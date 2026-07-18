from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from tensorflow import keras

from src.dataset import (
    DEFAULT_CLS_POSITIVE_COLUMNS,
    DEFAULT_FILTER_COLUMNS,
    DatasetConfig,
    download_and_build_dataset,
    filter_existing_files,
)
from src.dataset_provider import (
    apply_clahe_tf,
    as_tf_dataset,
    decode_image,
    hard_negatives_from_positives,
)

DEFAULT_ROI_NORM_COLUMNS: tuple[str, str, str, str] = (
    "pad_resized_xmin_norm",
    "pad_resized_ymin_norm",
    "pad_resized_xmax_norm",
    "pad_resized_ymax_norm",
)

# Misma ruta local que descarga ``DatasetConfig.splits_local`` desde GCS.
DATASET_SPLITS_PATH = DatasetConfig().splits_local


def resolve_dataset_splits_path(config) -> Path:
    """Ruta de splits: el dataset reducido usa un archivo aparte para no pisar el canonico."""
    path = DATASET_SPLITS_PATH
    if config.get("GENERAL", {}).get("REDUCED_DATASET"):
        return path.with_name(f"{path.stem}_reduced{path.suffix}")
    return path

__all__ = [
    "EpochTimer",
    "MemoryEpochLogger",
    "TrainingTimer",
    "sample_memory_usage",
    "run_training_stage",
    "apply_clahe_tf",
    "apply_probability_threshold",
    "build_positive_negative_dataset",
    "decode_image",
    "deduplicate_images",
    "dispose_model_builder",
    "logit_initial_bias",
    "predict_probs_and_labels",
    "prepare_dataset",
    "release_gpu_memory",
    "resample_train_for_patch",
    "resolve_dataset_splits_path",
    "resolve_steps_per_execution",
    "get_dataset_splits",
    "load_dataset_splits",
    "save_dataset_splits",
    "stratified_train_val_test_split",
    "threshold_best_f1",
    "threshold_recall_target",
    "threshold_youden_j",
    "undersample_negatives",
    "warmup",
]


def deduplicate_images(
    df: pd.DataFrame,
    *,
    patient_id_column: str = "patient_id",
    image_id_column: str = "image_id",
    path_column: str = "path",
    label_column: str = "cls",
    flag_columns: tuple[str, ...] | None = None,
    roi_norm_columns: tuple[str, str, str, str] = DEFAULT_ROI_NORM_COLUMNS,
    keep: Literal["first", "last"] = "first",
) -> pd.DataFrame:
    """One row per image: collapse CSV rows with multiple findings/boxes.

    Las columnas ROI (xmin/ymin/xmax/ymax normalizados) se agregan como union
    del bounding box (min en mins, max en maxs) para que avoid_roi evite todas
    las anotaciones de la imagen, no solo la primera fila.
    """
    if df.empty:
        return df.copy()

    key_cols = [patient_id_column, image_id_column]
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for deduplication: {missing}")

    out = df.copy()
    n_before = len(out)

    if flag_columns is None:
        flag_columns = tuple(
            c
            for c in (*DEFAULT_FILTER_COLUMNS, *DEFAULT_CLS_POSITIVE_COLUMNS)
            if c in out.columns
        )

    if label_column in out.columns:
        conflicts = (
            out.groupby(key_cols, dropna=False)[label_column].nunique().gt(1).sum()
        )
        if conflicts:
            raise ValueError(
                f"{conflicts} images have inconsistent {label_column} before deduplication."
            )

    agg: dict[str, str] = {
        patient_id_column: "first",
        image_id_column: "first",
    }
    if path_column in out.columns:
        agg[path_column] = "first"
    if label_column in out.columns:
        agg[label_column] = "max"
    for col in flag_columns:
        if col in out.columns:
            agg[col] = "max"

    xmin_c, ymin_c, xmax_c, ymax_c = roi_norm_columns
    if xmin_c in out.columns:
        agg[xmin_c] = "min"
    if ymin_c in out.columns:
        agg[ymin_c] = "min"
    if xmax_c in out.columns:
        agg[xmax_c] = "max"
    if ymax_c in out.columns:
        agg[ymax_c] = "max"

    for col in out.columns:
        if col not in agg:
            agg[col] = keep

    out = (
        out.groupby(key_cols, dropna=False, as_index=False)
        .agg(agg)
        .reset_index(drop=True)
    )

    n_removed = n_before - len(out)
    if n_removed:
        print(
            f"deduplicate_images: {n_before} -> {len(out)} rows "
            f"({n_removed} duplicates removed)"
        )

    return out


def prepare_dataset(config):
    """Descarga/arma el dataset y lo etiqueta en positivos/negativos.

    Une ``download_and_build_dataset`` (I/O + columna ``cls``) con
    ``build_positive_negative_dataset`` (split segun ``POSITIVE_MODE``) y devuelve
    el DataFrame binario listo para hacer los splits.
    """
    data = download_and_build_dataset(config)
    return build_positive_negative_dataset(config, data["ds"])


def build_positive_negative_dataset(config, ds):
    """Arma el dataset binario positivos/negativos segun ``POSITIVE_MODE``.

    - "mass": positivos = imagenes con ``Mass==1`` (tarea "deteccion de masas").
      Las imagenes con otro hallazgo pero sin masa se DESCARTAN (no son ni
      positivo ni negativo limpio).
    - "full": cualquier hallazgo cuenta como positivo (``cls==1``); no se descarta
      ninguna fila (positivos U negativos = dataset completo).

    Deduplica por imagen, fuerza ``cls`` a 1.0/0.0 y quita de los negativos las
    imagenes que ya aparecen como positivas (mismo patient_id/image_id).
    """
    positive_mode = config["GENERAL"]["POSITIVE_MODE"]
    if positive_mode == "mass":
        positive_mask = ds["Mass"] == 1
    elif positive_mode == "full":
        positive_mask = ds["cls"] == 1
    else:
        raise ValueError(f"POSITIVE_MODE desconocido: {positive_mode!r} (usar 'mass' o 'full')")

    ds_finding = deduplicate_images(ds[positive_mask].copy())
    ds_no_finding = deduplicate_images(ds[ds["cls"] == 0].copy())
    ds_finding["cls"] = 1.0
    ds_no_finding["cls"] = 0.0

    key = ["patient_id", "image_id"]
    ds_no_finding = (
        ds_no_finding.merge(ds_finding[key], on=key, how="left", indicator=True)
        .query("_merge == 'left_only'")
        .drop(columns="_merge")
    )
    ds = pd.concat([ds_finding, ds_no_finding], ignore_index=True)
    print(
        f"POSITIVE_MODE={positive_mode}: {len(ds_finding)} positivos / "
        f"{len(ds_no_finding)} negativos ({len(ds)} total)"
    )
    return ds


def stratified_train_val_test_split(
    config,
    ds,
    *,
    split_column="split",
    train_split_value="training",
    test_split_value="test",
    label_column="cls",
    group_column="patient_id",
):
    val_split = float(config["GENERAL"]["VALIDATION_SPLIT_RATIO"])
    if not (0.0 < val_split < 1.0):
        raise ValueError(
            f"VALIDATION_SPLIT_RATIO debe estar en (0, 1); recibido {val_split!r}"
        )
    seed = config["GENERAL"]["RANDOM_SEED"]
    tbl_training_full = ds[ds[split_column] == train_split_value].reset_index(drop=True)
    tbl_test = ds[ds[split_column] == test_split_value].reset_index(drop=True)

    n_splits = max(2, round(1 / val_split))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(
        sgkf.split(
            tbl_training_full,
            y=tbl_training_full[label_column],
            groups=tbl_training_full[group_column],
        )
    )

    tbl_train = tbl_training_full.iloc[train_idx].reset_index(drop=True)
    tbl_val = tbl_training_full.iloc[val_idx].reset_index(drop=True)
    return tbl_train, tbl_val, tbl_test


def _normalize_split_id_columns(
    df: pd.DataFrame,
    *,
    patient_id_column: str,
    image_id_column: str,
) -> pd.DataFrame:
    missing = [c for c in (patient_id_column, image_id_column) if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas de id para splits: {missing}")
    out = df[[patient_id_column, image_id_column]].copy()
    out[patient_id_column] = out[patient_id_column].astype(str)
    out[image_id_column] = out[image_id_column].astype(str)
    return out


def _ids_records(
    df: pd.DataFrame,
    *,
    patient_id_column: str,
    image_id_column: str,
) -> list[dict[str, str]]:
    ids = _normalize_split_id_columns(
        df, patient_id_column=patient_id_column, image_id_column=image_id_column
    )
    return ids.rename(
        columns={patient_id_column: "patient_id", image_id_column: "image_id"}
    ).to_dict(orient="records")


def _frame_from_split_ids(
    ds: pd.DataFrame,
    records: list[dict[str, Any]] | pd.DataFrame,
    *,
    split_name: str,
    patient_id_column: str,
    image_id_column: str,
) -> pd.DataFrame:
    """Reconstruye un split preservando el orden exacto del archivo de IDs."""
    if isinstance(records, pd.DataFrame):
        id_df = records[[patient_id_column, image_id_column]].copy()
    else:
        id_df = pd.DataFrame(records)
        if id_df.empty:
            id_df = pd.DataFrame(columns=["patient_id", "image_id"])
        id_df = id_df.rename(
            columns={"patient_id": patient_id_column, "image_id": image_id_column}
        )
        id_df = id_df[[patient_id_column, image_id_column]]

    id_df[patient_id_column] = id_df[patient_id_column].astype(str)
    id_df[image_id_column] = id_df[image_id_column].astype(str)
    id_df = id_df.reset_index(drop=True)
    id_df["_split_order"] = np.arange(len(id_df), dtype=np.int64)

    ds_keys = ds.copy()
    ds_keys[patient_id_column] = ds_keys[patient_id_column].astype(str)
    ds_keys[image_id_column] = ds_keys[image_id_column].astype(str)

    merged = id_df.merge(
        ds_keys,
        on=[patient_id_column, image_id_column],
        how="left",
        indicator=True,
    )
    missing = int((merged["_merge"] != "both").sum())
    if missing:
        raise ValueError(
            f"Split {split_name!r}: {missing}/{len(id_df)} ids no estan en el dataset actual. "
            "Regenera el archivo de splits o revisa POSITIVE_MODE / reduce."
        )
    return (
        merged.sort_values("_split_order", kind="mergesort")
        .drop(columns=["_merge", "_split_order"])
        .reset_index(drop=True)
    )


def save_dataset_splits(
    config,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    patient_id_column: str = "patient_id",
    image_id_column: str = "image_id",
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persiste los ids de train/val/test (orden incluido) en JSON o CSV.

    La ruta sale de ``resolve_dataset_splits_path`` (canonico o ``*_reduced``).
    La metadata (seed, positive_mode, ratios, ...) sale de ``config["GENERAL"]``.
    ``metadata`` extra se mergea encima. Guardar el train *despues* del
    undersample fija tambien el conjunto que ve cada epoca de entrenamiento.
    El formato se elige por extension: ``.json`` o ``.csv``.
    """
    general = config["GENERAL"]
    path = resolve_dataset_splits_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    meta = {
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "random_seed": general["RANDOM_SEED"],
        "positive_mode": general["POSITIVE_MODE"],
        "validation_split_ratio": general["VALIDATION_SPLIT_RATIO"],
        "no_finding_to_finding_ratio": general["NO_FINDING_TO_FINDING_RATIO"],
        "reduced_dataset": general["REDUCED_DATASET"],
        "includes_undersampled_train": True,
        **(metadata or {}),
    }

    if suffix == ".json":
        payload = {
            "meta": meta,
            "train": _ids_records(
                train, patient_id_column=patient_id_column, image_id_column=image_id_column
            ),
            "val": _ids_records(
                val, patient_id_column=patient_id_column, image_id_column=image_id_column
            ),
            "test": _ids_records(
                test, patient_id_column=patient_id_column, image_id_column=image_id_column
            ),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elif suffix == ".csv":
        frames = []
        for split_name, df in (("train", train), ("val", val), ("test", test)):
            ids = _normalize_split_id_columns(
                df, patient_id_column=patient_id_column, image_id_column=image_id_column
            )
            ids = ids.rename(
                columns={patient_id_column: "patient_id", image_id_column: "image_id"}
            )
            ids.insert(0, "split", split_name)
            ids.insert(1, "order", np.arange(len(ids), dtype=np.int64))
            frames.append(ids)
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        raise ValueError(f"Formato de splits no soportado: {suffix!r} (usa .json o .csv)")

    print(
        f"Splits guardados en {path} "
        f"(train={meta['n_train']}, val={meta['n_val']}, test={meta['n_test']})"
    )
    return path


def _assert_binary_split_ready(
    df: pd.DataFrame,
    *,
    split_name: str,
    label_column: str = "cls",
) -> None:
    """Falla temprano si un split esta vacio o no tiene ambas clases."""
    if len(df) == 0:
        raise ValueError(f"Split {split_name!r} quedo vacio")
    n_pos = int((df[label_column] >= 0.5).sum())
    n_neg = int(len(df) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Split {split_name!r} necesita ambas clases; "
            f"recibido pos={n_pos}, neg={n_neg}, total={len(df)}"
        )


def get_dataset_splits(config, ds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Devuelve train/val/test regenerando o cargando los ids fijos.

    Lo decide ``config["GENERAL"]["REGENERATE_SPLITS"]``:

    - ``True``: regenera el split estratificado, undersamplea los negativos del
      train y guarda los ids (train ya undersampleado) en ``DATASET_SPLITS_PATH``.
    - ``False``: carga los ids fijos desde ``DATASET_SPLITS_PATH`` (no regenera ni
      undersamplea).
    """
    if config["GENERAL"]["REGENERATE_SPLITS"]:
        train, val, test = stratified_train_val_test_split(config, ds)
        train = undersample_negatives(config, train)
        for name, frame in (("train", train), ("val", val), ("test", test)):
            _assert_binary_split_ready(frame, split_name=name)
        save_dataset_splits(config, train, val, test)
        return train, val, test
    train, val, test = load_dataset_splits(config, ds)
    for name, frame in (("train", train), ("val", val), ("test", test)):
        _assert_binary_split_ready(frame, split_name=name)
    return train, val, test


def load_dataset_splits(
    config,
    ds: pd.DataFrame,
    *,
    patient_id_column: str = "patient_id",
    image_id_column: str = "image_id",
    require_existing_files: bool = True,
    path_column: str = "path",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga train/val/test desde un JSON/CSV de ids, en el mismo orden guardado.

    La ruta sale de ``resolve_dataset_splits_path``. La validacion de IDs corre
    contra ``ds`` completo; luego, si ``require_existing_files`` es True, se
    descartan las filas cuya imagen no exista en disco (evita que tf.data falle
    con NotFoundError al leerlas).
    """
    path = resolve_dataset_splits_path(config)
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo de splits: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        train = _frame_from_split_ids(
            ds,
            payload["train"],
            split_name="train",
            patient_id_column=patient_id_column,
            image_id_column=image_id_column,
        )
        val = _frame_from_split_ids(
            ds,
            payload["val"],
            split_name="val",
            patient_id_column=patient_id_column,
            image_id_column=image_id_column,
        )
        test = _frame_from_split_ids(
            ds,
            payload["test"],
            split_name="test",
            patient_id_column=patient_id_column,
            image_id_column=image_id_column,
        )
        meta = payload.get("meta") or {}
    elif suffix == ".csv":
        raw = pd.read_csv(path, dtype={"patient_id": str, "image_id": str, "split": str})
        if "order" in raw.columns:
            raw = raw.sort_values(["split", "order"], kind="mergesort")
        train = _frame_from_split_ids(
            ds,
            raw.loc[raw["split"] == "train", ["patient_id", "image_id"]],
            split_name="train",
            patient_id_column=patient_id_column,
            image_id_column=image_id_column,
        )
        val = _frame_from_split_ids(
            ds,
            raw.loc[raw["split"] == "val", ["patient_id", "image_id"]],
            split_name="val",
            patient_id_column=patient_id_column,
            image_id_column=image_id_column,
        )
        test = _frame_from_split_ids(
            ds,
            raw.loc[raw["split"] == "test", ["patient_id", "image_id"]],
            split_name="test",
            patient_id_column=patient_id_column,
            image_id_column=image_id_column,
        )
        meta = {"n_train": len(train), "n_val": len(val), "n_test": len(test)}
    else:
        raise ValueError(f"Formato de splits no soportado: {suffix!r} (usa .json o .csv)")

    if require_existing_files:
        train = filter_existing_files(train, path_column)
        val = filter_existing_files(val, path_column)
        test = filter_existing_files(test, path_column)

    print(
        f"Splits cargados desde {path} "
        f"(train={len(train)}, val={len(val)}, test={len(test)}"
        + (f", meta_keys={sorted(meta.keys())}" if meta else "")
        + ")"
    )
    return train, val, test


def logit_initial_bias(n_positive: int, n_negative: int, *, eps: float = 1e-6) -> float:
    """Log-odds inicial; evita division por cero / ±inf con conteos nulos."""
    n_pos = max(float(n_positive), eps)
    n_neg = max(float(n_negative), eps)
    return float(np.log(n_pos / n_neg))


def resolve_steps_per_execution(
    n_rows: int,
    batch_size: int,
    *,
    max_steps: int = 32,
) -> int:
    """Acota steps_per_execution al numero real de batches por epoca."""
    steps_per_epoch = max(1, (int(n_rows) + int(batch_size) - 1) // int(batch_size))
    return max(1, min(int(max_steps), steps_per_epoch))


def _sigmoid(z) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def _eval_tf_dataset(dataset):
    """Vista ordenada y estable para evaluacion (InspectDataset o tf.data)."""
    if hasattr(dataset, "ordered"):
        return as_tf_dataset(dataset.ordered())
    return as_tf_dataset(dataset)


def predict_probs_and_labels(model, dataset) -> tuple[np.ndarray, np.ndarray]:
    """Una sola pasada sobre el dataset: devuelve (y_true, y_prob) alineados.

    Itera el pipeline ``tf.data`` una sola vez, tomando labels y logits del mismo
    batch (en vez de recorrerlo dos veces decodificando/croppeando las imagenes en
    cada pasada).
    """
    eval_ds = _eval_tf_dataset(dataset)
    keras_model = model.model if hasattr(model, "model") else model

    labels_batches: list[np.ndarray] = []
    logits_batches: list[np.ndarray] = []
    for xb, yb in eval_ds:
        labels_batches.append(np.asarray(yb, dtype=np.int64).reshape(-1))
        logits_batches.append(
            np.asarray(keras_model.predict_on_batch(xb), dtype=np.float64).reshape(-1)
        )

    if not labels_batches:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)

    y_true = np.concatenate(labels_batches)
    y_prob = _sigmoid(np.concatenate(logits_batches))
    return y_true, y_prob


def apply_probability_threshold(y_prob, threshold: float) -> np.ndarray:
    return (y_prob >= threshold).astype(np.int64)


def undersample_negatives(
    config,
    df: pd.DataFrame,
    *,
    label_column: str = "cls",
) -> pd.DataFrame:
    """Submuestrea negativos a `ratio`:1 respecto de los positivos. Solo para train.

    ``ratio`` y ``seed`` salen de ``config["GENERAL"]``.
    """
    ratio = float(config["GENERAL"]["NO_FINDING_TO_FINDING_RATIO"])
    seed = config["GENERAL"]["RANDOM_SEED"]
    pos = df[df[label_column] >= 0.5]
    neg = df[df[label_column] < 0.5]
    if len(pos) == 0:
        raise ValueError("undersample_negatives: no hay positivos en el DataFrame")
    n_neg = int(min(len(neg), int(round(len(pos) * ratio))))
    if n_neg <= 0:
        raise ValueError(
            f"undersample_negatives: n_neg={n_neg} invalido "
            f"(pos={len(pos)}, neg={len(neg)}, ratio={ratio})"
        )
    neg = neg.sample(n=n_neg, random_state=seed)
    out = pd.concat([pos, neg], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def resample_train_for_patch(
    train_df: pd.DataFrame,
    patch_ratio: dict[str, int | float],
    seed: int,
    *,
    label_column: str = "cls",
) -> tuple[pd.DataFrame, float, float]:
    """Remuestrea train en positivos + hard negatives + random negatives segun patch_ratio.

    ``patch_ratio`` usa claves POSITIVE, HARD_NEGATIVE y RANDOM_NEGATIVE (multiplicadores
    respecto del conteo de positivos). Devuelve (df, focal_alpha, initial_bias).
    """
    positive_train = train_df[train_df[label_column] == 1].copy()
    negative_pool = train_df[train_df[label_column] == 0].copy()
    n_positive = len(positive_train)

    def _resample(df: pd.DataFrame, n: int) -> pd.DataFrame:
        n = int(round(n))
        if n <= 0 or len(df) == 0:
            return df.iloc[0:0]
        if n <= len(df):
            return df.sample(n=n, random_state=seed)
        reps = pd.concat([df] * (n // len(df) + 1), ignore_index=True)
        return reps.sample(n=n, random_state=seed)

    positive_final = _resample(positive_train, n_positive * patch_ratio["POSITIVE"])
    hard_negative_final = _resample(hard_negatives_from_positives(positive_train), n_positive * patch_ratio["HARD_NEGATIVE"])
    random_negative_final = _resample(negative_pool, n_positive * patch_ratio["RANDOM_NEGATIVE"])
    train_patch = pd.concat([positive_final, hard_negative_final, random_negative_final], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    frac_neg = (train_patch[label_column] == 0).sum() / len(train_patch)
    bias = logit_initial_bias(int((train_patch[label_column] == 1).sum()), int((train_patch[label_column] == 0).sum()))
    print("resample_train_for_patch:", {"POSITIVE": len(positive_final), "HARD_NEGATIVE": len(hard_negative_final), "RANDOM_NEGATIVE": len(random_negative_final), "TOTAL": len(train_patch), "FRAC_NEG": round(float(frac_neg), 3)})
    return train_patch, float(frac_neg), bias


def _threshold_inputs_ok(y_true, y_prob) -> bool:
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    if y_true.size == 0 or y_prob.size == 0:
        return False
    classes = np.unique(y_true.astype(int))
    return classes.size >= 2


def threshold_youden_j(y_true, y_prob, *, default: float = 0.5) -> float:
    """Umbral que maximiza el indice J de Youden (sensibilidad + especificidad - 1)."""
    if not _threshold_inputs_ok(y_true, y_prob):
        print("Advertencia: threshold_youden_j sin ambas clases; se usa default", default)
        return float(default)
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    best = int(np.argmax(tpr - fpr))
    value = float(thr[best])
    if not np.isfinite(value):
        return float(default)
    return value


def threshold_best_f1(y_true, y_prob, *, default: float = 0.5) -> float:
    """Umbral que maximiza F1 sobre la clase positiva."""
    if not _threshold_inputs_ok(y_true, y_prob):
        print("Advertencia: threshold_best_f1 sin ambas clases; se usa default", default)
        return float(default)
    precision, recall, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    if f1.size == 0:
        return float(default)
    best = int(np.argmax(f1))
    value = float(thr[best])
    if not np.isfinite(value):
        return float(default)
    return value


def threshold_recall_target(y_true, y_prob, target_recall: float = 0.90, *, default: float = 0.5) -> float:
    """Umbral mas alto (mayor precision) que aun garantiza recall >= target_recall."""
    if not _threshold_inputs_ok(y_true, y_prob):
        print(
            "Advertencia: threshold_recall_target sin ambas clases; se usa default",
            default,
        )
        return float(default)
    precision, recall, thr = precision_recall_curve(y_true, y_prob)
    recall_t, precision_t = recall[:-1], precision[:-1]
    ok = np.flatnonzero(recall_t >= target_recall)
    if ok.size == 0:
        return threshold_best_f1(y_true, y_prob, default=default)
    best = int(ok[np.argmax(precision_t[ok])])
    value = float(thr[best])
    if not np.isfinite(value):
        return float(default)
    return value


def warmup(model, train_ds) -> float:
    """Un paso forward-only para compilar el grafo antes de model.fit() (sin actualizar pesos)."""
    t0 = time.perf_counter()
    try:
        xb, _ = next(iter(as_tf_dataset(train_ds)))
    except StopIteration as exc:
        raise ValueError("warmup: el dataset de train no produjo batches") from exc
    model.model.predict_on_batch(xb)
    elapsed = time.perf_counter() - t0
    print(f"  Warm-up completado en {elapsed:.1f}s")
    return elapsed


def release_gpu_memory(*, clear_keras_session: bool = True) -> None:
    """Libera VRAM/RAM entre experimentos.

    Usar ``clear_keras_session=False`` solo si un builder activo se reutiliza
    inmediatamente después (p. ej. patch_hardneg exitoso → abmil_patch_hardneg).
    """
    if clear_keras_session:
        tf.keras.backend.clear_session()
    gc.collect()


def dispose_model_builder(builder_obj) -> None:
    """Rompe referencias internas de un builder para que gc libere tras clear_session."""
    if builder_obj is None:
        return
    builder_obj.model = None
    builder_obj.backbone = None
    builder_obj.pretrained_builder = None


class TrainingTimer:
    """Wall-clock compartido entre las fases de entrenamiento multi-etapa."""

    def __init__(self) -> None:
        self._training_start: float | None = None
        self._stage_start: float | None = None
        self.current_stage: int | None = None
        self.stage_summaries: dict[int, dict[str, float]] = {}

    def start_training(self) -> None:
        self._training_start = time.perf_counter()

    def start_stage(self, stage: int) -> None:
        self.current_stage = stage
        self._stage_start = time.perf_counter()

    def elapsed_since_training_start(self) -> float:
        if self._training_start is None:
            return 0.0
        return time.perf_counter() - self._training_start

    def elapsed_since_stage_start(self) -> float:
        if self._stage_start is None:
            return 0.0
        return time.perf_counter() - self._stage_start

    def record_stage_summary(
        self,
        stage: int,
        *,
        setup_seconds: float,
        warmup_seconds: float,
        fit_seconds: float,
        checkpoint_seconds: float,
    ) -> dict[str, float]:
        summary = {
            "setup_seconds": setup_seconds,
            "warmup_seconds": warmup_seconds,
            "fit_seconds": fit_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "stage_wall_seconds": self.elapsed_since_stage_start(),
        }
        self.stage_summaries[stage] = summary
        return summary


def sample_memory_usage() -> dict[str, float]:
    """RAM del proceso y VRAM de GPU:0 en GB. Omite metricas no disponibles."""
    metrics: dict[str, float] = {}

    try:
        import psutil

        metrics["ram_rss_gb"] = psutil.Process().memory_info().rss / (1024**3)
    except Exception:
        pass

    try:
        if tf.config.list_physical_devices("GPU"):
            info = tf.config.experimental.get_memory_info("GPU:0")
            metrics["vram_current_gb"] = info["current"] / (1024**3)
            metrics["vram_peak_gb"] = info["peak"] / (1024**3)
    except Exception:
        pass

    return metrics


class MemoryEpochLogger(keras.callbacks.Callback):
    """Anade uso de RAM/VRAM a logs al final de cada epoca (Comet los recoge via CometEpochLogger)."""

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        logs.update(sample_memory_usage())


class EpochTimer(keras.callbacks.Callback):
    def __init__(self, training_timer: TrainingTimer | None = None):
        super().__init__()
        self.training_timer = training_timer

    def on_train_begin(self, logs=None):
        self.epoch_times = []
        self.fit_elapsed_times = []
        self._fit_start = time.perf_counter()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        epoch_wall = time.perf_counter() - self.epoch_start_time
        fit_elapsed = time.perf_counter() - self._fit_start
        self.epoch_times.append(epoch_wall)
        self.fit_elapsed_times.append(fit_elapsed)
        if logs is None:
            return

        logs["epoch_wall_seconds"] = epoch_wall
        logs["epoch_time_seconds"] = epoch_wall
        logs["fit_elapsed_seconds"] = fit_elapsed
        logs["total_elapsed_seconds"] = fit_elapsed
        if self.training_timer is not None:
            logs["global_elapsed_seconds"] = self.training_timer.elapsed_since_training_start()
            logs["stage_elapsed_seconds"] = self.training_timer.elapsed_since_stage_start()


_STAGE_EPOCH_KEYS = {
    1: "EPOCHS_FROZEN_BACKBONE",
    2: "EPOCHS_PARTIAL_BACKBONE",
    3: "EPOCHS_FULL_FINETUNE",
}


def run_training_stage(
    config,
    model,
    train_ds,
    val_ds,
    *,
    stage: int,
    training_timer: TrainingTimer,
    epoch_offset: int = 0,
    experiment=None,
    extra_callbacks=None,
    setup_fn=None,
):
    """Ejecuta una etapa de congelamiento con metricas de tiempo reales (wall-clock).

    Las epocas de cada etapa (1=frozen, 2=partial, 3=full) salen de
    ``config["TRAINING"]``; el resto (setup de descongelamiento) sigue llegando por
    ``setup_fn``.
    """
    from src.comet_logging import CometEpochLogger, log_stage_timing

    if stage not in _STAGE_EPOCH_KEYS:
        raise ValueError(f"stage invalido: {stage!r} (esperado 1, 2 o 3)")
    epochs = config["TRAINING"][_STAGE_EPOCH_KEYS[stage]]

    # El setup de freeze/unfreeze corre aunque se omita el fit, para no saltar fases.
    setup_seconds = 0.0
    if setup_fn is not None:
        t0 = time.perf_counter()
        setup_fn()
        setup_seconds = time.perf_counter() - t0

    if epochs <= 0:
        empty_history = keras.callbacks.History()
        empty_history.history = {}
        print(f"  Etapa {stage}: omitida (epochs=0)")
        return empty_history, None, None

    training_timer.start_stage(stage)

    warmup_seconds = warmup(model, train_ds)

    fit_t0 = time.perf_counter()
    callbacks = list(extra_callbacks or [])
    if experiment is not None:
        callbacks.insert(
            0,
            CometEpochLogger(experiment, epoch_offset=epoch_offset, stage=stage),
        )
    history = model.fit(
        train_ds,
        val_ds,
        epochs=epochs,
        callbacks=callbacks,
        training_timer=training_timer,
        stage=stage,
    )
    fit_seconds = time.perf_counter() - fit_t0

    ckpt_t0 = time.perf_counter()
    best_epoch = model.load_best_checkpoint()
    checkpoint_seconds = time.perf_counter() - ckpt_t0

    summary = training_timer.record_stage_summary(
        stage,
        setup_seconds=setup_seconds,
        warmup_seconds=warmup_seconds,
        fit_seconds=fit_seconds,
        checkpoint_seconds=checkpoint_seconds,
    )
    epochs_completed = len(history.history.get("loss", []))
    global_epoch = epoch_offset + epochs_completed
    if experiment is not None:
        log_stage_timing(experiment, stage, summary, step=global_epoch)

    print(
        f"  Etapa {stage}: {summary['stage_wall_seconds']:.1f}s total "
        f"(setup={setup_seconds:.1f}s, warmup={warmup_seconds:.1f}s, "
        f"fit={fit_seconds:.1f}s, checkpoint={checkpoint_seconds:.1f}s)"
    )
    return history, best_epoch, summary
