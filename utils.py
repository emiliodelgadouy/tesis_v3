from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from tensorflow import keras

from src.dataset import DEFAULT_CLS_POSITIVE_COLUMNS, DEFAULT_FILTER_COLUMNS
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

__all__ = [
    "EpochTimer",
    "MemoryEpochLogger",
    "TrainingTimer",
    "sample_memory_usage",
    "run_training_stage",
    "apply_clahe_tf",
    "apply_probability_threshold",
    "binary_report",
    "decode_image",
    "deduplicate_images",
    "dispose_model_builder",
    "labels_from_tf_dataset",
    "logit_initial_bias",
    "plot_binary_confusion_matrix",
    "predict_probabilities",
    "predict_probabilities_tta",
    "predict_probs_and_labels",
    "release_gpu_memory",
    "resample_train_for_patch",
    "resolve_steps_per_execution",
    "load_dataset_splits",
    "save_dataset_splits",
    "stratified_split",
    "threshold_best_f1",
    "threshold_max_recall",
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


def stratified_split(
    ds,
    val_split=0.20,
    seed=42,
    split_column="split",
    train_split_value="training",
    test_split_value="test",
    label_column="cls",
    group_column="patient_id",
):
    tbl_training_full = ds[ds[split_column] == train_split_value].reset_index(drop=True)
    tbl_test = ds[ds[split_column] == test_split_value].reset_index(drop=True)

    n_splits = round(1 / val_split)
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
    path: str | Path,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    patient_id_column: str = "patient_id",
    image_id_column: str = "image_id",
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persiste los ids de train/val/test (orden incluido) en JSON o CSV.

    Guardar el train *despues* del undersample fija tambien el conjunto que ve
    cada epoca de entrenamiento. El formato se elige por extension:
    ``.json`` (default recomendado) o ``.csv``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    meta = {
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
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


def load_dataset_splits(
    path: str | Path,
    ds: pd.DataFrame,
    *,
    patient_id_column: str = "patient_id",
    image_id_column: str = "image_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga train/val/test desde un JSON/CSV de ids, en el mismo orden guardado."""
    path = Path(path)
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

    print(
        f"Splits cargados desde {path} "
        f"(train={len(train)}, val={len(val)}, test={len(test)}"
        + (f", meta_keys={sorted(meta.keys())}" if meta else "")
        + ")"
    )
    return train, val, test


def logit_initial_bias(n_positive: int, n_negative: int) -> float:
    return float(np.log(n_positive / n_negative))


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


def labels_from_tf_dataset(dataset) -> np.ndarray:
    return np.concatenate(
        [np.asarray(yb, dtype=np.int64).reshape(-1) for _, yb in _eval_tf_dataset(dataset)]
    )


def predict_probabilities(model, dataset, *, verbose: int = 1) -> np.ndarray:
    logits = model.predict(dataset, verbose=verbose).astype(np.float64).reshape(-1)
    return _sigmoid(logits)


def predict_probs_and_labels(model, dataset, *, use_tta: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Una sola pasada sobre el dataset: devuelve (y_true, y_prob) alineados.

    Reemplaza el par ``labels_from_tf_dataset`` + ``predict_probabilities``, que
    recorre el pipeline ``tf.data`` dos veces (decodificando/croppeando las
    imagenes en cada pasada). Aca se itera una sola vez, tomando labels y logits
    del mismo batch. Con ``use_tta=True`` promedia la prediccion con su espejo
    horizontal (segunda pasada solo si TTA esta activo).
    """
    eval_ds = _eval_tf_dataset(dataset)
    keras_model = model.model if hasattr(model, "model") else model

    labels_batches: list[np.ndarray] = []
    logits_batches: list[np.ndarray] = []
    flip_logits_batches: list[np.ndarray] = []
    for xb, yb in eval_ds:
        labels_batches.append(np.asarray(yb, dtype=np.int64).reshape(-1))
        logits_batches.append(
            np.asarray(keras_model.predict_on_batch(xb), dtype=np.float64).reshape(-1)
        )
        if use_tta:
            flip_logits_batches.append(
                np.asarray(
                    keras_model.predict_on_batch(_flip_inputs_tta(xb)), dtype=np.float64
                ).reshape(-1)
            )

    y_true = np.concatenate(labels_batches)
    y_prob = _sigmoid(np.concatenate(logits_batches))
    if use_tta:
        y_prob = (y_prob + _sigmoid(np.concatenate(flip_logits_batches))) / 2.0
    return y_true, y_prob


def apply_probability_threshold(y_prob, threshold: float) -> np.ndarray:
    return (y_prob >= threshold).astype(np.int64)


def plot_binary_confusion_matrix(
    y_true,
    y_pred,
    *,
    title: str,
    display_labels: tuple[str, str] = ("Neg", "Pos"),
    figsize: tuple[float, float] = (5, 4),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred, labels=[0, 1]),
        display_labels=list(display_labels),
    ).plot(ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def threshold_max_recall(y_true, y_prob) -> tuple[float, float]:
    """Umbral que maximiza recall en validación (desempate por precisión)."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    recalls_t, precisions_t = recalls[:-1], precisions[:-1]
    max_recall = float(recalls_t.max())
    candidate_idx = np.flatnonzero(recalls_t == max_recall)
    best_i = int(candidate_idx[np.argmax(precisions_t[candidate_idx])])
    return float(thresholds[best_i]), max_recall


def undersample_negatives(
    df: pd.DataFrame,
    *,
    ratio: int = 3,
    seed: int = 42,
    label_column: str = "cls",
) -> pd.DataFrame:
    """Submuestrea negativos a `ratio`:1 respecto de los positivos. Solo para train."""
    pos = df[df[label_column] >= 0.5]
    neg = df[df[label_column] < 0.5]
    n_neg = min(len(neg), len(pos) * ratio)
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


def _flip_inputs_tta(x: tf.Tensor) -> tf.Tensor:
    """Flip horizontal para imagen (B,H,W,C) o bag MIL pre-tiled (B,K,H,W,C)."""
    if x.shape.rank == 5:
        return tf.reverse(x, axis=[3])
    return tf.image.flip_left_right(x)


def predict_probabilities_tta(model, dataset) -> np.ndarray:
    """TTA por flip horizontal: promedia probabilidades de imagen normal + espejada."""
    base = _eval_tf_dataset(dataset)
    flipped = base.map(
        lambda x, y: (_flip_inputs_tta(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    p_base = _sigmoid(model.predict(base, verbose=0).reshape(-1))
    p_flip = _sigmoid(model.predict(flipped, verbose=0).reshape(-1))
    return (p_base + p_flip) / 2.0


def threshold_youden_j(y_true, y_prob) -> float:
    """Umbral que maximiza el indice J de Youden (sensibilidad + especificidad - 1)."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    best = int(np.argmax(tpr - fpr))
    return float(thr[best])


def threshold_best_f1(y_true, y_prob) -> float:
    """Umbral que maximiza F1 sobre la clase positiva."""
    precision, recall, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best = int(np.argmax(f1))
    return float(thr[best])


def threshold_recall_target(y_true, y_prob, target_recall: float = 0.90) -> float:
    """Umbral mas alto (mayor precision) que aun garantiza recall >= target_recall."""
    precision, recall, thr = precision_recall_curve(y_true, y_prob)
    recall_t, precision_t = recall[:-1], precision[:-1]
    ok = np.flatnonzero(recall_t >= target_recall)
    if ok.size == 0:
        return threshold_best_f1(y_true, y_prob)
    best = int(ok[np.argmax(precision_t[ok])])
    return float(thr[best])


def binary_report(y_true, y_prob, threshold: float, *, title: str = "") -> dict:
    """Imprime metricas clinicas (sensibilidad, especificidad, PPV, NPV, F1) a un umbral."""
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) else 0.0
    balanced_acc = (sensitivity + specificity) / 2.0
    print(f"== {title} (umbral={threshold:.4f}) ==")
    print(f"  ROC-AUC ........ {roc_auc_score(y_true, y_prob):.4f}")
    print(f"  PR-AUC ......... {average_precision_score(y_true, y_prob):.4f}")
    print(f"  Sensibilidad ... {sensitivity:.4f}  (recall+)")
    print(f"  Especificidad .. {specificity:.4f}")
    print(f"  PPV (prec+) .... {ppv:.4f}")
    print(f"  NPV ............ {npv:.4f}")
    print(f"  F1 ............. {f1:.4f}")
    print(f"  Balanced acc ... {balanced_acc:.4f}")
    print(f"  Matriz [tn fp / fn tp]: [{tn} {fp} / {fn} {tp}]\n")
    return {
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "balanced_accuracy": balanced_acc,
    }


def warmup(model, train_ds) -> float:
    """Un paso forward-only para compilar el grafo antes de model.fit() (sin actualizar pesos)."""
    t0 = time.perf_counter()
    xb, _ = next(iter(as_tf_dataset(train_ds)))
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


def run_training_stage(
    model,
    train_ds,
    val_ds,
    *,
    stage: int,
    epochs: int,
    training_timer: TrainingTimer,
    epoch_offset: int = 0,
    experiment=None,
    extra_callbacks=None,
    setup_fn=None,
):
    """Ejecuta una etapa de congelamiento con metricas de tiempo reales (wall-clock)."""
    from src.comet_logging import CometEpochLogger, log_stage_timing

    if epochs <= 0:
        empty_history = keras.callbacks.History()
        empty_history.history = {}
        print(f"  Etapa {stage}: omitida (epochs=0)")
        return empty_history, None, None

    training_timer.start_stage(stage)

    setup_seconds = 0.0
    if setup_fn is not None:
        t0 = time.perf_counter()
        setup_fn()
        setup_seconds = time.perf_counter() - t0

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
