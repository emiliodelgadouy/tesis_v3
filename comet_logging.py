"""Configuracion y helpers de Comet ML para el notebook de tesis."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import comet_ml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tensorflow import keras

from src.dataset_provider import as_tf_dataset

# Defaults; el notebook puede sobreescribir COMET_* y pasarlos a login()/start_training_experiment().
COMET_API_KEY = os.environ.get("COMET_API_KEY", "")
COMET_PROJECT = "tesis-reni"
COMET_N_SAMPLE_IMAGES = 8
# Comet rechaza nombres de imagen/figura con mas de 100 caracteres.
COMET_MAX_IMAGE_NAME_LEN = 100


def login(*, api_key: str | None = None) -> None:
    """Autentica en Comet. Si api_key es vacío, usa env o ~/.comet.config."""
    key = COMET_API_KEY if api_key is None else api_key
    if key:
        comet_ml.login(api_key=key)
    else:
        comet_ml.login()


class CometEpochLogger(keras.callbacks.Callback):
    """Loguea metricas de Keras en Comet indexadas por epoch global (no por batch step)."""

    def __init__(self, experiment, *, epoch_offset: int = 0, stage: int | None = None):
        super().__init__()
        self.experiment = experiment
        self.epoch_offset = epoch_offset
        self.stage = stage

    def on_train_begin(self, logs=None):
        # Alinear step/epoch internos de Comet con el contador global de epocas.
        self.experiment.set_step(self.epoch_offset)
        self.experiment.set_epoch(self.epoch_offset)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metrics = {k: float(v) for k, v in logs.items() if v is not None}
        global_epoch = self.epoch_offset + epoch + 1
        if self.stage is not None:
            self.experiment.log_metric(
                "training_stage",
                self.stage,
                step=global_epoch,
                epoch=global_epoch,
            )
        self.experiment.log_metrics(metrics, step=global_epoch, epoch=global_epoch)


def log_stage_timing(experiment, stage: int, summary: dict[str, float], *, step: int) -> None:
    """Loguea el desglose wall-clock de una etapa de entrenamiento."""
    experiment.log_metrics(
        {f"stage_{stage}_{key}": float(value) for key, value in summary.items()},
        step=step,
        epoch=step,
    )


def log_training_timing_summary(experiment, training_timer) -> None:
    """Loguea tiempos totales y por etapa al finalizar el entrenamiento."""
    total_seconds = training_timer.elapsed_since_training_start()
    metrics = {"training_wall_seconds": total_seconds}
    for stage, summary in training_timer.stage_summaries.items():
        metrics[f"stage_{stage}_wall_seconds"] = summary["stage_wall_seconds"]
    experiment.log_metrics(metrics)


def start_training_experiment(
    *,
    experiment_name: str,
    run_config: dict[str, Any],
    backbone_description: str,
    architecture_catalog: str,
    model=None,
    project_name: str = COMET_PROJECT,
):
    experiment = comet_ml.start(
        project_name=project_name,
        experiment_config=comet_ml.ExperimentConfig(
            name=experiment_name,
            auto_metric_logging=False,
        ),
    )
    # Evita el callback auto-inyectado de Keras, que loguea por batch step.
    experiment.disable_mp()
    experiment.set_step(0)
    experiment.set_epoch(0)
    experiment.log_parameters(run_config)
    experiment.log_other("backbone_description", backbone_description)
    experiment.log_asset_data(architecture_catalog, "catalogo_arquitecturas.txt")
    if model is not None:
        experiment.set_model_graph(model)
        _log_model_param_counts(experiment, model)
    return experiment


def _log_model_param_counts(experiment, model) -> None:
    """Loguea la cantidad de parametros del modelo (totales, entrenables y no entrenables)."""
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    non_trainable = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
    experiment.log_metric("model_total_params", trainable + non_trainable)
    experiment.log_metric("model_trainable_params", trainable)
    experiment.log_metric("model_non_trainable_params", non_trainable)


def _plot_confusion_matrix(y_true, y_pred, *, title: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred, labels=[0, 1]),
        display_labels=["Neg", "Pos"],
    ).plot(ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    return fig


def _log_split_confusion_matrices(
    experiment,
    *,
    backbone_name: str,
    split_name: str,
    y_true,
    y_pred_default,
    y_pred_youden,
    prob_threshold: float,
    thr_youden: float,
    show_plots: bool,
) -> None:
    split_key = split_name.lower()
    experiment.log_confusion_matrix(
        y_true=y_true,
        y_predicted=y_pred_default,
        labels=["Neg", "Pos"],
        title=f"{split_name} — umbral {prob_threshold}",
        file_name=f"confusion_matrix_{split_key}_default.json",
    )
    experiment.log_confusion_matrix(
        y_true=y_true,
        y_predicted=y_pred_youden,
        labels=["Neg", "Pos"],
        title=f"{split_name} — umbral Youden J = {thr_youden:.4f}",
        file_name=f"confusion_matrix_{split_key}_youden.json",
    )

    fig_cm_default = _plot_confusion_matrix(
        y_true,
        y_pred_default,
        title=f"{backbone_name} — {split_name} — umbral = {prob_threshold}",
    )
    experiment.log_figure(figure_name=f"confusion_matrix_{split_key}_default", figure=fig_cm_default)
    if show_plots:
        plt.show()
    plt.close(fig_cm_default)

    fig_cm_youden = _plot_confusion_matrix(
        y_true,
        y_pred_youden,
        title=f"{backbone_name} — {split_name} — umbral Youden J = {thr_youden:.4f}",
    )
    experiment.log_figure(figure_name=f"confusion_matrix_{split_key}_youden", figure=fig_cm_youden)
    if show_plots:
        plt.show()
    plt.close(fig_cm_youden)


def _bag_to_montage(bag: np.ndarray) -> np.ndarray:
    """Une las instancias (K, H, W, C) de un bag MIL en una imagen-grilla."""
    bag = np.asarray(bag)
    k = bag.shape[0]
    cols = int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))
    ph, pw = bag.shape[1], bag.shape[2]
    channels = bag.shape[3] if bag.ndim == 4 else 1
    canvas = np.zeros((rows * ph, cols * pw, channels), dtype=bag.dtype)
    for idx in range(k):
        r, c = divmod(idx, cols)
        canvas[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw] = bag[idx]
    return canvas


def _prepare_image_for_logging(img: np.ndarray) -> np.ndarray:
    """Devuelve float32 en [0, 1] listo para imshow."""
    img = np.asarray(img)
    if img.ndim == 4:
        img = _bag_to_montage(img)
    arr = img.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _fixed_sample_indices(y_true: np.ndarray, *, n_samples: int, random_seed: int) -> list[int]:
    """Muestras estables entre experimentos, balanceadas por etiqueta cuando se puede."""
    if n_samples <= 0:
        return []

    groups = {
        "neg": np.flatnonzero(y_true == 0),
        "pos": np.flatnonzero(y_true == 1),
    }
    per_group = max(1, n_samples // len(groups))
    picked: list[int] = []
    for idxs in groups.values():
        if idxs.size:
            picked.extend(idxs[:per_group].tolist())

    if len(picked) < n_samples:
        remaining = [i for i in range(len(y_true)) if i not in picked]
        extra = min(n_samples - len(picked), len(remaining))
        if extra:
            picked.extend(
                np.random.default_rng(random_seed).choice(remaining, size=extra, replace=False).tolist()
            )
    return picked[:n_samples]


def _sens_spec(y_true, y_prob, thr: float) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = (np.asarray(y_prob) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return sens, spec


def _log_final_weights(experiment, *, backbone_name: str, final_weights_path) -> None:
    weights_path = Path(final_weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"No existe el archivo de pesos final: {weights_path}")

    model_name = f"{backbone_name}_final_weights"
    try:
        experiment.log_model(name=model_name, file_or_folder=str(weights_path))
    except TypeError:
        experiment.log_model(model_name, str(weights_path))
    experiment.log_other("final_weights_file", weights_path.name)


def _comet_slug(value: str) -> str:
    slug = re.sub(r"[^\w.\-]+", "_", str(value).strip())
    return slug.strip("_") or "unknown"


def _resolve_experiment_name(experiment, experiment_name: str | None) -> str:
    if experiment_name:
        return _comet_slug(experiment_name)
    for attr in ("get_name", "name"):
        if hasattr(experiment, attr):
            val = getattr(experiment, attr)
            if callable(val):
                val = val()
            if val:
                return _comet_slug(val)
    return "experiment"


def _input_sample_name(
    *,
    experiment_name: str,
    split_name: str,
    dataset_idx: int,
    label: int,
    row: Any | None,
    max_len: int = COMET_MAX_IMAGE_NAME_LEN,
) -> str:
    prefix_parts = [experiment_name, "pre_augment", _comet_slug(split_name)]
    id_segment: str | None = None
    if row is not None:
        id_parts: list[str] = []
        for col in ("patient_id", "image_id"):
            if col in row.index and pd.notna(row[col]):
                id_parts.append(_comet_slug(row[col]))
        if id_parts:
            id_segment = "__".join(id_parts)
        elif "path" in row.index and pd.notna(row["path"]):
            id_segment = _comet_slug(Path(str(row["path"])).stem)

    # idx siempre presente -> garantiza unicidad aunque haya que recortar id_segment.
    suffix_parts = [f"idx{dataset_idx:04d}", f"label{label}"]
    fixed = "/".join(prefix_parts + suffix_parts)
    if id_segment is None:
        return fixed[:max_len]

    # Espacio disponible para el id_segment respetando el limite de Comet (+1 por el "/").
    available = max_len - len(fixed) - 1
    if available <= 0:
        return fixed[:max_len]
    if len(id_segment) > available:
        id_segment = id_segment[:available]
    return "/".join(prefix_parts + [id_segment] + suffix_parts)


def _collect_dataset_images_labels(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Lee imagenes/etiquetas en orden estable (pre-augmentacion del tf.data)."""
    if hasattr(dataset, "ordered"):
        source = dataset.ordered()
    else:
        source = as_tf_dataset(dataset)

    images_batches: list[np.ndarray] = []
    labels_batches: list[np.ndarray] = []
    for images_batch, labels_batch in source:
        images_batches.append(np.asarray(images_batch))
        labels_batches.append(np.asarray(labels_batch).reshape(-1))

    if not images_batches:
        raise ValueError("El dataset no produjo batches al loguear muestras de entrada.")

    images = np.concatenate(images_batches, axis=0)
    labels = np.concatenate(labels_batches, axis=0).astype(int).reshape(-1)
    if len(images) != len(labels):
        raise ValueError(
            f"Mismatch imagenes ({len(images)}) vs etiquetas ({len(labels)}) al loguear muestras."
        )
    return images, labels


def _array_to_pil_rgb(arr: np.ndarray):
    from PIL import Image

    normalized = _prepare_image_for_logging(arr)
    if normalized.ndim == 2:
        normalized = np.stack([normalized, normalized, normalized], axis=-1)
    elif normalized.shape[-1] == 1:
        normalized = np.repeat(normalized, 3, axis=-1)
    rgb_u8 = (np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(rgb_u8, mode="RGB")


def _sample_metadata(
    *,
    experiment_name: str,
    split_name: str,
    dataset_idx: int,
    label: int,
    row: Any | None,
    image_shape: tuple[int, ...],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "experiment_name": experiment_name,
        "split": split_name,
        "dataset_idx": int(dataset_idx),
        "label": int(label),
        "image_shape": [int(dim) for dim in image_shape],
    }
    if row is None:
        return metadata
    for col in ("patient_id", "image_id", "laterality", "view", "path"):
        if col in row.index and pd.notna(row[col]):
            metadata[col] = str(row[col])
    return metadata


def log_deterministic_input_samples(
    experiment,
    dataset,
    *,
    experiment_name: str | None = None,
    random_seed: int,
    n_samples: int = COMET_N_SAMPLE_IMAGES,
    split_name: str = "train",
) -> None:
    """Loguea a Comet imagenes individuales post-resize / pre-augmentacion."""
    if n_samples <= 0:
        return

    images, y_true = _collect_dataset_images_labels(dataset)
    picked = _fixed_sample_indices(y_true, n_samples=n_samples, random_seed=random_seed)
    if not picked:
        return

    exp_name = _resolve_experiment_name(experiment, experiment_name)
    source_table = getattr(dataset, "source_table", None)
    if source_table is not None:
        source_table = source_table.reset_index(drop=True)

    for slot, idx in enumerate(picked):
        label = int(y_true[idx] >= 0.5)
        row = source_table.iloc[idx] if source_table is not None and idx < len(source_table) else None
        name = _input_sample_name(
            experiment_name=exp_name,
            split_name=split_name,
            dataset_idx=idx,
            label=label,
            row=row,
        )
        pil_image = _array_to_pil_rgb(images[idx])
        metadata = _sample_metadata(
            experiment_name=exp_name,
            split_name=split_name,
            dataset_idx=idx,
            label=label,
            row=row,
            image_shape=tuple(int(dim) for dim in np.asarray(images[idx]).shape),
        )
        metadata["sample_slot"] = int(slot)
        experiment.log_image(
            image_data=pil_image,
            name=name,
            metadata=metadata,
        )


def log_dataset_class_counts(
    experiment,
    *,
    train=None,
    val=None,
    test=None,
    label_column: str = "cls",
) -> None:
    """Loguea cantidad de muestras clase 0 y 1 por split (train/val/test) en Comet."""
    metrics: dict[str, int] = {}
    for split_name, df in (("train", train), ("val", val), ("test", test)):
        if df is None or len(df) == 0:
            continue
        labels = np.asarray(df[label_column]).reshape(-1)
        n_pos = int(np.sum(labels >= 0.5))
        n_neg = int(len(labels) - n_pos)
        metrics[f"{split_name}_neg"] = n_neg
        metrics[f"{split_name}_pos"] = n_pos
        metrics[f"{split_name}_n"] = n_neg + n_pos
    if metrics:
        experiment.log_metrics(metrics)


def log_keras_eval_metrics(
    experiment,
    model,
    *,
    train_ds=None,
    val_ds=None,
    test_ds=None,
) -> None:
    """Loguea metricas nativas de Keras (evaluate) para train/val/test en Comet.

    Para train usa la vista ordenada (sin shuffle) cuando el dataset es
    InspectDataset, de modo que el orden de muestras sea estable.
    """
    for split_name, dataset in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        if dataset is None:
            continue
        eval_ds = dataset.ordered() if split_name == "train" and hasattr(dataset, "ordered") else dataset
        metrics = model.evaluate(eval_ds, return_dict=True)
        experiment.log_metrics(
            {f"{split_name}_{key}": float(value) for key, value in metrics.items()}
        )


def _log_split_eval(
    experiment,
    *,
    backbone_name: str,
    split_name: str,
    y_true,
    y_prob,
    y_pred_default,
    y_pred_youden,
    prob_threshold: float,
    thr_youden: float,
    thr_recall90: float,
    show_plots: bool,
) -> None:
    """Loguea TODO lo de un split: matrices, curvas ROC/PR y escalares clinicos."""
    key = split_name.lower()
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)

    _log_split_confusion_matrices(
        experiment,
        backbone_name=backbone_name,
        split_name=split_name,
        y_true=y_true,
        y_pred_default=y_pred_default,
        y_pred_youden=y_pred_youden,
        prob_threshold=prob_threshold,
        thr_youden=thr_youden,
        show_plots=show_plots,
    )

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))

    # Curvas nativas de Comet: interactivas y comparables entre experimentos.
    experiment.log_curve(f"{key}_roc", x=fpr.tolist(), y=tpr.tolist())
    experiment.log_curve(f"{key}_pr", x=rec.tolist(), y=prec.tolist())

    fig_roc_pr, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4))
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], "--", color="gray")
    ax_roc.set_xlabel("1 - Especificidad (FPR)")
    ax_roc.set_ylabel("Sensibilidad (TPR)")
    ax_roc.set_title(f"ROC — {split_name} ({backbone_name})")
    ax_roc.legend(loc="lower right")
    ax_pr.plot(rec, prec, label=f"AP = {pr_auc:.3f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"Precision-Recall — {split_name}")
    ax_pr.legend(loc="upper right")
    plt.tight_layout()
    experiment.log_figure(figure_name=f"roc_pr_curves_{key}", figure=fig_roc_pr)
    if show_plots:
        plt.show()
    plt.close(fig_roc_pr)

    sens_y, spec_y = _sens_spec(y_true, y_prob, thr_youden)
    sens_r, spec_r = _sens_spec(y_true, y_prob, thr_recall90)

    experiment.log_metrics(
        {
            f"{key}_n": int(len(y_true)),
            f"{key}_neg": int(len(y_true) - np.sum(y_true >= 0.5)),
            f"{key}_pos": int(np.sum(y_true >= 0.5)),
            f"{key}_roc_auc": round(roc_auc, 4),
            f"{key}_pr_auc": round(pr_auc, 4),
            f"{key}_sens_youden": round(float(sens_y), 4),
            f"{key}_spec_youden": round(float(spec_y), 4),
            f"{key}_sens_recall90": round(float(sens_r), 4),
            f"{key}_spec_recall90": round(float(spec_r), 4),
        }
    )


def log_test_results(
    experiment,
    *,
    backbone_name: str,
    dataset,
    y_test_true,
    y_test_prob,
    y_pred_default,
    y_pred_youden,
    thr_youden: float,
    thr_recall90: float,
    prob_threshold: float,
    best_val_metric: float,
    random_seed: int,
    y_val_true=None,
    y_val_prob=None,
    y_val_pred_default=None,
    y_val_pred_youden=None,
    y_train_true=None,
    y_train_prob=None,
    y_train_pred_default=None,
    y_train_pred_youden=None,
    n_sample_images: int = COMET_N_SAMPLE_IMAGES,
    final_weights_path=None,
    show_plots: bool = True,
) -> str:
    """Loguea evaluacion completa a Comet de forma simetrica para train/val/test.

    Para cada split disponible loguea: matrices de confusion (umbral default y
    Youden), curvas ROC/PR nativas + figura, y escalares clinicos prefijados con
    el nombre del split ({split}_roc_auc, {split}_pr_auc, {split}_sens_youden,
    {split}_spec_youden, {split}_sens_recall90, {split}_spec_recall90, {split}_n,
    {split}_pos). Train y val son opcionales: solo se loguean si se pasan sus
    predicciones/probabilidades.
    """
    splits = [
        (
            "Train",
            y_train_true,
            y_train_prob,
            y_train_pred_default,
            y_train_pred_youden,
        ),
        (
            "Val",
            y_val_true,
            y_val_prob,
            y_val_pred_default,
            y_val_pred_youden,
        ),
        (
            "Test",
            y_test_true,
            y_test_prob,
            y_pred_default,
            y_pred_youden,
        ),
    ]

    for split_name, y_true, y_prob, y_pd, y_py in splits:
        if any(v is None for v in (y_true, y_prob, y_pd, y_py)):
            continue
        _log_split_eval(
            experiment,
            backbone_name=backbone_name,
            split_name=split_name,
            y_true=y_true,
            y_prob=y_prob,
            y_pred_default=y_pd,
            y_pred_youden=y_py,
            prob_threshold=prob_threshold,
            thr_youden=thr_youden,
            thr_recall90=thr_recall90,
            show_plots=show_plots,
        )

    # Escalares globales (umbrales elegidos en validacion + mejor metrica de val).
    experiment.log_metrics(
        {
            "val_best_auc": round(float(best_val_metric), 4),
            "thr_youden": round(float(thr_youden), 4),
            "thr_recall90": round(float(thr_recall90), 4),
        }
    )

    if final_weights_path is not None:
        _log_final_weights(
            experiment,
            backbone_name=backbone_name,
            final_weights_path=final_weights_path,
        )

    url = experiment.url
    experiment.end()
    return url
