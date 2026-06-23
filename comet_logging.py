"""Configuracion y helpers de Comet ML para el notebook de tesis."""

from __future__ import annotations

import os
from typing import Any

import comet_ml
import matplotlib.pyplot as plt
import numpy as np
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
from src.utils import apply_probability_threshold

# Defaults; el notebook puede sobreescribir COMET_* y pasarlos a login()/start_training_experiment().
COMET_API_KEY = os.environ.get("COMET_API_KEY", "")
COMET_PROJECT = "tesis-reni"
COMET_N_SAMPLE_IMAGES = 8


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


def start_training_experiment(
    *,
    experiment_name: str,
    run_config: dict[str, Any],
    backbone_description: str,
    architecture_catalog: str,
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
    return experiment


def _plot_confusion_matrix(y_true, y_pred, *, title: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred, labels=[0, 1]),
        display_labels=["Neg", "Pos"],
    ).plot(ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    return fig


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


def _sens_spec(y_true, y_prob, thr: float) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = (np.asarray(y_prob) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return sens, spec


def _log_sample_predictions(
    experiment,
    dataset,
    y_true,
    y_prob,
    threshold: float,
    *,
    random_seed: int,
    n_samples: int = COMET_N_SAMPLE_IMAGES,
):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    y_pred = apply_probability_threshold(y_prob, threshold)

    images = np.concatenate(
        [np.asarray(x_batch) for x_batch, _ in as_tf_dataset(dataset)],
        axis=0,
    )
    if len(images) != len(y_true):
        raise ValueError(
            f"Mismatch imagenes ({len(images)}) vs etiquetas ({len(y_true)}) al loguear muestras."
        )

    groups = {
        "TP": np.flatnonzero((y_true == 1) & (y_pred == 1)),
        "TN": np.flatnonzero((y_true == 0) & (y_pred == 0)),
        "FP": np.flatnonzero((y_true == 0) & (y_pred == 1)),
        "FN": np.flatnonzero((y_true == 1) & (y_pred == 0)),
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
    picked = picked[:n_samples]

    n_cols = min(4, max(1, len(picked)))
    n_rows = int(np.ceil(len(picked) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, idx in zip(axes, picked):
        img = images[idx]
        if img.ndim == 4:
            img = _bag_to_montage(img)
        if img.dtype in (np.float16, np.float32, np.float64):
            img = img / 255.0 if img.max() > 1.0 else img
            img = np.clip(img.astype(np.float32), 0.0, 1.0)
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(
            f"y={y_true[idx]} | p={y_prob[idx]:.3f} | pred={y_pred[idx]}",
            fontsize=9,
        )
        ax.axis("off")
        experiment.log_image(
            img,
            name=f"samples/test_{idx:04d}",
            metadata={
                "y_true": int(y_true[idx]),
                "prob": float(y_prob[idx]),
                "pred": int(y_pred[idx]),
            },
        )

    for ax in axes[len(picked) :]:
        ax.axis("off")

    fig.suptitle(f"Muestras test — umbral Youden J = {threshold:.4f}", y=1.02)
    plt.tight_layout()
    return fig


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
    n_sample_images: int = COMET_N_SAMPLE_IMAGES,
    show_plots: bool = True,
) -> str:
    experiment.log_confusion_matrix(
        y_true=y_test_true,
        y_predicted=y_pred_default,
        labels=["Neg", "Pos"],
        title=f"Test — umbral {prob_threshold}",
        file_name="confusion_matrix_test_default.json",
    )
    experiment.log_confusion_matrix(
        y_true=y_test_true,
        y_predicted=y_pred_youden,
        labels=["Neg", "Pos"],
        title=f"Test — umbral Youden J = {thr_youden:.4f}",
        file_name="confusion_matrix_test_youden.json",
    )

    fig_cm_default = _plot_confusion_matrix(
        y_test_true,
        y_pred_default,
        title=f"{backbone_name} — Test — umbral = {prob_threshold}",
    )
    experiment.log_figure(figure_name="confusion_matrix_test_default", figure=fig_cm_default)
    if show_plots:
        plt.show()
    plt.close(fig_cm_default)

    fig_cm_youden = _plot_confusion_matrix(
        y_test_true,
        y_pred_youden,
        title=f"{backbone_name} — Test — umbral Youden J = {thr_youden:.4f}",
    )
    experiment.log_figure(figure_name="confusion_matrix_test_youden", figure=fig_cm_youden)
    if show_plots:
        plt.show()
    plt.close(fig_cm_youden)

    fpr, tpr, _ = roc_curve(y_test_true, y_test_prob)
    prec, rec, _ = precision_recall_curve(y_test_true, y_test_prob)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4))
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test_true, y_test_prob):.3f}")
    ax_roc.plot([0, 1], [0, 1], "--", color="gray")
    ax_roc.set_xlabel("1 - Especificidad (FPR)")
    ax_roc.set_ylabel("Sensibilidad (TPR)")
    ax_roc.set_title(f"ROC — Test ({backbone_name})")
    ax_roc.legend(loc="lower right")
    ax_pr.plot(rec, prec, label=f"AP = {average_precision_score(y_test_true, y_test_prob):.3f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall — Test")
    ax_pr.legend(loc="upper right")
    plt.tight_layout()
    experiment.log_figure(figure_name="roc_pr_curves_test", figure=fig)
    if show_plots:
        plt.show()
    plt.close(fig)

    fig_samples = _log_sample_predictions(
        experiment,
        dataset,
        y_test_true,
        y_test_prob,
        thr_youden,
        random_seed=random_seed,
        n_samples=n_sample_images,
    )
    experiment.log_figure(figure_name="sample_predictions_test", figure=fig_samples)
    if show_plots:
        plt.show()
    plt.close(fig_samples)

    sens_y, spec_y = _sens_spec(y_test_true, y_test_prob, thr_youden)
    sens_r, spec_r = _sens_spec(y_test_true, y_test_prob, thr_recall90)

    experiment.log_metrics(
        {
            "n_test": int(len(y_test_true)),
            "test_pos": int(np.sum(np.asarray(y_test_true) >= 0.5)),
            "val_best_auc": round(float(best_val_metric), 4),
            "test_roc_auc": round(float(roc_auc_score(y_test_true, y_test_prob)), 4),
            "test_pr_auc": round(float(average_precision_score(y_test_true, y_test_prob)), 4),
            "thr_youden": round(thr_youden, 4),
            "sens_youden": round(float(sens_y), 4),
            "spec_youden": round(float(spec_y), 4),
            "thr_recall90": round(thr_recall90, 4),
            "sens_recall90": round(float(sens_r), 4),
            "spec_recall90": round(float(spec_r), 4),
        }
    )

    url = experiment.url
    experiment.end()
    return url
