"""Métricas y evaluación offline para clasificación por parches."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from tensorflow import keras

from src.plots import plot_history


def _as_numpy(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


class OneVsRestAUC(keras.metrics.AUC):
    """AUC binaria: una clase vs el resto (etiquetas one-hot)."""

    def __init__(self, class_index: int, name="ovr_auc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_index = class_index

    def update_state(self, y_true, y_pred, sample_weight=None):
        idx = self.class_index
        super().update_state(y_true[:, idx], y_pred[:, idx], sample_weight)


def predict_multiclass(model, dataset, *, has_sample_weight: bool = False):
    y_true, y_prob = [], []
    for batch in dataset:
        if has_sample_weight:
            xb, yb, _ = batch
        else:
            xb, yb = batch
        prob = _as_numpy(model(xb, training=False))
        y_true.append(np.argmax(_as_numpy(yb), axis=-1))
        y_prob.append(prob)
    return np.concatenate(y_true), np.concatenate(y_prob)


def predict_multihead(model, dataset, head: str):
    y_true, y_prob = [], []
    for xb, yb in dataset:
        out = model(xb, training=False)
        if not isinstance(out, dict):
            raise TypeError(f"Salida multhead esperada dict, recibió {type(out)}")
        prob = _as_numpy(out[head])
        y_true.append(np.argmax(_as_numpy(yb[head]), axis=-1))
        y_prob.append(prob)
    return np.concatenate(y_true), np.concatenate(y_prob)


def evaluate_multiclass(
    model,
    dataset,
    class_names,
    *,
    title: str,
    has_sample_weight: bool = False,
    positive_class: int | None = None,
):
    y_true, y_prob = predict_multiclass(
        model, dataset, has_sample_weight=has_sample_weight
    )
    y_pred = y_prob.argmax(axis=1)
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            digits=3,
            zero_division=0,
        )
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"Macro F1: {macro_f1:.3f} | Weighted F1: {weighted_f1:.3f}")
    if positive_class is not None and len(np.unique(y_true)) > 1:
        y_bin = (y_true == positive_class).astype(np.int32)
        try:
            auc = roc_auc_score(y_bin, y_prob[:, positive_class])
            print(f"AUC clase '{class_names[positive_class]}': {auc:.3f}")
        except ValueError:
            print(f"AUC clase '{class_names[positive_class]}': n/a (una sola clase)")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))),
        display_labels=class_names,
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}


def evaluate_binary_head(
    model,
    dataset,
    head: str,
    class_names,
    *,
    title: str,
):
    y_true, y_prob = predict_multihead(model, dataset, head)
    y_pred = y_prob.argmax(axis=1)
    print(f"\n{'=' * 60}\n{title} — {head}\n{'=' * 60}")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=3,
            zero_division=0,
        )
    )
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob[:, 1])
        print(f"AUC (clase positiva={class_names[1]}): {auc:.3f}")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred, labels=[0, 1]),
        display_labels=class_names,
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{title} — {head}")
    plt.tight_layout()
    plt.show()
    return {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}


__all__ = [
    "OneVsRestAUC",
    "evaluate_binary_head",
    "evaluate_multiclass",
    "plot_history",
    "predict_multiclass",
    "predict_multihead",
]
