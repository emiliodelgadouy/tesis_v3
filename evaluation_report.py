"""Evaluación reproducible para el experimento A/B de etiquetas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def select_validation_thresholds(
    y_true,
    y_prob,
) -> dict[str, float]:
    """Selecciona todos los umbrales usando exclusivamente validación."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1.0 - fpr

    balanced_index = int(np.nanargmax((tpr + specificity) / 2.0))

    def threshold_for_specificity(target: float) -> float:
        candidates = np.flatnonzero(specificity >= target)
        if not len(candidates):
            return float("inf")
        best = candidates[np.argmax(tpr[candidates])]
        return float(thresholds[best])

    def threshold_for_sensitivity(target: float) -> float:
        candidates = np.flatnonzero(tpr >= target)
        if not len(candidates):
            return float("-inf")
        best = candidates[np.argmax(specificity[candidates])]
        return float(thresholds[best])

    return {
        "balanced_accuracy": float(thresholds[balanced_index]),
        "specificity_90": threshold_for_specificity(0.90),
        "specificity_95": threshold_for_specificity(0.95),
        "sensitivity_90": threshold_for_sensitivity(0.90),
    }


def compute_metrics(
    frame: pd.DataFrame,
    thresholds: dict[str, float],
) -> dict[str, float]:
    """Calcula métricas sin seleccionar nada sobre este DataFrame."""
    y_true = frame["y_true"].to_numpy(np.int64)
    y_prob = frame["y_prob"].to_numpy(np.float64)
    if len(np.unique(y_true)) < 2:
        return {}

    prevalence = float(y_true.mean())
    pr_auc = float(average_precision_score(y_true, y_prob))
    balanced_pred = y_prob >= thresholds["balanced_accuracy"]
    cm = confusion_matrix(y_true, balanced_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in cm.ravel())

    pred_spec90 = y_prob >= thresholds["specificity_90"]
    cm_spec90 = confusion_matrix(y_true, pred_spec90, labels=[0, 1])
    tn90, fp90, fn90, tp90 = (int(value) for value in cm_spec90.ravel())

    pred_spec95 = y_prob >= thresholds["specificity_95"]
    cm_spec95 = confusion_matrix(y_true, pred_spec95, labels=[0, 1])
    tn95, fp95, fn95, tp95 = (int(value) for value in cm_spec95.ravel())

    pred_sens90 = y_prob >= thresholds["sensitivity_90"]
    cm_sens90 = confusion_matrix(y_true, pred_sens90, labels=[0, 1])
    tn_s90, fp_s90, fn_s90, tp_s90 = (
        int(value) for value in cm_sens90.ravel()
    )

    return {
        "n": int(len(frame)),
        "patients": int(frame["patient_id"].astype(str).nunique()),
        "positives": int(y_true.sum()),
        "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": pr_auc,
        "pr_lift": _safe_divide(pr_auc, prevalence),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, balanced_pred)),
        "accuracy": float(accuracy_score(y_true, balanced_pred)),
        "sensitivity": _safe_divide(tp, tp + fn),
        "specificity": _safe_divide(tn, tn + fp),
        "sensitivity_at_val_spec90": _safe_divide(tp90, tp90 + fn90),
        "specificity_at_val_spec90": _safe_divide(tn90, tn90 + fp90),
        "sensitivity_at_val_spec95": _safe_divide(tp95, tp95 + fn95),
        "specificity_at_val_spec95": _safe_divide(tn95, tn95 + fp95),
        "specificity_at_val_sens90": _safe_divide(tn_s90, tn_s90 + fp_s90),
        "sensitivity_at_val_sens90": _safe_divide(tp_s90, tp_s90 + fn_s90),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def bootstrap_metrics_by_patient(
    frame: pd.DataFrame,
    thresholds: dict[str, float],
    *,
    samples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap cluster: remuestrea pacientes y conserva todas sus imágenes."""
    groups = [
        group.reset_index(drop=True)
        for _, group in frame.groupby(frame["patient_id"].astype(str), sort=True)
    ]
    if not groups:
        raise ValueError("No hay pacientes para bootstrap")
    rng = np.random.default_rng(seed)
    rows = []
    for sample_index in range(samples):
        sampled_indices = rng.integers(0, len(groups), size=len(groups))
        sampled = pd.concat(
            [groups[index] for index in sampled_indices], ignore_index=True
        )
        metrics = compute_metrics(sampled, thresholds)
        if metrics:
            metrics["bootstrap_sample"] = sample_index
            rows.append(metrics)
    return pd.DataFrame(rows)


def confidence_intervals(
    bootstrap: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for column in bootstrap.select_dtypes(include=[np.number]).columns:
        if column in {
            "bootstrap_sample",
            "n",
            "patients",
            "positives",
            "tn",
            "fp",
            "fn",
            "tp",
        }:
            continue
        values = bootstrap[column].replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        result[f"{column}_ci_low"] = float(values.quantile(alpha / 2.0))
        result[f"{column}_ci_high"] = float(values.quantile(1.0 - alpha / 2.0))
    return result


def subgroup_metrics(
    frame: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    rows = [{"stratum": "ALL", "value": "ALL", **compute_metrics(frame, thresholds)}]
    for column in ("view", "breast_density", "density"):
        if column not in frame.columns:
            continue
        for value, group in frame.groupby(column, dropna=False):
            metrics = compute_metrics(group, thresholds)
            rows.append(
                {
                    "stratum": column,
                    "value": str(value),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def plot_roc_pr(
    frames: dict[str, pd.DataFrame],
    output_path: str | Path,
    *,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, frame in frames.items():
        y_true = frame["y_true"].to_numpy(np.int64)
        y_prob = frame["y_prob"].to_numpy(np.float64)
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        axes[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_true, y_prob):.3f})")
        axes[1].plot(
            recall,
            precision,
            label=f"{name} ({average_precision_score(y_true, y_prob):.3f})",
        )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set(xlabel="1 - especificidad", ylabel="Sensibilidad", title="ROC")
    axes[1].set(xlabel="Sensibilidad", ylabel="Precisión", title="Precision–Recall")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(
    frame: pd.DataFrame,
    threshold: float,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    y_true = frame["y_true"].to_numpy(np.int64)
    y_pred = frame["y_prob"].to_numpy(np.float64) >= threshold
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, str(value), ha="center", va="center")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Neg", "Pos"],
        yticklabels=["Neg", "Pos"],
        xlabel="Predicción",
        ylabel="Real",
        title=title,
    )
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
