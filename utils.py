from __future__ import annotations

import time
from typing import Literal

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
from src.dataset_provider import apply_clahe_tf, as_tf_dataset, decode_image

__all__ = [
    "EpochTimer",
    "apply_clahe_tf",
    "apply_probability_threshold",
    "binary_report",
    "decode_image",
    "deduplicate_images",
    "labels_from_tf_dataset",
    "logit_initial_bias",
    "plot_binary_confusion_matrix",
    "predict_probabilities",
    "predict_probabilities_tta",
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
    keep: Literal["first", "last"] = "first",
) -> pd.DataFrame:
    """One row per image: collapse CSV rows with multiple findings/boxes."""
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


def logit_initial_bias(n_positive: int, n_negative: int) -> float:
    return float(np.log(n_positive / n_negative))


def _sigmoid(z) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def labels_from_tf_dataset(dataset) -> np.ndarray:
    return np.concatenate(
        [np.asarray(yb, dtype=np.int64).reshape(-1) for _, yb in dataset]
    )


def predict_probabilities(model, dataset, *, verbose: int = 1) -> np.ndarray:
    logits = model.predict(dataset, verbose=verbose).astype(np.float64).reshape(-1)
    return _sigmoid(logits)


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


def predict_probabilities_tta(model, dataset) -> np.ndarray:
    """TTA por flip horizontal: promedia probabilidades de imagen normal + espejada."""
    base = as_tf_dataset(dataset)
    flipped = base.map(
        lambda x, y: (tf.image.flip_left_right(x), y),
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


def warmup(model, train_ds) -> None:
    """Un step dummy para disparar la compilacion del grafo antes de model.fit()."""
    t0 = time.perf_counter()
    xb, yb = next(iter(as_tf_dataset(train_ds)))
    model.model.train_on_batch(xb, yb)
    print(f"  Warm-up completado en {time.perf_counter() - t0:.1f}s")


class EpochTimer(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []
        self.total_elapsed_times = []
        self._fit_start = time.perf_counter()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.perf_counter() - self.epoch_start_time
        self.epoch_times.append(elapsed)
        total = time.perf_counter() - self._fit_start
        self.total_elapsed_times.append(total)
        if logs is not None:
            logs["epoch_time_seconds"] = elapsed
            logs["total_elapsed_seconds"] = total
