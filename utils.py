from __future__ import annotations

import time
from typing import Literal

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tensorflow import keras

from src.dataset import DEFAULT_CLS_POSITIVE_COLUMNS, DEFAULT_FILTER_COLUMNS
from src.dataset_provider import apply_clahe_tf, decode_image

__all__ = [
    "EpochTimer",
    "apply_clahe_tf",
    "decode_image",
    "deduplicate_images",
    "stratified_split",
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
