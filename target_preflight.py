"""Validaciones y estadísticas previas al entrenamiento del A/B."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.dataset import ALL_FINDING_COLUMNS
from src.target_ab import assert_disjoint_patients
from src.targets import validate_birads_labels

FINDING_COLUMNS = ALL_FINDING_COLUMNS


def _split_summary(split_name: str, frame: pd.DataFrame) -> dict[str, Any]:
    labels = frame["cls"].astype(int)
    positives = int(labels.sum())
    images = int(len(frame))
    prevalence = positives / images if images else float("nan")
    return {
        "split": split_name,
        "images": images,
        "patients": int(frame["patient_id"].astype(str).nunique()),
        "positives": positives,
        "negatives": images - positives,
        "prevalence": prevalence,
        "always_negative_accuracy": 1.0 - prevalence,
    }


def _distribution_rows(
    split_name: str,
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    categorical = [
        column
        for column in ("view", "breast_density", "density", "finding_categories")
        if column in frame.columns
    ]
    for class_value, class_frame in frame.groupby("cls", dropna=False):
        class_count = len(class_frame)
        for column in categorical:
            counts = class_frame[column].fillna("<NULL>").astype(str).value_counts()
            for value, count in counts.items():
                rows.append(
                    {
                        "split": split_name,
                        "class": int(class_value),
                        "feature": column,
                        "value": value,
                        "count": int(count),
                        "fraction_within_class": float(count / class_count),
                    }
                )
        for column in FINDING_COLUMNS:
            if column not in class_frame.columns:
                continue
            present = int(pd.to_numeric(class_frame[column], errors="coerce").fillna(0).eq(1).sum())
            rows.append(
                {
                    "split": split_name,
                    "class": int(class_value),
                    "feature": "finding",
                    "value": column,
                    "count": present,
                    "fraction_within_class": float(present / class_count),
                }
            )
    return rows


def validate_ab_preflight(
    *,
    raw_df: pd.DataFrame,
    train_natural: pd.DataFrame,
    train_effective: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str | Path,
    target_mode: str,
    require_files: bool = True,
) -> dict[str, Any]:
    """Falla temprano ante labels/splits inválidos y persiste el diagnóstico."""
    validate_birads_labels(raw_df)
    assert_disjoint_patients(train_effective, val, test)

    frames = {
        "train_natural": train_natural,
        "train": train_effective,
        "validation": val,
        "test": test,
    }
    for split_name, frame in frames.items():
        if frame.empty:
            raise ValueError(f"Split {split_name} vacío")
        if frame["cls"].nunique() != 2:
            raise ValueError(f"Split {split_name} no contiene ambas clases")
        duplicate_images = frame.duplicated(["patient_id", "image_id"]).sum()
        if duplicate_images:
            raise ValueError(
                f"Split {split_name} contiene {int(duplicate_images)} imágenes duplicadas"
            )
        if require_files and "path" in frame.columns:
            missing_files = int((~frame["path"].map(os.path.isfile)).sum())
            if missing_files:
                raise FileNotFoundError(
                    f"Split {split_name}: faltan {missing_files} imágenes"
                )

    summaries = [_split_summary(name, frame) for name, frame in frames.items()]
    distributions = [
        row
        for name, frame in frames.items()
        for row in _distribution_rows(name, frame)
    ]
    by_split = {row["split"]: row for row in summaries}
    result = {
        "target_mode": target_mode,
        "birads_invalid_or_null": 0,
        "patient_overlap": 0,
        "natural_prevalence": by_split["train_natural"]["prevalence"],
        "effective_prevalence": by_split["train"]["prevalence"],
        "splits": summaries,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output_dir / "preflight_counts.csv", index=False)
    pd.DataFrame(distributions).to_csv(
        output_dir / "preflight_distributions.csv", index=False
    )
    (output_dir / "preflight.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result
