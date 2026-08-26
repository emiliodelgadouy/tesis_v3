"""Definiciones puras de los objetivos binarios del proyecto."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

TARGET_MASS = "mass"
TARGET_BIRADS_RECALL = "birads_recall"
LEGACY_TARGET_FULL = "full"
VALID_TARGET_MODES = (TARGET_MASS, TARGET_BIRADS_RECALL, LEGACY_TARGET_FULL)

BIRADS_NEGATIVE = frozenset({"BI-RADS 1", "BI-RADS 2"})
BIRADS_POSITIVE = frozenset({"BI-RADS 3", "BI-RADS 4", "BI-RADS 5"})
BIRADS_VALID = BIRADS_NEGATIVE | BIRADS_POSITIVE


def resolve_target_mode(config: Mapping) -> str:
    """Resuelve TARGET_MODE y conserva POSITIVE_MODE como alias legacy."""
    general = config.get("GENERAL", config)
    target = general.get("TARGET_MODE")
    legacy = general.get("POSITIVE_MODE")
    if target is None:
        target = legacy
    if target is None:
        raise KeyError("Falta GENERAL.TARGET_MODE (o el alias legacy POSITIVE_MODE)")
    target = str(target).strip().lower().replace("-", "_")
    if target not in VALID_TARGET_MODES:
        raise ValueError(
            f"TARGET_MODE desconocido: {target!r}; usar {VALID_TARGET_MODES}"
        )
    return target


def source_frame_name(target_mode: str) -> str:
    """Nombre del frame de ``download_and_build_dataset`` para este target."""
    target_mode = str(target_mode).strip().lower().replace("-", "_")
    if target_mode == TARGET_BIRADS_RECALL:
        return "ds_raw"
    if target_mode in (TARGET_MASS, LEGACY_TARGET_FULL):
        return "ds"
    raise ValueError(
        f"TARGET_MODE desconocido: {target_mode!r}; usar {VALID_TARGET_MODES}"
    )


def validate_birads_labels(
    df: pd.DataFrame,
    *,
    birads_column: str = "breast_birads",
    patient_id_column: str = "patient_id",
    image_id_column: str = "image_id",
) -> None:
    """Valida cobertura, vocabulario y consistencia BI-RADS por imagen."""
    required = [patient_id_column, image_id_column, birads_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas para birads_recall: {missing}")

    values = df[birads_column]
    null_count = int(values.isna().sum())
    invalid_values = sorted(
        str(value) for value in values.dropna().unique() if value not in BIRADS_VALID
    )
    if null_count or invalid_values:
        raise ValueError(
            "breast_birads contiene valores nulos o inválidos: "
            f"nulls={null_count}, invalid={invalid_values}"
        )

    inconsistent = (
        df.groupby([patient_id_column, image_id_column], dropna=False)[birads_column]
        .nunique(dropna=False)
        .gt(1)
    )
    if inconsistent.any():
        raise ValueError(
            f"{int(inconsistent.sum())} imágenes tienen breast_birads inconsistente"
        )


def target_masks(
    df: pd.DataFrame,
    target_mode: str,
    *,
    birads_column: str = "breast_birads",
) -> tuple[pd.Series, pd.Series]:
    """Devuelve máscaras (positiva, negativa) antes de deduplicar imágenes."""
    target_mode = str(target_mode).strip().lower().replace("-", "_")
    if target_mode == TARGET_MASS:
        return df["Mass"].eq(1), df["cls"].eq(0)
    if target_mode == LEGACY_TARGET_FULL:
        return df["cls"].eq(1), df["cls"].eq(0)
    if target_mode == TARGET_BIRADS_RECALL:
        validate_birads_labels(df, birads_column=birads_column)
        values = df[birads_column]
        return values.isin(BIRADS_POSITIVE), values.isin(BIRADS_NEGATIVE)
    raise ValueError(
        f"TARGET_MODE desconocido: {target_mode!r}; usar {VALID_TARGET_MODES}"
    )
