"""Utilidades puras para el experimento A/B de etiquetas."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PATIENT_SPLIT_NAMES = ("train", "val", "test")


def _canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _existing_patient_assignments(payload: dict[str, Any]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for split_name in PATIENT_SPLIT_NAMES:
        for record in payload.get(split_name, []):
            patient_id = str(record["patient_id"])
            previous = assignments.setdefault(patient_id, split_name)
            if previous != split_name:
                raise ValueError(
                    f"Paciente {patient_id} aparece en {previous} y {split_name}"
                )
    return assignments


def build_patient_split_manifest(
    raw_df: pd.DataFrame,
    canonical_split_payload: dict[str, Any],
    *,
    validation_ratio: float = 0.20,
    seed: int = 42,
    patient_id_column: str = "patient_id",
    upstream_split_column: str = "split",
) -> dict[str, Any]:
    """Extiende el split mass existente a todos los pacientes, una sola vez."""
    required = [patient_id_column, upstream_split_column]
    missing = [column for column in required if column not in raw_df.columns]
    if missing:
        raise KeyError(f"Faltan columnas para crear el manifiesto A/B: {missing}")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio debe estar en (0, 1)")

    patient_source = raw_df[required].drop_duplicates()
    inconsistent = patient_source.groupby(patient_id_column)[
        upstream_split_column
    ].nunique()
    if inconsistent.gt(1).any():
        raise ValueError(
            f"{int(inconsistent.gt(1).sum())} pacientes cruzan splits upstream"
        )
    upstream = (
        patient_source.drop_duplicates(patient_id_column)
        .set_index(patient_id_column)[upstream_split_column]
        .astype(str)
        .to_dict()
    )
    assignments = _existing_patient_assignments(canonical_split_payload)

    unknown_existing = sorted(set(assignments).difference(upstream))
    if unknown_existing:
        raise ValueError(
            f"{len(unknown_existing)} pacientes del split canónico no están en metadata"
        )
    conflicts = [
        patient_id
        for patient_id, split_name in assignments.items()
        if upstream[patient_id] == "test" and split_name != "test"
    ]
    if conflicts:
        raise ValueError(
            f"{len(conflicts)} pacientes upstream test no están en test canónico"
        )

    for patient_id, source_split in upstream.items():
        if source_split == "test":
            previous = assignments.setdefault(patient_id, "test")
            if previous != "test":
                raise ValueError(
                    f"Paciente upstream test {patient_id} asignado a {previous}"
                )
        elif source_split != "training":
            raise ValueError(f"Split upstream desconocido: {source_split!r}")

    training_patients = sorted(
        patient_id
        for patient_id, source_split in upstream.items()
        if source_split == "training"
    )
    missing_training = [
        patient_id for patient_id in training_patients if patient_id not in assignments
    ]
    existing_val = sum(assignments.get(patient_id) == "val" for patient_id in training_patients)
    target_val = round(len(training_patients) * validation_ratio)
    missing_val_count = min(
        len(missing_training), max(0, target_val - existing_val)
    )

    # Orden pseudoaleatorio estable, independiente de las etiquetas.
    ranked_missing = sorted(
        missing_training,
        key=lambda patient_id: hashlib.sha256(
            f"{seed}:{patient_id}".encode("utf-8")
        ).hexdigest(),
    )
    missing_val = set(ranked_missing[:missing_val_count])
    for patient_id in missing_training:
        assignments[patient_id] = "val" if patient_id in missing_val else "train"

    records = [
        {"patient_id": patient_id, "split": assignments[patient_id]}
        for patient_id in sorted(assignments)
    ]
    counts = {
        split_name: sum(record["split"] == split_name for record in records)
        for split_name in PATIENT_SPLIT_NAMES
    }
    manifest = {
        "meta": {
            "version": 1,
            "seed": int(seed),
            "validation_ratio": float(validation_ratio),
            "strategy": "preserve_canonical_then_hash_extend",
            "canonical_split_sha256": _canonical_json_hash(canonical_split_payload),
            "patient_counts": counts,
            "new_training_patients": len(missing_training),
            "new_validation_patients": missing_val_count,
        },
        "patients": records,
    }
    manifest["meta"]["manifest_sha256"] = _canonical_json_hash(records)
    return manifest


def save_patient_split_manifest(
    manifest: dict[str, Any], path: str | Path, *, overwrite: bool = False
) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if _canonical_json_hash(existing) != _canonical_json_hash(manifest):
            raise FileExistsError(
                f"El manifiesto existente difiere y no se regenerará: {path}"
            )
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_patient_split_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No existe el manifiesto A/B: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = manifest.get("patients", [])
    expected = (manifest.get("meta") or {}).get("manifest_sha256")
    actual = _canonical_json_hash(records)
    if expected != actual:
        raise ValueError(
            f"Hash inválido en manifiesto A/B: esperado={expected}, actual={actual}"
        )
    _manifest_assignment_map(manifest)
    return manifest


def _manifest_assignment_map(manifest: dict[str, Any]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for record in manifest.get("patients", []):
        patient_id = str(record["patient_id"])
        split_name = str(record["split"])
        if split_name not in PATIENT_SPLIT_NAMES:
            raise ValueError(f"Split inválido en manifiesto: {split_name!r}")
        previous = assignments.setdefault(patient_id, split_name)
        if previous != split_name:
            raise ValueError(f"Paciente duplicado con splits distintos: {patient_id}")
    if not assignments:
        raise ValueError("El manifiesto A/B no contiene pacientes")
    return assignments


def apply_patient_split_manifest(
    target_df: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    patient_id_column: str = "patient_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aplica el mismo mapa por paciente a cualquier definición de target."""
    assignments = _manifest_assignment_map(manifest)
    out = target_df.copy()
    out["_ab_split"] = out[patient_id_column].astype(str).map(assignments)
    missing = out["_ab_split"].isna()
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} imágenes pertenecen a pacientes fuera del manifiesto"
        )
    splits = tuple(
        out.loc[out["_ab_split"] == split_name]
        .drop(columns="_ab_split")
        .reset_index(drop=True)
        for split_name in PATIENT_SPLIT_NAMES
    )
    return splits


def undersample_training_negatives(
    train_df: pd.DataFrame,
    *,
    max_negatives_per_positive: float = 3.0,
    seed: int = 42,
    label_column: str = "cls",
) -> pd.DataFrame:
    """Limita negativos por positivo y conserva todos los positivos."""
    positive = train_df[train_df[label_column].eq(1)]
    negative = train_df[train_df[label_column].eq(0)]
    if positive.empty or negative.empty:
        raise ValueError("Train necesita ambas clases antes del undersampling")
    max_negative = int(np.floor(len(positive) * max_negatives_per_positive))
    if len(negative) > max_negative:
        negative = negative.sample(n=max_negative, random_state=seed)
    return (
        pd.concat([positive, negative], ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def assert_disjoint_patients(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    patient_id_column: str = "patient_id",
) -> None:
    patient_sets = {
        "train": set(train[patient_id_column].astype(str)),
        "val": set(val[patient_id_column].astype(str)),
        "test": set(test[patient_id_column].astype(str)),
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = patient_sets[left] & patient_sets[right]
        if overlap:
            raise ValueError(
                f"Leakage: {len(overlap)} pacientes compartidos entre {left} y {right}"
            )
