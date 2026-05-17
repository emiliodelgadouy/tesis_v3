from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.dataset import DEFAULT_CLS_POSITIVE_COLUMNS, DEFAULT_FILTER_COLUMNS

Severity = Literal["error", "warning", "info"]


@dataclass
class ValidationIssue:
    code: str
    severity: Severity
    message: str
    count: int = 0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetValidationReport:
    name: str
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def merge(self, other: DatasetValidationReport) -> DatasetValidationReport:
        return DatasetValidationReport(
            name=f"{self.name} + {other.name}",
            issues=[*self.issues, *other.issues],
        )


def _issue(
    code: str,
    severity: Severity,
    message: str,
    *,
    count: int = 0,
    **details: Any,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        severity=severity,
        message=message,
        count=count,
        details=details,
    )


def _sample_records(df: pd.DataFrame, columns: list[str], n: int = 5) -> list[dict[str, Any]]:
    if df.empty:
        return []
    cols = [column for column in columns if column in df.columns]
    sample = df.head(n)
    if not cols:
        return sample.to_dict(orient="records")
    return sample[cols].to_dict(orient="records")


def validate_dataframe(
    df: pd.DataFrame,
    *,
    name: str = "dataset",
    path_column: str = "path",
    label_column: str = "cls",
    group_column: str = "patient_id",
    image_id_column: str = "image_id",
    cls_positive_columns: tuple[str, ...] = DEFAULT_CLS_POSITIVE_COLUMNS,
    check_files: bool = True,
    max_missing_file_samples: int = 5,
) -> DatasetValidationReport:
    """Valida colisiones, contradicciones de etiqueta y archivos faltantes en una tabla."""
    report = DatasetValidationReport(name=name)

    if df.empty:
        report.issues.append(
            _issue("empty_dataframe", "error", f"{name}: el DataFrame está vacío.")
        )
        return report

    required = {path_column, label_column, group_column, image_id_column}
    missing = sorted(required - set(df.columns))
    if missing:
        report.issues.append(
            _issue(
                "missing_columns",
                "error",
                f"{name}: faltan columnas obligatorias {missing}.",
                columns=missing,
            )
        )
        return report

    dup_mask = df.duplicated(subset=[path_column], keep=False)
    dup_rows = int(dup_mask.sum())
    if dup_rows:
        dup_paths = int(df.loc[dup_mask, path_column].nunique())
        report.issues.append(
            _issue(
                "duplicate_paths",
                "warning",
                f"{name}: {dup_rows} filas comparten {dup_paths} paths duplicados "
                "(misma imagen repetida en la tabla).",
                count=dup_rows,
                unique_paths=dup_paths,
                samples=_sample_records(
                    df.loc[dup_mask].sort_values(path_column),
                    [path_column, label_column, group_column, image_id_column],
                ),
            )
        )

    key_dup_mask = df.duplicated(subset=[group_column, image_id_column], keep=False)
    key_dup_rows = int(key_dup_mask.sum())
    if key_dup_rows:
        report.issues.append(
            _issue(
                "duplicate_patient_image",
                "warning",
                f"{name}: {key_dup_rows} filas repiten la pareja "
                f"({group_column}, {image_id_column}).",
                count=key_dup_rows,
                samples=_sample_records(
                    df.loc[key_dup_mask].sort_values([group_column, image_id_column]),
                    [path_column, label_column, group_column, image_id_column],
                ),
            )
        )

    label_variants = df.groupby(path_column, dropna=False)[label_column].nunique()
    conflicting_paths = label_variants[label_variants > 1]
    if not conflicting_paths.empty:
        conflict_rows = df[df[path_column].isin(conflicting_paths.index)]
        report.issues.append(
            _issue(
                "label_conflict_on_path",
                "error",
                f"{name}: {len(conflicting_paths)} paths tienen etiquetas {label_column} "
                "inconsistentes.",
                count=int(len(conflicting_paths)),
                samples=_sample_records(
                    conflict_rows.sort_values(path_column),
                    [path_column, label_column, group_column, image_id_column],
                ),
            )
        )

    available_positive = [column for column in cls_positive_columns if column in df.columns]
    if available_positive and label_column in df.columns:
        expected_positive = df[available_positive].eq(1).any(axis=1)
        expected_cls = expected_positive.astype(df[label_column].dtype)
        mismatch_mask = df[label_column] != expected_cls
        mismatch_count = int(mismatch_mask.sum())
        if mismatch_count:
            report.issues.append(
                _issue(
                    "cls_positive_columns_mismatch",
                    "error",
                    f"{name}: {mismatch_count} filas tienen {label_column} "
                    f"incompatible con {available_positive}.",
                    count=mismatch_count,
                    samples=_sample_records(
                        df.loc[mismatch_mask],
                        [path_column, label_column, *available_positive[:3]],
                    ),
                )
            )

    if "No_Finding" in df.columns and available_positive:
        both_mask = df["No_Finding"].eq(1) & df[available_positive].eq(1).any(axis=1)
        both_count = int(both_mask.sum())
        if both_count:
            report.issues.append(
                _issue(
                    "no_finding_with_positive",
                    "warning",
                    f"{name}: {both_count} filas marcan No_Finding=1 y también un hallazgo positivo.",
                    count=both_count,
                    samples=_sample_records(
                        df.loc[both_mask],
                        [path_column, label_column, "No_Finding", available_positive[0]],
                    ),
                )
            )

    filter_cols = [column for column in DEFAULT_FILTER_COLUMNS if column in df.columns]
    if filter_cols and label_column in df.columns:
        has_finding_flag = df[filter_cols].eq(1).any(axis=1)
        no_rows_with_cls = (~has_finding_flag) & df[label_column].notna()
        orphan_count = int(no_rows_with_cls.sum())
        if orphan_count:
            report.issues.append(
                _issue(
                    "row_without_finding_flag",
                    "warning",
                    f"{name}: {orphan_count} filas no activan ninguna columna de "
                    f"filtrado {filter_cols}.",
                    count=orphan_count,
                )
            )

    if check_files:
        missing_mask = ~df[path_column].map(lambda value: Path(str(value)).is_file())
        missing_count = int(missing_mask.sum())
        if missing_count:
            report.issues.append(
                _issue(
                    "missing_image_files",
                    "error",
                    f"{name}: {missing_count} paths no existen en disco.",
                    count=missing_count,
                    samples=_sample_records(
                        df.loc[missing_mask],
                        [path_column, label_column, group_column],
                        n=max_missing_file_samples,
                    ),
                )
            )

    if not report.issues:
        report.issues.append(
            _issue(
                "dataframe_ok",
                "info",
                f"{name}: sin colisiones críticas ni contradicciones detectadas "
                f"({len(df)} filas).",
                count=len(df),
            )
        )

    return report


def validate_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    name: str = "splits",
    path_column: str = "path",
    label_column: str = "cls",
    group_column: str = "patient_id",
    image_id_column: str = "image_id",
) -> DatasetValidationReport:
    """Detecta fugas entre train/val/test (mismo paciente, path o imagen en varios splits)."""
    report = DatasetValidationReport(name=name)
    splits: dict[str, pd.DataFrame] = {
        "train": train,
        "val": val,
        "test": test,
    }

    for split_name, frame in splits.items():
        if frame.empty:
            report.issues.append(
                _issue(
                    "empty_split",
                    "warning",
                    f"{name}: el split {split_name!r} está vacío.",
                    split=split_name,
                )
            )

    pair_names = (("train", "val"), ("train", "test"), ("val", "test"))
    for left_name, right_name in pair_names:
        left = splits[left_name]
        right = splits[right_name]
        if left.empty or right.empty:
            continue

        patient_overlap = set(left[group_column]) & set(right[group_column])
        if patient_overlap:
            report.issues.append(
                _issue(
                    "patient_leak_between_splits",
                    "error",
                    f"{name}: {len(patient_overlap)} pacientes aparecen en "
                    f"{left_name} y {right_name} (fuga por {group_column}).",
                    count=len(patient_overlap),
                    splits=(left_name, right_name),
                    samples=sorted(list(patient_overlap))[:5],
                )
            )

        path_overlap = set(left[path_column]) & set(right[path_column])
        if path_overlap:
            report.issues.append(
                _issue(
                    "path_leak_between_splits",
                    "error",
                    f"{name}: {len(path_overlap)} paths compartidos entre "
                    f"{left_name} y {right_name}.",
                    count=len(path_overlap),
                    splits=(left_name, right_name),
                    samples=sorted(list(path_overlap))[:5],
                )
            )

        left_keys = set(zip(left[group_column], left[image_id_column], strict=False))
        right_keys = set(zip(right[group_column], right[image_id_column], strict=False))
        key_overlap = left_keys & right_keys
        if key_overlap:
            report.issues.append(
                _issue(
                    "image_key_leak_between_splits",
                    "error",
                    f"{name}: {len(key_overlap)} imágenes "
                    f"({group_column}, {image_id_column}) compartidas entre "
                    f"{left_name} y {right_name}.",
                    count=len(key_overlap),
                    splits=(left_name, right_name),
                    samples=[{"patient_id": k[0], "image_id": k[1]} for k in list(key_overlap)[:5]],
                )
            )

    combined = pd.concat(
        [
            train.assign(_split="train"),
            val.assign(_split="val"),
            test.assign(_split="test"),
        ],
        ignore_index=True,
    )
    cross_split_paths = combined.groupby(path_column)["_split"].nunique()
    leaked_paths = cross_split_paths[cross_split_paths > 1]
    if not leaked_paths.empty:
        leaked_rows = combined[combined[path_column].isin(leaked_paths.index)]
        report.issues.append(
            _issue(
                "path_in_multiple_splits",
                "error",
                f"{name}: {len(leaked_paths)} paths aparecen en más de un split.",
                count=int(len(leaked_paths)),
                samples=_sample_records(
                    leaked_rows.sort_values(path_column),
                    [path_column, label_column, "_split", group_column],
                ),
            )
        )

    if not any(issue.severity in ("error", "warning") for issue in report.issues):
        total = sum(len(frame) for frame in splits.values())
        report.issues.append(
            _issue(
                "splits_ok",
                "info",
                f"{name}: train/val/test sin fugas detectadas "
                f"(train={len(train)}, val={len(val)}, test={len(test)}, total={total}).",
                count=total,
            )
        )

    return report


def validate_dataset_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    full_df: pd.DataFrame | None = None,
    name: str = "dataset",
    check_files: bool = True,
    raise_on_error: bool = False,
) -> DatasetValidationReport:
    """Valida la tabla completa (opcional) y los tres splits de entrenamiento."""
    report = validate_splits(train, val, test, name=f"{name}/splits")

    for split_name, frame in (("train", train), ("val", val), ("test", test)):
        report = report.merge(
            validate_dataframe(
                frame,
                name=f"{name}/{split_name}",
                check_files=check_files,
            )
        )

    if full_df is not None:
        report = report.merge(
            validate_dataframe(
                full_df,
                name=f"{name}/full",
                check_files=check_files,
            )
        )

    if raise_on_error and not report.ok:
        messages = "\n".join(f"  - [{issue.code}] {issue.message}" for issue in report.errors)
        raise ValueError(f"Validación del dataset falló:\n{messages}")

    return report


def print_validation_report(report: DatasetValidationReport) -> None:
    """Imprime un resumen legible de la validación."""
    icon = { "error": "ERROR", "warning": "WARN ", "info": "OK   " }

    print(f"=== Validación: {report.name} ===")
    if report.ok:
        print("Estado: OK (sin errores)")
    else:
        print(f"Estado: FALLO ({len(report.errors)} error/es)")

    for issue in report.issues:
        prefix = icon[issue.severity]
        suffix = f" (n={issue.count})" if issue.count else ""
        print(f"[{prefix}] {issue.message}{suffix}")
        if issue.details.get("samples"):
            print(f"         ejemplos: {issue.details['samples']}")

    print(
        f"Resumen: {len(report.errors)} errores, "
        f"{len(report.warnings)} advertencias, "
        f"{sum(1 for i in report.issues if i.severity == 'info')} informativos"
    )
