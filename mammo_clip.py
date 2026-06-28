from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

DEFAULT_ZERO_SHOT_PROMPTS: dict[str, list[str]] = {
    "mass": ["no mass", "mass"],
    "calcification": ["no suspicious calcification", "suspicious calcification"],
    "malignancy": ["benign mammogram", "malignant mammogram"],
    "density": [
        "fatty breast density",
        "scattered fibroglandular breast density",
        "heterogeneously dense breast",
        "extremely dense breast",
    ],
}


@dataclass
class MammoClipZeroShotConfig:
    """Configuracion para correr MAMMO-CLIP como baseline externo.

    El paquete `mammoclip` se carga de forma opcional para no hacer pesado el entorno
    base de TensorFlow/Keras. La salida conserva identificadores de la tabla original
    y agrega columnas `mammoclip_*` con las predicciones devueltas por el paquete.
    """

    path_column: str = "path"
    id_columns: tuple[str, ...] = ("patient_id", "series_id", "image_id", "laterality", "view", "cls")
    output_csv: str | Path | None = "mammo_clip_zero_shot.csv"
    model_kwargs: dict[str, Any] = field(default_factory=dict)


def _import_mammoclip():
    try:
        import mammoclip  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Falta el paquete opcional `mammoclip`. Instalalo en el entorno donde vas "
            "a correr el baseline, por ejemplo:\n\n"
            "    pip install mammoclip\n\n"
            "MAMMO-CLIP usa PyTorch y descarga/checkpoints externos; por eso no se "
            "incluye como dependencia obligatoria del pipeline Keras."
        ) from exc
    return mammoclip


def load_mammoclip_model(**model_kwargs: Any):
    """Carga el modelo del paquete `mammoclip`.

    Se mantiene en un helper chico para que el notebook pueda construirlo una sola vez
    y reutilizarlo en varias llamadas a `predict_mammoclip_zero_shot`.
    """

    mammoclip = _import_mammoclip()
    model_cls = getattr(mammoclip, "MammoClipModel", None)
    if model_cls is None:
        raise AttributeError(
            "El paquete `mammoclip` instalado no expone `MammoClipModel`. "
            "Actualizalo o revisa la API del paquete."
        )
    return model_cls(**model_kwargs)


def _as_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        arr = value.detach().cpu().numpy()
        return _as_python_scalar(arr)
    return value


def _flatten_prediction(value: Any, prefix: str = "mammoclip") -> dict[str, Any]:
    value = _as_python_scalar(value)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, child in value.items():
            clean_key = str(key).strip().lower().replace(" ", "_").replace("-", "_")
            out.update(_flatten_prediction(child, f"{prefix}_{clean_key}"))
        return out
    if isinstance(value, (list, tuple)):
        return {
            f"{prefix}_{idx}": _as_python_scalar(item)
            for idx, item in enumerate(value)
        }
    return {prefix: value}


def _row_identity(row: pd.Series, config: MammoClipZeroShotConfig) -> dict[str, Any]:
    out = {
        column: row[column]
        for column in config.id_columns
        if column in row.index
    }
    out[config.path_column] = row[config.path_column]
    return out


def predict_mammoclip_zero_shot(
    df: pd.DataFrame,
    prompts: Mapping[str, list[str]] | None = None,
    *,
    model: Any | None = None,
    config: MammoClipZeroShotConfig | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Corre MAMMO-CLIP zero-shot sobre filas con paths de mamografias.

    Parameters
    ----------
    df:
        Tabla con al menos `config.path_column`.
    prompts:
        Diccionario de tareas a clases textuales. Si no se indica, usa prompts
        razonables para masa, calcificaciones, malignidad y densidad.
    model:
        Instancia ya cargada de `mammoclip.MammoClipModel`. Si es None se carga una.
    config:
        Configuracion de columnas y salida.
    limit:
        Limita la cantidad de filas, util para smoke tests antes de correr todo.
    """

    config = config or MammoClipZeroShotConfig()
    prompts = dict(prompts or DEFAULT_ZERO_SHOT_PROMPTS)
    if config.path_column not in df.columns:
        raise KeyError(f"Falta la columna de paths: {config.path_column!r}")
    if not prompts:
        raise ValueError("`prompts` no puede estar vacio.")

    table = df.reset_index(drop=True)
    if limit is not None:
        table = table.iloc[: int(limit)].copy()

    model = model or load_mammoclip_model(**config.model_kwargs)
    records: list[dict[str, Any]] = []
    for idx, row in table.iterrows():
        path = str(row[config.path_column])
        prediction = model.predict(path, prompts)
        record = _row_identity(row, config)
        record["mammoclip_row_index"] = int(idx)
        record.update(_flatten_prediction(prediction))
        records.append(record)

    result = pd.DataFrame(records)
    if config.output_csv is not None:
        output_csv = Path(config.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
    return result
