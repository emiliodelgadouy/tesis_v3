from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

from tensorflow import keras


def medical_weights_dir() -> Path:
    custom = os.environ.get("MEDICAL_WEIGHTS_DIR")
    base = Path(custom) if custom else Path.home() / ".keras" / "medical_weights"
    base.mkdir(parents=True, exist_ok=True)
    return base


def import_gdown():
    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "Se requiere 'gdown' para descargar pesos pre-entrenados medicos. "
            "Instalalo con: pip install gdown"
        ) from exc
    return gdown


def load_legacy_h5_model(weights_path: Path):
    """Carga un modelo Keras guardado en formato legacy (.h5 de Keras 2).

    Keras 3 rechaza nombres de capa con '/' (p. ej. 'conv1/conv') al deserializar
    estos .h5 antiguos. ``tf_keras`` (el paquete de compatibilidad Keras 2) los
    carga sin problema; lo instalamos al vuelo si no esta disponible.
    """
    try:
        import tf_keras  # type: ignore
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "tf-keras"],
            check=True,
        )
        import tf_keras  # type: ignore

    return tf_keras.models.load_model(str(weights_path), compile=False)


def transfer_legacy_weights(target_model: keras.Model, weights_path: Path) -> None:
    """Transfiere por posicion los pesos de un .h5 legacy a un modelo Keras 3.

    Construimos la arquitectura limpia en Keras 3 (nombres validos) y copiamos
    los pesos del modelo legacy en el mismo orden topologico. Las arquitecturas
    son identicas (misma familia de ``keras.applications``), por lo que el orden
    y las formas coinciden; validamos ambas cosas para fallar de forma explicita.
    """
    legacy_model = load_legacy_h5_model(weights_path)
    source_weights = legacy_model.get_weights()
    target_weights = target_model.get_weights()

    if len(source_weights) != len(target_weights):
        raise ValueError(
            "No se pueden transferir los pesos legacy: el modelo guardado tiene "
            f"{len(source_weights)} tensores y el modelo destino {len(target_weights)}. "
            "Verifica que la arquitectura coincide."
        )
    for index, (src, dst) in enumerate(zip(source_weights, target_weights)):
        if src.shape != dst.shape:
            raise ValueError(
                f"Forma incompatible al transferir el peso {index}: "
                f"{src.shape} (legacy) vs {dst.shape} (destino)."
            )
    target_model.set_weights(source_weights)


def extract_zip_bundle(zip_path: Path, cache_dir: Path, *, min_bytes: int) -> bool:
    """Extrae un ZIP en cache si existe y supera el tamano minimo. Devuelve True si ok."""
    if not zip_path.is_file():
        return False
    if zip_path.stat().st_size < min_bytes:
        zip_path.unlink(missing_ok=True)
        return False
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(cache_dir)
    except zipfile.BadZipFile:
        zip_path.unlink(missing_ok=True)
        return False
    return True
