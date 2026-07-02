from __future__ import annotations

import os
from pathlib import Path

from tensorflow import keras


def medical_weights_dir() -> Path:
    base = Path(os.environ["MEDICAL_WEIGHTS_DIR"]) if os.environ.get("MEDICAL_WEIGHTS_DIR") else Path.home() / ".keras" / "medical_weights"
    base.mkdir(parents=True, exist_ok=True)
    return base


def transfer_legacy_weights(target_model: keras.Model, weights_path: Path) -> None:
    import tf_keras

    legacy_weights = tf_keras.models.load_model(str(weights_path), compile=False).get_weights()
    target_model.set_weights(legacy_weights)
