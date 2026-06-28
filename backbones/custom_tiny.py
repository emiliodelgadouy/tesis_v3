from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

DESCRIPTION = (
    "CNN minima (2 capas conv + ReLU, entrada 64×64) entrenada desde cero. "
    "Sirve como baseline de overhead del pipeline y no usa transfer learning."
)


def preprocess_input(x):
    return x


def CustomTinyBackbone(
    weights: str | None = None,
    include_top: bool = False,
    input_shape: tuple[int, int, int] | None = None,
    name: str = "custom_tiny",
    **kwargs,
) -> keras.Model:
    """Backbone minimo para medir overhead de entrenamiento/inferencia."""
    del weights

    inputs = keras.Input(shape=input_shape or (64, 64, 3), name="input")
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)
    x = layers.Conv2D(8, 3, strides=2, padding="same", activation="relu", name="conv_1")(x)
    x = layers.Conv2D(16, 3, strides=2, padding="same", activation="relu", name="conv_2")(x)

    if include_top:
        classes = kwargs.pop("classes", 1000)
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = layers.Dense(classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs, x, name=name)
