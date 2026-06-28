from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from .base import Backbone, DEFAULT_WEIGHTS

_DESCRIPTION = (
    "CNN minima (2 capas conv + ReLU, entrada 64×64) entrenada desde cero. "
    "Sirve como baseline de overhead del pipeline y no usa transfer learning."
)


class CustomTinyBackbone(Backbone):
    key = "customtiny"
    keras_name = "custom_tiny"
    input_size = (64, 64)
    description = _DESCRIPTION
    default_weights = None

    def preprocess_input(self, x):
        return x

    def build(
        self,
        *,
        weights=DEFAULT_WEIGHTS,
        include_top: bool = False,
        input_shape: tuple[int, int, int] | None = None,
        **kwargs,
    ) -> keras.Model:
        del weights

        inputs = keras.Input(
            shape=self._default_input_shape(input_shape), name="input"
        )
        x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)
        x = layers.Conv2D(
            8, 3, strides=2, padding="same", activation="relu", name="conv_1"
        )(x)
        x = layers.Conv2D(
            16, 3, strides=2, padding="same", activation="relu", name="conv_2"
        )(x)

        if include_top:
            classes = kwargs.pop("classes", 1000)
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            x = layers.Dense(classes, activation="softmax", name="predictions")(x)

        return keras.Model(inputs, x, name=self.keras_name)
