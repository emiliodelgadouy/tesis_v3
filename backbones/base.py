from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from tensorflow import keras

from ._types import InputSize, ModelFactory, PreprocessFunction

DEFAULT_WEIGHTS = object()


class Backbone(ABC):
    """Contrato de un backbone registrable: metadatos, preproceso y construccion."""

    key: str
    keras_name: str
    input_size: InputSize
    description: str
    default_weights: str | None

    @abstractmethod
    def preprocess_input(self, x):
        """Normaliza la entrada como en el preentrenamiento del backbone."""

    @abstractmethod
    def build(
        self,
        *,
        weights=DEFAULT_WEIGHTS,
        include_top: bool = False,
        input_shape: tuple[int, int, int] | None = None,
        **kwargs,
    ) -> keras.Model:
        """Instancia el ``keras.Model`` convolucional (sin cabeza de clasificacion)."""

    @property
    def name(self) -> str:
        """Alias legacy de ``keras_name`` (``BackboneConfig.name``)."""
        return self.keras_name

    @property
    def model_fn(self) -> ModelFactory:
        """Callable compatible con la API antigua de ``keras.applications``."""

        def factory(
            weights=DEFAULT_WEIGHTS,
            include_top: bool = False,
            input_shape: tuple[int, int, int] | None = None,
            name: str | None = None,
            **kwargs,
        ) -> keras.Model:
            del name  # Keras Application usa su propio nombre interno; no forzamos aqui.
            return self.build(
                weights=weights,
                include_top=include_top,
                input_shape=input_shape,
                **kwargs,
            )

        return factory

    def format_description(self) -> str:
        h, w = self.input_size
        weights = self.default_weights or "none (desde cero)"
        return (
            f"Arquitectura: {self.key}\n"
            f"Entrada: {h}×{w}×3\n"
            f"Pesos por defecto: {weights}\n\n"
            f"{self.description}"
        )

    def resolve(
        self, input_size: InputSize | None = None
    ) -> tuple[keras.Model, PreprocessFunction, InputSize]:
        if input_size is None:
            input_size = self.input_size
        height, width = input_size
        model = self.build(input_shape=(height, width, 3))
        return model, self.preprocess_input, input_size

    def _resolve_weights(self, weights) -> str | None:
        if weights is DEFAULT_WEIGHTS:
            return self.default_weights
        return weights

    def _default_input_shape(
        self, input_shape: tuple[int, int, int] | None
    ) -> tuple[int, int, int]:
        height, width = self.input_size
        return input_shape or (height, width, 3)


class KerasApplicationBackbone(Backbone):
    """Wrapper uniforme sobre ``tensorflow.keras.applications``."""

    def __init__(
        self,
        key: str,
        keras_name: str,
        application: ModelFactory,
        preprocess_fn: PreprocessFunction,
        input_size: InputSize,
        description: str,
        *,
        default_weights: str | None = "imagenet",
    ) -> None:
        self.key = key
        self.keras_name = keras_name
        self._application = application
        self._preprocess_fn = preprocess_fn
        self.input_size = input_size
        self.description = description
        self.default_weights = default_weights

    def preprocess_input(self, x):
        return self._preprocess_fn(x)

    def build(
        self,
        *,
        weights=DEFAULT_WEIGHTS,
        include_top: bool = False,
        input_shape: tuple[int, int, int] | None = None,
        **kwargs,
    ) -> keras.Model:
        return self._application(
            weights=self._resolve_weights(weights),
            include_top=include_top,
            input_shape=self._default_input_shape(input_shape),
            **kwargs,
        )


def keras_application(
    key: str,
    model_fn: ModelFactory,
    preprocess_fn: PreprocessFunction,
    input_size: InputSize,
    description: str,
    *,
    keras_name: str | None = None,
    default_weights: str | None = "imagenet",
) -> KerasApplicationBackbone:
    """Fabrica declarativa para backbones de ``keras.applications``."""
    return KerasApplicationBackbone(
        key=key,
        keras_name=keras_name or key,
        application=model_fn,
        preprocess_fn=preprocess_fn,
        input_size=input_size,
        description=description,
        default_weights=default_weights,
    )
