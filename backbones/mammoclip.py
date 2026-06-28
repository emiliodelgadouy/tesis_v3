from __future__ import annotations

from tensorflow import keras

from .base import Backbone, DEFAULT_WEIGHTS

_DESCRIPTION = (
    "MAMMO-CLIP (MICCAI 2024): modelo vision-lenguaje para mamografia basado en "
    "EfficientNet y BioClinicalBERT, preentrenado con pares mamograma-informe. "
    "No es un backbone Keras entrenable dentro de este pipeline; usalo como baseline "
    "zero-shot o extractor externo con `src.mammo_clip`."
)


class MammoClipBackbone(Backbone):
    key = "mammoclip"
    keras_name = "mammoclip"
    input_size = (224, 224)
    description = _DESCRIPTION
    default_weights = "mammoclip"

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
        del weights, include_top, input_shape, kwargs
        raise NotImplementedError(
            "MAMMO-CLIP es un modelo PyTorch/vision-lenguaje, no un `keras.Model` "
            "compatible con `ModelBuilder`. Para usarlo en esta tesis, instalá el paquete "
            "`mammoclip` y llamá a `predict_mammoclip_zero_shot(...)` desde `src.mammo_clip` "
            "como baseline externo. Si querés usar embeddings en ABMIL/CLAM, precomputalos "
            "fuera del grafo Keras y entrená una cabeza sobre esos vectores."
        )
