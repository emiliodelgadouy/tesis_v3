from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input

from .base import Backbone, DEFAULT_WEIGHTS
from .medical_weights import import_gdown, medical_weights_dir

CHEXNET_FILE_ID = "19BllaOvs2x5PLV_vlWMy4i8LapLb2j6b"
CHEXNET_FILE_NAME = "CheXNet_weights.h5"
CHEXNET_NUM_CLASSES = 14

_DESCRIPTION = (
    "DenseNet-121 preentrenada con CheXNet (Rajpurkar et al.) sobre NIH ChestX-ray14 "
    "(14 patologias toracicas, cabeza multietiqueta descartada). Especializada en "
    "radiografia de torax; preprocesamiento DenseNet. Entrada 224×224."
)


def _ensure_chexnet_weights():
    cache_dir = medical_weights_dir() / "chexnet"
    cache_dir.mkdir(parents=True, exist_ok=True)
    weights_path = cache_dir / CHEXNET_FILE_NAME
    if weights_path.is_file():
        return weights_path

    gdown = import_gdown()
    gdown.download(
        id=CHEXNET_FILE_ID,
        output=str(weights_path),
        quiet=False,
    )
    if not weights_path.is_file():
        raise FileNotFoundError(f"No se pudo descargar CheXNet a {weights_path}")
    return weights_path


class CheXNetBackbone(Backbone):
    key = "chexnet"
    keras_name = "chexnet_densenet121"
    input_size = (224, 224)
    description = _DESCRIPTION
    default_weights = "chexnet"

    def preprocess_input(self, x):
        return densenet_preprocess_input(x)

    def build(
        self,
        *,
        weights=DEFAULT_WEIGHTS,
        include_top: bool = False,
        input_shape: tuple[int, int, int] | None = None,
        **kwargs,
    ) -> keras.Model:
        weights = self._resolve_weights(weights)
        input_shape = self._default_input_shape(input_shape)

        if include_top:
            raise ValueError(
                "CheXNet solo soporta include_top=False (la cabeza original es de 14 clases)."
            )

        backbone = DenseNet121(
            weights=None, include_top=False, input_shape=input_shape, **kwargs
        )

        if weights == "chexnet":
            weights_path = _ensure_chexnet_weights()
            gap = layers.GlobalAveragePooling2D(name="avg_pool")(backbone.output)
            predictions = layers.Dense(
                CHEXNET_NUM_CLASSES, activation="sigmoid", name="predictions"
            )(gap)
            full_model = keras.Model(
                inputs=backbone.input, outputs=predictions, name="chexnet_full"
            )
            full_model.load_weights(str(weights_path))
        elif weights == "imagenet":
            imagenet_backbone = DenseNet121(
                weights="imagenet", include_top=False, input_shape=input_shape, **kwargs
            )
            backbone.set_weights(imagenet_backbone.get_weights())
        elif weights is None:
            pass
        else:
            backbone.load_weights(str(weights))

        backbone._name = self.keras_name
        return backbone
