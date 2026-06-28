from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

from .medical_weights import import_gdown, medical_weights_dir

# CheXNet (DenseNet121 + Dense(14) sigmoid) sobre NIH ChestX-ray14.
# https://github.com/brucechou1983/CheXNet-Keras
CHEXNET_FILE_ID = "19BllaOvs2x5PLV_vlWMy4i8LapLb2j6b"
CHEXNET_FILE_NAME = "CheXNet_weights.h5"
CHEXNET_NUM_CLASSES = 14

DESCRIPTION = (
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


def CheXNetDenseNet121(
    weights="chexnet",
    include_top: bool = False,
    input_shape: tuple[int, int, int] | None = None,
    name: str = "chexnet_densenet121",
    **kwargs,
) -> keras.Model:
    if include_top:
        raise ValueError(
            "CheXNet solo soporta include_top=False (la cabeza original es de 14 clases)."
        )

    backbone = DenseNet121(
        weights=None, include_top=False, input_shape=input_shape, **kwargs
    )

    if weights == "chexnet":
        # El .h5 publico contiene DenseNet121 + GlobalAveragePooling2D + Dense(14) sigmoid.
        # Reconstruimos esa arquitectura, cargamos pesos, y nos quedamos solo con el backbone.
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

    backbone._name = name
    return backbone
