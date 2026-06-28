from __future__ import annotations

from dataclasses import dataclass

from tensorflow import keras

from src.backbones import chexnet, custom_cnn, custom_tiny, efficientnet, efficientnet_v2, radimagenet, vgg
from src.backbones._types import InputSize, ModelFactory, PreprocessFunction
from src.backbones.chexnet import CheXNetDenseNet121
from src.backbones.custom_cnn import CustomCnnBackbone
from src.backbones.custom_tiny import CustomTinyBackbone
from src.backbones.efficientnet import (
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7,
)
from src.backbones.efficientnet_v2 import (
    EfficientNetV2B0,
    EfficientNetV2B1,
    EfficientNetV2B2,
    EfficientNetV2B3,
    EfficientNetV2L,
    EfficientNetV2M,
    EfficientNetV2S,
)
from src.backbones.radimagenet import (
    RadImageNetDenseNet121,
    RadImageNetInceptionResNetV2,
    RadImageNetInceptionV3,
    RadImageNetResNet50,
)


ARCHITECTURE_DESCRIPTIONS: dict[str, str] = {
    "customtiny": custom_tiny.DESCRIPTION,
    "customcnn": custom_cnn.DESCRIPTION,
    **vgg.DESCRIPTIONS,
    **efficientnet.DESCRIPTIONS,
    **efficientnet_v2.DESCRIPTIONS,
    **radimagenet.DESCRIPTIONS,
    "chexnet": chexnet.DESCRIPTION,
}


def get_backbone_description(name: str) -> str:
    """Texto descriptivo de una arquitectura registrada."""
    key = _normalize_name(name)
    config = get_backbone_config(name)
    body = ARCHITECTURE_DESCRIPTIONS.get(
        key,
        f"Backbone '{name}' registrado sin descripcion detallada.",
    )
    h, w = config.input_size
    weights = config.default_weights or "none (desde cero)"
    return (
        f"Arquitectura: {key}\n"
        f"Entrada: {h}×{w}×3\n"
        f"Pesos por defecto: {weights}\n\n"
        f"{body}"
    )


def format_all_architecture_descriptions() -> str:
    """Catalogo de todas las arquitecturas disponibles (para informes / Comet)."""
    lines = [
        "Catalogo de backbones disponibles",
        "=" * 40,
        "",
    ]
    for key in available_backbones():
        lines.append(get_backbone_description(key))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


@dataclass
class BackboneConfig:
    name: str
    model_fn: ModelFactory
    preprocess_input: PreprocessFunction
    input_size: InputSize
    default_weights: str | None = "imagenet"


BACKBONES: dict[str, BackboneConfig] = {
    "customtiny": BackboneConfig(
        "custom_tiny", CustomTinyBackbone, custom_tiny.preprocess_input, (64, 64),
        default_weights=None,
    ),
    "customcnn": BackboneConfig(
        "custom_cnn",
        CustomCnnBackbone,
        custom_tiny.preprocess_input,
        (224, 224),
        default_weights=None,
    ),
    "vgg16": BackboneConfig("vgg16", vgg.VGG16, vgg.vgg16_preprocess_input, (224, 224)),
    "vgg19": BackboneConfig("vgg19", vgg.VGG19, vgg.vgg19_preprocess_input, (224, 224)),
    "efficientnetb0": BackboneConfig("efficientnetb0", EfficientNetB0, efficientnet.preprocess_input, (224, 224)),
    "efficientnetb1": BackboneConfig("efficientnetb1", EfficientNetB1, efficientnet.preprocess_input, (240, 240)),
    "efficientnetb2": BackboneConfig("efficientnetb2", EfficientNetB2, efficientnet.preprocess_input, (260, 260)),
    "efficientnetb3": BackboneConfig("efficientnetb3", EfficientNetB3, efficientnet.preprocess_input, (300, 300)),
    "efficientnetb4": BackboneConfig("efficientnetb4", EfficientNetB4, efficientnet.preprocess_input, (380, 380)),
    "efficientnetb5": BackboneConfig("efficientnetb5", EfficientNetB5, efficientnet.preprocess_input, (456, 456)),
    "efficientnetb6": BackboneConfig("efficientnetb6", EfficientNetB6, efficientnet.preprocess_input, (528, 528)),
    "efficientnetb7": BackboneConfig("efficientnetb7", EfficientNetB7, efficientnet.preprocess_input, (600, 600)),
    "efficientnetv2b0": BackboneConfig("efficientnetv2b0", EfficientNetV2B0, efficientnet_v2.preprocess_input, (224, 224)),
    "efficientnetv2b1": BackboneConfig("efficientnetv2b1", EfficientNetV2B1, efficientnet_v2.preprocess_input, (240, 240)),
    "efficientnetv2b2": BackboneConfig("efficientnetv2b2", EfficientNetV2B2, efficientnet_v2.preprocess_input, (260, 260)),
    "efficientnetv2b3": BackboneConfig("efficientnetv2b3", EfficientNetV2B3, efficientnet_v2.preprocess_input, (300, 300)),
    "efficientnetv2s": BackboneConfig("efficientnetv2s", EfficientNetV2S, efficientnet_v2.preprocess_input, (384, 384)),
    "efficientnetv2m": BackboneConfig("efficientnetv2m", EfficientNetV2M, efficientnet_v2.preprocess_input, (480, 480)),
    "efficientnetv2l": BackboneConfig("efficientnetv2l", EfficientNetV2L, efficientnet_v2.preprocess_input, (480, 480)),
    "radimagenetdensenet121": BackboneConfig(
        "radimagenet_densenet121",
        RadImageNetDenseNet121,
        radimagenet.preprocess_input,
        (224, 224),
        default_weights="radimagenet",
    ),
    "radimagenetresnet50": BackboneConfig(
        "radimagenet_resnet50",
        RadImageNetResNet50,
        radimagenet.preprocess_input,
        (224, 224),
        default_weights="radimagenet",
    ),
    "radimagenetinceptionv3": BackboneConfig(
        "radimagenet_inceptionv3",
        RadImageNetInceptionV3,
        radimagenet.preprocess_input,
        # Los pesos oficiales se entrenaron con input 224x224 (no 299x299).
        (224, 224),
        default_weights="radimagenet",
    ),
    "radimagenetinceptionresnetv2": BackboneConfig(
        "radimagenet_inceptionresnetv2",
        RadImageNetInceptionResNetV2,
        radimagenet.preprocess_input,
        # Los pesos oficiales se entrenaron con input 224x224 (no 299x299).
        (224, 224),
        default_weights="radimagenet",
    ),
    "chexnet": BackboneConfig(
        "chexnet_densenet121",
        CheXNetDenseNet121,
        chexnet.preprocess_input,
        (224, 224),
        default_weights="chexnet",
    ),
}


_DEFAULT_WEIGHTS = object()


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "")


def available_backbones() -> tuple[str, ...]:
    return tuple(BACKBONES)


def get_backbone_config(name: str) -> BackboneConfig:
    key = _normalize_name(name)
    if key not in BACKBONES:
        available = ", ".join(available_backbones())
        raise ValueError(f"Backbone '{name}' is not available. Options: {available}")
    return BACKBONES[key]


def get_preprocess_input(name: str) -> PreprocessFunction:
    return get_backbone_config(name).preprocess_input


def get_input_size(name: str) -> InputSize:
    return get_backbone_config(name).input_size


def build_backbone(
    name: str,
    weights=_DEFAULT_WEIGHTS,
    include_top: bool = False,
    input_shape: tuple[int, int, int] | None = None,
    **kwargs,
) -> keras.Model:
    config = get_backbone_config(name)
    if weights is _DEFAULT_WEIGHTS:
        weights = config.default_weights
    height, width = config.input_size
    # NB: no forzamos `name=config.name`. Las aplicaciones EfficientNetV2 de Keras
    # reutilizan el argumento `name` como clave de DEFAULT_BLOCKS_ARGS (espera
    # "efficientnetv2-b0"), por lo que un nombre normalizado dispara un KeyError.
    # Los backbones propios ya usan por defecto exactamente config.name.
    return config.model_fn(
        weights=weights,
        include_top=include_top,
        input_shape=input_shape or (height, width, 3),
        **kwargs,
    )


def resolve_backbone(backbone_name, input_size: InputSize | None = None):
    if input_size is None:
        input_size = get_input_size(backbone_name)
    height, width = input_size
    backbone = build_backbone(backbone_name, input_shape=(height, width, 3))
    preprocess_input = get_preprocess_input(backbone_name)
    return backbone, preprocess_input, input_size
