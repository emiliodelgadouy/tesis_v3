from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from tensorflow import keras
from tensorflow.keras.applications import (
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7,
    EfficientNetV2B0,
    EfficientNetV2B1,
    EfficientNetV2B2,
    EfficientNetV2B3,
    EfficientNetV2L,
    EfficientNetV2M,
    EfficientNetV2S,
    VGG16,
    VGG19,
)
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_v2_preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input


ModelFactory = Callable[..., keras.Model]
PreprocessFunction = Callable
InputSize = tuple[int, int]


def custom_tiny_preprocess_input(x):
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
    x = keras.layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)
    x = keras.layers.Conv2D(8, 3, strides=2, padding="same", activation="relu", name="conv_1")(x)
    x = keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu", name="conv_2")(x)

    if include_top:
        classes = kwargs.pop("classes", 1000)
        x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = keras.layers.Dense(classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs, x, name=name)


@dataclass(frozen=True)
class BackboneConfig:
    name: str
    model_fn: ModelFactory
    preprocess_input: PreprocessFunction
    input_size: InputSize


BACKBONES: dict[str, BackboneConfig] = {
    "customtiny": BackboneConfig("custom_tiny", CustomTinyBackbone, custom_tiny_preprocess_input, (64, 64)),
    "vgg16": BackboneConfig("vgg16", VGG16, vgg16_preprocess_input, (224, 224)),
    "vgg19": BackboneConfig("vgg19", VGG19, vgg19_preprocess_input, (224, 224)),
    "efficientnetb0": BackboneConfig("efficientnetb0", EfficientNetB0, efficientnet_preprocess_input, (224, 224)),
    "efficientnetb1": BackboneConfig("efficientnetb1", EfficientNetB1, efficientnet_preprocess_input, (240, 240)),
    "efficientnetb2": BackboneConfig("efficientnetb2", EfficientNetB2, efficientnet_preprocess_input, (260, 260)),
    "efficientnetb3": BackboneConfig("efficientnetb3", EfficientNetB3, efficientnet_preprocess_input, (300, 300)),
    "efficientnetb4": BackboneConfig("efficientnetb4", EfficientNetB4, efficientnet_preprocess_input, (380, 380)),
    "efficientnetb5": BackboneConfig("efficientnetb5", EfficientNetB5, efficientnet_preprocess_input, (456, 456)),
    "efficientnetb6": BackboneConfig("efficientnetb6", EfficientNetB6, efficientnet_preprocess_input, (528, 528)),
    "efficientnetb7": BackboneConfig("efficientnetb7", EfficientNetB7, efficientnet_preprocess_input, (600, 600)),
    "efficientnetv2b0": BackboneConfig("efficientnetv2b0", EfficientNetV2B0, efficientnet_v2_preprocess_input, (224, 224)),
    "efficientnetv2b1": BackboneConfig("efficientnetv2b1", EfficientNetV2B1, efficientnet_v2_preprocess_input, (240, 240)),
    "efficientnetv2b2": BackboneConfig("efficientnetv2b2", EfficientNetV2B2, efficientnet_v2_preprocess_input, (260, 260)),
    "efficientnetv2b3": BackboneConfig("efficientnetv2b3", EfficientNetV2B3, efficientnet_v2_preprocess_input, (300, 300)),
    "efficientnetv2s": BackboneConfig("efficientnetv2s", EfficientNetV2S, efficientnet_v2_preprocess_input, (384, 384)),
    "efficientnetv2m": BackboneConfig("efficientnetv2m", EfficientNetV2M, efficientnet_v2_preprocess_input, (480, 480)),
    "efficientnetv2l": BackboneConfig("efficientnetv2l", EfficientNetV2L, efficientnet_v2_preprocess_input, (480, 480)),
}


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
    weights: str | None = "imagenet",
    include_top: bool = False,
    input_shape: tuple[int, int, int] | None = None,
    **kwargs,
) -> keras.Model:
    config = get_backbone_config(name)
    height, width = config.input_size
    return config.model_fn(
        weights=weights,
        include_top=include_top,
        input_shape=input_shape or (height, width, 3),
        name=config.name,
        **kwargs,
    )
def resolve_backbone(backbone_name):
    backbone = build_backbone(backbone_name)
    preprocess_input = get_preprocess_input(backbone_name)
    input_size = get_input_size(backbone_name)
    return backbone, preprocess_input, input_size