from __future__ import annotations

from tensorflow.keras.applications import (
    EfficientNetV2B0,
    EfficientNetV2B1,
    EfficientNetV2B2,
    EfficientNetV2B3,
    EfficientNetV2L,
    EfficientNetV2M,
    EfficientNetV2S,
)
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

from .base import Backbone, keras_application

_DESCRIPTIONS = {
    "efficientnetv2b0": (
        "EfficientNetV2-B0 (Tan & Le, 2021): bloques Fused-MBConv y entrenamiento progresivo; "
        "mejor trade-off velocidad/precision que EfficientNet v1. ImageNet, entrada 224×224."
    ),
    "efficientnetv2b1": "EfficientNetV2-B1. ImageNet, entrada 240×240.",
    "efficientnetv2b2": "EfficientNetV2-B2. ImageNet, entrada 260×260.",
    "efficientnetv2b3": "EfficientNetV2-B3. ImageNet, entrada 300×300.",
    "efficientnetv2s": "EfficientNetV2-S (small). ImageNet, entrada 384×384.",
    "efficientnetv2m": "EfficientNetV2-M (medium). ImageNet, entrada 480×480.",
    "efficientnetv2l": "EfficientNetV2-L (large). ImageNet, entrada 480×480.",
}

_VARIANTS: tuple[tuple[str, type, tuple[int, int]], ...] = (
    ("efficientnetv2b0", EfficientNetV2B0, (224, 224)),
    ("efficientnetv2b1", EfficientNetV2B1, (240, 240)),
    ("efficientnetv2b2", EfficientNetV2B2, (260, 260)),
    ("efficientnetv2b3", EfficientNetV2B3, (300, 300)),
    ("efficientnetv2s", EfficientNetV2S, (384, 384)),
    ("efficientnetv2m", EfficientNetV2M, (480, 480)),
    ("efficientnetv2l", EfficientNetV2L, (480, 480)),
)

BACKBONES: tuple[Backbone, ...] = tuple(
    keras_application(key, model_fn, preprocess_input, input_size, _DESCRIPTIONS[key])
    for key, model_fn, input_size in _VARIANTS
)
