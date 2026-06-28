from __future__ import annotations

from tensorflow import keras

from ._types import InputSize, PreprocessFunction
from .base import Backbone, DEFAULT_WEIGHTS
from .chexnet import CheXNetBackbone
from .custom_cnn import CustomCnnBackbone
from .custom_tiny import CustomTinyBackbone
from .efficientnet import BACKBONES as EFFICIENTNET_BACKBONES
from .efficientnet_v2 import BACKBONES as EFFICIENTNET_V2_BACKBONES
from .mammoclip import MammoClipBackbone
from .radimagenet import BACKBONES as RADIMAGENET_BACKBONES
from .vgg import BACKBONES as VGG_BACKBONES

_BACKBONE_INSTANCES: tuple[Backbone, ...] = (
    CustomTinyBackbone(),
    CustomCnnBackbone(),
    *VGG_BACKBONES,
    *EFFICIENTNET_BACKBONES,
    *EFFICIENTNET_V2_BACKBONES,
    *RADIMAGENET_BACKBONES,
    CheXNetBackbone(),
    MammoClipBackbone(),
)

BACKBONES: dict[str, Backbone] = {b.key: b for b in _BACKBONE_INSTANCES}

# Alias legacy del dataclass anterior.
BackboneConfig = Backbone


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "")


def available_backbones() -> tuple[str, ...]:
    return tuple(BACKBONES)


def get_backbone(name: str) -> Backbone:
    key = normalize_name(name)
    if key not in BACKBONES:
        available = ", ".join(available_backbones())
        raise ValueError(f"Backbone '{name}' is not available. Options: {available}")
    return BACKBONES[key]


get_backbone_config = get_backbone


def get_preprocess_input(name: str) -> PreprocessFunction:
    return get_backbone(name).preprocess_input


def get_input_size(name: str) -> InputSize:
    return get_backbone(name).input_size


def get_backbone_description(name: str) -> str:
    return get_backbone(name).format_description()


def format_all_architecture_descriptions() -> str:
    lines = [
        "Catalogo de backbones disponibles",
        "=" * 40,
        "",
    ]
    for key in available_backbones():
        lines.append(get_backbone_description(key))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_backbone(
    name: str,
    weights=DEFAULT_WEIGHTS,
    include_top: bool = False,
    input_shape: tuple[int, int, int] | None = None,
    **kwargs,
) -> keras.Model:
    return get_backbone(name).build(
        weights=weights,
        include_top=include_top,
        input_shape=input_shape,
        **kwargs,
    )


def resolve_backbone(
    backbone_name: str, input_size: InputSize | None = None
) -> tuple[keras.Model, PreprocessFunction, InputSize]:
    return get_backbone(backbone_name).resolve(input_size)
