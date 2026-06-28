"""Backbones como objetos registrables; el provider solo reexporta el registro."""

from .base import Backbone, DEFAULT_WEIGHTS, KerasApplicationBackbone, keras_application
from .chexnet import CheXNetBackbone
from .custom_cnn import CustomCnnBackbone
from .custom_tiny import CustomTinyBackbone
from .mammoclip import MammoClipBackbone
from .radimagenet import RadImageNetBackbone
from .registry import (
    BACKBONES,
    BackboneConfig,
    available_backbones,
    build_backbone,
    format_all_architecture_descriptions,
    get_backbone,
    get_backbone_config,
    get_backbone_description,
    get_input_size,
    get_preprocess_input,
    normalize_name,
    resolve_backbone,
)

__all__ = [
    "BACKBONES",
    "Backbone",
    "BackboneConfig",
    "DEFAULT_WEIGHTS",
    "KerasApplicationBackbone",
    "CheXNetBackbone",
    "CustomCnnBackbone",
    "CustomTinyBackbone",
    "MammoClipBackbone",
    "RadImageNetBackbone",
    "available_backbones",
    "build_backbone",
    "format_all_architecture_descriptions",
    "get_backbone",
    "get_backbone_config",
    "get_backbone_description",
    "get_input_size",
    "get_preprocess_input",
    "keras_application",
    "normalize_name",
    "resolve_backbone",
]
