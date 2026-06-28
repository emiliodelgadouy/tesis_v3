"""Factory de backbones: delega en el registro OOP de ``src.backbones``."""

from src.backbones.registry import (
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
    resolve_backbone,
)

__all__ = [
    "BACKBONES",
    "BackboneConfig",
    "available_backbones",
    "build_backbone",
    "format_all_architecture_descriptions",
    "get_backbone",
    "get_backbone_config",
    "get_backbone_description",
    "get_input_size",
    "get_preprocess_input",
    "resolve_backbone",
]
