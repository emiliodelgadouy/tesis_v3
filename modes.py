"""Modos de entrenamiento disponibles para los experimentos."""

from __future__ import annotations

TRAINING_MODES = (
    "simple",
    "abmil",
    "abmil_patch_hardneg",
    "abmil_patch_hardneg_guided",
    "full",
    "patch",
    "patch_hardneg",
)

MIL_MODES = (
    "abmil",
    "abmil_patch_hardneg",
    "abmil_patch_hardneg_guided",
)

PATCH_PRETRAINED_MIL_MODES = (
    "abmil_patch_hardneg",
    "abmil_patch_hardneg_guided",
)

# Clave sin guiones ni underscores (ver normalize_mode).
_NORMALIZED_TO_MODE = {
    "simple": "simple",
    "abmil": "abmil",
    "abmilpatchhardneg": "abmil_patch_hardneg",
    "abmilpatchhardnegguided": "abmil_patch_hardneg_guided",
    "full": "full",
    "patch": "patch",
    "patchhardneg": "patch_hardneg",
}


def normalize_mode(name: str) -> str:
    key = name.strip().lower().replace("-", "").replace("_", "")
    mode = _NORMALIZED_TO_MODE.get(key)
    if mode is not None:
        return mode
    raise ValueError(
        f"MODE '{name}' no disponible. Opciones: {', '.join(m.upper() for m in TRAINING_MODES)}"
    )


def is_mil_mode(mode: str) -> bool:
    return normalize_mode(mode) in MIL_MODES


def uses_patch_pretraining(mode: str) -> bool:
    return normalize_mode(mode) in PATCH_PRETRAINED_MIL_MODES


def is_patch_mode(mode: str) -> bool:
    return normalize_mode(mode) in ("patch", "patch_hardneg")


def resolve_training_mode(
    *,
    mode: str | None = None,
    mil_mode: bool | None = None,
    architecture_name: str | None = None,
    bag_mode: bool | None = None,
) -> str:
    """Unifica MODE con parametros legacy (mil_mode, architecture_name, bag_mode).

    Si ``mode`` esta seteado, siempre gana; los aliases legacy solo aplican si
    ``mode is None``.
    """
    if mode is not None:
        return normalize_mode(mode)

    if mil_mode is False:
        return "simple"
    if architecture_name is not None:
        return normalize_mode(architecture_name)
    if mil_mode is True:
        return "abmil"
    if bag_mode is True:
        return "abmil"
    if bag_mode is False:
        return "simple"
    return "simple"


def resolve_mode_kwargs(kwargs: dict) -> dict:
    """Extrae aliases legacy de kwargs y devuelve una copia con `mode` unificado."""
    out = dict(kwargs)
    mode = out.pop("mode", None)
    patch_mode = out.pop("patch_mode", None)
    resolved = resolve_training_mode(
        mode=mode,
        mil_mode=out.pop("mil_mode", None),
        architecture_name=out.pop("architecture_name", None),
        bag_mode=out.pop("bag_mode", None),
    )
    if patch_mode is True and resolved == "simple":
        resolved = "patch"
    out["mode"] = resolved
    return out
