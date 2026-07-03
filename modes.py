"""Modos de entrenamiento: SIMPLE | ABMIL | HIGHRES | PATCH."""

from __future__ import annotations

TRAINING_MODES = ("simple", "abmil", "highres", "patch")

# Entrada fija para MODE=highres (todos los backbones usan el mismo H×W).
HIGHRES_INPUT_SIZE: tuple[int, int] = (512, 512)

MODE_DESCRIPTIONS: dict[str, str] = {
    "simple": (
        "Clasificador de imagen completa: backbone + GAP + MLP. Baseline sin MIL."
    ),
    "highres": (
        "Igual que SIMPLE (backbone + GAP + MLP, sin MIL) pero con entrada fija "
        f"{HIGHRES_INPUT_SIZE[0]}×{HIGHRES_INPUT_SIZE[1]} para todos los backbones, "
        "en lugar de reescalar al input nativo de cada arquitectura. El backbone es "
        "totalmente convolucional + GAP, asi que acepta esa resolucion sin modificar "
        "capas; mayor uso de memoria y tiempo que SIMPLE."
    ),
    "abmil": (
        "ABMIL (Ilse et al., 2018): atencion gated sobre instancias del bag, "
        "agregacion ponderada y clasificador a nivel imagen. Sin supervision "
        "explicita a nivel parche."
    ),
    "patch": (
        "Clasificador de parches: backbone + GAP + MLP sobre crops (roi / "
        "avoid_roi / uniforme segun etiqueta). Mismo grafo que SIMPLE; el "
        "dataset activa patch_mode."
    ),
}


def normalize_mode(name: str) -> str:
    key = name.strip().lower().replace("-", "").replace("_", "")
    if key in MODE_DESCRIPTIONS:
        return key
    raise ValueError(
        f"MODE '{name}' no disponible. Opciones: {', '.join(m.upper() for m in TRAINING_MODES)}"
    )


def is_mil_mode(mode: str) -> bool:
    return normalize_mode(mode) == "abmil"


def is_patch_mode(mode: str) -> bool:
    return normalize_mode(mode) == "patch"


def resolve_training_mode(
    *,
    mode: str | None = None,
    mil_mode: bool | None = None,
    architecture_name: str | None = None,
    bag_mode: bool | None = None,
) -> str:
    """Unifica MODE con parametros legacy (mil_mode, architecture_name, bag_mode).

    Si ``mode`` es distinto de ``simple``, tiene prioridad sobre los aliases legacy.
    """
    has_legacy = (
        mil_mode is not None
        or architecture_name is not None
        or bag_mode is not None
    )
    if mode is not None:
        normalized = normalize_mode(mode)
        if not has_legacy or normalized != "simple":
            return normalized

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


def get_mode_description(name: str) -> str:
    return MODE_DESCRIPTIONS[normalize_mode(name)]


def format_all_mode_descriptions() -> str:
    lines = ["Catalogo de modos de entrenamiento", "=" * 40, ""]
    for key in TRAINING_MODES:
        lines.append(f"{key.upper()}\n{MODE_DESCRIPTIONS[key]}\n")
    return "\n".join(lines).rstrip() + "\n"


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
