"""Modos de entrenamiento: SIMPLE | ABMIL | FULL | PATCH."""

from __future__ import annotations

TRAINING_MODES = ("simple", "abmil", "abmil_patch_hardneg", "full", "patch", "patch_hardneg")

# Clave sin guiones ni underscores (ver normalize_mode).
_NORMALIZED_TO_MODE = {
    "simple": "simple",
    "abmil": "abmil",
    "abmilpatchhardneg": "abmil_patch_hardneg",
    "full": "full",
    "patch": "patch",
    "patchhardneg": "patch_hardneg",
}

# Entrada fija para MODE=full (todos los backbones usan el mismo H×W).
FULL_INPUT_SIZE: tuple[int, int] = (512, 512)

MODE_DESCRIPTIONS: dict[str, str] = {
    "simple": (
        "Clasificador de imagen completa: backbone + GAP + MLP. Baseline sin MIL."
    ),
    "full": (
        "Igual que SIMPLE (backbone + GAP + MLP, sin MIL) pero con entrada fija "
        f"{FULL_INPUT_SIZE[0]}×{FULL_INPUT_SIZE[1]} para todos los backbones, "
        "en lugar de reescalar al input nativo de cada arquitectura. El backbone es "
        "totalmente convolucional + GAP, asi que acepta esa resolucion sin modificar "
        "capas; mayor uso de memoria y tiempo que SIMPLE."
    ),
    "abmil": (
        "ABMIL (Ilse et al., 2018): atencion gated sobre instancias del bag, "
        "agregacion ponderada y clasificador a nivel imagen. Sin supervision "
        "explicita a nivel parche."
    ),
    "abmil_patch_hardneg": (
        "ABMIL con encoder inicializado desde un PatchModelBuilder patch_hardneg "
        "preentrenado (backbone + proyeccion densa + clasificador transferidos)."
    ),
    "patch": (
        "Clasificador de parches: backbone + GAP + MLP sobre crops (roi / "
        "avoid_roi / uniforme segun etiqueta). Mismo grafo que SIMPLE; el "
        "dataset activa patch_mode. Train con positivos + randneg."
    ),
    "patch_hardneg": (
        "Igual que PATCH (mismo grafo y patch_mode) pero el train mezcla "
        "positivos, hard negatives (misma imagen con hallazgo, etiqueta 0) "
        "y randneg, tipicamente repartiendo el balance neg:pos entre ambos tipos."
    ),
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
    return normalize_mode(mode) in ("abmil", "abmil_patch_hardneg")


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
