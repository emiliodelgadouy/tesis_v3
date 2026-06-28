"""Modos de entrenamiento: SIMPLE | ABMIL | CLAM."""

from __future__ import annotations

TRAINING_MODES = ("simple", "abmil", "clam", "native")

MODE_DESCRIPTIONS: dict[str, str] = {
    "simple": (
        "Clasificador de imagen completa: backbone + GAP + MLP. Baseline sin MIL."
    ),
    "native": (
        "Igual que SIMPLE (backbone + GAP + MLP, sin MIL) pero alimentando la imagen "
        "a su RESOLUCION ORIGINAL en lugar de reescalarla al input nativo del backbone. "
        "Al ser el backbone totalmente convolucional + GAP, acepta cualquier resolucion "
        "sin modificar capas; evita comprimir la imagen a costa de mayor uso de memoria."
    ),
    "abmil": (
        "ABMIL (Ilse et al., 2018): atencion gated sobre instancias del bag, "
        "agregacion ponderada y clasificador a nivel imagen. Sin supervision "
        "explicita a nivel parche."
    ),
    "clam": (
        "CLAM-SB (Lu et al., 2021): atencion gated + clasificador de bag y "
        "clustering a nivel instancia con pseudo-etiquetas en los parches mas y "
        "menos atendidos (top-k / bottom-k). Perdida combinada bag + instancia."
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
    return normalize_mode(mode) in ("abmil", "clam")


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
    resolved = resolve_training_mode(
        mode=mode,
        mil_mode=out.pop("mil_mode", None),
        architecture_name=out.pop("architecture_name", None),
        bag_mode=out.pop("bag_mode", None),
    )
    out["mode"] = resolved
    return out
