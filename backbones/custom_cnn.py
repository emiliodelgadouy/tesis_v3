from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from .custom_tiny import preprocess_input

DESCRIPTION = (
    "CNN compacta entrenada desde cero (4 bloques conv 3×3 + BN + ReLU + pooling, "
    "entrada 224×224). Pensada para datasets medicos pequenos con regularizacion "
    "fuerte (dropout) y sin pesos preentrenados."
)


def _conv_bn_relu(
    x,
    filters: int,
    kernel_size: int,
    *,
    strides: int = 1,
    name: str,
) -> keras.KerasTensor:
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    return x


def _residual_block(
    x,
    filters: int,
    name: str,
    *,
    strides: int = 1,
    dropout: float = 0.0,
) -> keras.KerasTensor:
    shortcut = x

    x = _conv_bn_relu(x, filters, 3, strides=strides, name=f"{name}_a")
    x = layers.Conv2D(
        filters, 3, padding="same", use_bias=False, name=f"{name}_b_conv"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_b_bn")(x)

    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters,
            1,
            strides=strides,
            padding="same",
            use_bias=False,
            name=f"{name}_proj_conv",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name=f"{name}_drop")(x)
    return x


def _residual_stage(
    x,
    filters: int,
    blocks: int,
    name: str,
    *,
    first_stride: int = 1,
    dropout: float = 0.15,
) -> keras.KerasTensor:
    x = _residual_block(
        x, filters, f"{name}_block0", strides=first_stride, dropout=dropout
    )
    for block_idx in range(1, blocks):
        x = _residual_block(
            x, filters, f"{name}_block{block_idx}", dropout=dropout
        )
    return x


def CustomCnnBackbone(
    weights: str | None = None,
    include_top: bool = False,
    input_shape: tuple[int, int, int] | None = None,
    name: str = "custom_cnn",
    **kwargs,
) -> keras.Model:
    """ResNet-18 desde cero (sin transfer learning).

    Baseline para el dataset completo (~20k imagenes filtradas, ~1.8k positivos):
    stem convolucional, cuatro etapas con bloques residuales (2 por etapa) y
    canales 64-128-256-512 (~11M parametros). Escala adecuada frente a
    transfer learning sin ser tan pesada como ResNet-50+. La cabeza clasificadora
    la agrega ModelBuilder.
    """
    if weights not in (None, "none"):
        raise ValueError(
            f"CustomCnnBackbone no usa pesos preentrenados (recibido weights={weights!r})."
        )

    inputs = keras.Input(shape=input_shape or (224, 224, 3), name="input")
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)

    # Stem ResNet: 224 -> 112 -> 56
    x = _conv_bn_relu(x, 64, 7, strides=2, name="stem")
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="stem_pool")(x)

    # 56 -> 28 -> 14 -> 7
    x = _residual_stage(x, 64, 2, "stage1", dropout=0.05)
    x = _residual_stage(x, 128, 2, "stage2", first_stride=2, dropout=0.10)
    x = _residual_stage(x, 256, 2, "stage3", first_stride=2, dropout=0.15)
    x = _residual_stage(x, 512, 2, "stage4", first_stride=2, dropout=0.20)

    if include_top:
        classes = kwargs.pop("classes", 1000)
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = layers.Dense(classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs, x, name=name)
