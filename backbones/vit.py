from __future__ import annotations

import warnings
from typing import ClassVar

import numpy as np
import scipy.ndimage
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.imagenet_utils import preprocess_input as imagenet_preprocess

from .base import Backbone, DEFAULT_WEIGHTS, InputSize

BASE_URL = "https://github.com/faustomorales/vit-keras/releases/download/dl"
WEIGHTS_CLASSES = {"imagenet21k": 21_843, "imagenet21k+imagenet2012": 1_000}

CONFIG_B = {
    "dropout": 0.1,
    "mlp_dim": 3072,
    "num_heads": 12,
    "num_layers": 12,
    "hidden_size": 768,
}

CONFIG_L = {
    "dropout": 0.1,
    "mlp_dim": 4096,
    "num_heads": 16,
    "num_layers": 24,
    "hidden_size": 1024,
}


def preprocess_input(x):
    return imagenet_preprocess(x, data_format=None, mode="tf")


class ClassToken(layers.Layer):
    def build(self, input_shape):
        self.hidden_size = input_shape[-1]
        self.cls = self.add_weight(
            shape=(1, 1, self.hidden_size),
            initializer="zeros",
            trainable=True,
            name="cls",
            dtype="float32",
        )

    def call(self, inputs):
        batch_size = keras.ops.shape(inputs)[0]
        cls_broadcasted = keras.ops.broadcast_to(self.cls, [batch_size, 1, self.hidden_size])
        return keras.ops.concatenate(
            [keras.ops.cast(cls_broadcasted, dtype=inputs.dtype), inputs],
            axis=1,
        )


class AddPositionEmbs(layers.Layer):
    def build(self, input_shape):
        self.pe = self.add_weight(
            shape=(1, input_shape[1], input_shape[2]),
            initializer="random_normal",
            trainable=True,
            name="pos_embedding",
            dtype="float32",
        )

    def call(self, inputs):
        return inputs + keras.ops.cast(self.pe, dtype=inputs.dtype)


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        if hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} debe ser divisible por num_heads={self.num_heads}"
            )
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // self.num_heads
        self.query_dense = layers.Dense(hidden_size, name="query")
        self.key_dense = layers.Dense(hidden_size, name="key")
        self.value_dense = layers.Dense(hidden_size, name="value")
        self.combine_heads = layers.Dense(hidden_size, name="out")

    def attention(self, query, key, value):
        score = keras.ops.matmul(query, keras.ops.transpose(key, axes=[0, 1, 3, 2]))
        dim_key = keras.ops.cast(keras.ops.shape(key)[-1], score.dtype)
        scaled_score = score / keras.ops.sqrt(dim_key)
        weights = keras.ops.softmax(scaled_score, axis=-1)
        output = keras.ops.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = keras.ops.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return keras.ops.transpose(x, axes=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = keras.ops.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = keras.ops.transpose(attention, axes=[0, 2, 1, 3])
        concat_attention = keras.ops.reshape(attention, (batch_size, -1, self.hidden_size))
        output = self.combine_heads(concat_attention)
        return output, weights


class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, mlp_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

    def build(self, input_shape):
        self.att = MultiHeadSelfAttention(
            num_heads=self.num_heads,
            name="MultiHeadDotProductAttention_1",
        )
        self.mlpblock = keras.Sequential(
            [
                layers.Dense(self.mlp_dim, activation="linear", name=f"{self.name}_Dense_0"),
                layers.Lambda(lambda x: keras.activations.gelu(x, approximate=False)),
                layers.Dropout(self.dropout),
                layers.Dense(input_shape[-1], name=f"{self.name}_Dense_1"),
                layers.Dropout(self.dropout),
            ],
            name="MlpBlock_3",
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, name="LayerNorm_0")
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, name="LayerNorm_2")
        self.dropout_layer = layers.Dropout(self.dropout)

    def call(self, inputs, training=False):
        x = self.layernorm1(inputs)
        x, weights = self.att(x)
        x = self.dropout_layer(x, training=training)
        x = x + inputs
        y = self.layernorm2(x)
        y = self.mlpblock(y)
        return x + y, weights


class PatchTokensToSpatial(layers.Layer):
    """Convierte tokens de parche (sin CLS) en un mapa espacial para GAP del pipeline."""

    def __init__(self, grid_shape: tuple[int, int], **kwargs):
        super().__init__(**kwargs)
        self.grid_shape = grid_shape

    def call(self, tokens):
        patch_tokens = tokens[:, 1:, :]
        batch_size = keras.ops.shape(patch_tokens)[0]
        grid_h, grid_w = self.grid_shape
        return keras.ops.reshape(
            patch_tokens,
            (batch_size, grid_h, grid_w, keras.ops.shape(patch_tokens)[-1]),
        )


def _apply_embedding_weights(target_layer, source_weights, num_x_patches, num_y_patches):
    expected_shape = target_layer.weights[0].shape
    if expected_shape != source_weights.shape:
        token, grid = source_weights[0, :1], source_weights[0, 1:]
        sin = int(np.sqrt(grid.shape[0]))
        warnings.warn(
            f"Reescalando posiciones de ViT de {sin}x{sin} a {num_y_patches}x{num_x_patches}",
            UserWarning,
            stacklevel=2,
        )
        zoom = (num_y_patches / sin, num_x_patches / sin, 1)
        grid = scipy.ndimage.zoom(grid.reshape(sin, sin, -1), zoom, order=1).reshape(
            num_x_patches * num_y_patches,
            -1,
        )
        source_weights = np.concatenate([token, grid], axis=0)[np.newaxis]
    target_layer.set_weights([source_weights])


def _load_vit_weights(model: keras.Model, params_path: str, num_x_patches: int, num_y_patches: int) -> None:
    params_dict = np.load(params_path, allow_pickle=False)
    source_keys = list(params_dict.keys())
    source_keys_used: list[str] = []
    n_transformers = len(
        {
            "/".join(key.split("/")[:2])
            for key in source_keys
            if key.startswith("Transformer/encoderblock_")
        }
    )
    n_transformers_out = sum(layer.name.startswith("Transformer_encoderblock_") for layer in model.layers)
    if n_transformers != n_transformers_out:
        raise ValueError(
            f"ViT: {n_transformers_out} bloques en el modelo vs {n_transformers} en los pesos"
        )

    matches = []
    for tidx in range(n_transformers):
        encoder = model.get_layer(f"Transformer_encoderblock_{tidx}")
        source_prefix = f"Transformer/encoderblock_{tidx}"
        matches.extend(
            [
                {
                    "layer": layer,
                    "keys": [f"{source_prefix}/{norm}/{name}" for name in ["scale", "bias"]],
                }
                for norm, layer in [
                    ("LayerNorm_0", encoder.layernorm1),
                    ("LayerNorm_2", encoder.layernorm2),
                ]
            ]
            + [
                {
                    "layer": encoder.mlpblock.get_layer(f"{source_prefix.replace('/', '_')}_Dense_{mlpdense}"),
                    "keys": [
                        f"{source_prefix}/MlpBlock_3/Dense_{mlpdense}/{name}"
                        for name in ["kernel", "bias"]
                    ],
                }
                for mlpdense in [0, 1]
            ]
            + [
                {
                    "layer": layer,
                    "keys": [
                        f"{source_prefix}/MultiHeadDotProductAttention_1/{attvar}/{name}"
                        for name in ["kernel", "bias"]
                    ],
                    "reshape": True,
                }
                for attvar, layer in [
                    ("query", encoder.att.query_dense),
                    ("key", encoder.att.key_dense),
                    ("value", encoder.att.value_dense),
                    ("out", encoder.att.combine_heads),
                ]
            ]
        )

    matches.extend(
        [
            {"layer": model.get_layer("embedding"), "keys": ["embedding/kernel", "embedding/bias"]},
            {"layer": model.get_layer("class_token"), "keys": ["cls"]},
            {
                "layer": model.get_layer("Transformer_encoder_norm"),
                "keys": [f"Transformer/encoder_norm/{name}" for name in ["scale", "bias"]],
            },
        ]
    )

    _apply_embedding_weights(
        target_layer=model.get_layer("Transformer_posembed_input"),
        source_weights=params_dict["Transformer/posembed_input/pos_embedding"],
        num_x_patches=num_x_patches,
        num_y_patches=num_y_patches,
    )
    source_keys_used.append("Transformer/posembed_input/pos_embedding")

    for match in matches:
        source_keys_used.extend(match["keys"])
        source_weights = [params_dict[key] for key in match["keys"]]
        if match.get("reshape", False):
            source_weights = [
                source.reshape(expected.shape)
                for source, expected in zip(source_weights, match["layer"].get_weights())
            ]
        match["layer"].set_weights(source_weights)

    unused = set(source_keys).difference(source_keys_used)
    if unused:
        warnings.warn(f"Pesos ViT no usados: {unused}", UserWarning, stacklevel=2)


def _build_vit_model(
    *,
    name: str,
    input_shape: tuple[int, int, int],
    patch_size: int,
    config: dict,
) -> keras.Model:
    height, width, _channels = input_shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"input_shape={input_shape} debe ser multiplo de patch_size={patch_size}"
        )
    grid_h = height // patch_size
    grid_w = width // patch_size

    inputs = keras.Input(shape=input_shape, name="input")
    x = layers.Conv2D(
        filters=config["hidden_size"],
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="embedding",
    )(inputs)
    x = layers.Reshape((grid_h * grid_w, config["hidden_size"]), name="patch_tokens")(x)
    x = ClassToken(name="class_token")(x)
    x = AddPositionEmbs(name="Transformer_posembed_input")(x)
    for block_idx in range(config["num_layers"]):
        x, _ = TransformerBlock(
            num_heads=config["num_heads"],
            mlp_dim=config["mlp_dim"],
            dropout=config["dropout"],
            name=f"Transformer_encoderblock_{block_idx}",
        )(x)
    x = layers.LayerNormalization(epsilon=1e-6, name="Transformer_encoder_norm")(x)
    outputs = PatchTokensToSpatial((grid_h, grid_w), name="spatial_features")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def _download_vit_weights(vit_size: str, weights_tag: str) -> str:
    if weights_tag not in WEIGHTS_CLASSES:
        raise ValueError(f"Pesos ViT no soportados: {weights_tag!r}")
    filename = f"ViT-{vit_size}_{weights_tag}.npz"
    return keras.utils.get_file(filename, f"{BASE_URL}/{filename}", cache_subdir="weights")


class ViTBackbone(Backbone):
    patch_size: ClassVar[int]
    vit_size: ClassVar[str]
    config: ClassVar[dict]
    input_size: ClassVar[InputSize] = (224, 224)
    default_weights = "imagenet"
    weights_tag = "imagenet21k+imagenet2012"

    def preprocess_input(self, x):
        return preprocess_input(x)

    def build(
        self,
        *,
        weights=DEFAULT_WEIGHTS,
        include_top: bool = False,
        input_shape: tuple[int, int, int] | None = None,
        **kwargs,
    ) -> keras.Model:
        if include_top:
            raise ValueError("ViT del pipeline usa include_top=False; la cabeza la agrega ModelBuilder")
        weights = self.coalesce_weights(weights)
        shape = self.input_shape_or_default(input_shape)
        model = _build_vit_model(
            name=self.key,
            input_shape=shape,
            patch_size=self.patch_size,
            config=self.config,
        )
        if weights == "imagenet":
            params_path = _download_vit_weights(self.vit_size, self.weights_tag)
            num_y_patches = shape[0] // self.patch_size
            num_x_patches = shape[1] // self.patch_size
            _load_vit_weights(model, params_path, num_x_patches, num_y_patches)
        elif weights is not None:
            model.load_weights(str(weights))
        model._name = self.key
        return model


class ViTB16Backbone(ViTBackbone):
    key = "vitb16"
    patch_size = 16
    vit_size = "B_16"
    config = CONFIG_B


class ViTB32Backbone(ViTBackbone):
    key = "vitb32"
    patch_size = 32
    vit_size = "B_32"
    config = CONFIG_B


class ViTL16Backbone(ViTBackbone):
    key = "vitl16"
    patch_size = 16
    vit_size = "L_16"
    config = CONFIG_L
