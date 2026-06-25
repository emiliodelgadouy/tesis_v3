"""CLAM-SB (Lu et al., Nat Biomed Eng 2021) en TensorFlow/Keras.

Single-branch CLAM con atencion gated, clasificador a nivel bag y perdida de
clustering a nivel instancia sobre los parches mas y menos atendidos.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.modes import (
    MODE_DESCRIPTIONS,
    format_all_mode_descriptions,
    normalize_mode,
)

# Aliases legacy
MIL_ARCHITECTURE_DESCRIPTIONS = {
    k: v for k, v in MODE_DESCRIPTIONS.items() if k != "simple"
}


def get_mil_architecture_description(name: str) -> str:
    key = normalize_mode(name)
    return MIL_ARCHITECTURE_DESCRIPTIONS.get(
        key,
        f"Arquitectura MIL '{name}' sin descripcion detallada.",
    )


format_all_mil_architecture_descriptions = format_all_mode_descriptions
normalize_mil_architecture_name = normalize_mode


def _effective_k_sample(k_sample: int, num_instances: tf.Tensor) -> tf.Tensor:
    k = tf.minimum(int(k_sample), num_instances // 2)
    return tf.maximum(k, 1)


def _instance_loss_for_class(
    h: tf.Tensor,
    attention: tf.Tensor,
    bag_labels: tf.Tensor,
    class_idx: int,
    classifier: layers.Layer,
    *,
    k_sample: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Perdida de instancia para la rama in-the-class de un solo subtipo."""
    mask_bool = tf.equal(bag_labels, class_idx)
    mask_count = tf.reduce_sum(tf.cast(mask_bool, tf.int32))
    feature_dim = tf.shape(h)[-1]
    k = _effective_k_sample(k_sample, tf.shape(h)[1])

    def _compute():
        h_b = tf.boolean_mask(h, mask_bool)
        a_b = tf.boolean_mask(attention, mask_bool)
        b = tf.shape(h_b)[0]
        top_p_idx = tf.nn.top_k(a_b, k=k).indices
        top_n_idx = tf.nn.top_k(-a_b, k=k).indices
        batch_idx = tf.tile(tf.reshape(tf.range(b), (-1, 1)), [1, k])
        top_p = tf.gather_nd(h_b, tf.stack([batch_idx, top_p_idx], axis=-1))
        top_n = tf.gather_nd(h_b, tf.stack([batch_idx, top_n_idx], axis=-1))
        all_inst = tf.reshape(
            tf.concat([top_p, top_n], axis=1), (-1, feature_dim)
        )
        logits = classifier(all_inst)
        targets = tf.concat(
            [
                tf.ones(b * k, dtype=tf.int32),
                tf.zeros(b * k, dtype=tf.int32),
            ],
            axis=0,
        )
        per_inst = keras.losses.sparse_categorical_crossentropy(
            targets, logits, from_logits=True
        )
        return tf.reduce_sum(per_inst), b

    def _zero():
        return tf.constant(0.0, dtype=tf.float32), tf.constant(0, dtype=tf.int32)

    return tf.cond(mask_count > 0, _compute, _zero)


def clam_instance_clustering_loss(
    h: tf.Tensor,
    attention: tf.Tensor,
    bag_labels: tf.Tensor,
    instance_classifiers: list[layers.Layer],
    *,
    k_sample: int,
) -> tf.Tensor:
    """Perdida de clustering CLAM-SB para clasificacion binaria (n_classes=2).

    Solo activa el clasificador de instancia de la rama in-the-class segun la
    etiqueta del bag (pseudo-etiquetas en top-k y bottom-k por atencion).
    """
    bag_labels = tf.reshape(tf.cast(bag_labels, tf.int32), [-1])
    attention = tf.squeeze(attention, axis=-1)

    total_loss = tf.constant(0.0, dtype=tf.float32)
    active = tf.constant(0, dtype=tf.int32)
    for class_idx, classifier in enumerate(instance_classifiers):
        class_loss, count = _instance_loss_for_class(
            h,
            attention,
            bag_labels,
            class_idx,
            classifier,
            k_sample=k_sample,
        )
        total_loss += class_loss
        active += count

    return tf.cond(
        active > 0,
        lambda: total_loss / tf.cast(active, tf.float32),
        lambda: tf.constant(0.0, dtype=tf.float32),
    )


class CLAMAttentionBlock(layers.Layer):
    """Bloque de atencion CLAM-SB: agrega el bag y expone estado para la loss de instancias."""

    def __init__(
        self,
        attention_dim: int = 128,
        gated: bool = True,
        k_sample: int = 8,
        n_classes: int = 2,
        **kwargs,
    ):
        kwargs.setdefault("dtype", "float32")
        super().__init__(**kwargs)
        self.attention_dim = int(attention_dim)
        self.gated = bool(gated)
        self.k_sample = int(k_sample)
        self.n_classes = int(n_classes)
        self.last_h = None
        self.last_attention = None
        self.instance_classifiers = [
            layers.Dense(2, dtype="float32", name=f"instance_classifier_{i}")
            for i in range(self.n_classes)
        ]

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        self.V = self.add_weight(
            name="V",
            shape=(feature_dim, self.attention_dim),
            initializer="glorot_uniform",
            trainable=True,
            dtype="float32",
        )
        if self.gated:
            self.U = self.add_weight(
                name="U",
                shape=(feature_dim, self.attention_dim),
                initializer="glorot_uniform",
                trainable=True,
                dtype="float32",
            )
        self.w = self.add_weight(
            name="w",
            shape=(self.attention_dim, 1),
            initializer="glorot_uniform",
            trainable=True,
            dtype="float32",
        )
        for classifier in self.instance_classifiers:
            classifier.build((None, feature_dim))
        super().build(input_shape)

    def call(self, inputs, training=False):
        h = tf.cast(inputs, tf.float32)
        vh = tf.tanh(tf.matmul(h, self.V))
        if self.gated:
            vh = vh * tf.sigmoid(tf.matmul(h, self.U))
        scores = tf.matmul(vh, self.w)
        attention = tf.nn.softmax(scores, axis=1)
        self.last_h = h
        self.last_attention = attention
        return tf.reduce_sum(attention * h, axis=1)

    def compute_instance_loss(self, bag_labels: tf.Tensor) -> tf.Tensor:
        if self.last_h is None or self.last_attention is None:
            return tf.constant(0.0, dtype=tf.float32)
        return clam_instance_clustering_loss(
            self.last_h,
            self.last_attention,
            bag_labels,
            self.instance_classifiers,
            k_sample=self.k_sample,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "attention_dim": self.attention_dim,
                "gated": self.gated,
                "k_sample": self.k_sample,
                "n_classes": self.n_classes,
            }
        )
        return config


class CLAMTrainingModel(keras.Model):
    """Modelo MIL con perdida combinada bag + clustering de instancias (CLAM)."""

    def __init__(
        self,
        *args,
        clam_attention_layer_name: str = "clam_attention",
        bag_loss_weight: float = 0.7,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.clam_attention_layer_name = clam_attention_layer_name
        self.bag_loss_weight = float(bag_loss_weight)

    def _clam_layer(self) -> CLAMAttentionBlock:
        return self.get_layer(self.clam_attention_layer_name)

    def _combined_loss(self, x, y, *, training: bool):
        y_pred = self(x, training=training)
        bag_loss = self.compiled_loss(
            y, y_pred, regularization_losses=self.losses
        )
        inst_loss = self._clam_layer().compute_instance_loss(y)
        total = (
            self.bag_loss_weight * bag_loss
            + (1.0 - self.bag_loss_weight) * inst_loss
        )
        return total, bag_loss, inst_loss, y_pred

    def train_step(self, data):
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            total_loss, bag_loss, inst_loss, y_pred = self._combined_loss(
                x, y, training=True
            )
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)
        return {
            **{metric.name: metric.result() for metric in self.metrics},
            "loss": total_loss,
            "bag_loss": bag_loss,
            "instance_loss": inst_loss,
        }

    def test_step(self, data):
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)
        total_loss, bag_loss, inst_loss, y_pred = self._combined_loss(
            x, y, training=False
        )
        self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)
        return {
            **{metric.name: metric.result() for metric in self.metrics},
            "loss": total_loss,
            "bag_loss": bag_loss,
            "instance_loss": inst_loss,
        }
