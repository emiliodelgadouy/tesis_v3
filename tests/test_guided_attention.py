import unittest

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.modes import is_mil_mode, normalize_mode
from src.model_builder.factory import create_model_builder
from src.model_builder.layers import GuidedGatedAttentionPooling


def _config():
    return {
        "GENERAL": {"METRIC_TO_MAXIMIZE": "auc"},
        "TRAINING": {
            "FOCAL_GAMMA": 2.0,
            "AGGRESSIVE_AUGMENTATION": False,
            "EARLY_STOPPING_PATIENCE": 2,
            "REDUCE_LR_PATIENCE": 1,
        },
        "MIL": {
            "BAG_KERAS_TILING": True,
            "ATTENTION_DIM": 4,
            "ATTENTION_GATED": True,
            "GUIDED_ATTENTION_TEMPERATURE": 1.0,
            "GUIDED_ATTENTION_STRENGTH": 1.0,
        },
        "FULL": {"BAG_GRID": (2, 2)},
    }


def _backbone():
    inputs = keras.Input((8, 8, 3))
    outputs = layers.Conv2D(4, 3, padding="same", name="tiny_conv")(inputs)
    return keras.Model(inputs, outputs, name="tiny_backbone")


def _patch_builder():
    backbone = _backbone()
    builder = create_model_builder(
        _config(),
        (8, 8),
        backbone,
        lambda x: x,
        mode="patch_hardneg",
        initial_bias=0.0,
        focal_alpha=0.75,
        top_dense=6,
        dropout=0.0,
        jit_compile=False,
        steps_per_execution=1,
    )
    builder.build()
    dense = builder.model.get_layer("dense")
    dense.set_weights(
        [
            np.full_like(dense.get_weights()[0], 0.25),
            np.full_like(dense.get_weights()[1], 0.5),
        ]
    )
    output = builder.model.get_layer("output")
    output.set_weights(
        [
            np.full_like(output.get_weights()[0], 7.0),
            np.full_like(output.get_weights()[1], 3.0),
        ]
    )
    # El test no usa checkpoints; evita que el builder intente cargarlos.
    builder._global_checkpoint_loaded = True
    return builder


def _transferred_builder(mode):
    source = _patch_builder()
    target = create_model_builder(
        _config(),
        (8, 8),
        source.backbone,
        source.preprocess_input,
        mode=mode,
        initial_bias=-1.0,
        focal_alpha=0.75,
        pretrained_builder=source,
        top_dense=6,
        dropout=0.0,
        jit_compile=False,
        steps_per_execution=1,
    )
    target.build()
    return source, target


class GuidedAttentionTest(unittest.TestCase):
    def test_guided_attention_starts_from_patch_logits(self):
        features = tf.constant([[[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]]])
        guide_logits = tf.constant([[[-1.0], [0.0], [2.0]]])
        layer = GuidedGatedAttentionPooling(
            attention_dim=3,
            guide_temperature=1.0,
            guide_strength=1.0,
        )

        pooled = layer([features, guide_logits])

        expected_attention = tf.nn.softmax(guide_logits, axis=1)
        expected_pooled = tf.reduce_sum(expected_attention * features, axis=1)
        np.testing.assert_allclose(layer.last_attention.numpy(), expected_attention.numpy(), atol=1e-6)
        np.testing.assert_allclose(pooled.numpy(), expected_pooled.numpy(), atol=1e-6)

    def test_plain_transfer_keeps_a_new_trainable_bag_output(self):
        source, target = _transferred_builder("abmil_patch_hardneg")

        source_output = source.model.get_layer("output").get_weights()
        bag_output = target.model.get_layer("output")

        self.assertTrue(bag_output.trainable)
        self.assertNotIn("td_instance_output", {layer.name for layer in target.model.layers})
        self.assertFalse(np.array_equal(bag_output.get_weights()[0], source_output[0]))
        self.assertFalse(target.model.get_layer("td_instance_dense").layer.trainable)

    def test_guided_transfer_copies_patch_head_only_to_instance_guide(self):
        source, target = _transferred_builder("abmil_patch_hardneg_guided")

        source_output = source.model.get_layer("output").get_weights()
        instance_output = target.model.get_layer("td_instance_output").layer
        bag_output = target.model.get_layer("output")

        np.testing.assert_allclose(instance_output.get_weights()[0], source_output[0])
        np.testing.assert_allclose(instance_output.get_weights()[1], source_output[1])
        self.assertFalse(instance_output.trainable)
        self.assertTrue(bag_output.trainable)
        self.assertFalse(np.array_equal(bag_output.get_weights()[0], source_output[0]))
        self.assertEqual(target.model.get_layer("guided_attention_pooling").w.numpy().sum(), 0.0)

    def test_transferred_layers_unfreeze_after_stage_one(self):
        _, target = _transferred_builder("abmil_patch_hardneg_guided")

        target.make_backbone_partially_trainable(trainable_fraction=0.5, learning_rate=1e-4)

        self.assertTrue(target.model.get_layer("td_instance_dense").layer.trainable)
        self.assertTrue(target.model.get_layer("td_instance_output").layer.trainable)

    def test_guided_mode_normalization_and_dispatch(self):
        self.assertEqual(normalize_mode("ABMIL-PATCH-HARDNEG-GUIDED"), "abmil_patch_hardneg_guided")
        self.assertTrue(is_mil_mode("abmil_patch_hardneg_guided"))


if __name__ == "__main__":
    unittest.main()
