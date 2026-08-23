import unittest

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.model_builder.factory import create_model_builder


def _config():
    return {
        "GENERAL": {"METRIC_TO_MAXIMIZE": "pr_auc"},
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
        },
        "FULL": {"BAG_GRID": (2, 2)},
        "MULTIBRANCH": {"FUSION_DIM": 5},
    }


def _backbone(size, name):
    inputs = keras.Input((*size, 3))
    outputs = layers.Conv2D(4, 3, padding="same", name=f"{name}_conv")(inputs)
    return keras.Model(inputs, outputs, name=name)


class MultibranchModelTest(unittest.TestCase):
    def _standalone_sources(self):
        full = create_model_builder(
            _config(),
            (8, 8),
            _backbone((8, 8), "pretrained_full_source"),
            lambda x: x,
            mode="full",
            top_dense=6,
            dropout=0.0,
            initial_bias=0.0,
            jit_compile=False,
            steps_per_execution=1,
        )
        full.build()
        abmil = create_model_builder(
            _config(),
            (4, 4),
            _backbone((4, 4), "pretrained_abmil_source"),
            lambda x: x,
            mode="abmil",
            top_dense=6,
            dropout=0.0,
            initial_bias=0.0,
            jit_compile=False,
            steps_per_execution=1,
        )
        abmil.build()
        return full, abmil

    def test_builds_one_output_from_full_and_local_branches(self):
        builder = create_model_builder(
            _config(),
            (8, 8),
            _backbone((8, 8), "full_source"),
            lambda x: x,
            mode="multibranch",
            tile_backbone=_backbone((4, 4), "tile_source"),
            tile_preprocess_input=lambda x: x,
            tile_size=(4, 4),
            top_dense=6,
            dropout=0.0,
            initial_bias=0.0,
            jit_compile=False,
            steps_per_execution=1,
        )
        builder.build()

        logits = builder.model(tf.zeros((2, 8, 8, 3)), training=False)
        self.assertEqual(tuple(logits.shape), (2, 1))
        self.assertIn("full_backbone", {layer.name for layer in builder.model.layers})
        self.assertIn("td_tile_backbone", {layer.name for layer in builder.model.layers})
        diagnostics = builder.branch_fusion_weight_norms()
        self.assertGreater(diagnostics["fusion_full_weight_norm"], 0.0)
        self.assertGreater(diagnostics["fusion_local_weight_norm"], 0.0)

    def test_partial_finetune_updates_both_encoders(self):
        builder = create_model_builder(
            _config(),
            (8, 8),
            _backbone((8, 8), "full_source"),
            lambda x: x,
            mode="multibranch",
            tile_backbone=_backbone((4, 4), "tile_source"),
            tile_preprocess_input=lambda x: x,
            tile_size=(4, 4),
            top_dense=6,
            jit_compile=False,
            steps_per_execution=1,
        )
        builder.build()
        builder.make_backbone_partially_trainable(
            trainable_fraction=0.5,
            learning_rate=1e-4,
        )

        self.assertTrue(builder.backbone.trainable)
        self.assertTrue(builder.tile_backbone.trainable)

    def test_transfers_both_specialists_and_freezes_them_for_stage_one(self):
        full_source, abmil_source = self._standalone_sources()
        builder = create_model_builder(
            _config(),
            (8, 8),
            _backbone((8, 8), "fresh_full_target"),
            lambda x: x,
            mode="multibranch",
            tile_backbone=_backbone((4, 4), "fresh_tile_target"),
            tile_preprocess_input=lambda x: x,
            tile_size=(4, 4),
            pretrained_full_builder=full_source,
            pretrained_abmil_builder=abmil_source,
            top_dense=6,
            dropout=0.0,
            initial_bias=0.0,
            jit_compile=False,
            steps_per_execution=1,
        )
        builder.build()

        diagnostics = builder.branch_transfer_diagnostics()
        self.assertEqual(diagnostics["transfer_applied"], 1.0)
        for name, value in diagnostics.items():
            if name.endswith("max_abs_diff"):
                self.assertEqual(value, 0.0, name)
        self.assertFalse(builder.model.get_layer("full_dense").trainable)
        self.assertFalse(builder.model.get_layer("td_instance_dense").layer.trainable)
        self.assertFalse(builder.model.get_layer("attention_pooling").trainable)

        builder.make_backbone_partially_trainable(
            trainable_fraction=0.5,
            learning_rate=1e-4,
        )
        self.assertTrue(builder.model.get_layer("full_dense").trainable)
        self.assertTrue(builder.model.get_layer("td_instance_dense").layer.trainable)
        self.assertTrue(builder.model.get_layer("attention_pooling").trainable)


if __name__ == "__main__":
    unittest.main()
