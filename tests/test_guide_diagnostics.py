import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.guide_diagnostics import _tile_roi_overlap, evaluate_patch_guide_localization


def _config():
    return {
        "GENERAL": {
            "RANDOM_SEED": 42,
            "USE_CLAHE": False,
            "BATCH_SIZE": 4,
        },
        "MIL": {
            "BATCH_SIZE": 2,
            "BAG_KERAS_TILING": True,
            "GUIDED_ATTENTION_TEMPERATURE": 1.0,
        },
        "PATCH": {"RESIZE_TO_BAG_CANVAS": True},
        "FULL": {"BAG_GRID": (2, 2), "BAG_CANVAS_MODE": "resize"},
    }


def _mean_intensity_model():
    inputs = keras.Input((4, 4, 3))
    channels = layers.GlobalAveragePooling2D()(inputs)
    logits = layers.Lambda(
        lambda values: tf.reduce_mean(values, axis=1, keepdims=True)
    )(channels)
    return keras.Model(inputs, logits)


class PatchGuideDiagnosticsTest(unittest.TestCase):
    def test_tile_roi_overlap_matches_grid_geometry(self):
        overlap = _tile_roi_overlap((0.55, 0.05, 0.95, 0.45), (2, 2))

        self.assertEqual(int(np.argmax(overlap)), 1)
        self.assertGreater(overlap[1], 0.0)
        self.assertEqual(int(np.count_nonzero(overlap)), 1)

    def test_evaluation_uses_all_tiles_and_finds_roi(self):
        with tempfile.TemporaryDirectory() as tmp:
            positive = np.zeros((8, 8, 3), dtype=np.uint8)
            positive[:4, :4] = 255
            negative = np.zeros((8, 8, 3), dtype=np.uint8)
            positive_path = Path(tmp) / "positive.png"
            negative_path = Path(tmp) / "negative.png"
            tf.io.write_file(str(positive_path), tf.io.encode_png(positive))
            tf.io.write_file(str(negative_path), tf.io.encode_png(negative))

            table = pd.DataFrame(
                {
                    "path": [str(positive_path), str(negative_path)],
                    "cls": [1.0, 0.0],
                    "laterality": ["L", "L"],
                    "image_id": ["positive", "negative"],
                    "pad_resized_xmin_norm": [0.05, np.nan],
                    "pad_resized_ymin_norm": [0.05, np.nan],
                    "pad_resized_xmax_norm": [0.45, np.nan],
                    "pad_resized_ymax_norm": [0.45, np.nan],
                }
            )
            builder = SimpleNamespace(
                IMG_SIZE=(4, 4),
                model=_mean_intensity_model(),
            )

            result = evaluate_patch_guide_localization(
                _config(),
                builder,
                val=table,
                show_examples=False,
            )

            summary = result.summary.loc["val"]
            self.assertEqual(summary["roi_top1_hit_rate"], 1.0)
            self.assertEqual(summary["roi_top3_hit_rate"], 1.0)
            self.assertEqual(summary["bag_max_roc_auc"], 1.0)
            positive_row = result.per_image["val"].iloc[0]
            self.assertEqual(positive_row["top_tile_index"], 0)
            self.assertEqual(len(positive_row["patch_probabilities"]), 4)


if __name__ == "__main__":
    unittest.main()
