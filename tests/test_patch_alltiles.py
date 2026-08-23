import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.dataset_provider import build_dataset_provider
from src.utils import expand_grid_patch_table


def _config():
    return {
        "GENERAL": {"RANDOM_SEED": 42},
        "FULL": {"BAG_GRID": (2, 2)},
        "PATCH_ALLTILES": {
            "MIN_ROI_TILE_OVERLAP": 0.0,
            "HARD_NEGATIVE_TO_POSITIVE_RATIO": 1.0,
            "RANDOM_NEGATIVE_TO_POSITIVE_RATIO": 1.0,
        },
    }


def _table():
    return pd.DataFrame(
        {
            "path": ["positive.png", "negative.png"],
            "image_id": ["positive", "negative"],
            "laterality": ["L", "L"],
            "cls": [1.0, 0.0],
            "pad_resized_xmin_norm": [0.05, np.nan],
            "pad_resized_ymin_norm": [0.05, np.nan],
            "pad_resized_xmax_norm": [0.45, np.nan],
            "pad_resized_ymax_norm": [0.45, np.nan],
        }
    )


class PatchAllTilesTableTest(unittest.TestCase):
    def test_exhaustive_table_labels_every_grid_tile(self):
        expanded = expand_grid_patch_table(_config(), _table(), training=False)

        self.assertEqual(len(expanded), 8)
        self.assertEqual(int(expanded["cls"].sum()), 1)
        positive = expanded[expanded["_bag_cls"] >= 0.5]
        self.assertEqual(int(positive.loc[positive["cls"] >= 0.5, "_patch_tile_index"].iloc[0]), 0)
        self.assertEqual(set(expanded["_patch_tile_index"]), {0, 1, 2, 3})

    def test_training_table_samples_both_negative_sources(self):
        expanded = expand_grid_patch_table(_config(), _table(), training=True)

        self.assertEqual(int((expanded["cls"] >= 0.5).sum()), 1)
        self.assertEqual(int(((expanded["_bag_cls"] >= 0.5) & (expanded["cls"] < 0.5)).sum()), 1)
        self.assertEqual(int((expanded["_bag_cls"] < 0.5).sum()), 1)

    def test_lateralization_flips_the_positive_tile(self):
        table = _table().iloc[[0]].copy()
        table["laterality"] = "R"
        expanded = expand_grid_patch_table(_config(), table, training=False)

        positive_tile = int(expanded.loc[expanded["cls"] >= 0.5, "_patch_tile_index"].iloc[0])
        self.assertEqual(positive_tile, 1)

    def test_provider_extracts_the_explicit_grid_tile(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            image[:4, :4] = 10
            image[:4, 4:] = 20
            image[4:, :4] = 30
            image[4:, 4:] = 40
            path = Path(tmp) / "grid.png"
            tf.io.write_file(str(path), tf.io.encode_png(image))
            table = pd.DataFrame(
                {
                    "path": [str(path)] * 4,
                    "cls": [0.0] * 4,
                    "_patch_tile_index": [0, 1, 2, 3],
                }
            )
            provider = build_dataset_provider(
                None,
                (4, 4),
                4,
                mode="patch_alltiles",
                bag_grid=(2, 2),
                bag_canvas_mode="resize",
                use_clahe=False,
                cache_dataset=False,
            )

            images, _ = next(iter(provider.build_eval(table).unwrap()))
            means = tf.reduce_mean(images, axis=(1, 2, 3)).numpy()
            np.testing.assert_allclose(means, [10, 20, 30, 40], atol=1e-5)

    def test_patch_tiles_match_the_pretiled_abmil_bag_after_lateralization(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
            path = Path(tmp) / "grid.png"
            tf.io.write_file(str(path), tf.io.encode_png(image))
            bag_table = pd.DataFrame(
                {"path": [str(path)], "cls": [1.0], "laterality": ["R"]}
            )
            patch_table = bag_table.loc[bag_table.index.repeat(4)].reset_index(drop=True)
            patch_table["_patch_tile_index"] = np.arange(4, dtype=np.int32)

            common = {
                "bag_grid": (2, 2),
                "bag_canvas_mode": "resize",
                "use_clahe": False,
                "cache_dataset": False,
                "lateralize": True,
            }
            patch_provider = build_dataset_provider(
                None,
                (4, 4),
                4,
                mode="patch_alltiles",
                **common,
            )
            bag_provider = build_dataset_provider(
                None,
                (4, 4),
                1,
                mode="abmil",
                bag_keras_tiling=False,
                **common,
            )

            patch_images, _ = next(iter(patch_provider.build_eval(patch_table).unwrap()))
            bags, _ = next(iter(bag_provider.build_eval(bag_table).unwrap()))
            np.testing.assert_allclose(patch_images.numpy(), bags.numpy()[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
