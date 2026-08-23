from tensorflow import keras
from tensorflow.keras import layers

from src.model_builder.simple import SimpleModelBuilder


class PatchModelBuilder(SimpleModelBuilder):
    # clasificador de parches: mismo grafo que simple; el crop lo hace el dataset
    model_name = "patch"


class PatchHardnegModelBuilder(PatchModelBuilder):
    model_name = "patch_hardneg"


class PatchAllTilesModelBuilder(PatchModelBuilder):
    """Clasificador patch entrenado sobre candidatos de toda la grilla MIL."""

    model_name = "patch_alltiles"

    def augmentation_seq(self):
        """Replica la augmentación por tile usada por ABMIL pre-tileado."""
        layers_list: list[layers.Layer] = []
        if not self.lateralized_inputs:
            layers_list.append(layers.RandomFlip("horizontal", name="aug_flip_h"))
        layers_list.extend(
            [
                layers.RandomRotation(0.03, fill_mode="reflect", name="aug_rot"),
                layers.RandomZoom(
                    height_factor=(0.0, 0.10),
                    width_factor=(0.0, 0.10),
                    fill_mode="reflect",
                    name="aug_zoom",
                ),
                layers.RandomTranslation(
                    height_factor=0.10,
                    width_factor=0.10,
                    fill_mode="reflect",
                    name="aug_translate",
                ),
                layers.RandomContrast(0.15, name="aug_contrast"),
                layers.RandomBrightness(
                    0.15,
                    value_range=(0.0, 255.0),
                    name="aug_brightness",
                ),
            ]
        )
        return keras.Sequential(layers_list, name="augmentation_mil_tile")
