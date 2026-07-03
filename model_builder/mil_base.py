from tensorflow import keras
from tensorflow.keras import layers

from src.model_builder.base import BaseModelBuilder
from src.model_builder.layers import BagTiling


class MilModelBuilderBase(BaseModelBuilder):
    def __init__(self, *args, bag_size=None, attention_dim=128, attention_gated=True, bag_grid=(3, 3), bag_keras_tiling=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.bag_grid = (int(bag_grid[0]), int(bag_grid[1]))
        self.bag_size = int(bag_size) if bag_size is not None else self.bag_grid[0] * self.bag_grid[1]
        self.attention_dim = attention_dim
        self.attention_gated = attention_gated
        self.bag_keras_tiling = bag_keras_tiling

    def augmentation_seq(self):
        # Augmentacion moderada: menos jitter que aggressive_augmentation porque
        # el tiling fijo ya mueve el contenido entre tiles en cada transformacion.
        return keras.Sequential([
            layers.RandomFlip("horizontal", name="aug_flip_h"),
            layers.RandomRotation(0.03, fill_mode="reflect", name="aug_rot"),
            layers.RandomZoom(height_factor=(0.0, 0.10), width_factor=(0.0, 0.10), fill_mode="reflect", name="aug_zoom"),
            layers.RandomTranslation(height_factor=0.10, width_factor=0.10, fill_mode="reflect", name="aug_translate"),
            layers.RandomContrast(0.15, name="aug_contrast"),
            layers.RandomBrightness(0.15, value_range=(0.0, 255.0), name="aug_brightness"),
        ], name="augmentation_mil")

    def mil_inputs(self):
        # bag de K parches (K, H, W, 3) — el dataset ya arma los tiles
        return keras.Input(shape=(self.bag_size, self.IMG_SIZE[0], self.IMG_SIZE[1], 3), name="bag")

    def mil_inputs_full(self):
        # imagen completa q despues se parte en tiles adentro del modelo
        rows, cols = self.bag_grid
        h, w = self.IMG_SIZE
        return keras.Input(shape=(rows * h, cols * w, 3), name="bag_full_image")

    def encode_instances(self, inputs):
        # corre backbone en cada instancia del bag con TimeDistributed
        x = layers.TimeDistributed(self.augmentation_seq(), name="td_augmentation")(inputs)
        x = layers.TimeDistributed(layers.Lambda(self.preprocess_input, name="preprocess_input"), name="td_preprocess")(x)
        x = layers.TimeDistributed(self.backbone, name="td_backbone")(x)
        x = layers.TimeDistributed(layers.GlobalAveragePooling2D(name="gap"), name="td_gap")(x)
        x = layers.TimeDistributed(layers.Dense(self.top_dense, activation="relu", name="instance_dense"), name="td_instance_dense")(x)
        return layers.TimeDistributed(layers.Dropout(self.dropout, name="instance_dropout"), name="td_instance_dropout")(x)

    def encode_instances_keras_tiling(self, inputs):
        # augment/preprocess en imagen entera y despues tilea con BagTiling
        x = self.augmentation(inputs)
        x = self.preprocess(x)
        x = BagTiling(self.bag_grid, name="bag_tiling")(x)
        x = layers.TimeDistributed(self.backbone, name="td_backbone")(x)
        x = layers.TimeDistributed(layers.GlobalAveragePooling2D(name="gap"), name="td_gap")(x)
        x = layers.TimeDistributed(layers.Dense(self.top_dense, activation="relu", name="instance_dense"), name="td_instance_dense")(x)
        return layers.TimeDistributed(layers.Dropout(self.dropout, name="instance_dropout"), name="td_instance_dropout")(x)

    def bag_dropout(self, x):
        return layers.Dropout(self.dropout, dtype="float32", name="bag_dropout")(x)

    def pool_instances(self, x):
        raise NotImplementedError

    def wrap_model(self, inputs, outputs):
        return keras.Model(inputs, outputs, name=self.model_name)

    def build_td_tiling(self):
        inputs = self.mil_inputs()
        x = self.encode_instances(inputs)
        x = self.pool_instances(x)
        x = self.bag_dropout(x)
        outputs = self.output(x)
        self.model = self.wrap_model(inputs, outputs)
        return self.compile()

    def build_keras_tiling(self):
        inputs = self.mil_inputs_full()
        x = self.encode_instances_keras_tiling(inputs)
        x = self.pool_instances(x)
        x = self.bag_dropout(x)
        outputs = self.output(x)
        self.model = self.wrap_model(inputs, outputs)
        return self.compile()

    def build(self):
        return self.build_keras_tiling() if self.bag_keras_tiling else self.build_td_tiling()
