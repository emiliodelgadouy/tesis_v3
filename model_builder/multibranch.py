import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from src.model_builder.base import BaseModelBuilder
from src.model_builder.layers import BagTiling, GatedAttentionPooling


class MultibranchModelBuilder(BaseModelBuilder):
    """Fusiona contexto global FULL con evidencia local ABMIL.

    Recibe un único canvas grande. Una rama lo procesa completo y la otra lo
    divide en la misma grilla usada por ABMIL. Los dos encoders son separados
    para no forzar que una misma red resuelva simultáneamente escalas distintas.
    """

    model_name = "multibranch"

    def __init__(
        self,
        *args,
        tile_backbone,
        tile_preprocess_input,
        tile_size,
        bag_grid=(3, 3),
        attention_dim=128,
        attention_gated=True,
        fusion_dim=128,
        pretrained_full_builder=None,
        pretrained_abmil_builder=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if tile_backbone is None or tile_preprocess_input is None:
            raise ValueError("multibranch requiere un encoder y preprocess para tiles")
        self.bag_grid = (int(bag_grid[0]), int(bag_grid[1]))
        self.tile_size = (int(tile_size[0]), int(tile_size[1]))
        expected_canvas = (
            self.bag_grid[0] * self.tile_size[0],
            self.bag_grid[1] * self.tile_size[1],
        )
        if tuple(self.IMG_SIZE) != expected_canvas:
            raise ValueError(
                f"IMG_SIZE={self.IMG_SIZE} no coincide con grilla/tile "
                f"{self.bag_grid}×{self.tile_size}={expected_canvas}"
            )

        # Los wrappers dan nombres únicos a dos modelos Keras de la misma familia.
        self.backbone = keras.Model(
            self.backbone.input,
            self.backbone.output,
            name="full_backbone",
        )
        self.tile_backbone = keras.Model(
            tile_backbone.input,
            tile_backbone.output,
            name="tile_backbone",
        )
        self.tile_backbone.trainable = self.backbone.trainable
        self.tile_preprocess_input = tile_preprocess_input
        self.attention_dim = int(attention_dim)
        self.attention_gated = bool(attention_gated)
        self.fusion_dim = int(fusion_dim)
        if (pretrained_full_builder is None) != (pretrained_abmil_builder is None):
            raise ValueError(
                "El transfer multibranch requiere juntos los builders FULL y ABMIL"
            )
        self.pretrained_full_builder = pretrained_full_builder
        self.pretrained_abmil_builder = pretrained_abmil_builder
        self._branch_transfer_diagnostics = {}

    @staticmethod
    def _keep_batch_norm_frozen(model):
        for layer in model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

    def keep_batch_norm_frozen(self):
        self._keep_batch_norm_frozen(self.backbone)
        self._keep_batch_norm_frozen(self.tile_backbone)

    @staticmethod
    def _set_partial(model, trainable_fraction):
        total = len(model.layers)
        freeze_until = total - max(1, round(total * trainable_fraction))
        model.trainable = True
        for layer in model.layers[:freeze_until]:
            layer.trainable = False
        for layer in model.layers[freeze_until:]:
            layer.trainable = True

    def make_backbone_partially_trainable(
        self,
        trainable_fraction=0.30,
        learning_rate=None,
        train_batch_norm=False,
    ):
        self._set_transferred_branch_heads_trainable(True)
        self._set_partial(self.backbone, trainable_fraction)
        self._set_partial(self.tile_backbone, trainable_fraction)
        if not train_batch_norm:
            self.keep_batch_norm_frozen()
        if learning_rate is not None:
            self.learning_rate = learning_rate
        return self.compile()

    def make_backbone_trainable(
        self,
        trainable=True,
        learning_rate=None,
        train_batch_norm=False,
    ):
        self._set_transferred_branch_heads_trainable(trainable)
        self.backbone.trainable = trainable
        self.tile_backbone.trainable = trainable
        if trainable and not train_batch_norm:
            self.keep_batch_norm_frozen()
        if learning_rate is not None:
            self.learning_rate = learning_rate
        return self.compile()

    def build(self):
        inputs = self.inputs()
        augmented = self.augmentation(inputs)

        full = layers.Lambda(
            self.preprocess_input,
            name="full_preprocess_input",
        )(augmented)
        full = self.backbone(full)
        full = layers.GlobalAveragePooling2D(name="full_gap")(full)
        full = layers.Dense(
            self.top_dense,
            activation="relu",
            name="full_dense",
        )(full)
        full = layers.Dropout(self.dropout, name="full_dropout")(full)
        full = layers.LayerNormalization(name="full_feature_norm")(full)

        tiles = BagTiling(self.bag_grid, name="bag_tiling")(augmented)
        tiles = layers.TimeDistributed(
            layers.Lambda(
                self.tile_preprocess_input,
                name="tile_preprocess_input",
            ),
            name="td_tile_preprocess",
        )(tiles)
        tiles = layers.TimeDistributed(
            self.tile_backbone,
            name="td_tile_backbone",
        )(tiles)
        tiles = layers.TimeDistributed(
            layers.GlobalAveragePooling2D(name="tile_gap"),
            name="td_tile_gap",
        )(tiles)
        tiles = layers.TimeDistributed(
            layers.Dense(
                self.top_dense,
                activation="relu",
                name="instance_dense",
            ),
            name="td_instance_dense",
        )(tiles)
        tiles = layers.TimeDistributed(
            layers.Dropout(self.dropout, name="instance_dropout"),
            name="td_instance_dropout",
        )(tiles)
        local = GatedAttentionPooling(
            attention_dim=self.attention_dim,
            gated=self.attention_gated,
            name="attention_pooling",
        )(tiles)
        local = layers.LayerNormalization(name="local_feature_norm")(local)

        fused = layers.Concatenate(name="full_local_features")([full, local])
        fused = layers.Dense(
            self.fusion_dim,
            activation="relu",
            name="fusion_dense",
        )(fused)
        fused = layers.Dropout(self.dropout, name="fusion_dropout")(fused)
        outputs = self.output(fused)

        self.model = keras.Model(inputs, outputs, name=self.model_name)
        self._transfer_pretrained_branches()
        return self.compile()

    def _set_transferred_branch_heads_trainable(self, trainable):
        if self.pretrained_full_builder is None or self.model is None:
            return
        self.model.get_layer("full_dense").trainable = trainable
        self.model.get_layer("td_instance_dense").layer.trainable = trainable
        self.model.get_layer("attention_pooling").trainable = trainable

    @staticmethod
    def _max_abs_weight_diff(source_weights, target_weights):
        if len(source_weights) != len(target_weights):
            return float("inf")
        differences = [
            float(np.max(np.abs(source - target)))
            for source, target in zip(source_weights, target_weights)
            if source.size
        ]
        return max(differences, default=0.0)

    def _transfer_pretrained_branches(self):
        """Inicializa ambas ramas desde modelos mamográficos ya entrenados.

        FULL aporta su encoder y proyección global. ABMIL aporta el encoder de
        tiles, la proyección de instancia y la atención. Los clasificadores
        binarios se descartan porque la salida multibranch debe aprenderse nueva.
        """
        full_source = self.pretrained_full_builder
        abmil_source = self.pretrained_abmil_builder
        if full_source is None:
            return
        if full_source.model is None or abmil_source.model is None:
            raise RuntimeError("Los modelos FULL/ABMIL fuente deben estar construidos")

        try:
            source_full_dense = full_source.model.get_layer("dense")
            source_instance_dense = abmil_source.model.get_layer(
                "td_instance_dense"
            ).layer
            source_attention = abmil_source.model.get_layer("attention_pooling")
            target_full_dense = self.model.get_layer("full_dense")
            target_instance_dense = self.model.get_layer("td_instance_dense").layer
            target_attention = self.model.get_layer("attention_pooling")
        except ValueError as exc:
            raise RuntimeError(
                "Los checkpoints fuente no tienen la arquitectura FULL/ABMIL esperada"
            ) from exc

        self.backbone.set_weights(full_source.backbone.get_weights())
        target_full_dense.set_weights(source_full_dense.get_weights())
        self.tile_backbone.set_weights(abmil_source.backbone.get_weights())
        target_instance_dense.set_weights(source_instance_dense.get_weights())
        target_attention.set_weights(source_attention.get_weights())

        self._branch_transfer_diagnostics = {
            "transfer_applied": 1.0,
            "transfer_full_backbone_max_abs_diff": self._max_abs_weight_diff(
                full_source.backbone.get_weights(), self.backbone.get_weights()
            ),
            "transfer_full_dense_max_abs_diff": self._max_abs_weight_diff(
                source_full_dense.get_weights(), target_full_dense.get_weights()
            ),
            "transfer_abmil_backbone_max_abs_diff": self._max_abs_weight_diff(
                abmil_source.backbone.get_weights(), self.tile_backbone.get_weights()
            ),
            "transfer_abmil_dense_max_abs_diff": self._max_abs_weight_diff(
                source_instance_dense.get_weights(), target_instance_dense.get_weights()
            ),
            "transfer_abmil_attention_max_abs_diff": self._max_abs_weight_diff(
                source_attention.get_weights(), target_attention.get_weights()
            ),
        }
        self._set_transferred_branch_heads_trainable(False)
        print(
            "Transfer multibranch aplicado: FULL (backbone+dense) + "
            "ABMIL (backbone+instance_dense+attention); clasificadores descartados"
        )

    def branch_transfer_diagnostics(self) -> dict[str, float]:
        return dict(self._branch_transfer_diagnostics)

    def branch_fusion_weight_norms(self) -> dict[str, float]:
        """Magnitud relativa de cada mitad de entrada de la cabeza de fusión."""
        kernel = self.model.get_layer("fusion_dense").get_weights()[0]
        split = self.top_dense
        full_norm = float(np.linalg.norm(kernel[:split]))
        local_norm = float(np.linalg.norm(kernel[split:]))
        total = full_norm + local_norm
        return {
            "fusion_full_weight_norm": full_norm,
            "fusion_local_weight_norm": local_norm,
            "fusion_local_weight_fraction": local_norm / total if total else 0.0,
        }
