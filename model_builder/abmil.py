from tensorflow.keras import layers

from src.model_builder.layers import (
    GatedAttentionPooling,
    GuidedGatedAttentionPooling,
    LearnedGuidedGatedAttentionPooling,
)
from src.model_builder.mil_base import MilModelBuilderBase


class AbmilModelBuilder(MilModelBuilderBase):
    model_name = "abmil"

    def pool_instances(self, x):
        # atencion gated sobre instancias (Ilse et al.)
        return GatedAttentionPooling(attention_dim=self.attention_dim, gated=self.attention_gated, name="attention_pooling")(x)

    def build_keras_tiling(self):
        # Conserva la identidad del submodo (p.ej. abmil_patch_hardneg).
        self.model_name = f"{type(self).model_name}_keras_tiling"
        return super().build_keras_tiling()


class AbmilPatchHardnegModelBuilder(AbmilModelBuilder):
    model_name = "abmil_patch_hardneg"

    def build(self):
        if self.pretrained_builder is None:
            raise ValueError("abmil_patch_hardneg requiere pretrained_builder entrenado en patch_hardneg")
        return super().build()


class AbmilPatchHardnegGuidedModelBuilder(AbmilPatchHardnegModelBuilder):
    """ABMIL cuya atencion parte de los logits patch_hardneg por instancia."""

    model_name = "abmil_patch_hardneg_guided"

    def __init__(self, *args, guided_attention_temperature=1.0, guided_attention_strength=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.guided_attention_temperature = float(guided_attention_temperature)
        self.guided_attention_strength = float(guided_attention_strength)

    def _guided_inputs(self, features):
        guide_logits = layers.TimeDistributed(
            layers.Dense(1, dtype="float32", name="instance_output"),
            name="td_instance_output",
        )(features)
        return [self.instance_dropout(features), guide_logits]

    def encode_instances(self, inputs):
        return self._guided_inputs(self.encode_instance_features(inputs))

    def encode_instances_keras_tiling(self, inputs):
        return self._guided_inputs(self.encode_instance_features_keras_tiling(inputs))

    def pool_instances(self, inputs):
        return GuidedGatedAttentionPooling(
            attention_dim=self.attention_dim,
            gated=self.attention_gated,
            guide_temperature=self.guided_attention_temperature,
            guide_strength=self.guided_attention_strength,
            name="guided_attention_pooling",
        )(inputs)

    def _unfreeze_transferred_patch_layers(self):
        """Mantiene fija la guía patch legacy durante el fine-tuning MIL.

        La guía es la única supervisión espacial de este modo. Si se libera en
        la etapa 2, la loss de bag puede reescribir los logits patch y borrar
        silenciosamente el prior de localización medido antes. El backbone y
        la atención MIL siguen la transición normal de etapas.
        """
        return None


class AbmilPatchAlltilesModelBuilder(AbmilPatchHardnegModelBuilder):
    """Transfer simple desde un patch head entrenado sobre toda la grilla."""

    model_name = "abmil_patch_alltiles"

    def build(self):
        if self.pretrained_builder is None:
            raise ValueError(
                f"{self.model_name} requiere pretrained_builder entrenado en patch_alltiles"
            )
        return AbmilModelBuilder.build(self)


class AbmilPatchAlltilesGatedModelBuilder(AbmilPatchAlltilesModelBuilder):
    """Guía patch opcional mediante una compuerta escalar aprendible."""

    model_name = "abmil_patch_alltiles_gated"

    def __init__(self, *args, guided_attention_temperature=1.0, initial_guide_gate=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.guided_attention_temperature = float(guided_attention_temperature)
        self.initial_guide_gate = float(initial_guide_gate)

    def _guided_inputs(self, features):
        guide_logits = layers.TimeDistributed(
            layers.Dense(1, dtype="float32", name="instance_output"),
            name="td_instance_output",
        )(features)
        return [self.instance_dropout(features), guide_logits]

    def encode_instances(self, inputs):
        return self._guided_inputs(self.encode_instance_features(inputs))

    def encode_instances_keras_tiling(self, inputs):
        return self._guided_inputs(self.encode_instance_features_keras_tiling(inputs))

    def pool_instances(self, inputs):
        return LearnedGuidedGatedAttentionPooling(
            attention_dim=self.attention_dim,
            gated=self.attention_gated,
            guide_temperature=self.guided_attention_temperature,
            initial_guide_gate=self.initial_guide_gate,
            name="learned_guided_attention_pooling",
        )(inputs)

    def guide_gate_value(self) -> float:
        layer = self.model.get_layer("learned_guided_attention_pooling")
        return float(layer.guide_gate.numpy())


class AbmilPatchAlltilesScoreModelBuilder(AbmilPatchAlltilesModelBuilder):
    """Añade el logit patch al embedding y deja que ABMIL decida cómo usarlo."""

    model_name = "abmil_patch_alltiles_score"

    def _score_features(self, features):
        guide_logits = layers.TimeDistributed(
            layers.Dense(1, dtype="float32", name="instance_output"),
            name="td_instance_output",
        )(features)
        dropped = self.instance_dropout(features)
        return layers.Concatenate(axis=-1, name="instance_features_with_patch_score")(
            [dropped, guide_logits]
        )

    def encode_instances(self, inputs):
        return self._score_features(self.encode_instance_features(inputs))

    def encode_instances_keras_tiling(self, inputs):
        return self._score_features(self.encode_instance_features_keras_tiling(inputs))
