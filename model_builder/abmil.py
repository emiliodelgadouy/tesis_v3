from src.model_builder.layers import GatedAttentionPooling
from src.model_builder.mil_base import MilModelBuilderBase


class AbmilModelBuilder(MilModelBuilderBase):
    model_name = "abmil"

    def pool_instances(self, x):
        # atencion gated sobre instancias (Ilse et al.)
        return GatedAttentionPooling(attention_dim=self.attention_dim, gated=self.attention_gated, name="attention_pooling")(x)

    def build_keras_tiling(self):
        self.model_name = "abmil_keras_tiling"
        return super().build_keras_tiling()
