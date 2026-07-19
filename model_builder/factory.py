from src.modes import is_mil_mode, normalize_mode
from src.model_builder.abmil import AbmilModelBuilder, AbmilPatchHardnegGuidedModelBuilder, AbmilPatchHardnegModelBuilder
from src.model_builder.full import FullModelBuilder
from src.model_builder.patch import PatchHardnegModelBuilder, PatchModelBuilder
from src.model_builder.simple import SimpleModelBuilder

_BUILDERS = {
    "simple": SimpleModelBuilder,
    "full": FullModelBuilder,
    "abmil": AbmilModelBuilder,
    "abmil_patch_hardneg": AbmilPatchHardnegModelBuilder,
    "abmil_patch_hardneg_guided": AbmilPatchHardnegGuidedModelBuilder,
    "patch": PatchModelBuilder,
    "patch_hardneg": PatchHardnegModelBuilder,
}


def create_model_builder(config, IMG_SIZE, backbone, preprocess_input, *, mode="simple", initial_bias=None, focal_alpha=0.90, bag_size=None, pretrained_builder=None, checkpoint_prefix=None, lateralized_inputs=False, steps_per_execution=32, backbone_trainable=False, top_dense=256, dropout=0.4, learning_rate=1e-3, reduce_lr_factor=0.5, min_lr=1e-7, checkpoint_monitor=None, jit_compile=True):
    # Los hiperparametros compartidos salen del CONFIG del notebook; lo especifico
    # de cada corrida (backbone, mode, initial_bias, ...) sigue llegando por argumento.
    general = config["GENERAL"]
    training = config["TRAINING"]
    mil_config = config["MIL"]
    metric_to_maximize = general["METRIC_TO_MAXIMIZE"]
    monitor_mode = "min" if metric_to_maximize == "loss" else "max"

    # elige el builder segun MODE (simple / full / abmil / patch / patch_hardneg)
    mode = normalize_mode(mode)
    common = dict(IMG_SIZE=IMG_SIZE, backbone=backbone, preprocess_input=preprocess_input, backbone_trainable=backbone_trainable, top_dense=top_dense, dropout=dropout, learning_rate=learning_rate, focal_alpha=focal_alpha, focal_gamma=training["FOCAL_GAMMA"], metric_to_maximize=metric_to_maximize, checkpoint_monitor=checkpoint_monitor, monitor_mode=monitor_mode, early_stopping_patience=training["EARLY_STOPPING_PATIENCE"], reduce_lr_patience=training["REDUCE_LR_PATIENCE"], reduce_lr_factor=reduce_lr_factor, min_lr=min_lr, aggressive_augmentation=training["AGGRESSIVE_AUGMENTATION"], initial_bias=initial_bias, pretrained_builder=pretrained_builder, jit_compile=jit_compile, steps_per_execution=steps_per_execution, checkpoint_prefix=checkpoint_prefix, lateralized_inputs=lateralized_inputs)
    if is_mil_mode(mode):
        full_cfg = config.get("FULL") or {}
        mil = dict(
            bag_size=bag_size,
            attention_dim=mil_config["ATTENTION_DIM"],
            attention_gated=mil_config["ATTENTION_GATED"],
            bag_grid=full_cfg.get("BAG_GRID", (3, 3)),
            bag_keras_tiling=mil_config["BAG_KERAS_TILING"],
        )
        if mode == "abmil_patch_hardneg_guided":
            mil.update(
                guided_attention_temperature=mil_config.get("GUIDED_ATTENTION_TEMPERATURE", 1.0),
                guided_attention_strength=mil_config.get("GUIDED_ATTENTION_STRENGTH", 1.0),
            )
        return _BUILDERS[mode](**common, **mil)
    return _BUILDERS[mode](**common)
