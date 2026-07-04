from src.modes import normalize_mode
from src.model_builder.abmil import AbmilModelBuilder
from src.model_builder.highres import HighresModelBuilder
from src.model_builder.patch import PatchModelBuilder
from src.model_builder.simple import SimpleModelBuilder

_BUILDERS = {
    "simple": SimpleModelBuilder,
    "highres": HighresModelBuilder,
    "abmil": AbmilModelBuilder,
    "patch": PatchModelBuilder,
    "patch_hardneg": PatchModelBuilder,
}


def create_model_builder(IMG_SIZE, backbone, preprocess_input, backbone_trainable=False, top_dense=256, dropout=0.4, learning_rate=1e-3, focal_alpha=0.90, focal_gamma=2.0, metric_to_maximize="pr_auc", checkpoint_monitor=None, monitor_mode="max", early_stopping_patience=8, reduce_lr_patience=4, reduce_lr_factor=0.5, min_lr=1e-7, aggressive_augmentation=False, initial_bias=None, mode="simple", bag_size=None, attention_dim=128, attention_gated=True, bag_grid=(3, 3), bag_keras_tiling=False, pretrained_builder=None, jit_compile=True, steps_per_execution=32):
    # elige el builder segun MODE (simple / highres / abmil / patch / patch_hardneg)
    mode = normalize_mode(mode)
    common = dict(IMG_SIZE=IMG_SIZE, backbone=backbone, preprocess_input=preprocess_input, backbone_trainable=backbone_trainable, top_dense=top_dense, dropout=dropout, learning_rate=learning_rate, focal_alpha=focal_alpha, focal_gamma=focal_gamma, metric_to_maximize=metric_to_maximize, checkpoint_monitor=checkpoint_monitor, monitor_mode=monitor_mode, early_stopping_patience=early_stopping_patience, reduce_lr_patience=reduce_lr_patience, reduce_lr_factor=reduce_lr_factor, min_lr=min_lr, aggressive_augmentation=aggressive_augmentation, initial_bias=initial_bias, pretrained_builder=pretrained_builder, jit_compile=jit_compile, steps_per_execution=steps_per_execution)
    if mode == "abmil":
        mil = dict(bag_size=bag_size, attention_dim=attention_dim, attention_gated=attention_gated, bag_grid=bag_grid, bag_keras_tiling=bag_keras_tiling)
        return _BUILDERS[mode](**common, **mil)
    return _BUILDERS[mode](**common)
