import tensorflow as tf

from src.backbones import resolve_backbone
from src.model_builder.factory import create_model_builder


def _minimal_config():
    return {
        "GENERAL": {"METRIC_TO_MAXIMIZE": "auc", "BATCH_SIZE": 2},
        "TRAINING": {
            "FOCAL_GAMMA": 2.0,
            "AGGRESSIVE_AUGMENTATION": False,
            "EARLY_STOPPING_PATIENCE": 4,
            "REDUCE_LR_PATIENCE": 2,
        },
        "MIL": {
            "ATTENTION_DIM": 128,
            "ATTENTION_GATED": True,
            "BAG_KERAS_TILING": True,
        },
    }


def test_vitb16_outputs_spatial_features_for_gap():
    backbone, preprocess, size = resolve_backbone("vitb16", input_size=(224, 224))
    x = tf.random.uniform((2, size[0], size[1], 3), maxval=255.0)
    features = backbone(preprocess(x), training=False)
    assert features.shape.rank == 4
    assert tuple(features.shape[1:3]) == (14, 14)
    assert int(features.shape[-1]) == 768


def test_vitb16_builds_simple_model():
    backbone, preprocess, size = resolve_backbone("vitb16", input_size=(224, 224))
    builder = create_model_builder(
        _minimal_config(),
        size,
        backbone,
        preprocess,
        mode="simple",
        jit_compile=False,
    )
    builder.build()
    logits = builder.model(tf.zeros((2, size[0], size[1], 3)), training=False)
    assert tuple(logits.shape) == (2, 1)
