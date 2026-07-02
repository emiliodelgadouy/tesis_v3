from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import ClassVar

import gdown
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121, InceptionResNetV2, InceptionV3, ResNet50

from .base import Backbone, DEFAULT_WEIGHTS, InputSize, ModelFactory
from .medical_weights import medical_weights_dir, transfer_legacy_weights

BUNDLE_ID = "1UgYviv2K6QPM1SCexqqab5-yTgwoAFEc"
BUNDLE_ZIP = "RadImageNet_tensorflow_pretrained_bundle.zip"

WEIGHT_PATTERNS: dict[str, tuple[str, ...]] = {
    "densenet121": (r"densenet[_-]?121",),
    "resnet50": (r"resnet[_-]?50",),
    "inceptionv3": (r"inception[_-]?v3(?!.*resnet)",),
    "irv2": (r"irv2", r"inception[_-]?resnet[_-]?v2"),
}


def preprocess(x):
    from tensorflow.keras.applications.imagenet_utils import preprocess_input as imagenet_preprocess

    return imagenet_preprocess(x, mode="caffe") / 255.0


def list_h5_files(cache_dir: Path) -> list[Path]:
    return sorted(cache_dir.rglob("*.h5"))


def ensure_cache() -> Path:
    cache_dir = medical_weights_dir() / "radimagenet"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if list_h5_files(cache_dir):
        return cache_dir
    zip_path = cache_dir / BUNDLE_ZIP
    gdown.download(f"https://drive.google.com/uc?id={BUNDLE_ID}", str(zip_path), quiet=False)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cache_dir)
    zip_path.unlink(missing_ok=True)
    return cache_dir


def weights_path(model_key: str) -> Path:
    cache_dir = ensure_cache()
    patterns = [re.compile(p, re.IGNORECASE) for p in WEIGHT_PATTERNS[model_key]]
    for h5_file in list_h5_files(cache_dir):
        if any(p.search(h5_file.name) for p in patterns):
            return h5_file
    raise FileNotFoundError(model_key)


class RadImageNetBackbone(Backbone):
    model_key: ClassVar[str]
    base_model_fn: ClassVar[ModelFactory]
    input_size: ClassVar[InputSize] = (224, 224)
    default_weights = "radimagenet"

    def preprocess_input(self, x):
        return preprocess(x)

    def build(self, *, weights=DEFAULT_WEIGHTS, include_top: bool = False, input_shape: tuple[int, int, int] | None = None, **kwargs) -> keras.Model:
        weights = self.coalesce_weights(weights)
        shape = self.input_shape_or_default(input_shape)
        base_model_fn = self.__class__.base_model_fn
        if weights == "radimagenet":
            model = base_model_fn(weights=None, include_top=False, input_shape=shape, **kwargs)
            transfer_legacy_weights(model, weights_path(self.__class__.model_key))
        elif weights == "imagenet":
            model = base_model_fn(weights="imagenet", include_top=False, input_shape=shape, **kwargs)
        else:
            model = base_model_fn(weights=None, include_top=False, input_shape=shape, **kwargs)
            if weights is not None:
                model.load_weights(str(weights))
        model._name = self.key
        return model


class RadImageNetDenseNet121Backbone(RadImageNetBackbone):
    key = "radimagenetdensenet121"
    model_key = "densenet121"
    base_model_fn = DenseNet121


class RadImageNetResNet50Backbone(RadImageNetBackbone):
    key = "radimagenetresnet50"
    model_key = "resnet50"
    base_model_fn = ResNet50


class RadImageNetInceptionV3Backbone(RadImageNetBackbone):
    key = "radimagenetinceptionv3"
    model_key = "inceptionv3"
    base_model_fn = InceptionV3


class RadImageNetInceptionResNetV2Backbone(RadImageNetBackbone):
    key = "radimagenetinceptionresnetv2"
    model_key = "irv2"
    base_model_fn = InceptionResNetV2
