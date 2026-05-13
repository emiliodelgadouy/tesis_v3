from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import (
    DenseNet121,
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7,
    EfficientNetV2B0,
    EfficientNetV2B1,
    EfficientNetV2B2,
    EfficientNetV2B3,
    EfficientNetV2L,
    EfficientNetV2M,
    EfficientNetV2S,
    InceptionResNetV2,
    InceptionV3,
    ResNet50,
    VGG16,
    VGG19,
)
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_v2_preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input


ModelFactory = Callable[..., keras.Model]
PreprocessFunction = Callable
InputSize = tuple[int, int]


def custom_tiny_preprocess_input(x):
    return x


def CustomTinyBackbone(
    weights: str | None = None,
    include_top: bool = False,
    input_shape: tuple[int, int, int] | None = None,
    name: str = "custom_tiny",
    **kwargs,
) -> keras.Model:
    """Backbone minimo para medir overhead de entrenamiento/inferencia."""
    del weights

    inputs = keras.Input(shape=input_shape or (64, 64, 3), name="input")
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)
    x = layers.Conv2D(8, 3, strides=2, padding="same", activation="relu", name="conv_1")(x)
    x = layers.Conv2D(16, 3, strides=2, padding="same", activation="relu", name="conv_2")(x)

    if include_top:
        classes = kwargs.pop("classes", 1000)
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = layers.Dense(classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs, x, name=name)


# ---------------------------------------------------------------------------
# Backbones medicos: RadImageNet (Mei et al., Radiology AI 2022) y CheXNet
# (Rajpurkar et al., port de brucechou1983).
#
# Los pesos oficiales solo se distribuyen via Google Drive, por lo que se
# descargan con gdown y se cachean localmente (override con MEDICAL_WEIGHTS_DIR).
# ---------------------------------------------------------------------------

# Carpeta oficial con los .h5 de RadImageNet en formato TF/Keras.
# https://github.com/BMEII-AI/RadImageNet
RADIMAGENET_FOLDER_ID = "1Es7cK1hv7zNHJoUW0tI0e6nLFVYTqPqK"

# Bundle ZIP oficial con los cuatro modelos TF (~1.8 GB). Los .h5 quedan en
# RadImageNet_models/ al extraer. Se usa si download_folder falla (muy comun sin cookies).
RADIMAGENET_TF_BUNDLE_ID = "1UgYviv2K6QPM1SCexqqab5-yTgwoAFEc"
RADIMAGENET_BUNDLE_ZIP_NAME = "RadImageNet_tensorflow_pretrained_bundle.zip"
# Evita tratar como ZIP valido una pagina HTML de aviso de Google (~2 KB).
MIN_RADIMAGENET_ZIP_BYTES = 100 * 1024 * 1024

# Patrones para localizar cada arquitectura una vez descargada la carpeta.
# El negative-lookahead en inception_v3 evita matchear el archivo de IRV2.
RADIMAGENET_NAME_PATTERNS: dict[str, tuple[str, ...]] = {
    "densenet121": (r"densenet[_-]?121",),
    "resnet50": (r"resnet[_-]?50",),
    "inceptionv3": (r"inception[_-]?v3(?!.*resnet)",),
    "irv2": (r"irv2", r"inception[_-]?resnet[_-]?v2"),
}

# CheXNet (DenseNet121 + Dense(14) sigmoid) sobre NIH ChestX-ray14.
# https://github.com/brucechou1983/CheXNet-Keras
CHEXNET_FILE_ID = "19BllaOvs2x5PLV_vlWMy4i8LapLb2j6b"
CHEXNET_FILE_NAME = "CheXNet_weights.h5"
CHEXNET_NUM_CLASSES = 14


def _medical_weights_dir() -> Path:
    custom = os.environ.get("MEDICAL_WEIGHTS_DIR")
    base = Path(custom) if custom else Path.home() / ".keras" / "medical_weights"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _import_gdown():
    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "Se requiere 'gdown' para descargar pesos pre-entrenados medicos. "
            "Instalalo con: pip install gdown"
        ) from exc
    return gdown


def _list_radimagenet_h5(cache_dir: Path) -> list[Path]:
    return sorted(cache_dir.rglob("*.h5"))


def _ensure_radimagenet_weights() -> Path:
    cache_dir = _medical_weights_dir() / "radimagenet"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if _list_radimagenet_h5(cache_dir):
        return cache_dir

    gdown = _import_gdown()

    # 1) Reintentar extraccion si ya hay un ZIP parcial/corrupto en cache.
    zip_path = cache_dir / RADIMAGENET_BUNDLE_ZIP_NAME
    if zip_path.is_file():
        if zip_path.stat().st_size < MIN_RADIMAGENET_ZIP_BYTES:
            zip_path.unlink(missing_ok=True)
        else:
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(cache_dir)
            except zipfile.BadZipFile:
                zip_path.unlink(missing_ok=True)
            else:
                if _list_radimagenet_h5(cache_dir):
                    try:
                        zip_path.unlink()
                    except OSError:
                        pass
                    return cache_dir

    # 2) Carpeta de Drive (archivos sueltos). Historicamente fallaba sin cookies
    # y desde 2025-2026 la carpeta original devuelve 404, asi que solo se intenta
    # si el usuario fuerza el modo con RADIMAGENET_TRY_FOLDER=1.
    if os.environ.get("RADIMAGENET_TRY_FOLDER", "").lower() in {"1", "true", "yes"}:
        print(
            f"Intentando descargar RadImageNet (carpeta de Google Drive) en {cache_dir} ...\n"
            "Si falla, se usara el bundle ZIP oficial (~1.8 GB, una sola vez)."
        )
        try:
            gdown.download_folder(
                id=RADIMAGENET_FOLDER_ID,
                output=str(cache_dir),
                quiet=False,
                use_cookies=True,
            )
        except Exception as exc:  # gdown.exceptions.DownloadError u otros
            print(
                f"Aviso: fallo la descarga de la carpeta de Drive ({exc!s}). "
                "Continuando con el bundle ZIP oficial."
            )
        if _list_radimagenet_h5(cache_dir):
            return cache_dir

    # 3) Bundle oficial (contiene RadImageNet_models/*.h5).
    bundle_url = f"https://drive.google.com/uc?id={RADIMAGENET_TF_BUNDLE_ID}"
    print(
        "Descargando el bundle ZIP oficial de RadImageNet (~1.8 GB). "
        "Puede tardar varios minutos; al terminar se borra el ZIP para ahorrar espacio."
    )
    try:
        gdown.download(bundle_url, str(zip_path), quiet=False)
    except Exception as exc:
        zip_path.unlink(missing_ok=True)
        raise FileNotFoundError(
            "No se pudieron descargar los pesos RadImageNet desde Google Drive "
            f"({exc!s}).\n"
            "Opciones:\n"
            f"  1) Reintentar mas tarde (Drive limita descargas masivas).\n"
            "  2) Descargar manualmente el ZIP desde "
            f"https://drive.google.com/file/d/{RADIMAGENET_TF_BUNDLE_ID}/view "
            f"y copiarlo a {zip_path} antes de reintentar.\n"
            "  3) Bajar los .h5 individuales (ver https://github.com/BMEII-AI/RadImageNet) "
            f"y colocarlos bajo {cache_dir} (o setear MEDICAL_WEIGHTS_DIR)."
        ) from exc

    if not zip_path.is_file() or zip_path.stat().st_size < MIN_RADIMAGENET_ZIP_BYTES:
        zip_path.unlink(missing_ok=True)
        raise FileNotFoundError(
            "No se pudieron obtener los pesos RadImageNet: el bundle ZIP descargado "
            "esta vacio o es demasiado pequeno (probable aviso HTML de Google Drive por "
            "limite de descargas).\n"
            "Opciones:\n"
            f"  1) Borrar {zip_path} y reintentar mas tarde (Drive limita descargas masivas).\n"
            "  2) Descargar manualmente el ZIP desde "
            f"https://drive.google.com/file/d/{RADIMAGENET_TF_BUNDLE_ID}/view y copiarlo a "
            f"{zip_path} antes de reintentar.\n"
            "  3) Bajar los .h5 individuales desde https://github.com/BMEII-AI/RadImageNet "
            f"y colocarlos bajo {cache_dir} (o setear MEDICAL_WEIGHTS_DIR y copiar a "
            f"…/radimagenet/).\n"
            "  4) En Colab, montar Drive y exportar MEDICAL_WEIGHTS_DIR a una ruta persistente."
        )

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cache_dir)
    try:
        zip_path.unlink()
    except OSError:
        pass

    if not _list_radimagenet_h5(cache_dir):
        raise FileNotFoundError(
            f"No se encontraron archivos .h5 despues de extraer el bundle en {cache_dir}."
        )
    return cache_dir


def _radimagenet_weights_path(model_key: str) -> Path:
    cache_dir = _ensure_radimagenet_weights()
    h5_files = _list_radimagenet_h5(cache_dir)
    patterns = [re.compile(p, re.IGNORECASE) for p in RADIMAGENET_NAME_PATTERNS[model_key]]
    for h5_file in h5_files:
        if any(pattern.search(h5_file.name) for pattern in patterns):
            return h5_file
    all_files = sorted(p for p in cache_dir.rglob("*") if p.is_file())
    sample = [str(p.relative_to(cache_dir)) for p in all_files[:50]]
    extra = f"\n... y {len(all_files) - 50} archivos mas." if len(all_files) > 50 else ""
    raise FileNotFoundError(
        f"No se encontro archivo de pesos de RadImageNet para '{model_key}' en {cache_dir}.\n"
        f"Archivos .h5 detectados: {[f.name for f in h5_files]}\n"
        f"Archivos en cache (muestra): {sample}{extra}"
    )


def _ensure_chexnet_weights() -> Path:
    cache_dir = _medical_weights_dir() / "chexnet"
    cache_dir.mkdir(parents=True, exist_ok=True)
    weights_path = cache_dir / CHEXNET_FILE_NAME
    if weights_path.is_file():
        return weights_path

    gdown = _import_gdown()
    print(f"Descargando pesos de CheXNet a {weights_path} (~30 MB) ...")
    gdown.download(
        id=CHEXNET_FILE_ID,
        output=str(weights_path),
        quiet=False,
    )
    if not weights_path.is_file():
        raise FileNotFoundError(f"No se pudo descargar CheXNet a {weights_path}")
    return weights_path


def radimagenet_preprocess_input(x):
    """Preprocesamiento usado en el sample code oficial de RadImageNet (rescale 1/255)."""
    return x / 255.0


def chexnet_preprocess_input(x):
    return densenet_preprocess_input(x)


def _build_radimagenet_backbone(
    base_model_fn: ModelFactory,
    model_key: str,
    weights,
    include_top: bool,
    input_shape: tuple[int, int, int] | None,
    name: str,
    **kwargs,
) -> keras.Model:
    if include_top:
        raise ValueError(
            "Los pesos RadImageNet solo soportan include_top=False (sin cabeza)."
        )

    if weights == "radimagenet":
        weights_path = _radimagenet_weights_path(model_key)
        # Los .h5 oficiales son archivos *_notop.h5 guardados con Keras 2.4 que
        # incluyen arquitectura + pesos y ya no tienen cabeza de clasificacion;
        # el input esta fijado a (None, 224, 224, 3). Cargarlos directamente con
        # load_model preserva todos los pesos exactamente (incluyendo BN).
        saved = keras.models.load_model(str(weights_path), compile=False)

        inputs = saved.inputs[0] if saved.inputs else saved.input
        output_tensor = saved.outputs[0] if saved.outputs else saved.output

        if len(output_tensor.shape) != 4:
            last_spatial_layer = None
            for layer in reversed(saved.layers):
                try:
                    layer_out = layer.output
                    if isinstance(layer_out, list):
                        layer_out = layer_out[0]
                    if len(layer_out.shape) == 4:
                        last_spatial_layer = layer
                        break
                except Exception:
                    continue
            if last_spatial_layer is None:
                raise ValueError(
                    f"No se encontro ninguna capa espacial (4D) en el modelo RadImageNet "
                    f"cargado desde {weights_path}. Verifica que el .h5 es correcto."
                )
            output_tensor = last_spatial_layer.output

        base_model = keras.Model(inputs=inputs, outputs=output_tensor, name=name)
        print(
            f"RadImageNet '{model_key}': backbone cargado desde '{weights_path.name}', "
            f"{len(base_model.layers)} capas, input {tuple(inputs.shape)}, "
            f"output {tuple(output_tensor.shape)}."
        )
    elif weights == "imagenet":
        base_model = base_model_fn(
            weights="imagenet", include_top=False, input_shape=input_shape, **kwargs
        )
    elif weights is None:
        base_model = base_model_fn(
            weights=None, include_top=False, input_shape=input_shape, **kwargs
        )
    else:
        base_model = base_model_fn(
            weights=None, include_top=False, input_shape=input_shape, **kwargs
        )
        base_model.load_weights(str(weights))

    base_model._name = name
    return base_model


def RadImageNetDenseNet121(
    weights="radimagenet",
    include_top: bool = False,
    input_shape: tuple[int, int, int] | None = None,
    name: str = "radimagenet_densenet121",
    **kwargs,
) -> keras.Model:
    return _build_radimagenet_backbone(
        DenseNet121, "densenet121", weights, include_top, input_shape, name, **kwargs
    )


def RadImageNetResNet50(
    weights="radimagenet",
    include_top: bool = False,
    input_shape: tuple[int, int, int] | None = None,
    name: str = "radimagenet_resnet50",
    **kwargs,
) -> keras.Model:
    return _build_radimagenet_backbone(
        ResNet50, "resnet50", weights, include_top, input_shape, name, **kwargs
    )


def RadImageNetInceptionV3(
    weights="radimagenet",
    include_top: bool = False,
    input_shape: tuple[int, int, int] | None = None,
    name: str = "radimagenet_inceptionv3",
    **kwargs,
) -> keras.Model:
    return _build_radimagenet_backbone(
        InceptionV3, "inceptionv3", weights, include_top, input_shape, name, **kwargs
    )


def RadImageNetInceptionResNetV2(
    weights="radimagenet",
    include_top: bool = False,
    input_shape: tuple[int, int, int] | None = None,
    name: str = "radimagenet_inceptionresnetv2",
    **kwargs,
) -> keras.Model:
    return _build_radimagenet_backbone(
        InceptionResNetV2, "irv2", weights, include_top, input_shape, name, **kwargs
    )


def CheXNetDenseNet121(
    weights="chexnet",
    include_top: bool = False,
    input_shape: tuple[int, int, int] | None = None,
    name: str = "chexnet_densenet121",
    **kwargs,
) -> keras.Model:
    if include_top:
        raise ValueError(
            "CheXNet solo soporta include_top=False (la cabeza original es de 14 clases)."
        )

    backbone = DenseNet121(
        weights=None, include_top=False, input_shape=input_shape, **kwargs
    )

    if weights == "chexnet":
        # El .h5 publico contiene DenseNet121 + GlobalAveragePooling2D + Dense(14) sigmoid.
        # Reconstruimos esa arquitectura, cargamos pesos, y nos quedamos solo con el backbone.
        weights_path = _ensure_chexnet_weights()
        gap = layers.GlobalAveragePooling2D(name="avg_pool")(backbone.output)
        predictions = layers.Dense(
            CHEXNET_NUM_CLASSES, activation="sigmoid", name="predictions"
        )(gap)
        full_model = keras.Model(
            inputs=backbone.input, outputs=predictions, name="chexnet_full"
        )
        full_model.load_weights(str(weights_path))
    elif weights == "imagenet":
        imagenet_backbone = DenseNet121(
            weights="imagenet", include_top=False, input_shape=input_shape, **kwargs
        )
        backbone.set_weights(imagenet_backbone.get_weights())
    elif weights is None:
        pass
    else:
        backbone.load_weights(str(weights))

    backbone._name = name
    return backbone


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BackboneConfig:
    name: str
    model_fn: ModelFactory
    preprocess_input: PreprocessFunction
    input_size: InputSize
    default_weights: str | None = "imagenet"


BACKBONES: dict[str, BackboneConfig] = {
    "customtiny": BackboneConfig(
        "custom_tiny", CustomTinyBackbone, custom_tiny_preprocess_input, (64, 64),
        default_weights=None,
    ),
    "vgg16": BackboneConfig("vgg16", VGG16, vgg16_preprocess_input, (224, 224)),
    "vgg19": BackboneConfig("vgg19", VGG19, vgg19_preprocess_input, (224, 224)),
    "efficientnetb0": BackboneConfig("efficientnetb0", EfficientNetB0, efficientnet_preprocess_input, (224, 224)),
    "efficientnetb1": BackboneConfig("efficientnetb1", EfficientNetB1, efficientnet_preprocess_input, (240, 240)),
    "efficientnetb2": BackboneConfig("efficientnetb2", EfficientNetB2, efficientnet_preprocess_input, (260, 260)),
    "efficientnetb3": BackboneConfig("efficientnetb3", EfficientNetB3, efficientnet_preprocess_input, (300, 300)),
    "efficientnetb4": BackboneConfig("efficientnetb4", EfficientNetB4, efficientnet_preprocess_input, (380, 380)),
    "efficientnetb5": BackboneConfig("efficientnetb5", EfficientNetB5, efficientnet_preprocess_input, (456, 456)),
    "efficientnetb6": BackboneConfig("efficientnetb6", EfficientNetB6, efficientnet_preprocess_input, (528, 528)),
    "efficientnetb7": BackboneConfig("efficientnetb7", EfficientNetB7, efficientnet_preprocess_input, (600, 600)),
    "efficientnetv2b0": BackboneConfig("efficientnetv2b0", EfficientNetV2B0, efficientnet_v2_preprocess_input, (224, 224)),
    "efficientnetv2b1": BackboneConfig("efficientnetv2b1", EfficientNetV2B1, efficientnet_v2_preprocess_input, (240, 240)),
    "efficientnetv2b2": BackboneConfig("efficientnetv2b2", EfficientNetV2B2, efficientnet_v2_preprocess_input, (260, 260)),
    "efficientnetv2b3": BackboneConfig("efficientnetv2b3", EfficientNetV2B3, efficientnet_v2_preprocess_input, (300, 300)),
    "efficientnetv2s": BackboneConfig("efficientnetv2s", EfficientNetV2S, efficientnet_v2_preprocess_input, (384, 384)),
    "efficientnetv2m": BackboneConfig("efficientnetv2m", EfficientNetV2M, efficientnet_v2_preprocess_input, (480, 480)),
    "efficientnetv2l": BackboneConfig("efficientnetv2l", EfficientNetV2L, efficientnet_v2_preprocess_input, (480, 480)),
    "radimagenetdensenet121": BackboneConfig(
        "radimagenet_densenet121",
        RadImageNetDenseNet121,
        radimagenet_preprocess_input,
        (224, 224),
        default_weights="radimagenet",
    ),
    "radimagenetresnet50": BackboneConfig(
        "radimagenet_resnet50",
        RadImageNetResNet50,
        radimagenet_preprocess_input,
        (224, 224),
        default_weights="radimagenet",
    ),
    "radimagenetinceptionv3": BackboneConfig(
        "radimagenet_inceptionv3",
        RadImageNetInceptionV3,
        radimagenet_preprocess_input,
        # Los pesos oficiales se entrenaron con input 224x224 (no 299x299).
        (224, 224),
        default_weights="radimagenet",
    ),
    "radimagenetinceptionresnetv2": BackboneConfig(
        "radimagenet_inceptionresnetv2",
        RadImageNetInceptionResNetV2,
        radimagenet_preprocess_input,
        # Los pesos oficiales se entrenaron con input 224x224 (no 299x299).
        (224, 224),
        default_weights="radimagenet",
    ),
    "chexnet": BackboneConfig(
        "chexnet_densenet121",
        CheXNetDenseNet121,
        chexnet_preprocess_input,
        (224, 224),
        default_weights="chexnet",
    ),
}


_DEFAULT_WEIGHTS = object()


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "")


def available_backbones() -> tuple[str, ...]:
    return tuple(BACKBONES)


def get_backbone_config(name: str) -> BackboneConfig:
    key = _normalize_name(name)
    if key not in BACKBONES:
        available = ", ".join(available_backbones())
        raise ValueError(f"Backbone '{name}' is not available. Options: {available}")
    return BACKBONES[key]


def get_preprocess_input(name: str) -> PreprocessFunction:
    return get_backbone_config(name).preprocess_input


def get_input_size(name: str) -> InputSize:
    return get_backbone_config(name).input_size


def build_backbone(
    name: str,
    weights=_DEFAULT_WEIGHTS,
    include_top: bool = False,
    input_shape: tuple[int, int, int] | None = None,
    **kwargs,
) -> keras.Model:
    config = get_backbone_config(name)
    if weights is _DEFAULT_WEIGHTS:
        weights = config.default_weights
    height, width = config.input_size
    return config.model_fn(
        weights=weights,
        include_top=include_top,
        input_shape=input_shape or (height, width, 3),
        name=config.name,
        **kwargs,
    )


def resolve_backbone(backbone_name, input_size: InputSize | None = None):
    if input_size is None:
        input_size = get_input_size(backbone_name)
    height, width = input_size
    backbone = build_backbone(backbone_name, input_shape=(height, width, 3))
    preprocess_input = get_preprocess_input(backbone_name)
    return backbone, preprocess_input, input_size
