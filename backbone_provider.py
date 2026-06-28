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


def _conv_bn_relu(
    x,
    filters: int,
    kernel_size: int,
    *,
    strides: int = 1,
    name: str,
) -> keras.KerasTensor:
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    return x


def _residual_block(
    x,
    filters: int,
    name: str,
    *,
    strides: int = 1,
    dropout: float = 0.0,
) -> keras.KerasTensor:
    shortcut = x

    x = _conv_bn_relu(x, filters, 3, strides=strides, name=f"{name}_a")
    x = layers.Conv2D(
        filters, 3, padding="same", use_bias=False, name=f"{name}_b_conv"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_b_bn")(x)

    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters,
            1,
            strides=strides,
            padding="same",
            use_bias=False,
            name=f"{name}_proj_conv",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name=f"{name}_drop")(x)
    return x


def _residual_stage(
    x,
    filters: int,
    blocks: int,
    name: str,
    *,
    first_stride: int = 1,
    dropout: float = 0.15,
) -> keras.KerasTensor:
    x = _residual_block(
        x, filters, f"{name}_block0", strides=first_stride, dropout=dropout
    )
    for block_idx in range(1, blocks):
        x = _residual_block(
            x, filters, f"{name}_block{block_idx}", dropout=dropout
        )
    return x


def CustomCnnBackbone(
    weights: str | None = None,
    include_top: bool = False,
    input_shape: tuple[int, int, int] | None = None,
    name: str = "custom_cnn",
    **kwargs,
) -> keras.Model:
    """ResNet-18 desde cero (sin transfer learning).

    Baseline para el dataset completo (~20k imagenes filtradas, ~1.8k positivos):
    stem convolucional, cuatro etapas con bloques residuales (2 por etapa) y
    canales 64-128-256-512 (~11M parametros). Escala adecuada frente a
    transfer learning sin ser tan pesada como ResNet-50+. La cabeza clasificadora
    la agrega ModelBuilder.
    """
    if weights not in (None, "none"):
        raise ValueError(
            f"CustomCnnBackbone no usa pesos preentrenados (recibido weights={weights!r})."
        )

    inputs = keras.Input(shape=input_shape or (224, 224, 3), name="input")
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)

    # Stem ResNet: 224 -> 112 -> 56
    x = _conv_bn_relu(x, 64, 7, strides=2, name="stem")
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="stem_pool")(x)

    # 56 -> 28 -> 14 -> 7
    x = _residual_stage(x, 64, 2, "stage1", dropout=0.05)
    x = _residual_stage(x, 128, 2, "stage2", first_stride=2, dropout=0.10)
    x = _residual_stage(x, 256, 2, "stage3", first_stride=2, dropout=0.15)
    x = _residual_stage(x, 512, 2, "stage4", first_stride=2, dropout=0.20)

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


def _load_legacy_h5_model(weights_path: Path):
    """Carga un modelo Keras guardado en formato legacy (.h5 de Keras 2).

    Keras 3 rechaza nombres de capa con '/' (p. ej. 'conv1/conv') al deserializar
    estos .h5 antiguos. ``tf_keras`` (el paquete de compatibilidad Keras 2) los
    carga sin problema; lo instalamos al vuelo si no esta disponible.
    """
    try:
        import tf_keras  # type: ignore
    except ImportError:
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "tf-keras"],
            check=True,
        )
        import tf_keras  # type: ignore

    return tf_keras.models.load_model(str(weights_path), compile=False)


def _transfer_legacy_weights(target_model: keras.Model, weights_path: Path) -> None:
    """Transfiere por posicion los pesos de un .h5 legacy a un modelo Keras 3.

    Construimos la arquitectura limpia en Keras 3 (nombres validos) y copiamos
    los pesos del modelo legacy en el mismo orden topologico. Las arquitecturas
    son identicas (misma familia de ``keras.applications``), por lo que el orden
    y las formas coinciden; validamos ambas cosas para fallar de forma explicita.
    """
    legacy_model = _load_legacy_h5_model(weights_path)
    source_weights = legacy_model.get_weights()
    target_weights = target_model.get_weights()

    if len(source_weights) != len(target_weights):
        raise ValueError(
            "No se pueden transferir los pesos legacy: el modelo guardado tiene "
            f"{len(source_weights)} tensores y el modelo destino {len(target_weights)}. "
            "Verifica que la arquitectura coincide."
        )
    for index, (src, dst) in enumerate(zip(source_weights, target_weights)):
        if src.shape != dst.shape:
            raise ValueError(
                f"Forma incompatible al transferir el peso {index}: "
                f"{src.shape} (legacy) vs {dst.shape} (destino)."
            )
    target_model.set_weights(source_weights)


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
    gdown.download(
        id=CHEXNET_FILE_ID,
        output=str(weights_path),
        quiet=False,
    )
    if not weights_path.is_file():
        raise FileNotFoundError(f"No se pudo descargar CheXNet a {weights_path}")
    return weights_path


def radimagenet_preprocess_input(x):
    """Preprocesamiento fiel al entrenamiento TF oficial de RadImageNet.

    El codigo TF de RadImageNet uso ``ImageDataGenerator(preprocessing_function=
    preprocess_input, rescale=1/255)``: primero ``preprocess_input`` en modo
    ``caffe`` (RGB->BGR y resta de la media de ImageNet, sin escalar) y LUEGO
    ``/255``. Es decir ``(BGR - [103.94, 116.78, 123.68]) / 255`` (rango
    ~[-0.49, 0.59]). Usar solo ``x/255`` ([0,1]) NO coincide con la distribucion
    de entrada esperada por los pesos y degrada fuertemente las features.
    """
    from tensorflow.keras.applications.imagenet_utils import preprocess_input

    x = preprocess_input(x, mode="caffe")
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
        # Los .h5 oficiales son *_notop.h5 guardados con Keras 2.4 (arquitectura +
        # pesos, sin cabeza). Keras 3 no puede deserializar su arquitectura porque
        # las capas usan nombres con '/' (p. ej. 'conv1/conv'). Reconstruimos la
        # arquitectura limpia en Keras 3 y transferimos los pesos por posicion
        # desde el modelo legacy (cargado con tf_keras), preservando BN incluido.
        base_model = base_model_fn(
            weights=None, include_top=False, input_shape=input_shape, **kwargs
        )
        _transfer_legacy_weights(base_model, weights_path)
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
# Descripciones para documentacion y logging (p. ej. Comet)
# ---------------------------------------------------------------------------

ARCHITECTURE_DESCRIPTIONS: dict[str, str] = {
    "customtiny": (
        "CNN minima (2 capas conv + ReLU, entrada 64×64) entrenada desde cero. "
        "Sirve como baseline de overhead del pipeline y no usa transfer learning."
    ),
    "customcnn": (
        "CNN compacta entrenada desde cero (4 bloques conv 3×3 + BN + ReLU + pooling, "
        "entrada 224×224). Pensada para datasets medicos pequenos con regularizacion "
        "fuerte (dropout) y sin pesos preentrenados."
    ),
    "vgg16": (
        "VGG-16 (Simonyan & Zisserman, 2014): 13 capas conv + 3 FC, stack de filtros "
        "3×3. Pesos ImageNet; buen extractor generico pero mas pesada que arquitecturas "
        "modernas. Entrada 224×224."
    ),
    "vgg19": (
        "VGG-19: igual familia que VGG-16 con tres bloques conv adicionales. Mas "
        "capacidad y parametros; preentrenada en ImageNet. Entrada 224×224."
    ),
    "efficientnetb0": (
        "EfficientNet-B0 (Tan & Le, 2019): escalado compuesto de profundidad, ancho y "
        "resolucion; Mobile Inverted Bottleneck (MBConv). La variante mas liviana de la "
        "familia B. ImageNet, entrada 224×224."
    ),
    "efficientnetb1": "EfficientNet-B1: mayor ancho y resolucion que B0. ImageNet, entrada 240×240.",
    "efficientnetb2": "EfficientNet-B2: capacidad intermedia. ImageNet, entrada 260×260.",
    "efficientnetb3": "EfficientNet-B3: equilibrio frecuente entre costo y rendimiento. ImageNet, entrada 300×300.",
    "efficientnetb4": "EfficientNet-B4: modelo grande de la serie B. ImageNet, entrada 380×380.",
    "efficientnetb5": "EfficientNet-B5. ImageNet, entrada 456×456.",
    "efficientnetb6": "EfficientNet-B6. ImageNet, entrada 528×528.",
    "efficientnetb7": "EfficientNet-B7: maxima escala de la serie original. ImageNet, entrada 600×600.",
    "efficientnetv2b0": (
        "EfficientNetV2-B0 (Tan & Le, 2021): bloques Fused-MBConv y entrenamiento progresivo; "
        "mejor trade-off velocidad/precision que EfficientNet v1. ImageNet, entrada 224×224."
    ),
    "efficientnetv2b1": "EfficientNetV2-B1. ImageNet, entrada 240×240.",
    "efficientnetv2b2": "EfficientNetV2-B2. ImageNet, entrada 260×260.",
    "efficientnetv2b3": "EfficientNetV2-B3. ImageNet, entrada 300×300.",
    "efficientnetv2s": "EfficientNetV2-S (small). ImageNet, entrada 384×384.",
    "efficientnetv2m": "EfficientNetV2-M (medium). ImageNet, entrada 480×480.",
    "efficientnetv2l": "EfficientNetV2-L (large). ImageNet, entrada 480×480.",
    "radimagenetdensenet121": (
        "DenseNet-121 preentrenada en RadImageNet (Mei et al., Radiology AI 2022): ~1,35 M "
        "imagenes de 7 modalidades (CT, MRI, US, etc.). Transfer learning orientado a "
        "imagen medica; entrada 224×224, preprocesamiento rescale 1/255."
    ),
    "radimagenetresnet50": (
        "ResNet-50 preentrenada en RadImageNet. Residual connections; buen compromiso "
        "entre profundidad y entrenabilidad en dominio medico. Entrada 224×224."
    ),
    "radimagenetinceptionv3": (
        "InceptionV3 preentrenada en RadImageNet (pesos oficiales a 224×224, no 299×299). "
        "Modulos inception multi-escala. Dominio medico."
    ),
    "radimagenetinceptionresnetv2": (
        "Inception-ResNet-v2 preentrenada en RadImageNet. Combina bloques inception con "
        "conexiones residuales; alta capacidad. Entrada 224×224."
    ),
    "chexnet": (
        "DenseNet-121 preentrenada con CheXNet (Rajpurkar et al.) sobre NIH ChestX-ray14 "
        "(14 patologias toracicas, cabeza multietiqueta descartada). Especializada en "
        "radiografia de torax; preprocesamiento DenseNet. Entrada 224×224."
    ),
}


def get_backbone_description(name: str) -> str:
    """Texto descriptivo de una arquitectura registrada."""
    key = _normalize_name(name)
    config = get_backbone_config(name)
    body = ARCHITECTURE_DESCRIPTIONS.get(
        key,
        f"Backbone '{name}' registrado sin descripcion detallada.",
    )
    h, w = config.input_size
    weights = config.default_weights or "none (desde cero)"
    return (
        f"Arquitectura: {key}\n"
        f"Entrada: {h}×{w}×3\n"
        f"Pesos por defecto: {weights}\n\n"
        f"{body}"
    )


def format_all_architecture_descriptions() -> str:
    """Catalogo de todas las arquitecturas disponibles (para informes / Comet)."""
    lines = [
        "Catalogo de backbones disponibles",
        "=" * 40,
        "",
    ]
    for key in available_backbones():
        lines.append(get_backbone_description(key))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass
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
    "customcnn": BackboneConfig(
        "custom_cnn",
        CustomCnnBackbone,
        custom_tiny_preprocess_input,
        (224, 224),
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
    # NB: no forzamos `name=config.name`. Las aplicaciones EfficientNetV2 de Keras
    # reutilizan el argumento `name` como clave de DEFAULT_BLOCKS_ARGS (espera
    # "efficientnetv2-b0"), por lo que un nombre normalizado dispara un KeyError.
    # Los backbones propios ya usan por defecto exactamente config.name.
    return config.model_fn(
        weights=weights,
        include_top=include_top,
        input_shape=input_shape or (height, width, 3),
        **kwargs,
    )


def resolve_backbone(backbone_name, input_size: InputSize | None = None):
    if input_size is None:
        input_size = get_input_size(backbone_name)
    height, width = input_size
    backbone = build_backbone(backbone_name, input_shape=(height, width, 3))
    preprocess_input = get_preprocess_input(backbone_name)
    return backbone, preprocess_input, input_size
