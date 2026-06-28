from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from tensorflow import keras
from tensorflow.keras.applications import (
    DenseNet121,
    InceptionResNetV2,
    InceptionV3,
    ResNet50,
)

from ._types import InputSize, ModelFactory
from .base import Backbone, DEFAULT_WEIGHTS
from .medical_weights import (
    extract_zip_bundle,
    import_gdown,
    medical_weights_dir,
    transfer_legacy_weights,
)

RADIMAGENET_FOLDER_ID = "1Es7cK1hv7zNHJoUW0tI0e6nLFVYTqPqK"
RADIMAGENET_TF_BUNDLE_ID = "1UgYviv2K6QPM1SCexqqab5-yTgwoAFEc"
RADIMAGENET_BUNDLE_ZIP_NAME = "RadImageNet_tensorflow_pretrained_bundle.zip"
MIN_RADIMAGENET_ZIP_BYTES = 100 * 1024 * 1024

RADIMAGENET_NAME_PATTERNS: dict[str, tuple[str, ...]] = {
    "densenet121": (r"densenet[_-]?121",),
    "resnet50": (r"resnet[_-]?50",),
    "inceptionv3": (r"inception[_-]?v3(?!.*resnet)",),
    "irv2": (r"irv2", r"inception[_-]?resnet[_-]?v2"),
}

_DESCRIPTIONS = {
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
}


def _radimagenet_preprocess_input(x):
    from tensorflow.keras.applications.imagenet_utils import preprocess_input as imagenet_preprocess

    x = imagenet_preprocess(x, mode="caffe")
    return x / 255.0


def _list_radimagenet_h5(cache_dir: Path) -> list[Path]:
    return sorted(cache_dir.rglob("*.h5"))


def _ensure_radimagenet_weights() -> Path:
    cache_dir = medical_weights_dir() / "radimagenet"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if _list_radimagenet_h5(cache_dir):
        return cache_dir

    gdown = import_gdown()

    zip_path = cache_dir / RADIMAGENET_BUNDLE_ZIP_NAME
    if extract_zip_bundle(zip_path, cache_dir, min_bytes=MIN_RADIMAGENET_ZIP_BYTES):
        if _list_radimagenet_h5(cache_dir):
            try:
                zip_path.unlink()
            except OSError:
                pass
            return cache_dir

    if os.environ.get("RADIMAGENET_TRY_FOLDER", "").lower() in {"1", "true", "yes"}:
        try:
            gdown.download_folder(
                id=RADIMAGENET_FOLDER_ID,
                output=str(cache_dir),
                quiet=False,
                use_cookies=True,
            )
        except Exception as exc:
            print(
                f"Aviso: fallo la descarga de la carpeta de Drive ({exc!s}). "
                "Continuando con el bundle ZIP oficial."
            )
        if _list_radimagenet_h5(cache_dir):
            return cache_dir

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


@dataclass(frozen=True)
class RadImageNetBackbone(Backbone):
    """Backbone Keras con pesos RadImageNet transferidos desde .h5 legacy."""

    key: str
    keras_name: str
    model_key: str
    base_model_fn: ModelFactory
    description: str
    input_size: InputSize = (224, 224)
    default_weights: str | None = "radimagenet"

    def preprocess_input(self, x):
        return _radimagenet_preprocess_input(x)

    def build(
        self,
        *,
        weights=DEFAULT_WEIGHTS,
        include_top: bool = False,
        input_shape: tuple[int, int, int] | None = None,
        **kwargs,
    ) -> keras.Model:
        weights = self._resolve_weights(weights)
        input_shape = self._default_input_shape(input_shape)

        if include_top:
            raise ValueError(
                "Los pesos RadImageNet solo soportan include_top=False (sin cabeza)."
            )

        if weights == "radimagenet":
            weights_path = _radimagenet_weights_path(self.model_key)
            base_model = self.base_model_fn(
                weights=None, include_top=False, input_shape=input_shape, **kwargs
            )
            transfer_legacy_weights(base_model, weights_path)
        elif weights == "imagenet":
            base_model = self.base_model_fn(
                weights="imagenet", include_top=False, input_shape=input_shape, **kwargs
            )
        elif weights is None:
            base_model = self.base_model_fn(
                weights=None, include_top=False, input_shape=input_shape, **kwargs
            )
        else:
            base_model = self.base_model_fn(
                weights=None, include_top=False, input_shape=input_shape, **kwargs
            )
            base_model.load_weights(str(weights))

        base_model._name = self.keras_name
        return base_model


_VARIANTS: tuple[tuple[str, str, str, ModelFactory], ...] = (
    ("radimagenetdensenet121", "radimagenet_densenet121", "densenet121", DenseNet121),
    ("radimagenetresnet50", "radimagenet_resnet50", "resnet50", ResNet50),
    ("radimagenetinceptionv3", "radimagenet_inceptionv3", "inceptionv3", InceptionV3),
    (
        "radimagenetinceptionresnetv2",
        "radimagenet_inceptionresnetv2",
        "irv2",
        InceptionResNetV2,
    ),
)

BACKBONES: tuple[Backbone, ...] = tuple(
    RadImageNetBackbone(
        key=key,
        keras_name=keras_name,
        model_key=model_key,
        base_model_fn=model_fn,
        description=_DESCRIPTIONS[key],
    )
    for key, keras_name, model_key, model_fn in _VARIANTS
)
