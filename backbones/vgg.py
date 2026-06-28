from __future__ import annotations

from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input

DESCRIPTIONS = {
    "vgg16": (
        "VGG-16 (Simonyan & Zisserman, 2014): 13 capas conv + 3 FC, stack de filtros "
        "3×3. Pesos ImageNet; buen extractor generico pero mas pesada que arquitecturas "
        "modernas. Entrada 224×224."
    ),
    "vgg19": (
        "VGG-19: igual familia que VGG-16 con tres bloques conv adicionales. Mas "
        "capacidad y parametros; preentrenada en ImageNet. Entrada 224×224."
    ),
}
