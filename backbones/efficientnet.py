from __future__ import annotations

from tensorflow.keras.applications import (
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7,
)
from tensorflow.keras.applications.efficientnet import preprocess_input

DESCRIPTIONS = {
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
}
