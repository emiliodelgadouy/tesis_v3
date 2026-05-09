from __future__ import annotations
import os
import random
from pathlib import Path
from shutil import copy2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.applications import EfficientNetB3, EfficientNetB6
from tensorflow.keras.applications.efficientnet import preprocess_input


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_device() -> str:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return "CPU"
    details = tf.config.experimental.get_device_details(gpus[0])
    name = (details.get("device_name") or "").strip()
    return f"GPU: {name}" if name else "GPU"