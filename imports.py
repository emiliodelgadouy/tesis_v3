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


def set_random_seeds(config) -> None:
    """Fija las semillas (random/numpy/tf) desde ``config["GENERAL"]["RANDOM_SEED"]``."""
    seed = config["GENERAL"]["RANDOM_SEED"]
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)