from __future__ import annotations
import random

import numpy as np
import tensorflow as tf


def set_random_seeds(config) -> None:
    """Fija las semillas (random/numpy/tf) desde ``config["GENERAL"]["RANDOM_SEED"]``."""
    seed = config["GENERAL"]["RANDOM_SEED"]
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
