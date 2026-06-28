from __future__ import annotations

from typing import Callable

from tensorflow import keras

ModelFactory = Callable[..., keras.Model]
PreprocessFunction = Callable
InputSize = tuple[int, int]
