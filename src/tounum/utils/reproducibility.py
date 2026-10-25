"""Random seeds and TensorFlow device configuration."""

from __future__ import annotations

import contextlib
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow/Keras (as ``keras.utils.set_random_seed``)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import keras

        keras.utils.set_random_seed(seed)
    except ImportError:  # pragma: no cover - TensorFlow is a hard dependency
        pass


def configure_gpu_memory_growth() -> None:
    """Allocate GPU memory progressively instead of reserving the whole card."""
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        with contextlib.suppress(RuntimeError):  # GPUs already initialised
            tf.config.experimental.set_memory_growth(gpu, True)
