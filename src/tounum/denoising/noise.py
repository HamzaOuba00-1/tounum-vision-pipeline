"""Synthetic noise models (images in [0, 1]).

* gaussian    - additive sensor grain;
* salt_pepper - dead pixels / dust, applied to all channels of a pixel;
* speckle     - multiplicative grain, stronger on bright areas.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

NOISE_TYPES = ("gaussian", "salt_pepper", "speckle")


def gaussian(x, intensity):
    import tensorflow as tf

    return tf.clip_by_value(x + intensity * tf.random.normal(tf.shape(x)), 0.0, 1.0)


def salt_pepper(x, intensity):
    import tensorflow as tf

    draw = tf.random.uniform(tf.shape(x)[:-1])[..., tf.newaxis]
    x = tf.where(draw < intensity / 2, 0.0, x)
    return tf.where(draw > 1 - intensity / 2, 1.0, x)


def speckle(x, intensity):
    import tensorflow as tf

    return tf.clip_by_value(x + x * intensity * tf.random.normal(tf.shape(x)), 0.0, 1.0)


NOISE_FUNCTIONS = {"gaussian": gaussian, "salt_pepper": salt_pepper, "speckle": speckle}


def add_noise(x, noise: str, intensity: float):
    if noise not in NOISE_FUNCTIONS:
        raise ValueError(f"Unknown noise '{noise}'. Choose from {NOISE_TYPES}.")
    return NOISE_FUNCTIONS[noise](x, intensity)


def fixed_noisy_sets(
    images: np.ndarray, levels: Mapping[str, Mapping[str, float]], seed: int
) -> dict[tuple[str, str], np.ndarray]:
    """Noise drawn once per (noise, level), so every method sees the same inputs.

    The seed of each combination is ``seed + 10 * noise_index + level_index``,
    as in the notebook (``bruiter_une_fois``).
    """
    import tensorflow as tf

    result = {}
    constant = tf.constant(images, tf.float32)
    for k, (noise, noise_levels) in enumerate(levels.items()):
        for n, (level, intensity) in enumerate(noise_levels.items()):
            tf.random.set_seed(seed + 10 * k + n)
            result[(noise, level)] = add_noise(constant, noise, intensity).numpy()
    return result
