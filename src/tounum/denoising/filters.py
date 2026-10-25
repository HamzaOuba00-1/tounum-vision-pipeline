"""Classical denoising baselines (3x3 neighbourhood) and image-quality metrics."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def no_filter(x: np.ndarray) -> np.ndarray:
    return x


def mean_filter(x: np.ndarray) -> np.ndarray:
    return ndimage.uniform_filter(x, size=(1, 3, 3, 1))


def gaussian_filter(x: np.ndarray) -> np.ndarray:
    return ndimage.gaussian_filter(x, sigma=(0, 1, 1, 0))


def median_filter(x: np.ndarray) -> np.ndarray:
    return ndimage.median_filter(x, size=(1, 3, 3, 1))


BASELINES = {
    "no_filter": no_filter,
    "mean_filter": mean_filter,
    "gaussian_filter": gaussian_filter,
    "median_filter": median_filter,
}


def psnr_ssim(images: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Mean PSNR (dB) and SSIM against the clean images (pixels in [0, 1])."""
    import tensorflow as tf

    a = tf.constant(np.clip(images, 0, 1), tf.float32)
    b = tf.constant(reference, tf.float32)
    return {
        "psnr": float(tf.reduce_mean(tf.image.psnr(a, b, 1.0))),
        "ssim": float(tf.reduce_mean(tf.image.ssim(a, b, 1.0))),
    }
