"""Denoise photos of any size with a trained autoencoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from tounum.denoising.train import METADATA_FILENAME, MODEL_FILENAME
from tounum.utils.runs import read_json


def pad_to_multiple(x: np.ndarray, multiple: int = 4) -> np.ndarray:
    """Reflect-pad an H x W x C array so that H and W are multiples of ``multiple``."""
    height, width = x.shape[:2]
    pad_h, pad_w = -height % multiple, -width % multiple
    mode = "reflect" if height > pad_h and width > pad_w else "edge"
    return np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)


def limit_size(image: np.ndarray, max_side: int | None) -> np.ndarray:
    """Optionally downscale very large photos before denoising (memory safety)."""
    if max_side is None or max(image.shape[:2]) <= max_side:
        return image
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    resized = Image.fromarray(image).resize(
        (max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.uint8)


class Denoiser:
    """Wraps the trained autoencoder; same behaviour as the notebook's ``debruiter``."""

    def __init__(self, model, size_multiple: int = 4, max_side: int | None = None) -> None:
        self.model = model
        self.size_multiple = size_multiple
        self.max_side = max_side

    @classmethod
    def from_run(cls, run_dir: str | Path, max_side: int | None = None) -> Denoiser:
        import keras

        run_dir = Path(run_dir)
        metadata = read_json(run_dir / METADATA_FILENAME)
        model_path = run_dir / MODEL_FILENAME
        if not model_path.is_file():
            raise FileNotFoundError(f"Denoiser model not found: {model_path}")
        return cls(keras.models.load_model(model_path), metadata.get("size_multiple", 4), max_side)

    def denoise(self, image: np.ndarray | Image.Image) -> np.ndarray:
        """uint8 RGB in, uint8 RGB out (same size, unless ``max_side`` downscales it)."""
        array = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError("Expected an RGB image of shape (H, W, 3).")
        array = limit_size(array.astype(np.uint8), self.max_side)
        height, width = array.shape[:2]
        x = pad_to_multiple(array.astype("float32") / 255.0, self.size_multiple)
        y = self.model.predict(x[np.newaxis], verbose=0)[0, :height, :width]
        return (np.clip(y, 0, 1) * 255).astype("uint8")
