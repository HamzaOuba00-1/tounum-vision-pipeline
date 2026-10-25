"""Photo loading, splitting and near-duplicate removal (notebook *Livrable 2*).

* Training photos: the ``Photo`` folder of the classification dataset,
  randomly sampled down to ``max_train_images`` (seed 42).
* Evaluation photos: the denoising dataset, shuffled (seed 42) and cut in two
  halves - validation then test.
* Every photo is converted to RGB, centre-cropped and resized to 128x128.
* Training photos that are near-duplicates of an evaluation photo (16x16 grey
  thumbnails, mean absolute difference < 0.02) are removed to avoid leakage.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from tounum.utils.logging_config import get_logger

LOGGER = get_logger("denoising.data")


@dataclass
class DenoisingData:
    x_train_u8: np.ndarray  # (n, size, size, 3) uint8
    x_val: np.ndarray  # float32 in [0, 1]
    x_test: np.ndarray  # float32 in [0, 1]
    val_paths: list[str]
    test_paths: list[str]
    duplicates_removed: int


def list_images(folder: Path, extensions: Sequence[str]) -> list[str]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder}. See data/README.md.")
    allowed = {e.lower() for e in extensions}
    return sorted(str(p) for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in allowed)


def read_photo(path: str | Path, size: int) -> np.ndarray | None:
    """RGB, centre crop + resize to ``size`` x ``size`` (uint8), or ``None`` if unreadable."""
    try:
        with Image.open(path) as image:
            image.draft("RGB", (size * 2, size * 2))
            image = ImageOps.fit(image.convert("RGB"), (size, size))
            return np.asarray(image, dtype=np.uint8)
    except (OSError, ValueError) as error:
        LOGGER.warning("Unreadable photo skipped: %s (%s)", path, error)
        return None


def load_photos(paths: Sequence[str], size: int, workers: int = 8) -> tuple[np.ndarray, list[str]]:
    """Read photos in parallel; returns the stacked array and the paths actually read."""
    with ThreadPoolExecutor(max_workers=workers) as executor:
        images = list(executor.map(lambda p: read_photo(p, size), paths))
    kept = [(img, p) for img, p in zip(images, paths, strict=True) if img is not None]
    if not kept:
        raise ValueError("None of the photos could be read.")
    return np.stack([img for img, _ in kept]), [p for _, p in kept]


def thumbnails(images: np.ndarray) -> np.ndarray:
    """Flattened 16x16 grey thumbnails in [0, 1], used for duplicate detection."""
    import tensorflow as tf

    grey = images.astype("float32").mean(axis=-1, keepdims=True)
    if grey.max() > 1:
        grey = grey / 255.0
    return tf.image.resize(grey, (16, 16)).numpy().reshape(len(images), -1)


def near_duplicate_mask(train: np.ndarray, evaluation: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask of training photos too close to any evaluation photo."""
    train_thumbs = thumbnails(train)
    eval_thumbs = thumbnails(evaluation)
    return np.array([np.abs(eval_thumbs - thumb).mean(axis=1).min() < threshold for thumb in train_thumbs])


def split_eval_paths(paths: Sequence[str], seed: int) -> tuple[list[str], list[str]]:
    """Shuffle and cut the evaluation photos in half: validation first, then test."""
    frame = pd.DataFrame({"path": list(paths)}).sample(frac=1, random_state=seed)
    frame = frame.reset_index(drop=True)
    half = len(frame) // 2
    return list(frame.loc[: half - 1, "path"]), list(frame.loc[half:, "path"])


def prepare_denoising_data(
    train_dir: Path,
    eval_dir: Path,
    size: int,
    extensions: Sequence[str],
    max_train: int,
    duplicate_threshold: float,
    seed: int,
    cache_dir: Path | None = None,
) -> DenoisingData:
    """Load the three sets exactly as in the notebook."""
    train_paths = pd.Series(list_images(train_dir, extensions))
    if train_paths.empty:
        raise ValueError(f"No training photo in {train_dir}")
    train_paths = train_paths.sample(min(max_train, len(train_paths)), random_state=seed)
    eval_paths = list_images(eval_dir, extensions)
    if len(eval_paths) < 2:
        raise ValueError(f"At least two evaluation photos are needed in {eval_dir}")

    cache_file = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"train_{size}_{len(train_paths)}_seed{seed}.npy"
    if cache_file is not None and cache_file.exists():
        x_train_u8 = np.load(cache_file)
        LOGGER.info("Training photos loaded from cache %s", cache_file)
    else:
        x_train_u8, _ = load_photos(list(train_paths), size)
        if cache_file is not None:
            np.save(cache_file, x_train_u8)

    val_paths, test_paths = split_eval_paths(eval_paths, seed)
    x_val_u8, val_paths = load_photos(val_paths, size)
    x_test_u8, test_paths = load_photos(test_paths, size)
    x_val = x_val_u8.astype("float32") / 255.0
    x_test = x_test_u8.astype("float32") / 255.0

    duplicates = near_duplicate_mask(x_train_u8, np.concatenate([x_val, x_test]), duplicate_threshold)
    x_train_u8 = x_train_u8[~duplicates]
    LOGGER.info(
        "Train %s (duplicates removed: %d) | val %s | test %s",
        x_train_u8.shape,
        int(duplicates.sum()),
        x_val.shape,
        x_test.shape,
    )
    return DenoisingData(x_train_u8, x_val, x_test, val_paths, test_paths, int(duplicates.sum()))
