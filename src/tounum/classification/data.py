"""Data ingestion, validation, caching and splitting for the photo classifier.

Faithful to notebook *Livrable 1*:

* five source classes, one folder each; ``Photo`` -> label 1, the rest -> 0;
* the known corrupted file ``Painting/painting_02662.jpg`` is excluded;
* images are decoded with PIL (``draft`` for fast JPEG decoding), converted
  to RGB, resized to 180x180 (bilinear, no aspect-ratio preservation) and
  cached as JPEG (quality 95), keyed by the MD5 of their source path;
* unreadable files are dropped and reported;
* stratified 70 / 15 / 15 split on the *source* class (seed 42).
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from tounum.utils.logging_config import get_logger

LOGGER = get_logger("classification.data")

# Very large scans are legitimate inputs: disable PIL's decompression-bomb guard
# (same as the notebook).
Image.MAX_IMAGE_PIXELS = None

JPEG_CACHE_QUALITY = 95


class DataValidationError(ValueError):
    """Raised when the dataset does not match the expected structure."""


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------
def build_inventory(
    data_dir: Path,
    source_classes: Sequence[str],
    positive_class: str,
    extensions: Sequence[str],
    known_invalid_files: Sequence[str] = (),
) -> pd.DataFrame:
    """List every image of every class folder.

    Returns a DataFrame with columns ``path`` (absolute), ``source_class`` and
    ``label`` (1 for ``positive_class``, 0 otherwise).
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Classification dataset not found: {data_dir}. See data/README.md for the expected layout."
        )
    allowed = {ext.lower() for ext in extensions}
    records = []
    for source_class in source_classes:
        class_dir = data_dir / source_class
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in allowed:
                records.append(
                    {
                        "path": str(path.resolve()),
                        "source_class": source_class,
                        "label": int(source_class == positive_class),
                    }
                )
    frame = pd.DataFrame(records, columns=["path", "source_class", "label"])

    invalid = {str((data_dir / rel).resolve()) for rel in known_invalid_files}
    excluded = frame["path"].isin(invalid)
    if excluded.any():
        LOGGER.info("Excluding %d known invalid file(s)", int(excluded.sum()))
        frame = frame[~excluded]
    return frame.reset_index(drop=True)


def validate_inventory(frame: pd.DataFrame, source_classes: Sequence[str]) -> None:
    """Check the inventory before any expensive processing."""
    if frame.empty:
        raise DataValidationError("No image found in the classification dataset.")
    missing = set(source_classes) - set(frame["source_class"])
    if missing:
        raise DataValidationError(f"Classes without any image: {sorted(missing)}")
    if set(frame["label"]) != {0, 1}:
        raise DataValidationError("Both labels (0 and 1) must be present.")
    duplicated = frame["path"].duplicated()
    if duplicated.any():
        raise DataValidationError(f"{int(duplicated.sum())} duplicated path(s) in the inventory.")


def class_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Image count per source class and per binary label."""
    return frame.groupby(["source_class", "label"]).size().rename("images").reset_index()


# ---------------------------------------------------------------------------
# Image preprocessing (shared by training and prediction)
# ---------------------------------------------------------------------------
def load_and_resize(source: str | Path | Image.Image, image_size: tuple[int, int]) -> np.ndarray:
    """Decode an image, convert it to RGB and resize it (uint8, H x W x 3).

    ``image_size`` is ``(height, width)``. Identical to the notebook's
    ``load_and_resize``: JPEG draft decoding at twice the target size, then a
    bilinear resize that does not preserve the aspect ratio.
    """
    height, width = image_size
    if isinstance(source, Image.Image):
        image = source.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)
    with Image.open(source) as image:
        image.draft("RGB", (width * 2, height * 2))
        image = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def jpeg_roundtrip(array: np.ndarray, quality: int = JPEG_CACHE_QUALITY) -> np.ndarray:
    """Encode with PIL (as the cache does) and decode with TensorFlow (as training does).

    Training images are read from the JPEG cache by ``tf.io.decode_jpeg``, so
    they carry JPEG artefacts and TensorFlow's decoding. Reproducing both steps
    at prediction time removes the train/serve skew of the notebook's
    ``predict_image`` (which resized with ``tf.image.resize`` instead).
    """
    import tensorflow as tf

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=quality)
    return tf.io.decode_jpeg(buffer.getvalue(), channels=3).numpy()


def preprocess_for_model(source: str | Path | Image.Image, image_size: tuple[int, int]) -> np.ndarray:
    """Exact preprocessing seen by the model during training (float32, 0-255)."""
    return jpeg_roundtrip(load_and_resize(source, image_size)).astype(np.float32)


# ---------------------------------------------------------------------------
# Resized-image cache
# ---------------------------------------------------------------------------
def cache_path_for(source_path: str, cache_dir: Path) -> Path:
    return cache_dir / (hashlib.md5(source_path.encode()).hexdigest() + ".jpg")


def _prepare_one(source_path: str, cache_dir: Path, image_size: tuple[int, int]) -> str | None:
    target = cache_path_for(source_path, cache_dir)
    if target.exists():
        return str(target)
    try:
        temporary = target.with_suffix(".tmp")
        Image.fromarray(load_and_resize(source_path, image_size)).save(
            temporary, format="JPEG", quality=JPEG_CACHE_QUALITY
        )
        temporary.replace(target)  # atomic: no half-written cache file
        return str(target)
    except (OSError, ValueError) as error:  # corrupted / unreadable image
        LOGGER.warning("Unreadable image skipped: %s (%s)", source_path, error)
        return None


def build_image_cache(
    frame: pd.DataFrame, cache_dir: Path, image_size: tuple[int, int], workers: int = 8
) -> pd.DataFrame:
    """Resize every image once into ``cache_dir`` and add a ``cache_path`` column.

    Unreadable images are removed from the returned frame.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        cache_paths = list(executor.map(lambda p: _prepare_one(p, cache_dir, image_size), frame["path"]))
    result = frame.copy()
    result["cache_path"] = cache_paths
    unreadable = result["cache_path"].isna()
    if unreadable.any():
        LOGGER.warning("%d unreadable image(s) removed", int(unreadable.sum()))
    return result[~unreadable].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sampling and splitting
# ---------------------------------------------------------------------------
def subsample(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Stratified sub-sample on the source class (notebook ``DATA_FRACTION``)."""
    if fraction >= 1.0:
        return frame
    sample, _ = train_test_split(
        frame, train_size=fraction, stratify=frame["source_class"], random_state=seed
    )
    return sample.reset_index(drop=True)


def split_dataset(
    frame: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int
) -> dict[str, pd.DataFrame]:
    """Stratified train / validation / test split on the source class."""
    dev, test = train_test_split(
        frame, test_size=test_fraction, stratify=frame["source_class"], random_state=seed
    )
    train, val = train_test_split(
        dev,
        test_size=val_fraction / (1 - test_fraction),
        stratify=dev["source_class"],
        random_state=seed,
    )
    splits = {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
    assert_disjoint(splits)
    return splits


def assert_disjoint(splits: dict[str, pd.DataFrame]) -> None:
    """Guard against leakage: no image may belong to two splits."""
    names = list(splits)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            common = set(splits[first]["path"]) & set(splits[second]["path"])
            if common:
                raise DataValidationError(f"{len(common)} image(s) shared between '{first}' and '{second}'.")


# ---------------------------------------------------------------------------
# tf.data pipelines
# ---------------------------------------------------------------------------
def make_dataset(
    frame: pd.DataFrame,
    image_size: tuple[int, int],
    batch_size: int,
    training: bool = False,
    seed: int = 42,
):
    """Batched ``tf.data`` pipeline reading the cached JPEGs (pixels in [0, 255]).

    Shuffling is applied to the training set only. Normalisation happens
    inside the model (``Rescaling`` layer).
    """
    import tensorflow as tf

    paths = frame["cache_path"].to_numpy()
    labels = frame["label"].to_numpy(dtype=np.float32)

    def load_image(path, label):
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        image.set_shape([*image_size, 3])
        return tf.cast(image, tf.float32), tf.cast(label, tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(frame), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
