"""Synthetic datasets and tiny configurations shared by the tests.

No real data is needed: images are generated on the fly, models are tiny
and trained for one or two epochs, and the ImageNet CNN of the captioning
module is replaced by a small Keras model with the same output shape.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest
from PIL import Image

from tounum.config import config_from_dict

CLASSES = ["Painting", "Photo", "Schematics", "Sketch", "Text"]


def _photo_like(rng: np.random.Generator, size: int) -> np.ndarray:
    """Smooth colour gradient + a disc: compressible, 'natural' image."""
    y, x = np.mgrid[0:size, 0:size] / size
    base = rng.uniform(0.2, 0.8, size=3)
    image = np.stack([base[c] + 0.3 * (x if c % 2 else y) for c in range(3)], axis=-1)
    cy, cx, r = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7), rng.uniform(0.1, 0.25)
    image[(y - cy) ** 2 + (x - cx) ** 2 < r**2] *= 0.5
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def _document_like(rng: np.random.Generator, size: int) -> np.ndarray:
    """White page with dark horizontal strokes (text / schematics)."""
    image = np.full((size, size, 3), 245, dtype=np.uint8)
    for row in range(2, size - 2, 4):
        start, length = rng.integers(0, size // 3), rng.integers(size // 3, size)
        image[row, start : start + length] = rng.integers(0, 60)
    return image


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def classification_dir(tmp_path, rng):
    root = tmp_path / "data" / "raw" / "classification"
    for source_class in CLASSES:
        folder = root / source_class
        folder.mkdir(parents=True)
        for i in range(10):
            maker = _photo_like if source_class == "Photo" else _document_like
            Image.fromarray(maker(rng, 48)).save(folder / f"{source_class.lower()}_{i:03d}.jpg")
    # one corrupted file, which must be skipped with a warning
    (root / "Sketch" / "broken.jpg").write_bytes(b"not a jpeg")
    return root


@pytest.fixture
def denoising_dirs(tmp_path, rng):
    eval_dir = tmp_path / "data" / "raw" / "denoising"
    eval_dir.mkdir(parents=True)
    for i in range(6):
        Image.fromarray(_photo_like(rng, 40)).save(eval_dir / f"eval_{i}.png")
    return eval_dir


@pytest.fixture
def coco_dir(tmp_path, rng):
    """Fake MS COCO: 40 images with 5 captions each, plus one image missing on disk."""
    root = tmp_path / "data" / "raw" / "coco"
    images = root / "train2014"
    images.mkdir(parents=True)
    subjects = ["a dog", "a cat", "two people", "a red bus", "a man"]
    places = ["on the grass", "in a kitchen", "near the beach", "on a street"]
    annotations = []
    for image_id in range(1, 41):
        Image.fromarray(_photo_like(rng, 32)).save(images / f"COCO_train2014_{image_id:012d}.jpg")
        for k in range(5):
            caption = f"{subjects[(image_id + k) % 5].capitalize()} {places[image_id % 4]}."
            annotations.append({"image_id": image_id, "caption": caption})
    annotations.append({"image_id": 999, "caption": "An image that is not on disk."})
    (root / "annotations").mkdir()
    (root / "annotations" / "captions_train2014.json").write_text(json.dumps({"annotations": annotations}))
    return root


@pytest.fixture
def tiny_config(tmp_path):
    """Configuration with tiny models and epochs; paths inside ``tmp_path``."""
    return config_from_dict(
        {
            "seed": 42,
            "paths": {
                "root": ".",
                "classification_data": "data/raw/classification",
                "denoising_train_data": "data/raw/classification/Photo",
                "denoising_eval_data": "data/raw/denoising",
                "coco_annotations": "data/raw/coco/annotations/captions_train2014.json",
                "coco_images": "data/raw/coco/train2014",
                "interim": "data/interim",
                "artifacts": "artifacts",
            },
            "classification": {
                "image_size": [32, 32],
                "batch_size": 8,
                "max_epochs": 2,
                "val_fraction": 0.2,
                "test_fraction": 0.2,
                "cache_workers": 2,
                "known_invalid_files": ["Painting/painting_000.jpg"],
                "candidates": [
                    {"name": "cnn_reference", "variant": "reference"},
                    {"name": "cnn_4_blocks", "variant": "four_blocks"},
                ],
            },
            "denoising": {
                "image_size": 16,
                "max_train_images": 8,
                "batch_size": 4,
                "max_epochs": 1,
                "compare_architectures": ["A0_base", "A2_unet"],
            },
            "captioning": {
                "n_test": 6,
                "n_val": 6,
                "n_train": 24,
                "vocab_size": 30,
                "max_epochs": 2,
                "patience": 1,
                "generation_batch_size": 4,
                "feature_batch_size": 4,
                "model": {"units": 16, "embedding_dim": 8, "batch_size": 8},
            },
        },
        base_dir=tmp_path,
    )


@pytest.fixture
def fake_extractor():
    """Stands in for InceptionResNetV2: (B, 299, 299, 3) -> (B, 8, 8, 1536)."""
    import keras
    from keras import layers

    keras.utils.set_random_seed(0)
    inputs = keras.Input(shape=(None, None, 3))
    x = layers.Resizing(8, 8)(inputs)
    outputs = layers.Conv2D(1536, 1, activation="relu")(x)
    return keras.Model(inputs, outputs, name="fake_extractor")
