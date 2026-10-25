"""Transfer learning: frozen ImageNet CNNs and the feature cache.

The CNN is frozen, so each photo always gives the same features: they are
computed once and stored as one ``float16`` ``.npy`` file per image
(8x8 grid -> 64 regions x C values). Training then reads these files.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from tounum.utils.logging_config import get_logger

LOGGER = get_logger("captioning.features")

CNN_INPUT_SIZE = (299, 299)
# name -> (keras.applications constructor name, preprocess module, channels)
CNNS = {
    "InceptionV3": ("InceptionV3", "inception_v3", 2048),
    "InceptionResNetV2": ("InceptionResNetV2", "inception_resnet_v2", 1536),
}


def feature_channels(cnn: str) -> int:
    if cnn not in CNNS:
        raise ValueError(f"Unknown CNN '{cnn}'. Choose from {list(CNNS)}.")
    return CNNS[cnn][2]


def create_extractor(cnn: str, weights: str | None = "imagenet"):
    """Frozen CNN without its classification head (``include_top=False``)."""
    import keras

    feature_channels(cnn)
    constructor = getattr(keras.applications, CNNS[cnn][0])
    base = constructor(include_top=False, weights=weights)
    base.trainable = False
    return keras.Model(base.input, base.output, name=f"extractor_{cnn}")


def preprocess_images(images, cnn: str):
    """Pixels in [0, 255] -> the [-1, 1] range expected by the Inception CNNs."""
    import keras
    import tensorflow as tf

    module = getattr(keras.applications, CNNS[cnn][1])
    return module.preprocess_input(tf.cast(images, tf.float32))


def extract(extractor, images) -> np.ndarray:
    """(batch, 8, 8, C) feature maps -> (batch, 64, C)."""
    import tensorflow as tf

    output = extractor(images, training=False)
    return tf.reshape(output, (tf.shape(output)[0], -1, output.shape[-1])).numpy()


def read_image_299(path):
    """Decode (JPEG, PNG, ...) to RGB and resize to 299x299 (bilinear)."""
    import tensorflow as tf

    image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    return tf.image.resize(image, CNN_INPUT_SIZE)


def resize_array_299(image: np.ndarray):
    """Same resize as ``read_image_299`` for an in-memory uint8 RGB array."""
    import tensorflow as tf

    return tf.image.resize(tf.convert_to_tensor(image), CNN_INPUT_SIZE)


def feature_path(cache_dir: Path, cnn: str, image_id: int | str) -> Path:
    return Path(cache_dir) / cnn / f"{image_id}.npy"


def cache_features(
    image_ids: Sequence[int],
    image_paths: Sequence[str],
    cnn: str,
    cache_dir: Path,
    extractor,
    batch_size: int = 16,
) -> int:
    """Compute and store the features of images not yet cached. Returns the number computed."""
    import tensorflow as tf

    (Path(cache_dir) / cnn).mkdir(parents=True, exist_ok=True)
    todo = [
        (i, p)
        for i, p in zip(image_ids, image_paths, strict=True)
        if not feature_path(cache_dir, cnn, i).exists()
    ]
    if not todo:
        LOGGER.info("%s: all features already cached", cnn)
        return 0
    ids, paths = zip(*todo, strict=True)
    batches = (
        tf.data.Dataset.from_tensor_slices(list(paths))
        .map(read_image_299, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(2)
    )
    offset = 0
    for images in batches:
        features = extract(extractor, preprocess_images(images, cnn))
        for image_id, feature in zip(ids[offset : offset + len(features)], features, strict=True):
            target = feature_path(cache_dir, cnn, image_id)
            temporary = target.with_suffix(".tmp.npy")
            np.save(temporary, feature.astype("float16"))
            temporary.replace(target)
        offset += len(features)
    LOGGER.info("%s: %d image features computed", cnn, len(todo))
    return len(todo)


def load_features(image_ids: Sequence[int], cnn: str, cache_dir: Path) -> np.ndarray:
    paths = [feature_path(cache_dir, cnn, i) for i in image_ids]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} feature file(s) missing, e.g. {missing[0]}")
    return np.stack([np.load(p) for p in paths]).astype("float32")
