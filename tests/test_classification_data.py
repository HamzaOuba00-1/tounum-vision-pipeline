import numpy as np
import pytest

from tounum.classification import data as cdata

CLASSES = ["Painting", "Photo", "Schematics", "Sketch", "Text"]
EXT = [".jpg", ".png"]


def test_inventory_labels_and_known_invalid_exclusion(classification_dir):
    frame = cdata.build_inventory(classification_dir, CLASSES, "Photo", EXT, ["Painting/painting_000.jpg"])
    assert set(frame.columns) == {"path", "source_class", "label"}
    assert (frame.loc[frame["source_class"] == "Photo", "label"] == 1).all()
    assert (frame.loc[frame["source_class"] != "Photo", "label"] == 0).all()
    assert not frame["path"].str.endswith("painting_000.jpg").any()
    assert (frame["source_class"] == "Painting").sum() == 9


def test_missing_dataset_and_class_folder(tmp_path, classification_dir):
    with pytest.raises(FileNotFoundError):
        cdata.build_inventory(tmp_path / "nope", CLASSES, "Photo", EXT)
    with pytest.raises(FileNotFoundError, match="Missing class folder"):
        cdata.build_inventory(classification_dir, [*CLASSES, "Maps"], "Photo", EXT)


def test_validate_inventory_errors(classification_dir):
    frame = cdata.build_inventory(classification_dir, CLASSES, "Photo", EXT)
    cdata.validate_inventory(frame, CLASSES)
    with pytest.raises(cdata.DataValidationError, match="Both labels"):
        cdata.validate_inventory(frame[frame["label"] == 0], ["Painting"])
    with pytest.raises(cdata.DataValidationError, match="duplicated"):
        cdata.validate_inventory(frame.iloc[[0, 0, *range(len(frame))]], CLASSES)


def test_cache_skips_unreadable_and_is_reused(classification_dir, tmp_path):
    frame = cdata.build_inventory(classification_dir, CLASSES, "Photo", EXT)
    assert frame["path"].str.endswith("broken.jpg").any()
    cached = cdata.build_image_cache(frame, tmp_path / "cache", (32, 32), workers=2)
    assert len(cached) == len(frame) - 1
    assert not cached["path"].str.endswith("broken.jpg").any()
    first = {p: (tmp_path / "cache" / p).stat().st_mtime_ns for p in cached["cache_path"]}
    again = cdata.build_image_cache(frame, tmp_path / "cache", (32, 32), workers=2)
    assert {p: (tmp_path / "cache" / p).stat().st_mtime_ns for p in again["cache_path"]} == first


def test_split_is_stratified_and_disjoint(classification_dir, tmp_path):
    frame = cdata.build_inventory(classification_dir, CLASSES, "Photo", EXT)
    splits = cdata.split_dataset(frame, 0.2, 0.2, seed=42)
    assert sum(len(s) for s in splits.values()) == len(frame)
    for split in splits.values():
        assert set(split["source_class"]) == set(CLASSES)
    again = cdata.split_dataset(frame, 0.2, 0.2, seed=42)
    assert list(again["test"]["path"]) == list(splits["test"]["path"])  # reproducible


def test_assert_disjoint_detects_leakage(classification_dir):
    frame = cdata.build_inventory(classification_dir, CLASSES, "Photo", EXT)
    with pytest.raises(cdata.DataValidationError, match="shared"):
        cdata.assert_disjoint({"train": frame, "test": frame.head(3)})


def test_subsample_keeps_every_class(classification_dir):
    frame = cdata.build_inventory(classification_dir, CLASSES, "Photo", EXT)
    sample = cdata.subsample(frame, 0.5, seed=1)
    assert len(sample) < len(frame)
    assert set(sample["source_class"]) == set(CLASSES)
    assert cdata.subsample(frame, 1.0, seed=1) is frame


def test_prediction_preprocessing_matches_training_pipeline(classification_dir, tmp_path):
    """The predictor must see exactly the pixels the model was trained on."""
    import tensorflow as tf

    frame = cdata.build_inventory(classification_dir, CLASSES, "Photo", EXT).head(5)
    cached = cdata.build_image_cache(frame, tmp_path / "cache", (32, 32), workers=1)
    for _, row in cached.iterrows():
        training_pixels = tf.io.decode_jpeg(tf.io.read_file(row["cache_path"]), channels=3).numpy()
        prediction_pixels = cdata.preprocess_for_model(row["path"], (32, 32))
        assert prediction_pixels.shape == (32, 32, 3)
        np.testing.assert_array_equal(prediction_pixels, training_pixels)


def test_make_dataset_shapes(classification_dir, tmp_path):
    frame = cdata.build_inventory(classification_dir, CLASSES, "Photo", EXT)
    cached = cdata.build_image_cache(frame, tmp_path / "cache", (32, 32), workers=2)
    images, labels = next(iter(cdata.make_dataset(cached, (32, 32), 4, training=True)))
    assert images.shape == (4, 32, 32, 3) and labels.shape == (4,)
    assert float(images.numpy().max()) > 1.0  # raw pixels: normalisation is inside the model
