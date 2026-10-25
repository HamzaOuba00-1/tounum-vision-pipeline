from pathlib import Path

import pytest

from tounum.config import ConfigError, config_from_dict, load_config


def test_default_config_loads_with_notebook_values():
    config = load_config()
    assert config.seed == 42
    assert config.classification.image_size == (180, 180)
    assert [c.variant for c in config.classification.candidates] == [
        "reference",
        "regularized",
        "four_blocks",
    ]
    assert config.denoising.architecture == "A2_unet"
    assert config.captioning.model.cnn == "InceptionResNetV2"


def test_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config()
    assert config.path("artifacts") == Path(__file__).resolve().parents[1] / "artifacts"


def test_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="Unknown key"):
        config_from_dict({"classification": {"imagesize": [1, 1]}}, tmp_path)


@pytest.mark.parametrize(
    "section, values",
    [
        ("classification", {"data_fraction": 0}),
        ("classification", {"val_fraction": 0.5, "test_fraction": 0.5}),
        ("denoising", {"image_size": 30}),
        ("denoising", {"train_noise": "blur"}),
        ("captioning", {"decoding": "beam"}),
    ],
)
def test_invalid_values_are_rejected(tmp_path, section, values):
    with pytest.raises(ConfigError):
        config_from_dict({section: values}, tmp_path)


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")
