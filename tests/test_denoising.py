import numpy as np
import pytest

from tounum.denoising import data as ddata
from tounum.denoising.filters import BASELINES, psnr_ssim
from tounum.denoising.models import ARCHITECTURES, build_denoiser
from tounum.denoising.noise import add_noise, fixed_noisy_sets
from tounum.denoising.predict import Denoiser, limit_size, pad_to_multiple
from tounum.denoising.train import METADATA_FILENAME, MODEL_FILENAME, compare_architectures, train_denoiser
from tounum.utils.runs import create_run_dir, read_json

LEVELS = {
    "gaussian": {"low": 0.05, "medium": 0.10, "high": 0.20},
    "salt_pepper": {"low": 0.02, "medium": 0.05, "high": 0.10},
    "speckle": {"low": 0.10, "medium": 0.25, "high": 0.50},
}


@pytest.fixture
def clean_images(rng):
    return rng.uniform(0.2, 0.8, size=(4, 16, 16, 3)).astype("float32")


def test_noises_stay_in_range_and_change_the_image(clean_images):
    for noise in LEVELS:
        noisy = add_noise(clean_images, noise, 0.2).numpy()
        assert noisy.min() >= 0 and noisy.max() <= 1
        assert not np.allclose(noisy, clean_images)
    with pytest.raises(ValueError):
        add_noise(clean_images, "blur", 0.1)


def test_salt_pepper_rate(rng):
    x = np.full((1, 200, 200, 3), 0.5, dtype="float32")
    noisy = add_noise(x, "salt_pepper", 0.10).numpy()
    altered = (noisy[..., 0] != 0.5).mean()
    assert 0.08 < altered < 0.12
    assert ((noisy == 0) | (noisy == 1) | (noisy == 0.5)).all()


def test_fixed_noise_is_reproducible(clean_images):
    a = fixed_noisy_sets(clean_images, LEVELS, seed=1000)
    b = fixed_noisy_sets(clean_images, LEVELS, seed=1000)
    assert set(a) == {(n, level) for n in LEVELS for level in ("low", "medium", "high")}
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_baselines_and_metrics(clean_images):
    noisy = add_noise(clean_images, "gaussian", 0.1).numpy()
    for apply_filter in BASELINES.values():
        assert apply_filter(noisy).shape == noisy.shape
    perfect = psnr_ssim(clean_images, clean_images)
    assert perfect["ssim"] == pytest.approx(1.0)
    assert psnr_ssim(noisy, clean_images)["psnr"] < 40


def test_parameter_counts_match_notebook():
    expected = {"A0_base": 29_507, "A1_wide": 298_755, "A2_unet": 158_659, "A3_strided": 75_651}
    assert {name: build().count_params() for name, build in ARCHITECTURES.items()} == expected


@pytest.mark.parametrize("name", list(ARCHITECTURES))
def test_models_are_fully_convolutional(name):
    model = build_denoiser(name)
    for shape in [(1, 16, 16, 3), (1, 24, 40, 3)]:
        assert model.predict(np.zeros(shape, "float32"), verbose=0).shape == shape


def test_pad_and_limit_size():
    x = np.zeros((10, 13, 3))
    assert pad_to_multiple(x).shape == (12, 16, 3)
    assert pad_to_multiple(np.zeros((2, 3, 3))).shape == (4, 4, 3)  # tiny image: edge padding
    big = np.zeros((100, 50, 3), dtype=np.uint8)
    assert limit_size(big, 40).shape == (40, 20, 3)
    assert limit_size(big, None) is big


def test_denoiser_keeps_size_of_arbitrary_images(rng):
    denoiser = Denoiser(build_denoiser("A2_unet"))
    image = rng.integers(0, 255, size=(21, 30, 3), dtype=np.uint8)
    output = denoiser.denoise(image)
    assert output.shape == image.shape and output.dtype == np.uint8
    with pytest.raises(ValueError):
        denoiser.denoise(np.zeros((10, 10)))


def test_near_duplicates_are_detected(rng):
    evaluation = rng.uniform(size=(3, 16, 16, 3)).astype("float32")
    train = np.concatenate(
        [(evaluation[:1] * 255).astype(np.uint8), rng.integers(0, 255, (2, 16, 16, 3), np.uint8)]
    )
    mask = ddata.near_duplicate_mask(train, evaluation, 0.02)
    assert mask.tolist() == [True, False, False]


def test_eval_split_halves():
    val, test = ddata.split_eval_paths([f"p{i}" for i in range(7)], seed=42)
    assert len(val) == 3 and len(test) == 4 and not set(val) & set(test)


def test_end_to_end_training_and_comparison(classification_dir, denoising_dirs, tiny_config):
    run_dir = create_run_dir(tiny_config.path("artifacts"), "denoising", "test")
    results = train_denoiser(tiny_config, run_dir)
    assert results["architecture"] == "A2_unet" and results["val_images"] == 3
    assert set(results["validation_medium_noise"]) == set(LEVELS)
    for name in [MODEL_FILENAME, METADATA_FILENAME, "test_results.csv", "history.csv", "metrics.json"]:
        assert (run_dir / name).is_file(), name
    assert read_json(run_dir / METADATA_FILENAME)["size_multiple"] == 4
    assert Denoiser.from_run(run_dir).denoise(np.zeros((18, 22, 3), np.uint8)).shape == (18, 22, 3)

    table = compare_architectures(tiny_config, create_run_dir(tiny_config.path("artifacts"), "cmp", "t"))
    assert list(table.index) == ["A0_base", "A2_unet"]
    assert (table["psnr"] > 0).all()


def test_missing_folders_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        ddata.list_images(tmp_path / "missing", [".png"])
