"""Training and evaluation of the denoising autoencoder (notebook *Livrable 2*).

``train_denoiser`` reproduces the final protocol: the selected architecture
(A2 U-Net, MSE loss, learning rate 0.003) is trained on the three noises at
once ("mixed"), with early stopping on the validation set (medium noise),
then evaluated **once** on the test set against the classical filters, for
every noise type and intensity.

``compare_architectures`` reproduces the architecture comparison of the
notebook (gaussian noise, BCE, learning rate 0.001).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from tounum.config import Config, DenoisingConfig
from tounum.denoising.data import DenoisingData, prepare_denoising_data
from tounum.denoising.filters import BASELINES, psnr_ssim
from tounum.denoising.models import build_denoiser
from tounum.denoising.noise import add_noise, fixed_noisy_sets
from tounum.utils.logging_config import get_logger
from tounum.utils.reproducibility import configure_gpu_memory_growth, set_seed
from tounum.utils.runs import write_csv, write_json

LOGGER = get_logger("denoising.train")

MODEL_FILENAME = "denoiser.keras"
METADATA_FILENAME = "denoiser_metadata.json"


def training_dataset(x_train_u8: np.ndarray, noise: str, levels: dict, batch_size: int, seed: int):
    """(noisy, clean) pairs generated on the fly.

    Photos are shuffled, scaled to [0, 1], randomly flipped, then noised with
    an intensity drawn uniformly between the "low" and "high" levels. For
    ``noise="mixed"`` the noise type is also drawn at random for each photo.
    The noise changes at every epoch, so the model never sees the same noisy
    photo twice.
    """
    import tensorflow as tf

    noise_names = list(levels)

    def noise_with(x, name):
        intensity = tf.random.uniform([], levels[name]["low"], levels[name]["high"])
        return add_noise(x, name, intensity)

    def make_noisy(x):
        if noise == "mixed":
            choice = tf.random.uniform([], 0, len(noise_names), dtype=tf.int32)
            return tf.switch_case(choice, [lambda n=n: noise_with(x, n) for n in noise_names])
        return noise_with(x, noise)

    ds = tf.data.Dataset.from_tensor_slices(x_train_u8)
    ds = ds.shuffle(min(len(x_train_u8), 5000), seed=seed)
    ds = ds.map(lambda img: tf.cast(img, tf.float32) / 255.0)
    ds = ds.map(tf.image.random_flip_left_right)
    ds = ds.map(lambda clean: (make_noisy(clean), clean), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(2)


def fit_denoiser(
    data: DenoisingData,
    val_noisy: dict,
    cfg: DenoisingConfig,
    architecture: str,
    noise: str,
    loss: str,
    learning_rate: float,
    seed: int,
):
    """Train one autoencoder; validation uses the medium level of the learned noise(s)."""
    import keras

    set_seed(seed)
    model = build_denoiser(architecture)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=loss)

    val_noises = list(cfg.noise_levels) if noise == "mixed" else [noise]
    x_val = np.concatenate([val_noisy[(n, "medium")] for n in val_noises])
    y_val = np.concatenate([data.x_val] * len(val_noises))

    start = time.perf_counter()
    history = model.fit(
        training_dataset(data.x_train_u8, noise, cfg.noise_levels, cfg.batch_size, seed),
        epochs=cfg.max_epochs,
        validation_data=(x_val, y_val),
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=cfg.early_stopping_patience, restore_best_weights=True
            )
        ],
        shuffle=False,  # the tf.data pipeline already shuffles the photos
        verbose=2,
    )
    LOGGER.info(
        "%s trained: %d epochs in %.0f s",
        architecture,
        len(history.history["loss"]),
        time.perf_counter() - start,
    )
    return model, pd.DataFrame(history.history)


def baseline_table(noisy: dict, clean: np.ndarray, level: str = "medium") -> pd.DataFrame:
    """PSNR / SSIM of the classical filters, for every noise at one level."""
    rows = []
    for (noise, noise_level), x in noisy.items():
        if noise_level != level:
            continue
        for name, apply_filter in BASELINES.items():
            rows.append({"noise": noise, "level": level, "method": name, **psnr_ssim(apply_filter(x), clean)})
    return pd.DataFrame(rows)


def evaluate_against_baselines(model, noisy: dict, clean: np.ndarray) -> pd.DataFrame:
    """Every noise x level: classical filters and the autoencoder."""
    rows = []
    for (noise, level), x in noisy.items():
        for name, apply_filter in BASELINES.items():
            rows.append({"noise": noise, "level": level, "method": name, **psnr_ssim(apply_filter(x), clean)})
        prediction = model.predict(x, batch_size=16, verbose=0)
        rows.append({"noise": noise, "level": level, "method": "autoencoder", **psnr_ssim(prediction, clean)})
    return pd.DataFrame(rows)


def _load(config: Config) -> DenoisingData:
    cfg = config.denoising
    return prepare_denoising_data(
        config.path("denoising_train_data"),
        config.path("denoising_eval_data"),
        cfg.image_size,
        cfg.image_extensions,
        cfg.max_train_images,
        cfg.duplicate_threshold,
        config.seed,
        cache_dir=config.path("interim") / "denoising_cache",
    )


def train_denoiser(config: Config, run_dir: Path) -> dict:
    """Train the configured final model and evaluate it once on the test set."""
    configure_gpu_memory_growth()
    cfg = config.denoising
    data = _load(config)
    val_noisy = fixed_noisy_sets(data.x_val, cfg.noise_levels, cfg.val_noise_seed)
    test_noisy = fixed_noisy_sets(data.x_test, cfg.noise_levels, cfg.test_noise_seed)

    write_csv(run_dir / "validation_baselines.csv", baseline_table(val_noisy, data.x_val))
    model, history = fit_denoiser(
        data, val_noisy, cfg, cfg.architecture, cfg.train_noise, cfg.loss, cfg.learning_rate, config.seed
    )
    write_csv(run_dir / "history.csv", history, index=True)

    # Cross-noise validation table (which noise the model handles)
    validation = {
        noise: psnr_ssim(model.predict(val_noisy[(noise, "medium")], batch_size=16, verbose=0), data.x_val)
        for noise in cfg.noise_levels
    }

    # ---- single, final evaluation on the test set ---------------------------
    test_table = evaluate_against_baselines(model, test_noisy, data.x_test)
    write_csv(run_dir / "test_results.csv", test_table)
    clean_check = psnr_ssim(model.predict(data.x_test, batch_size=16, verbose=0), data.x_test)

    medium = test_table[test_table["level"] == "medium"]
    best_filter = medium[medium["method"].isin(list(BASELINES)[1:])].groupby("noise")["psnr"].max()
    autoencoder = medium[medium["method"] == "autoencoder"].set_index("noise")["psnr"]
    results = {
        "architecture": cfg.architecture,
        "loss": cfg.loss,
        "learning_rate": cfg.learning_rate,
        "train_noise": cfg.train_noise,
        "train_images": int(len(data.x_train_u8)),
        "val_images": int(len(data.x_val)),
        "test_images": int(len(data.x_test)),
        "duplicates_removed": data.duplicates_removed,
        "validation_medium_noise": validation,
        "test_medium_psnr_gain_vs_best_filter": (autoencoder - best_filter).to_dict(),
        "test_clean_image_check": clean_check,
    }
    write_json(run_dir / "metrics.json", results)
    write_csv(run_dir / "test_paths.csv", pd.DataFrame({"path": data.test_paths}))

    model.save(run_dir / MODEL_FILENAME)
    write_json(
        run_dir / METADATA_FILENAME,
        {
            "architecture": cfg.architecture,
            "image_size_trained": cfg.image_size,
            "size_multiple": 4,
            "noise_levels": cfg.noise_levels,
            "seed": config.seed,
        },
    )
    LOGGER.info("Denoiser saved to %s", run_dir)
    return results


def compare_architectures(config: Config, run_dir: Path) -> pd.DataFrame:
    """Architecture comparison of the notebook: gaussian noise, BCE, lr 0.001."""
    configure_gpu_memory_growth()
    cfg = config.denoising
    data = _load(config)
    val_noisy = fixed_noisy_sets(data.x_val, cfg.noise_levels, cfg.val_noise_seed)
    rows = []
    for architecture in cfg.compare_architectures:
        model, history = fit_denoiser(
            data, val_noisy, cfg, architecture, "gaussian", "binary_crossentropy", 0.001, config.seed
        )
        write_csv(run_dir / "histories" / f"{architecture}.csv", history, index=True)
        scores = psnr_ssim(
            model.predict(val_noisy[("gaussian", "medium")], batch_size=16, verbose=0), data.x_val
        )
        rows.append({"architecture": architecture, "parameters": model.count_params(), **scores})
    table = pd.DataFrame(rows).set_index("architecture")
    write_csv(run_dir / "architecture_comparison.csv", table, index=True)
    return table
