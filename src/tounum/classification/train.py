"""End-to-end training of the photo classifier (notebook *Livrable 1*).

Protocol (unchanged from the notebook):

1. inventory + validation of the dataset, resized-image cache;
2. stratified 70 / 15 / 15 split on the source class;
3. class weights computed on the training set only;
4. every candidate CNN is trained with ReduceLROnPlateau + EarlyStopping
   (``restore_best_weights``) monitored on the validation loss;
5. the model with the best validation F1 macro (threshold 0.5) is kept;
6. the decision threshold is tuned on the validation set;
7. the test set is evaluated **once**, with the frozen model and threshold.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from tounum.classification import data as cdata
from tounum.classification.evaluate import (
    compute_metrics,
    confusion_frame,
    plateau_analysis,
    select_threshold,
    source_class_analysis,
)
from tounum.classification.model import build_classifier
from tounum.config import Config
from tounum.utils.logging_config import get_logger
from tounum.utils.reproducibility import configure_gpu_memory_growth, set_seed
from tounum.utils.runs import write_csv, write_json

LOGGER = get_logger("classification.train")

MODEL_FILENAME = "classifier.keras"
METADATA_FILENAME = "classifier_metadata.json"


def compute_class_weights(labels: np.ndarray) -> dict[int, float]:
    """'Balanced' class weights, from the training labels only."""
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(c): float(w) for c, w in zip(classes, weights, strict=True)}


def prepare_splits(config: Config) -> dict[str, pd.DataFrame]:
    """Inventory, validation, cache, optional sub-sample and split."""
    cfg = config.classification
    frame = cdata.build_inventory(
        config.path("classification_data"),
        cfg.source_classes,
        cfg.positive_class,
        cfg.image_extensions,
        cfg.known_invalid_files,
    )
    cdata.validate_inventory(frame, cfg.source_classes)
    LOGGER.info("Inventory: %d images", len(frame))

    height, width = cfg.image_size
    cache_dir = config.path("interim") / "classification_cache" / f"{height}x{width}"
    start = time.perf_counter()
    frame = cdata.build_image_cache(frame, cache_dir, cfg.image_size, cfg.cache_workers)
    LOGGER.info("Image cache ready in %.1f s (%s)", time.perf_counter() - start, cache_dir)
    cdata.validate_inventory(frame, cfg.source_classes)

    frame = cdata.subsample(frame, cfg.data_fraction, config.seed)
    splits = cdata.split_dataset(frame, cfg.val_fraction, cfg.test_fraction, config.seed)
    for name, split in splits.items():
        LOGGER.info("%-5s: %d images", name, len(split))
    return splits


def _fit(model, train_ds, val_ds, class_weights, config: Config, start_epoch: int):
    import keras

    cfg = config.classification
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", **cfg.reduce_lr),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(cfg.early_stopping["patience"]),
            min_delta=cfg.early_stopping["min_delta"],
            restore_best_weights=True,
            start_from_epoch=start_epoch,
        ),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.max_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        shuffle=False,  # the tf.data pipeline already shuffles the training set
        verbose=2,
    )
    return pd.DataFrame(history.history)


def predict_probabilities(model, dataset) -> np.ndarray:
    return model.predict(dataset, verbose=0).reshape(-1)


def train_classifier(config: Config, run_dir: Path) -> dict:
    """Run the full protocol and write model, metrics and splits into ``run_dir``."""
    configure_gpu_memory_growth()
    cfg = config.classification
    splits = prepare_splits(config)
    for name, split in splits.items():
        write_csv(run_dir / "splits" / f"{name}.csv", split[["path", "source_class", "label"]])

    datasets = {
        "train": cdata.make_dataset(splits["train"], cfg.image_size, cfg.batch_size, True, config.seed),
        "val": cdata.make_dataset(splits["val"], cfg.image_size, cfg.batch_size),
        "test": cdata.make_dataset(splits["test"], cfg.image_size, cfg.batch_size),
    }
    class_weights = compute_class_weights(splits["train"]["label"].to_numpy())
    LOGGER.info("Class weights (train only): %s", class_weights)

    y_val = splits["val"]["label"].to_numpy()
    models, val_rows, val_probabilities = {}, [], {}
    for candidate in cfg.candidates:
        set_seed(config.seed)
        LOGGER.info("Training candidate '%s' (%s)", candidate.name, candidate.variant)
        model = build_classifier(
            candidate.variant,
            cfg.image_size,
            cfg.learning_rate,
            cfg.clipnorm,
            cfg.enriched_augmentation,
            name=candidate.name,
        )
        start = time.perf_counter()
        history = _fit(
            model,
            datasets["train"],
            datasets["val"],
            class_weights,
            config,
            candidate.early_stopping_start_epoch,
        )
        LOGGER.info("'%s' trained in %.1f s", candidate.name, time.perf_counter() - start)
        write_csv(run_dir / "histories" / f"{candidate.name}.csv", history, index=True)

        probabilities = predict_probabilities(model, datasets["val"])
        metrics = compute_metrics(y_val, probabilities, threshold=0.5)
        plateau = plateau_analysis(history["val_loss"])
        val_rows.append({"model": candidate.name, "variant": candidate.variant, **metrics, **plateau})
        models[candidate.name] = model
        val_probabilities[candidate.name] = probabilities

    comparison = pd.DataFrame(val_rows).set_index("model")
    write_csv(run_dir / "validation_comparison.csv", comparison, index=True)
    best_name = str(comparison["f1_macro"].idxmax())
    best_model = models[best_name]
    LOGGER.info("Selected on validation F1 macro: %s", best_name)

    grid = cfg.threshold_grid
    threshold, best_f1, curve = select_threshold(
        y_val, val_probabilities[best_name], grid["start"], grid["stop"], grid["num"]
    )
    write_csv(run_dir / "threshold_curve.csv", curve)
    LOGGER.info("Threshold tuned on validation: %.2f (F1 macro %.4f)", threshold, best_f1)

    # ---- single, final evaluation on the test set ---------------------------
    test_probabilities = predict_probabilities(best_model, datasets["test"])
    y_test = splits["test"]["label"].to_numpy()
    results = {
        "selected_model": best_name,
        "decision_threshold": threshold,
        "validation": compute_metrics(y_val, val_probabilities[best_name], threshold),
        "test": compute_metrics(y_test, test_probabilities, threshold),
        "constant_other_accuracy_test": float((y_test == 0).mean()),
    }
    write_json(run_dir / "metrics.json", results)
    write_csv(
        run_dir / "test_confusion_matrix.csv",
        confusion_frame(y_test, test_probabilities, threshold),
        index=True,
    )
    for split_name, frame, probs in [
        ("val", splits["val"], val_probabilities[best_name]),
        ("test", splits["test"], test_probabilities),
    ]:
        write_csv(
            run_dir / f"{split_name}_by_source_class.csv",
            source_class_analysis(frame, probs, threshold, cfg.positive_class, cfg.source_classes),
        )
        predictions = frame[["path", "source_class", "label"]].copy()
        predictions["probability_photo"] = probs
        write_csv(run_dir / f"{split_name}_predictions.csv", predictions)

    best_model.save(run_dir / MODEL_FILENAME)
    write_json(
        run_dir / METADATA_FILENAME,
        {
            "model_name": best_name,
            "image_size": list(cfg.image_size),
            "decision_threshold": threshold,
            "classes": {"0": "Other", "1": cfg.positive_class},
            "source_classes": cfg.source_classes,
            "seed": config.seed,
        },
    )
    LOGGER.info("Test F1 macro: %.4f | artifacts: %s", results["test"]["f1_macro"], run_dir)
    return results
