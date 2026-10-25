"""CNN architectures compared in notebook *Livrable 1*.

* ``reference``   - 3 conv blocks, no augmentation, no dropout;
* ``regularized`` - same + flip/rotation/zoom augmentation + dropout;
* ``four_blocks`` - regularized + a 4th conv block (256 filters), which
  halves the dense layer and was the model retained in the notebook.
"""

from __future__ import annotations

VARIANTS = ("reference", "regularized", "four_blocks")


def build_classifier(
    variant: str,
    image_size: tuple[int, int] = (180, 180),
    learning_rate: float = 3e-4,
    clipnorm: float = 1.0,
    enriched_augmentation: bool = False,
    name: str | None = None,
):
    """Build and compile one of the notebook's CNN variants.

    The model takes raw pixels in [0, 255]; normalisation is part of the
    model so that training and prediction share it.
    """
    import keras
    from keras import layers

    if variant not in VARIANTS:
        raise ValueError(f"Unknown classifier variant '{variant}'. Choose from {VARIANTS}.")
    regularized = variant in {"regularized", "four_blocks"}

    model_layers: list = [layers.Input(shape=(*image_size, 3), name="image")]
    if regularized:
        augmentation = [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
        if enriched_augmentation:  # typical scanning defects (notebook option)
            augmentation += [
                layers.RandomTranslation(0.1, 0.1),
                layers.RandomBrightness(0.1),
                layers.RandomContrast(0.1),
            ]
        model_layers.append(keras.Sequential(augmentation, name="augmentation"))

    model_layers.append(layers.Rescaling(1.0 / 255.0, name="normalisation"))
    if regularized and enriched_augmentation:
        model_layers.append(layers.GaussianNoise(0.02, name="scan_noise"))

    filters = [32, 64, 128] + ([256] if variant == "four_blocks" else [])
    for index, n_filters in enumerate(filters, start=1):
        model_layers += [
            layers.Conv2D(n_filters, 3, padding="same", activation="relu", name=f"conv_{index}"),
            layers.MaxPooling2D(name=f"pool_{index}"),
        ]
    if regularized:
        model_layers.append(layers.Dropout(0.2, name="dropout_conv"))
    model_layers += [layers.Flatten(name="flatten"), layers.Dense(128, activation="relu", name="dense")]
    if regularized:
        model_layers.append(layers.Dropout(0.25, name="dropout_dense"))
    model_layers.append(layers.Dense(1, activation="sigmoid", dtype="float32", name="probability_photo"))

    model = keras.Sequential(model_layers, name=name or f"cnn_{variant}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model
