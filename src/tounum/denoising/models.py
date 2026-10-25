"""Convolutional autoencoders compared in notebook *Livrable 2*.

All models are fully convolutional (input ``(None, None, 3)``), so a model
trained on 128x128 crops can denoise a photo of any size whose sides are
multiples of 4. Output: sigmoid, 3 channels in [0, 1].
"""

from __future__ import annotations


def _layers():
    from keras import Input, Model
    from keras.layers import Concatenate, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D

    return Input, Model, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Concatenate


def build_a0_base():
    """Workshop autoencoder: 2 x (Conv 32 + MaxPool), 2 x (Conv 32 + UpSampling)."""
    Input, Model, Conv2D, MaxPooling2D, UpSampling2D, _, _ = _layers()
    inputs = Input(shape=(None, None, 3))
    x = Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = MaxPooling2D(2, padding="same")(x)
    x = Conv2D(32, 3, activation="relu", padding="same")(x)
    code = MaxPooling2D(2, padding="same")(x)
    x = Conv2D(32, 3, activation="relu", padding="same")(code)
    x = UpSampling2D(2)(x)
    x = Conv2D(32, 3, activation="relu", padding="same")(x)
    x = UpSampling2D(2)(x)
    outputs = Conv2D(3, 3, activation="sigmoid", padding="same")(x)
    return Model(inputs, outputs, name="A0_base")


def build_a1_wide():
    """Same structure as A0 with 64 / 128 filters."""
    Input, Model, Conv2D, MaxPooling2D, UpSampling2D, _, _ = _layers()
    inputs = Input(shape=(None, None, 3))
    x = Conv2D(64, 3, activation="relu", padding="same")(inputs)
    x = MaxPooling2D(2, padding="same")(x)
    x = Conv2D(128, 3, activation="relu", padding="same")(x)
    code = MaxPooling2D(2, padding="same")(x)
    x = Conv2D(128, 3, activation="relu", padding="same")(code)
    x = UpSampling2D(2)(x)
    x = Conv2D(64, 3, activation="relu", padding="same")(x)
    x = UpSampling2D(2)(x)
    outputs = Conv2D(3, 3, activation="sigmoid", padding="same")(x)
    return Model(inputs, outputs, name="A1_wide")


def build_a2_unet():
    """U-Net style skip connections (model retained in the notebook)."""
    Input, Model, Conv2D, MaxPooling2D, UpSampling2D, _, Concatenate = _layers()
    inputs = Input(shape=(None, None, 3))
    c1 = Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = MaxPooling2D(2, padding="same")(c1)
    c2 = Conv2D(64, 3, activation="relu", padding="same")(x)
    code = MaxPooling2D(2, padding="same")(c2)
    x = Conv2D(64, 3, activation="relu", padding="same")(code)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, c2])
    x = Conv2D(64, 3, activation="relu", padding="same")(x)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, c1])
    x = Conv2D(32, 3, activation="relu", padding="same")(x)
    outputs = Conv2D(3, 3, activation="sigmoid", padding="same")(x)
    return Model(inputs, outputs, name="A2_unet")


def build_a3_strided():
    """Learned down/up-sampling: strided Conv2D and Conv2DTranspose."""
    Input, Model, Conv2D, _, _, Conv2DTranspose, _ = _layers()
    inputs = Input(shape=(None, None, 3))
    x = Conv2D(32, 3, strides=2, activation="relu", padding="same")(inputs)
    code = Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
    x = Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same")(code)
    x = Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(x)
    outputs = Conv2D(3, 3, activation="sigmoid", padding="same")(x)
    return Model(inputs, outputs, name="A3_strided")


ARCHITECTURES = {
    "A0_base": build_a0_base,
    "A1_wide": build_a1_wide,
    "A2_unet": build_a2_unet,
    "A3_strided": build_a3_strided,
}


def build_denoiser(name: str):
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{name}'. Choose from {list(ARCHITECTURES)}.")
    return ARCHITECTURES[name]()
