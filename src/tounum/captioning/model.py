"""Encoder, Bahdanau attention and RNN decoder (notebook ``captioning_modele.py``).

Architecture A (workshop): InceptionV3 features, GRU, zero initial state.
Architecture B (final):    InceptionResNetV2 features, LSTM, initial state
computed from the mean of the 64 image regions.

The layer **attribute names** (``fc``, ``embedding``, ``cellule``, ``fc1``,
``fc2``, ``attention.W1/W2/V``, ``init_h``, ``init_c``) are kept identical to
the notebook so that weight files saved by the notebook can still be loaded.
"""

from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf
from keras import layers

from tounum.config import CaptionModelConfig

N_REGIONS = 64  # 8 x 8 feature grid


@keras.utils.register_keras_serializable(package="tounum")
class CNNEncoder(keras.Model):
    """Projects each of the 64 regions to ``embedding_dim`` values (Dense + ReLU)."""

    def __init__(self, embedding_dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.fc = layers.Dense(embedding_dim)

    def call(self, x):
        return tf.nn.relu(self.fc(x))

    def get_config(self):
        return {"embedding_dim": self.embedding_dim}


@keras.utils.register_keras_serializable(package="tounum")
class BahdanauAttention(keras.Model):
    """Additive attention: one weight per region, given the RNN state."""

    def __init__(self, units: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        scores = self.V(tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))
        weights = tf.nn.softmax(scores, axis=1)  # (batch, 64, 1), sums to 1
        context = tf.reduce_sum(weights * features, axis=1)
        return context, weights

    def get_config(self):
        return {"units": self.units}


@keras.utils.register_keras_serializable(package="tounum")
class RNNDecoder(keras.Model):
    """Writes the caption word by word (teacher forcing during training)."""

    def __init__(
        self,
        embedding_dim: int,
        units: int,
        vocab_size: int,
        rnn: str = "gru",
        initial_state: bool = False,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.decoder_config = {
            "embedding_dim": embedding_dim,
            "units": units,
            "vocab_size": vocab_size,
            "rnn": rnn,
            "initial_state": initial_state,
            "dropout": dropout,
        }
        self.units = units
        self.rnn = rnn
        self.use_initial_state = initial_state
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        recurrent = layers.GRU if rnn == "gru" else layers.LSTM
        # Attribute name kept from the notebook ("cellule") for weight compatibility.
        self.cellule = recurrent(
            units, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform"
        )
        self.fc1 = layers.Dense(units)
        self.dropout = layers.Dropout(dropout)
        self.fc2 = layers.Dense(vocab_size)
        self.attention = BahdanauAttention(units)
        if initial_state:
            self.init_h = layers.Dense(units, activation="tanh")
            if rnn == "lstm":
                self.init_c = layers.Dense(units, activation="tanh")

    def initial_rnn_state(self, features):
        """Zeros (architecture A) or a projection of the mean region (architecture B)."""
        batch = tf.shape(features)[0]
        if not self.use_initial_state:
            n_states = 1 if self.rnn == "gru" else 2
            return [tf.zeros((batch, self.units)) for _ in range(n_states)]
        summary = tf.reduce_mean(features, axis=1)
        state = [self.init_h(summary)]
        if self.rnn == "lstm":
            state.append(self.init_c(summary))
        return state

    def call(self, x, features, state, training=False):
        context, attention_weights = self.attention(features, state[0])
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        output, *state = self.cellule(x, initial_state=state)
        y = self.fc1(output)
        y = tf.reshape(y, (-1, y.shape[2]))
        y = self.dropout(y, training=training)
        return self.fc2(y), state, attention_weights

    def get_config(self):
        return dict(self.decoder_config)


def build_caption_model(config: CaptionModelConfig, vocab_size: int, feature_channels: int):
    """Create encoder + decoder and build their weights with a dummy call."""
    encoder = CNNEncoder(config.embedding_dim)
    decoder = RNNDecoder(
        config.embedding_dim, config.units, vocab_size, config.rnn, config.initial_state, config.dropout
    )
    features = encoder(tf.zeros((1, N_REGIONS, feature_channels)))
    decoder(tf.zeros((1, 1), tf.int32), features, decoder.initial_rnn_state(features))
    return encoder, decoder


def generate(
    encoder,
    decoder,
    features: np.ndarray,
    start_id: int,
    max_len: int,
    mode: str = "greedy",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Write captions from image features only (no reference caption).

    ``greedy`` takes the most probable word (deterministic, retained in the
    notebook); ``sample`` draws the word from the predicted distribution
    (workshop method). Returns word ids (n, max_len) and attention (n, max_len, 64).
    """
    if mode not in {"greedy", "sample"}:
        raise ValueError("mode must be 'greedy' or 'sample'")
    n = features.shape[0]
    encoded = encoder(tf.cast(features, tf.float32))
    state = decoder.initial_rnn_state(encoded)
    word = tf.fill([n, 1], start_id)
    ids, attentions = [], []
    for step in range(max_len):
        logits, state, attention = decoder(word, encoded, state)
        if mode == "greedy":
            next_word = tf.argmax(logits, axis=-1, output_type=tf.int32)
        else:
            next_word = tf.cast(tf.random.stateless_categorical(logits, 1, seed=[seed, step])[:, 0], tf.int32)
        ids.append(next_word)
        attentions.append(attention[..., 0])
        word = next_word[:, None]
    return tf.stack(ids, 1).numpy(), tf.stack(attentions, 1).numpy()
