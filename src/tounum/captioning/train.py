"""Training of the captioning model (notebook *Livrable 3*).

* the vocabulary (5 000 most frequent words) and the maximum caption length
  (99th percentile) are computed on the **training captions only**;
* training uses teacher forcing and a masked sparse cross-entropy;
* after each epoch the validation loss and the validation **BLEU-4**
  (greedy captions written by the model alone) are computed;
* early stopping: training stops when the validation BLEU-4 has not improved
  for ``patience`` epochs and the weights of the best epoch are restored;
* the test set is evaluated **once**, after the model is frozen.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tounum.captioning import features as feats
from tounum.captioning.data import coco_filename, load_captions, split_image_ids
from tounum.captioning.metrics import corpus_bleu_scores
from tounum.captioning.model import N_REGIONS, build_caption_model, generate
from tounum.captioning.text import (
    Vocabulary,
    add_markers,
    clean_caption,
    max_caption_length,
    to_padded_sequences,
)
from tounum.config import CaptionModelConfig, Config
from tounum.utils.logging_config import get_logger
from tounum.utils.reproducibility import configure_gpu_memory_growth, set_seed
from tounum.utils.runs import write_csv, write_json

LOGGER = get_logger("captioning.train")

ENCODER_WEIGHTS = "encoder.weights.h5"
DECODER_WEIGHTS = "decoder.weights.h5"
METADATA_FILENAME = "captioner_metadata.json"


@dataclass
class CaptionData:
    """Text side of the dataset, ready for training."""

    vocab: Vocabulary
    max_len: int
    sequences: dict[int, np.ndarray]  # image_id -> (n_captions, max_len)
    references: dict[int, list[list[str]]]  # image_id -> cleaned human captions


def prepare_text(
    captions: dict[int, list[str]], train_ids, all_ids, vocab_size: int, percentile: float
) -> CaptionData:
    """Markers, vocabulary and length limit fitted on the training captions only."""
    marked = {i: [add_markers(c) for c in captions[i]] for i in all_ids}
    train_texts = [c for i in train_ids for c in marked[i]]
    vocab = Vocabulary.fit(train_texts, vocab_size)
    max_len = max_caption_length([len(c.split()) for c in train_texts], percentile)
    sequences = {i: to_padded_sequences(vocab, marked[i], max_len) for i in all_ids}
    references = {i: [clean_caption(c).split() for c in captions[i]] for i in all_ids}
    return CaptionData(vocab, max_len, sequences, references)


def masked_loss(real, pred):
    """Sparse categorical cross-entropy ignoring ``<pad>`` (id 0)."""
    import keras
    import tensorflow as tf

    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")(real, pred)
    mask = tf.cast(tf.math.not_equal(real, 0), loss.dtype)
    return tf.reduce_mean(loss * mask)


def training_dataset(
    image_ids, sequences, cnn: str, cache_dir: Path, batch_size: int, max_len: int, seed: int
):
    """One (features, caption) pair per caption, shuffled, read from the feature cache."""
    import tensorflow as tf

    paths, targets = [], []
    for image_id in image_ids:
        for sequence in sequences[image_id]:
            paths.append(str(feats.feature_path(cache_dir, cnn, image_id)))
            targets.append(sequence)
    if not targets:
        raise ValueError("No training caption.")
    channels = feats.feature_channels(cnn)

    def load(path, target):
        return np.load(path.decode()).astype("float32"), target

    def set_shapes(f, c):
        f.set_shape((N_REGIONS, channels))
        c.set_shape((max_len,))
        return f, c

    ds = tf.data.Dataset.from_tensor_slices((paths, np.asarray(targets, dtype="int32")))
    ds = ds.shuffle(len(targets), seed=seed)
    ds = ds.map(
        lambda p, c: tf.numpy_function(load, [p, c], [tf.float32, tf.int32]),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.map(set_shapes)
    # drop_remainder: constant batch shape, train_step is traced only once
    return ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE), len(targets)


def write_captions(
    encoder, decoder, features, vocab: Vocabulary, max_len: int, mode="greedy", batch_size=250
):
    """Captions (lists of words) and attention maps for a set of images."""
    hypotheses, attentions = [], []
    for start in range(0, len(features), batch_size):
        ids, attention = generate(
            encoder, decoder, features[start : start + batch_size], vocab.start_id, max_len, mode, seed=start
        )
        hypotheses += [vocab.decode(row) for row in ids]
        attentions.append(attention)
    return hypotheses, np.concatenate(attentions) if attentions else np.empty((0, max_len, 64))


def bleu_for(encoder, decoder, features, image_ids, text: CaptionData, mode="greedy", batch_size=250):
    hypotheses, _ = write_captions(encoder, decoder, features, text.vocab, text.max_len, mode, batch_size)
    return corpus_bleu_scores([text.references[i] for i in image_ids], hypotheses), hypotheses


def fit_captioner(
    model_cfg: CaptionModelConfig,
    text: CaptionData,
    train_ids,
    val_ids,
    val_features: np.ndarray,
    cache_dir: Path,
    max_epochs: int,
    patience: int,
    seed: int,
    generation_batch_size: int = 250,
):
    """Custom training loop with early stopping on validation BLEU-4."""
    import keras
    import tensorflow as tf

    set_seed(seed)
    channels = feats.feature_channels(model_cfg.cnn)
    encoder, decoder = build_caption_model(model_cfg, text.vocab.size, channels)
    optimizer = keras.optimizers.Adam(learning_rate=model_cfg.learning_rate)
    dataset, n_captions = training_dataset(
        train_ids, text.sequences, model_cfg.cnn, cache_dir, model_cfg.batch_size, text.max_len, seed
    )
    LOGGER.info("Training on %d captions of %d images", n_captions, len(train_ids))
    start_id, max_len = text.vocab.start_id, text.max_len

    def batch_loss(images, target, training):
        loss = 0.0
        encoded = encoder(images)
        state = decoder.initial_rnn_state(encoded)
        word = tf.expand_dims(tf.fill([tf.shape(target)[0]], start_id), 1)
        for i in range(1, max_len):
            predictions, state, _ = decoder(word, encoded, state, training=training)
            loss += masked_loss(target[:, i], predictions)
            word = tf.expand_dims(target[:, i], 1)  # teacher forcing
        return loss / (max_len - 1)

    @tf.function(reduce_retracing=True)
    def train_step(images, target):
        with tf.GradientTape() as tape:
            loss = batch_loss(images, target, training=True)
        variables = encoder.trainable_variables + decoder.trainable_variables
        optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables, strict=True))
        return loss

    @tf.function(reduce_retracing=True)
    def val_step(images, target):
        return batch_loss(images, target, training=False)

    val_pairs = [(k, s) for k, i in enumerate(val_ids) for s in text.sequences[i]]
    history = {"loss": [], "val_loss": [], "bleu4_val": []}
    best, best_weights, waited = -1.0, None, 0
    for epoch in range(max_epochs):
        tic = time.perf_counter()
        losses = [float(train_step(images, target)) for images, target in dataset]
        if not losses:
            raise ValueError("Training set smaller than one batch; lower batch_size.")
        val_losses = []
        for k in range(0, len(val_pairs), 256):
            chunk = val_pairs[k : k + 256]
            images = tf.constant(val_features[[a for a, _ in chunk]], tf.float32)
            val_losses.append(float(val_step(images, tf.constant(np.stack([s for _, s in chunk])))))
        bleu4 = bleu_for(encoder, decoder, val_features, val_ids, text, batch_size=generation_batch_size)[0][
            "BLEU-4"
        ]
        history["loss"].append(float(np.mean(losses)))
        history["val_loss"].append(float(np.mean(val_losses)) if val_losses else float("nan"))
        history["bleu4_val"].append(bleu4)
        LOGGER.info(
            "Epoch %2d | loss %.3f | val loss %.3f | val BLEU-4 %.3f | %.0f s",
            epoch + 1,
            history["loss"][-1],
            history["val_loss"][-1],
            bleu4,
            time.perf_counter() - tic,
        )
        if bleu4 > best:
            best, waited = bleu4, 0
            best_weights = (encoder.get_weights(), decoder.get_weights())
        else:
            waited += 1
            if waited >= patience:
                LOGGER.info("Early stopping: no BLEU-4 improvement for %d epochs", patience)
                break
    encoder.set_weights(best_weights[0])
    decoder.set_weights(best_weights[1])
    return encoder, decoder, pd.DataFrame(history)


def train_captioner(config: Config, run_dir: Path, extractor=None) -> dict:
    """Full protocol: split, text, features, training, validation, single test evaluation.

    ``extractor`` can be injected (tests); by default the ImageNet CNN of the
    configuration is created, which downloads its weights on first use.
    """
    configure_gpu_memory_growth()
    cfg, model_cfg = config.captioning, config.captioning.model
    captions = load_captions(config.path("coco_annotations"))
    images_dir = config.path("coco_images")
    splits = split_image_ids(captions, images_dir, cfg.n_test, cfg.n_val, cfg.n_train, config.seed)
    LOGGER.info(
        "Images: train %d | val %d | test %d (available %d)",
        len(splits.train),
        len(splits.val),
        len(splits.test),
        splits.available,
    )
    all_ids = splits.train + splits.val + splits.test
    text = prepare_text(captions, splits.train, all_ids, cfg.vocab_size, cfg.max_len_percentile)
    LOGGER.info("Vocabulary: %d words kept | max caption length: %d", cfg.vocab_size, text.max_len)

    cache_dir = config.path("interim") / "caption_features"
    if extractor is None:
        extractor = feats.create_extractor(model_cfg.cnn)
    feats.cache_features(
        all_ids,
        [str(images_dir / coco_filename(i)) for i in all_ids],
        model_cfg.cnn,
        cache_dir,
        extractor,
        cfg.feature_batch_size,
    )
    del extractor  # not needed during training: free memory

    val_features = feats.load_features(splits.val, model_cfg.cnn, cache_dir)
    encoder, decoder, history = fit_captioner(
        model_cfg,
        text,
        splits.train,
        splits.val,
        val_features,
        cache_dir,
        cfg.max_epochs,
        cfg.patience,
        config.seed,
        cfg.generation_batch_size,
    )
    write_csv(run_dir / "history.csv", history, index=True)

    gen_batch = cfg.generation_batch_size
    val_scores = {
        mode: bleu_for(encoder, decoder, val_features, splits.val, text, mode, gen_batch)[0]
        for mode in ("greedy", "sample")
    }
    control_ids = splits.train[: cfg.n_val]
    train_scores = bleu_for(
        encoder,
        decoder,
        feats.load_features(control_ids, model_cfg.cnn, cache_dir),
        control_ids,
        text,
        batch_size=gen_batch,
    )[0]

    # ---- single, final evaluation on the test set ---------------------------
    test_features = feats.load_features(splits.test, model_cfg.cnn, cache_dir)
    test_scores, test_hyp = bleu_for(
        encoder, decoder, test_features, splits.test, text, cfg.decoding, gen_batch
    )
    write_csv(
        run_dir / "test_captions.csv",
        pd.DataFrame(
            {
                "image_id": splits.test,
                "generated": [" ".join(h) for h in test_hyp],
                "human_1": [captions[i][0] for i in splits.test],
            }
        ),
    )
    results = {
        "model": asdict(model_cfg),
        "train_images": len(splits.train),
        "best_epoch": int(history["bleu4_val"].idxmax()) + 1,
        "epochs_run": len(history),
        "validation": val_scores,
        "train_control": train_scores,
        "test": test_scores,
        "decoding": cfg.decoding,
    }
    write_json(run_dir / "metrics.json", results)
    write_json(run_dir / "splits.json", {"train": splits.train, "val": splits.val, "test": splits.test})

    encoder.save_weights(run_dir / ENCODER_WEIGHTS)
    decoder.save_weights(run_dir / DECODER_WEIGHTS)
    write_json(
        run_dir / METADATA_FILENAME,
        {
            "model": asdict(model_cfg),
            "vocabulary": text.vocab.to_dict(),
            "vocab_size": text.vocab.size,
            "max_len": text.max_len,
            "decoding": cfg.decoding,
        },
    )
    LOGGER.info("Test BLEU-4 %.3f | artifacts: %s", test_scores["BLEU-4"], run_dir)
    return results
