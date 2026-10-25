"""Caption new images with a trained model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tounum.captioning import features as feats
from tounum.captioning.model import build_caption_model, generate
from tounum.captioning.text import Vocabulary
from tounum.captioning.train import DECODER_WEIGHTS, ENCODER_WEIGHTS, METADATA_FILENAME
from tounum.config import CaptionModelConfig
from tounum.utils.runs import read_json


class Captioner:
    def __init__(
        self, encoder, decoder, extractor, vocab: Vocabulary, max_len: int, cnn: str, decoding="greedy"
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.extractor = extractor
        self.vocab = vocab
        self.max_len = max_len
        self.cnn = cnn
        self.decoding = decoding

    @classmethod
    def from_run(cls, run_dir: str | Path, extractor=None) -> Captioner:
        """Rebuild the model from its metadata and load the trained weights."""
        run_dir = Path(run_dir)
        metadata = read_json(run_dir / METADATA_FILENAME)
        model_cfg = CaptionModelConfig(**metadata["model"])
        vocab = Vocabulary.from_dict(metadata["vocabulary"])
        encoder, decoder = build_caption_model(
            model_cfg, metadata["vocab_size"], feats.feature_channels(model_cfg.cnn)
        )
        for model, filename in [(encoder, ENCODER_WEIGHTS), (decoder, DECODER_WEIGHTS)]:
            path = run_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing weights: {path}")
            model.load_weights(path)
        if extractor is None:
            extractor = feats.create_extractor(model_cfg.cnn)
        return cls(
            encoder,
            decoder,
            extractor,
            vocab,
            metadata["max_len"],
            model_cfg.cnn,
            metadata.get("decoding", "greedy"),
        )

    def features_from_arrays(self, images: list[np.ndarray]) -> np.ndarray:
        import tensorflow as tf

        batch = tf.stack([feats.resize_array_299(np.asarray(img)) for img in images])
        return feats.extract(self.extractor, feats.preprocess_images(batch, self.cnn))

    def caption_arrays(self, images: list[np.ndarray]) -> list[str]:
        """uint8 RGB arrays -> captions (same preprocessing as the feature cache)."""
        if not images:
            return []
        ids, _ = generate(
            self.encoder,
            self.decoder,
            self.features_from_arrays(images),
            self.vocab.start_id,
            self.max_len,
            self.decoding,
        )
        return [" ".join(self.vocab.decode(row)) for row in ids]

    def caption_paths(self, paths: list[str | Path]) -> list[str]:
        """Image files -> captions, decoded exactly like the training feature cache."""
        import tensorflow as tf

        if not paths:
            return []
        batch = tf.stack([feats.read_image_299(str(p)) for p in paths])
        features = feats.extract(self.extractor, feats.preprocess_images(batch, self.cnn))
        ids, _ = generate(
            self.encoder, self.decoder, features, self.vocab.start_id, self.max_len, self.decoding
        )
        return [" ".join(self.vocab.decode(row)) for row in ids]
