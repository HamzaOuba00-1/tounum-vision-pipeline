import numpy as np
import pytest

from tounum.captioning import features as feats
from tounum.captioning.data import load_captions, split_image_ids
from tounum.captioning.metrics import corpus_bleu_scores, sentence_bleu4
from tounum.captioning.model import build_caption_model, generate
from tounum.captioning.predict import Captioner
from tounum.captioning.train import ENCODER_WEIGHTS, METADATA_FILENAME, masked_loss, train_captioner
from tounum.config import CaptionModelConfig
from tounum.utils.runs import create_run_dir, read_json


def test_load_and_split_per_image(coco_dir):
    captions = load_captions(coco_dir / "annotations" / "captions_train2014.json")
    assert len(captions[1]) == 5 and 999 in captions
    splits = split_image_ids(captions, coco_dir / "train2014", n_test=5, n_val=5, n_train=20, seed=42)
    assert splits.available == 40  # image 999 is not on disk
    sets = [set(splits.train), set(splits.val), set(splits.test)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    again = split_image_ids(captions, coco_dir / "train2014", 5, 5, 20, seed=42)
    assert again.test == splits.test
    with pytest.raises(ValueError, match="Requested"):
        split_image_ids(captions, coco_dir / "train2014", 5, 5, 100, seed=42)


def test_missing_annotations(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_captions(tmp_path / "missing.json")


def test_bleu():
    refs = [[["a", "dog", "on", "the", "grass"]], [["a", "cat", "on", "a", "bed"]]]
    perfect = corpus_bleu_scores(refs, [r[0] for r in refs])
    assert perfect["BLEU-4"] == pytest.approx(1.0)
    assert corpus_bleu_scores(refs, [["a"], ["b"]])["BLEU-4"] < 0.1
    assert sentence_bleu4(refs[0], refs[0][0]) == pytest.approx(1.0)


def test_masked_loss_ignores_padding():
    import tensorflow as tf

    logits = tf.random.normal((4, 10), seed=1)
    first_word_loss = tf.keras.losses.sparse_categorical_crossentropy(
        tf.constant([3]), logits[:1], from_logits=True
    )
    # only the first position is a real word: the 3 <pad> positions contribute 0 to the mean
    assert float(masked_loss(tf.constant([3, 0, 0, 0]), logits)) == pytest.approx(
        float(first_word_loss[0]) / 4, rel=1e-5
    )
    assert float(masked_loss(tf.constant([0, 0, 0, 0]), logits)) == 0.0


@pytest.mark.parametrize(
    "cnn, rnn, initial_state, n_states",
    [("InceptionV3", "gru", False, 1), ("InceptionResNetV2", "lstm", True, 2)],
)
def test_architectures_a_and_b(cnn, rnn, initial_state, n_states):
    cfg = CaptionModelConfig(cnn=cnn, rnn=rnn, initial_state=initial_state, units=16, embedding_dim=8)
    channels = feats.feature_channels(cnn)
    encoder, decoder = build_caption_model(cfg, vocab_size=30, feature_channels=channels)
    features = np.random.rand(3, 64, channels).astype("float32")
    encoded = encoder(features)
    state = decoder.initial_rnn_state(encoded)
    assert len(state) == n_states
    assert initial_state == (float(np.abs(state[0].numpy()).sum()) > 0)
    logits, _, attention = decoder(np.full((3, 1), 2), encoded, state)
    assert logits.shape == (3, 30) and attention.shape == (3, 64, 1)
    np.testing.assert_allclose(attention.numpy().sum(axis=1), 1.0, rtol=1e-5)

    ids_a, att = generate(encoder, decoder, features, start_id=2, max_len=5)
    ids_b, _ = generate(encoder, decoder, features, start_id=2, max_len=5)
    assert ids_a.shape == (3, 5) and att.shape == (3, 5, 64)
    np.testing.assert_array_equal(ids_a, ids_b)  # greedy decoding is deterministic
    assert generate(encoder, decoder, features, 2, 5, mode="sample")[0].shape == (3, 5)


def test_feature_cache(coco_dir, tmp_path, fake_extractor):
    ids = [1, 2, 3]
    paths = [str(coco_dir / "train2014" / f"COCO_train2014_{i:012d}.jpg") for i in ids]
    assert feats.cache_features(ids, paths, "InceptionResNetV2", tmp_path, fake_extractor, 2) == 3
    assert feats.cache_features(ids, paths, "InceptionResNetV2", tmp_path, fake_extractor, 2) == 0
    loaded = feats.load_features(ids, "InceptionResNetV2", tmp_path)
    assert loaded.shape == (3, 64, 1536) and loaded.dtype == np.float32
    with pytest.raises(FileNotFoundError):
        feats.load_features([42], "InceptionResNetV2", tmp_path)


def test_end_to_end_captioning(coco_dir, tiny_config, fake_extractor):
    run_dir = create_run_dir(tiny_config.path("artifacts"), "captioning", "test")
    results = train_captioner(tiny_config, run_dir, extractor=fake_extractor)
    assert results["train_images"] == 24
    assert set(results["test"]) == {"BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"}
    assert set(results["validation"]) == {"greedy", "sample"}
    for name in [ENCODER_WEIGHTS, METADATA_FILENAME, "history.csv", "test_captions.csv", "splits.json"]:
        assert (run_dir / name).is_file(), name
    splits = read_json(run_dir / "splits.json")
    assert not set(splits["train"]) & set(splits["test"])

    captioner = Captioner.from_run(run_dir, extractor=fake_extractor)
    image = (np.random.default_rng(0).uniform(size=(40, 50, 3)) * 255).astype(np.uint8)
    captions = captioner.caption_arrays([image, image])
    assert len(captions) == 2 and captions[0] == captions[1]
    path = coco_dir / "train2014" / "COCO_train2014_000000000001.jpg"
    assert isinstance(captioner.caption_paths([path])[0], str)


@pytest.mark.parametrize(
    "cfg, expected",
    [
        (CaptionModelConfig(cnn="InceptionV3", rnn="gru", initial_state=False, dropout=0.0), 6_603_658),
        (CaptionModelConfig(cnn="InceptionResNetV2", rnn="lstm", initial_state=True, dropout=0.0), 7_259_018),
        (CaptionModelConfig(cnn="InceptionV3", rnn="gru", initial_state=False, embedding_dim=512), 9_325_962),
    ],
)
def test_parameter_counts_match_notebook(cfg, expected):
    """Architecture A, B and tuned A: same trainable parameters as the notebook (vocabulary 5 001)."""
    encoder, decoder = build_caption_model(
        cfg, vocab_size=5001, feature_channels=feats.feature_channels(cfg.cnn)
    )
    assert encoder.count_params() + decoder.count_params() == expected
