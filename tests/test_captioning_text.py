import numpy as np
import pytest

from tounum.captioning.text import (
    Vocabulary,
    add_markers,
    clean_caption,
    max_caption_length,
    to_padded_sequences,
)

TEXTS = [
    add_markers(t)
    for t in [
        "A dog runs on the grass.",
        "A dog and a cat!",
        "Two people on the beach",
        "A cat, sleeping on a bed",
        "a red bus on the street",
    ]
]


def test_clean_caption():
    assert clean_caption("  A Dog's  ball, (red)! ") == "a dog s ball red"
    assert add_markers("Hi!") == "<start> hi <end>"


def test_vocabulary_matches_legacy_keras_tokenizer():
    """Same indices as keras.preprocessing.text.Tokenizer used by the notebook."""
    text_module = pytest.importorskip("tensorflow.keras.preprocessing.text")
    tokenizer = text_module.Tokenizer(num_words=8, oov_token="<unk>", filters="")
    tokenizer.fit_on_texts(TEXTS)
    vocab = Vocabulary.fit(TEXTS, num_words=8)
    for word, index in tokenizer.word_index.items():
        assert vocab.word_index[word] == index
    for text in TEXTS + ["<start> a zebra on the moon <end>"]:
        assert vocab.encode(text) == tokenizer.texts_to_sequences([text])[0]


def test_special_tokens_and_rare_words():
    vocab = Vocabulary.fit(TEXTS, num_words=6)
    assert vocab.word_index["<pad>"] == 0 and vocab.unk_id == 1
    assert vocab.size == 7
    assert all(i < 6 for text in TEXTS for i in vocab.encode(text))
    assert vocab.encode("zebra") == [vocab.unk_id]


def test_decode_stops_at_end_and_drops_padding():
    vocab = Vocabulary.fit(TEXTS, num_words=50)
    ids = vocab.encode("<start> a dog <end> cat") + [0, 0]
    assert vocab.decode(ids[1:]) == ["a", "dog"]


def test_serialisation_roundtrip():
    vocab = Vocabulary.fit(TEXTS, num_words=10)
    restored = Vocabulary.from_dict(vocab.to_dict())
    assert restored.start_id == vocab.start_id and restored.end_id == vocab.end_id
    assert restored.encode(TEXTS[0]) == vocab.encode(TEXTS[0])


def test_padding_and_length_filter():
    vocab = Vocabulary.fit(TEXTS, num_words=50)
    sequences = to_padded_sequences(vocab, TEXTS, max_len=7)
    assert sequences.shape == (2, 7) and sequences.dtype == np.int32  # 3 captions are too long
    sequences = to_padded_sequences(vocab, TEXTS, max_len=9)
    assert len(sequences) == len(TEXTS)
    assert (sequences[:, -1] == 0).any()
    assert max_caption_length([5, 6, 7, 30], percentile=50) == 6


def test_from_index_word_rejects_nothing_and_keeps_ids():
    vocab = Vocabulary.from_index_word(
        {"0": "<pad>", "1": "<unk>", "2": "a", "3": "<start>", "4": "<end>"}, 5
    )
    assert vocab.encode("<start> a b <end>") == [3, 2, 1, 4]
    with pytest.raises(KeyError):
        Vocabulary.from_index_word({"1": "x"}, 2).start_id  # noqa: B018
