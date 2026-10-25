"""Caption text preprocessing: cleaning, vocabulary, sequences.

``Vocabulary`` re-implements the behaviour of the (legacy, Keras 2)
``keras.preprocessing.text.Tokenizer(num_words=TOP_K, oov_token="<unk>",
filters="")`` used in the notebook, so that indices are identical, while
being plain Python and JSON-serialisable:

* index 0 is ``<pad>``, index 1 is ``<unk>``;
* the other words are indexed by decreasing frequency (ties: first seen first);
* only indices ``< num_words`` are emitted; rarer words become ``<unk>``.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Iterable, Sequence

import numpy as np

PAD, UNK, START, END = "<pad>", "<unk>", "<start>", "<end>"


def clean_caption(text: str) -> str:
    """Lower-case and replace every non alphanumeric character with a space."""
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return " ".join(text.split())


def add_markers(text: str) -> str:
    return f"{START} {clean_caption(text)} {END}"


class Vocabulary:
    def __init__(self, num_words: int, word_index: dict[str, int]) -> None:
        self.num_words = int(num_words)
        self.word_index = dict(word_index)
        self.word_index[PAD] = 0
        self.index_word = {index: word for word, index in self.word_index.items()}

    # -- construction ---------------------------------------------------------
    @classmethod
    def fit(cls, texts: Iterable[str], num_words: int) -> Vocabulary:
        counts: OrderedDict[str, int] = OrderedDict()
        for text in texts:
            for word in text.lower().split():
                counts[word] = counts.get(word, 0) + 1
        ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)  # stable
        words = [UNK] + [word for word, _ in ordered if word != UNK]
        return cls(num_words, {word: index for index, word in enumerate(words, start=1)})

    @classmethod
    def from_index_word(cls, index_word: dict[int | str, str], num_words: int) -> Vocabulary:
        """Rebuild from a saved ``index_word`` mapping (e.g. the notebook's JSON)."""
        word_index = {word: int(index) for index, word in index_word.items() if int(index) != 0}
        return cls(num_words, word_index)

    # -- properties -----------------------------------------------------------
    @property
    def size(self) -> int:
        """Size of the embedding / output layer (``num_words + 1``, as in the notebook)."""
        return self.num_words + 1

    @property
    def unk_id(self) -> int:
        return self.word_index[UNK]

    @property
    def start_id(self) -> int:
        return self.word_index[START]

    @property
    def end_id(self) -> int:
        return self.word_index[END]

    # -- conversion -----------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        ids = []
        for word in text.lower().split():
            index = self.word_index.get(word)
            ids.append(self.unk_id if index is None or index >= self.num_words else index)
        return ids

    def decode(self, ids: Iterable[int]) -> list[str]:
        """Ids -> words, stopping at ``<end>`` and dropping ``<pad>``."""
        words = []
        for index in ids:
            index = int(index)
            if index == self.end_id:
                break
            if index != 0:
                words.append(self.index_word.get(index, UNK))
        return words

    def to_dict(self) -> dict:
        kept = {str(i): w for i, w in self.index_word.items() if i < self.num_words}
        return {"num_words": self.num_words, "index_word": kept}

    @classmethod
    def from_dict(cls, data: dict) -> Vocabulary:
        return cls.from_index_word(data["index_word"], data["num_words"])


def max_caption_length(tokenised_lengths: Sequence[int], percentile: float = 99) -> int:
    """Maximum caption length kept (percentile of the training lengths, markers included)."""
    return int(np.percentile(tokenised_lengths, percentile))


def to_padded_sequences(vocab: Vocabulary, texts: Sequence[str], max_len: int) -> np.ndarray:
    """Encode, drop captions longer than ``max_len`` and post-pad with 0."""
    encoded = [ids for ids in (vocab.encode(t) for t in texts) if len(ids) <= max_len]
    result = np.zeros((len(encoded), max_len), dtype=np.int32)
    for row, ids in enumerate(encoded):
        result[row, : len(ids)] = ids
    return result
