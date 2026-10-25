"""BLEU scores against the 5 human captions of each image (NLTK)."""

from __future__ import annotations

from collections.abc import Sequence

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu

_SMOOTHING = SmoothingFunction().method1


def corpus_bleu_scores(
    references: Sequence[Sequence[Sequence[str]]], hypotheses: Sequence[Sequence[str]]
) -> dict[str, float]:
    """Corpus BLEU-1 to BLEU-4 (uniform n-gram weights, smoothing method 1)."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    return {
        f"BLEU-{n}": float(
            corpus_bleu(references, hypotheses, weights=tuple([1 / n] * n), smoothing_function=_SMOOTHING)
        )
        for n in (1, 2, 3, 4)
    }


def sentence_bleu4(references: Sequence[Sequence[str]], hypothesis: Sequence[str]) -> float:
    return float(sentence_bleu(references, hypothesis, smoothing_function=_SMOOTHING))
