"""
Answer-level metrics for open QA — the standard SQuAD-style pair:

  - exact match (EM): after normalisation, the prediction is identical to
    the gold answer
  - token F1: overlap in words between prediction and gold, so a partially
    right answer ("different diets" vs. "diets") earns partial credit that
    EM would score as zero

Normalisation (lowercase, strip punctuation and the articles a/an/the,
collapse whitespace) is the one from the SQuAD evaluation script, so
"The Federalist Party." and "federalist party" count as the same answer.
"""
from __future__ import annotations

import re
import string
from collections import Counter

import numpy as np

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)  # both empty -> 1, one empty -> 0
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def is_abstention(prediction: str) -> bool:
    """The RAG prompt tells the model to say "I don't know" rather than guess."""
    p = str(prediction).lower()
    return any(phrase in p for phrase in ("don't know", "dont know", "do not know"))


def bootstrap_ci(values, n_resamples: int = 2000, seed: int = 42) -> tuple[float, float]:
    """95% percentile-bootstrap confidence interval for the mean of `values`."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
