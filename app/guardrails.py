"""Guardrails: refuse when retrieval is weak, flag answers not grounded in the context."""
from __future__ import annotations
import re

from .retrieval import tokenize


def retrieval_confident(coverages: list[float], min_coverage: float) -> bool:
    return bool(coverages) and max(coverages) >= min_coverage


def grounding_score(answer: str, contexts: list[str]) -> float:
    """Share of the answer's content words that appear in the retrieved context."""
    a = set(tokenize(re.sub(r"\[\d+\]", " ", answer)))
    if not a:
        return 0.0
    return len(a & set(tokenize(" ".join(contexts)))) / len(a)
