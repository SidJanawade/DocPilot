"""LLM backends: Anthropic when ANTHROPIC_API_KEY is set, otherwise an offline extractive fallback."""
from __future__ import annotations
import os
import re

from .config import LLM_MODEL
from .retrieval import tokenize

SYSTEM = (
    "You answer questions using ONLY the numbered context passages. Cite passages like [1]. "
    "If the context does not contain the answer, reply exactly: I don't know based on the docs."
)


class ExtractiveLLM:
    """Picks the best sentences from the context. Deterministic and free; used for tests/CI."""

    name = "extractive"

    def generate(self, question: str, contexts: list[str]) -> str:
        qt = set(tokenize(question))
        scored = []
        for i, ctx in enumerate(contexts, 1):
            for sent in re.split(r"(?<=[.!?])\s+", ctx):
                scored.append((len(qt & set(tokenize(sent))) - 0.05 * i, i, sent))
        best = sorted(scored, key=lambda x: -x[0])[:2]
        return " ".join(f"{s} [{i}]" for _, i, s in best) if best else "I don't know based on the docs."


class AnthropicLLM:
    name = "anthropic"

    def __init__(self):
        import anthropic

        self.client = anthropic.Anthropic()

    def generate(self, question: str, contexts: list[str]) -> str:
        ctx = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
        msg = self.client.messages.create(
            model=LLM_MODEL,
            max_tokens=500,
            system=SYSTEM,
            messages=[{"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")


def get_llm():
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return AnthropicLLM()
        except ImportError:
            pass
    return ExtractiveLLM()
