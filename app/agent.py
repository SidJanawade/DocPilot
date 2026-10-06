"""Tool-using agent: routes between calculator, clarification, and the guarded RAG pipeline."""
from __future__ import annotations
import ast
import operator as op
import re
import time

from .guardrails import grounding_score, retrieval_confident
from .retrieval import query_coverage, tokenize

_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Mod: op.mod}
REFUSAL = "I don't know based on the docs."


def safe_eval(expr: str):
    """Evaluate arithmetic only (no names, calls, or attribute access)."""

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        raise ValueError("unsupported expression")

    return ev(ast.parse(expr.strip(), mode="eval").body)


def extract_math(q: str):
    m = re.search(r"(?<![\w.])(\(?\d[\d\s+\-*/%().]*\d\)?)(?![\w.])", q)
    return m.group(1) if m and re.search(r"[+\-*/%]", m.group(1)) else None


class Metrics:
    def __init__(self):
        self.requests = self.refusals = self.cache_hits = 0
        self.latencies: list[float] = []

    def snapshot(self) -> dict:
        lat = sorted(self.latencies)

        def pct(p):
            return round(lat[min(int(len(lat) * p), len(lat) - 1)], 2) if lat else 0.0

        return {
            "requests": self.requests,
            "refusals": self.refusals,
            "cache_hits": self.cache_hits,
            "latency_ms_p50": pct(0.5),
            "latency_ms_p95": pct(0.95),
        }


class Agent:
    def __init__(self, retriever, llm, cache, k: int = 4, min_coverage: float = 0.34):
        self.retriever, self.llm, self.cache = retriever, llm, cache
        self.k, self.min_coverage = k, min_coverage
        self.metrics = Metrics()

    def ask(self, question: str) -> dict:
        t0 = time.perf_counter()
        q = question.strip()
        if not q:
            raise ValueError("empty question")
        trace: list[str] = []
        out = self._route(q, trace)
        out["trace"] = trace
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        self.metrics.requests += 1
        self.metrics.latencies.append(out["latency_ms"])
        self.metrics.refusals += int(out.get("refused", False))
        self.metrics.cache_hits += int(out.get("cached", False))
        return out

    @staticmethod
    def _simple(answer: str, **kw) -> dict:
        return {"answer": answer, "sources": [], "refused": False, "cached": False, "grounded": True, **kw}

    def _route(self, q: str, trace: list[str]) -> dict:
        expr = extract_math(q)
        if expr:
            trace.append("tool:calculator")
            try:
                return self._simple(f"{expr.strip()} = {safe_eval(expr)}")
            except (ValueError, SyntaxError, ZeroDivisionError):
                trace.append("calculator_failed")
        if len(tokenize(q)) < 2:
            trace.append("clarify")
            return self._simple("Could you add more detail to your question?")

        cached = self.cache.get(q)
        if cached:
            trace.append("cache_hit")
            return {**cached, "cached": True}

        trace.append("retrieve:hybrid+rerank")
        hits = self.retriever.search(q, k=self.k, mode="hybrid", rerank=True)
        qt = tokenize(q)
        covs = [query_coverage(qt, h.chunk.heading + " " + h.chunk.text) for h in hits[:3]]
        if not retrieval_confident(covs, self.min_coverage):
            trace.append("guardrail:low_retrieval_confidence")
            return self._simple(REFUSAL, refused=True)

        contexts = [h.chunk.text for h in hits]
        trace.append(f"generate:{self.llm.name}")
        answer = self.llm.generate(q, contexts)
        score = grounding_score(answer, contexts)
        grounded = score >= 0.7 or answer.strip() == REFUSAL
        trace.append(f"guardrail:grounding={score:.2f}")
        if not grounded:
            answer = REFUSAL
        out = {
            "answer": answer,
            "sources": [{"doc": h.chunk.doc, "heading": h.chunk.heading, "score": round(h.score, 4)} for h in hits],
            "refused": answer == REFUSAL,
            "cached": False,
            "grounded": grounded,
        }
        if not out["refused"]:
            self.cache.put(q, out)
        return out
