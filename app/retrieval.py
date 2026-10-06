"""Hybrid retrieval: BM25 + dense vectors fused with Reciprocal Rank Fusion, plus a reranker."""
from __future__ import annotations
import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass

from .ingest import Chunk

STOP = frozenset(
    "a an the is are was were be to of in on for and or it its this that with as at by from how do does "
    "what which when where why who can i my you your should shouldn don doesn t s".split()
)
_TOKEN = re.compile(r"[a-z0-9_]+")


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us")) else w


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in _TOKEN.findall(text.lower()) if w not in STOP and len(w) > 1]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / max(len(docs), 1)
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            for t in query:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


class HashEmbedder:
    """Dependency-free hashed uni+bigram vectors. Offline default; swap for STEmbedder for semantics."""

    def __init__(self, dim: int = 1 << 14):
        self.dim = dim

    def _h(self, s: str) -> int:
        return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % self.dim

    def embed(self, text: str):
        toks = tokenize(text)
        feats = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
        vec = {k: 1 + math.log(c) for k, c in Counter(self._h(f) for f in feats).items()}
        norm = math.sqrt(sum(x * x for x in vec.values())) or 1.0
        return {k: x / norm for k, x in vec.items()}

    def sim(self, a, b) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(x * b.get(k, 0.0) for k, x in a.items())


class STEmbedder:
    """Real semantic embeddings (pip install sentence-transformers)."""

    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name)

    def embed(self, text: str):
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def sim(self, a, b) -> float:
        return float(sum(x * y for x, y in zip(a, b)))


def get_embedder(name: str):
    return STEmbedder() if name == "st" else HashEmbedder()


def query_coverage(query_tokens: list[str], text: str) -> float:
    q = set(query_tokens)
    return len(q & set(tokenize(text))) / len(q) if q else 0.0


@dataclass
class Hit:
    chunk: Chunk
    score: float


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder):
        self.chunks, self.emb = chunks, embedder
        self.bm25 = BM25([tokenize(c.heading + " " + c.text) for c in chunks])
        self.vecs = [embedder.embed(c.heading + " " + c.text) for c in chunks]

    @staticmethod
    def _ranks(scores):
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return {i: r for r, i in enumerate(order)}

    def search(self, query: str, k: int = 4, mode: str = "hybrid", rerank: bool = False) -> list[Hit]:
        qt = tokenize(query)
        bm = self.bm25.scores(qt)
        qv = self.emb.embed(query)
        dn = [self.emb.sim(qv, v) for v in self.vecs]
        if mode == "bm25":
            fused = bm
        elif mode == "dense":
            fused = dn
        else:  # Reciprocal Rank Fusion
            rb, rd = self._ranks(bm), self._ranks(dn)
            fused = [1 / (60 + rb[i]) + 1 / (60 + rd[i]) for i in range(len(bm))]
        pool = max(k, 15) if rerank else k
        order = sorted(range(len(fused)), key=lambda i: -fused[i])[:pool]
        hits = [Hit(self.chunks[i], fused[i]) for i in order]
        return self._rerank(qt, hits)[:k] if rerank else hits

    @staticmethod
    def _rerank(qt: list[str], hits: list[Hit]) -> list[Hit]:
        """Cheap reranker: blend fused score with query-term coverage + bigram match."""
        if not hits:
            return hits
        top = max(h.score for h in hits) or 1.0
        bigrams = {a + "_" + b for a, b in zip(qt, qt[1:])}
        rescored = []
        for h in hits:
            toks = tokenize(h.chunk.text)
            have = {a + "_" + b for a, b in zip(toks, toks[1:])}
            bonus = len(bigrams & have) / len(bigrams) if bigrams else 0.0
            cov = query_coverage(qt, h.chunk.heading + " " + h.chunk.text)
            rescored.append(Hit(h.chunk, 0.5 * h.score / top + 0.35 * cov + 0.15 * bonus))
        return sorted(rescored, key=lambda h: -h.score)
