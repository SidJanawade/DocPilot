"""Semantic cache: reuse answers for near-duplicate questions."""
from __future__ import annotations
from collections import OrderedDict


class SemanticCache:
    def __init__(self, embedder, threshold: float = 0.9, max_size: int = 256):
        self.emb, self.threshold, self.max_size = embedder, threshold, max_size
        self.items: OrderedDict[str, tuple] = OrderedDict()
        self.hits = self.misses = 0

    def get(self, query: str):
        qv = self.emb.embed(query)
        best, best_sim = None, 0.0
        for key, (vec, _) in self.items.items():
            s = self.emb.sim(qv, vec)
            if s > best_sim:
                best, best_sim = key, s
        if best is not None and best_sim >= self.threshold:
            self.items.move_to_end(best)
            self.hits += 1
            return self.items[best][1]
        self.misses += 1
        return None

    def put(self, query: str, value: dict):
        self.items[query] = (self.emb.embed(query), value)
        while len(self.items) > self.max_size:
            self.items.popitem(last=False)
