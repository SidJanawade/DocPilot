"""Evaluation harness: retrieval quality per strategy + end-to-end refusal/grounding checks.

Usage: python evals/run_evals.py [--min-hit 0.8] [--min-refusal 0.9]
Exits non-zero if thresholds are missed, so CI can block quality regressions.
"""
from __future__ import annotations
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402
from app.agent import Agent  # noqa: E402
from app.cache import SemanticCache  # noqa: E402
from app.ingest import load_chunks  # noqa: E402
from app.llm import get_llm  # noqa: E402
from app.retrieval import HybridRetriever, get_embedder  # noqa: E402

K = 3
STRATEGIES = [("bm25", "bm25", False), ("dense", "dense", False), ("hybrid (RRF)", "hybrid", False), ("hybrid + rerank", "hybrid", True)]


def load_golden():
    return [json.loads(l) for l in (ROOT / "evals" / "golden_set.jsonl").read_text().splitlines() if l.strip()]


def is_hit(hit, item) -> bool:
    return hit.chunk.doc == item["doc"] and item["must"].lower() in hit.chunk.text.lower()


def eval_retrieval(retriever, golden):
    rows = []
    answerable = [g for g in golden if g["answerable"]]
    for name, mode, rerank in STRATEGIES:
        hits_at_k, rr, lat = 0, [], []
        for g in answerable:
            t0 = time.perf_counter()
            res = retriever.search(g["q"], k=K, mode=mode, rerank=rerank)
            lat.append((time.perf_counter() - t0) * 1000)
            rank = next((i for i, h in enumerate(res, 1) if is_hit(h, g)), None)
            hits_at_k += rank is not None
            rr.append(1 / rank if rank else 0.0)
        rows.append((name, hits_at_k / len(answerable), statistics.mean(rr), statistics.mean(lat)))
    return rows


def eval_end_to_end(agent, golden):
    ground, answered, refused_ok = [], 0, 0
    n_unans = n_ans = 0
    for g in golden:
        out = agent.ask(g["q"])
        if g["answerable"]:
            n_ans += 1
            if not out["refused"]:
                answered += 1
                ground.append(1.0 if out["grounded"] else 0.0)
        else:
            n_unans += 1
            refused_ok += out["refused"]
    return {
        "answer_rate": answered / n_ans,
        "grounded_rate": statistics.mean(ground) if ground else 0.0,
        "refusal_accuracy": refused_ok / n_unans,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-hit", type=float, default=0.8)
    ap.add_argument("--min-refusal", type=float, default=0.9)
    args = ap.parse_args()

    emb = get_embedder(config.EMBEDDER)
    chunks = load_chunks(config.DOCS_DIR)
    retriever = HybridRetriever(chunks, emb)
    golden = load_golden()

    rows = eval_retrieval(retriever, golden)
    agent = Agent(retriever, get_llm(), SemanticCache(emb, config.CACHE_THRESHOLD), config.TOP_K, config.MIN_COVERAGE)
    e2e = eval_end_to_end(agent, golden)

    # semantic cache effect: ask every answerable question again
    qs = [g["q"] for g in golden if g["answerable"]]
    fresh = Agent(retriever, get_llm(), SemanticCache(emb, config.CACHE_THRESHOLD), config.TOP_K, config.MIN_COVERAGE)
    cold_ms = statistics.mean(fresh.ask(q)["latency_ms"] for q in qs)
    warm_ms = statistics.mean(fresh.ask(q)["latency_ms"] for q in qs)

    lines = [
        f"Embedder: `{config.EMBEDDER}` | LLM: `{agent.llm.name}` | chunks: {len(chunks)} | questions: {len(golden)}",
        "",
        f"| Retrieval strategy | Hit@{K} | MRR | Avg latency (ms) |",
        "|---|---|---|---|",
        *[f"| {n} | {h:.2f} | {m:.2f} | {l:.2f} |" for n, h, m, l in rows],
        "",
        "| End-to-end metric | Score |",
        "|---|---|",
        f"| Answer rate (answerable Qs) | {e2e['answer_rate']:.2f} |",
        f"| Grounded answers | {e2e['grounded_rate']:.2f} |",
        f"| Refusal accuracy (unanswerable Qs) | {e2e['refusal_accuracy']:.2f} |",
        "",
        f"Semantic cache: cold {cold_ms:.2f} ms -> warm {warm_ms:.2f} ms avg per request",
    ]
    report = "\n".join(lines)
    print(report)
    (ROOT / "evals" / "results.md").write_text(report + "\n")

    best_hit = rows[-1][1]
    failed = []
    if best_hit < args.min_hit:
        failed.append(f"Hit@{K} {best_hit:.2f} < {args.min_hit}")
    if e2e["refusal_accuracy"] < args.min_refusal:
        failed.append(f"refusal accuracy {e2e['refusal_accuracy']:.2f} < {args.min_refusal}")
    if failed:
        print("\nEVAL FAILED: " + "; ".join(failed))
        sys.exit(1)
    print("\nEVAL PASSED")


if __name__ == "__main__":
    main()
