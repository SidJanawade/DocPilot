import pytest

from app import config
from app.agent import Agent, extract_math, safe_eval
from app.cache import SemanticCache
from app.ingest import load_chunks
from app.llm import ExtractiveLLM
from app.retrieval import BM25, HashEmbedder, HybridRetriever, tokenize


@pytest.fixture(scope="module")
def agent():
    emb = HashEmbedder()
    retriever = HybridRetriever(load_chunks(config.DOCS_DIR), emb)
    return Agent(retriever, ExtractiveLLM(), SemanticCache(emb, 0.9))


def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("The containers are running") == ["container", "running"]


def test_bm25_prefers_matching_doc():
    bm = BM25([tokenize("docker volume persists data"), tokenize("kubernetes pod replica")])
    s = bm.scores(tokenize("volume data"))
    assert s[0] > s[1]


def test_chunks_are_heading_aware():
    chunks = load_chunks(config.DOCS_DIR)
    assert chunks and {"docker.md", "kubernetes.md", "fastapi.md"} <= {c.doc for c in chunks}


def test_hybrid_retrieves_right_doc(agent):
    hit = agent.retriever.search("How do I roll back a deployment?", k=1, rerank=True)[0]
    assert hit.chunk.doc == "kubernetes.md"


def test_answer_has_sources_and_citation(agent):
    out = agent.ask("Which Service type is only reachable inside the cluster?")
    assert not out["refused"] and out["sources"] and "ClusterIP" in out["answer"]


def test_out_of_scope_is_refused(agent):
    assert agent.ask("What is the capital of Australia?")["refused"]


def test_semantic_cache_hit_on_repeat(agent):
    q = "How do I make my production image smaller?"
    agent.ask(q)
    assert agent.ask(q)["cached"]


def test_calculator_tool_and_safety():
    assert safe_eval("2 + 3 * 4") == 14
    assert extract_math("what is 12 / 4?") == "12 / 4"
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('ls')")


def test_empty_question_rejected(agent):
    with pytest.raises(ValueError):
        agent.ask("   ")
