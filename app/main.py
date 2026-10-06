from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config
from .agent import Agent
from .cache import SemanticCache
from .ingest import load_chunks
from .llm import get_llm
from .retrieval import HybridRetriever, get_embedder

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    emb = get_embedder(config.EMBEDDER)
    chunks = load_chunks(config.DOCS_DIR)
    state["agent"] = Agent(
        HybridRetriever(chunks, emb), get_llm(), SemanticCache(emb, config.CACHE_THRESHOLD), config.TOP_K, config.MIN_COVERAGE
    )
    state["chunks"] = len(chunks)
    yield


app = FastAPI(title="DocPilot", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@app.post("/ask")
def ask(req: AskRequest):
    try:
        return state["agent"].ask(req.question)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/health")
def health():
    return {"status": "ok", "chunks": state.get("chunks", 0)}


@app.get("/metrics")
def metrics():
    return state["agent"].metrics.snapshot()
