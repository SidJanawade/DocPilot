import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = Path(os.getenv("DOCPILOT_DOCS", ROOT / "data" / "docs"))
EMBEDDER = os.getenv("DOCPILOT_EMBEDDER", "hash")  # "hash" (offline) or "st" (sentence-transformers)
LLM_MODEL = os.getenv("DOCPILOT_MODEL", "claude-sonnet-4-6")
TOP_K = int(os.getenv("DOCPILOT_TOP_K", "4"))
CACHE_THRESHOLD = float(os.getenv("DOCPILOT_CACHE_THRESHOLD", "0.9"))
MIN_COVERAGE = float(os.getenv("DOCPILOT_MIN_COVERAGE", "0.34"))
