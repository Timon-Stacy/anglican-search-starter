"""Central configuration: paths, embedding model, chunking parameters.

Everything that might reasonably change lives here so the rest of the pipeline
imports from one place. In particular the embedding model name is here and
nowhere else, so swapping it is a one-line change (until an index has actually
been built, after which the vector dimension is fixed).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Project layout ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = PROJECT_ROOT / "sample_data"
# DB/index paths default to the project root but are overridable via env, so the
# same code runs locally and in a container with mounted volumes (e.g. on HPE
# Private Cloud AI the Job mounts a PVC at /data and sets ANGLICAN_DB/INDEX).
DB_PATH = Path(os.environ.get("ANGLICAN_DB", str(PROJECT_ROOT / "library.db")))
INDEX_PATH = Path(os.environ.get("ANGLICAN_INDEX", str(PROJECT_ROOT / "index.faiss")))

# Embedding model -----------------------------------------------------------
# Qwen/Qwen3-Embedding-0.6B: current top-tier retrieval in its class, with
# Matryoshka (MRL) support so we truncate 1024 -> 512 dims to halve index RAM
# and ~double CPU vector-search speed (the deployment target is a CPU-only
# server). Qwen3 is asymmetric and instruction-aware: queries get an
# "Instruct: <task>\nQuery:" prefix, passages are embedded raw.
#
# NOTE: changing the embedder or EMBEDDING_TRUNCATE_DIM invalidates the index —
# clear embeddings_status and rebuild index.faiss (re-embed). Re-chunking is NOT
# needed (chunks are model-agnostic text; Qwen3's 32k context easily fits them).
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
# Full native dimension (GPU-served, so no need to truncate for RAM/CPU speed).
EMBEDDING_TRUNCATE_DIM = None  # set e.g. 512 to halve index size on a CPU box
EMBEDDING_DIM = 1024
QUERY_PREFIX = (
    "Instruct: Given a question about Christian theology, retrieve passages "
    "from historical Anglican texts that answer it\nQuery:"
)
PASSAGE_PREFIX = ""  # Qwen3 documents are embedded without an instruction

# Reranker (cross-encoder) applied to FAISS candidates. Default bge-reranker-v2-m3
# (~568M, Apache-2.0) is great on a GPU. On a CPU-only serving box override both
# of these via env for low latency, e.g.:
#   ANGLICAN_RERANKER=cross-encoder/ms-marco-MiniLM-L-6-v2  ANGLICAN_RERANK_POOL=30
RERANKER_MODEL = os.environ.get("ANGLICAN_RERANKER", "BAAI/bge-reranker-v2-m3")
DEFAULT_RERANK_POOL = int(os.environ.get("ANGLICAN_RERANK_POOL", "80"))


@dataclass(frozen=True)
class ChunkConfig:
    """Token budget for chunking. Targets ~300-500 tokens with light overlap."""

    target_tokens: int = 400
    max_tokens: int = 480  # hard ceiling, stays under the model's 512 limit
    min_tokens: int = 80   # don't emit tiny dangling chunks if avoidable
    overlap_tokens: int = 60


DEFAULT_CHUNK_CONFIG = ChunkConfig()


def estimate_tokens(text: str) -> int:
    """Fast, offline token estimate (~1.3 subword tokens per whitespace word).

    Used for previewing/tuning chunk boundaries without downloading the model.
    The indexing step swaps in the model's real tokenizer for exact counts.
    """
    return int(len(text.split()) * 1.3) + 1
