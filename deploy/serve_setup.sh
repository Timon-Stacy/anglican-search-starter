#!/usr/bin/env bash
# Provision a CPU-only host to serve the Anglican search MCP server.
# Installs CPU PyTorch (no CUDA), the package, and pre-caches the serving models.
# Works on x86_64 and arm64 Linux (Ubuntu 22.04/24.04 recommended).
#
#   git clone https://github.com/Timon-Stacy/anglican-search-starter
#   cd anglican-search-starter
#   bash deploy/serve_setup.sh
set -euo pipefail

# Light, fast CPU reranker by default (override before running to change it).
ANGLICAN_RERANKER="${ANGLICAN_RERANKER:-cross-encoder/ms-marco-MiniLM-L-6-v2}"

# 1. uv
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2. CPU virtualenv. We install torch from the CPU index explicitly (the
#    project pins the cu128 GPU build, which we do NOT want here), then install
#    the rest and the package itself without re-resolving torch.
echo "[setup] building CPU environment ..."
uv venv --python 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install \
  "sentence-transformers>=3.0" "faiss-cpu>=1.8" "mcp>=1.2" \
  "pyspellchecker>=0.9.0" "numpy>=1.26" "tqdm>=4.66" \
  "fastapi>=0.110" "uvicorn>=0.27" "jinja2>=3.1" "python-multipart>=0.0.9" "itsdangerous>=2.1"
uv pip install --no-deps -e .

# 3. Pre-cache the serving models so the first query isn't slow.
echo "[setup] caching models (embedder + reranker: $ANGLICAN_RERANKER) ..."
.venv/bin/python - "$ANGLICAN_RERANKER" <<'PY'
import sys
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", truncate_dim=512)
CrossEncoder(sys.argv[1])
print("models cached")
PY

mkdir -p data
echo
echo "[setup] done. Next: copy library-serve.db and index.faiss into ./data, then:"
echo "  ANGLICAN_DB=\$PWD/data/library-serve.db ANGLICAN_INDEX=\$PWD/data/index.faiss \\"
echo "  ANGLICAN_RERANKER=$ANGLICAN_RERANKER ANGLICAN_RERANK_POOL=30 \\"
echo "  uv run anglican-search-mcp"
