#!/usr/bin/env bash
# Provision an Intel Arc (XPU) box to embed and/or serve the Anglican search
# engine. Installs the XPU PyTorch build, the package, and pre-caches the models.
# Nvidia boxes use the default cu128 build instead (`uv sync`); CPU-only serving
# boxes use deploy/serve_setup.sh. The engine auto-detects the device at runtime
# (anglican_search/device.py) — this script just installs the right torch wheel.
#
#   git clone https://github.com/Timon-Stacy/anglican-search-starter
#   cd anglican-search-starter
#   bash deploy/gpu_setup_arc.sh
#
# HOST PREREQ (one-time, not done here): the Intel GPU kernel driver + compute
# runtime must already be installed.
#   * Linux (Ubuntu 24.04): install Intel's client-GPU compute runtime
#     (intel-opencl-icd, libze1 / level-zero) from Intel's apt repo. The torch
#     XPU wheel bundles the oneAPI math libs, so you do NOT need a full oneAPI
#     SDK install — just the kernel driver + runtime loaders.
#   * Windows: install the latest Intel Arc graphics driver.
# Verify after: `ANGLICAN_DEVICE=xpu uv run python scripts/check_env.py` must
# print "GPU matmul OK".
set -euo pipefail

# Full reranker is fine on a GPU; override before running to use a lighter one.
ANGLICAN_RERANKER="${ANGLICAN_RERANKER:-BAAI/bge-reranker-v2-m3}"

# 1. uv
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2. XPU virtualenv. Install torch from the Intel XPU index explicitly (the
#    project pins the cu128 Nvidia build, which we do NOT want here), then the
#    rest of the deps and the package without re-resolving torch.
echo "[setup] building Intel XPU environment ..."
uv venv --python 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/xpu
uv pip install \
  "sentence-transformers>=3.0" "faiss-cpu>=1.8" "mcp>=1.2" \
  "transformers>=4.40" "pyspellchecker>=0.9.0" "numpy>=1.26" "tqdm>=4.66" \
  "fastapi>=0.110" "uvicorn>=0.27" "jinja2>=3.1" "python-multipart>=0.0.9" "itsdangerous>=2.1"
uv pip install --no-deps -e .

# 3. Prove the GPU actually runs a kernel before we trust it for embedding.
echo "[setup] verifying XPU ..."
ANGLICAN_DEVICE=xpu .venv/bin/python scripts/check_env.py

# 4. Pre-cache the models so the first query/embed isn't slow.
echo "[setup] caching models (embedder + reranker: $ANGLICAN_RERANKER) ..."
.venv/bin/python - "$ANGLICAN_RERANKER" <<'PY'
import sys
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
CrossEncoder(sys.argv[1])
print("models cached")
PY

echo
echo "[setup] done. Build the index on the Arc GPU with a small batch (XPU uses"
echo "eager attention, and the A380 has only ~6 GB VRAM — 16 is a safe default):"
echo "  ANGLICAN_DEVICE=xpu uv run python -m anglican_search.embed_library --phase all --encode-batch 16"
echo "On 'XPU out of memory', halve it (8); to go faster if it's stable, try 24."
