# GPU embed image for the Anglican library.
#
# The torch cu128 wheels bundle the CUDA runtime, so a slim Python base is
# enough — at runtime the container uses the host NVIDIA driver via the
# Kubernetes GPU device plugin (request nvidia.com/gpu: 1). The Qwen3 embedder
# is pre-baked into the image so the embed job needs no model egress, which
# suits an air-gapped private cloud.
#
# Build:  docker build -t anglican-embed:latest .
# Run:    docker run --gpus all -v /host/data:/data anglican-embed:latest

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/opt/hf-cache \
    ANGLICAN_DB=/data/library.db \
    ANGLICAN_INDEX=/data/index.faiss \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates && rm -rf /var/lib/apt/lists/*

# uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer (cached unless deps change). uv.lock is optional.
COPY pyproject.toml uv.lock* ./
COPY src ./src
RUN uv sync --no-dev

# Pre-bake the embedder (~1.2 GB) into the image cache so runtime needs no
# Hugging Face egress. (The reranker is not needed for embedding.)
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', truncate_dim=512)"

COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# /data holds the input DB and output index (mount a PVC here).
VOLUME ["/data"]

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--phase", "all", "--encode-batch", "512"]
