# Anglican Library Search

Local semantic + keyword search over a personal library of ~1,500 OCR'd
19th-century Anglican theological texts, exposed to LLM clients as an MCP tool.

The pipeline cleans noisy OCR (long-s correction, de-hyphenation, boilerplate
and running-head removal), chunks the text, embeds it into a FAISS index,
reranks with a cross-encoder, and serves it over the Model Context Protocol.

## Architecture

```
books.content ──▶ clean + chunk ──▶ SQLite (chunks, FTS5) ──▶ embed ──▶ FAISS
                                                                  │
   query ──▶ embed ──▶ FAISS top-N ──▶ metadata filter ──▶ rerank ──▶ results
```

- **Embedder:** `Qwen/Qwen3-Embedding-0.6B`, truncated to 512-d via Matryoshka
  (smaller index + faster CPU search). Swappable in `config.py`.
- **Reranker:** `BAAI/bge-reranker-v2-m3` cross-encoder over the FAISS candidates.
- **Store:** SQLite holds chunk text + metadata keyed so the FAISS vector id
  equals the chunk id; FTS5 provides exact keyword search.
- **Filters:** author, category, title, and publication-year range
  (year/publisher enriched from Internet Archive metadata).
- **Server:** an MCP stdio server exposing `search_anglican_library`.

## Setup

Requires [uv](https://docs.astral.sh/uv/). The GPU build of PyTorch (CUDA 12.8)
is pulled automatically from the configured index.

```bash
uv sync
uv run python scripts/check_env.py        # verify GPU / dependencies
```

### Compute backends (Nvidia / Intel Arc / CPU)

The engine auto-detects the accelerator at runtime (`anglican_search/device.py`):
`cuda` (Nvidia) → `xpu` (Intel Arc) → `cpu`. Only the embedder and reranker move
to the GPU; FAISS always runs on CPU, so the search path is identical everywhere.
Force a backend with `ANGLICAN_DEVICE=cuda|xpu|cpu`.

`uv sync` installs the Nvidia (cu128) build. For other hardware, install the
matching torch wheel instead of syncing torch:

| Hardware | Install | Notes |
|---|---|---|
| Nvidia (default) | `uv sync` | cu128 wheels (Blackwell-capable) |
| **Intel Arc (e.g. A380)** | `bash deploy/gpu_setup_arc.sh` | XPU wheels; needs the Intel GPU driver/runtime on the host |
| CPU-only serving | `bash deploy/serve_setup.sh` | cpu wheels |

Verify the GPU actually runs a kernel: `ANGLICAN_DEVICE=xpu uv run python
scripts/check_env.py` (it prints the detected device and a matmul checksum).

## Build the index

`library.db` (the SQLite store with cleaned chunks) is the input.

```bash
uv run python -m anglican_search.embed_library --phase all
```

The build is resumable: re-running continues from `embeddings_status`. To enrich
books with publication year/publisher from Internet Archive:

```bash
uv run python -m anglican_search.enrich_ia
```

## Search

```bash
# semantic + rerank
uv run python -m anglican_search.search --q "the eternal generation of the Son" --k 5
# with filters
uv run python -m anglican_search.search --q "church authority" --category "Church History" --year-min 1850
# literal full-text
uv run python -m anglican_search.search --q "Athanasian Creed" --literal
```

## MCP server

```bash
uv run anglican-search-mcp
```

Register it with an MCP client by pointing `command` at `uv run
anglican-search-mcp` with `cwd` set to this repo. The tool accepts a query plus
optional `top_k`, `mode` (`semantic`/`literal`), `rerank`, and the metadata
filters.

## Repository layout

| Path | Purpose |
|---|---|
| `src/anglican_search/parse.py` `clean.py` `longs.py` `chunk.py` | Ingestion: header parsing, OCR cleanup, chunking |
| `src/anglican_search/embed_library.py` | Re-chunk + embed into FAISS (resumable) |
| `src/anglican_search/search.py` | Semantic + literal search, filters, rerank |
| `src/anglican_search/server.py` | MCP stdio server |
| `src/anglican_search/enrich_ia.py` | Internet Archive metadata enrichment |
| `scripts/` | Env check + chunk inspection utilities |
| `notebooks/embed_on_pcai.ipynb` | Build the index on a GPU notebook workspace |
| `deploy/` | Dockerfile + Kubernetes manifests + deployment guide |

## Bilson AI — hosted web service

`src/bilson_ai/` is a FastAPI app that turns the search engine into a product:
a marketing site, user signup/login (hashed passwords + sessions), API-key
issue/revoke, per-user monthly usage quotas, an admin panel, a documented REST
API (`POST /v1/search`), plus health and legal pages. It loads the search model
once and shares it across requests.

```bash
uv sync --extra web                 # or pip install the [web] extra
BILSON_SECRET=$(openssl rand -hex 32) uv run bilson-ai   # serves on 127.0.0.1:8001
```

## Deployment

- **Build the index** on a GPU (Kubernetes job or notebook): [deploy/README.md](deploy/README.md).
- **Host the service** on a CPU box (specs, providers, slim DB, TLS): [deploy/SERVE.md](deploy/SERVE.md).
- **Full server runbook** (Bilson AI + systemd + Caddy + admin, step by step): [deploy/SETUP.md](deploy/SETUP.md).
