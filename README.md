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

## Deployment

See [deploy/README.md](deploy/README.md) for running the index build as a GPU
job (Kubernetes) or notebook on a private-cloud AI platform, and shipping the
resulting `index.faiss` to a CPU serving host.
