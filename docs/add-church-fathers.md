# Adding the Schaff Church Fathers set (ANF + NPNF)

Adds the 38-volume Schaff edition — Ante-Nicene Fathers and Nicene & Post-Nicene
Fathers (series 1 & 2) — to the library, **one book row per work**, from CCEL's
proofread digital text (public domain, far cleaner than archive.org OCR).

Do all of this on a **rented GPU box** (Ubuntu, this repo cloned, internet): it
needs to download from CCEL, chunk with the model tokenizer, and embed on a GPU.
Start from your existing `library-clean.db` (it already holds the 1,921 books /
1.23M chunks) so the Fathers are added to it.

## 1. Review the split (no DB writes)
```bash
uv pip install lxml          # or: uv sync --extra ingest
uv run python scripts/ingest_ccel_fathers.py --dry-run
```
This downloads the 38 ThML volumes into `ccel_cache/` (resumable) and prints a
manifest: `author | title | chars` per work. Skim it — especially the **NPNF2
mixed-author volumes** (npnf202/203/209/211/213), which get a volume-level or
"Various" author since those volumes lack per-work author markers; the work title
is always the reliable citation anchor.

## 2. Ingest into the DB
```bash
uv run python scripts/ingest_ccel_fathers.py --db library-clean.db
```
Each work → a `books` row: `title "<work> (ANF/NPNF1/NPNF2 <vol>)"`, author,
`category "Church Fathers"`, CCEL `source_url`, `year` (edition), `content`.
Idempotent via a new `books.ccel_id` column — re-running skips works already in.

## 3. Re-chunk + rebuild the index (full, on the GPU)
The repaired DB's `embeddings_status` is empty and the index was built separately,
so the clean path is a **full rebuild** (re-embeds existing + new together):
```bash
rm -f index.faiss
ANGLICAN_DB=library-clean.db ANGLICAN_INDEX=index.faiss \
  uv run python -m anglican_search.embed_library --phase all --encode-batch 256
```
`--phase all` re-chunks only the new books (existing are skipped via
`rechunk_status`), then embeds every chunk into a fresh `index.faiss`. On a rented
GPU this is the same ~30-60 min as the original build.

*Faster alternative:* re-chunk only (`--phase rechunk`), then run
`notebooks/build_index_cuda.ipynb` (big batches, single write) to build the index.

*Cheaper/incremental alternative* (avoids re-embedding the 1.23M you have): mark
the existing chunks embedded first so only the new ones are processed and appended
to the current `index.faiss`:
```bash
python -c "import sqlite3,datetime; c=sqlite3.connect('library-clean.db'); \
c.execute(\"INSERT OR IGNORE INTO embeddings_status (chunk_id, embedded_at) \
SELECT id, datetime('now') FROM chunks\"); c.commit()"
# (keep index.faiss in place, then run --phase all as above)
```

## 4. Deploy
Copy the updated `library-clean.db` + `index.faiss` to the NAS and restart:
```bash
scp library-clean.db index.faiss USER@NAS:~/Docker/rag/anglican-search-starter/deploy/docker/home/data/
# on the NAS (data/library.db is what the app reads):
mv data/library-clean.db data/library.db   # if your app points at library.db
cd ~/Docker/rag/anglican-search-starter/deploy/docker/home && sudo docker compose up -d
```

## 5. Update the prompts
The corpus now spans the early Church Fathers (2nd-8th c., Schaff edition) as well
as the Anglican texts. Edit `SERVER_INSTRUCTIONS` (`src/anglican_search/mcp_tool.py`)
and `docs/research-prompt.md` to say so, rebuild `bilson`, and re-paste the project
prompt. Users can filter to just the patristics with `category: "Church Fathers"`.
