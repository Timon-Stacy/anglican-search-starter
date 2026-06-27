#!/usr/bin/env python3
"""Re-chunk (with OCR cleanup) and embed the library into a FAISS index.

This replaces the original raw, character-based ``make_chunks`` with the
validated cleanup pipeline (long-s correction, boilerplate/noise stripping,
paragraph + real-token-aware chunking) and turns the embed step into a
streaming, crash-safe loop suitable for ~1M chunks.

Two phases, each independently resumable:

  rechunk  Read books.content, clean + chunk, write chunks (start_char/end_char
           are offsets into books.content). Tracked per-book in rechunk_status,
           so an interrupted run resumes where it left off. FTS5 triggers are
           dropped for the bulk pass, then the index is rebuilt and triggers
           recreated.

  embed    Embed not-yet-embedded chunks in windows: encode -> add to FAISS
           (IndexIDMap2, vector id == chunks.id) -> save index -> mark
           embeddings_status. Memory-bounded and crash-safe to the last window.

Usage:
    uv run python -m anglican_search.embed_library --phase all
    uv run python -m anglican_search.embed_library --phase rechunk
    uv run python -m anglican_search.embed_library --phase embed
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from .config import (
    DB_PATH,
    DEFAULT_CHUNK_CONFIG,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDING_TRUNCATE_DIM,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    INDEX_PATH,
    INDEX_TYPE,
    PASSAGE_PREFIX,
)
from .device import select_device, supports_fp16
from .pipeline import process_body

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------
# Database / schema
# --------------------------------------------------------------------------
def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id      INTEGER NOT NULL,
            chunk_index  INTEGER NOT NULL,
            start_char   INTEGER NOT NULL,
            end_char     INTEGER NOT NULL,
            text         TEXT NOT NULL,
            UNIQUE(book_id, chunk_index),
            FOREIGN KEY(book_id) REFERENCES books(id) ON DELETE CASCADE
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings_status (
            chunk_id     INTEGER PRIMARY KEY,
            embedded_at  TEXT NOT NULL,
            FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
        )
    """)
    # Tracks which books have been re-chunked with the cleanup pipeline.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rechunk_status (
            book_id     INTEGER PRIMARY KEY,
            n_chunks    INTEGER NOT NULL,
            chunked_at  TEXT NOT NULL,
            FOREIGN KEY(book_id) REFERENCES books(id) ON DELETE CASCADE
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_book_id ON chunks(book_id)")
    conn.commit()


# --- FTS5 maintenance ------------------------------------------------------
def fts_exists(conn: sqlite3.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
    ).fetchone() is not None


def ensure_fts(conn: sqlite3.Connection) -> None:
    if not fts_exists(conn):
        conn.execute("""
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                text, content='chunks', content_rowid='id'
            )
        """)
        conn.commit()


def drop_fts_triggers(conn: sqlite3.Connection) -> None:
    for name in ("chunks_ai", "chunks_ad", "chunks_au"):
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
    conn.commit()


def create_fts_triggers(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
            INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
        END;
    """)
    conn.commit()


def rebuild_fts(conn: sqlite3.Connection) -> None:
    ensure_fts(conn)
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    conn.commit()


# --------------------------------------------------------------------------
# Re-chunk phase
# --------------------------------------------------------------------------
def make_token_counter(model_name: str):
    """A cached counter using the embedding model's real tokenizer."""
    tok = AutoTokenizer.from_pretrained(model_name)

    @lru_cache(maxsize=200_000)
    def count(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    return count


def books_to_rechunk(conn: sqlite3.Connection, limit: int | None = None) -> list[int]:
    sql = """
        SELECT b.id FROM books b
        LEFT JOIN rechunk_status r ON r.book_id = b.id
        WHERE b.content IS NOT NULL AND length(trim(b.content)) > 0
          AND r.book_id IS NULL
        ORDER BY b.id
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return [row[0] for row in conn.execute(sql).fetchall()]


def rechunk_phase(conn: sqlite3.Connection, model_name: str, limit: int | None = None) -> None:
    in_scope = conn.execute(
        "SELECT COUNT(*) FROM books WHERE content IS NOT NULL AND length(trim(content))>0"
    ).fetchone()[0]
    todo = books_to_rechunk(conn, limit=limit)
    log(f"== Re-chunk phase ==  {in_scope} books in scope, {len(todo)} remaining this run")
    if not todo:
        log("All in-scope books already re-chunked.")
        return

    count_tokens = make_token_counter(model_name)
    drop_fts_triggers(conn)  # bulk insert without per-row FTS churn

    t0 = time.time()
    done = 0
    total_chunks = 0
    for bid in todo:
        row = conn.execute("SELECT content FROM books WHERE id=?", (bid,)).fetchone()
        content = row[0] if row else None
        if not content or not content.strip():
            continue
        chunks, _stats = process_body(content, count_tokens, DEFAULT_CHUNK_CONFIG)
        with conn:  # atomic per book -> resumable
            conn.execute("DELETE FROM chunks WHERE book_id=?", (bid,))
            conn.executemany(
                "INSERT INTO chunks (book_id, chunk_index, start_char, end_char, text) "
                "VALUES (?,?,?,?,?)",
                [(bid, c.chunk_index, c.char_start, c.char_end, c.text) for c in chunks],
            )
            conn.execute(
                "INSERT OR REPLACE INTO rechunk_status (book_id, n_chunks, chunked_at) "
                "VALUES (?,?,?)",
                (bid, len(chunks), time.strftime("%Y-%m-%d %H:%M:%S")),
            )
        done += 1
        total_chunks += len(chunks)
        if done % 50 == 0 or done == len(todo):
            rate = done / max(time.time() - t0, 1e-6)
            eta = (len(todo) - done) / max(rate, 1e-6)
            log(f"  re-chunked {done}/{len(todo)} books, {total_chunks} chunks "
                f"({rate:.1f} books/s, ETA {eta/60:.1f} min)")

    log("Rebuilding FTS5 index from cleaned chunks...")
    rebuild_fts(conn)
    create_fts_triggers(conn)
    log("Re-chunk phase complete.")


# --------------------------------------------------------------------------
# Embed phase
# --------------------------------------------------------------------------
def load_index(path: str, dim: int) -> faiss.Index:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        idx = faiss.read_index(path)
        if idx.d != dim:
            raise RuntimeError(
                f"Index dim {idx.d} != model dim {dim}. Delete {path} to rebuild."
            )
        log(f"Loaded FAISS index with {idx.ntotal} vectors.")
        return idx
    if INDEX_TYPE == "flat":
        log("Creating new FAISS index (IndexIDMap2 over IndexFlatIP, exact cosine).")
        return faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
    log(f"Creating new FAISS HNSW index (M={HNSW_M}, efC={HNSW_EF_CONSTRUCTION}, cosine).")
    base = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    base.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    base.hnsw.efSearch = HNSW_EF_SEARCH
    return faiss.IndexIDMap2(base)


def load_unembedded(conn: sqlite3.Connection, limit: int) -> list[tuple[int, str]]:
    rows = conn.execute(
        """
        SELECT c.id, c.text FROM chunks c
        LEFT JOIN embeddings_status e ON e.chunk_id = c.id
        WHERE e.chunk_id IS NULL
        ORDER BY c.id
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def mark_embedded(conn: sqlite3.Connection, ids: list[int]) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO embeddings_status (chunk_id, embedded_at) VALUES (?,?)",
            [(i, now) for i in ids],
        )


def embed_phase(
    conn: sqlite3.Connection,
    index_path: str,
    model_name: str,
    encode_batch: int = 256,
    window: int = 50_000,
    limit: int | None = None,
    fp16: bool = True,
    max_minutes: float | None = None,
) -> None:
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    already = conn.execute("SELECT COUNT(*) FROM embeddings_status").fetchone()[0]
    log(f"== Embed phase ==  {total} chunks, {already} already embedded")
    if already >= total:
        log("Everything already embedded.")
        return

    device = select_device()  # cuda (Nvidia) | xpu (Intel Arc) | cpu
    model = SentenceTransformer(model_name, device=device, truncate_dim=EMBEDDING_TRUNCATE_DIM)
    if fp16 and supports_fp16(device):
        model = model.half()  # ~2x throughput, half VRAM; fine for inference
    try:
        dim = model.get_embedding_dimension()
    except AttributeError:
        dim = model.get_sentence_embedding_dimension()
    if dim != EMBEDDING_DIM:
        log(f"Note: model dim {dim} differs from config EMBEDDING_DIM {EMBEDDING_DIM}.")
    index = load_index(index_path, dim)
    log(f"Model {model_name} on {device}, dim={dim}")

    processed = 0
    t0 = time.time()
    while True:
        if limit is not None and processed >= limit:
            break
        win = window if limit is None else min(window, limit - processed)
        rows = load_unembedded(conn, limit=win)
        if not rows:
            break
        ids = [r[0] for r in rows]
        texts = [PASSAGE_PREFIX + r[1] for r in rows]

        emb = model.encode(
            texts,
            batch_size=encode_batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype("float32")

        index.add_with_ids(emb, np.asarray(ids, dtype=np.int64))
        faiss.write_index(index, index_path)   # persist BEFORE marking (no data loss)
        mark_embedded(conn, ids)

        processed += len(ids)
        done_total = already + processed
        elapsed = time.time() - t0
        rate = processed / max(elapsed, 1e-6)
        eta = (total - done_total) / max(rate, 1e-6)
        log(f"  embedded {done_total}/{total} ({rate:.0f} chunks/s, "
            f"ETA {eta/60:.1f} min). Index ntotal={index.ntotal}")

        if max_minutes is not None and elapsed / 60 >= max_minutes:
            log(f"Reached time cap ({max_minutes} min) after {elapsed/60:.1f} min. "
                f"Stopping cleanly at a saved window — {total - done_total} chunks remain. "
                f"Re-run the same command to resume.")
            return

    log(f"Embed phase done. Index saved to {index_path} ({index.ntotal} vectors).")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--index", default=str(INDEX_PATH))
    ap.add_argument("--model", default=EMBEDDING_MODEL)
    ap.add_argument("--phase", choices=["rechunk", "embed", "all"], default="all")
    ap.add_argument("--encode-batch", type=int, default=256)
    ap.add_argument("--window", type=int, default=50_000, help="Chunks per save cycle")
    ap.add_argument("--limit", type=int, default=None, help="Cap chunks embedded this run")
    ap.add_argument("--rechunk-limit", type=int, default=None, help="Cap books re-chunked this run")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="Stop the embed phase after ~this many minutes (at a saved window)")
    ap.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable fp16 (use fp32)")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        raise SystemExit(f"DB not found: {args.db}")

    conn = connect(args.db)
    ensure_schema(conn)
    ensure_fts(conn)

    if args.phase in ("rechunk", "all"):
        rechunk_phase(conn, args.model, limit=args.rechunk_limit)
    if args.phase in ("embed", "all"):
        embed_phase(conn, args.index, args.model,
                    encode_batch=args.encode_batch, window=args.window,
                    limit=args.limit, fp16=args.fp16, max_minutes=args.max_minutes)

    conn.close()


if __name__ == "__main__":
    main()
