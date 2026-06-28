"""Search the Anglican library: semantic (FAISS) + cross-encoder rerank, and
literal (FTS5). Both support metadata filters (author / category / year / title).

Pipeline for semantic search:
    embed query -> FAISS top-N candidates -> apply metadata filters ->
    cross-encoder rerank the top of those -> return top_k.

The reranker reorders by true query-passage relevance, which fixes the cases
where pure vector similarity ranks a loosely-related passage too high. Filters
are applied between retrieval and rerank so they never waste the rerank budget.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import threading
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np

from .config import (
    DB_PATH,
    DEFAULT_RERANK_POOL,
    EMBEDDING_MODEL,
    EMBEDDING_TRUNCATE_DIM,
    HNSW_EF_SEARCH,
    INDEX_PATH,
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    RERANKER_MODEL,
    SQLITE_MMAP_BYTES,
)
from .device import model_load_kwargs, rerank_device, select_device, supports_fp16


@dataclass
class Filters:
    author: str | None = None
    category: str | None = None
    year_min: int | None = None
    year_max: int | None = None
    title: str | None = None
    book_ids: list[int] | None = None

    def any(self) -> bool:
        return any(
            v is not None for v in (
                self.author, self.category, self.year_min,
                self.year_max, self.title, self.book_ids,
            )
        )

    def sql(self) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if self.author:
            clauses.append("b.author LIKE ?"); params.append(f"%{self.author}%")
        if self.category:
            clauses.append("b.category LIKE ?"); params.append(f"%{self.category}%")
        if self.title:
            clauses.append("b.title LIKE ?"); params.append(f"%{self.title}%")
        if self.year_min is not None:
            clauses.append("b.year >= ?"); params.append(self.year_min)
        if self.year_max is not None:
            clauses.append("b.year <= ?"); params.append(self.year_max)
        if self.book_ids:
            clauses.append(f"b.id IN ({','.join('?' * len(self.book_ids))})")
            params.extend(self.book_ids)
        return clauses, params


class Searcher:
    def __init__(
        self,
        db_path: str = str(DB_PATH),
        index_path: str = str(INDEX_PATH),
        model_name: str = EMBEDDING_MODEL,
        reranker_name: str = RERANKER_MODEL,
    ) -> None:
        self.db_path = db_path
        self.index_path = index_path
        self.model_name = model_name
        self.reranker_name = reranker_name
        # Per-thread SQLite connections — a single connection isn't safe to share
        # across the threadpool threads serving concurrent requests.
        self._local = threading.local()
        # One GPU at serve time, so serialize inference: concurrent requests queue
        # cleanly instead of contending on CUDA (which wouldn't parallelize anyway).
        self._gpu_lock = threading.Lock()
        # Guards the FAISS index so live additions (new books) never race searches.
        self._index_lock = threading.Lock()
        self._model = None
        self._index = None
        self._reranker = None
        self._hnsw = None  # the HNSW struct (if any), for runtime efSearch tuning

    # -- lazy heavy resources ------------------------------------------------
    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.db_path, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA busy_timeout = 5000")
            c.execute(f"PRAGMA mmap_size = {SQLITE_MMAP_BYTES}")  # DB pages in RAM
            c.execute("PRAGMA cache_size = -65536")               # ~64 MB page cache
            self._local.conn = c
        return c

    @property
    def index(self) -> faiss.Index:
        if self._index is None:
            self._index = faiss.read_index(self.index_path)
            # IndexIDMap2.index comes back as a base Index pointer — downcast to
            # reach the concrete HNSW so efSearch is actually settable.
            try:
                inner = faiss.downcast_index(self._index.index)
                if hasattr(inner, "hnsw"):
                    inner.hnsw.efSearch = HNSW_EF_SEARCH
                    self._hnsw = inner.hnsw
            except Exception:  # noqa: BLE001 - flat index, nothing to tune
                pass
        return self._index

    def _ensure_ef(self, want: int) -> None:
        """HNSW needs efSearch >= k to return k good results; bump if needed."""
        if self._hnsw is not None and self._hnsw.efSearch < want:
            self._hnsw.efSearch = want

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            device = select_device()  # cuda (Nvidia) | xpu (Intel Arc) | cpu
            model = SentenceTransformer(
                self.model_name, device=device, truncate_dim=EMBEDDING_TRUNCATE_DIM,
                model_kwargs=model_load_kwargs(device) or None,
            )
            if supports_fp16(device):
                model = model.half()  # fp16: faster inference + less VRAM
            if model.get_sentence_embedding_dimension() != self.index.d:
                raise RuntimeError(
                    "Model dim != index dim; index built with a different model."
                )
            self._model = model
        return self._model

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            device = rerank_device()  # CPU on Arc (XPU cross-encoders SIGBUS); GPU on CUDA
            mkw = model_load_kwargs(device) or None
            try:
                ce = CrossEncoder(self.reranker_name, device=device, model_kwargs=mkw)
            except TypeError:
                # Older CrossEncoder has no model_kwargs param; eager not forced.
                ce = CrossEncoder(self.reranker_name, device=device)
            if supports_fp16(device):
                try:
                    ce.model.half()  # fp16 reranker
                except Exception:  # noqa: BLE001 - keep fp32 if unsupported
                    pass
            self._reranker = ce
        return self._reranker

    # -- helpers -------------------------------------------------------------
    _SELECT = """
        SELECT c.id, c.book_id, c.text, c.start_char, c.end_char,
               b.title, b.author, b.category, b.source_url, b.year, b.publisher
        FROM chunks c JOIN books b ON b.id = c.book_id
    """

    def _fetch(self, ids: list[int], filters: Filters) -> dict[int, sqlite3.Row]:
        if not ids:
            return {}
        where = [f"c.id IN ({','.join('?' * len(ids))})"]
        params: list[Any] = list(ids)
        fclauses, fparams = filters.sql()
        where += fclauses
        params += fparams
        rows = self.conn.execute(
            self._SELECT + " WHERE " + " AND ".join(where), params
        ).fetchall()
        return {r["id"]: r for r in rows}

    @staticmethod
    def _format(row: sqlite3.Row, score: float) -> dict[str, Any]:
        r = dict(row)
        return {
            "score": float(score),
            "book_id": int(r["book_id"]),
            "title": r["title"],
            "author": r["author"],
            "category": r["category"],
            "year": r.get("year"),
            "publisher": r.get("publisher"),
            "url": r["source_url"],
            "char_start": r.get("start_char"),
            "char_end": r.get("end_char"),
            "text": r["text"],
        }

    def _rerank_scores(self, query: str, texts: list[str]) -> list[float]:
        pairs = [[query, t] for t in texts]
        return [float(s) for s in self.reranker.predict(pairs, convert_to_numpy=True)]

    # -- search modes --------------------------------------------------------
    def semantic(
        self,
        query: str,
        k: int = 5,
        *,
        rerank: bool = True,
        fetch_k: int | None = None,
        rerank_pool: int = DEFAULT_RERANK_POOL,
        filters: Filters | None = None,
    ) -> list[dict[str, Any]]:
        filters = filters or Filters()
        if fetch_k is None:
            # Over-fetch when filtering so enough candidates survive. Kept modest
            # for HNSW (efSearch must track k); still ample for filtering.
            fetch_k = 600 if filters.any() else 200

        with self._gpu_lock:
            qv = self.model.encode(
                [QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True
            ).astype("float32")
        self._ensure_ef(max(fetch_k, k))
        with self._index_lock:
            scores, ids = self.index.search(qv, max(fetch_k, k))
        id_list = [int(i) for i in ids[0] if i != -1]
        rows = self._fetch(id_list, filters)

        # Keep FAISS order, drop ids filtered out.
        ordered = [(cid, float(s)) for cid, s in zip(id_list, scores[0]) if cid in rows]

        if rerank and ordered:
            pool = ordered[:rerank_pool]
            with self._gpu_lock:
                rr = self._rerank_scores(query, [rows[cid]["text"] for cid, _ in pool])
            ranked = sorted(zip(pool, rr), key=lambda t: t[1], reverse=True)
            return [self._format(rows[cid], score) for (cid, _), score in ranked[:k]]
        return [self._format(rows[cid], s) for cid, s in ordered[:k]]

    def literal(
        self, query: str, k: int = 5, *, filters: Filters | None = None
    ) -> list[dict[str, Any]]:
        filters = filters or Filters()
        fclauses, fparams = filters.sql()
        extra = (" AND " + " AND ".join(fclauses)) if fclauses else ""
        sql = f"""
            SELECT c.id, c.book_id, c.text, c.start_char, c.end_char,
                   b.title, b.author, b.category, b.source_url, b.year, b.publisher,
                   rank AS score
            FROM chunks_fts fts
            JOIN chunks c ON c.id = fts.rowid
            JOIN books b ON b.id = c.book_id
            WHERE chunks_fts MATCH ?{extra}
            ORDER BY rank
            LIMIT ?
        """
        rows = self.conn.execute(sql, [query, *fparams, k]).fetchall()
        return [self._format(r, r["score"]) for r in rows]

    def search(
        self, query: str, k: int = 5, semantic: bool = True, **kw: Any
    ) -> list[dict[str, Any]]:
        return self.semantic(query, k, **kw) if semantic else self.literal(query, k, **kw)

    # -- browse (manual library exploration) --------------------------------
    @staticmethod
    def _browse_where(q: str | None, category: str | None) -> tuple[str, list[Any]]:
        where = ["EXISTS (SELECT 1 FROM chunks c WHERE c.book_id = b.id)"]
        params: list[Any] = []
        if q:
            where.append("(b.title LIKE ? OR b.author LIKE ?)")
            params += [f"%{q}%", f"%{q}%"]
        if category:
            where.append("b.category = ?")
            params.append(category)
        return "WHERE " + " AND ".join(where), params

    def categories(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT category FROM books "
            "WHERE category IS NOT NULL AND category != '' ORDER BY category"
        ).fetchall()
        return [r[0] for r in rows]

    def count_books(self, q: str | None = None, category: str | None = None) -> int:
        where, params = self._browse_where(q, category)
        return self.conn.execute(f"SELECT COUNT(*) FROM books b {where}", params).fetchone()[0]

    def list_books(self, q: str | None = None, category: str | None = None,
                   limit: int = 40, offset: int = 0) -> list[sqlite3.Row]:
        where, params = self._browse_where(q, category)
        return self.conn.execute(
            f"""SELECT b.id, b.title, b.author, b.category, b.year, b.source_url,
                       (SELECT COUNT(*) FROM chunks c WHERE c.book_id = b.id) AS n_chunks
                FROM books b {where}
                ORDER BY b.author, b.title LIMIT ? OFFSET ?""",
            params + [limit, offset],
        ).fetchall()

    def get_book(self, book_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT id, title, author, category, year, publisher, source_url "
            "FROM books WHERE id = ?", (book_id,)
        ).fetchone()

    def count_book_chunks(self, book_id: int) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE book_id = ?", (book_id,)
        ).fetchone()[0]

    def book_chunks(self, book_id: int, limit: int = 25, offset: int = 0) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT chunk_index, text, start_char, end_char FROM chunks "
            "WHERE book_id = ? ORDER BY chunk_index LIMIT ? OFFSET ?",
            (book_id, limit, offset),
        ).fetchall()

    # -- live additions (new books imported at runtime) ---------------------
    def embed_passages(self, texts: list[str]):
        """Embed passage texts (no query prefix) -> float32 matrix for FAISS."""
        with self._gpu_lock:
            return self.model.encode(
                [PASSAGE_PREFIX + t for t in texts],
                normalize_embeddings=True, convert_to_numpy=True,
            ).astype("float32")

    def add_to_index(self, ids: list[int], vectors) -> None:
        """Add new vectors to the live index (id == chunks.id). Thread-safe."""
        import numpy as np

        with self._index_lock:
            self.index.add_with_ids(vectors, np.asarray(ids, dtype=np.int64))

    def save_index(self, path: str | None = None) -> None:
        with self._index_lock:
            faiss.write_index(self.index, path or self.index_path)


# Module-level singleton so the model/index load once per process (MCP server).
_searcher: Searcher | None = None


def get_searcher(**kw: Any) -> Searcher:
    global _searcher
    if _searcher is None:
        _searcher = Searcher(**kw)
    return _searcher


def main() -> None:
    ap = argparse.ArgumentParser(description="Search the Anglican library.")
    ap.add_argument("--q", required=True, help="Query string")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--literal", action="store_true", help="FTS5 literal search instead of semantic")
    ap.add_argument("--no-rerank", dest="rerank", action="store_false", help="Skip cross-encoder rerank")
    ap.add_argument("--author")
    ap.add_argument("--category")
    ap.add_argument("--title")
    ap.add_argument("--year-min", type=int)
    ap.add_argument("--year-max", type=int)
    ap.add_argument("--model", default=EMBEDDING_MODEL)
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--index", default=str(INDEX_PATH))
    args = ap.parse_args()

    s = Searcher(db_path=args.db, index_path=args.index, model_name=args.model)
    filters = Filters(author=args.author, category=args.category, title=args.title,
                      year_min=args.year_min, year_max=args.year_max)
    if args.literal:
        results = s.literal(args.q, k=args.k, filters=filters)
    else:
        results = s.semantic(args.q, k=args.k, rerank=args.rerank, filters=filters)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
