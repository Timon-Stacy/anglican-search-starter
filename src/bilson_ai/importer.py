"""Background importer: turns an approved submission into a live, searchable book.

On approval a submission id is enqueued; a single worker thread downloads the
source text, inserts the book + cleaned chunks into the library DB, embeds the
new chunks, and adds them to the *live* FAISS index (so the book is immediately
searchable — no restart). Failures are recorded on the submission; chunks still
land in the DB (literal/FTS searchable) and can be embedded later by
embed_library if the GPU step fails.
"""

from __future__ import annotations

import queue
import sqlite3
import threading
import time
import urllib.request

from anglican_search.embed_library import make_token_counter
from anglican_search.pipeline import process_body

from . import submissions

_UA = {"User-Agent": "bilson-ai/0.1 (library import)"}


def fetch_text(source_type: str, source_id: str) -> tuple[str | None, str | None]:
    """Download raw text for an IA or Gutenberg source. Returns (text, url)."""
    if source_type == "ia":
        urls = [f"https://archive.org/download/{source_id}/{source_id}_djvu.txt",
                f"https://archive.org/download/{source_id}/{source_id}.txt"]
    elif source_type == "gutenberg":
        urls = [f"https://www.gutenberg.org/files/{source_id}/{source_id}-0.txt",
                f"https://www.gutenberg.org/ebooks/{source_id}.txt.utf-8"]
    else:
        return None, None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=45) as r:
                text = r.read().decode("utf-8", "replace")
            if text.strip():
                return text, url
        except Exception:
            continue
    return None, None


class Importer:
    def __init__(self, searcher, db_path: str, index_path: str):
        self.searcher = searcher
        self.db_path = db_path
        self.index_path = index_path
        self._count_tokens = None
        self._q: queue.Queue[int] = queue.Queue()
        threading.Thread(target=self._run, daemon=True, name="book-importer").start()

    def enqueue(self, sub_id: int) -> None:
        self._q.put(sub_id)

    def _run(self) -> None:
        while True:
            sub_id = self._q.get()
            try:
                self._import(sub_id)
            except Exception as e:  # noqa: BLE001 - record on the submission
                submissions.set_status(sub_id, "failed", str(e)[:300])

    def _import(self, sub_id: int) -> None:
        sub = submissions.get(sub_id)
        if not sub or sub["status"] != "approved":
            return
        stype, sid = sub["source_type"], sub["source_id"]

        text, url = fetch_text(stype, sid)
        if not text:
            submissions.set_status(sub_id, "failed", "download failed or unsupported source")
            return

        conn = sqlite3.connect(self.db_path)
        try:
            col = submissions.SOURCE_COLUMN.get(stype)
            cur = conn.execute(
                f"INSERT INTO books ({col}, title, category, source_url, content, status, approved) "
                "VALUES (?,?,?,?,?, 'ok', 1)",
                (sid, sub["title"] or sid, "Submitted", url, text),
            )
            book_id = cur.lastrowid
            conn.commit()

            if self._count_tokens is None:
                self._count_tokens = make_token_counter(self.searcher.model_name)
            chunks, _stats = process_body(text, self._count_tokens)
            conn.executemany(
                "INSERT INTO chunks (book_id, chunk_index, start_char, end_char, text) "
                "VALUES (?,?,?,?,?)",
                [(book_id, c.chunk_index, c.char_start, c.char_end, c.text) for c in chunks],
            )
            conn.commit()
            rows = conn.execute(
                "SELECT id, text FROM chunks WHERE book_id=? ORDER BY chunk_index", (book_id,)
            ).fetchall()
        finally:
            conn.close()

        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        # Embed + add to the live index, then persist + record embedding status.
        vectors = self.searcher.embed_passages(texts)
        self.searcher.add_to_index(ids, vectors)
        self.searcher.save_index(self.index_path)
        conn = sqlite3.connect(self.db_path)
        try:
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            conn.executemany(
                "INSERT OR IGNORE INTO embeddings_status (chunk_id, embedded_at) VALUES (?,?)",
                [(i, now) for i in ids],
            )
            conn.commit()
        finally:
            conn.close()
        submissions.set_status(sub_id, "imported", f"book_id {book_id}, {len(chunks)} chunks")
