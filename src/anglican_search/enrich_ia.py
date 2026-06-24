#!/usr/bin/env python3
"""Enrich books with Internet Archive metadata (year, publisher, language).

For every book that has an `ia_title_id` and hasn't been enriched yet, fetch
https://archive.org/metadata/<id> and pull a usable publication year (plus
publisher/language) so the search layer can filter by date. Resumable (skips
rows already stamped with `ia_meta_at`), concurrent, and polite.

    uv run python -m anglican_search.enrich_ia
    uv run python -m anglican_search.enrich_ia --limit 50   # small test
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import DB_PATH

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_UA = {"User-Agent": "anglican-search/0.1 (personal library metadata enrichment)"}
_YEAR_RE = re.compile(r"\b(1[4-9]\d{2}|20\d{2})\b")  # plausible printing years


def _first(value):
    """IA metadata fields are sometimes lists; take the first scalar."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def parse_year(meta: dict) -> int | None:
    for key in ("year", "date"):
        raw = _first(meta.get(key))
        if raw:
            m = _YEAR_RE.search(str(raw))
            if m:
                return int(m.group(1))
    return None


def fetch(ia_id: str) -> dict | None:
    url = f"https://archive.org/metadata/{ia_id}"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8", "replace"))
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return None


def enrich_one(book_id: int, ia_id: str) -> tuple[int, int | None, str | None, str | None]:
    data = fetch(ia_id) or {}
    meta = data.get("metadata", {}) if isinstance(data, dict) else {}
    return (
        book_id,
        parse_year(meta),
        _first(meta.get("publisher")),
        _first(meta.get("language")),
    )


def ensure_columns(conn: sqlite3.Connection) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(books)")}
    for name, decl in (
        ("year", "INTEGER"),
        ("publisher", "TEXT"),
        ("language", "TEXT"),
        ("ia_meta_at", "TEXT"),
    ):
        if name not in cols:
            conn.execute(f"ALTER TABLE books ADD COLUMN {name} {decl}")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_books_year ON books(year)")
    conn.commit()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    ensure_columns(conn)

    sql = "SELECT id, ia_title_id FROM books WHERE ia_title_id IS NOT NULL AND ia_meta_at IS NULL"
    if args.limit:
        sql += f" LIMIT {int(args.limit)}"
    todo = conn.execute(sql).fetchall()
    print(f"Enriching {len(todo)} books from Internet Archive ({args.workers} workers)...", flush=True)
    if not todo:
        return

    done = found = 0
    t0 = time.time()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(enrich_one, bid, iaid): bid for bid, iaid in todo}
        for fut in as_completed(futures):
            book_id, year, publisher, language = fut.result()
            conn.execute(
                "UPDATE books SET year=?, publisher=?, language=?, ia_meta_at=? WHERE id=?",
                (year, publisher, language, now, book_id),
            )
            done += 1
            found += year is not None
            if done % 100 == 0 or done == len(todo):
                conn.commit()
                rate = done / max(time.time() - t0, 1e-6)
                print(f"  {done}/{len(todo)} processed, {found} with a year "
                      f"({rate:.1f}/s, ETA {(len(todo)-done)/max(rate,1e-6)/60:.1f} min)", flush=True)
    conn.commit()
    print(f"Done. {found}/{done} books now have a publication year.", flush=True)


if __name__ == "__main__":
    main()
