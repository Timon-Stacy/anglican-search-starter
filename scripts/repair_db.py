"""Repair a library.db whose `embeddings_status` table is corrupt.

The corruption observed is localised to the `embeddings_status` b-tree (plus a
freelist-size mismatch); every table that holds real data — `books`, `chunks`,
`chunks_fts`, `rechunk_status` — reads fine. `embeddings_status` only tracks
*which chunks have already been embedded*, so when you're rebuilding the FAISS
index from scratch it is disposable.

Strategy (non-destructive — never touches the source):
  1. Copy library.db -> the output path (a snapshot you can fall back to).
  2. On the COPY, remove the corrupt table from the schema via writable_schema
     (this avoids walking its damaged b-tree, which a normal DROP would do).
  3. VACUUM the copy. VACUUM rewrites the whole file from the live objects in
     sqlite_master, so it reclaims the orphaned pages AND rebuilds the freelist —
     both problems disappear.
  4. Recreate `embeddings_status` empty so the embed pipeline can repopulate it.
  5. PRAGMA integrity_check — refuse to report success unless it returns "ok".

    uv run python scripts/repair_db.py library.db library-clean.db

Then point the build at the clean DB:
    ANGLICAN_DB=$PWD/library-clean.db ANGLICAN_DEVICE=xpu \
    uv run python -m anglican_search.embed_library --phase embed
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import time

_EMB_SCHEMA = """
CREATE TABLE embeddings_status (
    chunk_id     INTEGER PRIMARY KEY,
    embedded_at  TEXT NOT NULL,
    FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
)
"""


def _log(msg: str) -> None:
    print(f"[repair] {msg}", flush=True)


def repair(src: str, dst: str) -> int:
    if not os.path.exists(src):
        _log(f"source not found: {src}")
        return 2
    if os.path.abspath(src) == os.path.abspath(dst):
        _log("source and destination must differ (source is never modified)")
        return 2
    if os.path.exists(dst):
        _log(f"destination already exists, refusing to overwrite: {dst}")
        return 2

    size_gb = os.path.getsize(src) / 1e9
    _log(f"copying {src} -> {dst} ({size_gb:.2f} GB) ...")
    t0 = time.time()
    shutil.copyfile(src, dst)
    _log(f"copied in {time.time() - t0:.1f}s")

    conn = sqlite3.connect(dst)
    try:
        # 1) Confirm the damage is where we think it is and the rest is readable.
        for tbl, expect_ok in (("books", True), ("chunks", True),
                               ("rechunk_status", True)):
            n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            _log(f"readable: {tbl} = {n:,} rows")

        # 2) Remove the corrupt table from the schema without walking its b-tree.
        _log("dropping corrupt embeddings_status from schema (writable_schema) ...")
        conn.execute("PRAGMA writable_schema = ON")
        conn.execute("DELETE FROM sqlite_master WHERE name = 'embeddings_status'")
        conn.execute("PRAGMA writable_schema = OFF")
        conn.commit()
    finally:
        conn.close()

    # 3) VACUUM in a fresh connection: rebuilds the file, fixes the freelist,
    #    reclaims the orphaned pages from the removed table.
    conn = sqlite3.connect(dst)
    try:
        _log("VACUUM (rebuilding the file — this is the slow step) ...")
        t1 = time.time()
        conn.execute("VACUUM")
        conn.commit()
        _log(f"VACUUM done in {time.time() - t1:.1f}s")

        # 4) Recreate the tracking table, empty.
        _log("recreating empty embeddings_status ...")
        conn.executescript(_EMB_SCHEMA)
        conn.commit()

        # 5) Verify.
        _log("integrity_check ...")
        rows = [r[0] for r in conn.execute("PRAGMA integrity_check").fetchall()]
        final_size = os.path.getsize(dst) / 1e9
    finally:
        conn.close()

    if rows == ["ok"]:
        _log(f"OK — {dst} is clean ({final_size:.2f} GB). integrity_check = ok")
        _log("embeddings_status is empty: the next embed run rebuilds the index "
             "from scratch (re-embeds all chunks).")
        return 0
    _log("integrity_check FAILED:")
    for r in rows[:20]:
        _log(f"  {r}")
    return 1


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else "library.db"
    dst = sys.argv[2] if len(sys.argv) > 2 else "library-clean.db"
    return repair(src, dst)


if __name__ == "__main__":
    raise SystemExit(main())
