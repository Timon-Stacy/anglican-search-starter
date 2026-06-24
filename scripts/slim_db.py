"""Make a slim serving copy of the SQLite store.

Drops `books.content` (the raw OCR text — only needed to *re-chunk*, never to
serve) and the `citations` table, then compacts. Saves ~1.7 GB, so the serving
box needs less disk and RAM. Run this on the build machine, then ship only the
slim DB + index.faiss to the server.

    uv run python scripts/slim_db.py library.db library-serve.db
"""

from __future__ import annotations

import os
import sqlite3
import sys


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "library.db"
    dst = sys.argv[2] if len(sys.argv) > 2 else "library-serve.db"
    if os.path.exists(dst):
        os.remove(dst)

    print(f"snapshot {src} -> {dst} ...", flush=True)
    c = sqlite3.connect(src)
    c.execute(f"VACUUM INTO '{dst}'")  # consistent, compacted copy (handles WAL)
    c.close()

    c = sqlite3.connect(dst)
    try:
        c.execute("ALTER TABLE books DROP COLUMN content")
    except sqlite3.OperationalError as e:
        print("  (content column already absent:", e, ")")
    c.execute("DROP TABLE IF EXISTS citations")
    c.commit()
    print("compacting ...", flush=True)
    c.execute("VACUUM")
    c.close()
    print(f"done: {dst} is {os.path.getsize(dst) / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
