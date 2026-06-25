"""User book submissions: URL parsing, library/queue dedup, and CRUD."""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone

from . import db

# Maps source type -> the books-table column that stores its native id.
SOURCE_COLUMN = {"ia": "ia_title_id", "gutenberg": "gutenberg_id", "google": "gb_title_id"}


def parse_source(url: str) -> tuple[str | None, str | None]:
    """Detect the source + id from a submitted URL (mirrors download.py)."""
    url = (url or "").strip()
    m = re.search(r"archive\.org/details/([^/?#]+)", url)
    if m:
        return "ia", m.group(1)
    m = re.search(r"gutenberg\.org/ebooks/(\d+)", url)
    if m:
        return "gutenberg", m.group(1)
    if "books.google." in url:
        return "google", ""  # recognised, but not auto-importable
    return None, None


def in_library(library_conn: sqlite3.Connection, source_type: str, source_id: str) -> bool:
    col = SOURCE_COLUMN.get(source_type)
    if not col or not source_id:
        return False
    row = library_conn.execute(
        f"SELECT 1 FROM books WHERE {col} = ? LIMIT 1", (source_id,)
    ).fetchone()
    return row is not None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def existing(source_type: str, source_id: str) -> sqlite3.Row | None:
    conn = db.connect()
    try:
        return conn.execute(
            "SELECT * FROM submissions WHERE source_type=? AND source_id=? "
            "AND status IN ('pending','approved','imported') LIMIT 1",
            (source_type, source_id),
        ).fetchone()
    finally:
        conn.close()


def create(user_id: int, url: str, source_type: str, source_id: str,
           title: str = "", note: str = "") -> int:
    conn = db.connect()
    try:
        cur = conn.execute(
            "INSERT INTO submissions (user_id, url, source_type, source_id, title, note, "
            "status, created_at) VALUES (?,?,?,?,?,?, 'pending', ?)",
            (user_id, url, source_type, source_id, title.strip(), note.strip(), _now()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def by_status(status: str, limit: int = 200) -> list[sqlite3.Row]:
    conn = db.connect()
    try:
        return conn.execute(
            """SELECT s.*, u.email AS submitter
               FROM submissions s LEFT JOIN users u ON u.id = s.user_id
               WHERE s.status = ? ORDER BY s.created_at DESC LIMIT ?""",
            (status, limit),
        ).fetchall()
    finally:
        conn.close()


def get(sub_id: int) -> sqlite3.Row | None:
    conn = db.connect()
    try:
        return conn.execute("SELECT * FROM submissions WHERE id=?", (sub_id,)).fetchone()
    finally:
        conn.close()


def set_status(sub_id: int, status: str, detail: str | None = None) -> None:
    conn = db.connect()
    try:
        conn.execute(
            "UPDATE submissions SET status=?, detail=COALESCE(?, detail), reviewed_at=? WHERE id=?",
            (status, detail, _now(), sub_id),
        )
        conn.commit()
    finally:
        conn.close()
