"""SQLite schema + connection for accounts, API keys, and usage."""

from __future__ import annotations

import sqlite3

from .config import ACCOUNTS_DB

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    is_admin      INTEGER NOT NULL DEFAULT 0,
    is_active     INTEGER NOT NULL DEFAULT 1,
    monthly_limit INTEGER,                 -- NULL => use the global default
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash     TEXT UNIQUE NOT NULL,     -- sha256 of the key (key shown once)
    key_prefix   TEXT NOT NULL,            -- e.g. bk_ab12cd34 for display
    name         TEXT,
    is_active    INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT NOT NULL,
    last_used_at TEXT
);

-- Monthly usage aggregated per user (period = 'YYYY-MM').
CREATE TABLE IF NOT EXISTS usage (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    period  TEXT NOT NULL,
    count   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, period)
);

CREATE INDEX IF NOT EXISTS idx_keys_user ON api_keys(user_id);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(ACCOUNTS_DB), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def init_db() -> None:
    conn = connect()
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()
