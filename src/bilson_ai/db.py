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

-- Proposed books awaiting admin review/import.
CREATE TABLE IF NOT EXISTS submissions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER REFERENCES users(id) ON DELETE SET NULL,
    url         TEXT NOT NULL,
    source_type TEXT,                 -- ia | gutenberg | google
    source_id   TEXT,
    title       TEXT,
    note        TEXT,
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending|approved|imported|rejected|failed
    detail      TEXT,
    created_at  TEXT NOT NULL,
    reviewed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_sub_status ON submissions(status);

-- OAuth 2.1 authorization server (for MCP clients that connect via the spec's
-- auth flow, e.g. custom-connector UIs). Static API keys still work too.
CREATE TABLE IF NOT EXISTS oauth_clients (
    client_id     TEXT PRIMARY KEY,        -- issued by dynamic client registration
    client_name   TEXT,
    redirect_uris TEXT NOT NULL,           -- JSON array of allowed redirect URIs
    created_at    TEXT NOT NULL
);

-- Short-lived authorization codes (single-use; only the SHA-256 is stored).
CREATE TABLE IF NOT EXISTS oauth_codes (
    code_hash      TEXT PRIMARY KEY,
    client_id      TEXT NOT NULL,
    user_id        INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    redirect_uri   TEXT NOT NULL,
    code_challenge TEXT NOT NULL,          -- PKCE S256 challenge
    scope          TEXT,
    expires_at     INTEGER NOT NULL,       -- unix epoch
    created_at     TEXT NOT NULL
);

-- Access/refresh tokens (only SHA-256 stored, like API keys).
CREATE TABLE IF NOT EXISTS oauth_tokens (
    token_hash   TEXT PRIMARY KEY,
    refresh_hash TEXT UNIQUE,
    client_id    TEXT NOT NULL,
    user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    scope        TEXT,
    expires_at   INTEGER NOT NULL,         -- unix epoch (access token)
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_oauth_tokens_user ON oauth_tokens(user_id);
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
