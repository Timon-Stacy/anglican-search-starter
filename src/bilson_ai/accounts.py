"""Account, API-key, and usage logic — including password/key hashing.

Security choices:
- Passwords: PBKDF2-HMAC-SHA256, 200k iterations, per-user random salt, stored
  as `pbkdf2_sha256$<iters>$<salt_hex>$<hash_hex>`; verified in constant time.
- API keys: shown to the user once; only the SHA-256 of the key is stored, so a
  DB leak does not expose usable keys.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import time
from datetime import datetime, timezone

from . import db
from .config import API_KEY_PREFIX, DEFAULT_MONTHLY_LIMIT

_PBKDF2_ITERS = 200_000


# --- password hashing ------------------------------------------------------
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ITERS)
    return f"pbkdf2_sha256${_PBKDF2_ITERS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_hex, hash_hex = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(),
                                 bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _period() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


# --- users -----------------------------------------------------------------
def create_user(email: str, password: str, is_admin: bool = False) -> int:
    email = email.strip().lower()
    conn = db.connect()
    try:
        cur = conn.execute(
            "INSERT INTO users (email, password_hash, is_admin, created_at) VALUES (?,?,?,?)",
            (email, hash_password(password), int(is_admin), _now()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_user_by_email(email: str) -> sqlite3.Row | None:
    conn = db.connect()
    try:
        return conn.execute("SELECT * FROM users WHERE email=?",
                            (email.strip().lower(),)).fetchone()
    finally:
        conn.close()


def get_user(user_id: int) -> sqlite3.Row | None:
    conn = db.connect()
    try:
        return conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    finally:
        conn.close()


def authenticate(email: str, password: str) -> sqlite3.Row | None:
    user = get_user_by_email(email)
    if user and user["is_active"] and verify_password(password, user["password_hash"]):
        return user
    return None


# --- API keys --------------------------------------------------------------
def create_api_key(user_id: int, name: str = "") -> str:
    """Create a key, store only its hash, and return the full key (shown once)."""
    raw = API_KEY_PREFIX + secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw.encode()).hexdigest()
    conn = db.connect()
    try:
        conn.execute(
            "INSERT INTO api_keys (user_id, key_hash, key_prefix, name, created_at) "
            "VALUES (?,?,?,?,?)",
            (user_id, key_hash, raw[:11], name or "default", _now()),
        )
        conn.commit()
    finally:
        conn.close()
    return raw


def list_keys(user_id: int) -> list[sqlite3.Row]:
    conn = db.connect()
    try:
        return conn.execute(
            "SELECT * FROM api_keys WHERE user_id=? ORDER BY created_at DESC", (user_id,)
        ).fetchall()
    finally:
        conn.close()


def revoke_key(user_id: int, key_id: int) -> None:
    conn = db.connect()
    try:
        conn.execute("UPDATE api_keys SET is_active=0 WHERE id=? AND user_id=?",
                     (key_id, user_id))
        conn.commit()
    finally:
        conn.close()


def resolve_key(raw_key: str) -> tuple[sqlite3.Row, sqlite3.Row] | None:
    """Return (user, key) for a valid, active key whose owner is active."""
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    conn = db.connect()
    try:
        key = conn.execute(
            "SELECT * FROM api_keys WHERE key_hash=? AND is_active=1", (key_hash,)
        ).fetchone()
        if not key:
            return None
        user = conn.execute("SELECT * FROM users WHERE id=? AND is_active=1",
                            (key["user_id"],)).fetchone()
        if not user:
            return None
        conn.execute("UPDATE api_keys SET last_used_at=? WHERE id=?", (_now(), key["id"]))
        conn.commit()
        return user, key
    finally:
        conn.close()


# --- usage / quota ---------------------------------------------------------
def monthly_limit(user: sqlite3.Row) -> int:
    return user["monthly_limit"] if user["monthly_limit"] is not None else DEFAULT_MONTHLY_LIMIT


def usage_this_month(user_id: int) -> int:
    conn = db.connect()
    try:
        row = conn.execute("SELECT count FROM usage WHERE user_id=? AND period=?",
                           (user_id, _period())).fetchone()
        return row["count"] if row else 0
    finally:
        conn.close()


def record_use(user_id: int, n: int = 1) -> None:
    conn = db.connect()
    try:
        conn.execute(
            "INSERT INTO usage (user_id, period, count) VALUES (?,?,?) "
            "ON CONFLICT(user_id, period) DO UPDATE SET count = count + ?",
            (user_id, _period(), n, n),
        )
        conn.commit()
    finally:
        conn.close()


# --- admin -----------------------------------------------------------------
def list_users() -> list[sqlite3.Row]:
    conn = db.connect()
    try:
        return conn.execute(
            """SELECT u.*,
                      (SELECT COUNT(*) FROM api_keys k WHERE k.user_id=u.id AND k.is_active=1) AS active_keys,
                      (SELECT count FROM usage g WHERE g.user_id=u.id AND g.period=?) AS used
               FROM users u ORDER BY u.created_at DESC""",
            (_period(),),
        ).fetchall()
    finally:
        conn.close()


def admin_set(user_id: int, *, is_active: int | None = None,
              is_admin: int | None = None, monthly_limit: int | None = None) -> None:
    sets, params = [], []
    if is_active is not None:
        sets.append("is_active=?"); params.append(is_active)
    if is_admin is not None:
        sets.append("is_admin=?"); params.append(is_admin)
    if monthly_limit is not None:
        sets.append("monthly_limit=?"); params.append(monthly_limit)
    if not sets:
        return
    params.append(user_id)
    conn = db.connect()
    try:
        conn.execute(f"UPDATE users SET {', '.join(sets)} WHERE id=?", params)
        conn.commit()
    finally:
        conn.close()
