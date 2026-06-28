"""OAuth 2.1 authorization server for the MCP endpoint (per the MCP auth spec).

Lets MCP clients that don't take a pre-shared key — the one-click "custom
connector" UIs — connect by the standard flow instead: dynamic client
registration (RFC 7591) + authorization-code grant with PKCE (S256), issuing
bearer tokens bound to a Bilson user account. The static API keys still work for
the REST API and for clients (CLIs, curl) that send a key directly.

Security choices, mirroring accounts.py:
- Authorization codes and tokens are random (`secrets`), single-use for codes,
  and only their SHA-256 is stored — a DB leak exposes no usable code/token.
- PKCE S256 is required; the verifier is checked in constant time.
- Codes are short-lived (5 min) and bound to client_id + redirect_uri.
- Public clients only (no client secret); auth is the user's session at the
  consent step plus PKCE at the token step.

This module is pure logic + storage; the HTTP endpoints live in app.py (they need
the session and templates). The base URL is passed in so it works on any host.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timezone

from . import db

ACCESS_TTL = 3600            # access token: 1 hour
REFRESH_TTL = 30 * 24 * 3600  # refresh token: 30 days (kept for issued_at math)
CODE_TTL = 300               # authorization code: 5 minutes
DEFAULT_SCOPE = "search"


class OAuthError(Exception):
    """An OAuth error to surface as {"error": code, "error_description": desc}."""

    def __init__(self, code: str, desc: str = "") -> None:
        self.code = code
        self.desc = desc
        super().__init__(code)


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _now() -> int:
    return int(time.time())


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _base(url: str) -> str:
    return url.rstrip("/")


# --- discovery metadata ----------------------------------------------------
def protected_resource_metadata(base_url: str) -> dict:
    """RFC 9728 — tells the client which authorization server protects /mcp."""
    base = _base(base_url)
    return {
        "resource": f"{base}/mcp",
        "authorization_servers": [base],
        "bearer_methods_supported": ["header"],
        "scopes_supported": [DEFAULT_SCOPE],
    }


def authorization_server_metadata(base_url: str) -> dict:
    """RFC 8414 — the authorization server's endpoints and capabilities."""
    base = _base(base_url)
    return {
        "issuer": base,
        "authorization_endpoint": f"{base}/oauth/authorize",
        "token_endpoint": f"{base}/oauth/token",
        "registration_endpoint": f"{base}/oauth/register",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
        "scopes_supported": [DEFAULT_SCOPE],
    }


# --- dynamic client registration (RFC 7591) --------------------------------
def register_client(client_name: str, redirect_uris: list[str]) -> str:
    client_id = "mcp_" + secrets.token_urlsafe(16)
    conn = db.connect()
    try:
        conn.execute(
            "INSERT INTO oauth_clients (client_id, client_name, redirect_uris, created_at) "
            "VALUES (?,?,?,?)",
            (client_id, client_name or "", json.dumps(list(redirect_uris)), _now_str()),
        )
        conn.commit()
    finally:
        conn.close()
    return client_id


def get_client(client_id: str):
    conn = db.connect()
    try:
        return conn.execute(
            "SELECT * FROM oauth_clients WHERE client_id=?", (client_id,)
        ).fetchone()
    finally:
        conn.close()


def client_allows_redirect(client_id: str, redirect_uri: str) -> bool:
    c = get_client(client_id)
    if not c or not redirect_uri:
        return False
    try:
        return redirect_uri in json.loads(c["redirect_uris"])
    except Exception:  # noqa: BLE001
        return False


# --- authorization code ----------------------------------------------------
def create_code(client_id: str, user_id: int, redirect_uri: str,
                code_challenge: str, scope: str | None) -> str:
    code = secrets.token_urlsafe(32)
    conn = db.connect()
    try:
        conn.execute(
            "INSERT INTO oauth_codes (code_hash, client_id, user_id, redirect_uri, "
            "code_challenge, scope, expires_at, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (_sha256(code), client_id, user_id, redirect_uri, code_challenge,
             scope or DEFAULT_SCOPE, _now() + CODE_TTL, _now_str()),
        )
        conn.commit()
    finally:
        conn.close()
    return code


def verify_pkce(verifier: str, challenge: str) -> bool:
    """PKCE S256: base64url(sha256(verifier)) == challenge, compared constant-time."""
    if not verifier or not challenge:
        return False
    digest = hashlib.sha256(verifier.encode()).digest()
    calc = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return hmac.compare_digest(calc, challenge)


def _pop_code(code: str):
    """Fetch and delete (single-use) a code row by its hash."""
    h = _sha256(code) if code else ""
    conn = db.connect()
    try:
        row = conn.execute("SELECT * FROM oauth_codes WHERE code_hash=?", (h,)).fetchone()
        if row:
            conn.execute("DELETE FROM oauth_codes WHERE code_hash=?", (h,))
            conn.commit()
        return row
    finally:
        conn.close()


def exchange_code(code: str, client_id: str, redirect_uri: str,
                  code_verifier: str) -> dict:
    """Authorization-code grant. Returns an OAuth token response or raises OAuthError."""
    row = _pop_code(code)  # single-use: gone after this, even on failure
    if not row:
        raise OAuthError("invalid_grant", "unknown or already-used authorization code")
    if row["expires_at"] < _now():
        raise OAuthError("invalid_grant", "authorization code expired")
    if row["client_id"] != client_id:
        raise OAuthError("invalid_grant", "client_id mismatch")
    if row["redirect_uri"] != redirect_uri:
        raise OAuthError("invalid_grant", "redirect_uri mismatch")
    if not verify_pkce(code_verifier, row["code_challenge"]):
        raise OAuthError("invalid_grant", "PKCE verification failed")
    return _issue_tokens(client_id, row["user_id"], row["scope"])


def _issue_tokens(client_id: str, user_id: int, scope: str | None) -> dict:
    access = secrets.token_urlsafe(32)
    refresh = secrets.token_urlsafe(32)
    conn = db.connect()
    try:
        conn.execute(
            "INSERT INTO oauth_tokens (token_hash, refresh_hash, client_id, user_id, "
            "scope, expires_at, created_at) VALUES (?,?,?,?,?,?,?)",
            (_sha256(access), _sha256(refresh), client_id, user_id,
             scope or DEFAULT_SCOPE, _now() + ACCESS_TTL, _now_str()),
        )
        conn.commit()
    finally:
        conn.close()
    return {
        "access_token": access,
        "token_type": "Bearer",
        "expires_in": ACCESS_TTL,
        "refresh_token": refresh,
        "scope": scope or DEFAULT_SCOPE,
    }


def refresh(refresh_token: str, client_id: str) -> dict:
    """Refresh-token grant (rotates the refresh token)."""
    h = _sha256(refresh_token) if refresh_token else ""
    conn = db.connect()
    try:
        row = conn.execute("SELECT * FROM oauth_tokens WHERE refresh_hash=?", (h,)).fetchone()
        if row:
            conn.execute("DELETE FROM oauth_tokens WHERE refresh_hash=?", (h,))
            conn.commit()
    finally:
        conn.close()
    if not row:
        raise OAuthError("invalid_grant", "unknown refresh token")
    if row["client_id"] != client_id:
        raise OAuthError("invalid_grant", "client_id mismatch")
    return _issue_tokens(client_id, row["user_id"], row["scope"])


def resolve_token(access_token: str):
    """Return the active user row for a valid, unexpired access token, else None.
    Used by the MCP auth middleware (same shape as accounts.resolve_key's user)."""
    if not access_token:
        return None
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT * FROM oauth_tokens WHERE token_hash=?", (_sha256(access_token),)
        ).fetchone()
        if not row or row["expires_at"] < _now():
            return None
        return conn.execute(
            "SELECT * FROM users WHERE id=? AND is_active=1", (row["user_id"],)
        ).fetchone()
    finally:
        conn.close()
