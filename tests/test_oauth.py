"""Gate tests for the MCP OAuth 2.1 authorization server (bilson_ai/oauth.py).

Covers the security-critical paths: PKCE S256, single-use/expiring authorization
codes, redirect-uri binding, token resolution, and refresh rotation. Uses a
throwaway SQLite DB (no network, no models), so it's fast and deterministic.
"""

from __future__ import annotations

import base64
import hashlib
import secrets

import pytest

from bilson_ai import db, oauth


@pytest.fixture
def tmpdb(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "ACCOUNTS_DB", str(tmp_path / "accounts.db"))
    db.init_db()


def _make_user(email: str = "a@b.com") -> int:
    conn = db.connect()
    try:
        cur = conn.execute(
            "INSERT INTO users (email, password_hash, is_admin, is_active, created_at) "
            "VALUES (?,?,0,1,?)", (email, "x", oauth._now_str()))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def _pkce():
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


# --- PKCE (pure) ----------------------------------------------------------
def test_pkce_roundtrip():
    v, c = _pkce()
    assert oauth.verify_pkce(v, c)


def test_pkce_rejects_wrong_verifier():
    _, c = _pkce()
    assert not oauth.verify_pkce("not-the-verifier", c)


def test_pkce_rejects_empty():
    assert not oauth.verify_pkce("", "x")
    assert not oauth.verify_pkce("x", "")


# --- discovery metadata (pure) --------------------------------------------
def test_metadata_uses_base_url_without_trailing_slash():
    prm = oauth.protected_resource_metadata("https://lib.example/")
    assert prm["resource"] == "https://lib.example/mcp"
    assert prm["authorization_servers"] == ["https://lib.example"]
    asm = oauth.authorization_server_metadata("https://lib.example")
    assert asm["issuer"] == "https://lib.example"
    assert asm["authorization_endpoint"] == "https://lib.example/oauth/authorize"
    assert asm["token_endpoint"] == "https://lib.example/oauth/token"
    assert asm["registration_endpoint"] == "https://lib.example/oauth/register"
    assert asm["code_challenge_methods_supported"] == ["S256"]
    assert asm["grant_types_supported"] == ["authorization_code", "refresh_token"]


# --- client registration / redirect binding -------------------------------
def test_register_and_redirect_allowlist(tmpdb):
    cid = oauth.register_client("Test App", ["https://app/cb"])
    assert cid.startswith("mcp_")
    assert oauth.client_allows_redirect(cid, "https://app/cb")
    assert not oauth.client_allows_redirect(cid, "https://evil/cb")
    assert not oauth.client_allows_redirect("nonexistent", "https://app/cb")


# --- full authorization-code + PKCE flow ----------------------------------
def test_full_flow_issues_and_resolves_token(tmpdb):
    uid = _make_user()
    cid = oauth.register_client("App", ["https://app/cb"])
    v, c = _pkce()
    code = oauth.create_code(cid, uid, "https://app/cb", c, "search")
    tok = oauth.exchange_code(code, cid, "https://app/cb", v)
    assert tok["token_type"] == "Bearer" and tok["access_token"]
    user = oauth.resolve_token(tok["access_token"])
    assert user is not None and user["id"] == uid


def test_code_is_single_use(tmpdb):
    uid = _make_user()
    cid = oauth.register_client("App", ["https://app/cb"])
    v, c = _pkce()
    code = oauth.create_code(cid, uid, "https://app/cb", c, "search")
    oauth.exchange_code(code, cid, "https://app/cb", v)
    with pytest.raises(oauth.OAuthError):
        oauth.exchange_code(code, cid, "https://app/cb", v)


def test_wrong_verifier_rejected(tmpdb):
    uid = _make_user()
    cid = oauth.register_client("App", ["https://app/cb"])
    _, c = _pkce()
    code = oauth.create_code(cid, uid, "https://app/cb", c, "search")
    with pytest.raises(oauth.OAuthError):
        oauth.exchange_code(code, cid, "https://app/cb", "wrong-verifier")


def test_redirect_uri_mismatch_rejected(tmpdb):
    uid = _make_user()
    cid = oauth.register_client("App", ["https://app/cb"])
    v, c = _pkce()
    code = oauth.create_code(cid, uid, "https://app/cb", c, "search")
    with pytest.raises(oauth.OAuthError):
        oauth.exchange_code(code, cid, "https://other/cb", v)


def test_client_mismatch_rejected(tmpdb):
    uid = _make_user()
    cid = oauth.register_client("App", ["https://app/cb"])
    v, c = _pkce()
    code = oauth.create_code(cid, uid, "https://app/cb", c, "search")
    with pytest.raises(oauth.OAuthError):
        oauth.exchange_code(code, "some-other-client", "https://app/cb", v)


def test_expired_code_rejected(tmpdb, monkeypatch):
    uid = _make_user()
    cid = oauth.register_client("App", ["https://app/cb"])
    v, c = _pkce()
    code = oauth.create_code(cid, uid, "https://app/cb", c, "search")
    monkeypatch.setattr(oauth, "_now", lambda: 9_999_999_999)  # far future
    with pytest.raises(oauth.OAuthError):
        oauth.exchange_code(code, cid, "https://app/cb", v)


# --- token resolution + refresh -------------------------------------------
def test_resolve_token_rejects_garbage(tmpdb):
    assert oauth.resolve_token("") is None
    assert oauth.resolve_token("not-a-real-token") is None


def test_expired_access_token_not_resolved(tmpdb, monkeypatch):
    uid = _make_user()
    cid = oauth.register_client("App", ["https://app/cb"])
    v, c = _pkce()
    tok = oauth.exchange_code(
        oauth.create_code(cid, uid, "https://app/cb", c, "search"), cid, "https://app/cb", v)
    monkeypatch.setattr(oauth, "_now", lambda: 9_999_999_999)
    assert oauth.resolve_token(tok["access_token"]) is None


def test_refresh_rotates_and_invalidates_old(tmpdb):
    uid = _make_user()
    cid = oauth.register_client("App", ["https://app/cb"])
    v, c = _pkce()
    tok = oauth.exchange_code(
        oauth.create_code(cid, uid, "https://app/cb", c, "search"), cid, "https://app/cb", v)
    tok2 = oauth.refresh(tok["refresh_token"], cid)
    assert tok2["access_token"] != tok["access_token"]
    assert oauth.resolve_token(tok2["access_token"])["id"] == uid
    with pytest.raises(oauth.OAuthError):       # old refresh token is rotated out
        oauth.refresh(tok["refresh_token"], cid)


def test_refresh_wrong_client_rejected(tmpdb):
    uid = _make_user()
    cid = oauth.register_client("App", ["https://app/cb"])
    v, c = _pkce()
    tok = oauth.exchange_code(
        oauth.create_code(cid, uid, "https://app/cb", c, "search"), cid, "https://app/cb", v)
    with pytest.raises(oauth.OAuthError):
        oauth.refresh(tok["refresh_token"], "other-client")
