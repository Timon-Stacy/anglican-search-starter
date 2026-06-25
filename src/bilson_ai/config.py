"""Bilson AI configuration (all env-overridable for deployment)."""

from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

BRAND = "Bilson AI"
TAGLINE = "An MCP server that lets your AI search a curated library of historical Anglican theology — and cite it."

# Public base URL (no trailing slash), used to show the MCP endpoint on the
# dashboard. In production set e.g. BILSON_PUBLIC_URL=https://library.example.
# Falls back to the request's own base URL when unset.
PUBLIC_URL = os.environ.get("BILSON_PUBLIC_URL", "").rstrip("/")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Accounts/keys/usage live in their own DB (separate from the search data).
ACCOUNTS_DB = Path(os.environ.get("BILSON_DB", str(PROJECT_ROOT / "accounts.db")))

# Signed-session secret. MUST be set in production (export BILSON_SECRET=...);
# a random one is generated otherwise so dev works, but sessions won't survive a
# restart.
SECRET_KEY = os.environ.get("BILSON_SECRET", "")
if not SECRET_KEY:
    SECRET_KEY = secrets.token_hex(32)
    print("[bilson] WARNING: BILSON_SECRET not set — using an ephemeral secret "
          "(sessions reset on restart).", file=sys.stderr)

# Default monthly request quota per user (override per-user in the admin panel).
DEFAULT_MONTHLY_LIMIT = int(os.environ.get("BILSON_MONTHLY_LIMIT", "1000"))

HOST = os.environ.get("BILSON_HOST", "127.0.0.1")
PORT = int(os.environ.get("BILSON_PORT", "8001"))

# First-run admin bootstrap (optional): if set, this email is granted admin on
# signup. Otherwise use `python -m bilson_ai.admin create-admin ...`.
ADMIN_EMAIL = os.environ.get("BILSON_ADMIN_EMAIL", "").strip().lower()

API_KEY_PREFIX = "bk_"
