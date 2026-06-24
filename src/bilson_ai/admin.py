"""Admin bootstrap CLI.

    uv run python -m bilson_ai.admin create-admin you@example.com 'a-strong-password'

Creates the user if needed and grants admin. Re-run to promote an existing user.
"""

from __future__ import annotations

import sys

from . import accounts
from .db import init_db


def main() -> None:
    init_db()
    args = sys.argv[1:]
    if len(args) == 3 and args[0] == "create-admin":
        email, password = args[1].strip().lower(), args[2]
        existing = accounts.get_user_by_email(email)
        if existing:
            accounts.admin_set(existing["id"], is_admin=1, is_active=1)
            print(f"Promoted {email} to admin.")
        else:
            uid = accounts.create_user(email, password, is_admin=True)
            print(f"Created admin {email} (id={uid}).")
    else:
        print("usage: python -m bilson_ai.admin create-admin <email> <password>")
        sys.exit(1)


if __name__ == "__main__":
    main()
