#!/usr/bin/env bash
# Entry point for the embed job. Ensures the SQLite DB is present (optionally
# staging it from ANGLICAN_DB_URL on first run, so a persistent PVC only
# downloads once), then runs the resumable embed pipeline.
set -euo pipefail

DB="${ANGLICAN_DB:-/data/library.db}"
INDEX="${ANGLICAN_INDEX:-/data/index.faiss}"
mkdir -p "$(dirname "$DB")" "$(dirname "$INDEX")"

if [ ! -f "$DB" ] && [ -n "${ANGLICAN_DB_URL:-}" ]; then
  echo "[entrypoint] $DB missing — downloading from ANGLICAN_DB_URL ..."
  curl -fSL "$ANGLICAN_DB_URL" -o "$DB"
fi

if [ ! -f "$DB" ]; then
  echo "[entrypoint] ERROR: $DB not found. Mount the database at /data, or set" >&2
  echo "             ANGLICAN_DB_URL to download it on first run." >&2
  exit 1
fi

echo "[entrypoint] DB=$DB  INDEX=$INDEX"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || \
  echo "[entrypoint] WARNING: no GPU visible — embedding will run on CPU (slow)."

# Resumable: re-running continues from embeddings_status if interrupted.
exec python -m anglican_search.embed_library "$@"
