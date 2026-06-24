"""MCP server exposing the Anglican library as a search tool over stdio.

Run directly:
    uv run python -m anglican_search.server
or via the installed console script:
    anglican-search-mcp

Paths default to config.DB_PATH / config.INDEX_PATH but can be overridden with
the ANGLICAN_DB and ANGLICAN_INDEX environment variables (handy for pointing at
a test index, or a moved library).

MCP client config example (in your client's MCP servers config):
    {
      "mcpServers": {
        "anglican-library": {
          "command": "uv",
          "args": ["run", "anglican-search-mcp"],
          "cwd": "C:\\\\path\\\\to\\\\anglican_search_starter"
        }
      }
    }
"""

from __future__ import annotations

import os
import sys

from mcp.server.fastmcp import FastMCP

from .config import DB_PATH, EMBEDDING_MODEL, INDEX_PATH
from .search import Filters, Searcher

mcp = FastMCP("anglican-library")

_searcher: Searcher | None = None


def _get_searcher() -> Searcher:
    global _searcher
    if _searcher is None:
        _searcher = Searcher(
            db_path=os.environ.get("ANGLICAN_DB", str(DB_PATH)),
            index_path=os.environ.get("ANGLICAN_INDEX", str(INDEX_PATH)),
            model_name=os.environ.get("ANGLICAN_MODEL", EMBEDDING_MODEL),
        )
    return _searcher


def _format(results: list[dict], query: str) -> str:
    if not results:
        return f'No results for "{query}".'
    lines = [f'Top {len(results)} results for "{query}":\n']
    for i, r in enumerate(results, 1):
        author = r.get("author") or "Unknown"
        year = f", {r['year']}" if r.get("year") else ""
        snippet = " ".join(r["text"].split())
        if len(snippet) > 1200:
            snippet = snippet[:1200] + " […]"
        lines.append(
            f"[{i}] {r['title']} — {author}{year} "
            f"(book_id {r['book_id']}, score {r['score']:.3f})\n"
            f"    source: {r.get('url') or 'n/a'}  chars {r.get('char_start')}-{r.get('char_end')}\n"
            f"    {snippet}\n"
        )
    return "\n".join(lines)


@mcp.tool()
def search_anglican_library(
    query: str,
    top_k: int = 5,
    mode: str = "semantic",
    rerank: bool = True,
    author: str | None = None,
    category: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    title: str | None = None,
) -> str:
    """Search a personal library of ~1,500 Anglican theological texts (OCR'd
    19th-century books, e.g. on the Trinity, Christology, the Athanasian Creed).

    Semantic search embeds the query, retrieves candidates with FAISS, and
    reorders them with a cross-encoder reranker for precision. Returns passages
    with attribution (title, author, year, book_id, source URL, char offsets)
    so results can be cited.

    Args:
        query: A natural-language question/topic (semantic) or keywords (literal).
        top_k: Number of passages to return (default 5, max 25).
        mode: "semantic" (meaning-based, default) or "literal" (exact FTS keywords).
        rerank: Apply the cross-encoder reranker to semantic results (default True).
        author: Restrict to books whose author contains this string (e.g. "Waterland").
        category: Restrict to a category (e.g. "Church History", "Liturgics").
        year_min: Only books published in or after this year.
        year_max: Only books published in or before this year.
        title: Restrict to books whose title contains this string.
    """
    top_k = max(1, min(int(top_k), 25))
    filters = Filters(author=author, category=category, title=title,
                      year_min=year_min, year_max=year_max)
    try:
        s = _get_searcher()
        if mode.lower() == "literal":
            results = s.literal(query, k=top_k, filters=filters)
        else:
            results = s.semantic(query, k=top_k, rerank=rerank, filters=filters)
    except FileNotFoundError:
        return ("The FAISS index is not built yet. Run the embedding step first:\n"
                "  uv run python -m anglican_search.embed_library --phase all")
    except Exception as e:  # noqa: BLE001 - surface a usable message to the model
        return f"Search error: {e}"
    return _format(results, query)


def _warmup() -> None:
    """Load the model + index (and run one encode) in the main thread before
    serving. Loading lazily inside a tool call deadlocks under FastMCP's async
    dispatch, and pre-warming also makes the first real query fast. If the index
    isn't built yet, semantic search is simply unavailable until it is.
    """
    try:
        # rerank=True forces the embedder, index, AND cross-encoder to all load
        # in the main thread now — loading any of them inside a tool call
        # deadlocks under FastMCP's async dispatch.
        _get_searcher().semantic("warmup", k=1, rerank=True)
        print("[anglican] model + index + reranker ready.", file=sys.stderr, flush=True)
    except FileNotFoundError:
        print("[anglican] no FAISS index yet — run embed_library first; "
              "literal search still works.", file=sys.stderr, flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[anglican] warmup skipped: {e}", file=sys.stderr, flush=True)


def main() -> None:
    print("[anglican] starting, warming up...", file=sys.stderr, flush=True)
    _warmup()
    mcp.run()  # stdio transport by default


if __name__ == "__main__":
    main()
