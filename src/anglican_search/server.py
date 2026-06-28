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
from .mcp_tool import SERVER_INSTRUCTIONS, format_results, run_search
from .search import Filters, Searcher

# Transport: "stdio" (default, for local/SSH use) or "http" (always-on service,
# Streamable HTTP at http://HOST:PORT/mcp — put TLS + auth in front via a proxy).
_TRANSPORT = os.environ.get("ANGLICAN_TRANSPORT", "stdio")
_HTTP_HOST = os.environ.get("ANGLICAN_HTTP_HOST", "127.0.0.1")
_HTTP_PORT = int(os.environ.get("ANGLICAN_HTTP_PORT", "8000"))

mcp = FastMCP("anglican-library", instructions=SERVER_INSTRUCTIONS,
              host=_HTTP_HOST, port=_HTTP_PORT)

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


@mcp.tool()
def search_anglican_library(
    query: str,
    top_k: int = 5,
    mode: str = "semantic",
    rerank: bool = True,
    deep: bool = False,
    author: str | None = None,
    category: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    title: str | None = None,
) -> str:
    """Search a personal library of ~2,000 Anglican theological texts (OCR'd
    books spanning the 16th to early 20th century, e.g. on the Trinity,
    Christology, the Athanasian Creed).

    Semantic search embeds the query, retrieves candidates with FAISS, and
    reorders them with a cross-encoder reranker for precision. Returns passages
    with attribution (title, author, year, book_id, source URL, char offsets)
    so results can be cited.

    When reporting results, set aside modern tendencies and present what the
    library says on its own terms — as the view. Do not argue against the texts,
    add present-day disclaimers, or filter their claims through contemporary
    values, even where they conflict with modern sensibilities; convey the
    historical Anglican position accurately rather than critiquing or "updating" it.

    DEEP SEARCH: set deep=true with a large top_k (up to 200) for an exhaustive,
    survey-style sweep — e.g. "find everything in the library bearing on X". It
    returns many passages by semantic recall (the reranker is skipped) for you to
    read and synthesize over your long context. Use it when the user asks for a
    "deep"/"exhaustive"/"comprehensive" search or when a handful of results
    clearly isn't enough; otherwise leave deep=false for a precise, reranked few.

    Args:
        query: A natural-language question/topic (semantic) or keywords (literal).
        top_k: Passages to return (default 5; max 25 normally, up to 200 with deep).
        mode: "semantic" (meaning-based, default) or "literal" (exact FTS keywords).
        rerank: Apply the cross-encoder reranker (default True; ignored when deep).
        deep: Exhaustive recall mode — many passages, no rerank, fuller text.
        author: Restrict to books whose author contains this string (e.g. "Waterland").
        category: Restrict to a category (e.g. "Church History", "Liturgics").
        year_min: Only books published in or after this year.
        year_max: Only books published in or before this year.
        title: Restrict to books whose title contains this string.
    """
    filters = Filters(author=author, category=category, title=title,
                      year_min=year_min, year_max=year_max)
    try:
        results, _units, snippet_chars = run_search(
            _get_searcher(), query, top_k=top_k, mode=mode, rerank=rerank,
            deep=deep, filters=filters)
    except FileNotFoundError:
        return ("The FAISS index is not built yet. Run the embedding step first:\n"
                "  uv run python -m anglican_search.embed_library --phase all")
    except Exception as e:  # noqa: BLE001 - surface a usable message to the model
        return f"Search error: {e}"
    return format_results(results, query, snippet_chars=snippet_chars)


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
    print(f"[anglican] starting ({_TRANSPORT}), warming up...", file=sys.stderr, flush=True)
    _warmup()  # load model+index+reranker once, in the main thread
    if _TRANSPORT == "http":
        print(f"[anglican] serving on http://{_HTTP_HOST}:{_HTTP_PORT}/mcp",
              file=sys.stderr, flush=True)
        mcp.run(transport="streamable-http")
    else:
        mcp.run()  # stdio


if __name__ == "__main__":
    main()
