"""The MCP endpoint, mounted into the web service at /mcp.

This is the primary way to use the library: an LLM's MCP client connects to
`/mcp` with a Bilson API key. It is authenticated by the *same* API keys as the
REST API, metered against the *same* monthly quota, and shares the *one* loaded
search engine — so MCP, the REST API, and the website are a single product.
"""

from __future__ import annotations

import contextvars
import json

from mcp.server.fastmcp import FastMCP

from anglican_search.mcp_tool import TOOL_DOC, format_results, run_search
from anglican_search.search import Filters

from . import accounts

# The authenticated user for the current MCP request. Set by the auth middleware
# in the event-loop task; anyio copies the context into the tool's worker thread.
_user: contextvars.ContextVar[dict | None] = contextvars.ContextVar("mcp_user", default=None)

# Shared search engine, injected by the web app at startup.
_engine: dict = {"searcher": None, "batcher": None}


def configure(searcher, batcher) -> None:
    _engine["searcher"] = searcher
    _engine["batcher"] = batcher


# stateless + JSON responses: each request is self-contained (clean behind a
# reverse proxy) and returns application/json rather than an SSE stream, which
# avoids any interaction with the app's HTTP middleware.
mcp = FastMCP("anglican-library", stateless_http=True, json_response=True)


@mcp.tool()
def search_anglican_library(
    query: str, top_k: int = 5, mode: str = "semantic", rerank: bool = True,
    deep: bool = False, author: str | None = None, category: str | None = None,
    year_min: int | None = None, year_max: int | None = None, title: str | None = None,
) -> str:
    searcher, batcher = _engine["searcher"], _engine["batcher"]
    if searcher is None:
        return "The library search engine is not ready yet."
    filters = Filters(author=author, category=category, title=title,
                      year_min=year_min, year_max=year_max)
    try:
        results, units, snippet = run_search(
            searcher, query, top_k=top_k, mode=mode, rerank=rerank, deep=deep,
            filters=filters, batcher=batcher)
    except Exception as e:  # noqa: BLE001
        return f"Search error: {e}"
    user = _user.get()
    if user is not None:
        accounts.record_use(user["id"], n=units)
    return format_results(results, query, snippet_chars=snippet)


search_anglican_library.__doc__ = TOOL_DOC  # single source of truth for the tool description

asgi = mcp.streamable_http_app()  # the mountable ASGI app (route at /mcp)


async def _send_json(send, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


class MCPAuthMiddleware:
    """Gate the mounted /mcp endpoint with a Bilson API key + monthly quota."""

    def __init__(self, app, prefix: str = "/mcp"):
        self.app = app
        self.prefix = prefix

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or not scope.get("path", "").startswith(self.prefix):
            return await self.app(scope, receive, send)
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        auth = headers.get(b"authorization", b"").decode()
        key = (auth[7:].strip() if auth.lower().startswith("bearer ")
               else headers.get(b"x-api-key", b"").decode())
        resolved = accounts.resolve_key(key) if key else None
        if not resolved:
            return await _send_json(send, 401, {"error": "invalid_api_key"})
        user, _k = resolved
        if accounts.usage_this_month(user["id"]) >= accounts.monthly_limit(user):
            return await _send_json(send, 429, {"error": "quota_exceeded"})
        token = _user.set(dict(user))
        try:
            await self.app(scope, receive, send)
        finally:
            _user.reset(token)
