"""Bilson AI web service: marketing site, accounts, API keys, admin, REST API.

Run:  uv run bilson-ai           (uvicorn on BILSON_HOST:BILSON_PORT)
The search model is loaded once at startup and shared by every request.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from anglican_search.batch import BatchedSearch
from anglican_search.config import DEEP_MAX_TOP_K, MAX_TOP_K
from anglican_search.search import Filters, get_searcher

from . import accounts, mcp_app as mcpmod, submissions
from .config import (BRAND, DEFAULT_MONTHLY_LIMIT, HOST, PORT, PUBLIC_URL,
                     SECRET_KEY, TAGLINE, ADMIN_EMAIL)
from .db import init_db
from .importer import Importer

_HERE = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(_HERE / "templates"))
templates.env.globals["brand"] = BRAND


class RateLimiter:
    """Simple in-memory sliding-window limiter (per process / single worker)."""

    def __init__(self, max_hits: int, window_sec: float):
        self.max = max_hits
        self.window = window_sec
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._hits.setdefault(key, [])
            cutoff = now - self.window
            drop = 0
            while drop < len(q) and q[drop] < cutoff:
                drop += 1
            if drop:
                del q[:drop]
            if len(q) >= self.max:
                return False
            q.append(now)
            return True


# Manual UI search throttle (the API path is metered separately by quota).
_ui_limiter = RateLimiter(int(os.environ.get("BILSON_UI_RATE", "30")), 60.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Load + warm the search engine once (embedder + index + reranker), in the
    # main thread at startup — lazy-loading inside a request deadlocks.
    app.state.searcher = get_searcher()
    app.state.batcher = None
    app.state.importer = None
    try:
        app.state.searcher.semantic("warmup", k=1, rerank=True)
        # Dynamic micro-batching layer for the high-traffic search endpoints.
        app.state.batcher = BatchedSearch(app.state.searcher)
        # Background worker that imports approved book submissions.
        app.state.importer = Importer(app.state.searcher,
                                      app.state.searcher.db_path,
                                      app.state.searcher.index_path)
        mcpmod.configure(app.state.searcher, app.state.batcher)
        print("[bilson] ready: MCP (/mcp) + REST (/v1) + website, one shared engine.", flush=True)
    except Exception as e:  # noqa: BLE001 - serve the site even if the index isn't built yet
        print(f"[bilson] search engine unavailable ({e}); search will error.", flush=True)
    # Run the mounted MCP endpoint's session manager for the life of the app.
    async with mcpmod.mcp.session_manager.run():
        yield


app = FastAPI(title=BRAND, docs_url=None, redoc_url=None, lifespan=lifespan)
# /mcp is gated by Bilson API keys + quota (pure-ASGI middleware, runs before routing).
app.add_middleware(mcpmod.MCPAuthMiddleware)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, same_site="lax", https_only=False)
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    return resp


# --- helpers ---------------------------------------------------------------
def current_user(request: Request):
    uid = request.session.get("user_id")
    return accounts.get_user(uid) if uid else None


def page(request: Request, name: str, **ctx):
    # Starlette 1.x signature: request first; it's injected into the context.
    return templates.TemplateResponse(request, name, {"user": current_user(request), **ctx})


def _api_key_from(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key")


# --- marketing / static pages ---------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return page(request, "index.html", tagline=TAGLINE)


@app.get("/docs", response_class=HTMLResponse)
def docs(request: Request):
    mcp_url = (PUBLIC_URL or str(request.base_url).rstrip("/")) + "/mcp"
    return page(request, "docs.html", mcp_url=mcp_url)


@app.get("/legal", response_class=HTMLResponse)
def legal(request: Request):
    return page(request, "legal.html")


@app.get("/health")
def health(request: Request):
    ok = getattr(request.app.state, "searcher", None) is not None
    return {"status": "ok", "search_engine": "ready" if ok else "unavailable"}


# --- auth ------------------------------------------------------------------
@app.get("/signup", response_class=HTMLResponse)
def signup_form(request: Request):
    return page(request, "signup.html")


@app.post("/signup")
def signup(request: Request, email: str = Form(...), password: str = Form(...)):
    email = email.strip().lower()
    if "@" not in email or "." not in email.split("@")[-1]:
        return page(request, "signup.html", error="Enter a valid email address.")
    if len(password) < 8:
        return page(request, "signup.html", error="Password must be at least 8 characters.")
    if accounts.get_user_by_email(email):
        return page(request, "signup.html", error="That email is already registered.")
    uid = accounts.create_user(email, password, is_admin=bool(ADMIN_EMAIL) and email == ADMIN_EMAIL)
    request.session["user_id"] = uid
    return RedirectResponse("/dashboard", status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return page(request, "login.html")


@app.post("/login")
def login(request: Request, email: str = Form(...), password: str = Form(...)):
    user = accounts.authenticate(email, password)
    if not user:
        return page(request, "login.html", error="Invalid email or password.")
    request.session["user_id"] = user["id"]
    return RedirectResponse("/dashboard", status_code=303)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=303)


# --- dashboard / keys ------------------------------------------------------
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    user = current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    new_key = request.session.pop("new_key", None)
    mcp_url = (PUBLIC_URL or str(request.base_url).rstrip("/")) + "/mcp"
    return page(request, "dashboard.html",
                keys=accounts.list_keys(user["id"]),
                used=accounts.usage_this_month(user["id"]),
                limit=accounts.monthly_limit(user),
                new_key=new_key, mcp_url=mcp_url)


@app.post("/keys/create")
def keys_create(request: Request, name: str = Form("")):
    user = current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    request.session["new_key"] = accounts.create_api_key(user["id"], name.strip())
    return RedirectResponse("/dashboard", status_code=303)


@app.post("/keys/{key_id}/revoke")
def keys_revoke(request: Request, key_id: int):
    user = current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    accounts.revoke_key(user["id"], key_id)
    return RedirectResponse("/dashboard", status_code=303)


# --- admin -----------------------------------------------------------------
@app.get("/admin", response_class=HTMLResponse)
def admin(request: Request):
    user = current_user(request)
    if not user or not user["is_admin"]:
        return RedirectResponse("/login", status_code=303)
    return page(request, "admin.html", users=accounts.list_users(),
                default_limit=DEFAULT_MONTHLY_LIMIT)


@app.post("/admin/user/{user_id}")
def admin_update(request: Request, user_id: int, action: str = Form(...),
                 value: str = Form("")):
    admin_user = current_user(request)
    if not admin_user or not admin_user["is_admin"]:
        return RedirectResponse("/login", status_code=303)
    if action == "toggle_active":
        u = accounts.get_user(user_id)
        accounts.admin_set(user_id, is_active=0 if u["is_active"] else 1)
    elif action == "toggle_admin":
        u = accounts.get_user(user_id)
        accounts.admin_set(user_id, is_admin=0 if u["is_admin"] else 1)
    elif action == "set_limit" and value.strip().isdigit():
        accounts.admin_set(user_id, monthly_limit=int(value))
    return RedirectResponse("/admin", status_code=303)


# --- manual UI: search + browse (logged-in users) -------------------------
@app.get("/search", response_class=HTMLResponse)
def search_page(request: Request, q: str = "", mode: str = "semantic", top_k: int = 10):
    user = current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    results = []
    rate_limited = False
    searcher = getattr(request.app.state, "searcher", None)
    batcher = getattr(request.app.state, "batcher", None)
    if q.strip() and searcher is not None:
        if not _ui_limiter.allow(f"u{user['id']}"):
            rate_limited = True
        else:
            top_k = max(1, min(top_k, 25))
            if mode == "literal":
                results = searcher.literal(q, k=top_k)
            elif batcher is not None:
                results = batcher.search(q, k=top_k, rerank=True)
            else:
                results = searcher.semantic(q, k=top_k, rerank=True)
    return page(request, "search.html", q=q, mode=mode, top_k=top_k,
                results=results, rate_limited=rate_limited)


@app.get("/library", response_class=HTMLResponse)
def library(request: Request, q: str = "", category: str = "", p: int = 1):
    user = current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    searcher = getattr(request.app.state, "searcher", None)
    per = 40
    if searcher is None:
        return page(request, "library.html", books=[], total=0, categories=[],
                    q=q, category=category, p=1, per=per)
    p = max(1, p)
    return page(request, "library.html",
                books=searcher.list_books(q or None, category or None, per, (p - 1) * per),
                total=searcher.count_books(q or None, category or None),
                categories=searcher.categories(), q=q, category=category, p=p, per=per)


@app.get("/library/{book_id}", response_class=HTMLResponse)
def book_detail(request: Request, book_id: int, p: int = 1):
    user = current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    searcher = getattr(request.app.state, "searcher", None)
    book = searcher.get_book(book_id) if searcher else None
    if not book:
        return page(request, "book.html", book=None, chunks=[], total=0, p=1, per=25)
    per = 25
    p = max(1, p)
    return page(request, "book.html", book=book,
                chunks=searcher.book_chunks(book_id, per, (p - 1) * per),
                total=searcher.count_book_chunks(book_id), p=p, per=per)


# --- book submissions ------------------------------------------------------
@app.get("/submit", response_class=HTMLResponse)
def submit_form(request: Request):
    if not current_user(request):
        return RedirectResponse("/login", status_code=303)
    return page(request, "submit.html")


@app.post("/submit", response_class=HTMLResponse)
def submit(request: Request, url: str = Form(...), title: str = Form(""), note: str = Form("")):
    user = current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    stype, sid = submissions.parse_source(url)
    if stype is None:
        return page(request, "submit.html",
                    error="Unrecognised link. Use Internet Archive, Project Gutenberg, "
                          "Google Books, or HathiTrust.")
    searcher = getattr(request.app.state, "searcher", None)
    if searcher is not None and submissions.in_library(searcher.conn, stype, sid):
        return page(request, "submit.html", info="That book is already in the library.")
    if submissions.existing(stype, sid):
        return page(request, "submit.html", info="That book has already been submitted.")
    submissions.create(user["id"], url.strip(), stype, sid, title, note)
    if stype in submissions.AUTO_SOURCES:
        msg = "Thanks! Your submission was added to the review queue."
    else:
        msg = ("Thanks! Added to the queue. Google Books / HathiTrust text usually isn't "
               "downloadable, so an admin may import it manually.")
    return page(request, "submit.html", info=msg)


@app.get("/admin/queue", response_class=HTMLResponse)
def admin_queue(request: Request):
    user = current_user(request)
    if not user or not user["is_admin"]:
        return RedirectResponse("/login", status_code=303)
    return page(request, "queue.html",
                pending=submissions.by_status("pending"),
                needs_manual=submissions.by_status("needs_manual"),
                recent=(submissions.by_status("imported", 10)
                        + submissions.by_status("failed", 10)
                        + submissions.by_status("rejected", 10)))


@app.get("/admin/queue/{sub_id}", response_class=HTMLResponse)
def admin_submission(request: Request, sub_id: int):
    user = current_user(request)
    if not user or not user["is_admin"]:
        return RedirectResponse("/login", status_code=303)
    return page(request, "submission.html", sub=submissions.get(sub_id))


@app.post("/admin/queue/{sub_id}")
def admin_queue_action(request: Request, sub_id: int, action: str = Form(...)):
    user = current_user(request)
    if not user or not user["is_admin"]:
        return RedirectResponse("/login", status_code=303)
    if action == "approve":
        submissions.set_status(sub_id, "approved")
        importer = getattr(request.app.state, "importer", None)
        if importer is not None:
            importer.enqueue(sub_id)
        else:
            submissions.set_status(sub_id, "failed", "importer unavailable (index not loaded)")
    elif action == "reject":
        submissions.set_status(sub_id, "rejected")
    return RedirectResponse("/admin/queue", status_code=303)


@app.post("/admin/queue/{sub_id}/manual")
def admin_manual_import(request: Request, sub_id: int,
                        text: str = Form(...), title: str = Form("")):
    user = current_user(request)
    if not user or not user["is_admin"]:
        return RedirectResponse("/login", status_code=303)
    importer = getattr(request.app.state, "importer", None)
    if importer is None:
        submissions.set_status(sub_id, "failed", "importer unavailable (index not loaded)")
        return RedirectResponse("/admin/queue", status_code=303)
    if not text.strip():
        return RedirectResponse(f"/admin/queue/{sub_id}", status_code=303)
    if title.strip():
        submissions.set_title(sub_id, title.strip())
    submissions.set_status(sub_id, "approved")
    importer.enqueue(sub_id, text=text)
    return RedirectResponse("/admin/queue", status_code=303)


# --- REST API --------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    mode: str = "semantic"          # "semantic" | "literal"
    rerank: bool = True
    deep: bool = False              # recall mode: many passages, no rerank
    author: str | None = None
    category: str | None = None
    year_min: int | None = None
    year_max: int | None = None
    title: str | None = None


@app.get("/v1/verify")
def verify(request: Request):
    """Lightweight key check (used by the reverse proxy to gate the MCP route)."""
    raw = _api_key_from(request)
    if not raw or not accounts.resolve_key(raw):
        return JSONResponse({"error": "invalid_api_key"}, status_code=401)
    return {"ok": True}


@app.post("/v1/search")
def api_search(req: SearchRequest, request: Request):
    raw = _api_key_from(request)
    resolved = accounts.resolve_key(raw) if raw else None
    if not resolved:
        return JSONResponse({"error": "invalid_api_key"}, status_code=401)
    user, _key = resolved

    used, limit = accounts.usage_this_month(user["id"]), accounts.monthly_limit(user)
    if used >= limit:
        return JSONResponse(
            {"error": "quota_exceeded", "used": used, "limit": limit}, status_code=429)

    searcher = getattr(request.app.state, "searcher", None)
    batcher = getattr(request.app.state, "batcher", None)
    if searcher is None:
        return JSONResponse({"error": "search_unavailable"}, status_code=503)

    filters = Filters(author=req.author, category=req.category, title=req.title,
                      year_min=req.year_min, year_max=req.year_max)
    # Deep mode: return many passages by recall (no rerank) for a long-context
    # model to synthesize. It costs more data, so it counts as more quota units.
    if req.deep:
        top_k = max(1, min(req.top_k, DEEP_MAX_TOP_K))
        use_rerank = False
    else:
        top_k = max(1, min(req.top_k, MAX_TOP_K))
        use_rerank = req.rerank
    try:
        if req.mode == "literal":
            results = searcher.literal(req.query, k=top_k, filters=filters)
        elif batcher is not None:
            results = batcher.search(req.query, k=top_k, rerank=use_rerank, filters=filters)
        else:
            results = searcher.semantic(req.query, k=top_k, rerank=use_rerank, filters=filters)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": "search_error", "detail": str(e)}, status_code=500)

    accounts.record_use(user["id"], n=(-(-top_k // 25) if req.deep else 1))
    return {"query": req.query, "count": len(results), "deep": req.deep, "results": results}


@app.get("/robots.txt", response_class=PlainTextResponse)
def robots():
    return "User-agent: *\nDisallow: /dashboard\nDisallow: /admin\nDisallow: /v1/\nDisallow: /mcp\n"


# Mount the MCP endpoint last: the explicit routes above match first, and /mcp
# falls through to it. Auth + quota are enforced by MCPAuthMiddleware.
app.mount("/", mcpmod.asgi)


def main() -> None:
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
