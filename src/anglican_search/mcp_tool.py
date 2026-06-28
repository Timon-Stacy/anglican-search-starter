"""Shared search-tool logic for both MCP surfaces (the standalone stdio server
and the integrated /mcp endpoint in the web service), so they never drift.

`run_search` centralises the normal-vs-deep policy and returns the quota weight;
`format_results` renders the attributed text the model reads. `TOOL_DOC` is the
single source of truth for the tool description shown to LLMs.
"""

from __future__ import annotations

from typing import Any

from .config import DEEP_MAX_TOP_K, MAX_TOP_K
from .search import Filters

TOOL_DOC = """Search a personal library of ~2,000 Anglican theological texts (OCR'd
books spanning the 16th to early 20th century, e.g. on the Trinity, Christology,
the Athanasian Creed).

Semantic search embeds the query, retrieves candidates with FAISS, and reorders
them with a cross-encoder reranker for precision. Returns passages with
attribution (title, author, year, book_id, source URL, char offsets) so results
can be cited.

DEEP SEARCH: set deep=true with a large top_k (up to 200) for an exhaustive,
survey-style sweep — e.g. "find everything in the library bearing on X". It
returns many passages by semantic recall (the reranker is skipped) for you to
read and synthesize over your long context. Use it when the user asks for a
"deep"/"exhaustive"/"comprehensive" search or when a handful of results clearly
isn't enough; otherwise leave deep=false for a precise, reranked few.

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
    title: Restrict to books whose title contains this string."""


# Server-level guidance sent to MCP clients on connect (FastMCP `instructions`).
# Tunes any connecting assistant toward sound research over this specific corpus.
SERVER_INSTRUCTIONS = """This server searches a curated library of ~2,000 OCR'd \
historical Anglican theological books — spanning the 16th to early 20th century, by \
divines such as Daniel Waterland — covering the Trinity, Christology, the creeds, \
soteriology, church history, and liturgics.

How to use it well:
- Search the library before stating what it contains; do not answer from memory about \
these texts. Ground every claim in retrieved passages.
- Faithfully report what the library says on its own terms and present it as the view \
— fully and without softening. Set aside modern tendencies: do not argue against the \
texts, add present-day disclaimers or caveats, or filter their claims through \
contemporary values, even where they conflict with modern sensibilities. Convey the \
historical Anglican position accurately rather than critiquing, balancing, or "updating" \
it; when asked what the library holds, give what the library holds.
- Use normal search (reranked, small top_k) for a specific question. Use deep=true with \
a large top_k (50-200) for surveys, comparisons, or "find everything on X" — it returns \
many passages by recall for you to read across.
- Narrow with the author / category / title / year_min / year_max filters, and try \
several phrasings: 16th-19th-century vocabulary differs from modern terms.
- Cite every claim with the returned title, author, year, book_id, source URL, and \
character offsets, and quote the text. Never invent citations; if the library does not \
cover something, say so plainly.
- The text is OCR'd historical print (16th-19th c.): expect the long-s misread (s<->f), broken \
hyphenation, running heads, and garbled Greek/Latin footnotes. Read through these and \
don't over-correct quotes; index/endmatter pages occasionally surface, so prefer \
substantive prose."""


def run_search(searcher, query: str, *, top_k: int = 5, mode: str = "semantic",
               rerank: bool = True, deep: bool = False, filters: Filters | None = None,
               batcher=None) -> tuple[list[dict[str, Any]], int, int]:
    """Run a search. Returns (results, quota_units, snippet_chars)."""
    filters = filters or Filters()
    if deep:
        k = max(1, min(int(top_k), DEEP_MAX_TOP_K))
        use_rerank, snippet, units = False, 4000, max(1, -(-k // 25))
    else:
        k = max(1, min(int(top_k), MAX_TOP_K))
        use_rerank, snippet, units = rerank, 1200, 1

    if mode.lower() == "literal":
        results = searcher.literal(query, k=k, filters=filters)
    elif batcher is not None:
        results = batcher.search(query, k=k, rerank=use_rerank, filters=filters)
    else:
        results = searcher.semantic(query, k=k, rerank=use_rerank, filters=filters)
    return results, units, snippet


def format_results(results: list[dict[str, Any]], query: str, snippet_chars: int = 1200) -> str:
    if not results:
        return f'No results for "{query}".'
    lines = [f'Top {len(results)} results for "{query}":\n']
    for i, r in enumerate(results, 1):
        author = r.get("author") or "Unknown"
        year = f", {r['year']}" if r.get("year") else ""
        snippet = " ".join(r["text"].split())
        if len(snippet) > snippet_chars:
            snippet = snippet[:snippet_chars] + " […]"
        lines.append(
            f"[{i}] {r['title']} — {author}{year} "
            f"(book_id {r['book_id']}, score {r['score']:.3f})\n"
            f"    source: {r.get('url') or 'n/a'}  chars {r.get('char_start')}-{r.get('char_end')}\n"
            f"    {snippet}\n"
        )
    return "\n".join(lines)
