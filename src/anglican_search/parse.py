"""Parse the metadata header and split off the OCR body.

Header shape (consistent across the corpus):

    # <Title>

    Author: <Author>
    Category: <Category>
    Book ID: <numeric ID>
    Status: ok
    Input URL: <archive.org details URL>
    Source URL: <archive.org _djvu.txt URL>

    ---
    <raw OCR body...>
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BookMeta:
    book_id: int | None
    title: str
    author: str
    category: str
    status: str
    input_url: str
    source_url: str


@dataclass
class ParsedFile:
    meta: BookMeta
    body: str
    # 1-based line number in the original file where the body begins. Chunk line
    # ranges are reported in original-file coordinates so they're clickable.
    body_start_line: int


_KEY_RE = re.compile(r"^([A-Za-z][A-Za-z ]+?):\s*(.*)$")
_KEY_MAP = {
    "author": "author",
    "category": "category",
    "book id": "book_id",
    "status": "status",
    "input url": "input_url",
    "source url": "source_url",
}


def read_text_smart(path: Path) -> str:
    """Read a file as UTF-8, falling back to cp1252.

    The archive.org _djvu.txt sources are typically Windows-1252, not UTF-8:
    their curly apostrophes/quotes (0x91-0x94) and em dashes (0x97) are invalid
    UTF-8 and would otherwise be destroyed into U+FFFD replacement characters.
    UTF-8-strict succeeds on genuine UTF-8 files, so the fallback only fires for
    the legacy-encoded ones, where cp1252 is the right interpretation.
    """
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def parse_file(path: Path) -> ParsedFile:
    """Parse one .md file into metadata + raw body text."""
    raw = read_text_smart(path)
    lines = raw.splitlines()

    fields: dict[str, str] = {}
    title = ""
    body_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not title and stripped.startswith("# "):
            title = stripped[2:].strip()
            continue
        if stripped == "---":
            body_start = i + 1
            break
        m = _KEY_RE.match(stripped)
        if m:
            key = m.group(1).strip().lower()
            if key in _KEY_MAP:
                fields[_KEY_MAP[key]] = m.group(2).strip()

    # book_id: prefer the header value, fall back to the filename prefix
    # (files are named like "01911_-_Waterland__...md").
    book_id = _to_int(fields.get("book_id"))
    if book_id is None:
        fm = re.match(r"0*(\d+)", path.stem)
        if fm:
            book_id = int(fm.group(1))

    meta = BookMeta(
        book_id=book_id,
        title=title or path.stem,
        author=fields.get("author", ""),
        category=fields.get("category", ""),
        status=fields.get("status", ""),
        input_url=fields.get("input_url", ""),
        source_url=fields.get("source_url", ""),
    )
    body = "\n".join(lines[body_start:])
    return ParsedFile(meta=meta, body=body, body_start_line=body_start + 1)


def _to_int(value: str | None) -> int | None:
    if not value:
        return None
    m = re.search(r"\d+", value)
    return int(m.group(0)) if m else None
