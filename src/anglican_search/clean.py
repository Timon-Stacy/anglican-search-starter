"""Adaptive OCR cleanup for 19th-century scanned theological texts.

Design notes (grounded in the three sample volumes, not assumptions):

* Long-s ("ſ" misread as "f") is *volume-specific*: Scott (01914) had ~1,800
  occurrences, Waterland and Vogan almost none. So correction is whole-word and
  dictionary-style (a curated map), never a blanket f->s substitution, which
  would wreck the clean volumes (e.g. turn "of" into "os").

* Boilerplate runs well past the first 100 lines and includes pure OCR garbage
  from scanned title/ornamental pages (e.g. "SYTIMIVIG at EO"). A fixed line
  cutoff fails, so we detect where sustained real prose begins instead.

* Running heads / page numbers / signature marks ("4 INTRODUCTION.",
  "AND SUBJECTS. xiii", "D 3 S:.") are interleaved every ~30-40 lines at page
  breaks and are dropped as noise.

* Greek/Latin footnote quotes were OCR'd into garbled ASCII (zero real Greek
  codepoints in the samples), so they can't be detected as Greek; they survive
  as low-signal noise the embedder largely ignores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .longs import correct_text as fix_long_s  # dictionary-driven long-s fix

# ---------------------------------------------------------------------------
# Noise-line detection (running heads, page numbers, signature marks)
# ---------------------------------------------------------------------------
_PAGE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$")
_ROMAN_RE = re.compile(r"^\s*[ivxlcdm]{1,8}\.?\s*$", re.IGNORECASE)


def is_noise_line(line: str) -> bool:
    """True for lines that are page furniture rather than prose."""
    s = line.strip()
    if not s:
        return False  # blanks are handled separately (paragraph breaks)
    if _PAGE_NUM_RE.match(s) or _ROMAN_RE.match(s):
        return True
    # very short symbol/garble fragments ("D 3 S:.", "~", "@")
    letters = [c for c in s if c.isalpha()]
    if len(s) <= 3:
        return True
    if not letters:
        return True  # punctuation/number-only line
    # short, mostly-uppercase lines are running heads ("AND SUBJECTS. xiii",
    # "4 INTRODUCTION.", "NAMES OF LECTURERS,")
    words = s.split()
    if len(words) <= 6 and len(letters) <= 32:
        upper = sum(c.isupper() for c in letters)
        if upper / len(letters) >= 0.7:
            return True
    return False


# ---------------------------------------------------------------------------
# Content-start detection (skip front matter / title-page OCR garbage)
# ---------------------------------------------------------------------------
def _is_prose(line: str) -> bool:
    """Heuristic: does this line look like a real sentence fragment?"""
    s = line.strip()
    words = s.split()
    if len(words) < 5:
        return False
    letters = sum(c.isalpha() for c in s)
    if letters / max(len(s), 1) < 0.6:  # too many symbols/digits -> garble
        return False
    lower = sum(c.islower() for c in s)
    upper = sum(c.isupper() for c in s)
    if lower <= upper:  # all-caps -> heading/running head, not prose
        return False
    wordlike = sum(1 for w in words if re.fullmatch(r"[A-Za-z]{2,}", w.strip(",.;:'\"()[]")))
    return wordlike >= 4


def find_content_start(lines: list[str], search_limit: int = 600) -> int:
    """Index of the first line of the first sustained run of real prose.

    Scans a sliding window; the first window with a clear majority of prose
    lines marks where real content begins. Capped so we never eat deep into a
    volume; returns 0 if no clear start is found.
    """
    window = 8
    need = 5
    limit = min(len(lines), search_limit)
    for i in range(limit):
        win = lines[i : i + window]
        if sum(_is_prose(ln) for ln in win) >= need and _is_prose(lines[i]):
            return i
    return 0


# ---------------------------------------------------------------------------
# Dehyphenation + paragraph assembly
# ---------------------------------------------------------------------------
@dataclass
class Paragraph:
    text: str          # cleaned text
    char_start: int    # offset into the original body/content (inclusive)
    char_end: int      # offset into the original body/content (exclusive)


_WS_RE = re.compile(r"[ \t]+")


@dataclass
class CleanStats:
    total_lines: int
    blank_lines: int
    noise_lines_removed: int
    content_start_offset: int  # lines skipped as leading boilerplate
    long_s_corrections: int


def build_paragraphs(
    body: str,
    apply_long_s: bool = True,
) -> tuple[list[Paragraph], CleanStats]:
    """Turn a raw OCR body into cleaned, char-offset-tracked paragraphs.

    Pipeline: drop leading boilerplate -> drop noise lines -> group blank-line
    separated runs into paragraphs -> dehyphenate line wraps -> normalise
    whitespace -> long-s correction. Each paragraph records the [char_start,
    char_end) span of the ORIGINAL `body` it was derived from, so a chunk can be
    located back in books.content regardless of cleaning.
    """
    raw_lines = body.split("\n")
    total = len(raw_lines)
    blank = sum(1 for ln in raw_lines if not ln.strip())

    # Char offset of each line within `body` (len + 1 for the split-out "\n").
    line_start: list[int] = []
    pos = 0
    for ln in raw_lines:
        line_start.append(pos)
        pos += len(ln) + 1
    line_end = [line_start[i] + len(raw_lines[i]) for i in range(total)]

    start = find_content_start(raw_lines)

    # Keep (char_start, char_end, text); empty text marks a paragraph break.
    kept: list[tuple[int, int, str]] = []
    noise_removed = 0
    for i in range(start, total):
        line = raw_lines[i]
        if line.strip() and is_noise_line(line):
            noise_removed += 1
            kept.append((line_start[i], line_end[i], ""))
            continue
        kept.append((line_start[i], line_end[i], line))

    # Group into paragraphs on blank lines.
    paragraphs: list[Paragraph] = []
    cur: list[tuple[int, int, str]] = []

    def flush() -> None:
        if not cur:
            return
        text = _join_lines([t for _, _, t in cur])
        if text.strip():
            paragraphs.append(
                Paragraph(text=text, char_start=cur[0][0], char_end=cur[-1][1])
            )
        cur.clear()

    for cs, ce, text in kept:
        if text.strip():
            cur.append((cs, ce, text))
        else:
            flush()
    flush()

    long_s_total = 0
    if apply_long_s:
        for p in paragraphs:
            p.text, n = fix_long_s(p.text)
            long_s_total += n

    stats = CleanStats(
        total_lines=total,
        blank_lines=blank,
        noise_lines_removed=noise_removed,
        content_start_offset=start,
        long_s_corrections=long_s_total,
    )
    return paragraphs, stats


def _join_lines(lines: list[str]) -> str:
    """Join wrapped lines into a paragraph, repairing hyphenated line breaks."""
    out: list[str] = []
    for i, line in enumerate(lines):
        token = _WS_RE.sub(" ", line.strip())
        if not token:
            continue
        # Hyphen at end of a wrapped line followed by a lowercase continuation
        # => join without the hyphen ("Col-" + "lege" -> "College").
        if out and out[-1].endswith("-"):
            prev = out[-1]
            nxt = token
            if re.search(r"[A-Za-z]-$", prev) and nxt[:1].islower():
                out[-1] = prev[:-1] + nxt
                continue
        out.append(token)
    return " ".join(out)
