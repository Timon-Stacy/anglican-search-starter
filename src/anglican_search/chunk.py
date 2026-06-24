"""Paragraph-aware, token-budgeted chunking.

Paragraphs are treated as atomic and packed into ~target_tokens chunks with a
small trailing overlap. Keeping paragraphs whole is also what satisfies the
"don't split in the middle of a footnote" requirement: inline footnote blocks
travel with their paragraph rather than being cut across a chunk boundary.

A paragraph larger than the hard ceiling (rare) is split on sentence
boundaries as a fallback. Every chunk carries the original-file line range of
the source text so a result can be located in the .md again.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from .clean import Paragraph
from .config import ChunkConfig

TokenCounter = Callable[[str], int]

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


@dataclass
class Chunk:
    text: str
    chunk_index: int
    char_start: int
    char_end: int
    n_tokens: int


def chunk_paragraphs(
    paragraphs: list[Paragraph],
    count_tokens: TokenCounter,
    cfg: ChunkConfig,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf: list[Paragraph] = []
    buf_tokens = 0

    def emit() -> None:
        nonlocal buf, buf_tokens
        if not buf:
            return
        text = "\n\n".join(p.text for p in buf)
        chunks.append(
            Chunk(
                text=text,
                chunk_index=len(chunks),
                char_start=buf[0].char_start,
                char_end=buf[-1].char_end,
                n_tokens=count_tokens(text),
            )
        )
        # Seed the next buffer with trailing paragraphs for overlap.
        overlap: list[Paragraph] = []
        otok = 0
        for p in reversed(buf):
            t = count_tokens(p.text)
            if otok + t > cfg.overlap_tokens:
                break
            overlap.insert(0, p)
            otok += t
        buf = overlap
        buf_tokens = otok

    for para in paragraphs:
        ptok = count_tokens(para.text)

        if ptok > cfg.max_tokens:
            emit()  # close out whatever is buffered first
            buf, buf_tokens = [], 0
            for piece in _split_oversized(para, count_tokens, cfg):
                chunks.append(
                    Chunk(
                        text=piece.text,
                        chunk_index=len(chunks),
                        char_start=piece.char_start,
                        char_end=piece.char_end,
                        n_tokens=count_tokens(piece.text),
                    )
                )
            continue

        if buf_tokens + ptok > cfg.target_tokens and buf_tokens >= cfg.min_tokens:
            emit()
        buf.append(para)
        buf_tokens += ptok

    emit()
    # Renumber after the fact so indices are contiguous and overlap-stable.
    for i, c in enumerate(chunks):
        c.chunk_index = i
    return chunks


def _split_oversized(
    para: Paragraph, count_tokens: TokenCounter, cfg: ChunkConfig
) -> list[Paragraph]:
    """Split a too-large paragraph on sentence boundaries."""
    sentences = _SENT_RE.split(para.text)
    out: list[Paragraph] = []
    cur: list[str] = []
    cur_tok = 0
    for sent in sentences:
        st = count_tokens(sent)
        if cur and cur_tok + st > cfg.target_tokens:
            out.append(Paragraph(" ".join(cur), para.char_start, para.char_end))
            cur, cur_tok = [], 0
        cur.append(sent)
        cur_tok += st
    if cur:
        out.append(Paragraph(" ".join(cur), para.char_start, para.char_end))
    return out
