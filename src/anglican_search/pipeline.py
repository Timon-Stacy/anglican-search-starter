"""End-to-end body -> chunks orchestration, shared by re-chunking and inspection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunk import Chunk, TokenCounter, chunk_paragraphs
from .clean import CleanStats, build_paragraphs
from .config import ChunkConfig, DEFAULT_CHUNK_CONFIG, estimate_tokens
from .parse import BookMeta, parse_file


def process_body(
    body: str,
    count_tokens: TokenCounter = estimate_tokens,
    cfg: ChunkConfig = DEFAULT_CHUNK_CONFIG,
    apply_long_s: bool = True,
) -> tuple[list[Chunk], CleanStats]:
    """Clean and chunk a raw OCR body string (e.g. books.content)."""
    paragraphs, stats = build_paragraphs(body, apply_long_s=apply_long_s)
    chunks = chunk_paragraphs(paragraphs, count_tokens, cfg)
    return chunks, stats


@dataclass
class ProcessedFile:
    meta: BookMeta
    chunks: list[Chunk]
    stats: CleanStats


def process_file(
    path: Path,
    count_tokens: TokenCounter = estimate_tokens,
    cfg: ChunkConfig = DEFAULT_CHUNK_CONFIG,
    apply_long_s: bool = True,
) -> ProcessedFile:
    """Parse a .md file and clean+chunk its body (used by the inspector)."""
    parsed = parse_file(path)
    chunks, stats = process_body(parsed.body, count_tokens, cfg, apply_long_s)
    return ProcessedFile(meta=parsed.meta, chunks=chunks, stats=stats)
