"""Run the ingestion pipeline on sample files and print results for tuning.

This is the "show me the cleanup on real text" tool. It reports per-file stats
(boilerplate skipped, noise lines removed, long-s corrections), shows a few
chunks, and surfaces the most common unrecognised f-words so the long-s map can
be grown against actual data.

    uv run python scripts/inspect_chunks.py            # all of sample_data
    uv run python scripts/inspect_chunks.py <file.md>  # one file
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The Windows console code page mangles curly quotes/dashes into "�" on display
# even when the underlying string is correct UTF-8; force UTF-8 output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from anglican_search.config import SAMPLE_DIR  # noqa: E402
from anglican_search.longs import is_real_word  # noqa: E402
from anglican_search.parse import parse_file  # noqa: E402
from anglican_search.pipeline import process_file  # noqa: E402

_FWORD_RE = re.compile(r"[A-Za-z]*f[A-Za-z]*")


def unresolved_f_words(text: str, top: int = 12) -> list[tuple[str, int]]:
    """f-containing tokens in the CORRECTED text that aren't real words.

    These are what's genuinely left: long-s the corrector couldn't resolve,
    OCR garbage, or proper nouns. Legitimate f-words are dictionary hits and
    excluded.
    """
    counter: Counter[str] = Counter()
    for w in _FWORD_RE.findall(text):
        if len(w) >= 4 and not is_real_word(w):
            counter[w] += 1
    return counter.most_common(top)


def show_file(path: Path) -> None:
    parsed = parse_file(path)
    result = process_file(path)
    m, s = result.meta, result.stats

    print("=" * 78)
    print(f"{m.title}")
    print(f"  book_id={m.book_id}  author={m.author!r}  category={m.category!r}")
    print(f"  source={m.source_url}")
    print("-" * 78)
    print(
        f"  lines={s.total_lines}  blank={s.blank_lines} "
        f"({s.blank_lines / max(s.total_lines,1):.0%})  "
        f"boilerplate_skipped={s.content_start_offset}  "
        f"noise_removed={s.noise_lines_removed}"
    )
    print(f"  long_s_corrections={s.long_s_corrections}  chunks={len(result.chunks)}")
    if result.chunks:
        toks = [c.n_tokens for c in result.chunks]
        print(f"  tokens/chunk: min={min(toks)} avg={sum(toks)//len(toks)} max={max(toks)}")

    corrected = "\n".join(c.text for c in result.chunks)
    fw = unresolved_f_words(corrected)
    if fw:
        print("  unresolved f-words (post-fix):",
              ", ".join(f"{w}({n})" for w, n in fw))

    print("-" * 78)
    for c in result.chunks[:2]:
        _show_chunk(c)
    if len(result.chunks) > 4:
        print("  ...")
        _show_chunk(result.chunks[len(result.chunks) // 2])


def _show_chunk(c) -> None:
    preview = c.text.replace("\n", " ")
    if len(preview) > 480:
        preview = preview[:480] + " […]"
    print(f"  [chunk {c.chunk_index}] chars {c.char_start}-{c.char_end}  ~{c.n_tokens} tok")
    print(f"    {preview}")
    print()


def main() -> int:
    args = sys.argv[1:]
    if args:
        paths = [Path(a) for a in args]
    else:
        paths = sorted(SAMPLE_DIR.glob("*.md"))
    if not paths:
        print("No .md files found.")
        return 1
    for p in paths:
        show_file(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
