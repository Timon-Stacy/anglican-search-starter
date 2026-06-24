"""Dictionary-driven long-s ("ſ" OCR'd as "f") correction.

Algorithm, per whitespace token containing 'f':

1. If a curated override applies (archaic spellings the modern dictionary won't
   contain, e.g. "fhew" -> "shew"), use it.
2. If the token is already a valid English word, leave it untouched. This is
   what protects legitimate f-words ("after", "different", "office", "of",
   "if") from ever being altered.
3. Otherwise enumerate every subset of the token's 'f' positions, flip those to
   's', and keep the highest-frequency candidate that is a real word.

Step 3 enforces the typography for free: a long-s was never the *final* letter
and never adjacent to a real 'f', so candidates that violate those rules simply
aren't dictionary words and are rejected. "felf" -> "self" (flip first f only;
"sels"/"fels" aren't words), "Philofophy" -> "Philosophy", "Effence" ->
"Essence". A token like "after" is already valid, so it's left alone.

Ambiguous collisions where the f-form is itself a valid word (e.g. "fame" that
should be "same") are intentionally left unchanged — corrupting a genuine word
is worse than missing one long-s case.
"""

from __future__ import annotations

import re
from functools import lru_cache

from spellchecker import SpellChecker

# Archaic / domain spellings the modern frequency dictionary lacks, so the
# enumeration in step 3 can't recover them. Extend as needed.
_CURATED: dict[str, str] = {
    "fhew": "shew", "fhewn": "shewn", "fhewed": "shewed",
    "fheweth": "sheweth", "fhewing": "shewing", "fhews": "shews",
    "fo": "so", " alfo": "also",
    "thofe": "those", "thefe": "these", "whofe": "whose",
    "doft": "dost", "haft": "hast", "waft": "wast",
}

_spell = SpellChecker(distance=1)  # only the frequency dictionary is used
_FREQ = _spell.word_frequency

# The bundled dictionary is American English. Teach it common British and
# archaic spellings that pervade this corpus, so they're (a) protected from
# being "corrected" (step 2 leaves any known word alone) and (b) not flagged as
# noise by the diagnostics.
_PROTECTED_WORDS = """
defence defences offence offences pretence pretences licence licences
favour favours favoured favouring favourite favourites favourable favourably
honour honours honoured honourable colour colours behaviour saviour saviours
neighbour endeavour endeavours endeavoured labour labours laboured ardour
fervour vigour splendour rigour succour valour humour odour clamour candour
demeanour parlour harbour centre centres theatre sceptre fibre fibres metre
practise practised judgement judgements acknowledgement connexion reflexion
hath doth dost hast wast thou thee thy thine ye unto whilst amongst
shew shews shewn shewed shewing sheweth thyself
""".split()
_FREQ.load_words(_PROTECTED_WORDS)

_TOKEN_RE = re.compile(r"[A-Za-z]+")


def _match_case(template: str, source: str) -> str:
    if source.isupper():
        return template.upper()
    if source[:1].isupper():
        return template.capitalize()
    return template


@lru_cache(maxsize=300_000)
def _fix_lower(low: str) -> str:
    """Correct one lowercase token; returns it unchanged if no fix applies."""
    if low in _CURATED:
        return _CURATED[low]
    if low in _FREQ:  # already a real word -> never touch it
        return low

    positions = [i for i, c in enumerate(low) if c == "f"]
    if not positions or len(positions) > 5:
        return low

    best: str | None = None
    best_freq = -1
    chars = list(low)
    n = len(positions)
    for mask in range(1, 1 << n):  # every non-empty subset of f-positions
        for j in range(n):
            chars[positions[j]] = "s" if (mask >> j) & 1 else "f"
        cand = "".join(chars)
        if cand in _FREQ:
            f = _FREQ[cand]
            if f > best_freq:
                best_freq, best = f, cand
    # restore (not strictly needed since we rebuild each loop, but keep clean)
    return best if best is not None else low


def correct_token(token: str) -> str:
    if "f" not in token and "F" not in token:
        return token
    low = token.lower()
    fixed = _fix_lower(low)
    return token if fixed == low else _match_case(fixed, token)


def correct_text(text: str) -> tuple[str, int]:
    """Apply long-s correction across a string. Returns (text, num_corrections)."""
    count = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal count
        tok = m.group(0)
        out = correct_token(tok)
        if out != tok:
            count += 1
        return out

    return _TOKEN_RE.sub(repl, text), count


def is_real_word(token: str) -> bool:
    """Dictionary membership (lowercased). Used by diagnostics."""
    return token.lower() in _FREQ
