"""Ingest the Schaff Church Fathers set (ANF + NPNF 1 & 2) from CCEL into the
library, one book row per work.

Source: CCEL's ThML (proofread digital text — far cleaner than archive.org OCR),
public domain. 38 volumes: Ante-Nicene Fathers (anf01-10), Nicene & Post-Nicene
Fathers series 1 (npnf101-114) and series 2 (npnf201-214).

The two series are marked up differently, so we detect per volume:
  * ANF style: each work has a <ThML.head> with a workID/DC.Title. Works run
    sequentially, so the text between consecutive work-heads is that work's body;
    the author is the <div1> heading the text actually lives under (the per-work
    authorID is sometimes wrong, e.g. Barnabas tagged "ignatius").
  * NPNF style: no per-work heads — each substantive <div1> is a work (e.g. "The
    Confessions"); the author is volume-level (see NPNF_AUTHOR).

Each work becomes a books row: title "<work> (<abbr> <vol>)", author, category
"Church Fathers", source_url (CCEL), content (clean text), year (edition).
Idempotent via a new books.ccel_id column.

    # review the split first (no DB writes, no downloads kept beyond the cache):
    uv run python scripts/ingest_ccel_fathers.py --dry-run
    # then ingest into the DB:
    uv run python scripts/ingest_ccel_fathers.py --db library.db

Needs lxml (`uv pip install lxml`). After ingest, embed the new chunks
(incremental) and ship db+index to the serving box — see the deploy docs.
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import time
import urllib.request
from collections import Counter

from lxml import etree

CCEL_URL = "https://ccel.org/ccel/schaff/{vol}.xml"
_UA = {"User-Agent": "anglican-library/0.1 (Church Fathers corpus ingest)"}

# (volume id, series label, abbr, volume number, edition year)
_ANF = [(f"anf{i:02d}", "Ante-Nicene Fathers", "ANF", i, 1885) for i in range(1, 11)]
_NPNF1 = [(f"npnf1{i:02d}", "Nicene and Post-Nicene Fathers, Series 1", "NPNF1", i, 1887)
          for i in range(1, 15)]
_NPNF2 = [(f"npnf2{i:02d}", "Nicene and Post-Nicene Fathers, Series 2", "NPNF2", i, 1890)
          for i in range(1, 15)]
VOLUMES = _ANF + _NPNF1 + _NPNF2

# Volume-level author for the NPNF volumes (which lack per-work author markers).
# NPNF2 mixed volumes get the primary author / "Various" — review with --dry-run.
NPNF_AUTHOR = {
    **{f"npnf1{i:02d}": "Augustine of Hippo" for i in range(1, 9)},
    **{f"npnf1{i:02d}": "John Chrysostom" for i in range(9, 15)},
    "npnf201": "Eusebius of Caesarea", "npnf202": "Socrates and Sozomen",
    "npnf203": "Various (Theodoret, Jerome, Rufinus)", "npnf204": "Athanasius",
    "npnf205": "Gregory of Nyssa", "npnf206": "Jerome",
    "npnf207": "Cyril of Jerusalem and Gregory Nazianzen", "npnf208": "Basil of Caesarea",
    "npnf209": "Various (Hilary of Poitiers, John of Damascus)", "npnf210": "Ambrose of Milan",
    "npnf211": "Various (Sulpitius Severus, Vincent, Cassian)",
    "npnf212": "Leo the Great and Gregory the Great", "npnf213": "Various (Gregory the Great, et al.)",
    "npnf214": "The Seven Ecumenical Councils",
}

_FRONT_MATTER = re.compile(
    r"^\s*(title page|contents?|table of contents|preface|prolegomena|chief events|"
    r"index|indexes|indices|index of|bibliography|errata|publisher|translator|"
    r"editor|introductory notice to|advertisement|note by the)",
    re.IGNORECASE,
)
_INDEX_PAGE = re.compile(r"index of (subjects|texts|scripture|greek|words)|:\s*index", re.IGNORECASE)
_MINOR = {"of", "the", "and", "to", "in", "a", "an", "on", "for", "with", "de"}


def _local(tag) -> str:
    return tag if isinstance(tag, str) else ""


def titlecase(s: str) -> str:
    s = re.sub(r"[_\s]+", " ", (s or "")).strip()
    words = s.split()
    out = []
    for i, w in enumerate(words):
        lw = w.lower()
        out.append(lw if (lw in _MINOR and 0 < i < len(words) - 1) else lw.capitalize())
    return " ".join(out)


def clean_title(s: str) -> str:
    return re.sub(r"[_\s]+", " ", (s or "")).strip()


def _ptext(el) -> str:
    """Joined text of a <p>, with footnote (<note>) *content* dropped but the
    paragraph text that resumes after a note (its tail) kept. Whitespace normalised."""
    if any(_local(a.tag) == "note" for a in el.iterancestors()):
        return ""  # this <p> is itself inside a footnote — skip entirely

    parts: list[str] = []

    def walk(node) -> None:
        if _local(node.tag) != "note":  # drop a note's text + descendants...
            if node.text:
                parts.append(node.text)
            for child in node:
                walk(child)
        if node.tail:  # ...but keep the text that follows it in the parent's flow
            parts.append(node.tail)

    if el.text:
        parts.append(el.text)
    for child in el:
        walk(child)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _div1_title_of(el):
    for a in el.iterancestors():
        if _local(a.tag) == "div1":
            return a.get("title") or a.get("shorttitle")
    return None


def parse_anf(body):
    """ANF style: works delimited by <ThML.head> work markers, in document order."""
    works, cur = [], None
    for el in body.iter():
        tag = _local(el.tag)
        if tag == "ThML.head":
            wid = el.find("electronicEdInfo/workID")
            aid = el.find("electronicEdInfo/authorID")
            ttl = el.find("electronicEdInfo/DC/DC.Title")
            if wid is None or not (wid.text or "").strip():
                continue
            if (aid is not None and (aid.text or "").strip() == "schaff"):
                continue  # the volume-level head, not a work
            cur = {"title": clean_title(ttl.text if ttl is not None else (el.get("title") or "")),
                   "authorid": (aid.text or "").strip() if aid is not None else "",
                   "parts": [], "div1": Counter()}
            works.append(cur)
        elif tag == "p" and cur is not None:
            txt = _ptext(el)
            if txt:
                cur["parts"].append(txt)
                d1 = _div1_title_of(el)
                if d1:
                    cur["div1"][d1] += 1
    out = []
    for w in works:
        text = "\n\n".join(w["parts"])
        if len(text) < 200:
            continue
        author = titlecase(w["div1"].most_common(1)[0][0]) if w["div1"] else titlecase(w["authorid"])
        out.append((author, w["title"], text))
    return out


def parse_npnf(body, vol):
    """NPNF style: each substantive <div1> is a work; author is volume-level."""
    author = NPNF_AUTHOR.get(vol, "")
    out = []
    for d1 in body.iter("div1"):
        title = clean_title(d1.get("title") or "")
        if not title or _FRONT_MATTER.match(title) or _INDEX_PAGE.search(title):
            continue
        text = "\n\n".join(t for t in (_ptext(p) for p in d1.iter("p")) if t)
        if len(text) < 200:
            continue
        out.append((author, title, text))
    return out


def works_for_volume(path, vol):
    tree = etree.parse(path, etree.XMLParser(recover=True, huge_tree=True))
    body = tree.getroot().find("ThML.body")
    if body is None:
        return []
    # ANF if it has real per-work heads; else NPNF (div1 = work).
    has_work_heads = any(
        (h.find("electronicEdInfo/workID") is not None
         and (h.find("electronicEdInfo/authorID") is None
              or (h.find("electronicEdInfo/authorID").text or "").strip() != "schaff"))
        for h in body.iter("ThML.head"))
    return parse_anf(body) if has_work_heads else parse_npnf(body, vol)


def fetch_volume(vol: str, cache: str) -> str:
    path = os.path.join(cache, f"{vol}.xml")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(cache, exist_ok=True)
    url = CCEL_URL.format(vol=vol)
    print(f"  downloading {url} ...", flush=True)
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    with open(path, "wb") as f:
        f.write(data)
    time.sleep(1)  # be polite to CCEL
    return path


# --- DB ---------------------------------------------------------------------
def ensure_ccel_column(conn: sqlite3.Connection) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(books)")}
    if "ccel_id" not in cols:
        conn.execute("ALTER TABLE books ADD COLUMN ccel_id TEXT")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_books_ccel ON books(ccel_id) "
                 "WHERE ccel_id IS NOT NULL")
    conn.commit()


def insert_work(conn, ccel_id, title, author, category, source_url, year, content) -> bool:
    if conn.execute("SELECT 1 FROM books WHERE ccel_id=?", (ccel_id,)).fetchone():
        return False
    conn.execute(
        "INSERT INTO books (ccel_id, title, author, category, source_url, content, "
        "year, status, approved) VALUES (?,?,?,?,?,?,?, 'ok', 1)",
        (ccel_id, title, author, category, source_url, content, year))
    conn.commit()
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="library.db", help="library DB to ingest into")
    ap.add_argument("--cache", default="ccel_cache", help="dir to cache downloaded ThML")
    ap.add_argument("--dry-run", action="store_true", help="parse + print manifest, no DB writes")
    ap.add_argument("--only", default="", help="comma-separated volume ids (e.g. anf01,npnf101)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    vols = [v for v in VOLUMES if not args.only or v[0] in args.only.split(",")]
    conn = None
    if not args.dry_run:
        if not os.path.exists(args.db):
            raise SystemExit(f"DB not found: {args.db}")
        conn = sqlite3.connect(args.db)
        ensure_ccel_column(conn)

    grand_works = grand_chars = grand_new = 0
    for vol, series, abbr, num, year in vols:
        print(f"== {abbr} vol {num} ({vol}) ==", flush=True)
        path = fetch_volume(vol, args.cache)
        works = works_for_volume(path, vol)
        vch = sum(len(t) for _, _, t in works)
        grand_works += len(works); grand_chars += vch
        print(f"  {len(works)} works, {vch:,} chars", flush=True)
        for n, (author, title, text) in enumerate(works, 1):
            full_title = f"{title} ({abbr} {num})"
            ccel_id = f"{vol}#{n:03d}"
            url = CCEL_URL.format(vol=vol).replace(".xml", "")
            if args.dry_run:
                print(f"    [{n:>2}] {author:<28} | {title[:54]:<54} | {len(text):>7}")
            elif insert_work(conn, ccel_id, full_title, author, "Church Fathers", url, year, text):
                grand_new += 1
    print(f"\nTOTAL: {grand_works} works, {grand_chars:,} chars"
          + ("" if args.dry_run else f", {grand_new} new books inserted"))
    if conn:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
