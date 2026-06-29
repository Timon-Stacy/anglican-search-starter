"""Gate tests for the CCEL Schaff-Fathers parser (scripts/ingest_ccel_fathers.py).

Covers the tricky bits that would silently corrupt the corpus if they regressed:
ANF author taken from the div1 the text lives under (not the sometimes-wrong
per-work authorID), footnote stripping, NPNF div1-as-work with front-matter/index
filtering, and title casing. Skipped if lxml isn't installed.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

pytest.importorskip("lxml")
from lxml import etree  # noqa: E402

_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "ingest_ccel_fathers.py"
_spec = importlib.util.spec_from_file_location("ccel_ingest", _PATH)
ccel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ccel)


def _body(xml: str):
    return etree.fromstring(xml.encode()).find("ThML.body")


ANF = """<ThML><ThML.body>
<div1 title="CLEMENT OF ROME">
  <ThML.head><electronicEdInfo><workID>a</workID><authorID>clement_rome</authorID>
  <DC><DC.Title>First Epistle to the Corinthians</DC.Title></DC></electronicEdInfo></ThML.head>
  <p>{clem}</p>
</div1>
<div1 title="BARNABAS">
  <ThML.head><electronicEdInfo><workID>b</workID><authorID>ignatius</authorID>
  <DC><DC.Title>Epistle of Barnabas</DC.Title></DC></electronicEdInfo></ThML.head>
  <p>{barn} keep<note>DROP THIS FOOTNOTE</note> tail</p>
</div1>
</ThML.body></ThML>""".format(clem="alpha " * 60, barn="beta " * 60)

NPNF = """<ThML><ThML.body>
<div1 title="Title Page"><p>front</p></div1>
<div1 title="The Confessions"><p>{conf}</p></div1>
<div1 title="The Confessions of St. Augustin: Index of Subjects"><p>{idx}</p></div1>
</ThML.body></ThML>""".format(conf="gamma " * 60, idx="delta " * 60)


# --- ANF: author from div1, footnotes dropped ----------------------------
def test_anf_extracts_works_with_div1_author():
    works = ccel.parse_anf(_body(ANF))
    titles = {t: a for a, t, _ in works}
    assert "First Epistle to the Corinthians" in titles
    assert "Epistle of Barnabas" in titles


def test_anf_author_comes_from_div1_not_buggy_authorid():
    works = {t: a for a, t, _ in ccel.parse_anf(_body(ANF))}
    # the work-head's authorID for Barnabas is the wrong "ignatius"; the div1 wins
    assert works["Epistle of Barnabas"] == "Barnabas"
    assert works["First Epistle to the Corinthians"] == "Clement of Rome"


def test_anf_drops_footnotes():
    text = {t: x for _, t, x in ccel.parse_anf(_body(ANF))}["Epistle of Barnabas"]
    assert "DROP THIS FOOTNOTE" not in text
    assert "keep" in text and "tail" in text


# --- NPNF: div1 = work, front matter + index filtered --------------------
def test_npnf_uses_div1_works_and_volume_author():
    works = ccel.parse_npnf(_body(NPNF), "npnf101")
    titles = [t for _, t, _ in works]
    assert titles == ["The Confessions"]          # Title Page + Index dropped
    assert works[0][0] == "Augustine of Hippo"     # volume-level author


# --- helpers --------------------------------------------------------------
def test_titlecase_keeps_minor_words_lower():
    assert ccel.titlecase("CLEMENT OF ROME") == "Clement of Rome"
    assert ccel.titlecase("JUSTIN MARTYR") == "Justin Martyr"


def test_index_page_regex_matches_index_titles():
    assert ccel._INDEX_PAGE.search("The Confessions: Index of Subjects")
    assert not ccel._INDEX_PAGE.search("The Confessions")
