"""Turn a research-paper PDF into clean, reading-order plain text.

Naive PDF text extraction is the single biggest source of robotic, garbled
audio: two-column papers get read across the gutter, words break across line
wraps as ``obser- vations``, and the bibliography gets read aloud as a wall of
author initials and years. This module fixes those three problems specifically,
because they are the ones you actually hear.

The output is a list of paragraphs in the order a human would read them, with
figure/table captions, page furniture, and everything after ``References``
dropped.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass
class Block:
    """A text block with its bounding box, as PyMuPDF reports it."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str


# A block is page furniture (running header / footer / page number) if it sits
# in the top or bottom margin and is short. Fractions of page height.
_TOP_MARGIN = 0.06
_BOTTOM_MARGIN = 0.94

# Caption blocks read terribly ("Figure 3 dot ..." mid-thought), so we drop a
# block whose first words announce a figure or table.
_CAPTION_START = re.compile(r"^\s*(fig(?:ure)?\.?|table|tab\.|algorithm)\s*\.?\s*\d", re.IGNORECASE)

# Section headers in IEEE style: "II. RELATED WORK", "A. Threat Model". We keep
# the title (it's a useful spoken signpost) but drop the enumerator, which would
# otherwise be read as a roman numeral or a stray letter.
_SECTION_HEADER = re.compile(r"^\s*(?:[IVXLC]+|[A-Z]|\d+)\.\s+(?P<title>[A-Z].*?)\s*$")

# Everything from the bibliography onward is unlistenable; stop here.
_REFERENCES = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE)


def extract_paragraphs(pdf_path: str) -> list[str]:
    """Read `pdf_path` and return its body as reading-order paragraphs."""
    doc = fitz.open(pdf_path)
    paragraphs: list[str] = []
    for page in doc:
        for block in _ordered_blocks(page):
            para = _clean_block(block.text)
            if para is None:
                continue
            if _REFERENCES.match(para):
                return paragraphs  # bibliography and appendices: stop reading
            paragraphs.append(para)
    return paragraphs


def _ordered_blocks(page: fitz.Page) -> list[Block]:
    """Return this page's text blocks in human reading order.

    Research papers are two-column. PyMuPDF hands back blocks in a layout order
    that zig-zags across the gutter, so we re-sort: a block belongs to the left
    column if its horizontal centre is left of the page midline, otherwise the
    right column. Full-width blocks (the title, a spanning figure) cross the
    midline; we bin them with the left column so the title still sorts to the
    top by its y-coordinate. Then it's left column top-to-bottom, then right.
    """
    mid = page.rect.width / 2
    top = page.rect.height * _TOP_MARGIN
    bottom = page.rect.height * _BOTTOM_MARGIN

    ordered: list[Block] = []
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        if not text.strip():
            continue
        # Drop running headers/footers and page numbers in the margins.
        if (y1 < top or y0 > bottom) and len(text.strip()) < 60:
            continue
        ordered.append(Block(x0, y0, x1, y1, text))

    crosses = lambda b: b.x0 < mid < b.x1  # noqa: E731 — spanning block
    column = lambda b: 0 if (crosses(b) or (b.x0 + b.x1) / 2 < mid) else 1  # noqa: E731
    ordered.sort(key=lambda b: (column(b), round(b.y0)))
    return ordered


def _clean_block(text: str) -> str | None:
    """Normalise one block into a single spoken paragraph, or drop it.

    Returns ``None`` for blocks that should not be read aloud (captions,
    leftover single-symbol blocks). Otherwise returns text with ligatures
    folded, line-wrap hyphenation rejoined, and inline citation brackets removed.
    """
    text = unicodedata.normalize("NFKC", text)

    if _CAPTION_START.match(text):
        return None

    # Rejoin words split across a line wrap: "obser-\nvations" -> "observations".
    text = re.sub(r"-\n(?=\w)", "", text)
    # Any remaining newline inside a block is a soft wrap, not a paragraph break.
    text = text.replace("\n", " ")
    # Inline numeric citations, including ranges like "[3]-[7]": drop the whole
    # run so we don't leave a dangling dash where the citations used to be.
    text = re.sub(r"\[[\d,\s–—-]+\](?:\s*[–—-]\s*\[[\d,\s–—-]+\])*", "", text)
    # Strip a list bullet at the start of a block ("• A closed-set …").
    text = re.sub(r"^\s*[•▪◦·‣∙-]\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # A section header: keep the title, drop its enumerator so we don't read
    # "one dot introduction". A trailing period gives the synth a clean pause.
    header = _SECTION_HEADER.match(text)
    if header:
        title = header.group("title")
        return title if title.endswith(".") else title + "."

    # Drop blocks with almost no letters (stray equation fragments, lone glyphs).
    if sum(c.isalpha() for c in text) < 3:
        return None

    return text
