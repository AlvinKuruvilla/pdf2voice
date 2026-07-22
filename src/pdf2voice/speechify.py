"""Rewrite academic prose into something that *sounds* right when spoken.

Extraction gives clean text; this makes it clean *speech*. The difference is
all the written shorthand a reader's eye skips but a text-to-speech model reads
literally: "e.g." becomes "ee gee", a bare URL becomes a spelled-out mess, and
"et al." lands as "et al period". We expand the common offenders and delete the
things that have no spoken form at all.
"""

from __future__ import annotations

import re

# Written abbreviations and their spoken form. Order matters only in that the
# multi-word keys ("et al.") are matched before their prefixes could interfere;
# we apply them longest-first to be safe.
_ABBREVIATIONS: dict[str, str] = {
    "e.g.": "for example",
    "i.e.": "that is",
    "et al.": "and colleagues",
    "cf.": "compare",
    "etc.": "and so on",
    "vs.": "versus",
    "w.r.t.": "with respect to",
    "Fig.": "Figure",
    "Eq.": "Equation",
    "Sec.": "Section",
    "Tab.": "Table",
    "approx.": "approximately",
}

# Symbols and Greek letters that appear in the notation-heavy results sections.
# Left unmapped they are read as silence or garbled, so we give each a spoken
# word. This is not a math renderer — just the handful that actually show up.
_SYMBOLS: dict[str, str] = {
    "@": " at ", "×": " times ", "±": " plus or minus ",
    "−": " minus ", "≈": " approximately ", "≤": " at most ", "≥": " at least ",
    "≠": " not equal to ", "=": " equals ", "<": " less than ", ">": " greater than ",
    "%": " percent ", "→": " to ", "∈": " in ", "∆": " delta ", "Δ": " delta ",
    "·": " ", "{": " ", "}": " ",
    "α": " alpha ", "β": " beta ", "γ": " gamma ", "λ": " lambda ",
    "σ": " sigma ", "μ": " mu ", "θ": " theta ",
}

# Things with no useful spoken form: URLs, DOIs, emails. Delete them outright.
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_DOI = re.compile(r"\bdoi:\s*\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


def speechify(paragraph: str) -> str:
    """Return `paragraph` rewritten for natural-sounding synthesis."""
    text = _URL.sub("", paragraph)
    text = _DOI.sub("", text)
    text = _EMAIL.sub("", text)

    for written, spoken in sorted(_ABBREVIATIONS.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(re.escape(written), spoken, text, flags=re.IGNORECASE)

    for symbol, spoken in _SYMBOLS.items():
        text = text.replace(symbol, spoken)

    # Numeric ranges and ratios read wrong as punctuation: "0.98–0.99" and
    # "1:N" should be spoken "to", not "dash" / "colon". Both rules are anchored
    # to digits so compounds ("text-based", "COVID-19") are untouched.
    text = re.sub(r"(?<=\d)\s*[–—-]\s*(?=\d)", " to ", text)
    text = re.sub(r"\b([0-9N]+)\s*:\s*([0-9N]+)\b", r"\1 to \2", text)

    # Rejoin the thousands separator when extraction split "1,098" into "1, 098",
    # which would otherwise be read as "one, ninety-eight".
    text = re.sub(r"(?<=\d),\s+(?=\d{3}\b)", ",", text)
    # A dangling dash left where a citation was removed ("content –.") is noise.
    text = re.sub(r"\s+[–—-]\s*(?=[.,;:])", "", text)

    # Collapse whitespace left behind by deletions and tidy spacing before
    # punctuation so the model doesn't insert an odd pause.
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
