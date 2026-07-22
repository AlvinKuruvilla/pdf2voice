"""pdf2voice: turn research-paper PDFs into natural-sounding audio."""

from .extract import extract_paragraphs
from .speechify import speechify
from .tts import synthesize

__all__ = ["extract_paragraphs", "speechify", "synthesize"]
