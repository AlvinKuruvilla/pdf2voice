"""Command-line entry point: ``pdf2voice paper.pdf -o paper.mp3``."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .extract import extract_paragraphs
from .speechify import speechify
from .tts import synthesize

# A handful of Kokoro's more natural English voices, for --list-voices and as a
# hint in --help. Prefix a = American, b = British; f = female, m = male.
_VOICES = ["af_heart", "af_bella", "af_nicole", "am_michael", "am_puck", "bf_emma", "bm_george"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pdf2voice",
        description="Turn a research-paper PDF into natural-sounding audio.",
    )
    parser.add_argument("pdf", type=Path, help="path to the input PDF")
    parser.add_argument(
        "-o", "--output", type=Path,
        help="output audio file (.mp3, .m4a, or .wav). Defaults to the PDF name as .mp3.",
    )
    parser.add_argument(
        "-v", "--voice", default="af_heart",
        help=f"Kokoro voice (default: af_heart). Some options: {', '.join(_VOICES)}",
    )
    parser.add_argument(
        "-s", "--speed", type=float, default=1.0,
        help="speaking rate multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--text-only", action="store_true",
        help="print the cleaned, speech-ready text and exit without synthesising",
    )
    args = parser.parse_args(argv)

    if not args.pdf.is_file():
        parser.error(f"no such file: {args.pdf}")

    # Extraction and normalisation are cheap; do them first so --text-only is
    # instant and so a bad PDF fails before we pay to load the TTS model.
    paragraphs = [speechify(p) for p in extract_paragraphs(str(args.pdf))]
    paragraphs = [p for p in paragraphs if p]
    if not paragraphs:
        parser.error(f"no readable text found in {args.pdf}")

    if args.text_only:
        print("\n\n".join(paragraphs))
        return 0

    out_path = args.output or args.pdf.with_suffix(".mp3")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    words = sum(len(p.split()) for p in paragraphs)
    print(f"Extracted {len(paragraphs)} paragraphs (~{words} words). Synthesising with '{args.voice}'…")

    started = time.monotonic()
    written = synthesize(
        paragraphs, str(out_path), voice=args.voice, speed=args.speed,
        on_progress=_progress,
    )
    print(f"\nDone in {time.monotonic() - started:.0f}s → {written}")
    return 0


def _progress(done: int, total: int) -> None:
    """Render a single-line progress bar to stderr."""
    width = 30
    filled = int(width * done / total)
    bar = "█" * filled + "░" * (width - filled)
    sys.stderr.write(f"\r  [{bar}] {done}/{total} segments")
    sys.stderr.flush()


if __name__ == "__main__":
    raise SystemExit(main())
