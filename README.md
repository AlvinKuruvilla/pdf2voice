# pdf2voice

Turn a research-paper PDF into natural-sounding audio you can actually listen to
on a walk. It reads the paper in the right order, skips the parts you don't want
narrated (bibliography, figure captions, page numbers), and speaks the rest with
a local neural voice — no API keys, no per-use cost, nothing leaves your machine.

## Quickstart

```
git clone https://github.com/AlvinKuruvilla/pdf2voice
cd pdf2voice
uv sync
uv run pdf2voice paper.pdf          # -> paper.mp3
```

Two optional system packages improve the output:

```
brew install ffmpeg      # enables MP3/M4A output (WAV works without it)
brew install espeak-ng   # better pronunciation of rare/technical words
```

## Usage

```
uv run pdf2voice paper.pdf                 # -> paper.mp3
uv run pdf2voice paper.pdf -o talk.m4a     # format follows the extension (.mp3/.m4a/.wav)
uv run pdf2voice paper.pdf -v am_michael   # pick the voice
uv run pdf2voice paper.pdf -s 1.15         # speak faster (default 1.0)
uv run pdf2voice paper.pdf --text-only     # print the cleaned narration, synth nothing
```

`--text-only` is the fast way to see exactly what will be spoken before spending
minutes on synthesis.

## How it works

Three stages, each in its own module so you can inspect the text before paying
for synthesis:

```text
PDF ──extract.py──▶ reading-order paragraphs ──speechify.py──▶ spoken text ──tts.py──▶ audio
```

- **`extract.py`** — the part that decides whether the audio is coherent.
  Research papers are two-column, so it re-sorts PyMuPDF's blocks into true
  reading order (left column top-to-bottom, then right), rejoins words broken
  across line wraps (`obser-\nvations` → `observations`), strips inline citation
  brackets and list bullets, drops running headers/footers and figure captions,
  and stops at the bibliography.
- **`speechify.py`** — makes clean text sound right *spoken*. Expands the
  shorthand a reader skips but a synth reads literally (`e.g.` → "for example",
  `et al.` → "and colleagues"), deletes URLs/DOIs, and gives the notation-heavy
  results sections a fighting chance (`α` → "alpha", `1:N` → "one to N",
  `0.98–0.99` → "0.98 to 0.99", `p < 0.05` → "p less than 0.05").
- **`tts.py`** — [Kokoro](https://github.com/hexgrad/kokoro), an 82M-parameter
  open-weights model, does the voice. This module packs paragraphs into
  model-sized segments, trims the silence Kokoro pads onto each one, then stitches
  them back with sentence- and paragraph-length pauses so it paces like a person,
  and encodes to MP3/M4A via ffmpeg (WAV if ffmpeg isn't installed).

## Voices

American (`af_`/`am_`) and British (`bf_`/`bm_`) voices ship with Kokoro; `f`/`m`
is the voice's gender. A few natural ones: `af_heart` (default), `af_bella`,
`am_michael`, `bf_emma`. Set with `-v/--voice`.

## Trade-offs

- **Pacing is deliberate.** The default voice speaks at ~115 words per minute —
  clear and unhurried, but slower than an audiobook. Use `-s 1.15` for a brisker
  ~140 wpm; it stays natural on Kokoro.
- **Notation is approximated, not rendered.** Common symbols, Greek letters,
  comparison operators, numeric ranges and ratios are spoken correctly. What is
  *not* handled: subscripts and superscripts, which PyMuPDF flattens into the
  baseline (`t_release` becomes `trelease`), and multi-line equations. The
  abstract, introduction, related work, and conclusion — the parts worth
  listening to — come out clean; a heavy methods section still has rough patches.
  This is a deliberate line: full LaTeX-math-to-speech, and subscript
  reconstruction from font metrics, is a project of its own and only benefits
  sections you'd read off the screen anyway.
- **Line-wrap hyphens are always removed**, which occasionally merges a genuine
  compound (`closed-set` wrapped as `closed-\nset` → `closedset`). Soft
  justification hyphens vastly outnumber real ones, so removing wins on average;
  the spoken difference is negligible.
- **CPU synthesis is roughly real-time.** A 10-page paper takes a few minutes and
  runs entirely offline once the model weights are cached.

## Requirements

Python 3.12+, [uv](https://docs.astral.sh/uv/). Everything else is pulled in by
`uv sync`. First run downloads the Kokoro weights (~few hundred MB), cached after.
