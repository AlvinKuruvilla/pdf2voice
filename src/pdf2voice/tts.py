"""Synthesise paragraphs into one continuous, naturally-paced audio file.

Kokoro produces the voice; this module handles everything around it: splitting
paragraphs into segments the model can handle, stitching the segments back
together with pauses that mimic how a person paces sentences and paragraphs, and
encoding the result to WAV or (via ffmpeg) a compact MP3/M4A.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 24_000  # Kokoro's fixed output rate.

# Kokoro's model caps a segment at ~510 phoneme tokens. Characters are a coarse
# proxy; 400 keeps us comfortably under even for phoneme-dense text, while
# staying long enough that prosody spans a full clause rather than a few words.
_MAX_SEGMENT_CHARS = 400

# Pauses inserted between segments. A beat between sentences and a longer breath
# between paragraphs is most of what separates "read aloud" from "listed off".
_SENTENCE_GAP_S = 0.28
_PARAGRAPH_GAP_S = 0.55

# Sentence boundary: end punctuation followed by whitespace and a capital or
# quote. Abbreviations are already expanded in speechify, so this rarely misfires.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z“\"(])")


def synthesize(
    paragraphs: list[str],
    out_path: str,
    voice: str = "af_heart",
    speed: float = 1.0,
    on_progress=None,
) -> Path:
    """Synthesise `paragraphs` to `out_path` and return the written path.

    `on_progress(done, total)` is called after each segment if provided, so a
    CLI can show a progress bar without this module knowing anything about the
    terminal.
    """
    from kokoro import KPipeline  # imported lazily; loading torch is slow

    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")

    segments = _segment(paragraphs)
    pieces: list[np.ndarray] = []
    for done, (text, ends_paragraph) in enumerate(segments, start=1):
        pieces.append(_render(pipeline, text, voice, speed))
        gap = _PARAGRAPH_GAP_S if ends_paragraph else _SENTENCE_GAP_S
        pieces.append(np.zeros(int(gap * SAMPLE_RATE), dtype=np.float32))
        if on_progress is not None:
            on_progress(done, len(segments))

    audio = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    return _write(audio, Path(out_path))


def _segment(paragraphs: list[str]) -> list[tuple[str, bool]]:
    """Pack paragraphs into (segment_text, ends_paragraph) chunks.

    Each chunk is at most `_MAX_SEGMENT_CHARS`, formed by greedily grouping whole
    sentences. The last chunk of a paragraph is flagged so the caller can insert
    a longer pause there.
    """
    segments: list[tuple[str, bool]] = []
    for para in paragraphs:
        sentences = [
            piece
            for sentence in _SENTENCE_BOUNDARY.split(para)
            for piece in _split_oversized(sentence)
        ]
        chunk = ""
        chunks: list[str] = []
        for sentence in sentences:
            if chunk and len(chunk) + len(sentence) + 1 > _MAX_SEGMENT_CHARS:
                chunks.append(chunk)
                chunk = sentence
            else:
                chunk = f"{chunk} {sentence}".strip()
        if chunk:
            chunks.append(chunk)
        for i, c in enumerate(chunks):
            segments.append((c, i == len(chunks) - 1))
    return segments


# Below this amplitude (~ -46 dBFS) a sample counts as silence for trimming.
_SILENCE_FLOOR = 0.005


def _split_oversized(sentence: str) -> list[str]:
    """Break a sentence longer than the segment cap on clause boundaries.

    Kokoro silently truncates a segment past its token limit, dropping the tail
    from the audio. Sentences over the cap are rare in practice, but when one
    appears we split it at commas/semicolons (and, as a last resort, hard-slice)
    so no words are lost. Short sentences pass through untouched.
    """
    if len(sentence) <= _MAX_SEGMENT_CHARS:
        return [sentence]

    pieces: list[str] = []
    current = ""
    for clause in re.split(r"(?<=[,;:])\s+", sentence):
        if current and len(current) + len(clause) + 1 > _MAX_SEGMENT_CHARS:
            pieces.append(current)
            current = clause
        else:
            current = f"{current} {clause}".strip()
    if current:
        pieces.append(current)

    # A single clause still over the cap (no punctuation to break on): hard-slice.
    out: list[str] = []
    for piece in pieces:
        while len(piece) > _MAX_SEGMENT_CHARS:
            out.append(piece[:_MAX_SEGMENT_CHARS])
            piece = piece[_MAX_SEGMENT_CHARS:]
        out.append(piece)
    return out


def _render(pipeline, text: str, voice: str, speed: float) -> np.ndarray:
    """Synthesise one segment to a float32 waveform, silence-trimmed.

    A segment has no internal newline, so Kokoro yields exactly one result; we
    concatenate defensively in case a future version splits further. Kokoro pads
    each segment with leading/trailing silence; left in, that padding stacks up
    across ~90 segments into seconds of dead air. We trim it here so the *only*
    pauses between segments are the calibrated gaps `synthesize` inserts.
    """
    parts = [
        r.audio.detach().cpu().numpy().astype(np.float32)
        for r in pipeline(text, voice=voice, speed=speed)
    ]
    audio = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
    return _trim_silence(audio)


def _trim_silence(audio: np.ndarray) -> np.ndarray:
    """Strip near-silent samples from the start and end of `audio`.

    Internal pauses (at commas and periods) are left untouched — they are what
    makes the delivery sound human; only the boundary padding is removed.
    """
    loud = np.flatnonzero(np.abs(audio) > _SILENCE_FLOOR)
    if loud.size == 0:
        return audio
    return audio[loud[0] : loud[-1] + 1]


def _write(audio: np.ndarray, out_path: Path) -> Path:
    """Write `audio` to `out_path`, encoding via ffmpeg for non-WAV formats."""
    if out_path.suffix.lower() == ".wav":
        sf.write(str(out_path), audio, SAMPLE_RATE)
        return out_path

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        fallback = out_path.with_suffix(".wav")
        sf.write(str(fallback), audio, SAMPLE_RATE)
        print(f"ffmpeg not found; wrote {fallback} instead of {out_path.name}")
        return fallback

    # Write a temporary WAV, then let ffmpeg encode to the requested container.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, SAMPLE_RATE)
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(tmp_path),
             "-b:a", "192k", str(out_path)],
            check=True,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    return out_path
