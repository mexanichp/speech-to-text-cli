#!/usr/bin/env python3
"""Transcribe a whole recording in one pass, as the quality ceiling.

The live pipeline hears the speech a few seconds at a time. This hears all of
it at once, with the full right-context the live path can never have, and so
produces the best transcript these two models are capable of on this audio.
That is the number to chase, not a rival implementation: the goal stated in
CLAUDE.md is this transcript's quality at the live path's latency.

Two stages, matching the two the live pipeline runs:

``asr``
    Qwen3-ASR over the audio, one window. Long recordings are cut at the
    quietest point inside a search band rather than at a fixed offset, so a
    chunk boundary lands in a pause instead of mid-word.

``clean``
    The same text model and the same prompt the cleanup sidecar uses, over the
    ASR output. Run because the live path runs it: comparing a cleaned live
    transcript against a raw ceiling would score the cleanup pass instead of
    the streaming.

Usage::

    .venv/bin/python scripts/oracle.py in.wav -o scratch/oracle.md
    .venv/bin/python scripts/oracle.py in.wav --stage asr    # ceiling, no cleanup
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import wave

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

SAMPLE_RATE = 16000

#: Longest audio handed to one ASR forward pass, in seconds.
#:
#: Not a model limit. Memory and time both grow with the window, and beyond a
#: few minutes a single pass on this class of machine is slow enough that the
#: script stops being usable as a check you run after a change. The point of
#: this tool is right-context, and a chunk this size has plenty.
DEFAULT_CHUNK_SEC = 120.0

#: How far either side of a chunk boundary to search for a pause to cut at.
SEEK_SEC = 10.0

#: Window the quietest-point search averages over, in seconds.
QUIET_WIN_SEC = 0.2


def read_wav(path: str) -> np.ndarray:
    """Read a 16 kHz mono PCM wav as f32 in [-1, 1]."""
    with wave.open(path, "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1:
            sys.exit(
                f"{path}: need 16 kHz mono, got {w.getframerate()} Hz "
                f"{w.getnchannels()}ch. Convert with:\n"
                f"  ffmpeg -i in.mp4 -vn -ac 1 -ar 16000 -acodec pcm_s16le out.wav"
            )
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def quietest_near(pcm: np.ndarray, target: int, seek: int) -> int:
    """Return the quietest sample offset within ``seek`` of ``target``.

    A chunk boundary that lands mid-word costs two words at every seam, which
    is exactly the damage this script exists to measure the absence of.
    """
    lo = max(0, target - seek)
    hi = min(len(pcm), target + seek)
    if hi - lo < SAMPLE_RATE:
        return target

    win = int(QUIET_WIN_SEC * SAMPLE_RATE)
    band = pcm[lo:hi]
    # Mean absolute amplitude over a sliding window, via a prefix sum.
    csum = np.cumsum(np.abs(band), dtype=np.float64)
    energy = (csum[win:] - csum[:-win]) / win
    return lo + int(np.argmin(energy)) + win // 2


def chunks(pcm: np.ndarray, chunk_sec: float) -> list[tuple[int, int]]:
    """Split into (start, end) sample ranges, cutting at pauses."""
    if chunk_sec <= 0 or len(pcm) <= chunk_sec * SAMPLE_RATE:
        return [(0, len(pcm))]

    step = int(chunk_sec * SAMPLE_RATE)
    seek = int(SEEK_SEC * SAMPLE_RATE)
    out, start = [], 0
    while len(pcm) - start > step:
        cut = quietest_near(pcm, start + step, seek)
        out.append((start, cut))
        start = cut
    out.append((start, len(pcm)))
    return out


def run_asr(pcm: np.ndarray, model_id: str, language: str | None, chunk_sec: float) -> str:
    from mlx_audio.stt.utils import load_model

    say(f"loading {model_id}")
    model = load_model(model_id)

    parts = []
    for i, (a, b) in enumerate(chunks(pcm, chunk_sec), 1):
        say(f"asr window {i}: {(b - a) / SAMPLE_RATE:.1f}s")
        at = time.monotonic()
        out = model.generate(
            pcm[a:b],
            language=language,
            temperature=0.0,
            system_prompt=None,
            # Same budget rule as the sidecar: a transcript cannot need more
            # text tokens per second than the audio carries.
            max_tokens=int((b - a) / SAMPLE_RATE * 12.5) + 64,
        )
        text = getattr(out, "text", None)
        parts.append((str(out) if text is None else text).strip())
        say(f"  {int((time.monotonic() - at) * 1000)} ms")
    return " ".join(p for p in parts if p)


#: Words the content check lets a reply add or drop freely.
#:
#: Mirrors ``FUNCTION_WORDS`` in ``src/cleanup.rs``. The Rust list is the
#: authoritative one: it guards the transcript the speaker keeps, while this
#: guards a reference the speaker never sees. They are kept in step by hand, and
#: a drift costs a reference that is scored slightly differently from the live
#: path rather than a wrong transcript.
FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "so", "then", "than", "that", "this", "these", "those",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "have", "has",
    "had", "because", "though", "although", "since", "whether", "as", "if", "unless",
    "who", "whom", "whose", "which", "what", "where", "when", "why", "how", "while",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "of", "to", "in", "on", "at", "for", "with", "by", "from", "into", "onto", "about", "over",
    "under", "through", "between", "up", "down", "out", "off", "there", "here",
    "it", "its", "they", "them", "their", "he", "him", "his", "she", "her", "we", "us", "our",
    "you", "your", "i", "my", "me", "all", "any", "some", "such", "no", "not", "own", "same",
    "very", "too", "more", "most", "only", "also", "even", "still", "again", "just", "ever",
    "never", "always", "already",
}

#: Share of the input's content words a reply may lose, as a denominator.
MAY_LOSE_DEN = 4


def stem(word: str) -> str:
    """Strip a word to the form two spellings of it share. See ``cleanup::stem``."""
    base = "".join(c for c in word.lower() if c.isalnum() or c == "'")
    for suffix in ("ing", "ed", "es", "ly", "s"):
        if len(base) > len(suffix) + 2 and base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def content(text: str) -> set[str]:
    """The content-word stems of a passage."""
    return {s for s in (stem(w) for w in text.split()) if s and s not in FUNCTION_WORDS}


def accepts(before: str, after: str) -> str | None:
    """Why a reply must be refused, or ``None`` if it may be kept.

    The same two-directional check the host runs in ``cleanup::check``, and the
    reason this script has one at all: the ceiling is produced by the same model
    and the same prompt as the live path, so it inherits the same failures. Run
    unchecked it did, and a 327 s recording came back opening with 33 words the
    speaker never said, copied out of a worked example in the prompt. A
    reference carrying invented text is worse than no reference: every later
    measurement is scored against it.
    """
    said, kept = content(before), content(after)
    if kept - said:
        return f"invented {sorted(kept - said)[:6]}"
    if len(said - kept) * MAY_LOSE_DEN > len(said):
        return f"dropped {len(said - kept)} of {len(said)} content word(s)"
    return None


def run_cleanup(text: str, model_id: str) -> str:
    """Run the cleanup sidecar's own prompt over the text, in sentence batches."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sidecar"))
    # The sidecar points sys.stdout at stderr on import, which is right for a
    # process whose stdout is protocol and wrong for this one. Reuse its prompt
    # and its repair(), not its stream discipline.
    real_stdout = sys.stdout
    import cleanup_sidecar as cs

    sys.stdout = real_stdout

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    say(f"loading {model_id}")
    model, tok = load(model_id)
    sampler = make_sampler(temp=0.0)

    lines = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    lines = [s if s.endswith((".", "?", "!")) else s + "." for s in lines]

    # Batched to match what the live path hands the model. A pass over the
    # whole transcript at once is a different task with a different failure
    # mode, and CLAUDE.md 8 records that longer batches provoke summarising.
    out = []
    for i in range(0, len(lines), cs_batch := 12):
        batch = "\n".join(lines[i : i + cs_batch])
        say(f"cleanup batch {i // cs_batch + 1}: {len(lines[i:i + cs_batch])} sentence(s)")
        reply = cs.repair(model, tok, sampler, generate, batch)
        # Refused the way the host refuses: the batch stands as the recognizer
        # left it. A reference is worth having only if every word in it was
        # said, so an unusable reply costs the repair rather than the run.
        if why := accepts(batch, reply):
            say(f"  refused ({why}) — keeping the batch as recognized")
            reply = batch
        out.append(reply)
    return "\n".join(out)


def say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("wav", help="16 kHz mono wav")
    ap.add_argument("-o", "--out", help="write here as well as to stdout")
    ap.add_argument("--stage", choices=["asr", "clean", "both"], default="both")
    ap.add_argument("--model", default="mlx-community/Qwen3-ASR-1.7B-8bit")
    ap.add_argument("--cleanup-model", default="mlx-community/Qwen3-4B-4bit")
    ap.add_argument("--language", default="en")
    ap.add_argument("--chunk-s", type=float, default=DEFAULT_CHUNK_SEC,
                    help="0 for a single window over the whole recording")
    args = ap.parse_args()

    pcm = read_wav(args.wav)
    say(f"{args.wav}: {len(pcm) / SAMPLE_RATE:.1f}s")

    text = run_asr(pcm, args.model, args.language, args.chunk_s)
    if args.stage in ("clean", "both"):
        raw = text
        text = run_cleanup(raw, args.cleanup_model)
        if args.out:
            with open(args.out + ".asr", "w", encoding="utf-8") as f:
                f.write(raw + "\n")

    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        say(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
