#!/usr/bin/env python3
"""Qwen3-ASR inference sidecar.

Speaks newline-delimited JSON over stdin and stdout. The Rust host owns capture,
voice activity detection, buffering and the transcript state machine; this
process runs only the forward pass.

Protocol
--------
Request::

    {"id": <int>, "pcm": "<base64 f32le @16kHz mono>", "hint": <bool>}

Reply::

    {"id": <int>, "text": "<hypothesis>", "ms": <int>}
    {"id": <int>, "error": "<message>"}

Readiness banner, emitted once the model is resident::

    {"event": "ready", "model": "<repo>", "load_ms": <int>}

Every reply echoes the request id, so the host can detect a stream that has
fallen out of step. The id is null only for a request that could not be parsed.

``hint`` selects between no system prompt, which is what almost every window
receives, and the fixed vocabulary hint supplied once at startup. It is a
boolean rather than a string: the host chooses whether the prompt applies but
never what it contains, so no request field exists through which transcript text
could reach the model.

Invariants
----------
stdout carries protocol output only. The real stdout is duplicated to a private
handle and ``sys.stdout`` is redirected to stderr before the heavy imports, so a
stray ``print`` anywhere in the dependency tree lands in the log rather than
desynchronising the host.
"""

import argparse
import base64
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_PROTOCOL = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
sys.stdout = sys.stderr

import numpy as np

SAMPLE_RATE = 16000

#: Shortest buffer worth a forward pass, in seconds. The model's chunker yields
#: nothing useful below this, so shorter windows return empty immediately.
MIN_AUDIO_SEC = 0.35

#: Audio token rate of the model, in hertz.
#:
#: A transcript cannot legitimately need more text tokens per second than the
#: audio carries, which makes this the right ceiling on generation. The
#: library's own default permits several minutes of output for any window, so a
#: decoder caught in a repetition loop runs to that cap and the pass takes as
#: long as generating it. Budgeting against the audio bounds the worst case
#: without affecting a healthy decode, which never approaches it.
TOKENS_PER_SEC = 12.5

#: Additional tokens allowed regardless of audio length, covering the prompt
#: scaffolding emitted before any transcript.
MIN_TOKEN_BUDGET = 64


def log(msg: str) -> None:
    """Write a diagnostic to stderr, which the host surfaces as a notice."""
    print(msg, file=sys.stderr, flush=True)


def emit(obj: dict) -> None:
    """Write one protocol message to the private stdout handle."""
    _PROTOCOL.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _PROTOCOL.flush()


def token_budget(samples: int) -> int:
    """Return the generation cap for a window of ``samples`` audio samples."""
    return int(samples / SAMPLE_RATE * TOKENS_PER_SEC) + MIN_TOKEN_BUDGET


def warm_up(model, language: str | None, system_prompt: str | None) -> None:
    """Run the model once per prompt state to absorb graph compilation.

    Both states are warmed because the session uses both, and the hinted pass
    decides a spoken command, which is the worst moment to pay a first-call
    penalty. Failures are non-fatal.
    """
    for prompt in {None, system_prompt} if system_prompt else {None}:
        try:
            model.generate(
                np.zeros(SAMPLE_RATE, dtype=np.float32),
                language=language,
                system_prompt=prompt,
            )
        except Exception as exc:
            log(f"warmup failed (non-fatal): {exc}")


def transcribe(model, pcm, language: str | None, system_prompt: str | None):
    """Transcribe one window and return its text.

    ``system_prompt`` must never carry transcript text. The template places it
    verbatim ahead of the audio, so a prompt derived from what the speaker said
    becomes the strongest signal on a window holding no speech and is copied
    out; the echo is then committed and returns in the next prompt. The fixed
    vocabulary hint is safe because it is set once at startup, cannot grow, and
    cannot contain dictation.
    """
    out = model.generate(
        pcm,
        language=language,
        temperature=0.0,
        system_prompt=system_prompt,
        max_tokens=token_budget(len(pcm)),
    )
    text = getattr(out, "text", None)
    return str(out) if text is None else text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-ASR-1.7B-8bit")
    ap.add_argument("--language", default=None, help="force a language; omit to auto-detect")
    ap.add_argument("--system-prompt", default=None, help="fixed vocabulary hint")
    args = ap.parse_args()

    t0 = time.monotonic()
    from mlx_audio.stt.utils import load_model

    model = load_model(args.model)
    load_ms = int((time.monotonic() - t0) * 1000)

    warm_up(model, args.language, args.system_prompt)
    emit({"event": "ready", "model": args.model, "load_ms": load_ms})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            emit({"id": None, "error": f"bad json: {exc}"})
            continue

        req_id = req.get("id")

        try:
            pcm = np.frombuffer(base64.b64decode(req["pcm"]), dtype=np.float32)

            if len(pcm) < MIN_AUDIO_SEC * SAMPLE_RATE:
                emit({"id": req_id, "text": "", "ms": 0})
                continue

            t = time.monotonic()
            prompt = args.system_prompt if req.get("hint") else None
            text = transcribe(model, pcm, args.language, prompt)
            ms = int((time.monotonic() - t) * 1000)

            emit({"id": req_id, "text": text.strip(), "ms": ms})

        except Exception as exc:
            emit({"id": req_id, "error": f"{type(exc).__name__}: {exc}"})

    return 0


if __name__ == "__main__":
    sys.exit(main())
