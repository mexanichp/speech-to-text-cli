#!/usr/bin/env python3
"""Qwen3-ASR inference sidecar.

Speaks newline-delimited JSON over stdin/stdout. The Rust host owns capture,
VAD, buffering and the transcript state machine; this process only runs the
forward pass.

Protocol
--------
in   {"id": <int>, "pcm": "<base64 f32le @16kHz mono>"}
out  {"id": <int>, "text": "<hypothesis>", "ms": <int>}
out  {"id": <int>, "error": "<message>"}

On startup, once the model is resident:
out  {"event": "ready", "model": "<repo>", "load_ms": <int>}

Every reply echoes the request's `id`, so the host can tell that the stream is
still in step. `id` is null only on a request that could not be parsed at all.

The protocol stream is only correct if nothing else writes to it, and imports
below pull in a large dependency tree that is entitled to print. So the real
stdout is duplicated to a private handle and `sys.stdout` is redirected to
stderr before any of them load: a stray `print()` anywhere now lands in the log
instead of desynchronising the host.
"""

import argparse
import base64
import json
import os
import sys
import time

# Progress bars would interleave with the protocol stream on the terminal.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Claim stdout for the protocol before anything else can write to it, then point
# `sys.stdout` at stderr so ordinary printing is harmless. Must happen before
# the heavy imports below.
_PROTOCOL = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
sys.stdout = sys.stderr

import numpy as np

SAMPLE_RATE = 16000

# The model's own chunker splits anything shorter than this into nothing
# useful; below it we return empty rather than burning a forward pass.
MIN_AUDIO_SEC = 0.35


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def emit(obj: dict) -> None:
    _PROTOCOL.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _PROTOCOL.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    # Only reached when the script is run by hand; the Rust host always passes
    # --model explicitly. Kept in step with the CLI's default regardless.
    ap.add_argument("--model", default="mlx-community/Qwen3-ASR-1.7B-8bit")
    ap.add_argument("--language", default=None, help="force a language; omit to auto-detect")
    args = ap.parse_args()

    t0 = time.monotonic()
    from mlx_audio.stt.utils import load_model

    model = load_model(args.model)
    load_ms = int((time.monotonic() - t0) * 1000)

    # First call carries graph-compilation cost. Burn it on silence now so the
    # first real utterance isn't penalised.
    try:
        model.generate(np.zeros(SAMPLE_RATE, dtype=np.float32), language=args.language)
    except Exception as exc:  # warmup is best-effort
        log(f"warmup failed (non-fatal): {exc}")

    emit({"event": "ready", "model": args.model, "load_ms": load_ms})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            # No id to echo — the host treats an unattributed reply as the
            # answer to whatever it last sent, which it is.
            emit({"id": None, "error": f"bad json: {exc}"})
            continue

        req_id = req.get("id")

        try:
            pcm = np.frombuffer(base64.b64decode(req["pcm"]), dtype=np.float32)

            if len(pcm) < MIN_AUDIO_SEC * SAMPLE_RATE:
                emit({"id": req_id, "text": "", "ms": 0})
                continue

            t = time.monotonic()
            # DO NOT pass system_prompt=<transcript here>. It looks like context
            # carry-over and is not.
            #
            # `_build_prompt` drops the string verbatim into the system turn of
            # the chat template, ahead of the audio. Whenever the audio carries
            # no speech — an open mic in a room with any background noise — the
            # transcript is then the strongest signal in the context and the
            # decoder simply copies it out. Measured on this model: with a
            # transcript prompt, silence and noise at every level returned that
            # transcript verbatim; with no prompt, the same audio returned "".
            #
            # It also self-reinforces, because the echo gets committed and comes
            # back in the next prompt.
            #
            # And it bought nothing: transcription of real speech was
            # byte-identical with and without it. The open weights have no
            # context biasing — that is Qwen3-ASR-Flash, the API-only model.
            out = model.generate(
                pcm,
                language=args.language,
                temperature=0.0,
            )
            ms = int((time.monotonic() - t) * 1000)

            text = getattr(out, "text", None)
            if text is None:
                text = str(out)

            emit({"id": req_id, "text": text.strip(), "ms": ms})

        except Exception as exc:
            emit({"id": req_id, "error": f"{type(exc).__name__}: {exc}"})

    return 0


if __name__ == "__main__":
    sys.exit(main())
