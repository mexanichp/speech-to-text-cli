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

# Qwen3-ASR emits audio tokens at 12.5 Hz (80 ms/token), so a transcript can
# never legitimately need more text tokens than that per second of audio —
# ordinary English needs roughly a third of it. `generate`'s own default is
# max_tokens=8192, which is 655 s of allowed output for *any* window however
# short, so a decoder that gets stuck in a repetition loop simply runs to that
# cap and the forward pass takes as long as it takes.
#
# Reported live: a 63.6 s buffer took ~67 s in a single pass, which stretched the
# host's adaptive tick to ~84 s. Nothing was decoded or drawn for over a minute
# and the capture channel backed up behind it. Budgeting tokens against the audio
# bounds that to something the tick stretch can actually absorb, and costs
# nothing on a healthy decode because a healthy decode never approaches it.
TOKENS_PER_SEC = 12.5

# Headroom for short buffers, where the per-second budget alone would be tighter
# than the prompt scaffolding the model emits before any transcript.
MIN_TOKEN_BUDGET = 64


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
    # Set once, at spawn, from the host's static configuration — never per
    # request, and never from the transcript. See the note at the generate call.
    ap.add_argument("--system-prompt", default=None, help="fixed vocabulary hint")
    args = ap.parse_args()

    t0 = time.monotonic()
    from mlx_audio.stt.utils import load_model

    model = load_model(args.model)
    load_ms = int((time.monotonic() - t0) * 1000)

    # First call carries graph-compilation cost. Burn it on silence now so the
    # first real utterance isn't penalised.
    try:
        model.generate(
            np.zeros(SAMPLE_RATE, dtype=np.float32),
            language=args.language,
            # Same prompt as the real calls, so warmup compiles the graph the
            # session will actually use rather than one a token shorter.
            system_prompt=args.system_prompt,
        )
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
            # DO NOT pass the transcript here, whatever else this argument
            # carries. That is the distinction, and it is the whole of it.
            #
            # `_build_prompt` drops the string verbatim into the system turn of
            # the chat template, ahead of the audio. With the **transcript** in
            # it, any window holding no speech — an open mic in a room with
            # background noise — made that transcript the strongest signal in the
            # context and the decoder copied it straight out. It also
            # self-reinforced: the echo was committed and came back next time.
            #
            # `args.system_prompt` is safe because it is the opposite kind of
            # string: fixed for the session, set once from the host's CLI flags,
            # and never derived from anything the speaker said. It cannot grow,
            # cannot contain dictation, and cannot feed back. Measured on
            # `Qwen3-ASR-1.7B-8bit` with a command list in it — 20s of noise at
            # −38 dBFS returned "" exactly as with no prompt, 41s of dictation
            # was byte-identical, and it fixed "Luna, roll back" to
            # "Luna rollback".
            #
            # The structural guarantee is that the *protocol* has no text field.
            # A per-request prompt is the shape this failure needs, and it does
            # not exist.
            out = model.generate(
                pcm,
                language=args.language,
                temperature=0.0,
                system_prompt=args.system_prompt,
                max_tokens=int(len(pcm) / SAMPLE_RATE * TOKENS_PER_SEC) + MIN_TOKEN_BUDGET,
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
