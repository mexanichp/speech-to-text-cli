#!/usr/bin/env python3
"""Transcript cleanup sidecar.

Speaks newline-delimited JSON over stdin and stdout. The Rust host owns the
transcript; this process only re-reads a run of settled sentences and says where
it thinks the sentence boundaries and punctuation should have gone.

Protocol
--------
Request::

    {"id": <int>, "text": "<one settled sentence per line>"}

Reply::

    {"id": <int>, "text": "<repaired text>", "ms": <int>}
    {"id": <int>, "error": "<message>"}

Readiness banner, emitted once the model is resident::

    {"event": "ready", "model": "<repo>", "load_ms": <int>}

Why this is a separate process
------------------------------
It shares the GPU with speech recognition, and recognition is on the latency
path while this is not. Run in the host's loop, a pass here stalls the live one;
stalled long enough, the loop swallows a whole utterance per iteration and stops
trimming the audio buffer at all. Its own process means the host can hand work
over and collect it whenever it happens to be ready.

Invariants
----------
stdout carries protocol output only. The real stdout is duplicated to a private
handle and ``sys.stdout`` is redirected to stderr before the heavy imports, so a
stray ``print`` anywhere in the dependency tree lands in the log rather than
desynchronising the host.

The model is never shown anything but transcript text, and its reply is never
trusted: the host checks that no content word was invented before it accepts a
single edit.
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_PROTOCOL = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
sys.stdout = sys.stderr

#: Ceiling on generated tokens, as a multiple of the tokens sent in.
#:
#: The task is a rewrite, not a continuation, so a correct reply is about as
#: long as its input. The multiple leaves room for punctuation splitting one
#: token into several and bounds a decoder caught in a repetition loop, which
#: would otherwise run to the library default of several thousand tokens.
TOKEN_BUDGET_MULTIPLE = 2

#: Floor on that budget, covering the shortest inputs where the multiple alone
#: would leave no room at all.
MIN_TOKEN_BUDGET = 64

DEFAULT_PROMPT = """You are given a transcript produced by a speech recognizer that could only hear a few seconds of audio at a time.

Because of that, the line breaks and full stops in the input are unreliable. The recognizer put a full stop wherever it ran out of audio, so a line often ends in the middle of a sentence, and the next line often carries on from it. Where two chunks of audio overlapped, the same few words appear at the end of one line and the start of the next.

Re-read the whole passage and write out what the speaker actually said.

- Join lines that are two halves of one sentence, and remove the full stop between them.
- Where the end of one line repeats the start of the next, that repeat is an artifact: keep one copy.
- Where a short line repeats words that already appear in a neighbouring line, that line is an artifact: drop it.
- Split a line that runs two separate thoughts together.
- Fix capitalization and punctuation, and grammatical agreement where a wrong split caused it.
- Keep every word that carries meaning. Do not invent words, do not swap a word for a synonym, and do not drop a clause.
- Write one sentence per line.

Rewrite only the passage you are given. Output only the corrected passage."""

#: Worked examples, carried as turns rather than as text inside the prompt.
#:
#: They used to sit in the system message under ``Input:``/``Output:``
#: headings, and the model copied one of them into a reply: a 327 s recording
#: came back opening with 33 words of the second example, none of which the
#: speaker said. The host's content check refused that batch, which is the check
#: working, but a refused batch is a run of transcript left unrepaired.
#:
#: A worked example written inside the instruction is text sitting in the same
#: turn as the text to rewrite, and nothing in the format says which of the two
#: to rewrite. As turns the boundary is the one the chat template already draws,
#: so the model has no reason to continue the example rather than the input.
#:
#: This is the same failure §8 records for the recognition sidecar's transcript
#: prompt, in the one place the decision record says a static prompt is safe. It
#: is safe from *self-reinforcement*, because these words never come from the
#: speaker. It was never safe from being copied out.
EXAMPLES: list[tuple[str, str]] = [
    (
        "So I deliberately take very slowly here because I want to test how the product behaves on.\n"
        "such a long poses.\n"
        "What is too important to know is how exactly the algorithm commits the messages, and whether or not it commits them.\n"
        "Correctly.",
        "So I deliberately take very slowly here, because I want to test how the product behaves on such a long poses.\n"
        "What is too important to know is how exactly the algorithm commits the messages, and whether or not it commits them correctly.",
    ),
    (
        "and then be pretty quick and be very fluently.\n"
        "Pick lots of words in just a matter of seconds, so this tool has to recognize.\n"
        "So this too has to recognize both of these scenarios.",
        "And then be pretty quick and be very fluently, pick lots of words in just a matter of seconds.\n"
        "So this too has to recognize both of these scenarios.",
    ),
]

#: The prompt may be overridden for tuning without editing this file.
SYSTEM_PROMPT = DEFAULT_PROMPT
_override = os.environ.get("STT_CLEANUP_PROMPT")
if _override and os.path.exists(_override):
    SYSTEM_PROMPT = open(_override, encoding="utf-8").read()



def log(msg: str) -> None:
    """Write a diagnostic to stderr, which the host surfaces as a notice."""
    print(msg, file=sys.stderr, flush=True)


def emit(obj: dict) -> None:
    """Write one protocol message to the private stdout handle."""
    _PROTOCOL.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _PROTOCOL.flush()


def repair(model, tok, sampler, generate, text: str) -> str:
    """Return the model's reading of ``text``, or ``text`` if it produced none."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for source, corrected in EXAMPLES:
        messages.append({"role": "user", "content": source})
        messages.append({"role": "assistant", "content": corrected})
    messages.append({"role": "user", "content": text})

    prompt = tok.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False
    )
    # Budgeted against the passage, not against the whole prompt. The reply is a
    # rewrite of the passage and nothing else, so the examples and the rules are
    # not room the decoder is entitled to spend. Measuring the prompt instead
    # let the budget grow with every example added to it.
    budget = max(MIN_TOKEN_BUDGET, len(tok.encode(text)) * TOKEN_BUDGET_MULTIPLE)
    out = generate(model, tok, prompt=prompt, max_tokens=budget, sampler=sampler)
    return (out or "").strip() or text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-4B-4bit")
    args = ap.parse_args()

    started = time.time()
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model, tok = load(args.model)
    sampler = make_sampler(temp=0.0)

    # One short pass to absorb graph compilation, so the first real batch is not
    # charged for it. Failures here are not fatal; the host tolerates a slow
    # first reply and would rather have the process than not.
    try:
        repair(model, tok, sampler, generate, "A sentence.\nAnd another.")
    except Exception as exc:  # noqa: BLE001
        log(f"warmup failed (non-fatal): {exc}")

    emit(
        {
            "event": "ready",
            "model": args.model,
            "load_ms": int((time.time() - started) * 1000),
        }
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            text = req.get("text") or ""
            if not text.strip():
                emit({"id": rid, "text": "", "ms": 0})
                continue
            at = time.time()
            emit(
                {
                    "id": rid,
                    "text": repair(model, tok, sampler, generate, text),
                    "ms": int((time.time() - at) * 1000),
                }
            )
        except Exception as exc:  # noqa: BLE001
            emit({"id": rid, "error": f"{type(exc).__name__}: {exc}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
