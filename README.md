speech-to-text-cli
---

![Static Badge](https://img.shields.io/badge/project_status-mvp-lightgrey?style=flat&label=project%20status)
![Static Badge](https://img.shields.io/badge/platform-Apple%20Silicon-rgb(240,239,235))
![GitHub License](https://img.shields.io/badge/license-Apache%202.0-rgb(240,239,235))

## Abstract

Local real-time speech-to-text written in 🦀 Rust.

Text appears while you speak — dim while it can still change, plain once it
settles. Nothing leaves the machine.

Built on [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), chosen for
accuracy on accented and non-native English: 16.07% WER across 16 accent groups,
against Whisper large-v3's 21.30%.

## How it works

Qwen3-ASR is an offline model whose native streaming mode requires vLLM, which
requires CUDA. Streaming is therefore synthesised — the model re-runs on a growing
audio buffer every 500 ms, and a word settles once three consecutive hypotheses
agree on it. Voice activity detection keeps inference off silence and marks the
end of an utterance, which commits its final word.

Rust owns capture, VAD, windowing, the transcript state machine and rendering.
A Python/MLX sidecar owns the forward pass, over NDJSON on stdio.

Measured on an M3 Max: ~215 ms inference on a 6 s buffer, ~1.2 s for a word to
stabilise, 72% duty cycle over a 42 s unbroken passage.

## Setup

Requires Apple Silicon, Rust and Python 3.10+.

```sh
python3 -m venv .venv
.venv/bin/pip install "mlx-audio[stt]"
cargo build --release
```

The model (~1.8 GB) downloads on first run.

## Usage

```sh
# live microphone
./target/release/speech-to-text-cli --language en

# replay a 16 kHz mono WAV through the same pipeline
./target/release/speech-to-text-cli --simulate audio.wav
```

Omit `--language` to auto-detect across 52 languages.

## The live document

The transcript is a document, not a log scrolling past. It owns the screen while
you dictate and reaches your scrollback in full on exit, which is what makes it
editable by voice.

```
  This sentence has settled.
  So has this one.
│ This is the sentence you are saying right now,
│ and this part is still being decoded

  listening · 3 sentences · Luna: delete discard keep undo clear copy
```

| rendering | meaning |
|---|---|
| dim | still being decoded; this can change on its own |
| plain | settled; nothing changes it but you |
| `│` | the utterance in flight, as against the transcript |

A sentence goes plain once two more sentences follow it. Until then the model
still has the whole buffer and is free to revise it.

A pause is a full stop only provisionally. Resume within `--continue-ms` and the
audio is merged and re-decoded as one utterance, so the model — not a timer —
decides whether that was one thought or two. Merging happens on the audio, never
on the text, because punctuation across the seam needs the prosody.

## Commands

Six verbs, each addressed to the assistant by name. The comma is optional, and a
verb the model conjugated on you — "Luna deletes" — is the same command.

| spoken | effect |
|---|---|
| `Luna, delete` | drop the last sentence in the transcript |
| `Luna, discard` | drop everything you just said, however it split |
| `Luna, keep` | file it now instead of waiting out the settle |
| `Luna, clear` | throw away the whole transcript |
| `Luna, copy` | put the transcript on the clipboard |
| `Luna, undo` | put back what the last delete, discard or clear took |

`delete` reaches settled text, one sentence per go. `discard` is scoped to what
you are still saying and cannot reach further back, so it is safe to say without
looking at the screen. Every removal is undoable.

Nothing may come between the name and the verb, so these all stay dictation:

```
Luna went to the store.
We should discard the first draft.
The lunar module landed safely.
Luna, please copy
```

Rename the assistant with `--assistant Jarvis`. Pick a name you do not otherwise
dictate — "Luna deletes the offer." is a delete.

## Where your text goes

stdout on exit, one sentence per line. Redirect it with `> notes.txt`, or read it
out of your scrollback.

A session file under `~/.local/state/speech-to-text-cli/`, rewritten as you speak.
It is removed on a clean exit unless `--persist`, and survives every exit that is
not clean — which is what makes `clear` and `delete` safe to offer.

```sh
speech-to-text-cli --resume            # continue the most recent session
speech-to-text-cli --resume notes.txt  # continue a named file, read-only
```

`--resume` restores text, not session state: undo history does not survive it.

## Options

| flag | default | effect |
|---|---|---|
| `--assistant` | `Luna` | what to call it when giving an instruction |
| `--language` | auto | force a language, e.g. `en` |
| `--device` | system | input device name substring |
| `--simulate` | — | replay a 16 kHz mono WAV instead of the microphone |
| `--agreement` | `3` | hypotheses that must agree before text stops being dim |
| `--interval-ms` | `500` | shortest gap between re-runs; stretches on a long buffer |
| `--endpoint-ms` | `600` | silence that ends an utterance |
| `--open-ms` | `150` | how long a sound must last to count as speech |
| `--continue-ms` | `15000` | shortest silence before text settles; adapts upward |
| `--continue-max-ms` | `30000` | ceiling on that adaptation; equal to the floor disables it |
| `--trim-after-s` | `30` | when to start looking for a pause to trim the buffer at |
| `--rms-floor` | `-40` | silence floor in dBFS |
| `--persist` | off | keep the session file on exit |
| `--resume` | off | continue a previous session; bare, the most recent one |
| `--quiet` | off | suppress the "settling N s behind" notice |
| `--model` | Qwen3-ASR-1.7B-8bit | any MLX Qwen3-ASR repository |

`STT_TRACE=<file>` writes one line per sentence reaching the transcript, naming
the part of the pipeline that filed it, plus every buffer trim taken or refused.
File only, never the terminal, and free when unset.

## Troubleshooting

**Stray "Okay." or "Oh." when you aren't speaking.** Your room is above the
silence floor, and the model is transcribing what it is handed. Raise it with
`--rms-floor -30`: above your room, below your voice.

**Your keyboard is being transcribed.** Raise `--open-ms` instead — a keystroke
is louder than the room, so any floor that rejects one rejects a quiet voice too.
Duration separates them where level cannot. 300 ms is the ceiling, past which
first words start clipping.

## Limitations

- macOS and Apple Silicon only; `copy` shells out to `pbcopy`.
- Accuracy figures are Qwen's own benchmark, not independently replicated.
- Continuous background sound is not gated. Fast typing, a TV or a conversation
  across the room passes both defences, and a better microphone position beats
  any flag.
- The live view has no scrollback. A document taller than the terminal scrolls
  out of sight; it is still there, still deletable, still printed on exit.
- Correcting a word to a different one is not detected. Self-repair catches
  "regularing… regular", not "Tuesday… Wednesday".
- Never pausing costs lag, and past a minute of one unbroken sentence the buffer
  is trimmed anyway and can split it. Pausing for breath is free.
- Piped output shows nothing until exit, then prints the whole transcript.

See [CLAUDE.md](CLAUDE.md) for the decision record: what was measured, what was
rejected, and why.
