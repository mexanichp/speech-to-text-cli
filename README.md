# speech-to-text-cli
🦀 Local real-time speech-to-text for Apple Silicon

![Static Badge](https://img.shields.io/badge/project_status-mvp-lightgrey?style=flat&label=project%20status)
![Static Badge](https://img.shields.io/badge/platform-Apple%20Silicon-rgb(240,239,235))
![GitHub License](https://img.shields.io/badge/license-Apache%202.0-rgb(240,239,235))

Dictation with live provisional text, built on [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) for accented and non-native English. Nothing leaves the machine.

## Setup

Apple Silicon, Rust, Python 3.10+. The model (~1.8 GB) downloads on first run.

```sh
python3 -m venv .venv
.venv/bin/pip install "mlx-audio[stt]"
cargo build --release
```

## Usage

```sh
# microphone
./target/release/speech-to-text-cli --language en

# 16 kHz mono WAV
./target/release/speech-to-text-cli --simulate audio.wav
```

Dim text is still being decoded. Plain text has settled. The transcript prints to stdout on exit and autosaves to `~/.local/state/speech-to-text-cli/`.

```
  This sentence has settled.
  So has this one.
│ This is the sentence you are saying right now,
│ and this part is still being decoded

  listening · 3 sentences · Luna: delete discard keep undo clear copy
```

## Commands

| spoken | effect |
|---|---|
| `Luna, delete` | drop the last sentence in the transcript |
| `Luna, discard` | drop everything you just said |
| `Luna, keep` | file it now instead of waiting out the settle |
| `Luna, clear` | throw away the whole transcript |
| `Luna, copy` | put the transcript on the clipboard |
| `Luna, undo` | put back what the last delete, discard or clear took |

The comma is optional. `Luna deletes` is the same command. Nothing may come between the name and the verb. Rename with `--assistant`.

## Options

| flag | default | effect |
|---|---|---|
| `--assistant` | `Luna` | name that prefixes a spoken command |
| `--language` | auto | force a language, e.g. `en` |
| `--device` | system | input device name substring |
| `--simulate` | | replay a 16 kHz mono WAV instead of the microphone |
| `--agreement` | `3` | hypotheses that must agree before text settles |
| `--interval-ms` | `500` | shortest gap between re-runs |
| `--endpoint-ms` | `600` | silence that ends an utterance |
| `--open-ms` | `150` | how long a sound must last to count as speech |
| `--continue-ms` | `15000` | shortest silence before text settles |
| `--continue-max-ms` | `30000` | ceiling on settle adaptation |
| `--trim-after-s` | `30` | when to look for a pause to trim the buffer at |
| `--rms-floor` | `-40` | silence floor in dBFS |
| `--persist` | off | keep the session file on exit |
| `--resume [PATH]` | off | continue a previous session |
| `--quiet` | off | suppress the settling-behind notice |
| `--model` | `Qwen3-ASR-1.7B-8bit` | any MLX Qwen3-ASR repository |

`STT_TRACE=<file>` logs every sentence filed and every buffer trim.

## Decision record

[CLAUDE.md](CLAUDE.md)
