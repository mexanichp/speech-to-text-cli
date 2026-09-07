# speech-to-text-cli
🦀 Local real-time speech-to-text for Apple Silicon

[![CI/CD status](https://img.shields.io/github/actions/workflow/status/mexanichp/speech-to-text-cli/release.yml?style=flat&label=ci%2Fcd)](https://github.com/mexanichp/speech-to-text-cli/actions/workflows/release.yml)
[![Latest version](https://img.shields.io/github/v/release/mexanichp/speech-to-text-cli?style=flat&label=version&color=5C97CB)](https://github.com/mexanichp/speech-to-text-cli/releases/latest)
![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-9c4b34)

Transcribes speech in real time with [Qwen3-ASR-1.7B](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit), for accuracy on accented and non-native English, and polishes with [Qwen3-4B](https://huggingface.co/mlx-community/Qwen3-4B-4bit). Local native MLX support for Apple silicon with spoken commands.

## Setup

Install from current repo tap:

```sh
brew install mexanichp/tap/speech-to-text-cli
```

Alternatively, build the repo:

```sh
cargo build --release
```
> Dependencies:
> - [Rust](https://rustup.rs)
> - [uv](https://github.com/astral-sh/uv)

## Usage

```sh
# Transcribe from the microphone.
speech-to-text-cli --language en

# Transcribe a 16 kHz mono WAV file.
speech-to-text-cli --simulate audio.wav
```

The session file lives temporarily in `~/.local/state/speech-to-text-cli/`, and `--persist` keeps it after the session ends.

## Commands
Assuming the assistant parameter was not overridden:

| Command | Effect |
|---|---|
| `Luna, delete` | Drops the last sentence in the transcript |
| `Luna, discard` | Drops the last sentence you said |
| `Luna, keep` | Files the text now instead of waiting out the settle |
| `Luna, clear` | Throws away the whole transcript |
| `Luna, copy` | Runs the cleanup pass to the end, and then puts the transcript on the clipboard as prose |
| `Luna, undo` | Puts back what the last delete, discard, or clear took |

To change the assistant's name, see options below.

## Options

| Flag | Default | Effect |
|---|---|---|
| `--assistant` | `Luna` | Sets the name that prefixes a spoken command |
| `--language` | Auto | Forces a language, for example `en` |
| `--device` | System | Selects the input device whose name contains this substring |
| `--simulate` | | Replays a 16 kHz mono WAV file instead of the microphone |
| `--agreement` | `3` | Sets how many hypotheses must agree before text settles |
| `--interval-ms` | `400` | Sets the shortest gap between re-runs |
| `--endpoint-ms` | `600` | Sets the silence that ends an utterance |
| `--open-ms` | `150` | Sets how long a sound must last to count as speech |
| `--continue-ms` | `15000` | Sets the shortest silence before text settles |
| `--continue-max-ms` | `30000` | Sets the ceiling on settle adaptation |
| `--trim-after-s` | `12` | Sets the audio buffer length to hold to, which sets the latency |
| `--rms-floor` | `-40` | Sets the silence floor in dBFS |
| `--persist` | Off | Keeps the session file when you exit |
| `--resume [PATH]` | Off | Continues a previous session |
| `--quiet` | Off | Suppresses the settling-behind notice |
| `--model` | `mlx-community/Qwen3-ASR-1.7B-8bit` | Selects any MLX Qwen3-ASR repository |
| `--cleanup-model` | `mlx-community/Qwen3-4B-4bit` | Selects the text model that repairs sentence boundaries |
| `--no-cleanup` | Off | Leaves the transcript exactly as recognized |

| Variable | Effect |
|---|---|
| `STT_TRACE=FILE` | Logs every filed sentence, every buffer trim, and every cleanup pass to FILE |
| `STT_PYTHON=PATH` | Runs the sidecars on the interpreter at PATH instead of the managed one |

## Decision record

[CLAUDE.md](CLAUDE.md)
