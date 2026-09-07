# speech-to-text-cli
🦀 Local real-time speech-to-text for Apple Silicon

[![CI/CD status](https://img.shields.io/github/actions/workflow/status/mexanichp/speech-to-text-cli/release.yml?style=flat&label=ci%2Fcd&labelColor=EE5340)](https://github.com/mexanichp/speech-to-text-cli/actions/workflows/release.yml)
[![Latest version](https://img.shields.io/github/v/release/mexanichp/speech-to-text-cli?style=flat&label=version&color=4AC9E3&labelColor=EE5340)](https://github.com/mexanichp/speech-to-text-cli/releases/latest)
![Platform: Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon-EE5340?labelColor=4AC9E3)
![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-4AC9E3?labelColor=EE5340)

Transcribes speech in real time with [Qwen3-ASR-1.7B](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit), which is selected for accuracy on accented and non-native English, and re-punctuates the settled sentences with [Qwen3-4B](https://huggingface.co/mlx-community/Qwen3-4B-4bit). All inference runs on the local machine.

## Setup

```sh
brew install mexanichp/tap/speech-to-text-cli
```

To build it yourself on Apple Silicon, install [Rust](https://rustup.rs) and [uv](https://github.com/astral-sh/uv), and then run the following command:

```sh
cargo build --release
```

The first run creates the Python environment in `~/.cache/speech-to-text-cli` and downloads two models of about 1.8 GB and 2.3 GB.

## Usage

```sh
# Transcribe from the microphone.
./target/release/speech-to-text-cli --language en

# Transcribe a 16 kHz mono WAV file.
./target/release/speech-to-text-cli --simulate audio.wav
```

The transcript prints to stdout as prose when you exit, and it autosaves to `~/.local/state/speech-to-text-cli/` while you talk.

```
  Text with no mark has been read back in context and is finished.
· This sentence is transcribed, and the cleanup pass has not reached it.
│ This is the sentence you are saying right now,
│ and this part is still being decoded

  listening · 3 sentences · Luna: delete discard keep undo clear copy
```

| How it looks | What it means |
|---|---|
| Dim, `│` | The recognizer is decoding it and can still change the words |
| Plain, `·` | The recognizer is finished, and the cleanup pass can still re-punctuate or re-join it |
| Plain, no mark | The text is finished, and only you move it now |

## Commands

| Command | Effect |
|---|---|
| `Luna, delete` | Drops the last sentence in the transcript |
| `Luna, discard` | Drops the last sentence you said |
| `Luna, keep` | Files the text now instead of waiting out the settle |
| `Luna, clear` | Throws away the whole transcript |
| `Luna, copy` | Runs the cleanup pass to the end, and then puts the transcript on the clipboard as prose |
| `Luna, undo` | Puts back what the last delete, discard, or clear took |

The comma is optional, and `Luna deletes` is the same command. Don't put anything between the name and the verb. To rename the assistant, use `--assistant`.

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
