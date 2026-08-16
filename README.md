# speech-to-text-cli

Local real-time speech-to-text for Apple Silicon. Text appears **while you
speak** and firms up as the model gains context — provisional words render dim,
committed words render normally.

Built around [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), chosen
for accuracy on **accented and non-native English** (16.07% WER across 16 accent
groups, vs Whisper large-v3's 21.30%). Nothing leaves the machine.

## How it works

Qwen3-ASR is an offline model — its native streaming mode requires vLLM, which
requires CUDA. So streaming behaviour is synthesised: the model re-runs on a
growing audio buffer every 500ms, and a word is committed once **three
consecutive hypotheses agree** on it (LocalAgreement-3). Voice activity
detection gates inference away from silence and marks utterance ends, which is
what commits the final word of a sentence.

Rust owns capture, VAD, windowing, the transcript state machine and rendering.
A Python/MLX sidecar owns only the forward pass, over newline-delimited JSON on
stdio.

## Talking to it

The transcript is a **live document**, not a log scrolling past. It takes over
the screen while you dictate and is written to your terminal's scrollback in
full when you exit. That is what makes it editable by voice — you can throw away
a sentence you already kept, which would be impossible if it had already been
printed.

```
  This sentence is kept.
  So is this one.
│ This one has settled, but you haven't kept it yet.
│ and this part is still being decoded

  listening · 2 kept, 1 pending · Luna: commit reject clear copy undo
```

Three states, and the rendering is the whole signal:

| how it looks | what it means |
|---|---|
| dim | still being decoded — this can change on its own |
| plain, `│` in the margin | settled; nothing will change it but you |
| plain, no margin | you kept it |

Five spoken commands, all addressed to the assistant by name:

```
"Luna, commit"     keep everything settled so far
"Luna, reject"     drop the last sentence — including one you already kept
"Luna, clear"      throw away everything
"Luna, copy"       put the transcript on the clipboard, and keep it
"Luna, undo"       put back whatever the last reject or clear removed
                   ("Luna, rollback" is the same command)
```

The comma is optional; "Luna commit" works identically, and so does a verb the
model conjugated on you — "Luna rejects" is the same as "Luna reject", which
matters because that substitution is one the model really makes. Rename the
assistant with `--assistant Jarvis` if you'd rather; pick a name you don't
otherwise dictate, since "Luna rejects the offer." is a reject.

`copy` takes the whole transcript and marks it kept, on the reasoning that
choosing to copy something is how you approve it. It stops at text that has
settled — anything still being decoded is left alone, because that is the part
the pipeline might still rewrite.

Nothing you can say is unrecoverable: `reject` and `clear` are both undone by
`undo`, and repeating `undo` walks back a run of them. To remove two sentences
on purpose, leave a beat between the two rejects — **a `reject` or `clear`
repeated within 3 seconds is ignored**, because you can't see the first one land
until you stop speaking, and the usual reason people repeat a command is that
they think it wasn't heard.

Commands are recognised **only** as the wake word immediately followed by the
verb, and they take effect the moment you stop speaking rather than waiting out
the settle. That strictness is deliberate: missing a command costs you one
repetition, but a false one deletes something you meant to keep. All of these
stay dictation:

```
Luna went to the store.
I will commit the change tomorrow.
Luna, please commit          ← nothing may come between the name and the verb
```

Nothing is lost if you never say `commit` — everything you dictated is printed
on exit either way, and a count of what you never confirmed goes to stderr.

## Where your text goes

Two places, and only one of them is a file.

**stdout, on exit.** The whole transcript, one sentence per line. Redirect it if
you want it somewhere (`> notes.txt`), or it lives in your terminal's scrollback.

**A session file, as you speak.** Written to
`~/.local/state/speech-to-text-cli/` on every change. This is what makes `clear`
and `reject` survivable, and it's why killing the process, a panic, or closing
the window can't take your dictation with them.

That file is **deleted on a clean exit** — by then stdout has the transcript and
the file is just litter. Pass `--persist` to keep it; the path is printed on the
way out. An unclean exit leaves it behind either way, which is the whole point:
the file survives exactly the cases that would otherwise lose your text.

```sh
./target/release/speech-to-text-cli --persist
# ...
# transcript saved to ~/.local/state/speech-to-text-cli/session-20260816-124445.txt
```

### Picking a session back up

If a session ends badly, the next launch tells you:

```
note: a previous session is on disk (14 sentence(s)) — pass --resume to continue it: …
```

`--resume` recovers the most recent one and carries on dictating into it — same
file, so you don't collect a stale copy per crash. `--resume <path>` resumes a
specific transcript instead, and in that case the file you named is only ever
*read*: a new session file is opened beside it, so a clean exit can't delete
something you pointed at.

```sh
./target/release/speech-to-text-cli --resume            # continue where you stopped
./target/release/speech-to-text-cli --resume notes.txt  # continue a specific file
```

Recovered text comes back **kept**, so `Luna, copy` works immediately and you
can keep dictating onto the end of it. If there's nothing to resume the command
fails outright rather than starting empty — dictating into a fresh session
believing it's the old one is the kind of thing you'd only notice at the end.

It restores *text*, not session state: the undo history doesn't survive, so a
resumed session can't take back what the previous one did.

## Pausing mid-thought

Silence alone cannot tell "I'm finished" from "I'm thinking" — both are just
silence. So a pause is treated as a full stop only *provisionally*: text settles
after **six seconds** of silence, and if you resume before that, the audio is
merged and re-decoded as one utterance.

The model decides whether that is one thought or two:

```
"So here is the thing I wanted to"  +  "say, and that's why it matters."
  → So here is the thing I wanted to say, and that's why it matters.

"This is the first complete sentence."  +  "And this is a totally separate one."
  → This is the first complete sentence.
    And this is a totally separate one.
```

Merging happens on the **audio**, never on the text — the model needs the
prosody to get punctuation and capitalisation right across the seam.

Because a paused sentence can still be rewritten, it stays **dim** until the six
seconds are up. Dim always means "this can still change on its own" — including
across a pause, where the sentence looks finished but the pipeline is not done
with it.

The merged audio carries the **real pause length**, so the model judges the
boundary on how long you actually stopped. Verified from both sides:

```
4s pause → "This is the first thought, and this continues it."
8s pause → "This is the first thought."
           "This is a separate thought."
```

`--continue-ms` is one number doing two jobs: how long text stays dim (shorter
feels more responsive) and how long a thinking pause can run before one thought
becomes two sentences (longer is more forgiving). Six seconds splits it.

Measured on an M3 Max: **~215ms** inference on a 6s buffer (28× realtime), and
**72% duty cycle** sustained over a 42-second unbroken passage — about 1.2s for
a word to stabilise on a typical buffer.

## Talking without stopping

Nothing cuts you off. There is no maximum utterance length and no hard cut.

Because the settle window is six seconds, nearly every natural pause folds back into
the same audio buffer, so a long stretch of dictation is genuinely one growing
window. Past `--trim-after-s` (30s) the buffer is trimmed — but **only at a
pause you already left**, chosen as the longest silence available, and if you
left none then nothing is cut at all. A word is never sliced.

That bound is not arbitrary. Inference and lag both scale with buffer length:

| buffer | inference | time for a word to stabilise |
|---|---|---|
| 20s | 0.6s | ~2.2s |
| 60s | ~2.1s | ~7s |
| 600s | ~20s | **~70s** |

An untrimmed ten-minute monologue would still transcribe *correctly* — just a
minute behind you, which is useless when the point is reading along as you
speak.

## Fixing what you just said

Stumble over a word and just say it again. There is nothing to learn and no
command to remember — the false start is dropped and your correction kept:

```
"We should rewrite the regularing… regular expression parser."
  → We should rewrite the regular expression parser.
```

The model transcribes the stutter verbatim, so this happens downstream of it.
It works mid-breath, and it reaches words that are already on screen in plain
text — you'll see the false start dim and disappear as you say the correction.

A hesitation noise in between is fine: "the regularing, uh, regular expression"
works too.

### When it fires, and when it doesn't

A correction is only recognised when one word is a **strict prefix** of the
other — a second attempt at the *same* word. That is a deliberately narrow
rule, because the alternative is mangling ordinary speech. All of these are
left exactly as spoken:

```
These changes change the behavior of the parser.
The tests test the parser thoroughly.
He was at work, working late on the report.
The main maintenance window is on Sunday.
Take the form from the office.
```

A near-miss is always treated as dictation. Deleting a word you meant to keep
is worse than leaving a stutter for you to edit later.

**What it cannot do** is swap one word for a different one. "Let's meet on
Tuesday… Wednesday" keeps both, because the two words have nothing in common to
match on. Detecting that needs either a wake word or a model that understands
what you meant, and neither is here.

Repeating a word exactly — "the the" — is also left alone on purpose: "I had
had enough" is a real sentence and there is no way to tell them apart.

See [CLAUDE.md](CLAUDE.md) for the full decision record and the alternatives
that were rejected.

## Setup

Requires Apple Silicon, Rust, and Python 3.10+.

```sh
python3 -m venv .venv
.venv/bin/pip install "mlx-audio[stt]"
cargo build --release
```

The model (~1.8GB) downloads on first run. Pass `--model
mlx-community/Qwen3-ASR-0.6B-8bit` for a ~600MB model at roughly half the
inference cost, and about half a WER point worse on accents.

## Usage

```sh
# Live microphone
./target/release/speech-to-text-cli --language en

# Replay a 16kHz mono WAV through the same pipeline
./target/release/speech-to-text-cli --simulate audio.wav --language en
```

Omit `--language` to auto-detect across 52 languages.

### Tuning

| Flag | Default | Effect |
|---|---|---|
| `--assistant` | `Luna` | What to call it when giving it an instruction |
| `--agreement` | 3 | Hypotheses that must agree before text stops being dim. Lower = faster, jitterier |
| `--interval-ms` | 500 | *Shortest* gap between re-transcriptions. Stretches automatically if inference on a long buffer outgrows it |
| `--endpoint-ms` | 600 | Silence that ends an utterance |
| `--open-ms` | 150 | How long a sound must last to count as speech. Raise it if your keyboard gets transcribed |
| `--continue-ms` | 6000 | Silence before text settles. Resume before this and it is re-decoded as one utterance |
| `--persist` | off | Keep the session file on exit instead of deleting it |
| `--resume [PATH]` | off | Continue a previous session; bare, it recovers the most recent one |
| `--trim-after-s` | 30 | Trim the buffer past this, at a pause you already left. Never cuts mid-word |
| `--rms-floor` | -40 | Silence floor in dBFS. Raise it if a quiet room produces stray lines |
| `--quiet` | off | Suppress the "refresh slowed" notice |
| `--model` | `mlx-community/Qwen3-ASR-1.7B-8bit` | Any MLX Qwen3-ASR repo |

`--agreement 2` reaches ~0.7s to stable, at the cost of more visible revision.

### Stray "Okay." or "Oh." when you aren't speaking

Your room is above the silence floor. The model is not inventing these — it is
being handed non-speech audio and transcribing it, which is the only thing it
can do. The floor rejects audio that is *quiet*, not audio that is *non-speech*,
so a desk fan at −38 dBFS clears the default −40 and the detector fires on it.
Measured, from a file containing no speech at all:

| 20s of | default | `--rms-floor -30` |
|---|---|---|
| noise at −48 dBFS | *(nothing)* | *(nothing)* |
| noise at −38 dBFS | `Oh.` | *(nothing)* |

```sh
./target/release/speech-to-text-cli --rms-floor -30
```

Set it above your room and below your voice. Too high and quiet speech is
dropped; start at -30 and go from there.

### Your keyboard is being transcribed

Raising the floor is the wrong first move here, and it is worth knowing why: a
keystroke is *louder* than the room, so any floor that rejects one also rejects
a quiet voice. Level is the wrong axis. Duration is the right one — a keystroke
is 3 to 6 frames of sound where the shortest syllable is a dozen:

```sh
./target/release/speech-to-text-cli --open-ms 250
```

The onset delay this adds costs you nothing: 300 ms of audio is kept ahead of
every utterance, so the word that opened it is still transcribed in full. Going
past 300 ms starts clipping first words, so that is the ceiling.

This gates *isolated* noise — a keystroke, a mouse click, a creaking chair —
which is the kind that produces stray lines, because a lone click opens a window
holding almost nothing and a near-empty window is what makes the model guess.
Continuous fast typing is not gated by any duration setting, since it reads as
continuous sound; there, `--rms-floor` is the only knob, and it only helps if
your voice is louder than your keyboard. A headset or any mic that is not
sitting on the desk fixes it properly.

## Limitations

- macOS prompts for mic permission on first run.
- Accuracy figures come from Qwen's own benchmark and are not independently
  replicated.
- **The live view has no scrollback of its own.** A document taller than your
  terminal scrolls older sentences out of sight. They are still there, still
  rejectable and still printed on exit — you just can't look at them.
- **Undo only reaches text removal.** There is no way to un-keep a sentence you
  committed, short of rejecting it and saying it again. `Luna, copy` keeps what
  it copies, so that applies to a copy too.
- **Continuous background sound is not gated.** `--open-ms` rejects isolated
  noise and `--rms-floor` rejects quiet noise; sustained speech-like sound that
  is louder than your room — fast typing, a TV, a conversation across the room —
  passes both, and the model transcribes it, because from here that is what it
  is. Gating is inherently imperfect and a better microphone position beats any
  flag.
- **Resume restores text, not session state.** Undo history doesn't survive, so
  a resumed session can't take back what the previous one did.
- `--resume` only ever offers the *newest* session. An older one is still
  reachable by path, but nothing tells you it's there.
- Session files pile up under `--persist`; nothing prunes them.
- `Luna, copy` uses `pbcopy`, so it is macOS-only like the rest of the pipeline.
- Piped or redirected output shows nothing until the process exits, then prints
  the whole transcript. The text is identical to what a terminal shows.
- Self-repair only reaches words in the utterance you're still speaking. After
  a pause it usually still works, because the audio is merged and re-decoded as
  one — but once the settle window closes, the false start is out of reach.
- Correcting a word to a completely different one is not detected. `Luna,
  reject` won't help either: it works on whole sentences, not words.
- A four-letter false start ("conf… configure") is not caught. The threshold
  that would catch it also catches "just justify".
- If you truly never pause, the buffer keeps growing and the transcript falls
  further behind. That is the deliberate trade — latency comes back when you
  stop, a mangled word never does.
