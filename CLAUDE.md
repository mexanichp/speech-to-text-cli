# speech-to-text-cli

Real-time speech-to-text CLI in Rust. Local-only inference, live provisional
text that stabilizes as you speak, targeting strong accuracy on accented and
non-native English. Apple Silicon, macOS.

**This file is the decision record.** It holds the choices that were made, the
invariants that hold the pipeline together, and the things that cost a
measurement to learn. It is not a description of the code.

**The source is authoritative.** Where this file disagrees with the code, the
code is right and this file is the bug.

Nothing here should need editing when a constant, a default or a test count
changes. If a sentence here goes stale for that reason it was too specific:
name the constant instead of restating its value, and state the finding instead
of the number that produced it.

---

## 1. Product intent

Live dictation where the speaker reads along while talking and can verify
capture in real time. Text appears provisionally, then files into a document.

Provisional text is a feature, not noise. It is the correctness signal the
interaction is built on: the speaker sees an error the moment it happens, and
can say `Luna, delete` rather than discovering it afterwards.

Text moves dim to plain exactly once, in one direction, and never back.

---

## 2. Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | **Qwen3-ASR** as the ASR model | Best measured accuracy on accented English at a cost this class of machine can sustain |
| D2 | **MLX** runtime | Only mature local runtime for this model on macOS |
| D3 | **Python sidecar**, NDJSON over stdio | MLX is Python/Swift-only. Rust owns the pipeline, Python owns the forward pass |
| D4 | **Sliding window + LocalAgreement-n** | Produces provisional text while speaking; native streaming is vLLM-only |
| D5 | **VAD for gating and segmentation** | Suppresses hallucination on silence, supplies the endpoint |
| D6 | **Commit latency is the budget**, and it is reported to the speaker when exceeded | It is what the speaker sees. Duty cycle is pinned by construction and says nothing |
| D7 | **The transcript is a live document, not scrollback** | `delete` must reach a sentence filed minutes ago, and scrollback cannot be taken back |
| D8 | **Trim the buffer at a pause, never mid-word, and only where the text is filed** | An utterance is never truncated; what is bounded is how far behind the transcript falls |
| D9 | **Wake word for document operations only** | "throw that away" has no acoustic signal; a self-repair does |
| D10 | **Autosave every change; delete only on a clean exit** | The paths that lose the transcript cannot run cleanup, so the file survives them |
| D11 | **Commitment is logical, not timed** | A sentence files once further sentences sit behind it unrevised |

### D1: why Qwen3-ASR

Primary criterion is accent robustness across native and non-native English.

Dialog-Accented English, averaged over 16 accent groups
([Qwen3-ASR Technical Report](https://arxiv.org/html/2601.21337v1)):

| Model | WER |
|---|---|
| **Qwen3-ASR-1.7B** | **16.07%** |
| Qwen3-ASR-0.6B | 16.62% |
| FunASR-MLT-Nano | 19.96% |
| Whisper-large-v3 | 21.30% |
| Gemini-2.5-Pro | 23.85% |
| GPT-4o-Transcribe | 28.56% |

Qwen's own internal benchmark, not independently replicated. Treat the ordering
as more reliable than the values. Also Apache 2.0, 52 languages.

1.7B costs roughly twice 0.6B rather than the order of magnitude first assumed,
which is what makes the better accuracy affordable. 0.6B stays supported
through `--model` for slower hosts.

### Rejected alternatives

- **NVIDIA Parakeet / Canary**: best streaming architecture, but English-native
  biased. NVIDIA's own cards say it "may underperform on heavily accented
  speech". Disqualified on the primary criterion. Also NVIDIA Open Model
  License, no ONNX export, and sherpa-onnx lacks true streaming RNNT
  ([issue #3573](https://github.com/k2-fsa/sherpa-onnx/issues/3573)).
- **Whisper large-v3 / turbo**: several points worse on accented speech.
  `whisper-rs` would have been the cleanest Rust path.
- **Streaming Zipformer via sherpa-rs**: the only true streaming that works in
  Rust today, accuracy well below both. Would win at a sub-500ms budget.
- **Qwen3-ASR native streaming (vLLM)**: vLLM requires CUDA.
- **qfuxa/qwen3-asr-0.6b-streaming**: causal encoder fine-tune, worse live WER
  and worse commit latency than sliding-windowing the base model.

---

## 3. Architecture

```
┌─────────────────────────── Rust ────────────────────────────┐
│  cpal (mic, device rate)                                    │
│      └→ rubato resample → 16kHz mono f32                    │
│          └→ VAD: gate, endpoint, silence runs               │
│              └→ audio buffer + tick scheduler               │
│                  └→ buffer trim (cut at a filed boundary)   │
│                      └────────┐                             │
│                               ↓ NDJSON / stdin              │
└───────────────────────────────┼─────────────────────────────┘
                                ↓
┌────────────────── Python sidecar (MLX) ─────────────────────┐
│  Qwen3-ASR → hypothesis (no prompt; hint only on ask)       │
└───────────────────────────────┬─────────────────────────────┘
                                ↓ NDJSON / stdout
┌───────────────────────────────┼─────────────────────────────┐
│  self-repair → LocalAgreement-n window                      │
│      └→ logical commitment → document (filed sentences)     │
│          └→ spoken commands (endpoint only)                 │
│              └→ autosave + alternate-screen live view       │
│                  └→ real scrollback, once, on exit          │
└─────────────────────────────────────────────────────────────┘
```

Rust does everything except the forward pass. The transcript state machine is
the actual engineering: no library ships it, because the commit policy is
application-specific.

---

## 4. Layout

```
src/main.rs        orchestration: tick loop, trim bands, endpoint handling,
                   `Settle` (adaptive hold), `Dip` (cut candidates), commands
src/audio.rs       cpal capture, downmix, resample to 16kHz; --simulate path
src/vad.rs         earshot + RMS floor, open/close hysteresis, two silence runs
src/transcript.rs  LocalAgreement window, the sentence document, logical
                   commitment, the filed-text overlap test, undo
src/text.rs        sentence splitting, opener veto, continuation deferral
src/repair.rs      self-repair detection ("regularing… regular")
src/command.rs     wake-word command parsing and the vocabulary hint
src/store.rs       session autosave, resume, clipboard
src/trace.rs       filing-path trace, off unless STT_TRACE names a file
src/render.rs      alternate-screen live document, wrap-safe row layout
src/sidecar.rs     subprocess handle + NDJSON protocol
sidecar/asr_sidecar.py   MLX inference; stdout is protocol, stderr is logs
```

Every module carries its own tests. `cargo test` is the gate.

---

## 5. How text reaches the document

Two layers, and keeping them distinct is the whole design.

**The window** is LocalAgreement: a word is agreed once it appears in the
longest common prefix of the last `n` hypotheses. Agreed is a claim about the
*pipeline*.

**The document** is a `Vec<Sentence>`: what the speaker has finished saying. A
sentence there is plain on screen and only the speaker can change it.

### Filing paths

Each is tagged in the `STT_TRACE` output. Every one of them files text that was
decoded with full right-context, which is the property to preserve when adding
another.

| trace tag | when | decoded from |
|---|---|---|
| `logical` | enough complete sentences sit behind it | the whole buffer |
| `endpoint-settled` | the settled prefix of a finalized utterance | the whole utterance |
| `settle` | the hold expired with no continuation | the whole utterance |
| `endpoint` | the settle is disabled, or the utterance is empty | the whole utterance |
| `command` | dictation segments of an utterance carrying an instruction | the whole utterance |
| `stale` | a pending utterance was displaced (defensive, unreachable) | the whole utterance |
| `eos-held` / `eos` | end of stream | the whole buffer |
| `trim-forced` | the forced band, where the cut happens regardless | a whole-buffer decode, located by the head |

`Transcript::settled_words` is the rule: file everything except the last
`KEEP_SENTENCES` complete sentences, plus one more if the boundary reads as a
continued clause (`text::opens_a_continuation`), plus one more if the window
does not end on terminal punctuation.

Commitment declines while `command::names` matches the prefix about to be
filed, so a wake word is never filed as dictation before the endpoint reads it
as an instruction.

### The buffer trim

`cut_point` answers *where* the audio may be cut. The bar falls as the buffer
grows, because holding out for a sentence boundary is affordable only while
latency is not already suffering:

| band | eligible gaps |
|---|---|
| below the eager band | none |
| eager band | `SENTENCE_DIP_FRAMES`, a sentence boundary |
| past the threshold | `DIP_FRAMES`, a clause |
| past the desperate multiple | any recorded silence, down to `MIN_DIP_FRAMES` |
| nothing eligible, past the forced multiple | `last_resort`: any gap at all with refusals ignored, else the quietest frame in the buffer |

Longest gap wins; latest breaks a tie. Gaps come from `Vad::quiet_run`, which
is reset by any voiced frame, not from `silence_run`, which survives a click.

`Transcript::spent_by` answers *whether*. It cuts only when the head decode is
entirely filed **and** the cut lands exactly on a filed sentence end. The
second condition is what the first cannot supply: a head decoding to
`It tries to solve the problem.` against a filed `It tries to solve the problem
of coordination.` passes the first and is precisely the cut that must not be
made, because the rest of that sentence stays in the buffer and later decodes
as though it stood alone.

A refusal is remembered on the gap (`Dip::refused_until`), keyed on
`Transcript::filed_words` rather than on the document revision, since the answer
cannot change until at least the stranded words have been filed. A refusal takes
every later gap with it, because a later cut severs the same sentence and more
of it. A refused probe costs a forward pass, charged to the next tick as
`probe_debt`: it delays the next pass rather than stealing time from it, and
shows up in the reported commit latency.

In the forced band the cut happens regardless. Filing the severed head is what
used to damage the transcript, so the backstop files from a whole-buffer decode
instead, using `words_covering` to locate the cut inside it.

### The settle

A pause is only a full stop in hindsight. On endpoint the audio is retained; if
speech resumes within the hold it is merged and the whole thing re-decoded as
one utterance.

**Merge at the audio level, never at the text level.** Splicing two
independently decoded strings cannot produce correct punctuation or
capitalisation; handing the model the merged audio can, because it sees the
prosody.

`Settle` adapts: a decaying maximum of the silences the speaker went on talking
through, with a margin, clamped between `--continue-ms` and `--continue-max-ms`.
A single pause moves it one step rather than straight to the ceiling, so growth
needs a habit and a one-off interruption washes out. It never reads the text,
only the speaker's pause habit. A zero floor switches adaptation off, since that
is an explicit instruction not to hold.

Consequence: every ordinary pause merges, so a session merges continuously and
the buffer never resets on its own. That is what makes the trim load-bearing.

### Spoken commands

Wake word immediately before the verb, nothing in between. A third-person `-s`
is the same command, because the speaker said the verb and the recogniser
conjugated it. Past and progressive forms stay dictation.

| verb | scope |
|---|---|
| `delete` | the newest sentence in the document, however old |
| `discard` | the newest sentence **this utterance** filed, and nothing else |
| `keep` | files what is in flight and ends the settle, moving no text |
| `clear` | the whole document |
| `copy` | the whole document, to the system clipboard |
| `undo` / `rollback` | the last removal |

Every verb changes the transcript. That is the entry requirement: a verb that
moves no text cannot help the speaker and can still fire by accident, so it is
pure downside on both axes that matter, the vocabulary hint and the
false-command surface.

`discard` cannot walk backwards into settled text however many times it is
said: `main.rs::apply` maintains the count and `Transcript::discard_last` takes
it as an argument. Saying it with nothing in flight removes nothing rather than
falling back on the newest sentence. Silently widening the scope of a
destructive command to whatever happens to be nearest is what this module
refuses to do.

`keep` is the answer to the pipeline's own commonest warning. When commit
latency exceeds its target the notice says to say it, because it drops the held
audio and starts the next utterance short. Pausing does not do this: an endpoint
holds the audio and a continuation merges it back.

Detection runs on finalized utterances only. `repair()` is pure and safe
per-hypothesis; commands destroy text, and the window re-decodes the same audio
every tick, so per-hypothesis detection would empty the document one sentence
per tick.

A command usually follows a pause, so the merge puts both halves in one buffer
and the model punctuates them as running on: `This is my first sentence, Luna,
delete.` `text::close_sentence` fixes the last word of dictation *terminated by*
a command, which is not a guess: the words that followed were provably not part
of the sentence. Text appearing after a command is left alone, since nothing
establishes that it ended.

### Self-repair

`repair.rs` drops a false start when the speaker says the word again. The rule
is strict prefix containment, with different stem thresholds each way:
backtrack ("regularing" then "regular") is safe with a short stem once `-s`/`-es`
is excluded, since English essentially never puts a word before its own prefix.
Extension ("conf" then "configure") needs a longer stem and a non-inflectional
ending, because English does put a word before its own derivation ("at work
working late").

It reaches text the window has already agreed, so `push_hypothesis` cuts
`committed` back to where it still agrees with the newest hypothesis. Nothing
else in the state machine ever shortens it, and without that the replaced word
survives forever.

### Persistence

The document plus the in-flight text is written to the state directory on every
change, via a temporary file and a rename. Including the in-flight text is what
makes recovery independent of when text files: a speaker who never pauses
reaches the document late.

A clean exit removes the file unless `--persist`, since stdout has the
transcript by then. An unclean exit cannot remove it. That asymmetry is the
design: the file survives exactly the cases that lose text.

Bare `--resume` takes the most recently *written* file by mtime and **adopts it
in place**, so a clean exit without `--persist` deletes the file it just
recovered. `--resume <path>` treats the file as read-only and opens a new one
beside it, since adopting it would let a clean exit delete something the user
pointed us at. Two ordering requirements, both found by tests: the "a previous
session is on disk" hint is computed before this session's file exists, or it
finds our own; and the resume load happens before anything creates a file, or a
mistyped path leaves an empty file that becomes the newest and poisons the next
bare `--resume`.

---

## 6. Tuning

Every flag is documented on `Args` in `main.rs`, and the defaults live there.
The ones whose behaviour is not obvious from the name:

| flag | behaviour |
|---|---|
| `--interval-ms` | a **floor**, not a period. The next tick stretches with inference so the duty cycle cannot exceed 100% |
| `--agreement` | commit latency is roughly `(n-1) x tick + inference` |
| `--continue-ms` | the settle floor. Raising it is close to free on ordinary dictation, where every pause is already shorter |
| `--continue-max-ms` | ceiling on settle adaptation. Set equal to the floor to pin the hold |
| `--trim-after-s` | when to start *looking* for a cut, not where to cut. See below |
| `--open-ms` | duration, not level: this is the keyboard defence. Past the pre-roll it clips first words |
| `--rms-floor` | dBFS. Set above the room and below the voice. Needs `allow_negative_numbers` on the clap command, or the negative value parses as a flag |

**`KEEP_SENTENCES` is what actually sets the buffer length, not
`--trim-after-s`.** The trim may only cut where a filed sentence ended, so the
buffer can never be shorter than what commitment is still holding back. Swept
across the range where the endorsement test still functions, the flag does not
move the buffer at all; it does something only when it is low enough that every
trim is forced, which bypasses the endorsement test and damages the transcript.
The program warns below a floor for that reason.

**Two things scale with the buffer and one of them is the one to watch.**
Inference grows sub-linearly with buffer length, because encoder cost rises
while decode, driven by transcript length, dominates. Duty cycle is pinned by
the tick stretch and therefore cannot report anything. Commit latency grows on
both terms, is what the speaker sees, and is what the notice reports.

**Benchmark with speech, not silence.** Silence produces no tokens to decode
and flatters the model badly.

Protocol overhead is the whole buffer re-encoded as base64 JSON every tick, and
it is single-digit milliseconds against a forward pass of hundreds. Do not
optimise the wire format; it was measured and it is noise.

---

## 7. Invariants

Break one of these and the failure is usually silent.

1. **Nothing writes to the terminal except `Renderer`.** A bare `eprintln!`
   while the live view is up corrupts the frame. Off-thread diagnostics go
   through notice channels; the sidecar's stderr is piped, never inherited;
   `STT_TRACE` writes to a file only.
2. **Every emitted row is strictly narrower than the terminal**, gutter and
   decoration included. At exactly the last column terminals disagree about
   whether the cursor advanced, which makes any row arithmetic off by one.
3. **The per-request protocol carries audio and a boolean.** There is no
   free-text field, so transcript text cannot reach the model by any path. See
   the prompt echo in §8.
4. **Nothing reaches the scrollback until the session ends.** Dim means the
   pipeline may rewrite this; plain means only the speaker can. The live view
   and the piped path produce identical text.
5. **Commands are parsed from finalized utterances only**, and the audio that
   carried one is dropped, never retained for continuation. Retention is the
   only thing that could replay a command.
6. **`scan()` stops at an endpoint.** One loop iteration can cover more audio
   than the endpoint threshold, so a single batch can hold one utterance ending
   and the next beginning. Running past it reports the onset before the caller
   has a pending utterance to merge into, and leaves the detector open so no
   further onset ever arrives: the held utterance becomes unreachable, not
   merely late.
7. **The main loop blocks for one chunk and then drains the rest.** Several
   passes run off the tick and can land back to back; taking one chunk per
   iteration left the loop behind when the next pass started, and dropped
   capture buffers do not degrade gracefully, they shred the audio.
8. **Every hypothesis in the agreement deque was stripped against the same
   `filed`.** Filing mid-window therefore clears the deque, or the comparison
   is between sequences that no longer begin at the same word.
9. **Filing text is not the same operation as cutting audio.** Filing removes
   nothing: the audio stays and the model keeps re-decoding it. Cutting removes
   context from every future decode. Conflating them is what made early
   commitment look impossible.
10. **A decode of severed audio is not evidence about a sentence boundary.**
    The model punctuates whatever it is handed, so a severed head comes back
    with a full stop and a severed tail comes back capitalised. Both confident,
    both wrong, neither detectable from the text. The test is therefore "is this
    text already in the document?", never anything about how the text looks.

---

## 8. Measured, do not re-research

### About the model

- Qwen3-ASR is **AED**, 8x downsampling, **12.5 Hz token rate** (80ms/token).
- The AuT encoder has a dynamic attention window, so streaming is trained into
  the weights. Unimplemented outside vLLM: an MLX port is engineering, not
  research.
- Open weights have **no context biasing / hotwords**. That is Qwen3-ASR-Flash,
  the API-only model.
- Timestamps require the separate Qwen3-ForcedAligner.
- Audio contract everywhere: **16 kHz, mono, f32**. Microphones do not deliver
  it, so resampling is mandatory.
- Model formats are containers; a runtime must implement the *architecture*.
  Converting to GGUF is meaningless, since whisper.cpp only knows Whisper.
- `generate` defaults to a token cap of several minutes of output for any
  window, so a decoder caught in a repetition loop does not fail, it runs to the
  cap. The sidecar budgets against the audio token rate instead, which never
  binds on real speech and cuts the worst case by an order of magnitude.

### The transcript prompt echo

`mlx_audio` drops `system_prompt` verbatim into the system turn ahead of the
audio. Feeding the transcript back as context made the model **copy it out
verbatim** on any window holding no speech, and it self-reinforced: the echo was
committed, joined the history, and returned in the next prompt. It bought
nothing, byte-identical output on real speech with and without.

A **static** prompt is not this bug. The reasoning turns on the prompt being
derived from what the speaker said. The fixed command list produces
byte-identical output on dictation, silence and noise, and it fixes a verb the
recogniser otherwise splits into two words, which the parser correctly refuses
with no way for the speaker to see why.

It does bias in one measured direction: it pulls ambiguous audio toward the wake
word (`Moon a` to `Luna.`, `Lunar respect` to `Luna respect.`). It never
invented a verb across deliberate near-misses. So the hint runs only on a
finalized utterance that did not already parse as a command, and the hinted
reading replaces the unprompted one **only if it parses as a command**, which
means every word the speaker keeps comes from the unprompted pass.

Gating that retry on "did the transcript name the assistant?" was tried and
removed: it assumes the recogniser spelled the wake word correctly, which is
exactly what fails when the hint is needed. A fuzzier gate only moves the
boundary and adds a second thing to be wrong about.

### Signals that do not work

- **Terminal punctuation is not a completeness signal.** The model punctuates
  fragments as though they stood alone. On the input where a completeness test
  would have to work, every fragment ended in a full stop. Any future attempt to
  decide "has this thought finished?" from one utterance's text should start by
  explaining what it does differently.
- **Seam length does not decide a sentence boundary.** Swept across audio
  conditions and seam lengths from zero to several seconds, the boundary never
  moved once, including a two-second gap read as a comma. The boundary is
  prosodic and semantic. Anyone reaching for the merge gap to fix a split
  sentence should read §5 first. Caveat: this was synthesized speech, whose
  prosody is unambiguous by construction. It rules out a strong dependence, not
  a marginal one.
- **Punctuation does not mark a self-repair.** The repair seam came back clean
  while an innocent control (`He was at work, working late`) got a comma. It is
  anti-correlated.
- **Edit distance cannot detect a self-repair.** `form`/`from`,
  `quiet`/`quite`, `trial`/`trail` are one or two edits apart and all appear in
  ordinary sentences. The rule is strict prefix containment only.

### Failure modes defended against

- **Hallucination on silence.** An open mic in a quiet room produces confident
  boilerplate. The RMS floor ANDed with the detector fixes it; do not remove the
  floor, the detector alone is not sufficient. Know what it does not cover: it
  rejects audio that is *quiet*, not audio that is *non-speech*. An unremarkable
  room clears the default floor and gets transcribed, which is why `--rms-floor`
  exists.
- **A keyboard transcribed as speech.** Level is the wrong axis: a keystroke is
  louder than the room by construction, so any floor that rejects one also
  rejects a quiet voice. Duration is right, and the separation is wide: an
  isolated keystroke yields a handful of voiced frames where a syllable yields
  dozens. Sustained typing is different again, chaining into runs no open window
  can gate, and there the model itself declines the audio. Testing this needs
  room tone, not digital silence: the detector is stateful and adapts, and a
  buffer of exact zeros makes the test pass for the wrong reason.
- **An unbounded buffer.** A speaker who leaves no gap the VAD calls quiet, or a
  television in the room below the voice and above the floor, produces no
  endpoints, no trims and a buffer that grows for as long as the session runs.
  Every trim band is a filter over recorded gaps, and a filter over an empty
  list is empty. `last_resort` is the band below the bands. The transcript stays
  correct throughout, which is what makes this hard to notice: the session
  simply gets slower until it stops keeping up.
- **Protocol desync.** The sidecar `dup()`s the real stdout and points
  `sys.stdout` at stderr before the heavy imports, and every reply echoes the
  request id. Without both, one stray `print()` anywhere in the dependency tree
  would offset every later exchange by one, silently pairing each window with
  the previous window's hypothesis.
- **A refused window is not a dead session.** `Err` means the transport is gone
  and the session cannot continue; `Reply::Failed` means the model refused one
  buffer and the next pass will probably succeed. Failures are reported through
  `notice()` and are **not** suppressed by `--quiet`, which covers only the
  self-correcting latency notice.
- **Empty hypotheses poisoning LocalAgreement.** The sidecar returns `""` for
  any buffer below its minimum, and the first tick after onset is barely longer
  than the pre-roll. Admitting that empty word list drives the common prefix to
  zero for the next `n` ticks. An empty hypothesis is absence of evidence, not
  evidence of silence.
- **Abbreviations split as sentences.** `The U.S. government said so.` came out
  as two lines. Guards: a lone letter before the stop is an initial, and text
  resuming in lower case means the sentence is still going. Lower case rather
  than requiring upper, because caseless scripts must still split on `。`.
- **A benchmark harness that was itself the bottleneck.** `simulate` paced by
  sleeping *between* sends, and `sleep` guarantees at least its argument, so the
  loop accumulated send cost plus scheduler latency, which is large when the
  main thread is saturating the GPU. It made the pipeline look far slower than
  realtime while it was keeping up. It now paces against an absolute deadline.
  **Never measure throughput with a relative-sleep pacer**; the failure is
  silent and reads as "the system under test is slow", which is what you were
  looking for.

---

## 9. Reproducing things

Which path filed a line:

```sh
STT_TRACE=/tmp/s.trace ./target/release/speech-to-text-cli --simulate f.wav
```

Never add an `eprintln!` for this. See invariant 1.

Synthesizing test speech. `[[slnc <ms>]]` inserts a real pause, which is what
exercises the endpoint, the merge and the settle boundary:

```sh
say -v Samantha -o t.aiff "A sentence. [[slnc 1500]] Luna, delete."
afconvert -f WAVE -d LEI16@16000 -c 1 t.aiff t.wav
```

`say` will not produce a disfluency: "regularing regular" comes back as one
hyphenated token, so `repair.rs` cannot be exercised this way. A short gap is
not enough to make the model file two sentences from one merged utterance;
that needs two topically unrelated sentences and a few seconds of silence.

Terminal output. Redirecting stdout is not enough, since the renderer detects a
non-TTY and skips the live view:

```sh
script -q out.raw ./target/release/speech-to-text-cli --simulate f.wav
```

Frames are separated by `\x1b[1;1H` and every row is `\x1b[{r};1H\x1b[2K` plus
content, so a small screen emulator is enough. Worth asserting: the final
transcript appears **after** `\x1b[?1049l`, and no in-flight row is plain.

Reproducing a persistence question: synthesize a passage with a pause between
every sentence so every gap merges, run it under `--simulate --persist`, and
poll the session file. Reasoning about persistence without a repro has produced
two wrong diagnoses; the repro is cheap.

Dependency notes:

- `rubato` is pinned; 5.x reworked its API around `audioadapter`.
- `cpal` renamed `Device::name()` to `description()` in the version used here.
- `earshot` needs exactly **256-sample** frames at 16 kHz.

---

## 10. Known gaps

**Read every measurement in this file with the caveat attached.** Almost all of
it is synthesized speech, which has clean prosody and no genuine hesitation. It
is evidence about *mechanism*, whether a path fires, whether a cut lands,
whether a string splits, and not about how a human hesitating mid-thought comes
out. Real dictation is what several of these need.

- **`KEEP_SENTENCES` wants real dictation.** It is the one setting changed on a
  judgement call rather than to repair a defect, and it is what sets the buffer
  length.
- **The forced backstop can still file a fragment.** In that band the cut
  happens regardless; the whole-buffer decode usually places it, but when it
  cannot, the severed head is filed. Reachable on demand by setting
  `--trim-after-s` low enough that every trim is forced.
- **The forced path clears no `filed` state**, leaving the record describing
  audio that has been cut away. Contained today because `covered()` also tries
  the suffix alignment, but that is a fallback, not a fix.
- **`FILED_MEMORY` caps how long a head decode can be and still be endorsed.**
  Not reachable at ordinary buffer sizes; would be at a very large
  `--trim-after-s`.
- **Re-tokenisation now decides whether audio is discarded.** The trim's
  endorsement runs through the same overlap budget that dedupes filed text, so a
  word that comes back re-tokenised past the budget refuses a cut it should
  take. It fails safe, the buffer grows rather than text being lost, but the
  budget is doing two jobs and was sized for one.
- **The first pause past the settle floor is always lost.** A pause is only
  measurable once it has ended, so a speaker whose habit exceeds the floor pays
  one fragment before `Settle` adapts. Nothing short of predicting the pause
  fixes it.
- **`Settle` learns from silence, not from meaning.** A speaker who stops
  mid-topic and one who has finished are indistinguishable to it, so a long
  interruption stretches the hold for the next few utterances.
- **The trim seam is still a seam.** Cutting at the longest pause lands on a
  sentence boundary most of the time, and nothing re-punctuates across it the
  way the merge does.
- **Arbitrary substitutions are undetectable.** "Tuesday… Wednesday" shares no
  prefix, and `delete` does not recover it, since it operates on sentences
  rather than words. Nothing short of a wake word or a semantic model does.
- **Cross-utterance repair.** A repair only reaches words in the current window.
- **No scrollback in the live view.** The document can grow past the screen;
  older sentences stay deletable and are printed at exit, but there is no way to
  look at them.
- **Nothing watches the duty cycle.** The notice reports commit latency, which
  is the right quantity for the speaker, but it means nothing reports inference
  genuinely exceeding realtime. A gap in reporting rather than in behaviour.
- **Sentence splitting never breaks after a one-letter word**, so `The answer is
  A. It follows.` is one sentence. The initials guard is worth more than the
  case.
- **Repetition-loop detection** (the compression-ratio heuristic) is
  unimplemented. The token budget bounds the damage without detecting the loop.
- **Session files accumulate under `--persist`**, and `--resume` only ever
  offers the newest.
- **Resume restores text, not session state.** The undo stack and the audio are
  gone, so a resumed session cannot undo what the previous one did.
- **`copy` is macOS-only**, which matches D2 but would need replacing if the
  pipeline ever left Apple Silicon.
