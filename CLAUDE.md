# speech-to-text-cli

Real-time speech-to-text CLI in Rust. Local-only inference, live provisional
text that stabilizes as you speak, targeting strong accuracy on **accented and
non-native English**.

Host: Apple M3 Max, macOS. Rust 1.91-nightly, edition 2024.

---

## 1. Product intent

Live dictation where the user **reads along while speaking** and can verify
capture in real time. Text appears provisionally, then stabilizes. This enables
spoken correction ("revert the last sentence") because the user sees an error
the moment it happens rather than after the fact.

This is a deliberate product stance: provisional text is a feature, not noise.
It is the correctness signal the interaction is built on.

---

## 2. Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | **Qwen3-ASR-0.6B** as the ASR model | Best measured accuracy on accented English; small enough to sliding-window in realtime |
| D2 | **MLX** runtime (Apple Silicon GPU) | Only mature local runtime for this model on macOS; no CUDA available |
| D3 | **Python sidecar**, NDJSON over stdio | MLX is Python/Swift-only. Rust owns the pipeline, Python owns inference |
| D4 | **Sliding window + LocalAgreement-3** | Produces provisional/committed text *while speaking*; native streaming is vLLM-only |
| D5 | **VAD for gating + segmentation** | Suppresses hallucination on silence; supplies utterance-final commit trigger |
| D6 | Latency target: **≤2s to stable** | User-specified. Buys LocalAgreement n=3 |
| D7 | Rust implements only the pipeline | Capture, VAD, buffering, transcript state machine, rendering |
| D8 | **The transcript is a live document, not scrollback** | `reject` has to reach committed text, and scrollback is the one region a program cannot take back (§6) |
| D9 | **Trim at a pause, never mid-word** | An utterance is never truncated; what gets bounded is how far behind the transcript falls (§3) |
| D10 | **Wake word for document operations only** | "keep this" / "throw that away" have no acoustic signal at all, unlike a self-repair (§6) |
| D11 | **Autosave every change; delete only on a clean exit** | The paths that used to lose the transcript are exactly the ones that cannot run cleanup, so the file survives them for free (§6) |

### D1 — Why Qwen3-ASR-0.6B

Primary criterion is accent robustness across native *and* non-native English.

Dialog-Accented English, averaged over 16 accent groups
([Qwen3-ASR Technical Report](https://arxiv.org/html/2601.21337v1)):

| Model | WER |
|---|---|
| Qwen3-ASR-1.7B | 16.07% |
| **Qwen3-ASR-0.6B** | **16.62%** |
| FunASR-MLT-Nano | 19.96% |
| Doubao-ASR | 20.41% |
| Whisper-large-v3 | 21.30% |
| Gemini-2.5-Pro | 23.85% |
| GPT-4o-Transcribe | 28.56% |

Caveat: this is Qwen's **own internal benchmark**, not independently
replicated. Treat the ordering as more reliable than the absolute values. The
result is consistent with theory — multilingual training (52 languages, 22
Chinese dialects) regularizes for the L1 phonology carried in non-native speech.

Also: Apache 2.0, 52 languages, LibriSpeech clean 1.63% (1.7B) vs Whisper
large-v3's 1.51%, GigaSpeech 8.45% vs 9.76%.

**1.7B over 0.6B — reversed, on measurement.** This decision originally went the
other way, on the estimate that 1.7B ran at ~7× RTF and so could not sustain the
sliding window past ~7s of buffer. That estimate was wrong by about a factor of
two, in the same direction and for the same reason §3's published 0.6B figure
was wrong: it came from a source, not from this machine.

Measured here (M3 Max, `Qwen3-ASR-1.7B-8bit`, real speech, median of 3):

| buffer | 0.6B | 1.7B | 1.7B duty @500ms |
|---|---|---|---|
| 1s | 63ms | 98ms | 20% |
| 2s | 75ms | 120ms | 24% |
| 4s | 96ms | 170ms | 34% |
| 6s | 128ms | 215ms | 43% |
| 8s | 154ms | 284ms | 57% |
| 12s | 199ms | 398ms | 80% |
| 14s | — | 480ms | 96% |
| 16s | — | 506ms | **101% — over realtime** |
| 20s | 318ms | 624ms | 125% |

1.7B is **~2× the cost of 0.6B, not ~7×**, and it sustains the sliding window
up to a 16s buffer. Warm load is indistinguishable (~1.07s for both, measured in
fresh processes; the resident footprint is 2.3GB against ~0.7GB). Commit latency
at the default n=3 / 500ms is ~1.1–1.2s for typical buffers — inside D6's 2s
target with room to spare.

So the accuracy is now affordable: 16.07% vs 16.62% WER on accented English, the
primary criterion. 0.6B remains a supported `--model` for slower hosts.

The 16s crossover is handled by the **adaptive tick** (§3), not by cutting the
window short. Buffers now routinely run past it — the window is no longer capped
at all, only trimmed at pauses (§3) — and the measured duty cycle over a 42s
passage is 72%, so the crossover is real but well inside what the tick absorbs.

### Rejected alternatives

- **NVIDIA Parakeet / Canary** — best-in-class streaming architecture
  (unified-en-0.6b: 5.91% offline → 6.52% @560ms) but **English-native biased**.
  NVIDIA's own cards state it "may underperform on heavily accented speech… or
  non-native English speakers." Training lineage (LibriSpeech, Fisher, VCTK,
  VoxPopuli, Europarl) has documented gaps — no Indian-resident speakers in
  LibriSpeech/Switchboard/WSJ; VoxPopuli skews European L1. Disqualified on the
  primary criterion. Also: NVIDIA Open Model License, no ONNX export, and
  sherpa-onnx lacks true streaming RNNT support ([issue #3573](https://github.com/k2-fsa/sherpa-onnx/issues/3573), open).
- **Whisper large-v3 / turbo** — 21.30% on accents, 4.7 points worse than
  Qwen3-ASR-0.6B. `whisper-rs` would have been the cleanest Rust path; accuracy
  on the primary criterion overrode that.
- **Streaming Zipformer via sherpa-rs** — only *true* streaming that works in
  Rust today, but accuracy well below both. Would have been the pick at a
  sub-500ms budget.
- **Qwen3-ASR native streaming (vLLM)** — vLLM requires CUDA. `vllm-metal`
  exists but targets mlx-community LLMs, not this audio tower. No official Qwen
  response on a non-vLLM path.
- **qfuxa/qwen3-asr-0.6b-streaming** — causal encoder fine-tune, runs on MPS
  without vLLM, but 17.6% live WER and ~4.1s p50 commit latency. Worse than
  sliding-windowing the base model.
- **Existing local models** — MLX Whisper and faster-whisper weights in
  `~/.cache/huggingface`, plus superwhisper's Argmax models. The Whisper weights
  lose on accuracy; the Argmax CoreML models are proprietary and licensed to
  that app.

---

## 3. Latency budget

```
commit_latency ≈ (n − 1) × rerun_interval + inference_time
```

**Measured on this machine** (M3 Max, real speech, median of 3 runs per point).
The full 0.6B-vs-1.7B table is in D1; the default is now **1.7B**:

| buffer | 1.7B inference | RTF | duty @500ms |
|---|---|---|---|
| 1.0s | 98ms | 10.2× | 20% |
| 2.0s | 120ms | 16.6× | 24% |
| 4.0s | 170ms | 23.6× | 34% |
| 6.0s | 215ms | **27.9×** | **43%** |
| 14.0s | 480ms | 29.2× | 96% |
| 16.0s | 506ms | 31.6× | 101% |

Inference grows with the buffer but sub-linearly — encoder cost rises while
decode, driven by transcript length, dominates. Benchmark with **speech, not
silence**: silence produces no tokens to decode and flatters the model badly.

Resulting commit latency at the default n=3 / 500ms: **~1.1–1.2s** for typical
2–6s buffers, inside the 2s target at ≤43% duty. n=2 (~0.7s) is still
affordable, and so is a shorter interval.

**Duty cycle remains the constraint to watch**, since it scales with buffer
length; `main.rs` warns when inference exceeds the tick interval, which is the
point at which the system falls behind realtime permanently. Buffer trim policy
is therefore load-bearing. Cut at committed sentence boundaries, preferably at
VAD-detected silence.

### The tick stretches; the window does not shrink

With 1.7B the duty-cycle constraint binds for real, at a 16s buffer. It is
defended by making `--interval-ms` a **floor** rather than a fixed period: after
each hypothesis the next tick is set to `max(interval, inference × 1.25)`.

Why not the two obvious alternatives:

- **Raising the interval** costs `(n−1) × interval` of commit latency on *every*
  commit — ~1.2s to ~1.67s at 750ms — to fix a problem that only exists in the
  last few seconds of an unusually long utterance.
- **Shrinking the window** pays for it by truncating the speaker instead,
  which is worse: it damages the transcript rather than the refresh rate.

Stretching pays only when the buffer is actually long. Typical 2–6s buffers
never reach the floor and keep the full ~1.2s commit latency.

**Why the cliff is sharp, and why this is not optional.** The main loop used to
take one audio chunk per iteration. Once inference exceeded a tick it consumed a
single chunk per forward pass, the capture channel backed up, and the cpal
callback dropped buffers — the model was then handed shredded audio. Measured on
a 24s continuous passage, identical input, tick policy the only difference:

| policy | output |
|---|---|
| adaptive | `So a benchmark run on silence would flatter the model considerably and tell.` |
| fixed 500ms | `So. a. bench. run. on silence. would. flat. the. model.` |

Degradation is therefore **not** graceful without this, and the fragmented form
is the signature to recognise. With it, the same passage reached a 19.9s buffer
(644ms inference, tick stretched to 805ms, 80% duty) and stayed intact.

The last inference predicts the next one well — cost grows smoothly with buffer
length, and within an utterance the buffer only grows. The stretch resets at
each endpoint, since the next utterance starts short.

**Draining one chunk per iteration was itself half the cliff.** The tick stretch
keeps the *steady state* under 100%, but three passes deliberately run off the
tick and can land back to back: the endpoint final pass, the immediate re-decode
when a continuation merges, and the extra pass over the head when the buffer is
trimmed. Each buffers tens of chunks, and taking one per iteration meant the
loop was still behind when the next pass started. The main loop now blocks for
one chunk and then `try_iter`s the rest, so a burst costs latency that pauses
give back rather than costing words. `CAPTURE_BACKLOG` (512 chunks, several
seconds) is the second half of that defence.

### The buffer trim: cut at a pause, never mid-word

`--max-window-s` is gone; `--trim-after-s` (default 30) replaces it, and the
difference is not the name. The old flag was a **hard cut at a sample offset**,
which sliced whatever word happened to be there. The new one is a *trigger to go
looking for a gap the speaker already left* — a run of ≥12 silent frames
(~190ms) inside an utterance, well under `--endpoint-ms` so it catches the
pauses that are not endpoints. If there is no such gap, **nothing is cut** and
the buffer keeps growing. The speaker is never truncated; what is bounded is how
far behind the transcript falls.

**Prefer the longest gap, not the most recent one.** A ~200ms gap is usually a
comma, a ~500ms one usually a sentence boundary, so cutting at the longest
available candidate splits the transcript where it was going to split anyway.
Measured on a 42s passage, identical audio, selection rule the only difference:

| rule | output around the seam |
|---|---|
| most recent gap | `If I never leave such a gap,` / `Then nothing gets cut at all…` |
| **longest gap** | `If I never leave such a gap, then nothing gets cut at all…` |

It fixed two bad splits and one misrecognition (`That trait` → `That trade`),
taking the same passage from 9 broken sentences to 7 clean ones. The merge seam
is pushed as a candidate too, since the reconstructed pause is by construction
the longest silence in the buffer.

**"Never cut" needed a bound, because "keeps growing" had none.** D9 says a
buffer with no gap in it is simply not cut, and that the cost is latency. That
was wrong about the cost. Reported live: a speaker who never left 190ms reached a
**63.6s buffer**, at which point one forward pass ran ~67s and the tick stretched
to ~84s — the session stopped responding, and (before the fix below) nothing had
reached the document either.

So the trim now records every silence down to `MIN_DIP_FRAMES` (~64ms) and
raises the bar rather than the candidate list:

| buffer | eligible gaps |
|---|---|
| < `--trim-after-s` | *(no trim)* |
| ≥ `--trim-after-s` | ≥ ~190ms — D9's quality bar, unchanged |
| ≥ 2 × `--trim-after-s` | any recorded silence, longest still preferred |

A ~64ms silence can be the closure of a stop consonant rather than a word
boundary, so cutting there really can clip one. That is the trade being made
knowingly: a bad cut is a bad sentence, an unbounded buffer is not an outcome at
all. It only ever *adds* candidates in a situation that had none, and the
longest-gap preference is untouched, so nothing changes for a speaker who pauses
normally. Verified with `--trim-after-s 10` on 80s of continuous speech: the
buffer peaked at **13.3s** against the 20s bound.

### Measured duty cycle, end to end

On the 42s passage above, at `--trim-after-s 30` with everything else default:
**63 passes, 29.8s of inference for 41.6s of audio — 72% duty**, of which only
1.4s was off-tick. Wall time 44.7s against an ideal of 45.4s. So the pipeline
sustains realtime with ~28% headroom at a 30s trim threshold.

Protocol overhead is **~7ms per pass** — the whole buffer is re-sent as base64
JSON every tick (66MB over that run) and it does not matter. Do not bother
optimising the wire format; it was measured and it is noise next to the forward
pass.

**The simulator was lying, and any timing measured before this is suspect.**
`audio::simulate` paced itself by sleeping 20ms *between* sends. `sleep`
guarantees *at least* its argument, so the loop accumulated the send cost plus
whatever the scheduler added — and under load, with the main thread saturating
the GPU and Python contending for CPU, that was a lot. The same 41.6s file took
**58s** to send, which made the pipeline look ~40% slower than realtime when it
was in fact keeping up at 72%. It now paces against an absolute deadline
(`start + 20ms × n`). Never measure throughput with a relative-sleep pacer.

Model load: **~1.0s** warm (~42s first run, including download), plus ~1.5–2s of
warmup before the sidecar reports ready — so budget **~3s** of startup, not 1s.

**Utterance-final words** never receive right-context from later speech, so
sliding-window alone leaves them provisional forever. VAD close (400–700ms
silence) forces a final pass that commits the tail. The two mechanisms compose:
sliding-window covers the interior, VAD close covers the terminus.

---

## 4. Architecture

```
┌─────────────────────────── Rust ────────────────────────────┐
│  cpal (mic, 48kHz)                                          │
│      └→ rubato resample → 16kHz mono f32                    │
│          └→ ring buffer (lock-free, RT-safe producer)       │
│              └→ VAD: gate + endpoint detection              │
│                  └→ window scheduler (every 500ms)          │
│                      └────────┐                             │
│                               ↓ NDJSON / stdin              │
└───────────────────────────────┼─────────────────────────────┘
                                ↓
┌────────────────── Python sidecar (MLX) ─────────────────────┐
│  Qwen3-ASR-0.6B  →  hypothesis text                         │
└───────────────────────────────┬─────────────────────────────┘
                                ↓ NDJSON / stdout
┌───────────────────────────────┼─────────────────────────────┐
│  LocalAgreement-3 committer   ↓                             │
│      └→ window state (agreed | provisional)                 │
│          └→ command parse (wake word) ──→ commit / reject   │
│              └→ document (sentences, each kept or pending)  │
│                  └→ alternate-screen live view              │
│                      └→ real scrollback, once, on exit      │
└─────────────────────────────────────────────────────────────┘
```

**Division of labor:** Rust does everything except the forward pass. The
transcript state machine (§6) is the actual engineering — no library ships it,
because the commit policy is application-specific.

---

## 5. Continuation after a hesitation pause

`endpoint_ms` conflates two acoustically identical events: **terminal silence**
(thought finished) and **hesitation silence** (mid-thought pause). No duration
threshold separates them — people pause 1–2s while thinking and 400ms between
finished sentences. Raising the threshold just trades one failure for the other.

So a pause is only a full stop *in hindsight*. On endpoint the audio is retained
(`Pending` in `main.rs`). If speech resumes within a grace window, the committed
line is erased, the audio is merged with the new window, and the whole thing is
re-decoded as one utterance.

**Merge at the audio level, never at the text level.** Splicing two
independently decoded strings cannot produce correct punctuation or
capitalisation; handing the model the merged audio can, because it sees the
prosody. Measured:

| input | output |
|---|---|
| `"So here is the thing I wanted to"` alone | `"So here is the thing I wanted to."` |
| `"say, and that's why it matters."` alone | `"Say, and that's why it matters."` |
| **merged audio** | `"So here is the thing I wanted to say, and that's why it matters."` |
| `"This is the first complete sentence."` + `"And this is a totally separate one."` merged, 1.4s gap | `"This is the first complete sentence. And this is a totally separate one."` |

The model arbitrates: it fuses genuine continuations and re-separates unrelated
utterances, unprompted. We never decide — we only supply the audio.

### Two measured findings that shape the tuning

**Terminal punctuation is not a completeness signal for this model.** Qwen3-ASR
punctuates whatever it is handed: on the fragment "So here is the thing I wanted
to" it returns `"...wanted to."` So *absence* of a full stop indicates
incompleteness, but *presence* proves nothing.

This used to drive `looks_incomplete`, a trailing-function-word check that gave
a cleanly-finished sentence only a quarter of the grace window. **That heuristic
is gone**, and deliberately: the grace is now a flat `--continue-ms` for every
utterance. It existed to limit pointless re-decoding when a settled sentence had
already been printed to scrollback, and nothing is printed any more (§6) — a
settling sentence just sits dim in the live document, so leaving it open costs
nothing worth guessing about. Do not reintroduce it to save recompute; the
measurement below says recompute was never the expensive part.

**The merge must carry the real pause.** Originally it did not, and that was a
self-inflicted bug rather than a limit of the model. On endpoint the window ends
~600ms into the silence (where VAD fired) and the pre-roll trim discarded the
rest, so the merged audio always showed the model ~0.9s of gap — whether the
speaker had paused 1 second or 30. Merging then fused sentences that a human
would never join:

| gap the model saw | result |
|---|---|
| real 1.4s silence | `"...complete sentence. And this is a totally separate one."` |
| pipeline merge, compressed to ~0.9s | `"...complete sentence, and this is a totally separate one."` |

`gap_samples` in `main.rs` now counts the silence the pre-roll trim throws away
and splices it back on merge, capped at `MAX_GAP_SAMPLES` (1.2s — beyond about a
second the model already reads a boundary, so faithfully reproducing a 30s pause
would only cost encoder time).

**Consequence: generous `continue_ms` values are safe.** Verified against known
ground truth at 0 / 2000 / 4000 / 8000 ms — continuations fuse, separate
utterances stay separate, transcription accuracy unchanged. Raising it costs
recompute and on-screen movement, not correctness.

**Default is 6000ms.** This is one number doing two jobs that pull in opposite
directions, which is why it is worth stating rather than tuning by feel:

- it is how long text stays **dim** — shorter is more responsive;
- it is how long a **thinking pause** may run before one thought becomes two
  sentences — longer is more forgiving.

It went 4000 → 10000 → 6000. Ten seconds settled text so late the display felt
unresponsive; six keeps the merge generous while halving the wait. Verified end
to end from both sides at each setting:

| gap | result |
|---|---|
| 4s (inside the window) | `This is the first thought, and this continues it.` — merged, one sentence, comma across the seam |
| 8s (past it, would have merged at 10s) | `This is the first thought.` / `This is a separate thought.` |
| 12s (past it) | two sentences, no merge |

The cost is real but is not accuracy: **every** ordinary pause still falls inside
6s, so a dictation session merges continuously and the buffer never resets on its
own. That is precisely what makes the §3 trim load-bearing rather than a nicety
— without it, a ten-minute session would be a single 600s window.

### Terminal safety: never write what you may have to take back

The original design printed the sentence to scrollback at the endpoint and
rewrote it with `Renderer::retract()` if a continuation arrived. That was wrong
on two counts, and both were user-visible.

**It made plain text lie.** Dim means "can still change", plain means "never
again" — that is the correctness signal §1 says the whole interaction is built
on. A sentence inside its grace window can be re-decoded word for word, so
printing it plain claimed a permanence the pipeline did not have.

**And the fallback silently dropped the feature.** `retract()` refused whenever
it could not be sure the text was still on screen — in particular after any
`notice()`. Refusing left the sentence standing, which reads as harmless, but it
also skipped `retract_last_sentence()` and the audio merge: one thought stayed
split across two sentences, with nothing on screen to indicate it.

So a held sentence is now never written at all. `Renderer::settling()` keeps it
**dim in the live region**, and `finalize()` runs only once the grace closes —
on a terminal and when piped alike, which is what §5 wanted anyway ("the merge
benefit is content, not cosmetics, and must survive redirection"). The two paths
no longer differ in content at all.

Consequences worth keeping:

- The merge is now **unconditional**. There is no refusal case, because there is
  nothing on screen to take back.
- `retract()`, `revisable` and `is_tty()` are gone.
- Styling downgrades from partly-plain to fully-dim at the endpoint. That is
  correct, not a glitch: the instant an utterance becomes retractable, *every*
  word in it is provisional again.

**This argument was later followed one step further.** If plain text must not
appear while the *pipeline* can still rewrite it, then a `reject` command that
lets the *speaker* rewrite it means the transcript cannot live in the scrollback
at all — see §6. `Renderer::settling` and `Renderer::held` are gone with the
rest; a settling sentence is now simply a dim entry in a document that is
re-derived from the model on every frame, so there is nothing to restore after a
notice and no cursor arithmetic to get wrong.

## 6. Transcript state machine

The core of the product. Owns:

- **Committed vs provisional** as *data*, not styling. Committed spans are
  append-only; provisional is a replaceable tail.
- **LocalAgreement-n**: a word is promoted to committed once it appears in the
  longest common prefix of the last `n` hypotheses.
- **Dedup, two layers**: (a) within-window via common-prefix tracking of how
  much has already been emitted; (b) across-window via the audio timestamp of
  the last cut — any re-emitted text before it is discarded.
- **Context carry-over — removed, do not reintroduce.** The last ~200 words of
  committed transcript used to be fed back as `system_prompt`. It did nothing
  for accuracy and replayed the transcript on any non-speech window. §7 has the
  measurements.
- **Addressable spans**: each committed span carries an id + audio time range,
  so a future "revert the last sentence" has a referent.

### The document, and why nothing reaches scrollback until exit

`Transcript` is two layers now, and keeping them distinct is the whole design.

The **window** is LocalAgreement as before: `committed` there is a claim about
the *pipeline* — these words are agreed and nothing upstream will revise them.
The **document** is a `Vec<Sentence>`, each carrying `committed: bool`, which is
a claim about the *speaker* — they said "Luna, commit" out loud.

`finalize_window()` deliberately does **not** file into the document. The caller
decides, because an utterance that has merely ended may still be a hesitation
(§5); filing there would mean un-filing on merge. That is also why
`retract_last_sentence()` is gone — a pending sentence is no longer in the
document to retract.

**Scrollback is the one region a program cannot take back.** `Luna, reject` has
to reach a sentence the speaker already committed, so the transcript cannot be
appended to the scrollback while the session runs. It lives on the **alternate
screen**, redrawn from the model every frame, and is written to the real
scrollback exactly once, on exit — the moment it genuinely stops being editable.

The styling contract survives this intact, and it is worth being precise about
why. It was always a promise about the *pipeline*:

| rendering | meaning |
|---|---|
| dim | the pipeline may still rewrite this |
| plain, `│` gutter | settled; only the speaker can change it now |
| plain, no gutter | the speaker committed it |
| `⋮` gutter, top row | more transcript above than the screen holds |

The last of those is about the screen rather than the text, and it exists
because its absence cost two wrong diagnoses. **The live tail is never capped**
— a speaker who has not finished has to read what they are saying (§1) — so on
a long buffer the document scrolls out of view. It used to do that silently,
which is indistinguishable from the document being empty, which is what it
actually was. Nothing is hidden to make room for the marker: it replaces a
gutter that was already budgeted, so it cannot widen a row (§7 records the last
elision marker doing exactly that).

A speaker rejecting their own sentence is not the pipeline changing its mind, so
plain text still means what it always meant.

Consequences:

- **The document is unbounded**, and `RETAINED_WORDS` is gone. It was right to
  prune when this was a scratch buffer whose only consumer was retraction; it is
  the deliverable now, and dropping the front of it would discard what the
  speaker came here to produce. A five-hour session is about a megabyte. What
  has to stay bounded is *rendering*, and `build_rows` walks back from the newest
  sentence and stops once the screen is full — so an hour of dictation costs the
  same frame as the first sentence.
- **`prev_rows` is gone.** Frames address every row by number (`\x1b[{r};1H`)
  and clear it first, so drift is not merely avoided but unrepresentable. The
  strict-width invariant stays: rows must still be narrower than the terminal,
  gutter included, or the terminal's own wrap fires.
- **Ctrl-C must be caught.** The alternate screen has to be left, or the user
  gets a shell they cannot see themselves typing into. `SIGINT`/`SIGTERM` set a
  flag, the loop exits at the next iteration, and the transcript is still
  written. A *second* signal restores the terminal with `libc::write` and
  `_exit(130)` — async-signal-safe, unlike anything in `Renderer` — for the case
  where the loop is blocked in a long forward pass.
- **`Renderer::drop` also restores**, so a panic cannot leave the terminal
  wedged either.
- Uncommitted sentences are still printed at exit. Losing text because someone
  forgot the magic words would be indefensible; the count goes to **stderr** so
  it cannot pollute the transcript on stdout.
- The document is also on disk throughout, which is what makes the destructive
  commands safe — see *The session file* below.

### Spoken commands

`command.rs`. `Luna, commit` keeps everything settled so far; `Luna, reject`
drops the newest sentence, **including one already committed**. The name is
`--assistant`, default `Luna`.

**This reverses the removal recorded below, and the two are not in conflict.**
Self-repair replaced the wake word for *fixing a word*, and that was right —
nobody says "admin" before stuttering, they just say the word again, and there
is an acoustic signal to key off. But "keep this" and "throw that away" have no
acoustic signal whatsoever. They are document operations, not speech events.
Nothing short of a wake word can express them, which is exactly why one earns
its place here and did not earn it there.

**Detection runs on finalized utterances only, and this is a correctness
requirement rather than an optimisation.** `repair()` is safe per-hypothesis
because it is pure. Commands are not: `reject` destroys a sentence, and the
sliding window re-decodes the same audio every 500ms, so per-hypothesis
detection would empty the document one sentence per tick.

Two rules together make a command fire exactly once:

1. It is parsed from the finalized utterance, not from a hypothesis.
2. The audio that carried it is **dropped, never retained for continuation**.
   Retention is the only thing that could replay it, so not retaining it is the
   entire fix. A command therefore also skips the settle period — making the
   speaker wait ten seconds to learn whether they were heard would defeat the
   point.

**Matching is wake word immediately followed by the verb**, compared through
`normalize`, which is what makes "Luna, commit" and "Luna commit" the same input
— the comma is punctuation the model chose and the parser never sees it. Nothing
may come between them. The asymmetry justifies the strictness: a *missed*
command costs one repetition, a *false* one deletes text the speaker meant to
keep. Measured controls, all left as dictation: `Luna went to the store.`,
`I will commit the change tomorrow.`, `Commit Luna`, `Luna please commit`,
`Luna committed`.

**A third-person `-s` on the verb is the same command**, and this is the one
place the strictness is relaxed. Reported live: a spoken "Luna, reject" comes
back from the model as `Luna rejects.` The speaker said the verb — the model
conjugated it — so reading that as dictation punishes them for a transcription
they can neither see nor influence, and it is the one failure the wake word
cannot rescue, because they *did* say the wake word. It applies to every verb,
not just `reject`: `commits`, `clears`, `copies`, `rollbacks`, `undoes`.

The cost is real and worth stating: `Luna rejects the offer.` is now a reject.
It is bounded rather than open-ended — `--assistant` retargets the name,
`Debounce` stops a repeat compounding it, `rollback` takes it back — and the
allowance stops at `-s`. `Luna committed` and `Luna rejecting` stay dictation,
because a past or progressive form is a plausible thing to say *about* a person
called Luna in a way the bare present tense following the name is not.

Segments are applied **in spoken order**, not with commands hoisted. "That's the
wrong sentence. Luna, reject." must file the text before the reject reaches it,
or the reject deletes the sentence before the one the speaker just finished.

**The merge creates a seam comma that has to be closed.** Because a command
usually follows a pause, §5 merges both halves into one buffer, and the model
then punctuates them as running on. Measured end to end:

| spoken | model returns | after stripping the command |
|---|---|---|
| "This is my first sentence." + "Luna, commit" | `This is my first sentence, Luna, commit.` | `This is my first sentence,` ← dangling |

`text::close_sentence` fixes the last word of a Text segment that is *terminated
by a command*. This is not a guess about what the speaker meant: the words that
followed were provably not part of the sentence, so the sentence ended there.
Text appearing *after* a command is left alone — nothing establishes that it
ended.

Five verbs: `commit`, `reject`, `clear`, `copy`, and `rollback` — the last of
which also answers to `undo`. Two words for one command is otherwise not
something this module does; `rollback` is the operation's name everywhere else
here, `undo` is what people actually say, and both still require the wake word
immediately before them so the extra surface costs nothing. Worth knowing that
the model is not equally sure of them: measured, `Luna, undo` came back as
`Luna, Andú.` inside a longer utterance while firing correctly in isolation.
The failure is benign — an unrecognised command is filed as dictation — and
having the second verb is the recourse.

`copy` takes the **whole document, and commits it** — copying is itself an act
of approval, the same argument `--resume` makes about a recovered file.

This reverses an earlier "committed sentences only" reading, on use. That
reading was defensible in the abstract and useless in practice: reported from a
live session as `copy` reporting *0 kept, 4 pending* after the speaker had
waited out the settle window and assumed a settled sentence was a kept one. A
speaker who says "copy" has looked at the text and decided to do something with
it, which is what commitment records; making them say two commands to express
one intention was ceremony, not safety.

The boundary that **is** load-bearing survives untouched, and it is a different
one: `copy` stops at the *document*. An utterance still inside its grace window
is not in there yet, so text the **pipeline** may still rewrite is never copied
and never committed. The rejected reading guarded against the speaker not having
vouched for text; this one guards against the text not being final, which is the
guarantee only this program can make. The commit lands only if `pbcopy`
succeeded, since that is what the approval attaches to.

It shells out to `pbcopy` with both output streams silenced — a child writing to
the terminal would land in the middle of a frame, the same invariant that made
the sidecar's stderr a pipe rather than inherited.

**Auto-committing on the settle timer was the alternative, and was rejected.**
It would have collapsed the three-level ladder in §6: the `│` gutter would only
ever flash for six seconds, `Luna, commit` would become a no-op, and "kept"
would stop meaning the speaker approved anything — a timer would be doing the
vouching. Copy-commits gets the same practical result at the moment the speaker
actually acts.

#### Repeats are suppressed on a timer, not per utterance

`Luna, reject` twice in quick succession is one intention said twice. The
speaker cannot see a command land until the utterance ends, so a repeat inside
roughly one reaction time means they think they were not heard — and taking it
literally deletes a second sentence.

**The obvious implementation is wrong**, and measurably so. Collapsing adjacent
commands inside `split` catches nothing in practice: 700ms between two "Luna,
reject" is past `--endpoint-ms`, so they arrive as *two separate utterances* and
`split` never sees them together. Measured, both fired and two sentences went.

`command::Debounce` therefore works on wall-clock time, with a 3s window — sized
from that measurement, where the two firings landed ~1.7s apart. It guards
`Reject` and `Clear` only. **`Rollback` is deliberately unguarded**: saying
"undo, undo, undo" is how a run of rejects is walked back out, so there the
repetition *is* the intention. Each suppressed repeat re-stamps the window, so a
burst of five collapses to one rather than letting every second one through.

A suppressed command still reports itself. Silence is indistinguishable from not
being heard, which is exactly what provokes the repeat.

#### Rollback, and why `clear` is safe to have

`clear` throws away the whole document, which is only defensible because
`rollback` puts it back. Undo entries are **deltas, not snapshots**: a reject
remembers one sentence, where snapshotting would cost a copy of an
unbounded transcript per step. Depth 64.

Only text-*removing* commands are recorded. Undoing a `commit` would mean taking
back an approval, which removes nothing and is not what anyone means by undo —
and, more importantly, must not consume the undo entry that does have text
behind it.

Verified end to end on synthesized speech: commit keeps, reject drops a
committed sentence and leaves the document empty, `clear` removes 2 and
`rollback` restores both with their commit state intact, `copy` moves the kept
text onto the real clipboard, a repeated reject is suppressed with a notice,
`--assistant Jarvis` retargets everything, and `Luna went to the store.`
survives as dictation.

### The session file

`store.rs`. The document is written to `~/.local/state/speech-to-text-cli/` on
every change, and **that is what makes `clear` and `reject` safe to offer**.

Before this the transcript reached stdout once, at exit, and three paths skipped
that write entirely: a second Ctrl-C (`libc::_exit`, which by design runs no
destructors), a panic (the renderer's `Drop` restores the terminal but writes
nothing), and SIGHUP when the window closes. Adding a blunt irreversible command
on top of data whose only copy was process memory would have been indefensible.

**The deletion rule is the interesting part.** A *clean* exit removes the file
unless `--persist`, because stdout already has the transcript and leaving a file
behind after every session is litter. An *unclean* exit cannot remove it — there
is no code running to do so — and that asymmetry is the design rather than an
oversight: the file survives precisely the cases that used to lose text, and is
cleaned up in the one case that never did. Verified with `kill -9` mid-session:
the settled sentence was on disk, with no `--persist`.

Writes go through a temporary file and a rename, so a crash midway leaves either
the old transcript or the new one, never half of one. Cost is nothing on the hot
path — the file is rewritten when a sentence settles or a command runs, a few
times a minute, and `Transcript::revision` makes an unchanged document free. A
failed write reports **once** and the session continues on memory alone; losing
the safety net is not a reason to end a dictation session.

#### The file holds the utterance in flight too, and has to

"Written on every change to the document" was not the same guarantee as "holds
what the speaker has said", and the gap between them was the whole passage.
Reported from a live session as `Luna, copy` copying nothing and the session file
being empty of everything just dictated.

Mid-passage the **buffer trim is the only thing that files anything**. §5 merges
every ordinary pause, so an utterance never settles, `finalize_window` files
nothing by design, and the grace never closes. Between trims the text exists
only in `Pending` and `Transcript::committed` — process memory, which is exactly
what this module was written to stop relying on. The document, `copy` and the
file all read `script.document()`, so one empty document explains all three
reported symptoms at once.

`Store::save` therefore takes the in-flight text as well, and writes it as
trailing lines. Measured on a 41 s continuously-merging passage:

| | file at T+20s | after `kill -9` at T+20s |
|---|---|---|
| document only | *(empty — 0 document entries until the first trim at ~T+33s)* | nothing |
| **with in-flight** | 4 sentences | the same 4 sentences |

Three decisions inside it:

- **Rate-limited, not written through.** The document is written the instant it
  moves; a *growing* in-flight line moves every tick and would put an `fsync` on
  the tick for no gain. 2 s bounds what a crash takes to about a clause.
- **A *shrinking* one is written immediately.** It shrinks because the document
  absorbed it, and deferring that would leave the file briefly holding the same
  words twice.
- **Split with `split_sentences`, like `file()` does.** One line is one sentence
  everywhere in this format. Written whole, the 20 s recovery above came back as
  a single 230-character line, which `--resume` would hand back as one
  unrejectable blob.

The line is deliberately unmarked, because the file has to stay a clean
transcript — `--persist` hands it to a human and `--resume` reads it straight
back. A clean exit files the tail first and then saves with an empty tail, so an
in-flight line only ever survives into a file a crash left behind.

#### Resume

`--resume` bare recovers the most recently *written* session file — by mtime,
not by name, because the name records when a session started and after a resume
the interesting one is whichever was written last. `--resume <path>` takes a
specific file.

Three decisions worth keeping:

**Recovered text comes back kept.** The file stores text, not the workflow state
that produced it, and there is nowhere to put that state without making the file
something other than a clean transcript — which it has to stay, because
`--persist` hands it to a human. Restoring as *pending* was the alternative and
is worse: `Luna, copy` would come back empty on a resumed session, which is the
opposite of picking up where you left off. Choosing to resume a transcript is
itself an approval of it.

**A recovered session is continued in place, but only if it is ours.** A file
found by us in our own state directory is adopted — same file, no duplicate, and
`--persist` governs it as it would any other. A path the user *named* is
read-only, and a new session file is created beside it, because adopting it
would mean a clean exit deleting something they pointed us at. Verified: an
explicit `--resume ~/mynotes.txt` left the file byte-identical.

**`restore` is not undoable.** `rollback` takes back what the speaker did during
a session, and this happened before the session began. Letting undo reach it
would mean one "Luna, undo" emptying a transcript they had only just recovered.

Two things the tests caught that are worth not re-learning:

- The "a previous session is on disk" hint has to be computed **before** the
  store opens this session's file, or `newest_session()` finds our own and
  reports it back to us. It did exactly that.
- Storage is resolved before the sidecar spawns, so `--resume` with nothing to
  resume fails in 0.09s rather than after three seconds of model warmup. It also
  **fails** rather than starting empty: dictating into a fresh session believing
  it is the old one is the sort of thing discovered only at the end.

### Self-repair

`repair.rs`. When the speaker corrects themselves mid-flow — "the regularing…
regular expression" — the false start is dropped and the correction kept.

**Wake-word editing commands were removed and replaced by this.** The grammar
worked as specified and its tests passed, but it asked the speaker to remember
a command language and say "admin" mid-sentence, which is not what people do
when they misspeak. What they actually do is say the word again. So the
detector now watches for that directly, and there is no command language left
to learn.

That reasoning still holds **for repair**, and it is worth keeping straight now
that a wake word exists again for document operations (above). The dividing line
is whether the intent has an acoustic signature: saying the word again is one,
so repair needs no command; "throw that away" is not, so it needs nothing else.

The cost of dropping the wake word *for repair* is real and is stated under
*What this cannot do* below: an arbitrary substitution like "Tuesday…
Wednesday" is still not expressible. `Luna, reject` does not recover it either
— it operates on whole sentences, not words.

**The model transcribes disfluencies verbatim**, which is what makes the
feature reachable. Measured on `Qwen3-ASR-1.7B-8bit`:

| spoken | transcript |
|---|---|
| "The regularing · regular expression is broken." | `The regularing regular expression is broken.` |
| "I need to conf · configure the server today." | `I need to conf configure the server today.` |

#### Two signals that do not work, both measured

**Punctuation.** The obvious cue is the hesitation pause, which the model
writes as a comma. It does not, here — and it is actively misleading:

| spoken | transcript |
|---|---|
| "The regularing · regular expression…" | `The regularing regular expression…` — no comma |
| "He was at work working late…" | `He was at work, working late…` — comma |

The repair seam came back clean while an innocent control got the comma. On
this model punctuation is anti-correlated with repair, so it is not usable.

**Edit distance.** `form`/`from`, `quiet`/`quite`, `trial`/`trail`,
`casual`/`causal` are one or two edits apart, and "take the form from the
office" is an ordinary sentence. The rule is therefore **prefix containment
only** — one word must be a strict prefix of the other, which `form`/`from` is
not.

#### The asymmetry, which is what keeps it safe

Containment alone still is not enough, and the two directions fail differently,
so `repair.rs` gives them different thresholds.

**Longer first** — "regularing" then "regular". English essentially never puts a
word immediately before its own prefix. The one systematic exception is a plural
noun meeting its verb, so excluding the `-s`/`-es` extension covers it and the
stem can be short (4).

**Shorter first** — "conf" then "configure". Genuinely ambiguous, because
English *does* put a word before its own derivation. So it needs a longer stem
(5) *and* an extension that is not a plain inflection. Every counterexample
found is a four-letter word, which is where the 5 comes from.

Measured controls, all left untouched end to end:

| sentence | why it is not a repair |
|---|---|
| `These changes change the behavior of the parser.` | plural meeting its verb (`-s`) |
| `The tests test the parser thoroughly.` | same |
| `He was at work, working late on the report.` | derivation, 4-letter stem |
| `The main maintenance window is on Sunday.` | same |
| `Just justify the text and be done.` | same |
| `Take the form from the office.` | no containment |

#### What this cannot do

- **Arbitrary substitutions.** "Let's meet on Tuesday… Wednesday" is not a
  repair this can see, because the two words share nothing orthographically.
  Verified: it comes out as `Let's meet on Tuesday. Wednesday.` This is the
  capability the wake word bought, and dropping it is the price of not having
  one. Nothing short of a wake word or a semantic model recovers it.
- **Exact repetition.** "the the" is left alone on purpose. It is the commonest
  disfluency of all, but "I had had enough" and "he said that that was fine"
  are ordinary English and nothing here separates them.
- **Fillers the model renders as real words.** "regularing · uh · regular" was
  transcribed `The regularing a regular expression is broken.` — the "uh"
  became an article, which breaks the pair. `FILLER` deliberately excludes
  articles: eating a real "a" is worse than missing this.
- **Four-letter truncations.** "conf"/"configure" is not caught, because
  catching it would also catch "just"/"justify".

The bar is the one the commands were held to: **a near-miss is dictation, never
a guess.** Deleting a word the speaker meant to keep is much worse than leaving
a stutter on screen.

#### Why it is a pure function

The sliding window re-decodes the same audio every tick, so a repair sits in
*every* hypothesis from the moment it is spoken until the utterance ends.
Anything stateful would apply it repeatedly.

`repair()` is therefore a pure function of one hypothesis. Running it twice
gives the same answer, so it is safe to apply to provisional text, and a
re-decode that no longer hears the false start simply stops dropping it — with
nothing to undo. This is the one property worth keeping from the command
design, and it is why nothing here is destructive: unlike a sentence removal,
a repair never needs to wait for the window to close.

#### The one invariant this bends, deliberately

A repair reaches text the window has already committed — the false start is
plain on screen several ticks before the correction is spoken. So
`push_hypothesis` cuts `committed` back to where it still agrees with the
newest hypothesis. Nothing else in the state machine ever shortens it, and
without that the replaced word survives forever: every later comparison is made
against `committed.len()`.

Observed live, on `--simulate` through a pty:

```
We should probably rewrite the regular ⟨ring.⟩        ← "regular" committed plain
We should probably rewrite the ⟨regularing.⟩          ← committed shrinks back to "the"
We should probably rewrite the ⟨regular expression.⟩  ← repaired
```

That means plain text in the *live region* can change. The styling contract is
unharmed, because it was always a promise about the pipeline: dim means the
pipeline may still rewrite this, plain means it will not. Scrollback is
untouched by any of this — the final line was byte-identical to the same
sentence recorded without the stutter.

---

## 7. Key technical facts (verified — do not re-research)

- Qwen3-ASR is **AED**, 8× downsampling, **12.5 Hz token rate** (80ms/token).
  0.6B = 180M AuT encoder + Qwen3-0.6B LLM. 1.7B = 300M encoder + Qwen3-1.7B.
- The AuT encoder has a **dynamic attention window (1s–8s)**, so streaming is
  *trained into the weights*. It is unimplemented outside vLLM, not
  architecturally absent — an MLX port is engineering, not research.
- Official streaming is **vLLM-only**, gives up batching and timestamps.
- Open weights have **no context biasing / hotwords**. That is Qwen3-ASR-Flash,
  the API-only model.
- Timestamps require the separate **Qwen3-ForcedAligner-0.6B** (~42.9ms, 11 langs).
- Audio contract everywhere: **16 kHz, mono, f32**. Mic delivers 48kHz →
  resampling is mandatory.
- Feature frontend matches Whisper's: 128 mel bins, n_fft=400, hop_length=160.
- Whisper's fixed 30s window does **not** apply to Qwen3-ASR (variable length).
- Model formats are containers; a runtime must implement the *architecture*.
  Converting Qwen3-ASR to GGUF is meaningless — whisper.cpp only knows Whisper.

### Failure modes to defend against

- **Hallucination on silence** — AED models emit confident garbage on
  near-empty audio. **Observed here**: an open mic in a quiet room produced
  "Okay." and "The." because the neural VAD fired on ambient noise. Mitigated
  with an RMS floor (~-40 dBFS) ANDed with the detector in `vad.rs`; 15s of
  silence now yields nothing. Do not remove the floor — the detector alone is
  not sufficient.

  **Know what the floor does not cover.** It rejects *quiet* audio, not
  *non-speech* audio. Measured: white noise at −34 dBFS clears the floor
  comfortably, and only the neural detector rejects it. Speech-like background —
  a TV, a conversation across the room — passes both, and the model then
  transcribes it, because from the pipeline's point of view that is what it is.
  Gating is the only defence here and it is inherently imperfect; what must not
  happen is that imperfect gating gets *amplified* into replaying the whole
  transcript, which is the bug below.

  **The default floor is not universal, and it was not tunable — now it is.**
  Reported from a live session as stray `Okay.` and `Oh.` lines appearing while
  the speaker said nothing. Reproduced here from noise alone, with no speech in
  the file at all:

  | 20s of | default −40 dBFS | `--rms-floor -30` |
  |---|---|---|
  | digital silence | *(nothing)* | *(nothing)* |
  | noise at −48 dBFS | *(nothing)* | *(nothing)* |
  | noise at **−38 dBFS** | **`Oh.`** | *(nothing)* |

  −38 dBFS is an unremarkable room — a desk fan clears −40 on every frame, the
  detector then fires on it, and the model transcribes what it is handed. The
  floor was a hard-coded constant with `with_floor` reachable only from tests,
  so a speaker in a noisier room had no recourse at all. `--rms-floor` now
  exposes it in dBFS. Set it above the room and below the voice; real speech
  still passes at −30, verified end to end.

  Note `allow_negative_numbers` on the `Args` command. Without it clap reads the
  `-30` in `--rms-floor -30` as another flag and refuses to start — which it did,
  silently enough that the first verification run looked like a *success*
  (empty output from a program that never started resembles empty output from a
  gate that worked).

- **A keyboard transcribed as speech (fixed)** — reported from a live session:
  typing while the mic was open produced text. **The floor cannot fix this and
  raising it is the wrong instinct**, which is why it gets its own entry rather
  than a bigger number in the one above. A keystroke is *louder than the room by
  construction* — it is the loudest thing in a quiet office — so any floor that
  rejects one also rejects a quiet voice. Level is the wrong axis.

  Duration is the right one, and the numbers are not close. Measured against
  `earshot` directly, with a room idling above the floor so the detector is what
  decides:

  | energy | consecutive voiced frames |
  |---|---|
  | 32 ms (one keystroke) | **3–6** |
  | 64 ms | 6 |
  | 96 ms | 8 |
  | 128 ms | 11 |
  | a spoken syllable | dozens |

  The detector lags one to two frames past the end of the energy, which is what
  turns a 2-frame burst into a 4-frame run. `open_frames` was hard-coded at 48 ms
  — **3 frames** — so a single keystroke cleared it with a frame to spare. It is
  now `--open-ms`, defaulting to 150 ms (9 frames): above an isolated keystroke,
  below any syllable.

  **Know what this does not cover**, in the same spirit as the floor above — and
  this was measured *after* the fix was written, against a first draft of these
  notes that claimed it sat "above everything a keyboard produces". It does not.
  On synthesized key noise at ~7 keystrokes a second, with the keycap and desk
  resonance that follows each press, `earshot` chains straight across the gaps
  and returns runs of **23 to 51 consecutive voiced frames** — 370 to 800 ms.
  No sane open window gates that.

  What saves that case is the model: 20s of the same file returned **nothing at
  all**, at both 48 ms and 150 ms. Which locates the bug precisely. Sustained
  typing produces a long buffer of obvious non-speech, and the ASR declines it.
  An *isolated* keystroke produces a window holding a fraction of a second of
  near-silence — and that is exactly the input the first failure mode in this
  section is about, the one an AED model answers with "Okay." So the reported
  symptom lives in the regime a duration gate does fix, and the regime it does
  not fix was never the one producing text.

  Onset latency is not the cost it appears to be: the 300 ms pre-roll means the
  speech that opened the utterance is still in the buffer, so nothing is clipped.
  **Values past 300 ms would start clipping first words** — that pre-roll is the
  real ceiling on this flag.

  The same threshold now also governs *continuing*: a silence run is cancelled
  only by `open_frames` of sustained voicing. Without that, typing through a
  thinking pause reset the run on every keystroke and the endpoint — which is
  what commits the tail of the sentence and closes the buffer — never arrived.
  Silence still only accrues on genuinely silent frames, so this can never
  *cause* an endpoint; it can only stop a click from postponing one.

  **Testing this needs room tone, not digital silence.** `earshot` is stateful
  and adapts to what it is shown, and a buffer of exact zeros is a signal no
  microphone produces: fed it, the RMS floor rejects every frame either side of a
  keystroke and nothing gets through at any setting, so the test passes for the
  wrong reason. `Room` in `vad.rs` puts a fixed-seed room tone under everything
  at about −35 dBFS, which is the condition the bug actually occurs in.
- **Transcript replay through the context prompt (fixed — never do this again)**
  — reported as "it repeats the last sentence when there is background noise
  and I am not speaking". It was not the model hallucinating. We were handing it
  the answer.

  `mlx_audio`'s `_build_prompt` drops `system_prompt` **verbatim into the system
  turn of the chat template**, ahead of the audio. So when a window held no
  speech, the transcript was the strongest signal in the context and the decoder
  copied it out. Worse, it self-reinforces: the echo is committed, joins the
  history, and comes back in the next prompt.

  Measured on `Qwen3-ASR-0.6B-8bit`, 2s windows:

  | audio | with transcript prompt | no prompt |
  |---|---|---|
  | digital silence | the transcript, verbatim | `"The."` |
  | noise −46 dBFS | the transcript, verbatim | `""` |
  | noise −34 dBFS | the transcript, verbatim | `""` |
  | noise −26 dBFS | the transcript, verbatim | `""` |

  And it bought **nothing**. On real speech the output was byte-identical with
  and without the prompt — including the capitalisation case §5 cites, where
  `"say, …"` alone still came back `"Say, …"` either way. Consistent with the
  fact two sections down: the open weights have no context biasing. That is
  Qwen3-ASR-Flash, the API-only model.

  End to end, on speech followed by 20s of speech-like babble: with the prompt,
  the transcript replayed four times and then degenerated into `"So."` forever;
  without it, one garbled line for the babble and nothing else.

  **Invariant: the per-window protocol carries audio and nothing else.**
  `transcribe` takes no text argument, so there is no code path by which the
  transcript can reach the model.

  **Refined on measurement — a *static* prompt is not this bug.** The reasoning
  above turns on the prompt being dynamic and derived from what the speaker
  said: that is what makes it the strongest signal on an empty window, and what
  makes the echo self-reinforce. A fixed command list has neither property, and
  the difference is measured rather than argued. On `Qwen3-ASR-1.7B-8bit`, with
  `"Commands: Luna commit, Luna reject, …"` as `system_prompt`:

  | audio | no prompt | with the command list |
  |---|---|---|
  | 3s digital silence | `""` | `""` |
  | 20s noise at −38 dBFS — *the reported echo condition* | `""` | `""` |
  | speech-like babble | transcribed | **byte-identical** |
  | 41s of ordinary dictation | — | **byte-identical** |
  | `Luna went to the store.` + `I will commit the change tomorrow.` | dictation | dictation, no false command |

  And it fixes a command that was silently unreachable: spoken "Luna, rollback"
  comes back as `Luna, roll back.` — two words, which `command.rs` correctly
  refuses, since nothing may come between the wake word and the verb. With the
  hint it is `Luna rollback.` and fires. Verified end to end on synthesized
  speech: reject then rollback both landed, and the sentence came back.

  So the ban is on the *transcript*, not on the argument. `command::hint` is
  passed **once, at spawn**, as a CLI flag to the sidecar, specifically so the
  dangerous shape — per-request text — does not exist to be reached for later.
  Note this does not contradict "no context biasing" below: the hint is not
  biasing recognition of arbitrary vocabulary, it is fixing how the model
  *segments* two known words.

- **Repetition loops** — decoder gets stuck. The compression-ratio heuristic
  (output that gzips too well is looping) plus temperature fallback is still
  unimplemented and still worth having. What *is* implemented is the token
  budget below, which bounds the damage without detecting the loop.

- **A runaway forward pass, from `max_tokens=8192` (fixed)** — reported as a
  63.6s buffer whose refresh "slowed to 84K", i.e. a ~67s forward pass setting
  the adaptive tick to ~84s. The session looked dead for over a minute.

  `Qwen3ASRModel.generate` defaults to **`max_tokens=8192`**. At the model's
  12.5 Hz token rate that is 655 *seconds* of permitted output for any window
  however short, so a decoder that gets stuck does not fail — it runs to the cap,
  and the pass takes as long as generating 8192 tokens takes. Nothing in the
  pipeline bounded it, because the tick stretch is derived *from* `infer_ms` and
  therefore scales with the failure instead of containing it.

  The budget is now `audio_seconds × 12.5 + 64`. Measured on
  `Qwen3-ASR-1.7B-8bit`, capped versus the 8192 default on identical audio:

  | audio | budget | capped | default | text |
  |---|---|---|---|---|
  | 30.0s | 439 tok | 0.98s | 0.97s | byte-identical (570 chars) |
  | 63.6s | 859 tok | 1.82s | 1.81s | byte-identical (1030 chars) |

  So a healthy 63.6s pass is **1.8s, not 67s**, the cap never binds on real
  speech — 859 tokens against ~258 actually emitted, 3.3× headroom — and the
  worst case drops ~10×. 12.5 is the model's own audio token rate rather than a
  tuned number, which is why it is the right ceiling: a transcript cannot
  legitimately need more text tokens per second than the audio carries.

  **The tick was deliberately not clamped.** An obvious fix is a ceiling on the
  stretch, and it makes things worse: `tick = max(interval, infer × 1.25)` fires
  the next pass 0.25 × `infer` after the last one *finishes*, so clamping below
  that starts passes back to back and pins the duty cycle at 100% — the exact
  cliff §3 exists to stay off. The stretch is right; what was wrong was an
  unbounded `infer` feeding it. Fix the input, not the response.
- **Seam artifacts** — dropped/duplicated words at window boundaries. The
  context carry-over that was supposed to address this never worked (see above),
  so if seam artifacts appear, the fix has to come from window overlap or the
  agreement policy, not from prompting.
- **Redraw debris (fixed)** — the live region looked like it was duplicating
  whole sentences on real speech. It was **not** a transcript bug; the state
  machine was correct throughout. `\x1b[2K\r` clears *one* physical row, so once
  the line wrapped past the terminal width the redraw left the earlier rows on
  screen and re-printed below them. Repeat count grew with sentence length, and
  text spliced mid-word at the wrap column.

  Two invariants in `render.rs` now prevent it, and both matter:
  1. Track the previous draw's row count; move up that many rows and `\x1b[0J`.
  2. **Break lines explicitly, every row strictly under terminal width.** Never
     rely on terminal auto-wrap for row accounting — at exactly the last column,
     terminals disagree on whether the cursor advanced (deferred wrap), which
     makes cursor-up off by one. This caused a residual stale row even after
     fix 1.

  The live region is capped to 5 rows showing the newest words. The cap is
  cosmetic only: `finalize()` writes the complete sentence to scrollback.

- **Redraw debris, second cause (fixed)** — duplication reappeared, always
  starting the moment a "falling behind" warning printed. Different root cause,
  same symptom.

  `prev_rows` is the renderer's model of the screen, and **it is only valid if
  the renderer is the sole writer to the terminal.** A bare `eprintln!` while
  the live region is on screen advances the cursor behind the renderer's back;
  the next redraw then counts rows from the wrong origin and clears the wrong
  region. Three writers violated this: the falling-behind warning, the cpal
  error callback, and the sidecar (spawned `Stdio::inherit()`).

  **Invariant: nothing writes to the terminal except through `Renderer`.**
  `Renderer::notice()` clears the live region first and resets `prev_rows`.
  Off-thread writers (audio callback, sidecar stderr) send to notice channels
  that the main loop drains through `notice()`. Sidecar stderr is now piped,
  never inherited. If you add an `eprintln!` anywhere reachable while the live
  region is up, this bug comes straight back.

  The warning is also rate-limited to one per 10s and suppressible with
  `--quiet` — falling behind is self-correcting once the speaker pauses, so it
  is informational and must never flood the terminal.

- **Stale words left on screen (fixed)** — `live()` returned early when both
  word lists were empty, so anything already drawn stayed on screen until some
  later hypothesis redrew over it. Emptiness is now a state to render, not a
  reason to skip rendering. The case that originally exposed this was a spoken
  command clearing `committed` mid-utterance; the commands are gone, but the
  early return was wrong on its own terms and the fix stays.

- **Redraw debris, third cause (fixed)** — same symptom again, two more places
  where a row could reach *exactly* the terminal width and arm deferred wrap:

  1. `usable()` clamped its result **up** to a floor of 8. Below width 9 that
     handed back a budget wider than the terminal itself. Clamps on a width
     budget must only ever go down; `dimensions()` now floors the width at 2 so
     `usable()` always has a column to give that is still strictly inside.
  2. The `…` elision marker was prepended to the first retained row *outside*
     the width budget, so a full row plus the marker landed exactly on the
     boundary — at any width, not just narrow ones. `layout()` now re-wraps one
     column narrower when a marker is needed. Re-wrapping cannot flip the
     elision decision, because narrower rows only ever produce more of them.

  **Invariant, restated: every emitted row is strictly narrower than the
  terminal, including anything decorative added afterwards.**

- **Protocol desync (fixed)** — the sidecar's stdout is the protocol stream, and
  that was enforced by convention only. A single stray `print()` anywhere in the
  MLX/HF dependency tree would either kill startup or, worse, shift every later
  exchange by one — each window silently receiving the *previous* window's
  hypothesis, forever, with no error raised anywhere.

  Two independent defences, both needed. The sidecar now `dup()`s the real
  stdout to a private handle and points `sys.stdout` at stderr **before the
  heavy imports**, so stray output lands in the log. And every reply echoes the
  request's `id`, which `sidecar.rs` checks and uses to resynchronise rather
  than trusting line order. Verified by injecting a stray `print()`: it surfaced
  as a notice and the transcript was unaffected.

- **A refused window is not a dead session (fixed)** — `transcribe` used to
  `bail!` on a per-request error, which propagated out of `main` and exited,
  discarding a transcript the user was in the middle of speaking. The two cases
  are now distinct in the type: `Err` means the transport is gone and the
  session genuinely cannot continue, `Reply::Failed` means the model refused one
  buffer and the next forward pass will probably succeed. Failures are reported
  through `notice()`, rate-limited to one per 10s, and **not** suppressed by
  `--quiet` — that flag covers the self-correcting duty-cycle warning, not real
  faults. Verified by failing every second window: the full transcript still
  came out.

- **Plain text that could still change (fixed)** — the styling contract was
  broken at the seam the product is built around. On a terminal, a sentence was
  written to scrollback in plain text at the endpoint and stayed retractable for
  up to `continue_ms`, so text the display called confirmed could still be
  erased and re-decoded. Reproducible by pausing mid-thought and resuming.
  Held sentences now stay dim in the live region until the grace closes; see
  §5 *Terminal safety* for the full reasoning and what it let us delete.

  **Invariant: nothing reaches the scrollback while anything upstream can still
  rewrite it.** Dim is the only honest rendering of text that may change.

  Superseded by the stronger form in §6: nothing reaches the scrollback *at
  all* until the session ends, because `Luna, reject` means the speaker can
  rewrite text the pipeline is finished with.

- **A benchmark harness that was itself the bottleneck (fixed)** — the pipeline
  looked ~40% slower than realtime on a long passage. It was not. `simulate`
  paced by sleeping 20ms *between* sends, and `sleep` guarantees *at least* its
  argument, so the loop accumulated the send cost plus scheduler latency — which
  is large when the main thread is saturating the GPU and Python is contending
  for CPU. A 41.6s file took 58s to send. Pacing against an absolute deadline
  (`start + 20ms × n`) brought the same run to 44.7s wall against a 45.4s ideal,
  and the true duty cycle to 72%.

  **The lesson generalises: never measure throughput against a relative-sleep
  pacer.** Any timing number produced before this fix should be re-measured
  before being trusted, and the failure is silent — it reads as "the system
  under test is slow", which is exactly what you were looking for.

- **One chunk per loop iteration (fixed)** — the tick stretch keeps the steady
  state under 100% duty, but the endpoint pass, the merge re-decode and the trim
  pass all run *off* the tick and can land back to back. Each buffers tens of
  chunks, and draining one per iteration meant the loop was still behind when
  the next pass began. The loop now blocks for one chunk and `try_iter`s the
  rest. `CAPTURE_BACKLOG` (512) is the second layer: it converts a burst into
  recoverable latency instead of dropped buffers, and dropped buffers do not
  degrade gracefully (see the fragmented output above).

- **Protocol overhead is not worth optimising (measured, do not re-litigate)** —
  the entire buffer is re-encoded as base64 JSON on every tick, 66MB over a 42s
  passage. Measured round-trip minus `infer_ms` is **~7ms per pass**. Sending
  deltas, or int16 instead of f32, would be real work for noise. The forward
  pass is the cost.

- **Abbreviations split as sentences (fixed)** — `split_sentences` treated any
  terminal punctuation followed by whitespace as a boundary, and the old comment
  claiming abbreviations survive was simply wrong: `"The U.S. government said
  so."` came out as two lines, and `"I read J. R. R. Tolkien last year."` as
  four. Two guards, both narrow: a lone letter before the stop is an initial,
  and text resuming in **lower case** means the sentence is still going. The
  second tests for lower case rather than requiring upper on purpose — CJK is
  caseless and must still split on `。`.

- **Empty hypotheses poisoning LocalAgreement (fixed)** — the sidecar returns
  `""` for any buffer under `MIN_AUDIO_SEC` (0.35s), and the first tick after
  onset carries barely more than the 300ms pre-roll — measured at 0.32s for
  typical chunk sizes, so this fired on most utterances. Admitting that empty
  word list to the agreement window drove the common prefix to zero for the next
  `n` ticks, stalling commitment and blanking the provisional tail off screen
  mid-sentence. `push_hypothesis` now discards it: an empty hypothesis is
  absence of evidence, not evidence of silence.

- **The merge threw away the trim's cut candidates (fixed)** — reported as
  sentences vanishing from the screen once the buffer passed the threshold, then
  as `Luna, copy` copying nothing and the session file being empty. Three
  symptoms, one cause: **the document was empty**, because the trim is the only
  thing that files text mid-passage (see §6, *The file holds the utterance in
  flight too*) and the trim was being starved of places to cut.

  `dips` were cleared at every endpoint, including when the audio was retained
  in `Pending`. A continuation puts that audio back at the *front* of the merged
  buffer with its pauses exactly where they were, but the record of them was
  already gone — so a merged buffer offered only its own seam, one candidate per
  merge, however many sentences it held. §5 merges every ordinary pause, so this
  is the normal path in a real session, not an edge case. `Pending` now carries
  its dips and the merge splices them back at their original offsets.

  **The lesson worth keeping is the coupling, not the bug.** The trim was
  documented as a latency control — "what is bounded is how far behind the
  transcript falls". It is also the *only* thing that gets text out of process
  memory during a long passage, so starving it silently disabled the crash
  safety net. Anything that changes trim policy has to be read as changing
  persistence policy too.

  Diagnosis note, because it cost two wrong answers: the first reading was
  "renderer starves the document" (`build_rows` draws the whole live tail before
  any document rows, which is real but was not this), and the second asserted
  the text was safe on disk. Both were reasoning without a repro. The repro is
  cheap — `say` a passage with `[[slnc 1000]]` between sentences so every gap
  merges, run it under `--simulate --persist`, and poll the session file. Do
  that first.

---

## 8. Layout

```
src/main.rs        orchestration: window scheduling, tick loop, trim, commands
src/audio.rs       cpal capture, downmix, resample to 16kHz; --simulate path
src/vad.rs         earshot VAD + RMS floor, open/close hysteresis, dip runs
src/transcript.rs  LocalAgreement-n window + the sentence document
src/repair.rs      self-repair detection ("regularing… regular")
src/command.rs     wake-word commands + repeat debounce
src/store.rs       session autosave + clipboard
src/text.rs        shared word/sentence helpers
src/render.rs      alternate-screen live document, wrap-safe row layout
src/sidecar.rs     subprocess handle + NDJSON protocol
sidecar/asr_sidecar.py   MLX inference; stdout is protocol, stderr is logs
```

Testing terminal output: pipe through a pty with
`script -q out.raw ./target/release/speech-to-text-cli --simulate f.wav`, then
replay the escape stream through a screen emulator. Redirecting stdout is not
enough — the renderer detects a non-TTY and skips the live view entirely.

Frames are separated by `\x1b[1;1H` and every row is `\x1b[{r};1H\x1b[2K` plus
content, so a ~20-line emulator is enough to check any of it. Two things worth
asserting on the captured stream: that the final transcript appears **after**
`\x1b[?1049l` (i.e. in the real scrollback, not the alternate screen), and that
the gutter/dim ladder matches the state you expect.

Synthesizing test speech, which is how the command and trim cases were verified:

```sh
say -v Samantha -o t.aiff "A sentence. [[slnc 1500]] Luna, commit."
afconvert -f WAVE -d LEI16@16000 -c 1 t.aiff t.wav
```

`[[slnc <ms>]]` inserts a real pause, which is what exercises the endpoint, the
merge and the settle boundary. Note that `say` will not reliably produce a
*disfluency*: "regularing regular" comes back as `regular-irregular`, one
hyphenated token, so `repair.rs` cannot be exercised this way.

Dependency notes:
- `rubato` pinned to **0.16.2** — 5.x reworked its API around `audioadapter`.
- `cpal` 0.18 renamed `Device::name()` to `description()`, and `SampleRate` is
  now a plain `u32`.
- `earshot` needs exactly **256-sample** frames at 16 kHz.

## 9. Status

MVP working end to end. Verified via `--simulate` on 3 utterances: all
transcribed correctly, VAD endpointed each one, provisional text visibly
converged to committed.

Live mic path exercised with real speech. Transcription quality was good; the
apparent "duplicated sentences" reported from that session were a rendering bug
(see §7), now fixed and covered by tests.

Self-repair works (§6). Verified end to end via `--simulate` on synthesized
speech: a backtracked word is replaced, a long utterance carrying one comes out
byte-identical to the same sentence recorded without it, and six control
sentences that merely resemble repairs are untouched.

**Live document, spoken commands and the pause-aligned trim all work**, verified
end to end on synthesized speech at the current defaults (99 unit tests green):

| case | result |
|---|---|
| `"First sentence. Luna, commit."` | filed and marked kept; seam comma closed to a full stop |
| `"…should disappear. Luna, reject."` | dropped; document empty |
| `"First sentence. Luna, commit. Luna, reject."` | **a committed sentence is dropped** — the requirement that forced the document model |
| `"Luna, clear."` then `"Luna, rollback."` | 2 removed, 2 restored with commit state intact |
| `"Luna, copy."` | the whole document lands on the real system clipboard, and is marked kept |
| `"Luna, rejects."` | fires — and `--assistant Jarvis` on the same audio shows the model really did write `Luna rejects.` |
| `"Luna, reject."` twice, 1.7s apart | second suppressed with a notice; one sentence dropped |
| 20s of synthesized typing over room tone | nothing transcribed, at `--open-ms` 48 and 150 alike |
| `--assistant Jarvis` | retargets every command and the banner |
| `"Luna went to the store."` | left as dictation |
| 4s gap | merged, one sentence, comma across the seam |
| 8s / 12s gap | two sentences, no merge |
| 42s unbroken passage | trimmed at pauses, 7 clean sentences, no word cut, 72% duty |
| `kill -9` mid-session, no `--persist` | settled text recovered from the session file |
| `kill -9` at T+20s of a continuously merging passage, before any trim | 4 sentences recovered, split one per line, with stdout empty as designed |
| 41s passage, 1s pauses throughout (every gap merges) | file tracks the passage from T+6s; before the fix it held nothing until the first trim at ~T+33s |
| 80s of continuous speech, `--trim-after-s 10` | buffer peaked at 13.3s against the 20s bound; transcript clean |
| 63.6s buffer, token budget vs the 8192 default | byte-identical text, 1.82s vs 1.81s — the cap never binds on real speech |
| `"…will be dropped. Luna, reject. Luna, rollback."` | both fire; the sentence is dropped and restored — `rollback` was unreachable before the vocabulary hint |
| command list as `system_prompt`, 20s noise / 41s dictation / controls | no echo, byte-identical, no false command |
| next launch after that kill | announces the orphaned session, naming it and its size |
| `--resume` | recovers it, continues dictating, **adopts the file in place** — still one file |
| `--resume ~/mynotes.txt` | loads it, leaves it byte-identical, opens a new session file |
| `--resume` with nothing to resume | fails in 0.09s, exit 1, before the model loads |

Not yet done:
- **Arbitrary substitutions.** "Tuesday… Wednesday" cannot be detected — the
  words share nothing orthographically. `Luna, reject` does not recover it:
  it operates on sentences, not words. See §6 *What this cannot do*.
- **Cross-utterance repair.** A repair only reaches words in the current
  window. After a pause the §5 merge usually puts them back in it, but once the
  grace closes the false start is out of reach.
- Repetition-loop detection (compression-ratio heuristic). Now more relevant
  than it was: buffers routinely reach 30s, which is where long-generation
  failures live.
- **No scrollback in the live view.** The document can grow past the screen and
  older sentences scroll out of sight. They are still in the document, still
  rejectable and still printed at exit — but there is no way to look at them.
- **The trim seam is still a seam.** Cutting at the longest pause makes it land
  at a sentence boundary most of the time, but "most of the time" is not always,
  and nothing re-punctuates across it the way the §5 merge does.
- **Undo does not reach a `commit`.** Only text removal is recorded, so there is
  no way to un-keep a sentence short of rejecting and re-dictating it.
- **Session files accumulate under `--persist`.** Nothing prunes the state
  directory, and `--resume` only ever offers the newest one — an older orphan is
  reachable by path but is never mentioned.
- **Resume restores text, not session state.** Commit marks, the undo stack and
  the audio are all gone, so a resumed session cannot undo what the previous one
  did.
- **`copy` is macOS-only** (`pbcopy`), which matches D2 but would need replacing
  if the pipeline ever left Apple Silicon.
