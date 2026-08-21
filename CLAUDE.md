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

**"Stabilizes" means *files*, not *agrees*.** Text used to go plain the moment
LocalAgreement agreed it, mid-utterance, and that was withdrawn on report — see
§6, *Where plain begins*. Everything in flight is dim; the transition the
speaker reads is the one into the document, which happens once and never
reverses.

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
| D8 | **The transcript is a live document, not scrollback** | `delete` has to reach a sentence filed minutes ago, and scrollback is the one region a program cannot take back (§6) |
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

**Duty cycle is no longer the constraint to watch, and saying so was hiding the
one that is.** It scales with buffer length only until the tick stretch engages,
and after that `tick = infer × 1.25` pins it at exactly 80% *by construction* —
so the number cannot move, and a notice keyed on it fires whenever inference
merely grows. Measured over a 109s session, the old "refresh slowed" condition
was true for **48% of it**, on a pipeline that was keeping up throughout.

What actually degrades is **commit latency**, `(n−1) × tick + inference`, which
grows on both terms with the buffer, is what the speaker sees, and had nothing
watching it: over that same session it exceeded D6's 2s target for **36% of the
time**, peaking at 3.06s. `main.rs` now reports that instead, and phrases it as
something the speaker can act on — pausing endpoints the utterance, which files
it and starts the next one short.

Buffer trim policy is therefore load-bearing, and load-bearing for latency
rather than for duty. Cut at committed sentence boundaries, preferably at
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
| < ½ × `--trim-after-s` | *(no trim)* |
| ≥ ½ × `--trim-after-s` | ≥ ~480ms — a sentence boundary, nothing less |
| ≥ `--trim-after-s` | ≥ ~190ms — D9's quality bar, unchanged |
| ≥ 2 × `--trim-after-s` | any recorded silence, longest still preferred |

A ~64ms silence can be the closure of a stop consonant rather than a word
boundary, so cutting there really can clip one. That is the trade being made
knowingly: a bad cut is a bad sentence, an unbounded buffer is not an outcome at
all. It only ever *adds* candidates in a situation that had none, and the
longest-gap preference is untouched, so nothing changes for a speaker who pauses
normally. Verified with `--trim-after-s 10` on 80s of continuous speech: the
buffer peaked at **13.3s** against the 20s bound.

### Trimming early at a good gap, rather than late at whatever is left

The band at **½ × `--trim-after-s`** is new, and the argument for it is that
waiting was buying nothing. Between 15s and 30s the old policy refused to cut at
all, while a sentence-grade gap was available the entire time — measured over a
109s session, **50% of 202 recorded gaps were ≥480ms, and every one of the 21
snapshots taken had one**. So the buffer was being carried to 30s, and the
latency paid for it, purely because nothing was looking yet.

The bar in that band is *higher* than the one a late trim settles for, which is
what makes it safe: it can only ever produce a cut better than the one that
would otherwise have happened at 30s.

Measured on the same 109s passage, everything else default:

| | old | with the eager band |
|---|---|---|
| buffer, median / max | 13.7s / **30.0s** | 9.4s / **15.0s** |
| commit latency, median / max | 1.61s / **3.03s** | 1.33s / **1.94s** |
| time past D6's 2s target | **35%** | **0%** |
| duty cycle | **81%** | **49%** |
| trims | 3 | 7 |
| transcript | — | byte-identical |

**Duty went down while the trim count went up**, which is the result to hold on
to: more trims are more forward passes, but every one of them is on a shorter
buffer, and inference grows faster than linearly in buffer length. Trimming more
often is *cheaper*, not dearer.

### Longest gap still wins, and latest-wins was tried and measured worse

Cutting at the longest gap looks wrong once the buffer is what you are trying to
bound, because the **merge seam** wins that contest by construction — the
reconstructed pause is capped at `MAX_GAP_SAMPLES` (1.2s), longer than
`--endpoint-ms` permits any within-utterance gap to be — and it sits wherever
the last merge happened. Measured, that took 9.8s off a 30.3s buffer and left
20.5s standing; two of three trims in one session did the same.

So latest-wins was tried, guarded by the ≥480ms floor, on the argument that once
every candidate is sentence-grade, position is free to decide. **The argument
was wrong.** Same audio, five rules:

| rule | buffer max | commit max | seams |
|---|---|---|---|
| longest, ≥480ms | 15.0s | 1.94s | clean |
| longest, ≥190ms | 15.0s | 1.93s | clean |
| latest, ≥576ms | 14.9s | 1.85s | clean |
| **latest, ≥480ms** | 14.8s | 1.81s | **`The measurement should come first,`** |

That stranded entry is the exact signature recorded above for most-recent-wins.
A 480ms floor does not separate a comma from a boundary as cleanly as the
~200/~500ms rule of thumb implies — the speaker had left ~500ms after a comma.

The decisive column is the first: **every rule bounds the buffer identically.**
The eager band did all of it and the selection rule contributed nothing, so
there was never anything to buy by reversing a measured decision. The ≥480ms
floor is kept as insurance rather than tuning: with longest-wins selecting,
dropping it to ~190ms produced a byte-identical transcript.

### Where to cut was only half the question

Every rule above decides *where* in the audio to cut. None of them asked whether
the text either side of the cut had been settled, and that is what produced the
failure reported live at the shipped defaults:

```text
It tries to solve the problem.
Of coordination between participants, a single node can communicate with the other nodes.
```

One spoken sentence, one deliberate mid-sentence pause, two document entries.
The capitalised `Of` is the tell — it is a fragment that was decoded **alone**
and punctuated as though it stood by itself, which is §5's `say,` -> `Say,`
finding happening at a seam the pipeline created.

**The trim did it, and by construction rather than by bad luck.** The merge seam
is the longest gap in the buffer *always*: `scan` stops at an endpoint, so no
within-utterance gap can exceed `--endpoint-ms` (36 frames), while a seam
carries the reconstructed pause (up to `MAX_GAP_SAMPLES`, 75 frames). So
longest-wins selects it every time, and §5 arranges for every ordinary pause
to merge — so this was the normal trim, not an edge case. The trim then decoded
`window[..cut]` **alone** and filed whatever came back.

Measured on the reported sentence, same audio, three decodes:

| decoded | text |
|---|---|
| whole buffer | `It tries to solve the problem of coordination between participants. A single node can communicate with the other nodes.` |
| head alone, cut at the pause | `It tries to solve the problem.` |
| tail alone | `Of coordination between participants, a single node can communicate with the other nodes.` |

The middle row is the reported first line and the bottom row is the reported
second line, character for character. So the pipeline was asking the model
exactly the right question — the whole buffer, merged, prosody intact — and then
throwing that answer away in favour of the answer to a worse one.

### The head decode is a question now, not a filing

The trim still decodes `window[..cut]`, and still costs that forward pass, but
what it does with the result is inverted:

- the text is compared against the document (`Transcript::unfiled`);
- **zero words unaccounted for** — every word of the head is already filed, from
  a decode that had the whole buffer in front of it — the audio is spent, so it
  is cut away and **nothing is filed**;
- **anything else** — the cut would strand text that has never been filed, so
  the trim gives up this tick and the buffer grows instead.

Nothing about *where* changed. What changed is that the trim stopped being a
filing path at all: on the ordinary path it now only ever discards audio whose
text is already in the document, and logical commitment (§6) — which reads the
whole-buffer decode — is what files.

Reproduced end to end under `--simulate`, at the shipped defaults, on a passage
with a 1.6s mid-sentence pause 17s in:

| | transcript around the pause |
|---|---|
| before | `…and it tries to solve the problem.` / `Of coordination between participants, …` |
| **after** | `…and it tries to solve the problem of coordination between participants.` |

The trace records the moment: `trim-refused  cut at 17.50s would strand 60
word(s)`.

**The cost is latency, and it is not small.** On a 37s passage of ten sentences,
same audio, everything else default:

| | before | after |
|---|---|---|
| transcript | ground truth | ground truth — one numeral differs, `three` vs `3` |
| trims taken | 3 | 2 |
| trims refused (a forward pass each) | 0 | 9 |
| commit latency past D6's 2s target | never | once, 2.0s at a 20s buffer |

On the mid-pause repro the buffer ran to 27s where the old code cut at 17s, and
commit latency peaked at 2.9s against 2.1s. That is the trade taken knowingly
and it is the same one D9 states: **latency is recoverable and a filed fragment
is not.** A refused trim costs seconds of lag that any pause gives back; the cut
it refused costs a sentence that can never be repaired, because plain text never
changes.

Two things bound the cost:

- A refusal is **remembered on the gap itself** (`Dip::refused`), keyed on
  `Transcript::revision`, so the same cut is not re-probed every tick — only
  when the document moves and the answer could have changed. Holding it beside
  the buffer instead, and resetting it at each endpoint, cost 17 refusals where
  9 would do, because §5 merges every ordinary pause and each merge restarted
  the walk.
- A refusal takes **everything after it** out of contention as well as the gap
  probed, since a later cut severs the same sentence and more of it. So the trim
  walks backwards through candidates, one pass per tick, and each pass is on a
  shorter head than the last.

### The backstop, and what it still costs

Refusing is only bounded while something else eventually files the text.
Logical commitment does, on ordinary dictation. It does **not** for a speaker
producing one very long sentence, or one who keeps naming the assistant (which
makes `commit_settled` decline), and there the refusal has no bound at all —
which is the 63.6s buffer above, the one that took ~67s in a single forward pass
and stopped responding for a minute.

So past **2 × `--trim-after-s`** (60s at the defaults) the trim gives up and
files the head decode, exactly as it used to. That is the same shape of argument
as the desperate band and deliberately the same number: by then the pipeline has
already stopped holding out for a good place to cut, and holding out for
well-founded text on top of that would be insisting on quality in a situation
that has run out of it.

**This is the one path that can still produce the reported damage**, and it is
traced as `trim-forced` for exactly that reason. Verified by making it fire, with
`--trim-after-s 8` so the ceiling is 16s:

```text
So we should review the metrics before.
The end of the week, because I think the.
```

That is the failure, reproduced on demand. At the default the same passage needs
60s of continuously un-filed speech to reach it. Which speakers reach it is not
known without real dictation: a speaker whose sentences run 20s each produces
the three sentences commitment needs only after ~60s.

### Measured duty cycle, end to end

On the 42s passage above, at `--trim-after-s 30` with everything else default:
**63 passes, 29.8s of inference for 41.6s of audio — 72% duty**, of which only
1.4s was off-tick. Wall time 44.7s against an ideal of 45.4s. So the pipeline
sustains realtime with ~28% headroom at a 30s trim threshold.

Re-measured on the 109s session above with the eager band in place: **49% duty**,
wall time 112.1s against 109.2s of audio. The headroom roughly doubled, and the
cause is entirely the shorter buffer.

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
warmup before the sidecar reports ready. Warmup runs **twice** — once with the
command hint and once without, because §7 now uses both prompt states and a
command is the worst place to pay graph compilation. So budget **~4.5s** of
startup, not 1s.

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
│  Qwen3-ASR-1.7B → hypothesis (no prompt; hint only on ask)  │
└───────────────────────────────┬─────────────────────────────┘
                                ↓ NDJSON / stdout
┌───────────────────────────────┼─────────────────────────────┐
│  LocalAgreement-3 committer   ↓                             │
│      └→ window state (agreed | provisional)                 │
│          └→ command parse (wake word) ──→ delete / undo   │
│              └→ document (finished sentences)               │
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
is gone**, and deliberately: the hold is the same for every utterance, and only
its *length* varies (see `Settle`, below). It existed to limit pointless
re-decoding when a settled sentence had already been printed to scrollback, and
nothing is printed any more (§6) — a settling sentence just sits dim in the live
document, so leaving it open costs nothing worth guessing about. Do not
reintroduce it to save recompute; the measurement below says recompute was never
the expensive part.

**And the measurement has since got much sharper.** The shredded passage below
came back as six fragments, and **every one of them ended in a full stop** —
`Okay, so let's speak about the politicians.`, `See.`, `To be doing very well.`
So this is not a heuristic that is merely weak; on the exact input where a
completeness test would have to work, it has no signal at all. Any future
attempt to decide "has this thought finished?" from the text of one utterance
should start by explaining what it does differently, because punctuation is
settled: it cannot.

`Settle` is not that heuristic returning by another door. It never reads the
text; it measures the *speaker*, and it only ever adjusts how long to wait
before filing — never whether the text looks ready.

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

### The seam length does not decide the boundary — measured, and it supersedes the table above

The table immediately above says the model reads a ~1.4s silence as a sentence
boundary and a ~0.9s one as a comma. **That does not reproduce under a
controlled test, and the mechanism it implies is not there.**

It matters because it was the leading candidate for the failure in §3. The seam
the model actually receives is 892–2092ms — trailing silence inside the held
audio (`close_frames`, 592ms) plus the spliced gap (0–1200ms) plus the retained
pre-roll (≤300ms) — so on that reading an ordinary ~1.2s hesitation would land
above the split threshold and be *instructed* to become two sentences. The fix
would have been to normalise every seam to one fixed length, which the pipeline
can do exactly, since it synthesises the silence itself.

It was gated on a probe first. Identical speech either side, seam length the
only variable, `Qwen3-ASR-1.7B-8bit`:

| audio | 0 / 300 / 600 / 900 / 1400 / 2092 / 4000 ms of seam |
|---|---|
| `"It tries to solve the problem"` + `"of coordination between participants, …"`, spliced | **byte-identical at every length** — fused, one sentence |
| the same passage from one continuous synthesis, its own pause shortened in post | **byte-identical at every length** |
| `"This is the first complete sentence."` + `"And this is a totally separate one."` | **byte-identical** — a comma at every length, including 2092ms |
| the same pair from one synthesis with `[[slnc 2000]]` | **byte-identical** — a comma at every length |
| `"The deployment finished at noon."` + `"My cat is asleep on the keyboard."` | **byte-identical** — two sentences at every length, including 0ms |

Five conditions, nine lengths, and the boundary never moved once. The third and
fourth rows are the table above's own example, at 2.2s of silence, coming back
with a comma.

So the boundary is **prosodic and semantic, not durational**, which is the
stronger form of what §5 already says: the model arbitrates, and it arbitrates
on what it hears in the speech rather than on how long the gap is. That is
consistent with the row above it — merging *unrelated* sentences still returns
two sentences, at any seam length, including none at all.

**What the original table probably measured** is the pipeline, not the model: it
compared a file with a real pause against *the merge path*, which differs in
more than silence length. Comparing two things that differ in two ways and
attributing the difference to one of them is the error, and it is worth naming
because the fix it justified — `gap_samples`, `MAX_GAP_SAMPLES` — is still in
the code.

**Nothing was changed on the strength of this.** The seam is still reconstructed
exactly as described above. A null result licenses *not* making a change; it
does not license making the opposite one, and normalising the seam would cost a
re-measurement of §3's trim policy to buy an effect measured at zero. What it
does license is disbelieving the mechanism: **a mid-sentence pause of any length
is not what splits a sentence here, and anyone reaching for `MAX_GAP_SAMPLES` to
fix a split boundary should read §3 first.**

Caveat, stated because it is the whole weakness of the probe: this is
synthesized speech, whose prosody is clean and unambiguous by construction. A
real hesitation is precisely the ambiguous case where a weak durational cue
could still tip a decision the model is otherwise unsure about. The probe rules
out a strong dependence, not a marginal one. It needs re-running on a recorded
human pause before the caveat can be dropped.

**Consequence: generous `continue_ms` values are safe.** Verified against known
ground truth at 0 / 2000 / 4000 / 8000 ms — continuations fuse, separate
utterances stay separate, transcription accuracy unchanged. Raising it costs
recompute and on-screen movement, not correctness.

**It went 4000 → 10000 → 6000, and 6000 was wrong.** The number was tuned
against how quickly text goes plain, on the reading that it was one number doing
two jobs pulling in opposite directions:

- it is how long text stays **dim** — shorter is more responsive;
- it is how long a **thinking pause** may run before one thought becomes two
  sentences — longer is more forgiving.

The second job is not one a duration can do, and it never was. That is now
`Settle` in `main.rs`, and the default floor is **15000ms**.

### The settle is a display parameter, and treating it as a linguistic one shredded a passage

Reported live, and reproduced exactly. A speaker thinking out loud with ~7s
pauses, dictating a passage that is two sentences:

| | transcript |
|---|---|
| filed on the 6s timer | `Okay, so let's speak about the politicians.` / `And politics, which I rather feel.` / `Is a little bit off the rails.` / `I wonder if this discrepancy.` / `Works well so far because it doesn't seem.` / `To be doing very well.` |
| **audio held together** | `Okay, so let's speak about the politicians and politics, which I rather feel is a little bit off the rails.` / `I wonder if this discrepancy works well so far, because it doesn't seem to be doing very well.` |

The second is verbatim ground truth — right fusion, right separation into two,
right commas.

**Read the capitalisation in the first row.** `And`, `Is`, `Works`, `To` are each
a fragment that was decoded *alone* and therefore punctuated as a whole
sentence: this section's own `say,` → `Say,` finding, happening six times in one
passage. So filing early does not merely put the sentence boundaries in the
wrong place. It damages the words either side of every boundary, permanently,
because plain text never changes.

**The margin was 300ms.** A 7s speaking pause reaches the settle as a ~6.3s
gap — `--endpoint-ms` is spent detecting the silence before the hold even
starts — against a 6000ms window. The speaker was on the wrong side of a cliff
by a rounding error, which is why the damage was total rather than occasional.

### Why holding is the safe direction, and why it is nearly free

The two ways to be wrong are not comparable:

- **Hold too long** — text stays dim a few seconds more. The store already
  writes in-flight text (§6), so a crash takes nothing either. Recoverable.
- **File too early** — the passage is cut mid-thought and the surface forms
  around the cut are wrong. **Not** recoverable.

So the policy leans hard toward holding. What makes that affordable is that the
timer is *unreachable on ordinary dictation* — every ordinary pause is already
shorter than the floor, so it never fires at all. Measured over a normal-paced
passage, settle as the only variable:

| | 6s settle | 600s settle |
|---|---|---|
| transcript | — | **byte-identical** |
| buffer max | 14.8s | 14.9s |
| commit latency max | 1.70s | 1.66s |
| forward passes | 33 | 33 |
| inference total | 10.4s | 10.4s |

Nothing moves, because nothing reaches the timer. Raising the floor changes
behaviour only for the speaker it was damaging.

The slow passage above does pay for it, and the bill is small: buffer max 2.9s →
14.2s, inference 3.8s → 7.0s over 47s of audio, commit latency max 1.17s →
1.37s — still inside D6's 2s target.

### It adapts, because a fixed floor is not enough

A larger fixed number is paid by everyone every time they stop for good. Worse,
it does not actually solve the problem — it just moves the cliff. Measured on a
passage with 20s pauses:

| policy | entries |
|---|---|
| fixed 6s | 4 fragments |
| **fixed 15s** | **4 fragments** — the cliff moved, nothing else |
| 15s floor + adaptive | **2** |

So `Settle` learns. It remembers the last five silences the speaker went on
talking through and holds for the longest of them plus half again, clamped
between `--continue-ms` and `--continue-max-ms` (default 60s). Traced on that
same passage:

```text
16.843 EXPIRE   held=15.01s settle=15.00s     ← the first pause is lost
16.843 FILE     "The first part of the sentence."
21.021 RESUMED  gap=19.18s settle_now=28.77s  ← and pays for every one after it
42.180 RESUMED  gap=19.41s settle_now=29.12s
42.181 MERGE    held_audio=1.78s
63.843 RESUMED  gap=19.29s settle_now=29.12s
63.844 MERGE    held_audio=5.24s
```

**One fragment is still lost, and that is inherent**: a pause is only measurable
once it ends, so the first one past the floor cannot be held. Everything after
it is.

**It learns from every resumption, including the ones it got wrong.** The gap is
recorded whether or not the hold was still open when speech restarted. Learning
only from *merged* pauses would mean never learning about a pause longer than
the current settle — which is the only kind that has ever caused damage, and is
precisely the blindness §7 records the deleted `mentions_wake` gate having.

A zero floor (`--continue-ms 0`) switches adaptation off: it is an explicit
instruction not to hold, and adapting past it would be overriding the speaker.

**Consequence, unchanged and now load-bearing twice over:** every ordinary pause
merges, so a dictation session merges continuously and the buffer never resets
on its own. That is what makes the §3 trim load-bearing rather than a nicety —
without it, a ten-minute session would be a single 600s window.

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
  word in it is provisional again. **Superseded, and in the direction this
  sentence points**: if the endpoint makes every word provisional again, the
  words were provisional all along, so nothing in flight is plain any more and
  there is no downgrade left to explain. See §6, *Where plain begins*.

**This argument was later followed one step further.** If plain text must not
appear while the *pipeline* can still rewrite it, then a `delete` command that
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
The **document** is a `Vec<Sentence>`: what the speaker has finished saying.

A `Sentence` used to carry a second `committed: bool` — a claim about the
*speaker*, set by `Luna, commit`. That flag is gone; see *The approval flag, and
why it was decay rather than design* below.

### Logical commitment: a sentence files when other sentences follow it

**What files a sentence is its position in the transcript, not the clock.**
`Transcript::settled_prefix` keeps the last [`KEEP_SENTENCES`] (2) of the window
and files everything before them; `main.rs::commit_settled` runs it after every
tick.

The rule is one sentence long: **a sentence is settled once enough further
sentences exist behind it that the model has demonstrably stopped revising it.**

It is deliberately *not* a completeness test on the sentence itself. §5 measured
that the model punctuates whatever it is handed — the six fragments of the
reported failure each ended in a full stop — so terminal punctuation says
nothing about whether a thought has finished. What it marks reliably is where
the model *chose* a boundary, and a boundary that survives more speech arriving
behind it has had all the right-context it will ever get. §7 has the numbers:
60 re-checks, 4 revisions, all four `3` ↔ `three`.

Two consequences worth knowing before touching it:

- **The audio stays.** Filing is a claim about text, not a reason to discard
  anything, and `Transcript::filed` strips the re-decoded copy off every later
  hypothesis. Cutting the audio is the trim's job and is measured to be
  destructive when done early (§7) — these are different operations and the
  distinction is the whole reason this works at all.
- **It declines when the assistant is named.** A command must be parsed from a
  finalized utterance and never from a hypothesis (below), and filing early
  would put `Luna, delete` in the transcript as dictation, permanently, before
  the endpoint ever read it as an instruction. `command::names` is the guard;
  it is coarser than `has_command` because a command can straddle the boundary
  between what is settled and what is not.

Measured on a 37s passage of ten sentences, against filing by trim alone:

| | trim only | with commitment |
|---|---|---|
| first sentence plain at | 17.6s | **14.9s** |
| document grows | in batches of 2 | **one sentence at a time** |
| recovered by `kill -9` at T+25s | 2 sentences | **4** |
| transcript | ground truth | **byte-identical** |

The last row is the one that matters: this buys cadence, not accuracy. The
transcript was already correct.

`KEEP_SENTENCES` is 2 rather than 1 on the measurement's own terms. One is
enough for stability, and 1 was tested and produces ground truth on every
passage here — but the model punctuating fragments means the last "sentence" in
the window is usually the one still being spoken, so keeping 2 is what actually
delivers the one *complete* sentence of right-context the measurement vouches
for. It costs one sentence of dim time against an irreversible operation.

#### Commitment is the only path that files mid-passage now

The trim used to file as well, and **that filing was the reported failure in
§3**: it decoded the severed head and put whatever came back into the document.
It no longer files on the ordinary path at all. The paths that can reach the
document are now:

| path | what decoded the text it files |
|---|---|
| logical commitment | the whole buffer |
| the settle expiring, or an endpoint with nothing to hold | the whole utterance |
| a spoken command's dictation segments | the whole utterance |
| end of stream | the whole buffer |
| the trim, past `2 × --trim-after-s` only | **the severed head** — the backstop, §3 |

Every row but the last files text produced with all the right-context there was.
That is the property to preserve when adding a sixth.

**Two consequences that are easy to miss.**

A command can no longer be executed by a trim. It used to be possible: the trim
ran `resolve` and `apply` over a head decode, which is not a finalized utterance
however much it resembles one. Commands are now parsed at endpoints and at end
of stream only — the rule §6 states — with the forced backstop the one
exception, and it inherits that with the rest of the old behaviour.

And §7's "the trim is the only thing that gets text out of process memory during
a long passage" is now **false, and its safety argument has to be re-derived
rather than assumed**. What carries the passage is logical commitment plus the
in-flight line the store writes (§6, *The file holds the utterance in flight
too*). Verified after the change: `kill -9` at T+20s of a continuously-merging
passage still recovers everything spoken up to ~2s before the kill.

#### Knowing which path filed a sentence

Six paths reach the document and a transcript does not say which one produced a
given line. That cost two wrong diagnoses of the §3 failure before it was
traced, so `STT_TRACE=<file>` now records one line per filing, per trim
decision, and per notice:

```text
14.615  logical    The deployment finished at noon, and everything looked stable.
17.231  trim-refused  cut at 7.43s would strand 11 word(s)
18.580  trim       cut at 3.12s, head already filed
40.784  eos-held   The database migration is scheduled for Saturday morning.
```

It writes to a **file**, never to the terminal, and that is a correctness
requirement rather than a preference: §7 records a redraw bug whose cause was a
bare `eprintln!` while the live region was up. Unset, every call is an atomic
load and a return.

### The sentence a fragment cannot be

`text::split_sentences` refuses a boundary when the text after it opens with a
word that cannot begin an English sentence — `of`, `which`, `whom`, `whose`,
`than` — unless it is an idiom (`Of course`, `Which is why`) or set off by a
comma within two words (`Of course,`).

This is **insurance, not the fix** for §3's failure, and the distinction is
worth keeping straight: the two halves of that report came from two different
forward passes, so they never met in one string for this to see. What it catches
is the same damage arriving inside a single decode.

**It is not `looks_incomplete` returning by another door**, which §5 forbids.
That heuristic read the text *before* a boundary and guessed whether the speaker
had finished a thought — a question about intent, which §5 measured the model
gives no signal for. This reads the text *after* a boundary and asks whether the
string can grammatically **be** a sentence. "Of coordination between
participants" is not an unfinished thought that might yet be finished; it is a
prepositional phrase, and no continuation makes it a sentence, because what it
modifies is on the other side of the split. That is a property of the string,
decidable from the string.

The list is five words long, and what is missing from it is the argument.
`to`, `for`, `with`, `from`, `by`, `in`, `on`, `at` all look like members and
are not, because fronting an adverbial is ordinary English — "For now we wait.",
"By Friday the release ships." `and` and `but` are missing for the stronger
version of the same reason: they open spoken sentences constantly. A false
positive merges two real sentences into one document entry, costing `delete`
granularity and a sentence of filing delay; a false negative leaves a permanent
fragment. Both costs are real, which is why the list is short rather than empty.

### Filing is still the caller's decision at an endpoint

`finalize_window()` deliberately does **not** file into the document. The caller
decides, because an utterance that has merely ended may still be a hesitation
(§5); filing there would mean un-filing on merge. That is also why
`retract_last_sentence()` is gone — a pending sentence is no longer in the
document to retract.

**Scrollback is the one region a program cannot take back.** `Luna, delete` has
to reach a sentence the speaker already committed, so the transcript cannot be
appended to the scrollback while the session runs. It lives on the **alternate
screen**, redrawn from the model every frame, and is written to the real
scrollback exactly once, on exit — the moment it genuinely stops being editable.

The styling contract survives this intact, and it is worth being precise about
why. It was always a promise about the *pipeline*:

| rendering | meaning |
|---|---|
| dim | the pipeline may still rewrite this |
| plain | settled; only the speaker can change it now |
| `│` gutter | the utterance in flight, as against the transcript |
| `⋮` gutter, top row | more transcript above than the screen holds |

The gutter is deliberately *not* on the same axis as the styling. It answers "is
this the sentence being spoken right now?", and it is now the only thing
distinguishing the agreed head of the live tail, which is dim like the rest of
it.

#### Where plain begins, and why it moved

The boundary used to sit at **LocalAgreement**: the agreed head of the live tail
rendered plain while the provisional remainder stayed dim. It now sits at the
**document**. Everything in flight is dim; text goes plain when it files.

Reported live as text appearing white and then turning grey again on the next
breath. The sequence looks like a rendering fault and is not:

1. speaking — LocalAgreement promotes the head, which renders plain;
2. the speaker pauses — the utterance moves to `Pending` and renders wholly dim;
3. they resume inside `--continue-ms` — the audio merges, `finalize_window` has
   already emptied `committed`, and the re-decode starts from nothing.

**The dim was never the wrong half.** Plain claims the pipeline is finished with
a word, and inside an utterance that claim is false by construction: the §5
merge re-decodes the *entire* buffer and can rewrite every word in it, which is
precisely why the merge happens at the audio level rather than the text level.
The promotion at step 1 was the lie; steps 2 and 3 were the pipeline telling the
truth late.

The cost is real and is not argued away: a word can no longer be watched
stabilising inside an utterance. That signal was only ever trustworthy for an
utterance that never merged, and §5 arranges for almost every ordinary pause to
merge — so it was a promise kept sometimes, which is not a promise. Held
absolutely it is worth more: text moves dim → plain exactly once, in one
direction, and never back.

**LocalAgreement is untouched as data.** This section's own first line draws
that distinction — committed versus provisional is state, not styling — and only
the styling was taken away. `committed` and `provisional` still arrive at the
renderer as separate slices, and the state machine still depends on the
difference.

Consequence worth knowing before tuning anything: **filing is what promotes, so
whatever files decides when text goes white.** The trim used to be the only
thing that did mid-passage — §5 merges every ordinary pause, so an utterance
never settles on its own — which made `--trim-after-s 30` show nothing plain for
the first ~30s of continuous speech, and made trim cadence a display property as
well as the latency control §3 treats it as.

Logical commitment (above) is now the main one, and it promotes on sentence
structure rather than on buffer length: measured on a 37s passage, the first
sentence went plain at 14.9s instead of 17.6s and the document then grew one
sentence at a time rather than two at a time. The trim still files whatever is
left over when it cuts, so the coupling §7 records for persistence is unchanged
— it is simply no longer the only path.

The last of those is about the screen rather than the text, and it exists
because its absence cost two wrong diagnoses. **The live tail is never capped**
— a speaker who has not finished has to read what they are saying (§1) — so on
a long buffer the document scrolls out of view. It used to do that silently,
which is indistinguishable from the document being empty, which is what it
actually was. Nothing is hidden to make room for the marker: it replaces a
gutter that was already budgeted, so it cannot widen a row (§7 records the last
elision marker doing exactly that).

A speaker deleting their own sentence is not the pipeline changing its mind, so
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
- Every sentence is printed at exit, unconditionally. Losing text because someone
  forgot a magic word would be indefensible — and since that was true from the
  start, no magic word ever gated it, which is the argument below.
- The document is also on disk throughout, which is what makes the destructive
  commands safe — see *The session file* below.

### The approval flag, and why it was decay rather than design

`Sentence.committed` is **removed**, along with `Luna, commit`, `commit_all`,
the third rung of the styling ladder and the "N kept, M pending" status line.
This is not a reversal of a measurement; it is the removal of something whose
last load-bearing role had already been taken away and whose corpse nobody
noticed.

The tell was that by the end it had **four consumers and all four were
display**: the gutter character, the status counter, an exit note on stderr, and
the setter. Nothing branched on it. `copy` took the whole document, `finish()`
printed the whole document, `delete` and `clear` ignored it.

Four things make the reading "leftover" rather than "cosmetic but intentional":

- **Its one real job was deleted on measured use.** `copy` used to take kept
  sentences only. That was reversed above, on a live session reporting *0 kept,
  4 pending*. Nothing gated on the flag after that.
- **The session file cannot represent it.** `store::load` returned everything
  `committed: true`, so `--resume` silently approved a transcript the speaker
  had not approved. A state the product's own persistence format cannot store,
  whose loss was never filed as a bug, is not load-bearing. The round trip is
  now lossless by construction.
- **Every way to give it teeth is already rejected here, on principle.** Gating
  output — "losing text because someone forgot the magic words would be
  indefensible". Auto-commit on the settle timer — rejected below. Protecting a
  committed sentence from `delete` — the *inverse* of the requirement that
  forced the document model into existence.
- **The ladder argument does not survive re-reading.** The styling contract is
  about **mutability**: dim = the pipeline may rewrite this, plain = it will
  not. Commit changed no mutability whatsoever — a committed sentence was
  exactly as deletable as a pending one. So the third rung was not extending
  that contract, it was orthogonal decoration sitting on it. The `│` gutter now
  carries a question that *is* answerable — in flight, or filed — which is what
  it was always actually distinguishing.

What it cost, meanwhile, was not nothing: a fifth verb in the model's vocabulary
hint (§7), and `commit` was the most ordinary English word of the set — "I will
commit the change tomorrow" — plus its `-s` surface, `Luna commits`. **A verb
that moves no text is pure false-command risk**, since a false match can only
ever destroy. That is the entry requirement now: every remaining verb changes
the transcript.

The one honest argument the other way is that "kept" marked how far the speaker
had reviewed. There is nowhere to review *to*: the live view has no scrollback
(§9), so an older sentence cannot be looked at, only deleted blind.

### Spoken commands

`command.rs`. `Luna, delete` drops the newest sentence; `Luna, undo` puts it
back. The name is `--assistant`, default `Luna`.

#### Two scopes, `delete` and `discard`

They are not synonyms, and naming them as though they were — the verb was
`reject` for a while, against `discard` — hid the one distinction they exist to
draw. An utterance becomes **one document entry per sentence** via
`split_sentences`, so "drop the last sentence" and "forget what I just said" are
the same instruction only when the model heard one sentence in it.

- `delete` is **document-scoped**: the newest entry, whatever it is and however
  long ago it was filed.
- `discard` is **utterance-scoped**: everything the current utterance filed.

Measured end to end, same audio, verb the only difference — `"The deployment
finished at noon."` / 3s / `"My cat is asleep on the keyboard."` / 1.2s / verb:

| verb | result |
|---|---|
| `keep` | `keep — filed 2 sentence(s)`, both printed |
| `discard` | `discard — dropped 2 sentence(s)`, document empty |
| `delete` | would leave the deployment line standing |

**`discard` reaches further than it used to, because an utterance is longer than
it used to be.** The §5 settle now holds across a thinking pause, so two
sentences separated by an 8s pause are *one* utterance where they were two.
Measured, same audio, settle the only variable:

| settle | result of `Luna, discard` |
|---|---|
| 6s (old) | `dropped 1 sentence(s)` — the earlier sentence survives |
| 15s (current) | `dropped 2 sentence(s)` — both go |

This is left as it is, and the reasoning is worth recording because §6 otherwise
forbids exactly this. It is not the scope *widening to whatever is nearest* —
which is what that rule bans, and what `delete` is for. It is the scope tracking
its own definition: `discard` has always meant "everything this utterance
filed", and the utterance really is bigger now. It is also announced — every
branch reports its count through `notice` — and `undo` restores the group as one
entry.

What bounds it in practice is the trim: each `apply` counts only what *that*
finalized utterance filed, so a discard cannot reach past the last trim.

**The scope is supplied by the caller, and that is the safety argument.**
`Transcript::discard_last` takes a count; `main.rs::apply` maintains it as the
number of sentences *this* utterance has filed and not yet closed off. So a
discard cannot walk backwards into the transcript however many times it is said
— verified, three in a row leave a settled sentence untouched. Saying it with
nothing in flight removes **nothing** rather than falling back on the newest
sentence, because silently widening the scope of a destructive command to
whatever happens to be nearest is precisely what this module refuses to do.

Every branch of `apply` that moves text has to keep that count honest, and the
failure otherwise is specific: without `delete` decrementing it,
`"Sentence. Luna, delete. Luna, discard."` deletes the sentence and then
discards a *second* one this utterance never filed. `clear` and `rollback` zero
it; `copy` leaves it alone, because copying moves no text.

#### `keep`, and the honest case against it

`keep` files what the utterance said and ends the settle. **It moves no text of
its own**, and that has to be stated plainly because it is the property that
killed `commit`.

The reason is the §5 merge. Speaking inside the grace window merges the pending
audio into the command's utterance, and the command branch in `main.rs` files
the text and drops the audio *without* creating a new `Pending`. So **every**
command already ends the settle early — `copy`, `delete` and `clear` all do it.
By the time `Keep` runs, the Text segment ahead of it has already filed the
words.

What is left for it is thin but real, and it is three things `commit` never had:

- it is the only way to end the settle **without a side effect**;
- it marks the thought finished, so the next utterance starts from a fresh short
  buffer instead of merging into a growing one — a §3 latency win the speaker
  can ask for, since no duration threshold can tell terminal from hesitation
  silence and they are the only one who knows;
- it closes the utterance off from a following `discard`.

That is weaker than the other five verbs and should be re-read as a candidate
for removal if it goes unused. The entry requirement was **every verb must move
text**; this one moves text *in time* rather than in content, which is a
weakening of the rule, admitted here rather than argued away.

**This reverses the removal recorded below, and the two are not in conflict.**
Self-repair replaced the wake word for *fixing a word*, and that was right —
nobody says "admin" before stuttering, they just say the word again, and there
is an acoustic signal to key off. But "throw that away" and "put it back" have
no acoustic signal whatsoever. They are document operations, not speech events.
Nothing short of a wake word can express them, which is exactly why one earns
its place here and did not earn it there.

**Every verb must move text**, which is the entry requirement `commit` failed.
A verb that changes no text cannot help the speaker and can still fire by
accident, so it is pure downside on both axes that matter here: the hint it adds
to the model's context (§7) and the false-command surface below.

**Detection runs on finalized utterances only, and this is a correctness
requirement rather than an optimisation.** `repair()` is safe per-hypothesis
because it is pure. Commands are not: `delete` destroys a sentence, and the
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
`normalize`, which is what makes "Luna, delete" and "Luna delete" the same input
— the comma is punctuation the model chose and the parser never sees it. Nothing
may come between them. The asymmetry justifies the strictness: a *missed*
command costs one repetition, a *false* one deletes text the speaker meant to
keep. Measured controls, all left as dictation: `Luna went to the store.`,
`I will copy the file tomorrow.`, `Copy Luna`, `Luna please copy`,
`Luna deleted`.

**A third-person `-s` on the verb is the same command**, and this is the one
place the strictness is relaxed. Reported live: a spoken "Luna, delete" comes
back from the model as `Luna deletes.` The speaker said the verb — the model
conjugated it — so reading that as dictation punishes them for a transcription
they can neither see nor influence, and it is the one failure the wake word
cannot rescue, because they *did* say the wake word. It applies to every verb,
not just `delete`: `discards`, `keeps`, `clears`, `copies`, `rollbacks`, `undoes`.

The cost is real and worth stating: `Luna deletes the offer.` is now a delete.
It is bounded rather than open-ended — `--assistant` retargets the name,
`rollback` takes it back — and the allowance stops at `-s`. `Luna deleted` and
`Luna deleting` stay dictation, because a past or progressive form is a
plausible thing to say *about* a person called Luna in a way the bare present
tense following the name is not.

Segments are applied **in spoken order**, not with commands hoisted. "That's the
wrong sentence. Luna, delete." must file the text before the delete reaches it,
or the delete removes the sentence before the one the speaker just finished.

**The merge creates a seam comma that has to be closed.** Because a command
usually follows a pause, §5 merges both halves into one buffer, and the model
then punctuates them as running on. Measured end to end:

| spoken | model returns | after stripping the command |
|---|---|---|
| "This is my first sentence." + "Luna, commit" | `This is my first sentence, Luna, commit.` | `This is my first sentence,` ← dangling |

(Measured on the `commit` verb, which no longer exists. The seam is a property
of the merge, not of which word followed the wake word, so it transfers.)

`text::close_sentence` fixes the last word of a Text segment that is *terminated
by a command*. This is not a guess about what the speaker meant: the words that
followed were provably not part of the sentence, so the sentence ended there.
Text appearing *after* a command is left alone — nothing establishes that it
ended.

Six verbs: `delete`, `discard`, `keep`, `clear`, `copy`, and `rollback` — the
last of which also answers to `undo`. Two words for one command is otherwise not
something this module does; `rollback` is the operation's name everywhere else
here, `undo` is what people actually say, and both still require the wake word
immediately before them so the extra surface costs nothing. Worth knowing that
the model is not equally sure of them: measured, `Luna, undo` came back as
`Luna, Andú.` inside a longer utterance while firing correctly in isolation.
The failure is benign — an unrecognised command is filed as dictation — and
having the second verb is the recourse.

`copy` takes the **whole document**, with no approval filter in front of it.

This reverses an earlier "committed sentences only" reading, on use. That
reading was defensible in the abstract and useless in practice: reported from a
live session as `copy` reporting *0 kept, 4 pending* after the speaker had
waited out the settle window and reasonably assumed settled text was theirs.
Making them say two commands to express one intention was ceremony, not safety.
The flag it filtered on has since gone entirely (§6), which retires the question.

The boundary that **is** load-bearing survives untouched, and it is a different
one: `copy` stops at the *document*. An utterance still inside its grace window
is not in there yet, so text the **pipeline** may still rewrite is never copied.
The rejected reading guarded against the speaker not having vouched for text;
this one guards against the text not being final, which is the guarantee only
this program can make.

It shells out to `pbcopy` with both output streams silenced — a child writing to
the terminal would land in the middle of a frame, the same invariant that made
the sidecar's stderr a pipe rather than inherited.

**Auto-committing on the settle timer was considered and rejected**, and it is
worth recording that the *reason* given at the time was wrong in a way that
pointed at the real answer. The argument was that a timer doing the vouching
would collapse the three-level ladder and make `Luna, commit` a no-op. Both were
true. What went unnoticed is that `commit` was *already* a no-op in every sense
that reaches text — so the objection was really an argument for deleting the
rung, not for defending it. That is what eventually happened.

#### Repeat suppression — removed, and what it turned on

There used to be a `command::Debounce`: a 3s wall-clock window that swallowed a
second `Reject` or `Clear`. The argument was that the speaker cannot see a
command land until the utterance ends, so a repeat inside roughly one reaction
time is one intention said twice — and taking it literally deletes a second
sentence.

**It is gone, because the vocabulary hint removed the premise.** The guard was
built for a speaker who repeats themselves *because they doubt they were heard*,
and that doubt was well founded when `Luna, rollback` came back as
`Luna, roll back.` and silently did nothing (§7). With the command list as
`system_prompt` the commands are recognised on the first try, so a second
`delete` is far better read as someone who means it twice — and the guard was
costing them exactly that, with no recourse but waiting out a window they cannot
see either.

The asymmetry that justified it has flipped with it. It used to weigh *a false
extra delete* against *one repetition*; it now weighs *a swallowed command*
against *one `undo`*. `rollback` was always the reason a command as blunt as
`clear` is safe to offer, and it was never guarded — "undo, undo, undo" is how a
run of deletes is walked back out. That recourse covers the removal too.

Two findings from it are worth keeping, because they cost measurements:

- **Collapsing adjacent commands inside `split` would not have worked anyway.**
  700ms between two "Luna, delete" is past `--endpoint-ms`, so they arrive as
  *two separate utterances* and `split` never sees them together. Measured, both
  fired and two sentences went — which is the behaviour the code is back to, and
  the reason the guard had to be wall-clock rather than syntactic.
- **Every command still reports itself through `notice`.** Silence is
  indistinguishable from not being heard, which is exactly what provokes a
  repeat. That was true when a suppressed command had to announce itself, and
  the notice is now the whole acknowledgement.

#### Rollback, and why `clear` is safe to have

`clear` throws away the whole document, which is only defensible because
`rollback` puts it back. Undo entries are **deltas, not snapshots**: a delete
remembers one sentence, where snapshotting would cost a copy of an
unbounded transcript per step. Depth 64.

Only text-*removing* commands are recorded, which since the removal of `commit`
is every command except `copy`. Undoing a copy would remove nothing and would
consume the entry that does have text behind it.

Verified end to end on synthesized speech: delete drops a sentence and leaves
the document empty, `clear` removes 2 and `rollback` restores both, `copy` moves
the text onto the real clipboard, `--assistant Jarvis` retargets everything, and
`Luna went to the store.` survives as dictation.

### The session file

`store.rs`. The document is written to `~/.local/state/speech-to-text-cli/` on
every change, and **that is what makes `clear` and `delete` safe to offer**.

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

Mid-passage the **buffer trim was the only thing that filed anything**. §5
merges every ordinary pause, so an utterance never settles, `finalize_window`
files nothing by design, and the grace never closes. Between trims the text
existed only in `Pending` and `Transcript::committed` — process memory, which is
exactly what this module was written to stop relying on. The document, `copy`
and the file all read `script.document()`, so one empty document explained all
three reported symptoms at once.

Logical commitment (§6) has since made the document fill steadily rather than
only at a trim, which shortens the exposure — measured, a `kill -9` at T+25s of
a 37s passage recovered 4 settled sentences where it used to recover 2. **It
does not remove the need for any of what follows.** The last two sentences are
always unfiled by construction, and a speaker who has said only one has nothing
committed at all, so the in-flight text still has to reach the file.

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

**The round trip is lossless, and it took removing a field to make it so.** The
file stores text, and a `Sentence` is now only its words, so what goes in is what
comes back. It used to store text while a `Sentence` also carried an approval
flag, which the format had nowhere to put: `load` therefore returned everything
`committed: true` and a resumed session silently approved itself. That was
rationalised at the time — "choosing to resume a transcript is itself an approval
of it" — and the rationalisation was the tell. A field the persistence format
cannot represent, whose loss has to be argued away rather than fixed, is not
carrying weight. §6 records what came of noticing that.

**A recovered session is continued in place, but only if it is ours.** A file
found by us in our own state directory is adopted — same file, no duplicate, and
`--persist` governs it as it would any other. A path the user *named* is
read-only, and a new session file is created beside it, because adopting it
would mean a clean exit deleting something they pointed us at. Verified: an
explicit `--resume ~/mynotes.txt` left the file byte-identical.

**`restore` is not undoable.** `rollback` takes back what the speaker did during
a session, and this happened before the session began. Letting undo reach it
would mean one "Luna, undo" emptying a transcript they had only just recovered.

**A resumed session is deleted on a clean exit, and that is sharper than it
sounds.** `--persist` governs an adopted file exactly as it governs a fresh one,
so bare `--resume` without `--persist` **removes the file it just recovered**.
The rationale is the same one as for a fresh session — stdout has the transcript
by then — and it is consistent, but the two cases are not equally forgiving: a
file reached by `--resume` exists *because something already went wrong*, and
bare `--resume` picks it by mtime, so it can adopt and then delete a file the
speaker never named. Observed for real during this work, on a file holding a
live session's only copy. Left as designed, because the argument for it holds
and `--persist` is the escape hatch — but if it is ever changed, this is why.

**A failed `--resume` used to leave an empty file behind, and it poisoned the
flag that would have found the real one (fixed).** `Store::new` writes its file
up front, and the resume target was loaded *after* that — so a mistyped path
exited non-zero having created `session-<now>.txt`, empty. Since
`newest_session()` goes by mtime, that empty file was then the newest, so the
next bare `--resume` adopted it, recovered nothing, and deleted it on the way
out. One typo could put the real transcript permanently out of reach of the
recovery flag. The load now happens **before** anything creates a session file.

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
Wednesday" is still not expressible. `Luna, delete` does not recover it either
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

- **The command hint does bias, just not the way the transcript did — so it is
  off on the windows the speaker dictates into** *(the gating rule below is
  superseded by the entry after it; the measurements are unchanged and are why
  the hint is still absent from every live window)* — raised as "the model must
  never hallucinate a command on keystrokes or background noise; it is biased
  toward them because the system prompt names them." Half right, and the half
  that is right was not visible in the table above, because that table only
  asked whether the hint *echoes*.

  It does not echo. What it does is **pull ambiguous audio toward the wake
  word**. Measured on `Qwen3-ASR-1.7B-8bit`, same audio, prompt the only
  difference:

  | spoken | no prompt | with the hint |
  |---|---|---|
  | "Moon a" | `Muna.` | **`Luna.`** |
  | "Luna respect" | `Lunar respect.` | **`Luna respect.`** |
  | "Luna coffee" | `Luna Coffee.` | `Luna coffee.` |

  Three cases, all in the same direction. Nothing here is a false *command* —
  across eight deliberate near-misses (`Luna respect / here / coffee / project /
  clearly / undertow`, "Moon a wreck") **the hint never invented a verb**. But it
  makes the wake-word half of a trigger reachable from audio that contains no
  wake word, on exactly the windows where nobody addressed the assistant.

  Everything else came back empty with *and* without it: digital silence, white
  noise at −45 dBFS, room tone at −38, 60 Hz hum, isolated keystrokes and
  sustained typing. **Noise was never the problem.** Near-miss speech is.

  **The fix is to stop paying for the hint on windows that cannot benefit from
  it.** Measured across every verb and two voices, the unprompted decode already
  handles four of the five:

  | spoken | no prompt | with the hint |
  |---|---|---|
  | `Luna, delete` / `clear` / `copy` / `undo` | parses | parses |
  | `Luna, rollback` | `Luna, roll back.` — **missed** | `Luna rollback.` |

  One row disagrees, and in it the wake word is already recognised. So:
  **decode unprompted; if the result names the assistant but carries no verb,
  re-decode that same buffer with the hint and take commands from it.**
  `main.rs::resolve`, gated on `command::mentions_wake`.

  - Ordinary dictation, noise, keystrokes and near-misses never see the hint,
    so its one measured bias has nothing to act on.
  - `rollback` still works, which is the only thing the hint ever bought.
  - The hinted reading replaces the unprompted one **only if it found a
    command**, so text the speaker keeps is never prompt-influenced.

  Verified end to end. A merged `That was wrong. Luna, rollback.` — which comes
  back unprompted as `That was wrong, Luna. Roll back.` — triggered **exactly one
  hinted pass out of 16** and the undo landed. A 30-pass run over dictation, the
  `Moon a` near-miss, sustained typing, room tone and `The reviewers rejected
  it.` triggered **zero**, and the near-miss stayed `Muna` instead of being
  pulled to `Luna`.

- **The gate above is removed: the hint now runs on every finalized utterance
  that did not already parse as a command.** The design it replaces is the one
  immediately above, and the reason is not new evidence — it is that the gate
  assumed away the failure it existed to fix.

  `mentions_wake` asked, of the **unprompted** transcript, "was the assistant
  named?" That is a question about whether the model wrote the wake word
  correctly. The two rows the entry above is built on are the model writing it
  *incorrectly*: `Moon a` -> `Muna.`, `Luna respect` -> `Lunar respect.` So a
  speaker who says "Luna, delete" and is transcribed `Lunar delete.` fails the
  gate, never gets the hinted pass, and has their instruction filed as dictation
  with nothing on screen to say so. The wake word cannot rescue them, because
  they *did* say it — which §6 already identifies as the one failure mode with
  no recourse, and the gate guaranteed it.

  A fuzzier gate (edit distance, prefix containment, joined token pairs) only
  moves the boundary and adds a second thing to be wrong about. So there is no
  gate.

  **Why that is safe is a different question from why it is affordable, and the
  safety half is structural.** The acceptance test is unchanged: the hinted
  reading replaces the unprompted one *only if it parses as a command*. `Muna.`
  pulled to `Luna.` is discarded, because a bare wake word with no verb is not a
  command. Every word the speaker keeps still comes from the unprompted pass.
  Loosening the gate changed only the decision to **spend a forward pass**, and
  that decision cannot damage text — which is the point the original entry
  missed by holding a cheap decision to the bar of a destructive one.

  Verified end to end with the hint now reaching every one of these:

  | spoken | result |
  |---|---|
  | `The lunar module landed safely.` | unchanged — **not** pulled to `Luna` |
  | `I will copy the file tomorrow.` | dictation |
  | `We should discard the first draft.` | dictation |
  | `Please keep the receipt for me.` | dictation |
  | `Luna went to the store.` | dictation |
  | `…will be dropped. Luna, delete. Luna, rollback.` | both fire, sentence restored |

  The `lunar` row is the one to keep: it is the near-miss the old gate rejected,
  it now gets the hint every time, and it still comes back `lunar`.

  **Cost, stated rather than buried: one extra forward pass per endpoint**, paid
  on the utterances that carry no command, which is most of them. It lands where
  the loop is already off-tick, so `CAPTURE_BACKLOG` absorbs it and §3's warning
  about back-to-back off-tick passes applies directly — this makes that worse,
  not better. If it proves too expensive the fix is to hint a **bounded tail** of
  the buffer rather than to put the gate back; a command is a few words at the
  end of an utterance, so a 3–4s tail is ~150ms against ~1s for a 30s buffer.
  Not implemented, because it needs a seam between the unprompted head and the
  hinted tail and the cost has not yet been shown to matter.

  **The structural invariant is strengthened, not weakened.** The per-request
  field is a **boolean**, not a string: the host chooses whether the fixed prompt
  applies, never what it says. There is still no text field, so the transcript
  still cannot reach the model by any path — which is the property the original
  ban was protecting, and the reason "pass it once at spawn" was only ever a
  means to it.

  Cost: one extra forward pass on an utterance that names the assistant without
  commanding it, and a second warmup at startup so the rare hinted pass does not
  pay graph compilation while the speaker waits. Startup is ~4.4s against ~3s.

  **What this does not fix**, and is a different mechanism: `Luna copies the
  file.` is still a `copy`, prompt or no prompt. That is the `-s` allowance in
  §6, whose cost is stated there, and it is unaffected either way.

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
  all* until the session ends, because `Luna, delete` means the speaker can
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

- **A whole utterance dropped when a pause was shorter than a forward pass
  (fixed)** — the worst failure this program has had: **84 words, 30.5s of
  speech, gone silently.** Found by instrumenting a 109s session rather than
  reported, which is the only reason it was found at all — nothing on screen and
  nothing in the session file said anything had been lost.

  It is a direct consequence of the entry below. The loop drains the whole
  capture backlog per iteration, so **one iteration covers as much audio as the
  last forward pass took** — ~870ms at a 30s buffer, ~2s at 60s. That is longer
  than `--endpoint-ms`, so a single batch of VAD frames can contain an utterance
  ending *and* the next one beginning. Traced:

  ```text
  30.973s ONSET     {"pending": false, "win": 30.48}
  30.973s ENDPOINT  {"buf": 30.48}
  34.952s OVERWRITE {"lost_words": 84, "lost_audio": 30.48, "age_ms": 2226}
  ```

  Three independent things then had to go wrong, and they went wrong on their
  own:

  1. **The onset was observed before the `Pending` existed.** `main.rs` checks
     the continuation merge *above* the endpoint handler that creates it, so the
     onset merged nothing and was discarded.
  2. **The VAD was left active.** That same onset re-opened it after the
     endpoint closed it, so it emitted only `Speaking` afterwards and **no
     second `Onset` ever came**. The merge is edge-triggered, so the held
     utterance was not merely late — it was unreachable.
  3. **`pending = Some(…)` was a bare assignment.** The next endpoint wrote over
     the held utterance and its text went with it, 2.2s into its own 6s grace.

  The fix is at the source: `scan()` **stops at an endpoint** and leaves the
  remaining frames in the buffer, so the two events land in separate iterations
  with the caller's `Pending` built in between. The window is split at the
  endpoint rather than taken whole, so the next utterance keeps its own opening
  audio instead of having it spliced onto the end of the last one. And the
  assignment files any held utterance first — unreachable now, but this is the
  one place where being wrong costs a whole utterance, so it does not rely on
  the argument.

  **It scaled with the buffer, which is what made it dangerous rather than
  merely rare.** The longer the buffer, the wider the window in which a
  resumption can be swallowed, *and* the more speech is riding on the `Pending`
  that gets overwritten. The §3 trim work independently shortened the buffer,
  which shrinks the exposure — but the ordering is the fix and the buffer length
  was only ever the amplifier.

  Verified end to end: the same session went from 12 transcript entries to 18,
  with the passage's first eight sentences restored, and zero overwrite events.

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
  transcript falls". It was also, at the time, the *only* thing that got text
  out of process memory during a long passage, so starving it silently disabled
  the crash safety net. Anything that changes trim policy has to be read as
  changing persistence policy too.

  **Superseded on the specifics, and the coupling is the reason to say so.** The
  trim files nothing on the ordinary path any more (§3): logical commitment and
  the in-flight line carry the passage instead. So the *lesson* holds exactly —
  changing trim policy changed persistence again, and had to be re-verified —
  while the sentence it was written about is no longer true. Re-verified after
  the change with `kill -9` at T+20s of a continuously merging passage: the
  session file held everything up to ~2s before the kill.

  Diagnosis note, because it cost two wrong answers: the first reading was
  "renderer starves the document" (`build_rows` draws the whole live tail before
  any document rows, which is real but was not this), and the second asserted
  the text was safe on disk. Both were reasoning without a repro. The repro is
  cheap — `say` a passage with `[[slnc 1000]]` between sentences so every gap
  merges, run it under `--simulate --persist`, and poll the session file. Do
  that first.

- **One spoken sentence filed as two entries, because the trim decoded severed
  audio (fixed)** — reported live at the shipped defaults, from a deliberate
  mid-sentence pause:

  ```text
  It tries to solve the problem.
  Of coordination between participants, a single node can communicate with the other nodes.
  ```

  Not a model failure. Handed the merged buffer the model returns the sentence
  whole; the pipeline asked it that question, then filed the answer to a
  different one — the head decoded alone, without the rest of the sentence in
  front of it. §3 has the three-way decode and the fix. Two things are worth
  repeating here because they generalise:

  **A decode of severed audio is not evidence about a sentence boundary.** The
  model punctuates whatever it is handed (§5), so a head that ends mid-sentence
  comes back with a full stop on it and a tail that starts mid-sentence comes
  back capitalised. Both are confident, both are wrong, and neither is
  detectable from the text alone — which is why the test is "is this text
  already in the document?" rather than anything about how the text looks.

  **The seam-length reading was the obvious explanation and it was wrong.** The
  merge splices up to 1.2s of silence and the total seam reaches ~2.1s, which
  looks exactly like a boundary cue. Measured across five audio conditions and
  seams from 0 to 4000ms, the boundary never moved (§5). Checking the mechanism
  before building on it is what kept a flag out of the CLI that would have done
  nothing.

- **Cutting audio early damages the transcript; filing *text* early does not.
  The two are not the same operation (measured — do not re-litigate)** — this
  distinction cost a wrong conclusion before it was drawn, so it is worth
  stating before either measurement.

  **Cutting audio** removes context from every *future* decode. **Filing text**
  removes nothing: the audio stays in the buffer and the model keeps re-decoding
  all of it. Only the first is destructive, and conflating them made the second
  look impossible.

  Cutting early, on a 47s passage of 7s thinking pauses, audio held together so
  the trim threshold is the only variable:

  | `--trim-after-s` | trims | transcript |
  |---|---|---|
  | 30 | 1 | two sentences, verbatim ground truth |
  | 20 | 2 | `…which I rather feel` / `Is a little bit off the rails.` — **stranded, no terminal punctuation** |
  | 12 | 5 | six fragments — *identical to the broken baseline* |
  | 8 | 5 | six fragments |

  At `--trim-after-s 20` this is §9's "the trim seam is still a seam" caught in
  the act. **The distinction that matters is *where* the cut lands, not how
  early it is**: the `30` row also filed early — it cut at 11.18s, long before
  the passage ended — and was perfect, because that cut fell on a true sentence
  boundary.

  Filing text early is a different question, and the answer is the opposite.
  Logging every hypothesis the pipeline actually saw across three passages and
  every merge in them, then asking whether a sentence ever changed once another
  sentence had appeared behind it:

  | | |
  |---|---|
  | non-final sentences re-checked | **60** |
  | revisions | **4** |
  | of those, `3` ↔ `three` | **4** |

  No boundary moved, no word changed, no capitalisation changed. That is what
  `Transcript::settled_prefix` rests on, and it is why commitment is a count of
  sentences rather than a duration.

- **Logical commitment is delicate in three specific ways, and all three were
  found the hard way** — `settled_prefix` files a sentence while its audio is
  still in the buffer, so the same words come back on every later decode. Each
  of these produced visible transcript damage before it was fixed.

  1. **The filed-text memory cannot be a word count.** A count indexes into one
     decode, and the two that matter are of different audio — the trim decodes
     `window[..cut]`, every tick decodes the whole buffer. `Transcript::filed`
     matches on *content* instead.

  2. **The overlap test has to tolerate a substitution.** Exact matching
     collapses to zero on the one instability re-decoding is measured to have.
     Observed:

     ```text
     13.01  logical  There was a spike around 3 in the afternoon.
     13.55  trim     I think the latency numbers … around three in the afternoon.
     ```

     One word came back re-tokenised, no suffix matched, and two whole sentences
     were filed a second time. The budget is proportional and floors at zero, so
     short runs must still match exactly — this is *confirming a match known to
     be there*, not searching for a coincidental one.

  3. **Everything filed must be remembered, not just what commitment filed.**
     The trim files through `push_sentence`, and its head decode can transcribe
     a whole sentence from the part of it that was in the head — leaving the
     rest of that sentence in the retained tail to be decoded again. Recording
     only logical commits left the trim's own output unprotected, which filed
     `The database migration is scheduled for Saturday morning.` twice.

     Still true, and now load-bearing in the other direction: the same
     filed-text memory is what the trim *asks* before cutting (§3), so the
     substitution budget in point 2 is now deciding whether audio gets
     discarded, not only whether text gets duplicated. A budget too tight
     refuses good cuts and grows the buffer; too loose discards audio holding
     text nobody has filed.

  And one that is not about the overlap at all: **filing mid-window has to clear
  the agreement deque.** The retained hypotheses were stripped against a shorter
  `filed` than the next one will be, so their words no longer start at the same
  place and `common_prefix_len` compares misaligned sequences. It produced
  spliced nonsense — `We will migration is scheduled for Saturday morning.` —
  and the invariant to hold onto is that **every hypothesis in the deque was
  stripped against the same `filed`.** The cost is `agreement_n` ticks before
  the next word is agreed, paid only when a sentence files.

  All four only appear once commitment is eager enough to fire mid-passage,
  which is the entire point of the mechanism. They are hazards of the design,
  not of a setting: they were surfaced by lowering `KEEP_SENTENCES` to 1 and
  they are latent at any value.

---

## 8. Layout

```
src/main.rs        orchestration: window scheduling, tick loop, trim, commands
                   `Settle` — how long an utterance is held (§5), adaptive
                   the trim's endorsement test and `Dip::refused` (§3)
src/audio.rs       cpal capture, downmix, resample to 16kHz; --simulate path
src/vad.rs         earshot VAD + RMS floor, open/close hysteresis, dip runs
src/transcript.rs  LocalAgreement-n window + the sentence document
                   logical commitment (§6) + the filed-text overlap test
src/repair.rs      self-repair detection ("regularing… regular")
src/command.rs     wake-word command parsing
src/store.rs       session autosave + clipboard
src/text.rs        shared word/sentence helpers + the implausible-opener veto
src/trace.rs       filing-path trace, off unless `STT_TRACE` names a file
src/render.rs      alternate-screen live document, wrap-safe row layout
src/sidecar.rs     subprocess handle + NDJSON protocol
sidecar/asr_sidecar.py   MLX inference; stdout is protocol, stderr is logs
                         prompt is off unless the request asks for it (§7)
                         — asked for once per finalized utterance, never on a
                         window the speaker is dictating into
```

Reproducing a filing-path question — which of the six paths in §6 put this
line in the document? — starts with `STT_TRACE`:

```sh
STT_TRACE=/tmp/s.trace ./target/release/speech-to-text-cli --simulate f.wav
```

Never add an `eprintln!` for this instead. §7 records the redraw bug that comes
straight back if anything writes to the terminal behind the renderer.

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
say -v Samantha -o t.aiff "A sentence. [[slnc 1500]] Luna, delete."
afconvert -f WAVE -d LEI16@16000 -c 1 t.aiff t.wav
```

`[[slnc <ms>]]` inserts a real pause, which is what exercises the endpoint, the
merge and the settle boundary. Note that a 1.4s gap is **not** enough to make
the model file two sentences from one merged utterance — measured, it commas the
seam instead. Two topically unrelated sentences and a 3s gap does it, which is
what the `delete`/`discard` scope difference has to be verified against. Note that `say` will not reliably produce a
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
end to end on synthesized speech at the current defaults (147 unit tests green).

**Read the audio column with the caveat attached.** Everything below is
synthesized speech, which has clean prosody and no genuine hesitation — so it is
evidence about *mechanism* (does this code path fire, does this cut land, does
this string split) and not about the thing the §3 failure is made of. The
mid-sentence-pause rows reproduce the reported bug mechanically and show it gone;
they do not establish that a human hesitating mid-thought now comes out whole.
That needs real dictation, and so does sizing the backstop below.

| case | result |
|---|---|
| `"First sentence. Luna, copy."` | filed; seam comma closed to a full stop |
| `"…should disappear. Luna, delete."` | dropped; document empty |
| `"First sentence. Luna, copy. Luna, delete."` | **a sentence already copied is dropped** — the requirement that forced the document model |
| `"Luna, clear."` then `"Luna, rollback."` | 2 removed, 2 restored |
| `"Luna, copy."` | the whole document lands on the real system clipboard |
| `"Luna, deletes."` | fires — and `--assistant Jarvis` on the same audio shows the model really did write the conjugated form |
| `"Luna, delete."` twice, 1.7s apart | both fire; two sentences dropped, each announced — the debounce that used to collapse them is gone (§6) |
| 20s of synthesized typing over room tone | nothing transcribed, at `--open-ms` 48 and 150 alike |
| `--assistant Jarvis` | retargets every command and the banner |
| `"Luna went to the store."` | left as dictation |
| 4s gap | merged, one sentence, comma across the seam |
| 8s / 12s gap | merged at the 15s floor, and the model **still returns two sentences** — over-merging is arbitrated, not damaging |
| **47s passage, 7s thinking pauses** | **6 fragments → 2 sentences, verbatim ground truth** — the reported bug; `And politics` / `Is a little bit` / `To be doing very well` were fragments decoded alone |
| same passage, normal-paced control | byte-identical at a 6s and a 600s settle: same buffer max, same commit latency, same 33 passes |
| 66s passage, 20s pauses | fixed 6s and fixed 15s both give 4 fragments; adaptive gives **2** — the first pause is still lost, every later one is held |
| 33s passage, 14s pauses | 2 sentences (was 3 fragments, one with a spurious `?`) |
| same 47s passage, `--trim-after-s` 30 / 20 / 12 / 8 | 2 sentences / a stranded `Is a little bit off the rails.` / 6 fragments / 6 fragments — filing early damages text whatever triggers it |
| `Luna, delete` after an 8s pause | fires — the command merges with the held utterance and still parses |
| **37s passage, 10 sentences, logical commitment** | ground truth, and the document grows **one sentence at a time from 14.9s** where filing by trim alone grew two at a time from 17.6s |
| same, `kill -9` at T+25s | **4 settled sentences recovered**, against 2 before |
| same, `KEEP_SENTENCES` 1 vs 2 | both ground truth *after* the three overlap bugs in §7; 1 was what exposed all of them |
| `Luna, delete` after 4 committed sentences | fires, drops the newest — and the wake word is never filed as dictation by a hypothesis |
| live view through a pty, 37s passage | **175 in-flight rows all dim, 229 document rows all plain**, widest 79 at width 80 |
| `Luna, discard` across an 8s pause | `dropped 2 sentence(s)` where the 6s settle dropped 1 — the scope tracks the longer utterance, announced and undoable (§6) |
| `kill -9` at T+30s of the slow passage | the **fused** first sentence recovered from the session file, plus the partial second |
| 42s unbroken passage | trimmed at pauses, 7 clean sentences, no word cut, 72% duty |
| `kill -9` mid-session, no `--persist` | settled text recovered from the session file |
| `kill -9` at T+20s of a continuously merging passage, before any trim | 4 sentences recovered, split one per line, with stdout empty as designed |
| 41s passage, 1s pauses throughout (every gap merges) | file tracks the passage from T+6s; before the fix it held nothing until the first trim at ~T+33s |
| 80s of continuous speech, `--trim-after-s 10` | buffer peaked at 13.3s against the 20s bound; transcript clean |
| 63.6s buffer, token budget vs the 8192 default | byte-identical text, 1.82s vs 1.81s — the cap never binds on real speech |
| `"…will be dropped. Luna, delete. Luna, rollback."` | both fire; the sentence is dropped and restored — `rollback` was unreachable before the vocabulary hint |
| command list as `system_prompt`, 20s noise / 41s dictation / controls | no echo, byte-identical, no false command |
| hint always on vs off: `"Moon a"`, `"Luna respect"` | `Muna.` / `Lunar respect.` unprompted — the hint pulled both toward the wake word, which is why no live window ever sees it (§7) |
| merged `"That was wrong. Luna, rollback."` | 1 hinted pass of 16; undo landed — the retry path doing its one job |
| dictation + near-miss + typing + room tone + `"The reviewers rejected it."` | **0 hinted passes of 30**, under the gate that has since been removed; near-miss stayed `Muna` |
| **`delete` vs `discard`**: 2 sentences in one breath, verb the only difference | `keep` filed 2 and printed both; `discard` dropped 2 and left the document empty — the scope difference `delete` alone cannot express |
| `"Luna, discard."` x3 with one settled sentence | removes nothing; a discard cannot reach past the current utterance |
| `"…thought. Luna, discard."` then `"Luna, undo."` | dropped and restored as **one** undo entry |
| hint on **every** finalized utterance: `"The lunar module landed safely."` | unchanged — the near-miss the old gate rejected now gets the hint every time and still comes back `lunar`, not `Luna` |
| same, `"I will copy the file tomorrow."` / `"We should discard the first draft."` / `"Please keep the receipt for me."` | all dictation; no false command from a hinted pass |
| next launch after that kill | announces the orphaned session, naming it and its size |
| `--resume` | recovers it, continues dictating, **adopts the file in place** — still one file |
| `--resume ~/mynotes.txt` | loads it, leaves it byte-identical, opens a new session file |
| `--resume` with nothing to resume | fails in 0.09s, exit 1, before the model loads |
| `--resume <bad path>` | fails, exit 1, and now leaves **no** empty session file behind to poison the next bare `--resume` |
| plain run / `--trim-after-s` / `--persist`, with another session's file present | that file is untouched — only bare `--resume` adopts, and therefore only it can delete one |
| 109s dictation, endpoint and onset in one VAD batch | **84 words / 30.5s recovered** — 12 transcript entries became 18; the overwrite that dropped them cannot happen now |
| same session, eager trim band | buffer max 30.0s → **15.0s**, commit latency max 3.03s → **1.94s**, duty 81% → **49%**, transcript byte-identical |
| same session, five gap-selection rules | only latest-wins-at-480ms damaged a seam; every rule bounded the buffer identically, so longest-wins was kept |
| live view through a pty, 40s clip | **211 in-flight rows all dim, 386 document rows all plain** — nothing renders plain while the pipeline can still rewrite it |
| widest emitted row at width 80 | 79 columns — the strict-width invariant holds with the new styling |
| live view through a pty, 47s slow passage | **258 in-flight rows all dim, 88 document rows all plain**, widest row 79 at width 80 — the settle change does not touch the styling contract |
| **mid-sentence pause, 1.6s, 17s into a passage** | before: `…it tries to solve the problem.` / `Of coordination between participants, …` — the reported bug. After: **one sentence, whole** |
| same audio, seam length swept 0 → 4000 ms (5 conditions, 9 lengths) | **the boundary never moved once** — the seam-length explanation is measured dead (§5) |
| head / tail / whole-buffer decode of that passage | `It tries to solve the problem.` / `Of coordination between participants, …` / the sentence whole — the failure, isolated to the severed decode |
| 37s passage, 10 sentences, endorsement test on vs off | same transcript bar one numeral (`three` vs `3`); 3 trims → 2 taken and 9 refused; commit latency past 2s once, at 2.0s |
| the mid-pause repro, endorsement test on vs off | buffer 17s → 27s, commit latency 2.1s → 2.9s — the latency this trade costs, stated |
| `--trim-after-s 8` on the same passage (ceiling 16s) | `So we should review the metrics before.` / `The end of the week, because I think the.` — the backstop reproducing the old damage on demand |
| `kill -9` at T+20s, trim no longer filing | the passage still recovered from the session file, to within ~2s of the kill |
| `split_sentences` opener veto | the reported fragment stays joined; `And` / `But` / `To be fair` / `For now` / `By Friday` / `In fact` / `Of course` / `Which is why` all still split |

Not yet done:
- **The audio is still cut by position — but a cut that lands mid-sentence no
  longer files anything.** The trim asks the document before cutting (§3), so
  the fragment it used to leave is gone from the ordinary path. The cut itself
  is still chosen by pause length and still lands wherever it lands; what
  changed is that a bad position now costs a refusal and some buffer instead of
  a sentence. Locating the boundary *in the audio* remains impossible here:
  punctuation cannot do it (the model punctuates fragments), timestamps need a
  second model, and — measured, §5 — pause length carries no boundary signal at
  all.
- **The forced backstop can still file a fragment, and nobody knows how often.**
  Past `2 × --trim-after-s` the trim files the severed head, because an
  unbounded buffer is not an outcome (§3). Reproduced on demand at
  `--trim-after-s 8`. At the default it needs 60s of speech that logical
  commitment has not filed — which needs three sentences, so a speaker of very
  long sentences reaches it far sooner than one of short ones. **This is the
  residual risk of the whole change and it needs real dictation to size.** The
  candidate fix is to file from the *whole-buffer* decode there rather than the
  head, keeping only the incomplete tail in the window; it was not built because
  a provisional word whose audio sits before the cut could be lost, and that
  cannot be measured on synthesized speech either.
- **Refusing a trim costs a forward pass, and a passage of long sentences pays
  it repeatedly.** Measured at 9 refusals over a 37s passage. The memo on
  `Dip::refused` bounds the repeats to one per candidate per document change,
  and the refusals are cheaper than the passes they replace on the accepted
  path (`resolve` no longer runs there) — but a speaker who never lets the
  document move pays a pass per candidate per tick-batch, on a growing buffer.
- **The `3` ↔ `three` instability now decides whether audio is discarded.** The
  trim's endorsement test runs through the same overlap budget that dedupes
  filed text (§7), so a decode that re-tokenises a word past the budget refuses
  a cut it should have taken. It fails safe — the buffer grows rather than text
  being lost — but the budget is now doing two jobs and was sized for one.
- **The first pause past the settle floor is always lost.** A pause can only be
  measured once it has ended, so a speaker whose habit exceeds `--continue-ms`
  pays one fragment before `Settle` adapts. Nothing short of predicting the
  pause fixes it; raising the floor only moves which speaker pays.
- **`Settle` learns from silence, not from meaning.** A speaker who stops for a
  minute mid-topic and one who stops for a minute because they have finished are
  indistinguishable to it, so a long interruption stretches the hold for the
  next few utterances. Bounded by `--continue-max-ms` and forgotten after five
  pauses, but it is a real source of "why is my text still dim".
- **Arbitrary substitutions.** "Tuesday… Wednesday" cannot be detected — the
  words share nothing orthographically. `Luna, delete` does not recover it:
  it operates on sentences, not words. See §6 *What this cannot do*.
- **Cross-utterance repair.** A repair only reaches words in the current
  window. After a pause the §5 merge usually puts them back in it, but once the
  grace closes the false start is out of reach.
- Repetition-loop detection (compression-ratio heuristic). Less pressing than it
  was — the eager trim band (§3) brought the measured buffer maximum from 30.0s
  to 15.0s, so the long-generation regime is entered far less often — but the
  token budget still bounds the damage without detecting the loop.
- **No scrollback in the live view.** The document can grow past the screen and
  older sentences scroll out of sight. They are still in the document, still
  deletable and still printed at exit — but there is no way to look at them.
- **The trim seam is still a seam.** Cutting at the longest pause makes it land
  at a sentence boundary most of the time, but "most of the time" is not always,
  and nothing re-punctuates across it the way the §5 merge does. The eager band
  (§3) made trims roughly twice as frequent without costing a seam on the
  measured passage — but that is one passage, and the failure mode it would show
  up as is a stranded half-sentence, which is exactly what latest-wins produced
  when it was tried. Worth re-checking against real dictation before trusting
  the frequency further.
- **Duty cycle no longer has a warning at all.** The notice now reports commit
  latency (§3), which is the right quantity for the speaker — but it means
  nothing watches the case where inference genuinely exceeds realtime and the
  capture channel backs up. The tick stretch makes that self-correcting in
  steady state, and `CAPTURE_BACKLOG` absorbs the bursts, so this is a gap in
  reporting rather than in behaviour.
- **Undo does not reach a `keep`.** Only text removal is recorded, and `keep`
  removes nothing — so there is no way to reopen a settle the speaker ended
  early. The recourse is `delete` and re-dictating.
- **Every finished thought costs a second forward pass**, since the hinted
  re-decode is no longer gated (§7). Measured cost is one pass on a buffer that
  was just decoded, at the moment the loop is already off its tick. The bounded
  tail described in §7 is the fix if it starts to matter; it is not built.
- **`keep` is the weakest verb here and is a removal candidate.** It moves no
  text of its own — the merge files the utterance before the verb runs — so it
  bends the "every verb must move text" rule to "moves text in time". §6 has
  the full argument both ways.
- **Session files accumulate under `--persist`.** Nothing prunes the state
  directory, and `--resume` only ever offers the newest one — an older orphan is
  reachable by path but is never mentioned.
- **Resume restores text, not session state.** The undo stack and the audio are
  gone, so a resumed session cannot undo what the previous one did.
- **`copy` is macOS-only** (`pbcopy`), which matches D2 but would need replacing
  if the pipeline ever left Apple Silicon.
