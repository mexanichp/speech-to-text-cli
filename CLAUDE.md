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
can say `Luna, delete` rather than discovering it afterwards. Repair is at
sentence granularity and that is deliberate; see D9.

Text moves through three tiers, in one direction, and never back: dim while the
recogniser may still rewrite the words, plain and marked once it is finished
with them, plain and unmarked once the cleanup pass has read them in context.
The deliverable is the third tier.

---

## 2. Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | **Qwen3-ASR** as the ASR model | Best measured accuracy on accented English at a cost this class of machine can sustain |
| D2 | **MLX** runtime | Only mature local runtime for this model on macOS |
| D3 | **Python sidecar**, NDJSON over stdio | MLX is Python/Swift-only. Rust owns the pipeline, Python owns the forward pass |
| D4 | **Sliding window + LocalAgreement-n** | Produces provisional text while speaking; native streaming is vLLM-only |
| D5 | **VAD for gating and segmentation** | Suppresses hallucination on silence, supplies the endpoint |
| D6 | **Commit latency is the budget**, and the system holds it rather than asking the speaker to | It is what the speaker sees. Duty cycle is pinned by construction and says nothing |
| D7 | **The transcript is a live document, not scrollback** | `delete` must reach a sentence filed minutes ago, and scrollback cannot be taken back |
| D8 | **Trim the buffer at a pause, never mid-word** | A word is never sliced; what is bounded is how far behind the transcript falls. Landing on a filed sentence end is preferred, not required — see D12 |
| D9 | **Wake word for document operations only** | "throw that away" has no acoustic signal; a self-repair does |
| D10 | **Autosave every change; delete only on a clean exit** | The paths that lose the transcript cannot run cleanup, so the file survives them |
| D11 | **Commitment is logical, not timed** | A sentence files once further sentences sit behind it unrevised |
| D12 | **The buffer is held short and the seams are repaired afterwards** | Latency is set by buffer length. Nothing else can be traded for it |
| D13 | **A second, text-only model cleans up settled text**, in its own process | Quality wants whole-passage context, latency wants none. One decode cannot serve both |
| D14 | **Three tiers, and the tier is visible** | The cleanup pass moves plain text, so a two-tier screen would be lying |
| D15 | **Cleanup batches overlap by one sentence** | A seam landing on a batch boundary is otherwise unjoinable: one half finalizes before the other is read |
| D16 | **The pass is told which full stops the trim invented**, by sending those lines without one | The host knows from the audio what the pass cannot infer from the text |

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

### D12: hold the buffer short

Inference grows with buffer length and commit latency grows with both, so the
buffer is the only lever that moves what the speaker sees. Measured on real
speech: 92 ms at a 5 s buffer, 141 ms at 10 s, 501 ms at 44 s, 2507 ms at 190 s.

The trim used to refuse any cut that did not land on a filed sentence end,
because a bad cut damages the document permanently. That refusal is what pinned
the buffer: on a real recording it turned down 19 of 22 candidates and the
buffer reached 88 s against a 30 s setting. Cutting anyway is only affordable
because D13 repairs what the cut breaks.

### D13: the cleanup pass

A separate process, a separate model, no audio. It reads a run of settled
sentences and says where the sentence boundaries should have been.

It runs **off the latency path and is never waited on**, except at exit and on
a spoken `copy`. That is not a style preference: it shares the GPU with
recognition, and a blocking pass here is exactly the condition that starves the
main loop (§8). Both exceptions are moments when the speaker has asked for the
finished text and has stopped talking.

Its reply is **checked, not trusted**, and the check runs in both directions.
`cleanup::check` rejects a reading that carries a content word the input did not
have, and one that has lost more than a quarter of the content words the input
did have. Function words are exempt either way, since fixing agreement broken by
a bad split is the point.

Both halves were found by measurement, not foresight; see §8.

### D15: batches overlap

The pass reads a run of settled sentences and finalizes it. Batches used to abut
exactly, so a seam falling on a batch boundary put its two halves in different
passes and no pass ever saw both: the first half went finalized and out of
scope while the second was still settled. On a real session the speaker got
`but it seems.` as one finished sentence and `That it happened exactly between
two paragraphs...` as the next, one tier apart, with nothing left in the
pipeline that could put them back together. That is the defect this project was
handed as a bug report.

The last sentence of an accepted batch therefore stays settled and opens the
next one. The property to hold is not that batches overlap but that **no
adjacent pair is ever split across batches**, which is what makes every seam
reachable by some pass.

Carried only when the reply leaves something behind to finalize. A reply that
joined the whole batch into one sentence has nothing to hand forward, and
holding it back would leave the next batch starting where this one started and
growing without bound.

The alternative was to include the previous batch's tail as read-only context
and let the pass rewrite it. That edits finalized text, which invariant 4 does
not allow. Carrying costs one sentence of delay to the third tier and allows
nothing new.

### D16: the seam mark

§11 called this the one structural change worth the risk, and the reason is that
the pass is given text and nothing else while the host knows from the audio
exactly which full stops it invented. A forced trim cuts at the best pause
available, which is often not a sentence end, so what it files stops where the
audio stopped. Both halves then read as grammatical sentences and nothing in the
words says one of them was severed.

The mark is the **absence** of a full stop: a line the trim cut is sent without
one. That is the truth about the line, it costs no tokens, there is no marker
for the model to copy into its reply, nothing to strip out afterwards, and no
word for the content check to read as invented. The pass's own rules already say
to join a line that carries on into the next.

It stays a hint rather than an instruction because the reply is re-punctuated
per line before it is spliced; see §8.

**Ablated, because the first write-up of this section guessed wrong about it.**
Run with the mark suppressed and everything else in place, boundary precision on
recording 3 comes out 0.69-0.76 over two runs, against 0.75-0.86 over three runs
with it. **Recall does not move**: 0.90-0.95 either way.

Two things follow, and only one of them is solid. The over-joining this file
first blamed on the mark is not the mark's; it belongs to the deduplication, and
recall says so at every sample. That the mark *buys* precision is suggestive
rather than established — the medians are 0.86 against 0.72, but the ranges
touch at 0.75, and precision on this recording turns out to vary by 0.11 within
one configuration. §8 says a single run is not evidence; three runs are not much
more, and the honest reading is that the effect is real in direction and not
yet pinned in size.

Invariant 3 does not forbid it. That invariant is about transcript text reaching
the *recognition* model, and this pass is handed transcript text by
construction. What it gets here is less than before, not more.

### D14: three tiers

Settled text is not final: the cleanup pass may re-join or re-split it. A screen
showing two tiers would have to either dim settled text, claiming the recogniser
might still change the words when it will not, or leave it unmarked, claiming it
is finished when it is not. The tier goes in the gutter, which already exists to
say which layer a row belongs to.

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
│      └→ logical commitment → document (settled sentences)   │
│          └→ seam de-duplication at the filing boundary      │
│              └→ spoken commands (endpoint only)             │
│                  └→ autosave + alternate-screen live view   │
│                      └────────┐                             │
└───────────────────────────────┼─────────────────────────────┘
                                ↓ NDJSON / stdio, never awaited
┌────────── Python sidecar (mlx-lm, text only) ───────────────┐
│  re-punctuate, re-join, re-split a run of settled sentences │
└───────────────────────────────┬─────────────────────────────┘
                                ↓
┌───────────────────────────────┼─────────────────────────────┐
│  content-word check → finalized sentences                   │
│      └→ prose to the clipboard and to scrollback, on exit   │
└─────────────────────────────────────────────────────────────┘
```

Rust does everything except the forward pass. The transcript state machine is
the actual engineering: no library ships it, because the commit policy is
application-specific.

---

## 4. Layout

```
src/main.rs        orchestration: tick loop, trim bands, endpoint handling,
                   `Settle` (adaptive hold), `Dip` (cut candidates), commands,
                   `Pass` (the cleanup pass and the one batch it has out)
src/audio.rs       cpal capture, downmix, resample to 16kHz; --simulate path
src/vad.rs         earshot + RMS floor, open/close hysteresis, two silence runs
src/transcript.rs  LocalAgreement window, the sentence document, logical
                   commitment, the filed-text overlap test, undo
src/text.rs        sentence splitting, opener veto, continuation deferral
src/repair.rs      self-repair detection ("regularing… regular")
src/command.rs     wake-word command parsing and the vocabulary hint
src/cleanup.rs     cleanup sidecar handle, the content-word check
src/store.rs       session autosave, resume, clipboard
src/trace.rs       filing-path trace, off unless STT_TRACE names a file
src/ablate.rs      switches one guard off for a measurement, via STT_ABLATE
src/render.rs      alternate-screen live document, wrap-safe row layout
src/sidecar.rs     subprocess handle + NDJSON protocol
sidecar/asr_sidecar.py       MLX inference; stdout is protocol, stderr is logs
sidecar/cleanup_sidecar.py   mlx-lm text repair; same protocol discipline
scripts/oracle.py            whole-recording transcription: the ceiling, §9
scripts/score.py             WER and sentence-boundary scoring, §9
scripts/bench.py             repeats and interleaves configurations, §9
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
| `trim-forced` | the trim cut where no filed boundary endorsed it | a whole-buffer decode, located by the head |
| `cleanup` / `cleanup-refused` / `cleanup-stale` | the cleanup pass returned | settled text, re-read as a passage |

`trim-forced` is the **ordinary** path now, not a backstop. Endorsement by
`spent_by` is preferred where it happens to hold, because it is free; where it
does not, the cut is taken anyway and the text comes from the whole-buffer
decode. See D12.

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

| band | eligible gaps | endorsement |
|---|---|---|
| below the eager band | none | |
| eager band | `SENTENCE_DIP_FRAMES`, a sentence boundary | required |
| past the threshold | `DIP_FRAMES`, a clause | preferred, not required |
| past the desperate multiple | any recorded silence, down to `MIN_DIP_FRAMES` | preferred, not required |
| nothing eligible, past `LAST_RESORT_MULTIPLE` | `last_resort`: any gap at all with refusals ignored, else the quietest frame in the buffer | none |

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

Past the threshold the cut happens regardless. Filing the severed head is what
damages the transcript, so this path files from a whole-buffer decode instead,
using `words_covering` to locate the cut inside it.

Three things then clean up after it, and every one was found by running a real
recording rather than by reasoning. They are not alternatives; each catches a
shape the others structurally cannot see:

- **`covered_from_a_boundary`.** An endpoint files whole sentences while their
  audio sits in the held utterance, and a continuation splices that audio back
  to the front of the buffer. The filed record then starts *earlier* than the
  next head decode, so the front match misses, and that decode is shorter than
  the record's tail, so the whole-tail match misses too. The text reads as
  unfiled and is filed a second time. The third alignment anchors on recorded
  sentence ends, which are the only places a decode of restored audio can begin.
- **`drop_seam_repeat`.** The head's last words were decoded with no
  right-context while the next decode covers the same speech with the rest of
  the sentence behind it, so the overlap appears twice, spelled differently the
  second time: `so this tool has to recognize` against `So this too has to
  recognize`. The *earlier* copy is dropped, because it is the one decoded
  blind and because dropping the later one leaves a sentence starting
  mid-clause. Two exceptions, both measured. The two copies need not be the same
  length, so the match runs through `same_speech` rather than `same_run`; see
  §8. And where the earlier copy is already **finalized** it is not the
  pipeline's to rewrite, so there the incoming copy gives up the overlap
  instead. Refusing to act was the third option and the wrong one, since nothing
  else can see a seam: `already_in_the_document` excludes a run ending at the
  tail's end, which is exactly what a seam is.
- **`already_in_the_document`.** The same fault as the first, at a place no
  boundary can anchor: a forced trim files a slice starting *inside* a sentence
  the document already holds, `To know is how exactly the algorithm commits the
  messages` arriving behind a sentence that already says it. Here the *incoming*
  copy goes, the reverse of the seam, because those words are already in place
  and only what follows them is new.

  Asked repeatedly rather than once, because one answer exposes the next:
  dropping a matched prefix leaves a remainder that is a different question. It
  also answers for a run below `SEAM_MIN`, where no substitution budget is
  affordable and the whole fragment must therefore match outright. Both were
  found the same way, by a one-word sentence reading `Converge.` A stub like
  that is the worst of the outcomes it sits between: it keeps none of the
  meaning and still costs a sentence boundary.

**Order is load-bearing.** The seam is removed first, which takes the repeated
words out of the document's tail; anything still matching after that is held
elsewhere and is a different fault. The other order fails, because a repeat
running to the end of the document is a seam but its shorter prefixes do not
reach the end, so the stale check claims it one word early and truncates the
wrong copy.

Both compare through the seam matchers rather than `align`, which is tuned for
insertion and far too permissive here. **Both ends of a run are
anchored exactly**, and that is not decoration. A purely proportional budget
lets a thirteen-word run with three wrong words on the end read as a match, and
those three words are exactly the new ones the repeat is followed by.

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
| `copy` | the whole document, to the system clipboard, cleanup pass first |
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

`copy` waits for the cleanup pass, which is the one thing on the main loop that
does. It is the only command whose entire purpose is the finished text, and the
document as it stands always carries seams the pass has not reached: the pass
runs `LAG` sentences behind and needs `MIN_BATCH` of them, so a tail of several
sentences is uncleaned at any moment. Copying that silently is the worst of the
options, because an unrepaired seam is a grammatical sentence that stops early
and looks like nothing is wrong. Measured on recording 2: `copy` handed over six
uncleaned sentences of fourteen, and one of them was cut mid-clause.

The wait is bounded, and what the budget does not cover is **said** rather than
finalized quietly. See invariant 11 for why waiting here is safe and waiting
elsewhere is not.

The flush runs with **no lag**, which is the one thing `LAG` exists to prevent,
so `copy` also zeroes the `discard` budget. Without that, a `copy` that re-joins
a sentence this utterance filed into one holding older text leaves a following
`discard` to take the older text with it. Zeroing is right rather than merely
safe: the flush finalizes what it reads, and nothing finalized is in flight.

`keep` drops the held audio and starts the next utterance short, which pausing
does not do: an endpoint holds the audio and a continuation merges it back.

It used to be what the latency notice told the speaker to say. That notice now
states the fact and asks for nothing, because handing the speaker a chore
mid-sentence is not a way to run a pipeline, and because the buffer bound of D12
is what actually holds the latency.

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
| `--trim-after-s` | the buffer length being held to. This is the latency control |
| `--cleanup-model` | text model for the cleanup pass. 4B is the floor that works; see §8 |
| `--no-cleanup` | leaves trim seams in the transcript. Only useful for measuring what the pass is worth |
| `--open-ms` | duration, not level: this is the keyboard defence. Past the pre-roll it clips first words |
| `--rms-floor` | dBFS. Set above the room and below the voice. Needs `allow_negative_numbers` on the clap command, or the negative value parses as a flag |

**`--trim-after-s` is now the buffer length and therefore the latency control.**
It used to do nothing, because the trim could only cut where a filed sentence
ended and so the buffer was pinned by `KEEP_SENTENCES` instead. Swept across its
whole useful range it did not move the buffer at all. Since D12 dropped the
endorsement requirement it does what its name says: on the reference recording
the peak buffer went from 88.6 s to 25.2 s and p95 commit latency from 3.63 s to
1.31 s.

**`--endpoint-ms` is not a latency lever.** Swept over 600 / 900 / 1200 ms it
moved the utterance count from 51 to 45 to 42 and left the peak buffer at 62 s
throughout. What bounds the buffer is whether the trim may cut, not how often
the speaker endpoints.

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
   recogniser may rewrite these words; plain means it is finished with them; an
   unmarked gutter means the cleanup pass has read them too. Text moves through
   those in one direction and never back. The live view and the piped path
   produce identical text.
   
5. **Commands are parsed from finalized utterances only**, and the audio that
   carried one is dropped, never retained for continuation. Retention is the
   only thing that could replay a command.
6. **`scan()` reports at most one boundary and stops there.** One loop
   iteration can cover more audio than the endpoint threshold, so a single batch
   can hold one utterance ending and the next beginning. Running past an
   *endpoint* reports the onset before the caller has a pending utterance to
   merge into, and leaves the detector open so no further onset ever arrives:
   the held utterance becomes unreachable, not merely late. Running past an
   *onset* starves the pipeline; see §8.

7. **The main loop blocks for one chunk and then drains the rest**, up to
   `MAX_INGEST_SAMPLES`. Several passes run off the tick and can land back to
   back; taking one chunk per iteration left the loop behind when the next pass
   started, and dropped capture buffers do not degrade gracefully, they shred
   the audio. The bound discards nothing: what is left stays in the channel and
   arrives on the next iteration, which is immediate.
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
11. **Nothing on the main loop ever waits on the cleanup pass**, except the two
    flushes: at exit, and on a spoken `copy`. It shares the GPU with
    recognition, so a blocking pass anywhere else reproduces the starvation of
    §8 exactly. Both exceptions are points where the speaker has asked for the
    deliverable and is not talking — `copy` is parsed only from a finalized
    utterance, so the endpoint has just fired and the buffer has just been
    trimmed. `Pass::flush` is the only thing that may block, and it is bounded.

    A flush **collects the batch already out before submitting another**.
    Replies come back in the order the batches went out, so submitting first
    pairs every reply with the following batch's range, and the shift runs to
    the end of the flush. The symptom is the tail of the document arriving
    unrepaired, which is indistinguishable from a pass that simply had nothing
    to say.

12. **A reply line without a full stop is punctuated before it is spliced**,
    never run into the line after it. The batch is sent with the stop removed
    from every line the trim cut, so without this a pass that declined to join
    has its reply read as a join and the mark stops being a hint. See §8.

13. **The cleanup pass may restructure text but may not introduce a word.**
    `cleanup::invents_content` is the check, and every path that refuses still
    finalizes the run where it stands, or it would be offered to the next pass
    forever.

14. **`Transcript::filed` follows the recogniser, not the document.** It exists
    to strip already-filed text out of the *next hypothesis*, and the next
    hypothesis comes from audio, so it has to describe what the audio decodes
    to. The cleanup pass and `drop_seam_repeat` both edit the document and
    deliberately leave it alone.

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

### The cleanup model

Measured on the reference recording's own damage, with a strict prompt:

| model | on a five-fragment split | on a clause the audio lost |
|---|---|---|
| Qwen3-0.6B-4bit | no-op: joined the lines, left every fragment | no-op |
| Qwen3-1.7B-4bit | no-op | **invented** `they just wait for someone else` |
| Qwen3-4B-4bit | rejoined it correctly | declined to invent |

4B is the floor. Below it the pass either does nothing or fabricates, and
fabrication is the worse failure because it is fluent.

**A one-directional check misses the commoner failure.** Guarding only against
invented words looks sufficient and is not: handed eight sentences, 4B returned
four, every surviving word legitimately said, and a third of the transcript
gone. It reads as a clean reply. The check has to test that what was said
survives as well as that nothing new appeared, and the second half is the one
that fires in practice.

**A bigger buffer is not a safer buffer.** Swept over `--trim-after-s`, 12 s was
best on *both* axes, 8.6% WER and 0.91 boundary F1, against 16.4%/0.61 at 16 s
and 36.8%/0.33 at 20 s. Longer buffers file longer runs, longer runs make longer
cleanup batches, and a longer batch is what provokes the summarising failure
above. The flags interact; do not tune one of them alone. 4B also kept the
recogniser's own errors (`too`, `poses`) rather than smoothing them away, which
is the behaviour to preserve: this pass fixes structure, not hearing.

**Prompt shape matters more than prompt length.** Rewriting the rules as a
numbered procedure made the model narrate its steps instead of emitting text,
and WER went from 11.2% to 46.7%. Prose rules with worked examples work; an
ordered checklist does not.

**A batch of one sentence is useless.** The pass exists to join sentences split
across a seam, and handed them one at a time it can only re-punctuate each in
isolation. `MIN_BATCH` is why, and it was found by wiring it up without one.

### The loop starvation

Found by running a real recording rather than by reading the code.

The loop drains the whole capture backlog per iteration, and `scan` reported an
onset and the endpoint behind it in the same call. The caller merges on the
onset and returns to the top of the loop on the endpoint, so that batch skipped
both the buffer trim and the live inference pass. Self-reinforcing: the buffer
grows, the next pass is slower, the batch after it spans more audio still.

On the reference recording the pipeline **stopped running live passes
altogether at 160 s of 190 s** and the buffer reached 88.6 s against a 30 s
setting. Confirmed by re-running with the faster 0.6B model, where the loop
stayed alive and the buffer held at 61 s.

`scan` therefore reports at most one boundary per call. The symptom this
produces is worth recognising: text still files, so the transcript looks like it
is working, while the live view goes quiet and never recovers.

### Measuring this pipeline

WER alone cannot see what the cleanup pass does. Stripping punctuation to
compare word streams makes a sentence split in two identical to one left whole,
and splitting is the defect being repaired. Score **sentence boundaries
separately**; on the reference recording the shipped pipeline reads 8.6% WER
against a baseline 7.9%, and boundary F1 0.91 against 0.86.

Read the trade the same way: a shorter buffer costs word accuracy at the seams
and buys latency and boundary accuracy. p95 commit latency 3.63 s to 1.31 s,
peak buffer 88.6 s to 25.2 s, live passes per session 100 to 184.

**The pipeline is not deterministic across runs, so measure more than one.**
Inference timing decides when ticks and trims land, which decides what is in the
buffer, which decides the text. The same binary on the same file did 13, 15 and
19 forced trims on three consecutive runs. Spread over those runs: WER 7.9% to
8.6%, boundary F1 0.78 to 0.91. A single run is not evidence that a change
helped, and one bad run cost an afternoon before this was noticed.

It is not reliably *in*determinate either, which is the trap. At the buffer
length now held to, two consecutive runs of recording 1 came out byte-identical,
same tick count and same trim count. A run a couple of hours earlier on the same
binary differed by three words and took a visibly different path: 170 ticks
against 185, five endpoints against six. So a pair of matching runs says nothing
about the next one. Measure across a gap, not back to back.

### The ceiling

The same two models, run over a whole recording in one pass, are what this
pipeline is trying to equal. `scripts/oracle.py` produces it. The number that
reframes the problem: **206 s of speech costs about 3 s of inference**, so the
ceiling is not expensive, it is only unavailable live. What the live path buys
is not throughput. It is seeing the words while you are still saying them.

Measured against the ceiling, on recording 2 (`golden_2`):

| | WER | boundary P | boundary R |
|---|---|---|---|
| before the exit-flush and `copy` fixes | 2.3% | 0.77 | 1.00 |
| after | 1.7% | 0.83 | 1.00 |

Two things this says that the aggregate numbers above do not.

**Almost every boundary defect is a spurious boundary, not a missing one.** On
recording 2 recall is 1.00 at both measurements: the live path invents
boundaries and never loses one. Recording 1 has a single genuine miss, and it is
a boundary that *moved* rather than vanished — the ceiling breaks after
`...the spoken language of a person.` and the live path breaks a clause earlier,
which scores once against recall and once against precision. So the defect to
chase is the same one in both cases: a boundary in the wrong place, which is a
trim seam. Nothing yet measured is a boundary the pipeline simply failed to
produce.

**The ceiling inherits the recogniser's errors and beats the buffer's.** On
recording 1 the ceiling scores 3.9% WER and 0.91 boundary F1 against the human
transcript, and its remaining errors are mishearings the short buffer also makes
(`poses` for `pauses`, `tool` for `two`). But it hears `and sometimes they just
keep it up`, which the live path renders `and sometimes they just so some`. §10
recorded that as a word the recogniser lost. It is not: it is a word the
*buffer* lost, and a longer window recovers it. That distinction decides whether
the fix is a better cleanup model or a better trim.

### The cleanup prompt copies itself out

The worked examples used to sit in the system message under `Input:`/`Output:`
headings. On a 327 s recording the model **copied one of them into its reply**,
opening a transcript with 33 words the speaker never said. A worked example
written inside the instruction is text sitting in the same turn as the text to
rewrite, and nothing in that format says which of the two to rewrite. Carried as
real user/assistant turns the boundary is the one the chat template already
draws, and the echo stops.

This is the same failure §8 records for the recognition sidecar's transcript
prompt, in the one place this file said a static prompt was safe. It is safe
from *self-reinforcement*, because those words never come from the speaker. It
was never safe from being copied out.

Two things it cost, and only one of them was visible. In the live path
`cleanup::check` refused the batch, which is the check working, but a refused
batch is a run of transcript left unrepaired, and the batch it hit was the exit
flush — the tail, which is the part the speaker reads. In `scripts/oracle.py`
there was no check at all, so **the ceiling silently inherited the invented
words** and every measurement scored against it was wrong. The oracle now runs
the same two-directional check the host runs; see §9.

### Recognising filed text

- **The edit budget may not floor at zero.** `align` spent a budget
  proportional to the shorter run, which meant a filed sentence below
  `OVERLAP_TOLERANCE` words had to be re-decoded *exactly* to be recognised as
  filed. One inserted word defeated it outright: `So, not sure what's going on
  there.` was filed at an endpoint, the continuation spliced its audio back, the
  re-decode came back `So I'm not sure what's going on there.`, every alignment
  failed on the second word and the speaker got the sentence twice. One edit is
  now allowed from `OVERLAP_MIN_RUN` words up. Below that a free edit really
  would match almost anything; above it the commoner failure is the reverse.
- **A seam is not only a substitution.** `same_run` compares position by
  position and so requires the two copies to have the same number of words. They
  routinely do not, because the blind copy has a word the later one lacks:
  `first paragraph and second paragraph connect, converge` against `First
  paragraph and second paragraph converge`, and `such artifacts like previous
  one` against `Such artifacts, like the previous one`. Both read as new text.
  `same_speech` anchors both ends exactly, as `same_run` does, and spends the
  same proportional budget on edits of any kind.

### What the deduplication is worth

The first comparison run through `scripts/bench.py`, four runs per arm,
interleaved, recording 3. `dedup-off` is `edit-floor`, `same-speech`,
`stale-loop` and `short-fragment` all switched off at once.

| | WER | boundary P | boundary R |
|---|---|---|---|
| shipping | 3.3 (3.3-3.3) | 0.86 (0.86-0.90) | 0.90 (0.90-0.95) |
| dedup-off | 7.4 (5.3-12.8) | 0.91 (0.86-0.95) | 0.95 (0.95-1.00) |

**WER separates and nothing else does.** The deduplication is worth between two
and nine points of WER, the largest single effect measured on this pipeline, and
that settles that it stays. Its cost in boundaries is **not established**: both
boundary ranges overlap, and the medians lean the way the earlier single-run
work claimed without the ranges backing it. "The dedup costs the recall" is the
leading hypothesis and still not a finding.

Note which arm is the unstable one. `dedup-off` spans 5.3-12.8% WER while
shipping does not move at all, because without the guards the transcript grows a
different set of duplicates depending on where the trims land. A guard that
removes a failure mode removes its variance with it.

### Batch size is the cleanup pass's strongest lever

Three arms, three runs each, interleaved, recording 3. `min8` moves
`MIN_BATCH` from 4 to 8; `quiet` offers a two-sentence batch once the document
has been still for longer than a settle.

| | WER | boundary P | boundary R |
|---|---|---|---|
| base (`MIN_BATCH` 4) | 3.3 (3.3-3.3) | 0.86 (0.79-0.86) | 0.90 (0.90-0.95) |
| quiet | 3.5 (3.5-4.0) | 0.82 (0.81-0.82) | 0.90 (0.85-0.90) |
| **min8** | 3.3 (3.3-3.3) | **0.95 (0.95-0.95)** | 0.90 (0.90-0.90) |

**`min8` separates on precision and costs nothing.** 0.95 against 0.86, no
overlap, with WER and recall identical. §11 had been claiming for a while that
the pass declines a join when it cannot see enough of the passage around the
seam; this is that claim with a number on it, and it is the largest quality
lever measured on this pass. What it costs is depth: the tail no pass sees
during a session is `LAG + MIN_BATCH`, so it went from seven sentences to
eleven. `copy` and the exit flush still cover it.

**The quiet batch is a loss, and it is the same finding from the other side.**
Offering two sentences when the speaker pauses is §11's own suggestion for
reaching that tail, and it separates *downward* on WER, 3.5-4.0 against 3.3. Of
course it does: a two-sentence batch is the "batch too small" failure already in
this section, and worse than merely useless, because what it finalizes it also
removes from the larger batch that would have come. The tail is better left
alone than cleaned badly, and the two levers were pulling opposite ways all
along.

### Reading a sweep of more than two arms

`bench.py` first reported one overlap verdict per metric across *all* arms, so a
single overlapping pair marked the whole row `OVERLAPPING`. On the sweep above
that hid the only result in it: `min8` separates cleanly from both other arms on
precision while those two overlap each other. Comparisons are pairwise now. A
sweep is not one question.

### Repetition inside one session is not enough

Those four shipping runs came out at 3.30% WER four times, to the digit. Earlier
the same binary produced 2.8%, 3.5% and 4.2% on the same recording. Nothing
changed in between except that the earlier runs were spread over an hour of
other work and these four were back to back.

§8 already said it — *measure across a gap, not back to back* — and this is what
ignoring it costs. `bench.py` interleaves arms, which makes the **comparison**
sound, but its ranges understate the true spread because consecutive runs share
a machine state. Read a separated comparison as real; do not quote the range as
the variance of the number.

### The rule that rode along with the prompt fix

Moving the examples into turns fixed the echo. The same edit also added a rule
telling the pass to drop a short line that repeats a neighbour, which was a
content change smuggled in with a structure change, in a prompt this file says
is shape-sensitive.

Ablated with `STT_CLEANUP_PROMPT` pointing at the same prompt minus that one
line: 2.8%/3.3% WER and precision 0.86/0.75 without it, against 2.8%/4.2% and
0.86/0.86 with it. **Inconclusive**, and the ranges overlap on both axes. It is
kept because the mechanism is sound — the pipeline demonstrably files short
duplicate fragments the host guards cannot always see, and `cleanup::check`
bounds what a drop instruction can cost — but nothing here is evidence that it
earns its place. Take it out first if the pass ever starts losing text.

### The forced trim: what did not work

**Re-decoding the whole buffer to place the cut made it worse.** The forced trim
locates the cut inside a whole-buffer decode, and when the head cannot be placed
it falls back on filing the severed head, which invariant 10 says is not
evidence. The obvious repair is to stop standing the committed agreement window
in for that decode and take a fresh whole-buffer pass whenever placement fails.
Measured on recording 3, that took WER from 5.1% to **11.2%** and insertions
from 16 to **42**, and the count of unplaceable heads did not even fall. The
extra pass files text rounded up to a sentence boundary past the cut, whose
audio is still buffered and is filed again later. Do not re-try this without an
answer to that.

### Keeping the seam mark a hint

The batch is sent with the full stop removed from every line the trim cut (D16),
and `apply_cleanup` flattens the reply before re-splitting it on punctuation. So
a pass that read the mark and **declined** to join had its reply read as a join
anyway: the two lines met with no punctuation between them. Measured, that cost
two sentence boundaries the recogniser had got right, and it is what turns a
hint into an instruction. Each reply line is therefore re-punctuated before the
reply is flattened: one sentence per line is what the pass is asked for, so a
line without a stop is one it forgot to punctuate.

Recovering that took boundary precision from 0.75 to 0.86 and WER from 3.5% to
2.8% on the same recording.

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

### Scoring a change against the ceiling

`scratch/` holds the real recordings. It is gitignored, so this is the one part
of the project a fresh checkout cannot reproduce; the procedure is here so that
whoever has the files can re-derive everything from them.

The reference is **not** a human transcript. It is the same two models run over
the whole recording at once, with the full right-context the live path can never
have. It runs the **same content check the host runs**, and that is not
decoration: unchecked, its cleanup pass copied a worked example out of its own
prompt and opened `oracle_3.md` with 33 words nobody said. A reference carrying
invented text is worse than no reference, because every later measurement is
scored against it. `scripts/oracle.py` mirrors `cleanup::check`; the Rust one is
authoritative and they are kept in step by hand. That makes it the best these weights can do on this audio, which is the
target §11 sets, and it is cheap enough to regenerate whenever a model changes:
206 s of speech takes about 3 s of inference.

```sh
ffmpeg -v error -i scratch/<recording>.mp4 -vn -ac 1 -ar 16000 -acodec pcm_s16le g.wav

# The ceiling. Writes the cleaned transcript, and the raw ASR beside it as .asr.
.venv/bin/python scripts/oracle.py g.wav -o scratch/oracle_<n>.md

# The pipeline under test.
./target/release/speech-to-text-cli --simulate g.wav --language en > out.txt

# Both numbers, and the word alignment behind them.
.venv/bin/python scripts/score.py out.txt --ref scratch/oracle_<n>.md --wake Luna --diff
```

`--wake` drops sentences naming the assistant. The speaker's `Luna copy` is an
instruction the pipeline correctly keeps out of the document and the ceiling
correctly transcribes, and without this the feature scores as five deletions.

Read **precision and recall separately**; the summary F1 hides which way the
pipeline is wrong. Precision is what a spurious boundary costs and is the number
that moves. A recall miss is rarer and usually means a boundary *moved*, which
costs one of each, so a change that trades them evenly is not an improvement
however the F1 reads.

The human transcript beside recording 1 is still worth keeping, because the
ceiling inherits the recogniser's mishearings and a human does not. Use it to
answer "did the model hear this?", and the ceiling to answer "did the streaming
cost us anything the model heard?". They are different questions and only the
second one is actionable here.

### Comparing two configurations

**A single run cannot answer any question this project asks.** §8 says the
pipeline is not deterministic; what it did not say, until it was measured, is
how wide that is. Boundary precision on recording 3 spans **0.11** across runs
of one binary, which is larger than most of the effects worth chasing. Two
findings were written up from single-run comparisons before this was noticed,
and one of them was wrong.

`scripts/bench.py` repeats each configuration and **interleaves** the arms:

```sh
scripts/bench.py scratch/g3.wav --ref scratch/oracle_3.md --runs 4 --wake Luna \
    --arm shipping \
    --arm no-mark:STT_ABLATE=seam-mark
```

Round-robin rather than one arm to completion, because an hour of runs drifts
and interleaving puts the drift on every arm instead of on whichever ran second.
It reports a median and a full range per arm, and says outright whether the
ranges overlap. **Overlapping ranges are not a result.** Reach for the median
only once they separate.

`STT_TUNE` moves a constant instead of switching a guard off, `name=number`,
comma-separated, listed by `src/ablate.rs::TUNABLE`. It is what makes the sweep
§11 keeps asking for a single command rather than a rebuild per point:

```sh
scripts/bench.py scratch/g3.wav --ref scratch/oracle_3.md --runs 3 --wake Luna \
    --arm min4 --arm min8:STT_TUNE=min-batch=8
```

Pairs in an arm are separated by `;`, because `STT_ABLATE` takes commas of its
own.

`STT_ABLATE` switches one guard off, comma-separated, listed by
`src/ablate.rs::KNOWN`. Environment rather than a flag, for the same reason
`STT_TRACE` is: apparatus, not a setting. A name it does not recognise is a
startup error rather than a no-op, because an arm that silently does nothing
produces two identical runs and the conclusion that the guard does not matter.

Ablating a prompt rule needs no rebuild. `STT_CLEANUP_PROMPT` points the sidecar
at a file that replaces the system prompt, examples excluded, so a single line
can be added or removed and measured against the same binary:

```sh
STT_CLEANUP_PROMPT=$PWD/scratch/variant.txt \
  ./target/release/speech-to-text-cli --simulate g.wav --language en > out.txt
```

Dependency notes:

- `rubato` is pinned; 5.x reworked its API around `audioadapter`.
- `cpal` renamed `Device::name()` to `description()` in the version used here.
- `earshot` needs exactly **256-sample** frames at 16 kHz.

---

## 10. Known gaps

**Read every measurement in this file with the caveat attached.** Most of it is
synthesized speech, which has clean prosody and no genuine hesitation. It is
evidence about *mechanism*, whether a path fires, whether a cut lands, whether a
string splits. The exception is the recording in `scratch/`, which is real
speech with a human transcript beside it, and which is what §8's latency,
boundary and WER numbers come from. It is one speaker on one topic.

### The trade this design makes

- **Word accuracy at the seams is slightly worse than a long buffer gave.** 8.6%
  against 7.9% on the reference recording. Bought: p95 commit latency from
  3.63 s to 1.31 s, and boundary F1 from 0.86 to 0.91. The right trade for live
  dictation, and the wrong one for batch transcription of a file.
- **The cleanup pass cannot recover a word a seam destroyed.** Where the live
  path decoded `and sometimes they just so some`, the speaker had said `and
  sometimes they just keep it up`. No text-only model can put that back, and one
  that tries is fabricating. It stays a fragment. This was recorded as a word
  the *recogniser* lost; §8 corrects that, because the whole-recording pass
  hears it. What lost it is the cut.

### The cleanup pass

- **One batch at a time.** A speaker producing sentences faster than a pass
  completes leaves settled text piling up behind the pass. It catches up in the
  next silence, on a spoken `copy` and at the exit flush, so this is a delay to
  the third tier rather than a loss.
- **A tail of `LAG + MIN_BATCH` sentences is never cleaned during a session**,
  and raising `MIN_BATCH` to 8 made that tail deeper, not shallower. It is the
  price of the precision the larger batch buys, and it is paid in the one place
  it is most visible: the tail is the speaker's most recent work, which is the
  part they are reading. `copy` and the exit both flush it. Offering a short
  batch to cover it was tried and is a measured loss; see §8. But it means the speaker's most recent work is also the work the
  pass has not read, which is why `copy` and the exit both flush. On recording 2
  that tail was six sentences of fourteen. See §11.
- **A seam whose halves both read as grammatical still survives the pass
  sometimes.** Measured: `...and seems like the buffer.` / `It is still
  overflowing a little...` D16 now tells the pass which of those full stops the
  trim invented, and D15 guarantees it sees both halves at once, which is what
  the two conditions for joining were. It still declines on some of them:
  recording 3 ends with about three seams unjoined out of a session, and that is
  now the largest single item in the gap to the ceiling.
- **The exit flush is time-boxed.** A long document that cannot be cleaned
  inside the budget exits with its tail finalized where it stood, seams and all.
- **`OVERLAP_MIN_RUN` buys an edit at four words, and four words is short.**
  The floor exists because a filed sentence below `OVERLAP_TOLERANCE` words had
  to be re-decoded exactly to be recognised, which cost a duplicated sentence
  (§8). The cost of the fix runs the other way: two genuinely different
  four-word runs that differ by one word in the middle now align, so a decode
  can read as already filed when it is not, and the trim then discards audio
  holding unfiled speech. `align` anchors neither end, which is what makes four
  words thin. Never observed; the measured failure needed seven words, so a
  floor of six would keep the fix and halve the exposure. Not changed, because
  trading one unmeasured risk for another is not an improvement.
- **A carried sentence is still the seam logic's to edit.** D15 leaves the last
  sentence of a cleaned batch `Settled` so the next batch can join across it,
  and `drop_seam_repeat` refuses only `Finalized` text. So a sentence the
  cleanup pass has just rewritten — possibly by joining two halves — can have
  its tail truncated by a later seam drop, which the old abutting batches
  prevented by finalizing it immediately. Never observed, never looked for. If
  it fires, the fix is to refuse on `cut == false` rather than on the tier.
- **`Sentence::cut` is not persisted.** A resumed session comes back with every
  seam mark cleared, so the cleanup pass re-reads that text with no idea where
  the trim cut it. A special case of resume restoring text rather than state,
  but a new one, and it is silent.
- **The recall cost of the deduplication is not attributed.** Three changes
  landed together — the edit floor above, `same_speech`, and asking the stale
  check repeatedly — and recording 3 lost boundaries that recording 1 and 2 did
  not. Whichever of the three it is, the mechanism is the same: a repeat that
  spans a whole sentence takes the sentence, and the boundary with it.
- **The content-word check is a crude stemmer over a fixed function-word list.**
  It fails safe, rejecting a legitimate repair rather than admitting an invented
  word, but a rejection is silent apart from the trace.
- **A reply that is fluent, word-legal and wrong is accepted.** `cleanup::check`
  bounds what a reply may add and what it may lose; it cannot tell whether the
  restructuring was right. `scripts/score.py` now puts a number on the outcome
  (§9), but only over a whole recording, and only when someone runs it.

### The buffer and the trim

- **`FILED_MEMORY` caps how long a head decode can be and still be endorsed.**
  Not reachable at the buffer lengths this now holds to.
- **The forced path still leaves `filed` describing audio that has been cut**,
  contained by `covered_from_a_boundary` and `drop_seam_repeat` rather than
  fixed. Both are recognisers of a mess rather than a reason it cannot occur.
- **`drop_seam_repeat` can fire on a genuine repetition.** A speaker who ends
  one sentence and begins the next with the same four words loses the first
  copy. Judged rarer than the seam it exists for.
- **The forced trim path executes spoken commands**, which invariant 5 says only
  a finalized utterance may do. Pre-existing, and it matters more now that this
  path is the ordinary one: a false detection mid-buffer deletes a sentence.
  Never observed on the reference recording.
- **A long endpoint pass still stalls the loop.** Two whole-utterance decodes
  run at each endpoint, and on the reference recording that left gaps of up to
  7.4 s between live passes even with the buffer bounded. Bounded now, but not
  small.

### Speech and segmentation

- **The first pause past the settle floor is always lost.** A pause is only
  measurable once it has ended, so a speaker whose habit exceeds the floor pays
  one fragment before `Settle` adapts.
- **`Settle` learns from silence, not from meaning.** A speaker who stops
  mid-topic and one who has finished are indistinguishable to it.
- **Paragraphs come only from an expired hold.** A speaker who never pauses for
  the full settle gets one paragraph, however long the session.
- **Arbitrary substitutions are undetectable.** `Tuesday`/`Wednesday` shares no
  prefix, and `delete` does not recover it, since it operates on sentences
  rather than words. Accepted: re-saying the sentence is the repair.
- **Cross-utterance repair.** A self-repair only reaches words in the current
  window.
- **Sentence splitting never breaks after a one-letter word**, so `The answer is
  A. It follows.` is one sentence. The initials guard is worth more than the
  case.

### Reporting and persistence

- **Nothing watches the duty cycle.** The notice reports commit latency, which
  is the right quantity for the speaker, but nothing reports inference genuinely
  exceeding realtime.
- **Repetition-loop detection** (the compression-ratio heuristic) is
  unimplemented. The token budget bounds the damage without detecting the loop.
- **Session files accumulate under `--persist`**, and `--resume` only ever
  offers the newest.
- **Resume restores text, not session state.** The undo stack and the audio are
  gone, and resumed sentences come back as settled, so the cleanup pass runs
  over them again.
- **The session file is one sentence per line and carries no tier**, so a
  resumed session cannot tell what had already been cleaned.

### Deliberately not gaps

Scrollback in the live view, word-level spoken repair and a non-macOS clipboard
were all considered and are **not** wanted. The tool is for one speaker on one
machine who reads the last few lines while talking, re-says a sentence when it
comes out wrong, and pastes the result somewhere else.

A standing WER harness was on this list and has come off it. What is wanted is
still not CI: it is the two scripts in `scripts/`, run by hand against a
recording, because §8 records that this pipeline is not deterministic and a
gate that flaps is worse than no gate. See §9 for how, and §11 for why.

---

## 11. The gap to the ceiling

**The goal, stated as a measurement.** The transcript that comes out of a live
session should be the transcript `scripts/oracle.py` produces from the same
audio. That is the same models with the same weights, so the gap is entirely
attributable to hearing the speech a few seconds at a time, and it is a gap the
scripts in §9 can put a number on after every change.

It is closer than it looks. On recording 2 the live path is 1.7% WER and
boundary precision 0.83 against the ceiling, from **two** spurious sentence
boundaries in a 174-word passage. Neither is a hard problem in general. Both are
the same specific one.

**The ceiling is only a ceiling if it is checked.** Nothing above holds when the
reference is wrong, and it was: see §8 on the cleanup prompt copying itself out.
Recording 3 first scored 12.3% WER against a contaminated `oracle_3.md` and
6.0% against a corrected one, on the same transcript. Regenerate the reference
after any change to the cleanup prompt or model, and read the first few lines of
it before believing a number.

### The scoreboard

`scratch/` is gitignored, so this is the inventory rather than the data:

| file | what it is |
|---|---|
| `<date>.mp4`, `golden_2.mp4`, `golden_3.mp4` | the recordings, real speech, one speaker |
| `<date>.md` | a **human** transcript of recording 1 |
| `golden_2.md` | recording 2 as the shipped pipeline heard it, kept as the symptom |
| `golden_3_actual_current_transcript.md` | recording 3 as it was heard when it was reported, kept as the symptom |
| `golden_3_ideal.md` | recording 3 as the speaker meant it, written out by hand |
| `oracle_1.md`, `oracle_2.md`, `oracle_3.md` | the ceiling, from `scripts/oracle.py` |
| `oracle_*.md.asr` | the ceiling before its cleanup pass |

Recording 3 is the speaker describing these defects while provoking them, which
makes it the sharpest of the three: the transcript names the artifact it
contains.

Live path against the ceiling. Recording 3 is quoted as a range over two runs,
because §8 records that a single run is not evidence:

| | WER | boundary P | boundary R |
|---|---|---|---|
| recording 1, before | 6.6% | 0.82 | 0.82 |
| recording 1, after | 6.5% | 0.82 | 0.82 |
| recording 2, before | 1.7% | 0.83 | 1.00 |
| recording 2, after | 1.7% | **1.00** | 1.00 |
| recording 3, before | 6.0% | 0.83 | 1.00 |
| recording 3, after | 3.3% | **0.95** | 0.90 |

Read that trade honestly.

**Recording 2 is now exact.** The two spurious boundaries this section was
written to complain about are gone, at the same WER. It is the shortest and
cleanest of the three, so treat it as the existence proof rather than the
typical case.

**Recording 1 did not move at all**, on any of the four numbers, and did not
move again when `MIN_BATCH` doubled. It is the recording with the long
deliberate pauses, so it endpoints cleanly and forces few trims, and almost
nothing here applies to it. That is the result to want from a change aimed at
trim seams: no effect where there are no seams. Recording 2 is the same story at
the other end — already exact, and unmoved by the batch change.

**Recording 3 traded one boundary for everything else.** WER roughly halved,
duplicate insertions went from 14 to 4, and precision went from 0.83 to 0.95.
Recall fell from 1.00 to 0.90, which is two boundaries: one of them is the
*reference's* fault, since the ceiling breaks after `So this version is far from
ideal.` where the speaker ran straight on and the live path is right. The other
is real. §8's ablation places that cost on the **deduplication** rather than the
seam mark, though the boundary ranges there overlapped, so it stays a hypothesis
with the medians behind it.

§9 says to read the two numbers separately and not to accept an even trade. This
one is not even, and on two of the three recordings it costs nothing at all.

The `before` rows for recordings 1 and 2 are this file's historical numbers,
measured against oracles built with the pre-fix cleanup prompt. The oracles were
regenerated for the `after` rows, so those two comparisons are not perfectly
controlled. Recording 3 is, and it is the one the changes were aimed at.

Regenerate the whole table rather than one cell.

### What the gap is made of

Every defect measured so far is a **trim seam**: the audio buffer was cut where
the speaker did not finish a sentence, and the two halves were decoded and filed
as though each stood alone. It shows up three ways, in decreasing order of how
much it costs.

1. **A seam the cleanup pass never sees.** The pass runs `LAG` sentences behind
   and needs `MIN_BATCH` of them, so a tail of up to `LAG + MIN_BATCH`
   sentences is never offered to it while the speaker is still talking. A
   variant of this was the bug report that produced recording 3, and it was not
   the tail: batches abutted, so a seam landing *on a batch boundary* was never
   offered to any pass at all, at any point in the session. That is D15. On
   recording 2 that tail was six sentences of fourteen, and one of them was cut
   mid-clause. The `copy` and exit flushes now reach it, which is why the
   measured seam count fell without the pass getting any better at joining. It
   is a delay closed at the last moment, not a fix: the speaker reads the third
   tier while they talk, and this tail is the part they are reading.
2. **A seam the cleanup pass sees and declines to join.** Measured: `...and
   seems like the buffer.` followed by `It is still overflowing a little...`
   The speaker said `the buffer is still overflowing`. Both halves read as
   grammatical sentences, so nothing in the *text* says one of them was cut, and
   the pass is given nothing but text. Handed the pair alone, Qwen3-4B-4bit
   still declines. Handed the run of five it sits in, it joins them.
3. **A word the seam destroyed outright.** `and sometimes they just keep it up`
   became `and sometimes they just so some`. No text-only pass recovers that,
   and one that tries is fabricating. The ceiling gets it right, so the audio
   carries it; the buffer is what threw it away.

### What to try, in order

**~~Give the pass more context, not a bigger model.~~** Right, and now with a
number: `MIN_BATCH` from 4 to 8 takes boundary precision on recording 3 from
0.86 to 0.95, separated, at identical WER and recall (§8). It was the largest
quality lever left. `LAG` and `BATCH` are still unswept, and `BATCH` probably
never binds — batches ran at exactly `MIN_BATCH` all session at 4, so what binds
is the minimum against how fast the document grows, not the cap.

**~~Make the pass reach the tail during the session.~~** Tried the way this
entry suggested, an undersized batch once the document had been still for longer
than a settle, and it **measured worse** on both axes. See §8; the short version
is that this entry and the `MIN_BATCH` one below were pulling in opposite
directions, and the measurement says batch size wins. The tail stays uncleaned
during the session and the two flushes stay the answer for it.

**~~Tell the pass where the seams are.~~** Done; this is D16, and D15 is the
other half of it. The risk this entry predicted — teaching the pass to join at
every seam — **did not materialise**: ablated, the mark leaves recall untouched.
It appears to buy precision, by about 0.14 of median, but see D16 for why that
number is not yet worth trusting. The entry was wrong about where the danger
was.

**~~Get enough runs to say anything about precision.~~** Half done.
`scripts/bench.py` and `STT_ABLATE` are the apparatus and §9 is how to use them.
The other half is still missing: a way to spread runs across a gap, since §8 now
records that consecutive runs of one binary agree to the digit while runs an
hour apart do not. Until that exists, read a separated comparison as real and
treat every unseparated boundary number in this file as unsettled.

**~~Find which dedup change costs the boundaries.~~** Premature. Switching all
four off moved WER decisively and left both boundary ranges overlapping (§8), so
there is no established group effect to bisect. Re-ask it once runs can be
spread across a gap.

**Do not reach for a longer buffer.** §8 swept it: 12 s was best on both axes,
and longer buffers were worse on both. The trade is already at its optimum and
the remaining gap is not on that axis.

### What is not worth trying

- **A better cleanup model.** 4B is the floor and the failures above are not
  fluency failures; the model declines to join because the text is genuinely
  ambiguous, which a larger model is also entitled to do.
- **Prosody at the seam.** §8 swept seam length across audio conditions and the
  boundary never moved once.
- **Anything aimed at boundaries the pipeline failed to produce.** It does not
  produce too few, it produces them in the wrong places. Recall is 1.00 on
  recording 2, and recording 1's single miss is a boundary that moved.
