# findings.md — buffer oversize and product-logic review

A fresh-eyes audit of the pipeline, focused on the reported symptom: *the buffer
fills quickly as soon as the user speaks, and performance degrades.*

Nothing here is taken from the prose in `CLAUDE.md` or from the doc comments.
Every claim below is either derived from the constants in the source or measured
on a running pipeline, and the measurement is given. Where the code's own
documentation contradicts what the code does, that is called out — several of
the load-bearing claims in `CLAUDE.md` turn out to be false.

---

## 0. How this was measured

Three synthesized passages through `--simulate`, plus `STT_TRACE` instrumentation
added to the main loop (`tick`, `probe`, `merge` lines; all behind the existing
env var, no cost when unset):

| file | what it is |
|---|---|
| `long.wav` | 53 s, 10 sentences, ~1 s pauses — ordinary dictation |
| `fast.wav` | 45 s, 10 sentences, 350 ms pauses — nothing endpoints |
| `tv.wav` | `long.wav` with a background talker at −31 dBFS |
| `mid.wav` | a 1.6 s pause dropped *mid-sentence* — the §3 regression case |

A faithful baseline binary was reconstructed in an isolated copy of the tree so
that before/after is a controlled comparison rather than a memory of one.

**Headline result**, `long.wav`, same audio, same model, transcript unchanged:

| | before | after |
|---|---|---|
| trim probes refused | 18 | **7** |
| inference thrown away on them | 8.45 s — **24 % of everything the GPU did** | **2.48 s (8 %)** |
| buffer, max / median | 27.2 s / 20.2 s | **17.4 s / 13.1 s** |
| commit latency, max / median | 2.62 s / 1.88 s | **2.09 s / 1.39 s** |
| time past D6's 2 s target | **43 %** | **1 %** |
| hypotheses drawn | 56 | **75** |
| duty cycle | 67 % | 60 % |
| transcript | ground truth | ground truth, byte-identical |

On `fast.wav`, the harder passage (350 ms pauses, nothing endpoints): past-target
time 29 % → **3 %**, buffer max 27.7 s → **17.0 s**, median 17.0 s → **11.6 s**,
wasted inference 24 % → **6 %**, transcript unchanged.

---

## 1. `unfiled()` asks the wrong question — the root cause

**Severity: high. Fixed.**

The trim proposes a cut, decodes the audio ahead of it, and asks the document
"is this text already filed?" via `Transcript::unfiled`. That routes to
`overlap`, which finds *the longest **suffix** of `filed` that is a prefix of the
decode*.

A head decode is a **prefix** of the buffer. It lines up with the *front* of
what has been filed. A suffix search cannot see it.

Consequence: `unfiled` returned zero **only when the head decoded to exactly
everything filed**. Demonstrated directly:

```
filed words: 19        (two sentences in the document)
unfiled(head covering ONLY sentence 1) = 9     ← a verbatim filed sentence, "all new"
unfiled(head covering BOTH)            = 0
```

So every cut earlier than the exact filed frontier was refused — at a forward
pass each — and past `FORCED_TRIM_MULTIPLE` the head was then filed a *second*
time. Observed end to end on `tv.wav --trim-after-s 12`, baseline binary:

```
The deployment finished at noon, and everything looked stable.
I think the latency numbers are worth a second look, because the spike around three ...
The deployment finished at noon, and everything looked stable.      ← duplicate
I think the latency numbers are worth a second look, because the spike around 3 ...  ← duplicate
```

Twelve lines for a ten-sentence passage.

**The fix is not simply "match the prefix too."** Matching a prefix alone would
authorise the cut `CLAUDE.md` §3 exists to prevent: a head decoding to `It tries
to solve the problem.` against a filed `It tries to solve the problem of
coordination.` has every word in the document and is exactly the cut that
strands half a sentence.

The real rule is two conditions, now in `Transcript::spent_by`:

1. every word ahead of the cut is filed, **front-aligned**; and
2. the match ends **where a filed sentence ended** — tracked with
   `filed_ends`/`filed_base`.

Plus `Transcript::forget_filed_prefix`, because `filed` is documented as "words
filed from the audio **still in the buffer**" and the trim was cutting the audio
away while leaving the words behind — which is what put the two out of step in
the first place.

Trim decisions are now legible in the trace, and cut whole sentences at a time:

```
trim  cut at 3.12s, spends 9 filed word(s)
trim  cut at 8.56s, spends 26 filed word(s)
```

---

## 2. `KEEP_SENTENCES`, not `--trim-after-s`, is what sets the buffer

**Severity: high. Fixed (default changed 2 → 1).**

`--trim-after-s` is documented as bounding "how far behind the transcript can
fall". Swept on `long.wav`, everything else default:

| `--trim-after-s` | buffer max / median | past 2 s target | forced trims | transcript |
|---|---|---|---|---|
| 6 | 13.1 s / 8.7 s | 0 % | **7 of 7** | **damaged** |
| 15 | 24.8 s / 17.5 s | 23 % | 0 | clean |
| 20 | 23.3 s / 17.2 s | 23 % | 0 | clean |
| 30 *(default)* | 25.1 s / 17.1 s | 18 % | 0 | clean |
| 120 | 53.5 s / 15.5 s | 44 % | 0 | clean |

**Across the whole range where the safety test still functions — 15 to 30 — the
buffer does not move.** 17.5 / 17.2 / 17.1 s. The flag does something only when
it is low enough that *every* trim is forced past `FORCED_TRIM_MULTIPLE`, which
bypasses the endorsement test entirely and produced the documented damage on
demand:

```
The reviewers rejected the first draft.
Because it did not explain the trade-offs clearly enough, let us write that section again ...
```

(ground truth: `The reviewers rejected the first draft because it did not explain
the trade-offs clearly enough.` / `Let us write that section again ...`)

The reason is finding 1: the trim may only cut where a filed sentence ended, so
the buffer can never be smaller than what commitment is still holding back. The
actual control is `KEEP_SENTENCES`, a private constant with no CLI exposure.
Same audio, one value changed:

| | KEEP = 2 | KEEP = 1 |
|---|---|---|
| `long.wav` transcript | ground truth | **byte-identical** |
| `fast.wav` transcript | ground truth | **byte-identical** |
| `mid.wav` (the §3 case) | sentence kept whole | **kept whole**; differs only by a comma before three `and`s |
| buffer max / median | 25.1 s / 17.2 s | **17.4 s / 13.1 s** |
| commit latency max / median | 2.57 s / 1.73 s | **2.09 s / 1.39 s** |
| past D6's 2 s target | 17 % | **1 %** |

**The stated reason for 2 does not survive reading the code.** It was that "the
last sentence in the window is usually the one still being spoken, so keeping 2
delivers the one *complete* sentence of right-context". But `settled_prefix`
already handles that separately — it counts `KEEP_SENTENCES + 1` whenever the
window does not end on terminal punctuation. So 2 was buying a *second* complete
sentence of right-context beyond what the measurement vouches for, and the
measurement it cites (60 re-checks of a sentence that already had **another
sentence** after it, 4 revisions, all `3` ↔ `three`) is evidence for 1.

Changed to 1. This is the one change here that is a judgement call on measured
evidence rather than the repair of an outright defect; the residual risk is
recorded at the constant.

---

## 3. `silence_run` is not a run of silence, so the trim cut mid-word

**Severity: high. Fixed.**

`Vad::silence_run` is documented as "consecutive silent frames" and consumed by
`scan` to place cut points. It is not that. It is only reset by `open_frames`
(9 frames / 150 ms) of *sustained* voicing — deliberately, because that is the
keystroke defence — so **the span it measures can contain speech**, and the
midpoint derived from it can land inside a word.

Measured on synthesized speech, comparing each recorded cut candidate against the
frame's actual RMS:

| passage | candidates whose midpoint is in loud audio |
|---|---|
| `p1.wav` (15 s) | **5 of 8** |
| `long.wav` (53 s) | **12 of 22** |

Including `dip @18.856s claimed=208 ms true_silence=0 ms` — above `DIP_FRAMES`,
so an *eligible* cut point, squarely inside a word. The claimed length is also
inflated by the voiced frames it spans, so a non-gap can win longest-wins
selection.

Two features in direct conflict, and nothing in the code or the docs noticed the
coupling: endpointing wants "has the speaker stopped, ignoring clicks", the trim
wants "is there really nothing here".

Fixed by giving them separate counters — `Vad::quiet_run`, reset by any voiced
frame at all. After: **0 of 7** and **0 of 20**. `silence_run` is untouched, so
the keystroke defence and its test are unaffected.

A pure level test (`rms < floor`, ignoring the detector) was tried first and is
*wrong*: the existing `Room` fixture catches it, because a real room idles above
−40 dBFS and a level-only rule yields no cut points at all. Recorded at the
constant so it is not tried again.

---

## 4. A refused trim probe costs a forward pass the tick does not know about

**Severity: medium. Fixed.**

When `cut_point` proposes a cut, the loop always spends `transcribe(&window[..cut])`
first. On refusal there is deliberately no `else` — control falls through to the
ordinary tick, which spends `transcribe(&window)` as well, because `last_tick`
was never reset. Meanwhile:

```rust
tick = interval.max(infer + infer / 4);   // infer = the full-window pass only
```

Measured on `long.wav`: a refused probe at a 23.7 s buffer cost 676 ms, the tick
pass immediately after cost 749 ms, `tick` was 936 ms — **152 % duty on that
interval**, on the code path whose comment promises the stretch keeps it under
100 %. It never showed up as dropped audio because `CAPTURE_BACKLOG` absorbs
isolated bursts; it is a latent spiral that arms itself precisely when the buffer
is already long (at 60 s the probe and the pass are ~1.8 s each against a 2.25 s
tick).

Fixed by charging the tick — but **only the excess** the interval's own idle
cannot cover. Charging the whole probe was tried and measured worse on the
quantity D6 cares about: identical transcript, duty 61 % → 59 %, but peak commit
latency 2.19 s → **3.49 s** and past-target 11 % → 18 %. The excess-only rule
holds each interval at exactly 100 % and binds on 9 ticks out of 63.

---

## 5. The refusal memo lapsed on any filing, and did not cover new gaps

**Severity: medium. Fixed.**

Two separate leaks in `Dip::refused`:

**(a) Wrong clock.** It was keyed on `Transcript::revision`, which moves on *any*
filing. A cut that would strand 50 words cannot become safe until 50 more words
are filed — but a 9-word sentence landing lapsed the memo and bought another
forward pass on a question whose answer had not changed. Measured on `long.wav`:
five offsets probed twice (11.68 s, 18.00 s, 18.16 s, 18.74 s, 21.89 s), **2.6 s
of inference**. Now keyed on a monotone `Transcript::filed_words` with the
threshold `filed + stranded`, which is a sound lower bound.

**(b) A refusal is about a region, not about the gaps that existed at the time.**
`refuse_from` marked existing dips; every merge then appended a *fresh*
sentence-boundary gap near the buffer end with no memo. That gap is the longest
and latest candidate, and sits inside the two-or-three sentences commitment
always holds back — so it was proposed, probed and refused **every single
cycle**. It accounted for **7 of 9 refusals**. New gaps now inherit the strictest
threshold already recorded.

---

## 6. There is no actual bound on buffer length

**Severity: high. Fixed.**

`DESPERATE_TRIM_MULTIPLE` ("any recorded silence") and `FORCED_TRIM_MULTIPLE`
("stop asking whether the text is filed") are presented as backstops. Both are
*filters over `dips`* — and a filter over an empty list is still empty. A speaker
who leaves no gap the VAD calls quiet has **no bound on the buffer whatsoever**.
The "63.6 s buffer" `CLAUDE.md` §3 records as fixed was not fixed, only made
rarer.

Reproduced, and it is not exotic — a television in the same room, at −31 dBFS,
below the speaker and above `--rms-floor`:

| `tv.wav`, 53 s | baseline |
|---|---|
| endpoints | **0** — the utterance never ends |
| merges | 0 |
| trims taken | **0** |
| buffer at end of file | **53.4 s, still growing** |
| forward pass | 1334 ms and rising |
| commit latency | 4.7 s |

The transcript was still *correct*, which is what makes this hard to notice: the
session simply gets slower until it stops keeping up.

Fixed with `last_resort()`, a band below the bands, reached past
`FORCED_TRIM_MULTIPLE`: take any gap at all with refusals ignored, and failing
that the **quietest frame in the buffer**. Verified firing at
`--trim-after-s 12`: buffer pinned at 24.6 s against a 24 s bound, 10 correct
sentences.

---

## 7. The only advice the program gives does not work

**Severity: medium (product). Fixed.**

The falling-behind notice said *"pause for a moment and it will catch up"*, on
the reasoning that a pause endpoints the utterance, which files it and starts the
next one from a short buffer. **The first half is true and the second is not.**
An endpoint does not file — it creates a `Pending` and *holds the audio* for
`--continue-ms`, and if the speaker resumes inside that window §5 merges the
whole thing back. The buffer is restored, not reset.

So the advice was correct only for a pause longer than the settle — 15 s by
default, longer once `Settle` has adapted — which is not what "for a moment"
means. Measured on `long.wav`: **9 merges, and the buffer was never once reset by
a pause.**

The program already has the thing that does what the notice promised: `keep`
ends the settle without a side effect, so the audio is dropped rather than held.
The notice now says so, and it is verified:

```
19.539  notice   text is settling 2.2s behind — 17s buffer, 531 ms per pass;
                 say "Luna, keep" to file what you have said and start fresh
23.692  notice   keep — filed 3 sentence(s)
buffer:  17.22s  →  0.32s
```

Worth noting that `CLAUDE.md` §9 lists `keep` as the weakest verb and a candidate
for removal for lack of use. It is the answer to the pipeline's own commonest
warning.

---

## 8. The forced backstop re-filed text it already had

**Severity: low. Fixed.**

Past `FORCED_TRIM_MULTIPLE`, if every word ahead of the cut was already filed but
the cut did not land on a sentence boundary, the old code still filed the head
decode — putting text the document already held into it a second time. Half a
sentence is stranded either way, but a duplicate is worse than a fragment. It now
cuts and files nothing, traced as
`trim-forced  cut at Xs mid-sentence; text already filed, nothing added`.

---

## 9. `gap_samples` / `MAX_GAP_SAMPLES` is inert for ordinary pauses

**Severity: none (documentation). Not changed.**

`CLAUDE.md` §5 presents `gap_samples` as the fix for "the merge must carry the
real pause", and §3 states the merge seam "wins that contest by construction —
the reconstructed pause is capped at `MAX_GAP_SAMPLES` (1.2 s), longer than
`--endpoint-ms` permits any within-utterance gap to be."

Neither holds at ordinary pause lengths. The arithmetic is

```
gap_samples ≈ pause − endpoint_ms (600 ms) − PREROLL (300 ms)
```

so a 1 s pause contributes ~100 ms and anything under ~900 ms contributes
nothing. Measured on `long.wav`: **`gap=0.00s` on 8 of 9 merges** — the seam dip
is guarded by `if gap > 0`, so most merges contribute no seam candidate at all.
The model sees a seam of ~892 ms regardless, which is the ~0.9 s §5 describes as
the *bug* the fix was for.

Left alone deliberately. §5's own later seam-length sweep (five conditions, nine
lengths, boundary never moved) says the seam length has no effect on the
transcript, and the location is already covered by the pre-endpoint gap. Raising
the seam's rank would bias longest-wins further toward *late* cuts, which
finding 5(b) shows is the expensive direction. The claims in the docs are simply
wrong and should be corrected.

---

## 10. A failed startup poisons the recovery flag

**Severity: medium. Fixed.**

`CLAUDE.md` §6 records this being fixed: `Store::new` writes its session file up
front, so a run that fails afterwards leaves an empty `session-<now>.txt`, and
because `newest_session()` goes by mtime that empty file *is* the newest — so
the next bare `--resume` adopts it, recovers nothing, and without `--persist`
deletes it, putting the real transcript permanently out of reach of the flag
meant to find it.

**The fix only covered a failing `--resume`.** It reordered the resume *load*
before store creation, but left store creation ahead of every other fallible
startup step. Observed during this work — two runs with an unreadable input
path:

```
-rw-r--r--  614  session-20260816-230151.txt     ← the real orphaned transcript
-rw-r--r--    0  session-20260821-044352.txt     ← newest, would be adopted
-rw-r--r--    0  session-20260821-044353.txt     ← newest, would be adopted
```

The store now opens after the sidecar and the capture device are up, so the file
is created only once the session is genuinely going to happen. Verified: a run
with a bad `--simulate` path leaves the state directory exactly as it found it.

---

## 11. Nothing warned that `--trim-after-s` had been set below its useful range

**Severity: low. Fixed.**

Finding 2 shows that below ~10 the trim stops checking whether the text it
severs has been filed, on most or all cuts. That is not a slower setting, it is a
different and worse mode, and the flag's documentation reads as though the value
were a pure latency/quality dial. A startup warning now names the threshold and
what happens past it. Not clamped — a low value is a legitimate thing to ask for,
and overriding it would be deciding for the speaker.

---

## 12. Hypotheses tested and rejected

Recorded because "we measured this and it wasn't the problem" is worth as much as
a fix, and because two of these looked obviously right.

- **The tie-break is selecting late cuts.** Every genuine sentence boundary
  saturates at 36 frames (`close_frames − 1`), so "longest gap wins, latest
  breaks the tie" degenerates into latest-wins — and every refused probe was the
  latest eligible dip, right at the `MIN_SEGMENT_SAMPLES` margin. Reversing the
  tie-break to prefer the earliest changed **nothing**: 9 refusals either way,
  buffer max 22.8 → 22.7 s. Reverted. The real cause was 5(b).
- **The eager trim band is inert because no gap reaches 480 ms.** Derived from
  the constants and wrong: a pause that endpoints leaves a 36-frame (576 ms)
  gap recorded just before the endpoint, which is sentence-grade and travels back
  through the merge with `Pending.dips`. The band works.
- **The buffer floor is set by commitment regardless of `--trim-after-s`.**
  Right within 15–30 s (finding 2), wrong outside it — the flag does control the
  buffer at 6 and at 120, at the cost of transcript damage and latency
  respectively.
- **A noisy room breaks endpointing.** Gaussian room noise at −35 dBFS does not:
  `earshot` rejects it and the passage transcribes normally. It takes
  *speech-like* background to break it (finding 6).
- **`quiet_run` should be a pure level test.** Provable invariant, and wrong —
  it yields zero cut points in a room that idles above the floor (finding 3).

---

## 13. Left unfixed

- **`FILED_MEMORY` interacts with `filed_base`.** When the 256-word cap pops
  words off the front, `filed_base` is advanced so boundaries stay correct, but a
  head decode longer than 256 words can no longer be endorsed. Not reachable at
  the measured buffer sizes (~40 words); would be at very long
  `--trim-after-s`.
- **The forced path still clears no `filed` state.** It leaves the deque
  describing audio that has been cut away, so the next front-aligned comparison
  starts in the wrong place until something clears it. Contained today because
  `covered()` also tries the suffix alignment, but that is a fallback, not a fix.
- **`is_initial` never splits after a one-letter word**, so `The answer is A. It
  follows.` is one sentence. Rare in dictation; the guard is worth more than the
  case.
- **Everything here is synthesized speech.** It has no hesitation, no
  disfluency, and clean prosody. It is evidence about *mechanism* — does this
  path fire, does this cut land, does this string split — and not about how a
  human hesitating mid-thought comes out. The `KEEP_SENTENCES` change in
  particular wants real dictation before it is trusted.
