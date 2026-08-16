//! Real-time speech-to-text CLI.
//!
//! Rust owns capture, VAD, windowing, the transcript state machine and
//! rendering. A Python/MLX sidecar owns the forward pass. See CLAUDE.md for
//! the decision record.

mod audio;
mod command;
mod render;
mod repair;
mod sidecar;
mod store;
mod text;
mod transcript;
mod vad;

use anyhow::{Context, Result};
use clap::Parser;
use command::{Command, Segment};
use render::Renderer;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use transcript::Transcript;
use vad::{FRAME, Vad, VadEvent};

#[derive(Parser)]
#[command(
    about = "Local real-time speech-to-text with live provisional text",
    // `--rms-floor -30` is the natural way to write a dBFS level, and without
    // this clap reads the value as another flag and refuses to start.
    allow_negative_numbers = true
)]
struct Args {
    /// MLX model repo.
    ///
    /// 1.7B is the default: measured on this machine it runs at roughly half
    /// the speed of 0.6B, not the ~7× the published figures implied, which
    /// leaves it comfortably able to sustain the sliding window. Pass
    /// `mlx-community/Qwen3-ASR-0.6B-8bit` for more headroom on a slower host.
    #[arg(long, default_value = "mlx-community/Qwen3-ASR-1.7B-8bit")]
    model: String,

    /// What to call the assistant when giving it an instruction.
    ///
    /// "<name>, commit" keeps everything settled so far; "<name>, reject"
    /// drops the most recent sentence, including one already committed. The
    /// comma is optional — it is punctuation the model chose, and the parser
    /// never sees it.
    #[arg(long, default_value = "Luna")]
    assistant: String,

    /// Force a language (e.g. "en"). Omit to auto-detect.
    #[arg(long)]
    language: Option<String>,

    /// Input device name substring. Omit for the system default.
    #[arg(long)]
    device: Option<String>,

    /// Feed a 16 kHz mono WAV through the pipeline instead of the microphone.
    #[arg(long, conflicts_with = "device")]
    simulate: Option<PathBuf>,

    /// Shortest gap between re-runs of the sliding window, in milliseconds.
    ///
    /// A floor, not a fixed period: when inference on a long buffer outgrows
    /// it, the gap stretches to fit rather than letting the duty cycle pass
    /// 100%. Lowering this speeds up commitment on the short buffers where the
    /// model has headroom, and does nothing on the long ones where it does not.
    #[arg(long, default_value_t = 500)]
    interval_ms: u64,

    /// LocalAgreement depth: how many consecutive hypotheses must contain a
    /// word before it stops being provisional. Higher is calmer but slower.
    #[arg(long, default_value_t = 3)]
    agreement: usize,

    /// Silence needed to end an utterance, in milliseconds.
    #[arg(long, default_value_t = 600)]
    endpoint_ms: u32,

    /// Sound must last this long, in milliseconds, before it counts as speech.
    ///
    /// This is the defence against a keyboard, and it is a *duration* test
    /// because level is not usable here: a keystroke is louder than the room by
    /// construction, so any floor that rejects one also rejects a quiet voice.
    /// What a keystroke is not is sustained — measured, one produces 3 to 6
    /// voiced frames where the shortest syllable produces a dozen.
    ///
    /// Raise it if typing, a mouse, or a creaking chair still gets transcribed;
    /// lower it if a short first word is being missed. The onset latency it adds
    /// is not lost audio — a 300 ms pre-roll is retained, so the speech that
    /// opened the utterance is still in the buffer — but values past that
    /// pre-roll would start clipping first words.
    ///
    /// It gates *isolated* transients, which is the case that produces stray
    /// text. Sustained fast typing reads as continuous energy to the detector
    /// and is not gated by any duration threshold; `--rms-floor` is the knob
    /// for that, and it works only if your voice is louder than your keyboard.
    #[arg(long, default_value_t = vad::DEFAULT_OPEN_MS)]
    open_ms: u32,

    /// Trim the audio buffer once it passes this many seconds.
    ///
    /// **Never cuts mid-word.** It cuts at the longest silence the speaker
    /// already left — between sentences, around a breath — and if they left no
    /// such gap it does not cut at all. An utterance is therefore never
    /// truncated; what this bounds is how far behind the transcript can fall.
    ///
    /// It has to bound something, because inference and commit latency both
    /// scale with the buffer: measured, a 20s buffer costs ~620ms per pass and
    /// ~2.2s to stabilise a word, and extrapolating the same curve a 600s
    /// buffer costs ~20s per pass and over a minute to stabilise. That is still
    /// *correct*, just uselessly far behind someone reading along as they
    /// speak.
    #[arg(long, default_value_t = 30)]
    trim_after_s: u64,

    /// Silence, in milliseconds, before an utterance settles.
    ///
    /// Until it elapses the sentence stays dim and its audio is retained: if
    /// the speaker resumes, the audio is merged and the whole thing re-decoded
    /// as one utterance, so the model re-punctuates across the seam rather than
    /// us splicing two independently decoded fragments. 0 settles immediately.
    ///
    /// This is one number doing two jobs, and they pull in opposite directions:
    /// it is both how long text stays dim (shorter is more responsive) and how
    /// long a thinking pause can run before it becomes two sentences instead of
    /// one fused thought (longer is more forgiving). 6s splits the difference.
    ///
    /// Raising it is safe for accuracy — the merged audio carries the real
    /// pause, so the model re-separates utterances that were never one thought.
    /// What it costs is recompute and on-screen movement.
    #[arg(long, default_value_t = 6_000)]
    continue_ms: u64,

    /// Keep the session file on exit instead of deleting it.
    ///
    /// The transcript is written to disk as you speak either way — that is what
    /// makes `clear` and `reject` recoverable, and what stops a panic or a
    /// second Ctrl-C from taking the session with it. Without this the file is
    /// removed on a **clean** exit, once stdout has the transcript. An unclean
    /// exit leaves it behind regardless, since nothing is running to delete it.
    #[arg(long)]
    persist: bool,

    /// Pick up a previous session, appending to its transcript.
    ///
    /// Bare, it recovers the most recently written session file — which is what
    /// a crash, a panic or a closed window leaves behind. Given a path, it
    /// resumes that file instead.
    ///
    /// Recovered text comes back **kept**: the file records text rather than
    /// the workflow state that produced it, and choosing to resume a transcript
    /// is itself an approval of it.
    ///
    /// Fails rather than starting empty when there is nothing to resume. The
    /// alternative is dictating into a fresh session believing it is the old
    /// one, and only finding out at the end.
    #[arg(long, value_name = "PATH", num_args = 0..=1)]
    resume: Option<Option<PathBuf>>,

    /// Silence floor in dBFS: frames quieter than this are never speech,
    /// whatever the neural detector says.
    ///
    /// Raise it if a quiet room produces stray "Okay." or "Oh." lines. Those
    /// are the model being handed non-speech, not the model inventing things
    /// unprompted: the floor rejects audio that is *quiet*, not audio that is
    /// *non-speech*, so a room idling above it clears the gate on every frame.
    /// Measured here, noise at −38 dBFS is enough to produce "Oh." on its own.
    /// Set it above your room and below your voice.
    #[arg(long, default_value_t = vad::DEFAULT_FLOOR_DBFS)]
    rms_floor: f32,

    /// Suppress the "refresh slowed" notice entirely.
    #[arg(long)]
    quiet: bool,

    /// Python interpreter for the sidecar.
    #[arg(long, default_value = ".venv/bin/python")]
    python: PathBuf,

    /// Sidecar script.
    #[arg(long, default_value = "sidecar/asr_sidecar.py")]
    script: PathBuf,
}

/// Silence retained ahead of speech onset so the first phoneme isn't clipped.
const PREROLL_SAMPLES: usize = (audio::TARGET_RATE as usize * 300) / 1000;

/// Ceiling on silence spliced back into a merged utterance.
///
/// Anything past about a second already reads as a sentence boundary, so
/// reproducing a 30s pause faithfully would only cost encoder time. Capping
/// keeps the merged buffer bounded while still showing the model a pause it
/// cannot mistake for a breath.
const MAX_GAP_SAMPLES: usize = (audio::TARGET_RATE as usize * 1200) / 1000;

/// Silent frames that mark a gap worth cutting at: ~190ms.
///
/// Well under any sane `--endpoint-ms`, because the whole point is to find the
/// pauses that are *not* endpoints — the ones inside a passage the speaker is
/// still in the middle of.
const DIP_FRAMES: usize = 12;

/// Shortest silence recorded as a *possible* cut point: ~64ms.
///
/// Deliberately below [`DIP_FRAMES`], so these are not gaps anyone would choose:
/// 64ms of silence can be the closure of a stop consonant rather than a word
/// boundary, and cutting there risks clipping one. They are recorded because the
/// alternative turned out to be worse than a clipped consonant.
///
/// D9 says an utterance is never truncated and that a buffer with no gap in it
/// simply keeps growing. Reported live, that has no bound: a speaker who never
/// left 190ms drove the buffer to **63.6s**, where a single forward pass took
/// ~67s and stretched the tick to ~84s. A bad cut is a bad sentence; an
/// unbounded buffer is a session that stops responding.
const MIN_DIP_FRAMES: usize = 4;

/// How far past `--trim-after-s` the buffer may go before the trim stops holding
/// out for a comfortable gap.
///
/// Below this only [`DIP_FRAMES`] gaps are eligible, which is D9's quality bar
/// unchanged. Past it every recorded silence is fair game — longest still
/// preferred, so this only ever *adds* candidates the trim would otherwise have
/// had none of. By then the speaker has demonstrably not left a good gap, and
/// there is no bound on how long waiting for one would take.
const DESPERATE_TRIM_MULTIPLE: usize = 2;

/// A gap the speaker left inside an utterance: somewhere the buffer may be cut
/// without slicing a word.
///
/// The length is kept because gaps are not equally good places to cut. A ~200ms
/// gap is usually a comma; a ~500ms one is usually the end of a sentence, and
/// cutting there splits the transcript where it was going to split anyway.
/// Measured on a 42s passage, taking the most recent gap rather than the
/// longest one cut mid-clause and left `If I never leave such a gap,` standing
/// as its own entry, comma and all.
#[derive(Clone, Copy)]
struct Dip {
    at: usize,
    frames: usize,
}

/// Audio kept on each side of a trim.
///
/// The head needs enough to be worth decoding on its own; the tail needs enough
/// that the next window is not starting from a sliver. Both at 2s, which the
/// measured curve puts at ~120ms of inference — cheap on either side.
const MIN_SEGMENT_SAMPLES: usize = audio::TARGET_RATE as usize * 2;

/// Minimum gap between repeats of the same rate-limited notice.
const NOTICE_EVERY: Duration = Duration::from_secs(10);

/// Longest the live view may go without a repaint, so the settle countdown
/// advances and notices expire on time.
const REPAINT_EVERY: Duration = Duration::from_millis(200);

/// Set from a signal handler; the loop exits at the next iteration so the
/// transcript is still written and the terminal still restored.
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// An utterance that has ended but whose audio is kept, in case the pause that
/// ended it turns out to have been a hesitation.
struct Pending {
    audio: Vec<f32>,
    /// Gaps already found in `audio`, at offsets into it.
    ///
    /// Held with the audio rather than discarded at the endpoint, because a
    /// continuation puts that audio back at the *front* of the merged buffer
    /// with its pauses exactly where they were. Dropping them left a merged
    /// buffer with no cut candidates except its own seam — one per merge — so a
    /// session that merges continuously, which §5 means every ordinary pause
    /// does, could hold a dozen sentences and offer the trim a handful of places
    /// to cut instead of every pause in the passage.
    ///
    /// That is not a latency bug. The trim is the only thing that files text
    /// into the document mid-passage, so starving it leaves `Luna, copy` and the
    /// session file empty of everything the speaker has said.
    dips: Vec<Dip>,
    /// Dim in the live view, and deliberately not in the document yet: a
    /// continuation re-decodes it and can rewrite every word.
    words: Vec<String>,
    at: Instant,
}

extern "C" fn on_signal(_: libc::c_int) {
    // A second Ctrl-C means the first one is not getting through — almost
    // certainly because the loop is blocked in a long forward pass. Rather than
    // leave the user stuck, restore the terminal here and leave immediately.
    // Both calls are async-signal-safe, which `print!` and `Renderer` are not.
    if INTERRUPTED.swap(true, Ordering::SeqCst) {
        const RESTORE: &[u8] = b"\x1b[?1049l\x1b[?25h";
        unsafe {
            libc::write(1, RESTORE.as_ptr().cast(), RESTORE.len());
            libc::_exit(130);
        }
    }
}

fn install_signal_handlers() {
    // Ctrl-C must not kill the process outright: the alternate screen would be
    // left entered, handing back a shell the user cannot see themselves typing
    // into, and the transcript would go with it.
    unsafe {
        libc::signal(libc::SIGINT, on_signal as libc::sighandler_t);
        libc::signal(libc::SIGTERM, on_signal as libc::sighandler_t);
    }
}

/// File a finished utterance into the document, one entry per sentence.
///
/// Splitting matters beyond line breaks: `reject` removes one document entry,
/// so a re-decoded continuation the model judged to be two separate thoughts
/// has to arrive as two separately rejectable sentences.
fn file(script: &mut Transcript, words: &[String]) {
    let text = words.join(" ");
    for part in text::split_sentences(&text) {
        script.push_sentence(part.split_whitespace().map(str::to_string).collect());
    }
}

/// Run a finalized utterance: dictation is filed, instructions are executed, in
/// the order they were spoken.
///
/// Every branch reports through `notice`, including the ones that did nothing.
/// A spoken command has no other acknowledgement — the speaker cannot see a
/// button depress — so silence is indistinguishable from not being heard, and
/// they would simply say it again. For `reject` that means deleting a second
/// sentence.
fn apply(
    script: &mut Transcript,
    words: &[String],
    wake: &str,
    screen: &mut Renderer,
    debounce: &mut command::Debounce,
) {
    for segment in command::split(words, wake) {
        // A text-removing command repeated before the speaker could have seen
        // the first one land is the same intention said twice. Saying so is
        // essential rather than polite: silence here is indistinguishable from
        // not being heard, which is what provokes the repeat in the first place.
        if let Segment::Run(cmd) = segment
            && !debounce.allows(cmd, Instant::now())
        {
            screen.notice(&format!(
                "ignored a repeated {} — pause a moment and say it again to do it twice",
                match cmd {
                    Command::Reject => "reject",
                    _ => "clear",
                }
            ));
            continue;
        }

        match segment {
            Segment::Text(text) => file(script, &text),

            Segment::Run(Command::Commit) => {
                let n = script.commit_all();
                screen.notice(&match n {
                    0 => "commit — nothing new to keep".to_string(),
                    1 => "commit — kept 1 sentence".to_string(),
                    n => format!("commit — kept {n} sentences"),
                });
            }

            Segment::Run(Command::Reject) => {
                let msg = match script.reject_last() {
                    Some(dropped) => format!("reject — dropped \u{201c}{}\u{201d}", dropped.text()),
                    None => "reject — nothing left to drop".to_string(),
                };
                screen.notice(&msg);
            }

            Segment::Run(Command::Clear) => {
                let n = script.clear();
                screen.notice(&match n {
                    0 => "clear — already empty".to_string(),
                    n => format!("clear — removed {n} sentence(s); say \"{wake}, undo\" to get them back"),
                });
            }

            Segment::Run(Command::Rollback) => {
                let msg = match script.rollback() {
                    Some(what) => format!("undo — restored {what}"),
                    None => "undo — nothing to take back".to_string(),
                };
                screen.notice(&msg);
            }

            Segment::Run(Command::Copy) => screen.notice(&copy(script)),
        }
    }
}

/// `Luna, copy` — put the transcript on the clipboard, and keep what it took.
///
/// **Copying is itself an act of approval**, which is the same argument
/// `--resume` already makes about a recovered file: you looked at the text and
/// chose to do something with it. So this no longer refuses to copy a document
/// nobody said "commit" over — it copies the whole document and marks it kept,
/// which is the reading that matches what the speaker just did.
///
/// The narrow part worth keeping is the other boundary: this stops at the
/// *document*. An utterance still inside its grace window is not in there yet,
/// so text the **pipeline** may still rewrite is never copied and never
/// committed. Approval is the speaker's to give, and there is nothing settled
/// to give it to yet.
///
/// The commit happens only if the clipboard write succeeded, since it is that
/// write the approval attaches to.
fn copy(script: &mut Transcript) -> String {
    let text: Vec<String> = script
        .document()
        .iter()
        .map(transcript::Sentence::text)
        .collect();

    // Worth saying rather than doing silently: the alternative is overwriting
    // whatever the speaker had on the clipboard with an empty string.
    if text.is_empty() {
        return "copy — nothing to copy yet".to_string();
    }

    let n = text.len();
    match store::copy_to_clipboard(&(text.join("\n") + "\n")) {
        Ok(()) => match script.commit_all() {
            0 => format!("copy — {n} sentence(s) on the clipboard"),
            kept => format!("copy — {n} sentence(s) on the clipboard, {kept} newly kept"),
        },
        Err(e) => format!("copy failed: {e}"),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    install_signal_handlers();

    // Storage is settled before the sidecar loads, so a bad `--resume` fails in
    // milliseconds instead of after three seconds of model warmup.
    //
    // The lookup has to happen before anything creates *this* session's file,
    // or it finds our own and reports it as the previous one.
    let previous = store::newest_session();

    // The bool distinguishes a file we found in our own state directory from a
    // path the user named. Only the former is ours to write to or delete, and
    // continuing it in place is what stops every crash from leaving a stale
    // duplicate behind.

    let resume: Option<(PathBuf, bool)> = match &args.resume {
        None => None,
        Some(Some(path)) => Some((path.clone(), false)),
        Some(None) => Some((
            previous.clone().context(
                "nothing to resume — no previous session found. \
                 Pass a path to resume a specific transcript.",
            )?,
            true,
        )),
    };

    // Non-fatal: an unwritable state directory is a reason to dictate without
    // a safety net, not a reason to refuse to dictate.
    let opened = match &resume {
        Some((path, true)) => Ok(store::Store::adopt(path.clone(), args.persist)),
        _ => store::Store::new(args.persist),
    };
    let mut store = match opened {
        Ok(store) => {
            eprintln!("autosaving to {}", store.path().display());
            Some(store)
        }
        Err(e) => {
            eprintln!("warning: no autosave ({e:#}) — the transcript is memory-only");
            None
        }
    };

    let mut recovered: Vec<transcript::Sentence> = Vec::new();
    if let Some((path, _)) = &resume {
        // Fatal on failure: the speaker asked for this text back, and starting
        // without it would be discovered far too late.
        let loaded = store::load(path).context("resuming a previous session")?;
        eprintln!("resumed {} sentence(s) from {}", loaded.len(), path.display());
        recovered = loaded;
    } else if let Some(path) = &previous {
        // Recovery is worthless if nobody knows it happened. This is the only
        // place a crashed session announces that it left something behind, so
        // it carries the size too — that is what decides whether resuming is
        // worth it.
        let n = store::load(path).map_or(0, |s| s.len());
        eprintln!(
            "note: a previous session is on disk ({n} sentence(s)) \
             — pass --resume to continue it: {}",
            path.display()
        );
    }


    // The command vocabulary, derived from `--assistant` so a retargeted wake
    // word is announced too. Fixed for the session and never touched again —
    // the per-window protocol carries audio and nothing else.
    let mut asr = sidecar::Sidecar::spawn(
        &args.python,
        &args.script,
        &args.model,
        args.language.as_deref(),
        &command::hint(&args.assistant),
    )
    .context("starting inference sidecar")?;

    let capture = match &args.simulate {
        Some(path) => audio::simulate(path)?,
        None => audio::start(args.device.as_deref())?,
    };

    let mut vad = Vad::new(args.open_ms, args.endpoint_ms, args.rms_floor);
    let mut script = Transcript::new(args.agreement);
    script.restore(recovered);

    // Audio for the current utterance, at 16 kHz.
    let mut window: Vec<f32> = Vec::new();
    let mut vad_cursor = 0usize;
    let mut has_speech = false;
    let mut last_tick = Instant::now();

    // Offsets into `window` where the speaker left a short gap. These are the
    // only places the buffer may be cut.
    let mut dips: Vec<Dip> = Vec::new();

    // Far enough back that the first notice of each kind is never held. Checked
    // because `Instant` counts from boot and the plain subtraction panics on
    // underflow — reachable, if barely, on a machine that just started.
    let long_ago = Instant::now()
        .checked_sub(NOTICE_EVERY)
        .unwrap_or_else(Instant::now);
    let mut last_slowed = long_ago;
    let mut last_failed = long_ago;

    let mut debounce = command::Debounce::default();
    let mut pending: Option<Pending> = None;
    let continue_ms = Duration::from_millis(args.continue_ms);
    // Silence discarded since the last endpoint, measured in samples so it
    // stays correct regardless of wall-clock pacing.
    let mut gap_samples = 0usize;

    let interval = Duration::from_millis(args.interval_ms);
    // Effective gap between hypotheses. `interval` is the floor; this stretches
    // when inference outgrows it so the duty cycle cannot pass 100%.
    let mut tick = interval;
    let trim_after = args.trim_after_s as usize * audio::TARGET_RATE as usize;


    eprintln!(
        "listening — speak; say \"{name}, commit\" to keep what you said, \
         then reject / clear / copy / undo; Ctrl-C to stop",
        name = args.assistant
    );

    // Last, so every line above reaches the ordinary terminal. Once this
    // exists it owns the screen and nothing may write there behind its back.
    let mut screen = Renderer::new();
    let mut last_paint = long_ago;

    // Set aside rather than returned, so a dead sidecar still ends with the
    // transcript written out instead of taking it down with the process.
    let mut fatal: Option<anyhow::Error> = None;

    loop {
        if INTERRUPTED.load(Ordering::SeqCst) {
            break;
        }

        // Diagnostics from the audio and sidecar threads. They must go through
        // the renderer: anything that writes to the terminal behind its back
        // lands in the middle of the frame.
        for msg in capture.notices.try_iter().chain(asr.notices().try_iter()) {
            screen.notice(&msg);
        }

        // Everything the pipeline is holding that the document does not have
        // yet. Between trims that is the *whole* passage: §5 merges every
        // ordinary pause, so an utterance never settles and `finalize_window`
        // deliberately files nothing. Handing it to the store is what keeps a
        // crash mid-passage from taking minutes of speech with it.
        let in_flight = match &pending {
            Some(p) => p.words.join(" "),
            None => {
                let mut live = script.committed().join(" ");
                for word in script.provisional() {
                    if !live.is_empty() {
                        live.push(' ');
                    }
                    live.push_str(word);
                }
                live
            }
        };

        // A no-op unless the document or the utterance in flight actually
        // moved; the latter is rate-limited inside. Placed before every
        // `continue` below so no mutation can reach the end of an iteration
        // unwritten.
        if let Some(store) = &mut store
            && let Some(err) = store.save(script.document(), script.revision(), &in_flight)
        {
            screen.notice(&err);
        }

        // Repainting from the top means every `continue` below still leaves the
        // screen current, including the pre-roll trim that skips the rest of
        // the iteration.
        if last_paint.elapsed() >= REPAINT_EVERY {
            last_paint = Instant::now();
            let state = match &pending {
                Some(p) => render::State::Settling(
                    continue_ms
                        .saturating_sub(p.at.elapsed())
                        .as_millis()
                        .div_ceil(1000) as u64,
                ),
                None if vad.is_active() => render::State::Speaking,
                None => render::State::Listening,
            };
            screen.draw(&render::Frame {
                document: script.document(),
                settling: pending.as_ref().map_or(&[], |p| &p.words),
                committed: script.committed(),
                provisional: script.provisional(),
                state,
                wake: &args.assistant,
            });
        }

        match capture.pcm.recv_timeout(Duration::from_millis(50)) {
            Ok(chunk) => {
                window.extend_from_slice(&chunk);
                // Then take everything else already queued. Consuming a single
                // chunk per iteration was the mechanism that turned one slow
                // forward pass into a permanent backlog: a pass that runs off
                // the tick — an endpoint, a merge, a trim — buffers tens of
                // chunks, and draining them one per iteration meant the loop
                // was still behind when the next pass started. This costs
                // nothing when the queue is empty, which is the normal case.
                for chunk in capture.pcm.try_iter() {
                    window.extend_from_slice(&chunk);
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }

        // Advance VAD over whole frames only.
        let mut endpointed = false;
        let mut onset = false;
        while vad_cursor + FRAME <= window.len() {
            let ev = vad.push(&window[vad_cursor..vad_cursor + FRAME]);
            vad_cursor += FRAME;
            match ev {
                VadEvent::Onset => {
                    has_speech = true;
                    onset = true;
                }
                VadEvent::Speaking => {
                    has_speech = true;
                    // A gap inside an utterance. Cutting at its midpoint leaves
                    // trailing silence on the head and pre-roll on the tail, so
                    // neither side starts or ends mid-phoneme.
                    //
                    // The run is still growing, so the entry is re-centred and
                    // re-measured on every further silent frame; `silence_run`
                    // resets on the next voiced one, which is what makes the
                    // first frame past the threshold a new gap rather than a
                    // continuation of the last.
                    let run = vad.silence_run();
                    if run >= MIN_DIP_FRAMES {
                        let started = vad_cursor.saturating_sub(run * FRAME);
                        let dip = Dip {
                            at: started + run * FRAME / 2,
                            frames: run,
                        };
                        // Whether this extends the last gap is decided by where
                        // the run *started*, not by it being one frame longer
                        // than last time. A keystroke mid-pause holds the run
                        // steady for a frame or two instead of extending it,
                        // and that must not register as a second gap — an
                        // earlier gap always ended before this one began, so
                        // its midpoint is always behind `started`.
                        match dips.last_mut() {
                            Some(last) if last.at >= started => *last = dip,
                            _ => dips.push(dip),
                        }
                    }
                }
                VadEvent::Endpoint => endpointed = true,
                VadEvent::Idle => {}
            }
        }

        // The grace closed with no continuation, so the utterance was a full
        // stop after all. Only now does it join the document.
        if let Some(p) = &pending
            && p.at.elapsed() >= continue_ms
        {
            let expired = pending.take().expect("just checked");
            file(&mut script, &expired.words);
            last_paint = long_ago;
        }

        // A pause is only a full stop in hindsight. If the speaker resumes soon
        // enough, the sentence that ended was a hesitation — merge the audio
        // and re-decode the whole thing as one utterance, so the model
        // re-punctuates across the seam rather than us splicing two fragments.
        //
        // `take_if`, not `take`: a bare take would remove the sentence before
        // the grace check and drop it on the floor if the check then failed.
        if onset
            && let Some(p) = pending.take_if(|p| p.at.elapsed() < continue_ms)
        {
            // Reconstruct the real pause: trailing silence already in the held
            // audio, plus what the pre-roll trim threw away, plus the pre-roll.
            let gap = gap_samples.min(MAX_GAP_SAMPLES);
            let held = p.audio.len();
            let mut merged = p.audio;
            merged.resize(merged.len() + gap, 0.0);
            let shift = merged.len();
            merged.extend_from_slice(&window);

            // Everything recorded since the endpoint now sits `shift` samples
            // later.
            for dip in &mut dips {
                dip.at += shift;
            }
            if gap > 0 {
                // The seam is the longest silence in the buffer by
                // construction, so it is also the best place to cut.
                dips.push(Dip {
                    at: held + gap / 2,
                    frames: gap / FRAME,
                });
            }
            // The held audio is back at the front of the buffer, so the pauses
            // found in it are still where they were and are still the only
            // places it may be cut. These are what a long merged passage is made
            // of: the seam alone offers one candidate per merge, which is not
            // enough to keep a continuously merging session inside
            // `--trim-after-s`.
            dips.extend(p.dips);
            dips.sort_unstable_by_key(|d| d.at);

            window = merged;
            vad_cursor += shift;
            // Any instant already in the past makes the tick below fire on this
            // iteration, so the merged audio is decoded immediately rather than
            // after another interval of stale text on screen.
            last_tick = long_ago;
            gap_samples = 0;
        }

        // Outside an utterance, keep only a short pre-roll so idle silence
        // doesn't accumulate into the window.
        if !has_speech && window.len() > PREROLL_SAMPLES {
            let drop = window.len() - PREROLL_SAMPLES;
            window.drain(..drop);
            vad_cursor = vad_cursor.saturating_sub(drop);
            dips.clear();
            // Remember how much silence was discarded. A continuation splices
            // it back, so the model judges the boundary on the pause the
            // speaker actually took rather than on the fixed ~0.9s that
            // endpoint trailing-silence plus pre-roll would otherwise leave.
            gap_samples += drop;
            continue;
        }

        if endpointed {
            if has_speech {
                // A failure here costs the final pass over the utterance, so
                // the tail stays uncommitted — but everything the sliding
                // window already agreed on is still finalized below rather
                // than discarded.
                match asr.transcribe(&window) {
                    Ok(sidecar::Reply::Hypothesis(hyp)) => {
                        script.push_hypothesis(&hyp.text);
                    }
                    Ok(sidecar::Reply::Failed(why)) => {
                        screen.notice(&format!("sidecar failed on the final pass: {why}"));
                    }
                    Err(e) => {
                        fatal = Some(e);
                        break;
                    }
                }

                let utterance = script.finalize_window();
                if command::has_command(&utterance, &args.assistant) {
                    // Instructions run now — making the speaker wait out the
                    // settle period to find out whether they were heard would
                    // defeat the point of having them.
                    //
                    // And the audio is dropped rather than retained: a command
                    // replayed into a merged window would fire a second time,
                    // which for `reject` means silently eating another
                    // sentence. Retention is the only thing that could replay
                    // it, so not retaining it is the whole fix.
                    apply(&mut script, &utterance, &args.assistant, &mut screen, &mut debounce);
                } else if utterance.is_empty() || continue_ms.is_zero() {
                    file(&mut script, &utterance);
                } else {
                    pending = Some(Pending {
                        audio: std::mem::take(&mut window),
                        // Taken rather than left to the `dips.clear()` below.
                        // This audio may come back at the front of a merged
                        // buffer, and its pauses have to come back with it.
                        dips: std::mem::take(&mut dips),
                        words: utterance,
                        at: Instant::now(),
                    });
                    gap_samples = 0;
                }
            }
            window.clear();
            vad_cursor = 0;
            dips.clear();
            has_speech = false;
            last_tick = Instant::now();
            // The next utterance starts from a short buffer, so the stretch
            // this one may have earned does not carry over.
            tick = interval;
            last_paint = long_ago;
            continue;
        }

        // The buffer has outgrown the trim threshold. Cut it at a gap the
        // speaker already left, never mid-word — and if they have not left one,
        // do not cut at all. An utterance is never truncated; the cost of
        // refusing to cut is latency, which is recoverable, where the cost of
        // cutting mid-word is a mangled transcript, which is not.
        // Hold out for a comfortable gap until the buffer is far enough past the
        // threshold that waiting for one costs more than taking a short one.
        let dip_floor = if window.len() >= trim_after.saturating_mul(DESPERATE_TRIM_MULTIPLE) {
            MIN_DIP_FRAMES
        } else {
            DIP_FRAMES
        };

        if window.len() >= trim_after
            && let Some(cut) = dips
                .iter()
                .filter(|d| {
                    d.frames >= dip_floor
                        && d.at >= MIN_SEGMENT_SAMPLES
                        && window.len() - d.at >= MIN_SEGMENT_SAMPLES
                })
                // Longest gap wins, latest breaks the tie: the longer a pause,
                // the more likely it is a sentence boundary rather than a
                // comma, and cutting at one splits the transcript where it was
                // going to split anyway.
                .max_by_key(|d| (d.frames, d.at))
                .map(|d| d.at)
        {
            // Decode the head alone. Finalizing against a hypothesis of the
            // *whole* buffer would commit words whose audio is about to stay
            // behind in the tail, and they would then be transcribed twice.
            match asr.transcribe(&window[..cut]) {
                Ok(sidecar::Reply::Hypothesis(hyp)) => {
                    script.push_hypothesis(&hyp.text);
                }
                Ok(sidecar::Reply::Failed(why)) => {
                    screen.notice(&format!("sidecar failed on the trim pass: {why}"));
                }
                Err(e) => {
                    fatal = Some(e);
                    break;
                }
            }

            // Straight into the document, with no settle period: the speaker is
            // still mid-passage, so there is no pause here to reconsider.
            let head = script.finalize_window();
            apply(&mut script, &head, &args.assistant, &mut screen, &mut debounce);

            window.drain(..cut);
            vad_cursor = vad_cursor.saturating_sub(cut);
            dips.retain(|d| d.at > cut);
            for dip in &mut dips {
                dip.at -= cut;
            }
            last_tick = Instant::now();
            tick = interval;
            last_paint = long_ago;
            continue;
        }

        if has_speech && vad.is_active() && last_tick.elapsed() >= tick {
            last_tick = Instant::now();

            // One refused window is not a reason to end the session and throw
            // away the transcript: the next window is a fresh forward pass and
            // usually succeeds. Rate-limited, since a sidecar that is failing
            // persistently would otherwise flood the status line every tick.
            // Not suppressed by --quiet, which covers the duty-cycle notice
            // below — that one is self-correcting, this one is a real fault.
            let hyp = match asr.transcribe(&window) {
                Ok(sidecar::Reply::Hypothesis(hyp)) => hyp,
                Ok(sidecar::Reply::Failed(why)) => {
                    if last_failed.elapsed() >= NOTICE_EVERY {
                        last_failed = Instant::now();
                        screen.notice(&format!("sidecar failed on this window: {why}"));
                    }
                    continue;
                }
                Err(e) => {
                    fatal = Some(e);
                    break;
                }
            };

            // Keep the duty cycle under 100%, which is a hard cliff rather than
            // a gradual slowdown: the loop takes one audio chunk per iteration,
            // so once inference no longer fits inside a tick it consumes a
            // single chunk per forward pass, the capture channel backs up, and
            // the audio callback starts dropping buffers outright.
            //
            // The last inference predicts the next one well: cost grows
            // smoothly with buffer length, and within an utterance the buffer
            // only grows.
            let infer = Duration::from_millis(hyp.infer_ms);
            tick = interval.max(infer + infer / 4);

            // Informational rather than a warning — the system is keeping up,
            // just refreshing less often. Only worth saying once the stretch is
            // big enough to notice.
            let stretched = tick >= interval + interval / 5;
            if stretched && !args.quiet && last_slowed.elapsed() >= NOTICE_EVERY {
                last_slowed = Instant::now();
                screen.notice(&format!(
                    "{:.1}s buffer, inference {} ms — refresh slowed to {} ms",
                    window.len() as f32 / audio::TARGET_RATE as f32,
                    hyp.infer_ms,
                    tick.as_millis(),
                ));
            }

            script.push_hypothesis(&hyp.text);
            last_paint = long_ago;
        }
    }

    // A held sentence that never got its continuation. The stream ending is as
    // final as the grace expiring, so it earns its place either way, and it
    // precedes the tail because it was spoken first.
    if let Some(p) = pending.take() {
        file(&mut script, &p.words);
    }

    // Anything still in flight when the stream ended. Commands are honoured
    // here too: an instruction spoken as the last thing before Ctrl-C should
    // not be silently downgraded into dictation.
    let tail = script.finalize_window();
    apply(&mut script, &tail, &args.assistant, &mut screen, &mut debounce);

    // One last write, for anything settled on the way out. Nothing is in flight
    // any more — the tail was just finalized and filed — so this also clears the
    // partial line any earlier tick left behind.
    if let Some(store) = &mut store {
        store.save(script.document(), script.revision(), "");
    }

    // Gives the terminal back, then writes the transcript where it persists.
    screen.finish(script.document());

    // Only reached on a clean exit, which is exactly the case where stdout has
    // the transcript and the session file is redundant. The paths that used to
    // lose text never get here, so their file survives.
    if let Some(kept) = store.and_then(store::Store::finish) {
        eprintln!("transcript saved to {}", kept.display());
    }

    match fatal {
        Some(e) => Err(e),
        None => Ok(()),
    }
}
