//! Real-time speech-to-text CLI.
//!
//! Rust owns capture, voice activity detection, windowing, the transcript state
//! machine and rendering. A Python sidecar owns the forward pass.
//!
//! This module is the orchestrator: it parses configuration, starts the audio
//! source and the sidecar, and runs the main loop that schedules inference,
//! trims the audio buffer, applies spoken commands and drives the renderer.

mod audio;
mod command;
mod render;
mod repair;
mod sidecar;
mod store;
mod text;
mod trace;
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
    allow_negative_numbers = true
)]
/// Command-line configuration.
struct Args {
    /// MLX model repository to load.
    ///
    /// The 1.7B default sustains the sliding window on Apple Silicon. Pass a
    /// 0.6B repository for more headroom on a slower host.
    #[arg(long, default_value = "mlx-community/Qwen3-ASR-1.7B-8bit")]
    model: String,

    /// Name that prefixes a spoken command.
    ///
    /// Say `NAME, delete` to drop the newest sentence, or `NAME, discard` for
    /// the newest sentence of what you are saying now. Also keep, undo, clear
    /// and copy. The comma is optional.
    #[arg(long, default_value = "Luna")]
    assistant: String,

    /// Force a language, for example "en". Omit to auto-detect.
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
    /// A floor rather than a fixed period: when inference on a long buffer
    /// outgrows it, the gap stretches so the duty cycle cannot exceed 100%.
    /// Lowering it speeds up commitment on short buffers only.
    #[arg(long, default_value_t = 500)]
    interval_ms: u64,

    /// Hypotheses that must agree before a word stops being provisional.
    ///
    /// Higher is calmer but slower. Values below 2 are raised to 2.
    #[arg(long, default_value_t = 3)]
    agreement: usize,

    /// Silence needed to end an utterance, in milliseconds.
    #[arg(long, default_value_t = 600)]
    endpoint_ms: u32,

    /// Sound must last this long, in milliseconds, before it counts as speech.
    ///
    /// This is the defence against a keyboard, and it tests duration because
    /// level cannot: a keystroke is louder than the room, so any floor that
    /// rejects one also rejects a quiet voice.
    ///
    /// Raise it if typing or a creaking chair is transcribed; lower it if a
    /// short first word is missed. It gates isolated transients only, so
    /// sustained typing needs `--rms-floor` instead. Values beyond the 300 ms
    /// pre-roll begin clipping first words.
    #[arg(long, default_value_t = vad::DEFAULT_OPEN_MS)]
    open_ms: u32,

    /// Look for a place to trim the audio buffer once it passes this many
    /// seconds.
    ///
    /// Never cuts mid-word: it cuts at a silence the speaker already left, and
    /// only where the text going with it has been filed. This bounds how far
    /// behind the transcript can fall, since inference and commit latency both
    /// grow with the buffer.
    ///
    /// Values below 10 are accepted but warned about, because the buffer then
    /// routinely passes the point at which the trim stops checking whether the
    /// text it severs has been filed.
    #[arg(long, default_value_t = 30)]
    trim_after_s: u64,

    /// Shortest silence, in milliseconds, before an utterance settles.
    ///
    /// Until it elapses the sentence stays dim and its audio is retained, so
    /// that speech resuming within the window is merged and re-decoded as one
    /// utterance rather than spliced from two. 0 settles immediately.
    ///
    /// This is a display parameter, not a linguistic one. Raising it is close
    /// to free on ordinary dictation, where every pause is already shorter than
    /// the floor; what it costs is how long your last words stay dim.
    #[arg(long, default_value_t = 15_000)]
    continue_ms: u64,

    /// Longest the settle may stretch to for a slow speaker, in milliseconds.
    ///
    /// The hold grows to cover pauses you have actually spoken through, one
    /// step per pause, so reaching this takes a habit rather than a single
    /// interruption.
    ///
    /// It is also how long a silence still counts as a continuation, and a
    /// continuation splices the previous utterance's audio back into the buffer
    /// instead of letting it reset. Set equal to `--continue-ms` to pin the
    /// hold and disable adaptation.
    #[arg(long, default_value_t = 30_000)]
    continue_max_ms: u64,

    /// Keep the session file on exit instead of deleting it.
    ///
    /// The transcript is written to disk as you speak either way. Without this
    /// the file is removed on a clean exit, once stdout has the transcript. An
    /// unclean exit leaves it behind regardless.
    #[arg(long)]
    persist: bool,

    /// Resume a previous session, appending to its transcript.
    ///
    /// Bare, it recovers the most recently written session file. Given a path,
    /// it resumes that file and leaves it untouched.
    ///
    /// Fails rather than starting empty when there is nothing to resume.
    #[arg(long, value_name = "PATH", num_args = 0..=1)]
    resume: Option<Option<PathBuf>>,

    /// Silence floor in dBFS: quieter frames are never speech.
    ///
    /// Raise it if a quiet room produces stray words. The floor rejects audio
    /// that is quiet, not audio that is non-speech, so set it above your room
    /// and below your voice.
    #[arg(long, default_value_t = vad::DEFAULT_FLOOR_DBFS)]
    rms_floor: f32,

    /// Suppress the notice reporting that text is settling behind the speaker.
    #[arg(long)]
    quiet: bool,

    /// Python interpreter for the sidecar.
    #[arg(long, default_value = ".venv/bin/python")]
    python: PathBuf,

    /// Sidecar script.
    #[arg(long, default_value = "sidecar/asr_sidecar.py")]
    script: PathBuf,
}

/// Silence retained ahead of speech onset, in samples, so the first phoneme is
/// not clipped.
const PREROLL_SAMPLES: usize = (audio::TARGET_RATE as usize * 300) / 1000;

/// Ceiling on silence spliced back into a merged utterance, in samples.
///
/// Beyond roughly a second the recogniser already reads a boundary, so
/// reproducing a longer pause faithfully would only cost encoder time.
const MAX_GAP_SAMPLES: usize = (audio::TARGET_RATE as usize * 1200) / 1000;

/// Frames of silence marking a gap worth cutting at, roughly 190 ms.
///
/// Well below any sane endpoint threshold, since these are the pauses inside a
/// passage the speaker is still in the middle of.
const DIP_FRAMES: usize = 12;

/// Shortest silence recorded as a possible cut point, roughly 64 ms.
///
/// Below [`DIP_FRAMES`], so these are not gaps anyone would choose: a silence
/// this short can be the closure of a stop consonant rather than a word
/// boundary. They are recorded because an unbounded buffer is worse than a
/// clipped consonant.
const MIN_DIP_FRAMES: usize = 4;

/// Silence that reads as a sentence boundary rather than a comma, roughly
/// 480 ms.
///
/// The bar the trim holds to while the buffer is still short enough to afford
/// waiting for a good cut.
const SENTENCE_DIP_FRAMES: usize = 30;

/// Fraction of the trim threshold at which the trim starts looking, provided
/// what it finds is sentence-grade.
///
/// Taking a good cut early is free, because the bar in this band is higher than
/// the one a later trim settles for.
const EAGER_TRIM_DIVISOR: usize = 2;

/// Multiple of the trim threshold past which any recorded silence becomes an
/// eligible cut point.
///
/// Only ever adds candidates in a situation that had none; the preference for
/// the longest gap is unchanged.
const DESPERATE_TRIM_MULTIPLE: usize = 2;

/// Multiple of the trim threshold past which the trim cuts regardless of
/// whether the text it severs has been filed.
///
/// Refusing to cut is bounded only while something else eventually files the
/// text. For a speaker producing one very long sentence nothing does, and an
/// unbounded buffer eventually stops the session responding.
///
/// Text filed here comes from a decode of the whole buffer where the two
/// decodes can be aligned, so a forced cut does not by itself put a severed
/// boundary in the document.
const FORCED_TRIM_MULTIPLE: usize = 2;

/// A silence the speaker left inside an utterance: somewhere the buffer may be
/// cut without slicing a word.
///
/// Length is retained because gaps are not equally good places to cut. A longer
/// pause is more likely to be a sentence boundary than a comma, so cutting at
/// the longest available candidate splits the transcript where it was going to
/// split anyway.
#[derive(Clone, Copy)]
struct Dip {
    at: usize,
    frames: usize,
    /// Words that must be filed before this cut is worth probing again.
    ///
    /// Probing costs a forward pass, and the answer cannot change until the
    /// document does. A refusal that stranded `n` words cannot become an
    /// acceptance until at least `n` further words are filed, because the
    /// stranded words are by construction the next ones in document order.
    ///
    /// Held on the gap rather than beside the buffer, because a refusal names a
    /// place in the audio and this is the structure that follows the audio
    /// through merges and trims.
    refused_until: Option<u64>,
}

/// Minimum audio kept on each side of a trim, in samples.
///
/// The head needs enough to be worth decoding alone; the tail needs enough that
/// the next window does not start from a sliver.
const MIN_SEGMENT_SAMPLES: usize = audio::TARGET_RATE as usize * 2;

/// Minimum gap between repeats of the same rate-limited notice.
const NOTICE_EVERY: Duration = Duration::from_secs(10);

/// Commit latency past which the live text is far enough behind the speaker to
/// be worth reporting.
///
/// Reported in preference to the duty cycle, which is pinned by construction
/// and therefore says nothing. What degrades is the delay before a word stops
/// being provisional, which is what the speaker sees.
const COMMIT_TARGET: Duration = Duration::from_millis(2_000);

/// Longest the live view may go without a repaint, so the settle countdown
/// advances and notices expire on time.
const REPAINT_EVERY: Duration = Duration::from_millis(200);

/// Set from a signal handler; the loop exits at the next iteration so the
/// transcript is still written and the terminal still restored.
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// Fraction of the remembered pause that survives each resumption.
///
/// A one-off interruption must not hold the settle open for the rest of the
/// session, so what a pause teaches has to fade. It fades slowly, because
/// holding too long costs dim time and is recoverable whereas filing too early
/// costs the transcript and is not.
///
/// Slow enough that a habit survives a paragraph of fluency. Growth is bounded
/// separately, in [`Settle::resumed`], so the fast decay that a single
/// unbounded observation would have required is unnecessary.
const PAUSE_DECAY_NUM: u32 = 127;
const PAUSE_DECAY_DEN: u32 = 128;

/// Margin applied to the observed pause when computing the hold: the speaker
/// paused that long once, so the next may be a little longer.
const PAUSE_MARGIN: u32 = 3;
const PAUSE_MARGIN_DIVISOR: u32 = 2;

/// How long a finished utterance is held before it files.
///
/// A pause is only a full stop in hindsight, so the audio is retained and
/// merged if speech resumes within the hold. The hold is a duration, but it
/// never inspects the text: it measures the speaker's pause habit and adjusts
/// only how long to wait.
///
/// # Adaptation
///
/// The hold starts at a floor and grows to cover pauses the speaker has
/// demonstrably spoken through, bounded by a ceiling. Growth is gradual, so a
/// habit reaches the ceiling in a few pauses while a single interruption
/// cannot.
///
/// Every resumption is observed, including those where the hold had already
/// expired. Learning only from pauses short enough to have been merged would be
/// blind to the only kind that damages a transcript.
///
/// # Bias
///
/// Holding too long leaves text dim for longer and is recoverable, since
/// in-flight text is persisted. Filing too early cuts a passage mid-thought and
/// damages the words either side of the cut permanently. The policy therefore
/// leans toward holding.
struct Settle {
    /// Shortest hold, from `--continue-ms`.
    floor: Duration,
    /// Longest hold adaptation may reach, from `--continue-max-ms`.
    ceiling: Duration,
    /// The speaker's demonstrated pause habit, as a decaying maximum of the
    /// silences they went on talking through.
    observed: Duration,
}

impl Settle {
    /// Creates a hold policy.
    ///
    /// A zero floor disables adaptation entirely, since it is an explicit
    /// instruction not to hold. A ceiling below the floor is raised to it.
    fn new(floor_ms: u64, ceiling_ms: u64) -> Self {
        let floor = Duration::from_millis(floor_ms);
        let ceiling = if floor.is_zero() {
            Duration::ZERO
        } else {
            Duration::from_millis(ceiling_ms).max(floor)
        };
        Self {
            floor,
            ceiling,
            observed: Duration::ZERO,
        }
    }

    /// Records that speech restarted after `gap` of silence.
    ///
    /// A single silence may raise the hold by at most one step, never straight
    /// to the ceiling, so growth requires the pause to recur at the longer
    /// hold. A silence is only evidence about how the speaker thinks if they
    /// were thinking, and an interruption is indistinguishable from a pause in
    /// the audio.
    ///
    /// This also bounds the buffer, since the hold decides how long a silence
    /// still counts as a continuation and a continuation splices the previous
    /// utterance's audio back in rather than letting it reset.
    fn resumed(&mut self, gap: Duration) {
        let faded = self.observed.saturating_mul(PAUSE_DECAY_NUM) / PAUSE_DECAY_DEN;
        self.observed = faded.max(gap.min(self.window()));
    }

    /// The current hold duration, clamped between the floor and the ceiling.
    fn window(&self) -> Duration {
        (self.observed.saturating_mul(PAUSE_MARGIN) / PAUSE_MARGIN_DIVISOR)
            .clamp(self.floor, self.ceiling)
    }
}

/// An utterance that has ended but whose audio is retained, in case the pause
/// that ended it turns out to have been a hesitation.
struct Pending {
    /// Retained audio, replayed into the buffer if the speaker resumes.
    audio: Vec<f32>,
    /// Gaps already found in `audio`, at offsets into it.
    ///
    /// Retained with the audio because a continuation puts that audio back at
    /// the front of the merged buffer with its pauses where they were. Without
    /// them a merged buffer would offer the trim only its own seam, one
    /// candidate per merge, however many sentences it holds.
    dips: Vec<Dip>,
    /// The unsettled tail of the utterance, shown dim and not yet filed: a
    /// continuation re-decodes it and may rewrite every word.
    words: Vec<String>,
    /// When the utterance ended, against which the hold is measured.
    at: Instant,
}

/// Signal handler for interrupt and termination.
///
/// The first signal sets a flag so the loop exits normally, leaving the
/// terminal restored and the transcript written. A second signal restores the
/// terminal and exits immediately, for the case where the loop is blocked in a
/// long forward pass. Only async-signal-safe calls are used on that path.
extern "C" fn on_signal(_: libc::c_int) {
    if INTERRUPTED.swap(true, Ordering::SeqCst) {
        const RESTORE: &[u8] = b"\x1b[?1049l\x1b[?25h";
        unsafe {
            libc::write(1, RESTORE.as_ptr().cast(), RESTORE.len());
            libc::_exit(130);
        }
    }
}

/// Installs the handler for `SIGINT` and `SIGTERM`.
///
/// Required because the default disposition would leave the alternate screen
/// entered, returning the user to a shell they cannot see.
fn install_signal_handlers() {
    unsafe {
        libc::signal(libc::SIGINT, on_signal as libc::sighandler_t);
        libc::signal(libc::SIGTERM, on_signal as libc::sighandler_t);
    }
}

/// Files a finished utterance into the document, one entry per sentence.
///
/// Splitting matters beyond line breaks: a spoken command removes one document
/// entry, so a re-decoded continuation the recogniser judged to be two thoughts
/// must arrive as two separately removable sentences.
///
/// # Returns
///
/// How many entries were filed, which is what bounds the scope of `discard`.
/// Counted from what actually landed rather than from the split, since empty
/// sentences are dropped.
fn file(script: &mut Transcript, words: &[String], why: &str) -> usize {
    let text = words.join(" ");
    let mut filed = 0;
    for part in text::split_sentences(&text) {
        let words: Vec<String> = part.split_whitespace().map(str::to_string).collect();
        if !words.is_empty() {
            trace::note(why, part);
            script.push_sentence(words);
            filed += 1;
        }
    }
    filed
}

/// Files whatever the current window has finished saying.
///
/// This is what makes commitment logical rather than timed: it asks the
/// recogniser's own output rather than a clock, on the basis that a sentence
/// with further complete sentences behind it has been given every chance to be
/// revised and has not been.
///
/// Declines while the assistant is named in the text about to be filed. Filing
/// a wake word as dictation would put it in the transcript permanently, before
/// the endpoint could read it as an instruction. The check is scoped to the
/// prefix being filed rather than the whole window: text past that prefix is
/// still parsed for commands at the endpoint, and widening the scope would stop
/// commitment for as long as a command sat in the window, which is what bounds
/// the buffer.
fn commit_settled(script: &mut Transcript, wake: &str) {
    let n = script.settled_prefix();
    if n == 0 || command::names(&script.committed()[..n], wake) {
        return;
    }
    trace::note("logical", &script.committed()[..n].join(" "));
    script.file_settled(n);
}

/// Proposes where the buffer may be cut, if anywhere.
///
/// Answers where, not whether. The caller applies a second test, since a
/// silence can be a good place to cut acoustically and still fall in the middle
/// of a sentence the document does not hold. This test is about the audio:
/// cutting anywhere but a silence slices a word, whatever the transcript says.
///
/// The bar falls as the buffer grows, because holding out for a sentence
/// boundary is affordable only while latency is not already suffering:
///
/// | buffer | eligible gaps |
/// |---|---|
/// | below the eager band | none |
/// | eager band | [`SENTENCE_DIP_FRAMES`], a sentence boundary |
/// | past the threshold | [`DIP_FRAMES`], a clause |
/// | past [`DESPERATE_TRIM_MULTIPLE`] | any recorded silence |
///
/// The longest gap wins and the latest breaks a tie, since a longer pause is
/// more likely to be a sentence boundary. Candidates turned down by the caller
/// are skipped until enough has been filed for the answer to have changed.
///
/// # Returns
///
/// An offset into a window of `window_len` samples, or `None` to keep growing.
fn cut_point(
    dips: &[Dip],
    window_len: usize,
    trim_after: usize,
    filed_words: u64,
) -> Option<usize> {
    if window_len < trim_after / EAGER_TRIM_DIVISOR {
        return None;
    }

    let dip_floor = if window_len >= trim_after.saturating_mul(DESPERATE_TRIM_MULTIPLE) {
        MIN_DIP_FRAMES
    } else if window_len >= trim_after {
        DIP_FRAMES
    } else {
        SENTENCE_DIP_FRAMES
    };

    dips.iter()
        .filter(|d| {
            d.frames >= dip_floor
                && d.at >= MIN_SEGMENT_SAMPLES
                && window_len.saturating_sub(d.at) >= MIN_SEGMENT_SAMPLES
                && d.refused_until.is_none_or(|until| filed_words >= until)
        })
        .max_by_key(|d| (d.frames, d.at))
        .map(|d| d.at)
}

/// Chooses a cut when the speaker has left nowhere good and the buffer can no
/// longer grow.
///
/// Every band in [`cut_point`] filters the recorded gaps, and a filter over an
/// empty list is still empty, so a speaker who leaves no silence at all has no
/// bound on the buffer. Speech-like background noise produces exactly that: the
/// detector never closes, no gap is ever recorded, and the buffer grows until
/// the session stops keeping up.
///
/// Runs in two steps: any recorded gap regardless of refusals, since a refusal
/// is advice about text quality and the buffer can no longer afford it; failing
/// that, the quietest frame in the buffer, which is not a gap but the least bad
/// place available.
///
/// # Returns
///
/// An offset into `window`, or `None` if no position leaves usable audio on
/// both sides.
fn last_resort(dips: &[Dip], window: &[f32]) -> Option<usize> {
    let usable = |at: usize| {
        at >= MIN_SEGMENT_SAMPLES && window.len().saturating_sub(at) >= MIN_SEGMENT_SAMPLES
    };

    if let Some(dip) = dips
        .iter()
        .filter(|d| d.frames >= MIN_DIP_FRAMES && usable(d.at))
        .max_by_key(|d| (d.frames, d.at))
    {
        return Some(dip.at);
    }

    let first = MIN_SEGMENT_SAMPLES.div_ceil(FRAME);
    let last = window.len().saturating_sub(MIN_SEGMENT_SAMPLES) / FRAME;
    (first..last)
        .min_by(|&a, &b| {
            let e = |i: usize| {
                window[i * FRAME..(i + 1) * FRAME]
                    .iter()
                    .map(|s| s * s)
                    .sum::<f32>()
            };
            e(a).total_cmp(&e(b))
        })
        .map(|f| f * FRAME)
}

/// Records that a cut was probed and turned down.
///
/// Everything from `cut` onward is turned down with it, since a later cut
/// severs the same sentence and more of it. Marking only the gap probed would
/// spend a forward pass per gap rediscovering that.
///
/// The threshold is a lower bound rather than the exact one, which would need a
/// probe. Erring low costs a wasted pass, where erring high would silently stop
/// bounding the buffer. An existing threshold is never lowered.
fn refuse_from(dips: &mut [Dip], cut: usize, stranded: usize, filed_words: u64) {
    let until = filed_words + stranded.max(1) as u64;
    for dip in dips.iter_mut().filter(|d| d.at >= cut) {
        dip.refused_until = Some(dip.refused_until.map_or(until, |had| had.max(until)));
    }
}

/// What advancing the detector over newly-arrived audio revealed.
#[derive(Debug, Default, PartialEq, Eq)]
struct Scan {
    /// Speech began.
    onset: bool,
    /// An utterance ended. Everything past `cursor` belongs to the next one.
    endpointed: bool,
}

/// Advances the detector over whole frames, stopping at an endpoint.
///
/// Stopping is a correctness requirement. The main loop drains the whole
/// capture backlog per iteration, so one call can cover as much audio as the
/// last forward pass took — longer than the endpoint threshold, meaning a
/// single batch can contain one utterance ending and the next beginning.
///
/// Running past the endpoint loses the second event silently: the onset is
/// reported before the caller has built the pending utterance it should merge
/// into, and because that onset reopens the detector no further onset is ever
/// emitted, leaving the held utterance unreachable rather than merely late.
///
/// Frames after the endpoint remain in the buffer and are scanned on the next
/// iteration, by which time the caller has filed its pending utterance.
///
/// Silences are recorded in `dips` as they are found, and a run still growing
/// updates the existing entry rather than adding a second one.
fn scan(
    vad: &mut Vad,
    window: &[f32],
    cursor: &mut usize,
    dips: &mut Vec<Dip>,
    has_speech: &mut bool,
) -> Scan {
    let mut out = Scan::default();

    while *cursor + FRAME <= window.len() {
        let ev = vad.push(&window[*cursor..*cursor + FRAME]);
        *cursor += FRAME;
        match ev {
            VadEvent::Onset => {
                *has_speech = true;
                out.onset = true;
            }
            VadEvent::Speaking => {
                *has_speech = true;
                let run = vad.quiet_run();
                if run >= MIN_DIP_FRAMES {
                    let started = cursor.saturating_sub(run * FRAME);
                    let inherited = dips.iter().filter_map(|d| d.refused_until).max();
                    let dip = Dip {
                        at: started + run * FRAME / 2,
                        frames: run,
                        refused_until: inherited,
                    };
                    match dips.last_mut() {
                        Some(last) if last.at >= started => {
                            let known = last.refused_until;
                            *last = Dip {
                                refused_until: known.max(inherited),
                                ..dip
                            };
                        }
                        _ => dips.push(dip),
                    }
                }
            }
            VadEvent::Endpoint => {
                out.endpointed = true;
                break;
            }
            VadEvent::Idle => {}
        }
    }

    out
}

/// Decides what a finalized utterance says, re-asking with the command
/// vocabulary when the first reading found no instruction.
///
/// The hint fixes verbs the recogniser otherwise splits, and also pulls
/// ambiguous audio toward the wake word. It is applied to every finalized
/// utterance rather than only those that appear to name the assistant, because
/// such a gate assumes the recogniser spelled the wake word correctly, which is
/// exactly what fails when the hint is needed.
///
/// That is safe because the hinted reading replaces the unprompted one only if
/// it parses as a command. A bare wake word with no verb is discarded, so every
/// word the speaker keeps still comes from the unprompted pass. The gate only
/// ever decided whether to spend a forward pass, which cannot damage text.
///
/// # Returns
///
/// `Ok(None)` when the unprompted reading stands. The caller retains ownership
/// of that reading in every case, including `Err`, so a dead transport does not
/// also cost the speaker the sentence they just finished.
///
/// # Errors
///
/// Propagates a transport failure. A refused window is reported and treated as
/// no result.
fn resolve(
    asr: &mut sidecar::Sidecar,
    pcm: &[f32],
    words: &[String],
    wake: &str,
    screen: &mut Renderer,
) -> Result<Option<Vec<String>>> {
    if words.is_empty() || command::has_command(words, wake) {
        return Ok(None);
    }

    match asr.transcribe_hinted(pcm)? {
        sidecar::Reply::Hypothesis(hyp) => {
            let hinted = repair::repair(hyp.text.split_whitespace());
            Ok(command::has_command(&hinted, wake).then_some(hinted))
        }
        sidecar::Reply::Failed(why) => {
            screen.notice(&format!("sidecar failed on the command pass: {why}"));
            Ok(None)
        }
    }
}

/// Applies a finalized utterance: dictation is filed and instructions are
/// executed, in the order they were spoken.
///
/// Every branch reports through the renderer, including those that did nothing.
/// A spoken command has no other acknowledgement, and silence is
/// indistinguishable from not being heard, which provokes a repeat.
///
/// # Scope of `discard`
///
/// A local count tracks the sentences this utterance filed and has not closed
/// off, and is the only budget `discard` may spend from. Every branch that
/// moves text keeps it honest: `delete` decrements it, `clear` and `rollback`
/// zero it, and `copy` leaves it alone because copying moves no text.
///
/// Without that, an utterance combining a delete and a discard would remove a
/// second sentence the utterance never filed.
fn apply(
    script: &mut Transcript,
    words: &[String],
    wake: &str,
    screen: &mut Renderer,
    why: &str,
) {
    let mut live = 0usize;

    for segment in command::split(words, wake) {
        match segment {
            Segment::Text(text) => live += file(script, &text, why),

            Segment::Run(Command::Delete) => {
                let msg = match script.delete_last() {
                    Some(dropped) => format!("delete — dropped \u{201c}{}\u{201d}", dropped.text()),
                    None => "delete — nothing left to drop".to_string(),
                };
                live = live.saturating_sub(1);
                screen.notice(&msg);
            }

            Segment::Run(Command::Discard) => {
                let n = script.discard_last(live.min(1));
                live -= n;
                screen.notice(&match n {
                    0 => format!(
                        "discard — nothing in flight; say \"{wake}, delete\" for the last sentence"
                    ),
                    _ => format!(
                        "discard — dropped the last sentence; \"{wake}, undo\" puts it back"
                    ),
                });
            }

            Segment::Run(Command::Keep) => {
                let n = live;
                live = 0;
                screen.notice(&match n {
                    0 => "keep — nothing in flight".to_string(),
                    n => format!("keep — filed {n} sentence(s)"),
                });
            }

            Segment::Run(Command::Clear) => {
                let n = script.clear();
                live = 0;
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
                live = 0;
                screen.notice(&msg);
            }

            Segment::Run(Command::Copy) => screen.notice(&copy(script)),
        }
    }
}

/// Copies the transcript to the clipboard.
///
/// Takes the whole document, with no approval filter. The boundary that does
/// apply is different: this stops at the document, so an utterance still inside
/// its hold is not copied. That guards against copying text the pipeline may
/// still rewrite, which is the guarantee only this program can make.
///
/// # Returns
///
/// A message for the status line, reporting success or failure.
fn copy(script: &Transcript) -> String {
    let text: Vec<String> = script
        .document()
        .iter()
        .map(transcript::Sentence::text)
        .collect();

    if text.is_empty() {
        return "copy — nothing to copy yet".to_string();
    }

    let n = text.len();
    match store::copy_to_clipboard(&(text.join("\n") + "\n")) {
        Ok(()) => format!("copy — {n} sentence(s) on the clipboard"),
        Err(e) => format!("copy failed: {e}"),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    if command::normalized_name(&args.assistant).is_empty() {
        anyhow::bail!(
            "--assistant needs a name with letters or digits in it;              {:?} would leave every spoken command unmatchable",
            args.assistant
        );
    }

    install_signal_handlers();
    trace::init();

    let previous = store::newest_session();

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

    let mut recovered: Vec<transcript::Sentence> = Vec::new();
    if let Some((path, _)) = &resume {
        let loaded = store::load(path).context("resuming a previous session")?;
        eprintln!("resumed {} sentence(s) from {}", loaded.len(), path.display());
        recovered = loaded;
    } else if let Some(path) = &previous {
        let n = store::load(path).map_or(0, |s| s.len());
        eprintln!(
            "note: a previous session is on disk ({n} sentence(s)) \
             — pass --resume to continue it: {}",
            path.display()
        );
    }

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

    let mut vad = Vad::new(args.open_ms, args.endpoint_ms, args.rms_floor);
    let agreement_n = args.agreement.max(2);
    let mut script = Transcript::new(args.agreement);
    script.restore(recovered);

    let mut window: Vec<f32> = Vec::new();
    let mut vad_cursor = 0usize;
    let mut has_speech = false;
    let mut last_tick = Instant::now();

    let mut dips: Vec<Dip> = Vec::new();

    let long_ago = Instant::now()
        .checked_sub(NOTICE_EVERY)
        .unwrap_or_else(Instant::now);
    let mut last_slowed = long_ago;
    let mut last_failed = long_ago;

    let mut pending: Option<Pending> = None;
    let mut settle = Settle::new(args.continue_ms, args.continue_max_ms);
    let mut last_endpoint: Option<Instant> = None;
    let mut gap_samples = 0usize;

    let interval = Duration::from_millis(args.interval_ms);
    let mut tick = interval;
    let mut probe_debt = Duration::ZERO;
    let mut last_infer = Duration::ZERO;
    let trim_after = args.trim_after_s as usize * audio::TARGET_RATE as usize;

    const TRIM_FLOOR_S: u64 = 10;
    if args.trim_after_s < TRIM_FLOOR_S {
        eprintln!(
            "warning: --trim-after-s {} is below {TRIM_FLOOR_S}; the buffer will routinely pass \
             {}s, past which the trim stops checking whether the text it severs has been filed \
             and may split a sentence in two",
            args.trim_after_s,
            args.trim_after_s * FORCED_TRIM_MULTIPLE as u64,
        );
    }

    eprintln!(
        "listening — just speak; say \"{name}, delete\" to drop the last \
         sentence anywhere, or \"{name}, discard\" for the last one you just \
         said; then keep / undo / clear / copy; Ctrl-C to stop",
        name = args.assistant
    );

    let mut screen = Renderer::new();
    let mut last_paint = long_ago;

    let mut fatal: Option<anyhow::Error> = None;

    loop {
        if INTERRUPTED.load(Ordering::SeqCst) {
            break;
        }

        for msg in capture.notices.try_iter().chain(asr.notices().try_iter()) {
            screen.notice(&msg);
        }

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

        if let Some(store) = &mut store
            && let Some(err) = store.save(script.document(), script.revision(), &in_flight)
        {
            screen.notice(&err);
        }

        if last_paint.elapsed() >= REPAINT_EVERY {
            last_paint = Instant::now();
            let state = match &pending {
                Some(p) => render::State::Settling(
                    settle
                        .window()
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
                for chunk in capture.pcm.try_iter() {
                    window.extend_from_slice(&chunk);
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }

        let Scan { onset, endpointed } =
            scan(&mut vad, &window, &mut vad_cursor, &mut dips, &mut has_speech);

        if onset && let Some(ended) = last_endpoint.take() {
            settle.resumed(ended.elapsed());
        }

        if let Some(p) = &pending
            && p.at.elapsed() >= settle.window()
        {
            let expired = pending.take().expect("just checked");
            file(&mut script, &expired.words, "settle");
            script.forget_filed();
            last_paint = long_ago;
        }

        if onset
            && let Some(p) = pending.take_if(|p| p.at.elapsed() < settle.window())
        {
            let gap = gap_samples.min(MAX_GAP_SAMPLES);
            let held = p.audio.len();
            let mut merged = p.audio;
            merged.resize(merged.len() + gap, 0.0);
            let shift = merged.len();
            merged.extend_from_slice(&window);

            for dip in &mut dips {
                dip.at += shift;
            }
            if gap > 0 {
                dips.push(Dip {
                    at: held + gap / 2,
                    frames: gap / FRAME,
                    refused_until: p.dips.iter().filter_map(|d| d.refused_until).max(),
                });
            }
            dips.extend(p.dips);
            dips.sort_unstable_by_key(|d| d.at);

            trace::note(
                "merge",
                &format!(
                    "held={:.2}s gap={:.2}s buf={:.2}s dips={}",
                    held as f32 / audio::TARGET_RATE as f32,
                    gap as f32 / audio::TARGET_RATE as f32,
                    merged.len() as f32 / audio::TARGET_RATE as f32,
                    dips.len(),
                ),
            );
            window = merged;
            vad_cursor += shift;
            last_tick = long_ago;
            gap_samples = 0;
        }

        if !has_speech && window.len() > PREROLL_SAMPLES {
            let drop = window.len() - PREROLL_SAMPLES;
            window.drain(..drop);
            vad_cursor = vad_cursor.saturating_sub(drop);
            dips.clear();
            gap_samples += drop;
            continue;
        }

        if endpointed {
            let ended = {
                let after = window.split_off(vad_cursor.min(window.len()));
                std::mem::replace(&mut window, after)
            };
            vad_cursor = 0;
            dips.retain(|d| d.at < ended.len());

            if has_speech {
                match asr.transcribe(&ended) {
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

                let mut utterance = script.finalize_window();
                match resolve(&mut asr, &ended, &utterance, &args.assistant, &mut screen) {
                    Ok(Some(hinted)) => utterance = hinted,
                    Ok(None) => {}
                    Err(e) => fatal = Some(e),
                }

                if command::has_command(&utterance, &args.assistant) {
                    apply(&mut script, &utterance, &args.assistant, &mut screen, "command");
                    script.forget_filed();
                } else if utterance.is_empty() || settle.window().is_zero() {
                    file(&mut script, &utterance, "endpoint");
                    script.forget_filed();
                } else {
                    if let Some(stale) = pending.take() {
                        file(&mut script, &stale.words, "stale");
                    }

                    let settled = transcript::settled_words(&utterance);
                    let held = utterance.split_off(settled);
                    if !utterance.is_empty() {
                        file(&mut script, &utterance, "endpoint-settled");
                    }

                    pending = Some(Pending {
                        audio: ended,
                        dips: std::mem::take(&mut dips),
                        words: held,
                        at: Instant::now(),
                    });
                    gap_samples = 0;
                }
            }
            dips.clear();
            has_speech = false;
            last_endpoint = Some(Instant::now());
            last_tick = Instant::now();
            tick = interval;
            probe_debt = Duration::ZERO;
            last_paint = long_ago;
            if fatal.is_some() {
                break;
            }
            continue;
        }

        let proposal = cut_point(&dips, window.len(), trim_after, script.filed_words()).or_else(
            || {
                (window.len() >= trim_after.saturating_mul(FORCED_TRIM_MULTIPLE))
                    .then(|| last_resort(&dips, &window))
                    .flatten()
            },
        );

        if let Some(cut) = proposal {
            let probe_at = Instant::now();
            let head = match asr.transcribe(&window[..cut]) {
                Ok(sidecar::Reply::Hypothesis(hyp)) => Some(hyp.text),
                Ok(sidecar::Reply::Failed(why)) => {
                    screen.notice(&format!("sidecar failed on the trim pass: {why}"));
                    None
                }
                Err(e) => {
                    fatal = Some(e);
                    break;
                }
            };

            trace::note(
                "probe",
                &format!(
                    "buf={:.2}s cut={:.2}s head_pass={}ms filed={}",
                    window.len() as f32 / audio::TARGET_RATE as f32,
                    cut as f32 / audio::TARGET_RATE as f32,
                    probe_at.elapsed().as_millis(),
                    script.filed_words(),
                ),
            );

            let forced = window.len() >= trim_after.saturating_mul(FORCED_TRIM_MULTIPLE);
            let stranded = head.as_deref().map(|text| script.unfiled(text));
            let spent = head.as_deref().and_then(|text| script.spent_by(text));

            let cut_it = match (spent, stranded, forced) {
                (Some(k), _, _) => {
                    trace::note(
                        "trim",
                        &format!(
                            "cut at {:.2}s, spends {k} filed word(s)",
                            cut as f32 / audio::TARGET_RATE as f32
                        ),
                    );
                    script.forget_filed_prefix(k);
                    true
                }

                (None, Some(0), true) => {
                    trace::note(
                        "trim-forced",
                        &format!(
                            "cut at {:.2}s mid-sentence; text already filed, nothing added",
                            cut as f32 / audio::TARGET_RATE as f32
                        ),
                    );
                    let k = head.as_deref().map_or(0, |t| script.filed_prefix_of(t));
                    script.forget_filed_prefix(k);
                    true
                }

                (None, Some(n), true) => {
                    let text = head.unwrap_or_default();
                    let spent_here = script.filed_prefix_of(&text);

                    let mut whole = script.finalize_window();
                    if whole.is_empty() {
                        match asr.transcribe(&window) {
                            Ok(sidecar::Reply::Hypothesis(h)) => {
                                script.push_hypothesis(&h.text);
                                whole = script.finalize_window();
                            }
                            Ok(sidecar::Reply::Failed(why)) => screen
                                .notice(&format!("sidecar failed on the forced trim: {why}")),
                            Err(e) => fatal = Some(e),
                        }
                    }

                    let raw = text::words(&text);
                    let head_words = raw[raw.len().saturating_sub(n)..].to_vec();
                    let covers = transcript::words_covering(&whole, &head_words);

                    let mut words = if covers > 0 {
                        trace::note(
                            "trim-forced",
                            &format!(
                                "cut at {:.2}s, filing {covers} of {} word(s) of the whole-buffer decode",
                                cut as f32 / audio::TARGET_RATE as f32,
                                whole.len(),
                            ),
                        );
                        whole[..covers].to_vec()
                    } else {
                        trace::note(
                            "trim-forced",
                            &format!(
                                "cut at {:.2}s, head unplaceable (whole={} head={}); \
                                 filing {n} severed word(s)",
                                cut as f32 / audio::TARGET_RATE as f32,
                                whole.len(),
                                head_words.len(),
                            ),
                        );
                        script.push_hypothesis(&text);
                        script.finalize_window()
                    };

                    match resolve(&mut asr, &window[..cut], &words, &args.assistant, &mut screen) {
                        Ok(Some(hinted)) => words = hinted,
                        Ok(None) => {}
                        Err(e) => fatal = Some(e),
                    }
                    apply(&mut script, &words, &args.assistant, &mut screen, "trim-forced");
                    script.forget_filed_prefix(spent_here);
                    true
                }

                (None, stranded, false) | (None, stranded @ None, true) => {
                    if let Some(n) = stranded {
                        let idle = tick.saturating_sub(last_infer);
                        probe_debt += probe_at.elapsed().saturating_sub(idle);
                        trace::note(
                            "trim-refused",
                            &format!(
                                "cut at {:.2}s would strand {n} word(s)",
                                cut as f32 / audio::TARGET_RATE as f32
                            ),
                        );
                        refuse_from(&mut dips, cut, n, script.filed_words());
                    }
                    false
                }
            };

            if fatal.is_some() {
                break;
            }

            if cut_it {
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
        }

        if has_speech && vad.is_active() && last_tick.elapsed() >= tick + probe_debt {
            last_tick = Instant::now();
            let charged = std::mem::take(&mut probe_debt);

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

            let infer = Duration::from_millis(hyp.infer_ms);
            last_infer = infer;
            tick = interval.max(infer + infer / 4);

            let commit = tick * (agreement_n - 1) as u32 + charged + infer;
            if commit > COMMIT_TARGET && !args.quiet && last_slowed.elapsed() >= NOTICE_EVERY {
                last_slowed = Instant::now();
                screen.notice(&format!(
                    "text is settling {:.1}s behind \u{2014} {:.0}s buffer, {} ms per pass; \
                     say \u{201c}{name}, keep\u{201d} to file what you have said and start fresh",
                    commit.as_secs_f32(),
                    window.len() as f32 / audio::TARGET_RATE as f32,
                    hyp.infer_ms,
                    name = args.assistant,
                ));
            }

            trace::note(
                "tick",
                &format!(
                    "buf={:.2}s infer={}ms tick={}ms debt={}ms commit={:.2}s dips={} \
                     cwords={} settled={} filed={}",
                    window.len() as f32 / audio::TARGET_RATE as f32,
                    hyp.infer_ms,
                    tick.as_millis(),
                    charged.as_millis(),
                    commit.as_secs_f32(),
                    dips.len(),
                    script.committed().len(),
                    script.settled_prefix(),
                    script.filed_words(),
                ),
            );

            script.push_hypothesis(&hyp.text);
            commit_settled(&mut script, &args.assistant);
            last_paint = long_ago;
        }
    }

    if let Some(p) = pending.take() {
        file(&mut script, &p.words, "eos-held");
    }

    let mut tail = script.finalize_window();
    if fatal.is_none()
        && let Ok(Some(hinted)) =
            resolve(&mut asr, &window, &tail, &args.assistant, &mut screen)
    {
        tail = hinted;
    }
    apply(&mut script, &tail, &args.assistant, &mut screen, "eos");

    if let Some(store) = &mut store {
        store.save(script.document(), script.revision(), "");
    }

    screen.finish(script.document());

    if let Some(kept) = store.and_then(store::Store::finish) {
        eprintln!("transcript saved to {}", kept.display());
    }

    match fatal {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Defaults, so the tests read against the shipped configuration.
    fn settle() -> Settle {
        Settle::new(15_000, 60_000)
    }

    fn secs(n: u64) -> Duration {
        Duration::from_secs(n)
    }

    /// Nothing observed yet: the floor is the whole policy.
    #[test]
    fn the_settle_starts_at_the_floor() {
        assert_eq!(settle().window(), secs(15));
    }

    /// The reported failure, at the level the policy decides it. A speaker
    /// pausing to think reaches here having already spent `--endpoint-ms`, so a
    /// 7s pause arrives as ~6.3s — which the old fixed 6s settle filed, cutting
    /// a two-sentence passage into six fragments.
    #[test]
    fn an_ordinary_thinking_pause_is_inside_the_floor() {
        let s = settle();
        assert!(
            Duration::from_millis(6_300) < s.window(),
            "a 7s thinking pause must not file the utterance"
        );
    }

    /// Past the floor the speaker teaches it. One fragment is still lost — the
    /// first pause is observed only when it ends — and everything after it is
    /// held, which is the whole of what adaptation buys.
    #[test]
    fn a_slow_speaker_stretches_the_settle_past_the_floor() {
        let mut s = settle();
        assert!(secs(19) > s.window(), "a 20s pause starts out too long to hold");

        s.resumed(Duration::from_millis(19_180));
        assert!(
            secs(19) < s.window(),
            "having seen one, the next must be held: {:?}",
            s.window()
        );
    }

    /// A pause shorter than the floor must not drag the settle *down*. The
    /// floor is a floor.
    #[test]
    fn a_brisk_speaker_never_shortens_the_settle() {
        let mut s = settle();
        for _ in 0..10 {
            s.resumed(Duration::from_millis(400));
        }
        assert_eq!(s.window(), secs(15));
    }

    /// A thinking habit must survive a run of fluency. This is how people
    /// dictate — think, flow for a paragraph, think again — and the first
    /// version of this dropped the window off a cliff on the fifth flowing
    /// pause, so every later thinking pause was shredded afresh.
    #[test]
    fn a_thinking_habit_survives_a_paragraph_of_fluency() {
        let mut s = settle();
        s.resumed(secs(20));

        for i in 0..12 {
            s.resumed(Duration::from_millis(1_500));
            assert!(
                s.window() >= secs(20),
                "a 20s pause must still be held after {} flowing pauses, got {:?}",
                i + 1,
                s.window()
            );
        }
    }

    /// It still has to fade, or one interruption pins the settle open for the
    /// session. Gradually, though — there must be no pause at which behaviour
    /// changes abruptly.
    #[test]
    fn a_one_off_interruption_washes_out_gradually() {
        let mut s = settle();
        s.resumed(secs(300));
        assert!(
            s.window() < s.ceiling,
            "one interruption must not reach the ceiling: {:?}",
            s.window()
        );

        let mut seen = vec![s.window()];
        for _ in 0..200 {
            s.resumed(Duration::from_millis(1_500));
            seen.push(s.window());
        }
        assert_eq!(*seen.last().expect("non-empty"), secs(15), "it does return to the floor");

        let worst = seen
            .windows(2)
            .map(|w| w[0].saturating_sub(w[1]))
            .max()
            .expect("non-empty");
        assert!(worst <= secs(2), "decay must be gradual, worst step was {worst:?}");
    }

    /// A single absence must move the hold one step rather than teaching the
    /// ceiling outright, which made the hold jump with nothing on screen to
    /// explain it and take dozens of ordinary pauses to come back.
    #[test]
    fn a_single_long_absence_moves_the_hold_one_step_not_to_the_ceiling() {
        let mut s = settle();
        let before = s.window();
        s.resumed(secs(600));
        let after = s.window();

        assert!(after > before, "it must still learn something");
        assert!(
            after < s.ceiling,
            "one absence must not reach the ceiling: {after:?}"
        );
        assert_eq!(after, before * PAUSE_MARGIN / PAUSE_MARGIN_DIVISOR);
    }

    /// A habit still reaches the ceiling, because the pause happens again at
    /// the longer hold and teaches from there. Growth is gradual, not capped.
    #[test]
    fn a_repeated_pause_still_reaches_the_ceiling() {
        let mut s = settle();
        for _ in 0..5 {
            s.resumed(secs(40));
        }
        assert_eq!(s.window(), s.ceiling);
    }

    /// `--continue-ms 0` is documented as settling immediately. Adapting past
    /// it would override an explicit instruction not to hold.
    #[test]
    fn a_zero_floor_switches_adaptation_off() {
        let mut s = Settle::new(0, 60_000);
        assert!(s.window().is_zero());
        s.resumed(secs(20));
        assert!(s.window().is_zero(), "an explicit 0 must stay 0");
    }

    /// `--continue-max-ms` below `--continue-ms` must not invert the two.
    #[test]
    fn a_ceiling_under_the_floor_does_not_invert() {
        let mut s = Settle::new(15_000, 5_000);
        assert_eq!(s.window(), secs(15));
        s.resumed(secs(30));
        assert_eq!(s.window(), secs(15), "pinning the ceiling disables adaptation");
    }

    fn words(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    fn texts(script: &Transcript) -> Vec<String> {
        script
            .document()
            .iter()
            .map(transcript::Sentence::text)
            .collect()
    }

    const RATE: usize = audio::TARGET_RATE as usize;

    /// Longest silence a merge seam can carry, in milliseconds.
    const MAX_GAP_MS: usize = 1200;

    fn dip(at_s: f32, ms: usize) -> Dip {
        Dip {
            at: (at_s * RATE as f32) as usize,
            frames: ms * RATE / 1000 / FRAME,
            refused_until: None,
        }
    }

    /// Default `--trim-after-s`, in samples.
    const TRIM_AFTER: usize = 30 * RATE;

    /// Below half the threshold nothing is cut, however good the gap.
    #[test]
    fn a_short_buffer_is_never_cut() {
        let gaps = [dip(5.0, 576)];
        assert_eq!(cut_point(&gaps, 14 * RATE, TRIM_AFTER, 0), None);
        assert_eq!(cut_point(&gaps, 15 * RATE, TRIM_AFTER, 0), Some(5 * RATE));
    }

    /// The eager band holds out for a sentence boundary. A comma is not one,
    /// and waiting costs latency that the later bands will spend anyway.
    #[test]
    fn the_eager_band_refuses_a_comma_and_takes_a_boundary() {
        let comma = [dip(6.0, 208)];
        assert_eq!(cut_point(&comma, 16 * RATE, TRIM_AFTER, 0), None, "a comma is not a boundary");

        assert_eq!(cut_point(&comma, 30 * RATE, TRIM_AFTER, 0), Some(6 * RATE));

        let boundary = [dip(6.0, 512)];
        assert_eq!(cut_point(&boundary, 16 * RATE, TRIM_AFTER, 0), Some(6 * RATE));
    }

    /// A speaker who never leaves a real gap still gets a bounded buffer.
    #[test]
    fn the_desperate_band_accepts_any_recorded_silence() {
        let scrappy = [dip(20.0, 80)];
        assert_eq!(cut_point(&scrappy, 40 * RATE, TRIM_AFTER, 0), None);
        assert_eq!(cut_point(&scrappy, 60 * RATE, TRIM_AFTER, 0), Some(20 * RATE));
    }

    /// Longest wins, and latest only breaks a tie. Latest-wins was tried here
    /// and measured worse: guarded by the same 480ms floor it still stranded
    /// `The measurement should come first,` as its own entry, while bounding the
    /// buffer no better than this does. See [`cut_point`].
    #[test]
    fn the_cut_is_the_longest_boundary_not_merely_the_latest() {
        let merge_seam = dip(9.8, MAX_GAP_MS);
        let sentence_boundary = dip(18.0, 512);
        let later_shorter_boundary = dip(25.5, 496);
        let gaps = [merge_seam, sentence_boundary, later_shorter_boundary];
        let cut = cut_point(&gaps, 30 * RATE, TRIM_AFTER, 0).expect("a cut");
        assert_eq!(cut, (9.8 * RATE as f32) as usize, "the longest gap wins");

        let tied = [dip(8.0, 512), dip(21.0, 512)];
        assert_eq!(
            cut_point(&tied, 30 * RATE, TRIM_AFTER, 0),
            Some(21 * RATE),
            "equal gaps: prefer the one that removes more"
        );
    }

    /// Both segments have to be worth decoding on their own.
    #[test]
    fn a_cut_never_leaves_a_sliver_on_either_side() {
        assert_eq!(cut_point(&[dip(1.0, 576)], 20 * RATE, TRIM_AFTER, 0), None);
        assert_eq!(cut_point(&[dip(19.5, 576)], 20 * RATE, TRIM_AFTER, 0), None);
        assert!(cut_point(&[dip(10.0, 576)], 20 * RATE, TRIM_AFTER, 0).is_some());
    }

    /// No gaps at all means no cut — never a fallback to a sample offset.
    #[test]
    fn nothing_is_cut_when_the_speaker_left_no_gap() {
        assert_eq!(cut_point(&[], 60 * RATE, TRIM_AFTER, 0), None);
    }

    /// A cut that was probed and turned down takes everything after it out of
    /// contention, so the next tick walks back to the candidate before it
    /// rather than spending another forward pass on the same answer.
    #[test]
    fn a_refused_cut_hands_the_search_back_to_the_candidate_before_it() {
        let mut gaps = [dip(6.0, 512), dip(14.0, 1200), dip(22.0, 800)];

        assert_eq!(cut_point(&gaps, 30 * RATE, TRIM_AFTER, 100), Some(14 * RATE));

        refuse_from(&mut gaps, 14 * RATE, 20, 100);
        assert_eq!(
            cut_point(&gaps, 30 * RATE, TRIM_AFTER, 100),
            Some(6 * RATE),
            "22s is later than the refusal and must not be reconsidered"
        );

        refuse_from(&mut gaps, 6 * RATE, 5, 100);
        assert_eq!(cut_point(&gaps, 30 * RATE, TRIM_AFTER, 100), None);
    }

    /// A refusal lapses only once enough has been filed for the answer to have
    /// changed. Keying on the document version instead lapses it on any filing,
    /// re-probing cuts whose answer cannot have moved.
    #[test]
    fn a_refusal_lapses_only_once_enough_has_been_filed() {
        let mut gaps = [dip(6.0, 512), dip(14.0, 1200)];
        refuse_from(&mut gaps, 6 * RATE, 50, 100);
        assert_eq!(cut_point(&gaps, 30 * RATE, TRIM_AFTER, 100), None);

        assert_eq!(
            cut_point(&gaps, 30 * RATE, TRIM_AFTER, 109),
            None,
            "a 9-word sentence cannot have filed 50 words of stranded text"
        );

        assert_eq!(
            cut_point(&gaps, 30 * RATE, TRIM_AFTER, 150),
            Some(14 * RATE),
            "enough has been filed: every candidate is worth asking about again"
        );
    }

    /// A second refusal must not shorten the wait an earlier one imposed.
    /// `refuse_from` marks everything past the cut, and a gap already waiting on
    /// more than this refusal needs is waiting for a reason.
    #[test]
    fn a_refusal_never_lowers_a_threshold_already_set() {
        let mut gaps = [dip(6.0, 512), dip(14.0, 1200)];
        refuse_from(&mut gaps, 14 * RATE, 60, 0);
        refuse_from(&mut gaps, 6 * RATE, 5, 0);

        assert_eq!(
            cut_point(&gaps, 30 * RATE, TRIM_AFTER, 10),
            Some(6 * RATE),
            "the 14s gap still needs 60 words filed, not 5"
        );
        assert_eq!(cut_point(&gaps, 30 * RATE, TRIM_AFTER, 60), Some(14 * RATE));
    }

    /// A refusal must survive everything that moves the buffer, which is why it
    /// is held on the gap rather than beside it: a merge splices held gaps back
    /// at their original offsets.
    #[test]
    fn a_refusal_travels_with_the_gap_through_a_merge() {
        let mut held = [dip(6.0, 512), dip(14.0, 1200)];
        refuse_from(&mut held, 6 * RATE, 40, 0);

        let shift = 20 * RATE;
        let mut merged: Vec<Dip> = held.to_vec();
        merged.push(dip(2.0 + 20.0, 576));
        merged.sort_unstable_by_key(|d| d.at);

        assert_eq!(
            cut_point(&merged, 40 * RATE, TRIM_AFTER, 10),
            Some(shift + 2 * RATE),
            "only the gap the merge is seeing for the first time is worth a pass"
        );
    }

    /// The predicate the trim cuts on, end to end: the document decides whether
    /// the head is spent, and nothing else is consulted. A cut that would sever
    /// a sentence mid-way is declined and the buffer grows instead.
    #[test]
    fn the_trim_cuts_only_where_a_filed_sentence_ends() {
        let mut script = Transcript::new(2);
        script.push_sentence(words("It tries to solve the problem of coordination."));

        assert_eq!(
            script.spent_by("It tries to solve the problem of coordination."),
            Some(8)
        );

        assert_eq!(script.unfiled("It tries to solve the problem."), 0, "the words are all filed");
        assert_eq!(
            script.spent_by("It tries to solve the problem."),
            None,
            "but the rest of that sentence is still in the buffer"
        );
    }

    /// A cut at an earlier sentence boundary must be taken. Requiring the head
    /// to account for everything filed refuses it, reporting a sentence already
    /// in the document as new and eventually filing it twice.
    #[test]
    fn a_cut_at_an_earlier_sentence_boundary_is_taken_not_refused() {
        let mut script = Transcript::new(2);
        script.push_sentence(words("The deployment finished at noon, and everything looked stable."));
        script.push_sentence(words("I think the latency numbers are worth a second look."));

        assert_eq!(
            script.spent_by("The deployment finished at noon, and everything looked stable."),
            Some(9),
            "a whole filed sentence with more filed behind it is a safe cut"
        );

        script.forget_filed_prefix(9);
        assert_eq!(
            script.spent_by("I think the latency numbers are worth a second look."),
            Some(10)
        );
    }

    /// Nothing filed yet: no cut can be endorsed, whatever the head says.
    #[test]
    fn nothing_filed_means_nothing_to_spend() {
        let script = Transcript::new(2);
        assert_eq!(script.spent_by("Some words the model produced."), None);
        assert_eq!(script.spent_by(""), Some(0));
    }

    /// A 200 Hz tone at `amplitude`, `frames` frames of it, continuous in phase
    /// so `earshot` sees a coherent signal rather than a chopped one.
    fn tone(frames: usize, amplitude: f32, phase: &mut usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(frames * FRAME);
        for _ in 0..frames {
            for n in 0..FRAME {
                let t = (*phase * FRAME + n) as f32 / 16_000.0;
                out.push(amplitude * (2.0 * std::f32::consts::PI * 200.0 * t).sin());
            }
            *phase += 1;
        }
        out
    }

    /// One batch can hold an utterance ending and the next beginning, since the
    /// loop drains the whole backlog. Running past the endpoint reports an onset
    /// before the caller has a pending utterance to merge it into, and leaves the
    /// detector active so no further onset ever arrives. Stopping at the endpoint
    /// puts the two events in separate iterations.
    #[test]
    fn a_scan_stops_at_an_endpoint_so_the_next_onset_survives() {
        let mut vad = Vad::new(150, 600, vad::DEFAULT_FLOOR_DBFS);
        let mut phase = 0usize;

        let mut window = tone(60, 0.3, &mut phase);
        window.extend(tone(60, 0.0, &mut phase));
        window.extend(tone(60, 0.3, &mut phase));

        let mut cursor = 0usize;
        let mut dips = Vec::new();
        let mut has_speech = false;

        let first = scan(&mut vad, &window, &mut cursor, &mut dips, &mut has_speech);
        assert!(first.endpointed, "the pause has to end the utterance");
        assert!(
            cursor < window.len(),
            "the scan must leave the resumption for the next iteration, \
             not consume it before the caller can hold anything"
        );

        has_speech = false;
        let resumed = scan(&mut vad, &window, &mut cursor, &mut dips, &mut has_speech);
        assert!(resumed.onset, "the resumption must still be reported");
        assert!(!resumed.endpointed, "and must not be mistaken for another end");
        assert!(has_speech);
    }

    /// The ordinary case has to keep working: no endpoint means the scan
    /// consumes every whole frame it was given.
    #[test]
    fn a_scan_without_an_endpoint_consumes_the_whole_buffer() {
        let mut vad = Vad::new(150, 600, vad::DEFAULT_FLOOR_DBFS);
        let mut phase = 0usize;
        let window = tone(40, 0.3, &mut phase);

        let mut cursor = 0usize;
        let mut dips = Vec::new();
        let mut has_speech = false;

        let out = scan(&mut vad, &window, &mut cursor, &mut dips, &mut has_speech);
        assert!(out.onset && !out.endpointed);
        assert_eq!(cursor, window.len(), "nothing may be left behind");
    }

    /// Runs one finalized utterance through the dispatch.
    ///
    /// The renderer is inert under test, since stdout is not a terminal. That is
    /// the same path a piped session takes, so nothing is stubbed out.
    fn say(script: &mut Transcript, utterance: &str) {
        let mut screen = Renderer::new();
        apply(script, &words(utterance), "Luna", &mut screen, "test");
    }

    /// Both verbs take one sentence. What differs is how far back each reaches,
    /// and `discard` taking only the newest is the point: an utterance spans a
    /// paragraph once thinking pauses merge into it, so taking the whole
    /// utterance threw the paragraph away to correct one sentence.
    #[test]
    fn discard_drops_only_the_newest_sentence_of_the_utterance() {
        let mut script = Transcript::new(2);
        say(&mut script, "First thought. Second thought. Luna, discard.");
        assert_eq!(texts(&script), vec!["First thought."], "only the newest goes");

        let mut script = Transcript::new(2);
        say(
            &mut script,
            "First thought. Second thought. Luna, discard. Luna, discard.",
        );
        assert!(script.document().is_empty(), "{:?}", texts(&script));

        let mut script = Transcript::new(2);
        say(&mut script, "Older sentence.");
        say(
            &mut script,
            "First thought. Second thought. Luna, discard. Luna, discard. Luna, discard.",
        );
        assert_eq!(texts(&script), vec!["Older sentence."]);
    }

    /// The safety property that makes `discard` sayable without looking at the
    /// screen: it is scoped to this breath and cannot walk backwards into the
    /// transcript, however many times it is said.
    #[test]
    fn discard_cannot_reach_an_earlier_utterance() {
        let mut script = Transcript::new(2);
        say(&mut script, "Settled a while ago.");

        for _ in 0..3 {
            say(&mut script, "Luna, discard.");
        }
        assert_eq!(texts(&script), vec!["Settled a while ago."]);

        say(&mut script, "Luna, delete.");
        assert!(script.document().is_empty());
    }

    /// Both verbs in one breath must not overlap. Without `live` tracking the
    /// delete, the discard would take a second sentence this utterance never
    /// filed — a destructive command silently widening its own scope.
    #[test]
    fn a_delete_and_a_discard_in_one_breath_do_not_overlap() {
        let mut script = Transcript::new(2);
        say(&mut script, "Older sentence.");
        say(&mut script, "New sentence. Luna, delete. Luna, discard.");
        assert_eq!(texts(&script), vec!["Older sentence."]);
    }

    /// Addressing the assistant must not stop commitment. A command sits at the
    /// end of an utterance and stays in the window until the endpoint, so a
    /// guard scoped to the whole window would block every filing from the moment
    /// the wake word was decoded — and commitment is what bounds the buffer.
    #[test]
    fn a_wake_word_in_the_tail_does_not_block_commitment() {
        let mut script = Transcript::new(2);
        for _ in 0..3 {
            script.push_hypothesis("One. Two. Three. Luna delete");
        }
        commit_settled(&mut script, "Luna");
        assert!(
            !script.document().is_empty(),
            "the settled prefix must still file"
        );
        assert!(
            !texts(&script).iter().any(|s| s.contains("Luna")),
            "and must stop short of the wake word: {:?}",
            texts(&script)
        );
    }

    /// The half of the guard that is load-bearing: a wake word inside the text
    /// about to be filed would be put in the transcript as dictation,
    /// permanently, before the endpoint ever read it as an instruction.
    #[test]
    fn a_wake_word_in_the_settled_prefix_still_blocks_commitment() {
        let mut script = Transcript::new(2);
        for _ in 0..3 {
            script.push_hypothesis("Luna delete. Two. Three. Four.");
        }
        commit_settled(&mut script, "Luna");
        assert!(
            script.document().is_empty(),
            "the wake word must not be filed as dictation: {:?}",
            texts(&script)
        );
    }

    /// `keep` moves no text of its own — the merge filed it before the verb ran
    /// — so this is the whole of what it does beyond ending the settle early.
    #[test]
    fn keep_closes_the_caption_off_from_a_later_discard() {
        let mut script = Transcript::new(2);
        say(&mut script, "Worth keeping. Luna, keep. Luna, discard.");
        assert_eq!(texts(&script), vec!["Worth keeping."]);
    }

    /// One instruction, one undo entry, end to end. Two discards are two
    /// instructions, so walking them back takes two undos — which is what makes
    /// the repetition above a safe way to reach further.
    #[test]
    fn undo_puts_back_a_discarded_sentence() {
        let mut script = Transcript::new(2);
        say(&mut script, "One thought. Another thought. Luna, discard.");
        assert_eq!(texts(&script), vec!["One thought."]);

        say(&mut script, "Luna, undo.");
        assert_eq!(texts(&script), vec!["One thought.", "Another thought."]);

        let mut script = Transcript::new(2);
        say(
            &mut script,
            "One thought. Another thought. Luna, discard. Luna, discard.",
        );
        assert!(script.document().is_empty());
        say(&mut script, "Luna, undo. Luna, undo.");
        assert_eq!(texts(&script), vec!["One thought.", "Another thought."]);
    }

    /// A discard that follows a clear must take nothing and — more importantly
    /// — must not consume the undo entry the clear left, or the clear becomes
    /// unrecoverable and `clear` stops being safe to offer at all.
    #[test]
    fn a_discard_after_a_clear_leaves_the_undo_entry_alone() {
        let mut script = Transcript::new(2);
        say(&mut script, "Old one.");
        say(&mut script, "New one. Luna, clear. Luna, discard.");
        assert!(script.document().is_empty());

        say(&mut script, "Luna, undo.");
        assert_eq!(texts(&script), vec!["Old one.", "New one."]);
    }

    /// Dictation that merely names the assistant is still dictation, through
    /// the whole dispatch rather than just the parser.
    #[test]
    fn ordinary_speech_survives_the_dispatch() {
        let mut script = Transcript::new(2);
        say(&mut script, "Luna went to the store.");
        say(&mut script, "We should discard the first draft.");
        assert_eq!(
            texts(&script),
            vec!["Luna went to the store.", "We should discard the first draft."]
        );
    }
}
