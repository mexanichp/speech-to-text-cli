//! Voice activity detection with hysteresis.
//!
//! Two jobs, deliberately kept distinct:
//!   * **gating** — never invoke the model on silence. AED models hallucinate
//!     confident boilerplate on near-empty audio.
//!   * **endpointing** — signal that an utterance is complete, which is what
//!     lets the transcript commit its tail.
//!
//! # Duration is the defence against a keyboard, not level
//!
//! A keyboard is the loudest thing in an otherwise quiet room and the hardest
//! noise for a level gate to reject: a keystroke is broadband, clears any floor
//! set below a speaking voice, and the neural detector is perfectly willing to
//! call it speech. What it is *not* is sustained — a keystroke is one or two
//! frames where the shortest syllable is a dozen. So both hysteresis counters
//! are duration tests, and both of them matter here:
//!
//!   * `open_frames` decides whether a burst becomes an utterance at all.
//!   * The **silence run is cancelled only by sustained voicing**, so typing
//!     through a pause cannot keep postponing the endpoint that would have
//!     committed the sentence.

use earshot::Detector;

/// earshot requires exactly this frame size at 16 kHz (16 ms).
pub const FRAME: usize = 256;

const SPEECH_THRESHOLD: f32 = 0.5;

/// Default silence floor, in dBFS RMS.
///
/// The neural VAD alone fires on ambient room noise, and an AED model handed
/// near-silence emits confident boilerplate ("Okay.", "The."). So this floor is
/// belt-and-braces with the detector rather than redundant.
///
/// **It is a floor, not a speech detector, and the default is not universal.**
/// It rejects audio that is *quiet*, not audio that is *non-speech*. A room
/// that idles above this — measured here at −38 dBFS, which is an unremarkable
/// desk fan — clears it on every frame, the detector then fires on the noise,
/// and the model dutifully returns "Oh." Reproduced from noise alone. That is
/// what `--rms-floor` is for: raise it above the room, below your voice.
pub const DEFAULT_FLOOR_DBFS: f32 = -40.0;

/// Default sustained-speech requirement, in milliseconds.
///
/// Measured against `earshot` rather than guessed. A 32 ms burst of energy —
/// one keystroke — comes back as **four consecutive voiced frames**, because
/// the detector lags two frames past the end of the signal. Longer bursts scale
/// the same way: 64 ms gives 6 frames, 96 ms gives 8, and only at ~128 ms of
/// genuinely continuous energy does it reach 11.
///
/// 150 ms is 9 frames: above an isolated keystroke, below any syllable. The old
/// 48 ms was 3 frames, which one keystroke cleared with a frame to spare.
///
/// **Know what this does not cover**, in the same spirit as the floor above.
/// It gates *isolated* transients — which is the case that produces text,
/// because a lone keystroke opens a window holding a fraction of a second of
/// near-silence, and that is precisely what an AED model answers with "Okay."
/// It does **not** gate sustained fast typing: measured on synthesized key
/// noise at ~7 keystrokes a second, the detector chains across the gaps and
/// returns runs of 23 to 51 consecutive voiced frames, far past any sane open
/// window. In that regime the backstop is the ASR itself, which was measured
/// returning nothing at all for 20 s of it.
///
/// Onset latency is not the cost it looks like: the window keeps a 300 ms
/// pre-roll, so the speech that opened the utterance is still in the buffer.
/// Raising this past that pre-roll would start clipping first words.
pub const DEFAULT_OPEN_MS: u32 = 150;

/// dBFS is the unit a room is measured in; the comparison wants an amplitude.
fn floor_from_dbfs(dbfs: f32) -> f32 {
    10f32.powf(dbfs / 20.0)
}

pub struct Vad {
    detector: Detector,
    rms_floor: f32,
    /// Consecutive voiced frames before energy counts as speech at all: what
    /// opens an utterance, and what it takes to cancel a silence run.
    open_frames: usize,
    /// Consecutive silence frames needed to close one.
    close_frames: usize,
    speech_run: usize,
    silence_run: usize,
    active: bool,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum VadEvent {
    /// Speech began.
    Onset,
    /// Speech is ongoing.
    Speaking,
    /// Enough silence has elapsed: the utterance is complete.
    Endpoint,
    /// Silence, outside an utterance.
    Idle,
}

impl Vad {
    /// `open_ms` is how long energy must last before it counts as speech —
    /// the transient gate, see the module docs. `floor_dbfs` is the silence
    /// floor; see [`DEFAULT_FLOOR_DBFS`].
    pub fn new(open_ms: u32, close_ms: u32, floor_dbfs: f32) -> Self {
        Self::with_floor(open_ms, close_ms, floor_from_dbfs(floor_dbfs))
    }

    fn with_floor(open_ms: u32, close_ms: u32, rms_floor: f32) -> Self {
        let frame_ms = (FRAME as f32 / 16.0) as u32; // 16 ms
        Self {
            detector: Detector::default(),
            rms_floor,
            open_frames: (open_ms / frame_ms).max(1) as usize,
            close_frames: (close_ms / frame_ms).max(1) as usize,
            speech_run: 0,
            silence_run: 0,
            active: false,
        }
    }

    /// Feed exactly [`FRAME`] samples of 16 kHz mono f32.
    pub fn push(&mut self, frame: &[f32]) -> VadEvent {
        debug_assert_eq!(frame.len(), FRAME);

        let voiced =
            rms(frame) >= self.rms_floor && self.detector.predict_f32(frame) >= SPEECH_THRESHOLD;

        if voiced {
            self.speech_run += 1;
            // Sustained voicing, and only that, cancels accumulated silence.
            //
            // A keyboard click is one or two frames of broadband energy that
            // clears the floor easily and that the detector calls speech. If a
            // single such frame reset the run, typing through a thinking pause
            // would postpone the endpoint indefinitely — and the endpoint is
            // what commits the tail of the sentence and closes the buffer. The
            // same threshold that decides an utterance may *open* decides that
            // one may *continue*, which is the only consistent reading of it.
            if self.speech_run >= self.open_frames {
                self.silence_run = 0;
            }
        } else {
            self.silence_run += 1;
            self.speech_run = 0;
        }

        if self.active {
            if self.silence_run >= self.close_frames {
                self.active = false;
                VadEvent::Endpoint
            } else {
                VadEvent::Speaking
            }
        } else if self.speech_run >= self.open_frames {
            self.active = true;
            VadEvent::Onset
        } else {
            VadEvent::Idle
        }
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Consecutive silent frames seen so far.
    ///
    /// Exposed for the buffer trim in `main.rs`, which needs the *short*
    /// silences an endpoint deliberately ignores. A long utterance still has
    /// gaps in it — between sentences, around a breath — and they are the only
    /// places the audio can be cut without slicing a word in half.
    pub fn silence_run(&self) -> usize {
        self.silence_run
    }
}

fn rms(frame: &[f32]) -> f32 {
    (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(vad: &mut Vad, frames: usize, amplitude: f32) -> Vec<VadEvent> {
        (0..frames)
            .map(|i| {
                // 200 Hz tone: loud enough to register as periodic energy.
                let frame: Vec<f32> = (0..FRAME)
                    .map(|n| {
                        let t = (i * FRAME + n) as f32 / 16_000.0;
                        amplitude * (2.0 * std::f32::consts::PI * 200.0 * t).sin()
                    })
                    .collect();
                vad.push(&frame)
            })
            .collect()
    }

    /// A microphone in a room that is not digitally silent.
    ///
    /// **The room tone is the point.** `earshot` is stateful and adapts to what
    /// it is shown, and digital silence is a signal no microphone produces: fed
    /// it, the RMS floor rejects every frame either side of a keystroke and
    /// nothing gets through at any setting — which is not the situation being
    /// reproduced. A real room idles *above* −40 dBFS (§7: a desk fan does), so
    /// the floor is cleared continuously and the detector alone decides. That
    /// is the case where duration is the only defence left.
    ///
    /// Phase and noise are continuous and fixed-seed across the whole run,
    /// because the detector remembers what it has been shown.
    struct Room {
        vad: Vad,
        phase: usize,
        seed: u32,
    }

    impl Room {
        fn new(open_ms: u32) -> Self {
            Self {
                vad: Vad::new(open_ms, 600, DEFAULT_FLOOR_DBFS),
                phase: 0,
                seed: 12_345,
            }
        }

        /// A 200 Hz tone at `amplitude`, riding on the room tone. 0.0 is the
        /// room alone — quiet, but never digitally silent.
        fn feed(&mut self, frames: usize, amplitude: f32) -> Vec<VadEvent> {
            let mut events = Vec::with_capacity(frames);
            for _ in 0..frames {
                let mut frame = Vec::with_capacity(FRAME);
                for n in 0..FRAME {
                    self.seed = self.seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let room = (self.seed >> 8) as f32 / 8_388_608.0 - 1.0;
                    let t = (self.phase * FRAME + n) as f32 / 16_000.0;
                    frame.push(
                        0.03 * room
                            + amplitude * (2.0 * std::f32::consts::PI * 200.0 * t).sin(),
                    );
                }
                self.phase += 1;
                events.push(self.vad.push(&frame));
            }
            events
        }

        fn voice(&mut self, frames: usize) -> Vec<VadEvent> {
            self.feed(frames, 0.3)
        }

        /// One keystroke: 32 ms of energy, loud against the room.
        ///
        /// Measured against `earshot`, a burst this short comes back as a run
        /// of **3 to 6 voiced frames** — the detector lags a frame or two past
        /// the end of the energy. Every one of those clears a 48 ms (3-frame)
        /// open window and none clears a 150 ms (9-frame) one, which is the
        /// entire content of the fix.
        fn keystroke(&mut self) -> Vec<VadEvent> {
            self.feed(2, 0.3)
        }

        /// Typing at an ordinary rate: a keystroke every ~160 ms.
        fn typing(&mut self) -> Vec<VadEvent> {
            let mut events = Vec::new();
            for _ in 0..15 {
                events.extend(self.keystroke());
                events.extend(self.feed(8, 0.0));
            }
            events
        }
    }

    #[test]
    fn quiet_input_never_opens_an_utterance() {
        let mut vad = Vad::new(48, 600, DEFAULT_FLOOR_DBFS);
        // Below the floor: ambient noise must not trigger inference.
        let events = feed(&mut vad, 100, 0.001);
        assert!(events.iter().all(|e| *e == VadEvent::Idle));
        assert!(!vad.is_active());
    }

    /// dBFS is what a room is measured in; the detector wants amplitude.
    #[test]
    fn the_floor_converts_from_dbfs() {
        assert!((floor_from_dbfs(-40.0) - 0.01).abs() < 1e-6);
        assert!((floor_from_dbfs(0.0) - 1.0).abs() < 1e-6);
        assert!(floor_from_dbfs(-30.0) > floor_from_dbfs(-40.0));
    }

    /// The reason `--rms-floor` exists. A room that idles above the floor
    /// clears the gate on every frame, the detector then fires on it, and the
    /// model transcribes it — reproduced end to end from noise alone, as "Oh."
    ///
    /// The floor's contract is that it overrides the detector on level alone,
    /// so raising it above the room silences that whatever the detector thinks.
    /// The signal here is one the detector *does* accept, which is the only
    /// case where the floor is what decides.
    #[test]
    fn raising_the_floor_gates_out_audio_the_detector_accepts() {
        // Amplitude 0.3 is ~0.21 RMS (−13.5 dBFS); the detector opens on it.
        let mut lenient = Vad::new(48, 600, DEFAULT_FLOOR_DBFS);
        assert!(
            feed(&mut lenient, 100, 0.3).contains(&VadEvent::Onset),
            "precondition: the detector accepts this signal"
        );

        // A floor above that level must reject it regardless.
        let mut strict = Vad::new(48, 600, -10.0);
        assert!(
            feed(&mut strict, 100, 0.3)
                .iter()
                .all(|e| *e == VadEvent::Idle),
            "the floor must override the detector on level alone"
        );
        assert!(!strict.is_active());
    }

    /// The reported failure, at the frame level: typing while the mic is open
    /// produced text.
    ///
    /// A keystroke is loud — it clears any floor set below a speaking voice —
    /// so level cannot reject it and only duration can. The two windows here
    /// are run on byte-identical audio, which is what makes this a test of the
    /// threshold rather than of the detector.
    #[test]
    fn typing_opens_a_short_window_and_not_the_default_one() {
        let mut lenient = Room::new(48);
        assert!(
            lenient.typing().contains(&VadEvent::Onset),
            "precondition: 48 ms is what let a keystroke through"
        );

        let mut room = Room::new(DEFAULT_OPEN_MS);
        assert!(
            room.typing().iter().all(|e| *e == VadEvent::Idle),
            "typing must not open an utterance at the default window"
        );
        assert!(!room.vad.is_active());

        // And a voice still gets through the wider window, in the same room.
        assert!(
            room.voice(60).contains(&VadEvent::Onset),
            "sustained speech must still open an utterance"
        );
    }

    /// Typing through a thinking pause used to reset the silence run on every
    /// keystroke, so the endpoint that commits the sentence never arrived.
    /// Cancelling silence now takes as much sustained voicing as opening an
    /// utterance does.
    #[test]
    fn a_keystroke_does_not_cancel_accumulated_silence() {
        let mut room = Room::new(DEFAULT_OPEN_MS);
        room.voice(40);
        assert!(room.vad.is_active(), "precondition: an utterance is open");

        room.feed(12, 0.0);
        let accrued = room.vad.silence_run();
        assert!(accrued > 0, "precondition: silence is accruing to an endpoint");

        // One keystroke, mid-pause. The detector does call it speech, so this
        // passes only if the run was *held* — were the frames rejected outright
        // the run would have grown instead.
        room.keystroke();
        assert_eq!(
            room.vad.silence_run(),
            accrued,
            "a keystroke must not put the endpoint 600 ms back"
        );

        // Speech resuming still clears it. Not on the first frame — the
        // detector takes a moment to re-acquire — which costs nothing, because
        // silence only ever accrues on a genuinely silent frame.
        room.voice(25);
        assert_eq!(room.vad.silence_run(), 0, "sustained speech must cancel it");
    }

    #[test]
    fn silence_after_speech_endpoints() {
        let mut vad = Vad::new(48, 100, DEFAULT_FLOOR_DBFS);
        feed(&mut vad, 40, 0.3);
        let quiet = feed(&mut vad, 40, 0.0);
        assert!(
            quiet.contains(&VadEvent::Endpoint),
            "expected an endpoint once silence exceeded the close window"
        );
    }

    /// The buffer trim cuts at short silences the endpoint ignores, so the run
    /// has to be observable while the utterance is still open.
    #[test]
    fn the_silence_run_is_visible_before_the_endpoint() {
        // 600 ms to close, so a handful of silent frames is still "Speaking".
        let mut vad = Vad::new(48, 600, DEFAULT_FLOOR_DBFS);
        feed(&mut vad, 40, 0.3);
        assert_eq!(vad.silence_run(), 0, "speech resets the run");

        let events = feed(&mut vad, 10, 0.0);
        assert!(
            events.iter().all(|e| *e == VadEvent::Speaking),
            "10 frames is 160 ms — nowhere near the 600 ms close window"
        );
        assert_eq!(vad.silence_run(), 10, "and yet the gap is measurable");
        assert!(vad.is_active(), "the utterance is still open");

        // Speech resuming clears it, so each gap is counted separately.
        feed(&mut vad, 5, 0.3);
        assert_eq!(vad.silence_run(), 0);
    }
}




