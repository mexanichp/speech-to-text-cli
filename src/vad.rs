//! Voice activity detection with hysteresis.
//!
//! Serves two distinct purposes:
//!
//! - Gating, so the recogniser is never invoked on silence. Attention-based
//!   models emit confident boilerplate when handed near-empty audio.
//! - Endpointing, which signals that an utterance is complete and lets the
//!   transcript commit its tail.
//!
//! # Transient rejection
//!
//! Level cannot distinguish a keystroke from speech: a keystroke is broadband
//! and louder than the room, so any floor that rejects one also rejects a quiet
//! voice. Duration can, since a keystroke spans a few frames where the shortest
//! syllable spans dozens. Both hysteresis counters are therefore duration
//! tests, and the silence run is cancelled only by sustained voicing, so typing
//! through a pause cannot postpone the endpoint indefinitely.
//!
//! # Two silence counters
//!
//! [`Vad::silence_run`] accrues toward an endpoint and deliberately survives a
//! brief voiced burst, so the span it measures may contain speech.
//! [`Vad::quiet_run`] is reset by any voiced frame at all and is the one the
//! buffer trim must use, since it needs spans that are genuinely empty.

use earshot::Detector;

/// Frame size the detector requires, in samples at 16 kHz.
pub const FRAME: usize = 256;

/// Detector confidence above which a frame counts as voiced.
const SPEECH_THRESHOLD: f32 = 0.5;

/// Samples per millisecond at the 16 kHz contract.
const SAMPLES_PER_MS: usize = 16;

/// Duration of one [`FRAME`], in milliseconds.
const FRAME_MS: u32 = (FRAME / SAMPLES_PER_MS) as u32;

/// Default silence floor, in dBFS RMS.
///
/// Combined with the detector rather than redundant to it: the detector alone
/// fires on ambient room noise, and the recogniser then transcribes it.
///
/// This rejects audio that is quiet, not audio that is non-speech, so it is not
/// universal. A room idling above it clears the gate on every frame. The
/// `--rms-floor` option exists to place it above the room and below the
/// speaker's voice.
pub const DEFAULT_FLOOR_DBFS: f32 = -40.0;

/// Default sustained-speech requirement, in milliseconds.
///
/// Nine frames: above an isolated keystroke, which yields four to six voiced
/// frames because the detector lags the signal, and below any syllable.
///
/// Gates isolated transients only. Sustained typing reads as continuous energy
/// and is not rejected by any duration threshold; there the recogniser itself
/// declines the audio.
///
/// Raising this does not lose speech while it stays within the caller's
/// pre-roll, since the audio that opened the utterance is still buffered.
/// Beyond the pre-roll it would clip first words.
pub const DEFAULT_OPEN_MS: u32 = 150;

/// Converts a dBFS level to the linear amplitude the RMS comparison uses.
fn floor_from_dbfs(dbfs: f32) -> f32 {
    10f32.powf(dbfs / 20.0)
}

/// Frame-by-frame speech detector with open and close hysteresis.
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
    /// Consecutive genuinely quiet frames, reset by any voiced frame.
    ///
    /// Separate from `silence_run` because the two consumers need opposite
    /// things: endpointing asks whether the speaker has stopped, ignoring
    /// clicks, while the buffer trim asks whether a span is empty enough to cut
    /// without slicing a word. A keystroke inside a pause splits this into two
    /// shorter spans rather than yielding one long span running through it.
    ///
    /// Not a pure level test. A frame counts as voiced only when it clears the
    /// floor *and* the detector agrees, so quiet spans include audio that is
    /// loud but unrecognised. Gating on level alone would be provable but
    /// yields no cut points at all in a room idling above the floor.
    quiet_run: usize,
    active: bool,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
/// What one frame revealed about the utterance in progress.
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
    /// Creates a detector.
    ///
    /// # Parameters
    ///
    /// - `open_ms`: sustained energy required before audio counts as speech.
    /// - `close_ms`: silence required to end an utterance.
    /// - `floor_dbfs`: level below which a frame is never voiced; see
    ///   [`DEFAULT_FLOOR_DBFS`].
    ///
    /// Each duration is rounded down to whole frames, with a floor of one.
    pub fn new(open_ms: u32, close_ms: u32, floor_dbfs: f32) -> Self {
        Self::with_floor(open_ms, close_ms, floor_from_dbfs(floor_dbfs))
    }

    /// Creates a detector from a linear amplitude floor rather than dBFS.
    fn with_floor(open_ms: u32, close_ms: u32, rms_floor: f32) -> Self {
        let frame_ms = FRAME_MS;
        Self {
            detector: Detector::default(),
            rms_floor,
            open_frames: (open_ms / frame_ms).max(1) as usize,
            close_frames: (close_ms / frame_ms).max(1) as usize,
            speech_run: 0,
            silence_run: 0,
            quiet_run: 0,
            active: false,
        }
    }

    /// Advances the detector by one frame.
    ///
    /// # Panics
    ///
    /// Debug builds assert that `frame` is exactly [`FRAME`] samples of 16 kHz
    /// mono f32.
    pub fn push(&mut self, frame: &[f32]) -> VadEvent {
        debug_assert_eq!(frame.len(), FRAME);

        let voiced =
            rms(frame) >= self.rms_floor && self.detector.predict_f32(frame) >= SPEECH_THRESHOLD;

        if voiced {
            self.speech_run += 1;
            self.quiet_run = 0;
            if self.speech_run >= self.open_frames {
                self.silence_run = 0;
            }
        } else {
            self.silence_run += 1;
            self.quiet_run += 1;
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

    /// Reports whether an utterance is currently open.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Silent frames accrued toward an endpoint.
    ///
    /// A voiced burst shorter than the open threshold holds this steady rather
    /// than clearing it, so the span it covers may contain speech. Use
    /// [`Vad::quiet_run`] where the audio must be genuinely empty.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn silence_run(&self) -> usize {
        self.silence_run
    }

    /// Consecutive frames of genuinely quiet audio.
    ///
    /// Reset by a single voiced frame, so the span reported contains no speech.
    /// These are the only offsets at which the buffer may be cut without
    /// slicing a word.
    pub fn quiet_run(&self) -> usize {
        self.quiet_run
    }
}

/// Root-mean-square amplitude of one frame.
fn rms(frame: &[f32]) -> f32 {
    (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(vad: &mut Vad, frames: usize, amplitude: f32) -> Vec<VadEvent> {
        (0..frames)
            .map(|i| {
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
    /// reproduced. A real room idles above the default floor, so the floor is
    /// cleared continuously and the detector alone decides. That is the case
    /// where duration is the only defence left.
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

        /// Synthesizes one keystroke: 32 ms of energy, loud against the room.
        ///
        /// A burst this short yields three to six voiced frames, because the
        /// detector lags the signal. All clear a three-frame open window and
        /// none clears a nine-frame one.
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

    /// The floor overrides the detector on level alone, which is why raising it
    /// above a room the detector accepts silences that room. Uses a signal the
    /// detector accepts, since that is the only case where the floor decides.
    #[test]
    fn raising_the_floor_gates_out_audio_the_detector_accepts() {
        let mut lenient = Vad::new(48, 600, DEFAULT_FLOOR_DBFS);
        assert!(
            feed(&mut lenient, 100, 0.3).contains(&VadEvent::Onset),
            "precondition: the detector accepts this signal"
        );

        let mut strict = Vad::new(48, 600, -10.0);
        assert!(
            feed(&mut strict, 100, 0.3)
                .iter()
                .all(|e| *e == VadEvent::Idle),
            "the floor must override the detector on level alone"
        );
        assert!(!strict.is_active());
    }

    /// Typing with the microphone open must not open an utterance. Both windows
    /// run on byte-identical audio, so this tests the threshold rather than the
    /// detector.
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

        assert!(
            room.voice(60).contains(&VadEvent::Onset),
            "sustained speech must still open an utterance"
        );
    }

    /// Cancelling accumulated silence requires as much sustained voicing as
    /// opening an utterance, so typing through a pause cannot postpone the
    /// endpoint indefinitely.
    #[test]
    fn a_keystroke_does_not_cancel_accumulated_silence() {
        let mut room = Room::new(DEFAULT_OPEN_MS);
        room.voice(40);
        assert!(room.vad.is_active(), "precondition: an utterance is open");

        room.feed(12, 0.0);
        let accrued = room.vad.silence_run();
        assert!(accrued > 0, "precondition: silence is accruing to an endpoint");

        room.keystroke();
        assert_eq!(
            room.vad.silence_run(),
            accrued,
            "a keystroke must not put the endpoint 600 ms back"
        );

        room.voice(25);
        assert_eq!(room.vad.silence_run(), 0, "sustained speech must cancel it");
    }

    /// The two silence counters must disagree here. A keystroke ends a quiet run
    /// but not a silence run, which is why only the quiet run may be used to
    /// choose a cut point.
    #[test]
    fn a_keystroke_ends_a_quiet_run_even_though_it_does_not_end_a_silence_run() {
        let mut room = Room::new(DEFAULT_OPEN_MS);
        room.voice(40);
        assert!(room.vad.is_active(), "precondition: an utterance is open");

        room.feed(12, 0.0);
        let accrued = room.vad.silence_run();
        assert!(accrued > 0, "precondition: silence is accruing");
        assert_eq!(room.vad.quiet_run(), accrued, "with no voicing the two agree");

        room.keystroke();
        assert_eq!(
            room.vad.silence_run(),
            accrued,
            "the endpoint must not be postponed by a click"
        );
        assert_eq!(
            room.vad.quiet_run(),
            0,
            "but the audio is not empty here, so it is not a place to cut"
        );

        room.feed(5, 0.0);
        let after = room.vad.quiet_run();
        assert!(
            (1..=5).contains(&after),
            "the quiet run restarts at the click, not before it: {after}"
        );
        assert!(
            room.vad.silence_run() > after,
            "while the endpoint still counts the whole pause"
        );
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

        feed(&mut vad, 5, 0.3);
        assert_eq!(vad.silence_run(), 0);
    }
}
