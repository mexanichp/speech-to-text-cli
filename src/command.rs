//! Spoken editing commands: "Luna, commit" and "Luna, reject".
//!
//! The wake word is back, and this time it is not competing with self-repair —
//! the two solve different problems. `repair.rs` fixes a word the speaker
//! stumbled over, which needs no command because saying it again *is* the
//! signal. These are document operations — keep what I said, throw away what I
//! said — and there is no acoustic signal for those at all. Nothing short of a
//! wake word can express them, which is exactly why the wake word earns its
//! place here and did not earn it there.
//!
//! # Why detection happens on finalized utterances only
//!
//! `repair()` is safe to run on provisional text because it is pure: the same
//! false start sits in every hypothesis, so re-deriving the edit each tick is
//! idempotent. **Commands are not pure.** `Reject` destroys a sentence, and the
//! sliding window re-decodes the same audio every 500ms — running it per
//! hypothesis would delete the document one sentence per tick.
//!
//! So a command fires only when the utterance is finalized, and the audio that
//! carried it is then dropped rather than retained for continuation. Those two
//! rules together are what make a command fire exactly once. See the
//! `Pending` handling in `main.rs`.
//!
//! # The matching rule
//!
//! Wake word immediately followed by the verb, compared through
//! [`normalize`], which is what makes "Luna, commit" and "Luna commit"
//! the same input. Nothing may come between them.
//!
//! That is deliberately strict, and the asymmetry justifies it: a **missed**
//! command costs the speaker one repetition, while a **false** command deletes
//! text they meant to keep. So this holds the same bar as `repair.rs` — a
//! near-miss is dictation, never a guess — and pays for it in the direction
//! that is merely annoying rather than destructive.
//!
//! # The one thing that is not a near-miss
//!
//! A third-person `-s` on the verb is the *same* command. Observed live, a
//! spoken "Luna, reject" comes back from the model as `Luna rejects.` — the
//! speaker said the verb and the model conjugated it, so treating that as
//! dictation punishes them for a transcription they cannot see and cannot
//! influence. It is also the one failure the wake word cannot rescue, because
//! they did say the wake word.
//!
//! It does widen the false-command surface, and honestly: "Luna rejects the
//! offer." is now a reject. That is the cost of the wake word being a name,
//! and it is bounded rather than open-ended — `--assistant` retargets it,
//! `Debounce` stops a repeat compounding it, and `rollback` takes it back.
//! What was **not** widened is the past and progressive: "Luna committed" and
//! "Luna rejecting" stay dictation, because those are plausible things to say
//! *about* someone called Luna in a way the bare present tense is not.

use crate::text::{close_sentence, normalize};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Command {
    /// Keep everything settled so far.
    Commit,
    /// Drop the most recent sentence, whatever state it was in — including one
    /// already committed.
    Reject,
    /// Throw away the whole document.
    Clear,
    /// Put the kept text on the system clipboard.
    Copy,
    /// Undo the last thing that removed text.
    Rollback,
}

/// Every verb, in the order they are announced to the model.
///
/// Kept beside [`Command::exact`] rather than derived from it, with a test below
/// that fails if anything here stops parsing — the two must not drift, because a
/// verb the model is told about but the parser does not know is worse than one
/// it was never told about.
const VERBS: [&str; 6] = ["commit", "reject", "clear", "copy", "undo", "rollback"];

/// The command vocabulary as one line, for the ASR model to be told about.
///
/// # This is the exception to "the sidecar receives audio and nothing else"
///
/// §7 forbids feeding text to the model, on a measurement: the **transcript** as
/// `system_prompt` was replayed verbatim whenever a window held no speech, and
/// it self-reinforced because the echo was committed and came back in the next
/// prompt. That reasoning turns on the prompt being *dynamic and derived from
/// what the speaker said*. This one is neither, and the difference is measurable
/// rather than argued — on `Qwen3-ASR-1.7B-8bit`:
///
/// | case | no prompt | with this |
/// |---|---|---|
/// | 20s of noise at −38 dBFS | `""` | `""` — no echo |
/// | 41s of ordinary dictation | — | byte-identical |
/// | `Luna went to the store.` | dictation | dictation — no false command |
/// | spoken "Luna, rollback" | `Luna, roll back.` | `Luna rollback.` |
///
/// That last row is why it earns its place. `roll back` is two words, so the
/// parser correctly refuses it — nothing may come between the wake word and the
/// verb — and the speaker has no way to see why their command did nothing.
///
/// The invariant survives in a stronger form: this is passed **once, at spawn**,
/// from configuration. The per-window protocol still carries audio and nothing
/// else, so no code path exists by which the transcript could reach the model.
pub fn hint(wake: &str) -> String {
    let spoken: Vec<String> = VERBS.iter().map(|verb| format!("{wake} {verb}")).collect();
    format!("Commands: {}.", spoken.join(", "))
}

impl Command {
    /// The verb as spoken, or as the model chose to write it.
    ///
    /// A third-person `-s` counts as the same command, and that is a
    /// transcription concern rather than a grammar: observed live, a spoken
    /// "Luna, reject" comes back as `Luna rejects.` often enough to matter.
    /// The speaker said the verb; the model conjugated it. Refusing that
    /// reading would mean the command silently fails for reasons invisible to
    /// the person saying it — the one failure the wake word cannot help with,
    /// since they *did* say the wake word.
    ///
    /// Only the `-s` inflection, and deliberately not the rest. "Luna
    /// committed" and "Luna rejecting" stay dictation: a past or progressive
    /// form is a plausible thing to say *about* someone called Luna, where the
    /// present tense following the name directly is not — and a false command
    /// deletes text.
    fn from_verb(word: &str) -> Option<Self> {
        let word = normalize(word);
        Self::exact(&word).or_else(|| {
            // Every stem the word could be an `-s` form of, not just the first
            // rule that fires: "copies" needs the `y`, "undoes" needs the `e`
            // kept, and a verb ending in `e` would need it kept too.
            [
                word.strip_suffix("ies").map(|stem| format!("{stem}y")),
                word.strip_suffix("es").map(str::to_string),
                word.strip_suffix('s').map(str::to_string),
            ]
            .into_iter()
            .flatten()
            .find_map(|stem| Self::exact(&stem))
        })
    }

    fn exact(word: &str) -> Option<Self> {
        match word {
            "commit" => Some(Self::Commit),
            "reject" => Some(Self::Reject),
            "clear" => Some(Self::Clear),
            "copy" => Some(Self::Copy),
            // Two words for one command, which is otherwise not something this
            // module does. "rollback" is the name the operation has everywhere
            // else here; "undo" is the one people actually say out loud. Both
            // still require the wake word immediately before them, so the extra
            // surface costs nothing.
            "rollback" | "undo" => Some(Self::Rollback),
            _ => None,
        }
    }
}

/// One piece of a finalized utterance: dictation, or an instruction.
#[derive(Debug, PartialEq, Eq)]
pub enum Segment {
    Text(Vec<String>),
    Run(Command),
}

/// Split a finalized utterance into dictation and commands, in spoken order.
///
/// Order is preserved rather than commands being hoisted, because it is the
/// only reading that stays predictable when both appear in one breath:
/// "That's the wrong sentence. Luna, reject." has to file the text *before* the
/// reject reaches it, or the reject deletes the wrong thing entirely.
///
/// An empty or punctuation-only wake word never matches. `normalize` maps both
/// to `""`, and so would any word with no alphanumerics in it, so without the
/// guard `--assistant ""` would fire on every stray comma.
pub fn split(words: &[String], wake: &str) -> Vec<Segment> {
    let wake = normalize(wake);
    if wake.is_empty() {
        return vec![Segment::Text(words.to_vec())];
    }

    let mut out = Vec::new();
    let mut text: Vec<String> = Vec::new();
    let mut i = 0;

    while i < words.len() {
        let command = (normalize(&words[i]) == wake)
            .then(|| words.get(i + 1).and_then(|w| Command::from_verb(w)))
            .flatten();

        match command {
            Some(cmd) => {
                if !text.is_empty() {
                    // The speaker stopped dictating here — what came next was
                    // an instruction — so the passage ends here even if the
                    // model punctuated it as running on. It does exactly that:
                    // the merge in `main.rs` hands it one buffer holding both
                    // halves, and measured, "This is my first sentence."
                    // followed by "Luna, commit" comes back as
                    // `This is my first sentence, Luna, commit.` Removing the
                    // command then leaves the comma dangling.
                    let mut done = std::mem::take(&mut text);
                    close_sentence(&mut done);
                    if !done.is_empty() {
                        out.push(Segment::Text(done));
                    }
                }
                out.push(Segment::Run(cmd));
                i += 2;
            }
            None => {
                text.push(words[i].clone());
                i += 1;
            }
        }
    }

    if !text.is_empty() {
        out.push(Segment::Text(text));
    }
    out
}

/// How long a text-removing command suppresses a repeat of itself.
///
/// Sized from the failure it prevents. The speaker cannot see a command take
/// effect until the utterance ends, so a repetition inside roughly one reaction
/// time is the *same* intention said twice — and taking it literally deletes a
/// second sentence they never meant to lose.
///
/// Measured on synthesized speech, "Luna, reject." twice with a 700ms gap
/// between them lands the two firings ~1.7s apart, because 700ms of silence is
/// past `--endpoint-ms` and therefore two separate utterances. Three seconds
/// covers that comfortably while still letting a deliberate second reject
/// through after a beat.
const REPEAT_WINDOW: std::time::Duration = std::time::Duration::from_secs(3);

/// Suppresses a destructive command repeated before the speaker could have seen
/// the first one land.
///
/// Only `Reject` and `Clear` are guarded. `Rollback` is deliberately not:
/// saying "undo, undo, undo" is how a run of rejects gets walked back out, so
/// there the repetition *is* the intention. `Commit` and `Copy` remove nothing,
/// and repeating them is harmless.
#[derive(Default)]
pub struct Debounce {
    last: Option<(Command, std::time::Instant)>,
}

impl Debounce {
    /// Should this command run? Records it as the newest when it may.
    ///
    /// `now` is a parameter rather than read inside so the window is testable
    /// without sleeping.
    pub fn allows(&mut self, command: Command, now: std::time::Instant) -> bool {
        let guarded = matches!(command, Command::Reject | Command::Clear);

        if guarded
            && let Some((previous, at)) = self.last
            && previous == command
            && now.duration_since(at) < REPEAT_WINDOW
        {
            // Deliberately re-stamped: a burst of five stays suppressed as one
            // rather than letting every second repeat through.
            self.last = Some((command, now));
            return false;
        }

        self.last = guarded.then_some((command, now));
        true
    }
}

/// Does this utterance carry an instruction?
///
/// The caller needs this before deciding whether to retain the audio for a
/// possible continuation: an utterance holding a command must never be replayed
/// into a later window, or the command fires a second time.
pub fn has_command(words: &[String], wake: &str) -> bool {
    split(words, wake)
        .iter()
        .any(|s| matches!(s, Segment::Run(_)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    fn split_str(s: &str) -> Vec<Segment> {
        split(&words(s), "Luna")
    }

    fn text(s: &str) -> Segment {
        Segment::Text(words(s))
    }

    /// The comma is the model's business, not ours: `normalize` removes it, so
    /// both spoken forms are the same command.
    #[test]
    fn the_comma_is_optional() {
        assert_eq!(split_str("Luna, commit"), vec![Segment::Run(Command::Commit)]);
        assert_eq!(split_str("Luna commit"), vec![Segment::Run(Command::Commit)]);
        assert_eq!(split_str("Luna, reject."), vec![Segment::Run(Command::Reject)]);
        assert_eq!(split_str("LUNA COMMIT"), vec![Segment::Run(Command::Commit)]);
    }

    /// The ordinary case: a thought, a pause, then the instruction. The merge
    /// in `main.rs` hands both to this pass as one utterance.
    #[test]
    fn text_before_a_command_is_kept_and_ordered_first() {
        assert_eq!(
            split_str("This is my sentence. Luna, commit"),
            vec![text("This is my sentence."), Segment::Run(Command::Commit)]
        );
    }

    /// Order is the whole point. If the reject were hoisted ahead of the text
    /// it would delete the sentence *before* the one the speaker just finished.
    #[test]
    fn text_after_a_command_stays_after_it() {
        assert_eq!(
            split_str("Luna, reject. Let me try again."),
            vec![Segment::Run(Command::Reject), text("Let me try again.")]
        );
    }

    /// Measured end to end. The merge hands the model one buffer holding the
    /// sentence and the instruction, so it runs them together and commas the
    /// seam; removing the command must not leave that comma behind.
    #[test]
    fn the_seam_comma_the_merge_creates_is_closed() {
        assert_eq!(
            split_str("This is my first sentence, Luna, commit."),
            vec![text("This is my first sentence."), Segment::Run(Command::Commit)]
        );
    }

    /// Text *after* a command is still being spoken, so it is left alone —
    /// nothing has established that it ended.
    #[test]
    fn text_after_a_command_is_not_force_terminated() {
        assert_eq!(
            split_str("Luna, reject. and then I kept talking"),
            vec![
                Segment::Run(Command::Reject),
                text("and then I kept talking")
            ]
        );
    }

    /// Parsing stays a faithful transcription of what was said. Suppressing a
    /// repeat is `Debounce`'s job, because the repeat usually arrives in a
    /// *later* utterance — 700ms of silence is past `--endpoint-ms` — which
    /// this function never gets to see.
    #[test]
    fn parsing_reports_every_command_it_hears() {
        assert_eq!(
            split_str("Luna reject Luna reject"),
            vec![Segment::Run(Command::Reject), Segment::Run(Command::Reject)]
        );
        assert_eq!(
            split_str("Luna commit Luna copy"),
            vec![Segment::Run(Command::Commit), Segment::Run(Command::Copy)]
        );
    }

    #[test]
    fn commands_separated_by_dictation_are_distinct() {
        assert_eq!(
            split_str("Luna, reject. A new sentence. Luna, reject."),
            vec![
                Segment::Run(Command::Reject),
                text("A new sentence."),
                Segment::Run(Command::Reject)
            ]
        );
    }

    mod debounce {
        use super::*;
        use std::time::{Duration, Instant};

        /// The case this exists for: the speaker repeats because they have not
        /// seen the first one land, and the second would delete a sentence they
        /// never meant to lose.
        #[test]
        fn a_repeat_inside_the_window_is_suppressed() {
            let mut d = Debounce::default();
            let t0 = Instant::now();

            assert!(d.allows(Command::Reject, t0));
            assert!(!d.allows(Command::Reject, t0 + Duration::from_millis(1700)));
            assert!(!d.allows(Command::Reject, t0 + Duration::from_millis(2900)));
        }

        /// A burst must collapse to one, not to every other one. Each repeat
        /// re-stamps, so the window slides forward.
        #[test]
        fn a_long_burst_stays_suppressed_throughout() {
            let mut d = Debounce::default();
            let t0 = Instant::now();
            assert!(d.allows(Command::Reject, t0));

            for i in 1..=6 {
                let at = t0 + Duration::from_millis(1500 * i);
                assert!(!d.allows(Command::Reject, at), "repeat {i} got through");
            }
        }

        /// Waiting it out is how a second removal is expressed on purpose.
        #[test]
        fn a_deliberate_repeat_after_the_window_runs() {
            let mut d = Debounce::default();
            let t0 = Instant::now();
            assert!(d.allows(Command::Reject, t0));
            assert!(d.allows(Command::Reject, t0 + Duration::from_millis(3100)));
        }

        /// Different commands never mask each other, in either order.
        #[test]
        fn an_unrelated_command_in_between_is_not_blocked() {
            let mut d = Debounce::default();
            let t0 = Instant::now();
            let soon = t0 + Duration::from_millis(200);

            assert!(d.allows(Command::Reject, t0));
            assert!(d.allows(Command::Clear, soon), "a clear is not a reject");
            assert!(
                d.allows(Command::Reject, soon + Duration::from_millis(200)),
                "and the clear displaced the reject rather than extending it"
            );
        }

        /// Repeating "undo" is how a run of rejects is walked back out, so the
        /// repetition there *is* the intention.
        #[test]
        fn harmless_and_intentionally_repeatable_commands_are_never_guarded() {
            for command in [Command::Rollback, Command::Commit, Command::Copy] {
                let mut d = Debounce::default();
                let t0 = Instant::now();
                assert!(d.allows(command, t0), "{command:?}");
                assert!(d.allows(command, t0), "{command:?} must repeat freely");
                assert!(d.allows(command, t0), "{command:?} must repeat freely");
            }
        }

        /// An unguarded command must not leave a stale entry that a later
        /// reject is then measured against.
        #[test]
        fn an_unguarded_command_clears_the_guard() {
            let mut d = Debounce::default();
            let t0 = Instant::now();

            assert!(d.allows(Command::Reject, t0));
            assert!(d.allows(Command::Commit, t0));
            assert!(
                d.allows(Command::Reject, t0),
                "the commit ended the reject's window"
            );
        }
    }

    #[test]
    fn every_verb_is_recognised() {
        for (spoken, want) in [
            ("Luna, commit", Command::Commit),
            ("Luna, reject", Command::Reject),
            ("Luna, clear", Command::Clear),
            ("Luna, copy", Command::Copy),
            ("Luna, rollback", Command::Rollback),
            ("Luna, undo", Command::Rollback),
        ] {
            assert_eq!(split_str(spoken), vec![Segment::Run(want)], "{spoken:?}");
        }
    }

    /// Observed live: the model writes a spoken "Luna, reject" as
    /// `Luna rejects.` The speaker said the verb, so the command has to fire —
    /// and every verb takes the same treatment, since there is nothing special
    /// about `reject` except that it is the one caught doing it.
    #[test]
    fn a_conjugated_verb_is_the_same_command() {
        for (spoken, want) in [
            ("Luna, commits", Command::Commit),
            ("Luna rejects", Command::Reject),
            ("Luna, clears", Command::Clear),
            // Both the spelling the model would choose and the one it might.
            ("Luna, copies", Command::Copy),
            ("Luna, copys", Command::Copy),
            ("Luna, rollbacks", Command::Rollback),
            ("Luna, undoes", Command::Rollback),
            ("Luna, undos", Command::Rollback),
        ] {
            assert_eq!(split_str(spoken), vec![Segment::Run(want)], "{spoken:?}");
        }
    }

    /// The allowance stops at `-s`. A past or progressive form is a plausible
    /// thing to say about a person called Luna, and a false command deletes
    /// text — so the line is drawn where the tense stops being one a speaker
    /// giving an instruction would ever use.
    #[test]
    fn other_conjugations_are_still_dictation() {
        for line in [
            "Luna committed",
            "Luna rejecting",
            "Luna cleared",
            "Luna copied",
            "Luna undone",
        ] {
            assert_eq!(split_str(line), vec![text(line)], "must be left alone: {line:?}");
        }
    }

    /// The hint and the parser must not drift: a verb the model is told about
    /// but the parser cannot read is worse than one it was never told about,
    /// because the speaker then says a command that is heard and ignored.
    #[test]
    fn every_announced_verb_is_one_the_parser_accepts() {
        for verb in VERBS {
            assert_eq!(
                split_str(&format!("Luna {verb}")),
                vec![Segment::Run(Command::from_verb(verb).expect("announced"))],
                "announced but not parsed: {verb:?}"
            );
        }
    }

    /// It is announced to the model verbatim, so it has to read as the words a
    /// speaker would actually say — and it has to follow `--assistant`.
    #[test]
    fn the_hint_names_the_configured_wake_word() {
        assert_eq!(
            hint("Luna"),
            "Commands: Luna commit, Luna reject, Luna clear, Luna copy, \
             Luna undo, Luna rollback."
        );
        assert!(hint("Jarvis").starts_with("Commands: Jarvis commit,"));
        assert!(!hint("Jarvis").contains("Luna"));
    }

    /// The wake word is configurable, so nothing may be hard-coded to "luna".
    #[test]
    fn the_wake_word_is_configurable() {
        let w = words("Jarvis, commit");
        assert_eq!(split(&w, "Jarvis"), vec![Segment::Run(Command::Commit)]);
        // And the default no longer fires on it.
        assert_eq!(split(&w, "Luna"), vec![Segment::Text(w.clone())]);
    }

    /// A near-miss is dictation. Deleting a sentence the speaker meant to keep
    /// is far worse than making them repeat the command.
    #[test]
    fn ordinary_speech_is_not_a_command() {
        for line in [
            // The wake word alone, as a name.
            "Luna went to the store.",
            // The verb alone, which is an ordinary English word.
            "I will commit the change tomorrow.",
            "The reviewers reject bad patches.",
            // Right words, wrong order.
            "Commit Luna",
            // Something between them: nothing may come between the two.
            "Luna please commit",
            "Luna, I want you to commit",
            // A verb that is merely similar.
            "Luna committed",
            "Luna rejecting",
        ] {
            assert_eq!(
                split_str(line),
                vec![text(line)],
                "must be left alone: {line:?}"
            );
        }
    }

    /// The wake word must survive as dictation when it is not driving a verb,
    /// including immediately before text that merely starts with one.
    #[test]
    fn a_bare_wake_word_is_kept_as_text() {
        assert_eq!(split_str("Luna"), vec![text("Luna")]);
        assert_eq!(
            split_str("I told Luna about it."),
            vec![text("I told Luna about it.")]
        );
    }

    /// `normalize` maps an empty wake word and any punctuation-only token to
    /// the same empty string, so without the guard every stray comma in the
    /// transcript would be read as the wake word.
    #[test]
    fn an_empty_wake_word_never_matches() {
        let w = words(", commit and , reject");
        assert_eq!(split(&w, ""), vec![Segment::Text(w.clone())]);
        assert_eq!(split(&w, "..."), vec![Segment::Text(w)]);
    }

    #[test]
    fn empty_input_produces_no_segments() {
        assert!(split(&[], "Luna").is_empty());
    }

    /// What `main.rs` uses to decide whether the audio may be retained for a
    /// continuation. A retained command would fire a second time.
    #[test]
    fn has_command_sees_what_split_sees() {
        assert!(has_command(&words("Luna, commit"), "Luna"));
        assert!(has_command(&words("Some text. Luna, reject"), "Luna"));
        assert!(!has_command(&words("Luna went home."), "Luna"));
        assert!(!has_command(&[], "Luna"));
    }
}
