//! Parsing of spoken editing commands from a finalized utterance.
//!
//! A command is the wake word immediately followed by a verb. Nothing may come
//! between them, and matching is done on normalized forms, so the comma in
//! "Luna, delete" is punctuation the recogniser chose and is never seen here.
//!
//! # Command scopes
//!
//! `delete` and `discard` both remove one sentence and differ only in reach.
//! `delete` takes the newest sentence in the document, however long ago it was
//! filed. `discard` takes the newest sentence of the utterance in flight and
//! can reach nothing older; said with nothing in flight it removes nothing
//! rather than falling through to the transcript. That bound is what makes
//! `discard` safe to say without looking at the screen.
//!
//! # Detection is restricted to finalized utterances
//!
//! [`crate::repair`] may run on provisional text because it is pure. Commands
//! are not: they destroy text, and the sliding window re-decodes the same audio
//! every tick, so per-hypothesis detection would empty the document one
//! sentence per tick.
//!
//! Two rules together make a command fire exactly once. It is parsed only from
//! a finalized utterance, and the audio carrying it is dropped rather than
//! retained for continuation, since retention is the only thing that could
//! replay it.
//!
//! # Matching bias
//!
//! Strictness is asymmetric by design: a missed command costs one repetition,
//! whereas a false command destroys text the speaker meant to keep. A near-miss
//! is therefore treated as dictation.
//!
//! The single exception is a third-person `-s` on the verb, which is the same
//! command. The recogniser conjugates a spoken "Luna, delete" into "Luna
//! deletes" often enough to matter, and rejecting that would punish the speaker
//! for a transcription they can neither see nor influence. Past and progressive
//! forms remain dictation, since those are plausible things to say about a
//! person.

use crate::text::{close_sentence, normalize};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
/// An instruction parsed from a finalized utterance.
pub enum Command {
    /// Drop the most recent sentence in the document.
    Delete,
    /// Drop the newest sentence of the utterance in flight. Cannot reach past
    /// the current breath; with nothing in flight it removes nothing.
    Discard,
    /// File what this utterance said and end the settle now.
    Keep,
    /// Throw away the whole document.
    Clear,
    /// Put the transcript on the system clipboard.
    Copy,
    /// Undo the last thing that removed text.
    Rollback,
}

/// Every verb, in the order they are announced to the recogniser.
///
/// Maintained beside [`Command::exact`] rather than derived from it, with a
/// test that fails if any entry stops parsing. A verb the recogniser is told
/// about but the parser does not know is worse than one it never heard of.
///
/// A short list is safer twice over: each entry is a phrase the recogniser is
/// primed with before it hears anything, and each is an ordinary English word
/// that a false match could destroy text with. Every verb here changes the
/// transcript, which is the requirement for inclusion.
const VERBS: [&str; 7] = [
    "delete", "discard", "keep", "clear", "copy", "undo", "rollback",
];

/// Formats the command vocabulary as a single hint line for the recogniser.
///
/// Passed once at spawn, never per request. The distinction matters: a prompt
/// derived from what the speaker said is replayed verbatim on any window
/// holding no speech, and self-reinforces as the echo is committed and returns
/// in the next prompt. This prompt is fixed at startup, cannot grow, and cannot
/// contain dictation.
///
/// It earns its place by fixing verbs the recogniser otherwise splits, such as
/// "rollback" arriving as two words, which the parser correctly refuses with no
/// way for the speaker to see why.
pub fn hint(wake: &str) -> String {
    let spoken: Vec<String> = VERBS.iter().map(|verb| format!("{wake} {verb}")).collect();
    format!("Commands: {}.", spoken.join(", "))
}

impl Command {
    /// Parses a verb as spoken, or as the recogniser chose to write it.
    ///
    /// Accepts the third-person `-s` form, which is a transcription artefact
    /// rather than a different word: the speaker said the verb and the
    /// recogniser conjugated it. Past and progressive forms are rejected, since
    /// those are plausible things to say about a person bearing the wake word
    /// as a name.
    fn from_verb(word: &str) -> Option<Self> {
        let word = normalize(word);
        Self::exact(&word).or_else(|| {
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

    /// Parses a verb in its exact spoken form.
    fn exact(word: &str) -> Option<Self> {
        match word {
            "delete" => Some(Self::Delete),
            "discard" => Some(Self::Discard),
            "keep" => Some(Self::Keep),
            "clear" => Some(Self::Clear),
            "copy" => Some(Self::Copy),
            "rollback" | "undo" => Some(Self::Rollback),
            _ => None,
        }
    }
}

/// One piece of a finalized utterance: dictation, or an instruction.
#[derive(Debug, PartialEq, Eq)]
/// One piece of a finalized utterance: dictation, or an instruction.
pub enum Segment {
    Text(Vec<String>),
    Run(Command),
}

/// Splits a finalized utterance into dictation and commands, in spoken order.
///
/// Order is preserved rather than commands being hoisted, because that is the
/// only reading that stays predictable when both appear in one breath: text
/// preceding a `delete` must be filed before the delete reaches it.
///
/// Dictation terminated by a command is closed with [`close_sentence`], since
/// the words that followed were provably not part of it. Text appearing after a
/// command is left alone, as nothing establishes that it ended.
///
/// A wake word with no alphanumerics never matches, which would otherwise fire
/// on every stray punctuation token.
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

/// The wake word in the form the parser compares against.
///
/// Exposed so a caller can verify at startup that the configured name can match
/// anything. A name of punctuation alone normalizes to empty and would make
/// every command silently unreachable.
pub fn normalized_name(wake: &str) -> String {
    normalize(wake)
}

/// Reports whether the assistant is named anywhere in these words.
///
/// Used to decline an action, never to decline to look. Logical commitment
/// consults this before filing text early: filing a wake word as dictation puts
/// it in the transcript permanently, before the endpoint can read it as an
/// instruction.
///
/// Deliberately coarser than [`has_command`], because a wake word whose verb
/// has not yet been decoded is a command in progress. A false positive delays
/// one filing; a false negative files a word that should have been a command.
pub fn names(words: &[String], wake: &str) -> bool {
    let wake = normalize(wake);
    !wake.is_empty() && words.iter().any(|w| normalize(w) == wake)
}

/// Reports whether a finalized utterance carries an instruction.
///
/// The caller needs this before deciding whether to retain the audio for a
/// possible continuation: audio holding a command must never be replayed into a
/// later window, or the command fires twice.
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
        assert_eq!(split_str("Luna, copy"), vec![Segment::Run(Command::Copy)]);
        assert_eq!(split_str("Luna copy"), vec![Segment::Run(Command::Copy)]);
        assert_eq!(split_str("Luna, delete."), vec![Segment::Run(Command::Delete)]);
        assert_eq!(split_str("LUNA COPY"), vec![Segment::Run(Command::Copy)]);
    }

    /// The ordinary case: a thought, a pause, then the instruction. The merge
    /// in `main.rs` hands both to this pass as one utterance.
    #[test]
    fn text_before_a_command_is_kept_and_ordered_first() {
        assert_eq!(
            split_str("This is my sentence. Luna, copy"),
            vec![text("This is my sentence."), Segment::Run(Command::Copy)]
        );
    }

    /// Order is the whole point. If the delete were hoisted ahead of the text
    /// it would delete the sentence *before* the one the speaker just finished.
    #[test]
    fn text_after_a_command_stays_after_it() {
        assert_eq!(
            split_str("Luna, delete. Let me try again."),
            vec![Segment::Run(Command::Delete), text("Let me try again.")]
        );
    }

    /// A merged buffer holding a sentence and an instruction is punctuated as
    /// running on, so removing the command must not leave the seam comma
    /// behind.
    #[test]
    fn the_seam_comma_the_merge_creates_is_closed() {
        assert_eq!(
            split_str("This is my first sentence, Luna, copy."),
            vec![text("This is my first sentence."), Segment::Run(Command::Copy)]
        );
    }

    /// Text *after* a command is still being spoken, so it is left alone —
    /// nothing has established that it ended.
    #[test]
    fn text_after_a_command_is_not_force_terminated() {
        assert_eq!(
            split_str("Luna, delete. and then I kept talking"),
            vec![
                Segment::Run(Command::Delete),
                text("and then I kept talking")
            ]
        );
    }

    /// Every command heard is reported. Repeats are not collapsed here or
    /// downstream, and could not be collapsed here in any case: a repeat usually
    /// arrives in a later utterance, and this sees only one.
    #[test]
    fn parsing_reports_every_command_it_hears() {
        assert_eq!(
            split_str("Luna delete Luna delete"),
            vec![Segment::Run(Command::Delete), Segment::Run(Command::Delete)]
        );
        assert_eq!(
            split_str("Luna clear Luna copy"),
            vec![Segment::Run(Command::Clear), Segment::Run(Command::Copy)]
        );
    }

    #[test]
    fn commands_separated_by_dictation_are_distinct() {
        assert_eq!(
            split_str("Luna, delete. A new sentence. Luna, delete."),
            vec![
                Segment::Run(Command::Delete),
                text("A new sentence."),
                Segment::Run(Command::Delete)
            ]
        );
    }

    /// Two deletes in one breath remove two sentences. A repeat is read as
    /// meaning it twice, with undo as the recourse.
    #[test]
    fn a_repeated_command_is_not_collapsed() {
        assert_eq!(
            split_str("Luna, delete. Luna, delete."),
            vec![Segment::Run(Command::Delete), Segment::Run(Command::Delete)]
        );
        assert_eq!(
            split_str("Luna, clear. Luna, clear."),
            vec![Segment::Run(Command::Clear), Segment::Run(Command::Clear)]
        );
    }

    #[test]
    fn every_verb_is_recognised() {
        for (spoken, want) in [
            ("Luna, delete", Command::Delete),
            ("Luna, discard", Command::Discard),
            ("Luna, keep", Command::Keep),
            ("Luna, clear", Command::Clear),
            ("Luna, copy", Command::Copy),
            ("Luna, rollback", Command::Rollback),
            ("Luna, undo", Command::Rollback),
        ] {
            assert_eq!(split_str(spoken), vec![Segment::Run(want)], "{spoken:?}");
        }
    }

    /// The recogniser conjugates a spoken verb, so the `-s` form must fire.
    /// Applies to every verb, not only the one first observed.
    #[test]
    fn a_conjugated_verb_is_the_same_command() {
        for (spoken, want) in [
            ("Luna deletes", Command::Delete),
            ("Luna, discards", Command::Discard),
            ("Luna, keeps", Command::Keep),
            ("Luna, clears", Command::Clear),
            ("Luna, copies", Command::Copy),
            ("Luna, copys", Command::Copy),
            ("Luna, rollbacks", Command::Rollback),
            ("Luna, undoes", Command::Rollback),
            ("Luna, undos", Command::Rollback),
        ] {
            assert_eq!(split_str(spoken), vec![Segment::Run(want)], "{spoken:?}");
        }
    }

    /// The allowance stops at `-s`. Past and progressive forms are plausible
    /// things to say about a person, and a false command deletes text.
    #[test]
    fn other_conjugations_are_still_dictation() {
        for line in [
            "Luna deleted",
            "Luna deleting",
            "Luna discarded",
            "Luna kept",
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
            "Commands: Luna delete, Luna discard, Luna keep, Luna clear, \
             Luna copy, Luna undo, Luna rollback."
        );
        assert!(hint("Jarvis").starts_with("Commands: Jarvis delete,"));
        assert!(!hint("Jarvis").contains("Luna"));
    }

    /// The wake word is configurable, so nothing may be hard-coded to "luna".
    #[test]
    fn the_wake_word_is_configurable() {
        let w = words("Jarvis, copy");
        assert_eq!(split(&w, "Jarvis"), vec![Segment::Run(Command::Copy)]);
        assert_eq!(split(&w, "Luna"), vec![Segment::Text(w.clone())]);
    }

    /// A near-miss is dictation. Deleting a sentence the speaker meant to keep
    /// is far worse than making them repeat the command.
    #[test]
    fn ordinary_speech_is_not_a_command() {
        for line in [
            "Luna went to the store.",
            "I will copy the change tomorrow.",
            "The reviewers delete bad patches.",
            "We should discard the first draft.",
            "Please keep the receipt.",
            "Copy Luna",
            "Luna please copy",
            "Luna, I want you to copy",
            "Luna deleted",
            "Luna deleting",
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
        let w = words(", copy and , delete");
        assert_eq!(split(&w, ""), vec![Segment::Text(w.clone())]);
        assert_eq!(split(&w, "..."), vec![Segment::Text(w)]);
    }

    #[test]
    fn empty_input_produces_no_segments() {
        assert!(split(&[], "Luna").is_empty());
    }

    /// What `main.rs` uses twice: to decide whether the audio may be retained
    /// for a continuation — a retained command would fire a second time — and
    /// to decide whether the hinted re-decode has anything left to find.
    #[test]
    fn has_command_sees_what_split_sees() {
        assert!(has_command(&words("Luna, copy"), "Luna"));
        assert!(has_command(&words("Some text. Luna, delete"), "Luna"));
        assert!(!has_command(&words("Luna went home."), "Luna"));
        assert!(!has_command(&[], "Luna"));
        assert!(!has_command(&words("Luna, commit"), "Luna"));

        assert!(!has_command(&words("Luna, roll back."), "Luna"));
        assert!(!has_command(&words("Lunar delete."), "Luna"));
        assert!(!has_command(&words("Muna delete."), "Luna"));
    }
}
