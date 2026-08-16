//! Self-repair: the speaker starts a word, hears it come out wrong, and says
//! it again — "the regularing… regular expression".
//!
//! Qwen3-ASR transcribes the disfluency verbatim, so the reparandum lands in
//! the transcript unless something takes it out. Measured on
//! `Qwen3-ASR-1.7B-8bit`:
//!
//! | spoken | transcript |
//! |---|---|
//! | "The regularing · regular expression is broken." | `The regularing regular expression is broken.` |
//! | "I need to conf · configure the server today." | `I need to conf configure the server today.` |
//!
//! This is the same problem the wake-word command grammar had — telling an edit
//! apart from dictation with no acoustic signal to lean on — but without the
//! wake word to make it easy, because nobody says "admin" before stuttering.
//! The whole defence is therefore in the pair rule below, and the bar is the
//! same one the commands were held to: **a near-miss is dictation, never a
//! guess.** Deleting a word the speaker meant to keep is much worse than
//! leaving a stutter on screen for them to edit later.
//!
//! # Why not punctuation, and why not edit distance
//!
//! The obvious signal is the pause: the model writes hesitation as a comma. It
//! does not, here. Measured, the repair seam came back *unpunctuated*
//! (`regularing regular`) while an innocent control got the comma
//! (`at work, working late`). Punctuation is anti-correlated with repair on
//! this model, so it is not usable.
//!
//! Edit distance is worse than useless. `form`/`from`, `quiet`/`quite`,
//! `trial`/`trail`, `casual`/`causal` are all one or two edits apart, and "the
//! form from the office" is an ordinary thing to say. So the rule is **prefix
//! containment only** — one word must be a strict prefix of the other, which
//! `form`/`from` is not.
//!
//! # The asymmetry
//!
//! Containment alone still is not enough, and the two directions fail
//! differently, so they get different thresholds.
//!
//! **Longer first** — "regularing" then "regular". English essentially never
//! puts a word immediately before its own prefix. The one systematic exception
//! is a plural noun meeting its verb: "the tests test the parser", "these
//! changes change the behavior" — both measured, both real. That exception is
//! exactly the `-s`/`-es` extension, so excluding it is enough and the stem can
//! be short.
//!
//! **Shorter first** — "conf" then "configure". This direction is genuinely
//! ambiguous, because English *does* put a word before its own derivation: "at
//! work, working late" (measured), "the main maintenance window", "just justify
//! the text". So it needs both a longer stem and an extension that is not a
//! plain inflection. The cost is that four-letter truncations like
//! "conf"/"configure" are not caught; that is deliberate, since catching them
//! would also catch "just"/"justify".
//!
//! # Not handled, on purpose
//!
//! A word repeated exactly — "the the" — is left alone. It is the commonest
//! disfluency of all, but "I had had enough" and "he said that that was fine"
//! are ordinary English, and there is no way to separate them here. The
//! speaker asked for a correction *to a new word*, and this only does that.

use crate::text::normalize;

/// Shortest stem for a backtrack ("regularing" -> "regular").
///
/// Short because the direction is already safe: the plural-verb exception is
/// excluded below, and nothing else in English lands a word directly before its
/// own prefix.
const MIN_STEM_BACKTRACK: usize = 4;

/// Shortest stem for an extension ("regul" -> "regular").
///
/// Longer than the backtrack because this direction has real counterexamples,
/// and every one found is a four-letter word: `work`/`working`,
/// `just`/`justify`, `main`/`maintenance`, `data`/`database`, `test`/`tests`.
/// Five cuts all of them.
const MIN_STEM_EXTENSION: usize = 5;

/// Extensions that make a backtrack ordinary English rather than a repair:
/// a plural noun followed by its verb, as in "the tests test the parser".
const PLURAL: &[&str] = &["s", "es"];

/// Extensions that make an extension ordinary English rather than a repair —
/// "at work working late", "this test tests the parser".
const INFLECTION: &[&str] = &[
    "s", "es", "ed", "d", "ing", "er", "est", "ly", "ness", "ment",
];

/// Hesitation noises that may sit between a false start and its correction
/// without breaking the pair.
const FILLER: &[&str] = &["uh", "um", "er", "erm", "ah", "hmm", "mm", "sorry"];

/// Longest run of hesitation allowed between the two halves of a repair.
const MAX_INTERREGNUM: usize = 3;

/// Drop reparandums, keeping the correction.
///
/// A pure function of one hypothesis, which is what makes it safe to run on
/// provisional text. The sliding window re-decodes the same audio every tick,
/// so the repair sits in every hypothesis from the moment it is spoken; running
/// this twice gives the same answer, and a re-decode that no longer hears the
/// false start simply stops dropping it, with nothing to undo.
///
/// Never returns an empty list for a non-empty one: the correction is always
/// kept, only what it replaced is dropped.
pub fn repair<'a>(words: impl Iterator<Item = &'a str>) -> Vec<String> {
    let mut kept: Vec<String> = Vec::new();

    for word in words {
        let pause = interregnum(&kept);
        if let Some(at) = kept.len().checked_sub(pause + 1)
            && is_revision(&kept[at], word)
        {
            // Take the false start and the hesitation after it with it.
            kept.truncate(at);
        }
        kept.push(word.to_string());
    }

    kept
}

/// Is `b` the speaker's second attempt at `a`?
fn is_revision(a: &str, b: &str) -> bool {
    let (a, b) = (normalize(a), normalize(b));

    // Digits are not stutters. "2020 2021" shares no prefix anyway, but "20"
    // followed by "2020" would, and a spoken figure is not a repair.
    if a.is_empty() || a == b || !is_word(&a) || !is_word(&b) {
        return false;
    }

    // Characters, not bytes: this is a threshold on how much word the two share,
    // and `len()` would make it five times looser on any script outside ASCII.
    if let Some(extension) = a.strip_prefix(&b) {
        // "regularing" then "regular".
        return b.chars().count() >= MIN_STEM_BACKTRACK && !PLURAL.contains(&extension);
    }
    if let Some(extension) = b.strip_prefix(&a) {
        // "regul" then "regular".
        return a.chars().count() >= MIN_STEM_EXTENSION && !INFLECTION.contains(&extension);
    }
    false
}

fn is_word(normalized: &str) -> bool {
    !normalized.is_empty() && normalized.chars().all(char::is_alphabetic)
}

/// How many trailing words of `kept` are hesitation rather than dictation.
fn interregnum(kept: &[String]) -> usize {
    let mut n = 0;
    while n < MAX_INTERREGNUM {
        let Some(word) = kept.len().checked_sub(n + 1).map(|i| normalize(&kept[i])) else {
            break;
        };

        // "I mean" is the canonical spoken repair marker. "mean" on its own is
        // an ordinary verb, so the "I" is required.
        let preceded_by_i = kept
            .len()
            .checked_sub(n + 2)
            .is_some_and(|i| normalize(&kept[i]) == "i");
        if word == "mean" && preceded_by_i {
            n += 2;
            continue;
        }

        if FILLER.contains(&word.as_str()) {
            n += 1;
            continue;
        }
        break;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fix(text: &str) -> String {
        repair(text.split_whitespace()).join(" ")
    }

    /// The measured transcript of "The regularing · regular expression is
    /// broken", which is the case this module exists for.
    #[test]
    fn a_backtracked_word_is_replaced() {
        assert_eq!(
            fix("The regularing regular expression is broken."),
            "The regular expression is broken."
        );
    }

    /// The other direction: a truncated false start, finished on the second go.
    #[test]
    fn a_truncated_false_start_is_replaced() {
        assert_eq!(fix("I need to regul regular expressions"), "I need to regular expressions");
        assert_eq!(fix("open the config configuration file"), "open the configuration file");
    }

    #[test]
    fn hesitation_between_the_halves_is_absorbed() {
        assert_eq!(fix("the regularing uh regular expression"), "the regular expression");
        assert_eq!(fix("the regularing um, er, regular expression"), "the regular expression");
        assert_eq!(fix("the regularing I mean regular expression"), "the regular expression");
    }

    /// "mean" without the "I" is an ordinary verb and must not open a gap.
    #[test]
    fn mean_alone_is_not_a_hesitation() {
        assert_eq!(fix("the regularing mean regular"), "the regularing mean regular");
    }

    /// Correcting the correction.
    #[test]
    fn a_chain_of_attempts_keeps_only_the_last() {
        assert_eq!(fix("the regul regularing regular expression"), "the regular expression");
    }

    /// Measured controls. Each is a real sentence the model returned, and each
    /// is what a similarity-based rule would have destroyed.
    #[test]
    fn ordinary_english_is_untouched() {
        for line in [
            // Plural noun meeting its verb — the whole reason -s is excluded.
            "These changes change the behavior of the parser.",
            "The tests test the parser thoroughly.",
            "The results result from a bad merge.",
            // A word before its own derivation.
            "He was at work, working late on the report.",
            "The main maintenance window is on Sunday.",
            "Just justify the text and be done.",
            "This test tests the parser.",
            // One or two edits apart, but neither contains the other.
            "Take the form from the office.",
            "That is not quiet quite right.",
            // Short shared prefixes that mean nothing.
            "I found one online store.",
            "Put the car carpet in the hall.",
            "He did it for form's sake.",
        ] {
            assert_eq!(fix(line), line, "must be left alone: {line:?}");
        }
    }

    /// Exact repetition is deliberately out of scope: "had had" and "that that"
    /// are ordinary English and nothing here can tell them from a stutter.
    #[test]
    fn exact_repetition_is_left_alone() {
        assert_eq!(fix("I had had enough of it"), "I had had enough of it");
        assert_eq!(fix("the the regular expression"), "the the regular expression");
    }

    /// Numbers share prefixes constantly and are never stutters.
    #[test]
    fn digits_are_never_a_repair() {
        assert_eq!(fix("in 20 2020 we shipped"), "in 20 2020 we shipped");
        assert_eq!(fix("section 1 1.1 covers it"), "section 1 1.1 covers it");
    }

    /// The sliding window hands the same text to this pass on every tick, and a
    /// re-decode that revises the repair away has to be able to withdraw it.
    #[test]
    fn repairing_is_idempotent() {
        let once = fix("The regularing regular expression is broken.");
        assert_eq!(fix(&once), once);
    }

    /// A correction always survives; only what it replaced goes.
    #[test]
    fn the_last_word_is_never_dropped() {
        for line in ["regularing regular", "regul regular", "hello", "uh uh uh"] {
            assert!(!repair(line.split_whitespace()).is_empty(), "emptied: {line:?}");
        }
    }

    #[test]
    fn empty_input_stays_empty() {
        assert_eq!(repair("".split_whitespace()), Vec::<String>::new());
    }

    /// The stem thresholds count characters, not bytes. Measured in bytes, a
    /// two-character CJK stem is six and clears a five-byte bar it should not.
    #[test]
    fn stem_thresholds_are_measured_in_characters() {
        assert_eq!(fix("\u{4e2d}\u{56fd} \u{4e2d}\u{56fd}\u{4eba}"), "\u{4e2d}\u{56fd} \u{4e2d}\u{56fd}\u{4eba}");
        // Four Cyrillic characters is under the backtrack stem, as for ASCII.
        assert_eq!(fix("\u{0440}\u{0435}\u{0433} \u{0440}\u{0435}\u{0433}\u{0443}"), "\u{0440}\u{0435}\u{0433} \u{0440}\u{0435}\u{0433}\u{0443}");
    }

    /// Punctuation and case are the model's business, not the pair rule's.
    #[test]
    fn punctuation_does_not_hide_a_repair() {
        assert_eq!(fix("The Regularing, regular expression."), "The regular expression.");
    }
}
