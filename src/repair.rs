//! Removal of self-repairs, where a speaker restarts a word they misspoke.
//!
//! The recogniser transcribes disfluency verbatim, so "the regularing… regular
//! expression" reaches the transcript with the false start intact. [`repair`]
//! drops the false start and keeps the correction.
//!
//! # Detection rule
//!
//! A pair qualifies only when one word is a strict prefix of the other. Two
//! signals that appear usable are not: the recogniser does not punctuate the
//! hesitation seam, and edit distance cannot separate `form`/`from` or
//! `quiet`/`quite` from genuine repairs.
//!
//! Containment alone is insufficient, and the two directions fail differently,
//! so each has its own stem threshold:
//!
//! - Backtrack ("regularing" then "regular"). English rarely places a word
//!   before its own prefix; the systematic exception is a plural noun meeting
//!   its verb ("the tests test the parser"), which the `-s`/`-es` exclusion
//!   covers. A short stem therefore suffices.
//! - Extension ("conf" then "configure"). English does place a word before its
//!   own derivation ("at work working late"), so this requires a longer stem
//!   and an extension that is not a plain inflection.
//!
//! # Limitations
//!
//! Exact repetition ("the the") is never treated as a repair, because "I had
//! had enough" is ordinary English. Substitutions that share no prefix
//! ("Tuesday… Wednesday") are not detectable here. Truncations shorter than
//! [`MIN_STEM_EXTENSION`] are not caught, since catching them would also catch
//! "just"/"justify".
//!
//! A false positive deletes a word the speaker intended to keep, which is worse
//! than leaving a visible stutter, so an ambiguous pair is treated as
//! dictation.

use crate::text::normalize;

/// Shortest stem accepted for a backtrack, in characters.
///
/// "regularing" then "regular". Short because [`PLURAL`] already excludes the
/// only systematic counterexample in this direction.
const MIN_STEM_BACKTRACK: usize = 4;

/// Shortest stem accepted for an extension, in characters.
///
/// "conf" then "configure". Longer than [`MIN_STEM_BACKTRACK`] because this
/// direction has real counterexamples — `work`/`working`, `just`/`justify`,
/// `main`/`maintenance` — all of which are four letters.
const MIN_STEM_EXTENSION: usize = 5;

/// Suffixes that make a backtrack ordinary English: a plural noun followed by
/// its verb, as in "the tests test the parser".
const PLURAL: &[&str] = &["s", "es"];

/// Suffixes that make an extension ordinary English, as in "at work working
/// late".
const INFLECTION: &[&str] = &[
    "s", "es", "ed", "d", "ing", "er", "est", "ly", "ness", "ment",
];

/// Hesitation noises that may sit between a false start and its correction
/// without breaking the pair.
const FILLER: &[&str] = &["uh", "um", "er", "erm", "ah", "hmm", "mm", "sorry"];

/// Longest run of hesitation allowed between the two halves of a repair.
const MAX_INTERREGNUM: usize = 3;

/// Removes false starts, keeping each correction.
///
/// Pure and idempotent with respect to one hypothesis, which is what makes it
/// safe on provisional text: the sliding window re-decodes the same audio every
/// tick, so applying this repeatedly yields the same result, and a later decode
/// that no longer contains the false start simply stops dropping it.
///
/// # Returns
///
/// The words with reparandums removed. Never empty for a non-empty input, as
/// only the replaced words are dropped.
pub fn repair<'a>(words: impl Iterator<Item = &'a str>) -> Vec<String> {
    let mut kept: Vec<String> = Vec::new();

    for word in words {
        let pause = interregnum(&kept);
        if let Some(at) = kept.len().checked_sub(pause + 1)
            && is_revision(&kept[at], word)
        {
            kept.truncate(at);
        }
        kept.push(word.to_string());
    }

    kept
}

/// Reports whether `b` is a second attempt at `a`.
///
/// Requires strict prefix containment in either direction, subject to the stem
/// thresholds and inflection exclusions described in the module documentation.
/// Comparison is on normalized forms, and lengths count characters rather than
/// bytes so the thresholds hold outside ASCII.
fn is_revision(a: &str, b: &str) -> bool {
    let (a, b) = (normalize(a), normalize(b));

    if a.is_empty() || a == b || !is_word(&a) || !is_word(&b) {
        return false;
    }

    if let Some(extension) = a.strip_prefix(&b) {
        return b.chars().count() >= MIN_STEM_BACKTRACK && !PLURAL.contains(&extension);
    }
    if let Some(extension) = b.strip_prefix(&a) {
        return a.chars().count() >= MIN_STEM_EXTENSION && !INFLECTION.contains(&extension);
    }
    false
}

/// Reports whether a normalized token is alphabetic, excluding numerals from
/// the pair rule: spoken figures share prefixes without being repairs.
fn is_word(normalized: &str) -> bool {
    !normalized.is_empty() && normalized.chars().all(char::is_alphabetic)
}

/// Counts trailing words of `kept` that are hesitation rather than dictation.
///
/// These may separate a false start from its correction without breaking the
/// pair, up to [`MAX_INTERREGNUM`].
fn interregnum(kept: &[String]) -> usize {
    let mut n = 0;
    while n < MAX_INTERREGNUM {
        let Some(word) = kept.len().checked_sub(n + 1).map(|i| normalize(&kept[i])) else {
            break;
        };

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
            "These changes change the behavior of the parser.",
            "The tests test the parser thoroughly.",
            "The results result from a bad merge.",
            "He was at work, working late on the report.",
            "The main maintenance window is on Sunday.",
            "Just justify the text and be done.",
            "This test tests the parser.",
            "Take the form from the office.",
            "That is not quiet quite right.",
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
        assert_eq!(fix("\u{0440}\u{0435}\u{0433} \u{0440}\u{0435}\u{0433}\u{0443}"), "\u{0440}\u{0435}\u{0433} \u{0440}\u{0435}\u{0433}\u{0443}");
    }

    /// Punctuation and case are the model's business, not the pair rule's.
    #[test]
    fn punctuation_does_not_hide_a_repair() {
        assert_eq!(fix("The Regularing, regular expression."), "The regular expression.");
    }
}
