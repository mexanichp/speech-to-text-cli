//! Word and sentence primitives shared across the pipeline.
//!
//! The transcript state machine, the repair pass, the command parser and the
//! renderer all need the same three questions answered: whether two words are
//! the same, whether a word ends a sentence, and where one sentence ends and
//! the next begins.
//!
//! These answers must be identical for every caller. If they diverge, the state
//! machine and the display disagree about where sentence boundaries lie, and
//! sentence boundaries determine what `delete` and `discard` can reach.

const TERMINALS: [char; 6] = ['.', '!', '?', '\u{3002}', '\u{ff1f}', '\u{ff01}'];

/// Punctuation that joins clauses rather than ending them.
const JOINERS: [char; 5] = [',', ';', ':', '\u{3001}', '\u{ff0c}'];

/// Reports whether a word closes a sentence.
///
/// Trailing quotes and brackets are ignored, so `end."` still closes.
pub fn ends_sentence(word: &str) -> bool {
    word.trim_end_matches(['"', '\'', ')', ']', '\u{201d}', '\u{2019}'])
        .ends_with(TERMINALS)
}

/// Terminates the last word of a passage known to have ended.
///
/// Replaces a trailing clause joiner with a full stop, and appends one if the
/// word carries no terminal punctuation. Tokens that are punctuation alone are
/// removed first, so no bare `.` is left standing as a word.
///
/// Applied only where the caller has external evidence that the passage ended —
/// in [`crate::command`], that the following words were an instruction. Text
/// that merely looks finished is left alone, since the recogniser punctuates
/// fragments as though they were sentences.
pub fn close_sentence(words: &mut Vec<String>) {
    while words.last().is_some_and(|w| normalize(w).is_empty()) {
        words.pop();
    }

    let Some(last) = words.last_mut() else {
        return;
    };
    if ends_sentence(last) {
        return;
    }
    *last = format!("{}.", last.trim_end_matches(JOINERS));
}

/// Reduces a word to its comparison form: alphanumerics only, lower case.
///
/// Two words that differ only in case or punctuation compare equal, so a
/// cosmetic revision between decodes does not stall agreement.
///
/// Punctuation is discarded, so this cannot be used to compare sentence
/// boundaries. Callers needing that must use [`ends_sentence`].
pub fn normalize(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Splits text on the sentence boundaries the recogniser chose.
///
/// A boundary is terminal punctuation followed by whitespace. Three guards
/// suppress false boundaries: single-letter initials ([`is_initial`]), text
/// resuming in lower case ([`resumes_lowercase`]), and openers that cannot
/// begin a sentence ([`implausible_opener`]). Decimals need no guard, since no
/// whitespace follows the point.
///
/// The document stores one entry per sentence, so this determines what `delete`
/// and `discard` can address, not merely how lines are broken.
///
/// # Returns
///
/// The sentences in order, or the whole input as a single element when no
/// boundary is found.
pub fn split_sentences(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let chars: Vec<(usize, char)> = text.char_indices().collect();

    for (i, (idx, ch)) in chars.iter().enumerate() {
        if !TERMINALS.contains(ch) {
            continue;
        }
        let mut end = idx + ch.len_utf8();
        let mut j = i + 1;
        while let Some((next_idx, next_ch)) = chars.get(j) {
            if TERMINALS.contains(next_ch) || *next_ch == '"' || *next_ch == '\'' {
                end = next_idx + next_ch.len_utf8();
                j += 1;
            } else {
                break;
            }
        }
        if !chars.get(j).is_some_and(|(_, c)| c.is_whitespace()) {
            continue;
        }
        if is_initial(&chars, i) || resumes_lowercase(&chars, j) {
            continue;
        }
        if implausible_opener(&text[chars[j].0..]) {
            continue;
        }
        let part = text[start..end].trim();
        if !part.is_empty() {
            parts.push(part);
        }
        start = end;
    }

    let tail = text[start..].trim();
    if !tail.is_empty() {
        parts.push(tail);
    }
    if parts.is_empty() { vec![text] } else { parts }
}

/// Reports whether the stop at `at` closes a single-letter initial.
///
/// Covers the `S.` of "U.S." and the `R.` of "J. R. R. Tolkien", neither of
/// which is a sentence boundary. Requires exactly one letter before the stop,
/// so longer abbreviations are unaffected.
fn is_initial(chars: &[(usize, char)], at: usize) -> bool {
    let Some((_, prev)) = at.checked_sub(1).and_then(|k| chars.get(k)) else {
        return false;
    };
    if !prev.is_alphabetic() {
        return false;
    }
    at.checked_sub(2)
        .and_then(|k| chars.get(k))
        .is_none_or(|(_, c)| !c.is_alphabetic())
}

/// Reports whether text from `from` resumes in lower case, indicating the stop
/// was internal punctuation.
///
/// Tests for lower case rather than requiring upper case, because caseless
/// scripts must still split.
fn resumes_lowercase(chars: &[(usize, char)], from: usize) -> bool {
    chars[from..]
        .iter()
        .map(|(_, c)| *c)
        .find(|c| !c.is_whitespace())
        .is_some_and(char::is_lowercase)
}

/// Words that cannot begin an English sentence.
///
/// Each requires an antecedent that a split would place on the other side of
/// the boundary, so text opening with one is a fragment however it was
/// capitalised.
///
/// The list excludes `to`, `for`, `with`, `from`, `by`, `in`, `on`, `at`,
/// `and` and `but`: all front real sentences in ordinary speech, and vetoing
/// them would merge sentences the speaker did separate.
const BARE_RELATIONAL: [&str; 5] = ["of", "which", "whom", "whose", "than"];

/// Two-word openings that begin with a [`BARE_RELATIONAL`] word yet are
/// ordinary sentence openers.
const IDIOMATIC_OPENERS: [(&str, &str); 3] = [("of", "course"), ("which", "is"), ("which", "was")];

/// Leading words searched for the comma that marks an opener as an aside.
///
/// Two rather than more, so that "Of coordination between participants," stays
/// vetoed: its comma falls at the fourth word and marks a clause boundary
/// rather than an aside.
const OPENER_COMMA_WINDOW: usize = 2;

/// Reports whether splitting before `rest` would strand a fragment.
///
/// Tests whether the text after a boundary can grammatically be a sentence,
/// which is a property of the string. This is distinct from judging whether the
/// text *before* a boundary sounds finished: that is a question about the
/// speaker's intent, which the recogniser answers better from the audio and
/// which no lexical rule here attempts.
///
/// Two exemptions narrow it: a whitelisted [`IDIOMATIC_OPENERS`] pair, and an
/// opener set off by a comma within [`OPENER_COMMA_WINDOW`] words.
///
/// A false positive merges two real sentences into one entry, costing `delete`
/// granularity. A false negative leaves a permanent fragment. [`BARE_RELATIONAL`]
/// is short because the first cost is real and exists at all because the second
/// is worse.
fn implausible_opener(rest: &str) -> bool {
    let mut words = rest.split_whitespace();
    let Some(first) = words.next() else {
        return false;
    };
    let head = normalize(first);
    if !BARE_RELATIONAL.contains(&head.as_str()) {
        return false;
    }

    let second = words.next();
    if let Some(second) = second
        && IDIOMATIC_OPENERS.contains(&(head.as_str(), normalize(second).as_str()))
    {
        return false;
    }

    let aside = std::iter::once(first)
        .chain(second)
        .take(OPENER_COMMA_WINDOW)
        .any(|w| w.ends_with(JOINERS));

    !aside
}

/// Split a decode into words, the one way the rest of the pipeline does it.
pub fn words(text: &str) -> Vec<String> {
    text.split_whitespace().map(str::to_string).collect()
}

/// Words that open a sentence but are also how a continued clause begins.
///
/// Unlike [`BARE_RELATIONAL`] these are not vetoed, because they front real
/// sentences in ordinary speech. They are used only to require more evidence
/// before a boundary is made permanent; see [`opens_a_continuation`].
///
/// Restricted to true coordinators plus `because`. Subordinators such as `if`,
/// `when` and `while` front real sentences too often to be worth the delay, and
/// `which` is already vetoed outright.
const CLAUSE_CONTINUERS: [&str; 7] = ["and", "but", "or", "nor", "yet", "so", "because"];

/// Reports whether text reads as a continuation of the sentence before it.
///
/// Used to defer filing across a boundary, never to refuse one. The recogniser
/// ends a sentence at a hesitation while that hesitation is the end of the
/// audio, then withdraws the boundary once it hears the continuation. Filing in
/// between splits one spoken sentence permanently, and the text after such a
/// boundary characteristically opens with a coordinator.
///
/// Requiring every boundary to wait instead would slow the whole session; this
/// spends the delay only where the risk is. A false positive delays one filing
/// by one sentence, whereas a false negative splits a sentence permanently.
pub fn opens_a_continuation(text: &str) -> bool {
    text.split_whitespace()
        .next()
        .is_some_and(|w| CLAUSE_CONTINUERS.contains(&normalize(w).as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reported shape: what follows the boundary is a coordinate clause,
    /// which is what a hesitation promoted to a full stop looks like.
    #[test]
    fn a_coordinate_clause_reads_as_a_continuation() {
        for text in [
            "and accurately, and knows when to commit the messages.",
            "And politics, which I rather feel is off the rails.",
            "But the numbers do not agree.",
            "so the buffer does not overflow.",
            "because it did not explain the trade-offs.",
        ] {
            assert!(opens_a_continuation(text), "should defer: {text}");
        }
    }

    /// Ordinary sentence openings must not be deferred. This costs a filing
    /// delay rather than a wrong transcript, but it is paid on every sentence
    /// that trips it, so the list has to stay narrow.
    #[test]
    fn an_ordinary_opening_is_not_a_continuation() {
        for text in [
            "Also, take a look at the current transcript.",
            "There was a spike around three in the afternoon.",
            "If you look at the graph it is obvious.",
            "When I ran it again the error was gone.",
            "While that is true, it is not the whole story.",
            "That is the sort of thing that gets forgotten.",
            "However, it still works that way.",
            "",
        ] {
            assert!(!opens_a_continuation(text), "should not defer: {text}");
        }
    }

    fn words(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    #[test]
    fn quoted_and_caseless_sentence_endings_count() {
        assert!(ends_sentence("said.\""));
        assert!(ends_sentence("\u{4f60}\u{597d}\u{3002}"));
        assert!(!ends_sentence("mid-sentence"));
    }

    #[test]
    fn normalize_strips_case_and_punctuation() {
        assert_eq!(normalize("Regularing,"), "regularing");
        assert_eq!(normalize("real-time"), "realtime");
        assert_eq!(normalize("..."), "");
    }

    /// The measured case: the merge hands the model one buffer holding the
    /// sentence and the instruction, so it commas the seam.
    #[test]
    fn a_dangling_comma_becomes_a_full_stop() {
        let mut w = words("This is my first sentence,");
        close_sentence(&mut w);
        assert_eq!(w.join(" "), "This is my first sentence.");
    }

    #[test]
    fn unpunctuated_text_gains_a_full_stop() {
        let mut w = words("no punctuation at all");
        close_sentence(&mut w);
        assert_eq!(w.join(" "), "no punctuation at all.");
    }

    /// Already-terminal punctuation is the model's, and better than ours.
    #[test]
    fn an_existing_ending_is_left_exactly_as_it_is() {
        for line in [
            "It already ends properly.",
            "Does it though?",
            "Absolutely!",
            "She said \"go.\"",
            "\u{4f60}\u{597d}\u{3002}",
            "I had no idea...",
        ] {
            let mut w = words(line);
            close_sentence(&mut w);
            assert_eq!(w.join(" "), line, "must be left alone: {line:?}");
        }
    }

    /// A stop must never be appended to punctuation standing on its own, or a
    /// bare "." ends up in the transcript as a word.
    #[test]
    fn punctuation_only_tokens_are_dropped_rather_than_terminated() {
        let mut w = words("real words ,");
        close_sentence(&mut w);
        assert_eq!(w.join(" "), "real words.");

        let mut only = words(", ...");
        close_sentence(&mut only);
        assert!(only.is_empty());

        let mut nothing: Vec<String> = Vec::new();
        close_sentence(&mut nothing);
        assert!(nothing.is_empty());
    }

    #[test]
    fn merged_decode_splits_into_sentences() {
        assert_eq!(
            split_sentences("This is the first complete sentence. And this is a totally separate one."),
            vec![
                "This is the first complete sentence.",
                "And this is a totally separate one."
            ]
        );
    }

    #[test]
    fn fused_continuation_stays_one_sentence() {
        let s = "So here is the thing I wanted to say, and that's why it matters.";
        assert_eq!(split_sentences(s), vec![s]);
    }

    #[test]
    fn decimals_and_abbreviations_do_not_split() {
        assert_eq!(split_sentences("It cost 3.5 million."), vec!["It cost 3.5 million."]);
    }

    /// The stop after an initial is followed by a space, so the whitespace rule
    /// alone broke these lines in the middle of a sentence.
    #[test]
    fn initials_do_not_split() {
        let s = "The U.S. government said so.";
        assert_eq!(split_sentences(s), vec![s]);

        let s = "I read J. R. R. Tolkien last year.";
        assert_eq!(split_sentences(s), vec![s]);
    }

    #[test]
    fn a_lowercase_resumption_is_not_a_boundary() {
        let s = "She joined N.A.S.A. and never looked back.";
        assert_eq!(split_sentences(s), vec![s]);
    }

    /// Caseless scripts must still split, which is why the guard tests for
    /// lower case rather than requiring upper.
    #[test]
    fn caseless_scripts_still_split() {
        assert_eq!(
            split_sentences("\u{4f60}\u{597d}\u{3002} \u{518d}\u{89c1}\u{3002}"),
            vec!["\u{4f60}\u{597d}\u{3002}", "\u{518d}\u{89c1}\u{3002}"]
        );
    }

    #[test]
    fn unterminated_text_is_kept_whole() {
        assert_eq!(split_sentences("no terminal punctuation"), vec!["no terminal punctuation"]);
        assert_eq!(split_sentences(""), vec![""]);
    }

    /// The reported failure, as a string. A fragment opening with a bare
    /// preposition is not a sentence, however the model capitalised it.
    #[test]
    fn a_fragment_that_cannot_be_a_sentence_is_not_split_off() {
        let s = "It tries to solve the problem. Of coordination between participants, \
                 a single node can communicate with the other nodes.";
        assert_eq!(split_sentences(s), vec![s]);
    }

    /// Every other bare relational word, in the same shape.
    #[test]
    fn the_other_bare_relational_openers_are_vetoed_too() {
        for opener in [
            "Which the scheduler then retries on the next tick.",
            "Whom the committee had already interviewed twice.",
            "Whose latency budget nobody had ever measured.",
            "Than the previous release managed on the same hardware.",
        ] {
            let s = format!("The design has a flaw. {opener}");
            assert_eq!(split_sentences(&s), vec![s.as_str()], "must not split: {opener}");
        }
    }

    /// The controls. Every one of these is a legitimate way to open a spoken
    /// sentence, and merging it into its predecessor would cost a document
    /// entry the speaker can delete.
    #[test]
    fn legitimate_openers_still_split() {
        for opener in [
            "And that is why it matters.",
            "But nobody had measured it.",
            "To be fair, the numbers were noisy.",
            "For example, the trim never fires.",
            "For now we wait.",
            "By the way, the release shipped.",
            "By Friday the release will ship.",
            "With respect, that is not what happened.",
            "With the migration done we can go home.",
            "In fact the opposite is true.",
            "From here the path is obvious.",
            "That said, it is worth checking.",
            "Of course the answer is no.",
            "Which is why the buffer keeps growing.",
            "Which was never the plan.",
            "Of course, the answer is no.",
        ] {
            let s = format!("The design has a flaw. {opener}");
            assert_eq!(
                split_sentences(&s),
                vec!["The design has a flaw.", opener],
                "must still split: {opener}"
            );
        }
    }

    /// The comma window is two words, not four. A comma further out is a clause
    /// boundary inside the fragment rather than an aside setting it off, which
    /// is exactly the shape of the reported failure.
    #[test]
    fn a_late_comma_does_not_rescue_a_fragment() {
        let s = "It tries to solve the problem. Of coordination and consensus, which is hard.";
        assert_eq!(split_sentences(s), vec![s]);
    }

    #[test]
    fn runs_of_terminal_punctuation_are_absorbed() {
        assert_eq!(
            split_sentences("Really?! I had no idea..."),
            vec!["Really?!", "I had no idea..."]
        );
    }
}
