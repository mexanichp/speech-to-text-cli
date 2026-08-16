//! Word and sentence helpers shared by the transcript, the repair pass, the
//! command parser and the renderer.
//!
//! They all need the same questions answered — "are these the same word?",
//! "does this word end a sentence?", "where does one sentence stop and the next
//! begin?" — and every answer has to be identical everywhere, or the state
//! machine and the display disagree about where a sentence is.

const TERMINALS: [char; 6] = ['.', '!', '?', '\u{3002}', '\u{ff1f}', '\u{ff01}'];

/// Punctuation that joins clauses rather than ending them.
const JOINERS: [char; 5] = [',', ';', ':', '\u{3001}', '\u{ff0c}'];

/// Does this word close a sentence? Trailing quotes and brackets don't count.
pub fn ends_sentence(word: &str) -> bool {
    word.trim_end_matches(['"', '\'', ')', ']', '\u{201d}', '\u{2019}'])
        .ends_with(TERMINALS)
}

/// Force the last word to end a sentence, for a passage that is known to have
/// ended even though the model punctuated it as though it ran on.
///
/// Only the caller can know that, which is why this is not applied anywhere
/// text merely *looks* finished. See `command.rs`, where the evidence is that
/// the next words the speaker said were an instruction rather than dictation.
pub fn close_sentence(words: &mut Vec<String>) {
    // A token that is nothing but punctuation has no sentence to close, and
    // appending a stop to it would leave a bare "." standing as a word.
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

/// Compare words ignoring case and punctuation, so cosmetic revisions don't
/// stall commitment and so "regularing," matches "regularing".
pub fn normalize(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Split an utterance on the sentence boundaries the model chose.
///
/// A boundary is terminal punctuation followed by whitespace. Decimals survive
/// because the following character is not a space; abbreviations need the two
/// guards below, because "U.S. government" does put a space after the stop.
///
/// This is load-bearing for `Luna, reject`, not just for line breaks: the
/// document stores one entry per sentence, so a re-decoded continuation that
/// the model split into two thoughts must arrive as two rejectable units.
pub fn split_sentences(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let chars: Vec<(usize, char)> = text.char_indices().collect();

    for (i, (idx, ch)) in chars.iter().enumerate() {
        if !TERMINALS.contains(ch) {
            continue;
        }
        // Absorb any run of closing punctuation ("...", "?!").
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

/// Is the stop at `at` closing a single-letter initial — the `S.` of "U.S.",
/// the `R.` of "J. R. R. Tolkien"? Never a sentence boundary.
fn is_initial(chars: &[(usize, char)], at: usize) -> bool {
    let Some((_, prev)) = at.checked_sub(1).and_then(|k| chars.get(k)) else {
        return false;
    };
    if !prev.is_alphabetic() {
        return false;
    }
    // Exactly one letter before the stop, so nothing longer is caught.
    at.checked_sub(2)
        .and_then(|k| chars.get(k))
        .is_none_or(|(_, c)| !c.is_alphabetic())
}

/// Does the text after `from` resume in lower case? Then the stop was internal
/// punctuation and the sentence is still going.
///
/// Deliberately tests for *lower* case rather than requiring upper: scripts
/// without case, Chinese in particular, are caseless and must still split.
fn resumes_lowercase(chars: &[(usize, char)], from: usize) -> bool {
    chars[from..]
        .iter()
        .map(|(_, c)| *c)
        .find(|c| !c.is_whitespace())
        .is_some_and(char::is_lowercase)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // What the model actually returns when two utterances are re-decoded
        // together and it judges them distinct.
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

    #[test]
    fn runs_of_terminal_punctuation_are_absorbed() {
        assert_eq!(
            split_sentences("Really?! I had no idea..."),
            vec!["Really?!", "I had no idea..."]
        );
    }
}
