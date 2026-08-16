//! Transcript state machine: LocalAgreement-n commitment, and the document it
//! commits into.
//!
//! Two layers, deliberately separate:
//!
//! * **The window.** A word is promoted from provisional to agreed once it
//!   appears in the longest common prefix of the last `n` hypotheses. Words are
//!   compared in normalised form so a punctuation-only revision ("real time" ->
//!   "real-time") doesn't block commitment, but the surface form emitted is
//!   always taken from the most recent hypothesis.
//!
//! * **The document.** Finished sentences, each carrying whether the *speaker*
//!   has committed it. This is the deliverable, and it is mutable until the
//!   session ends, because `Luna, reject` has to be able to reach any sentence
//!   in it.
//!
//! # Two meanings of "committed", and why they are not the same word
//!
//! The window's `committed` is a claim about the **pipeline**: these words are
//! agreed and nothing upstream will revise them. A `Sentence`'s `committed` is
//! a claim about the **speaker**: they said so out loud.
//!
//! The styling contract is unaffected by the second one. Dim still means "the
//! pipeline may rewrite this" and plain still means it will not. A speaker
//! rejecting their own sentence is not the pipeline changing its mind, so plain
//! text keeps its promise even though a rejected sentence can leave the screen.

use crate::repair::repair;
use crate::text::normalize;
use std::collections::VecDeque;

/// A finished sentence in the document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sentence {
    pub words: Vec<String>,
    /// Set by `Luna, commit`. Purely a record of what the speaker approved —
    /// it does **not** make the sentence unreachable, because `Luna, reject`
    /// must be able to take back a committed sentence too.
    pub committed: bool,
}

impl Sentence {
    pub fn text(&self) -> String {
        self.words.join(" ")
    }
}

pub struct Transcript {
    agreement_n: usize,
    /// Last `n` hypotheses for the current window, as word lists.
    hypotheses: VecDeque<Vec<String>>,
    /// Agreed within the current window.
    committed: Vec<String>,
    /// Finished sentences, oldest first.
    ///
    /// Deliberately unbounded. It used to be pruned to a few hundred words,
    /// which was right when it was a scratch buffer whose only consumer was
    /// retraction — but it is the transcript now, and dropping the front of it
    /// would silently discard what the speaker came here to produce. A five
    /// hour session is on the order of a megabyte. What actually has to stay
    /// bounded is the *rendering* cost, and that is the renderer's problem:
    /// it walks back from the newest sentence and stops once the screen is
    /// full.
    document: Vec<Sentence>,
    /// What the last text-removing commands took away, newest last.
    undo: Vec<Undo>,
    /// Bumped on every change to the document, so the autosave can tell whether
    /// there is anything to write without diffing or cloning it.
    revision: u64,
}

/// Enough to put back what one destructive command removed.
///
/// Stored as a delta rather than a document snapshot. A reject removes exactly
/// one sentence, so remembering that sentence costs one sentence — where
/// snapshotting would cost a copy of the whole transcript per undo step, and
/// the transcript is deliberately unbounded.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Undo {
    Rejected(Sentence),
    Cleared(Vec<Sentence>),
}

/// How many destructive commands can be taken back.
///
/// Deep enough that walking a series of rejects back out is never blocked by
/// the limit, bounded because a `Cleared` entry holds a whole document.
const UNDO_DEPTH: usize = 64;

impl Transcript {
    pub fn new(agreement_n: usize) -> Self {
        Self {
            agreement_n: agreement_n.max(2),
            hypotheses: VecDeque::new(),
            committed: Vec::new(),
            document: Vec::new(),
            undo: Vec::new(),
            revision: 0,
        }
    }

    /// Feed a fresh hypothesis for the current window. Returns the words newly
    /// promoted to agreed.
    ///
    /// Self-repairs are resolved first, on every hypothesis, because the pass
    /// is pure: the same false start sits in the audio from the moment it is
    /// spoken until the utterance ends, so re-deriving the edit each tick is
    /// both stable and self-correcting — a re-decode that no longer hears the
    /// stutter simply stops dropping it.
    ///
    /// Commands are **not** resolved here, for exactly the reason repairs are.
    /// `Luna, reject` destroys a sentence, so re-deriving it every tick would
    /// empty the document. Commands run once, on the finalized utterance; see
    /// `command.rs`.
    ///
    /// An empty hypothesis is discarded rather than admitted to the agreement
    /// window. It means "no evidence", not "evidence of nothing": the sidecar
    /// returns one for any buffer under its minimum length, which includes the
    /// first tick of most utterances — the window is barely longer than the
    /// pre-roll at that point. Letting it in would drive the common prefix to
    /// zero for the next `n` ticks, stalling commitment and blanking the
    /// provisional tail off the screen mid-sentence.
    pub fn push_hypothesis(&mut self, text: &str) -> Vec<String> {
        let words = repair(text.split_whitespace());

        if words.is_empty() {
            return Vec::new();
        }

        self.hypotheses.push_back(words);
        while self.hypotheses.len() > self.agreement_n {
            self.hypotheses.pop_front();
        }

        // A repair reaches words this window already agreed on — the false
        // start is on screen in plain text before the speaker has finished
        // saying the correction — so the agreed prefix is no longer guaranteed
        // to *be* a prefix of the newest hypothesis. Cut it back to where the
        // two still agree, or the replaced word stays on screen forever: every
        // later comparison is made against `committed.len()`, and nothing else
        // ever shortens it.
        let keep = {
            let latest = self.hypotheses.back().expect("just pushed");
            self.committed
                .iter()
                .zip(latest)
                .take_while(|(a, b)| normalize(a) == normalize(b))
                .count()
        };
        self.committed.truncate(keep);

        if self.hypotheses.len() < self.agreement_n {
            return Vec::new();
        }

        let stable = self.common_prefix_len();
        if stable <= self.committed.len() {
            return Vec::new();
        }

        // Surface forms come from the newest hypothesis.
        let latest = self.hypotheses.back().expect("non-empty");
        let fresh: Vec<String> = latest[self.committed.len()..stable].to_vec();
        self.committed.extend_from_slice(&fresh);
        fresh
    }

    /// Words in the current window that are still subject to revision.
    pub fn provisional(&self) -> &[String] {
        match self.hypotheses.back() {
            Some(latest) if latest.len() > self.committed.len() => &latest[self.committed.len()..],
            _ => &[],
        }
    }

    /// Words in the current window the pipeline has stopped revising.
    pub fn committed(&self) -> &[String] {
        &self.committed
    }

    /// The utterance is over: nothing more will arrive to disambiguate the
    /// tail, so agree to it wholesale and close the window.
    ///
    /// This is what rescues utterance-final words, which sliding-window
    /// agreement alone would leave provisional forever.
    ///
    /// It deliberately does **not** file anything into the document. The caller
    /// owns that decision, because an utterance that has merely ended may still
    /// be a hesitation — its audio is held for `--continue-ms` and re-decoded
    /// if the speaker resumes. Filing here would mean un-filing there.
    ///
    /// Returns the finished utterance, empty when the window held no speech the
    /// model was willing to transcribe.
    pub fn finalize_window(&mut self) -> Vec<String> {
        if let Some(latest) = self.hypotheses.back()
            && latest.len() > self.committed.len()
        {
            let tail: Vec<String> = latest[self.committed.len()..].to_vec();
            self.committed.extend_from_slice(&tail);
        }

        self.hypotheses.clear();
        std::mem::take(&mut self.committed)
    }

    /// Seed the document from a recovered session, before anything is spoken.
    ///
    /// Not undoable, and deliberately so: `rollback` takes back things the
    /// speaker did during a session, and this happened before the session
    /// started. Letting an undo reach it would mean "undo" could empty a
    /// transcript the speaker had only just recovered.
    pub fn restore(&mut self, sentences: Vec<Sentence>) {
        if sentences.is_empty() {
            return;
        }
        self.document = sentences;
        self.revision += 1;
    }

    /// File a finished sentence into the document, uncommitted.
    pub fn push_sentence(&mut self, words: Vec<String>) {
        if words.is_empty() {
            return;
        }
        self.document.push(Sentence {
            words,
            committed: false,
        });
        self.revision += 1;
    }

    /// `Luna, commit` — the speaker approves everything settled so far.
    ///
    /// Returns how many sentences this newly approved, so the caller can say so
    /// rather than leaving the speaker guessing whether they were heard.
    pub fn commit_all(&mut self) -> usize {
        let mut n = 0;
        for sentence in &mut self.document {
            if !sentence.committed {
                sentence.committed = true;
                n += 1;
            }
        }
        if n > 0 {
            self.revision += 1;
        }
        n
    }

    /// `Luna, reject` — drop the newest sentence, **whatever its state**.
    ///
    /// Committed sentences are reachable on purpose. Commitment records that
    /// the speaker approved the text, not that the text has been handed to
    /// something that cannot give it back: nothing reaches the scrollback until
    /// the session ends, so there is always something to take back.
    pub fn reject_last(&mut self) -> Option<Sentence> {
        let dropped = self.document.pop()?;
        self.remember(Undo::Rejected(dropped.clone()));
        self.revision += 1;
        Some(dropped)
    }

    /// `Luna, clear` — throw away the whole document.
    ///
    /// Returns how many sentences went. Recoverable with [`rollback`], which is
    /// the only reason a command this blunt is safe to have.
    ///
    /// [`rollback`]: Transcript::rollback
    pub fn clear(&mut self) -> usize {
        if self.document.is_empty() {
            return 0;
        }
        let gone = std::mem::take(&mut self.document);
        let n = gone.len();
        self.remember(Undo::Cleared(gone));
        self.revision += 1;
        n
    }

    /// `Luna, rollback` — put back what the last destructive command removed.
    ///
    /// Returns a description of what came back, for the notice. Only `reject`
    /// and `clear` are recorded: undoing a *commit* would mean taking back an
    /// approval, which removes no text and is not what anyone means when they
    /// ask to undo.
    pub fn rollback(&mut self) -> Option<String> {
        match self.undo.pop()? {
            Undo::Rejected(sentence) => {
                let text = sentence.text();
                self.document.push(sentence);
                self.revision += 1;
                Some(format!("\u{201c}{text}\u{201d}"))
            }
            Undo::Cleared(mut sentences) => {
                let n = sentences.len();
                // Anything said since the clear stays, and stays newest: the
                // restored text was spoken first.
                sentences.append(&mut self.document);
                self.document = sentences;
                self.revision += 1;
                Some(format!("{n} sentence(s)"))
            }
        }
    }

    fn remember(&mut self, entry: Undo) {
        self.undo.push(entry);
        if self.undo.len() > UNDO_DEPTH {
            self.undo.remove(0);
        }
    }

    pub fn document(&self) -> &[Sentence] {
        &self.document
    }

    /// Changes to the document so far. Only ever compared for equality — the
    /// autosave writes when it differs from what it last wrote.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Longest prefix on which every retained hypothesis agrees.
    fn common_prefix_len(&self) -> usize {
        let shortest = self.hypotheses.iter().map(Vec::len).min().unwrap_or(0);

        let first = self.hypotheses.front().expect("non-empty");
        (0..shortest)
            .take_while(|&i| {
                let target = normalize(&first[i]);
                self.hypotheses.iter().all(|h| normalize(&h[i]) == target)
            })
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    fn file(t: &mut Transcript, s: &str) {
        t.push_sentence(words(s));
    }

    fn texts(t: &Transcript) -> Vec<String> {
        t.document().iter().map(Sentence::text).collect()
    }

    fn uncommitted(t: &Transcript) -> usize {
        t.document().iter().filter(|s| !s.committed).count()
    }

    #[test]
    fn commits_only_on_n_way_agreement() {
        let mut t = Transcript::new(3);

        assert!(t.push_hypothesis("the quick").is_empty(), "n=1: too early");
        assert!(t.push_hypothesis("the quick brown").is_empty(), "n=2: too early");

        // The oldest retained hypothesis only reached "quick", so that is as
        // far as three-way agreement extends.
        let fresh = t.push_hypothesis("the quick brown fox");
        assert_eq!(fresh, vec!["the", "quick"]);
        assert_eq!(t.provisional(), ["brown", "fox"]);

        // As the window slides forward, agreement catches up.
        let fresh = t.push_hypothesis("the quick brown fox jumps");
        assert_eq!(fresh, vec!["brown"]);
    }

    #[test]
    fn unstable_tail_is_not_committed() {
        let mut t = Transcript::new(2);
        t.push_hypothesis("i test real time");
        // Tail disagrees, so only the agreed prefix commits.
        let fresh = t.push_hypothesis("i test really");
        assert_eq!(fresh, vec!["i", "test"]);
    }

    #[test]
    fn cosmetic_revision_does_not_block_commit() {
        let mut t = Transcript::new(2);
        t.push_hypothesis("hello world");
        // Case and punctuation changed, but the words did not: commit anyway,
        // and emit the newest surface form.
        let fresh = t.push_hypothesis("Hello, world");
        assert_eq!(fresh, vec!["Hello,", "world"]);
    }

    #[test]
    fn finalize_commits_the_tail() {
        let mut t = Transcript::new(3);
        t.push_hypothesis("hello there");
        let sentence = t.finalize_window();
        assert_eq!(sentence, vec!["hello", "there"]);
        assert!(t.provisional().is_empty());
        assert!(t.committed().is_empty(), "window resets after finalize");
    }

    /// The window and the document are separate layers. An utterance that has
    /// merely ended may still turn out to be a hesitation, so filing it is the
    /// caller's call — made once the grace period closes, not here.
    #[test]
    fn finalizing_a_window_does_not_file_it() {
        let mut t = Transcript::new(2);
        t.push_hypothesis("hello there");
        t.finalize_window();
        assert!(t.document().is_empty(), "the caller files, not the window");
    }

    /// The sidecar returns "" for any buffer under its minimum length, which
    /// includes the first tick of most utterances. Admitting that to the
    /// agreement window drove the common prefix to zero and blanked the
    /// provisional tail off the screen.
    #[test]
    fn an_empty_hypothesis_is_ignored() {
        let mut t = Transcript::new(2);
        t.push_hypothesis("");
        assert!(t.provisional().is_empty(), "nothing to show yet");

        t.push_hypothesis("hello world");
        assert_eq!(t.provisional(), ["hello", "world"]);

        // The empty one must not have consumed a slot in the agreement window,
        // nor reset the prefix once real text is flowing.
        assert_eq!(t.push_hypothesis("hello world"), vec!["hello", "world"]);

        t.push_hypothesis("");
        assert_eq!(
            t.provisional(),
            [] as [String; 0],
            "everything is committed, so nothing is left provisional"
        );
        assert_eq!(t.committed(), ["hello", "world"], "committed text survives");
    }

    #[test]
    fn an_empty_final_pass_still_commits_what_was_agreed() {
        let mut t = Transcript::new(2);
        t.push_hypothesis("keep this");
        t.push_hypothesis("keep this");
        t.push_hypothesis("");
        assert_eq!(t.finalize_window(), vec!["keep", "this"]);
    }

    /// The mid-sentence repair, end to end. The false start is agreed several
    /// ticks before the speaker says the correction, so the repair has to reach
    /// text the window already promoted to plain. Nothing else in this state
    /// machine ever shortens `committed`.
    #[test]
    fn a_repair_takes_back_an_already_committed_word() {
        let mut t = Transcript::new(2);
        t.push_hypothesis("the regularing");
        assert_eq!(t.push_hypothesis("the regularing").len(), 2);
        assert_eq!(t.committed(), ["the", "regularing"]);

        // The speaker says it again, correctly.
        t.push_hypothesis("the regularing regular");
        assert_eq!(t.committed(), ["the"], "the committed false start must go");
        assert_eq!(t.provisional(), ["regular"]);

        t.push_hypothesis("the regularing regular expression");
        assert_eq!(
            t.finalize_window(),
            ["the", "regular", "expression"],
            "the reparandum never reaches the transcript"
        );
    }

    /// The repair sits in the audio, so every tick of the window re-reports it.
    /// Re-deriving it each time has to converge rather than eat the sentence.
    #[test]
    fn a_repair_holds_steady_across_reruns() {
        let mut t = Transcript::new(3);
        for _ in 0..6 {
            t.push_hypothesis("the regularing regular expression is broken.");
        }
        assert_eq!(t.finalize_window(), words("the regular expression is broken."));
    }

    /// A re-decode that no longer hears the false start must put the word back.
    /// Nothing was destroyed, so this costs only a redraw.
    #[test]
    fn a_revised_hypothesis_withdraws_the_repair() {
        let mut t = Transcript::new(2);
        t.push_hypothesis("the regularing regular");
        // The next pass over the same audio heard it differently, and there is
        // no repair any more — only dictation.
        for _ in 0..2 {
            t.push_hypothesis("the popular regular");
        }
        assert_eq!(t.finalize_window(), words("the popular regular"));
    }

    /// Ordinary speech that merely looks like a repair must survive the whole
    /// pipeline, not just the pair rule.
    #[test]
    fn a_plural_meeting_its_verb_survives_commitment() {
        let mut t = Transcript::new(2);
        for _ in 0..3 {
            t.push_hypothesis("These changes change the behavior of the parser.");
        }
        assert_eq!(
            t.finalize_window(),
            words("These changes change the behavior of the parser.")
        );
    }

    #[test]
    fn sentences_are_filed_uncommitted() {
        let mut t = Transcript::new(2);
        file(&mut t, "first one.");
        file(&mut t, "second one.");
        assert_eq!(uncommitted(&t), 2);
        assert!(t.document().iter().all(|s| !s.committed));
    }

    #[test]
    fn an_empty_sentence_is_never_filed() {
        let mut t = Transcript::new(2);
        t.push_sentence(Vec::new());
        assert!(t.document().is_empty());
    }

    #[test]
    fn commit_marks_everything_settled_and_reports_how_much() {
        let mut t = Transcript::new(2);
        file(&mut t, "first one.");
        file(&mut t, "second one.");

        assert_eq!(t.commit_all(), 2);
        assert_eq!(uncommitted(&t), 0);

        // Nothing new to approve, so nothing is reported.
        assert_eq!(t.commit_all(), 0);

        // A sentence spoken afterwards is uncommitted again.
        file(&mut t, "third one.");
        assert_eq!(uncommitted(&t), 1);
        assert_eq!(t.commit_all(), 1);
    }

    /// The requirement that drove the document model: reject reaches *any*
    /// sentence, not just one the pipeline is still holding.
    #[test]
    fn reject_reaches_a_committed_sentence() {
        let mut t = Transcript::new(2);
        file(&mut t, "keep this one.");
        file(&mut t, "drop this one.");
        t.commit_all();

        let dropped = t.reject_last().expect("a sentence to reject");
        assert_eq!(dropped.text(), "drop this one.");
        assert!(dropped.committed, "it really had been committed");
        assert_eq!(texts(&t), vec!["keep this one."]);
    }

    /// Repeated rejects walk backwards, and crossing from uncommitted into
    /// committed text is not a boundary.
    #[test]
    fn rejects_walk_backwards_across_the_commit_boundary() {
        let mut t = Transcript::new(2);
        file(&mut t, "one.");
        file(&mut t, "two.");
        t.commit_all();
        file(&mut t, "three.");

        assert_eq!(t.reject_last().map(|s| s.text()), Some("three.".into()));
        assert_eq!(t.reject_last().map(|s| s.text()), Some("two.".into()));
        assert_eq!(texts(&t), vec!["one."]);
    }

    #[test]
    fn rejecting_an_empty_document_is_a_no_op() {
        let mut t = Transcript::new(2);
        assert_eq!(t.reject_last(), None);
        assert!(t.document().is_empty());
    }

    /// A resumed session must be usable immediately: kept, so `copy` works and
    /// the status line does not claim a pile of unconfirmed text.
    #[test]
    fn restore_seeds_the_document_as_kept() {
        let mut t = Transcript::new(2);
        t.restore(vec![
            Sentence { words: words("From last time."), committed: true },
            Sentence { words: words("And more."), committed: true },
        ]);

        assert_eq!(texts(&t), vec!["From last time.", "And more."]);
        assert_eq!(uncommitted(&t), 0);

        // New dictation lands after it, pending as usual.
        file(&mut t, "Said just now.");
        assert_eq!(uncommitted(&t), 1);
        assert_eq!(
            texts(&t),
            vec!["From last time.", "And more.", "Said just now."]
        );
    }

    /// `rollback` takes back what the speaker did *this* session. Letting it
    /// reach the seed would mean one "undo" emptying a transcript they had only
    /// just recovered.
    #[test]
    fn rollback_cannot_reach_a_restored_session() {
        let mut t = Transcript::new(2);
        t.restore(vec![Sentence { words: words("Recovered."), committed: true }]);
        assert_eq!(t.rollback(), None);
        assert_eq!(texts(&t), vec!["Recovered."]);
    }

    #[test]
    fn clear_takes_the_whole_document_and_reports_the_count() {
        let mut t = Transcript::new(2);
        file(&mut t, "one.");
        file(&mut t, "two.");
        t.commit_all();
        file(&mut t, "three.");

        assert_eq!(t.clear(), 3, "committed and pending alike");
        assert!(t.document().is_empty());
        // Nothing left to take, and nothing recorded for it either.
        assert_eq!(t.clear(), 0);
    }

    /// The reason a command as blunt as `clear` is safe to have at all.
    #[test]
    fn rollback_restores_a_clear() {
        let mut t = Transcript::new(2);
        file(&mut t, "one.");
        file(&mut t, "two.");
        t.commit_all();
        t.clear();

        assert_eq!(t.rollback(), Some("2 sentence(s)".into()));
        assert_eq!(texts(&t), vec!["one.", "two."]);
        assert!(
            t.document().iter().all(|s| s.committed),
            "commit state comes back with the text"
        );
    }

    #[test]
    fn rollback_restores_a_reject() {
        let mut t = Transcript::new(2);
        file(&mut t, "keep.");
        file(&mut t, "oops.");
        t.reject_last();

        assert_eq!(t.rollback(), Some("\u{201c}oops.\u{201d}".into()));
        assert_eq!(texts(&t), vec!["keep.", "oops."]);
    }

    /// Repeated rejects have to be walkable all the way back out.
    #[test]
    fn rollback_unwinds_several_rejects_in_order() {
        let mut t = Transcript::new(2);
        for s in ["one.", "two.", "three."] {
            file(&mut t, s);
        }
        t.reject_last();
        t.reject_last();
        assert_eq!(texts(&t), vec!["one."]);

        t.rollback();
        t.rollback();
        assert_eq!(texts(&t), vec!["one.", "two.", "three."]);
        assert_eq!(t.rollback(), None, "nothing left to undo");
    }

    /// Text spoken after a clear must survive the undo, and stay newest — the
    /// restored sentences were spoken first.
    #[test]
    fn rollback_after_a_clear_keeps_what_was_said_since() {
        let mut t = Transcript::new(2);
        file(&mut t, "before.");
        t.clear();
        file(&mut t, "after.");

        t.rollback();
        assert_eq!(texts(&t), vec!["before.", "after."]);
    }

    /// A commit removes no text, so there is nothing for an undo to restore
    /// and it must not consume the entry that has.
    #[test]
    fn commit_is_not_something_rollback_takes_back() {
        let mut t = Transcript::new(2);
        file(&mut t, "one.");
        file(&mut t, "two.");
        t.reject_last();
        t.commit_all();

        assert_eq!(t.rollback(), Some("\u{201c}two.\u{201d}".into()));
        assert_eq!(texts(&t), vec!["one.", "two."]);
    }

    #[test]
    fn rollback_on_a_fresh_session_is_a_no_op() {
        let mut t = Transcript::new(2);
        assert_eq!(t.rollback(), None);
    }

    /// The autosave writes when this moves, so every mutation has to move it
    /// and nothing else may.
    #[test]
    fn the_revision_tracks_every_document_change() {
        let mut t = Transcript::new(2);
        let mut seen = t.revision();
        let mut bumped = |t: &Transcript, what: &str| {
            assert!(t.revision() > seen, "{what} must bump the revision");
            seen = t.revision();
        };

        file(&mut t, "one.");
        bumped(&t, "push");
        t.commit_all();
        bumped(&t, "commit");
        t.reject_last();
        bumped(&t, "reject");
        t.rollback();
        bumped(&t, "rollback");
        t.clear();
        bumped(&t, "clear");

        // Reading is not a change, and neither is a no-op command.
        let quiet = t.revision();
        t.push_hypothesis("still speaking");
        let _ = t.document();
        t.commit_all();
        t.clear();
        assert_eq!(t.revision(), quiet, "no-ops must not force a write");
    }

    /// Rejecting must not disturb the utterance being spoken right now — the
    /// command arrives in its own window, and the speaker may carry straight on.
    #[test]
    fn a_reject_leaves_the_live_window_alone() {
        let mut t = Transcript::new(2);
        file(&mut t, "already filed.");
        t.push_hypothesis("still speaking");
        t.push_hypothesis("still speaking");

        t.reject_last();
        assert_eq!(t.committed(), ["still", "speaking"]);
    }
}
