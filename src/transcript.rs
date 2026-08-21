//! Transcript state machine: agreement over hypotheses, and the document they
//! commit into.
//!
//! Two layers, deliberately separate.
//!
//! The **window** applies LocalAgreement-n: a word is promoted from provisional
//! to agreed once it appears in the longest common prefix of the last `n`
//! hypotheses. Words are compared in normalized form so a punctuation-only
//! revision does not stall agreement, and surface forms are always taken from
//! the newest hypothesis.
//!
//! The **document** holds finished sentences. It is the deliverable, and stays
//! mutable until the session ends because spoken commands must be able to reach
//! any sentence in it.
//!
//! # Filing and re-decoding
//!
//! A sentence files while its audio is still in the buffer, so every later
//! decode of that buffer reproduces it. [`Transcript::filed`] records what has
//! been filed and strips it from each incoming hypothesis. Matching is on
//! content rather than word counts, because the decodes being compared cover
//! different spans of audio.
//!
//! # Styling contract
//!
//! `committed` here is a claim about the pipeline: these words are agreed and
//! nothing upstream will revise them. That is what the renderer's dim and plain
//! styling reflects. A speaker deleting their own sentence is not the pipeline
//! changing its mind, so plain text keeps its promise even when a sentence
//! leaves the screen.

use crate::repair::repair;
use crate::text::{ends_sentence, normalize, opens_a_continuation, split_sentences};
use std::collections::VecDeque;

/// A finished sentence in the document.
///
/// Deliberately just the words. There is no per-sentence approval flag any
/// more: it made no sentence any more or less reachable, printable, copyable or
/// recoverable than its neighbours, which is the whole of what this type is
/// for.
#[derive(Debug, Clone, PartialEq, Eq)]
/// One finished sentence in the document.
pub struct Sentence {
    pub words: Vec<String>,
}

impl Sentence {
    /// The sentence as a single space-separated string.
    pub fn text(&self) -> String {
        self.words.join(" ")
    }
}

/// The agreement window and the document it commits into.
pub struct Transcript {
    /// Hypotheses that must agree before a word is promoted.
    agreement_n: usize,
    /// Last `n` hypotheses for the current window, as word lists.
    hypotheses: VecDeque<Vec<String>>,
    /// Agreed within the current window.
    committed: Vec<String>,
    /// Finished sentences, oldest first.
    ///
    /// Unbounded: this is the deliverable, so dropping the front of it would
    /// discard the session. A five-hour session is on the order of a megabyte.
    /// Rendering cost is bounded by the renderer instead, which walks back from
    /// the newest sentence and stops when the screen is full.
    document: Vec<Sentence>,
    /// What the last text-removing commands took away, newest last.
    undo: Vec<Undo>,
    /// Incremented on every document change, so persistence can detect movement
    /// without diffing or cloning.
    revision: u64,
    /// Words already filed from audio still in the buffer, oldest first.
    ///
    /// A sentence files while its audio remains in the window, so every later
    /// decode reproduces it. Stripping the overlap between this and each fresh
    /// decode is what stops it being filed twice.
    ///
    /// Stored as content rather than a word count, because the decodes being
    /// compared cover different spans of audio and a count would index into
    /// only one of them.
    filed: VecDeque<String>,
    /// Total words ever filed, counted rather than retained.
    ///
    /// Monotone, which is what the buffer trim needs: a refused cut that
    /// stranded `n` words cannot become acceptable until at least `n` further
    /// words are filed, since the stranded words are by construction the next
    /// ones in document order. [`Transcript::revision`] is the wrong clock for
    /// that, because it advances on any filing however small.
    filed_words: u64,
    /// Absolute index of `filed[0]` — how many words have been dropped off the
    /// front of it.
    filed_base: u64,
    /// Absolute index one past the end of each filed sentence, oldest first.
    ///
    /// The trim needs to know not only whether text is filed but whether a cut
    /// falls *between* filed sentences. Cutting inside one leaves its tail in
    /// the buffer with nothing before it, and the tail then decodes as though it
    /// were a sentence of its own.
    filed_ends: VecDeque<u64>,
}

/// Complete sentences held back in the window rather than filed.
///
/// A sentence files once this many complete sentences sit behind it, on the
/// basis that a boundary surviving further speech is one the recogniser will
/// keep.
///
/// One, plus the extra sentence that [`opens_a_continuation`] requires at
/// boundaries which read as a continued clause. Raising this instead makes
/// every boundary wait, which slows the whole session to guard a hazard that
/// occurs at particular boundaries, and filing less often also leaves the trim
/// fewer places it may cut.
///
/// This is the blunt instrument if real dictation still splits sentences.
const KEEP_SENTENCES: usize = 1;

/// Words of overlap that buy one tolerated edit when matching filed text.
///
/// A run shorter than this must match exactly, which stops a coincidence being
/// read as an overlap, while a sentence-length run survives the occasional
/// re-tokenised word that re-decoding produces.
const OVERLAP_TOLERANCE: usize = 8;

/// Filed words retained for the overlap test.
///
/// Need only cover words whose audio is still buffered, which the trim bounds
/// at a few dozen. Set generously, since the cost is a handful of strings and
/// the failure it prevents is a duplicated sentence.
const FILED_MEMORY: usize = 256;

/// One reversible removal.
///
/// Stored as a delta rather than a document snapshot, since the document is
/// unbounded and snapshotting would cost a copy of it per undo step.
///
/// The variants differ in where restored sentences go, which is why `Discarded`
/// and `Cleared` are not one case.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Undo {
    /// One sentence removed from the end of the document.
    Deleted(Sentence),
    /// The newest sentences, restored to the tail so that text preceding them
    /// stays older.
    Discarded(Vec<Sentence>),
    /// The whole document, restored at the front so that anything said since
    /// stays newer.
    Cleared(Vec<Sentence>),
}

/// How many destructive commands can be taken back.
///
/// Deep enough that a run of deletes can always be walked back out, bounded
/// because a [`Undo::Cleared`] entry holds a whole document.
const UNDO_DEPTH: usize = 64;

impl Transcript {
    /// Creates an empty transcript.
    ///
    /// # Parameters
    ///
    /// - `agreement_n`: hypotheses that must agree before a word is promoted,
    ///   clamped to a minimum of two.
    pub fn new(agreement_n: usize) -> Self {
        Self {
            agreement_n: agreement_n.max(2),
            hypotheses: VecDeque::new(),
            committed: Vec::new(),
            document: Vec::new(),
            undo: Vec::new(),
            revision: 0,
            filed: VecDeque::new(),
            filed_words: 0,
            filed_base: 0,
            filed_ends: VecDeque::new(),
        }
    }

    /// Admits a fresh hypothesis for the current window.
    ///
    /// Self-repairs are resolved on every hypothesis, since that pass is pure
    /// and therefore stable under repetition. Commands are not resolved here:
    /// they destroy text, so re-deriving them each tick would empty the
    /// document. They run once, on the finalized utterance.
    ///
    /// Text already filed is stripped first, so nothing downstream needs to
    /// know that filing happened mid-window.
    ///
    /// Agreed surface forms are refreshed from the newest hypothesis. Agreement
    /// compares through [`normalize`], which discards punctuation, so without
    /// this the window could retain a sentence boundary the recogniser had
    /// since withdrawn — and [`Transcript::settled_prefix`] decides what to file
    /// by splitting on exactly that punctuation.
    ///
    /// An empty hypothesis is discarded rather than admitted, since it means
    /// absence of evidence: the sidecar returns one for any buffer below its
    /// minimum length, and admitting it would drive the common prefix to zero
    /// for the next `n` ticks.
    ///
    /// # Returns
    ///
    /// The words newly promoted to agreed, which may be empty.
    pub fn push_hypothesis(&mut self, text: &str) -> Vec<String> {
        let mut words = repair(text.split_whitespace());

        words.drain(..self.overlap(&words));

        if words.is_empty() {
            return Vec::new();
        }

        self.hypotheses.push_back(words);
        while self.hypotheses.len() > self.agreement_n {
            self.hypotheses.pop_front();
        }

        let keep = {
            let latest = self.hypotheses.back().expect("just pushed");
            self.committed
                .iter()
                .zip(latest)
                .take_while(|(a, b)| normalize(a) == normalize(b))
                .count()
        };
        self.committed.truncate(keep);

        {
            let latest = self.hypotheses.back().expect("just pushed");
            for (word, fresh) in self.committed.iter_mut().zip(latest) {
                if word != fresh {
                    word.clone_from(fresh);
                }
            }
        }

        if self.hypotheses.len() < self.agreement_n {
            return Vec::new();
        }

        let stable = self.common_prefix_len();
        if stable <= self.committed.len() {
            return Vec::new();
        }

        let latest = self.hypotheses.back().expect("non-empty");
        let fresh: Vec<String> = latest[self.committed.len()..stable].to_vec();
        self.committed.extend_from_slice(&fresh);
        fresh
    }

    /// Matches a suffix of [`Transcript::filed`] against a prefix of `words`.
    ///
    /// This is the alignment a whole-buffer hypothesis takes: the buffer holds a
    /// tail of the filed audio, and a fresh decode starts at the beginning of
    /// that tail.
    ///
    /// Searched longest-first, because a shorter match is a coincidence when a
    /// longer one exists.
    ///
    /// # Returns
    ///
    /// How many leading words of `words` are already filed, or zero.
    fn overlap(&self, words: &[String]) -> usize {
        let filed: Vec<&String> = self.filed.iter().collect();
        let most = filed.len().min(words.len());
        (1..=most)
            .rev()
            .find_map(|k| {
                let tail = &filed[filed.len() - k..];
                let (took, gave) = align(tail, words);
                (took == k).then_some(gave)
            })
            .unwrap_or(0)
    }

    /// Matches the front of [`Transcript::filed`] against the front of `words`.
    ///
    /// This is the alignment a *head* decode takes, since a head is a prefix of
    /// the buffer and lines up with the start of what has been filed. A suffix
    /// search cannot see that alignment.
    ///
    /// # Returns
    ///
    /// `(filed_consumed, words_consumed)`. The two differ whenever the aligner
    /// absorbed an edit, and callers must not substitute one for the other.
    fn prefix_overlap(&self, words: &[String]) -> (usize, usize) {
        let filed: Vec<&String> = self.filed.iter().collect();
        align(&filed, words)
    }

    /// Drops the first `n` filed words, whose audio has just been cut away.
    ///
    /// [`Transcript::filed`] covers words whose audio is still in the buffer, so
    /// a trim must prune it in step with the cut. Otherwise the list starts
    /// earlier than the audio and [`Transcript::prefix_overlap`] can no longer
    /// align at the front.
    pub fn forget_filed_prefix(&mut self, n: usize) {
        let n = n.min(self.filed.len());
        self.filed.drain(..n);
        self.filed_base += n as u64;
        self.forget_stale_ends();
    }

    /// Filed words a head decode covers, whether or not it ends a sentence.
    ///
    /// [`Transcript::spent_by`] is the question asked before a cut. This is what
    /// the forced backstop needs once it has stopped asking: the cut happens
    /// regardless, and these words leave the buffer with the audio behind them.
    pub fn filed_prefix_of(&self, text: &str) -> usize {
        self.prefix_overlap(&repair(text.split_whitespace())).0
    }

    /// Reports how many filed words a proposed cut would spend, or `None` if
    /// the cut must not be made.
    ///
    /// # Parameters
    ///
    /// - `text`: what the audio ahead of the proposed cut decodes to on its own.
    ///
    /// # Returns
    ///
    /// `Some(k)` only when both conditions hold:
    ///
    /// 1. Every word of `text` is already filed, so the cut can lose no text.
    /// 2. The cut lands exactly where a filed sentence ended, so it cannot
    ///    strand the second half of one.
    ///
    /// The second condition is what the first cannot supply. A head decoding to
    /// `It tries to solve the problem.` against a filed `It tries to solve the
    /// problem of coordination.` satisfies the first and is precisely the cut
    /// that must not be made, because the rest of that sentence remains in the
    /// buffer and would later decode as though it stood alone.
    ///
    /// A head that decodes to nothing yields `Some(0)`: there is no text to
    /// lose, so the audio is spent.
    pub fn spent_by(&self, text: &str) -> Option<usize> {
        let words = repair(text.split_whitespace());
        if words.is_empty() {
            return Some(0);
        }

        let (took_filed, took_words) = self.prefix_overlap(&words);
        if took_words < words.len() {
            return None;
        }
        let ends_a_sentence = self
            .filed_ends
            .contains(&(self.filed_base + took_filed as u64));
        ends_a_sentence.then_some(took_filed)
    }

    /// Counts words of a fresh decode that the document does not already hold.
    ///
    /// Zero means the audio behind the decode is spent: nothing is lost by
    /// discarding it and nothing new would be filed by keeping it.
    ///
    /// Stripped exactly as [`Transcript::push_hypothesis`] strips, so the two
    /// agree by construction rather than by being kept in step.
    pub fn unfiled(&self, text: &str) -> usize {
        let words = repair(text.split_whitespace());
        words.len() - self.covered(&words)
    }

    /// Leading words of a decode that are already filed, under whichever
    /// alignment matches better.
    ///
    /// Both occur. `filed` normally starts where the buffer starts, which suits
    /// a head decode, but a path that files without pruning can leave it
    /// starting earlier, in which case the buffer aligns with its tail. Both
    /// are anchored matches against text known to be in the buffer rather than
    /// searches for a coincidence, so taking the larger answer is safe.
    fn covered(&self, words: &[String]) -> usize {
        self.prefix_overlap(words).1.max(self.overlap(words))
    }

    /// Counts leading agreed words that are safe to file now.
    ///
    /// Only agreed words are eligible, since provisional words are still moving
    /// and filing is irreversible. See [`settled_words`] for the rule.
    pub fn settled_prefix(&self) -> usize {
        settled_words(&self.committed)
    }
}

/// Aligns two word runs from their fronts, tolerating a few edits.
///
/// Tolerates substitutions, insertions and deletions. A positional comparison
/// handles only substitutions: one inserted word shifts every later position,
/// so every later pair disagrees and the match collapses from complete to
/// nothing. Filed text then goes unrecognised, is not stripped, and is filed a
/// second time, arriving split where the strip should have begun.
///
/// The edit budget is proportional to the shorter run and floors at zero, so
/// short runs must match exactly. This confirms a match already believed to be
/// present rather than searching for a coincidental one, and over a handful of
/// words a single free edit would match almost anything.
///
/// # Returns
///
/// `(a_consumed, b_consumed)` at the last position where the two agreed
/// outright. These differ once an edit has been absorbed, which is the reason
/// both are reported. A run ending in an edit reports the match before it, so a
/// wrong guess costs alignment length rather than falsely claiming text is
/// filed.
fn align(a: &[&String], b: &[String]) -> (usize, usize) {
    let budget = a.len().min(b.len()) / OVERLAP_TOLERANCE;
    let same = |x: &str, y: &str| normalize(x) == normalize(y);

    let (mut i, mut j, mut spent) = (0usize, 0usize, 0usize);
    let (mut ai, mut bj) = (0usize, 0usize);

    while i < a.len() && j < b.len() {
        if same(a[i], &b[j]) {
            i += 1;
            j += 1;
            ai = i;
            bj = j;
            continue;
        }
        if spent == budget {
            break;
        }
        let a_has_extra_word = i + 1 < a.len() && same(a[i + 1], &b[j]);
        let b_has_extra_word = j + 1 < b.len() && same(a[i], &b[j + 1]);

        if a_has_extra_word {
            i += 1;
        } else if b_has_extra_word {
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
        spent += 1;
    }

    (ai, bj)
}

/// Measures how far into `whole` the audio behind `head` reaches, rounded up to
/// a sentence boundary.
///
/// Used by the forced trim, which must cut regardless of whether the text is
/// filed. The cut is unavoidable there, but filing the severed head decode is
/// not: `whole` is a decode of the entire buffer, made with full right-context,
/// and `head` is only needed to locate the cut within it.
///
/// Rounds up because rounding down would file less text than the discarded
/// audio carries, losing the words in between outright. Rounding up files text
/// whose audio is still buffered, which the overlap test already handles.
///
/// # Returns
///
/// A word count into `whole`, or zero when the two decodes cannot be aligned
/// well enough to place the cut.
pub fn words_covering(whole: &[String], head: &[String]) -> usize {
    if whole.is_empty() || head.is_empty() {
        return 0;
    }

    let refs: Vec<&String> = whole.iter().collect();
    let (reaches, matched) = align(&refs, head);
    if matched * 2 < head.len() {
        return 0;
    }

    let text = whole.join(" ");
    let mut seen = 0usize;
    for part in split_sentences(&text) {
        seen += part.split_whitespace().count();
        if seen >= reaches {
            return seen.min(whole.len());
        }
    }
    whole.len()
}

/// Counts leading words of a finished word list that are safe to file.
///
/// A sentence is settled once [`KEEP_SENTENCES`] complete sentences follow it,
/// on the basis that a boundary surviving further speech has had all the
/// right-context it will get. A boundary whose next sentence reads as a
/// continued clause requires one more; see [`opens_a_continuation`].
///
/// The requirement is charged per boundary rather than accumulated across the
/// walk, so a run of such boundaries does not stall filing: an earlier boundary
/// already has the later sentences behind it.
///
/// An incomplete trailing sentence does not count toward the requirement.
///
/// Free-standing because two callers need it over different lists: the
/// agreement window, and a finalized utterance, which never passes through the
/// window at all.
///
/// # Returns
///
/// A word count, or zero when nothing is settled.
pub fn settled_words(committed: &[String]) -> usize {
    {
        let complete = committed.last().is_some_and(|w| ends_sentence(w));

        let text = committed.join(" ");
        let sentences = split_sentences(&text);
        let keep = if complete {
            KEEP_SENTENCES
        } else {
            KEEP_SENTENCES + 1
        };

        let mut cut = sentences.len().saturating_sub(keep);
        while cut > 0 {
            let needed = keep + usize::from(opens_a_continuation(sentences[cut]));
            if sentences.len() - cut >= needed {
                break;
            }
            cut -= 1;
        }
        if cut == 0 {
            return 0;
        }

        sentences[..cut]
            .iter()
            .map(|s| s.split_whitespace().count())
            .sum()
    }
}

impl Transcript {
    /// Files the first `n` agreed words as whole sentences.
    ///
    /// `n` comes from [`Transcript::settled_prefix`], with the caller free to
    /// refuse in between: filing must not happen while the text names the
    /// assistant, since a command has to be parsed from a finalized utterance
    /// rather than a hypothesis.
    ///
    /// The words move out of the agreement window and into the filed record, so
    /// every later decode of the same audio has them stripped.
    ///
    /// Clears the hypothesis deque. Retained hypotheses were stripped against a
    /// shorter filed record than the next one will be, so leaving them would
    /// compare sequences that no longer begin at the same word. The cost is
    /// `agreement_n` ticks before the next promotion, paid only when a sentence
    /// files.
    ///
    /// # Returns
    ///
    /// How many sentences were filed.
    pub fn file_settled(&mut self, n: usize) -> usize {
        let n = n.min(self.committed.len());
        if n == 0 {
            return 0;
        }

        let text = self.committed.drain(..n).collect::<Vec<_>>().join(" ");

        self.hypotheses.clear();

        let mut filed = 0;
        for part in split_sentences(&text) {
            let words: Vec<String> = part.split_whitespace().map(str::to_string).collect();
            if !words.is_empty() {
                self.push_sentence(words);
                filed += 1;
            }
        }

        filed
    }

    /// Forgets all filed text, because the audio behind it has been discarded.
    ///
    /// Called when a window is abandoned rather than trimmed. Without it, a
    /// later passage opening with the words an earlier one closed on would have
    /// them silently stripped.
    pub fn forget_filed(&mut self) {
        self.filed.clear();
        self.filed_base = self.filed_words;
        self.filed_ends.clear();
    }

    /// Discards sentence boundaries that no longer describe retained words.
    fn forget_stale_ends(&mut self) {
        while self.filed_ends.front().is_some_and(|&e| e <= self.filed_base) {
            self.filed_ends.pop_front();
        }
    }

    /// Words in the current window still subject to revision.
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

    /// Closes the window, agreeing to whatever remains provisional.
    ///
    /// Nothing further will arrive to disambiguate the tail, so this is what
    /// rescues utterance-final words that sliding-window agreement alone would
    /// leave provisional indefinitely.
    ///
    /// Deliberately files nothing. The caller owns that decision, because an
    /// utterance that has merely ended may still be a hesitation whose audio is
    /// held and re-decoded if the speaker resumes.
    ///
    /// # Returns
    ///
    /// The finished utterance, empty when the window held no transcribable
    /// speech.
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

    /// Seeds the document from a recovered session, before anything is spoken.
    ///
    /// Deliberately not undoable: rollback reverses what the speaker did during
    /// a session, and this happened before it began. Letting undo reach it
    /// would allow one command to empty a just-recovered transcript.
    pub fn restore(&mut self, sentences: Vec<Sentence>) {
        if sentences.is_empty() {
            return;
        }
        self.document = sentences;
        self.revision += 1;
    }

    /// Files one finished sentence into the document.
    ///
    /// Every word filed here is also recorded for the overlap test, which must
    /// hold for every filing path rather than only logical commitment: a head
    /// decode can transcribe a whole sentence from the part of it that was in
    /// the head, leaving the remainder in the buffer to be decoded again.
    ///
    /// Empty sentences are ignored.
    pub fn push_sentence(&mut self, words: Vec<String>) {
        if words.is_empty() {
            return;
        }
        for word in &words {
            self.filed.push_back(word.clone());
        }
        self.filed_words += words.len() as u64;
        self.filed_ends.push_back(self.filed_words);
        while self.filed.len() > FILED_MEMORY {
            self.filed.pop_front();
            self.filed_base += 1;
        }
        self.forget_stale_ends();
        self.document.push(Sentence { words });
        self.revision += 1;
    }

    /// Removes the newest sentence in the document.
    ///
    /// Every sentence stays reachable for the whole session, which is why
    /// nothing reaches the scrollback until it ends.
    ///
    /// # Returns
    ///
    /// The removed sentence, or `None` when the document is empty.
    pub fn delete_last(&mut self) -> Option<Sentence> {
        let dropped = self.document.pop()?;
        self.remember(Undo::Deleted(dropped.clone()));
        self.revision += 1;
        Some(dropped)
    }

    /// Removes the newest `n` sentences as one undoable unit.
    ///
    /// `n` is supplied by the caller rather than discovered here, and that is
    /// the whole safety argument: `main.rs::apply` passes the newest sentence
    /// of the *current utterance* and nothing else, so this can never walk
    /// backwards into settled text however many times it is said. That is what
    /// makes it safe to say without looking at the screen — unlike `delete`,
    /// which reaches whatever is newest in the document.
    ///
    /// It still takes a count rather than assuming one, because the caller is
    /// where the scope lives: `n` is clamped to what the utterance actually
    /// filed, so a `discard` with nothing in flight removes nothing rather than
    /// falling through to the transcript.
    ///
    /// One undo entry for the whole group, because it was one instruction.
    /// Returns how many went.
    pub fn discard_last(&mut self, n: usize) -> usize {
        let n = n.min(self.document.len());
        if n == 0 {
            return 0;
        }
        let gone = self.document.split_off(self.document.len() - n);
        self.remember(Undo::Discarded(gone));
        self.revision += 1;
        n
    }

    /// Removes the entire document.
    ///
    /// Safe to offer only because [`Transcript::rollback`] restores it.
    ///
    /// # Returns
    ///
    /// How many sentences were removed.
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

    /// Restores what the last text-removing command took away.
    ///
    /// Only removals are recorded. Commands that leave the document unchanged
    /// must not consume an entry that has a sentence behind it.
    ///
    /// # Returns
    ///
    /// A description of what was restored, for reporting, or `None` when there
    /// is nothing to undo.
    pub fn rollback(&mut self) -> Option<String> {
        match self.undo.pop()? {
            Undo::Deleted(sentence) => {
                let text = sentence.text();
                self.document.push(sentence);
                self.revision += 1;
                Some(format!("\u{201c}{text}\u{201d}"))
            }
            Undo::Discarded(mut sentences) => {
                let n = sentences.len();
                self.document.append(&mut sentences);
                self.revision += 1;
                Some(format!("{n} sentence(s)"))
            }
            Undo::Cleared(mut sentences) => {
                let n = sentences.len();
                sentences.append(&mut self.document);
                self.document = sentences;
                self.revision += 1;
                Some(format!("{n} sentence(s)"))
            }
        }
    }

    /// Records a removal for undo, discarding the oldest entry past
    /// [`UNDO_DEPTH`].
    fn remember(&mut self, entry: Undo) {
        self.undo.push(entry);
        if self.undo.len() > UNDO_DEPTH {
            self.undo.remove(0);
        }
    }

    /// The filed sentences, oldest first.
    pub fn document(&self) -> &[Sentence] {
        &self.document
    }

    /// Document version, compared only for equality.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Total words filed since the session began. Monotone.
    pub fn filed_words(&self) -> u64 {
        self.filed_words
    }

    /// Length of the longest prefix on which every retained hypothesis agrees,
    /// compared through [`normalize`].
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

    /// Agree on `text` by repeating it until LocalAgreement promotes it, then
    /// file whatever logical commitment says is settled.
    fn agree_and_commit(t: &mut Transcript, text: &str) -> usize {
        for _ in 0..4 {
            t.push_hypothesis(text);
        }
        let n = t.settled_prefix();
        t.file_settled(n)
    }

    /// Nothing files until a sentence has [`KEEP_SENTENCES`] complete sentences
    /// behind it. The rule counts sentences rather than measuring a duration,
    /// because the recogniser punctuates fragments and terminal punctuation
    /// alone therefore says nothing.
    #[test]
    fn a_sentence_files_once_enough_others_follow_it() {
        let sentences = |n: usize| {
            (1..=n).map(|i| format!("S{i}.")).collect::<Vec<_>>().join(" ")
        };

        let mut t = Transcript::new(3);
        assert_eq!(
            agree_and_commit(&mut t, &sentences(KEEP_SENTENCES)),
            0,
            "nothing behind them yet"
        );
        assert_eq!(texts(&t), Vec::<String>::new());

        let mut t = Transcript::new(3);
        assert_eq!(agree_and_commit(&mut t, &sentences(KEEP_SENTENCES + 1)), 1);
        assert_eq!(texts(&t), vec!["S1."], "the newest stay in the window");

        let mut t = Transcript::new(3);
        assert_eq!(agree_and_commit(&mut t, &sentences(KEEP_SENTENCES + 2)), 2);
        assert_eq!(texts(&t), vec!["S1.", "S2."]);
    }

    /// Replays the hypothesis sequence a real session produced: the recogniser
    /// ends a sentence at a hesitation, then withdraws that boundary once it
    /// hears the continuation. Filing in between splits one spoken sentence
    /// into two permanent entries, the second beginning in lower case.
    ///
    /// Exercises both halves of the fix: the window must not retain a boundary
    /// no live hypothesis claims, and one complete sentence of right-context is
    /// not enough to trust a boundary with.
    #[test]
    fn a_boundary_the_model_later_withdraws_is_not_filed() {
        let mut t = Transcript::new(3);

        let tick = |t: &mut Transcript, text: &str| {
            t.push_hypothesis(text);
            let n = t.settled_prefix();
            t.file_settled(n);
        };

        for _ in 0..3 {
            tick(&mut t, "But the most important feature here is that it works smoothly.");
        }
        for _ in 0..3 {
            tick(
                &mut t,
                "But the most important feature here is that it works smoothly. \
                 And accurately, and knows when to commit the messages, so the buffer does not overflow.",
            );
        }
        for _ in 0..3 {
            tick(
                &mut t,
                "But the most important feature here is that it works smoothly and accurately, \
                 and knows when to commit the messages, so the buffer does not overflow. \
                 Also, take a look at the current transcript.",
            );
        }

        let all: Vec<String> = texts(&t)
            .into_iter()
            .chain(std::iter::once(t.committed().join(" ")))
            .collect();

        assert!(
            !all.iter().any(|s| s.starts_with("and ")),
            "a remainder was filed as its own sentence: {all:?}"
        );
        assert!(
            all.iter()
                .any(|s| s.contains("works smoothly and accurately")),
            "the sentence must survive whole: {all:?}"
        );
        assert!(
            !all.iter().any(|s| s.ends_with("works smoothly.")),
            "the withdrawn boundary must not have been filed: {all:?}"
        );
    }

    /// A run of continuation-shaped boundaries must not stall commitment, or a
    /// speaker who strings clauses together would file nothing, leaving the trim
    /// nowhere to cut and the buffer to grow until the forced backstop severs a
    /// sentence.
    #[test]
    fn a_run_of_continuations_still_commits() {
        let mut t = Transcript::new(3);
        for _ in 0..4 {
            t.push_hypothesis("First one. And a second. And a third. And a fourth.");
        }
        let n = t.settled_prefix();
        assert!(n > 0, "commitment must not stall on chained clauses");
        t.file_settled(n);
        assert_eq!(texts(&t), vec!["First one.", "And a second."]);
    }

    /// Agreement compares through `normalize`, so a punctuation-only revision
    /// never withdraws an agreed word. Without refreshing surface forms the
    /// window retains a boundary no live hypothesis claims, and filing splits on
    /// exactly that punctuation.
    #[test]
    fn a_revised_punctuation_mark_reaches_the_committed_window() {
        let mut t = Transcript::new(3);
        for _ in 0..3 {
            t.push_hypothesis("It works smoothly.");
        }
        assert_eq!(t.committed().join(" "), "It works smoothly.");

        for _ in 0..3 {
            t.push_hypothesis("It works smoothly, and accurately.");
        }
        assert_eq!(
            t.committed().join(" "),
            "It works smoothly, and accurately.",
            "the boundary the model withdrew must not survive in the window"
        );
    }

    /// The in-progress tail is not a complete sentence and must not be counted
    /// as one, or every filed sentence is one sentence short of the
    /// right-context it was promised.
    #[test]
    fn an_unfinished_tail_does_not_count_as_right_context() {
        let mut t = Transcript::new(3);
        let finished = agree_and_commit(&mut t, "One. Two. Three. Four.");

        let mut u = Transcript::new(3);
        let unfinished = agree_and_commit(&mut u, "One. Two. Three. Four and");
        assert_eq!(
            unfinished,
            finished - 1,
            "an in-progress tail must not be counted as a complete sentence"
        );
    }

    /// Filed text whose audio is still in the buffer comes back on every later
    /// decode. It must be stripped, or the document grows a copy per tick.
    #[test]
    fn re_decoding_filed_audio_does_not_file_it_again() {
        let mut t = Transcript::new(3);
        let filed = agree_and_commit(&mut t, "One. Two. Three. Four.");
        let once = texts(&t);
        assert_eq!(once.len(), filed, "precondition: something filed");

        for _ in 0..8 {
            t.push_hypothesis("One. Two. Three. Four.");
            let n = t.settled_prefix();
            t.file_settled(n);
        }
        assert_eq!(
            texts(&t),
            once,
            "filed once, however often the audio is re-decoded"
        );
    }

    /// A single re-tokenised word must not defeat the overlap. Measured, the
    /// model returns `3` and `three` for the same audio on different passes,
    /// and an exact run match collapses to zero on it — which filed two whole
    /// sentences a second time.
    #[test]
    fn the_overlap_survives_a_re_tokenised_word() {
        let mut t = Transcript::new(3);
        agree_and_commit(&mut t, "There was a spike around 3 in the afternoon. Two. Three. Four.");
        let after_first = texts(&t);
        assert!(
            after_first
                .first()
                .is_some_and(|s| s.contains("spike around 3")),
            "precondition: the spike sentence filed with the numeral: {after_first:?}"
        );

        for _ in 0..4 {
            t.push_hypothesis("There was a spike around three in the afternoon. Two. Three. Four.");
            let n = t.settled_prefix();
            t.file_settled(n);
        }
        assert_eq!(texts(&t), after_first, "the spike sentence must not be filed twice");
    }

    /// One inserted word must not hide filed text. A positional comparison
    /// shifts at the insertion and matches nothing after it, so the filed text
    /// goes unstripped and is filed a second time, split where the strip should
    /// have begun.
    #[test]
    fn an_inserted_word_does_not_hide_filed_text() {
        let mut t = Transcript::new(3);
        file(
            &mut t,
            "What is important to know is how exactly the algorithm commits the messages.",
        );

        let seen = "What is too important to know is how exactly the algorithm commits the messages.";
        assert_eq!(
            t.unfiled(seen),
            0,
            "an inserted word must not make filed text look new"
        );
    }

    /// A dropped word is the same hazard in the other direction.
    #[test]
    fn a_dropped_word_does_not_hide_filed_text() {
        let mut t = Transcript::new(3);
        file(
            &mut t,
            "What is too important to know is how exactly the algorithm commits the messages.",
        );
        let seen = "What is important to know is how exactly the algorithm commits the messages.";
        assert_eq!(t.unfiled(seen), 0);
    }

    /// The tolerance must not turn into a licence to match anything. Two
    /// genuinely different sentences of the same length share nothing, and a
    /// budget that let them align would discard audio holding unfiled speech.
    #[test]
    fn unrelated_text_is_never_read_as_filed() {
        let mut t = Transcript::new(3);
        file(
            &mut t,
            "The deployment finished at noon and everything looked stable today.",
        );
        let seen = "My cat is asleep on the keyboard again this afternoon somehow.";
        assert_eq!(
            t.unfiled(seen),
            seen.split_whitespace().count(),
            "nothing here is filed"
        );
    }

    /// …but a short run still has to match exactly, or a decode that merely
    /// opens on the word the last filed sentence closed on loses that word.
    #[test]
    fn a_short_overlap_is_not_forgiven_a_mismatch() {
        let mut t = Transcript::new(3);
        file(&mut t, "Done.");
        assert_eq!(t.overlap(&words("Different words entirely")), 0);
        assert_eq!(t.overlap(&words("Done. and so on")), 1);
    }

    /// Anything filed by *any* path has to be remembered, not just logical
    /// commitment. The trim files through `push_sentence`, and its head decode
    /// can transcribe a whole sentence from the part of it that was in the head
    /// — leaving the rest in the retained tail to be decoded again.
    #[test]
    fn text_filed_by_the_trim_is_also_protected() {
        let mut t = Transcript::new(3);
        file(&mut t, "The database migration is scheduled for Saturday morning.");

        t.push_hypothesis("The database migration is scheduled for Saturday morning. We will need someone.");
        assert_eq!(
            t.committed().len() + t.provisional().len(),
            words("We will need someone.").len(),
            "the filed sentence must be stripped, leaving only what follows it"
        );
    }

    /// The predicate the trim cuts on. Zero means the audio behind the head is
    /// spent and severing it can lose nothing; anything else means the cut would
    /// strand unfiled text.
    #[test]
    fn unfiled_counts_what_a_cut_would_strand() {
        let mut t = Transcript::new(3);
        file(&mut t, "It tries to solve the problem of coordination.");

        assert_eq!(
            t.unfiled("It tries to solve the problem of coordination."),
            0,
            "a head whose text is entirely filed is safe to cut away"
        );
        assert_eq!(
            t.unfiled("It tries to solve the problem of coordination. A single node can talk."),
            5,
            "text past the filed prefix is what a cut would strand"
        );
        assert_eq!(
            t.unfiled("A single node can talk."),
            5,
            "nothing in common with the document means nothing is spent"
        );
        assert_eq!(t.unfiled(""), 0, "a head that decoded to nothing is spent too");
    }

    /// Once the audio is gone nothing can re-decode it, so the memory has to be
    /// dropped — otherwise a later passage opening on the words an earlier one
    /// closed with would have them silently eaten.
    #[test]
    fn forgetting_filed_audio_stops_the_overlap_reaching_a_new_passage() {
        let mut t = Transcript::new(3);
        file(&mut t, "Thank you.");
        assert_eq!(t.overlap(&words("Thank you very much.")), 2);

        t.forget_filed();
        assert_eq!(t.overlap(&words("Thank you very much.")), 0);
    }

    /// Filing mid-window must not leave the agreement deque holding hypotheses
    /// that were stripped against a shorter `filed` — their words no longer
    /// start at the same place, and comparing them produced spliced nonsense
    /// like `We will migration is scheduled for Saturday morning.`
    #[test]
    fn filing_restarts_agreement_so_the_deque_stays_aligned() {
        let mut t = Transcript::new(3);
        agree_and_commit(&mut t, "One. Two. Three. Four.");
        assert!(t.hypotheses.is_empty(), "the deque must be cleared on filing");

        for _ in 0..4 {
            t.push_hypothesis("One. Two. Three. Four five.");
        }
        let live = [t.committed(), t.provisional()].concat().join(" ");
        assert!(!live.contains("One"), "filed text must not reappear: {live:?}");
    }

    #[test]
    fn commits_only_on_n_way_agreement() {
        let mut t = Transcript::new(3);

        assert!(t.push_hypothesis("the quick").is_empty(), "n=1: too early");
        assert!(t.push_hypothesis("the quick brown").is_empty(), "n=2: too early");

        let fresh = t.push_hypothesis("the quick brown fox");
        assert_eq!(fresh, vec!["the", "quick"]);
        assert_eq!(t.provisional(), ["brown", "fox"]);

        let fresh = t.push_hypothesis("the quick brown fox jumps");
        assert_eq!(fresh, vec!["brown"]);
    }

    #[test]
    fn unstable_tail_is_not_committed() {
        let mut t = Transcript::new(2);
        t.push_hypothesis("i test real time");
        let fresh = t.push_hypothesis("i test really");
        assert_eq!(fresh, vec!["i", "test"]);
    }

    #[test]
    fn cosmetic_revision_does_not_block_commit() {
        let mut t = Transcript::new(2);
        t.push_hypothesis("hello world");
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
    fn an_empty_sentence_is_never_filed() {
        let mut t = Transcript::new(2);
        t.push_sentence(Vec::new());
        assert!(t.document().is_empty());
    }

    /// The requirement that drove the document model: delete reaches *any*
    /// sentence, however long ago it was filed, not just one the pipeline is
    /// still holding.
    #[test]
    fn deletes_walk_backwards_through_the_document() {
        let mut t = Transcript::new(2);
        for s in ["one.", "two.", "three."] {
            file(&mut t, s);
        }

        let dropped = t.delete_last().expect("a sentence to delete");
        assert_eq!(dropped.text(), "three.");
        assert_eq!(t.delete_last().map(|s| s.text()), Some("two.".into()));
        assert_eq!(texts(&t), vec!["one."]);
    }

    #[test]
    fn deleting_an_empty_document_is_a_no_op() {
        let mut t = Transcript::new(2);
        assert_eq!(t.delete_last(), None);
        assert!(t.document().is_empty());
    }

    /// A resumed session must be usable immediately, and indistinguishable from
    /// one that was dictated: new speech simply lands after it.
    #[test]
    fn restore_seeds_the_document() {
        let mut t = Transcript::new(2);
        t.restore(vec![
            Sentence { words: words("From last time.") },
            Sentence { words: words("And more.") },
        ]);

        assert_eq!(texts(&t), vec!["From last time.", "And more."]);

        file(&mut t, "Said just now.");
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
        t.restore(vec![Sentence { words: words("Recovered.") }]);
        assert_eq!(t.rollback(), None);
        assert_eq!(texts(&t), vec!["Recovered."]);
    }

    /// `discard` removes the live caption, which is a *group* of sentences, and
    /// one instruction has to be one undo entry — otherwise walking back a
    /// discard of three sentences takes three "undo"s the speaker never asked
    /// for.
    #[test]
    fn discard_takes_a_group_and_undoes_as_one() {
        let mut t = Transcript::new(2);
        for s in ["one.", "two.", "three."] {
            file(&mut t, s);
        }

        assert_eq!(t.discard_last(2), 2);
        assert_eq!(texts(&t), vec!["one."]);

        assert_eq!(t.rollback(), Some("2 sentence(s)".into()));
        assert_eq!(texts(&t), vec!["one.", "two.", "three."]);
        assert_eq!(t.rollback(), None, "one instruction, one entry");
    }

    /// The scope comes from the caller, so this must hold the line at both ends:
    /// told nothing, it removes nothing rather than falling back on the newest
    /// sentence; told more than exists, it clamps rather than panicking.
    #[test]
    fn discard_never_reaches_further_than_it_is_told() {
        let mut t = Transcript::new(2);
        file(&mut t, "settled.");

        assert_eq!(t.discard_last(0), 0, "nothing in flight removes nothing");
        assert_eq!(texts(&t), vec!["settled."]);
        assert_eq!(t.rollback(), None, "and records no undo entry for it");

        assert_eq!(t.discard_last(9), 1, "clamped to the document");
        assert!(t.document().is_empty());
    }

    /// A discard took the *newest* sentences, so they go back on the tail —
    /// where `delete` puts its one sentence, and unlike `clear`, which took
    /// everything and therefore restores at the front.
    #[test]
    fn a_discard_is_restored_to_the_tail() {
        let mut t = Transcript::new(2);
        file(&mut t, "before.");
        file(&mut t, "oops.");

        t.discard_last(1);
        file(&mut t, "since.");
        t.rollback();

        assert_eq!(texts(&t), vec!["before.", "since.", "oops."]);
    }

    #[test]
    fn clear_takes_the_whole_document_and_reports_the_count() {
        let mut t = Transcript::new(2);
        for s in ["one.", "two.", "three."] {
            file(&mut t, s);
        }

        assert_eq!(t.clear(), 3);
        assert!(t.document().is_empty());
        assert_eq!(t.clear(), 0);
    }

    /// The reason a command as blunt as `clear` is safe to have at all.
    #[test]
    fn rollback_restores_a_clear() {
        let mut t = Transcript::new(2);
        file(&mut t, "one.");
        file(&mut t, "two.");
        t.clear();

        assert_eq!(t.rollback(), Some("2 sentence(s)".into()));
        assert_eq!(texts(&t), vec!["one.", "two."]);
    }

    #[test]
    fn rollback_restores_a_delete() {
        let mut t = Transcript::new(2);
        file(&mut t, "keep.");
        file(&mut t, "oops.");
        t.delete_last();

        assert_eq!(t.rollback(), Some("\u{201c}oops.\u{201d}".into()));
        assert_eq!(texts(&t), vec!["keep.", "oops."]);
    }

    /// Repeated deletes have to be walkable all the way back out.
    #[test]
    fn rollback_unwinds_several_deletes_in_order() {
        let mut t = Transcript::new(2);
        for s in ["one.", "two.", "three."] {
            file(&mut t, s);
        }
        t.delete_last();
        t.delete_last();
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
        t.delete_last();
        bumped(&t, "delete");
        t.rollback();
        bumped(&t, "rollback");
        t.discard_last(1);
        bumped(&t, "discard");
        t.rollback();
        bumped(&t, "undo of a discard");
        t.clear();
        bumped(&t, "clear");

        let quiet = t.revision();
        t.push_hypothesis("still speaking");
        let _ = t.document();
        t.clear();
        t.delete_last();
        t.discard_last(3);
        assert_eq!(t.revision(), quiet, "no-ops must not force a write");
    }

    /// Deleting must not disturb the utterance being spoken right now — the
    /// command arrives in its own window, and the speaker may carry straight on.
    #[test]
    fn a_delete_leaves_the_live_window_alone() {
        let mut t = Transcript::new(2);
        file(&mut t, "already filed.");
        t.push_hypothesis("still speaking");
        t.push_hypothesis("still speaking");

        t.delete_last();
        assert_eq!(t.committed(), ["still", "speaking"]);
    }
}
