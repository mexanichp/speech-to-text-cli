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

/// How far through the pipeline a finished sentence has come.
///
/// The two tiers exist because they make different promises. A settled
/// sentence is what the recogniser heard, and the cleanup pass may still
/// re-punctuate or re-segment it. A finalized one has been through that pass
/// and is the deliverable.
///
/// The distinction is visible to the speaker rather than internal, which is the
/// point: without it the screen would claim a permanence the pipeline does not
/// have, since text does move after it goes plain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Transcribed and filed. The cleanup pass has not reached it.
    Settled,
    /// The cleanup pass has been over it. Nothing moves it but the speaker.
    Finalized,
}

/// Whether the last sentence of a cleaned run is finalized with the rest.
///
/// The cleanup pass exists to join a sentence split across a trim seam, and it
/// can only join what it is shown together. Batches used to abut exactly, so a
/// seam falling on a batch boundary put the two halves in different passes and
/// no pass ever saw both: the first half went finalized and out of scope while
/// the second was still settled. Measured on a real session, the speaker got
/// `but it seems.` as one finished sentence and `That it happened exactly
/// between two paragraphs...` as the next, one tier apart, with nothing left
/// that could put them back together.
///
/// [`Tail::Carry`] leaves the last sentence settled instead, so it opens the
/// next batch and every adjacent pair is read together by some pass. It is the
/// cheapest form of "give the pass more context": no larger batch, no larger
/// model, and no finalized text ever edited, which is what including the
/// previous batch's tail as context would have required.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tail {
    /// Finalize the whole run. Nothing follows that could join to it.
    Close,
    /// Hold the last sentence back, so the next batch begins with it.
    Carry,
}

/// A finished sentence in the document.
///
/// The words and how far they have come. There is no per-sentence approval flag
/// any more: it made no sentence any more or less reachable, printable,
/// copyable or recoverable than its neighbours, which is the whole of what this
/// type is for. The tier is different, because it changes what the sentence is
/// allowed to do next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sentence {
    pub words: Vec<String>,
    pub tier: Tier,
    /// Whether a paragraph break belongs before this sentence.
    ///
    /// Set from the one signal the pipeline has that a speaker finished a
    /// thought rather than drew breath: a hold that expired without a
    /// continuation. Everything shorter merges, and merging is what says the
    /// speaker was still going.
    pub paragraph: bool,
    /// Whether the audio behind this sentence was cut off at its end.
    ///
    /// A forced trim cuts the buffer at the best pause available, which is
    /// often not the end of a sentence, so what is filed stops where the audio
    /// stopped and the rest of the thought arrives as the next sentence. The
    /// full stop between them is the recogniser's, not the speaker's.
    ///
    /// The host knows this at the moment it cuts. The cleanup pass, which is
    /// given text and nothing else, has to infer it, and §11 records the case
    /// where it cannot: both halves read as grammatical sentences and nothing
    /// in the words says one of them was severed. This is that knowledge, kept
    /// so it can be handed over.
    pub cut: bool,
}

impl Sentence {
    /// A sentence as the recogniser filed it.
    pub fn settled(words: Vec<String>) -> Self {
        Self { words, tier: Tier::Settled, paragraph: false, cut: false }
    }

    /// A sentence the cleanup pass has finished with.
    pub fn finalized(words: Vec<String>) -> Self {
        Self { words, tier: Tier::Finalized, paragraph: false, cut: false }
    }

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
    /// Set when the next sentence filed should open a paragraph.
    opens_paragraph: bool,
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

/// Words of the document searched for a repeat of incoming text.
///
/// Only the recent tail. A repeat arrives from a decode of audio still in the
/// buffer, and the buffer is bounded, so a match further back than this is a
/// coincidence rather than a re-file.
const REPEAT_WINDOW: usize = 80;

/// Shortest repeat at a trim seam worth removing.
///
/// Below this a match is as likely to be a coincidence as a seam: ordinary
/// English repeats three-word runs all the time.
const SEAM_MIN: usize = 4;

/// Longest repeat looked for at a seam.
const SEAM_MAX: usize = 16;

/// Words of a seam repeat that buy one tolerated mismatch.
///
/// More generous than the budget used for filed text, and deliberately so. The
/// two sides of a seam are decodes of the same speech with different amounts of
/// right-context, so the second reading routinely differs by a word: `so this
/// tool has to recognize` against `So this too has to recognize`. A budget
/// tight enough to reject a coincidence would reject that too.
const SEAM_TOLERANCE: usize = 4;

/// Words of overlap that buy one tolerated edit when matching filed text.
///
/// A sentence-length run survives the occasional re-tokenised word that
/// re-decoding produces, and a longer one survives proportionally more.
const OVERLAP_TOLERANCE: usize = 8;

/// Shortest run that is allowed a single edit regardless of the proportion.
///
/// The proportional budget alone floors at zero below [`OVERLAP_TOLERANCE`]
/// words, which means a filed sentence shorter than that had to be re-decoded
/// *exactly* to be recognised as filed. One inserted word defeated it outright:
/// `So, not sure what's going on there.` was filed at an endpoint, the audio was
/// merged back by a continuation, and the re-decode came back `So I'm not sure
/// what's going on there.` Every alignment failed on the second word, the
/// sentence read as unfiled, and the speaker got it twice.
///
/// Short runs still cannot be matched loosely: below this they must agree
/// outright. What buys the edit is that every caller anchors the match against
/// text believed to be in the buffer already, rather than searching for one.
const OVERLAP_MIN_RUN: usize = 4;

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
            opens_paragraph: false,
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

    /// Matches `words` against the record starting from a filed sentence end.
    ///
    /// The third alignment, and the one the other two cannot express. A head
    /// decode begins where the buffer begins. The record normally begins there
    /// too, but not once an endpoint has filed whole sentences and a
    /// continuation has spliced their audio back: the record then starts
    /// *earlier* than the decode, so the front match misses, and the decode is
    /// shorter than the record's tail, so the whole-tail match misses as well.
    /// The text reads as unfiled and is filed a second time, which is the
    /// duplicate a speaker sees.
    ///
    /// Anchored on the recorded sentence ends rather than on any offset. Those
    /// are the only places a decode of restored audio can begin, and an
    /// unanchored search over ordinary words finds a coincidence sooner or
    /// later.
    fn covered_from_a_boundary(&self, words: &[String]) -> usize {
        let filed: Vec<&String> = self.filed.iter().collect();
        self.filed_ends
            .iter()
            .filter_map(|&end| end.checked_sub(self.filed_base))
            .filter_map(|at| usize::try_from(at).ok())
            .filter(|&at| at < filed.len())
            .map(|at| align(&filed[at..], words).1)
            .max()
            .unwrap_or(0)
    }

    /// Leading words of a decode that are already filed, under whichever
    /// alignment matches best.
    ///
    /// All three occur. `filed` normally starts where the buffer starts, which
    /// suits a head decode; a path that files without pruning can leave it
    /// starting earlier, in which case the buffer aligns with its tail; and an
    /// endpoint followed by a continuation leaves it starting earlier than a
    /// decode that is also shorter than that tail, which only
    /// [`Transcript::covered_from_a_boundary`] can see. Each is an anchored
    /// match against text known to be in the buffer rather than a search for a
    /// coincidence, so taking the largest answer is safe.
    fn covered(&self, words: &[String]) -> usize {
        self.prefix_overlap(words)
            .1
            .max(self.overlap(words))
            .max(self.covered_from_a_boundary(words))
    }

    /// Counts leading agreed words that are safe to file now.
    ///
    /// Only agreed words are eligible, since provisional words are still moving
    /// and filing is irreversible. See [`settled_words`] for the rule.
    pub fn settled_prefix(&self) -> usize {
        settled_words(&self.committed)
    }
}

/// Renders a document as flowing prose.
///
/// Sentences run together with a space and paragraphs break with a blank line,
/// which is what the text has to look like to be pasted into anything that
/// wraps its own lines. One sentence per line is right for a list and wrong for
/// a paragraph, and dictation is mostly paragraphs.
pub fn prose(document: &[Sentence]) -> String {
    let mut out = String::new();
    for sentence in document {
        if !out.is_empty() {
            out.push_str(if sentence.paragraph { "\n\n" } else { " " });
        }
        out.push_str(&sentence.text());
    }
    if !out.is_empty() {
        out.push('\n');
    }
    out
}

/// Aligns two word runs from their fronts, tolerating a few edits.
///
/// Tolerates substitutions, insertions and deletions. A positional comparison
/// handles only substitutions: one inserted word shifts every later position,
/// so every later pair disagrees and the match collapses from complete to
/// nothing. Filed text then goes unrecognised, is not stripped, and is filed a
/// second time, arriving split where the strip should have begun.
///
/// The edit budget is proportional to the shorter run, with one edit allowed
/// from [`OVERLAP_MIN_RUN`] words up. This confirms a match already believed to
/// be present rather than searching for a coincidental one; below that length a
/// single free edit would match almost anything, and above it the commoner
/// failure is the reverse, a filed sentence going unrecognised over one word.
///
/// # Returns
///
/// `(a_consumed, b_consumed)` at the last position where the two agreed
/// outright. These differ once an edit has been absorbed, which is the reason
/// both are reported. A run ending in an edit reports the match before it, so a
/// wrong guess costs alignment length rather than falsely claiming text is
/// filed.
fn align(a: &[&String], b: &[String]) -> (usize, usize) {
    let shorter = a.len().min(b.len());
    let floor = usize::from(shorter >= OVERLAP_MIN_RUN && !crate::ablate::off("edit-floor"));
    let budget = (shorter / OVERLAP_TOLERANCE).max(floor);
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
        let paragraph = std::mem::take(&mut self.opens_paragraph);
        let mut words = words;
        // Order matters, and this is the order. The seam goes first, which
        // takes the repeated words out of the document's tail; anything still
        // matching after that is a stretch the document holds *elsewhere*,
        // which is a different fault with a different answer.
        //
        // The other order does not work: a repeat that runs to the end of the
        // document is a seam, but its shorter prefixes do not, so the stale
        // check claims the seam one word early and truncates the wrong copy.
        //
        // Both run for a sentence opening a paragraph as well. They used to be
        // skipped there, on the reasoning that a paragraph follows a silence
        // long enough that nothing can have been decoded twice across it. That
        // is an argument about the *audio*, and these two guards are about the
        // text the document already holds: the settle files the held words and
        // forgets the filed record in the same breath, so a paragraph opening
        // was the one place in the pipeline with no duplicate defence at all.
        let seam = self.drop_seam_repeat(&words);
        words.drain(..seam.min(words.len()));
        if words.is_empty() {
            return;
        }

        // Asked repeatedly, because one answer can expose the next. Dropping a
        // matched prefix leaves a remainder that is a different question, and
        // on a real recording the remainder was a single word: six words of
        // `First paragraph and second paragraph converge.` arrived behind text
        // that already held five of them, and the one word left over became a
        // sentence reading `Converge.` A stub like that is worse than either
        // outcome it sits between — it keeps none of the meaning and costs a
        // sentence boundary. Each pass drops at least one word, so this ends.
        loop {
            let stale = self.already_in_the_document(&words);
            if stale == 0 {
                break;
            }
            words.drain(..stale.min(words.len()));
            if words.is_empty() {
                return;
            }
            if crate::ablate::off("stale-loop") {
                break;
            }
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
        self.document.push(Sentence { paragraph, ..Sentence::settled(words) });
        self.revision += 1;
    }

    /// Whether two runs of the same length are the same words.
    ///
    /// A few substitutions are allowed in the middle, because the two copies
    /// come from decodes of the same audio with different right-context and
    /// routinely differ by a word. None are allowed at either end.
    ///
    /// The end anchors are what stop a run growing past where it matches. A
    /// proportional budget spent at the tail lets a thirteen-word run with
    /// three wrong words on the end read as a match, and those three words are
    /// exactly the new ones the repeat is followed by.
    fn same_run(a: &[&String], b: &[&String]) -> bool {
        let n = a.len();
        if n == 0 || n != b.len() {
            return false;
        }
        let same = |x: &String, y: &String| normalize(x) == normalize(y);
        same(a[0], b[0])
            && same(a[n - 1], b[n - 1])
            && a.iter().zip(b).filter(|(x, y)| !same(x, y)).count() <= n / SEAM_TOLERANCE
    }

    /// Whether two runs are two decodes of the same speech, at whatever lengths.
    ///
    /// [`Transcript::same_run`] compares position by position and so requires
    /// the two copies to have the same number of words. A seam often does not:
    /// the earlier copy was decoded with no right-context and the model guessed
    /// a word the later copy does not have. Measured on one recording, twice in
    /// the same session — `first paragraph and second paragraph connect,
    /// converge` against `First paragraph and second paragraph converge`, and
    /// `such artifacts like previous one` against `Such artifacts, like the
    /// previous one`. Both read as new text, and the speaker got them twice.
    ///
    /// Both ends are still anchored exactly, which is what stops a run growing
    /// past where it matches, and the budget in between is the same proportion
    /// [`SEAM_TOLERANCE`] sets, spent on edits of any kind rather than on
    /// substitutions alone.
    fn same_speech(a: &[&String], b: &[&String]) -> bool {
        if crate::ablate::off("same-speech") {
            return Self::same_run(a, b);
        }
        let (n, m) = (a.len(), b.len());
        if n == 0 || m == 0 {
            return false;
        }
        let same = |x: &String, y: &String| normalize(x) == normalize(y);
        if !same(a[0], b[0]) || !same(a[n - 1], b[m - 1]) {
            return false;
        }

        let budget = n.min(m) / SEAM_TOLERANCE;
        if n.abs_diff(m) > budget {
            return false;
        }

        // Levenshtein over the two runs, which are bounded by `SEAM_MAX`.
        let mut row: Vec<usize> = (0..=m).collect();
        for (i, x) in a.iter().enumerate() {
            let mut diagonal = row[0];
            row[0] = i + 1;
            for (j, y) in b.iter().enumerate() {
                let cost = usize::from(!same(x, y));
                let next = (row[j] + 1).min(row[j + 1] + 1).min(diagonal + cost);
                diagonal = row[j + 1];
                row[j + 1] = next;
            }
        }
        row[m] <= budget
    }

    /// Leading words of `incoming` that the document already holds.
    ///
    /// A different repeat from the seam one, and it needs a different answer. A
    /// forced trim files from a whole-buffer decode located by the head, and
    /// when the filed-text strip fails to recognise its own output that decode
    /// starts *inside* a sentence already in the document: `To know is how
    /// exactly the algorithm commits the messages` arriving behind a sentence
    /// that already says it. It matches no sentence boundary, so
    /// [`Transcript::covered_from_a_boundary`] cannot see it, and it does not
    /// sit at the document's end, so [`Transcript::drop_seam_repeat`] cannot
    /// either.
    ///
    /// Here the *incoming* copy is the one to drop, the reverse of the seam
    /// case: the words are already in the document in their proper place, and
    /// what follows them in this decode is the only part that is new.
    ///
    /// Searched longest-first over the recent tail only, with the same
    /// substitution budget a seam gets, since the two copies come from decodes
    /// of the same audio with different context.
    ///
    /// Runs *after* [`Transcript::drop_seam_repeat`], which has by then taken
    /// the seam out of the tail. A match ending at the tail's end is excluded
    /// anyway, but only the ordering makes that exclusion reliable: the
    /// shorter prefixes of an end-anchored repeat do not end at the end.
    fn already_in_the_document(&self, incoming: &[String]) -> usize {
        let tail: Vec<&String> = self
            .document
            .iter()
            .flat_map(|s| s.words.iter())
            .rev()
            .take(REPEAT_WINDOW)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        let head: Vec<&String> = incoming.iter().collect();
        let most = incoming.len().min(tail.len());
        if let Some(k) = (SEAM_MIN..=most).rev().find(|&k| {
            // Ending at the tail's end is the seam case, which
            // `drop_seam_repeat` has already dealt with.
            (0..tail.len().saturating_sub(k))
                .any(|i| Self::same_run(&tail[i..i + k], &head[..k]))
        }) {
            return k;
        }

        // Below `SEAM_MIN` no substitution budget is affordable, so nothing
        // above can see a two- or three-word fragment. Those exist: the forced
        // trim files whatever the severed head decoded to, and on a real
        // recording that put `Such artifacts, like.` into the document three
        // words behind `...duplicates, such artifacts like previous one.` One
        // word is enough to bother the speaker and too short for any tolerance,
        // so the whole fragment has to match a run of the tail outright, and
        // only then is it read as something the trim left behind.
        let short = incoming.len() < SEAM_MIN
            && incoming.len() <= tail.len()
            && !crate::ablate::off("short-fragment");
        let repeated = || {
            tail.windows(incoming.len())
                .any(|run| run.iter().zip(incoming).all(|(x, y)| normalize(x) == normalize(y)))
        };
        match short && !incoming.is_empty() && repeated() {
            true => incoming.len(),
            false => 0,
        }
    }

    /// Drops a seam repeat from the tail of the sentence before this one.
    ///
    /// The trim cuts the audio at a pause and the head is decoded on its own, so
    /// its last words were transcribed with no right-context at all, while the
    /// next decode covers the same speech with the rest of the sentence behind
    /// it. Where the two overlap the speaker sees their words twice, spelled
    /// slightly differently the second time.
    ///
    /// The earlier copy is the one dropped, because it is the one decoded blind,
    /// and because dropping the later one would leave the sentence that survives
    /// starting mid-clause.
    ///
    /// Only a settled sentence is edited. A finalized one has been read in full
    /// by the cleanup pass and is no longer the pipeline's to rewrite, so there
    /// the *incoming* copy goes instead and the caller is told how much of it to
    /// drop. That reverses the usual preference deliberately: the rule is to
    /// keep the copy decoded with the most context, and a finalized sentence has
    /// had more context than any decode. Refusing to act at all was the third
    /// option and the wrong one, since nothing else can see this repeat —
    /// [`Transcript::already_in_the_document`] excludes a run ending at the
    /// tail's end, which is exactly what a seam is.
    ///
    /// Compared through [`Transcript::same_run`] rather than [`align`], since a
    /// seam differs by substitution rather than by insertion, and an aligner
    /// tolerant enough to absorb one over four words would match almost
    /// anything.
    ///
    /// # Returns
    ///
    /// Leading words of `incoming` the caller must drop, which is zero whenever
    /// this was able to edit the document itself.
    fn drop_seam_repeat(&mut self, incoming: &[String]) -> usize {
        let Some(prev) = self.document.last_mut() else {
            return 0;
        };

        let tail: Vec<&String> = prev.words.iter().collect();
        let head: Vec<&String> = incoming.iter().collect();
        let most = tail.len().min(head.len()).min(SEAM_MAX);

        // Longest first, and the two copies are allowed to disagree about how
        // many words the overlap took. `cut` is what leaves the document's tail
        // and `took` is what the incoming copy spends on the same speech; they
        // differ by exactly the word one decode had and the other did not.
        let found = (SEAM_MIN..=most).rev().find_map(|k| {
            let widths = [k, k + 1, k.saturating_sub(1)];
            widths.into_iter().find_map(|cut| {
                let fits = cut >= SEAM_MIN && cut <= tail.len();
                (fits && Self::same_speech(&tail[tail.len() - cut..], &head[..k]))
                    .then_some((cut, k))
            })
        });
        let Some((cut, took)) = found else {
            return 0;
        };

        if prev.tier != Tier::Settled {
            return took;
        }
        prev.words.truncate(prev.words.len() - cut);
        if prev.words.is_empty() {
            self.document.pop();
        }
        self.revision += 1;
        0
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

    /// Records that the newest sentence stops where the audio was cut.
    ///
    /// Called by the forced trim, which is the only thing that ends a sentence
    /// for a reason having nothing to do with what the speaker said. The mark
    /// is what [`Transcript::cleanup_batch`]'s caller hands to the pass so it
    /// knows which full stop to distrust.
    pub fn mark_cut(&mut self) {
        if let Some(last) = self.document.last_mut() {
            last.cut = true;
        }
    }

    /// Records that the speaker stopped, so the next sentence opens a paragraph.
    ///
    /// Called where a hold expired rather than merged. That is the only place
    /// the pipeline learns the difference between a pause and a full stop, and
    /// it learns it by waiting rather than by reading the text, which §8 has
    /// twice found does not carry the signal.
    pub fn begin_paragraph(&mut self) {
        self.opens_paragraph = !self.document.is_empty();
    }

    /// The oldest run of settled sentences far enough behind to be cleaned up.
    ///
    /// Holds back the newest `lag` sentences of the document. Those are close
    /// enough to the speaker that a spoken command may still reach them, and
    /// restructuring a sentence under a `delete` aimed at it is the one way the
    /// cleanup pass could cost the speaker text rather than tidy it.
    ///
    /// Held to at least `min` sentences, which is what makes the pass worth
    /// running at all. Its main job is to join a sentence that was split across
    /// a trim seam, and a batch of one has nothing to join to: handed the
    /// sentences one at a time it can only re-punctuate each in isolation,
    /// which is the state it was brought in to repair.
    ///
    /// Capped at `max` so one pass cannot be handed an entire session, which
    /// would make its cost grow without bound as the document does.
    ///
    /// Contiguous by construction: it stops at the first sentence that is not
    /// settled, so a finalized sentence restored by `undo` cannot be swept back
    /// through the pass.
    pub fn cleanup_batch(
        &self,
        lag: usize,
        min: usize,
        max: usize,
    ) -> Option<std::ops::Range<usize>> {
        let from = self.document.iter().position(|s| s.tier == Tier::Settled)?;
        let run_end = self.document[from..]
            .iter()
            .position(|s| s.tier != Tier::Settled)
            .map_or(self.document.len(), |k| from + k);

        let limit = self.document.len().saturating_sub(lag).min(run_end);
        (limit.saturating_sub(from) >= min.max(1)).then(|| from..limit.min(from + max))
    }

    /// Replaces a settled run with the cleanup pass's reading of it.
    ///
    /// An empty `replacement` marks the run finalized where it stands, which is
    /// what a refused or unusable pass gets. Forward progress either way: a run
    /// that stayed settled would be offered to the next pass forever.
    ///
    /// # What this deliberately does not touch
    ///
    /// [`Transcript::filed`] keeps the recogniser's words, not these. It exists
    /// to strip text already filed out of the *next hypothesis*, and the next
    /// hypothesis comes from the audio, so it has to describe what the audio
    /// decodes to. The document describes what the speaker gets. Letting the
    /// cleanup pass edit both would break stripping for any sentence whose
    /// audio is still in the buffer.
    ///
    /// The undo stack is likewise untouched. Undo takes back what the *speaker*
    /// removed, and the cleanup pass is not the speaker; putting its edits on
    /// that stack would make `undo` mean two different things.
    pub fn finalize(
        &mut self,
        range: std::ops::Range<usize>,
        replacement: Vec<Vec<String>>,
        tail: Tail,
    ) {
        if range.start >= self.document.len() {
            return;
        }
        let range = range.start..range.end.min(self.document.len());

        if replacement.is_empty() {
            for sentence in &mut self.document[range] {
                sentence.tier = Tier::Finalized;
            }
        } else {
            // The paragraph break belongs to the position, not to the words,
            // so it stays with whatever now stands first in the run.
            let opens = self.document[range.clone()]
                .first()
                .is_some_and(|s| s.paragraph);
            let mut cleaned: Vec<Sentence> = replacement
                .into_iter()
                .filter(|w| !w.is_empty())
                .map(Sentence::finalized)
                .collect();
            if let Some(first) = cleaned.first_mut() {
                first.paragraph = opens;
            }
            // Carried only when something is left to finalize. A reply that
            // joined the whole run into one sentence has nothing to hand
            // forward, and holding it back would leave the batch starting where
            // it started and growing, forever.
            if tail == Tail::Carry && cleaned.len() > 1
                && let Some(last) = cleaned.last_mut()
            {
                last.tier = Tier::Settled;
            }
            self.document.splice(range, cleaned);
        }
        self.revision += 1;
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

    /// A batch of one has nothing to join to, which is the pass's whole job.
    #[test]
    fn a_cleanup_batch_is_held_back_until_it_is_worth_running() {
        let mut t = Transcript::new(2);
        for i in 0..6 {
            t.push_sentence(words(&format!("Sentence number {i}.")));
        }

        // Six settled, three held back for spoken commands: three available,
        // which is below the minimum worth sending.
        assert_eq!(t.cleanup_batch(3, 4, 12), None);

        t.push_sentence(words("One more."));
        assert_eq!(t.cleanup_batch(3, 4, 12), Some(0..4));
        assert_eq!(t.cleanup_batch(3, 4, 2), Some(0..2), "capped by max");
        assert_eq!(t.cleanup_batch(0, 1, 12), Some(0..7), "the flush takes it all");
    }

    /// The pass has to make progress even when it says nothing useful, or the
    /// same run is offered to it forever.
    /// The defect this exists for. A sentence split across a trim seam is only
    /// joinable by a pass that sees both halves, and abutting batches put a
    /// seam that lands on a batch boundary permanently out of reach: the first
    /// half finalized with its batch, the second opened the next one.
    ///
    /// The property to hold is not "batches overlap" but "no adjacent pair is
    /// split", so the test asks the second batch whether it starts on the
    /// sentence the first one ended on.
    /// The tail the pass never reaches during a session is `lag + min`
    /// sentences deep. `min` is what decides how deep, and `lag` is the floor
    /// nothing may reach, because a spoken command may still be aimed there.
    /// Both are arguments rather than constants precisely so this is testable
    /// and so §9 can sweep them.
    #[test]
    fn the_minimum_sets_the_reach_and_the_lag_is_the_floor() {
        let mut t = Transcript::new(3);
        for s in ["One.", "Two.", "Three.", "Four.", "Five."] {
            file(&mut t, s);
        }

        assert_eq!(t.cleanup_batch(3, 4, 12), None, "a full batch is not there yet");
        assert_eq!(
            t.cleanup_batch(3, 2, 12),
            Some(0..2),
            "a quiet document gives up everything but the lag"
        );
        assert_eq!(
            t.document().len() - 2,
            3,
            "and what it holds back is exactly the lag"
        );
    }

    #[test]
    fn the_next_batch_begins_where_the_last_one_ended() {
        let mut t = Transcript::new(3);
        for s in ["One.", "Two.", "Three.", "Four.", "Five.", "Six.", "Seven.", "Eight."] {
            file(&mut t, s);
        }

        let first = t.cleanup_batch(3, 4, 12).expect("enough settled sentences");
        let ended_on = t.document()[first.end - 1].text();
        let reply: Vec<Vec<String>> =
            t.document()[first.clone()].iter().map(|s| s.words.clone()).collect();
        t.finalize(first.clone(), reply, Tail::Carry);

        for s in ["Nine.", "Ten.", "Eleven.", "Twelve."] {
            file(&mut t, s);
        }
        let second = t.cleanup_batch(3, 4, 12).expect("a second batch");
        assert_eq!(
            t.document()[second.start].text(),
            ended_on,
            "the seam between the two batches has to be inside one of them"
        );
        assert_eq!(second.start, first.end - 1, "overlapping by exactly one");
    }

    /// The carry may not stall the pass. A reply that joined a whole batch into
    /// one sentence has nothing to hand forward, and holding that one back
    /// would leave the next batch starting where this one started.
    #[test]
    fn a_batch_joined_into_one_sentence_is_still_finalized() {
        let mut t = Transcript::new(3);
        for s in ["One.", "Two.", "Three.", "Four.", "Five.", "Six.", "Seven.", "Eight."] {
            file(&mut t, s);
        }
        let range = t.cleanup_batch(3, 4, 12).expect("a batch");
        t.finalize(range.clone(), vec![words("One two three four five.")], Tail::Carry);

        assert_eq!(t.document()[0].tier, Tier::Finalized, "forward progress or nothing moves");

        for s in ["Nine.", "Ten.", "Eleven.", "Twelve."] {
            file(&mut t, s);
        }
        assert_eq!(
            t.cleanup_batch(3, 4, 12).map(|r| r.start),
            Some(1),
            "the next batch has to start past the joined sentence"
        );
    }

    #[test]
    fn a_refused_batch_is_finalized_where_it_stands() {
        let mut t = Transcript::new(2);
        for i in 0..8 {
            t.push_sentence(words(&format!("Sentence number {i}.")));
        }
        let range = t.cleanup_batch(3, 4, 12).expect("a batch");
        let before: Vec<Vec<String>> = t.document()[range.clone()]
            .iter()
            .map(|s| s.words.clone())
            .collect();

        t.finalize(range.clone(), Vec::new(), Tail::Close);

        assert!(t.document()[range.clone()].iter().all(|s| s.tier == Tier::Finalized));
        assert_eq!(
            t.document()[range.clone()]
                .iter()
                .map(|s| s.words.clone())
                .collect::<Vec<_>>(),
            before,
            "refusing must not change a word"
        );
        let next = t.cleanup_batch(3, 4, 12);
        assert!(next.is_none_or(|r| r.start >= range.end), "never offered again");
    }

    /// Joining two settled sentences into one is the pass's ordinary output.
    #[test]
    fn a_cleanup_reply_replaces_the_run_it_was_given() {
        let mut t = Transcript::new(2);
        t.push_sentence(words("Keep me."));
        t.push_sentence(words("The product behaves on."));
        t.push_sentence(words("Such a long pause."));

        t.finalize(1..3, vec![words("The product behaves on such a long pause.")], Tail::Close);

        assert_eq!(t.document().len(), 2);
        assert_eq!(t.document()[1].text(), "The product behaves on such a long pause.");
        assert_eq!(t.document()[1].tier, Tier::Finalized);
        assert_eq!(t.document()[0].tier, Tier::Settled, "outside the run, untouched");
    }

    /// The seam repeat the speaker actually sees, with the substitution that
    /// stops an exact comparison finding it.
    #[test]
    fn a_seam_repeat_loses_the_copy_that_was_decoded_blind() {
        let mut t = Transcript::new(2);
        t.push_sentence(words(
            "Pick lots of words in just a matter of seconds, so this tool has to recognize.",
        ));
        t.push_sentence(words("So this too has to recognize both of these scenarios."));

        assert_eq!(
            t.document()[0].text(),
            "Pick lots of words in just a matter of seconds,",
            "the blind tail goes, the sentence that survives keeps its start"
        );
        assert_eq!(
            t.document()[1].text(),
            "So this too has to recognize both of these scenarios."
        );
    }

    /// The other repeat, taken from a run where the filed-text strip failed and
    /// a forced trim filed a slice from the middle of a sentence the document
    /// already held. It matches no sentence boundary and does not sit at the
    /// document's end, so neither of the other two guards can see it.
    #[test]
    fn text_the_document_already_holds_elsewhere_is_not_filed_again() {
        let mut t = Transcript::new(2);
        t.push_sentence(words(
            "What is too important to know is how exactly the algorithm commits the \
             messages, and whether or not it commits them.",
        ));
        t.push_sentence(words(
            "To know is how exactly the algorithm commits the messages, and whether or not.",
        ));

        assert_eq!(t.document().len(), 1, "the whole re-file was already on record");
    }

    /// The same guard must not swallow the new words behind a repeat.
    ///
    /// A sentence stands between the repeat and the document's end here, which
    /// is what makes it a re-file rather than a seam.
    #[test]
    fn only_the_repeated_opening_is_dropped() {
        let mut t = Transcript::new(2);
        t.push_sentence(words(
            "What is too important to know is how exactly the algorithm commits the messages.",
        ));
        t.push_sentence(words("Alternatively, many things could go wrong."));
        t.push_sentence(words(
            "To know is how exactly the algorithm commits the messages, and whether or not.",
        ));

        assert_eq!(t.document().len(), 3);
        assert_eq!(t.document()[2].text(), "and whether or not.");
    }

    /// A seam whose two copies disagree about how many words the overlap took.
    /// Both measured on one recording, and neither is visible to a matcher that
    /// compares position by position: the lengths differ, so it declines before
    /// it looks at a word.
    #[test]
    fn a_seam_repeat_survives_a_word_only_one_copy_has() {
        let mut t = Transcript::new(2);
        file(&mut t, "Also, when the first paragraph and second paragraph connect, converge.");
        file(&mut t, "First paragraph and second paragraph converge.");

        assert_eq!(
            texts(&t),
            vec![
                "Also, when the".to_string(),
                "First paragraph and second paragraph converge.".to_string()
            ],
            "the blind copy gives up the overlap, the later one keeps it"
        );
    }

    #[test]
    fn a_seam_repeat_survives_an_inserted_article() {
        let mut t = Transcript::new(2);
        file(
            &mut t,
            "Obviously, there could be some duplicates, like audio duplicates, such artifacts like previous one.",
        );
        file(&mut t, "Such artifacts, like the previous one, I saw, was clear.");

        assert_eq!(
            texts(&t),
            vec![
                "Obviously, there could be some duplicates, like audio duplicates,".to_string(),
                "Such artifacts, like the previous one, I saw, was clear.".to_string()
            ],
            "one copy of the overlap, and it is the one decoded with context"
        );
    }

    /// Dropping a matched prefix can leave a remainder that is itself a
    /// repeat, and a one-word remainder is the worst of both outcomes: it keeps
    /// none of the meaning and costs a sentence boundary. Measured, it read
    /// `Converge.`
    #[test]
    fn a_repeat_that_would_leave_a_stub_is_dropped_whole() {
        let mut t = Transcript::new(2);
        file(&mut t, "Also, when the first paragraph and second paragraph can vary, converge.");
        file(&mut t, "First paragraph and second paragraph converge.");

        assert_eq!(texts(&t).len(), 1, "no stub left behind: {:?}", texts(&t));
    }

    /// The forced trim knows the full stop it just created is its own, and the
    /// cleanup pass cannot tell from the words. The mark is how it is told.
    #[test]
    fn a_forced_cut_marks_the_sentence_it_ended() {
        let mut t = Transcript::new(2);
        file(&mut t, "It has already everything in place.");
        assert!(!t.document()[0].cut, "nothing has been cut yet");

        t.mark_cut();
        assert!(t.document()[0].cut);

        file(&mut t, "And to eliminate any duplicates as we go.");
        assert!(t.document()[0].cut, "the mark stays with the sentence it describes");
        assert!(!t.document()[1].cut, "and does not spread to the next one");
    }

    /// The looser matcher must not turn into one that matches anything. Two
    /// runs that happen to start and end alike are not the same speech.
    #[test]
    fn a_run_that_only_shares_its_ends_is_not_a_seam() {
        let mut t = Transcript::new(2);
        file(&mut t, "The build failed because the tests were flaky today.");
        file(&mut t, "The release went out without a hitch today.");

        assert_eq!(texts(&t).len(), 2, "nothing here is a repeat: {:?}", texts(&t));
        assert!(texts(&t)[0].ends_with("today."), "the first sentence is untouched");
    }

    /// A fragment too short for any tolerance to reach. The forced trim files
    /// whatever the severed head decoded to, and on a real recording that put
    /// `Such artifacts, like.` into the document three words behind the sentence
    /// that already said it. Nothing above `SEAM_MIN` can see three words.
    #[test]
    fn a_short_fragment_the_document_already_holds_is_dropped() {
        let mut t = Transcript::new(2);
        file(
            &mut t,
            "Obviously, there could be some duplicates, like audio duplicates, such artifacts like previous one.",
        );
        file(&mut t, "Such artifacts, like.");

        assert_eq!(texts(&t).len(), 1, "the fragment is a repeat: {:?}", texts(&t));
    }

    /// The short match has to be exact, or every `Yes.` in a session collapses
    /// into the first one.
    #[test]
    fn a_short_sentence_the_document_does_not_hold_survives() {
        let mut t = Transcript::new(2);
        file(&mut t, "Obviously, there could be some duplicates, like audio duplicates.");
        file(&mut t, "Such artifacts, though.");

        assert_eq!(texts(&t).len(), 2, "nothing in the tail says this: {:?}", texts(&t));
    }

    /// A finalized sentence has been read in full by the pass and is no longer
    /// the pipeline's to edit.
    #[test]
    fn a_seam_repeat_never_edits_finalized_text() {
        let mut t = Transcript::new(2);
        t.push_sentence(words("So this tool has to recognize."));
        t.finalize(0..1, Vec::new(), Tail::Close);
        t.push_sentence(words("So this tool has to recognize both of these."));

        assert_eq!(t.document().len(), 2);
        assert_eq!(t.document()[0].text(), "So this tool has to recognize.");
        assert_eq!(
            t.document()[1].text(),
            "both of these.",
            "the finalized copy stands, so the incoming one gives up the repeat"
        );
    }

    /// What the clipboard carries.
    #[test]
    fn prose_runs_sentences_together_and_breaks_paragraphs() {
        let mut t = Transcript::new(2);
        t.push_sentence(words("First thought."));
        t.push_sentence(words("Still the same one."));
        t.begin_paragraph();
        t.push_sentence(words("A new one."));

        assert_eq!(
            prose(t.document()),
            "First thought. Still the same one.\n\nA new one.\n"
        );
        assert_eq!(prose(&[]), "", "nothing to deliver is not a blank line");
    }

    /// The duplicate a speaker actually sees, reduced to the state that causes
    /// it.
    ///
    /// An endpoint files whole sentences while their audio sits in the held
    /// utterance; a continuation splices that audio back to the front of the
    /// buffer; the trim then decodes a head of it. The record starts earlier
    /// than that decode, so no front match is possible, and the decode is
    /// shorter than the record's tail, so no whole-tail match is possible
    /// either. Read as unfiled, the sentence is filed a second time.
    #[test]
    fn a_head_decode_of_restored_audio_is_recognised_as_filed() {
        let mut t = Transcript::new(2);
        t.push_sentence(words("Alpha beta gamma delta."));
        t.push_sentence(words(
            "So my entire speech represents both the quick passage and the slow passage.",
        ));

        assert_eq!(
            t.unfiled("So my entire speech represents both."),
            0,
            "every word of this head is already in the document"
        );
        assert_eq!(
            t.unfiled("So my entire speech represents both the quick passage and the slow \
                       passage. Thank you for your attention."),
            5,
            "and only the words past it are new"
        );
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

    /// The same hazard on a run too short for a proportional budget to buy an
    /// edit, which is the shape it actually took on a real recording.
    ///
    /// `So, not sure what's going on there.` was filed at an endpoint, the
    /// continuation spliced its audio back into the buffer, and the re-decode
    /// came back `So I'm not sure what's going on there.` Seven words, one
    /// insertion, and every alignment failed on the second word: the speaker
    /// got the sentence twice.
    #[test]
    fn a_short_filed_sentence_survives_one_inserted_word() {
        let mut t = Transcript::new(3);
        file(&mut t, "So, not sure what's going on there.");

        let seen = "So I'm not sure what's going on there.";
        assert_eq!(
            t.unfiled(seen),
            0,
            "the sentence is filed; nothing here is new: {:?}",
            texts(&t)
        );
    }

    /// The floor is a floor, not a licence. Three words must still agree
    /// outright, or an overlap becomes something the aligner can find anywhere.
    #[test]
    fn a_run_below_the_floor_still_matches_exactly() {
        let mut t = Transcript::new(3);
        file(&mut t, "Ship it.");
        let seen = "Skip it.";
        assert_eq!(
            t.unfiled(seen),
            seen.split_whitespace().count(),
            "two words that differ are not an overlap"
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
            Sentence::settled(words("From last time.")),
            Sentence::settled(words("And more.")),
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
        t.restore(vec![Sentence::settled(words("Recovered."))]);
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
