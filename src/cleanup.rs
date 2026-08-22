//! The cleanup pass: a second model, over text only, off the latency path.
//!
//! Speech recognition holds the audio buffer short so live text keeps up with
//! the speaker. That is bought by cutting the buffer at pauses that are not
//! always sentence boundaries, which leaves the document split where the
//! speaker did not pause and joined where they did. This module reads a run of
//! settled sentences and says where the boundaries should have been.
//!
//! # Why a separate process
//!
//! It shares the GPU with recognition, and recognition is on the latency path
//! while this is not. Run inline, a pass here stalls the live one; stalled long
//! enough, the main loop swallows a whole utterance per iteration and stops
//! trimming the buffer at all, which is the failure this whole design exists to
//! avoid. The host therefore hands work over and collects it whenever it is
//! ready, and never waits.
//!
//! # Why the reply is checked
//!
//! A language model asked to tidy a transcript will also, given the chance,
//! improve it: `and sometimes they just so someone` came back as `and sometimes
//! they just wait for someone else` from a smaller model, which is fluent,
//! confident and something the speaker never said. Local recognition's one real
//! guarantee is that the words are the speaker's, so [`invents_content`] rejects
//! any reply carrying a content word the input did not have. Function words are
//! exempt: fixing an agreement error is the point.

use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TryRecvError};
use serde::Deserialize;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};

/// Sentences held back from the pass, newest first.
///
/// They are close enough to the speaker that a spoken command may still be
/// aimed at them, and restructuring a sentence out from under a `delete` is the
/// one way this pass could cost text rather than tidy it.
pub const LAG: usize = 3;

/// Fewest sentences worth handing to one pass.
///
/// The pass exists to join sentences split across a trim seam, and a batch of
/// one has nothing to join to. Below this the document is left alone and the
/// sentences accumulate until there are enough of them to read together.
///
/// **This is the strongest lever measured on the cleanup pass.** At four, which
/// is what it was, batches came out at exactly four all session and boundary
/// precision on recording 3 sat at 0.86. At eight it is 0.95, separated across
/// three interleaved runs, at identical WER and recall. §11 had been saying for
/// a while that the pass declines a join when it cannot see enough of the
/// passage; this is that, with a number on it.
///
/// It is bounded above by the summarising failure in §8 and below by this, so
/// it is a window rather than a direction. Eight is the only point above four
/// that has been measured.
pub const MIN_BATCH: usize = 8;

/// Most sentences handed to one pass.
///
/// Bounds both the prompt and the time a single reply can take, so the cost of
/// a pass does not grow with the length of the session.
pub const BATCH: usize = 12;

/// Words that may be added or dropped freely.
///
/// Fixing agreement broken by a bad split is the point of the pass, and that
/// means moving these. Everything else has to have been said.
const FUNCTION_WORDS: &[&str] = &[
    // articles, conjunctions, copulas
    "a", "an", "the", "and", "or", "but", "so", "then", "than", "that", "this", "these", "those",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "have", "has",
    "had", "because", "though", "although", "since", "whether", "as", "if", "unless",
    // relative and interrogative pronouns, which a re-join routinely restores
    "who", "whom", "whose", "which", "what", "where", "when", "why", "how", "while",
    // modals, which a re-split routinely breaks
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    // prepositions and particles
    "of", "to", "in", "on", "at", "for", "with", "by", "from", "into", "onto", "about", "over",
    "under", "through", "between", "up", "down", "out", "off", "there", "here",
    // pronouns and determiners
    "it", "its", "they", "them", "their", "he", "him", "his", "she", "her", "we", "us", "our",
    "you", "your", "i", "my", "me", "all", "any", "some", "such", "no", "not", "own", "same",
    // degree and focus adverbs
    "very", "too", "more", "most", "only", "also", "even", "still", "again", "just", "ever",
    "never", "always", "already",
];

/// One batch handed to the pass.
struct Job {
    seq: u64,
    text: String,
}

/// One batch come back.
pub struct Done {
    pub seq: u64,
    /// The reply, one sentence per line. Empty when the pass had nothing to say.
    pub text: String,
}

/// What [`Cleanup::poll`] found.
pub enum Progress {
    /// The batch that is out has not come back yet.
    Waiting,
    /// A batch finished.
    Done(Done),
    /// The sidecar is gone and nothing further will come back.
    ///
    /// Reported once, on the transition. A model that fails to load leaves a
    /// process that exits, a channel that closes and a batch that is out
    /// forever, and without this the pass stops for the rest of the session
    /// with nothing on screen to say so.
    Lost,
}

/// A running cleanup sidecar, or nothing if it could not be started.
pub struct Cleanup {
    jobs: Sender<Job>,
    done: Receiver<Done>,
    notices: Receiver<String>,
    child: Child,
    /// Set while a batch is out, so only one is ever in flight.
    busy: bool,
    /// Cleared once the sidecar is known to be gone, so [`Progress::Lost`] is
    /// reported once rather than on every iteration of the main loop.
    alive: bool,
    next: u64,
}

#[derive(Deserialize)]
struct Reply {
    #[serde(default)]
    id: Option<u64>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    event: Option<String>,
}

impl Cleanup {
    /// Starts the sidecar and the thread that talks to it.
    ///
    /// Returns once the process exists, not once the model is resident. Loading
    /// takes seconds and there is nothing to clean up in the first seconds of a
    /// session, so waiting would only delay the speaker.
    ///
    /// # Errors
    ///
    /// Fails only if the process cannot be spawned. A model that fails to load
    /// shows up later as a dead channel and is reported as a notice.
    pub fn start(python: &Path, script: &Path, model: &str) -> Result<Self> {
        let mut child = Command::new(python)
            .arg(script)
            .arg("--model")
            .arg(model)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| {
                format!("spawning cleanup sidecar: {} {}", python.display(), script.display())
            })?;

        let (jobs_tx, jobs_rx) = crossbeam_channel::unbounded::<Job>();
        let (done_tx, done_rx) = crossbeam_channel::unbounded::<Done>();
        let (note_tx, note_rx) = crossbeam_channel::unbounded::<String>();

        let errors = note_tx.clone();
        let mut stdin = child.stdin.take().context("cleanup sidecar stdin")?;
        let stdout = child.stdout.take().context("cleanup sidecar stdout")?;
        let stderr = child.stderr.take().context("cleanup sidecar stderr")?;

        // Piped rather than inherited: anything the model's dependencies print
        // would land on the live view and corrupt the frame.
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if !line.trim().is_empty() {
                    let _ = note_tx.send(format!("cleanup: {line}"));
                }
            }
        });

        std::thread::spawn(move || {
            let mut out = BufReader::new(stdout);
            let mut line = String::new();

            // The readiness banner. Anything else here means the process died
            // before it loaded, and the job loop below will see the closed pipe.
            if out.read_line(&mut line).unwrap_or(0) == 0 {
                return;
            }

            for job in jobs_rx {
                let request = serde_json::json!({ "id": job.seq, "text": job.text });
                if writeln!(stdin, "{request}").is_err() || stdin.flush().is_err() {
                    return;
                }
                line.clear();
                if out.read_line(&mut line).unwrap_or(0) == 0 {
                    return;
                }
                // Every job is answered, including the ones that go wrong.
                // Skipping the reply instead would leave the host holding a
                // batch that never comes back, and the pass would stop for the
                // rest of the session without stopping anything else.
                let text = match serde_json::from_str::<Reply>(&line) {
                    // Same discipline as the recognition sidecar: every reply
                    // echoes its request id, and one that does not means the
                    // stream has slipped a message. Pairing a batch with the
                    // previous batch's reply would splice one run of sentences
                    // over another.
                    Ok(reply) if reply.event.is_none() && reply.id == Some(job.seq) => {
                        if let Some(why) = reply.error {
                            let _ = errors.send(format!("cleanup pass failed on a batch: {why}"));
                        }
                        reply.text.unwrap_or_default()
                    }
                    Ok(_) => {
                        let _ = errors.send("cleanup pass replied out of step".to_string());
                        String::new()
                    }
                    Err(_) => String::new(),
                };
                if done_tx.send(Done { seq: job.seq, text }).is_err() {
                    return;
                }
            }
        });

        Ok(Self {
            jobs: jobs_tx,
            done: done_rx,
            notices: note_rx,
            child,
            busy: false,
            alive: true,
            next: 0,
        })
    }

    /// Whether a batch is already out.
    pub fn busy(&self) -> bool {
        self.busy
    }

    /// Whether the sidecar is still answering.
    ///
    /// False once a poll has found the channel closed. Submitting after that
    /// succeeds silently, because the job channel is unbounded and its receiver
    /// is what died, so the caller has to ask.
    pub fn alive(&self) -> bool {
        self.alive
    }

    /// Hands one batch over. Never blocks.
    ///
    /// # Returns
    ///
    /// The sequence number the reply will carry, or `None` if the sidecar is
    /// gone.
    pub fn submit(&mut self, text: String) -> Option<u64> {
        let seq = self.next;
        self.next += 1;
        self.jobs.send(Job { seq, text }).ok()?;
        self.busy = true;
        Some(seq)
    }

    /// Collects a finished batch if one is waiting. Never blocks.
    pub fn poll(&mut self) -> Progress {
        match self.done.try_recv() {
            Ok(done) => {
                self.busy = false;
                Progress::Done(done)
            }
            Err(TryRecvError::Empty) => Progress::Waiting,
            Err(TryRecvError::Disconnected) => {
                self.busy = false;
                match std::mem::take(&mut self.alive) {
                    true => Progress::Lost,
                    false => Progress::Waiting,
                }
            }
        }
    }

    /// Blocks for a batch already out, for the flush at end of session.
    ///
    /// The only place this waits. Nothing is live any more, so the cost is the
    /// speaker's exit rather than their latency, and the alternative is handing
    /// them a transcript with the seams still in it.
    pub fn wait(&mut self, within: std::time::Duration) -> Option<Done> {
        let got = self.done.recv_timeout(within);
        self.busy = false;
        // A timeout is a slow batch, not a dead sidecar. Conflating them
        // retires the pass for the rest of the session over one long reply,
        // and the `copy` flush runs on a budget short enough to hit.
        self.alive &= !matches!(got, Err(RecvTimeoutError::Disconnected));
        got.ok()
    }

    /// Diagnostics from the sidecar's stderr.
    pub fn notices(&self) -> &Receiver<String> {
        &self.notices
    }

    /// Closes stdin so the sidecar exits on EOF.
    pub fn finish(mut self) {
        drop(self.jobs);
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Strips a word to the form two spellings of it share.
///
/// Case, punctuation and the inflections a rewrite legitimately changes. Crude
/// on purpose: this decides whether a word was *said*, not what it means, and
/// an over-eager stem costs a rejected edit rather than invented text.
fn stem(word: &str) -> String {
    let base: String = word
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '\'')
        .flat_map(char::to_lowercase)
        .collect();

    for suffix in ["ing", "ed", "es", "ly", "s"] {
        if base.len() > suffix.len() + 2
            && let Some(cut) = base.strip_suffix(suffix)
        {
            return cut.to_string();
        }
    }
    base
}

/// The share of the input's content words a reply may lose before it is
/// refused, as a denominator: a quarter.
///
/// Some loss is correct. A seam repeat is removed by dropping one copy, and a
/// re-join drops filler. Losing a quarter of what was said is not any of those.
const MAY_LOSE_DEN: usize = 4;

/// What the checker made of a reply.
#[derive(Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Restructured, and every word still accounted for.
    Accept,
    /// Carries a content word the speaker never said.
    Invented,
    /// Lost too much of what the speaker did say.
    Dropped,
}

/// Whether a reply may be spliced into the document.
///
/// The check the whole pass rests on, and it has to run in **both** directions.
///
/// A model asked to tidy will also invent, and an invented clause is fluent,
/// confident and indistinguishable from the rest by eye: a 1.7B model turned
/// `and sometimes they just so someone` into `and sometimes they just wait for
/// someone else`.
///
/// A model handed a long passage will also summarise it, which is the failure
/// that a one-directional check misses entirely. Measured: a batch of eight
/// sentences came back as four, every word of the four legitimately said, and
/// half the passage gone. It read as a clean reply and cost the speaker a third
/// of their transcript.
///
/// Content words only. Function words move freely, because fixing agreement
/// broken by a bad split is the point of the pass.
pub fn check(before: &[String], after: &[String]) -> Verdict {
    let stems = |words: &[String]| -> std::collections::HashSet<String> {
        words
            .iter()
            .map(|w| stem(w))
            .filter(|s| !s.is_empty() && !FUNCTION_WORDS.contains(&s.as_str()))
            .collect()
    };

    let said = stems(before);
    let kept = stems(after);

    if kept.difference(&said).next().is_some() {
        return Verdict::Invented;
    }
    if said.difference(&kept).count() * MAY_LOSE_DEN > said.len() {
        return Verdict::Dropped;
    }
    Verdict::Accept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    /// The measured failure, kept as the case this check exists for. A 1.7B
    /// model turned a mis-segmented clause into a fluent one that says something
    /// the speaker never said.
    #[test]
    fn an_invented_clause_is_rejected() {
        let before = words("and sometimes they just so someone can actually be slow");
        let after = words("and sometimes they just wait for someone else to be slow");
        assert_eq!(check(&before, &after), Verdict::Invented, "\"wait\" was never said");
    }

    /// The measured failure a one-directional check missed: a batch of eight
    /// sentences came back as four, every surviving word legitimately said.
    #[test]
    fn a_summarised_reply_is_rejected() {
        let before = words(
            "So this tool has to recognize both of these scenarios and make sure it commits \
             the messages logically and not other way. So my entire speech represents both \
             the quick passage and the slow passage. Thank you for your attention to this \
             matter. Bye now.",
        );
        let after = words(
            "So this tool has to recognize both of these scenarios and make sure it commits \
             the messages logically.",
        );
        assert_eq!(check(&before, &after), Verdict::Dropped);
    }

    /// Re-punctuating, re-joining and re-casing are the whole point, so none of
    /// them may trip the check.
    #[test]
    fn restructuring_is_allowed() {
        let before = words("What is too important to know. Is. How exactly. The algorithm.");
        let after = words("What is too important to know is how exactly the algorithm.");
        assert_eq!(check(&before, &after), Verdict::Accept);
    }

    /// Agreement broken by a bad split is what the pass is asked to fix, and
    /// fixing it moves function words and inflections.
    #[test]
    fn fixing_agreement_is_allowed() {
        let before = words("For person, will sometimes slows down");
        let after = words("For a person who sometimes slows down");
        assert_eq!(check(&before, &after), Verdict::Accept);

        let before = words("they was speaking quick");
        let after = words("they were speaking quickly");
        assert_eq!(check(&before, &after), Verdict::Accept);
    }

    /// Dropping words is not invention. A seam repeat is removed by dropping
    /// one copy, and that has to pass.
    #[test]
    fn dropping_a_repeat_is_allowed() {
        let before = words("so this tool has to recognize So this too has to recognize both");
        let after = words("So this too has to recognize both");
        assert_eq!(check(&before, &after), Verdict::Accept);
    }
}
