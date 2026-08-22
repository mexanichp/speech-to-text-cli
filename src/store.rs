//! Persistence for the transcript: the session file and the clipboard.
//!
//! The document is written to disk on every change, which is what makes the
//! destructive spoken commands safe to offer and what stops an unclean exit
//! from taking the session with it. Three exit paths write nothing on the way
//! out: a second interrupt, a panic, and `SIGHUP` when the terminal closes.
//!
//! Writing is not on the hot path. The file is rewritten when a sentence
//! settles or a command runs, a few times a minute rather than per tick.
//!
//! # Deletion policy
//!
//! A clean exit removes the file unless `--persist` was given, since stdout has
//! the transcript by then. An unclean exit cannot remove it, because no code is
//! running to do so. That asymmetry is deliberate: the file survives exactly
//! the cases that lose text, and is cleaned up in the case that never does.

use anyhow::{Context, Result};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::transcript::Sentence;

/// Longest the utterance in flight may go unwritten.
///
/// The document is written the instant it changes, but in-flight text changes
/// every tick and syncing that often would cost without benefit. Between
/// filings the in-flight buffer holds the whole passage, so it cannot go
/// unwritten either. This bounds what a crash can take to roughly one clause.
const IN_FLIGHT_EVERY: Duration = Duration::from_secs(2);

/// The session file, rewritten whenever the document changes.
pub struct Store {
    path: PathBuf,
    persist: bool,
    /// Revision last written, so an unchanged document costs nothing.
    written: Option<u64>,
    /// In-flight text as of the last write, so an unchanged utterance is free
    /// too and only genuine movement pays.
    in_flight: String,
    last_write: Instant,
    /// Set once writing fails, so a full disk reports once instead of on every
    /// sentence for the rest of the session.
    failed: bool,
}

impl Store {
    /// Creates a new session file in the state directory.
    ///
    /// The file is created immediately so that an unwritable directory fails
    /// here rather than at the first sentence.
    ///
    /// # Parameters
    ///
    /// - `persist`: keep the file on a clean exit instead of deleting it.
    ///
    /// # Errors
    ///
    /// Fails if the state directory cannot be created or written to.
    pub fn new(persist: bool) -> Result<Self> {
        let dir = state_dir();
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating {}", dir.display()))?;

        let path = dir.join(format!("session-{}.txt", stamp()));
        std::fs::write(&path, "").with_context(|| format!("creating {}", path.display()))?;

        Ok(Self {
            path,
            persist,
            written: None,
            in_flight: String::new(),
            last_write: Instant::now(),
            failed: false,
        })
    }

    /// Continues writing to a session file this program left behind.
    ///
    /// Continuing in place rather than opening a new file keeps the state
    /// directory from accumulating one stale copy per crash.
    ///
    /// Valid only for a file found in this program's own state directory. A
    /// path the user named must be treated as read-only, since adopting it
    /// would let a clean exit delete a file they supplied.
    pub fn adopt(path: PathBuf, persist: bool) -> Self {
        Self {
            path,
            persist,
            written: None,
            in_flight: String::new(),
            last_write: Instant::now(),
            failed: false,
        }
    }

    /// Path of the file being written.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Writes the document and any in-flight text, if either has changed.
    ///
    /// # Parameters
    ///
    /// - `revision`: document version, compared for equality so an unchanged
    ///   document costs nothing.
    /// - `in_flight`: text the pipeline holds that has not reached the document
    ///   yet. Including it is what keeps recovery independent of when text is
    ///   filed, since a passage the speaker never pauses in reaches the
    ///   document late.
    ///
    /// Growing in-flight text is rate-limited to [`IN_FLIGHT_EVERY`]. Shrinking
    /// text is written through, because it shrinks when the document absorbs
    /// it and deferring would leave the file holding the same words twice.
    ///
    /// # Returns
    ///
    /// An error message on the first failure and `None` thereafter. A failed
    /// autosave is reported once rather than ending the session, since the
    /// in-memory document is intact and still reaches stdout at exit.
    pub fn save(&mut self, document: &[Sentence], revision: u64, in_flight: &str) -> Option<String> {
        if self.failed {
            return None;
        }

        let settled = self.written != Some(revision);
        if !settled && self.in_flight == in_flight {
            return None;
        }

        if !settled
            && in_flight.len() > self.in_flight.len()
            && self.last_write.elapsed() < IN_FLIGHT_EVERY
        {
            return None;
        }

        self.written = Some(revision);
        self.in_flight = in_flight.to_string();
        self.last_write = Instant::now();

        match write_atomically(&self.path, document, in_flight) {
            Ok(()) => None,
            Err(e) => {
                self.failed = true;
                Some(format!("autosave failed, transcript is memory-only: {e}"))
            }
        }
    }

    /// Completes a clean exit, removing the file unless persistence was
    /// requested.
    ///
    /// # Returns
    ///
    /// The path when the file was kept, so the caller can report it.
    pub fn finish(self) -> Option<PathBuf> {
        if self.persist {
            return Some(self.path);
        }
        let _ = std::fs::remove_file(&self.path);
        None
    }
}

/// Finds the most recently written session file.
///
/// Selected by modification time rather than name, because a name records when
/// a session started and a resumed session continues an older file.
///
/// Partial `.tmp` files are ignored.
pub fn newest_session() -> Option<PathBuf> {
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;

    for entry in std::fs::read_dir(state_dir()).ok()?.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "txt") {
            continue;
        }
        let Ok(at) = entry.metadata().and_then(|m| m.modified()) else {
            continue;
        };
        if best.as_ref().is_none_or(|(newest, _)| at > *newest) {
            best = Some((at, path));
        }
    }

    best.map(|(_, path)| path)
}

/// Reads a session file back into sentences.
///
/// One line is one sentence, which is the whole of the format. Blank lines are
/// skipped. Because a sentence is only its words, the round trip is lossless.
///
/// # Errors
///
/// Fails if the file cannot be read.
pub fn load(path: &Path) -> Result<Vec<Sentence>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;

    Ok(text
        .lines()
        .map(|line| line.split_whitespace().map(str::to_string).collect::<Vec<_>>())
        .filter(|words| !words.is_empty())
        .map(Sentence::settled)
        .collect())
}

/// Writes the transcript via a temporary file and an atomic rename.
///
/// A crash midway therefore leaves either the previous transcript or the new
/// one, never a partial file.
///
/// In-flight text is appended as trailing lines, split into sentences like the
/// rest, and is deliberately unmarked: `--persist` hands the file to a person
/// and `--resume` reads it straight back, so it must stay a clean transcript.
///
/// # Errors
///
/// Fails if the temporary file cannot be written, synced or renamed.
fn write_atomically(path: &Path, document: &[Sentence], in_flight: &str) -> Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut file = std::fs::File::create(&tmp)?;
        for sentence in document {
            writeln!(file, "{}", sentence.text())?;
        }
        for part in crate::text::split_sentences(in_flight.trim()) {
            if !part.is_empty() {
                writeln!(file, "{part}")?;
            }
        }
        file.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Directory holding session files, under the user's state directory.
fn state_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map_or_else(std::env::temp_dir, PathBuf::from);
    home.join(".local/state/speech-to-text-cli")
}

/// Formats the current UTC time as `YYYYmmdd-HHMMSS` for a session filename.
///
/// Computed rather than taken from a date crate: the only requirements are that
/// concurrent sessions differ and that a person can identify their own.
fn stamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());

    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);

    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = era * 400 + yoe + i64::from(month <= 2);

    format!("{year:04}{month:02}{day:02}-{hh:02}{mm:02}{ss:02}")
}

/// Copies text to the system clipboard via `pbcopy`.
///
/// Shells out rather than taking a dependency, since the runtime is macOS-only
/// in any case. The child's output streams are silenced: the renderer owns the
/// terminal, and a child writing to it would corrupt the frame.
///
/// # Errors
///
/// Fails if `pbcopy` cannot be spawned, written to, or exits non-zero.
pub fn copy_to_clipboard(text: &str) -> Result<()> {
    let mut child = Command::new("pbcopy")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("spawning pbcopy")?;

    child
        .stdin
        .take()
        .context("pbcopy stdin")?
        .write_all(text.as_bytes())
        .context("writing to pbcopy")?;

    let status = child.wait().context("waiting for pbcopy")?;
    anyhow::ensure!(status.success(), "pbcopy exited with {status}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sentence(s: &str) -> Sentence {
        Sentence::settled(s.split_whitespace().map(str::to_string).collect())
    }

    /// The timestamp only has to be sortable and human-legible, but a wrong
    /// civil-from-days conversion is the kind of thing that silently produces
    /// month 13 on one day of the year.
    #[test]
    fn the_stamp_is_a_plausible_utc_datetime() {
        let s = stamp();
        assert_eq!(s.len(), 15, "YYYYmmdd-HHMMSS: {s}");
        assert_eq!(&s[8..9], "-");

        let num = |r: std::ops::Range<usize>| s[r].parse::<u32>().expect("digits");
        assert!((2020..2100).contains(&num(0..4)), "year: {s}");
        assert!((1..=12).contains(&num(4..6)), "month: {s}");
        assert!((1..=31).contains(&num(6..8)), "day: {s}");
        assert!(num(9..11) < 24 && num(11..13) < 60 && num(13..15) < 60, "time: {s}");
    }

    /// The file is the transcript, not a debug dump: every sentence, one per
    /// line, exactly as stdout prints them.
    #[test]
    fn the_file_holds_the_document_verbatim() {
        let dir = std::env::temp_dir().join(format!("stt-store-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("t.txt");

        let doc = [sentence("Kept one."), sentence("Still pending.")];
        write_atomically(&path, &doc, "").expect("write");

        assert_eq!(
            std::fs::read_to_string(&path).expect("read"),
            "Kept one.\nStill pending.\n"
        );

        write_atomically(&path, &doc[..1], "").expect("rewrite");
        assert_eq!(std::fs::read_to_string(&path).expect("read"), "Kept one.\n");

        assert!(!path.with_extension("tmp").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Resume has to be lossless for the thing the file actually stores: the
    /// text, and where one sentence ends and the next begins.
    #[test]
    fn a_document_survives_a_write_and_load_round_trip() {
        let dir = std::env::temp_dir().join(format!("stt-round-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("t.txt");

        let doc = [
            sentence("First one, with a comma."),
            sentence("Second one!"),
            sentence("\u{4f60}\u{597d}\u{3002}"),
        ];
        write_atomically(&path, &doc, "").expect("write");
        let back = load(&path).expect("load");

        let texts: Vec<String> = back.iter().map(Sentence::text).collect();
        assert_eq!(
            texts,
            ["First one, with a comma.", "Second one!", "\u{4f60}\u{597d}\u{3002}"]
        );
        assert_eq!(back, doc.to_vec());

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("stt-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir.join("t.txt")
    }

    /// In-flight text must reach the file, or a passage the speaker never
    /// pauses in is absent from it until something files.
    #[test]
    fn the_utterance_in_flight_reaches_the_file() {
        let path = scratch("inflight");
        let doc = [sentence("Already filed.")];

        write_atomically(&path, &doc, "and this is still being said").expect("write");
        assert_eq!(
            std::fs::read_to_string(&path).expect("read"),
            "Already filed.\nand this is still being said\n"
        );

        let back = load(&path).expect("load");
        assert_eq!(
            back.iter().map(Sentence::text).collect::<Vec<_>>(),
            ["Already filed.", "and this is still being said"]
        );

        write_atomically(&path, &[], "nothing has settled yet").expect("write");
        assert_eq!(
            std::fs::read_to_string(&path).expect("read"),
            "nothing has settled yet\n"
        );

        write_atomically(&path, &[], "One thought. Then a second. And a third").expect("write");
        assert_eq!(
            std::fs::read_to_string(&path).expect("read"),
            "One thought.\nThen a second.\nAnd a third\n"
        );

        write_atomically(&path, &doc, "").expect("write");
        assert_eq!(std::fs::read_to_string(&path).expect("read"), "Already filed.\n");

        let _ = std::fs::remove_dir_all(path.parent().expect("dir"));
    }

    /// A settling document is written through; a growing utterance is not, or
    /// there would be an `fsync` on every tick for a line that changes anyway.
    #[test]
    fn a_growing_utterance_is_rate_limited_but_the_document_is_not() {
        let path = scratch("throttle");
        let mut store = Store::adopt(path.clone(), true);
        let doc = [sentence("One.")];

        assert_eq!(store.save(&doc, 1, "i am talking"), None);
        assert_eq!(
            std::fs::read_to_string(&path).expect("read"),
            "One.\ni am talking\n"
        );

        assert_eq!(store.save(&doc, 1, "i am talking and talking"), None);
        assert_eq!(
            std::fs::read_to_string(&path).expect("read"),
            "One.\ni am talking\n",
            "a growing utterance must not write through on every tick"
        );

        let doc = [sentence("One."), sentence("Two.")];
        assert_eq!(store.save(&doc, 2, "i am talking and talking"), None);
        assert_eq!(
            std::fs::read_to_string(&path).expect("read"),
            "One.\nTwo.\ni am talking and talking\n"
        );

        let _ = std::fs::remove_dir_all(path.parent().expect("dir"));
    }

    /// The in-flight line shrinks when the document absorbs it. Deferring that
    /// would leave the file briefly holding the same words twice — once as a
    /// sentence and once as the tail — which is worse than a stale tail.
    #[test]
    fn an_absorbed_utterance_is_cleared_immediately() {
        let path = scratch("absorb");
        let mut store = Store::adopt(path.clone(), true);

        store.save(&[], 0, "this will settle");
        assert_eq!(std::fs::read_to_string(&path).expect("read"), "this will settle\n");

        store.save(&[], 0, "");
        assert_eq!(
            std::fs::read_to_string(&path).expect("read"),
            "",
            "a shrinking tail must not wait out the rate limit"
        );

        let _ = std::fs::remove_dir_all(path.parent().expect("dir"));
    }

    /// An unchanged session must still cost nothing, which is what kept this
    /// off the hot path in the first place.
    #[test]
    fn an_unchanged_session_is_never_rewritten() {
        let path = scratch("unchanged");
        let mut store = Store::adopt(path.clone(), true);
        let doc = [sentence("Settled.")];

        store.save(&doc, 1, "");
        let before = std::fs::metadata(&path).and_then(|m| m.modified()).expect("mtime");

        for _ in 0..10 {
            assert_eq!(store.save(&doc, 1, ""), None);
        }
        let after = std::fs::metadata(&path).and_then(|m| m.modified()).expect("mtime");
        assert_eq!(before, after, "nothing moved, so nothing may be written");

        let _ = std::fs::remove_dir_all(path.parent().expect("dir"));
    }

    /// A file half-written by a crash, or one a user has edited by hand, must
    /// not produce empty sentences the renderer then has to cope with.
    #[test]
    fn blank_lines_and_trailing_whitespace_are_not_sentences() {
        let dir = std::env::temp_dir().join(format!("stt-blank-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("t.txt");

        std::fs::write(&path, "Real one.\n\n   \n\nAnother.\n\n").expect("write");
        let back = load(&path).expect("load");
        assert_eq!(
            back.iter().map(Sentence::text).collect::<Vec<_>>(),
            ["Real one.", "Another."]
        );

        std::fs::write(&path, "").expect("write");
        assert!(load(&path).expect("load").is_empty(), "an empty session is empty");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
