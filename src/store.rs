//! Getting the transcript out of process memory: the session file, and the
//! clipboard.
//!
//! # Why the session file exists
//!
//! The document used to live only in `Vec<Sentence>` and reach stdout once, at
//! exit. That was survivable while the only way to lose text was closing the
//! terminal, but three exit paths skipped the write entirely — a second Ctrl-C
//! (`libc::_exit`, which by design runs no destructors), a panic (the renderer's
//! `Drop` restores the terminal but writes nothing), and SIGHUP when the window
//! closes. Adding `Luna, clear` on top of that would have meant a blunt,
//! irreversible command operating on data whose only copy was process memory.
//!
//! So the document is written to disk on every change. Nothing here is on the
//! hot path: the file is rewritten when a sentence settles or a command runs,
//! which is a few times a minute, not per tick.
//!
//! # Why the file is deleted, and exactly when
//!
//! Leaving a file behind after every dictation session would be litter, so a
//! **clean** exit removes it unless `--persist` is given. An *unclean* exit
//! cannot remove it — there is no code running to do so — and that asymmetry is
//! the feature rather than an accident: the file survives precisely the cases
//! that used to lose the transcript, and is cleaned up in the case that never
//! did.

use anyhow::{Context, Result};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::transcript::Sentence;

/// Longest the utterance in flight may go unwritten.
///
/// The document is written the instant it changes, which is a few times a
/// minute. In-flight text changes on every tick, and putting an `fsync` on the
/// tick would be real cost for no gain — but leaving it out entirely was worse
/// than it looked. §5 means every ordinary pause merges, so between trims
/// *nothing* reaches the document and the in-flight buffer is where the whole
/// passage lives. Two seconds bounds what a crash can take to about one clause.
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
    /// Create the session file. `persist` keeps it on a clean exit.
    pub fn new(persist: bool) -> Result<Self> {
        let dir = state_dir();
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating {}", dir.display()))?;

        let path = dir.join(format!("session-{}.txt", stamp()));
        // Create it up front: a session that crashes before saying anything
        // should still leave evidence it ran, and an unwritable directory
        // should fail now rather than at the first sentence.
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

    /// Continue writing to a session file this program left behind.
    ///
    /// Resuming in place rather than starting a new file is what keeps the
    /// state directory from filling up with a stale copy per crash: the
    /// recovered session *is* this session, so it has one file, and `--persist`
    /// governs it exactly as it would any other.
    ///
    /// Only ever used on a file found in our own state directory. A path the
    /// user named is treated as read-only — see `Store::new` at the call site —
    /// because adopting it would mean a clean exit deleting something they
    /// pointed us at.
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

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Write the document, plus whatever is still in flight, if either moved.
    ///
    /// `in_flight` is the text the pipeline holds that has not reached the
    /// document yet. Passing it is what stops the safety net from depending on
    /// the buffer trim: mid-passage the trim is the *only* thing that files
    /// anything, so a passage the speaker never paused long enough to settle
    /// used to exist nowhere but process memory — exactly the situation this
    /// module was written to rule out.
    ///
    /// Returns an error message the first time it fails and `None` afterwards.
    /// Losing the autosave is worth saying once; it is not worth ending a
    /// dictation session over, because the in-memory document is still intact
    /// and still reaches stdout at exit.
    pub fn save(&mut self, document: &[Sentence], revision: u64, in_flight: &str) -> Option<String> {
        if self.failed {
            return None;
        }

        let settled = self.written != Some(revision);
        if !settled && self.in_flight == in_flight {
            return None;
        }

        // A growing utterance moves on every tick, so it is rate-limited rather
        // than written through. A *shrinking* one is not: it shrinks because the
        // document absorbed it, and deferring that would leave the file
        // momentarily claiming the same words twice.
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

    /// Clean exit. Removes the file unless `--persist` was given; returns the
    /// path when it was kept, so the caller can say where it is.
    pub fn finish(self) -> Option<PathBuf> {
        if self.persist {
            return Some(self.path);
        }
        let _ = std::fs::remove_file(&self.path);
        None
    }
}

/// The most recently touched session file, if there is one.
///
/// By modification time rather than by name: the name records when a session
/// *started*, and after a resume the interesting one is whichever was written
/// to last.
pub fn newest_session() -> Option<PathBuf> {
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;

    for entry in std::fs::read_dir(state_dir()).ok()?.flatten() {
        let path = entry.path();
        // `.tmp` is a rename that never landed, so it is at best a duplicate of
        // the file beside it and at worst half-written.
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

/// Read a session file back into sentences.
///
/// **Everything comes back kept.** The file records text, not the workflow
/// state that produced it, and asking to resume a transcript is itself an act
/// of approval — you looked at a file and said "that one". Restoring it as
/// pending would also mean `Luna, copy` came back empty on a resumed session,
/// which is the opposite of picking up where you left off.
pub fn load(path: &Path) -> Result<Vec<Sentence>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;

    Ok(text
        .lines()
        .map(|line| line.split_whitespace().map(str::to_string).collect::<Vec<_>>())
        .filter(|words| !words.is_empty())
        .map(|words| Sentence {
            words,
            committed: true,
        })
        .collect())
}

/// Write via a temporary file and rename.
///
/// A rename is atomic, so a crash midway leaves either the previous transcript
/// or the new one — never a half-written file, which is the one outcome worse
/// than not having autosaved at all.
fn write_atomically(path: &Path, document: &[Sentence], in_flight: &str) -> Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut file = std::fs::File::create(&tmp)?;
        for sentence in document {
            writeln!(file, "{}", sentence.text())?;
        }
        // The passage still being spoken, last and unmarked. It is deliberately
        // not distinguished from settled text: the file has to stay a clean
        // transcript, because `--persist` hands it to a human and `--resume`
        // reads it straight back. A clean exit files the tail first, so this
        // only survives into a file a crash left behind — which is the only case
        // where a partial last sentence beats no last sentence.
        //
        // Split the same way `file()` splits a finished utterance, so one line
        // is one sentence everywhere in this format. Measured on a 20 s passage
        // killed with `kill -9`, writing it whole recovered four sentences as a
        // single 230-character line — which `--resume` would then hand back as
        // one unrejectable blob.
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

fn state_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map_or_else(std::env::temp_dir, PathBuf::from);
    home.join(".local/state/speech-to-text-cli")
}

/// `YYYYmmdd-HHMMSS` in UTC, computed rather than pulled in with a date crate.
///
/// The only requirement is that concurrent sessions get different filenames and
/// that a human can tell which one is theirs.
fn stamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());

    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);

    // Civil-from-days, shifting the epoch to 0000-03-01 so leap days land at
    // the end of the cycle and the year/month arithmetic stays branch-free.
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

/// Put text on the system clipboard.
///
/// `pbcopy` rather than a crate: this project is macOS-only by construction
/// (§2 — MLX, Apple Silicon), and shelling out avoids a dependency that would
/// pull in a window-server connection for one string.
///
/// Both output streams are silenced. The live view owns the terminal, and a
/// child process writing to it would land in the middle of a frame — the same
/// invariant that made the sidecar's stderr a pipe rather than inherited.
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

    fn sentence(s: &str, committed: bool) -> Sentence {
        Sentence {
            words: s.split_whitespace().map(str::to_string).collect(),
            committed,
        }
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

    /// The file is the transcript, not a debug dump: committed and pending
    /// sentences alike, one per line, exactly as stdout prints them.
    #[test]
    fn the_file_holds_the_document_verbatim() {
        let dir = std::env::temp_dir().join(format!("stt-store-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("t.txt");

        let doc = [sentence("Kept one.", true), sentence("Still pending.", false)];
        write_atomically(&path, &doc, "").expect("write");

        assert_eq!(
            std::fs::read_to_string(&path).expect("read"),
            "Kept one.\nStill pending.\n"
        );

        // Rewriting replaces rather than appends, so a reject really shrinks
        // the file instead of leaving the dropped sentence behind.
        write_atomically(&path, &doc[..1], "").expect("rewrite");
        assert_eq!(std::fs::read_to_string(&path).expect("read"), "Kept one.\n");

        // And the temporary never survives a successful write.
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
            sentence("First one, with a comma.", true),
            sentence("Second one!", false),
            sentence("\u{4f60}\u{597d}\u{3002}", false),
        ];
        write_atomically(&path, &doc, "").expect("write");
        let back = load(&path).expect("load");

        let texts: Vec<String> = back.iter().map(Sentence::text).collect();
        assert_eq!(
            texts,
            ["First one, with a comma.", "Second one!", "\u{4f60}\u{597d}\u{3002}"]
        );
        // Everything returns kept: the file records text, not workflow state,
        // and asking to resume it is an approval of it.
        assert!(back.iter().all(|s| s.committed));

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("stt-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir.join("t.txt")
    }

    /// The failure the user hit: mid-passage the buffer trim is the only thing
    /// that files anything into the document, so a long merged passage left the
    /// session file empty of everything that had been said. The in-flight line
    /// is what makes the safety net independent of the trim.
    #[test]
    fn the_utterance_in_flight_reaches_the_file() {
        let path = scratch("inflight");
        let doc = [sentence("Already filed.", false)];

        write_atomically(&path, &doc, "and this is still being said").expect("write");
        assert_eq!(
            std::fs::read_to_string(&path).expect("read"),
            "Already filed.\nand this is still being said\n"
        );

        // It has to come back as ordinary recoverable text, since the whole
        // point is that `--resume` picks it up after a crash.
        let back = load(&path).expect("load");
        assert_eq!(
            back.iter().map(Sentence::text).collect::<Vec<_>>(),
            ["Already filed.", "and this is still being said"]
        );

        // And a passage that has not settled at all is still not lost.
        write_atomically(&path, &[], "nothing has settled yet").expect("write");
        assert_eq!(
            std::fs::read_to_string(&path).expect("read"),
            "nothing has settled yet\n"
        );

        // One line is one sentence everywhere in this format, so a long
        // in-flight passage is split the same way `file()` splits a finished
        // one. Measured: 20 s killed with `kill -9` recovered four sentences as
        // a single 230-character line before this.
        write_atomically(&path, &[], "One thought. Then a second. And a third").expect("write");
        assert_eq!(
            std::fs::read_to_string(&path).expect("read"),
            "One thought.\nThen a second.\nAnd a third\n"
        );

        // An empty tail must not leave a blank line behind: `split_sentences`
        // returns the input whole when it finds no boundary, empty included.
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
        let doc = [sentence("One.", false)];

        assert_eq!(store.save(&doc, 1, "i am talking"), None);
        assert_eq!(
            std::fs::read_to_string(&path).expect("read"),
            "One.\ni am talking\n"
        );

        // More words, same document, well inside the window: deferred.
        assert_eq!(store.save(&doc, 1, "i am talking and talking"), None);
        assert_eq!(
            std::fs::read_to_string(&path).expect("read"),
            "One.\ni am talking\n",
            "a growing utterance must not write through on every tick"
        );

        // The document moving overrides the rate limit, because a settled
        // sentence is exactly what this file exists to hold.
        let doc = [sentence("One.", false), sentence("Two.", false)];
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

        // Same revision, so only the in-flight line moved — and it shrank.
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
        let doc = [sentence("Settled.", true)];

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
