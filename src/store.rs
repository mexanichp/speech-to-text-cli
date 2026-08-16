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

use crate::transcript::Sentence;

/// The session file, rewritten whenever the document changes.
pub struct Store {
    path: PathBuf,
    persist: bool,
    /// Revision last written, so an unchanged document costs nothing.
    written: Option<u64>,
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
            failed: false,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Write the document if it has changed since the last write.
    ///
    /// Returns an error message the first time it fails and `None` afterwards.
    /// Losing the autosave is worth saying once; it is not worth ending a
    /// dictation session over, because the in-memory document is still intact
    /// and still reaches stdout at exit.
    pub fn save(&mut self, document: &[Sentence], revision: u64) -> Option<String> {
        if self.failed || self.written == Some(revision) {
            return None;
        }
        self.written = Some(revision);

        match write_atomically(&self.path, document) {
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
fn write_atomically(path: &Path, document: &[Sentence]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut file = std::fs::File::create(&tmp)?;
        for sentence in document {
            writeln!(file, "{}", sentence.text())?;
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
        write_atomically(&path, &doc).expect("write");

        assert_eq!(
            std::fs::read_to_string(&path).expect("read"),
            "Kept one.\nStill pending.\n"
        );

        // Rewriting replaces rather than appends, so a reject really shrinks
        // the file instead of leaving the dropped sentence behind.
        write_atomically(&path, &doc[..1]).expect("rewrite");
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
        write_atomically(&path, &doc).expect("write");
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
