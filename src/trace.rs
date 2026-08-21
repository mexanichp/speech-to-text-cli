//! Diagnostic trace of the path each sentence took into the document.
//!
//! Several code paths can file a sentence — logical commitment, the settle
//! expiring, an endpoint, a spoken command, the buffer trim, end of stream —
//! and a finished transcript does not record which one produced a given line.
//! This module attributes them.
//!
//! Disabled unless the `STT_TRACE` environment variable names a file. When
//! disabled, [`note`] costs one atomic load.
//!
//! # Invariants
//!
//! Output goes to the named file only, never to stdout or stderr. The renderer
//! owns the terminal while a session runs, and any write behind its back
//! corrupts the frame it is drawing.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Destination file and the session start used as the timestamp origin.
struct Sink {
    file: File,
    start: Instant,
}

static SINK: OnceLock<Option<Mutex<Sink>>> = OnceLock::new();

/// Returns the trace sink, opening it on first use.
///
/// Yields `None` when `STT_TRACE` is unset or the file cannot be opened.
fn sink() -> Option<&'static Mutex<Sink>> {
    SINK.get_or_init(|| {
        let path = std::env::var_os("STT_TRACE")?;
        let file = OpenOptions::new().create(true).append(true).open(path).ok()?;
        Some(Mutex::new(Sink {
            file,
            start: Instant::now(),
        }))
    })
    .as_ref()
}

/// Opens the trace file so timestamps are relative to session start.
///
/// Optional: [`note`] opens the file on demand. Calling this first only fixes
/// the timestamp origin at a well-defined point.
pub fn init() {
    sink();
}

/// Records one event, timestamped from session start.
///
/// # Parameters
///
/// - `kind`: the originating path, such as `logical`, `trim` or `endpoint`.
/// - `detail`: text identifying the event, typically the sentence filed.
///
/// Write failures are ignored: losing a diagnostic must not disturb a session,
/// and there is no safe channel on which to report it.
pub fn note(kind: &str, detail: &str) {
    let Some(sink) = sink() else { return };
    let Ok(mut sink) = sink.lock() else { return };
    let at = sink.start.elapsed().as_secs_f32();
    let _ = writeln!(sink.file, "{at:8.3}  {kind:<9}  {detail}");
    let _ = sink.file.flush();
}
