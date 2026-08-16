//! Terminal rendering: a live document the speaker can still edit by voice.
//!
//! # Why this is a full-screen view now
//!
//! The renderer used to append finished sentences to the scrollback, and that
//! made "once a sentence is printed, nothing can unprint it" a hard limit of
//! the product. `Luna, reject` has to reach a sentence the speaker already
//! committed, so the transcript cannot live in the scrollback while the session
//! is running — scrollback is exactly the one region a program cannot take
//! back.
//!
//! So the whole document lives on the **alternate screen** and is redrawn from
//! the model on every frame. Nothing is ever "taken back" there either; the
//! frame is simply re-derived, which is why there is still no `retract` and no
//! upward cursor arithmetic anywhere in this file. On exit the alternate screen
//! is dropped and the finished transcript is written to the real scrollback
//! once, in full — the point at which it genuinely can no longer change.
//!
//! # Invariants that carried over, and still bite
//!
//! 1. **Every emitted row is strictly narrower than the terminal**, including
//!    the gutter and anything decorative. At exactly the last column terminals
//!    disagree about whether the cursor advanced (deferred wrap), so relying on
//!    their auto-wrap makes row accounting off by one. We break lines
//!    ourselves; the terminal's own wrap must never fire.
//! 2. **Nothing writes to the terminal except this module** while the live view
//!    is up. A bare `eprintln!` from any thread lands in the middle of the
//!    frame. Off-thread diagnostics go through [`Renderer::notice`], which puts
//!    them in the status line instead.
//!
//! Absolute cursor positioning replaced the old relative arithmetic, so
//! `prev_rows` is gone: each frame addresses every row it paints by number and
//! clears it first, which cannot drift no matter what the previous frame did.
//!
//! # What the styling promises
//!
//! Three levels, and the ladder is the product:
//!
//! | rendering | meaning |
//! |---|---|
//! | dim text | the **pipeline** may still rewrite this |
//! | plain text, `│` gutter | settled; only the **speaker** can change it now |
//! | plain text, no gutter | the speaker committed it |
//!
//! Plus one that is about the screen rather than the text: a `⋮` gutter on the
//! top row means there is more transcript above than fits. The live tail is
//! never truncated to make room for history — §1 — so on a long buffer the
//! document scrolls out of view, and this says so instead of letting it vanish
//! silently.
//!
//! Dim still means exactly what it always meant. A speaker rejecting their own
//! sentence is not the pipeline changing its mind, so plain text keeps its
//! promise even though a committed sentence can still leave the screen.

use crate::transcript::Sentence;
use std::io::{IsTerminal, Write, stdout};

const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

const ALT_ENTER: &str = "\x1b[?1049h";
const ALT_LEAVE: &str = "\x1b[?1049l";
const CURSOR_HIDE: &str = "\x1b[?25l";
const CURSOR_SHOW: &str = "\x1b[?25h";

/// Marks a sentence the speaker has not committed yet.
const PENDING_MARK: char = '\u{2502}';

/// Marks the top row when there is more transcript above it than fits.
///
/// The live tail is drawn in full and outranks history — that is the §1 stance,
/// and it is deliberate: a speaker who has not finished has to be able to read
/// what they are saying. The consequence is that on a long buffer or a short
/// terminal the document scrolls out of view, and **it used to do so silently**,
/// which is exactly how an empty document (see §6) went undiagnosed as a
/// rendering problem for two rounds.
///
/// So this is not a cap. Everything live still renders; this only stops the
/// screen from claiming that what it shows is all there is.
const MORE_MARK: char = '\u{22ee}';

/// How long a diagnostic holds the status line before the ordinary status
/// comes back.
const NOTICE_HOLD: std::time::Duration = std::time::Duration::from_secs(6);

/// What the pipeline is doing, for the status line.
pub enum State {
    Listening,
    Speaking,
    /// Ended, but still inside the grace window: seconds left before it settles.
    Settling(u64),
}

/// Everything one frame needs. Borrowed, because the renderer owns no
/// transcript state — it is a pure function of the model plus the terminal
/// size, which is what makes a full redraw always correct.
pub struct Frame<'a> {
    pub document: &'a [Sentence],
    /// An utterance that ended and is waiting out the grace window. Dim: a
    /// continuation re-decodes it from the audio and can rewrite every word.
    pub settling: &'a [String],
    /// In-flight, agreed by LocalAgreement.
    pub committed: &'a [String],
    /// In-flight, still subject to revision.
    pub provisional: &'a [String],
    pub state: State,
    pub wake: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gutter {
    /// The speaker committed this.
    None,
    /// Settled or in flight — `Luna, commit` has not reached it.
    Pending,
    /// Top row only: there is more transcript above than the screen holds.
    ///
    /// It displaces the row's own commit mark, which is the right trade — the
    /// rows it stands for are off screen entirely, and their absence matters
    /// more than one visible row's state.
    More,
}

struct Row<'a> {
    gutter: Gutter,
    cells: Vec<(&'a str, bool)>,
}

pub struct Renderer {
    tty: bool,
    notice: Option<(String, std::time::Instant)>,
}

impl Renderer {
    /// Takes over the screen when stdout is a terminal.
    ///
    /// Piped output gets no live view at all — the transcript is written once,
    /// at the end, by [`finish`](Renderer::finish). The two paths differ in
    /// presentation only; the text they produce is identical, which is the
    /// property that keeps `--simulate | diff` an honest test.
    pub fn new() -> Self {
        let tty = stdout().is_terminal();
        if tty {
            let mut out = stdout().lock();
            let _ = write!(out, "{ALT_ENTER}{CURSOR_HIDE}");
            let _ = out.flush();
        }
        Self { tty, notice: None }
    }

    /// Show a diagnostic without disturbing the transcript.
    ///
    /// The old renderer had to erase the live region to print one of these, and
    /// then restore whatever had been settling in it. A frame is re-derived
    /// from the model every draw, so there is nothing to restore: the message
    /// simply occupies the status line for a few seconds.
    pub fn notice(&mut self, msg: &str) {
        self.notice = Some((msg.to_string(), std::time::Instant::now()));
        if !self.tty {
            // Nothing owns the terminal, so a diagnostic can just be printed.
            eprintln!("{msg}");
        }
    }

    pub fn draw(&mut self, frame: &Frame) {
        if !self.tty {
            return;
        }

        let (width, height) = dimensions();
        let (gutter_w, text_w) = budget(width);
        let body_rows = height.saturating_sub(1).max(1);

        let rows = build_rows(frame, text_w, body_rows);
        let status = self.status_line(frame, usable(width));

        // Bottom-aligned: the newest text sits immediately above the status
        // line and stays there. The speaker is reading along with what they are
        // saying right now, so it must not move around under their eye.
        let top_blank = body_rows - rows.len();

        let mut buf = String::new();
        for r in 0..body_rows {
            at(&mut buf, r + 1);
            if let Some(row) = r.checked_sub(top_blank).and_then(|i| rows.get(i)) {
                buf.push_str(&style_row(row, gutter_w));
            }
        }
        at(&mut buf, height);
        buf.push_str(&status);

        let mut out = stdout().lock();
        let _ = write!(out, "{buf}");
        let _ = out.flush();
    }

    fn status_line(&mut self, frame: &Frame, usable: usize) -> String {
        if let Some((msg, at)) = &self.notice {
            if at.elapsed() < NOTICE_HOLD {
                return format!("{DIM}{}{RESET}", clip(msg, usable));
            }
            self.notice = None;
        }

        let pending = frame.document.iter().filter(|s| !s.committed).count();
        let state = match frame.state {
            State::Listening => "listening".to_string(),
            State::Speaking => "speaking".to_string(),
            State::Settling(s) => format!("settling {s}s"),
        };

        // Kept short enough to fit an 80-column terminal whole. `clip` will
        // save us on anything narrower, but it cuts mid-word, so the common
        // case should not need it.
        let text = format!(
            "{state} \u{b7} {} kept, {pending} pending \u{b7} {}: commit reject clear copy undo",
            frame.document.len() - pending,
            frame.wake,
        );
        format!("{DIM}{}{RESET}", clip(&text, usable))
    }

    /// Give the terminal back and write the transcript where it will persist.
    ///
    /// This is the one moment the text genuinely stops being editable, and it
    /// is also the only moment anything reaches the scrollback — so the styling
    /// contract holds trivially: everything written here is plain, and nothing
    /// can change it afterwards.
    pub fn finish(&mut self, document: &[Sentence]) {
        if self.tty {
            let mut out = stdout().lock();
            let _ = write!(out, "{ALT_LEAVE}{CURSOR_SHOW}");
            let _ = out.flush();
            self.tty = false;
        }

        let mut out = stdout().lock();
        for sentence in document {
            let _ = writeln!(out, "{}", sentence.text());
        }
        let _ = out.flush();

        // Uncommitted text is still printed — losing what someone dictated
        // because they forgot the magic words would be indefensible. But it
        // goes on the record, on stderr so it cannot pollute the transcript.
        let pending = document.iter().filter(|s| !s.committed).count();
        if pending > 0 {
            eprintln!("\n({pending} sentence(s) were never committed)");
        }
    }
}

impl Drop for Renderer {
    /// Restores the terminal on any exit path, including a panic — unwinding
    /// runs this. Leaving the alternate screen entered would hand the user back
    /// a shell they cannot see what they are typing in.
    fn drop(&mut self) {
        if self.tty {
            let mut out = stdout().lock();
            let _ = write!(out, "{ALT_LEAVE}{CURSOR_SHOW}");
            let _ = out.flush();
        }
    }
}

/// Move to the start of row `row`, clearing it.
fn at(buf: &mut String, row: usize) {
    buf.push_str(&format!("\x1b[{row};1H\x1b[2K"));
}

/// Newest-first row assembly, stopping as soon as the screen is full.
///
/// Walking backwards is what keeps drawing independent of session length: an
/// hour of dictation costs the same frame as the first sentence, because
/// everything above the top row is never touched. This is why the document
/// itself needs no size bound.
fn build_rows<'a>(frame: &Frame<'a>, text_w: usize, max_rows: usize) -> Vec<Row<'a>> {
    let mut rows: Vec<Row<'a>> = Vec::new();

    // The live tail is always drawn: it is what the speaker is reading along
    // with, so it outranks any amount of history.
    let live: Vec<(&str, bool)> = if frame.settling.is_empty() {
        frame
            .committed
            .iter()
            .map(|w| (w.as_str(), false))
            .chain(frame.provisional.iter().map(|w| (w.as_str(), true)))
            .collect()
    } else {
        frame.settling.iter().map(|w| (w.as_str(), true)).collect()
    };

    if !live.is_empty() {
        for cells in wrap(live.into_iter(), text_w).into_iter().rev() {
            rows.push(Row {
                gutter: Gutter::Pending,
                cells,
            });
        }
    }

    // Whether anything the document holds could not be drawn. Tracked rather
    // than inferred afterwards, because the two ways it happens are different:
    // the loop below runs out of screen, or the live tail alone already
    // overflowed it.
    let mut elided = rows.len() > max_rows;

    for sentence in frame.document.iter().rev() {
        if rows.len() >= max_rows {
            elided = true;
            break;
        }
        let gutter = if sentence.committed {
            Gutter::None
        } else {
            Gutter::Pending
        };
        let cells = sentence.words.iter().map(|w| (w.as_str(), false));
        for cells in wrap(cells, text_w).into_iter().rev() {
            rows.push(Row { gutter, cells });
        }
    }

    rows.truncate(max_rows);
    rows.reverse();

    // The marker replaces the top row's own gutter rather than being prepended
    // to its text. §7 records why: an elision marker added *outside* the width
    // budget put a full row exactly on the terminal boundary and armed deferred
    // wrap. The gutter is already budgeted and already one column plus a space,
    // so this cannot change any row's width.
    if elided && let Some(top) = rows.first_mut() {
        top.gutter = Gutter::More;
    }
    rows
}

fn style_row(row: &Row, gutter_w: usize) -> String {
    let mut out = String::new();
    if gutter_w > 0 {
        match row.gutter {
            Gutter::Pending => out.push_str(&format!("{DIM}{PENDING_MARK}{RESET} ")),
            Gutter::More => out.push_str(&format!("{DIM}{MORE_MARK}{RESET} ")),
            Gutter::None => out.push_str(&" ".repeat(gutter_w)),
        }
    }
    out.push_str(&style_line(&row.cells));
    out
}

/// Rows must stay strictly under the terminal width so auto-wrap never fires.
///
/// The floor is 1, not some comfortable minimum: clamping *up* would hand back
/// a budget wider than the terminal and every row would auto-wrap. A
/// one-column terminal renders unreadably, which is fine; it must not render
/// *wrongly*.
fn usable(width: usize) -> usize {
    width.saturating_sub(1).max(1)
}

/// Split the usable width between the gutter and the text.
///
/// The gutter is dropped entirely rather than squeezed when there is no room
/// for it, because a two-column decoration on a six-column terminal would push
/// the text back out to the auto-wrap boundary the whole module exists to stay
/// inside.
fn budget(width: usize) -> (usize, usize) {
    let usable = usable(width);
    if usable >= 6 {
        (2, usable - 2)
    } else {
        (0, usable)
    }
}

/// Greedy word wrap, reporting exact row counts because it breaks the lines
/// itself rather than leaving it to the terminal.
fn wrap<'a>(
    tokens: impl Iterator<Item = (&'a str, bool)>,
    usable: usize,
) -> Vec<Vec<(&'a str, bool)>> {
    let mut lines: Vec<Vec<(&str, bool)>> = vec![Vec::new()];
    let mut used = 0usize;

    for (word, dim) in tokens {
        for piece in hard_split(word, usable) {
            let len = piece.chars().count();
            let need = if used == 0 { len } else { len + 1 };

            if used > 0 && used + need > usable {
                lines.push(Vec::new());
                used = 0;
            }
            used += if used == 0 { len } else { len + 1 };
            lines.last_mut().expect("non-empty").push((piece, dim));
        }
    }
    lines
}

/// Provisional words are always a suffix of the line, so one dim run suffices.
fn style_line(line: &[(&str, bool)]) -> String {
    let split = line.iter().position(|(_, dim)| *dim).unwrap_or(line.len());
    let plain: Vec<&str> = line[..split].iter().map(|(w, _)| *w).collect();
    let dim: Vec<&str> = line[split..].iter().map(|(w, _)| *w).collect();

    match (plain.is_empty(), dim.is_empty()) {
        (true, true) => String::new(),
        (false, true) => plain.join(" "),
        (true, false) => format!("{DIM}{}{RESET}", dim.join(" ")),
        (false, false) => format!("{} {DIM}{}{RESET}", plain.join(" "), dim.join(" ")),
    }
}

/// Break a token too long for any line, so wrapping can never stall.
fn hard_split(word: &str, usable: usize) -> Vec<&str> {
    if word.chars().count() <= usable {
        return vec![word];
    }
    let mut parts = Vec::new();
    let (mut start, mut count) = (0usize, 0usize);
    for (i, _) in word.char_indices() {
        if count == usable {
            parts.push(&word[start..i]);
            start = i;
            count = 0;
        }
        count += 1;
    }
    parts.push(&word[start..]);
    parts
}

/// Truncate to a column budget, counting characters rather than bytes.
fn clip(text: &str, usable: usize) -> String {
    text.chars().take(usable).collect()
}

fn dimensions() -> (usize, usize) {
    let (width, height) = terminal_size::terminal_size()
        .map(|(w, h)| (w.0 as usize, h.0 as usize))
        .unwrap_or((80, 24));
    // Floored at 2 so [`usable`] always has at least one column to give that is
    // still strictly inside the terminal, and at 2 rows so there is a body row
    // as well as a status line.
    (width.max(2), height.max(2))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    fn sentence(s: &str, committed: bool) -> Sentence {
        Sentence {
            words: words(s),
            committed,
        }
    }

    fn frame<'a>(
        document: &'a [Sentence],
        committed: &'a [String],
        provisional: &'a [String],
    ) -> Frame<'a> {
        Frame {
            document,
            settling: &[],
            committed,
            provisional,
            state: State::Speaking,
            wake: "Luna",
        }
    }

    /// Rendered width of a row, escapes excluded.
    fn visible(row: &str) -> usize {
        row.replace(DIM, "").replace(RESET, "").chars().count()
    }

    /// The invariant the whole module rests on: no row may reach the terminal
    /// width, or the terminal's own wrap fires and the row count stops matching
    /// the screen.
    fn assert_rows_fit(rows: &[Row], gutter_w: usize, width: usize) {
        for row in rows {
            let painted = style_row(row, gutter_w);
            assert!(
                visible(&painted) < width,
                "row of {} chars must stay under width {width}: {painted:?}",
                visible(&painted)
            );
        }
    }

    #[test]
    fn the_live_tail_is_split_into_committed_and_provisional() {
        let doc = [];
        let committed = words("hello");
        let provisional = words("world");
        let rows = build_rows(&frame(&doc, &committed, &provisional), 40, 5);

        assert_eq!(rows.len(), 1);
        assert_eq!(style_line(&rows[0].cells), format!("hello {DIM}world{RESET}"));
    }

    /// The three-level ladder, which is the product: dim text is the pipeline's
    /// to change, the gutter marks what the speaker has not approved yet.
    #[test]
    fn committed_sentences_lose_the_pending_gutter() {
        let doc = [sentence("Approved.", true), sentence("Not yet.", false)];
        let live = Vec::new();
        let rows = build_rows(&frame(&doc, &live, &live), 40, 5);

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].gutter, Gutter::None);
        assert_eq!(rows[1].gutter, Gutter::Pending);
        assert!(style_row(&rows[0], 2).starts_with("  "));
        assert!(style_row(&rows[1], 2).contains(PENDING_MARK));
    }

    /// A settling sentence renders exactly like provisional text, because that
    /// is what it is: a continuation inside the grace window re-decodes it and
    /// can change every word. Plain text has to keep meaning "the pipeline is
    /// done with this".
    #[test]
    fn a_settling_sentence_is_entirely_dim() {
        let doc = [];
        let settling = words("this might still change");
        let live = Vec::new();
        let f = Frame {
            document: &doc,
            settling: &settling,
            committed: &live,
            provisional: &live,
            state: State::Settling(7),
            wake: "Luna",
        };
        let rows = build_rows(&f, 40, 5);
        assert_eq!(
            style_line(&rows[0].cells),
            format!("{DIM}this might still change{RESET}")
        );
    }

    /// The live tail outranks history: it is what the speaker is reading along
    /// with, so it must survive however much text is above it.
    #[test]
    fn the_live_tail_survives_a_full_screen_of_history() {
        let doc: Vec<Sentence> = (0..50)
            .map(|i| sentence(&format!("Sentence number {i}."), true))
            .collect();
        let committed = words("still");
        let provisional = words("speaking");
        let rows = build_rows(&frame(&doc, &committed, &provisional), 40, 5);

        assert_eq!(rows.len(), 5, "never more than the screen holds");
        let last = style_line(&rows[4].cells);
        assert!(last.contains("still") && last.contains("speaking"), "{last:?}");
        // And the newest history is what got kept above it.
        assert!(style_line(&rows[3].cells).contains("49"));
    }

    /// History scrolling off is allowed — the live tail outranks it — but it
    /// must never do so silently. That silence is what made an empty document
    /// look like a rendering bug for two rounds of diagnosis.
    #[test]
    fn history_pushed_off_screen_is_marked_rather_than_hidden() {
        let doc: Vec<Sentence> = (0..30)
            .map(|i| sentence(&format!("Sentence number {i}."), true))
            .collect();
        let live = Vec::new();
        let rows = build_rows(&frame(&doc, &live, &live), 40, 5);

        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0].gutter, Gutter::More, "the top row must say so");
        assert!(style_row(&rows[0], 2).contains(MORE_MARK));
        // Only the top row, and the rest keep their own meaning.
        assert!(rows[1..].iter().all(|r| r.gutter != Gutter::More));
    }

    /// The case the speaker actually hit: an utterance so long it fills the
    /// screen on its own. It is still drawn in full — never capped — but the
    /// document behind it is no longer silently gone.
    #[test]
    fn a_live_tail_that_fills_the_screen_still_marks_the_history_behind_it() {
        let doc = [sentence("Filed earlier.", true)];
        let committed = words(&"word ".repeat(200));
        let rows = build_rows(&frame(&doc, &committed, &[]), 40, 4);

        assert_eq!(rows.len(), 4, "the live tail keeps every row it needs");
        assert_eq!(rows[0].gutter, Gutter::More);
        assert!(
            rows.iter().all(|r| r.cells.iter().all(|(w, _)| *w == "word")),
            "and none of it was dropped to make room for history"
        );
    }

    /// A screen that holds everything must not claim otherwise.
    #[test]
    fn nothing_is_marked_when_it_all_fits() {
        let doc = [sentence("One.", true), sentence("Two.", false)];
        let live = words("still talking");
        let rows = build_rows(&frame(&doc, &live, &[]), 40, 20);
        assert!(rows.iter().all(|r| r.gutter != Gutter::More));
    }

    /// The marker replaces a gutter that was already budgeted, so it cannot
    /// push a row onto the wrap boundary — the failure §7 records for the last
    /// elision marker this file had.
    #[test]
    fn the_marker_never_widens_a_row() {
        for width in 2..=24 {
            let (gutter_w, text_w) = budget(width);
            let doc: Vec<Sentence> = (0..20).map(|i| sentence(&format!("row {i} here"), i % 2 == 0)).collect();
            let live = words("tail end of it");
            let rows = build_rows(&frame(&doc, &live, &[]), text_w, 3);
            assert_rows_fit(&rows, gutter_w, width);
        }
    }

    #[test]
    fn a_long_sentence_wraps_and_every_row_fits() {
        let doc = [sentence(&"word ".repeat(60), false)];
        let live = Vec::new();
        let rows = build_rows(&frame(&doc, &live, &live), budget(80).1, 20);
        assert!(rows.len() > 1, "300 chars at width 80 must wrap");
        assert_rows_fit(&rows, 2, 80);
    }

    /// A terminal narrower than the gutter used to be handed a text budget that
    /// pushed rows back out to the auto-wrap boundary.
    #[test]
    fn narrow_terminals_still_produce_rows_that_fit() {
        for width in 2..=12 {
            let (gutter_w, text_w) = budget(width);
            let doc = [sentence("some words here", false), sentence("more", true)];
            let live = words("tail end");
            let rows = build_rows(&frame(&doc, &live, &[]), text_w, 4);
            assert_rows_fit(&rows, gutter_w, width);
        }
    }

    #[test]
    fn an_overlong_token_is_hard_split_rather_than_stalling() {
        let doc = [sentence(&"x".repeat(300), false)];
        let live = Vec::new();
        let rows = build_rows(&frame(&doc, &live, &live), budget(80).1, 20);
        assert!(rows.len() >= 4);
        assert_rows_fit(&rows, 2, 80);
    }

    #[test]
    fn multibyte_text_does_not_panic_or_overflow() {
        let doc = [sentence(&"привет мир ".repeat(30), false)];
        let live = words("ещё");
        let (gutter_w, text_w) = budget(40);
        let rows = build_rows(&frame(&doc, &live, &[]), text_w, 6);
        assert_rows_fit(&rows, gutter_w, 40);
    }

    /// The status line is a row like any other and must not reach the width.
    #[test]
    fn the_status_line_is_clipped_to_the_terminal() {
        let doc = [sentence("kept.", true)];
        let live = Vec::new();
        let mut r = Renderer {
            tty: false,
            notice: None,
        };

        for width in [2usize, 10, 40, 80, 200] {
            let line = r.status_line(&frame(&doc, &live, &live), usable(width));
            let visible = line.replace(DIM, "").replace(RESET, "").chars().count();
            assert!(visible < width, "status of {visible} at width {width}");
        }
    }

    /// It clips mid-word, so an ordinary terminal should not need clipping at
    /// all — otherwise the command hints are permanently truncated.
    #[test]
    fn the_status_line_fits_an_eighty_column_terminal_whole() {
        let doc = [sentence("kept.", true), sentence("pending.", false)];
        let live = Vec::new();
        let mut r = Renderer {
            tty: false,
            notice: None,
        };

        let line = r.status_line(&frame(&doc, &live, &live), usable(80));
        assert!(line.contains("undo"), "the last hint survived whole: {line:?}");
    }

    #[test]
    fn an_empty_session_renders_nothing_rather_than_panicking() {
        let doc = [];
        let live = Vec::new();
        let rows = build_rows(&frame(&doc, &live, &live), 40, 5);
        assert!(rows.is_empty());
    }
}
