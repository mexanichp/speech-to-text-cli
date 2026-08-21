//! Terminal rendering of the live, still-editable transcript.
//!
//! The document is drawn on the alternate screen and re-derived from the model
//! on every frame. It cannot be appended to the scrollback while the session
//! runs, because spoken commands must be able to reach any sentence in it and
//! scrollback is the one region a program cannot take back. On exit the
//! alternate screen is dropped and the finished transcript is written to the
//! real scrollback once, at the point it can no longer change.
//!
//! When stdout is not a terminal there is no live view; the transcript is
//! written once at the end. The two paths produce identical text.
//!
//! # Styling contract
//!
//! | rendering | meaning |
//! |---|---|
//! | dim | the pipeline may still rewrite this |
//! | plain | filed; only the speaker can change it now |
//!
//! Everything in flight is dim, whether or not agreement has promoted it, and
//! text becomes plain when it files. Agreement is not rendered: inside an
//! utterance a continuation re-decodes the whole buffer and may rewrite any
//! word, so a plain word there would be claiming a permanence the pipeline does
//! not have.
//!
//! The gutter answers a different question, namely whether a row belongs to the
//! utterance being spoken now or to the transcript behind it.
//!
//! # Invariants
//!
//! 1. Every emitted row is strictly narrower than the terminal, including the
//!    gutter and any decoration added afterwards. At exactly the last column
//!    terminals disagree about whether the cursor advanced, so lines are broken
//!    here and the terminal's own wrap must never fire.
//! 2. Nothing else writes to the terminal while the live view is up. Diagnostics
//!    raised on other threads go through [`Renderer::notice`], which places them
//!    in the status line.
//!
//! Each frame addresses every row by absolute position and clears it before
//! painting, so no state carries between frames and the layout cannot drift.

use crate::transcript::Sentence;
use std::io::{IsTerminal, Write, stdout};

const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

const ALT_ENTER: &str = "\x1b[?1049h";
const ALT_LEAVE: &str = "\x1b[?1049l";
const CURSOR_HIDE: &str = "\x1b[?25l";
const CURSOR_SHOW: &str = "\x1b[?25h";

/// Marks the utterance in flight, as against the transcript behind it.
const LIVE_MARK: char = '\u{2502}';

/// Marks the top row when more transcript exists above than fits on screen.
///
/// The live tail is always drawn in full and outranks history, since a speaker
/// who has not finished must be able to read what they are saying. On a long
/// buffer or a short terminal the document therefore scrolls out of view, and
/// this marker distinguishes that from an empty document.
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

/// Everything one frame needs, borrowed from the model.
///
/// The renderer owns no transcript state, so a frame is a pure function of this
/// and the terminal size. That is what makes a full redraw always correct.
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
/// Marker drawn in the left margin of a row.
enum Gutter {
    /// A filed sentence: the pipeline is finished with it.
    None,
    /// The utterance in flight, settling or still being spoken.
    Live,
    /// Top row only: there is more transcript above than the screen holds.
    ///
    /// It displaces the row's own mark, which is the right trade — the rows it
    /// stands for are off screen entirely, and their absence matters more than
    /// one visible row's state.
    More,
}

/// One laid-out row: its gutter and its styled words.
struct Row<'a> {
    gutter: Gutter,
    cells: Vec<(&'a str, bool)>,
}

/// Owns the terminal while a session runs.
pub struct Renderer {
    tty: bool,
    notice: Option<(String, std::time::Instant)>,
}

impl Renderer {
    /// Takes over the screen when stdout is a terminal.
    ///
    /// Piped output gets no live view; the transcript is written once by
    /// [`Renderer::finish`]. The two paths differ only in presentation and
    /// produce identical text.
    pub fn new() -> Self {
        let tty = stdout().is_terminal();
        if tty {
            let mut out = stdout().lock();
            let _ = write!(out, "{ALT_ENTER}{CURSOR_HIDE}");
            let _ = out.flush();
        }
        Self { tty, notice: None }
    }

    /// Shows a diagnostic in the status line for [`NOTICE_HOLD`].
    ///
    /// The only permitted way to surface a message while the live view is up.
    /// Nothing is disturbed, because the next frame is re-derived from the
    /// model in any case. Without a terminal the message goes to stderr.
    pub fn notice(&mut self, msg: &str) {
        crate::trace::note("notice", msg);
        self.notice = Some((msg.to_string(), std::time::Instant::now()));
        if !self.tty {
            eprintln!("{msg}");
        }
    }

    /// Paints one frame, bottom-aligned so the newest text does not move.
    ///
    /// A no-op when stdout is not a terminal.
    pub fn draw(&mut self, frame: &Frame) {
        if !self.tty {
            return;
        }

        let (width, height) = dimensions();
        let (gutter_w, text_w) = budget(width);
        let body_rows = height.saturating_sub(1).max(1);

        let rows = build_rows(frame, text_w, body_rows);
        let status = self.status_line(frame, usable(width));

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

    /// Builds the status line: an active notice if one is held, otherwise the
    /// pipeline state, sentence count and command vocabulary.
    fn status_line(&mut self, frame: &Frame, usable: usize) -> String {
        if let Some((msg, at)) = &self.notice {
            if at.elapsed() < NOTICE_HOLD {
                return format!("{DIM}{}{RESET}", clip(msg, usable));
            }
            self.notice = None;
        }

        let state = match frame.state {
            State::Listening => "listening".to_string(),
            State::Speaking => "speaking".to_string(),
            State::Settling(s) => format!("settling {s}s"),
        };

        let n = frame.document.len();
        let sentences = if n == 1 {
            "1 sentence".to_string()
        } else {
            format!("{n} sentences")
        };

        let text = format!(
            "{state} \u{b7} {sentences} \u{b7} {}: delete discard keep undo clear copy",
            frame.wake,
        );
        format!("{DIM}{}{RESET}", clip(&text, usable))
    }

    /// Restores the terminal and writes the finished transcript to stdout.
    ///
    /// This is the only point at which anything reaches the scrollback, and the
    /// point at which the text stops being editable. Every sentence is written
    /// unconditionally.
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
    }
}

impl Drop for Renderer {
    /// Restores the terminal on every exit path, including unwinding from a
    /// panic. Leaving the alternate screen entered would return the user to a
    /// shell in which they cannot see what they type.
    fn drop(&mut self) {
        if self.tty {
            let mut out = stdout().lock();
            let _ = write!(out, "{ALT_LEAVE}{CURSOR_SHOW}");
            let _ = out.flush();
        }
    }
}

/// Appends a move to the start of row `row`, clearing that row.
fn at(buf: &mut String, row: usize) {
    buf.push_str(&format!("\x1b[{row};1H\x1b[2K"));
}

/// Builds the visible rows, newest first, stopping once the screen is full.
///
/// Walking backwards keeps the cost of a frame independent of session length,
/// which is why the document itself needs no size bound.
///
/// # Returns
///
/// Rows in display order, at most `max_rows` of them, with the top row marked
/// when anything was elided.
fn build_rows<'a>(frame: &Frame<'a>, text_w: usize, max_rows: usize) -> Vec<Row<'a>> {
    let mut rows: Vec<Row<'a>> = Vec::new();

    let live: Vec<(&str, bool)> = if frame.settling.is_empty() {
        frame
            .committed
            .iter()
            .chain(frame.provisional.iter())
            .map(|w| (w.as_str(), true))
            .collect()
    } else {
        frame.settling.iter().map(|w| (w.as_str(), true)).collect()
    };

    if !live.is_empty() {
        for cells in wrap(live.into_iter(), text_w).into_iter().rev() {
            rows.push(Row {
                gutter: Gutter::Live,
                cells,
            });
        }
    }

    let mut elided = rows.len() > max_rows;

    for sentence in frame.document.iter().rev() {
        if rows.len() >= max_rows {
            elided = true;
            break;
        }
        let cells = sentence.words.iter().map(|w| (w.as_str(), false));
        for cells in wrap(cells, text_w).into_iter().rev() {
            rows.push(Row {
                gutter: Gutter::None,
                cells,
            });
        }
    }

    rows.truncate(max_rows);
    rows.reverse();

    if elided && let Some(top) = rows.first_mut() {
        top.gutter = Gutter::More;
    }
    rows
}

/// Renders one row, gutter included.
fn style_row(row: &Row, gutter_w: usize) -> String {
    let mut out = String::new();
    if gutter_w > 0 {
        match row.gutter {
            Gutter::Live => out.push_str(&format!("{DIM}{LIVE_MARK}{RESET} ")),
            Gutter::More => out.push_str(&format!("{DIM}{MORE_MARK}{RESET} ")),
            Gutter::None => out.push_str(&" ".repeat(gutter_w)),
        }
    }
    out.push_str(&style_line(&row.cells));
    out
}

/// Columns usable for a row, always strictly inside the terminal width.
///
/// Floored at one rather than a comfortable minimum: clamping upward would
/// return a budget wider than the terminal and every row would auto-wrap. A
/// very narrow terminal may render unreadably but must not render wrongly.
fn usable(width: usize) -> usize {
    width.saturating_sub(1).max(1)
}

/// Splits the usable width between the gutter and the text.
///
/// The gutter is dropped entirely rather than narrowed when there is no room,
/// since decoration must never push text back to the auto-wrap boundary.
///
/// # Returns
///
/// `(gutter_width, text_width)`, where a gutter width of zero means none.
fn budget(width: usize) -> (usize, usize) {
    let usable = usable(width);
    if usable >= 6 {
        (2, usable - 2)
    } else {
        (0, usable)
    }
}

/// Wraps styled words greedily into lines of at most `usable` columns.
///
/// Breaking lines here rather than relying on the terminal is what makes the
/// row count exact.
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

/// Renders one line, emitting a single dim run.
///
/// Dim words are always a suffix of a line, so one run always suffices.
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

/// Splits a token too long for any line into pieces that fit, so wrapping
/// cannot stall.
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

/// Truncates text to a column budget, counting characters rather than bytes.
fn clip(text: &str, usable: usize) -> String {
    text.chars().take(usable).collect()
}

/// Terminal size, floored so there is always at least one usable column and
/// one body row.
///
/// # Returns
///
/// `(width, height)`, defaulting to 80x24 when the size cannot be determined.
fn dimensions() -> (usize, usize) {
    let (width, height) = terminal_size::terminal_size()
        .map(|(w, h)| (w.0 as usize, h.0 as usize))
        .unwrap_or((80, 24));
    (width.max(2), height.max(2))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    fn sentence(s: &str) -> Sentence {
        Sentence { words: words(s) }
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

    /// Everything in flight is dim, agreed or not. The split is still tracked
    /// as data — it arrives here in two slices — it just no longer earns two
    /// renderings, because the merge can rewrite the agreed half too.
    #[test]
    fn the_whole_live_tail_is_dim_including_the_agreed_head() {
        let doc = [];
        let committed = words("hello");
        let provisional = words("world");
        let rows = build_rows(&frame(&doc, &committed, &provisional), 40, 5);

        assert_eq!(rows.len(), 1);
        assert_eq!(
            style_line(&rows[0].cells),
            format!("{DIM}hello world{RESET}"),
            "agreed and provisional must render identically"
        );
    }

    /// The gutter answers "is this the utterance in flight, or the transcript?"
    /// Filed sentences are the transcript however recently they were filed, so
    /// none of them carries it — including the one filed a moment ago.
    #[test]
    fn the_gutter_marks_the_live_tail_and_nothing_else() {
        let doc = [sentence("Filed a while ago."), sentence("Filed just now.")];
        let committed = words("still");
        let provisional = words("speaking");
        let rows = build_rows(&frame(&doc, &committed, &provisional), 40, 5);

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].gutter, Gutter::None);
        assert_eq!(rows[1].gutter, Gutter::None);
        assert_eq!(rows[2].gutter, Gutter::Live, "the in-flight row");

        assert!(style_row(&rows[0], 2).starts_with("  "));
        assert!(style_row(&rows[1], 2).starts_with("  "));
        assert!(style_row(&rows[2], 2).contains(LIVE_MARK));
    }

    /// The gutter is not the dim/plain axis, and it is now the *only* thing
    /// distinguishing an agreed live row: it is dim like the rest of the tail,
    /// and marked because it is still the sentence being spoken.
    #[test]
    fn an_agreed_live_row_is_dim_and_still_marked() {
        let doc = [];
        let committed = words("already agreed");
        let rows = build_rows(&frame(&doc, &committed, &[]), 40, 5);

        assert_eq!(rows[0].gutter, Gutter::Live);
        assert_eq!(
            style_line(&rows[0].cells),
            format!("{DIM}already agreed{RESET}"),
            "nothing in flight renders plain"
        );
    }

    /// Text must never move from plain back to dim. Walks the three states an
    /// utterance passes through and requires the live tail to be dim in all of
    /// them, with only the document plain.
    #[test]
    fn text_never_moves_from_plain_back_to_dim() {
        let doc = [sentence("Filed earlier.")];

        let plain_rows = |rows: &[Row]| -> Vec<String> {
            rows.iter()
                .filter(|r| r.gutter == Gutter::None)
                .map(|r| style_line(&r.cells))
                .collect()
        };
        let live_rows = |rows: &[Row]| -> Vec<String> {
            rows.iter()
                .filter(|r| r.gutter == Gutter::Live)
                .map(|r| style_line(&r.cells))
                .collect()
        };

        let committed = words("the agreed head");
        let provisional = words("and the tail");
        let speaking = build_rows(&frame(&doc, &committed, &provisional), 40, 10);

        let settling = words("the agreed head and the tail");
        let empty = Vec::new();
        let paused = build_rows(
            &Frame {
                document: &doc,
                settling: &settling,
                committed: &empty,
                provisional: &empty,
                state: State::Settling(5),
                wake: "Luna",
            },
            40,
            10,
        );

        let reheard = words("the agreed head and the tail again");
        let resumed = build_rows(&frame(&doc, &empty, &reheard), 40, 10);

        for (what, rows) in [
            ("speaking", &speaking),
            ("settling", &paused),
            ("merged", &resumed),
        ] {
            for row in live_rows(rows) {
                assert!(
                    row.starts_with(DIM),
                    "{what}: in-flight text must be dim, got {row:?}"
                );
            }
            assert_eq!(
                plain_rows(rows),
                vec!["Filed earlier."],
                "{what}: only the document renders plain"
            );
        }
    }

    /// A settling sentence renders like provisional text, because a
    /// continuation within the hold re-decodes it and may change every word.
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
            .map(|i| sentence(&format!("Sentence number {i}.")))
            .collect();
        let committed = words("still");
        let provisional = words("speaking");
        let rows = build_rows(&frame(&doc, &committed, &provisional), 40, 5);

        assert_eq!(rows.len(), 5, "never more than the screen holds");
        let last = style_line(&rows[4].cells);
        assert!(last.contains("still") && last.contains("speaking"), "{last:?}");
        assert!(style_line(&rows[3].cells).contains("49"));
    }

    /// History scrolling off is allowed — the live tail outranks it — but it
    /// must never do so silently. That silence is what made an empty document
    /// look like a rendering bug for two rounds of diagnosis.
    #[test]
    fn history_pushed_off_screen_is_marked_rather_than_hidden() {
        let doc: Vec<Sentence> = (0..30)
            .map(|i| sentence(&format!("Sentence number {i}.")))
            .collect();
        let live = Vec::new();
        let rows = build_rows(&frame(&doc, &live, &live), 40, 5);

        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0].gutter, Gutter::More, "the top row must say so");
        assert!(style_row(&rows[0], 2).contains(MORE_MARK));
        assert!(rows[1..].iter().all(|r| r.gutter != Gutter::More));
    }

    /// The case the speaker actually hit: an utterance so long it fills the
    /// screen on its own. It is still drawn in full — never capped — but the
    /// document behind it is no longer silently gone.
    #[test]
    fn a_live_tail_that_fills_the_screen_still_marks_the_history_behind_it() {
        let doc = [sentence("Filed earlier.")];
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
        let doc = [sentence("One."), sentence("Two.")];
        let live = words("still talking");
        let rows = build_rows(&frame(&doc, &live, &[]), 40, 20);
        assert!(rows.iter().all(|r| r.gutter != Gutter::More));
    }

    /// The elision marker replaces a gutter that is already budgeted, so it
    /// cannot push a row onto the wrap boundary.
    #[test]
    fn the_marker_never_widens_a_row() {
        for width in 2..=24 {
            let (gutter_w, text_w) = budget(width);
            let doc: Vec<Sentence> = (0..20).map(|i| sentence(&format!("row {i} here"))).collect();
            let live = words("tail end of it");
            let rows = build_rows(&frame(&doc, &live, &[]), text_w, 3);
            assert_rows_fit(&rows, gutter_w, width);
        }
    }

    #[test]
    fn a_long_sentence_wraps_and_every_row_fits() {
        let doc = [sentence(&"word ".repeat(60))];
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
            let doc = [sentence("some words here"), sentence("more")];
            let live = words("tail end");
            let rows = build_rows(&frame(&doc, &live, &[]), text_w, 4);
            assert_rows_fit(&rows, gutter_w, width);
        }
    }

    #[test]
    fn an_overlong_token_is_hard_split_rather_than_stalling() {
        let doc = [sentence(&"x".repeat(300))];
        let live = Vec::new();
        let rows = build_rows(&frame(&doc, &live, &live), budget(80).1, 20);
        assert!(rows.len() >= 4);
        assert_rows_fit(&rows, 2, 80);
    }

    #[test]
    fn multibyte_text_does_not_panic_or_overflow() {
        let doc = [sentence(&"привет мир ".repeat(30))];
        let live = words("ещё");
        let (gutter_w, text_w) = budget(40);
        let rows = build_rows(&frame(&doc, &live, &[]), text_w, 6);
        assert_rows_fit(&rows, gutter_w, 40);
    }

    /// The status line is a row like any other and must not reach the width.
    #[test]
    fn the_status_line_is_clipped_to_the_terminal() {
        let doc = [sentence("kept.")];
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

    /// One count, and it agrees with itself grammatically. This replaced
    /// "N kept, M pending", whose second number was the most visible trace of a
    /// distinction that changed nothing about the text.
    #[test]
    fn the_status_line_counts_sentences_once() {
        let live = Vec::new();
        let mut r = Renderer { tty: false, notice: None };
        let line = |r: &mut Renderer, doc: &[Sentence]| {
            r.status_line(&frame(doc, &live, &live), usable(80))
        };

        assert!(line(&mut r, &[]).contains("0 sentences"));
        assert!(line(&mut r, &[sentence("One.")]).contains("1 sentence \u{b7}"));
        assert!(line(&mut r, &[sentence("One."), sentence("Two.")]).contains("2 sentences"));

        let all = line(&mut r, &[sentence("One.")]);
        assert!(!all.contains("kept") && !all.contains("pending"), "{all:?}");
    }

    /// It clips mid-word, so an ordinary terminal should not need clipping at
    /// all — otherwise the command hints are permanently truncated.
    #[test]
    fn the_status_line_fits_an_eighty_column_terminal_whole() {
        let doc = [sentence("kept."), sentence("pending.")];
        let live = Vec::new();
        let mut r = Renderer {
            tty: false,
            notice: None,
        };

        let line = r.status_line(&frame(&doc, &live, &live), usable(80));
        assert!(line.contains("copy"), "the last hint survived whole: {line:?}");
    }

    #[test]
    fn an_empty_session_renders_nothing_rather_than_panicking() {
        let doc = [];
        let live = Vec::new();
        let rows = build_rows(&frame(&doc, &live, &live), 40, 5);
        assert!(rows.is_empty());
    }
}
