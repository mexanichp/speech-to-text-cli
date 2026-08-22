//! Turning one guard off, for measurement only.
//!
//! Every change in §8 that was worth keeping was kept because a run with it and
//! a run without it came out different. Doing that by editing the source and
//! rebuilding works, and it is how D16 was settled, but it costs a rebuild per
//! arm, it cannot interleave two arms against the same thermal state, and it
//! leaves nothing behind that the next person can re-run.
//!
//! `STT_ABLATE` names guards to disable, comma-separated. Environment rather
//! than a flag, for the same reason `STT_TRACE` is: this is apparatus, not a
//! setting, and it has no business in `--help` where it would read as something
//! a speaker might want.
//!
//! ```sh
//! STT_ABLATE=seam-mark,same-speech ./target/release/speech-to-text-cli --simulate g.wav
//! ```
//!
//! Unknown names are rejected loudly at startup rather than ignored. A silently
//! misspelled arm produces two identical runs and the conclusion "no effect",
//! which is the one wrong answer this file exists to prevent.

use std::collections::HashSet;
use std::sync::OnceLock;

/// Every guard that can be switched off, and what switching it off does.
///
/// The list is exhaustive on purpose: it is what `--help` would carry if this
/// were a flag, and it is what makes a misspelling detectable.
pub const KNOWN: &[(&str, &str)] = &[
    ("seam-mark", "send cut lines with their full stop, so the pass cannot see the seam (D16)"),
    ("edit-floor", "no edit for runs below OVERLAP_TOLERANCE, so short filed text must match exactly"),
    ("same-speech", "match seams at equal lengths only, as `same_run` does"),
    ("stale-loop", "ask the stale check once rather than until it stops finding anything"),
    ("short-fragment", "leave fragments below SEAM_MIN alone"),
];

/// Numeric constants a measurement may move, and what each one is.
///
/// Separate from [`KNOWN`] because these are not guards: nothing is switched
/// off, a number is moved and the pipeline is asked what it thinks. §11 has
/// been asking for a sweep of these three since it was written, and the reason
/// it never happened is that moving a constant meant a rebuild per point.
pub const TUNABLE: &[(&str, &str)] = &[
    ("lag", "sentences held back from the pass (cleanup::LAG)"),
    ("min-batch", "fewest sentences worth one pass (cleanup::MIN_BATCH)"),
    ("batch", "most sentences in one pass (cleanup::BATCH)"),
];

static OFF: OnceLock<HashSet<String>> = OnceLock::new();
static TUNED: OnceLock<Vec<(String, usize)>> = OnceLock::new();

/// Reads `STT_ABLATE` once.
///
/// # Errors
///
/// Names a guard that does not exist, listing the ones that do.
pub fn init() -> Result<(), String> {
    let raw = std::env::var("STT_ABLATE").unwrap_or_default();
    let asked: HashSet<String> = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase)
        .collect();

    if let Some(unknown) = asked.iter().find(|a| !KNOWN.iter().any(|(k, _)| k == a)) {
        let known = KNOWN.iter().map(|(k, _)| *k).collect::<Vec<_>>().join(", ");
        return Err(format!("STT_ABLATE: no such guard {unknown:?}; known: {known}"));
    }

    let _ = OFF.set(asked);

    let raw = std::env::var("STT_TUNE").unwrap_or_default();
    let mut moved: Vec<(String, usize)> = Vec::new();
    for pair in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (name, value) = pair.split_once('=').ok_or_else(|| {
            format!("STT_TUNE: {pair:?} is not name=number")
        })?;
        let name = name.trim().to_lowercase();
        if !TUNABLE.iter().any(|(k, _)| *k == name) {
            let known = TUNABLE.iter().map(|(k, _)| *k).collect::<Vec<_>>().join(", ");
            return Err(format!("STT_TUNE: no such constant {name:?}; known: {known}"));
        }
        let value: usize = value
            .trim()
            .parse()
            .map_err(|_| format!("STT_TUNE: {name} wants a number, got {value:?}"))?;
        moved.push((name, value));
    }
    let _ = TUNED.set(moved);
    Ok(())
}

/// The value a measurement asked for, or the shipped constant.
///
/// The default is passed in rather than duplicated here, so the constant stays
/// where it is documented and this cannot drift from it.
pub fn tune(name: &str, default: usize) -> usize {
    TUNED
        .get()
        .and_then(|moved| moved.iter().find(|(k, _)| k == name))
        .map_or(default, |(_, v)| *v)
}

/// Whether `guard` has been switched off. False in a session that never called
/// [`init`], which is every test.
pub fn off(guard: &str) -> bool {
    OFF.get().is_some_and(|set| set.contains(guard))
}

/// What was switched off or moved, for the trace and the run's own record.
pub fn describe() -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    if let Some(set) = OFF.get() {
        let mut names: Vec<&str> = set.iter().map(String::as_str).collect();
        names.sort_unstable();
        parts.extend(names.into_iter().map(str::to_string));
    }
    if let Some(moved) = TUNED.get() {
        parts.extend(moved.iter().map(|(k, v)| format!("{k}={v}")));
    }
    match parts.is_empty() {
        true => None,
        false => Some(parts.join(",")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A misspelled guard is the failure this module is built to make loud: it
    /// produces two identical runs and the conclusion "no effect".
    #[test]
    fn every_known_guard_has_a_description() {
        for (name, why) in KNOWN {
            assert!(!name.is_empty() && !why.is_empty(), "{name} needs a description");
            assert_eq!(*name, name.to_lowercase(), "names are matched case-folded");
        }
    }

    /// Nothing is off unless a session asked for it, so the tests and the
    /// shipped binary take the same path.
    #[test]
    fn nothing_is_off_by_default() {
        assert!(!off("seam-mark"));
        assert!(!off("edit-floor"));
    }

    /// A constant nobody moved reads as itself, which is what keeps the
    /// shipped defaults in `cleanup.rs` rather than duplicated here.
    #[test]
    fn an_untuned_constant_is_its_default() {
        assert_eq!(tune("min-batch", 4), 4);
        assert_eq!(tune("lag", 3), 3);
    }

    #[test]
    fn every_tunable_has_a_description() {
        for (name, why) in TUNABLE {
            assert!(!name.is_empty() && !why.is_empty(), "{name} needs a description");
            assert_eq!(*name, name.to_lowercase());
        }
    }
}
