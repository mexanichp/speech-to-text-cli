//! Locating the Python half of the pipeline.
//!
//! The binary owns everything except the forward pass, but the forward pass
//! needs an interpreter and two scripts, and neither of those can be assumed to
//! sit below the working directory once the binary is installed rather than
//! built in place.
//!
//! Both are solved the same way: the scripts are compiled into the binary and
//! written out to a cache directory, and the interpreter is either found or
//! built. What ships is one file.
//!
//! # Resolution order
//!
//! An explicit `--python` wins, then `STT_PYTHON`, then a `.venv` below the
//! working directory, then the managed environment under [`cache_dir`]. The
//! `.venv` rule is what keeps the development flow in this repository working
//! unchanged: build, run, and the interpreter that gets picked is the one the
//! measurements were taken with.
//!
//! # Invariants
//!
//! Everything here runs before [`crate::render::Renderer`] takes the terminal,
//! so it is the one part of the program that may write to stderr directly and
//! may hand a child process inherited stdio. Invariant 1 applies from the moment
//! the live view is up, which is after this module is finished.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The recognition sidecar, carried by the binary.
const ASR: &str = include_str!("../sidecar/asr_sidecar.py");

/// The cleanup sidecar, carried by the binary.
const CLEANUP: &str = include_str!("../sidecar/cleanup_sidecar.py");

/// The pinned Python environment, carried by the binary.
const REQUIREMENTS: &str = include_str!("../requirements.txt");

/// Interpreter version the managed environment is built at.
///
/// Pinned rather than left to whatever the host has, and pinned to the version
/// every measurement in `CLAUDE.md` was taken against. `uv` downloads a managed
/// build when the host has no such interpreter, so this costs a download rather
/// than a prerequisite. Every package in `requirements.txt` has an arm64 wheel
/// here, which is what makes a bootstrap a download and never a compile.
const PYTHON_VERSION: &str = "3.14";

/// Name of the file recording which pinned set an environment was built from.
const STAMP: &str = ".stt-requirements";

/// Where the Python side of the pipeline was found.
pub struct Layout {
    /// Interpreter that can import `mlx`.
    pub python: PathBuf,
    /// Recognition sidecar script, on disk.
    pub asr: PathBuf,
    /// Cleanup sidecar script, on disk.
    pub cleanup: PathBuf,
}

/// Finds or builds everything the sidecars need to start.
///
/// # Parameters
///
/// - `python`: an explicit `--python`, taken as given and never probed. A caller
///   who names an interpreter has said what they want, and failing loudly on it
///   beats silently bootstrapping a second one.
/// - `asr`, `cleanup`: explicit script paths, overriding the embedded copies.
///   For editing a sidecar without a rebuild.
///
/// # Errors
///
/// Fails if a path the caller named does not exist, if the cache directory
/// cannot be written, or if no interpreter could be found and none could be
/// built.
pub fn resolve(python: Option<&Path>, asr: Option<&Path>, cleanup: Option<&Path>) -> Result<Layout> {
    Ok(Layout {
        python: interpreter(python)?,
        asr: match asr {
            Some(path) => given(path, "--script")?,
            None => materialize("asr_sidecar.py", ASR)?,
        },
        cleanup: match cleanup {
            Some(path) => given(path, "--cleanup-script")?,
            None => materialize("cleanup_sidecar.py", CLEANUP)?,
        },
    })
}

/// Accepts a path the caller named, or reports that it is not there.
///
/// A missing file here is a typo rather than a condition to recover from, so it
/// is reported against the flag that carried it.
fn given(path: &Path, flag: &str) -> Result<PathBuf> {
    if !path.exists() {
        bail!("{flag} points at {}, which does not exist", path.display());
    }
    Ok(path.to_path_buf())
}

/// Writes an embedded script to the cache directory, and returns where.
///
/// Rewritten only when the content differs, so an upgraded binary replaces the
/// scripts it carries without the common case paying a write. Comparing content
/// rather than stamping a version also covers the case that matters during
/// development, where the version has not changed but the script has.
fn materialize(name: &str, body: &str) -> Result<PathBuf> {
    let dir = cache_dir().join("sidecar");
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;

    let path = dir.join(name);
    if std::fs::read_to_string(&path).is_ok_and(|on_disk| on_disk == body) {
        return Ok(path);
    }
    std::fs::write(&path, body).with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

/// Walks the resolution order, building the managed environment if it has to.
fn interpreter(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return given(path, "--python");
    }
    if let Some(set) = std::env::var_os("STT_PYTHON") {
        return given(Path::new(&set), "STT_PYTHON");
    }

    // The development flow: a `.venv` below the working directory is the one
    // this repository's README builds and every measurement was taken with.
    let local = PathBuf::from(".venv/bin/python");
    if local.exists() {
        return Ok(local);
    }

    let managed = cache_dir().join("venv");
    let python = managed.join("bin/python");
    if python.exists() && stamp_matches(&managed) {
        return Ok(python);
    }

    bootstrap(&managed)?;
    Ok(python)
}

/// Whether the managed environment was built from the pinned set we carry.
///
/// The stamp is written only after a successful install, so a bootstrap that
/// died halfway leaves no stamp and is retried rather than trusted. It also
/// makes an upgraded binary that moved a pin rebuild rather than run against the
/// environment the previous one left.
fn stamp_matches(venv: &Path) -> bool {
    std::fs::read_to_string(venv.join(STAMP)).is_ok_and(|found| found == REQUIREMENTS)
}

/// Builds the managed environment with `uv`.
///
/// Progress is inherited rather than captured: this runs once, it takes tens of
/// seconds, and a silent pause there reads as a hang. The first run downloads
/// several gigabytes of model weights immediately afterwards, so a visible
/// install is the smaller half of what the user is already waiting for.
fn bootstrap(venv: &Path) -> Result<()> {
    let probe = Command::new("uv")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    if !probe.is_ok_and(|status| status.success()) {
        bail!(
            "no Python environment for the sidecars, and `uv` is not on PATH to build one.\n\
             \n    brew install uv\n\n\
             Or point at an interpreter that already has the packages in \
             requirements.txt:\n\
             \n    --python /path/to/python"
        );
    }

    let dir = cache_dir();
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let reqs = dir.join("requirements.txt");
    std::fs::write(&reqs, REQUIREMENTS).with_context(|| format!("writing {}", reqs.display()))?;

    eprintln!("setting up the Python environment in {} (once)", dir.display());

    run(Command::new("uv")
        .arg("venv")
        .arg("--python")
        .arg(PYTHON_VERSION)
        .arg(venv))?;

    run(Command::new("uv")
        .arg("pip")
        .arg("install")
        .arg("--python")
        .arg(venv.join("bin/python"))
        .arg("--requirement")
        .arg(&reqs))?;

    std::fs::write(venv.join(STAMP), REQUIREMENTS)
        .with_context(|| format!("stamping {}", venv.display()))?;
    Ok(())
}

/// Runs one bootstrap step, failing on anything but a clean exit.
fn run(cmd: &mut Command) -> Result<()> {
    let shown = format!("{cmd:?}");
    let status = cmd.status().with_context(|| format!("running {shown}"))?;
    if !status.success() {
        bail!("{shown} failed with {status}");
    }
    Ok(())
}

/// Directory holding the managed environment and the extracted scripts.
///
/// Under the user's cache directory rather than the state directory holding
/// sessions: everything here is rebuildable from the binary, so losing it costs
/// one bootstrap rather than a transcript.
fn cache_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map_or_else(std::env::temp_dir, PathBuf::from);
    home.join(".cache/speech-to-text-cli")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_scripts_are_the_sidecars() {
        // The scripts are compiled in, so what ships is whatever was in the tree
        // at build time. This is the tripwire for embedding the wrong file.
        assert!(ASR.contains("--model"), "asr sidecar takes a model argument");
        assert!(CLEANUP.contains("--model"), "cleanup sidecar takes a model argument");
    }

    #[test]
    fn the_pinned_set_names_both_runtimes() {
        // A bootstrap omitting either one produces an environment that starts and
        // then fails on the first forward pass.
        assert!(REQUIREMENTS.contains("mlx-audio=="));
        assert!(REQUIREMENTS.contains("mlx-lm=="));
    }

    #[test]
    fn a_named_path_that_is_missing_is_reported_against_its_flag() {
        let err = given(Path::new("/nonexistent/python"), "--python").unwrap_err();
        assert!(err.to_string().contains("--python"), "{err}");
    }

    #[test]
    fn materializing_is_idempotent() {
        let first = materialize("asr_sidecar.py", ASR).expect("write");
        let again = materialize("asr_sidecar.py", ASR).expect("rewrite");
        assert_eq!(first, again);
        assert_eq!(std::fs::read_to_string(&first).expect("read"), ASR);
    }

    #[test]
    fn a_stale_stamp_forces_a_rebuild() {
        let dir = std::env::temp_dir().join(format!("stt-stamp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");

        assert!(!stamp_matches(&dir), "no stamp is not a match");
        std::fs::write(dir.join(STAMP), "numpy==1.0.0\n").expect("write");
        assert!(!stamp_matches(&dir), "a different pin is not a match");
        std::fs::write(dir.join(STAMP), REQUIREMENTS).expect("write");
        assert!(stamp_matches(&dir), "the carried pin matches");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
