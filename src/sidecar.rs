//! Handle to the Python/MLX inference process.
//!
//! Rust owns the pipeline; this process owns only the forward pass. They speak
//! newline-delimited JSON over stdio — no socket, no port, and the sidecar dies
//! with its parent.

use anyhow::{Context, Result, anyhow, bail};
use base64::{Engine, engine::general_purpose::STANDARD as B64};
use crossbeam_channel::{Receiver, unbounded};
use serde::Deserialize;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Duration;

#[derive(Deserialize)]
struct Envelope {
    /// Echoed back from the request. `None` on a message the sidecar could not
    /// attribute to one — a startup banner, or a request it failed to parse.
    #[serde(default)]
    id: Option<i64>,
    #[serde(default)]
    event: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    ms: Option<u64>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    load_ms: Option<u64>,
}

pub struct Hypothesis {
    pub text: String,
    pub infer_ms: u64,
}

/// What came back for one window.
///
/// The distinction is between "this window failed" and "the sidecar is gone".
/// Only the second is fatal: the model can fail on a single buffer and be
/// perfectly healthy for the next one, and ending a dictation session over that
/// would throw away the transcript the user is in the middle of speaking.
pub enum Reply {
    Hypothesis(Hypothesis),
    /// The sidecar rejected this window and is still running.
    Failed(String),
}

pub struct Sidecar {
    child: Child,
    /// `Option` so `Drop` can close it and let the sidecar exit on EOF rather
    /// than being killed mid-flight.
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    notices: Receiver<String>,
    next_id: i64,
}

impl Sidecar {
    /// Spawns the sidecar and blocks until the model is resident and warm.
    pub fn spawn(python: &Path, script: &Path, model: &str, language: Option<&str>) -> Result<Self> {
        let mut cmd = Command::new(python);
        cmd.arg(script).arg("--model").arg(model);
        if let Some(lang) = language {
            cmd.arg("--language").arg(lang);
        }

        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Piped, not inherited: the sidecar must not write to the terminal
            // directly, or it corrupts the renderer's row tracking.
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("spawning sidecar: {} {}", python.display(), script.display()))?;

        let stdin = child.stdin.take().expect("piped");
        let stdout = BufReader::new(child.stdout.take().expect("piped"));

        let (notice_tx, notice_rx) = unbounded::<String>();
        let stderr = child.stderr.take().expect("piped");
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if notice_tx.send(format!("sidecar: {line}")).is_err() {
                    return;
                }
            }
        });

        let mut sc = Self {
            child,
            stdin: Some(stdin),
            stdout,
            notices: notice_rx,
            next_id: 0,
        };

        // No live region exists yet, so startup diagnostics can print directly.
        match sc.read_envelope() {
            Ok(ready) if ready.event.as_deref() == Some("ready") => {
                eprintln!("model loaded in {} ms", ready.load_ms.unwrap_or(0));
                Ok(sc)
            }
            other => {
                // The sidecar's own stderr explains the failure far better than
                // "closed stdout" does, so surface it.
                let detail: Vec<String> = sc.notices.try_iter().collect();
                let reason = match other {
                    Ok(env) => format!("unexpected message: {:?}", env.error),
                    Err(e) => e.to_string(),
                };
                if detail.is_empty() {
                    bail!("sidecar failed to start: {reason}");
                }
                bail!("sidecar failed to start: {reason}\n{}", detail.join("\n"));
            }
        }
    }

    /// Diagnostics from the sidecar. Drain these through the renderer — never
    /// let them reach the terminal on their own.
    pub fn notices(&self) -> &Receiver<String> {
        &self.notices
    }

    /// Transcribe a window. Blocks until the hypothesis comes back — measured
    /// at ~130 ms for a 6 s buffer, well inside the tick interval.
    ///
    /// `Err` means the transport is broken and the session cannot continue.
    /// A window the sidecar merely refused comes back as [`Reply::Failed`].
    ///
    /// Audio only. There is deliberately no text field: feeding the transcript
    /// back as a prompt made the model replay it verbatim whenever the window
    /// held no speech. See the note in `asr_sidecar.py`.
    pub fn transcribe(&mut self, pcm: &[f32]) -> Result<Reply> {
        let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();

        self.next_id += 1;
        let req = serde_json::json!({
            "id": self.next_id,
            "pcm": B64.encode(&bytes),
        });

        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| anyhow!("sidecar stdin already closed"))?;
        writeln!(stdin, "{req}").context("writing to sidecar")?;
        stdin.flush()?;

        let resp = self.read_reply()?;
        if let Some(err) = resp.error {
            return Ok(Reply::Failed(err));
        }

        Ok(Reply::Hypothesis(Hypothesis {
            text: resp.text.unwrap_or_default(),
            infer_ms: resp.ms.unwrap_or(0),
        }))
    }

    /// Read until the reply to the request just sent.
    ///
    /// The `id` check is what keeps the stream in step. stdout is supposed to
    /// carry protocol only, but one stray line from a library inside the
    /// sidecar would otherwise shift every later exchange by one: each window
    /// would silently receive the *previous* window's hypothesis, forever, with
    /// no error anywhere. Skipping unmatched lines resynchronises instead.
    fn read_reply(&mut self) -> Result<Envelope> {
        loop {
            let env = self.read_envelope()?;
            match (env.event.as_deref(), env.id) {
                // A banner, never a reply.
                (Some(_), _) => continue,
                (None, Some(id)) if id == self.next_id => return Ok(env),
                // Left over from an earlier desynchronised exchange.
                (None, Some(id)) if id < self.next_id => continue,
                (None, Some(id)) => {
                    bail!("sidecar replied to request {id}, which was never sent")
                }
                // The sidecar could not attribute it — a request it failed to
                // parse. It consumed our line, so this *is* our reply.
                (None, None) => return Ok(env),
            }
        }
    }

    fn read_envelope(&mut self) -> Result<Envelope> {
        let mut line = String::new();
        let n = self.stdout.read_line(&mut line)?;
        if n == 0 {
            return Err(anyhow!("sidecar closed stdout (process died?)"));
        }
        serde_json::from_str(line.trim())
            .with_context(|| format!("parsing sidecar output: {}", line.trim()))
    }
}

impl Drop for Sidecar {
    fn drop(&mut self) {
        // Closing stdin gives the sidecar EOF, so it unwinds its own read loop
        // and releases MLX/multiprocessing resources cleanly. Kill only if it
        // fails to take the hint.
        self.stdin.take();

        for _ in 0..50 {
            match self.child.try_wait() {
                Ok(Some(_)) => return,
                Ok(None) => std::thread::sleep(Duration::from_millis(20)),
                Err(_) => break,
            }
        }

        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
