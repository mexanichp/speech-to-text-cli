//! Handle to the Python/MLX inference subprocess.
//!
//! Rust owns the pipeline; the subprocess owns only the forward pass. They
//! communicate in newline-delimited JSON over stdio, so the sidecar needs no
//! socket and terminates with its parent.
//!
//! # Protocol
//!
//! Each request carries an id, base64 f32 PCM at 16 kHz, and a flag selecting
//! whether the fixed vocabulary hint applies. Each reply echoes the request id.
//!
//! # Invariants
//!
//! The request has no free-text field. The host selects whether the prompt
//! given at [`Sidecar::spawn`] applies, and can never supply prompt text, so no
//! code path allows transcript content to reach the model.

use anyhow::{Context, Result, anyhow, bail};
use base64::{Engine, engine::general_purpose::STANDARD as B64};
use crossbeam_channel::{Receiver, unbounded};
use serde::Deserialize;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Duration;

/// One decoded line of the sidecar's stdout stream.
#[derive(Deserialize)]
struct Envelope {
    /// Request id echoed back, or `None` for an unattributable message such as
    /// the startup banner or a request that failed to parse.
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

/// A successful transcription of one audio window.
pub struct Hypothesis {
    /// Transcribed text, already trimmed by the sidecar.
    pub text: String,
    /// Wall-clock cost of the forward pass, in milliseconds.
    pub infer_ms: u64,
}

/// Outcome of transcribing one window.
///
/// Distinct from the `Err` case of the transcribe methods: this enum reports
/// that the sidecar is alive and answered, whereas `Err` reports that the
/// transport is gone and the session cannot continue. A model that refuses one
/// buffer is usually healthy for the next, so refusal must not end the session.
pub enum Reply {
    /// The window was transcribed.
    Hypothesis(Hypothesis),
    /// The sidecar rejected this window and is still running.
    Failed(String),
}

/// Owning handle to a running sidecar process.
pub struct Sidecar {
    child: Child,
    /// Optional so [`Drop`] can close it, letting the sidecar exit on EOF
    /// rather than being killed mid-request.
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    notices: Receiver<String>,
    next_id: i64,
}

impl Sidecar {
    /// Spawns the sidecar and blocks until the model is resident and warm.
    ///
    /// # Parameters
    ///
    /// - `prompt`: fixed vocabulary hint, passed once as a command-line
    ///   argument. Individual requests may select it but may not replace it.
    ///   An empty string omits it entirely.
    ///
    /// # Errors
    ///
    /// Fails if the process cannot be spawned, or if its first message is not
    /// the readiness banner. The sidecar's captured stderr is included in the
    /// error, since it usually explains the cause.
    pub fn spawn(
        python: &Path,
        script: &Path,
        model: &str,
        language: Option<&str>,
        prompt: &str,
    ) -> Result<Self> {
        let mut cmd = Command::new(python);
        cmd.arg(script).arg("--model").arg(model);
        if let Some(lang) = language {
            cmd.arg("--language").arg(lang);
        }
        if !prompt.is_empty() {
            cmd.arg("--system-prompt").arg(prompt);
        }

        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
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

        match sc.read_envelope() {
            Ok(ready) if ready.event.as_deref() == Some("ready") => {
                eprintln!("model loaded in {} ms", ready.load_ms.unwrap_or(0));
                Ok(sc)
            }
            other => {
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

    /// Diagnostics captured from the sidecar's stderr.
    ///
    /// Drain these through the renderer. Writing them to the terminal directly
    /// corrupts the frame the renderer is drawing.
    pub fn notices(&self) -> &Receiver<String> {
        &self.notices
    }

    /// Transcribes a window with no system prompt, blocking until it returns.
    ///
    /// This is the ordinary path, taking nearly every window in a session.
    /// Only audio reaches the model, so nothing the speaker said can bias the
    /// result.
    ///
    /// # Errors
    ///
    /// Returns `Err` only when the transport is broken and the session cannot
    /// continue. A window the sidecar refuses yields [`Reply::Failed`].
    pub fn transcribe(&mut self, pcm: &[f32]) -> Result<Reply> {
        self.send(pcm, false)
    }

    /// Transcribes a window with the vocabulary hint given at [`Sidecar::spawn`].
    ///
    /// The hint improves recognition of command verbs but also biases ambiguous
    /// audio toward the wake word, so it is reserved for re-deciding a finalized
    /// utterance and is never applied to a window being dictated into.
    ///
    /// Only the prompt fixed at spawn can be selected; no prompt text may be
    /// supplied here.
    ///
    /// # Errors
    ///
    /// As [`Sidecar::transcribe`].
    pub fn transcribe_hinted(&mut self, pcm: &[f32]) -> Result<Reply> {
        self.send(pcm, true)
    }

    /// Encodes and sends one window, then waits for its reply.
    fn send(&mut self, pcm: &[f32], hint: bool) -> Result<Reply> {
        let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();

        self.next_id += 1;
        let req = serde_json::json!({
            "id": self.next_id,
            "pcm": B64.encode(&bytes),
            "hint": hint,
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

    /// Reads until the reply matching the request just sent.
    ///
    /// Banners and replies to earlier requests are skipped, which resynchronises
    /// the stream if a stray line ever reaches stdout. Without the id check such
    /// a line would offset every later exchange by one, silently pairing each
    /// window with the previous window's hypothesis.
    ///
    /// # Errors
    ///
    /// Fails if the stream closes, a line does not parse, or the sidecar
    /// answers a request that was never sent.
    fn read_reply(&mut self) -> Result<Envelope> {
        loop {
            let env = self.read_envelope()?;
            match (env.event.as_deref(), env.id) {
                (Some(_), _) => continue,
                (None, Some(id)) if id == self.next_id => return Ok(env),
                (None, Some(id)) if id < self.next_id => continue,
                (None, Some(id)) => {
                    bail!("sidecar replied to request {id}, which was never sent")
                }
                (None, None) => return Ok(env),
            }
        }
    }

    /// Reads and parses the next line of the protocol stream.
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
    /// Closes stdin so the sidecar exits on EOF, then waits briefly before
    /// killing it, so a request in flight is not truncated.
    fn drop(&mut self) {
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
