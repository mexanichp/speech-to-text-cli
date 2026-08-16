//! Microphone capture, downmixed to mono and resampled to the 16 kHz f32
//! contract every ASR component in this pipeline expects.
//!
//! The cpal callback is a real-time context, so it only downmixes and hands
//! off. Resampling — which allocates and is stateful — happens on its own
//! thread.

use anyhow::{Context, Result, anyhow, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::time::Duration;

pub const TARGET_RATE: u32 = 16_000;

/// Samples per resampler call at the input rate.
const RESAMPLE_CHUNK: usize = 1024;

/// Capture chunks buffered between the audio thread and the main loop.
///
/// This is a **latency budget, not a queue size**, and it is load-bearing.
/// Once it overflows the audio callback drops buffers, and the model is then
/// handed audio with holes in it — which does not degrade gracefully. Measured,
/// the same passage comes back as `So. a. bench. run. on silence.` instead of a
/// sentence, and no amount of catching up afterwards repairs it.
///
/// The main loop consumes one chunk per iteration, so it drains nothing at all
/// while it is blocked in a forward pass. The tick stretch keeps the steady
/// state under 100% duty, but three places deliberately run a pass *off* the
/// tick and can therefore land back to back:
///
/// * the final pass at an endpoint,
/// * the immediate re-decode when a continuation merges,
/// * the extra pass over the head when the buffer is trimmed.
///
/// At 512 chunks this holds several seconds either way — comfortably more than
/// that burst — so falling behind costs a few seconds of lag that pauses then
/// give back, instead of costing words outright. Latency is recoverable;
/// shredded audio is not.
const CAPTURE_BACKLOG: usize = 512;

pub struct Capture {
    /// 16 kHz mono f32, in arbitrary-sized chunks.
    pub pcm: Receiver<Vec<f32>>,
    /// Diagnostics raised off the audio and resampler threads. Drain these
    /// through the renderer; writing them to the terminal directly corrupts
    /// its row tracking.
    pub notices: Receiver<String>,
    /// Dropping the stream stops capture, so the caller must hold it.
    _stream: Option<cpal::Stream>,
}

/// Feed a WAV file through the pipeline at wall-clock pace, as if it were the
/// microphone. The only way to exercise capture -> VAD -> commit end to end
/// without a human speaking.
pub fn simulate(path: &std::path::Path) -> Result<Capture> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();

    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    };

    let channels = spec.channels as usize;
    let mono: Vec<f32> = if channels == 1 {
        raw
    } else {
        raw.chunks(channels)
            .map(|f| f.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    if spec.sample_rate != TARGET_RATE {
        bail!(
            "simulate expects {} Hz mono; {} is {} Hz",
            TARGET_RATE,
            path.display(),
            spec.sample_rate
        );
    }

    eprintln!(
        "audio: simulating {} ({:.2}s) at realtime pace",
        path.display(),
        mono.len() as f32 / TARGET_RATE as f32
    );

    let (tx, rx) = bounded::<Vec<f32>>(CAPTURE_BACKLOG);
    // 20 ms per tick, matching a typical capture callback cadence.
    let step = TARGET_RATE as usize / 50;

    std::thread::spawn(move || {
        // Paced against an absolute schedule, not by sleeping between sends.
        //
        // `sleep` guarantees *at least* its argument, so a loop that sleeps 20ms
        // per chunk drifts by however long each iteration took plus whatever the
        // scheduler added — and it adds a lot here, because the main thread is
        // saturating the GPU and a Python process is contending for the CPU.
        // Measured on a 41.6s file, the relative form took 58s to send, which
        // made the pipeline look ~40% slower than realtime when it was in fact
        // keeping up: the harness, not the system under test, was late.
        let start = std::time::Instant::now();
        let mut sent = 0u32;
        // Trailing silence so the VAD endpoints the final utterance.
        let silence = vec![0.0f32; step];
        let blocks = mono.chunks(step).map(Some).chain((0..40).map(|_| None));

        for block in blocks {
            let chunk = block.map_or_else(|| silence.clone(), <[f32]>::to_vec);
            if tx.send(chunk).is_err() {
                return;
            }
            sent += 1;
            let due = start + Duration::from_millis(20) * sent;
            if let Some(wait) = due.checked_duration_since(std::time::Instant::now()) {
                std::thread::sleep(wait);
            }
        }
    });

    // Nothing runs off-thread here that can raise diagnostics.
    let (_tx, notices) = unbounded::<String>();

    Ok(Capture {
        pcm: rx,
        notices,
        _stream: None,
    })
}

pub fn start(device_name: Option<&str>) -> Result<Capture> {
    let host = cpal::default_host();

    let device = match device_name {
        Some(want) => host
            .input_devices()?
            .find(|d| {
                d.description()
                    .map(|desc| desc.name().contains(want))
                    .unwrap_or(false)
            })
            .ok_or_else(|| anyhow!("no input device matching {want:?}"))?,
        None => host
            .default_input_device()
            .ok_or_else(|| anyhow!("no default input device"))?,
    };

    let config = device
        .default_input_config()
        .context("querying default input config")?;

    let in_rate = config.sample_rate();
    let channels = config.channels() as usize;

    let label = device
        .description()
        .map(|d| d.name().to_string())
        .unwrap_or_else(|_| "?".into());
    eprintln!("audio: {label} @ {in_rate} Hz, {channels} ch -> {TARGET_RATE} Hz mono");

    let (raw_tx, raw_rx) = bounded::<Vec<f32>>(CAPTURE_BACKLOG);
    let (pcm_tx, pcm_rx) = bounded::<Vec<f32>>(CAPTURE_BACKLOG);
    let (notice_tx, notice_rx) = unbounded::<String>();
    let err_tx = notice_tx.clone();

    let stream = device.build_input_stream(
        config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mono: Vec<f32> = if channels == 1 {
                data.to_vec()
            } else {
                data.chunks(channels)
                    .map(|f| f.iter().sum::<f32>() / channels as f32)
                    .collect()
            };
            // Dropping under backpressure is correct here: better to lose a
            // buffer than to block the audio thread.
            let _ = raw_tx.try_send(mono);
        },
        move |err| {
            let _ = err_tx.send(format!("audio stream error: {err}"));
        },
        None,
    )?;

    stream.play()?;

    std::thread::spawn(move || {
        if let Err(e) = resample_loop(raw_rx, pcm_tx, in_rate) {
            let _ = notice_tx.send(format!("resampler stopped: {e}"));
        }
    });

    Ok(Capture {
        pcm: pcm_rx,
        notices: notice_rx,
        _stream: Some(stream),
    })
}

fn resample_loop(raw: Receiver<Vec<f32>>, out: Sender<Vec<f32>>, in_rate: u32) -> Result<()> {
    // Already at the target rate: pass through untouched.
    if in_rate == TARGET_RATE {
        for chunk in raw {
            if out.send(chunk).is_err() {
                break;
            }
        }
        return Ok(());
    }

    let params = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.95,
        oversampling_factor: 128,
        interpolation: SincInterpolationType::Linear,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        TARGET_RATE as f64 / in_rate as f64,
        1.0,
        params,
        RESAMPLE_CHUNK,
        1,
    )?;

    let mut pending: Vec<f32> = Vec::with_capacity(RESAMPLE_CHUNK * 2);

    for chunk in raw {
        pending.extend_from_slice(&chunk);

        while pending.len() >= RESAMPLE_CHUNK {
            let block: Vec<f32> = pending.drain(..RESAMPLE_CHUNK).collect();
            let resampled = resampler.process(&[block], None)?;
            if out.send(resampled[0].clone()).is_err() {
                return Ok(());
            }
        }
    }

    Ok(())
}
