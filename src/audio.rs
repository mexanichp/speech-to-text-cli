//! Audio capture, downmixed to mono and resampled to 16 kHz f32.
//!
//! Every component downstream of this module expects that format, so it is the
//! one place sample rates and channel counts are handled.
//!
//! Capture runs on a real-time thread that only downmixes and forwards.
//! Resampling allocates and is stateful, so it runs on its own thread.
//!
//! [`simulate`] substitutes a WAV file for the microphone at wall-clock pace,
//! which is how the pipeline is exercised end to end without a speaker.

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
/// This is a latency budget rather than a queue size. The main loop drains
/// nothing while blocked in a forward pass, and several passes run off the
/// regular tick and can land back to back, so the buffer must absorb a burst
/// of them. At 512 chunks it holds several seconds.
///
/// On overflow the capture callback drops buffers, and audio with gaps in it
/// does not degrade gracefully: the model returns fragments rather than
/// sentences, and later passes cannot repair it. Lag is recoverable, so this
/// is sized to trade lag for integrity.
const CAPTURE_BACKLOG: usize = 512;

/// Live audio source plus the diagnostics raised while producing it.
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

/// Feeds a WAV file through the pipeline at wall-clock pace, replacing the
/// microphone.
///
/// Pacing follows an absolute deadline rather than sleeping between sends, so
/// scheduler delay and send cost do not accumulate. A relative-sleep pacer
/// makes the pipeline appear slower than realtime when it is keeping up.
///
/// Trailing silence is appended so the final utterance reaches an endpoint.
///
/// # Errors
///
/// Fails if the file cannot be read, or if it is not 16 kHz.
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
    let step = TARGET_RATE as usize / 50;

    std::thread::spawn(move || {
        let start = std::time::Instant::now();
        let mut sent = 0u32;
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

    let (_tx, notices) = unbounded::<String>();

    Ok(Capture {
        pcm: rx,
        notices,
        _stream: None,
    })
}

/// Opens an input device and starts capture.
///
/// # Parameters
///
/// - `device_name`: substring matched against device names; `None` selects the
///   system default.
///
/// # Errors
///
/// Fails if no host input device matches, or if the stream cannot be built or
/// started.
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

/// Resamples mono chunks from `in_rate` to [`TARGET_RATE`] until the input
/// channel closes.
///
/// Runs off the capture thread because resampling allocates and holds state,
/// neither of which is permitted in a real-time audio callback.
fn resample_loop(raw: Receiver<Vec<f32>>, out: Sender<Vec<f32>>, in_rate: u32) -> Result<()> {
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
