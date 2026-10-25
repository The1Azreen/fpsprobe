//! Renders overlay outputs for a [`Layout`]:
//! - the source video with the graphs composited on top (HDR stays HDR), and/or
//! - the graphs alone on a transparent canvas as ProRes 4444, for video editors.
//!
//! The video is decoded once, on a worker thread, as 10-bit 4:2:0 (P010, what the
//! hardware decoders and encoders use natively, so FFmpeg converts nothing). Each frame is
//! measured on the GPU when the layout needs nits or scopes, its overlay is drawn
//! on the CPU by [`Canvas`] and blended in on the GPU ([`Compositor`]), and FFmpeg
//! only encodes the result (copying the audio from the source).

use crate::canvas::{Canvas, FrameData};
use crate::composite::Compositor;
use crate::layout::Layout;
use crate::nits::{NitsFrame, NitsGpu, Transfer};
use crate::progress::Progress;
use crate::timing::Timeline;
use crate::video::{self, VideoInfo};
use anyhow::{Context, Result, anyhow, bail};
use std::collections::VecDeque;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Stdio};
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;

/// Nits that panel white maps to in HDR output (BT.2408 graphics white).
const HDR_GRAPHICS_WHITE: f32 = 203.0;
/// Frames queued between drawing, compositing and FFmpeg's stdin.
const WRITE_QUEUE: usize = 4;

#[derive(Clone, Debug)]
pub enum Output {
    /// Source video with the overlay composited on top.
    Overlay { path: PathBuf, encoder: Option<String> },
    /// Graphs only, transparent background, ProRes 4444 at the given size.
    GraphsOnly { path: PathBuf, width: usize, height: usize },
}

pub struct RenderJob<'a> {
    pub input: &'a Path,
    pub info: &'a VideoInfo,
    pub layout: &'a Layout,
    pub timeline: &'a Timeline,
    /// Per-frame FPS and frame time (see [`Timeline::fps_series`]), including frame 0.
    pub fps: &'a [f64],
    pub frametime_ms: &'a [f64],
    pub compare: Option<Compare<'a>>,
    pub outputs: &'a [Output],
    pub hwaccel: Option<&'a str>,
    pub sdr_white: f32,
}

/// A second capture's FPS and frame time, sampled at this video's frames.
#[derive(Clone, Copy)]
pub struct Compare<'a> {
    pub fps: &'a [f64],
    pub frametime_ms: &'a [f64],
}

impl RenderJob<'_> {
    fn frames(&self) -> usize {
        self.fps.len()
    }

    /// Size frames are decoded at: 4:2:0 packing on the GPU needs a multiple of 4x2,
    /// so odd sizes are padded with black (and cropped again when encoding).
    fn decode_size(&self) -> (usize, usize) {
        (self.info.width.div_ceil(4) * 4, self.info.height.div_ceil(2) * 2)
    }
}

/// Picks the first hardware HEVC encoder that actually works on this machine,
/// falling back to x264 (x265 for HDR). FFmpeg builds list encoders whose
/// hardware is absent, so each candidate is tried on a tiny test frame.
fn pick_encoder(hdr: bool) -> String {
    let works = |enc: &str| {
        let fmt = if hdr { "format=p010le" } else { "format=nv12" };
        video::command("ffmpeg")
            .args(["-v", "error", "-f", "lavfi", "-i", "color=s=256x256:d=0.1", "-vf", fmt])
            .args(["-frames:v", "1", "-c:v", enc, "-f", "null", "-"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    };
    ["hevc_nvenc", "hevc_amf", "hevc_qsv"].into_iter().find(|e| works(e)).unwrap_or(software_encoder(hdr)).to_string()
}

/// CPU encoder for when no hardware encoder works (x265 keeps HDR in 10-bit).
pub fn software_encoder(hdr: bool) -> &'static str {
    if hdr { "libx265" } else { "libx264" }
}

/// A hardware encoder crashed while rendering. That is usually the driver, so
/// rendering again with [`software_encoder`] tends to work.
#[derive(Debug)]
pub struct EncoderCrashed {
    pub encoder: String,
    details: String,
}

impl std::fmt::Display for EncoderCrashed {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "ffmpeg failed while rendering:\n{}", self.details)
    }
}

impl std::error::Error for EncoderCrashed {}

fn encoder_args(encoder: &str) -> Vec<&'static str> {
    match encoder {
        "hevc_nvenc" | "h264_nvenc" => vec!["-preset", "p5", "-rc", "vbr", "-cq", "20"],
        "hevc_amf" | "h264_amf" => vec!["-quality", "quality", "-rc", "cqp", "-qp_i", "20", "-qp_p", "20"],
        "hevc_qsv" | "h264_qsv" => vec!["-global_quality", "22"],
        "libx265" => vec!["-preset", "fast", "-crf", "20"],
        _ => vec!["-preset", "veryfast", "-crf", "18"],
    }
}

/// Maps 8-bit SDR overlay values to code values of the output transfer, so overlay
/// white lands at graphics white instead of an HDR format's peak (identity for SDR).
fn panel_lut(transfer: Transfer) -> [u8; 256] {
    let mut lut = [0u8; 256];
    for (v, out) in lut.iter_mut().enumerate() {
        let linear = (v as f32 / 255.0).powf(2.2);
        let e = match transfer {
            Transfer::Pq => {
                let y = (linear * HDR_GRAPHICS_WHITE / 10000.0).max(0.0);
                let (m1, m2, c1, c2, c3) = (0.159_301_76_f32, 78.84375f32, 0.8359375f32, 18.851_563_f32, 18.6875f32);
                let yp = y.powf(m1);
                ((c1 + c2 * yp) / (1.0 + c3 * yp)).powf(m2)
            }
            Transfer::Hlg => {
                // Graphics white sits at 75% HLG signal (scene light 0.265).
                let l = linear * 0.265;
                if l <= 1.0 / 12.0 { (3.0 * l).sqrt() } else { 0.17883277 * (12.0 * l - 0.28466892).ln() + 0.559_910_7 }
            }
            Transfer::Sdr => linear.powf(1.0 / 2.2),
        };
        *out = (e.clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    lut
}

/// Straight (un-premultiplied) RGBA, which editors expect from ProRes 4444.
fn straight_alpha(premultiplied: &[u8]) -> Vec<u8> {
    let mut out = premultiplied.to_vec();
    for px in out.as_chunks_mut::<4>().0 {
        let a = px[3] as u32;
        if a > 0 && a < 255 {
            for c in &mut px[..3] {
                *c = ((*c as u32 * 255 + a / 2) / a).min(255) as u8;
            }
        }
    }
    out
}

/// An overlay drawn on the main thread, with the source frame it goes on (video only).
type Overlay = (Option<Arc<Vec<u8>>>, Vec<u8>);

/// An FFmpeg process and the threads feeding it. Drawing, GPU compositing and
/// writing to FFmpeg each run on their own thread, so they overlap.
struct Sink {
    /// What this FFmpeg writes, for error reports.
    label: String,
    /// The hardware encoder this FFmpeg uses, if any.
    hw_encoder: Option<String>,
    canvas: Canvas,
    child: Child,
    /// FFmpeg's last messages, for the error report.
    stderr: Option<JoinHandle<String>>,
    overlays: Option<mpsc::SyncSender<Overlay>>,
    workers: Vec<JoinHandle<Result<()>>>,
}

impl Sink {
    /// Starts FFmpeg. With a compositor, overlays are blended into their source
    /// frames; without, they are written as straight-alpha RGBA.
    fn spawn(
        mut cmd: std::process::Command,
        label: String,
        canvas: Canvas,
        compositor: Option<Compositor>,
    ) -> Result<Self> {
        let mut child = cmd.stdin(Stdio::piped()).stderr(Stdio::piped()).spawn().context("failed to run ffmpeg")?;
        let mut stdin = child.stdin.take().unwrap();
        let stderr = video::capture_stderr(&mut child);
        let (overlays, overlay_rx) = mpsc::sync_channel::<Overlay>(WRITE_QUEUE);
        let (frames, frame_rx) = mpsc::sync_channel::<Vec<u8>>(WRITE_QUEUE);
        // Stops quietly when FFmpeg stops reading; its exit status tells why.
        let writer = std::thread::spawn(move || {
            for f in frame_rx {
                if stdin.write_all(&f).is_err() {
                    break;
                }
            }
            Ok(())
        });
        let converter = std::thread::spawn(move || -> Result<()> {
            let mut compositor = compositor;
            for (source, overlay) in overlay_rx {
                let out = match (&mut compositor, source) {
                    (Some(c), Some(frame)) => match c.push(&frame, &overlay)? {
                        Some(out) => out,
                        None => continue,
                    },
                    (Some(_), None) => bail!("no decoded frame to draw the graphs on"),
                    (None, _) => straight_alpha(&overlay),
                };
                if frames.send(out).is_err() {
                    return Ok(());
                }
            }
            // Frames still on the GPU.
            if let Some(c) = &mut compositor {
                for out in c.flush()? {
                    if frames.send(out).is_err() {
                        break;
                    }
                }
            }
            Ok(())
        });
        Ok(Self {
            label,
            hw_encoder: None,
            canvas,
            child,
            stderr,
            overlays: Some(overlays),
            workers: vec![converter, writer],
        })
    }

    /// Queues the overlay just drawn; false once the output has stopped taking frames.
    fn send(&mut self, source: Option<&Arc<Vec<u8>>>) -> bool {
        let overlay = self.canvas.pixels().to_vec();
        self.overlays.as_ref().is_some_and(|tx| tx.send((source.cloned(), overlay)).is_ok())
    }

    /// Closes FFmpeg's input and waits for it and the feeding threads.
    fn finish(&mut self, kill: bool) -> Result<(std::process::ExitStatus, Option<anyhow::Error>)> {
        drop(self.overlays.take());
        if kill {
            let _ = self.child.kill();
        }
        let mut error = None;
        for w in self.workers.drain(..) {
            match w.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => error = error.or(Some(e)),
                Err(_) => error = error.or(Some(anyhow!("render thread panicked"))),
            }
        }
        Ok((self.child.wait()?, error))
    }
}

fn video_sink(job: &RenderJob, path: &Path, encoder: Option<&str>, frames_out: usize) -> Result<Sink> {
    let info = job.info;
    let rect = job.layout.bounds_px(info.width as f32, info.height as f32).context("no graphs are enabled")?;
    let canvas = Canvas::new(info.width, info.height, rect)?;
    let hdr = info.color.is_hdr();
    let encoder = encoder.map(str::to_string).unwrap_or_else(|| pick_encoder(hdr));
    let (dw, dh) = job.decode_size();
    let compositor = Compositor::new(dw, dh, rect, info.color, &panel_lut(info.color.transfer))?;

    let trc = match info.color.transfer {
        Transfer::Pq => "smpte2084",
        Transfer::Hlg => "arib-std-b67",
        Transfer::Sdr => "bt709",
    };
    let (primaries, matrix) = if info.color.bt2020 { ("bt2020", "bt2020nc") } else { ("bt709", "bt709") };
    let range = if info.color.full_range { "pc" } else { "tv" };
    let tags = ["-color_primaries", primaries, "-color_trc", trc, "-colorspace", matrix, "-color_range", range];

    let mut cmd = video::command("ffmpeg");
    cmd.args(["-v", "error", "-stats", "-y", "-f", "rawvideo", "-pix_fmt", "p010le", "-s"])
        .arg(format!("{dw}x{dh}"))
        .args(["-framerate", &info.fps.to_string()]);
    // Tagging the raw input too keeps FFmpeg from converting every frame's colours
    // in software (5x slower, and a needless conversion).
    cmd.args(tags).args(["-i", "-"]);
    cmd.arg("-i").arg(job.input);
    cmd.args(["-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-b:a", "256k"]);
    // Encoders need even sizes; padding added for decoding is cropped off again.
    let (ow, oh) = (info.width / 2 * 2, info.height / 2 * 2);
    if (ow, oh) != (dw, dh) {
        cmd.args(["-vf", &format!("crop={ow}:{oh}:0:0")]);
    }
    cmd.args(["-frames:v", &frames_out.to_string(), "-c:v", &encoder]);
    cmd.args(encoder_args(&encoder));
    // Hardware encoders take P010 / NV12 as is; x264 / x265 want planar formats.
    let out_fmt = match (encoder.starts_with("lib"), hdr) {
        (false, true) => "p010le",
        (false, false) => "nv12",
        (true, true) => "yuv420p10le",
        (true, false) => "yuv420p",
    };
    cmd.args(["-pix_fmt", out_fmt]);
    cmd.args(tags);
    if hdr && (encoder.starts_with("hevc_") || encoder == "libx265") {
        cmd.args(["-profile:v", "main10"]);
    }
    cmd.arg(path);
    let label = format!("video with graphs ({encoder}) to {}", path.display());
    let mut sink = Sink::spawn(cmd, label, canvas, Some(compositor))?;
    sink.hw_encoder = (!encoder.starts_with("lib")).then_some(encoder);
    Ok(sink)
}

fn graphs_sink(job: &RenderJob, path: &Path, width: usize, height: usize, frames_out: usize) -> Result<Sink> {
    let canvas = Canvas::full(width, height)?;
    let mut cmd = video::command("ffmpeg");
    cmd.args(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s"])
        .arg(format!("{width}x{height}"))
        .args(["-framerate", &job.info.fps.to_string(), "-i", "-"]);
    cmd.args(["-frames:v", &frames_out.to_string()]);
    cmd.args(["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", "-vendor", "apl0"]);
    cmd.args(["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]);
    cmd.arg(path);
    Sink::spawn(cmd, format!("graphs only (prores_ks) to {}", path.display()), canvas, None)
}

/// One decoded source frame: its measurement and/or its pixels.
pub struct SourceFrame {
    pub nits: Option<NitsFrame>,
    /// P010 at [`RenderJob::decode_size`], when outputs need the picture.
    pub yuv: Option<Vec<u8>>,
}

/// Decodes the video once on a worker thread, measuring nits on the GPU when
/// `scope_flags` is given and keeping the pixels when `keep_pixels` is set.
/// Frames arrive in order on the receiver.
pub fn spawn_source(
    input: &Path,
    info: &VideoInfo,
    (width, height): (usize, usize),
    hwaccel: Option<&str>,
    sdr_white: f32,
    scope_flags: Option<u32>,
    keep_pixels: bool,
) -> Result<(mpsc::Receiver<SourceFrame>, JoinHandle<Result<()>>)> {
    let mut gpu = scope_flags.map(|f| NitsGpu::new(width, height, info.color, sdr_white, f)).transpose()?;
    let (rx, mut child) = video::spawn_decoder(input, hwaccel, (width, height), "p010le", width * height * 3)?;
    let (tx, frames) = mpsc::sync_channel::<SourceFrame>(4);
    let handle = std::thread::spawn(move || -> Result<()> {
        let result = (|| {
            // Nits come back in batches, so pixels wait here for their measurement.
            let mut pending: VecDeque<Option<Vec<u8>>> = VecDeque::new();
            let emit = |nits: Vec<NitsFrame>, pending: &mut VecDeque<Option<Vec<u8>>>| {
                nits.into_iter()
                    .all(|n| tx.send(SourceFrame { nits: Some(n), yuv: pending.pop_front().flatten() }).is_ok())
            };
            for buf in rx {
                let open = match gpu.as_mut() {
                    Some(g) => {
                        let done = g.push(&buf)?;
                        pending.push_back(keep_pixels.then_some(buf));
                        emit(done, &mut pending)
                    }
                    None => tx.send(SourceFrame { nits: None, yuv: Some(buf) }).is_ok(),
                };
                if !open {
                    return Ok(()); // consumer finished or failed
                }
            }
            if let Some(g) = gpu.as_mut() {
                let done = g.finish()?;
                emit(done, &mut pending);
            }
            Ok(())
        })();
        let _ = child.kill();
        let _ = child.wait();
        result
    });
    Ok((frames, handle))
}

/// FFmpeg's exit status in words. Windows reports crashes as NTSTATUS codes
/// (0xC...), which FFmpeg never returns itself.
fn describe_exit(status: std::process::ExitStatus) -> String {
    match status.code() {
        Some(0) => "exited early".into(),
        Some(code) if (code as u32) >= 0xC000_0000 => {
            let what = match code as u32 {
                0xC000_0005 => "access violation",
                0xC000_0409 => "stack buffer overrun",
                0xC000_001D => "illegal instruction",
                0xC000_0017 | 0xC000_012D => "out of memory",
                _ => "crash",
            };
            format!("crashed with 0x{:08X} ({what}), probably in the encoder or GPU driver", code as u32)
        }
        Some(code) => format!("exited with code {code}"),
        None => "was terminated".into(),
    }
}

/// Renders all requested outputs in one pass. Returns per-frame nits (without
/// scope data) when the layout needed them.
pub fn render(job: &RenderJob, progress: &Progress) -> Result<Option<Vec<NitsFrame>>> {
    let info = job.info;
    let wants_nits = job.layout.needs_nits();
    // Outputs run at the nominal rate; with a variable frame rate a source frame can
    // cover several output frames (or none), so frames are drawn per output slot.
    let slots = job.timeline.slots(job.frames());
    let total_slots = slots.last().map_or(0, |s| s + 1);

    let mut sinks = Vec::new();
    for out in job.outputs {
        sinks.push(match out {
            Output::Overlay { path, encoder } => video_sink(job, path, encoder.as_deref(), total_slots)?,
            Output::GraphsOnly { path, width, height } => graphs_sink(job, path, *width, *height, total_slots)?,
        });
    }
    let keep_pixels = job.outputs.iter().any(|o| matches!(o, Output::Overlay { .. }));
    // Decoding and nits run on their own thread so they overlap with drawing and encoding.
    let source = if wants_nits || keep_pixels {
        let flags = wants_nits.then(|| job.layout.scope_flags());
        Some(spawn_source(job.input, info, job.decode_size(), job.hwaccel, job.sdr_white, flags, keep_pixels)?)
    } else {
        None
    };

    progress.stage("Rendering", total_slots);
    let mut measured: Vec<NitsFrame> = Vec::new();
    let (mut peak_hist, mut avg_hist) = (Vec::new(), Vec::new());
    let mut max_cll = 0f32;
    let mut failed = false;
    let mut current: Option<NitsFrame> = None;
    let mut pixels: Option<Arc<Vec<u8>>> = None;
    let mut next = 0usize;
    let result: Result<()> = {
        for slot in 0..total_slots {
            if progress.is_cancelled() {
                break;
            }
            while next < slots.len() && slots[next] <= slot {
                let src = source.as_ref().and_then(|(frames, _)| frames.recv().ok());
                let (nf, yuv) = src.map_or((None, None), |s| (s.nits, s.yuv));
                if let Some(f) = &nf {
                    max_cll = max_cll.max(f.peak);
                }
                peak_hist.push(nf.as_ref().map_or(f64::NAN, |f| f.peak_lum as f64));
                avg_hist.push(nf.as_ref().map_or(f64::NAN, |f| f.avg_lum as f64));
                if let Some(mut f) = std::mem::replace(&mut current, nf) {
                    f.scopes = None;
                    measured.push(f);
                }
                if let Some(yuv) = yuv {
                    pixels = Some(Arc::new(yuv));
                }
                next += 1;
            }
            let data = FrameData {
                frame: next - 1,
                capture_fps: info.fps,
                fps: job.fps,
                frametime_ms: job.frametime_ms,
                compare_fps: job.compare.map(|c| c.fps),
                compare_frametime_ms: job.compare.map(|c| c.frametime_ms),
                nits: current.as_ref(),
                peak_history: &peak_hist,
                avg_history: &avg_hist,
                max_cll,
                source_pixels: (info.width * info.height) as f32,
            };
            for sink in &mut sinks {
                sink.canvas.draw(job.layout, &data);
                if !sink.send(pixels.as_ref()) {
                    failed = true; // FFmpeg exited; its error is reported below
                }
            }
            if failed {
                break;
            }
            progress.advance();
        }
        Ok(())
    };
    if let Some(mut f) = current.take() {
        f.scopes = None;
        measured.push(f);
    }

    let mut ffmpeg_errors = Vec::new();
    let mut crashed_encoder = None;
    for sink in &mut sinks {
        let (status, worker_error) = sink.finish(progress.is_cancelled() || result.is_err())?;
        let messages = sink.stderr.take().and_then(|h| h.join().ok()).unwrap_or_default();
        if status.code().is_some_and(|c| c as u32 >= 0xC000_0000) && sink.hw_encoder.is_some() {
            crashed_encoder = sink.hw_encoder.clone();
        }
        if let Some(e) = worker_error.filter(|_| !progress.is_cancelled() && result.is_ok()) {
            ffmpeg_errors.push(format!("{}: {e:#}", sink.label));
        } else if (!status.success() || failed) && !progress.is_cancelled() && result.is_ok() {
            let mut report = format!("{}: {}", sink.label, describe_exit(status));
            if !messages.is_empty() {
                report += &format!("\n{messages}");
            }
            ffmpeg_errors.push(report);
        }
    }
    if let Some((frames, handle)) = source {
        drop(frames); // unblocks the worker if it is still sending
        handle.join().map_err(|_| anyhow!("decoding worker panicked"))??;
    }
    result?;
    if progress.is_cancelled() {
        bail!("cancelled");
    }
    if !ffmpeg_errors.is_empty() {
        let details = ffmpeg_errors.join("\n");
        if let Some(encoder) = crashed_encoder {
            return Err(EncoderCrashed { encoder, details }.into());
        }
        bail!("ffmpeg failed while rendering:\n{details}");
    }
    Ok(wants_nits.then_some(measured))
}
