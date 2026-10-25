//! Renders overlay outputs for a [`Layout`]:
//! - the source video with the graphs composited on top (HDR stays HDR), and/or
//! - the graphs alone on a transparent canvas as ProRes 4444, for video editors.
//!
//! Each frame's overlay is drawn on the CPU by [`Canvas`] and piped to FFmpeg.
//! When the layout needs nits/scopes, the video is decoded a second time on a
//! worker thread and measured on the GPU in step with rendering.

use crate::canvas::{Canvas, FrameData};
use crate::layout::Layout;
use crate::nits::{NitsFrame, NitsGpu, Transfer};
use crate::progress::Progress;
use crate::timing::Timeline;
use crate::video::{self, VideoInfo};
use anyhow::{Context, Result, anyhow, bail};
use std::collections::VecDeque;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Stdio};
use std::sync::mpsc;

/// Nits that panel white maps to in HDR output (BT.2408 graphics white).
const HDR_GRAPHICS_WHITE: f32 = 203.0;

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
    let fallback = if hdr { "libx265" } else { "libx264" };
    ["hevc_nvenc", "hevc_amf", "hevc_qsv"].into_iter().find(|e| works(e)).unwrap_or(fallback).to_string()
}

fn encoder_args(encoder: &str) -> Vec<&'static str> {
    match encoder {
        "hevc_nvenc" | "h264_nvenc" => vec!["-preset", "p5", "-rc", "vbr", "-cq", "20"],
        "hevc_amf" | "h264_amf" => vec!["-quality", "quality", "-rc", "cqp", "-qp_i", "20", "-qp_p", "20"],
        "hevc_qsv" | "h264_qsv" => vec!["-global_quality", "22"],
        "libx265" => vec!["-preset", "fast", "-crf", "20"],
        _ => vec!["-preset", "veryfast", "-crf", "18"],
    }
}

/// Maps 8-bit SDR overlay values to code values of the HDR output transfer, so
/// overlay white lands at graphics white instead of the format's peak.
fn hdr_panel_lut(transfer: Transfer) -> [u8; 256] {
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

/// How overlay pixels are converted before being piped to FFmpeg.
enum PixelTransform {
    /// Premultiplied RGBA as drawn (FFmpeg overlay is told alpha=premultiplied).
    None,
    /// Premultiplied RGBA with colour channels re-encoded for an HDR output.
    Hdr(Box<[u8; 256]>),
    /// Straight (un-premultiplied) alpha, which editors expect from ProRes 4444.
    Straight,
}

struct Sink {
    /// What this FFmpeg writes, for error reports.
    label: String,
    canvas: Canvas,
    child: Child,
    /// FFmpeg's last messages, for the error report.
    stderr: Option<std::thread::JoinHandle<String>>,
    stdin: Option<ChildStdin>,
    transform: PixelTransform,
    buf: Vec<u8>,
}

impl Sink {
    fn write_frame(&mut self) -> std::io::Result<()> {
        let data = self.canvas.pixels();
        let out: &[u8] = match &self.transform {
            PixelTransform::None => data,
            PixelTransform::Hdr(lut) => {
                self.buf.clear();
                self.buf.extend_from_slice(data);
                for px in self.buf.as_chunks_mut::<4>().0 {
                    px[0] = lut[px[0] as usize];
                    px[1] = lut[px[1] as usize];
                    px[2] = lut[px[2] as usize];
                }
                &self.buf
            }
            PixelTransform::Straight => {
                self.buf.clear();
                self.buf.extend_from_slice(data);
                for px in self.buf.as_chunks_mut::<4>().0 {
                    let a = px[3] as u32;
                    if a > 0 && a < 255 {
                        for c in &mut px[..3] {
                            *c = ((*c as u32 * 255 + a / 2) / a).min(255) as u8;
                        }
                    }
                }
                &self.buf
            }
        };
        self.stdin.as_mut().expect("stdin open while rendering").write_all(out)
    }
}

fn overlay_sink(job: &RenderJob, path: &Path, encoder: Option<&str>) -> Result<Sink> {
    let info = job.info;
    let (bx, by, bw, bh) =
        job.layout.bounds_px(info.width as f32, info.height as f32).context("no graphs are enabled")?;
    let canvas = Canvas::new(info.width, info.height, (bx, by, bw, bh))?;
    let hdr = info.color.is_hdr();
    let encoder = encoder.map(str::to_string).unwrap_or_else(|| pick_encoder(hdr));

    let mut cmd = video::command("ffmpeg");
    cmd.args(["-v", "error", "-stats", "-y"]);
    if let Some(hw) = job.hwaccel {
        cmd.args(["-hwaccel", hw]);
    }
    cmd.arg("-i").arg(job.input);
    cmd.args(["-f", "rawvideo", "-pix_fmt", "rgba", "-s"]).arg(format!("{bw}x{bh}")).args([
        "-framerate",
        &info.fps.to_string(),
        "-i",
        "-",
    ]);
    // Both streams start at t=0 so overlay frame N lands on video frame N. With a
    // variable frame rate the overlay runs at the nominal rate and each video frame
    // takes the overlay frame nearest its timestamp, hence the half-frame shift.
    let (ov_fmt, out_fmt) = if hdr { ("yuv420p10", "yuv420p10le") } else { ("auto", "yuv420p") };
    let shift = if job.timeline.is_variable() { format!("-{}/TB", 0.5 / info.fps) } else { String::new() };
    cmd.arg("-filter_complex").arg(format!(
        "[0:v]setpts=PTS-STARTPTS[v0];[1:v]setpts=PTS-STARTPTS{shift}[ov];\
         [v0][ov]overlay={bx}:{by}:alpha=premultiplied:eof_action=pass:format={ov_fmt},format={out_fmt}[v]"
    ));
    cmd.args(["-map", "[v]", "-map", "0:a?", "-c:a", "aac", "-b:a", "256k"]);
    cmd.args(["-frames:v", &job.frames().to_string(), "-c:v", &encoder]);
    cmd.args(encoder_args(&encoder));
    if hdr {
        let trc = if info.color.transfer == Transfer::Hlg { "arib-std-b67" } else { "smpte2084" };
        cmd.args(["-color_primaries", "bt2020", "-color_trc", trc, "-colorspace", "bt2020nc"]);
        if encoder.starts_with("hevc_") || encoder == "libx265" {
            cmd.args(["-profile:v", "main10"]);
        }
    }
    cmd.arg(path);
    let mut child = cmd.stdin(Stdio::piped()).stderr(Stdio::piped()).spawn().context("failed to run ffmpeg")?;
    let stdin = child.stdin.take();
    let stderr = video::capture_stderr(&mut child);
    let transform =
        if hdr { PixelTransform::Hdr(Box::new(hdr_panel_lut(info.color.transfer))) } else { PixelTransform::None };
    let label = format!("video with graphs ({encoder}) to {}", path.display());
    Ok(Sink { label, canvas, child, stderr, stdin, transform, buf: Vec::new() })
}

fn graphs_sink(job: &RenderJob, path: &Path, width: usize, height: usize) -> Result<Sink> {
    let canvas = Canvas::full(width, height)?;
    let mut cmd = video::command("ffmpeg");
    cmd.args(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s"])
        .arg(format!("{width}x{height}"))
        .args(["-framerate", &job.info.fps.to_string(), "-i", "-"]);
    cmd.args(["-frames:v", &job.timeline.slots(job.frames()).last().map_or(0, |s| s + 1).to_string()]);
    cmd.args(["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", "-vendor", "apl0"]);
    cmd.args(["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]);
    cmd.arg(path);
    let mut child = cmd.stdin(Stdio::piped()).stderr(Stdio::piped()).spawn().context("failed to run ffmpeg")?;
    let stdin = child.stdin.take();
    let stderr = video::capture_stderr(&mut child);
    let label = format!("graphs only (prores_ks) to {}", path.display());
    Ok(Sink { label, canvas, child, stderr, stdin, transform: PixelTransform::Straight, buf: Vec::new() })
}

/// Decodes the video a second time and measures nits frame by frame.
struct NitsStream {
    gpu: NitsGpu,
    rx: mpsc::Receiver<Vec<u8>>,
    child: Child,
    ready: VecDeque<NitsFrame>,
    done: bool,
}

impl NitsStream {
    fn next(&mut self) -> Result<Option<NitsFrame>> {
        while self.ready.is_empty() && !self.done {
            match self.rx.recv() {
                Ok(buf) => self.ready.extend(self.gpu.push(&buf)?),
                Err(_) => {
                    self.ready.extend(self.gpu.finish()?);
                    self.done = true;
                }
            }
        }
        Ok(self.ready.pop_front())
    }
}

/// Starts nits measurement on a worker thread; frames arrive in order on the receiver.
pub fn spawn_nits_worker(
    input: &Path,
    info: &VideoInfo,
    hwaccel: Option<&str>,
    sdr_white: f32,
    scope_flags: u32,
) -> Result<(mpsc::Receiver<NitsFrame>, std::thread::JoinHandle<Result<()>>)> {
    let (nw, nh) = (info.width / 4 * 4, info.height / 2 * 2);
    let gpu = NitsGpu::new(nw, nh, info.color, sdr_white, scope_flags)?;
    let (rx, child) = video::spawn_decoder(input, hwaccel, (nw, nh), "yuv420p10le", gpu.frame_bytes())?;
    let mut stream = NitsStream { gpu, rx, child, ready: VecDeque::new(), done: false };
    let (tx, frames) = mpsc::sync_channel::<NitsFrame>(16);
    let handle = std::thread::spawn(move || -> Result<()> {
        let result = (|| {
            while let Some(f) = stream.next()? {
                if tx.send(f).is_err() {
                    break; // consumer finished or failed
                }
            }
            Ok(())
        })();
        let _ = stream.child.kill();
        let _ = stream.child.wait();
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

    let mut sinks = Vec::new();
    for out in job.outputs {
        sinks.push(match out {
            Output::Overlay { path, encoder } => overlay_sink(job, path, encoder.as_deref())?,
            Output::GraphsOnly { path, width, height } => graphs_sink(job, path, *width, *height)?,
        });
    }
    // Nits run on their own thread (decode + GPU) so they overlap with drawing and encoding.
    let worker = if wants_nits {
        Some(spawn_nits_worker(job.input, info, job.hwaccel, job.sdr_white, job.layout.scope_flags())?)
    } else {
        None
    };

    // Outputs run at the nominal rate; with a variable frame rate a source frame can
    // cover several output frames (or none), so frames are drawn per output slot.
    let slots = job.timeline.slots(job.frames());
    let total_slots = slots.last().map_or(0, |s| s + 1);
    progress.stage("Rendering", total_slots);
    let mut measured: Vec<NitsFrame> = Vec::new();
    let (mut peak_hist, mut avg_hist) = (Vec::new(), Vec::new());
    let mut max_cll = 0f32;
    let mut failed = false;
    let mut current: Option<NitsFrame> = None;
    let mut next = 0usize;
    for slot in 0..total_slots {
        if progress.is_cancelled() {
            break;
        }
        while next < slots.len() && slots[next] <= slot {
            let nf = worker.as_ref().and_then(|(frames, _)| frames.recv().ok());
            if let Some(f) = &nf {
                max_cll = max_cll.max(f.peak);
            }
            peak_hist.push(nf.as_ref().map_or(f64::NAN, |f| f.peak_lum as f64));
            avg_hist.push(nf.as_ref().map_or(f64::NAN, |f| f.avg_lum as f64));
            if let Some(mut f) = std::mem::replace(&mut current, nf) {
                f.scopes = None;
                measured.push(f);
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
            if sink.write_frame().is_err() {
                failed = true; // FFmpeg exited; its own error is printed on stderr
            }
        }
        if failed {
            break;
        }
        progress.advance();
    }
    if let Some(mut f) = current.take() {
        f.scopes = None;
        measured.push(f);
    }

    let mut ffmpeg_errors = Vec::new();
    for sink in &mut sinks {
        drop(sink.stdin.take());
        if progress.is_cancelled() {
            let _ = sink.child.kill();
        }
        let status = sink.child.wait()?;
        let messages = sink.stderr.take().and_then(|h| h.join().ok()).unwrap_or_default();
        if !status.success() || failed {
            let mut report = format!("{}: {}", sink.label, describe_exit(status));
            if !messages.is_empty() {
                report += &format!("\n{messages}");
            }
            ffmpeg_errors.push(report);
        }
    }
    if let Some((frames, handle)) = worker {
        drop(frames); // unblocks the worker if it is still sending
        handle.join().map_err(|_| anyhow!("nits worker panicked"))??;
    }
    if progress.is_cancelled() {
        bail!("cancelled");
    }
    if !ffmpeg_errors.is_empty() {
        bail!("ffmpeg failed while rendering:\n{}", ffmpeg_errors.join("\n"));
    }
    Ok(wants_nits.then_some(measured))
}
