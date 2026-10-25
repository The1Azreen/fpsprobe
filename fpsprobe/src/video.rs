//! FFmpeg helpers: probing a video and decoding it to raw frames.

use crate::nits::ColorInfo;
use anyhow::{Context, Result, bail};
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread::JoinHandle;

#[derive(Clone, Debug)]
pub struct VideoInfo {
    pub width: usize,
    pub height: usize,
    pub fps: f64,
    /// Container duration in seconds, when known (used for progress estimates).
    pub duration: Option<f64>,
    pub color: ColorInfo,
}

impl VideoInfo {
    /// Approximate frame count for progress bars.
    pub fn estimated_frames(&self) -> Option<usize> {
        self.duration.map(|d| (d * self.fps).round() as usize)
    }
}

pub fn probe(path: &Path) -> Result<VideoInfo> {
    let out = command("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,color_transfer,color_space,color_range:format=duration",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(path)
        .output()
        .context("failed to run ffprobe (is FFmpeg on PATH?)")?;
    if !out.status.success() {
        bail!("ffprobe failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    let text = String::from_utf8(out.stdout)?;
    let field = |key: &str| {
        text.lines().find_map(|l| l.trim().strip_prefix(key).and_then(|r| r.strip_prefix('='))).unwrap_or("unknown")
    };
    let rate = field("r_frame_rate");
    let (num, den) = rate.split_once('/').unwrap_or((rate, "1"));
    Ok(VideoInfo {
        width: field("width").parse().context("ffprobe: missing width")?,
        height: field("height").parse().context("ffprobe: missing height")?,
        fps: num.parse::<f64>()? / den.parse::<f64>()?,
        duration: field("duration").parse().ok(),
        color: ColorInfo::from_probe(field("color_transfer"), field("color_space"), field("color_range")),
    })
}

/// Reads the presentation time of every video frame from packet timestamps
/// (demuxing only, no decoding) in the background. Dropping it stops ffprobe.
pub struct FrameTimes {
    child: Child,
    reader: Option<JoinHandle<std::io::Result<String>>>,
}

impl FrameTimes {
    pub fn spawn(path: &Path) -> Result<Self> {
        let mut child = command("ffprobe")
            .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time", "-of", "csv=p=0"])
            .arg(path)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("failed to run ffprobe (is FFmpeg installed and on PATH?)")?;
        let mut stdout = child.stdout.take().unwrap();
        let reader = std::thread::spawn(move || {
            let mut text = String::new();
            stdout.read_to_string(&mut text).map(|_| text)
        });
        Ok(Self { child, reader: Some(reader) })
    }

    /// Seconds from the first frame, in presentation order. `None` when any
    /// timestamp is missing.
    pub fn finish(mut self) -> Result<Option<Vec<f64>>> {
        let text = self.reader.take().unwrap().join().map_err(|_| anyhow::anyhow!("ffprobe reader panicked"))??;
        if !self.child.wait()?.success() {
            bail!("ffprobe failed to read frame timestamps");
        }
        parse_frame_times(&text)
    }
}

impl Drop for FrameTimes {
    fn drop(&mut self) {
        // Still running when analysis was cancelled; ffprobe would read the whole file.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub fn frame_times(path: &Path) -> Result<Option<Vec<f64>>> {
    FrameTimes::spawn(path)?.finish()
}

fn parse_frame_times(text: &str) -> Result<Option<Vec<f64>>> {
    let parsed: Option<Vec<f64>> =
        text.lines().map(str::trim).filter(|l| !l.is_empty()).map(|l| l.trim_end_matches(',').parse().ok()).collect();
    let Some(mut times) = parsed.filter(|t| !t.is_empty()) else { return Ok(None) };
    // Packets arrive in decode order; B-frames make that differ from presentation order.
    times.sort_by(f64::total_cmp);
    let first = times[0];
    times.iter_mut().for_each(|t| *t -= first);
    Ok(Some(times))
}

/// Hardware decoders worth trying on this platform, best first.
const HWACCELS: &[&str] = if cfg!(windows) {
    &["d3d11va", "cuda", "qsv"]
} else if cfg!(target_os = "macos") {
    &["videotoolbox"]
} else {
    &["cuda", "vaapi", "qsv", "vulkan"]
};

/// Resolves the decoder choice: "auto" becomes the first hardware decoder that can
/// decode this video (or software decoding), anything else is passed through.
pub fn resolve_hwaccel(choice: Option<&str>, input: &Path) -> Option<String> {
    match choice {
        Some("auto") => HWACCELS.iter().find(|hw| hw_decodes(input, hw)).map(|hw| hw.to_string()),
        other => other.map(str::to_string),
    }
}

/// FFmpeg quietly falls back to software decoding when a hardware decoder fails to
/// initialise, so success means a clean exit with nothing printed at error level.
fn hw_decodes(input: &Path, hw: &str) -> bool {
    command("ffmpeg")
        .args(["-v", "error", "-nostdin", "-hwaccel", hw, "-i"])
        .arg(input)
        .args(["-an", "-sn", "-frames:v", "3", "-f", "null", "-"])
        .stdout(Stdio::null())
        .output()
        .is_ok_and(|o| o.status.success() && o.stderr.is_empty())
}

/// Forwards a child's stderr to ours (FFmpeg's progress line included) and keeps its
/// last few messages for error reports: the GUI has no console to show them on.
pub fn capture_stderr(child: &mut Child) -> Option<JoinHandle<String>> {
    let mut stderr = child.stderr.take()?;
    Some(std::thread::spawn(move || {
        const KEEP: usize = 8;
        let (mut tail, mut line) = (VecDeque::new(), Vec::new());
        let mut buf = [0u8; 4096];
        let mut out = std::io::stderr();
        while let Ok(n) = stderr.read(&mut buf) {
            if n == 0 {
                break;
            }
            let _ = out.write_all(&buf[..n]);
            for &b in &buf[..n] {
                if b != b'\n' && b != b'\r' {
                    line.push(b);
                    continue;
                }
                let text = String::from_utf8_lossy(&line).trim().to_string();
                line.clear();
                // Progress lines ("frame= ... speed=") aren't errors.
                if !text.is_empty() && !text.starts_with("frame=") {
                    tail.push_back(text);
                    if tail.len() > KEEP {
                        tail.pop_front();
                    }
                }
            }
        }
        tail.into_iter().collect::<Vec<_>>().join("\n")
    }))
}

/// Builds a command for an FFmpeg tool without flashing a console window from the GUI.
pub fn command(program: &str) -> Command {
    #[allow(unused_mut)]
    let mut cmd = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

/// Spawns FFmpeg decoding to raw `pix_fmt` frames of `width`x`height` (cropped, or
/// padded with black on the right and bottom), and streams them over a channel.
pub fn spawn_decoder(
    input: &Path,
    hwaccel: Option<&str>,
    (width, height): (usize, usize),
    pix_fmt: &str,
    frame_bytes: usize,
) -> Result<(mpsc::Receiver<Vec<u8>>, Child)> {
    let mut cmd = command("ffmpeg");
    cmd.args(["-v", "error", "-nostdin"]);
    if let Some(hw) = hwaccel {
        cmd.args(["-hwaccel", hw]);
    }
    cmd.arg("-i").arg(input);
    // Sizing keeps frames packable into u32 words on the GPU.
    // passthrough keeps FFmpeg from inserting or dropping frames, which would fake the result.
    cmd.args(["-an", "-sn", "-fps_mode", "passthrough", "-vf"])
        .arg(format!("pad=max(iw\\,{width}):max(ih\\,{height}):0:0,crop={width}:{height}:0:0,format={pix_fmt}"))
        .args(["-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]);
    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("failed to run ffmpeg (is FFmpeg installed and on PATH?)")?;
    let mut stdout = child.stdout.take().unwrap();
    let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(8);
    std::thread::spawn(move || {
        loop {
            let mut buf = vec![0u8; frame_bytes];
            if stdout.read_exact(&mut buf).is_err() || tx.send(buf).is_err() {
                break;
            }
        }
    });
    Ok((rx, child))
}
