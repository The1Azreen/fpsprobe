//! FFmpeg helpers: probing a video and decoding it to raw frames.

use crate::nits::ColorInfo;
use anyhow::{Context, Result, bail};
use std::io::Read;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;

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

/// Spawns FFmpeg decoding to raw `pix_fmt` frames cropped to `width`x`height`,
/// and streams them over a channel.
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
    // Cropping keeps frames packable into u32 words on the GPU.
    // passthrough keeps FFmpeg from inserting or dropping frames, which would fake the result.
    cmd.args(["-an", "-sn", "-fps_mode", "passthrough", "-vf"])
        .arg(format!("crop={width}:{height}:0:0,format={pix_fmt}"))
        .args(["-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]);
    let mut child = cmd.stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn().context("failed to run ffmpeg")?;
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
