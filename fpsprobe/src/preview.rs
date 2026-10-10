//! Low-resolution video frames for the GUI preview at a chosen time.
//! HDR is tone-mapped to SDR for display only.

use crate::video::{self, VideoInfo};
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::mpsc;

/// FFmpeg filter chain producing `w`x`h` RGBA, tone-mapping HDR sources to SDR.
fn preview_filter(info: &VideoInfo, w: usize, h: usize) -> String {
    let scale = format!("scale={w}:{h}:flags=bilinear");
    if info.color.is_hdr() {
        format!(
            "{scale},zscale=t=linear:npl=100,format=gbrpf32le,tonemap=hable:desat=0,\
             zscale=t=bt709:m=bt709:p=bt709:r=pc,format=rgba"
        )
    } else {
        format!("{scale},format=rgba")
    }
}

/// Decodes one frame at `time` seconds as `w`x`h` RGBA.
pub fn grab_frame(input: &Path, info: &VideoInfo, time: f64, w: usize, h: usize) -> Result<Vec<u8>> {
    let out = video::command("ffmpeg")
        .args(["-v", "error", "-nostdin", "-ss", &format!("{:.3}", time.max(0.0))])
        .arg("-i")
        .arg(input)
        .args(["-an", "-sn", "-frames:v", "1", "-vf", &preview_filter(info, w, h), "-f", "rawvideo", "-"])
        .stderr(Stdio::null())
        .output()
        .context("failed to run ffmpeg")?;
    if out.stdout.len() < w * h * 4 {
        bail!("no frame at {time:.2}s");
    }
    Ok(out.stdout[..w * h * 4].to_vec())
}

/// A decoded preview frame and the time it was requested for.
pub struct Grabbed {
    pub time: f64,
    pub size: (usize, usize),
    pub rgba: Result<Vec<u8>>,
}

/// Background frame grabber: only the most recent request is decoded, so
/// dragging a slider never queues up stale work.
pub struct FrameGrabber {
    requests: mpsc::Sender<(PathBuf, VideoInfo, f64, usize, usize)>,
    results: mpsc::Receiver<Grabbed>,
}

impl Default for FrameGrabber {
    fn default() -> Self {
        let (req_tx, req_rx) = mpsc::channel::<(PathBuf, VideoInfo, f64, usize, usize)>();
        let (res_tx, res_rx) = mpsc::channel();
        std::thread::spawn(move || {
            while let Ok(mut req) = req_rx.recv() {
                while let Ok(newer) = req_rx.try_recv() {
                    req = newer;
                }
                let (input, info, time, w, h) = req;
                let rgba = grab_frame(&input, &info, time, w, h);
                if res_tx.send(Grabbed { time, size: (w, h), rgba }).is_err() {
                    break;
                }
            }
        });
        Self { requests: req_tx, results: res_rx }
    }
}

impl FrameGrabber {
    pub fn request(&self, input: &Path, info: &VideoInfo, time: f64, w: usize, h: usize) {
        let _ = self.requests.send((input.to_path_buf(), info.clone(), time, w, h));
    }

    /// The newest finished frame, if any arrived since the last call.
    pub fn poll(&self) -> Option<Grabbed> {
        self.results.try_iter().last()
    }
}
