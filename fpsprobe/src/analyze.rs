//! Turns per-tile difference statistics into new/duplicate decisions and FPS numbers.
//!
//! Compressed captures never contain bit-identical duplicates: encoder noise makes
//! every frame differ slightly. trdrop's rule ("any pixel differs by more than N")
//! therefore counts almost every frame as new. Instead we score each frame by the
//! mean absolute difference of its *most-changed tile* (small real motion still
//! stands out, while noise spread over the frame doesn't add up), then split the
//! scores into a noise cluster and a motion cluster with Otsu's method in log space.

use crate::gpu::GpuDiff;
use crate::progress::Progress;
use crate::video::{self, VideoInfo};
use anyhow::{Result, bail};
use std::path::Path;

pub const TILE: usize = 64;

#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub width: usize,
    pub height: usize,
}

impl Geometry {
    /// Width is cropped to a multiple of 4 so frames pack into u32 words on the GPU.
    pub fn for_video(info: &VideoInfo) -> Self {
        Self { width: info.width / 4 * 4, height: info.height }
    }
    pub fn tiles_x(&self) -> usize {
        self.width.div_ceil(TILE)
    }
    pub fn tiles_y(&self) -> usize {
        self.height.div_ceil(TILE)
    }
    pub fn tiles(&self) -> usize {
        self.tiles_x() * self.tiles_y()
    }
    pub fn frame_bytes(&self) -> usize {
        self.width * self.height
    }
    /// Pixel count of a tile (edge tiles are partial).
    pub fn tile_pixels(&self, tile: usize) -> usize {
        let (tx, ty) = (tile % self.tiles_x(), tile / self.tiles_x());
        let w = TILE.min(self.width - tx * TILE);
        let h = TILE.min(self.height - ty * TILE);
        w * h
    }
    /// Marks every tile touching one of the `x,y,w,h` pixel rectangles.
    pub fn ignored_tiles(&self, rects: &[[usize; 4]]) -> Vec<bool> {
        let mut ignored = vec![false; self.tiles()];
        for [x, y, w, h] in rects {
            for ty in y / TILE..((y + h).div_ceil(TILE)).min(self.tiles_y()) {
                for tx in x / TILE..((x + w).div_ceil(TILE)).min(self.tiles_x()) {
                    ignored[ty * self.tiles_x() + tx] = true;
                }
            }
        }
        ignored
    }
}

/// Raw comparison of frame N against frame N-1.
#[derive(Clone, Debug)]
pub struct FrameStats {
    /// Pixels whose luma difference exceeds the trdrop "pixel difference" margin.
    pub compat_count: u64,
    /// Sum of absolute luma differences per 64x64 tile.
    pub tile_sad: Vec<u32>,
}

/// CPU reference implementation; produces exactly the same stats as diff.wgsl.
pub fn cpu_diff(geo: Geometry, prev: &[u8], cur: &[u8], compat_thr: u32) -> FrameStats {
    let mut tile_sad = vec![0u32; geo.tiles()];
    let mut compat = 0u64;
    let tx = geo.tiles_x();
    for y in 0..geo.height {
        let row = y * geo.width;
        let trow = (y / TILE) * tx;
        for (x, (&a, &b)) in prev[row..row + geo.width].iter().zip(&cur[row..row + geo.width]).enumerate() {
            let d = a.abs_diff(b) as u32;
            tile_sad[trow + x / TILE] += d;
            compat += (d > compat_thr) as u64;
        }
    }
    FrameStats { compat_count: compat, tile_sad }
}

pub struct DiffOptions<'a> {
    pub hwaccel: Option<&'a str>,
    pub use_gpu: bool,
    pub pixel_difference: u32,
    pub max_frames: Option<usize>,
}

/// Decodes the video as 8-bit luma and compares each frame with the previous one.
/// Returns per-frame stats (frame 1 onwards) and the number of decoded frames.
pub fn run_diff(input: &Path, info: &VideoInfo, opts: &DiffOptions, progress: &Progress) -> Result<(Vec<FrameStats>, usize)> {
    let geo = Geometry::for_video(info);
    let mut gpu = if opts.use_gpu { Some(GpuDiff::new(geo, opts.pixel_difference)?) } else { None };
    let (rx, mut child) = video::spawn_decoder(input, opts.hwaccel, (geo.width, geo.height), "gray", geo.frame_bytes())?;
    progress.stage("Analysing frame pacing", opts.max_frames.or(info.estimated_frames()).unwrap_or(0));
    let mut stats: Vec<FrameStats> = Vec::new();
    let mut prev: Option<Vec<u8>> = None;
    let mut frames = 0usize;
    for frame in rx {
        if opts.max_frames.is_some_and(|m| frames >= m) || progress.is_cancelled() {
            break;
        }
        frames += 1;
        progress.advance();
        match gpu.as_mut() {
            Some(g) => stats.extend(g.push(&frame)?),
            None => {
                if let Some(p) = &prev {
                    stats.push(cpu_diff(geo, p, &frame, opts.pixel_difference));
                }
                prev = Some(frame);
            }
        }
    }
    if let Some(g) = gpu.as_mut() {
        stats.extend(g.finish()?);
    }
    let _ = child.kill();
    let _ = child.wait();
    if progress.is_cancelled() {
        bail!("cancelled");
    }
    if frames < 2 {
        bail!("need at least 2 decoded frames, got {frames}");
    }
    Ok((stats, frames))
}

pub struct FrameResult {
    pub score: f64,
    /// Index of the tile that produced `score`.
    pub max_tile: usize,
    pub is_new: bool,
}

pub struct Analysis {
    pub frames: Vec<FrameResult>,
    pub threshold: f64,
    pub threshold_source: &'static str,
}

/// Used when the score distribution isn't bimodal (no duplicates, or all duplicates).
const FALLBACK_THRESHOLD: f64 = 1.5;
/// Estimated thresholds outside this range are treated as unreliable.
const MIN_THRESHOLD: f64 = 0.2;
const MAX_THRESHOLD: f64 = 8.0;
/// The motion cluster must be this many times the noise cluster to count as bimodal.
const MIN_SEPARATION: f64 = 3.0;

/// Mean absolute difference of the most-changed tile, and that tile's index.
fn frame_score(geo: Geometry, s: &FrameStats, ignored: &[bool]) -> (f64, usize) {
    s.tile_sad
        .iter()
        .enumerate()
        .filter(|(t, _)| !ignored[*t])
        .map(|(t, &sad)| (sad as f64 / geo.tile_pixels(t) as f64, t))
        .fold((0.0, 0), |a, b| if b.0 > a.0 { b } else { a })
}

/// Otsu threshold over log10 scores; returns (threshold, low-class mean, high-class mean).
fn otsu(scores: &[f64]) -> Option<(f64, f64, f64)> {
    const BINS: usize = 512;
    let logs: Vec<f64> = scores.iter().map(|s| (s + 0.01).log10()).collect();
    let (lo, hi) = logs.iter().fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    if hi - lo < 1e-6 {
        return None;
    }
    let bin = |v: f64| (((v - lo) / (hi - lo)) * (BINS - 1) as f64).round() as usize;
    let mut hist = [0f64; BINS];
    for &v in &logs {
        hist[bin(v)] += 1.0;
    }
    let center = |i: usize| lo + (hi - lo) * i as f64 / (BINS - 1) as f64;
    let total = logs.len() as f64;
    let sum_all: f64 = (0..BINS).map(|i| hist[i] * center(i)).sum();
    let (mut w0, mut sum0, mut best, mut best_i) = (0.0, 0.0, -1.0, 0);
    for i in 0..BINS - 1 {
        w0 += hist[i];
        sum0 += hist[i] * center(i);
        let w1 = total - w0;
        if w0 == 0.0 || w1 == 0.0 {
            continue;
        }
        let (m0, m1) = (sum0 / w0, (sum_all - sum0) / w1);
        let between = w0 * w1 * (m0 - m1).powi(2);
        if between > best {
            best = between;
            best_i = i;
        }
    }
    let t_log = (center(best_i) + center(best_i + 1)) / 2.0;
    let thr = 10f64.powf(t_log) - 0.01;
    let mean = |it: Vec<f64>| it.iter().sum::<f64>() / it.len().max(1) as f64;
    let low = mean(scores.iter().copied().filter(|&s| s <= thr).collect());
    let high = mean(scores.iter().copied().filter(|&s| s > thr).collect());
    Some((thr, low, high))
}

pub fn analyze(geo: Geometry, stats: &[FrameStats], ignored: &[bool], forced: Option<f64>) -> Analysis {
    let scored: Vec<(f64, usize)> = stats.iter().map(|s| frame_score(geo, s, ignored)).collect();
    let scores: Vec<f64> = scored.iter().map(|s| s.0).collect();
    let (threshold, threshold_source) = match forced {
        Some(t) => (t, "user"),
        None => match otsu(&scores) {
            Some((t, low, high))
                if (MIN_THRESHOLD..=MAX_THRESHOLD).contains(&t) && high >= MIN_SEPARATION * low.max(0.05) =>
            {
                (t, "estimated from noise floor")
            }
            _ => (FALLBACK_THRESHOLD, "fallback (scores not bimodal)"),
        },
    };
    let frames = scored
        .into_iter()
        .map(|(score, max_tile)| FrameResult { score, max_tile, is_new: score > threshold })
        .collect();
    Analysis { frames, threshold, threshold_source }
}

struct Summary {
    avg_fps: f64,
    min_1s_fps: f64,
    low_1pct_fps: f64,
    unique: usize,
}

/// `new_flags` excludes frame 0, which is always treated as new.
fn summarize(new_flags: &[bool], capture_fps: f64) -> Summary {
    let total = new_flags.len() + 1;
    let unique = 1 + new_flags.iter().filter(|&&n| n).count();
    let duration = total as f64 / capture_fps;

    // Per-second FPS over a sliding one-second window (as trdrop does).
    let window = capture_fps.round().max(1.0) as usize;
    let all: Vec<bool> = std::iter::once(true).chain(new_flags.iter().copied()).collect();
    let min_1s_fps = if all.len() >= window {
        all.windows(window).map(|w| w.iter().filter(|&&n| n).count()).min().unwrap_or(0) as f64
            * capture_fps
            / window as f64
    } else {
        f64::NAN
    };

    // Frame times between consecutive new frames; 1% low = FPS at the 99th percentile frame time.
    let idx: Vec<usize> = all.iter().enumerate().filter(|(_, n)| **n).map(|(i, _)| i).collect();
    let mut ft: Vec<f64> = idx.windows(2).map(|w| (w[1] - w[0]) as f64 / capture_fps).collect();
    ft.sort_by(f64::total_cmp);
    let low_1pct_fps = if ft.is_empty() {
        f64::NAN
    } else {
        1.0 / ft[((ft.len() as f64 * 0.99).ceil() as usize).clamp(1, ft.len()) - 1]
    };

    Summary { avg_fps: unique as f64 / duration, min_1s_fps, low_1pct_fps, unique }
}

/// Human-readable summary comparing fpsprobe with trdrop's original rule.
pub fn summary_text(a: &Analysis, capture_fps: f64, stats: &[FrameStats]) -> String {
    let new: Vec<bool> = a.frames.iter().map(|f| f.is_new).collect();
    let compat: Vec<bool> = stats.iter().map(|s| s.compat_count > 0).collect();
    let s = summarize(&new, capture_fps);
    let c = summarize(&compat, capture_fps);
    let total = new.len() + 1;
    let mut out = String::new();
    out += &format!("threshold        {:.3} ({})
", a.threshold, a.threshold_source);
    out += &format!("                 {:>10} {:>14}
", "fpsprobe", "trdrop-rule");
    out += &format!("unique frames    {:>10} {:>14}   of {total}
", s.unique, c.unique);
    out += &format!("average fps      {:>10.2} {:>14.2}
", s.avg_fps, c.avg_fps);
    out += &format!("min 1s fps       {:>10.2} {:>14.2}
", s.min_1s_fps, c.min_1s_fps);
    out += &format!("1% low fps       {:>10.2} {:>14.2}
", s.low_1pct_fps, c.low_1pct_fps);
    out
}
