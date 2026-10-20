//! Turns per-tile difference statistics into new/duplicate decisions.
//!
//! Compressed captures never contain bit-identical duplicates: encoder noise makes
//! every frame differ slightly. trdrop's rule ("any pixel differs by more than N")
//! therefore counts almost every frame as new. Instead we score each frame by the
//! mean absolute difference of its *most-changed tile* (small real motion still
//! stands out, while noise spread over the frame doesn't add up), then split the
//! scores into a noise cluster and a motion cluster with Otsu's method in log space,
//! or at the widest gap between them when Otsu can't tell them apart.

use crate::gpu::GpuDiff;
use crate::progress::Progress;
use crate::video::{self, VideoInfo};
use anyhow::{Result, bail};
use std::path::Path;

pub const TILE: usize = 64;

#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    /// Width of the decoded frames: the source width padded to a multiple of 4 so
    /// frames pack into u32 words on the GPU. The padding is black in every frame.
    pub width: usize,
    pub height: usize,
    /// Real width of the video.
    pub source_width: usize,
}

impl Geometry {
    pub fn for_video(info: &VideoInfo) -> Self {
        Self { width: info.width.div_ceil(4) * 4, height: info.height, source_width: info.width }
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
    /// Real pixel count of a tile (edge tiles are partial; padding doesn't count).
    pub fn tile_pixels(&self, tile: usize) -> usize {
        let (tx, ty) = (tile % self.tiles_x(), tile / self.tiles_x());
        let w = TILE.min(self.source_width - tx * TILE);
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
#[derive(Clone, Debug, PartialEq)]
pub struct FrameStats {
    /// Pixels whose luma difference exceeds the trdrop "pixel difference" margin.
    pub compat_count: u64,
    /// Sum of absolute luma differences per 64x64 tile.
    pub tile_sad: Vec<u32>,
}

impl FrameStats {
    /// Reduces the per-tile sums to what the analysis keeps.
    pub fn score(&self, geo: Geometry, ignored: &[bool]) -> FrameScore {
        let mut row_scores = vec![f32::NAN; geo.tiles_y()];
        let (mut score, mut max_tile) = (0.0, 0);
        for (t, &sad) in self.tile_sad.iter().enumerate() {
            if ignored[t] {
                continue;
            }
            let v = sad as f64 / geo.tile_pixels(t) as f64;
            if v > score {
                (score, max_tile) = (v, t);
            }
            let row = &mut row_scores[t / geo.tiles_x()];
            *row = if row.is_nan() { v as f32 } else { row.max(v as f32) };
        }
        FrameScore { score, max_tile, compat_count: self.compat_count, row_scores }
    }
}

/// What is kept of each frame comparison: a few numbers instead of every tile's
/// sum, so memory stays small on long recordings.
#[derive(Clone, Debug)]
pub struct FrameScore {
    /// Mean absolute luma difference of the most-changed tile.
    pub score: f64,
    /// Index of the tile that produced `score`.
    pub max_tile: usize,
    pub compat_count: u64,
    /// Score of the most-changed tile in each row of tiles (NaN = whole row ignored).
    pub row_scores: Vec<f32>,
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
    /// Regions (x, y, w, h in source pixels) excluded from scoring.
    pub ignore: &'a [[usize; 4]],
}

/// Decodes the video as 8-bit luma and compares each frame with the previous one.
/// Returns per-frame scores (frame 1 onwards) and the number of decoded frames.
pub fn run_diff(
    input: &Path,
    info: &VideoInfo,
    opts: &DiffOptions,
    progress: &Progress,
) -> Result<(Vec<FrameScore>, usize)> {
    let geo = Geometry::for_video(info);
    let ignored = geo.ignored_tiles(opts.ignore);
    let mut gpu = if opts.use_gpu { Some(GpuDiff::new(geo, opts.pixel_difference)?) } else { None };
    let (rx, mut child) =
        video::spawn_decoder(input, opts.hwaccel, (geo.width, geo.height), "gray", geo.frame_bytes())?;
    progress.stage("Analysing frame pacing", opts.max_frames.or(info.estimated_frames()).unwrap_or(0));
    let mut scores: Vec<FrameScore> = Vec::new();
    let mut prev: Option<Vec<u8>> = None;
    let mut frames = 0usize;
    for frame in rx {
        if opts.max_frames.is_some_and(|m| frames >= m) || progress.is_cancelled() {
            break;
        }
        frames += 1;
        progress.advance();
        match gpu.as_mut() {
            Some(g) => scores.extend(g.push(&frame)?.iter().map(|s| s.score(geo, &ignored))),
            None => {
                if let Some(p) = &prev {
                    scores.push(cpu_diff(geo, p, &frame, opts.pixel_difference).score(geo, &ignored));
                }
                prev = Some(frame);
            }
        }
    }
    if let Some(g) = gpu.as_mut() {
        scores.extend(g.finish()?.iter().map(|s| s.score(geo, &ignored)));
    }
    let _ = child.kill();
    let _ = child.wait();
    if progress.is_cancelled() {
        bail!("cancelled");
    }
    if frames < 2 {
        bail!("need at least 2 decoded frames, got {frames}");
    }
    Ok((scores, frames))
}

pub struct Analysis {
    /// New/duplicate flag for every frame, including frame 0 (always new).
    pub is_new: Vec<bool>,
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
/// Fewest scores the gap estimate needs on each side of the gap.
const MIN_GAP_SIDE: usize = 3;
/// A one-second stretch whose median score is below this many thresholds barely
/// changes, so a low frame rate there may just be a still scene.
const LOW_MOTION_FACTOR: f64 = 5.0;

fn log_score(s: f64) -> f64 {
    (s + 0.01).log10()
}

fn from_log(l: f64) -> f64 {
    10f64.powf(l) - 0.01
}

/// Otsu threshold over log10 scores; returns (threshold, low-class mean, high-class mean).
fn otsu(scores: &[f64]) -> Option<(f64, f64, f64)> {
    const BINS: usize = 512;
    let logs: Vec<f64> = scores.iter().map(|&s| log_score(s)).collect();
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
    // Empty bins between the clusters all score the same; take the middle of that
    // run so the threshold sits mid-gap, not right against the noise.
    let (mut w0, mut sum0, mut best, mut best_i, mut best_end) = (0.0, 0.0, -1.0, 0, 0);
    for (i, &h) in hist.iter().enumerate().take(BINS - 1) {
        w0 += h;
        sum0 += h * center(i);
        let w1 = total - w0;
        if w0 == 0.0 || w1 == 0.0 {
            continue;
        }
        let (m0, m1) = (sum0 / w0, (sum_all - sum0) / w1);
        let between = w0 * w1 * (m0 - m1).powi(2);
        if between > best {
            best = between;
            (best_i, best_end) = (i, i);
        } else if between == best && best_end + 1 == i {
            best_end = i;
        }
    }
    let thr = from_log((center(best_i) + center(best_end + 1)) / 2.0);
    let mean = |it: Vec<f64>| it.iter().sum::<f64>() / it.len().max(1) as f64;
    let low = mean(scores.iter().copied().filter(|&s| s <= thr).collect());
    let high = mean(scores.iter().copied().filter(|&s| s > thr).collect());
    Some((thr, low, high))
}

/// Middle of the widest gap between sorted log scores that lands in the plausible
/// threshold range. Catches videos where duplicates are rare, or include exact
/// zeros, and Otsu splits off the wrong cluster.
fn widest_gap(scores: &[f64]) -> Option<f64> {
    let mut logs: Vec<f64> = scores.iter().map(|&s| log_score(s)).collect();
    logs.sort_by(f64::total_cmp);
    let n = logs.len();
    (MIN_GAP_SIDE..n.saturating_sub(MIN_GAP_SIDE - 1))
        .map(|i| (logs[i] - logs[i - 1], from_log((logs[i] + logs[i - 1]) / 2.0)))
        .filter(|&(gap, thr)| gap >= MIN_SEPARATION.log10() && (MIN_THRESHOLD..=MAX_THRESHOLD).contains(&thr))
        .max_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, thr)| thr)
}

pub fn analyze(scores: &[FrameScore], forced: Option<f64>) -> Analysis {
    let values: Vec<f64> = scores.iter().map(|s| s.score).collect();
    let (threshold, threshold_source) = match forced {
        Some(t) => (t, "user"),
        None => match otsu(&values) {
            Some((t, low, high))
                if (MIN_THRESHOLD..=MAX_THRESHOLD).contains(&t) && high >= MIN_SEPARATION * low.max(0.05) =>
            {
                (t, "estimated from noise floor")
            }
            _ => match widest_gap(&values) {
                Some(t) => (t, "estimated from gap between noise and motion"),
                None => (FALLBACK_THRESHOLD, "fallback (scores not bimodal)"),
            },
        },
    };
    let is_new = std::iter::once(true).chain(values.iter().map(|&s| s > threshold)).collect();
    Analysis { is_new, threshold, threshold_source }
}

/// Frames (including frame 0) whose trailing one-second stretch barely changes:
/// a still scene repeats frames whatever the game's frame rate, so FPS there is
/// not a measure of performance.
pub fn low_motion(scores: &[FrameScore], threshold: f64, capture_fps: f64) -> Vec<bool> {
    let window = capture_fps.round().max(1.0) as usize;
    let limit = LOW_MOTION_FACTOR * threshold;
    let mut out = vec![false; scores.len() + 1];
    let mut buf = Vec::with_capacity(window);
    for i in 1..=scores.len() {
        buf.clear();
        buf.extend(scores[i.saturating_sub(window)..i].iter().map(|s| s.score));
        buf.sort_by(f64::total_cmp);
        out[i] = buf[buf.len() / 2] < limit;
    }
    out[0] = out.get(1).copied().unwrap_or(false);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scores(values: &[f64]) -> Vec<FrameScore> {
        values.iter().map(|&score| FrameScore { score, max_tile: 0, compat_count: 0, row_scores: vec![] }).collect()
    }

    /// Deterministic spread of `n` values around `centre` (within ±30%).
    fn cluster(centre: f64, n: usize) -> impl Iterator<Item = f64> {
        (0..n).map(move |i| centre * (0.7 + 0.6 * ((i * 37) % 101) as f64 / 100.0))
    }

    #[test]
    fn otsu_splits_thirty_in_sixty() {
        let v: Vec<f64> = cluster(0.3, 500).zip(cluster(40.0, 500)).flat_map(|(a, b)| [a, b]).collect();
        let a = analyze(&scores(&v), None);
        assert_eq!(a.threshold_source, "estimated from noise floor");
        // Mid-gap (log scale) between noise (up to 0.39) and motion (from 28).
        assert!(a.threshold > 2.0 && a.threshold < 6.0, "{}", a.threshold);
        assert_eq!(a.is_new.iter().filter(|&&n| n).count(), 501);
    }

    #[test]
    fn gap_handles_rare_duplicates_and_zeros() {
        // Like a locked 60 fps capture: a handful of exact repeats, a few noise-level
        // duplicates, and everything else clearly moving.
        let mut v = vec![0.0; 11];
        v.extend(cluster(0.3, 15));
        v.extend(cluster(60.0, 11_000));
        v.push(1.8); // slow but real motion
        let a = analyze(&scores(&v), None);
        assert!(a.threshold > 0.4 && a.threshold < 1.8, "{}", a.threshold);
        assert_eq!(a.is_new.iter().filter(|&&n| !n).count(), 26);
    }

    #[test]
    fn widest_gap_skips_implausible_gaps() {
        // Gaps below 0.2 (zeros to noise) and above 8 (slow to fast motion) are ignored.
        let mut v = vec![0.0; 5];
        v.extend(cluster(0.3, 10));
        v.extend([1.8, 1.9, 2.0]);
        v.extend(cluster(60.0, 50));
        let t = widest_gap(&v).unwrap();
        assert!(t > 0.4 && t < 1.8, "{t}");
    }

    #[test]
    fn falls_back_without_duplicates() {
        let v: Vec<f64> = cluster(50.0, 1000).collect();
        let a = analyze(&scores(&v), None);
        assert_eq!(a.threshold, FALLBACK_THRESHOLD);
        assert!(a.is_new.iter().all(|&n| n));
    }

    #[test]
    fn flags_still_stretches() {
        let mut v: Vec<f64> = cluster(50.0, 120).collect();
        v.extend(cluster(2.0, 120));
        let low = low_motion(&scores(&v), 1.0, 60.0);
        assert!(!low[100]);
        assert!(low[230]);
    }

    #[test]
    fn padding_does_not_dilute_edge_tiles() {
        let info = VideoInfo {
            width: 130,
            height: 64,
            fps: 60.0,
            duration: None,
            color: crate::nits::ColorInfo::from_probe("bt709", "bt709", "tv"),
        };
        let geo = Geometry::for_video(&info);
        assert_eq!((geo.width, geo.tiles_x()), (132, 3));
        assert_eq!(geo.tile_pixels(2), 2 * 64);
        // A change confined to the two real columns of the last tile scores fully.
        let prev = vec![0u8; geo.frame_bytes()];
        let mut cur = prev.clone();
        for y in 0..64 {
            cur[y * geo.width + 128] = 10;
            cur[y * geo.width + 129] = 10;
        }
        let s = cpu_diff(geo, &prev, &cur, 0).score(geo, &vec![false; geo.tiles()]);
        assert_eq!(s.max_tile, 2);
        assert!((s.score - 10.0).abs() < 1e-9);
    }

    #[test]
    fn gpu_matches_cpu() {
        let geo = Geometry { width: 200, height: 136, source_width: 198 };
        let Ok(mut gpu) = GpuDiff::new(geo, 3) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let frames: Vec<Vec<u8>> = (0..5u32)
            .map(|f| (0..geo.frame_bytes() as u32).map(|i| ((i * 7 + f * 13 + (i / 200) * f) % 251) as u8).collect())
            .collect();
        let mut from_gpu = Vec::new();
        for f in &frames {
            from_gpu.extend(gpu.push(f).unwrap());
        }
        from_gpu.extend(gpu.finish().unwrap());
        let from_cpu: Vec<FrameStats> = frames.windows(2).map(|w| cpu_diff(geo, &w[0], &w[1], 3)).collect();
        assert_eq!(from_gpu, from_cpu);
    }
}
