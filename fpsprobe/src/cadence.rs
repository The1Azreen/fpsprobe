//! Patterns in the new/duplicate sequence that say something about the capture
//! rather than the game: drops from a capture clock that runs slightly fast, and
//! (experimental) torn frames.

use crate::analyze::{FrameScore, TILE};
use crate::timing::Timeline;

/// Shortest repeat interval accepted as a capture clock mismatch. Faster regular
/// duplicates are much more likely to be the game's own pacing.
const MIN_PERIOD_S: f64 = 2.0;
/// At most one isolated duplicate per this many seconds, on average, before the
/// game is assumed not to be locked to the capture rate and detection is skipped.
const MAX_DROP_RATE_S: f64 = 1.0;
/// Fewest drops that must line up before a pattern is believed.
const MIN_HITS: usize = 4;
/// How far (in frames) a drop may sit from the fitted grid and still be on it.
const GRID_TOLERANCE: f64 = 2.5;

/// Single duplicate frames repeating at a fixed interval: the capture device runs
/// slightly faster than the source (e.g. 60 vs 59.94 Hz) and shows one frame twice
/// every so often. They are not the game dropping frames.
#[derive(Clone, Debug)]
pub struct CaptureDrops {
    /// Indices (including frame 0) of the duplicates caused by the capture.
    pub frames: Vec<usize>,
    pub period_s: f64,
}

impl CaptureDrops {
    /// How far the capture clock is off the source, as a fraction (0.001 = 0.1%).
    pub fn clock_offset(&self, fps: f64) -> f64 {
        1.0 / (self.period_s * fps)
    }
}

/// Finds duplicates that repeat on a fixed grid. `is_new` includes frame 0.
pub fn capture_drops(is_new: &[bool], timeline: &Timeline) -> Option<CaptureDrops> {
    let fps = timeline.fps;
    // Isolated duplicates: the game is otherwise delivering every frame. The first
    // couple of frames are skipped because recordings often start on a repeat.
    let drops: Vec<usize> =
        (2..is_new.len().saturating_sub(1)).filter(|&i| !is_new[i] && is_new[i - 1] && is_new[i + 1]).collect();
    let duration = timeline.duration(is_new.len());
    if drops.len() < MIN_HITS || drops.len() as f64 > duration / MAX_DROP_RATE_S {
        return None;
    }
    let times: Vec<f64> = drops.iter().map(|&i| timeline.time(i)).collect();

    // Candidate grids: each pair of neighbouring drops, assuming 1-4 periods between them.
    let mut best: Option<(f64, Vec<usize>, f64)> = None;
    for w in times.windows(2) {
        for m in 1..=4 {
            let period = (w[1] - w[0]) / m as f64;
            if period < MIN_PERIOD_S {
                continue;
            }
            let Some((hits, period)) = fit_grid(&times, w[0], period, fps) else { continue };
            let k = |t: f64| ((t - w[0]) / period).round() as i64;
            let span = (k(times[*hits.last().unwrap()]) - k(times[hits[0]]) + 1) as f64;
            // Grid points without a drop count against a grid: a short period always
            // finds hits, but leaves most of its points empty.
            let score = hits.len() as f64 - 0.5 * (span - hits.len() as f64);
            if best.as_ref().is_none_or(|b| score > b.2 + 1e-9 || (score > b.2 - 1e-9 && period > b.0)) {
                best = Some((period, hits, score));
            }
        }
    }
    let (period_s, hits, _) = best?;
    // A believable grid is mostly filled: a few points may be hidden when the game
    // repeated a frame at the same moment (the duplicate is then not isolated).
    let first = times[hits[0]];
    let points = ((times[*hits.last().unwrap()] - first) / period_s).round() as usize + 1;
    if hits.len() < MIN_HITS || (points - hits.len()) * 3 > hits.len() {
        return None;
    }
    Some(CaptureDrops { frames: hits.iter().map(|&h| drops[h]).collect(), period_s })
}

/// Drops on the grid `anchor + k * period`, refitting the period by least squares
/// so a slightly wrong starting period doesn't lose drops far from the anchor.
/// Returns indices into `times` and the refined period.
fn fit_grid(times: &[f64], anchor: f64, mut period: f64, fps: f64) -> Option<(Vec<usize>, f64)> {
    let tol = GRID_TOLERANCE / fps;
    let mut hits = Vec::new();
    for pass in 0..3 {
        // The first pass widens the tolerance with distance from the anchor.
        let slack = if pass == 0 { 0.5 / fps } else { 0.0 };
        hits = (0..times.len())
            .filter(|&i| {
                let k = ((times[i] - anchor) / period).round();
                (times[i] - anchor - k * period).abs() <= tol + slack * k.abs()
            })
            .collect();
        if hits.len() < 2 {
            return None;
        }
        // Least-squares slope of drop time against grid index.
        let ks: Vec<f64> = hits.iter().map(|&i| ((times[i] - anchor) / period).round()).collect();
        let n = ks.len() as f64;
        let (mk, mt) = (ks.iter().sum::<f64>() / n, hits.iter().map(|&i| times[i]).sum::<f64>() / n);
        let var: f64 = ks.iter().map(|k| (k - mk).powi(2)).sum();
        if var == 0.0 {
            return None;
        }
        let cov: f64 = ks.iter().zip(&hits).map(|(k, &i)| (k - mk) * (times[i] - mt)).sum();
        period = cov / var;
        if period < MIN_PERIOD_S {
            return None;
        }
    }
    Some((hits, period))
}

/// Each side of a tear must cover at least this share of the rows. Thin bands at the
/// edges are usually letterboxing or HUD strips starting to move, not tears.
const MIN_TEAR_BAND: f64 = 0.2;

/// Experimental tear detection, per row of tiles, so the reported line is approximate.
///
/// A torn frame shows the previous image above (or below) the tear and a new one on
/// the other side. Compared with the previous captured frame, one band of rows is
/// unchanged and the rest changed; in the next frame the unchanged band catches up.
/// Requiring that catch-up rules out scenes that are simply still in one half. The
/// catch-up frame itself looks split the other way at the same row and is skipped.
///
/// `scores[j]` is frame j+1's comparison. Returns the tear's approximate y position
/// in pixels for each frame (including frame 0).
pub fn tears(scores: &[FrameScore], threshold: f64) -> Vec<Option<usize>> {
    let thr = threshold as f32;
    let rows: Vec<&[f32]> = scores.iter().map(|s| s.row_scores.as_slice()).collect();
    let mut out = vec![None; rows.len() + 1];
    // Row and orientation (unchanged band on top?) of the previous frame's tear.
    let mut prev: Option<(usize, bool)> = None;
    for j in 0..rows.len().saturating_sub(1) {
        let found = prev.take();
        let (cur, next) = (rows[j], rows[j + 1]);
        let valid: Vec<usize> = (0..cur.len()).filter(|&r| !cur[r].is_nan()).collect();
        if valid.len() < 2 {
            continue;
        }
        let moving: Vec<bool> = valid.iter().map(|&r| cur[r] > thr).collect();
        // Exactly one switch between unchanged and changed rows.
        let switches: Vec<usize> = (1..moving.len()).filter(|&k| moving[k] != moving[k - 1]).collect();
        let [k] = switches[..] else { continue };
        let min_band = (MIN_TEAR_BAND * valid.len() as f64).ceil() as usize;
        if k < min_band || valid.len() - k < min_band {
            continue;
        }
        let still_on_top = !moving[0];
        if found.is_some_and(|(row, top)| row == valid[k] && top != still_on_top) {
            continue;
        }
        let still: Vec<usize> = if still_on_top { valid[..k].to_vec() } else { valid[k..].to_vec() };
        let caught_up = still.iter().filter(|&&r| next[r] > thr).count();
        if caught_up * 2 > still.len() {
            out[j + 1] = Some(valid[k] * TILE);
            prev = Some((valid[k], still_on_top));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_drops(n: usize, at: impl Iterator<Item = usize>) -> Vec<bool> {
        let mut v = vec![true; n];
        for i in at {
            v[i] = false;
        }
        v
    }

    #[test]
    fn finds_clock_mismatch_drops() {
        // 60 vs 59.94 Hz: one repeat every 1001 frames (16.68 s), with a game hitch mixed in.
        let t = Timeline::constant(60.0);
        let mut drops: Vec<usize> = (0..10).map(|k| 500 + k * 1001).collect();
        drops.push(3000);
        let new = with_drops(11_000, drops.into_iter());
        let c = capture_drops(&new, &t).expect("pattern found");
        assert_eq!(c.frames.len(), 10);
        assert!(!c.frames.contains(&3000));
        assert!((c.period_s - 1001.0 / 60.0).abs() < 0.02, "{}", c.period_s);
        assert!((c.clock_offset(60.0) - 0.000999).abs() < 0.00005);
    }

    #[test]
    fn tolerates_jitter_and_a_hidden_drop() {
        // Drops wobble by a frame (capture timing) and one lands on a game hitch.
        let t = Timeline::constant(60.0);
        let wobble = [0i64, 1, -1, 0, 1, 0, -1, 0];
        let drops = (0..8).filter(|&k| k != 4).map(|k| (300 + k as i64 * 980 + wobble[k]) as usize);
        let c = capture_drops(&with_drops(8_500, drops), &t).expect("pattern found");
        assert_eq!(c.frames.len(), 7);
    }

    #[test]
    fn ignores_random_hitches() {
        let t = Timeline::constant(60.0);
        let drops = [400usize, 1700, 2100, 4900, 5300, 8800, 9100];
        assert!(capture_drops(&with_drops(10_000, drops.into_iter()), &t).is_none());
    }

    #[test]
    fn ignores_game_cadence() {
        // A game at 50 fps in a 60 fps capture repeats a frame every 6 frames.
        let t = Timeline::constant(60.0);
        assert!(capture_drops(&with_drops(6_000, (5..6_000).step_by(6)), &t).is_none());
    }

    fn rows(r: Vec<Vec<f32>>) -> Vec<FrameScore> {
        r.into_iter().map(|row_scores| FrameScore { score: 0.0, max_tile: 0, compat_count: 0, row_scores }).collect()
    }

    #[test]
    fn detects_tear_and_its_row() {
        let thr = 1.0;
        let moving = 20.0f32;
        // Frame 1: top 3 rows unchanged, bottom 3 changed. Frame 2: top catches up.
        let scores = rows(vec![
            vec![0.3, 0.3, 0.3, moving, moving, moving],
            vec![moving, moving, moving, 0.3, 0.3, 0.3],
            vec![moving; 6],
        ]);
        let t = tears(&scores, thr);
        assert_eq!(t[1], Some(3 * 64));
        assert_eq!(t[2], None); // the catch-up frame, split the other way at the same row
        assert_eq!(t[3], None); // last frame has no next frame to confirm
    }

    #[test]
    fn thin_edge_band_is_not_a_tear() {
        let mut first = vec![20.0f32; 10];
        first[0] = 0.3; // top row still, then catches up
        let scores = rows(vec![first, vec![20.0; 10], vec![20.0; 10]]);
        assert!(tears(&scores, 1.0).iter().all(Option::is_none));
    }

    #[test]
    fn still_sky_is_not_a_tear() {
        let still = vec![0.3, 0.3, 20.0, 20.0];
        let scores = rows(vec![still.clone(), still.clone(), still]);
        assert!(tears(&scores, 1.0).iter().all(Option::is_none));
    }
}
