//! Frame timing: when each captured frame was shown, and the FPS / frame-time
//! numbers that follow from which frames were new.
//!
//! Most captures have a constant frame rate, where frame i is shown at i / fps.
//! Variable frame rate recordings (OBS, phones) carry real timestamps instead, and
//! frame times are only right when those are used.

/// Share of frame intervals that must be off the nominal spacing before a video is
/// treated as variable frame rate.
const VFR_SHARE: f64 = 0.005;
/// How far (in frames) an interval may be off the nominal spacing and still count
/// as regular. Covers containers that round timestamps to milliseconds.
const VFR_TOLERANCE: f64 = 0.25;

#[derive(Clone, Debug)]
pub struct Timeline {
    /// Nominal capture rate: graph scales, window lengths and output frame rate.
    pub fps: f64,
    /// Seconds from the first frame, when the frame rate is variable.
    times: Option<Vec<f64>>,
}

impl Timeline {
    pub fn constant(fps: f64) -> Self {
        Self { fps, times: None }
    }

    /// Uses `timestamps` only when they show a variable frame rate. Constant-rate
    /// timestamps add nothing but rounding jitter. `frames` is the decoded count.
    pub fn from_timestamps(fps: f64, timestamps: Option<Vec<f64>>, frames: usize) -> Self {
        let Some(mut times) = timestamps.filter(|t| frames >= 2 && t.len() >= frames) else {
            return Self::constant(fps);
        };
        times.truncate(frames);
        let nominal = 1.0 / fps;
        let irregular = times.windows(2).filter(|w| (w[1] - w[0] - nominal).abs() > VFR_TOLERANCE * nominal).count();
        if (irregular as f64) < VFR_SHARE * (frames - 1) as f64 {
            return Self::constant(fps);
        }
        // Timestamps close to the nominal grid are put on it, so millisecond rounding
        // doesn't show up as 16 / 17 ms frame-time jitter.
        for t in &mut times {
            let slot = (*t * fps).round();
            if (*t * fps - slot).abs() <= VFR_TOLERANCE {
                *t = slot / fps;
            }
        }
        Self { fps, times: Some(times) }
    }

    pub fn is_variable(&self) -> bool {
        self.times.is_some()
    }

    /// When frame `i` is shown, in seconds from the first frame.
    pub fn time(&self, i: usize) -> f64 {
        match &self.times {
            Some(t) if i < t.len() => t[i],
            Some(t) => t[t.len() - 1] + (i + 1 - t.len()) as f64 / self.fps,
            None => i as f64 / self.fps,
        }
    }

    /// Length covered by `frames` frames; the last one lasts one nominal frame.
    pub fn duration(&self, frames: usize) -> f64 {
        if frames == 0 { 0.0 } else { self.time(frames - 1) + 1.0 / self.fps }
    }

    /// Frame index at the nominal rate that each frame is shown in. Rendered outputs
    /// run at the nominal rate, so a variable-rate frame can cover several slots.
    pub fn slots(&self, frames: usize) -> Vec<usize> {
        let mut last = 0;
        (0..frames)
            .map(|i| {
                last = last.max((self.time(i) * self.fps).round() as usize);
                last
            })
            .collect()
    }

    /// Per-frame FPS over a trailing one-second window (NaN until a full second has
    /// elapsed) and most recent frame time in ms. `is_new` includes frame 0.
    pub fn fps_series(&self, is_new: &[bool]) -> (Vec<f64>, Vec<f64>) {
        let mut fps = Vec::with_capacity(is_new.len());
        let mut frametime_ms = Vec::with_capacity(is_new.len());
        let window = self.fps.round().max(1.0) as usize;
        let mut in_window = 0usize;
        let mut oldest = 0usize;
        let mut last_new = 0usize;
        let mut last_ft = 1000.0 / self.fps;
        for i in 0..is_new.len() {
            in_window += is_new[i] as usize;
            match self.times {
                // A window of whole frames keeps fractional rates (59.94) steady.
                None => {
                    if i >= window {
                        in_window -= is_new[i - window] as usize;
                    }
                    fps.push(if i + 1 >= window { in_window as f64 * self.fps / window as f64 } else { f64::NAN });
                }
                // Frames shown in the last second, with a little slack for rounded timestamps.
                Some(_) => {
                    let now = self.time(i);
                    while now - self.time(oldest) > 1.0 - 0.25 / self.fps {
                        in_window -= is_new[oldest] as usize;
                        oldest += 1;
                    }
                    fps.push(if now + 1.0 / self.fps >= 1.0 - 1e-9 { in_window as f64 } else { f64::NAN });
                }
            }
            if is_new[i] && i > 0 {
                last_ft = (self.time(i) - self.time(last_new)) * 1000.0;
                last_new = i;
            }
            frametime_ms.push(last_ft);
        }
        (fps, frametime_ms)
    }
}

/// Headline frame-rate numbers for one set of new/duplicate decisions.
#[derive(Clone, Debug, serde::Serialize)]
pub struct PacingStats {
    pub unique: usize,
    pub avg_fps: f64,
    pub min_1s_fps: f64,
    /// FPS at the 99th-percentile frame time.
    pub low_1pct_p99: f64,
    /// FPS from the average of the slowest 1% of frame times.
    pub low_1pct_avg: f64,
}

impl PacingStats {
    /// `is_new` includes frame 0.
    pub fn new(is_new: &[bool], timeline: &Timeline) -> Self {
        let unique = is_new.iter().filter(|&&n| n).count();
        let (fps, _) = timeline.fps_series(is_new);
        let min_1s_fps = min_ignoring_nan(fps.iter().copied());

        let idx: Vec<usize> = (0..is_new.len()).filter(|&i| is_new[i]).collect();
        let mut ft: Vec<f64> = idx.windows(2).map(|w| timeline.time(w[1]) - timeline.time(w[0])).collect();
        ft.sort_by(f64::total_cmp);
        let (low_1pct_p99, low_1pct_avg) = if ft.is_empty() {
            (f64::NAN, f64::NAN)
        } else {
            let p99 = ft[((ft.len() as f64 * 0.99).ceil() as usize).clamp(1, ft.len()) - 1];
            let worst = &ft[ft.len() - (ft.len() as f64 * 0.01).ceil().max(1.0) as usize..];
            (1.0 / p99, worst.len() as f64 / worst.iter().sum::<f64>())
        };
        Self {
            unique,
            avg_fps: unique as f64 / timeline.duration(is_new.len()),
            min_1s_fps,
            low_1pct_p99,
            low_1pct_avg,
        }
    }
}

/// Smallest non-NaN value, or NaN when there is none.
pub fn min_ignoring_nan(values: impl Iterator<Item = f64>) -> f64 {
    values.filter(|v| !v.is_nan()).fold(f64::NAN, f64::min)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern(p: &[bool], n: usize) -> Vec<bool> {
        (0..n).map(|i| p[i % p.len()]).collect()
    }

    #[test]
    fn constant_60_with_every_frame_new() {
        let t = Timeline::constant(60.0);
        let s = PacingStats::new(&vec![true; 600], &t);
        assert_eq!(s.unique, 600);
        assert!((s.avg_fps - 60.0).abs() < 1e-9);
        assert!((s.min_1s_fps - 60.0).abs() < 1e-9);
        assert!((s.low_1pct_p99 - 60.0).abs() < 1e-6);
        assert!((s.low_1pct_avg - 60.0).abs() < 1e-6);
    }

    #[test]
    fn thirty_in_sixty() {
        let t = Timeline::constant(60.0);
        let s = PacingStats::new(&pattern(&[true, false], 600), &t);
        assert!((s.avg_fps - 30.0).abs() < 1e-9);
        assert!((s.min_1s_fps - 30.0).abs() < 1e-9);
        let (fps, ft) = t.fps_series(&pattern(&[true, false], 600));
        assert!(fps[58].is_nan() && fps[59] == 30.0);
        assert!((ft[100] - 33.333).abs() < 0.01);
    }

    #[test]
    fn one_percent_lows_differ_on_rare_hitches() {
        // 1000 frame intervals at 60 fps, 15 of them doubled (one dropped frame each).
        let mut new = vec![true; 1016];
        for k in 0..15 {
            new[50 + k * 60] = false;
        }
        let s = PacingStats::new(&new, &Timeline::constant(60.0));
        // 1.5% of intervals are 33.3 ms: both the p99 and the slowest-1% average see them.
        assert!((s.low_1pct_p99 - 30.0).abs() < 1e-6);
        assert!((s.low_1pct_avg - 30.0).abs() < 1e-6);
        // Only 5 hitches: below the 99th percentile, but still in the slowest 1% average.
        let mut new = vec![true; 1006];
        for k in 0..5 {
            new[50 + k * 60] = false;
        }
        let s = PacingStats::new(&new, &Timeline::constant(60.0));
        assert!((s.low_1pct_p99 - 60.0).abs() < 1e-6);
        assert!(s.low_1pct_avg < 50.0);
    }

    #[test]
    fn millisecond_timestamps_stay_constant() {
        let ts: Vec<f64> = (0..600).map(|i| (i as f64 * 1000.0 / 60.0).round() / 1000.0).collect();
        assert!(!Timeline::from_timestamps(60.0, Some(ts), 600).is_variable());
    }

    #[test]
    fn variable_timestamps_drive_frame_times() {
        // 60 fps with every 30th frame missing from the recording (a 33 ms gap), in
        // millisecond timestamps.
        let ts: Vec<f64> =
            (0..660).filter(|i| i % 30 != 29).map(|i| (i as f64 * 1000.0 / 60.0).round() / 1000.0).collect();
        let n = ts.len();
        let t = Timeline::from_timestamps(60.0, Some(ts), n);
        assert!(t.is_variable());
        let s = PacingStats::new(&vec![true; n], &t);
        assert!((s.avg_fps - 58.0).abs() < 0.2, "{}", s.avg_fps);
        assert!((s.min_1s_fps - 58.0).abs() < 1.01, "{}", s.min_1s_fps);
        let (_, ft) = t.fps_series(&vec![true; n]);
        assert!((ft[29] - 33.333).abs() < 0.01);
        assert!((ft[10] - 16.667).abs() < 0.01, "{}", ft[10]);
        // The gap leaves one output slot without a frame of its own.
        let slots = t.slots(n);
        assert_eq!(slots[28], 28);
        assert_eq!(slots[29], 30);
    }
}
