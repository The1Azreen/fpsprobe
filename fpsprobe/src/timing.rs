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

        let (low_1pct_p99, low_1pct_avg) = one_percent_lows(is_new, timeline, |_, _| true);
        Self {
            unique,
            avg_fps: unique as f64 / timeline.duration(is_new.len()),
            min_1s_fps,
            low_1pct_p99,
            low_1pct_avg,
        }
    }
}

/// The two 1% lows (p99 and average of the slowest 1%) over the frame times
/// between consecutive new frames, keeping only intervals for which `keep(first,
/// last)` holds. NaN when no interval is kept.
pub fn one_percent_lows(is_new: &[bool], timeline: &Timeline, keep: impl Fn(usize, usize) -> bool) -> (f64, f64) {
    let idx: Vec<usize> = (0..is_new.len()).filter(|&i| is_new[i]).collect();
    let mut ft: Vec<f64> =
        idx.windows(2).filter(|w| keep(w[0], w[1])).map(|w| timeline.time(w[1]) - timeline.time(w[0])).collect();
    if ft.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    ft.sort_by(f64::total_cmp);
    let p99 = ft[((ft.len() as f64 * 0.99).ceil() as usize).clamp(1, ft.len()) - 1];
    let worst = &ft[ft.len() - (ft.len() as f64 * 0.01).ceil().max(1.0) as usize..];
    (1.0 / p99, worst.len() as f64 / worst.iter().sum::<f64>())
}

/// A frame that came later than the target frame rate allows.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Hitch {
    /// The frame shown too long, and when it appeared (seconds from the start).
    pub frame: usize,
    pub time_s: f64,
    /// How long it stayed on screen.
    pub frame_time_ms: f64,
}

/// How well a game holds the frame rate it runs at most of the time.
#[derive(Clone, Debug, serde::Serialize)]
pub struct TargetPacing {
    /// The capture rate divided by the most common number of captured frames per
    /// game frame: 30 for a game locked to 30 in a 60 Hz capture.
    pub fps: f64,
    /// Game frames considered.
    pub frames: usize,
    /// Frames that stayed on screen at least half a capture frame longer than the target allows.
    pub late_frames: usize,
    pub on_time_percent: f64,
    /// The longest late frames, longest first.
    pub worst: Vec<Hitch>,
}

/// Finds the target frame rate from the median frame time and measures the frames
/// that missed it, over the intervals between new frames for which `keep(first,
/// last)` holds. None when no interval is kept.
pub fn target_pacing(
    is_new: &[bool],
    timeline: &Timeline,
    keep: impl Fn(usize, usize) -> bool,
    worst: usize,
) -> Option<TargetPacing> {
    let idx: Vec<usize> = (0..is_new.len()).filter(|&i| is_new[i]).collect();
    let mut intervals: Vec<(usize, f64)> = idx
        .windows(2)
        .filter(|w| keep(w[0], w[1]))
        .map(|w| (w[0], timeline.time(w[1]) - timeline.time(w[0])))
        .collect();
    if intervals.is_empty() {
        return None;
    }
    let mut sorted: Vec<f64> = intervals.iter().map(|i| i.1).collect();
    sorted.sort_by(f64::total_cmp);
    let capture = timeline.fps;
    let per_frame = (sorted[sorted.len() / 2] * capture).round().max(1.0);
    let late_after = (per_frame + 0.5) / capture;

    intervals.retain(|i| i.1 > late_after);
    let late_frames = intervals.len();
    // Longest first; equal ones in time order.
    intervals.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    Some(TargetPacing {
        fps: capture / per_frame,
        frames: sorted.len(),
        late_frames,
        on_time_percent: 100.0 * (1.0 - late_frames as f64 / sorted.len() as f64),
        worst: intervals
            .iter()
            .take(worst)
            .map(|&(frame, dt)| Hitch { frame, time_s: timeline.time(frame), frame_time_ms: dt * 1000.0 })
            .collect(),
    })
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

    #[test]
    fn locked_thirty_with_two_hitches() {
        // 30 in 60, except frames 398 and 798 stay up for 4 captured frames (66.7 ms).
        let mut new = pattern(&[true, false], 1200);
        new[400] = false;
        new[800] = false;
        let t = Timeline::constant(60.0);
        let p = target_pacing(&new, &t, |_, _| true, 10).unwrap();
        assert!((p.fps - 30.0).abs() < 1e-9);
        assert_eq!(p.late_frames, 2);
        assert_eq!((p.worst[0].frame, p.worst[1].frame), (398, 798));
        assert!((p.worst[0].frame_time_ms - 66.667).abs() < 0.01);
        assert!((p.worst[0].time_s - 398.0 / 60.0).abs() < 1e-9);
        assert!((p.on_time_percent - 100.0 * (1.0 - 2.0 / 597.0)).abs() < 1e-9);
        // Every frame new: a target of 60 with nothing late.
        let t60 = target_pacing(&pattern(&[true], 600), &t, |_, _| true, 10).unwrap();
        assert_eq!((t60.fps, t60.late_frames), (60.0, 0));
        // Nothing kept: nothing to report.
        assert!(target_pacing(&new, &t, |_, _| false, 10).is_none());
    }
}
