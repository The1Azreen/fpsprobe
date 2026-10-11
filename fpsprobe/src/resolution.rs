//! EXPERIMENTAL: estimates a game's internal rendering resolution from captured video.
//!
//! Upscaling leaves a periodic trace on the sampling grid of the source image:
//! - integer (nearest-neighbour) scaling only puts edges on multiples of N pixels;
//! - linear-style interpolation makes the second derivative peak at source samples.
//!
//! For each axis we sum a luma difference across the picture, remove the slowly
//! varying content, and look for the periodicity with a periodogram. The period
//! gives the scale factor, and active size / scale the native size. The source
//! type, [`SourceKind`], decides what to look for:
//! - Retro / Interlaced: |first difference| (edges), scale 2..16. Every repeat of
//!   the grid shows up at multiples of 1/scale, so candidates are scored on all of
//!   their harmonics, as in pitch detection, which avoids picking 2x or 1/2x.
//! - Modern: |second difference| (interpolation), scale 1..2. Its frequency (above
//!   Nyquist) aliases to 1 - 1/scale, which is what we search for.
//!
//! Limits: temporal anti-aliasing and reconstruction (TAA, FSR, DLSS) and heavy
//! compression weaken the pattern, and the encoder's own 8/16-pixel block grid
//! can mimic common modern scales; results come with a confidence score.

use crate::progress::Progress;
use crate::video::{self, VideoInfo};
use anyhow::{Context, Result, bail};
use std::io::{BufWriter, Read, Write};
use std::path::Path;
use std::process::Stdio;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SourceKind {
    /// Retro console through an integer (or near-integer) scaler.
    Retro,
    /// 480i/576i console through a deinterlacer (RetroTINK, Morph, ...).
    Interlaced,
    /// Modern game, possibly with dynamic resolution, upscaled by 1x..2x.
    Modern,
}

impl SourceKind {
    pub const ALL: [SourceKind; 3] = [SourceKind::Retro, SourceKind::Interlaced, SourceKind::Modern];

    pub fn label(self) -> &'static str {
        match self {
            SourceKind::Retro => "Retro (integer scaled)",
            SourceKind::Interlaced => "Interlaced (480i via deinterlacer)",
            SourceKind::Modern => "Modern (dynamic resolution)",
        }
    }

    /// Search band in cycles per pixel.
    fn band(self) -> (f64, f64) {
        match self {
            SourceKind::Retro | SourceKind::Interlaced => (1.0 / 16.0, 0.5),
            SourceKind::Modern => (0.02, 0.5),
        }
    }

    /// Retro scalers repeat pixels (edges matter); modern ones interpolate (curvature matters).
    fn second_difference(self) -> bool {
        self == SourceKind::Modern
    }

    /// Native size as a fraction of the measured length, from the detected frequency.
    fn native_fraction(self, f: f64) -> f64 {
        match self {
            SourceKind::Retro | SourceKind::Interlaced => f,
            SourceKind::Modern => 1.0 - f,
        }
    }
}

/// Minimum peak-to-median periodogram ratio to report an estimate at all.
const MIN_CONFIDENCE: f64 = 12.0;
/// Modern detections that coincide with the encoder's block grid (multiples of
/// 1/32 cycles per pixel) need this much more evidence: HEVC/H.264 block edges
/// produce the same periodicity at native resolution.
const CODEC_GRID_FACTOR: f64 = 40.0;

#[derive(Clone, Copy, Debug)]
pub struct AxisEstimate {
    /// Native pixels along this axis for the whole active picture.
    pub native: f64,
    /// Output pixels per native pixel.
    pub scale: f64,
    /// Integer scale factor, when the scale is within a few percent of one.
    pub integer: Option<u32>,
    /// Peak-to-median periodogram ratio (higher is more certain).
    pub confidence: f64,
    /// Phase of the detected grid (radians), used to spot alternating fields.
    phase: f64,
    /// The detected period matches the video encoder's block grid (Modern only).
    pub codec_aligned: bool,
}

impl AxisEstimate {
    pub fn confidence_label(&self) -> &'static str {
        match self.confidence {
            c if c >= 40.0 => "high",
            c if c >= 15.0 => "medium",
            _ => "low",
        }
    }
}

#[derive(Clone, Debug)]
pub struct SampleEstimate {
    pub time: f64,
    pub x: Option<AxisEstimate>,
    pub y: Option<AxisEstimate>,
}

#[derive(Clone, Debug)]
pub struct ResolutionReport {
    pub kind: SourceKind,
    /// Active picture (x, y, w, h) inside the frame, excluding black borders.
    pub active: (usize, usize, usize, usize),
    /// Estimate from all samples combined (best for a fixed resolution).
    pub overall_x: Option<AxisEstimate>,
    pub overall_y: Option<AxisEstimate>,
    pub samples: Vec<SampleEstimate>,
    /// Interlaced mode: share of consecutive frame pairs whose vertical grid shifts
    /// by half a period (fields alternating, as with bob deinterlacing).
    pub field_alternation: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct ResolutionOptions {
    pub kind: SourceKind,
    /// Frames (or frame pairs, for interlaced) analysed per second of video.
    pub samples_per_second: f64,
    /// Regions (x, y, w, h) excluded, e.g. a HUD drawn at output resolution.
    pub ignore: Vec<[usize; 4]>,
    pub hwaccel: Option<String>,
    pub max_samples: Option<usize>,
}

// ---- signal processing ----------------------------------------------------------

/// Sum of |first| or |second difference| of luma along one axis, per position,
/// over a rectangle. `along_x`: profile over columns (detects horizontal scaling).
fn axis_profile(
    luma: &[u8],
    width: usize,
    rect: (usize, usize, usize, usize),
    mask: &[[usize; 4]],
    along_x: bool,
    second: bool,
) -> Vec<f64> {
    let (rx, ry, rw, rh) = rect;
    let len = if along_x { rw } else { rh };
    let mut e = vec![0f64; len];
    if rw < 3 || rh < 3 {
        return e;
    }
    let masked = |x: usize, y: usize| mask.iter().any(|&[mx, my, mw, mh]| x >= mx && x < mx + mw && y >= my && y < my + mh);
    for y in ry..ry + rh {
        let row = &luma[y * width..(y + 1) * width];
        for x in rx..rx + rw {
            let d = if along_x {
                if x == rx || x + 1 == rx + rw {
                    continue;
                }
                if second { row[x - 1] as i32 - 2 * row[x] as i32 + row[x + 1] as i32 } else { row[x] as i32 - row[x - 1] as i32 }
            } else {
                if y == ry || y + 1 == ry + rh {
                    continue;
                }
                let above = luma[(y - 1) * width + x] as i32;
                if second { above - 2 * row[x] as i32 + luma[(y + 1) * width + x] as i32 } else { row[x] as i32 - above }
            };
            if d != 0 && (mask.is_empty() || !masked(x, y)) {
                let i = if along_x { x - rx } else { y - ry };
                e[i] += d.unsigned_abs() as f64;
            }
        }
    }
    e
}

/// Removes slowly varying content (a centred moving average) and zeroes the edges.
fn detrend(e: &[f64]) -> Vec<f64> {
    const HALF: usize = 8;
    let n = e.len();
    let mut prefix = vec![0f64; n + 1];
    for (i, v) in e.iter().enumerate() {
        prefix[i + 1] = prefix[i] + v;
    }
    (0..n)
        .map(|i| {
            if i < HALF + 1 || i + HALF + 1 >= n {
                return 0.0;
            }
            let (a, b) = (i - HALF, i + HALF + 1);
            e[i] - (prefix[b] - prefix[a]) / (b - a) as f64
        })
        .collect()
}

/// Strongest fundamental periodicity of `s` in `band`: (frequency, confidence, phase).
/// Harmonics of a period-p signal sit at multiples of its frequency (folded at
/// Nyquist), so among strong peaks we keep the lowest-frequency one.
fn find_period(s: &[f64], band: (f64, f64)) -> Option<(f64, f64, f64)> {
    let n = s.len();
    let energy: f64 = s.iter().map(|v| v * v).sum();
    if n < 32 || energy <= 0.0 {
        return None;
    }
    let df = 1.0 / (4.0 * n as f64);
    let steps = ((band.1 - band.0) / df).floor() as usize + 1;
    let mut power = Vec::with_capacity(steps);
    let mut phases = Vec::with_capacity(steps);
    for k in 0..steps {
        let f = band.0 + k as f64 * df;
        let (ws, wc) = (-std::f64::consts::TAU * f).sin_cos();
        let (mut zr, mut zi) = (1.0f64, 0.0f64);
        let (mut ar, mut ai) = (0.0f64, 0.0f64);
        for &v in s {
            ar += v * zr;
            ai += v * zi;
            (zr, zi) = (zr * wc - zi * ws, zr * ws + zi * wc);
        }
        // Normalized so white noise averages ~1.
        power.push((ar * ar + ai * ai) / energy);
        phases.push(ai.atan2(ar));
    }
    let mut sorted = power.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2].max(1e-9);
    let max = *sorted.last()?;
    if max / median < MIN_CONFIDENCE {
        return None;
    }
    // Local maxima that are at least a quarter of the strongest peak.
    let peak = (1..power.len().saturating_sub(1))
        .find(|&k| power[k] >= power[k - 1] && power[k] >= power[k + 1] && power[k] >= 0.25 * max && power[k] / median >= MIN_CONFIDENCE)
        .or_else(|| (power[power.len() - 1] >= 0.25 * max).then(|| power.len() - 1))?;
    // Parabolic refinement of the peak position.
    let offset = if peak > 0 && peak + 1 < power.len() {
        let (a, b, c) = (power[peak - 1], power[peak], power[peak + 1]);
        let denom = a - 2.0 * b + c;
        if denom.abs() > 1e-12 { (0.5 * (a - c) / denom).clamp(-0.5, 0.5) } else { 0.0 }
    } else {
        0.0
    };
    let f = band.0 + (peak as f64 + offset) * df;
    Some((f, power[peak] / median, phases[peak]))
}

/// Normalized periodogram value and phase of `s` at frequency `f` (cycles/pixel).
fn power_at(s: &[f64], energy: f64, f: f64) -> (f64, f64) {
    let (ws, wc) = (-std::f64::consts::TAU * f).sin_cos();
    let (mut zr, mut zi) = (1.0f64, 0.0f64);
    let (mut ar, mut ai) = (0.0f64, 0.0f64);
    for &v in s {
        ar += v * zr;
        ai += v * zi;
        (zr, zi) = (zr * wc - zi * ws, zr * ws + zi * wc);
    }
    ((ar * ar + ai * ai) / energy, ai.atan2(ar))
}

/// Retro / interlaced: finds the pixel-repeat period by scoring each candidate on
/// all of its harmonics k/p (k/p <= 0.5). A grid with period p also explains
/// candidates p/2 (every other harmonic) and 2p (half of its harmonics), so the
/// true period scores highest; near-ties go to the longer period, which explains
/// more harmonics. Returns (frequency 1/p, confidence, phase at 1/p).
fn find_scale_comb(s: &[f64]) -> Option<(f64, f64, f64)> {
    let energy: f64 = s.iter().map(|v| v * v).sum();
    if s.len() < 32 || energy <= 0.0 {
        return None;
    }
    let mut cache: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
    let mut power = |f: f64| *cache.entry((f * 1e6).round() as u64).or_insert_with(|| power_at(s, energy, f).0);
    let mut candidates = Vec::new();
    // Periods below 2 px would put the fundamental above Nyquist, where it folds
    // onto another candidate's frequency.
    let mut p = 2.0f64;
    while p <= 16.0 {
        let harmonics = ((0.5 * p).floor() as usize).clamp(1, 8);
        let score = (1..=harmonics).map(|k| power(k as f64 / p)).sum::<f64>() / harmonics as f64;
        candidates.push((p, score));
        p *= 1.002;
    }
    let mut scores: Vec<f64> = candidates.iter().map(|c| c.1).collect();
    scores.sort_by(f64::total_cmp);
    let median = scores[scores.len() / 2].max(1e-9);
    let best = *scores.last()?;
    if best / median < MIN_CONFIDENCE {
        return None;
    }
    let (mut p, score) = candidates.iter().copied().max_by(|a, b| a.1.total_cmp(&b.1))?;
    // The best-scoring period can be a fraction of the true one (its harmonics are
    // a subset). A multiple m*p is real only if the harmonics it adds, those that
    // are not harmonics of p, carry energy too: true for a x9 line-doubled grid
    // seen as x3, not for a x4 grid tested as x8.
    let base = p;
    for m in 2..=(16.0 / base).floor() as usize {
        let q = base * m as f64;
        let added: Vec<f64> = (1..=((0.5 * q).floor() as usize).min(16)).filter(|k| k % m != 0).map(|k| power(k as f64 / q)).collect();
        if !added.is_empty() && added.iter().sum::<f64>() / added.len() as f64 >= 0.25 * score {
            p = q;
        }
    }
    let (_, phase) = power_at(s, energy, 1.0 / p);
    Some((1.0 / p, score / median, phase))
}

/// Whether frequency `f` lies on the encoder block grid (a multiple of 1/32).
fn on_codec_grid(f: f64) -> bool {
    let k = f * 32.0;
    (k - k.round()).abs() < 0.12 && k.round() >= 1.0
}

fn estimate_axis(profile: &[f64], kind: SourceKind, full_len: usize) -> Option<AxisEstimate> {
    let signal = detrend(profile);
    let (f, confidence, phase) = match kind {
        SourceKind::Modern => find_period(&signal, kind.band())?,
        _ => find_scale_comb(&signal)?,
    };
    let codec_aligned = kind == SourceKind::Modern && on_codec_grid(f);
    if codec_aligned && confidence < MIN_CONFIDENCE * CODEC_GRID_FACTOR {
        return None;
    }
    let fraction = kind.native_fraction(f);
    let scale = 1.0 / fraction;
    let rounded = scale.round();
    let integer = (kind != SourceKind::Modern && rounded >= 2.0 && (scale - rounded).abs() < 0.04).then_some(rounded as u32);
    let native = match integer {
        Some(n) => full_len as f64 / n as f64,
        None => full_len as f64 * fraction,
    };
    Some(AxisEstimate { native, scale, integer, confidence, phase, codec_aligned })
}

/// Estimates both axes over `region` of a luma frame. Native sizes are reported for
/// a full picture of `full` (w, h) at the same scale (e.g. the active area).
pub fn analyze_region(
    luma: &[u8],
    width: usize,
    region: (usize, usize, usize, usize),
    full: (usize, usize),
    kind: SourceKind,
) -> (Option<AxisEstimate>, Option<AxisEstimate>) {
    let second = kind.second_difference();
    let x = estimate_axis(&axis_profile(luma, width, region, &[], true, second), kind, full.0);
    let y = estimate_axis(&axis_profile(luma, width, region, &[], false, second), kind, full.1);
    (x, y)
}

/// Bounding box of the picture inside black borders, from per-row/column maxima.
fn active_area(col_max: &[u8], row_max: &[u8]) -> (usize, usize, usize, usize) {
    let black = col_max.iter().chain(row_max).copied().min().unwrap_or(0);
    let threshold = black.saturating_add(12);
    let span = |v: &[u8]| {
        let first = v.iter().position(|&m| m > threshold).unwrap_or(0);
        let last = v.iter().rposition(|&m| m > threshold).unwrap_or(v.len().saturating_sub(1));
        (first, (last + 1).saturating_sub(first).max(1))
    };
    let (x, w) = span(col_max);
    let (y, h) = span(row_max);
    (x, y, w, h)
}

// ---- decoding -------------------------------------------------------------------

/// Full-resolution 8-bit luma of the frame at `time`.
pub fn grab_luma(input: &Path, info: &VideoInfo, time: f64) -> Result<Vec<u8>> {
    let out = video::command("ffmpeg")
        .args(["-v", "error", "-nostdin", "-ss", &format!("{:.3}", time.max(0.0))])
        .arg("-i")
        .arg(input)
        .args(["-an", "-sn", "-frames:v", "1", "-vf", "format=gray", "-f", "rawvideo", "-"])
        .stderr(Stdio::null())
        .output()
        .context("failed to run ffmpeg")?;
    let size = info.width * info.height;
    if out.stdout.len() < size {
        bail!("no frame at {time:.2}s");
    }
    Ok(out.stdout[..size].to_vec())
}

// ---- whole-video analysis -------------------------------------------------------

pub fn analyze(input: &Path, info: &VideoInfo, opts: &ResolutionOptions, progress: &Progress) -> Result<ResolutionReport> {
    let (w, h) = (info.width, info.height);
    let rate = opts.samples_per_second.clamp(0.05, info.fps);
    // Interlaced: pairs of consecutive frames, to compare their field phase.
    let every = (info.fps / rate).round().max(2.0) as usize;
    let filter = match opts.kind {
        SourceKind::Interlaced => format!("select=lt(mod(n\\,{every})\\,2),format=gray"),
        _ => format!("fps={rate},format=gray"),
    };
    let mut cmd = video::command("ffmpeg");
    cmd.args(["-v", "error", "-nostdin"]);
    if let Some(hw) = &opts.hwaccel {
        cmd.args(["-hwaccel", hw]);
    }
    cmd.arg("-i").arg(input).args(["-an", "-sn", "-fps_mode", "passthrough", "-vf", &filter, "-f", "rawvideo", "-"]);
    let mut child = cmd.stdout(Stdio::piped()).stderr(Stdio::null()).spawn().context("failed to run ffmpeg")?;
    let mut stdout = child.stdout.take().unwrap();

    let per_sample = if opts.kind == SourceKind::Interlaced { 2.0 } else { 1.0 };
    let expected = info.duration.map(|d| (d * rate * per_sample) as usize).unwrap_or(0);
    progress.stage("Measuring resolution (experimental)", opts.max_samples.unwrap_or(expected).min(expected.max(1)));

    let mut frames: Vec<(f64, Vec<u8>)> = Vec::new();
    let mut col_max = vec![0u8; w];
    let mut row_max = vec![0u8; h];
    let mut buf = vec![0u8; w * h];
    let mut index = 0usize;
    while stdout.read_exact(&mut buf).is_ok() {
        if progress.is_cancelled() || opts.max_samples.is_some_and(|m| index >= m) {
            break;
        }
        // Ignored regions (e.g. a scaler's on-screen menu) can extend past the
        // picture, so they don't count towards finding its borders either.
        for (y, row) in buf.chunks_exact(w).enumerate() {
            let masked: Vec<(usize, usize)> = opts
                .ignore
                .iter()
                .filter(|&&[_, my, _, mh]| y >= my && y < my + mh)
                .map(|&[mx, _, mw, _]| (mx, mx + mw))
                .collect();
            for (x, &v) in row.iter().enumerate() {
                if masked.iter().any(|&(a, b)| x >= a && x < b) {
                    continue;
                }
                col_max[x] = col_max[x].max(v);
                row_max[y] = row_max[y].max(v);
            }
        }
        let time = match opts.kind {
            SourceKind::Interlaced => ((index / 2) * every + index % 2) as f64 / info.fps,
            _ => index as f64 / rate,
        };
        frames.push((time, buf.clone()));
        index += 1;
        progress.advance();
    }
    let _ = child.kill();
    let _ = child.wait();
    if progress.is_cancelled() {
        bail!("cancelled");
    }
    if frames.is_empty() {
        bail!("no frames decoded");
    }

    let active = active_area(&col_max, &row_max);
    let (_, _, aw, ah) = active;
    let mut sum_x = vec![0f64; aw];
    let mut sum_y = vec![0f64; ah];
    let mut samples = Vec::with_capacity(frames.len());
    for (time, luma) in &frames {
        let second = opts.kind.second_difference();
        let px = axis_profile(luma, w, active, &opts.ignore, true, second);
        let py = axis_profile(luma, w, active, &opts.ignore, false, second);
        sum_x.iter_mut().zip(&px).for_each(|(a, b)| *a += b);
        sum_y.iter_mut().zip(&py).for_each(|(a, b)| *a += b);
        samples.push(SampleEstimate {
            time: *time,
            x: estimate_axis(&px, opts.kind, aw),
            y: estimate_axis(&py, opts.kind, ah),
        });
    }

    // Bob deinterlacing shifts each field by half a line: the vertical grid phase
    // flips by ~pi between consecutive frames.
    let field_alternation = (opts.kind == SourceKind::Interlaced).then(|| {
        let pairs: Vec<bool> = samples
            .chunks_exact(2)
            .filter_map(|p| {
                let (a, b) = (p[0].y?, p[1].y?);
                ((a.scale - b.scale).abs() < 0.05 * a.scale).then(|| {
                    let d = (b.phase - a.phase).rem_euclid(std::f64::consts::TAU);
                    (d - std::f64::consts::PI).abs() < std::f64::consts::FRAC_PI_2
                })
            })
            .collect();
        if pairs.is_empty() { f64::NAN } else { pairs.iter().filter(|&&a| a).count() as f64 / pairs.len() as f64 }
    });

    Ok(ResolutionReport {
        kind: opts.kind,
        active,
        overall_x: estimate_axis(&sum_x, opts.kind, aw),
        overall_y: estimate_axis(&sum_y, opts.kind, ah),
        samples,
        field_alternation: field_alternation.filter(|v| !v.is_nan()),
    })
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

impl ResolutionReport {
    /// Confident per-sample native heights (or widths), sorted.
    fn confident(&self, vertical: bool) -> Vec<f64> {
        let mut v: Vec<f64> = self
            .samples
            .iter()
            .filter_map(|s| if vertical { s.y } else { s.x })
            .filter(|a| a.confidence >= 15.0)
            .map(|a| a.native)
            .collect();
        v.sort_by(f64::total_cmp);
        v
    }

    pub fn summary(&self) -> String {
        let (ax, ay, aw, ah) = self.active;
        let mut out = format!("EXPERIMENTAL resolution estimate ({})\n", self.kind.label());
        out += &format!("active picture   {aw}x{ah} at ({ax},{ay})\n");
        let describe = |a: &Option<AxisEstimate>, name: &str| match a {
            Some(a) => {
                let scale = match a.integer {
                    Some(n) => format!("integer x{n}"),
                    None => format!("x{:.3}", a.scale),
                };
                let codec = if a.codec_aligned { ", matches encoder block grid" } else { "" };
                format!("{name:<16} {:.0} px  (scale {scale}, confidence {} {:.0}{codec})\n", a.native, a.confidence_label(), a.confidence)
            }
            None => format!("{name:<16} no upscaling pattern found (native, or hidden by filtering/compression)\n"),
        };
        match self.kind {
            SourceKind::Modern => {
                for (vertical, name) in [(false, "native width"), (true, "native height")] {
                    let v = self.confident(vertical);
                    let other = self.confident(!vertical);
                    if v.is_empty() && !other.is_empty() {
                        // Dynamic resolution scales both axes together, so derive this
                        // axis from the other one's scale.
                        let (own, theirs) = if vertical { (ah, aw) } else { (aw, ah) };
                        let ratio = own as f64 / theirs as f64;
                        out += &format!(
                            "{name:<16} median ~{:.0} px (assumed from the other axis; not measured confidently)\n",
                            percentile(&other, 0.5) * ratio
                        );
                    } else if v.is_empty() {
                        out += &format!("{name:<16} no confident samples\n");
                    } else {
                        out += &format!(
                            "{name:<16} median {:.0} px, range {:.0}-{:.0} ({} of {} samples confident)\n",
                            percentile(&v, 0.5),
                            v[0],
                            v[v.len() - 1],
                            v.len(),
                            self.samples.len()
                        );
                    }
                }
            }
            SourceKind::Retro => {
                out += &describe(&self.overall_x, "native width");
                out += &describe(&self.overall_y, "native height");
            }
            SourceKind::Interlaced => {
                out += &describe(&self.overall_x, "native width");
                // Per displayed frame (what each deinterlaced frame shows) vs all
                // frames combined (both fields interleaved over time).
                let per_frame = self.confident(true);
                if !per_frame.is_empty() {
                    out += &format!("{:<16} {:.0} lines per displayed frame (median of {} frames)\n", "native height", percentile(&per_frame, 0.5), per_frame.len());
                }
                out += &describe(&self.overall_y, "both fields");
            }
        }
        if let Some(f) = self.field_alternation {
            out += &format!(
                "field alternation {:.0}% of frame pairs{}\n",
                f * 100.0,
                if f > 0.7 { " -> fields are shown one at a time (bob deinterlacing)" } else { "" }
            );
        }
        out
    }

    pub fn write_csv(&self, path: &Path) -> Result<()> {
        let mut w = BufWriter::new(std::fs::File::create(path)?);
        writeln!(w, "time_s,native_width,scale_x,confidence_x,native_height,scale_y,confidence_y")?;
        let cell = |a: &Option<AxisEstimate>| match a {
            Some(a) => format!("{:.1},{:.4},{:.1}", a.native, a.scale, a.confidence),
            None => ",,".into(),
        };
        for s in &self.samples {
            writeln!(w, "{:.3},{},{}", s.time, cell(&s.x), cell(&s.y))?;
        }
        Ok(())
    }
}
