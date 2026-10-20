//! One export job end to end: frame-pacing analysis, optional nits measurement,
//! rendered outputs, the CSV and the summary. Shared by the CLI and the GUI.

use crate::analyze::{self, Analysis, DiffOptions, FrameScore, Geometry};
use crate::cadence::{self, CaptureDrops};
use crate::layout::Layout;
use crate::nits::{self, NitsFrame, NitsGpu};
use crate::progress::Progress;
use crate::render::{self, Compare, Output, RenderJob};
use crate::timing::{self, PacingStats, Timeline};
use crate::video::{self, VideoInfo};
use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct ExportJob {
    pub input: PathBuf,
    pub layout: Layout,
    pub csv: Option<PathBuf>,
    /// Source video with the graphs on top.
    pub video: Option<PathBuf>,
    /// Graphs only (ProRes 4444 with alpha): path, width, height.
    pub graphs: Option<(PathBuf, usize, usize)>,
    /// Machine-readable summary (JSON).
    pub summary_json: Option<PathBuf>,
    /// Measure nits for the CSV even if no nits graph is shown.
    pub nits_in_csv: bool,
    /// FFmpeg hardware decoder, "auto" to pick one, or None for software decoding.
    pub hwaccel: Option<String>,
    pub encoder: Option<String>,
    pub use_gpu: bool,
    pub pixel_difference: u32,
    pub threshold: Option<f64>,
    /// Regions (x, y, w, h in source pixels) excluded from frame comparison.
    pub ignore: Vec<[usize; 4]>,
    pub sdr_white: f32,
    pub max_frames: Option<usize>,
    /// Count duplicates caused by the capture clock (see [`cadence::capture_drops`]) as game frames.
    pub fix_capture_drops: bool,
    /// Experimental tear detection.
    pub tears: bool,
    /// A second capture whose FPS and frame-time lines are drawn beside this one's.
    pub compare: Option<PathBuf>,
}

impl ExportJob {
    pub fn new(input: PathBuf) -> Self {
        Self {
            input,
            layout: Layout::default(),
            csv: None,
            video: None,
            graphs: None,
            summary_json: None,
            nits_in_csv: false,
            hwaccel: Some("auto".into()),
            encoder: None,
            use_gpu: true,
            pixel_difference: 0,
            threshold: None,
            ignore: Vec::new(),
            sdr_white: 100.0,
            max_frames: None,
            fix_capture_drops: true,
            tears: false,
            compare: None,
        }
    }
}

pub struct ExportReport {
    /// Human-readable summary.
    pub summary: String,
    pub stats: RunSummary,
}

/// Everything measured for one video; also written as JSON.
#[derive(Clone, Debug, Serialize)]
pub struct RunSummary {
    pub input: PathBuf,
    pub width: usize,
    pub height: usize,
    pub capture_fps: f64,
    pub color: String,
    pub decoder: Option<String>,
    pub variable_frame_rate: bool,
    pub frames: usize,
    pub analysis_seconds: f64,
    pub threshold: f64,
    pub threshold_source: String,
    /// fpsprobe's result, with capture drops counted as game frames when enabled.
    pub fps: PacingStats,
    /// trdrop's original rule (any pixel differs by more than the margin).
    pub trdrop_rule: PacingStats,
    pub capture_drops: Option<CaptureDropSummary>,
    /// Time where the picture barely changes, so a low frame rate may be a still scene.
    pub low_motion_seconds: f64,
    /// Lowest one-second FPS outside low-motion stretches.
    pub min_1s_fps_with_motion: f64,
    /// Torn frames (experimental), when tear detection ran.
    pub torn_frames: Option<usize>,
    pub nits: Option<NitsSummary>,
    pub compare: Option<Box<RunSummary>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CaptureDropSummary {
    pub frames: usize,
    pub period_seconds: f64,
    /// How far the capture clock is off the source, in percent.
    pub clock_offset_percent: f64,
    pub counted_as_game_frames: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct NitsSummary {
    pub peak_luminance: f32,
    pub median_frame_peak_luminance: f32,
    pub avg_luminance: f64,
    pub max_cll: f32,
    pub max_fall: f32,
}

impl RunSummary {
    pub fn text(&self) -> String {
        let mut out = format!("input: {}x{} @ {:.3} fps, {}\n", self.width, self.height, self.capture_fps, self.color);
        out += &format!("decoder: {}\n", self.decoder.as_deref().unwrap_or("software"));
        if self.variable_frame_rate {
            out += "timing: variable frame rate, using frame timestamps\n";
        }
        out += &format!("analysed {} frames in {:.1}s\n", self.frames, self.analysis_seconds);
        out += &format!("threshold        {:.3} ({})\n", self.threshold, self.threshold_source);
        let (a, c) = (&self.fps, &self.trdrop_rule);
        out += &format!("                 {:>10} {:>14}\n", "fpsprobe", "trdrop-rule");
        out += &format!("unique frames    {:>10} {:>14}   of {}\n", a.unique, c.unique, self.frames);
        out += &format!("average fps      {:>10.2} {:>14.2}\n", a.avg_fps, c.avg_fps);
        out += &format!("min 1s fps       {:>10.2} {:>14.2}\n", a.min_1s_fps, c.min_1s_fps);
        out += &format!(
            "1% low (p99)     {:>10.2} {:>14.2}   99th-percentile frame time\n",
            a.low_1pct_p99, c.low_1pct_p99
        );
        out += &format!(
            "1% low (avg)     {:>10.2} {:>14.2}   average of the slowest 1%\n",
            a.low_1pct_avg, c.low_1pct_avg
        );
        if let Some(d) = &self.capture_drops {
            out += &format!(
                "capture drops    {} repeats, one every {:.2} s: capture clock {:.3}% fast{}; {}\n",
                d.frames,
                d.period_seconds,
                d.clock_offset_percent,
                if (0.08..0.12).contains(&d.clock_offset_percent) { " (like 60 vs 59.94 Hz)" } else { "" },
                if d.counted_as_game_frames { "counted as game frames" } else { "left as drops" },
            );
        }
        if self.low_motion_seconds > 0.0 {
            out += &format!(
                "low motion       {:.1} s barely changing (low FPS there may be a still scene); min 1s fps elsewhere {:.2}\n",
                self.low_motion_seconds, self.min_1s_fps_with_motion
            );
        }
        if let Some(t) = self.torn_frames {
            let pct = 100.0 * t as f64 / a.unique.max(1) as f64;
            out += &format!("torn frames      {t} ({pct:.2}% of new frames, experimental)\n");
        }
        if let Some(n) = &self.nits {
            // Luminance is what the eye sees; MaxCLL uses max(R,G,B), which saturated
            // colours push well above the luminance peak.
            out += &format!(
                "peak luminance   {:>10.0} nits  (brightest frame; median frame {:.0})\n",
                n.peak_luminance, n.median_frame_peak_luminance
            );
            out += &format!("avg luminance    {:>10.0} nits  (whole clip)\n", n.avg_luminance);
            out += &format!("MaxCLL           {:>10.0} nits  (HDR metadata: brightest max(R,G,B) pixel)\n", n.max_cll);
            out += &format!("MaxFALL          {:>10.0} nits  (HDR metadata: brightest frame average)\n", n.max_fall);
        }
        if let Some(b) = &self.compare {
            out += &format!("\ncompare (B, second line on the graphs): {}\n", b.input.display());
            out += &b.text();
        }
        out
    }
}

/// Frame-pacing results for one video.
struct Pacing {
    info: VideoInfo,
    timeline: Timeline,
    scores: Vec<FrameScore>,
    analysis: Analysis,
    /// Final new/duplicate flags (including frame 0), after capture-drop correction.
    is_new: Vec<bool>,
    capture: Option<CaptureDrops>,
    low_motion: Vec<bool>,
    tears: Option<Vec<Option<usize>>>,
    seconds: f64,
}

impl Pacing {
    fn frames(&self) -> usize {
        self.is_new.len()
    }

    fn summary(&self, input: &Path, decoder: Option<String>, fix_capture_drops: bool) -> RunSummary {
        let fps = PacingStats::new(&self.is_new, &self.timeline);
        let compat: Vec<bool> = std::iter::once(true).chain(self.scores.iter().map(|s| s.compat_count > 0)).collect();
        let (series, _) = self.timeline.fps_series(&self.is_new);
        let min_1s_fps_with_motion =
            timing::min_ignoring_nan(series.iter().zip(&self.low_motion).filter(|(_, low)| !**low).map(|(f, _)| *f));
        RunSummary {
            input: input.to_path_buf(),
            width: self.info.width,
            height: self.info.height,
            capture_fps: self.info.fps,
            color: self.info.color.describe(),
            decoder,
            variable_frame_rate: self.timeline.is_variable(),
            frames: self.frames(),
            analysis_seconds: self.seconds,
            threshold: self.analysis.threshold,
            threshold_source: self.analysis.threshold_source.to_string(),
            fps,
            trdrop_rule: PacingStats::new(&compat, &self.timeline),
            capture_drops: self.capture.as_ref().map(|c| CaptureDropSummary {
                frames: c.frames.len(),
                period_seconds: c.period_s,
                clock_offset_percent: 100.0 * c.clock_offset(self.info.fps),
                counted_as_game_frames: fix_capture_drops,
            }),
            low_motion_seconds: self.low_motion.iter().filter(|&&l| l).count() as f64 / self.info.fps,
            min_1s_fps_with_motion,
            torn_frames: self.tears.as_ref().map(|t| t.iter().flatten().count()),
            nits: None,
            compare: None,
        }
    }
}

fn analyse(input: &Path, job: &ExportJob, hwaccel: Option<&str>, progress: &Progress) -> Result<Pacing> {
    let info = video::probe(input)?;
    let start = Instant::now();
    // Timestamps need their own demux pass; running it beside decoding hides its cost.
    let ts_input = input.to_path_buf();
    let timestamps = std::thread::spawn(move || video::frame_times(&ts_input));
    let diff_opts = DiffOptions {
        hwaccel,
        use_gpu: job.use_gpu,
        pixel_difference: job.pixel_difference,
        max_frames: job.max_frames,
        ignore: &job.ignore,
    };
    let (scores, frames) = analyze::run_diff(input, &info, &diff_opts, progress)?;
    // Without timestamps the frame rate is taken as constant, as before.
    let timestamps = timestamps.join().ok().and_then(Result::ok).flatten();
    let timeline = Timeline::from_timestamps(info.fps, timestamps, frames);
    let analysis = analyze::analyze(&scores, job.threshold);
    let capture = cadence::capture_drops(&analysis.is_new, &timeline);
    let mut is_new = analysis.is_new.clone();
    if job.fix_capture_drops {
        for &i in capture.iter().flat_map(|c| &c.frames) {
            is_new[i] = true;
        }
    }
    let low_motion = analyze::low_motion(&scores, analysis.threshold, info.fps);
    let tears = job.tears.then(|| cadence::tears(&scores, analysis.threshold));
    Ok(Pacing {
        info,
        timeline,
        scores,
        analysis,
        is_new,
        capture,
        low_motion,
        tears,
        seconds: start.elapsed().as_secs_f64(),
    })
}

/// `series` (one value per frame of `b`) sampled at the times of `a`'s frames;
/// NaN once `b` has ended.
fn resample(a: &Timeline, a_frames: usize, b: &Timeline, series: &[f64]) -> Vec<f64> {
    let end = b.duration(series.len());
    let mut j = 0;
    (0..a_frames)
        .map(|i| {
            let t = a.time(i);
            while j + 1 < series.len() && b.time(j + 1) <= t + 1e-6 {
                j += 1;
            }
            if t < end { series[j] } else { f64::NAN }
        })
        .collect()
}

pub fn run(job: &ExportJob, progress: &Progress) -> Result<ExportReport> {
    let hwaccel = video::resolve_hwaccel(job.hwaccel.as_deref(), &job.input);
    let pacing = analyse(&job.input, job, hwaccel.as_deref(), progress)?;
    let mut stats = pacing.summary(&job.input, hwaccel.clone(), job.fix_capture_drops);
    let info = &pacing.info;
    let frames = pacing.frames();

    let compare = match &job.compare {
        Some(path) => {
            let hw = video::resolve_hwaccel(job.hwaccel.as_deref(), path);
            let b = analyse(path, job, hw.as_deref(), progress).context("analysing the comparison video")?;
            stats.compare = Some(Box::new(b.summary(path, hw, job.fix_capture_drops)));
            let (fps, ft) = b.timeline.fps_series(&b.is_new);
            Some((
                resample(&pacing.timeline, frames, &b.timeline, &fps),
                resample(&pacing.timeline, frames, &b.timeline, &ft),
            ))
        }
        None => None,
    };

    let mut outputs = Vec::new();
    if let Some(path) = &job.video {
        outputs.push(Output::Overlay { path: path.clone(), encoder: job.encoder.clone() });
    }
    if let Some((path, width, height)) = &job.graphs {
        outputs.push(Output::GraphsOnly { path: path.clone(), width: *width, height: *height });
    }
    let mut render_seconds = None;
    let nits_frames = if !outputs.is_empty() {
        let start = Instant::now();
        let (fps, frametime_ms) = pacing.timeline.fps_series(&pacing.is_new);
        let rj = RenderJob {
            input: &job.input,
            info,
            layout: &job.layout,
            timeline: &pacing.timeline,
            fps: &fps,
            frametime_ms: &frametime_ms,
            compare: compare.as_ref().map(|(f, t)| Compare { fps: f, frametime_ms: t }),
            outputs: &outputs,
            hwaccel: hwaccel.as_deref(),
            sdr_white: job.sdr_white,
        };
        let n = render::render(&rj, progress)?;
        render_seconds = Some(start.elapsed().as_secs_f64());
        match n {
            Some(n) => Some(n),
            None if job.nits_in_csv && job.csv.is_some() => {
                Some(measure_nits(job, info, hwaccel.as_deref(), frames, progress)?)
            }
            None => None,
        }
    } else if job.nits_in_csv || job.layout.needs_nits() {
        Some(measure_nits(job, info, hwaccel.as_deref(), frames, progress)?)
    } else {
        None
    };

    if let Some(n) = nits_frames.as_deref().filter(|n| !n.is_empty()) {
        let (max_cll, max_fall) = nits::content_light_levels(n);
        let mut peaks: Vec<f32> = n.iter().map(|f| f.peak_lum).collect();
        peaks.sort_by(f32::total_cmp);
        let pct = |q: f64| peaks[((peaks.len() - 1) as f64 * q).round() as usize];
        stats.nits = Some(NitsSummary {
            peak_luminance: pct(1.0),
            median_frame_peak_luminance: pct(0.5),
            avg_luminance: n.iter().map(|f| f.avg_lum as f64).sum::<f64>() / n.len() as f64,
            max_cll,
            max_fall,
        });
    }

    let mut summary = stats.text();
    if let Some(s) = render_seconds {
        summary += &format!("rendered in {s:.1}s\n");
    }
    if let Some(path) = &job.csv {
        write_csv(path, &pacing, nits_frames.as_deref())?;
        summary += &format!("CSV written to {}\n", path.display());
    }
    if let Some(path) = &job.summary_json {
        std::fs::write(path, serde_json::to_string_pretty(&stats)?)?;
        summary += &format!("summary written to {}\n", path.display());
    }
    Ok(ExportReport { summary, stats })
}

fn write_csv(path: &Path, p: &Pacing, nits_frames: Option<&[NitsFrame]>) -> Result<()> {
    let geo = Geometry::for_video(&p.info);
    let mut w = BufWriter::new(std::fs::File::create(path)?);
    write!(w, "frame,time_s,score,max_tile_x,max_tile_y,compat_count,new,compat_new,capture_drop,low_motion")?;
    if p.tears.is_some() {
        write!(w, ",tear_y")?;
    }
    if nits_frames.is_some() {
        write!(w, ",peak_nits,avg_nits,peak_luminance_nits,avg_luminance_nits")?;
    }
    writeln!(w)?;
    let capture: Vec<usize> = p.capture.iter().flat_map(|c| c.frames.iter().copied()).collect();
    for i in 0..p.frames() {
        write!(w, "{i},{:.6},", p.timeline.time(i))?;
        match i.checked_sub(1) {
            None => write!(w, ",,,,1,1")?,
            Some(j) => {
                let s = &p.scores[j];
                write!(
                    w,
                    "{:.4},{},{},{},{},{}",
                    s.score,
                    s.max_tile % geo.tiles_x(),
                    s.max_tile / geo.tiles_x(),
                    s.compat_count,
                    p.is_new[i] as u8,
                    (s.compat_count > 0) as u8
                )?;
            }
        }
        write!(w, ",{},{}", capture.binary_search(&i).is_ok() as u8, p.low_motion[i] as u8)?;
        if let Some(t) = &p.tears {
            write!(w, ",{}", t[i].map(|y| y.to_string()).unwrap_or_default())?;
        }
        if let Some(f) = nits_frames.and_then(|n| n.get(i)) {
            write!(w, ",{:.2},{:.2},{:.2},{:.2}", f.peak, f.avg, f.peak_lum, f.avg_lum)?;
        }
        writeln!(w)?;
    }
    Ok(())
}

/// Nits measurement pass without rendering.
fn measure_nits(
    job: &ExportJob,
    info: &VideoInfo,
    hwaccel: Option<&str>,
    max_frames: usize,
    progress: &Progress,
) -> Result<Vec<NitsFrame>> {
    let (w, h) = (info.width / 4 * 4, info.height / 2 * 2);
    let mut gpu = NitsGpu::new(w, h, info.color, job.sdr_white, 0)?;
    let (rx, mut child) = video::spawn_decoder(&job.input, hwaccel, (w, h), "yuv420p10le", gpu.frame_bytes())?;
    progress.stage("Measuring nits", max_frames);
    let mut out = Vec::new();
    for frame in rx.into_iter().take(max_frames) {
        if progress.is_cancelled() {
            break;
        }
        out.extend(gpu.push(&frame)?);
        progress.advance();
    }
    out.extend(gpu.finish()?);
    let _ = child.kill();
    let _ = child.wait();
    if progress.is_cancelled() {
        bail!("cancelled");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_follows_time_and_ends_with_b() {
        let a = Timeline::constant(60.0);
        let b = Timeline::constant(30.0);
        let series: Vec<f64> = (0..30).map(|i| i as f64).collect(); // one second of B
        let r = resample(&a, 120, &b, &series);
        assert_eq!((r[0], r[1], r[2], r[59]), (0.0, 0.0, 1.0, 29.0));
        assert!(r[60].is_nan());
    }
}
