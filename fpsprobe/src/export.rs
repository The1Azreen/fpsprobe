//! One export job end to end: frame-pacing analysis, optional nits measurement,
//! rendered outputs and the CSV. Shared by the CLI and the GUI.

use crate::analyze::{self, DiffOptions, Geometry};
use crate::layout::Layout;
use crate::nits::{self, NitsFrame, NitsGpu};
use crate::progress::Progress;
use crate::render::{self, Output, RenderJob};
use crate::video;
use anyhow::{Result, bail};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
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
    /// Measure nits for the CSV even if no nits graph is shown.
    pub nits_in_csv: bool,
    pub hwaccel: Option<String>,
    pub encoder: Option<String>,
    pub use_gpu: bool,
    pub pixel_difference: u32,
    pub threshold: Option<f64>,
    /// Regions (x, y, w, h in source pixels) excluded from frame comparison.
    pub ignore: Vec<[usize; 4]>,
    pub sdr_white: f32,
    pub max_frames: Option<usize>,
}

impl ExportJob {
    pub fn new(input: PathBuf) -> Self {
        Self {
            input,
            layout: Layout::default(),
            csv: None,
            video: None,
            graphs: None,
            nits_in_csv: false,
            hwaccel: Some("d3d11va".into()),
            encoder: None,
            use_gpu: true,
            pixel_difference: 0,
            threshold: None,
            ignore: Vec::new(),
            sdr_white: 100.0,
            max_frames: None,
        }
    }
}

pub struct ExportReport {
    pub summary: String,
}

pub fn run(job: &ExportJob, progress: &Progress) -> Result<ExportReport> {
    let info = video::probe(&job.input)?;
    let hwaccel = job.hwaccel.as_deref();
    let mut summary = format!(
        "input: {}x{} @ {:.3} fps, {}\n",
        info.width,
        info.height,
        info.fps,
        info.color.describe()
    );

    let start = Instant::now();
    let diff_opts =
        DiffOptions { hwaccel, use_gpu: job.use_gpu, pixel_difference: job.pixel_difference, max_frames: job.max_frames };
    let (stats, frames) = analyze::run_diff(&job.input, &info, &diff_opts, progress)?;
    summary += &format!("analysed {frames} frames in {:.1}s\n", start.elapsed().as_secs_f64());
    let geo = Geometry::for_video(&info);
    let ignored = geo.ignored_tiles(&job.ignore);
    let result = analyze::analyze(geo, &stats, &ignored, job.threshold);
    summary += &analyze::summary_text(&result, info.fps, &stats);
    let is_new: Vec<bool> = std::iter::once(true).chain(result.frames.iter().map(|f| f.is_new)).collect();

    let mut outputs = Vec::new();
    if let Some(path) = &job.video {
        outputs.push(Output::Overlay { path: path.clone(), encoder: job.encoder.clone() });
    }
    if let Some((path, width, height)) = &job.graphs {
        outputs.push(Output::GraphsOnly { path: path.clone(), width: *width, height: *height });
    }
    let nits_frames = if !outputs.is_empty() {
        let start = Instant::now();
        let rj = RenderJob {
            input: &job.input,
            info: &info,
            layout: &job.layout,
            is_new: &is_new,
            outputs: &outputs,
            hwaccel,
            sdr_white: job.sdr_white,
        };
        let n = render::render(&rj, progress)?;
        summary += &format!("rendered in {:.1}s\n", start.elapsed().as_secs_f64());
        match n {
            Some(n) => Some(n),
            None if job.nits_in_csv && job.csv.is_some() => Some(measure_nits(job, &info, frames, progress)?),
            None => None,
        }
    } else if job.nits_in_csv || job.layout.needs_nits() {
        Some(measure_nits(job, &info, frames, progress)?)
    } else {
        None
    };

    if let Some(n) = nits_frames.as_deref().filter(|n| !n.is_empty()) {
        // Luminance is what the eye sees; MaxCLL uses max(R,G,B), which saturated
        // colours push well above the luminance peak.
        let (max_cll, max_fall) = nits::content_light_levels(n);
        let mut peaks: Vec<f32> = n.iter().map(|f| f.peak_lum).collect();
        peaks.sort_by(f32::total_cmp);
        let pct = |q: f64| peaks[((peaks.len() - 1) as f64 * q).round() as usize];
        let avg_lum = n.iter().map(|f| f.avg_lum as f64).sum::<f64>() / n.len() as f64;
        summary += &format!("peak luminance   {:>10.0} nits  (brightest frame; median frame {:.0})\n", pct(1.0), pct(0.5));
        summary += &format!("avg luminance    {avg_lum:>10.0} nits  (whole clip)\n");
        summary += &format!("MaxCLL           {max_cll:>10.0} nits  (HDR metadata: brightest max(R,G,B) pixel)\n");
        summary += &format!("MaxFALL          {max_fall:>10.0} nits  (HDR metadata: brightest frame average)\n");
    }

    if let Some(path) = &job.csv {
        let mut w = BufWriter::new(std::fs::File::create(path)?);
        write!(w, "frame,time_s,score,max_tile_x,max_tile_y,compat_count,new,compat_new")?;
        if nits_frames.is_some() {
            write!(w, ",peak_nits,avg_nits,peak_luminance_nits,avg_luminance_nits")?;
        }
        writeln!(w)?;
        for i in 0..is_new.len() {
            write!(w, "{i},{:.6},", i as f64 / info.fps)?;
            match i.checked_sub(1) {
                None => write!(w, ",,,,1,1")?,
                Some(j) => {
                    let (s, f) = (&stats[j], &result.frames[j]);
                    write!(
                        w,
                        "{:.4},{},{},{},{},{}",
                        f.score,
                        f.max_tile % geo.tiles_x(),
                        f.max_tile / geo.tiles_x(),
                        s.compat_count,
                        f.is_new as u8,
                        (s.compat_count > 0) as u8
                    )?;
                }
            }
            if let Some(f) = nits_frames.as_ref().and_then(|n| n.get(i)) {
                write!(w, ",{:.2},{:.2},{:.2},{:.2}", f.peak, f.avg, f.peak_lum, f.avg_lum)?;
            }
            writeln!(w)?;
        }
        summary += &format!("CSV written to {}\n", path.display());
    }
    Ok(ExportReport { summary })
}

/// Nits measurement pass without rendering.
fn measure_nits(job: &ExportJob, info: &video::VideoInfo, max_frames: usize, progress: &Progress) -> Result<Vec<NitsFrame>> {
    let (w, h) = (info.width / 4 * 4, info.height / 2 * 2);
    let mut gpu = NitsGpu::new(w, h, info.color, job.sdr_white, 0)?;
    let (rx, mut child) = video::spawn_decoder(&job.input, job.hwaccel.as_deref(), (w, h), "yuv420p10le", gpu.frame_bytes())?;
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
