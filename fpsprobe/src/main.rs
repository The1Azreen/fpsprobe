//! fpsprobe command line: analyse a captured video and export CSV / overlay video / graphs.

use anyhow::Result;
use clap::{Parser, ValueEnum};
use fpsprobe::export::{self, ExportJob};
use fpsprobe::layout::Layout;
use fpsprobe::progress::Progress;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Backend {
    Gpu,
    Cpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Hwaccel {
    D3d11va,
    Cuda,
    Qsv,
    Vulkan,
    None,
}

#[derive(Parser, Debug)]
#[command(about = "GPU-accelerated FPS / frame-time and HDR analysis for captured video")]
struct Args {
    /// Input video file
    input: PathBuf,
    /// Write per-frame results to this CSV file
    #[arg(long)]
    csv: Option<PathBuf>,
    /// Render the video with the graphs on top to this file (e.g. out.mp4)
    #[arg(long)]
    render: Option<PathBuf>,
    /// Render the graphs alone on a transparent background (ProRes 4444 .mov) for video editors
    #[arg(long)]
    graphs: Option<PathBuf>,
    /// Size of the --graphs canvas, WIDTHxHEIGHT
    #[arg(long, default_value = "1920x1080", value_parser = parse_size)]
    graphs_size: (usize, usize),
    /// Overlay layout saved from the GUI (default: FPS graphs, plus nits graphs with --nits)
    #[arg(long)]
    layout: Option<PathBuf>,
    /// Measure brightness in nits (HDR10/PQ, HLG or SDR): MaxCLL/MaxFALL and CSV columns
    #[arg(long)]
    nits: bool,
    /// Nits assumed for SDR reference white when measuring SDR video
    #[arg(long, default_value_t = 100.0)]
    sdr_white: f32,
    /// Where to compute frame differences
    #[arg(long, value_enum, default_value_t = Backend::Gpu)]
    backend: Backend,
    /// FFmpeg hardware decoder
    #[arg(long, value_enum, default_value_t = Hwaccel::D3d11va)]
    hwaccel: Hwaccel,
    /// FFmpeg video encoder for --render (default: first working of hevc_nvenc, hevc_amf, hevc_qsv, libx264)
    #[arg(long)]
    encoder: Option<String>,
    /// trdrop-compatible pixel difference margin (0-255) for the `compat` column
    #[arg(long, default_value_t = 0)]
    pixel_difference: u32,
    /// Force the duplicate threshold (mean abs luma diff of the most-changed tile)
    /// instead of estimating it from the video's noise floor
    #[arg(long)]
    threshold: Option<f64>,
    /// Ignore a region (HUD, clock, overlay) as x,y,w,h in pixels; repeatable.
    /// Any 64x64 tile touching the region is excluded from scoring.
    #[arg(long = "ignore", value_parser = parse_rect)]
    ignore: Vec<[usize; 4]>,
    /// Stop after this many frames
    #[arg(long)]
    max_frames: Option<usize>,
}

fn parse_rect(s: &str) -> Result<[usize; 4], String> {
    let v: Vec<usize> = s.split(',').map(|p| p.trim().parse()).collect::<Result<_, _>>().map_err(|e| format!("{e}"))?;
    v.try_into().map_err(|_| "expected x,y,w,h".to_string())
}

fn parse_size(s: &str) -> Result<(usize, usize), String> {
    let (w, h) = s.split_once(['x', 'X']).ok_or("expected WIDTHxHEIGHT")?;
    Ok((w.trim().parse().map_err(|e| format!("{e}"))?, h.trim().parse().map_err(|e| format!("{e}"))?))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let layout = match &args.layout {
        Some(path) => Layout::load(path)?,
        None if args.nits => Layout::preset_fps_hdr(),
        None => Layout::preset_fps(),
    };
    let job = ExportJob {
        layout,
        csv: args.csv,
        video: args.render,
        graphs: args.graphs.map(|p| (p, args.graphs_size.0, args.graphs_size.1)),
        nits_in_csv: args.nits,
        hwaccel: match args.hwaccel {
            Hwaccel::D3d11va => Some("d3d11va".into()),
            Hwaccel::Cuda => Some("cuda".into()),
            Hwaccel::Qsv => Some("qsv".into()),
            Hwaccel::Vulkan => Some("vulkan".into()),
            Hwaccel::None => None,
        },
        encoder: args.encoder,
        use_gpu: args.backend == Backend::Gpu,
        pixel_difference: args.pixel_difference,
        threshold: args.threshold,
        ignore: args.ignore,
        sdr_white: args.sdr_white,
        max_frames: args.max_frames,
        ..ExportJob::new(args.input)
    };

    // Report each stage as it starts.
    let progress = Arc::new(Progress::default());
    let watcher = {
        let progress = progress.clone();
        std::thread::spawn(move || {
            let mut last = String::new();
            while Arc::strong_count(&progress) > 1 {
                let (stage, _, _) = progress.snapshot();
                if stage != last && !stage.is_empty() {
                    eprintln!("{stage}...");
                    last = stage;
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        })
    };
    let report = export::run(&job, &progress);
    drop(progress);
    let _ = watcher.join();
    print!("{}", report?.summary);
    Ok(())
}
