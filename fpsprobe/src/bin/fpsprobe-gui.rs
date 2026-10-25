//! fpsprobe GUI: arrange the overlay graphs on a live preview, then export
//! CSV, the video with graphs, and/or the graphs alone for a video editor.

#![windows_subsystem = "windows"]

use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};
use fpsprobe::canvas::{Canvas, FrameData};
use fpsprobe::export::{self, ExportJob, ExportReport};
use fpsprobe::layout::{Layout, RectPx, Snap, Widget, snap_move, snap_resize};
use fpsprobe::nits::{HIST_BINS, NitsFrame, ScopeData, WAVE_COLS, WAVE_ROWS, XY_BINS};
use fpsprobe::preview::FrameGrabber;
use fpsprobe::progress::Progress;
use fpsprobe::resolution::{self, AxisEstimate, ResolutionOptions, ResolutionReport, SourceKind};
use fpsprobe::timing::Timeline;
use fpsprobe::video::{self, VideoInfo};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;

/// Longest side of the preview image; the layout is resolution independent.
const PREVIEW_MAX_W: usize = 1280;
const GRAPH_SIZES: [(&str, (usize, usize)); 4] =
    [("4K", (3840, 2160)), ("1440p", (2560, 1440)), ("1080p", (1920, 1080)), ("720p", (1280, 720))];
/// Encoders for the video with graphs; "auto" picks a hardware one and falls back to software if it crashes.
const ENCODERS: [&str; 6] = ["auto", "hevc_amf", "hevc_nvenc", "hevc_qsv", "libx265", "libx264"];
const HWACCELS: [&str; 8] = ["auto", "d3d11va", "cuda", "qsv", "vulkan", "vaapi", "videotoolbox", "none"];

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1560.0, 920.0]).with_title("fpsprobe"),
        ..Default::default()
    };
    eframe::run_native("fpsprobe", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}

/// Everything remembered between sessions.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
struct Settings {
    layout: Layout,
    export_csv: bool,
    csv_nits: bool,
    export_video: bool,
    export_graphs: bool,
    graphs_size: (usize, usize),
    hwaccel: String,
    sdr_white: f32,
    auto_threshold: bool,
    threshold: f64,
    ignore: String,
    preview_target: PreviewTarget,
    output_dir: Option<PathBuf>,
    res_kind: SourceKind,
    res_rate: f64,
    /// Show the experimental internal-resolution tools (off by default).
    show_experimental: bool,
    /// Snap to the grid while dragging without holding Shift (Shift then turns it off).
    snap_always: bool,
    /// Grid spacing in canvas pixels.
    grid_px: u32,
    /// Prevents moving or resizing any graph.
    lock_layout: bool,
    /// Count periodic capture-clock repeats as game frames.
    fix_capture_drops: bool,
    /// Experimental tear detection.
    detect_tears: bool,
    export_summary: bool,
    /// Second capture drawn beside this one on the FPS and frame-time graphs.
    compare: Option<PathBuf>,
    /// Encoder for the video with graphs (see ENCODERS).
    encoder: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            layout: Layout::preset_fps(),
            export_csv: true,
            csv_nits: false,
            export_video: true,
            export_graphs: false,
            graphs_size: (1920, 1080),
            hwaccel: "auto".into(),
            sdr_white: 100.0,
            auto_threshold: true,
            threshold: 1.5,
            ignore: String::new(),
            preview_target: PreviewTarget::Video,
            output_dir: None,
            res_kind: SourceKind::Modern,
            res_rate: 2.0,
            show_experimental: false,
            snap_always: false,
            grid_px: 16,
            lock_layout: false,
            fix_capture_drops: true,
            detect_tears: false,
            export_summary: false,
            compare: None,
            encoder: "auto".into(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum PreviewTarget {
    /// Overlay on the source video (canvas = video resolution).
    Video,
    /// Graphs-only export (canvas = chosen export resolution).
    Graphs,
}

#[derive(Clone, Copy, PartialEq)]
enum DragMode {
    Move,
    Resize,
}

struct Drag {
    index: usize,
    mode: DragMode,
    start_pointer: Pos2,
    start: (f32, f32, f32, f32),
}

struct RunningJob {
    progress: Arc<Progress>,
    handle: JoinHandle<anyhow::Result<ExportReport>>,
}

struct App {
    settings: Settings,
    input: Option<PathBuf>,
    info: Option<VideoInfo>,
    status: String,
    selected: Option<usize>,
    drag: Option<Drag>,
    demo: Demo,
    demo_frame: usize,
    animate: bool,
    canvas: Option<(usize, usize, Canvas)>,
    texture: Option<egui::TextureHandle>,
    layout_changed: bool,
    redraw: bool,
    job: Option<RunningJob>,
    report: String,
    /// Real video frame shown behind the graphs.
    grabber: FrameGrabber,
    video_tex: Option<egui::TextureHandle>,
    video_time: f64,
    video_error: Option<String>,
    /// A frame was requested from the grabber and hasn't arrived yet.
    frame_pending: bool,
    // Experimental resolution measurement.
    res_job: Option<(Arc<Progress>, JoinHandle<anyhow::Result<ResolutionReport>>)>,
    res_report: Option<ResolutionReport>,
    res_error: Option<String>,
    /// Pixel counter: drags in the preview select a region instead of moving graphs.
    counter_mode: bool,
    counter_drag: Option<Pos2>,
    /// Selected region as fractions of the frame.
    counter_sel: Option<Rect>,
    counter: Option<Counter>,
    /// Alignment lines the dragged graph locked onto (canvas px), drawn while dragging.
    guides: (Vec<f32>, Vec<f32>),
}

/// Result of the pixel counter on one region of the current frame.
struct Counter {
    tex: egui::TextureHandle,
    region: (usize, usize, usize, usize),
    /// Full picture the estimate refers to (active area if measured, else the frame).
    full: (usize, usize),
    x: Option<AxisEstimate>,
    y: Option<AxisEstimate>,
    steps: f64,
    run: f64,
    vertical: bool,
    open: bool,
}

impl App {
    fn new(cc: &eframe::CreationContext) -> Self {
        cc.egui_ctx.set_theme(egui::Theme::Dark);
        let settings: Settings = cc.storage.and_then(|s| eframe::get_value(s, eframe::APP_KEY)).unwrap_or_default();
        let mut app = Self {
            settings,
            input: None,
            info: None,
            status: "Choose a video to analyse.".into(),
            selected: None,
            drag: None,
            demo: Demo::new(),
            demo_frame: 420,
            animate: false,
            canvas: None,
            texture: None,
            layout_changed: true,
            redraw: true,
            job: None,
            report: String::new(),
            grabber: FrameGrabber::default(),
            video_tex: None,
            video_time: 0.0,
            video_error: None,
            frame_pending: false,
            res_job: None,
            res_report: None,
            res_error: None,
            counter_mode: false,
            counter_drag: None,
            counter_sel: None,
            counter: None,
            guides: (Vec::new(), Vec::new()),
        };
        // A video passed on the command line (or via "Open with") is opened directly.
        if let Some(path) = std::env::args_os().nth(1).map(PathBuf::from) {
            app.open_video(path);
        }
        app
    }

    /// Canvas size the layout is being designed for.
    fn target_size(&self) -> (usize, usize) {
        match self.settings.preview_target {
            PreviewTarget::Video => self.info.as_ref().map_or((1920, 1080), |i| (i.width, i.height)),
            PreviewTarget::Graphs => self.settings.graphs_size,
        }
    }

    fn open_video(&mut self, path: PathBuf) {
        match video::probe(&path) {
            Ok(info) => {
                self.status = format!(
                    "{}x{} @ {:.2} fps, {}{}",
                    info.width,
                    info.height,
                    info.fps,
                    info.color.describe(),
                    info.duration.map(|d| format!(", {d:.1} s")).unwrap_or_default()
                );
                self.video_time = info.duration.unwrap_or(0.0) * 0.25;
                self.info = Some(info);
                self.input = Some(path);
                self.layout_changed = true;
                self.video_tex = None;
                self.request_frame();
            }
            Err(e) => self.status = format!("Could not read video: {e:#}"),
        }
    }

    fn start_export(&mut self) {
        let Some(input) = self.input.clone() else { return };
        let s = &self.settings;
        let dir = s.output_dir.clone().or_else(|| input.parent().map(PathBuf::from)).unwrap_or_default();
        let stem = input.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "output".into());
        let mut ignore = Vec::new();
        for part in s.ignore.split(';').map(str::trim).filter(|p| !p.is_empty()) {
            let v: Result<Vec<usize>, _> = part.split(',').map(|n| n.trim().parse()).collect();
            match v.ok().and_then(|v| <[usize; 4]>::try_from(v).ok()) {
                Some(r) => ignore.push(r),
                None => {
                    self.report = format!("Ignore regions: '{part}' is not x,y,w,h");
                    return;
                }
            }
        }
        let job = ExportJob {
            layout: s.layout.clone(),
            csv: s.export_csv.then(|| dir.join(format!("{stem}_fps.csv"))),
            video: s.export_video.then(|| dir.join(format!("{stem}_analysis.mp4"))),
            graphs: s.export_graphs.then(|| (dir.join(format!("{stem}_graphs.mov")), s.graphs_size.0, s.graphs_size.1)),
            summary_json: s.export_summary.then(|| dir.join(format!("{stem}_summary.json"))),
            fix_capture_drops: s.fix_capture_drops,
            tears: s.show_experimental && s.detect_tears,
            compare: s.compare.clone(),
            encoder: (s.encoder != "auto").then(|| s.encoder.clone()),
            nits_in_csv: s.csv_nits,
            hwaccel: (s.hwaccel != "none").then(|| s.hwaccel.clone()),
            threshold: (!s.auto_threshold).then_some(s.threshold),
            ignore,
            sdr_white: s.sdr_white,
            ..ExportJob::new(input)
        };
        let progress = Arc::new(Progress::default());
        let p = progress.clone();
        let handle = std::thread::spawn(move || export::run(&job, &p));
        self.report = format!("Exporting to {}", dir.display());
        self.job = Some(RunningJob { progress, handle });
    }

    fn start_resolution(&mut self) {
        let (Some(input), Some(info)) = (self.input.clone(), self.info.clone()) else { return };
        let opts = ResolutionOptions {
            kind: self.settings.res_kind,
            samples_per_second: self.settings.res_rate,
            ignore: parse_rects(&self.settings.ignore).unwrap_or_default(),
            hwaccel: None,
            max_samples: None,
        };
        let progress = Arc::new(Progress::default());
        let p = progress.clone();
        let handle = std::thread::spawn(move || resolution::analyze(&input, &info, &opts, &p));
        self.res_error = None;
        self.res_job = Some((progress, handle));
    }

    fn poll_resolution(&mut self, ctx: &egui::Context) {
        let Some((_, handle)) = &self.res_job else { return };
        if handle.is_finished() {
            let (_, handle) = self.res_job.take().unwrap();
            match handle.join() {
                Ok(Ok(r)) => self.res_report = Some(r),
                Ok(Err(e)) => self.res_error = Some(format!("{e:#}")),
                Err(_) => self.res_error = Some("resolution analysis crashed".into()),
            }
        } else {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }

    /// Runs the pixel counter on the selected region of the frame at the slider position.
    fn run_counter(&mut self, ctx: &egui::Context) {
        let (Some(input), Some(info), Some(sel)) = (&self.input, &self.info, self.counter_sel) else { return };
        let (w, h) = (info.width as f32, info.height as f32);
        let x0 = (sel.min.x * w).clamp(0.0, w - 1.0) as usize;
        let y0 = (sel.min.y * h).clamp(0.0, h - 1.0) as usize;
        let rw = ((sel.width() * w) as usize).min(info.width - x0);
        let rh = ((sel.height() * h) as usize).min(info.height - y0);
        if rw < 16 || rh < 16 {
            self.res_error = Some("Pixel counter: select a larger region (at least 16x16 video pixels).".into());
            return;
        }
        let luma = match resolution::grab_luma(input, info, self.video_time) {
            Ok(l) => l,
            Err(e) => {
                self.res_error = Some(format!("{e:#}"));
                return;
            }
        };
        let full = self.res_report.as_ref().map_or((info.width, info.height), |r| (r.active.2, r.active.3));
        let (ex, ey) = resolution::analyze_region(&luma, info.width, (x0, y0, rw, rh), full, self.settings.res_kind);
        let mut rgba = Vec::with_capacity(rw * rh * 4);
        for y in y0..y0 + rh {
            for &v in &luma[y * info.width + x0..y * info.width + x0 + rw] {
                rgba.extend_from_slice(&[v, v, v, 255]);
            }
        }
        let image = egui::ColorImage::from_rgba_unmultiplied([rw, rh], &rgba);
        let tex = ctx.load_texture("pixel-counter", image, egui::TextureOptions::NEAREST);
        let (steps, run, vertical) =
            self.counter.as_ref().map_or((10.0, rw as f64, true), |c| (c.steps, c.run, c.vertical));
        self.counter =
            Some(Counter { tex, region: (x0, y0, rw, rh), full, x: ex, y: ey, steps, run, vertical, open: true });
    }

    fn poll_job(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.job else { return };
        if job.handle.is_finished() {
            let job = self.job.take().unwrap();
            self.report = match job.handle.join() {
                Ok(Ok(r)) => format!("Done.\n\n{}", r.summary),
                Ok(Err(e)) => format!("Export failed: {e:#}"),
                Err(_) => "Export crashed.".into(),
            };
        } else {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }

    fn mark_layout_changed(&mut self) {
        self.layout_changed = true;
    }

    /// Size video frames are decoded at for the preview (the preview canvas size).
    fn video_frame_size(&self) -> Option<(usize, usize)> {
        let info = self.info.as_ref()?;
        let w = info.width.min(PREVIEW_MAX_W);
        Some((w, ((info.height as f64 * w as f64 / info.width as f64).round() as usize).max(2)))
    }

    fn request_frame(&mut self) {
        if let (Some(input), Some(info), Some((w, h))) = (&self.input, &self.info, self.video_frame_size()) {
            self.grabber.request(input, info, self.video_time, w, h);
            self.frame_pending = true;
        }
    }

    /// Uploads a newly decoded preview frame to the GPU.
    fn update_video_frame(&mut self, ctx: &egui::Context) {
        if let Some(g) = self.grabber.poll() {
            // An older frame may arrive while a newer request is still being decoded.
            self.frame_pending = (g.time - self.video_time).abs() > 1e-6;
            match g.rgba {
                Ok(rgba) => {
                    let image = egui::ColorImage::from_rgba_unmultiplied([g.size.0, g.size.1], &rgba);
                    match &mut self.video_tex {
                        Some(t) => t.set(image, egui::TextureOptions::LINEAR),
                        None => self.video_tex = Some(ctx.load_texture("video", image, egui::TextureOptions::LINEAR)),
                    }
                    self.video_error = None;
                }
                Err(e) => self.video_error = Some(format!("{e:#}")),
            }
        }
        if self.frame_pending {
            ctx.request_repaint_after(std::time::Duration::from_millis(30));
        }
    }

    fn video_controls(&mut self, ui: &mut egui::Ui) {
        let Some(duration) = self.info.as_ref().and_then(|i| i.duration) else {
            ui.label("Open a video to preview it behind the graphs.");
            return;
        };
        ui.horizontal(|ui| {
            ui.label("Frame");
            let mut t = self.video_time;
            ui.spacing_mut().slider_width = (ui.available_width() - 200.0).max(120.0);
            if ui.add(egui::Slider::new(&mut t, 0.0..=duration).show_value(false)).changed() {
                self.video_time = t;
                self.request_frame();
            }
            ui.monospace(format!("{} / {}", clock(self.video_time), clock(duration)));
        });
        if let Some(e) = &self.video_error {
            ui.colored_label(Color32::from_rgb(230, 110, 110), e);
        }
    }

    // ---- panels ----------------------------------------------------------------

    fn left_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Video");
        if ui.button("Open video…").clicked()
            && let Some(path) = rfd::FileDialog::new()
                .add_filter("Video", &["mp4", "mkv", "mov", "avi", "ts", "m2ts", "webm"])
                .pick_file()
        {
            self.open_video(path);
        }
        if let Some(p) = &self.input {
            ui.label(p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
        }
        ui.small(&self.status);

        ui.separator();
        ui.heading("Layout");
        ui.horizontal_wrapped(|ui| {
            if ui.button("FPS (DF style)").clicked() {
                self.settings.layout = Layout::preset_fps();
                self.selected = None;
                self.mark_layout_changed();
            }
            if ui.button("FPS + HDR").clicked() {
                self.settings.layout = Layout::preset_fps_hdr();
                self.selected = None;
                self.mark_layout_changed();
            }
            if ui.button("HDR scopes grid").clicked() {
                self.settings.layout = Layout::preset_hdr_grid();
                self.settings.preview_target = PreviewTarget::Graphs;
                self.selected = None;
                self.mark_layout_changed();
            }
        });
        ui.add_space(4.0);
        let mut changed = false;
        for (i, w) in self.settings.layout.widgets.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                changed |= ui.checkbox(&mut w.enabled, "").changed();
                if ui.selectable_label(self.selected == Some(i), w.kind.label()).clicked() {
                    self.selected = Some(i);
                    if !w.enabled {
                        w.enabled = true;
                        changed = true;
                    }
                }
            });
        }
        if changed {
            self.mark_layout_changed();
        }
        ui.horizontal(|ui| {
            if ui.button("Save layout…").clicked()
                && let Some(path) =
                    rfd::FileDialog::new().add_filter("Layout", &["json"]).set_file_name("layout.json").save_file()
                && let Err(e) = self.settings.layout.save(&path)
            {
                self.report = format!("{e:#}");
            }
            if ui.button("Load layout…").clicked()
                && let Some(path) = rfd::FileDialog::new().add_filter("Layout", &["json"]).pick_file()
            {
                match Layout::load(&path) {
                    Ok(l) => {
                        self.settings.layout = l;
                        self.selected = None;
                        self.mark_layout_changed();
                    }
                    Err(e) => self.report = format!("{e:#}"),
                }
            }
        });

        ui.separator();
        ui.heading("Analysis");
        egui::Grid::new("analysis").num_columns(2).show(ui, |ui| {
            ui.label("Hardware decode");
            egui::ComboBox::from_id_salt("hwaccel").selected_text(&self.settings.hwaccel).show_ui(ui, |ui| {
                for h in HWACCELS {
                    ui.selectable_value(&mut self.settings.hwaccel, h.to_string(), h);
                }
            });
            ui.end_row();
            ui.label("Duplicate threshold");
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.settings.auto_threshold, "auto");
                ui.add_enabled(
                    !self.settings.auto_threshold,
                    egui::DragValue::new(&mut self.settings.threshold).range(0.05..=50.0).speed(0.05),
                );
            });
            ui.end_row();
            ui.label("SDR white (nits)");
            ui.add(egui::DragValue::new(&mut self.settings.sdr_white).range(50.0..=1000.0).speed(1.0));
            ui.end_row();
        });
        ui.checkbox(&mut self.settings.fix_capture_drops, "Count capture-clock repeats as game frames").on_hover_text(
            "Single repeated frames at a fixed interval come from a capture device running slightly fast, not from the game.",
        );
        ui.horizontal(|ui| {
            let name = self.settings.compare.as_ref().and_then(|p| p.file_name()).map(|n| n.to_string_lossy());
            ui.label(format!("Compare with: {}", name.as_deref().unwrap_or("none")));
            if ui.button("Choose…").clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .add_filter("Video", &["mp4", "mkv", "mov", "avi", "ts", "m2ts", "webm"])
                    .pick_file()
            {
                self.settings.compare = Some(path);
            }
            if self.settings.compare.is_some() && ui.button("Clear").clicked() {
                self.settings.compare = None;
            }
        });
        ui.label("Ignore regions (HUD, clocks): x,y,w,h; …");
        ui.text_edit_singleline(&mut self.settings.ignore);

        ui.separator();
        if ui.checkbox(&mut self.settings.show_experimental, "Show experimental features").changed()
            && !self.settings.show_experimental
        {
            // Hiding the section also leaves pixel-counter mode.
            self.counter_mode = false;
            self.counter_sel = None;
            self.counter = None;
        }
        if self.settings.show_experimental {
            ui.checkbox(&mut self.settings.detect_tears, "Detect torn frames (experimental)");
            egui::CollapsingHeader::new("Experimental: internal resolution")
                .default_open(false)
                .show(ui, |ui| self.resolution_panel(ui));
        }
    }

    fn resolution_panel(&mut self, ui: &mut egui::Ui) {
        ui.small("Estimates the game's rendering resolution from upscaling patterns. Treat results as a hint and check them with the pixel counter.");
        egui::Grid::new("res").num_columns(2).show(ui, |ui| {
            ui.label("Source");
            egui::ComboBox::from_id_salt("res_kind").selected_text(self.settings.res_kind.label()).show_ui(ui, |ui| {
                for k in SourceKind::ALL {
                    ui.selectable_value(&mut self.settings.res_kind, k, k.label());
                }
            });
            ui.end_row();
            ui.label("Samples / second");
            ui.add(egui::DragValue::new(&mut self.settings.res_rate).range(0.2..=10.0).speed(0.1));
            ui.end_row();
        });
        match &self.res_job {
            Some((progress, _)) => {
                let (stage, done, total) = progress.snapshot();
                let frac = if total > 0 { done as f32 / total as f32 } else { 0.0 };
                ui.add(egui::ProgressBar::new(frac.min(1.0)).text(format!("{stage} {done}/{total}")));
                if ui.button("Cancel").clicked() {
                    progress.cancel();
                }
            }
            None => {
                if ui.add_enabled(self.input.is_some(), egui::Button::new("Measure resolution")).clicked() {
                    self.start_resolution();
                }
            }
        }
        if let Some(e) = &self.res_error {
            ui.colored_label(Color32::from_rgb(230, 110, 110), e);
        }
        if let Some(r) = &self.res_report {
            ui.add(egui::Label::new(egui::RichText::new(r.summary()).monospace().size(10.5)).wrap());
            if r.samples.len() > 1 {
                resolution_graph(ui, r);
            }
        }

        ui.add_space(6.0);
        let can_count = self.input.is_some() && self.settings.preview_target == PreviewTarget::Video;
        ui.add_enabled_ui(can_count, |ui| {
            if ui.checkbox(&mut self.counter_mode, "Pixel counter").changed() && !self.counter_mode {
                self.counter_sel = None;
            }
        });
        ui.small(if can_count {
            "Drag a box over a sharp, slightly slanted edge in the preview to zoom in and count its steps."
        } else {
            "Open a video and preview \"Video with graphs\" to use it."
        });
    }

    fn right_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Selected graph");
        let (tw, th) = self.target_size();
        let (tw, th) = (tw as f32, th as f32);
        let mut changed = false;
        match self.selected.and_then(|i| self.settings.layout.widgets.get_mut(i)) {
            None => {
                ui.label("Click a graph in the preview, or pick one on the left.");
            }
            Some(w) => {
                ui.strong(w.kind.label());
                ui.horizontal(|ui| {
                    changed |= ui.checkbox(&mut w.enabled, "Shown").changed();
                    ui.checkbox(&mut w.locked, "Locked");
                });
                egui::Grid::new("geom").num_columns(2).show(ui, |ui| {
                    let mut px = [w.x * tw, w.y * th, w.w * tw, w.h * th];
                    for (name, v) in ["X", "Y", "Width", "Height"].iter().zip(px.iter_mut()) {
                        ui.label(*name);
                        changed |= ui.add(egui::DragValue::new(v).speed(1.0).suffix(" px")).changed();
                        ui.end_row();
                    }
                    w.x = (px[0] / tw).clamp(0.0, 1.0);
                    w.y = (px[1] / th).clamp(0.0, 1.0);
                    w.w = (px[2] / tw).clamp(0.01, 1.0);
                    w.h = (px[3] / th).clamp(0.01, 1.0);
                    ui.label("Background");
                    changed |= ui.add(egui::Slider::new(&mut w.bg_opacity, 0.0..=1.0)).changed();
                    ui.end_row();
                    if w.kind.has_history() {
                        ui.label("History");
                        changed |= ui.add(egui::Slider::new(&mut w.history_seconds, 1.0..=60.0).suffix(" s")).changed();
                        ui.end_row();
                    }
                });
                if ui.button("Reset to default size").clicked() {
                    let (bw, bh) = w.kind.base_size();
                    // Base sizes are for a 1080p canvas.
                    let scale = th / 1080.0;
                    w.w = (bw * scale / tw).min(1.0);
                    w.h = (bh * scale / th).min(1.0);
                    changed = true;
                }
            }
        }
        if changed {
            self.mark_layout_changed();
        }

        ui.separator();
        ui.heading("Export");
        let s = &mut self.settings;
        ui.checkbox(&mut s.export_csv, "CSV (per-frame data)");
        ui.indent("csvnits", |ui| {
            ui.add_enabled(s.export_csv, egui::Checkbox::new(&mut s.csv_nits, "include nits columns"))
        });
        ui.checkbox(&mut s.export_video, "Video with graphs (.mp4, HDR kept)");
        ui.indent("encoder", |ui| {
            ui.horizontal(|ui| {
                ui.label("Encoder");
                egui::ComboBox::from_id_salt("encoder").selected_text(&s.encoder).show_ui(ui, |ui| {
                    for e in ENCODERS {
                        ui.selectable_value(&mut s.encoder, e.to_string(), e);
                    }
                });
            })
            .response
            .on_hover_text(
                "auto: the GPU's encoder (AMD AMF, NVIDIA NVENC or Intel QSV), rendering again in software \
                 if it crashes. libx265 / libx264 are slower but don't depend on the GPU driver.",
            );
        });
        ui.checkbox(&mut s.export_graphs, "Graphs only (ProRes 4444 + alpha .mov)");
        ui.checkbox(&mut s.export_summary, "Summary (.json)");
        ui.horizontal(|ui| {
            ui.label("Graphs size");
            if graphs_size_picker(ui, &mut s.graphs_size, self.info.as_ref())
                && s.preview_target == PreviewTarget::Graphs
            {
                self.layout_changed = true;
            }
        });
        ui.horizontal(|ui| {
            let dir =
                s.output_dir.as_ref().map(|d| d.display().to_string()).unwrap_or_else(|| "next to the video".into());
            ui.label(format!("Save to: {dir}"));
        });
        ui.horizontal(|ui| {
            if ui.button("Choose folder…").clicked()
                && let Some(d) = rfd::FileDialog::new().pick_folder()
            {
                s.output_dir = Some(d);
            }
            if s.output_dir.is_some() && ui.button("Use video's folder").clicked() {
                s.output_dir = None;
            }
        });

        ui.add_space(8.0);
        let anything = s.export_csv || s.export_video || s.export_graphs;
        let graphs_needed = s.export_video || s.export_graphs;
        let has_graphs = s.layout.enabled().next().is_some();
        match &self.job {
            Some(job) => {
                let (stage, done, total) = job.progress.snapshot();
                let frac = if total > 0 { done as f32 / total as f32 } else { 0.0 };
                ui.add(egui::ProgressBar::new(frac.min(1.0)).text(format!("{stage}  {done}/{total}")));
                if ui.button("Cancel").clicked() {
                    job.progress.cancel();
                }
            }
            None => {
                let ready = self.input.is_some() && anything && (!graphs_needed || has_graphs);
                if ui.add_enabled(ready, egui::Button::new("Export").min_size(Vec2::new(120.0, 30.0))).clicked() {
                    self.start_export();
                }
                if self.input.is_none() {
                    ui.small("Open a video first.");
                } else if graphs_needed && !has_graphs {
                    ui.small("Enable at least one graph.");
                }
            }
        }
        ui.add_space(6.0);
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.add(egui::Label::new(egui::RichText::new(&self.report).monospace().size(11.0)).wrap());
        });
    }

    fn preview(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Preview for:");
            let before = self.settings.preview_target;
            ui.selectable_value(&mut self.settings.preview_target, PreviewTarget::Video, "Video with graphs");
            ui.selectable_value(&mut self.settings.preview_target, PreviewTarget::Graphs, "Graphs only");
            if before != self.settings.preview_target {
                self.layout_changed = true;
            }
            let (tw, th) = self.target_size();
            if self.settings.preview_target == PreviewTarget::Graphs {
                if graphs_size_picker(ui, &mut self.settings.graphs_size, self.info.as_ref()) {
                    self.layout_changed = true;
                }
            } else {
                ui.label(format!("canvas {tw}x{th}"));
            }
            ui.separator();
            ui.checkbox(&mut self.animate, "Animate demo data");
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.settings.lock_layout, "Lock layout");
            ui.separator();
            ui.checkbox(&mut self.settings.snap_always, "Always snap to grid");
            ui.label("Grid");
            ui.add(egui::DragValue::new(&mut self.settings.grid_px).range(2..=256).suffix(" px"));
            ui.separator();
            ui.small("Drag to move, corner to resize. Shift: grid  \u{b7}  Ctrl: align to other graphs  \u{b7}  Arrows: nudge (Shift: one grid step)");
        });

        let ctx = ui.ctx().clone();
        self.update_video_frame(&ctx);
        if self.settings.preview_target == PreviewTarget::Video {
            self.video_controls(ui);
        }

        // Fit the target aspect ratio into the available space.
        let (tw, th) = self.target_size();
        let avail = ui.available_size();
        let aspect = tw as f32 / th as f32;
        let size = if avail.x / avail.y > aspect {
            Vec2::new(avail.y * aspect, avail.y)
        } else {
            Vec2::new(avail.x, avail.x / aspect)
        };
        let (response, painter) = ui.allocate_painter(size, Sense::click_and_drag());
        let rect = response.rect;

        self.update_preview_image(ui.ctx());
        match (&self.video_tex, self.settings.preview_target) {
            (Some(tex), PreviewTarget::Video) => {
                painter.image(tex.id(), rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
            }
            _ => background(&painter, rect, self.settings.preview_target),
        }
        if let Some(tex) = &self.texture {
            painter.image(tex.id(), rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
        }

        let to_screen = |w: &Widget| {
            Rect::from_min_size(
                Pos2::new(rect.min.x + w.x * rect.width(), rect.min.y + w.y * rect.height()),
                Vec2::new(w.w * rect.width(), w.h * rect.height()),
            )
        };
        let handle_size = 12.0;
        let hit = |pos: Pos2, widgets: &[Widget]| -> Option<(usize, DragMode)> {
            widgets.iter().enumerate().rev().filter(|(_, w)| w.enabled).find_map(|(i, w)| {
                let r = to_screen(w);
                let handle = Rect::from_center_size(r.max, Vec2::splat(handle_size * 1.6));
                if handle.contains(pos) {
                    Some((i, DragMode::Resize))
                } else if r.contains(pos) {
                    Some((i, DragMode::Move))
                } else {
                    None
                }
            })
        };

        if self.counter_mode {
            self.counter_interaction(ui.ctx(), &response, rect, &painter);
            return;
        }
        if let Some(pos) = response.hover_pos() {
            match hit(pos, &self.settings.layout.widgets) {
                Some((_, DragMode::Resize)) => ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeNwSe),
                Some((_, DragMode::Move)) => ui.ctx().set_cursor_icon(egui::CursorIcon::Grab),
                None => {}
            }
        }
        if response.drag_started()
            && let Some(pos) = response.interact_pointer_pos()
            && let Some((index, mode)) = hit(pos, &self.settings.layout.widgets)
        {
            let w = &self.settings.layout.widgets[index];
            self.selected = Some(index);
            if !self.settings.lock_layout && !w.locked {
                self.drag = Some(Drag { index, mode, start_pointer: pos, start: (w.x, w.y, w.w, w.h) });
            }
        }
        if response.dragged()
            && let (Some(d), Some(pos)) = (&self.drag, response.interact_pointer_pos())
        {
            // Work in canvas pixels so grid steps and alignment are exact.
            let (cw, ch) = (tw as f32, th as f32);
            let delta = pos - d.start_pointer;
            let (dx, dy) = (delta.x / rect.width() * cw, delta.y / rect.height() * ch);
            let (x, y, ww, wh) = (d.start.0 * cw, d.start.1 * ch, d.start.2 * cw, d.start.3 * ch);
            let modifiers = ui.input(|i| i.modifiers);
            let snap = Snap {
                grid: (self.settings.snap_always != modifiers.shift).then_some(self.settings.grid_px as f32),
                align: modifiers.ctrl || modifiers.command,
                threshold: 10.0 * cw / rect.width(),
            };
            let others: Vec<RectPx> = self
                .settings
                .layout
                .widgets
                .iter()
                .enumerate()
                .filter(|(i, w)| *i != d.index && w.enabled)
                .map(|(_, w)| w.rect_px(cw, ch))
                .collect();
            let snapped = match d.mode {
                DragMode::Move => snap_move((x + dx, y + dy, ww, wh), &others, (cw, ch), snap),
                DragMode::Resize => snap_resize((x, y, ww + dx, wh + dy), &others, (cw, ch), snap, 0.02 * cw.min(ch)),
            };
            let (nx, ny, nw, nh) = snapped.rect;
            let w = &mut self.settings.layout.widgets[d.index];
            (w.x, w.y, w.w, w.h) = (nx / cw, ny / ch, nw / cw, nh / ch);
            self.guides = (snapped.guides_x, snapped.guides_y);
            self.layout_changed = true;
        }
        if response.drag_stopped() {
            self.drag = None;
            self.guides = (Vec::new(), Vec::new());
        }
        self.nudge_selected(ui, (tw, th));
        if response.clicked() {
            self.selected =
                response.interact_pointer_pos().and_then(|p| hit(p, &self.settings.layout.widgets)).map(|(i, _)| i);
        }

        // While dragging: the grid (when snapping to it) and the alignment lines in use.
        if self.drag.is_some() {
            let grid_on = self.settings.snap_always != ui.input(|i| i.modifiers.shift);
            let (sx, sy) = (rect.width() / tw as f32, rect.height() / th as f32);
            let step = self.settings.grid_px as f32;
            if grid_on && step * sx >= 6.0 {
                let faint = Stroke::new(1.0, Color32::from_white_alpha(22));
                let mut gx = 0.0;
                while gx <= tw as f32 {
                    let x = rect.min.x + gx * sx;
                    painter.line_segment([Pos2::new(x, rect.min.y), Pos2::new(x, rect.max.y)], faint);
                    gx += step;
                }
                let mut gy = 0.0;
                while gy <= th as f32 {
                    let y = rect.min.y + gy * sy;
                    painter.line_segment([Pos2::new(rect.min.x, y), Pos2::new(rect.max.x, y)], faint);
                    gy += step;
                }
            }
            let guide = Stroke::new(1.5, Color32::from_rgb(255, 90, 200));
            for &gx in &self.guides.0 {
                let x = rect.min.x + gx * sx;
                painter.line_segment([Pos2::new(x, rect.min.y), Pos2::new(x, rect.max.y)], guide);
            }
            for &gy in &self.guides.1 {
                let y = rect.min.y + gy * sy;
                painter.line_segment([Pos2::new(rect.min.x, y), Pos2::new(rect.max.x, y)], guide);
            }
        }

        // Outlines and resize handles on top of the rendered preview.
        let all_locked = self.settings.lock_layout;
        for (i, w) in self.settings.layout.widgets.iter().enumerate().filter(|(_, w)| w.enabled) {
            let r = to_screen(w);
            let selected = self.selected == Some(i);
            let locked = all_locked || w.locked;
            let color = match (selected, locked) {
                (true, false) => Color32::from_rgb(90, 170, 255),
                (true, true) => Color32::from_rgb(255, 170, 60),
                (false, true) => Color32::from_rgba_unmultiplied(255, 170, 60, 90),
                (false, false) => Color32::from_white_alpha(70),
            };
            painter.rect_stroke(r, 0.0, Stroke::new(if selected { 2.0 } else { 1.0 }, color), StrokeKind::Inside);
            painter.rect_filled(
                Rect::from_center_size(r.max - Vec2::splat(handle_size / 2.0), Vec2::splat(handle_size)),
                2.0,
                color,
            );
            if selected {
                let label = if locked { format!("{} (locked)", w.kind.label()) } else { w.kind.label().to_string() };
                painter.text(
                    r.min + Vec2::new(4.0, -14.0),
                    egui::Align2::LEFT_TOP,
                    label,
                    egui::FontId::proportional(12.0),
                    color,
                );
            }
        }
    }

    /// Arrow keys move the selected graph by one canvas pixel, or one grid step with Shift.
    fn nudge_selected(&mut self, ui: &egui::Ui, (tw, th): (usize, usize)) {
        if ui.ctx().egui_wants_keyboard_input() || self.settings.lock_layout {
            return;
        }
        let Some(w) = self.selected.and_then(|i| self.settings.layout.widgets.get_mut(i)) else { return };
        if w.locked {
            return;
        }
        let (mut dx, mut dy, shift) = ui.input(|i| {
            let k = |key| i.key_pressed(key) as i32 as f32;
            (
                k(egui::Key::ArrowRight) - k(egui::Key::ArrowLeft),
                k(egui::Key::ArrowDown) - k(egui::Key::ArrowUp),
                i.modifiers.shift,
            )
        });
        if dx == 0.0 && dy == 0.0 {
            return;
        }
        let step = if shift { self.settings.grid_px as f32 } else { 1.0 };
        dx *= step;
        dy *= step;
        let (cw, ch) = (tw as f32, th as f32);
        w.x = ((w.x * cw + dx) / cw).clamp(0.0, 1.0 - w.w);
        w.y = ((w.y * ch + dy) / ch).clamp(0.0, 1.0 - w.h);
        self.layout_changed = true;
    }

    /// Pixel-counter mode: dragging selects a region; releasing analyses it.
    fn counter_interaction(
        &mut self,
        ctx: &egui::Context,
        response: &egui::Response,
        rect: Rect,
        painter: &egui::Painter,
    ) {
        let to_norm = |p: Pos2| {
            Pos2::new(
                ((p.x - rect.min.x) / rect.width()).clamp(0.0, 1.0),
                ((p.y - rect.min.y) / rect.height()).clamp(0.0, 1.0),
            )
        };
        ctx.set_cursor_icon(egui::CursorIcon::Crosshair);
        if response.drag_started() {
            self.counter_drag = response.interact_pointer_pos().map(to_norm);
        }
        if let (Some(start), Some(pos)) = (self.counter_drag, response.interact_pointer_pos())
            && response.dragged()
        {
            self.counter_sel = Some(Rect::from_two_pos(start, to_norm(pos)));
        }
        if response.drag_stopped() {
            self.counter_drag = None;
            self.run_counter(ctx);
        }
        if let Some(sel) = self.counter_sel {
            let screen = Rect::from_min_max(
                Pos2::new(rect.min.x + sel.min.x * rect.width(), rect.min.y + sel.min.y * rect.height()),
                Pos2::new(rect.min.x + sel.max.x * rect.width(), rect.min.y + sel.max.y * rect.height()),
            );
            painter.rect_stroke(screen, 0.0, Stroke::new(2.0, Color32::from_rgb(255, 200, 60)), StrokeKind::Outside);
        }
        painter.text(
            rect.min + Vec2::new(8.0, 8.0),
            egui::Align2::LEFT_TOP,
            "Pixel counter: drag a box over an edge",
            egui::FontId::proportional(14.0),
            Color32::from_rgb(255, 200, 60),
        );
    }

    fn counter_window(&mut self, ctx: &egui::Context) {
        let Some(c) = &mut self.counter else { return };
        let mut open = c.open;
        egui::Window::new("Pixel counter (experimental)").open(&mut open).default_width(560.0).show(ctx, |ui| {
            let (x, y, w, h) = c.region;
            ui.label(format!("Region {w}x{h} at ({x},{y}) of the frame, zoomed (nearest neighbour)."));
            let zoom = (520.0 / w as f32).min(360.0 / h as f32).max(1.0);
            ui.add(egui::Image::from_texture(egui::load::SizedTexture::new(c.tex.id(), Vec2::new(w as f32 * zoom, h as f32 * zoom))));
            ui.separator();
            ui.strong("Automatic estimate for this region");
            let line = |a: &Option<AxisEstimate>, name: &str, full: usize| match a {
                Some(a) => format!(
                    "{name}: {:.0} of {full} px (scale x{:.3}{}, confidence {})",
                    a.native,
                    a.scale,
                    a.integer.map(|n| format!(", integer x{n}")).unwrap_or_default(),
                    a.confidence_label()
                ),
                None => format!("{name}: no clear pattern in this region"),
            };
            ui.label(line(&c.x, "Width", c.full.0));
            ui.label(line(&c.y, "Height", c.full.1));
            ui.separator();
            ui.strong("Manual count");
            ui.small("Along a slanted edge, count the stair steps and how many output pixels they span in the same direction.");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut c.vertical, true, "Vertical steps");
                ui.selectable_value(&mut c.vertical, false, "Horizontal steps");
            });
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut c.steps).range(1.0..=4000.0).speed(0.2));
                ui.label("steps over");
                ui.add(egui::DragValue::new(&mut c.run).range(1.0..=8000.0).speed(1.0));
                ui.label("output pixels");
            });
            let full = if c.vertical { c.full.1 } else { c.full.0 };
            ui.label(format!("= {:.0} native px of {full} ({:.1}%)", full as f64 * c.steps / c.run, 100.0 * c.steps / c.run));
        });
        c.open = open;
    }

    /// Re-renders the preview image when the layout, target or demo frame changes.
    fn update_preview_image(&mut self, ctx: &egui::Context) {
        let (tw, th) = self.target_size();
        let pw = tw.min(PREVIEW_MAX_W);
        let ph = ((th as f64 * pw as f64 / tw as f64).round() as usize).max(2);
        if self.canvas.as_ref().is_none_or(|(w, h, _)| (*w, *h) != (pw, ph)) {
            self.canvas = Canvas::full(pw, ph).ok().map(|c| (pw, ph, c));
            self.layout_changed = true;
        }
        if self.animate {
            self.demo_frame = (self.demo_frame + 1) % self.demo.len();
            self.redraw = true;
            ctx.request_repaint();
        }
        if !(self.layout_changed || self.redraw) {
            return;
        }
        let Some((_, _, canvas)) = self.canvas.as_mut() else { return };
        if self.layout_changed {
            canvas.invalidate();
        }
        canvas.draw(&self.settings.layout, &self.demo.frame_data(self.demo_frame));
        let image = egui::ColorImage::from_rgba_premultiplied([pw, ph], canvas.pixels());
        match &mut self.texture {
            Some(t) => t.set(image, egui::TextureOptions::LINEAR),
            None => self.texture = Some(ctx.load_texture("preview", image, egui::TextureOptions::LINEAR)),
        }
        self.layout_changed = false;
        self.redraw = false;
    }
}

/// Preset and custom size for the graphs-only export. Returns true if it changed.
fn graphs_size_picker(ui: &mut egui::Ui, size: &mut (usize, usize), info: Option<&VideoInfo>) -> bool {
    let mut changed = false;
    let current = GRAPH_SIZES
        .iter()
        .find(|(_, s)| s == size)
        .map(|(name, s)| format!("{name} ({}x{})", s.0, s.1))
        .unwrap_or_else(|| format!("{}x{}", size.0, size.1));
    egui::ComboBox::from_id_salt(ui.next_auto_id()).selected_text(current).show_ui(ui, |ui| {
        for (name, s) in GRAPH_SIZES {
            changed |= ui.selectable_value(size, s, format!("{name} ({}x{})", s.0, s.1)).changed();
        }
        if let Some(i) = info {
            changed |= ui
                .selectable_value(size, (i.width, i.height), format!("Same as video ({}x{})", i.width, i.height))
                .changed();
        }
    });
    changed |= ui.add(egui::DragValue::new(&mut size.0).range(64..=7680)).changed();
    ui.label("x");
    changed |= ui.add(egui::DragValue::new(&mut size.1).range(64..=4320)).changed();
    changed
}

/// Parses "x,y,w,h; x,y,w,h" into rectangles.
fn parse_rects(text: &str) -> Option<Vec<[usize; 4]>> {
    text.split(';')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(|part| {
            part.split(',')
                .map(|n| n.trim().parse().ok())
                .collect::<Option<Vec<usize>>>()
                .and_then(|v| v.try_into().ok())
        })
        .collect()
}

/// Native height per sample over time; colour shows confidence.
fn resolution_graph(ui: &mut egui::Ui, r: &ResolutionReport) {
    let (resp, painter) = ui.allocate_painter(Vec2::new(ui.available_width(), 90.0), Sense::hover());
    let rect = resp.rect;
    painter.rect_filled(rect, 2.0, Color32::from_gray(28));
    let full = r.active.3 as f64;
    let t_max = r.samples.last().map_or(1.0, |s| s.time).max(1e-3);
    let lo = full * 0.4;
    let to_y = |v: f64| rect.max.y - ((v - lo) / (full - lo)).clamp(0.0, 1.0) as f32 * rect.height();
    for frac in [0.5, 0.75, 1.0] {
        let y = to_y(full * frac);
        painter.line_segment(
            [Pos2::new(rect.min.x, y), Pos2::new(rect.max.x, y)],
            Stroke::new(1.0, Color32::from_gray(55)),
        );
        painter.text(
            Pos2::new(rect.min.x + 2.0, y - 1.0),
            egui::Align2::LEFT_BOTTOM,
            format!("{:.0}", full * frac),
            egui::FontId::proportional(9.0),
            Color32::from_gray(140),
        );
    }
    for s in &r.samples {
        let x = rect.min.x + (s.time / t_max) as f32 * rect.width();
        let (y, color) = match s.y {
            Some(a) if a.confidence >= 40.0 => (to_y(a.native), Color32::from_rgb(110, 210, 120)),
            Some(a) if a.confidence >= 15.0 => (to_y(a.native), Color32::from_rgb(230, 190, 80)),
            Some(a) => (to_y(a.native), Color32::from_gray(120)),
            None => (to_y(full), Color32::from_gray(70)),
        };
        painter.circle_filled(Pos2::new(x, y), 2.5, color);
    }
    resp.on_hover_text("Native height per sample. Green = high confidence, yellow = medium, grey = low / none (drawn at output height).");
}

/// Seconds as m:ss.s
fn clock(t: f64) -> String {
    format!("{}:{:04.1}", (t / 60.0) as u64, t % 60.0)
}

/// Checkerboard for the transparent graphs export, a dark gradient standing in for video.
fn background(painter: &egui::Painter, rect: Rect, target: PreviewTarget) {
    match target {
        PreviewTarget::Graphs => {
            let cell = 16.0;
            painter.rect_filled(rect, 0.0, Color32::from_gray(48));
            let (nx, ny) = ((rect.width() / cell).ceil() as i32, (rect.height() / cell).ceil() as i32);
            for y in 0..ny {
                for x in (y % 2..nx).step_by(2) {
                    let min = rect.min + Vec2::new(x as f32 * cell, y as f32 * cell);
                    let r = Rect::from_min_size(min, Vec2::splat(cell)).intersect(rect);
                    painter.rect_filled(r, 0.0, Color32::from_gray(64));
                }
            }
        }
        PreviewTarget::Video => {
            let mut mesh = egui::Mesh::default();
            mesh.colored_vertex(rect.left_top(), Color32::from_rgb(40, 62, 92));
            mesh.colored_vertex(rect.right_top(), Color32::from_rgb(52, 72, 98));
            mesh.colored_vertex(rect.right_bottom(), Color32::from_rgb(38, 46, 40));
            mesh.colored_vertex(rect.left_bottom(), Color32::from_rgb(30, 40, 34));
            mesh.add_triangle(0, 1, 2);
            mesh.add_triangle(0, 2, 3);
            painter.add(mesh);
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "video",
                egui::FontId::proportional(28.0),
                Color32::from_white_alpha(25),
            );
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_job(&ctx);
        self.poll_resolution(&ctx);
        self.counter_window(&ctx);
        egui::Panel::left("settings").resizable(true).default_size(300.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| self.left_panel(ui));
        });
        egui::Panel::right("inspector").resizable(true).default_size(330.0).show(ui, |ui| self.right_panel(ui));
        egui::CentralPanel::default().show(ui, |ui| self.preview(ui));
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.settings);
    }
}

/// Synthetic data so the layout can be designed without analysing a video first.
struct Demo {
    fps: Vec<f64>,
    frametime_ms: Vec<f64>,
    peak: Vec<f64>,
    avg: Vec<f64>,
    nits: Vec<NitsFrame>,
}

impl Demo {
    fn new() -> Self {
        // 20 s at 60 Hz: a locked 60, a stretch of uneven ~45, then 30.
        let n = 1200;
        let mut rng = Lcg(7);
        let mut is_new = Vec::with_capacity(n);
        for i in 0..n {
            let new = match i {
                _ if i < 400 => true,
                _ if i < 800 => i % 4 != 3 || rng.next() < 0.1,
                _ => i % 2 == 0,
            };
            is_new.push(new);
        }
        let (fps, frametime_ms) = Timeline::constant(60.0).fps_series(&is_new);
        let peak: Vec<f64> = (0..n).map(|i| 900.0 + 500.0 * ((i as f64) / 90.0).sin().abs()).collect();
        let avg: Vec<f64> = (0..n).map(|i| 110.0 + 25.0 * ((i as f64) / 140.0).cos()).collect();
        // A handful of distinct scope frames, cycled while animating.
        let nits = (0..8).map(|k| demo_nits(k, &mut rng, peak[k * 100] as f32, avg[k * 100] as f32)).collect();
        Self { fps, frametime_ms, peak, avg, nits }
    }

    fn len(&self) -> usize {
        self.fps.len()
    }

    fn frame_data(&self, frame: usize) -> FrameData<'_> {
        FrameData {
            frame,
            capture_fps: 60.0,
            fps: &self.fps,
            frametime_ms: &self.frametime_ms,
            compare_fps: None,
            compare_frametime_ms: None,
            nits: Some(&self.nits[(frame / 8) % self.nits.len()]),
            peak_history: &self.peak,
            avg_history: &self.avg,
            max_cll: 3000.0,
            source_pixels: (1920 * 1080) as f32,
        }
    }
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        (0..4).map(|_| self.next()).sum::<f64>() - 2.0
    }
}

fn log_bin(nits: f64, bins: usize) -> usize {
    (fpsprobe::canvas::log_axis(nits) * (bins - 1) as f64).round() as usize
}

/// Plausible-looking scope data: a scene with sky highlights, mid-tones and shadows.
fn demo_nits(seed: usize, rng: &mut Lcg, peak: f32, avg: f32) -> NitsFrame {
    let phase = seed as f64 * 0.7;
    let mut wave = vec![0u32; 3 * WAVE_ROWS * WAVE_COLS];
    let mut hist = vec![0u32; 4 * HIST_BINS];
    let mut cie = vec![0u32; XY_BINS * XY_BINS];
    let mut vec = vec![0u32; XY_BINS * XY_BINS];
    for col in 0..WAVE_COLS {
        let t = col as f64 / WAVE_COLS as f64;
        for _ in 0..40 {
            // Pick a layer: sky, midtones or shadows.
            let r = rng.next();
            let base = if r < 0.25 {
                600.0 + 300.0 * (t * 6.0 + phase).sin()
            } else if r < 0.7 {
                60.0 * (1.0 + (t * 11.0 + phase).sin()) + 20.0
            } else {
                3.0 + 4.0 * rng.next()
            };
            let tint = [1.0 + 0.3 * rng.gauss(), 1.0 + 0.2 * rng.gauss(), 0.9 + 0.4 * rng.gauss()];
            for c in 0..3 {
                let nits = (base * tint[c] * (1.0 + 0.15 * rng.gauss())).max(0.0);
                let row = log_bin(nits, WAVE_ROWS);
                wave[(c * WAVE_ROWS + row) * WAVE_COLS + col] += 400;
                hist[(c + 1) * HIST_BINS + log_bin(nits, HIST_BINS)] += 50;
            }
            hist[log_bin(base, HIST_BINS)] += 50;
            // Chromaticity scattered around D65, stretched toward green/yellow and blue.
            let (x, y) = (
                0.3127 + 0.05 * rng.gauss() + 0.03 * (tint[0] - tint[2]),
                0.329 + 0.05 * rng.gauss() + 0.02 * (tint[1] - 1.0),
            );
            let (bx, by) = ((x / 0.8 * XY_BINS as f64) as usize, (y / 0.9 * XY_BINS as f64) as usize);
            if bx < XY_BINS && by < XY_BINS {
                cie[by * XY_BINS + bx] += 3;
            }
            let (cb, cr) = (0.06 * rng.gauss() + 0.04 * (tint[2] - 1.0), 0.05 * rng.gauss() + 0.03 * (tint[0] - 1.0));
            let (vx, vy) = (((cb + 0.5) * XY_BINS as f64) as usize, ((cr + 0.5) * XY_BINS as f64) as usize);
            if vx < XY_BINS && vy < XY_BINS {
                vec[vy * XY_BINS + vx] += 3;
            }
        }
    }
    NitsFrame {
        peak: peak * 1.8,
        avg: avg * 1.6,
        peak_lum: peak,
        avg_lum: avg,
        scopes: Some(ScopeData { wave: Some(wave), hist: Some(hist), cie: Some(cie), vec: Some(vec) }),
        ..Default::default()
    }
}
