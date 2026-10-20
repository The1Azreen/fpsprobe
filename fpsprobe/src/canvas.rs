//! Draws an overlay [`Layout`] into an RGBA image, one video frame at a time.
//!
//! Static parts (panel backgrounds, plot frames, guides, axis labels, CIE gamut
//! outlines, ...) are drawn once into a template; every frame starts from a copy
//! of it and only draws the data. Pixels are premultiplied RGBA.

use crate::layout::{Layout, Widget, WidgetKind};
use crate::nits::{HIST_BINS, NitsFrame, WAVE_COLS, WAVE_ROWS, XY_BINS};
use ab_glyph::{Font, FontRef, ScaleFont};
use anyhow::{Context, Result};
use tiny_skia::{Paint, PathBuilder, Pixmap, Stroke, Transform};

const FONT: &[u8] = include_bytes!("../assets/fonts/FjallaOne-Regular.ttf");

/// Everything a frame of the overlay displays.
pub struct FrameData<'a> {
    pub frame: usize,
    pub capture_fps: f64,
    /// Per-frame FPS over a trailing one-second window (NaN = unknown yet).
    pub fps: &'a [f64],
    /// Per-frame most recent frame time in ms.
    pub frametime_ms: &'a [f64],
    pub nits: Option<&'a NitsFrame>,
    /// Per-frame luminance peak / average in nits (NaN = not measured).
    pub peak_history: &'a [f64],
    pub avg_history: &'a [f64],
    pub max_cll: f32,
    /// Source video pixel count (normalizes scope densities).
    pub source_pixels: f32,
}

/// Log-nits axis shared with the GPU bins: 0 nits at 0, 10000 nits at 1.
pub fn log_axis(n: f64) -> f64 {
    ((n.max(0.0) + 1.0).log10() / 10001f64.log10()).clamp(0.0, 1.0)
}

const DECADES: [(f64, &str); 5] = [(1.0, "1"), (10.0, "10"), (100.0, "100"), (1000.0, "1k"), (10000.0, "10k")];

/// CIE 1931 2° spectral locus (x, y), 380-700 nm.
const LOCUS: [(f32, f32); 41] = [
    (0.1741, 0.0050),
    (0.1733, 0.0048),
    (0.1714, 0.0051),
    (0.1644, 0.0109),
    (0.1566, 0.0177),
    (0.1440, 0.0297),
    (0.1241, 0.0578),
    (0.1096, 0.0868),
    (0.0913, 0.1327),
    (0.0687, 0.2007),
    (0.0454, 0.2950),
    (0.0235, 0.4127),
    (0.0082, 0.5384),
    (0.0039, 0.6548),
    (0.0139, 0.7502),
    (0.0389, 0.8120),
    (0.0743, 0.8338),
    (0.1142, 0.8262),
    (0.1547, 0.8059),
    (0.1929, 0.7816),
    (0.2296, 0.7543),
    (0.2658, 0.7243),
    (0.3016, 0.6923),
    (0.3373, 0.6589),
    (0.3731, 0.6245),
    (0.4087, 0.5896),
    (0.4441, 0.5547),
    (0.4788, 0.5202),
    (0.5125, 0.4866),
    (0.5448, 0.4544),
    (0.5752, 0.4242),
    (0.6029, 0.3965),
    (0.6270, 0.3725),
    (0.6482, 0.3514),
    (0.6658, 0.3340),
    (0.6801, 0.3197),
    (0.6915, 0.3083),
    (0.7006, 0.2993),
    (0.7079, 0.2920),
    (0.7190, 0.2809),
    (0.7347, 0.2653),
];
/// Name, primaries (x, y) and outline colour.
type Gamut = (&'static str, [(f32, f32); 3], [u8; 4]);
const GAMUTS: [Gamut; 3] = [
    ("Rec.709", [(0.64, 0.33), (0.30, 0.60), (0.15, 0.06)], [235, 235, 235, 170]),
    ("P3", [(0.68, 0.32), (0.265, 0.69), (0.15, 0.06)], [240, 210, 110, 150]),
    ("Rec.2020", [(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)], [120, 220, 240, 150]),
];
const CIE_X_RANGE: f32 = 0.8;
const CIE_Y_RANGE: f32 = 0.9;

pub struct Canvas {
    pixmap: Pixmap,
    font: FontRef<'static>,
    canvas_w: f32,
    canvas_h: f32,
    /// Top-left of this image within the canvas (the image may cover only part of it).
    ox: f32,
    oy: f32,
    template: Option<Vec<u8>>,
    building: bool,
    /// Waveform bin count -> brightness, keyed by the source pixel count.
    wave_lut: (f32, Vec<u8>),
    cie_colors: Vec<[u8; 3]>,
    vec_colors: Vec<[u8; 3]>,
}

impl Canvas {
    /// An image covering `region` (x, y, w, h in canvas pixels) of a `canvas_w` x `canvas_h` canvas.
    pub fn new(canvas_w: usize, canvas_h: usize, region: (usize, usize, usize, usize)) -> Result<Self> {
        let (x, y, w, h) = region;
        let pixmap = Pixmap::new(w as u32, h as u32).context("overlay region is empty")?;
        let font = FontRef::try_from_slice(FONT).context("bad overlay font")?;
        Ok(Self {
            pixmap,
            font,
            canvas_w: canvas_w as f32,
            canvas_h: canvas_h as f32,
            ox: x as f32,
            oy: y as f32,
            template: None,
            building: false,
            wave_lut: (0.0, Vec::new()),
            cie_colors: cie_colors(),
            vec_colors: vec_colors(),
        })
    }

    /// An image covering the whole canvas.
    pub fn full(canvas_w: usize, canvas_h: usize) -> Result<Self> {
        Self::new(canvas_w, canvas_h, (0, 0, canvas_w, canvas_h))
    }

    pub fn width(&self) -> u32 {
        self.pixmap.width()
    }
    pub fn height(&self) -> u32 {
        self.pixmap.height()
    }
    /// Premultiplied RGBA pixels.
    pub fn pixels(&self) -> &[u8] {
        self.pixmap.data()
    }

    /// Forces the static template to be redrawn (call after changing the layout).
    pub fn invalidate(&mut self) {
        self.template = None;
    }

    pub fn draw(&mut self, layout: &Layout, d: &FrameData) {
        if self.template.is_none() {
            self.pixmap.data_mut().fill(0);
            self.building = true;
            for w in layout.enabled() {
                self.widget(w, d);
            }
            self.building = false;
            self.template = Some(self.pixmap.data().to_vec());
        }
        if let Some(t) = &self.template {
            self.pixmap.data_mut().copy_from_slice(t);
        }
        for w in layout.enabled() {
            self.widget(w, d);
        }
    }

    fn widget(&mut self, w: &Widget, d: &FrameData) {
        let (x, y, ww, wh) = w.rect_px(self.canvas_w, self.canvas_h);
        let r = Rect { x: x - self.ox, y: y - self.oy, w: ww, h: wh };
        let (bw, bh) = w.kind.base_size();
        let s = (ww / bw).min(wh / bh).max(0.2);
        if self.building {
            self.fill(r, [0, 0, 0, (w.bg_opacity.clamp(0.0, 1.0) * 255.0) as u8]);
        }
        match w.kind {
            WidgetKind::FpsCounter => self.fps_counter(r, s, d),
            WidgetKind::FpsGraph => self.fps_graph(r, s, w, d),
            WidgetKind::FrametimeGraph => self.frametime_graph(r, s, w, d),
            WidgetKind::NitsReadout => self.nits_readout(r, s, d),
            WidgetKind::NitsGraph => self.nits_graph(r, s, w, d),
            WidgetKind::Waveform => self.waveform(r, s, d),
            WidgetKind::Histogram => self.histogram(r, s, d),
            WidgetKind::Cie => self.cie(r, s, d),
            WidgetKind::Vectorscope => self.vectorscope(r, s, d),
        }
    }

    // ---- widgets -------------------------------------------------------------

    fn fps_counter(&mut self, r: Rect, s: f32, d: &FrameData) {
        if self.building {
            return;
        }
        let fps = d.fps.get(d.frame).copied().unwrap_or(f64::NAN);
        let ft = d.frametime_ms.get(d.frame).copied().unwrap_or(f64::NAN);
        let fps_text = if fps.is_nan() { "FPS --".to_string() } else { format!("FPS {fps:.0}") };
        let base = r.y + r.h * 0.5 + 22.0 * s;
        self.text(&fps_text, r.x + 20.0 * s, base, 62.0 * s, [255, 255, 255]);
        if !ft.is_nan() {
            self.text(&format!("{ft:.1} ms"), r.x + 250.0 * s, base, 32.0 * s, [200, 200, 200]);
        }
    }

    /// Plot area below a small title line.
    fn titled_plot(&mut self, r: Rect, s: f32, title: &str, left_margin: f32) -> Rect {
        let pad = 10.0 * s;
        if !self.building {
            self.text(title, r.x + pad, r.y + pad + 15.0 * s, 18.0 * s, [170, 170, 170]);
        }
        Rect {
            x: r.x + pad + left_margin * s,
            y: r.y + pad + 24.0 * s,
            w: r.w - 2.0 * pad - left_margin * s,
            h: r.h - 2.0 * pad - 24.0 * s,
        }
    }

    fn fps_graph(&mut self, r: Rect, s: f32, w: &Widget, d: &FrameData) {
        let plot = self.titled_plot(r, s, "FPS", 0.0);
        let max = d.capture_fps.max(30.0);
        let map = move |v: f64| v.clamp(0.0, max) / max;
        if self.building {
            let guides: Vec<(f64, String)> =
                [30.0, 60.0, 120.0].into_iter().filter(|&g| g < max).map(|g| (g, format!("{g:.0}"))).collect();
            self.plot_frame(plot, s, &map, &guides, false);
        } else {
            self.series(plot, s, d, w.history_seconds, &[(d.fps, [80, 220, 120])], &map);
        }
    }

    fn frametime_graph(&mut self, r: Rect, s: f32, w: &Widget, d: &FrameData) {
        let plot = self.titled_plot(r, s, "FRAME TIME", 0.0);
        // Up to 4 capture frames (at least 50 ms) so 16.7 / 33.3 ms cadences stay readable.
        let max = (4000.0 / d.capture_fps).max(50.0);
        let map = move |v: f64| v.clamp(0.0, max) / max;
        if self.building {
            // Only label cadences the capture rate can actually show, so labels don't overlap.
            let guides: Vec<(f64, String)> = [8.3, 16.7, 33.3, 66.7]
                .into_iter()
                .filter(|&g| g < max && g >= 990.0 / d.capture_fps)
                .map(|g| (g, format!("{g:.1} ms")))
                .collect();
            self.plot_frame(plot, s, &map, &guides, false);
        } else {
            self.series(plot, s, d, w.history_seconds, &[(d.frametime_ms, [255, 170, 60])], &map);
        }
    }

    fn nits_readout(&mut self, r: Rect, s: f32, d: &FrameData) {
        if self.building {
            return;
        }
        let base = r.y + r.h * 0.5 + 16.0 * s;
        let Some(n) = d.nits else {
            self.text("nits: no data", r.x + 20.0 * s, base, 30.0 * s, [200, 200, 200]);
            return;
        };
        self.text(&format!("PEAK {:.0}", n.peak_lum), r.x + 20.0 * s, base, 44.0 * s, [255, 230, 120]);
        self.text(&format!("AVG {:.0} nits", n.avg_lum), r.x + 260.0 * s, base, 30.0 * s, [120, 220, 255]);
        self.text(&format!("MaxCLL {:.0}", d.max_cll), r.x + r.w - 170.0 * s, base, 22.0 * s, [150, 150, 150]);
    }

    fn nits_graph(&mut self, r: Rect, s: f32, w: &Widget, d: &FrameData) {
        let plot = self.titled_plot(r, s, "NITS  peak / avg", 36.0);
        if self.building {
            let ticks: Vec<(f64, String)> = DECADES.iter().skip(1).map(|(v, l)| (*v, l.to_string())).collect();
            self.plot_frame(plot, s, &log_axis, &ticks, true);
        } else {
            let lines: [(&[f64], [u8; 3]); 2] = [(d.peak_history, [255, 230, 120]), (d.avg_history, [120, 220, 255])];
            self.series(plot, s, d, w.history_seconds, &lines, &log_axis);
        }
    }

    fn waveform(&mut self, r: Rect, s: f32, d: &FrameData) {
        let plot = self.titled_plot(r, s, "WAVEFORM  nits", 36.0);
        if self.building {
            let ticks: Vec<(f64, String)> = DECADES.iter().map(|(v, l)| (*v, l.to_string())).collect();
            self.plot_frame(plot, s, &log_axis, &ticks, true);
            return;
        }
        let Some(wave) = d.nits.and_then(|n| n.scopes.as_ref()).and_then(|sc| sc.wave.as_deref()) else { return };
        let per_col = d.source_pixels / WAVE_COLS as f32;
        if self.wave_lut.0 != per_col {
            // Density relative to an even spread over all rows; dense bins saturate.
            let norm = WAVE_ROWS as f32 / per_col.max(1.0);
            let len = (8.0 * 6.0 / norm).ceil() as usize + 1;
            self.wave_lut =
                (per_col, (0..len).map(|c| ((1.0 - (-(c as f32) * norm / 6.0).exp()) * 255.0) as u8).collect());
        }
        let Some(px) = self.clip(plot) else { return };
        let lut = &self.wave_lut.1;
        let level = |count: u32| lut[(count as usize).min(lut.len() - 1)];
        // Bin column for each visible pixel column, accounting for clipping at the left edge.
        let cols: Vec<usize> =
            (0..px.w).map(|i| ((i + px.skip_x) * WAVE_COLS / px.full_w).min(WAVE_COLS - 1)).collect();
        let plane = WAVE_ROWS * WAVE_COLS;
        let pw = self.pixmap.width() as usize;
        let data = self.pixmap.data_mut();
        for py in 0..px.h {
            let row = ((px.full_h - 1 - (py + px.skip_y)) * WAVE_ROWS / px.full_h).min(WAVE_ROWS - 1) * WAVE_COLS;
            let line = (px.y + py) * pw + px.x;
            for (i, &col) in cols.iter().enumerate() {
                let add = [level(wave[row + col]), level(wave[plane + row + col]), level(wave[2 * plane + row + col])];
                if add != [0, 0, 0] {
                    add_px(data, (line + i) * 4, add);
                }
            }
        }
    }

    fn histogram(&mut self, r: Rect, s: f32, d: &FrameData) {
        let plot = self.titled_plot(r, s, "HISTOGRAM  nits", 18.0);
        let label_h = 20.0 * s;
        let row_h = (plot.h - label_h) / 4.0;
        let rows = ["L", "R", "G", "B"];
        let colors = [[225u8, 225, 225], [240, 110, 110], [130, 230, 130], [120, 160, 250]];
        if self.building {
            for (i, name) in rows.iter().enumerate() {
                let row = Rect { x: plot.x, y: plot.y + row_h * i as f32, w: plot.w, h: row_h - 3.0 * s };
                self.fill(row, [255, 255, 255, 14]);
                self.text(name, plot.x - 16.0 * s, row.y + row.h * 0.5 + 6.0 * s, 16.0 * s, colors[i]);
            }
            for (v, label) in DECADES {
                let x = plot.x + log_axis(v) as f32 * plot.w;
                self.fill(Rect { x, y: plot.y, w: s.max(1.0), h: plot.h - label_h }, [255, 255, 255, 50]);
                self.text(label, x - 8.0 * s, plot.y + plot.h - 2.0 * s, 16.0 * s, [200, 170, 90]);
            }
            return;
        }
        let Some(hist) = d.nits.and_then(|n| n.scopes.as_ref()).and_then(|sc| sc.hist.as_deref()) else { return };
        for (i, color) in colors.iter().enumerate() {
            let bins = &hist[i * HIST_BINS..(i + 1) * HIST_BINS];
            // Height relative to the tallest bin, square-root scaled so small bins stay visible.
            let max = bins.iter().copied().max().unwrap_or(0).max(1) as f32;
            let row = Rect { x: plot.x, y: plot.y + row_h * i as f32, w: plot.w, h: row_h - 3.0 * s };
            let n = row.w.max(1.0) as usize;
            for c in 0..n {
                let b = (c * HIST_BINS / n).min(HIST_BINS - 1);
                let v = (bins[b] as f32 / max).sqrt();
                if v <= 0.0 {
                    continue;
                }
                let bar = v * row.h;
                self.fill_solid(
                    Rect { x: row.x + c as f32, y: row.y + row.h - bar, w: 1.0, h: bar },
                    [color[0], color[1], color[2], 210],
                );
            }
        }
    }

    /// Square plot area centred in the widget below the title.
    fn square_plot(&mut self, r: Rect, s: f32, title: &str) -> Rect {
        let area = self.titled_plot(r, s, title, 0.0);
        let side = area.w.min(area.h);
        Rect { x: area.x + (area.w - side) / 2.0, y: area.y + (area.h - side) / 2.0, w: side, h: side }
    }

    fn cie(&mut self, r: Rect, s: f32, d: &FrameData) {
        let plot = self.square_plot(r, s, "CIE 1931 xy");
        let to_px = |x: f32, y: f32| (plot.x + x / CIE_X_RANGE * plot.w, plot.y + plot.h - y / CIE_Y_RANGE * plot.h);
        if self.building {
            self.fill(plot, [255, 255, 255, 10]);
            let locus: Vec<(f32, f32)> = LOCUS.iter().map(|&(x, y)| to_px(x, y)).collect();
            self.polyline(&locus, true, s, [255, 255, 255, 110]);
            for (name, prim, rgba) in GAMUTS {
                let pts: Vec<(f32, f32)> = prim.iter().map(|&(x, y)| to_px(x, y)).collect();
                self.polyline(&pts, true, s * 1.2, rgba);
                self.text(name, pts[1].0 + 6.0 * s, pts[1].1 + 4.0 * s, 15.0 * s, [rgba[0], rgba[1], rgba[2]]);
            }
            let (wx, wy) = to_px(0.3127, 0.3290);
            self.fill(Rect { x: wx - 2.0 * s, y: wy - 2.0 * s, w: 4.0 * s, h: 4.0 * s }, [255, 255, 255, 200]);
            return;
        }
        let Some(bins) = d.nits.and_then(|n| n.scopes.as_ref()).and_then(|sc| sc.cie.as_deref()) else { return };
        let colors = std::mem::take(&mut self.cie_colors);
        self.density(plot, bins, &colors, |bx, by| {
            (bx as f32 / XY_BINS as f32, 1.0 - (by + 1) as f32 / XY_BINS as f32)
        });
        self.cie_colors = colors;
    }

    fn vectorscope(&mut self, r: Rect, s: f32, d: &FrameData) {
        let plot = self.square_plot(r, s, "VECTORSCOPE");
        if self.building {
            let (cx, cy, rad) = (plot.x + plot.w / 2.0, plot.y + plot.h / 2.0, plot.w / 2.0);
            let circle: Vec<(f32, f32)> = (0..96)
                .map(|i| {
                    let a = i as f32 / 96.0 * std::f32::consts::TAU;
                    (cx + rad * a.cos(), cy + rad * a.sin())
                })
                .collect();
            self.polyline(&circle, true, s, [255, 255, 255, 110]);
            self.polyline(&[(cx - rad, cy), (cx + rad, cy)], false, s, [255, 255, 255, 60]);
            self.polyline(&[(cx, cy - rad), (cx, cy + rad)], false, s, [255, 255, 255, 60]);
            // Skin-tone line, 123 degrees from the +Cb axis.
            let a = 123f32.to_radians();
            self.polyline(&[(cx, cy), (cx + rad * a.cos(), cy - rad * a.sin())], false, s, [255, 255, 255, 140]);
            return;
        }
        let Some(bins) = d.nits.and_then(|n| n.scopes.as_ref()).and_then(|sc| sc.vec.as_deref()) else { return };
        let colors = std::mem::take(&mut self.vec_colors);
        // Bin (Cb, Cr): Cb to the right, Cr up.
        self.density(plot, bins, &colors, |bx, by| {
            (bx as f32 / XY_BINS as f32, 1.0 - (by + 1) as f32 / XY_BINS as f32)
        });
        self.vec_colors = colors;
    }

    /// Draws a 2D count histogram, each bin tinted by `colors` and brightened by log density.
    /// `place` maps a bin to the top-left of its cell as fractions of the plot.
    fn density(&mut self, plot: Rect, bins: &[u32], colors: &[[u8; 3]], place: impl Fn(usize, usize) -> (f32, f32)) {
        let max = bins.iter().copied().max().unwrap_or(0);
        if max == 0 {
            return;
        }
        // Normalize against a quarter of the peak so a dominant colour (e.g. white) doesn't hide the rest.
        let reference = ((max as f32) * 0.25).max(2.0).ln_1p();
        let cell = (plot.w / XY_BINS as f32).max(1.0);
        for by in 0..XY_BINS {
            for bx in 0..XY_BINS {
                let c = bins[by * XY_BINS + bx];
                if c == 0 {
                    continue;
                }
                let a = ((c as f32).ln_1p() / reference).clamp(0.15, 1.0);
                let (fx, fy) = place(bx, by);
                let col = colors[by * XY_BINS + bx];
                let rgba = [
                    (col[0] as f32 * a) as u8,
                    (col[1] as f32 * a) as u8,
                    (col[2] as f32 * a) as u8,
                    (a * 255.0) as u8,
                ];
                self.fill_solid(Rect { x: plot.x + fx * plot.w, y: plot.y + fy * plot.h, w: cell, h: cell }, rgba);
            }
        }
    }

    // ---- primitives ----------------------------------------------------------

    /// Plot background, horizontal guides and labels (left of the plot when `labels_left`).
    fn plot_frame(
        &mut self,
        plot: Rect,
        s: f32,
        map: &dyn Fn(f64) -> f64,
        guides: &[(f64, String)],
        labels_left: bool,
    ) {
        self.fill(plot, [255, 255, 255, 18]);
        for (v, label) in guides {
            let y = plot.y + plot.h - map(*v) as f32 * plot.h;
            self.fill(Rect { x: plot.x, y: y - s.max(1.0) / 2.0, w: plot.w, h: s.max(1.0) }, [255, 255, 255, 55]);
            if labels_left {
                self.text(label, plot.x - 34.0 * s, y + 6.0 * s, 16.0 * s, [200, 170, 90]);
            } else {
                self.text(label, plot.x + 4.0 * s, y - 3.0 * s, 16.0 * s, [170, 170, 170]);
            }
        }
    }

    /// Scrolling line graphs of the last `seconds`, newest sample at the right edge.
    /// Drawn as one vertical span per pixel column (min..max of the samples landing
    /// there, joined to the previous column): far cheaper than stroking a 1000+
    /// point anti-aliased path every frame, and it looks the same.
    fn series(
        &mut self,
        plot: Rect,
        s: f32,
        d: &FrameData,
        seconds: f32,
        lines: &[(&[f64], [u8; 3])],
        map: &dyn Fn(f64) -> f64,
    ) {
        let span = (seconds.max(0.5) as f64 * d.capture_fps).round().max(2.0) as usize;
        let start = d.frame.saturating_sub(span);
        let thickness = (2.5 * s).max(1.0);
        let step_cols = (plot.w / span as f32).ceil().max(1.0);
        let ypos = |v: f64| plot.y + plot.h - map(v) as f32 * plot.h;
        for (values, rgb) in lines {
            let last = d.frame.min(values.len().saturating_sub(1));
            if values.is_empty() || start > last {
                continue;
            }
            let mut col: Option<f32> = None;
            let (mut lo, mut hi) = (0f32, 0f32);
            let mut last_y: Option<f32> = None;
            let rgba = [rgb[0], rgb[1], rgb[2], 255];
            for i in start..=last {
                if values[i].is_nan() {
                    last_y = None;
                    continue;
                }
                let px = (plot.x + plot.w - ((d.frame - i) as f32 / span as f32) * plot.w).floor();
                let py = ypos(values[i]);
                if col == Some(px) {
                    lo = lo.min(py);
                    hi = hi.max(py);
                } else {
                    if let Some(c) = col {
                        self.fill_solid(
                            Rect { x: c, y: lo - thickness / 2.0, w: step_cols, h: hi - lo + thickness },
                            rgba,
                        );
                    }
                    col = Some(px);
                    (lo, hi) = (py, py);
                    if let Some(l) = last_y {
                        (lo, hi) = (lo.min(l), hi.max(l));
                    }
                }
                last_y = Some(py);
            }
            if let Some(c) = col {
                self.fill_solid(Rect { x: c, y: lo - thickness / 2.0, w: step_cols, h: hi - lo + thickness }, rgba);
            }
        }
    }

    /// Integer pixel bounds of `r` clipped to the image, plus how much was cut off.
    fn clip(&self, r: Rect) -> Option<Clip> {
        let (pw, ph) = (self.pixmap.width() as i64, self.pixmap.height() as i64);
        let (x0, y0) = (r.x.round() as i64, r.y.round() as i64);
        let (x1, y1) = ((r.x + r.w).round() as i64, (r.y + r.h).round() as i64);
        let (cx0, cy0, cx1, cy1) = (x0.max(0), y0.max(0), x1.min(pw), y1.min(ph));
        if cx1 <= cx0 || cy1 <= cy0 {
            return None;
        }
        Some(Clip {
            x: cx0 as usize,
            y: cy0 as usize,
            w: (cx1 - cx0) as usize,
            h: (cy1 - cy0) as usize,
            skip_x: (cx0 - x0) as usize,
            skip_y: (cy0 - y0) as usize,
            full_w: (x1 - x0).max(1) as usize,
            full_h: (y1 - y0).max(1) as usize,
        })
    }

    /// Alpha-blends a premultiplied colour over a rectangle.
    fn fill(&mut self, r: Rect, rgba: [u8; 4]) {
        let Some(c) = self.clip(r) else { return };
        let pw = self.pixmap.width() as usize;
        let a = rgba[3] as f32 / 255.0;
        let src = [rgba[0] as f32 * a, rgba[1] as f32 * a, rgba[2] as f32 * a, rgba[3] as f32];
        let data = self.pixmap.data_mut();
        for y in c.y..c.y + c.h {
            for x in c.x..c.x + c.w {
                let o = (y * pw + x) * 4;
                for k in 0..4 {
                    data[o + k] = (src[k] + data[o + k] as f32 * (1.0 - a)).round() as u8;
                }
            }
        }
    }

    /// Overwrites a rectangle with an already-premultiplied colour.
    fn fill_solid(&mut self, r: Rect, rgba: [u8; 4]) {
        let Some(c) = self.clip(r) else { return };
        let pw = self.pixmap.width() as usize;
        let data = self.pixmap.data_mut();
        for y in c.y..c.y + c.h {
            let row = (y * pw + c.x) * 4;
            for px in data[row..row + c.w * 4].as_chunks_mut::<4>().0 {
                px.copy_from_slice(&rgba);
            }
        }
    }

    fn polyline(&mut self, pts: &[(f32, f32)], closed: bool, width: f32, rgba: [u8; 4]) {
        let mut pb = PathBuilder::new();
        for (i, &(x, y)) in pts.iter().enumerate() {
            if i == 0 {
                pb.move_to(x, y);
            } else {
                pb.line_to(x, y);
            }
        }
        if closed {
            pb.close();
        }
        let Some(path) = pb.finish() else { return };
        let mut paint = Paint::default();
        paint.set_color_rgba8(rgba[0], rgba[1], rgba[2], rgba[3]);
        paint.anti_alias = true;
        let stroke = Stroke { width: width.max(1.0), ..Default::default() };
        self.pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
    }

    /// Draws text with its baseline at `y`, blending into the premultiplied pixmap.
    fn text(&mut self, text: &str, x: f32, y: f32, px: f32, rgb: [u8; 3]) {
        let font = self.font.as_scaled(px);
        let (pw, ph) = (self.pixmap.width() as i32, self.pixmap.height() as i32);
        let data = self.pixmap.data_mut();
        let mut caret = x;
        let mut prev = None;
        for ch in text.chars() {
            let id = font.glyph_id(ch);
            if let Some(p) = prev {
                caret += font.kern(p, id);
            }
            let glyph = id.with_scale_and_position(px, ab_glyph::point(caret, y));
            caret += font.h_advance(id);
            prev = Some(id);
            let Some(outlined) = self.font.outline_glyph(glyph) else { continue };
            let b = outlined.px_bounds();
            outlined.draw(|gx, gy, cov| {
                let (ix, iy) = (b.min.x as i32 + gx as i32, b.min.y as i32 + gy as i32);
                if ix < 0 || iy < 0 || ix >= pw || iy >= ph {
                    return;
                }
                let o = ((iy * pw + ix) * 4) as usize;
                let a = cov.clamp(0.0, 1.0);
                for c in 0..3 {
                    data[o + c] = (rgb[c] as f32 * a + data[o + c] as f32 * (1.0 - a)).round() as u8;
                }
                data[o + 3] = (255.0 * a + data[o + 3] as f32 * (1.0 - a)).round() as u8;
            });
        }
    }
}

/// Additively brightens one premultiplied pixel.
fn add_px(data: &mut [u8], o: usize, add: [u8; 3]) {
    for c in 0..3 {
        data[o + c] = data[o + c].saturating_add(add[c]);
    }
    data[o + 3] = data[o + 3].max(add[0].max(add[1]).max(add[2]));
}

#[derive(Clone, Copy, Debug)]
struct Rect {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

struct Clip {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    /// Pixels of the unclipped rectangle cut off at the left / top.
    skip_x: usize,
    skip_y: usize,
    full_w: usize,
    full_h: usize,
}

/// Display colour of each CIE xy bin: the chromaticity itself, at full brightness, in sRGB.
fn cie_colors() -> Vec<[u8; 3]> {
    let mut out = vec![[0u8; 3]; XY_BINS * XY_BINS];
    for by in 0..XY_BINS {
        for bx in 0..XY_BINS {
            let x = (bx as f32 + 0.5) / XY_BINS as f32 * CIE_X_RANGE;
            let y = ((by as f32 + 0.5) / XY_BINS as f32 * CIE_Y_RANGE).max(1e-3);
            let (cx, cy, cz) = (x / y, 1.0, (1.0 - x - y) / y);
            let r = 3.2406 * cx - 1.5372 * cy - 0.4986 * cz;
            let g = -0.9689 * cx + 1.8758 * cy + 0.0415 * cz;
            let b = 0.0557 * cx - 0.2040 * cy + 1.0570 * cz;
            out[by * XY_BINS + bx] = normalized_srgb([r, g, b]);
        }
    }
    out
}

/// Display colour of each vectorscope bin: that chroma at mid luma (BT.709).
fn vec_colors() -> Vec<[u8; 3]> {
    let mut out = vec![[0u8; 3]; XY_BINS * XY_BINS];
    for by in 0..XY_BINS {
        for bx in 0..XY_BINS {
            let cb = (bx as f32 + 0.5) / XY_BINS as f32 - 0.5;
            let cr = (by as f32 + 0.5) / XY_BINS as f32 - 0.5;
            let y = 0.6;
            let rgb = [y + 1.5748 * cr, y - 0.1873 * cb - 0.4681 * cr, y + 1.8556 * cb];
            out[by * XY_BINS + bx] = normalized_srgb(rgb.map(|v| v.max(0.0).powf(2.2)));
        }
    }
    out
}

/// Linear RGB -> 8-bit sRGB-ish colour with the brightest channel at full scale.
fn normalized_srgb(rgb: [f32; 3]) -> [u8; 3] {
    let rgb = rgb.map(|v| v.max(0.0));
    let m = rgb[0].max(rgb[1]).max(rgb[2]).max(1e-6);
    rgb.map(|v| ((v / m).powf(1.0 / 2.2) * 255.0).round() as u8)
}
