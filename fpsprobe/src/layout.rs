//! Overlay layout: which graphs are shown and where.
//!
//! Widget rectangles are stored as fractions of the output canvas, so one layout
//! works for a 1080p overlay, a 4K source video and a 1440p graphs-only export.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::nits::{SCOPE_CIE, SCOPE_HIST, SCOPE_VEC, SCOPE_WAVE};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WidgetKind {
    FpsCounter,
    FpsGraph,
    FrametimeGraph,
    NitsReadout,
    NitsGraph,
    Waveform,
    Histogram,
    Cie,
    Vectorscope,
}

impl WidgetKind {
    pub const ALL: [WidgetKind; 9] = [
        WidgetKind::FpsCounter,
        WidgetKind::FpsGraph,
        WidgetKind::FrametimeGraph,
        WidgetKind::NitsReadout,
        WidgetKind::NitsGraph,
        WidgetKind::Waveform,
        WidgetKind::Histogram,
        WidgetKind::Cie,
        WidgetKind::Vectorscope,
    ];

    pub fn label(self) -> &'static str {
        match self {
            WidgetKind::FpsCounter => "FPS counter",
            WidgetKind::FpsGraph => "FPS graph",
            WidgetKind::FrametimeGraph => "Frame-time graph",
            WidgetKind::NitsReadout => "Nits readout (peak / avg)",
            WidgetKind::NitsGraph => "Nits over time",
            WidgetKind::Waveform => "HDR waveform (RGB nits)",
            WidgetKind::Histogram => "HDR histogram (L/R/G/B)",
            WidgetKind::Cie => "CIE chromaticity",
            WidgetKind::Vectorscope => "Vectorscope",
        }
    }

    /// Whether the widget needs the nits/scopes measurement pass.
    pub fn needs_nits(self) -> bool {
        !matches!(self, WidgetKind::FpsCounter | WidgetKind::FpsGraph | WidgetKind::FrametimeGraph)
    }

    pub fn scope_flag(self) -> u32 {
        match self {
            WidgetKind::Waveform => SCOPE_WAVE,
            WidgetKind::Histogram => SCOPE_HIST,
            WidgetKind::Cie => SCOPE_CIE,
            WidgetKind::Vectorscope => SCOPE_VEC,
            _ => 0,
        }
    }

    /// Whether the widget scrolls a time history (and so has a history length).
    pub fn has_history(self) -> bool {
        matches!(self, WidgetKind::FpsGraph | WidgetKind::FrametimeGraph | WidgetKind::NitsGraph)
    }

    /// Design size in pixels on a 1080p canvas; content scales with the actual box.
    pub fn base_size(self) -> (f32, f32) {
        match self {
            WidgetKind::FpsCounter => (420.0, 90.0),
            WidgetKind::FpsGraph | WidgetKind::FrametimeGraph => (680.0, 140.0),
            WidgetKind::NitsReadout => (680.0, 80.0),
            WidgetKind::NitsGraph => (680.0, 120.0),
            WidgetKind::Waveform => (680.0, 260.0),
            WidgetKind::Histogram => (680.0, 320.0),
            WidgetKind::Cie | WidgetKind::Vectorscope => (400.0, 400.0),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Widget {
    pub kind: WidgetKind,
    pub enabled: bool,
    /// Rectangle as fractions of the canvas (0..1).
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    /// Background panel opacity (0 = none, 1 = solid black).
    pub bg_opacity: f32,
    /// Seconds of history for scrolling graphs.
    pub history_seconds: f32,
    /// Locked graphs can't be moved or resized in the editor.
    #[serde(default)]
    pub locked: bool,
}

impl Widget {
    /// Creates a widget from a rectangle in 1920x1080 reference pixels.
    fn at(kind: WidgetKind, enabled: bool, x: f32, y: f32, w: f32, h: f32) -> Self {
        Self {
            kind,
            enabled,
            x: x / 1920.0,
            y: y / 1080.0,
            w: w / 1920.0,
            h: h / 1080.0,
            bg_opacity: 0.78,
            history_seconds: 10.0,
            locked: false,
        }
    }

    /// Rectangle in pixels on a canvas of the given size.
    pub fn rect_px(&self, canvas_w: f32, canvas_h: f32) -> (f32, f32, f32, f32) {
        (self.x * canvas_w, self.y * canvas_h, self.w * canvas_w, self.h * canvas_h)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Layout {
    pub widgets: Vec<Widget>,
}

impl Default for Layout {
    fn default() -> Self {
        Self::preset_fps()
    }
}

impl Layout {
    /// Every widget kind appears exactly once; this fills in any missing (disabled).
    fn complete(mut self) -> Self {
        for kind in WidgetKind::ALL {
            if !self.widgets.iter().any(|w| w.kind == kind) {
                let (bw, bh) = kind.base_size();
                self.widgets.push(Widget::at(kind, false, 40.0, 40.0, bw, bh));
            }
        }
        self
    }

    /// Digital Foundry style: counter top-left, FPS and frame-time graphs along the bottom.
    pub fn preset_fps() -> Self {
        Self {
            widgets: vec![
                Widget::at(WidgetKind::FpsCounter, true, 24.0, 24.0, 420.0, 90.0),
                Widget::at(WidgetKind::FpsGraph, true, 24.0, 760.0, 900.0, 145.0),
                Widget::at(WidgetKind::FrametimeGraph, true, 24.0, 915.0, 900.0, 141.0),
            ],
        }
        .complete()
    }

    /// FPS on the left, nits readout / waveform / nits graph on the right (the CLI's --nits look).
    pub fn preset_fps_hdr() -> Self {
        Self {
            widgets: vec![
                Widget::at(WidgetKind::FpsCounter, true, 24.0, 676.0, 720.0, 90.0),
                Widget::at(WidgetKind::FpsGraph, true, 24.0, 766.0, 720.0, 145.0),
                Widget::at(WidgetKind::FrametimeGraph, true, 24.0, 911.0, 720.0, 145.0),
                Widget::at(WidgetKind::NitsReadout, true, 760.0, 676.0, 720.0, 80.0),
                Widget::at(WidgetKind::Waveform, true, 760.0, 756.0, 720.0, 190.0),
                Widget::at(WidgetKind::NitsGraph, true, 760.0, 946.0, 720.0, 110.0),
            ],
        }
        .complete()
    }

    /// The four HDR scopes filling the canvas, as in HDRScopes.
    pub fn preset_hdr_grid() -> Self {
        Self {
            widgets: vec![
                Widget::at(WidgetKind::Waveform, true, 0.0, 0.0, 1340.0, 536.0),
                Widget::at(WidgetKind::Cie, true, 1346.0, 0.0, 574.0, 536.0),
                Widget::at(WidgetKind::Histogram, true, 0.0, 544.0, 1340.0, 536.0),
                Widget::at(WidgetKind::Vectorscope, true, 1346.0, 544.0, 574.0, 536.0),
            ],
        }
        .complete()
    }

    pub fn enabled(&self) -> impl Iterator<Item = &Widget> {
        self.widgets.iter().filter(|w| w.enabled)
    }

    pub fn needs_nits(&self) -> bool {
        self.enabled().any(|w| w.kind.needs_nits())
    }

    pub fn scope_flags(&self) -> u32 {
        self.enabled().fold(0, |f, w| f | w.kind.scope_flag())
    }

    /// Bounding box of enabled widgets in canvas pixels, as (x, y, w, h).
    pub fn bounds_px(&self, canvas_w: f32, canvas_h: f32) -> Option<(usize, usize, usize, usize)> {
        let mut it = self.enabled().map(|w| w.rect_px(canvas_w, canvas_h));
        let first = it.next()?;
        let (mut x0, mut y0, mut x1, mut y1) = (first.0, first.1, first.0 + first.2, first.1 + first.3);
        for (x, y, w, h) in it {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x + w);
            y1 = y1.max(y + h);
        }
        let x0 = x0.floor().clamp(0.0, canvas_w - 2.0) as usize;
        let y0 = y0.floor().clamp(0.0, canvas_h - 2.0) as usize;
        let x1 = x1.ceil().clamp(x0 as f32 + 2.0, canvas_w) as usize;
        let y1 = y1.ceil().clamp(y0 as f32 + 2.0, canvas_h) as usize;
        Some((x0, y0, x1 - x0, y1 - y0))
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let layout: Layout = serde_json::from_str(&text).context("invalid layout file")?;
        Ok(layout.complete())
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::write(path, serde_json::to_string_pretty(self)?).with_context(|| format!("writing {}", path.display()))
    }
}

// ---- snapping for the layout editor ---------------------------------------------------

/// A rectangle in canvas pixels: (x, y, w, h).
pub type RectPx = (f32, f32, f32, f32);

/// How a moved or resized rectangle should snap. Distances are in canvas pixels.
#[derive(Clone, Copy, Debug)]
pub struct Snap {
    /// Snap to a grid of this spacing.
    pub grid: Option<f32>,
    /// Snap edges and centres to other rectangles and to the canvas edges and centre.
    pub align: bool,
    /// How close (in canvas pixels) an edge must be to an alignment line to snap.
    pub threshold: f32,
}

/// Result of snapping: the adjusted rectangle and the alignment lines it locked onto
/// (canvas x positions and y positions), for drawing guides.
#[derive(Clone, Debug, Default)]
pub struct Snapped {
    pub rect: RectPx,
    pub guides_x: Vec<f32>,
    pub guides_y: Vec<f32>,
}

/// Lines other rectangles and the canvas offer along one axis: edges and centres.
fn alignment_lines(others: &[RectPx], canvas_len: f32, vertical: bool) -> Vec<f32> {
    let mut lines = vec![0.0, canvas_len / 2.0, canvas_len];
    for &(x, y, w, h) in others {
        let (start, len) = if vertical { (y, h) } else { (x, w) };
        lines.extend([start, start + len / 2.0, start + len]);
    }
    lines
}

/// Smallest shift that puts one of `anchors` on one of `lines`, if within `threshold`.
fn nearest_shift(anchors: &[f32], lines: &[f32], threshold: f32) -> Option<(f32, f32)> {
    anchors
        .iter()
        .flat_map(|&a| lines.iter().map(move |&l| (l - a, l)))
        .filter(|(d, _)| d.abs() <= threshold)
        .min_by(|a, b| a.0.abs().total_cmp(&b.0.abs()))
}

fn round_to(v: f32, step: f32) -> f32 {
    (v / step).round() * step
}

/// Snaps a rectangle being moved. Alignment (if enabled and something is close)
/// takes priority over the grid on each axis. The result stays inside the canvas.
pub fn snap_move(rect: RectPx, others: &[RectPx], canvas: (f32, f32), snap: Snap) -> Snapped {
    let (mut x, mut y, w, h) = rect;
    let mut out = Snapped::default();
    for vertical in [false, true] {
        let (pos, len, canvas_len) = if vertical { (y, h, canvas.1) } else { (x, w, canvas.0) };
        let mut new = pos;
        let mut aligned = None;
        if snap.align {
            let lines = alignment_lines(others, canvas_len, vertical);
            aligned = nearest_shift(&[pos, pos + len / 2.0, pos + len], &lines, snap.threshold);
        }
        match (aligned, snap.grid) {
            (Some((shift, line)), _) => {
                new = pos + shift;
                if vertical { out.guides_y.push(line) } else { out.guides_x.push(line) }
            }
            (None, Some(g)) if g > 0.0 => new = round_to(pos, g),
            _ => {}
        }
        let new = new.clamp(0.0, (canvas_len - len).max(0.0));
        if vertical { y = new } else { x = new }
    }
    out.rect = (x, y, w, h);
    out
}

/// Snaps a rectangle being resized from its bottom-right corner (the top-left stays put).
pub fn snap_resize(rect: RectPx, others: &[RectPx], canvas: (f32, f32), snap: Snap, min_size: f32) -> Snapped {
    let (x, y, mut w, mut h) = rect;
    let mut out = Snapped::default();
    for vertical in [false, true] {
        let (start, len, canvas_len) = if vertical { (y, h, canvas.1) } else { (x, w, canvas.0) };
        let mut end = start + len;
        let mut aligned = None;
        if snap.align {
            let lines = alignment_lines(others, canvas_len, vertical);
            aligned = nearest_shift(&[end], &lines, snap.threshold);
        }
        match (aligned, snap.grid) {
            (Some((shift, line)), _) => {
                end += shift;
                if vertical { out.guides_y.push(line) } else { out.guides_x.push(line) }
            }
            (None, Some(g)) if g > 0.0 => end = round_to(end, g),
            _ => {}
        }
        let new_len = (end - start).clamp(min_size, (canvas_len - start).max(min_size));
        if vertical { h = new_len } else { w = new_len }
    }
    out.rect = (x, y, w, h);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const CANVAS: (f32, f32) = (1920.0, 1080.0);

    #[test]
    fn grid_rounds_position() {
        let s = Snap { grid: Some(16.0), align: false, threshold: 0.0 };
        let r = snap_move((101.0, 57.0, 300.0, 100.0), &[], CANVAS, s).rect;
        assert_eq!((r.0, r.1), (96.0, 64.0));
    }

    #[test]
    fn aligns_left_edge_to_another_graph() {
        let s = Snap { grid: None, align: true, threshold: 10.0 };
        let other = (500.0, 200.0, 400.0, 100.0);
        let snapped = snap_move((506.0, 600.0, 300.0, 100.0), &[other], CANVAS, s);
        assert_eq!(snapped.rect.0, 500.0);
        assert_eq!(snapped.guides_x, vec![500.0]);
    }

    #[test]
    fn aligns_centre_to_canvas_centre() {
        let s = Snap { grid: None, align: true, threshold: 10.0 };
        // Width 400, x 757 -> centre 957, canvas centre 960.
        let r = snap_move((757.0, 300.0, 400.0, 100.0), &[], CANVAS, s).rect;
        assert_eq!(r.0, 760.0);
    }

    #[test]
    fn far_from_lines_falls_back_to_grid() {
        let s = Snap { grid: Some(32.0), align: true, threshold: 5.0 };
        let r = snap_move((300.0, 300.0, 100.0, 100.0), &[], CANVAS, s).rect;
        assert_eq!((r.0, r.1), (288.0, 288.0));
    }

    #[test]
    fn move_stays_inside_canvas() {
        let s = Snap { grid: Some(16.0), align: false, threshold: 0.0 };
        let r = snap_move((1900.0, -20.0, 300.0, 100.0), &[], CANVAS, s).rect;
        assert_eq!((r.0, r.1), (1620.0, 0.0));
    }

    #[test]
    fn resize_snaps_bottom_right_corner() {
        let s = Snap { grid: Some(16.0), align: false, threshold: 0.0 };
        let r = snap_resize((100.0, 100.0, 303.0, 205.0), &[], CANVAS, s, 20.0).rect;
        assert_eq!((r.0, r.1, r.2, r.3), (100.0, 100.0, 300.0, 204.0));
    }

    #[test]
    fn resize_aligns_right_edge_to_other_graph() {
        let s = Snap { grid: None, align: true, threshold: 10.0 };
        let other = (900.0, 0.0, 300.0, 100.0); // right edge at 1200
        let r = snap_resize((100.0, 400.0, 1094.0, 100.0), &[other], CANVAS, s, 20.0).rect;
        assert_eq!(r.2, 1100.0);
    }
}
