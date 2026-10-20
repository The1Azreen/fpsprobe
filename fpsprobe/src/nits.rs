//! GPU measurement of absolute luminance (nits) and HDR scopes for HDR10, HLG and SDR video.
//!
//! Frames arrive as 10-bit 4:2:0 (yuv420p10le) and are converted per pixel to
//! linear RGB in nits on the GPU. Per frame we report:
//! - peak / average of max(R,G,B): the quantities MaxCLL and MaxFALL are built from
//! - peak / average luminance (Y)
//! - optionally scope data: RGB waveform, L/R/G/B histogram, CIE xy, vectorscope.

use anyhow::{Context, Result, anyhow};
use bytemuck::{Pod, Zeroable};

pub const WAVE_COLS: usize = 512;
pub const WAVE_ROWS: usize = 256;
pub const HIST_BINS: usize = 256;
/// Bins per axis of the CIE xy (x 0..0.8, y 0..0.9) and vectorscope (Cb/Cr -0.5..0.5) plots.
pub const XY_BINS: usize = 256;
const BATCH: usize = 8;

pub const SCOPE_WAVE: u32 = 1;
pub const SCOPE_HIST: u32 = 2;
pub const SCOPE_CIE: u32 = 4;
pub const SCOPE_VEC: u32 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transfer {
    Pq,
    Sdr,
    Hlg,
}

#[derive(Clone, Copy, Debug)]
pub struct ColorInfo {
    pub transfer: Transfer,
    pub bt2020: bool,
    pub full_range: bool,
}

impl ColorInfo {
    /// Interprets ffprobe's color_transfer / color_space / color_range strings.
    pub fn from_probe(transfer: &str, space: &str, range: &str) -> Self {
        let transfer = match transfer {
            "smpte2084" => Transfer::Pq,
            "arib-std-b67" => Transfer::Hlg,
            _ => Transfer::Sdr,
        };
        // Untagged HDR is assumed to be BT.2020, as HDR10/HLG always are.
        let bt2020 = space.starts_with("bt2020") || (transfer != Transfer::Sdr && space == "unknown");
        Self { transfer, bt2020, full_range: range == "pc" }
    }

    pub fn is_hdr(&self) -> bool {
        self.transfer != Transfer::Sdr
    }

    pub fn describe(&self) -> String {
        let t = match self.transfer {
            Transfer::Pq => "HDR10 / PQ",
            Transfer::Hlg => "HLG",
            Transfer::Sdr => "SDR",
        };
        let m = if self.bt2020 { "BT.2020" } else { "BT.709" };
        let r = if self.full_range { "full" } else { "limited" };
        format!("{t}, {m}, {r} range")
    }
}

/// Per-frame scope data; each field is present when that scope was requested.
#[derive(Clone, Debug, Default)]
pub struct ScopeData {
    /// [R,G,B][row][col], rows bottom (0 nits) to top (10000 nits).
    pub wave: Option<Vec<u32>>,
    /// [L,R,G,B][bin], bins on the same log-nits axis as the waveform.
    pub hist: Option<Vec<u32>>,
    /// [y bin][x bin] chromaticity counts.
    pub cie: Option<Vec<u32>>,
    /// [Cr bin][Cb bin] chroma counts.
    pub vec: Option<Vec<u32>>,
}

#[derive(Clone, Debug, Default)]
pub struct NitsFrame {
    /// Brightest max(R,G,B) in the frame (the per-frame MaxCLL candidate).
    pub peak: f32,
    /// Frame average of max(R,G,B) (the per-frame MaxFALL candidate).
    pub avg: f32,
    pub peak_lum: f32,
    pub avg_lum: f32,
    pub scopes: Option<ScopeData>,
}

/// Where each scope lives inside one frame's slot of the scope buffer (in u32s).
#[derive(Clone, Copy, Debug, Default)]
struct ScopeLayout {
    wave: Option<usize>,
    hist: Option<usize>,
    cie: Option<usize>,
    vec: Option<usize>,
    words: usize,
}

impl ScopeLayout {
    fn new(flags: u32) -> Self {
        let mut l = ScopeLayout::default();
        let mut add = |flag: u32, size: usize| {
            (flags & flag != 0).then(|| {
                let off = l.words;
                l.words += size;
                off
            })
        };
        l.wave = add(SCOPE_WAVE, 3 * WAVE_ROWS * WAVE_COLS);
        l.hist = add(SCOPE_HIST, 4 * HIST_BINS);
        l.cie = add(SCOPE_CIE, XY_BINS * XY_BINS);
        l.vec = add(SCOPE_VEC, XY_BINS * XY_BINS);
        l
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    width: u32,
    height: u32,
    u_off: u32,
    v_off: u32,
    wave_cols: u32,
    wave_rows: u32,
    scope_flags: u32,
    wg_base: u32,
    scope_base: u32,
    hist_off: u32,
    cie_off: u32,
    vec_off: u32,
    transfer: u32,
    full_range: u32,
    bt2020: u32,
    _pad0: u32,
    sdr_white: f32,
    cr_r: f32,
    cb_g: f32,
    cr_g: f32,
    cb_b: f32,
    kr: f32,
    kg: f32,
    kb: f32,
}

pub struct NitsGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    frame: wgpu::Buffer,
    params: wgpu::Buffer,
    params_stride: u64,
    wg_stats: wgpu::Buffer,
    scopes: wgpu::Buffer,
    wg_readback: wgpu::Buffer,
    scope_readback: wgpu::Buffer,
    base: Params,
    groups: (u32, u32),
    wg_count: usize,
    layout: ScopeLayout,
    pixels: usize,
    pending: usize,
}

impl NitsGpu {
    /// `scope_flags` is a combination of SCOPE_* (0 for statistics only).
    pub fn new(width: usize, height: usize, color: ColorInfo, sdr_white: f32, scope_flags: u32) -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .context("no GPU adapter found")?;

        let layout = ScopeLayout::new(scope_flags);
        let samples = width * height * 3 / 2;
        let frame_bytes = (samples * 2) as u64;
        // A 4-byte placeholder keeps the binding valid when no scopes are requested.
        let scope_bytes = (layout.words.max(1) * BATCH * 4) as u64;
        let need = frame_bytes.max(scope_bytes);
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size =
            limits.max_storage_buffer_binding_size.max(need).min(adapter.limits().max_storage_buffer_binding_size);
        limits.max_buffer_size = limits.max_buffer_size.max(need).min(adapter.limits().max_buffer_size);
        if limits.max_storage_buffer_binding_size < need {
            return Err(anyhow!("frame of {frame_bytes} bytes exceeds GPU storage buffer limit"));
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("fpsprobe-nits"),
            required_limits: limits,
            ..Default::default()
        }))?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nits"),
            source: wgpu::ShaderSource::Wgsl(include_str!("nits.wgsl").into()),
        });
        let buffer_entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nits"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Params>() as u64),
                    },
                    count: None,
                },
                buffer_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
                buffer_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nits"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nits"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let groups = ((width / 2).div_ceil(16) as u32, (height / 2).div_ceil(16) as u32);
        let wg_count = (groups.0 * groups.1) as usize;
        let buf = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
        };
        use wgpu::BufferUsages as U;
        let frame = buf("nits-frame", frame_bytes, U::STORAGE | U::COPY_DST);
        let align = device.limits().min_uniform_buffer_offset_alignment as u64;
        let params_stride = (std::mem::size_of::<Params>() as u64).div_ceil(align) * align;
        let params = buf("nits-params", params_stride * BATCH as u64, U::UNIFORM | U::COPY_DST);
        let wg_bytes = (wg_count * 4 * BATCH * 4) as u64;
        let wg_stats = buf("nits-wg", wg_bytes, U::STORAGE | U::COPY_SRC);
        let scopes = buf("nits-scopes", scope_bytes, U::STORAGE | U::COPY_SRC | U::COPY_DST);
        let wg_readback = buf("nits-wg-rb", wg_bytes, U::MAP_READ | U::COPY_DST);
        let scope_readback = buf("nits-scopes-rb", scope_bytes, U::MAP_READ | U::COPY_DST);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<Params>() as u64),
                    }),
                },
                wgpu::BindGroupEntry { binding: 1, resource: frame.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wg_stats.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: scopes.as_entire_binding() },
            ],
        });

        // YCbCr -> R'G'B' and luminance weights for the stream's matrix.
        let (kr, kb) = if color.bt2020 { (0.2627f32, 0.0593f32) } else { (0.2126, 0.0722) };
        let kg = 1.0 - kr - kb;
        let base = Params {
            width: width as u32,
            height: height as u32,
            u_off: (width * height) as u32,
            v_off: (width * height + width * height / 4) as u32,
            wave_cols: WAVE_COLS as u32,
            wave_rows: WAVE_ROWS as u32,
            scope_flags,
            wg_base: 0,
            scope_base: 0,
            hist_off: layout.hist.unwrap_or(0) as u32,
            cie_off: layout.cie.unwrap_or(0) as u32,
            vec_off: layout.vec.unwrap_or(0) as u32,
            transfer: match color.transfer {
                Transfer::Pq => 0,
                Transfer::Sdr => 1,
                Transfer::Hlg => 2,
            },
            full_range: color.full_range as u32,
            bt2020: color.bt2020 as u32,
            _pad0: 0,
            sdr_white,
            cr_r: 2.0 * (1.0 - kr),
            cb_g: 2.0 * kb * (1.0 - kb) / kg,
            cr_g: 2.0 * kr * (1.0 - kr) / kg,
            cb_b: 2.0 * (1.0 - kb),
            kr,
            kg,
            kb,
        };

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group,
            frame,
            params,
            params_stride,
            wg_stats,
            scopes,
            wg_readback,
            scope_readback,
            base,
            groups,
            wg_count,
            layout,
            pixels: width * height,
            pending: 0,
        })
    }

    pub fn frame_bytes(&self) -> usize {
        self.pixels * 3
    }

    /// Queues one frame; returns completed frames, in order, whenever a batch fills.
    pub fn push(&mut self, frame: &[u8]) -> Result<Vec<NitsFrame>> {
        let slot = self.pending;
        let words = self.layout.words;
        let params =
            Params { wg_base: (slot * self.wg_count * 4) as u32, scope_base: (slot * words) as u32, ..self.base };
        let params_offset = slot as u64 * self.params_stride;
        self.queue.write_buffer(&self.params, params_offset, bytemuck::bytes_of(&params));
        self.queue.write_buffer(&self.frame, 0, frame);

        let mut enc = self.device.create_command_encoder(&Default::default());
        if words > 0 {
            enc.clear_buffer(&self.scopes, (slot * words * 4) as u64, Some((words * 4) as u64));
        }
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[params_offset as u32]);
            pass.dispatch_workgroups(self.groups.0, self.groups.1, 1);
        }
        self.pending += 1;
        if self.pending == BATCH {
            self.flush(Some(enc))
        } else {
            self.queue.submit([enc.finish()]);
            Ok(Vec::new())
        }
    }

    pub fn finish(&mut self) -> Result<Vec<NitsFrame>> {
        self.flush(None)
    }

    fn flush(&mut self, last: Option<wgpu::CommandEncoder>) -> Result<Vec<NitsFrame>> {
        let n = self.pending;
        let mut bufs = Vec::new();
        if let Some(enc) = last {
            bufs.push(enc.finish());
        }
        if n == 0 {
            if !bufs.is_empty() {
                self.queue.submit(bufs);
            }
            return Ok(Vec::new());
        }
        let words = self.layout.words;
        let wg_bytes = (n * self.wg_count * 4 * 4) as u64;
        let scope_bytes = (n * words * 4) as u64;
        let mut copy = self.device.create_command_encoder(&Default::default());
        copy.copy_buffer_to_buffer(&self.wg_stats, 0, &self.wg_readback, 0, wg_bytes);
        if words > 0 {
            copy.copy_buffer_to_buffer(&self.scopes, 0, &self.scope_readback, 0, scope_bytes);
        }
        bufs.push(copy.finish());
        self.queue.submit(bufs);

        let wg_slice = self.wg_readback.slice(0..wg_bytes);
        let scope_slice = self.scope_readback.slice(0..if words > 0 { scope_bytes } else { 4 });
        let (tx, rx) = std::sync::mpsc::channel();
        let tx2 = tx.clone();
        wg_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        if words > 0 {
            scope_slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx2.send(r);
            });
        } else {
            drop(tx2);
        }
        self.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .map_err(|e| anyhow!("GPU poll failed: {e:?}"))?;
        for r in rx {
            r.map_err(|e| anyhow!("readback map failed: {e:?}"))?;
        }

        let mut out = Vec::with_capacity(n);
        {
            let wg_view = wg_slice.get_mapped_range().map_err(|e| anyhow!("readback view failed: {e:?}"))?;
            let wg: &[f32] = bytemuck::cast_slice(&wg_view);
            let scope_view = if words > 0 {
                Some(scope_slice.get_mapped_range().map_err(|e| anyhow!("readback view failed: {e:?}"))?)
            } else {
                None
            };
            let scope_words: Option<&[u32]> = scope_view.as_ref().map(|v| bytemuck::cast_slice(v));
            let l = self.layout;
            for slot in 0..n {
                let stats = &wg[slot * self.wg_count * 4..(slot + 1) * self.wg_count * 4];
                let mut f = NitsFrame::default();
                let (mut sum, mut sum_lum) = (0f64, 0f64);
                for g in stats.as_chunks::<4>().0 {
                    f.peak = f.peak.max(g[0]);
                    sum += g[1] as f64;
                    f.peak_lum = f.peak_lum.max(g[2]);
                    sum_lum += g[3] as f64;
                }
                f.avg = (sum / self.pixels as f64) as f32;
                f.avg_lum = (sum_lum / self.pixels as f64) as f32;
                f.scopes = scope_words.map(|w| {
                    let s = &w[slot * words..(slot + 1) * words];
                    let take = |off: Option<usize>, len: usize| off.map(|o| s[o..o + len].to_vec());
                    ScopeData {
                        wave: take(l.wave, 3 * WAVE_ROWS * WAVE_COLS),
                        hist: take(l.hist, 4 * HIST_BINS),
                        cie: take(l.cie, XY_BINS * XY_BINS),
                        vec: take(l.vec, XY_BINS * XY_BINS),
                    }
                });
                out.push(f);
            }
        }
        self.wg_readback.unmap();
        if words > 0 {
            self.scope_readback.unmap();
        }
        self.pending = 0;
        Ok(out)
    }
}

/// Peak and average over a clip, in the sense of CTA-861.3 MaxCLL / MaxFALL.
pub fn content_light_levels(frames: &[NitsFrame]) -> (f32, f32) {
    let max_cll = frames.iter().map(|f| f.peak).fold(0.0, f32::max);
    let max_fall = frames.iter().map(|f| f.avg).fold(0.0, f32::max);
    (max_cll, max_fall)
}
