//! GPU compositing of the overlay into decoded video frames.
//!
//! Frames are 10-bit 4:2:0 (P010) straight from the decoder; the overlay is
//! the premultiplied RGBA drawn by [`crate::canvas::Canvas`]. Blending here, on the
//! frames fpsprobe already decodes, leaves FFmpeg only the encoding: no second
//! decode and no CPU overlay filter, which was the slowest part of rendering.

use crate::nits::ColorInfo;
use anyhow::{Context, Result, anyhow};
use bytemuck::{Pod, Zeroable};
use std::sync::mpsc;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    width: u32,
    height: u32,
    uv_off: u32,
    _pad3: u32,
    ox: u32,
    oy: u32,
    ow: u32,
    oh: u32,
    block_x: u32,
    block_y: u32,
    full_range: u32,
    _pad0: u32,
    kr: f32,
    kb: f32,
    _pad1: f32,
    _pad2: f32,
}

/// Frames in flight: the GPU blends one while the caller draws the next overlay.
const SLOTS: usize = 2;

type MapResult = mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>;

/// GPU buffers for one frame in flight.
struct Slot {
    frame: wgpu::Buffer,
    overlay: wgpu::Buffer,
    readback: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Submitted work whose result hasn't been collected yet.
    pending: Option<(wgpu::SubmissionIndex, MapResult)>,
}

/// Blends a fixed overlay rectangle into frames of one size.
pub struct Compositor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    slots: Vec<Slot>,
    /// Slot the next frame goes into; it also holds the oldest frame in flight.
    next: usize,
    groups: (u32, u32),
    frame_bytes: usize,
    overlay_bytes: usize,
}

impl Compositor {
    /// Frames are `width` x `height` (multiples of 4 and 2); the overlay covers
    /// `rect` (x, y, w, h) inside them. `lut` maps straight overlay code values to
    /// the frame's transfer (see `render::hdr_panel_lut`; identity for SDR).
    pub fn new(
        width: usize,
        height: usize,
        rect: (usize, usize, usize, usize),
        color: ColorInfo,
        lut: &[u8; 256],
    ) -> Result<Self> {
        if !width.is_multiple_of(4) || !height.is_multiple_of(2) {
            return Err(anyhow!("frame size {width}x{height} must be a multiple of 4x2"));
        }
        let (ox, oy, ow, oh) = rect;
        if ox + ow > width || oy + oh > height || ow == 0 || oh == 0 {
            return Err(anyhow!("overlay {ow}x{oh} at {ox},{oy} doesn't fit a {width}x{height} frame"));
        }
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .context("no GPU adapter found")?;
        let frame_bytes = width * height * 3;
        let overlay_bytes = ow * oh * 4;
        let need = frame_bytes.max(overlay_bytes) as u64;
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size =
            limits.max_storage_buffer_binding_size.max(need).min(adapter.limits().max_storage_buffer_binding_size);
        limits.max_buffer_size = limits.max_buffer_size.max(need).min(adapter.limits().max_buffer_size);
        if limits.max_storage_buffer_binding_size < need {
            return Err(anyhow!("frame of {frame_bytes} bytes exceeds GPU storage buffer limit"));
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("fpsprobe-composite"),
            required_limits: limits,
            ..Default::default()
        }))?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("composite"),
            source: wgpu::ShaderSource::Wgsl(include_str!("composite.wgsl").into()),
        });
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        use wgpu::BufferBindingType as B;
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite"),
            entries: &[
                entry(0, B::Uniform),
                entry(1, B::Storage { read_only: false }),
                entry(2, B::Storage { read_only: true }),
                entry(3, B::Storage { read_only: true }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("composite"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("composite"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Blocks of 4x2 pixels touching the overlay.
        let (bx0, by0) = (ox / 4, oy / 2);
        let (bx1, by1) = ((ox + ow).div_ceil(4), (oy + oh).div_ceil(2));
        let groups = (((bx1 - bx0) as u32).div_ceil(16), ((by1 - by0) as u32).div_ceil(8));
        let (kr, kb) = if color.bt2020 { (0.2627f32, 0.0593f32) } else { (0.2126, 0.0722) };
        let params = Params {
            width: width as u32,
            height: height as u32,
            uv_off: (width * height) as u32,
            _pad3: 0,
            ox: ox as u32,
            oy: oy as u32,
            ow: ow as u32,
            oh: oh as u32,
            block_x: bx0 as u32,
            block_y: by0 as u32,
            full_range: color.full_range as u32,
            _pad0: 0,
            kr,
            kb,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        use wgpu::BufferUsages as U;
        use wgpu::util::DeviceExt;
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("composite-params"),
            contents: bytemuck::bytes_of(&params),
            usage: U::UNIFORM,
        });
        let lut_words: Vec<u32> = lut.iter().map(|&v| v as u32).collect();
        let lut = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("composite-lut"),
            contents: bytemuck::cast_slice(&lut_words),
            usage: U::STORAGE,
        });
        let buf = |label, size: usize, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage,
                mapped_at_creation: false,
            })
        };
        let slots = (0..SLOTS)
            .map(|_| {
                let frame = buf("composite-frame", frame_bytes, U::STORAGE | U::COPY_DST | U::COPY_SRC);
                let overlay = buf("composite-overlay", overlay_bytes, U::STORAGE | U::COPY_DST);
                let readback = buf("composite-rb", frame_bytes, U::MAP_READ | U::COPY_DST);
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: frame.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: overlay.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: lut.as_entire_binding() },
                    ],
                });
                Slot { frame, overlay, readback, bind_group, pending: None }
            })
            .collect();
        Ok(Self { device, queue, pipeline, slots, next: 0, groups, frame_bytes, overlay_bytes })
    }

    /// Starts blending `overlay` (premultiplied RGBA of the rectangle) into `frame`
    /// (P010). Frames come back in order, [`SLOTS`] - 1 calls later: this
    /// returns the oldest finished frame once the pipeline is full.
    pub fn push(&mut self, frame: &[u8], overlay: &[u8]) -> Result<Option<Vec<u8>>> {
        if frame.len() != self.frame_bytes || overlay.len() != self.overlay_bytes {
            return Err(anyhow!("frame or overlay has the wrong size"));
        }
        let i = self.next;
        self.next = (i + 1) % SLOTS;
        let done = self.collect(i)?;
        let slot = &mut self.slots[i];
        self.queue.write_buffer(&slot.frame, 0, frame);
        self.queue.write_buffer(&slot.overlay, 0, overlay);
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &slot.bind_group, &[]);
            pass.dispatch_workgroups(self.groups.0, self.groups.1, 1);
        }
        enc.copy_buffer_to_buffer(&slot.frame, 0, &slot.readback, 0, self.frame_bytes as u64);
        let index = self.queue.submit([enc.finish()]);
        let (tx, rx) = mpsc::channel();
        slot.readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        slot.pending = Some((index, rx));
        Ok(done)
    }

    /// The frames still in flight, oldest first.
    pub fn flush(&mut self) -> Result<Vec<Vec<u8>>> {
        let mut out = Vec::new();
        for k in 0..SLOTS {
            out.extend(self.collect((self.next + k) % SLOTS)?);
        }
        Ok(out)
    }

    /// Blends one frame and waits for it.
    pub fn composite(&mut self, frame: &[u8], overlay: &[u8]) -> Result<Vec<u8>> {
        let mut out = self.flush()?;
        out.extend(self.push(frame, overlay)?);
        out.extend(self.flush()?);
        out.pop().ok_or_else(|| anyhow!("no composited frame"))
    }

    /// Waits for slot `i`'s frame, if it has one, and copies it out.
    fn collect(&mut self, i: usize) -> Result<Option<Vec<u8>>> {
        let slot = &mut self.slots[i];
        let Some((index, rx)) = slot.pending.take() else { return Ok(None) };
        self.device
            .poll(wgpu::PollType::Wait { submission_index: Some(index), timeout: None })
            .map_err(|e| anyhow!("GPU poll failed: {e:?}"))?;
        rx.recv()?.map_err(|e| anyhow!("readback map failed: {e:?}"))?;
        let out =
            slot.readback.slice(..).get_mapped_range().map_err(|e| anyhow!("readback view failed: {e:?}"))?.to_vec();
        slot.readback.unmap();
        Ok(Some(out))
    }
}

/// CPU reference of composite.wgsl, for tests.
#[cfg(test)]
fn composite_cpu(
    frame: &mut [u16],
    (width, height): (usize, usize),
    (ox, oy, ow, oh): (usize, usize, usize, usize),
    overlay: &[u8],
    color: ColorInfo,
    lut: &[u8; 256],
) {
    let (kr, kb) = if color.bt2020 { (0.2627f32, 0.0593f32) } else { (0.2126, 0.0722) };
    let pixel = |x: usize, y: usize| -> [f32; 4] {
        if x < ox || y < oy || x >= ox + ow || y >= oy + oh {
            return [0.0; 4];
        }
        let i = ((y - oy) * ow + (x - ox)) * 4;
        let a = overlay[i + 3] as f32 / 255.0;
        if overlay[i + 3] == 0 {
            return [0.0; 4];
        }
        let c = |k: usize| lut[((overlay[i + k] as f32 / a).round() as usize).min(255)] as f32 / 255.0;
        let (r, g, b) = (c(0), c(1), c(2));
        let yp = kr * r + (1.0 - kr - kb) * g + kb * b;
        let (cb, cr) = ((b - yp) / (2.0 * (1.0 - kb)), (r - yp) / (2.0 * (1.0 - kr)));
        if color.full_range {
            [yp * 1023.0, 512.0 + cb * 1023.0, 512.0 + cr * 1023.0, a]
        } else {
            [64.0 + yp * 876.0, 512.0 + cb * 896.0, 512.0 + cr * 896.0, a]
        }
    };
    // P010: 10-bit values in the top bits; U and V interleaved after the Y plane.
    let code = |v: f32| (v.round().clamp(0.0, 1023.0) as u16) << 6;
    let (uv_off, cw) = (width * height, width / 2);
    for y in 0..height {
        for x in 0..width {
            let p = pixel(x, y);
            let d = &mut frame[y * width + x];
            let old = (*d >> 6) as f32;
            *d = code(old + (p[0] - old) * p[3]);
        }
    }
    for cy in 0..height / 2 {
        for cx in 0..cw {
            let q = [
                pixel(2 * cx, 2 * cy),
                pixel(2 * cx + 1, 2 * cy),
                pixel(2 * cx, 2 * cy + 1),
                pixel(2 * cx + 1, 2 * cy + 1),
            ];
            let sum_a: f32 = q.iter().map(|p| p[3]).sum();
            for c in [1, 2] {
                let d = &mut frame[uv_off + 2 * (cy * cw + cx) + c - 1];
                let sum_ac: f32 = q.iter().map(|p| p[3] * p[c]).sum();
                *d = code(sum_ac / 4.0 + (1.0 - sum_a / 4.0) * (*d >> 6) as f32);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> [u8; 256] {
        std::array::from_fn(|i| i as u8)
    }

    /// Mid-grey P010 frame.
    fn grey(w: usize, h: usize) -> Vec<u16> {
        let mut f = vec![512u16 << 6; w * h * 3 / 2];
        f[..w * h].fill(500 << 6);
        f
    }

    fn bytes(samples: &[u16]) -> Vec<u8> {
        samples.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    #[test]
    fn opaque_white_is_limited_range_white() {
        let color = ColorInfo::from_probe("bt709", "bt709", "tv");
        let (w, h) = (8, 4);
        let mut frame = grey(w, h);
        let overlay = vec![255u8; 4 * 4 * 4]; // 4x4 white, fully opaque, at (2, 0)
        composite_cpu(&mut frame, (w, h), (2, 0, 4, 4), &overlay, color, &identity());
        assert_eq!(frame[2] >> 6, 940); // luma white
        assert_eq!(frame[1] >> 6, 500); // outside the overlay
        assert_eq!(frame[w * h + 2] >> 6, 512); // U of a fully covered 2x2 group: neutral
        assert_eq!(frame[w * h + 3] >> 6, 512); // and its V
    }

    #[test]
    fn gpu_matches_cpu() {
        let color = ColorInfo::from_probe("smpte2084", "bt2020nc", "tv");
        let (w, h) = (64, 32);
        let rect = (6, 3, 37, 21); // odd position and size
        let lut: [u8; 256] = std::array::from_fn(|i| (255 - i / 2) as u8);
        let Ok(mut gpu) = Compositor::new(w, h, rect, color, &lut) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let mut frame: Vec<u16> = (0..w * h * 3 / 2).map(|i| ((64 + (i * 37) % 876) as u16) << 6).collect();
        // Premultiplied pixels with varying alpha, including transparent ones.
        let overlay: Vec<u8> = (0..rect.2 * rect.3)
            .flat_map(|i| {
                let a = ((i * 53) % 256) as u8;
                let c = |k: usize| ((((i * k * 29) % 256) * a as usize) / 255) as u8;
                [c(1), c(2), c(3), a]
            })
            .collect();
        let out = gpu.composite(&bytes(&frame), &overlay).unwrap();
        composite_cpu(&mut frame, (w, h), rect, &overlay, color, &lut);
        let got: Vec<u16> = out.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect();
        // GPU float rounding may differ by a code value, or by a LUT step where
        // un-premultiplying lands on .5.
        let diffs: Vec<u16> = got.iter().zip(&frame).map(|(a, b)| (a >> 6).abs_diff(b >> 6)).collect();
        let off = diffs.iter().filter(|&&d| d > 1).count();
        assert!(diffs.iter().all(|&d| d <= 4), "largest difference {:?}", diffs.iter().max());
        assert!(off * 200 < diffs.len(), "{off} of {} samples differ by more than 1", diffs.len());
    }
}
