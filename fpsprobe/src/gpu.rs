//! GPU backend: uploads luma frames and computes per-tile SAD with a compute shader.
//!
//! Stats for up to `BATCH` frames are accumulated in one storage buffer and read
//! back together, so the CPU only ever waits on the GPU once per batch.

use crate::analyze::{FrameStats, Geometry, TILE};
use anyhow::{Context, Result, anyhow};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const BATCH: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    width_words: u32,
    height: u32,
    tiles_x: u32,
    compat_thr: u32,
    base: u32,
    _pad: [u32; 3],
}

pub struct GpuDiff {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    /// Two bind groups so frame buffers ping-pong instead of being copied:
    /// bind_groups[i] reads (frames[1-i], frames[i]) as (prev, cur).
    bind_groups: [wgpu::BindGroup; 2],
    frames: [wgpu::Buffer; 2],
    params: wgpu::Buffer,
    params_stride: u64,
    stats: wgpu::Buffer,
    readback: wgpu::Buffer,
    geo: Geometry,
    compat_thr: u32,
    slot_words: usize,
    /// Frame index of the next upload (frame 0 has no predecessor, so no dispatch).
    next_index: usize,
    pending: usize,
}

impl GpuDiff {
    pub fn new(geo: Geometry, compat_thr: u32) -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .context("no GPU adapter found")?;

        let frame_bytes = geo.frame_bytes() as u64;
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size =
            limits.max_storage_buffer_binding_size.max(frame_bytes).min(adapter.limits().max_storage_buffer_binding_size);
        limits.max_buffer_size = limits.max_buffer_size.max(frame_bytes).min(adapter.limits().max_buffer_size);
        if limits.max_storage_buffer_binding_size < frame_bytes {
            return Err(anyhow!("frame of {frame_bytes} bytes exceeds GPU storage buffer limit"));
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("fpsprobe"),
            required_limits: limits,
            ..Default::default()
        }))?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("diff"),
            source: wgpu::ShaderSource::Wgsl(include_str!("diff.wgsl").into()),
        });
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("diff"),
            entries: &[
                // Params use a dynamic offset so each batch slot gets its own block.
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
                storage(1, true),
                storage(2, true),
                storage(3, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("diff"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("diff"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let frames = [0, 1].map(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(if i == 0 { "frame0" } else { "frame1" }),
                size: frame_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        // One Params block per batch slot, each at a uniform-offset-aligned stride.
        let align = device.limits().min_uniform_buffer_offset_alignment as u64;
        let params_stride = (std::mem::size_of::<Params>() as u64).div_ceil(align) * align;
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: params_stride * BATCH as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Slot layout: [compat_count, tile_sad[0..tiles]]
        let slot_words = 1 + geo.tiles();
        let stats_bytes = (slot_words * BATCH * 4) as u64;
        let stats = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stats"),
            contents: &vec![0u8; stats_bytes as usize],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: stats_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_groups = [0usize, 1].map(|cur| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &params,
                            offset: 0,
                            size: wgpu::BufferSize::new(std::mem::size_of::<Params>() as u64),
                        }),
                    },
                    wgpu::BindGroupEntry { binding: 1, resource: frames[1 - cur].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: frames[cur].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: stats.as_entire_binding() },
                ],
            })
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_groups,
            frames,
            params,
            params_stride,
            stats,
            readback,
            geo,
            compat_thr,
            slot_words,
            next_index: 0,
            pending: 0,
        })
    }

    /// Uploads one frame and, if it has a predecessor, queues its diff.
    /// Returns stats for completed frames whenever a batch fills up.
    pub fn push(&mut self, frame: &[u8]) -> Result<Vec<FrameStats>> {
        let cur = self.next_index % 2;
        self.queue.write_buffer(&self.frames[cur], 0, frame);
        let first = self.next_index == 0;
        self.next_index += 1;
        if first {
            self.queue.submit([]);
            return Ok(Vec::new());
        }

        let slot = self.pending;
        let params = Params {
            width_words: (self.geo.width / 4) as u32,
            height: self.geo.height as u32,
            tiles_x: self.geo.tiles_x() as u32,
            compat_thr: self.compat_thr,
            base: (slot * self.slot_words) as u32,
            _pad: [0; 3],
        };
        let params_offset = slot as u64 * self.params_stride;
        self.queue.write_buffer(&self.params, params_offset, bytemuck::bytes_of(&params));

        let mut enc = self.device.create_command_encoder(&Default::default());
        // Only the compat counter accumulates; tile SADs are overwritten.
        enc.clear_buffer(&self.stats, (slot * self.slot_words * 4) as u64, Some(4));
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_groups[cur], &[params_offset as u32]);
            pass.dispatch_workgroups(self.geo.tiles_x() as u32, self.geo.tiles_y() as u32, 1);
        }
        self.pending += 1;

        let mut out = Vec::new();
        if self.pending == BATCH {
            out = self.flush_with(Some(enc))?;
        } else {
            self.queue.submit([enc.finish()]);
        }
        Ok(out)
    }

    /// Reads back any frames still pending on the GPU.
    pub fn finish(&mut self) -> Result<Vec<FrameStats>> {
        self.flush_with(None)
    }

    fn flush_with(&mut self, last: Option<wgpu::CommandEncoder>) -> Result<Vec<FrameStats>> {
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
        let bytes = (n * self.slot_words * 4) as u64;
        let mut copy = self.device.create_command_encoder(&Default::default());
        copy.copy_buffer_to_buffer(&self.stats, 0, &self.readback, 0, bytes);
        bufs.push(copy.finish());
        self.queue.submit(bufs);

        let slice = self.readback.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .map_err(|e| anyhow!("GPU poll failed: {e:?}"))?;
        rx.recv()?.map_err(|e| anyhow!("readback map failed: {e:?}"))?;

        let mut out = Vec::with_capacity(n);
        {
            let view = slice.get_mapped_range().map_err(|e| anyhow!("readback view failed: {e:?}"))?;
            let words: &[u32] = bytemuck::cast_slice(&view);
            for slot in words.chunks_exact(self.slot_words) {
                out.push(FrameStats { compat_count: slot[0] as u64, tile_sad: slot[1..].to_vec() });
            }
        }
        self.readback.unmap();
        self.pending = 0;
        Ok(out)
    }
}

const _: () = assert!(TILE == 64, "diff.wgsl hardcodes 64x64 tiles");
