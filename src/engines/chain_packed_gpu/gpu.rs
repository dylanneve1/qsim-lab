//! The wgpu driver: device setup, per-pass op programs and tables, and the
//! chunk pipeline (host gather → upload → unpack → sub-stages → pack →
//! download → host scatter, `slots` chunks in flight).

use super::{
    gather_chunk, gpu_plans, next_base, scatter_chunk, GpuRun, HalfCodec, HostStore, NMANT,
};
use crate::circuit::SimError;
use crate::engines::blocked::gpu_export::{GpuOp, GpuSubStage};
use crate::engines::blocked::{BlockConfig, Stage};
use crate::engines::chain_lowprec::LowPrec;
use crate::engines::chain_packed::GpuStagePlan;
use crate::engines::chain_sweep::SweepPlan;
use num_complex::Complex64;
use std::time::Instant;

const SHADER: &str = include_str!("kernels.wgsl");
/// Workgroup size of every kernel.
const WG: u32 = 256;
/// Bytes of one `Params` slot in the params buffer (dynamic-offset aligned).
const PSLOT: u64 = 256;
/// Words of the op program uniform (`array<vec4<u32>, 4096>`).
const PROG_WORDS: usize = 4 * 4096;
const NONE: u32 = u32::MAX;

/// Options of the GPU backend.
#[derive(Clone, Debug)]
pub struct GpuOptions {
    /// Cache-block (sub-stage buffer) bits the shader is built for; the
    /// config's nested block must not exceed it (`2^bits · 8` bytes of
    /// workgroup memory).
    pub nested_bits: usize,
    /// Amplitudes per streamed chunk (rounded to whole gathered blocks).
    pub chunk_amps: usize,
    /// Chunks in flight.
    pub slots: usize,
    /// Backends to try (default: Vulkan, then DX12, then any).
    pub backends: Option<wgpu::Backends>,
    /// Print per-pass timings to stderr.
    pub progress: bool,
}

impl Default for GpuOptions {
    fn default() -> Self {
        GpuOptions {
            nested_bits: 12,
            chunk_amps: 1 << 27,
            slots: 3,
            backends: None,
            progress: false,
        }
    }
}

/// A GPU device ready to run packed sweeps.
pub struct GpuSweeper {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    unpack: wgpu::ComputePipeline,
    substage: wgpu::ComputePipeline,
    pack: wgpu::ComputePipeline,
    info: String,
    opts: GpuOptions,
    max_binding: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Params {
    l: u32,
    g: u32,
    c0: u32,
    fresh: u32,
    base_in: i32,
    base_out: i32,
    bits: u32,
    maxv: u32,
    block: u32,
    wpb: u32,
    sl: u32,
    lowb: u32,
    himask: u32,
    outmask: u32,
    nx: u32,
    total: u32,
    dec_off: u32,
    thr_off: u32,
    pad0: u32,
    pad1: u32,
}

impl Params {
    fn bytes(&self) -> [u8; 80] {
        let w = [
            self.l,
            self.g,
            self.c0,
            self.fresh,
            self.base_in as u32,
            self.base_out as u32,
            self.bits,
            self.maxv,
            self.block,
            self.wpb,
            self.sl,
            self.lowb,
            self.himask,
            self.outmask,
            self.nx,
            self.total,
            self.dec_off,
            self.thr_off,
            self.pad0,
            self.pad1,
        ];
        let mut out = [0u8; 80];
        for (o, v) in out.chunks_exact_mut(4).zip(w) {
            o.copy_from_slice(&v.to_le_bytes());
        }
        out
    }
}

fn err(what: &'static str) -> SimError {
    SimError::NotSupported { what }
}

/// 2D dispatch shape for `groups` workgroups: `(nx, ny)`.
fn shape(groups: u32) -> (u32, u32) {
    if groups <= 65535 {
        (groups.max(1), 1)
    } else {
        (32768, groups.div_ceil(32768))
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Encodes the ops of a sub-stage into the program words, appending the
/// diagonal tables to `tables` (as f32 bits).
fn encode_sub(s: &GpuSubStage, tables: &mut Vec<u32>) -> Vec<u32> {
    let mut w = vec![0u32];
    let mut nops = 0u32;
    let fb = |m: &[f32; 8]| m.map(f32::to_bits);
    for op in &s.ops {
        match op {
            GpuOp::U1 {
                t,
                m,
                kind,
                cin,
                cout,
            } => {
                w.extend([
                    1,
                    *t,
                    *kind as u32,
                    *cin,
                    *cout as u32,
                    (*cout >> 32) as u32,
                ]);
                w.extend(fb(m));
                nops += 1;
            }
            GpuOp::Swap { a, b } => {
                w.extend([2, *a, *b]);
                nops += 1;
            }
            GpuOp::Pair {
                t1,
                m1,
                real1,
                t2,
                m2,
                real2,
                cx,
            } => {
                w.extend([3, *t1, *t2, *real1 as u32, *real2 as u32, *cx as u32]);
                w.extend(fb(m1));
                w.extend(fb(m2));
                nops += 1;
            }
            GpuOp::Diag(gs) => {
                for g in gs {
                    // variant offsets, then the tables of each variant
                    let voff = tables.len() as u32;
                    tables.resize(tables.len() + g.variants.len(), NONE);
                    for (i, v) in g.variants.iter().enumerate() {
                        if let Some(t) = v {
                            tables[voff as usize + i] = tables.len() as u32;
                            for part in [&t.lor, &t.loi, &t.hr, &t.hi] {
                                tables.extend(part.iter().map(|x| x.to_bits()));
                            }
                        }
                    }
                    w.extend([4, g.cmask, g.cpat, g.lb, g.conds.len() as u32, voff]);
                    for &(om, op) in &g.conds {
                        w.extend([om as u32, (om >> 32) as u32, op as u32, (op >> 32) as u32]);
                    }
                    nops += 1;
                }
            }
        }
    }
    w[0] = nops;
    w
}

/// Per-slot device and staging buffers.
struct Slot {
    pk: wgpu::Buffer,
    cd: wgpu::Buffer,
    work: wgpu::Buffer,
    up: wgpu::Buffer,
    down: wgpu::Buffer,
    /// Chunk in flight: (pass-local chunk index, submission).
    busy: Option<(usize, wgpu::SubmissionIndex)>,
    up_mapped: bool,
}

impl GpuSweeper {
    /// Opens the first suitable adapter (high performance preferred).
    pub fn new(opts: &GpuOptions) -> Result<Self, String> {
        let nb = opts.nested_bits;
        let need_wg = (8u64 << nb) as u32;
        let tries = match opts.backends {
            Some(b) => vec![b],
            None => vec![
                wgpu::Backends::VULKAN,
                wgpu::Backends::DX12,
                wgpu::Backends::all(),
            ],
        };
        let mut last = String::from("no adapter");
        for b in tries {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: b,
                ..wgpu::InstanceDescriptor::new_without_display_handle()
            });
            let adapter = match pollster::block_on(instance.request_adapter(
                &wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                },
            )) {
                Ok(a) => a,
                Err(e) => {
                    last = format!("{b:?}: {e}");
                    continue;
                }
            };
            let al = adapter.limits();
            if al.max_compute_workgroup_storage_size < need_wg + 64 {
                last = format!(
                    "{b:?}: workgroup memory {} < {}",
                    al.max_compute_workgroup_storage_size, need_wg
                );
                continue;
            }
            let ai = adapter.get_info();
            let limits = wgpu::Limits {
                max_compute_workgroup_storage_size: al.max_compute_workgroup_storage_size,
                max_storage_buffer_binding_size: al.max_storage_buffer_binding_size,
                max_buffer_size: al.max_buffer_size,
                max_uniform_buffer_binding_size: al.max_uniform_buffer_binding_size.max(65536),
                max_compute_invocations_per_workgroup: al
                    .max_compute_invocations_per_workgroup
                    .max(WG),
                max_compute_workgroup_size_x: al.max_compute_workgroup_size_x.max(WG),
                max_storage_buffers_per_shader_stage: al.max_storage_buffers_per_shader_stage,
                ..wgpu::Limits::default()
            };
            let (device, queue) = match pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("qsim packed sweep"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    ..Default::default()
                },
            )) {
                Ok(d) => d,
                Err(e) => {
                    last = format!("{b:?}: {e}");
                    continue;
                }
            };
            let src = SHADER
                .replace("__MAXN__", &(1u32 << nb).to_string())
                .replace("__WG__", &WG.to_string());
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("packed kernels"),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            let st = |ro: bool| wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: ro },
                has_dynamic_offset: false,
                min_binding_size: None,
            };
            let un = |size: u64| wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(size),
            };
            let types = [
                st(false),
                st(false),
                st(false),
                st(true),
                st(false),
                un(80),
                un((PROG_WORDS * 4) as u64),
                st(true),
            ];
            let entries: Vec<_> = types
                .iter()
                .enumerate()
                .map(|(i, ty)| wgpu::BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: *ty,
                    count: None,
                })
                .collect();
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("packed layout"),
                entries: &entries,
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("packed pipeline layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let mk = |entry: &str| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pl),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            };
            let (unpack, substage, pack) = (mk("unpack"), mk("substage"), mk("pack"));
            let info = format!(
                "{} ({:?}, {:?}, driver {} {})",
                ai.name, ai.backend, ai.device_type, ai.driver, ai.driver_info
            );
            return Ok(GpuSweeper {
                device,
                queue,
                layout,
                unpack,
                substage,
                pack,
                info,
                opts: opts.clone(),
                max_binding: al.max_storage_buffer_binding_size.min(al.max_buffer_size),
            });
        }
        Err(last)
    }

    /// Adapter description.
    pub fn adapter_info(&self) -> String {
        self.info.clone()
    }

    fn buffer(&self, label: &str, size: u64, usage: wgpu::BufferUsages, mapped: bool) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16).next_multiple_of(16),
            usage,
            mapped_at_creation: mapped,
        })
    }

    fn wait(&self, idx: &wgpu::SubmissionIndex) {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(idx.clone()),
                timeout: None,
            })
            .expect("GPU poll");
    }

    /// Runs `stages` on a packed register in format `lp` (`intB:bN:h`) on
    /// the GPU and returns the `|0..0>` amplitude times `plan.scale`. The
    /// result is bit-identical to `run_packed(plan, stages, lp, cfg)` (see
    /// the module docs for the conditions on `cfg`).
    pub fn run(
        &self,
        plan: &SweepPlan,
        stages: &[Stage],
        lp: &LowPrec,
        cfg: &BlockConfig,
    ) -> Result<GpuRun, SimError> {
        let codec = HalfCodec::new(lp)?;
        let plans = gpu_plans(stages, plan.width, cfg)?;
        if plans
            .iter()
            .any(|p| p.subs.iter().any(|s| s.l > self.opts.nested_bits))
        {
            return Err(err(
                "GPU packed sweep: a cache block exceeds the shader's (lower cfg.block_bytes)",
            ));
        }
        if codec.block > 64 {
            return Err(err("GPU packed sweep: scale blocks of at most 64 amplitudes"));
        }
        let mut store = HostStore::new(plan.width, &codec)?;
        let width = plan.width;
        let maxl = plans.iter().map(|p| p.l).max().unwrap_or(0);
        if plans.iter().any(|p| (1usize << p.bc) < codec.block) {
            return Err(err(
                "GPU packed sweep: a stage's contiguous runs are shorter than the scale block",
            ));
        }
        let chunk = self
            .opts
            .chunk_amps
            .next_power_of_two()
            .max(1 << maxl)
            .min(1 << width);
        if (chunk as u64) * 8 > self.max_binding {
            return Err(err("GPU packed sweep: chunk work buffer exceeds the binding limit"));
        }
        let bb = codec.block_bytes();
        let nsb = chunk / codec.block; // scale blocks per chunk
        let pk_bytes = (nsb * bb) as u64;
        let cd_bytes = (nsb * 2) as u64;
        let stage_bytes = pk_bytes + cd_bytes.next_multiple_of(4);
        use wgpu::BufferUsages as U;
        let mut slots: Vec<Slot> = (0..self.opts.slots.max(1))
            .map(|_| Slot {
                pk: self.buffer("pk", pk_bytes, U::STORAGE | U::COPY_DST | U::COPY_SRC, false),
                cd: self.buffer("cd", cd_bytes, U::STORAGE | U::COPY_DST | U::COPY_SRC, false),
                work: self.buffer("work", chunk as u64 * 8, U::STORAGE, false),
                up: self.buffer("up", stage_bytes, U::MAP_WRITE | U::COPY_SRC, true),
                down: self.buffer("down", stage_bytes, U::MAP_READ | U::COPY_DST, false),
                busy: None,
                up_mapped: true,
            })
            .collect();
        // codec tables: dec rows then thresholds
        let mut cvec = codec.dec.clone();
        let thr_off = cvec.len() as u32;
        cvec.extend_from_slice(&codec.thr);
        debug_assert_eq!(codec.thr.len(), NMANT * codec.maxv as usize);
        let codec_buf = self.buffer("codec", cvec.len() as u64 * 4, U::STORAGE | U::COPY_DST, false);
        self.queue.write_buffer(&codec_buf, 0, &f32_bytes(&cvec));
        let stats = self.buffer("stats", 16, U::STORAGE | U::COPY_DST | U::COPY_SRC, false);
        let stats_rb = self.buffer("stats rb", 16, U::MAP_READ | U::COPY_DST, false);

        let t_all = Instant::now();
        let (mut uf, mut of, mut inx) = (0u64, 0u64, 0u64);
        let mut pass_secs = Vec::with_capacity(plans.len());
        for (pi, p) in plans.iter().enumerate() {
            let t0 = Instant::now();
            let st = self.run_pass(
                &mut store, &codec, p, chunk, &mut slots, &codec_buf, thr_off, &stats, &stats_rb,
            )?;
            uf += st[1] as u64;
            of += st[2] as u64;
            inx += st[3] as u64;
            pass_secs.push(t0.elapsed().as_secs_f64());
            if self.opts.progress {
                eprintln!(
                    "  gpu pass {}/{} l={} bc={} subs={} {:.2}s (t={:.1}s)",
                    pi + 1,
                    plans.len(),
                    p.l,
                    p.bc,
                    p.subs.len(),
                    t0.elapsed().as_secs_f64(),
                    t_all.elapsed().as_secs_f64()
                );
            }
        }
        let secs = t_all.elapsed().as_secs_f64();
        let a = store.amplitude(&codec, 0);
        Ok(GpuRun {
            amp: Complex64::new(a.re as f64, a.im as f64) * plan.scale,
            passes: plans.len(),
            store_bytes: store.bytes(),
            underflow: uf,
            overflow: of,
            inexact: inx,
            secs,
            pass_secs,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run_pass(
        &self,
        store: &mut HostStore,
        codec: &HalfCodec,
        p: &GpuStagePlan,
        chunk: usize,
        slots: &mut [Slot],
        codec_buf: &wgpu::Buffer,
        thr_off: u32,
        stats: &wgpu::Buffer,
        stats_rb: &wgpu::Buffer,
    ) -> Result<[i32; 4], SimError> {
        use wgpu::BufferUsages as U;
        let width = store.width;
        let nblocks = 1usize << (width - p.l);
        let g = (chunk >> p.l).min(nblocks);
        let nchunks = nblocks / g;
        let amps = g << p.l;
        let nsb = amps / codec.block;
        let bb = codec.block_bytes();
        let (pk_len, cd_len) = ((nsb * bb) as u64, (nsb * 2) as u64);
        let cd_off = pk_len; // within the staging buffers
        let base_in = store.base;
        let base_out = next_base(store.maxexp);

        // programs (one per sub-stage, 256-byte aligned) and tables
        let mut tables: Vec<u32> = Vec::new();
        let mut prog: Vec<u8> = Vec::new();
        let mut prog_off = Vec::new();
        for s in &p.subs {
            let w = encode_sub(s, &mut tables);
            if w.len() > PROG_WORDS {
                return Err(err("GPU packed sweep: a cache block has too many ops"));
            }
            prog_off.push(prog.len() as u32);
            prog.extend(w.iter().flat_map(|x| x.to_le_bytes()));
            prog.resize(prog.len().next_multiple_of(256), 0);
        }
        prog.resize(prog.len() + PROG_WORDS * 4, 0);
        tables.push(0);
        let prog_buf = self.buffer("prog", prog.len() as u64, U::UNIFORM | U::COPY_DST, false);
        self.queue.write_buffer(&prog_buf, 0, &prog);
        let tab_buf = self.buffer("tables", tables.len() as u64 * 4, U::STORAGE | U::COPY_DST, false);
        let tb: Vec<u8> = tables.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.queue.write_buffer(&tab_buf, 0, &tb);

        // params: per chunk, unpack + one per sub-stage + pack
        let nk = p.subs.len() + 2;
        let mut params = vec![0u8; nchunks * nk * PSLOT as usize];
        let fmt = Params {
            l: p.l as u32,
            g: g as u32,
            fresh: store.fresh as u32,
            base_in,
            base_out,
            bits: codec.bits,
            maxv: codec.maxv,
            block: codec.block as u32,
            wpb: (bb / 4) as u32,
            dec_off: 0,
            thr_off,
            ..Default::default()
        };
        let full = (1usize << p.l) - 1;
        let mut disp: Vec<(u32, u32)> = Vec::with_capacity(nk);
        for ci in 0..nchunks {
            let c0 = (ci * g) as u32;
            for k in 0..nk {
                let mut q = Params { c0, ..fmt };
                let groups;
                if k == 0 || k == nk - 1 {
                    let items = if k == 0 { nsb } else { nsb.div_ceil(2) };
                    groups = (items as u32).div_ceil(WG);
                    q.total = if k == 0 { nsb as u32 } else { nsb as u32 };
                } else {
                    let s = &p.subs[k - 1];
                    let lowb = s.inner_mask.trailing_ones() as usize;
                    q.sl = s.l as u32;
                    q.lowb = lowb as u32;
                    q.himask = (s.inner_mask >> lowb << lowb) as u32;
                    q.outmask = (full & !s.inner_mask) as u32;
                    let wgs = g << (p.l - s.l);
                    groups = wgs as u32;
                    q.total = wgs as u32;
                }
                let (nx, ny) = shape(groups);
                q.nx = nx;
                if ci == 0 {
                    disp.push((nx, ny));
                }
                let o = (ci * nk + k) * PSLOT as usize;
                params[o..o + 80].copy_from_slice(&q.bytes());
            }
        }
        let par_buf = self.buffer("params", params.len() as u64, U::UNIFORM | U::COPY_DST, false);
        self.queue.write_buffer(&par_buf, 0, &params);
        let mut init = Vec::new();
        for v in [i32::MIN, 0, 0, 0] {
            init.extend(v.to_le_bytes());
        }
        self.queue.write_buffer(stats, 0, &init);

        let groups: Vec<wgpu::BindGroup> = slots
            .iter()
            .map(|s| {
                let whole = |b: &wgpu::Buffer| b.as_entire_binding();
                let sized = |b: &wgpu::Buffer, size: u64| {
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: b,
                        offset: 0,
                        size: wgpu::BufferSize::new(size),
                    })
                };
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("packed bind group"),
                    layout: &self.layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: whole(&s.pk) },
                        wgpu::BindGroupEntry { binding: 1, resource: whole(&s.cd) },
                        wgpu::BindGroupEntry { binding: 2, resource: whole(&s.work) },
                        wgpu::BindGroupEntry { binding: 3, resource: whole(codec_buf) },
                        wgpu::BindGroupEntry { binding: 4, resource: whole(stats) },
                        wgpu::BindGroupEntry { binding: 5, resource: sized(&par_buf, 80) },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: sized(&prog_buf, (PROG_WORDS * 4) as u64),
                        },
                        wgpu::BindGroupEntry { binding: 7, resource: whole(&tab_buf) },
                    ],
                })
            })
            .collect();

        let ns = slots.len();
        let finish = |slot: &mut Slot, store: &mut HostStore| {
            let Some((ci, idx)) = slot.busy.take() else {
                return;
            };
            self.wait(&idx);
            slot.up_mapped = true;
            {
                let v = slot.down.get_mapped_range(..).expect("map down");
                let data = &v[..pk_len as usize];
                let cb = &v[cd_off as usize..(cd_off + cd_len) as usize];
                let codes: Vec<u16> = cb
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                scatter_chunk(store, codec, p, ci * g, g, data, &codes);
            }
            slot.down.unmap();
        };
        for ci in 0..nchunks {
            let si = ci % ns;
            finish(&mut slots[si], store);
            let slot = &mut slots[si];
            // `up` was re-mapped with the submission `finish` waited on
            assert!(slot.up_mapped, "upload staging not mapped");
            {
                let mut v = slot.up.get_mapped_range_mut(..).expect("map up range");
                let mut codes = vec![0u16; nsb];
                let (data, rest) = v.split_at_mut(pk_len as usize);
                gather_chunk(store, codec, p, ci * g, g, data, &mut codes);
                for (o, c) in rest[..cd_len as usize].chunks_exact_mut(2).zip(&codes) {
                    o.copy_from_slice(&c.to_le_bytes());
                }
            }
            slot.up.unmap();
            slot.up_mapped = false;
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            enc.copy_buffer_to_buffer(&slot.up, 0, &slot.pk, 0, pk_len);
            enc.copy_buffer_to_buffer(&slot.up, cd_off, &slot.cd, 0, cd_len.next_multiple_of(4));
            {
                let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                for (k, &(nx, ny)) in disp.iter().enumerate() {
                    let pipe = if k == 0 {
                        &self.unpack
                    } else if k == nk - 1 {
                        &self.pack
                    } else {
                        &self.substage
                    };
                    let po = ((ci * nk + k) as u64 * PSLOT) as u32;
                    let gofs = if k == 0 || k == nk - 1 { 0 } else { prog_off[k - 1] };
                    cp.set_pipeline(pipe);
                    cp.set_bind_group(0, &groups[si], &[po, gofs]);
                    cp.dispatch_workgroups(nx, ny, 1);
                }
            }
            enc.copy_buffer_to_buffer(&slot.pk, 0, &slot.down, 0, pk_len);
            enc.copy_buffer_to_buffer(&slot.cd, 0, &slot.down, cd_off, cd_len.next_multiple_of(4));
            let idx = self.queue.submit([enc.finish()]);
            slot.down.map_async(wgpu::MapMode::Read, .., |r| r.expect("map down"));
            slot.up.map_async(wgpu::MapMode::Write, .., |r| r.expect("map up"));
            slot.busy = Some((ci, idx));
        }
        for slot in slots.iter_mut() {
            finish(slot, store);
        }
        // pass statistics
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(stats, 0, stats_rb, 0, 16);
        let idx = self.queue.submit([enc.finish()]);
        stats_rb.map_async(wgpu::MapMode::Read, .., |r| r.expect("map stats"));
        self.wait(&idx);
        let st: [i32; 4] = {
            let v = stats_rb.get_mapped_range(..).expect("stats range");
            std::array::from_fn(|i| i32::from_le_bytes(v[4 * i..4 * i + 4].try_into().unwrap()))
        };
        stats_rb.unmap();
        store.fresh = false;
        store.base = base_out;
        if st[0] != i32::MIN {
            store.maxexp = st[0];
        }
        Ok(st)
    }
}
