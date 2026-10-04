//! Metal (Apple GPU) backend for the dense state vector, single precision.
//!
//! Enabled with the `metal` cargo feature, macOS only. Apple GPUs have no
//! f64 arithmetic, so amplitudes are `Complex<f32>`.
//!
//! The state lives in one `MTLBuffer` with shared storage: on Apple silicon
//! the CPU and the GPU see the same physical memory, so
//! [`MetalState::amplitudes`] is a zero-copy view and there are no
//! host/device transfers.
//!
//! Execution reuses the CPU blocked executor's front end ([`crate::blocked`]):
//! gates are lowered to [`KOp`]s, runs of single-qubit gates are multiplied
//! together ([`fuse_1q`]), the op list is cut into stages of at most
//! `tg_bits` inner qubits ([`plan_stages`]) and diagonal terms are moved
//! together ([`schedule_diag`]). Each stage is one compute dispatch with one
//! threadgroup per assignment of the outer qubits: the threadgroup gathers
//! its `2^l` amplitudes into threadgroup memory (32 KiB = 4096 `float2` on
//! the M1), applies every op of the stage there and writes them back, so
//! DRAM is streamed once per stage. A run of diagonal terms (a QFT's
//! controlled phases) is one pass: each amplitude sums the log-factors of
//! the terms whose condition it meets and does one `exp`/`sincos`.
//!
//! [`MetalSim::apply_kops_naive`] is the unfused baseline: one dispatch over
//! the whole state per op.

use crate::blocked::{fuse_1q, lower_gates, plan_stages, schedule_diag, KOp};
use crate::circuit::{check_gate, Circuit, Op, SimError};
use crate::gate::{Gate, Mat2};
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions,
    MTLSize,
};
use num_complex::{Complex32, Complex64};
use std::collections::HashMap;
use std::sync::Mutex;

const SRC: &str = include_str!("metal_sv.metal");
const MAX_OUTER: usize = 32;
const MAX_HI: usize = 256;
/// Largest register: indices are 32-bit in the kernels and 2^31 `float2`
/// would be 16 GiB anyway.
pub const MAX_QUBITS: usize = 30;

#[repr(C)]
#[derive(Clone, Copy)]
struct StageHdr {
    l: u32,
    b: u32,
    nout: u32,
    nops: u32,
    outer: [u32; MAX_OUTER],
    hioff: [u32; MAX_HI],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct OpG {
    kind: u32,
    t: u32,
    cin: u32,
    omask: u32,
    opat: u32,
    a: u32,
    nterms: u32,
    pad: u32,
    m: [f32; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct Term {
    imask: u32,
    ipat: u32,
    omask: u32,
    opat: u32,
    lnmag: f32,
    theta: f32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GHdr {
    kind: u32,
    t: u32,
    ctrl: u32,
    a: u32,
    mask: u32,
    pat: u32,
    fre: f32,
    fim: f32,
    m: [f32; 8],
}

const K_COMPLEX: u32 = 0;
const K_REAL: u32 = 1;
const K_X: u32 = 2;
const K_SWAP: u32 = 3;
const K_DIAG: u32 = 4;

/// Tuning knobs of the Metal executor.
#[derive(Clone, Debug)]
pub struct MetalConfig {
    /// Inner qubits per stage (threadgroup buffer of `2^tg_bits` amplitudes;
    /// at most 12 on Apple GPUs, whose threadgroup memory is 32 KiB).
    pub tg_bits: usize,
    /// Maximum number of non-contiguous (high) inner qubits per stage
    /// (at most 8).
    pub slots: usize,
    /// Threads per threadgroup.
    pub threads: usize,
    /// Multiply runs of single-qubit gates together first.
    pub fuse_1q: bool,
    /// Reorder diagonal terms into as few runs as possible.
    pub schedule_diag: bool,
}

impl Default for MetalConfig {
    fn default() -> Self {
        MetalConfig {
            tg_bits: 12,
            slots: 6,
            threads: 512,
            fuse_1q: true,
            schedule_diag: true,
        }
    }
}

/// A Metal device with the compiled kernels.
pub struct MetalSim {
    device: Device,
    queue: CommandQueue,
    stage: Mutex<HashMap<usize, ComputePipelineState>>,
    gate_pairs: ComputePipelineState,
    gate_phase: ComputePipelineState,
    init_basis: ComputePipelineState,
    scale4: ComputePipelineState,
}

/// An `n`-qubit f32 state vector in a shared-storage Metal buffer.
pub struct MetalState {
    buf: Buffer,
    n: usize,
}

impl MetalState {
    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// Zero-copy view of the amplitudes (unified memory). Only valid while no
    /// GPU work on this state is in flight; every `MetalSim` method waits for
    /// its command buffer before returning.
    pub fn amplitudes(&self) -> &[Complex32] {
        unsafe {
            // SAFETY: the buffer holds 2^n Complex32 (8 bytes, align 4) and
            // is page aligned; Complex32 is repr(C) { re, im } like float2.
            std::slice::from_raw_parts(self.buf.contents() as *const Complex32, 1 << self.n)
        }
    }

    /// Mutable zero-copy view of the amplitudes (see [`MetalState::amplitudes`]).
    pub fn amplitudes_mut(&mut self) -> &mut [Complex32] {
        unsafe {
            // SAFETY: as in `amplitudes`; `&mut self` excludes other views.
            std::slice::from_raw_parts_mut(self.buf.contents() as *mut Complex32, 1 << self.n)
        }
    }
}

/// A gate list compiled for one register size: per-stage headers plus the
/// op and term tables in shared buffers.
pub struct MetalPlan {
    n: usize,
    tg_bits: usize,
    threads: usize,
    hdrs: Vec<StageHdr>,
    /// (first op, group count) per stage
    spans: Vec<(usize, usize)>,
    ops: Buffer,
    terms: Buffer,
}

impl MetalPlan {
    /// Number of stages, i.e. full passes over the state vector.
    pub fn num_stages(&self) -> usize {
        self.hdrs.len()
    }
}

/// Error of the Metal backend: a gate-validation error or a Metal/setup
/// failure.
#[derive(Debug)]
pub enum MetalError {
    Sim(SimError),
    Metal(String),
}

impl std::fmt::Display for MetalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetalError::Sim(e) => write!(f, "{e}"),
            MetalError::Metal(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for MetalError {}

impl From<MetalError> for MetalError {
    fn from(e: SimError) -> Self {
        MetalError::Sim(e)
    }
}

fn err(s: impl Into<String>) -> MetalError {
    MetalError::Metal(s.into())
}

fn mat_f32(m: &Mat2) -> [f32; 8] {
    [
        m[0][0].re as f32,
        m[0][0].im as f32,
        m[0][1].re as f32,
        m[0][1].im as f32,
        m[1][0].re as f32,
        m[1][0].im as f32,
        m[1][1].re as f32,
        m[1][1].im as f32,
    ]
}

/// Maps a physical bit mask to buffer bits (position in `inner`), returning
/// (buffer mask, leftover outer physical mask).
fn split_mask(mask: usize, pos: &[i32]) -> (u32, u32) {
    let mut inner = 0u32;
    let mut outer = 0u32;
    let mut m = mask;
    while m != 0 {
        let q = m.trailing_zeros() as usize;
        m &= m - 1;
        if pos[q] >= 0 {
            inner |= 1 << pos[q];
        } else {
            outer |= 1 << q;
        }
    }
    (inner, outer)
}

fn split_pat(mask: usize, pat: usize, pos: &[i32]) -> (u32, u32, u32, u32) {
    let (im, om) = split_mask(mask, pos);
    let (ip, op) = split_mask(pat & mask, pos);
    (im, ip, om, op)
}

impl MetalSim {
    /// Opens the system default Metal device and compiles the kernels.
    pub fn new() -> Result<Self, MetalError> {
        let device = Device::system_default().ok_or_else(|| err("no Metal device"))?;
        let queue = device.new_command_queue();
        let lib = device
            .new_library_with_source(SRC, &CompileOptions::new())
            .map_err(|e| err(format!("Metal compile: {e}")))?;
        let pso = |name: &str| -> Result<ComputePipelineState, MetalError> {
            let f = lib
                .get_function(name, None)
                .map_err(|e| err(format!("Metal function {name}: {e}")))?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| err(format!("Metal pipeline {name}: {e}")))
        };
        Ok(MetalSim {
            gate_pairs: pso("gate_pairs")?,
            gate_phase: pso("gate_phase")?,
            init_basis: pso("init_basis")?,
            scale4: pso("scale4")?,
            stage: Mutex::new(HashMap::new()),
            device,
            queue,
        })
    }

    /// Device name, e.g. "Apple M1 Pro".
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    fn stage_pso(&self, tg_bits: usize) -> Result<ComputePipelineState, MetalError> {
        let mut map = self.stage.lock().unwrap();
        if let Some(p) = map.get(&tg_bits) {
            return Ok(p.clone());
        }
        let src = format!("#define TG_BITS {tg_bits}\n{SRC}");
        let lib = self
            .device
            .new_library_with_source(&src, &CompileOptions::new())
            .map_err(|e| err(format!("Metal compile: {e}")))?;
        let f = lib
            .get_function("stage", None)
            .map_err(|e| err(format!("Metal function stage: {e}")))?;
        let p = self
            .device
            .new_compute_pipeline_state_with_function(&f)
            .map_err(|e| err(format!("Metal pipeline stage: {e}")))?;
        map.insert(tg_bits, p.clone());
        Ok(p)
    }

    /// Allocates an `n`-qubit state in `|0...0>`.
    pub fn alloc(&self, n: usize) -> Result<MetalState, MetalError> {
        if !(1..=MAX_QUBITS).contains(&n) {
            return Err(err(format!("Metal backend supports 1..={MAX_QUBITS} qubits")));
        }
        let bytes = 8u64 << n;
        if bytes > self.device.max_buffer_length() {
            return Err(err(format!(
                "state of {bytes} bytes exceeds the Metal buffer limit {}",
                self.device.max_buffer_length()
            )));
        }
        let buf = self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        let s = MetalState { buf, n };
        self.set_basis(&s, 0);
        Ok(s)
    }

    /// Sets the state to the basis state `|index>` (on the GPU).
    pub fn set_basis(&self, s: &MetalState, index: usize) {
        assert!(index < 1 << s.n);
        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.init_basis);
        enc.set_buffer(0, Some(&s.buf), 0);
        let idx = index as u32;
        enc.set_bytes(1, 4, &idx as *const u32 as *const _);
        let threads = (1u64 << s.n) / 2;
        let w = 256.min(threads.max(1));
        enc.dispatch_threads(MTLSize::new(threads.max(1), 1, 1), MTLSize::new(w, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    /// Bandwidth probe: `passes` in-place scalings of the whole state (one
    /// read and one write per amplitude per pass) in one command buffer.
    /// Returns the wall time in seconds.
    pub fn scale_passes(&self, s: &MetalState, passes: usize) -> f64 {
        let t = std::time::Instant::now();
        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.scale4);
        enc.set_buffer(0, Some(&s.buf), 0);
        let f = 1.0f32;
        enc.set_bytes(1, 4, &f as *const f32 as *const _);
        let threads = ((1u64 << s.n) / 2).max(1);
        for _ in 0..passes {
            enc.dispatch_threads(
                MTLSize::new(threads, 1, 1),
                MTLSize::new(256.min(threads), 1, 1),
            );
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        t.elapsed().as_secs_f64()
    }

    /// Compiles executor ops for an `n`-qubit register.
    pub fn compile(&self, n: usize, ops: &[KOp], cfg: &MetalConfig) -> Result<MetalPlan, MetalError> {
        if !(1..=MAX_QUBITS).contains(&n) {
            return Err(err(format!("Metal backend supports 1..={MAX_QUBITS} qubits")));
        }
        if !(1..=12).contains(&cfg.tg_bits) || cfg.slots > 8 || cfg.slots == 0 {
            return Err(err("MetalConfig: need 1 <= tg_bits <= 12 and 1 <= slots <= 8"));
        }
        let fused;
        let ops = if cfg.fuse_1q {
            fused = fuse_1q(ops, n, false);
            &fused[..]
        } else {
            ops
        };
        let l = cfg.tg_bits.min(n);
        let stages = plan_stages(ops, n, l, cfg.slots);
        let mut hdrs = Vec::with_capacity(stages.len());
        let mut spans = Vec::with_capacity(stages.len());
        let mut gops: Vec<OpG> = Vec::new();
        let mut terms: Vec<Term> = Vec::new();
        for st in &stages {
            let sops = if cfg.schedule_diag {
                schedule_diag(&st.ops)
            } else {
                st.ops.clone()
            };
            let l = st.inner.len();
            let mut pos = vec![-1i32; n];
            for (j, &q) in st.inner.iter().enumerate() {
                pos[q] = j as i32;
            }
            let b = st.inner.iter().enumerate().take_while(|(j, &q)| *j == q).count();
            if l - b > 8 {
                return Err(err("stage has more than 8 high inner qubits"));
            }
            let mut hdr = StageHdr {
                l: l as u32,
                b: b as u32,
                nout: (n - l) as u32,
                nops: 0,
                outer: [0; MAX_OUTER],
                hioff: [0; MAX_HI],
            };
            for (k, q) in (0..n).filter(|&q| pos[q] < 0).enumerate() {
                hdr.outer[k] = q as u32;
            }
            for (hi, off) in hdr.hioff.iter_mut().enumerate().take(1 << (l - b)) {
                let mut o = 0u32;
                for k in 0..(l - b) {
                    if hi >> k & 1 == 1 {
                        o |= 1 << st.inner[b + k];
                    }
                }
                *off = o;
            }
            let first = gops.len();
            let mut i = 0;
            while i < sops.len() {
                match sops[i] {
                    KOp::U1 { q, m, ctrl } => {
                        let (cin, cout) = split_mask(ctrl, &pos);
                        let t = pos[q];
                        assert!(t >= 0, "U1 target must be inner");
                        let x = m[0][0] == Complex64::new(0.0, 0.0)
                            && m[1][1] == Complex64::new(0.0, 0.0)
                            && m[0][1] == Complex64::new(1.0, 0.0)
                            && m[1][0] == Complex64::new(1.0, 0.0);
                        let real = m.iter().flatten().all(|z| z.im == 0.0);
                        gops.push(OpG {
                            kind: if x {
                                K_X
                            } else if real {
                                K_REAL
                            } else {
                                K_COMPLEX
                            },
                            t: t as u32,
                            cin,
                            omask: cout,
                            opat: cout,
                            m: mat_f32(&m),
                            ..Default::default()
                        });
                        i += 1;
                    }
                    KOp::Swap { a, b } => {
                        let (pa, pb) = (pos[a], pos[b]);
                        assert!(pa >= 0 && pb >= 0, "swap qubits must be inner");
                        let (lo, hi) = (pa.min(pb) as u32, pa.max(pb) as u32);
                        gops.push(OpG {
                            kind: K_SWAP,
                            t: lo,
                            a: hi,
                            ..Default::default()
                        });
                        i += 1;
                    }
                    KOp::Phase { .. } => {
                        let t0 = terms.len();
                        while i < sops.len() {
                            let KOp::Phase { mask, pat, f } = sops[i] else {
                                break;
                            };
                            let (imask, ipat, omask, opat) = split_pat(mask, pat, &pos);
                            terms.push(Term {
                                imask,
                                ipat,
                                omask,
                                opat,
                                lnmag: f.norm().ln() as f32,
                                theta: f.arg() as f32,
                                ..Default::default()
                            });
                            i += 1;
                        }
                        gops.push(OpG {
                            kind: K_DIAG,
                            a: t0 as u32,
                            nterms: (terms.len() - t0) as u32,
                            ..Default::default()
                        });
                    }
                }
            }
            hdr.nops = (gops.len() - first) as u32;
            spans.push((first, 1usize << (n - l)));
            hdrs.push(hdr);
        }
        let mk = |bytes: &[u8]| {
            if bytes.is_empty() {
                self.device.new_buffer(64, MTLResourceOptions::StorageModeShared)
            } else {
                self.device.new_buffer_with_data(
                    bytes.as_ptr() as *const _,
                    bytes.len() as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            }
        };
        let ops_bytes = unsafe {
            // SAFETY: OpG is repr(C) plain old data.
            std::slice::from_raw_parts(
                gops.as_ptr() as *const u8,
                gops.len() * std::mem::size_of::<OpG>(),
            )
        };
        let term_bytes = unsafe {
            // SAFETY: Term is repr(C) plain old data.
            std::slice::from_raw_parts(
                terms.as_ptr() as *const u8,
                terms.len() * std::mem::size_of::<Term>(),
            )
        };
        Ok(MetalPlan {
            n,
            tg_bits: cfg.tg_bits,
            threads: cfg.threads,
            hdrs,
            spans,
            ops: mk(ops_bytes),
            terms: mk(term_bytes),
        })
    }

    /// Runs a compiled plan (one command buffer, one dispatch per stage) and
    /// waits for it.
    pub fn run(&self, s: &mut MetalState, plan: &MetalPlan) -> Result<(), MetalError> {
        if plan.n != s.n {
            return Err(err("plan compiled for another register size"));
        }
        if plan.hdrs.is_empty() {
            return Ok(());
        }
        let pso = self.stage_pso(plan.tg_bits)?;
        let maxt = pso.max_total_threads_per_threadgroup() as usize;
        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&s.buf), 0);
        enc.set_buffer(3, Some(&plan.terms), 0);
        for (h, &(first, groups)) in plan.hdrs.iter().zip(&plan.spans) {
            // threads: no more than one per amplitude pair is useful
            let tpg = plan.threads.min(maxt).min((1usize << h.l).div_ceil(2)).max(1);
            enc.set_bytes(
                1,
                std::mem::size_of::<StageHdr>() as u64,
                h as *const StageHdr as *const _,
            );
            enc.set_buffer(2, Some(&plan.ops), (first * std::mem::size_of::<OpG>()) as u64);
            enc.dispatch_thread_groups(
                MTLSize::new(groups as u64, 1, 1),
                MTLSize::new(tpg as u64, 1, 1),
            );
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(())
    }

    /// Fused execution of executor ops (compile + run).
    pub fn apply_kops(
        &self,
        s: &mut MetalState,
        ops: &[KOp],
        cfg: &MetalConfig,
    ) -> Result<(), MetalError> {
        let plan = self.compile(s.n, ops, cfg)?;
        self.run(s, &plan)
    }

    /// Fused execution of a gate list.
    pub fn apply_gates(
        &self,
        s: &mut MetalState,
        gates: &[Gate],
        cfg: &MetalConfig,
    ) -> Result<(), MetalError> {
        for g in gates {
            check_gate(g, s.n)?;
        }
        self.apply_kops(s, &lower_gates(gates), cfg)
    }

    /// Fused execution of a circuit of unitary gates.
    pub fn apply_circuit(
        &self,
        s: &mut MetalState,
        c: &Circuit,
        cfg: &MetalConfig,
    ) -> Result<(), MetalError> {
        let gates = circuit_gates(c)?;
        self.apply_gates(s, &gates, cfg)
    }

    /// Unfused baseline: one dispatch over the whole state per op (no
    /// fusion, no blocking).
    pub fn apply_kops_naive(&self, s: &mut MetalState, ops: &[KOp]) -> Result<(), MetalError> {
        let n = s.n;
        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_buffer(0, Some(&s.buf), 0);
        for op in ops {
            let mut h = GHdr::default();
            let (pso, threads) = match *op {
                KOp::U1 { q, m, ctrl } => {
                    h.kind = 0;
                    h.t = q as u32;
                    h.ctrl = ctrl as u32;
                    h.m = mat_f32(&m);
                    (&self.gate_pairs, 1u64 << (n - 1))
                }
                KOp::Swap { a, b } => {
                    if n < 2 {
                        return Err(err("swap on a 1-qubit register"));
                    }
                    h.kind = 3;
                    h.t = a.min(b) as u32;
                    h.a = a.max(b) as u32;
                    (&self.gate_pairs, 1u64 << (n - 2))
                }
                KOp::Phase { mask, pat, f } => {
                    h.mask = mask as u32;
                    h.pat = (pat & mask) as u32;
                    h.fre = f.re as f32;
                    h.fim = f.im as f32;
                    (&self.gate_phase, 1u64 << n)
                }
            };
            enc.set_compute_pipeline_state(pso);
            enc.set_bytes(
                1,
                std::mem::size_of::<GHdr>() as u64,
                &h as *const GHdr as *const _,
            );
            enc.dispatch_threads(
                MTLSize::new(threads, 1, 1),
                MTLSize::new(256.min(threads), 1, 1),
            );
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(())
    }
}

/// The unitary gates of a circuit (error on measurements, noise, ...).
pub fn circuit_gates(c: &Circuit) -> Result<Vec<Gate>, MetalError> {
    c.ops
        .iter()
        .enumerate()
        .map(|(i, op)| match op {
            Op::Gate(g) => Ok(*g),
            _ => Err(MetalError::Sim(SimError::MeasurementNotSupported {
                backend: "metal",
                op_index: i,
            })),
        })
        .collect()
}
