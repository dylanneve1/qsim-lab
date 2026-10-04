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
//! controlled phases) is one pass over the threadgroup buffer: the terms are
//! grouped by a shared "pivot" condition so that inside a group the
//! log-factor is a sum of per-bit terms, tabulated (on the host) over the
//! low and high 6 bits of the buffer index; each amplitude adds two table
//! entries per matching group and does one `exp`/`sincos`.
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
struct Group {
    pmask: u32,
    ppat: u32,
    tab: u32,
    o0: u32,
    on: u32,
    lin: u32,
    pad: [u32; 2],
}

/// Outer factor of a pivot group: `f` when `(base & omask) == opat`.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
struct OTerm {
    omask: u32,
    opat: u32,
    f: [f32; 2],
}

/// One diagonal term in buffer coordinates: multiply by `f` where
/// `(e & imask) == ipat` and the outer bits match `(omask, opat)`.
#[derive(Clone, Copy, Debug)]
struct DTerm {
    imask: u32,
    ipat: u32,
    omask: u32,
    opat: u32,
    f: Complex64,
}

/// Splits a run of diagonal terms into pivot groups.
///
/// A term `f` on buffer indices with `(e & imask) == ipat` (and outer bits
/// matching `(omask, opat)`) joins one group:
/// - with an outer condition: the group of pivot `(imask, ipat)`, as an
///   outer factor (one scalar per threadgroup, computed on the GPU);
/// - otherwise the group of pivot `imask \ {d}` for one of its bits `d`,
///   as a factor depending on the single bit `d` (chosen so that as many
///   terms as possible share a pivot; a QFT's controlled phases after `H(j)`
///   all share pivot `{j}`).
///
/// Inside a group the single-bit factors multiply into two 64-entry tables
/// over the low and high 6 bits of the buffer index (`l <= 12`), computed
/// in f64 on the host. Returns (group headers, tables, outer factors); the
/// group's `tab` / `o0` are relative to the returned vectors.
fn diag_groups(terms: &[DTerm]) -> (Vec<Group>, Vec<[f32; 2]>, Vec<OTerm>) {
    use std::collections::BTreeMap;
    type Key = (u32, u32);
    #[derive(Clone, Copy)]
    enum Part {
        Outer(u32, u32),
        Const,
        Lin(u32, u32),
    }
    let cands = |t: &DTerm| -> Vec<(Key, Part)> {
        if t.omask != 0 {
            vec![((t.imask, t.ipat), Part::Outer(t.omask, t.opat))]
        } else if t.imask == 0 {
            vec![((0, 0), Part::Const)]
        } else {
            let mut v = Vec::new();
            let mut m = t.imask;
            while m != 0 {
                let d = m.trailing_zeros();
                m &= m - 1;
                let bit = 1u32 << d;
                v.push((
                    (t.imask & !bit, t.ipat & !bit),
                    Part::Lin(d, (t.ipat >> d) & 1),
                ));
            }
            v
        }
    };
    let mut count: BTreeMap<Key, usize> = BTreeMap::new();
    for t in terms {
        for (k, _) in cands(t) {
            *count.entry(k).or_default() += 1;
        }
    }
    #[derive(Default)]
    struct Acc {
        /// log-factor (theta, ln|f|) tables, exponentiated at the end
        tab: Vec<[f64; 2]>,
        lin: bool,
        outer: BTreeMap<(u32, u32), Complex64>,
    }
    let mut groups: BTreeMap<Key, Acc> = BTreeMap::new();
    for t in terms {
        let (key, part) = cands(t)
            .into_iter()
            .max_by_key(|(k, _)| count[k])
            .expect("at least one candidate");
        let g = groups.entry(key).or_insert_with(|| Acc {
            tab: vec![[0.0; 2]; 128],
            ..Default::default()
        });
        let v = [t.f.arg(), t.f.norm().ln()];
        match part {
            Part::Outer(om, op) => {
                *g.outer.entry((om, op)).or_insert(Complex64::new(1.0, 0.0)) *= t.f;
            }
            Part::Const => {
                g.lin = true;
                for x in g.tab.iter_mut().take(64) {
                    x[0] += v[0];
                    x[1] += v[1];
                }
            }
            Part::Lin(d, val) => {
                g.lin = true;
                let (off, local) = if d < 6 { (0, d) } else { (64, d - 6) };
                for x in 0..64u32 {
                    if (x >> local) & 1 == val {
                        g.tab[off + x as usize][0] += v[0];
                        g.tab[off + x as usize][1] += v[1];
                    }
                }
            }
        }
    }
    let mut hdrs = Vec::with_capacity(groups.len());
    let mut tabs = Vec::new();
    let mut oterms = Vec::new();
    for ((pmask, ppat), g) in groups {
        let tab = tabs.len() as u32;
        if g.lin {
            tabs.extend(g.tab.iter().map(|x| {
                let z = Complex64::from_polar(x[1].exp(), x[0]);
                [z.re as f32, z.im as f32]
            }));
        }
        let o0 = oterms.len() as u32;
        oterms.extend(g.outer.iter().map(|(&(omask, opat), f)| OTerm {
            omask,
            opat,
            f: [f.re as f32, f.im as f32],
        }));
        hdrs.push(Group {
            pmask,
            ppat,
            tab,
            o0,
            on: oterms.len() as u32 - o0,
            lin: g.lin as u32,
            pad: [0; 2],
        });
    }
    (hdrs, tabs, oterms)
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
const K_BATCH: u32 = 5;

/// Tuning knobs of the Metal executor.
#[derive(Clone, Debug)]
pub struct MetalConfig {
    /// Inner qubits per stage (threadgroup buffer of `2^tg_bits` amplitudes;
    /// at most 12 on Apple GPUs, whose threadgroup memory is 32 KiB).
    pub tg_bits: usize,
    /// Maximum number of non-contiguous (high) inner qubits per stage
    /// (2..=8).
    pub slots: usize,
    /// Threads per threadgroup.
    pub threads: usize,
    /// Multiply runs of single-qubit gates together first.
    pub fuse_1q: bool,
    /// Reorder diagonal terms into as few runs as possible.
    pub schedule_diag: bool,
    /// Apply up to this many consecutive uncontrolled single-qubit gates on
    /// distinct qubits as one op whose `2^batch` amplitudes per thread stay
    /// in registers (1 = off, at most 4). Shared-memory kernel only.
    pub batch: usize,
    /// Use the register-resident kernel: each thread keeps
    /// `2^tg_bits / threads` amplitudes in registers for the whole stage;
    /// gates on the buffer bits above `log2(threads)` are register-local,
    /// the lowest 5 bits use SIMD shuffles and the bits in between go
    /// through threadgroup memory. Best with `slots = tg_bits -
    /// log2(threads)`, so the high (gathered) qubits are the local ones.
    pub regs: bool,
}

impl Default for MetalConfig {
    fn default() -> Self {
        // Register kernel, 2^9 amplitudes per threadgroup of 64 threads
        // (8 per thread): best or within 10% of best for both QFT-26 and
        // brickwork-26 in a sweep on an M1 Pro (research/metal.md).
        MetalConfig {
            tg_bits: 9,
            slots: 4,
            threads: 64,
            fuse_1q: true,
            schedule_diag: true,
            batch: 3,
            regs: true,
        }
    }
}

/// A Metal device with the compiled kernels.
pub struct MetalSim {
    device: Device,
    queue: CommandQueue,
    stage: Mutex<HashMap<(bool, usize, usize), ComputePipelineState>>,
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
    regs: bool,
    tg_bits: usize,
    threads: usize,
    hdrs: Vec<StageHdr>,
    /// per stage: (U1 ops, swaps, diagonal runs, pivot groups, high inner qubits)
    stats: Vec<[usize; 5]>,
    /// (first op, group count) per stage
    spans: Vec<(usize, usize)>,
    ops: Buffer,
    groups: Buffer,
    tabs: Buffer,
    oterms: Buffer,
    mats: Buffer,
}

impl MetalPlan {
    /// Number of stages, i.e. full passes over the state vector.
    pub fn num_stages(&self) -> usize {
        self.hdrs.len()
    }

    /// Per stage: (U1 ops, swaps, diagonal runs, pivot groups, inner
    /// qubits outside the contiguous low range).
    pub fn stage_stats(&self) -> &[[usize; 5]] {
        &self.stats
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

impl From<SimError> for MetalError {
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

    fn stage_pso(
        &self,
        regs: bool,
        tg_bits: usize,
        tpg: usize,
    ) -> Result<ComputePipelineState, MetalError> {
        let mut map = self.stage.lock().unwrap();
        if let Some(p) = map.get(&(regs, tg_bits, tpg)) {
            return Ok(p.clone());
        }
        // QSIM_METAL_DEFINES: extra preprocessor lines for kernel experiments
        // (diagnostics only; e.g. "#define DBG_NO_OTERMS 1")
        let extra = std::env::var("QSIM_METAL_DEFINES").unwrap_or_default();
        let src = format!(
            "{}\n#define TG_BITS {tg_bits}\n#define TPG {tpg}\n#define TPG_BITS {}\n{SRC}",
            extra.replace(';', "\n"),
            tpg.trailing_zeros()
        );
        let fname = if regs { "stage_reg" } else { "stage" };
        let lib = self
            .device
            .new_library_with_source(&src, &CompileOptions::new())
            .map_err(|e| err(format!("Metal compile: {e}")))?;
        let f = lib
            .get_function(fname, None)
            .map_err(|e| err(format!("Metal function {fname}: {e}")))?;
        let p = self
            .device
            .new_compute_pipeline_state_with_function(&f)
            .map_err(|e| err(format!("Metal pipeline stage: {e}")))?;
        if (p.max_total_threads_per_threadgroup() as usize) < tpg {
            return Err(err(format!(
                "stage kernel supports at most {} threads per threadgroup",
                p.max_total_threads_per_threadgroup()
            )));
        }
        map.insert((regs, tg_bits, tpg), p.clone());
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
        // slots >= 2: a SWAP of two high qubits needs both in one stage
        // (plan_stages would otherwise emit a stage wider than tg_bits).
        if !(2..=12).contains(&cfg.tg_bits) || !(2..=8).contains(&cfg.slots) {
            return Err(err("MetalConfig: need 2 <= tg_bits <= 12 and 2 <= slots <= 8"));
        }
        if !(1..=4).contains(&cfg.batch) {
            return Err(err("MetalConfig: batch must be 1..=4"));
        }
        if !cfg.threads.is_power_of_two() || cfg.threads > 1024 {
            return Err(err("MetalConfig: threads must be a power of two <= 1024"));
        }
        let fused;
        let ops = if cfg.fuse_1q {
            fused = fuse_1q(ops, n, false);
            &fused[..]
        } else {
            ops
        };
        let l = cfg.tg_bits.min(n);
        // the register kernel has no batch op (its local gates are already
        // register-resident)
        let batch = if cfg.regs { 1 } else { cfg.batch };
        let stages = plan_stages(ops, n, l, cfg.slots);
        let mut hdrs = Vec::with_capacity(stages.len());
        let mut stats = Vec::with_capacity(stages.len());
        let mut spans = Vec::with_capacity(stages.len());
        let mut gops: Vec<OpG> = Vec::new();
        let mut groups: Vec<Group> = Vec::new();
        let mut tabs: Vec<[f32; 2]> = Vec::new();
        let mut oterms: Vec<OTerm> = Vec::new();
        let mut mats: Vec<[f32; 8]> = Vec::new();
        for st in &stages {
            let sops = if cfg.schedule_diag {
                schedule_diag(&st.ops)
            } else {
                st.ops.clone()
            };
            let l = st.inner.len();
            if l > cfg.tg_bits.min(n) || (cfg.regs && l != cfg.tg_bits.min(n)) {
                return Err(err(format!(
                    "stage needs {l} inner qubits, more than tg_bits = {}",
                    cfg.tg_bits
                )));
            }
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
                    KOp::U1 { q, m, ctrl: 0 }
                        if batch >= 2
                            && matches!(sops.get(i + 1), Some(KOp::U1 { ctrl: 0, q: q2, .. }) if *q2 != q) =>
                    {
                        // consecutive uncontrolled single-qubit gates on
                        // distinct qubits: one register-resident batch
                        let mut tg: Vec<(u32, Mat2)> = vec![(pos[q] as u32, m)];
                        let mut j = i + 1;
                        while tg.len() < batch {
                            match sops.get(j) {
                                Some(&KOp::U1 { q, m, ctrl: 0 })
                                    if tg.iter().all(|&(t, _)| t != pos[q] as u32) =>
                                {
                                    tg.push((pos[q] as u32, m));
                                    j += 1;
                                }
                                _ => break,
                            }
                        }
                        // the gates act on distinct qubits, so they commute
                        tg.sort_by_key(|&(t, _)| t);
                        let packed = tg.iter().enumerate().fold(0u32, |p, (k, &(t, _))| p | t << (8 * k));
                        let m0 = mats.len() as u32;
                        mats.extend(tg.iter().map(|(_, m)| mat_f32(m)));
                        gops.push(OpG {
                            kind: K_BATCH,
                            t: tg.len() as u32,
                            a: packed,
                            nterms: m0,
                            ..Default::default()
                        });
                        i = j;
                    }
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
                        let mut run = Vec::new();
                        while i < sops.len() {
                            let KOp::Phase { mask, pat, f } = sops[i] else {
                                break;
                            };
                            let (imask, ipat, omask, opat) = split_pat(mask, pat, &pos);
                            run.push(DTerm {
                                imask,
                                ipat,
                                omask,
                                opat,
                                f,
                            });
                            i += 1;
                        }
                        let (gh, gt, go) = diag_groups(&run);
                        let g0 = groups.len();
                        let toff = tabs.len() as u32;
                        let ooff = oterms.len() as u32;
                        groups.extend(gh.into_iter().map(|mut g| {
                            g.tab += toff;
                            g.o0 += ooff;
                            g
                        }));
                        tabs.extend(gt);
                        oterms.extend(go);
                        gops.push(OpG {
                            kind: K_DIAG,
                            a: g0 as u32,
                            nterms: (groups.len() - g0) as u32,
                            ..Default::default()
                        });
                    }
                }
            }
            hdr.nops = (gops.len() - first) as u32;
            let mut stt = [0usize, 0, 0, 0, l - b];
            for o in &gops[first..] {
                match o.kind {
                    K_SWAP => stt[1] += 1,
                    K_DIAG => {
                        stt[2] += 1;
                        stt[3] += o.nterms as usize;
                    }
                    K_BATCH => stt[0] += o.t as usize,
                    _ => stt[0] += 1,
                }
            }
            stats.push(stt);
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
        let group_bytes = unsafe {
            // SAFETY: Group is repr(C) plain old data.
            std::slice::from_raw_parts(
                groups.as_ptr() as *const u8,
                groups.len() * std::mem::size_of::<Group>(),
            )
        };
        let oterm_bytes = unsafe {
            // SAFETY: OTerm is repr(C) plain old data.
            std::slice::from_raw_parts(
                oterms.as_ptr() as *const u8,
                oterms.len() * std::mem::size_of::<OTerm>(),
            )
        };
        let mat_bytes = unsafe {
            // SAFETY: [f32; 8] is plain old data (two float4).
            std::slice::from_raw_parts(mats.as_ptr() as *const u8, mats.len() * 32)
        };
        let tab_bytes = unsafe {
            // SAFETY: [f32; 2] is plain old data (layout of float2).
            std::slice::from_raw_parts(tabs.as_ptr() as *const u8, tabs.len() * 8)
        };
        Ok(MetalPlan {
            n,
            regs: cfg.regs,
            // the register kernel is compiled for exactly 2^l elements; the
            // shared-memory kernel needs at least two elements per thread
            tg_bits: if cfg.regs { l } else { cfg.tg_bits },
            threads: if cfg.regs {
                cfg.threads.min(1 << l)
            } else {
                cfg.threads.min(1 << (cfg.tg_bits - 1))
            },
            hdrs,
            stats,
            spans,
            ops: mk(ops_bytes),
            groups: mk(group_bytes),
            tabs: mk(tab_bytes),
            oterms: mk(oterm_bytes),
            mats: mk(mat_bytes),
        })
    }

    /// Runs a compiled plan (one command buffer, one dispatch per stage) and
    /// waits for it.
    pub fn run(&self, s: &mut MetalState, plan: &MetalPlan) -> Result<(), MetalError> {
        self.run_range(s, plan, 0..plan.hdrs.len(), false)
    }

    /// Diagnostics: runs every stage in its own command buffer and returns
    /// the wall time of each. With `load_store_only`, the stages gather and
    /// scatter their amplitudes but apply no ops (the memory-traffic floor
    /// of the plan; the state is left unchanged).
    pub fn profile(
        &self,
        s: &mut MetalState,
        plan: &MetalPlan,
        load_store_only: bool,
    ) -> Result<Vec<f64>, MetalError> {
        (0..plan.hdrs.len())
            .map(|i| {
                let t = std::time::Instant::now();
                self.run_range(s, plan, i..i + 1, load_store_only)?;
                Ok(t.elapsed().as_secs_f64())
            })
            .collect()
    }

    fn run_range(
        &self,
        s: &mut MetalState,
        plan: &MetalPlan,
        range: std::ops::Range<usize>,
        load_store_only: bool,
    ) -> Result<(), MetalError> {
        if plan.n != s.n {
            return Err(err("plan compiled for another register size"));
        }
        if range.is_empty() {
            return Ok(());
        }
        let tpg = plan.threads;
        let pso = self.stage_pso(plan.regs, plan.tg_bits, tpg)?;
        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&s.buf), 0);
        enc.set_buffer(3, Some(&plan.groups), 0);
        enc.set_buffer(4, Some(&plan.tabs), 0);
        enc.set_buffer(5, Some(&plan.oterms), 0);
        enc.set_buffer(6, Some(&plan.mats), 0);
        for i in range {
            let (first, groups) = plan.spans[i];
            let mut h = plan.hdrs[i];
            if load_store_only {
                h.nops = 0;
            }
            enc.set_bytes(
                1,
                std::mem::size_of::<StageHdr>() as u64,
                &h as *const StageHdr as *const _,
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
