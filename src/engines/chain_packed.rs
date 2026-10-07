//! Packed low-precision storage for the chain-sweep bond register.
//!
//! [`crate::engines::chain_lowprec`] measures the fidelity cost of storing
//! the register in `b`-bit block floating point by *emulating* the format
//! (an f32 register rounded after every pass). This module stores it for
//! real: the register lives in RAM as packed `b`-bit integers (`b` = 2..=8)
//! plus one scale per block of `B` amplitudes, and every memory pass
//! streams it: for each block of a stage of the blocked executor, the
//! stage's contiguous runs are unpacked into the executor's f32 buffer, the
//! stage's ops run in f32, and the buffer is re-quantized and packed back
//! in place. Peak memory is the packed store plus one `2^l` f32 buffer per
//! thread.
//!
//! The rounding is the emulator's own code (same block statistic, same
//! scale, same round-to-nearest-even, same decode `q * (1/s)` in f64 then
//! f32), so with the same stages and kernels the packed run is
//! bit-identical to the emulated one ([`run_emulated_stages`]); the tests
//! check that.
//!
//! Layout: block `k` covers register indices `kB..(k+1)B`; its `2B` real
//! components (re, im of each amplitude in turn) are stored as offset
//! binary `q + (2^(b-1) - 1)`, LSB first, in `B·b/4` bytes (`B` must be a
//! multiple of 4, at most [`MAX_BLOCK`]). Scales: the f32 step (`intB:bN`), or with `:h` a 16-bit
//! code (10-bit fraction of the 11-bit mantissa, 6-bit exponent relative to
//! a per-pass base; code 0 = all-zero block). The per-pass base is set from
//! the largest exponent of the previous pass, leaving 22 binades of
//! headroom above it and 40 below; blocks outside that window are counted
//! ([`PackedRun::underflow`] / [`PackedRun::overflow`]) and, below it,
//! stored as zero. Neither happens on the doped-Clifford circuit.

use crate::circuit::SimError;
use crate::engines::blocked::{
    plan_stages_lookahead, prepare_stage_cfg, run_compiled_on_block, run_prepared_on_block,
    BlockConfig, CompiledKOps, KOp, PreparedStage, Stage,
};
use crate::engines::chain_lowprec::{int_step, quantize, Format, LowPrec, Scaling};
use crate::engines::chain_sweep::SweepPlan;
use crate::engines::statevector::StateVector;
use num_complex::{Complex32, Complex64};
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};

/// Headroom (binades) of the `:h` exponent window above the previous
/// pass's largest block exponent.
const H_UP: i32 = 22;

/// Largest scale block (amplitudes) the packed store supports.
pub const MAX_BLOCK: usize = 1024;

/// Unpacks `out.len()` (a multiple of 8) `b`-bit codes stored LSB first.
#[inline]
fn unpack_codes(bytes: &[u8], b: u32, out: &mut [u8]) {
    match b {
        8 => out.copy_from_slice(bytes),
        4 => {
            for (o, &x) in out.chunks_exact_mut(2).zip(bytes) {
                o[0] = x & 15;
                o[1] = x >> 4;
            }
        }
        _ => {
            // 8 codes = b bytes
            let mask = (1u64 << b) - 1;
            for (o, g) in out.chunks_exact_mut(8).zip(bytes.chunks_exact(b as usize)) {
                let w = g
                    .iter()
                    .enumerate()
                    .fold(0u64, |w, (i, &x)| w | (x as u64) << (8 * i));
                for (t, o) in o.iter_mut().enumerate() {
                    *o = ((w >> (b as usize * t)) & mask) as u8;
                }
            }
        }
    }
}

/// Packs `codes.len()` (a multiple of 8) `b`-bit codes LSB first.
#[inline]
fn pack_codes(codes: &[u8], b: u32, bytes: &mut [u8]) {
    match b {
        8 => bytes.copy_from_slice(codes),
        4 => {
            for (x, c) in bytes.iter_mut().zip(codes.chunks_exact(2)) {
                *x = c[0] | c[1] << 4;
            }
        }
        _ => {
            for (g, c) in bytes.chunks_exact_mut(b as usize).zip(codes.chunks_exact(8)) {
                let w = c
                    .iter()
                    .enumerate()
                    .fold(0u64, |w, (t, &x)| w | (x as u64) << (b as usize * t));
                for (i, g) in g.iter_mut().enumerate() {
                    *g = (w >> (8 * i)) as u8;
                }
            }
        }
    }
}

/// A raw pointer that may be shared between the rayon workers of one pass:
/// every block writes a disjoint byte range.
#[derive(Clone, Copy)]
struct SyncPtr<T>(*mut T);
// SAFETY: only used for disjoint per-block writes inside one pass (see
// `PackedStore::run_stage`); the store outlives the pass.
unsafe impl<T> Sync for SyncPtr<T> {}
unsafe impl<T> Send for SyncPtr<T> {}

/// The packed register.
pub struct PackedStore {
    width: usize,
    lp: LowPrec,
    bits: u32,
    block: usize,
    bpb: usize,
    maxv: f64,
    data: Vec<u8>,
    s32: Vec<f32>,
    s16: Vec<u16>,
    /// Still `|0..0>` (nothing stored yet).
    fresh: bool,
    /// `:h` exponent base of the blocks currently stored.
    base: i32,
    /// Largest block exponent of the blocks currently stored.
    maxexp: i32,
    underflow: u64,
    overflow: u64,
    /// Thread-nanoseconds spent unpacking, computing and packing.
    thread_ns: [u64; 3],
}

fn exp_of(x: f64) -> i32 {
    ((x.to_bits() >> 52) & 0x7ff) as i32 - 1023
}

impl PackedStore {
    /// Whether `lp` can be stored packed: an integer format of 2..=8 bits
    /// with per-block scaling (`B` a multiple of 4), round to nearest.
    pub fn supports(lp: &LowPrec) -> bool {
        matches!(lp.format, Format::Int(b) if (2..=8).contains(&b))
            && matches!(lp.scaling, Scaling::Block(b) if b % 4 == 0 && b <= MAX_BLOCK)
            && !lp.stochastic
    }

    /// Bytes a `width`-bit register takes in format `lp` (data + scales).
    pub fn bytes_for(width: usize, lp: &LowPrec) -> Option<usize> {
        if !Self::supports(lp) {
            return None;
        }
        let (Format::Int(b), Scaling::Block(bl)) = (lp.format, lp.scaling) else {
            return None;
        };
        let n = 1usize << width;
        let nb = n.div_ceil(bl);
        Some(nb * bl * b as usize / 4 + nb * if lp.half_scale { 2 } else { 4 })
    }

    /// Allocates a `width`-bit register in `|0..0>`; fails cleanly when the
    /// memory is not available.
    pub fn new(width: usize, lp: &LowPrec) -> Result<Self, SimError> {
        let (Format::Int(bits), Scaling::Block(block)) = (lp.format, lp.scaling) else {
            return Err(SimError::NotSupported {
                what: "packed storage needs an intB:bN format",
            });
        };
        if !Self::supports(lp) {
            return Err(SimError::NotSupported {
                what: "packed storage needs intB:bN with B in 2..=8, N a multiple of 4, no :sr",
            });
        }
        let n = 1usize << width;
        let block = block.min(n);
        if n % block != 0 || block % 4 != 0 {
            return Err(SimError::NotSupported {
                what: "packed storage: the scale block does not tile the register",
            });
        }
        let nb = n / block;
        let bpb = block * bits as usize / 4;
        let total = (nb * bpb + nb * if lp.half_scale { 2 } else { 4 }) as u128;
        let alloc = |_: std::collections::TryReserveError| SimError::TooLarge {
            what: "packed chain-sweep store (allocation failed)",
            bytes: total,
            limit: 0,
        };
        let mut data = Vec::new();
        data.try_reserve_exact(nb * bpb).map_err(alloc)?;
        data.resize(nb * bpb, 0u8);
        let (mut s32, mut s16) = (Vec::new(), Vec::new());
        if lp.half_scale {
            s16.try_reserve_exact(nb).map_err(alloc)?;
            s16.resize(nb, 0u16);
        } else {
            s32.try_reserve_exact(nb).map_err(alloc)?;
            s32.resize(nb, 0f32);
        }
        let maxv = ((1u64 << (bits - 1)) - 1) as f64;
        Ok(PackedStore {
            width,
            lp: *lp,
            bits,
            block,
            bpb,
            maxv,
            data,
            s32,
            s16,
            fresh: true,
            base: 0,
            maxexp: exp_of(1.0 / maxv),
            underflow: 0,
            overflow: 0,
            thread_ns: [0; 3],
        })
    }

    /// Bytes held (data + scales).
    pub fn bytes(&self) -> usize {
        self.data.len() + 4 * self.s32.len() + 2 * self.s16.len()
    }

    /// Step of block `k` (0 = all-zero block), stored with exponent base `base`.
    #[inline]
    fn step(s32: *const f32, s16: *const u16, half: bool, base: i32, k: usize) -> f64 {
        // SAFETY: k < number of blocks (callers index within the register).
        unsafe {
            if half {
                let c = *s16.add(k);
                if c == 0 {
                    0.0
                } else {
                    let e = (c >> 10) as i32 - 1 + base;
                    let m = 1024 + (c & 0x3ff) as i64;
                    m as f64 * f64::from_bits(((e - 10 + 1023) as u64) << 52)
                }
            } else {
                *s32.add(k) as f64
            }
        }
    }

    /// Decodes `out.len()` amplitudes starting at register index `start`
    /// (a whole number of blocks).
    #[inline]
    fn unpack_run(&self, ptrs: &Ptrs, start: usize, out: &mut [Complex32]) {
        if self.fresh {
            out.fill(Complex32::new(0.0, 0.0));
            if start == 0 {
                out[0] = Complex32::new(1.0, 0.0);
            }
            return;
        }
        let b = self.bits;
        let off = self.maxv as i32;
        let mut codes = [0u8; 2 * MAX_BLOCK];
        let codes = &mut codes[..2 * self.block];
        for (j, out) in out.chunks_exact_mut(self.block).enumerate() {
            let k = start / self.block + j;
            let step = Self::step(ptrs.s32.0, ptrs.s16.0, self.lp.half_scale, self.base, k);
            if step == 0.0 {
                out.fill(Complex32::new(0.0, 0.0));
                continue;
            }
            let s = 1.0 / step;
            let inv = 1.0 / s;
            // SAFETY: block k's bytes lie inside `data`.
            let bytes =
                unsafe { std::slice::from_raw_parts(ptrs.data.0.add(k * self.bpb), self.bpb) };
            unpack_codes(bytes, b, codes);
            for (z, c) in out.iter_mut().zip(codes.chunks_exact(2)) {
                z.re = ((c[0] as i32 - off) as f64 * inv) as f32;
                z.im = ((c[1] as i32 - off) as f64 * inv) as f32;
            }
        }
    }

    /// Quantizes and stores `v.len()` amplitudes at register index `start`.
    #[inline]
    fn pack_run(&self, ptrs: &Ptrs, start: usize, v: &[Complex32], st: &PassStats) {
        let b = self.bits;
        let off = self.maxv as i32;
        let mut codes = [0u8; 2 * MAX_BLOCK];
        let codes = &mut codes[..2 * self.block];
        // one atomic update per run, not per block (contention)
        let mut maxexp = i32::MIN;
        for (j, v) in v.chunks_exact(self.block).enumerate() {
            let k = start / self.block + j;
            // the emulator's block statistic, in f32
            let m = v
                .iter()
                .fold(0.0f32, |m, z| m.max(z.re.abs()).max(z.im.abs())) as f64;
            let mut step = if m == 0.0 || !m.is_finite() {
                0.0
            } else {
                int_step(m, &self.lp)
            };
            // SAFETY: block k's scale slot and bytes are written by this
            // block only (blocks of a pass are disjoint).
            unsafe {
                if self.lp.half_scale {
                    let mut code = 0u16;
                    if step != 0.0 {
                        let e = exp_of(step);
                        let ec = e - ptrs.base_out + 1;
                        maxexp = maxexp.max(e);
                        if ec < 1 {
                            st.underflow.fetch_add(1, Ordering::Relaxed);
                            step = 0.0;
                        } else if ec > 63 {
                            st.overflow.fetch_add(1, Ordering::Relaxed);
                            step = 0.0;
                        } else {
                            let frac = ((step.to_bits() >> 42) & 0x3ff) as u16;
                            code = ((ec as u16) << 10) | frac;
                            debug_assert_eq!(
                                Self::step(std::ptr::null(), &code, true, ptrs.base_out, 0),
                                step
                            );
                        }
                    }
                    *ptrs.s16.0.add(k) = code;
                } else {
                    *ptrs.s32.0.add(k) = step as f32;
                }
            }
            // SAFETY: as above.
            let bytes =
                unsafe { std::slice::from_raw_parts_mut(ptrs.data.0.add(k * self.bpb), self.bpb) };
            if step == 0.0 {
                // a zero block stores offset-binary zero everywhere
                codes.fill(off as u8);
            } else {
                // `round_to` for an integer format, inlined: the nearest
                // integer (ties to even), saturated at ±maxv
                let s = 1.0 / step;
                let maxv = self.maxv;
                for (c, z) in codes.chunks_exact_mut(2).zip(v) {
                    c[0] = ((z.re as f64 * s).round_ties_even().clamp(-maxv, maxv) as i32 + off)
                        as u8;
                    c[1] = ((z.im as f64 * s).round_ties_even().clamp(-maxv, maxv) as i32 + off)
                        as u8;
                }
            }
            pack_codes(codes, b, bytes);
        }
        if maxexp != i32::MIN {
            st.maxexp.fetch_max(maxexp, Ordering::Relaxed);
        }
    }

    fn ptrs(&mut self, base_out: i32) -> Ptrs {
        Ptrs {
            data: SyncPtr(self.data.as_mut_ptr()),
            s32: SyncPtr(self.s32.as_mut_ptr()),
            s16: SyncPtr(self.s16.as_mut_ptr()),
            base_out,
        }
    }

    /// Runs one stage (one memory pass) on the store.
    pub fn run_stage(&mut self, st: &Stage, cfg: &BlockConfig) -> Result<(), SimError> {
        let ex = StageExec::new(st, self.width, cfg);
        self.run_exec(&ex)
    }

    fn run_exec(&mut self, ex: &StageExec) -> Result<(), SimError> {
        if (1usize << ex.bc) < self.block {
            return Err(SimError::NotSupported {
                what: "packed storage: a stage's contiguous runs are shorter than the scale block",
            });
        }
        let base_out = self.maxexp + H_UP - 62;
        let ptrs = self.ptrs(base_out);
        let stats = PassStats {
            maxexp: AtomicI32::new(i32::MIN),
            underflow: AtomicU64::new(0),
            overflow: AtomicU64::new(0),
        };
        let me = &*self;
        let t = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
        (0..ex.nblocks()).into_par_iter().for_each(|c| {
            with_block_buf(ex.l, |buf| {
                let t0 = std::time::Instant::now();
                ex.for_each_run(c, |reg, off, len| {
                    me.unpack_run(&ptrs, reg, &mut buf[off..off + len])
                });
                let t1 = std::time::Instant::now();
                ex.compute(buf, c);
                let t2 = std::time::Instant::now();
                ex.for_each_run(c, |reg, off, len| {
                    me.pack_run(&ptrs, reg, &buf[off..off + len], &stats)
                });
                let t3 = std::time::Instant::now();
                for (a, d) in t.iter().zip([t1 - t0, t2 - t1, t3 - t2]) {
                    a.fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
                }
            })
        });
        for (a, d) in self.thread_ns.iter_mut().zip(&t) {
            *a += d.load(Ordering::Relaxed);
        }
        self.fresh = false;
        self.base = base_out;
        let mx = stats.maxexp.load(Ordering::Relaxed);
        if mx != i32::MIN {
            self.maxexp = mx;
        }
        self.underflow += stats.underflow.load(Ordering::Relaxed);
        self.overflow += stats.overflow.load(Ordering::Relaxed);
        Ok(())
    }

    /// The amplitude at register index `i`.
    pub fn amplitude(&mut self, i: usize) -> Complex32 {
        let start = i / self.block * self.block;
        let mut v = vec![Complex32::new(0.0, 0.0); self.block];
        let ptrs = self.ptrs(0);
        self.unpack_run(&ptrs, start, &mut v);
        v[i - start]
    }

    /// Decodes the whole register (tests and small registers only).
    pub fn to_vec(&mut self) -> Vec<Complex32> {
        let mut v = vec![Complex32::new(0.0, 0.0); 1usize << self.width];
        let ptrs = self.ptrs(0);
        self.unpack_run(&ptrs, 0, &mut v);
        v
    }
}

struct Ptrs {
    data: SyncPtr<u8>,
    s32: SyncPtr<f32>,
    s16: SyncPtr<u16>,
    base_out: i32,
}

struct PassStats {
    maxexp: AtomicI32,
    underflow: AtomicU64,
    overflow: AtomicU64,
}

/// Spreads the low bits of `x` over the set bits of `mask` (software pdep).
#[inline]
fn deposit(mut x: usize, mut mask: usize) -> usize {
    let mut out = 0;
    while mask != 0 && x != 0 {
        let b = mask & mask.wrapping_neg();
        if x & 1 == 1 {
            out |= b;
        }
        x >>= 1;
        mask &= mask - 1;
    }
    out
}

thread_local! {
    static BLOCK_BUF: RefCell<Vec<Complex32>> = const { RefCell::new(Vec::new()) };
}

/// Runs `f` on this thread's `2^l`-amplitude gather buffer.
fn with_block_buf<R>(l: usize, f: impl FnOnce(&mut [Complex32]) -> R) -> R {
    BLOCK_BUF.with(|b| {
        let mut b = b.borrow_mut();
        if b.len() != 1 << l {
            *b = Vec::new();
            b.resize(1 << l, Complex32::new(0.0, 0.0));
        }
        f(&mut b[..])
    })
}

/// How a block's ops are run on the gather buffer.
enum Compute {
    /// The stage prepared as one block (bit-identical to the emulator's
    /// `run_lowprec` when the stage covers the whole register).
    Direct(PreparedStage<f32>),
    /// The stage's ops re-planned with the cache-blocked executor inside
    /// the buffer (L2-sized sub-blocks): the fast path for big buffers.
    Nested(CompiledKOps<f32>),
}

/// One stage, compiled for gathered execution: buffer bit `j` is register
/// qubit `inner[j]`; the ops are renumbered so that the inner qubits are
/// `0..l` and the outer ones `l..width` (in order), which makes every
/// block a contiguous range of the renumbered register.
struct StageExec {
    l: usize,
    width: usize,
    inner_mask: usize,
    outer_mask: usize,
    bc: usize,
    compute: Compute,
    simd: bool,
}

fn remap_mask(m: usize, pos: &[usize]) -> usize {
    let mut out = 0;
    let mut x = m;
    while x != 0 {
        let q = x.trailing_zeros() as usize;
        out |= 1 << pos[q];
        x &= x - 1;
    }
    out
}

impl StageExec {
    fn new(st: &Stage, width: usize, cfg: &BlockConfig) -> Self {
        let l = st.inner.len();
        let mut pos = vec![usize::MAX; width];
        for (j, &q) in st.inner.iter().enumerate() {
            pos[q] = j;
        }
        let mut k = l;
        for p in pos.iter_mut() {
            if *p == usize::MAX {
                *p = k;
                k += 1;
            }
        }
        let ops: Vec<KOp> = st
            .ops
            .iter()
            .map(|o| match *o {
                KOp::U1 { q, m, ctrl } => KOp::U1 {
                    q: pos[q],
                    m,
                    ctrl: remap_mask(ctrl, &pos),
                },
                KOp::Phase { mask, pat, f } => KOp::Phase {
                    mask: remap_mask(mask, &pos),
                    pat: remap_mask(pat, &pos),
                    f,
                },
                KOp::Swap { a, b } => KOp::Swap {
                    a: pos[a],
                    b: pos[b],
                },
            })
            .collect();
        let inner_mask = st.inner.iter().fold(0usize, |m, &q| m | 1 << q);
        let full = if width >= usize::BITS as usize {
            usize::MAX
        } else {
            (1usize << width) - 1
        };
        let nested = BlockConfig {
            fuse_1q: false,
            ..cfg.clone()
        };
        let l2 = nested.block_bits(width, std::mem::size_of::<Complex32>());
        let compute = if l2 + 2 <= l && std::env::var_os("CS_DIRECT").is_none() {
            Compute::Nested(StateVector::<f32>::compile_kops(width, &ops, &nested))
        } else {
            Compute::Direct(prepare_stage_cfg::<f32>(
                &Stage {
                    inner: (0..l).collect(),
                    ops,
                },
                width,
                cfg,
            ))
        };
        StageExec {
            l,
            width,
            inner_mask,
            outer_mask: full & !inner_mask,
            bc: inner_mask.trailing_ones() as usize,
            compute,
            simd: cfg.simd,
        }
    }

    fn nblocks(&self) -> usize {
        1usize << (self.width - self.l)
    }

    /// Calls `f(register index, buffer offset, length)` for every
    /// contiguous run of block `c`.
    #[inline]
    fn for_each_run(&self, c: usize, mut f: impl FnMut(usize, usize, usize)) {
        let base = deposit(c, self.outer_mask);
        let runlen = 1usize << self.bc;
        for r in 0..1usize << (self.l - self.bc) {
            f(base | deposit(r << self.bc, self.inner_mask), r << self.bc, runlen);
        }
    }

    /// Runs the stage's ops on the gathered block `c`.
    fn compute(&self, buf: &mut [Complex32], c: usize) {
        let hi = c << self.l;
        match &self.compute {
            Compute::Direct(p) => run_prepared_on_block(buf, p, self.simd, hi),
            Compute::Nested(p) => run_compiled_on_block(p, buf, hi),
        }
    }
}

/// Stages (memory passes) for a packed sweep: the blocked executor's
/// look-ahead stage planner with blocks of `2^l` amplitudes of which at most
/// `slots` qubits are gathered (the low `l - slots` qubits are contiguous
/// and must hold at least one scale block). `ops` are used as given (fuse
/// them first with [`crate::engines::blocked::fuse_1q`] if wanted).
pub fn packed_stages(ops: &[KOp], width: usize, l: usize, slots: usize) -> Vec<Stage> {
    plan_stages_lookahead(ops, width, l.min(width), slots)
}

/// Result of a packed sweep.
#[derive(Clone, Debug)]
pub struct PackedRun {
    /// The amplitude (times the plan's scale).
    pub amp: Complex64,
    /// Memory passes (= roundings).
    pub passes: usize,
    /// Bytes of the packed store.
    pub store_bytes: usize,
    /// `:h` blocks below / above the per-pass exponent window.
    pub underflow: u64,
    /// See `underflow`.
    pub overflow: u64,
    /// Wall-clock seconds of the passes.
    pub secs: f64,
    /// Thread-seconds spent unpacking, computing and packing.
    pub thread_secs: [f64; 3],
}

/// Runs `stages` on a packed register in format `lp` and returns the
/// `|0..0>` amplitude times `plan.scale`. `cfg` is used for stage
/// preparation and kernels (its block size and fusion settings are not:
/// the stages are given).
pub fn run_packed(
    plan: &SweepPlan,
    stages: &[Stage],
    lp: &LowPrec,
    cfg: &BlockConfig,
) -> Result<PackedRun, SimError> {
    let mut store = PackedStore::new(plan.width, lp)?;
    let t = std::time::Instant::now();
    let progress = std::env::var_os("CS_PROGRESS").is_some();
    for (j, st) in stages.iter().enumerate() {
        store.run_stage(st, cfg)?;
        if progress {
            eprintln!(
                "  pass {}/{} t={:.1}s",
                j + 1,
                stages.len(),
                t.elapsed().as_secs_f64()
            );
        }
    }
    let secs = t.elapsed().as_secs_f64();
    let a = store.amplitude(0);
    Ok(PackedRun {
        amp: Complex64::new(a.re as f64, a.im as f64) * plan.scale,
        passes: stages.len(),
        store_bytes: store.bytes(),
        underflow: store.underflow,
        overflow: store.overflow,
        secs,
        thread_secs: store.thread_ns.map(|t| t as f64 * 1e-9),
    })
}

/// The emulated reference for [`run_packed`]: an f32 register, the same
/// stages gathered and computed block by block exactly as in the packed
/// run, rounded with the emulator's [`quantize`] after every stage. The
/// packed run must agree with it bit for bit.
pub fn run_emulated_stages(
    plan: &SweepPlan,
    stages: &[Stage],
    lp: &LowPrec,
    cfg: &BlockConfig,
) -> Result<Complex64, SimError> {
    let mut sv = StateVector::<f32>::try_new(plan.width)?;
    for st in stages {
        let ex = StageExec::new(st, plan.width, cfg);
        let amps = SyncPtr(sv.amplitudes_mut().as_mut_ptr());
        (0..ex.nblocks()).into_par_iter().for_each(|c| {
            let amps = amps;
            with_block_buf(ex.l, |buf| {
                // SAFETY: blocks are disjoint sets of register indices.
                ex.for_each_run(c, |reg, off, len| unsafe {
                    buf[off..off + len]
                        .copy_from_slice(std::slice::from_raw_parts(amps.0.add(reg), len))
                });
                ex.compute(buf, c);
                ex.for_each_run(c, |reg, off, len| unsafe {
                    std::slice::from_raw_parts_mut(amps.0.add(reg), len)
                        .copy_from_slice(&buf[off..off + len])
                });
            })
        });
        quantize(sv.amplitudes_mut(), lp);
    }
    Ok(sv.amplitude(0) * plan.scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize, seed: u64) -> Vec<Complex32> {
        let mut z = seed;
        (0..n)
            .map(|i| {
                z = z.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let a = ((z >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * (1.0 + i as f32 / 7.0);
                z = z.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let b = ((z >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 3e-3;
                Complex32::new(a, b)
            })
            .collect()
    }

    /// Pack + unpack of arbitrary data equals the emulator's `quantize`.
    #[test]
    fn roundtrip_matches_quantize() {
        for f in ["int4:b16:h", "int5:b64", "int5:b16:h", "int6:b64", "int3:b16", "int8:b256"] {
            let lp = LowPrec::parse(f).unwrap();
            let w = 10;
            let mut v = ramp(1 << w, 7);
            v[40..60].fill(Complex32::new(0.0, 0.0)); // zero blocks
            v[300] = Complex32::new(1e-9, 0.0); // tiny block maximum
            let mut q = v.clone();
            quantize(&mut q, &lp);
            let mut s = PackedStore::new(w, &lp).unwrap();
            s.fresh = false;
            let ptrs = s.ptrs(s.maxexp + H_UP - 62);
            let st = PassStats {
                maxexp: AtomicI32::new(i32::MIN),
                underflow: AtomicU64::new(0),
                overflow: AtomicU64::new(0),
            };
            s.pack_run(&ptrs, 0, &v, &st);
            s.base = ptrs.base_out;
            assert_eq!(st.overflow.load(Ordering::Relaxed), 0);
            let got = s.to_vec();
            if st.underflow.load(Ordering::Relaxed) == 0 {
                assert_eq!(got, q, "{f}");
            } else {
                // only the tiny block may differ (flushed to zero)
                assert!(lp.half_scale, "{f}");
                for (i, (a, b)) in got.iter().zip(&q).enumerate() {
                    if i / lp_block(&lp) != 300 / lp_block(&lp) {
                        assert_eq!(a, b, "{f} at {i}");
                    }
                }
            }
            assert_eq!(s.bytes(), PackedStore::bytes_for(w, &lp).unwrap());
        }
    }

    fn lp_block(lp: &LowPrec) -> usize {
        match lp.scaling {
            Scaling::Block(b) => b,
            _ => unreachable!(),
        }
    }

    #[test]
    fn sizes() {
        let lp = LowPrec::parse("int4:b16:h").unwrap();
        // 2^35 amplitudes: 32 GiB data + 4 GiB scales
        assert_eq!(PackedStore::bytes_for(35, &lp), Some(36 << 30));
        let lp = LowPrec::parse("int5:b64").unwrap();
        assert_eq!(PackedStore::bytes_for(35, &lp), Some(42 << 30));
        assert!(!PackedStore::supports(&LowPrec::parse("int4:b64:sr").unwrap()));
        assert!(!PackedStore::supports(&LowPrec::parse("bf16").unwrap()));
        assert!(!PackedStore::supports(&LowPrec::parse("int5:b6").unwrap()));
    }

    #[test]
    fn identity_stage_keeps_basis_state() {
        let lp = LowPrec::parse("int4:b16:h").unwrap();
        let mut s = PackedStore::new(8, &lp).unwrap();
        let st = Stage {
            inner: (0..8).collect(),
            ops: vec![],
        };
        s.run_stage(&st, &BlockConfig::default()).unwrap();
        let v = s.to_vec();
        assert!((v[0].re - 1.0).abs() < 1e-3 && v[1..].iter().all(|z| z.norm() == 0.0));
    }
}
