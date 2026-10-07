//! GPU (wgpu: Vulkan / DX12) backend of the packed chain sweep.
//!
//! [`crate::engines::chain_packed`] keeps the bond register in RAM as
//! packed `b`-bit integers with one scale per block and streams it once per
//! memory pass. This module runs the passes on a GPU: per pass the packed
//! register is streamed through the GPU in chunks of gathered blocks
//! (host gather → upload → unpack → the stage's ops in f32 → re-quantize
//! and pack → download → host scatter), several chunks in flight.
//!
//! **Bit-exactness.** The result is bit-identical to
//! [`crate::engines::chain_packed::run_packed`] with the same stages and a
//! config accepted by [`cfg_exportable`] (FMA kernel tier, i.e. AVX-512
//! disabled, no dense fusion, no L1 tiling):
//! - the ops are the CPU's own compiled plan per gathered block
//!   ([`gpu_stage_plan`]), whose per-element arithmetic is a fixed sequence
//!   of f32 `fma`/`mul` that the shader repeats; the only f64 part of the
//!   CPU kernels (the diagonal groups' factor tables) is precomputed on the
//!   host for every activity pattern;
//! - the format is `intB:bN:h` (`N` a multiple of 16). Its f64 encode /
//!   decode is reproduced without f64 on the GPU: the block step's 11-bit
//!   mantissa and exponent come from integer arithmetic on the block
//!   maximum ([`step_parts`]), and the rounding `round(x · (1/step))` /
//!   decode `q · (1/(1/step))` are tables over the 1024 step mantissas
//!   ([`HalfCodec`]): thresholds and decoded values at a normalised step,
//!   moved to the block's exponent by exact power-of-two scaling.
//!
//! [`emulate_packed`] runs exactly the GPU's pipeline on the CPU (same
//! plan export, same tables, same per-element order); the tests check it
//! against `run_packed` bit for bit, and the GPU against it.

use crate::circuit::SimError;
use crate::engines::blocked::gpu_export::{run_sub_stage_ref, GpuSubStage};
use crate::engines::blocked::{BlockConfig, Stage};
use crate::engines::chain_lowprec::{round_to, Format, LowPrec, Scaling};
use crate::engines::chain_packed::{deposit, gpu_stage_plan, GpuStagePlan, H_UP};
use crate::engines::chain_sweep::SweepPlan;
use num_complex::{Complex32, Complex64};
use rayon::prelude::*;

pub use crate::engines::blocked::gpu_export::cfg_exportable;

#[cfg(feature = "wgpu")]
pub mod gpu;

/// Number of 11-bit step mantissas (`1024..2048`).
pub const NMANT: usize = 1024;

/// Encode / decode tables of an `intB:bN:h` format.
#[derive(Clone, Debug)]
pub struct HalfCodec {
    /// Bits per component.
    pub bits: u32,
    /// Amplitudes per scale block.
    pub block: usize,
    /// Largest stored magnitude, `2^(bits-1) - 1`.
    pub maxv: u32,
    /// `dec[mant * (2 maxv + 1) + q + maxv]`: the CPU's decoded value of
    /// `q` for the step `1024 + mant` (exponent 10).
    pub dec: Vec<f32>,
    /// `thr[mant * maxv + n]`: smallest `x >= 0` (f32) that the CPU rounds
    /// to at least `n + 1` for the step `1024 + mant`.
    pub thr: Vec<f32>,
}

/// The CPU's integer for `x` at step `step` (`round_to(x * (1/step))`).
fn cpu_q(x: f32, step: f64, format: Format) -> f64 {
    let s = 1.0 / step;
    round_to(x as f64 * s, format, None)
}

impl HalfCodec {
    /// Tables for `lp`; fails unless it is `intB:bN:h` with `B` in 2..=8,
    /// `N` a multiple of 16 and round to nearest.
    pub fn new(lp: &LowPrec) -> Result<Self, SimError> {
        let (Format::Int(bits), Scaling::Block(block)) = (lp.format, lp.scaling) else {
            return Err(SimError::NotSupported {
                what: "GPU packed storage needs an intB:bN:h format",
            });
        };
        if !(2..=8).contains(&bits) || block % 16 != 0 || !lp.half_scale || lp.stochastic {
            return Err(SimError::NotSupported {
                what: "GPU packed storage needs intB:bN:h with B in 2..=8, N a multiple of 16, no :sr",
            });
        }
        let maxv = (1u32 << (bits - 1)) - 1;
        let w = 2 * maxv as usize + 1;
        let mut dec = vec![0f32; NMANT * w];
        let mut thr = vec![0f32; NMANT * maxv as usize];
        for mant in 0..NMANT {
            let step = (1024 + mant) as f64;
            // decode exactly as `PackedStore::unpack_run`
            let s = 1.0 / step;
            let inv = 1.0 / s;
            for qi in 0..w {
                let q = qi as i64 - maxv as i64;
                dec[mant * w + qi] = (q as f64 * inv) as f32;
            }
            // encode: q(x) is monotone in x >= 0; binary search on the bits
            for n in 0..maxv as usize {
                let target = (n + 1) as f64;
                let (mut lo, mut hi) = (0u32, 0x7f80_0000u32); // q(lo) < target <= q(hi)
                debug_assert!(cpu_q(f32::from_bits(hi), step, lp.format) >= target);
                while hi - lo > 1 {
                    let mid = lo + (hi - lo) / 2;
                    if cpu_q(f32::from_bits(mid), step, lp.format) >= target {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                thr[mant * maxv as usize + n] = f32::from_bits(hi);
            }
        }
        Ok(HalfCodec {
            bits,
            block,
            maxv,
            dec,
            thr,
        })
    }

    /// Bytes of packed data per scale block.
    pub fn block_bytes(&self) -> usize {
        self.block * self.bits as usize / 4
    }
}

/// `2^k` as an f32 (`-126 <= k <= 127`).
#[inline]
pub fn pow2f(k: i32) -> f32 {
    debug_assert!((-126..=127).contains(&k));
    f32::from_bits(((k + 127) as u32) << 23)
}

/// The block step of a block with largest component `m` (finite, `> 0`)
/// as `(e, mant)`: `step = (1024 + mant) · 2^(e - 10)`, equal to the CPU's
/// `int_step(m)` for a `:h` format with largest integer `maxv`. Integer
/// arithmetic only: `m / maxv` is rounded up to 11 significant bits; the
/// CPU's f64 quotient never crosses a multiple of the 11-bit grid (its
/// distance from one is at least `2^-24` relative, far above the f64
/// rounding error), so the two agree.
#[inline]
pub fn step_parts(m: f32, maxv: u32) -> (i32, u32) {
    let b = m.to_bits();
    let (ex, fr) = ((b >> 23) & 0xff, b & 0x7f_ffff);
    let (mut mm, mut ee) = if ex == 0 {
        (fr, -149)
    } else {
        (fr | 0x80_0000, ex as i32 - 150)
    };
    let sh = mm.leading_zeros();
    mm <<= sh;
    ee -= sh as i32;
    let (q, r) = (mm / maxv, mm % maxv);
    let bl = 32 - q.leading_zeros();
    let s = bl - 11;
    let mut mant = q >> s;
    if q & ((1 << s) - 1) != 0 || r != 0 {
        mant += 1;
    }
    let mut e = bl as i32 - 1 + ee;
    if mant == 2048 {
        mant = 1024;
        e += 1;
    }
    (e, mant - 1024)
}

/// Statistics of one pass (as `PackedStore` keeps them).
#[derive(Clone, Copy, Debug, Default)]
pub struct PassStats {
    /// Largest block exponent written (`i32::MIN`: none).
    pub maxexp: i32,
    /// Blocks below the `:h` window (stored as zero).
    pub underflow: u64,
    /// Blocks above it.
    pub overflow: u64,
    /// Values whose exact power-of-two scaling would leave the normal f32
    /// range (the GPU result could then differ from the CPU's; never on
    /// the doped-Clifford circuit).
    pub inexact: u64,
}

/// The packed register on the host: same layout as `PackedStore` (data
/// block `k` at bytes `k·bpb..`, one 16-bit scale code per block).
pub struct HostStore {
    /// Register qubits.
    pub width: usize,
    /// Packed data.
    pub data: Vec<u8>,
    /// Scale codes.
    pub codes: Vec<u16>,
    /// Still `|0..0>`.
    pub fresh: bool,
    /// `:h` exponent base of the stored blocks.
    pub base: i32,
    /// Largest stored block exponent.
    pub maxexp: i32,
}

impl HostStore {
    /// A `width`-qubit register in `|0..0>`; fails cleanly when the memory
    /// is not available.
    pub fn new(width: usize, codec: &HalfCodec) -> Result<Self, SimError> {
        let n = 1usize << width;
        if n % codec.block != 0 {
            return Err(SimError::NotSupported {
                what: "GPU packed storage: the scale block does not tile the register",
            });
        }
        let nb = n / codec.block;
        let total = (nb * codec.block_bytes() + 2 * nb) as u128;
        let alloc = |_| SimError::TooLarge {
            what: "packed chain-sweep store (allocation failed)",
            bytes: total,
            limit: 0,
        };
        let mut data = Vec::new();
        data.try_reserve_exact(nb * codec.block_bytes()).map_err(alloc)?;
        data.resize(nb * codec.block_bytes(), 0u8);
        let mut codes = Vec::new();
        codes.try_reserve_exact(nb).map_err(alloc)?;
        codes.resize(nb, 0u16);
        Ok(HostStore {
            width,
            data,
            codes,
            fresh: true,
            base: 0,
            maxexp: (((1.0 / codec.maxv as f64).to_bits() >> 52) & 0x7ff) as i32 - 1023,
        })
    }

    /// Bytes held.
    pub fn bytes(&self) -> usize {
        self.data.len() + 2 * self.codes.len()
    }

    /// Decodes `out.len()` amplitudes from register index `start` (whole
    /// blocks), as the GPU's unpack kernel does.
    pub fn decode(&self, codec: &HalfCodec, start: usize, out: &mut [Complex32]) {
        if self.fresh {
            out.fill(Complex32::new(0.0, 0.0));
            if start == 0 {
                out[0] = Complex32::new(1.0, 0.0);
            }
            return;
        }
        let bpb = codec.block_bytes();
        for (j, o) in out.chunks_exact_mut(codec.block).enumerate() {
            let k = start / codec.block + j;
            let mut flags = 0;
            decode_block(
                codec,
                self.codes[k],
                self.base,
                &self.data[k * bpb..(k + 1) * bpb],
                o,
                &mut flags,
            );
        }
    }

    /// The amplitude at register index `i`.
    pub fn amplitude(&self, codec: &HalfCodec, i: usize) -> Complex32 {
        let start = i / codec.block * codec.block;
        let mut v = vec![Complex32::new(0.0, 0.0); codec.block];
        self.decode(codec, start, &mut v);
        v[i - start]
    }
}

/// Decodes one block (the unpack kernel's per-block code).
pub fn decode_block(
    codec: &HalfCodec,
    code: u16,
    base: i32,
    bytes: &[u8],
    out: &mut [Complex32],
    inexact: &mut u64,
) {
    if code == 0 {
        out.fill(Complex32::new(0.0, 0.0));
        return;
    }
    let e = (code >> 10) as i32 - 1 + base;
    let mant = (code & 0x3ff) as usize;
    let k = e - 10;
    let w = 2 * codec.maxv as usize + 1;
    let row = &codec.dec[mant * w..(mant + 1) * w];
    let sc = if (-126..=127).contains(&k) {
        pow2f(k)
    } else {
        *inexact += 1;
        0.0
    };
    let (b, mask) = (codec.bits, (1u64 << codec.bits) - 1);
    let (mut acc, mut na, mut bi) = (0u64, 0u32, 0usize);
    let mut next = || {
        while na < b {
            acc |= (bytes[bi] as u64) << na;
            bi += 1;
            na += 8;
        }
        let u = (acc & mask) as usize;
        acc >>= b;
        na -= b;
        let d = row[u];
        let v = d * sc;
        if d != 0.0 && v.abs() < f32::MIN_POSITIVE {
            *inexact += 1;
        }
        v
    };
    for z in out.iter_mut() {
        z.re = next();
        z.im = next();
    }
}

/// Quantizes and packs one block (the pack kernel's per-block code);
/// returns the scale code.
pub fn encode_block(
    codec: &HalfCodec,
    v: &[Complex32],
    base_out: i32,
    bytes: &mut [u8],
    st: &mut PassStats,
) -> u16 {
    let m = v
        .iter()
        .fold(0.0f32, |m, z| m.max(z.re.abs()).max(z.im.abs()));
    let mut code = 0u16;
    let mut zero = true;
    let (mut mant, mut k) = (0usize, 0i32);
    if m != 0.0 && m.is_finite() {
        let (e, mt) = step_parts(m, codec.maxv);
        st.maxexp = st.maxexp.max(e);
        let ec = e - base_out + 1;
        if ec < 1 {
            st.underflow += 1;
        } else if ec > 63 {
            st.overflow += 1;
        } else {
            code = ((ec as u16) << 10) | mt as u16;
            zero = false;
            mant = mt as usize;
            k = -(e - 10);
        }
    }
    let maxv = codec.maxv as usize;
    let thr = &codec.thr[mant * maxv..(mant + 1) * maxv];
    // |x| · 2^k, exact (k may exceed the f32 exponent range: two steps)
    let (k1, k2) = if k > 126 { (126, k - 126) } else { (k, 0) };
    if !(-126..=126).contains(&k1) || !(-126..=126).contains(&k2) {
        st.inexact += 1;
    }
    let (s1, s2) = (pow2f(k1.clamp(-126, 126)), pow2f(k2.clamp(-126, 126)));
    let (b, off) = (codec.bits, codec.maxv as i64);
    let (mut acc, mut na, mut bi) = (0u64, 0u32, 0usize);
    let mut put = |x: f32| {
        let q = if zero {
            0
        } else {
            let a = x.abs() * s1 * s2;
            let n = thr.iter().take_while(|&&t| a >= t).count() as i64;
            if x < 0.0 {
                -n
            } else {
                n
            }
        };
        acc |= ((q + off) as u64) << na;
        na += b;
        while na >= 8 {
            bytes[bi] = acc as u8;
            acc >>= 8;
            na -= 8;
            bi += 1;
        }
    };
    for z in v {
        put(z.re);
        put(z.im);
    }
    code
}

/// `:h` exponent base of the next pass, from the largest exponent stored.
pub fn next_base(maxexp: i32) -> i32 {
    maxexp + H_UP - 62
}

/// Compiles the stages for the GPU (see [`gpu_stage_plan`]).
pub fn gpu_plans(
    stages: &[Stage],
    width: usize,
    cfg: &BlockConfig,
) -> Result<Vec<GpuStagePlan>, SimError> {
    if !cfg_exportable(cfg) {
        return Err(SimError::NotSupported {
            what: "GPU packed sweep: the config must use the FMA kernel tier (QSIM_NO_AVX512 or avx512 off), no dense fusion, no L1 tiling",
        });
    }
    stages
        .iter()
        .map(|st| {
            gpu_stage_plan(st, width, cfg).map_err(|_| SimError::NotSupported {
                what: "GPU packed sweep: stage plan not exportable",
            })
        })
        .collect()
}

/// Runs the plan of one stage on one gathered block `buf` (`2^l`
/// amplitudes, gathered block `c`), sub-stage by sub-stage, as the GPU.
pub fn run_block_ref(p: &GpuStagePlan, buf: &mut [Complex32], c: usize) {
    let hi = (c as u64) << p.l;
    let full = (1usize << p.l) - 1;
    let mut sub = Vec::new();
    for s in &p.subs {
        let outer = full & !s.inner_mask;
        sub.resize(1usize << s.l, Complex32::new(0.0, 0.0));
        for cc in 0..1usize << (p.l - s.l) {
            let base = deposit(cc, outer);
            for (j, z) in sub.iter_mut().enumerate() {
                *z = buf[base | deposit(j, s.inner_mask)];
            }
            run_sub_stage_ref(s, &mut sub, hi | base as u64);
            for (j, z) in sub.iter().enumerate() {
                buf[base | deposit(j, s.inner_mask)] = *z;
            }
        }
    }
}

/// Register index of run `r` of gathered block `c`.
#[inline]
pub fn run_start(p: &GpuStagePlan, c: usize, r: usize) -> usize {
    deposit(c, p.outer_mask) | deposit(r << p.bc, p.inner_mask)
}

/// One pass of the GPU pipeline on the CPU (reference for the GPU).
pub fn emulate_stage(store: &mut HostStore, codec: &HalfCodec, p: &GpuStagePlan) -> PassStats {
    let base_out = next_base(store.maxexp);
    let bpb = codec.block_bytes();
    let runlen = 1usize << p.bc;
    let nblocks = 1usize << (store.width - p.l);
    let nruns = 1usize << (p.l - p.bc);
    let mut total = PassStats {
        maxexp: i32::MIN,
        ..Default::default()
    };
    for c in 0..nblocks {
        let mut buf = vec![Complex32::new(0.0, 0.0); 1 << p.l];
        for r in 0..nruns {
            let reg = run_start(p, c, r);
            store.decode(codec, reg, &mut buf[r * runlen..(r + 1) * runlen]);
        }
        run_block_ref(p, &mut buf, c);
        for r in 0..nruns {
            let reg = run_start(p, c, r);
            for (j, v) in buf[r * runlen..(r + 1) * runlen]
                .chunks_exact(codec.block)
                .enumerate()
            {
                let k = reg / codec.block + j;
                let code = encode_block(
                    codec,
                    v,
                    base_out,
                    &mut store.data[k * bpb..(k + 1) * bpb],
                    &mut total,
                );
                store.codes[k] = code;
            }
        }
    }
    store.fresh = false;
    store.base = base_out;
    if total.maxexp != i32::MIN {
        store.maxexp = total.maxexp;
    }
    total
}

/// Result of a GPU (or emulated GPU) packed sweep.
#[derive(Clone, Debug)]
pub struct GpuRun {
    /// The amplitude (times the plan's scale).
    pub amp: Complex64,
    /// Memory passes.
    pub passes: usize,
    /// Bytes of the packed store.
    pub store_bytes: usize,
    /// Blocks below / above the `:h` window.
    pub underflow: u64,
    /// See `underflow`.
    pub overflow: u64,
    /// Values outside the exact-scaling range (should be 0).
    pub inexact: u64,
    /// Wall-clock seconds of the passes.
    pub secs: f64,
    /// Per-pass seconds.
    pub pass_secs: Vec<f64>,
}

/// The GPU pipeline run on the CPU: same plans, tables and order as the
/// GPU backend. Bit-identical to `run_packed` (tests) and to the GPU.
pub fn emulate_packed(
    plan: &SweepPlan,
    stages: &[Stage],
    lp: &LowPrec,
    cfg: &BlockConfig,
) -> Result<GpuRun, SimError> {
    let codec = HalfCodec::new(lp)?;
    let plans = gpu_plans(stages, plan.width, cfg)?;
    let mut store = HostStore::new(plan.width, &codec)?;
    let t = std::time::Instant::now();
    let (mut uf, mut of, mut inx) = (0, 0, 0);
    let mut pass_secs = Vec::new();
    for p in &plans {
        let t0 = std::time::Instant::now();
        let s = emulate_stage(&mut store, &codec, p);
        pass_secs.push(t0.elapsed().as_secs_f64());
        uf += s.underflow;
        of += s.overflow;
        inx += s.inexact;
    }
    let a = store.amplitude(&codec, 0);
    Ok(GpuRun {
        amp: Complex64::new(a.re as f64, a.im as f64) * plan.scale,
        passes: plans.len(),
        store_bytes: store.bytes(),
        underflow: uf,
        overflow: of,
        inexact: inx,
        secs: t.elapsed().as_secs_f64(),
        pass_secs,
    })
}

/// Shape statistics of the GPU plans (for planning / the `--count` mode).
#[derive(Clone, Debug, Default)]
pub struct PlanStats {
    /// Memory passes.
    pub passes: usize,
    /// Sub-stages (VRAM sweeps of the f32 work buffer) over all passes.
    pub subs: usize,
    /// Largest sub-stage buffer bits.
    pub max_sub_l: usize,
    /// Ops over all sub-stages.
    pub ops: usize,
    /// Most outer-conditioned terms in one diagonal group.
    pub max_conds: usize,
    /// Diagonal-table bytes (all variants) of the largest pass.
    pub max_table_bytes: usize,
    /// Smallest contiguous run bits.
    pub min_bc: usize,
}

/// Statistics of `plans`.
pub fn plan_stats(plans: &[GpuStagePlan]) -> PlanStats {
    use crate::engines::blocked::gpu_export::GpuOp;
    let mut s = PlanStats {
        passes: plans.len(),
        min_bc: usize::MAX,
        ..Default::default()
    };
    for p in plans {
        s.min_bc = s.min_bc.min(p.bc);
        let mut tb = 0;
        for sub in &p.subs {
            s.subs += 1;
            s.max_sub_l = s.max_sub_l.max(sub.l);
            s.ops += sub.ops.len();
            for op in &sub.ops {
                if let GpuOp::Diag(gs) = op {
                    for g in gs {
                        s.max_conds = s.max_conds.max(g.conds.len());
                        tb += g
                            .variants
                            .iter()
                            .flatten()
                            .map(|t| 4 * (t.lor.len() + t.loi.len() + t.hr.len() + t.hi.len()))
                            .sum::<usize>();
                    }
                }
            }
        }
        s.max_table_bytes = s.max_table_bytes.max(tb);
    }
    s
}

/// Sub-stages of `p` (re-exported for the GPU driver).
pub fn subs(p: &GpuStagePlan) -> &[GpuSubStage] {
    &p.subs
}

/// Parallel host gather of the packed runs of gathered blocks
/// `c0..c0 + g` into `data` / `codes` (block order, run order).
pub fn gather_chunk(
    store: &HostStore,
    codec: &HalfCodec,
    p: &GpuStagePlan,
    c0: usize,
    g: usize,
    data: &mut [u8],
    codes: &mut [u16],
) {
    let runlen = 1usize << p.bc;
    let nruns = 1usize << (p.l - p.bc);
    let bpr = runlen / codec.block * codec.block_bytes();
    let cpr = runlen / codec.block;
    data[..g * nruns * bpr]
        .par_chunks_mut(bpr)
        .zip(codes[..g * nruns * cpr].par_chunks_mut(cpr))
        .enumerate()
        .for_each(|(i, (d, cd))| {
            let reg = run_start(p, c0 + i / nruns, i % nruns);
            let k = reg / codec.block;
            d.copy_from_slice(&store.data[k * codec.block_bytes()..][..bpr]);
            cd.copy_from_slice(&store.codes[k..k + cpr]);
        });
}

/// A raw pointer shared by the scatter workers (disjoint writes).
#[derive(Clone, Copy)]
struct SyncPtr<T>(*mut T);
// SAFETY: only used for disjoint per-run writes in `scatter_chunk`.
unsafe impl<T> Sync for SyncPtr<T> {}
unsafe impl<T> Send for SyncPtr<T> {}

/// Inverse of [`gather_chunk`].
pub fn scatter_chunk(
    store: &mut HostStore,
    codec: &HalfCodec,
    p: &GpuStagePlan,
    c0: usize,
    g: usize,
    data: &[u8],
    codes: &[u16],
) {
    let runlen = 1usize << p.bc;
    let nruns = 1usize << (p.l - p.bc);
    let bb = codec.block_bytes();
    let bpr = runlen / codec.block * bb;
    let cpr = runlen / codec.block;
    let (dp, cp) = (
        SyncPtr(store.data.as_mut_ptr()),
        SyncPtr(store.codes.as_mut_ptr()),
    );
    let (dl, cl) = (store.data.len(), store.codes.len());
    data[..g * nruns * bpr]
        .par_chunks(bpr)
        .zip(codes[..g * nruns * cpr].par_chunks(cpr))
        .enumerate()
        .for_each(|(i, (d, cd))| {
            let (dp, cp) = (dp, cp);
            let reg = run_start(p, c0 + i / nruns, i % nruns);
            let k = reg / codec.block;
            assert!(k * bb + bpr <= dl && k + cpr <= cl);
            // SAFETY: runs of different (block, run) pairs are disjoint
            // register ranges, so these writes never overlap; bounds
            // checked above.
            unsafe {
                std::ptr::copy_nonoverlapping(d.as_ptr(), dp.0.add(k * bb), bpr);
                std::ptr::copy_nonoverlapping(cd.as_ptr(), cp.0.add(k), cpr);
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::chain_lowprec::int_step;

    #[test]
    fn step_parts_match_int_step() {
        for f in ["int4:b16:h", "int5:b16:h", "int8:b32:h", "int2:b16:h", "int3:b64:h"] {
            let lp = LowPrec::parse(f).unwrap();
            let codec = HalfCodec::new(&lp).unwrap();
            let mut z = 12345u64;
            for i in 0..200_000u32 {
                z = z.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let m = match i % 4 {
                    0 => f32::from_bits((z >> 33) as u32 & 0x7f7f_ffff),
                    1 => f32::from_bits(((z >> 40) as u32 & 0x7f_ffff) | (100 << 23)),
                    2 => (i as f32) * 0.125,
                    _ => f32::from_bits((z >> 41) as u32 & 0x7f_ffff), // subnormal
                };
                if m == 0.0 || !m.is_finite() {
                    continue;
                }
                let step = int_step(m as f64, &lp);
                let (e, mant) = step_parts(m, codec.maxv);
                let got = (1024 + mant) as f64 * 2f64.powi(e - 10);
                assert_eq!(got, step, "{f} m={m:e}");
                assert_eq!(((step.to_bits() >> 52) & 0x7ff) as i32 - 1023, e);
            }
        }
    }

    #[test]
    fn codec_matches_cpu_rounding() {
        let lp = LowPrec::parse("int4:b16:h").unwrap();
        let codec = HalfCodec::new(&lp).unwrap();
        let mut z = 99u64;
        for _ in 0..200_000 {
            z = z.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let mant = (z >> 50) as usize % NMANT;
            let e = ((z >> 20) % 40) as i32 - 30;
            let step = (1024 + mant) as f64 * 2f64.powi(e - 10);
            let x = ((z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) * 7.6 * step;
            let x = x as f32;
            let want = cpu_q(x, step, lp.format);
            let a = x.abs() * pow2f(-(e - 10));
            let n = codec.thr[mant * 7..(mant + 1) * 7]
                .iter()
                .take_while(|&&t| a >= t)
                .count() as f64;
            assert_eq!(if x < 0.0 { -n } else { n }, want, "x={x:e} step={step:e}");
        }
    }
}
