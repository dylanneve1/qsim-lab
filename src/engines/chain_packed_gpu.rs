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
//!   ([`Codec`]): thresholds and decoded values at a normalised step,
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
pub mod gen;
#[cfg(feature = "wgpu")]
pub mod gpu;

/// Number of 11-bit step mantissas (`1024..2048`) of the `:h` formats.
pub const NMANT: usize = 1024;

/// Encode / decode tables of a packed format.
#[derive(Clone, Debug)]
pub struct Codec {
    /// Bits per component.
    pub bits: u32,
    /// Amplitudes per scale block.
    pub block: usize,
    /// Largest stored magnitude, `2^(bits-1) - 1`.
    pub maxv: u32,
    /// `:h` scales (16-bit codes); else f32 steps.
    pub half: bool,
    /// `:h`: `dec[mant * (2 maxv + 1) + q + maxv]`, the CPU's decoded value
    /// of `q` for the step `1024 + mant` (exponent 10).
    pub dec: Vec<f32>,
    /// `:h`: `thr[mant * maxv + n]`, smallest `x >= 0` (f32) that the CPU
    /// rounds to at least `n + 1` for the step `1024 + mant`.
    pub thr: Vec<f32>,
    /// f32 steps: `k` of [`exact32::recip_back`] for each 24-bit step
    /// mantissa, 2 bits each (16 per word).
    pub ktab: Vec<u32>,
}

/// The CPU's integer for `x` at step `step` (`round_to(x * (1/step))`).
fn cpu_q(x: f32, step: f64, format: Format) -> f64 {
    let s = 1.0 / step;
    round_to(x as f64 * s, format, None)
}

impl Codec {
    /// Tables for `lp`; fails unless it is `intB:bN` or `intB:bN:h` with
    /// `B` in 2..=8, `N` a multiple of 16 (at most 64) and round to nearest.
    pub fn new(lp: &LowPrec) -> Result<Self, SimError> {
        let (Format::Int(bits), Scaling::Block(block)) = (lp.format, lp.scaling) else {
            return Err(SimError::NotSupported {
                what: "GPU packed storage needs an intB:bN[:h] format",
            });
        };
        if !(2..=8).contains(&bits) || block % 16 != 0 || block > 64 || lp.stochastic {
            return Err(SimError::NotSupported {
                what: "GPU packed storage needs intB:bN[:h] with B in 2..=8, N in {16, 32, 48, 64}, no :sr",
            });
        }
        let maxv = (1u32 << (bits - 1)) - 1;
        let mut c = Codec {
            bits,
            block,
            maxv,
            half: lp.half_scale,
            dec: Vec::new(),
            thr: Vec::new(),
            ktab: Vec::new(),
        };
        if lp.half_scale {
            let w = 2 * maxv as usize + 1;
            c.dec = vec![0f32; NMANT * w];
            c.thr = vec![0f32; NMANT * maxv as usize];
            for mant in 0..NMANT {
                let step = (1024 + mant) as f64;
                // decode exactly as `PackedStore::unpack_run`
                let s = 1.0 / step;
                let inv = 1.0 / s;
                for qi in 0..w {
                    let q = qi as i64 - maxv as i64;
                    c.dec[mant * w + qi] = (q as f64 * inv) as f32;
                }
                // encode: q(x) is monotone in x >= 0; binary search on the bits
                for n in 0..maxv as usize {
                    let target = (n + 1) as f64;
                    let (mut lo, mut hi) = (0u32, 0x7f80_0000u32);
                    while hi - lo > 1 {
                        let mid = lo + (hi - lo) / 2;
                        if cpu_q(f32::from_bits(mid), step, lp.format) >= target {
                            hi = mid;
                        } else {
                            lo = mid;
                        }
                    }
                    c.thr[mant * maxv as usize + n] = f32::from_bits(hi);
                }
            }
        } else {
            c.ktab = exact32::k_table();
        }
        Ok(c)
    }

    /// Bytes of packed data per scale block.
    pub fn block_bytes(&self) -> usize {
        self.block * self.bits as usize / 4
    }

    /// Bytes of one block scale (2 for `:h` codes, 4 for f32 steps).
    pub fn scale_bytes(&self) -> usize {
        if self.half {
            2
        } else {
            4
        }
    }

    fn k_of(&self, sm: u32) -> i32 {
        let i = (sm - (1 << 23)) as usize;
        (((self.ktab[i >> 4] >> ((i & 15) * 2)) & 3) as i32) - 1
    }
}

/// Exact integer emulation (u32 operations only, as in the shader) of the
/// f32-step formats' f64 arithmetic: the step `(m / maxv) as f32`, the
/// encode `round(x · (1/step))` and the decode `(q · (1/(1/step))) as f32`.
pub mod exact32 {
    /// `a · b` as `(hi, lo)`.
    #[inline]
    pub fn mul32(a: u32, b: u32) -> (u32, u32) {
        let (a0, a1, b0, b1) = (a & 0xffff, a >> 16, b & 0xffff, b >> 16);
        let (p00, p01, p10, p11) = (a0 * b0, a0 * b1, a1 * b0, a1 * b1);
        let mid = (p00 >> 16) + (p01 & 0xffff) + (p10 & 0xffff);
        let lo = (p00 & 0xffff) | (mid << 16);
        let hi = p11 + (p01 >> 16) + (p10 >> 16) + (mid >> 16);
        (hi, lo)
    }

    #[inline]
    fn bitlen(x: u32) -> u32 {
        32 - x.leading_zeros()
    }

    /// Bit length of a 3-limb number (limb 0 lowest).
    #[inline]
    pub fn bitlen3(p: [u32; 3]) -> u32 {
        if p[2] != 0 {
            64 + bitlen(p[2])
        } else if p[1] != 0 {
            32 + bitlen(p[1])
        } else {
            bitlen(p[0])
        }
    }

    #[inline]
    fn shr3(p: [u32; 3], d: u32) -> [u32; 3] {
        let (w, s) = ((d / 32) as usize, d % 32);
        let mut out = [0u32; 3];
        for (i, o) in out.iter_mut().enumerate() {
            let j = i + w;
            if j < 3 {
                let mut v = p[j] >> s;
                if s > 0 && j + 1 < 3 {
                    v |= p[j + 1] << (32 - s);
                }
                *o = v;
            }
        }
        out
    }

    #[inline]
    fn bit3(p: [u32; 3], i: u32) -> bool {
        i < 96 && (p[(i / 32) as usize] >> (i % 32)) & 1 == 1
    }

    /// Whether any of the bits `0..n` of `p` is set.
    #[inline]
    fn low3(p: [u32; 3], n: u32) -> bool {
        let mut any = false;
        for (i, &v) in p.iter().enumerate() {
            let lo = 32 * i as u32;
            if n >= lo + 32 {
                any |= v != 0;
            } else if n > lo {
                any |= v & ((1u32 << (n - lo)) - 1) != 0;
            }
        }
        any
    }

    /// `p / 2^d` rounded to nearest, ties to even.
    #[inline]
    pub fn rne_shift3(p: [u32; 3], d: u32) -> [u32; 3] {
        if d == 0 {
            return p;
        }
        let mut q = shr3(p, d);
        if bit3(p, d - 1) && (low3(p, d - 1) || q[0] & 1 == 1) {
            q[0] = q[0].wrapping_add(1);
            if q[0] == 0 {
                q[1] = q[1].wrapping_add(1);
                if q[1] == 0 {
                    q[2] = q[2].wrapping_add(1);
                }
            }
        }
        q
    }

    /// The f32 step `(m / maxv) as f32` (via f64, as the CPU) of a block
    /// with largest component `m > 0` (finite) as `(sm, es)`:
    /// `step = sm · 2^es`, `sm` in `[2^23, 2^24)`. The f64 quotient never
    /// lands on an f32 rounding tie unless it is exact (`maxv` is odd), so
    /// this is the direct rounding of the exact quotient.
    #[inline]
    pub fn step32(m: f32, maxv: u32) -> (u32, i32) {
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
        let s = bitlen(q) - 24;
        let mut sm = q >> s;
        let half = 1u32 << (s - 1);
        let low = q & ((1u32 << s) - 1);
        if low > half || (low == half && (r != 0 || sm & 1 == 1)) {
            sm += 1;
        }
        let mut es = s as i32 + ee;
        if sm == 1 << 24 {
            sm = 1 << 23;
            es += 1;
        }
        (sm, es)
    }

    /// f32 bits of `sm · 2^es` (`None` outside the normal range).
    #[inline]
    pub fn f32_bits(sm: u32, es: i32) -> Option<u32> {
        let e = es + 23 + 127;
        (1..=254)
            .contains(&e)
            .then(|| ((e as u32) << 23) | (sm & 0x7f_ffff))
    }

    /// `fl64(1 / sm)` for `sm` in `[2^23, 2^24)` as `([lo, hi], adj)`:
    /// `R · 2^(adj - 76)`, `R` a 53-bit integer.
    #[inline]
    pub fn recip(sm: u32) -> ([u32; 2], i32) {
        // floor(2^76 / sm) in 8-bit digits (2^76 = 16 · 256^9)
        let mut rem = 16u32;
        let (mut qh, mut ql) = (0u32, 0u32);
        for _ in 0..9 {
            rem <<= 8;
            let d = rem / sm;
            rem -= d * sm;
            qh = (qh << 8) | (ql >> 24);
            ql = (ql << 8) | d;
        }
        let twice = rem << 1;
        if twice > sm || (twice == sm && ql & 1 == 1) {
            ql = ql.wrapping_add(1);
            if ql == 0 {
                qh += 1;
            }
        }
        if qh == 1 << 21 {
            ([0, 1 << 20], 1)
        } else {
            ([ql, qh], 0)
        }
    }

    /// The CPU's stored integer `round(x · fl64(1/step))` clamped to
    /// `±maxv`, for `step = sm · 2^es` with `recip(sm) = (r, adj)`.
    #[inline]
    pub fn enc_q(x: f32, r: [u32; 2], adj: i32, es: i32, maxv: u32) -> i32 {
        let b = x.to_bits();
        let ax = b & 0x7fff_ffff;
        if ax == 0 {
            return 0;
        }
        let ex = ax >> 23;
        let (xm, xe) = if ex == 0 {
            (ax, -149)
        } else {
            ((ax & 0x7f_ffff) | 0x80_0000, ex as i32 - 150)
        };
        let (h0, l0) = mul32(xm, r[0]);
        let (h1, l1) = mul32(xm, r[1]);
        let p1 = h0.wrapping_add(l1);
        let p2 = h1 + (p1 < h0) as u32;
        let p = [l0, p1, p2];
        // x · (1/step) = P · 2^t
        let t = xe - 76 + adj - es;
        let d1 = bitlen3(p).saturating_sub(53);
        let p53 = rne_shift3(p, d1);
        let f = -(t + d1 as i32);
        let n = if f <= 0 {
            maxv
        } else if f >= 64 {
            0
        } else {
            let q = rne_shift3(p53, f as u32);
            if q[1] != 0 || q[2] != 0 {
                maxv
            } else {
                q[0].min(maxv)
            }
        };
        if b >> 31 == 1 {
            -(n as i32)
        } else {
            n as i32
        }
    }

    /// The CPU's decoded value `(q · fl64(1/fl64(1/step))) as f32` for
    /// `step = sm · 2^es`, where `fl64(1/fl64(1/sm)) = (sm·2^29 + k)·2^-29`
    /// ([`k_table`]). Also returns whether the result left the normal
    /// range (then it may differ from the CPU's).
    #[inline]
    pub fn dec_v(q: i32, sm: u32, es: i32, k: i32) -> (f32, bool) {
        if q == 0 {
            return (0.0, false);
        }
        let aq = q.unsigned_abs();
        let a = aq * sm;
        // N = a · 2^29 + aq · k
        let (mut hi, mut lo) = (a >> 3, a << 29);
        let t = aq * k.unsigned_abs();
        if k >= 0 {
            let l2 = lo.wrapping_add(t);
            hi += (l2 < lo) as u32;
            lo = l2;
        } else {
            hi -= (lo < t) as u32;
            lo = lo.wrapping_sub(t);
        }
        let n = [lo, hi, 0];
        let d1 = bitlen3(n).saturating_sub(53);
        let n53 = rne_shift3(n, d1);
        let d2 = bitlen3(n53).saturating_sub(24);
        let mut m = rne_shift3(n53, d2)[0];
        let mut dd = (d1 + d2) as i32;
        if m == 1 << 24 {
            m = 1 << 23;
            dd += 1;
        }
        let bl = bitlen(m);
        let e = bl as i32 - 1 + dd + es - 29;
        let field = (m << (24 - bl)) & 0x7f_ffff;
        let bad = !(-126..=127).contains(&e);
        let bits = ((q < 0) as u32) << 31 | (((e + 127).clamp(1, 254) as u32) << 23) | field;
        (f32::from_bits(bits), bad)
    }

    /// `k` with `fl64(1/fl64(1/sm)) = (sm·2^29 + k)·2^-29` for every
    /// `sm` in `[2^23, 2^24)`, stored as `k + 1` in 2 bits (16 per word).
    pub fn k_table() -> Vec<u32> {
        let n = 1usize << 23;
        let mut t = vec![0u32; n / 16];
        for (w, word) in t.iter_mut().enumerate() {
            let mut v = 0u32;
            for j in 0..16 {
                let sm = (1u64 << 23) + (w * 16 + j) as u64;
                let s = sm as f64;
                let inv = 1.0 / (1.0 / s);
                let k = (inv * (1u64 << 29) as f64) as i64 - ((sm as i64) << 29);
                assert!((-1..=1).contains(&k), "k = {k} for sm = {sm}");
                v |= ((k + 1) as u32) << (2 * j);
            }
            *word = v;
        }
        t
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
    /// Values whose exact emulation would leave the normal f32 range (the
    /// GPU result could then differ from the CPU's; never on the
    /// doped-Clifford circuit).
    pub inexact: u64,
}

/// The packed register on the host: same layout as `PackedStore` (data
/// block `k` at bytes `k·bpb..`, one scale per block: a 16-bit code for
/// `:h`, else the f32 step, little-endian).
pub struct HostStore {
    /// Register qubits.
    pub width: usize,
    /// Packed data.
    pub data: Vec<u8>,
    /// Block scales (`scale_bytes` each).
    pub scales: Vec<u8>,
    /// Bytes per scale.
    pub sb: usize,
    /// Still `|0..0>`.
    pub fresh: bool,
    /// `:h` exponent base of the stored blocks.
    pub base: i32,
    /// Largest stored block exponent (`:h`).
    pub maxexp: i32,
}

impl HostStore {
    /// A `width`-qubit register in `|0..0>`; fails cleanly when the memory
    /// is not available.
    pub fn new(width: usize, codec: &Codec) -> Result<Self, SimError> {
        let n = 1usize << width;
        if n % codec.block != 0 {
            return Err(SimError::NotSupported {
                what: "GPU packed storage: the scale block does not tile the register",
            });
        }
        let nb = n / codec.block;
        let sb = codec.scale_bytes();
        let total = (nb * (codec.block_bytes() + sb)) as u128;
        let alloc = |_| SimError::TooLarge {
            what: "packed chain-sweep store (allocation failed)",
            bytes: total,
            limit: 0,
        };
        let mut data = Vec::new();
        data.try_reserve_exact(nb * codec.block_bytes())
            .map_err(alloc)?;
        data.resize(nb * codec.block_bytes(), 0u8);
        let mut scales = Vec::new();
        scales.try_reserve_exact(nb * sb).map_err(alloc)?;
        scales.resize(nb * sb, 0u8);
        Ok(HostStore {
            width,
            data,
            scales,
            sb,
            fresh: true,
            base: 0,
            maxexp: (((1.0 / codec.maxv as f64).to_bits() >> 52) & 0x7ff) as i32 - 1023,
        })
    }

    /// Bytes held.
    pub fn bytes(&self) -> usize {
        self.data.len() + self.scales.len()
    }

    /// Scale of block `k`.
    pub fn scale(&self, k: usize) -> u32 {
        let s = &self.scales[k * self.sb..(k + 1) * self.sb];
        if self.sb == 2 {
            u16::from_le_bytes([s[0], s[1]]) as u32
        } else {
            u32::from_le_bytes([s[0], s[1], s[2], s[3]])
        }
    }

    fn set_scale(&mut self, k: usize, v: u32) {
        let sb = self.sb;
        self.scales[k * sb..(k + 1) * sb].copy_from_slice(&v.to_le_bytes()[..sb]);
    }

    /// Decodes `out.len()` amplitudes from register index `start` (whole
    /// blocks), as the GPU's unpack kernel does.
    pub fn decode(&self, codec: &Codec, start: usize, out: &mut [Complex32]) {
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
                self.scale(k),
                self.base,
                &self.data[k * bpb..(k + 1) * bpb],
                o,
                &mut flags,
            );
        }
    }

    /// The amplitude at register index `i`.
    pub fn amplitude(&self, codec: &Codec, i: usize) -> Complex32 {
        let start = i / codec.block * codec.block;
        let mut v = vec![Complex32::new(0.0, 0.0); codec.block];
        self.decode(codec, start, &mut v);
        v[i - start]
    }
}

/// Unpacks the `b`-bit offset-binary values of a block, LSB first.
fn unpack_ints(bytes: &[u8], b: u32, n: usize) -> impl Iterator<Item = u32> + '_ {
    let mask = (1u64 << b) - 1;
    let (mut acc, mut na, mut bi) = (0u64, 0u32, 0usize);
    (0..n).map(move |_| {
        while na < b {
            acc |= (bytes[bi] as u64) << na;
            bi += 1;
            na += 8;
        }
        let u = (acc & mask) as u32;
        acc >>= b;
        na -= b;
        u
    })
}

/// Decodes one block (the unpack kernel's per-block code).
pub fn decode_block(
    codec: &Codec,
    sc: u32,
    base: i32,
    bytes: &[u8],
    out: &mut [Complex32],
    inexact: &mut u64,
) {
    if sc == 0 {
        out.fill(Complex32::new(0.0, 0.0));
        return;
    }
    let maxv = codec.maxv as i32;
    let vals: Vec<f32> = if codec.half {
        let e = (sc >> 10) as i32 - 1 + base;
        let mant = (sc & 0x3ff) as usize;
        let k = e - 10;
        let w = 2 * codec.maxv as usize + 1;
        let row = &codec.dec[mant * w..(mant + 1) * w];
        let s = if (-126..=127).contains(&k) {
            pow2f(k)
        } else {
            *inexact += 1;
            0.0
        };
        unpack_ints(bytes, codec.bits, 2 * out.len())
            .map(|u| {
                let d = row[u as usize];
                let v = d * s;
                if d != 0.0 && v.abs() < f32::MIN_POSITIVE {
                    *inexact += 1;
                }
                v
            })
            .collect()
    } else {
        let ex = (sc >> 23) & 0xff;
        if ex == 0 {
            *inexact += 1;
        }
        let sm = (sc & 0x7f_ffff) | 0x80_0000;
        let es = ex as i32 - 127 - 23;
        let k = codec.k_of(sm);
        unpack_ints(bytes, codec.bits, 2 * out.len())
            .map(|u| {
                let (v, bad) = exact32::dec_v(u as i32 - maxv, sm, es, k);
                *inexact += bad as u64;
                v
            })
            .collect()
    };
    for (z, c) in out.iter_mut().zip(vals.chunks_exact(2)) {
        *z = Complex32::new(c[0], c[1]);
    }
}

/// Quantizes and packs one block (the pack kernel's per-block code);
/// returns the scale.
pub fn encode_block(
    codec: &Codec,
    v: &[Complex32],
    base_out: i32,
    bytes: &mut [u8],
    st: &mut PassStats,
) -> u32 {
    let m = v
        .iter()
        .fold(0.0f32, |m, z| m.max(z.re.abs()).max(z.im.abs()));
    let maxv = codec.maxv;
    let mut sc = 0u32;
    let qs: Vec<i32> = if codec.half {
        let mut zero = true;
        let (mut mant, mut k) = (0usize, 0i32);
        if m != 0.0 && m.is_finite() {
            let (e, mt) = step_parts(m, maxv);
            st.maxexp = st.maxexp.max(e);
            let ec = e - base_out + 1;
            if ec < 1 {
                st.underflow += 1;
            } else if ec > 63 {
                st.overflow += 1;
            } else {
                sc = ((ec as u32) << 10) | mt;
                zero = false;
                mant = mt as usize;
                k = -(e - 10);
            }
        }
        let thr = &codec.thr[mant * maxv as usize..(mant + 1) * maxv as usize];
        // |x| · 2^k, exact (k may exceed the f32 exponent range: two steps)
        let (k1, k2) = if k > 126 { (126, k - 126) } else { (k, 0) };
        if !(-126..=126).contains(&k1) || !(-126..=126).contains(&k2) {
            st.inexact += 1;
        }
        let (s1, s2) = (pow2f(k1.clamp(-126, 126)), pow2f(k2.clamp(-126, 126)));
        v.iter()
            .flat_map(|z| [z.re, z.im])
            .map(|x| {
                if zero {
                    return 0;
                }
                let a = x.abs() * s1 * s2;
                let n = thr.iter().take_while(|&&t| a >= t).count() as i32;
                if x < 0.0 {
                    -n
                } else {
                    n
                }
            })
            .collect()
    } else if m != 0.0 && m.is_finite() {
        let (sm, es) = exact32::step32(m, maxv);
        match exact32::f32_bits(sm, es) {
            Some(b) => sc = b,
            None => {
                st.inexact += 1;
                sc = 0;
            }
        }
        let (r, adj) = exact32::recip(sm);
        v.iter()
            .flat_map(|z| [z.re, z.im])
            .map(|x| {
                if sc == 0 {
                    0
                } else {
                    exact32::enc_q(x, r, adj, es, maxv)
                }
            })
            .collect()
    } else {
        vec![0; 2 * v.len()]
    };
    let (b, off) = (codec.bits, maxv as i64);
    let (mut acc, mut na, mut bi) = (0u64, 0u32, 0usize);
    for q in qs {
        acc |= ((q as i64 + off) as u64) << na;
        na += b;
        while na >= 8 {
            bytes[bi] = acc as u8;
            acc >>= 8;
            na -= 8;
            bi += 1;
        }
    }
    sc
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
pub fn emulate_stage(store: &mut HostStore, codec: &Codec, p: &GpuStagePlan) -> PassStats {
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
                store.set_scale(k, code);
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
    let codec = Codec::new(lp)?;
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

/// Register-residency analysis of `plans` for `rbits` register bits per
/// thread chosen per sub-stage (the most targeted buffer bits): returns
/// (ops, register-local ops, shared-memory phases).
pub fn reg_stats(plans: &[GpuStagePlan], rbits: usize) -> (usize, usize, usize) {
    use crate::engines::blocked::gpu_export::GpuOp;
    let (mut ops, mut local, mut phases) = (0, 0, 0);
    for p in plans {
        for sub in &p.subs {
            let mut hist = vec![0usize; sub.l];
            let tmask = |op: &GpuOp| -> u32 {
                match op {
                    GpuOp::U1 { t, .. } => 1 << t,
                    GpuOp::Swap { a, b } => (1 << a) | (1 << b),
                    GpuOp::Pair { t1, t2, .. } => (1 << t1) | (1 << t2),
                    GpuOp::Diag(_) => 0,
                }
            };
            for op in &sub.ops {
                let m = tmask(op);
                for (b, h) in hist.iter_mut().enumerate() {
                    *h += (m >> b & 1) as usize;
                }
            }
            let mut order: Vec<usize> = (0..sub.l).collect();
            order.sort_by_key(|&b| std::cmp::Reverse(hist[b]));
            let rmask: u32 = order.iter().take(rbits).fold(0, |m, &b| m | 1 << b);
            let mut in_shared = false;
            for op in &sub.ops {
                ops += 1;
                let m = tmask(op);
                if m & !rmask == 0 {
                    local += 1;
                    in_shared = false;
                } else if !in_shared {
                    phases += 1;
                    in_shared = true;
                }
            }
        }
    }
    (ops, local, phases)
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

/// A raw pointer shared by the gather / scatter workers (disjoint writes).
#[derive(Clone, Copy)]
struct SyncPtr<T>(*mut T);
// SAFETY: only used for disjoint per-run writes in `gather_chunk` /
// `scatter_chunk`.
unsafe impl<T> Sync for SyncPtr<T> {}
unsafe impl<T> Send for SyncPtr<T> {}

/// Bytes of packed data and of scales per run of `p` (staging layout:
/// all runs' data, then all runs' scales).
pub fn run_bytes(codec: &Codec, p: &GpuStagePlan) -> (usize, usize) {
    let bpr = (1usize << p.bc) / codec.block;
    (bpr * codec.block_bytes(), bpr * codec.scale_bytes())
}

/// Parallel host gather of the packed runs of gathered blocks
/// `c0..c0 + g` (block order, run order): data to `data`, scales to `sc`.
///
/// # Safety
/// `data` / `sc` must be valid for writes of `g · 2^(l-bc)` times
/// [`run_bytes`] bytes each, and not accessed otherwise meanwhile (they may
/// point to write-only mapped memory: only written, never read).
pub unsafe fn gather_chunk(
    store: &HostStore,
    codec: &Codec,
    p: &GpuStagePlan,
    c0: usize,
    g: usize,
    data: *mut u8,
    sc: *mut u8,
) {
    let nruns = 1usize << (p.l - p.bc);
    let (bpr, spr) = run_bytes(codec, p);
    let (bb, sb) = (codec.block_bytes(), codec.scale_bytes());
    let (dp, sp) = (SyncPtr(data), SyncPtr(sc));
    (0..g * nruns).into_par_iter().for_each(|i| {
        let (dp, sp) = (dp, sp);
        let k = run_start(p, c0 + i / nruns, i % nruns) / codec.block;
        let (src_d, src_s) = (&store.data[k * bb..][..bpr], &store.scales[k * sb..][..spr]);
        // SAFETY: run `i` writes bytes `i·bpr..` / `i·spr..` only (disjoint
        // across `i`), inside the ranges the caller guarantees.
        unsafe {
            std::ptr::copy_nonoverlapping(src_d.as_ptr(), dp.0.add(i * bpr), bpr);
            std::ptr::copy_nonoverlapping(src_s.as_ptr(), sp.0.add(i * spr), spr);
        }
    });
}

/// Inverse of [`gather_chunk`]: writes the runs back into the store.
pub fn scatter_chunk(
    store: &mut HostStore,
    codec: &Codec,
    p: &GpuStagePlan,
    c0: usize,
    g: usize,
    data: &[u8],
    sc: &[u8],
) {
    let nruns = 1usize << (p.l - p.bc);
    let (bpr, spr) = run_bytes(codec, p);
    let (bb, sb) = (codec.block_bytes(), codec.scale_bytes());
    assert!(data.len() >= g * nruns * bpr && sc.len() >= g * nruns * spr);
    let (dp, sp) = (
        SyncPtr(store.data.as_mut_ptr()),
        SyncPtr(store.scales.as_mut_ptr()),
    );
    let (dl, sl) = (store.data.len(), store.scales.len());
    (0..g * nruns).into_par_iter().for_each(|i| {
        let (dp, sp) = (dp, sp);
        let k = run_start(p, c0 + i / nruns, i % nruns) / codec.block;
        assert!(k * bb + bpr <= dl && k * sb + spr <= sl);
        // SAFETY: runs of different (block, run) pairs are disjoint register
        // ranges, so these writes never overlap; bounds checked above.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr().add(i * bpr), dp.0.add(k * bb), bpr);
            std::ptr::copy_nonoverlapping(sc.as_ptr().add(i * spr), sp.0.add(k * sb), spr);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::chain_lowprec::int_step;

    #[test]
    fn step_parts_match_int_step() {
        for f in [
            "int4:b16:h",
            "int5:b16:h",
            "int8:b32:h",
            "int2:b16:h",
            "int3:b64:h",
        ] {
            let lp = LowPrec::parse(f).unwrap();
            let codec = Codec::new(&lp).unwrap();
            let mut z = 12345u64;
            for i in 0..200_000u32 {
                z = z
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
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
        let codec = Codec::new(&lp).unwrap();
        let mut z = 99u64;
        for _ in 0..200_000 {
            z = z
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
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

    #[test]
    fn exact32_matches_cpu_f64() {
        use super::exact32::*;
        for f in ["int6:b64", "int4:b16", "int5:b32", "int8:b64", "int2:b16"] {
            let lp = LowPrec::parse(f).unwrap();
            let codec = Codec::new(&lp).unwrap();
            let maxv = codec.maxv;
            let mut z = 777u64;
            let mut rnd = || {
                z = z
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                z
            };
            for i in 0..300_000u32 {
                let r0 = rnd();
                let m = match i % 3 {
                    0 => f32::from_bits(
                        ((r0 >> 40) as u32 & 0x7f_ffff) | (((r0 >> 20) % 60 + 80) as u32) << 23,
                    ),
                    1 => ((r0 >> 11) as f64 / (1u64 << 53) as f64) as f32 * 1e-3 + 1e-30,
                    _ => (maxv as f32) * f32::from_bits((((r0 >> 20) % 60 + 80) as u32) << 23), // exact quotients
                };
                let step = int_step(m as f64, &lp);
                let (sm, es) = step32(m, maxv);
                assert_eq!(sm as f64 * 2f64.powi(es), step, "{f} step m={m:e}");
                assert_eq!(f32_bits(sm, es), Some((step as f32).to_bits()));
                let (rr, adj) = recip(sm);
                let rv = ((rr[1] as u64) << 32 | rr[0] as u64) as f64 * 2f64.powi(adj - 76);
                assert_eq!(rv, 1.0 / sm as f64, "{f} recip sm={sm}");
                // encode: random x in [-m, m], plus exact ties
                for j in 0..4 {
                    let r1 = rnd();
                    let x = if j < 3 {
                        (((r1 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32 * m
                    } else {
                        let n = (r1 >> 40) % maxv as u64;
                        ((n as f64 + 0.5) * step * if r1 & 1 == 1 { -1.0 } else { 1.0 }) as f32
                    };
                    let want = cpu_q(x, step, lp.format);
                    assert_eq!(
                        enc_q(x, rr, adj, es, maxv) as f64,
                        want,
                        "{f} enc x={x:e} step={step:e}"
                    );
                }
                // decode every q
                let k = codec.k_of(sm);
                let inv = 1.0 / (1.0 / step);
                for q in -(maxv as i32)..=maxv as i32 {
                    let want = (q as f64 * inv) as f32;
                    let (got, bad) = dec_v(q, sm, es, k);
                    assert!(!bad);
                    assert_eq!(got.to_bits(), want.to_bits(), "{f} dec q={q} step={step:e}");
                }
            }
        }
    }
}
