//! Reduced-precision storage for the chain-sweep bond register.
//!
//! The exact D=70 sweep of the IBM doped-Clifford circuit carries a 2^35
//! register: 256 GiB in complex64. Storing it in 16 bits per real component
//! (4 bytes per amplitude) halves that, 8 bits quarters it. This module
//! measures what that costs in fidelity by *emulating* the storage format:
//! the register is held in f32, every memory pass (a stage of the blocked
//! executor, or a single op, or a whole worldline) runs in f32, and at the
//! end of each pass every amplitude is rounded to the storage format and
//! back. That is bit-for-bit what a kernel that loads 16-bit values,
//! computes in f32 and stores 16-bit values would produce (the encoder is
//! round-to-nearest-even with gradual underflow and saturation), so the
//! fidelity numbers carry over; the memory saving itself needs packed-storage
//! kernels, which this module does not provide.
//!
//! Formats: bf16 / fp16, the OCP fp8 (E4M3, E5M2), fp6 (E3M2, E2M3) and
//! fp4 (E2M1) minifloats, and `b`-bit block floating point (a signed
//! integer per component with one scale per block, [`Format::Int`]).
//! Rounding is to nearest (ties to even) or stochastic.
//!
//! Scaling: the sweep uses unnormalised bond sums and the register norm
//! drifts by about half a bit per qubit, and fp16 / fp8 have a small
//! exponent range. [`Scaling::Block`] gives every block of `B` amplitudes a
//! shared scale chosen so the block's largest component lands at (integer
//! formats) or just below (floating formats, power-of-two scale) the
//! format's maximum; [`Scaling::Global`] does the same with one scale for
//! the whole register.

use crate::circuit::SimError;
use crate::engines::blocked::{fuse_1q, plan_stages_lookahead, BlockConfig, KOp};
use crate::engines::chain_sweep::SweepPlan;
use crate::engines::statevector::StateVector;
use num_complex::{Complex32, Complex64};
use rayon::prelude::*;

/// A storage format (per real component).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// IEEE single (no rounding: the f32 reference).
    F32,
    /// bfloat16: 8 exponent bits, 7 mantissa bits.
    Bf16,
    /// IEEE half: 5 exponent bits, 10 mantissa bits.
    Fp16,
    /// OCP fp8 E4M3 (max 448, no infinities).
    E4m3,
    /// OCP fp8 E5M2.
    E5m2,
    /// OCP fp6 E3M2 (max 28).
    E3m2,
    /// OCP fp6 E2M3 (max 7.5).
    E2m3,
    /// OCP fp4 E2M1 (max 6).
    E2m1,
    /// Signed integer of this many bits (symmetric, `±(2^(b-1)-1)`), used
    /// with a per-block scale: block floating point.
    Int(u32),
    /// Lloyd–Max quantizer for a unit Gaussian with `2^b` levels (b = 2 or
    /// 3), used with a per-block RMS scale.
    Lm(u32),
}

/// Positive Lloyd–Max reconstruction levels for a unit Gaussian.
const LM2: [f64; 2] = [0.4528, 1.5104];
const LM3: [f64; 4] = [0.2451, 0.7560, 1.3439, 2.1520];

fn lm_levels(b: u32) -> &'static [f64] {
    if b == 2 {
        &LM2
    } else {
        &LM3
    }
}

impl Format {
    /// (mantissa bits, minimum normal exponent, largest finite value) of a
    /// floating format.
    fn params(self) -> (i32, i32, f64) {
        match self {
            Format::F32 => (23, -126, f32::MAX as f64),
            Format::Bf16 => (7, -126, 3.389_531_389_251_535e38),
            Format::Fp16 => (10, -14, 65504.0),
            Format::E4m3 => (3, -6, 448.0),
            Format::E5m2 => (2, -14, 57344.0),
            Format::E3m2 => (2, -2, 28.0),
            Format::E2m3 => (3, 0, 7.5),
            Format::E2m1 => (1, 0, 6.0),
            Format::Int(b) => (0, 0, ((1u64 << (b - 1)) - 1) as f64),
            Format::Lm(b) => (0, 0, *lm_levels(b).last().unwrap()),
        }
    }

    /// Storage bits per real component (without the block scales).
    pub fn bits(self) -> u32 {
        match self {
            Format::F32 => 32,
            Format::Bf16 | Format::Fp16 => 16,
            Format::E4m3 | Format::E5m2 => 8,
            Format::E3m2 | Format::E2m3 => 6,
            Format::E2m1 => 4,
            Format::Int(b) | Format::Lm(b) => b,
        }
    }

    fn name(self) -> String {
        match self {
            Format::F32 => "f32".into(),
            Format::Bf16 => "bf16".into(),
            Format::Fp16 => "fp16".into(),
            Format::E4m3 => "e4m3".into(),
            Format::E5m2 => "e5m2".into(),
            Format::E3m2 => "e3m2".into(),
            Format::E2m3 => "e2m3".into(),
            Format::E2m1 => "e2m1".into(),
            Format::Int(b) => format!("int{b}"),
            Format::Lm(b) => format!("lm{b}"),
        }
    }
}

/// How amplitudes are scaled before rounding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scaling {
    /// Store the values as they are.
    None,
    /// One scale for the whole register, per pass.
    Global,
    /// One scale per block of this many amplitudes, per pass.
    Block(usize),
}

/// Storage format, scaling and rounding mode.
///
/// Floating formats get power-of-two scales (one exponent byte per block);
/// integer formats get the exact scale `max|component| / (2^(b-1)-1)`,
/// stored as f32, or with `half_scale` as an f16-precision mantissa
/// (rounded up) under one exponent per pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LowPrec {
    /// Number format of each real component.
    pub format: Format,
    /// Scaling applied before rounding.
    pub scaling: Scaling,
    /// Stochastic (unbiased) rounding instead of round-to-nearest-even.
    pub stochastic: bool,
    /// Integer formats: store the block scale in f16 instead of f32.
    pub half_scale: bool,
}

impl LowPrec {
    /// Parses `fmt[:g|:bN][:sr][:h]`, e.g. `bf16`, `fp16:b1024`, `e4m3:g`,
    /// `int4:b64:sr`, `int6:b256:h`.
    pub fn parse(s: &str) -> Option<Self> {
        let mut it = s.split(':');
        let f = it.next()?;
        let format = match f {
            "f32" => Format::F32,
            "bf16" => Format::Bf16,
            "fp16" => Format::Fp16,
            "e4m3" => Format::E4m3,
            "e5m2" => Format::E5m2,
            "e3m2" => Format::E3m2,
            "e2m3" => Format::E2m3,
            "e2m1" => Format::E2m1,
            "lm2" => Format::Lm(2),
            "lm3" => Format::Lm(3),
            _ => Format::Int(
                f.strip_prefix("int")?
                    .parse()
                    .ok()
                    .filter(|b| (2..=16).contains(b))?,
            ),
        };
        let mut lp = LowPrec {
            format,
            scaling: Scaling::None,
            stochastic: false,
            half_scale: false,
        };
        for opt in it {
            match opt {
                "g" => lp.scaling = Scaling::Global,
                "sr" => lp.stochastic = true,
                "h" => lp.half_scale = true,
                b => {
                    lp.scaling =
                        Scaling::Block(b.strip_prefix('b')?.parse().ok().filter(|&n| n > 0)?)
                }
            }
        }
        // an unscaled integer format has no meaning
        if matches!(format, Format::Int(_) | Format::Lm(_)) && lp.scaling == Scaling::None {
            return None;
        }
        Some(lp)
    }

    /// Storage bits per real component including the block scales.
    pub fn bits_per_component(&self) -> f64 {
        let scale_bits = match self.format {
            Format::Int(_) | Format::Lm(_) if self.half_scale => 16.0,
            Format::Int(_) | Format::Lm(_) => 32.0,
            _ => 8.0,
        };
        self.format.bits() as f64
            + match self.scaling {
                Scaling::Block(b) => scale_bits / (2 * b) as f64,
                _ => 0.0,
            }
    }
}

impl std::fmt::Display for LowPrec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format.name())?;
        match self.scaling {
            Scaling::None => {}
            Scaling::Global => write!(f, ":g")?,
            Scaling::Block(b) => write!(f, ":b{b}")?,
        }
        if self.stochastic {
            write!(f, ":sr")?;
        }
        if self.half_scale {
            write!(f, ":h")?;
        }
        Ok(())
    }
}

#[inline]
fn pow2(k: i32) -> f64 {
    f64::from_bits(((k + 1023) as u64) << 52)
}

/// Rounds `x` to the format: nearest with ties to even, or stochastically
/// (`u` uniform in `[0, 1)`: round up with probability equal to the
/// fractional position); gradual underflow; saturation at the largest
/// finite value. Integer formats round to the nearest integer.
#[inline]
pub fn round_to(x: f64, format: Format, u: Option<f64>) -> f64 {
    if format == Format::F32 {
        return x as f32 as f64;
    }
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    if let Format::Lm(b) = format {
        // nearest level (exact zeros, i.e. structurally free register
        // halves, are kept: they need not be stored)
        let a = x.abs();
        let lv = lm_levels(b);
        let mut best = lv[0];
        for &l in lv {
            if (a - l).abs() < (a - best).abs() {
                best = l;
            }
        }
        return best.copysign(x);
    }
    let (m, emin, maxv) = format.params();
    let q = if let Format::Int(_) = format {
        0
    } else {
        let e = ((x.to_bits() >> 52) & 0x7ff) as i32 - 1023;
        e.max(emin) - m
    };
    let y = x * pow2(-q);
    let r = match u {
        Some(u) => (y + u).floor(),
        None => y.round_ties_even(),
    } * pow2(q);
    r.clamp(-maxv, maxv)
}

/// Scale `s` (values are stored as `round(x * s)`) for a block whose
/// largest component is `maxabs`.
pub(crate) fn scale_for(maxabs: f64, lp: &LowPrec) -> f64 {
    if maxabs == 0.0 || !maxabs.is_finite() {
        return 1.0;
    }
    let (_, _, maxv) = lp.format.params();
    match lp.format {
        // `maxabs` is the block RMS here (see `block_stat`)
        Format::Lm(_) => {
            let r = if lp.half_scale {
                let e = ((maxabs.to_bits() >> 52) & 0x7ff) as i32 - 1023;
                let q = pow2(e - 10);
                (maxabs / q).round() * q
            } else {
                maxabs as f32 as f64
            };
            1.0 / r
        }
        Format::Int(_) => 1.0 / int_step(maxabs, lp),
        _ => pow2((maxv / maxabs).log2().floor() as i32),
    }
}

/// Quantisation step of an integer block format (`lp.format` must be
/// [`Format::Int`]) for a block whose largest component is `maxabs` (> 0,
/// finite): an f32 value, or with `half_scale` an 11-significant-bit
/// mantissa rounded up. The block scale is `s = 1 / step` and a component
/// `x` is stored as `round_to(x * s)`, decoded as `q * (1 / s)`.
pub(crate) fn int_step(maxabs: f64, lp: &LowPrec) -> f64 {
    let (_, _, maxv) = lp.format.params();
    let step = maxabs / maxv;
    if lp.half_scale {
        // an f16 mantissa (11 significant bits, rounded up so the
        // block never saturates) on a per-pass global exponent
        let e = ((step.to_bits() >> 52) & 0x7ff) as i32 - 1023;
        let q = pow2(e - 10);
        (step / q).ceil() * q
    } else {
        step as f32 as f64
    }
}

#[inline]
fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

static SR_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn round_block(v: &mut [Complex32], s: f64, lp: &LowPrec, seed: Option<u64>) {
    let inv = 1.0 / s;
    let mut st = seed.unwrap_or(0);
    let mut u = || {
        seed.map(|_| {
            st = splitmix(st);
            (st >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
        })
    };
    for z in v {
        z.re = (round_to(z.re as f64 * s, lp.format, u()) * inv) as f32;
        z.im = (round_to(z.im as f64 * s, lp.format, u()) * inv) as f32;
    }
}

/// The statistic a block scale is set from: the RMS over the nonzero
/// components for Lloyd–Max formats, the largest component otherwise.
fn block_stat(v: &[Complex32], lp: &LowPrec) -> f64 {
    if let Format::Lm(_) = lp.format {
        let (mut s2, mut n) = (0.0f64, 0usize);
        for z in v {
            for c in [z.re, z.im] {
                if c != 0.0 {
                    s2 += (c as f64) * (c as f64);
                    n += 1;
                }
            }
        }
        if n == 0 {
            0.0
        } else {
            (s2 / n as f64).sqrt()
        }
    } else {
        maxabs(v)
    }
}

pub(crate) fn maxabs(v: &[Complex32]) -> f64 {
    v.iter()
        .fold(0.0f32, |m, z| m.max(z.re.abs()).max(z.im.abs())) as f64
}

/// Rounds every amplitude of `amps` to the storage format and back.
pub fn quantize(amps: &mut [Complex32], lp: &LowPrec) {
    const CHUNK: usize = 1 << 14;
    let pass = SR_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let seed = |i: usize| {
        lp.stochastic
            .then(|| splitmix(pass.wrapping_mul(0x2545_f491_4f6c_dd1d) ^ i as u64))
    };
    match lp.scaling {
        Scaling::None => amps
            .par_chunks_mut(CHUNK)
            .enumerate()
            .for_each(|(i, c)| round_block(c, 1.0, lp, seed(i))),
        Scaling::Global => {
            let m = amps.par_chunks(CHUNK).map(maxabs).reduce(|| 0.0, f64::max);
            let s = scale_for(m, lp);
            amps.par_chunks_mut(CHUNK)
                .enumerate()
                .for_each(|(i, c)| round_block(c, s, lp, seed(i)));
        }
        Scaling::Block(b) => amps
            .par_chunks_mut(b)
            .enumerate()
            .for_each(|(i, c)| round_block(c, scale_for(block_stat(c, lp), lp), lp, seed(i))),
    }
}

/// Where the rounding happens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Granularity {
    /// After every executor op (worst case: no fusion at all).
    Op,
    /// After every stage of the blocked executor (one memory pass each).
    Stage,
    /// After every worldline (one rounding per qubit of the chain).
    Qubit,
    /// After every `m` executor ops (a fixed pass rate, independent of the
    /// register width and cache size).
    Every(usize),
}

/// Splits a plan into memory passes at the given granularity. Stage passes
/// are the blocked executor's own stages for `cfg` (after 1q fusion).
pub fn passes(plan: &SweepPlan, cfg: &BlockConfig, g: Granularity) -> Vec<Vec<KOp>> {
    let n = plan.width;
    match g {
        Granularity::Op => plan.ops.iter().map(|o| vec![*o]).collect(),
        Granularity::Every(m) => plan.ops.chunks(m.max(1)).map(|c| c.to_vec()).collect(),
        Granularity::Qubit => plan
            .qubit_ops
            .windows(2)
            .map(|w| plan.ops[w[0]..w[1]].to_vec())
            .filter(|v| !v.is_empty())
            .collect(),
        Granularity::Stage => {
            let ops = if cfg.fuse_1q {
                fuse_1q(&plan.ops, n, cfg.split_phases)
            } else {
                plan.ops.clone()
            };
            let l = cfg.block_bits(n, std::mem::size_of::<Complex32>());
            plan_stages_lookahead(&ops, n, l, cfg.slots)
                .into_iter()
                .map(|s| s.ops)
                .collect()
        }
    }
}

/// Result of a reduced-precision sweep.
#[derive(Clone, Debug)]
pub struct LowPrecRun {
    /// The amplitude (times the plan's scale).
    pub amp: Complex64,
    /// Number of roundings (memory passes).
    pub passes: usize,
    /// Register fidelity against an f64 run after pass `j` (only with
    /// `trace_every > 0`): `(pass index, |<ref|lp>|^2 / (|ref|^2 |lp|^2))`.
    pub trace: Vec<(usize, f64)>,
}

/// Runs `passes` on an f32 register, rounding to `lp` after each pass.
/// With `trace_every > 0` an f64 register runs alongside and the register
/// fidelity is recorded every `trace_every` passes (and after the last).
pub fn run_lowprec(
    plan: &SweepPlan,
    passes: &[Vec<KOp>],
    lp: &LowPrec,
    trace_every: usize,
) -> Result<LowPrecRun, SimError> {
    let cfg = BlockConfig {
        fuse_1q: false,
        ..BlockConfig::default()
    };
    let mut sv = StateVector::<f32>::try_new(plan.width)?;
    let mut rf = if trace_every > 0 {
        Some(StateVector::<f64>::try_new(plan.width)?)
    } else {
        None
    };
    let mut trace = Vec::new();
    for (j, p) in passes.iter().enumerate() {
        sv.apply_kops_blocked(p, &cfg);
        quantize(sv.amplitudes_mut(), lp);
        if let Some(r) = rf.as_mut() {
            r.apply_kops_blocked(p, &cfg);
            if (j + 1) % trace_every == 0 || j + 1 == passes.len() {
                let (ov, nr, nl) = r
                    .amplitudes()
                    .par_iter()
                    .zip(sv.amplitudes().par_iter())
                    .map(|(a, b)| {
                        let b = Complex64::new(b.re as f64, b.im as f64);
                        (a.conj() * b, a.norm_sqr(), b.norm_sqr())
                    })
                    .reduce(
                        || (Complex64::new(0.0, 0.0), 0.0, 0.0),
                        |x, y| (x.0 + y.0, x.1 + y.1, x.2 + y.2),
                    );
                let f = if nr > 0.0 && nl > 0.0 {
                    ov.norm_sqr() / (nr * nl)
                } else {
                    f64::NAN
                };
                trace.push((j + 1, f));
            }
        }
    }
    Ok(LowPrecRun {
        amp: sv.amplitude(0) * plan.scale,
        passes: passes.len(),
        trace,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_matches_known_values() {
        // bf16: 1 + 2^-8 is a tie between 1 and 1 + 2^-7 -> even (1)
        assert_eq!(round_to(1.0 + 2f64.powi(-8), Format::Bf16, None), 1.0);
        assert_eq!(
            round_to(1.0 + 3.0 * 2f64.powi(-8), Format::Bf16, None),
            1.0 + 2.0 * 2f64.powi(-7)
        );
        // fp16 max and saturation, smallest subnormal 2^-24
        assert_eq!(round_to(70000.0, Format::Fp16, None), 65504.0);
        assert_eq!(round_to(2f64.powi(-24), Format::Fp16, None), 2f64.powi(-24));
        assert_eq!(round_to(2f64.powi(-26), Format::Fp16, None), 0.0);
        // e4m3: 3 mantissa bits, max 448, subnormal step 2^-9
        assert_eq!(round_to(1.0625, Format::E4m3, None), 1.0);
        assert_eq!(round_to(1.1875, Format::E4m3, None), 1.25);
        assert_eq!(round_to(1000.0, Format::E4m3, None), 448.0);
        assert_eq!(
            round_to(-3.0 * 2f64.powi(-9), Format::E4m3, None),
            -3.0 * 2f64.powi(-9)
        );
    }

    #[test]
    fn block_scaling_keeps_tiny_values() {
        let mut v: Vec<Complex32> = (0..64)
            .map(|i| Complex32::new(1e-12 * (1.0 + i as f32 / 64.0), -3e-13))
            .collect();
        let orig = v.clone();
        quantize(&mut v, &LowPrec::parse("fp16:b16").unwrap());
        for (a, b) in v.iter().zip(&orig) {
            assert!((a - b).norm() <= 2e-3 * b.norm(), "{a} vs {b}");
        }
        // unscaled fp16 flushes them to zero
        let mut w = orig.clone();
        quantize(&mut w, &LowPrec::parse("fp16").unwrap());
        assert!(w.iter().all(|z| z.norm() == 0.0));
    }

    #[test]
    fn parse_roundtrip() {
        for s in [
            "bf16",
            "fp16:g",
            "fp16:b1024",
            "e4m3:b256",
            "e5m2",
            "f32",
            "int4:b64:sr",
            "int6:b256:h",
            "e2m1:b32",
        ] {
            assert_eq!(LowPrec::parse(s).unwrap().to_string(), s);
        }
        assert!(LowPrec::parse("fp16:b0").is_none());
        assert!(LowPrec::parse("int8").is_none());
        assert!(LowPrec::parse("int1:b8").is_none());
        let lp = LowPrec::parse("int4:b64:h").unwrap();
        assert!((lp.bits_per_component() - 4.125).abs() < 1e-12);
    }

    #[test]
    fn int_block_and_stochastic_rounding() {
        // int3 (levels -3..3) with an exact block scale: max maps to 3
        let mut v = vec![Complex32::new(3.0, -1.0), Complex32::new(0.9, 2.2)];
        quantize(&mut v, &LowPrec::parse("int3:b2").unwrap());
        assert_eq!(v, vec![Complex32::new(3.0, -1.0), Complex32::new(1.0, 2.0)]);
        // stochastic rounding is unbiased: the mean of many roundings of 0.3 (step 1) is 0.3
        let n = 20000;
        let mut w = vec![Complex32::new(0.3, -0.7); n];
        w.push(Complex32::new(3.0, 0.0)); // fixes the global scale at step 1
        quantize(&mut w, &LowPrec::parse("int3:g:sr").unwrap());
        let (mr, mi) = w[..n].iter().fold((0.0f64, 0.0f64), |a, z| {
            (a.0 + z.re as f64, a.1 + z.im as f64)
        });
        assert!((mr / n as f64 - 0.3).abs() < 0.02, "{}", mr / n as f64);
        assert!((mi / n as f64 + 0.7).abs() < 0.02, "{}", mi / n as f64);
        assert!(w[..n]
            .iter()
            .all(|z| (z.re == 0.0 || z.re == 1.0) && (z.im == -1.0 || z.im == 0.0)));
    }
}
