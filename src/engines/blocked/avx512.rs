//! AVX-512 kernels of the blocked state-vector executor (x86_64 only), the
//! tier above the AVX2+FMA build; selected at run time when the CPU has
//! AVX-512 F/DQ/VL/BW, `BlockConfig::avx512` is set and `QSIM_NO_AVX512` is
//! unset (see `research/performance/avx512.md`).
//!
//! The executor's generic kernels are written for autovectorisation over
//! long contiguous runs. With 16 `f32` (8 `f64`) lanes per register, every
//! target or control below bit 4 (3) sits *inside* a vector, where those
//! kernels fall back to short runs or scalar code. The explicit kernels here
//! work on whole 512-bit vectors at every qubit position:
//!
//! * **1-qubit gates** (`u1`): for a target inside the vector, the partner
//!   amplitude is one lane permutation away (`vpermps`), and the new value is
//!   `A x + B perm(x)` with per-lane coefficient vectors `A`, `B` (`m00`/`m01`
//!   on lanes whose target bit is 0, `m11`/`m10` where it is 1). Controls
//!   inside the vector are folded into the same vectors (`A = 1`, `B = 0` on
//!   lanes whose controls are not all set), so no lane is masked at run
//!   time; X gates are lane blends. Targets above the vector are the usual
//!   vertical pairs `(x, y)`, with the same per-lane trick for in-vector
//!   controls. Controls above the vector select which vectors are visited.
//! * **Pairs** (`pair`): two 1-qubit gates (and an optional CNOT between
//!   their bits) in one sweep, each gate either inside the vector or
//!   vertical.
//! * **Dense `2^k x 2^k` blocks** (`dense`, k = 2, 3): tiles of `2^k`
//!   vectors; every target inside the vector is first exchanged with a free
//!   vector bit by two-source permutes (`vpermt2ps`), so the matrix-vector
//!   product is purely vertical with broadcast matrix entries.
//!
//! Everything else (diagonal blocks, swaps, tiny blocks with fewer index
//! bits than a vector has lanes) runs the generic kernels, compiled here with
//! the AVX-512 features enabled. All kernels agree with the portable path to
//! rounding error (`tests/engines/avx512.rs`).

use super::{
    apply_lop, diag_pass_tables, for_each_run, prof, Buf, DiagPass, DiagScratch, LOp, Prepared,
    Seg, UKind, View, C1,
};
use crate::engines::statevector::Real;
use core::arch::x86_64::*;

/// 512-bit vector operations for one amplitude type: `f32` (16 lanes) or
/// `f64` (8 lanes). All methods are `unsafe`: they must only be reached from
/// code compiled with the AVX-512 F/DQ/VL/BW and FMA target features, on a
/// CPU that has them (the dispatch in `super::run_ops` guarantees both).
pub(super) trait Simd: Real {
    /// The vector type.
    type V: Copy;
    /// log2 of the number of lanes.
    const LB: usize;
    /// Unaligned load of one vector.
    unsafe fn ld(p: *const Self) -> Self::V;
    /// Unaligned store of one vector.
    unsafe fn st(p: *mut Self, v: Self::V);
    /// Broadcast.
    unsafe fn splat(x: Self) -> Self::V;
    /// Vector from per-lane values (set-up code only).
    unsafe fn lanes(f: impl Fn(usize) -> Self) -> Self::V;
    /// `a * b + c`, one rounding.
    unsafe fn fma(a: Self::V, b: Self::V, c: Self::V) -> Self::V;
    /// `c - a * b`, one rounding.
    unsafe fn fnma(a: Self::V, b: Self::V, c: Self::V) -> Self::V;
    /// `a * b`.
    unsafe fn vmul(a: Self::V, b: Self::V) -> Self::V;
    /// Lane `l` of the result is lane `idx[l]` of `x`.
    unsafe fn perm(idx: __m512i, x: Self::V) -> Self::V;
    /// Lane `l` of the result is lane `idx[l]` of the concatenation `a ++ b`.
    unsafe fn perm2(a: Self::V, idx: __m512i, b: Self::V) -> Self::V;
    /// Lanes whose bit in `k` is set come from `b`, the others from `a`.
    unsafe fn blend(k: u16, a: Self::V, b: Self::V) -> Self::V;
    /// Index vector for [`Simd::perm`] / [`Simd::perm2`] (set-up code only).
    unsafe fn idx(f: impl Fn(usize) -> usize) -> __m512i;
}

impl Simd for f32 {
    type V = __m512;
    const LB: usize = 4;
    #[inline(always)]
    unsafe fn ld(p: *const f32) -> __m512 {
        _mm512_loadu_ps(p)
    }
    #[inline(always)]
    unsafe fn st(p: *mut f32, v: __m512) {
        _mm512_storeu_ps(p, v)
    }
    #[inline(always)]
    unsafe fn splat(x: f32) -> __m512 {
        _mm512_set1_ps(x)
    }
    #[inline(always)]
    unsafe fn lanes(f: impl Fn(usize) -> f32) -> __m512 {
        let a: [f32; 16] = std::array::from_fn(f);
        _mm512_loadu_ps(a.as_ptr())
    }
    #[inline(always)]
    unsafe fn fma(a: __m512, b: __m512, c: __m512) -> __m512 {
        _mm512_fmadd_ps(a, b, c)
    }
    #[inline(always)]
    unsafe fn fnma(a: __m512, b: __m512, c: __m512) -> __m512 {
        _mm512_fnmadd_ps(a, b, c)
    }
    #[inline(always)]
    unsafe fn vmul(a: __m512, b: __m512) -> __m512 {
        _mm512_mul_ps(a, b)
    }
    #[inline(always)]
    unsafe fn perm(idx: __m512i, x: __m512) -> __m512 {
        _mm512_permutexvar_ps(idx, x)
    }
    #[inline(always)]
    unsafe fn perm2(a: __m512, idx: __m512i, b: __m512) -> __m512 {
        _mm512_permutex2var_ps(a, idx, b)
    }
    #[inline(always)]
    unsafe fn blend(k: u16, a: __m512, b: __m512) -> __m512 {
        _mm512_mask_blend_ps(k, a, b)
    }
    #[inline(always)]
    unsafe fn idx(f: impl Fn(usize) -> usize) -> __m512i {
        let a: [i32; 16] = std::array::from_fn(|i| f(i) as i32);
        _mm512_loadu_si512(a.as_ptr().cast())
    }
}

impl Simd for f64 {
    type V = __m512d;
    const LB: usize = 3;
    #[inline(always)]
    unsafe fn ld(p: *const f64) -> __m512d {
        _mm512_loadu_pd(p)
    }
    #[inline(always)]
    unsafe fn st(p: *mut f64, v: __m512d) {
        _mm512_storeu_pd(p, v)
    }
    #[inline(always)]
    unsafe fn splat(x: f64) -> __m512d {
        _mm512_set1_pd(x)
    }
    #[inline(always)]
    unsafe fn lanes(f: impl Fn(usize) -> f64) -> __m512d {
        let a: [f64; 8] = std::array::from_fn(f);
        _mm512_loadu_pd(a.as_ptr())
    }
    #[inline(always)]
    unsafe fn fma(a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        _mm512_fmadd_pd(a, b, c)
    }
    #[inline(always)]
    unsafe fn fnma(a: __m512d, b: __m512d, c: __m512d) -> __m512d {
        _mm512_fnmadd_pd(a, b, c)
    }
    #[inline(always)]
    unsafe fn vmul(a: __m512d, b: __m512d) -> __m512d {
        _mm512_mul_pd(a, b)
    }
    #[inline(always)]
    unsafe fn perm(idx: __m512i, x: __m512d) -> __m512d {
        _mm512_permutexvar_pd(idx, x)
    }
    #[inline(always)]
    unsafe fn perm2(a: __m512d, idx: __m512i, b: __m512d) -> __m512d {
        _mm512_permutex2var_pd(a, idx, b)
    }
    #[inline(always)]
    unsafe fn blend(k: u16, a: __m512d, b: __m512d) -> __m512d {
        _mm512_mask_blend_pd(k as u8, a, b)
    }
    #[inline(always)]
    unsafe fn idx(f: impl Fn(usize) -> usize) -> __m512i {
        let a: [i64; 8] = std::array::from_fn(|i| f(i) as i64);
        _mm512_loadu_si512(a.as_ptr().cast())
    }
}

/// Lanes per vector.
#[inline(always)]
fn nl<S: Simd>() -> usize {
    1 << S::LB
}

/// Lane mask with bit `i` set where `f(i)`.
#[inline(always)]
fn lane_mask<S: Simd>(f: impl Fn(usize) -> bool) -> u16 {
    (0..nl::<S>())
        .filter(|&i| f(i))
        .fold(0u16, |k, i| k | (1 << i))
}

/// Complex coefficients `A` (applied to the amplitude itself) and `B`
/// (applied to its partner), per lane.
#[derive(Clone, Copy)]
struct Coef<V> {
    ar: V,
    ai: V,
    br: V,
    bi: V,
}

/// Builds coefficient vectors from per-lane `(A, B)` = `((ar, ai), (br, bi))`.
#[inline(always)]
unsafe fn coef<S: Simd>(f: impl Fn(usize) -> ((S, S), (S, S))) -> Coef<S::V> {
    Coef {
        ar: S::lanes(|i| f(i).0 .0),
        ai: S::lanes(|i| f(i).0 .1),
        br: S::lanes(|i| f(i).1 .0),
        bi: S::lanes(|i| f(i).1 .1),
    }
}

/// `A x + B p` (complex; `REAL`: imaginary coefficients are zero).
#[inline(always)]
unsafe fn cmac<S: Simd, const REAL: bool>(
    c: &Coef<S::V>,
    xr: S::V,
    xi: S::V,
    pr: S::V,
    pi: S::V,
) -> (S::V, S::V) {
    if REAL {
        (
            S::fma(c.ar, xr, S::vmul(c.br, pr)),
            S::fma(c.ar, xi, S::vmul(c.br, pi)),
        )
    } else {
        let r = S::fma(
            c.ar,
            xr,
            S::fnma(c.ai, xi, S::fnma(c.bi, pi, S::vmul(c.br, pr))),
        );
        let i = S::fma(
            c.ar,
            xi,
            S::fma(c.ai, xr, S::fma(c.bi, pr, S::vmul(c.br, pi))),
        );
        (r, i)
    }
}

/// The four entries of a 2x2 matrix `[m0 m1; m2 m3]` given as executor
/// coefficients `[m0r, m1r, m2r, m3r, m0i, m1i, m2i, m3i]`, per lane
/// active/inactive: `(m0, m1)` for the amplitude with target bit 0, `(m3,
/// m2)` for bit 1, identity on inactive lanes.
#[inline(always)]
fn entries<S: Simd>(m: &[S; 8], bit: usize, active: bool) -> ((S, S), (S, S)) {
    let [m0r, m1r, m2r, m3r, m0i, m1i, m2i, m3i] = *m;
    match (active, bit) {
        (false, _) => ((S::one(), S::zero()), (S::zero(), S::zero())),
        (true, 0) => ((m0r, m0i), (m1r, m1i)),
        _ => ((m3r, m3i), (m2r, m2i)),
    }
}

/// 1-qubit gate `m` (kind `kind`) on buffer bit `t`, active where the
/// in-buffer controls `cin` are all set, on a buffer of `2^l` amplitudes.
/// Returns `false` (doing nothing) if the buffer is smaller than a vector.
#[inline(always)]
unsafe fn u1<S: Simd>(
    re: &mut [S],
    im: &mut [S],
    l: usize,
    t: usize,
    m: &[S; 8],
    kind: UKind,
    cin: usize,
) -> bool {
    let lb = S::LB;
    if l < lb || re.len() < 1 << l || im.len() < 1 << l {
        return false;
    }
    let cin_lo = cin & (nl::<S>() - 1);
    let cvec = cin >> lb;
    let nvb = l - lb;
    let active = |i: usize| i & cin_lo == cin_lo;
    let (pr, pi) = (re.as_mut_ptr(), im.as_mut_ptr());
    if t < lb {
        let s = 1usize << t;
        let idx = S::idx(|i| i ^ s);
        match kind {
            UKind::X => {
                let k = lane_mask::<S>(active);
                for_each_run(nvb, cvec, |b, run| {
                    for v in (b | cvec)..(b | cvec) + run {
                        let o = v << lb;
                        let (xr, xi) = (S::ld(pr.add(o)), S::ld(pi.add(o)));
                        S::st(pr.add(o), S::blend(k, xr, S::perm(idx, xr)));
                        S::st(pi.add(o), S::blend(k, xi, S::perm(idx, xi)));
                    }
                });
            }
            UKind::Real | UKind::Complex => {
                let c = coef::<S>(|i| entries(m, (i >> t) & 1, active(i)));
                macro_rules! go {
                    ($real:literal) => {
                        for_each_run(nvb, cvec, |b, run| {
                            for v in (b | cvec)..(b | cvec) + run {
                                let o = v << lb;
                                let (xr, xi) = (S::ld(pr.add(o)), S::ld(pi.add(o)));
                                let (yr, yi) = cmac::<S, $real>(
                                    &c,
                                    xr,
                                    xi,
                                    S::perm(idx, xr),
                                    S::perm(idx, xi),
                                );
                                S::st(pr.add(o), yr);
                                S::st(pi.add(o), yi);
                            }
                        })
                    };
                }
                if kind == UKind::Real {
                    go!(true)
                } else {
                    go!(false)
                }
            }
        }
        return true;
    }
    // Vertical: x at vector v (target bit 0), y at v | tv (target bit 1).
    let tv = 1usize << (t - lb);
    let fixed = cvec | tv;
    match kind {
        UKind::X => {
            let k = lane_mask::<S>(active);
            for_each_run(nvb, fixed, |b, run| {
                for v in (b | cvec)..(b | cvec) + run {
                    let (ox, oy) = (v << lb, (v | tv) << lb);
                    let (xr, xi) = (S::ld(pr.add(ox)), S::ld(pi.add(ox)));
                    let (yr, yi) = (S::ld(pr.add(oy)), S::ld(pi.add(oy)));
                    S::st(pr.add(ox), S::blend(k, xr, yr));
                    S::st(pi.add(ox), S::blend(k, xi, yi));
                    S::st(pr.add(oy), S::blend(k, yr, xr));
                    S::st(pi.add(oy), S::blend(k, yi, xi));
                }
            });
        }
        UKind::Real | UKind::Complex => {
            let c0 = coef::<S>(|i| entries(m, 0, active(i)));
            let c1 = coef::<S>(|i| entries(m, 1, active(i)));
            macro_rules! go {
                ($real:literal) => {
                    for_each_run(nvb, fixed, |b, run| {
                        for v in (b | cvec)..(b | cvec) + run {
                            let (ox, oy) = (v << lb, (v | tv) << lb);
                            let (xr, xi) = (S::ld(pr.add(ox)), S::ld(pi.add(ox)));
                            let (yr, yi) = (S::ld(pr.add(oy)), S::ld(pi.add(oy)));
                            let (ar, ai) = cmac::<S, $real>(&c0, xr, xi, yr, yi);
                            let (br, bi) = cmac::<S, $real>(&c1, yr, yi, xr, xi);
                            S::st(pr.add(ox), ar);
                            S::st(pi.add(ox), ai);
                            S::st(pr.add(oy), br);
                            S::st(pi.add(oy), bi);
                        }
                    })
                };
            }
            if kind == UKind::Real {
                go!(true)
            } else {
                go!(false)
            }
        }
    }
    true
}

/// A 2x2 gate prepared for [`pair`]: per-lane coefficients for the
/// in-vector case (`lane`, with the lane permutation `idx`) or broadcast
/// coefficients for the two halves of a vertical pair (`v0`, `v1`).
struct Gate2<V> {
    real: bool,
    idx: __m512i,
    lane: Coef<V>,
    v0: Coef<V>,
    v1: Coef<V>,
}

#[inline(always)]
unsafe fn gate2<S: Simd>(m: &[S; 8], kind: UKind, t: usize) -> Gate2<S::V> {
    let s = 1usize << t.min(S::LB - 1);
    Gate2 {
        real: kind != UKind::Complex,
        idx: S::idx(|i| i ^ s),
        lane: coef::<S>(|i| entries(m, (i >> t) & 1, true)),
        v0: coef::<S>(|_| entries(m, 0, true)),
        v1: coef::<S>(|_| entries(m, 1, true)),
    }
}

/// Applies `g` to one vector whose target bit is inside the vector.
#[inline(always)]
unsafe fn on_lane<S: Simd>(g: &Gate2<S::V>, x: (S::V, S::V)) -> (S::V, S::V) {
    let (pr, pi) = (S::perm(g.idx, x.0), S::perm(g.idx, x.1));
    if g.real {
        cmac::<S, true>(&g.lane, x.0, x.1, pr, pi)
    } else {
        cmac::<S, false>(&g.lane, x.0, x.1, pr, pi)
    }
}

/// Applies `g` to a vertical pair (`x`: target bit 0, `y`: target bit 1).
#[inline(always)]
#[allow(clippy::type_complexity)]
unsafe fn on_pair<S: Simd>(
    g: &Gate2<S::V>,
    x: (S::V, S::V),
    y: (S::V, S::V),
) -> ((S::V, S::V), (S::V, S::V)) {
    if g.real {
        (
            cmac::<S, true>(&g.v0, x.0, x.1, y.0, y.1),
            cmac::<S, true>(&g.v1, y.0, y.1, x.0, x.1),
        )
    } else {
        (
            cmac::<S, false>(&g.v0, x.0, x.1, y.0, y.1),
            cmac::<S, false>(&g.v1, y.0, y.1, x.0, x.1),
        )
    }
}

/// Lanes in `k` from `y`, the others from `x` (re and im).
#[inline(always)]
unsafe fn blend2<S: Simd>(k: u16, x: (S::V, S::V), y: (S::V, S::V)) -> (S::V, S::V) {
    (S::blend(k, x.0, y.0), S::blend(k, x.1, y.1))
}

#[inline(always)]
unsafe fn ld2<S: Simd>(pr: *mut S, pi: *mut S, o: usize) -> (S::V, S::V) {
    (S::ld(pr.add(o)), S::ld(pi.add(o)))
}

#[inline(always)]
unsafe fn st2<S: Simd>(pr: *mut S, pi: *mut S, o: usize, x: (S::V, S::V)) {
    S::st(pr.add(o), x.0);
    S::st(pi.add(o), x.1);
}

/// `m1` on bit `t1` and `m2` on bit `t2` (`t1 < t2`, uncontrolled), then for
/// `cx = 1` a CNOT `t1 -> t2` and for `cx = 2` a CNOT `t2 -> t1`, in one
/// sweep (the executor's `LOp::Pair`). Each gate is applied inside the
/// vector or across vectors depending on its bit. Returns `false` if the
/// buffer is smaller than a vector.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn pair<S: Simd>(
    re: &mut [S],
    im: &mut [S],
    l: usize,
    t1: usize,
    m1: &[S; 8],
    k1: UKind,
    t2: usize,
    m2: &[S; 8],
    k2: UKind,
    cx: u8,
) -> bool {
    let lb = S::LB;
    if l < lb || re.len() < 1 << l || im.len() < 1 << l {
        return false;
    }
    let nvb = l - lb;
    let (pr, pi) = (re.as_mut_ptr(), im.as_mut_ptr());
    let (g1, g2) = (gate2::<S>(m1, k1, t1), gate2::<S>(m2, k2, t2));
    if t2 < lb {
        // both inside the vector
        let k1m = lane_mask::<S>(|i| (i >> t1) & 1 == 1);
        let k2m = lane_mask::<S>(|i| (i >> t2) & 1 == 1);
        for v in 0..1usize << nvb {
            let o = v << lb;
            let x = on_lane::<S>(&g2, on_lane::<S>(&g1, ld2::<S>(pr, pi, o)));
            let x = match cx {
                // control t1: lanes with bit t1 set take their t2 partner
                1 => blend2::<S>(k1m, x, (S::perm(g2.idx, x.0), S::perm(g2.idx, x.1))),
                2 => blend2::<S>(k2m, x, (S::perm(g1.idx, x.0), S::perm(g1.idx, x.1))),
                _ => x,
            };
            st2::<S>(pr, pi, o, x);
        }
    } else if t1 < lb {
        // t1 inside the vector, t2 vertical
        let tv = 1usize << (t2 - lb);
        let k1m = lane_mask::<S>(|i| (i >> t1) & 1 == 1);
        for_each_run(nvb, tv, |b, run| {
            for v in b..b + run {
                let (ox, oy) = (v << lb, (v | tv) << lb);
                let x = on_lane::<S>(&g1, ld2::<S>(pr, pi, ox));
                let y = on_lane::<S>(&g1, ld2::<S>(pr, pi, oy));
                let (x, y) = on_pair::<S>(&g2, x, y);
                let (x, y) = match cx {
                    1 => (blend2::<S>(k1m, x, y), blend2::<S>(k1m, y, x)),
                    2 => (x, (S::perm(g1.idx, y.0), S::perm(g1.idx, y.1))),
                    _ => (x, y),
                };
                st2::<S>(pr, pi, ox, x);
                st2::<S>(pr, pi, oy, y);
            }
        });
    } else {
        // both vertical: a0 (bits t1, t2 = 0, 0), a1 (1, 0), a2 (0, 1), a3 (1, 1)
        let (s1, s2) = (1usize << (t1 - lb), 1usize << (t2 - lb));
        for_each_run(nvb, s1 | s2, |b, run| {
            for v in b..b + run {
                let o = [v << lb, (v | s1) << lb, (v | s2) << lb, (v | s1 | s2) << lb];
                let a: [(S::V, S::V); 4] = std::array::from_fn(|j| ld2::<S>(pr, pi, o[j]));
                let (b0, b1) = on_pair::<S>(&g1, a[0], a[1]);
                let (b2, b3) = on_pair::<S>(&g1, a[2], a[3]);
                let (c0, c2) = on_pair::<S>(&g2, b0, b2);
                let (c1, c3) = on_pair::<S>(&g2, b1, b3);
                let (c1, c2, c3) = match cx {
                    1 => (c3, c2, c1),
                    2 => (c1, c3, c2),
                    _ => (c1, c2, c3),
                };
                st2::<S>(pr, pi, o[0], c0);
                st2::<S>(pr, pi, o[1], c1);
                st2::<S>(pr, pi, o[2], c2);
                st2::<S>(pr, pi, o[3], c3);
            }
        });
    }
    true
}

/// Exchanges lane bit `t[i]` with tile-vector bit `i` for the first `a`
/// targets (`xa`/`xb`: the two-source permutes; an involution). A function,
/// not a closure: closures do not inherit `#[target_feature]`, and an
/// out-of-line closure would call every permute out of line.
#[inline(always)]
unsafe fn exchange<S: Simd, const K: usize, const D: usize>(
    x: &mut [S::V; D],
    a: usize,
    xa: &[__m512i; K],
    xb: &[__m512i; K],
) {
    for i in 0..a {
        for v in 0..D {
            if (v >> i) & 1 == 0 {
                let w = v | (1 << i);
                let (p, q) = (x[v], x[w]);
                x[v] = S::perm2(p, xa[i], q);
                x[w] = S::perm2(p, xb[i], q);
            }
        }
    }
}

/// Dense `D x D` unitary (`D = 2^K`, row-major `mr`/`mi`, local index bit
/// `j` = buffer bit `t[j]`, `t` ascending) on a buffer of `2^l` amplitudes.
/// Returns `false` if the buffer has fewer than `K` index bits above the
/// vector.
#[inline(always)]
unsafe fn dense<S: Simd, const K: usize, const D: usize>(
    re: &mut [S],
    im: &mut [S],
    l: usize,
    t: [usize; K],
    mr: &[S],
    mi: &[S],
) -> bool {
    let lb = S::LB;
    if l < lb + K || re.len() < 1 << l || im.len() < 1 << l || mr.len() < D * D || mi.len() < D * D
    {
        return false;
    }
    let tmask: usize = t.iter().map(|&q| 1usize << q).sum();
    let a = t.iter().filter(|&&q| q < lb).count();
    // vector-index bit i <-> buffer bit lst[i]: partners of the in-vector
    // targets (lowest free bits >= lb), then the targets above the vector
    let mut lst = [0usize; K];
    let mut p = lb;
    for slot in lst.iter_mut().take(a) {
        while (tmask >> p) & 1 == 1 {
            p += 1;
        }
        *slot = p;
        p += 1;
    }
    lst[a..K].copy_from_slice(&t[a..K]);
    let off: [usize; D] = std::array::from_fn(|v| {
        (0..K)
            .filter(|&i| (v >> i) & 1 == 1)
            .map(|i| 1usize << lst[i])
            .sum()
    });
    let mask = lst.iter().map(|&q| 1usize << q).sum::<usize>() | (nl::<S>() - 1);
    // exchange of lane bit t[i] with vector bit i (an involution)
    let lanes = nl::<S>();
    let xa: [__m512i; K] = std::array::from_fn(|i| {
        let j = 1usize << t[i].min(lb - 1);
        S::idx(|l| if l & j == 0 { l } else { lanes + (l ^ j) })
    });
    let xb: [__m512i; K] = std::array::from_fn(|i| {
        let j = 1usize << t[i].min(lb - 1);
        S::idx(|l| if l & j == 0 { l | j } else { lanes + l })
    });
    let (pr, pi) = (re.as_mut_ptr(), im.as_mut_ptr());
    let (mr, mi) = (mr.as_ptr(), mi.as_ptr());
    let zero = S::splat(S::zero());
    let mut base = 0usize;
    for _ in 0..1usize << (l - lb - K) {
        let mut xr = [zero; D];
        let mut xi = [zero; D];
        for v in 0..D {
            xr[v] = S::ld(pr.add(base + off[v]));
            xi[v] = S::ld(pi.add(base + off[v]));
        }
        exchange::<S, K, D>(&mut xr, a, &xa, &xb);
        exchange::<S, K, D>(&mut xi, a, &xa, &xb);
        let mut yr = xr;
        let mut yi = xi;
        for r in 0..D {
            let (a, b) = (S::splat(*mr.add(r * D)), S::splat(*mi.add(r * D)));
            let mut ar = S::fnma(b, xi[0], S::vmul(a, xr[0]));
            let mut ai = S::fma(b, xr[0], S::vmul(a, xi[0]));
            for c in 1..D {
                let (a, b) = (S::splat(*mr.add(r * D + c)), S::splat(*mi.add(r * D + c)));
                ar = S::fnma(b, xi[c], S::fma(a, xr[c], ar));
                ai = S::fma(b, xr[c], S::fma(a, xi[c], ai));
            }
            yr[r] = ar;
            yi[r] = ai;
        }
        exchange::<S, K, D>(&mut yr, a, &xa, &xb);
        exchange::<S, K, D>(&mut yi, a, &xa, &xb);
        for v in 0..D {
            S::st(pr.add(base + off[v]), yr[v]);
            S::st(pi.add(base + off[v]), yi[v]);
        }
        base = ((base | mask) + 1) & !mask;
    }
    true
}

/// Single-pass diagonal block (`super::DiagPass`): per row `h`, the
/// factor of lane `i` of vector `v` is `lo[x] * hi[h] * L_h[i] * V_h[v]`,
/// where `L_h` (lane bits) and `V_h` (vector bits) collect the crossing
/// terms active in the row. Returns `false` if a row is narrower than a
/// vector.
#[inline(always)]
unsafe fn diag_pass<S: Simd>(
    re: &mut [S],
    im: &mut [S],
    l: usize,
    d: &DiagPass<S>,
    base: usize,
    sc: &mut DiagScratch<S>,
) -> bool {
    let lb = S::LB;
    let lo_bits = l.min(d.lo_bits);
    if lo_bits < lb || re.len() < 1 << l || im.len() < 1 << l {
        return false;
    }
    let (lor, loi, _, hi, cross) = diag_pass_tables(d, base, sc);
    let (pr, pi) = (re.as_mut_ptr(), im.as_mut_ptr());
    let (tr, ti) = (lor.as_ptr(), loi.as_ptr());
    let nv = 1usize << (lo_bits - lb);
    let lane_bits = (1u32 << lb) - 1;
    #[allow(clippy::needless_range_loop)]
    for h in 0..1usize << (l - lo_bits) {
        let mut e = [[C1; 2]; super::LO_BITS];
        let mut used = 0u32;
        for c in cross {
            if h & c.hmask == c.hpat {
                e[c.a][c.va as usize] *= c.f;
                used |= 1 << c.a;
            }
        }
        let s = hi[h];
        let lane_used = used & lane_bits;
        let (lr, li) = if lane_used == 0 {
            (S::splat(S::from_f64(s.re)), S::splat(S::from_f64(s.im)))
        } else {
            let f = |i: usize| {
                let mut z = s;
                let mut m = lane_used;
                while m != 0 {
                    let a = m.trailing_zeros() as usize;
                    m &= m - 1;
                    z *= e[a][(i >> a) & 1];
                }
                z
            };
            (
                S::lanes(|i| S::from_f64(f(i).re)),
                S::lanes(|i| S::from_f64(f(i).im)),
            )
        };
        let vec_used = used >> lb;
        let row = h << lo_bits;
        for v in 0..nv {
            let (cr, ci) = if vec_used == 0 {
                (lr, li)
            } else {
                let mut z = C1;
                let mut m = vec_used;
                while m != 0 {
                    let b = m.trailing_zeros() as usize;
                    m &= m - 1;
                    z *= e[lb + b][(v >> b) & 1];
                }
                let (zr, zi) = (S::splat(S::from_f64(z.re)), S::splat(S::from_f64(z.im)));
                (
                    S::fnma(li, zi, S::vmul(lr, zr)),
                    S::fma(li, zr, S::vmul(lr, zi)),
                )
            };
            let o = v << lb;
            let (ar, ai) = (S::ld(tr.add(o)), S::ld(ti.add(o)));
            let fr = S::fnma(ai, ci, S::vmul(ar, cr));
            let fi = S::fma(ai, cr, S::vmul(ar, ci));
            let p = row + o;
            let (xr, xi) = (S::ld(pr.add(p)), S::ld(pi.add(p)));
            S::st(pr.add(p), S::fnma(xi, fi, S::vmul(xr, fr)));
            S::st(pi.add(p), S::fma(xi, fr, S::vmul(xr, fi)));
        }
    }
    true
}

/// The target features of every function below; each op kind gets its own
/// non-inlined function (instantiated for `f32` and `f64` in this crate
/// only), which keeps the per-type stage loop small for the compiler.
macro_rules! avx512_fn {
    ($(#[$m:meta])* fn $name:ident<$s:ident>($($arg:ident: $ty:ty),* $(,)?) $body:block) => {
        $(#[$m])*
        #[target_feature(enable = "avx512f,avx512dq,avx512vl,avx512bw,avx2,fma")]
        #[inline(never)]
        #[allow(clippy::too_many_arguments)]
        unsafe fn $name<$s: Simd>($($arg: $ty),*) $body
    };
}

avx512_fn! {
    /// The generic kernels (autovectorised with AVX-512 enabled), one copy
    /// per type: ops without an explicit kernel (pivot-form diagonal blocks,
    /// swaps) and the explicit kernels' fallbacks on blocks smaller than
    /// they need.
    fn op_generic<S>(v: &mut View<S>, l: usize, op: &LOp<S>, base: usize, sc: &mut DiagScratch<S>) {
        apply_lop::<S, true>(v, l, op, base, sc);
    }
}

avx512_fn! {
    /// 1-qubit op (explicit kernel, generic one on blocks below a vector).
    fn op_u1<S>(v: &mut View<S>, l: usize, t: usize, m: &[S; 8], kind: UKind, cin: usize) {
        if !u1::<S>(v.re, v.im, l, t, m, kind, cin) {
            super::apply_u1::<S, true>(v, l, t, m, kind, cin);
        }
    }
}

avx512_fn! {
    /// Pair op (explicit kernel, generic one on blocks below a vector).
    fn op_pair<S>(v: &mut View<S>, l: usize, op: &LOp<S>, base: usize, sc: &mut DiagScratch<S>) {
        if let LOp::Pair { t1, m1, k1, t2, m2, k2, cx } = op {
            if pair::<S>(v.re, v.im, l, *t1, m1, *k1, *t2, m2, *k2, *cx) {
                return;
            }
        }
        op_generic::<S>(v, l, op, base, sc);
    }
}

avx512_fn! {
    /// Dense op (explicit kernel, generic one on blocks too small for it).
    fn op_dense<S>(v: &mut View<S>, l: usize, op: &LOp<S>, base: usize, sc: &mut DiagScratch<S>) {
        if let LOp::Dense { k, t, re, im } = op {
            let done = match k {
                2 => dense::<S, 2, 4>(v.re, v.im, l, [t[0], t[1]], re, im),
                _ => dense::<S, 3, 8>(v.re, v.im, l, [t[0], t[1], t[2]], re, im),
            };
            if done {
                return;
            }
        }
        op_generic::<S>(v, l, op, base, sc);
    }
}

avx512_fn! {
    /// Single-pass diagonal op (explicit kernel, generic one below a vector).
    fn op_diag_pass<S>(v: &mut View<S>, l: usize, op: &LOp<S>, base: usize, sc: &mut DiagScratch<S>) {
        if let LOp::DiagPass(d) = op {
            if diag_pass::<S>(v.re, v.im, l, d, base, sc) {
                return;
            }
        }
        op_generic::<S>(v, l, op, base, sc);
    }
}

/// Applies one op with the explicit kernels where they apply and the
/// generic ones (compiled with AVX-512 enabled) otherwise.
#[inline(always)]
unsafe fn apply_op<S: Simd>(
    v: &mut View<S>,
    l: usize,
    op: &LOp<S>,
    base: usize,
    sc: &mut DiagScratch<S>,
) {
    match op {
        LOp::U1 {
            t,
            m,
            kind,
            cin,
            cout,
        } => {
            if base & cout == *cout {
                op_u1::<S>(v, l, *t, m, *kind, *cin);
            }
        }
        LOp::Pair { .. } => op_pair::<S>(v, l, op, base, sc),
        LOp::Dense { .. } => op_dense::<S>(v, l, op, base, sc),
        LOp::DiagPass(_) => op_diag_pass::<S>(v, l, op, base, sc),
        LOp::Diag(_) | LOp::Swap { .. } => op_generic::<S>(v, l, op, base, sc),
    }
}

/// Profile bucket of an op (same buckets as the generic executor).
fn prof_kind<S: Simd>(op: &LOp<S>) -> usize {
    match op {
        LOp::U1 { t, kind, cin, .. } => {
            if (cin | (1 << t)) & (nl::<S>() - 1) != 0 {
                3
            } else {
                *kind as usize
            }
        }
        LOp::Swap { .. } => 4,
        LOp::Pair { .. } => 2,
        LOp::Diag(_) | LOp::DiagPass(_) => 5,
        LOp::Dense { .. } => 8,
    }
}

/// The stage's ops on one block (inlined into the two concrete,
/// target-feature entry points below, which are compiled once, in this
/// crate, instead of in every crate that instantiates the executor).
///
/// # Safety
/// The CPU must support AVX-512 F/DQ/VL/BW, AVX2 and FMA.
#[inline(always)]
unsafe fn run_ops_t<S: Simd>(
    p: &Prepared<S>,
    buf: &mut Buf<S>,
    base: usize,
    sc: &mut DiagScratch<S>,
) {
    if let Some(tp) = &p.tile {
        let k = tp.k;
        let tile = 1usize << k;
        let lowmask = tile - 1;
        for seg in &tp.segs {
            match seg {
                Seg::Tile(run) => {
                    for (ti, (re, im)) in buf
                        .re
                        .chunks_exact_mut(tile)
                        .zip(buf.im.chunks_exact_mut(tile))
                        .enumerate()
                    {
                        let mut v = View { re, im };
                        for op in run {
                            match op {
                                LOp::U1 {
                                    t,
                                    m,
                                    kind,
                                    cin,
                                    cout,
                                } => {
                                    // controls above the tile select tiles
                                    let chi = cin >> k;
                                    if base & cout == *cout && ti & chi == chi {
                                        op_u1::<S>(&mut v, k, *t, m, *kind, cin & lowmask);
                                    }
                                }
                                other => apply_op::<S>(&mut v, k, other, base, sc),
                            }
                        }
                    }
                }
                Seg::Full(op) => apply_op::<S>(&mut buf.view(), p.l, op, base, sc),
            }
        }
        return;
    }
    let mut v = buf.view();
    if prof::on() {
        for op in &p.ops {
            let t0 = std::time::Instant::now();
            apply_op::<S>(&mut v, p.l, op, base, sc);
            prof::add(prof_kind(op), t0);
        }
        return;
    }
    for op in &p.ops {
        apply_op::<S>(&mut v, p.l, op, base, sc);
    }
}

/// [`run_ops_t`] for `f32`, compiled with AVX-512.
///
/// # Safety
/// The CPU must support AVX-512 F/DQ/VL/BW, AVX2 and FMA.
#[target_feature(enable = "avx512f,avx512dq,avx512vl,avx512bw,avx2,fma")]
#[inline(never)]
unsafe fn run_ops_f32(
    p: &Prepared<f32>,
    buf: &mut Buf<f32>,
    base: usize,
    sc: &mut DiagScratch<f32>,
) {
    run_ops_t::<f32>(p, buf, base, sc)
}

/// [`run_ops_t`] for `f64`, compiled with AVX-512.
///
/// # Safety
/// As [`run_ops_f32`].
#[target_feature(enable = "avx512f,avx512dq,avx512vl,avx512bw,avx2,fma")]
#[inline(never)]
unsafe fn run_ops_f64(
    p: &Prepared<f64>,
    buf: &mut Buf<f64>,
    base: usize,
    sc: &mut DiagScratch<f64>,
) {
    run_ops_t::<f64>(p, buf, base, sc)
}

/// Runs a prepared stage's ops on one block with the AVX-512 kernels.
///
/// # Safety
/// The CPU must support AVX-512 F/DQ/VL/BW, AVX2 and FMA
/// (`super::avx512_available()`).
#[inline]
pub(super) unsafe fn run_ops<T: Real>(
    p: &Prepared<T>,
    buf: &mut Buf<T>,
    base: usize,
    sc: &mut DiagScratch<T>,
) {
    use std::any::TypeId;
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        // SAFETY: `T` is `f32`, so these are the same types.
        run_ops_f32(
            &*(p as *const Prepared<T>).cast::<Prepared<f32>>(),
            &mut *(buf as *mut Buf<T>).cast::<Buf<f32>>(),
            base,
            &mut *(sc as *mut DiagScratch<T>).cast::<DiagScratch<f32>>(),
        )
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        // SAFETY: `T` is `f64`.
        run_ops_f64(
            &*(p as *const Prepared<T>).cast::<Prepared<f64>>(),
            &mut *(buf as *mut Buf<T>).cast::<Buf<f64>>(),
            base,
            &mut *(sc as *mut DiagScratch<T>).cast::<DiagScratch<f64>>(),
        )
    } else {
        super::run_ops_impl::<T, false>(p, buf, base, sc)
    }
}

/// Interleaved `Complex` amplitudes -> split re/im buffer slices, two
/// vectors per step (`vpermt2ps` deinterleave), scalar tail.
#[inline(always)]
unsafe fn load_t<S: Simd>(re: &mut [S], im: &mut [S], run: &[num_complex::Complex<S>]) {
    let n = run.len().min(re.len()).min(im.len());
    let lanes = nl::<S>();
    let even = S::idx(|i| 2 * i);
    let odd = S::idx(|i| 2 * i + 1);
    let src = run.as_ptr().cast::<S>();
    let (pr, pi) = (re.as_mut_ptr(), im.as_mut_ptr());
    let mut k = 0;
    while k + lanes <= n {
        let a = S::ld(src.add(2 * k));
        let b = S::ld(src.add(2 * k + lanes));
        S::st(pr.add(k), S::perm2(a, even, b));
        S::st(pi.add(k), S::perm2(a, odd, b));
        k += lanes;
    }
    for j in k..n {
        re[j] = run[j].re;
        im[j] = run[j].im;
    }
}

/// Split re/im buffer slices -> interleaved `Complex` amplitudes.
#[inline(always)]
unsafe fn store_t<S: Simd>(re: &[S], im: &[S], run: &mut [num_complex::Complex<S>]) {
    let n = run.len().min(re.len()).min(im.len());
    let lanes = nl::<S>();
    let half = lanes / 2;
    let lo = S::idx(|p| if p % 2 == 0 { p / 2 } else { lanes + p / 2 });
    let hi = S::idx(|p| {
        if p % 2 == 0 {
            half + p / 2
        } else {
            lanes + half + p / 2
        }
    });
    let dst = run.as_mut_ptr().cast::<S>();
    let (pr, pi) = (re.as_ptr(), im.as_ptr());
    let mut k = 0;
    while k + lanes <= n {
        let r = S::ld(pr.add(k));
        let i = S::ld(pi.add(k));
        S::st(dst.add(2 * k), S::perm2(r, lo, i));
        S::st(dst.add(2 * k + lanes), S::perm2(r, hi, i));
        k += lanes;
    }
    for j in k..n {
        run[j] = num_complex::Complex::new(re[j], im[j]);
    }
}

type C32 = num_complex::Complex<f32>;
type C64 = num_complex::Complex<f64>;

#[target_feature(enable = "avx512f,avx512dq,avx512vl,avx512bw,avx2,fma")]
#[inline(never)]
unsafe fn load_f32(re: &mut [f32], im: &mut [f32], run: &[C32]) {
    load_t::<f32>(re, im, run)
}

#[target_feature(enable = "avx512f,avx512dq,avx512vl,avx512bw,avx2,fma")]
#[inline(never)]
unsafe fn load_f64(re: &mut [f64], im: &mut [f64], run: &[C64]) {
    load_t::<f64>(re, im, run)
}

#[target_feature(enable = "avx512f,avx512dq,avx512vl,avx512bw,avx2,fma")]
#[inline(never)]
unsafe fn store_f32(re: &[f32], im: &[f32], run: &mut [C32]) {
    store_t::<f32>(re, im, run)
}

#[target_feature(enable = "avx512f,avx512dq,avx512vl,avx512bw,avx2,fma")]
#[inline(never)]
unsafe fn store_f64(re: &[f64], im: &[f64], run: &mut [C64]) {
    store_t::<f64>(re, im, run)
}

/// `buf[off..off + run.len()] = run` (split layout), AVX-512 build.
///
/// # Safety
/// As [`run_ops`]: the CPU must support AVX-512 F/DQ/VL/BW, AVX2 and FMA.
#[inline]
pub(super) unsafe fn load_run<T: Real>(
    buf: &mut Buf<T>,
    off: usize,
    run: &[num_complex::Complex<T>],
) {
    use std::any::TypeId;
    let end = off + run.len();
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        // SAFETY: `T` is `f32` (`Complex<T>` and `Complex<f32>` are the same type).
        let b = &mut *(buf as *mut Buf<T>).cast::<Buf<f32>>();
        let r = &*(run as *const [num_complex::Complex<T>] as *const [num_complex::Complex<f32>]);
        load_f32(&mut b.re[off..end], &mut b.im[off..end], r)
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        // SAFETY: `T` is `f64`.
        let b = &mut *(buf as *mut Buf<T>).cast::<Buf<f64>>();
        let r = &*(run as *const [num_complex::Complex<T>] as *const [num_complex::Complex<f64>]);
        load_f64(&mut b.re[off..end], &mut b.im[off..end], r)
    } else {
        super::load_run(buf, off, run)
    }
}

/// `run = buf[off..off + run.len()]` (interleaved), AVX-512 build.
///
/// # Safety
/// As [`run_ops`].
#[inline]
pub(super) unsafe fn store_run<T: Real>(
    buf: &Buf<T>,
    off: usize,
    run: &mut [num_complex::Complex<T>],
) {
    use std::any::TypeId;
    let end = off + run.len();
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        // SAFETY: `T` is `f32`.
        let b = &*(buf as *const Buf<T>).cast::<Buf<f32>>();
        let r = &mut *(run as *mut [num_complex::Complex<T>] as *mut [num_complex::Complex<f32>]);
        store_f32(&b.re[off..end], &b.im[off..end], r)
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        // SAFETY: `T` is `f64`.
        let b = &*(buf as *const Buf<T>).cast::<Buf<f64>>();
        let r = &mut *(run as *mut [num_complex::Complex<T>] as *mut [num_complex::Complex<f64>]);
        store_f64(&b.re[off..end], &b.im[off..end], r)
    } else {
        super::store_run(buf, off, run)
    }
}
