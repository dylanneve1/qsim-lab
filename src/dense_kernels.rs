//! SoA (split re/im) kernels for dense `2^K x 2^K` unitaries on a cache
//! block, written so that LLVM vectorises them on any target (SSE2 baseline,
//! AVX2 when the caller is compiled with it).
//!
//! # Lane transposition
//!
//! The kernel works on *tiles* of `2^K` vectors of [`LANES`] contiguous
//! amplitudes. For every target qubit `t >= 3` the pair partner lives in a
//! different vector, so the arithmetic is purely vertical. A target `t < 3`
//! would sit inside a vector (lane bit `t`); instead of gathering, the tile
//! first *exchanges* that lane bit with a spare higher index bit (a
//! `2 x LANES` butterfly of constant shuffles, identical for every kernel),
//! runs the vertical matrix-vector product, and exchanges back. The matrix
//! product code is therefore shared by all target positions: 1 target-set
//! independent instantiation per `(K, T, FMA)`.

use crate::statevector::Real;

/// Amplitudes per vector in a tile (3 index bits).
pub(crate) const LANES: usize = 8;
type V<T> = [T; LANES];

/// Inserts a zero bit at each position of `fixed` (ascending) into `x`.
#[inline(always)]
fn insert_zeros(mut x: usize, mut fixed: usize) -> usize {
    while fixed != 0 {
        let f = fixed.trailing_zeros();
        fixed &= fixed - 1;
        let lowm = (1usize << f) - 1;
        x = ((x & !lowm) << 1) | (x & lowm);
    }
    x
}

/// `acc + a * x`, fused when `FMA` (callers set it only when the code is
/// compiled with the `fma` target feature, otherwise `mul_add` would be a
/// libm call).
#[inline(always)]
fn mac<T: Real, const FMA: bool>(acc: T, a: T, x: T) -> T {
    if FMA {
        a.mul_add(x, acc)
    } else {
        acc + a * x
    }
}

/// Exchanges lane bit `J` of the amplitudes with the "vector bit" that
/// distinguishes `a` (bit = 0) from `b` (bit = 1).
#[inline(always)]
fn exchange<T: Real, const J: usize>(a: &mut V<T>, b: &mut V<T>) {
    let (x, y) = (*a, *b);
    for l in 0..LANES {
        if (l >> J) & 1 == 0 {
            a[l] = x[l];
            b[l] = x[l | (1 << J)];
        } else {
            a[l] = y[l ^ (1 << J)];
            b[l] = y[l];
        }
    }
}

/// Applies [`exchange`] for lane bit `J` against vector-index bit `I` to
/// all `D` vectors of a tile.
#[inline(always)]
fn exchange_all<T: Real, const J: usize, const I: usize, const D: usize>(x: &mut [V<T>; D]) {
    for v in 0..D {
        if (v >> I) & 1 == 0 {
            let (lo, hi) = x.split_at_mut(v | (1 << I));
            exchange::<T, J>(&mut lo[v], &mut hi[0]);
        }
    }
}

/// Position of the `i`-th set bit of `pm`.
const fn nth_bit(pm: usize, i: usize) -> usize {
    let mut m = pm;
    let mut k = 0;
    while k < i {
        m &= m - 1;
        k += 1;
    }
    m.trailing_zeros() as usize
}

/// `exchange_all` for lane bit `j` (a monomorphisation-time constant, so the
/// `match` folds away) against vector bit `I`.
#[inline(always)]
fn exchange_bit<T: Real, const I: usize, const D: usize>(x: &mut [V<T>; D], j: usize) {
    match j {
        0 => exchange_all::<T, 0, I, D>(x),
        1 => exchange_all::<T, 1, I, D>(x),
        _ => exchange_all::<T, 2, I, D>(x),
    }
}

/// Exchanges every in-lane target (the set bits of `PM`, ascending) of the
/// tile with its partner vector bit (`i`-th set bit <-> vector bit `i`).
#[inline(always)]
fn exchange_lows<T: Real, const PM: usize, const D: usize>(x: &mut [V<T>; D]) {
    let a = PM.count_ones();
    if a >= 1 {
        exchange_bit::<T, 0, D>(x, nth_bit(PM, 0));
    }
    if a >= 2 {
        exchange_bit::<T, 1, D>(x, nth_bit(PM, 1));
    }
    if a >= 3 {
        exchange_bit::<T, 2, D>(x, nth_bit(PM, 2));
    }
}

/// `y = M x` for `D` vectors of complex lanes; `mr`/`mi` are row-major.
#[inline(always)]
fn matvec<T: Real, const FMA: bool, const D: usize>(
    mr: &[T],
    mi: &[T],
    xr: &[V<T>; D],
    xi: &[V<T>; D],
) -> ([V<T>; D], [V<T>; D]) {
    let mut yr = [[T::zero(); LANES]; D];
    let mut yi = [[T::zero(); LANES]; D];
    for r in 0..D {
        let mut ar = [T::zero(); LANES];
        let mut ai = [T::zero(); LANES];
        for c in 0..D {
            let (a, b) = (mr[r * D + c], mi[r * D + c]);
            let nb = -b;
            for m in 0..LANES {
                ar[m] = mac::<T, FMA>(mac::<T, FMA>(ar[m], a, xr[c][m]), nb, xi[c][m]);
                ai[m] = mac::<T, FMA>(mac::<T, FMA>(ai[m], a, xi[c][m]), b, xr[c][m]);
            }
        }
        yr[r] = ar;
        yi[r] = ai;
    }
    (yr, yi)
}

/// Per-call constants of a tile kernel (computed once, out of line).
struct TilePlan<const D: usize> {
    /// Buffer offset of tile vector `v` relative to the tile base.
    off: [usize; D],
    /// Buffer bits that vary inside a tile (3 lane bits + the vector bits).
    mask: usize,
}

/// Vector-index bit `i` <-> buffer bit `lst[i]`: first the partner bits for
/// the in-lane targets (lowest free bits >= 3), then the high targets.
#[inline(never)]
fn tile_plan<const K: usize, const D: usize>(t: [usize; K], a: usize) -> TilePlan<D> {
    let tmask: usize = t.iter().map(|&q| 1usize << q).sum();
    let mut lst = [0usize; K];
    let mut p = 3;
    for slot in lst.iter_mut().take(a) {
        while (tmask >> p) & 1 == 1 {
            p += 1;
        }
        *slot = p;
        p += 1;
    }
    lst[a..K].copy_from_slice(&t[a..K]);
    let mut off = [0usize; D];
    for (v, o) in off.iter_mut().enumerate() {
        for (i, &b) in lst.iter().enumerate() {
            if (v >> i) & 1 == 1 {
                *o |= 1 << b;
            }
        }
    }
    let mask = lst.iter().map(|&q| 1usize << q).sum::<usize>() | 7;
    TilePlan { off, mask }
}

/// Applies the `D x D` matrix (`D = 2^K`, target-local index bit `j` =
/// buffer bit `t[j]`, `t` ascending) to the SoA buffer of `2^l` amplitudes.
/// `PM` is the mask of targets below bit 3. Requires `l >= 3 + K`.
#[inline(always)]
fn apply_tile<T: Real, const FMA: bool, const K: usize, const D: usize, const PM: usize>(
    re: &mut [T],
    im: &mut [T],
    l: usize,
    t: [usize; K],
    mr: &[T],
    mi: &[T],
) {
    debug_assert!(D == 1 << K && l >= 3 + K);
    debug_assert_eq!(PM.count_ones() as usize, t.iter().filter(|&&q| q < 3).count());
    let plan = tile_plan::<K, D>(t, PM.count_ones() as usize);
    let (re, im) = (&mut re[..1 << l], &mut im[..1 << l]);
    let (mr, mi) = (&mr[..D * D], &mi[..D * D]);
    let (off, mask) = (plan.off, plan.mask);
    let count = 1usize << (l - 3 - K);
    let mut base = 0usize;
    for _ in 0..count {
        let mut xr = [[T::zero(); LANES]; D];
        let mut xi = [[T::zero(); LANES]; D];
        for v in 0..D {
            let o = base + off[v];
            xr[v] = re[o..o + LANES].try_into().unwrap();
            xi[v] = im[o..o + LANES].try_into().unwrap();
        }
        exchange_lows::<T, PM, D>(&mut xr);
        exchange_lows::<T, PM, D>(&mut xi);
        let (mut yr, mut yi) = matvec::<T, FMA, D>(mr, mi, &xr, &xi);
        exchange_lows::<T, PM, D>(&mut yr);
        exchange_lows::<T, PM, D>(&mut yi);
        for v in 0..D {
            let o = base + off[v];
            re[o..o + LANES].copy_from_slice(&yr[v]);
            im[o..o + LANES].copy_from_slice(&yi[v]);
        }
        // next index with all bits of `mask` clear
        base = ((base | mask) + 1) & !mask;
    }
}

/// Plain gather/scatter version for blocks too small for [`apply_tile`]
/// (`l < 3 + K`).
#[inline(always)]
pub(crate) fn apply_scalar<T: Real, const K: usize, const D: usize>(
    re: &mut [T],
    im: &mut [T],
    l: usize,
    t: [usize; K],
    mr: &[T],
    mi: &[T],
) {
    let tmask: usize = t.iter().map(|&q| 1usize << q).sum();
    let mut off = [0usize; D];
    for (v, o) in off.iter_mut().enumerate() {
        for (i, &b) in t.iter().enumerate() {
            if (v >> i) & 1 == 1 {
                *o |= 1 << b;
            }
        }
    }
    for k in 0..1usize << (l - K) {
        let base = insert_zeros(k, tmask);
        let mut xr = [T::zero(); D];
        let mut xi = [T::zero(); D];
        for v in 0..D {
            xr[v] = re[base + off[v]];
            xi[v] = im[base + off[v]];
        }
        for r in 0..D {
            let (mut sr, mut si) = (T::zero(), T::zero());
            for c in 0..D {
                let (a, b) = (mr[r * D + c], mi[r * D + c]);
                sr = sr + a * xr[c] - b * xi[c];
                si = si + a * xi[c] + b * xr[c];
            }
            re[base + off[r]] = sr;
            im[base + off[r]] = si;
        }
    }
}

/// Dispatches a dense op: tile kernel when the block is big enough.
#[inline(always)]
pub(crate) fn apply_dense<T: Real, const FMA: bool, const K: usize, const D: usize>(
    re: &mut [T],
    im: &mut [T],
    l: usize,
    t: [usize; K],
    mr: &[T],
    mi: &[T],
) {
    if l >= 3 + K {
        let pm: usize = t.iter().filter(|&&q| q < 3).map(|&q| 1usize << q).sum();
        macro_rules! go {
            ($pm:literal) => {
                apply_tile::<T, FMA, K, D, $pm>(re, im, l, t, mr, mi)
            };
        }
        match pm {
            0 => go!(0),
            1 => go!(1),
            2 => go!(2),
            3 if K >= 2 => go!(3),
            4 => go!(4),
            5 if K >= 2 => go!(5),
            6 if K >= 2 => go!(6),
            7 if K >= 3 => go!(7),
            _ => unreachable!("target mask {pm:#b} for K = {K}"),
        }
    } else {
        apply_scalar::<T, K, D>(re, im, l, t, mr, mi);
    }
}
