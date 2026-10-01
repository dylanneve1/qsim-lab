//! Cache-blocked, fused execution of gate sequences on a state vector.
//!
//! The plain [`StateVector::apply_gate`] path makes one pass over all `2^n`
//! amplitudes per gate, so a deep circuit on 24+ qubits is limited by DRAM
//! bandwidth. This module trades those passes for passes over a block that
//! fits in the L2 cache:
//!
//! 1. Gates are lowered to a tiny IR ([`KOp`]): a (multi-)controlled 2x2
//!    unitary, a diagonal phase term `f` on indices with `(i & mask) == pat`,
//!    or a SWAP. Runs of single-qubit gates on the same qubit are multiplied
//!    together (in f64) first.
//! 2. The gate list is cut into *stages*. Each stage has an *inner* set of
//!    `L` qubits: the low qubits `0..b` plus up to `slots` higher qubits.
//!    Only qubits on which a gate acts non-diagonally as a target (or SWAP)
//!    must be inner; controls and diagonal gates may involve any qubit,
//!    because for a fixed assignment of the outer qubits a control is either
//!    on or off and a diagonal gate is a diagonal gate on the inner qubits.
//! 3. A stage is executed chunk by chunk: the `2^L` amplitudes that share
//!    one assignment of the outer qubits are gathered into a buffer (they
//!    form `2^(L-b')` contiguous runs), all of the stage's gates are applied
//!    there, and the buffer is written back. DRAM is streamed once per
//!    stage instead of once per gate.
//! 4. Consecutive diagonal terms in a stage are applied together: they are
//!    grouped by a "pivot" condition and each group is one multiplication
//!    pass by a product of per-bit factor tables (see [`DiagBlock`]).

use crate::circuit::{check_gate, Circuit, Op, SimError};
use crate::gate::{mat2_mul, Gate, Mat2};
use crate::statevector::{Real, StateVector};
use num_complex::{Complex, Complex64};
use rayon::prelude::*;

const C0: Complex64 = Complex64::new(0.0, 0.0);
const C1: Complex64 = Complex64::new(1.0, 0.0);
const XMAT: Mat2 = [[C0, C1], [C1, C0]];

/// Executor IR: one operation on physical qubits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KOp {
    /// 2x2 unitary `m` on qubit `q`, applied where all qubits in the bit mask
    /// `ctrl` are 1.
    U1 { q: usize, m: Mat2, ctrl: usize },
    /// Multiplies every amplitude whose index satisfies `(i & mask) == pat`
    /// by `f`.
    Phase {
        mask: usize,
        pat: usize,
        f: Complex64,
    },
    /// Exchanges qubits `a` and `b`.
    Swap { a: usize, b: usize },
}

impl KOp {
    /// Bit mask of every qubit the op involves.
    fn touches(&self) -> usize {
        match *self {
            KOp::U1 { q, ctrl, .. } => (1 << q) | ctrl,
            KOp::Phase { mask, .. } => mask,
            KOp::Swap { a, b } => (1 << a) | (1 << b),
        }
    }

    /// Bit mask of the qubits that must be inner (in the cache block).
    fn needs_inner(&self) -> usize {
        match *self {
            KOp::U1 { q, .. } => 1 << q,
            KOp::Phase { .. } => 0,
            KOp::Swap { a, b } => (1 << a) | (1 << b),
        }
    }
}

fn push_diag1(out: &mut Vec<KOp>, q: usize, d0: Complex64, d1: Complex64) {
    if d0 != C1 {
        out.push(KOp::Phase {
            mask: 1 << q,
            pat: 0,
            f: d0,
        });
    }
    if d1 != C1 {
        out.push(KOp::Phase {
            mask: 1 << q,
            pat: 1 << q,
            f: d1,
        });
    }
}

/// Lowers a gate to executor ops (appended to `out`).
pub fn lower_gate(g: &Gate, out: &mut Vec<KOp>) {
    match *g {
        Gate::Cnot(c, t) => out.push(KOp::U1 {
            q: t,
            m: XMAT,
            ctrl: 1 << c,
        }),
        Gate::Ccx(a, b, t) => out.push(KOp::U1 {
            q: t,
            m: XMAT,
            ctrl: (1 << a) | (1 << b),
        }),
        Gate::Cz(a, b) => {
            let m = (1 << a) | (1 << b);
            out.push(KOp::Phase {
                mask: m,
                pat: m,
                f: Complex64::new(-1.0, 0.0),
            })
        }
        Gate::CPhase(a, b, th) => {
            let m = (1 << a) | (1 << b);
            out.push(KOp::Phase {
                mask: m,
                pat: m,
                f: Complex64::from_polar(1.0, th),
            })
        }
        Gate::Swap(a, b) => out.push(KOp::Swap { a, b }),
        ref g1 => {
            let q = g1.qubits()[0];
            if let Some((d0, d1)) = g1.diagonal_1q() {
                push_diag1(out, q, d0, d1);
            } else {
                out.push(KOp::U1 {
                    q,
                    m: g1.matrix_1q().expect("single-qubit gate"),
                    ctrl: 0,
                });
            }
        }
    }
}

/// Lowers a gate list.
pub fn lower_gates<'a>(gates: impl IntoIterator<Item = &'a Gate>) -> Vec<KOp> {
    let mut out = Vec::new();
    for g in gates {
        lower_gate(g, &mut out);
    }
    out
}

fn emit_1q(out: &mut Vec<KOp>, q: usize, m: Mat2) {
    if m[0][1] == C0 && m[1][0] == C0 {
        push_diag1(out, q, m[0][0], m[1][1]);
    } else {
        out.push(KOp::U1 { q, m, ctrl: 0 });
    }
}

/// Multiplies together runs of uncontrolled single-qubit ops on the same
/// qubit (products taken in f64). The result is the same unitary up to
/// rounding.
pub fn fuse_1q(ops: &[KOp], n: usize) -> Vec<KOp> {
    let mut pending: Vec<Option<Mat2>> = vec![None; n];
    let mut out = Vec::with_capacity(ops.len());
    let accumulate = |pending: &mut Vec<Option<Mat2>>, q: usize, m: Mat2| {
        pending[q] = Some(match pending[q] {
            Some(p) => mat2_mul(&m, &p),
            None => m,
        });
    };
    for op in ops {
        match *op {
            KOp::U1 { q, m, ctrl: 0 } => accumulate(&mut pending, q, m),
            KOp::Phase { mask, pat, f } if mask.count_ones() == 1 => {
                let q = mask.trailing_zeros() as usize;
                let m = if pat == 0 {
                    [[f, C0], [C0, C1]]
                } else {
                    [[C1, C0], [C0, f]]
                };
                accumulate(&mut pending, q, m);
            }
            _ => {
                let mut t = op.touches();
                while t != 0 {
                    let q = t.trailing_zeros() as usize;
                    t &= t - 1;
                    if let Some(m) = pending[q].take() {
                        emit_1q(&mut out, q, m);
                    }
                }
                out.push(*op);
            }
        }
    }
    for (q, p) in pending.into_iter().enumerate() {
        if let Some(m) = p {
            emit_1q(&mut out, q, m);
        }
    }
    out
}

/// Tuning knobs of the blocked executor.
#[derive(Clone, Debug)]
pub struct BlockConfig {
    /// Size of the per-thread cache block in bytes (rounded down to a power
    /// of two amplitudes).
    pub block_bytes: usize,
    /// Maximum number of high (non-contiguous) qubits gathered per stage.
    pub slots: usize,
    /// Multiply runs of single-qubit gates together first.
    pub fuse_1q: bool,
    /// Registers with at most this many qubits run as one block on the
    /// calling thread.
    pub small_n: usize,
}

impl Default for BlockConfig {
    fn default() -> Self {
        BlockConfig {
            block_bytes: 256 << 10,
            slots: 6,
            fuse_1q: true,
            small_n: 12,
        }
    }
}

impl BlockConfig {
    /// Block size `L` (in qubits) for an `n`-qubit register of `elem`-byte
    /// amplitudes.
    fn block_bits(&self, n: usize, elem: usize) -> usize {
        let l = (self.block_bytes / elem).max(2).ilog2() as usize;
        if n <= self.small_n || n <= l.min(4) {
            n
        } else {
            // keep at least four chunks so every thread gets work
            l.min(n - 2)
        }
    }
}

/// One stage of a plan: the inner qubits (ascending; buffer bit `j` is
/// physical qubit `inner[j]`) and the ops applied while they are cached.
#[derive(Clone, Debug)]
pub struct Stage {
    pub inner: Vec<usize>,
    pub ops: Vec<KOp>,
}

/// Splits `ops` into stages for an `n`-qubit register with blocks of `l`
/// qubits of which at most `slots` are outside the contiguous low range.
pub fn plan_stages(ops: &[KOp], n: usize, l: usize, slots: usize) -> Vec<Stage> {
    if l >= n {
        return vec![Stage {
            inner: (0..n).collect(),
            ops: ops.to_vec(),
        }];
    }
    let slots = slots.min(l);
    let b = l - slots;
    let low = (1usize << b) - 1;
    let finish = |hi: usize, ops: Vec<KOp>| {
        let mut mask = low | hi;
        let mut q = b;
        while (mask.count_ones() as usize) < l {
            mask |= 1 << q;
            q += 1;
        }
        Stage {
            inner: (0..n).filter(|q| mask >> q & 1 == 1).collect(),
            ops,
        }
    };
    let mut stages = Vec::new();
    let mut hi = 0usize;
    let mut cur: Vec<KOp> = Vec::new();
    for op in ops {
        let need = op.needs_inner() & !low;
        let merged = hi | need;
        if merged.count_ones() as usize > slots {
            stages.push(finish(hi, std::mem::take(&mut cur)));
            hi = need;
        } else {
            hi = merged;
        }
        cur.push(*op);
    }
    if !cur.is_empty() {
        stages.push(finish(hi, cur));
    }
    stages
}

// ----- prepared (typed, buffer-relative) stages ---------------------------

/// A factor applied to one group of a diagonal block: `f` when the chunk's
/// outer bits match (`omask`, `opat`) and, if `bit` is set, when buffer bit
/// `bit.0` equals `bit.1`.
#[derive(Clone, Debug)]
struct LinTerm {
    bit: Option<(usize, bool)>,
    omask: usize,
    opat: usize,
    f: Complex64,
}

/// Terms applied in one pass over the buffer indices with
/// `(j & cmask) == cpat`. Each term depends on at most one buffer bit, so
/// the product over terms factorises into a table over the low 8 bits and a
/// table over the rest.
#[derive(Clone, Debug)]
struct DiagGroup {
    cmask: usize,
    cpat: usize,
    terms: Vec<LinTerm>,
}

/// A run of consecutive diagonal terms.
#[derive(Clone, Debug)]
struct DiagBlock {
    groups: Vec<DiagGroup>,
}

#[derive(Clone, Debug)]
enum LOp<T: Real> {
    U1 {
        t: usize,
        m: [T; 8],
        is_x: bool,
        cin: usize,
        cout: usize,
    },
    Swap {
        a: usize,
        b: usize,
    },
    Diag(DiagBlock),
}

struct Prepared<T: Real> {
    l: usize,
    inner_mask: usize,
    ops: Vec<LOp<T>>,
}

fn cvt<T: Real>(z: Complex64) -> Complex<T> {
    Complex::new(T::from_f64(z.re), T::from_f64(z.im))
}

/// Phys mask -> (buffer mask, outer phys mask).
fn split_mask(mask: usize, pos: &[Option<usize>]) -> (usize, usize) {
    let (mut inn, mut out) = (0usize, 0usize);
    let mut m = mask;
    while m != 0 {
        let q = m.trailing_zeros() as usize;
        m &= m - 1;
        match pos[q] {
            Some(j) => inn |= 1 << j,
            None => out |= 1 << q,
        }
    }
    (inn, out)
}

/// Remaps the bits of `pat` that lie in `mask` (phys) to buffer positions.
fn map_pat(mask: usize, pat: usize, pos: &[Option<usize>]) -> (usize, usize) {
    split_mask(pat & mask, pos)
}

fn build_diag_block(terms: &[KOp], pos: &[Option<usize>]) -> DiagBlock {
    struct T2 {
        imask: usize,
        ipat: usize,
        omask: usize,
        opat: usize,
        f: Complex64,
    }
    let mut multi = Vec::new(); // >= 2 inner bits
    let mut single = Vec::new(); // <= 1 inner bit
    for op in terms {
        let KOp::Phase { mask, pat, f } = *op else {
            unreachable!()
        };
        let (imask, omask) = split_mask(mask, pos);
        let (ipat, opat) = map_pat(mask, pat, pos);
        let t = T2 {
            imask,
            ipat,
            omask,
            opat,
            f,
        };
        if imask.count_ones() >= 2 {
            multi.push(t);
        } else {
            single.push(t);
        }
    }
    let mut groups: Vec<DiagGroup> = Vec::new();
    let group_of = |groups: &mut Vec<DiagGroup>, cmask: usize, cpat: usize| -> usize {
        if let Some(i) = groups
            .iter()
            .position(|g| g.cmask == cmask && g.cpat == cpat)
        {
            i
        } else {
            groups.push(DiagGroup {
                cmask,
                cpat,
                terms: Vec::new(),
            });
            groups.len() - 1
        }
    };
    // Terms on >= 2 inner bits: condition on all but one bit. For 2-bit
    // terms pick the (bit, value) pivot shared by the most terms, greedily.
    let mut rest: Vec<T2> = Vec::new();
    for t in multi {
        if t.imask.count_ones() > 2 {
            let lin = t.imask.trailing_zeros() as usize;
            let cmask = t.imask & !(1 << lin);
            let g = group_of(&mut groups, cmask, t.ipat & cmask);
            groups[g].terms.push(LinTerm {
                bit: Some((lin, t.ipat >> lin & 1 == 1)),
                omask: t.omask,
                opat: t.opat,
                f: t.f,
            });
        } else {
            rest.push(t);
        }
    }
    while !rest.is_empty() {
        let mut counts: std::collections::HashMap<(usize, bool), usize> = Default::default();
        for t in &rest {
            let mut m = t.imask;
            while m != 0 {
                let j = m.trailing_zeros() as usize;
                m &= m - 1;
                *counts.entry((j, t.ipat >> j & 1 == 1)).or_default() += 1;
            }
        }
        let (&(pj, pv), _) = counts
            .iter()
            .max_by_key(|(k, c)| (**c, std::cmp::Reverse(**k)))
            .expect("non-empty");
        let g = group_of(&mut groups, 1 << pj, (pv as usize) << pj);
        let mut keep = Vec::new();
        for t in rest {
            if t.imask >> pj & 1 == 1 && (t.ipat >> pj & 1 == 1) == pv {
                let other = (t.imask & !(1 << pj)).trailing_zeros() as usize;
                groups[g].terms.push(LinTerm {
                    bit: Some((other, t.ipat >> other & 1 == 1)),
                    omask: t.omask,
                    opat: t.opat,
                    f: t.f,
                });
            } else {
                keep.push(t);
            }
        }
        rest = keep;
    }
    // Terms on <= 1 inner bit: fold a 1-bit term into a pivot group with the
    // same condition if there is one, otherwise into the unconditioned group.
    for t in single {
        let (cmask, cpat, bit) = if t.imask == 0 {
            (0, 0, None)
        } else if let Some(i) = groups
            .iter()
            .position(|g| g.cmask == t.imask && g.cpat == t.ipat)
        {
            groups[i].terms.push(LinTerm {
                bit: None,
                omask: t.omask,
                opat: t.opat,
                f: t.f,
            });
            continue;
        } else {
            let j = t.imask.trailing_zeros() as usize;
            (0, 0, Some((j, t.ipat != 0)))
        };
        let g = group_of(&mut groups, cmask, cpat);
        groups[g].terms.push(LinTerm {
            bit,
            omask: t.omask,
            opat: t.opat,
            f: t.f,
        });
    }
    DiagBlock { groups }
}

fn prepare<T: Real>(st: &Stage, n: usize) -> Prepared<T> {
    let mut pos = vec![None; n];
    let mut inner_mask = 0usize;
    for (j, &q) in st.inner.iter().enumerate() {
        pos[q] = Some(j);
        inner_mask |= 1 << q;
    }
    let mut ops = Vec::new();
    let mut i = 0;
    while i < st.ops.len() {
        match st.ops[i] {
            KOp::U1 { q, m, ctrl } => {
                let (cin, cout) = split_mask(ctrl, &pos);
                ops.push(LOp::U1 {
                    t: pos[q].expect("target is inner"),
                    m: [
                        T::from_f64(m[0][0].re),
                        T::from_f64(m[0][1].re),
                        T::from_f64(m[1][0].re),
                        T::from_f64(m[1][1].re),
                        T::from_f64(m[0][0].im),
                        T::from_f64(m[0][1].im),
                        T::from_f64(m[1][0].im),
                        T::from_f64(m[1][1].im),
                    ],
                    is_x: m == XMAT,
                    cin,
                    cout,
                });
                i += 1;
            }
            KOp::Swap { a, b } => {
                let (a, b) = (pos[a].expect("inner"), pos[b].expect("inner"));
                ops.push(LOp::Swap {
                    a: a.min(b),
                    b: a.max(b),
                });
                i += 1;
            }
            KOp::Phase { .. } => {
                let start = i;
                while i < st.ops.len() && matches!(st.ops[i], KOp::Phase { .. }) {
                    i += 1;
                }
                ops.push(LOp::Diag(build_diag_block(&st.ops[start..i], &pos)));
            }
        }
    }
    Prepared {
        l: st.inner.len(),
        inner_mask,
        ops,
    }
}

// ----- buffer kernels -----------------------------------------------------
//
// Inside a block the amplitudes are kept as separate real and imaginary
// arrays (structure of arrays): with interleaved `Complex<T>` LLVM does not
// vectorise the complex arithmetic, while the split layout vectorises
// cleanly (about 4x faster per amplitude in L2, see EXPERIMENTS-sv.md).

/// A block of amplitudes in split (SoA) layout.
struct Buf<T> {
    re: Vec<T>,
    im: Vec<T>,
}

/// Inserts a zero bit at each position of `fixed` (ascending) into `x`.
#[inline]
fn insert_zeros(mut x: usize, mut fixed: usize) -> usize {
    while fixed != 0 {
        let f = fixed.trailing_zeros();
        fixed &= fixed - 1;
        let lowm = (1usize << f) - 1;
        x = ((x & !lowm) << 1) | (x & lowm);
    }
    x
}

/// Calls `f(base, run)` for every maximal run of indices `j < 2^l` with
/// `(j & fixed) == 0`: `base..base+run` are those indices.
#[inline]
fn for_each_run(l: usize, fixed: usize, mut f: impl FnMut(usize, usize)) {
    let low = fixed.trailing_zeros().min(l as u32) as usize;
    let run = 1usize << low;
    let nfree = l - fixed.count_ones() as usize;
    let count = 1usize << (nfree - low);
    let upper = fixed >> low << low;
    for k in 0..count {
        f(insert_zeros(k << low, upper), run);
    }
}

/// `(a, b) <- (m0 a + m1 b, m2 a + m3 b)` elementwise; `m` holds the real
/// parts then the imaginary parts of the four entries.
#[inline(always)]
fn u1_kernel<T: Real>(ar: &mut [T], ai: &mut [T], br: &mut [T], bi: &mut [T], m: &[T; 8]) {
    let [m0r, m1r, m2r, m3r, m0i, m1i, m2i, m3i] = *m;
    let len = ar.len();
    let (ai, br, bi) = (&mut ai[..len], &mut br[..len], &mut bi[..len]);
    for k in 0..len {
        let (xr, xi, yr, yi) = (ar[k], ai[k], br[k], bi[k]);
        ar[k] = m0r * xr - m0i * xi + m1r * yr - m1i * yi;
        ai[k] = m0r * xi + m0i * xr + m1r * yi + m1i * yr;
        br[k] = m2r * xr - m2i * xi + m3r * yr - m3i * yi;
        bi[k] = m2r * xi + m2i * xr + m3r * yi + m3i * yr;
    }
}

/// Splits `v` into `v[lo..lo+run]` and `v[hi..hi+run]` (`lo + run <= hi`).
#[inline(always)]
fn two<T>(v: &mut [T], lo: usize, hi: usize, run: usize) -> (&mut [T], &mut [T]) {
    let (a, b) = v.split_at_mut(hi);
    (&mut a[lo..lo + run], &mut b[..run])
}

fn apply_u1<T: Real>(buf: &mut Buf<T>, l: usize, t: usize, m: &[T; 8], is_x: bool, cin: usize) {
    let s = 1usize << t;
    let Buf { re, im } = buf;
    if cin == 0 {
        for (cr, ci) in re.chunks_exact_mut(2 * s).zip(im.chunks_exact_mut(2 * s)) {
            let (ar, br) = cr.split_at_mut(s);
            let (ai, bi) = ci.split_at_mut(s);
            if is_x {
                ar.swap_with_slice(br);
                ai.swap_with_slice(bi);
            } else {
                u1_kernel(ar, ai, br, bi, m);
            }
        }
        return;
    }
    for_each_run(l, cin | s, |base, run| {
        let base = base | cin;
        let (ar, br) = two(re, base, base + s, run);
        let (ai, bi) = two(im, base, base + s, run);
        if is_x {
            ar.swap_with_slice(br);
            ai.swap_with_slice(bi);
        } else {
            u1_kernel(ar, ai, br, bi, m);
        }
    });
}

fn apply_swap<T: Real>(buf: &mut Buf<T>, l: usize, a: usize, b: usize) {
    let (sa, sb) = (1usize << a, 1usize << b);
    let Buf { re, im } = buf;
    for_each_run(l, sa | sb, |base, run| {
        let (x, y) = two(re, base + sa, base + sb, run);
        x.swap_with_slice(y);
        let (x, y) = two(im, base + sa, base + sb, run);
        x.swap_with_slice(y);
    });
}

const LO_BITS: usize = 8;

/// Scratch tables reused across diagonal passes.
#[derive(Default)]
struct DiagScratch<T> {
    lo: Vec<Complex64>,
    hi: Vec<Complex64>,
    lor: Vec<T>,
    loi: Vec<T>,
}

/// `a[k] *= (lr[k] + i li[k]) * h` on a run.
#[inline(always)]
fn diag_kernel<T: Real>(ar: &mut [T], ai: &mut [T], lr: &[T], li: &[T], h: Complex<T>) {
    let len = ar.len();
    let (ai, lr, li) = (&mut ai[..len], &lr[..len], &li[..len]);
    for k in 0..len {
        let fr = lr[k] * h.re - li[k] * h.im;
        let fi = lr[k] * h.im + li[k] * h.re;
        let (xr, xi) = (ar[k], ai[k]);
        ar[k] = xr * fr - xi * fi;
        ai[k] = xr * fi + xi * fr;
    }
}

/// `out[x] = init * prod_j e[j][bit j of x]` for `x < 2^e.len()`.
fn product_table(out: &mut Vec<Complex64>, e: &[[Complex64; 2]], init: Complex64) {
    out.clear();
    out.resize(1 << e.len(), init);
    for (j, ej) in e.iter().enumerate() {
        let w = 1 << j;
        for x in (0..w).rev() {
            let v = out[x];
            out[x + w] = v * ej[1];
            out[x] = v * ej[0];
        }
    }
}

fn apply_diag_group<T: Real>(
    buf: &mut Buf<T>,
    l: usize,
    g: &DiagGroup,
    base: usize,
    sc: &mut DiagScratch<T>,
) {
    // Collect per-bit factors for the active terms.
    let mut e = [[C1; 2]; 64];
    let mut s = C1;
    let mut any = false;
    for t in &g.terms {
        if base & t.omask != t.opat {
            continue;
        }
        any = true;
        match t.bit {
            None => s *= t.f,
            Some((j, v)) => e[j][v as usize] *= t.f,
        }
    }
    if !any {
        return;
    }
    let lb = l.min(LO_BITS);
    // Tables (built in f64): lo over bits 0..lb, hi over bits lb..l with
    // the scalar folded in.
    product_table(&mut sc.lo, &e[..lb], C1);
    product_table(&mut sc.hi, &e[lb..l], s);
    let (lo, hi) = (&sc.lo, &sc.hi);
    sc.lor.clear();
    sc.loi.clear();
    sc.lor.extend(lo.iter().map(|z| T::from_f64(z.re)));
    sc.loi.extend(lo.iter().map(|z| T::from_f64(z.im)));
    let lomask = (1usize << lb) - 1;
    let (cm_lo, cp_lo) = (g.cmask & lomask, g.cpat & lomask);
    let (cm_hi, cp_hi) = (g.cmask >> lb, g.cpat >> lb);
    let Buf { re, im } = buf;
    for (h, &hv) in hi.iter().enumerate() {
        if h & cm_hi != cp_hi {
            continue;
        }
        let hv: Complex<T> = cvt(hv);
        let rr = &mut re[h << lb..(h + 1) << lb];
        let ri = &mut im[h << lb..(h + 1) << lb];
        if cm_lo == 0 {
            diag_kernel(rr, ri, &sc.lor, &sc.loi, hv);
        } else {
            for_each_run(lb, cm_lo, |b0, run| {
                let b0 = b0 | cp_lo;
                diag_kernel(
                    &mut rr[b0..b0 + run],
                    &mut ri[b0..b0 + run],
                    &sc.lor[b0..b0 + run],
                    &sc.loi[b0..b0 + run],
                    hv,
                );
            });
        }
    }
}

fn run_ops<T: Real>(p: &Prepared<T>, buf: &mut Buf<T>, base: usize, sc: &mut DiagScratch<T>) {
    let l = p.l;
    for op in &p.ops {
        match op {
            LOp::U1 {
                t,
                m,
                is_x,
                cin,
                cout,
            } => {
                if base & cout == *cout {
                    apply_u1(buf, l, *t, m, *is_x, *cin);
                }
            }
            LOp::Swap { a, b } => apply_swap(buf, l, *a, *b),
            LOp::Diag(d) => {
                for g in &d.groups {
                    apply_diag_group(buf, l, g, base, sc);
                }
            }
        }
    }
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

/// Byte-wise lookup tables for `extract(x, mask)` (software pext).
struct Pext {
    tables: Vec<[u32; 256]>,
}

impl Pext {
    fn new(mask: usize, bits: usize) -> Self {
        let nbytes = bits.div_ceil(8).max(1);
        let tables = (0..nbytes)
            .map(|k| {
                let below = (mask & ((1usize << (8 * k)) - 1)).count_ones();
                let m = (mask >> (8 * k)) & 0xff;
                let mut t = [0u32; 256];
                for (v, slot) in t.iter_mut().enumerate() {
                    let mut out = 0u32;
                    let mut i = 0;
                    for bit in 0..8 {
                        if m >> bit & 1 == 1 {
                            if v >> bit & 1 == 1 {
                                out |= 1 << i;
                            }
                            i += 1;
                        }
                    }
                    *slot = out << below;
                }
                t
            })
            .collect();
        Pext { tables }
    }
    #[inline]
    fn get(&self, x: usize) -> usize {
        let mut out = 0u32;
        for (k, t) in self.tables.iter().enumerate() {
            out |= t[(x >> (8 * k)) & 0xff];
        }
        out as usize
    }
}

#[inline]
fn load_run<T: Real>(buf: &mut Buf<T>, off: usize, run: &[Complex<T>]) {
    let r = &mut buf.re[off..off + run.len()];
    let i = &mut buf.im[off..off + run.len()];
    for ((a, b), z) in r.iter_mut().zip(i.iter_mut()).zip(run) {
        *a = z.re;
        *b = z.im;
    }
}

#[inline]
fn store_run<T: Real>(buf: &Buf<T>, off: usize, run: &mut [Complex<T>]) {
    let r = &buf.re[off..off + run.len()];
    let i = &buf.im[off..off + run.len()];
    for ((a, b), z) in r.iter().zip(i.iter()).zip(run.iter_mut()) {
        *z = Complex::new(*a, *b);
    }
}

fn new_buf<T: Real>(l: usize) -> Buf<T> {
    Buf {
        re: vec![T::zero(); 1 << l],
        im: vec![T::zero(); 1 << l],
    }
}

fn run_stage<T: Real>(amps: &mut [Complex<T>], n: usize, p: &Prepared<T>) {
    let l = p.l;
    if l >= n {
        let mut buf = new_buf::<T>(l);
        let mut sc = DiagScratch::default();
        load_run(&mut buf, 0, amps);
        run_ops(p, &mut buf, 0, &mut sc);
        store_run(&buf, 0, amps);
        return;
    }
    let full = (1usize << n) - 1;
    let outer_phys = full & !p.inner_mask;
    let init = || (new_buf::<T>(l), DiagScratch::default());
    if p.inner_mask == (1usize << l) - 1 {
        // Inner qubits are the low ones: chunks are contiguous.
        amps.par_chunks_mut(1 << l)
            .enumerate()
            .for_each_init(init, |(buf, sc), (c, chunk)| {
                load_run(buf, 0, chunk);
                run_ops(p, buf, c << l, sc);
                store_run(buf, 0, chunk);
            });
        return;
    }
    // Gather path. Runs of 2^bc contiguous amplitudes; the run index rho
    // has slot bits (inner qubits >= bc) and outer bits.
    let bc = p.inner_mask.trailing_ones() as usize;
    let runlen = 1usize << bc;
    let nrun_bits = n - bc;
    let smask = p.inner_mask >> bc;
    let omask = !smask & ((1usize << nrun_bits) - 1);
    let ns = l - bc;
    let ps = Pext::new(smask, nrun_bits);
    let po = Pext::new(omask, nrun_bits);
    let mut ord: Vec<Option<&mut [Complex<T>]>> = (0..1usize << nrun_bits).map(|_| None).collect();
    for (rho, run) in amps.chunks_exact_mut(runlen).enumerate() {
        let key = (po.get(rho) << ns) | ps.get(rho);
        ord[key] = Some(run);
    }
    ord.par_chunks_mut(1 << ns)
        .enumerate()
        .for_each_init(init, |(buf, sc), (c, runs)| {
            for (r, run) in runs.iter().enumerate() {
                let run = run.as_ref().expect("every run is assigned");
                load_run(buf, r << bc, run);
            }
            run_ops(p, buf, deposit(c, outer_phys), sc);
            for (r, run) in runs.iter_mut().enumerate() {
                let run = run.as_mut().expect("every run is assigned");
                store_run(buf, r << bc, run);
            }
        });
}

impl<T: Real> StateVector<T> {
    /// Applies executor ops with cache blocking (see the module docs).
    pub fn apply_kops_blocked(&mut self, ops: &[KOp], cfg: &BlockConfig) {
        let n = self.num_qubits();
        let fused;
        let ops = if cfg.fuse_1q {
            fused = fuse_1q(ops, n);
            &fused[..]
        } else {
            ops
        };
        let l = cfg.block_bits(n, std::mem::size_of::<Complex<T>>());
        let stages = plan_stages(ops, n, l, cfg.slots);
        let trace = std::env::var_os("QSIM_TRACE").is_some();
        let amps = self.amplitudes_mut();
        for st in &stages {
            let t0 = std::time::Instant::now();
            let p = prepare::<T>(st, n);
            let t1 = std::time::Instant::now();
            run_stage(amps, n, &p);
            if trace {
                eprintln!(
                    "stage inner={:?} ops={} lops={} prep={:.2}ms run={:.2}ms",
                    st.inner,
                    st.ops.len(),
                    p.ops.len(),
                    (t1 - t0).as_secs_f64() * 1e3,
                    t1.elapsed().as_secs_f64() * 1e3
                );
            }
        }
    }

    /// Applies a gate sequence with the blocked executor.
    pub fn apply_gates_blocked(
        &mut self,
        gates: &[Gate],
        cfg: &BlockConfig,
    ) -> Result<(), SimError> {
        for g in gates {
            check_gate(g, self.num_qubits())?;
        }
        let ops = lower_gates(gates);
        self.apply_kops_blocked(&ops, cfg);
        Ok(())
    }

    /// Like [`StateVector::apply_circuit`], with the blocked executor.
    pub fn apply_circuit_blocked(
        &mut self,
        c: &Circuit,
        cfg: &BlockConfig,
    ) -> Result<(), SimError> {
        let gates: Vec<Gate> = c
            .ops
            .iter()
            .map(|op| match op {
                Op::Gate(g) => *g,
                Op::Measure(_) => {
                    panic!("apply_circuit_blocked: use Circuit::run for measurements")
                }
            })
            .collect();
        self.apply_gates_blocked(&gates, cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pext_and_deposit() {
        let mask = 0b1011_0010_0110usize;
        let p = Pext::new(mask, 12);
        for x in 0..4096usize {
            let e = p.get(x);
            assert_eq!(deposit(e, mask), x & mask);
        }
    }

    #[test]
    fn runs_cover_subspace() {
        let l = 7;
        let fixed = 0b0100101;
        let mut seen = vec![false; 1 << l];
        for_each_run(l, fixed, |base, run| {
            for (j, sj) in seen.iter_mut().enumerate().skip(base).take(run) {
                assert_eq!(j & fixed, 0);
                assert!(!*sj);
                *sj = true;
            }
        });
        assert_eq!(seen.iter().filter(|&&s| s).count(), 1 << (l - 3));
    }
}
