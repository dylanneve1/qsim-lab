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
        // Other multi-qubit gates (ISwap, ISwapdg) have no dedicated kernel:
        // lower their exact decomposition instead.
        ref g2 if g2.qubits().len() > 1 => {
            for d in g2.decompose_to_clifford_rz() {
                lower_gate(&d, out);
            }
        }
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

fn is_real(m: &Mat2) -> bool {
    m.iter().flatten().all(|z| z.im == 0.0)
}

/// Writes a non-diagonal, non-real 2x2 unitary as `D_a R D_b` with `R` real
/// and `D_a`, `D_b` diagonal: with `p0 = u00/|u00|` (or 1), `p1 = u10/|u10|`
/// and `q = u01/(|u01| p0)`, `D_a = diag(p0, p1)`, `D_b = diag(1, q)` and
/// `R = D_a^* U D_b^*`, whose entries are real up to rounding (unitarity
/// forces `arg u11 = arg u10 + arg u01 - arg u00 + π`).
/// Returns `(D_b, R, D_a)`.
fn split_phases(m: &Mat2) -> ([Complex64; 2], Mat2, [Complex64; 2]) {
    let unit = |z: Complex64| if z == C0 { C1 } else { z / z.norm() };
    let p0 = unit(m[0][0]);
    let p1 = unit(m[1][0]);
    let q = unit(m[0][1]) / p0;
    let da = [p0, p1];
    let db = [C1, q];
    let mut r = [[C0; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            r[i][j] = Complex64::new((da[i].conj() * m[i][j] * db[j].conj()).re, 0.0);
        }
    }
    (db, r, da)
}

fn emit_1q(out: &mut Vec<KOp>, q: usize, m: Mat2, split: bool) {
    if m[0][1] == C0 && m[1][0] == C0 {
        push_diag1(out, q, m[0][0], m[1][1]);
    } else if !split || is_real(&m) {
        out.push(KOp::U1 { q, m, ctrl: 0 });
    } else {
        let (db, r, da) = split_phases(&m);
        push_diag1(out, q, db[0], db[1]);
        out.push(KOp::U1 { q, m: r, ctrl: 0 });
        push_diag1(out, q, da[0], da[1]);
    }
}

/// Multiplies together runs of uncontrolled single-qubit ops on the same
/// qubit (products taken in f64). The result is the same unitary up to
/// rounding. With `split`, a fused gate that is neither real nor diagonal is
/// emitted as phase, real rotation, phase (see [`split_phases`]): the real
/// kernel needs 6 instead of 16 flops per amplitude and the phases merge
/// with other diagonal terms.
pub fn fuse_1q(ops: &[KOp], n: usize, split: bool) -> Vec<KOp> {
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
                        emit_1q(&mut out, q, m, split);
                    }
                }
                out.push(*op);
            }
        }
    }
    for (q, p) in pending.into_iter().enumerate() {
        if let Some(m) = p {
            emit_1q(&mut out, q, m, split);
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
    /// Split fused 1q gates into phase / real rotation / phase.
    ///
    /// **Off by default: known to give wrong amplitudes.** The audit found a
    /// 10-gate, 5-qubit circuit (`Y(0) CX(0,4) Rx(0,π/4) Z H X S Z T H` on
    /// qubit 0) where enabling it changes amplitudes by 0.26; see
    /// `tests/blocked.rs::split_phases_regression` and `research/sv.md`.
    /// Do not enable until that is fixed.
    pub split_phases: bool,
    /// Reorder diagonal terms within a stage (they commute with every op
    /// not targeting their qubits) so they form as few passes as possible.
    pub schedule_diag: bool,
    /// Use the fused-multiply-add kernels: on x86_64 the AVX2+FMA build of
    /// the chunk kernels when the CPU has them (checked at run time; the
    /// portable kernels are used otherwise); on aarch64 the same kernels
    /// with `mul_add` (NEON `fmla`, part of the baseline, no `unsafe`).
    /// Results agree with the portable path to rounding error.
    pub simd: bool,
}

impl Default for BlockConfig {
    fn default() -> Self {
        BlockConfig {
            block_bytes: 256 << 10,
            slots: 6,
            fuse_1q: true,
            small_n: 12,
            split_phases: false,
            schedule_diag: true,
            simd: true,
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

/// Reorders a stage's ops so that the diagonal terms form as few contiguous
/// runs as possible, without changing the unitary.
///
/// Non-diagonal ops are put in ASAP layers of the commutation DAG (two ops
/// commute unless one acts non-diagonally on a qubit the other uses; a
/// control acts diagonally, so diagonal terms commute with controls). A
/// diagonal term may then sit in any gap between its last predecessor's
/// layer and its first successor's layer; picking the fewest gaps that hit
/// every term's window is interval stabbing, solved greedily.
pub fn schedule_diag(ops: &[KOp]) -> Vec<KOp> {
    let nbits = ops
        .iter()
        .map(|op| usize::BITS - op.touches().leading_zeros())
        .max()
        .unwrap_or(0) as usize;
    let bits = |m: usize| (0..nbits).filter(move |q| m >> q & 1 == 1);
    // Forward pass: ASAP layer of every op (diagonal terms take the layer
    // of their latest predecessor, i.e. they sit in the gap after it).
    let mut t_tgt = vec![0usize; nbits]; // non-diagonal action on q
    let mut t_ctl = vec![0usize; nbits]; // used as a control
    let mut t_diag = vec![0usize; nbits]; // diagonal term on q
    let mut layer = vec![0usize; ops.len()];
    for (i, op) in ops.iter().enumerate() {
        match *op {
            KOp::U1 { q, ctrl, .. } => {
                let mut e = t_tgt[q].max(t_ctl[q]).max(t_diag[q]);
                for c in bits(ctrl) {
                    e = e.max(t_tgt[c]);
                }
                let e = e + 1;
                layer[i] = e;
                t_tgt[q] = e;
                for c in bits(ctrl) {
                    t_ctl[c] = t_ctl[c].max(e);
                }
            }
            KOp::Swap { a, b } => {
                let e = 1 + [a, b]
                    .iter()
                    .map(|&q| t_tgt[q].max(t_ctl[q]).max(t_diag[q]))
                    .max()
                    .unwrap_or(0);
                layer[i] = e;
                t_tgt[a] = e;
                t_tgt[b] = e;
            }
            KOp::Phase { mask, .. } => {
                let e = bits(mask).map(|q| t_tgt[q]).max().unwrap_or(0);
                layer[i] = e;
                for q in bits(mask) {
                    t_diag[q] = t_diag[q].max(e);
                }
            }
        }
    }
    let top = layer.iter().copied().max().unwrap_or(0);
    // Backward pass: the latest gap each diagonal term may move to.
    let mut first_tgt = vec![top + 1; nbits];
    let mut hi = vec![0usize; ops.len()];
    for (i, op) in ops.iter().enumerate().rev() {
        match *op {
            KOp::U1 { q, .. } => first_tgt[q] = first_tgt[q].min(layer[i]),
            KOp::Swap { a, b } => {
                first_tgt[a] = first_tgt[a].min(layer[i]);
                first_tgt[b] = first_tgt[b].min(layer[i]);
            }
            KOp::Phase { mask, .. } => {
                hi[i] = bits(mask).map(|q| first_tgt[q]).min().unwrap_or(top + 1) - 1;
            }
        }
    }
    // Interval stabbing over gaps 0..=top, greedily by *left* end (latest
    // left end first): every point is some term's earliest gap. Points at
    // left ends keep terms next to the op that created their window (in a
    // QFT, all phases conditioned on the qubit just rotated), which keeps
    // each run to few pivot groups.
    let mut diag: Vec<usize> = (0..ops.len())
        .filter(|&i| matches!(ops[i], KOp::Phase { .. }))
        .collect();
    diag.sort_by_key(|&i| (std::cmp::Reverse(layer[i]), i));
    let mut gap_of = vec![usize::MAX; ops.len()];
    let mut point: Option<usize> = None;
    for i in diag {
        let p = match point {
            Some(p) if p <= hi[i] => p,
            _ => {
                point = Some(layer[i]);
                layer[i]
            }
        };
        gap_of[i] = p;
    }
    let mut out = Vec::with_capacity(ops.len());
    let mut by_slot: Vec<Vec<usize>> = vec![Vec::new(); 2 * top + 2];
    for (i, op) in ops.iter().enumerate() {
        // slot 2g: diagonal terms in gap g; slot 2l - 1: ops of layer l
        let slot = match op {
            KOp::Phase { .. } => 2 * gap_of[i],
            _ => 2 * layer[i] - 1,
        };
        by_slot[slot].push(i);
    }
    for slot in by_slot {
        out.extend(slot.into_iter().map(|i| ops[i]));
    }
    out
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

#[derive(Clone, Copy, Debug, PartialEq)]
enum UKind {
    X,
    Real,
    Complex,
}

#[derive(Clone, Debug)]
enum LOp<T: Real> {
    U1 {
        t: usize,
        m: [T; 8],
        kind: UKind,
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

#[inline(always)]
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
                    kind: if m == XMAT {
                        UKind::X
                    } else if is_real(&m) {
                        UKind::Real
                    } else {
                        UKind::Complex
                    },
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

/// Calls `f(base, run)` for every maximal run of indices `j < 2^l` with
/// `(j & fixed) == 0`: `base..base+run` are those indices.
#[inline(always)]
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

/// `a * b + c`: a fused multiply-add when `F` (only ever instantiated with
/// `F = true` inside `#[target_feature(enable = "fma")]` code, where it
/// lowers to one `vfmadd` instruction; with `F = false` it is the portable
/// two-rounding `a * b + c`, so the fallback never calls a software `fma`).
#[inline(always)]
fn fma<T: Real, const F: bool>(a: T, b: T, c: T) -> T {
    if F {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// `(a, b) <- (m0 a + m1 b, m2 a + m3 b)` elementwise; `m` holds the real
/// parts then the imaginary parts of the four entries.
#[inline(always)]
fn u1_kernel<T: Real, const F: bool>(
    ar: &mut [T],
    ai: &mut [T],
    br: &mut [T],
    bi: &mut [T],
    m: &[T; 8],
) {
    let [m0r, m1r, m2r, m3r, m0i, m1i, m2i, m3i] = *m;
    let len = ar.len();
    let (ai, br, bi) = (&mut ai[..len], &mut br[..len], &mut bi[..len]);
    for k in 0..len {
        let (xr, xi, yr, yi) = (ar[k], ai[k], br[k], bi[k]);
        let (a, b) = cmul2::<T, F>([m0r, m0i, m1r, m1i], xr, xi, yr, yi);
        let (c, d) = cmul2::<T, F>([m2r, m2i, m3r, m3i], xr, xi, yr, yi);
        ar[k] = a;
        ai[k] = b;
        br[k] = c;
        bi[k] = d;
    }
}

/// `(m0 + i m0i) (xr + i xi) + (m1 + i m1i) (yr + i yi)` for
/// `m = [m0, m0i, m1, m1i]`.
#[inline(always)]
fn cmul2<T: Real, const F: bool>(m: [T; 4], xr: T, xi: T, yr: T, yi: T) -> (T, T) {
    let [m0r, m0i, m1r, m1i] = m;
    let re = fma::<T, F>(
        m0r,
        xr,
        fma::<T, F>(-m0i, xi, fma::<T, F>(m1r, yr, -(m1i * yi))),
    );
    let im = fma::<T, F>(
        m0r,
        xi,
        fma::<T, F>(m0i, xr, fma::<T, F>(m1r, yi, m1i * yr)),
    );
    (re, im)
}

/// Splits `v` into `v[lo..lo+run]` and `v[hi..hi+run]` (`lo + run <= hi`).
#[inline(always)]
fn two<T>(v: &mut [T], lo: usize, hi: usize, run: usize) -> (&mut [T], &mut [T]) {
    let (a, b) = v.split_at_mut(hi);
    (&mut a[lo..lo + run], &mut b[..run])
}

/// Index bits below which runs are too short for slice kernels.
const SMALL: usize = 3;

/// Calls `f(i, i + s)` for every index `i < 2^l` with `(i & fixed) == cin`
/// (`fixed` contains the target bit `s` and the control bits `cin`), for
/// masks with bits in `SMALL`: runs are taken over the other bits so that
/// they are at least 4 long, and the low bits are tested per element.
#[inline(always)]
fn small_pairs(l: usize, fixed: usize, cin: usize, s: usize, mut f: impl FnMut(usize, usize)) {
    let (fs, cs) = (fixed & SMALL, cin & SMALL);
    let cbig = cin & !SMALL;
    for_each_run(l, fixed & !SMALL, |base, run| {
        let base = base | cbig;
        for k in 0..run {
            if k & fs == cs {
                f(base + k, base + k + s);
            }
        }
    });
}

/// `(a, b) <- (m0 a + m1 b, m2 a + m3 b)` for a real matrix.
#[inline(always)]
fn u1_real_kernel<T: Real, const F: bool>(
    ar: &mut [T],
    ai: &mut [T],
    br: &mut [T],
    bi: &mut [T],
    m: &[T; 8],
) {
    let [m0, m1, m2, m3, ..] = *m;
    let len = ar.len();
    let (ai, br, bi) = (&mut ai[..len], &mut br[..len], &mut bi[..len]);
    for k in 0..len {
        let (xr, xi, yr, yi) = (ar[k], ai[k], br[k], bi[k]);
        ar[k] = fma::<T, F>(m0, xr, m1 * yr);
        ai[k] = fma::<T, F>(m0, xi, m1 * yi);
        br[k] = fma::<T, F>(m2, xr, m3 * yr);
        bi[k] = fma::<T, F>(m2, xi, m3 * yi);
    }
}

#[inline(always)]
fn u1_slices<T: Real, const F: bool>(
    ar: &mut [T],
    ai: &mut [T],
    br: &mut [T],
    bi: &mut [T],
    m: &[T; 8],
    kind: UKind,
) {
    match kind {
        UKind::X => {
            ar.swap_with_slice(br);
            ai.swap_with_slice(bi);
        }
        UKind::Real => u1_real_kernel::<T, F>(ar, ai, br, bi, m),
        UKind::Complex => u1_kernel::<T, F>(ar, ai, br, bi, m),
    }
}

/// Uncontrolled 2x2 gate on target bit `TB < 2`: the pairs lie inside
/// aligned groups of 8, handled with fixed index maps so the four pairs of a
/// group are computed as one short vector (about 2x faster than walking
/// runs of length 1 or 2).
#[inline(always)]
fn u1_group8<T: Real, const F: bool, const TB: usize, const K: u8>(
    re: &mut [T],
    im: &mut [T],
    m: &[T; 8],
) {
    let s = 1usize << TB;
    let lo: [usize; 4] = std::array::from_fn(|k| ((k >> TB) << (TB + 1)) | (k & (s - 1)));
    let [m0r, m1r, m2r, m3r, m0i, m1i, m2i, m3i] = *m;
    for (gr, gi) in re.chunks_exact_mut(8).zip(im.chunks_exact_mut(8)) {
        let xr: [T; 4] = std::array::from_fn(|k| gr[lo[k]]);
        let xi: [T; 4] = std::array::from_fn(|k| gi[lo[k]]);
        let yr: [T; 4] = std::array::from_fn(|k| gr[lo[k] + s]);
        let yi: [T; 4] = std::array::from_fn(|k| gi[lo[k] + s]);
        for k in 0..4 {
            let (ar, ai, br, bi) = match K {
                0 => (yr[k], yi[k], xr[k], xi[k]),
                1 => (
                    fma::<T, F>(m0r, xr[k], m1r * yr[k]),
                    fma::<T, F>(m0r, xi[k], m1r * yi[k]),
                    fma::<T, F>(m2r, xr[k], m3r * yr[k]),
                    fma::<T, F>(m2r, xi[k], m3r * yi[k]),
                ),
                _ => {
                    let (a, b) = cmul2::<T, F>([m0r, m0i, m1r, m1i], xr[k], xi[k], yr[k], yi[k]);
                    let (c, d) = cmul2::<T, F>([m2r, m2i, m3r, m3i], xr[k], xi[k], yr[k], yi[k]);
                    (a, b, c, d)
                }
            };
            gr[lo[k]] = ar;
            gi[lo[k]] = ai;
            gr[lo[k] + s] = br;
            gi[lo[k] + s] = bi;
        }
    }
}

#[inline(always)]
fn apply_u1<T: Real, const F: bool>(
    buf: &mut Buf<T>,
    l: usize,
    t: usize,
    m: &[T; 8],
    kind: UKind,
    cin: usize,
) {
    let s = 1usize << t;
    let Buf { re, im } = buf;
    let fixed = cin | s;
    if cin == 0 && t < 2 && l >= 3 {
        match (t, kind) {
            (0, UKind::X) => u1_group8::<T, F, 0, 0>(re, im, m),
            (0, UKind::Real) => u1_group8::<T, F, 0, 1>(re, im, m),
            (0, UKind::Complex) => u1_group8::<T, F, 0, 2>(re, im, m),
            (_, UKind::X) => u1_group8::<T, F, 1, 0>(re, im, m),
            (_, UKind::Real) => u1_group8::<T, F, 1, 1>(re, im, m),
            (_, UKind::Complex) => u1_group8::<T, F, 1, 2>(re, im, m),
        }
        return;
    }
    if fixed & SMALL != 0 {
        let [m0r, m1r, m2r, m3r, m0i, m1i, m2i, m3i] = *m;
        match kind {
            UKind::X => small_pairs(l, fixed, cin, s, |i, j| {
                re.swap(i, j);
                im.swap(i, j);
            }),
            UKind::Real => small_pairs(l, fixed, cin, s, |i, j| {
                let (xr, xi, yr, yi) = (re[i], im[i], re[j], im[j]);
                re[i] = fma::<T, F>(m0r, xr, m1r * yr);
                im[i] = fma::<T, F>(m0r, xi, m1r * yi);
                re[j] = fma::<T, F>(m2r, xr, m3r * yr);
                im[j] = fma::<T, F>(m2r, xi, m3r * yi);
            }),
            UKind::Complex => small_pairs(l, fixed, cin, s, |i, j| {
                let (xr, xi, yr, yi) = (re[i], im[i], re[j], im[j]);
                (re[i], im[i]) = cmul2::<T, F>([m0r, m0i, m1r, m1i], xr, xi, yr, yi);
                (re[j], im[j]) = cmul2::<T, F>([m2r, m2i, m3r, m3i], xr, xi, yr, yi);
            }),
        }
        return;
    }
    if cin == 0 {
        for (cr, ci) in re.chunks_exact_mut(2 * s).zip(im.chunks_exact_mut(2 * s)) {
            let (ar, br) = cr.split_at_mut(s);
            let (ai, bi) = ci.split_at_mut(s);
            u1_slices::<T, F>(ar, ai, br, bi, m, kind);
        }
        return;
    }
    for_each_run(l, fixed, |base, run| {
        let base = base | cin;
        let (ar, br) = two(re, base, base + s, run);
        let (ai, bi) = two(im, base, base + s, run);
        u1_slices::<T, F>(ar, ai, br, bi, m, kind);
    });
}

#[inline(always)]
fn apply_swap<T: Real>(buf: &mut Buf<T>, l: usize, a: usize, b: usize) {
    let (sa, sb) = (1usize << a, 1usize << b);
    let Buf { re, im } = buf;
    if (sa | sb) & SMALL != 0 {
        // pairs (i | sa, i | sb) with both bits clear in i
        small_pairs(l, sa | sb, 0, sb - sa, |i, j| {
            re.swap(i + sa, j + sa);
            im.swap(i + sa, j + sa);
        });
        return;
    }
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
fn diag_kernel<T: Real, const F: bool>(
    ar: &mut [T],
    ai: &mut [T],
    lr: &[T],
    li: &[T],
    h: Complex<T>,
) {
    let len = ar.len();
    let (ai, lr, li) = (&mut ai[..len], &lr[..len], &li[..len]);
    for k in 0..len {
        let fr = fma::<T, F>(lr[k], h.re, -(li[k] * h.im));
        let fi = fma::<T, F>(lr[k], h.im, li[k] * h.re);
        let (xr, xi) = (ar[k], ai[k]);
        ar[k] = fma::<T, F>(xr, fr, -(xi * fi));
        ai[k] = fma::<T, F>(xr, fi, xi * fr);
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

#[inline(always)]
fn apply_diag_group<T: Real, const F: bool>(
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
            diag_kernel::<T, F>(rr, ri, &sc.lor, &sc.loi, hv);
        } else if cm_lo & SMALL != 0 {
            for x in 0..rr.len() {
                if x & cm_lo == cp_lo {
                    let fr = fma::<T, F>(sc.lor[x], hv.re, -(sc.loi[x] * hv.im));
                    let fi = fma::<T, F>(sc.lor[x], hv.im, sc.loi[x] * hv.re);
                    let (xr, xi) = (rr[x], ri[x]);
                    rr[x] = fma::<T, F>(xr, fr, -(xi * fi));
                    ri[x] = fma::<T, F>(xr, fi, xi * fr);
                }
            }
        } else {
            for_each_run(lb, cm_lo, |b0, run| {
                let b0 = b0 | cp_lo;
                diag_kernel::<T, F>(
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

/// Runs the ops of a prepared stage on one buffer, using the AVX2+FMA build
/// of the kernels when `simd` is set (see [`simd_available`]).
fn run_ops<T: Real>(
    p: &Prepared<T>,
    buf: &mut Buf<T>,
    base: usize,
    sc: &mut DiagScratch<T>,
    simd: bool,
) {
    #[cfg(target_arch = "x86_64")]
    if simd {
        // SAFETY: `simd` is only ever true when `simd_available()` returned
        // true, i.e. the running CPU supports AVX2 and FMA.
        unsafe { return run_ops_avx2(p, buf, base, sc) };
    }
    #[cfg(target_arch = "aarch64")]
    if simd {
        // NEON with fused multiply-add is part of the aarch64 baseline, so
        // no target feature or run-time check is needed: `mul_add` lowers to
        // one `fmla`/`fmadd` and `simd_available()` is simply true.
        return run_ops_impl::<T, true>(p, buf, base, sc);
    }
    let _ = simd;
    run_ops_impl::<T, false>(p, buf, base, sc);
}

/// The same kernels, compiled with AVX2 and FMA enabled.
///
/// # Safety
/// The caller must have verified that the CPU supports `avx2` and `fma`.
/// Everything reached from here is `#[inline(always)]`, so it is compiled
/// as part of this function with those features; the body is safe Rust.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn run_ops_avx2<T: Real>(
    p: &Prepared<T>,
    buf: &mut Buf<T>,
    base: usize,
    sc: &mut DiagScratch<T>,
) {
    run_ops_impl::<T, true>(p, buf, base, sc)
}

/// Whether the FMA kernels can run on this CPU: AVX2+FMA detected at run
/// time on x86_64, always on aarch64.
pub fn simd_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

/// Optional per-kernel time accounting (env `QSIM_PROF`), for profiling.
mod prof {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    pub const NAMES: [&str; 8] = [
        "u1 x",
        "u1 real",
        "u1 complex",
        "u1 small-bit",
        "swap",
        "diag",
        "load",
        "store",
    ];
    pub static ON: AtomicBool = AtomicBool::new(false);
    pub static NS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
    #[inline]
    pub fn on() -> bool {
        ON.load(Ordering::Relaxed)
    }
    #[inline]
    pub fn add(k: usize, t: std::time::Instant) {
        NS[k].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    pub fn report() {
        let v: Vec<String> = NAMES
            .iter()
            .zip(NS.iter())
            .map(|(n, a)| format!("{n}={:.1}ms", a.swap(0, Ordering::Relaxed) as f64 / 1e6))
            .collect();
        eprintln!("prof (thread-summed): {}", v.join(" "));
    }
}

#[inline(always)]
fn run_ops_impl<T: Real, const F: bool>(
    p: &Prepared<T>,
    buf: &mut Buf<T>,
    base: usize,
    sc: &mut DiagScratch<T>,
) {
    if prof::on() {
        return run_ops_prof::<T, F>(p, buf, base, sc);
    }
    let l = p.l;
    for op in &p.ops {
        match op {
            LOp::U1 {
                t,
                m,
                kind,
                cin,
                cout,
            } => {
                if base & cout == *cout {
                    apply_u1::<T, F>(buf, l, *t, m, *kind, *cin);
                }
            }
            LOp::Swap { a, b } => apply_swap(buf, l, *a, *b),
            LOp::Diag(d) => {
                for g in &d.groups {
                    apply_diag_group::<T, F>(buf, l, g, base, sc);
                }
            }
        }
    }
}

#[inline(always)]
fn run_ops_prof<T: Real, const F: bool>(
    p: &Prepared<T>,
    buf: &mut Buf<T>,
    base: usize,
    sc: &mut DiagScratch<T>,
) {
    let l = p.l;
    for op in &p.ops {
        let t0 = std::time::Instant::now();
        match op {
            LOp::U1 {
                t,
                m,
                kind,
                cin,
                cout,
            } => {
                if base & cout == *cout {
                    apply_u1::<T, F>(buf, l, *t, m, *kind, *cin);
                }
                let k = if (cin | (1 << t)) & SMALL != 0 {
                    3
                } else {
                    *kind as usize
                };
                prof::add(k, t0);
            }
            LOp::Swap { a, b } => {
                apply_swap(buf, l, *a, *b);
                prof::add(4, t0);
            }
            LOp::Diag(d) => {
                for g in &d.groups {
                    apply_diag_group::<T, F>(buf, l, g, base, sc);
                }
                prof::add(5, t0);
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

fn run_stage<T: Real>(amps: &mut [Complex<T>], n: usize, p: &Prepared<T>, simd: bool) {
    let l = p.l;
    if l >= n {
        let mut buf = new_buf::<T>(l);
        let mut sc = DiagScratch::default();
        load_run(&mut buf, 0, amps);
        run_ops(p, &mut buf, 0, &mut sc, simd);
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
                let t0 = std::time::Instant::now();
                load_run(buf, 0, chunk);
                if prof::on() {
                    prof::add(6, t0);
                }
                run_ops(p, buf, c << l, sc, simd);
                let t0 = std::time::Instant::now();
                store_run(buf, 0, chunk);
                if prof::on() {
                    prof::add(7, t0);
                }
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
            let t0 = std::time::Instant::now();
            for (r, run) in runs.iter().enumerate() {
                let run = run.as_ref().expect("every run is assigned");
                load_run(buf, r << bc, run);
            }
            if prof::on() {
                prof::add(6, t0);
            }
            run_ops(p, buf, deposit(c, outer_phys), sc, simd);
            let t0 = std::time::Instant::now();
            for (r, run) in runs.iter_mut().enumerate() {
                let run = run.as_mut().expect("every run is assigned");
                store_run(buf, r << bc, run);
            }
            if prof::on() {
                prof::add(7, t0);
            }
        });
}

impl<T: Real> StateVector<T> {
    /// Applies executor ops with cache blocking (see the module docs).
    pub fn apply_kops_blocked(&mut self, ops: &[KOp], cfg: &BlockConfig) {
        let n = self.num_qubits();
        let fused;
        let ops = if cfg.fuse_1q {
            fused = fuse_1q(ops, n, cfg.split_phases);
            &fused[..]
        } else {
            ops
        };
        let l = cfg.block_bits(n, std::mem::size_of::<Complex<T>>());
        let stages = plan_stages(ops, n, l, cfg.slots);
        let trace = std::env::var_os("QSIM_TRACE").is_some();
        prof::ON.store(
            std::env::var_os("QSIM_PROF").is_some(),
            std::sync::atomic::Ordering::Relaxed,
        );
        let simd = cfg.simd && simd_available();
        let amps = self.amplitudes_mut();
        for st in &stages {
            let t0 = std::time::Instant::now();
            let p = if cfg.schedule_diag {
                prepare::<T>(
                    &Stage {
                        inner: st.inner.clone(),
                        ops: schedule_diag(&st.ops),
                    },
                    n,
                )
            } else {
                prepare::<T>(st, n)
            };
            let t1 = std::time::Instant::now();
            run_stage(amps, n, &p, simd);
            if trace {
                let groups: Vec<usize> = p
                    .ops
                    .iter()
                    .filter_map(|o| match o {
                        LOp::Diag(d) => Some(d.groups.len()),
                        _ => None,
                    })
                    .collect();
                eprintln!(
                    "stage inner={:?} ops={} lops={} diag groups={:?} prep={:.2}ms run={:.2}ms",
                    st.inner,
                    st.ops.len(),
                    p.ops.len(),
                    groups,
                    (t1 - t0).as_secs_f64() * 1e3,
                    t1.elapsed().as_secs_f64() * 1e3
                );
            }
        }
        if prof::on() {
            prof::report();
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
                // Measurements, resets, noise channels and classically
                // conditioned ops are not unitary gates; the blocked
                // executor only handles gate sequences.
                _ => panic!(
                    "apply_circuit_blocked: only unitary gates are supported; \
                     use Circuit::run for measurements, resets, noise and classical control"
                ),
            })
            .collect();
        self.apply_gates_blocked(&gates, cfg)
    }
}

/// Prepares and executes blocked stages repeatedly on chunks of `2^c` amplitudes.
pub struct BlockedChunkExecutor<T: Real> {
    c: usize,
    stages: Vec<Prepared<T>>,
    /// Use the AVX2+FMA kernels (`cfg.simd` and the CPU supports them).
    simd: bool,
}

impl<T: Real> BlockedChunkExecutor<T> {
    /// Builds an executor for gates acting on local qubits `0..c`.
    pub fn new(gates: &[Gate], c: usize, cfg: &BlockConfig) -> Result<Self, SimError> {
        for g in gates {
            check_gate(g, c)?;
        }
        let ops = lower_gates(gates);
        Ok(Self::from_kops(&ops, c, cfg))
    }

    /// Builds an executor from lowered ops on qubits `0..c`.
    pub fn from_kops(ops: &[KOp], c: usize, cfg: &BlockConfig) -> Self {
        let fused;
        let ops = if cfg.fuse_1q {
            fused = fuse_1q(ops, c, cfg.split_phases);
            &fused[..]
        } else {
            ops
        };
        let l = cfg.block_bits(c, std::mem::size_of::<Complex<T>>());
        let stages = plan_stages(ops, c, l, cfg.slots);
        let prepared = stages
            .into_iter()
            .map(|st| {
                if cfg.schedule_diag {
                    prepare::<T>(
                        &Stage {
                            inner: st.inner.clone(),
                            ops: schedule_diag(&st.ops),
                        },
                        c,
                    )
                } else {
                    prepare::<T>(&st, c)
                }
            })
            .collect();
        BlockedChunkExecutor {
            c,
            stages: prepared,
            simd: cfg.simd && simd_available(),
        }
    }

    /// Applies the pre-compiled stages to an in-RAM chunk of `2^c` amplitudes.
    pub fn apply_to_chunk(&self, chunk: &mut [Complex<T>]) {
        assert_eq!(chunk.len(), 1 << self.c);
        for p in &self.stages {
            run_stage(chunk, self.c, p, self.simd);
        }
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
