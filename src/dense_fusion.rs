//! Dense k-qubit gate fusion for the blocked state-vector executor
//! (`BlockConfig::dense_fusion`, default off).
//!
//! Inside one stage of the blocked executor (all targets cached in the
//! block), consecutive ops whose qubits fit in a set of at most `k <= 3`
//! inner qubits are multiplied into one dense `2^k x 2^k` unitary, applied
//! in a single pass over the block by the kernels in
//! [`crate::dense_kernels`]. This trades arithmetic for passes: a fused
//! 2-qubit block costs a 4x4 complex matrix-vector product per amplitude
//! group instead of two or three separate sweeps.
//!
//! Origin: exp/sv-monomial / wip/fusion-avx2 (unreviewed WIP, Oct 1),
//! ported onto the current executor, reviewed and differential-tested
//! (`tests/dense_fusion.rs`). Measured result: `research/performance/dense-fusion.md`.
//!
//! # Ordering argument
//!
//! A pending block only ever contains ops on its own qubits, and any later
//! op touching one of those qubits is either multiplied into the block or
//! flushes it first. Ops emitted while a block is pending therefore act on
//! disjoint qubits and commute with it, so the output is the input product
//! with commuting factors reordered.

use crate::blocked::KOp;
use crate::gate::Mat2;
use num_complex::Complex64;

const C0: Complex64 = Complex64::new(0.0, 0.0);
const C1: Complex64 = Complex64::new(1.0, 0.0);

/// A dense unitary on sorted physical qubits `qs`: `mat` is row-major of
/// size `2^|qs|`, local index bit `j` = qubit `qs[j]`.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseOp {
    pub qs: Vec<usize>,
    pub mat: Vec<Complex64>,
}

/// One op of a fused stage.
#[derive(Clone, Debug, PartialEq)]
pub enum FusedOp {
    /// An executor op left as is.
    Plain(KOp),
    /// A dense unitary on 2 or 3 qubits.
    Dense(DenseOp),
}

fn bits(mut m: usize) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        if m == 0 {
            None
        } else {
            let q = m.trailing_zeros() as usize;
            m &= m - 1;
            Some(q)
        }
    })
}

impl DenseOp {
    /// Embeds this op into the sorted superset `new_qs` (identity on the
    /// extra qubits).
    fn lift(&self, new_qs: &[usize]) -> DenseOp {
        if self.qs == new_qs {
            return self.clone();
        }
        let d_new = 1usize << new_qs.len();
        let d_old = 1usize << self.qs.len();
        let mut extra = 0usize;
        let mut map = Vec::with_capacity(self.qs.len());
        for (j, q) in new_qs.iter().enumerate() {
            match self.qs.iter().position(|x| x == q) {
                Some(p) => map.push((j, p)),
                None => extra |= 1 << j,
            }
        }
        let local = |x: usize| map.iter().fold(0, |o, &(j, p)| o | (((x >> j) & 1) << p));
        let mut mat = vec![C0; d_new * d_new];
        for y in 0..d_new {
            for x in 0..d_new {
                if y & extra == x & extra {
                    mat[y * d_new + x] = self.mat[local(y) * d_old + local(x)];
                }
            }
        }
        DenseOp {
            qs: new_qs.to_vec(),
            mat,
        }
    }

    /// `next * self` (apply `self` first) on the union of the qubits.
    fn then(&self, next: &DenseOp) -> DenseOp {
        let mut qs = self.qs.clone();
        qs.extend(next.qs.iter().copied().filter(|q| !self.qs.contains(q)));
        qs.sort_unstable();
        let a = next.lift(&qs);
        let b = self.lift(&qs);
        let d = 1usize << qs.len();
        let mut mat = vec![C0; d * d];
        for i in 0..d {
            for k in 0..d {
                let aik = a.mat[i * d + k];
                if aik == C0 {
                    continue;
                }
                for j in 0..d {
                    mat[i * d + j] += aik * b.mat[k * d + j];
                }
            }
        }
        DenseOp { qs, mat }
    }
}

/// `m` on qubit `q`, active where all qubits of `ctrl` are 1.
fn u1_dense(q: usize, m: &Mat2, ctrl: usize) -> DenseOp {
    let mut qs: Vec<usize> = bits(ctrl | (1 << q)).collect();
    qs.sort_unstable();
    let d = 1usize << qs.len();
    let tp = qs.iter().position(|&x| x == q).unwrap();
    let cm: usize = qs
        .iter()
        .enumerate()
        .filter(|&(_, &x)| (ctrl >> x) & 1 == 1)
        .map(|(j, _)| 1 << j)
        .sum();
    let mut mat = vec![C0; d * d];
    for y in 0..d {
        for x in 0..d {
            if x & cm != cm {
                if y == x {
                    mat[y * d + x] = C1;
                }
            } else if (x ^ y) & !(1 << tp) == 0 {
                mat[y * d + x] = m[(y >> tp) & 1][(x >> tp) & 1];
            }
        }
    }
    DenseOp { qs, mat }
}

fn swap_dense(a: usize, b: usize) -> DenseOp {
    let qs = vec![a.min(b), a.max(b)];
    let mut mat = vec![C0; 16];
    for x in 0..4usize {
        let y = ((x & 1) << 1) | (x >> 1);
        mat[y * 4 + x] = C1;
    }
    DenseOp { qs, mat }
}

/// Multiplies amplitudes with `(i & mask) == pat` by `f`.
fn phase_dense(mask: usize, pat: usize, f: Complex64) -> DenseOp {
    let qs: Vec<usize> = bits(mask).collect();
    let d = 1usize << qs.len();
    let lp: usize = qs
        .iter()
        .enumerate()
        .filter(|&(_, &q)| (pat >> q) & 1 == 1)
        .map(|(j, _)| 1 << j)
        .sum();
    let mut mat = vec![C0; d * d];
    for x in 0..d {
        mat[x * d + x] = if x == lp { f } else { C1 };
    }
    DenseOp { qs, mat }
}

struct Block {
    op: DenseOp,
    /// The input ops multiplied in, in an order equivalent to the input.
    ops: Vec<KOp>,
}

/// Fuses the ops of one stage. `inner` = bit mask of the qubits cached in
/// the block (only those can be dense targets); `max_k` in `2..=3`.
///
/// Fusable: 1-qubit and controlled 1-qubit gates and swaps whose qubits are
/// all inner and number at most `max_k`. Diagonal phases are multiplied in
/// only when all their qubits already belong to one pending block (they are
/// cheap on their own and the diagonal scheduler groups them).
///
/// Cost rule: a group on `k >= 2` qubits becomes a dense op only if it
/// contains at least `min_ops` uncontrolled 1-qubit gates with a dense
/// 2x2 matrix (no zero entry; `0` = `2^k`);
/// otherwise its ops are emitted unchanged. A dense `2^k x 2^k` product
/// costs `2^k` complex multiply-adds per amplitude, the same as `2^(k-1)`
/// separate 1-qubit gates, and the dense kernel is slower per flop than the
/// specialised 1-qubit kernels, while controlled gates, swaps and phases
/// are cheaper than a 1-qubit gate. So fusing brickwork's `U1 U1 CNOT` is a
/// net loss (0.87-0.98x on the M1 Pro), while a generic two-qubit unitary
/// written as 3 x (`U1 U1 CNOT`) gains 2x (`research/performance/dense-fusion.md`).
/// Only those gates are counted, so chains of cheap permutations (CNOT,
/// CCX, X, swaps: arithmetic circuits) are never fused. `min_ops = 1` fuses every group of two or
/// more gates (for tests and experiments). A group on one qubit always becomes a
/// single 1-qubit gate.
pub fn fuse_stage(
    ops: &[KOp],
    n: usize,
    inner: usize,
    max_k: usize,
    min_ops: usize,
) -> Vec<FusedOp> {
    assert!(
        (2..=3).contains(&max_k),
        "dense fusion width must be 2 or 3"
    );
    let mut out = Vec::with_capacity(ops.len());
    let mut blocks: Vec<Option<Block>> = Vec::new();
    let mut owner: Vec<Option<usize>> = vec![None; n];

    let flush = |b: usize,
                 blocks: &mut [Option<Block>],
                 owner: &mut [Option<usize>],
                 out: &mut Vec<FusedOp>| {
        let Some(bl) = blocks[b].take() else { return };
        for &q in &bl.op.qs {
            owner[q] = None;
        }
        let k = bl.op.qs.len();
        let need = if min_ops == 0 { 1 << k } else { min_ops };
        let heavy = bl
            .ops
            .iter()
            .filter(
                |o| matches!(o, KOp::U1 { ctrl: 0, m, .. } if m.iter().flatten().all(|z| *z != C0)),
            )
            .count();
        if bl.ops.len() == 1 || (k >= 2 && min_ops != 1 && heavy < need) {
            out.extend(bl.ops.into_iter().map(FusedOp::Plain));
        } else if k == 1 {
            let m = &bl.op.mat;
            out.push(FusedOp::Plain(KOp::U1 {
                q: bl.op.qs[0],
                m: [[m[0], m[1]], [m[2], m[3]]],
                ctrl: 0,
            }));
        } else {
            out.push(FusedOp::Dense(bl.op));
        }
    };

    for op in ops {
        let touches = match *op {
            KOp::U1 { q, ctrl, .. } => (1 << q) | ctrl,
            KOp::Phase { mask, .. } => mask,
            KOp::Swap { a, b } => (1 << a) | (1 << b),
        };
        let mut owners: Vec<usize> = Vec::new();
        for q in bits(touches) {
            if let Some(b) = owner[q] {
                if !owners.contains(&b) {
                    owners.push(b);
                }
            }
        }
        let candidate = if touches & !inner != 0 || touches.count_ones() as usize > max_k {
            None
        } else {
            match *op {
                KOp::U1 { q, m, ctrl } => Some(u1_dense(q, &m, ctrl)),
                KOp::Swap { a, b } => Some(swap_dense(a, b)),
                KOp::Phase { mask, pat, f } => {
                    let inside = owners.len() == 1
                        && blocks[owners[0]]
                            .as_ref()
                            .is_some_and(|bl| bits(mask).all(|q| bl.op.qs.contains(&q)));
                    inside.then(|| phase_dense(mask, pat, f))
                }
            }
        };
        let Some(d) = candidate else {
            for b in owners {
                flush(b, &mut blocks, &mut owner, &mut out);
            }
            out.push(FusedOp::Plain(*op));
            continue;
        };
        let mut union = touches;
        for &b in &owners {
            union |= blocks[b]
                .as_ref()
                .unwrap()
                .op
                .qs
                .iter()
                .map(|&q| 1usize << q)
                .sum::<usize>();
        }
        let idx = if owners.is_empty() || union.count_ones() as usize > max_k {
            for b in owners {
                flush(b, &mut blocks, &mut owner, &mut out);
            }
            blocks.push(Some(Block {
                op: d,
                ops: vec![*op],
            }));
            blocks.len() - 1
        } else {
            // Pending blocks on disjoint qubits commute: multiply them
            // together, then apply the new op.
            let mut acc = blocks[owners[0]].take().unwrap();
            for &b in &owners[1..] {
                let o = blocks[b].take().unwrap();
                acc.op = acc.op.then(&o.op);
                acc.ops.extend(o.ops);
            }
            acc.op = acc.op.then(&d);
            acc.ops.push(*op);
            blocks[owners[0]] = Some(acc);
            owners[0]
        };
        for &q in &blocks[idx].as_ref().unwrap().op.qs {
            owner[q] = Some(idx);
        }
    }
    for b in 0..blocks.len() {
        flush(b, &mut blocks, &mut owner, &mut out);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_1_SQRT_2;

    fn h() -> Mat2 {
        let h = Complex64::new(FRAC_1_SQRT_2, 0.0);
        [[h, h], [h, -h]]
    }

    #[test]
    fn h_squared_is_identity() {
        let a = u1_dense(0, &h(), 0);
        let id = a.then(&a);
        assert_eq!(id.qs, vec![0]);
        for (i, z) in id.mat.iter().enumerate() {
            let want = if i == 0 || i == 3 { C1 } else { C0 };
            assert!((z - want).norm() < 1e-12);
        }
    }

    #[test]
    fn bell_column() {
        let x = [[C0, C1], [C1, C0]];
        let bell = u1_dense(0, &h(), 0).then(&u1_dense(1, &x, 1));
        assert_eq!(bell.qs, vec![0, 1]);
        let r = FRAC_1_SQRT_2;
        let col0: Vec<f64> = (0..4).map(|y| bell.mat[y * 4].re).collect();
        assert!((col0[0] - r).abs() < 1e-12 && (col0[3] - r).abs() < 1e-12);
        assert!(col0[1].abs() < 1e-12 && col0[2].abs() < 1e-12);
    }

    #[test]
    fn lone_ops_pass_through() {
        let x = [[C0, C1], [C1, C0]];
        let ops = [
            KOp::U1 {
                q: 1,
                m: x,
                ctrl: 1,
            },
            KOp::Swap { a: 2, b: 3 },
        ];
        let f = fuse_stage(&ops, 4, 0b1111, 3, 0);
        assert_eq!(f, vec![FusedOp::Plain(ops[0]), FusedOp::Plain(ops[1])]);
    }

    #[test]
    fn outer_qubits_are_not_fused() {
        let ops = [
            KOp::U1 {
                q: 0,
                m: h(),
                ctrl: 0,
            },
            KOp::U1 {
                q: 0,
                m: h(),
                ctrl: 1 << 3,
            },
        ];
        let f = fuse_stage(&ops, 4, 0b0111, 3, 0);
        assert!(f.iter().all(|o| matches!(o, FusedOp::Plain(_))));
    }

    #[test]
    fn cost_rule() {
        let x = [[C0, C1], [C1, C0]];
        // U1 U1 CNOT: 2 uncontrolled 1-qubit gates < 4 -> unchanged
        let ops = [
            KOp::U1 {
                q: 0,
                m: h(),
                ctrl: 0,
            },
            KOp::U1 {
                q: 1,
                m: h(),
                ctrl: 0,
            },
            KOp::U1 {
                q: 1,
                m: x,
                ctrl: 1,
            },
        ];
        let f = fuse_stage(&ops, 2, 0b11, 2, 0);
        assert_eq!(
            f,
            ops.iter().map(|&o| FusedOp::Plain(o)).collect::<Vec<_>>()
        );
        // forced: one dense op
        let f = fuse_stage(&ops, 2, 0b11, 2, 1);
        assert!(matches!(&f[..], [FusedOp::Dense(d)] if d.qs == vec![0, 1]));
    }
}
