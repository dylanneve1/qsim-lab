//! k-qubit dense gate fusion for the blocked state-vector executor.
//!
//! Groups compatible consecutive gates into dense unitary operations on
//! up to `k <= 4` qubits (dimension <= 16x16), raising arithmetic intensity
//! and eliminating intermediate memory passes over cache blocks.

use crate::blocked::KOp;
use crate::gate::Mat2;
use num_complex::Complex64;

const C0: Complex64 = Complex64::new(0.0, 0.0);
const C1: Complex64 = Complex64::new(1.0, 0.0);

/// A fused dense unitary operation on a set of sorted target qubits.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseOp {
    /// Target qubits, sorted ascending.
    pub qs: Vec<usize>,
    /// Unitary matrix of size `(1 << qs.len())^2`, row-major.
    pub mat: Vec<Complex64>,
}

impl DenseOp {
    pub fn new(qs: Vec<usize>, mat: Vec<Complex64>) -> Self {
        let dim = 1 << qs.len();
        assert_eq!(mat.len(), dim * dim);
        Self { qs, mat }
    }

    /// Creates an identity dense op on `qs`.
    pub fn identity(qs: Vec<usize>) -> Self {
        let dim = 1 << qs.len();
        let mut mat = vec![C0; dim * dim];
        for i in 0..dim {
            mat[i * dim + i] = C1;
        }
        Self { qs, mat }
    }

    /// Lifts this dense op to a superset of qubits `new_qs` (sorted ascending).
    pub fn lift(&self, new_qs: &[usize]) -> DenseOp {
        if self.qs == new_qs {
            return self.clone();
        }
        let d_new = 1usize << new_qs.len();
        let mut new_mat = vec![C0; d_new * d_new];

        // Mask of positions in new_qs that are NOT in self.qs
        let mut mask_not_q = 0usize;
        let mut q_pos = Vec::with_capacity(self.qs.len());
        for (j, &q) in new_qs.iter().enumerate() {
            if let Some(pos) = self.qs.iter().position(|&x| x == q) {
                q_pos.push((j, pos));
            } else {
                mask_not_q |= 1 << j;
            }
        }

        let map_to_q = |x: usize| -> usize {
            let mut out = 0usize;
            for &(new_idx, old_idx) in &q_pos {
                out |= ((x >> new_idx) & 1) << old_idx;
            }
            out
        };

        let d_old = 1usize << self.qs.len();

        for y in 0..d_new {
            for x in 0..d_new {
                if (y & mask_not_q) == (x & mask_not_q) {
                    let y_q = map_to_q(y);
                    let x_q = map_to_q(x);
                    new_mat[y * d_new + x] = self.mat[y_q * d_old + x_q];
                }
            }
        }

        DenseOp {
            qs: new_qs.to_vec(),
            mat: new_mat,
        }
    }

    /// Multiplies `next` after `self`: returns `next * self`.
    pub fn compose(&self, next: &DenseOp) -> DenseOp {
        let mut merged_qs = self.qs.clone();
        for &q in &next.qs {
            if !merged_qs.contains(&q) {
                merged_qs.push(q);
            }
        }
        merged_qs.sort_unstable();

        let a = next.lift(&merged_qs);
        let b = self.lift(&merged_qs);

        let d = 1usize << merged_qs.len();
        let mut c_mat = vec![C0; d * d];

        for i in 0..d {
            let i_off = i * d;
            for k in 0..d {
                let aik = a.mat[i_off + k];
                if aik == C0 {
                    continue;
                }
                let k_off = k * d;
                for j in 0..d {
                    c_mat[i_off + j] += aik * b.mat[k_off + j];
                }
            }
        }

        DenseOp {
            qs: merged_qs,
            mat: c_mat,
        }
    }
}

/// Converts a 1-qubit controlled/uncontrolled unitary `U1` into a `DenseOp`.
pub fn u1_to_dense(q: usize, m: &Mat2, ctrl: usize) -> DenseOp {
    let mut qs = vec![q];
    let mut c = ctrl;
    while c != 0 {
        let cq = c.trailing_zeros() as usize;
        c &= c - 1;
        qs.push(cq);
    }
    qs.sort_unstable();

    let d = 1usize << qs.len();
    let q_pos = qs.iter().position(|&x| x == q).unwrap();
    let mut ctrl_mask = 0usize;
    for (idx, &cq) in qs.iter().enumerate() {
        if (ctrl >> cq) & 1 == 1 {
            ctrl_mask |= 1 << idx;
        }
    }

    let mut mat = vec![C0; d * d];
    for y in 0..d {
        for x in 0..d {
            if ctrl != 0 && (x & ctrl_mask) != ctrl_mask {
                // Control not active: identity
                if y == x {
                    mat[y * d + x] = C1;
                }
            } else {
                // Control active (or uncontrolled): acts on bit q_pos
                let x_no_q = x & !(1 << q_pos);
                let y_no_q = y & !(1 << q_pos);
                if x_no_q == y_no_q {
                    let bx = (x >> q_pos) & 1;
                    let by = (y >> q_pos) & 1;
                    mat[y * d + x] = m[by][bx];
                }
            }
        }
    }

    DenseOp { qs, mat }
}

/// Converts a `Swap` into a `DenseOp`.
pub fn swap_to_dense(a: usize, b: usize) -> DenseOp {
    let mut qs = vec![a, b];
    qs.sort_unstable();
    let d = 4;
    let mut mat = vec![C0; d * d];
    let pos_a = qs.iter().position(|&x| x == a).unwrap();
    let pos_b = qs.iter().position(|&x| x == b).unwrap();

    for x in 0..4 {
        let ba = (x >> pos_a) & 1;
        let bb = (x >> pos_b) & 1;
        let y = (x & !((1 << pos_a) | (1 << pos_b))) | (ba << pos_b) | (bb << pos_a);
        mat[y * d + x] = C1;
    }

    DenseOp { qs, mat }
}

/// Converts a diagonal phase term to a `DenseOp`.
pub fn phase_to_dense(mask: usize, pat: usize, f: Complex64) -> DenseOp {
    let mut qs = Vec::new();
    let mut m = mask;
    while m != 0 {
        let q = m.trailing_zeros() as usize;
        m &= m - 1;
        qs.push(q);
    }
    qs.sort_unstable();
    let d = 1usize << qs.len();

    let mut pat_in_qs = 0usize;
    for (idx, &q) in qs.iter().enumerate() {
        if (pat >> q) & 1 == 1 {
            pat_in_qs |= 1 << idx;
        }
    }

    let mut mat = vec![C0; d * d];
    for x in 0..d {
        mat[x * d + x] = if x == pat_in_qs { f } else { C1 };
    }

    DenseOp { qs, mat }
}

/// Fuses consecutive gates on up to `max_k` qubits (2 <= `max_k` <= 3).
pub fn fuse_dense_ops(ops: &[KOp], n: usize, max_k: usize) -> Vec<KOp> {
    if max_k < 2 {
        return ops.to_vec();
    }

    let mut out = Vec::with_capacity(ops.len());
    let mut active: Vec<Option<DenseOp>> = Vec::new();
    let mut wire_owner: Vec<Option<usize>> = vec![None; n];

    let flush_block = |b_idx: usize,
                       active: &mut Vec<Option<DenseOp>>,
                       wire_owner: &mut Vec<Option<usize>>,
                       out: &mut Vec<KOp>| {
        if let Some(b) = active[b_idx].take() {
            for &q in &b.qs {
                wire_owner[q] = None;
            }
            if b.qs.len() == 1 {
                let q = b.qs[0];
                let m = [[b.mat[0], b.mat[1]], [b.mat[2], b.mat[3]]];
                out.push(KOp::U1 { q, m, ctrl: 0 });
            } else {
                out.push(KOp::Dense(std::sync::Arc::new(b)));
            }
        }
    };

    for op in ops {
        // Decide how to represent this op
        let candidate = match op {
            KOp::U1 { q, m, ctrl } => {
                let n_qubits = 1 + ctrl.count_ones() as usize;
                if n_qubits <= max_k {
                    Some(u1_to_dense(*q, m, *ctrl))
                } else {
                    None
                }
            }
            KOp::Swap { a, b } => Some(swap_to_dense(*a, *b)),
            KOp::Phase { mask, pat, f } => {
                // If all qubits in mask already belong to the same active block, absorb it!
                let mut qs = Vec::new();
                let mut m = *mask;
                while m != 0 {
                    let q = m.trailing_zeros() as usize;
                    m &= m - 1;
                    qs.push(q);
                }
                let owners: std::collections::HashSet<usize> = qs
                    .iter()
                    .filter_map(|&q| wire_owner.get(q).copied().flatten())
                    .collect();
                if owners.len() == 1 {
                    let b_idx = *owners.iter().next().unwrap();
                    let b = active[b_idx].as_ref().unwrap();
                    if qs.iter().all(|q| b.qs.contains(q)) {
                        Some(phase_to_dense(*mask, *pat, *f))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            KOp::Dense(d) => {
                if d.qs.len() <= max_k {
                    Some((**d).clone())
                } else {
                    None
                }
            }
        };

        let Some(dense_op) = candidate else {
            // Cannot be absorbed as a dense op. Flush all blocks on the touched wires!
            let t = op.touches();
            let mut owners = Vec::new();
            for q in 0..n {
                if (t >> q) & 1 == 1 {
                    if let Some(b) = wire_owner[q] {
                        if !owners.contains(&b) {
                            owners.push(b);
                        }
                    }
                }
            }
            for b in owners {
                flush_block(b, &mut active, &mut wire_owner, &mut out);
            }
            out.push(op.clone());
            continue;
        };

        // Find all active blocks owning any of dense_op.qs
        let mut owners = Vec::new();
        for &q in &dense_op.qs {
            if let Some(b) = wire_owner[q] {
                if !owners.contains(&b) {
                    owners.push(b);
                }
            }
        }

        if owners.is_empty() {
            // Start a new block
            let b_idx = active.len();
            for &q in &dense_op.qs {
                wire_owner[q] = Some(b_idx);
            }
            active.push(Some(dense_op));
        } else if owners.len() == 1 {
            let b_idx = owners[0];
            let b = active[b_idx].as_ref().unwrap();
            let mut merged_qs = b.qs.clone();
            for &q in &dense_op.qs {
                if !merged_qs.contains(&q) {
                    merged_qs.push(q);
                }
            }
            if merged_qs.len() <= max_k {
                // Merge into existing block
                let composed = b.compose(&dense_op);
                for &q in &merged_qs {
                    wire_owner[q] = Some(b_idx);
                }
                active[b_idx] = Some(composed);
            } else {
                // Would exceed max_k: flush block and start fresh
                flush_block(b_idx, &mut active, &mut wire_owner, &mut out);
                let new_idx = active.len();
                for &q in &dense_op.qs {
                    wire_owner[q] = Some(new_idx);
                }
                active.push(Some(dense_op));
            }
        } else {
            // Touches multiple blocks: check if union fits in max_k
            let mut merged_qs = dense_op.qs.clone();
            for &b_idx in &owners {
                for &q in &active[b_idx].as_ref().unwrap().qs {
                    if !merged_qs.contains(&q) {
                        merged_qs.push(q);
                    }
                }
            }
            if merged_qs.len() <= max_k {
                // Merge all blocks together
                merged_qs.sort_unstable();
                let mut combined = active[owners[0]].take().unwrap();
                for &b_idx in &owners[1..] {
                    let next_b = active[b_idx].take().unwrap();
                    combined = combined.compose(&next_b);
                }
                combined = combined.compose(&dense_op);
                let target_idx = owners[0];
                for &q in &combined.qs {
                    wire_owner[q] = Some(target_idx);
                }
                active[target_idx] = Some(combined);
            } else {
                // Exceeds max_k: flush all touching blocks
                for b_idx in owners {
                    flush_block(b_idx, &mut active, &mut wire_owner, &mut out);
                }
                let new_idx = active.len();
                for &q in &dense_op.qs {
                    wire_owner[q] = Some(new_idx);
                }
                active.push(Some(dense_op));
            }
        }
    }

    // Flush all remaining active blocks
    for b_idx in 0..active.len() {
        flush_block(b_idx, &mut active, &mut wire_owner, &mut out);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_1_SQRT_2;

    #[test]
    fn hadamard_squared_is_identity() {
        let h = FRAC_1_SQRT_2;
        let h_mat = [
            [Complex64::new(h, 0.0), Complex64::new(h, 0.0)],
            [Complex64::new(h, 0.0), Complex64::new(-h, 0.0)],
        ];
        let op1 = u1_to_dense(0, &h_mat, 0);
        let op2 = u1_to_dense(0, &h_mat, 0);
        let id = op1.compose(&op2);
        assert_eq!(id.qs, vec![0]);
        assert!((id.mat[0] - C1).norm() < 1e-12);
        assert!((id.mat[1] - C0).norm() < 1e-12);
        assert!((id.mat[2] - C0).norm() < 1e-12);
        assert!((id.mat[3] - C1).norm() < 1e-12);
    }

    #[test]
    fn cnot_bell_state_matrix() {
        let h = FRAC_1_SQRT_2;
        let h_mat = [
            [Complex64::new(h, 0.0), Complex64::new(h, 0.0)],
            [Complex64::new(h, 0.0), Complex64::new(-h, 0.0)],
        ];
        let x_mat = [
            [C0, C1],
            [C1, C0],
        ];
        let h0 = u1_to_dense(0, &h_mat, 0);
        let cnot01 = u1_to_dense(1, &x_mat, 1 << 0); // ctrl=0, tgt=1
        let bell = h0.compose(&cnot01);
        assert_eq!(bell.qs, vec![0, 1]);
        // Column 0 should be (|00> + |11>)/sqrt(2), i.e. index 0 and index 3 have amplitude 1/sqrt(2)
        assert!((bell.mat[0 * 4 + 0] - Complex64::new(h, 0.0)).norm() < 1e-12);
        assert!((bell.mat[1 * 4 + 0] - C0).norm() < 1e-12);
        assert!((bell.mat[2 * 4 + 0] - C0).norm() < 1e-12);
        assert!((bell.mat[3 * 4 + 0] - Complex64::new(h, 0.0)).norm() < 1e-12);
    }
}
