//! Approximate chain-sweep amplitudes: the bond register of
//! [`chain_sweep`](crate::engines::chain_sweep) held as a truncated MPS.
//!
//! The exact sweep carries a dense vector over the `≈ D/2` bond bits of the
//! current cut (`2^(D/2)` entries). Here the same compiled op stream
//! ([`SweepPlan`]) acts on a matrix product state over the register bits
//! instead, with the bond dimension capped at `χ` and singular values below
//! a relative cutoff dropped after every two-bit op. Register bit `k` sits
//! at chain position `pos[k]`; ops on two bits that are not neighbours are
//! routed with SWAPs (the permutation is kept, not undone). With the exact
//! sweep's slot allocation the register bits are, to a good approximation,
//! in time order, and almost every two-bit op acts on neighbours.
//!
//! The ops are generally not unitary (projectors, unnormalised Hadamards).
//! The MPS is kept normalised in mixed-canonical form with the norm in a
//! separate log-scale, so every truncation is locally optimal and its
//! discarded weight is exact. Truncation is a projection, not a
//! renormalisation: the amplitude returned is that of the projected state.
//! [`MpsSweepResult::fid_est`] is the product of the kept weights.

use crate::engines::blocked::KOp;
use crate::engines::chain_sweep::SweepPlan;
use crate::engines::mps::robust_thin_svd;
use crate::gate::{Mat2, Mat4};
use faer::Mat;
use num_complex::Complex64;

type C = Complex64;
const C0: C = C::new(0.0, 0.0);
const C1: C = C::new(1.0, 0.0);

/// One site tensor, stored as `data[(l * 2 + s) * dr + r]`.
#[derive(Clone, Debug)]
struct Site {
    dl: usize,
    dr: usize,
    data: Vec<C>,
}

/// Counters of a [`BoundaryMps`] run.
#[derive(Clone, Copy, Debug, Default)]
pub struct MpsCounters {
    /// Two-site SVDs (including routing SWAPs).
    pub svds: usize,
    /// Routing SWAPs.
    pub swaps: usize,
    /// SVDs that dropped weight above the cutoff because of `χ`.
    pub truncations: usize,
    /// Largest bond dimension seen.
    pub peak_chi: usize,
    /// Largest total site-tensor size seen (bytes, complex f64).
    pub peak_bytes: usize,
}

/// A register of `width` bits as an MPS, starting in `|0..0>`.
#[derive(Clone, Debug)]
pub struct BoundaryMps {
    sites: Vec<Site>,
    /// Chain position of each register bit.
    pos: Vec<usize>,
    /// Register bit at each chain position.
    bit_at: Vec<usize>,
    center: usize,
    chi: usize,
    cutoff: f64,
    /// `ln` of the norm factor carried outside the (normalised) tensors.
    log_scale: f64,
    /// Product of kept weights of all truncations.
    fid: f64,
    /// The state became exactly zero.
    zero: bool,
    /// Counters.
    pub counters: MpsCounters,
}

impl BoundaryMps {
    /// `|0..0>` on `width` bits, bond cap `chi`, relative SVD cutoff `cutoff`
    /// (singular values with `s^2 / Σ s^2 < cutoff` are dropped).
    pub fn new(width: usize, chi: usize, cutoff: f64) -> Self {
        let site = Site {
            dl: 1,
            dr: 1,
            data: vec![C1, C0],
        };
        BoundaryMps {
            sites: vec![site; width.max(1)],
            pos: (0..width.max(1)).collect(),
            bit_at: (0..width.max(1)).collect(),
            center: 0,
            chi: chi.max(1),
            cutoff,
            log_scale: 0.0,
            fid: 1.0,
            zero: false,
            counters: MpsCounters::default(),
        }
    }

    /// Product of the kept weights so far (truncation fidelity estimate).
    pub fn fid_est(&self) -> f64 {
        self.fid
    }

    /// Bond dimensions along the chain.
    pub fn bond_dims(&self) -> Vec<usize> {
        let w = self.sites.len();
        self.sites[..w - 1].iter().map(|s| s.dr).collect()
    }

    /// Register bits in chain order.
    pub fn order(&self) -> &[usize] {
        &self.bit_at
    }

    fn bytes(&self) -> usize {
        self.sites.iter().map(|s| s.data.len() * 16).sum()
    }

    fn left_mat(&self, i: usize) -> Mat<C> {
        let s = &self.sites[i];
        Mat::from_fn(s.dl * 2, s.dr, |r, c| s.data[r * s.dr + c])
    }

    fn right_mat(&self, i: usize) -> Mat<C> {
        let s = &self.sites[i];
        Mat::from_fn(s.dl, 2 * s.dr, |r, c| s.data[r * 2 * s.dr + c])
    }

    fn set_left(&mut self, i: usize, m: &Mat<C>) {
        let (dl, dr) = (m.nrows() / 2, m.ncols());
        let mut data = Vec::with_capacity(dl * 2 * dr);
        for r in 0..dl * 2 {
            for c in 0..dr {
                data.push(m[(r, c)]);
            }
        }
        self.sites[i] = Site { dl, dr, data };
    }

    fn set_right(&mut self, i: usize, m: &Mat<C>) {
        let (dl, dr) = (m.nrows(), m.ncols() / 2);
        let mut data = Vec::with_capacity(dl * 2 * dr);
        for r in 0..dl {
            for c in 0..2 * dr {
                data.push(m[(r, c)]);
            }
        }
        self.sites[i] = Site { dl, dr, data };
    }

    fn move_center(&mut self, to: usize) {
        while self.center < to {
            let i = self.center;
            let qr = self.left_mat(i).qr();
            let (q, r) = (qr.compute_thin_Q(), qr.thin_R().to_owned());
            self.set_left(i, &q);
            let next = &r * &self.right_mat(i + 1);
            self.set_right(i + 1, &next);
            self.center += 1;
        }
        while self.center > to {
            let i = self.center;
            let qr = self.right_mat(i).adjoint().qr();
            let (q, r) = (qr.compute_thin_Q(), qr.thin_R().to_owned());
            self.set_right(i, &q.adjoint().to_owned());
            let prev = &self.left_mat(i - 1) * r.adjoint();
            self.set_left(i - 1, &prev);
            self.center -= 1;
        }
    }

    /// Normalises the centre tensor, moving its norm into the log-scale.
    fn renorm_center(&mut self) {
        let s = &mut self.sites[self.center];
        let n2: f64 = s.data.iter().map(|z| z.norm_sqr()).sum();
        if n2 == 0.0 || !n2.is_finite() {
            self.zero = true;
            return;
        }
        let inv = 1.0 / n2.sqrt();
        for z in &mut s.data {
            *z *= inv;
        }
        self.log_scale += 0.5 * n2.ln();
    }

    /// Applies a 2x2 matrix (any, not necessarily unitary) to register bit `q`.
    pub fn apply_1(&mut self, q: usize, m: &Mat2) {
        if self.zero {
            return;
        }
        let p = self.pos[q];
        self.move_center(p);
        let s = &mut self.sites[p];
        let dr = s.dr;
        for l in 0..s.dl {
            for r in 0..dr {
                let a0 = s.data[(l * 2) * dr + r];
                let a1 = s.data[(l * 2 + 1) * dr + r];
                s.data[(l * 2) * dr + r] = m[0][0] * a0 + m[0][1] * a1;
                s.data[(l * 2 + 1) * dr + r] = m[1][0] * a0 + m[1][1] * a1;
            }
        }
        self.renorm_center();
    }

    /// 4x4 matrix on chain positions `i, i+1` (index `2*s_i + s_{i+1}`),
    /// then a truncated SVD; the centre ends at `i+1`.
    fn apply_adjacent(&mut self, i: usize, g: &Mat4) {
        self.move_center(i);
        let (dl, dr) = (self.sites[i].dl, self.sites[i + 1].dr);
        let theta = &self.left_mat(i) * &self.right_mat(i + 1);
        let mut t2 = Mat::<C>::zeros(dl * 2, 2 * dr);
        for l in 0..dl {
            for r in 0..dr {
                let v = [
                    theta[(l * 2, r)],
                    theta[(l * 2, dr + r)],
                    theta[(l * 2 + 1, r)],
                    theta[(l * 2 + 1, dr + r)],
                ];
                for (o, row) in g.iter().enumerate() {
                    let val = row[0] * v[0] + row[1] * v[1] + row[2] * v[2] + row[3] * v[3];
                    t2[(l * 2 + (o >> 1), (o & 1) * dr + r)] = val;
                }
            }
        }
        let (u, sv, v) = svd(&t2);
        self.counters.svds += 1;
        let total: f64 = sv.iter().map(|s| s * s).sum();
        if total == 0.0 || !total.is_finite() {
            self.zero = true;
            return;
        }
        let mut keep = 0;
        let mut kept = 0.0;
        let mut cut_by_chi = false;
        for &s in &sv {
            let w = s * s;
            if keep > 0 && w / total < self.cutoff {
                break;
            }
            if keep >= self.chi {
                cut_by_chi = true;
                break;
            }
            keep += 1;
            kept += w;
        }
        if cut_by_chi {
            self.counters.truncations += 1;
        }
        self.fid *= (kept / total).min(1.0);
        self.log_scale += 0.5 * kept.ln();
        let norm = kept.sqrt();
        let left = Mat::from_fn(dl * 2, keep, |r, c| u[(r, c)]);
        let right = Mat::from_fn(keep, 2 * dr, |r, c| v[(c, r)].conj() * (sv[r] / norm));
        self.set_left(i, &left);
        self.set_right(i + 1, &right);
        self.center = i + 1;
        self.counters.peak_chi = self.counters.peak_chi.max(keep);
        let b = self.bytes();
        self.counters.peak_bytes = self.counters.peak_bytes.max(b);
    }

    fn swap_positions(&mut self, i: usize) {
        let mut g = [[C0; 4]; 4];
        g[0][0] = C1;
        g[1][2] = C1;
        g[2][1] = C1;
        g[3][3] = C1;
        self.apply_adjacent(i, &g);
        self.counters.swaps += 1;
        let (a, b) = (self.bit_at[i], self.bit_at[i + 1]);
        self.bit_at.swap(i, i + 1);
        self.pos[a] = i + 1;
        self.pos[b] = i;
    }

    /// Applies a 4x4 matrix to register bits `(a, b)` (index `2*bit(a) +
    /// bit(b)`), routing `b` next to `a` with SWAPs if needed.
    pub fn apply_2(&mut self, a: usize, b: usize, m: &Mat4) {
        if self.zero {
            return;
        }
        assert_ne!(a, b);
        // move b towards a, from whichever side it is on
        while self.pos[b] > self.pos[a] + 1 {
            let p = self.pos[b];
            self.swap_positions(p - 1);
        }
        while self.pos[b] + 1 < self.pos[a] {
            let p = self.pos[b];
            self.swap_positions(p);
        }
        let (pa, pb) = (self.pos[a], self.pos[b]);
        if pa < pb {
            self.apply_adjacent(pa, m);
        } else {
            self.apply_adjacent(pb, &crate::gate::mat4_swap_qubits(m));
        }
    }

    /// Applies one executor op.
    pub fn apply_kop(&mut self, op: &KOp) {
        match *op {
            KOp::U1 { q, m, ctrl } => {
                if ctrl == 0 {
                    self.apply_1(q, &m);
                } else {
                    assert_eq!(ctrl.count_ones(), 1, "multi-controlled op");
                    let c = ctrl.trailing_zeros() as usize;
                    let mut g = [[C0; 4]; 4];
                    g[0][0] = C1;
                    g[1][1] = C1;
                    g[2][2] = m[0][0];
                    g[2][3] = m[0][1];
                    g[3][2] = m[1][0];
                    g[3][3] = m[1][1];
                    self.apply_2(c, q, &g);
                }
            }
            KOp::Phase { mask, pat, f } => match mask.count_ones() {
                1 => {
                    let q = mask.trailing_zeros() as usize;
                    let d = if pat == 0 {
                        [[f, C0], [C0, C1]]
                    } else {
                        [[C1, C0], [C0, f]]
                    };
                    self.apply_1(q, &d);
                }
                2 => {
                    let a = mask.trailing_zeros() as usize;
                    let b = (mask & !(1 << a)).trailing_zeros() as usize;
                    let o = 2 * ((pat >> a) & 1) + ((pat >> b) & 1);
                    let mut g = [[C0; 4]; 4];
                    for (k, row) in g.iter_mut().enumerate() {
                        row[k] = if k == o { f } else { C1 };
                    }
                    self.apply_2(a, b, &g);
                }
                k => panic!("phase on {k} bits"),
            },
            KOp::Swap { a, b } => {
                // relabel: exchange the chain positions of the two bits
                let (pa, pb) = (self.pos[a], self.pos[b]);
                self.pos.swap(a, b);
                self.bit_at[pa] = b;
                self.bit_at[pb] = a;
            }
        }
    }

    /// `<0..0|ψ>` including the carried scale.
    pub fn zero_amplitude(&self) -> C {
        if self.zero {
            return C0;
        }
        let mut v = vec![C1];
        for s in &self.sites {
            let mut nv = vec![C0; s.dr];
            for (l, &vl) in v.iter().enumerate() {
                for (r, z) in nv.iter_mut().enumerate() {
                    *z += vl * s.data[(l * 2) * s.dr + r];
                }
            }
            v = nv;
        }
        v[0] * self.log_scale.exp()
    }

    /// `ln` of the carried norm factor (the tensors are normalised).
    pub fn log_scale(&self) -> f64 {
        self.log_scale
    }

    /// Whether the state is exactly zero.
    pub fn is_zero(&self) -> bool {
        self.zero
    }

    /// The tensors as a dense vector over the register bits (bit `k` of the
    /// index = register bit `k`), including the carried scale. For tests.
    pub fn to_dense(&self) -> Vec<C> {
        let w = self.sites.len();
        if self.zero {
            return vec![C0; 1 << w];
        }
        // contract left to right: rows = chain-order index (position 0 = msb)
        let mut cur: Vec<C> = vec![C1]; // [idx][r]
        let mut dr = 1;
        for s in &self.sites {
            let rows = cur.len() / dr;
            let mut next = vec![C0; rows * 2 * s.dr];
            for i in 0..rows {
                for l in 0..dr {
                    let a = cur[i * dr + l];
                    if a == C0 {
                        continue;
                    }
                    for p in 0..2 {
                        for r in 0..s.dr {
                            next[((i * 2 + p) * s.dr) + r] += a * s.data[(l * 2 + p) * s.dr + r];
                        }
                    }
                }
            }
            cur = next;
            dr = s.dr;
        }
        let sc = self.log_scale.exp();
        let mut out = vec![C0; 1 << w];
        for (i, &z) in cur.iter().enumerate() {
            let mut idx = 0;
            for p in 0..w {
                if (i >> (w - 1 - p)) & 1 == 1 {
                    idx |= 1 << self.bit_at[p];
                }
            }
            out[idx] = z * sc;
        }
        out
    }
}

/// Thin SVD for the truncation step. faer's SVD occasionally fails to
/// converge on the exactly degenerate matrices this circuit produces (seen
/// at D = 36 on `128x128` matrices, even after `mps::robust_thin_svd`'s
/// retries and after a `1e-13` perturbation). The fallback diagonalises the
/// smaller Gram matrix (`M M†` or `M† M`) instead: singular values then
/// carry an absolute error of about `1e-16 s_max^2` in `s^2`, far below
/// the truncation error, and vectors with `s = 0` are zero (they are always
/// below the cutoff).
fn svd(m: &Mat<C>) -> (Mat<C>, Vec<f64>, Mat<C>) {
    let finite = |s: &[f64], u: &Mat<C>, v: &Mat<C>| {
        s.iter().all(|x| x.is_finite())
            && (0..u.nrows()).all(|r| (0..u.ncols()).all(|c| u[(r, c)].re.is_finite()))
            && (0..v.nrows()).all(|r| (0..v.ncols()).all(|c| v[(r, c)].re.is_finite()))
    };
    if let Ok(svd) = m.thin_svd() {
        let k = svd.S().column_vector().nrows();
        let s: Vec<f64> = (0..k).map(|i| svd.S().column_vector()[i].re).collect();
        let (u, v) = (svd.U().to_owned(), svd.V().to_owned());
        if finite(&s, &u, &v) {
            return (u, s, v);
        }
    }
    gram_svd(m).unwrap_or_else(|| robust_thin_svd(m))
}

/// Thin SVD from the eigendecomposition of the smaller Gram matrix.
fn gram_svd(m: &Mat<C>) -> Option<(Mat<C>, Vec<f64>, Mat<C>)> {
    let wide = m.nrows() <= m.ncols();
    let a = if wide {
        m.to_owned()
    } else {
        m.adjoint().to_owned()
    };
    let g = &a * a.adjoint();
    let eig = g.self_adjoint_eigen(faer::Side::Lower).ok()?;
    let k = a.nrows();
    // eigenvalues ascending: reverse
    let order: Vec<usize> = (0..k).rev().collect();
    let s: Vec<f64> = order
        .iter()
        .map(|&i| eig.S().column_vector()[i].re.max(0.0).sqrt())
        .collect();
    let left = Mat::from_fn(k, k, |r, c| eig.U()[(r, order[c])]);
    // right vectors: a† left_c / s_c
    let ah_l = a.adjoint() * &left;
    let smax = s.first().copied().unwrap_or(0.0);
    let right = Mat::from_fn(a.ncols(), k, |r, c| {
        if s[c] > 1e-150 * smax.max(1e-300) {
            ah_l[(r, c)] / s[c]
        } else {
            C0
        }
    });
    Some(if wide {
        (left, s, right)
    } else {
        (right, s, left)
    })
}

/// Result of an approximate sweep.
#[derive(Clone, Copy, Debug)]
pub struct MpsSweepResult {
    /// Approximate amplitude (projection of the exact one onto the kept
    /// subspaces, times the plan's scale).
    pub amp: C,
    /// Product of the kept weights over all truncations.
    pub fid_est: f64,
    /// Counters.
    pub counters: MpsCounters,
}

/// Runs a compiled sweep plan on a boundary MPS with bond cap `chi`.
pub fn amplitude_mps(plan: &SweepPlan, chi: usize, cutoff: f64) -> MpsSweepResult {
    let mut m = BoundaryMps::new(plan.width, chi, cutoff);
    for op in &plan.ops {
        m.apply_kop(op);
        if m.is_zero() {
            break;
        }
    }
    MpsSweepResult {
        amp: m.zero_amplitude() * plan.scale,
        fid_est: m.fid_est(),
        counters: m.counters,
    }
}

/// Runs the plan's ops for qubits `0..upto` on a boundary MPS and returns
/// it (e.g. to inspect bond dimensions after each qubit).
pub fn run_prefix(plan: &SweepPlan, upto: usize, chi: usize, cutoff: f64) -> BoundaryMps {
    let mut m = BoundaryMps::new(plan.width, chi, cutoff);
    for op in &plan.ops[..plan.qubit_ops[upto]] {
        m.apply_kop(op);
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gram_svd_reconstructs() {
        for (r, c) in [(5, 9), (9, 5), (6, 6)] {
            let m = Mat::from_fn(r, c, |i, j| {
                let x = ((i * 31 + j * 17 + 3) % 23) as f64 / 7.0 - 1.5;
                C::new(x, ((i * 13 + j * 5) % 11) as f64 / 5.0 - 1.0)
            });
            let (u, s, v) = gram_svd(&m).unwrap();
            assert!(s.windows(2).all(|w| w[0] >= w[1]));
            for i in 0..r {
                for j in 0..c {
                    let z: C = (0..s.len())
                        .map(|k| u[(i, k)] * s[k] * v[(j, k)].conj())
                        .sum();
                    assert!((z - m[(i, j)]).norm() < 1e-10, "{r}x{c} ({i},{j})");
                }
            }
        }
    }
}
