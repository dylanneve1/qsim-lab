//! Matrix product state (MPS) simulation with SVD truncation.
//!
//! The state is written as a chain of tensors `A[q]` of shape
//! `(χ_left, 2, χ_right)`; an amplitude is the product of one 2D slice per
//! qubit: `<s_0 ... s_{n-1}|ψ> = A[0][s_0] A[1][s_1] ... A[n-1][s_{n-1}]`.
//! The bond dimension `χ` needed across a cut is the Schmidt rank of the
//! state across that cut, so memory is `O(n χ^2)`: product and GHZ states
//! need `χ ≤ 2` at any size, while a generic (volume-law) state needs
//! `χ = 2^{n/2}` in the middle and an MPS is then no better than a state
//! vector.
//!
//! Gates on neighbouring qubits are applied by contracting the two tensors,
//! applying the 4x4 matrix and splitting them again with an SVD; singular
//! values beyond `max_bond` (or below a relative cutoff) are dropped and the
//! discarded weight is recorded. The chain is kept in mixed-canonical form
//! (an orthogonality centre is moved with QR decompositions), which makes
//! each truncation locally optimal and its error equal to the discarded
//! weight. Gates on distant qubits are routed with SWAPs.

use crate::circuit::{check_gate, SimError, Simulator};
use crate::gate::{mat4_swap_qubits, Gate, Mat2, Mat4};

use faer::Mat;
use num_complex::Complex64;
use rand::{Rng, RngCore};

type C = Complex64;

/// One site tensor, stored as `data[(l * 2 + s) * dr + r]`.
#[derive(Clone, Debug)]
struct Site {
    dl: usize,
    dr: usize,
    data: Vec<C>,
}

impl Site {
    #[inline]
    fn at(&self, l: usize, s: usize, r: usize) -> C {
        self.data[(l * 2 + s) * self.dr + r]
    }
}

/// A matrix product state.
#[derive(Clone, Debug)]
pub struct Mps {
    n: usize,
    sites: Vec<Site>,
    /// Orthogonality centre: sites left of it are left-isometric, sites
    /// right of it are right-isometric.
    center: usize,
    max_bond: usize,
    cutoff: f64,
    /// Product of (1 - discarded weight) over all truncations: an estimate
    /// of the fidelity with the exact state.
    fidelity: f64,
    truncations: usize,
}

impl Mps {
    /// `|0...0>` on `n` qubits with bond dimension capped at `max_bond`.
    pub fn new(n: usize, max_bond: usize) -> Self {
        assert!(n >= 1 && max_bond >= 1);
        let site = Site {
            dl: 1,
            dr: 1,
            data: vec![C::new(1.0, 0.0), C::new(0.0, 0.0)],
        };
        Mps {
            n,
            sites: vec![site; n],
            center: 0,
            max_bond,
            cutoff: 1e-14,
            fidelity: 1.0,
            truncations: 0,
        }
    }

    /// Relative cutoff: singular values with `s^2 / sum s^2` below this are
    /// dropped even if the bond cap is not reached. Default `1e-14`.
    pub fn set_cutoff(&mut self, cutoff: f64) {
        self.cutoff = cutoff;
    }

    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// Resets the MPS back to `|0...0>` without reallocating site tensor buffers.
    pub fn reset_all(&mut self) {
        for site in &mut self.sites {
            site.dl = 1;
            site.dr = 1;
            site.data.clear();
            site.data.push(C::new(1.0, 0.0));
            site.data.push(C::new(0.0, 0.0));
        }
        self.center = 0;
        self.fidelity = 1.0;
        self.truncations = 0;
    }

    /// Bond dimensions between neighbouring sites (`n - 1` entries).
    pub fn bond_dims(&self) -> Vec<usize> {
        self.sites[..self.n - 1].iter().map(|s| s.dr).collect()
    }

    pub fn max_bond_dim(&self) -> usize {
        self.bond_dims().into_iter().max().unwrap_or(1)
    }

    /// Estimated fidelity with the exact (untruncated) state.
    pub fn fidelity_estimate(&self) -> f64 {
        self.fidelity
    }

    /// Number of SVDs that actually dropped weight.
    pub fn truncation_count(&self) -> usize {
        self.truncations
    }

    /// Bytes used by the site tensors.
    pub fn bytes(&self) -> usize {
        self.sites.iter().map(|s| s.data.len() * 16).sum()
    }

    // ----- canonical form ----------------------------------------------

    fn site_matrix_left(&self, i: usize) -> Mat<C> {
        // rows (l, s), cols r
        let s = &self.sites[i];
        Mat::from_fn(s.dl * 2, s.dr, |row, col| s.data[row * s.dr + col])
    }

    fn site_matrix_right(&self, i: usize) -> Mat<C> {
        // rows l, cols (s, r)
        let s = &self.sites[i];
        Mat::from_fn(s.dl, 2 * s.dr, |row, col| s.data[row * 2 * s.dr + col])
    }

    fn set_from_left(&mut self, i: usize, m: &Mat<C>) {
        let dl = m.nrows() / 2;
        let dr = m.ncols();
        let mut data = vec![C::default(); dl * 2 * dr];
        for row in 0..dl * 2 {
            for col in 0..dr {
                data[row * dr + col] = m[(row, col)];
            }
        }
        self.sites[i] = Site { dl, dr, data };
    }

    fn set_from_right(&mut self, i: usize, m: &Mat<C>) {
        let dl = m.nrows();
        let dr = m.ncols() / 2;
        let mut data = vec![C::default(); dl * 2 * dr];
        for row in 0..dl {
            for col in 0..2 * dr {
                data[row * 2 * dr + col] = m[(row, col)];
            }
        }
        self.sites[i] = Site { dl, dr, data };
    }

    /// Moves the orthogonality centre to site `to` with QR decompositions.
    fn move_center(&mut self, to: usize) {
        while self.center < to {
            let i = self.center;
            let qr = self.site_matrix_left(i).qr();
            let (q, r) = (qr.compute_thin_Q(), qr.thin_R().to_owned());
            self.set_from_left(i, &q);
            let next = &r * &self.site_matrix_right(i + 1);
            self.set_from_right(i + 1, &next);
            self.center += 1;
        }
        while self.center > to {
            let i = self.center;
            // M = L Q with Q having orthonormal rows: QR of M†.
            let qr = self.site_matrix_right(i).adjoint().qr();
            let (q, r) = (qr.compute_thin_Q(), qr.thin_R().to_owned());
            self.set_from_right(i, &q.adjoint().to_owned());
            let prev = &self.site_matrix_left(i - 1) * r.adjoint();
            self.set_from_left(i - 1, &prev);
            self.center -= 1;
        }
    }

    // ----- gates --------------------------------------------------------

    /// Applies a 2x2 unitary to qubit `q` (no change of bond dimension).
    pub fn apply_1q(&mut self, q: usize, m: &Mat2) {
        let s = &mut self.sites[q];
        let dr = s.dr;
        for l in 0..s.dl {
            for r in 0..dr {
                let a0 = s.data[(l * 2) * dr + r];
                let a1 = s.data[(l * 2 + 1) * dr + r];
                s.data[(l * 2) * dr + r] = m[0][0] * a0 + m[0][1] * a1;
                s.data[(l * 2 + 1) * dr + r] = m[1][0] * a0 + m[1][1] * a1;
            }
        }
    }

    /// Applies a 4x4 unitary to sites `i, i+1` (matrix indexed by
    /// `2*s_i + s_{i+1}`), then splits with a truncated SVD.
    fn apply_2q_adjacent(&mut self, i: usize, g: &Mat4) {
        self.move_center(i);
        let (dl, dr) = (self.sites[i].dl, self.sites[i + 1].dr);
        // theta[(l, s1), (s2, r)] = sum_m A[l, s1, m] B[m, s2, r]
        let theta = &self.site_matrix_left(i) * &self.site_matrix_right(i + 1);
        debug_assert_eq!((theta.nrows(), theta.ncols()), (dl * 2, 2 * dr));
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
                    let (s1, s2) = (o >> 1, o & 1);
                    t2[(l * 2 + s1, s2 * dr + r)] = val;
                }
            }
        }
        // faer returns singular values sorted in non-increasing order
        let svd = t2.thin_svd().expect("SVD did not converge");
        let (u, v) = (svd.U(), svd.V());
        let sv: Vec<f64> = (0..u.ncols())
            .map(|k| svd.S().column_vector()[k].re)
            .collect();
        let total: f64 = sv.iter().map(|s| s * s).sum();
        let mut keep = 0;
        let mut kept_w = 0.0;
        for &s in &sv {
            let w = s * s;
            if keep >= self.max_bond || (keep > 0 && w / total < self.cutoff) {
                break;
            }
            keep += 1;
            kept_w += w;
        }
        let discarded = ((total - kept_w) / total).max(0.0);
        if discarded > 1e-15 {
            self.fidelity *= 1.0 - discarded;
            self.truncations += 1;
        }
        let norm = kept_w.sqrt();
        let left = Mat::from_fn(dl * 2, keep, |r, c| u[(r, c)]);
        let right = Mat::from_fn(keep, 2 * dr, |r, c| v[(c, r)].conj() * (sv[r] / norm));
        self.set_from_left(i, &left);
        self.set_from_right(i + 1, &right);
        self.center = i + 1;
    }

    /// Applies a 4x4 unitary to qubits `(a, b)` (matrix indexed by
    /// `2*bit(a) + bit(b)`). Non-adjacent qubits are brought together with
    /// SWAPs and moved back afterwards.
    pub fn apply_2q(&mut self, a: usize, b: usize, m: &Mat4) {
        assert_ne!(a, b);
        let (lo, hi) = (a.min(b), a.max(b));
        let swap = Gate::Swap(0, 1).matrix_2q().expect("swap");
        // move qubit `hi` down to site lo+1
        for k in (lo + 1..hi).rev() {
            self.apply_2q_adjacent(k, &swap);
        }
        let mm = if a == lo { *m } else { mat4_swap_qubits(m) };
        self.apply_2q_adjacent(lo, &mm);
        for k in lo + 1..hi {
            self.apply_2q_adjacent(k, &swap);
        }
    }

    pub fn apply_gate(&mut self, g: &Gate) -> Result<(), SimError> {
        check_gate(g, self.n)?;
        if matches!(g, Gate::I(_)) {
            return Ok(());
        }
        if let Some(m) = g.matrix_1q() {
            let q = g.qubits()[0];
            self.apply_1q(q, &m);
        } else if let Some(m) = g.matrix_2q() {
            let qs = g.qubits();
            self.apply_2q(qs[0], qs[1], &m);
        } else {
            for h in g.decompose_to_clifford_rz() {
                self.apply_gate(&h)?;
            }
        }
        Ok(())
    }

    // ----- readout ------------------------------------------------------

    /// The amplitude `<bits|ψ>` (bit `q` of `bits` = qubit `q`), for up to
    /// 128 qubits.
    pub fn amplitude(&self, bits: u128) -> C {
        let mut v = vec![C::new(1.0, 0.0)];
        for (q, s) in self.sites.iter().enumerate() {
            let b = ((bits >> q) & 1) as usize;
            let mut nv = vec![C::default(); s.dr];
            for (l, vl) in v.iter().enumerate() {
                for (r, o) in nv.iter_mut().enumerate() {
                    *o += vl * s.at(l, b, r);
                }
            }
            v = nv;
        }
        v[0]
    }

    /// `<ψ|ψ>`, computed by contracting the chain with its conjugate.
    pub fn norm_sqr(&self) -> f64 {
        // E[l, l'] environment, start 1x1
        let mut e = Mat::<C>::from_fn(1, 1, |_, _| C::new(1.0, 0.0));
        for s in &self.sites {
            let mut ne = Mat::<C>::zeros(s.dr, s.dr);
            for p in 0..2 {
                let a = Mat::from_fn(s.dl, s.dr, |l, r| s.at(l, p, r));
                ne += a.adjoint() * &e * &a;
            }
            e = ne;
        }
        e[(0, 0)].re
    }

    /// Probability that qubit `q` reads 1.
    pub fn prob_one(&mut self, q: usize) -> f64 {
        self.move_center(q);
        let s = &self.sites[q];
        let mut p = [0.0; 2];
        for l in 0..s.dl {
            for (b, pb) in p.iter_mut().enumerate() {
                for r in 0..s.dr {
                    *pb += s.at(l, b, r).norm_sqr();
                }
            }
        }
        p[1] / (p[0] + p[1])
    }

    /// Measures qubit `q`, collapsing the state.
    pub fn measure_qubit<R: Rng + ?Sized>(&mut self, q: usize, rng: &mut R) -> bool {
        let p1 = self.prob_one(q);
        let outcome = rng.random::<f64>() < p1;
        let p = if outcome { p1 } else { 1.0 - p1 };
        let k = 1.0 / p.sqrt();
        let s = &mut self.sites[q];
        let dr = s.dr;
        for l in 0..s.dl {
            for b in 0..2 {
                for r in 0..dr {
                    let x = &mut s.data[(l * 2 + b) * dr + r];
                    *x = if (b == 1) == outcome {
                        *x * k
                    } else {
                        C::default()
                    };
                }
            }
        }
        outcome
    }

    /// Resets qubit `q` to |0>, collapsing the state.
    pub fn reset_qubit<R: Rng + ?Sized>(&mut self, q: usize, rng: &mut R) {
        if self.measure_qubit(q, rng) {
            self.apply_gate(&Gate::X(q)).expect("valid qubit");
        }
    }

    /// Draws `shots` bitstrings without collapsing the state (for up to 128
    /// qubits). With the centre at site 0 every other site is
    /// right-isometric, so conditional probabilities can be read off left
    /// to right in `O(n χ^2)` per shot.
    pub fn sample<R: Rng + ?Sized>(&mut self, shots: usize, rng: &mut R) -> Vec<u128> {
        assert!(self.n <= 128);
        self.move_center(0);
        let mut out = Vec::with_capacity(shots);
        for _ in 0..shots {
            let mut v = vec![C::new(1.0, 0.0)];
            let mut bits = 0u128;
            for (q, s) in self.sites.iter().enumerate() {
                let mut w = [vec![C::default(); s.dr], vec![C::default(); s.dr]];
                for (b, wb) in w.iter_mut().enumerate() {
                    for (l, vl) in v.iter().enumerate() {
                        for (r, o) in wb.iter_mut().enumerate() {
                            *o += vl * s.at(l, b, r);
                        }
                    }
                }
                let p0: f64 = w[0].iter().map(|z| z.norm_sqr()).sum();
                let p1: f64 = w[1].iter().map(|z| z.norm_sqr()).sum();
                let b = usize::from(rng.random::<f64>() * (p0 + p1) >= p0);
                let pb = if b == 1 { p1 } else { p0 };
                let k = 1.0 / pb.sqrt();
                v = w[b].iter().map(|z| z * k).collect();
                bits |= (b as u128) << q;
            }
            out.push(bits);
        }
        out
    }
}

impl Simulator for Mps {
    fn name(&self) -> &'static str {
        "mps"
    }
    fn num_qubits(&self) -> usize {
        self.n
    }
    fn apply(&mut self, gate: &Gate) -> Result<(), SimError> {
        self.apply_gate(gate)
    }
    fn measure(&mut self, q: usize, rng: &mut dyn RngCore) -> Result<bool, SimError> {
        if q >= self.n {
            return Err(SimError::QubitOutOfRange {
                qubit: q,
                num_qubits: self.n,
            });
        }
        Ok(self.measure_qubit(q, rng))
    }
    fn reset(&mut self, q: usize, rng: &mut dyn RngCore) -> Result<(), SimError> {
        if q >= self.n {
            return Err(SimError::QubitOutOfRange {
                qubit: q,
                num_qubits: self.n,
            });
        }
        self.reset_qubit(q, rng);
        Ok(())
    }
    fn reset_all(&mut self) -> Result<(), SimError> {
        self.reset_all();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn ghz_100_has_bond_two() {
        let n = 100;
        let mut m = Mps::new(n, 64);
        m.apply_gate(&Gate::H(0)).unwrap();
        for q in 1..n {
            m.apply_gate(&Gate::Cnot(q - 1, q)).unwrap();
        }
        assert_eq!(m.max_bond_dim(), 2);
        let h = std::f64::consts::FRAC_1_SQRT_2;
        assert!((m.amplitude(0).norm() - h).abs() < 1e-10);
        assert!((m.amplitude((1u128 << n) - 1).norm() - h).abs() < 1e-10);
        assert!(m.amplitude(1).norm() < 1e-10);
        assert!((m.norm_sqr() - 1.0).abs() < 1e-10);
        let mut rng = StdRng::seed_from_u64(4);
        for s in m.sample(20, &mut rng) {
            assert!(s == 0 || s == (1u128 << n) - 1);
        }
    }

    #[test]
    fn long_range_gate() {
        let mut m = Mps::new(5, 16);
        m.apply_gate(&Gate::H(4)).unwrap();
        m.apply_gate(&Gate::Cnot(4, 0)).unwrap();
        let h = std::f64::consts::FRAC_1_SQRT_2;
        assert!((m.amplitude(0b10001).norm() - h).abs() < 1e-10);
        assert!((m.amplitude(0).norm() - h).abs() < 1e-10);
        assert!((m.norm_sqr() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn truncation_reduces_fidelity() {
        let mut rng = StdRng::seed_from_u64(11);
        let n = 10;
        let mut m = Mps::new(n, 2);
        for _ in 0..6 {
            for q in 0..n {
                m.apply_gate(&Gate::Ry(q, rng.random::<f64>() * 3.0))
                    .unwrap();
            }
            for q in (0..n - 1).step_by(2) {
                m.apply_gate(&Gate::Cnot(q, q + 1)).unwrap();
            }
            for q in (1..n - 1).step_by(2) {
                m.apply_gate(&Gate::Cnot(q, q + 1)).unwrap();
            }
        }
        assert!(m.max_bond_dim() <= 2);
        assert!(m.fidelity_estimate() < 0.99);
        assert!(m.truncation_count() > 0);
        assert!((m.norm_sqr() - 1.0).abs() < 1e-9);
    }
}
