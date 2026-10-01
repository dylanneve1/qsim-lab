//! Fused semiclassical rounds for the permutation oracle.
//!
//! Between rounds the recycled control is `|0>`, so the state is
//! `|0> ⊗ |ψ>` and only `ψ` (the work register) needs storing. One round
//! (`H`, controlled `U`, `Phase(φ)`, `H`) maps it to
//!
//! ```text
//! |0> (ψ + e^{iφ} Uψ)/2  +  |1> (ψ − e^{iφ} Uψ)/2
//! ```
//!
//! so `P(1) = ‖ψ − e^{iφ}Uψ‖² / 4` and the collapsed, recycled state is
//! `(ψ ± e^{iφ}Uψ) / (2 √P)`. That is the same linear algebra the gate path
//! performs on `n + 1` qubits (the tests check equal distributions to
//! 1e-12), done in two passes over `2^n` amplitudes instead of ~7 passes
//! over `2^(n+1)`, and without storing the control qubit at all.
//!
//! The circuit being simulated is unchanged: `n + 1` qubits, one recycled
//! control. Only the bookkeeping of the control qubit is analytic.

use super::{mod_inverse, mul_mod, Instance, Oracle, OrderFindingState};
use crate::gate::Gate;
use crate::sparse::AmpMap;
use crate::statevector::Real;
use num_complex::{Complex, Complex64};
use num_traits::Zero;
use rayon::prelude::*;

fn not_gate_level() -> ! {
    panic!("fused states only run whole permutation-oracle rounds")
}

/// Dense work register `ψ` (`2^n` amplitudes) plus a buffer for `Uψ`.
#[derive(Clone, Debug)]
pub struct FusedDense<T: Real> {
    psi: Vec<Complex<T>>,
    v: Vec<Complex<T>>,
    /// `e^{iφ}` of the current round.
    ph: Complex<T>,
    /// `P(1)` of the current round (computed in the gather pass).
    p1: f64,
}

fn cvt<T: Real>(z: Complex64) -> Complex<T> {
    Complex::new(T::from_f64(z.re), T::from_f64(z.im))
}

impl<T: Real> FusedDense<T> {
    pub fn new(inst: &Instance) -> Self {
        assert_eq!(inst.oracle, Oracle::Permutation);
        let len = 1usize << inst.m;
        let mut psi = vec![Complex::zero(); len];
        psi[1] = Complex::new(T::one(), T::zero()); // work register |1>
        Self {
            psi,
            v: vec![Complex::zero(); len],
            ph: Complex::new(T::one(), T::zero()),
            p1: 0.0,
        }
    }

    /// The work-register amplitudes.
    pub fn work(&self) -> &[Complex<T>] {
        &self.psi
    }
}

/// `v[z] = ψ[z · inv mod N]` for `z < N`, `v[z] = ψ[z]` otherwise, with the
/// modular index advanced incrementally (no division per amplitude).
/// Returns `Σ |ψ − ph·v|²` (computed in the same pass).
fn gather<T: Real>(
    psi: &[Complex<T>],
    v: &mut [Complex<T>],
    inv: u64,
    n_mod: u64,
    ph: Complex<T>,
) -> f64 {
    let n = n_mod as usize;
    let inv = inv as usize;
    let chunk = 4096.min(v.len());
    v.par_chunks_mut(chunk)
        .enumerate()
        .map(|(ci, out)| {
            let z0 = ci * chunk;
            let mut src = if z0 < n {
                ((z0 as u128 * inv as u128) % n as u128) as usize
            } else {
                0
            };
            let mut acc = 0.0f64;
            for (o, z) in out.iter_mut().zip(z0..) {
                if z < n {
                    *o = psi[src];
                    src += inv;
                    if src >= n {
                        src -= n;
                    }
                } else {
                    *o = psi[z];
                }
                acc += (psi[z] - ph * *o).norm_sqr().to_f64();
            }
            acc
        })
        .sum()
}

impl<T: Real> OrderFindingState for FusedDense<T> {
    fn gate(&mut self, _g: &Gate) {
        not_gate_level()
    }
    fn ctrl_mul(&mut self, _m: usize, _mult: u64, _inv: u64, _n: u64) {
        not_gate_level()
    }
    fn round(&mut self, inst: &Instance, i: usize, y_low: u128) {
        let mult = inst.mults[inst.t - 1 - i];
        let phi = if y_low != 0 {
            Instance::correction(i, y_low)
        } else {
            0.0
        };
        self.ph = cvt(Complex64::from_polar(1.0, phi));
        let s = gather(
            &self.psi,
            &mut self.v,
            mod_inverse(mult, inst.n_mod),
            inst.n_mod,
            self.ph,
        );
        self.p1 = s / 4.0;
    }
    fn prob_one(&self, q: usize) -> f64 {
        assert_eq!(q, 0);
        self.p1
    }
    fn collapse(&mut self, q: usize, outcome: bool) {
        let p1 = self.prob_one(q);
        let p = if outcome { p1 } else { 1.0 - p1 };
        assert!(p > 0.0, "cannot collapse onto a zero-probability outcome");
        let ph = if outcome { -self.ph } else { self.ph };
        let k = T::from_f64(0.5 / p.sqrt());
        self.psi
            .par_iter_mut()
            .zip(self.v.par_iter())
            .for_each(|(a, &b)| *a = (*a + ph * b).scale(k));
    }
    fn reset_control(&mut self, _bit: bool) {}
    fn bytes(&self) -> usize {
        (self.psi.len() + self.v.len()) * std::mem::size_of::<Complex<T>>()
    }
    fn stored(&self) -> usize {
        self.psi.len()
    }
}

/// Sparse work register `ψ` (hash map `y -> amplitude`); a round only
/// records `U` and `φ`, `P(1)` and the collapse are computed by lookups into
/// `ψ`, so one map is rebuilt per round.
#[derive(Clone, Debug)]
pub struct FusedSparse {
    psi: AmpMap,
    ph: Complex64,
    mult: u64,
    inv: u64,
    n_mod: u64,
    p1: f64,
    peak: usize,
}

impl FusedSparse {
    pub fn new(inst: &Instance) -> Self {
        assert_eq!(inst.oracle, Oracle::Permutation);
        let mut psi = AmpMap::default();
        psi.insert(1, Complex64::new(1.0, 0.0));
        Self {
            psi,
            ph: Complex64::new(1.0, 0.0),
            mult: 1,
            inv: 1,
            n_mod: inst.n_mod,
            p1: 0.0,
            peak: 1,
        }
    }

    pub fn nnz(&self) -> usize {
        self.psi.len()
    }

    fn u(&self, y: u64) -> u64 {
        if y < self.n_mod {
            mul_mod(y, self.mult, self.n_mod)
        } else {
            y
        }
    }
    fn u_inv(&self, y: u64) -> u64 {
        if y < self.n_mod {
            mul_mod(y, self.inv, self.n_mod)
        } else {
            y
        }
    }

    /// For one stored `j`: the pair `(ψ_j, (Uψ)_j)` at `k = j`, and, if
    /// `U j` is not stored, the pair `(0, ψ_j)` at `k = U j`. Over all `j`
    /// this visits every `k` in `supp ψ ∪ U(supp ψ)` exactly once.
    fn pairs(&self, j: u64, a: Complex64) -> [(u64, Complex64, Complex64); 2] {
        let zero = Complex64::new(0.0, 0.0);
        let b = self.psi.get(&self.u_inv(j)).copied().unwrap_or(zero);
        let uj = self.u(j);
        let second = if self.psi.contains_key(&uj) {
            (u64::MAX, zero, zero) // marker: nothing to emit
        } else {
            (uj, zero, a)
        };
        [(j, a, b), second]
    }
}

impl OrderFindingState for FusedSparse {
    fn gate(&mut self, _g: &Gate) {
        not_gate_level()
    }
    fn ctrl_mul(&mut self, _m: usize, _mult: u64, _inv: u64, _n: u64) {
        not_gate_level()
    }
    fn round(&mut self, inst: &Instance, i: usize, y_low: u128) {
        self.mult = inst.mults[inst.t - 1 - i];
        self.inv = mod_inverse(self.mult, inst.n_mod);
        let phi = if y_low != 0 {
            Instance::correction(i, y_low)
        } else {
            0.0
        };
        self.ph = Complex64::from_polar(1.0, phi);
        let ph = self.ph;
        let s: f64 = self
            .psi
            .par_iter()
            .map(|(&j, &a)| {
                self.pairs(j, a)
                    .iter()
                    .filter(|e| e.0 != u64::MAX)
                    .map(|&(_, x, y)| (x - ph * y).norm_sqr())
                    .sum::<f64>()
            })
            .sum();
        self.p1 = s / 4.0;
    }
    fn prob_one(&self, q: usize) -> f64 {
        assert_eq!(q, 0);
        self.p1
    }
    fn collapse(&mut self, q: usize, outcome: bool) {
        let p1 = self.prob_one(q);
        let p = if outcome { p1 } else { 1.0 - p1 };
        assert!(p > 0.0, "cannot collapse onto a zero-probability outcome");
        let ph = if outcome { -self.ph } else { self.ph };
        let k = 0.5 / p.sqrt();
        let zero = Complex64::new(0.0, 0.0);
        let entries: Vec<(u64, Complex64)> = self
            .psi
            .par_iter()
            .flat_map_iter(|(&j, &a)| {
                self.pairs(j, a)
                    .into_iter()
                    .filter(|e| e.0 != u64::MAX)
                    .map(|(key, x, y)| (key, (x + ph * y) * k))
            })
            .filter(|e| e.1 != zero)
            .collect();
        let mut out = AmpMap::with_capacity_and_hasher(entries.len(), Default::default());
        out.extend(entries);
        self.peak = self.peak.max(out.len());
        self.psi = out;
    }
    fn reset_control(&mut self, _bit: bool) {}
    fn bytes(&self) -> usize {
        self.psi.capacity() * (8 + 16 + 1)
    }
    fn stored(&self) -> usize {
        self.peak.max(self.psi.len())
    }
}
