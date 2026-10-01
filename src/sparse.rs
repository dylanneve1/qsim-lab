//! Exact sparse state vector: a hash map from basis index to amplitude.
//!
//! Memory and time are `O(#nonzero amplitudes)` instead of `O(2^n)`. That
//! only pays off for states that stay sparse in the computational basis,
//! e.g. the work register of Shor's order finding (support `{a^k mod N}`)
//! or circuits made of classical reversible gates (X, CNOT, Toffoli, SWAP)
//! with few superposed qubits. A dense layer of Hadamards makes it
//! `2^n` entries with hash-map overhead, i.e. much worse than
//! [`crate::StateVector`].
//!
//! The arithmetic per amplitude is the same as the dense kernels'
//! (`m00 a0 + m01 a1`, ...), so results agree with the dense f64 state to
//! rounding. Amplitudes that become exactly `0.0` are dropped; nothing else
//! is pruned (no thresholds), so the representation is exact.

use crate::circuit::{check_gate, SimError, Simulator};
use crate::gate::{Gate, Mat2, Mat4};
use num_complex::Complex64;
use rand::{Rng, RngCore};
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// A fast, non-cryptographic hasher for `u64` keys (one multiply + xor
/// shift). Basis indices are adversarial-free, so DoS resistance is moot.
#[derive(Default, Clone, Copy)]
pub struct IndexHasher(u64);

impl Hasher for IndexHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
    }
    fn write_u64(&mut self, k: u64) {
        let x = k.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.0 = x ^ (x >> 29);
    }
}

pub type AmpMap = HashMap<u64, Complex64, BuildHasherDefault<IndexHasher>>;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Sparse exact state on up to 64 qubits.
#[derive(Clone, Debug)]
pub struct SparseState {
    n: usize,
    amps: AmpMap,
    /// Largest number of stored amplitudes seen so far.
    peak: usize,
}

fn map_with_capacity(c: usize) -> AmpMap {
    AmpMap::with_capacity_and_hasher(c, Default::default())
}

impl SparseState {
    /// `|0...0>` on `n <= 64` qubits.
    pub fn new(n: usize) -> Self {
        Self::basis_state(n, 0)
    }

    /// The basis state `|index>`.
    pub fn basis_state(n: usize, index: u64) -> Self {
        assert!(n <= 64, "SparseState supports at most 64 qubits");
        assert!(n == 64 || index >> n == 0, "basis index out of range");
        let mut amps = map_with_capacity(1);
        amps.insert(index, Complex64::new(1.0, 0.0));
        Self { n, amps, peak: 1 }
    }

    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// Number of stored (non-zero) amplitudes.
    pub fn nnz(&self) -> usize {
        self.amps.len()
    }

    /// Largest [`SparseState::nnz`] seen since construction.
    pub fn peak_nnz(&self) -> usize {
        self.peak.max(self.amps.len())
    }

    fn note_peak(&mut self) {
        self.peak = self.peak.max(self.amps.len());
    }

    /// Rough heap footprint: `capacity * (key + value + 1 control byte)`.
    pub fn bytes(&self) -> usize {
        self.amps.capacity() * (8 + 16 + 1)
    }

    pub fn amplitude(&self, index: u64) -> Complex64 {
        self.amps.get(&index).copied().unwrap_or(ZERO)
    }

    /// Iterator over the stored `(index, amplitude)` pairs (unordered).
    pub fn iter(&self) -> impl Iterator<Item = (u64, Complex64)> + '_ {
        self.amps.iter().map(|(&k, &v)| (k, v))
    }

    pub fn norm_sqr(&self) -> f64 {
        self.amps.values().map(|a| a.norm_sqr()).sum()
    }

    /// Dense copy of the amplitudes (for cross-checks on small `n`).
    pub fn to_dense(&self) -> Vec<Complex64> {
        assert!(self.n <= 30);
        let mut v = vec![ZERO; 1 << self.n];
        for (&k, &a) in &self.amps {
            v[k as usize] = a;
        }
        v
    }

    fn check_q(&self, q: usize) {
        assert!(q < self.n, "qubit {q} out of range for {} qubits", self.n);
    }

    /// Multiplies amplitudes whose bit `q` is 0 by `d0` and 1 by `d1`.
    pub fn apply_diagonal_1q(&mut self, q: usize, d0: Complex64, d1: Complex64) {
        self.check_q(q);
        let bit = 1u64 << q;
        let one = Complex64::new(1.0, 0.0);
        for (k, a) in self.amps.iter_mut() {
            let d = if k & bit != 0 { d1 } else { d0 };
            if d != one {
                *a *= d;
            }
        }
        self.amps.retain(|_, a| *a != ZERO);
    }

    /// Applies a general single-qubit matrix to qubit `q`.
    pub fn apply_1q_matrix(&mut self, q: usize, m: &Mat2) {
        self.check_q(q);
        let bit = 1u64 << q;
        let old = std::mem::take(&mut self.amps);
        let mut out = map_with_capacity(old.len() * 2);
        for (&k, &a) in &old {
            let k0 = k & !bit;
            let k1 = k | bit;
            // each pair is visited once: from k0 if it is stored, else from k1
            if k & bit != 0 && old.contains_key(&k0) {
                continue;
            }
            let (a0, a1) = if k & bit == 0 {
                (a, old.get(&k1).copied().unwrap_or(ZERO))
            } else {
                (ZERO, a)
            };
            let b0 = m[0][0] * a0 + m[0][1] * a1;
            let b1 = m[1][0] * a0 + m[1][1] * a1;
            if b0 != ZERO {
                out.insert(k0, b0);
            }
            if b1 != ZERO {
                out.insert(k1, b1);
            }
        }
        self.amps = out;
        self.note_peak();
    }

    /// Applies a general two-qubit matrix to `(a, b)` (`a` is the more
    /// significant bit of the 4x4 matrix index, as in [`Gate::matrix_2q`]).
    pub fn apply_2q_matrix(&mut self, qa: usize, qb: usize, m: &Mat4) {
        self.check_q(qa);
        self.check_q(qb);
        let (ba, bb) = (1u64 << qa, 1u64 << qb);
        let old = std::mem::take(&mut self.amps);
        let mut out = map_with_capacity(old.len() * 2);
        let idx = |base: u64, s: usize| -> u64 {
            base | if s & 2 != 0 { ba } else { 0 } | if s & 1 != 0 { bb } else { 0 }
        };
        for &k in old.keys() {
            let base = k & !(ba | bb);
            // visit each group of four once, from its first stored member
            let first = (0..4)
                .map(|s| idx(base, s))
                .find(|i| old.contains_key(i))
                .expect("k itself is stored");
            if first != k {
                continue;
            }
            let v: [Complex64; 4] =
                std::array::from_fn(|s| old.get(&idx(base, s)).copied().unwrap_or(ZERO));
            for (r, row) in m.iter().enumerate() {
                let x = row[0] * v[0] + row[1] * v[1] + row[2] * v[2] + row[3] * v[3];
                if x != ZERO {
                    out.insert(idx(base, r), x);
                }
            }
        }
        self.amps = out;
        self.note_peak();
    }

    /// Applies a classical reversible map on basis indices. `f` must be a
    /// bijection on the stored support (checked: a collision panics).
    pub fn apply_permutation(&mut self, f: impl Fn(u64) -> u64) {
        let old = std::mem::take(&mut self.amps);
        let mut out = map_with_capacity(old.len());
        for (k, a) in old {
            let j = f(k);
            assert!(
                out.insert(j, a).is_none(),
                "apply_permutation: f is not injective"
            );
        }
        self.amps = out;
    }

    /// Parallel classical reversible map on basis indices using Rayon.
    pub fn apply_permutation_par(&mut self, f: impl Fn(u64) -> u64 + Sync + Send) {
        let old = std::mem::take(&mut self.amps);
        let pairs: Vec<(u64, Complex64)> = old
            .into_par_iter()
            .map(|(k, a)| {
                let j = f(k);
                (j, a)
            })
            .collect();
        let mut out = map_with_capacity(pairs.len());
        for (j, a) in pairs {
            assert!(
                out.insert(j, a).is_none(),
                "apply_permutation_par: f is not injective"
            );
        }
        self.amps = out;
    }

    /// In-place permutation for maps that fix most keys: only the keys for
    /// which `f(k) != k` are moved.
    fn permute_sparse(&mut self, f: impl Fn(u64) -> u64) {
        let moved: Vec<(u64, u64, Complex64)> = self
            .amps
            .iter()
            .filter_map(|(&k, &a)| {
                let j = f(k);
                (j != k).then_some((k, j, a))
            })
            .collect();
        for &(k, _, _) in &moved {
            self.amps.remove(&k);
        }
        for (_, j, a) in moved {
            assert!(
                self.amps.insert(j, a).is_none(),
                "permutation is not injective"
            );
        }
    }

    /// Probability that measuring qubit `q` gives 1.
    pub fn prob_one(&self, q: usize) -> f64 {
        self.check_q(q);
        let bit = 1u64 << q;
        self.amps
            .iter()
            .filter(|(k, _)| *k & bit != 0)
            .map(|(_, a)| a.norm_sqr())
            .sum()
    }

    /// Projects qubit `q` onto `outcome` and renormalises; returns the
    /// probability of that outcome.
    pub fn collapse(&mut self, q: usize, outcome: bool) -> f64 {
        let p1 = self.prob_one(q);
        let p = if outcome { p1 } else { 1.0 - p1 };
        assert!(p > 0.0, "cannot collapse onto a zero-probability outcome");
        let bit = 1u64 << q;
        let want = if outcome { bit } else { 0 };
        let k = 1.0 / p.sqrt();
        self.amps.retain(|i, a| {
            *a *= k;
            i & bit == want
        });
        p
    }

    pub fn measure_qubit<R: Rng + ?Sized>(&mut self, q: usize, rng: &mut R) -> bool {
        let p1 = self.prob_one(q);
        let outcome = rng.random::<f64>() < p1;
        self.collapse(q, outcome);
        outcome
    }

    /// Applies a gate in place.
    pub fn apply_gate(&mut self, g: &Gate) -> Result<(), SimError> {
        check_gate(g, self.n)?;
        let b = |q: usize| 1u64 << q;
        match *g {
            Gate::I(_) => {}
            Gate::X(q) => {
                let m = b(q);
                self.apply_permutation(|k| k ^ m);
            }
            Gate::Cnot(c, t) => {
                let (mc, mt) = (b(c), b(t));
                self.permute_sparse(|k| if k & mc != 0 { k ^ mt } else { k });
            }
            Gate::Ccx(c1, c2, t) => {
                let (mc, mt) = (b(c1) | b(c2), b(t));
                self.permute_sparse(|k| if k & mc == mc { k ^ mt } else { k });
            }
            Gate::Swap(x, y) => {
                let (mx, my) = (b(x), b(y));
                self.permute_sparse(|k| {
                    if ((k & mx != 0) as u8) ^ ((k & my != 0) as u8) == 1 {
                        k ^ mx ^ my
                    } else {
                        k
                    }
                });
            }
            Gate::Cz(x, y) => self.apply_cphase(x, y, Complex64::new(-1.0, 0.0)),
            Gate::CPhase(x, y, th) => self.apply_cphase(x, y, Complex64::from_polar(1.0, th)),
            Gate::ISwap(..) | Gate::ISwapdg(..) => {
                let qs = g.qubits();
                let m = g.matrix_2q().expect("2q gate");
                // matrix_2q is in terms of the gate's own (a, b) order
                self.apply_2q_matrix(qs[0], qs[1], &m);
            }
            ref g1 => {
                let q = g1.qubits()[0];
                if let Some((d0, d1)) = g1.diagonal_1q() {
                    self.apply_diagonal_1q(q, d0, d1);
                } else {
                    let m = g1.matrix_1q().expect("single-qubit gate");
                    self.apply_1q_matrix(q, &m);
                }
            }
        }
        Ok(())
    }

    fn apply_cphase(&mut self, x: usize, y: usize, ph: Complex64) {
        let m = (1u64 << x) | (1u64 << y);
        for (k, a) in self.amps.iter_mut() {
            if k & m == m {
                *a *= ph;
            }
        }
    }

    /// Applies every gate of a measurement-free circuit.
    pub fn apply_circuit(&mut self, c: &crate::Circuit) -> Result<(), SimError> {
        for g in c.gates() {
            self.apply_gate(g)?;
        }
        Ok(())
    }
}

impl Simulator for SparseState {
    fn name(&self) -> &'static str {
        "sparse"
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
    fn reset_all(&mut self) -> Result<(), SimError> {
        *self = Self::new(self.n);
        Ok(())
    }
}
