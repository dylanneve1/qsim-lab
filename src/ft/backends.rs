//! Physical backends for [`crate::ft::machine::Machine`].

use super::core::{PX, PZ};
use super::machine::Phys;
use crate::gate::Gate;
use crate::statevector::StateVector;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Exact Pauli frame: the physical state is `F · |ideal⟩` with `F` a Pauli
/// (one byte per qubit, bit 0 = X, bit 1 = Z; global phases dropped). Valid
/// for any ideal state (including encoded magic states) because every
/// physical operation is Clifford, a Pauli measurement, or a preparation.
/// Measurements return the flip relative to the ideal outcome.
#[derive(Clone, Debug, Default)]
pub struct FrameBackend {
    pub frame: Vec<u8>,
}

impl Phys for FrameBackend {
    fn ensure(&mut self, n: usize) {
        if self.frame.len() < n {
            self.frame.resize(n, 0);
        }
    }
    #[inline]
    fn prep0(&mut self, q: usize) {
        self.frame[q] = 0;
    }
    #[inline]
    fn prep_plus(&mut self, q: usize) {
        self.frame[q] = 0;
    }
    #[inline]
    fn prep_t(&mut self, q: usize) {
        self.frame[q] = 0;
    }
    #[inline]
    fn h(&mut self, q: usize) {
        let f = self.frame[q];
        self.frame[q] = ((f & 1) << 1) | (f >> 1);
    }
    #[inline]
    fn s(&mut self, q: usize) {
        let f = self.frame[q];
        self.frame[q] = f ^ ((f & 1) << 1);
    }
    #[inline]
    fn sdg(&mut self, q: usize) {
        self.s(q)
    }
    #[inline]
    fn cnot(&mut self, c: usize, t: usize) {
        let fc = self.frame[c];
        let ft = self.frame[t];
        self.frame[t] = ft ^ (fc & 1);
        self.frame[c] = fc ^ (ft & 2);
    }
    #[inline]
    fn meas_z(&mut self, q: usize) -> bool {
        self.frame[q] & 1 == 1
    }
    #[inline]
    fn meas_x(&mut self, q: usize) -> bool {
        self.frame[q] & 2 == 2
    }
    #[inline]
    fn pauli(&mut self, q: usize, code: u8) {
        self.frame[q] ^= code;
    }
    fn frame_ref(&self) -> Option<&[u8]> {
        Some(&self.frame)
    }
}

/// Dense state vector with real (Born-rule) measurements, for validating the
/// frame model on small cases. Fixed capacity.
pub struct DenseBackend {
    pub sv: StateVector<f64>,
    pub rng: StdRng,
    cap: usize,
}

impl DenseBackend {
    pub fn new(cap: usize, seed: u64) -> Self {
        DenseBackend {
            sv: StateVector::new(cap),
            rng: StdRng::seed_from_u64(seed),
            cap,
        }
    }
    fn g(&mut self, g: Gate) {
        self.sv.apply_gate(&g).unwrap();
    }
    fn reset(&mut self, q: usize) {
        self.sv.reset_qubit(q, &mut self.rng);
    }
}

impl Phys for DenseBackend {
    fn ensure(&mut self, n: usize) {
        assert!(
            n <= self.cap,
            "dense backend capacity {} exceeded ({n})",
            self.cap
        );
    }
    fn prep0(&mut self, q: usize) {
        self.reset(q);
    }
    fn prep_plus(&mut self, q: usize) {
        self.reset(q);
        self.g(Gate::H(q));
    }
    fn prep_t(&mut self, q: usize) {
        self.reset(q);
        self.g(Gate::H(q));
        self.g(Gate::T(q));
    }
    fn h(&mut self, q: usize) {
        self.g(Gate::H(q));
    }
    fn s(&mut self, q: usize) {
        self.g(Gate::S(q));
    }
    fn sdg(&mut self, q: usize) {
        self.g(Gate::Sdg(q));
    }
    fn cnot(&mut self, c: usize, t: usize) {
        self.g(Gate::Cnot(c, t));
    }
    fn meas_z(&mut self, q: usize) -> bool {
        self.sv.measure_qubit(q, &mut self.rng)
    }
    fn meas_x(&mut self, q: usize) -> bool {
        self.g(Gate::H(q));
        let r = self.sv.measure_qubit(q, &mut self.rng);
        self.g(Gate::H(q));
        r
    }
    fn pauli(&mut self, q: usize, code: u8) {
        if code & PZ != 0 {
            self.g(Gate::Z(q));
        }
        if code & PX != 0 {
            self.g(Gate::X(q));
        }
    }
}
