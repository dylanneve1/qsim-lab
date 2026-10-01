//! Stochastic Pauli noise channels and noise models for Monte Carlo trajectories.
//!
//! All noise channels in this module are stochastic Pauli operations, which means
//! they map stabilizer states to stabilizer states and can be simulated exactly on
//! the stabilizer tableau, state vector, and matrix product state backends.
//!
//! # Noise Channels
//!
//! * **Single-qubit depolarizing channel**: with probability `p`, an error occurs,
//!   chosen uniformly from `{X, Y, Z}` (each with probability `p / 3`).
//!   With probability `1 - p`, the identity is applied.
//! * **Two-qubit depolarizing channel**: with probability `p`, an error occurs,
//!   chosen uniformly from the 15 non-trivial Pauli pairs
//!   `{I, X, Y, Z} ⊗ {I, X, Y, Z} \ {I ⊗ I}` (each with probability `p / 15`).
//! * **Measurement flip (readout error)**: with probability `p_meas`, a classical
//!   measurement outcome is inverted (`0 <-> 1`).
//! * **Reset error**: with probability `p_reset`, an `X` flip is applied immediately
//!   after a qubit reset, leaving it in `|1>` instead of `|0>`.

use crate::circuit::{SimError, Simulator};
use crate::gate::Gate;
use rand::{Rng, RngCore};

/// A noise model for Monte Carlo trajectory simulation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NoiseModel {
    /// Single-qubit depolarizing error rate applied after each 1-qubit gate.
    pub p_1q: f64,
    /// Two-qubit depolarizing error rate applied after each 2-qubit gate.
    pub p_2q: f64,
    /// Measurement readout error rate (probability of inverting outcome).
    pub p_meas: f64,
    /// Reset error rate (probability of X flip after reset).
    pub p_reset: f64,
}

impl NoiseModel {
    /// An ideal noiseless model.
    pub fn none() -> Self {
        Self::default()
    }

    /// Uniform depolarizing error rate `p` on 1q gates, 2q gates, readout, and reset.
    pub fn uniform(p: f64) -> Self {
        Self {
            p_1q: p,
            p_2q: p,
            p_meas: p,
            p_reset: p,
        }
    }

    /// Gate-only depolarizing error rates.
    pub fn gate_depolarizing(p_1q: f64, p_2q: f64) -> Self {
        Self {
            p_1q,
            p_2q,
            p_meas: 0.0,
            p_reset: 0.0,
        }
    }

    /// Standard circuit-level noise model with gate error `p_gate` and readout error `p_meas`.
    pub fn circuit_level(p_gate: f64, p_meas: f64) -> Self {
        Self {
            p_1q: p_gate,
            p_2q: p_gate,
            p_meas,
            p_reset: p_gate,
        }
    }

    pub fn with_p1(mut self, p: f64) -> Self {
        self.p_1q = p;
        self
    }

    pub fn with_p2(mut self, p: f64) -> Self {
        self.p_2q = p;
        self
    }

    pub fn with_meas(mut self, p: f64) -> Self {
        self.p_meas = p;
        self
    }

    pub fn with_reset(mut self, p: f64) -> Self {
        self.p_reset = p;
        self
    }

    /// True if all error probabilities are zero.
    pub fn is_noiseless(&self) -> bool {
        self.p_1q <= 0.0 && self.p_2q <= 0.0 && self.p_meas <= 0.0 && self.p_reset <= 0.0
    }
}

/// Samples a single-qubit depolarizing error on qubit `q` with probability `p`.
///
/// Returns `Some(Gate::X(q))`, `Some(Gate::Y(q))`, or `Some(Gate::Z(q))` each
/// with probability `p / 3`, or `None` with probability `1 - p`.
pub fn sample_depolarizing_1q<R: RngCore + ?Sized>(p: f64, q: usize, rng: &mut R) -> Option<Gate> {
    if p <= 0.0 {
        return None;
    }
    let r: f64 = rng.random();
    if r < p {
        let sub = r / p;
        if sub < 1.0 / 3.0 {
            Some(Gate::X(q))
        } else if sub < 2.0 / 3.0 {
            Some(Gate::Y(q))
        } else {
            Some(Gate::Z(q))
        }
    } else {
        None
    }
}

/// Samples a two-qubit depolarizing error on qubits `(a, b)` with probability `p`.
///
/// With probability `p`, uniformly chooses one of the 15 non-identity Pauli pairs
/// `{I, X, Y, Z}^2 \ {II}`, each with probability `p / 15`. Returns a list of 1 or 2
/// single-qubit gates representing the error.
pub fn sample_depolarizing_2q<R: RngCore + ?Sized>(
    p: f64,
    a: usize,
    b: usize,
    rng: &mut R,
) -> Vec<Gate> {
    if p <= 0.0 {
        return Vec::new();
    }
    let r: f64 = rng.random();
    if r < p {
        let k = rng.random_range(1..16usize);
        let pa = k / 4;
        let pb = k % 4;
        let mut errs = Vec::with_capacity(2);
        match pa {
            1 => errs.push(Gate::X(a)),
            2 => errs.push(Gate::Y(a)),
            3 => errs.push(Gate::Z(a)),
            _ => {}
        }
        match pb {
            1 => errs.push(Gate::X(b)),
            2 => errs.push(Gate::Y(b)),
            3 => errs.push(Gate::Z(b)),
            _ => {}
        }
        errs
    } else {
        Vec::new()
    }
}

/// Applies gate-level depolarizing noise following gate `g`.
pub fn apply_gate_noise<S: Simulator + ?Sized, R: RngCore + ?Sized>(
    sim: &mut S,
    g: &Gate,
    noise: &NoiseModel,
    rng: &mut R,
) -> Result<(), SimError> {
    let qs = g.qubits();
    match qs.len() {
        1 if noise.p_1q > 0.0 => {
            if let Some(err) = sample_depolarizing_1q(noise.p_1q, qs[0], rng) {
                sim.apply(&err)?;
            }
        }
        2 if noise.p_2q > 0.0 => {
            for err in sample_depolarizing_2q(noise.p_2q, qs[0], qs[1], rng) {
                sim.apply(&err)?;
            }
        }
        _ => {}
    }
    Ok(())
}
