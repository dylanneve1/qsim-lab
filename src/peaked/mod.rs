//! Peak finding: the most likely output bitstring of a circuit and its probability.
//!
//! [`find_peak`] works on any unitary circuit the tensor-network engine can
//! contract: it computes every single-qubit marginal `<Z_q>` exactly
//! ([`crate::engines::tn::expectation`], light cones only), takes the bitstring
//! of their signs, and returns its exact probability
//! ([`crate::engines::tn::amplitude`]). For a peaked circuit the signs of the
//! marginals are the peak.
//!
//! [`hqap`] first reduces heuristic peaked circuits built by identity insertion
//! (a trained core `R ▷ P` with obfuscated `U ▷ U†` blocks in between, as in
//! the Quantum Advantage Tracker's `peaked_circuit_P11/P12`) to their core.
//!
//! Lab notebook: `research/simulability/peaked-circuits.md`.

pub mod hqap;

use crate::circuit::{Circuit, SimError};
use crate::engines::tn::{self, Pauli, TnOptions};

/// A circuit's peak: the sign bitstring of the single-qubit marginals.
#[derive(Clone, Debug)]
pub struct Peak {
    /// `bits[q]` is the peak's bit on qubit `q`.
    pub bits: Vec<bool>,
    /// Exact probability `|<bits|C|0>|^2` of that bitstring.
    pub probability: f64,
    /// `<Z_q>` for every qubit.
    pub marginals: Vec<f64>,
}

impl Peak {
    /// The bitstring with qubit 0 as the leftmost character.
    pub fn bitstring(&self) -> String {
        bitstring(&self.bits)
    }
}

/// `bits` as a string with qubit 0 as the leftmost character.
pub fn bitstring(bits: &[bool]) -> String {
    bits.iter().map(|&b| if b { '1' } else { '0' }).collect()
}

/// Exact single-qubit marginals, their sign bitstring and its exact probability.
pub fn find_peak(c: &Circuit, opts: &TnOptions) -> Result<Peak, SimError> {
    let mut marginals = Vec::with_capacity(c.num_qubits);
    for q in 0..c.num_qubits {
        let (z, _) = tn::expectation(c, &[(q, Pauli::Z)], opts)?;
        marginals.push(z);
    }
    let bits: Vec<bool> = marginals.iter().map(|&z| z < 0.0).collect();
    let (amp, _) = tn::amplitude(c, &bits, opts)?;
    Ok(Peak {
        bits,
        probability: amp.norm_sqr(),
        marginals,
    })
}
