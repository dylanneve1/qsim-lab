//! Free-fermion (fermionic Gaussian, matchgate) circuits: a detector that
//! finds the structure in an arbitrary gate list, and an engine that
//! simulates such circuits in polynomial time (docs/ENGINE_GAUSSIAN.md).
//!
//! * [`detect`] fuses the gates into one- and two-qubit blocks (undoing SWAP
//!   networks by relabelling wires), looks for a Jordan–Wigner order in which
//!   every block is a Gaussian map of the Majorana operators, and reports
//!   how far the circuit is from that: [`GaussianReport::gaussian_fraction`],
//!   [`GaussianReport::max_residual`], and the diagonal interaction phases
//!   `exp(i g n_a n_b)` ([`InteractionPhase`]).
//! * [`GaussianState`] evolves the `2n x 2n` Majorana covariance matrix,
//!   O(n) per block, and reads out `⟨Z_i⟩`, Z-products by Wick's theorem
//!   (Pfaffians), basis-state probabilities (Pfaffians, O(n^3)) and samples
//!   (O(n^3) per shot). Amplitudes are not available: the covariance matrix
//!   does not carry the global phase.
//! * [`pt2`] treats weak interaction phases `exp(i λ g n_a n_b)` of a
//!   number-conserving circuit by exact second-order perturbation theory in
//!   `λ` (Wick's theorem on the single-particle correlation matrix).
//!
//! The planner ([`crate::planner::Engine::Gaussian`]) chooses this engine
//! when the circuit is exactly Gaussian.
//!
//! ```
//! use qsim_lab::Circuit;
//! use qsim_lab::engines::gaussian::{self, GaussianOptions, GaussianState};
//!
//! // a hopping (XX + YY) rotation between two modes, on top of |10>
//! let mut c = Circuit::new(2);
//! c.x(0).h(0).h(1).cnot(0, 1).rz(1, 0.3).cnot(0, 1).h(0).h(1);
//! let r = gaussian::detect(&c, &Default::default());
//! assert!(r.exact);
//! let (st, _) = GaussianState::from_circuit(&c, &GaussianOptions::default()).unwrap();
//! assert!(st.expectation_z(0).unwrap().abs() <= 1.0);
//! ```

mod detect;
pub mod pt2;
mod state;

pub use detect::{
    compile, detect, DetectOptions, GaussOp, GaussianProgram, GaussianReport, InteractionPhase,
    Ordering,
};
pub use state::{covariance_bytes, pfaffian, GaussianOptions, GaussianState, InteractionPolicy};

use crate::circuit::{Circuit, SimError};

/// `⟨Π_{q ∈ obs} Z_q⟩` of `c|0^n⟩` on the Gaussian engine (errors as in
/// [`GaussianState::from_circuit`]).
pub fn expectation_z_product(
    c: &Circuit,
    obs: &[usize],
    opts: &GaussianOptions,
) -> Result<f64, SimError> {
    let (st, _) = GaussianState::from_circuit(c, opts)?;
    st.expectation_z_product(obs)
}

/// Predicted floating-point work of the engine: `48 n` per block for the
/// covariance update, plus `read_out` (e.g. `(4/3)(2n)^3` per shot).
pub fn work(report: &GaussianReport) -> f64 {
    48.0 * report.blocks as f64 * report.n.max(1) as f64
}
