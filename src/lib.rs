//! # qsim-lab
//!
//! A small, from-scratch quantum circuit simulator written to explore how
//! different simulation methods scale:
//!
//! * [`statevector`]: dense `2^n` amplitudes. Exact, handles any gate, but
//!   memory doubles with every qubit.
//! * [`stabilizer`]: the Aaronson–Gottesman CHP tableau. `O(n^2)` bits and
//!   fast, but only for Clifford circuits (H, S, CNOT, Paulis, measurement).
//! * [`pauli_path`]: Clifford+T via Pauli-path summation. Polynomial in `n`,
//!   exponential in the number of T gates.
//! * [`mps`]: matrix product states with SVD truncation. Cost is set by the
//!   entanglement (bond dimension `χ`), not by `n`.
//!
//! All backends consume the same [`Circuit`]/[`Gate`] types and implement
//! [`Simulator`].

pub mod algorithms;
pub mod bench;
pub mod circuit;
pub mod gate;
pub mod mps;
pub mod pauli_path;
pub mod stabilizer;
pub mod statevector;

pub use circuit::{Circuit, Op, SimError, Simulator};
pub use gate::Gate;
pub use mps::Mps;
pub use stabilizer::Tableau;
pub use statevector::{StateVector, StateVectorF32, StateVectorF64};
