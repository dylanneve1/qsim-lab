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
//!
//! ```
//! use qsim_lab::{Circuit, StateVectorF32, Tableau};
//! use rand::SeedableRng;
//!
//! let mut c = Circuit::new(3);
//! c.h(0).cnot(0, 1).cnot(1, 2).measure_all();
//! let mut rng = rand::rngs::StdRng::seed_from_u64(1);
//!
//! let mut sv = StateVectorF32::new(3);
//! let bits = c.run(&mut sv, &mut rng).unwrap(); // dense simulation
//! assert!(bits.iter().all(|&b| b == bits[0]));
//!
//! let mut tab = Tableau::new(3);
//! let bits = c.run(&mut tab, &mut rng).unwrap(); // stabilizer simulation
//! assert!(bits.iter().all(|&b| b == bits[0]));
//! ```

pub mod adaptive;
pub mod adaptive_bench;
pub mod algorithms;
pub mod bench;
pub mod blocked;
pub mod circuit;
pub mod compile;
pub mod dag;
pub mod ft;
pub mod gate;
pub mod hsf;
pub mod magic_atlas;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal_sv;
pub mod monitored;
pub mod mps;
pub mod mps_cost;
pub mod noise;
pub mod ooc;
pub mod ooc_window;
pub mod pauli_frame;
pub mod pauli_path;
pub mod pipeline;
pub mod planner;
pub mod qasm;
pub mod qec;
pub mod shor;
pub mod shor_arith;
pub mod shor_ge;
pub mod shor_mbu;
pub mod shor_ripple;
pub mod shor_superopt;
pub mod shor_window;
pub mod simulability;
pub mod sparse;
pub mod stabilizer;
pub mod statevector;
pub mod stim_io;

pub use circuit::{Circuit, CircuitStats, Op, SimError, Simulator};
pub use gate::Gate;
pub use hsf::{HsfOptions, HybridSchrodingerFeynman};
pub use mps::Mps;
pub use noise::NoiseModel;
pub use ooc::{OocConfig, OocStateVector, OocStats};
pub use qec::{DecodingGraph, RepetitionCode, SurfaceCode, UnionFindDecoder};
pub use sparse::SparseState;
pub use stabilizer::Tableau;
pub use statevector::{StateVector, StateVectorF32, StateVectorF64};
