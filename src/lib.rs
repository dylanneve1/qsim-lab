//! # qsim-lab
//!
//! A small, from-scratch quantum circuit simulator written to explore how
//! different simulation methods scale:
//!
//! * [`engines::statevector`]: dense `2^n` amplitudes. Exact, handles any gate, but
//!   memory doubles with every qubit.
//! * [`engines::stabilizer`]: the Aaronson–Gottesman CHP tableau. `O(n^2)` bits and
//!   fast, but only for Clifford circuits (H, S, CNOT, Paulis, measurement).
//! * [`engines::pauli_path`]: Clifford+T via Pauli-path summation. Polynomial in `n`,
//!   exponential in the number of T gates.
//! * [`engines::mps`]: matrix product states with SVD truncation. Cost is set by the
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

#![warn(missing_docs)]

pub mod algorithms;
pub mod bench;
pub mod chem;
pub mod circuit;
pub mod compile;
pub mod dag;
pub mod engines;
pub mod error;
pub mod ft;
pub mod gate;
pub mod graph;
pub mod io;
pub mod magic_atlas;
pub mod noise;
pub mod peaked;
pub mod pipeline;
pub mod planner;
pub mod qec;
pub mod shor;
pub mod simulability;

// Pre-reorganisation module paths (`qsim_lab::engines::statevector`, `qsim_lab::shor::ge`,
// `qsim_lab::io::qasm`, ...), kept so downstream code keeps compiling. New code
// should use the paths above.
#[doc(hidden)]
pub use bench::adaptive as adaptive_bench;
#[cfg(all(feature = "metal", target_os = "macos"))]
#[doc(hidden)]
pub use engines::metal_sv;
#[doc(hidden)]
pub use engines::{
    adaptive, blocked, dense_fusion, hsf, monitored, mps, mps_cost, ooc, ooc_window, pauli_frame,
    pauli_path, sparse, spd, stab_rank, stabilizer, statevector,
};
#[doc(hidden)]
pub use io::{qasm, stim as stim_io};
#[doc(hidden)]
pub use shor::{
    arith as shor_arith, ge as shor_ge, mbu as shor_mbu, ripple as shor_ripple,
    superopt as shor_superopt, window as shor_window,
};

pub use circuit::{Circuit, CircuitStats, Op, SimError, Simulator};
pub use engines::hsf::{HsfOptions, HybridSchrodingerFeynman};
pub use engines::mps::Mps;
pub use engines::ooc::{OocConfig, OocStateVector, OocStats};
pub use engines::sparse::SparseState;
pub use engines::stabilizer::Tableau;
pub use engines::statevector::{StateVector, StateVectorF32, StateVectorF64};
pub use error::{Error, Result};
pub use gate::Gate;
pub use noise::NoiseModel;
pub use qec::{DecodingGraph, RepetitionCode, SurfaceCode, UnionFindDecoder};
