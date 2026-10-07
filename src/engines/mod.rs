//! Exact simulation engines. Every engine consumes the same
//! [`Circuit`](crate::Circuit) / [`Gate`](crate::Gate) types; most implement
//! [`Simulator`](crate::Simulator).
//!
//! Dense state vector:
//! * [`statevector`] — `2^n` amplitudes, any gate (f32 / f64).
//! * [`blocked`] — cache-blocked, fused executor for the state vector (the default dense path).
//! * [`dense_fusion`] — dense k-qubit gate fusion for the blocked executor.
//! * [`ooc`], [`ooc_window`] — out-of-core (disk-backed) state vector and its pass scheduler.
//! * [`dist`] — two-node distributed state vector (MPI-style global-qubit swaps over a
//!   byte-stream link: TCP, an SSH session, or in-process channels).
//! * `metal_sv` — Apple-GPU (Metal) f32 state vector (`--features metal`, macOS only).
//!
//! Structured / low-rank representations:
//! * [`sparse`] — sparse state vector (hash map of nonzero amplitudes).
//! * [`stabilizer`] — Aaronson–Gottesman tableau, SymPhase and fast detector samplers.
//! * [`pauli_path`], [`pauli_frame`] — Clifford+T by Pauli-path summation / rotation frame.
//! * [`adaptive`] — adaptive representation switching for Clifford+T circuits.
//! * [`stab_rank`] — branching-rank (sum of stabilizer states) simulator.
//! * [`mps`], [`mps_cost`] — matrix product states, and a cost predictor for exact MPS runs.
//! * [`hsf`] — hybrid Schrödinger–Feynman simulation.
//! * [`tn`] — exact tensor-network contraction (amplitudes, batches, Pauli expectations).
//! * [`chain_sweep`] — exact amplitudes of open-chain CZ circuits by sweeping a `D/2`-bit
//!   bond register along the chain (runs on the blocked CPU and Metal executors).
//! * [`chain_lowprec`] — emulated 16-bit / 8-bit storage of the chain-sweep register (fidelity cost).
//! * [`chain_packed`] — the same register stored packed (`b`-bit ints + block scales), streamed per pass.
//! * [`spd`] — sparse Pauli dynamics for kicked-Ising Trotter circuits.
//! * [`monitored`] — monitored Clifford+T circuits (measurement-induced transitions).

pub mod adaptive;
pub mod blocked;
pub mod chain_lowprec;
pub mod chain_packed;
pub mod chain_packed_gpu;
pub mod chain_sweep;
pub mod dense_fusion;
pub(crate) mod dense_kernels;
pub mod dist;
pub mod hsf;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal_sv;
pub mod monitored;
pub mod mps;
pub mod mps_cost;
pub mod ooc;
pub mod ooc_window;
pub mod pauli_frame;
pub mod pauli_path;
pub mod sparse;
pub mod spd;
pub mod stab_rank;
pub mod stabilizer;
pub mod statevector;
pub mod tn;
