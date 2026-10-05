//! Fault-tolerant Shor at the gate level: Shor's order finding for N = 15 run
//! on concatenated-Steane logical qubits (level 1 = `[[7,1,3]]`, level 2 =
//! `[[49,1,9]]`) with circuit-level depolarizing noise on every physical
//! location, Steane error correction with verified ancillas, transversal
//! Cliffords, T gates by magic-state injection + teleportation, and
//! hierarchical hard-decision decoding. See `research/shor/ft-shor.md`.
//!
//! Simulation method (exact for stochastic Pauli noise): every physical
//! operation is a Clifford gate, a Pauli measurement, a preparation of
//! |0⟩/|+⟩/|T⟩ or a classically controlled Clifford, so the physical state is
//! always `F · Enc(|ψ⟩)` with `F` a Pauli frame and `|ψ⟩` the *ideal*
//! logical state. `F` is tracked bit-wise on the physical qubits; `|ψ⟩` is a
//! dense vector over the handful of logical qubits. Measurement outcomes are
//! the ideal (Born-sampled) outcome XOR the decoded frame flip, and every
//! feed-forward uses the recorded outcome — so decoder failures and noisy
//! magic states act on `|ψ⟩` exactly as physics dictates (including the
//! non-Pauli logical errors they cause through T gadgets). The dense backend
//! re-runs the same machine on a full physical state vector to validate this.

pub mod backends;
pub mod core;
pub mod logical;
pub mod machine;
pub mod shor;
