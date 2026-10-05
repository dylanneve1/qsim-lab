<<<<<<< HEAD
//! Circuit file formats: OpenQASM 2.0 ([`qasm`]), Stim's `.stim` format ([`stim`])
//! and the `.qc` format of the Feynman T-count benchmarks ([`qc`]).
=======
//! Circuit file formats: OpenQASM 2.0 ([`qasm`]), Stim's `.stim` format ([`stim`]),
//! and ONNX export for graph viewers such as Netron ([`onnx`]).
>>>>>>> exp/netron

pub mod onnx;
pub mod qasm;
pub mod qc;
pub mod stim;
