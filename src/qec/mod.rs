//! Quantum error correction (QEC) codes, decoders, and threshold benchmarks.

pub mod bposd;
pub mod color;
pub mod decoder;
pub mod dem;
pub mod distance;
pub mod repetition;
pub mod schedules;
pub mod surface;

pub use decoder::{DecodingGraph, GraphEdge, UnionFindDecoder};
pub use dem::{CircuitFaults, DemSampler, ErrorMechanism, FaultKind, Pauli, Signature};
pub use repetition::{MemoryExperimentResult, RepetitionCode};
pub use schedules::{
    BiasedDemSamplerV2, Permutation, Schedule, ScheduleResult, ScheduledSurfaceCode,
};
pub use surface::{SamplingMethod, SurfaceCode};
