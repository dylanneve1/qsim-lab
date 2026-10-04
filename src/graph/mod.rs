//! Graph compiler: treat a circuit like a compute graph.
//!
//! * [`param`]: circuits with symbolic (affine) angles.
//! * [`compiled`]: compile once (light cone, components, lowering, fusion,
//!   stage plan, constant folding, constant-prefix state), then
//!   [`CompiledCircuit::bind`] recomputes only the numbers that depend on
//!   the parameters.
//! * [`observable`]: Pauli-sum observables.
//!
//! Write-up and measurements: `research/graph-compiler.md`.

pub mod compiled;
pub mod observable;
pub mod param;
pub mod rewrite;

pub use compiled::{BoundCircuit, CompiledCircuit, GraphOptions, GraphStats};
pub use observable::{Observable, PauliTerm};
pub use param::{Angle, POp, ParamCircuit};
pub use rewrite::{phase_regions, RewriteOptions, RewriteStats};
