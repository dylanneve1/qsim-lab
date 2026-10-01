//! Backend-independent circuit optimisation and dispatch.
//!
//! Every pass here is *exact*: it never changes the quantity the caller
//! asked for (amplitudes up to a tracked global phase, or the outcome
//! distribution of the measurements). The passes do less work before any
//! backend runs:
//!
//! * [`peephole`]: commutation-aware cancellation and merging of gates
//!   (`H H`, `CNOT CNOT`, `T T -> S`, `Rz(a) Rz(b) -> Rz(a+b)` across
//!   commuting gates, removal of identity rotations), tracking the global
//!   phase it introduces.
//! * [`analysis`]: backward light cones, connected components of the
//!   qubit-interaction graph, the classical "monomial" suffix before
//!   terminal measurements and the causal Clifford prefix.
//! * [`stabsv`]: a phase-exact stabilizer simulator for up to 64 qubits
//!   that turns a Clifford prefix into a state vector in one pass over the
//!   support (instead of one pass over `2^n` amplitudes per gate).
//! * [`plan`]: puts it together: per-component backend choice
//!   (tableau / state vector / Pauli paths) for sampling, full state
//!   vectors and expectation values.

pub mod analysis;
pub mod peephole;
pub mod plan;
pub mod stabsv;
pub mod stateprop;

pub use peephole::{optimize, Optimized, PeepholeOptions};
pub use plan::{
    compile_sampling, compile_unitary, expectation_z_product, SamplingPlan, UnitaryPlan,
};

use crate::gate::Gate;

/// The qubits of a gate without allocating: `(qubits, count)`.
#[inline]
pub(crate) fn qubits_of(g: &Gate) -> ([usize; 3], usize) {
    use Gate::*;
    match *g {
        H(q) | X(q) | Y(q) | Z(q) | S(q) | Sdg(q) | T(q) | Tdg(q) => ([q, 0, 0], 1),
        Rx(q, _) | Ry(q, _) | Rz(q, _) | Phase(q, _) => ([q, 0, 0], 1),
        Cnot(a, b) | Cz(a, b) | Swap(a, b) | CPhase(a, b, _) => ([a, b, 0], 2),
        Ccx(a, b, t) => ([a, b, t], 3),
    }
}

/// How a gate acts on one of its qubits, for commutation checks.
///
/// A gate is a sum of tensor products; if on qubit `q` every factor lies in
/// `span{I, Z}` the gate is `Z`-type on `q`, if every factor lies in
/// `span{I, X}` it is `X`-type. Two gates commute if on every shared qubit
/// they have the same (non-`Other`) type, because then every pair of tensor
/// terms commutes factor by factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Axis {
    Z,
    X,
    Other,
}

#[inline]
pub(crate) fn axis_on(g: &Gate, q: usize) -> Axis {
    use Gate::*;
    match *g {
        Z(_) | S(_) | Sdg(_) | T(_) | Tdg(_) | Phase(..) | Rz(..) | Cz(..) | CPhase(..) => Axis::Z,
        X(_) | Rx(..) => Axis::X,
        Cnot(c, _) => {
            if q == c {
                Axis::Z
            } else {
                Axis::X
            }
        }
        Ccx(_, _, t) => {
            if q == t {
                Axis::X
            } else {
                Axis::Z
            }
        }
        H(_) | Y(_) | Ry(..) | Swap(..) => Axis::Other,
    }
}

/// True if `g` maps computational basis states to (phases times) basis
/// states, i.e. its matrix is a permutation times a diagonal.
pub(crate) fn is_monomial(g: &Gate) -> bool {
    use Gate::*;
    !matches!(g, H(_) | Rx(..) | Ry(..))
}

/// True if the gate is diagonal in the computational basis.
pub(crate) fn is_diagonal(g: &Gate) -> bool {
    use Gate::*;
    matches!(
        g,
        Z(_) | S(_) | Sdg(_) | T(_) | Tdg(_) | Phase(..) | Rz(..) | Cz(..) | CPhase(..)
    )
}
