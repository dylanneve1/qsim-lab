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
pub mod phasefold;
pub mod plan;
pub mod repeat;
pub mod stabsv;
pub mod stateprop;

pub use peephole::{optimize, Optimized, PeepholeOptions};
pub use phasefold::{phase_fold, phase_fold_with_stats, PhaseFoldStats};
pub use plan::{
    compile_sampling, compile_unitary, expectation_z_product, SamplingPlan, UnitaryPlan,
};

use crate::circuit::{Circuit, Op, SimError};
use crate::gate::Gate;

/// The qubits of a gate without allocating: `(qubits, count)`.
#[inline]
pub(crate) fn qubits_of(g: &Gate) -> ([usize; 3], usize) {
    use Gate::*;
    match *g {
        I(q) | H(q) | X(q) | Y(q) | Z(q) | S(q) | Sdg(q) | T(q) | Tdg(q) | Sx(q) | Sxdg(q) => {
            ([q, 0, 0], 1)
        }
        Rx(q, _) | Ry(q, _) | Rz(q, _) | Phase(q, _) | U(q, ..) => ([q, 0, 0], 1),
        Cnot(a, b) | Cz(a, b) | Swap(a, b) | ISwap(a, b) | ISwapdg(a, b) | CPhase(a, b, _) => {
            ([a, b, 0], 2)
        }
        Ccx(a, b, t) => ([a, b, t], 3),
    }
}

/// The qubits an operation touches: `(qubits, count)`. For a classically
/// controlled gate these are the gate's qubits (the classical bit it reads
/// is not a qubit; see [`classical_dependency`]).
#[inline]
pub(crate) fn op_qubits(op: &Op) -> ([usize; 3], usize) {
    match *op {
        Op::Gate(ref g) | Op::ClassicControlled { gate: ref g, .. } => qubits_of(g),
        Op::Measure(q)
        | Op::Reset(q)
        | Op::XFlip(q, _)
        | Op::YFlip(q, _)
        | Op::ZFlip(q, _)
        | Op::Depolarize1q(q, _) => ([q, 0, 0], 1),
        Op::Depolarize2q(a, b, _) => ([a, b, 0], 2),
    }
}

/// The index (in program order, as in [`Circuit::run`](crate::circuit::Circuit::run))
/// of the measurement outcome an operation reads, if any.
#[inline]
pub(crate) fn classical_dependency(op: &Op) -> Option<usize> {
    match *op {
        Op::ClassicControlled { meas_index, .. } => Some(meas_index),
        Op::Gate(_)
        | Op::Measure(_)
        | Op::Reset(_)
        | Op::XFlip(..)
        | Op::YFlip(..)
        | Op::ZFlip(..)
        | Op::Depolarize1q(..)
        | Op::Depolarize2q(..) => None,
    }
}

/// True for operations that are unitary gates (everything except
/// measurements, resets, classically controlled gates and noise channels).
#[inline]
pub(crate) fn is_unitary_op(op: &Op) -> bool {
    match op {
        Op::Gate(_) => true,
        Op::Measure(_)
        | Op::Reset(_)
        | Op::ClassicControlled { .. }
        | Op::XFlip(..)
        | Op::YFlip(..)
        | Op::ZFlip(..)
        | Op::Depolarize1q(..)
        | Op::Depolarize2q(..) => false,
    }
}

/// Checks what [`Circuit::run`](crate::circuit::Circuit::run) would reject
/// at run time, so the passes can rely on it: every qubit index is in
/// range, no operation uses a qubit twice, and every classically
/// controlled gate reads a measurement that happened before it. Noise
/// probabilities must lie in `[0, 1]` (outside it `run` would not apply
/// the channel its name describes).
pub fn validate(c: &Circuit) -> Result<(), SimError> {
    let n = c.num_qubits;
    let mut measured = 0;
    for op in &c.ops {
        let (qs, k) = op_qubits(op);
        for (i, &q) in qs[..k].iter().enumerate() {
            if q >= n {
                return Err(SimError::QubitOutOfRange {
                    qubit: q,
                    num_qubits: n,
                });
            }
            if qs[..i].contains(&q) {
                return Err(match op {
                    Op::Gate(g) | Op::ClassicControlled { gate: g, .. } => {
                        SimError::RepeatedQubit(*g)
                    }
                    _ => SimError::NotSupported {
                        what: "a two-qubit noise channel acting twice on one qubit",
                    },
                });
            }
        }
        let prob = match *op {
            Op::XFlip(_, p)
            | Op::YFlip(_, p)
            | Op::ZFlip(_, p)
            | Op::Depolarize1q(_, p)
            | Op::Depolarize2q(_, _, p) => Some(p),
            Op::Gate(_) | Op::Measure(_) | Op::Reset(_) | Op::ClassicControlled { .. } => None,
        };
        if prob.is_some_and(|p| !(0.0..=1.0).contains(&p)) {
            return Err(SimError::NotSupported {
                what: "a noise probability outside [0, 1]",
            });
        }
        if let Some(m) = classical_dependency(op) {
            if m >= measured {
                return Err(SimError::ClassicalBitOutOfRange {
                    bit: m,
                    available: measured,
                });
            }
        }
        if matches!(op, Op::Measure(_)) {
            measured += 1;
        }
    }
    Ok(())
}

/// Fails with [`SimError::NotSupported`] unless every op is a unitary gate.
pub(crate) fn require_unitary(c: &Circuit, what: &'static str) -> Result<(), SimError> {
    if c.ops.iter().all(is_unitary_op) {
        Ok(())
    } else {
        Err(SimError::NotSupported { what })
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
        // The identity lies in span{I, Z} (and span{I, X}); either works.
        I(_) => Axis::Z,
        Z(_) | S(_) | Sdg(_) | T(_) | Tdg(_) | Phase(..) | Rz(..) | Cz(..) | CPhase(..) => Axis::Z,
        // Sx = e^{iπ/4} Rx(π/2) = (1+i)/2 I + (1-i)/2 X.
        X(_) | Rx(..) | Sx(_) | Sxdg(_) => Axis::X,
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
        // iSWAP contains XX + YY terms; U is a general rotation.
        H(_) | Y(_) | Ry(..) | U(..) | Swap(..) | ISwap(..) | ISwapdg(..) => Axis::Other,
    }
}

/// True if `g` maps computational basis states to (phases times) basis
/// states, i.e. its matrix is a permutation times a diagonal. `U` is
/// treated as non-monomial whatever its angles (always safe: monomial
/// gates only enable the classical-suffix rewrite).
pub(crate) fn is_monomial(g: &Gate) -> bool {
    use Gate::*;
    match g {
        I(_) | X(_) | Y(_) | Z(_) | S(_) | Sdg(_) | T(_) | Tdg(_) | Rz(..) | Phase(..) => true,
        // iSWAP: |01> -> i|10>, |10> -> i|01>: a SWAP times phases.
        Cnot(..) | Cz(..) | Swap(..) | ISwap(..) | ISwapdg(..) | CPhase(..) | Ccx(..) => true,
        H(_) | Sx(_) | Sxdg(_) | Rx(..) | Ry(..) | U(..) => false,
    }
}

/// True if the gate is diagonal in the computational basis (`U` counts as
/// non-diagonal, which is always safe).
pub(crate) fn is_diagonal(g: &Gate) -> bool {
    use Gate::*;
    match g {
        I(_) | Z(_) | S(_) | Sdg(_) | T(_) | Tdg(_) | Phase(..) | Rz(..) | Cz(..) | CPhase(..) => {
            true
        }
        H(_) | X(_) | Y(_) | Sx(_) | Sxdg(_) | Rx(..) | Ry(..) | U(..) => false,
        Cnot(..) | Swap(..) | ISwap(..) | ISwapdg(..) | Ccx(..) => false,
    }
}
