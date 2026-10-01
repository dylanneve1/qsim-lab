//! A minimal circuit representation and the `Simulator` trait that every
//! backend implements.

use crate::gate::Gate;
use rand::{Rng, RngCore};
use std::fmt;

/// Errors a backend can report.
#[derive(Clone, Debug, PartialEq)]
pub enum SimError {
    /// The backend cannot represent this gate (e.g. T on a stabilizer tableau).
    Unsupported { backend: &'static str, gate: Gate },
    /// A qubit index is out of range.
    QubitOutOfRange { qubit: usize, num_qubits: usize },
    /// A gate was given the same qubit twice.
    RepeatedQubit(Gate),
    /// The requested register would exceed the crate's memory cap.
    TooLarge {
        what: &'static str,
        bytes: u128,
        limit: u128,
    },
    /// The Pauli-path simulator exceeded its term budget.
    TooManyTerms { terms: usize, limit: usize },
}

impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimError::Unsupported { backend, gate } => {
                write!(f, "the {backend} backend does not support {gate:?}")
            }
            SimError::QubitOutOfRange { qubit, num_qubits } => {
                write!(f, "qubit {qubit} out of range for {num_qubits} qubits")
            }
            SimError::RepeatedQubit(g) => write!(f, "gate {g:?} uses a qubit twice"),
            SimError::TooLarge { what, bytes, limit } => write!(
                f,
                "{what} would need {bytes} bytes, above the {limit}-byte limit"
            ),
            SimError::TooManyTerms { terms, limit } => {
                write!(f, "{terms} Pauli terms exceeds the limit of {limit}")
            }
        }
    }
}

impl std::error::Error for SimError {}

/// Checks qubit indices of a gate against the register size.
pub fn check_gate(g: &Gate, num_qubits: usize) -> Result<(), SimError> {
    let qs = g.qubits();
    for (i, &q) in qs.iter().enumerate() {
        if q >= num_qubits {
            return Err(SimError::QubitOutOfRange {
                qubit: q,
                num_qubits,
            });
        }
        if qs[..i].contains(&q) {
            return Err(SimError::RepeatedQubit(*g));
        }
    }
    Ok(())
}

/// The common interface of all backends.
pub trait Simulator {
    /// Short backend name, used in error messages and benchmark output.
    fn name(&self) -> &'static str;
    fn num_qubits(&self) -> usize;
    /// Applies a gate in place.
    fn apply(&mut self, gate: &Gate) -> Result<(), SimError>;
    /// Measures qubit `q` in the computational basis, collapsing the state.
    fn measure(&mut self, q: usize, rng: &mut dyn RngCore) -> Result<bool, SimError>;
}

/// One instruction of a circuit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    Gate(Gate),
    Measure(usize),
}

/// An ordered list of gates and measurements on `num_qubits` qubits.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Circuit {
    pub num_qubits: usize,
    pub ops: Vec<Op>,
}

macro_rules! builder_1q {
    ($($name:ident => $variant:ident),*) => {$(
        pub fn $name(&mut self, q: usize) -> &mut Self {
            self.gate(Gate::$variant(q))
        }
    )*};
}

impl Circuit {
    pub fn new(num_qubits: usize) -> Self {
        Circuit {
            num_qubits,
            ops: Vec::new(),
        }
    }

    pub fn gate(&mut self, g: Gate) -> &mut Self {
        self.ops.push(Op::Gate(g));
        self
    }

    builder_1q!(h => H, x => X, y => Y, z => Z, s => S, sdg => Sdg, t => T, tdg => Tdg);

    pub fn rx(&mut self, q: usize, theta: f64) -> &mut Self {
        self.gate(Gate::Rx(q, theta))
    }
    pub fn ry(&mut self, q: usize, theta: f64) -> &mut Self {
        self.gate(Gate::Ry(q, theta))
    }
    pub fn rz(&mut self, q: usize, theta: f64) -> &mut Self {
        self.gate(Gate::Rz(q, theta))
    }
    pub fn phase(&mut self, q: usize, theta: f64) -> &mut Self {
        self.gate(Gate::Phase(q, theta))
    }
    pub fn cnot(&mut self, c: usize, t: usize) -> &mut Self {
        self.gate(Gate::Cnot(c, t))
    }
    pub fn cz(&mut self, a: usize, b: usize) -> &mut Self {
        self.gate(Gate::Cz(a, b))
    }
    pub fn swap(&mut self, a: usize, b: usize) -> &mut Self {
        self.gate(Gate::Swap(a, b))
    }
    pub fn cphase(&mut self, a: usize, b: usize, theta: f64) -> &mut Self {
        self.gate(Gate::CPhase(a, b, theta))
    }
    pub fn ccx(&mut self, a: usize, b: usize, t: usize) -> &mut Self {
        self.gate(Gate::Ccx(a, b, t))
    }
    pub fn measure(&mut self, q: usize) -> &mut Self {
        self.ops.push(Op::Measure(q));
        self
    }
    pub fn measure_all(&mut self) -> &mut Self {
        for q in 0..self.num_qubits {
            self.measure(q);
        }
        self
    }

    /// Appends all operations of `other` (which must not be wider).
    pub fn append(&mut self, other: &Circuit) -> &mut Self {
        assert!(other.num_qubits <= self.num_qubits);
        self.ops.extend_from_slice(&other.ops);
        self
    }

    /// Iterates over the gates, skipping measurements.
    pub fn gates(&self) -> impl Iterator<Item = &Gate> + '_ {
        self.ops.iter().filter_map(|op| match op {
            Op::Gate(g) => Some(g),
            Op::Measure(_) => None,
        })
    }

    pub fn num_gates(&self) -> usize {
        self.gates().count()
    }

    /// Number of T/T† gates.
    pub fn t_count(&self) -> usize {
        self.gates().filter(|g| g.is_t()).count()
    }

    pub fn is_clifford(&self) -> bool {
        self.gates().all(|g| g.is_clifford())
    }

    /// The inverse circuit (gates reversed and inverted). Panics if the
    /// circuit contains measurements.
    pub fn inverse(&self) -> Circuit {
        let ops = self
            .ops
            .iter()
            .rev()
            .map(|op| match op {
                Op::Gate(g) => Op::Gate(g.inverse()),
                Op::Measure(_) => panic!("cannot invert a measurement"),
            })
            .collect();
        Circuit {
            num_qubits: self.num_qubits,
            ops,
        }
    }

    /// Runs the circuit on a simulator and returns the measurement outcomes
    /// in program order.
    pub fn run<S: Simulator + ?Sized>(
        &self,
        sim: &mut S,
        rng: &mut dyn RngCore,
    ) -> Result<Vec<bool>, SimError> {
        let mut out = Vec::new();
        for op in &self.ops {
            match op {
                Op::Gate(g) => sim.apply(g)?,
                Op::Measure(q) => out.push(sim.measure(*q, rng)?),
            }
        }
        Ok(out)
    }

    /// A random Clifford circuit with `depth` layers. Each layer applies a
    /// random single-qubit Clifford generator to every qubit and then a
    /// random two-qubit gate on random pairs.
    pub fn random_clifford<R: Rng + ?Sized>(num_qubits: usize, depth: usize, rng: &mut R) -> Self {
        Self::random_layers(num_qubits, depth, 0.0, rng)
    }

    /// Like [`Circuit::random_clifford`] but each single-qubit slot is a T
    /// gate with probability `t_prob`.
    pub fn random_clifford_t<R: Rng + ?Sized>(
        num_qubits: usize,
        depth: usize,
        t_prob: f64,
        rng: &mut R,
    ) -> Self {
        Self::random_layers(num_qubits, depth, t_prob, rng)
    }

    fn random_layers<R: Rng + ?Sized>(n: usize, depth: usize, t_prob: f64, rng: &mut R) -> Self {
        let mut c = Circuit::new(n);
        for _ in 0..depth {
            for q in 0..n {
                if t_prob > 0.0 && rng.random_bool(t_prob) {
                    c.gate(if rng.random_bool(0.5) {
                        Gate::T(q)
                    } else {
                        Gate::Tdg(q)
                    });
                    continue;
                }
                let g = match rng.random_range(0..6) {
                    0 => Gate::H(q),
                    1 => Gate::S(q),
                    2 => Gate::Sdg(q),
                    3 => Gate::X(q),
                    4 => Gate::Y(q),
                    _ => Gate::Z(q),
                };
                c.gate(g);
            }
            if n >= 2 {
                for _ in 0..n / 2 {
                    let a = rng.random_range(0..n);
                    let mut b = rng.random_range(0..n - 1);
                    if b >= a {
                        b += 1;
                    }
                    let g = match rng.random_range(0..3) {
                        0 => Gate::Cnot(a, b),
                        1 => Gate::Cz(a, b),
                        _ => Gate::Swap(a, b),
                    };
                    c.gate(g);
                }
            }
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn builder_and_counts() {
        let mut c = Circuit::new(3);
        c.h(0).t(1).cnot(0, 1).tdg(2).measure_all();
        assert_eq!(c.num_gates(), 4);
        assert_eq!(c.t_count(), 2);
        assert!(!c.is_clifford());
        assert_eq!(c.ops.len(), 7);
    }

    #[test]
    fn random_clifford_is_clifford() {
        let mut rng = StdRng::seed_from_u64(1);
        let c = Circuit::random_clifford(6, 10, &mut rng);
        assert!(c.is_clifford());
        assert!(c.gates().all(|g| check_gate(g, 6).is_ok()));
    }

    #[test]
    fn check_gate_rejects_bad_qubits() {
        assert!(check_gate(&Gate::Cnot(1, 1), 3).is_err());
        assert!(check_gate(&Gate::H(3), 3).is_err());
        assert!(check_gate(&Gate::Ccx(0, 1, 2), 3).is_ok());
    }
}
