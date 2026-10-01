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
    /// A classical bit index referenced by a conditional operation was out of range.
    ClassicalBitOutOfRange { bit: usize, available: usize },
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
            SimError::ClassicalBitOutOfRange { bit, available } => {
                write!(
                    f,
                    "classical bit {bit} is out of range ({available} available)"
                )
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
    /// Resets qubit `q` to the computational basis state |0>.
    fn reset(&mut self, q: usize, rng: &mut dyn RngCore) -> Result<(), SimError> {
        if q >= self.num_qubits() {
            return Err(SimError::QubitOutOfRange {
                qubit: q,
                num_qubits: self.num_qubits(),
            });
        }
        let outcome = self.measure(q, rng)?;
        if outcome {
            self.apply(&Gate::X(q))?;
        }
        Ok(())
    }
}

/// One instruction of a circuit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    Gate(Gate),
    Measure(usize),
    Reset(usize),
    /// Classical condition: apply `gate` if measured bit `meas_index` equals `target_value`.
    ClassicControlled {
        gate: Gate,
        meas_index: usize,
        target_value: bool,
    },
    /// Stochastic Pauli X flip on qubit `q` with probability `p`.
    XFlip(usize, f64),
    /// Stochastic Pauli Y flip on qubit `q` with probability `p`.
    YFlip(usize, f64),
    /// Stochastic Pauli Z flip on qubit `q` with probability `p`.
    ZFlip(usize, f64),
    /// Single-qubit depolarizing error on qubit `q` with probability `p`.
    Depolarize1q(usize, f64),
    /// Two-qubit depolarizing error on qubits `(a, b)` with probability `p`.
    Depolarize2q(usize, usize, f64),
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
    pub fn reset(&mut self, q: usize) -> &mut Self {
        self.ops.push(Op::Reset(q));
        self
    }
    /// Conditionally applies `gate` if measured bit `meas_index` is `true`.
    pub fn c_if(&mut self, meas_index: usize, gate: Gate) -> &mut Self {
        self.ops.push(Op::ClassicControlled {
            gate,
            meas_index,
            target_value: true,
        });
        self
    }
    /// Conditionally applies `gate` if measured bit `meas_index` equals `target_value`.
    pub fn classic_controlled(
        &mut self,
        gate: Gate,
        meas_index: usize,
        target_value: bool,
    ) -> &mut Self {
        self.ops.push(Op::ClassicControlled {
            gate,
            meas_index,
            target_value,
        });
        self
    }
    /// Stochastic Pauli X flip on qubit `q` with probability `p`.
    pub fn x_flip(&mut self, q: usize, p: f64) -> &mut Self {
        self.ops.push(Op::XFlip(q, p));
        self
    }
    /// Stochastic Pauli Y flip on qubit `q` with probability `p`.
    pub fn y_flip(&mut self, q: usize, p: f64) -> &mut Self {
        self.ops.push(Op::YFlip(q, p));
        self
    }
    /// Stochastic Pauli Z flip on qubit `q` with probability `p`.
    pub fn z_flip(&mut self, q: usize, p: f64) -> &mut Self {
        self.ops.push(Op::ZFlip(q, p));
        self
    }
    /// Single-qubit depolarizing error on qubit `q` with probability `p`.
    pub fn depolarize_1q(&mut self, q: usize, p: f64) -> &mut Self {
        self.ops.push(Op::Depolarize1q(q, p));
        self
    }
    /// Two-qubit depolarizing error on qubits `(a, b)` with probability `p`.
    pub fn depolarize_2q(&mut self, a: usize, b: usize, p: f64) -> &mut Self {
        self.ops.push(Op::Depolarize2q(a, b, p));
        self
    }

    /// Appends all operations of `other` (which must not be wider).
    pub fn append(&mut self, other: &Circuit) -> &mut Self {
        assert!(other.num_qubits <= self.num_qubits);
        self.ops.extend_from_slice(&other.ops);
        self
    }

    /// Iterates over the gates, skipping measurements, resets, and noise channels.
    pub fn gates(&self) -> impl Iterator<Item = &Gate> + '_ {
        self.ops.iter().filter_map(|op| match op {
            Op::Gate(g) => Some(g),
            _ => None,
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
        self.ops.iter().all(|op| match op {
            Op::Gate(g) => g.is_clifford(),
            Op::ClassicControlled { gate, .. } => gate.is_clifford(),
            _ => true,
        })
    }

    /// The inverse circuit (gates reversed and inverted). Panics if the
    /// circuit contains non-gate operations.
    pub fn inverse(&self) -> Circuit {
        let ops = self
            .ops
            .iter()
            .rev()
            .map(|op| match op {
                Op::Gate(g) => Op::Gate(g.inverse()),
                _ => panic!("cannot invert a circuit with non-gate operations"),
            })
            .collect();
        Circuit {
            num_qubits: self.num_qubits,
            ops,
        }
    }

    /// Runs the circuit on a simulator with a noise model and returns the
    /// measurement outcomes in program order.
    pub fn run_noisy<S: Simulator + ?Sized>(
        &self,
        sim: &mut S,
        noise: &crate::noise::NoiseModel,
        rng: &mut dyn RngCore,
    ) -> Result<Vec<bool>, SimError> {
        let mut out = Vec::new();
        for op in &self.ops {
            match op {
                Op::Gate(g) => {
                    sim.apply(g)?;
                    crate::noise::apply_gate_noise(sim, g, noise, rng)?;
                }
                Op::Measure(q) => {
                    let mut b = sim.measure(*q, rng)?;
                    if noise.p_meas > 0.0 && rng.random::<f64>() < noise.p_meas {
                        b = !b;
                    }
                    out.push(b);
                }
                Op::Reset(q) => {
                    sim.reset(*q, rng)?;
                    if noise.p_reset > 0.0 && rng.random::<f64>() < noise.p_reset {
                        sim.apply(&Gate::X(*q))?;
                    }
                }
                Op::ClassicControlled {
                    gate,
                    meas_index,
                    target_value,
                } => {
                    if *meas_index >= out.len() {
                        return Err(SimError::ClassicalBitOutOfRange {
                            bit: *meas_index,
                            available: out.len(),
                        });
                    }
                    if out[*meas_index] == *target_value {
                        sim.apply(gate)?;
                        crate::noise::apply_gate_noise(sim, gate, noise, rng)?;
                    }
                }
                Op::XFlip(q, p) => {
                    if *p > 0.0 && rng.random::<f64>() < *p {
                        sim.apply(&Gate::X(*q))?;
                    }
                }
                Op::YFlip(q, p) => {
                    if *p > 0.0 && rng.random::<f64>() < *p {
                        sim.apply(&Gate::Y(*q))?;
                    }
                }
                Op::ZFlip(q, p) => {
                    if *p > 0.0 && rng.random::<f64>() < *p {
                        sim.apply(&Gate::Z(*q))?;
                    }
                }
                Op::Depolarize1q(q, p) => {
                    if let Some(err) = crate::noise::sample_depolarizing_1q(*p, *q, rng) {
                        sim.apply(&err)?;
                    }
                }
                Op::Depolarize2q(a, b, p) => {
                    for err in crate::noise::sample_depolarizing_2q(*p, *a, *b, rng) {
                        sim.apply(&err)?;
                    }
                }
            }
        }
        Ok(out)
    }

    /// Runs the circuit on a simulator and returns the measurement outcomes
    /// in program order.
    pub fn run<S: Simulator + ?Sized>(
        &self,
        sim: &mut S,
        rng: &mut dyn RngCore,
    ) -> Result<Vec<bool>, SimError> {
        self.run_noisy(sim, &crate::noise::NoiseModel::none(), rng)
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
