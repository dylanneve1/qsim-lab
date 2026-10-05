//! A minimal circuit representation and the `Simulator` trait that every
//! backend implements.

use crate::gate::Gate;
use rand::{Rng, RngCore};
use std::fmt;

/// Errors a backend can report.
#[derive(Clone, Debug, PartialEq)]
pub enum SimError {
    /// The backend cannot represent this gate (e.g. T on a stabilizer tableau).
    Unsupported {
        /// Name of the backend that rejected the gate.
        backend: &'static str,
        /// The rejected gate.
        gate: Gate,
    },
    /// A qubit index is out of range.
    QubitOutOfRange {
        /// The offending qubit index.
        qubit: usize,
        /// Register width the index was checked against.
        num_qubits: usize,
    },
    /// A gate was given the same qubit twice.
    RepeatedQubit(Gate),
    /// The requested register would exceed the crate's memory cap.
    TooLarge {
        /// What was being allocated.
        what: &'static str,
        /// Bytes the allocation would need.
        bytes: u128,
        /// The cap, in bytes.
        limit: u128,
    },
    /// The Pauli-path simulator exceeded its term budget.
    TooManyTerms {
        /// Number of Pauli terms reached.
        terms: usize,
        /// The configured term budget.
        limit: usize,
    },
    /// The backend computes amplitudes of a unitary circuit and cannot apply
    /// the non-unitary operation (measurement, reset, noise channel or
    /// classically conditioned gate) at position `op_index` of `Circuit::ops`.
    MeasurementNotSupported {
        /// Name of the backend.
        backend: &'static str,
        /// Index into `Circuit::ops` of the unsupported operation.
        op_index: usize,
    },
    /// A classical bit index referenced by a conditional operation was out of range.
    ClassicalBitOutOfRange {
        /// The requested measurement-record index.
        bit: usize,
        /// Number of measurement outcomes recorded so far.
        available: usize,
    },
    /// The request needs something this circuit does not have (e.g. a
    /// state vector or amplitude of a circuit with measurements or noise).
    NotSupported {
        /// Description of the unsupported request.
        what: &'static str,
    },
    /// Failed to parse OpenQASM source.
    QasmError(String),
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
            SimError::MeasurementNotSupported { backend, op_index } => write!(
                f,
                "the {backend} backend cannot simulate the non-unitary operation at op {op_index}"
            ),
            SimError::ClassicalBitOutOfRange { bit, available } => {
                write!(
                    f,
                    "classical bit {bit} is out of range ({available} available)"
                )
            }
            SimError::NotSupported { what } => write!(f, "not supported: {what}"),
            SimError::QasmError(msg) => write!(f, "OpenQASM error: {msg}"),
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
    /// Number of qubits in the simulated register.
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
    /// Resets the simulator state back to |0...0> without reallocating buffers.
    fn reset_all(&mut self) -> Result<(), SimError>;
    /// Applies a sequence of gates in order. Backends that can fuse or
    /// reorder a run of gates (the state vector's cache-blocked executor)
    /// override this; the default applies them one by one. The result must
    /// equal gate-by-gate application (up to floating-point rounding).
    fn apply_gates(&mut self, gates: &[Gate]) -> Result<(), SimError> {
        for g in gates {
            self.apply(g)?;
        }
        Ok(())
    }
}

/// One instruction of a circuit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    /// Apply a unitary gate.
    Gate(Gate),
    /// Measure qubit `q` in the computational basis. Outcomes are appended to the
    ///  measurement record in program order; [`Op::ClassicControlled`] refers to them
    ///  by their position in that record.
    Measure(usize),
    /// Reset qubit `q` to |0>.
    Reset(usize),
    /// Classical condition: apply `gate` if measured bit `meas_index` equals `target_value`.
    ClassicControlled {
        /// The gate applied when the condition holds.
        gate: Gate,
        /// Index into the measurement record (0 = the first `Measure` executed).
        meas_index: usize,
        /// Outcome the recorded bit must equal for `gate` to be applied.
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
    /// Register width; every qubit index in `ops` must be below this.
    pub num_qubits: usize,
    /// Operations in program order.
    pub ops: Vec<Op>,
}

macro_rules! builder_1q {
    ($($name:ident => $variant:ident),*) => {$(
        /// Appends the corresponding single-qubit gate on qubit `q`.
        pub fn $name(&mut self, q: usize) -> &mut Self {
            self.gate(Gate::$variant(q))
        }
    )*};
}

impl Circuit {
    /// An empty circuit on `num_qubits` qubits.
    pub fn new(num_qubits: usize) -> Self {
        Circuit {
            num_qubits,
            ops: Vec::new(),
        }
    }

    /// Appends gate `g` (qubit indices are not checked here).
    pub fn gate(&mut self, g: Gate) -> &mut Self {
        self.ops.push(Op::Gate(g));
        self
    }

    builder_1q!(
        i => I,
        h => H,
        x => X,
        y => Y,
        z => Z,
        s => S,
        sdg => Sdg,
        t => T,
        tdg => Tdg,
        sx => Sx,
        sxdg => Sxdg
    );

    /// Appends `Rx(theta)` on qubit `q`; `theta` in radians.
    pub fn rx(&mut self, q: usize, theta: f64) -> &mut Self {
        self.gate(Gate::Rx(q, theta))
    }
    /// Appends `Ry(theta)` on qubit `q`; `theta` in radians.
    pub fn ry(&mut self, q: usize, theta: f64) -> &mut Self {
        self.gate(Gate::Ry(q, theta))
    }
    /// Appends `Rz(theta)` on qubit `q`; `theta` in radians.
    pub fn rz(&mut self, q: usize, theta: f64) -> &mut Self {
        self.gate(Gate::Rz(q, theta))
    }
    /// Appends the phase gate `diag(1, e^{i theta})` on qubit `q`.
    pub fn phase(&mut self, q: usize, theta: f64) -> &mut Self {
        self.gate(Gate::Phase(q, theta))
    }
    /// Appends the universal gate `U(theta, phi, lambda)` on qubit `q`.
    pub fn u(&mut self, q: usize, theta: f64, phi: f64, lambda: f64) -> &mut Self {
        self.gate(Gate::U(q, theta, phi, lambda))
    }
    /// Appends a CNOT with control `c` and target `t`.
    pub fn cnot(&mut self, c: usize, t: usize) -> &mut Self {
        self.gate(Gate::Cnot(c, t))
    }
    /// Appends a controlled-Z on qubits `a` and `b`.
    pub fn cz(&mut self, a: usize, b: usize) -> &mut Self {
        self.gate(Gate::Cz(a, b))
    }
    /// Appends a SWAP of qubits `a` and `b`.
    pub fn swap(&mut self, a: usize, b: usize) -> &mut Self {
        self.gate(Gate::Swap(a, b))
    }
    /// Appends an iSWAP on qubits `a` and `b`.
    pub fn iswap(&mut self, a: usize, b: usize) -> &mut Self {
        self.gate(Gate::ISwap(a, b))
    }
    /// Appends an inverse iSWAP on qubits `a` and `b`.
    pub fn iswapdg(&mut self, a: usize, b: usize) -> &mut Self {
        self.gate(Gate::ISwapdg(a, b))
    }
    /// Appends a controlled phase `diag(1, 1, 1, e^{i theta})` on qubits `a` and `b`.
    pub fn cphase(&mut self, a: usize, b: usize, theta: f64) -> &mut Self {
        self.gate(Gate::CPhase(a, b, theta))
    }
    /// Appends a Toffoli with controls `a`, `b` and target `t`.
    pub fn ccx(&mut self, a: usize, b: usize, t: usize) -> &mut Self {
        self.gate(Gate::Ccx(a, b, t))
    }
    /// Appends a computational-basis measurement of qubit `q`.
    pub fn measure(&mut self, q: usize) -> &mut Self {
        self.ops.push(Op::Measure(q));
        self
    }
    /// Appends a measurement of every qubit, in order `0..num_qubits`.
    pub fn measure_all(&mut self) -> &mut Self {
        for q in 0..self.num_qubits {
            self.measure(q);
        }
        self
    }
    /// Appends a reset of qubit `q` to |0>.
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

    /// Number of unitary gates (`Op::Gate`); conditional gates are not counted.
    pub fn num_gates(&self) -> usize {
        self.gates().count()
    }

    /// Number of T/T† gates.
    pub fn t_count(&self) -> usize {
        self.gates().filter(|g| g.is_t()).count()
    }

    /// True when every gate, including classically conditioned ones, is Clifford.
    ///  Measurements, resets and noise channels do not affect the result.
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
        // Maximal unitary gate segments (no measurement, reset, noise op or
        // classically conditioned op in between) are handed to the backend
        // as one batch so it can use its fastest executor. Gate noise draws
        // from the RNG after every gate, so noisy gates are applied singly.
        let batching = noise.p_1q <= 0.0 && noise.p_2q <= 0.0;
        let mut batch: Vec<Gate> = Vec::new();
        for op in &self.ops {
            if !batch.is_empty() && !matches!(op, Op::Gate(_)) {
                sim.apply_gates(&batch)?;
                batch.clear();
            }
            match op {
                Op::Gate(g) if batching => batch.push(*g),
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
        sim.apply_gates(&batch)?;
        Ok(out)
    }

    /// Calculates the circuit depth (the length of the critical path).
    pub fn depth(&self) -> usize {
        let mut wire_depth = vec![0usize; self.num_qubits];
        for op in &self.ops {
            let qs = match op {
                Op::Gate(g) => g.qubits(),
                Op::Measure(q)
                | Op::Reset(q)
                | Op::XFlip(q, _)
                | Op::YFlip(q, _)
                | Op::ZFlip(q, _)
                | Op::Depolarize1q(q, _) => vec![*q],
                Op::ClassicControlled { gate, .. } => gate.qubits(),
                Op::Depolarize2q(a, b, _) => vec![*a, *b],
            };
            let max_d = qs
                .iter()
                .map(|&q| wire_depth.get(q).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);
            let next_d = max_d + 1;
            for &q in &qs {
                if q < wire_depth.len() {
                    wire_depth[q] = next_d;
                }
            }
        }
        wire_depth.into_iter().max().unwrap_or(0)
    }

    /// Computes summary statistics of the circuit.
    pub fn stats(&self) -> CircuitStats {
        let mut g1 = 0;
        let mut g2 = 0;
        let mut g3 = 0;
        let mut cliff = 0;
        let mut t_count = 0;
        let mut meas = 0;
        for op in &self.ops {
            match op {
                Op::Gate(g) => {
                    match g.arity() {
                        1 => g1 += 1,
                        2 => g2 += 1,
                        3 => g3 += 1,
                        _ => {}
                    }
                    if g.is_clifford() {
                        cliff += 1;
                    }
                    if g.is_t() {
                        t_count += 1;
                    }
                }
                Op::Measure(_) => meas += 1,
                _ => {}
            }
        }
        CircuitStats {
            num_qubits: self.num_qubits,
            total_ops: self.ops.len(),
            total_gates: g1 + g2 + g3,
            depth: self.depth(),
            gates_1q: g1,
            gates_2q: g2,
            gates_3q: g3,
            clifford_gates: cliff,
            t_gates: t_count,
            measurements: meas,
        }
    }

    /// Peephole optimization pass: cancels adjacent self-inverses, merges
    /// rotations on identical axes, and eliminates identity/zero-angle gates.
    ///
    /// The result equals the original **up to a global phase**: `Rx`, `Ry`
    /// and `Rz` by a multiple of 2π are `-I`, not `I`, and are removed. That
    /// is unobservable for a whole circuit but matters if the output is used
    /// as a controlled sub-circuit. Only directly adjacent operations are
    /// merged; measurements, resets and conditional ops act as barriers.
    pub fn optimize(&self) -> Circuit {
        let mut ops: Vec<Op> = Vec::new();
        for op in &self.ops {
            match op {
                Op::Gate(g) => {
                    // Skip identity gates
                    if matches!(g, Gate::I(_)) {
                        continue;
                    }
                    // Skip near-zero rotation gates
                    match *g {
                        Gate::Rx(_, t) | Gate::Ry(_, t) | Gate::Rz(_, t) | Gate::Phase(_, t)
                            if (t % (2.0 * std::f64::consts::PI)).abs() < 1e-12 =>
                        {
                            continue;
                        }
                        Gate::CPhase(_, _, t)
                            if (t % (2.0 * std::f64::consts::PI)).abs() < 1e-12 =>
                        {
                            continue;
                        }
                        _ => {}
                    }
                    // Try to simplify against previous gates
                    let mut merged = false;
                    if let Some(Op::Gate(prev)) = ops.last().copied() {
                        if let Some(combined) = try_combine_gates(&prev, g) {
                            ops.pop();
                            if let Some(c) = combined {
                                ops.push(Op::Gate(c));
                            }
                            merged = true;
                        }
                    }
                    if !merged {
                        ops.push(Op::Gate(*g));
                    }
                }
                Op::Measure(q) => {
                    ops.push(Op::Measure(*q));
                }
                other => {
                    ops.push(*other);
                }
            }
        }
        Circuit {
            num_qubits: self.num_qubits,
            ops,
        }
    }

    /// Renders an ASCII text diagram of the circuit.
    pub fn draw(&self) -> String {
        if self.num_qubits == 0 {
            return String::new();
        }
        let mut wires: Vec<String> = (0..self.num_qubits)
            .map(|q| format!("q{q:<2}: ──"))
            .collect();

        for op in &self.ops {
            match op {
                Op::Gate(g) => match *g {
                    Gate::I(q) => {
                        append_1q(&mut wires, q, "[I]");
                    }
                    Gate::H(q) => {
                        append_1q(&mut wires, q, "[H]");
                    }
                    Gate::X(q) => {
                        append_1q(&mut wires, q, "[X]");
                    }
                    Gate::Y(q) => {
                        append_1q(&mut wires, q, "[Y]");
                    }
                    Gate::Z(q) => {
                        append_1q(&mut wires, q, "[Z]");
                    }
                    Gate::S(q) => {
                        append_1q(&mut wires, q, "[S]");
                    }
                    Gate::Sdg(q) => {
                        append_1q(&mut wires, q, "[S†]");
                    }
                    Gate::T(q) => {
                        append_1q(&mut wires, q, "[T]");
                    }
                    Gate::Tdg(q) => {
                        append_1q(&mut wires, q, "[T†]");
                    }
                    Gate::Sx(q) => {
                        append_1q(&mut wires, q, "[√X]");
                    }
                    Gate::Sxdg(q) => {
                        append_1q(&mut wires, q, "[√X†]");
                    }
                    Gate::Rx(q, _) => {
                        append_1q(&mut wires, q, "[Rx]");
                    }
                    Gate::Ry(q, _) => {
                        append_1q(&mut wires, q, "[Ry]");
                    }
                    Gate::Rz(q, _) => {
                        append_1q(&mut wires, q, "[Rz]");
                    }
                    Gate::Phase(q, _) => {
                        append_1q(&mut wires, q, "[P]");
                    }
                    Gate::U(q, _, _, _) => {
                        append_1q(&mut wires, q, "[U]");
                    }
                    Gate::Cnot(c, t) => {
                        append_2q(&mut wires, c, t, "■", "X");
                    }
                    Gate::Cz(a, b) => {
                        append_2q(&mut wires, a, b, "■", "■");
                    }
                    Gate::Swap(a, b) => {
                        append_2q(&mut wires, a, b, "X", "X");
                    }
                    Gate::ISwap(a, b) => {
                        append_2q(&mut wires, a, b, "iX", "iX");
                    }
                    Gate::ISwapdg(a, b) => {
                        append_2q(&mut wires, a, b, "iX†", "iX†");
                    }
                    Gate::CPhase(a, b, _) => {
                        append_2q(&mut wires, a, b, "■", "P");
                    }
                    Gate::Ccx(a, b, t) => {
                        append_3q(&mut wires, a, b, t, "■", "■", "X");
                    }
                },
                Op::Measure(q) => {
                    append_1q(&mut wires, *q, "[M]");
                }
                Op::Reset(q) => {
                    append_1q(&mut wires, *q, "[R]");
                }
                _ => {}
            }
        }

        // Add trailing wire end
        for w in &mut wires {
            w.push_str("──");
        }
        wires.join("\n")
    }

    /// Serializes this circuit into OpenQASM 2.0 format. Fails for operations
    /// OpenQASM 2.0 cannot express (classically conditioned gates, noise).
    pub fn to_qasm(&self) -> Result<String, SimError> {
        crate::io::qasm::to_qasm(self)
    }

    /// Parses an OpenQASM 2.0 program into a [`Circuit`].
    pub fn from_qasm(source: &str) -> Result<Circuit, SimError> {
        crate::io::qasm::from_qasm(source)
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

/// Summary statistics of a quantum circuit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CircuitStats {
    /// Register width.
    pub num_qubits: usize,
    /// Number of entries in `ops`, of every kind.
    pub total_ops: usize,
    /// Unitary gates (`Op::Gate`) of arity 1, 2 or 3.
    pub total_gates: usize,
    /// Circuit depth as computed by [`Circuit::depth`].
    pub depth: usize,
    /// Single-qubit unitary gates.
    pub gates_1q: usize,
    /// Two-qubit unitary gates.
    pub gates_2q: usize,
    /// Three-qubit unitary gates (Toffoli).
    pub gates_3q: usize,
    /// Unitary gates that are Clifford.
    pub clifford_gates: usize,
    /// T and T† gates.
    pub t_gates: usize,
    /// `Measure` operations.
    pub measurements: usize,
}

impl fmt::Display for Circuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.draw())
    }
}

fn try_combine_gates(a: &Gate, b: &Gate) -> Option<Option<Gate>> {
    use Gate::*;
    match (*a, *b) {
        // Self-inverses
        (H(q1), H(q2)) | (X(q1), X(q2)) | (Y(q1), Y(q2)) | (Z(q1), Z(q2)) if q1 == q2 => Some(None),
        (S(q1), Sdg(q2)) | (Sdg(q1), S(q2)) if q1 == q2 => Some(None),
        (T(q1), Tdg(q2)) | (Tdg(q1), T(q2)) if q1 == q2 => Some(None),
        (Sx(q1), Sxdg(q2)) | (Sxdg(q1), Sx(q2)) if q1 == q2 => Some(None),
        (Cnot(c1, t1), Cnot(c2, t2)) if c1 == c2 && t1 == t2 => Some(None),
        (Cz(a1, b1), Cz(a2, b2)) if (a1 == a2 && b1 == b2) || (a1 == b2 && b1 == a2) => Some(None),
        (Swap(a1, b1), Swap(a2, b2)) if (a1 == a2 && b1 == b2) || (a1 == b2 && b1 == a2) => {
            Some(None)
        }
        (ISwap(a1, b1), ISwapdg(a2, b2)) | (ISwapdg(a1, b1), ISwap(a2, b2))
            if (a1 == a2 && b1 == b2) || (a1 == b2 && b1 == a2) =>
        {
            Some(None)
        }
        // Combinations
        (S(q1), S(q2)) if q1 == q2 => Some(Some(Z(q1))),
        (Sdg(q1), Sdg(q2)) if q1 == q2 => Some(Some(Z(q1))),
        (T(q1), T(q2)) if q1 == q2 => Some(Some(S(q1))),
        (Tdg(q1), Tdg(q2)) if q1 == q2 => Some(Some(Sdg(q1))),
        // Continuous rotations
        (Rx(q1, t1), Rx(q2, t2)) if q1 == q2 => {
            let t = t1 + t2;
            if (t % (2.0 * std::f64::consts::PI)).abs() < 1e-12 {
                Some(None)
            } else {
                Some(Some(Rx(q1, t)))
            }
        }
        (Ry(q1, t1), Ry(q2, t2)) if q1 == q2 => {
            let t = t1 + t2;
            if (t % (2.0 * std::f64::consts::PI)).abs() < 1e-12 {
                Some(None)
            } else {
                Some(Some(Ry(q1, t)))
            }
        }
        (Rz(q1, t1), Rz(q2, t2)) if q1 == q2 => {
            let t = t1 + t2;
            if (t % (2.0 * std::f64::consts::PI)).abs() < 1e-12 {
                Some(None)
            } else {
                Some(Some(Rz(q1, t)))
            }
        }
        (Phase(q1, t1), Phase(q2, t2)) if q1 == q2 => {
            let t = t1 + t2;
            if (t % (2.0 * std::f64::consts::PI)).abs() < 1e-12 {
                Some(None)
            } else {
                Some(Some(Phase(q1, t)))
            }
        }
        (CPhase(a1, b1, t1), CPhase(a2, b2, t2))
            if (a1 == a2 && b1 == b2) || (a1 == b2 && b1 == a2) =>
        {
            let t = t1 + t2;
            if (t % (2.0 * std::f64::consts::PI)).abs() < 1e-12 {
                Some(None)
            } else {
                Some(Some(CPhase(a1, b1, t)))
            }
        }
        _ => None,
    }
}

fn append_1q(wires: &mut [String], target: usize, label: &str) {
    let tag = format!("{label:^5}");
    for (q, w) in wires.iter_mut().enumerate() {
        if q == target {
            w.push_str(&tag);
            w.push_str("──");
        } else {
            w.push_str("───────");
        }
    }
}

fn append_2q(wires: &mut [String], a: usize, b: usize, label_a: &str, label_b: &str) {
    let lo = a.min(b);
    let hi = a.max(b);
    let tag_a = format!("{label_a:^5}");
    let tag_b = format!("{label_b:^5}");
    for (q, w) in wires.iter_mut().enumerate() {
        if q == a {
            w.push_str(&tag_a);
            w.push_str("──");
        } else if q == b {
            w.push_str(&tag_b);
            w.push_str("──");
        } else if q > lo && q < hi {
            w.push_str("  │  ──");
        } else {
            w.push_str("───────");
        }
    }
}

fn append_3q(wires: &mut [String], a: usize, b: usize, c: usize, la: &str, lb: &str, lc: &str) {
    let lo = a.min(b).min(c);
    let hi = a.max(b).max(c);
    for (q, w) in wires.iter_mut().enumerate() {
        if q == a {
            w.push_str(&format!("{la:^5}──"));
        } else if q == b {
            w.push_str(&format!("{lb:^5}──"));
        } else if q == c {
            w.push_str(&format!("{lc:^5}──"));
        } else if q > lo && q < hi {
            w.push_str("  │  ──");
        } else {
            w.push_str("───────");
        }
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
    fn circuit_depth_and_stats() {
        let mut c = Circuit::new(3);
        c.h(0).cnot(0, 1).h(2).cnot(1, 2);
        assert_eq!(c.depth(), 3);
        let st = c.stats();
        assert_eq!(st.depth, 3);
        assert_eq!(st.gates_1q, 2);
        assert_eq!(st.gates_2q, 2);
        assert_eq!(st.total_gates, 4);
    }

    #[test]
    fn circuit_optimization_cancels_inverses() {
        let mut c = Circuit::new(2);
        c.h(0)
            .h(0)
            .x(1)
            .x(1)
            .t(0)
            .tdg(0)
            .s(1)
            .s(1)
            .cnot(0, 1)
            .cnot(0, 1);
        let opt = c.optimize();
        assert_eq!(opt.num_gates(), 1);
        assert_eq!(opt.ops[0], Op::Gate(Gate::Z(1)));
    }

    #[test]
    fn circuit_draw() {
        let mut c = Circuit::new(2);
        c.h(0).x(1).cnot(0, 1).measure_all();
        let s = c.draw();
        assert!(s.contains("q0 :"));
        assert!(s.contains("[H]"));
        assert!(s.contains("[X]"));
        assert!(s.contains("■"));
        assert!(s.contains("[M]"));
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
