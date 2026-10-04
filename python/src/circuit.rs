//! `qsimlab._native.CircuitCore`: the engine circuit plus the metadata the
//! Python layer carries (global phase, readout error, QEC annotations).
//!
//! The user-facing `qsimlab.Circuit` (python/qsimlab/circuit.py) wraps one
//! of these as `circuit._core`; native functions take `&PyCircuit`.

use crate::convert::{gate_from_parts, gate_parts, parse_condition};
use crate::errors::{circuit_err, map_sim_err, qerr, unsupported, value_err};
use crate::threads::heavy;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use qsim_lab::circuit::check_gate;
use qsim_lab::noise::NoiseModel;
use qsim_lab::{Circuit, Op, SimError};
use std::sync::Arc;

/// Everything a Python circuit is. Cheap to snapshot (`Arc`), copied on
/// write.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CircuitData {
    pub circuit: Circuit,
    /// Radians; multiplies every amplitude by `exp(i global_phase)`.
    pub global_phase: f64,
    /// Probability that each measurement result is flipped (Stim `M(p)`).
    pub readout_error: f64,
    /// Detectors: sets of absolute measurement indices.
    pub detectors: Vec<Vec<usize>>,
    /// `observables[k]`: measurement indices of logical observable `k`.
    pub observables: Vec<Vec<usize>>,
    /// Built with `repeat()` / Stim `REPEAT`: the repeat pass is worth running.
    pub has_repeats: bool,
}

impl CircuitData {
    pub fn new(n: usize) -> Self {
        CircuitData {
            circuit: Circuit::new(n),
            ..Default::default()
        }
    }

    pub fn num_measurements(&self) -> usize {
        self.circuit
            .ops
            .iter()
            .filter(|o| matches!(o, Op::Measure(_)))
            .count()
    }

    /// The readout error as an engine noise model (gate noise off).
    pub fn noise_model(&self) -> NoiseModel {
        NoiseModel {
            p_meas: self.readout_error,
            ..NoiseModel::default()
        }
    }

    /// True if every op is a gate.
    pub fn is_unitary(&self) -> bool {
        self.circuit.ops.iter().all(|o| matches!(o, Op::Gate(_)))
    }

    /// The circuit with *terminal* measurements removed (measurements after
    /// which nothing touches the qubit). Other non-gate ops are kept.
    pub fn without_terminal_measurements(&self) -> Circuit {
        let c = &self.circuit;
        let conditionals = c
            .ops
            .iter()
            .any(|o| matches!(o, Op::ClassicControlled { .. }));
        let mut touched = vec![false; c.num_qubits];
        let mut keep = vec![true; c.ops.len()];
        for (i, op) in c.ops.iter().enumerate().rev() {
            match op {
                Op::Measure(q) if !touched[*q] && !conditionals => keep[i] = false,
                _ => {
                    for q in op_qubits(op) {
                        touched[q] = true;
                    }
                }
            }
        }
        Circuit {
            num_qubits: c.num_qubits,
            ops: c
                .ops
                .iter()
                .zip(keep)
                .filter_map(|(o, k)| k.then_some(*o))
                .collect(),
        }
    }
}

pub fn op_qubits(op: &Op) -> Vec<usize> {
    match op {
        Op::Gate(g) => g.qubits(),
        Op::ClassicControlled { gate, .. } => gate.qubits(),
        Op::Measure(q)
        | Op::Reset(q)
        | Op::XFlip(q, _)
        | Op::YFlip(q, _)
        | Op::ZFlip(q, _)
        | Op::Depolarize1q(q, _) => vec![*q],
        Op::Depolarize2q(a, b, _) => vec![*a, *b],
    }
}

fn remap_gate(g: &qsim_lab::Gate, map: &[usize]) -> qsim_lab::Gate {
    let (name, qs, ps) = gate_parts(g);
    let qs: Vec<usize> = qs.iter().map(|&q| map[q]).collect();
    gate_from_parts(name, &qs, &ps).expect("remapped gate keeps its arity")
}

fn remap_op(op: &Op, map: &[usize], meas_offset: usize) -> Op {
    match *op {
        Op::Gate(g) => Op::Gate(remap_gate(&g, map)),
        Op::Measure(q) => Op::Measure(map[q]),
        Op::Reset(q) => Op::Reset(map[q]),
        Op::ClassicControlled {
            gate,
            meas_index,
            target_value,
        } => Op::ClassicControlled {
            gate: remap_gate(&gate, map),
            meas_index: meas_index + meas_offset,
            target_value,
        },
        Op::XFlip(q, p) => Op::XFlip(map[q], p),
        Op::YFlip(q, p) => Op::YFlip(map[q], p),
        Op::ZFlip(q, p) => Op::ZFlip(map[q], p),
        Op::Depolarize1q(q, p) => Op::Depolarize1q(map[q], p),
        Op::Depolarize2q(a, b, p) => Op::Depolarize2q(map[a], map[b], p),
    }
}

/// Python-visible description of one op: `(name, qubits, params, c_if)`.
pub type OpParts = (&'static str, Vec<usize>, Vec<f64>, Option<(usize, bool)>);

/// The [`OpParts`] of an engine op.
pub fn op_parts(op: &Op) -> OpParts {
    match *op {
        Op::Gate(g) => {
            let (n, q, p) = gate_parts(&g);
            (n, q, p, None)
        }
        Op::ClassicControlled {
            gate,
            meas_index,
            target_value,
        } => {
            let (n, q, p) = gate_parts(&gate);
            (n, q, p, Some((meas_index, target_value)))
        }
        Op::Measure(q) => ("measure", vec![q], vec![], None),
        Op::Reset(q) => ("reset", vec![q], vec![], None),
        Op::XFlip(q, p) => ("x_error", vec![q], vec![p], None),
        Op::YFlip(q, p) => ("y_error", vec![q], vec![p], None),
        Op::ZFlip(q, p) => ("z_error", vec![q], vec![p], None),
        Op::Depolarize1q(q, p) => ("depolarize1", vec![q], vec![p], None),
        Op::Depolarize2q(a, b, p) => ("depolarize2", vec![a, b], vec![p], None),
    }
}

fn check_prob(name: &str, p: f64) -> PyResult<()> {
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        return Err(circuit_err(format!(
            "{name}: probability {p} is not in [0, 1]"
        )));
    }
    Ok(())
}

/// The native circuit. Use `qsimlab.Circuit`; this class is private.
#[pyclass(name = "CircuitCore", module = "qsimlab._native", skip_from_py_object)]
#[derive(Clone)]
pub struct PyCircuit {
    pub data: Arc<CircuitData>,
}

impl PyCircuit {
    pub fn from_data(data: CircuitData) -> Self {
        PyCircuit {
            data: Arc::new(data),
        }
    }
    /// Read access for other binding modules.
    pub fn data(&self) -> &CircuitData {
        &self.data
    }
    /// A cheap snapshot that can be moved into a GIL-free closure.
    pub fn snapshot(&self) -> Arc<CircuitData> {
        Arc::clone(&self.data)
    }
    fn data_mut(&mut self) -> &mut CircuitData {
        Arc::make_mut(&mut self.data)
    }
    fn check_qubits(&self, qs: &[usize]) -> PyResult<()> {
        let n = self.data.circuit.num_qubits;
        for (i, &q) in qs.iter().enumerate() {
            if q >= n {
                return Err(map_sim_err(SimError::QubitOutOfRange {
                    qubit: q,
                    num_qubits: n,
                }));
            }
            if qs[..i].contains(&q) {
                return Err(circuit_err(format!("qubit {q} is used twice")));
            }
        }
        Ok(())
    }
    fn check_meas_indices(&self, idx: &[usize]) -> PyResult<()> {
        let m = self.data.num_measurements();
        if let Some(&k) = idx.iter().find(|&&k| k >= m) {
            return Err(map_sim_err(SimError::ClassicalBitOutOfRange {
                bit: k,
                available: m,
            }));
        }
        Ok(())
    }
}

#[pymethods]
impl PyCircuit {
    #[new]
    fn py_new(num_qubits: usize) -> Self {
        PyCircuit::from_data(CircuitData::new(num_qubits))
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.data.circuit.num_qubits
    }
    #[getter]
    fn num_ops(&self) -> usize {
        self.data.circuit.ops.len()
    }
    #[getter]
    fn num_measurements(&self) -> usize {
        self.data.num_measurements()
    }
    #[getter]
    fn get_global_phase(&self) -> f64 {
        self.data.global_phase
    }
    #[setter]
    fn set_global_phase(&mut self, v: f64) -> PyResult<()> {
        if !v.is_finite() {
            return Err(value_err("global_phase must be finite"));
        }
        self.data_mut().global_phase = v;
        Ok(())
    }
    #[getter]
    fn get_readout_error(&self) -> f64 {
        self.data.readout_error
    }
    #[setter]
    fn set_readout_error(&mut self, p: f64) -> PyResult<()> {
        check_prob("readout_error", p)?;
        self.data_mut().readout_error = p;
        Ok(())
    }
    #[getter]
    fn detectors(&self) -> Vec<Vec<usize>> {
        self.data.detectors.clone()
    }
    #[getter]
    fn observables(&self) -> Vec<Vec<usize>> {
        self.data.observables.clone()
    }
    #[getter]
    fn get_has_repeats(&self) -> bool {
        self.data.has_repeats
    }
    #[setter]
    fn set_has_repeats(&mut self, v: bool) {
        self.data_mut().has_repeats = v;
    }

    /// Appends a gate. `c_if`: `None`, a measurement index, or `(index, value)`.
    #[pyo3(signature = (name, qubits, params=Vec::new(), c_if=None))]
    fn append_gate(
        &mut self,
        name: &str,
        qubits: Vec<usize>,
        params: Vec<f64>,
        c_if: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let g = gate_from_parts(name, &qubits, &params)?;
        let cond = parse_condition(c_if)?;
        check_gate(&g, self.data.circuit.num_qubits).map_err(map_sim_err)?;
        if let Some((k, _)) = cond {
            self.check_meas_indices(&[k])?;
        }
        let op = match cond {
            None => Op::Gate(g),
            Some((meas_index, target_value)) => Op::ClassicControlled {
                gate: g,
                meas_index,
                target_value,
            },
        };
        self.data_mut().circuit.ops.push(op);
        Ok(())
    }

    /// Appends one measurement per qubit (in order); returns the index of
    /// the first new measurement.
    fn append_measure(&mut self, qubits: Vec<usize>) -> PyResult<usize> {
        for &q in &qubits {
            self.check_qubits(&[q])?;
        }
        let first = self.data.num_measurements();
        let c = &mut self.data_mut().circuit;
        for q in qubits {
            c.ops.push(Op::Measure(q));
        }
        Ok(first)
    }

    fn append_reset(&mut self, qubits: Vec<usize>) -> PyResult<()> {
        for &q in &qubits {
            self.check_qubits(&[q])?;
        }
        let c = &mut self.data_mut().circuit;
        for q in qubits {
            c.ops.push(Op::Reset(q));
        }
        Ok(())
    }

    /// Noise channel: `x_error`, `y_error`, `z_error`, `depolarize1` (one
    /// qubit each), `depolarize2` (qubit pairs).
    fn append_noise(&mut self, kind: &str, qubits: Vec<usize>, p: f64) -> PyResult<()> {
        check_prob(kind, p)?;
        let mut ops = Vec::new();
        match kind {
            "depolarize2" => {
                if qubits.len() % 2 != 0 {
                    return Err(circuit_err("depolarize2 takes qubit pairs"));
                }
                for pair in qubits.chunks(2) {
                    self.check_qubits(pair)?;
                    ops.push(Op::Depolarize2q(pair[0], pair[1], p));
                }
            }
            "x_error" | "y_error" | "z_error" | "depolarize1" => {
                for &q in &qubits {
                    self.check_qubits(&[q])?;
                    ops.push(match kind {
                        "x_error" => Op::XFlip(q, p),
                        "y_error" => Op::YFlip(q, p),
                        "z_error" => Op::ZFlip(q, p),
                        _ => Op::Depolarize1q(q, p),
                    });
                }
            }
            _ => return Err(circuit_err(format!("unknown noise channel '{kind}'"))),
        }
        self.data_mut().circuit.ops.extend(ops);
        Ok(())
    }

    fn add_detector(&mut self, meas: Vec<usize>) -> PyResult<usize> {
        self.check_meas_indices(&meas)?;
        let d = self.data_mut();
        d.detectors.push(meas);
        Ok(d.detectors.len() - 1)
    }

    fn add_observable(&mut self, index: usize, meas: Vec<usize>) -> PyResult<()> {
        self.check_meas_indices(&meas)?;
        let d = self.data_mut();
        if d.observables.len() <= index {
            d.observables.resize(index + 1, Vec::new());
        }
        d.observables[index].extend(meas);
        Ok(())
    }

    /// Appends `other` `reps` times, its qubit `i` placed on `qubits[i]`
    /// (default: the identity map). Measurement indices in conditionals
    /// and QEC annotations are shifted.
    #[pyo3(signature = (other, qubits=None, reps=1))]
    fn extend(
        &mut self,
        other: &PyCircuit,
        qubits: Option<Vec<usize>>,
        reps: usize,
    ) -> PyResult<()> {
        let o = other.snapshot();
        let n = self.data.circuit.num_qubits;
        let map = match qubits {
            Some(m) => {
                if m.len() != o.circuit.num_qubits {
                    return Err(circuit_err(format!(
                        "qubit map has {} entries for a {}-qubit circuit",
                        m.len(),
                        o.circuit.num_qubits
                    )));
                }
                self.check_qubits(&m)?;
                m
            }
            None => {
                if o.circuit.num_qubits > n {
                    return Err(circuit_err(format!(
                        "cannot append a {}-qubit circuit to a {n}-qubit one",
                        o.circuit.num_qubits
                    )));
                }
                (0..o.circuit.num_qubits).collect()
            }
        };
        let total = o.circuit.ops.len().saturating_mul(reps);
        if total > 2_000_000_000 {
            return Err(qerr(
                "ResourceLimitError",
                "repeat would create more than 2e9 ops",
            ));
        }
        let per_rep = o.num_measurements();
        let d = self.data_mut();
        d.circuit.ops.reserve(total);
        let base = d.num_measurements();
        for rep in 0..reps {
            let off = base + rep * per_rep;
            for op in &o.circuit.ops {
                d.circuit.ops.push(remap_op(op, &map, off));
            }
            for det in &o.detectors {
                d.detectors.push(det.iter().map(|k| k + off).collect());
            }
            for (k, obs) in o.observables.iter().enumerate() {
                if d.observables.len() <= k {
                    d.observables.resize(k + 1, Vec::new());
                }
                d.observables[k].extend(obs.iter().map(|m| m + off));
            }
        }
        d.global_phase += o.global_phase * reps as f64;
        d.has_repeats |= o.has_repeats || reps > 1;
        if o.readout_error > 0.0 && d.readout_error != o.readout_error {
            if d.readout_error == 0.0 {
                d.readout_error = o.readout_error;
            } else {
                return Err(circuit_err(
                    "cannot combine circuits with different readout_error",
                ));
            }
        }
        Ok(())
    }

    /// `[(name, qubits, params, c_if)]` in program order.
    fn instructions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for op in &self.data.circuit.ops {
            let (name, qs, ps, cond) = op_parts(op);
            let t = PyTuple::new(
                py,
                [
                    name.into_pyobject(py)?.into_any(),
                    PyTuple::new(py, qs)?.into_any(),
                    PyTuple::new(py, ps)?.into_any(),
                    cond.into_pyobject(py)?.into_any(),
                ],
            )?;
            list.append(t)?;
        }
        Ok(list)
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let snap = self.snapshot();
        let s = heavy(py, Some(1), move || snap.circuit.stats());
        let c = &self.data.circuit;
        let d = PyDict::new(py);
        d.set_item("num_qubits", s.num_qubits)?;
        d.set_item("total_ops", s.total_ops)?;
        d.set_item("total_gates", s.total_gates)?;
        d.set_item("depth", s.depth)?;
        d.set_item("gates_1q", s.gates_1q)?;
        d.set_item("gates_2q", s.gates_2q)?;
        d.set_item("gates_3q", s.gates_3q)?;
        d.set_item("clifford_gates", s.clifford_gates)?;
        d.set_item("t_gates", s.t_gates)?;
        d.set_item("measurements", s.measurements)?;
        let count = |f: &dyn Fn(&Op) -> bool| c.ops.iter().filter(|o| f(o)).count();
        d.set_item(
            "noise_channels",
            count(&|o| {
                matches!(
                    o,
                    Op::XFlip(..)
                        | Op::YFlip(..)
                        | Op::ZFlip(..)
                        | Op::Depolarize1q(..)
                        | Op::Depolarize2q(..)
                )
            }),
        )?;
        d.set_item("resets", count(&|o| matches!(o, Op::Reset(_))))?;
        d.set_item(
            "conditionals",
            count(&|o| matches!(o, Op::ClassicControlled { .. })),
        )?;
        d.set_item("is_clifford", c.is_clifford())?;
        d.set_item("is_unitary", self.data.is_unitary())?;
        d.set_item("detectors", self.data.detectors.len())?;
        d.set_item("observables", self.data.observables.len())?;
        Ok(d)
    }

    fn draw(&self) -> String {
        self.data.circuit.draw()
    }

    fn to_qasm(&self) -> PyResult<String> {
        self.data.circuit.to_qasm().map_err(|e| match e {
            SimError::QasmError(m) => unsupported(format!("OpenQASM 2 export: {m}")),
            e => map_sim_err(e),
        })
    }

    #[staticmethod]
    fn from_qasm(py: Python<'_>, source: String) -> PyResult<PyCircuit> {
        let r = heavy(py, Some(1), move || qsim_lab::qasm::from_qasm(&source));
        Ok(PyCircuit::from_data(CircuitData {
            circuit: r.map_err(map_sim_err)?,
            ..Default::default()
        }))
    }

    fn to_stim(&self) -> PyResult<String> {
        let d = &self.data;
        qsim_lab::stim_io::to_stim(&d.circuit, &d.noise_model(), &d.detectors, &d.observables)
            .map_err(|e| unsupported(format!("Stim export: {}", e.0)))
    }

    #[staticmethod]
    fn from_stim(py: Python<'_>, source: String) -> PyResult<PyCircuit> {
        let has_repeats = source.contains("REPEAT");
        let r = heavy(py, Some(1), move || qsim_lab::stim_io::parse_stim(&source));
        let p = r.map_err(|e| qerr("ParseError", e.0))?;
        if p.noise.p_1q != 0.0 || p.noise.p_2q != 0.0 || p.noise.p_reset != 0.0 {
            return Err(unsupported("Stim import produced implicit gate noise"));
        }
        Ok(PyCircuit::from_data(CircuitData {
            circuit: p.circuit,
            global_phase: 0.0,
            readout_error: p.noise.p_meas,
            detectors: p.detectors,
            observables: p.observables,
            has_repeats,
        }))
    }

    fn copy(&self) -> PyCircuit {
        PyCircuit::from_data((*self.data).clone())
    }

    /// The inverse of a unitary circuit (gates reversed and inverted,
    /// global phase negated).
    fn inverse(&self) -> PyResult<PyCircuit> {
        if !self.data.is_unitary() {
            return Err(unsupported(
                "inverse() needs a unitary circuit (no measurement, reset, conditional or noise)",
            ));
        }
        Ok(PyCircuit::from_data(CircuitData {
            circuit: self.data.circuit.inverse(),
            global_phase: -self.data.global_phase,
            ..Default::default()
        }))
    }

    /// The circuit without terminal measurements (and their detectors and
    /// observables).
    fn remove_final_measurements(&self) -> PyCircuit {
        PyCircuit::from_data(CircuitData {
            circuit: self.data.without_terminal_measurements(),
            global_phase: self.data.global_phase,
            readout_error: 0.0,
            has_repeats: self.data.has_repeats,
            ..Default::default()
        })
    }

    fn __eq__(&self, other: &PyCircuit) -> bool {
        self.data == other.data
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCircuit>()?;
    let gates = PyDict::new(m.py());
    for &(name, nq, np) in crate::convert::GATES {
        gates.set_item(name, (nq, np))?;
    }
    m.add("GATES", gates)?;
    let aliases = PyDict::new(m.py());
    for &(a, c) in crate::convert::ALIASES {
        aliases.set_item(a, c)?;
    }
    m.add("GATE_ALIASES", aliases)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_measurements_are_the_ones_nothing_follows() {
        let mut d = CircuitData::new(2);
        d.circuit.h(0).measure(0).cnot(0, 1).measure(0).measure(1);
        let c = d.without_terminal_measurements();
        // the first measure of qubit 0 is followed by a CNOT: kept
        assert_eq!(c.ops.len(), 3);
        assert!(matches!(c.ops[1], Op::Measure(0)));
    }
}
