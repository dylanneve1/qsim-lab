//! ONNX export of circuits, so a circuit can be opened in a neural-network
//! graph viewer such as [Netron](https://netron.app).
//!
//! The circuit becomes an ONNX graph in which every operation is a node and
//! every qubit is a chain of wire tensors:
//!
//! * graph inputs `q0 … q{n-1}` are the qubits (`complex64[2]`, one qubit's
//!   amplitudes, so a viewer shows a qubit-shaped type on every wire);
//! * each gate is a node in the custom domain [`DOMAIN`] (`H`, `CX`, `RZ`,
//!   `CCX`, …; Stim-style names, see [`op_type`]) that consumes the current
//!   wires of its qubits, in argument order, and produces new ones named
//!   `q{i}_{k}` (the `k`-th update of qubit `i`);
//! * `Measure` nodes also output a classical tensor `m{j}` (`bool` scalar),
//!   where `j` is the position in the measurement record; classically
//!   controlled gates (`IF_X`, `IF_P`, …) take `m{j}` as an extra input, so the
//!   classical dependency is drawn as an edge;
//! * optional detectors (`DETECTOR`, output `D{i}`) and logical observables
//!   (`OBSERVABLE_INCLUDE`, output `L{k}`) take the measurement tensors they
//!   read as inputs, which draws the detector structure of a QEC circuit;
//! * graph outputs are the final wire of every qubit, every measurement
//!   record entry, and every detector / observable.
//!
//! Node attributes carry the exact operation: `qubits` (INTS), angles
//! (`theta`, `phi`, `lambda` as FLOAT), noise probabilities (`p`),
//! measurement-record indices (`record`, `records`) and conditions
//! (`value`). FLOAT attributes are 32-bit, so every parametrised node also
//! has a doc string with the full `f64` value (and a `π` fraction when the
//! angle is one), which viewers show as the node's description.
//!
//! The file is written with a small built-in protobuf encoder (no
//! dependencies), IR version 8, opset imports `ai.onnx` 18 and
//! [`DOMAIN`] 1. Viewers do not execute the graph: the custom operators have
//! no ONNX schema, which is allowed for non-default domains.
//!
//! Lab notebook: `research/compiler/netron-export.md`.

use crate::circuit::{check_gate, Circuit, Op, SimError};
use crate::gate::Gate;
use std::f64::consts::PI;
use std::fmt::Write as _;

/// The ONNX operator domain used for every circuit node.
pub const DOMAIN: &str = "qsim";

/// ONNX IR version written to the model.
pub const IR_VERSION: i64 = 8;

/// Options for [`to_onnx`].
#[derive(Clone, Debug, Default)]
pub struct OnnxOptions {
    /// Graph name shown by viewers (default `"circuit"`).
    pub name: Option<String>,
    /// Free text stored as the model's doc string (e.g. how the circuit was
    /// built). A summary line with the circuit's statistics is always added.
    pub doc: Option<String>,
    /// Detectors, each a set of measurement-record indices (as in
    /// [`crate::io::stim::StimProgram::detectors`]); one `DETECTOR` node each.
    pub detectors: Vec<Vec<usize>>,
    /// Logical observables, each a set of measurement-record indices
    /// ([`crate::io::stim::StimProgram::observables`]); one
    /// `OBSERVABLE_INCLUDE` node each.
    pub observables: Vec<Vec<usize>>,
    /// Export only the first `max_ops` operations (`None`: all). Detectors and
    /// observables that read a measurement beyond the cut are dropped; the
    /// truncation is recorded in the model metadata.
    pub max_ops: Option<usize>,
    /// Extra `key = value` pairs for the model metadata (shown by viewers as
    /// model properties).
    pub metadata: Vec<(String, String)>,
}

/// The ONNX operator name of a gate (Stim-style, upper case).
pub fn op_type(g: &Gate) -> &'static str {
    use Gate::*;
    match g {
        I(_) => "I",
        H(_) => "H",
        X(_) => "X",
        Y(_) => "Y",
        Z(_) => "Z",
        S(_) => "S",
        Sdg(_) => "S_DAG",
        T(_) => "T",
        Tdg(_) => "T_DAG",
        Sx(_) => "SQRT_X",
        Sxdg(_) => "SQRT_X_DAG",
        Rx(..) => "RX",
        Ry(..) => "RY",
        Rz(..) => "RZ",
        Phase(..) => "P",
        U(..) => "U",
        Cnot(..) => "CX",
        Cz(..) => "CZ",
        Swap(..) => "SWAP",
        ISwap(..) => "ISWAP",
        ISwapdg(..) => "ISWAP_DAG",
        CPhase(..) => "CP",
        Ccx(..) => "CCX",
    }
}

/// The named real parameters of a gate, in the order of its constructor.
fn gate_params(g: &Gate) -> Vec<(&'static str, f64)> {
    use Gate::*;
    match *g {
        Rx(_, t) | Ry(_, t) | Rz(_, t) | Phase(_, t) | CPhase(_, _, t) => vec![("theta", t)],
        U(_, th, ph, la) => vec![("theta", th), ("phi", ph), ("lambda", la)],
        _ => Vec::new(),
    }
}

/// `x` as a short multiple of `π` (`"π/4"`, `"-3π/8"`, `"2π"`) when it is
/// one to within 1e-12 (denominators up to 2^20 or up to 12), else `None`.
pub fn pi_fraction(x: f64) -> Option<String> {
    if !x.is_finite() {
        return None;
    }
    if x == 0.0 {
        return Some("0".to_string());
    }
    let r = x / PI;
    let dens = (1..=12u64).chain((4..=20).map(|k| 1u64 << k));
    for q in dens {
        let p = (r * q as f64).round();
        if p == 0.0 || (r * q as f64 - p).abs() > 1e-12 * q as f64 {
            continue;
        }
        let (p, q) = reduce(p as i64, q as i64);
        let sign = if p < 0 { "-" } else { "" };
        let num = match p.unsigned_abs() {
            1 => "π".to_string(),
            a => format!("{a}π"),
        };
        return Some(if q == 1 {
            format!("{sign}{num}")
        } else {
            format!("{sign}{num}/{q}")
        });
    }
    None
}

fn reduce(p: i64, q: i64) -> (i64, i64) {
    let (mut a, mut b) = (p.unsigned_abs(), q.unsigned_abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    let g = a.max(1) as i64;
    (p / g, q / g)
}

/// Encodes `circuit` as an ONNX model (the bytes of a `.onnx` file).
///
/// Fails if an operation names a qubit outside the register, repeats a qubit,
/// or a classically controlled gate / detector / observable reads a
/// measurement that has not happened yet.
///
/// ```
/// use qsim_lab::{io::onnx, Circuit};
///
/// let mut c = Circuit::new(3);
/// c.h(0).cnot(0, 1).cnot(1, 2).measure_all();
/// let bytes = onnx::to_onnx(&c, &onnx::OnnxOptions::default()).unwrap();
/// std::fs::write(std::env::temp_dir().join("ghz.onnx"), &bytes).unwrap(); // open in Netron
/// assert!(bytes.len() > 100);
/// ```
pub fn to_onnx(circuit: &Circuit, opts: &OnnxOptions) -> Result<Vec<u8>, SimError> {
    let n = circuit.num_qubits;
    let ops_all = &circuit.ops;
    let n_ops = opts.max_ops.map_or(ops_all.len(), |m| m.min(ops_all.len()));
    let ops = &ops_all[..n_ops];

    // current wire name of every qubit, and how many times it was updated
    let mut wire: Vec<String> = (0..n).map(|q| format!("q{q}")).collect();
    let mut version = vec![0usize; n];
    let mut nodes = Vec::with_capacity(ops.len());
    let mut wires_made: Vec<String> = Vec::new();
    let mut n_meas = 0usize;

    let check_q = |q: usize| -> Result<(), SimError> {
        if q >= n {
            Err(SimError::QubitOutOfRange {
                qubit: q,
                num_qubits: n,
            })
        } else {
            Ok(())
        }
    };
    let check_rec = |j: usize, available: usize| -> Result<(), SimError> {
        if j >= available {
            Err(SimError::ClassicalBitOutOfRange { bit: j, available })
        } else {
            Ok(())
        }
    };

    for (idx, op) in ops.iter().enumerate() {
        // (op_type, qubits, extra classical inputs, extra classical outputs, attributes, doc)
        let mut node = Node::default();
        let qubits: Vec<usize> = match op {
            Op::Gate(g) => {
                check_gate(g, n)?;
                node.op_type = op_type(g).to_string();
                let params = gate_params(g);
                node.doc = param_doc(g, &params);
                for (name, v) in params {
                    node.attrs.push(Attr::Float(name, v as f32));
                }
                g.qubits()
            }
            Op::ClassicControlled {
                gate,
                meas_index,
                target_value,
            } => {
                check_gate(gate, n)?;
                check_rec(*meas_index, n_meas)?;
                node.op_type = format!("IF_{}", op_type(gate));
                let params = gate_params(gate);
                let mut doc = format!(
                    "{} applied if measurement m{} == {}",
                    op_type(gate),
                    meas_index,
                    u8::from(*target_value)
                );
                if let Some(d) = param_doc(gate, &params) {
                    doc.push_str("; ");
                    doc.push_str(&d);
                }
                node.doc = Some(doc);
                for (name, v) in params {
                    node.attrs.push(Attr::Float(name, v as f32));
                }
                node.attrs.push(Attr::Int("record", *meas_index as i64));
                node.attrs
                    .push(Attr::Int("value", i64::from(u8::from(*target_value))));
                node.extra_inputs.push(format!("m{meas_index}"));
                gate.qubits()
            }
            Op::Measure(q) => {
                check_q(*q)?;
                node.op_type = "Measure".to_string();
                node.attrs.push(Attr::Int("record", n_meas as i64));
                node.doc = Some(format!(
                    "Z-basis measurement of q{q}; outcome is measurement-record entry m{n_meas}"
                ));
                node.extra_outputs.push(format!("m{n_meas}"));
                n_meas += 1;
                vec![*q]
            }
            Op::Reset(q) => {
                check_q(*q)?;
                node.op_type = "Reset".to_string();
                vec![*q]
            }
            Op::XFlip(q, p) | Op::YFlip(q, p) | Op::ZFlip(q, p) | Op::Depolarize1q(q, p) => {
                check_q(*q)?;
                node.op_type = match op {
                    Op::XFlip(..) => "X_ERROR",
                    Op::YFlip(..) => "Y_ERROR",
                    Op::ZFlip(..) => "Z_ERROR",
                    _ => "DEPOLARIZE1",
                }
                .to_string();
                node.attrs.push(Attr::Float("p", *p as f32));
                node.doc = Some(format!("p = {p:?}"));
                vec![*q]
            }
            Op::Depolarize2q(a, b, p) => {
                check_q(*a)?;
                check_q(*b)?;
                if a == b {
                    return Err(SimError::NotSupported {
                        what: "DEPOLARIZE2 on a repeated qubit",
                    });
                }
                node.op_type = "DEPOLARIZE2".to_string();
                node.attrs.push(Attr::Float("p", *p as f32));
                node.doc = Some(format!("p = {p:?}"));
                vec![*a, *b]
            }
        };
        node.attrs.insert(
            0,
            Attr::Ints("qubits", qubits.iter().map(|&q| q as i64).collect()),
        );
        node.name = format!("{idx}:{}", node.op_type);
        for &q in &qubits {
            node.inputs.push(wire[q].clone());
        }
        node.inputs.append(&mut node.extra_inputs);
        for &q in &qubits {
            version[q] += 1;
            wire[q] = format!("q{q}_{}", version[q]);
            node.outputs.push(wire[q].clone());
            wires_made.push(wire[q].clone());
        }
        node.outputs.append(&mut node.extra_outputs);
        nodes.push(node);
    }

    // detectors and observables (dropped if they read past a max_ops cut)
    let truncated = n_ops < ops_all.len();
    let mut n_det = 0usize;
    let mut dropped = 0usize;
    for (kind, sets) in [
        ("DETECTOR", &opts.detectors),
        ("OBSERVABLE_INCLUDE", &opts.observables),
    ] {
        for (i, recs) in sets.iter().enumerate() {
            if let Some(&bad) = recs.iter().find(|&&j| j >= n_meas) {
                if truncated {
                    dropped += 1;
                    continue;
                }
                return Err(SimError::ClassicalBitOutOfRange {
                    bit: bad,
                    available: n_meas,
                });
            }
            let out = if kind == "DETECTOR" {
                n_det += 1;
                format!("D{i}")
            } else {
                format!("L{i}")
            };
            let mut node = Node {
                name: format!("{kind}:{i}"),
                op_type: kind.to_string(),
                inputs: recs.iter().map(|j| format!("m{j}")).collect(),
                outputs: vec![out],
                ..Node::default()
            };
            node.attrs.push(Attr::Ints(
                "records",
                recs.iter().map(|&j| j as i64).collect(),
            ));
            if kind != "DETECTOR" {
                node.attrs.push(Attr::Int("index", i as i64));
            }
            node.doc = Some(format!("parity of measurements {recs:?}"));
            nodes.push(node);
        }
    }

    // ----- encode -----
    let stats = Circuit {
        num_qubits: n,
        ops: ops.to_vec(),
    }
    .stats();
    let name = opts.name.clone().unwrap_or_else(|| "circuit".to_string());
    let mut doc = format!(
        "qsim-lab circuit: {} qubits, {} ops ({} gates: {} 1q, {} 2q, {} 3q; T-count {}), {} measurements, depth {}",
        n,
        stats.total_ops,
        stats.total_gates,
        stats.gates_1q,
        stats.gates_2q,
        stats.gates_3q,
        stats.t_gates,
        stats.measurements,
        stats.depth
    );
    if truncated {
        let _ = write!(
            doc,
            ". Truncated: first {n_ops} of {} ops exported",
            ops_all.len()
        );
    }
    if let Some(d) = &opts.doc {
        doc.push_str(".\n");
        doc.push_str(d);
    }

    let mut meta: Vec<(String, String)> = vec![
        ("qubits".into(), n.to_string()),
        ("ops".into(), stats.total_ops.to_string()),
        ("gates_1q".into(), stats.gates_1q.to_string()),
        ("gates_2q".into(), stats.gates_2q.to_string()),
        ("gates_3q".into(), stats.gates_3q.to_string()),
        ("t_count".into(), stats.t_gates.to_string()),
        ("measurements".into(), stats.measurements.to_string()),
        ("depth".into(), stats.depth.to_string()),
    ];
    if !opts.detectors.is_empty() || !opts.observables.is_empty() {
        meta.push(("detectors".into(), n_det.to_string()));
        meta.push(("observables".into(), opts.observables.len().to_string()));
    }
    if truncated {
        meta.push((
            "truncated".into(),
            format!("first {n_ops} of {} ops", ops_all.len()),
        ));
        if dropped > 0 {
            meta.push((
                "dropped_detectors_and_observables".into(),
                dropped.to_string(),
            ));
        }
    }
    meta.extend(opts.metadata.iter().cloned());

    let mut graph = Pb::default();
    for node in &nodes {
        graph.msg(1, |p| node.encode(p));
    }
    graph.str(2, &name);
    graph.str(10, &doc);
    for q in 0..n {
        graph.msg(11, |p| value_info(p, &format!("q{q}"), Kind::Qubit));
    }
    // an untouched qubit's output is its input wire (pass-through)
    for w in &wire {
        graph.msg(12, |p| value_info(p, w, Kind::Qubit));
    }
    for j in 0..n_meas {
        graph.msg(12, |p| value_info(p, &format!("m{j}"), Kind::Bit));
    }
    for node in &nodes {
        if node.op_type == "DETECTOR" || node.op_type == "OBSERVABLE_INCLUDE" {
            graph.msg(12, |p| value_info(p, &node.outputs[0], Kind::Bit));
        }
    }
    let finals: std::collections::HashSet<&str> = wire.iter().map(String::as_str).collect();
    for w in &wires_made {
        if !finals.contains(w.as_str()) {
            graph.msg(13, |p| value_info(p, w, Kind::Qubit));
        }
    }

    let mut model = Pb::default();
    model.int(1, IR_VERSION);
    model.str(2, "qsim-lab");
    model.str(3, env!("CARGO_PKG_VERSION"));
    model.str(4, "lab.qsim");
    model.int(5, 1);
    model.str(6, &doc);
    model.bytes(7, &graph.0);
    for (domain, version) in [("", 18i64), (DOMAIN, 1)] {
        model.msg(8, |p| {
            p.str(1, domain);
            p.int(2, version);
        });
    }
    for (k, v) in &meta {
        model.msg(14, |p| {
            p.str(1, k);
            p.str(2, v);
        });
    }
    Ok(model.0)
}

fn param_doc(g: &Gate, params: &[(&'static str, f64)]) -> Option<String> {
    if params.is_empty() {
        return None;
    }
    let mut s = String::new();
    let _ = write!(s, "{}(", op_type(g));
    for (i, (name, v)) in params.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        let _ = write!(s, "{name} = {v:?}");
        if let Some(f) = pi_fraction(*v) {
            let _ = write!(s, " = {f}");
        }
    }
    s.push(')');
    Some(s)
}

// ---------------------------------------------------------------------------
// minimal protobuf encoder (proto2 wire format, unpacked repeated scalars)

#[derive(Default)]
struct Pb(Vec<u8>);

impl Pb {
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.0.push((v as u8) | 0x80);
            v >>= 7;
        }
        self.0.push(v as u8);
    }
    fn key(&mut self, field: u32, wire_type: u8) {
        self.varint((u64::from(field) << 3) | u64::from(wire_type));
    }
    fn int(&mut self, field: u32, v: i64) {
        self.key(field, 0);
        self.varint(v as u64);
    }
    fn float(&mut self, field: u32, v: f32) {
        self.key(field, 5);
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn bytes(&mut self, field: u32, b: &[u8]) {
        self.key(field, 2);
        self.varint(b.len() as u64);
        self.0.extend_from_slice(b);
    }
    fn str(&mut self, field: u32, s: &str) {
        self.bytes(field, s.as_bytes());
    }
    fn msg(&mut self, field: u32, f: impl FnOnce(&mut Pb)) {
        let mut inner = Pb::default();
        f(&mut inner);
        self.bytes(field, &inner.0);
    }
}

enum Attr {
    Float(&'static str, f32),
    Int(&'static str, i64),
    Ints(&'static str, Vec<i64>),
}

#[derive(Default)]
struct Node {
    name: String,
    op_type: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    extra_inputs: Vec<String>,
    extra_outputs: Vec<String>,
    attrs: Vec<Attr>,
    doc: Option<String>,
}

impl Node {
    // onnx.NodeProto: input 1, output 2, name 3, op_type 4, attribute 5,
    // doc_string 6, domain 7
    fn encode(&self, p: &mut Pb) {
        for i in &self.inputs {
            p.str(1, i);
        }
        for o in &self.outputs {
            p.str(2, o);
        }
        p.str(3, &self.name);
        p.str(4, &self.op_type);
        for a in &self.attrs {
            // onnx.AttributeProto: name 1, f 2, i 3, ints 8, type 20
            // (AttributeType FLOAT = 1, INT = 2, INTS = 7)
            p.msg(5, |q| match a {
                Attr::Float(name, v) => {
                    q.str(1, name);
                    q.float(2, *v);
                    q.int(20, 1);
                }
                Attr::Int(name, v) => {
                    q.str(1, name);
                    q.int(3, *v);
                    q.int(20, 2);
                }
                Attr::Ints(name, vs) => {
                    q.str(1, name);
                    for v in vs {
                        q.int(8, *v);
                    }
                    q.int(20, 7);
                }
            });
        }
        if let Some(d) = &self.doc {
            p.str(6, d);
        }
        p.str(7, DOMAIN);
    }
}

#[derive(Clone, Copy)]
enum Kind {
    /// One qubit: `complex64[2]`.
    Qubit,
    /// One classical bit: `bool` scalar.
    Bit,
}

// onnx.ValueInfoProto: name 1, type 2; TypeProto: tensor_type 1;
// TypeProto.Tensor: elem_type 1, shape 2; TensorShapeProto: dim 1;
// Dimension: dim_value 1. TensorProto.DataType: BOOL = 9, COMPLEX64 = 14.
fn value_info(p: &mut Pb, name: &str, kind: Kind) {
    p.str(1, name);
    p.msg(2, |t| {
        t.msg(1, |tt| {
            match kind {
                Kind::Qubit => tt.int(1, 14),
                Kind::Bit => tt.int(1, 9),
            }
            tt.msg(2, |s| {
                if let Kind::Qubit = kind {
                    s.msg(1, |d| d.int(1, 2));
                }
            });
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pi_fractions() {
        assert_eq!(pi_fraction(PI / 4.0).as_deref(), Some("π/4"));
        assert_eq!(pi_fraction(-3.0 * PI / 8.0).as_deref(), Some("-3π/8"));
        assert_eq!(pi_fraction(2.0 * PI).as_deref(), Some("2π"));
        assert_eq!(pi_fraction(PI / 3.0).as_deref(), Some("π/3"));
        assert_eq!(pi_fraction(-PI / 1024.0).as_deref(), Some("-π/1024"));
        assert_eq!(pi_fraction(0.0).as_deref(), Some("0"));
        assert_eq!(pi_fraction(1.0), None);
        assert_eq!(pi_fraction(f64::NAN), None);
    }

    #[test]
    fn varint_encoding() {
        let mut p = Pb::default();
        p.varint(0);
        p.varint(1);
        p.varint(127);
        p.varint(128);
        p.varint(300);
        assert_eq!(p.0, vec![0, 1, 127, 0x80, 1, 0xac, 0x02]);
        let mut p = Pb::default();
        p.int(1, -1);
        // key 0x08, then -1 as a 10-byte varint
        assert_eq!(p.0.len(), 11);
        assert_eq!(p.0[0], 0x08);
        assert_eq!(p.0[10], 0x01);
    }
}
