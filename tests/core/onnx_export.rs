//! ONNX export (`qsim_lab::io::onnx`): the bytes are decoded with an
//! independent minimal protobuf reader and checked against the circuit:
//! node per op, SSA wiring (each wire produced once and consumed at most
//! once, every input defined before use), attributes, classical edges,
//! detectors, truncation and validation errors. A lossless reconstruction of
//! the gate list from the graph closes the loop.

use qsim_lab::circuit::{Circuit, Op, SimError};
use qsim_lab::gate::Gate;
use qsim_lab::io::onnx::{to_onnx, OnnxOptions};
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

// ---------------------------------------------------------------- decoder

#[derive(Debug, Clone)]
enum Val {
    Varint(u64),
    Fixed32(u32),
    Fixed64(u64),
    Bytes(Vec<u8>),
}

fn varint(b: &[u8], i: &mut usize) -> u64 {
    let mut v = 0u64;
    let mut shift = 0;
    loop {
        let byte = b[*i];
        *i += 1;
        v |= u64::from(byte & 0x7f) << shift;
        if byte < 0x80 {
            return v;
        }
        shift += 7;
        assert!(shift < 64, "varint too long");
    }
}

/// All (field, value) pairs of one message, in order.
fn fields(b: &[u8]) -> Vec<(u32, Val)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let key = varint(b, &mut i);
        let field = (key >> 3) as u32;
        let v = match key & 7 {
            0 => Val::Varint(varint(b, &mut i)),
            1 => {
                let v = u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
                i += 8;
                Val::Fixed64(v)
            }
            2 => {
                let len = varint(b, &mut i) as usize;
                let v = b[i..i + len].to_vec();
                i += len;
                Val::Bytes(v)
            }
            5 => {
                let v = u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
                i += 4;
                Val::Fixed32(v)
            }
            wt => panic!("unexpected wire type {wt}"),
        };
        out.push((field, v));
    }
    assert_eq!(i, b.len(), "message overran its length");
    out
}

fn get_all(m: &[(u32, Val)], f: u32) -> Vec<&Val> {
    m.iter().filter(|(k, _)| *k == f).map(|(_, v)| v).collect()
}
fn bytes(v: &Val) -> &[u8] {
    match v {
        Val::Bytes(b) => b,
        other => panic!("expected bytes, got {other:?}"),
    }
}
fn string(v: &Val) -> String {
    String::from_utf8(bytes(v).to_vec()).expect("utf8")
}
fn int(v: &Val) -> i64 {
    match v {
        Val::Varint(x) => *x as i64,
        other => panic!("expected varint, got {other:?}"),
    }
}
fn strings(m: &[(u32, Val)], f: u32) -> Vec<String> {
    get_all(m, f).into_iter().map(string).collect()
}
fn one_string(m: &[(u32, Val)], f: u32) -> Option<String> {
    let v = get_all(m, f);
    assert!(v.len() <= 1, "field {f} repeated");
    v.first().map(|v| string(v))
}

#[derive(Debug, Clone)]
enum A {
    F(f32),
    I(i64),
    Is(Vec<i64>),
}

#[derive(Debug, Clone)]
struct DNode {
    name: String,
    op: String,
    domain: String,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attrs: HashMap<String, A>,
    doc: Option<String>,
}

#[derive(Debug)]
struct DModel {
    ir_version: i64,
    producer: String,
    opsets: Vec<(String, i64)>,
    metadata: HashMap<String, String>,
    graph_name: String,
    nodes: Vec<DNode>,
    inputs: Vec<(String, i64, Vec<i64>)>,
    outputs: Vec<(String, i64, Vec<i64>)>,
    value_info: Vec<(String, i64, Vec<i64>)>,
}

fn decode_value_info(b: &[u8]) -> (String, i64, Vec<i64>) {
    let m = fields(b);
    let name = one_string(&m, 1).unwrap();
    let ty = fields(bytes(get_all(&m, 2)[0]));
    let tensor = fields(bytes(get_all(&ty, 1)[0]));
    let elem = int(get_all(&tensor, 1)[0]);
    let shape = fields(bytes(get_all(&tensor, 2)[0]));
    let dims = get_all(&shape, 1)
        .into_iter()
        .map(|d| int(get_all(&fields(bytes(d)), 1)[0]))
        .collect();
    (name, elem, dims)
}

fn decode(model: &[u8]) -> DModel {
    let m = fields(model);
    let ir_version = int(get_all(&m, 1)[0]);
    let producer = one_string(&m, 2).unwrap();
    let opsets = get_all(&m, 8)
        .into_iter()
        .map(|o| {
            let f = fields(bytes(o));
            (one_string(&f, 1).unwrap_or_default(), int(get_all(&f, 2)[0]))
        })
        .collect();
    let metadata = get_all(&m, 14)
        .into_iter()
        .map(|e| {
            let f = fields(bytes(e));
            (one_string(&f, 1).unwrap(), one_string(&f, 2).unwrap())
        })
        .collect();
    let g = fields(bytes(get_all(&m, 7)[0]));
    let graph_name = one_string(&g, 2).unwrap();
    let nodes = get_all(&g, 1)
        .into_iter()
        .map(|nb| {
            let n = fields(bytes(nb));
            let attrs = get_all(&n, 5)
                .into_iter()
                .map(|ab| {
                    let a = fields(bytes(ab));
                    let name = one_string(&a, 1).unwrap();
                    let ty = int(get_all(&a, 20)[0]);
                    let val = match ty {
                        1 => match get_all(&a, 2)[0] {
                            Val::Fixed32(x) => A::F(f32::from_bits(*x)),
                            other => panic!("float attr {other:?}"),
                        },
                        2 => A::I(int(get_all(&a, 3)[0])),
                        7 => A::Is(get_all(&a, 8).into_iter().map(int).collect()),
                        t => panic!("attr type {t}"),
                    };
                    (name, val)
                })
                .collect();
            DNode {
                name: one_string(&n, 3).unwrap(),
                op: one_string(&n, 4).unwrap(),
                domain: one_string(&n, 7).unwrap_or_default(),
                inputs: strings(&n, 1),
                outputs: strings(&n, 2),
                attrs,
                doc: one_string(&n, 6),
            }
        })
        .collect();
    let vi = |f: u32| {
        get_all(&g, f)
            .into_iter()
            .map(|b| decode_value_info(bytes(b)))
            .collect::<Vec<_>>()
    };
    DModel {
        ir_version,
        producer,
        opsets,
        metadata,
        graph_name,
        nodes,
        inputs: vi(11),
        outputs: vi(12),
        value_info: vi(13),
    }
}

// ---------------------------------------------------------------- checks

/// SSA + linearity: every tensor produced once (or a graph input), every
/// node input defined earlier, every quantum wire consumed at most once,
/// graph outputs defined, every value typed.
fn check_wiring(d: &DModel) {
    let mut defined: HashSet<String> = d.inputs.iter().map(|(n, _, _)| n.clone()).collect();
    let mut consumed_quantum: HashSet<String> = HashSet::new();
    let typed: HashMap<String, (i64, Vec<i64>)> = d
        .inputs
        .iter()
        .chain(&d.outputs)
        .chain(&d.value_info)
        .map(|(n, e, s)| (n.clone(), (*e, s.clone())))
        .collect();
    let mut names = HashSet::new();
    for n in &d.nodes {
        assert!(names.insert(n.name.clone()), "duplicate node name {}", n.name);
        assert_eq!(n.domain, "qsim");
        for i in &n.inputs {
            assert!(defined.contains(i), "{} reads undefined {i}", n.name);
            let (elem, _) = typed.get(i).unwrap_or_else(|| panic!("untyped {i}"));
            if *elem == 14 {
                assert!(
                    consumed_quantum.insert(i.clone()),
                    "quantum wire {i} consumed twice (at {})",
                    n.name
                );
            }
        }
        for o in &n.outputs {
            assert!(defined.insert(o.clone()), "{o} produced twice");
            assert!(typed.contains_key(o), "untyped {o}");
        }
    }
    for (o, _, _) in &d.outputs {
        assert!(defined.contains(o), "graph output {o} never produced");
    }
    // a quantum wire that is not a graph output must be consumed (no dangling wires)
    let outs: HashSet<&str> = d.outputs.iter().map(|(n, _, _)| n.as_str()).collect();
    for (name, (elem, _)) in &typed {
        if *elem == 14 && !outs.contains(name.as_str()) {
            assert!(consumed_quantum.contains(name), "dangling wire {name}");
        }
    }
}

fn qubits_attr(n: &DNode) -> Vec<usize> {
    match n.attrs.get("qubits") {
        Some(A::Is(v)) => v.iter().map(|&q| q as usize).collect(),
        other => panic!("{}: qubits attr {other:?}", n.name),
    }
}

fn exact_param(n: &DNode, name: &str) -> f64 {
    // the doc string carries the f64 value: "... name = <f64> ..."
    let doc = n.doc.as_ref().expect("parametrised node has a doc string");
    let key = format!("{name} = ");
    let start = doc.find(&key).expect("param in doc") + key.len();
    let rest = &doc[start..];
    let end = rest
        .find(|c: char| c == ' ' || c == ',' || c == ')')
        .unwrap_or(rest.len());
    rest[..end].parse().expect("f64")
}

/// Rebuilds the unitary/measurement program from the graph.
fn reconstruct(d: &DModel, num_qubits: usize) -> Circuit {
    let mut c = Circuit::new(num_qubits);
    for n in &d.nodes {
        let q = |i: usize| qubits_attr(n)[i];
        let th = || exact_param(n, "theta");
        let g = match n.op.trim_start_matches("IF_") {
            "I" => Some(Gate::I(q(0))),
            "H" => Some(Gate::H(q(0))),
            "X" => Some(Gate::X(q(0))),
            "Y" => Some(Gate::Y(q(0))),
            "Z" => Some(Gate::Z(q(0))),
            "S" => Some(Gate::S(q(0))),
            "S_DAG" => Some(Gate::Sdg(q(0))),
            "T" => Some(Gate::T(q(0))),
            "T_DAG" => Some(Gate::Tdg(q(0))),
            "SQRT_X" => Some(Gate::Sx(q(0))),
            "SQRT_X_DAG" => Some(Gate::Sxdg(q(0))),
            "RX" => Some(Gate::Rx(q(0), th())),
            "RY" => Some(Gate::Ry(q(0), th())),
            "RZ" => Some(Gate::Rz(q(0), th())),
            "P" => Some(Gate::Phase(q(0), th())),
            "U" => Some(Gate::U(
                q(0),
                th(),
                exact_param(n, "phi"),
                exact_param(n, "lambda"),
            )),
            "CX" => Some(Gate::Cnot(q(0), q(1))),
            "CZ" => Some(Gate::Cz(q(0), q(1))),
            "SWAP" => Some(Gate::Swap(q(0), q(1))),
            "ISWAP" => Some(Gate::ISwap(q(0), q(1))),
            "ISWAP_DAG" => Some(Gate::ISwapdg(q(0), q(1))),
            "CP" => Some(Gate::CPhase(q(0), q(1), th())),
            "CCX" => Some(Gate::Ccx(q(0), q(1), q(2))),
            _ => None,
        };
        let p = || match n.attrs.get("p") {
            Some(A::F(p)) => f64::from(*p),
            other => panic!("p attr {other:?}"),
        };
        let op = if let Some(g) = g {
            if n.op.starts_with("IF_") {
                let (Some(A::I(rec)), Some(A::I(val))) = (n.attrs.get("record"), n.attrs.get("value"))
                else {
                    panic!("IF_ node without record/value")
                };
                assert_eq!(n.inputs.last().unwrap(), &format!("m{rec}"));
                Op::ClassicControlled {
                    gate: g,
                    meas_index: *rec as usize,
                    target_value: *val == 1,
                }
            } else {
                Op::Gate(g)
            }
        } else {
            match n.op.as_str() {
                "Measure" => Op::Measure(q(0)),
                "Reset" => Op::Reset(q(0)),
                "X_ERROR" => Op::XFlip(q(0), p()),
                "Y_ERROR" => Op::YFlip(q(0), p()),
                "Z_ERROR" => Op::ZFlip(q(0), p()),
                "DEPOLARIZE1" => Op::Depolarize1q(q(0), p()),
                "DEPOLARIZE2" => Op::Depolarize2q(q(0), q(1), p()),
                "DETECTOR" | "OBSERVABLE_INCLUDE" => continue,
                other => panic!("unknown op {other}"),
            }
        };
        c.ops.push(op);
    }
    c
}

/// Noise probabilities are stored as f32 attributes (and in the doc string);
/// compare them at f32 precision, everything else exactly.
fn same_program(a: &Circuit, b: &Circuit) {
    assert_eq!(a.ops.len(), b.ops.len());
    for (x, y) in a.ops.iter().zip(&b.ops) {
        let f32eq = |p: f64, q: f64| (p as f32) == (q as f32);
        let ok = match (x, y) {
            (Op::XFlip(a, p), Op::XFlip(b, q))
            | (Op::YFlip(a, p), Op::YFlip(b, q))
            | (Op::ZFlip(a, p), Op::ZFlip(b, q))
            | (Op::Depolarize1q(a, p), Op::Depolarize1q(b, q)) => a == b && f32eq(*p, *q),
            (Op::Depolarize2q(a, b, p), Op::Depolarize2q(c, d, q)) => {
                a == c && b == d && f32eq(*p, *q)
            }
            _ => x == y,
        };
        assert!(ok, "{x:?} != {y:?}");
    }
}

fn export(c: &Circuit, opts: &OnnxOptions) -> DModel {
    let bytes = to_onnx(c, opts).expect("export");
    let d = decode(&bytes);
    check_wiring(&d);
    d
}

// ---------------------------------------------------------------- tests

#[test]
fn ghz_graph_structure() {
    let mut c = Circuit::new(3);
    c.h(0).cnot(0, 1).cnot(1, 2).measure_all();
    let d = export(&c, &OnnxOptions::default());
    assert_eq!(d.ir_version, 8);
    assert_eq!(d.producer, "qsim-lab");
    assert!(d.opsets.contains(&("qsim".to_string(), 1)));
    assert!(d.opsets.contains(&(String::new(), 18)));
    assert_eq!(d.graph_name, "circuit");
    let ops: Vec<&str> = d.nodes.iter().map(|n| n.op.as_str()).collect();
    assert_eq!(ops, ["H", "CX", "CX", "Measure", "Measure", "Measure"]);
    // inputs q0..q2 as complex64[2]
    assert_eq!(
        d.inputs,
        (0..3)
            .map(|q| (format!("q{q}"), 14, vec![2]))
            .collect::<Vec<_>>()
    );
    // CX(0,1) reads the H output of q0 and the input wire of q1
    assert_eq!(d.nodes[1].inputs, ["q0_1", "q1"]);
    assert_eq!(d.nodes[1].outputs, ["q0_2", "q1_1"]);
    // measurements output the qubit wire and the record bit
    assert_eq!(d.nodes[3].outputs, ["q0_3", "m0"]);
    let outs: Vec<&str> = d.outputs.iter().map(|(n, _, _)| n.as_str()).collect();
    assert_eq!(outs, ["q0_3", "q1_3", "q2_2", "m0", "m1", "m2"]);
    assert!(d.outputs[3..].iter().all(|(_, e, s)| *e == 9 && s.is_empty()));
    assert_eq!(d.metadata["qubits"], "3");
    assert_eq!(d.metadata["measurements"], "3");
    assert_eq!(d.metadata["gates_2q"], "2");
    same_program(&reconstruct(&d, 3), &c);
}

#[test]
fn every_gate_kind_roundtrips_exactly() {
    let mut c = Circuit::new(4);
    let angles = [PI / 4.0, -PI / 1024.0, 0.123456789012345, 1e-300, 7.0];
    for &a in &angles {
        c.rx(0, a).ry(1, a).rz(2, a).phase(3, a);
        c.u(1, a, -a, 2.0 * a);
        c.gate(Gate::CPhase(3, 0, a));
    }
    c.i(0).h(1).x(2).y(3).z(0).s(1).sdg(2).t(3).tdg(0).sx(1).sxdg(2);
    c.gate(Gate::Cz(0, 3))
        .gate(Gate::Swap(1, 2))
        .gate(Gate::ISwap(2, 3))
        .gate(Gate::ISwapdg(3, 2))
        .gate(Gate::Ccx(3, 1, 0));
    c.measure(2);
    c.ops.push(Op::Reset(2));
    c.c_if(0, Gate::Phase(1, -PI / 8.0));
    c.c_if(0, Gate::Cnot(3, 0));
    c.ops.push(Op::ClassicControlled {
        gate: Gate::X(1),
        meas_index: 0,
        target_value: false,
    });
    c.ops.push(Op::XFlip(0, 0.125));
    c.ops.push(Op::YFlip(1, 1e-3));
    c.ops.push(Op::ZFlip(2, 0.3));
    c.ops.push(Op::Depolarize1q(3, 0.01));
    c.ops.push(Op::Depolarize2q(0, 1, 0.02));
    let d = export(&c, &OnnxOptions::default());
    assert_eq!(d.nodes.len(), c.ops.len());
    same_program(&reconstruct(&d, 4), &c);
    // f32 attribute present for display; pi fraction in the doc string
    let rz = d.nodes.iter().find(|n| n.op == "RZ").unwrap();
    assert!(matches!(rz.attrs.get("theta"), Some(A::F(x)) if *x == (PI / 4.0) as f32));
    assert!(rz.doc.as_deref().unwrap().contains("π/4"));
    // the conditional nodes draw an edge from the measurement
    let cif: Vec<&DNode> = d.nodes.iter().filter(|n| n.op.starts_with("IF_")).collect();
    assert_eq!(cif.len(), 3);
    assert!(cif.iter().all(|n| n.inputs.last().unwrap() == "m0"));
    assert_eq!(cif[1].op, "IF_CX");
    assert!(cif[1].inputs[0].starts_with("q3_") && cif[1].inputs[1].starts_with("q0_"));
}

#[test]
fn semiclassical_shor_classical_edges() {
    let c = qsim_lab::shor::semiclassical_circuit(15, 2);
    let d = export(&c, &OnnxOptions::default());
    same_program(&reconstruct(&d, c.num_qubits), &c);
    let producers: HashMap<&str, usize> = d
        .nodes
        .iter()
        .enumerate()
        .flat_map(|(i, n)| n.outputs.iter().map(move |o| (o.as_str(), i)))
        .collect();
    let mut n_if = 0;
    for (i, n) in d.nodes.iter().enumerate() {
        if n.op.starts_with("IF_") {
            n_if += 1;
            let m = n.inputs.last().unwrap();
            let p = producers[m.as_str()];
            assert!(p < i && d.nodes[p].op == "Measure");
        }
    }
    assert!(n_if > 0);
}

#[test]
fn stim_detectors_and_observables() {
    let src = "R 0 1 2 3 4\nH 0\nCX 0 1 1 2\nX_ERROR(0.01) 0 1 2\nM 0 1 2\nDETECTOR rec[-1] rec[-2]\nDETECTOR rec[-2] rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-1]\n";
    let prog = qsim_lab::io::stim::parse_stim(src).unwrap();
    let opts = OnnxOptions {
        detectors: prog.detectors.clone(),
        observables: prog.observables.clone(),
        ..Default::default()
    };
    let d = export(&prog.circuit, &opts);
    let dets: Vec<&DNode> = d.nodes.iter().filter(|n| n.op == "DETECTOR").collect();
    assert_eq!(dets.len(), 2);
    for (det, recs) in dets.iter().zip(&prog.detectors) {
        let want: Vec<String> = recs.iter().map(|j| format!("m{j}")).collect();
        assert_eq!(det.inputs, want);
    }
    let obs: Vec<&DNode> = d
        .nodes
        .iter()
        .filter(|n| n.op == "OBSERVABLE_INCLUDE")
        .collect();
    assert_eq!(obs.len(), 1);
    assert_eq!(obs[0].outputs, ["L0"]);
    let outs: HashSet<&str> = d.outputs.iter().map(|(n, _, _)| n.as_str()).collect();
    assert!(outs.contains("D0") && outs.contains("D1") && outs.contains("L0"));
    assert_eq!(d.metadata["detectors"], "2");
    same_program(&reconstruct(&d, prog.circuit.num_qubits), &prog.circuit);
}

#[test]
fn truncation_and_untouched_qubits() {
    let mut c = Circuit::new(5); // qubit 4 never used
    c.h(0).cnot(0, 1).measure(0).measure(1);
    let opts = OnnxOptions {
        max_ops: Some(3),
        detectors: vec![vec![0], vec![0, 1]],
        ..Default::default()
    };
    let d = export(&c, &opts);
    assert_eq!(d.nodes.iter().filter(|n| n.op != "DETECTOR").count(), 3);
    // the detector reading m1 (past the cut) is dropped, the other kept
    assert_eq!(d.nodes.iter().filter(|n| n.op == "DETECTOR").count(), 1);
    assert_eq!(d.metadata["truncated"], "first 3 of 4 ops");
    assert_eq!(d.metadata["dropped_detectors_and_observables"], "1");
    // untouched qubit 4 passes straight through
    assert!(d.outputs.iter().any(|(n, _, _)| n == "q4"));
}

#[test]
fn invalid_programs_are_rejected() {
    let mut c = Circuit::new(2);
    c.gate(Gate::Cnot(0, 2));
    assert!(matches!(
        to_onnx(&c, &OnnxOptions::default()),
        Err(SimError::QubitOutOfRange { qubit: 2, .. })
    ));
    let mut c = Circuit::new(2);
    c.gate(Gate::Cz(1, 1));
    assert!(matches!(
        to_onnx(&c, &OnnxOptions::default()),
        Err(SimError::RepeatedQubit(_))
    ));
    let mut c = Circuit::new(2);
    c.c_if(0, Gate::X(1)); // reads a measurement that has not happened
    assert!(matches!(
        to_onnx(&c, &OnnxOptions::default()),
        Err(SimError::ClassicalBitOutOfRange { bit: 0, available: 0 })
    ));
    let mut c = Circuit::new(1);
    c.measure(0);
    let opts = OnnxOptions {
        detectors: vec![vec![1]],
        ..Default::default()
    };
    assert!(matches!(
        to_onnx(&c, &opts),
        Err(SimError::ClassicalBitOutOfRange { bit: 1, available: 1 })
    ));
}

#[test]
fn large_circuit_is_well_formed() {
    // a gate-level Shor circuit: thousands of nodes, still SSA and linear
    let c = qsim_lab::shor::semiclassical_ripple_circuit(21, 2);
    assert!(c.ops.len() > 1000);
    let d = export(&c, &OnnxOptions::default());
    assert_eq!(d.nodes.len(), c.ops.len());
    same_program(&reconstruct(&d, c.num_qubits), &c);
}
