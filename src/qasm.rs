//! OpenQASM 2.0 parser and serializer for quantum circuits.

use crate::circuit::{Circuit, Op, SimError};
use crate::gate::Gate;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Serializes a [`Circuit`] into an OpenQASM 2.0 program string.
pub fn to_qasm(circuit: &Circuit) -> String {
    let mut out = String::new();
    out.push_str("OPENQASM 2.0;\n");
    out.push_str("include \"qelib1.inc\";\n");
    let n = circuit.num_qubits;
    out.push_str(&format!("qreg q[{n}];\n"));
    out.push_str(&format!("creg c[{n}];\n"));

    let mut meas_idx = 0;
    for op in &circuit.ops {
        match op {
            Op::Gate(g) => match *g {
                Gate::I(q) => out.push_str(&format!("id q[{q}];\n")),
                Gate::H(q) => out.push_str(&format!("h q[{q}];\n")),
                Gate::X(q) => out.push_str(&format!("x q[{q}];\n")),
                Gate::Y(q) => out.push_str(&format!("y q[{q}];\n")),
                Gate::Z(q) => out.push_str(&format!("z q[{q}];\n")),
                Gate::S(q) => out.push_str(&format!("s q[{q}];\n")),
                Gate::Sdg(q) => out.push_str(&format!("sdg q[{q}];\n")),
                Gate::T(q) => out.push_str(&format!("t q[{q}];\n")),
                Gate::Tdg(q) => out.push_str(&format!("tdg q[{q}];\n")),
                Gate::Sx(q) => out.push_str(&format!("sx q[{q}];\n")),
                Gate::Sxdg(q) => out.push_str(&format!("sxdg q[{q}];\n")),
                Gate::Rx(q, th) => out.push_str(&format!("rx({th:.16}) q[{q}];\n")),
                Gate::Ry(q, th) => out.push_str(&format!("ry({th:.16}) q[{q}];\n")),
                Gate::Rz(q, th) => out.push_str(&format!("rz({th:.16}) q[{q}];\n")),
                Gate::Phase(q, th) => out.push_str(&format!("u1({th:.16}) q[{q}];\n")),
                Gate::U(q, th, ph, lam) => {
                    out.push_str(&format!("u3({th:.16},{ph:.16},{lam:.16}) q[{q}];\n"))
                }
                Gate::Cnot(c, t) => out.push_str(&format!("cx q[{c}],q[{t}];\n")),
                Gate::Cz(a, b) => out.push_str(&format!("cz q[{a}],q[{b}];\n")),
                Gate::Swap(a, b) => out.push_str(&format!("swap q[{a}],q[{b}];\n")),
                Gate::ISwap(a, b) => {
                    // Decomposition into standard gates for compatibility
                    out.push_str(&format!("swap q[{a}],q[{b}];\n"));
                    out.push_str(&format!("cz q[{a}],q[{b}];\n"));
                    out.push_str(&format!("s q[{a}];\n"));
                    out.push_str(&format!("s q[{b}];\n"));
                }
                Gate::ISwapdg(a, b) => {
                    out.push_str(&format!("swap q[{a}],q[{b}];\n"));
                    out.push_str(&format!("cz q[{a}],q[{b}];\n"));
                    out.push_str(&format!("sdg q[{a}];\n"));
                    out.push_str(&format!("sdg q[{b}];\n"));
                }
                Gate::CPhase(a, b, th) => out.push_str(&format!("cp({th:.16}) q[{a}],q[{b}];\n")),
                Gate::Ccx(a, b, t) => out.push_str(&format!("ccx q[{a}],q[{b}],q[{t}];\n")),
            },
            Op::Measure(q) => {
                out.push_str(&format!("measure q[{q}] -> c[{meas_idx}];\n"));
                meas_idx += 1;
            }
            Op::Reset(q) => {
                out.push_str(&format!("reset q[{q}];\n"));
            }
            _ => {}
        }
    }
    out
}

/// Parses a mathematical expression in QASM parameters (numbers, pi, -, +, *, /).
fn parse_param(expr: &str) -> Result<f64, String> {
    let s = expr.trim();
    if s.is_empty() {
        return Err("empty expression".to_string());
    }
    // Handle leading signs
    if let Some(rest) = s.strip_prefix('-') {
        return parse_param(rest).map(|v| -v);
    }
    if let Some(rest) = s.strip_prefix('+') {
        return parse_param(rest);
    }
    // Handle division (e.g. pi/2, -pi/4)
    if let Some((num, den)) = s.split_once('/') {
        let n = parse_param(num)?;
        let d = parse_param(den)?;
        if d == 0.0 {
            return Err("division by zero".to_string());
        }
        return Ok(n / d);
    }
    // Handle multiplication
    if let Some((a, b)) = s.split_once('*') {
        let n = parse_param(a)?;
        let d = parse_param(b)?;
        return Ok(n * d);
    }
    // Constants or literals
    if s.eq_ignore_ascii_case("pi") {
        return Ok(PI);
    }
    s.parse::<f64>()
        .map_err(|e| format!("cannot parse number '{s}': {e}"))
}

/// Parses comma-separated parameter list inside parentheses: `(p1, p2, ...)`.
fn parse_param_list(s: &str) -> Result<Vec<f64>, String> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(Vec::new());
    }
    let mut params = Vec::new();
    for p in s.split(',') {
        params.push(parse_param(p)?);
    }
    Ok(params)
}

/// Parses an OpenQASM 2.0 program string into a [`Circuit`].
pub fn from_qasm(source: &str) -> Result<Circuit, SimError> {
    // 1. Strip comments
    let mut clean_lines = Vec::new();
    for line in source.lines() {
        let code = match line.split_once("//") {
            Some((code, _)) => code,
            None => line,
        };
        clean_lines.push(code);
    }
    let full_code = clean_lines.join(" ");

    // 2. Split statements by semicolon
    let mut qreg_offsets: HashMap<String, usize> = HashMap::new();
    let mut total_qubits = 0usize;

    // First pass: find all qregs and their offsets
    for stmt in full_code.split(';') {
        let trimmed = stmt.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut words = trimmed.split_whitespace();
        let cmd = words.next().unwrap_or("");
        if cmd == "qreg" {
            let rest = words.collect::<Vec<_>>().join("");
            if let Some((name, size_str)) = rest.split_once('[') {
                if let Some(size_str) = size_str.strip_suffix(']') {
                    let size: usize = size_str.trim().parse().map_err(|e| {
                        SimError::QasmError(format!("invalid qreg size '{size_str}': {e}"))
                    })?;
                    qreg_offsets.insert(name.trim().to_string(), total_qubits);
                    total_qubits += size;
                }
            }
        }
    }

    if total_qubits == 0 {
        // If no qreg was declared, scan for referenced qubit indices like q[0]
        let mut max_q = 0usize;
        let mut found = false;
        for part in full_code.split(|c: char| c == '[' || c == ']') {
            if let Ok(idx) = part.trim().parse::<usize>() {
                max_q = max_q.max(idx + 1);
                found = true;
            }
        }
        if found {
            total_qubits = max_q;
            qreg_offsets.insert("q".to_string(), 0);
        }
    }

    let mut circuit = Circuit::new(total_qubits);

    let resolve_qubit = |arg: &str| -> Result<usize, SimError> {
        let s = arg.trim();
        if let Some((reg, idx_str)) = s.split_once('[') {
            if let Some(idx_str) = idx_str.strip_suffix(']') {
                let idx: usize = idx_str
                    .trim()
                    .parse()
                    .map_err(|e| SimError::QasmError(format!("bad qubit index '{idx_str}': {e}")))?;
                let offset = qreg_offsets.get(reg.trim()).copied().unwrap_or(0);
                let q = offset + idx;
                if q >= total_qubits {
                    return Err(SimError::QubitOutOfRange {
                        qubit: q,
                        num_qubits: total_qubits,
                    });
                }
                return Ok(q);
            }
        }
        // Direct integer
        if let Ok(q) = s.parse::<usize>() {
            if q >= total_qubits {
                return Err(SimError::QubitOutOfRange {
                    qubit: q,
                    num_qubits: total_qubits,
                });
            }
            return Ok(q);
        }
        Err(SimError::QasmError(format!("cannot resolve qubit argument '{s}'")))
    };

    // Second pass: parse gates and measurements
    for stmt in full_code.split(';') {
        let trimmed = stmt.trim();
        if trimmed.is_empty() {
            continue;
        }
        let (gate_spec, rest) = if trimmed.contains('(') {
            let close_idx = trimmed.find(')').ok_or_else(|| {
                SimError::QasmError(format!("unmatched '(' in '{trimmed}'"))
            })?;
            let gate_spec = trimmed[..=close_idx].trim();
            let rest = trimmed[close_idx + 1..].trim();
            (gate_spec, rest)
        } else {
            match trimmed.split_once(char::is_whitespace) {
                Some((w, r)) => (w.trim(), r.trim()),
                None => (trimmed, ""),
            }
        };

        if gate_spec.starts_with("OPENQASM")
            || gate_spec == "include"
            || gate_spec == "qreg"
            || gate_spec == "creg"
            || gate_spec == "barrier"
        {
            continue;
        }

        if gate_spec == "measure" {
            // measure q[0] -> c[0]
            if let Some((q_part, _)) = rest.split_once("->") {
                let q = resolve_qubit(q_part)?;
                circuit.measure(q);
            }
            continue;
        }

        if gate_spec == "reset" {
            let q = resolve_qubit(rest)?;
            circuit.reset(q);
            continue;
        }

        // Gate with optional params, e.g. "rx(0.5) q[0]" or "h q[0]"
        let (gate_name, params) = if let Some((gname, rest_params)) = gate_spec.split_once('(') {
            let pstr = rest_params.strip_suffix(')').ok_or_else(|| {
                SimError::QasmError(format!("unmatched parenthesis in '{gate_spec}'"))
            })?;
            let p = parse_param_list(pstr).map_err(SimError::QasmError)?;
            (gname.trim(), p)
        } else {
            (gate_spec, Vec::new())
        };

        let args: Vec<usize> = rest
            .split(',')
            .map(|a| a.trim())
            .filter(|a| !a.is_empty())
            .map(&resolve_qubit)
            .collect::<Result<Vec<_>, _>>()?;

        match gate_name.to_lowercase().as_str() {
            "id" => {
                if args.len() == 1 {
                    circuit.i(args[0]);
                }
            }
            "h" => {
                if args.len() == 1 {
                    circuit.h(args[0]);
                }
            }
            "x" => {
                if args.len() == 1 {
                    circuit.x(args[0]);
                }
            }
            "y" => {
                if args.len() == 1 {
                    circuit.y(args[0]);
                }
            }
            "z" => {
                if args.len() == 1 {
                    circuit.z(args[0]);
                }
            }
            "s" => {
                if args.len() == 1 {
                    circuit.s(args[0]);
                }
            }
            "sdg" => {
                if args.len() == 1 {
                    circuit.sdg(args[0]);
                }
            }
            "t" => {
                if args.len() == 1 {
                    circuit.t(args[0]);
                }
            }
            "tdg" => {
                if args.len() == 1 {
                    circuit.tdg(args[0]);
                }
            }
            "sx" => {
                if args.len() == 1 {
                    circuit.sx(args[0]);
                }
            }
            "sxdg" => {
                if args.len() == 1 {
                    circuit.sxdg(args[0]);
                }
            }
            "rx" => {
                if args.len() == 1 && !params.is_empty() {
                    circuit.rx(args[0], params[0]);
                }
            }
            "ry" => {
                if args.len() == 1 && !params.is_empty() {
                    circuit.ry(args[0], params[0]);
                }
            }
            "rz" => {
                if args.len() == 1 && !params.is_empty() {
                    circuit.rz(args[0], params[0]);
                }
            }
            "u1" | "p" | "phase" => {
                if args.len() == 1 && !params.is_empty() {
                    circuit.phase(args[0], params[0]);
                }
            }
            "u3" | "u" => {
                if args.len() == 1 && params.len() >= 3 {
                    circuit.u(args[0], params[0], params[1], params[2]);
                }
            }
            "cx" | "cnot" => {
                if args.len() == 2 {
                    circuit.cnot(args[0], args[1]);
                }
            }
            "cz" => {
                if args.len() == 2 {
                    circuit.cz(args[0], args[1]);
                }
            }
            "swap" => {
                if args.len() == 2 {
                    circuit.swap(args[0], args[1]);
                }
            }
            "iswap" => {
                if args.len() == 2 {
                    circuit.iswap(args[0], args[1]);
                }
            }
            "iswapdg" | "iswap_adj" => {
                if args.len() == 2 {
                    circuit.iswapdg(args[0], args[1]);
                }
            }
            "cp" | "cphase" => {
                if args.len() == 2 && !params.is_empty() {
                    circuit.cphase(args[0], args[1], params[0]);
                }
            }
            "ccx" => {
                if args.len() == 3 {
                    circuit.ccx(args[0], args[1], args[2]);
                }
            }
            unknown => {
                return Err(SimError::QasmError(format!(
                    "unrecognized QASM gate: '{unknown}'"
                )));
            }
        }
    }

    Ok(circuit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qasm_roundtrip() {
        let mut c = Circuit::new(3);
        c.h(0)
            .cnot(0, 1)
            .rz(1, 0.5)
            .swap(1, 2)
            .measure_all();
        let qasm = to_qasm(&c);
        let parsed = from_qasm(&qasm).unwrap();
        assert_eq!(parsed.num_qubits, 3);
        assert_eq!(parsed.ops.len(), c.ops.len());
        assert_eq!(parsed.num_gates(), c.num_gates());
    }

    #[test]
    fn parse_openqasm_features() {
        let qasm = r#"
            // Example Bell circuit
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            h q[0];
            cx q[0], q[1];
            rz(pi / 2) q[1];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
        "#;
        let c = from_qasm(qasm).unwrap();
        assert_eq!(c.num_qubits, 2);
        assert_eq!(c.num_gates(), 3);
    }
}
