//! OpenQASM 2.0 parser and serializer for quantum circuits.

use crate::circuit::{Circuit, Op, SimError};
use crate::gate::Gate;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Serializes a [`Circuit`] into an OpenQASM 2.0 program string.
///
/// Angles are written in Rust's shortest round-trip form, so
/// `from_qasm(to_qasm(c))` reproduces every parameter exactly. The classical
/// register has one bit per measurement (measurement `k` writes `c[k]`).
///
/// Classically conditioned gates and stochastic noise channels have no
/// faithful OpenQASM 2.0 form (`if` compares a whole register), so they are
/// reported as an error rather than silently dropped.
pub fn to_qasm(circuit: &Circuit) -> Result<String, SimError> {
    let mut out = String::new();
    out.push_str("OPENQASM 2.0;\n");
    out.push_str("include \"qelib1.inc\";\n");
    let n = circuit.num_qubits;
    out.push_str(&format!("qreg q[{n}];\n"));
    let num_meas = circuit
        .ops
        .iter()
        .filter(|op| matches!(op, Op::Measure(_)))
        .count();
    if num_meas > 0 {
        out.push_str(&format!("creg c[{num_meas}];\n"));
    }

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
                Gate::Rx(q, th) => out.push_str(&format!("rx({th:?}) q[{q}];\n")),
                Gate::Ry(q, th) => out.push_str(&format!("ry({th:?}) q[{q}];\n")),
                Gate::Rz(q, th) => out.push_str(&format!("rz({th:?}) q[{q}];\n")),
                Gate::Phase(q, th) => out.push_str(&format!("u1({th:?}) q[{q}];\n")),
                Gate::U(q, th, ph, lam) => {
                    out.push_str(&format!("u3({th:?},{ph:?},{lam:?}) q[{q}];\n"))
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
                Gate::CPhase(a, b, th) => out.push_str(&format!("cu1({th:?}) q[{a}],q[{b}];\n")),
                Gate::Ccx(a, b, t) => out.push_str(&format!("ccx q[{a}],q[{b}],q[{t}];\n")),
            },
            Op::Measure(q) => {
                out.push_str(&format!("measure q[{q}] -> c[{meas_idx}];\n"));
                meas_idx += 1;
            }
            Op::Reset(q) => {
                out.push_str(&format!("reset q[{q}];\n"));
            }
            other => {
                return Err(SimError::QasmError(format!(
                    "{other:?} has no OpenQASM 2.0 equivalent"
                )))
            }
        }
    }
    Ok(out)
}

/// Parses an OpenQASM 2.0 parameter expression.
///
/// Grammar (the OpenQASM 2.0 expression language, standard precedence,
/// left-associative except `^`):
///
/// ```text
/// expr   := term (('+' | '-') term)*
/// term   := unary (('*' | '/') unary)*
/// unary  := ('-' | '+') unary | power
/// power  := atom ('^' unary)?
/// atom   := number | 'pi' | '(' expr ')' | func '(' expr ')'
/// func   := sin | cos | tan | exp | ln | sqrt
/// ```
fn parse_param(expr: &str) -> Result<f64, String> {
    let mut p = ExprParser {
        s: expr.as_bytes(),
        i: 0,
    };
    let v = p.expr()?;
    p.skip_ws();
    if p.i != p.s.len() {
        return Err(format!(
            "unexpected '{}' in expression '{}'",
            &expr[p.i..],
            expr.trim()
        ));
    }
    if !v.is_finite() {
        return Err(format!("expression '{}' is not finite", expr.trim()));
    }
    Ok(v)
}

struct ExprParser<'a> {
    s: &'a [u8],
    i: usize,
}

impl ExprParser<'_> {
    fn skip_ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.s.get(self.i).copied()
    }

    fn eat(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn expr(&mut self) -> Result<f64, String> {
        let mut v = self.term()?;
        loop {
            if self.eat(b'+') {
                v += self.term()?;
            } else if self.eat(b'-') {
                v -= self.term()?;
            } else {
                return Ok(v);
            }
        }
    }

    fn term(&mut self) -> Result<f64, String> {
        let mut v = self.unary()?;
        loop {
            if self.eat(b'*') {
                v *= self.unary()?;
            } else if self.eat(b'/') {
                let d = self.unary()?;
                if d == 0.0 {
                    return Err("division by zero".to_string());
                }
                v /= d;
            } else {
                return Ok(v);
            }
        }
    }

    fn unary(&mut self) -> Result<f64, String> {
        if self.eat(b'-') {
            Ok(-self.unary()?)
        } else if self.eat(b'+') {
            self.unary()
        } else {
            self.power()
        }
    }

    fn power(&mut self) -> Result<f64, String> {
        let base = self.atom()?;
        if self.eat(b'^') {
            // Right-associative: a^b^c = a^(b^c); binds tighter than unary minus
            // on its left operand, as in the OpenQASM 2.0 grammar.
            let e = self.unary()?;
            Ok(base.powf(e))
        } else {
            Ok(base)
        }
    }

    fn atom(&mut self) -> Result<f64, String> {
        match self.peek() {
            None => Err("unexpected end of expression".to_string()),
            Some(b'(') => {
                self.i += 1;
                let v = self.expr()?;
                if !self.eat(b')') {
                    return Err("missing ')'".to_string());
                }
                Ok(v)
            }
            Some(c) if c.is_ascii_digit() || c == b'.' => self.number(),
            Some(c) if c.is_ascii_alphabetic() => {
                let start = self.i;
                while self.i < self.s.len()
                    && (self.s[self.i].is_ascii_alphanumeric() || self.s[self.i] == b'_')
                {
                    self.i += 1;
                }
                let name = std::str::from_utf8(&self.s[start..self.i])
                    .unwrap_or("")
                    .to_ascii_lowercase();
                if name == "pi" {
                    return Ok(PI);
                }
                let f: fn(f64) -> f64 = match name.as_str() {
                    "sin" => f64::sin,
                    "cos" => f64::cos,
                    "tan" => f64::tan,
                    "exp" => f64::exp,
                    "ln" => f64::ln,
                    "sqrt" => f64::sqrt,
                    _ => return Err(format!("unknown identifier '{name}'")),
                };
                if !self.eat(b'(') {
                    return Err(format!("expected '(' after '{name}'"));
                }
                let v = self.expr()?;
                if !self.eat(b')') {
                    return Err(format!("missing ')' after argument of '{name}'"));
                }
                Ok(f(v))
            }
            Some(c) => Err(format!("unexpected character '{}'", c as char)),
        }
    }

    /// A decimal literal, optionally with an exponent (`1.5e-3`).
    fn number(&mut self) -> Result<f64, String> {
        let start = self.i;
        while self.i < self.s.len() && (self.s[self.i].is_ascii_digit() || self.s[self.i] == b'.') {
            self.i += 1;
        }
        if self.i < self.s.len() && (self.s[self.i] == b'e' || self.s[self.i] == b'E') {
            let save = self.i;
            self.i += 1;
            if self.i < self.s.len() && (self.s[self.i] == b'+' || self.s[self.i] == b'-') {
                self.i += 1;
            }
            let digits = self.i;
            while self.i < self.s.len() && self.s[self.i].is_ascii_digit() {
                self.i += 1;
            }
            if self.i == digits {
                self.i = save; // not an exponent after all
            }
        }
        let text = std::str::from_utf8(&self.s[start..self.i]).unwrap_or("");
        text.parse::<f64>()
            .map_err(|e| format!("cannot parse number '{text}': {e}"))
    }
}

/// Splits `s` at commas that are not inside parentheses.
fn split_top_level(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0;
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&s[start..]);
    out
}

/// Index of the `)` matching the first `(` in `s`, if any.
fn matching_paren(s: &str) -> Option<usize> {
    let open = s.find('(')?;
    let mut depth = 0i32;
    for (i, c) in s[open..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Parses comma-separated parameter list inside parentheses: `(p1, p2, ...)`.
fn parse_param_list(s: &str) -> Result<Vec<f64>, String> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(Vec::new());
    }
    split_top_level(s).into_iter().map(parse_param).collect()
}

/// `(qubits, parameters)` for every gate name the parser understands.
fn gate_arity(name: &str) -> Option<(usize, usize)> {
    Some(match name {
        "id" | "h" | "x" | "y" | "z" | "s" | "sdg" | "t" | "tdg" | "sx" | "sxdg" => (1, 0),
        "rx" | "ry" | "rz" | "u1" | "p" | "phase" => (1, 1),
        "u3" | "u" => (1, 3),
        "cx" | "cnot" | "cz" | "swap" | "iswap" => (2, 0),
        "cp" | "cu1" | "cphase" => (2, 1),
        "ccx" => (3, 0),
        _ => return None,
    })
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
    // register name -> (offset, size)
    let mut qreg_offsets: HashMap<String, (usize, usize)> = HashMap::new();
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
                    qreg_offsets.insert(name.trim().to_string(), (total_qubits, size));
                    total_qubits += size;
                }
            }
        }
    }

    if total_qubits == 0 {
        // If no qreg was declared, scan for referenced qubit indices like q[0]
        let mut max_q = 0usize;
        let mut found = false;
        for part in full_code.split(['[', ']']) {
            if let Ok(idx) = part.trim().parse::<usize>() {
                max_q = max_q.max(idx + 1);
                found = true;
            }
        }
        if found {
            total_qubits = max_q;
            qreg_offsets.insert("q".to_string(), (0, total_qubits));
        }
    }

    let mut circuit = Circuit::new(total_qubits);

    let resolve_qubit = |arg: &str| -> Result<usize, SimError> {
        let s = arg.trim();
        if let Some((reg, idx_str)) = s.split_once('[') {
            if let Some(idx_str) = idx_str.strip_suffix(']') {
                let idx: usize = idx_str.trim().parse().map_err(|e| {
                    SimError::QasmError(format!("bad qubit index '{idx_str}': {e}"))
                })?;
                let (offset, size) = qreg_offsets.get(reg.trim()).copied().ok_or_else(|| {
                    SimError::QasmError(format!("unknown register '{}'", reg.trim()))
                })?;
                if idx >= size {
                    return Err(SimError::QasmError(format!(
                        "index {idx} out of range for register '{}' of size {size}",
                        reg.trim()
                    )));
                }
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
        Err(SimError::QasmError(format!(
            "cannot resolve qubit argument '{s}'"
        )))
    };

    // Second pass: parse gates and measurements
    for stmt in full_code.split(';') {
        let trimmed = stmt.trim();
        if trimmed.is_empty() {
            continue;
        }
        let (gate_spec, rest) = if trimmed.contains('(') {
            let close_idx = matching_paren(trimmed)
                .ok_or_else(|| SimError::QasmError(format!("unmatched '(' in '{trimmed}'")))?;
            let gate_spec = trimmed[..=close_idx].trim();
            let rest = trimmed[close_idx + 1..].trim();
            if rest.contains(['(', ')']) {
                return Err(SimError::QasmError(format!(
                    "unbalanced parentheses in '{trimmed}'"
                )));
            }
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

        let lname = gate_name.to_lowercase();
        if let Some((nq, np)) = gate_arity(&lname) {
            if args.len() != nq || params.len() != np {
                return Err(SimError::QasmError(format!(
                    "'{gate_name}' takes {nq} qubit(s) and {np} parameter(s), got {} and {}",
                    args.len(),
                    params.len()
                )));
            }
        }
        match lname.as_str() {
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
            "cp" | "cu1" | "cphase" => {
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
        c.h(0).cnot(0, 1).rz(1, 0.5).swap(1, 2).measure_all();
        let qasm = to_qasm(&c).unwrap();
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
