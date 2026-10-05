//! Reader for the `.qc` circuit format of Amy's Feynman toolkit, the format
//! of the standard Clifford+T T-count benchmarks (Amy, Maslov & Mosca 2014;
//! used by Feynman, PyZX, TODD and AlphaTensor-Quantum). See
//! `research/compiler/todd.md`.
//!
//! A file declares its qubits (`.v a b c`), the primary inputs (`.i a b`;
//! the other qubits are ancillas that start in `|0>`), optional outputs
//! (`.o`), and a gate list between `BEGIN` and `END`. Gate lines are a name
//! followed by qubit names:
//!
//! | line | gate |
//! |---|---|
//! | `H a`, `X a`, `Y a`, `Z a` | single-qubit Clifford |
//! | `S a` / `P a`, `S* a` / `P* a` | `S`, `S†` |
//! | `T a`, `T* a` | `T`, `T†` |
//! | `tof a` / `tof a b` / `tof a b c` | `X`, `CNOT(a, b)`, Toffoli (last name is the target) |
//! | `cnot a b` | `CNOT(a, b)` |
//! | `Z a b` / `Zd a b` | controlled-Z |
//! | `Z a b c` / `Zd a b c` | doubly controlled Z (`CCZ`) |
//! | `swap a b` | SWAP |
//!
//! `Zd` is the inverse of `Z`; the controlled versions are Hermitian, so
//! they are the same gate. Multi-controlled gates with more than two
//! controls are rejected.

use std::collections::HashMap;
use std::fmt;

/// A gate of a `.qc` circuit, on qubit indices (positions in the `.v` line).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QcGate {
    /// Hadamard.
    H(usize),
    /// Pauli X.
    X(usize),
    /// Pauli Y.
    Y(usize),
    /// Pauli Z.
    Z(usize),
    /// `S = diag(1, i)`.
    S(usize),
    /// `S†`.
    Sdg(usize),
    /// `T = diag(1, e^{iπ/4})`.
    T(usize),
    /// `T†`.
    Tdg(usize),
    /// `Cnot(control, target)`.
    Cnot(usize, usize),
    /// Controlled Z (symmetric).
    Cz(usize, usize),
    /// Doubly controlled Z (symmetric), 7 T gates in the standard decomposition.
    Ccz(usize, usize, usize),
    /// Toffoli `(control, control, target)`, 7 T gates in the standard decomposition.
    Toffoli(usize, usize, usize),
    /// SWAP.
    Swap(usize, usize),
}

/// A parsed `.qc` circuit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QcCircuit {
    /// Qubit names, in `.v` order (qubit `i` is `names[i]`).
    pub names: Vec<String>,
    /// `inputs[i]` is true when qubit `i` is a primary input (`.i`); the
    /// others are ancillas initialised to `|0>`.
    pub inputs: Vec<bool>,
    /// The gates, in time order.
    pub gates: Vec<QcGate>,
}

/// A `.qc` parse error: the 1-based line number and what was wrong.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QcError {
    /// 1-based line number (0 when the error is not tied to a line).
    pub line: usize,
    /// Description of the problem.
    pub msg: String,
}

impl fmt::Display for QcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, ".qc format, line {}: {}", self.line, self.msg)
    }
}

impl std::error::Error for QcError {}

impl From<QcError> for crate::error::Error {
    fn from(e: QcError) -> Self {
        crate::error::Error::Parse(e.to_string())
    }
}

impl QcCircuit {
    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.names.len()
    }

    /// T-count with every `CCZ` and Toffoli counted as 7 (their standard
    /// Clifford+T decomposition), plus the explicit `T`/`T†` gates.
    pub fn t_count(&self) -> usize {
        self.gates
            .iter()
            .map(|g| match g {
                QcGate::T(_) | QcGate::Tdg(_) => 1,
                QcGate::Ccz(..) | QcGate::Toffoli(..) => 7,
                _ => 0,
            })
            .sum()
    }

    /// The circuit as a [`Circuit`](crate::Circuit) (`CCZ` as `H·Toffoli·H`
    /// on its last qubit).
    pub fn to_circuit(&self) -> crate::Circuit {
        use crate::gate::Gate;
        let mut c = crate::Circuit::new(self.num_qubits());
        for g in &self.gates {
            match *g {
                QcGate::H(q) => c.gate(Gate::H(q)),
                QcGate::X(q) => c.gate(Gate::X(q)),
                QcGate::Y(q) => c.gate(Gate::Y(q)),
                QcGate::Z(q) => c.gate(Gate::Z(q)),
                QcGate::S(q) => c.gate(Gate::S(q)),
                QcGate::Sdg(q) => c.gate(Gate::Sdg(q)),
                QcGate::T(q) => c.gate(Gate::T(q)),
                QcGate::Tdg(q) => c.gate(Gate::Tdg(q)),
                QcGate::Cnot(a, b) => c.gate(Gate::Cnot(a, b)),
                QcGate::Cz(a, b) => c.gate(Gate::Cz(a, b)),
                QcGate::Swap(a, b) => c.gate(Gate::Swap(a, b)),
                QcGate::Toffoli(a, b, t) => c.gate(Gate::Ccx(a, b, t)),
                QcGate::Ccz(a, b, t) => c
                    .gate(Gate::H(t))
                    .gate(Gate::Ccx(a, b, t))
                    .gate(Gate::H(t)),
            };
        }
        c
    }
}

/// Parses a `.qc` file.
pub fn parse_qc(src: &str) -> Result<QcCircuit, QcError> {
    let mut names: Vec<String> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut input_names: Option<Vec<String>> = None;
    let mut gates = Vec::new();
    let mut in_body = false;
    let mut seen_end = false;
    for (ln, raw) in src.lines().enumerate() {
        let ln = ln + 1;
        let err = |msg: String| QcError { line: ln, msg };
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let mut toks = line.split_whitespace();
        let head = toks.next().unwrap_or("");
        let args: Vec<&str> = toks.collect();
        if !in_body {
            match head {
                ".v" => {
                    for a in &args {
                        if index.insert(a.to_string(), names.len()).is_some() {
                            return Err(err(format!("qubit `{a}` declared twice")));
                        }
                        names.push(a.to_string());
                    }
                }
                ".i" => input_names = Some(args.iter().map(|s| s.to_string()).collect()),
                ".o" | ".c" | ".ov" => {}
                "BEGIN" => {
                    if seen_end {
                        return Err(err("only one BEGIN/END block is supported".into()));
                    }
                    in_body = true;
                }
                _ => return Err(err(format!("unexpected `{head}` outside BEGIN/END"))),
            }
            continue;
        }
        if head == "END" {
            in_body = false;
            seen_end = true;
            continue;
        }
        let mut qs = Vec::with_capacity(args.len());
        for a in &args {
            match index.get(*a) {
                Some(&q) => {
                    if qs.contains(&q) {
                        return Err(err(format!("qubit `{a}` used twice in one gate")));
                    }
                    qs.push(q)
                }
                None => return Err(err(format!("unknown qubit `{a}`"))),
            }
        }
        let g = match (head, qs.as_slice()) {
            ("H", &[q]) => QcGate::H(q),
            ("X", &[q]) | ("tof", &[q]) | ("not", &[q]) => QcGate::X(q),
            ("Y", &[q]) => QcGate::Y(q),
            ("Z", &[q]) | ("Zd", &[q]) => QcGate::Z(q),
            ("S", &[q]) | ("P", &[q]) => QcGate::S(q),
            ("S*", &[q]) | ("P*", &[q]) => QcGate::Sdg(q),
            ("T", &[q]) => QcGate::T(q),
            ("T*", &[q]) => QcGate::Tdg(q),
            ("tof", &[a, b]) | ("cnot", &[a, b]) => QcGate::Cnot(a, b),
            ("Z", &[a, b]) | ("Zd", &[a, b]) => QcGate::Cz(a, b),
            ("Z", &[a, b, c]) | ("Zd", &[a, b, c]) => QcGate::Ccz(a, b, c),
            ("tof", &[a, b, t]) => QcGate::Toffoli(a, b, t),
            ("swap", &[a, b]) => QcGate::Swap(a, b),
            _ => {
                return Err(err(format!(
                    "unsupported gate `{head}` with {} qubit(s)",
                    qs.len()
                )))
            }
        };
        gates.push(g);
    }
    if in_body {
        return Err(QcError {
            line: 0,
            msg: "missing END".into(),
        });
    }
    let inputs = match input_names {
        None => vec![true; names.len()],
        Some(list) => {
            let mut v = vec![false; names.len()];
            for a in list {
                match index.get(&a) {
                    Some(&q) => v[q] = true,
                    None => {
                        return Err(QcError {
                            line: 0,
                            msg: format!("input `{a}` is not a declared qubit"),
                        })
                    }
                }
            }
            v
        }
    };
    Ok(QcCircuit {
        names,
        inputs,
        gates,
    })
}

/// Writes a circuit in `.qc` form (qubits named `q0, q1, ...`, every qubit
/// declared an input).
pub fn to_qc(c: &crate::Circuit) -> Result<String, QcError> {
    use crate::circuit::Op;
    use crate::gate::Gate;
    let n = c.num_qubits;
    let names: Vec<String> = (0..n).map(|q| format!("q{q}")).collect();
    let mut s = String::new();
    s.push_str(".v");
    for nm in &names {
        s.push(' ');
        s.push_str(nm);
    }
    s.push_str("\n.i");
    for nm in &names {
        s.push(' ');
        s.push_str(nm);
    }
    s.push_str("\n\nBEGIN\n");
    for op in &c.ops {
        let line = match op {
            Op::Gate(g) => match *g {
                Gate::H(q) => format!("H {}", names[q]),
                Gate::X(q) => format!("X {}", names[q]),
                Gate::Y(q) => format!("Y {}", names[q]),
                Gate::Z(q) => format!("Z {}", names[q]),
                Gate::S(q) => format!("S {}", names[q]),
                Gate::Sdg(q) => format!("S* {}", names[q]),
                Gate::T(q) => format!("T {}", names[q]),
                Gate::Tdg(q) => format!("T* {}", names[q]),
                Gate::Cnot(a, b) => format!("tof {} {}", names[a], names[b]),
                Gate::Cz(a, b) => format!("Z {} {}", names[a], names[b]),
                Gate::Swap(a, b) => format!("swap {} {}", names[a], names[b]),
                Gate::Ccx(a, b, t) => format!("tof {} {} {}", names[a], names[b], names[t]),
                other => {
                    return Err(QcError {
                        line: 0,
                        msg: format!("gate {other:?} has no .qc form"),
                    })
                }
            },
            _ => {
                return Err(QcError {
                    line: 0,
                    msg: "only unitary gates have a .qc form".into(),
                })
            }
        };
        s.push_str(&line);
        s.push('\n');
    }
    s.push_str("END\n");
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tof3() {
        let src = ".v 1 2 3 4 5\n.i 1 2 3 4\n\nBEGIN\nH 5\nZ 1 2 5\nH 5\ntof 1 2\nT* 3\nEND\n";
        let c = parse_qc(src).unwrap();
        assert_eq!(c.num_qubits(), 5);
        assert_eq!(c.inputs, vec![true, true, true, true, false]);
        assert_eq!(
            c.gates,
            vec![
                QcGate::H(4),
                QcGate::Ccz(0, 1, 4),
                QcGate::H(4),
                QcGate::Cnot(0, 1),
                QcGate::Tdg(2)
            ]
        );
        assert_eq!(c.t_count(), 8);
    }

    #[test]
    fn rejects_unknown() {
        assert!(parse_qc(".v a\nBEGIN\nfoo a\nEND\n").is_err());
        assert!(parse_qc(".v a\nBEGIN\nH b\nEND\n").is_err());
        assert!(parse_qc(".v a b\nBEGIN\ntof a a\nEND\n").is_err());
        assert!(parse_qc(".v a\nBEGIN\nH a\n").is_err());
    }
}
