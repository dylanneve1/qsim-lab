//! Conversions between Python values and engine types: gate names,
//! bitstrings, Pauli strings, memory budgets.

use crate::errors::{circuit_err, qerr, value_err};
use pyo3::prelude::*;
use pyo3::types::{PyString, PyTuple};
use qsim_lab::Gate;

/// `(name, num_qubits, num_params)` of every gate of the engine IR, in the
/// canonical (Python) spelling.
pub const GATES: &[(&str, usize, usize)] = &[
    ("i", 1, 0),
    ("h", 1, 0),
    ("x", 1, 0),
    ("y", 1, 0),
    ("z", 1, 0),
    ("s", 1, 0),
    ("sdg", 1, 0),
    ("t", 1, 0),
    ("tdg", 1, 0),
    ("sx", 1, 0),
    ("sxdg", 1, 0),
    ("rx", 1, 1),
    ("ry", 1, 1),
    ("rz", 1, 1),
    ("p", 1, 1),
    ("u", 1, 3),
    ("cx", 2, 0),
    ("cz", 2, 0),
    ("swap", 2, 0),
    ("iswap", 2, 0),
    ("iswapdg", 2, 0),
    ("cp", 2, 1),
    ("ccx", 3, 0),
];

/// Aliases accepted by `Circuit.append`.
pub const ALIASES: &[(&str, &str)] = &[
    ("id", "i"),
    ("cnot", "cx"),
    ("phase", "p"),
    ("u1", "p"),
    ("u3", "u"),
    ("cphase", "cp"),
    ("cu1", "cp"),
    ("toffoli", "ccx"),
];

pub fn canonical_name(name: &str) -> Option<&'static str> {
    let lower = name.to_ascii_lowercase();
    let lower = ALIASES
        .iter()
        .find(|(a, _)| *a == lower)
        .map(|(_, c)| *c)
        .unwrap_or(lower.as_str())
        .to_string();
    GATES.iter().find(|g| g.0 == lower).map(|g| g.0)
}

/// Builds a gate from its name, qubits and parameters, checking arity.
pub fn gate_from_parts(name: &str, qs: &[usize], ps: &[f64]) -> PyResult<Gate> {
    let canon = canonical_name(name).ok_or_else(|| {
        circuit_err(format!(
            "unknown gate '{name}'; known gates: {}",
            GATES.iter().map(|g| g.0).collect::<Vec<_>>().join(", ")
        ))
    })?;
    let &(_, nq, np) = GATES.iter().find(|g| g.0 == canon).unwrap();
    if qs.len() != nq || ps.len() != np {
        return Err(circuit_err(format!(
            "gate '{canon}' takes {nq} qubit(s) and {np} parameter(s), got {} and {}",
            qs.len(),
            ps.len()
        )));
    }
    if let Some(p) = ps.iter().find(|p| !p.is_finite()) {
        return Err(circuit_err(format!(
            "gate '{canon}': parameter {p} is not finite"
        )));
    }
    use Gate::*;
    Ok(match canon {
        "i" => I(qs[0]),
        "h" => H(qs[0]),
        "x" => X(qs[0]),
        "y" => Y(qs[0]),
        "z" => Z(qs[0]),
        "s" => S(qs[0]),
        "sdg" => Sdg(qs[0]),
        "t" => T(qs[0]),
        "tdg" => Tdg(qs[0]),
        "sx" => Sx(qs[0]),
        "sxdg" => Sxdg(qs[0]),
        "rx" => Rx(qs[0], ps[0]),
        "ry" => Ry(qs[0], ps[0]),
        "rz" => Rz(qs[0], ps[0]),
        "p" => Phase(qs[0], ps[0]),
        "u" => U(qs[0], ps[0], ps[1], ps[2]),
        "cx" => Cnot(qs[0], qs[1]),
        "cz" => Cz(qs[0], qs[1]),
        "swap" => Swap(qs[0], qs[1]),
        "iswap" => ISwap(qs[0], qs[1]),
        "iswapdg" => ISwapdg(qs[0], qs[1]),
        "cp" => CPhase(qs[0], qs[1], ps[0]),
        "ccx" => Ccx(qs[0], qs[1], qs[2]),
        _ => unreachable!(),
    })
}

/// `(name, qubits, params)` of a gate, canonical spelling.
pub fn gate_parts(g: &Gate) -> (&'static str, Vec<usize>, Vec<f64>) {
    use Gate::*;
    let qs = g.qubits();
    let (name, ps): (&'static str, Vec<f64>) = match *g {
        I(_) => ("i", vec![]),
        H(_) => ("h", vec![]),
        X(_) => ("x", vec![]),
        Y(_) => ("y", vec![]),
        Z(_) => ("z", vec![]),
        S(_) => ("s", vec![]),
        Sdg(_) => ("sdg", vec![]),
        T(_) => ("t", vec![]),
        Tdg(_) => ("tdg", vec![]),
        Sx(_) => ("sx", vec![]),
        Sxdg(_) => ("sxdg", vec![]),
        Rx(_, t) => ("rx", vec![t]),
        Ry(_, t) => ("ry", vec![t]),
        Rz(_, t) => ("rz", vec![t]),
        Phase(_, t) => ("p", vec![t]),
        U(_, a, b, c) => ("u", vec![a, b, c]),
        Cnot(..) => ("cx", vec![]),
        Cz(..) => ("cz", vec![]),
        Swap(..) => ("swap", vec![]),
        ISwap(..) => ("iswap", vec![]),
        ISwapdg(..) => ("iswapdg", vec![]),
        CPhase(_, _, t) => ("cp", vec![t]),
        Ccx(..) => ("ccx", vec![]),
    };
    (name, qs, ps)
}

/// Parses one bitstring: a Python int (bit q = qubit q) or a str of
/// `0`/`1` whose rightmost character is qubit 0 (optional `0b` prefix,
/// `_` separators allowed).
pub fn parse_bitstring(obj: &Bound<'_, PyAny>, n: usize) -> PyResult<u128> {
    let x: u128 = if let Ok(s) = obj.cast::<PyString>() {
        let s: String = s.extract()?;
        let s = s.as_str();
        let t = s.trim().trim_start_matches("0b").replace('_', "");
        if t.is_empty() || !t.chars().all(|c| c == '0' || c == '1') {
            return Err(circuit_err(format!(
                "bitstring '{s}' must contain only 0 and 1 (rightmost = qubit 0)"
            )));
        }
        if t.len() != n {
            return Err(circuit_err(format!(
                "bitstring '{s}' has {} characters for a {n}-qubit circuit",
                t.len()
            )));
        }
        u128::from_str_radix(&t, 2).map_err(|e| circuit_err(e.to_string()))?
    } else {
        obj.extract::<u128>().map_err(|_| {
            circuit_err(format!(
                "a basis state must be a non-negative int or a 0/1 string, got {}",
                obj.repr().map(|r| r.to_string()).unwrap_or_default()
            ))
        })?
    };
    if n < 128 && x >> n != 0 {
        return Err(qerr(
            "QubitIndexError",
            format!("basis state {x} does not fit in {n} qubits"),
        ));
    }
    Ok(x)
}

pub fn parse_bitstrings(obj: &Bound<'_, PyAny>, n: usize) -> PyResult<Vec<u128>> {
    if n > 128 {
        return Err(qerr(
            "UnsupportedOperationError",
            "amplitudes are indexed up to 128 qubits",
        ));
    }
    if obj.is_instance_of::<PyString>() {
        return Ok(vec![parse_bitstring(obj, n)?]);
    }
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        out.push(parse_bitstring(&item?, n)?);
    }
    Ok(out)
}

/// One Pauli string: overall sign and `(qubit, op)` with op 1 = X, 2 = Y, 3 = Z.
#[derive(Clone, Debug, PartialEq)]
pub struct PauliTerm {
    pub sign: f64,
    pub ops: Vec<(usize, u8)>,
}

impl PauliTerm {
    pub fn masks(&self) -> (u128, u128, u32) {
        let (mut xm, mut zm, mut ny) = (0u128, 0u128, 0u32);
        for &(q, o) in &self.ops {
            match o {
                1 => xm |= 1 << q,
                2 => {
                    xm |= 1 << q;
                    zm |= 1 << q;
                    ny += 1;
                }
                _ => zm |= 1 << q,
            }
        }
        (xm, zm, ny)
    }
}

fn pauli_code(c: char) -> Option<u8> {
    match c.to_ascii_uppercase() {
        'I' => Some(0),
        'X' => Some(1),
        'Y' => Some(2),
        'Z' => Some(3),
        _ => None,
    }
}

/// Parses a Pauli string (see API.md §5): dense `"XIZ"` (rightmost =
/// qubit 0, length `n`) or sparse `"X0 Z2"` / `"X0*Z2"`, optional sign.
pub fn parse_pauli(s: &str, n: usize) -> PyResult<PauliTerm> {
    let bad = |why: &str| value_err(format!("bad Pauli string '{s}': {why}"));
    let mut t = s.trim();
    let mut sign = 1.0;
    if let Some(r) = t.strip_prefix('-') {
        sign = -1.0;
        t = r.trim_start();
    } else if let Some(r) = t.strip_prefix('+') {
        t = r.trim_start();
    }
    let mut ops: Vec<(usize, u8)> = Vec::new();
    if t.chars().any(|c| c.is_ascii_digit()) {
        let chars: Vec<char> = t.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if c.is_whitespace() || c == '*' || c == ',' {
                i += 1;
                continue;
            }
            let code = pauli_code(c).ok_or_else(|| bad("expected one of I, X, Y, Z"))?;
            i += 1;
            let start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            if start == i {
                return Err(bad("every Pauli letter needs a qubit index in sparse form"));
            }
            let q: usize = chars[start..i]
                .iter()
                .collect::<String>()
                .parse()
                .map_err(|_| bad("qubit index"))?;
            if q >= n {
                return Err(qerr(
                    "QubitIndexError",
                    format!("Pauli '{s}': qubit {q} out of range for {n} qubits"),
                ));
            }
            if ops.iter().any(|&(p, _)| p == q) {
                return Err(bad("qubit named twice"));
            }
            if code != 0 {
                ops.push((q, code));
            }
        }
    } else {
        let letters: Vec<char> = t.chars().filter(|c| !c.is_whitespace()).collect();
        let identity = letters.is_empty() || letters.iter().all(|&c| c == 'I' || c == 'i');
        if !identity && letters.len() != n {
            return Err(bad(&format!(
                "dense form needs {n} letters (rightmost = qubit 0), got {}; \
                 use the sparse form like 'X0 Z3' otherwise",
                letters.len()
            )));
        }
        for (k, &c) in letters.iter().rev().enumerate() {
            let code = pauli_code(c).ok_or_else(|| bad("expected one of I, X, Y, Z"))?;
            if code != 0 {
                ops.push((k, code));
            }
        }
    }
    ops.sort_unstable();
    Ok(PauliTerm { sign, ops })
}

/// Memory budget in bytes from an int, float or a string like `"4GiB"`,
/// `"512 MB"`, `"1e9"`.
pub fn parse_memory(obj: &Bound<'_, PyAny>) -> PyResult<u128> {
    if let Ok(v) = obj.extract::<u128>() {
        return Ok(v);
    }
    if let Ok(v) = obj.extract::<f64>() {
        if v.is_finite() && v >= 0.0 {
            return Ok(v as u128);
        }
    }
    let s: String = obj
        .extract()
        .map_err(|_| value_err("memory budget must be an int (bytes) or a string like '4GiB'"))?;
    parse_memory_str(&s)
}

pub fn parse_memory_str(s: &str) -> PyResult<u128> {
    let t = s.trim().replace([' ', '_'], "");
    // the unit is the trailing run of letters (so "1e9" stays a number)
    let split = t.trim_end_matches(|c: char| c.is_ascii_alphabetic()).len();
    let (num, unit) = t.split_at(split);
    let v: f64 = num
        .parse()
        .map_err(|_| value_err(format!("bad memory size '{s}'")))?;
    let mult: f64 = match unit.to_ascii_lowercase().as_str() {
        "" | "b" => 1.0,
        "k" | "kb" => 1e3,
        "m" | "mb" => 1e6,
        "g" | "gb" => 1e9,
        "t" | "tb" => 1e12,
        "kib" => 1024.0,
        "mib" => 1024f64.powi(2),
        "gib" => 1024f64.powi(3),
        "tib" => 1024f64.powi(4),
        _ => return Err(value_err(format!("bad memory unit in '{s}'"))),
    };
    if !(v.is_finite() && v >= 0.0) {
        return Err(value_err(format!("bad memory size '{s}'")));
    }
    Ok((v * mult) as u128)
}

/// `(meas_index, value)` from `c_if=None | int | (int, bool)`.
pub fn parse_condition(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<(usize, bool)>> {
    let Some(o) = obj else { return Ok(None) };
    if o.is_none() {
        return Ok(None);
    }
    if let Ok(t) = o.cast::<PyTuple>() {
        if t.len() != 2 {
            return Err(value_err(
                "c_if must be an int or a (measurement, value) pair",
            ));
        }
        let k: usize = t.get_item(0)?.extract()?;
        let v = t.get_item(1)?;
        let v: bool = if let Ok(b) = v.extract::<bool>() {
            b
        } else {
            match v.extract::<i64>()? {
                0 => false,
                1 => true,
                _ => return Err(value_err("c_if value must be 0/1 or a bool")),
            }
        };
        return Ok(Some((k, v)));
    }
    if o.extract::<bool>().is_ok() && o.is_instance_of::<pyo3::types::PyBool>() {
        return Err(value_err(
            "c_if takes a measurement index (int) or a (measurement, value) pair, not a bool",
        ));
    }
    Ok(Some((o.extract::<usize>()?, true)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_gate_round_trips_through_its_parts() {
        for &(name, nq, np) in GATES {
            let qs: Vec<usize> = (0..nq).collect();
            let ps: Vec<f64> = (0..np).map(|k| 0.1 + k as f64).collect();
            let g = gate_from_parts(name, &qs, &ps).unwrap();
            let (n2, q2, p2) = gate_parts(&g);
            assert_eq!((n2, q2, p2), (name, qs, ps));
        }
    }

    #[test]
    fn memory_strings() {
        assert_eq!(parse_memory_str("4GiB").unwrap(), 4 << 30);
        assert_eq!(parse_memory_str("512 MB").unwrap(), 512_000_000);
        assert_eq!(parse_memory_str("1e9").unwrap(), 1_000_000_000);
        assert_eq!(parse_memory_str("2k").unwrap(), 2000);
    }

    #[test]
    fn pauli_masks() {
        let t = PauliTerm {
            sign: 1.0,
            ops: vec![(0, 1), (2, 2), (3, 3)],
        };
        assert_eq!(t.masks(), (0b0101, 0b1100, 1));
    }
}
