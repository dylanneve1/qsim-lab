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

// ---------------------------------------------------------------------------
// Parser
//
// A complete OpenQASM 2.0 front end: tokenizer with line numbers, `qreg` /
// `creg` (laid out in declaration order), user `gate` definitions (expanded
// as macros, with parameters), `opaque` (rejected when used), register
// broadcasting (`h q;`, `cx a,b;` over equal-size registers, `measure q -> c;`),
// `barrier` (ignored), `reset`, `if (c == v) op;` for a one-bit register
// (becomes a classically controlled gate on the measurement that last wrote
// that bit), and the qelib1.inc gate set plus the common Qiskit extensions.
//
// Gates with a direct engine equivalent map to it exactly (so
// `from_qasm(to_qasm(c)) == c`); the others are expanded with exact
// decompositions (as matrices, global phase included, following Qiskit's
// gate definitions, e.g. `rz(θ) = exp(-iθZ/2)`, `crz`, `cu3`, `rxx`, ...).

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Id(String),
    Num(f64),
    Str(String),
    Sym(&'static str),
}

#[derive(Clone, Debug)]
struct Token {
    tok: Tok,
    line: usize,
}

fn qerr(line: usize, msg: impl std::fmt::Display) -> SimError {
    SimError::QasmError(format!("line {line}: {msg}"))
}

fn tokenize(src: &str) -> Result<Vec<Token>, SimError> {
    let b = src.as_bytes();
    let mut i = 0;
    let mut line = 1;
    let mut out = Vec::new();
    const SYMS: [&str; 16] = [
        "->", "==", ";", ",", "(", ")", "[", "]", "{", "}", "+", "-", "*", "/", "^", ".",
    ];
    while i < b.len() {
        let c = b[i];
        if c == b'\n' {
            line += 1;
            i += 1;
        } else if c.is_ascii_whitespace() {
            i += 1;
        } else if b[i..].starts_with(b"//") {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i..].starts_with(b"/*") {
            i += 2;
            while i < b.len() && !b[i..].starts_with(b"*/") {
                if b[i] == b'\n' {
                    line += 1;
                }
                i += 1;
            }
            i += 2;
        } else if c.is_ascii_alphabetic() || c == b'_' {
            let s = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            out.push(Token {
                tok: Tok::Id(src[s..i].to_string()),
                line,
            });
        } else if c.is_ascii_digit() || (c == b'.' && i + 1 < b.len() && b[i + 1].is_ascii_digit())
        {
            let s = i;
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.') {
                i += 1;
            }
            if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                let save = i;
                i += 1;
                if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
                    i += 1;
                }
                if i < b.len() && b[i].is_ascii_digit() {
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                } else {
                    i = save;
                }
            }
            let v: f64 = src[s..i]
                .parse()
                .map_err(|_| qerr(line, format!("bad number '{}'", &src[s..i])))?;
            out.push(Token {
                tok: Tok::Num(v),
                line,
            });
        } else if c == b'"' {
            let s = i + 1;
            i += 1;
            while i < b.len() && b[i] != b'"' {
                i += 1;
            }
            if i >= b.len() {
                return Err(qerr(line, "unterminated string"));
            }
            out.push(Token {
                tok: Tok::Str(src[s..i].to_string()),
                line,
            });
            i += 1;
        } else if let Some(sym) = SYMS.iter().find(|s| b[i..].starts_with(s.as_bytes())) {
            out.push(Token {
                tok: Tok::Sym(sym),
                line,
            });
            i += sym.len();
        } else {
            return Err(qerr(line, format!("unexpected character '{}'", c as char)));
        }
    }
    Ok(out)
}

/// A parameter expression, kept as tokens and evaluated per call (gate
/// bodies refer to the gate's parameters).
#[derive(Clone, Debug)]
struct Expr(Vec<Token>);

struct ExprEval<'a> {
    t: &'a [Token],
    i: usize,
    env: &'a HashMap<String, f64>,
    line: usize,
}

impl ExprEval<'_> {
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.i).map(|t| &t.tok)
    }
    fn err(&self, msg: &str) -> SimError {
        qerr(self.line, format!("bad parameter expression: {msg}"))
    }
    fn eat(&mut self, s: &str) -> bool {
        if matches!(self.peek(), Some(Tok::Sym(x)) if *x == s) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn expr(&mut self) -> Result<f64, SimError> {
        let mut v = self.term()?;
        loop {
            if self.eat("+") {
                v += self.term()?;
            } else if self.eat("-") {
                v -= self.term()?;
            } else {
                return Ok(v);
            }
        }
    }
    fn term(&mut self) -> Result<f64, SimError> {
        let mut v = self.unary()?;
        loop {
            if self.eat("*") {
                v *= self.unary()?;
            } else if self.eat("/") {
                let d = self.unary()?;
                if d == 0.0 {
                    return Err(self.err("division by zero"));
                }
                v /= d;
            } else {
                return Ok(v);
            }
        }
    }
    fn unary(&mut self) -> Result<f64, SimError> {
        if self.eat("-") {
            Ok(-self.unary()?)
        } else if self.eat("+") {
            self.unary()
        } else {
            self.power()
        }
    }
    fn power(&mut self) -> Result<f64, SimError> {
        let base = self.atom()?;
        if self.eat("^") {
            let e = self.unary()?;
            Ok(base.powf(e))
        } else {
            Ok(base)
        }
    }
    fn atom(&mut self) -> Result<f64, SimError> {
        match self.peek().cloned() {
            Some(Tok::Num(v)) => {
                self.i += 1;
                Ok(v)
            }
            Some(Tok::Sym("(")) => {
                self.i += 1;
                let v = self.expr()?;
                if !self.eat(")") {
                    return Err(self.err("missing ')'"));
                }
                Ok(v)
            }
            Some(Tok::Id(name)) => {
                self.i += 1;
                if name == "pi" {
                    return Ok(PI);
                }
                if let Some(v) = self.env.get(&name) {
                    return Ok(*v);
                }
                let f: fn(f64) -> f64 = match name.as_str() {
                    "sin" => f64::sin,
                    "cos" => f64::cos,
                    "tan" => f64::tan,
                    "exp" => f64::exp,
                    "ln" => f64::ln,
                    "sqrt" => f64::sqrt,
                    _ => return Err(self.err(&format!("unknown identifier '{name}'"))),
                };
                if !self.eat("(") {
                    return Err(self.err(&format!("'{name}' needs parentheses")));
                }
                let v = self.expr()?;
                if !self.eat(")") {
                    return Err(self.err("missing ')'"));
                }
                Ok(f(v))
            }
            _ => Err(self.err("expected a number, 'pi', a parameter or '('")),
        }
    }
}

impl Expr {
    fn eval(&self, env: &HashMap<String, f64>, line: usize) -> Result<f64, SimError> {
        let mut e = ExprEval {
            t: &self.0,
            i: 0,
            env,
            line,
        };
        let v = e.expr()?;
        if e.i != self.0.len() {
            return Err(e.err("trailing tokens"));
        }
        if !v.is_finite() {
            return Err(e.err("value is not finite"));
        }
        Ok(v)
    }
}

/// A qubit or bit argument: a whole register or one element.
#[derive(Clone, Debug)]
enum Arg {
    Reg(String),
    Bit(String, usize),
}

/// One gate call inside a `gate` body.
#[derive(Clone, Debug)]
struct BodyCall {
    name: String,
    params: Vec<Expr>,
    args: Vec<String>,
    line: usize,
}

#[derive(Clone, Debug)]
struct GateDef {
    params: Vec<String>,
    qargs: Vec<String>,
    body: Vec<BodyCall>,
    opaque: bool,
}

/// qelib1.inc gates (and Qiskit's usual extensions) without a direct engine
/// gate, defined in terms of gates that have one. Exact as matrices.
const BUILTIN_DEFS: &str = "
gate u2(phi,lambda) q { u3(pi/2,phi,lambda) q; }
gate u0(gamma) q { id q; }
gate r(theta,phi) q { u3(theta,phi-pi/2,-phi+pi/2) q; }
gate cy a,b { sdg b; cx a,b; s b; }
gate ch a,b { s b; h b; t b; cx a,b; tdg b; h b; sdg b; }
gate crz(lambda) a,b { rz(lambda/2) b; cx a,b; rz(-lambda/2) b; cx a,b; }
gate cry(theta) a,b { ry(theta/2) b; cx a,b; ry(-theta/2) b; cx a,b; }
gate crx(theta) a,b { p(pi/2) b; cx a,b; u3(-theta/2,0,0) b; cx a,b; u3(theta/2,-pi/2,0) b; }
gate cu3(theta,phi,lambda) c,t { p((lambda+phi)/2) c; p((lambda-phi)/2) t; cx c,t; u3(-theta/2,0,-(phi+lambda)/2) t; cx c,t; u3(theta/2,phi,0) t; }
gate cu(theta,phi,lambda,gamma) c,t { p(gamma) c; cu3(theta,phi,lambda) c,t; }
gate csx a,b { h b; cp(pi/2) a,b; h b; }
gate cswap a,b,c { cx c,b; ccx a,b,c; cx c,b; }
gate rzz(theta) a,b { cx a,b; rz(theta) b; cx a,b; }
gate rxx(theta) a,b { h a; h b; rzz(theta) a,b; h a; h b; }
gate ryy(theta) a,b { rx(pi/2) a; rx(pi/2) b; rzz(theta) a,b; rx(-pi/2) a; rx(-pi/2) b; }
gate rzx(theta) a,b { h b; rzz(theta) a,b; h b; }
gate ecr a,b { rzx(pi/4) a,b; x a; rzx(-pi/4) a,b; }
gate dcx a,b { cx a,b; cx b,a; }
gate iswapdg a,b { sdg a; sdg b; cz a,b; swap a,b; }
gate rccx a,b,c { u2(0,pi) c; u1(pi/4) c; cx b,c; u1(-pi/4) c; cx a,c; u1(pi/4) c; cx b,c; u1(-pi/4) c; u2(0,pi) c; }
";

/// Engine gate for a name with a direct equivalent.
fn native_gate(name: &str, q: &[usize], p: &[f64]) -> Option<(usize, usize, Option<Gate>)> {
    let (nq, np) = match name {
        "id" | "i" | "h" | "x" | "y" | "z" | "s" | "sdg" | "t" | "tdg" | "sx" | "sxdg" => (1, 0),
        "rx" | "ry" | "rz" | "u1" | "p" | "phase" => (1, 1),
        "u3" | "u" | "U" => (1, 3),
        "cx" | "CX" | "cnot" | "cz" | "swap" | "iswap" => (2, 0),
        "cp" | "cu1" | "cphase" => (2, 1),
        "ccx" | "toffoli" => (3, 0),
        _ => return None,
    };
    if q.len() != nq || p.len() != np {
        return Some((nq, np, None));
    }
    let g = match name {
        "id" | "i" => Gate::I(q[0]),
        "h" => Gate::H(q[0]),
        "x" => Gate::X(q[0]),
        "y" => Gate::Y(q[0]),
        "z" => Gate::Z(q[0]),
        "s" => Gate::S(q[0]),
        "sdg" => Gate::Sdg(q[0]),
        "t" => Gate::T(q[0]),
        "tdg" => Gate::Tdg(q[0]),
        "sx" => Gate::Sx(q[0]),
        "sxdg" => Gate::Sxdg(q[0]),
        "rx" => Gate::Rx(q[0], p[0]),
        "ry" => Gate::Ry(q[0], p[0]),
        "rz" => Gate::Rz(q[0], p[0]),
        "u1" | "p" | "phase" => Gate::Phase(q[0], p[0]),
        "u3" | "u" | "U" => Gate::U(q[0], p[0], p[1], p[2]),
        "cx" | "CX" | "cnot" => Gate::Cnot(q[0], q[1]),
        "cz" => Gate::Cz(q[0], q[1]),
        "swap" => Gate::Swap(q[0], q[1]),
        "iswap" => Gate::ISwap(q[0], q[1]),
        "cp" | "cu1" | "cphase" => Gate::CPhase(q[0], q[1], p[0]),
        _ => Gate::Ccx(q[0], q[1], q[2]),
    };
    Some((nq, np, Some(g)))
}

struct Parser {
    toks: Vec<Token>,
    i: usize,
    qregs: Vec<(String, usize, usize)>,
    cregs: Vec<(String, usize, usize)>,
    num_clbits: usize,
    defs: HashMap<String, GateDef>,
    /// Measurement index that last wrote each classical bit.
    last_write: Vec<Option<usize>>,
    num_meas: usize,
    ops: Vec<Op>,
}

impl Parser {
    fn line(&self) -> usize {
        self.toks
            .get(self.i)
            .or(self.toks.last())
            .map_or(0, |t| t.line)
    }
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.i).map(|t| &t.tok)
    }
    fn next(&mut self) -> Result<Tok, SimError> {
        let t = self
            .toks
            .get(self.i)
            .map(|t| t.tok.clone())
            .ok_or_else(|| qerr(self.line(), "unexpected end of input"))?;
        self.i += 1;
        Ok(t)
    }
    fn eat(&mut self, s: &str) -> bool {
        if matches!(self.peek(), Some(Tok::Sym(x)) if *x == s) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, s: &str) -> Result<(), SimError> {
        if self.eat(s) {
            Ok(())
        } else {
            Err(qerr(
                self.line(),
                format!("expected '{s}', found {}", self.describe()),
            ))
        }
    }
    fn describe(&self) -> String {
        match self.peek() {
            Some(Tok::Id(s)) => format!("'{s}'"),
            Some(Tok::Num(v)) => format!("'{v}'"),
            Some(Tok::Str(s)) => format!("\"{s}\""),
            Some(Tok::Sym(s)) => format!("'{s}'"),
            None => "end of input".into(),
        }
    }
    fn ident(&mut self) -> Result<String, SimError> {
        match self.next()? {
            Tok::Id(s) => Ok(s),
            _ => {
                self.i -= 1;
                Err(qerr(
                    self.line(),
                    format!("expected a name, found {}", self.describe()),
                ))
            }
        }
    }
    fn uint(&mut self) -> Result<usize, SimError> {
        match self.next()? {
            Tok::Num(v) if v >= 0.0 && v.fract() == 0.0 => Ok(v as usize),
            _ => {
                self.i -= 1;
                Err(qerr(
                    self.line(),
                    format!("expected a non-negative integer, found {}", self.describe()),
                ))
            }
        }
    }
    /// Tokens of one expression, up to a top-level `,` or the closing `)`.
    fn expr_tokens(&mut self) -> Result<Expr, SimError> {
        let mut depth = 0usize;
        let mut out = Vec::new();
        loop {
            match self.peek() {
                None => return Err(qerr(self.line(), "unterminated parameter list")),
                Some(Tok::Sym(",")) if depth == 0 => break,
                Some(Tok::Sym(")")) if depth == 0 => break,
                Some(Tok::Sym(";")) => return Err(qerr(self.line(), "unbalanced parentheses")),
                Some(Tok::Sym("(")) => depth += 1,
                Some(Tok::Sym(")")) => depth -= 1,
                _ => {}
            }
            out.push(self.toks[self.i].clone());
            self.i += 1;
        }
        if out.is_empty() {
            return Err(qerr(self.line(), "empty parameter"));
        }
        Ok(Expr(out))
    }
    fn param_list(&mut self) -> Result<Vec<Expr>, SimError> {
        let mut ps = Vec::new();
        if self.eat("(") {
            if self.eat(")") {
                return Ok(ps);
            }
            loop {
                ps.push(self.expr_tokens()?);
                if self.eat(")") {
                    break;
                }
                self.expect(",")?;
            }
        }
        Ok(ps)
    }
    fn arg(&mut self) -> Result<Arg, SimError> {
        let name = self.ident()?;
        if self.eat("[") {
            let k = self.uint()?;
            self.expect("]")?;
            Ok(Arg::Bit(name, k))
        } else {
            Ok(Arg::Reg(name))
        }
    }
    fn reg(
        &self,
        regs: &[(String, usize, usize)],
        a: &Arg,
        kind: &str,
    ) -> Result<Vec<usize>, SimError> {
        let line = self.line();
        let name = match a {
            Arg::Reg(n) | Arg::Bit(n, _) => n,
        };
        let &(_, off, size) = regs
            .iter()
            .find(|r| &r.0 == name)
            .ok_or_else(|| qerr(line, format!("unknown {kind} register '{name}'")))?;
        match a {
            Arg::Reg(_) => Ok((off..off + size).collect()),
            Arg::Bit(_, k) => {
                if *k >= size {
                    Err(qerr(
                        line,
                        format!("index {k} out of range for register '{name}' of size {size}"),
                    ))
                } else {
                    Ok(vec![off + k])
                }
            }
        }
    }
    /// Broadcast a list of register/element arguments into argument tuples.
    fn broadcast(&self, args: &[Vec<usize>]) -> Result<Vec<Vec<usize>>, SimError> {
        let len = args
            .iter()
            .map(|a| a.len())
            .filter(|&l| l != 1)
            .max()
            .unwrap_or(1);
        if args.iter().any(|a| a.len() != 1 && a.len() != len) {
            return Err(qerr(
                self.line(),
                "registers of different sizes in one statement",
            ));
        }
        Ok((0..len)
            .map(|k| {
                args.iter()
                    .map(|a| if a.len() == 1 { a[0] } else { a[k] })
                    .collect()
            })
            .collect())
    }

    fn emit(
        &mut self,
        name: &str,
        params: &[f64],
        qubits: &[usize],
        cond: Option<(usize, bool)>,
        line: usize,
        depth: usize,
    ) -> Result<(), SimError> {
        for (i, q) in qubits.iter().enumerate() {
            if qubits[..i].contains(q) {
                return Err(qerr(line, format!("'{name}' uses qubit {q} twice")));
            }
        }
        if let Some((nq, np, g)) = native_gate(name, qubits, params) {
            let g = g.ok_or_else(|| {
                qerr(
                    line,
                    format!(
                        "'{name}' takes {nq} qubit(s) and {np} parameter(s), got {} and {}",
                        qubits.len(),
                        params.len()
                    ),
                )
            })?;
            self.ops.push(match cond {
                None => Op::Gate(g),
                Some((meas_index, target_value)) => Op::ClassicControlled {
                    gate: g,
                    meas_index,
                    target_value,
                },
            });
            return Ok(());
        }
        let def = self
            .defs
            .get(name)
            .cloned()
            .ok_or_else(|| qerr(line, format!("unrecognized QASM gate: '{name}'")))?;
        if def.opaque {
            return Err(qerr(
                line,
                format!("opaque gate '{name}' has no definition"),
            ));
        }
        if def.params.len() != params.len() || def.qargs.len() != qubits.len() {
            return Err(qerr(
                line,
                format!(
                    "'{name}' takes {} qubit(s) and {} parameter(s), got {} and {}",
                    def.qargs.len(),
                    def.params.len(),
                    qubits.len(),
                    params.len()
                ),
            ));
        }
        if depth > 64 {
            return Err(qerr(line, "gate definitions nest too deeply (recursive?)"));
        }
        let env: HashMap<String, f64> = def
            .params
            .iter()
            .cloned()
            .zip(params.iter().copied())
            .collect();
        for call in &def.body {
            let ps = call
                .params
                .iter()
                .map(|e| e.eval(&env, call.line))
                .collect::<Result<Vec<_>, _>>()?;
            let qs = call
                .args
                .iter()
                .map(|a| {
                    def.qargs
                        .iter()
                        .position(|x| x == a)
                        .map(|k| qubits[k])
                        .ok_or_else(|| {
                            qerr(
                                call.line,
                                format!("unknown qubit argument '{a}' in gate '{name}'"),
                            )
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.emit(&call.name, &ps, &qs, cond, call.line, depth + 1)?;
        }
        Ok(())
    }

    fn gate_def(&mut self, opaque: bool) -> Result<(), SimError> {
        let name = self.ident()?;
        let mut params = Vec::new();
        if self.eat("(") && !self.eat(")") {
            loop {
                params.push(self.ident()?);
                if self.eat(")") {
                    break;
                }
                self.expect(",")?;
            }
        }
        let mut qargs = vec![self.ident()?];
        while self.eat(",") {
            qargs.push(self.ident()?);
        }
        let mut body = Vec::new();
        if opaque {
            self.expect(";")?;
        } else {
            self.expect("{")?;
            while !self.eat("}") {
                let line = self.line();
                let cname = self.ident()?;
                if cname == "barrier" {
                    while !self.eat(";") {
                        self.next()?;
                    }
                    continue;
                }
                let ps = self.param_list()?;
                let mut args = vec![self.ident()?];
                while self.eat(",") {
                    args.push(self.ident()?);
                }
                self.expect(";")?;
                body.push(BodyCall {
                    name: cname,
                    params: ps,
                    args,
                    line,
                });
            }
        }
        self.defs.insert(
            name,
            GateDef {
                params,
                qargs,
                body,
                opaque,
            },
        );
        Ok(())
    }

    /// A quantum operation (gate call, measure, reset), optionally conditioned.
    fn qop(&mut self, cond: Option<(usize, bool)>) -> Result<(), SimError> {
        let line = self.line();
        let name = self.ident()?;
        match name.as_str() {
            "measure" => {
                if cond.is_some() {
                    return Err(qerr(line, "conditional measurement is not supported"));
                }
                let q = self.arg()?;
                self.expect("->")?;
                let c = self.arg()?;
                self.expect(";")?;
                let qs = self.reg(&self.qregs, &q, "quantum")?;
                let cs = self.reg(&self.cregs, &c, "classical")?;
                if qs.len() != cs.len() {
                    return Err(qerr(line, "measure: register sizes differ"));
                }
                for (q, c) in qs.into_iter().zip(cs) {
                    self.ops.push(Op::Measure(q));
                    self.last_write[c] = Some(self.num_meas);
                    self.num_meas += 1;
                }
            }
            "reset" => {
                if cond.is_some() {
                    return Err(qerr(line, "conditional reset is not supported"));
                }
                let q = self.arg()?;
                self.expect(";")?;
                for q in self.reg(&self.qregs, &q, "quantum")? {
                    self.ops.push(Op::Reset(q));
                }
            }
            "barrier" => {
                while !self.eat(";") {
                    self.next()?;
                }
            }
            _ => {
                let pexprs = self.param_list()?;
                let empty = HashMap::new();
                let ps = pexprs
                    .iter()
                    .map(|e| e.eval(&empty, line))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut args = Vec::new();
                if !matches!(self.peek(), Some(Tok::Sym(";"))) {
                    args.push(self.arg()?);
                    while self.eat(",") {
                        args.push(self.arg()?);
                    }
                }
                self.expect(";")?;
                let lists = args
                    .iter()
                    .map(|a| self.reg(&self.qregs, a, "quantum"))
                    .collect::<Result<Vec<_>, _>>()?;
                if lists.is_empty() {
                    return Err(qerr(line, format!("'{name}' has no qubit arguments")));
                }
                for qs in self.broadcast(&lists)? {
                    self.emit(&name, &ps, &qs, cond, line, 0)?;
                }
            }
        }
        Ok(())
    }

    fn program(&mut self) -> Result<(), SimError> {
        while self.peek().is_some() {
            let line = self.line();
            let word = match self.peek() {
                Some(Tok::Id(s)) => s.clone(),
                _ => return Err(qerr(line, format!("unexpected {}", self.describe()))),
            };
            match word.as_str() {
                "OPENQASM" => {
                    self.i += 1;
                    match self.next()? {
                        Tok::Num(v) if (2.0..3.0).contains(&v) => {}
                        _ => return Err(qerr(line, "only OpenQASM 2.x is supported")),
                    }
                    self.expect(";")?;
                }
                "include" => {
                    self.i += 1;
                    match self.next()? {
                        Tok::Str(s) if s == "qelib1.inc" => {}
                        Tok::Str(s) => {
                            return Err(qerr(
                                line,
                                format!("cannot include \"{s}\" (only qelib1.inc is built in)"),
                            ))
                        }
                        _ => return Err(qerr(line, "include needs a file name")),
                    }
                    self.expect(";")?;
                }
                "qreg" | "creg" => {
                    self.i += 1;
                    let name = self.ident()?;
                    self.expect("[")?;
                    let size = self.uint()?;
                    self.expect("]")?;
                    self.expect(";")?;
                    let regs = if word == "qreg" {
                        &mut self.qregs
                    } else {
                        &mut self.cregs
                    };
                    if regs.iter().any(|r| r.0 == name) {
                        return Err(qerr(line, format!("register '{name}' declared twice")));
                    }
                    let off = regs.last().map_or(0, |r| r.1 + r.2);
                    regs.push((name, off, size));
                    if word == "creg" {
                        self.num_clbits = off + size;
                        self.last_write.resize(self.num_clbits, None);
                    }
                }
                "gate" | "opaque" => {
                    self.i += 1;
                    self.gate_def(word == "opaque")?;
                }
                "if" => {
                    self.i += 1;
                    self.expect("(")?;
                    let creg = self.ident()?;
                    self.expect("==")?;
                    let v = self.uint()?;
                    self.expect(")")?;
                    let &(_, off, size) =
                        self.cregs.iter().find(|r| r.0 == creg).ok_or_else(|| {
                            qerr(line, format!("unknown classical register '{creg}'"))
                        })?;
                    if size != 1 || v > 1 {
                        return Err(qerr(
                            line,
                            format!(
                                "if ({creg} == {v}): only one-bit registers can be conditioned on \
                                 (the engine conditions on a single measurement)"
                            ),
                        ));
                    }
                    let meas = self.last_write[off].ok_or_else(|| {
                        qerr(line, format!("if ({creg} == {v}): '{creg}' is read before any measurement writes it"))
                    })?;
                    self.qop(Some((meas, v == 1)))?;
                }
                _ => self.qop(None)?,
            }
        }
        Ok(())
    }
}

/// Parses an OpenQASM 2.0 program into a [`Circuit`].
///
/// Qubits of all `qreg`s are numbered in declaration order. Measurement `k`
/// (in program order) is measurement record `k` of the circuit, whatever
/// classical bit it writes. See the module notes above for what is supported.
pub fn from_qasm(source: &str) -> Result<Circuit, SimError> {
    let mut defs_parser = Parser {
        toks: tokenize(BUILTIN_DEFS)?,
        i: 0,
        qregs: Vec::new(),
        cregs: Vec::new(),
        num_clbits: 0,
        defs: HashMap::new(),
        last_write: Vec::new(),
        num_meas: 0,
        ops: Vec::new(),
    };
    defs_parser.program()?;
    let mut p = Parser {
        toks: tokenize(source)?,
        defs: defs_parser.defs,
        ..defs_parser
    };
    p.i = 0;
    p.program()?;
    let n = p.qregs.last().map_or(0, |r| r.1 + r.2);
    Ok(Circuit {
        num_qubits: n,
        ops: p.ops,
    })
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

    #[test]
    fn gate_definitions_broadcast_and_conditionals() {
        let src = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg a[2];
            qreg b[1];
            creg c[2];
            creg f[1];
            gate g(t) x, y { h x; cx x, y; rz(t/2) y; }
            h a;
            g(0.5) a[1], b[0];
            cx a, b;
            measure a -> c;
            measure b[0] -> f[0];
            if (f == 1) x a[0];
        "#;
        let c = from_qasm(src).unwrap();
        assert_eq!(c.num_qubits, 3);
        assert_eq!(
            &c.ops[..5],
            &[
                Op::Gate(Gate::H(0)),
                Op::Gate(Gate::H(1)),
                Op::Gate(Gate::H(1)),
                Op::Gate(Gate::Cnot(1, 2)),
                Op::Gate(Gate::Rz(2, 0.25)),
            ]
        );
        assert_eq!(c.ops[5], Op::Gate(Gate::Cnot(0, 2)));
        assert_eq!(c.ops[6], Op::Gate(Gate::Cnot(1, 2)));
        assert_eq!(
            c.ops.last(),
            Some(&Op::ClassicControlled {
                gate: Gate::X(0),
                meas_index: 2,
                target_value: true
            })
        );
    }

    #[test]
    fn qelib_extras_expand() {
        let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\ncswap q[0],q[1],q[2];\nrzz(0.3) q[0],q[1];\nu2(0,pi) q[2];\n";
        let c = from_qasm(src).unwrap();
        assert_eq!(c.ops[0], Op::Gate(Gate::Cnot(2, 1)));
        assert_eq!(c.ops[1], Op::Gate(Gate::Ccx(0, 1, 2)));
        assert_eq!(c.ops[4], Op::Gate(Gate::Rz(1, 0.3)));
        assert!(matches!(c.ops[6], Op::Gate(Gate::U(2, ..))));
    }

    #[test]
    fn errors_name_the_line() {
        let e = from_qasm("OPENQASM 2.0;\nqreg q[1];\n\nfoo q[0];\n").unwrap_err();
        assert!(e.to_string().contains("line 4"), "{e}");
        assert!(from_qasm("qreg q[2]; creg c[2]; if (c == 1) x q[0];").is_err());
        assert!(from_qasm("qreg q[1]; gate r2 a { r2 a; } r2 q[0];").is_err());
    }
}
