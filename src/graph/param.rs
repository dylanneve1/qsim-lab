//! Parameterised circuits: gates whose angles are affine functions of a
//! parameter vector `θ`.
//!
//! An [`Angle`] is `c0 + Σ_j c_j θ_{i_j}`. That covers the sweep workloads
//! the graph compiler targets: QAOA (`γ_l · w_e`, `β_l`), hardware-efficient
//! VQE ansätze (one free angle per gate), Trotter steps (`J · dt` with `dt`
//! a parameter) and parameter-shift gradients (`θ_i ± π/2`).
//!
//! [`ParamCircuit::bind`] turns a parameter vector into an ordinary
//! [`Circuit`]; it is the reference every compiled path is checked against
//! and the "recompile per bind" baseline of the benchmarks.

use crate::circuit::{Circuit, SimError};
use crate::gate::Gate;

/// `c0 + Σ coef · θ[index]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Angle {
    pub c0: f64,
    pub terms: Vec<(u32, f64)>,
}

impl Angle {
    /// A constant angle.
    pub fn constant(v: f64) -> Self {
        Angle {
            c0: v,
            terms: Vec::new(),
        }
    }

    /// `θ[i]`.
    pub fn param(i: usize) -> Self {
        Angle::scaled(i, 1.0)
    }

    /// `coef · θ[i]`.
    pub fn scaled(i: usize, coef: f64) -> Self {
        Angle {
            c0: 0.0,
            terms: vec![(i as u32, coef)],
        }
    }

    /// `self + other` (terms on the same parameter are merged).
    pub fn plus(&self, other: &Angle) -> Angle {
        let mut a = self.clone();
        a.c0 += other.c0;
        for &(i, c) in &other.terms {
            match a.terms.iter_mut().find(|t| t.0 == i) {
                Some(t) => t.1 += c,
                None => a.terms.push((i, c)),
            }
        }
        a.terms.retain(|t| t.1 != 0.0);
        a
    }

    /// `k · self`.
    pub fn times(&self, k: f64) -> Angle {
        Angle {
            c0: self.c0 * k,
            terms: self
                .terms
                .iter()
                .map(|&(i, c)| (i, c * k))
                .filter(|t| t.1 != 0.0)
                .collect(),
        }
    }

    /// No parameter dependence.
    pub fn is_const(&self) -> bool {
        self.terms.is_empty()
    }

    /// Value at `params`.
    #[inline]
    pub fn eval(&self, params: &[f64]) -> f64 {
        self.terms
            .iter()
            .fold(self.c0, |s, &(i, c)| s + c * params[i as usize])
    }

    /// Parameters this angle depends on.
    pub fn params(&self) -> impl Iterator<Item = usize> + '_ {
        self.terms.iter().map(|t| t.0 as usize)
    }
}

impl From<f64> for Angle {
    fn from(v: f64) -> Self {
        Angle::constant(v)
    }
}

/// One op of a [`ParamCircuit`].
#[derive(Clone, Debug, PartialEq)]
pub enum POp {
    /// A gate with fixed angles.
    Fixed(Gate),
    Rx(usize, Angle),
    Ry(usize, Angle),
    Rz(usize, Angle),
    /// `diag(1, e^{iθ})`.
    Phase(usize, Angle),
    /// `diag(1, 1, 1, e^{iθ})`.
    CPhase(usize, usize, Angle),
    /// `exp(-i θ/2 Z⊗Z)`; bound as `CNOT · Rz(θ) · CNOT` (exactly equal,
    /// global phase included).
    Rzz(usize, usize, Angle),
    /// `exp(-i θ/2 X⊗X)`; bound as `H⊗H · Rzz(θ) · H⊗H`.
    Rxx(usize, usize, Angle),
    /// `U(θ, φ, λ)`.
    U(usize, Angle, Angle, Angle),
}

impl POp {
    /// Qubits in argument order.
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            POp::Fixed(g) => g.qubits(),
            POp::Rx(q, _) | POp::Ry(q, _) | POp::Rz(q, _) | POp::Phase(q, _) | POp::U(q, ..) => {
                vec![*q]
            }
            POp::CPhase(a, b, _) | POp::Rzz(a, b, _) | POp::Rxx(a, b, _) => vec![*a, *b],
        }
    }

    /// The angles of this op (empty for [`POp::Fixed`]).
    pub fn angles(&self) -> Vec<&Angle> {
        match self {
            POp::Fixed(_) => vec![],
            POp::Rx(_, a)
            | POp::Ry(_, a)
            | POp::Rz(_, a)
            | POp::Phase(_, a)
            | POp::CPhase(_, _, a)
            | POp::Rzz(_, _, a)
            | POp::Rxx(_, _, a) => vec![a],
            POp::U(_, a, b, c) => vec![a, b, c],
        }
    }

    /// Does the op depend on a parameter?
    pub fn is_param(&self) -> bool {
        self.angles().iter().any(|a| !a.is_const())
    }

    /// Diagonal in the computational basis for every parameter value.
    pub fn is_diagonal(&self) -> bool {
        match self {
            POp::Fixed(g) => match g {
                Gate::Cz(..) | Gate::CPhase(..) => true,
                g if g.arity() == 1 => g.diagonal_1q().is_some(),
                _ => false,
            },
            POp::Rz(..) | POp::Phase(..) | POp::CPhase(..) | POp::Rzz(..) => true,
            _ => false,
        }
    }

    /// Same op on relabelled qubits.
    pub fn map_qubits(&self, f: impl Fn(usize) -> usize) -> POp {
        match self {
            POp::Fixed(g) => POp::Fixed(crate::hsf::map_gate(*g, f)),
            POp::Rx(q, a) => POp::Rx(f(*q), a.clone()),
            POp::Ry(q, a) => POp::Ry(f(*q), a.clone()),
            POp::Rz(q, a) => POp::Rz(f(*q), a.clone()),
            POp::Phase(q, a) => POp::Phase(f(*q), a.clone()),
            POp::CPhase(x, y, a) => POp::CPhase(f(*x), f(*y), a.clone()),
            POp::Rzz(x, y, a) => POp::Rzz(f(*x), f(*y), a.clone()),
            POp::Rxx(x, y, a) => POp::Rxx(f(*x), f(*y), a.clone()),
            POp::U(q, a, b, c) => POp::U(f(*q), a.clone(), b.clone(), c.clone()),
        }
    }

    /// The fixed gates this op equals at `params` (appended to `out`).
    pub fn bind_into(&self, params: &[f64], out: &mut Vec<Gate>) {
        match self {
            POp::Fixed(g) => out.push(*g),
            POp::Rx(q, a) => out.push(Gate::Rx(*q, a.eval(params))),
            POp::Ry(q, a) => out.push(Gate::Ry(*q, a.eval(params))),
            POp::Rz(q, a) => out.push(Gate::Rz(*q, a.eval(params))),
            POp::Phase(q, a) => out.push(Gate::Phase(*q, a.eval(params))),
            POp::CPhase(x, y, a) => out.push(Gate::CPhase(*x, *y, a.eval(params))),
            POp::Rzz(x, y, a) => {
                out.push(Gate::Cnot(*x, *y));
                out.push(Gate::Rz(*y, a.eval(params)));
                out.push(Gate::Cnot(*x, *y));
            }
            POp::Rxx(x, y, a) => {
                out.push(Gate::H(*x));
                out.push(Gate::H(*y));
                out.push(Gate::Cnot(*x, *y));
                out.push(Gate::Rz(*y, a.eval(params)));
                out.push(Gate::Cnot(*x, *y));
                out.push(Gate::H(*x));
                out.push(Gate::H(*y));
            }
            POp::U(q, a, b, c) => {
                out.push(Gate::U(*q, a.eval(params), b.eval(params), c.eval(params)))
            }
        }
    }
}

/// A unitary circuit with parameterised angles.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParamCircuit {
    pub num_qubits: usize,
    pub num_params: usize,
    pub ops: Vec<POp>,
}

impl ParamCircuit {
    pub fn new(num_qubits: usize, num_params: usize) -> Self {
        ParamCircuit {
            num_qubits,
            num_params,
            ops: Vec::new(),
        }
    }

    /// Wraps a fixed circuit (gates only).
    pub fn from_circuit(c: &Circuit) -> Result<Self, SimError> {
        let mut p = ParamCircuit::new(c.num_qubits, 0);
        for op in &c.ops {
            match op {
                crate::circuit::Op::Gate(g) => p.ops.push(POp::Fixed(*g)),
                _ => {
                    return Err(SimError::NotSupported {
                        what: "graph compiler: only unitary gate sequences",
                    })
                }
            }
        }
        Ok(p)
    }

    /// Appends an op (checks qubits and parameter indices).
    pub fn push(&mut self, op: POp) -> &mut Self {
        let qs = op.qubits();
        for (i, &q) in qs.iter().enumerate() {
            assert!(q < self.num_qubits, "qubit {q} out of range");
            assert!(!qs[..i].contains(&q), "repeated qubit {q}");
        }
        for a in op.angles() {
            for p in a.params() {
                assert!(p < self.num_params, "parameter {p} out of range");
            }
        }
        self.ops.push(op);
        self
    }

    pub fn gate(&mut self, g: Gate) -> &mut Self {
        self.push(POp::Fixed(g))
    }
    pub fn rx(&mut self, q: usize, a: impl Into<Angle>) -> &mut Self {
        self.push(POp::Rx(q, a.into()))
    }
    pub fn ry(&mut self, q: usize, a: impl Into<Angle>) -> &mut Self {
        self.push(POp::Ry(q, a.into()))
    }
    pub fn rz(&mut self, q: usize, a: impl Into<Angle>) -> &mut Self {
        self.push(POp::Rz(q, a.into()))
    }
    pub fn phase(&mut self, q: usize, a: impl Into<Angle>) -> &mut Self {
        self.push(POp::Phase(q, a.into()))
    }
    pub fn cphase(&mut self, x: usize, y: usize, a: impl Into<Angle>) -> &mut Self {
        self.push(POp::CPhase(x, y, a.into()))
    }
    pub fn rzz(&mut self, x: usize, y: usize, a: impl Into<Angle>) -> &mut Self {
        self.push(POp::Rzz(x, y, a.into()))
    }
    pub fn rxx(&mut self, x: usize, y: usize, a: impl Into<Angle>) -> &mut Self {
        self.push(POp::Rxx(x, y, a.into()))
    }

    /// The fixed circuit at `params`.
    pub fn bind(&self, params: &[f64]) -> Result<Circuit, SimError> {
        if params.len() != self.num_params {
            return Err(SimError::NotSupported {
                what: "ParamCircuit::bind: wrong number of parameters",
            });
        }
        let mut gates = Vec::with_capacity(self.ops.len());
        for op in &self.ops {
            op.bind_into(params, &mut gates);
        }
        let mut c = Circuit::new(self.num_qubits);
        for g in gates {
            c.gate(g);
        }
        Ok(c)
    }

    /// Number of ops that depend on a parameter.
    pub fn num_param_ops(&self) -> usize {
        self.ops.iter().filter(|o| o.is_param()).count()
    }
}
