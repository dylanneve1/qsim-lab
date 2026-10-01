//! Dense exact fast paths for repeated unitary blocks.
//!
//! For a block `B` repeated `r` times on an `n`-qubit state vector
//! ([`run_dense`] picks per repeat):
//!
//! 1. **diagonal** `B` (only `Z S T Rz Phase CZ CPhase`): the phase
//!    polynomial of `B^r` is `r` times that of `B`: one diagonal pass;
//! 2. `B` on `k <= max_small_k` qubits: build the `2^k` unitary `U`, take
//!    `U^r` by repeated squaring, apply it once (a cost model decides);
//! 3. otherwise: lower/fuse/plan/prepare the blocked executor plan of `B`
//!    once and run it `r` times (saves compile time, not run time).

use super::cliff::{compact, remap_gate};
use super::{op_angles, op_with_angles, Node, Program};
use crate::blocked::{lower_gates, BlockConfig, CompiledKOps};
use crate::circuit::{check_gate, Op, SimError, Simulator};
use crate::gate::Gate;
use crate::statevector::{Real, StateVector};
use num_complex::{Complex, Complex64};
use std::collections::BTreeMap;
use std::f64::consts::PI;

/// Which fast paths [`run_dense`] may use.
#[derive(Clone, Debug)]
pub struct ExecOptions {
    pub diag: bool,
    pub small_unitary: bool,
    /// Largest block support (qubits) for the `2^k` unitary power.
    pub max_small_k: usize,
    pub reuse_plan: bool,
}

impl Default for ExecOptions {
    fn default() -> Self {
        ExecOptions {
            diag: true,
            small_unitary: true,
            max_small_k: 8,
            reuse_plan: true,
        }
    }
}

/// What [`run_dense`] did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExecStats {
    pub diag_blocks: usize,
    pub small_blocks: usize,
    pub reuse_blocks: usize,
    /// Repeats run copy by copy through the plain executor.
    pub plain_repeats: usize,
}

/// `(a * r) mod 2π` with the product and the reduction done in
/// double-double (fma), so the result is accurate to ~1 ulp of π even for
/// `a * r ~ 1e6`.
pub fn scaled_angle(a: f64, r: f64) -> f64 {
    const HI: f64 = std::f64::consts::TAU;
    const LO: f64 = 2.449_293_598_294_706_4e-16;
    let p = a * r;
    let e = a.mul_add(r, -p);
    let k = (p / HI).round();
    let t = k.mul_add(-HI, p);
    (t - k * LO) + e
}

/// A diagonal gate's contribution: phase polynomial
/// `φ(x) = g0 + Σ a_q x_q + Σ b_pq x_p x_q`.
#[derive(Clone, Debug, Default)]
pub struct DiagPoly {
    pub g0: f64,
    pub a: BTreeMap<usize, f64>,
    pub b: BTreeMap<(usize, usize), f64>,
}

impl DiagPoly {
    /// Adds `g`; false (and no change) if `g` is not diagonal.
    pub fn add(&mut self, g: &Gate) -> bool {
        use Gate::*;
        let mut a = |q: usize, v: f64| *self.a.entry(q).or_insert(0.0) += v;
        match *g {
            I(_) => {}
            Z(q) => a(q, PI),
            S(q) => a(q, PI / 2.0),
            Sdg(q) => a(q, -PI / 2.0),
            T(q) => a(q, PI / 4.0),
            Tdg(q) => a(q, -PI / 4.0),
            Phase(q, t) => a(q, t),
            Rz(q, t) => {
                self.g0 -= t / 2.0;
                *self.a.entry(q).or_insert(0.0) += t;
            }
            Cz(p, q) => {
                *self.b.entry((p.min(q), p.max(q))).or_insert(0.0) += PI;
            }
            CPhase(p, q, t) => {
                *self.b.entry((p.min(q), p.max(q))).or_insert(0.0) += t;
            }
            _ => return false,
        }
        true
    }

    /// `(gates, global phase)` of `self` scaled by `r` (so `B^r` for a
    /// diagonal `B`). Gates are `Phase`/`CPhase`.
    pub fn scaled_gates(&self, r: f64) -> (Vec<Gate>, f64) {
        let mut out = Vec::new();
        for (&q, &v) in &self.a {
            let t = scaled_angle(v, r);
            if t != 0.0 {
                out.push(Gate::Phase(q, t));
            }
        }
        for (&(p, q), &v) in &self.b {
            let t = scaled_angle(v, r);
            if t != 0.0 {
                out.push(Gate::CPhase(p, q, t));
            }
        }
        (out, scaled_angle(self.g0, r))
    }
}

/// `B^r` for a diagonal gate list as `(gates, global phase)`, exactly equal
/// as a unitary. `None` if some gate is not diagonal.
pub fn diag_power(body: &[Gate], r: u64) -> Option<(Vec<Gate>, f64)> {
    let mut d = DiagPoly::default();
    for g in body {
        if !d.add(g) {
            return None;
        }
    }
    Some(d.scaled_gates(r as f64))
}

// ------------------------------------------------------- 2^k unitaries

/// The `2^k × 2^k` unitary of `gates` (qubits `0..k`), row-major.
pub fn block_unitary(k: usize, gates: &[Gate]) -> Vec<Complex64> {
    let d = 1usize << k;
    let mut amps = vec![Complex64::new(0.0, 0.0); d * d];
    for c in 0..d {
        amps[(c << k) | c] = Complex64::new(1.0, 0.0);
    }
    let mut sv = StateVector::<f64>::from_amplitudes(amps);
    sv.apply_gates(gates).expect("gates on 0..k");
    let a = sv.amplitudes();
    // amps[(c << k) | row] = U[row][c]
    let mut u = vec![Complex64::new(0.0, 0.0); d * d];
    for r in 0..d {
        for c in 0..d {
            u[r * d + c] = a[(c << k) | r];
        }
    }
    u
}

fn mat_mul(a: &[Complex64], b: &[Complex64], d: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); d * d];
    for i in 0..d {
        for k in 0..d {
            let aik = a[i * d + k];
            if aik.re == 0.0 && aik.im == 0.0 {
                continue;
            }
            let (row, brow) = (&mut out[i * d..(i + 1) * d], &b[k * d..(k + 1) * d]);
            for j in 0..d {
                row[j] += aik * brow[j];
            }
        }
    }
    out
}

/// `U^e` by repeated squaring.
pub fn mat_pow(u: &[Complex64], d: usize, mut e: u64) -> Vec<Complex64> {
    let mut result = vec![Complex64::new(0.0, 0.0); d * d];
    for i in 0..d {
        result[i * d + i] = Complex64::new(1.0, 0.0);
    }
    let mut base = u.to_vec();
    while e > 0 {
        if e & 1 == 1 {
            result = mat_mul(&result, &base, d);
        }
        e >>= 1;
        if e > 0 {
            base = mat_mul(&base, &base, d);
        }
    }
    result
}

/// Applies a `2^k × 2^k` matrix (local bit `t` = qubit `qs[t]`) to a state.
pub fn apply_matrix<T: Real>(sv: &mut StateVector<T>, qs: &[usize], u: &[Complex64]) {
    let k = qs.len();
    let d = 1usize << k;
    let n = sv.num_qubits();
    let off: Vec<usize> = (0..d)
        .map(|j| {
            (0..k)
                .filter(|t| j >> t & 1 == 1)
                .map(|t| 1usize << qs[t])
                .sum()
        })
        .collect();
    let mut sorted = qs.to_vec();
    sorted.sort_unstable();
    let um: Vec<Complex<T>> = u
        .iter()
        .map(|z| Complex::new(T::from_f64(z.re), T::from_f64(z.im)))
        .collect();
    let amps = sv.amplitudes_mut();
    let mut v = vec![Complex::<T>::new(T::zero(), T::zero()); d];
    for b in 0..1usize << (n - k) {
        let mut base = b;
        for &q in &sorted {
            base = ((base >> q) << (q + 1)) | (base & ((1usize << q) - 1));
        }
        for j in 0..d {
            v[j] = amps[base | off[j]];
        }
        for i in 0..d {
            let mut s = Complex::<T>::new(T::zero(), T::zero());
            for j in 0..d {
                s = s + um[i * d + j] * v[j];
            }
            amps[base | off[i]] = s;
        }
    }
}

// ------------------------------------------------------------ executor

fn gates_of(ops: &[Op]) -> Result<Vec<Gate>, SimError> {
    ops.iter()
        .map(|op| match op {
            Op::Gate(g) => Ok(*g),
            _ => Err(SimError::NotSupported {
                what: "repeat::run_dense needs a unitary program",
            }),
        })
        .collect()
}

/// One copy of the body as gates, expanding nested repeats up to `limit`
/// gates. `None` if too long or not unitary.
fn flat_gates(nodes: &[Node], limit: usize) -> Option<Vec<Gate>> {
    fn go(nodes: &[Node], out: &mut Vec<Gate>, limit: usize) -> bool {
        for n in nodes {
            match n {
                Node::Ops(o) => {
                    for op in o {
                        match op {
                            Op::Gate(g) => out.push(*g),
                            _ => return false,
                        }
                    }
                }
                Node::Repeat { body, reps } => {
                    let mut one = Vec::new();
                    if !go(body, &mut one, limit) {
                        return false;
                    }
                    if one.len().saturating_mul(*reps) + out.len() > limit {
                        return false;
                    }
                    for _ in 0..*reps {
                        out.extend_from_slice(&one);
                    }
                }
                Node::Param {
                    shape,
                    reps,
                    angles,
                } => {
                    if shape.len() * reps + out.len() > limit {
                        return false;
                    }
                    for ang in angles.iter().take(*reps) {
                        let mut it = ang.iter().copied();
                        for op in shape {
                            let na = op_angles(op).len();
                            let a: Vec<f64> = it.by_ref().take(na).collect();
                            match op_with_angles(op, &a) {
                                Op::Gate(g) => out.push(g),
                                _ => return false,
                            }
                        }
                    }
                }
            }
        }
        true
    }
    let mut out = Vec::new();
    go(nodes, &mut out, limit).then_some(out)
}

struct Exec<'a, T: Real> {
    sv: &'a mut StateVector<T>,
    opts: &'a ExecOptions,
    stats: ExecStats,
    phase: f64,
}

impl<T: Real> Exec<'_, T> {
    fn apply(&mut self, gates: &[Gate]) -> Result<(), SimError> {
        for g in gates {
            check_gate(g, self.sv.num_qubits())?;
        }
        self.sv.apply_gates(gates)
    }

    fn nodes(&mut self, nodes: &[Node]) -> Result<(), SimError> {
        for n in nodes {
            match n {
                Node::Ops(o) => {
                    let g = gates_of(o)?;
                    self.apply(&g)?;
                }
                Node::Repeat { body, reps } => self.repeat(body, *reps)?,
                Node::Param {
                    shape,
                    reps,
                    angles,
                } => self.param(shape, *reps, angles)?,
            }
        }
        Ok(())
    }

    fn param(&mut self, shape: &[Op], reps: usize, angles: &[Vec<f64>]) -> Result<(), SimError> {
        let copy = |k: usize| -> Vec<Op> {
            let mut it = angles[k].iter().copied();
            shape
                .iter()
                .map(|op| {
                    let na = op_angles(op).len();
                    let a: Vec<f64> = it.by_ref().take(na).collect();
                    op_with_angles(op, &a)
                })
                .collect()
        };
        // All copies diagonal: the angles simply add.
        if self.opts.diag {
            let mut d = DiagPoly::default();
            let mut ok = true;
            'outer: for k in 0..reps {
                for op in copy(k) {
                    match op {
                        Op::Gate(g) if d.add(&g) => {}
                        _ => {
                            ok = false;
                            break 'outer;
                        }
                    }
                }
            }
            if ok {
                let (g, ph) = d.scaled_gates(1.0);
                self.phase += ph;
                self.stats.diag_blocks += 1;
                return self.apply(&g);
            }
        }
        // One batch per copy keeps cross-copy fusion; angles differ, so
        // nothing can be reused.
        let mut all = Vec::new();
        for k in 0..reps {
            all.extend(gates_of(&copy(k))?);
        }
        self.stats.plain_repeats += 1;
        self.apply(&all)
    }

    fn repeat(&mut self, body: &[Node], reps: usize) -> Result<(), SimError> {
        let n = self.sv.num_qubits();
        let Some(g) = flat_gates(body, 1 << 20) else {
            // too long to flatten: run the copies, each through the tree
            self.stats.plain_repeats += 1;
            for _ in 0..reps {
                self.nodes(body)?;
            }
            return Ok(());
        };
        if g.is_empty() || reps == 0 {
            return Ok(());
        }
        if self.opts.diag {
            if let Some((dg, ph)) = diag_power(&g, reps as u64) {
                self.phase += ph;
                self.stats.diag_blocks += 1;
                return self.apply(&dg);
            }
        }
        let (qs, local) = compact(&g);
        let k = qs.len();
        if self.opts.small_unitary && k <= self.opts.max_small_k && reps >= 2 {
            let kk = k as f64;
            let plain = reps as f64 * g.len() as f64 * (n as f64).exp2();
            let squarings = 2.0 * (reps as f64).log2().ceil().max(1.0);
            let small = g.len() as f64 * (2.0 * kk).exp2()
                + squarings * 4.0 * (3.0 * kk).exp2()
                + 2.0 * (n as f64 + kk).exp2();
            if small < plain {
                let u = block_unitary(k, &local);
                let up = mat_pow(&u, 1 << k, reps as u64);
                apply_matrix(self.sv, &qs, &up);
                self.stats.small_blocks += 1;
                return Ok(());
            }
        }
        if self.opts.reuse_plan && reps >= 2 {
            for gate in &g {
                check_gate(gate, n)?;
            }
            let cfg = BlockConfig::default();
            let plan: CompiledKOps<T> =
                StateVector::<T>::compile_kops(n, &lower_gates(g.iter()), &cfg);
            for _ in 0..reps {
                self.sv.run_compiled(&plan);
            }
            self.stats.reuse_blocks += 1;
            return Ok(());
        }
        self.stats.plain_repeats += 1;
        for _ in 0..reps {
            self.apply(&g)?;
        }
        let _ = remap_gate;
        Ok(())
    }
}

/// Runs a unitary program on `sv` with the fast paths in `opts`. Returns
/// what was used. The state ends up equal (up to rounding, global phase
/// included) to gate-by-gate application of [`Program::to_circuit`].
pub fn run_dense<T: Real>(
    p: &Program,
    sv: &mut StateVector<T>,
    opts: &ExecOptions,
) -> Result<ExecStats, SimError> {
    let mut e = Exec {
        sv,
        opts,
        stats: ExecStats::default(),
        phase: 0.0,
    };
    e.nodes(&p.nodes)?;
    if e.phase != 0.0 {
        let f = Complex::new(T::from_f64(e.phase.cos()), T::from_f64(e.phase.sin()));
        for a in e.sv.amplitudes_mut() {
            *a = *a * f;
        }
    }
    Ok(e.stats)
}
