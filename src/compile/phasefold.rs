//! Phase folding (Amy, Maslov & Mosca, "Polynomial-time T-depth
//! optimization of Clifford+T circuits via matroid partitioning", 2014).
//!
//! Between non-linear gates a circuit over `{CNOT, X, SWAP, Z-rotations}`
//! maps `|x>` to `e^{i f(x)} |Ax ⊕ b>`, with `f` a *phase polynomial*: a sum
//! of `θ_p · (p·x)` over parities `p`. Every wire carries an affine parity
//! of the *variables* `x`; two Z-rotations that act on the same parity can
//! be merged into one (`T·T = S`, `T·T† = I`, `Rz(a)·Rz(b) = Rz(a+b)`),
//! wherever they sit in the circuit, because both are diagonal in the
//! same basis and every gate in between only permutes that basis.
//!
//! Gates that are not affine on the computational basis (H, Rx, Ry, U,
//! SX, Toffoli target, measurement, …) give the qubits they change a
//! *fresh* variable. Rotations are then only merged when their parity
//! does not involve any fresh variable introduced in between. Soundness:
//! conjugating the later `Z`-string back through the gates between the
//! two rotations, it has no support on a qubit with a fresh variable
//! exactly when its parity does not contain that variable, so it commutes
//! with the gate that introduced it.
//!
//! The pass keeps every non-rotation gate in place and puts each merged
//! rotation at its *first* occurrence (the earlier position is always
//! reachable). Merged angles are tracked exactly in units of `π/4` plus
//! a floating residual (generic `Rz` / `Phase` angles); a merged rotation
//! that is an odd multiple of `π/4` is emitted as one `T`/`T†` (plus a
//! Clifford `S`/`Z`), a multiple of `π/2` as Cliffords, and anything else
//! as one `Phase` gate. The non-Clifford count of the output is never
//! larger than the input's, and the global phase is tracked:
//! `U_original = e^{i global_phase} U_circuit`.

use super::Optimized;
use crate::circuit::{Circuit, Op};
use crate::gate::Gate;
use std::collections::HashMap;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

/// Statistics of a phase-folding run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhaseFoldStats {
    /// Z-rotation gates seen.
    pub rotations_in: usize,
    /// Distinct parities that kept a non-trivial rotation.
    pub rotations_out: usize,
    /// Variables introduced (qubits + fresh ones after non-affine gates).
    pub variables: usize,
}

/// Tolerance for treating an angle as an exact multiple of `π/4` (the same
/// as the DAG peephole's).
const EPS: f64 = 1e-12;

struct Term {
    wire: usize,
    /// The wire's constant offset at the first occurrence.
    c1: bool,
    units: i64,
    resid: f64,
}

enum Slot {
    Keep(Op),
    Rot(usize),
}

/// `(units of π/4, residual)` with `θ = units·π/4 + residual`.
fn split_angle(theta: f64) -> (i64, f64) {
    let k = (theta / FRAC_PI_4).round();
    if k.abs() < 1e12 && (theta - k * FRAC_PI_4).abs() < EPS {
        (k as i64, 0.0)
    } else {
        (0, theta)
    }
}

/// The Z-rotation part of a gate: `(qubit, θ, phase)` with
/// `gate = e^{i phase} · diag(1, e^{iθ})`.
fn rotation_of(g: &Gate) -> Option<(usize, f64, f64)> {
    use Gate::*;
    Some(match *g {
        T(q) => (q, FRAC_PI_4, 0.0),
        Tdg(q) => (q, -FRAC_PI_4, 0.0),
        S(q) => (q, FRAC_PI_2, 0.0),
        Sdg(q) => (q, -FRAC_PI_2, 0.0),
        Z(q) => (q, std::f64::consts::PI, 0.0),
        Phase(q, t) => (q, t, 0.0),
        Rz(q, t) => (q, t, -t / 2.0),
        _ => return None,
    })
}

/// Gates after which the qubits in the returned list hold a fresh variable
/// (they are not affine on the computational basis, or not unitary).
fn fresh_wires(op: &Op) -> Vec<usize> {
    use Gate::*;
    match *op {
        Op::Gate(g) => match g {
            H(q) | Sx(q) | Sxdg(q) | Rx(q, _) | Ry(q, _) | U(q, ..) => vec![q],
            ISwap(a, b) | ISwapdg(a, b) => vec![a, b],
            Ccx(_, _, t) => vec![t],
            // affine, diagonal or handled explicitly
            _ => vec![],
        },
        Op::ClassicControlled { gate, .. } => gate.qubits(),
        Op::Measure(q)
        | Op::Reset(q)
        | Op::XFlip(q, _)
        | Op::YFlip(q, _)
        | Op::ZFlip(q, _)
        | Op::Depolarize1q(q, _) => vec![q],
        Op::Depolarize2q(a, b, _) => vec![a, b],
    }
}

/// Phase folding with default settings.
pub fn phase_fold(c: &Circuit) -> Optimized {
    phase_fold_with_stats(c).0
}

/// Phase folding, with statistics.
pub fn phase_fold_with_stats(c: &Circuit) -> (Optimized, PhaseFoldStats) {
    let n = c.num_qubits;
    // Number of variables: one per qubit plus one per fresh event.
    let fresh_events: usize = c.ops.iter().map(|op| fresh_wires(op).len()).sum();
    let nv = n + fresh_events;
    let w = nv.div_ceil(64).max(1);

    let mut par: Vec<Vec<u64>> = (0..n)
        .map(|q| {
            let mut v = vec![0u64; w];
            v[q / 64] |= 1 << (q % 64);
            v
        })
        .collect();
    let mut cst = vec![false; n];
    let mut next_var = n;

    let mut slots: Vec<Slot> = Vec::with_capacity(c.ops.len());
    let mut terms: Vec<Term> = Vec::new();
    let mut index: HashMap<Vec<u64>, usize> = HashMap::new();
    let mut phase = 0.0f64;
    let mut stats = PhaseFoldStats::default();

    let mut add_rot = |q: usize,
                       theta: f64,
                       par: &[Vec<u64>],
                       cst: &[bool],
                       slots: &mut Vec<Slot>,
                       terms: &mut Vec<Term>,
                       phase: &mut f64| {
        stats.rotations_in += 1;
        let key = &par[q];
        // diag(1, e^{iθ}) on value v = p·x ⊕ c
        let th = if cst[q] {
            *phase += theta; // e^{iθ(1 - px)} = e^{iθ} e^{-iθ px}
            -theta
        } else {
            theta
        };
        if key.iter().all(|&x| x == 0) {
            // parity of nothing: the value is the constant c; handled
            // above (th is irrelevant, phase already added when c = 1)
            return;
        }
        let (u, r) = split_angle(th);
        let id = match index.get(key.as_slice()) {
            Some(&id) => id,
            None => {
                let id = terms.len();
                terms.push(Term {
                    wire: q,
                    c1: cst[q],
                    units: 0,
                    resid: 0.0,
                });
                index.insert(key.clone(), id);
                slots.push(Slot::Rot(id));
                id
            }
        };
        terms[id].units += u;
        terms[id].resid += r;
    };

    for op in &c.ops {
        match *op {
            Op::Gate(g) => {
                if let Some((q, th, ph)) = rotation_of(&g) {
                    phase += ph;
                    add_rot(q, th, &par, &cst, &mut slots, &mut terms, &mut phase);
                    continue;
                }
                match g {
                    Gate::I(_) => {}
                    Gate::Y(q) => {
                        // Y = i · X · Z (Z first)
                        phase += FRAC_PI_2;
                        add_rot(
                            q,
                            std::f64::consts::PI,
                            &par,
                            &cst,
                            &mut slots,
                            &mut terms,
                            &mut phase,
                        );
                        cst[q] ^= true;
                        slots.push(Slot::Keep(Op::Gate(Gate::X(q))));
                    }
                    Gate::X(q) => {
                        cst[q] ^= true;
                        slots.push(Slot::Keep(*op));
                    }
                    Gate::Cnot(a, b) => {
                        let src = par[a].clone();
                        for (d, s) in par[b].iter_mut().zip(src) {
                            *d ^= s;
                        }
                        cst[b] ^= cst[a];
                        slots.push(Slot::Keep(*op));
                    }
                    Gate::Swap(a, b) => {
                        par.swap(a, b);
                        cst.swap(a, b);
                        slots.push(Slot::Keep(*op));
                    }
                    _ => {
                        for q in fresh_wires(op) {
                            par[q].iter_mut().for_each(|x| *x = 0);
                            par[q][next_var / 64] |= 1 << (next_var % 64);
                            cst[q] = false;
                            next_var += 1;
                        }
                        slots.push(Slot::Keep(*op));
                    }
                }
            }
            _ => {
                for q in fresh_wires(op) {
                    par[q].iter_mut().for_each(|x| *x = 0);
                    par[q][next_var / 64] |= 1 << (next_var % 64);
                    cst[q] = false;
                    next_var += 1;
                }
                slots.push(Slot::Keep(*op));
            }
        }
    }
    stats.variables = next_var;

    // Emission.
    let mut out = Circuit::new(n);
    for s in slots {
        match s {
            Slot::Keep(op) => out.ops.push(op),
            Slot::Rot(id) => {
                let t = &terms[id];
                let mut units = t.units.rem_euclid(8);
                let mut resid = t.resid;
                if resid.abs() < EPS {
                    resid = 0.0;
                }
                // Total angle A (mod 2π) of e^{iA·(p·x)}.
                if units == 0 && resid == 0.0 {
                    continue;
                }
                if t.c1 {
                    // e^{iA px} = e^{iA} · Phase(-A) on a wire holding ¬px
                    phase += units as f64 * FRAC_PI_4 + resid;
                    units = (8 - units).rem_euclid(8);
                    resid = -resid;
                }
                let q = t.wire;
                if resid != 0.0 {
                    out.ops
                        .push(Op::Gate(Gate::Phase(q, units as f64 * FRAC_PI_4 + resid)));
                } else {
                    let gs: &[Gate] = match units {
                        0 => &[],
                        1 => &[Gate::T(q)],
                        2 => &[Gate::S(q)],
                        3 => &[Gate::S(q), Gate::T(q)],
                        4 => &[Gate::Z(q)],
                        5 => &[Gate::Sdg(q), Gate::Tdg(q)],
                        6 => &[Gate::Sdg(q)],
                        _ => &[Gate::Tdg(q)],
                    };
                    out.ops.extend(gs.iter().map(|&g| Op::Gate(g)));
                }
                stats.rotations_out += 1;
            }
        }
    }
    (
        Optimized {
            circuit: out,
            global_phase: phase,
        },
        stats,
    )
}
