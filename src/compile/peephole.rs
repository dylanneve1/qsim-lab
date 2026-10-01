//! Commutation-aware peephole optimisation.
//!
//! Gates are appended one at a time to an output list. Each new gate walks
//! backwards over the earlier gates that share a qubit with it; it may pass
//! any gate it commutes with (see [`Axis`](super::Axis)) and stops at the
//! first one it does not. If on the way it meets a gate of the same family
//! on the same qubits, the two are merged:
//!
//! * self-inverse pairs cancel: `H H`, `X X`, `Y Y`, `CNOT CNOT`,
//!   `SWAP SWAP`, `CCX CCX`, `CZ CZ`;
//! * Z rotations merge: `Z, S, S†, T, T†, Phase(θ), Rz(θ)` are all
//!   `Phase(φ)` up to a global phase, so e.g. `T T -> S`, `S S† -> I`,
//!   `Rz(a) Rz(b) -> Rz(a+b)`;
//! * X and Y rotations merge the same way (`X Rx(θ) -> Rx(θ+π)` up to phase);
//! * controlled phases on the same pair merge (`CZ CPhase(θ) -> CPhase(θ+π)`).
//!
//! A merged gate is re-inserted at the earlier position and may cascade
//! (`S T T -> S S -> Z`). Rotations by multiples of `π/4` are rewritten to
//! named gates, so a circuit whose rotations happen to be Clifford becomes
//! visibly Clifford (and eligible for the tableau). Identity rotations are
//! removed. The global phase every rewrite introduces is accumulated in
//! [`Optimized::global_phase`], so amplitudes are reproduced exactly.
//!
//! Measurements are `Z`-type barriers: diagonal gates commute with a
//! computational-basis measurement (the projectors are diagonal), so they
//! may move across it, but nothing merges with it.
//!
//! Resets, noise channels and classically controlled gates are full
//! barriers on their qubits: nothing commutes past them and nothing merges
//! with them (they are copied through unchanged). Some of them do commute
//! with some gates (`ZFlip` with diagonal gates, say), but treating them as
//! opaque keeps the pass obviously correct for non-unitary circuits.

use super::{axis_on, op_qubits, qubits_of, Axis};
use crate::circuit::{Circuit, Op};
use crate::gate::Gate;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

/// Angles closer than this to a special value are treated as equal to it.
/// Snapping a rotation by `ε` changes amplitudes by at most `ε / 2`.
const ANGLE_EPS: f64 = 1e-13;

/// Tuning knobs for [`optimize`].
#[derive(Clone, Copy, Debug)]
pub struct PeepholeOptions {
    /// Maximum number of earlier gates (sharing a qubit) a new gate may be
    /// commuted past while looking for a merge partner.
    pub window: usize,
    /// Merge rotations and cancel inverse pairs (the main pass).
    pub merge: bool,
}

impl Default for PeepholeOptions {
    fn default() -> Self {
        PeepholeOptions {
            window: 1024,
            merge: true,
        }
    }
}

/// An optimised circuit: `U_original = e^{i global_phase} U_circuit`.
#[derive(Clone, Debug, PartialEq)]
pub struct Optimized {
    pub circuit: Circuit,
    pub global_phase: f64,
}

/// Runs the peephole pass with default options.
pub fn optimize(c: &Circuit) -> Optimized {
    optimize_with(c, PeepholeOptions::default())
}

/// Runs the peephole pass.
pub fn optimize_with(c: &Circuit, opts: PeepholeOptions) -> Optimized {
    let mut b = Builder {
        ops: Vec::with_capacity(c.ops.len()),
        per_q: vec![Vec::new(); c.num_qubits],
        phase: 0.0,
        opts,
    };
    for op in &c.ops {
        match *op {
            Op::Gate(g) => {
                let Some(g) = canonical(g, &mut b.phase) else {
                    continue;
                };
                if !opts.merge || !b.place(g, b.ops.len()) {
                    b.push_raw(Op::Gate(g));
                }
            }
            Op::Measure(_)
            | Op::Reset(_)
            | Op::ClassicControlled { .. }
            | Op::XFlip(..)
            | Op::YFlip(..)
            | Op::ZFlip(..)
            | Op::Depolarize1q(..)
            | Op::Depolarize2q(..) => b.push_raw(*op),
        }
    }
    let ops = b.ops.into_iter().flatten().collect();
    Optimized {
        circuit: Circuit {
            num_qubits: c.num_qubits,
            ops,
        },
        global_phase: b.phase.rem_euclid(TAU),
    }
}

struct Builder {
    /// Output ops; merged-away slots become `None`.
    ops: Vec<Option<Op>>,
    /// For every qubit, the indices of `ops` touching it, ascending.
    per_q: Vec<Vec<usize>>,
    phase: f64,
    opts: PeepholeOptions,
}

/// How an already placed op acts on qubit `q`, for commutation checks.
/// Non-unitary ops other than measurement are opaque barriers.
fn op_axis(op: &Op, q: usize) -> Axis {
    match op {
        Op::Gate(g) => axis_on(g, q),
        Op::Measure(_) => Axis::Z,
        Op::Reset(_)
        | Op::ClassicControlled { .. }
        | Op::XFlip(..)
        | Op::YFlip(..)
        | Op::ZFlip(..)
        | Op::Depolarize1q(..)
        | Op::Depolarize2q(..) => Axis::Other,
    }
}

/// Symmetric two-qubit gates commute with each other on the same pair.
fn symmetric_pair(g: &Gate) -> bool {
    matches!(g, Gate::Swap(..) | Gate::Cz(..) | Gate::CPhase(..))
}

fn commutes(h: &Op, g: &Gate) -> bool {
    let (gq, gn) = qubits_of(g);
    if let Op::Gate(hg) = h {
        if gn == 2 && symmetric_pair(g) && symmetric_pair(hg) && same_set(hg, g) {
            return true;
        }
    }
    gq[..gn].iter().all(|&q| {
        let (hq, hn) = op_qubits(h);
        if !hq[..hn].contains(&q) {
            return true;
        }
        let a = op_axis(h, q);
        a != Axis::Other && a == axis_on(g, q)
    })
}

fn same_set(a: &Gate, b: &Gate) -> bool {
    let (aq, an) = qubits_of(a);
    let (bq, bn) = qubits_of(b);
    an == bn && aq[..an].iter().all(|q| bq[..bn].contains(q))
}

impl Builder {
    fn push_raw(&mut self, op: Op) {
        let idx = self.ops.len();
        let (qs, k) = op_qubits(&op);
        for &q in &qs[..k] {
            self.per_q[q].push(idx);
        }
        self.ops.push(Some(op));
    }

    /// Tries to merge `g` into a live op with index `< before` that it can
    /// be commuted back to. Returns `true` if `g` was absorbed.
    fn place(&mut self, g: Gate, before: usize) -> bool {
        let (qs, k) = qubits_of(&g);
        // Cursors into the per-qubit index lists (positions of the first
        // entry >= before).
        let mut cur = [0usize; 3];
        for i in 0..k {
            let list = &self.per_q[qs[i]];
            cur[i] = list.partition_point(|&j| j < before);
        }
        let mut steps = 0;
        loop {
            // Largest index below the cursors over g's qubits.
            let mut best: Option<usize> = None;
            for i in 0..k {
                if cur[i] > 0 {
                    let j = self.per_q[qs[i]][cur[i] - 1];
                    best = Some(best.map_or(j, |b: usize| b.max(j)));
                }
            }
            let Some(j) = best else { return false };
            for i in 0..k {
                if cur[i] > 0 && self.per_q[qs[i]][cur[i] - 1] == j {
                    cur[i] -= 1;
                }
            }
            let Some(h) = self.ops[j] else { continue };
            steps += 1;
            if steps > self.opts.window {
                return false;
            }
            if let Op::Gate(hg) = h {
                if let Some(merged) = merge(&hg, &g, &mut self.phase) {
                    self.ops[j] = None;
                    if let Some(m) = merged {
                        if !self.place(m, j) {
                            self.ops[j] = Some(Op::Gate(m));
                        }
                    }
                    return true;
                }
            }
            if !commutes(&h, &g) {
                return false;
            }
        }
    }
}

/// Reduces `t` into `(-period/2, period/2]`.
fn reduce(t: f64, period: f64) -> f64 {
    let r = t.rem_euclid(period);
    if r > period / 2.0 {
        r - period
    } else {
        r
    }
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < ANGLE_EPS
}

/// `Z`-family gates as `e^{iγ} Phase(φ)`: returns `(q, φ, γ, is_rz)`.
fn zfamily(g: &Gate) -> Option<(usize, f64, f64, bool)> {
    use Gate::*;
    Some(match *g {
        Z(q) => (q, PI, 0.0, false),
        S(q) => (q, FRAC_PI_2, 0.0, false),
        Sdg(q) => (q, -FRAC_PI_2, 0.0, false),
        T(q) => (q, FRAC_PI_4, 0.0, false),
        Tdg(q) => (q, -FRAC_PI_4, 0.0, false),
        Phase(q, t) => (q, t, 0.0, false),
        Rz(q, t) => (q, t, -t / 2.0, true),
        _ => return None,
    })
}

/// `X`/`Y`-family gates as `e^{iγ} R(θ)`: returns `(q, θ, γ, is_y)`.
fn xyfamily(g: &Gate) -> Option<(usize, f64, f64, bool)> {
    use Gate::*;
    // X = e^{iπ/2} Rx(π), Y = e^{iπ/2} Ry(π)
    Some(match *g {
        X(q) => (q, PI, FRAC_PI_2, false),
        Rx(q, t) => (q, t, 0.0, false),
        Y(q) => (q, PI, FRAC_PI_2, true),
        Ry(q, t) => (q, t, 0.0, true),
        _ => return None,
    })
}

/// `e^{iγ} Phase(φ)` on `q` as a gate (or identity), adding any phase.
/// Prefers named gates; keeps `Rz` form if `as_rz`.
fn emit_z(q: usize, phi: f64, gamma: f64, as_rz: bool, phase: &mut f64) -> Option<Gate> {
    let k = phi / FRAC_PI_4;
    if near(k, k.round()) {
        *phase += gamma;
        return match (k.round() as i64).rem_euclid(8) {
            0 => None,
            1 => Some(Gate::T(q)),
            2 => Some(Gate::S(q)),
            4 => Some(Gate::Z(q)),
            6 => Some(Gate::Sdg(q)),
            7 => Some(Gate::Tdg(q)),
            _ => Some(Gate::Phase(q, reduce(phi, TAU))),
        };
    }
    if as_rz {
        // e^{iγ} Phase(φ) = e^{i(γ + φ/2)} Rz(φ); reduce φ mod 4π (exact).
        let t = reduce(phi, 2.0 * TAU);
        // Rz(φ) = Rz(t) since φ - t is a multiple of 4π.
        *phase += gamma + phi / 2.0;
        return Some(Gate::Rz(q, t));
    }
    *phase += gamma;
    Some(Gate::Phase(q, reduce(phi, TAU)))
}

/// `e^{iγ} R(θ)` (`Rx` or `Ry`) on `q` as a gate, adding any phase.
fn emit_xy(q: usize, theta: f64, gamma: f64, is_y: bool, phase: &mut f64) -> Option<Gate> {
    let t = reduce(theta, 2.0 * TAU); // R(θ + 4π) = R(θ)
    *phase += gamma;
    if near(t, 0.0) {
        return None;
    }
    if near(t.abs(), TAU) {
        *phase += PI; // R(2π) = -I
        return None;
    }
    let named = |q| if is_y { Gate::Y(q) } else { Gate::X(q) };
    if near(t, PI) {
        *phase -= FRAC_PI_2; // R(π) = -i P
        return Some(named(q));
    }
    if near(t, -PI) {
        *phase += FRAC_PI_2; // R(-π) = i P
        return Some(named(q));
    }
    Some(if is_y { Gate::Ry(q, t) } else { Gate::Rx(q, t) })
}

fn emit_cphase(a: usize, b: usize, t: f64) -> Option<Gate> {
    let t = reduce(t, TAU);
    if near(t, 0.0) {
        None
    } else if near(t.abs(), PI) {
        Some(Gate::Cz(a, b))
    } else {
        Some(Gate::CPhase(a, b, t))
    }
}

/// Normalises a single gate: removes identities and names rotations by
/// multiples of `π/4` (`π` for X/Y/controlled phase).
fn canonical(g: Gate, phase: &mut f64) -> Option<Gate> {
    use Gate::*;
    match g {
        Rz(q, t) => emit_z(q, t, -t / 2.0, true, phase),
        Phase(q, t) => emit_z(q, t, 0.0, false, phase),
        Rx(q, t) => emit_xy(q, t, 0.0, false, phase),
        Ry(q, t) => emit_xy(q, t, 0.0, true, phase),
        CPhase(a, b, t) => emit_cphase(a, b, t),
        g => Some(g),
    }
}

/// Merges `h` (earlier) and `g` (later) if they are in the same family on
/// the same qubits. `Some(None)`: they cancel; `Some(Some(m))`: replace
/// both by `m`; `None`: no merge.
fn merge(h: &Gate, g: &Gate, phase: &mut f64) -> Option<Option<Gate>> {
    use Gate::*;
    if let (Some((qa, pa, ga, ra)), Some((qb, pb, gb, rb))) = (zfamily(h), zfamily(g)) {
        if qa != qb {
            return None;
        }
        let both_rz = ra && rb;
        let (phi, gamma) = (pa + pb, ga + gb);
        if both_rz {
            // Rz(a) Rz(b) = Rz(a + b) exactly: no phase bookkeeping.
            return Some(emit_z(qa, phi, -phi / 2.0, true, phase));
        }
        return Some(emit_z(qa, phi, gamma, false, phase));
    }
    if let (Some((qa, ta, ga, ya)), Some((qb, tb, gb, yb))) = (xyfamily(h), xyfamily(g)) {
        if qa != qb || ya != yb {
            return None;
        }
        return Some(emit_xy(qa, ta + tb, ga + gb, ya, phase));
    }
    match (*h, *g) {
        (H(a), H(b)) if a == b => Some(None),
        (Cnot(a, b), Cnot(c, d)) if a == c && b == d => Some(None),
        (Swap(..), Swap(..)) if same_set(h, g) => Some(None),
        (Ccx(a, b, t), Ccx(c, d, u)) if t == u && ((a == c && b == d) || (a == d && b == c)) => {
            Some(None)
        }
        (Cz(..) | CPhase(..), Cz(..) | CPhase(..)) if same_set(h, g) => {
            let ang = |x: &Gate| match *x {
                Cz(..) => PI,
                CPhase(_, _, t) => t,
                _ => unreachable!(),
            };
            let (a, b) = match *h {
                Cz(a, b) | CPhase(a, b, _) => (a, b),
                _ => unreachable!(),
            };
            Some(emit_cphase(a, b, ang(h) + ang(g)))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statevector::StateVectorF64;
    use num_complex::Complex64;

    fn check_same(c: &Circuit) -> Optimized {
        let o = optimize(c);
        let mut a = StateVectorF64::new(c.num_qubits);
        a.apply_circuit(c).unwrap();
        let mut b = StateVectorF64::new(c.num_qubits);
        b.apply_circuit(&o.circuit).unwrap();
        let ph = Complex64::from_polar(1.0, o.global_phase);
        for i in 0..1 << c.num_qubits {
            let d = (a.amplitude(i) - b.amplitude(i) * ph).norm();
            assert!(d < 1e-12, "{i}: {d} {c:?} -> {o:?}");
        }
        o
    }

    #[test]
    fn cancels_inverse_pairs_through_commuting_gates() {
        let mut c = Circuit::new(3);
        c.h(0).cnot(0, 1).t(0).cnot(0, 2).tdg(0).cnot(0, 1).h(0);
        // T and T† commute through the CNOT controls and cancel; then
        // CNOT(0,1) .. CNOT(0,2) .. CNOT(0,1): CNOTs sharing a control
        // commute, so the CNOT(0,1) pair cancels too.
        let o = check_same(&c);
        assert_eq!(o.circuit.num_gates(), 3, "{:?}", o.circuit);
    }

    #[test]
    fn merges_rotations() {
        let mut c = Circuit::new(2);
        c.t(0).cz(0, 1).t(0).s(0).rz(1, 0.3).rz(1, -0.3).rx(0, 0.1);
        let o = check_same(&c);
        // T T S -> Z ; Rz pair cancels ; CZ, Z, Rx remain
        assert_eq!(o.circuit.num_gates(), 3, "{:?}", o.circuit);
        assert!(o.circuit.gates().any(|g| *g == Gate::Z(0)));
    }

    #[test]
    fn rz_to_named_gates_tracks_phase() {
        let mut c = Circuit::new(1);
        c.h(0)
            .rz(0, FRAC_PI_2)
            .rx(0, PI)
            .ry(0, -PI)
            .rz(0, 2.0 * TAU + 0.5);
        let o = check_same(&c);
        assert!(o.circuit.gates().any(|g| *g == Gate::S(0)));
    }

    #[test]
    fn qft_times_inverse_vanishes() {
        let mut c = crate::algorithms::qft(6);
        c.append(&crate::algorithms::qft(6).inverse());
        let o = check_same(&c);
        assert_eq!(o.circuit.num_gates(), 0);
    }

    #[test]
    fn diagonal_moves_across_measurement() {
        let mut c = Circuit::new(1);
        c.h(0).t(0).measure(0).tdg(0);
        let o = optimize(&c);
        assert_eq!(o.circuit.ops, vec![Op::Gate(Gate::H(0)), Op::Measure(0)]);
    }
}
