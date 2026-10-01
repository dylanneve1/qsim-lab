//! State-aware simplification: track qubits that are provably in one of
//! the six single-qubit stabilizer states (`|0>, |1>, |+>, |->, |+i>,
//! |-i>`) and are not entangled with anything.
//!
//! Circuits start in `|0...0>`, so many gates act on known product states:
//! oracles flip ancillas prepared in `|->` (phase kickback), controlled
//! phases are controlled by qubits still in a basis state (the QFT of a
//! basis state), Toffolis have a control in `|0>`. Every such gate is
//! replaced by something cheaper and exactly equivalent *on that state*:
//!
//! * a one-qubit gate mapping a known state to a known state is deleted
//!   (its phase goes into the global phase);
//! * `CNOT` with control `|0>`/`|1>` becomes nothing / `X(t)`; with target
//!   `|+>`/`|->` it becomes nothing / `Z(c)` (phase kickback);
//! * `CZ`, `CPhase` with a qubit in `|0>`/`|1>` become nothing / a phase
//!   on the other qubit; `CCX` reduces to `CNOT`, `X`, `CZ` or nothing;
//! * `SWAP` of two known qubits just exchanges the bookkeeping.
//!
//! A known qubit is not touched on the simulated wire (it stays `|0>`) until
//! a gate needs it as a genuine quantum input; then a short preparation
//! (`X`, `H`, `X H`, `H S` or `H S†`) is emitted. The output is therefore
//! never more than a few gates longer, and usually much shorter and less
//! entangling, which the component split then exploits.
//!
//! The rewrite is exact for the state prepared from `|0...0>` (including
//! the global phase, which is returned), not for the circuit as a unitary.

use crate::circuit::{Circuit, Op};
use crate::gate::Gate;
use num_complex::Complex64;
use std::f64::consts::FRAC_1_SQRT_2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum K {
    Z0,
    Z1,
    Xp,
    Xm,
    Yp,
    Ym,
    Unknown,
}

const KNOWN: [K; 6] = [K::Z0, K::Z1, K::Xp, K::Xm, K::Yp, K::Ym];

/// Preparation of a known state from `|0>`, in time order. The canonical
/// vector of each state is defined as the result of this preparation.
fn prep(k: K, q: usize) -> Vec<Gate> {
    match k {
        K::Z0 | K::Unknown => vec![],
        K::Z1 => vec![Gate::X(q)],
        K::Xp => vec![Gate::H(q)],
        K::Xm => vec![Gate::X(q), Gate::H(q)],
        K::Yp => vec![Gate::H(q), Gate::S(q)],
        K::Ym => vec![Gate::H(q), Gate::Sdg(q)],
    }
}

fn canon(k: K) -> [Complex64; 2] {
    let h = FRAC_1_SQRT_2;
    let c = |re, im| Complex64::new(re, im);
    match k {
        K::Z0 => [c(1.0, 0.0), c(0.0, 0.0)],
        K::Z1 => [c(0.0, 0.0), c(1.0, 0.0)],
        K::Xp => [c(h, 0.0), c(h, 0.0)],
        K::Xm => [c(h, 0.0), c(-h, 0.0)],
        K::Yp => [c(h, 0.0), c(0.0, h)],
        K::Ym => [c(h, 0.0), c(0.0, -h)],
        K::Unknown => unreachable!(),
    }
}

/// Simplifies `c` for the input `|0...0>`. Still-known qubits listed in
/// `keep` are prepared at the end, so the reduced final state of `keep`
/// is reproduced exactly (the full state, times `e^{i phase}`, if `keep`
/// is every qubit); measurements always see the right state.
pub fn propagate(c: &Circuit, keep: &[usize]) -> (Circuit, f64) {
    let mut p = Prop {
        k: vec![K::Z0; c.num_qubits],
        out: Circuit::new(c.num_qubits),
        phase: 0.0,
    };
    for op in &c.ops {
        match *op {
            Op::Measure(q) => {
                p.materialize(q);
                p.out.measure(q);
            }
            Op::Gate(g) => p.gate(g),
        }
    }
    for &q in keep {
        p.materialize(q);
    }
    (p.out, p.phase)
}

struct Prop {
    k: Vec<K>,
    out: Circuit,
    phase: f64,
}

impl Prop {
    fn materialize(&mut self, q: usize) {
        if self.k[q] != K::Unknown {
            for g in prep(self.k[q], q) {
                self.out.gate(g);
            }
            self.k[q] = K::Unknown;
        }
    }

    fn emit(&mut self, g: Gate, qs: &[usize]) {
        for &q in qs {
            self.materialize(q);
        }
        self.out.gate(g);
    }

    fn gate(&mut self, g: Gate) {
        use Gate::*;
        use K::*;
        let k = |q: usize| self.k[q];
        match g {
            Cnot(c, t) => match (k(c), k(t)) {
                (Z0, _) => {}
                (Z1, _) => self.gate(X(t)),
                (_, Xp) => {}
                (_, Xm) => self.gate(Z(c)),
                _ => self.emit(g, &[c, t]),
            },
            Cz(a, b) | CPhase(a, b, _) => {
                let on = |q| match g {
                    Cz(..) => Z(q),
                    CPhase(_, _, t) => Phase(q, t),
                    _ => unreachable!(),
                };
                match (k(a), k(b)) {
                    (Z0, _) | (_, Z0) => {}
                    (Z1, _) => self.gate(on(b)),
                    (_, Z1) => self.gate(on(a)),
                    _ => self.emit(g, &[a, b]),
                }
            }
            Ccx(a, b, t) => match (k(a), k(b), k(t)) {
                (Z0, _, _) | (_, Z0, _) => {}
                (Z1, _, _) => self.gate(Cnot(b, t)),
                (_, Z1, _) => self.gate(Cnot(a, t)),
                (_, _, Xp) => {}
                (_, _, Xm) => self.gate(Cz(a, b)),
                _ => self.emit(g, &[a, b, t]),
            },
            Swap(a, b) => {
                if k(a) == Unknown || k(b) == Unknown {
                    // The known side's wire still holds |0>; swap it over.
                    self.out.gate(g);
                }
                self.k.swap(a, b);
            }
            g => {
                let q = g.qubits()[0];
                let kq = k(q);
                if kq == Unknown {
                    self.out.gate(g);
                    return;
                }
                let m = g.matrix_1q().expect("one-qubit gate");
                let v = canon(kq);
                let w = [
                    m[0][0] * v[0] + m[0][1] * v[1],
                    m[1][0] * v[0] + m[1][1] * v[1],
                ];
                for s in KNOWN {
                    let cs = canon(s);
                    let ov = cs[0].conj() * w[0] + cs[1].conj() * w[1];
                    // Accept only if w = ov * cs to within rounding (the
                    // residual is linear in any deviation, unlike |ov|).
                    let res = (w[0] - ov * cs[0]).norm() + (w[1] - ov * cs[1]).norm();
                    if res < 1e-13 {
                        self.k[q] = s;
                        self.phase += ov.arg();
                        return;
                    }
                }
                self.emit(g, &[q]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statevector::StateVectorF64;

    fn check(c: &Circuit) -> Circuit {
        let all: Vec<usize> = (0..c.num_qubits).collect();
        let (o, ph) = propagate(c, &all);
        let mut a = StateVectorF64::new(c.num_qubits);
        a.apply_circuit(c).unwrap();
        let mut b = StateVectorF64::new(c.num_qubits);
        b.apply_circuit(&o).unwrap();
        let ph = Complex64::from_polar(1.0, ph);
        for i in 0..1 << c.num_qubits {
            assert!((a.amplitude(i) - b.amplitude(i) * ph).norm() < 1e-12);
        }
        o
    }

    #[test]
    fn bernstein_vazirani_becomes_x_gates() {
        let c = crate::algorithms::bernstein_vazirani(6, 0b101101);
        let (o, _) = propagate(&c, &[]);
        assert!(o.gates().all(|g| matches!(g, Gate::X(_))), "{o:?}");
        assert_eq!(o.num_gates(), 4);
    }

    #[test]
    fn qft_of_basis_state_has_no_entangling_gates() {
        let n = 6;
        let mut c = Circuit::new(n);
        c.x(0).x(3).x(4);
        c.append(&crate::algorithms::qft(n));
        let o = check(&c);
        assert!(o
            .gates()
            .all(|g| g.arity() == 1 || matches!(g, Gate::Swap(..))));
    }
}
