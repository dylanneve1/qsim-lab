//! Constant folding through wires in a known computational basis state.
//!
//! Every wire starts in `|0>`. While a wire is known to be `|b>`:
//! * `X`, `Y` flip it (`Y` also contributes a global phase `±i`);
//! * a diagonal op whose qubits are all known is a global phase (with a
//!   parameterised angle if the op is parameterised); with some qubits
//!   known it shrinks (`CPhase` with a known `1` control becomes `Phase`,
//!   with a known `0` it vanishes; a known wire in an `Rzz`/gadget turns
//!   into a sign of the angle);
//! * `CNOT`/`Toffoli`/`CZ` with a known `0` control vanish, with known `1`
//!   controls they lose that control;
//! * `SWAP` of two known wires swaps the knowledge, of one known wire it
//!   stays (the knowledge moves).
//!
//! Anything else makes its wires unknown. The result is exactly equal to
//! the input (global phase included, via [`POp::Global`]).

use super::param::{Angle, POp, ParamCircuit};
use crate::gate::Gate;
use std::f64::consts::PI;

/// What [`fold_basis`] removed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FoldStats {
    pub removed: usize,
    pub simplified: usize,
}

/// Folds known basis-state wires (see the module docs).
pub fn fold_basis(pc: &ParamCircuit) -> (ParamCircuit, FoldStats) {
    let n = pc.num_qubits;
    let mut known: Vec<Option<bool>> = vec![Some(false); n];
    let mut out = ParamCircuit::new(n, pc.num_params);
    let mut st = FoldStats::default();
    let mut global = Angle::constant(0.0);
    let c = Angle::constant;
    for op in &pc.ops {
        let qs = op.qubits();
        if qs.iter().all(|&q| known[q].is_none()) {
            out.ops.push(op.clone());
            continue;
        }
        // e^{iφ·b} for a known bit b
        let mut emit: Vec<POp> = Vec::new();
        let mut handled = true;
        match op {
            POp::Global(a) => global = global.plus(a),
            POp::Fixed(g) => match *g {
                Gate::I(_) => {}
                Gate::X(q) => {
                    let b = known[q].expect("known");
                    known[q] = Some(!b);
                }
                Gate::Y(q) => {
                    // Y|0> = i|1>, Y|1> = -i|0>
                    let b = known[q].expect("known");
                    global = global.plus(&c(if b { -PI / 2.0 } else { PI / 2.0 }));
                    known[q] = Some(!b);
                }
                Gate::Z(q) | Gate::S(q) | Gate::Sdg(q) | Gate::T(q) | Gate::Tdg(q) => {
                    if known[q] == Some(true) {
                        let phi = match g {
                            Gate::Z(_) => PI,
                            Gate::S(_) => PI / 2.0,
                            Gate::Sdg(_) => -PI / 2.0,
                            Gate::T(_) => PI / 4.0,
                            _ => -PI / 4.0,
                        };
                        global = global.plus(&c(phi));
                    }
                }
                Gate::Rz(q, t) => {
                    global = global.plus(&c(if known[q] == Some(true) {
                        t / 2.0
                    } else {
                        -t / 2.0
                    }))
                }
                Gate::Phase(q, t) => {
                    if known[q] == Some(true) {
                        global = global.plus(&c(t));
                    }
                }
                Gate::Cnot(a, b) => match known[a] {
                    Some(false) => {}
                    Some(true) => match known[b] {
                        Some(v) => known[b] = Some(!v),
                        None => emit.push(POp::Fixed(Gate::X(b))),
                    },
                    None => {
                        // control unknown, target known: target becomes unknown
                        handled = false;
                    }
                },
                Gate::Cz(a, b) | Gate::CPhase(a, b, _) => {
                    let t = if let Gate::CPhase(_, _, t) = *g {
                        t
                    } else {
                        PI
                    };
                    match (known[a], known[b]) {
                        (Some(false), _) | (_, Some(false)) => {}
                        (Some(true), Some(true)) => global = global.plus(&c(t)),
                        (Some(true), None) => emit.push(POp::Phase(b, c(t))),
                        (None, Some(true)) => emit.push(POp::Phase(a, c(t))),
                        (None, None) => unreachable!(),
                    }
                }
                Gate::Ccx(a, b, t) => match (known[a], known[b]) {
                    (Some(false), _) | (_, Some(false)) => {}
                    (Some(true), Some(true)) => match known[t] {
                        Some(v) => known[t] = Some(!v),
                        None => emit.push(POp::Fixed(Gate::X(t))),
                    },
                    (Some(true), None) => {
                        emit.push(POp::Fixed(Gate::Cnot(b, t)));
                        known[t] = None;
                    }
                    (None, Some(true)) => {
                        emit.push(POp::Fixed(Gate::Cnot(a, t)));
                        known[t] = None;
                    }
                    (None, None) => handled = false,
                },
                Gate::Swap(a, b) => {
                    if known[a].is_none() || known[b].is_none() {
                        // the unknown state still has to move
                        emit.push(op.clone());
                    }
                    known.swap(a, b);
                }
                _ => handled = false,
            },
            POp::Rz(q, a) => {
                let s = if known[*q] == Some(true) { 0.5 } else { -0.5 };
                global = global.plus(&a.times(s));
            }
            POp::Phase(q, a) => {
                if known[*q] == Some(true) {
                    global = global.plus(a);
                }
            }
            POp::CPhase(x, y, a) => match (known[*x], known[*y]) {
                (Some(false), _) | (_, Some(false)) => {}
                (Some(true), Some(true)) => global = global.plus(a),
                (Some(true), None) => emit.push(POp::Phase(*y, a.clone())),
                (None, Some(true)) => emit.push(POp::Phase(*x, a.clone())),
                (None, None) => unreachable!(),
            },
            POp::Rzz(x, y, a) => {
                let v = vec![*x, *y];
                fold_zstring(&v, a, &known, &mut global, &mut emit);
            }
            POp::ZString(v, a) => fold_zstring(v, a, &known, &mut global, &mut emit),
            _ => handled = false,
        }
        if handled {
            if emit.is_empty() {
                st.removed += 1;
            } else {
                st.simplified += 1;
            }
            out.ops.extend(emit);
        } else {
            for &q in &qs {
                known[q] = None;
            }
            out.ops.push(op.clone());
        }
    }
    if !(global.is_const() && global.c0 == 0.0) {
        out.ops.push(POp::Global(global));
    }
    (out, st)
}

/// `exp(-iθ/2 Z^{⊗v})` with some wires known: each known `1` flips the
/// sign; all known gives a global phase.
fn fold_zstring(
    v: &[usize],
    a: &Angle,
    known: &[Option<bool>],
    global: &mut Angle,
    emit: &mut Vec<POp>,
) {
    let mut sign = 1.0;
    let mut rest = Vec::new();
    for &q in v {
        match known[q] {
            Some(true) => sign = -sign,
            Some(false) => {}
            None => rest.push(q),
        }
    }
    let a = a.times(sign);
    if rest.is_empty() {
        *global = global.plus(&a.times(-0.5));
    } else if rest.len() == 1 {
        emit.push(POp::Rz(rest[0], a));
    } else {
        emit.push(POp::ZString(rest, a));
    }
}
