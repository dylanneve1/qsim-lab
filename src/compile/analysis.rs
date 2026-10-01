//! Structural analyses and the exact rewrites built on them.

use super::{classical_dependency, is_diagonal, is_monomial, op_qubits, qubits_of};
use crate::circuit::{Circuit, Op};
use crate::gate::Gate;

/// Keeps only the operations in the backward light cone of the
/// measurements and of the `outputs` qubits (whose final state the caller
/// still needs).
///
/// Walking backwards, an op is kept if it touches a live qubit, and then
/// all its qubits become live. Every dropped op is a trace-preserving
/// channel (a gate, reset, noise channel, or a classically controlled gate,
/// which is a gate in every branch of the outcome record) on qubits that
/// nothing kept later touches, so tracing those qubits out shows that the
/// joint distribution of the measurement record, and the reduced state of
/// `outputs`, are unchanged.
///
/// Every measurement is kept: it is part of the outcome record (and
/// classically controlled ops address outcomes by their index). The
/// classical dependency of a kept `ClassicControlled` op is therefore
/// always kept, together with the light cone of the qubit it measured.
pub fn light_cone(c: &Circuit, outputs: &[usize]) -> Circuit {
    let mut live = vec![false; c.num_qubits];
    for &q in outputs {
        live[q] = true;
    }
    let mut keep = vec![false; c.ops.len()];
    for (i, op) in c.ops.iter().enumerate().rev() {
        let (qs, k) = op_qubits(op);
        let keep_op = match op {
            Op::Measure(_) => true,
            Op::Gate(_)
            | Op::Reset(_)
            | Op::ClassicControlled { .. }
            | Op::XFlip(..)
            | Op::YFlip(..)
            | Op::ZFlip(..)
            | Op::Depolarize1q(..)
            | Op::Depolarize2q(..) => qs[..k].iter().any(|&q| live[q]),
        };
        if keep_op {
            keep[i] = true;
            for &q in &qs[..k] {
                live[q] = true;
            }
        }
    }
    let out = Circuit {
        num_qubits: c.num_qubits,
        ops: c
            .ops
            .iter()
            .zip(&keep)
            .filter(|(_, &k)| k)
            .map(|(op, _)| *op)
            .collect(),
    };
    debug_assert_eq!(
        measurement_count(&out),
        measurement_count(c),
        "light cone must keep every measurement"
    );
    out
}

fn measurement_count(c: &Circuit) -> usize {
    c.ops.iter().filter(|o| matches!(o, Op::Measure(_))).count()
}

/// If the circuit is unitary gates followed by terminal measurements (no
/// gate touches a qubit after it is measured), returns the unitary body and
/// the measured qubit of each outcome in program order. Circuits with
/// resets, classically controlled gates or noise channels are never
/// terminal (`None`).
pub fn terminal_measurements(c: &Circuit) -> Option<(Circuit, Vec<usize>)> {
    let mut measured = vec![false; c.num_qubits];
    let mut body = Circuit::new(c.num_qubits);
    let mut meas = Vec::new();
    for op in &c.ops {
        match op {
            Op::Measure(q) => {
                measured[*q] = true;
                meas.push(*q);
            }
            Op::Gate(g) => {
                let (qs, k) = qubits_of(g);
                if qs[..k].iter().any(|&q| measured[q]) {
                    return None;
                }
                body.gate(*g);
            }
            Op::Reset(_)
            | Op::ClassicControlled { .. }
            | Op::XFlip(..)
            | Op::YFlip(..)
            | Op::ZFlip(..)
            | Op::Depolarize1q(..)
            | Op::Depolarize2q(..) => return None,
        }
    }
    Some((body, meas))
}

/// Splits a unitary circuit that is followed only by computational-basis
/// measurement into `body` and a classical suffix.
///
/// A gate belongs to the suffix if it is monomial (a permutation of basis
/// states times phases) and every later gate on its qubits does too. The
/// suffix then maps `|x>` to `phase(x) |π(x)>`, so measuring after it is
/// the same as measuring after `body` and applying `π` to the bit string.
/// The returned suffix holds only the permutation gates (`X`, `CNOT`,
/// `SWAP`, `CCX`; `Y` becomes `X`); its diagonal gates are dropped.
pub fn split_monomial_suffix(body: &Circuit) -> (Circuit, Vec<Gate>) {
    let n = body.num_qubits;
    let mut blocked = vec![false; n];
    let mut in_suffix = vec![false; body.ops.len()];
    for (i, op) in body.ops.iter().enumerate().rev() {
        let Op::Gate(g) = op else {
            panic!("split_monomial_suffix: body must be unitary")
        };
        let (qs, k) = qubits_of(g);
        if is_monomial(g) && qs[..k].iter().all(|&q| !blocked[q]) {
            in_suffix[i] = true;
        } else {
            for &q in &qs[..k] {
                blocked[q] = true;
            }
        }
    }
    let mut rest = Circuit::new(n);
    let mut suffix = Vec::new();
    for (op, &s) in body.ops.iter().zip(&in_suffix) {
        let Op::Gate(g) = op else { unreachable!() };
        if !s {
            rest.gate(*g);
        } else if !is_diagonal(g) {
            suffix.push(match *g {
                Gate::Y(q) => Gate::X(q),
                g => g,
            });
        }
    }
    (rest, suffix)
}

/// Applies a classical permutation suffix (from [`split_monomial_suffix`])
/// to a bit string.
pub fn apply_classical(bits: &mut [bool], suffix: &[Gate]) {
    for g in suffix {
        match *g {
            Gate::X(q) => bits[q] ^= true,
            Gate::Cnot(c, t) => bits[t] ^= bits[c],
            Gate::Swap(a, b) => bits.swap(a, b),
            Gate::Ccx(a, b, t) => bits[t] ^= bits[a] & bits[b],
            _ => unreachable!("not a classical gate: {g:?}"),
        }
    }
}

/// The qubits whose pre-suffix bits the measured outputs depend on.
pub fn suffix_inputs(n: usize, meas: &[usize], suffix: &[Gate]) -> Vec<bool> {
    let mut need = vec![false; n];
    for &q in meas {
        need[q] = true;
    }
    for g in suffix.iter().rev() {
        match *g {
            Gate::X(_) => {}
            Gate::Cnot(c, t) => need[c] |= need[t],
            Gate::Swap(a, b) => need.swap(a, b),
            Gate::Ccx(a, b, t) => {
                if need[t] {
                    need[a] = true;
                    need[b] = true;
                }
            }
            _ => unreachable!(),
        }
    }
    need
}

/// Connected components of the qubit-interaction graph: qubits are joined
/// by any multi-qubit gate or noise channel, and a classically controlled
/// gate joins its qubits to the qubit whose measurement it reads (a
/// classical dependency: the components could not be sampled
/// independently otherwise). Every qubit appears in exactly one component;
/// components are sorted by their smallest qubit and each is ascending.
///
/// The circuit must pass [`validate`](super::validate).
pub fn components(c: &Circuit) -> Vec<Vec<usize>> {
    let n = c.num_qubits;
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    fn union(p: &mut [usize], a: usize, b: usize) {
        let (a, b) = (find(p, a), find(p, b));
        if a != b {
            p[a.max(b)] = a.min(b);
        }
    }
    let mut meas_qubit = Vec::new();
    for op in &c.ops {
        let (qs, k) = op_qubits(op);
        for i in 1..k {
            union(&mut parent, qs[0], qs[i]);
        }
        if let Some(m) = classical_dependency(op) {
            union(&mut parent, qs[0], meas_qubit[m]);
        }
        if let Op::Measure(q) = op {
            meas_qubit.push(*q);
        }
    }
    let mut by_root: Vec<Vec<usize>> = vec![Vec::new(); n];
    for q in 0..n {
        let r = find(&mut parent, q);
        by_root[r].push(q);
    }
    by_root.into_iter().filter(|v| !v.is_empty()).collect()
}

/// The ops of `c` acting on `qubits` (which must be a union of
/// [`components`]), relabelled so `qubits[i]` becomes qubit `i`.
/// Classically controlled ops are renumbered to the measurement index
/// within the restricted circuit.
pub fn restrict(c: &Circuit, qubits: &[usize]) -> Circuit {
    let mut map = vec![usize::MAX; c.num_qubits];
    for (i, &q) in qubits.iter().enumerate() {
        map[q] = i;
    }
    // Local index of every global measurement in the restriction.
    let mut local_meas: Vec<Option<usize>> = Vec::new();
    let mut n_local = 0;
    let mut out = Circuit::new(qubits.len());
    for op in &c.ops {
        let (qs, k) = op_qubits(op);
        let inside = map[qs[0]] != usize::MAX;
        debug_assert!(
            qs[..k].iter().all(|&q| (map[q] != usize::MAX) == inside),
            "restrict: qubits must be a union of components"
        );
        if let Op::Measure(_) = op {
            local_meas.push(inside.then_some(n_local));
            if inside {
                n_local += 1;
            }
        }
        if !inside {
            continue;
        }
        let f = |q: usize| map[q];
        let new = match *op {
            Op::ClassicControlled {
                gate,
                meas_index,
                target_value,
            } => Op::ClassicControlled {
                gate: relabel(&gate, f),
                meas_index: local_meas[meas_index]
                    .expect("restrict: classical dependency outside the component"),
                target_value,
            },
            ref other => relabel_op(other, f),
        };
        out.ops.push(new);
    }
    out
}

/// The gate with every qubit `q` replaced by `f(q)`.
pub fn relabel(g: &Gate, f: impl Fn(usize) -> usize) -> Gate {
    use Gate::*;
    match *g {
        H(q) => H(f(q)),
        X(q) => X(f(q)),
        Y(q) => Y(f(q)),
        Z(q) => Z(f(q)),
        S(q) => S(f(q)),
        Sdg(q) => Sdg(f(q)),
        T(q) => T(f(q)),
        Tdg(q) => Tdg(f(q)),
        Rx(q, t) => Rx(f(q), t),
        Ry(q, t) => Ry(f(q), t),
        Rz(q, t) => Rz(f(q), t),
        Phase(q, t) => Phase(f(q), t),
        Cnot(a, b) => Cnot(f(a), f(b)),
        Cz(a, b) => Cz(f(a), f(b)),
        Swap(a, b) => Swap(f(a), f(b)),
        CPhase(a, b, t) => CPhase(f(a), f(b), t),
        Ccx(a, b, t) => Ccx(f(a), f(b), f(t)),
    }
}

/// The op with every qubit `q` replaced by `f(q)` (classical indices are
/// unchanged).
pub fn relabel_op(op: &Op, f: impl Fn(usize) -> usize) -> Op {
    match *op {
        Op::Gate(g) => Op::Gate(relabel(&g, f)),
        Op::Measure(q) => Op::Measure(f(q)),
        Op::Reset(q) => Op::Reset(f(q)),
        Op::ClassicControlled {
            gate,
            meas_index,
            target_value,
        } => Op::ClassicControlled {
            gate: relabel(&gate, f),
            meas_index,
            target_value,
        },
        Op::XFlip(q, p) => Op::XFlip(f(q), p),
        Op::YFlip(q, p) => Op::YFlip(f(q), p),
        Op::ZFlip(q, p) => Op::ZFlip(f(q), p),
        Op::Depolarize1q(q, p) => Op::Depolarize1q(f(q), p),
        Op::Depolarize2q(a, b, p) => Op::Depolarize2q(f(a), f(b), p),
    }
}

/// Removes every SWAP by relabelling the wires of all later operations.
///
/// Returns the new circuit and `wire_of`, where `wire_of[q]` is the wire
/// that holds logical qubit `q` at the end. Measurements are relabelled
/// too, so outcome records are unchanged; the final state of the original
/// circuit on qubit `q` is the final state of the new one on `wire_of[q]`.
/// A classically controlled SWAP is not eliminated (whether it happens
/// depends on the shot); it is relabelled like every other op.
pub fn eliminate_swaps(c: &Circuit) -> (Circuit, Vec<usize>) {
    let mut wire_of: Vec<usize> = (0..c.num_qubits).collect();
    let mut out = Circuit::new(c.num_qubits);
    for op in &c.ops {
        match op {
            Op::Gate(Gate::Swap(a, b)) => wire_of.swap(*a, *b),
            other => out.ops.push(relabel_op(other, |q| wire_of[q])),
        }
    }
    (out, wire_of)
}

/// Splits a circuit into its causal Clifford prefix and the rest.
///
/// A gate is in the prefix if it is Clifford and every earlier op on any of
/// its qubits is in the prefix. The prefix is closed under predecessors, so
/// running it first and then the remaining ops in their original order is
/// the same circuit (only commuting, qubit-disjoint ops change order).
/// The prefix contains only unitary gates: measurements, resets, noise
/// channels and classically controlled gates all go to the rest and close
/// their qubits, so the rest keeps every measurement in its original order.
pub fn clifford_prefix(c: &Circuit) -> (Circuit, Circuit) {
    let mut open = vec![true; c.num_qubits];
    let mut prefix = Circuit::new(c.num_qubits);
    let mut rest = Circuit::new(c.num_qubits);
    for op in &c.ops {
        let (qs, k) = op_qubits(op);
        let to_prefix = match op {
            Op::Gate(g) => g.is_clifford() && qs[..k].iter().all(|&q| open[q]),
            Op::Measure(_)
            | Op::Reset(_)
            | Op::ClassicControlled { .. }
            | Op::XFlip(..)
            | Op::YFlip(..)
            | Op::ZFlip(..)
            | Op::Depolarize1q(..)
            | Op::Depolarize2q(..) => false,
        };
        if to_prefix {
            prefix.ops.push(*op);
        } else {
            for &q in &qs[..k] {
                open[q] = false;
            }
            rest.ops.push(*op);
        }
    }
    (prefix, rest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn light_cone_of_one_qubit() {
        let mut c = Circuit::new(4);
        c.h(0).cnot(0, 1).h(2).cnot(2, 3).t(1).measure(1).x(3);
        let l = light_cone(&c, &[]);
        assert_eq!(l.num_gates(), 3);
    }

    #[test]
    fn suffix_extraction() {
        let mut c = Circuit::new(3);
        c.h(0).cnot(0, 1).t(1).h(1).cnot(1, 2).t(2).swap(0, 2).x(0);
        let (rest, suf) = split_monomial_suffix(&c);
        assert_eq!(rest.num_gates(), 4);
        assert_eq!(suf, vec![Gate::Cnot(1, 2), Gate::Swap(0, 2), Gate::X(0)]);
    }

    #[test]
    fn components_and_prefix() {
        let mut c = Circuit::new(5);
        c.h(0).cnot(0, 2).t(2).cnot(2, 1).h(3);
        assert_eq!(components(&c), vec![vec![0, 1, 2], vec![3], vec![4]]);
        let (p, r) = clifford_prefix(&c);
        assert_eq!(p.num_gates(), 3);
        assert_eq!(r.num_gates(), 2);
    }
}
