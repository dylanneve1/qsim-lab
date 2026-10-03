//! Generators of repeated circuits: Trotter steps, QAOA layers, Grover
//! iterations, repetition-code rounds, repeated brickwork. Used by the
//! tests and benchmarks of the repeat pass.

use crate::circuit::Circuit;
use crate::gate::Gate;
use rand::Rng;

/// `exp(-i J dt ZZ)` on `(a, b)` as `CNOT Rz CNOT`.
fn zz(c: &mut Circuit, a: usize, b: usize, theta: f64) {
    c.cnot(a, b).rz(b, theta).cnot(a, b);
}

/// One transverse-field Ising Trotter step on a chain.
pub fn tfim_step(n: usize, jdt: f64, hdt: f64) -> Circuit {
    let mut c = Circuit::new(n);
    for i in 0..n - 1 {
        zz(&mut c, i, i + 1, 2.0 * jdt);
    }
    for i in 0..n {
        c.rx(i, 2.0 * hdt);
    }
    c
}

/// `steps` identical Trotter steps (after an `H` layer).
pub fn trotter(n: usize, steps: usize, jdt: f64, hdt: f64) -> Circuit {
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    let step = tfim_step(n, jdt, hdt);
    for _ in 0..steps {
        c.append(&step);
    }
    c
}

/// QAOA on a ring: `H` layer, then for each layer a cost layer
/// (`CNOT Rz(γ) CNOT` per edge) and a mixer `Rx(β)`. Angles differ per layer.
pub fn qaoa_ring(n: usize, gammas: &[f64], betas: &[f64]) -> Circuit {
    assert_eq!(gammas.len(), betas.len());
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    for (g, b) in gammas.iter().zip(betas) {
        for i in 0..n {
            zz(&mut c, i, (i + 1) % n, *g);
        }
        for i in 0..n {
            c.rx(i, 2.0 * b);
        }
    }
    c
}

/// Multi-controlled Z for up to two qubits.
fn mcz(c: &mut Circuit, qs: &[usize], _anc: &[usize]) {
    match qs.len() {
        1 => {
            c.z(qs[0]);
        }
        _ => {
            c.cz(qs[0], qs[1]);
        }
    }
}

/// Grover search for `marked` on `n` data qubits (`n - 2` ancillas for
/// `n >= 3`), `iters` iterations of oracle + diffusion.
pub fn grover(n: usize, marked: usize, iters: usize) -> Circuit {
    let anc_n = n.saturating_sub(2);
    let mut c = Circuit::new(n + anc_n);
    let data: Vec<usize> = (0..n).collect();
    let anc: Vec<usize> = (n..n + anc_n).collect();
    for q in 0..n {
        c.h(q);
    }
    let mut it = Circuit::new(n + anc_n);
    // oracle
    for q in 0..n {
        if marked >> q & 1 == 0 {
            it.x(q);
        }
    }
    mcz_exact_cz(&mut it, &data, &anc);
    for q in 0..n {
        if marked >> q & 1 == 0 {
            it.x(q);
        }
    }
    // diffusion
    for q in 0..n {
        it.h(q).x(q);
    }
    mcz_exact_cz(&mut it, &data, &anc);
    for q in 0..n {
        it.x(q).h(q);
    }
    for _ in 0..iters {
        c.append(&it);
    }
    c
}

/// Multi-controlled Z via a Toffoli ladder: the last step is a `CZ`.
fn mcz_exact_cz(c: &mut Circuit, qs: &[usize], anc: &[usize]) {
    let k = qs.len();
    if k <= 2 {
        mcz(c, qs, anc);
        return;
    }
    c.ccx(qs[0], qs[1], anc[0]);
    for i in 2..k - 1 {
        c.ccx(qs[i], anc[i - 2], anc[i - 1]);
    }
    c.cz(anc[k - 3], qs[k - 1]);
    for i in (2..k - 1).rev() {
        c.ccx(qs[i], anc[i - 2], anc[i - 1]);
    }
    c.ccx(qs[0], qs[1], anc[0]);
}

/// `block` repeated `reps` times.
pub fn repeated(block: &Circuit, reps: usize) -> Circuit {
    let mut c = Circuit::new(block.num_qubits);
    for _ in 0..reps {
        c.append(block);
    }
    c
}

/// A random topological order of the circuit's dependency DAG (gates on a
/// shared qubit keep their order; everything else is shuffled): the same
/// circuit with commuting gates reordered. Unitary circuits only.
pub fn shuffle_commuting<R: Rng>(c: &Circuit, rng: &mut R) -> Circuit {
    use crate::circuit::Op;
    let ops = &c.ops;
    let m = ops.len();
    let qs = |op: &Op| -> Vec<usize> {
        match op {
            Op::Gate(g) => g.qubits(),
            _ => panic!("unitary circuits only"),
        }
    };
    let mut indeg = vec![0usize; m];
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); m];
    let mut last: Vec<Option<usize>> = vec![None; c.num_qubits];
    for (i, op) in ops.iter().enumerate() {
        let mut preds: Vec<usize> = Vec::new();
        for q in qs(op) {
            if let Some(p) = last[q] {
                if !preds.contains(&p) {
                    preds.push(p);
                }
            }
            last[q] = Some(i);
        }
        for p in preds {
            succ[p].push(i);
            indeg[i] += 1;
        }
    }
    let mut ready: Vec<usize> = (0..m).filter(|&i| indeg[i] == 0).collect();
    let mut out = Vec::with_capacity(m);
    while !ready.is_empty() {
        let k = rng.random_range(0..ready.len());
        let i = ready.swap_remove(k);
        out.push(ops[i]);
        for &s in &succ[i] {
            indeg[s] -= 1;
            if indeg[s] == 0 {
                ready.push(s);
            }
        }
    }
    Circuit {
        num_qubits: c.num_qubits,
        ops: out,
    }
}

/// A random Clifford gate list on `n` qubits.
pub fn random_clifford_gates<R: Rng>(n: usize, len: usize, rng: &mut R) -> Vec<Gate> {
    (0..len)
        .map(|_| {
            let a = rng.random_range(0..n);
            if n >= 2 && rng.random_bool(0.5) {
                let mut b = rng.random_range(0..n - 1);
                if b >= a {
                    b += 1;
                }
                match rng.random_range(0..4) {
                    0 => Gate::Cnot(a, b),
                    1 => Gate::Cz(a, b),
                    2 => Gate::Swap(a, b),
                    _ => Gate::ISwap(a, b),
                }
            } else {
                match rng.random_range(0..7) {
                    0 => Gate::H(a),
                    1 => Gate::S(a),
                    2 => Gate::Sdg(a),
                    3 => Gate::X(a),
                    4 => Gate::Y(a),
                    5 => Gate::Sx(a),
                    _ => Gate::Sxdg(a),
                }
            }
        })
        .collect()
}

fn qec_round(d: usize, first: bool) -> Circuit {
    let n = 2 * d - 1;
    let mut c = Circuit::new(n);
    for j in 0..d - 1 {
        let a = d + j;
        if !first {
            c.reset(a);
        }
    }
    for j in 0..d - 1 {
        let a = d + j;
        c.cnot(j, a).cnot(j + 1, a);
    }
    for j in 0..d - 1 {
        c.measure(d + j);
    }
    c
}

/// Noiseless repetition-code memory circuit as written by
/// `RepetitionCode::build_circuit` (data `0..d`, ancillas `d..2d-1`); with
/// `plus` the data start in `|+...+>` (random first syndrome round).
pub fn qec_memory(d: usize, rounds: usize, plus: bool) -> Circuit {
    let n = 2 * d - 1;
    let mut c = Circuit::new(n);
    if plus {
        for q in 0..d {
            c.h(q);
        }
    }
    for r in 0..rounds {
        c.append(&qec_round(d, r == 0));
    }
    for i in 0..d {
        c.measure(i);
    }
    c
}

/// The same experiment as a [`super::Program`] with an explicit repeat (no
/// `rounds` copies are materialised): round 0, then `rounds - 1` copies of
/// the reset-extract-measure round, then the data readout.
pub fn qec_program(d: usize, rounds: usize, plus: bool) -> super::Program {
    use super::{Node, Program};
    let n = 2 * d - 1;
    let mut pre = Circuit::new(n);
    if plus {
        for q in 0..d {
            pre.h(q);
        }
    }
    pre.append(&qec_round(d, true));
    let mut fin = Circuit::new(n);
    for i in 0..d {
        fin.measure(i);
    }
    let mut nodes = vec![Node::Ops(pre.ops)];
    if rounds > 1 {
        nodes.push(Node::Repeat {
            body: vec![Node::Ops(qec_round(d, false).ops)],
            reps: rounds - 1,
        });
    }
    nodes.push(Node::Ops(fin.ops));
    Program {
        num_qubits: n,
        nodes,
    }
}
