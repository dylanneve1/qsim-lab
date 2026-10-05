//! Structural deduplication: find repeated subgraphs *up to qubit
//! relabelling* without any repeat markers.
//!
//! The circuit is cut into blocks of at most `k` qubits along its wire DAG
//! (the greedy grouping a gate-fusion pass uses: an op joins the open block
//! of its qubits while the union stays within `k` qubits). Each block gets a
//! canonical form: its ops with qubits renamed in order of first use, gate
//! kinds, and angles (exact bits of every affine coefficient). Blocks with
//! equal canonical forms are one *class*; an instance is a class plus the
//! map from canonical to physical qubits. A second, coarser key ignores the
//! angles (*shape* classes: the same block with different angles, e.g. one
//! ansatz layer per parameter set).
//!
//! What a class buys: anything computed per block depends only on the class
//! (its unitary, its fused matrix, its rewritten form, its bound numbers for
//! one parameter vector), so it is computed once and instantiated through
//! the qubit map. [`block_unitaries`] measures that on block unitaries;
//! `CompiledCircuit` uses the same idea for its numeric recipes (identical
//! recipes are evaluated once per bind).

use super::param::{Angle, POp, ParamCircuit};
use crate::gate::Gate;
use num_complex::Complex64;
use std::collections::HashMap;

/// A block: ops (indices into the circuit) and its qubits in canonical order.
#[derive(Clone, Debug)]
pub struct Block {
    pub ops: Vec<usize>,
    /// `qubits[j]` = physical qubit of canonical qubit `j`.
    pub qubits: Vec<usize>,
    pub class: usize,
    pub shape: usize,
}

/// Result of [`analyse`].
#[derive(Clone, Debug, Default)]
pub struct Dedup {
    pub blocks: Vec<Block>,
    pub num_classes: usize,
    pub num_shapes: usize,
    /// Instances per class.
    pub class_count: Vec<usize>,
    pub shape_count: Vec<usize>,
    /// Ops in blocks whose class has at least two instances.
    pub covered_ops: usize,
    /// Same for shape classes.
    pub covered_ops_shape: usize,
    pub total_ops: usize,
    pub secs: f64,
}

impl Dedup {
    /// Fraction of ops in a repeated class.
    pub fn coverage(&self) -> f64 {
        self.covered_ops as f64 / self.total_ops.max(1) as f64
    }
    /// Fraction of ops in a repeated shape class.
    pub fn shape_coverage(&self) -> f64 {
        self.covered_ops_shape as f64 / self.total_ops.max(1) as f64
    }
    /// Per-block work saved by computing it once per class: `1 - classes /
    /// blocks` (counting only blocks with at least two ops).
    pub fn reuse(&self) -> f64 {
        1.0 - self.num_classes as f64 / self.blocks.len().max(1) as f64
    }
}

fn gate_code(g: &Gate) -> (u8, Vec<f64>) {
    use Gate::*;
    match *g {
        I(_) => (0, vec![]),
        H(_) => (1, vec![]),
        X(_) => (2, vec![]),
        Y(_) => (3, vec![]),
        Z(_) => (4, vec![]),
        S(_) => (5, vec![]),
        Sdg(_) => (6, vec![]),
        T(_) => (7, vec![]),
        Tdg(_) => (8, vec![]),
        Sx(_) => (9, vec![]),
        Sxdg(_) => (10, vec![]),
        Rx(_, t) => (11, vec![t]),
        Ry(_, t) => (12, vec![t]),
        Rz(_, t) => (13, vec![t]),
        Phase(_, t) => (14, vec![t]),
        U(_, a, b, c) => (15, vec![a, b, c]),
        Cnot(..) => (16, vec![]),
        Cz(..) => (17, vec![]),
        Swap(..) => (18, vec![]),
        ISwap(..) => (19, vec![]),
        ISwapdg(..) => (20, vec![]),
        CPhase(_, _, t) => (21, vec![t]),
        Ccx(..) => (22, vec![]),
    }
}

/// `(kind code, exact angle bits)` of an op; qubits are handled by the caller.
pub(crate) fn op_key(op: &POp, with_angles: bool, out: &mut Vec<u64>) {
    let push_angle = |a: &Angle, out: &mut Vec<u64>| {
        out.push(a.c0.to_bits());
        for &(i, c) in &a.terms {
            out.push(u64::from(i) | 1 << 40);
            out.push(c.to_bits());
        }
        out.push(u64::MAX);
    };
    match op {
        POp::Fixed(g) => {
            let (c, angles) = gate_code(g);
            out.push(u64::from(c));
            if with_angles {
                for a in angles {
                    out.push(a.to_bits());
                }
            }
        }
        _ => {
            let c: u64 = match op {
                POp::Rx(..) => 100,
                POp::Ry(..) => 101,
                POp::Rz(..) => 102,
                POp::Phase(..) => 103,
                POp::CPhase(..) => 104,
                POp::Rzz(..) => 105,
                POp::Rxx(..) => 106,
                POp::U(..) => 107,
                POp::ZString(qs, _) => 200 + qs.len() as u64,
                POp::Global(_) => 108,
                POp::Fixed(_) => unreachable!(),
            };
            out.push(c);
            if with_angles {
                for a in op.angles() {
                    push_angle(a, out);
                }
            }
        }
    }
}

/// Cuts `pc` into blocks of at most `k` qubits and classifies them.
pub fn analyse(pc: &ParamCircuit, k: usize) -> Dedup {
    let t0 = std::time::Instant::now();
    let n = pc.num_qubits;
    // greedy blocking along the wire order
    let mut open: Vec<Option<usize>> = vec![None; n]; // qubit -> open block
    let mut blocks: Vec<(Vec<usize>, Vec<usize>)> = Vec::new(); // (ops, qubits in first-use order)
    let mut alive: Vec<bool> = Vec::new();
    for (i, op) in pc.ops.iter().enumerate() {
        let qs = op.qubits();
        if qs.is_empty() {
            continue;
        }
        let mut bs: Vec<usize> = qs.iter().filter_map(|&q| open[q]).collect();
        bs.sort_unstable();
        bs.dedup();
        let mut union: Vec<usize> = Vec::new();
        for &b in &bs {
            for &q in &blocks[b].1 {
                if !union.contains(&q) {
                    union.push(q);
                }
            }
        }
        for &q in &qs {
            if !union.contains(&q) {
                union.push(q);
            }
        }
        if union.len() <= k && !bs.is_empty() {
            // merge every open block on these qubits into the first one
            let b0 = bs[0];
            for &b in &bs[1..] {
                let (ops, qubits) = std::mem::take(&mut blocks[b]);
                alive[b] = false;
                blocks[b0].0.extend(ops);
                for q in qubits {
                    if !blocks[b0].1.contains(&q) {
                        blocks[b0].1.push(q);
                    }
                    open[q] = Some(b0);
                }
            }
            blocks[b0].0.push(i);
            for &q in &qs {
                if !blocks[b0].1.contains(&q) {
                    blocks[b0].1.push(q);
                }
                open[q] = Some(b0);
            }
        } else {
            // close the blocks on these qubits; start a new one
            for &b in &bs {
                for &q in &blocks[b].1.clone() {
                    if open[q] == Some(b) {
                        open[q] = None;
                    }
                }
            }
            let b = blocks.len();
            blocks.push((vec![i], qs.clone()));
            alive.push(true);
            for &q in &qs {
                open[q] = Some(b);
            }
        }
    }
    // canonical keys
    let mut class_of: HashMap<Vec<u64>, usize> = HashMap::new();
    let mut shape_of: HashMap<Vec<u64>, usize> = HashMap::new();
    let mut out = Dedup {
        total_ops: pc.ops.iter().filter(|o| !o.qubits().is_empty()).count(),
        ..Default::default()
    };
    for (b, (mut ops, _)) in blocks.into_iter().enumerate() {
        if !alive[b] {
            continue;
        }
        ops.sort_unstable(); // merged blocks: program order (disjoint parts commute)
        let mut canon: Vec<usize> = Vec::new();
        let mut key = Vec::new();
        let mut skey = Vec::new();
        for &i in &ops {
            let op = &pc.ops[i];
            for q in op.qubits() {
                let j = match canon.iter().position(|&x| x == q) {
                    Some(j) => j,
                    None => {
                        canon.push(q);
                        canon.len() - 1
                    }
                };
                key.push(j as u64);
                skey.push(j as u64);
            }
            op_key(op, true, &mut key);
            op_key(op, false, &mut skey);
        }
        let nc = class_of.len();
        let class = *class_of.entry(key).or_insert(nc);
        let ns = shape_of.len();
        let shape = *shape_of.entry(skey).or_insert(ns);
        out.blocks.push(Block {
            ops,
            qubits: canon,
            class,
            shape,
        });
    }
    out.num_classes = class_of.len();
    out.num_shapes = shape_of.len();
    out.class_count = vec![0; out.num_classes];
    out.shape_count = vec![0; out.num_shapes];
    for b in &out.blocks {
        out.class_count[b.class] += 1;
        out.shape_count[b.shape] += 1;
    }
    for b in &out.blocks {
        if out.class_count[b.class] >= 2 {
            out.covered_ops += b.ops.len();
        }
        if out.shape_count[b.shape] >= 2 {
            out.covered_ops_shape += b.ops.len();
        }
    }
    out.secs = t0.elapsed().as_secs_f64();
    out
}

/// The `2^k × 2^k` unitary of a block on its canonical qubits (row-major,
/// local bit `j` = canonical qubit `j`), at `params`.
pub fn block_unitary(pc: &ParamCircuit, b: &Block, params: &[f64]) -> Vec<Complex64> {
    let k = b.qubits.len();
    let d = 1usize << k;
    let local: HashMap<usize, usize> = b.qubits.iter().enumerate().map(|(j, &q)| (q, j)).collect();
    let mut gates = Vec::new();
    for &i in &b.ops {
        pc.ops[i]
            .map_qubits(|q| local[&q])
            .bind_into(params, &mut gates);
    }
    let mut u = vec![Complex64::new(0.0, 0.0); d * d];
    // column x = U|x>
    for x in 0..d {
        let mut sv = crate::statevector::StateVectorF64::basis_state(k, x);
        for g in &gates {
            sv.apply_gate(g).expect("valid gate");
        }
        for (y, a) in sv.amplitudes().iter().enumerate() {
            u[y * d + x] = *a;
        }
    }
    u
}

/// Block unitaries for every block: computed per instance (`dedup =
/// false`) or once per class and shared (`dedup = true`). Returns the
/// unitaries (shared by class) and the number actually computed.
pub fn block_unitaries(
    pc: &ParamCircuit,
    d: &Dedup,
    params: &[f64],
    dedup: bool,
) -> (Vec<std::sync::Arc<Vec<Complex64>>>, usize) {
    let mut cache: HashMap<usize, std::sync::Arc<Vec<Complex64>>> = HashMap::new();
    let mut computed = 0;
    let out = d
        .blocks
        .iter()
        .map(|b| {
            if dedup {
                if let Some(u) = cache.get(&b.class) {
                    return u.clone();
                }
            }
            computed += 1;
            let u = std::sync::Arc::new(block_unitary(pc, b, params));
            if dedup {
                cache.insert(b.class, u.clone());
            }
            u
        })
        .collect();
    (out, computed)
}
