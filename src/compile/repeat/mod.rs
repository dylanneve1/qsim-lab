//! Exact exploitation of repeated blocks: `circuit = prefix · B^r · suffix`.
//!
//! * [`detect`] finds repeats (also nested, and *parameterised* ones where
//!   the gates agree and only the angles differ) in two ways and keeps the
//!   one that covers more gates:
//!   1. tandem repeats of the op sequence as written;
//!   2. tandem repeats of the *canonical layer sequence*: every maximal
//!      unitary run is cut into ASAP layers whose gates are sorted, so
//!      commuting reorderings of a block still match.
//!      Every such sequence is a topological order of the same dependency
//!      DAG, so the rewritten program is the same circuit.
//! * [`cliff`]: exact fast paths for Clifford blocks (symplectic map raised to
//!   the `r`-th power by squaring, Clifford synthesis, steady-state skipping of
//!   deterministic measurement rounds).
//! * [`exec`]: dense fast paths: diagonal folding, `2^k` unitary powers and
//!   compile-once blocked plans.
//!
//! Everything is exact up to floating-point rounding; nothing is on by
//! default (see `pipeline::SimOptions`).

pub mod cliff;
pub mod exec;
pub mod workloads;

use crate::circuit::{Circuit, Op};
use crate::gate::Gate;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A node of a [`Program`].
#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    /// Plain operations.
    Ops(Vec<Op>),
    /// `body` repeated `reps` times, bit-identical copies.
    Repeat { body: Vec<Node>, reps: usize },
    /// `reps` copies of the same gate sequence with different angles.
    /// `shape` is the first copy; `angles[k]` lists the angles of copy `k`
    /// (in op order, as returned by [`op_angles`]).
    Param {
        shape: Vec<Op>,
        reps: usize,
        angles: Vec<Vec<f64>>,
    },
}

/// A circuit as a tree of plain ops and repeated blocks.
#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    pub num_qubits: usize,
    pub nodes: Vec<Node>,
}

/// Tuning knobs of [`detect`].
#[derive(Clone, Debug)]
pub struct DetectOptions {
    /// Longest period (in ops, or in layers) considered.
    pub max_period: usize,
    /// A repeat must cover at least this many ops (all copies together).
    pub min_ops: usize,
    /// Also try the canonical layer form.
    pub layered: bool,
    /// Also find repeats whose copies differ only in angles.
    pub parameterised: bool,
}

impl Default for DetectOptions {
    fn default() -> Self {
        DetectOptions {
            max_period: 4096,
            min_ops: 8,
            layered: true,
            parameterised: true,
        }
    }
}

/// How much of a program sits inside repeats.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Report {
    pub total_ops: usize,
    pub total_gates: usize,
    /// Gates inside a repeat, all copies counted.
    pub covered_gates: usize,
    /// Gates a repeat lets a simulator skip (all copies but the first).
    pub saved_gates: usize,
    pub repeats: usize,
    pub param_repeats: usize,
    /// `(period in ops, reps, parameterised)` of the top-level repeats.
    pub blocks: Vec<(usize, usize, bool)>,
}

impl Report {
    pub fn coverage(&self) -> f64 {
        if self.total_gates == 0 {
            0.0
        } else {
            self.covered_gates as f64 / self.total_gates as f64
        }
    }
}

// ---------------------------------------------------------------- op keys

/// `(kind, qubits, angles)` of a gate; angle slots unused are 0.
fn gate_parts(g: &Gate) -> (u8, [usize; 3], [f64; 3]) {
    use Gate::*;
    match *g {
        I(q) => (0, [q, 0, 0], [0.0; 3]),
        H(q) => (1, [q, 0, 0], [0.0; 3]),
        X(q) => (2, [q, 0, 0], [0.0; 3]),
        Y(q) => (3, [q, 0, 0], [0.0; 3]),
        Z(q) => (4, [q, 0, 0], [0.0; 3]),
        S(q) => (5, [q, 0, 0], [0.0; 3]),
        Sdg(q) => (6, [q, 0, 0], [0.0; 3]),
        T(q) => (7, [q, 0, 0], [0.0; 3]),
        Tdg(q) => (8, [q, 0, 0], [0.0; 3]),
        Sx(q) => (9, [q, 0, 0], [0.0; 3]),
        Sxdg(q) => (10, [q, 0, 0], [0.0; 3]),
        Rx(q, a) => (11, [q, 0, 0], [a, 0.0, 0.0]),
        Ry(q, a) => (12, [q, 0, 0], [a, 0.0, 0.0]),
        Rz(q, a) => (13, [q, 0, 0], [a, 0.0, 0.0]),
        Phase(q, a) => (14, [q, 0, 0], [a, 0.0, 0.0]),
        U(q, a, b, c) => (15, [q, 0, 0], [a, b, c]),
        Cnot(a, b) => (16, [a, b, 0], [0.0; 3]),
        Cz(a, b) => (17, [a, b, 0], [0.0; 3]),
        Swap(a, b) => (18, [a, b, 0], [0.0; 3]),
        ISwap(a, b) => (19, [a, b, 0], [0.0; 3]),
        ISwapdg(a, b) => (20, [a, b, 0], [0.0; 3]),
        CPhase(a, b, t) => (21, [a, b, 0], [t, 0.0, 0.0]),
        Ccx(a, b, t) => (22, [a, b, t], [0.0; 3]),
    }
}

/// Number of angle parameters of a gate.
fn gate_nangles(g: &Gate) -> usize {
    use Gate::*;
    match g {
        Rx(..) | Ry(..) | Rz(..) | Phase(..) | CPhase(..) => 1,
        U(..) => 3,
        _ => 0,
    }
}

fn gate_with_angles(g: &Gate, a: &[f64]) -> Gate {
    use Gate::*;
    match *g {
        Rx(q, _) => Rx(q, a[0]),
        Ry(q, _) => Ry(q, a[0]),
        Rz(q, _) => Rz(q, a[0]),
        Phase(q, _) => Phase(q, a[0]),
        CPhase(p, q, _) => CPhase(p, q, a[0]),
        U(q, ..) => U(q, a[0], a[1], a[2]),
        other => other,
    }
}

/// The angles of an op (empty unless it is a plain parameterised gate).
pub fn op_angles(op: &Op) -> Vec<f64> {
    match op {
        Op::Gate(g) => {
            let (_, _, a) = gate_parts(g);
            a[..gate_nangles(g)].to_vec()
        }
        _ => Vec::new(),
    }
}

/// The op with its angles replaced (identity if it has none).
pub fn op_with_angles(op: &Op, a: &[f64]) -> Op {
    match op {
        Op::Gate(g) => Op::Gate(gate_with_angles(g, a)),
        other => *other,
    }
}

/// Hash key of an op; with `shape` the angles of plain gates are dropped.
fn op_hash(op: &Op, shape: bool) -> u64 {
    let mut h = DefaultHasher::new();
    match op {
        Op::Gate(g) => {
            let (k, q, a) = gate_parts(g);
            (0u8, k, q).hash(&mut h);
            if !shape {
                for x in a {
                    x.to_bits().hash(&mut h);
                }
            }
        }
        Op::Measure(q) => (1u8, q).hash(&mut h),
        Op::Reset(q) => (2u8, q).hash(&mut h),
        Op::ClassicControlled {
            gate,
            meas_index,
            target_value,
        } => {
            let (k, q, a) = gate_parts(gate);
            (3u8, k, q, meas_index, target_value).hash(&mut h);
            for x in a {
                x.to_bits().hash(&mut h);
            }
        }
        Op::XFlip(q, p) => (4u8, q, p.to_bits()).hash(&mut h),
        Op::YFlip(q, p) => (5u8, q, p.to_bits()).hash(&mut h),
        Op::ZFlip(q, p) => (6u8, q, p.to_bits()).hash(&mut h),
        Op::Depolarize1q(q, p) => (7u8, q, p.to_bits()).hash(&mut h),
        Op::Depolarize2q(a, b, p) => (8u8, a, b, p.to_bits()).hash(&mut h),
    }
    h.finish()
}

fn elem_hash(e: &[Op], shape: bool) -> u64 {
    let mut h = DefaultHasher::new();
    for op in e {
        op_hash(op, shape).hash(&mut h);
    }
    h.finish()
}

fn same_shape(a: &Op, b: &Op) -> bool {
    match (a, b) {
        (Op::Gate(x), Op::Gate(y)) => {
            let (kx, qx, _) = gate_parts(x);
            let (ky, qy, _) = gate_parts(y);
            kx == ky && qx == qy
        }
        _ => a == b,
    }
}

fn elems_same_shape(a: &[Op], b: &[Op]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same_shape(x, y))
}

// ------------------------------------------------------ canonical layers

/// Cuts the circuit into canonical elements: maximal unitary runs become
/// ASAP layers (gates sorted by `(kind, qubits)`), every other op is its own
/// element.
fn canonical_layers(c: &Circuit) -> Vec<Vec<Op>> {
    fn flush(run: &mut Vec<Gate>, n: usize, out: &mut Vec<Vec<Op>>) {
        if run.is_empty() {
            return;
        }
        let mut level = vec![0usize; n];
        let mut layers: Vec<Vec<Gate>> = Vec::new();
        for g in run.drain(..) {
            let (_, qs, _) = gate_parts(&g);
            let k = g.arity();
            let l = qs[..k].iter().map(|&q| level[q]).max().unwrap_or(0);
            for &q in &qs[..k] {
                level[q] = l + 1;
            }
            if layers.len() <= l {
                layers.push(Vec::new());
            }
            layers[l].push(g);
        }
        for mut layer in layers {
            layer.sort_by_key(|g| {
                let (k, q, _) = gate_parts(g);
                (q, k)
            });
            out.push(layer.into_iter().map(Op::Gate).collect());
        }
    }
    let mut out = Vec::new();
    let mut run = Vec::new();
    for op in &c.ops {
        match op {
            Op::Gate(g) => run.push(*g),
            other => {
                flush(&mut run, c.num_qubits, &mut out);
                out.push(vec![*other]);
            }
        }
    }
    flush(&mut run, c.num_qubits, &mut out);
    out
}

// ------------------------------------------------------ tandem repeats

#[derive(Clone, Copy, Debug)]
struct Rep {
    start: usize,
    p: usize,
    reps: usize,
    cover: usize,
}

/// Maximal tandem repeats (`reps >= 2`) of `h`, non-overlapping, chosen
/// greedily by covered ops (ties: shorter period). `eq` verifies candidates
/// exactly, so a hash collision cannot produce a wrong repeat.
fn find_repeats(
    h: &[u64],
    w: &[usize],
    opts: &DetectOptions,
    eq: &dyn Fn(usize, usize) -> bool,
) -> Vec<Rep> {
    let n = h.len();
    let pmax = opts
        .max_period
        .min(n / 2)
        .min(if n > 0 { 600_000_000 / n.max(1) } else { 0 });
    let mut cands: Vec<Rep> = Vec::new();
    for p in 1..=pmax {
        let mut i = 0;
        while i + p < n {
            if h[i] == h[i + p] {
                let a = i;
                while i + p < n && h[i] == h[i + p] {
                    i += 1;
                }
                let len = i - a;
                if len >= p {
                    let reps = (len + p) / p;
                    let cover = w[a + reps * p] - w[a];
                    if cover >= opts.min_ops {
                        cands.push(Rep {
                            start: a,
                            p,
                            reps,
                            cover,
                        });
                    }
                }
            } else {
                i += 1;
            }
        }
    }
    cands.sort_by(|x, y| y.cover.cmp(&x.cover).then(x.p.cmp(&y.p)));
    let mut taken: Vec<Rep> = Vec::new();
    for c in cands {
        let (a, b) = (c.start, c.start + c.p * c.reps);
        if taken
            .iter()
            .any(|t| a < t.start + t.p * t.reps && t.start < b)
        {
            continue;
        }
        // exact verification
        if (a..b - c.p).any(|j| !eq(j, j + c.p)) {
            continue;
        }
        taken.push(c);
    }
    taken.sort_by_key(|t| t.start);
    taken
}

fn prefix_weights(elems: &[Vec<Op>]) -> Vec<usize> {
    let mut w = Vec::with_capacity(elems.len() + 1);
    w.push(0);
    for e in elems {
        w.push(w.last().unwrap() + e.len());
    }
    w
}

fn flatten(elems: &[Vec<Op>]) -> Vec<Op> {
    elems.iter().flatten().copied().collect()
}

/// Exact (bit-identical copies) repeat tree over `elems`.
fn build_exact(elems: &[Vec<Op>], opts: &DetectOptions) -> Vec<Node> {
    if elems.len() < 2 {
        return vec![Node::Ops(flatten(elems))];
    }
    let h: Vec<u64> = elems.iter().map(|e| elem_hash(e, false)).collect();
    let w = prefix_weights(elems);
    let reps = find_repeats(&h, &w, opts, &|a, b| elems[a] == elems[b]);
    let mut out = Vec::new();
    let mut pos = 0;
    for r in reps {
        if r.start > pos {
            out.push(Node::Ops(flatten(&elems[pos..r.start])));
        }
        let body = build_exact(&elems[r.start..r.start + r.p], opts);
        out.push(Node::Repeat { body, reps: r.reps });
        pos = r.start + r.p * r.reps;
    }
    if pos < elems.len() {
        out.push(Node::Ops(flatten(&elems[pos..])));
    }
    out.retain(|n| !matches!(n, Node::Ops(o) if o.is_empty()));
    out
}

/// Second pass over plain-op stretches: repeats whose copies differ in angles.
fn param_pass(nodes: Vec<Node>, layered: bool, n: usize, opts: &DetectOptions) -> Vec<Node> {
    let mut out = Vec::new();
    for node in nodes {
        match node {
            Node::Ops(ops) if ops.len() >= 2 * opts.min_ops.max(2) => {
                let tmp = Circuit { num_qubits: n, ops };
                let elems: Vec<Vec<Op>> = if layered {
                    canonical_layers(&tmp)
                } else {
                    tmp.ops.iter().map(|o| vec![*o]).collect()
                };
                let h: Vec<u64> = elems.iter().map(|e| elem_hash(e, true)).collect();
                let w = prefix_weights(&elems);
                let reps =
                    find_repeats(&h, &w, opts, &|a, b| elems_same_shape(&elems[a], &elems[b]));
                let mut pos = 0;
                for r in reps {
                    if r.start > pos {
                        out.push(Node::Ops(flatten(&elems[pos..r.start])));
                    }
                    let copies: Vec<Vec<Op>> = (0..r.reps)
                        .map(|k| flatten(&elems[r.start + k * r.p..r.start + (k + 1) * r.p]))
                        .collect();
                    let shape = copies[0].clone();
                    let angles: Vec<Vec<f64>> = copies
                        .iter()
                        .map(|c| c.iter().flat_map(op_angles).collect())
                        .collect();
                    if angles.iter().all(|a| *a == angles[0]) {
                        out.push(Node::Repeat {
                            body: vec![Node::Ops(shape)],
                            reps: r.reps,
                        });
                    } else {
                        out.push(Node::Param {
                            shape,
                            reps: r.reps,
                            angles,
                        });
                    }
                    pos = r.start + r.p * r.reps;
                }
                if pos < elems.len() {
                    out.push(Node::Ops(flatten(&elems[pos..])));
                }
            }
            other => out.push(other),
        }
    }
    out.retain(|n| !matches!(n, Node::Ops(o) if o.is_empty()));
    out
}

/// Detects repeated blocks in `c`.
pub fn detect(c: &Circuit, opts: &DetectOptions) -> Program {
    let n = c.num_qubits;
    let flat: Vec<Vec<Op>> = c.ops.iter().map(|o| vec![*o]).collect();
    let mut candidates: Vec<Program> = Vec::new();
    let mut variants: Vec<(bool, Vec<Vec<Op>>)> = vec![(false, flat)];
    if opts.layered {
        variants.push((true, canonical_layers(c)));
    }
    for (layered, elems) in variants {
        let mut nodes = build_exact(&elems, opts);
        if opts.parameterised {
            nodes = param_pass(nodes, layered, n, opts);
        }
        candidates.push(Program {
            num_qubits: n,
            nodes,
        });
    }
    candidates
        .into_iter()
        .max_by(|a, b| {
            let (ra, rb) = (a.report(), b.report());
            ra.saved_gates
                .cmp(&rb.saved_gates)
                // prefer the program in the original order on ties
                .then(std::cmp::Ordering::Greater)
        })
        .expect("at least one candidate")
}

impl Node {
    fn count(&self) -> (usize, usize) {
        // (ops in one copy, gates in one copy)
        match self {
            Node::Ops(o) => (
                o.len(),
                o.iter().filter(|x| matches!(x, Op::Gate(_))).count(),
            ),
            Node::Repeat { body, reps } => {
                let (a, b) = body
                    .iter()
                    .map(|n| n.count())
                    .fold((0, 0), |s, x| (s.0 + x.0, s.1 + x.1));
                (a * reps, b * reps)
            }
            Node::Param { shape, reps, .. } => {
                let g = shape.iter().filter(|x| matches!(x, Op::Gate(_))).count();
                (shape.len() * reps, g * reps)
            }
        }
    }
}

impl Program {
    /// Plain program (no repeats) of a circuit.
    pub fn plain(c: &Circuit) -> Program {
        Program {
            num_qubits: c.num_qubits,
            nodes: vec![Node::Ops(c.ops.clone())],
        }
    }

    /// Expands every repeat (materialises `r` copies).
    pub fn to_circuit(&self) -> Circuit {
        fn go(nodes: &[Node], out: &mut Vec<Op>) {
            for n in nodes {
                match n {
                    Node::Ops(o) => out.extend_from_slice(o),
                    Node::Repeat { body, reps } => {
                        for _ in 0..*reps {
                            go(body, out);
                        }
                    }
                    Node::Param {
                        shape,
                        reps,
                        angles,
                    } => {
                        for ang in angles.iter().take(*reps) {
                            let mut it = ang.iter().copied();
                            for op in shape {
                                let na = op_angles(op).len();
                                let a: Vec<f64> = it.by_ref().take(na).collect();
                                out.push(op_with_angles(op, &a));
                            }
                        }
                    }
                }
            }
        }
        let mut ops = Vec::new();
        go(&self.nodes, &mut ops);
        Circuit {
            num_qubits: self.num_qubits,
            ops,
        }
    }

    /// Coverage statistics. Gates are counted at the outermost repeat.
    pub fn report(&self) -> Report {
        fn gates(nodes: &[Node]) -> usize {
            nodes.iter().map(|n| n.count().1).sum()
        }
        fn ops(nodes: &[Node]) -> usize {
            nodes.iter().map(|n| n.count().0).sum()
        }
        fn inner(nodes: &[Node], r: &mut Report) {
            for n in nodes {
                if let Node::Repeat { body, .. } = n {
                    r.repeats += 1;
                    inner(body, r);
                }
            }
        }
        let mut r = Report {
            total_ops: ops(&self.nodes),
            total_gates: gates(&self.nodes),
            ..Report::default()
        };
        for n in &self.nodes {
            match n {
                Node::Repeat { body, reps } => {
                    let one = gates(body);
                    r.covered_gates += one * reps;
                    r.saved_gates += one * (reps - 1);
                    r.repeats += 1;
                    r.blocks.push((ops(body), *reps, false));
                    inner(body, &mut r);
                }
                Node::Param { shape, reps, .. } => {
                    let one = shape.iter().filter(|x| matches!(x, Op::Gate(_))).count();
                    r.covered_gates += one * reps;
                    r.saved_gates += one * (reps - 1);
                    r.param_repeats += 1;
                    r.blocks.push((shape.len(), *reps, true));
                }
                Node::Ops(_) => {}
            }
        }
        r
    }
}

/// Result of [`rewrite`].
#[derive(Clone, Debug)]
pub struct Rewrite {
    pub circuit: Circuit,
    /// Global phase of `circuit` relative to the program (exact; only
    /// meaningful if `phase_exact`).
    pub global_phase: f64,
    /// False if a Clifford block was replaced by its synthesised power
    /// (equal only up to a global phase).
    pub phase_exact: bool,
    pub diag_collapsed: usize,
    pub clifford_collapsed: usize,
    /// Repeats that had no applicable fast path and were expanded.
    pub expanded: usize,
}

/// Rewrites a program into an ordinary circuit, replacing diagonal repeats
/// by their folded phase layer and (if `allow_clifford`) Clifford repeats by
/// the synthesised gates of their symplectic power when that is shorter.
/// Other repeats are expanded copy by copy.
pub fn rewrite(p: &Program, allow_clifford: bool) -> Rewrite {
    use cliff::power_gates;
    use exec::{diag_power, DiagPoly};
    struct St {
        ops: Vec<Op>,
        phase: f64,
        exact: bool,
        diag: usize,
        cliff: usize,
        expanded: usize,
    }
    fn gates_of(nodes: &[Node]) -> Option<Vec<Gate>> {
        let mut out = Vec::new();
        for n in nodes {
            match n {
                Node::Ops(o) => {
                    for op in o {
                        match op {
                            Op::Gate(g) => out.push(*g),
                            _ => return None,
                        }
                    }
                }
                Node::Repeat { body, reps } => {
                    let g = gates_of(body)?;
                    if g.len().saturating_mul(*reps) > 1 << 22 {
                        return None;
                    }
                    for _ in 0..*reps {
                        out.extend_from_slice(&g);
                    }
                }
                Node::Param { .. } => return None,
            }
        }
        Some(out)
    }
    fn go(nodes: &[Node], allow: bool, st: &mut St) {
        for n in nodes {
            match n {
                Node::Ops(o) => st.ops.extend_from_slice(o),
                Node::Repeat { body, reps } => {
                    if let Some(g) = gates_of(body) {
                        if let Some((dg, ph)) = diag_power(&g, *reps as u64) {
                            st.ops.extend(dg.into_iter().map(Op::Gate));
                            st.phase += ph;
                            st.diag += 1;
                            continue;
                        }
                        if allow && g.iter().all(|x| x.is_clifford()) {
                            let k = cliff::compact(&g).0.len();
                            if g.len() * reps > 4 * (k * k + 8) {
                                if let Some(pg) = power_gates(&g, *reps as u64) {
                                    if pg.len() < g.len() * reps {
                                        st.ops.extend(pg.into_iter().map(Op::Gate));
                                        st.exact = false;
                                        st.cliff += 1;
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                    st.expanded += 1;
                    for _ in 0..*reps {
                        go(body, allow, st);
                    }
                }
                Node::Param {
                    shape,
                    reps,
                    angles,
                } => {
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
                    let mut d = DiagPoly::default();
                    let ok = (0..*reps).all(|k| {
                        copy(k).iter().all(|op| match op {
                            Op::Gate(g) => d.add(g),
                            _ => false,
                        })
                    });
                    if ok {
                        let (g, ph) = d.scaled_gates(1.0);
                        st.ops.extend(g.into_iter().map(Op::Gate));
                        st.phase += ph;
                        st.diag += 1;
                    } else {
                        st.expanded += 1;
                        for k in 0..*reps {
                            st.ops.extend(copy(k));
                        }
                    }
                }
            }
        }
    }
    let mut st = St {
        ops: Vec::new(),
        phase: 0.0,
        exact: true,
        diag: 0,
        cliff: 0,
        expanded: 0,
    };
    go(&p.nodes, allow_clifford, &mut st);
    Rewrite {
        circuit: Circuit {
            num_qubits: p.num_qubits,
            ops: st.ops,
        },
        global_phase: st.phase,
        phase_exact: st.exact,
        diag_collapsed: st.diag,
        clifford_collapsed: st.cliff,
        expanded: st.expanded,
    }
}
