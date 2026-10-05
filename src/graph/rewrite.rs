//! Phase-polynomial region rewriting ("fuse monomial runs").
//!
//! A *region* is a maximal run of ops that are affine permutations
//! (`X`, `CNOT`, `SWAP`, the `X` part of `Y`) or diagonal (`Z`, `S`, `T`,
//! `Rz`, `Phase`, `CZ`, `CPhase`, `Rzz`, phase gadgets). Its unitary is a
//! permutation times a diagonal: tracking every wire as an affine parity
//! `P_w · x ⊕ c_w` of the region's *input* bits, every diagonal op becomes a
//! sum of parity rotations `exp(-i α/2 (-1)^{p·x})` (merged per parity: phase
//! folding) and
//!
//! ```text
//! region = N · D(x),   D = Π_p exp(-i α_p/2 Z^{⊗p})  (gadgets on the inputs)
//! ```
//!
//! where `N` is the permutation part. `N` is re-emitted as `X`s when its
//! linear part is the identity (a CNOT ladder and its inverse cancel), as
//! SWAPs when it is a wire permutation, and as the original permutation ops
//! otherwise. Each gadget runs as `2^(k-1)` diagonal terms in the blocked
//! executor (no CNOT passes).
//!
//! Typical wins: Pauli-gadget circuits (UCC ansätze, Trotterised Pauli
//! Hamiltonians: `ladder · Rz · ladder⁻¹`), QAOA written as `CNOT Rz CNOT`,
//! and phase folding of repeated rotations on the same parity.
//!
//! Before the region pass the ops can be re-scheduled (any topological order
//! of the wire DAG gives the same unitary): ready region ops are emitted
//! before ready non-region ops so regions grow as large as possible.

use super::param::{Angle, POp, ParamCircuit};
use crate::gate::Gate;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Options of [`phase_regions`].
#[derive(Clone, Debug)]
pub struct RewriteOptions {
    /// Re-schedule so region ops are contiguous.
    pub reorder: bool,
    /// Gadgets wider than this are costed as CNOT ladders.
    pub max_weight: usize,
}

impl Default for RewriteOptions {
    fn default() -> Self {
        RewriteOptions {
            reorder: true,
            max_weight: 6,
        }
    }
}

/// What the pass did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RewriteStats {
    /// Phase-polynomial regions examined.
    pub regions: usize,
    /// Regions actually replaced (the rewrite had to be strictly better).
    pub rewritten: usize,
    /// Permutation ops in all examined regions before the pass.
    pub perm_ops_before: usize,
    /// Permutation-network cost after the pass (unchanged regions counted at their old size).
    pub perm_ops_after: usize,
    /// Diagonal ops in all examined regions before the pass.
    pub diag_ops_before: usize,
    /// Phase gadgets emitted by the rewritten regions.
    pub gadgets_after: usize,
}

enum Kind {
    /// Permutation part.
    Perm,
    /// Diagonal part.
    Diag,
    /// `Y = i X Z`: both.
    Y,
    Other,
}

fn kind(op: &POp) -> Kind {
    match op {
        POp::Fixed(g) => match g {
            Gate::X(_) | Gate::Cnot(..) | Gate::Swap(..) => Kind::Perm,
            Gate::Y(_) => Kind::Y,
            Gate::I(_)
            | Gate::Z(_)
            | Gate::S(_)
            | Gate::Sdg(_)
            | Gate::T(_)
            | Gate::Tdg(_)
            | Gate::Rz(..)
            | Gate::Phase(..)
            | Gate::Cz(..)
            | Gate::CPhase(..) => Kind::Diag,
            _ => Kind::Other,
        },
        POp::Rz(..) | POp::Phase(..) | POp::CPhase(..) | POp::Rzz(..) | POp::ZString(..) => {
            Kind::Diag
        }
        _ => Kind::Other,
    }
}

fn is_region(op: &POp) -> bool {
    !matches!(kind(op), Kind::Other)
}

/// Decomposes a diagonal op into parity rotations `(wires, α)` meaning
/// `exp(-i α/2 (-1)^{⊕ wires})`, plus a global phase angle.
fn diag_terms(op: &POp, out: &mut Vec<(Vec<usize>, Angle)>, global: &mut Angle) {
    let c = Angle::constant;
    // e^{iφ v} for the bit v = ⊕ws: global φ/2, rotation (ws, φ)
    let mut bitphase = |ws: Vec<usize>, phi: Angle, out: &mut Vec<(Vec<usize>, Angle)>| {
        *global = global.plus(&phi.times(0.5));
        out.push((ws, phi));
    };
    match op {
        POp::Fixed(g) => match *g {
            Gate::I(_) => {}
            Gate::Z(q) => bitphase(vec![q], c(PI), out),
            Gate::S(q) => bitphase(vec![q], c(PI / 2.0), out),
            Gate::Sdg(q) => bitphase(vec![q], c(-PI / 2.0), out),
            Gate::T(q) => bitphase(vec![q], c(PI / 4.0), out),
            Gate::Tdg(q) => bitphase(vec![q], c(-PI / 4.0), out),
            Gate::Phase(q, t) => bitphase(vec![q], c(t), out),
            Gate::Rz(q, t) => out.push((vec![q], c(t))),
            Gate::Cz(a, b) => cphase(a, b, &c(PI), out, global),
            Gate::CPhase(a, b, t) => cphase(a, b, &c(t), out, global),
            _ => unreachable!("not diagonal"),
        },
        POp::Rz(q, a) => out.push((vec![*q], a.clone())),
        POp::Phase(q, a) => bitphase(vec![*q], a.clone(), out),
        POp::CPhase(x, y, a) => cphase(*x, *y, a, out, global),
        POp::Rzz(x, y, a) => out.push((vec![*x, *y], a.clone())),
        POp::ZString(qs, a) => out.push((qs.clone(), a.clone())),
        _ => unreachable!("not diagonal"),
    }
}

/// `e^{iθ ab} = e^{iθ/2 a} e^{iθ/2 b} e^{-iθ/2 (a⊕b)}`.
fn cphase(a: usize, b: usize, t: &Angle, out: &mut Vec<(Vec<usize>, Angle)>, global: &mut Angle) {
    let h = t.times(0.5);
    *global = global.plus(&t.times(0.25));
    out.push((vec![a], h.clone()));
    out.push((vec![b], h.clone()));
    out.push((vec![a, b], h.times(-1.0)));
}

/// Greedy topological re-schedule: emit every ready region op before any
/// ready non-region op.
fn reorder(pc: &ParamCircuit) -> Vec<POp> {
    let n = pc.num_qubits;
    let m = pc.ops.len();
    // predecessor counts over the wire DAG
    let mut last: Vec<Option<usize>> = vec![None; n];
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); m];
    let mut indeg = vec![0usize; m];
    for (i, op) in pc.ops.iter().enumerate() {
        let mut preds: Vec<usize> = op.qubits().iter().filter_map(|&q| last[q]).collect();
        preds.sort_unstable();
        preds.dedup();
        for p in preds {
            succ[p].push(i);
            indeg[i] += 1;
        }
        for q in op.qubits() {
            last[q] = Some(i);
        }
    }
    let mut ready_r: std::collections::BinaryHeap<std::cmp::Reverse<usize>> = Default::default();
    let mut ready_o: Vec<usize> = Vec::new();
    for (i, &d) in indeg.iter().enumerate() {
        if d == 0 {
            if is_region(&pc.ops[i]) {
                ready_r.push(std::cmp::Reverse(i));
            } else {
                ready_o.push(i);
            }
        }
    }
    let mut out = Vec::with_capacity(m);
    let release = |i: usize,
                   indeg: &mut Vec<usize>,
                   ready_r: &mut std::collections::BinaryHeap<std::cmp::Reverse<usize>>,
                   ready_o: &mut Vec<usize>| {
        for &s in &succ[i] {
            indeg[s] -= 1;
            if indeg[s] == 0 {
                if is_region(&pc.ops[s]) {
                    ready_r.push(std::cmp::Reverse(s));
                } else {
                    ready_o.push(s);
                }
            }
        }
    };
    while out.len() < m {
        if let Some(std::cmp::Reverse(i)) = ready_r.pop() {
            out.push(pc.ops[i].clone());
            release(i, &mut indeg, &mut ready_r, &mut ready_o);
        } else {
            let batch = std::mem::take(&mut ready_o);
            let mut batch = batch;
            batch.sort_unstable();
            for i in batch {
                out.push(pc.ops[i].clone());
                release(i, &mut indeg, &mut ready_r, &mut ready_o);
            }
        }
    }
    out
}

fn bits(m: u128) -> Vec<usize> {
    (0..128).filter(|&q| m >> q & 1 == 1).collect()
}

/// One region: returns the replacement ops if they are cheaper.
fn rewrite_region(
    n: usize,
    ops: &[POp],
    opts: &RewriteOptions,
    st: &mut RewriteStats,
) -> Option<Vec<POp>> {
    let mut p: Vec<u128> = (0..n).map(|w| 1u128 << w).collect();
    let mut c = vec![false; n];
    let mut terms: HashMap<u128, Angle> = HashMap::new();
    let mut order: Vec<u128> = Vec::new(); // first-seen order of parities
    let mut global = Angle::constant(0.0);
    let mut perm_ops: Vec<POp> = Vec::new();
    let mut diag_before = 0usize;
    let mut tmp = Vec::new();
    let mut add_rot = |ws: &[usize],
                       a: &Angle,
                       p: &[u128],
                       c: &[bool],
                       terms: &mut HashMap<u128, Angle>,
                       global: &mut Angle| {
        let (mut mask, mut cb) = (0u128, false);
        for &w in ws {
            mask ^= p[w];
            cb ^= c[w];
        }
        let a = if cb { a.times(-1.0) } else { a.clone() };
        if mask == 0 {
            *global = global.plus(&a.times(-0.5));
            return;
        }
        match terms.get_mut(&mask) {
            Some(t) => *t = t.plus(&a),
            None => {
                order.push(mask);
                terms.insert(mask, a);
            }
        }
    };
    for op in ops {
        match kind(op) {
            Kind::Perm | Kind::Y => {
                let g = match op {
                    POp::Fixed(g) => *g,
                    _ => unreachable!(),
                };
                if let Gate::Y(q) = g {
                    // Y = i X Z: Z first
                    diag_before += 1;
                    global = global.plus(&Angle::constant(PI / 2.0 + PI / 2.0));
                    add_rot(&[q], &Angle::constant(PI), &p, &c, &mut terms, &mut global);
                    c[q] ^= true;
                    perm_ops.push(POp::Fixed(Gate::X(q)));
                    continue;
                }
                match g {
                    Gate::X(q) => c[q] ^= true,
                    Gate::Cnot(a, b) => {
                        p[b] ^= p[a];
                        c[b] ^= c[a];
                    }
                    Gate::Swap(a, b) => {
                        p.swap(a, b);
                        c.swap(a, b);
                    }
                    _ => unreachable!(),
                }
                perm_ops.push(op.clone());
            }
            Kind::Diag => {
                diag_before += 1;
                tmp.clear();
                diag_terms(op, &mut tmp, &mut global);
                for (ws, a) in &tmp {
                    add_rot(ws, a, &p, &c, &mut terms, &mut global);
                }
            }
            Kind::Other => unreachable!(),
        }
    }
    // the new permutation network
    let identity = (0..n).all(|w| p[w] == 1u128 << w);
    let is_perm = !identity
        && p.iter().all(|m| m.count_ones() == 1)
        && p.iter().fold(0u128, |s, m| s | m).count_ones() as usize == n;
    let mut net: Vec<POp> = Vec::new();
    if identity || is_perm {
        // value(w) = x[src(w)] ⊕ c[w]; realise the wire permutation with SWAPs
        let mut src: Vec<usize> = p.iter().map(|m| m.trailing_zeros() as usize).collect();
        // cur[w] = which input currently sits on wire w
        let mut cur: Vec<usize> = (0..n).collect();
        let mut pos: Vec<usize> = (0..n).collect(); // pos[input] = wire
        for w in 0..n {
            let want = src[w];
            if cur[w] != want {
                let v = pos[want];
                net.push(POp::Fixed(Gate::Swap(w, v)));
                let (a, b) = (cur[w], cur[v]);
                cur.swap(w, v);
                pos[a] = v;
                pos[b] = w;
            }
        }
        src.clear();
        for (w, &cw) in c.iter().enumerate() {
            if cw {
                net.push(POp::Fixed(Gate::X(w)));
            }
        }
    } else {
        net = perm_ops.clone();
    }
    let gadgets: Vec<(u128, Angle)> = order
        .into_iter()
        .filter_map(|m| {
            let a = terms.remove(&m)?;
            (!(a.is_const() && (a.c0.rem_euclid(4.0 * PI)).abs() < 1e-15)).then_some((m, a))
        })
        .collect();
    let wide = |k: usize| if k > opts.max_weight { 2 * (k - 1) } else { 0 };
    let after_perm = net.len()
        + gadgets
            .iter()
            .map(|(m, _)| wide(m.count_ones() as usize))
            .sum::<usize>();
    let before_perm = perm_ops.len();
    st.regions += 1;
    st.perm_ops_before += before_perm;
    st.diag_ops_before += diag_before;
    let better =
        after_perm < before_perm || (after_perm == before_perm && gadgets.len() < diag_before);
    if !better {
        st.perm_ops_after += before_perm;
        return None;
    }
    st.rewritten += 1;
    st.perm_ops_after += after_perm;
    st.gadgets_after += gadgets.len();
    let mut out = Vec::with_capacity(gadgets.len() + net.len() + 1);
    if !(global.is_const() && global.c0 == 0.0) {
        out.push(POp::Global(global));
    }
    for (m, a) in gadgets {
        out.push(POp::ZString(bits(m), a));
    }
    out.extend(net);
    Some(out)
}

/// Runs the region rewrite over a whole circuit (see the module docs).
/// With `opts.reorder` both the re-scheduled and the original order are
/// tried and the one leaving fewer permutation ops (then fewer gadgets) is
/// kept: greedy re-scheduling can pull half of the next gadget's ladder into
/// a region and stop it from cancelling.
pub fn phase_regions(pc: &ParamCircuit, opts: &RewriteOptions) -> (ParamCircuit, RewriteStats) {
    let plain = regions_in_order(pc, pc.ops.clone(), opts);
    if !opts.reorder {
        return plain;
    }
    let re = regions_in_order(pc, reorder(pc), opts);
    let key =
        |r: &(ParamCircuit, RewriteStats)| (r.1.perm_ops_after, r.1.gadgets_after, r.0.ops.len());
    if key(&re) < key(&plain) {
        re
    } else {
        plain
    }
}

fn regions_in_order(
    pc: &ParamCircuit,
    ops: Vec<POp>,
    opts: &RewriteOptions,
) -> (ParamCircuit, RewriteStats) {
    let mut st = RewriteStats::default();
    let mut out = ParamCircuit::new(pc.num_qubits, pc.num_params);
    let mut i = 0;
    while i < ops.len() {
        if !is_region(&ops[i]) {
            out.ops.push(ops[i].clone());
            i += 1;
            continue;
        }
        let start = i;
        while i < ops.len() && is_region(&ops[i]) {
            i += 1;
        }
        let reg = &ops[start..i];
        match rewrite_region(pc.num_qubits, reg, opts, &mut st) {
            Some(r) => out.ops.extend(r),
            None => out.ops.extend_from_slice(reg),
        }
    }
    (out, st)
}
