//! Exactness oracle for the circuit DAG (`qsim_lab::dag`).
//!
//! Reference semantics: an exact branching simulator. Every measurement,
//! reset and noise channel splits the state into weighted branches, so a
//! circuit's full behaviour on |0…0> is the record-labelled mixed state
//! `ρ_r = Σ_{branches with record r} w |ψ><ψ|`. Two circuits are
//! equivalent iff these agree for every record `r` (this covers the outcome
//! distribution, the post-measurement states and classical control).
//! For unitary circuits amplitudes are also compared up to the tracked
//! global phase.

use num_complex::Complex64 as C;
use proptest::prelude::*;
use qsim_lab::dag::{self, Dag, PeepholeOptions};
use qsim_lab::{Circuit, Gate, Op, StateVectorF64};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Reference: exact branching simulation
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Branch {
    w: f64,
    sv: StateVectorF64,
    rec: Vec<bool>,
}

const P_EPS: f64 = 1e-14;

fn branches(c: &Circuit) -> Vec<Branch> {
    let mut bs = vec![Branch {
        w: 1.0,
        sv: StateVectorF64::new(c.num_qubits),
        rec: Vec::new(),
    }];
    let paulis = |q: usize| [Gate::I(q), Gate::X(q), Gate::Y(q), Gate::Z(q)];
    for op in &c.ops {
        let mut next = Vec::with_capacity(bs.len());
        for b in bs {
            match *op {
                Op::Gate(g) => {
                    let mut b = b;
                    b.sv.apply_gate(&g).unwrap();
                    next.push(b);
                }
                Op::Measure(q) | Op::Reset(q) => {
                    let p1 = b.sv.prob_one(q);
                    for (out, p) in [(false, 1.0 - p1), (true, p1)] {
                        if p <= P_EPS {
                            continue;
                        }
                        let mut nb = b.clone();
                        nb.sv.collapse(q, out);
                        nb.w *= p;
                        if matches!(op, Op::Measure(_)) {
                            nb.rec.push(out);
                        } else if out {
                            nb.sv.apply_gate(&Gate::X(q)).unwrap();
                        }
                        next.push(nb);
                    }
                }
                Op::ClassicControlled {
                    gate,
                    meas_index,
                    target_value,
                } => {
                    let mut b = b;
                    if b.rec[meas_index] == target_value {
                        b.sv.apply_gate(&gate).unwrap();
                    }
                    next.push(b);
                }
                Op::XFlip(q, p) | Op::YFlip(q, p) | Op::ZFlip(q, p) => {
                    let g = match *op {
                        Op::XFlip(..) => Gate::X(q),
                        Op::YFlip(..) => Gate::Y(q),
                        _ => Gate::Z(q),
                    };
                    push_mix(&mut next, b, &[(1.0 - p, vec![]), (p, vec![g])]);
                }
                Op::Depolarize1q(q, p) => {
                    let mut mix = vec![(1.0 - p, vec![])];
                    for g in &paulis(q)[1..] {
                        mix.push((p / 3.0, vec![*g]));
                    }
                    push_mix(&mut next, b, &mix);
                }
                Op::Depolarize2q(a, bq, p) => {
                    let mut mix = vec![(1.0 - p, vec![])];
                    for k in 1..16 {
                        mix.push((p / 15.0, vec![paulis(a)[k / 4], paulis(bq)[k % 4]]));
                    }
                    push_mix(&mut next, b, &mix);
                }
            }
        }
        bs = next;
    }
    bs
}

fn push_mix(next: &mut Vec<Branch>, b: Branch, mix: &[(f64, Vec<Gate>)]) {
    for (p, gs) in mix {
        if *p <= 0.0 {
            continue;
        }
        let mut nb = b.clone();
        nb.w *= p;
        for g in gs {
            nb.sv.apply_gate(g).unwrap();
        }
        next.push(nb);
    }
}

/// Record (projected onto `bits`, or the full record) -> reduced density
/// matrix on `keep_qubits` (row-major, dim 2^k).
fn labelled_states(
    c: &Circuit,
    bits: Option<&[usize]>,
    keep_qubits: &[usize],
) -> BTreeMap<Vec<bool>, Vec<C>> {
    let k = keep_qubits.len();
    let dim = 1usize << k;
    let n = c.num_qubits;
    let mut out: BTreeMap<Vec<bool>, Vec<C>> = BTreeMap::new();
    for b in branches(c) {
        let key: Vec<bool> = match bits {
            Some(bs) => bs.iter().map(|&i| b.rec[i]).collect(),
            None => b.rec.clone(),
        };
        let rho = out
            .entry(key)
            .or_insert_with(|| vec![C::new(0.0, 0.0); dim * dim]);
        let amps = b.sv.amplitudes();
        // Partial trace over the qubits not kept.
        let rest: Vec<usize> = (0..n).filter(|q| !keep_qubits.contains(q)).collect();
        for env in 0..(1usize << rest.len()) {
            let mut base = 0usize;
            for (j, &q) in rest.iter().enumerate() {
                base |= ((env >> j) & 1) << q;
            }
            let idx = |x: usize| {
                let mut i = base;
                for (j, &q) in keep_qubits.iter().enumerate() {
                    i |= ((x >> j) & 1) << q;
                }
                i
            };
            for x in 0..dim {
                let ax = amps[idx(x)];
                if ax.norm_sqr() == 0.0 {
                    continue;
                }
                for y in 0..dim {
                    rho[x * dim + y] += b.w * ax * amps[idx(y)].conj();
                }
            }
        }
    }
    out
}

fn assert_same_states(
    a: &BTreeMap<Vec<bool>, Vec<C>>,
    b: &BTreeMap<Vec<bool>, Vec<C>>,
    tol: f64,
    what: &str,
) {
    let zero_like = |m: &Vec<C>| m.iter().all(|x| x.norm() < tol);
    for (k, ma) in a {
        match b.get(k) {
            Some(mb) => {
                let d = ma
                    .iter()
                    .zip(mb)
                    .map(|(x, y)| (x - y).norm())
                    .fold(0.0, f64::max);
                assert!(d < tol, "{what}: record {k:?} differs by {d}");
            }
            None => assert!(zero_like(ma), "{what}: record {k:?} missing on one side"),
        }
    }
    for (k, mb) in b {
        if !a.contains_key(k) {
            assert!(zero_like(mb), "{what}: record {k:?} missing on one side");
        }
    }
}

fn full_states(c: &Circuit) -> BTreeMap<Vec<bool>, Vec<C>> {
    let all: Vec<usize> = (0..c.num_qubits).collect();
    labelled_states(c, None, &all)
}

fn amps(c: &Circuit) -> Vec<C> {
    let mut s = StateVectorF64::new(c.num_qubits);
    s.apply_circuit(c).unwrap();
    s.amplitudes().to_vec()
}

fn max_phase_diff(a: &[C], b: &[C], phase: f64) -> f64 {
    let ph = C::from_polar(1.0, phase);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - ph * y).norm())
        .fold(0.0, f64::max)
}

// ---------------------------------------------------------------------------
// Random circuits with every op type
// ---------------------------------------------------------------------------

fn angle(rng: &mut StdRng) -> f64 {
    match rng.random_range(0..8) {
        0 => PI / 4.0,
        1 => -PI / 4.0,
        2 => PI / 2.0,
        3 => PI,
        4 => 2.0 * PI,
        5 => 0.0,
        _ => rng.random_range(-4.0..4.0),
    }
}

fn random_gate(rng: &mut StdRng, n: usize, pool: &[usize]) -> Gate {
    use Gate::*;
    // Draw qubits from a small pool so gates collide (and cancel) often.
    let pick = |rng: &mut StdRng| pool[rng.random_range(0..pool.len())];
    let q = pick(rng);
    if n >= 2 && rng.random_bool(0.4) {
        let mut b = pick(rng);
        while b == q {
            b = rng.random_range(0..n);
        }
        if n >= 3 && rng.random_bool(0.1) {
            let mut t = rng.random_range(0..n);
            while t == q || t == b {
                t = rng.random_range(0..n);
            }
            return Ccx(q, b, t);
        }
        return match rng.random_range(0..6) {
            0 | 1 => Cnot(q, b),
            2 => Cz(q, b),
            3 => Swap(q, b),
            4 => CPhase(q, b, angle(rng)),
            _ => {
                if rng.random_bool(0.5) {
                    ISwap(q, b)
                } else {
                    ISwapdg(q, b)
                }
            }
        };
    }
    match rng.random_range(0..17) {
        0 => I(q),
        1 => H(q),
        2 => X(q),
        3 => Y(q),
        4 => Z(q),
        5 => S(q),
        6 => Sdg(q),
        7 => T(q),
        8 => Tdg(q),
        9 => Sx(q),
        10 => Sxdg(q),
        11 => Rx(q, angle(rng)),
        12 => Ry(q, angle(rng)),
        13 => Rz(q, angle(rng)),
        14 => Phase(q, angle(rng)),
        15 => U(q, angle(rng), angle(rng), angle(rng)),
        _ => T(q),
    }
}

/// A random circuit using every op type. `nonunitary` is the probability
/// that a slot is a measurement / reset / classical control / noise op.
fn random_circuit(seed: u64, n: usize, len: usize, nonunitary: f64) -> Circuit {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut c = Circuit::new(n);
    let pool: Vec<usize> = (0..n.min(3)).collect();
    let mut measured = 0usize;
    let probs = [0.0, 0.25, 1.0];
    for _ in 0..len {
        if rng.random_bool(nonunitary) {
            let q = rng.random_range(0..n);
            let p = probs[rng.random_range(0..3)];
            match rng.random_range(0..8) {
                0 | 1 => {
                    c.measure(q);
                    measured += 1;
                }
                2 => {
                    c.reset(q);
                }
                3 | 4 if measured > 0 => {
                    let m = rng.random_range(0..measured);
                    let g = random_gate(&mut rng, n, &pool);
                    c.classic_controlled(g, m, rng.random_bool(0.5));
                }
                5 => {
                    match rng.random_range(0..3) {
                        0 => c.x_flip(q, p),
                        1 => c.y_flip(q, p),
                        _ => c.z_flip(q, p),
                    };
                }
                6 => {
                    c.depolarize_1q(q, p);
                }
                7 if n >= 2 => {
                    let b = (q + 1 + rng.random_range(0..n - 1)) % n;
                    c.depolarize_2q(q, b, p);
                }
                _ => {
                    c.measure(q);
                    measured += 1;
                }
            }
        } else {
            let g = random_gate(&mut rng, n, &pool);
            c.gate(g);
        }
    }
    c
}

// ---------------------------------------------------------------------------
// Round trip
// ---------------------------------------------------------------------------

#[test]
fn round_trip_is_exact_on_every_op_type() {
    for seed in 0..300 {
        let n = 1 + (seed as usize % 5);
        let c = random_circuit(seed, n, 40, 0.35);
        let d = Dag::from_circuit(&c).unwrap();
        d.check_invariants().unwrap();
        assert_eq!(d.to_circuit(), c, "seed {seed}");
    }
}

#[test]
fn rejects_what_run_rejects() {
    let mut c = Circuit::new(2);
    c.c_if(0, Gate::X(0));
    assert!(Dag::from_circuit(&c).is_err());
    let mut c = Circuit::new(2);
    c.cnot(0, 2);
    assert!(Dag::from_circuit(&c).is_err());
    let mut c = Circuit::new(2);
    c.depolarize_2q(1, 1, 0.1);
    assert!(Dag::from_circuit(&c).is_err());
}

/// Any topological order of the DAG is a circuit with identical semantics.
#[test]
fn random_topological_orders_are_equivalent() {
    for seed in 0..60 {
        let n = 1 + (seed as usize % 4);
        let c = random_circuit(1000 + seed, n, 18, 0.35);
        let d = Dag::from_circuit(&c).unwrap();
        let reference = full_states(&c);
        let mut rng = StdRng::seed_from_u64(seed);
        let prio: Vec<u64> = (0..d.capacity()).map(|_| rng.random()).collect();
        let order = d.topo_order_with(|id| prio[id as usize]);
        let c2 = d.circuit_in_order(&order);
        // Same multiset of ops on every wire, in the same per-wire order.
        for q in 0..n {
            let wire = |c: &Circuit| -> Vec<String> {
                c.ops
                    .iter()
                    .filter(|op| {
                        let (qs, k) = dag::op_qubits(op);
                        qs[..k].contains(&q)
                    })
                    .map(|op| format!("{op:?}"))
                    .collect()
            };
            assert_eq!(wire(&c), wire(&c2), "seed {seed} wire {q}");
        }
        assert_same_states(&reference, &full_states(&c2), 1e-12, "topo order");
    }
}

/// Random legal slides keep the semantics and the invariants.
#[test]
fn random_slides_are_exact() {
    for seed in 0..60 {
        let n = 2 + (seed as usize % 3);
        let c = random_circuit(2000 + seed, n, 24, 0.3);
        let mut d = Dag::from_circuit(&c).unwrap();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut slid = 0;
        for _ in 0..200 {
            let ids: Vec<_> = d.node_ids().collect();
            let a = ids[rng.random_range(0..ids.len())];
            let succ = d.successors(a);
            if succ.is_empty() {
                continue;
            }
            let b = succ[rng.random_range(0..succ.len())];
            if d.can_slide(a, b) {
                d.slide(a, b).unwrap();
                slid += 1;
            } else {
                assert!(d.slide(a, b).is_err());
            }
            d.check_invariants().unwrap();
        }
        let c2 = d.to_circuit();
        assert_eq!(c2.ops.len(), c.ops.len());
        assert_same_states(&full_states(&c), &full_states(&c2), 1e-12, "slides");
        assert!(slid > 0 || c.ops.len() < 3, "seed {seed}: nothing slid");
    }
}

// ---------------------------------------------------------------------------
// Commutation is exact
// ---------------------------------------------------------------------------

fn embed_dense(g: &Gate, n: usize) -> Vec<C> {
    let dim = 1usize << n;
    let mut m = vec![C::new(0.0, 0.0); dim * dim];
    for col in 0..dim {
        let mut sv = StateVectorF64::basis_state(n, col);
        sv.apply_gate(g).unwrap();
        for (row, a) in sv.amplitudes().iter().enumerate() {
            m[row * dim + col] = *a;
        }
    }
    m
}

#[test]
fn ops_commute_is_sound_for_gates() {
    let n = 3;
    let dim = 1 << n;
    let mut rng = StdRng::seed_from_u64(7);
    let mut yes = 0;
    for _ in 0..4000 {
        let pool = [0, 1, 2];
        let a = random_gate(&mut rng, n, &pool);
        let b = random_gate(&mut rng, n, &pool);
        if dag::ops_commute(&Op::Gate(a), &Op::Gate(b)) {
            yes += 1;
            let (ma, mb) = (embed_dense(&a, n), embed_dense(&b, n));
            let mut d = 0.0f64;
            for i in 0..dim {
                for j in 0..dim {
                    let mut ab = C::new(0.0, 0.0);
                    let mut ba = C::new(0.0, 0.0);
                    for k in 0..dim {
                        ab += ma[i * dim + k] * mb[k * dim + j];
                        ba += mb[i * dim + k] * ma[k * dim + j];
                    }
                    d = d.max((ab - ba).norm());
                }
            }
            assert!(
                d < 1e-12,
                "{a:?} and {b:?} reported commuting, |[A,B]| = {d}"
            );
        }
    }
    assert!(yes > 500, "too few commuting pairs exercised: {yes}");
}

/// Non-unitary ops: commuting per the rules gives identical labelled states
/// in both orders.
#[test]
fn ops_commute_is_sound_for_channels() {
    let mut checked = 0;
    for seed in 0..3000u64 {
        let n = 2;
        let pre = random_circuit(seed, n, 4, 0.0);
        let pair = random_circuit(seed + 77_777, n, 2, 0.6);
        if pair.ops.len() != 2 {
            continue;
        }
        let (a, b) = (pair.ops[0], pair.ops[1]);
        if matches!(a, Op::ClassicControlled { .. }) || matches!(b, Op::ClassicControlled { .. }) {
            continue;
        }
        if matches!(a, Op::Measure(_)) && matches!(b, Op::Measure(_)) {
            continue;
        }
        if !dag::ops_commute(&a, &b) {
            continue;
        }
        let mut c1 = pre.clone();
        c1.ops.extend([a, b]);
        let mut c2 = pre.clone();
        c2.ops.extend([b, a]);
        assert_same_states(
            &full_states(&c1),
            &full_states(&c2),
            1e-12,
            "channel commute",
        );
        checked += 1;
    }
    assert!(checked > 300, "only {checked} commuting channel pairs");
}

// ---------------------------------------------------------------------------
// Peephole
// ---------------------------------------------------------------------------

fn check_peephole(c: &Circuit, opts: PeepholeOptions, what: &str) -> usize {
    let (o, _) = dag::optimize_with(c, opts).unwrap();
    assert!(o.circuit.ops.len() <= c.ops.len());
    if c.ops.iter().all(|op| matches!(op, Op::Gate(_))) {
        let d = max_phase_diff(&amps(c), &amps(&o.circuit), o.global_phase);
        assert!(d < 1e-12, "{what}: amplitude diff {d}");
    }
    assert_same_states(&full_states(c), &full_states(&o.circuit), 1e-12, what);
    c.ops.len() - o.circuit.ops.len()
}

#[test]
fn peephole_is_exact_on_unitary_circuits() {
    let mut removed = 0;
    for seed in 0..400 {
        let n = 1 + (seed as usize % 5);
        let c = random_circuit(5000 + seed, n, 40, 0.0);
        removed += check_peephole(&c, PeepholeOptions::default(), "unitary");
    }
    assert!(removed > 1000, "peephole removed only {removed} ops");
}

#[test]
fn peephole_is_exact_with_measurement_reset_noise_and_control() {
    for seed in 0..300 {
        let n = 1 + (seed as usize % 4);
        let c = random_circuit(9000 + seed, n, 30, 0.3);
        check_peephole(&c, PeepholeOptions::default(), "mixed");
        let adj = PeepholeOptions {
            commute: false,
            ..Default::default()
        };
        check_peephole(&c, adj, "mixed adjacent-only");
    }
}

/// The commutation-aware pass never leaves more gates than the adjacent-only
/// baseline of PR 1 on the same circuit.
#[test]
fn peephole_beats_adjacent_only_optimize() {
    let mut rng = StdRng::seed_from_u64(3);
    let (mut ours, mut base) = (0, 0);
    for _ in 0..50 {
        let c = Circuit::random_clifford_t(6, 20, 0.3, &mut rng);
        let o = dag::optimize(&c).unwrap();
        let b = c.optimize();
        assert!(o.circuit.num_gates() <= b.num_gates());
        ours += o.circuit.num_gates();
        base += b.num_gates();
    }
    assert!(ours < base, "{ours} vs {base}");
}

// ---------------------------------------------------------------------------
// Light cone and components
// ---------------------------------------------------------------------------

/// Flat-list reference light cone (same algorithm as the compile module's
/// `analysis::light_cone`: walk backwards, keep ops touching live qubits,
/// keep every measurement).
fn flat_light_cone(c: &Circuit, outputs: &[usize]) -> Circuit {
    let mut live = vec![false; c.num_qubits];
    for &q in outputs {
        live[q] = true;
    }
    let mut keep = vec![false; c.ops.len()];
    for (i, op) in c.ops.iter().enumerate().rev() {
        let (qs, k) = dag::op_qubits(op);
        let keep_op = matches!(op, Op::Measure(_)) || qs[..k].iter().any(|&q| live[q]);
        if keep_op {
            keep[i] = true;
            for &q in &qs[..k] {
                live[q] = true;
            }
        }
    }
    Circuit {
        num_qubits: c.num_qubits,
        ops: c
            .ops
            .iter()
            .zip(&keep)
            .filter(|(_, &k)| k)
            .map(|(op, _)| *op)
            .collect(),
    }
}

#[test]
fn light_cone_matches_flat_pass_and_is_exact() {
    for seed in 0..300 {
        let n = 1 + (seed as usize % 5);
        let c = random_circuit(13_000 + seed, n, 30, 0.25);
        let mut rng = StdRng::seed_from_u64(seed);
        let outputs: Vec<usize> = (0..n).filter(|_| rng.random_bool(0.3)).collect();
        let ours = dag::light_cone(&c, &outputs).unwrap();
        assert_eq!(ours, flat_light_cone(&c, &outputs), "seed {seed}");
        if n <= 4 {
            assert_same_states(
                &labelled_states(&c, None, &outputs),
                &labelled_states(&ours, None, &outputs),
                1e-12,
                "light cone",
            );
        }
    }
}

#[test]
fn marginal_light_cone_is_exact() {
    let mut dropped = 0;
    for seed in 0..300 {
        let n = 1 + (seed as usize % 4);
        let c = random_circuit(17_000 + seed, n, 30, 0.3);
        let nbits = c.ops.iter().filter(|o| matches!(o, Op::Measure(_))).count();
        let mut rng = StdRng::seed_from_u64(seed);
        let keep: Vec<usize> = (0..nbits).filter(|_| rng.random_bool(0.4)).collect();
        let outputs: Vec<usize> = (0..n).filter(|_| rng.random_bool(0.2)).collect();
        let (cone, bits) = dag::light_cone_marginal(&c, &keep, &outputs).unwrap();
        // Every requested bit survives (plus any a kept op reads).
        assert!(keep.iter().all(|b| bits.contains(b)));
        dropped += c.ops.len() - cone.ops.len();
        // The cone's record maps onto `bits`; compare the marginal on `keep`.
        let pos: Vec<usize> = keep
            .iter()
            .map(|b| bits.iter().position(|x| x == b).unwrap())
            .collect();
        assert_same_states(
            &labelled_states(&c, Some(&keep), &outputs),
            &labelled_states(&cone, Some(&pos), &outputs),
            1e-12,
            "marginal light cone",
        );
    }
    assert!(dropped > 500, "marginal cone dropped only {dropped} ops");
}

/// Flat-list reference components (same rule as the compile module).
fn flat_components(c: &Circuit) -> Vec<Vec<usize>> {
    let n = c.num_qubits;
    let mut comp: Vec<usize> = (0..n).collect();
    let mut meas_q = Vec::new();
    let relabel = |comp: &mut Vec<usize>, a: usize, b: usize| {
        let (x, y) = (comp[a], comp[b]);
        if x != y {
            let (lo, hi) = (x.min(y), x.max(y));
            for v in comp.iter_mut() {
                if *v == hi {
                    *v = lo;
                }
            }
        }
    };
    for op in &c.ops {
        let (qs, k) = dag::op_qubits(op);
        for i in 1..k {
            relabel(&mut comp, qs[0], qs[i]);
        }
        if let Op::ClassicControlled { meas_index, .. } = op {
            relabel(&mut comp, qs[0], meas_q[*meas_index]);
        }
        if let Op::Measure(q) = op {
            meas_q.push(*q);
        }
    }
    let mut out: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (q, &r) in comp.iter().enumerate() {
        out.entry(r).or_default().push(q);
    }
    out.into_values().collect()
}

#[test]
fn components_match_flat_pass() {
    for seed in 0..300 {
        let n = 1 + (seed as usize % 8);
        let mut rng = StdRng::seed_from_u64(seed);
        let len = rng.random_range(0..12);
        let c = random_circuit(21_000 + seed, n, len, 0.3);
        assert_eq!(
            dag::components(&c).unwrap(),
            flat_components(&c),
            "seed {seed}"
        );
    }
}

// ---------------------------------------------------------------------------
// Proptests
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn prop_round_trip(seed in any::<u64>(), n in 1usize..6, len in 0usize..60) {
        let c = random_circuit(seed, n, len, 0.3);
        let d = Dag::from_circuit(&c).unwrap();
        prop_assert!(d.check_invariants().is_ok());
        prop_assert_eq!(d.to_circuit(), c);
    }

    #[test]
    fn prop_peephole_exact(seed in any::<u64>(), n in 1usize..5, len in 0usize..40, nu in 0.0f64..0.4) {
        let c = random_circuit(seed, n, len, nu);
        check_peephole(&c, PeepholeOptions::default(), "prop");
    }

    #[test]
    fn prop_light_cone_exact(seed in any::<u64>(), n in 1usize..5, len in 0usize..30) {
        let c = random_circuit(seed, n, len, 0.3);
        let outputs: Vec<usize> = (0..n).filter(|q| (seed >> q) & 1 == 1).collect();
        let ours = dag::light_cone(&c, &outputs).unwrap();
        prop_assert_eq!(&ours, &flat_light_cone(&c, &outputs));
        assert_same_states(
            &labelled_states(&c, None, &outputs),
            &labelled_states(&ours, None, &outputs),
            1e-12,
            "prop light cone",
        );
    }
}

/// The oracle itself must notice a wrong rewrite.
#[test]
#[should_panic(expected = "differs")]
fn oracle_detects_a_non_commuting_swap() {
    let mut c1 = Circuit::new(2);
    c1.h(0).measure(0).x(1).cnot(0, 1).h(0).measure(1);
    let mut c2 = Circuit::new(2);
    c2.h(0).measure(0).x(1).h(0).cnot(0, 1).measure(1);
    assert_same_states(&full_states(&c1), &full_states(&c2), 1e-12, "oracle");
}
