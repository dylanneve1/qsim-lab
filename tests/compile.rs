//! The compiler passes must be exact: optimised plans reproduce amplitudes
//! (including the global phase) and exact outcome distributions.

mod common;

use common::*;
use num_complex::Complex64;
use proptest::prelude::*;
use qsim_lab::circuit::{Circuit, Op, SimError};
use qsim_lab::compile::analysis::{
    clifford_prefix, eliminate_swaps, light_cone, split_monomial_suffix,
};
use qsim_lab::compile::plan::{exact_outcome_distribution, PlanOptions};
use qsim_lab::compile::stabsv::clifford_statevector;
use qsim_lab::compile::stateprop::propagate;
use qsim_lab::compile::{compile_sampling, compile_unitary, expectation_z_product, optimize};
use qsim_lab::{Gate, StateVectorF64};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::f64::consts::PI;

/// Gates biased towards things that merge/cancel: a small alphabet, angles
/// that are often multiples of π/4, and qubits drawn from `qs`.
fn gate_strategy(qs: Vec<usize>) -> impl Strategy<Value = Gate> {
    let k = qs.len();
    let q1 = {
        let qs = qs.clone();
        (0..k).prop_map(move |i| qs[i])
    };
    let pair = {
        let qs = qs.clone();
        (0..k, 0..k.max(2) - 1).prop_map(move |(a, b)| {
            let b = if b >= a { b + 1 } else { b };
            (qs[a], qs[b.min(k - 1)])
        })
    };
    let angle = prop_oneof![(-8i32..8).prop_map(|m| m as f64 * PI / 4.0), -7.0f64..7.0,];
    let ones = prop_oneof![
        q1.clone().prop_map(Gate::H),
        q1.clone().prop_map(Gate::X),
        q1.clone().prop_map(Gate::Y),
        q1.clone().prop_map(Gate::Z),
        q1.clone().prop_map(Gate::S),
        q1.clone().prop_map(Gate::Sdg),
        q1.clone().prop_map(Gate::T),
        q1.clone().prop_map(Gate::Tdg),
        (q1.clone(), angle.clone()).prop_map(|(q, t)| Gate::Rx(q, t)),
        (q1.clone(), angle.clone()).prop_map(|(q, t)| Gate::Ry(q, t)),
        (q1.clone(), angle.clone()).prop_map(|(q, t)| Gate::Rz(q, t)),
        (q1, angle.clone()).prop_map(|(q, t)| Gate::Phase(q, t)),
    ];
    if k < 2 {
        return ones.boxed();
    }
    let twos = prop_oneof![
        pair.clone().prop_map(|(a, b)| Gate::Cnot(a, b)),
        pair.clone().prop_map(|(a, b)| Gate::Cz(a, b)),
        pair.clone().prop_map(|(a, b)| Gate::Swap(a, b)),
        (pair, angle).prop_map(|((a, b), t)| Gate::CPhase(a, b, t)),
    ];
    if k < 3 {
        return prop_oneof![3 => ones, 2 => twos].boxed();
    }
    let qs3 = qs.clone();
    let three = (0..k, 0..k, 0..k)
        .prop_filter("distinct", |(a, b, c)| a != b && b != c && a != c)
        .prop_map(move |(a, b, c)| Gate::Ccx(qs3[a], qs3[b], qs3[c]));
    prop_oneof![6 => ones, 4 => twos, 1 => three].boxed()
}

/// A circuit on `n` qubits whose gates mostly stay inside a few blocks
/// (so components appear), plus occasional measurements.
fn circuit_strategy(meas: bool) -> impl Strategy<Value = Circuit> {
    (2usize..=7, 1usize..=3).prop_flat_map(move |(n, blocks)| {
        let blocks = blocks.min(n);
        let block_of: Vec<Vec<usize>> = (0..blocks)
            .map(|b| (0..n).filter(|q| q % blocks == b).collect())
            .collect();
        let all: Vec<usize> = (0..n).collect();
        let strategies: Vec<_> = block_of
            .into_iter()
            .map(|qs| gate_strategy(qs).boxed())
            .collect();
        let gates = prop::collection::vec(
            prop_oneof![
                10 => prop::strategy::Union::new(strategies).prop_map(Some),
                1 => gate_strategy(all).prop_map(Some),
                2 => Just(None),
            ],
            0..50,
        );
        (gates, prop::collection::vec(0..n, 0..60)).prop_map(move |(gs, mq)| {
            let mut c = Circuit::new(n);
            let mut mi = 0;
            for g in gs {
                match g {
                    Some(g) => {
                        c.gate(g);
                    }
                    None if meas && !mq.is_empty() => {
                        c.measure(mq[mi % mq.len()] % n);
                        mi += 1;
                    }
                    None => {}
                }
            }
            c
        })
    })
}

fn assert_states_equal(
    a: &StateVectorF64,
    b: &StateVectorF64,
    phase: f64,
) -> Result<(), TestCaseError> {
    let ph = Complex64::from_polar(1.0, phase);
    for i in 0..a.amplitudes().len() {
        let d = (a.amplitude(i) - b.amplitude(i) * ph).norm();
        prop_assert!(d < 1e-12, "amplitude {} differs by {}", i, d);
    }
    Ok(())
}

fn dist_close(a: &BTreeMap<Vec<bool>, f64>, b: &BTreeMap<Vec<bool>, f64>) -> f64 {
    let mut keys: Vec<&Vec<bool>> = a.keys().chain(b.keys()).collect();
    keys.dedup();
    keys.iter()
        .map(|k| (a.get(*k).unwrap_or(&0.0) - b.get(*k).unwrap_or(&0.0)).abs())
        .fold(0.0, f64::max)
}

fn terminal_version(c: &Circuit, rng_seed: u64) -> Circuit {
    let mut rng = StdRng::seed_from_u64(rng_seed);
    let mut t = Circuit::new(c.num_qubits);
    for g in c.gates() {
        t.gate(*g);
    }
    for q in 0..c.num_qubits {
        if rng.random_bool(0.6) {
            t.measure(q);
        }
    }
    if rng.random_bool(0.3) && c.num_qubits > 0 {
        t.measure(0); // repeated measurement
    }
    t
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn peephole_preserves_amplitudes(c in circuit_strategy(false)) {
        let o = optimize(&c);
        prop_assert!(o.circuit.num_gates() <= c.num_gates());
        assert_states_equal(&sv_of(&c), &sv_of(&o.circuit), o.global_phase)?;
    }

    #[test]
    fn peephole_cancels_circuit_times_inverse(c in circuit_strategy(false)) {
        let mut cc = c.clone();
        cc.append(&c.inverse());
        let o = optimize(&cc);
        assert_states_equal(&sv_of(&cc), &sv_of(&o.circuit), o.global_phase)?;
        // A mirrored circuit always collapses completely.
        prop_assert_eq!(o.circuit.num_gates(), 0, "{:?}", o.circuit);
    }

    #[test]
    fn unitary_plan_preserves_amplitudes(c in circuit_strategy(false)) {
        let want = sv_of(&c);
        for opts in [PlanOptions::default(), PlanOptions { peephole: false, ..PlanOptions::default() }] {
            let p = compile_unitary(&c, opts).unwrap();
            let got = p.statevector::<f64>().unwrap();
            assert_states_equal(&want, &got, 0.0)?;
            let f = p.factored::<f64>().unwrap();
            for i in 0..1u128 << c.num_qubits {
                prop_assert!((f.amplitude(i) - want.amplitude(i as usize)).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn clifford_prefix_absorption_is_exact(c in circuit_strategy(false)) {
        let (prefix, rest) = clifford_prefix(&c);
        let mut s = clifford_statevector::<f64>(&prefix);
        s.apply_circuit(&rest).unwrap();
        assert_states_equal(&sv_of(&c), &s, 0.0)?;
    }

    #[test]
    fn sampling_plan_terminal_distribution_is_exact(c in circuit_strategy(false), seed in 0u64..1000) {
        let t = terminal_version(&c, seed);
        let want = exact_outcome_distribution(&t);
        let plan = compile_sampling(&t, PlanOptions::default()).unwrap();
        let got = plan.exact_distribution();
        prop_assert!(dist_close(&want, &got) < 1e-10, "{:?}\n{:?}", want, got);
    }

    #[test]
    fn sampling_plan_midcircuit_distribution_is_exact(c in circuit_strategy(true)) {
        let want = exact_outcome_distribution(&c);
        let plan = compile_sampling(&c, PlanOptions::default()).unwrap();
        let got = plan.exact_distribution();
        prop_assert!(dist_close(&want, &got) < 1e-10, "{:?}\n{:?}", want, got);
    }

    #[test]
    fn peephole_preserves_midcircuit_distribution(c in circuit_strategy(true)) {
        let want = exact_outcome_distribution(&c);
        let got = exact_outcome_distribution(&optimize(&c).circuit);
        prop_assert!(dist_close(&want, &got) < 1e-10);
    }

    #[test]
    fn light_cone_preserves_marginals(c in circuit_strategy(true), q in 0usize..7) {
        let want = exact_outcome_distribution(&c);
        let got = exact_outcome_distribution(&light_cone(&c, &[]));
        prop_assert!(dist_close(&want, &got) < 1e-10);
        // reduced state of an output qubit: <Z_q>
        let q = q % c.num_qubits;
        let u: Circuit = Circuit { num_qubits: c.num_qubits, ops: c.ops.iter().filter(|o| matches!(o, Op::Gate(_))).copied().collect() };
        let a = sv_of(&u).expectation_z(q);
        let b = sv_of(&light_cone(&u, &[q])).expectation_z(q);
        prop_assert!((a - b).abs() < 1e-10);
    }

    #[test]
    fn monomial_suffix_preserves_distribution(c in circuit_strategy(false)) {
        let (rest, suffix) = split_monomial_suffix(&c);
        let p = sv_of(&c).probabilities();
        let pr = sv_of(&rest).probabilities();
        let n = c.num_qubits;
        let mut q = vec![0.0; 1 << n];
        for (x, w) in pr.iter().enumerate() {
            let mut bits: Vec<bool> = (0..n).map(|i| (x >> i) & 1 == 1).collect();
            qsim_lab::compile::analysis::apply_classical(&mut bits, &suffix);
            let y = bits.iter().enumerate().fold(0, |a, (i, &b)| a | ((b as usize) << i));
            q[y] += w;
        }
        prop_assert!(max_abs_diff(&p, &q) < 1e-12);
    }

    #[test]
    fn state_propagation_preserves_state(c in circuit_strategy(false)) {
        let all: Vec<usize> = (0..c.num_qubits).collect();
        let (o, ph) = propagate(&c, &all);
        assert_states_equal(&sv_of(&c), &sv_of(&o), ph)?;
    }

    #[test]
    fn state_propagation_preserves_distribution(c in circuit_strategy(true)) {
        let (o, _) = propagate(&c, &[]);
        let want = exact_outcome_distribution(&c);
        prop_assert!(dist_close(&want, &exact_outcome_distribution(&o)) < 1e-10);
    }

    #[test]
    fn swap_elimination_preserves_state(c in circuit_strategy(true)) {
        let (o, wire_of) = eliminate_swaps(&c);
        prop_assert!(o.gates().all(|g| !matches!(g, Gate::Swap(..))));
        let want = exact_outcome_distribution(&c);
        prop_assert!(dist_close(&want, &exact_outcome_distribution(&o)) < 1e-10);
        let u = Circuit { num_qubits: c.num_qubits, ops: c.ops.iter().filter(|o| matches!(o, Op::Gate(_))).copied().collect() };
        let (ou, w) = eliminate_swaps(&u);
        let (a, b) = (sv_of(&u), sv_of(&ou));
        let n = c.num_qubits;
        for i in 0..1usize << n {
            let j = (0..n).fold(0, |acc, q| acc | (((i >> q) & 1) << w[q]));
            prop_assert!((a.amplitude(i) - b.amplitude(j)).norm() < 1e-12);
        }
        let _ = wire_of;
    }

    #[test]
    fn expectation_matches_statevector(c in circuit_strategy(false), mask in 1usize..128) {
        let n = c.num_qubits;
        let qs: Vec<usize> = (0..n).filter(|q| (mask >> q) & 1 == 1).collect();
        let s = sv_of(&c);
        let want: f64 = s.probabilities().iter().enumerate().map(|(i, p)| {
            let par = qs.iter().filter(|&&q| (i >> q) & 1 == 1).count() % 2;
            if par == 1 { -p } else { *p }
        }).sum();
        let got = expectation_z_product(&c, &qs, PlanOptions::default()).unwrap();
        prop_assert!((want - got).abs() < 1e-10, "{} vs {}", want, got);
    }
}

/// The real sampling paths (tableau, Pauli paths, state vector with
/// absorbed prefix, classical suffix) against the exact distribution.
#[test]
fn sampled_frequencies_match_exact_distribution() {
    let mut rng = StdRng::seed_from_u64(42);
    let shots = 40_000;
    let mut seen = std::collections::HashSet::new();
    for trial in 0..30 {
        let n = 3 + trial % 5;
        let mut c = random_universal(n, 12 + trial, &mut rng);
        if trial % 3 == 0 {
            c = Circuit::random_clifford(n, 4, &mut rng);
        }
        if trial % 3 == 1 {
            c = Circuit::random_clifford_t(n, 4, 0.1, &mut rng);
        }
        let t = terminal_version(&c, trial as u64);
        let plan = compile_sampling(&t, PlanOptions::default()).unwrap();
        for comp in plan.components() {
            seen.insert(format!("{:?}", comp.backend));
        }
        let exact = plan.exact_distribution();
        let want = exact_outcome_distribution(&t);
        assert!(dist_close(&exact, &want) < 1e-10);
        let samples = plan.sample::<f64, _>(shots, &mut rng).unwrap();
        let mut counts: BTreeMap<Vec<bool>, usize> = BTreeMap::new();
        for s in samples {
            *counts.entry(s).or_insert(0) += 1;
        }
        for (k, p) in &want {
            let f = *counts.get(k).unwrap_or(&0) as f64 / shots as f64;
            let sigma = (p * (1.0 - p)).max(0.0).sqrt() / (shots as f64).sqrt();
            assert!(
                (f - p).abs() < 6.0 * sigma + 1e-3,
                "trial {trial}: {k:?} {f} vs {p}"
            );
        }
        for k in counts.keys() {
            assert!(
                want.contains_key(k),
                "trial {trial}: impossible outcome {k:?}"
            );
        }
    }
    assert!(
        seen.contains("Tableau") && seen.contains("StateVector"),
        "{seen:?}"
    );
}

#[test]
fn pauli_path_dispatch_is_used_and_exact() {
    // 30 qubits, a few T gates, measure two qubits: too big for the state
    // vector, cheap for Pauli paths.
    let mut rng = StdRng::seed_from_u64(9);
    let n = 30;
    let mut c = Circuit::random_clifford(n, 3, &mut rng);
    c.t(0).append(&Circuit::random_clifford(n, 2, &mut rng));
    c.measure(1).measure(2);
    let plan = compile_sampling(&c, PlanOptions::default()).unwrap();
    assert!(plan.stats.max_sv_qubits() <= 20, "{:?}", plan.stats);
    let s = plan.sample::<f32, _>(100, &mut rng).unwrap();
    assert_eq!(s.len(), 100);
}

#[test]
fn independent_registers_beyond_memory_cap() {
    // Four independent 8-qubit registers = 32 qubits: impossible as one
    // state vector, trivial as four.
    let mut rng = StdRng::seed_from_u64(3);
    let mut c = Circuit::new(32);
    for r in 0..4 {
        let sub = random_universal(8, 60, &mut rng);
        for g in sub.gates() {
            c.gate(qsim_lab::compile::analysis::relabel(g, |q| q + 8 * r));
        }
    }
    let plan = compile_unitary(&c, PlanOptions::default()).unwrap();
    let f = plan.factored::<f64>().unwrap();
    // spot-check one amplitude against a per-register state vector product
    let x: u128 = 0x1234_5678;
    let mut want = Complex64::new(1.0, 0.0);
    for r in 0..4 {
        let sub =
            qsim_lab::compile::analysis::restrict(&c, &(8 * r..8 * r + 8).collect::<Vec<_>>());
        want *= sv_of(&sub).amplitude(((x >> (8 * r)) & 0xff) as usize);
    }
    assert!((f.amplitude(x) - want).norm() < 1e-12);
}

#[test]
fn qft_of_basis_state_factorises() {
    let n = 12;
    let mut c = Circuit::new(n);
    for q in [0, 2, 3, 7, 11] {
        c.x(q);
    }
    c.append(&qsim_lab::algorithms::qft(n));
    let plan = compile_unitary(&c, PlanOptions::default()).unwrap();
    assert!(plan.stats.max_sv_qubits() <= 1, "{:?}", plan.stats);
    let got = plan.statevector::<f64>().unwrap();
    let want = sv_of(&c);
    for i in 0..1 << n {
        assert!((got.amplitude(i) - want.amplitude(i)).norm() < 1e-12);
    }
}

// ---------------------------------------------------------------------------
// Non-unitary operations: resets, classical control and noise channels.
// ---------------------------------------------------------------------------

/// Independent reference for the outcome-record distribution of any
/// circuit: a breadth-first list of weighted trajectories `(state, p,
/// record)`, expanded op by op with the semantics of `Circuit::run`.
/// Deliberately written differently from the library's recursive
/// `exact_outcome_distribution`, which it also checks.
fn reference_distribution(c: &Circuit) -> BTreeMap<Vec<bool>, f64> {
    let mut out = BTreeMap::new();
    for (_, p, r) in reference_trajectories(c) {
        *out.entry(r).or_insert(0.0) += p;
    }
    out
}

/// The final classical-quantum state `sum_rec |rec><rec| (x) rho_rec`
/// (unnormalised `rho_rec`, row-major), from the same trajectories. Equal
/// cq states mean the rewrite is exact for everything an observer could
/// later do with the qubits, not just for the recorded outcomes.
fn reference_cq_state(c: &Circuit) -> BTreeMap<Vec<bool>, Vec<Complex64>> {
    let dim = 1usize << c.num_qubits;
    let mut out: BTreeMap<Vec<bool>, Vec<Complex64>> = BTreeMap::new();
    for (s, p, r) in reference_trajectories(c) {
        let rho = out
            .entry(r)
            .or_insert_with(|| vec![Complex64::new(0.0, 0.0); dim * dim]);
        for i in 0..dim {
            let ai = s.amplitude(i);
            if ai.norm_sqr() == 0.0 {
                continue;
            }
            for j in 0..dim {
                rho[i * dim + j] += ai * s.amplitude(j).conj() * p;
            }
        }
    }
    out
}

fn cq_close(
    a: &BTreeMap<Vec<bool>, Vec<Complex64>>,
    b: &BTreeMap<Vec<bool>, Vec<Complex64>>,
) -> f64 {
    let mut worst: f64 = 0.0;
    for k in a.keys().chain(b.keys()) {
        match (a.get(k), b.get(k)) {
            (Some(x), Some(y)) => {
                for (u, v) in x.iter().zip(y) {
                    worst = worst.max((u - v).norm());
                }
            }
            (Some(x), None) | (None, Some(x)) => {
                for u in x {
                    worst = worst.max(u.norm());
                }
            }
            (None, None) => unreachable!(),
        }
    }
    worst
}

/// Weighted pure-state trajectories `(normalised state, probability,
/// record)` of a circuit under the semantics of `Circuit::run`.
fn reference_trajectories(c: &Circuit) -> Vec<(StateVectorF64, f64, Vec<bool>)> {
    type Traj = (StateVectorF64, f64, Vec<bool>);
    let mut trajs: Vec<Traj> = vec![(StateVectorF64::new(c.num_qubits), 1.0, Vec::new())];
    let with_paulis = |trajs: Vec<Traj>, cases: &[(f64, Vec<Gate>)]| -> Vec<Traj> {
        let mut next = Vec::new();
        for (s, p, rec) in trajs {
            for (w, gs) in cases {
                if *w <= 0.0 {
                    continue;
                }
                let mut t = s.clone();
                for g in gs {
                    t.apply_gate(g).unwrap();
                }
                next.push((t, p * w, rec.clone()));
            }
        }
        next
    };
    let pauli = |k: usize, q: usize| -> Vec<Gate> {
        match k {
            1 => vec![Gate::X(q)],
            2 => vec![Gate::Y(q)],
            3 => vec![Gate::Z(q)],
            _ => vec![],
        }
    };
    for op in &c.ops {
        trajs = match *op {
            Op::Gate(g) => trajs
                .into_iter()
                .map(|(mut s, p, r)| {
                    s.apply_gate(&g).unwrap();
                    (s, p, r)
                })
                .collect(),
            Op::ClassicControlled {
                gate,
                meas_index,
                target_value,
            } => trajs
                .into_iter()
                .map(|(mut s, p, r)| {
                    if r[meas_index] == target_value {
                        s.apply_gate(&gate).unwrap();
                    }
                    (s, p, r)
                })
                .collect(),
            Op::Measure(q) | Op::Reset(q) => {
                let record = matches!(op, Op::Measure(_));
                let mut next = Vec::new();
                for (s, p, r) in trajs {
                    let p1 = s.prob_one(q);
                    for (b, pb) in [(false, 1.0 - p1), (true, p1)] {
                        if pb < 1e-14 {
                            continue;
                        }
                        let mut t = s.clone();
                        t.collapse(q, b);
                        let mut r2 = r.clone();
                        if record {
                            r2.push(b);
                        } else if b {
                            t.apply_gate(&Gate::X(q)).unwrap();
                        }
                        next.push((t, p * pb, r2));
                    }
                }
                next
            }
            Op::XFlip(q, p) => with_paulis(trajs, &[(1.0 - p, vec![]), (p, vec![Gate::X(q)])]),
            Op::YFlip(q, p) => with_paulis(trajs, &[(1.0 - p, vec![]), (p, vec![Gate::Y(q)])]),
            Op::ZFlip(q, p) => with_paulis(trajs, &[(1.0 - p, vec![]), (p, vec![Gate::Z(q)])]),
            Op::Depolarize1q(q, p) => {
                let mut cases = vec![(1.0 - p, vec![])];
                cases.extend((1..4).map(|k| (p / 3.0, pauli(k, q))));
                with_paulis(trajs, &cases)
            }
            Op::Depolarize2q(a, b, p) => {
                let mut cases = vec![(1.0 - p, vec![])];
                cases.extend((1..16).map(|k| {
                    let mut e = pauli(k / 4, a);
                    e.extend(pauli(k % 4, b));
                    (p / 15.0, e)
                }));
                with_paulis(trajs, &cases)
            }
        };
    }
    trajs
}

/// One generated instruction of a noisy circuit (before indices are fixed).
#[derive(Clone, Debug)]
enum Ins {
    Gate(Gate),
    Measure(usize),
    Reset(usize),
    /// Conditional gate reading "some earlier measurement" (`pick` modulo
    /// the number of measurements so far; dropped if there is none).
    Cond(Gate, usize, bool),
    Noise(usize, usize, usize, f64),
}

/// Circuits with every kind of op: gates (mostly inside blocks, so
/// components appear), measurements, resets, classically controlled gates
/// (often reading a measurement in another block) and noise channels.
/// Branching ops are capped so the exact reference stays small.
fn noisy_circuit_strategy() -> impl Strategy<Value = Circuit> {
    (2usize..=5, 1usize..=3).prop_flat_map(|(n, blocks)| {
        let blocks = blocks.min(n);
        let block_of: Vec<Vec<usize>> = (0..blocks)
            .map(|b| (0..n).filter(|q| q % blocks == b).collect())
            .collect();
        let gates = prop::strategy::Union::new(
            block_of
                .into_iter()
                .map(|qs| gate_strategy(qs).boxed())
                .collect::<Vec<_>>(),
        );
        let prob = prop_oneof![
            Just(0.0),
            Just(1.0),
            Just(0.5),
            (0.0f64..=1.0).prop_map(|p| p)
        ];
        let ins = prop_oneof![
            12 => gates.clone().prop_map(Ins::Gate),
            2 => (0..n).prop_map(Ins::Measure),
            1 => (0..n).prop_map(Ins::Reset),
            2 => (gates, 0usize..8, any::<bool>()).prop_map(|(g, m, v)| Ins::Cond(g, m, v)),
            2 => (0usize..5, 0..n, 0..n, prob).prop_map(|(k, a, b, p)| Ins::Noise(k, a, b, p)),
        ];
        prop::collection::vec(ins, 0..30).prop_map(move |ins| {
            let mut c = Circuit::new(n);
            let (mut meas, mut branchy, mut noise) = (0usize, 0, 0);
            for i in ins {
                match i {
                    Ins::Gate(g) => {
                        c.gate(g);
                    }
                    Ins::Measure(q) | Ins::Reset(q) if branchy < 6 => {
                        branchy += 1;
                        if matches!(i, Ins::Measure(_)) {
                            c.measure(q);
                            meas += 1;
                        } else {
                            c.reset(q);
                        }
                    }
                    Ins::Cond(g, m, v) if meas > 0 => {
                        c.classic_controlled(g, m % meas, v);
                    }
                    Ins::Noise(k, a, b, p) if noise < 3 => {
                        noise += 1;
                        match k {
                            0 => c.x_flip(a, p),
                            1 => c.y_flip(a, p),
                            2 => c.z_flip(a, p),
                            3 => c.depolarize_1q(a, p),
                            _ if a != b => c.depolarize_2q(a, b, p),
                            _ => c.depolarize_1q(a, p),
                        };
                    }
                    _ => {}
                }
            }
            c
        })
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(192))]

    #[test]
    fn noisy_reference_agrees_with_library_reference(c in noisy_circuit_strategy()) {
        let want = reference_distribution(&c);
        let got = exact_outcome_distribution(&c);
        prop_assert!(dist_close(&want, &got) < 1e-10, "{:?}\n{:?}", want, got);
    }

    #[test]
    fn peephole_preserves_noisy_cq_state(c in noisy_circuit_strategy()) {
        let o = optimize(&c);
        let d = cq_close(&reference_cq_state(&c), &reference_cq_state(&o.circuit));
        prop_assert!(d < 1e-10, "{}: {:?}\n{:?}", d, c, o);
    }

    #[test]
    fn state_propagation_preserves_noisy_cq_state(c in noisy_circuit_strategy()) {
        let all: Vec<usize> = (0..c.num_qubits).collect();
        let (o, _) = propagate(&c, &all);
        let d = cq_close(&reference_cq_state(&c), &reference_cq_state(&o));
        prop_assert!(d < 1e-10, "{}: {:?}\n{:?}", d, c, o);
    }

    #[test]
    fn light_cone_preserves_noisy_distribution(c in noisy_circuit_strategy()) {
        let want = reference_distribution(&c);
        let got = reference_distribution(&light_cone(&c, &[]));
        prop_assert!(dist_close(&want, &got) < 1e-10);
    }

    #[test]
    fn state_propagation_preserves_noisy_distribution(c in noisy_circuit_strategy()) {
        let (o, _) = propagate(&c, &[]);
        let want = reference_distribution(&c);
        prop_assert!(dist_close(&want, &reference_distribution(&o)) < 1e-10, "{:?}\n{:?}", c, o);
    }

    #[test]
    fn swap_elimination_preserves_noisy_cq_state(c in noisy_circuit_strategy()) {
        let (o, wire_of) = eliminate_swaps(&c);
        // Undo the final wire permutation with explicit SWAPs, then the cq
        // states must agree exactly.
        let mut fixed = o.clone();
        let mut at = wire_of.clone(); // at[q] = wire currently holding q
        for q in 0..c.num_qubits {
            let w = at[q];
            if w != q {
                fixed.swap(w, q);
                let other = at.iter().position(|&x| x == q).unwrap();
                at.swap(q, other);
            }
        }
        prop_assert!(cq_close(&reference_cq_state(&c), &reference_cq_state(&fixed)) < 1e-10);
    }

    #[test]
    fn clifford_prefix_keeps_noisy_circuit(c in noisy_circuit_strategy()) {
        let (prefix, rest) = clifford_prefix(&c);
        prop_assert!(prefix.ops.iter().all(|o| matches!(o, Op::Gate(g) if g.is_clifford())));
        let mut joined = prefix.clone();
        joined.append(&rest);
        prop_assert!(cq_close(&reference_cq_state(&c), &reference_cq_state(&joined)) < 1e-10);
    }

    #[test]
    fn sampling_plan_noisy_distribution_is_exact(c in noisy_circuit_strategy()) {
        let want = reference_distribution(&c);
        for opts in [
            PlanOptions::default(),
            PlanOptions { state_prop: false, ..PlanOptions::default() },
            PlanOptions { peephole: false, swap_elim: false, ..PlanOptions::default() },
            PlanOptions::none(),
        ] {
            let plan = compile_sampling(&c, opts).unwrap();
            let got = plan.exact_distribution();
            prop_assert!(dist_close(&want, &got) < 1e-10, "{:?}\n{:?}\n{:?}", opts, want, got);
        }
    }

    #[test]
    fn unitary_and_expectation_refuse_non_unitary(c in noisy_circuit_strategy()) {
        let unitary = c.ops.iter().all(|o| matches!(o, Op::Gate(_)));
        let u = compile_unitary(&c, PlanOptions::default());
        let e = expectation_z_product(&c, &[0], PlanOptions::default());
        if unitary {
            prop_assert!(u.is_ok() && e.is_ok());
        } else {
            let refused = |r: Option<SimError>| matches!(r, Some(SimError::NotSupported { .. }));
            prop_assert!(refused(u.err()));
            prop_assert!(refused(e.err()));
        }
    }
}

/// `prefix; g; barrier; g' ; suffix` with `g'` equal or inverse to `g` on a
/// shared qubit: the shape in which a peephole pass that wrongly commutes
/// gates through a non-unitary op would merge them.
fn sandwich_strategy() -> impl Strategy<Value = Circuit> {
    (2usize..=3).prop_flat_map(|n| {
        let all: Vec<usize> = (0..n).collect();
        let gates = || prop::collection::vec(gate_strategy((0..n).collect()), 0..6);
        let prob = prop_oneof![Just(1.0), Just(0.5), 0.0f64..=1.0];
        (
            gates(),
            gate_strategy(all),
            any::<bool>(),
            (0usize..9, 0..n, prob, any::<bool>()),
            gates(),
        )
            .prop_map(move |(pre, g, inv, (kind, j, p, v), post)| {
                let mut c = Circuit::new(n);
                c.h(0).measure(0); // a record for classical control
                for g in pre {
                    c.gate(g);
                }
                c.gate(g);
                let qs = g.qubits();
                let q = qs[j % qs.len()];
                let other = (q + 1) % n;
                match kind {
                    0 => c.measure(q),
                    1 => c.reset(q),
                    2 => c.x_flip(q, p),
                    3 => c.y_flip(q, p),
                    4 => c.z_flip(q, p),
                    5 => c.depolarize_1q(q, p),
                    6 => c.depolarize_2q(q, other, p),
                    7 => c.classic_controlled(Gate::H(q), 0, v),
                    _ => c.classic_controlled(Gate::Cnot(other, q), 0, v),
                };
                c.gate(if inv { g.inverse() } else { g });
                for g in post {
                    c.gate(g);
                }
                c
            })
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn peephole_respects_non_unitary_barriers(c in sandwich_strategy()) {
        let o = optimize(&c);
        let d = cq_close(&reference_cq_state(&c), &reference_cq_state(&o.circuit));
        prop_assert!(d < 1e-10, "{}: {:?}\n{:?}", d, c, o);
        let all: Vec<usize> = (0..c.num_qubits).collect();
        let (s, _) = propagate(&c, &all);
        let d = cq_close(&reference_cq_state(&c), &reference_cq_state(&s));
        prop_assert!(d < 1e-10, "stateprop {}: {:?}\n{:?}", d, c, s);
    }
}

/// The reference's noise semantics against `Circuit::run` itself (Monte
/// Carlo), and the compiled sampler's real sampling path against both.
#[test]
fn noisy_sampling_matches_run_and_reference() {
    let mut rng = StdRng::seed_from_u64(77);
    let mut circuits = Vec::new();
    // Teleportation-style feed-forward with noise, plus an independent
    // noisy register and an ancilla that is reset and reused.
    let mut c = Circuit::new(5);
    c.ry(0, 1.1).h(1).cnot(1, 2).cnot(0, 1).h(0);
    c.depolarize_1q(1, 0.2).measure(0).measure(1);
    c.classic_controlled(Gate::X(2), 1, true);
    c.classic_controlled(Gate::Z(2), 0, true);
    c.x_flip(2, 0.1).measure(2);
    c.h(3).t(3).z_flip(3, 0.3).h(3).measure(3);
    c.cnot(3, 4)
        .measure(4)
        .reset(4)
        .h(4)
        .depolarize_2q(3, 4, 0.4);
    c.measure(4);
    circuits.push(c);
    // Clifford + noise: dispatched to the tableau.
    let mut c = Circuit::new(4);
    c.h(0)
        .cnot(0, 1)
        .y_flip(1, 0.25)
        .measure(1)
        .reset(1)
        .cnot(0, 1);
    c.classic_controlled(Gate::S(2), 0, false)
        .h(2)
        .depolarize_1q(2, 0.6);
    c.measure(0).measure(1).measure(2).measure(3);
    circuits.push(c);
    let shots = 40_000;
    for (ci, c) in circuits.iter().enumerate() {
        let want = reference_distribution(c);
        let plan = compile_sampling(c, PlanOptions::default()).unwrap();
        assert!(dist_close(&want, &plan.exact_distribution()) < 1e-10);
        let mut from_run: BTreeMap<Vec<bool>, usize> = BTreeMap::new();
        for _ in 0..shots {
            let mut s = StateVectorF64::new(c.num_qubits);
            *from_run
                .entry(c.run(&mut s, &mut rng).unwrap())
                .or_insert(0) += 1;
        }
        let mut from_plan: BTreeMap<Vec<bool>, usize> = BTreeMap::new();
        for r in plan.sample::<f64, _>(shots, &mut rng).unwrap() {
            *from_plan.entry(r).or_insert(0) += 1;
        }
        for counts in [&from_run, &from_plan] {
            for (k, p) in &want {
                let f = *counts.get(k).unwrap_or(&0) as f64 / shots as f64;
                let sigma = (p * (1.0 - p)).max(0.0).sqrt() / (shots as f64).sqrt();
                assert!(
                    (f - p).abs() < 6.0 * sigma + 1e-3,
                    "circuit {ci}: {k:?} {f} vs {p}"
                );
            }
            for k in counts.keys() {
                assert!(want.contains_key(k), "circuit {ci}: impossible {k:?}");
            }
        }
    }
}

#[test]
fn non_unitary_ops_are_peephole_barriers() {
    // X Reset X: merging the X pair across the reset would leave |0>.
    let mut c = Circuit::new(2);
    c.x(0).reset(0).x(0).measure(0);
    assert_eq!(optimize(&c).circuit.num_gates(), 2);
    // H ZFlip H is a bit flip, not a phase flip.
    let mut c = Circuit::new(1);
    c.h(0).z_flip(0, 0.5).h(0).measure(0);
    assert_eq!(optimize(&c).circuit.num_gates(), 2);
    // T, conditional H, T†: the T pair must not cancel through it.
    let mut c = Circuit::new(2);
    c.h(0).measure(0).h(1).t(1);
    c.classic_controlled(Gate::H(1), 0, true);
    c.tdg(1).measure(1);
    let o = optimize(&c);
    assert_eq!(o.circuit.ops.len(), c.ops.len());
    let want = reference_distribution(&c);
    assert!(dist_close(&want, &reference_distribution(&o.circuit)) < 1e-12);
}

#[test]
fn classical_control_joins_components_and_light_cone() {
    // q0 is measured; q1 is only touched by a gate conditioned on it.
    let mut c = Circuit::new(3);
    c.h(0).measure(0);
    c.classic_controlled(Gate::X(1), 0, true);
    c.measure(1);
    c.h(2); // unmeasured, irrelevant
    let comps = qsim_lab::compile::analysis::components(&c);
    assert_eq!(comps, vec![vec![0, 1], vec![2]]);
    let l = light_cone(&c, &[]);
    assert_eq!(l.ops.len(), 4, "{l:?}");
    let plan = compile_sampling(&c, PlanOptions::default()).unwrap();
    let d = plan.exact_distribution();
    assert_eq!(d.len(), 2);
    assert!((d[&vec![true, true]] - 0.5).abs() < 1e-12);
    assert!((d[&vec![false, false]] - 0.5).abs() < 1e-12);
    // restrict renumbers the classical index to the component's record.
    let mut c = Circuit::new(2);
    c.h(0).measure(0).h(1).measure(1);
    c.classic_controlled(Gate::X(1), 1, true).measure(1);
    assert_eq!(
        qsim_lab::compile::analysis::components(&c),
        vec![vec![0], vec![1]]
    );
    let r = qsim_lab::compile::analysis::restrict(&c, &[1]);
    assert_eq!(
        r.ops,
        vec![
            Op::Gate(Gate::H(0)),
            Op::Measure(0),
            Op::ClassicControlled {
                gate: Gate::X(0),
                meas_index: 0,
                target_value: true
            },
            Op::Measure(0),
        ]
    );
}

#[test]
fn invalid_circuits_are_rejected_like_run() {
    let mut c = Circuit::new(2);
    c.h(0).classic_controlled(Gate::X(1), 0, true).measure(0);
    assert!(matches!(
        compile_sampling(&c, PlanOptions::default()),
        Err(SimError::ClassicalBitOutOfRange {
            bit: 0,
            available: 0
        })
    ));
    let mut s = StateVectorF64::new(2);
    let mut rng = StdRng::seed_from_u64(1);
    assert!(matches!(
        c.run(&mut s, &mut rng),
        Err(SimError::ClassicalBitOutOfRange { .. })
    ));
    let mut c = Circuit::new(2);
    c.reset(5);
    assert!(matches!(
        compile_sampling(&c, PlanOptions::default()),
        Err(SimError::QubitOutOfRange { qubit: 5, .. })
    ));
    let mut c = Circuit::new(2);
    c.x_flip(0, 1.5).measure(0);
    assert!(compile_sampling(&c, PlanOptions::default()).is_err());
}

#[test]
fn reset_makes_qubit_known_again() {
    // After a reset the ancilla is |0> in every shot, so state propagation
    // can drop the CNOT it controls and the plan splits into components.
    let mut c = Circuit::new(3);
    c.h(0).cnot(0, 1).measure(1).reset(1);
    c.cnot(1, 2).h(2).measure(2).measure(0);
    let (o, _) = propagate(&c, &[]);
    assert!(!o.gates().any(|g| matches!(g, Gate::Cnot(1, 2))), "{o:?}");
    let want = reference_distribution(&c);
    assert!(dist_close(&want, &reference_distribution(&o)) < 1e-12);
    let plan = compile_sampling(&c, PlanOptions::default()).unwrap();
    assert!(dist_close(&want, &plan.exact_distribution()) < 1e-12);
}
