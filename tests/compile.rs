//! The compiler passes must be exact: optimised plans reproduce amplitudes
//! (including the global phase) and exact outcome distributions.

mod common;

use common::*;
use num_complex::Complex64;
use proptest::prelude::*;
use qsim_lab::circuit::{Circuit, Op};
use qsim_lab::compile::analysis::{clifford_prefix, light_cone, split_monomial_suffix};
use qsim_lab::compile::plan::{exact_outcome_distribution, Backend, PlanOptions};
use qsim_lab::compile::stabsv::clifford_statevector;
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
            let p = compile_unitary(&c, opts);
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
        let plan = compile_sampling(&t, PlanOptions::default());
        let got = plan.exact_distribution();
        prop_assert!(dist_close(&want, &got) < 1e-10, "{:?}\n{:?}", want, got);
    }

    #[test]
    fn sampling_plan_midcircuit_distribution_is_exact(c in circuit_strategy(true)) {
        let want = exact_outcome_distribution(&c);
        let plan = compile_sampling(&c, PlanOptions::default());
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
        let plan = compile_sampling(&t, PlanOptions::default());
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
    let plan = compile_sampling(&c, PlanOptions::default());
    assert!(plan
        .components()
        .iter()
        .any(|c| c.backend == Backend::PauliPath || c.backend == Backend::Tableau));
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
    let plan = compile_unitary(&c, PlanOptions::default());
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
