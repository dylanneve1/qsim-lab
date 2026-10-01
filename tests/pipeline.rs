//! Exactness of the single entry point (`qsim_lab::pipeline::simulate`) and
//! of the blocked executor behind `Circuit::run`, against plain references:
//! the independent naive state vector of `audit_common`, gate-by-gate
//! `apply_gate` runs and the branching `exact_outcome_distribution`.

mod audit_common;
mod common;

use audit_common::RefSv;
use common::*;
use num_complex::Complex64;
use proptest::prelude::*;
use qsim_lab::circuit::{Circuit, Op, SimError, Simulator};
use qsim_lab::compile::plan::{exact_outcome_distribution, Backend};
use qsim_lab::compile::{compile_sampling, compile_unitary};
use qsim_lab::noise::{sample_depolarizing_1q, sample_depolarizing_2q};
use qsim_lab::pipeline::{plan_options, simulate, Budget, Output, Request};
use qsim_lab::{Gate, StateVectorF64};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;

fn rand_gate(n: usize, rng: &mut StdRng) -> Gate {
    let q = rng.random_range(0..n);
    let th = rng.random_range(-3.3..3.3);
    let other = |rng: &mut StdRng, not: &[usize]| loop {
        let x = rng.random_range(0..n);
        if !not.contains(&x) {
            break x;
        }
    };
    let pick = match n {
        1 => 17,
        2 => 22,
        _ => 23,
    };
    match rng.random_range(0..pick) {
        0 => Gate::H(q),
        1 => Gate::X(q),
        2 => Gate::Y(q),
        3 => Gate::Z(q),
        4 => Gate::S(q),
        5 => Gate::Sdg(q),
        6 => Gate::T(q),
        7 => Gate::Tdg(q),
        8 => Gate::Rx(q, th),
        9 => Gate::Ry(q, th),
        10 => Gate::Rz(q, th),
        11 => Gate::Phase(q, th),
        12 => Gate::Sx(q),
        13 => Gate::Sxdg(q),
        14 => Gate::I(q),
        15 => Gate::U(q, th, th * 0.7, -th * 1.3),
        16 => Gate::Rz(
            q,
            std::f64::consts::FRAC_PI_4 * rng.random_range(-4..5) as f64,
        ),
        17 => Gate::Cnot(q, other(rng, &[q])),
        18 => Gate::Cz(q, other(rng, &[q])),
        19 => Gate::Swap(q, other(rng, &[q])),
        20 => Gate::CPhase(q, other(rng, &[q]), th),
        21 => {
            if rng.random_bool(0.5) {
                Gate::ISwap(q, other(rng, &[q]))
            } else {
                Gate::ISwapdg(q, other(rng, &[q]))
            }
        }
        _ => {
            let b = other(rng, &[q]);
            let t = other(rng, &[q, b]);
            Gate::Ccx(q, b, t)
        }
    }
}

/// `len` random gates over every gate type; with `nonunit > 0`, that many
/// measurements, resets, classically conditioned gates and noise channels
/// are interleaved (at most one two-qubit channel).
fn random_ops(n: usize, len: usize, nonunit: usize, rng: &mut StdRng) -> Circuit {
    let mut c = Circuit::new(n);
    let mut slots: Vec<usize> = (0..nonunit).map(|_| rng.random_range(0..=len)).collect();
    slots.sort_unstable();
    let mut si = 0;
    let mut measured = 0;
    let mut have_2q_noise = false;
    for i in 0..=len {
        while si < slots.len() && slots[si] == i {
            si += 1;
            let q = rng.random_range(0..n);
            let p = rng.random_range(0.05..0.6);
            match rng.random_range(0..8) {
                0 | 1 => {
                    c.measure(q);
                    measured += 1;
                }
                2 => {
                    c.reset(q);
                }
                3 if measured > 0 => {
                    let g = loop {
                        let g = rand_gate(n, rng);
                        if g.qubits().len() <= 2 {
                            break g;
                        }
                    };
                    c.classic_controlled(g, rng.random_range(0..measured), rng.random_bool(0.5));
                }
                4 => {
                    c.x_flip(q, p);
                }
                5 => {
                    if rng.random_bool(0.5) {
                        c.y_flip(q, p);
                    } else {
                        c.z_flip(q, p);
                    }
                }
                6 => {
                    c.depolarize_1q(q, p);
                }
                _ if n >= 2 && !have_2q_noise => {
                    have_2q_noise = true;
                    let b = (q + 1 + rng.random_range(0..n - 1)) % n;
                    c.depolarize_2q(q, b, p);
                }
                _ => {
                    c.measure(q);
                    measured += 1;
                }
            }
        }
        if i < len {
            c.gate(rand_gate(n, rng));
        }
    }
    c
}

/// The pre-pipeline `Circuit::run` semantics: one `apply` per gate.
fn run_gatewise<S: Simulator>(c: &Circuit, sim: &mut S, rng: &mut StdRng) -> Vec<bool> {
    let mut out = Vec::new();
    for op in &c.ops {
        match op {
            Op::Gate(g) => sim.apply(g).unwrap(),
            Op::Measure(q) => out.push(sim.measure(*q, rng).unwrap()),
            Op::Reset(q) => sim.reset(*q, rng).unwrap(),
            Op::ClassicControlled {
                gate,
                meas_index,
                target_value,
            } => {
                if out[*meas_index] == *target_value {
                    sim.apply(gate).unwrap();
                }
            }
            Op::XFlip(q, p) | Op::YFlip(q, p) | Op::ZFlip(q, p) => {
                if *p > 0.0 && rng.random::<f64>() < *p {
                    sim.apply(&match op {
                        Op::XFlip(..) => Gate::X(*q),
                        Op::YFlip(..) => Gate::Y(*q),
                        _ => Gate::Z(*q),
                    })
                    .unwrap();
                }
            }
            Op::Depolarize1q(q, p) => {
                if let Some(e) = sample_depolarizing_1q(*p, *q, rng) {
                    sim.apply(&e).unwrap();
                }
            }
            Op::Depolarize2q(a, b, p) => {
                for e in sample_depolarizing_2q(*p, *a, *b, rng) {
                    sim.apply(&e).unwrap();
                }
            }
        }
    }
    out
}

fn apply_1q_matrix(s: &mut RefSv, q: usize, m: [[Complex64; 2]; 2]) {
    let old = s.a.clone();
    for (i, out) in s.a.iter_mut().enumerate() {
        let r = (i >> q) & 1;
        *out = m[r][0] * old[i & !(1 << q)] + m[r][1] * old[i | (1 << q)];
    }
}

/// Naive reference: `RefSv` plus textbook matrices for the gates it lacks.
fn ref_run(c: &Circuit) -> RefSv {
    let cx = Complex64::new;
    let mut s = RefSv::new(c.num_qubits);
    for g in c.gates() {
        match *g {
            Gate::I(_) => {}
            Gate::Sx(q) => apply_1q_matrix(
                &mut s,
                q,
                [[cx(0.5, 0.5), cx(0.5, -0.5)], [cx(0.5, -0.5), cx(0.5, 0.5)]],
            ),
            Gate::Sxdg(q) => apply_1q_matrix(
                &mut s,
                q,
                [[cx(0.5, -0.5), cx(0.5, 0.5)], [cx(0.5, 0.5), cx(0.5, -0.5)]],
            ),
            Gate::U(q, th, ph, lam) => {
                let (c2, s2) = ((th / 2.0).cos(), (th / 2.0).sin());
                let e = |t: f64| cx(t.cos(), t.sin());
                apply_1q_matrix(
                    &mut s,
                    q,
                    [[cx(c2, 0.0), -e(lam) * s2], [e(ph) * s2, e(ph + lam) * c2]],
                );
            }
            Gate::ISwap(a, b) | Gate::ISwapdg(a, b) => {
                let ph = if matches!(g, Gate::ISwap(..)) {
                    cx(0.0, 1.0)
                } else {
                    cx(0.0, -1.0)
                };
                let old = s.a.clone();
                for (i, &v) in old.iter().enumerate() {
                    let (ba, bb) = ((i >> a) & 1, (i >> b) & 1);
                    if ba != bb {
                        s.a[i ^ (1 << a) ^ (1 << b)] = v * ph;
                    }
                }
            }
            ref g => s.apply(g),
        }
    }
    s
}

fn dist_close(a: &BTreeMap<Vec<bool>, f64>, b: &BTreeMap<Vec<bool>, f64>) -> f64 {
    let mut keys: Vec<&Vec<bool>> = a.keys().chain(b.keys()).collect();
    keys.dedup();
    keys.iter()
        .map(|k| (a.get(*k).unwrap_or(&0.0) - b.get(*k).unwrap_or(&0.0)).abs())
        .fold(0.0, f64::max)
}

fn amplitudes(c: &Circuit) -> Vec<Complex64> {
    let n = c.num_qubits;
    let r = simulate(
        c,
        &Request::Amplitudes((0..1u128 << n).collect()),
        &Budget::default(),
    )
    .unwrap();
    match r.output {
        Output::Amplitudes(a) => a,
        o => panic!("{o:?}"),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// simulate(Amplitudes) = naive reference state vector, to 1e-12,
    /// including the global phase.
    #[test]
    fn simulate_amplitudes_match_reference(seed in any::<u64>(), n in 1usize..=8, len in 0usize..60) {
        let mut rng = StdRng::seed_from_u64(seed);
        let c = random_ops(n, len, 0, &mut rng);
        let want = ref_run(&c);
        let got = amplitudes(&c);
        for (i, a) in got.iter().enumerate() {
            prop_assert!((a - want.a[i]).norm() < 1e-12, "amp {} {} vs {}", i, a, want.a[i]);
        }
        let sv = sv_of(&c);
        for (i, a) in got.iter().enumerate() {
            prop_assert!((a - sv.amplitude(i)).norm() < 1e-12);
        }
    }

    /// Same, on circuits with several independent blocks and mostly
    /// Clifford gates (exercises components, state propagation, the
    /// Clifford prefix).
    #[test]
    fn simulate_amplitudes_structured(seed in any::<u64>(), len in 0usize..80) {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = 8;
        let mut c = Circuit::new(n);
        for _ in 0..len {
            let blk = rng.random_range(0..3);
            let qs: Vec<usize> = (0..n).filter(|q| q % 3 == blk).collect();
            let g = rand_gate(qs.len(), &mut rng);
            let m = |q: usize| qs[q];
            let g = match g {
                Gate::H(q) => Gate::H(m(q)),
                Gate::S(q) => Gate::S(m(q)),
                Gate::T(q) => Gate::T(m(q)),
                Gate::Rz(q, t) => Gate::Rz(m(q), t),
                Gate::Cnot(a, b) => Gate::Cnot(m(a), m(b)),
                Gate::Cz(a, b) => Gate::Cz(m(a), m(b)),
                Gate::Swap(a, b) => Gate::Swap(m(a), m(b)),
                Gate::X(q) => Gate::X(m(q)),
                _ => Gate::H(m(0)),
            };
            c.gate(g);
        }
        let want = ref_run(&c);
        for (i, a) in amplitudes(&c).iter().enumerate() {
            prop_assert!((a - want.a[i]).norm() < 1e-12);
        }
    }

    /// The plan the pipeline compiles has exactly the outcome distribution
    /// of the plain branching reference, for circuits with every op type.
    #[test]
    fn simulate_plan_distribution_is_exact(seed in any::<u64>(), n in 1usize..=6, len in 0usize..40, nonunit in 0usize..5) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut c = random_ops(n, len, nonunit, &mut rng);
        if rng.random_bool(0.5) {
            // terminal measurements (the sampling plan's full pipeline)
            for q in 0..n {
                if rng.random_bool(0.6) {
                    c.measure(q);
                }
            }
        }
        let want = exact_outcome_distribution(&c);
        let plan = compile_sampling(&c, plan_options()).unwrap();
        let got = plan.exact_distribution();
        prop_assert!(dist_close(&want, &got) < 1e-10, "{:?}\n{:?}", want, got);
    }

    /// Expectation values of Z products match the dense reference.
    #[test]
    fn simulate_expectation_matches_reference(seed in any::<u64>(), n in 1usize..=8, len in 0usize..50, mask in 1usize..256) {
        let mut rng = StdRng::seed_from_u64(seed);
        let c = random_ops(n, len, 0, &mut rng);
        let qs: Vec<usize> = (0..n).filter(|q| (mask >> q) & 1 == 1).collect();
        let s = ref_run(&c);
        let want: f64 = s.probs().iter().enumerate().map(|(i, p)| {
            if qs.iter().filter(|&&q| (i >> q) & 1 == 1).count() % 2 == 1 { -p } else { *p }
        }).sum();
        let got = match simulate(&c, &Request::Expectation(qs), &Budget::default()).unwrap().output {
            Output::Expectation(v) => v,
            o => panic!("{o:?}"),
        };
        prop_assert!((want - got).abs() < 1e-10, "{} vs {}", want, got);
    }

    /// `Circuit::run` (blocked executor between non-unitary ops) equals the
    /// gate-by-gate run: same outcomes with the same RNG, same final state
    /// to 1e-12, for circuits with every op type.
    #[test]
    fn run_blocked_matches_gatewise(seed in any::<u64>(), n in 1usize..=10, len in 0usize..80, nonunit in 0usize..8) {
        let mut rng = StdRng::seed_from_u64(seed);
        let c = random_ops(n, len, nonunit, &mut rng);
        let (mut a, mut b) = (StateVectorF64::new(n), StateVectorF64::new(n));
        let mut r1 = StdRng::seed_from_u64(seed ^ 7);
        let mut r2 = StdRng::seed_from_u64(seed ^ 7);
        let oa = c.run(&mut a, &mut r1).unwrap();
        let ob = run_gatewise(&c, &mut b, &mut r2);
        prop_assert_eq!(oa, ob);
        for i in 0..1usize << n {
            prop_assert!((a.amplitude(i) - b.amplitude(i)).norm() < 1e-12, "amp {}", i);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(12))]

    /// Registers above the cache-block size (several stages, gathered
    /// high qubits).
    #[test]
    fn run_blocked_matches_gatewise_large(seed in any::<u64>(), n in 15usize..=18, len in 20usize..90, nonunit in 0usize..4) {
        let mut rng = StdRng::seed_from_u64(seed);
        let c = random_ops(n, len, nonunit, &mut rng);
        let (mut a, mut b) = (StateVectorF64::new(n), StateVectorF64::new(n));
        let mut r1 = StdRng::seed_from_u64(seed ^ 9);
        let mut r2 = StdRng::seed_from_u64(seed ^ 9);
        let oa = c.run(&mut a, &mut r1).unwrap();
        let ob = run_gatewise(&c, &mut b, &mut r2);
        prop_assert_eq!(oa, ob);
        let mut worst = 0.0f64;
        for i in 0..1usize << n {
            worst = worst.max((a.amplitude(i) - b.amplitude(i)).norm());
        }
        prop_assert!(worst < 1e-12, "max |d amp| = {}", worst);
    }

    /// Unitary circuits of 15..=18 qubits through the pipeline (the
    /// factored state vector path) against the gate-by-gate vector.
    #[test]
    fn simulate_amplitudes_large(seed in any::<u64>(), n in 15usize..=17, len in 10usize..70) {
        let mut rng = StdRng::seed_from_u64(seed);
        let c = random_ops(n, len, 0, &mut rng);
        let want = sv_of(&c);
        let plan = compile_unitary(&c, plan_options()).unwrap();
        let f = plan.factored::<f64>().unwrap();
        let mut worst = 0.0f64;
        for i in 0..1usize << n {
            worst = worst.max((f.amplitude(i as u128) - want.amplitude(i)).norm());
        }
        prop_assert!(worst < 1e-12, "max |d amp| = {}", worst);
    }
}

/// Clifford+T circuits big enough for the compressed-state engine: the
/// pipeline must pick it, and its samples / expectations must match the
/// dense state vector.
#[test]
fn adaptive_is_chosen_and_exact() {
    let mut rng = StdRng::seed_from_u64(2024);
    for trial in 0..6 {
        let n = 14 + trial % 3;
        // random Clifford layers with 4 T gates in between
        let mut c = Circuit::new(n);
        for _ in 0..4 {
            c.append(&Circuit::random_clifford(n, 5, &mut rng));
            c.gate(Gate::T(rng.random_range(0..n)));
        }
        c.append(&Circuit::random_clifford(n, 5, &mut rng));
        let meas: Vec<usize> = (0..n).collect();
        let mut t = c.clone();
        for &q in &meas {
            t.measure(q);
        }
        // exact marginal of the measured bits from the dense vector
        let probs = sv_of(&c).probabilities();
        let shots = 60_000;
        let r = simulate(
            &t,
            &Request::Samples {
                shots,
                seed: 5 + trial as u64,
            },
            &Budget::default(),
        )
        .unwrap();
        if !meas.is_empty() {
            assert!(
                r.engines.iter().any(|e| e.2 == Backend::Adaptive),
                "trial {trial}: engines {:?}",
                r.engines
            );
        }
        let Output::Samples(samples) = r.output else {
            panic!()
        };
        // compare the marginal of the first min(3, k) measured bits
        let k = meas.len().min(3);
        for pat in 0..1usize << k {
            let want: f64 = probs
                .iter()
                .enumerate()
                .filter(|(i, _)| (0..k).all(|j| ((i >> meas[j]) & 1) == (pat >> j) & 1))
                .map(|(_, p)| p)
                .sum();
            let got = samples
                .iter()
                .filter(|s| (0..k).all(|j| s[j] == ((pat >> j) & 1 == 1)))
                .count() as f64
                / shots as f64;
            let sigma = (want * (1.0 - want)).sqrt() / (shots as f64).sqrt();
            assert!(
                (got - want).abs() < 5.0 * sigma + 1e-9,
                "trial {trial} pat {pat}: {got} vs {want}"
            );
        }
        // expectation of a Z product
        let qs: Vec<usize> = (0..n).filter(|_| rng.random_bool(0.3)).collect();
        if qs.is_empty() {
            continue;
        }
        let want: f64 = probs
            .iter()
            .enumerate()
            .map(|(i, p)| {
                if qs.iter().filter(|&&q| (i >> q) & 1 == 1).count() % 2 == 1 {
                    -p
                } else {
                    *p
                }
            })
            .sum();
        let Output::Expectation(got) = simulate(&c, &Request::Expectation(qs), &Budget::default())
            .unwrap()
            .output
        else {
            panic!()
        };
        assert!((got - want).abs() < 1e-10, "trial {trial}: {got} vs {want}");
    }
}

#[test]
fn budget_is_enforced() {
    let mut rng = StdRng::seed_from_u64(77);
    let mut c = random_universal(14, 120, &mut rng);
    c.measure_all();
    let tiny = Budget { mem_bytes: 64 };
    let r = simulate(&c, &Request::Samples { shots: 1, seed: 0 }, &tiny);
    assert!(matches!(r, Err(SimError::TooLarge { .. })), "{r:?}");
    let ok = simulate(
        &c,
        &Request::Samples { shots: 1, seed: 0 },
        &Budget::default(),
    );
    assert!(ok.is_ok());
}

#[test]
fn amplitudes_refuse_non_unitary() {
    let mut c = Circuit::new(2);
    c.h(0).measure(0);
    assert!(simulate(&c, &Request::Amplitudes(vec![0]), &Budget::default()).is_err());
}
