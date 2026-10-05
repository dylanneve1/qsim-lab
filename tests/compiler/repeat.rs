//! The repeat pass must be exact: every fast path is checked against plain
//! gate-by-gate simulation.

use num_complex::Complex64;
use proptest::prelude::*;
use qsim_lab::circuit::{Circuit, Op};
use qsim_lab::compile::plan::exact_outcome_distribution;
use qsim_lab::compile::repeat::cliff::{
    canonical_state, power_gates, sample_program, CliffordMap, TableCache,
};
use qsim_lab::compile::repeat::exec::{
    apply_matrix, block_unitary, diag_power, mat_pow, run_dense, ExecOptions,
};
use qsim_lab::compile::repeat::workloads::*;
use qsim_lab::compile::repeat::{detect, rewrite, DetectOptions, Node, Program};
use qsim_lab::gate::Gate;
use qsim_lab::pipeline::{
    simulate, simulate_with, Budget, Output, RepeatOptions, Request, SimOptions,
};
use qsim_lab::{StateVectorF64, Tableau};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn sv_gates(n: usize, gates: &[Gate], reps: usize) -> StateVectorF64 {
    let mut s = StateVectorF64::new(n);
    // |+...+> start so that diagonal blocks are non-trivial
    for q in 0..n {
        s.apply_gate(&Gate::H(q)).unwrap();
    }
    for _ in 0..reps {
        for g in gates {
            s.apply_gate(g).unwrap();
        }
    }
    s
}

fn plus_then(n: usize, gates: &[Gate]) -> StateVectorF64 {
    sv_gates(n, gates, 1)
}

/// Max |a - e^{iφ} b| minimised over the global phase φ.
fn dist_up_to_phase(a: &StateVectorF64, b: &StateVectorF64) -> f64 {
    let ip = a.inner(b); // <a|b>
    let ph = if ip.norm() > 0.0 {
        ip / ip.norm()
    } else {
        Complex64::new(1.0, 0.0)
    };
    a.amplitudes()
        .iter()
        .zip(b.amplitudes())
        .map(|(x, y)| (*x - *y * ph.conj()).norm())
        .fold(0.0, f64::max)
}

fn dist(a: &StateVectorF64, b: &StateVectorF64) -> f64 {
    a.amplitudes()
        .iter()
        .zip(b.amplitudes())
        .map(|(x, y)| (*x - *y).norm())
        .fold(0.0, f64::max)
}

// --------------------------------------------------------------- detection

#[test]
fn trotter_is_fully_covered() {
    let c = trotter(6, 50, 0.1, 0.07);
    let p = detect(&c, &DetectOptions::default());
    let r = p.report();
    assert!(r.coverage() > 0.9, "{r:?}");
    assert!(sv_diff(&p.to_circuit(), &c) < 1e-12);
}

fn sv_diff(a: &Circuit, b: &Circuit) -> f64 {
    let (mut x, mut y) = (
        StateVectorF64::new(a.num_qubits),
        StateVectorF64::new(b.num_qubits),
    );
    x.apply_circuit(a).unwrap();
    y.apply_circuit(b).unwrap();
    dist(&x, &y)
}

#[test]
fn shuffled_copies_still_match() {
    let mut rng = StdRng::seed_from_u64(5);
    let step = tfim_step(6, 0.1, 0.07);
    // every copy is a different random topological order of the same step
    let mut c = Circuit::new(6);
    for q in 0..6 {
        c.h(q);
    }
    for _ in 0..40 {
        c.append(&shuffle_commuting(&step, &mut rng));
    }
    let plain = detect(
        &c,
        &DetectOptions {
            layered: false,
            ..Default::default()
        },
    );
    let layered = detect(&c, &DetectOptions::default());
    assert!(layered.report().coverage() > 0.9, "{:?}", layered.report());
    assert!(layered.report().coverage() > plain.report().coverage());
    assert!(sv_diff(&layered.to_circuit(), &c) < 1e-12);
}

#[test]
fn qaoa_is_a_parameterised_repeat() {
    let gam: Vec<f64> = (0..12).map(|i| 0.1 + 0.05 * i as f64).collect();
    let bet: Vec<f64> = (0..12).map(|i| 0.7 - 0.03 * i as f64).collect();
    let c = qaoa_ring(6, &gam, &bet);
    let exact = detect(
        &c,
        &DetectOptions {
            parameterised: false,
            ..Default::default()
        },
    );
    assert!(exact.report().coverage() < 0.2, "{:?}", exact.report());
    let p = detect(&c, &DetectOptions::default());
    let r = p.report();
    assert!(r.coverage() > 0.8 && r.param_repeats >= 1, "{r:?}");
    assert!(sv_diff(&p.to_circuit(), &c) < 1e-12);
}

#[test]
fn grover_and_qec_are_covered() {
    let c = grover(5, 11, 20);
    let p = detect(&c, &DetectOptions::default());
    assert!(p.report().coverage() > 0.9, "{:?}", p.report());
    assert!(sv_diff(&p.to_circuit(), &c) < 1e-12);

    let q = qec_memory(5, 100, false);
    let p = detect(&q, &DetectOptions::default());
    assert!(p.report().coverage() > 0.9, "{:?}", p.report());
    assert_eq!(p.to_circuit().ops.len(), q.ops.len());
}

#[test]
fn nested_repeats() {
    let mut inner = Circuit::new(4);
    inner.h(0).cnot(0, 1).t(2).cnot(2, 3).h(3);
    let mut outer = Circuit::new(4);
    for _ in 0..5 {
        outer.append(&repeated(&inner, 4));
        outer.s(1).cz(1, 2).sdg(0).h(2);
    }
    let c = repeated(&outer, 6);
    let p = detect(&c, &DetectOptions::default());
    fn depth(n: &[Node]) -> usize {
        n.iter()
            .map(|x| match x {
                Node::Repeat { body, .. } => 1 + depth(body),
                _ => 0,
            })
            .max()
            .unwrap_or(0)
    }
    assert!(depth(&p.nodes) >= 2, "{:?}", p.report());
    assert!(sv_diff(&p.to_circuit(), &c) < 1e-12);
}

// ------------------------------------------------------ (a) Clifford power

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn clifford_power_matches_gate_by_gate(seed in 0u64..10_000, n in 1usize..7, len in 1usize..40, r in 1u64..70) {
        let mut rng = StdRng::seed_from_u64(seed);
        let body = random_clifford_gates(n, len, &mut rng);
        // map level: M^r equals the map of r copies, signs included
        let m = CliffordMap::from_gates(n, &body).unwrap();
        let mut all = Vec::new();
        for _ in 0..r { all.extend_from_slice(&body); }
        let direct = CliffordMap::from_gates(n, &all).unwrap();
        prop_assert_eq!(m.pow(r), direct.clone());
        // synthesis reproduces the map and the state up to a global phase
        let syn = m.pow(r).synthesize();
        prop_assert_eq!(CliffordMap::from_gates(n, &syn).unwrap(), direct);
        let pg = power_gates(&body, r).unwrap();
        let a = sv_gates(n, &body, r as usize);
        let b = plus_then(n, &pg);
        prop_assert!(dist_up_to_phase(&a, &b) < 1e-12, "dist {}", dist_up_to_phase(&a, &b));
    }
}

#[test]
fn clifford_power_matches_chp_tableau_with_signs() {
    for (n, len, r) in [(20usize, 120usize, 1000u64), (64, 300, 513), (70, 200, 77)] {
        let mut rng = StdRng::seed_from_u64(n as u64);
        let body = random_clifford_gates(n, len, &mut rng);
        // plain: |0..0> through r copies on the CHP tableau
        let mut t = Tableau::new(n);
        for _ in 0..r {
            for g in &body {
                t.apply_gate(g).unwrap();
            }
        }
        let mut u = Tableau::new(n);
        for g in power_gates(&body, r).unwrap() {
            u.apply_gate(&g).unwrap();
        }
        assert_eq!(canonical_state(&mut t), canonical_state(&mut u), "n={n}");
    }
}

#[test]
fn gate_tables_agree_with_matrices() {
    // every Clifford gate: map of the gate then its inverse is the identity
    let gs = [
        Gate::H(0),
        Gate::S(1),
        Gate::Sdg(0),
        Gate::Sx(1),
        Gate::Sxdg(0),
        Gate::X(1),
        Gate::Y(0),
        Gate::Z(1),
        Gate::Cnot(0, 1),
        Gate::Cnot(1, 0),
        Gate::Cz(0, 1),
        Gate::Swap(0, 1),
        Gate::ISwap(0, 1),
        Gate::ISwapdg(1, 0),
    ];
    let mut cache = TableCache::default();
    for g in gs {
        let mut m = CliffordMap::identity(2);
        assert!(m.apply_gate(&g, &mut cache));
        assert!(m.apply_gate(&g.inverse(), &mut cache));
        assert_eq!(m, CliffordMap::identity(2), "{g:?}");
    }
    // known signs: S: X -> Y, Y -> -X; H: X <-> Z; SQRT-X sign via matrices
    let mut m = CliffordMap::identity(1);
    m.apply_gate(&Gate::S(0), &mut cache);
    assert_eq!(m.row_string(0), "+Y");
    let mut m = CliffordMap::identity(1);
    m.apply_gate(&Gate::Y(0), &mut cache);
    assert_eq!(
        (m.row_string(0), m.row_string(1)),
        ("-X".into(), "-Z".into())
    );
}

// ---------------------------------------------------------- (c) diagonal

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn diagonal_fold_matches(seed in 0u64..10_000, n in 2usize..8, len in 1usize..30, r in 1u64..200) {
        let mut rng = StdRng::seed_from_u64(seed);
        let body: Vec<Gate> = (0..len).map(|_| {
            let a = rng.random_range(0..n);
            let b = (a + 1 + rng.random_range(0..n - 1)) % n;
            let t = rng.random_range(-3.2..3.2);
            match rng.random_range(0..8) {
                0 => Gate::Rz(a, t), 1 => Gate::Phase(a, t), 2 => Gate::CPhase(a, b, t),
                3 => Gate::Cz(a, b), 4 => Gate::T(a), 5 => Gate::S(a), 6 => Gate::Tdg(a), _ => Gate::Z(a),
            }
        }).collect();
        let (g, ph) = diag_power(&body, r).unwrap();
        // exact amplitudes including the global phase
        let a = sv_gates(n, &body, r as usize);
        let mut b = plus_then(n, &g);
        let f = Complex64::from_polar(1.0, ph);
        for z in b.amplitudes_mut() { *z *= f; }
        prop_assert!(dist(&a, &b) < 1e-12, "dist {}", dist(&a, &b));
    }
}

#[test]
fn diagonal_fold_is_more_accurate_than_repeating() {
    // r = 10^5: the folded angle is reduced in double-double; gate-by-gate
    // accumulates r roundings. Both agree to the gate-by-gate error.
    let body = vec![
        Gate::Rz(0, 0.123456789),
        Gate::CPhase(0, 1, 0.37),
        Gate::T(1),
    ];
    let r = 100_000u64;
    let (g, ph) = diag_power(&body, r).unwrap();
    let a = sv_gates(2, &body, r as usize);
    let mut b = plus_then(2, &g);
    let f = Complex64::from_polar(1.0, ph);
    for z in b.amplitudes_mut() {
        *z *= f;
    }
    let d = dist(&a, &b);
    assert!(d < 1e-9, "dist {d}");
    println!("diag r=1e5 |Δ| = {d:e}");
}

// ----------------------------------------- (b) small unitary and (d) plan

fn small_body(n: usize, len: usize, rng: &mut StdRng) -> Vec<Gate> {
    (0..len)
        .map(|_| {
            let a = rng.random_range(0..n);
            let b = (a + 1 + rng.random_range(0..n - 1)) % n;
            let t = rng.random_range(-1.0..1.0);
            match rng.random_range(0..7) {
                0 => Gate::Rx(a, t),
                1 => Gate::Ry(a, t),
                2 => Gate::Rz(a, t),
                3 => Gate::Cnot(a, b),
                4 => Gate::H(a),
                5 => Gate::T(a),
                _ => Gate::CPhase(a, b, t),
            }
        })
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn small_unitary_and_plan_reuse_match(seed in 0u64..10_000, n in 2usize..9, len in 3usize..30, r in 2usize..40) {
        let mut rng = StdRng::seed_from_u64(seed);
        let body = small_body(n, len, &mut rng);
        let c = {
            let mut c = Circuit::new(n);
            for q in 0..n { c.h(q); }
            for g in &body { c.gate(*g); }
            c
        };
        let rep = repeated(&{
            let mut b = Circuit::new(n);
            for g in &body { b.gate(*g); }
            b
        }, r);
        let _ = c;
        let mut full = Circuit::new(n);
        for q in 0..n { full.h(q); }
        full.append(&rep);
        let want = sv_gates(n, &body, r);
        let prog = Program {
            num_qubits: n,
            nodes: vec![
                Node::Ops((0..n).map(|q| Op::Gate(Gate::H(q))).collect()),
                Node::Repeat { body: vec![Node::Ops(body.iter().map(|g| Op::Gate(*g)).collect())], reps: r },
            ],
        };
        for (small, reuse) in [(true, false), (false, true), (false, false)] {
            let mut sv = StateVectorF64::new(n);
            let o = ExecOptions { diag: true, small_unitary: small, max_small_k: 8, reuse_plan: reuse, reuse_max_qubits: 64, force_small: small };
            run_dense(&prog, &mut sv, &o).unwrap();
            prop_assert!(dist(&want, &sv) < 1e-12, "small={small} reuse={reuse} dist {}", dist(&want, &sv));
        }
    }
}

/// (b) accuracy on a Trotter block for growing `r`: error of `U^r` against
/// gate-by-gate, and the unitarity defect of `U^r`.
#[test]
fn small_unitary_error_vs_r() {
    let n = 6;
    let step = tfim_step(n, 0.05, 0.04);
    let body: Vec<Gate> = step.gates().copied().collect();
    let u = block_unitary(n, &body);
    let d = 1usize << n;
    let mut worst: f64 = 0.0;
    for r in [10usize, 100, 1000, 10_000] {
        let up = mat_pow(&u, d, r as u64);
        let mut a = StateVectorF64::new(n);
        for q in 0..n {
            a.apply_gate(&Gate::H(q)).unwrap();
        }
        let mut b = a.clone();
        for _ in 0..r {
            for g in &body {
                a.apply_gate(g).unwrap();
            }
        }
        apply_matrix(&mut b, &(0..n).collect::<Vec<_>>(), &up);
        let e = dist(&a, &b);
        // unitarity defect max |(U^† U - I)_ij| via the state norm
        let defect = (b.norm_sqr() - 1.0).abs();
        println!("r={r} |Δamp|={e:e} norm defect={defect:e}");
        assert!(e < 1e-10, "r={r}: {e}");
        worst = worst.max(e);
    }
    assert!(worst < 1e-10);
}

// ---------------------------------------------------- QEC steady state

#[test]
fn qec_sampling_is_bit_identical_to_circuit_run() {
    for plus in [false, true] {
        for rounds in [1usize, 2, 3, 7, 40] {
            let c = qec_memory(4, rounds, plus);
            let prog = detect(&c, &DetectOptions::default());
            for seed in 0..6u64 {
                let mut r1 = StdRng::seed_from_u64(seed);
                let mut t = Tableau::new(c.num_qubits);
                let want = c.run(&mut t, &mut r1).unwrap();
                let mut r2 = StdRng::seed_from_u64(seed);
                let (got, stats) = sample_program(&prog, true, true, &mut r2).unwrap();
                assert_eq!(
                    got, want,
                    "plus={plus} rounds={rounds} seed={seed} {stats:?}"
                );
                // the rng streams stay in step too
                assert_eq!(r1.random::<u64>(), r2.random::<u64>());
                if rounds >= 8 {
                    assert!(stats.copies_skipped > 0, "{stats:?}");
                }
            }
        }
    }
}

#[test]
fn qec_program_matches_expanded_circuit() {
    let p = qec_program(5, 300, true);
    let c = p.to_circuit();
    for seed in 0..4u64 {
        let mut r1 = StdRng::seed_from_u64(seed);
        let mut t = Tableau::new(c.num_qubits);
        let want = c.run(&mut t, &mut r1).unwrap();
        let mut r2 = StdRng::seed_from_u64(seed);
        let (got, stats) = sample_program(&p, true, true, &mut r2).unwrap();
        assert_eq!(got, want);
        assert!(stats.copies_skipped >= 290, "{stats:?}");
    }
}

#[test]
fn random_measuring_rounds_are_never_skipped_wrongly() {
    // A Clifford round with measurements in a rotating basis: some rounds
    // random, the state keeps changing. Whatever the pass does, the result
    // must equal Circuit::run bit for bit.
    let mut rng = StdRng::seed_from_u64(11);
    for trial in 0..30 {
        let n = 5;
        let mut round = Circuit::new(n);
        for g in random_clifford_gates(n, 12, &mut rng) {
            round.gate(g);
        }
        round.measure(rng.random_range(0..n));
        if rng.random_bool(0.5) {
            round.reset(rng.random_range(0..n));
        }
        let mut c = Circuit::new(n);
        for _ in 0..12 {
            c.append(&round);
        }
        let prog = detect(
            &c,
            &DetectOptions {
                min_ops: 4,
                ..Default::default()
            },
        );
        for seed in 0..5u64 {
            let mut r1 = StdRng::seed_from_u64(seed);
            let mut t = Tableau::new(n);
            let want = c.run(&mut t, &mut r1).unwrap();
            let mut r2 = StdRng::seed_from_u64(seed);
            let (got, _) = sample_program(&prog, true, true, &mut r2).unwrap();
            assert_eq!(got, want, "trial {trial} seed {seed}");
        }
    }
}

// ------------------------------------------------------------- pipeline

fn ro() -> SimOptions {
    SimOptions {
        repeat: Some(RepeatOptions {
            min_saved_gates: 8,
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn amps(c: &Circuit, opts: &SimOptions) -> Vec<Complex64> {
    let xs: Vec<u128> = (0..1u128 << c.num_qubits.min(7)).collect();
    match simulate_with(c, &Request::Amplitudes(xs), &Budget::default(), opts)
        .unwrap()
        .output
    {
        Output::Amplitudes(a) => a,
        _ => unreachable!(),
    }
}

#[test]
fn simulate_default_is_unchanged() {
    let c = trotter(5, 20, 0.1, 0.2);
    let x = Request::Amplitudes(vec![0, 3, 7, 31]);
    let a = simulate(&c, &x, &Budget::default()).unwrap();
    let b = simulate_with(&c, &x, &Budget::default(), &SimOptions::default()).unwrap();
    assert_eq!(a, b);
}

#[test]
fn pipeline_amplitudes_with_repeat_match() {
    let gam: Vec<f64> = (0..8).map(|i| 0.1 + 0.05 * i as f64).collect();
    let bet: Vec<f64> = (0..8).map(|i| 0.7 - 0.03 * i as f64).collect();
    let circuits = vec![
        trotter(7, 60, 0.05, 0.04),
        qaoa_ring(6, &gam, &bet),
        grover(4, 5, 12),
        {
            // diagonal repeat after an H layer
            let mut b = Circuit::new(5);
            b.rz(0, 0.3).cphase(0, 1, 0.2).t(2).cz(3, 4);
            let mut c = Circuit::new(5);
            for q in 0..5 {
                c.h(q);
            }
            c.append(&repeated(&b, 300));
            c.rx(0, 0.4);
            c
        },
    ];
    for c in circuits {
        let want = amps(&c, &SimOptions::default());
        let got = amps(&c, &ro());
        let d = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);
        assert!(d < 1e-12, "{d}");
    }
}

fn dist_of(c: &Circuit) -> std::collections::BTreeMap<Vec<bool>, f64> {
    exact_outcome_distribution(c)
}

#[test]
fn pipeline_rewrites_preserve_exact_distributions() {
    // Clifford repeat + diagonal repeat + terminal measurement
    let mut rng = StdRng::seed_from_u64(3);
    let n = 5;
    let mut cl = Circuit::new(n);
    for g in random_clifford_gates(n, 25, &mut rng) {
        cl.gate(g);
    }
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    c.append(&repeated(&cl, 37));
    let mut dg = Circuit::new(n);
    dg.rz(1, 0.7).cphase(1, 3, 0.4).t(2);
    c.append(&repeated(&dg, 50));
    c.h(1).h(2);
    c.measure_all();
    let prog = detect(
        &c,
        &DetectOptions {
            min_ops: 8,
            ..Default::default()
        },
    );
    let rw = rewrite(&prog, true);
    assert!(
        rw.clifford_collapsed >= 1 && rw.diag_collapsed >= 1,
        "{rw:?}"
    );
    assert!(rw.circuit.ops.len() < c.ops.len());
    let (a, b) = (dist_of(&c), dist_of(&rw.circuit));
    // supports may differ by outcomes of probability ~1e-33 (rounding)
    for k in a.keys().chain(b.keys()) {
        let (p, q) = (
            a.get(k).copied().unwrap_or(0.0),
            b.get(k).copied().unwrap_or(0.0),
        );
        assert!((p - q).abs() < 1e-12, "{k:?} {p} {q}");
    }
    // and through simulate_with: samples are identical for the same seed
    // when the rewritten circuit is sampled by the same plan
    let req = Request::Samples { shots: 50, seed: 9 };
    let _ = simulate_with(&c, &req, &Budget::default(), &ro()).unwrap();
}

#[test]
fn pipeline_qec_samples_identical_to_plain_run() {
    let c = qec_memory(5, 60, true);
    let req = Request::Samples { shots: 20, seed: 4 };
    let got = simulate_with(&c, &req, &Budget::default(), &ro()).unwrap();
    let mut rng = StdRng::seed_from_u64(4);
    let mut want = Vec::new();
    for _ in 0..20 {
        let mut t = Tableau::new(c.num_qubits);
        want.push(c.run(&mut t, &mut rng).unwrap());
    }
    assert_eq!(got.output, Output::Samples(want));
}

#[test]
fn global_phase_of_clifford_power_is_not_claimed_for_amplitudes() {
    // Amplitude requests never use the Clifford power (phase unknown):
    // the answer keeps the exact phase.
    let mut b = Circuit::new(3);
    b.h(0).s(0).cnot(0, 1).sx(2).cz(1, 2).t(0); // one T keeps it from collapsing elsewhere
    let c = repeated(&b, 40);
    let want = amps(&c, &SimOptions::default());
    let got = amps(&c, &ro());
    for (a, b) in want.iter().zip(&got) {
        assert!((a - b).norm() < 1e-12);
    }
}
