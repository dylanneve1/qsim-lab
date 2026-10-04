//! Audit adapter for exp/compiler (49facf8): every compile pass / plan
//! against the independent reference in `tests/audit_common`.
//! Copy to `tests/` together with `tests/audit_common/`.
//!
//! * peephole `optimize`: e^{iφ}·U_opt == U (amplitudes, tracked phase)
//! * `compile_unitary` (random pass subsets): full vector and factored
//!   amplitudes, f64 and f32
//! * `expectation_z_product` (random subsets of passes and qubits)
//! * `compile_sampling`: `exact_distribution()` vs the reference outcome
//!   distribution, and chi-square of `sample()` (the real fast path) —
//!   terminal measurements (subset, random order, duplicates), Clifford
//!   (tableau), Clifford+T (Pauli paths), monomial suffixes, disconnected
//!   blocks, idle qubits, and mid-circuit measurements.

mod audit_common;

use audit_common::*;
use num_complex::Complex64 as C;
use qsim_lab::compile::plan::{compile_sampling, compile_unitary, PlanOptions};
use qsim_lab::compile::{expectation_z_product, optimize};
use qsim_lab::circuit::Op;
use qsim_lab::{Circuit, Gate};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::f64::consts::PI;

fn random_opts(rng: &mut StdRng) -> PlanOptions {
    if rng.random_bool(0.4) {
        return PlanOptions::default();
    }
    PlanOptions {
        peephole: rng.random_bool(0.5),
        light_cone: rng.random_bool(0.5),
        suffix: rng.random_bool(0.5),
        split: rng.random_bool(0.5),
        clifford_prefix: rng.random_bool(0.5),
        dispatch: rng.random_bool(0.5),
    }
}

/// Reference distribution of measurement records (program order).
fn ref_distribution(c: &Circuit) -> BTreeMap<Vec<bool>, f64> {
    fn go(c: &Circuit, i: usize, mut s: RefSv, p: f64, rec: &mut Vec<bool>, out: &mut BTreeMap<Vec<bool>, f64>) {
        let mut i = i;
        while i < c.ops.len() {
            match c.ops[i] {
                Op::Gate(g) => s.apply(&g),
                Op::Measure(q) => {
                    let p1 = s.prob_one(q);
                    for (o, po) in [(false, 1.0 - p1), (true, p1)] {
                        if po < 1e-13 {
                            continue;
                        }
                        let mut t = s.clone();
                        t.collapse(q, o);
                        rec.push(o);
                        go(c, i + 1, t, p * po, rec, out);
                        rec.pop();
                    }
                    return;
                }
            }
            i += 1;
        }
        *out.entry(rec.clone()).or_insert(0.0) += p;
    }
    let mut out = BTreeMap::new();
    go(c, 0, RefSv::new(c.num_qubits), 1.0, &mut Vec::new(), &mut out);
    out
}

fn dist_diff(a: &BTreeMap<Vec<bool>, f64>, b: &BTreeMap<Vec<bool>, f64>) -> f64 {
    let mut d = 0.0f64;
    for (k, v) in a {
        d = d.max((v - b.get(k).copied().unwrap_or(0.0)).abs());
    }
    for (k, v) in b {
        d = d.max((v - a.get(k).copied().unwrap_or(0.0)).abs());
    }
    d
}

/// Chi-square goodness of fit; panics on samples outside the support.
fn chi2_check(samples: &[Vec<bool>], p: &BTreeMap<Vec<bool>, f64>, ctx: &str) {
    let shots = samples.len() as f64;
    let mut counts: BTreeMap<Vec<bool>, usize> = BTreeMap::new();
    for s in samples {
        let ps = p.get(s).copied().unwrap_or(0.0);
        assert!(ps > 1e-12, "{ctx}: sampled record {s:?} has reference probability {ps}");
        *counts.entry(s.clone()).or_insert(0) += 1;
    }
    let (mut chi2, mut dof, mut rest_e, mut rest_o) = (0.0, 0usize, 0.0, 0.0);
    for (k, &pk) in p {
        let e = pk * shots;
        let o = counts.get(k).copied().unwrap_or(0) as f64;
        if e < 5.0 {
            rest_e += e;
            rest_o += o;
        } else {
            chi2 += (o - e).powi(2) / e;
            dof += 1;
        }
    }
    if rest_e >= 5.0 {
        chi2 += (rest_o - rest_e).powi(2) / rest_e;
        dof += 1;
    }
    let bound = dof as f64 + 6.0 * (2.0 * dof.max(1) as f64).sqrt() + 12.0;
    assert!(chi2 < bound, "{ctx}: chi2={chi2:.1} dof={dof} bound={bound:.1}");
}

/// Circuit families that trigger the different passes/backends.
fn family_circuit(rng: &mut StdRng, n: usize, fam: usize) -> Circuit {
    let depth = rng.random_range(1..50);
    match fam {
        // universal
        0 => random_circuit(rng, n, depth, false, false),
        // Clifford only (tableau)
        1 => random_circuit(rng, n, depth, true, false),
        // Clifford + a few T (Pauli paths)
        2 => {
            let mut c = random_circuit(rng, n, depth, true, false);
            for _ in 0..rng.random_range(1..4) {
                let q = edge_qubit(rng, n);
                c.gate(if rng.random_bool(0.5) { Gate::T(q) } else { Gate::Tdg(q) });
                c.append(&random_circuit(rng, n, 5, true, false));
            }
            c
        }
        // two disconnected blocks + idle qubit(s)
        3 => {
            let mut c = Circuit::new(n);
            if n >= 3 {
                let cut = rng.random_range(1..n - 1);
                for _ in 0..depth {
                    let lo = rng.random_bool(0.5);
                    let (base, size) = if lo { (0, cut) } else { (cut, n - 1 - cut) };
                    if size == 0 {
                        continue;
                    }
                    let sub = random_gate(rng, size, false, false);
                    if sub.qubits().iter().all(|&q| q < size) {
                        c.gate(qsim_lab::compile::analysis::relabel(&sub, |q| q + base));
                    }
                }
            }
            c
        }
        // universal body followed by a monomial suffix (classical post-processing)
        _ => {
            let mut c = random_circuit(rng, n, depth, false, false);
            for _ in 0..rng.random_range(1..12) {
                let g = loop {
                    let g = random_gate(rng, n, false, false);
                    if matches!(g, Gate::X(_) | Gate::Y(_) | Gate::Z(_) | Gate::S(_) | Gate::T(_) | Gate::Cnot(..) | Gate::Swap(..) | Gate::Cz(..) | Gate::Ccx(..) | Gate::CPhase(..)) {
                        break g;
                    }
                };
                c.gate(g);
            }
            c
        }
    }
}

#[test]
fn peephole_preserves_unitary_with_tracked_phase() {
    let mut worst = 0.0f64;
    for it in 0..40 * iters() {
        for &n in &[1usize, 2, 3, 5, 8] {
            let seed = base_seed() ^ 0xC0A1 ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let mut c = Circuit::new(n);
            for _ in 0..rng.random_range(1..60) {
                let g = random_gate(&mut rng, n, false, false);
                c.gate(g);
                match rng.random_range(0..5) {
                    0 => {
                        c.gate(g.inverse());
                    }
                    1 => {
                        c.gate(g);
                    }
                    2 => {
                        // a commuting gate in between, then the inverse
                        let q = g.qubits()[0];
                        c.gate(Gate::Rz(q, edge_angle(&mut rng)));
                        c.gate(g.inverse());
                    }
                    3 => {
                        let k = rng.random_range(-2..3) as f64;
                        let g2 = match g {
                            Gate::Rx(q, t) => Gate::Rx(q, 2.0 * PI * k - t),
                            Gate::Ry(q, t) => Gate::Ry(q, 2.0 * PI * k - t),
                            Gate::Rz(q, t) => Gate::Rz(q, 2.0 * PI * k - t),
                            Gate::Phase(q, t) => Gate::Phase(q, 2.0 * PI * k - t),
                            Gate::CPhase(a, b, t) => Gate::CPhase(b, a, 2.0 * PI * k - t),
                            g => g,
                        };
                        c.gate(g2);
                    }
                    _ => {}
                }
            }
            let o = optimize(&c);
            let r = RefSv::run(&c);
            let ro = RefSv::run(&o.circuit);
            let ph = C::from_polar(1.0, o.global_phase);
            let d = max_amp_diff(&r.a, ro.a.iter().map(|x| x * ph));
            worst = worst.max(d);
            assert!(d <= 1e-12, "peephole Δ={d:e} seed={seed} n={n}\n{:?}\n-> phase {} {:?}", c.ops, o.global_phase, o.circuit.ops);
        }
    }
    eprintln!("peephole worst Δ = {worst:e}");
}

#[test]
fn unitary_plan_matches_reference() {
    let mut worst = 0.0f64;
    for it in 0..15 * iters() {
        for &n in &[1usize, 2, 3, 4, 6, 9, 12] {
            for fam in 0..5 {
                let seed = base_seed() ^ 0xC0A2 ^ ((it as u64) << 20) ^ ((fam as u64) << 8) ^ n as u64;
                let mut rng = StdRng::seed_from_u64(seed);
                let c = family_circuit(&mut rng, n, fam);
                let opts = random_opts(&mut rng);
                let plan = compile_unitary(&c, opts);
                let r = RefSv::run(&c);
                let sv = plan.statevector::<f64>().unwrap();
                let d = max_amp_diff(&r.a, (0..1 << n).map(|i| sv.amplitude(i)));
                worst = worst.max(d);
                assert!(d <= 1e-12, "unitary plan f64 Δ={d:e} seed={seed} n={n} fam={fam} opts={opts:?}\n{:?}", c.ops);
                let f = plan.factored::<f64>().unwrap();
                let d = max_amp_diff(&r.a, (0..1u128 << n).map(|x| f.amplitude(x)));
                assert!(d <= 1e-12, "factored amplitude Δ={d:e} seed={seed} n={n} fam={fam} opts={opts:?}");
                let s32 = plan.statevector::<f32>().unwrap();
                let d = max_amp_diff(&r.a, (0..1 << n).map(|i| s32.amplitude(i)));
                assert!(d <= 1e-5, "unitary plan f32 Δ={d:e} seed={seed} n={n} fam={fam}");
            }
        }
    }
    eprintln!("unitary plan worst f64 Δ = {worst:e}");
}

#[test]
fn expectation_plan_matches_reference() {
    for it in 0..15 * iters() {
        for &n in &[1usize, 2, 3, 5, 8, 11] {
            for fam in 0..5 {
                let seed = base_seed() ^ 0xC0A3 ^ ((it as u64) << 20) ^ ((fam as u64) << 8) ^ n as u64;
                let mut rng = StdRng::seed_from_u64(seed);
                let c = family_circuit(&mut rng, n, fam);
                let opts = random_opts(&mut rng);
                let k = rng.random_range(1..=n.min(4));
                let mut qs: Vec<usize> = Vec::new();
                while qs.len() < k {
                    let q = edge_qubit(&mut rng, n);
                    if !qs.contains(&q) {
                        qs.push(q);
                    }
                }
                let v = expectation_z_product(&c, &qs, opts).unwrap();
                let p: String = (0..n).map(|q| if qs.contains(&q) { 'Z' } else { 'I' }).collect();
                let e = RefSv::run(&c).pauli_expectation(&p);
                assert!((v - e).abs() < 1e-10, "<Z{qs:?}> plan={v} ref={e} seed={seed} n={n} fam={fam} opts={opts:?}\n{:?}", c.ops);
            }
        }
    }
}

fn add_measurements(rng: &mut StdRng, c: &mut Circuit, n: usize) {
    match rng.random_range(0..4) {
        0 => {
            c.measure_all();
        }
        1 => {
            // random subset in random order
            let mut qs: Vec<usize> = (0..n).filter(|_| rng.random_bool(0.6)).collect();
            if qs.is_empty() {
                qs.push(n - 1);
            }
            for i in (1..qs.len()).rev() {
                let j = rng.random_range(0..=i);
                qs.swap(i, j);
            }
            for q in qs {
                c.measure(q);
            }
        }
        2 => {
            // duplicates
            for _ in 0..rng.random_range(1..2 * n + 2) {
                let q = edge_qubit(rng, n);
                c.measure(q);
            }
        }
        _ => {
            c.measure(0);
            c.measure(n - 1);
            c.measure(0);
        }
    }
}

#[test]
fn sampling_plan_terminal_matches_reference() {
    let mut tally: BTreeMap<String, usize> = BTreeMap::new();
    for it in 0..6 * iters() {
        for &n in &[1usize, 2, 3, 5, 7, 9] {
            for fam in 0..5 {
                let seed = base_seed() ^ 0xC0A4 ^ ((it as u64) << 20) ^ ((fam as u64) << 8) ^ n as u64;
                let mut rng = StdRng::seed_from_u64(seed);
                let mut c = family_circuit(&mut rng, n, fam);
                add_measurements(&mut rng, &mut c, n);
                let opts = random_opts(&mut rng);
                let plan = compile_sampling(&c, opts);
                let p = ref_distribution(&c);
                for comp in &plan.stats.components {
                    *tally.entry(format!("{:?}", comp.2)).or_insert(0) += 1;
                }
                *tally.entry(format!("suffix>0: {}", plan.stats.suffix_gates > 0)).or_insert(0) += 1;
                let ctx = format!("seed={seed} n={n} fam={fam} opts={opts:?} backends={:?}", plan.stats.components);
                let d = dist_diff(&plan.exact_distribution(), &p);
                assert!(d < 1e-10, "exact_distribution Δ={d:e} {ctx}\n{:?}", c.ops);
                let samples = plan.sample::<f64, _>(4000, &mut rng).unwrap();
                chi2_check(&samples, &p, &format!("f64 {ctx}"));
                let samples = plan.sample::<f32, _>(2000, &mut rng).unwrap();
                chi2_check(&samples, &p, &format!("f32 {ctx}"));
            }
        }
    }
    eprintln!("terminal sampling plans by backend: {tally:?}");
}

#[test]
fn sampling_plan_midcircuit_matches_reference() {
    for it in 0..10 * iters() {
        for &n in &[1usize, 2, 3, 5, 7] {
            let seed = base_seed() ^ 0xC0A5 ^ ((it as u64) << 20) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let fam = rng.random_range(0..5);
            let mut c = Circuit::new(n);
            for _ in 0..rng.random_range(1..4) {
                c.append(&family_circuit(&mut rng, n, fam));
                for _ in 0..rng.random_range(1..3) {
                    let q = edge_qubit(&mut rng, n);
                    c.measure(q);
                }
            }
            c.append(&family_circuit(&mut rng, n, fam));
            add_measurements(&mut rng, &mut c, n);
            let opts = random_opts(&mut rng);
            let plan = compile_sampling(&c, opts);
            let p = ref_distribution(&c);
            let ctx = format!("seed={seed} n={n} fam={fam} opts={opts:?} backends={:?}", plan.stats.components);
            let d = dist_diff(&plan.exact_distribution(), &p);
            assert!(d < 1e-10, "mid-circuit exact_distribution Δ={d:e} {ctx}\n{:?}", c.ops);
            let samples = plan.sample::<f64, _>(4000, &mut rng).unwrap();
            chi2_check(&samples, &p, &format!("mid f64 {ctx}"));
        }
    }
}

/// Wide Clifford+T / Clifford+Rz circuits with few measured qubits, so the
/// dispatcher actually picks the Pauli-path backend.
#[test]
fn sampling_plan_pauli_path_backend() {
    let mut used = 0;
    for it in 0..8 * iters() {
        for &n in &[8usize, 10, 12, 13] {
            let seed = base_seed() ^ 0xC0A6 ^ ((it as u64) << 20) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let mut c = random_circuit(&mut rng, n, 25, true, false);
            for _ in 0..rng.random_range(1..5) {
                let q = edge_qubit(&mut rng, n);
                c.gate(match rng.random_range(0..3) {
                    0 => Gate::T(q),
                    1 => Gate::Tdg(q),
                    _ => Gate::Rz(q, edge_angle(&mut rng)),
                });
                c.append(&random_circuit(&mut rng, n, 6, true, false));
            }
            let k = rng.random_range(1..4);
            for _ in 0..k {
                let q = edge_qubit(&mut rng, n);
                c.measure(q);
            }
            let plan = compile_sampling(&c, PlanOptions::default());
            if plan.stats.components.iter().any(|x| format!("{:?}", x.2) == "PauliPath") {
                used += 1;
            }
            let p = ref_distribution(&c);
            let ctx = format!("seed={seed} n={n} backends={:?}", plan.stats.components);
            let d = dist_diff(&plan.exact_distribution(), &p);
            assert!(d < 1e-10, "pauli-path plan exact_distribution Δ={d:e} {ctx}");
            let samples = plan.sample::<f64, _>(4000, &mut rng).unwrap();
            chi2_check(&samples, &p, &format!("pauli-path plan {ctx}"));
        }
    }
    eprintln!("Pauli-path backend used in {used} plans");
    assert!(used > 0, "dispatcher never chose Pauli paths");
}
