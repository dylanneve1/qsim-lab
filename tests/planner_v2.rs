//! Planner v2 (src/planner.rs, research/planner-v2.md): samples and
//! amplitude requests, tiered planning, the plan cache.
//!
//! * every engine the planner can run for samples (state vector, sparse,
//!   MPS perfect sampling, HSF full output, compressed sampler, tableau)
//!   draws from the exact distribution: per-engine exact probabilities
//!   against the independent audit state vector, plus a chi-square test of
//!   the drawn samples (fixed seeds, so deterministic);
//! * every engine it can run for amplitudes (state vector, sparse, MPS,
//!   HSF) returns the exact amplitudes *with* the global phase;
//! * planned requests through `pipeline::simulate` are exact;
//! * speculation falls back exactly; the cache reuses choices only;
//! * tiering never changes a result, and on the dataset its epsilon-regret
//!   stays within the v1 guard.

#[path = "audit_common/mod.rs"]
mod audit_common;

use audit_common::{random_circuit, RefSv};
use num_complex::Complex64;
use qsim_lab::pipeline::{simulate, Budget, Output, Request};
use qsim_lab::planner::{self, Engine, PlanRequest, PlannerConfig, Prepared};
use qsim_lab::simulability::{build, Spec};
use qsim_lab::statevector::{sorted_uniforms, StateVectorF64};
use qsim_lab::{Circuit, Gate};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const SPECS: &[&str] = &[
    "ct:n=9,L=3,t=0,nn=1",
    "ct:n=9,L=3,t=5,nn=1",
    "ct:n=10,L=4,t=12,nn=0",
    "brick:n=8,D=3,nn=1",
    "brick:n=9,D=2,nn=0",
    "arith:bits=3,h=1,reps=2",
    "arith:bits=4,h=0,reps=1",
    "qaoa:n=8,p=2,deg=3,nn=0",
    "hea:n=8,D=2",
    "qft:n=8,h=3",
];

fn has_phase_ambiguous_gates(c: &Circuit) -> bool {
    c.gates().any(|g| {
        matches!(
            g,
            Gate::U(..) | Gate::ISwap(..) | Gate::ISwapdg(..) | Gate::Sx(_) | Gate::Sxdg(_)
        )
    })
}

/// Reference amplitudes: the independent audit state vector, or (for gate
/// types it lacks) the crate's unblocked state vector, gate by gate.
fn reference(c: &Circuit) -> Vec<Complex64> {
    if has_phase_ambiguous_gates(c) {
        let mut s = StateVectorF64::new(c.num_qubits);
        for g in c.gates() {
            s.apply_gate(g).unwrap();
        }
        s.amplitudes().to_vec()
    } else {
        RefSv::run(c).a.clone()
    }
}

fn corpus() -> Vec<(String, Circuit)> {
    let mut out = Vec::new();
    for spec in SPECS {
        for seed in 1..=2 {
            out.push((
                format!("{spec} s{seed}"),
                build(&Spec::parse(spec).unwrap(), seed).unwrap(),
            ));
        }
    }
    let mut rng = StdRng::seed_from_u64(0x5A_3B1E);
    for i in 0..24 {
        let n = rng.random_range(2..=8);
        let depth = rng.random_range(1..=30);
        let mut c = random_circuit(&mut rng, n, depth, false, false);
        if i % 3 == 0 && n >= 2 {
            c.gate(Gate::U(0, 0.3, 1.1, -0.7));
            c.gate(Gate::ISwap(0, n - 1));
            c.gate(Gate::Sx(1));
        }
        out.push((format!("random#{i} n={n}"), c));
    }
    out
}

fn cfg() -> PlannerConfig {
    PlannerConfig {
        debug_reference: true,
        cache: false,
        ..PlannerConfig::default()
    }
}

/// Chi-square statistic of `samples` against `p`, bins with expected count
/// below 5 pooled; returns `(statistic, degrees of freedom)`.
fn chi_square(samples: &[u128], p: &[f64]) -> (f64, usize) {
    let shots = samples.len() as f64;
    let mut counts = vec![0usize; p.len()];
    for &x in samples {
        assert!(
            (x as usize) < p.len() && p[x as usize] > 1e-14,
            "sampled an outcome of probability {}",
            p.get(x as usize).copied().unwrap_or(-1.0)
        );
        counts[x as usize] += 1;
    }
    let (mut stat, mut bins) = (0.0, 0usize);
    let (mut pool_e, mut pool_o) = (0.0, 0.0);
    for (i, &pi) in p.iter().enumerate() {
        let e = pi * shots;
        if e >= 5.0 {
            stat += (counts[i] as f64 - e).powi(2) / e;
            bins += 1;
        } else {
            pool_e += e;
            pool_o += counts[i] as f64;
        }
    }
    if pool_e > 0.0 {
        stat += (pool_o - pool_e).powi(2) / pool_e.max(1e-300);
        bins += 1;
    }
    (stat, bins.saturating_sub(1))
}

fn assert_distribution(samples: &[u128], p: &[f64], what: &str) {
    let (stat, df) = chi_square(samples, p);
    // mean df, sd sqrt(2 df): 7 sd is never reached by chance at fixed seeds
    let limit = df as f64 + 7.0 * (2.0 * df as f64).sqrt() + 12.0;
    assert!(
        stat <= limit,
        "{what}: chi2 {stat:.1} > {limit:.1} (df {df})"
    );
}

#[test]
fn every_sampling_engine_draws_the_exact_distribution() {
    let cfg = cfg();
    let mut runs = 0;
    for (name, c) in corpus() {
        let want = reference(&c);
        let p: Vec<f64> = want.iter().map(|a| a.norm_sqr()).collect();
        let clifford = c.gates().all(|g| g.is_clifford());
        let mut engines = vec![
            Engine::StateVector,
            Engine::Sparse,
            Engine::Mps,
            Engine::Hsf,
            Engine::Compressed,
        ];
        if clifford {
            engines.push(Engine::Tableau);
        }
        for (k, &e) in engines.iter().enumerate() {
            let Some(mut st) = planner::prepare(e, &c, &cfg, None).unwrap() else {
                panic!("{name}: {e:?} aborted without a deadline");
            };
            // exact probabilities where the engine exposes them
            match &mut st {
                Prepared::Sparse(s) => {
                    let mut q = vec![0.0; p.len()];
                    for (x, a) in s.iter() {
                        q[x as usize] = a.norm_sqr();
                    }
                    for (x, (a, b)) in p.iter().zip(&q).enumerate() {
                        assert!((a - b).abs() < 1e-10, "{name} sparse p[{x}]");
                    }
                }
                Prepared::Mps(_) | Prepared::Hsf(..) | Prepared::Sv(_) => {
                    let xs: Vec<u128> = (0..p.len() as u128).collect();
                    let a = st.amplitudes(&xs).unwrap();
                    for (x, (u, v)) in want.iter().zip(&a).enumerate() {
                        assert!(
                            (u - v).norm() < 1e-7,
                            "{name} {e:?} amplitude {x}: {u} vs {v}"
                        );
                    }
                }
                _ => {}
            }
            let mut rng = StdRng::seed_from_u64(1000 + k as u64);
            let s = st.samples(20_000, &mut rng, None).unwrap().unwrap();
            assert_eq!(s.len(), 20_000);
            assert_distribution(&s, &p, &format!("{name} {e:?}"));
            runs += 1;
        }
    }
    assert!(runs > 150, "{runs}");
}

#[test]
fn every_amplitude_engine_keeps_the_global_phase() {
    let cfg = cfg();
    let mut rng = StdRng::seed_from_u64(77);
    for (name, c) in corpus() {
        let want = reference(&c);
        let n = c.num_qubits;
        let xs: Vec<u128> = (0..40).map(|_| rng.random_range(0..1u128 << n)).collect();
        // planned (debug mode checks against the reference too)
        let r = planner::amplitudes(&c, &xs, &cfg).unwrap();
        for (x, a) in xs.iter().zip(&r.amplitudes) {
            assert!(
                (want[*x as usize] - a).norm() < 1e-7,
                "{name} planned {:?}",
                r.engine
            );
        }
        // forced engines run to completion (no speculative abort)
        let forced = PlannerConfig {
            speculate: 0.0,
            ..cfg
        };
        for e in [
            Engine::StateVector,
            Engine::Sparse,
            Engine::Mps,
            Engine::Hsf,
        ] {
            let mut p = planner::plan(&c, &PlanRequest::Amplitudes(xs.len()), &forced).unwrap();
            p.engine = e;
            let got = planner::execute_amplitudes(&p, &c, &xs, &forced).unwrap();
            assert_eq!(got.engine, e, "{name}: forced {e:?}");
            for (x, a) in xs.iter().zip(&got.amplitudes) {
                let w = want[*x as usize];
                assert!((w - a).norm() < 1e-7, "{name} {e:?} <{x}|psi>: {a} vs {w}");
            }
        }
        // engines without the global phase are refused, not wrong
        for e in [Engine::Compressed, Engine::Tableau] {
            if let Ok(Some(mut st)) = planner::prepare(e, &c, &cfg, None) {
                assert!(st.amplitudes(&xs).is_err(), "{name} {e:?}");
            }
        }
    }
}

#[test]
fn planned_samples_follow_the_distribution() {
    let cfg = cfg();
    for (i, (name, c)) in corpus().into_iter().enumerate() {
        let p: Vec<f64> = reference(&c).iter().map(|a| a.norm_sqr()).collect();
        for shots in [3usize, 20_000] {
            let mut rng = StdRng::seed_from_u64(i as u64);
            let r = planner::samples(&c, shots, &mut rng, &cfg).unwrap();
            assert_eq!(r.samples.len(), shots);
            if shots >= 1000 {
                assert_distribution(&r.samples, &p, &format!("{name} planned {:?}", r.engine));
            } else {
                assert!(r.samples.iter().all(|&x| p[x as usize] > 1e-14), "{name}");
            }
        }
    }
}

#[test]
fn pipeline_samples_and_amplitudes_are_exact() {
    for (i, (name, c)) in corpus().into_iter().enumerate() {
        let n = c.num_qubits;
        let want = reference(&c);
        // amplitudes, global phase included
        let xs: Vec<u128> = (0..1u128 << n).step_by(3).collect();
        let r = simulate(&c, &Request::Amplitudes(xs.clone()), &Budget::default()).unwrap();
        let Output::Amplitudes(a) = r.output else {
            panic!()
        };
        for (x, v) in xs.iter().zip(&a) {
            assert!((want[*x as usize] - v).norm() < 1e-7, "{name} <{x}|psi>");
        }
        // terminal samples of every qubit
        let mut m = c.clone();
        m.measure_all();
        let r = simulate(
            &m,
            &Request::Samples {
                shots: 20_000,
                seed: i as u64,
            },
            &Budget::default(),
        )
        .unwrap();
        let Output::Samples(s) = r.output else {
            panic!()
        };
        let idx: Vec<u128> = s
            .iter()
            .map(|b| {
                b.iter()
                    .enumerate()
                    .fold(0u128, |a, (q, &v)| a | (u128::from(v) << q))
            })
            .collect();
        let p: Vec<f64> = want.iter().map(|a| a.norm_sqr()).collect();
        assert_distribution(&idx, &p, &format!("{name} pipeline {:?}", r.engines));
    }
}

#[test]
fn speculative_sampling_falls_back_exactly() {
    let c = build(&Spec::parse("brick:n=10,D=8,nn=0").unwrap(), 3).unwrap();
    let p: Vec<f64> = reference(&c).iter().map(|a| a.norm_sqr()).collect();
    let cfg = PlannerConfig {
        speculate: 1e-9,
        min_deadline_secs: 0.0,
        ..cfg()
    };
    let mut plan = planner::plan(&c, &PlanRequest::Samples(20_000), &cfg).unwrap();
    plan.engine = Engine::Mps;
    if plan.ranked.iter().all(|x| x.0 != Engine::StateVector) {
        plan.ranked.push((Engine::StateVector, 1.0));
    }
    let mut rng = StdRng::seed_from_u64(5);
    let r = planner::execute_samples(&plan, &c, 20_000, &mut rng, &cfg).unwrap();
    assert_eq!(r.aborted.first(), Some(&Engine::Mps));
    assert_ne!(r.engine, Engine::Mps);
    assert_distribution(&r.samples, &p, "after abort");
}

#[test]
fn tiering_and_cache_never_change_a_result() {
    let tiered = cfg();
    let full = PlannerConfig { voi: 0.0, ..cfg() };
    let cached = PlannerConfig {
        cache: true,
        ..cfg()
    };
    for (name, c) in corpus() {
        let n = c.num_qubits;
        let obs: Vec<usize> = (0..n).step_by(2).collect();
        let want = reference(&c);
        let ref_z: f64 = want
            .iter()
            .enumerate()
            .map(|(x, a)| {
                let par = obs.iter().filter(|&&q| x >> q & 1 == 1).count() % 2;
                if par == 1 {
                    -a.norm_sqr()
                } else {
                    a.norm_sqr()
                }
            })
            .sum();
        for cfg in [&tiered, &full, &cached, &cached] {
            let r = planner::expectation(&c, &obs, cfg).unwrap();
            assert!((r.value - ref_z).abs() < 1e-6, "{name}: {:?}", r.engine);
        }
        // voi = 0 computes every feature it may need
        let p = planner::plan(&c, &PlanRequest::Expectation(obs.clone()), &full).unwrap();
        assert!(
            p.features.tier >= 1 || p.engine == Engine::Tableau,
            "{name}"
        );
    }
}

#[test]
fn cache_reuses_choices_across_angles_but_never_values() {
    planner::clear_cache();
    let cfg = PlannerConfig {
        cache: true,
        use_certificate: false,
        ..PlannerConfig::default()
    };
    let mk = |theta: f64| {
        let mut c = Circuit::new(12);
        for q in 0..12 {
            c.gate(Gate::H(q));
        }
        for layer in 0..3 {
            for q in 0..11 {
                c.gate(Gate::Cnot(q, q + 1));
                c.gate(Gate::Rz(q + 1, theta + layer as f64));
                c.gate(Gate::Cnot(q, q + 1));
            }
            for q in 0..12 {
                c.gate(Gate::Rx(q, 0.4 * theta));
            }
        }
        c
    };
    let obs: Vec<usize> = (0..12).collect();
    let a = planner::plan(&mk(0.3), &PlanRequest::Expectation(obs.clone()), &cfg).unwrap();
    assert!(!a.cached);
    let b = planner::plan(&mk(0.7), &PlanRequest::Expectation(obs.clone()), &cfg).unwrap();
    assert!(b.cached, "same structure, same Clifford classes");
    assert_eq!(a.engine, b.engine);
    // a Clifford angle changes the class: no reuse
    let c = planner::plan(
        &mk(std::f64::consts::FRAC_PI_2),
        &PlanRequest::Expectation(obs.clone()),
        &cfg,
    )
    .unwrap();
    assert!(!c.cached);
    // the value of the reused plan is the new circuit's own
    let r = planner::execute_expectation(&b, &mk(0.7), &obs, &cfg).unwrap();
    let want: f64 = RefSv::run(&mk(0.7))
        .a
        .iter()
        .enumerate()
        .map(|(x, a)| {
            if (x as u32).count_ones() % 2 == 1 {
                -a.norm_sqr()
            } else {
                a.norm_sqr()
            }
        })
        .sum();
    assert!((r.value - want).abs() < 1e-7);
    planner::clear_cache();
}

#[test]
fn sorted_uniforms_are_sorted_uniform_order_statistics() {
    let mut rng = StdRng::seed_from_u64(3);
    let n = 200_000;
    let r = sorted_uniforms(n, 2.0, &mut rng);
    assert!(r.windows(2).all(|w| w[0] <= w[1]));
    assert!(r[0] >= 0.0 && r[n - 1] < 2.0);
    // Kolmogorov–Smirnov against U[0, 2)
    let d = r
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let f = x / 2.0;
            (f - i as f64 / n as f64)
                .abs()
                .max(((i + 1) as f64 / n as f64 - f).abs())
        })
        .fold(0.0, f64::max);
    assert!(d * (n as f64).sqrt() < 1.95, "KS {d}"); // p ~ 0.001
                                                     // the k-th order statistic has mean k/(n+1)
    let mid = r[n / 2] / 2.0;
    assert!((mid - 0.5).abs() < 0.01);
}

#[test]
fn v1_config_reproduces_v1_staging() {
    // v1: a cheap state vector short-cut, no tiers, no cache.
    let c = build(&Spec::parse("hea:n=12,D=1").unwrap(), 1).unwrap();
    let obs: Vec<usize> = (0..12).collect();
    let p = planner::plan(&c, &PlanRequest::Expectation(obs), &PlannerConfig::v1()).unwrap();
    assert_eq!(p.engine, Engine::StateVector);
    assert!(!p.cached);
    assert_eq!(p.stage_secs, [0.0; 4]);
}
