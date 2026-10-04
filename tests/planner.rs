//! Planner v0 and the MPS cost replay (src/planner.rs, src/mps_cost.rs).
//!
//! * the replay reproduces the real MPS engine's operation counts exactly
//!   when fed its bond trace;
//! * every bond bound dominates the real bond at every SVD (rigour);
//! * every engine the planner can choose (and every plan it makes) returns
//!   the reference value, checked against the independent audit state
//!   vector;
//! * the adder that made `Strategy::Auto` 350× slower than the frame;
//! * on the 314-instance simulability dataset, the planner's regret is no
//!   worse than the published leave-one-family-out result.

#[path = "audit_common/mod.rs"]
mod audit_common;

use audit_common::{random_circuit, RefSv};
use qsim_lab::adaptive::{self, AdaptiveOptions, Strategy};
use qsim_lab::mps_cost::{replay, replay_traced, BondSource, Estimator};
use qsim_lab::pauli_path::PauliSum;
use qsim_lab::planner::{self, Engine, PlanRequest, PlannerConfig};
use qsim_lab::simulability::{build, Spec};
use qsim_lab::{Circuit, Gate, Mps};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

const SPECS: &[&str] = &[
    "ct:n=9,L=3,t=0,nn=1",
    "ct:n=9,L=3,t=5,nn=1",
    "ct:n=10,L=4,t=12,nn=0",
    "ct:n=12,L=6,t=4,nn=1",
    "brick:n=8,D=3,nn=1",
    "brick:n=9,D=2,nn=0",
    "arith:bits=3,h=1,reps=2",
    "arith:bits=4,h=0,reps=1",
    "arith:bits=3,h=3,reps=3",
    "qaoa:n=8,p=2,deg=3,nn=0",
    "qaoa:n=9,p=1,deg=2,nn=1",
    "qaoa:n=10,p=2,deg=3,nn=1",
];

fn ref_z(c: &Circuit, obs: &[usize]) -> f64 {
    // the audit reference knows the textbook gate set; rewrite the rest
    // (U, iSWAP, Sx) with the crate's Clifford+Rz decomposition (exact up
    // to a global phase, which expectation values do not see).
    let mut d = Circuit::new(c.num_qubits);
    for g in c.gates() {
        match g {
            Gate::U(..) | Gate::ISwap(..) | Gate::ISwapdg(..) | Gate::Sx(_) | Gate::Sxdg(_) => {
                for h in g.decompose_to_clifford_rz() {
                    d.gate(h);
                }
            }
            _ => {
                d.gate(*g);
            }
        }
    }
    let s = RefSv::run(&d);
    s.a.iter()
        .enumerate()
        .map(|(x, a)| {
            let par = obs.iter().filter(|&&q| x >> q & 1 == 1).count() % 2;
            if par == 1 {
                -a.norm_sqr()
            } else {
                a.norm_sqr()
            }
        })
        .sum()
}

/// Family instances plus edge-biased random circuits with every gate type
/// (Toffoli, controlled phases, long-range SWAPs, U, iSWAP).
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
    let mut rng = StdRng::seed_from_u64(0x9_1A77);
    for i in 0..40 {
        let n = rng.random_range(2..=9);
        let depth = rng.random_range(1..=40);
        let mut c = random_circuit(&mut rng, n, depth, false, false);
        if i % 3 == 0 && n >= 2 {
            c.gate(Gate::U(0, 0.3, 1.1, -0.7));
            c.gate(Gate::ISwap(0, n - 1));
            c.gate(Gate::ISwapdg(n - 1, 0));
            c.gate(Gate::Sx(1));
        }
        out.push((format!("random#{i} n={n}"), c));
    }
    out
}

fn exact_mps(c: &Circuit) -> Mps {
    let mut m = Mps::new(c.num_qubits, 1 << 20);
    m.enable_trace();
    for g in c.gates() {
        m.apply_gate(g).unwrap();
    }
    m
}

#[test]
fn replay_with_trace_reproduces_engine_counts() {
    for (name, c) in corpus() {
        let m = exact_mps(&c);
        let r = replay(&c, BondSource::Trace(m.trace())).unwrap();
        assert_eq!(r.stats, m.stats(), "{name}");
        assert_eq!(
            r.max_bond,
            m.trace().iter().copied().max().unwrap_or(1).max(1) as usize,
            "{name}"
        );
    }
}

#[test]
fn every_bound_dominates_the_real_bond_at_every_step() {
    for (name, c) in corpus() {
        let m = exact_mps(&c);
        for e in Estimator::ALL {
            let b = replay_traced(&c, BondSource::Bound(e)).unwrap();
            assert_eq!(b.trace.len(), m.trace().len(), "{name} {e:?}");
            for (step, (x, y)) in b.trace.iter().zip(m.trace()).enumerate() {
                assert!(x >= y, "{name} {e:?} step {step}: bound {x} < real {y}");
            }
            let s = b.stats;
            assert!(
                s.svd_work >= m.stats().svd_work * (1.0 - 1e-12),
                "{name} {e:?}"
            );
        }
        // the best bound is the minimum of the others
        let best = replay_traced(&c, BondSource::Bound(Estimator::Best)).unwrap();
        for e in Estimator::ALL {
            let b = replay_traced(&c, BondSource::Bound(e)).unwrap();
            assert!(best.stats.svd_work <= b.stats.svd_work, "{name} {e:?}");
        }
    }
}

#[test]
fn stabilizer_bound_is_exact_on_clifford_circuits() {
    // for Clifford circuits the Stab bound is the exact Schmidt rank.
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..30 {
        let n = rng.random_range(2..=10);
        let c = random_circuit(&mut rng, n, 30, true, false);
        let m = exact_mps(&c);
        let b = replay_traced(&c, BondSource::Bound(Estimator::Stab)).unwrap();
        assert_eq!(b.trace, m.trace());
    }
}

fn check_engine(e: Engine, c: &Circuit, obs: &[usize], want: f64, name: &str) {
    let cfg = PlannerConfig {
        debug_reference: true,
        ..PlannerConfig::default()
    };
    let mut p = planner::plan(c, &PlanRequest::Expectation(obs.to_vec()), &cfg).unwrap();
    // force the engine (keep the ranking as fall-backs)
    p.engine = e;
    let got = planner::execute_expectation(&p, c, obs, &cfg).unwrap();
    assert!(
        (got.value - want).abs() < 1e-6,
        "{name}: forced {e:?} ran {:?}: {} vs {want}",
        got.engine,
        got.value
    );
}

#[test]
fn every_planned_and_forced_engine_is_exact() {
    let cfg = PlannerConfig {
        debug_reference: true,
        ..PlannerConfig::default()
    };
    let mut chosen: HashMap<Engine, usize> = HashMap::new();
    for (name, c) in corpus() {
        let n = c.num_qubits;
        for obs in [(0..n).collect::<Vec<_>>(), vec![n / 2], vec![0, n - 1]] {
            let want = ref_z(&c, &obs);
            let r = planner::expectation(&c, &obs, &cfg).unwrap();
            assert!(
                (r.value - want).abs() < 1e-6,
                "{name} obs {obs:?}: planned {:?}: {} vs {want}",
                r.engine,
                r.value
            );
            *chosen.entry(r.engine).or_default() += 1;
            for e in [
                Engine::StateVector,
                Engine::Sparse,
                Engine::Mps,
                Engine::Hsf,
                Engine::Compressed,
            ] {
                check_engine(e, &c, &obs, want, &name);
            }
        }
    }
    eprintln!("planned engines: {chosen:?}");
}

#[test]
fn probe_or_solve_is_exact() {
    // small caps exercise both outcomes: solved by the probe, and re-ranked
    // after a truncated probe.
    let (mut solved, mut reranked) = (0, 0);
    for cap in [2u32, 4, 16] {
        let cfg = PlannerConfig {
            debug_reference: true,
            probe_cap: Some(cap),
            probe_frac: 1e9,
            sv_shortcut_secs: 0.0,
            mps_feature_min_secs: 0.0,
            hsf_feature_min_secs: 0.0,
            ..PlannerConfig::default()
        };
        for (name, c) in corpus() {
            let n = c.num_qubits;
            let obs: Vec<usize> = (0..n).step_by(2).collect();
            let want = ref_z(&c, &obs);
            let p = planner::plan(&c, &PlanRequest::Expectation(obs.clone()), &cfg).unwrap();
            match p.probe {
                Some((true, false)) => solved += 1,
                Some((true, true)) => reranked += 1,
                _ => {}
            }
            let r = planner::execute_expectation(&p, &c, &obs, &cfg).unwrap();
            assert!(
                (r.value - want).abs() < 1e-6,
                "{name} cap {cap}: {} vs {want}",
                r.value
            );
        }
    }
    assert!(
        solved > 0 && reranked > 0,
        "solved {solved} reranked {reranked}"
    );
}

#[test]
fn speculative_mps_falls_back_exactly() {
    // A tiny deadline forces the abort path: the runner-up must answer.
    let c = build(&Spec::parse("brick:n=12,D=8,nn=0").unwrap(), 3).unwrap();
    let obs: Vec<usize> = (0..12).collect();
    let want = ref_z(&c, &obs);
    let cfg = PlannerConfig {
        speculate: 1e-9,
        min_deadline_secs: 0.0,
        ..PlannerConfig::default()
    };
    let mut p = planner::plan(&c, &PlanRequest::Expectation(obs.clone()), &cfg).unwrap();
    p.engine = Engine::Mps;
    let r = planner::execute_expectation(&p, &c, &obs, &cfg).unwrap();
    assert_eq!(r.aborted.first(), Some(&Engine::Mps));
    assert_ne!(r.engine, Engine::Mps);
    assert!((r.value - want).abs() < 1e-9);
}

/// `arith:bits=12,h=2,reps=1` (25 qubits): measured on the Mac, Auto
/// 1.47 s vs frame 4 ms (Auto handed over to a 25-qubit dense register
/// with 4 live terms). The fixed Auto must stay in the Heisenberg frame,
/// and the planner must pick the measured winner (sparse, 63 µs).
#[test]
fn adder_auto_misfire_is_fixed() {
    let c = build(&Spec::parse("arith:bits=12,h=2,reps=1").unwrap(), 1).unwrap();
    let n = c.num_qubits;
    let obs: Vec<usize> = (0..n).collect();
    let o = PauliSum::z_product(n, &obs);
    let r = adaptive::expectation(&c, &o, &AdaptiveOptions::default()).unwrap();
    // it may still finish densely, but only on a small register late in
    // the sweep (the old policy took the 25-qubit register at the start).
    assert!(
        r.switched_at.is_none() || r.dense_qubits <= 16,
        "Auto switched at {:?} to {} qubits",
        r.switched_at,
        r.dense_qubits
    );
    let f = adaptive::expectation(
        &c,
        &o,
        &AdaptiveOptions {
            strategy: Strategy::Frame,
            ..Default::default()
        },
    )
    .unwrap();
    assert!((r.value - f.value).abs() < 1e-12);
    // the old behaviour is still reachable with the guard off
    let old = adaptive::expectation(
        &c,
        &o,
        &AdaptiveOptions {
            flat_evidence: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(old.dense_qubits, 25);
    assert!((old.value - f.value).abs() < 1e-9);

    let p = planner::plan(
        &c,
        &PlanRequest::Expectation(obs.clone()),
        &PlannerConfig::default(),
    )
    .unwrap();
    assert_eq!(p.engine, Engine::Sparse, "ranked {:?}", p.ranked);
}

// ---------------------------------------------------------------------------
// Dataset regret

fn parse_csv_line(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut q = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '"' if q && chars.peek() == Some(&'"') => {
                cur.push('"');
                chars.next();
            }
            '"' => q = !q,
            ',' if !q => out.push(std::mem::take(&mut cur)),
            _ => cur.push(ch),
        }
    }
    out.push(cur);
    out
}

/// Geometric-mean regret of the planner on the Mac dataset (state engines,
/// request `<Z^{⊗n}>`), against research/simulability.md §5 (held-out
/// 1.25×, top-1 85 %).
#[test]
fn dataset_regret_no_worse_than_published() {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/research/data/simulability/raw"
    );
    let mut inst: HashMap<(String, u64), HashMap<String, (String, f64)>> = HashMap::new();
    for f in ["ct24", "ct32", "ctnn0", "brick", "arith", "qaoa"] {
        let text = std::fs::read_to_string(format!("{dir}/{f}.csv")).unwrap();
        let mut lines = text.lines();
        let head = parse_csv_line(lines.next().unwrap());
        let col = |k: &str| head.iter().position(|h| h == k).unwrap();
        let (cs, cd, ce, cst, csec, cwall) = (
            col("spec"),
            col("seed"),
            col("engine"),
            col("status"),
            col("secs"),
            col("wall"),
        );
        for line in lines {
            let r = parse_csv_line(line);
            let t = if r[cst] == "ok" {
                r[csec].parse().unwrap()
            } else {
                r[cwall].parse().unwrap_or(10.0)
            };
            inst.entry((r[cs].clone(), r[cd].parse().unwrap()))
                .or_default()
                .insert(r[ce].clone(), (r[cst].clone(), t));
        }
    }
    let name = |e: Engine| match e {
        Engine::Tableau => "tableau",
        Engine::StateVector => "sv",
        Engine::Sparse => "sparse",
        Engine::Mps => "mps",
        Engine::Hsf => "hsf",
        Engine::Compressed => "cstate",
        Engine::Zero => "zero",
    };
    // the model's choice (staging off: every feature computed)
    let cfg = PlannerConfig {
        use_certificate: false,
        sv_shortcut_secs: 0.0,
        mps_feature_min_secs: 0.0,
        hsf_feature_min_secs: 0.0,
        ..PlannerConfig::default()
    };
    // the shipped, staged planner (skips features that cannot pay off):
    // judged by the epsilon-regret (+1 ms on both sides), since staging
    // only gives up sub-millisecond differences.
    let staged = PlannerConfig {
        use_certificate: false,
        ..PlannerConfig::default()
    };
    let mut eps_log = 0.0f64;
    let (mut n, mut top1, mut logsum) = (0usize, 0usize, 0.0f64);
    let mut worst = (1.0f64, String::new());
    let mut keys: Vec<_> = inst.keys().cloned().collect();
    keys.sort();
    for key in keys {
        let runs = &inst[&key];
        let state = ["sv", "sparse", "mps", "hsf", "tableau", "cstate"];
        let best = state
            .iter()
            .filter_map(|e| runs.get(*e).filter(|r| r.0 == "ok").map(|r| (*e, r.1)))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        let Some((wname, wt)) = best else { continue };
        let timeout = runs
            .values()
            .filter(|r| r.0 == "timeout")
            .map(|r| r.1)
            .fold(0.0, f64::max);
        let timeout = if timeout > 0.0 { timeout } else { 10.0 };
        let c = build(&Spec::parse(&key.0).unwrap(), key.1).unwrap();
        let all: Vec<usize> = (0..c.num_qubits).collect();
        // rank state engines only (the dataset compares state engines; the
        // certificate short-cut is evaluated separately)
        let p = planner::plan(&c, &PlanRequest::Expectation(all), &cfg).unwrap();
        let chosen = name(p.engine);
        let t = match runs.get(chosen) {
            Some((s, t)) if s == "ok" => *t,
            _ => 2.0 * timeout,
        };
        n += 1;
        top1 += usize::from(chosen == wname);
        let reg = t / wt;
        logsum += reg.log10();
        let ps = planner::plan(
            &c,
            &PlanRequest::Expectation((0..c.num_qubits).collect()),
            &staged,
        )
        .unwrap();
        let ts = match runs.get(name(ps.engine)) {
            Some((s, t)) if s == "ok" => *t,
            _ => 2.0 * timeout,
        };
        eps_log += ((ts + 1e-3) / (wt + 1e-3)).log10();
        if reg > worst.0 {
            worst = (reg, format!("{} -> {chosen} (best {wname})", key.0));
        }
    }
    let geo = 10f64.powf(logsum / n as f64);
    let acc = top1 as f64 / n as f64;
    let geo_eps_staged = 10f64.powf(eps_log / n as f64);
    eprintln!(
        "planner on dataset: n={n} top1={acc:.3} geo regret={geo:.3} worst={worst:?}; \
         staged geo eps-regret={geo_eps_staged:.3}"
    );
    assert!(geo_eps_staged <= 1.15, "staged eps-regret {geo_eps_staged}");
    assert!(n >= 300);
    // published (held-out): 1.25x, 85 %, worst 161x. In-sample with the
    // replayed MPS work this is 1.09x / 88 % / 12.9x; guard with margin.
    assert!(geo <= 1.15, "geo regret {geo}");
    assert!(acc >= 0.85, "top-1 {acc}");
    assert!(worst.0 <= 20.0, "worst {worst:?}");
}
