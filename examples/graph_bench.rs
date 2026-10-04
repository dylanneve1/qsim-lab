//! Graph compiler benchmarks (research/graph-compiler.md).
//!
//! `cargo run --release --example graph_bench -- bind [workload n p binds]`
//!
//! Variants are interleaved, min of `REPS` (default 3) reported.

use qsim_lab::blocked::BlockConfig;
use qsim_lab::graph::observable::{diagonal_expectation, pauli_expectation};
use qsim_lab::graph::{Angle, CompiledCircuit, GraphOptions, Observable, ParamCircuit};
use qsim_lab::pipeline::{self, Budget, Output, Request};
use qsim_lab::{Gate, StateVectorF64};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

fn reps() -> usize {
    std::env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3)
}

/// Random 3-regular-ish graph (pairing model, simple edges only).
fn regular3(n: usize, seed: u64) -> Vec<(usize, usize)> {
    let mut rng = StdRng::seed_from_u64(seed);
    loop {
        let mut stubs: Vec<usize> = (0..n).flat_map(|q| [q, q, q]).collect();
        for i in (1..stubs.len()).rev() {
            stubs.swap(i, rng.random_range(0..=i));
        }
        let mut e: Vec<(usize, usize)> = stubs
            .chunks(2)
            .map(|c| (c[0].min(c[1]), c[0].max(c[1])))
            .collect();
        e.sort_unstable();
        let ok = e.iter().all(|&(a, b)| a != b) && e.windows(2).all(|w| w[0] != w[1]);
        if ok {
            return e;
        }
    }
}

pub fn qaoa(n: usize, p: usize) -> (ParamCircuit, Observable) {
    let edges = regular3(n, 7 + n as u64);
    let mut pc = ParamCircuit::new(n, 2 * p);
    for q in 0..n {
        pc.gate(Gate::H(q));
    }
    for l in 0..p {
        for &(a, b) in &edges {
            pc.rzz(a, b, Angle::param(2 * l));
        }
        for q in 0..n {
            pc.rx(q, Angle::scaled(2 * l + 1, 2.0));
        }
    }
    let mut o = Observable::new();
    for &(a, b) in &edges {
        o.zz(-0.5, a, b);
    }
    (pc, o)
}

/// Hardware-efficient ansatz: L layers of Ry, Rz on every qubit and a CNOT
/// ladder; observable: TFIM `Σ Z_i Z_{i+1} + 0.7 Σ X_i`.
pub fn hea(n: usize, layers: usize) -> (ParamCircuit, Observable) {
    let mut pc = ParamCircuit::new(n, 2 * n * layers);
    let mut k = 0;
    for _ in 0..layers {
        for q in 0..n {
            pc.ry(q, Angle::param(k));
            pc.rz(q, Angle::param(k + 1));
            k += 2;
        }
        for q in 0..n - 1 {
            pc.gate(Gate::Cnot(q, q + 1));
        }
    }
    let mut o = Observable::new();
    for q in 0..n - 1 {
        o.zz(1.0, q, q + 1);
    }
    for q in 0..n {
        o.add(0.7, &format!("X{q}")).unwrap();
    }
    (pc, o)
}

/// TFIM Trotter evolution, `steps` steps, parameters `(J dt, h dt)`;
/// observable `Σ Z_i`.
pub fn trotter(n: usize, steps: usize) -> (ParamCircuit, Observable) {
    let mut pc = ParamCircuit::new(n, 2);
    for _ in 0..steps {
        for q in 0..n - 1 {
            pc.rzz(q, q + 1, Angle::scaled(0, 2.0));
        }
        for q in 0..n {
            pc.rx(q, Angle::scaled(1, 2.0));
        }
    }
    let mut o = Observable::new();
    for q in 0..n {
        o.add(1.0 / n as f64, &format!("Z{q}")).unwrap();
    }
    (pc, o)
}

/// Same observable evaluation as the compiled path, on a plain state.
fn eval_obs(sv: &StateVectorF64, o: &Observable) -> f64 {
    let mut diag = Vec::new();
    let mut s = 0.0;
    for t in &o.terms {
        if t.x == 0 {
            diag.push((t.coef, t.z as u64));
        } else {
            s += t.coef * pauli_expectation(sv.amplitudes(), t.x as u64, t.z as u64);
        }
    }
    s + diagonal_expectation(sv.amplitudes(), &diag)
}

/// Baseline: bind to a plain circuit, run the blocked executor (lowering,
/// fusion, stage plan, preparation every time), evaluate.
fn baseline(pc: &ParamCircuit, o: &Observable, p: &[f64]) -> f64 {
    let c = pc.bind(p).unwrap();
    let mut sv = StateVectorF64::new(pc.num_qubits);
    sv.apply_circuit_blocked(&c, &BlockConfig::default())
        .unwrap();
    eval_obs(&sv, o)
}

/// User-level baseline: `pipeline::simulate` once per Z-product term.
fn baseline_pipeline(pc: &ParamCircuit, o: &Observable, p: &[f64]) -> Option<f64> {
    if !o.is_diagonal() {
        return None;
    }
    let c = pc.bind(p).unwrap();
    let mut s = 0.0;
    for t in &o.terms {
        let qs: Vec<usize> = (0..128).filter(|&q| t.z >> q & 1 == 1).collect();
        match pipeline::simulate(&c, &Request::Expectation(qs), &Budget::default())
            .ok()?
            .output
        {
            Output::Expectation(v) => s += t.coef * v,
            _ => return None,
        }
    }
    Some(s)
}

fn time<F: FnMut() -> R, R>(mut f: F) -> (f64, R) {
    let t = Instant::now();
    let r = f();
    (t.elapsed().as_secs_f64(), r)
}

fn build(w: &str, n: usize, p: usize) -> (ParamCircuit, Observable) {
    match w {
        "qaoa" => qaoa(n, p),
        "hea" => hea(n, p),
        "trotter" => trotter(n, p),
        _ => panic!("workload qaoa|hea|trotter"),
    }
}

fn bind_bench(w: &str, n: usize, p: usize, binds: usize, pipe: bool) {
    let (pc, o) = build(w, n, p);
    let mut rng = StdRng::seed_from_u64(11);
    let ps: Vec<Vec<f64>> = (0..binds)
        .map(|_| {
            (0..pc.num_params)
                .map(|_| rng.random_range(-1.5..1.5))
                .collect()
        })
        .collect();
    let opts = GraphOptions::default();
    let mut best = [f64::INFINITY; 8];
    let mut check = (0.0, 0.0);
    for _ in 0..reps() {
        let (tc, cc) = time(|| CompiledCircuit::compile(&pc, Some(&o), &opts).unwrap());
        best[0] = best[0].min(tc);
        let (tb, _) = time(|| {
            for p in &ps {
                std::hint::black_box(cc.bind(p).unwrap());
            }
        });
        best[1] = best[1].min(tb / binds as f64);
        let (t, a) = time(|| {
            ps.iter()
                .map(|p| cc.bind(p).unwrap().expectation().unwrap())
                .sum::<f64>()
        });
        best[2] = best[2].min(t / binds as f64);
        let (t, b) = time(|| ps.iter().map(|p| baseline(&pc, &o, p)).sum::<f64>());
        best[3] = best[3].min(t / binds as f64);
        check = (a, b);
        let (t, _) = time(|| cc.sweep_expectation(&ps).unwrap());
        best[4] = best[4].min(t / binds as f64);
        let (t, _) = time(|| ps.par_iter().map(|p| baseline(&pc, &o, p)).sum::<f64>());
        best[5] = best[5].min(t / binds as f64);
        if pipe {
            let k = binds.min(20);
            let (t, _) = time(|| {
                ps[..k]
                    .iter()
                    .map(|p| baseline_pipeline(&pc, &o, p).unwrap_or(f64::NAN))
                    .sum::<f64>()
            });
            best[6] = best[6].min(t / k as f64);
        }
    }
    assert!((check.0 - check.1).abs() < 1e-8 * binds as f64, "{check:?}");
    let cc = CompiledCircuit::compile(&pc, Some(&o), &opts).unwrap();
    let st = cc.stats();
    println!(
        "{w} n={n} p={p} ops={} cone={} parts={:?} binds={binds} | compile {:.3}ms bind {:.1}us | per-bind: compiled {:.3}ms baseline {:.3}ms ({:.2}x) | sweep: compiled-par {:.3}ms baseline-par {:.3}ms ({:.2}x) vs serial baseline {:.2}x{}",
        pc.ops.len(),
        st.ops_after_cone,
        st.parts,
        best[0] * 1e3,
        best[1] * 1e6,
        best[2] * 1e3,
        best[3] * 1e3,
        best[3] / best[2],
        best[4] * 1e3,
        best[5] * 1e3,
        best[5] / best[4],
        best[3] / best[4],
        if pipe {
            format!(" | pipeline-per-term {:.3}ms ({:.1}x vs compiled)", best[6] * 1e3, best[6] / best[2])
        } else {
            String::new()
        }
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pipe = std::env::var("PIPE").is_ok();
    match args.get(1).map(|s| s.as_str()) {
        Some("bind") if args.len() >= 6 => {
            let n = args[3].parse().unwrap();
            let p = args[4].parse().unwrap();
            let b = args[5].parse().unwrap();
            bind_bench(&args[2], n, p, b, pipe);
        }
        Some("bind") | None => {
            println!("# load {:?}", std::fs::read_to_string("/proc/loadavg").ok());
            for &(w, n, p, b) in &[
                ("qaoa", 10, 3, 2000),
                ("qaoa", 16, 3, 300),
                ("qaoa", 20, 3, 20),
                ("hea", 8, 4, 2000),
                ("hea", 12, 4, 1000),
                ("hea", 16, 4, 100),
                ("trotter", 12, 20, 500),
                ("trotter", 20, 20, 10),
            ] {
                bind_bench(w, n, p, b, pipe);
            }
        }
        _ => eprintln!("usage: graph_bench bind [workload n p binds]"),
    }
}
