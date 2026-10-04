//! Graph compiler benchmarks (research/graph-compiler.md).
//!
//! `cargo run --release --example graph_bench -- bind [workload n p binds]`
//!
//! Variants are interleaved, min of `REPS` (default 3) reported.

use qsim_lab::blocked::BlockConfig;
use qsim_lab::graph::observable::{diagonal_expectation, pauli_expectation};
use qsim_lab::graph::{
    phase_regions, Angle, CompiledCircuit, GraphOptions, Observable, POp, ParamCircuit,
    RewriteOptions,
};
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

/// QAOA with the cost layer written as `CNOT · Rz · CNOT` (as transpilers
/// emit it).
pub fn qaoa_cx(n: usize, p: usize) -> (ParamCircuit, Observable) {
    let (pc, o) = qaoa(n, p);
    let mut out = ParamCircuit::new(n, pc.num_params);
    for op in pc.ops {
        match op {
            POp::Rzz(a, b, ang) => {
                out.gate(Gate::Cnot(a, b));
                out.rz(b, ang);
                out.gate(Gate::Cnot(a, b));
            }
            op => {
                out.push(op);
            }
        }
    }
    (out, o)
}

/// Trotterised random Pauli Hamiltonian (weights 2..=4, JW-like strings
/// with X/Y ends and a Z run between), each term a gadget written out:
/// basis change, CNOT ladder, Rz(2 c dt), ladder⁻¹, basis change⁻¹.
/// Parameter: dt. Observable: the Z part of the Hamiltonian.
pub fn pauli_trotter(n: usize, steps: usize) -> (ParamCircuit, Observable) {
    let mut rng = StdRng::seed_from_u64(99 + n as u64);
    let mut terms: Vec<(Vec<(usize, char)>, f64)> = Vec::new();
    for _ in 0..3 * n {
        let a = rng.random_range(0..n - 1);
        let b = (a + rng.random_range(1..4)).min(n - 1);
        let mut s: Vec<(usize, char)> = Vec::new();
        let ends = ['X', 'Y', 'Z'];
        let ea = ends[rng.random_range(0..3)];
        let eb = if ea == 'Z' {
            'Z'
        } else {
            ends[rng.random_range(0..2)]
        };
        s.push((a, ea));
        for q in a + 1..b {
            s.push((q, 'Z'));
        }
        s.push((b, eb));
        terms.push((s, rng.random_range(-1.0..1.0)));
    }
    let mut pc = ParamCircuit::new(n, 1);
    let into = |pc: &mut ParamCircuit, q: usize, ch: char, inv: bool| match (ch, inv) {
        ('X', _) => {
            pc.gate(Gate::H(q));
        }
        ('Y', false) => {
            pc.gate(Gate::Sdg(q));
            pc.gate(Gate::H(q));
        }
        ('Y', true) => {
            pc.gate(Gate::H(q));
            pc.gate(Gate::S(q));
        }
        _ => {}
    };
    for _ in 0..steps {
        for (s, c) in &terms {
            for &(q, ch) in s {
                into(&mut pc, q, ch, false);
            }
            for w in s.windows(2) {
                pc.gate(Gate::Cnot(w[0].0, w[1].0));
            }
            pc.rz(s.last().unwrap().0, Angle::scaled(0, 2.0 * c));
            for w in s.windows(2).rev() {
                pc.gate(Gate::Cnot(w[0].0, w[1].0));
            }
            for &(q, ch) in s {
                into(&mut pc, q, ch, true);
            }
        }
    }
    let mut o = Observable::new();
    for (s, c) in &terms {
        if s.iter().all(|x| x.1 == 'Z') {
            let st: Vec<String> = s.iter().map(|(q, _)| format!("Z{q}")).collect();
            o.add(*c, &st.join(" ")).unwrap();
        }
    }
    if o.terms.is_empty() {
        o.add(1.0, "Z0").unwrap();
    }
    // start from a non-trivial product state
    let mut full = ParamCircuit::new(n, 1);
    for q in 0..n {
        full.ry(q, 0.3 + 0.1 * q as f64);
    }
    full.ops.extend(pc.ops);
    (full, o)
}

fn rewrite_bench(w: &str, n: usize, p: usize, binds: usize) {
    let (pc, o) = match w {
        "qaoa-cx" => qaoa_cx(n, p),
        "pauli" => pauli_trotter(n, p),
        _ => build(w, n, p),
    };
    let (rc, st) = phase_regions(&pc, &RewriteOptions::default());
    let mut rng = StdRng::seed_from_u64(12);
    let ps: Vec<Vec<f64>> = (0..binds)
        .map(|_| {
            (0..pc.num_params)
                .map(|_| rng.random_range(-1.5..1.5))
                .collect()
        })
        .collect();
    let opts = GraphOptions::default();
    let mut best = [f64::INFINITY; 5];
    let mut vals = [0.0; 3];
    for _ in 0..reps() {
        let (t, v) = time(|| ps.iter().map(|p| baseline(&pc, &o, p)).sum::<f64>());
        best[0] = best[0].min(t / binds as f64);
        vals[0] = v;
        let cc = CompiledCircuit::compile(&pc, Some(&o), &opts).unwrap();
        let (t, v) = time(|| {
            ps.iter()
                .map(|p| cc.bind(p).unwrap().expectation().unwrap())
                .sum::<f64>()
        });
        best[1] = best[1].min(t / binds as f64);
        vals[1] = v;
        let (tr, (rc2, _)) = time(|| phase_regions(&pc, &RewriteOptions::default()));
        best[3] = best[3].min(tr);
        let cr = CompiledCircuit::compile(&rc2, Some(&o), &opts).unwrap();
        let (t, v) = time(|| {
            ps.iter()
                .map(|p| cr.bind(p).unwrap().expectation().unwrap())
                .sum::<f64>()
        });
        best[2] = best[2].min(t / binds as f64);
        vals[2] = v;
    }
    assert!(
        (vals[0] - vals[1]).abs() < 1e-8 * binds as f64
            && (vals[0] - vals[2]).abs() < 1e-8 * binds as f64,
        "{vals:?}"
    );
    let ccr = CompiledCircuit::compile(&rc, Some(&o), &opts).unwrap();
    let cco = CompiledCircuit::compile(&pc, Some(&o), &opts).unwrap();
    println!(
        "{w} n={n} p={p} ops {}->{} regions {}/{} perm ops {}->{} gadgets {} | kops/stages {:?} -> {:?} | per-bind: baseline {:.3}ms compiled {:.3}ms rewritten {:.3}ms | rewrite vs compiled {:.2}x, vs baseline {:.2}x | rewrite pass {:.2}ms",
        pc.ops.len(),
        rc.ops.len(),
        st.rewritten,
        st.regions,
        st.perm_ops_before,
        st.perm_ops_after,
        st.gadgets_after,
        cco.stats().parts.iter().map(|p| (p.1, p.2)).collect::<Vec<_>>(),
        ccr.stats().parts.iter().map(|p| (p.1, p.2)).collect::<Vec<_>>(),
        best[0] * 1e3,
        best[1] * 1e3,
        best[2] * 1e3,
        best[1] / best[2],
        best[0] / best[2],
        best[3] * 1e3
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
        Some("rewrite") if args.len() >= 6 => {
            rewrite_bench(
                &args[2],
                args[3].parse().unwrap(),
                args[4].parse().unwrap(),
                args[5].parse().unwrap(),
            );
        }
        Some("rewrite") => {
            for &(w, n, p, b) in &[
                ("qaoa-cx", 12, 3, 200),
                ("qaoa-cx", 20, 3, 5),
                ("pauli", 12, 4, 50),
                ("pauli", 20, 2, 3),
                ("trotter", 12, 20, 200),
                ("hea", 12, 4, 200),
            ] {
                rewrite_bench(w, n, p, b);
            }
        }
        _ => eprintln!("usage: graph_bench bind|rewrite [workload n p binds]"),
    }
}
