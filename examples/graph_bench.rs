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
        let (t, _) = time(|| {
            ps.iter()
                .map(|p| {
                    CompiledCircuit::compile(&pc, Some(&o), &opts)
                        .unwrap()
                        .bind(p)
                        .unwrap()
                        .expectation()
                        .unwrap()
                })
                .sum::<f64>()
        });
        best[7] = best[7].min(t / binds as f64);
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
        "{w} n={n} p={p} ops={} cone={} parts={:?} binds={binds} | compile {:.3}ms bind {:.1}us | per-bind: compiled {:.3}ms recompile {:.3}ms ({:.2}x) baseline {:.3}ms ({:.2}x) | sweep: compiled-par {:.3}ms baseline-par {:.3}ms ({:.2}x) vs serial baseline {:.2}x{}",
        pc.ops.len(),
        st.ops_after_cone,
        st.parts,
        best[0] * 1e3,
        best[1] * 1e6,
        best[2] * 1e3,
        best[7] * 1e3,
        best[7] / best[2],
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
    // plain = no rewrite; forced = always the rewritten circuit; auto = the
    // default (rewrite kept only if the plan is cheaper)
    let mut opts = GraphOptions::default();
    opts.rewrite = None;
    let auto = GraphOptions::default();
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
        let ca = CompiledCircuit::compile(&pc, Some(&o), &auto).unwrap();
        let (t, _) = time(|| {
            ps.iter()
                .map(|p| ca.bind(p).unwrap().expectation().unwrap())
                .sum::<f64>()
        });
        best[4] = best[4].min(t / binds as f64);
    }
    assert!(
        (vals[0] - vals[1]).abs() < 1e-8 * binds as f64
            && (vals[0] - vals[2]).abs() < 1e-8 * binds as f64,
        "{vals:?}"
    );
    let ccr = CompiledCircuit::compile(&rc, Some(&o), &opts).unwrap();
    let cco = CompiledCircuit::compile(&pc, Some(&o), &opts).unwrap();
    println!(
        "{w} n={n} p={p} ops {}->{} regions {}/{} perm ops {}->{} gadgets {} | kops/stages {:?} -> {:?} | per-bind: baseline {:.3}ms compiled {:.3}ms rewritten {:.3}ms | rewrite vs compiled {:.2}x, vs baseline {:.2}x | rewrite pass {:.2}ms | auto {:.3}ms (chose rewrite: {})",
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
        best[3] * 1e3,
        best[4] * 1e3,
        CompiledCircuit::compile(&pc, Some(&o), &auto)
            .unwrap()
            .stats()
            .rewritten
    );
}

fn dedup_family(name: &str) -> (ParamCircuit, usize) {
    use qsim_lab::{algorithms, shor_ripple, shor_window};
    let wrap = |c: &qsim_lab::Circuit| ParamCircuit::from_circuit(c).unwrap();
    match name {
        "trotter" => (trotter(12, 50).0, 0),
        "qaoa" => (qaoa(16, 5).0, 0),
        "hea" => (hea(12, 6).0, 0),
        "pauli" => (pauli_trotter(12, 4).0, 0),
        "adder" => {
            let lay = shor_ripple::RippleLayout::new(8);
            (wrap(&shor_ripple::controlled_ua(&lay, lay.ctrl, 7, 221)), 0)
        }
        "window" => {
            let lay = shor_window::WindowLayout::new(6, 3);
            (wrap(&shor_window::controlled_ua(&lay, 7, 55)), 0)
        }
        "qft" => (wrap(&algorithms::qft(16)), 0),
        "brickwork" => {
            let mut rng = StdRng::seed_from_u64(5);
            (wrap(&algorithms::random_brickwork(16, 20, &mut rng)), 0)
        }
        _ => panic!("family"),
    }
}

fn dedup_bench() {
    use qsim_lab::graph::dedup::{analyse, block_unitaries};
    println!("# load {:?}", std::fs::read_to_string("/proc/loadavg").ok());
    for fam in [
        "trotter",
        "qaoa",
        "hea",
        "pauli",
        "adder",
        "window",
        "qft",
        "brickwork",
    ] {
        let (pc, _) = dedup_family(fam);
        let mut row = format!("{fam:9} n={:3} ops={:6}", pc.num_qubits, pc.ops.len());
        for k in [2usize, 3, 4, 5] {
            let d = analyse(&pc, k);
            row += &format!(
                " | k={k} blocks {} classes {} cov {:.1}% shape {:.1}% reuse {:.1}% ({:.2}ms)",
                d.blocks.len(),
                d.num_classes,
                100.0 * d.coverage(),
                100.0 * d.shape_coverage(),
                100.0 * d.reuse(),
                d.secs * 1e3
            );
        }
        println!("{row}");
        // compile-time effect: block unitaries per instance vs per class (k = 4)
        let d = analyse(&pc, 4);
        let p: Vec<f64> = (0..pc.num_params).map(|i| 0.3 + 0.1 * i as f64).collect();
        let mut best = [f64::INFINITY; 2];
        let mut counts = (0, 0);
        for _ in 0..reps() {
            let (t, (_, c0)) = time(|| block_unitaries(&pc, &d, &p, false));
            best[0] = best[0].min(t);
            let (t, (_, c1)) = time(|| block_unitaries(&pc, &d, &p, true));
            best[1] = best[1].min(t);
            counts = (c0, c1);
        }
        println!(
            "  block unitaries (k=4): per instance {} in {:.2}ms, per class {} in {:.2}ms ({:.1}x)",
            counts.0,
            best[0] * 1e3,
            counts.1,
            best[1] * 1e3,
            best[0] / best[1]
        );
        // bind-time effect of recipe CSE (parameterised families)
        if pc.num_params > 0 {
            let mut off = GraphOptions::default();
            off.dedup_recipes = false;
            let on = GraphOptions::default();
            let a = CompiledCircuit::compile(&pc, None, &on).unwrap();
            let b = CompiledCircuit::compile(&pc, None, &off).unwrap();
            let mut bb = [f64::INFINITY; 2];
            let reps_b = 200;
            for _ in 0..reps() {
                let (t, _) = time(|| {
                    for _ in 0..reps_b {
                        std::hint::black_box(b.bind(&p).unwrap());
                    }
                });
                bb[0] = bb[0].min(t / reps_b as f64);
                let (t, _) = time(|| {
                    for _ in 0..reps_b {
                        std::hint::black_box(a.bind(&p).unwrap());
                    }
                });
                bb[1] = bb[1].min(t / reps_b as f64);
            }
            println!(
                "  bind: recipes evaluated per op {:.1}us, once per class {:.1}us ({:.2}x)",
                bb[0] * 1e6,
                bb[1] * 1e6,
                bb[0] / bb[1]
            );
        }
    }
}

/// Long shallow chain A (MPS-friendly) + small deep all-to-all core B
/// (state-vector friendly), joined by `cuts` CZ gates in the middle.
pub fn chain_core(
    na: usize,
    nb: usize,
    da: usize,
    db: usize,
    cuts: usize,
    seed: u64,
) -> qsim_lab::Circuit {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = na + nb;
    let mut c = qsim_lab::Circuit::new(n);
    let u = |rng: &mut StdRng, q: usize| {
        Gate::U(
            q,
            rng.random_range(0.0..3.1),
            rng.random_range(0.0..6.2),
            rng.random_range(0.0..6.2),
        )
    };
    let half = |c: &mut qsim_lab::Circuit, rng: &mut StdRng, da: usize, db: usize| {
        for l in 0..da {
            for q in 0..na {
                c.gate(u(rng, q));
            }
            for q in (l % 2..na - 1).step_by(2) {
                c.gate(Gate::Cz(q, q + 1));
            }
        }
        for _ in 0..db {
            for q in na..n {
                c.gate(u(rng, q));
            }
            let mut qs: Vec<usize> = (na..n).collect();
            for i in (1..qs.len()).rev() {
                qs.swap(i, rng.random_range(0..=i));
            }
            for p in qs.chunks(2) {
                if p.len() == 2 {
                    c.gate(Gate::Cz(p[0], p[1]));
                }
            }
        }
    };
    half(&mut c, &mut rng, da / 2, db / 2);
    for _ in 0..cuts {
        let a = rng.random_range(0..na);
        let b = rng.random_range(na..n);
        c.gate(Gate::Cz(a, b));
    }
    half(&mut c, &mut rng, da - da / 2, db - db / 2);
    c
}

fn partition_bench(na: usize, nb: usize, da: usize, db: usize, cuts: usize, deadline: f64) {
    use qsim_lab::graph::partition::{cut_amplitudes, plan_cut};
    use qsim_lab::planner::{self, Engine, PlannerConfig};
    let c = chain_core(na, nb, da, db, cuts, 3);
    let n = na + nb;
    let mut rng = StdRng::seed_from_u64(4);
    let m = 64;
    let xs: Vec<u128> = (0..m)
        .map(|_| rng.random::<u128>() & ((1u128 << n) - 1))
        .collect();
    let cfg = PlannerConfig::default();
    let (tp, plan) = time(|| plan_cut(&c, m, &cfg, 10, nb + 4));
    let plan = plan.unwrap();
    println!(
        "chain_core na={na} nb={nb} da={da} db={db} cuts={cuts} gates={} | plan {:.2}s: cut {} |B|={} A:{:?} B:{:?} predicted {:.3}s; single best {:?}",
        c.num_gates(),
        tp,
        plan.cut,
        plan.in_a.iter().filter(|&&a| !a).count(),
        plan.side_a,
        plan.side_b,
        plan.predicted_secs,
        plan.single
    );
    let mut best = f64::INFINITY;
    let mut amps = Vec::new();
    for _ in 0..reps() {
        let (t, a) = time(|| cut_amplitudes(&c, &plan, &xs, &cfg).unwrap());
        best = best.min(t);
        amps = a;
    }
    let norm: f64 = amps.iter().map(|a| a.norm_sqr()).sum::<f64>();
    println!(
        "  partitioned: {:.3}s for {m} amplitudes (sum |a|^2 = {norm:.3e})",
        best
    );
    // every single engine, with a deadline
    let only = std::env::var("SINGLE").unwrap_or_default();
    for e in [
        Engine::StateVector,
        Engine::Mps,
        Engine::Hsf,
        Engine::Sparse,
    ] {
        if !only.is_empty() && !only.split(',').any(|x| x == e.name()) {
            continue;
        }
        let (t, r) = time(|| planner::prepare(e, &c, &cfg, Some(deadline)));
        let what = match r {
            Ok(Some(mut p)) => match p.amplitudes(&xs) {
                Ok(a) => {
                    let err = a
                        .iter()
                        .zip(&amps)
                        .map(|(x, y)| (x - y).norm())
                        .fold(0.0, f64::max);
                    format!("ok, max |diff| vs partitioned {err:.2e}")
                }
                Err(e) => format!("read-out error {e}"),
            },
            Ok(None) => format!("aborted (deadline {deadline}s / budget)"),
            Err(e) => format!("error: {e}"),
        };
        println!("  single {e:?}: {t:.2}s {what}");
    }
}

/// How much of each family a k-qubit dense-fusion cost rule would fuse:
/// blocks of at most k qubits (the dedup blocking = greedy fusion
/// grouping) that hold at least 2^k dense single-qubit gates (the rule of
/// `BlockConfig::dense_min_ops`, from the measured kernel cost: a dense
/// k-qubit pass costs about as much as 2^k single-qubit passes).
fn fusable_bench() {
    use qsim_lab::graph::dedup::analyse;
    let dense1 = |op: &POp| match op {
        POp::Rx(..) | POp::Ry(..) | POp::U(..) => true,
        POp::Fixed(g) => {
            g.arity() == 1
                && g.diagonal_1q().is_none()
                && !matches!(g, Gate::X(_) | Gate::Y(_) | Gate::I(_))
        }
        _ => false,
    };
    for fam in [
        "trotter",
        "qaoa",
        "hea",
        "pauli",
        "adder",
        "window",
        "qft",
        "brickwork",
    ] {
        let (pc, _) = dedup_family(fam);
        let mut row = format!("{fam:9}");
        for k in [2usize, 3, 4, 5] {
            let d = analyse(&pc, k);
            let (mut blocks, mut ops) = (0, 0);
            for b in &d.blocks {
                let m = b.ops.iter().filter(|&&i| dense1(&pc.ops[i])).count();
                if b.qubits.len() == k && m >= 1 << k {
                    blocks += 1;
                    ops += b.ops.len();
                }
            }
            row += &format!(
                " | k={k}: {blocks} blocks, {:.1}% of ops",
                100.0 * ops as f64 / d.total_ops.max(1) as f64
            );
        }
        println!("{row}");
    }
}

/// Basis-state folding: ops removed and compiled run time with folding on
/// and off (full state, so nothing else removes the work).
fn fold_bench() {
    use qsim_lab::graph::fold::fold_basis;
    use qsim_lab::{shor_ripple, shor_window};
    let wrap = |c: &qsim_lab::Circuit| ParamCircuit::from_circuit(c).unwrap();
    let cases: Vec<(&str, ParamCircuit)> = vec![
        ("adder n=4 (16q)", {
            let lay = shor_ripple::RippleLayout::new(4);
            wrap(&shor_ripple::controlled_ua(&lay, lay.ctrl, 7, 15))
        }),
        ("window n=4 w=2", {
            let lay = shor_window::WindowLayout::new(4, 2);
            wrap(&shor_window::controlled_ua(&lay, 7, 15))
        }),
        ("trotter n=16 s=10", trotter(16, 10).0),
        ("qaoa n=16 p=3", qaoa(16, 3).0),
    ];
    for (name, pc) in cases {
        let (f, st) = fold_basis(&pc);
        let p: Vec<f64> = (0..pc.num_params).map(|i| 0.4 + 0.1 * i as f64).collect();
        let mut off = GraphOptions::default();
        off.fold_basis = false;
        let on = GraphOptions::default();
        let mut best = [f64::INFINITY; 2];
        for _ in 0..reps() {
            let a = CompiledCircuit::compile(&pc, None, &off).unwrap();
            let (t, _) = time(|| a.bind(&p).unwrap().statevector().unwrap());
            best[0] = best[0].min(t);
            let b = CompiledCircuit::compile(&pc, None, &on).unwrap();
            let (t, _) = time(|| b.bind(&p).unwrap().statevector().unwrap());
            best[1] = best[1].min(t);
        }
        println!(
            "{name:20} n={} ops {} -> {} (removed {}, simplified {}) | run: fold off {:.3}ms on {:.3}ms ({:.2}x)",
            pc.num_qubits,
            pc.ops.len(),
            f.ops.len(),
            st.removed,
            st.simplified,
            best[0] * 1e3,
            best[1] * 1e3,
            best[0] / best[1]
        );
    }
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
        Some("dedup") => dedup_bench(),
        Some("fusable") => fusable_bench(),
        Some("fold") => fold_bench(),
        Some("partition") => {
            let a: Vec<usize> = args[2..7].iter().map(|x| x.parse().unwrap()).collect();
            let dl: f64 = args.get(7).map_or(30.0, |x| x.parse().unwrap());
            partition_bench(a[0], a[1], a[2], a[3], a[4], dl);
        }
        _ => eprintln!("usage: graph_bench bind|rewrite|dedup [workload n p binds]"),
    }
}
