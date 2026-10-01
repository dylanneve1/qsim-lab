//! DAG IR measurements: construction cost per op, and gate counts after the
//! adjacent-only peephole (`Circuit::optimize`, PR 1), the DAG peephole
//! restricted to direct wire neighbours, and the commutation-aware DAG
//! peephole, on random Clifford+T, QFT and Grover circuits.
//!
//! Usage: cargo run --release --example dag_bench [counts|timing|all]
//! Timings must be taken through the swarm's bench.sh lock.

use qsim_lab::algorithms::qft;
use qsim_lab::dag::{self, Dag, PeepholeOptions};
use qsim_lab::{Circuit, Gate, StateVectorF64};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

/// Toffoli in the standard 7-T Clifford+T decomposition.
fn ccx_clifford_t(c: &mut Circuit, a: usize, b: usize, t: usize) {
    c.h(t)
        .cnot(b, t)
        .tdg(t)
        .cnot(a, t)
        .t(t)
        .cnot(b, t)
        .tdg(t)
        .cnot(a, t);
    c.t(b).t(t).h(t).cnot(a, b).t(a).tdg(b).cnot(a, b);
}

/// Multi-controlled X with a V-chain of Toffolis (needs `controls.len()-2`
/// clean ancillas, returned to |0>).
fn mcx(c: &mut Circuit, controls: &[usize], t: usize, anc: &[usize], decompose: bool) {
    let ccx = |c: &mut Circuit, a: usize, b: usize, t: usize| {
        if decompose {
            ccx_clifford_t(c, a, b, t);
        } else {
            c.ccx(a, b, t);
        }
    };
    match controls.len() {
        1 => {
            c.cnot(controls[0], t);
        }
        2 => ccx(c, controls[0], controls[1], t),
        m => {
            let mut chain = vec![(controls[0], controls[1], anc[0])];
            for i in 2..m - 1 {
                chain.push((controls[i], anc[i - 2], anc[i - 1]));
            }
            for &(a, b, x) in &chain {
                ccx(c, a, b, x);
            }
            ccx(c, controls[m - 1], anc[m - 3], t);
            for &(a, b, x) in chain.iter().rev() {
                ccx(c, a, b, x);
            }
        }
    }
}

/// Grover search circuit on `n` data qubits (plus `n-3` ancillas for the
/// multi-controlled Z when `n > 3`), `iters` iterations, marked item
/// `marked`, final measurement of the data qubits.
pub fn grover_circuit(n: usize, marked: usize, iters: usize, decompose: bool) -> Circuit {
    let anc: Vec<usize> = (n..n + n.saturating_sub(3)).collect();
    let mut c = Circuit::new(n + anc.len());
    let data: Vec<usize> = (0..n).collect();
    let mcz = |c: &mut Circuit| {
        let t = data[n - 1];
        c.h(t);
        mcx(c, &data[..n - 1], t, &anc, decompose);
        c.h(t);
    };
    for &q in &data {
        c.h(q);
    }
    for _ in 0..iters {
        for &q in &data {
            if (marked >> q) & 1 == 0 {
                c.x(q);
            }
        }
        mcz(&mut c);
        for &q in &data {
            if (marked >> q) & 1 == 0 {
                c.x(q);
            }
        }
        for &q in &data {
            c.h(q);
            c.x(q);
        }
        mcz(&mut c);
        for &q in &data {
            c.x(q);
            c.h(q);
        }
    }
    for &q in &data {
        c.measure(q);
    }
    c
}

/// QFT with every controlled phase lowered to CNOTs and single-qubit
/// phases (what a transpiler targeting CNOT + 1q gates emits).
fn qft_lowered(n: usize) -> Circuit {
    let src = qft(n);
    let mut c = Circuit::new(n);
    for g in src.gates() {
        match *g {
            Gate::CPhase(a, b, th) => {
                c.phase(a, th / 2.0)
                    .cnot(a, b)
                    .phase(b, -th / 2.0)
                    .cnot(a, b);
                c.phase(b, th / 2.0);
            }
            g => {
                c.gate(g);
            }
        }
    }
    c
}

fn counts(label: &str, c: &Circuit) {
    let base = c.optimize();
    let adj = PeepholeOptions {
        commute: false,
        ..Default::default()
    };
    let (o_adj, _) = dag::optimize_with(c, adj).unwrap();
    let (o, st) = dag::optimize_with(c, PeepholeOptions::default()).unwrap();
    let g0 = c.num_gates();
    let pct = |x: usize| 100.0 * (g0 as f64 - x as f64) / g0.max(1) as f64;
    println!(
        "{label:<34} gates {g0:>7} | optimize() {:>7} ({:>5.1}%) | dag-adjacent {:>7} ({:>5.1}%) | dag-commute {:>7} ({:>5.1}%) | T {:>5} -> {:>5} / {:>5} | passes {} searches {} steps {}",
        base.num_gates(),
        pct(base.num_gates()),
        o_adj.circuit.num_gates(),
        pct(o_adj.circuit.num_gates()),
        o.circuit.num_gates(),
        pct(o.circuit.num_gates()),
        c.t_count(),
        base.t_count(),
        o.circuit.t_count(),
        st.passes,
        st.searches,
        st.walk_steps,
    );
    // Spot-check exactness on small unitary bodies.
    let body = |c: &Circuit| Circuit {
        num_qubits: c.num_qubits,
        ops: c
            .ops
            .iter()
            .filter(|op| matches!(op, qsim_lab::Op::Gate(_)))
            .copied()
            .collect(),
    };
    if c.num_qubits <= 14 {
        let mut a = StateVectorF64::new(c.num_qubits);
        a.apply_circuit(&body(c)).unwrap();
        let mut b = StateVectorF64::new(c.num_qubits);
        b.apply_circuit(&body(&o.circuit)).unwrap();
        let ph = num_complex::Complex64::from_polar(1.0, o.global_phase);
        let d = a
            .amplitudes()
            .iter()
            .zip(b.amplitudes())
            .map(|(x, y)| (x - ph * y).norm())
            .fold(0.0, f64::max);
        assert!(d < 1e-10, "{label}: max |Δamp| = {d}");
    }
}

fn run_counts() {
    println!("# gate counts (lower is better); dag passes run to a fixpoint");
    for &(n, depth, tp) in &[
        (8, 50, 0.2),
        (16, 100, 0.2),
        (24, 200, 0.3),
        (50, 400, 0.25),
    ] {
        for seed in 0..3u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let c = Circuit::random_clifford_t(n, depth, tp, &mut rng);
            counts(&format!("clifford+t n={n} d={depth} p={tp} s={seed}"), &c);
        }
    }
    for n in [8, 16, 32] {
        counts(&format!("qft n={n}"), &qft(n));
        counts(&format!("qft-lowered n={n}"), &qft_lowered(n));
        let mut rt = qft(n);
        rt.append(&qft(n).inverse());
        counts(&format!("qft·qft† n={n}"), &rt);
    }
    for n in [4, 5, 6, 8] {
        let iters = ((std::f64::consts::PI / 4.0) * ((1u64 << n) as f64).sqrt()).floor() as usize;
        let it = iters.min(4);
        counts(
            &format!("grover-ccx n={n} it={it}"),
            &grover_circuit(n, 0b1011 % (1 << n), it, false),
        );
        counts(
            &format!("grover-clifford+t n={n} it={it}"),
            &grover_circuit(n, 0b1011 % (1 << n), it, true),
        );
    }
}

fn time_min<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

fn run_timing() {
    println!("# timings, min of 5 (shared VM; run under bench.sh)");
    let mut rng = StdRng::seed_from_u64(42);
    for &(n, depth) in &[(32, 500), (64, 2000), (100, 4000)] {
        let c = Circuit::random_clifford_t(n, depth, 0.25, &mut rng);
        let ops = c.ops.len() as f64;
        let t_build = time_min(5, || {
            std::hint::black_box(Dag::from_circuit(&c).unwrap());
        });
        let d = Dag::from_circuit(&c).unwrap();
        let t_out = time_min(5, || {
            std::hint::black_box(d.to_circuit());
        });
        let t_base = time_min(5, || {
            std::hint::black_box(c.optimize());
        });
        let t_peep = time_min(5, || {
            std::hint::black_box(dag::optimize(&c).unwrap());
        });
        let adj = PeepholeOptions {
            commute: false,
            ..Default::default()
        };
        let t_adj = time_min(5, || {
            std::hint::black_box(dag::optimize_with(&c, adj).unwrap());
        });
        let mut d2 = Dag::from_circuit(&c).unwrap();
        dag::peephole(&mut d2, PeepholeOptions::default());
        let t_topo = time_min(5, || {
            std::hint::black_box(d2.topo_order());
        });
        let mut lb = String::new();
        for look_back in [0, 1, 4, 16] {
            let o = PeepholeOptions {
                look_back,
                ..Default::default()
            };
            let mut gates = 0;
            let t = time_min(5, || {
                gates = dag::optimize_with(&c, o).unwrap().0.circuit.num_gates();
            });
            lb += &format!(" lb{look_back}: {:.0} ns/op ({gates} gates)", 1e9 * t / ops);
        }
        println!("  look-back sweep:{lb}");
        let t_cone = time_min(5, || {
            std::hint::black_box(dag::light_cone(&c, &[0]).unwrap());
        });
        println!(
            "n={n:>3} ops={ops:>8} | from_circuit {:>6.1} ns/op | to_circuit {:>6.1} ns/op | optimize() {:>6.1} ns/op | dag peephole {:>7.1} ns/op | dag light cone {:>6.1} ns/op | dag adjacent-only {:>6.1} ns/op | topo after rewrite {:>6.1} ns/op",
            1e9 * t_build / ops,
            1e9 * t_out / ops,
            1e9 * t_base / ops,
            1e9 * t_peep / ops,
            1e9 * t_cone / ops,
            1e9 * t_adj / ops,
            1e9 * t_topo / ops,
        );
    }
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    if mode == "counts" || mode == "all" {
        run_counts();
    }
    if mode == "timing" || mode == "all" {
        run_timing();
    }
}
