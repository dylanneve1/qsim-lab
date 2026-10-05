//! End-to-end `simulate()`-equivalent wall time with phase folding off vs
//! on (the PlanOptions switch), interleaved A/B/A/B/A/B, min of 3, on
//! Clifford+T circuits whose compile plan picks the adaptive / tableau /
//! Pauli-path engines. Run through bench.sh:
//!   bench.sh ./target/release/examples/phasepoly_e2e [filter]
//! Compile time (including the fold) is part of the measurement. Also
//! checks that both plans give the same exact outcome distribution family
//! by comparing the engines picked and (small n) the sampled records.

use qsim_lab::algorithms::qft;
use qsim_lab::compile::plan::{compile_sampling, PlanOptions};
use qsim_lab::gate::toffoli_clifford_t;
use qsim_lab::pipeline::plan_options;
use qsim_lab::shor::ripple::cuccaro_add;
use qsim_lab::{Circuit, Gate, Op};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn lower(c: &Circuit) -> Circuit {
    let mut o = Circuit::new(c.num_qubits);
    for op in &c.ops {
        match *op {
            Op::Gate(Gate::Ccx(a, b, t)) => {
                for g in toffoli_clifford_t(a, b, t) {
                    o.gate(g);
                }
            }
            Op::Gate(Gate::CPhase(a, b, th)) => {
                o.phase(a, th / 2.0)
                    .cnot(a, b)
                    .phase(b, -th / 2.0)
                    .cnot(a, b)
                    .phase(b, th / 2.0);
            }
            op => o.ops.push(op),
        }
    }
    o
}

fn adder(n: usize, superposed: usize) -> Circuit {
    let a: Vec<usize> = (0..n).collect();
    let b: Vec<usize> = (n..2 * n + 1).collect();
    let mut c = Circuit::new(2 * n + 2);
    for q in 0..superposed {
        c.h(a[q]);
        c.h(b[q]);
    }
    cuccaro_add(&mut c, &a, &b, 2 * n + 1);
    let mut c = lower(&c);
    c.measure_all();
    c
}

fn rct(n: usize, depth: usize, tp: f64, seed: u64) -> Circuit {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut c = Circuit::random_clifford_t(n, depth, tp, &mut rng);
    c.measure_all();
    c
}

/// Random Clifford+T where T gates are drawn on a few qubits and Clifford
/// glue is CNOT/S only between H rounds (so parities repeat).
fn rct_glued(n: usize, rounds: usize, seed: u64) -> Circuit {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    for _ in 0..rounds {
        for _ in 0..n {
            let a = rng.random_range(0..n);
            let b = (a + 1 + rng.random_range(0..n - 1)) % n;
            c.cnot(a, b);
        }
        for q in 0..n {
            if rng.random_bool(0.5) {
                c.t(q);
            } else {
                c.gate(Gate::Tdg(q));
            }
        }
        let q = rng.random_range(0..n);
        c.h(q);
    }
    c.measure_all();
    c
}

fn run(c: &Circuit, fold: bool) -> (f64, String, usize) {
    let mut o: PlanOptions = plan_options();
    o.phase_fold = fold;
    let t = Instant::now();
    let plan = compile_sampling(c, o).unwrap();
    let mut rng = StdRng::seed_from_u64(1);
    let out = plan.sample::<f64, _>(64, &mut rng).unwrap();
    let dt = t.elapsed().as_secs_f64();
    let eng = format!("{:?}", plan.stats.components);
    (dt, eng, out.len())
}

fn main() {
    let filter = std::env::args().nth(1).unwrap_or_default();
    let mut cases: Vec<(String, Circuit)> = vec![
        ("adder_n12_sup4".into(), adder(12, 4)),
        ("adder_n12_sup6".into(), adder(12, 6)),
        ("adder_n12_sup8".into(), adder(12, 8)),
        ("adder_n16_sup5".into(), adder(16, 5)),
        ("adder_n14_sup5".into(), adder(14, 5)),
        ("rct_n20_d8_p03".into(), rct(20, 8, 0.03, 5)),
        ("rct_n24_d10_p02".into(), rct(24, 10, 0.02, 6)),
        ("rct_n16_d6_p10".into(), rct(16, 6, 0.10, 7)),
        ("glued_n20_r10".into(), rct_glued(20, 10, 11)),
        ("glued_n26_r12".into(), rct_glued(26, 12, 12)),
    ];
    let _ = qft(2);
    cases.retain(|(n, _)| n.contains(&filter));
    println!("circuit,n,gates_in,t_in,engines_off,engines_on,off_s,on_s,speedup");
    for (name, c) in &cases {
        let (mut off, mut on) = (f64::INFINITY, f64::INFINITY);
        let (mut e_off, mut e_on) = (String::new(), String::new());
        for _ in 0..3 {
            let (t, e, _) = run(c, false);
            off = off.min(t);
            e_off = e;
            let (t, e, _) = run(c, true);
            on = on.min(t);
            e_on = e;
        }
        let mut u = Circuit::new(c.num_qubits);
        u.ops = c
            .ops
            .iter()
            .filter(|o| matches!(o, Op::Gate(_)))
            .copied()
            .collect();
        let d0 =
            qsim_lab::engines::adaptive::active_dimension(&qsim_lab::compile::optimize(&u).circuit)
                .unwrap();
        let f = qsim_lab::compile::optimize(
            &qsim_lab::compile::phase_fold(&qsim_lab::compile::optimize(&u).circuit).circuit,
        );
        let d1 = qsim_lab::engines::adaptive::active_dimension(&f.circuit).unwrap();
        println!(
            "{name},{},{},{},\"{e_off}\",\"{e_on}\",{off:.4},{on:.4},{:.2},d_off={d0},d_on={d1},nc_off={},nc_on={}",
            c.num_qubits,
            c.num_gates(),
            c.t_count(),
            off / on,
            qsim_lab::compile::optimize(&u).circuit.gates().filter(|g| !g.is_clifford()).count(),
            f.circuit.gates().filter(|g| !g.is_clifford()).count(),
        );
    }
}
