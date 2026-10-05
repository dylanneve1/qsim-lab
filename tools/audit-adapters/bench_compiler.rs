//! Audit bench for exp/compiler: INTERLEAVED "always state vector"
//! (`compile_sampling(c, PlanOptions::none())` + 1000 f32 shots) vs compiled
//! (`compile_sampling(c, default)` + 1000 f32 shots, compile time included),
//! `reps` alternating pairs, min of each. Same workloads/seeds as
//! examples/compile_bench.rs.
//!
//! usage: bench_compiler <bv23|ghz24|cliffordt22|cliffordt20> [reps]

use qsim_lab::algorithms;
use qsim_lab::circuit::Circuit;
use qsim_lab::compile::plan::{compile_sampling, PlanOptions};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let reps: usize = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(5);
    let mut rng = StdRng::seed_from_u64(1);
    let c1 = Circuit::random_clifford_t(20, 20, 0.02, &mut rng);
    let c2 = Circuit::random_clifford_t(22, 30, 0.01, &mut rng);
    let c = match a[1].as_str() {
        "bv23" => algorithms::bernstein_vazirani(23, 0x2D_5A5Bu64 & ((1 << 23) - 1)),
        "ghz24" => {
            let mut c = algorithms::ghz(24);
            c.measure_all();
            c
        }
        "cliffordt20" => {
            let mut c = c1;
            c.measure_all();
            c
        }
        "cliffordt22" => {
            let mut c = c2;
            c.measure_all();
            c
        }
        _ => panic!("unknown workload"),
    };
    let mut srng = StdRng::seed_from_u64(7);
    let (mut tb, mut tc) = (Vec::new(), Vec::new());
    for _ in 0..reps {
        let t = Instant::now();
        let p0 = compile_sampling(&c, PlanOptions::none());
        std::hint::black_box(p0.sample::<f32, _>(1000, &mut srng).unwrap());
        tb.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        let p = compile_sampling(&c, PlanOptions::default());
        std::hint::black_box(p.sample::<f32, _>(1000, &mut srng).unwrap());
        tc.push(t.elapsed().as_secs_f64());
    }
    let mb = tb.iter().cloned().fold(f64::INFINITY, f64::min);
    let mc = tc.iter().cloned().fold(f64::INFINITY, f64::min);
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    println!(
        "{} n={} gates={}: always-SV min {mb:.4}s  compiled min {mc:.5}s  speedup {:.1}x  load_end {}",
        a[1],
        c.num_qubits,
        c.num_gates(),
        mb / mc,
        load.split_whitespace().next().unwrap_or("?")
    );
    println!("  base     {:?}", tb.iter().map(|t| format!("{t:.4}")).collect::<Vec<_>>());
    println!("  compiled {:?}", tc.iter().map(|t| format!("{t:.5}")).collect::<Vec<_>>());
}
