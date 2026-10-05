//! Audit bench for exp/sv: INTERLEAVED base (`apply_circuit`) vs blocked
//! (`apply_circuit_blocked`, default config) runs, min-of-`reps` each, same
//! binary. Prints the per-rep times, both minima and the ratio.
//!
//! usage: bench_sv_blocked <qft|brick|ghz> <n> <f32|f64> [reps]

use qsim_lab::algorithms;
use qsim_lab::blocked::BlockConfig;
use qsim_lab::circuit::Circuit;
use qsim_lab::statevector::{Real, StateVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

fn workload(name: &str, n: usize) -> Circuit {
    match name {
        "qft" => algorithms::qft(n),
        "ghz" => algorithms::ghz(n),
        "brick" => algorithms::random_brickwork(n, 20, &mut StdRng::seed_from_u64(42)),
        _ => panic!("unknown workload"),
    }
}

fn go<T: Real>(c: &Circuit, reps: usize) {
    let cfg = BlockConfig::default();
    let (mut tb, mut tn) = (Vec::new(), Vec::new());
    let mut maxd = 0.0f64;
    for _ in 0..reps {
        let mut a = StateVector::<T>::new(c.num_qubits);
        let t = Instant::now();
        a.apply_circuit(c).unwrap();
        tb.push(t.elapsed().as_secs_f64());
        let mut b = StateVector::<T>::new(c.num_qubits);
        let t = Instant::now();
        b.apply_circuit_blocked(c, &cfg).unwrap();
        tn.push(t.elapsed().as_secs_f64());
        for i in (0..1usize << c.num_qubits).step_by(97) {
            maxd = maxd.max((a.amplitude(i) - b.amplitude(i)).norm());
        }
    }
    let mb = tb.iter().cloned().fold(f64::INFINITY, f64::min);
    let mn = tn.iter().cloned().fold(f64::INFINITY, f64::min);
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    println!(
        "base min {mb:.4}s  blocked min {mn:.4}s  speedup {:.2}x  sampled max|Δamp| {maxd:.2e}  load_end {}",
        mb / mn,
        load.split_whitespace().next().unwrap_or("?")
    );
    println!("  base    {:?}", tb.iter().map(|t| format!("{t:.3}")).collect::<Vec<_>>());
    println!("  blocked {:?}", tn.iter().map(|t| format!("{t:.3}")).collect::<Vec<_>>());
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let c = workload(&a[1], a[2].parse().unwrap());
    let reps = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(5);
    println!("{} n={} {} gates={}", a[1], a[2], a[3], c.num_gates());
    if a[3] == "f32" {
        go::<f32>(&c, reps)
    } else {
        go::<f64>(&c, reps)
    }
}
