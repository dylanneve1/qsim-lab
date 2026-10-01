//! Single-thread kernel micro-benchmark for the blocked executor: `K` fused
//! complex single-qubit gates (and optionally CNOTs) on an `n`-qubit register
//! that fits in one block, reported as cycles per amplitude per op at an
//! assumed clock (3.228 GHz, override with env GHZ).
//!
//! usage: l1_micro <n,...> [f32|f64] [K] [u1|cnot|mix] [reps] [tbits]
//! `tbits` = restrict gate targets to qubits < tbits (default n)

use num_complex::Complex64;
use qsim_lab::blocked::{BlockConfig, KOp};
use qsim_lab::statevector::{Real, StateVector};
use std::time::Instant;

/// Assumed clock: M1 Pro P-core max is 3.228 GHz; override with `GHZ=`.
/// (A dependent-add calibration was tried and discarded: LLVM folds the chain.)
fn clock_ghz() -> f64 {
    3.228
}

fn gates(n: usize, k: usize, kind: &str, tb: usize) -> Vec<KOp> {
    let mut v = Vec::new();
    for i in 0..k {
        let q = (i * 7 + i / tb) % tb; // varied targets
        let th = 0.1 + i as f64 * 0.37;
        let (c, s) = (th.cos(), th.sin());
        let m = [
            [Complex64::new(c, 0.1 * s), Complex64::new(-s, 0.0)],
            [Complex64::new(s, 0.0), Complex64::new(c, -0.1 * s)],
        ];
        let want_cnot = match kind {
            "cnot" => true,
            "mix" => i % 3 == 2,
            _ => false,
        };
        if want_cnot && n > 1 {
            let c = (q + 1) % tb.max(2);
            let c = if c == q { (q + 1) % n } else { c };
            v.push(KOp::U1 {
                q,
                m: [
                    [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
                    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                ],
                ctrl: 1 << c,
            });
        } else {
            v.push(KOp::U1 { q, m, ctrl: 0 });
        }
    }
    v
}

fn run<T: Real>(ns: &[usize], k: usize, kind: &str, reps: usize, tbits: Option<usize>, ghz: f64) {
    println!("| n | prec | set KiB | op | ops | min ms | cyc/amp/op |");
    println!("|---|---|---|---|---|---|---|");
    for &n in ns {
        let tb = tbits.unwrap_or(n).min(n);
        let ops = gates(n, k, kind, tb);
        let cfg = BlockConfig {
            fuse_1q: false,
            small_n: n,
            ..BlockConfig::default()
        };
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let mut s = StateVector::<T>::new(n);
            let t = Instant::now();
            s.apply_kops_blocked(&ops, &cfg);
            best = best.min(t.elapsed().as_secs_f64());
        }
        let amps = (1usize << n) as f64;
        let cyc = best * ghz * 1e9 / amps / k as f64;
        println!(
            "| {n} | {} | {} | {kind} | {k} | {:.3} | {cyc:.3} |",
            std::any::type_name::<T>(),
            (1usize << n) * 2 * std::mem::size_of::<T>() >> 10,
            best * 1e3
        );
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let ns: Vec<usize> = a[1].split(',').map(|x| x.parse().unwrap()).collect();
    let prec = a.get(2).map(|s| s.as_str()).unwrap_or("f32");
    let k: usize = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(200);
    let kind = a.get(4).map(|s| s.as_str()).unwrap_or("u1");
    let reps: usize = a.get(5).map(|s| s.parse().unwrap()).unwrap_or(7);
    let tb = a.get(6).map(|s| s.parse().unwrap());
    let ghz = std::env::var("GHZ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(clock_ghz);
    println!("clock {ghz:.3} GHz (assumed)");
    match prec {
        "f32" => run::<f32>(&ns, k, kind, reps, tb, ghz),
        _ => run::<f64>(&ns, k, kind, reps, tb, ghz),
    }
}
