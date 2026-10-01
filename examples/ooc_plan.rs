//! Plan-only comparison of the out-of-core schedulers (no file I/O): number of
//! passes over the file for QFT / brickwork / random circuits.
//!
//! `cargo run --release --example ooc_plan`

use qsim_lab::algorithms;
use qsim_lab::circuit::Circuit;
use qsim_lab::ooc::schedule_ooc;
use qsim_lab::ooc_window::{schedule_window, WindowOptions};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

fn workload(name: &str, n: usize) -> Circuit {
    match name {
        "qft" => algorithms::qft(n),
        "brick" => {
            let mut rng = StdRng::seed_from_u64(42 + n as u64);
            algorithms::random_brickwork(n, 4, &mut rng)
        }
        "brick16" => {
            let mut rng = StdRng::seed_from_u64(42 + n as u64);
            algorithms::random_brickwork(n, 16, &mut rng)
        }
        _ => panic!("unknown workload {name}"),
    }
}

fn main() {
    println!("workload   n   c | swap-sched passes | window passes by k=2,3,4,5,6 (gate/perm) | plan ms");
    for wl in ["qft", "brick", "brick16"] {
        for n in [22usize, 24, 26, 28, 30, 32] {
            for c in [18usize, 20, 22] {
                if c >= n {
                    continue;
                }
                let circ = workload(wl, n);
                let old = schedule_ooc(&circ, n, c).map(|p| p.steps.len()).unwrap_or(0);
                let mut cells = String::new();
                let t0 = Instant::now();
                for k in [2usize, 3, 4, 5, 6] {
                    let opts = WindowOptions {
                        extra_bits: k,
                        ..WindowOptions::default()
                    };
                    match schedule_window(&circ, c, &opts) {
                        Ok(p) => cells += &format!(" {:>3}({}/{})", p.passes.len(), p.gate_passes, p.perm_passes),
                        Err(e) => cells += &format!(" err:{e}"),
                    }
                }
                println!(
                    "{wl:<8} {n:>3} {c:>3} | {old:>4} |{cells} | {:.0}",
                    t0.elapsed().as_secs_f64() * 1e3
                );
            }
        }
    }
}
