//! Surface-code threshold sweep driver (one CSV line per (d, p) point).
//!
//! ```text
//! cargo run --release --example surface_threshold -- point <d> <p> <shots> <seed> [dem|tableau] [circuit|phenom|weighted[:res]]
//! cargo run --release --example surface_threshold -- report <d>
//! ```
//!
//! Noise: `NoiseModel::circuit_level(p, p)` — depolarizing `p` after every 1- and
//! 2-qubit gate, readout flip `p`, reset flip `p`. Rounds = d. Shots are split
//! into chunks of 1000 run on `QSIM_THREADS` threads (default 2); chunk `i` uses
//! `StdRng::seed_from_u64(seed * 1_000_003 + i)`, so results are reproducible
//! for a fixed seed regardless of thread count.

use qsim_lab::noise::NoiseModel;
use qsim_lab::qec::{SamplingMethod, SurfaceCode};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::time::Instant;

fn wilson(k: usize, n: usize) -> (f64, f64) {
    let (k, n) = (k as f64, n as f64);
    let z = 1.959964f64;
    let p = k / n;
    let denom = 1.0 + z * z / n;
    let c = (p + z * z / (2.0 * n)) / denom;
    let h = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / denom;
    ((c - h).max(0.0), (c + h).min(1.0))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("report") => {
            let d: usize = args[1].parse().unwrap();
            let t = Instant::now();
            let sc = SurfaceCode::new(d, d);
            let noise = NoiseModel::circuit_level(0.001, 0.001);
            let mech = sc.detector_error_model(&noise);
            println!(
                "d={d} rounds={d} qubits={} detectors={} fault_locations={} merged_mechanisms={} build_s={:.3}",
                SurfaceCode::total_qubits(d),
                sc.faults.num_detectors,
                sc.faults.locations.len(),
                mech.len(),
                t.elapsed().as_secs_f64()
            );
            let mut by_weight = std::collections::BTreeMap::new();
            for m in &mech {
                *by_weight.entry(m.detectors.len()).or_insert(0usize) += 1;
            }
            println!("mechanisms by #detectors: {by_weight:?}");
            println!("graph: {:?}", sc.graph_report);
            println!(
                "graph-like distance: circuit graph {:?}, phenomenological graph {:?}",
                sc.decoder.graph.min_logical_weight(),
                SurfaceCode::with_phenomenological_decoder(d, d)
                    .decoder
                    .graph
                    .min_logical_weight()
            );
        }
        Some("point") => {
            let d: usize = args[1].parse().unwrap();
            let p: f64 = args[2].parse().unwrap();
            let shots: usize = args[3].parse().unwrap();
            let seed: u64 = args[4].parse().unwrap();
            let method = match args.get(5).map(String::as_str).unwrap_or("dem") {
                "dem" => SamplingMethod::DetectorErrorModel,
                "tableau" => SamplingMethod::Tableau,
                m => panic!("unknown method {m}"),
            };
            let decoder = args.get(6).map(String::as_str).unwrap_or("circuit");
            let noise = NoiseModel::circuit_level(p, p);
            let sc = match decoder {
                "circuit" => SurfaceCode::new(d, d),
                "phenom" => SurfaceCode::with_phenomenological_decoder(d, d),
                w if w.starts_with("weighted") => {
                    let res = w
                        .strip_prefix("weighted:")
                        .map(|r| r.parse().unwrap())
                        .unwrap_or(10);
                    SurfaceCode::new_weighted(d, d, &noise, res)
                }
                x => panic!("unknown decoder graph {x}"),
            };
            let threads: usize = std::env::var("QSIM_THREADS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(2);
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let chunk = 1000usize;
            let nchunks = shots.div_ceil(chunk);
            let t = Instant::now();
            let errors: usize = pool.install(|| {
                (0..nchunks)
                    .into_par_iter()
                    .map(|i| {
                        let n = chunk.min(shots - i * chunk);
                        let mut rng =
                            StdRng::seed_from_u64(seed.wrapping_mul(1_000_003) + i as u64);
                        sc.run_experiment(&noise, n, method, &mut rng).logical_errors
                    })
                    .sum()
            });
            let secs = t.elapsed().as_secs_f64();
            let (lo, hi) = wilson(errors, shots);
            println!(
                "{d},{d},{p},{shots},{errors},{:.6e},{lo:.6e},{hi:.6e},{secs:.2},{method:?},{decoder},{seed}",
                errors as f64 / shots as f64
            );
        }
        _ => eprintln!("usage: surface_threshold point <d> <p> <shots> <seed> [dem|tableau] [circuit|phenom|weighted[:res]] | report <d>"),
    }
}
