//! Deep-circuit check of truncated SPD against the exact state vector on a
//! heavy-hex patch (BFS ball of `k` qubits around Eagle qubit 62): the
//! 127-qubit runs have no exact reference beyond ~10 steps, this patch does.
//!
//! ```text
//! cargo run --release --example spoof_patch -- [k=24] [theta=0.6] [steps=20] [deltas=1e-3,1e-4,...] [branch_factor=1] [only_steps]
//! ```
//! Prints one JSON line per (steps, δ): exact <Z_root>, SPD value, norm, time.

use qsim_lab::spd::{simulate, KickedIsing, Lattice, PauliObs, SpdOptions};
use qsim_lab::StateVector;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let k: usize = a.first().map(|s| s.parse().unwrap()).unwrap_or(24);
    let theta: f64 = a.get(1).map(|s| s.parse().unwrap()).unwrap_or(0.6);
    let max_steps: usize = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(20);
    let deltas: Vec<f64> = a
        .get(3)
        .map(|s| s.split(',').map(|t| t.parse().unwrap()).collect())
        .unwrap_or(vec![1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]);
    let branch_factor: f64 = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(1.0);
    let only_steps: Option<usize> = a.get(5).map(|s| s.parse().unwrap());
    let patch = Lattice::eagle127().bfs_patch(62, k);
    eprintln!("patch: {} qubits, {} edges", patch.n, patch.edges.len());
    // exact: evolve step by step, record <Z_0> (qubit 0 of the patch = Eagle 62)
    let one = KickedIsing::new(patch.clone(), 1, theta).to_circuit();
    let mut sv: StateVector<f64> = StateVector::new(patch.n);
    let mut exact = vec![1.0];
    for _ in 0..max_steps {
        sv.apply_circuit(&one).unwrap();
        exact.push(sv.expectation_z(0));
    }
    for steps in [5, 10, 15, 20, 25, 30]
        .into_iter()
        .filter(|&s| s <= max_steps && only_steps.is_none_or(|o| o == s))
    {
        let m = KickedIsing::new(patch.clone(), steps, theta);
        for &delta in &deltas {
            let t = Instant::now();
            let r = simulate(
                &m,
                &PauliObs::z(0),
                &SpdOptions {
                    delta,
                    branch_factor,
                    max_terms: 20_000_000,
                    ..SpdOptions::default()
                },
            );
            println!(
                "{{\"k\":{k},\"theta\":{theta},\"steps\":{steps},\"delta\":{delta:e},\"branch_factor\":{branch_factor},\"edges\":{},\"exact\":{:.10},\"spd\":{:.10},\"err\":{:.3e},\"norm2\":{:.6},\"peak_terms\":{},\"aborted\":{},\"seconds\":{:.3}}}",
                patch.edges.len(),
                exact[steps],
                r.value,
                r.value - exact[steps],
                r.norm2,
                r.peak_terms,
                r.aborted,
                t.elapsed().as_secs_f64()
            );
            if r.aborted {
                break;
            }
        }
    }
}
