//! Audit adapter for exp/hsf: HybridSchrodingerFeynman amplitudes / full
//! state vs the independent reference, over edge-biased circuits, random
//! and degenerate partitions (|A| = 0, 1, n−1, n; interleaved wires), every
//! option combination. Copy to `tests/` with `tests/audit_common/`.

mod audit_common;

use audit_common::*;
use qsim_lab::hsf::{HsfOptions, HybridSchrodingerFeynman, LeafMode, SchmidtMode};
use qsim_lab::Circuit;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn random_opts(rng: &mut StdRng) -> HsfOptions {
    let mut o = HsfOptions::default();
    if rng.random_bool(0.3) {
        return o;
    }
    o.eliminate_swaps = rng.random_bool(0.5);
    o.asap = rng.random_bool(0.5);
    o.schmidt = [SchmidtMode::Analytic, SchmidtMode::Svd, SchmidtMode::MatrixUnits][rng.random_range(0..3)];
    o.leaf = [LeafMode::Forward, LeafMode::Bra, LeafMode::Auto][rng.random_range(0..3)];
    o.prune_zero = rng.random_bool(0.5);
    o.threads = rng.random_range(0..3);
    o.gemm_batch = [1usize, 2, 7, 64][rng.random_range(0..4)];
    o
}

fn random_partition(rng: &mut StdRng, n: usize) -> Vec<bool> {
    match rng.random_range(0..6) {
        0 => vec![true; n],
        1 => vec![false; n],
        2 => (0..n).map(|q| q == 0).collect(),
        3 => (0..n).map(|q| q % 2 == 0).collect(),
        4 => (0..n).map(|q| q < n / 2).collect(),
        _ => (0..n).map(|_| rng.random_bool(0.5)).collect(),
    }
}

#[test]
fn hsf_matches_reference() {
    let (mut ok, mut rejected, mut worst) = (0usize, 0usize, 0.0f64);
    let mut reasons = std::collections::BTreeMap::new();
    for it in 0..15 * iters() {
        for &n in &SIZES {
            let seed = base_seed() ^ 0x45F ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let depth = rng.random_range(1..50);
            // keep the number of crossing gates moderate
            let c: Circuit = random_circuit(&mut rng, n, depth, false, false);
            let part = random_partition(&mut rng, n);
            let opts = random_opts(&mut rng);
            let h = match HybridSchrodingerFeynman::new(&c, &part, opts.clone()) {
                Ok(h) => h,
                Err(e) => {
                    rejected += 1;
                    *reasons.entry(format!("{e}").chars().take(60).collect::<String>()).or_insert(0) += 1;
                    continue;
                }
            };
            if h.num_paths() > 1 << 14 {
                continue;
            }
            let r = RefSv::run(&c);
            let mut xs: Vec<usize> = vec![0, (1 << n) - 1];
            for _ in 0..6 {
                xs.push(rng.random_range(0..1usize << n));
            }
            let amps = h.amplitudes(&xs).unwrap();
            for (&x, a) in xs.iter().zip(&amps) {
                let d = (r.a[x] - a).norm();
                worst = worst.max(d);
                assert!(d <= 1e-12, "hsf amplitude x={x} Δ={d:e} seed={seed} n={n} part={part:?} opts={opts:?}\n{:?}", c.ops);
            }
            let a1 = h.amplitude(xs[2]).unwrap();
            assert!((r.a[xs[2]] - a1).norm() <= 1e-12, "single amplitude seed={seed}");
            if n <= 11 {
                let sv = h.state_vector().unwrap();
                let d = max_amp_diff(&r.a, sv.iter().copied());
                worst = worst.max(d);
                assert!(d <= 1e-12, "hsf state_vector Δ={d:e} seed={seed} n={n} part={part:?} opts={opts:?}\n{:?}", c.ops);
            }
            ok += 1;
        }
    }
    eprintln!("hsf: {ok} checked, {rejected} rejected {reasons:?}, worst Δ = {worst:e}");
    assert!(ok > 50);
}

#[test]
fn hsf_auto_partition_matches_reference() {
    for it in 0..10 * iters() {
        for &n in &[2usize, 5, 8, 12, 14] {
            let seed = base_seed() ^ 0x45A ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let c = random_circuit(&mut rng, n, 30, false, false);
            let h = match HybridSchrodingerFeynman::auto(&c, HsfOptions::default()) {
                Ok(h) => h,
                Err(_) => continue,
            };
            if h.num_paths() > 1 << 14 {
                continue;
            }
            let r = RefSv::run(&c);
            let xs: Vec<usize> = (0..10).map(|_| rng.random_range(0..1usize << n)).collect();
            for (&x, a) in xs.iter().zip(h.amplitudes(&xs).unwrap()) {
                assert!((r.a[x] - a).norm() <= 1e-12, "auto hsf x={x} seed={seed} n={n}");
            }
        }
    }
}
