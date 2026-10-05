//! The cache-blocked executor must reproduce the gate-by-gate state vector.

#[path = "../common/mod.rs"]
mod common;

use common::random_universal;
use proptest::prelude::*;
use qsim_lab::algorithms;
use qsim_lab::circuit::Circuit;
use qsim_lab::engines::blocked::BlockConfig;
use qsim_lab::engines::statevector::{Real, StateVector};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn max_diff<T: Real>(a: &StateVector<T>, b: &StateVector<T>) -> f64 {
    a.amplitudes()
        .iter()
        .zip(b.amplitudes())
        .map(|(x, y)| {
            let d = *x - *y;
            (d.re.to_f64().powi(2) + d.im.to_f64().powi(2)).sqrt()
        })
        .fold(0.0, f64::max)
}

/// A random (non-basis) starting state so every amplitude matters.
fn start<T: Real>(n: usize, seed: u64) -> StateVector<T> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut s = StateVector::<T>::new(n);
    for q in 0..n {
        s.apply_gate(&qsim_lab::Gate::Ry(q, rng.random::<f64>() * 3.0))
            .unwrap();
        s.apply_gate(&qsim_lab::Gate::Rz(q, rng.random::<f64>() * 3.0))
            .unwrap();
    }
    s
}

fn check<T: Real>(c: &Circuit, cfg: &BlockConfig, seed: u64, tol: f64) {
    let mut a = start::<T>(c.num_qubits, seed);
    let mut b = a.clone();
    a.apply_circuit(c).unwrap();
    b.apply_circuit_blocked(c, cfg).unwrap();
    let d = max_diff(&a, &b);
    assert!(d <= tol, "max |Δamp| = {d:e} (cfg {cfg:?})");
}

/// Configurations that force every code path at small n: tiny blocks,
/// few or many slots, with and without 1q fusion, L1 tiling and dense
/// k-qubit fusion.
fn configs() -> Vec<BlockConfig> {
    let mut v = Vec::new();
    for (kib, slots, fuse, split, sched, tile, dense) in [
        (1, 2, true, true, true, 0, 0),
        (1, 4, false, false, true, 128, 2),
        (2, 0, true, false, false, 256, 3),
        (4, 3, true, true, false, 512, 0),
        (256, 6, true, true, true, 4096, 2),
        (8, 3, true, false, true, 0, 3),
    ] {
        v.push(BlockConfig {
            block_bytes: kib << 10,
            slots,
            fuse_1q: fuse,
            small_n: 4,
            split_phases: split,
            schedule_diag: sched,
            simd: true,
            l1_tile_bytes: tile,
            dense_fusion: dense,
            dense_min_ops: 1,
        });
    }
    v
}

#[test]
fn algorithms_match_gate_by_gate() {
    for n in [3, 5, 9, 12, 14] {
        let mut rng = StdRng::seed_from_u64(n as u64);
        for c in [
            algorithms::ghz(n),
            algorithms::qft(n),
            algorithms::random_brickwork(n, 6, &mut rng),
        ] {
            for cfg in configs() {
                check::<f64>(&c, &cfg, 1, 1e-12);
                check::<f32>(&c, &cfg, 1, 1e-5);
            }
        }
    }
}

#[test]
fn default_config_at_18_qubits() {
    let mut rng = StdRng::seed_from_u64(9);
    let n = 18;
    let cfg = BlockConfig::default();
    check::<f64>(&algorithms::qft(n), &cfg, 2, 1e-12);
    check::<f32>(&algorithms::qft(n), &cfg, 2, 1e-5);
    let c = random_universal(n, 400, &mut rng);
    check::<f64>(&c, &cfg, 3, 1e-12);
    check::<f32>(&c, &cfg, 3, 1e-5);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn random_circuits_match(n in 3usize..11, len in 1usize..120, seed in 0u64..1000, ci in 0usize..5) {
        let mut rng = StdRng::seed_from_u64(seed);
        let c = random_universal(n, len, &mut rng);
        let cfg = &configs()[ci];
        check::<f64>(&c, cfg, seed, 1e-12);
        check::<f32>(&c, cfg, seed, 1e-5);
    }
}

/// Regression for the audit's minimal repro (exp/audit,
/// tools/audit-adapters/sv_blocked_repro_ea41235.rs): with `split_phases` on, this
/// circuit came out wrong by 0.26 in amplitude. The default config must
/// match the gate-by-gate path.
#[test]
fn split_phases_regression() {
    use qsim_lab::Gate::*;
    let gs = [
        Y(0),
        Cnot(0, 4),
        Rx(0, std::f64::consts::FRAC_PI_4),
        Z(0),
        H(0),
        X(0),
        S(0),
        Z(0),
        T(0),
        H(0),
    ];
    let mut c = Circuit::new(5);
    for g in gs {
        c.ops.push(qsim_lab::circuit::Op::Gate(g));
    }
    let cfg = BlockConfig::default();
    assert!(!cfg.split_phases, "split_phases must stay off until fixed");
    check::<f64>(&c, &cfg, 7, 1e-12);
    check::<f32>(&c, &cfg, 7, 1e-5);
}

/// Grover in executor IR (`algorithms::grover_kops`, MCZ as one diagonal
/// term) through every blocked config vs gate by gate, and vs the closed
/// form: after `k` iterations the marked amplitude is `sin((2k+1)θ)` with
/// `sin θ = 2^{-n/2}` (an independent check of both paths).
#[test]
fn grover_matches_gate_by_gate_and_closed_form() {
    for n in [3usize, 6, 9, 13] {
        let marked = 0b101 & ((1 << n) - 1);
        for k in [1usize, 3] {
            let a = algorithms::grover_state::<f64>(n, marked, k);
            let theta = (2f64.powf(-(n as f64) / 2.0)).asin();
            let want = ((2 * k + 1) as f64 * theta).sin();
            let got = a.amplitudes()[marked];
            assert!(
                (got.norm() - want.abs()).abs() < 1e-12,
                "n={n} k={k}: {got} vs {want}"
            );
            for cfg in configs() {
                let mut b = StateVector::<f64>::new(n);
                b.apply_kops_blocked(&algorithms::grover_kops(n, marked, k), &cfg);
                let d = max_diff(&a, &b);
                assert!(d <= 1e-12, "n={n} k={k} max |Δamp| = {d:e}");
            }
        }
    }
}
