//! Audit adapter for exp/sv: the cache-blocked executor
//! (`StateVector::apply_circuit_blocked`) against the independent reference
//! in `tests/audit_common`, over edge-biased circuits and adversarial
//! `BlockConfig`s (byte-sized blocks, 0..8 slots, fusion on/off, small_n
//! forcing the multi-chunk path even at n = 2).
//!
//! Copy to `tests/` of an exp/sv checkout together with `tests/audit_common/`.

mod audit_common;

use audit_common::*;
use qsim_lab::blocked::BlockConfig;
use qsim_lab::{StateVectorF32, StateVectorF64};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn configs(rng: &mut StdRng) -> Vec<BlockConfig> {
    let mut v = vec![BlockConfig::default()];
    for _ in 0..4 {
        v.push(BlockConfig {
            block_bytes: [8usize, 16, 32, 64, 256, 1024, 4096, 1 << 18][rng.random_range(0..8)],
            slots: rng.random_range(0..9),
            fuse_1q: rng.random_bool(0.5),
            small_n: rng.random_range(0..4),
        });
    }
    v
}

#[test]
fn blocked_f64_matches_reference() {
    let mut worst = 0.0f64;
    for it in 0..20 * iters() {
        for &n in &SIZES {
            let seed = base_seed() ^ 0xB10C ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let depth = rng.random_range(1..150);
            let c = random_circuit(&mut rng, n, depth, false, false);
            let r = RefSv::run(&c);
            for cfg in configs(&mut rng) {
                let mut sv = StateVectorF64::new(n);
                sv.apply_circuit_blocked(&c, &cfg).unwrap();
                let d = max_amp_diff(&r.a, (0..1 << n).map(|i| sv.amplitude(i)));
                worst = worst.max(d);
                assert!(d <= 1e-12, "blocked f64 Δ={d:e} seed={seed} n={n} cfg={cfg:?} circuit={:?}", c.ops);
            }
        }
    }
    eprintln!("blocked f64 worst |Δamp| = {worst:e}");
}

#[test]
fn blocked_f32_matches_reference() {
    let mut worst = 0.0f64;
    for it in 0..20 * iters() {
        for &n in &SIZES {
            let seed = base_seed() ^ 0xB10C32 ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let depth = rng.random_range(1..60);
            let c = random_circuit(&mut rng, n, depth, false, false);
            let r = RefSv::run(&c);
            for cfg in configs(&mut rng) {
                let mut sv = StateVectorF32::new(n);
                sv.apply_circuit_blocked(&c, &cfg).unwrap();
                let d = max_amp_diff(&r.a, (0..1 << n).map(|i| sv.amplitude(i)));
                worst = worst.max(d);
                assert!(d <= 1e-5, "blocked f32 Δ={d:e} seed={seed} n={n} cfg={cfg:?} circuit={:?}", c.ops);
            }
        }
    }
    eprintln!("blocked f32 worst |Δamp| = {worst:e}");
}

/// 16 and 18 qubits with the default config: the real multi-threaded,
/// multi-stage path.
#[test]
fn blocked_large_registers_match_reference() {
    for (k, &n) in [16usize, 18].iter().enumerate() {
        let seed = base_seed() ^ 0xB1A6 ^ k as u64;
        let mut rng = StdRng::seed_from_u64(seed);
        let c = random_circuit(&mut rng, n, 60 * iters(), false, false);
        let r = RefSv::run(&c);
        for cfg in [BlockConfig::default(), BlockConfig { block_bytes: 4096, slots: 2, fuse_1q: true, small_n: 0 }] {
            let mut sv = StateVectorF64::new(n);
            sv.apply_circuit_blocked(&c, &cfg).unwrap();
            let d = max_amp_diff(&r.a, (0..1 << n).map(|i| sv.amplitude(i)));
            assert!(d <= 1e-12, "blocked large f64 Δ={d:e} seed={seed} n={n} cfg={cfg:?}");
            let mut s32 = StateVectorF32::new(n);
            s32.apply_circuit_blocked(&c, &cfg).unwrap();
            let d32 = max_amp_diff(&r.a, (0..1 << n).map(|i| s32.amplitude(i)));
            assert!(d32 <= 1e-5, "blocked large f32 Δ={d32:e} seed={seed} n={n} cfg={cfg:?}");
        }
    }
}

/// Deep single-qubit runs: fusion multiplies long products in f64 then
/// rounds once; make sure nothing drifts (n=1..3, 500 gates on one qubit).
#[test]
fn blocked_long_fusion_runs() {
    for n in 1..=3 {
        let seed = base_seed() ^ 0xF05E ^ n as u64;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut c = qsim_lab::Circuit::new(n);
        for _ in 0..500 {
            let mut g;
            loop {
                g = random_gate(&mut rng, n, false, false);
                if g.qubits().len() == 1 {
                    break;
                }
            }
            c.gate(g);
        }
        let r = RefSv::run(&c);
        let cfg = BlockConfig { small_n: 0, block_bytes: 16, ..BlockConfig::default() };
        let mut sv = StateVectorF64::new(n);
        sv.apply_circuit_blocked(&c, &cfg).unwrap();
        let d = max_amp_diff(&r.a, (0..1 << n).map(|i| sv.amplitude(i)));
        assert!(d <= 1e-12, "fusion f64 Δ={d:e} n={n}");
        let mut s32 = StateVectorF32::new(n);
        s32.apply_circuit_blocked(&c, &cfg).unwrap();
        let d32 = max_amp_diff(&r.a, (0..1 << n).map(|i| s32.amplitude(i)));
        assert!(d32 <= 1e-5, "fusion f32 Δ={d32:e} n={n}");
    }
}
