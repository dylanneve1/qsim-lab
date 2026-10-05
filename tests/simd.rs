//! The runtime-dispatched AVX2+FMA chunk kernels must agree with the portable
//! kernels and with gate-by-gate application.
//!
//! On a CPU without AVX2+FMA `BlockConfig::simd` falls back to the portable
//! kernels, so these tests then compare the portable path with itself (and
//! still check it against gate-by-gate application).

mod common;

use common::random_universal;
use proptest::prelude::*;
use qsim_lab::algorithms;
use qsim_lab::circuit::Circuit;
use qsim_lab::engines::blocked::{simd_available, BlockConfig};
use qsim_lab::engines::statevector::{Real, StateVector};
use qsim_lab::Gate;
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

fn start<T: Real>(n: usize, seed: u64) -> StateVector<T> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut s = StateVector::<T>::new(n);
    for q in 0..n {
        s.apply_gate(&Gate::Ry(q, rng.random::<f64>() * 3.0))
            .unwrap();
        s.apply_gate(&Gate::Rz(q, rng.random::<f64>() * 3.0))
            .unwrap();
    }
    s
}

/// Tiny blocks and low `small_n` so even small registers exercise the
/// chunked, gathered and small-bit kernel paths.
fn cfgs(simd: bool) -> Vec<BlockConfig> {
    [(1, 2), (1, 4), (2, 0), (4, 3), (256, 6)]
        .into_iter()
        .map(|(kib, slots)| BlockConfig {
            block_bytes: kib << 10,
            slots,
            small_n: 4,
            simd,
            ..BlockConfig::default()
        })
        .collect()
}

/// dispatched vs portable vs gate-by-gate on one circuit.
fn check3<T: Real>(c: &Circuit, seed: u64, tol: f64) {
    let init = start::<T>(c.num_qubits, seed);
    let mut reference = init.clone();
    reference.apply_circuit(c).unwrap();
    for (fast, slow) in cfgs(true).iter().zip(cfgs(false).iter()) {
        let mut a = init.clone();
        let mut b = init.clone();
        a.apply_circuit_blocked(c, fast).unwrap();
        b.apply_circuit_blocked(c, slow).unwrap();
        let (d_ab, d_ar, d_br) = (
            max_diff(&a, &b),
            max_diff(&a, &reference),
            max_diff(&b, &reference),
        );
        assert!(d_ab <= tol, "dispatched vs portable {d_ab:e} ({fast:?})");
        assert!(
            d_ar <= tol,
            "dispatched vs gate-by-gate {d_ar:e} ({fast:?})"
        );
        assert!(d_br <= tol, "portable vs gate-by-gate {d_br:e} ({slow:?})");
    }
}

#[test]
fn reports_detection() {
    // Documents which path the other tests exercised.
    eprintln!("avx2+fma available: {}", simd_available());
}

#[test]
fn algorithms_agree() {
    for n in [3, 6, 9, 12, 14] {
        let mut rng = StdRng::seed_from_u64(n as u64);
        for c in [
            algorithms::ghz(n),
            algorithms::qft(n),
            algorithms::random_brickwork(n, 6, &mut rng),
        ] {
            check3::<f64>(&c, 1, 1e-12);
            check3::<f32>(&c, 1, 1e-5);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn random_circuits_agree(n in 3usize..12, len in 1usize..150, seed in 0u64..10_000) {
        let mut rng = StdRng::seed_from_u64(seed);
        let c = random_universal(n, len, &mut rng);
        check3::<f64>(&c, seed, 1e-12);
        check3::<f32>(&c, seed, 1e-5);
    }
}
