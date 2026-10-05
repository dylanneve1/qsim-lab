//! Nested L1 tiling (`BlockConfig::l1_tile_bytes`) must agree with the untiled
//! blocked executor and with gate-by-gate application: max |Δamplitude|
//! <= 1e-12 (f64) and <= 1e-5 (f32).

mod audit_common;
mod common;

use common::random_universal;
use proptest::prelude::*;
use qsim_lab::algorithms;
use qsim_lab::circuit::{Circuit, Op};
use qsim_lab::engines::blocked::{lower_gates, tile_stats, BlockConfig};
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

/// (block bytes, slots, L1 tile bytes) triples; `small_n = 4` and the sizes
/// are in bytes of `Complex<f64>` = 16 B (f32 uses half as many amplitudes),
/// so even 5..12-qubit registers get several tiles per block and gathered
/// (non-contiguous) blocks.
const GEOMS: [(usize, usize, usize); 7] = [
    (512, 2, 128),
    (1024, 3, 256),
    (1024, 3, 128),
    (4096, 4, 512),
    (4096, 2, 1024),
    (16 << 10, 6, 1 << 10),
    (256 << 10, 6, 32 << 10),
];

fn cfg(g: (usize, usize, usize), tile: bool, simd: bool) -> BlockConfig {
    BlockConfig {
        block_bytes: g.0,
        slots: g.1,
        l1_tile_bytes: if tile { g.2 } else { 0 },
        small_n: 4,
        simd,
        ..BlockConfig::default()
    }
}

fn check<T: Real>(c: &Circuit, seed: u64, tol: f64) {
    let init = start::<T>(c.num_qubits, seed);
    let mut reference = init.clone();
    reference.apply_circuit(c).unwrap();
    for g in GEOMS {
        for simd in [false, true] {
            let mut untiled = init.clone();
            let mut tiled = init.clone();
            untiled
                .apply_circuit_blocked(c, &cfg(g, false, simd))
                .unwrap();
            tiled.apply_circuit_blocked(c, &cfg(g, true, simd)).unwrap();
            let d_ref = max_diff(&tiled, &reference);
            let d_unt = max_diff(&tiled, &untiled);
            assert!(
                d_ref <= tol,
                "tiled vs gate-by-gate {d_ref:e} ({g:?}, simd {simd})"
            );
            assert!(
                d_unt <= tol,
                "tiled vs untiled {d_unt:e} ({g:?}, simd {simd})"
            );
        }
    }
}

#[test]
fn algorithms_agree() {
    for n in [4, 7, 10, 12, 13] {
        let mut rng = StdRng::seed_from_u64(n as u64);
        for c in [
            algorithms::ghz(n),
            algorithms::qft(n),
            algorithms::random_brickwork(n, 6, &mut rng),
        ] {
            check::<f64>(&c, 1, 1e-12);
            check::<f32>(&c, 1, 1e-5);
        }
    }
}

/// Tiling must actually engage on these tests (otherwise they prove nothing).
#[test]
fn tiling_engages_on_brickwork() {
    let mut rng = StdRng::seed_from_u64(7);
    let n = 12;
    let c = algorithms::random_brickwork(n, 8, &mut rng);
    let gates: Vec<Gate> = c
        .ops
        .iter()
        .filter_map(|o| match o {
            Op::Gate(g) => Some(*g),
            _ => None,
        })
        .collect();
    let ops = lower_gates(&gates);
    for g in GEOMS.iter().take(6) {
        let st = tile_stats::<f64>(&ops, n, &cfg(*g, true, false));
        assert!(st.tile_bits >= 3, "{g:?} {st:?}");
        assert!(st.tiled_ops > 0 && st.runs > 0, "{g:?} {st:?}");
        // every tile run must hold more than one op on average, else there
        // is no reuse
        assert!(st.tiled_ops >= st.runs, "{g:?} {st:?}");
    }
}

/// With a single stage (block covers the register) and ops already ordered
/// tile-first there is no reordering, so tiling must be bit-identical.
#[test]
fn bit_identical_without_reordering() {
    let n = 9;
    let mut c = Circuit::new(n);
    for q in 0..4 {
        c.ry(q, 0.3 + q as f64).rz(q, 0.7 * q as f64);
    }
    c.cnot(0, 1).cnot(2, 3).h(1).t(3);
    let init = start::<f64>(n, 3);
    let mut a = init.clone();
    let mut b = init.clone();
    let g = (1usize << 20, 6usize, 256usize); // block >= register
    a.apply_circuit_blocked(&c, &cfg(g, false, false)).unwrap();
    b.apply_circuit_blocked(&c, &cfg(g, true, false)).unwrap();
    assert_eq!(max_diff(&a, &b), 0.0);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn random_circuits_agree(n in 4usize..13, len in 1usize..160, seed in 0u64..10_000) {
        let mut rng = StdRng::seed_from_u64(seed);
        let c = random_universal(n, len, &mut rng);
        check::<f64>(&c, seed, 1e-12);
        check::<f32>(&c, seed, 1e-5);
    }

    /// Brickwork-like layers (many local ops) plus controlled ops whose
    /// controls sit above the tile.
    #[test]
    fn layered_circuits_agree(n in 5usize..13, depth in 1usize..8, seed in 0u64..10_000) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut c = algorithms::random_brickwork(n, depth, &mut rng);
        for _ in 0..depth * 2 {
            let (a, b) = (rng.random_range(0..n), rng.random_range(0..n));
            if a != b {
                c.cnot(a, b);
                c.cphase(a, b, rng.random::<f64>() * 3.0);
                c.rx(rng.random_range(0..n), rng.random::<f64>());
            }
        }
        check::<f64>(&c, seed, 1e-12);
        check::<f32>(&c, seed, 1e-5);
    }
}

/// Differential check against the independent naive state vector of
/// `audit_common` (edge-case angles and qubits, SWAP/CCX/CPhase included),
/// tiled and untiled, portable and SIMD kernels, f64 and f32.
#[test]
fn tiled_matches_audit_reference() {
    let mut rng = StdRng::seed_from_u64(audit_common::base_seed() ^ 0x11ee);
    for &n in &[3usize, 5, 6, 8, 9, 11, 12] {
        for _ in 0..audit_common::iters().max(4) {
            let depth = rng.random_range(1..120);
            let c = audit_common::random_circuit(&mut rng, n, depth, false, false);
            let r = audit_common::RefSv::run(&c);
            for g in GEOMS {
                for simd in [false, true] {
                    let mut a = StateVector::<f64>::new(n);
                    a.apply_circuit_blocked(&c, &cfg(g, true, simd)).unwrap();
                    let d = audit_common::max_amp_diff(&r.a, a.amplitudes().iter().copied());
                    assert!(d <= 1e-12, "f64 n={n} {g:?} simd {simd}: {d:e}");
                    let mut b = StateVector::<f32>::new(n);
                    b.apply_circuit_blocked(&c, &cfg(g, true, simd)).unwrap();
                    let d = audit_common::max_amp_diff(
                        &r.a,
                        b.amplitudes()
                            .iter()
                            .map(|z| num_complex::Complex64::new(z.re as f64, z.im as f64)),
                    );
                    assert!(d <= 1e-5, "f32 n={n} {g:?} simd {simd}: {d:e}");
                }
            }
        }
    }
}
