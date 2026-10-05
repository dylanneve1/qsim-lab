//! Dense k-qubit fusion (`BlockConfig::dense_fusion`) must agree with the
//! unfused blocked executor, gate-by-gate application and the independent
//! reference state vector of `audit_common`: max |Δamplitude| <= 1e-12 (f64)
//! and <= 1e-5 (f32), for k = 2 and 3, portable and SIMD kernels, with and
//! without nested L1 tiling, on blocks small enough that every kernel path
//! (scalar `l < 3 + k`, lane-exchange for targets below bit 3, gathered
//! blocks) is exercised.

#[path = "../audit_common/mod.rs"]
mod audit_common;
#[path = "../common/mod.rs"]
mod common;

use common::random_universal;
use proptest::prelude::*;
use qsim_lab::algorithms;
use qsim_lab::circuit::{Circuit, Op};
use qsim_lab::engines::blocked::{fusion_stats, lower_gates, BlockConfig};
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

/// (block bytes, slots, L1 tile bytes); `small_n = 4`. Sizes in bytes of
/// `Complex<f64>`, so 5..12-qubit registers get several (gathered) blocks.
const GEOMS: [(usize, usize, usize); 6] = [
    (256, 2, 0),
    (1024, 3, 0),
    (1024, 3, 256),
    (4096, 4, 0),
    (16 << 10, 6, 1 << 10),
    (256 << 10, 6, 0),
];

/// (width, cost-rule minimum) pairs: forced fusion of every group of two
/// or more gates (maximum kernel coverage) and the default rule.
const KS: [(usize, usize); 3] = [(2, 1), (3, 1), (3, 0)];

fn cfg(g: (usize, usize, usize), (k, min): (usize, usize), simd: bool) -> BlockConfig {
    BlockConfig {
        block_bytes: g.0,
        slots: g.1,
        l1_tile_bytes: g.2,
        small_n: 4,
        simd,
        dense_fusion: k,
        dense_min_ops: min,
        ..BlockConfig::default()
    }
}

fn check<T: Real>(c: &Circuit, seed: u64, tol: f64) {
    let init = start::<T>(c.num_qubits, seed);
    let mut reference = init.clone();
    reference.apply_circuit(c).unwrap();
    for g in GEOMS {
        for simd in [false, true] {
            let mut plain = init.clone();
            plain
                .apply_circuit_blocked(c, &cfg(g, (0, 0), simd))
                .unwrap();
            for k in KS {
                let mut fused = init.clone();
                fused.apply_circuit_blocked(c, &cfg(g, k, simd)).unwrap();
                let d_ref = max_diff(&fused, &reference);
                let d_pl = max_diff(&fused, &plain);
                assert!(
                    d_ref <= tol,
                    "k={k:?} vs gate-by-gate {d_ref:e} ({g:?}, simd {simd})"
                );
                assert!(
                    d_pl <= tol,
                    "k={k:?} vs unfused {d_pl:e} ({g:?}, simd {simd})"
                );
            }
        }
    }
}

#[test]
fn algorithms_agree() {
    for n in [4, 5, 7, 10, 12, 13] {
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

/// Fusion must actually produce dense ops on these tests.
#[test]
fn fusion_engages() {
    let mut rng = StdRng::seed_from_u64(7);
    let c = algorithms::random_brickwork(12, 6, &mut rng);
    let gates: Vec<Gate> = c
        .ops
        .iter()
        .map(|o| match o {
            Op::Gate(g) => *g,
            _ => unreachable!(),
        })
        .collect();
    let ops = lower_gates(&gates);
    for g in GEOMS {
        let off = fusion_stats::<f64>(&ops, 12, &cfg(g, (0, 0), true));
        let k2 = fusion_stats::<f64>(&ops, 12, &cfg(g, (2, 1), true));
        let k3 = fusion_stats::<f64>(&ops, 12, &cfg(g, (3, 1), true));
        assert_eq!(off.dense2 + off.dense3, 0);
        assert!(k2.dense2 > 0 && k2.dense3 == 0, "{g:?}: {k2:?}");
        assert!(k3.dense2 + k3.dense3 > 0, "{g:?}: {k3:?}");
        assert!(k2.passes < off.passes, "{g:?}: {off:?} -> {k2:?}");
    }
    // 3-qubit groups need two consecutive brick layers in one stage, which
    // happens when the register is one block (n <= small_n = 12 by default)
    let one_block = BlockConfig {
        dense_fusion: 3,
        dense_min_ops: 1,
        ..BlockConfig::default()
    };
    let k3 = fusion_stats::<f64>(&ops, 12, &one_block);
    assert!(k3.stages == 1 && k3.dense3 > 0, "{k3:?}");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn random_circuits_agree(n in 4usize..13, len in 1usize..160, seed in 0u64..10_000) {
        let mut rng = StdRng::seed_from_u64(seed);
        let c = random_universal(n, len, &mut rng);
        check::<f64>(&c, seed, 1e-12);
        check::<f32>(&c, seed, 1e-5);
    }

    /// Brickwork layers plus controlled ops, CCX and swaps (3-qubit groups,
    /// controls outside the block).
    #[test]
    fn layered_circuits_agree(n in 5usize..13, depth in 1usize..8, seed in 0u64..10_000) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut c = algorithms::random_brickwork(n, depth, &mut rng);
        for _ in 0..depth * 2 {
            let (a, b, d) = (rng.random_range(0..n), rng.random_range(0..n), rng.random_range(0..n));
            if a != b {
                c.cnot(a, b);
                c.cphase(a, b, rng.random::<f64>() * 3.0);
                c.rx(rng.random_range(0..n), rng.random::<f64>());
                c.swap(a, b);
                if d != a && d != b {
                    c.ccx(a, b, d);
                }
            }
        }
        check::<f64>(&c, seed, 1e-12);
        check::<f32>(&c, seed, 1e-5);
    }
}

/// Differential check against the independent naive state vector of
/// `audit_common` (edge-case angles and qubits, SWAP/CCX/CPhase included).
#[test]
fn fused_matches_audit_reference() {
    let mut rng = StdRng::seed_from_u64(audit_common::base_seed() ^ 0xde75e);
    for &n in &[2usize, 3, 5, 6, 8, 9, 11, 12] {
        for _ in 0..audit_common::iters().max(4) {
            let depth = rng.random_range(1..120);
            let c = audit_common::random_circuit(&mut rng, n, depth, false, false);
            let r = audit_common::RefSv::run(&c);
            for g in GEOMS {
                for simd in [false, true] {
                    for k in KS {
                        let mut a = StateVector::<f64>::new(n);
                        a.apply_circuit_blocked(&c, &cfg(g, k, simd)).unwrap();
                        let d = audit_common::max_amp_diff(&r.a, a.amplitudes().iter().copied());
                        assert!(d <= 1e-12, "f64 n={n} k={k:?} {g:?} simd {simd}: {d:e}");
                        let mut b = StateVector::<f32>::new(n);
                        b.apply_circuit_blocked(&c, &cfg(g, k, simd)).unwrap();
                        let d = audit_common::max_amp_diff(
                            &r.a,
                            b.amplitudes()
                                .iter()
                                .map(|z| num_complex::Complex64::new(z.re as f64, z.im as f64)),
                        );
                        assert!(d <= 1e-5, "f32 n={n} k={k:?} {g:?} simd {simd}: {d:e}");
                    }
                }
            }
        }
    }
}

/// Where the cost rule fuses nothing (arithmetic, QFT; brickwork in most
/// stages), the fused executor runs exactly the unfused op list:
/// bit-identical results.
#[test]
fn no_dense_op_is_bit_identical() {
    let mut rng = StdRng::seed_from_u64(3);
    for (c, must_be_empty) in [
        (qsim_lab::bench::cuccaro_adder(6), true),
        (algorithms::qft(14), true),
        (algorithms::random_brickwork(14, 8, &mut rng), false),
    ] {
        let n = c.num_qubits;
        let ops = lower_gates(
            &c.ops
                .iter()
                .map(|o| match o {
                    Op::Gate(g) => *g,
                    _ => unreachable!(),
                })
                .collect::<Vec<_>>(),
        );
        for g in GEOMS {
            let st = fusion_stats::<f32>(&ops, n, &cfg(g, (2, 0), true));
            if st.dense2 + st.dense3 > 0 {
                assert!(!must_be_empty, "{g:?}: {st:?}");
                continue;
            }
            let mut a = start::<f32>(n, 5);
            let mut b = a.clone();
            a.apply_circuit_blocked(&c, &cfg(g, (0, 0), true)).unwrap();
            b.apply_circuit_blocked(&c, &cfg(g, (2, 0), true)).unwrap();
            assert_eq!(a.amplitudes(), b.amplitudes(), "{g:?}");
        }
    }
}
