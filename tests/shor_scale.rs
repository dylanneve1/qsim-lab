//! Exactness cross-checks for the scaled Shor paths:
//! semiclassical vs full QFT, sparse vs dense, gate-level vs permutation.

use num_complex::Complex64;
use qsim_lab::algorithms::gcd;
use qsim_lab::shor::{self, Backend, Instance, Oracle};
use qsim_lab::shor_arith::{self, BeauregardLayout};
use qsim_lab::{Circuit, SparseState, StateVectorF64};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn bases(n: u64, count: usize) -> Vec<u64> {
    (2..n - 1).filter(|&a| gcd(a, n) == 1).take(count).collect()
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Lever 1: the one-control-qubit semiclassical QFT gives exactly the same
/// distribution of the measured integer as the textbook 3n-qubit circuit.
#[test]
fn semiclassical_equals_full_qft_distribution() {
    for (n, count) in [(15u64, 7), (21, 4), (33, 2), (35, 2)] {
        for a in bases(n, count) {
            let full = shor::full_qft_distribution(n, a);
            let inst = Instance::new(n, a, Oracle::Permutation);
            let semi =
                shor::semiclassical_distribution(&inst, shor::dense_initial::<f64>(&inst), 0.0);
            let total: f64 = semi.iter().sum();
            assert!((total - 1.0).abs() < 1e-12, "N={n} a={a} total {total}");
            let d = max_diff(&full, &semi);
            assert!(d < 1e-12, "N={n} a={a}: max |Δp| = {d:e}");
        }
    }
}

/// Lever 2 on Shor: the sparse state gives the same distribution as dense.
#[test]
fn sparse_equals_dense_semiclassical() {
    for (n, count) in [(15u64, 7), (21, 4), (35, 2), (39, 2)] {
        for a in bases(n, count) {
            let inst = Instance::new(n, a, Oracle::Permutation);
            let dense =
                shor::semiclassical_distribution(&inst, shor::dense_initial::<f64>(&inst), 0.0);
            let sparse = shor::semiclassical_distribution(&inst, shor::sparse_initial(&inst), 0.0);
            let d = max_diff(&dense, &sparse);
            assert!(d < 1e-12, "N={n} a={a}: max |Δp| = {d:e}");
        }
    }
}

/// Same seed, same measured bits, dense vs sparse sampling runs.
#[test]
fn sparse_and_dense_runs_agree() {
    for n in [15u64, 21, 55, 77, 91] {
        for a in bases(n, 3) {
            let inst = Instance::new(n, a, Oracle::Permutation);
            for seed in 0..3 {
                let r1 =
                    shor::order_finding(&inst, Backend::DenseF64, &mut StdRng::seed_from_u64(seed));
                let r2 =
                    shor::order_finding(&inst, Backend::Sparse, &mut StdRng::seed_from_u64(seed));
                assert_eq!(r1.measured, r2.measured, "N={n} a={a} seed={seed}");
            }
        }
    }
}

fn random_circuit<R: Rng>(n: usize, len: usize, rng: &mut R) -> Circuit {
    use qsim_lab::Gate::*;
    let mut c = Circuit::new(n);
    for _ in 0..len {
        let q = rng.random_range(0..n);
        let mut p = rng.random_range(0..n);
        if p == q {
            p = (q + 1) % n;
        }
        let r = rng.random_range(0..n);
        let th = rng.random::<f64>() * 6.3;
        let g = match rng.random_range(0..16) {
            0 => H(q),
            1 => X(q),
            2 => Y(q),
            3 => Z(q),
            4 => S(q),
            5 => T(q),
            6 => Rx(q, th),
            7 => Ry(q, th),
            8 => Rz(q, th),
            9 => Phase(q, th),
            10 => Cnot(q, p),
            11 => Cz(q, p),
            12 => Swap(q, p),
            13 => CPhase(q, p, th),
            14 => ISwap(q, p),
            _ => {
                if r != q && r != p {
                    Ccx(q, p, r)
                } else {
                    Sx(q)
                }
            }
        };
        c.gate(g);
    }
    c
}

/// Lever 2, generic: sparse vs dense amplitudes on random circuits.
#[test]
fn sparse_equals_dense_on_random_circuits() {
    let mut rng = StdRng::seed_from_u64(7);
    for trial in 0..60 {
        let n = 2 + trial % 9;
        let c = random_circuit(n, 10 + 3 * trial, &mut rng);
        let start = rng.random_range(0..1u64 << n);
        let mut d = StateVectorF64::basis_state(n, start as usize);
        d.apply_circuit(&c).unwrap();
        let mut s = SparseState::basis_state(n, start);
        s.apply_circuit(&c).unwrap();
        let sd = s.to_dense();
        let diff = d
            .amplitudes()
            .iter()
            .zip(&sd)
            .map(|(x, y)| (x - y).norm())
            .fold(0.0, f64::max);
        assert!(diff < 1e-12, "trial {trial}: {diff:e}");
    }
}

/// Measurements with the same seed collapse sparse and dense identically.
#[test]
fn sparse_measurement_matches_dense() {
    let mut rng = StdRng::seed_from_u64(11);
    for trial in 0..20 {
        let n = 3 + trial % 6;
        let mut c = random_circuit(n, 30, &mut rng);
        c.measure_all();
        let mut d = StateVectorF64::new(n);
        let mut s = SparseState::new(n);
        let seed = rng.random::<u64>();
        let od = c.run(&mut d, &mut StdRng::seed_from_u64(seed)).unwrap();
        let os = c.run(&mut s, &mut StdRng::seed_from_u64(seed)).unwrap();
        assert_eq!(od, os, "trial {trial}");
    }
}

/// Lever 3: the gate-level controlled-U_a equals the permutation oracle on
/// every relevant basis state (control, x < N, b = 0, ancilla = 0), with
/// amplitude 1 (phase included) and nothing left anywhere else.
#[test]
fn beauregard_controlled_ua_is_the_permutation() {
    for n in [15u64, 21, 35] {
        let m = shor::work_bits(n);
        let lay = BeauregardLayout::new(m);
        for a in bases(n, 3) {
            let circ = shor_arith::controlled_ua(&lay, 0, a, n);
            for ctrl in 0..2u64 {
                for x in 0..n {
                    let idx = ctrl | (x << 1);
                    let mut s = SparseState::basis_state(lay.num_qubits(), idx);
                    s.apply_circuit(&circ).unwrap();
                    let y = if ctrl == 1 { a * x % n } else { x };
                    let want = ctrl | (y << 1);
                    let amp = s.amplitude(want);
                    assert!(
                        (amp - Complex64::new(1.0, 0.0)).norm() < 1e-12,
                        "N={n} a={a} c={ctrl} x={x}: amp {amp}"
                    );
                    let rest: f64 = s
                        .iter()
                        .filter(|(k, _)| *k != want)
                        .map(|(_, v)| v.norm_sqr())
                        .sum();
                    assert!(rest < 1e-24, "leak {rest:e}");
                }
            }
        }
    }
}

/// Lever 3 on the whole algorithm: gate-level oracle + semiclassical QFT
/// gives the same outcome distribution as the permutation oracle.
#[test]
fn beauregard_semiclassical_distribution_matches_permutation() {
    for (n, a) in [(15u64, 7u64), (15, 2), (15, 11), (21, 2)] {
        let ip = Instance::new(n, a, Oracle::Permutation);
        let ib = Instance::new(n, a, Oracle::Beauregard);
        let p = shor::semiclassical_distribution(&ip, shor::dense_initial::<f64>(&ip), 1e-15);
        let b = shor::semiclassical_distribution(&ib, shor::dense_initial::<f64>(&ib), 1e-15);
        let d = max_diff(&p, &b);
        assert!(d < 1e-12, "N={n} a={a}: {d:e}");
    }
}

/// The semiclassical circuit with the existing Measure / Reset /
/// ClassicControlled ops, run through `Circuit::run`, measures the same bits
/// as the driver for the same seed.
#[test]
fn semiclassical_circuit_runs_through_classic_control() {
    for (n, a) in [(15u64, 7u64), (21, 2), (21, 5)] {
        let c = shor::semiclassical_circuit(n, a);
        let inst = Instance::new(n, a, Oracle::Beauregard);
        for seed in 0..4 {
            let mut sv = StateVectorF64::new(inst.qubits());
            let bits = c.run(&mut sv, &mut StdRng::seed_from_u64(seed)).unwrap();
            let y = bits
                .iter()
                .enumerate()
                .fold(0u128, |acc, (i, &b)| acc | (u128::from(b) << i));
            let run =
                shor::order_finding(&inst, Backend::DenseF64, &mut StdRng::seed_from_u64(seed));
            assert_eq!(y, run.measured, "N={n} a={a} seed={seed}");
        }
    }
}

#[test]
fn semiclassical_factors_small_numbers() {
    let mut rng = StdRng::seed_from_u64(3);
    for n in [15u64, 21, 33, 35, 143] {
        for backend in [Backend::DenseF64, Backend::Sparse] {
            let (f, _) = shor::factor_semiclassical(n, Oracle::Permutation, backend, 30, &mut rng);
            let (p, q) = f.unwrap_or_else(|| panic!("N={n} {backend:?}"));
            assert_eq!(p * q, n);
        }
    }
}

/// The cache-blocked executor and the per-gate path give the same
/// gate-level distribution.
#[test]
fn beauregard_blocked_matches_unblocked() {
    for (n, a) in [(15u64, 7u64), (21, 5)] {
        let blocked = Instance::new(n, a, Oracle::Beauregard);
        let mut plain = blocked.clone();
        plain.blocked = false;
        let p = shor::semiclassical_distribution(&plain, shor::dense_initial::<f64>(&plain), 1e-15);
        let b =
            shor::semiclassical_distribution(&blocked, shor::dense_initial::<f64>(&blocked), 1e-15);
        let d = max_diff(&p, &b);
        assert!(d < 1e-12, "N={n} a={a}: {d:e}");
    }
}

/// The fused rounds (control qubit handled analytically) give the same
/// distribution as the gate-by-gate n+1-qubit simulation, dense and sparse.
#[test]
fn fused_rounds_match_gate_path() {
    use qsim_lab::shor::fused::{FusedDense, FusedSparse};
    for (n, count) in [(15u64, 7), (21, 4), (35, 2), (39, 2)] {
        for a in bases(n, count) {
            let inst = Instance::new(n, a, Oracle::Permutation);
            let gate =
                shor::semiclassical_distribution(&inst, shor::dense_initial::<f64>(&inst), 0.0);
            let fd = shor::semiclassical_distribution(&inst, FusedDense::<f64>::new(&inst), 0.0);
            let fs = shor::semiclassical_distribution(&inst, FusedSparse::new(&inst), 0.0);
            assert!(max_diff(&gate, &fd) < 1e-12, "dense N={n} a={a}");
            assert!(max_diff(&gate, &fs) < 1e-12, "sparse N={n} a={a}");
            let f32d = shor::semiclassical_distribution(&inst, FusedDense::<f32>::new(&inst), 0.0);
            assert!(max_diff(&gate, &f32d) < 1e-5, "f32 N={n} a={a}");
        }
    }
}

#[test]
fn fused_runs_measure_the_same_bits() {
    for n in [15u64, 21, 55, 77, 91, 143, 1022117] {
        for a in bases(n, 3) {
            let inst = Instance::new(n, a, Oracle::Permutation);
            for seed in 0..2 {
                let r =
                    |b| shor::order_finding(&inst, b, &mut StdRng::seed_from_u64(seed)).measured;
                let base = r(Backend::DenseF64);
                assert_eq!(base, r(Backend::FusedF64), "N={n} a={a}");
                assert_eq!(base, r(Backend::FusedSparse), "N={n} a={a}");
            }
        }
    }
}

/// Ripple-carry oracle: gate-by-gate mode and reversible-block mode produce identical distributions.
#[test]
fn ripple_gate_by_gate_matches_reversible_block() {
    for (n, count) in [(15u64, 4), (21, 3)] {
        for a in bases(n, count) {
            let block_inst = Instance::new(n, a, Oracle::Ripple);
            let mut gate_inst = block_inst.clone();
            gate_inst.gate_by_gate = true;

            let block_dist = shor::semiclassical_distribution(
                &block_inst,
                shor::sparse_initial(&block_inst),
                1e-15,
            );
            let gate_dist = shor::semiclassical_distribution(
                &gate_inst,
                shor::sparse_initial(&gate_inst),
                1e-15,
            );
            let d = max_diff(&block_dist, &gate_dist);
            assert!(d < 1e-12, "N={n} a={a}: gate-by-gate vs block diff {d:e}");
        }
    }
}

/// Ripple-carry oracle gives exactly the same measurement distribution as the permutation oracle.
#[test]
fn ripple_distribution_matches_permutation() {
    for (n, count) in [(15u64, 7), (21, 4)] {
        for a in bases(n, count) {
            let perm_inst = Instance::new(n, a, Oracle::Permutation);
            let rip_inst = Instance::new(n, a, Oracle::Ripple);

            let perm_dist = shor::semiclassical_distribution(
                &perm_inst,
                shor::sparse_initial(&perm_inst),
                1e-15,
            );
            let rip_dist =
                shor::semiclassical_distribution(&rip_inst, shor::sparse_initial(&rip_inst), 1e-15);
            let d = max_diff(&perm_dist, &rip_dist);
            assert!(d < 1e-12, "N={n} a={a}: ripple vs perm diff {d:e}");
        }
    }
}

/// Same seed gives bit-identical measured integers for Ripple vs Permutation oracle
/// across small, medium, and 1e6 moduli.
#[test]
fn ripple_runs_measure_the_same_bits_as_permutation() {
    for (n, count) in [(15u64, 4), (21, 4), (143, 2), (1003, 1), (1005973, 1)] {
        for a in bases(n, count) {
            let perm_inst = Instance::new(n, a, Oracle::Permutation);
            let rip_inst = Instance::new(n, a, Oracle::Ripple);
            for seed in 0..2 {
                let r_perm = shor::order_finding(
                    &perm_inst,
                    Backend::Sparse,
                    &mut StdRng::seed_from_u64(seed),
                );
                let r_rip = shor::order_finding(
                    &rip_inst,
                    Backend::Sparse,
                    &mut StdRng::seed_from_u64(seed),
                );
                assert_eq!(
                    r_perm.measured, r_rip.measured,
                    "N={n} a={a} seed={seed}: perm measured {} != ripple measured {}",
                    r_perm.measured, r_rip.measured
                );
            }
        }
    }
}

/// Ripple circuit run via Circuit::run with classic control matches driver.
#[test]
fn ripple_circuit_runs_through_classic_control() {
    for (n, a) in [(15u64, 7u64), (21, 2)] {
        let c = shor::semiclassical_ripple_circuit(n, a);
        let inst = Instance::new(n, a, Oracle::Ripple);
        for seed in 0..3 {
            let mut sp = SparseState::basis_state(inst.qubits(), 0);
            let bits = c.run(&mut sp, &mut StdRng::seed_from_u64(seed)).unwrap();
            let y = bits
                .iter()
                .enumerate()
                .fold(0u128, |acc, (i, &b)| acc | (u128::from(b) << i));
            let run = shor::order_finding(&inst, Backend::Sparse, &mut StdRng::seed_from_u64(seed));
            assert_eq!(y, run.measured, "N={n} a={a} seed={seed}");
        }
    }
}

// ---------------------------------------------------------------------------
// Round 4: bit-sliced branch tracking (src/shor/sliced.rs) and the windowed
// table-lookup oracle (src/shor_window.rs).

fn dist<S: shor::OrderFindingState>(inst: &Instance, s: S) -> Vec<f64> {
    shor::semiclassical_distribution(inst, s, 1e-15)
}

/// Sliced branch tracking of the ripple circuit = the gate-by-gate sparse
/// state vector of the same circuit = the permutation oracle (exact
/// distributions of the measured integer).
#[test]
fn sliced_ripple_distribution_matches_gate_by_gate_and_permutation() {
    for (n, count) in [(15u64, 7), (21, 4), (33, 2), (35, 2)] {
        for a in bases(n, count) {
            let perm = Instance::new(n, a, Oracle::Permutation);
            let rip = Instance::new(n, a, Oracle::Ripple);
            let mut rip_gbg = rip.clone();
            rip_gbg.gate_by_gate = true;
            let d_perm = dist(&perm, shor::sparse_initial(&perm));
            let d_sl = dist(&rip, shor::sliced::SlicedState::<f64>::new(&rip));
            let d = max_diff(&d_perm, &d_sl);
            assert!(d < 1e-12, "N={n} a={a}: sliced vs permutation {d:e}");
            if n <= 21 {
                let d_gbg = dist(&rip_gbg, shor::sparse_initial(&rip_gbg));
                let d = max_diff(&d_gbg, &d_sl);
                assert!(
                    d < 1e-12,
                    "N={n} a={a}: sliced vs gate-by-gate sparse {d:e}"
                );
            }
        }
    }
}

/// The windowed oracle, simulated gate by gate on the sparse state vector
/// (every gate through `SparseState::apply_gate`) and by sliced branch
/// tracking, gives the permutation oracle's exact distribution.
#[test]
fn windowed_distribution_matches_permutation() {
    for (n, count) in [(15u64, 4), (21, 3), (35, 1)] {
        for w in [1usize, 2, 3, 4] {
            for a in bases(n, count) {
                let perm = Instance::new(n, a, Oracle::Permutation);
                let win = Instance::new(n, a, Oracle::Windowed(w));
                let d_perm = dist(&perm, shor::sparse_initial(&perm));
                let d_sl = dist(&win, shor::sliced::SlicedState::<f64>::new(&win));
                let d = max_diff(&d_perm, &d_sl);
                assert!(
                    d < 1e-12,
                    "N={n} w={w} a={a}: sliced windowed vs perm {d:e}"
                );
                if n <= 21 && win.qubits() <= 64 {
                    let d_gbg = dist(&win, shor::sparse_initial(&win));
                    let d = max_diff(&d_perm, &d_gbg);
                    assert!(
                        d < 1e-12,
                        "N={n} w={w} a={a}: gate-by-gate windowed vs perm {d:e}"
                    );
                }
            }
        }
    }
}

/// f32 amplitudes in the sliced state: distribution within 1e-5 of f64.
#[test]
fn sliced_f32_close_to_f64() {
    for (n, a) in [(21u64, 2u64), (35, 2), (55, 7)] {
        let inst = Instance::new(n, a, Oracle::Ripple);
        let d64 = dist(&inst, shor::sliced::SlicedState::<f64>::new(&inst));
        let d32 = dist(&inst, shor::sliced::SlicedState::<f32>::new(&inst));
        let d = max_diff(&d64, &d32);
        assert!(d < 1e-5, "N={n} a={a}: f32 vs f64 {d:e}");
    }
}

/// Same seed, same measured integer: sliced ripple / sliced windowed vs the
/// existing sparse ripple path and the fused permutation path, up to N ≈ 1e6
/// (64 qubits for ripple, 88 for windowed w=4).
#[test]
fn sliced_runs_measure_the_same_bits() {
    for (n, count) in [(15u64, 3), (143, 2), (1003, 1), (1_005_973, 1)] {
        for a in bases(n, count) {
            let perm = Instance::new(n, a, Oracle::Permutation);
            let rip = Instance::new(n, a, Oracle::Ripple);
            let win = Instance::new(n, a, Oracle::Windowed(4));
            for seed in 0..2 {
                let r = |inst: &Instance, b: Backend| {
                    shor::order_finding(inst, b, &mut StdRng::seed_from_u64(seed)).measured
                };
                let base = r(&perm, Backend::FusedSparse);
                assert_eq!(
                    base,
                    r(&rip, Backend::SlicedF64),
                    "ripple N={n} a={a} seed={seed}"
                );
                assert_eq!(
                    base,
                    r(&win, Backend::SlicedF64),
                    "windowed N={n} a={a} seed={seed}"
                );
                if n <= 1003 {
                    assert_eq!(base, r(&rip, Backend::Sparse), "sparse ripple N={n} a={a}");
                }
            }
        }
    }
}

/// Beyond the 64-qubit u64-key limit of the sparse state: a 24-bit modulus
/// (ripple: 76 qubits, windowed: 104) with a base of small order.
#[test]
fn sliced_beyond_64_qubits_matches_permutation() {
    // N = 4093 * 4099, λ = 4·3·11·31·683; a = g^(4·683) has order dividing 1023
    let n = 4093u64 * 4099;
    let a = (2..1000u64)
        .map(|g| qsim_lab::algorithms::pow_mod(g, 4 * 683, n))
        .find(|&a| a > 1 && gcd(a, n) == 1)
        .unwrap();
    let perm = Instance::new(n, a, Oracle::Permutation);
    let rip = Instance::new(n, a, Oracle::Ripple);
    let win = Instance::new(n, a, Oracle::Windowed(3));
    assert!(rip.qubits() > 64 && win.qubits() > 64);
    for seed in 0..2 {
        let r = |inst: &Instance, b: Backend| {
            shor::order_finding(inst, b, &mut StdRng::seed_from_u64(seed)).measured
        };
        let base = r(&perm, Backend::FusedSparse);
        assert_eq!(base, r(&rip, Backend::SlicedF64), "seed={seed}");
        assert_eq!(base, r(&win, Backend::SlicedF64), "seed={seed}");
    }
}
