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
                .fold(0u64, |acc, (i, &b)| acc | (u64::from(b) << i));
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
