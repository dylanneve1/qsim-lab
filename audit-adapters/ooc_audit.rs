//! Independent audit spot-check of exp/ooc (out-of-core SV) vs in-RAM blocked executor.
//!
//! Tests n ∈ {16, 17, 18, 19, 20} with forced small chunks (64 chunks per file),
//! checking exact amplitude agreement against the in-RAM blocked executor on:
//! 1. QFT circuits (all-to-all controlled rotations, long-range global gates)
//! 2. Edge-biased (0, n-1) alternating boundary entanglers
//! 3. Universal random circuits (deep, 40-50 gates with 1q, 2q, and Toffoli)
//! in both f64 (tolerance 1e-12) and f32 (tolerance 1e-5).

use qsim_lab::algorithms;
use qsim_lab::blocked::BlockConfig;
use qsim_lab::circuit::Circuit;
use qsim_lab::gate::Gate;
use qsim_lab::ooc::{OocConfig, OocStateVector};
use qsim_lab::statevector::{Real, StateVector};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn max_diff<T: Real>(a: &[num_complex::Complex<T>], b: &[num_complex::Complex<T>]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = *x - *y;
            (d.re.to_f64().powi(2) + d.im.to_f64().powi(2)).sqrt()
        })
        .fold(0.0f64, f64::max)
}

fn check_ooc_vs_blocked<T: Real>(c: &Circuit, chunk_bits: usize, tol: f64) {
    let n = c.num_qubits;
    let bcfg = BlockConfig::default();

    // 1. In-RAM blocked reference
    let mut in_ram = StateVector::<T>::new(n);
    in_ram.apply_circuit_blocked(c, &bcfg).unwrap();

    // 2. Out-of-core state vector with forced small chunk size
    let mut ooc = OocStateVector::<T>::temp(
        n,
        chunk_bits,
        OocConfig {
            chunk_bits,
            scratch_dir: None,
            block_config: bcfg,
            restore_order: true,
        },
    )
    .unwrap();

    let stats = ooc.simulate_circuit(c).unwrap();
    let ooc_amps = ooc.read_amplitudes().unwrap();

    let d = max_diff(&ooc_amps, in_ram.amplitudes());
    assert!(
        d <= tol,
        "n={n}, c={chunk_bits}: max |Δamp| = {d:e} exceeds tol {tol:e} (file_passes={}, swap_passes={})",
        stats.file_passes,
        stats.swap_passes
    );
}

/// Boundary-biased adversarial circuit stressing (0, n-1) swaps and global gates.
fn edge_biased_circuit(n: usize, depth: usize, seed: u64) -> Circuit {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut c = Circuit::new(n);
    for _ in 0..depth {
        // H on edge qubits
        c.h(0);
        c.h(n - 1);
        // Long-distance CNOT between 0 and n-1
        c.cnot(0, n - 1);
        // Phase rotations on middle and edge
        c.rz(0, rng.random_range(-3.14..3.14));
        c.rz(n / 2, rng.random_range(-3.14..3.14));
        c.rz(n - 1, rng.random_range(-3.14..3.14));
        c.cnot(n - 1, 0);
        // Toffoli across boundaries if n >= 3
        c.ccx(0, n / 2, n - 1);
        // Random local gate on global qubit
        let gq = rng.random_range(n / 2..n);
        c.rx(gq, rng.random_range(-3.14..3.14));
    }
    c
}

/// Random universal circuit
fn random_universal_circuit(n: usize, gates: usize, seed: u64) -> Circuit {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut c = Circuit::new(n);
    for _ in 0..gates {
        let q = rng.random_range(0..n);
        let th = rng.random_range(-3.14..3.14);
        let other = |rng: &mut StdRng, not: &[usize]| loop {
            let x = rng.random_range(0..n);
            if !not.contains(&x) {
                break x;
            }
        };
        let g = match rng.random_range(0..17) {
            0 => Gate::H(q),
            1 => Gate::X(q),
            2 => Gate::Y(q),
            3 => Gate::Z(q),
            4 => Gate::S(q),
            5 => Gate::Sdg(q),
            6 => Gate::T(q),
            7 => Gate::Tdg(q),
            8 => Gate::Rx(q, th),
            9 => Gate::Ry(q, th),
            10 => Gate::Rz(q, th),
            11 => Gate::Phase(q, th),
            12 => Gate::Cnot(q, other(&mut rng, &[q])),
            13 => Gate::Cz(q, other(&mut rng, &[q])),
            14 => Gate::Swap(q, other(&mut rng, &[q])),
            15 => Gate::CPhase(q, other(&mut rng, &[q]), th),
            _ => {
                let b = other(&mut rng, &[q]);
                let t = other(&mut rng, &[q, b]);
                Gate::Ccx(q, b, t)
            }
        };
        c.gate(g);
    }
    c
}

#[test]
fn spot_check_16_qubits_forced_small_chunks() {
    let n = 16;
    let c = 10; // 64 chunks (1,024 amplitudes per chunk)
    println!("Checking n=16, chunk_bits={c} (64 chunks)");

    // 1. QFT
    let qft = algorithms::qft(n);
    check_ooc_vs_blocked::<f64>(&qft, c, 1e-12);
    check_ooc_vs_blocked::<f32>(&qft, c, 1e-5);

    // 2. Edge-biased
    let edge = edge_biased_circuit(n, 10, 1601);
    check_ooc_vs_blocked::<f64>(&edge, c, 1e-12);
    check_ooc_vs_blocked::<f32>(&edge, c, 1e-5);

    // 3. Universal
    let univ = random_universal_circuit(n, 40, 1602);
    check_ooc_vs_blocked::<f64>(&univ, c, 1e-12);
    check_ooc_vs_blocked::<f32>(&univ, c, 1e-5);
}

#[test]
fn spot_check_17_qubits_forced_small_chunks() {
    let n = 17;
    let c = 11; // 64 chunks (2,048 amplitudes per chunk)
    println!("Checking n=17, chunk_bits={c} (64 chunks)");

    let qft = algorithms::qft(n);
    check_ooc_vs_blocked::<f64>(&qft, c, 1e-12);
    check_ooc_vs_blocked::<f32>(&qft, c, 1e-5);

    let univ = random_universal_circuit(n, 35, 1701);
    check_ooc_vs_blocked::<f64>(&univ, c, 1e-12);
    check_ooc_vs_blocked::<f32>(&univ, c, 1e-5);
}

#[test]
fn spot_check_18_qubits_forced_small_chunks() {
    let n = 18;
    let c = 12; // 64 chunks (4,096 amplitudes per chunk)
    println!("Checking n=18, chunk_bits={c} (64 chunks)");

    let qft = algorithms::qft(n);
    check_ooc_vs_blocked::<f64>(&qft, c, 1e-12);
    check_ooc_vs_blocked::<f32>(&qft, c, 1e-5);

    let edge = edge_biased_circuit(n, 12, 1801);
    check_ooc_vs_blocked::<f64>(&edge, c, 1e-12);
    check_ooc_vs_blocked::<f32>(&edge, c, 1e-5);

    let univ = random_universal_circuit(n, 40, 1802);
    check_ooc_vs_blocked::<f64>(&univ, c, 1e-12);
    check_ooc_vs_blocked::<f32>(&univ, c, 1e-5);
}

#[test]
fn spot_check_19_qubits_forced_small_chunks() {
    let n = 19;
    let c = 13; // 64 chunks (8,192 amplitudes per chunk)
    println!("Checking n=19, chunk_bits={c} (64 chunks)");

    let univ = random_universal_circuit(n, 35, 1901);
    check_ooc_vs_blocked::<f64>(&univ, c, 1e-12);
    check_ooc_vs_blocked::<f32>(&univ, c, 1e-5);
}

#[test]
fn spot_check_20_qubits_forced_small_chunks() {
    let n = 20;
    let c = 14; // 64 chunks (16,384 amplitudes per chunk)
    println!("Checking n=20, chunk_bits={c} (64 chunks)");

    let qft = algorithms::qft(n);
    check_ooc_vs_blocked::<f64>(&qft, c, 1e-12);
    check_ooc_vs_blocked::<f32>(&qft, c, 1e-5);

    let edge = edge_biased_circuit(n, 10, 2001);
    check_ooc_vs_blocked::<f64>(&edge, c, 1e-12);
    check_ooc_vs_blocked::<f32>(&edge, c, 1e-5);

    let univ = random_universal_circuit(n, 40, 2002);
    check_ooc_vs_blocked::<f64>(&univ, c, 1e-12);
    check_ooc_vs_blocked::<f32>(&univ, c, 1e-5);
}
