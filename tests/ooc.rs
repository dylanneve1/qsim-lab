//! Differential tests comparing out-of-core simulation against the in-RAM blocked executor.
//!
//! Requirement:
//! "It must be EXACT: identical amplitudes vs the in-RAM blocked executor at 16–22 qubits
//! with a small forced chunk size (proptests)."

mod common;

use common::random_universal;
use proptest::prelude::*;
use qsim_lab::algorithms;
use qsim_lab::blocked::BlockConfig;
use qsim_lab::circuit::Circuit;
use qsim_lab::ooc::{OocConfig, OocStateVector};
use qsim_lab::statevector::{Real, StateVector};
use rand::rngs::StdRng;
use rand::SeedableRng;

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

fn check_exact<T: Real>(c: &Circuit, chunk_bits: usize, tol: f64) {
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
        "n={n}, c={chunk_bits}: max |Δamp| = {d:e} exceeds tol {tol:e} (passes={}, swaps={})",
        stats.file_passes,
        stats.swap_passes
    );
}

#[test]
fn exact_16_qubits_small_chunks() {
    let n = 16;
    let c = 12; // 2^12 = 4,096 amplitudes per chunk, 16 chunks
    let mut rng = StdRng::seed_from_u64(1612);

    let circ = random_universal(n, 30, &mut rng);
    check_exact::<f64>(&circ, c, 1e-12);
    check_exact::<f32>(&circ, c, 1e-6);

    let qft = algorithms::qft(n);
    check_exact::<f64>(&qft, c, 1e-12);
}

#[test]
fn exact_18_qubits_small_chunks() {
    let n = 18;
    let c = 14; // 2^14 = 16,384 amplitudes per chunk, 16 chunks
    let mut rng = StdRng::seed_from_u64(1814);

    let circ = random_universal(n, 35, &mut rng);
    check_exact::<f64>(&circ, c, 1e-12);
    check_exact::<f32>(&circ, c, 1e-6);
}

#[test]
fn exact_20_qubits_small_chunks() {
    let n = 20;
    let c = 16; // 2^16 = 65,536 amplitudes per chunk, 16 chunks
    let mut rng = StdRng::seed_from_u64(2016);

    let circ = random_universal(n, 30, &mut rng);
    check_exact::<f64>(&circ, c, 1e-12);
    check_exact::<f32>(&circ, c, 1e-6);
}

#[test]
fn exact_22_qubits_small_chunks() {
    let n = 22;
    let c = 18; // 2^18 = 262,144 amplitudes per chunk, 16 chunks
    let mut rng = StdRng::seed_from_u64(2218);

    let circ = random_universal(n, 25, &mut rng);
    check_exact::<f64>(&circ, c, 1e-12);
    check_exact::<f32>(&circ, c, 1e-6);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    #[test]
    fn proptest_ooc_exact_16_to_20_qubits(
        n in 16usize..=20,
        chunk_bits in 12usize..=14,
        gate_count in 15usize..=35,
        seed in 1u64..1000
    ) {
        let mut rng = StdRng::seed_from_u64(seed);
        let circ = random_universal(n, gate_count, &mut rng);
        check_exact::<f64>(&circ, chunk_bits, 1e-12);
        check_exact::<f32>(&circ, chunk_bits, 1e-6);
    }
}
