//! Differential tests comparing out-of-core simulation against the in-RAM blocked executor.
//!
//! Requirement:
//! "It must be EXACT: identical amplitudes vs the in-RAM blocked executor at 16–22 qubits
//! with a small forced chunk size (proptests)."

#[path = "../common/mod.rs"]
mod common;

use common::random_universal;
use proptest::prelude::*;
use qsim_lab::algorithms;
use qsim_lab::circuit::Circuit;
use qsim_lab::engines::blocked::BlockConfig;
use qsim_lab::engines::ooc::{OocConfig, OocScheduler, OocStateVector};
use qsim_lab::engines::statevector::{Real, StateVector};
use qsim_lab::gate::Gate;
use rand::rngs::StdRng;
use rand::Rng;
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

/// Random circuit over EVERY gate type the simulator has (including `I`, `U`,
/// `Sx`, `Sxdg`, `ISwap`, `ISwapdg`, `Swap`, `CPhase`, `Ccx`).
fn random_all_gates(n: usize, len: usize, rng: &mut StdRng) -> Circuit {
    assert!(n >= 3);
    let mut c = Circuit::new(n);
    for _ in 0..len {
        let q = rng.random_range(0..n);
        let th = rng.random_range(-3.2..3.2);
        let other = |rng: &mut StdRng, not: &[usize]| loop {
            let x = rng.random_range(0..n);
            if !not.contains(&x) {
                break x;
            }
        };
        let g = match rng.random_range(0..23) {
            0 => Gate::I(q),
            1 => Gate::H(q),
            2 => Gate::X(q),
            3 => Gate::Y(q),
            4 => Gate::Z(q),
            5 => Gate::S(q),
            6 => Gate::Sdg(q),
            7 => Gate::T(q),
            8 => Gate::Tdg(q),
            9 => Gate::Sx(q),
            10 => Gate::Sxdg(q),
            11 => Gate::Rx(q, th),
            12 => Gate::Ry(q, th),
            13 => Gate::Rz(q, th),
            14 => Gate::Phase(q, th),
            15 => Gate::U(q, th, th * 0.37 + 0.1, -th * 0.61),
            16 => Gate::Cnot(q, other(rng, &[q])),
            17 => Gate::Cz(q, other(rng, &[q])),
            18 => Gate::Swap(q, other(rng, &[q])),
            19 => Gate::ISwap(q, other(rng, &[q])),
            20 => Gate::ISwapdg(q, other(rng, &[q])),
            21 => Gate::CPhase(q, other(rng, &[q]), th),
            _ => {
                let b = other(rng, &[q]);
                let t = other(rng, &[q, b]);
                Gate::Ccx(q, b, t)
            }
        };
        c.gate(g);
    }
    c
}

const SCHEDS: [OocScheduler; 2] = [OocScheduler::Swap, OocScheduler::Window];

#[derive(Clone, Copy)]
struct Knobs {
    chunk_bits: usize,
    scheduler: OocScheduler,
    group_bits: usize,
    overlap_io: bool,
    restore_order: bool,
}

fn check_exact_with<T: Real>(c: &Circuit, kn: Knobs, tol: f64) {
    let n = c.num_qubits;
    let bcfg = BlockConfig::default();

    // 1. In-RAM blocked reference
    let mut in_ram = StateVector::<T>::new(n);
    in_ram.apply_circuit_blocked(c, &bcfg).unwrap();

    // 2. Out-of-core state vector with forced small chunk size
    let mut ooc = OocStateVector::<T>::temp(
        n,
        kn.chunk_bits,
        OocConfig {
            chunk_bits: kn.chunk_bits,
            scratch_dir: None,
            block_config: bcfg,
            restore_order: kn.restore_order,
            scheduler: kn.scheduler,
            group_bits: kn.group_bits,
            overlap_io: kn.overlap_io,
        },
    )
    .unwrap();

    let stats = ooc.simulate_circuit(c).unwrap();
    let ooc_amps = ooc.read_amplitudes().unwrap();

    let d = max_diff(&ooc_amps, in_ram.amplitudes());
    assert!(
        d <= tol,
        "n={n}, c={}, k={}, sched={:?}: max |Δamp| = {d:e} exceeds tol {tol:e} (passes={}, swaps={})",
        kn.chunk_bits,
        kn.group_bits,
        kn.scheduler,
        stats.file_passes,
        stats.swap_passes
    );
    if kn.restore_order {
        assert!(ooc.layout().iter().enumerate().all(|(q, &p)| p == q));
    }
}

fn check_exact<T: Real>(c: &Circuit, chunk_bits: usize, tol: f64) {
    for scheduler in SCHEDS {
        check_exact_with::<T>(
            c,
            Knobs {
                chunk_bits,
                scheduler,
                group_bits: 3,
                overlap_io: true,
                restore_order: true,
            },
            tol,
        );
    }
}

#[test]
fn exact_16_qubits_small_chunks() {
    let n = 16;
    let c = 12; // 2^12 = 4,096 amplitudes per chunk, 16 chunks
    let mut rng = StdRng::seed_from_u64(1612);

    let circ = random_universal(n, 30, &mut rng);
    check_exact::<f64>(&circ, c, 1e-12);
    check_exact::<f32>(&circ, c, 1e-5);

    let qft = algorithms::qft(n);
    check_exact::<f64>(&qft, c, 1e-12);
}

#[test]
fn exact_18_qubits_small_chunks() {
    let n = 18;
    let c = 14;
    let mut rng = StdRng::seed_from_u64(1814);

    let circ = random_universal(n, 35, &mut rng);
    check_exact::<f64>(&circ, c, 1e-12);
    check_exact::<f32>(&circ, c, 1e-5);
}

#[test]
fn exact_20_qubits_small_chunks() {
    let n = 20;
    let c = 16;
    let mut rng = StdRng::seed_from_u64(2016);

    let circ = random_universal(n, 30, &mut rng);
    check_exact::<f64>(&circ, c, 1e-12);
    check_exact::<f32>(&circ, c, 1e-5);
}

#[test]
fn exact_22_qubits_small_chunks() {
    let n = 22;
    let c = 18;
    let mut rng = StdRng::seed_from_u64(2218);

    let circ = random_universal(n, 25, &mut rng);
    check_exact::<f64>(&circ, c, 1e-12);
    check_exact::<f32>(&circ, c, 1e-5);
}

/// Tiny chunks (almost every qubit is global): the hard case for the scheduler.
#[test]
fn exact_tiny_chunks_all_gate_types() {
    let mut rng = StdRng::seed_from_u64(77);
    for &(n, c) in &[(14usize, 4usize), (16, 5), (17, 6), (18, 7), (20, 8)] {
        let circ = random_all_gates(n, 60, &mut rng);
        check_exact::<f64>(&circ, c, 1e-12);
        check_exact::<f32>(&circ, c, 1e-5);
    }
}

#[test]
fn exact_structured_circuits_tiny_chunks() {
    let mut rng = StdRng::seed_from_u64(5);
    for &(n, c) in &[(14usize, 4usize), (16, 6), (18, 8)] {
        check_exact::<f64>(&algorithms::qft(n), c, 1e-12);
        check_exact::<f64>(&algorithms::ghz(n), c, 1e-12);
        check_exact::<f64>(&algorithms::random_brickwork(n, 4, &mut rng), c, 1e-12);
        check_exact::<f32>(&algorithms::qft(n), c, 1e-5);
    }
}

/// Window size sweep, I/O overlap on/off, and `restore_order=false`
/// (canonical order is reconstructed by `read_amplitudes`).
#[test]
fn exact_window_knobs() {
    let mut rng = StdRng::seed_from_u64(99);
    let circ = random_all_gates(15, 50, &mut rng);
    for &c in &[3usize, 5, 8] {
        for &k in &[3usize, 4, 5, 12] {
            for &overlap_io in &[false, true] {
                for &restore_order in &[true, false] {
                    check_exact_with::<f64>(
                        &circ,
                        Knobs {
                            chunk_bits: c,
                            scheduler: OocScheduler::Window,
                            group_bits: k,
                            overlap_io,
                            restore_order,
                        },
                        1e-12,
                    );
                }
            }
        }
    }
}

/// n == c (one chunk), n == c + 1, and a window covering all high qubits.
#[test]
fn exact_degenerate_shapes() {
    let mut rng = StdRng::seed_from_u64(31);
    for &(n, c) in &[(6usize, 6usize), (7, 6), (8, 3), (9, 8)] {
        let circ = random_all_gates(n, 40, &mut rng);
        check_exact::<f64>(&circ, c, 1e-12);
    }
}

/// A circuit is simulated in two halves on the same state (layout persists
/// across calls when order is not restored).
#[test]
fn exact_two_calls_keep_layout() {
    let mut rng = StdRng::seed_from_u64(1234);
    let n = 14;
    let a = random_all_gates(n, 30, &mut rng);
    let b = random_all_gates(n, 30, &mut rng);
    let mut full = a.clone();
    full.append(&b);
    let mut ref_sv = StateVector::<f64>::new(n);
    ref_sv
        .apply_circuit_blocked(&full, &BlockConfig::default())
        .unwrap();
    let mut ooc = OocStateVector::<f64>::temp(
        n,
        5,
        OocConfig {
            chunk_bits: 5,
            restore_order: false,
            ..OocConfig::default()
        },
    )
    .unwrap();
    ooc.simulate_circuit(&a).unwrap();
    ooc.simulate_circuit(&b).unwrap();
    let d = max_diff(&ooc.read_amplitudes().unwrap(), ref_sv.amplitudes());
    assert!(d <= 1e-12, "two-call layout drift {d:e}");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn proptest_ooc_exact_16_to_20_qubits(
        n in 14usize..=20,
        chunk_bits in 4usize..=9,
        group_bits in 3usize..=5,
        gate_count in 15usize..=60,
        overlap_io in any::<bool>(),
        restore_order in any::<bool>(),
        seed in 1u64..100_000
    ) {
        let mut rng = StdRng::seed_from_u64(seed);
        let circ = random_all_gates(n, gate_count, &mut rng);
        for scheduler in SCHEDS {
            // The swap scheduler always restores order.
            let kn = Knobs { chunk_bits, scheduler, group_bits, overlap_io,
                restore_order: restore_order || scheduler == OocScheduler::Swap };
            check_exact_with::<f64>(&circ, kn, 1e-12);
            check_exact_with::<f32>(&circ, kn, 1e-5);
        }
    }
}
