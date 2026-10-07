//! Chain-sweep amplitudes against the dense state vector: random brickwork
//! CZ circuits, the IBM doped-Clifford circuit truncated to small sizes, and
//! bond slicing (slice sum = full amplitude, each slice = a state vector of
//! the circuit with that CZ replaced by `P_k ⊗ Z^k`).

use num_complex::Complex64;
use qsim_lab::circuit::Circuit;
use qsim_lab::engines::chain_sweep::{self, compile, sliced_ops, truncate, ChainCircuit};
use qsim_lab::engines::statevector::StateVector;
use qsim_lab::gate::Gate;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

const QASM: &str = include_str!("../../research/chain-sweep/nq70_depth70_checks27_doped.qasm");

fn brickwork(n: usize, d: usize, rng: &mut StdRng) -> Circuit {
    let mut c = Circuit::new(n);
    for layer in 0..d {
        for q in 0..n {
            for _ in 0..rng.random_range(0..3) {
                c.gate(match rng.random_range(0..7) {
                    0 => Gate::H(q),
                    1 => Gate::S(q),
                    2 => Gate::Sx(q),
                    3 => Gate::Sxdg(q),
                    4 => Gate::T(q),
                    5 => Gate::Rz(q, rng.random_range(-3.0..3.0)),
                    _ => Gate::Ry(q, rng.random_range(-3.0..3.0)),
                });
            }
        }
        let mut a = layer % 2;
        while a + 1 < n {
            // skip a few CZs so the layers are irregular
            if rng.random_bool(0.85) {
                c.gate(Gate::Cz(a, a + 1));
            }
            a += 2;
        }
    }
    c
}

fn sv(c: &Circuit) -> StateVector<f64> {
    let mut s = StateVector::<f64>::new(c.num_qubits);
    s.apply_circuit(c).unwrap();
    s
}

fn close(a: Complex64, b: Complex64, scale: f64, tol: f64) -> bool {
    (a - b).norm() <= tol * scale
}

#[test]
fn random_brickwork_all_amplitudes() {
    let mut rng = StdRng::seed_from_u64(7);
    for (n, d) in [(2, 3), (3, 5), (5, 8), (6, 11), (7, 12), (8, 9)] {
        let c = brickwork(n, d, &mut rng);
        let s = sv(&c);
        let cc = ChainCircuit::from_circuit(&c).unwrap();
        let scale = (1.0 / (1u64 << n) as f64).sqrt();
        for x in 0..1u128 << n {
            let a = chain_sweep::amplitude(&cc, x).unwrap();
            let e = s.amplitude(x as usize);
            assert!(close(a, e, scale, 1e-10), "n={n} d={d} x={x}: {a} vs {e}");
        }
    }
}

#[test]
fn register_width_is_half_the_depth() {
    let c = Circuit::from_qasm(QASM).unwrap();
    for d in [10, 21, 40] {
        let t = truncate(&c, 30, d);
        let cc = ChainCircuit::from_circuit(&t).unwrap();
        let plan = compile(&cc, 0, &HashMap::new());
        assert!(
            plan.width <= d.div_ceil(2) + 1,
            "d={d} width={}",
            plan.width
        );
    }
}

#[test]
fn doped_circuit_truncated_matches_state_vector() {
    let c = Circuit::from_qasm(QASM).unwrap();
    let mut rng = StdRng::seed_from_u64(3);
    for (n, d) in [(12, 14), (14, 24)] {
        let t = truncate(&c, n, d);
        let s = sv(&t);
        let cc = ChainCircuit::from_circuit(&t).unwrap();
        let scale = (1.0 / (1u64 << n) as f64).sqrt();
        for _ in 0..6 {
            let x = rng.random_range(0..1u128 << n);
            let a = chain_sweep::amplitude(&cc, x).unwrap();
            let e = s.amplitude(x as usize);
            assert!(close(a, e, scale, 1e-10), "n={n} d={d} x={x}: {a} vs {e}");
        }
    }
}

#[test]
fn slices_sum_to_amplitude_and_match_state_vector() {
    let c = Circuit::from_qasm(QASM).unwrap();
    let (n, d) = (10, 12);
    let t = truncate(&c, n, d);
    let cc = ChainCircuit::from_circuit(&t).unwrap();
    // three bonds on the middle edge plus one elsewhere
    let mut bonds: Vec<usize> = cc.edge_bonds(n / 2 - 1).into_iter().take(3).collect();
    bonds.push(cc.edge_bonds(1)[0]);
    let x = 0b1011001110u128;
    let full = chain_sweep::amplitude(&cc, x).unwrap();
    let parts =
        chain_sweep::slice_amplitudes_cpu::<f64>(&cc, x, &bonds, &Default::default()).unwrap();
    let sum: Complex64 = parts.iter().sum();
    let scale = (1.0 / (1u64 << n) as f64).sqrt();
    assert!(close(sum, full, scale, 1e-10), "{sum} vs {full}");
    for (s, part) in parts.iter().enumerate() {
        let mut st = StateVector::<f64>::new(n);
        for (q, m, cz) in sliced_ops(&t, &bonds, s) {
            match cz {
                Some(r) => st.apply_gate(&Gate::Cz(q, r)).unwrap(),
                None => st.apply_1q_matrix(q, &m),
            }
        }
        let e = st.amplitude(x as usize);
        assert!(close(*part, e, scale, 1e-10), "slice {s}: {part} vs {e}");
    }
}

#[test]
fn f32_agrees_with_f64() {
    let c = Circuit::from_qasm(QASM).unwrap();
    let t = truncate(&c, 16, 20);
    let cc = ChainCircuit::from_circuit(&t).unwrap();
    let plan = compile(&cc, 12345, &HashMap::new());
    let a64 = chain_sweep::amplitude_cpu::<f64>(&plan, &Default::default()).unwrap();
    let a32 = chain_sweep::amplitude_cpu::<f32>(&plan, &Default::default()).unwrap();
    assert!((a64 - a32).norm() <= 1e-5 / 256.0, "{a64} vs {a32}");
}

#[test]
fn meet_in_the_middle_matches_sweep_and_slices() {
    let c = Circuit::from_qasm(QASM).unwrap();
    let (n, d) = (11, 13);
    let t = truncate(&c, n, d);
    let cc = ChainCircuit::from_circuit(&t).unwrap();
    let scale = (1.0 / (1u64 << n) as f64).sqrt();
    for (x, e) in [
        (0b10110011101u128, 5),
        (0b00111000101, 2),
        (0b11111000000, 8),
    ] {
        let full = chain_sweep::amplitude(&cc, x).unwrap();
        let ct = chain_sweep::cut_tensors_cpu::<f64>(&cc, x, e, &Default::default()).unwrap();
        assert_eq!(ct.bonds, cc.edge_bonds(e));
        let a = ct.amplitude();
        assert!(close(a, full, scale, 1e-10), "e={e}: {a} vs {full}");
        let pos = [0, 2, 3];
        let sums = ct.slice_sums(&pos);
        let bonds: Vec<usize> = pos.iter().map(|&p| ct.bonds[p]).collect();
        let parts =
            chain_sweep::slice_amplitudes_cpu::<f64>(&cc, x, &bonds, &Default::default()).unwrap();
        for (s, (p, q)) in sums.iter().zip(&parts).enumerate() {
            assert!(close(*p, *q, scale, 1e-10), "slice {s}: {p} vs {q}");
        }
    }
}
