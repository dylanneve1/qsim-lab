//! Property-based tests (proptest generates the circuits).

mod common;

use common::*;
use proptest::prelude::*;
use qsim_lab::circuit::Circuit;
use qsim_lab::pauli_path::{self, PauliSum, DEFAULT_MAX_TERMS};
use qsim_lab::{Gate, Mps, StateVectorF64, Tableau};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn gate_strategy(n: usize, clifford_only: bool) -> impl Strategy<Value = Gate> {
    let q = 0..n;
    let pair = (0..n, 0..n - 1).prop_map(|(a, b)| (a, if b >= a { b + 1 } else { b }));
    let angle = -3.2f64..3.2;
    let one = prop_oneof![
        q.clone().prop_map(Gate::H),
        q.clone().prop_map(Gate::S),
        q.clone().prop_map(Gate::Sdg),
        q.clone().prop_map(Gate::X),
        q.clone().prop_map(Gate::Y),
        q.clone().prop_map(Gate::Z),
        pair.clone().prop_map(|(a, b)| Gate::Cnot(a, b)),
        pair.clone().prop_map(|(a, b)| Gate::Cz(a, b)),
        pair.clone().prop_map(|(a, b)| Gate::Swap(a, b)),
    ];
    if clifford_only {
        one.boxed()
    } else {
        prop_oneof![
            4 => one,
            1 => q.clone().prop_map(Gate::T),
            1 => (q.clone(), angle.clone()).prop_map(|(q, t)| Gate::Rx(q, t)),
            1 => (q.clone(), angle.clone()).prop_map(|(q, t)| Gate::Ry(q, t)),
            1 => (q, angle.clone()).prop_map(|(q, t)| Gate::Rz(q, t)),
            1 => (pair, angle).prop_map(|((a, b), t)| Gate::CPhase(a, b, t)),
        ]
        .boxed()
    }
}

fn circuit_strategy(clifford_only: bool) -> impl Strategy<Value = Circuit> {
    (2usize..=6).prop_flat_map(move |n| {
        prop::collection::vec(gate_strategy(n, clifford_only), 0..40).prop_map(move |gs| {
            let mut c = Circuit::new(n);
            for g in gs {
                c.gate(g);
            }
            c
        })
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn unitarity_preserves_norm(c in circuit_strategy(false)) {
        let s = sv_of(&c);
        prop_assert!((s.norm_sqr() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn circuit_then_inverse_is_identity(c in circuit_strategy(false)) {
        let mut s = sv_of(&c);
        s.apply_circuit(&c.inverse()).unwrap();
        prop_assert!((s.amplitude(0).norm() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn tableau_distribution_equals_statevector(c in circuit_strategy(true)) {
        let n = c.num_qubits;
        let p = sv_of(&c).probabilities();
        let mut t = Tableau::new(n);
        let mut rng = StdRng::seed_from_u64(0);
        c.run(&mut t, &mut rng).unwrap();
        for (i, &pi) in p.iter().enumerate() {
            prop_assert!((t.probability(i) - pi).abs() < 1e-9);
        }
    }

    #[test]
    fn stabilizer_probabilities_are_dyadic(c in circuit_strategy(true)) {
        // Gottesman–Knill: outcomes are uniform over an affine subspace.
        let p = sv_of(&c).probabilities();
        let nz: Vec<f64> = p.iter().copied().filter(|&x| x > 1e-12).collect();
        prop_assert!(nz.len().is_power_of_two());
        for x in &nz {
            prop_assert!((x - 1.0 / nz.len() as f64).abs() < 1e-9);
        }
    }

    #[test]
    fn mps_without_truncation_is_exact(c in circuit_strategy(false)) {
        let n = c.num_qubits;
        let p = sv_of(&c).probabilities();
        let mut m = Mps::new(n, 64);
        let mut rng = StdRng::seed_from_u64(0);
        c.run(&mut m, &mut rng).unwrap();
        prop_assert!(max_abs_diff(&p, &mps_probabilities(&m)) < 1e-9);
        prop_assert!(m.max_bond_dim() <= 1 << (n / 2));
    }

    #[test]
    fn pauli_path_z_expectations(c in circuit_strategy(false), q in 0usize..6) {
        let n = c.num_qubits;
        let q = q % n;
        let s = sv_of(&c);
        let (ev, _) = pauli_path::expectation(&c, &PauliSum::z_product(n, &[q]), DEFAULT_MAX_TERMS).unwrap();
        prop_assert!((ev - s.expectation_z(q)).abs() < 1e-9);
    }

    #[test]
    fn measuring_twice_gives_same_result(c in circuit_strategy(true), seed in 0u64..1000) {
        let n = c.num_qubits;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut t = Tableau::new(n);
        c.run(&mut t, &mut rng).unwrap();
        let mut s = sv_of(&c);
        for q in 0..n {
            let a = t.measure_qubit(q, &mut rng);
            prop_assert_eq!(t.measure_qubit(q, &mut rng), a);
            let possible = if a { s.prob_one(q) > 1e-9 } else { s.prob_one(q) < 1.0 - 1e-9 };
            prop_assert!(possible);
            s.collapse(q, a);
        }
    }

    #[test]
    fn toffoli_truth_table(input in 0usize..8) {
        let mut s = StateVectorF64::basis_state(3, input);
        s.apply_gate(&Gate::Ccx(0, 1, 2)).unwrap();
        let want = if input & 3 == 3 { input ^ 4 } else { input };
        prop_assert!((s.amplitude(want).norm() - 1.0).abs() < 1e-12);
    }
}
