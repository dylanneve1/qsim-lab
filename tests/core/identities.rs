//! Gate identities, checked on random states (so they hold as operators,
//! not just on |0>).

#[path = "../common/mod.rs"]
mod common;

use common::*;
use qsim_lab::circuit::Circuit;
use qsim_lab::{Gate, StateVectorF64, Tableau};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// A random (non-stabilizer) 3-qubit state.
fn random_state(seed: u64) -> StateVectorF64 {
    let mut rng = StdRng::seed_from_u64(seed);
    sv_of(&random_universal(3, 30, &mut rng))
}

fn apply(s: &StateVectorF64, gates: &[Gate]) -> StateVectorF64 {
    let mut s = s.clone();
    for g in gates {
        s.apply_gate(g).unwrap();
    }
    s
}

/// Asserts the two sequences act identically (equal up to global phase).
fn same(a: &[Gate], b: &[Gate]) {
    for seed in 0..5 {
        let s = random_state(seed);
        let f = apply(&s, a).fidelity(&apply(&s, b));
        assert!((f - 1.0).abs() < 1e-10, "{a:?} != {b:?} (fidelity {f})");
    }
}

/// Asserts exact equality including global phase.
fn same_exact(a: &[Gate], b: &[Gate]) {
    let s = random_state(42);
    let (x, y) = (apply(&s, a), apply(&s, b));
    for i in 0..8 {
        assert!(
            (x.amplitude(i) - y.amplitude(i)).norm() < 1e-10,
            "{a:?} vs {b:?}"
        );
    }
}

use Gate::*;

#[test]
fn single_qubit_identities() {
    same_exact(&[H(0), H(0)], &[]);
    same_exact(&[S(1), S(1)], &[Z(1)]);
    same_exact(&[T(2), T(2)], &[S(2)]);
    same_exact(&[T(0), Tdg(0)], &[]);
    same_exact(&[S(0), Sdg(0)], &[]);
    same_exact(&[X(0), X(0)], &[]);
    same_exact(&[H(0), Z(0), H(0)], &[X(0)]);
    same_exact(&[H(0), X(0), H(0)], &[Z(0)]);
    same(&[X(1), Z(1)], &[Y(1)]); // XZ = -iY
    same(&[Rz(0, 0.7)], &[Phase(0, 0.7)]);
    same(&[Rx(2, std::f64::consts::PI)], &[X(2)]);
    same(&[Ry(2, std::f64::consts::PI)], &[Y(2)]);
    same(&[Rz(1, std::f64::consts::FRAC_PI_4)], &[T(1)]);
    same_exact(&[Rx(0, 0.3), Rx(0, 0.4)], &[Rx(0, 0.7)]);
}

#[test]
fn cnot_conjugation_rules() {
    // CNOT (X ⊗ I) CNOT = X ⊗ X   (control 0, target 1)
    same_exact(&[Cnot(0, 1), X(0), Cnot(0, 1)], &[X(0), X(1)]);
    // CNOT (I ⊗ X) CNOT = I ⊗ X
    same_exact(&[Cnot(0, 1), X(1), Cnot(0, 1)], &[X(1)]);
    // CNOT (Z ⊗ I) CNOT = Z ⊗ I
    same_exact(&[Cnot(0, 1), Z(0), Cnot(0, 1)], &[Z(0)]);
    // CNOT (I ⊗ Z) CNOT = Z ⊗ Z
    same_exact(&[Cnot(0, 1), Z(1), Cnot(0, 1)], &[Z(0), Z(1)]);
    // H⊗H CNOT H⊗H reverses control and target
    same_exact(&[H(0), H(2), Cnot(0, 2), H(0), H(2)], &[Cnot(2, 0)]);
    // CZ = (I⊗H) CNOT (I⊗H), symmetric
    same_exact(&[H(1), Cnot(0, 1), H(1)], &[Cz(0, 1)]);
    same_exact(&[Cz(0, 1)], &[Cz(1, 0)]);
    // SWAP = three CNOTs
    same_exact(&[Cnot(0, 2), Cnot(2, 0), Cnot(0, 2)], &[Swap(0, 2)]);
    same_exact(&[CPhase(0, 1, std::f64::consts::PI)], &[Cz(0, 1)]);
}

#[test]
fn decompositions_are_exact_up_to_phase() {
    for g in [
        Ccx(0, 1, 2),
        Ccx(2, 0, 1),
        CPhase(1, 2, 0.9),
        Rx(1, 1.3),
        Ry(0, -0.4),
    ] {
        same(&[g], &g.decompose_to_clifford_rz());
    }
}

#[test]
fn stabilizer_identities() {
    // The tableau must agree with these identities too (as stabilizer
    // groups, compared via outcome distributions).
    let check = |a: &[Gate], b: &[Gate]| {
        let mut pre = Circuit::new(3);
        pre.h(0).s(0).cnot(0, 1).h(2).cz(1, 2);
        let run = |gs: &[Gate]| {
            let mut t = Tableau::new(3);
            let mut c = pre.clone();
            for g in gs {
                c.gate(*g);
            }
            c.h(1); // rotate so phases become visible in the Z basis
            let mut rng = StdRng::seed_from_u64(0);
            c.run(&mut t, &mut rng).unwrap();
            (0..8).map(|i| t.probability(i)).collect::<Vec<_>>()
        };
        assert_eq!(run(a), run(b), "{a:?} vs {b:?}");
    };
    check(&[H(0), H(0)], &[]);
    check(&[S(1), S(1)], &[Z(1)]);
    check(&[Cnot(0, 1), X(0), Cnot(0, 1)], &[X(0), X(1)]);
    check(&[Cnot(0, 1), Z(1), Cnot(0, 1)], &[Z(0), Z(1)]);
    check(&[Sdg(2), S(2)], &[]);
}
