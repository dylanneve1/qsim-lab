//! The backends must agree with each other.

mod common;

use common::*;
use num_complex::Complex64;
use qsim_lab::circuit::{Circuit, Simulator};
use qsim_lab::engines::pauli_path::{self, PauliSum, DEFAULT_MAX_TERMS};
use qsim_lab::{Mps, StateVectorF32, StateVectorF64, Tableau};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[test]
fn stabilizer_matches_statevector_on_random_clifford_circuits() {
    let mut rng = StdRng::seed_from_u64(10);
    for trial in 0..60 {
        let n = 1 + trial % 8;
        let c = Circuit::random_clifford(n, 1 + trial % 12, &mut rng);
        let sv = sv_of(&c);
        let mut t = Tableau::new(n);
        c.run(&mut t, &mut rng).unwrap();
        let p_sv = sv.probabilities();
        let p_tab: Vec<f64> = (0..1 << n).map(|i| t.probability(i)).collect();
        assert!(
            max_abs_diff(&p_sv, &p_tab) < 1e-9,
            "trial {trial}: n={n}\n{p_sv:?}\n{p_tab:?}"
        );
    }
}

#[test]
fn stabilizer_signs_match_statevector_expectations() {
    // Every stabilizer generator P must satisfy <ψ|P|ψ> = +1 on the state
    // vector; checks the sign bits, not just the distribution.
    let mut rng = StdRng::seed_from_u64(11);
    for trial in 0..40 {
        let n = 1 + trial % 6;
        let c = Circuit::random_clifford(n, 6, &mut rng);
        let sv = sv_of(&c);
        let mut t = Tableau::new(n);
        c.run(&mut t, &mut rng).unwrap();
        for p in t.stabilizers() {
            let sign = if p.starts_with('-') { -1.0 } else { 1.0 };
            let ev = pauli_expectation_sv(&sv, &p[1..]);
            assert!(
                (ev - sign).abs() < 1e-9,
                "trial {trial}: {p} has <P> = {ev}"
            );
        }
    }
}

/// `<ψ|P|ψ>` for a Pauli string (qubit 0 first) by applying P to a copy.
fn pauli_expectation_sv(sv: &StateVectorF64, p: &str) -> f64 {
    use qsim_lab::Gate;
    let mut phi = sv.clone();
    for (q, ch) in p.chars().enumerate() {
        match ch {
            'X' => phi.apply_gate(&Gate::X(q)).unwrap(),
            'Y' => phi.apply_gate(&Gate::Y(q)).unwrap(),
            'Z' => phi.apply_gate(&Gate::Z(q)).unwrap(),
            _ => {}
        }
    }
    let v = sv.inner(&phi);
    assert!(v.im.abs() < 1e-9);
    v.re
}

#[test]
fn stabilizer_sampling_matches_statevector_distribution() {
    let mut rng = StdRng::seed_from_u64(12);
    for trial in 0..6 {
        let n = 4;
        let mut c = Circuit::random_clifford(n, 5, &mut rng);
        let p_sv = sv_of(&c).probabilities();
        c.measure_all();
        let shots = 4000;
        let mut counts = vec![0usize; 1 << n];
        for _ in 0..shots {
            let mut t = Tableau::new(n);
            let bits = c.run(&mut t, &mut rng).unwrap();
            let idx = bits
                .iter()
                .enumerate()
                .fold(0, |a, (q, &b)| a | (usize::from(b) << q));
            counts[idx] += 1;
        }
        for i in 0..1 << n {
            let f = counts[i] as f64 / shots as f64;
            // 5 sigma of a binomial with p <= 1/2 and 4000 shots is < 0.04
            assert!(
                (f - p_sv[i]).abs() < 0.04,
                "trial {trial} idx {i}: {f} vs {}",
                p_sv[i]
            );
            if p_sv[i] < 1e-12 {
                assert_eq!(counts[i], 0, "impossible outcome sampled");
            }
        }
    }
}

#[test]
fn mps_matches_statevector_on_random_universal_circuits() {
    let mut rng = StdRng::seed_from_u64(13);
    for trial in 0..30 {
        let n = 1 + trial % 7;
        let c = random_universal(n, 40, &mut rng);
        let sv = sv_of(&c);
        let mut m = Mps::new(n, 1 << n);
        c.run(&mut m, &mut rng).unwrap();
        assert!((m.norm_sqr() - 1.0).abs() < 1e-9);
        // full amplitude overlap, so phases are checked too (up to global phase)
        let overlap: Complex64 = (0..1usize << n)
            .map(|i| sv.amplitude(i).conj() * m.amplitude(i as u128))
            .sum();
        assert!(
            (overlap.norm() - 1.0).abs() < 1e-8,
            "trial {trial}: |<sv|mps>| = {}",
            overlap.norm()
        );
        assert!((m.fidelity_estimate() - 1.0).abs() < 1e-9);
    }
}

#[test]
fn mps_measurement_collapses_like_statevector() {
    let mut rng = StdRng::seed_from_u64(14);
    for _ in 0..20 {
        let n = 5;
        let c = random_universal(n, 30, &mut rng);
        let mut sv = sv_of(&c);
        let mut m = Mps::new(n, 32);
        c.run(&mut m, &mut rng).unwrap();
        for q in [2, 0, 4] {
            assert!((sv.prob_one(q) - m.prob_one(q)).abs() < 1e-9);
            let outcome = m.measure_qubit(q, &mut rng);
            if (if outcome {
                sv.prob_one(q)
            } else {
                1.0 - sv.prob_one(q)
            }) < 1e-12
            {
                panic!("MPS produced an impossible outcome");
            }
            sv.collapse(q, outcome);
            assert!(max_abs_diff(&sv.probabilities(), &mps_probabilities(&m)) < 1e-9);
        }
    }
}

#[test]
fn mps_sampling_matches_statevector_distribution() {
    let mut rng = StdRng::seed_from_u64(15);
    let n = 4;
    let c = random_universal(n, 30, &mut rng);
    let p = sv_of(&c).probabilities();
    let mut m = Mps::new(n, 16);
    c.run(&mut m, &mut rng).unwrap();
    let shots = 20_000;
    let mut counts = vec![0usize; 1 << n];
    for s in m.sample(shots, &mut rng) {
        counts[s as usize] += 1;
    }
    for i in 0..1 << n {
        assert!((counts[i] as f64 / shots as f64 - p[i]).abs() < 0.02);
    }
}

#[test]
fn pauli_paths_match_statevector() {
    let mut rng = StdRng::seed_from_u64(16);
    for trial in 0..40 {
        let n = 1 + trial % 6;
        let c = random_universal(n, 25, &mut rng);
        let sv = sv_of(&c);
        // random Pauli observable
        let p: String = (0..n)
            .map(|_| ['I', 'X', 'Y', 'Z'][rng.random_range(0..4)])
            .collect();
        let (ev, _) =
            pauli_path::expectation(&c, &PauliSum::from_str_single(&p), DEFAULT_MAX_TERMS).unwrap();
        let want = pauli_expectation_sv(&sv, &p);
        assert!(
            (ev - want).abs() < 1e-9,
            "trial {trial} {p}: {ev} vs {want}"
        );
    }
}

#[test]
fn pauli_path_marginals_match_statevector() {
    let mut rng = StdRng::seed_from_u64(17);
    for _ in 0..10 {
        let n = 5;
        let c = Circuit::random_clifford_t(n, 6, 0.2, &mut rng);
        let sv = sv_of(&c);
        let qs = [0, 3, 4];
        let marg = pauli_path::marginal_distribution(&c, &qs).unwrap();
        let mut want = vec![0.0; 8];
        for (i, p) in sv.probabilities().iter().enumerate() {
            let k = qs
                .iter()
                .enumerate()
                .fold(0, |a, (j, &q)| a | (((i >> q) & 1) << j));
            want[k] += p;
        }
        assert!(max_abs_diff(&marg, &want) < 1e-9);
    }
}

#[test]
fn single_and_double_precision_agree() {
    let mut rng = StdRng::seed_from_u64(18);
    let n = 15; // above the parallel threshold
    let c = random_universal(n, 200, &mut rng);
    let mut a = StateVectorF32::new(n);
    a.apply_circuit(&c).unwrap();
    let b = sv_of(&c);
    let max = (0..1 << n)
        .map(|i| (a.amplitude(i) - b.amplitude(i)).norm())
        .fold(0.0, f64::max);
    assert!(max < 1e-4, "max amplitude difference {max}");
    assert!((a.norm_sqr() - 1.0).abs() < 1e-4);
}

#[test]
fn all_backends_agree_on_ghz() {
    let mut rng = StdRng::seed_from_u64(19);
    let n = 12;
    let c = qsim_lab::algorithms::ghz(n);
    let sv = sv_of(&c);
    let mut t = Tableau::new(n);
    c.run(&mut t, &mut rng).unwrap();
    let mut m = Mps::new(n, 4);
    c.run(&mut m, &mut rng).unwrap();
    let all = (1usize << n) - 1;
    for i in [0, 1, 5, all - 1, all] {
        let p = sv.probabilities()[i];
        assert!((p - t.probability(i)).abs() < 1e-12);
        assert!((p - m.amplitude(i as u128).norm_sqr()).abs() < 1e-12);
    }
    assert_eq!(m.max_bond_dim(), 2);
    // Z0 Z(n-1) correlation is +1, single Z is 0
    let zz = PauliSum::z_product(n, &[0, n - 1]);
    assert!((pauli_path::expectation(&c, &zz, 10).unwrap().0 - 1.0).abs() < 1e-12);
    assert!(pauli_path::expectation_z(&c, 3).unwrap().abs() < 1e-12);
}

#[test]
fn backends_reject_what_they_cannot_do() {
    let mut t = Tableau::new(2);
    let mut c = Circuit::new(2);
    c.h(0).t(0);
    let mut rng = StdRng::seed_from_u64(0);
    assert!(c.run(&mut t, &mut rng).is_err());
    let mut s = StateVectorF64::new(2);
    assert!(s.apply(&qsim_lab::Gate::Cnot(0, 2)).is_err());
}
