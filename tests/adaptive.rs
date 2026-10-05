//! The adaptive simulator (tableau -> rotation frame -> compressed state
//! vector) must reproduce the state vector exactly: amplitudes (up to a
//! global phase), expectation values for every switching strategy, and the
//! full outcome distribution of its sampler.

mod common;

use common::*;
use proptest::prelude::*;
use qsim_lab::bench::skeleton_stabilizer;
use qsim_lab::circuit::Circuit;
use qsim_lab::engines::adaptive::{self, AdaptiveOptions, CompressedState, Strategy};
use qsim_lab::engines::pauli_path::{self, PauliSum, DEFAULT_MAX_TERMS};
use qsim_lab::gate::Gate;
use qsim_lab::StateVectorF64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TOL: f64 = 1e-10;

/// Rounds of (random Clifford block, T/T†/Rz on a random qubit).
fn clifford_t_rounds<R: Rng>(n: usize, depth: usize, t: usize, rng: &mut R) -> Circuit {
    let mut c = Circuit::new(n);
    for _ in 0..t {
        c.append(&Circuit::random_clifford(n, depth, rng));
        let q = rng.random_range(0..n);
        match rng.random_range(0..4) {
            0 => c.gate(Gate::Tdg(q)),
            1 => c.gate(Gate::Rz(q, rng.random_range(-3.0..3.0))),
            _ => c.t(q),
        };
    }
    c.append(&Circuit::random_clifford(n, depth, rng));
    c
}

fn random_pauli<R: Rng>(n: usize, rng: &mut R) -> String {
    (0..n)
        .map(|_| ['I', 'X', 'Y', 'Z'][rng.random_range(0..4)])
        .collect()
}

fn sv_pauli(sv: &StateVectorF64, p: &str) -> f64 {
    let mut phi = sv.clone();
    for (q, ch) in p.chars().enumerate() {
        match ch {
            'X' => phi.apply_gate(&Gate::X(q)).unwrap(),
            'Y' => phi.apply_gate(&Gate::Y(q)).unwrap(),
            'Z' => phi.apply_gate(&Gate::Z(q)).unwrap(),
            _ => {}
        }
    }
    sv.inner(&phi).re
}

/// Max |a - e^{iφ} b| after aligning the global phase on the largest
/// amplitude of `b`.
fn phase_aligned_diff(a: &StateVectorF64, b: &StateVectorF64) -> f64 {
    let (aa, bb) = (a.amplitudes(), b.amplitudes());
    let i = (0..bb.len())
        .max_by(|&i, &j| bb[i].norm().partial_cmp(&bb[j].norm()).unwrap())
        .unwrap();
    let ph = aa[i] / bb[i];
    let ph = ph / ph.norm();
    aa.iter()
        .zip(bb)
        .map(|(x, y)| (x - y * ph).norm())
        .fold(0.0, f64::max)
}

fn check_state(c: &Circuit, ctx: &str) -> CompressedState {
    let sv = sv_of(c);
    let cs = CompressedState::new(c, 20).unwrap();
    let full = cs.to_statevector();
    let diff = phase_aligned_diff(&full, &sv);
    assert!(diff < TOL, "{ctx}: amplitudes differ by {diff}");
    cs
}

fn check_distribution(cs: CompressedState, sv: &StateVectorF64, ctx: &str) {
    let want = sv.probabilities();
    let s = cs.sampler();
    let got = s.distribution();
    let diff = max_abs_diff(&got, &want);
    assert!(diff < TOL, "{ctx}: distribution differs by {diff}");
}

#[test]
fn compressed_state_matches_statevector() {
    let mut rng = StdRng::seed_from_u64(7);
    let mut max_d = 0;
    for trial in 0..60 {
        let n = 1 + trial % 10;
        let t = (trial * 7) % 25;
        let c = clifford_t_rounds(n, 2, t, &mut rng);
        let cs = check_state(&c, &format!("trial {trial} n={n} t={t}"));
        assert!(cs.active_qubits() <= n.min(t));
        max_d = max_d.max(cs.active_qubits());
    }
    assert!(max_d >= 8);
}

#[test]
fn compressed_state_matches_statevector_on_universal_circuits() {
    // Rx, Ry, U, CPhase, Toffoli, iSWAP ... all go through the Clifford+Rz
    // decomposition.
    let mut rng = StdRng::seed_from_u64(8);
    for trial in 0..40 {
        let n = 1 + trial % 8;
        let c = random_universal(n, 25, &mut rng);
        let cs = check_state(&c, &format!("universal trial {trial}"));
        let sv = sv_of(&c);
        for _ in 0..3 {
            let p = random_pauli(n, &mut rng);
            let v = cs.expectation(&PauliSum::from_str_single(&p));
            let want = sv_pauli(&sv, &p);
            assert!((v - want).abs() < TOL, "trial {trial} {p}: {v} vs {want}");
        }
        check_distribution(cs, &sv, &format!("universal trial {trial}"));
    }
}

#[test]
fn sampler_distribution_is_exact() {
    let mut rng = StdRng::seed_from_u64(9);
    let mut saw_random = false;
    let mut saw_dense = false;
    for trial in 0..60 {
        let n = 1 + trial % 12;
        let t = (trial * 5) % 20;
        let c = clifford_t_rounds(n, 2, t, &mut rng);
        let sv = sv_of(&c);
        let cs = CompressedState::new(&c, 20).unwrap();
        let s = cs.sampler();
        saw_random |= s.stats.random_bits > 0;
        saw_dense |= s.stats.dense_bits > 2;
        let diff = max_abs_diff(&s.distribution(), &sv.probabilities());
        assert!(diff < TOL, "trial {trial} n={n} t={t}: {diff}");
    }
    assert!(saw_random && saw_dense);
}

#[test]
fn sampler_handles_pure_clifford_and_trivial_circuits() {
    let mut rng = StdRng::seed_from_u64(10);
    for n in 1..10 {
        let c = Circuit::random_clifford(n, 6, &mut rng);
        let sv = sv_of(&c);
        let cs = CompressedState::new(&c, 20).unwrap();
        assert_eq!(cs.active_qubits(), 0);
        check_distribution(cs, &sv, &format!("clifford n={n}"));
    }
    // T on |0> is a phase: nothing becomes active.
    let mut c = Circuit::new(3);
    c.t(0).t(1).h(2);
    let cs = CompressedState::new(&c, 20).unwrap();
    assert_eq!(cs.active_qubits(), 0);
    check_distribution(cs, &sv_of(&c), "t on zero");
}

#[test]
fn sampler_shots_follow_the_distribution() {
    // Statistical check of the shot path itself (not only `distribution`):
    // total variation distance of 200k shots against the exact distribution.
    let mut rng = StdRng::seed_from_u64(11);
    for trial in 0..4 {
        let n = 6;
        let c = clifford_t_rounds(n, 2, 4 + 2 * trial, &mut rng);
        let want = sv_of(&c).probabilities();
        let s = CompressedState::new(&c, 20).unwrap().sampler();
        let shots = 200_000;
        let mut hist = vec![0.0; 1 << n];
        for b in s.sample_indices(shots, &mut rng) {
            hist[b as usize] += 1.0 / shots as f64;
        }
        let tvd: f64 = hist
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / 2.0;
        // E[TVD] ~ sqrt(2^n / shots) / 2 ≈ 0.009 here.
        assert!(tvd < 0.03, "trial {trial}: tvd {tvd}");
    }
}

fn strategies(m: usize) -> Vec<Strategy> {
    let mut v = vec![Strategy::Frame, Strategy::Dense, Strategy::Auto];
    for k in 0..=m {
        v.push(Strategy::SwitchAt(k));
    }
    v
}

#[test]
fn adaptive_expectation_matches_statevector_for_every_switch_point() {
    let mut rng = StdRng::seed_from_u64(12);
    let mut switched = 0;
    let mut nonzero = 0;
    for trial in 0..30 {
        let n = 2 + trial % 9;
        let t = 3 + (trial * 7) % 22;
        let c = clifford_t_rounds(n, 2, t, &mut rng);
        let sv = sv_of(&c);
        let mut obs: Vec<(PauliSum, f64)> = Vec::new();
        let zs: Vec<usize> = (0..n).filter(|_| rng.random_bool(0.5)).collect();
        let st = skeleton_stabilizer(&c, &zs);
        // Reference: the legacy Pauli-path engine (itself cross-checked
        // against the state vector in tests/pauli_frame.rs).
        let (want, _) = pauli_path::expectation_legacy(&c, &st, DEFAULT_MAX_TERMS).unwrap();
        obs.push((st, want));
        for _ in 0..2 {
            let p = random_pauli(n, &mut rng);
            let w = sv_pauli(&sv, &p);
            obs.push((PauliSum::from_str_single(&p), w));
        }
        for (o, w) in &obs {
            if w.abs() > 1e-6 {
                nonzero += 1;
            }
            for strat in strategies(t + 1) {
                let opt = AdaptiveOptions {
                    strategy: strat,
                    ..Default::default()
                };
                let r = adaptive::expectation(&c, o, &opt).unwrap();
                switched += r.switched_at.is_some() as usize;
                assert!(
                    (r.value - w).abs() < TOL,
                    "trial {trial} {strat:?}: {} vs {w} ({r:?})",
                    r.value
                );
            }
        }
    }
    assert!(
        switched > 100 && nonzero > 20,
        "switched {switched} nonzero {nonzero}"
    );
}

#[test]
fn stabilizer_observable_through_compressed_state_matches_frame() {
    let mut rng = StdRng::seed_from_u64(13);
    for trial in 0..20 {
        let n = 3 + trial % 10;
        let c = clifford_t_rounds(n, 2, 2 + trial, &mut rng);
        let zs: Vec<usize> = (0..n).filter(|_| rng.random_bool(0.5)).collect();
        let st = skeleton_stabilizer(&c, &zs);
        let (fv, _) = pauli_path::expectation_legacy(&c, &st, DEFAULT_MAX_TERMS).unwrap();
        let cs = CompressedState::new(&c, 20).unwrap();
        let v = cs.expectation(&st);
        assert!((v - fv).abs() < TOL, "trial {trial}: {v} vs {fv}");
    }
}

#[test]
fn adaptive_matches_frame_at_64_qubits() {
    // Beyond the state vector: every strategy against the frame engine.
    let mut rng = StdRng::seed_from_u64(14);
    let mut nonzero = 0;
    for &t in &[8usize, 16, 20, 24] {
        let c = clifford_t_rounds(64, 2, t, &mut rng);
        let zs: Vec<usize> = (0..64).filter(|_| rng.random_bool(0.3)).collect();
        let st = skeleton_stabilizer(&c, &zs);
        let (fv, _) = pauli_path::expectation(&c, &st, DEFAULT_MAX_TERMS).unwrap();
        nonzero += (fv.abs() > 1e-9) as usize;
        for strat in [
            Strategy::Dense,
            Strategy::Auto,
            Strategy::SwitchAt(t / 2),
            Strategy::Frame,
        ] {
            let opt = AdaptiveOptions {
                strategy: strat,
                max_dense_qubits: 20,
                ..Default::default()
            };
            let r = adaptive::expectation(&c, &st, &opt).unwrap();
            assert!(
                (r.value - fv).abs() < 1e-12,
                "t={t} {strat:?}: {} vs {fv}",
                r.value
            );
        }
        let cs = CompressedState::new(&c, 24).unwrap();
        assert!((cs.expectation(&st) - fv).abs() < 1e-12);
    }
    assert!(nonzero >= 3);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn prop_state_and_distribution_exact(seed in any::<u64>(), n in 1usize..9, len in 1usize..40) {
        let mut rng = StdRng::seed_from_u64(seed);
        let c = random_universal(n, len, &mut rng);
        let sv = sv_of(&c);
        let cs = CompressedState::new(&c, 20).unwrap();
        let diff = phase_aligned_diff(&cs.to_statevector(), &sv);
        prop_assert!(diff < TOL, "amplitudes {}", diff);
        let p = random_pauli(n, &mut rng);
        let want = sv_pauli(&sv, &p);
        let o = PauliSum::from_str_single(&p);
        prop_assert!((cs.expectation(&o) - want).abs() < TOL);
        let k = rng.random_range(0..=c.num_gates());
        for strat in [Strategy::Auto, Strategy::Dense, Strategy::SwitchAt(k)] {
            let opt = AdaptiveOptions { strategy: strat, ..Default::default() };
            let v = adaptive::expectation(&c, &o, &opt).unwrap().value;
            prop_assert!((v - want).abs() < TOL, "{:?}: {} vs {}", strat, v, want);
        }
        let got = cs.sampler().distribution();
        prop_assert!(max_abs_diff(&got, &sv.probabilities()) < TOL);
    }
}
