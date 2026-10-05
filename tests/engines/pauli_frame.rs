//! The rotation-frame Pauli-path engine must agree with the legacy engine
//! and with the state vector, for every combination of its exact
//! optimisations.

#[path = "../common/mod.rs"]
mod common;

use common::*;
use qsim_lab::circuit::Circuit;
use qsim_lab::engines::pauli_path::{self, FrameOptions, PauliSum, DEFAULT_MAX_TERMS};
use qsim_lab::gate::Gate;
use qsim_lab::StateVectorF64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn all_options() -> Vec<FrameOptions> {
    let mut v = Vec::new();
    for prune in [false, true] {
        for merge_rotations in [false, true] {
            for parallel in [false, true] {
                for fuse in [false, true] {
                    v.push(FrameOptions {
                        prune,
                        merge_rotations,
                        parallel,
                        fuse,
                        ..FrameOptions::default()
                    });
                }
            }
        }
    }
    v
}

fn random_pauli<R: Rng>(n: usize, rng: &mut R) -> String {
    (0..n)
        .map(|_| ['I', 'X', 'Y', 'Z'][rng.random_range(0..4)])
        .collect()
}

/// `<ψ|P|ψ>` by applying P to a copy of the state.
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

/// The benchmark's circuit family: rounds of (random Clifford block, T on a
/// random qubit), then a final Clifford block.
fn clifford_t_rounds<R: Rng>(n: usize, depth: usize, t: usize, rng: &mut R) -> Circuit {
    let mut c = Circuit::new(n);
    for _ in 0..t {
        c.append(&Circuit::random_clifford(n, depth, rng));
        let q = rng.random_range(0..n);
        if rng.random_bool(0.5) {
            c.t(q);
        } else {
            c.gate(Gate::Tdg(q));
        }
    }
    c.append(&Circuit::random_clifford(n, depth, rng));
    c
}

fn check(c: &Circuit, p: &str, want: f64, ctx: &str) {
    let obs = PauliSum::from_str_single(p);
    let (legacy, _) = pauli_path::expectation_legacy(c, &obs, DEFAULT_MAX_TERMS).unwrap();
    assert!(
        (legacy - want).abs() < 1e-9,
        "{ctx} {p}: legacy {legacy} vs {want}"
    );
    for opt in all_options() {
        let (v, _) = pauli_path::expectation_with(c, &obs, &opt).unwrap();
        assert!(
            (v - legacy).abs() < 1e-12 && (v - want).abs() < 1e-9,
            "{ctx} {p} {opt:?}: frame {v} vs legacy {legacy} vs sv {want}"
        );
    }
}

#[test]
fn frame_matches_statevector_on_universal_circuits() {
    let mut rng = StdRng::seed_from_u64(100);
    for trial in 0..60 {
        let n = 1 + trial % 7;
        let c = random_universal(n, 30, &mut rng);
        let sv = sv_of(&c);
        for _ in 0..3 {
            let p = random_pauli(n, &mut rng);
            check(&c, &p, sv_pauli(&sv, &p), &format!("trial {trial}"));
        }
    }
}

#[test]
fn frame_matches_statevector_on_clifford_t_rounds() {
    // Many T gates on few qubits: the span saturates early, so every stage
    // of the pruning and projection logic is exercised, and the values are
    // generically non-zero.
    let mut rng = StdRng::seed_from_u64(101);
    let mut nonzero = 0;
    for trial in 0..24 {
        let n = 4 + trial % 7;
        let t = 6 + (trial * 5) % 24;
        let c = clifford_t_rounds(n, 2, t, &mut rng);
        let sv = sv_of(&c);
        for k in 0..4 {
            let p = if k == 0 {
                let mut s = vec!['I'; n];
                s[0] = 'Z';
                s.into_iter().collect()
            } else {
                random_pauli(n, &mut rng)
            };
            let want = sv_pauli(&sv, &p);
            if want.abs() > 1e-6 {
                nonzero += 1;
            }
            check(&c, &p, want, &format!("trial {trial} n={n} t={t}"));
        }
    }
    assert!(nonzero > 20, "test circuits should have non-zero values");
}

#[test]
fn frame_matches_statevector_on_toffoli_circuits() {
    // Toffoli networks produce many rotations about equal axes, which the
    // merging step combines (often into Clifford angles).
    let mut rng = StdRng::seed_from_u64(102);
    for trial in 0..20 {
        let n = 3 + trial % 5;
        let mut c = Circuit::new(n);
        for q in 0..n {
            if rng.random_bool(0.5) {
                c.h(q);
            }
        }
        for _ in 0..8 {
            let a = rng.random_range(0..n);
            let b = (a + 1 + rng.random_range(0..n - 1)) % n;
            let mut t = rng.random_range(0..n);
            while t == a || t == b {
                t = rng.random_range(0..n);
            }
            c.gate(Gate::Ccx(a, b, t));
            if rng.random_bool(0.3) {
                c.h(rng.random_range(0..n));
            }
        }
        let sv = sv_of(&c);
        for _ in 0..4 {
            let p = random_pauli(n, &mut rng);
            check(&c, &p, sv_pauli(&sv, &p), &format!("toffoli trial {trial}"));
        }
    }
}

#[test]
fn rotation_merging_reduces_toffoli_t_count_work() {
    // CCX · CCX = I: with merging, every rotation cancels.
    let mut c = Circuit::new(3);
    c.h(0).h(1);
    c.gate(Gate::Ccx(0, 1, 2)).gate(Gate::Ccx(0, 1, 2));
    let obs = PauliSum::z_product(3, &[2]);
    let opt = FrameOptions {
        merge_rotations: true,
        ..FrameOptions::default()
    };
    let (v, st) = pauli_path::expectation_with(&c, &obs, &opt).unwrap();
    assert!((v - 1.0).abs() < 1e-12);
    assert_eq!(st.peak_terms, 1);
}

#[test]
fn frame_handles_multi_term_observables() {
    let mut rng = StdRng::seed_from_u64(103);
    for trial in 0..10 {
        let n = 5;
        let c = random_universal(n, 25, &mut rng);
        let sv = sv_of(&c);
        let ps: Vec<String> = (0..3).map(|_| random_pauli(n, &mut rng)).collect();
        // O = P0 - 0.5 P1 + 2 P2 (built by evaluating linearity).
        let want =
            sv_pauli(&sv, &ps[0]) - 0.5 * sv_pauli(&sv, &ps[1]) + 2.0 * sv_pauli(&sv, &ps[2]);
        let got: f64 = [(0, 1.0), (1, -0.5), (2, 2.0)]
            .iter()
            .map(|&(i, w)| {
                w * pauli_path::expectation(
                    &c,
                    &PauliSum::from_str_single(&ps[i]),
                    DEFAULT_MAX_TERMS,
                )
                .unwrap()
                .0
            })
            .sum();
        assert!((got - want).abs() < 1e-9, "trial {trial}");
    }
}

#[test]
fn frame_matches_legacy_on_wide_registers() {
    // More than one 64-bit word per string (W = 2 and W = 4 key sizes).
    let mut rng = StdRng::seed_from_u64(104);
    for (n, t) in [(70, 18), (130, 14), (200, 12)] {
        for trial in 0..3 {
            let c = clifford_t_rounds(n, 1, t, &mut rng);
            for k in 0..3 {
                let p: String = if k == 0 {
                    // a low-weight observable near the light cone
                    let mut s = vec!['I'; n];
                    s[rng.random_range(0..n)] = 'Z';
                    s.into_iter().collect()
                } else {
                    let mut s = vec!['I'; n];
                    for _ in 0..3 {
                        s[rng.random_range(0..n)] = ['X', 'Y', 'Z'][rng.random_range(0..3)];
                    }
                    s.into_iter().collect()
                };
                let obs = PauliSum::from_str_single(&p);
                let (want, _) =
                    pauli_path::expectation_legacy(&c, &obs, DEFAULT_MAX_TERMS).unwrap();
                for opt in all_options() {
                    let (v, _) = pauli_path::expectation_with(&c, &obs, &opt).unwrap();
                    assert!(
                        (v - want).abs() < 1e-12,
                        "n={n} trial {trial}: {v} vs {want} ({opt:?})"
                    );
                }
            }
        }
    }
}

#[test]
fn frame_matches_legacy_on_benchmark_sized_circuits_with_nonzero_values() {
    // 24 qubits, 30 T gates: too many paths to be trivial, small enough for
    // the state vector as an independent reference.
    let mut rng = StdRng::seed_from_u64(105);
    for trial in 0..3 {
        let n = 16;
        let c = clifford_t_rounds(n, 1, 30, &mut rng);
        let sv = sv_of(&c);
        for _ in 0..3 {
            let p = random_pauli(n, &mut rng);
            let want = sv_pauli(&sv, &p);
            let obs = PauliSum::from_str_single(&p);
            let (v, _) = pauli_path::expectation(&c, &obs, DEFAULT_MAX_TERMS).unwrap();
            assert!((v - want).abs() < 1e-9, "trial {trial} {p}: {v} vs {want}");
        }
    }
}

#[test]
fn adder_benchmark_circuit_is_correct() {
    // The bench adder computes a + b and the Pauli-path values match the
    // state vector.
    for bits in [1, 2, 3] {
        let c = qsim_lab::bench::cuccaro_adder(bits);
        let n = c.num_qubits;
        // Classical check on basis inputs (drop the initial H layer).
        let body: Vec<Gate> = c.gates().skip(2 * bits).copied().collect();
        for av in 0..1usize << bits {
            for bv in 0..1usize << bits {
                let mut idx = 0usize;
                for i in 0..bits {
                    idx |= (av >> i & 1) << (1 + 2 * i);
                    idx |= (bv >> i & 1) << (2 + 2 * i);
                }
                let mut s = StateVectorF64::basis_state(n, idx);
                for g in &body {
                    s.apply_gate(g).unwrap();
                }
                let out = (0..1usize << n)
                    .find(|&i| s.amplitude(i).norm() > 0.5)
                    .unwrap();
                let mut sum = 0usize;
                for i in 0..bits {
                    sum |= (out >> (2 + 2 * i) & 1) << i;
                }
                sum |= (out >> (n - 1) & 1) << bits;
                assert_eq!(sum, av + bv, "bits={bits} a={av} b={bv}");
            }
        }
        let sv = sv_of(&c);
        let mut rng = StdRng::seed_from_u64(106 + bits as u64);
        for k in 0..6 {
            let p: String = if k == 0 {
                let mut s = vec!['I'; n];
                s[n - 1] = 'Z';
                s.into_iter().collect()
            } else {
                random_pauli(n, &mut rng)
            };
            check(&c, &p, sv_pauli(&sv, &p), &format!("adder bits={bits}"));
        }
    }
}

/// `<ψ| K Z_S K† |ψ>` with `ψ = c|0>` and `K` the Clifford skeleton of `c`,
/// computed on the state vector as `<φ|Z_S|φ>`, `φ = K† ψ`.
fn sv_skeleton_stabilizer(c: &Circuit, zs: &[usize]) -> f64 {
    let mut s = sv_of(c);
    s.apply_circuit(&qsim_lab::bench::clifford_skeleton(c).inverse())
        .unwrap();
    let mut p = vec!['I'; c.num_qubits];
    for &q in zs {
        p[q] = 'Z';
    }
    sv_pauli(&s, &p.into_iter().collect::<String>())
}

fn random_subset<R: Rng>(n: usize, rng: &mut R) -> Vec<usize> {
    loop {
        let s: Vec<usize> = (0..n).filter(|_| rng.random_bool(0.3)).collect();
        if !s.is_empty() {
            break s;
        }
    }
}

#[test]
fn frame_matches_statevector_on_skeleton_stabilizers() {
    // Observables `K Z_S K†` (stabilizers of the Clifford skeleton) have
    // generically non-zero expectation values, so a pruning rule that threw
    // away a contributing term would show up as a wrong, non-zero number.
    // T counts run from well below n (pruning removes almost everything)
    // to several times n (the x span saturates and the projection merges).
    let mut rng = StdRng::seed_from_u64(107);
    let (mut nonzero, mut total, mut pruned, mut shrunk) = (0, 0, 0u64, 0);
    for trial in 0..36 {
        let n = 4 + trial % 11; // 4..=14
        let t = [n / 2, n, 2 * n, 3 * n][trial % 4].min(30);
        let c = clifford_t_rounds(n, 2, t, &mut rng);
        for _ in 0..3 {
            let zs = random_subset(n, &mut rng);
            let want = sv_skeleton_stabilizer(&c, &zs);
            let obs = qsim_lab::bench::skeleton_stabilizer(&c, &zs);
            total += 1;
            if want.abs() > 1e-6 {
                nonzero += 1;
            }
            let (legacy, lst) =
                pauli_path::expectation_legacy(&c, &obs, DEFAULT_MAX_TERMS).unwrap();
            assert!((legacy - want).abs() < 1e-9, "legacy {legacy} vs sv {want}");
            let mut peak_noprune = 0;
            for opt in all_options() {
                let (v, st) = pauli_path::expectation_with(&c, &obs, &opt).unwrap();
                assert!(
                    (v - want).abs() < 1e-9 && (v - legacy).abs() < 1e-12,
                    "trial {trial} n={n} t={t} S={zs:?} {opt:?}: frame {v}, legacy {legacy}, sv {want}"
                );
                if opt.prune {
                    pruned += st.pruned_terms;
                    if st.peak_terms < lst.peak_terms {
                        shrunk += 1;
                    }
                } else {
                    peak_noprune = peak_noprune.max(st.peak_terms);
                }
            }
            assert!(peak_noprune > 0);
        }
    }
    eprintln!("skeleton stabilizers: {nonzero}/{total} non-zero, {pruned} terms pruned");
    assert!(
        nonzero * 10 >= total * 8,
        "expected mostly non-zero values: {nonzero}/{total}"
    );
    assert!(pruned > 1000, "pruning must actually fire: {pruned}");
    assert!(shrunk > 0);
}

#[test]
fn frame_matches_legacy_at_64_qubits_with_nonzero_values() {
    // Beyond the state vector: the benchmark's circuit family at n = 64 with
    // skeleton-stabilizer observables, against the legacy engine.
    let mut rng = StdRng::seed_from_u64(108);
    let mut nonzero = 0;
    for (trial, t) in [8, 16, 20, 24].into_iter().enumerate() {
        let c = clifford_t_rounds(64, 3, t, &mut rng);
        for _ in 0..2 {
            let zs = random_subset(64, &mut rng);
            let obs = qsim_lab::bench::skeleton_stabilizer(&c, &zs);
            let (want, _) = pauli_path::expectation_legacy(&c, &obs, DEFAULT_MAX_TERMS).unwrap();
            if want.abs() > 1e-6 {
                nonzero += 1;
            }
            for opt in all_options() {
                let (v, _) = pauli_path::expectation_with(&c, &obs, &opt).unwrap();
                assert!(
                    (v - want).abs() < 1e-12,
                    "trial {trial} t={t}: {v} vs {want} ({opt:?})"
                );
            }
        }
    }
    assert!(
        nonzero >= 6,
        "values should be mostly non-zero: {nonzero}/8"
    );
}

#[test]
fn pruned_observable_has_exactly_zero_value() {
    // The x-span rule's strongest consequence: with fewer T gates than
    // qubits, a random Pauli observable is usually discarded before any
    // propagation, and its value must then be exactly 0 on the state vector.
    let mut rng = StdRng::seed_from_u64(109);
    let mut dropped = 0;
    for trial in 0..30 {
        let n = 8 + trial % 7;
        let c = clifford_t_rounds(n, 2, 3, &mut rng);
        let sv = sv_of(&c);
        let p = random_pauli(n, &mut rng);
        let (v, st) =
            pauli_path::expectation(&c, &PauliSum::from_str_single(&p), DEFAULT_MAX_TERMS).unwrap();
        let want = sv_pauli(&sv, &p);
        assert!((v - want).abs() < 1e-9, "{p}: {v} vs {want}");
        if st.peak_terms == 0 {
            dropped += 1;
            assert!(want.abs() < 1e-12, "pruned {p} but sv says {want}");
        }
    }
    assert!(dropped > 10, "rule should fire often here: {dropped}");
}

#[test]
fn benchmark_family_matches_statevector() {
    // The exact circuit family of `qsim bench clifford-t`, at sizes the
    // state vector can check, with both benchmark observables (`z0` and
    // `stab`) and T counts on both sides of n.
    let (mut z0_nonzero, mut stab_nonzero, mut total) = (0, 0, 0);
    for n in [8, 10, 12, 14] {
        let build = qsim_lab::bench::clifford_t_family(n, 3, 3 * n, 3);
        for t in [n / 2, n, 2 * n, 3 * n] {
            let c = build(t);
            let sv = sv_of(&c);
            let mut z0 = vec!['I'; n];
            z0[0] = 'Z';
            let z0: String = z0.into_iter().collect();
            let want_z0 = sv_pauli(&sv, &z0);
            let want_stab = sv_skeleton_stabilizer(&c, &[0]);
            let obs_z0 = PauliSum::from_str_single(&z0);
            let obs_stab = qsim_lab::bench::skeleton_stabilizer(&c, &[0]);
            for (obs, want) in [(&obs_z0, want_z0), (&obs_stab, want_stab)] {
                let (v, _) = pauli_path::expectation(&c, obs, DEFAULT_MAX_TERMS).unwrap();
                assert!((v - want).abs() < 1e-9, "n={n} t={t}: {v} vs sv {want}");
            }
            total += 1;
            z0_nonzero += (want_z0.abs() > 1e-9) as usize;
            stab_nonzero += (want_stab.abs() > 1e-9) as usize;
        }
    }
    eprintln!(
        "bench family: <Z_0> non-zero {z0_nonzero}/{total}, stab non-zero {stab_nonzero}/{total}"
    );
    assert!(stab_nonzero * 10 >= total * 8);
}
