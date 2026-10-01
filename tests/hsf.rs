//! Hybrid Schrödinger–Feynman must reproduce the f64 state vector exactly
//! (max |Δamplitude| <= 1e-12).

mod common;

use common::*;
use num_complex::Complex64;
use proptest::prelude::*;
use qsim_lab::circuit::{Circuit, SimError};
use qsim_lab::gate::Gate;
use qsim_lab::hsf::{
    auto_partition, cut_bits, two_block_circuit, HsfOptions, HybridSchrodingerFeynman, LeafMode,
    SchmidtMode,
};
use qsim_lab::StateVectorF64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TOL: f64 = 1e-12;

fn max_diff(a: &[Complex64], b: &[Complex64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

fn random_partition<R: Rng>(n: usize, rng: &mut R) -> Vec<bool> {
    (0..n).map(|_| rng.random_bool(0.5)).collect()
}

/// All option combinations worth cross-checking.
fn option_sets() -> Vec<HsfOptions> {
    let mut v = Vec::new();
    for &elim in &[true, false] {
        for &asap in &[true, false] {
            for &schmidt in &[
                SchmidtMode::Analytic,
                SchmidtMode::Svd,
                SchmidtMode::MatrixUnits,
            ] {
                for &leaf in &[LeafMode::Forward, LeafMode::Bra] {
                    v.push(HsfOptions {
                        eliminate_swaps: elim,
                        asap,
                        schmidt,
                        leaf,
                        ..HsfOptions::default()
                    });
                }
            }
        }
    }
    v
}

#[test]
fn full_output_matches_statevector_small_random() {
    let mut rng = StdRng::seed_from_u64(1);
    let opts = option_sets();
    for trial in 0..120 {
        let n = 2 + trial % 8;
        let c = random_universal(n, 4 + trial % 25, &mut rng);
        let want = sv_of(&c);
        let part = random_partition(n, &mut rng);
        let o = opts[trial % opts.len()].clone();
        let h = HybridSchrodingerFeynman::new(&c, &part, o.clone()).unwrap();
        let got = h.state_vector().unwrap();
        let d = max_diff(&got, want.amplitudes());
        assert!(
            d <= TOL,
            "trial {trial} n={n} {o:?} part={part:?}: {d}\n{c:?}"
        );
        // amplitude API agrees on every index
        let xs: Vec<usize> = (0..1 << n).collect();
        let amps = h.amplitudes(&xs).unwrap();
        let d = max_diff(&amps, want.amplitudes());
        assert!(d <= TOL, "trial {trial} amplitudes: {d}");
    }
}

/// Builds an `n`-qubit circuit with exactly `k` gates crossing the cut
/// `A = in_a`, cycling over every crossing gate type and both orientations,
/// plus random local gates of every type inside each block.
fn circuit_with_cuts<R: Rng>(
    n: usize,
    in_a: &[bool],
    k: usize,
    local: usize,
    rng: &mut R,
) -> Circuit {
    let a: Vec<usize> = (0..n).filter(|&q| in_a[q]).collect();
    let b: Vec<usize> = (0..n).filter(|&q| !in_a[q]).collect();
    let mut c = Circuit::new(n);
    let pick = |v: &[usize], rng: &mut R| v[rng.random_range(0..v.len())];
    let local_gate = |blk: &[usize], rng: &mut R| -> Gate {
        let q = pick(blk, rng);
        let th = rng.random_range(-3.0..3.0);
        let other = |rng: &mut R, not: &[usize]| loop {
            let x = pick(blk, rng);
            if !not.contains(&x) {
                break x;
            }
        };
        let r = if blk.len() >= 3 {
            rng.random_range(0..17)
        } else if blk.len() == 2 {
            rng.random_range(0..16)
        } else {
            rng.random_range(0..12)
        };
        match r {
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
            12 => Gate::Cnot(q, other(rng, &[q])),
            13 => Gate::Cz(q, other(rng, &[q])),
            14 => Gate::Swap(q, other(rng, &[q])),
            15 => Gate::CPhase(q, other(rng, &[q]), th),
            _ => {
                let x = other(rng, &[q]);
                let y = other(rng, &[q, x]);
                Gate::Ccx(q, x, y)
            }
        }
    };
    // positions of the crossing gates among the local ones
    let total = local + k;
    let mut is_cut = vec![false; total];
    let mut placed = 0;
    while placed < k {
        let i = rng.random_range(0..total);
        if !is_cut[i] {
            is_cut[i] = true;
            placed += 1;
        }
    }
    let mut kind = 0;
    for cut in is_cut {
        if cut {
            let (x, y) = (pick(&a, rng), pick(&b, rng));
            let th = rng.random_range(-3.0..3.0);
            // both orientations: the first argument alternates between A and B
            let (p, q) = if kind % 2 == 0 { (x, y) } else { (y, x) };
            let g = match (kind / 2) % 4 {
                0 => Gate::Cz(p, q),
                1 => Gate::Cnot(p, q),
                2 => Gate::CPhase(p, q, th),
                _ => {
                    // Toffoli split 1|2, alternating which wire is alone
                    let blk = if in_a[p] { &a } else { &b };
                    let z = if blk.len() > 1 {
                        loop {
                            let z = pick(blk, rng);
                            if z != p {
                                break z;
                            }
                        }
                    } else {
                        p
                    };
                    if z == p {
                        Gate::Cz(p, q)
                    } else if kind % 4 == 3 {
                        Gate::Ccx(p, z, q) // controls together, target alone
                    } else {
                        Gate::Ccx(q, p, z) // a control alone
                    }
                }
            };
            kind += 1;
            c.gate(g);
        } else {
            let blk = if rng.random_bool(0.5) { &a } else { &b };
            c.gate(local_gate(blk, rng));
        }
    }
    c
}

#[test]
fn medium_circuits_all_gate_types_full_output() {
    let mut rng = StdRng::seed_from_u64(2);
    for trial in 0..40 {
        let n = 6 + trial % 9; // 6..14
        let mut part = random_partition(n, &mut rng);
        part[0] = true;
        part[1] = false;
        let k = trial % 9;
        let c = circuit_with_cuts(n, &part, k, 60, &mut rng);
        let want = sv_of(&c);
        for o in [
            HsfOptions::default(),
            HsfOptions {
                eliminate_swaps: false,
                asap: false,
                threads: 1,
                ..HsfOptions::default()
            },
        ] {
            let h = HybridSchrodingerFeynman::new(&c, &part, o).unwrap();
            let d = max_diff(&h.state_vector().unwrap(), want.amplitudes());
            assert!(d <= TOL, "trial {trial} n={n} k={k}: {d}");
        }
    }
}

/// The headline accuracy gate: 20–22 qubits, 0..=12 crossing gates of every
/// type in both orientations, both cut orientations (A and B exchanged),
/// a batch of amplitudes in one sweep.
#[test]
fn large_circuits_amplitude_batches_match_statevector() {
    let mut rng = StdRng::seed_from_u64(3);
    for (i, k) in (0..=12).enumerate() {
        let n = if i % 3 == 2 { 22 } else { 20 };
        // interleaved random partition, roughly balanced
        let mut idx: Vec<usize> = (0..n).collect();
        for j in (1..n).rev() {
            idx.swap(j, rng.random_range(0..=j));
        }
        let mut part = vec![false; n];
        for &q in &idx[..n / 2 - (i % 2)] {
            part[q] = true;
        }
        let c = circuit_with_cuts(n, &part, k, 150, &mut rng);
        let want = sv_of(&c);
        let mut xs: Vec<usize> = (0..48).map(|_| rng.random_range(0..1usize << n)).collect();
        // include the largest amplitudes so the check is not vacuous
        let mut by_mag: Vec<usize> = (0..1usize << n).collect();
        by_mag.select_nth_unstable_by(8, |&x, &y| {
            want.amplitude(y)
                .norm()
                .partial_cmp(&want.amplitude(x).norm())
                .unwrap()
        });
        xs.extend_from_slice(&by_mag[..8]);
        let w: Vec<Complex64> = xs.iter().map(|&x| want.amplitude(x)).collect();
        let flipped: Vec<bool> = part.iter().map(|&p| !p).collect();
        for (pi, p) in [&part, &flipped].into_iter().enumerate() {
            for leaf in [LeafMode::Forward, LeafMode::Bra, LeafMode::Auto] {
                let o = HsfOptions {
                    leaf,
                    ..HsfOptions::default()
                };
                let h = HybridSchrodingerFeynman::new(&c, p, o).unwrap();
                let got = h.amplitudes(&xs).unwrap();
                let d = max_diff(&got, &w);
                assert!(
                    d <= TOL,
                    "n={n} k={k} orientation {pi} {leaf:?}: {d} (cuts {:?})",
                    h.cut_gates()
                );
            }
        }
    }
}

#[test]
fn full_output_twenty_qubits_with_cuts() {
    let mut rng = StdRng::seed_from_u64(4);
    for k in [0, 3, 7, 10] {
        let n = 20;
        let part: Vec<bool> = (0..n).map(|q| q % 3 != 0).collect(); // unbalanced 13|7
        let c = circuit_with_cuts(n, &part, k, 120, &mut rng);
        let want = sv_of(&c);
        let h = HybridSchrodingerFeynman::new(&c, &part, HsfOptions::default()).unwrap();
        let got = h.state_vector().unwrap();
        let d = max_diff(&got, want.amplitudes());
        assert!(d <= TOL, "k={k}: {d}");
        let norm: f64 = got.iter().map(|a| a.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }
}

#[test]
fn rank_two_cuts_give_two_paths_each() {
    let mut c = Circuit::new(4);
    c.h(0).h(1).h(2).h(3);
    c.cz(0, 2)
        .cnot(3, 1)
        .cphase(1, 3, 0.2)
        .ccx(0, 1, 2)
        .ccx(2, 3, 0);
    c.swap(0, 3); // removed by relabelling
    let part = [true, true, false, false];
    let h = HybridSchrodingerFeynman::new(&c, &part, HsfOptions::default()).unwrap();
    assert_eq!(h.num_cut_gates(), 5);
    assert!(h.cut_gates().iter().all(|&(_, r)| r == 2));
    assert_eq!(h.num_paths(), 32);
    // without SWAP elimination the SWAP is a rank-4 crossing gate
    let h2 = HybridSchrodingerFeynman::new(
        &c,
        &part,
        HsfOptions {
            eliminate_swaps: false,
            ..HsfOptions::default()
        },
    )
    .unwrap();
    assert_eq!(h2.num_paths(), 128);
    let want = sv_of(&c);
    for hh in [&h, &h2] {
        assert!(max_diff(&hh.state_vector().unwrap(), want.amplitudes()) <= TOL);
    }
}

#[test]
fn zero_pruning_is_exact_and_prunes() {
    // CZs on |0> controls: every P1 branch is exactly zero.
    let mut c = Circuit::new(6);
    for i in 0..3 {
        c.cz(i, i + 3);
    }
    c.h(0).h(3).cnot(0, 3).t(3).h(1);
    let part = [true, true, true, false, false, false];
    let want = sv_of(&c);
    for prune in [true, false] {
        let o = HsfOptions {
            prune_zero: prune,
            ..HsfOptions::default()
        };
        let h = HybridSchrodingerFeynman::new(&c, &part, o).unwrap();
        assert!(max_diff(&h.state_vector().unwrap(), want.amplitudes()) <= TOL);
    }
}

#[test]
fn measurements_are_rejected() {
    let mut c = Circuit::new(3);
    c.h(0).cnot(0, 1).measure(1).cnot(1, 2);
    let err = HybridSchrodingerFeynman::new(&c, &[true, false, false], HsfOptions::default())
        .unwrap_err();
    assert_eq!(
        err,
        SimError::MeasurementNotSupported {
            backend: "hsf",
            op_index: 2
        }
    );
    assert!(auto_partition(&c, &HsfOptions::default()).is_err());
    assert!(err.to_string().contains("non-unitary"));
}

#[test]
fn other_non_unitary_ops_are_rejected() {
    let mut a = Circuit::new(3);
    a.h(0).reset(1).cnot(0, 2);
    let mut b = Circuit::new(3);
    b.h(0).x_flip(1, 0.1).cnot(0, 2);
    let mut c = Circuit::new(3);
    c.h(0).measure(0).c_if(0, Gate::X(2));
    for circ in [a, b, c] {
        let err =
            HybridSchrodingerFeynman::new(&circ, &[true, false, false], HsfOptions::default())
                .unwrap_err();
        assert!(
            matches!(
                err,
                SimError::MeasurementNotSupported { backend: "hsf", .. }
            ),
            "{err:?}"
        );
    }
}

#[test]
fn bad_inputs_are_rejected() {
    let mut c = Circuit::new(3);
    c.h(0);
    let h = HybridSchrodingerFeynman::new(&c, &[true, false, true], HsfOptions::default()).unwrap();
    assert!(h.amplitude(8).is_err());
    let mut bad = Circuit::new(3);
    bad.cnot(1, 1);
    assert!(
        HybridSchrodingerFeynman::new(&bad, &[true, false, true], HsfOptions::default()).is_err()
    );
}

#[test]
fn memory_cap_is_enforced() {
    let mut rng = StdRng::seed_from_u64(5);
    let c = two_block_circuit(16, 8, 4, 6, false, &mut rng);
    let part: Vec<bool> = (0..16).map(|q| q < 8).collect();
    let tiny = HsfOptions {
        max_bytes: 4096,
        ..HsfOptions::default()
    };
    let h = HybridSchrodingerFeynman::new(&c, &part, tiny).unwrap();
    assert!(matches!(h.amplitudes(&[0]), Err(SimError::TooLarge { .. })));
    assert!(matches!(h.state_vector(), Err(SimError::TooLarge { .. })));
    // a cap that only fits one worker still works, with identical results
    let one = HsfOptions {
        max_bytes: 8 * 2 * 256 * 16 + 256 * 16 * 4,
        ..HsfOptions::default()
    };
    let h1 = HybridSchrodingerFeynman::new(&c, &part, one).unwrap();
    let h4 = HybridSchrodingerFeynman::new(&c, &part, HsfOptions::default()).unwrap();
    let xs = [0, 1, 12345, 65535];
    let d = max_diff(&h1.amplitudes(&xs).unwrap(), &h4.amplitudes(&xs).unwrap());
    assert!(d <= TOL);
    // full output beyond 2^26 amplitudes is refused, not attempted
    let big = Circuit::new(40);
    let hb = HybridSchrodingerFeynman::new(
        &big,
        &(0..40).map(|q| q < 20).collect::<Vec<_>>(),
        HsfOptions::default(),
    )
    .unwrap();
    assert!(matches!(hb.state_vector(), Err(SimError::TooLarge { .. })));
    assert!((hb.amplitude(0).unwrap() - Complex64::new(1.0, 0.0)).norm() < 1e-15);
}

#[test]
fn thread_counts_agree() {
    let mut rng = StdRng::seed_from_u64(6);
    let c = two_block_circuit(14, 7, 6, 8, false, &mut rng);
    let part: Vec<bool> = (0..14).map(|q| q < 7).collect();
    let want = sv_of(&c);
    for threads in [1, 2, 3, 8] {
        let h = HybridSchrodingerFeynman::new(
            &c,
            &part,
            HsfOptions {
                threads,
                ..HsfOptions::default()
            },
        )
        .unwrap();
        assert!(max_diff(&h.state_vector().unwrap(), want.amplitudes()) <= TOL);
    }
}

#[test]
fn auto_partition_finds_planted_cut() {
    let mut rng = StdRng::seed_from_u64(7);
    for trial in 0..10 {
        let n = 12 + 2 * (trial % 5);
        let na = n / 2;
        let k = trial % 4;
        let c = two_block_circuit(n, na, 6, k, false, &mut rng);
        // scramble qubit labels so the planted blocks are not contiguous
        let mut perm: Vec<usize> = (0..n).collect();
        for j in (1..n).rev() {
            perm.swap(j, rng.random_range(0..=j));
        }
        let mut s = Circuit::new(n);
        for g in c.gates() {
            s.gate(qsim_lab::hsf::map_gate(*g, |q| perm[q]));
        }
        let o = HsfOptions::default();
        let part = auto_partition(&s, &o).unwrap();
        let bits = cut_bits(&s, &part, &o).unwrap();
        assert!(
            bits as usize <= k,
            "trial {trial}: found {bits} > planted {k}"
        );
        let h = HybridSchrodingerFeynman::auto(&s, o).unwrap();
        assert!(h.num_paths() <= 1 << k);
        let want = sv_of(&s);
        assert!(max_diff(&h.state_vector().unwrap(), want.amplitudes()) <= TOL);
    }
}

#[test]
fn forty_qubit_two_block_amplitudes_are_consistent() {
    // Beyond the state vector: check internal consistency instead. Two
    // independent blocks (no crossing gates) give product amplitudes that
    // can be computed from 20-qubit state vectors directly.
    let mut rng = StdRng::seed_from_u64(8);
    let n = 40;
    let c = two_block_circuit(n, 20, 3, 0, false, &mut rng);
    let mut ca = Circuit::new(20);
    let mut cb = Circuit::new(20);
    for g in c.gates() {
        let qs = g.qubits();
        if qs[0] < 20 {
            ca.gate(*g);
        } else {
            cb.gate(qsim_lab::hsf::map_gate(*g, |q| q - 20));
        }
    }
    let (sa, sb) = (sv_of(&ca), sv_of(&cb));
    let part: Vec<bool> = (0..n).map(|q| q < 20).collect();
    let h = HybridSchrodingerFeynman::new(&c, &part, HsfOptions::default()).unwrap();
    let xs: Vec<usize> = (0..8).map(|_| rng.random_range(0..1usize << 40)).collect();
    let got = h.amplitudes(&xs).unwrap();
    for (x, g) in xs.iter().zip(got) {
        let want = sa.amplitude(x & 0xfffff) * sb.amplitude(x >> 20);
        assert!((g - want).norm() <= TOL);
    }
}

fn arb_gate(n: usize) -> impl Strategy<Value = Gate> {
    (0..17usize, 0..n, 0..n, 0..n, -3.2f64..3.2).prop_filter_map(
        "distinct",
        move |(k, a, b, t, th)| {
            let g = match k {
                0 => Gate::H(a),
                1 => Gate::X(a),
                2 => Gate::Y(a),
                3 => Gate::Z(a),
                4 => Gate::S(a),
                5 => Gate::Sdg(a),
                6 => Gate::T(a),
                7 => Gate::Tdg(a),
                8 => Gate::Rx(a, th),
                9 => Gate::Ry(a, th),
                10 => Gate::Rz(a, th),
                11 => Gate::Phase(a, th),
                12 => Gate::Cnot(a, b),
                13 => Gate::Cz(a, b),
                14 => Gate::Swap(a, b),
                15 => Gate::CPhase(a, b, th),
                _ => Gate::Ccx(a, b, t),
            };
            let qs = g.qubits();
            let ok = (0..qs.len()).all(|i| !qs[..i].contains(&qs[i]));
            ok.then_some(g)
        },
    )
}

fn arb_case() -> impl Strategy<Value = (Circuit, Vec<bool>, bool, bool)> {
    (2usize..9).prop_flat_map(|n| {
        (
            prop::collection::vec(arb_gate(n), 0..40),
            prop::collection::vec(any::<bool>(), n),
            any::<bool>(),
            any::<bool>(),
        )
            .prop_map(move |(gs, part, elim, asap)| {
                let mut c = Circuit::new(n);
                for g in gs {
                    c.gate(g);
                }
                (c, part, elim, asap)
            })
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_hsf_equals_statevector((c, part, elim, asap) in arb_case()) {
        let want = sv_of(&c);
        let o = HsfOptions { eliminate_swaps: elim, asap, ..HsfOptions::default() };
        let h = HybridSchrodingerFeynman::new(&c, &part, o).unwrap();
        let got = h.state_vector().unwrap();
        prop_assert!(max_diff(&got, want.amplitudes()) <= TOL);
    }

    #[test]
    fn prop_auto_partition_is_valid_and_exact((c, _part, elim, _asap) in arb_case()) {
        let o = HsfOptions { eliminate_swaps: elim, ..HsfOptions::default() };
        let h = HybridSchrodingerFeynman::auto(&c, o).unwrap();
        let want = sv_of(&c);
        prop_assert!(max_diff(&h.state_vector().unwrap(), want.amplitudes()) <= TOL);
        let _ = StateVectorF64::new(1);
    }
}
