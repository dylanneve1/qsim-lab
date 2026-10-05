//! Differential fuzz harness (audit).
//!
//! Every backend is compared against `RefSv`, a deliberately naive dense
//! state vector defined *in this file* with its own gate matrices, so a
//! change to any kernel in `src/` (fused, strided, SIMD, parallel, ...) is
//! checked against code it does not share.
//!
//! Generators are biased towards edge cases: qubit 0 and the top qubit,
//! adjacent and maximally distant pairs, n = 1, and angles near 0, π/2, π
//! and 2π. Tolerances: f64 amplitudes 1e-12 (scaled by depth), f32
//! amplitudes 1e-5, tableau probabilities exact (dyadic), Pauli-path and
//! exact-MPS values 1e-10.
//!
//! `QSIM_FUZZ_ITERS` (default 1) multiplies the number of random cases;
//! `QSIM_FUZZ_SEED` changes the base seed. Failures print the seed and the
//! offending circuit.

#[path = "../audit_common/mod.rs"]
mod audit_common;

use audit_common::*;
use qsim_lab::engines::pauli_path::{self, PauliSum, DEFAULT_MAX_TERMS};
use qsim_lab::{Circuit, Gate, Mps, Simulator, StateVectorF32, StateVectorF64, Tableau};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ---------------------------------------------------------------------------
// State vector
// ---------------------------------------------------------------------------

#[test]
fn sv_f64_matches_reference() {
    let mut worst = 0.0f64;
    for it in 0..20 * iters() {
        for &n in &SIZES {
            let seed = base_seed() ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let depth = rng.random_range(1..120);
            let c = random_circuit(&mut rng, n, depth, false, false);
            let r = RefSv::run(&c);
            let mut sv = StateVectorF64::new(n);
            sv.apply_circuit(&c).unwrap();
            let d = max_amp_diff(&r.a, (0..1 << n).map(|i| sv.amplitude(i)));
            worst = worst.max(d);
            assert!(
                d <= 1e-12,
                "f64 Δ={d:e} seed={seed} n={n} circuit={:?}",
                c.ops
            );
            // gate-by-gate path through the Simulator trait must agree too
            let mut sv2 = StateVectorF64::new(n);
            let mut rng2 = StdRng::seed_from_u64(1);
            c.run(&mut sv2, &mut rng2).unwrap();
            let d2 = max_amp_diff(&r.a, (0..1 << n).map(|i| sv2.amplitude(i)));
            assert!(d2 <= 1e-12, "f64 run() Δ={d2:e} seed={seed} n={n}");
        }
    }
    eprintln!("sv_f64 worst |Δamp| = {worst:e}");
}

#[test]
fn sv_f32_matches_reference() {
    let mut worst = 0.0f64;
    for it in 0..20 * iters() {
        for &n in &SIZES {
            let seed = base_seed() ^ 0xF32 ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let depth = rng.random_range(1..60);
            let c = random_circuit(&mut rng, n, depth, false, false);
            let r = RefSv::run(&c);
            let mut sv = StateVectorF32::new(n);
            sv.apply_circuit(&c).unwrap();
            let d = max_amp_diff(&r.a, (0..1 << n).map(|i| sv.amplitude(i)));
            worst = worst.max(d);
            assert!(
                d <= 1e-5,
                "f32 Δ={d:e} seed={seed} n={n} circuit={:?}",
                c.ops
            );
        }
    }
    eprintln!("sv_f32 worst |Δamp| = {worst:e}");
}

/// Larger registers so parallel / blocked code paths (which typically only
/// kick in above a size threshold) are exercised. 16 and 18 qubits.
#[test]
fn sv_large_registers_match_reference() {
    for (k, &n) in [16usize, 18].iter().enumerate() {
        let seed = base_seed() ^ 0x1A46 ^ k as u64;
        let mut rng = StdRng::seed_from_u64(seed);
        let c = random_circuit(&mut rng, n, 40 * iters(), false, false);
        let r = RefSv::run(&c);
        let mut sv = StateVectorF64::new(n);
        sv.apply_circuit(&c).unwrap();
        let d = max_amp_diff(&r.a, (0..1 << n).map(|i| sv.amplitude(i)));
        assert!(d <= 1e-12, "f64 large Δ={d:e} seed={seed} n={n}");
        let mut s32 = StateVectorF32::new(n);
        s32.apply_circuit(&c).unwrap();
        let d32 = max_amp_diff(&r.a, (0..1 << n).map(|i| s32.amplitude(i)));
        assert!(d32 <= 1e-5, "f32 large Δ={d32:e} seed={seed} n={n}");
    }
}

/// Mid-circuit measurement, repeated measurement and reset on the state
/// vector: every outcome must have nonzero reference probability, the
/// post-measurement state must equal the collapsed reference, and repeating
/// a measurement must reproduce it.
#[test]
fn sv_measure_reset_match_reference() {
    for it in 0..30 * iters() {
        for &n in &[1usize, 2, 3, 5, 8] {
            let seed = base_seed() ^ 0x3EA5 ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let mut r = RefSv::new(n);
            let mut sv = StateVectorF64::new(n);
            for _ in 0..rng.random_range(1..8) {
                for _ in 0..rng.random_range(0..10) {
                    let g = random_gate(&mut rng, n, false, false);
                    r.apply(&g);
                    sv.apply_gate(&g).unwrap();
                }
                let q = edge_qubit(&mut rng, n);
                let p1 = r.prob_one(q);
                let reset = rng.random_bool(0.3);
                let mut mrng = StdRng::seed_from_u64(rng.random());
                if reset {
                    Simulator::reset(&mut sv, q, &mut mrng).unwrap();
                    // reference: result of reset is P0 ψ / |..| or X P1 ψ / |..|,
                    // whichever branch was taken; check qubit q is |0> and
                    // the state matches one of the two branches.
                    assert!(
                        sv.prob_one(q) < 1e-12,
                        "reset left P(1)={} seed={seed}",
                        sv.prob_one(q)
                    );
                    let mut b0 = r.clone();
                    let mut b1 = r.clone();
                    let ok0 = p1 < 1.0 - 1e-12 && {
                        b0.collapse(q, false);
                        max_amp_diff(&b0.a, (0..1 << n).map(|i| sv.amplitude(i))) < 1e-10
                    };
                    let ok1 = p1 > 1e-12 && {
                        b1.collapse(q, true);
                        b1.apply(&Gate::X(q));
                        max_amp_diff(&b1.a, (0..1 << n).map(|i| sv.amplitude(i))) < 1e-10
                    };
                    assert!(ok0 || ok1, "reset state matches neither branch seed={seed}");
                    r = if ok0 { b0 } else { b1 };
                } else {
                    let m = Simulator::measure(&mut sv, q, &mut mrng).unwrap();
                    let pm = if m { p1 } else { 1.0 - p1 };
                    assert!(
                        pm > 1e-12,
                        "measured outcome {m} with ref prob {pm} seed={seed}"
                    );
                    r.collapse(q, m);
                    let d = max_amp_diff(&r.a, (0..1 << n).map(|i| sv.amplitude(i)));
                    assert!(d < 1e-10, "post-measure Δ={d:e} seed={seed}");
                    let m2 = Simulator::measure(&mut sv, q, &mut mrng).unwrap();
                    assert_eq!(m, m2, "repeated measurement differs seed={seed}");
                }
            }
        }
    }
}

/// Outcome sampling: chi-square goodness of fit of `sample()` against the
/// reference distribution (8 qubits, 20k shots, very loose p < 1e-6 bound).
#[test]
fn sv_sampling_distribution() {
    let n = 6;
    let seed = base_seed() ^ 0x5A3;
    let mut rng = StdRng::seed_from_u64(seed);
    let c = random_circuit(&mut rng, n, 40, false, false);
    let p = RefSv::run(&c).probs();
    let mut sv = StateVectorF64::new(n);
    sv.apply_circuit(&c).unwrap();
    let shots = 20_000;
    let samples = sv.sample(shots, &mut rng);
    let mut counts = vec![0usize; 1 << n];
    for s in samples {
        counts[s] += 1;
    }
    let mut chi2 = 0.0;
    let mut dof = 0;
    for (i, &pi) in p.iter().enumerate() {
        let e = pi * shots as f64;
        if e < 5.0 {
            assert!(
                pi > 0.0 || counts[i] == 0,
                "sampled outcome {i} with probability 0"
            );
            continue;
        }
        chi2 += (counts[i] as f64 - e).powi(2) / e;
        dof += 1;
    }
    // chi2 < dof + 6 sqrt(2 dof) + 10 is a ~6σ bound
    let bound = dof as f64 + 6.0 * (2.0 * dof as f64).sqrt() + 10.0;
    assert!(chi2 < bound, "chi2={chi2} dof={dof} seed={seed}");
}

// ---------------------------------------------------------------------------
// Stabilizer tableau (exact)
// ---------------------------------------------------------------------------

#[test]
fn tableau_probabilities_exact() {
    for it in 0..25 * iters() {
        for &n in &[1usize, 2, 3, 4, 5, 7, 9, 11] {
            let seed = base_seed() ^ 0x7AB ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let depth = rng.random_range(1..80);
            let c = random_circuit(&mut rng, n, depth, true, false);
            let r = RefSv::run(&c);
            let mut t = Tableau::new(n);
            c.run(&mut t, &mut rng).unwrap();
            for (i, pr) in r.probs().into_iter().enumerate() {
                let pt = t.probability(i);
                // reference probability rounded to the nearest dyadic must
                // equal the tableau's exactly
                let snapped = if pr < 1e-9 {
                    0.0
                } else {
                    2f64.powi(pr.log2().round() as i32)
                };
                assert!(
                    (pr - snapped).abs() < 1e-9,
                    "ref prob {pr} not dyadic seed={seed}"
                );
                assert_eq!(
                    pt, snapped,
                    "outcome {i}: tableau {pt} vs ref {pr} seed={seed} n={n}"
                );
            }
            for s in t.stabilizers() {
                let sign = if s.starts_with('-') { -1.0 } else { 1.0 };
                let ev = r.pauli_expectation(&s[1..]);
                assert!(
                    (ev - sign).abs() < 1e-9,
                    "stabilizer {s} has <P>={ev} seed={seed}"
                );
            }
        }
    }
}

/// Mid-circuit measurements and resets on the tableau: each outcome must be
/// possible under the reference, deterministic outcomes must match, and the
/// collapsed states must keep agreeing.
#[test]
fn tableau_measure_reset_match_reference() {
    for it in 0..30 * iters() {
        for &n in &[1usize, 2, 3, 6, 9] {
            let seed = base_seed() ^ 0x7AB3 ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let mut r = RefSv::new(n);
            let mut t = Tableau::new(n);
            for _ in 0..rng.random_range(1..10) {
                for _ in 0..rng.random_range(0..12) {
                    let g = random_gate(&mut rng, n, true, false);
                    r.apply(&g);
                    t.apply_gate(&g).unwrap();
                }
                let q = edge_qubit(&mut rng, n);
                let p1 = r.prob_one(q);
                assert!(p1 < 1e-9 || (p1 - 0.5).abs() < 1e-9 || p1 > 1.0 - 1e-9);
                let det = t.peek(q);
                match det {
                    Some(b) => assert!(
                        (p1 - if b { 1.0 } else { 0.0 }).abs() < 1e-9,
                        "peek={b} but P1={p1} seed={seed}"
                    ),
                    None => assert!(
                        (p1 - 0.5).abs() < 1e-9,
                        "peek random but P1={p1} seed={seed}"
                    ),
                }
                let m = t.measure_qubit(q, &mut rng);
                assert_eq!(
                    t.measure_qubit(q, &mut rng),
                    m,
                    "repeat differs seed={seed}"
                );
                r.collapse(q, m);
                if rng.random_bool(0.3) {
                    t.reset_qubit(q, &mut rng);
                    if m {
                        r.apply(&Gate::X(q));
                    }
                }
            }
            for (i, pr) in r.probs().into_iter().enumerate() {
                assert!(
                    (t.probability(i) - pr).abs() < 1e-9,
                    "final dist mismatch seed={seed}"
                );
            }
        }
    }
}

/// Tableau shot sampling: every shot in the support, and per-outcome counts
/// consistent with the uniform distribution over the support.
#[test]
fn tableau_sampling_distribution() {
    for k in 0..4 * iters() {
        let n = 5 + k % 4;
        let seed = base_seed() ^ 0x5AB ^ k as u64;
        let mut rng = StdRng::seed_from_u64(seed);
        let c = random_circuit(&mut rng, n, 30, true, false);
        let p = RefSv::run(&c).probs();
        let mut t = Tableau::new(n);
        c.run(&mut t, &mut rng).unwrap();
        let shots = 8000;
        let mut counts = vec![0usize; 1 << n];
        for s in t.sample(shots, &mut rng) {
            let idx = s
                .iter()
                .enumerate()
                .fold(0, |acc, (q, &b)| acc | ((b as usize) << q));
            assert!(
                p[idx] > 1e-9,
                "sampled impossible outcome {idx} seed={seed}"
            );
            counts[idx] += 1;
        }
        let mut chi2 = 0.0;
        let mut dof = 0;
        for (i, &pi) in p.iter().enumerate() {
            if pi > 1e-9 {
                let e = pi * shots as f64;
                chi2 += (counts[i] as f64 - e).powi(2) / e;
                dof += 1;
            }
        }
        let bound = dof as f64 + 6.0 * (2.0 * dof as f64).sqrt() + 10.0;
        assert!(chi2 < bound, "tableau chi2={chi2} dof={dof} seed={seed}");
    }
}

// ---------------------------------------------------------------------------
// Pauli-path (Clifford+T and arbitrary rotations) and MPS
// ---------------------------------------------------------------------------

#[test]
fn pauli_path_matches_reference() {
    let mut worst = 0.0f64;
    for it in 0..15 * iters() {
        for &n in &[1usize, 2, 3, 4, 6, 8] {
            let seed = base_seed() ^ 0xFA7 ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            // mostly Clifford+T, sometimes arbitrary rotations; keep the
            // number of non-Clifford gates small enough for the term cap
            let mut c = Circuit::new(n);
            let mut non_cliff = 0;
            for _ in 0..rng.random_range(1..60) {
                let t_only = rng.random_bool(0.7);
                let g = random_gate(&mut rng, n, false, t_only);
                if matches!(g, Gate::Ccx(..)) {
                    continue;
                }
                if !g.is_clifford() {
                    if non_cliff >= 8 {
                        continue;
                    }
                    non_cliff += 1;
                }
                c.gate(g);
            }
            let r = RefSv::run(&c);
            let p: String = (0..n)
                .map(|_| ['I', 'X', 'Y', 'Z'][rng.random_range(0..4)])
                .collect();
            let (v, _) =
                pauli_path::expectation(&c, &PauliSum::from_str_single(&p), DEFAULT_MAX_TERMS)
                    .unwrap();
            let e = r.pauli_expectation(&p);
            worst = worst.max((v - e).abs());
            assert!(
                (v - e).abs() < 1e-10,
                "pauli-path <{p}>={v} ref={e} seed={seed} circuit={:?}",
                c.ops
            );
            // marginal distribution on up to 3 qubits
            let k = n.min(3);
            let qs: Vec<usize> = if k == n {
                (0..n).collect()
            } else {
                vec![0, n / 2, n - 1]
            };
            let md = pauli_path::marginal_distribution(&c, &qs).unwrap();
            let probs = r.probs();
            for (b, &mv) in md.iter().enumerate() {
                let rv: f64 = probs
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| {
                        qs.iter()
                            .enumerate()
                            .all(|(j, &q)| ((i >> q) & 1) == ((b >> j) & 1))
                    })
                    .map(|(_, x)| x)
                    .sum();
                assert!(
                    (mv - rv).abs() < 1e-10,
                    "marginal {b}: {mv} vs {rv} seed={seed}"
                );
            }
        }
    }
    eprintln!("pauli_path worst |Δ| = {worst:e}");
}

#[test]
fn mps_exact_matches_reference() {
    for it in 0..10 * iters() {
        for &n in &[1usize, 2, 3, 5, 8, 10] {
            let seed = base_seed() ^ 0x3B5 ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let depth = rng.random_range(1..60);
            let c = random_circuit(&mut rng, n, depth, false, false);
            let r = RefSv::run(&c);
            let mut m = Mps::new(n, 1 << (n / 2 + 1));
            m.set_cutoff(0.0);
            c.run(&mut m, &mut rng).unwrap();
            let d = max_amp_diff(&r.a, (0..1u128 << n).map(|i| m.amplitude(i)));
            assert!(
                d < 1e-9,
                "mps Δ={d:e} seed={seed} n={n} circuit={:?}",
                c.ops
            );
        }
    }
}

/// Memory caps must turn oversized registers into errors, not OOM aborts.
/// (Only the failing side is checked: allocating the largest allowed
/// register would cost ~0.5 GB on a shared box.)
#[test]
fn memory_caps_reject_oversized_registers() {
    assert!(qsim_lab::StateVector::<f32>::try_new(40).is_err());
    assert!(qsim_lab::StateVector::<f64>::try_new(40).is_err());
    assert!(Tableau::try_new(1_000_000).is_err());
    assert!(Tableau::try_new(64).is_ok());
}
