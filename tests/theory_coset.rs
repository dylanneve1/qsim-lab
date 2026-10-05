//! Executable checks for the theorems of `research/theory/theory-coset.md`
//! (coset-representation error in the windowed Gidney–Ekerå Shor circuit,
//! `src/shor/ge.rs`, `GeOpts::coset = c`).
//!
//! Every test fails if the corresponding statement is false:
//! * `model_*`: the abstract permutation model of the coset circuit
//!   (tests/theory_coset_model) reproduces the gate-level engine's exact
//!   output distributions (differential test against
//!   `shor_ge::distribution`) and the logged TV values.
//! * Theorem A (Gram lemma): TV ≤ δ_rms + θ_rms + δ̄ + √(δ̄ θ̄) for every
//!   admissible reference family (shifted squares, majority sets).
//! * Lemma D (faithfulness / re-absorption): a branch is congruent at the
//!   end iff every *looked-up* register was in range; out-of-range values of
//!   the accumulator `b` are harmless and re-absorbed.
//! * Theorem B (power-of-two order): Φ_E depends only on E mod r, the
//!   output law lives on the exact support, TV = Σ_k|f_k^†κf_k/r² − 1/r|/2,
//!   TV = 0 ⇔ class supports disjoint, TV ≤ off(κ)/(2r); and the converse
//!   observation (TV > 0 for every base of non-power-of-two order tested).
//! * Theorem C (additive worst case): δ_E ≤ (#additions)·2^{−c} for the
//!   unshifted square reference, hence TV ≤ 4·A·2^{−c}.
//!
//! Run: `cargo test --release --test theory_coset` (≈ 1 min).

#[path = "theory_coset_model/mod.rs"]
mod model;
use model::*;
use qsim_lab::shor::ge::{self as shor_ge, ExpReg, GeOpts};
use qsim_lab::shor::mbu::MbuOpts;

// ------------------------------------------------------------- the model

#[test]
fn model_matches_gate_level_engine() {
    // exact whole distributions from the gate-level engine vs the
    // permutation model; the two index y differently (bit order), so we
    // compare the sorted probability vectors and the TV to the exact law
    for &(nm, a, we, c) in &[
        (21u64, 2u64, 2usize, 1usize),
        (21, 2, 1, 2),
        (15, 2, 2, 2),
        (33, 5, 2, 1),
        (35, 2, 2, 1),
    ] {
        let n = work_bits(nm);
        let regs = [ExpReg {
            len: 2 * n,
            base: a,
        }];
        let o = |c| GeOpts {
            we,
            wm: 2,
            mbu: MbuOpts::ALL,
            coset: c,
        };
        let pc = shor_ge::distribution(nm, &regs, &o(c), 0.0);
        let pe = shor_ge::distribution(nm, &regs, &o(0), 0.0);
        let mc = Model::new(nm, a, we, 2, c);
        let me = Model::new(nm, a, we, 2, 0);
        let qc = distribution(mc.t, &mc.supports());
        let qe = distribution(me.t, &me.supports());
        let sorted = |v: &[f64]| {
            let mut s = v.to_vec();
            s.sort_by(|x, y| x.partial_cmp(y).unwrap());
            s
        };
        for (x, y) in sorted(&pc).iter().zip(sorted(&qc)) {
            assert!(
                (x - y).abs() < 1e-9,
                "N={nm} c={c}: sorted coset laws differ"
            );
        }
        for (x, y) in sorted(&pe).iter().zip(sorted(&qe)) {
            assert!((x - y).abs() < 1e-9, "N={nm}: sorted exact laws differ");
        }
        let (t1, t2) = (tv(&pc, &pe), tv(&qc, &qe));
        eprintln!("N={nm} a={a} we={we} c={c}: TV engine {t1:.9} model {t2:.9}");
        assert!((t1 - t2).abs() < 1e-9);
    }
}

#[test]
fn model_reproduces_logged_tv() {
    // research/data/ge-shor/coset_exact.log (gate-level engine, Mac)
    for &(nm, a, we, c, want) in &[
        (21u64, 2u64, 2usize, 1usize, 0.245411),
        (21, 2, 2, 4, 0.071174),
        (33, 5, 2, 2, 0.439084),
        (55, 2, 2, 1, 0.754),
        (55, 2, 2, 3, 0.343),
        (51, 2, 2, 3, 0.0),
    ] {
        let got = tv_coset(nm, a, we, 2, c);
        assert!(
            (got - want).abs() < 1.5e-3,
            "N={nm} c={c} got {got} want {want}"
        );
    }
}

// ------------------------------------------------------- Theorem A (Gram)

#[test]
fn theorem_a_gram_lemma_bounds_tv() {
    for &(nm, a, we, wm, c) in &[
        (21u64, 2u64, 2usize, 2usize, 1usize),
        (21, 2, 2, 2, 3),
        (21, 2, 1, 2, 4),
        (33, 5, 2, 2, 2),
        (35, 2, 2, 3, 3),
        (55, 2, 2, 2, 1),
        (55, 2, 2, 2, 3),
        (55, 2, 2, 1, 2),
        (65, 2, 2, 2, 2),
        (51, 2, 2, 2, 2),
    ] {
        let mc = Model::new(nm, a, we, wm, c);
        let me = Model::new(nm, a, we, wm, 0);
        let supp = mc.supports();
        let cls = mc.classes();
        let tvv = tv(
            &distribution(mc.t, &supp),
            &distribution(me.t, &me.supports()),
        );
        let k = mc.chunks() as i64 + 1;
        let mut best = f64::INFINITY;
        for ox in -k..=k {
            for ob in -k..=k {
                let s = mc.lemma_square(&supp, &cls, ox, ob);
                assert!(
                    tvv <= s.bound + 1e-12,
                    "N={nm} c={c} ({ox},{ob}): TV {tvv} > bound {}",
                    s.bound
                );
                assert!(s.theta_mean <= s.delta_mean + 1e-12);
                best = best.min(s.bound);
            }
        }
        if let Some(s) = mc.lemma_majority(&supp, &cls) {
            assert!(
                tvv <= s.bound + 1e-12,
                "N={nm} c={c} majority: TV {tvv} > {}",
                s.bound
            );
            best = best.min(s.bound);
        }
        eprintln!("N={nm} a={a} we={we} wm={wm} c={c}: TV={tvv:.5} best lemma bound={best:.5} ratio={:.2}", tvv / best.max(1e-300));
    }
}

// ------------------------------------------ Lemma D (faithfulness, re-absorption)

#[test]
fn lemma_d_faithful_branches_are_congruent_and_b_wraps_reabsorb() {
    for &(nm, a, we, wm, c) in &[
        (21u64, 2u64, 2usize, 2usize, 3usize),
        (55, 2, 2, 2, 3),
        (65, 2, 1, 3, 2),
        (35, 2, 2, 2, 4),
    ] {
        let m = Model::new(nm, a, we, wm, c);
        let side = 1u64 << c;
        let (mut faithful, mut reabsorbed, mut total) = (0u64, 0u64, 0u64);
        for e in 0..1u64 << m.t {
            let d = m.digits(e);
            for j in 0..side {
                for j2 in 0..side {
                    let w = m.walk(&d, j, j2);
                    total += 1;
                    if w.faithful {
                        faithful += 1;
                        // congruent at the end: x ≡ g^E, b ≡ 0 as residues of
                        // the signed coset index (b may sit "below zero")
                        assert_eq!(
                            w.x,
                            (w.u as i128 + w.jx as i128 * nm as i128).rem_euclid(m.m() as i128)
                                as u64
                        );
                        assert_eq!(
                            w.b,
                            (w.jb as i128 * nm as i128).rem_euclid(m.m() as i128) as u64
                        );
                        // indices stay within the walk band of Theorem C
                        let dband = ((m.wins.len() as i64 + 1) / 2 + 1) * (m.chunks() as i64 - 1);
                        assert!(w.jx >= -dband && w.jx < side as i64 + dband);
                        if w.b_out.iter().any(|&o| o) {
                            reabsorbed += 1;
                        }
                    } else {
                        // unfaithful ⇔ some looked-up register left [0, L)
                        assert!(w.first_bad.is_some());
                    }
                }
            }
        }
        eprintln!(
            "N={nm} c={c}: faithful {:.4}, faithful but b temporarily out of range (counted 'deviant' by ge-shor) {:.4}",
            faithful as f64 / total as f64,
            reabsorbed as f64 / total as f64
        );
        assert!(reabsorbed > 0, "expected temporarily wrapped accumulators");
    }
}

// ---------------------------------------------- Theorem B (power-of-two r)

#[test]
fn theorem_b_power_of_two_order() {
    let mut cases = 0;
    let mut exact = 0;
    for nm in (5u64..=45).step_by(2) {
        for a in 2..nm - 1 {
            if gcd(a, nm) != 1 {
                continue;
            }
            let r = order(a, nm);
            if !r.is_power_of_two() {
                continue;
            }
            for we in 1..=2 {
                for c in 1..=2 {
                    let mc = Model::new(nm, a, we, 2, c);
                    // (i) Φ_E depends only on E mod r (asserted inside)
                    let (tv_k, off, mism) = mc.pow2_tv(r);
                    assert_eq!(mism, 0);
                    // (ii) the full distribution equals the κ formula and is
                    // supported on multiples of 2^t / r
                    let supp = mc.supports();
                    let p = distribution(mc.t, &supp);
                    let step = (1usize << mc.t) / r as usize;
                    for (y, &py) in p.iter().enumerate() {
                        if y % step != 0 {
                            assert!(py.abs() < 1e-12, "mass off the exact support");
                        }
                    }
                    let me = Model::new(nm, a, we, 2, 0);
                    let tvv = tv(&p, &distribution(me.t, &me.supports()));
                    assert!((tvv - tv_k).abs() < 1e-9);
                    // (iii) TV = 0 ⇔ κ = I, and TV ≤ off/(2r)
                    assert_eq!(tvv < 1e-12, off == 0.0);
                    assert!(tvv <= off / (2.0 * r as f64) + 1e-12);
                    cases += 1;
                    exact += (off == 0.0) as u32;
                }
            }
        }
    }
    eprintln!("power-of-two order: {exact}/{cases} configurations exactly reproduce the exact law");
    assert!(cases > 100);
}

#[test]
fn theorem_b_collision_example_exists() {
    // disjointness is generic but not guaranteed: N = 53, a = 30 (r = 4),
    // w_e = 1, c = 1 has a cross-class collision and TV = 1/32
    let mc = Model::new(53, 30, 1, 2, 1);
    let (tvv, off, _) = mc.pow2_tv(4);
    assert!(
        off > 0.0 && (tvv - 1.0 / 32.0).abs() < 1e-12,
        "TV {tvv} off {off}"
    );
}

#[test]
fn observation_non_power_of_two_order_always_deviates() {
    for nm in (5u64..=35).step_by(2) {
        for a in 2..nm - 1 {
            if gcd(a, nm) != 1 || order(a, nm).is_power_of_two() {
                continue;
            }
            let t = tv_coset(nm, a, 2, 2, 1);
            assert!(t > 1e-6, "N={nm} a={a}: TV = {t}");
        }
    }
}

// ------------------------------------------ Theorem C (additive worst case)

#[test]
fn theorem_c_additive_worst_case() {
    for &(nm, a, we, wm, c) in &[
        (21u64, 2u64, 2usize, 2usize, 3usize),
        (55, 2, 2, 2, 2),
        (55, 2, 1, 3, 4),
        (65, 2, 2, 2, 3),
    ] {
        let full = Model::new(nm, a, we, wm, c);
        let k_add = full.chunks();
        let side = 1i64 << c;
        // every prefix of the schedule is itself a circuit: check
        // δ_E(prefix) ≤ (additions so far)·2^{-c} for every E and prefix
        let mut worst: f64 = 0.0;
        for e in 0..1u64 << full.t {
            let d = full.digits(e);
            let mut bad = vec![0usize; full.wins.len()];
            for j in 0..side as u64 {
                for j2 in 0..side as u64 {
                    let w = full.walk(&d, j, j2);
                    for (k, &(jx, jb)) in w.traj.iter().enumerate() {
                        if !((0..side).contains(&jx) && (0..side).contains(&jb)) {
                            bad[k] += 1;
                        }
                    }
                }
            }
            for (k, &b) in bad.iter().enumerate() {
                let delta = b as f64 / (side * side) as f64;
                let adds = 2 * k_add * (k + 1);
                let ratio = delta / (adds as f64 / side as f64);
                worst = worst.max(ratio);
                assert!(
                    ratio <= 1.0 + 1e-12,
                    "N={nm} c={c} E={e} window {k}: δ={delta} > A·2^-c"
                );
            }
        }
        let tvv = tv_coset(nm, a, we, wm, c);
        let a_tot = 2 * k_add * full.wins.len();
        assert!(tvv <= 4.0 * a_tot as f64 / side as f64);
        eprintln!(
            "N={nm} c={c}: max δ_E/(A·2^-c) = {worst:.3}; TV={tvv:.4} vs 4A·2^-c = {:.3}",
            4.0 * a_tot as f64 / side as f64
        );
    }
}
