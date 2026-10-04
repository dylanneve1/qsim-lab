//! Validation of the fault-tolerant Shor simulator (`qsim_lab::ft`):
//! the Pauli-frame + logical-vector model against a full physical state
//! vector with real measurements, on small encoded circuits.

use qsim_lab::ft::core::Noise;
use qsim_lab::ft::logical::{Encoded, Logical, MagicMode};
use qsim_lab::ft::machine::FtConfig;
use qsim_lab::ft::shor::{ideal_distribution, run_shor15, NLOG15};

/// Clifford circuits with a deterministic ideal outcome: with identical
/// fault realisations, the frame engine and the dense engine must record the
/// same decoded outcome on every shot (faults keep the state a Pauli
/// image of the ideal one, so the decoded outcome is deterministic).
#[test]
fn frame_equals_dense_per_shot_clifford_with_ec() {
    let cfg = FtConfig::default();
    let mut mism = 0;
    let mut wrong = 0;
    let shots = 40;
    for s in 0..shots {
        let p = 0.03;
        let circ = |l: &mut dyn FnMut(u8)| {
            // prep0 H S S H meas  -> ideal 1 ;  ops encoded as small ints
            for op in [0u8, 1, 2, 2, 1, 9] {
                l(op);
            }
        };
        let mut outs = [false; 2];
        {
            let mut e = Encoded::frame(1, 1, Noise::new(p, 1000 + s), cfg, MagicMode::Raw, s);
            let mut r = false;
            circ(&mut |op| match op {
                0 => e.prep(0, false),
                1 => e.h(0),
                2 => e.s(0),
                _ => r = e.meas(0),
            });
            outs[0] = r;
        }
        {
            let mut e = Encoded::dense(1, 1, 21, Noise::new(p, 1000 + s), cfg, MagicMode::Raw, 77 + s);
            let mut r = false;
            circ(&mut |op| match op {
                0 => e.prep(0, false),
                1 => e.h(0),
                2 => e.s(0),
                _ => r = e.meas(0),
            });
            outs[1] = r;
        }
        if outs[0] != outs[1] {
            mism += 1;
        }
        if !outs[0] {
            wrong += 1;
        }
    }
    assert_eq!(mism, 0, "frame and dense disagree on {mism}/{shots} shots");
    // p = 3% is far above the level-1 pseudo-threshold: many shots fail,
    // which is what makes the per-shot comparison informative.
    eprintln!("per-shot: {mism} mismatches, {wrong}/{shots} logical failures");
    assert!(wrong > 0);
}

fn two_prop_z(a: u32, na: u32, b: u32, nb: u32) -> f64 {
    let (pa, pb) = (a as f64 / na as f64, b as f64 / nb as f64);
    let pp = (a + b) as f64 / (na + nb) as f64;
    let se = (pp * (1.0 - pp) * (1.0 / na as f64 + 1.0 / nb as f64)).sqrt();
    if se == 0.0 {
        0.0
    } else {
        (pa - pb) / se
    }
}

/// Non-Clifford circuits through the T gadget (injection, transversal CNOT,
/// transversal measurement, feed-forward S): output statistics of the frame
/// model vs. the dense physical simulation (EC between gadgets off to keep
/// the dense state at ≤ 16 qubits).
#[test]
fn frame_matches_dense_statistics_t_gadget() {
    let cfg = FtConfig { ec: false, inject_postselect: false };
    let p = 0.02;
    type Circ = fn(&mut dyn Logical) -> bool;
    let circuits: [(Circ, f64); 3] = [
        (|l| {
            l.prep(0, false);
            l.h(0);
            l.t(0);
            l.h(0);
            l.meas(0)
        }, 0.146),
        (|l| {
            l.prep(0, false);
            l.h(0);
            l.t(0);
            l.h(0);
            l.tdg(0);
            l.h(0);
            l.meas(0)
        }, 0.0),
        (|l| {
            l.prep(0, true);
            l.h(0);
            l.tdg(0);
            l.tdg(0);
            l.h(0);
            l.meas(0)
        }, 0.5),
    ];
    for (ci, (c, _)) in circuits.iter().enumerate() {
        let (nf, nd) = (20_000u32, 1_500u32);
        let mut kf = 0;
        for s in 0..nf as u64 {
            let mut e = Encoded::frame(1, 1, Noise::new(p, s), cfg, MagicMode::Raw, s);
            kf += c(&mut e) as u32;
        }
        let mut kd = 0;
        for s in 0..nd as u64 {
            let mut e = Encoded::dense(1, 1, 16, Noise::new(p, 9_000_000 + s), cfg, MagicMode::Raw, s);
            kd += c(&mut e) as u32;
        }
        let z = two_prop_z(kf, nf, kd, nd);
        eprintln!("circuit {ci}: frame {kf}/{nf}  dense {kd}/{nd}  z = {z:.2}");
        assert!(z.abs() < 4.0, "circuit {ci}: z = {z}");
    }
}

/// Noiseless encoded Shor (levels 1, 2) reproduces the ideal distribution.
#[test]
fn encoded_shor15_noiseless() {
    for k in [1usize, 2] {
        let ideal = ideal_distribution(7, 3);
        let n = if k == 1 { 400 } else { 20 };
        let mut hist = [0u32; 8];
        for s in 0..n {
            let mut e = Encoded::frame(k, NLOG15, Noise::new(0.0, s), FtConfig::default(), MagicMode::Raw, s);
            hist[run_shor15(&mut e, 7, 3) as usize] += 1;
        }
        for y in 0..8 {
            if ideal[y] == 0.0 {
                assert_eq!(hist[y], 0, "level {k}: y={y}");
            } else {
                let f = hist[y] as f64 / n as f64;
                let tol = if k == 1 { 0.08 } else { 0.25 };
                assert!((f - ideal[y]).abs() < tol, "level {k}: y={y} freq {f}");
            }
        }
    }
}

/// T-gadget 1-exRec with an ideal magic state: every single fault in the
/// gadget (transversal CNOT, ECs, transversal measurement, S/I slot) must leave
/// the logical state exact (state vector) and the data frame correctable.
#[test]
fn t_gadget_exrec_single_faults() {
    use qsim_lab::ft::machine::ideal_logical;
    for dagger in [false, true] {
        let run = |script: Vec<(u64, u8)>, seed: u64| -> (bool, u64) {
            let mut e = Encoded::frame(1, 1, Noise::scripted(script), FtConfig::default(), MagicMode::Model(0.0), seed);
            e.m.noise.suspended = true;
            e.prep(0, false);
            e.h(0);
            e.m.noise.suspended = false;
            if dagger {
                e.tdg(0)
            } else {
                e.t(0)
            }
            let nloc = e.m.noise.loc;
            let b = e.blocks[0];
            let clean = ideal_logical(&e.m.b.frame, 1, b) == (false, false);
            // ideal: T|+> (or T†|+>) on logical qubit 0, magic slot reset
            let sv = e.sv.as_ref().unwrap();
            let ph = if dagger { -1.0 } else { 1.0 } * std::f64::consts::FRAC_PI_4;
            let w = num_complex::Complex64::from_polar(std::f64::consts::FRAC_1_SQRT_2, ph);
            let r = std::f64::consts::FRAC_1_SQRT_2;
            // amplitudes over (q0, magic q1): magic collapsed to some value
            let mut best = 0.0f64;
            for m in 0..2usize {
                let a0 = sv.a[m << 1];
                let a1 = sv.a[(m << 1) | 1];
                let ov = (a0 * r + a1 * w.conj()).norm_sqr();
                best = best.max(ov);
            }
            (clean && (best - 1.0).abs() < 1e-9, nloc)
        };
        let (ok, nloc) = run(vec![], 1);
        assert!(ok);
        let mut bad = 0;
        for l in 0..nloc {
            for code in 1..=15u8 {
                for seed in 0..2 {
                    if !run(vec![(l, code)], seed).0 {
                        bad += 1;
                    }
                }
            }
        }
        assert_eq!(bad, 0, "dagger={dagger}: {bad} failing single faults of {}", nloc * 30);
    }
}

/// The clean-run estimator: in runs without any logical-level fault the
/// output must follow the ideal distribution exactly (the location structure
/// does not depend on outcomes, so the fault flag is independent of them).
#[test]
fn clean_runs_follow_ideal_distribution() {
    use qsim_lab::ft::logical::{Checked, Unencoded};
    let ideal = ideal_distribution(7, 3);
    for enc in [true, false] {
        let mut h = [0f64; 8];
        let mut nclean = 0f64;
        let mut nf = 0;
        for s in 0..30_000u64 {
            let (y, faulty) = if enc {
                let mut c = Checked(Encoded::frame(1, NLOG15, Noise::new(1e-3, s), FtConfig::default(), MagicMode::Raw, s));
                let y = run_shor15(&mut c, 7, 3);
                (y, c.0.counts.logical_fault)
            } else {
                let mut u = Unencoded::new(NLOG15, Noise::new(1e-2, s), false, s);
                let y = run_shor15(&mut u, 7, 3);
                (y, u.noise.faults.iter().sum::<u64>() > 0)
            };
            if !faulty {
                h[y as usize] += 1.0;
                nclean += 1.0;
            } else {
                nf += 1;
            }
        }
        let mut chi2 = 0.0;
        for y in 0..8 {
            if ideal[y] == 0.0 {
                assert_eq!(h[y], 0.0, "enc={enc}: clean run gave off-peak y={y}");
            } else {
                let e = nclean * ideal[y];
                chi2 += (h[y] - e).powi(2) / e;
            }
        }
        eprintln!("enc={enc}: clean {nclean}, faulty {nf}, chi2(3 dof) = {chi2:.2}");
        assert!(nf > 1000);
        assert!(chi2 < 16.3, "chi2 = {chi2} (p < 0.001)");
    }
}
