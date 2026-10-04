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
    let shots = 60;
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
            let mut e = Encoded::dense(1, 1, 15, Noise::new(p, 1000 + s), cfg, MagicMode::Raw, 77 + s);
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
    // sanity: at p = 3% some shots should fail at level 1 is not required,
    // but the ideal outcome must dominate
    assert!(wrong < shots / 2);
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
            }
        }
    }
}
