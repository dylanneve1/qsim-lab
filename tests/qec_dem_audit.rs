//! Audit: does the surface-code "fast detector sampler" reproduce the
//! detector statistics of full circuit-level tableau simulation?
//!
//! Both samplers are run on the same d=3, rounds=3 memory experiment with
//! the same `NoiseModel`. For each detector we compare firing rates with a
//! two-proportion z-test (Bonferroni-corrected over all detectors at an
//! overall false-alarm rate ~1e-4), and the same for the uncorrected
//! logical-flip and decoded logical error rates.
//!
//! Copied from the audit branch (`exp/audit`). The original drove the fast
//! sampler through the public `error_mechanisms` list of main @ 86e5d67 (a
//! hand-written phenomenological model) and FAILED there. That API is gone;
//! `fast_stats` now drives the circuit-derived sampler
//! (`SurfaceCode::dem_sampler`), which is what
//! `SamplingMethod::DetectorErrorModel` uses. Everything else is unchanged.
//!
//! It is `#[ignore]`d because it is slow (20k full-tableau shots); run it with
//! `cargo test --release --test qec_dem_audit -- --ignored --nocapture`.
//! `QSIM_DEM_SHOTS` overrides the shot count, `QSIM_DEM_D` the distance
//! (rounds = d), `QSIM_DEM_P` the physical error rate.

use qsim_lab::noise::NoiseModel;
use qsim_lab::qec::SurfaceCode;
use qsim_lab::Tableau;
use rand::rngs::StdRng;
use rand::SeedableRng;

struct Stats {
    det: Vec<usize>,
    raw_logical: usize,
    decoded_errors: usize,
}

fn tableau_stats(sc: &SurfaceCode, noise: &NoiseModel, shots: usize, rng: &mut StdRng) -> Stats {
    let circuit = sc.build_circuit();
    let nq = SurfaceCode::total_qubits(sc.d);
    let nd = sc.decoder.graph.num_nodes;
    let mut s = Stats {
        det: vec![0; nd],
        raw_logical: 0,
        decoded_errors: 0,
    };
    for _ in 0..shots {
        let mut t = Tableau::new(nq);
        let bits = circuit.run_noisy(&mut t, noise, rng).unwrap();
        let (defects, raw) = sc.extract_z_defects(&bits);
        for &d in &defects {
            s.det[d] += 1;
        }
        s.raw_logical += raw as usize;
        s.decoded_errors += (raw ^ sc.decoder.decode(&defects)) as usize;
    }
    s
}

fn fast_stats(sc: &SurfaceCode, noise: &NoiseModel, shots: usize, rng: &mut StdRng) -> Stats {
    let nd = sc.decoder.graph.num_nodes;
    let mut s = Stats {
        det: vec![0; nd],
        raw_logical: 0,
        decoded_errors: 0,
    };
    // The circuit-derived DEM sampler (every noise location, every Pauli).
    let sampler = sc.dem_sampler(noise);
    for _ in 0..shots {
        let (defects, flip) = sampler.sample(rng);
        for &d in &defects {
            s.det[d] += 1;
        }
        s.raw_logical += flip as usize;
        s.decoded_errors += (flip ^ sc.decoder.decode(&defects)) as usize;
    }
    s
}

/// two-proportion z statistic
fn z(a: usize, na: usize, b: usize, nb: usize) -> f64 {
    let (pa, pb) = (a as f64 / na as f64, b as f64 / nb as f64);
    let p = (a + b) as f64 / (na + nb) as f64;
    let se = (p * (1.0 - p) * (1.0 / na as f64 + 1.0 / nb as f64)).sqrt();
    if se == 0.0 {
        if pa == pb {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (pa - pb) / se
    }
}

#[test]
#[ignore = "slow acceptance test; run explicitly with --ignored"]
fn fast_detector_sampler_matches_tableau_d3() {
    let shots: usize = std::env::var("QSIM_DEM_SHOTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);
    let env = |k: &str| std::env::var(k).ok();
    let p: f64 = env("QSIM_DEM_P")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.005);
    let d: usize = env("QSIM_DEM_D").and_then(|s| s.parse().ok()).unwrap_or(3);
    let noise = NoiseModel::circuit_level(p, p);
    let sc = SurfaceCode::new(d, d);
    let mut rng = StdRng::seed_from_u64(7);
    let tab = tableau_stats(&sc, &noise, shots, &mut rng);
    let fast = fast_stats(&sc, &noise, shots, &mut rng);
    let nd = tab.det.len();
    // Bonferroni over nd detectors + 2 logical rates at total alpha 1e-4
    let zcrit = 4.9;
    let mut worst = 0.0f64;
    let mut failures = Vec::new();
    println!("det  tableau_rate  fast_rate  z");
    for d in 0..nd {
        let zz = z(tab.det[d], shots, fast.det[d], shots);
        println!(
            "{d:3}  {:.5}  {:.5}  {zz:+.1}",
            tab.det[d] as f64 / shots as f64,
            fast.det[d] as f64 / shots as f64
        );
        worst = worst.max(zz.abs());
        if zz.abs() > zcrit {
            failures.push(d);
        }
    }
    let zl = z(tab.raw_logical, shots, fast.raw_logical, shots);
    let ze = z(tab.decoded_errors, shots, fast.decoded_errors, shots);
    println!(
        "raw logical flip: tableau {:.5} fast {:.5} z={zl:+.1}",
        tab.raw_logical as f64 / shots as f64,
        fast.raw_logical as f64 / shots as f64
    );
    println!(
        "decoded logical error: tableau {:.5} fast {:.5} z={ze:+.1}",
        tab.decoded_errors as f64 / shots as f64,
        fast.decoded_errors as f64 / shots as f64
    );
    assert!(
        failures.is_empty() && zl.abs() < zcrit && ze.abs() < zcrit,
        "detector model disagrees with circuit: {} detectors beyond {zcrit}σ (worst {worst:.1}σ), logical z={zl:.1}, decoded z={ze:.1}",
        failures.len()
    );
}
