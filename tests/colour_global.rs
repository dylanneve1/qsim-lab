//! The globally searched d = 9 colour-code schedule (research/colour-global.md):
//! inside Kishony-Fowler's design space (one auxiliary per plaquette, the same
//! 6-step schedule for the X and Z halves, collision-free) it reaches circuit
//! distance 8 = d - 1, one more than Kishony-Fowler's d - floor((d+3)/6) = 7.
use qsim_lab::qec::color::{ColorCode, ColorNoise, ColorSchedule, KF_SCHEDULE};

fn load_schedule(text: &str) -> ColorSchedule {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: Vec<u8> = l.split_whitespace().map(|t| t.parse().unwrap()).collect();
            [v[0], v[1], v[2], v[3], v[4], v[5]]
        })
        .collect()
}

fn d9() -> (ColorCode, ColorSchedule) {
    let cc = ColorCode::new(9);
    let s = load_schedule(include_str!(
        "../research/data/colour-global/schedules/d9_global_D8.sched"
    ));
    (cc, s)
}

#[test]
fn d9_global_schedule_is_in_kf_design_space() {
    let (cc, s) = d9();
    assert_eq!(s.len(), cc.plaquettes.len());
    assert!(cc.collisions(&s).is_empty(), "{:?}", cc.collisions(&s));
    // every present position uses a step in 1..=6, so the round has 6 + 6 CNOT layers
    for (p, row) in cc.plaquettes.iter().zip(&s) {
        for k in 0..6 {
            if p.data[k].is_some() {
                assert!((1..=6).contains(&row[k]));
            }
        }
    }
    // detectors deterministic, no random variable reaches a detector
    let m = cc.memory(&s, 2, ColorNoise::Cnot(0.001));
    let _ = qsim_lab::qec::color::circuit_dem(&m.circuit, &m.noise, &m.detectors, &m.observables);
}

/// One round is an upper bound on the multi-round distance and is cheap; the
/// 3- and 9-round values (8, certified, both bases) are in the write-up.
#[test]
fn d9_global_schedule_beats_kf_circuit_distance() {
    let (cc, s) = d9();
    for x_basis in [false, true] {
        let m = cc.memory_basis(&s, 1, ColorNoise::Cnot(0.001), x_basis);
        let (r, cert) = m.z_distance(1, u64::MAX);
        assert!(cert);
        assert_eq!(r.weight, Some(8), "x_basis={x_basis}");
    }
    let m = cc.memory(
        &cc.uniform_schedule(KF_SCHEDULE),
        1,
        ColorNoise::Cnot(0.001),
    );
    let (k, _) = m.z_distance(1, u64::MAX);
    assert_eq!(k.weight, Some(7));
}

/// d = 11: Kishony-Fowler's 6 + 6-layer space cannot exceed their d_circ = 9
/// (DRAT-verified UNSAT, research/colour-global.md §6), but one extra CNOT
/// layer per half (steps 1..=7, still one schedule for both halves and
/// collision-free) reaches 10 = d - 1.
#[test]
fn d11_seven_layer_schedule_is_valid() {
    let cc = ColorCode::new(11);
    let s = load_schedule(include_str!(
        "../research/data/colour-global/schedules/d11_T7_D10.sched"
    ));
    assert_eq!(s.len(), cc.plaquettes.len());
    assert!(cc.collisions(&s).is_empty());
    let max = cc
        .plaquettes
        .iter()
        .zip(&s)
        .flat_map(|(p, row)| (0..6).filter(|&k| p.data[k].is_some()).map(|k| row[k]))
        .max();
    assert_eq!(max, Some(7));
    let m = cc.memory(&s, 2, ColorNoise::Cnot(0.001));
    let _ = qsim_lab::qec::color::circuit_dem(&m.circuit, &m.noise, &m.detectors, &m.observables);
}

/// About 2.5 minutes per basis in release mode: `cargo test --release -- --ignored`.
#[test]
#[ignore]
fn d11_seven_layer_schedule_has_distance_10() {
    let cc = ColorCode::new(11);
    let s = load_schedule(include_str!(
        "../research/data/colour-global/schedules/d11_T7_D10.sched"
    ));
    for x_basis in [false, true] {
        let m = cc.memory_basis(&s, 1, ColorNoise::Cnot(0.001), x_basis);
        let (r, cert) = m.z_distance(1, u64::MAX);
        assert!(cert);
        assert_eq!(r.weight, Some(10), "x_basis={x_basis}");
    }
}
