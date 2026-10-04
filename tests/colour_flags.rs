//! Flag-qubit colour-code circuits (research/qec/colour-flags.md): the hook-free
//! boundary construction of research/qec/colour-global.md §6 built as real
//! circuits (one flag qubit per boundary-touching plaquette, both halves).
use qsim_lab::qec::color::{circuit_dem, ColorCode, ColorNoise, ColorSchedule, KF_SCHEDULE};

fn load(text: &str) -> (ColorSchedule, Vec<bool>) {
    let mut s = Vec::new();
    let mut f = Vec::new();
    for l in text.lines().filter(|l| !l.trim().is_empty()) {
        let t: Vec<&str> = l.split_whitespace().collect();
        let v: Vec<u8> = t[..6].iter().map(|x| x.parse().unwrap()).collect();
        s.push([v[0], v[1], v[2], v[3], v[4], v[5]]);
        f.push(t.get(6) == Some(&"F"));
    }
    (s, f)
}

fn hf(d: usize) -> (ColorCode, ColorSchedule, Vec<bool>) {
    let text = match d {
        5 => include_str!("../research/data/colour-flags/schedules/d5_hf_T6.sched"),
        7 => include_str!("../research/data/colour-flags/schedules/d7_hf_T6.sched"),
        9 => include_str!("../research/data/colour-flags/schedules/d9_hf_T7.sched"),
        _ => unreachable!(),
    };
    let cc = ColorCode::new(d);
    let (s, f) = load(text);
    assert_eq!(s.len(), cc.plaquettes.len());
    (cc, s, f)
}

#[test]
fn flagged_plaquettes_are_the_boundary_ones() {
    for d in [5, 7, 9] {
        let (cc, s, f) = hf(d);
        assert_eq!(f, cc.boundary_plaquettes(), "d={d}");
        assert_eq!(f.iter().filter(|&&b| b).count(), 3 * d - 6);
        assert!(cc.collisions(&s).is_empty());
    }
}

#[test]
fn no_flags_is_the_unflagged_circuit() {
    let cc = ColorCode::new(5);
    let s = cc.uniform_schedule(KF_SCHEDULE);
    for xb in [false, true] {
        let a = cc.memory_basis(&s, 2, ColorNoise::Uniform(0.001), xb);
        let b = cc.memory_flagged(&s, &vec![false; s.len()], 2, ColorNoise::Uniform(0.001), xb);
        assert_eq!(
            format!("{:?}", a.circuit.ops),
            format!("{:?}", b.circuit.ops)
        );
        assert_eq!(a.detectors, b.detectors);
        assert_eq!(a.observables, b.observables);
    }
}

/// Flag windows: the first flag CNOT precedes the plaquette's 2nd data CNOT,
/// the second follows its (w-1)-th, both at steps where the auxiliary is idle.
#[test]
fn flag_slots_cover_every_multi_qubit_hook() {
    for d in [5, 7, 9] {
        let (cc, s, f) = hf(d);
        let t = cc.num_steps(&s) as u8;
        for (pi, sl) in cc.flag_slots(&s, &f).iter().enumerate() {
            assert_eq!(sl.is_some(), f[pi]);
            if let Some((s1, s2)) = sl {
                let p = &cc.plaquettes[pi];
                let mut ts: Vec<u8> = (0..6)
                    .filter(|&k| p.data[k].is_some())
                    .map(|k| s[pi][k])
                    .collect();
                ts.sort_unstable();
                let w = ts.len();
                assert!(*s1 < ts[1] && *s2 > ts[w - 2] && *s2 <= t + 1);
                assert!(!ts.contains(s1) && !ts.contains(s2));
            }
        }
    }
}

#[test]
fn resources() {
    let (cc, s, f) = hf(9);
    let r = cc.resources(&s, &f);
    assert_eq!((r.data, r.aux, r.flags), (61, 30, 21));
    assert_eq!(r.cnot_layers, 18); // 7 data layers + 1 pre + 1 post layer per half
    assert_eq!(r.cnots, 396); // K-F: 312
    let k = cc.resources(&cc.uniform_schedule(KF_SCHEDULE), &[]);
    assert_eq!((k.flags, k.cnot_layers, k.cnots), (0, 12, 312));
    for d in [5, 7] {
        let (cc, s, f) = hf(d);
        assert_eq!(cc.resources(&s, &f).cnot_layers, 16);
    }
}

/// Every detector (flags included) is deterministic, both bases, both noise
/// models, and noiseless shots are all zero (circuit_dem panics otherwise).
#[test]
fn flagged_detectors_are_deterministic() {
    for d in [5, 7] {
        let (cc, s, f) = hf(d);
        let kf = cc.uniform_schedule(KF_SCHEDULE);
        let bf = cc.boundary_plaquettes();
        for (sched, flags) in [(&s, &f), (&kf, &bf)] {
            for rounds in [1, 2] {
                for xb in [false, true] {
                    for noise in [ColorNoise::Cnot(0.001), ColorNoise::Uniform(0.001)] {
                        let m = cc.memory_flagged(sched, flags, rounds, noise, xb);
                        let nf = flags.iter().filter(|&&b| b).count();
                        assert_eq!(
                            m.flag_detector.iter().filter(|&&b| b).count(),
                            2 * nf * rounds
                        );
                        let dem = circuit_dem(&m.circuit, &m.noise, &m.detectors, &m.observables);
                        assert!(!dem.is_empty());
                        // no single fault is an undetectable logical
                        assert!(dem
                            .iter()
                            .all(|e| !e.detectors.is_empty() || e.observables == 0));
                    }
                }
            }
        }
    }
}

fn dist(cc: &ColorCode, s: &ColorSchedule, f: &[bool], rounds: usize, xb: bool) -> usize {
    let m = cc.memory_flagged(s, f, rounds, ColorNoise::Cnot(0.001), xb);
    let (r, cert) = m.z_distance(1, u64::MAX);
    assert!(cert);
    r.weight.unwrap()
}

/// Full circuit distance d with flagged boundary plaquettes (K-F: d - 1 = 4).
#[test]
fn d5_flagged_reaches_full_distance() {
    let (cc, s, f) = hf(5);
    for xb in [false, true] {
        assert_eq!(dist(&cc, &s, &f, 5, xb), 5);
    }
    // K-F's own schedule with the same flags also reaches 5 at d = 5
    assert_eq!(
        dist(
            &cc,
            &cc.uniform_schedule(KF_SCHEDULE),
            &cc.boundary_plaquettes(),
            5,
            false
        ),
        5
    );
}

#[test]
fn d7_flagged_reaches_full_distance() {
    let (cc, s, f) = hf(7);
    for xb in [false, true] {
        assert_eq!(dist(&cc, &s, &f, 1, xb), 7);
    }
    // K-F's schedule + boundary flags: only 6
    assert_eq!(
        dist(
            &cc,
            &cc.uniform_schedule(KF_SCHEDULE),
            &cc.boundary_plaquettes(),
            1,
            false
        ),
        6
    );
}

/// 7 rounds, both bases (~35 s each in release).
#[test]
#[ignore]
fn d7_flagged_full_rounds() {
    let (cc, s, f) = hf(7);
    for xb in [false, true] {
        assert_eq!(dist(&cc, &s, &f, 7, xb), 7);
    }
}

/// d = 9 (~100 s per basis in release): 9 with flags and a 7th data layer
/// (K-F: 7; best unflagged: 8).
#[test]
#[ignore]
fn d9_flagged_reaches_full_distance() {
    let (cc, s, f) = hf(9);
    for xb in [false, true] {
        assert_eq!(dist(&cc, &s, &f, 1, xb), 9);
    }
}
