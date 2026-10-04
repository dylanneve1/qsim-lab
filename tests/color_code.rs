//! Validation of the triangular 6.6.6 colour-code generator.
use qsim_lab::qec::color::{circuit_dem, ColorCode, ColorNoise, KF_SCHEDULE, TRI_OPTIMAL};
use qsim_lab::stabilizer::symphase::{SymPhaseSampler, VarDist};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

fn supports(cc: &ColorCode) -> Vec<Vec<usize>> {
    cc.plaquettes
        .iter()
        .map(|p| p.data.iter().flatten().copied().collect())
        .collect()
}

#[test]
fn sizes_and_commutation() {
    for d in [3usize, 5, 7, 9, 11] {
        let cc = ColorCode::new(d);
        assert_eq!(cc.num_data(), (3 * d * d + 1) / 4, "d={d}");
        assert_eq!(cc.plaquettes.len(), (cc.num_data() - 1) / 2, "d={d}");
        let s = supports(&cc);
        // X_f and Z_g commute for all faces f, g (self-dual CSS: even overlaps)
        for a in &s {
            assert!(a.len() == 4 || a.len() == 6, "weights 4/6 only");
            for b in &s {
                let ov = a.iter().filter(|q| b.contains(q)).count();
                assert_eq!(ov % 2, 0);
            }
        }
        // logical: odd weight d, commutes with every stabilizer
        let l = cc.logical_support();
        assert_eq!(l.len(), d);
        for a in &s {
            assert_eq!(a.iter().filter(|q| l.contains(q)).count() % 2, 0);
        }
        // three colours, each plaquette's neighbours have the other colours
        for (i, a) in s.iter().enumerate() {
            for (j, b) in s.iter().enumerate() {
                if i != j && a.iter().any(|q| b.contains(q)) {
                    assert_ne!(cc.plaquettes[i].color, cc.plaquettes[j].color);
                }
            }
        }
    }
}

/// Code-capacity distance by brute force (minimum weight X error with zero
/// syndrome and odd overlap with the Z logical).
#[test]
fn code_distance_is_d() {
    for d in [3usize, 5] {
        let cc = ColorCode::new(d);
        let s = supports(&cc);
        let n = cc.num_data();
        let l = cc.logical_support();
        let mut best = usize::MAX;
        for mask in 1u64..(1 << n) {
            let w = mask.count_ones() as usize;
            if w >= best {
                continue;
            }
            let ok = s
                .iter()
                .all(|f| f.iter().filter(|&&q| mask >> q & 1 == 1).count() % 2 == 0);
            if ok && l.iter().filter(|&&q| mask >> q & 1 == 1).count() % 2 == 1 {
                best = w;
            }
        }
        assert_eq!(best, d);
    }
}

#[test]
fn kf_schedule_is_collision_free() {
    for d in [3usize, 5, 7, 9, 11, 13] {
        let cc = ColorCode::new(d);
        assert!(
            cc.collisions(&cc.uniform_schedule(KF_SCHEDULE)).is_empty(),
            "d={d}"
        );
    }
}

fn check_deterministic(cc: &ColorCode, s: &qsim_lab::qec::color::ColorSchedule, rounds: usize) {
    for (noise, xb) in [
        (ColorNoise::Cnot(0.0), false),
        (ColorNoise::Uniform(0.0), false),
        (ColorNoise::Cnot(0.0), true),
        (ColorNoise::Uniform(0.0), true),
    ] {
        let m = cc.memory_basis(s, rounds, noise, xb);
        let sets: Vec<Vec<usize>> = m.detectors.iter().chain(&m.observables).cloned().collect();
        let smp = SymPhaseSampler::new(&m.circuit, &m.noise)
            .unwrap()
            .with_parities(&sets);
        assert!(smp.reference().iter().all(|&b| !b));
        assert!(
            smp.groups().iter().all(|g| g.dist != VarDist::Coin),
            "random detector"
        );
        let mut rng = StdRng::seed_from_u64(1);
        for shot in smp.sample(128, &mut rng) {
            assert!(shot.iter().all(|&b| !b));
        }
    }
}

#[test]
fn detectors_are_deterministic_for_any_schedule() {
    let mut rng = StdRng::seed_from_u64(5);
    for d in [3usize, 5, 7] {
        let cc = ColorCode::new(d);
        for rounds in [1, 2, 3] {
            check_deterministic(&cc, &cc.uniform_schedule(KF_SCHEDULE), rounds);
            check_deterministic(&cc, &cc.uniform_schedule([TRI_OPTIMAL; 3]), rounds);
            // random per-plaquette permutations (collisions allowed: the
            // CNOTs of one stabilizer type commute, so still deterministic)
            let s = cc
                .plaquettes
                .iter()
                .map(|_| {
                    let mut v = [1u8, 2, 3, 4, 5, 6];
                    v.shuffle(&mut rng);
                    v
                })
                .collect();
            check_deterministic(&cc, &s, rounds);
        }
    }
}

#[test]
fn noisy_dem_has_no_undetectable_single_logical_fault() {
    for d in [3usize, 5] {
        let cc = ColorCode::new(d);
        let m = cc.memory(
            &cc.uniform_schedule(KF_SCHEDULE),
            d,
            ColorNoise::Uniform(0.001),
        );
        let dem = circuit_dem(&m.circuit, &m.noise, &m.detectors, &m.observables);
        assert!(!dem.is_empty());
        assert!(dem
            .iter()
            .all(|e| !e.detectors.is_empty() || e.observables == 0));
    }
}

/// Exact circuit distances of the Kishony-Fowler schedule reproduce their
/// d_circ = d - floor((d+3)/6) (noisy-CNOT model), with exact counts of
/// minimum-weight logicals; cross-checked against MaxSAT/ILP in
/// research/qec-r4.md.
#[test]
fn kf_circuit_distance_matches_published_formula() {
    for (d, rounds, count) in [
        (3usize, 3usize, None),
        (5, 1, Some(55u64)),
        (5, 5, Some(388)),
        (7, 1, Some(883)),
    ] {
        let cc = ColorCode::new(d);
        let m = cc.memory(
            &cc.uniform_schedule(KF_SCHEDULE),
            rounds,
            ColorNoise::Cnot(0.001),
        );
        let (r, cert) = m.z_distance(u64::MAX, u64::MAX);
        assert!(cert);
        assert_eq!(r.weight, Some(d - (d + 3) / 6), "d={d} rounds={rounds}");
        if let Some(c) = count {
            assert_eq!(r.count, c);
        }
    }
    // a uniform schedule is hook-limited: Lee et al.'s tri-optimal at d=5
    let cc = ColorCode::new(5);
    let m = cc.memory(
        &cc.uniform_schedule([TRI_OPTIMAL; 3]),
        1,
        ColorNoise::Cnot(0.001),
    );
    let (r, _) = m.z_distance(u64::MAX, u64::MAX);
    assert!(r.weight.unwrap() < 4);
}

fn load_schedule(text: &str) -> qsim_lab::qec::color::ColorSchedule {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: Vec<u8> = l.split_whitespace().map(|t| t.parse().unwrap()).collect();
            [v[0], v[1], v[2], v[3], v[4], v[5]]
        })
        .collect()
}

/// The schedule found by the large-neighbourhood search at d = 5
/// (research/qec-r4.md §2.4): collision-free, deterministic detectors, same
/// circuit distance as Kishony-Fowler (4) with 197 instead of 388
/// minimum-weight logicals over 5 rounds.
#[test]
fn lns_d5_schedule_is_valid_and_has_fewer_min_weight_logicals() {
    let cc = ColorCode::new(5);
    let s = load_schedule(include_str!(
        "../research/data/qec-r4/schedules/d5_lns_r5.sched"
    ));
    assert_eq!(s.len(), cc.plaquettes.len());
    assert!(cc.collisions(&s).is_empty());
    check_deterministic(&cc, &s, 5);
    let (r, cert) = cc
        .memory(&s, 5, ColorNoise::Cnot(0.001))
        .z_distance(u64::MAX, u64::MAX);
    assert!(cert);
    assert_eq!((r.weight, r.count), (Some(4), 197));
    let (k, _) = cc
        .memory(
            &cc.uniform_schedule(KF_SCHEDULE),
            5,
            ColorNoise::Cnot(0.001),
        )
        .z_distance(u64::MAX, u64::MAX);
    assert_eq!((k.weight, k.count), (Some(4), 388));
}

/// X-basis memory: same distance as Z-basis for Kishony-Fowler (self-dual
/// code, same schedule in both halves).
#[test]
fn x_basis_memory_distance() {
    for d in [3usize, 5, 7] {
        let cc = ColorCode::new(d);
        let m = cc.memory_basis(
            &cc.uniform_schedule(KF_SCHEDULE),
            1,
            ColorNoise::Cnot(0.001),
            true,
        );
        let (r, cert) = m.z_distance(u64::MAX, u64::MAX);
        assert!(cert);
        assert_eq!(r.weight, Some(d - (d + 3) / 6), "d={d}");
    }
}
