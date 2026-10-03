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
    for noise in [ColorNoise::Cnot(0.0), ColorNoise::Uniform(0.0)] {
        let m = cc.memory(s, rounds, noise);
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
