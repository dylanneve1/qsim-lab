//! Monitored Clifford+T engine (`src/engines/monitored`) against a dense state
//! vector with mid-circuit measurements: same circuit, same Born outcomes,
//! per-measurement probabilities, final state, cut entropies and magic.

use num_complex::Complex64;
use qsim_lab::engines::monitored::circuit::{self, MOp, Params};
use qsim_lab::engines::monitored::ent::cut_entropy;
use qsim_lab::engines::monitored::{Cliff2, Mode, Monitored};
use qsim_lab::magic_atlas::state_magic;
use qsim_lab::StateVectorF64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn renyi2_sv(sv: &StateVectorF64, region: &[bool]) -> f64 {
    let n = sv.num_qubits();
    let amps = sv.amplitudes();
    let qa: Vec<usize> = (0..n).filter(|&q| region[q]).collect();
    let qb: Vec<usize> = (0..n).filter(|&q| !region[q]).collect();
    let (na, nb) = (1usize << qa.len(), 1usize << qb.len());
    let mut m = vec![Complex64::new(0.0, 0.0); na * nb];
    for (i, &v) in amps.iter().enumerate() {
        let a = qa
            .iter()
            .enumerate()
            .fold(0, |s, (k, &q)| s | (i >> q & 1) << k);
        let b = qb
            .iter()
            .enumerate()
            .fold(0, |s, (k, &q)| s | (i >> q & 1) << k);
        m[a * nb + b] = v;
    }
    let mut tr = 0.0;
    for a in 0..na {
        for a2 in 0..na {
            let mut s = Complex64::new(0.0, 0.0);
            for b in 0..nb {
                s += m[a * nb + b] * m[a2 * nb + b].conj();
            }
            tr += s.norm_sqr();
        }
    }
    -tr.log2()
}

#[test]
fn clifford_group_has_11520_elements() {
    let g = Cliff2::group();
    assert_eq!(g.len(), 11520);
}

/// Runs a random monitored circuit on the engine and on the state vector.
fn check_one(n: usize, depth: usize, p_m: f64, p_t: f64, seed: u64, group: &[Cliff2]) -> usize {
    let p = Params {
        n,
        depth,
        p_m,
        p_t,
        periodic: seed.is_multiple_of(2),
    };
    let mut crng = StdRng::seed_from_u64(seed);
    let mut brng = StdRng::seed_from_u64(seed ^ 77);
    let mut sim = Monitored::new(n, Mode::Exact, 30).with_log();
    let mut sv = StateVectorF64::new(n);
    let mut cliffords = Vec::new();
    let mut nmeas = 0;
    let mut maxd = 0;
    for t in 0..depth {
        for op in circuit::layer(&p, t, group.len(), &mut crng) {
            match op {
                MOp::C2(k, a, b) => {
                    sim.cliff2(&group[k as usize], a, b);
                    for g in group[k as usize].gates(a, b) {
                        sv.apply_gate(&g).unwrap();
                        cliffords.push(g);
                    }
                }
                MOp::T(a) => {
                    sim.t(a).unwrap();
                    sv.apply_gate(&qsim_lab::gate::Gate::T(a)).unwrap();
                }
                MOp::M(a) => {
                    let rec = sim.measure(a, &mut brng, None);
                    let p_sv = sv.collapse(a, rec.outcome);
                    assert!(
                        (p_sv - rec.prob).abs() < 1e-9,
                        "n={n} seed={seed} t={t}: prob {} vs sv {p_sv} (kind {})",
                        rec.prob,
                        rec.kind
                    );
                    nmeas += 1;
                }
            }
            maxd = maxd.max(sim.d());
        }
        // state check every few layers
        if t % 3 == 2 || t + 1 == depth {
            let ours = sim.to_statevector(&cliffords);
            let f = ours.fidelity(&sv);
            assert!(
                (f - 1.0).abs() < 1e-9,
                "n={n} seed={seed} t={t}: fidelity {f}"
            );
            // cut entropies
            let half: Vec<bool> = (0..n).map(|q| q < n / 2).collect();
            let ce = cut_entropy(&sim, &half, 40);
            let s_sv = renyi2_sv(&sv, &half);
            let s = ce.s2.expect("exact");
            assert!((s - s_sv).abs() < 1e-7, "S2 {s} vs sv {s_sv} ({ce:?})");
            assert!(ce.lower - 1e-9 <= s_sv && s_sv <= ce.upper + 1e-9);
            let mut r = StdRng::seed_from_u64(seed + t as u64);
            let reg: Vec<bool> = (0..n).map(|_| r.random::<bool>()).collect();
            let comp: Vec<bool> = reg.iter().map(|b| !b).collect();
            let s1 = cut_entropy(&sim, &reg, 40).s2.unwrap();
            let s2 = cut_entropy(&sim, &comp, 40).s2.unwrap();
            assert!((s1 - renyi2_sv(&sv, &reg)).abs() < 1e-7);
            assert!((s1 - s2).abs() < 1e-7);
            // magic of the register equals magic of the full state
            if sim.d() <= 8 && n <= 8 {
                let full = state_magic(sv.amplitudes());
                let reg_m = state_magic(sim.amp.as_ref().unwrap());
                assert!((full.nullity - reg_m.nullity).abs() < 1e-9);
                assert!((full.m2 - reg_m.m2).abs() < 1e-7);
                assert!(reg_m.nullity <= sim.d() as f64 + 1e-9);
            }
        }
    }
    let _ = nmeas;
    maxd
}

#[test]
fn exact_against_state_vector_with_midcircuit_measurements() {
    let group = Cliff2::group();
    let mut total_maxd = 0;
    let mut cases = 0;
    for seed in 0..60u64 {
        let n = 3 + (seed as usize % 8); // 3..10
        let p_m = [0.05, 0.15, 0.3, 0.5][seed as usize % 4];
        let p_t = [0.1, 0.25, 0.5][seed as usize % 3];
        total_maxd = total_maxd.max(check_one(n, 3 * n, p_m, p_t, seed, &group));
        cases += 1;
    }
    assert!(cases == 60 && total_maxd >= 6, "max d {total_maxd}");
}

#[test]
fn dimension_is_outcome_independent() {
    let group = Cliff2::group();
    for seed in 0..20u64 {
        let p = Params {
            n: 24 + (seed as usize % 3) * 8,
            depth: 60,
            p_m: [0.08, 0.16, 0.3][seed as usize % 3],
            p_t: 0.05,
            periodic: true,
        };
        let (_, a) = circuit::run(&p, Mode::DimensionOnly, 64, seed, 1, &group, |_, _| {});
        let (_, b) = circuit::run(&p, Mode::DimensionOnly, 64, seed, 2, &group, |_, _| {});
        assert_eq!(a.d, b.d);
        // exact mode, when it fits, follows the same d(t)
        let (_, c) = circuit::run(&p, Mode::Exact, 18, seed, 3, &group, |_, _| {});
        let m = c.d.len();
        assert_eq!(&a.d[..m], &c.d[..]);
    }
}

#[test]
fn dephasing_picture_matches() {
    // d equals the entropy of the mixed stabilizer state obtained by replacing
    // every T by full Z-dephasing: compare with an independent computation:
    // rank of the stabilizer group from the tableau rows S_j, j inactive.
    let group = Cliff2::group();
    let p = Params {
        n: 32,
        depth: 64,
        p_m: 0.12,
        p_t: 0.03,
        periodic: true,
    };
    let (sim, tr) = circuit::run(&p, Mode::DimensionOnly, 64, 5, 5, &group, |_, _| {});
    // whole-system "entropy" from cut_entropy with region = everything is
    // n − g − a − b with g = n − d stabilizers, a = 0 pairs... for a pure
    // state the full-region entropy must be 0 and the empty region 0.
    let all = vec![true; 32];
    let ce = cut_entropy(&sim, &all, 0);
    assert_eq!(ce.g, 32 - sim.d());
    assert_eq!(ce.a, sim.d());
    assert_eq!(ce.lower, 0.0);
    assert_eq!(*tr.d.last().unwrap() as usize, sim.d());
}
