//! The .stim writer/reader: round trips must give *identical* sampling
//! distributions (checked exactly, not statistically, by comparing the
//! canonical detector error structure of the SymPhase samplers).
use qsim_lab::qec::surface::SurfaceCode;
use qsim_lab::stabilizer::symphase::{SymPhaseSampler, VarDist};
use qsim_lab::stim_io::{parse_stim, to_stim};
use qsim_lab::{Circuit, Gate, NoiseModel, Op};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Canonical form of a sampler: the multiset of variable groups, each as its
/// list of (outcome probability bits, flipped-row set), plus the reference.
fn canonical(s: &SymPhaseSampler) -> (Vec<bool>, Vec<String>) {
    let mut cols: Vec<Vec<usize>> = vec![Vec::new(); s.num_vars()];
    for j in 0..s.num_measurements() {
        for &v in s.row(j) {
            cols[v as usize].push(j);
        }
    }
    let mut groups: Vec<String> = s
        .groups()
        .iter()
        .map(|g| {
            let mut outs: Vec<(String, Vec<usize>)> = g
                .dist
                .outcomes()
                .into_iter()
                .map(|(pat, p)| {
                    let mut sig: Vec<usize> = Vec::new();
                    for k in 0..g.dist.len() {
                        if pat >> k & 1 == 1 {
                            for &r in &cols[g.first as usize + k] {
                                if let Ok(i) = sig.binary_search(&r) {
                                    sig.remove(i);
                                } else {
                                    let i = sig.binary_search(&r).unwrap_err();
                                    sig.insert(i, r);
                                }
                            }
                        }
                    }
                    (format!("{p:.15e}"), sig)
                })
                .collect();
            outs.sort();
            let kind = match g.dist {
                VarDist::Coin => "coin",
                _ => "fault",
            };
            format!("{kind}:{outs:?}")
        })
        .collect();
    groups.sort();
    (s.reference().to_vec(), groups)
}

fn sampler(
    c: &Circuit,
    noise: &NoiseModel,
    dets: &[Vec<usize>],
    obs: &[Vec<usize>],
) -> SymPhaseSampler {
    let sets: Vec<Vec<usize>> = dets.iter().chain(obs).cloned().collect();
    SymPhaseSampler::new(c, noise).unwrap().with_parities(&sets)
}

#[test]
fn surface_code_round_trip_is_exactly_the_same_circuit() {
    for d in [3, 5] {
        let sc = SurfaceCode::new(d, d);
        let c = sc.build_circuit();
        let noise = NoiseModel::circuit_level(0.003, 0.002);
        let dets = sc.detector_records();
        let obs = vec![sc.observable_records()];
        let text = to_stim(&c, &noise, &dets, &obs).unwrap();
        let prog = parse_stim(&text).unwrap();
        assert_eq!(prog.detectors, dets);
        assert_eq!(prog.observables, obs);
        assert_eq!(prog.noise.p_meas, 0.002);
        let a = sampler(&c, &noise, &dets, &obs);
        let b = sampler(
            &prog.circuit,
            &prog.noise,
            &prog.detectors,
            &prog.observables,
        );
        assert!(
            a.reference().iter().all(|&r| !r),
            "detectors must be deterministic"
        );
        assert_eq!(canonical(&a), canonical(&b), "d={d}");
        // and the parsed program re-serialises to the same text
        assert_eq!(
            to_stim(
                &prog.circuit,
                &prog.noise,
                &prog.detectors,
                &prog.observables
            )
            .unwrap(),
            text
        );
    }
}

#[test]
fn random_clifford_circuits_round_trip_exactly() {
    let mut rng = StdRng::seed_from_u64(11);
    for _ in 0..200 {
        let n = rng.random_range(1..6usize);
        let mut c = Circuit::new(n);
        let mut m = 0;
        for _ in 0..rng.random_range(1..40) {
            let a = rng.random_range(0..n);
            let mut b = rng.random_range(0..n);
            if n > 1 {
                while b == a {
                    b = rng.random_range(0..n);
                }
            }
            let op = match rng.random_range(0..13) {
                0 => Op::Gate(Gate::H(a)),
                1 => Op::Gate(Gate::S(a)),
                2 => Op::Gate(Gate::Sdg(a)),
                3 => Op::Gate(Gate::X(a)),
                4 if n > 1 => Op::Gate(Gate::Cnot(a, b)),
                5 if n > 1 => Op::Gate(Gate::Cz(a, b)),
                6 if n > 1 => Op::Gate(Gate::Swap(a, b)),
                7 => {
                    m += 1;
                    Op::Measure(a)
                }
                8 => Op::Reset(a),
                9 => Op::XFlip(a, 0.1),
                10 => Op::ZFlip(a, 0.07),
                11 => Op::Depolarize1q(a, 0.05),
                12 if n > 1 => Op::Depolarize2q(a, b, 0.03),
                _ => Op::Gate(Gate::Y(a)),
            };
            c.ops.push(op);
        }
        let noise = NoiseModel {
            p_1q: 0.01,
            p_2q: 0.02,
            p_meas: if rng.random_bool(0.5) { 0.04 } else { 0.0 },
            p_reset: 0.03,
        };
        // raw measurements (not necessarily deterministic)
        let dets: Vec<Vec<usize>> = (0..m).map(|j| vec![j]).collect();
        let text = to_stim(&c, &noise, &dets, &[]).unwrap();
        let prog = parse_stim(&text).unwrap();
        let a = sampler(&c, &noise, &dets, &[]);
        let b = sampler(
            &prog.circuit,
            &prog.noise,
            &prog.detectors,
            &prog.observables,
        );
        assert_eq!(canonical(&a), canonical(&b), "{text}");
    }
}

#[test]
fn stim_generated_rotated_memory_parses_with_deterministic_detectors() {
    let text = include_str!("data/stim_rotated_memory_z_d3_p0.003.stim");
    let prog = parse_stim(text).unwrap();
    assert_eq!(prog.detectors.len(), 24);
    assert_eq!(prog.observables.len(), 1);
    assert_eq!(prog.circuit.num_qubits, 26);
    let s = sampler(
        &prog.circuit,
        &prog.noise,
        &prog.detectors,
        &prog.observables,
    );
    assert!(s.reference().iter().all(|&r| !r));
    // no coin (random) variable reaches a detector: all detectors deterministic
    assert!(s.groups().iter().all(|g| g.dist != VarDist::Coin));
    // noiseless version: every shot is all-zero
    let clean: String = text
        .lines()
        .filter(|l| !l.contains("ERROR") && !l.contains("DEPOLARIZE"))
        .collect::<Vec<_>>()
        .join("\n");
    let p = parse_stim(&clean).unwrap();
    let s = sampler(&p.circuit, &p.noise, &p.detectors, &p.observables);
    let mut rng = StdRng::seed_from_u64(3);
    for shot in s.sample(256, &mut rng) {
        assert!(shot.iter().all(|&b| !b));
    }
}
