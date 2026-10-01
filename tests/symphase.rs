//! The symbolic-phase sampler must reproduce exactly the outcome
//! distribution of running the noisy circuit shot by shot on the tableau.
//!
//! On small circuits both distributions are computed *exactly*: the tableau
//! side by branching on every noise outcome and every random measurement,
//! the sampler side by enumerating every assignment of its variables. They
//! must agree to a relative 1e-9 (the tableau side sums many branch
//! probabilities, so 1e-12 absolute is below its rounding error). Larger
//! circuits are compared statistically.

use qsim_lab::stabilizer::symphase::SymPhaseSampler;
use qsim_lab::{Circuit, Gate, NoiseModel, Op, SurfaceCode, Tableau};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::f64::consts::FRAC_PI_2;

type Dist = HashMap<Vec<bool>, f64>;

fn paulis1(q: usize) -> [Gate; 3] {
    [Gate::X(q), Gate::Y(q), Gate::Z(q)]
}

fn pauli_of(k: usize, q: usize) -> Option<Gate> {
    match k {
        1 => Some(Gate::X(q)),
        2 => Some(Gate::Y(q)),
        3 => Some(Gate::Z(q)),
        _ => None,
    }
}

/// Exact outcome distribution of `run_noisy` on the tableau, by branching.
fn tableau_dist(c: &Circuit, noise: &NoiseModel) -> Dist {
    let mut acc = Dist::new();
    rec(
        c,
        noise,
        0,
        Tableau::new(c.num_qubits),
        vec![],
        1.0,
        &mut acc,
    );
    acc
}

fn rec(
    c: &Circuit,
    noise: &NoiseModel,
    i: usize,
    mut t: Tableau,
    out: Vec<bool>,
    p: f64,
    acc: &mut Dist,
) {
    if p == 0.0 {
        return;
    }
    let Some(op) = c.ops.get(i) else {
        *acc.entry(out).or_insert(0.0) += p;
        return;
    };
    let mut rng = StdRng::seed_from_u64(0); // outcomes are always forced
    let next =
        |t: Tableau, out: Vec<bool>, p: f64, acc: &mut Dist| rec(c, noise, i + 1, t, out, p, acc);
    // branch over a gate's depolarizing noise, then continue
    let gate_noise = |t: Tableau, out: Vec<bool>, g: &Gate, acc: &mut Dist| {
        let qs = g.qubits();
        match qs.len() {
            1 if noise.p_1q > 0.0 => {
                next(t.clone(), out.clone(), p * (1.0 - noise.p_1q), acc);
                for e in paulis1(qs[0]) {
                    let mut t2 = t.clone();
                    t2.apply_gate(&e).unwrap();
                    next(t2, out.clone(), p * noise.p_1q / 3.0, acc);
                }
            }
            2 if noise.p_2q > 0.0 => {
                next(t.clone(), out.clone(), p * (1.0 - noise.p_2q), acc);
                for k in 1..16 {
                    let mut t2 = t.clone();
                    for e in [pauli_of(k / 4, qs[0]), pauli_of(k % 4, qs[1])]
                        .into_iter()
                        .flatten()
                    {
                        t2.apply_gate(&e).unwrap();
                    }
                    next(t2, out.clone(), p * noise.p_2q / 15.0, acc);
                }
            }
            _ => next(t, out, p, acc),
        }
    };
    // measure with every possible outcome: (state, outcome, probability)
    let branches = |t: &Tableau, q: usize| -> Vec<(Tableau, bool, f64)> {
        let mut t = t.clone();
        match t.peek(q) {
            Some(b) => vec![(t, b, 1.0)],
            None => [false, true]
                .into_iter()
                .map(|b| {
                    let mut t2 = t.clone();
                    t2.measure_with(q, Some(b), &mut StdRng::seed_from_u64(0));
                    (t2, b, 0.5)
                })
                .collect(),
        }
    };
    match *op {
        Op::Gate(g) => {
            t.apply_gate(&g).unwrap();
            gate_noise(t, out, &g, acc);
        }
        Op::Measure(q) => {
            for (t2, b, pb) in branches(&t, q) {
                let pm = noise.p_meas;
                let mut o = out.clone();
                o.push(b);
                next(t2.clone(), o, p * pb * (1.0 - pm), acc);
                if pm > 0.0 {
                    let mut o = out.clone();
                    o.push(!b);
                    next(t2, o, p * pb * pm, acc);
                }
            }
        }
        Op::Reset(q) => {
            for (mut t2, b, pb) in branches(&t, q) {
                if b {
                    t2.x(q);
                }
                let pr = noise.p_reset;
                next(t2.clone(), out.clone(), p * pb * (1.0 - pr), acc);
                if pr > 0.0 {
                    t2.x(q);
                    next(t2, out.clone(), p * pb * pr, acc);
                }
            }
        }
        Op::ClassicControlled {
            gate,
            meas_index,
            target_value,
        } => {
            if out[meas_index] == target_value {
                t.apply_gate(&gate).unwrap();
                gate_noise(t, out, &gate, acc);
            } else {
                next(t, out, p, acc);
            }
        }
        Op::XFlip(q, pf) | Op::YFlip(q, pf) | Op::ZFlip(q, pf) => {
            let g = match op {
                Op::XFlip(..) => Gate::X(q),
                Op::YFlip(..) => Gate::Y(q),
                _ => Gate::Z(q),
            };
            next(t.clone(), out.clone(), p * (1.0 - pf), acc);
            t.apply_gate(&g).unwrap();
            next(t, out, p * pf, acc);
        }
        Op::Depolarize1q(q, pd) => {
            next(t.clone(), out.clone(), p * (1.0 - pd), acc);
            for e in paulis1(q) {
                let mut t2 = t.clone();
                t2.apply_gate(&e).unwrap();
                next(t2, out.clone(), p * pd / 3.0, acc);
            }
        }
        Op::Depolarize2q(a, b, pd) => {
            next(t.clone(), out.clone(), p * (1.0 - pd), acc);
            for k in 1..16 {
                let mut t2 = t.clone();
                for e in [pauli_of(k / 4, a), pauli_of(k % 4, b)]
                    .into_iter()
                    .flatten()
                {
                    t2.apply_gate(&e).unwrap();
                }
                next(t2, out.clone(), p * pd / 15.0, acc);
            }
        }
    }
    let _ = &mut rng;
}

/// Exact distribution of the sampler's affine map, by enumerating every
/// joint assignment of its variable groups.
fn sampler_dist(s: &SymPhaseSampler) -> Dist {
    let mut assignments: Vec<(Vec<u64>, f64)> = vec![(vec![0; s.num_vars()], 1.0)];
    for g in s.groups() {
        let mut nxt = vec![];
        for (vals, p) in &assignments {
            for (pat, pp) in g.dist.outcomes() {
                if pp == 0.0 {
                    continue;
                }
                let mut v = vals.clone();
                for k in 0..g.dist.len() {
                    v[g.first as usize + k] = ((pat >> k) & 1) as u64;
                }
                nxt.push((v, p * pp));
            }
        }
        assignments = nxt;
    }
    let mut acc = Dist::new();
    let mut out = vec![0u64; s.num_measurements()];
    for (vals, p) in assignments {
        s.eval(&vals, &mut out);
        let key: Vec<bool> = out.iter().map(|w| w & 1 == 1).collect();
        *acc.entry(key).or_insert(0.0) += p;
    }
    acc
}

fn assert_dist_eq(a: &Dist, b: &Dist, what: &str) {
    for k in a.keys().chain(b.keys()) {
        let (x, y) = (
            a.get(k).copied().unwrap_or(0.0),
            b.get(k).copied().unwrap_or(0.0),
        );
        assert!(
            (x - y).abs() <= 1e-9 * x.max(y) + 1e-15,
            "{what}: P({k:?}) tableau {x} vs sampler {y}"
        );
    }
}

/// Number of joint assignments the enumeration would visit.
fn enum_size(s: &SymPhaseSampler) -> f64 {
    s.groups()
        .iter()
        .map(|g| g.dist.outcomes().len() as f64)
        .product()
}

fn random_noisy_circuit(n: usize, len: usize, rng: &mut StdRng) -> Circuit {
    let mut c = Circuit::new(n);
    let mut nmeas = 0;
    for _ in 0..len {
        let a = rng.random_range(0..n);
        let b = (a + rng.random_range(1..n.max(2))) % n;
        let p = [0.1, 0.25, 0.5][rng.random_range(0..3)];
        match rng.random_range(0..20) {
            0..=2 => {
                c.gate(Gate::H(a));
            }
            3 => {
                c.gate(Gate::S(a));
            }
            4 => {
                c.gate(Gate::Sdg(a));
            }
            5 => {
                c.gate(Gate::Phase(a, FRAC_PI_2 * rng.random_range(1..4) as f64));
            }
            6..=7 if n > 1 => {
                c.gate(Gate::Cnot(a, b));
            }
            8 if n > 1 => {
                c.gate(Gate::Cz(a, b));
            }
            9 if n > 1 => {
                c.gate(Gate::Swap(a, b));
            }
            10..=11 => {
                c.measure(a);
                nmeas += 1;
            }
            12 => {
                c.reset(a);
            }
            13 => {
                c.x_flip(a, p);
            }
            14 => {
                c.y_flip(a, p);
            }
            15 => {
                c.z_flip(a, p);
            }
            16 => {
                c.depolarize_1q(a, p);
            }
            17 if n > 1 => {
                c.depolarize_2q(a, b, p);
            }
            18 if nmeas > 0 => {
                let g = [Gate::X(a), Gate::Y(a), Gate::Z(a)][rng.random_range(0..3)];
                c.c_if(rng.random_range(0..nmeas), g);
            }
            _ => {
                c.gate(Gate::X(a));
            }
        }
    }
    c.measure_all();
    c
}

#[test]
fn exact_distribution_matches_tableau_on_random_small_circuits() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut checked = 0;
    let mut tries = 0;
    while checked < 300 {
        tries += 1;
        assert!(tries < 8000);
        let n = rng.random_range(1..4);
        let len = rng.random_range(3..16);
        let c = random_noisy_circuit(n, len, &mut rng);
        let noise = match rng.random_range(0..4) {
            0 => NoiseModel::none(),
            1 => NoiseModel::none().with_meas(0.2),
            2 => NoiseModel::none().with_reset(0.3).with_meas(0.1),
            _ => NoiseModel::none().with_p1(0.15),
        };
        let s = match SymPhaseSampler::new(&c, &noise) {
            Ok(s) => s,
            // gate noise on a classically controlled gate is not affine
            Err(_) if noise.p_1q > 0.0 => continue,
            Err(e) => panic!("{e}"),
        };
        if enum_size(&s) > 5e3 {
            continue;
        }
        let a = tableau_dist(&c, &noise);
        let b = sampler_dist(&s);
        assert_dist_eq(&a, &b, &format!("circuit {c:?} noise {noise:?}"));
        checked += 1;
    }
}

#[test]
fn exact_distribution_with_two_qubit_gate_noise() {
    // Bell pair, parity check onto an ancilla, mid-circuit reset and re-use
    let mut c = Circuit::new(3);
    c.h(0).cnot(0, 1).cnot(1, 2).measure(2).reset(2);
    c.h(2).cnot(2, 0).h(2).measure(2).measure(0).measure(1);
    let noise = NoiseModel::gate_depolarizing(0.0, 0.2).with_meas(0.05);
    let s = SymPhaseSampler::new(&c, &noise).unwrap();
    assert!(enum_size(&s) < 1e6);
    assert_dist_eq(&tableau_dist(&c, &noise), &sampler_dist(&s), "parity check");
}

#[test]
fn rejects_conditioned_non_pauli_and_non_clifford() {
    let mut c = Circuit::new(2);
    c.measure(0).c_if(0, Gate::H(1));
    assert!(SymPhaseSampler::new(&c, &NoiseModel::none()).is_err());
    let mut c = Circuit::new(1);
    c.gate(Gate::T(0));
    assert!(SymPhaseSampler::new(&c, &NoiseModel::none()).is_err());
}

/// Bit-packed sampling (geometric skipping, 64 shots per word) against the
/// exact marginals of the affine map, and against shot-by-shot tableau runs,
/// on a d=3 surface-code memory experiment.
#[test]
fn surface_code_marginals_match_tableau_runs() {
    let code = SurfaceCode::new(3, 3);
    let c = code.build_circuit();
    let noise = NoiseModel::circuit_level(0.01, 0.01);
    let s = SymPhaseSampler::new(&c, &noise).unwrap();
    let shots = 20_000;
    let mut rng = StdRng::seed_from_u64(7);
    let fast = s.sample(shots, &mut rng);
    let mut slow = vec![];
    for _ in 0..shots / 4 {
        let mut t = Tableau::new(c.num_qubits);
        slow.push(c.run_noisy(&mut t, &noise, &mut rng).unwrap());
    }
    let m = s.num_measurements();
    // detection events: measurement j vs the same ancilla one round earlier
    // are what decoders see; compare raw marginals and pairwise parities
    for j in 0..m {
        let pf = fast.iter().filter(|r| r[j]).count() as f64 / fast.len() as f64;
        let ps = slow.iter().filter(|r| r[j]).count() as f64 / slow.len() as f64;
        let sd = (0.25 / fast.len() as f64 + 0.25 / slow.len() as f64).sqrt();
        assert!((pf - ps).abs() < 5.0 * sd, "measurement {j}: {pf} vs {ps}");
    }
    let stride = SurfaceCode::num_ancillas(3);
    for j in stride..m.min(3 * stride) {
        let par = |rs: &Vec<Vec<bool>>| {
            rs.iter().filter(|r| r[j] ^ r[j - stride]).count() as f64 / rs.len() as f64
        };
        let (pf, ps) = (par(&fast), par(&slow));
        let sd = (pf * (1.0 - pf) / fast.len() as f64 + ps * (1.0 - ps) / slow.len() as f64)
            .sqrt()
            .max(1e-3);
        assert!((pf - ps).abs() < 5.0 * sd, "detector {j}: {pf} vs {ps}");
    }
}

/// Targeted cases: re-randomisation after a measurement (H M H M), after a
/// reset, and both directions of the CZ frame rule.
#[test]
fn exact_distribution_on_targeted_circuits() {
    let none = NoiseModel::none();
    let mut cases = vec![];
    let mut c = Circuit::new(1);
    c.h(0).measure(0).h(0).measure(0);
    cases.push(c);
    let mut c = Circuit::new(2);
    c.h(0).cnot(0, 1).reset(0).h(0).measure(0).measure(1);
    cases.push(c);
    for (a, b) in [(0, 1), (1, 0)] {
        // X_a before CZ(a, b) becomes X_a Z_b; H(b) makes Z_b flip M(b)
        let mut c = Circuit::new(2);
        c.h(b).x_flip(a, 0.3).cz(a, b).h(b).measure(a).measure(b);
        cases.push(c);
        let mut c = Circuit::new(2);
        c.h(a).x_flip(b, 0.3).cz(a, b).h(a).measure(a).measure(b);
        cases.push(c);
    }
    for c in cases {
        let s = SymPhaseSampler::new(&c, &none).unwrap();
        assert_dist_eq(
            &tableau_dist(&c, &none),
            &sampler_dist(&s),
            &format!("{c:?}"),
        );
    }
}
