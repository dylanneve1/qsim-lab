//! Differential tests of the graph compiler (`src/graph`) against the
//! independent reference state vector of `tests/audit_common`.

mod audit_common;

use audit_common::*;
use num_complex::Complex64;
use qsim_lab::graph::{Angle, CompiledCircuit, GraphOptions, Observable, POp, ParamCircuit};
use qsim_lab::{Circuit, Gate};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Random affine angle over `np` parameters (sometimes constant, sometimes
/// an edge value).
fn rand_angle(rng: &mut StdRng, np: usize) -> Angle {
    match rng.random_range(0..5) {
        0 => Angle::constant(edge_angle(rng)),
        1 if np > 0 => Angle::param(rng.random_range(0..np)),
        _ if np > 0 => {
            let mut a = Angle::constant(rng.random_range(-1.0..1.0));
            for _ in 0..rng.random_range(1..3) {
                a = a.plus(&Angle::scaled(
                    rng.random_range(0..np),
                    rng.random_range(-2.0..2.0),
                ));
            }
            a
        }
        _ => Angle::constant(rng.random_range(-3.0..3.0)),
    }
}

pub fn random_param_circuit(rng: &mut StdRng, n: usize, np: usize, len: usize) -> ParamCircuit {
    let mut pc = ParamCircuit::new(n, np);
    for _ in 0..len {
        let k = rng.random_range(0..12);
        let op = match k {
            0..=2 => POp::Fixed(random_gate(rng, n, false, true)),
            3 => POp::Rx(edge_qubit(rng, n), rand_angle(rng, np)),
            4 => POp::Ry(edge_qubit(rng, n), rand_angle(rng, np)),
            5 => POp::Rz(edge_qubit(rng, n), rand_angle(rng, np)),
            6 => POp::Phase(edge_qubit(rng, n), rand_angle(rng, np)),
            7 => POp::U(
                edge_qubit(rng, n),
                rand_angle(rng, np),
                rand_angle(rng, np),
                rand_angle(rng, np),
            ),
            8 | 9 | 10 if n >= 2 => {
                let (a, b) = edge_pair(rng, n);
                match k {
                    8 => POp::CPhase(a, b, rand_angle(rng, np)),
                    9 => POp::Rzz(a, b, rand_angle(rng, np)),
                    _ => POp::Rxx(a, b, rand_angle(rng, np)),
                }
            }
            _ => POp::Fixed(random_gate(rng, n, false, true)),
        };
        pc.push(op);
    }
    pc
}

fn rand_params(rng: &mut StdRng, np: usize) -> Vec<f64> {
    (0..np)
        .map(|_| {
            if rng.random_bool(0.2) {
                edge_angle(rng)
            } else {
                rng.random_range(-4.0..4.0)
            }
        })
        .collect()
}

fn apply_1q_matrix(s: &mut RefSv, q: usize, m: [[Complex64; 2]; 2]) {
    let old = s.a.clone();
    for (i, out) in s.a.iter_mut().enumerate() {
        let r = (i >> q) & 1;
        *out = m[r][0] * old[i & !(1 << q)] + m[r][1] * old[i | (1 << q)];
    }
}
/// Naive reference: `RefSv` plus textbook matrices for the gates it lacks.
fn ref_run(c: &Circuit) -> RefSv {
    let cx = Complex64::new;
    let mut s = RefSv::new(c.num_qubits);
    for g in c.gates() {
        match *g {
            Gate::I(_) => {}
            Gate::Sx(q) => apply_1q_matrix(
                &mut s,
                q,
                [[cx(0.5, 0.5), cx(0.5, -0.5)], [cx(0.5, -0.5), cx(0.5, 0.5)]],
            ),
            Gate::Sxdg(q) => apply_1q_matrix(
                &mut s,
                q,
                [[cx(0.5, -0.5), cx(0.5, 0.5)], [cx(0.5, 0.5), cx(0.5, -0.5)]],
            ),
            Gate::U(q, th, ph, lam) => {
                let (c2, s2) = ((th / 2.0).cos(), (th / 2.0).sin());
                let e = |t: f64| cx(t.cos(), t.sin());
                apply_1q_matrix(
                    &mut s,
                    q,
                    [[cx(c2, 0.0), -e(lam) * s2], [e(ph) * s2, e(ph + lam) * c2]],
                );
            }
            Gate::ISwap(a, b) | Gate::ISwapdg(a, b) => {
                let ph = if matches!(g, Gate::ISwap(..)) {
                    cx(0.0, 1.0)
                } else {
                    cx(0.0, -1.0)
                };
                let old = s.a.clone();
                for (i, &v) in old.iter().enumerate() {
                    let (ba, bb) = ((i >> a) & 1, (i >> b) & 1);
                    if ba != bb {
                        s.a[i ^ (1 << a) ^ (1 << b)] = v * ph;
                    }
                }
            }
            ref g => s.apply(g),
        }
    }
    s
}

fn reference(pc: &ParamCircuit, p: &[f64]) -> RefSv {
    ref_run(&pc.bind(p).unwrap())
}

fn rand_observable(rng: &mut StdRng, n: usize, terms: usize) -> (Observable, Vec<(f64, String)>) {
    let mut o = Observable::new();
    let mut dense = Vec::new();
    for _ in 0..terms {
        let mut sparse = String::new();
        let mut d = String::new();
        let diag_only = rng.random_bool(0.4);
        for q in 0..n {
            let ch = if rng.random_bool(0.35) {
                if diag_only {
                    'Z'
                } else {
                    ['X', 'Y', 'Z'][rng.random_range(0..3)]
                }
            } else {
                'I'
            };
            d.push(ch);
            if ch != 'I' {
                sparse.push_str(&format!("{ch}{q} "));
            }
        }
        let c = rng.random_range(-1.0..1.0);
        o.add(c, &sparse).unwrap();
        dense.push((c, d));
    }
    (o, dense)
}

fn option_sets() -> Vec<GraphOptions> {
    let mut v = vec![GraphOptions::default()];
    let mut o = GraphOptions::default();
    o.light_cone = false;
    o.components = false;
    o.prefix_cache = false;
    o.diagonal_suffix = false;
    v.push(o);
    let mut o = GraphOptions::default();
    o.block.small_n = 2; // force multi-stage plans on small registers
    o.block.block_bytes = 64;
    o.block.slots = 2;
    v.push(o);
    let mut o = GraphOptions::default();
    o.block.schedule_diag = false;
    o.block.simd = false;
    v.push(o);
    v
}

#[test]
fn statevector_matches_reference() {
    let mut rng = StdRng::seed_from_u64(base_seed() ^ 0x9a11);
    for case in 0..120 * iters() {
        let n = 1 + case % 7;
        let np = rng.random_range(0..5);
        let len = rng.random_range(0..40);
        let pc = random_param_circuit(&mut rng, n, np, len);
        for opts in option_sets() {
            let cc = CompiledCircuit::compile(&pc, None, &opts).unwrap();
            for _ in 0..3 {
                let p = rand_params(&mut rng, np);
                let r = reference(&pc, &p);
                let sv = cc.bind(&p).unwrap().statevector().unwrap();
                let d = max_amp_diff(&r.a, sv.amplitudes().iter().copied());
                assert!(
                    d < 1e-11,
                    "case {case} n={n} diff {d}\n{pc:?}\nparams {p:?}"
                );
                let xs: Vec<u128> = (0..(1u128 << n)).step_by(3).collect();
                let amps = cc.bind(&p).unwrap().amplitudes(&xs).unwrap();
                for (x, a) in xs.iter().zip(amps) {
                    assert!((a - r.a[*x as usize]).norm() < 1e-11);
                }
            }
        }
    }
}

#[test]
fn expectation_matches_reference() {
    let mut rng = StdRng::seed_from_u64(base_seed() ^ 0x9a12);
    for case in 0..120 * iters() {
        let n = 1 + case % 8;
        let np = rng.random_range(1..5);
        let len = rng.random_range(0..40);
        let pc = random_param_circuit(&mut rng, n, np, len);
        let nt = rng.random_range(1..5);
        let (obs, dense) = rand_observable(&mut rng, n, nt);
        for opts in option_sets() {
            let cc = CompiledCircuit::compile(&pc, Some(&obs), &opts).unwrap();
            for _ in 0..3 {
                let p = rand_params(&mut rng, np);
                let r = reference(&pc, &p);
                let want: f64 = dense.iter().map(|(c, s)| c * r.pauli_expectation(s)).sum();
                let got = cc.bind(&p).unwrap().expectation().unwrap();
                assert!(
                    (got - want).abs() < 1e-10,
                    "case {case}: {got} vs {want}\n{pc:?}\n{obs:?}\n{p:?}"
                );
            }
        }
    }
}

#[test]
fn sweep_matches_individual_binds() {
    let mut rng = StdRng::seed_from_u64(base_seed() ^ 0x9a13);
    let pc = random_param_circuit(&mut rng, 6, 3, 60);
    let (obs, _) = rand_observable(&mut rng, 6, 4);
    let cc = CompiledCircuit::compile(&pc, Some(&obs), &GraphOptions::default()).unwrap();
    let ps: Vec<Vec<f64>> = (0..50).map(|_| rand_params(&mut rng, 3)).collect();
    let sw = cc.sweep_expectation(&ps).unwrap();
    for (p, v) in ps.iter().zip(sw) {
        assert_eq!(v, cc.bind(p).unwrap().expectation().unwrap());
    }
}

#[test]
fn larger_registers_multi_stage() {
    // 14..16 qubits: real multi-stage plans with the default block size
    let mut rng = StdRng::seed_from_u64(base_seed() ^ 0x9a14);
    for n in [14usize, 16] {
        let pc = random_param_circuit(&mut rng, n, 4, 150);
        let cc = CompiledCircuit::compile(&pc, None, &GraphOptions::default()).unwrap();
        let p = rand_params(&mut rng, 4);
        let r = reference(&pc, &p);
        let sv = cc.bind(&p).unwrap().statevector().unwrap();
        let d = max_amp_diff(&r.a, sv.amplitudes().iter().copied());
        assert!(d < 1e-11, "n={n} diff {d}");
    }
}

#[test]
fn qaoa_ring_light_cone_and_components() {
    // two disjoint rings: components; ZZ observable on one edge: light cone
    let n = 10;
    let mut pc = ParamCircuit::new(n, 2);
    for q in 0..n {
        pc.gate(Gate::H(q));
    }
    for ring in [0..5usize, 5..10] {
        let qs: Vec<usize> = ring.collect();
        for i in 0..qs.len() {
            pc.rzz(qs[i], qs[(i + 1) % qs.len()], Angle::param(0));
        }
    }
    for q in 0..n {
        pc.rx(q, Angle::scaled(1, 2.0));
    }
    let mut obs = Observable::new();
    obs.zz(1.0, 0, 1).zz(0.5, 6, 7);
    let cc = CompiledCircuit::compile(&pc, Some(&obs), &GraphOptions::default()).unwrap();
    assert_eq!(cc.stats().parts.len(), 2);
    let p = [0.37, -1.1];
    let r = reference(&pc, &p);
    let want = r.pauli_expectation("ZZIIIIIIII") + 0.5 * r.pauli_expectation("IIIIIIZZII");
    let got = cc.bind(&p).unwrap().expectation().unwrap();
    assert!((got - want).abs() < 1e-12);
}
