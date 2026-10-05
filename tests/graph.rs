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
        let k = rng.random_range(0..14);
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
            11 => {
                let mut qs: Vec<usize> = (0..n).filter(|_| rng.random_bool(0.5)).collect();
                if qs.is_empty() {
                    qs.push(edge_qubit(rng, n));
                }
                for i in (1..qs.len()).rev() {
                    qs.swap(i, rng.random_range(0..=i));
                }
                POp::ZString(qs, rand_angle(rng, np))
            }
            12 => POp::Global(rand_angle(rng, np)),
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

/// Reference state of `pc` at `p`, global-phase ops included.
fn reference(pc: &ParamCircuit, p: &[f64]) -> RefSv {
    let mut r = ref_run(&pc.bind(p).unwrap());
    let g = Complex64::from_polar(1.0, pc.global_phase(p));
    for a in r.a.iter_mut() {
        *a *= g;
    }
    r
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
    o.patch_bind = false;
    v.push(o);
    // dense fusion forced on every group, with L1 tiling, both bind modes
    for patch in [true, false] {
        let mut o = GraphOptions::default();
        o.block.dense_fusion = 3;
        o.block.dense_min_ops = 1;
        o.block.l1_tile_bytes = 256;
        o.block.small_n = 3;
        o.block.block_bytes = 256;
        o.patch_bind = patch;
        v.push(o);
    }
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

fn region_heavy(rng: &mut StdRng, n: usize, np: usize, len: usize) -> ParamCircuit {
    let mut pc = ParamCircuit::new(n, np);
    for _ in 0..len {
        let q = edge_qubit(rng, n);
        let op = match rng.random_range(0..14) {
            0..=3 if n >= 2 => {
                let (a, b) = edge_pair(rng, n);
                POp::Fixed(Gate::Cnot(a, b))
            }
            4 => POp::Fixed(Gate::X(q)),
            5 => POp::Fixed(Gate::Y(q)),
            6 if n >= 2 => {
                let (a, b) = edge_pair(rng, n);
                POp::Fixed(Gate::Swap(a, b))
            }
            7 => POp::Rz(q, rand_angle(rng, np)),
            8 => POp::Phase(q, rand_angle(rng, np)),
            9 if n >= 2 => {
                let (a, b) = edge_pair(rng, n);
                match rng.random_range(0..4) {
                    0 => POp::CPhase(a, b, rand_angle(rng, np)),
                    1 => POp::Rzz(a, b, rand_angle(rng, np)),
                    2 => POp::Fixed(Gate::Cz(a, b)),
                    _ => POp::ZString(vec![a, b], rand_angle(rng, np)),
                }
            }
            10 => POp::Fixed(
                [
                    Gate::Z(q),
                    Gate::S(q),
                    Gate::Sdg(q),
                    Gate::T(q),
                    Gate::Tdg(q),
                ][rng.random_range(0..5)],
            ),
            11 if n >= 3 => {
                let (a, b, c) = distinct3(rng, n);
                POp::ZString(vec![a, b, c], rand_angle(rng, np))
            }
            12 => POp::Fixed(Gate::H(q)),
            _ => POp::Rx(q, rand_angle(rng, np)),
        };
        pc.push(op);
    }
    pc
}

#[test]
fn phase_regions_exact() {
    use qsim_lab::graph::{phase_regions, RewriteOptions};
    let mut rng = StdRng::seed_from_u64(base_seed() ^ 0x9a15);
    let mut rewritten = 0;
    for case in 0..200 * iters() {
        let n = 1 + case % 7;
        let np = rng.random_range(0..4);
        let len = rng.random_range(0..60);
        let pc = region_heavy(&mut rng, n, np, len);
        for (reorder, w) in [(true, 6), (false, 6), (true, 1)] {
            let (rc, st) = phase_regions(
                &pc,
                &RewriteOptions {
                    reorder,
                    max_weight: w,
                },
            );
            rewritten += st.rewritten;
            for gopts in [GraphOptions::default(), {
                let mut o = GraphOptions::default();
                o.max_zstring = 2;
                o
            }] {
                let cc = CompiledCircuit::compile(&rc, None, &gopts).unwrap();
                for _ in 0..2 {
                    let p = rand_params(&mut rng, np);
                    let r = reference(&pc, &p);
                    // the rewritten circuit's own bound form, up to its global phase
                    let r2 = reference(&rc, &p);
                    let d2 = max_amp_diff(&r.a, r2.a.iter().copied());
                    assert!(
                        d2 < 1e-10,
                        "case {case} bound rewrite diff {d2}\n{pc:?}\n{rc:?}"
                    );
                    let sv = cc.bind(&p).unwrap().statevector().unwrap();
                    let d = max_amp_diff(&r.a, sv.amplitudes().iter().copied());
                    assert!(d < 1e-10, "case {case} diff {d}\n{pc:?}\n{rc:?}\n{p:?}");
                }
            }
        }
    }
    assert!(rewritten > 100, "only {rewritten} regions rewritten");
}

#[test]
fn gadget_ladders_collapse() {
    use qsim_lab::graph::{phase_regions, RewriteOptions};
    // ladder · Rz · ladder^-1 written out gate by gate
    let n = 6;
    let mut pc = ParamCircuit::new(n, 1);
    for w in [[0usize, 1, 2, 3], [2, 3, 4, 5]] {
        for i in 0..3 {
            pc.gate(Gate::Cnot(w[i], w[i + 1]));
        }
        pc.rz(w[3], Angle::param(0));
        for i in (0..3).rev() {
            pc.gate(Gate::Cnot(w[i], w[i + 1]));
        }
    }
    let (rc, st) = phase_regions(&pc, &RewriteOptions::default());
    assert_eq!(st.perm_ops_after, 0, "{st:?} {rc:?}");
    assert_eq!(
        rc.ops
            .iter()
            .filter(|o| matches!(o, POp::ZString(..)))
            .count(),
        2
    );
}

#[test]
fn dedup_blocks_reproduce_the_circuit() {
    use qsim_lab::graph::dedup::{analyse, block_unitaries};
    let mut rng = StdRng::seed_from_u64(base_seed() ^ 0x9a16);
    for case in 0..60 * iters() {
        let n = 2 + case % 6;
        let np = rng.random_range(0..3);
        let len = rng.random_range(1..50);
        let pc = random_param_circuit(&mut rng, n, np, len);
        let p = rand_params(&mut rng, np);
        let r = reference(&pc, &p);
        for k in [2usize, 3, 4] {
            let d = analyse(&pc, k);
            let covered: usize = d.blocks.iter().map(|b| b.ops.len()).sum();
            assert_eq!(covered, d.total_ops);
            let (u_plain, c_plain) = block_unitaries(&pc, &d, &p, false);
            let (u_dedup, c_dedup) = block_unitaries(&pc, &d, &p, true);
            assert_eq!(c_dedup, d.num_classes);
            assert_eq!(c_plain, d.blocks.len());
            // apply blocks in order of their last op
            let mut order: Vec<usize> = (0..d.blocks.len()).collect();
            order.sort_by_key(|&b| *d.blocks[b].ops.last().unwrap());
            let mut s = RefSv::new(n);
            for &bi in &order {
                let b = &d.blocks[bi];
                assert!(b.qubits.len() <= k || b.ops.len() == 1);
                for u in [&u_plain[bi], &u_dedup[bi]] {
                    assert!(u
                        .iter()
                        .zip(u_plain[bi].iter())
                        .all(|(a, b)| (a - b).norm() < 1e-12));
                }
                let u = &u_dedup[bi];
                let dim = 1usize << b.qubits.len();
                let old = s.a.clone();
                for (x, out) in s.a.iter_mut().enumerate() {
                    let loc = b
                        .qubits
                        .iter()
                        .enumerate()
                        .fold(0, |l, (j, &q)| l | ((x >> q) & 1) << j);
                    let rest = b.qubits.iter().fold(x, |y, &q| y & !(1 << q));
                    let mut acc = Complex64::new(0.0, 0.0);
                    for y in 0..dim {
                        let src = b
                            .qubits
                            .iter()
                            .enumerate()
                            .fold(rest, |z, (j, &q)| z | ((y >> j) & 1) << q);
                        acc += u[loc * dim + y] * old[src];
                    }
                    *out = acc;
                }
            }
            // global-phase ops are not in any block
            let g = Complex64::from_polar(1.0, pc.global_phase(&p));
            let dd = max_amp_diff(&r.a, s.a.iter().map(|a| a * g));
            assert!(dd < 1e-10, "case {case} k={k} diff {dd}");
        }
    }
}

#[test]
fn dedup_finds_trotter_steps_and_recipe_cse_is_exact() {
    use qsim_lab::graph::dedup::analyse;
    let n = 8;
    let mut pc = ParamCircuit::new(n, 2);
    for _ in 0..10 {
        for q in 0..n - 1 {
            pc.rzz(q, q + 1, Angle::scaled(0, 2.0));
        }
        for q in 0..n {
            pc.rx(q, Angle::scaled(1, 2.0));
        }
    }
    let d = analyse(&pc, 4);
    assert!(d.coverage() > 0.9, "{}", d.coverage());
    let mut obs = Observable::new();
    obs.add(1.0, "Z3").unwrap().add(0.5, "X0 X1").unwrap();
    let mut off = GraphOptions::default();
    off.dedup_recipes = false;
    let a = CompiledCircuit::compile(&pc, Some(&obs), &GraphOptions::default()).unwrap();
    let b = CompiledCircuit::compile(&pc, Some(&obs), &off).unwrap();
    for p in [[0.1, 0.2], [-1.3, 0.77]] {
        let x = a.bind(&p).unwrap().expectation().unwrap();
        let y = b.bind(&p).unwrap().expectation().unwrap();
        assert_eq!(x, y);
    }
}

#[test]
fn partition_path_sum_is_exact() {
    use qsim_lab::graph::partition::{cut_amplitudes, cut_size, plan_cut, CutPlan};
    use qsim_lab::planner::{Engine, PlannerConfig};
    let mut rng = StdRng::seed_from_u64(base_seed() ^ 0x9a17);
    let cfg = PlannerConfig::default();
    for case in 0..40 * iters() {
        let n = 3 + case % 6;
        let in_a: Vec<bool> = (0..n).map(|q| q < 1 + case % (n - 1)).collect();
        let mut c = Circuit::new(n);
        for _ in 0..rng.random_range(5..40) {
            let g = random_gate(&mut rng, n, false, true);
            let qs = g.qubits();
            let crossing = qs.iter().any(|&q| in_a[q]) && !qs.iter().all(|&q| in_a[q]);
            if crossing && !matches!(g, Gate::Cz(..) | Gate::Cnot(..) | Gate::CPhase(..)) {
                continue;
            }
            c.gate(g);
        }
        let k = cut_size(&c, &in_a).unwrap();
        if k > 6 {
            continue;
        }
        let plan = CutPlan {
            in_a: in_a.clone(),
            cut: k,
            side_a: (Engine::StateVector, 0.0),
            side_b: (Engine::StateVector, 0.0),
            predicted_secs: 0.0,
            single: None,
            plan_secs: 0.0,
        };
        let xs: Vec<u128> = (0..1u128 << n).collect();
        let got = cut_amplitudes(&c, &plan, &xs, &cfg).unwrap();
        let r = ref_run(&c);
        let d = max_amp_diff(&r.a, got.into_iter());
        assert!(d < 1e-10, "case {case} cut {k} diff {d}\n{c:?}");
    }
    // the planner-priced plan also gives exact amplitudes
    let mut c = Circuit::new(8);
    for q in 0..8 {
        c.gate(Gate::H(q));
        c.gate(Gate::T(q));
    }
    for q in 0..3 {
        c.gate(Gate::Cnot(q, q + 1));
    }
    c.gate(Gate::Cz(3, 4));
    for q in 4..7 {
        c.gate(Gate::CPhase(q, q + 1, 0.3));
        c.gate(Gate::Rx(q, 0.7));
    }
    let plan = plan_cut(&c, 8, &cfg, 4, 6).unwrap();
    let xs: Vec<u128> = (0..256).collect();
    let got = cut_amplitudes(&c, &plan, &xs, &cfg).unwrap();
    let d = max_amp_diff(&ref_run(&c).a, got.into_iter());
    assert!(d < 1e-10, "{d} {plan:?}");
}

#[test]
fn fold_basis_exact() {
    use qsim_lab::graph::fold::fold_basis;
    let mut rng = StdRng::seed_from_u64(base_seed() ^ 0x9a18);
    let mut removed = 0;
    for case in 0..300 * iters() {
        let n = 1 + case % 7;
        let np = rng.random_range(0..3);
        let len = rng.random_range(0..40);
        // mostly monomial / diagonal ops so wires stay known for a while
        let mut pc = if rng.random_bool(0.5) {
            region_heavy(&mut rng, n, np, len)
        } else {
            random_param_circuit(&mut rng, n, np, len)
        };
        if n >= 3 && rng.random_bool(0.3) {
            let (a, b, c) = distinct3(&mut rng, n);
            pc.ops.insert(0, POp::Fixed(Gate::X(a)));
            pc.push(POp::Fixed(Gate::Ccx(a, b, c)));
        }
        let (f, st) = fold_basis(&pc);
        removed += st.removed + st.simplified;
        for _ in 0..2 {
            let p = rand_params(&mut rng, np);
            let r = reference(&pc, &p);
            let r2 = reference(&f, &p);
            let d = max_amp_diff(&r.a, r2.a.iter().copied());
            assert!(d < 1e-10, "case {case} diff {d}\n{pc:?}\n{f:?}");
        }
    }
    assert!(removed > 200, "{removed}");
}
