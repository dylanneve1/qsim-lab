//! Round-4 audit of `compile::repeat` (repeat-block fast paths): independent
//! differential fuzz against the naive reference of `tests/audit_common`.
//!
//! * `run_dense` on hand-built programs: repeat counts 0 / 1 / 2 / large,
//!   nested repeats, parameterised repeats, diagonal / Clifford / small /
//!   general non-Clifford bodies, every `ExecOptions` path forced in turn;
//!   amplitudes compared with the global phase;
//! * `detect` + `to_circuit` / `rewrite` on structured random circuits,
//!   including blocks with measurements, resets, classical control and
//!   Pauli flips: the whole instrument (every branch's unnormalised state)
//!   is compared;
//! * `pipeline::simulate_with` against the reference for amplitude,
//!   expectation and sampling requests (samples: exact support + 6σ
//!   frequency test), and it must never fail where `simulate` succeeds.
//!
//! `QSIM_FUZZ_ITERS` scales the number of cases (default 1 = CI size).

#![allow(clippy::needless_range_loop)]

#[path = "../audit_common/mod.rs"]
mod audit_common;
#[path = "../audit_r4/mod.rs"]
mod audit_r4;

use audit_common::{iters, RefSv};
use audit_r4::{apply, branches, compare, random_state, record_distribution_capped};

/// Sampling checks skipped because the reference tree was too big.
static SKIPPED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
use num_complex::Complex64 as C;
use qsim_lab::compile::repeat::exec::{run_dense, ExecOptions};
use qsim_lab::compile::repeat::workloads::{
    grover, qaoa_ring, qec_memory, random_clifford_gates, repeated, shuffle_commuting, trotter,
};
use qsim_lab::compile::repeat::{detect, op_angles, rewrite, DetectOptions, Node, Program};
use qsim_lab::pipeline::{
    simulate, simulate_with, Budget, Output, RepeatOptions, Request, SimOptions,
};
use qsim_lab::{Circuit, Gate, Op, StateVectorF64};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::{FRAC_PI_4, PI};

fn rng_for(test: u64) -> StdRng {
    StdRng::seed_from_u64(audit_common::base_seed() ^ (test << 40) ^ 0x5EED)
}

fn distinct(rng: &mut StdRng, n: usize, k: usize) -> Vec<usize> {
    let mut v: Vec<usize> = Vec::new();
    while v.len() < k {
        let x = rng.random_range(0..n);
        if !v.contains(&x) {
            v.push(x);
        }
    }
    v
}

fn angle(rng: &mut StdRng) -> f64 {
    match rng.random_range(0..4) {
        0 => rng.random_range(-8..8) as f64 * FRAC_PI_4,
        1 => rng.random_range(-1e2..1e2),
        _ => rng.random_range(-PI..PI),
    }
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Diag,
    Clifford,
    General,
}

fn gate_of(rng: &mut StdRng, n: usize, kind: Kind) -> Gate {
    let q = rng.random_range(0..n);
    let two = n >= 2 && rng.random_bool(0.35);
    match kind {
        Kind::Diag => {
            if two {
                let v = distinct(rng, n, 2);
                return if rng.random_bool(0.5) {
                    Gate::Cz(v[0], v[1])
                } else {
                    Gate::CPhase(v[0], v[1], angle(rng))
                };
            }
            match rng.random_range(0..8) {
                0 => Gate::Z(q),
                1 => Gate::S(q),
                2 => Gate::Sdg(q),
                3 => Gate::T(q),
                4 => Gate::Tdg(q),
                5 => Gate::Rz(q, angle(rng)),
                6 => Gate::I(q),
                _ => Gate::Phase(q, angle(rng)),
            }
        }
        Kind::Clifford => {
            if two {
                let v = distinct(rng, n, 2);
                return match rng.random_range(0..5) {
                    0 => Gate::Cz(v[0], v[1]),
                    1 => Gate::Swap(v[0], v[1]),
                    2 => Gate::ISwap(v[0], v[1]),
                    3 => Gate::ISwapdg(v[0], v[1]),
                    _ => Gate::Cnot(v[0], v[1]),
                };
            }
            match rng.random_range(0..9) {
                0 => Gate::H(q),
                1 => Gate::S(q),
                2 => Gate::Sdg(q),
                3 => Gate::X(q),
                4 => Gate::Y(q),
                5 => Gate::Z(q),
                6 => Gate::Sx(q),
                7 => Gate::Sxdg(q),
                _ => Gate::I(q),
            }
        }
        Kind::General => {
            if n >= 3 && rng.random_bool(0.1) {
                let v = distinct(rng, n, 3);
                return Gate::Ccx(v[0], v[1], v[2]);
            }
            match rng.random_range(0..6) {
                0 => gate_of(rng, n, Kind::Diag),
                1 | 2 => gate_of(rng, n, Kind::Clifford),
                3 => Gate::Rx(q, angle(rng)),
                4 => Gate::Ry(q, angle(rng)),
                _ => Gate::U(q, angle(rng), angle(rng), angle(rng)),
            }
        }
    }
}

fn body(rng: &mut StdRng, n: usize, kind: Kind, len: usize) -> Vec<Op> {
    (0..len).map(|_| Op::Gate(gate_of(rng, n, kind))).collect()
}

/// A random unitary program tree (depth <= 2) with all node kinds.
fn random_program(rng: &mut StdRng, n: usize, big: bool) -> Program {
    let reps_pool: &[usize] = if big {
        &[0, 1, 2, 3, 7, 64, 513, 4097]
    } else {
        &[0, 1, 2, 3, 5, 16]
    };
    let kind = |rng: &mut StdRng| match rng.random_range(0..3) {
        0 => Kind::Diag,
        1 => Kind::Clifford,
        _ => Kind::General,
    };
    let mut nodes = Vec::new();
    for _ in 0..rng.random_range(1..=4) {
        match rng.random_range(0..5) {
            0 => {
                let k = kind(rng);
                let len = rng.random_range(0..4);
                nodes.push(Node::Ops(body(rng, n, k, len)));
            }
            1 | 2 => {
                let k = kind(rng);
                let reps = reps_pool[rng.random_range(0..reps_pool.len())];
                let len = rng.random_range(1..8);
                let mut b = vec![Node::Ops(body(rng, n, k, len))];
                if rng.random_bool(0.4) {
                    // nested repeat
                    let inner_len = rng.random_range(1..5);
                    let inner = Node::Repeat {
                        body: vec![Node::Ops(body(rng, n, k, inner_len))],
                        reps: [0, 1, 2, 9][rng.random_range(0..4)],
                    };
                    b.insert(rng.random_range(0..2), inner);
                }
                nodes.push(Node::Repeat { body: b, reps });
            }
            3 => {
                // parameterised: same shape, different angles
                let k = kind(rng);
                let len = rng.random_range(1..6);
                let shape = body(rng, n, k, len);
                let reps = rng.random_range(1..6);
                let angles: Vec<Vec<f64>> = (0..reps)
                    .map(|_| {
                        shape
                            .iter()
                            .flat_map(|op| {
                                (0..op_angles(op).len())
                                    .map(|_| angle(rng))
                                    .collect::<Vec<_>>()
                            })
                            .collect()
                    })
                    .collect();
                nodes.push(Node::Param {
                    shape,
                    reps,
                    angles,
                });
            }
            _ => {
                // large repeat of a 1-2 gate body (support <= 2)
                let k = kind(rng);
                let reps = if big { 100_003 } else { 300 };
                let n_small = n.min(2);
                let len_ = rng.random_range(1..4);
                let b = body(rng, n_small, k, len_);
                nodes.push(Node::Repeat {
                    body: vec![Node::Ops(b)],
                    reps,
                });
            }
        }
    }
    Program {
        num_qubits: n,
        nodes,
    }
}

fn exec_variants() -> Vec<(&'static str, ExecOptions)> {
    let d = ExecOptions::default();
    vec![
        ("default", d.clone()),
        (
            "force_small",
            ExecOptions {
                force_small: true,
                ..d.clone()
            },
        ),
        (
            "no_diag",
            ExecOptions {
                diag: false,
                force_small: true,
                ..d.clone()
            },
        ),
        (
            "reuse_only",
            ExecOptions {
                diag: false,
                small_unitary: false,
                ..d.clone()
            },
        ),
        (
            "plain",
            ExecOptions {
                diag: false,
                small_unitary: false,
                reuse_plan: false,
                ..d.clone()
            },
        ),
        (
            "small_k2",
            ExecOptions {
                max_small_k: 2,
                force_small: true,
                ..d
            },
        ),
    ]
}

/// Reference: expand the program and apply gate by gate (a huge repeat
/// of a tiny body is expanded too: 1e5 x 3 gates on <= 64 amplitudes).
fn reference_run(p: &Program, init: &RefSv) -> RefSv {
    let mut s = init.clone();
    for g in p.to_circuit().gates() {
        apply(&mut s, g);
    }
    s
}

#[test]
fn audit_repeat_run_dense_matches_reference() {
    let mut rng = rng_for(1);
    for it in 0..60 * iters() {
        let n = rng.random_range(1..=6);
        let big = it % 4 == 0;
        let p = random_program(&mut rng, n, big);
        let init = random_state(n, &mut rng);
        let want = reference_run(&p, &init);
        for (name, opts) in exec_variants() {
            let mut sv = StateVectorF64::from_amplitudes(init.a.clone());
            run_dense(&p, &mut sv, &opts).expect("run_dense");
            let d = want
                .a
                .iter()
                .zip(sv.amplitudes())
                .map(|(x, y)| (x - y).norm())
                .fold(0.0, f64::max);
            // r up to 1e5: allow accumulated rounding of the reference too
            let tol = if big { 1e-7 } else { 1e-10 };
            assert!(d < tol, "#{it} {name}: diff {d}\n{p:?}");
        }
        // rewrite: exact with phase if phase_exact, else up to phase
        for allow in [false, true] {
            let rw = rewrite(&p, allow);
            let mut s = init.clone();
            for g in rw.circuit.gates() {
                apply(&mut s, g);
            }
            let ph = if rw.phase_exact {
                C::from_polar(1.0, rw.global_phase)
            } else {
                let ip: C = s.a.iter().zip(&want.a).map(|(x, y)| x.conj() * y).sum();
                ip / ip.norm()
            };
            let d = want
                .a
                .iter()
                .zip(&s.a)
                .map(|(x, y)| (x - ph * y).norm())
                .fold(0.0, f64::max);
            let tol = if big { 1e-7 } else { 1e-10 };
            assert!(d < tol, "#{it} rewrite(allow={allow}): diff {d}\n{p:?}");
        }
    }
}

/// Structured circuits: prefix . (block)^r . suffix with nested and
/// commuting-shuffled copies; blocks may contain measurements, resets,
/// classically controlled gates and Pauli flips.
fn structured(rng: &mut StdRng, n: usize, nonunitary: bool) -> Circuit {
    let kind = [Kind::Diag, Kind::Clifford, Kind::General][rng.random_range(0..3)];
    let mut blk = Circuit::new(n);
    let len_ = rng.random_range(2..10);
    for op in body(rng, n, kind, len_) {
        blk.ops.push(op);
    }
    let mut c = Circuit::new(n);
    let len_ = rng.random_range(0..4);
    for op in body(rng, n, Kind::General, len_) {
        c.ops.push(op);
    }
    let mut nmeas = 0usize;
    if nonunitary {
        // a measurement first so classical control has a bit
        c.measure(rng.random_range(0..n));
        nmeas += 1;
        let q = rng.random_range(0..n);
        match rng.random_range(0..5) {
            0 => {
                blk.measure(q);
            }
            1 => {
                blk.reset(q);
            }
            2 => blk.ops.push(Op::ClassicControlled {
                gate: gate_of(rng, n, Kind::General),
                meas_index: 0,
                target_value: rng.random_bool(0.5),
            }),
            3 => blk.ops.push(Op::XFlip(q, 0.2)),
            _ => {
                blk.measure(q);
                blk.ops.push(Op::ClassicControlled {
                    gate: gate_of(rng, n, Kind::Clifford),
                    meas_index: 0,
                    target_value: true,
                });
            }
        }
    }
    let reps = [1, 2, 3, 4, 6][rng.random_range(0..5)];
    for _ in 0..reps {
        let copy = if rng.random_bool(0.3) && !nonunitary {
            shuffle_commuting(&blk, rng)
        } else {
            blk.clone()
        };
        c.append(&copy);
    }
    if rng.random_bool(0.3) {
        // nested: (blk^2 . x)^2
        let mut outer = Circuit::new(n);
        outer.append(&blk).append(&blk);
        outer.gate(gate_of(rng, n, Kind::General));
        c.append(&outer).append(&outer);
    }
    let len_ = rng.random_range(0..3);
    for op in body(rng, n, Kind::General, len_) {
        c.ops.push(op);
    }
    let _ = nmeas;
    c
}

fn detect_opts(rng: &mut StdRng) -> DetectOptions {
    DetectOptions {
        min_ops: [2, 4, 8][rng.random_range(0..3)],
        layered: rng.random_bool(0.7),
        parameterised: rng.random_bool(0.7),
        ..DetectOptions::default()
    }
}

#[test]
fn audit_repeat_detect_and_rewrite_preserve_the_instrument() {
    let mut rng = rng_for(2);
    for it in 0..120 * iters() {
        let n = rng.random_range(1..=5);
        let nonunitary = it % 2 == 1;
        let c = structured(&mut rng, n, nonunitary);
        let p = detect(&c, &detect_opts(&mut rng));
        let init = random_state(n, &mut rng);
        let back = p.to_circuit();
        let d = compare(&c, &back, 0.0, &init);
        assert!(d < 1e-10, "#{it} to_circuit: {d}\n{:?}\n{p:?}", c.ops);
        for allow in [false, true] {
            let rw = rewrite(&p, allow);
            let (ba, bb) = (branches(&c, &init), branches(&rw.circuit, &init));
            // common phase: exact if claimed, else from the largest entry
            let ph = if rw.phase_exact {
                C::from_polar(1.0, rw.global_phase)
            } else {
                let mut best = (0.0, C::new(1.0, 0.0));
                for (k, v) in &ba {
                    if let Some(w) = bb.get(k) {
                        for (x, y) in v.iter().zip(w) {
                            if y.norm() > best.0 {
                                best = (y.norm(), x / y);
                            }
                        }
                    }
                }
                best.1 / best.1.norm()
            };
            let d = compare(&c, &rw.circuit, ph.arg(), &init);
            assert!(
                d < 1e-9,
                "#{it} rewrite(allow={allow}) diff {d}\n{:?}\n{:?}",
                c.ops,
                rw.circuit.ops
            );
        }
    }
}

fn ro(min_saved: usize) -> SimOptions {
    SimOptions {
        repeat: Some(RepeatOptions {
            min_saved_gates: min_saved,
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// True if the samples were checked against the exact distribution.
fn sampled_ok(c: &Circuit, shots: usize, seed: u64, what: &str) -> bool {
    let req = Request::Samples { shots, seed };
    let plain = simulate(c, &req, &Budget::default());
    let got = simulate_with(c, &req, &Budget::default(), &ro(1));
    let got = match (plain, got) {
        (Ok(_), Err(e)) => panic!("{what}: simulate_with failed where simulate succeeds: {e:?}"),
        (Err(_), _) => return false,
        (Ok(_), Ok(g)) => g,
    };
    let Output::Samples(s) = got.output else {
        unreachable!()
    };
    // the reference enumerates every branch: only check when that is small
    let Some(exact) = record_distribution_capped(c, 1 << 14) else {
        SKIPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return false;
    };
    let mut counts = std::collections::BTreeMap::<Vec<bool>, usize>::new();
    for r in &s {
        *counts.entry(r.clone()).or_insert(0) += 1;
    }
    for (rec, &k) in &counts {
        let p = exact.get(rec).copied().unwrap_or(0.0);
        assert!(
            p > 1e-12,
            "{what}: impossible record {rec:?} sampled {k} times"
        );
    }
    for (rec, &p) in &exact {
        let k = counts.get(rec).copied().unwrap_or(0) as f64;
        let mu = p * shots as f64;
        let sd = (shots as f64 * p * (1.0 - p)).sqrt().max(1.0);
        assert!(
            (k - mu).abs() < 6.0 * sd + 1.0,
            "{what}: record {rec:?}: {k} of {shots}, expected {mu:.1}"
        );
    }
    true
}

#[test]
fn audit_repeat_pipeline_against_reference() {
    let mut rng = rng_for(3);
    for it in 0..60 * iters() {
        let n = rng.random_range(2..=5);
        // unitary: amplitudes and expectations
        let c = structured(&mut rng, n, false);
        let mut want = RefSv::new(n);
        for g in c.gates() {
            apply(&mut want, g);
        }
        let xs: Vec<u128> = (0..1u128 << n).collect();
        let got = simulate_with(&c, &Request::Amplitudes(xs), &Budget::default(), &ro(1))
            .expect("amplitudes");
        let Output::Amplitudes(a) = got.output else {
            unreachable!()
        };
        let d = want
            .a
            .iter()
            .zip(&a)
            .map(|(x, y)| (x - y).norm())
            .fold(0.0, f64::max);
        assert!(d < 1e-10, "#{it} amplitudes {d}\n{:?}", c.ops);
        let len_ = rng.random_range(1..=n);
        let qs = distinct(&mut rng, n, len_);
        let got = simulate_with(
            &c,
            &Request::Expectation(qs.clone()),
            &Budget::default(),
            &ro(1),
        )
        .expect("expectation");
        let Output::Expectation(e) = got.output else {
            unreachable!()
        };
        let pstr: String = (0..n)
            .map(|q| if qs.contains(&q) { 'Z' } else { 'I' })
            .collect();
        let e_ref = want.pauli_expectation(&pstr);
        assert!((e - e_ref).abs() < 1e-9, "#{it} expectation {e} vs {e_ref}");
        // terminal measurement samples
        let mut cm = c.clone();
        cm.measure_all();
        let _ = sampled_ok(&cm, 3000, it as u64, &format!("#{it} terminal"));
        // mid-circuit measurement samples (Clifford -> tableau rounds)
        let cn = structured(&mut rng, n, true);
        let _ = sampled_ok(&cn, 2000, it as u64, &format!("#{it} mid-circuit"));
    }
    eprintln!(
        "sampling checks skipped (reference too large): {}",
        SKIPPED.load(std::sync::atomic::Ordering::Relaxed)
    );
}

/// Clifford rounds with deterministic and random measurements: the
/// steady-state skip must not change the record distribution.
#[test]
fn audit_repeat_steady_state_rounds() {
    let mut rng = rng_for(4);
    for it in 0..40 * iters() {
        let n = rng.random_range(2..=5);
        let anc = n - 1;
        // `scramble`: random Clifford gates inside the round, so rounds may
        // stay random (must never be skipped); else a pure parity check
        // that becomes deterministic after the first round (skipped).
        let scramble = it % 2 == 1;
        let mut round = Circuit::new(n);
        if scramble {
            for g in random_clifford_gates(n, rng.random_range(1..6), &mut rng) {
                round.gate(g);
            }
        }
        for q in 0..anc {
            if rng.random_bool(0.6) {
                round.cnot(q, anc);
            }
        }
        round.measure(anc).reset(anc);
        if rng.random_bool(0.3) {
            round.measure(rng.random_range(0..anc.max(1)));
        }
        let mut c = Circuit::new(n);
        for g in random_clifford_gates(n, rng.random_range(0..8), &mut rng) {
            c.gate(g);
        }
        let reps = if scramble {
            [2, 3, 5][it % 3]
        } else {
            [3, 8, 40][it % 3]
        };
        c.append(&repeated(&round, reps));
        let checked = sampled_ok(&c, 1500, it as u64, &format!("steady #{it}"));
        assert!(checked || scramble, "steady #{it}: reference too large");
    }
}

#[test]
fn audit_repeat_workloads_amplitudes() {
    let gam: Vec<f64> = (0..6).map(|i| 0.1 + 0.05 * i as f64).collect();
    let bet: Vec<f64> = (0..6).map(|i| 0.7 - 0.03 * i as f64).collect();
    for c in [
        trotter(6, 40, 0.05, 0.04),
        qaoa_ring(6, &gam, &bet),
        grover(4, 5, 12),
    ] {
        let mut want = RefSv::new(c.num_qubits);
        for g in c.gates() {
            apply(&mut want, g);
        }
        let xs: Vec<u128> = (0..1u128 << c.num_qubits).collect();
        let got = simulate_with(&c, &Request::Amplitudes(xs), &Budget::default(), &ro(1))
            .expect("amplitudes");
        let Output::Amplitudes(a) = got.output else {
            unreachable!()
        };
        let d = want
            .a
            .iter()
            .zip(&a)
            .map(|(x, y)| (x - y).norm())
            .fold(0.0, f64::max);
        assert!(d < 1e-10, "workload amplitudes {d}");
    }
    let q = qec_memory(3, 10, true);
    assert!(sampled_ok(&q, 1000, 1, "qec_memory d=3"));
}

/// Repeated terminal measurements: a repeat of `Measure` ops after a
/// repeated non-Clifford block must not make `simulate_with` fail.
#[test]
fn audit_repeat_repeated_terminal_measurements() {
    let mut blk = Circuit::new(3);
    blk.h(0).t(0).cnot(0, 1).ry(2, 0.3).cnot(1, 2);
    let mut c = repeated(&blk, 40);
    for _ in 0..10 {
        c.measure(0);
    }
    c.measure(1).measure(2);
    assert!(sampled_ok(&c, 2000, 7, "terminal measure repeat"));
}
