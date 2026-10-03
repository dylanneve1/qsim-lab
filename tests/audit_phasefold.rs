//! Round-4 audit of `compile::phase_fold`: independent differential fuzz
//! against the naive reference state vector of `tests/audit_common`.
//!
//! * unitary circuits (all gate kinds, incl. iSWAP / SX / U / Toffoli /
//!   controlled phases): full unitary, every basis column, n <= 6, and a
//!   random input state up to n = 12, global phase included;
//! * arithmetic circuits (Cuccaro adders, Toffoli ladders, Shor ripple
//!   controlled modular multiplication), lowered to Clifford+T;
//! * non-unitary circuits (mid-circuit measurement, reset, classically
//!   controlled gates incl. classically controlled rotations, Pauli flips,
//!   blocks containing measurements repeated many times): every branch of
//!   the outcome tree (measurement, reset and flip outcomes) is enumerated
//!   and the unnormalised branch state compared, global phase included.
//!   That is equality of the whole instrument, not only of the outcome
//!   distribution.
//!
//! `QSIM_FUZZ_ITERS` scales the number of cases (default 1 = CI size).

#![allow(clippy::needless_range_loop)]

mod audit_common;
mod audit_r4;

use audit_common::{iters, RefSv};
use audit_r4::{apply, basis, compare, random_state};
use num_complex::Complex64 as C;
use qsim_lab::compile::plan::{compile_unitary, PlanOptions};
use qsim_lab::compile::{optimize, phase_fold};
use qsim_lab::gate::toffoli_clifford_t;
use qsim_lab::shor_ripple::{controlled_ua, cuccaro_add, RippleLayout};
use qsim_lab::{Circuit, Gate, Op};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::{FRAC_PI_4, PI};

/// Branch-tree size guard: ops that split the reference's branch tree.
fn stochastic(c: &Circuit) -> usize {
    c.ops
        .iter()
        .filter(|o| !matches!(o, Op::Gate(_) | Op::ClassicControlled { .. }))
        .count()
}

/// At most this many branching ops per case (2^k branches in the reference).
const MAX_BRANCHING: usize = 11;

fn nc(c: &Circuit) -> usize {
    c.gates().filter(|g| !g.is_clifford()).count()
}

fn t_like(c: &Circuit) -> usize {
    c.gates().filter(|g| g.is_t()).count()
}

/// Folds `c` and checks it: full unitary for n <= 6, else two random
/// states (or the branch tree for non-unitary circuits).
fn check(c: &Circuit, rng: &mut StdRng, what: &str) {
    let o = phase_fold(c);
    let n = c.num_qubits;
    assert!(
        nc(&o.circuit) <= nc(c),
        "{what}: non-Clifford count grew {} -> {}",
        nc(c),
        nc(&o.circuit)
    );
    let mut worst = 0.0f64;
    if n <= 6 {
        for col in 0..1usize << n {
            worst = worst.max(compare(c, &o.circuit, o.global_phase, &basis(n, col)));
        }
    } else {
        for _ in 0..2 {
            worst = worst.max(compare(
                c,
                &o.circuit,
                o.global_phase,
                &random_state(n, rng),
            ));
        }
    }
    assert!(
        worst < 1e-9,
        "{what}: diff {worst}\nin:  {:?}\nout: {:?} phase {}",
        c.ops,
        o.circuit.ops,
        o.global_phase
    );
}

// ------------------------------------------------------------ generators

fn angle(rng: &mut StdRng) -> f64 {
    match rng.random_range(0..6) {
        0 | 1 => rng.random_range(-12..12) as f64 * FRAC_PI_4,
        2 => rng.random_range(-12..12) as f64 * FRAC_PI_4 + 1e-13,
        3 => rng.random_range(-1e3..1e3),
        4 => rng.random_range(-12..12) as f64 * PI / 8.0,
        _ => rng.random_range(-PI..PI),
    }
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

/// Any gate the library has, biased towards phase-polynomial gates.
fn any_gate(rng: &mut StdRng, n: usize) -> Gate {
    let q = rng.random_range(0..n);
    let top = if n >= 3 {
        30
    } else if n == 2 {
        27
    } else {
        17
    };
    let k = rng.random_range(0..top);
    match k {
        0 | 1 => Gate::T(q),
        2 | 3 => Gate::Tdg(q),
        4 => Gate::S(q),
        5 => Gate::Sdg(q),
        6 => Gate::Z(q),
        7 => Gate::X(q),
        8 => Gate::Y(q),
        9 => Gate::H(q),
        10 => Gate::Rz(q, angle(rng)),
        11 => Gate::Phase(q, angle(rng)),
        12 => Gate::Rx(q, angle(rng)),
        13 => Gate::Ry(q, angle(rng)),
        14 => Gate::Sx(q),
        15 => Gate::U(q, angle(rng), angle(rng), angle(rng)),
        16 => Gate::I(q),
        17..=20 => {
            let v = distinct(rng, n, 2);
            Gate::Cnot(v[0], v[1])
        }
        21 => {
            let v = distinct(rng, n, 2);
            Gate::Cz(v[0], v[1])
        }
        22 => {
            let v = distinct(rng, n, 2);
            Gate::Swap(v[0], v[1])
        }
        23 => {
            let v = distinct(rng, n, 2);
            Gate::ISwap(v[0], v[1])
        }
        24 => {
            let v = distinct(rng, n, 2);
            Gate::ISwapdg(v[0], v[1])
        }
        25 | 26 => {
            let v = distinct(rng, n, 2);
            Gate::CPhase(v[0], v[1], angle(rng))
        }
        _ => {
            let v = distinct(rng, n, 3);
            Gate::Ccx(v[0], v[1], v[2])
        }
    }
}

/// Clifford+T only, with a few `H` so parities recur.
fn ct_gate(rng: &mut StdRng, n: usize, h_rate: f64) -> Gate {
    let q = rng.random_range(0..n);
    if n >= 2 && rng.random_bool(0.45) {
        let v = distinct(rng, n, 2);
        return match rng.random_range(0..6) {
            0 => Gate::Swap(v[0], v[1]),
            1 => Gate::Cz(v[0], v[1]),
            _ => Gate::Cnot(v[0], v[1]),
        };
    }
    if rng.random_bool(h_rate) {
        return Gate::H(q);
    }
    match rng.random_range(0..8) {
        0 | 1 => Gate::T(q),
        2 | 3 => Gate::Tdg(q),
        4 => Gate::S(q),
        5 => Gate::X(q),
        6 => Gate::Y(q),
        _ => Gate::Z(q),
    }
}

/// A non-unitary op (or a gate) for mixed circuits.
fn mixed_op(rng: &mut StdRng, n: usize, nmeas: &mut usize) -> Op {
    let q = rng.random_range(0..n);
    match rng.random_range(0..40) {
        0..=2 => {
            *nmeas += 1;
            Op::Measure(q)
        }
        3 | 4 => Op::Reset(q),
        5..=7 if *nmeas > 0 => {
            let g = if rng.random_bool(0.5) {
                ct_gate(rng, n, 0.1)
            } else {
                any_gate(rng, n)
            };
            Op::ClassicControlled {
                gate: g,
                meas_index: rng.random_range(0..*nmeas),
                target_value: rng.random_bool(0.5),
            }
        }
        8 => Op::XFlip(q, rng.random_range(0.05..0.5)),
        9 => Op::ZFlip(q, rng.random_range(0.05..0.5)),
        10 => Op::YFlip(q, rng.random_range(0.05..0.5)),
        11..=25 => Op::Gate(ct_gate(rng, n, 0.15)),
        _ => Op::Gate(any_gate(rng, n)),
    }
}

fn lower_ccx(c: &Circuit) -> Circuit {
    let mut o = Circuit::new(c.num_qubits);
    for op in &c.ops {
        match *op {
            Op::Gate(Gate::Ccx(a, b, t)) => {
                for g in toffoli_clifford_t(a, b, t) {
                    o.gate(g);
                }
            }
            op => o.ops.push(op),
        }
    }
    o
}

// ------------------------------------------------------------ tests

fn rng_for(test: u64) -> StdRng {
    StdRng::seed_from_u64(audit_common::base_seed() ^ (test << 32))
}

#[test]
fn audit_fold_unitary_all_gates() {
    let mut rng = rng_for(1);
    for it in 0..120 * iters() {
        let n = rng.random_range(1..=8);
        let len = rng.random_range(1..=120);
        let mut c = Circuit::new(n);
        for _ in 0..len {
            c.gate(any_gate(&mut rng, n));
        }
        check(&c, &mut rng, &format!("all-gates #{it}"));
    }
}

#[test]
fn audit_fold_clifford_t_dense_parities() {
    let mut rng = rng_for(2);
    for it in 0..120 * iters() {
        let n = rng.random_range(2..=12);
        let len = rng.random_range(10..=300);
        let h_rate = [0.0, 0.02, 0.1][it % 3];
        let mut c = Circuit::new(n);
        for _ in 0..len {
            c.gate(ct_gate(&mut rng, n, h_rate));
        }
        let o = phase_fold(&c);
        if h_rate == 0.0 {
            // pure CNOT+T: one rotation per distinct parity at most
            assert!(t_like(&o.circuit) < (1usize << n));
        }
        check(&c, &mut rng, &format!("ct #{it}"));
    }
}

#[test]
fn audit_fold_non_unitary_instrument() {
    let mut rng = rng_for(3);
    let (mut it, mut done) = (0, 0);
    while done < 150 * iters() {
        it += 1;
        let n = rng.random_range(1..=6);
        let len = rng.random_range(1..=45);
        let mut c = Circuit::new(n);
        let mut nmeas = 0;
        for _ in 0..len {
            let op = mixed_op(&mut rng, n, &mut nmeas);
            c.ops.push(op);
        }
        if stochastic(&c) > MAX_BRANCHING {
            continue;
        }
        check(&c, &mut rng, &format!("mixed #{it}"));
        done += 1;
    }
}

/// A block with measurements, resets and classically controlled rotations,
/// repeated many times (like a QEC / repeat-until-success round).
#[test]
fn audit_fold_repeated_measuring_blocks() {
    let mut rng = rng_for(4);
    let (mut it, mut done) = (0usize, 0);
    while done < 60 * iters() {
        it += 1;
        let n = rng.random_range(2..=5);
        let mut body = Vec::new();
        let mut nmeas = 0;
        for _ in 0..rng.random_range(3..=10) {
            body.push(mixed_op(&mut rng, n, &mut nmeas));
        }
        // a T on a "data" qubit in every round, before and after the
        // measuring part, so folding has something to merge across rounds
        let reps = [1, 2, 3, 5][it % 4];
        let mut c = Circuit::new(n);
        let mut seen = 0;
        for _ in 0..reps {
            c.t(0);
            for op in &body {
                let op = match *op {
                    Op::ClassicControlled {
                        gate,
                        meas_index,
                        target_value,
                    } => Op::ClassicControlled {
                        gate,
                        meas_index: meas_index + seen,
                        target_value,
                    },
                    o => o,
                };
                c.ops.push(op);
            }
            seen += nmeas;
            c.cnot(0, n - 1).t(n - 1).cnot(0, n - 1);
        }
        if stochastic(&c) > MAX_BRANCHING {
            continue;
        }
        check(&c, &mut rng, &format!("rounds #{it}"));
        done += 1;
    }
}

/// Hand-made adversarial cases around measurement / reset / classical
/// control and constant wires.
#[test]
fn audit_fold_adversarial_cases() {
    let mut rng = rng_for(5);
    let mut cases: Vec<Circuit> = Vec::new();
    // T before and after a measurement of the same wire
    let mut c = Circuit::new(2);
    c.h(0).t(0).measure(0).t(0).h(0).t(0);
    cases.push(c);
    // parity reconstructed after a reset of a wire that held part of it
    let mut c = Circuit::new(3);
    c.h(0)
        .h(1)
        .h(2)
        .cnot(0, 1)
        .t(1)
        .cnot(1, 2)
        .reset(1)
        .cnot(2, 1)
        .t(1);
    c.cnot(0, 1).t(1);
    cases.push(c);
    // classically controlled T and X on a wire carrying a folded parity
    let mut c = Circuit::new(3);
    c.h(0).h(1).cnot(0, 2).t(2).measure(1);
    c.ops.push(Op::ClassicControlled {
        gate: Gate::X(2),
        meas_index: 0,
        target_value: true,
    });
    c.t(2);
    c.ops.push(Op::ClassicControlled {
        gate: Gate::T(0),
        meas_index: 0,
        target_value: false,
    });
    c.cnot(0, 2).tdg(0).t(2);
    cases.push(c);
    // constant wires: X then rotations, folded with negated angle
    let mut c = Circuit::new(2);
    c.x(0).t(0).cnot(0, 1).x(0).t(0).y(1).t(1).cnot(0, 1).t(1);
    cases.push(c);
    // reset gives a constant |0>: rotations on it are trivial
    let mut c = Circuit::new(2);
    c.h(0).reset(0).t(0).x(0).t(0).cnot(0, 1).tdg(1);
    cases.push(c);
    // flip noise between two rotations on the same parity
    let mut c = Circuit::new(2);
    c.h(0).h(1).cnot(0, 1).t(1);
    c.ops.push(Op::XFlip(0, 0.3));
    c.t(1).cnot(0, 1).t(1);
    c.ops.push(Op::ZFlip(1, 0.2));
    c.t(1);
    cases.push(c);
    // Rz global phase with constant wire
    let mut c = Circuit::new(1);
    c.x(0)
        .rz(0, 0.3)
        .x(0)
        .rz(0, 0.3)
        .rz(0, 2.0 * PI)
        .phase(0, -0.6);
    cases.push(c);
    // measured qubit reused as a control after measurement
    let mut c = Circuit::new(3);
    c.h(0)
        .cnot(0, 1)
        .t(1)
        .measure(0)
        .cnot(0, 1)
        .cnot(0, 1)
        .t(1)
        .ccx(0, 1, 2)
        .t(2);
    cases.push(c);
    for (i, c) in cases.iter().enumerate() {
        check(c, &mut rng, &format!("adversarial #{i}"));
    }
}

/// Arithmetic: Cuccaro adders (all inputs), Toffoli ladders, and the
/// controlled modular multiplication of Shor's ripple layout, lowered to
/// Clifford+T, on up to 12 qubits.
#[test]
fn audit_fold_arithmetic() {
    let mut rng = rng_for(6);
    // Cuccaro adder, n-bit: 2n + 2 qubits
    for nb in 1..=4 {
        let a: Vec<usize> = (0..nb).collect();
        let b: Vec<usize> = (nb..2 * nb + 1).collect();
        let c0 = 2 * nb + 1;
        let mut c = Circuit::new(2 * nb + 2);
        cuccaro_add(&mut c, &a, &b, c0);
        let low = lower_ccx(&c);
        let o = phase_fold(&low);
        eprintln!(
            "cuccaro n={nb}: T-like {} -> {}",
            t_like(&low),
            t_like(&o.circuit)
        );
        check(&low, &mut rng, &format!("cuccaro {nb}"));
        // also with superposed inputs (H on a and b) prepended
        let mut s = Circuit::new(2 * nb + 2);
        for &q in a.iter().chain(&b) {
            s.h(q);
        }
        s.ops.extend(low.ops.iter().copied());
        check(&s, &mut rng, &format!("cuccaro-sup {nb}"));
    }
    // Shor ripple: controlled modular multiplication, n = 2 (10 qubits)
    let lay = RippleLayout::new(2);
    for (a, nm) in [(2u64, 3u64), (1, 3)] {
        let c = controlled_ua(&lay, lay.ctrl, a, nm);
        let low = lower_ccx(&c);
        let o = phase_fold(&low);
        eprintln!(
            "ripple cmult a={a} N={nm}: {} qubits, T-like {} -> {}",
            low.num_qubits,
            t_like(&low),
            t_like(&o.circuit)
        );
        check(&low, &mut rng, &format!("ripple a={a} N={nm}"));
        // on valid inputs |ctrl>|x>|0...>: fold must act identically
        for x in 0..nm as usize {
            for ctrl in 0..2usize {
                let mut idx = ctrl << lay.ctrl;
                for (i, &q) in lay.x.iter().enumerate() {
                    idx |= ((x >> i) & 1) << q;
                }
                let d = compare(
                    &low,
                    &o.circuit,
                    o.global_phase,
                    &basis(low.num_qubits, idx),
                );
                assert!(d < 1e-9, "ripple basis input {idx}: {d}");
            }
        }
    }
    // Toffoli ladders, compute/uncompute
    for (nn, reps) in [(3, 1), (4, 2), (6, 1), (7, 2)] {
        let k = nn - 2;
        let mut c = Circuit::new(nn + k);
        for q in 0..nn {
            c.h(q);
        }
        for _ in 0..reps {
            c.ccx(0, 1, nn);
            for i in 1..k {
                c.ccx(nn + i - 1, i + 1, nn + i);
            }
            c.cnot(nn + k - 1, 0);
            for i in (1..k).rev() {
                c.ccx(nn + i - 1, i + 1, nn + i);
            }
            c.ccx(0, 1, nn);
        }
        check(&lower_ccx(&c), &mut rng, &format!("ladder {nn}x{reps}"));
    }
}

/// The plan option (`peephole -> fold -> peephole`) against the reference.
#[test]
fn audit_fold_plan_option_amplitudes() {
    let mut rng = rng_for(7);
    for it in 0..40 * iters() {
        let n = rng.random_range(2..=9);
        let mut c = Circuit::new(n);
        for _ in 0..rng.random_range(5..150) {
            let g = if rng.random_bool(0.7) {
                ct_gate(&mut rng, n, 0.1)
            } else {
                any_gate(&mut rng, n)
            };
            c.gate(g);
        }
        let mut reference = RefSv::new(n);
        for g in c.gates() {
            apply(&mut reference, g);
        }
        let opts = PlanOptions {
            phase_fold: true,
            ..PlanOptions::default()
        };
        let plan = compile_unitary(&c, opts).expect("compile");
        let amps = plan.statevector::<f64>().expect("sv");
        let d = reference
            .a
            .iter()
            .zip(amps.amplitudes())
            .map(|(x, y)| (x - y).norm())
            .fold(0.0, f64::max);
        assert!(d < 1e-9, "plan #{it}: {d}");
        // and the peephole result folds exactly too
        let p = optimize(&c);
        let o = phase_fold(&p.circuit);
        let mut s = RefSv::new(n);
        for g in o.circuit.gates() {
            apply(&mut s, g);
        }
        let ph = C::from_polar(1.0, p.global_phase + o.global_phase);
        let d2 = reference
            .a
            .iter()
            .zip(&s.a)
            .map(|(x, y)| (x - ph * y).norm())
            .fold(0.0, f64::max);
        assert!(d2 < 1e-9, "peephole+fold #{it}: {d2}");
    }
}
