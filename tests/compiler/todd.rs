//! Exactness of the T-count optimiser (`compile::todd`): every output is
//! checked three ways against its input (path-sum canonical form, exact
//! basis-state simulation over Z[ω], and the floating-point state-vector
//! engine on random states), corrupted outputs must fail every check, and
//! the benchmark T-counts must not regress.

#[path = "../common/mod.rs"]
mod common;

use common::*;
use num_complex::Complex64;
use qsim_lab::compile::todd::verify::{
    basis_equivalent, equivalent, path_sum, vgates_from_circuit, vgates_from_qc, VGate,
};
use qsim_lab::compile::todd::{self, PGate, PhaseCircuit, ToddOptions};
use qsim_lab::io::qc::parse_qc;
use qsim_lab::{Circuit, Gate, Op, StateVectorF64};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn data(name: &str) -> String {
    let p = format!(
        "{}/research/data/todd/circuits/{name}.qc",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

/// `max_x |A x - ω^g B x|` over `states` random universal input states.
fn sv_check(a: &Circuit, b: &Circuit, global: u8, states: usize, seed: u64) -> f64 {
    let n = a.num_qubits;
    let ph = Complex64::from_polar(1.0, global as f64 * std::f64::consts::FRAC_PI_4);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut worst = 0.0f64;
    for _ in 0..states {
        let prep = random_universal(n, 3 * n + 4, &mut rng);
        let mut sa = StateVectorF64::new(n);
        sa.apply_circuit(&prep).unwrap();
        let mut sb = sa.clone();
        sa.apply_circuit(a).unwrap();
        sb.apply_circuit(b).unwrap();
        let d = sa
            .amplitudes()
            .iter()
            .zip(sb.amplitudes())
            .map(|(x, y)| (x - ph * y).norm())
            .fold(0.0, f64::max);
        worst = worst.max(d);
    }
    worst
}

fn basis_inputs(n: usize, max: usize, seed: u64) -> Vec<u128> {
    if n <= 20 && (1usize << n) <= max {
        (0..(1u128 << n)).collect()
    } else {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..max)
            .map(|_| rng.random::<u128>() & ((1u128 << n) - 1))
            .collect()
    }
}

/// Optimises, then checks the output with all three methods.
fn optimise_and_check(pc: &PhaseCircuit, opts: &ToddOptions, sv_states: usize) -> (Circuit, u8) {
    let (out, rep) = todd::optimize(pc, opts);
    let g = todd::verify_output(pc, &out).expect("path-sum equivalence");
    assert_eq!(g, rep.global_phase, "reported global phase");
    assert!(rep.t_output <= rep.t_folded && rep.t_folded <= rep.t_input);
    let n = pc.num_qubits;
    let vo = vgates_from_circuit(&out).unwrap();
    let inputs = basis_inputs(n, 1 << 12, 5);
    assert_eq!(
        basis_equivalent(n, &pc.to_vgates(), &vo, &inputs),
        Ok(g),
        "exact basis-state check"
    );
    if n <= 12 && sv_states > 0 {
        let (orig, g0) = pc.to_circuit();
        // pc = ω^g0 orig, pc = ω^g out  =>  orig = ω^(g - g0) out
        let err = sv_check(&orig, &out, (g + 8 - g0) % 8, sv_states, 11);
        assert!(err < 1e-9, "state-vector check: {err}");
    }
    (out, g)
}

#[test]
fn benchmark_circuits_are_exact_and_do_not_regress() {
    // (name, best T-count reached by this optimiser when the test was
    // written; the test fails if a change makes it worse)
    let cases: &[(&str, usize)] = &[
        ("tof_3", 15),
        ("barenco_tof_3", 16),
        ("mod5_4", 8),
        ("tof_4", 23),
        ("barenco_tof_4", 28),
        ("mod_mult_55", 30),
        ("vbe_adder_3", 24),
        ("qft_4", 66),
        ("rc_adder_6", 47),
    ];
    for &(name, bound) in cases {
        let qc = parse_qc(&data(name)).unwrap();
        let mut pc = PhaseCircuit::from_qc(&qc);
        let before = pc.clone();
        if pc.reduce_hadamards() > 0 {
            // the rewrite is checked on every basis input
            let n = pc.num_qubits;
            let inputs: Vec<u128> = (0..(1u128 << n)).collect();
            assert_eq!(
                basis_equivalent(n, &before.to_vgates(), &pc.to_vgates(), &inputs),
                Ok(0),
                "{name}: Hadamard rewrite"
            );
        }
        let opts = ToddOptions {
            restarts: 4,
            ..Default::default()
        };
        let (out, _) = optimise_and_check(&pc, &opts, 8);
        let t = out
            .gates()
            .filter(|g| matches!(g, Gate::T(_) | Gate::Tdg(_)))
            .count();
        assert!(t <= bound, "{name}: T-count {t} > {bound}");
        // the .qc reader's own expansion agrees with the IR
        let a = path_sum(qc.num_qubits(), &vgates_from_qc(&qc), None);
        let b = path_sum(pc.num_qubits, &pc.to_vgates(), None);
        assert_eq!(equivalent(&a, &b), Some(0));
    }
}

fn random_phase_circuit(n: usize, len: usize, rng: &mut StdRng) -> PhaseCircuit {
    let mut gates = Vec::with_capacity(len);
    for _ in 0..len {
        let q = rng.random_range(0..n);
        let mut other = |not: &[usize], rng: &mut StdRng| loop {
            let x = rng.random_range(0..n);
            if !not.contains(&x) {
                break x;
            }
        };
        let g = match rng.random_range(0..12) {
            0 | 1 => PGate::H(q),
            2 => PGate::X(q),
            3 | 4 => PGate::Phase(q, rng.random_range(1..8)),
            5..=7 if n >= 2 => PGate::Cnot(q, other(&[q], rng)),
            8 if n >= 2 => PGate::Cz(q, other(&[q], rng)),
            9 if n >= 2 => PGate::Swap(q, other(&[q], rng)),
            10 | 11 if n >= 3 => {
                let b = other(&[q], rng);
                let c = other(&[q, b], rng);
                PGate::Ccz(q, b, c)
            }
            _ => PGate::Phase(q, 1),
        };
        gates.push(g);
    }
    PhaseCircuit {
        num_qubits: n,
        gates,
        global: rng.random_range(0..8),
    }
}

#[test]
fn random_circuits_are_exact() {
    let mut rng = StdRng::seed_from_u64(base_seed() ^ 0x70dd);
    for it in 0..(60 * iters()) {
        let n = 1 + it % 7;
        let len = 5 + (it * 7) % 60;
        let pc = random_phase_circuit(n, len, &mut rng);
        let opts = ToddOptions {
            restarts: it % 3,
            seed: it as u64,
            ..Default::default()
        };
        optimise_and_check(&pc, &opts, 4);
    }
}

#[test]
fn hadamard_free_ccz_networks_reach_known_counts() {
    // CCZ·CCZ on the same qubits is the identity.
    let mut pc = PhaseCircuit {
        num_qubits: 3,
        gates: vec![PGate::Ccz(0, 1, 2), PGate::Ccz(2, 0, 1)],
        global: 0,
    };
    let (out, _) = optimise_and_check(&pc, &ToddOptions::default(), 4);
    assert_eq!(out.gates().filter(|g| !g.is_clifford()).count(), 0);
    // CCZ(0,1,2)·CCZ(0,1,3) = CCZ(0,1,2⊕3) up to CNOTs: 7 T.
    pc.num_qubits = 4;
    pc.gates = vec![PGate::Ccz(0, 1, 2), PGate::Ccz(0, 1, 3)];
    let (out, _) = optimise_and_check(&pc, &ToddOptions::default(), 4);
    assert_eq!(out.gates().filter(|g| !g.is_clifford()).count(), 7);
}

/// Every single-gate corruption of a verified output is rejected by both
/// exact checks.
#[test]
fn corrupted_outputs_fail_verification() {
    let qc = parse_qc(&data("mod5_4")).unwrap();
    let pc = PhaseCircuit::from_qc(&qc);
    let (out, g) = optimise_and_check(&pc, &ToddOptions::default(), 0);
    let n = pc.num_qubits;
    let reference = path_sum(n, &pc.to_vgates(), None);
    let inputs: Vec<u128> = (0..(1u128 << n)).collect();
    let mut rng = StdRng::seed_from_u64(99);
    let mut mutants = 0;
    for i in 0..out.ops.len() {
        let Op::Gate(gate) = out.ops[i] else { continue };
        let replacement: Vec<Gate> = match gate {
            Gate::T(q) => vec![Gate::Tdg(q)],
            Gate::Tdg(q) => vec![Gate::T(q)],
            Gate::S(q) => vec![Gate::Sdg(q)],
            Gate::Sdg(q) => vec![Gate::S(q)],
            Gate::Z(q) => vec![],
            Gate::H(q) => vec![Gate::H((q + 1) % n)],
            Gate::Cnot(a, b) => vec![Gate::Cnot(b, a)],
            Gate::X(q) => vec![],
            _ => continue,
        };
        let mut bad = out.clone();
        bad.ops.splice(i..=i, replacement.into_iter().map(Op::Gate));
        let vb = vgates_from_circuit(&bad).unwrap();
        // a mutant can be equal to the original by accident (e.g. a CNOT
        // reversed between equal wire values); only count real changes
        let changed =
            basis_equivalent(n, &vgates_from_circuit(&out).unwrap(), &vb, &inputs) != Ok(0);
        if !changed {
            continue;
        }
        mutants += 1;
        let ps = path_sum(n, &vb, None);
        assert_eq!(
            equivalent(&reference, &ps),
            None,
            "path sum missed mutant {i}"
        );
        assert!(
            basis_equivalent(n, &pc.to_vgates(), &vb, &inputs).is_err(),
            "basis check missed mutant {i}"
        );
        assert!(todd::verify_output(&pc, &bad).is_err());
    }
    assert!(mutants > 20, "only {mutants} effective mutants");
    // an extra global phase is reported, not hidden
    let mut shifted = out.clone();
    shifted.ops.push(Op::Gate(Gate::X(0)));
    shifted.ops.push(Op::Gate(Gate::Z(0)));
    shifted.ops.push(Op::Gate(Gate::X(0)));
    shifted.ops.push(Op::Gate(Gate::Z(0)));
    let gv = todd::verify_output(&pc, &shifted).unwrap();
    assert_eq!(gv, (g + 4) % 8);
    let _ = rng.random::<u8>();
}

#[test]
fn qc_round_trip_and_ccz_expansion() {
    let qc = parse_qc(&data("barenco_tof_3")).unwrap();
    let pc = PhaseCircuit::from_qc(&qc);
    let (c, g0) = pc.to_circuit();
    assert_eq!(c.t_count(), qc.t_count());
    let back = PhaseCircuit::from_circuit(&c).unwrap();
    let a = path_sum(pc.num_qubits, &pc.to_vgates(), None);
    let b = path_sum(back.num_qubits, &back.to_vgates(), None);
    assert_eq!(equivalent(&a, &b), Some(g0));
    // and through the floating-point engine, all basis states
    let qcc = qc.to_circuit();
    let err = sv_check(&qcc, &c, g0, 6, 3);
    assert!(err < 1e-9);
    let v = vgates_from_qc(&qc);
    assert!(v.iter().any(|g| matches!(g, VGate::Ccz(..))));
}

/// Pauli-frame mode: the output's Hadamard structure differs from the
/// input's, so it is checked semantically: exact Z[ω] simulation of every
/// basis input, plus the floating-point engine on random states.
fn pauli_check(pc: &PhaseCircuit, opts: &ToddOptions) -> Circuit {
    let (out, rep) = todd::pauli::optimize_pauli(pc, opts);
    let n = pc.num_qubits;
    let vo = vgates_from_circuit(&out).unwrap();
    let inputs = basis_inputs(n, 1 << 12, 9);
    let g = basis_equivalent(n, &pc.to_vgates(), &vo, &inputs).expect("exact basis-state check");
    assert_eq!(g, rep.global_phase, "reported global phase");
    assert!(rep.t_output <= rep.t_merged && rep.t_merged <= rep.t_input);
    if n <= 12 {
        let (orig, g0) = pc.to_circuit();
        let err = sv_check(&orig, &out, (g + 8 - g0) % 8, 6, 17);
        assert!(err < 1e-9, "state-vector check: {err}");
    }
    out
}

#[test]
fn pauli_mode_random_circuits_are_exact() {
    let mut rng = StdRng::seed_from_u64(base_seed() ^ 0x9a01);
    for it in 0..(60 * iters()) {
        let n = 1 + it % 7;
        let len = 5 + (it * 11) % 70;
        let pc = random_phase_circuit(n, len, &mut rng);
        let opts = ToddOptions {
            restarts: it % 3,
            seed: it as u64,
            reassign_passes: it % 2,
            absorb_cliffords: it % 4 != 3,
            ..Default::default()
        };
        pauli_check(&pc, &opts);
    }
}

#[test]
fn pauli_mode_benchmarks_are_exact_and_do_not_regress() {
    let cases: &[(&str, usize)] = &[
        ("mod5_4", 8),
        ("tof_3", 15),
        ("barenco_tof_3", 16),
        ("vbe_adder_3", 24),
        ("mod_mult_55", 28),
        ("csla_mux_3", 49),
    ];
    for &(name, bound) in cases {
        let pc = PhaseCircuit::from_qc(&parse_qc(&data(name)).unwrap());
        let opts = ToddOptions {
            restarts: 2,
            reassign_passes: 1,
            ..Default::default()
        };
        let out = pauli_check(&pc, &opts);
        let t = out
            .gates()
            .filter(|g| matches!(g, Gate::T(_) | Gate::Tdg(_)))
            .count();
        assert!(t <= bound, "{name}: T-count {t} > {bound}");
    }
}

#[test]
fn diagonalisation_and_frames_agree_with_the_state_vector() {
    use qsim_lab::compile::todd::pauli::{diagonalize, rotations, Rotation};
    // a circuit's rotation list, re-synthesised one rotation at a time in
    // the input frame and followed by the Clifford skeleton, is the circuit
    let mut rng = StdRng::seed_from_u64(31);
    for it in 0..40 {
        let n = 1 + it % 5;
        let pc = random_phase_circuit(n, 30, &mut rng);
        let (rots, global) = rotations(&pc);
        let mut c = Circuit::new(n);
        let mut g = global;
        for Rotation { axis, k } in &rots {
            let (dg, rows) = diagonalize(std::slice::from_ref(axis));
            let k = if rows[0].sign {
                g = (g + k) % 8;
                (8 - k) % 8
            } else {
                *k
            };
            let gates: Vec<Gate> = dg
                .iter()
                .map(|d| match *d {
                    qsim_lab::compile::todd::pauli::DGate::H(q) => Gate::H(q),
                    qsim_lab::compile::todd::pauli::DGate::S(q) => Gate::S(q),
                    qsim_lab::compile::todd::pauli::DGate::Sdg(q) => Gate::Sdg(q),
                    qsim_lab::compile::todd::pauli::DGate::Cnot(a, b) => Gate::Cnot(a, b),
                })
                .collect();
            for &gt in &gates {
                c.gate(gt);
            }
            let wires: Vec<usize> = rows[0].z.ones().collect();
            let t = wires[0];
            for &w in &wires[1..] {
                c.cnot(w, t);
            }
            c.gate(Gate::Phase(t, k as f64 * std::f64::consts::FRAC_PI_4));
            for &w in wires[1..].iter().rev() {
                c.cnot(w, t);
            }
            for &gt in gates.iter().rev() {
                c.gate(match gt {
                    Gate::S(q) => Gate::Sdg(q),
                    Gate::Sdg(q) => Gate::S(q),
                    other => other,
                });
            }
        }
        for gate in &pc.gates {
            match *gate {
                PGate::H(q) => {
                    c.h(q);
                }
                PGate::X(q) => {
                    c.x(q);
                }
                PGate::Cnot(a, b) => {
                    c.cnot(a, b);
                }
                PGate::Swap(a, b) => {
                    c.swap(a, b);
                }
                PGate::Cz(a, b) => {
                    c.cz(a, b);
                }
                PGate::Phase(q, k) if k % 2 == 0 => {
                    c.gate(Gate::Phase(q, k as f64 * std::f64::consts::FRAC_PI_4));
                }
                PGate::Phase(q, k) => {
                    c.gate(Gate::Phase(q, (k - 1) as f64 * std::f64::consts::FRAC_PI_4));
                }
                PGate::Ccz(..) => {}
            }
        }
        let (orig, g0) = pc.to_circuit();
        // pc = ω^g · c and pc = ω^g0 · orig
        let err = sv_check(&orig, &c, (g + 8 - g0) % 8, 3, it as u64);
        assert!(err < 1e-9, "iteration {it}: {err}");
    }
}

#[test]
fn hadamard_rewrites_are_exact() {
    let mut rng = StdRng::seed_from_u64(base_seed() ^ 0x4ad);
    let mut removed_total = 0;
    for it in 0..(200 * iters()) {
        let n = 2 + it % 5;
        // bias towards Hadamard sandwiches around CNOT blocks
        let mut pc = random_phase_circuit(n, 10 + it % 40, &mut rng);
        for _ in 0..3 {
            let a = rng.random_range(0..n);
            let b = (a + 1 + rng.random_range(0..n - 1)) % n;
            let at = rng.random_range(0..=pc.gates.len());
            pc.gates.splice(
                at..at,
                [
                    PGate::H(a),
                    PGate::H(b),
                    PGate::Cnot(a, b),
                    PGate::H(a),
                    PGate::H(b),
                ],
            );
            let at = rng.random_range(0..=pc.gates.len());
            pc.gates.splice(
                at..at,
                [PGate::H(b), PGate::Cnot(a, b), PGate::X(b), PGate::H(b)],
            );
        }
        let before = pc.clone();
        removed_total += pc.reduce_hadamards();
        let inputs: Vec<u128> = (0..(1u128 << n)).collect();
        assert_eq!(
            basis_equivalent(n, &before.to_vgates(), &pc.to_vgates(), &inputs),
            Ok(0),
            "iteration {it}"
        );
        assert_eq!(before.t_count(), pc.t_count());
    }
    assert!(
        removed_total > 100,
        "rules fired only {removed_total} times"
    );
}

/// The record circuits committed in `research/data/todd/outputs/` are
/// re-verified against the originals from scratch: T-count, then the
/// path-sum identity when the Hadamards match, else exact simulation of
/// every basis input (all records with a changed Hadamard structure have
/// n ≤ 20).
#[test]
fn committed_record_circuits_verify() {
    let dir = format!("{}/research/data/todd/outputs", env!("CARGO_MANIFEST_DIR"));
    let Ok(listing) = std::fs::read_to_string(format!("{dir}/records.txt")) else {
        panic!("{dir}/records.txt missing");
    };
    let mut checked = 0;
    for line in listing
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        let (name, t_claim) = (f[0], f[1].parse::<usize>().unwrap());
        let orig = PhaseCircuit::from_qc(&parse_qc(&data(name)).unwrap());
        let text = std::fs::read_to_string(format!("{dir}/{name}.qc")).unwrap();
        let out_qc = parse_qc(&text).unwrap();
        assert_eq!(
            out_qc.num_qubits(),
            orig.num_qubits,
            "{name}: no ancillas added"
        );
        assert_eq!(out_qc.t_count(), t_claim, "{name}: T-count");
        let out = out_qc.to_circuit();
        let n = orig.num_qubits;
        let vo = vgates_from_circuit(&out).unwrap();
        let a = path_sum(n, &orig.to_vgates(), None);
        let b = path_sum(n, &vo, None);
        if equivalent(&a, &b).is_none() {
            assert!(n <= 20, "{name}: needs the path-sum identity");
            let inputs: Vec<u128> = (0..(1u128 << n)).collect();
            assert!(
                basis_equivalent(n, &orig.to_vgates(), &vo, &inputs).is_ok(),
                "{name}: exact basis-state check"
            );
        }
        checked += 1;
    }
    assert!(checked >= 2);
}
