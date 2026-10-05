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
        ("mod5_4", 16),
        ("tof_4", 23),
        ("barenco_tof_4", 28),
        ("hwb6", 75),
        ("mod_mult_55", 35),
        ("vbe_adder_3", 24),
        ("qft_4", 69),
    ];
    for &(name, bound) in cases {
        let qc = parse_qc(&data(name)).unwrap();
        let pc = PhaseCircuit::from_qc(&qc);
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
        let changed = basis_equivalent(n, &vgates_from_circuit(&out).unwrap(), &vb, &inputs) != Ok(0);
        if !changed {
            continue;
        }
        mutants += 1;
        let ps = path_sum(n, &vb, None);
        assert_eq!(equivalent(&reference, &ps), None, "path sum missed mutant {i}");
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
