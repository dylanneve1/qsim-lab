//! Audit adapter for PR #1 (improvements-and-optimizations): new gates
//! (I, Sx, Sxdg, U, ISwap, ISwapdg) on every backend, `Circuit::optimize`,
//! OpenQASM round trip and parameter parsing, `reset_all`.
//! Copy to `tests/` together with `tests/audit_common/`.
#![allow(clippy::field_reassign_with_default)]

mod audit_common;

use audit_common::*;
use num_complex::Complex64 as C;
use qsim_lab::engines::pauli_path::{self, PauliSum, DEFAULT_MAX_TERMS};
use qsim_lab::{Circuit, Gate, Mps, Simulator, StateVectorF32, StateVectorF64, Tableau};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

/// Reference application of the new gates, from textbook matrices
/// (Qiskit conventions: U(θ,φ,λ) = [[cos θ/2, −e^{iλ} sin θ/2],
/// [e^{iφ} sin θ/2, e^{i(φ+λ)} cos θ/2]]; iSWAP |01>,|10> → i|10>, i|01>).
fn ref_apply(r: &mut RefSv, g: &Gate) {
    let one_q = |r: &mut RefSv, q: usize, m: [[C; 2]; 2]| {
        let old = r.a.clone();
        for (i, out) in r.a.iter_mut().enumerate() {
            let b = (i >> q) & 1;
            *out = m[b][0] * old[i & !(1 << q)] + m[b][1] * old[i | (1 << q)];
        }
    };
    let h = 0.5;
    match *g {
        Gate::I(_) => {}
        Gate::Sx(q) => one_q(r, q, [[cx(h, h), cx(h, -h)], [cx(h, -h), cx(h, h)]]),
        Gate::Sxdg(q) => one_q(r, q, [[cx(h, -h), cx(h, h)], [cx(h, h), cx(h, -h)]]),
        Gate::U(q, th, ph, la) => {
            let (s, c) = ((th / 2.0).sin(), (th / 2.0).cos());
            let e = |t: f64| cx(t.cos(), t.sin());
            one_q(r, q, [[cx(c, 0.0), -e(la) * s], [e(ph) * s, e(ph + la) * c]])
        }
        Gate::ISwap(a, b) | Gate::ISwapdg(a, b) => {
            let ph = if matches!(g, Gate::ISwap(..)) { cx(0.0, 1.0) } else { cx(0.0, -1.0) };
            let old = r.a.clone();
            for i in 0..old.len() {
                let (ba, bb) = ((i >> a) & 1, (i >> b) & 1);
                if ba != bb {
                    let j = i ^ (1 << a) ^ (1 << b);
                    r.a[j] = old[i] * ph;
                }
            }
        }
        ref g => r.apply(g),
    }
}

fn ref_run(c: &Circuit) -> RefSv {
    let mut r = RefSv::new(c.num_qubits);
    for g in c.gates() {
        ref_apply(&mut r, g);
    }
    r
}

fn new_gate(rng: &mut StdRng, n: usize, clifford_only: bool) -> Gate {
    let q = edge_qubit(rng, n);
    let k = if n == 1 { rng.random_range(0..4) } else { rng.random_range(0..6) };
    let g = match k {
        0 => Gate::I(q),
        1 => Gate::Sx(q),
        2 => Gate::Sxdg(q),
        3 => Gate::U(q, edge_angle(rng), edge_angle(rng), edge_angle(rng)),
        4 => {
            let (a, b) = edge_pair(rng, n);
            Gate::ISwap(a, b)
        }
        _ => {
            let (a, b) = edge_pair(rng, n);
            Gate::ISwapdg(a, b)
        }
    };
    if clifford_only && !g.is_clifford() {
        return new_gate(rng, n, clifford_only);
    }
    g
}

fn mixed_circuit(rng: &mut StdRng, n: usize, depth: usize, clifford_only: bool) -> Circuit {
    let mut c = Circuit::new(n);
    for _ in 0..depth {
        let g = if rng.random_bool(0.4) {
            new_gate(rng, n, clifford_only)
        } else {
            random_gate(rng, n, clifford_only, false)
        };
        c.gate(g);
    }
    c
}

fn global_phase_diff(a: &[C], b: impl Iterator<Item = C>) -> (f64, f64) {
    // returns (max |Δ| as is, max |Δ| after removing the best global phase)
    let b: Vec<C> = b.collect();
    let ov: C = a.iter().zip(&b).map(|(x, y)| x.conj() * y).sum();
    let ph = if ov.norm() > 0.0 { ov / ov.norm() } else { cx(1.0, 0.0) };
    let raw = a.iter().zip(&b).map(|(x, y)| (x - y).norm()).fold(0.0, f64::max);
    let up = a.iter().zip(&b).map(|(x, y)| (x * ph - y).norm()).fold(0.0, f64::max);
    (raw, up)
}

#[test]
fn new_gates_statevector_and_mps_match_reference() {
    for it in 0..20 * iters() {
        for &n in &SIZES {
            let seed = base_seed() ^ 0x9E1 ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let depth = rng.random_range(1..80);
            let c = mixed_circuit(&mut rng, n, depth, false);
            let r = ref_run(&c);
            let mut sv = StateVectorF64::new(n);
            sv.apply_circuit(&c).unwrap();
            let d = max_amp_diff(&r.a, (0..1 << n).map(|i| sv.amplitude(i)));
            assert!(d <= 1e-12, "f64 Δ={d:e} seed={seed} n={n} {:?}", c.ops);
            let mut s32 = StateVectorF32::new(n);
            s32.apply_circuit(&c).unwrap();
            let d = max_amp_diff(&r.a, (0..1 << n).map(|i| s32.amplitude(i)));
            assert!(d <= 1e-5, "f32 Δ={d:e} seed={seed} n={n}");
            if n <= 10 {
                let mut m = Mps::new(n, 1 << (n / 2 + 1));
                m.set_cutoff(0.0);
                c.run(&mut m, &mut rng).unwrap();
                let d = max_amp_diff(&r.a, (0..1u128 << n).map(|i| m.amplitude(i)));
                assert!(d <= 1e-9, "mps Δ={d:e} seed={seed} n={n} {:?}", c.ops);
            }
        }
    }
}

#[test]
fn new_clifford_gates_tableau_exact() {
    for it in 0..25 * iters() {
        for &n in &[1usize, 2, 3, 5, 8, 11] {
            let seed = base_seed() ^ 0x9E2 ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let depth = rng.random_range(1..60);
            let c = mixed_circuit(&mut rng, n, depth, true);
            let r = ref_run(&c);
            let mut t = Tableau::new(n);
            c.run(&mut t, &mut rng).unwrap();
            for (i, pr) in r.probs().into_iter().enumerate() {
                let snapped = if pr < 1e-9 { 0.0 } else { 2f64.powi(pr.log2().round() as i32) };
                assert_eq!(t.probability(i), snapped, "outcome {i} seed={seed} n={n} {:?}", c.ops);
            }
            for s in t.stabilizers() {
                let sign = if s.starts_with('-') { -1.0 } else { 1.0 };
                let ev = r.pauli_expectation(&s[1..]);
                assert!((ev - sign).abs() < 1e-9, "stabilizer {s} <P>={ev} seed={seed} {:?}", c.ops);
            }
        }
    }
}

#[test]
fn new_gates_pauli_path_match_reference() {
    for it in 0..15 * iters() {
        for &n in &[1usize, 2, 3, 5, 7] {
            let seed = base_seed() ^ 0x9E3 ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let mut c = Circuit::new(n);
            let mut nc = 0;
            for _ in 0..rng.random_range(1..50) {
                let g = if rng.random_bool(0.5) {
                    new_gate(&mut rng, n, false)
                } else {
                    random_gate(&mut rng, n, false, true)
                };
                if matches!(g, Gate::Ccx(..)) {
                    continue;
                }
                if !g.is_clifford() {
                    if nc >= 6 {
                        continue;
                    }
                    nc += 1;
                }
                c.gate(g);
            }
            let r = ref_run(&c);
            let p: String = (0..n).map(|_| ['I', 'X', 'Y', 'Z'][rng.random_range(0..4)]).collect();
            let (v, _) = pauli_path::expectation(&c, &PauliSum::from_str_single(&p), DEFAULT_MAX_TERMS).unwrap();
            let e = r.pauli_expectation(&p);
            assert!((v - e).abs() < 1e-10, "<{p}> pauli={v} ref={e} seed={seed} {:?}", c.ops);
        }
    }
}

/// `optimize()` must preserve the state. We report raw and up-to-global-
/// phase differences separately: dropping Rx/Ry/Rz(2π) = −I changes the
/// global phase only.
#[test]
fn optimize_preserves_state() {
    let mut phase_only = 0;
    for it in 0..40 * iters() {
        for &n in &[1usize, 2, 3, 5, 8] {
            let seed = base_seed() ^ 0x0F7 ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            // circuits rich in cancellation: random gates, often followed by
            // their inverse, a repeat, or a rotation completing a multiple of 2π
            let mut c = Circuit::new(n);
            for _ in 0..rng.random_range(1..60) {
                let g = if rng.random_bool(0.3) { new_gate(&mut rng, n, false) } else { random_gate(&mut rng, n, false, false) };
                c.gate(g);
                match rng.random_range(0..4) {
                    0 => {
                        c.gate(g.inverse());
                    }
                    1 => {
                        c.gate(g);
                    }
                    2 => {
                        let comp = |t: f64, rng: &mut StdRng| 2.0 * PI * rng.random_range(-2..3) as f64 - t;
                        let g2 = match g {
                            Gate::Rx(q, t) => Gate::Rx(q, comp(t, &mut rng)),
                            Gate::Ry(q, t) => Gate::Ry(q, comp(t, &mut rng)),
                            Gate::Rz(q, t) => Gate::Rz(q, comp(t, &mut rng)),
                            Gate::Phase(q, t) => Gate::Phase(q, comp(t, &mut rng)),
                            Gate::CPhase(a, b, t) => Gate::CPhase(b, a, comp(t, &mut rng)),
                            g => g,
                        };
                        c.gate(g2);
                    }
                    _ => {}
                }
            }
            let o = c.optimize();
            assert!(o.num_gates() <= c.num_gates());
            let r = ref_run(&c);
            let ro = ref_run(&o);
            let (raw, up) = global_phase_diff(&r.a, ro.a.iter().copied());
            assert!(up <= 1e-10, "optimize changed the state: Δ(up to phase)={up:e} seed={seed} n={n}\n{:?}\n->\n{:?}", c.ops, o.ops);
            if raw > 1e-10 {
                phase_only += 1;
            }
        }
    }
    eprintln!("optimize: {phase_only} circuits changed only by a global phase");
}

#[test]
fn optimize_rx_2pi_global_phase() {
    // Rx(π)·Rx(π) = Rx(2π) = −I: optimize() deletes it; state picks up −1.
    let mut c = Circuit::new(1);
    c.h(0).rx(0, PI).rx(0, PI);
    let o = c.optimize();
    let (raw, up) = global_phase_diff(&ref_run(&c).a, ref_run(&o).a.iter().copied());
    eprintln!("Rx(π)Rx(π): optimized ops {:?}; raw Δ={raw:e}, up-to-phase Δ={up:e}", o.ops);
    assert!(up < 1e-12);
}

#[test]
fn qasm_round_trip_preserves_state() {
    for it in 0..20 * iters() {
        for &n in &[1usize, 2, 4, 7] {
            let seed = base_seed() ^ 0x0A5 ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let depth = rng.random_range(1..60);
            let c = mixed_circuit(&mut rng, n, depth, false);
            let s = c.to_qasm().expect("unitary circuit must serialise");
            let back = Circuit::from_qasm(&s).unwrap_or_else(|e| panic!("reparse failed: {e}\n{s}"));
            assert_eq!(back.num_qubits, n);
            let (raw, _) = global_phase_diff(&ref_run(&c).a, ref_run(&back).a.iter().copied());
            assert!(raw <= 1e-12, "qasm round trip Δ={raw:e} seed={seed}\n{s}");
        }
    }
}

/// Non-gate ops must survive serialisation (or serialisation must fail).
#[test]
fn qasm_round_trip_keeps_conditionals_and_noise() {
    let mut c = Circuit::new(2);
    c.h(0).measure(0).c_if(0, Gate::X(1)).depolarize_1q(1, 0.1).measure(1);
    // Either an exact round trip, or an explicit error (no silent drops).
    match c.to_qasm() {
        Err(e) => eprintln!("to_qasm rejects non-unitary ops: {e}"),
        Ok(s) => {
            let back = Circuit::from_qasm(&s).unwrap();
            assert_eq!(back.ops, c.ops, "ops changed by to_qasm:\n{s}");
        }
    }
    // c_if alone must round-trip (QASM 2 has `if`)
    let mut c2 = Circuit::new(2);
    c2.h(0).measure(0).c_if(0, Gate::X(1)).measure(1);
    if let Ok(s) = c2.to_qasm() {
        let back = Circuit::from_qasm(&s).unwrap();
        assert_eq!(back.ops, c2.ops, "c_if round trip:\n{s}");
    }
}

/// More measurements than qubits: emitted creg must be large enough.
#[test]
fn qasm_creg_large_enough_for_repeated_measurement() {
    let mut c = Circuit::new(1);
    c.h(0).measure(0).measure(0).measure(0);
    let s = c.to_qasm().unwrap();
    let creg: usize = s.lines().find(|l| l.starts_with("creg")).and_then(|l| l.split(['[', ']']).nth(1)).unwrap().parse().unwrap();
    assert!(creg >= 3, "creg c[{creg}] but 3 measurements are written:\n{s}");
}

/// Parameter expressions: standard left-to-right precedence.
#[test]
fn qasm_parameter_expressions() {
    let mut bad = Vec::new();
    for (expr, want) in [
        ("-(pi/2)", -PI / 2.0),
        ("2^3", 8.0),
        ("-2^2", -4.0),
        ("pi/-2", -PI / 2.0),
        ("cos(0)", 1.0),
        ("1.5e-3*2", 3e-3),
        ("pi/2", PI / 2.0),
        ("-pi/4", -PI / 4.0),
        ("2*pi/3", 2.0 * PI / 3.0),
        ("pi/2*3", 3.0 * PI / 2.0),
        ("1/2/4", 0.125),
        ("pi-1", PI - 1.0),
        ("pi/2+pi/4", 0.75 * PI),
        ("1e-3", 1e-3),
    ] {
        let src = format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\nrz({expr}) q[0];\n");
        match Circuit::from_qasm(&src) {
            Ok(c) => match c.ops[0] {
                qsim_lab::Op::Gate(Gate::Rz(_, t)) if (t - want).abs() < 1e-12 => {}
                ref o => bad.push(format!("rz({expr}) parsed as {o:?}, want {want}")),
            },
            Err(e) => bad.push(format!("rz({expr}) rejected: {e}")),
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
fn reset_all_restores_zero_state() {
    let mut rng = StdRng::seed_from_u64(base_seed() ^ 0xEE);
    for &n in &[1usize, 3, 15] {
        let c = mixed_circuit(&mut rng, n, 40, false);
        let mut sv = StateVectorF64::new(n);
        sv.apply_circuit(&c).unwrap();
        Simulator::reset_all(&mut sv).unwrap();
        assert!((sv.amplitude(0) - cx(1.0, 0.0)).norm() < 1e-15 && (sv.norm_sqr() - 1.0).abs() < 1e-15);
        let cc = mixed_circuit(&mut rng, n, 40, true);
        let mut t = Tableau::new(n);
        cc.run(&mut t, &mut rng).unwrap();
        Simulator::reset_all(&mut t).unwrap();
        assert_eq!(t.probability(0), 1.0);
        let mut m = Mps::new(n, 64);
        if n <= 10 {
            c.run(&mut m, &mut rng).unwrap();
            Simulator::reset_all(&mut m).unwrap();
            assert!((m.amplitude(0) - cx(1.0, 0.0)).norm() < 1e-12);
            // and keeps working afterwards
            c.run(&mut m, &mut rng).unwrap();
            let d = max_amp_diff(&ref_run(&c).a, (0..1u128 << n).map(|i| m.amplitude(i)));
            assert!(d < 1e-9, "mps after reset_all Δ={d:e}");
        }
    }
}

/// The blocked executor (on main since the PR merged main) must handle the
/// new gates too.
#[test]
fn new_gates_blocked_executor_match_reference() {
    use qsim_lab::engines::blocked::BlockConfig;
    for it in 0..10 * iters() {
        for &n in &SIZES {
            let seed = base_seed() ^ 0x9E5 ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let depth = rng.random_range(1..80);
            let c = mixed_circuit(&mut rng, n, depth, false);
            let r = ref_run(&c);
            let mut cfgs = vec![BlockConfig::default()];
            for _ in 0..3 {
                let mut k = BlockConfig::default();
                k.block_bytes = [8usize, 64, 1024, 1 << 18][rng.random_range(0..4)];
                k.slots = rng.random_range(0..8);
                k.fuse_1q = rng.random_bool(0.5);
                k.small_n = rng.random_range(0..4);
                cfgs.push(k);
            }
            for cfg in cfgs {
                let mut sv = StateVectorF64::new(n);
                sv.apply_circuit_blocked(&c, &cfg).unwrap();
                let d = max_amp_diff(&r.a, (0..1 << n).map(|i| sv.amplitude(i)));
                assert!(d <= 1e-12, "blocked new gates Δ={d:e} seed={seed} n={n} cfg={cfg:?} {:?}", c.ops);
            }
        }
    }
}
