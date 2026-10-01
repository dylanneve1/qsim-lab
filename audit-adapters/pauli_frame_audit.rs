//! Audit adapter for exp/pauli (PR 2): rotation-frame engine vs the
//! independent reference on adversarial circuits with NON-ZERO expectation
//! values. Circuit = L · W† · M · W, W a random Clifford (dense rotation
//! axes), M a sparse non-Clifford core (repeated / cancelling / same-axis
//! rotations, tiny and π angles, Ccx, CPhase), L a final 1q layer that makes
//! the observable (X/Y/Z on a random subset) close to ±1. t up to ~40 ≫ n so
//! x-span pruning and z projection fire. Copy to tests/ with audit_common.
mod audit_common;
use audit_common::*;
use qsim_lab::pauli_frame::FrameOptions;
use qsim_lab::pauli_path::{expectation_legacy, expectation_with, PauliSum};
use qsim_lab::{Circuit, Gate};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn core(rng: &mut StdRng, n: usize, k: usize, len: usize) -> Vec<Gate> {
    // non-Clifford activity confined to k qubits (plus Clifford glue)
    let qs: Vec<usize> = (0..k).map(|_| rng.random_range(0..n)).collect();
    let mut g = Vec::new();
    for _ in 0..len {
        let q = qs[rng.random_range(0..k)];
        let ang = [std::f64::consts::FRAC_PI_4, -std::f64::consts::FRAC_PI_4, std::f64::consts::PI, 1e-9, 0.3, 1.234, -2.5][rng.random_range(0..7)];
        let r = match rng.random_range(0..12) {
            0 => Gate::T(q),
            1 => Gate::Tdg(q),
            2 => Gate::Rz(q, ang),
            3 => Gate::Rx(q, ang),
            4 => Gate::Ry(q, ang),
            5 => Gate::Phase(q, ang),
            6 => { // rotation then its exact inverse (merge -> identity)
                g.push(Gate::Rz(q, ang));
                Gate::Rz(q, -ang)
            }
            7 => { // same-axis pair separated by a commuting gate
                g.push(Gate::Rz(q, ang));
                g.push(Gate::Z(q));
                Gate::T(q)
            }
            8 if n > 1 => { let o = qs[rng.random_range(0..k)]; if o == q { Gate::H(q) } else { Gate::CPhase(q, o, ang) } }
            9 if n > 2 => { let a = qs[rng.random_range(0..k)]; let b = qs[rng.random_range(0..k)]; if a == q || b == q || a == b { Gate::S(q) } else { Gate::Ccx(a, b, q) } }
            10 if n > 1 => { let o = qs[rng.random_range(0..k)]; if o == q { Gate::H(q) } else { Gate::Cnot(q, o) } }
            _ => Gate::H(q),
        };
        g.push(r);
    }
    g
}

fn build(rng: &mut StdRng, n: usize) -> (Circuit, String) {
    let mut w = Vec::new();
    for _ in 0..rng.random_range(0..4 * n + 2) {
        w.push(random_gate(rng, n, true, false));
    }
    let k = rng.random_range(1..=n.min(3));
    let len = rng.random_range(1..40);
    let m = core(rng, n, k, len);
    let mut c = Circuit::new(n);
    for g in &w { c.gate(*g); }
    for g in &m { c.gate(*g); }
    for g in w.iter().rev() { c.gate(g.inverse()); }
    let mut obs = String::new();
    for q in 0..n {
        let ch = if rng.random_bool(0.4) { ['X', 'Y', 'Z'][rng.random_range(0..3)] } else { 'I' };
        match ch { 'X' => { c.gate(Gate::H(q)); } 'Y' => { c.gate(Gate::H(q)); c.gate(Gate::S(q)); } _ => {} }
        obs.push(ch);
    }
    if !obs.contains(['X', 'Y', 'Z']) { obs.replace_range(0..1, "Z"); }
    (c, obs)
}

fn all_opts() -> Vec<FrameOptions> {
    let mut v = Vec::new();
    for b in 0..16u32 {
        for drop in [1e-14, 0.0] {
            v.push(FrameOptions { prune: b & 1 != 0, merge_rotations: b & 2 != 0, parallel: b & 4 != 0, fuse: b & 8 != 0, drop_below: drop, ..FrameOptions::default() });
        }
    }
    v
}

#[test]
fn frame_matches_reference_nonzero() {
    let (mut checked, mut nonzero, mut pruned_runs, mut worst) = (0usize, 0usize, 0usize, 0.0f64);
    let opts = all_opts();
    for it in 0..40 * iters() {
        for n in [1usize, 2, 3, 4, 5, 6, 8, 10, 12, 14] {
            let seed = base_seed() ^ 0xFA11 ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let (c, obs) = build(&mut rng, n);
            let r = RefSv::run(&c).pauli_expectation(&obs);
            if r.abs() > 1e-3 { nonzero += 1; }
            let p = PauliSum::from_str_single(&obs);
            let sel: Vec<&FrameOptions> = if n <= 8 { opts.iter().collect() } else { vec![&opts[rng.random_range(0..opts.len())], &opts[opts.len() - 1]] };
            for o in sel {
                let (v, st) = expectation_with(&c, &p, o).unwrap_or_else(|e| panic!("err {e} seed={seed}"));
                if st.pruned_terms > 0 { pruned_runs += 1; }
                let d = (v - r).abs();
                worst = worst.max(d);
                assert!(d <= 1e-9, "frame Δ={d:e} ref={r} got={v} seed={seed} n={n} obs={obs} opts={o:?}\n{:?}", c.ops);
            }
            let (l, _) = expectation_legacy(&c, &p, 1 << 22).unwrap();
            assert!((l - r).abs() <= 1e-9, "legacy Δ seed={seed}");
            checked += 1;
        }
    }
    eprintln!("pauli frame: {checked} circuits, {nonzero} with |<P>|>1e-3, {pruned_runs} option-runs with pruning, worst Δ={worst:e}");
    assert!(nonzero * 2 > checked);
    assert!(pruned_runs > 0);
}

/// Wide registers (64, 65, 128 qubits; crosses the 64-bit word boundary):
/// frame vs legacy on the same nonzero family (no SV possible).
#[test]
fn frame_matches_legacy_wide() {
    let mut nonzero = 0;
    for it in 0..6 * iters() {
        for n in [63usize, 64, 65, 128] {
            let seed = base_seed() ^ 0x64 ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let (c, obs) = build(&mut rng, n);
            let p = PauliSum::from_str_single(&obs);
            let (l, _) = match expectation_legacy(&c, &p, 1 << 21) { Ok(x) => x, Err(_) => continue };
            if l.abs() > 1e-3 { nonzero += 1; }
            for o in [FrameOptions::default(), FrameOptions { prune: false, merge_rotations: false, drop_below: 0.0, ..FrameOptions::default() }] {
                let (v, _) = expectation_with(&c, &p, &o).unwrap();
                assert!((v - l).abs() <= 1e-10, "wide Δ={:e} seed={seed} n={n}", (v - l).abs());
            }
        }
    }
    eprintln!("wide: {nonzero} nonzero");
    assert!(nonzero > 0);
}
