//! HSF on main (8139a6a) with PR 1 gates: helpers from pr1_gates_qasm.rs + the hsf_amplitudes test using mixed_circuit/ref_run.
#![allow(dead_code, unused_imports)]
mod audit_common;

use audit_common::*;
use num_complex::Complex64 as C;
use qsim_lab::pauli_path::{self, PauliSum, DEFAULT_MAX_TERMS};
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

use qsim_lab::hsf::{HsfOptions, HybridSchrodingerFeynman, LeafMode, SchmidtMode};
fn random_opts(rng: &mut StdRng) -> HsfOptions {
    let mut o = HsfOptions::default();
    if rng.random_bool(0.3) {
        return o;
    }
    o.eliminate_swaps = rng.random_bool(0.5);
    o.asap = rng.random_bool(0.5);
    o.schmidt = [SchmidtMode::Analytic, SchmidtMode::Svd, SchmidtMode::MatrixUnits][rng.random_range(0..3)];
    o.leaf = [LeafMode::Forward, LeafMode::Bra, LeafMode::Auto][rng.random_range(0..3)];
    o.prune_zero = rng.random_bool(0.5);
    o.threads = rng.random_range(0..3);
    o.gemm_batch = [1usize, 2, 7, 64][rng.random_range(0..4)];
    o
}

fn random_partition(rng: &mut StdRng, n: usize) -> Vec<bool> {
    match rng.random_range(0..6) {
        0 => vec![true; n],
        1 => vec![false; n],
        2 => (0..n).map(|q| q == 0).collect(),
        3 => (0..n).map(|q| q % 2 == 0).collect(),
        4 => (0..n).map(|q| q < n / 2).collect(),
        _ => (0..n).map(|_| rng.random_bool(0.5)).collect(),
    }
}

#[test]
fn hsf_matches_reference() {
    let (mut ok, mut rejected, mut worst) = (0usize, 0usize, 0.0f64);
    let mut reasons = std::collections::BTreeMap::new();
    for it in 0..15 * iters() {
        for &n in &SIZES {
            let seed = base_seed() ^ 0x45F ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let depth = rng.random_range(1..50);
            // keep the number of crossing gates moderate
            let c: Circuit = mixed_circuit(&mut rng, n, depth, false);
            let part = random_partition(&mut rng, n);
            let opts = random_opts(&mut rng);
            let h = match HybridSchrodingerFeynman::new(&c, &part, opts.clone()) {
                Ok(h) => h,
                Err(e) => {
                    rejected += 1;
                    *reasons.entry(format!("{e}").chars().take(60).collect::<String>()).or_insert(0) += 1;
                    continue;
                }
            };
            if h.num_paths() > 1 << 14 {
                continue;
            }
            let r = ref_run(&c);
            let mut xs: Vec<usize> = vec![0, (1 << n) - 1];
            for _ in 0..6 {
                xs.push(rng.random_range(0..1usize << n));
            }
            let amps = h.amplitudes(&xs).unwrap();
            for (&x, a) in xs.iter().zip(&amps) {
                let d = (r.a[x] - a).norm();
                worst = worst.max(d);
                assert!(d <= 1e-12, "hsf amplitude x={x} Δ={d:e} seed={seed} n={n} part={part:?} opts={opts:?}\n{:?}", c.ops);
            }
            let a1 = h.amplitude(xs[2]).unwrap();
            assert!((r.a[xs[2]] - a1).norm() <= 1e-12, "single amplitude seed={seed}");
            if n <= 11 {
                let sv = h.state_vector().unwrap();
                let d = max_amp_diff(&r.a, sv.iter().copied());
                worst = worst.max(d);
                assert!(d <= 1e-12, "hsf state_vector Δ={d:e} seed={seed} n={n} part={part:?} opts={opts:?}\n{:?}", c.ops);
            }
            ok += 1;
        }
    }
    eprintln!("hsf: {ok} checked, {rejected} rejected {reasons:?}, worst Δ = {worst:e}");
    assert!(ok > 50);
}

