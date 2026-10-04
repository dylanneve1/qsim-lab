//! Audit (PR 3, on main): reset of an ENTANGLED qubit, without a prior
//! measurement, must leave the partners in the reference mixture. Reference:
//! RefSv branches collapse(q,0) and collapse(q,1)+X(q), weighted by P(q),
//! then the rest of the circuit on each branch; final distribution mixed.
//! Checks Tableau::reset_qubit and Circuit::run (Op::Reset) on Tableau,
//! including several resets and gates after the reset. Chi-square, 6σ.
mod audit_common;
use audit_common::*;
use qsim_lab::{Circuit, Gate, Op, Tableau};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Mixture distribution of `c` (gates + Reset + final measure-all implied).
fn ref_dist(c: &Circuit) -> Vec<f64> {
    let n = c.num_qubits;
    let mut branches: Vec<(f64, RefSv)> = vec![(1.0, RefSv::new(n))];
    for op in &c.ops {
        match op {
            Op::Gate(g) => branches.iter_mut().for_each(|(_, s)| s.apply(g)),
            Op::Reset(q) => {
                let mut nb = Vec::new();
                for (w, s) in branches {
                    let p1 = s.prob_one(*q);
                    for (b, p) in [(false, 1.0 - p1), (true, p1)] {
                        if p > 1e-12 {
                            let mut s2 = s.clone();
                            s2.collapse(*q, b);
                            if b { s2.apply(&Gate::X(*q)); }
                            nb.push((w * p, s2));
                        }
                    }
                }
                branches = nb;
            }
            _ => unreachable!(),
        }
    }
    let mut d = vec![0.0; 1 << n];
    for (w, s) in &branches {
        for (i, p) in s.probs().iter().enumerate() { d[i] += w * p; }
    }
    d
}

fn chi_ok(d: &[f64], counts: &[usize], shots: usize) -> (bool, f64) {
    let mut chi = 0.0;
    let mut dof = 0usize;
    for (p, &k) in d.iter().zip(counts) {
        if *p < 1e-12 {
            if k > 0 { return (false, f64::INFINITY); }
            continue;
        }
        let e = p * shots as f64;
        chi += (k as f64 - e).powi(2) / e;
        dof += 1;
    }
    let dof = dof.saturating_sub(1).max(1) as f64;
    (chi <= dof + 6.0 * (2.0 * dof).sqrt() + 10.0, chi)
}

#[test]
fn tableau_reset_of_entangled_qubit_matches_mixture() {
    let shots = 4000;
    let mut nontrivial = 0;
    for it in 0..25 * iters() {
        for n in [2usize, 3, 4, 5] {
            let seed = base_seed() ^ 0x5E7 ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let mut c = Circuit::new(n);
            // entangle: H + CNOT chain gives a GHZ-like core, then random Cliffords
            c.gate(Gate::H(0));
            for q in 1..n { c.gate(Gate::Cnot(0, q)); }
            for _ in 0..rng.random_range(0..3 * n) { c.gate(random_gate(&mut rng, n, true, false)); }
            for _ in 0..rng.random_range(1..4) {
                c.reset(rng.random_range(0..n));
                for _ in 0..rng.random_range(0..2 * n) { c.gate(random_gate(&mut rng, n, true, false)); }
            }
            let d = ref_dist(&c);
            // is the outcome distribution different from "reset forced 0"? count anyway
            if d.iter().filter(|p| **p > 1e-12).count() > 1 { nontrivial += 1; }
            for path in 0..2 {
                let mut counts = vec![0usize; 1 << n];
                let mut srng = StdRng::seed_from_u64(seed ^ 0xABCD ^ path);
                for _ in 0..shots {
                    let mut t = Tableau::new(n);
                    if path == 0 {
                        for op in &c.ops {
                            match op {
                                Op::Gate(g) => t.apply_gate(g).unwrap(),
                                Op::Reset(q) => { t.reset_qubit(*q, &mut srng); }
                                _ => unreachable!(),
                            }
                        }
                    } else {
                        c.run(&mut t, &mut srng).unwrap();
                    }
                    let bits = t.measure_all(&mut srng);
                    let idx = bits.iter().enumerate().fold(0usize, |a, (q, &b)| a | ((b as usize) << q));
                    counts[idx] += 1;
                }
                let (ok, chi) = chi_ok(&d, &counts, shots);
                assert!(ok, "path={path} chi={chi} seed={seed} n={n}\nref={d:?}\ncounts={counts:?}\n{:?}", c.ops);
            }
        }
    }
    eprintln!("entangled reset: {nontrivial} circuits with non-deterministic outcome");
}

/// reset_all after arbitrary history (gates, measurements, resets, measure_all)
/// must be indistinguishable from a fresh tableau: identical peeks and
/// identical outcome sequences under the same RNG on a follow-up circuit,
/// and the follow-up distribution matches the reference.
#[test]
fn tableau_reset_all_after_history_is_fresh() {
    for it in 0..40 * iters() {
        for n in [1usize, 2, 3, 5, 9, 64, 65, 130] {
            let seed = base_seed() ^ 0x2E5A ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let mut t = Tableau::new(n);
            for _ in 0..rng.random_range(0..6) {
                for _ in 0..rng.random_range(0..3 * n.min(10)) { t.apply_gate(&random_gate(&mut rng, n, true, false)).unwrap(); }
                let q = rng.random_range(0..n);
                match rng.random_range(0..3) {
                    0 => { t.measure_qubit(q, &mut rng); }
                    1 => { t.reset_qubit(q, &mut rng); }
                    _ => { t.measure_all(&mut rng); }
                }
            }
            t.reset_all();
            let mut f = Tableau::new(n);
            let follow: Vec<Gate> = (0..rng.random_range(0..4 * n.min(10))).map(|_| random_gate(&mut rng, n, true, false)).collect();
            for g in &follow { t.apply_gate(g).unwrap(); f.apply_gate(g).unwrap(); }
            for q in 0..n { assert_eq!(t.peek(q), f.peek(q), "peek q={q} seed={seed} n={n}"); }
            let (mut r1, mut r2) = (StdRng::seed_from_u64(seed ^ 7), StdRng::seed_from_u64(seed ^ 7));
            for _ in 0..3 {
                let q = r1.random_range(0..n); let _ = r2.random_range(0..n);
                assert_eq!(t.measure_qubit(q, &mut r1), f.measure_qubit(q, &mut r2), "measure seed={seed}");
            }
            assert_eq!(t.measure_all(&mut r1), f.measure_all(&mut r2), "measure_all seed={seed} n={n}");
            if n <= 9 {
                // also the reference: fresh RefSv on the follow-up gates, peeks deterministic agree
                let mut r = RefSv::new(n);
                for g in &follow { r.apply(g); }
                let mut t2 = Tableau::new(n);
                t2.reset_all();
                for g in &follow { t2.apply_gate(g).unwrap(); }
                for q in 0..n {
                    let p1 = r.prob_one(q);
                    match t2.peek(q) { Some(b) => assert!((p1 - b as u8 as f64).abs() < 1e-9), None => assert!((p1 - 0.5).abs() < 1e-9) }
                }
            }
        }
    }
}
