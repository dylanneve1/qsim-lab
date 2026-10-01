//! Exactness of the sign-tracking `Tableau` against the frozen CHP
//! reference (`stabilizer::reference::RefTableau`).
//!
//! Both implementations consume exactly one `random_bool(0.5)` per random
//! measurement outcome and nothing else, and whether an outcome is random
//! is a property of the state, not of the representation. So with the same
//! seed, every measurement (and reset) outcome must be identical, and the
//! final states must have the same canonical stabilizer group, signs
//! included.

use proptest::prelude::*;
use qsim_lab::stabilizer::reference::RefTableau;
use qsim_lab::{Gate, Tableau};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::{FRAC_PI_2, PI};

#[derive(Clone, Copy, Debug)]
enum Step {
    G(Gate),
    Measure(usize),
    Reset(usize),
}

fn random_gate<R: Rng>(n: usize, rng: &mut R) -> Gate {
    let a = rng.random_range(0..n);
    let mut b = rng.random_range(0..n);
    if n > 1 {
        while b == a {
            b = rng.random_range(0..n);
        }
    }
    let two = n > 1;
    match rng.random_range(0..14) {
        0 => Gate::H(a),
        1 => Gate::S(a),
        2 => Gate::Sdg(a),
        3 => Gate::X(a),
        4 => Gate::Y(a),
        5 => Gate::Z(a),
        6 => Gate::Phase(a, FRAC_PI_2 * rng.random_range(-3..4) as f64),
        7 => Gate::Rz(a, FRAC_PI_2 * rng.random_range(-3..4) as f64),
        8 | 9 if two => Gate::Cnot(a, b),
        10 if two => Gate::Cz(a, b),
        11 if two => Gate::Swap(a, b),
        12 if two => Gate::CPhase(a, b, PI * rng.random_range(-2..3) as f64),
        _ => Gate::H(a),
    }
}

/// Random Clifford program: `len` steps, a fraction `pm` of them single
/// qubit measurements and `pr` resets.
fn program(n: usize, len: usize, pm: f64, pr: f64, seed: u64) -> Vec<Step> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len)
        .map(|_| {
            let u: f64 = rng.random();
            if u < pm {
                Step::Measure(rng.random_range(0..n))
            } else if u < pm + pr {
                Step::Reset(rng.random_range(0..n))
            } else {
                Step::G(random_gate(n, &mut rng))
            }
        })
        .collect()
}

fn run_new(n: usize, prog: &[Step], seed: u64) -> (Vec<bool>, Tableau) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut t = Tableau::new(n);
    let mut out = vec![];
    for s in prog {
        match *s {
            Step::G(g) => t.apply_gate(&g).unwrap(),
            Step::Measure(q) => out.push(t.measure_qubit(q, &mut rng)),
            Step::Reset(q) => out.push(t.reset_qubit(q, &mut rng)),
        }
    }
    (out, t)
}

fn run_ref(n: usize, prog: &[Step], seed: u64) -> (Vec<bool>, RefTableau) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut t = RefTableau::new(n);
    let mut out = vec![];
    for s in prog {
        match *s {
            Step::G(g) => t.apply_gate(&g).unwrap(),
            Step::Measure(q) => out.push(t.measure_qubit(q, &mut rng)),
            Step::Reset(q) => out.push(t.reset_qubit(q, &mut rng)),
        }
    }
    (out, t)
}

/// A Pauli string with a sign: `(-1)^neg * prod_q X^x_q Z^z_q`-style, with
/// (x, z) = (1, 1) meaning Y (as in the tableau).
#[derive(Clone, PartialEq, Eq, Debug)]
struct Pauli {
    neg: bool,
    x: Vec<bool>,
    z: Vec<bool>,
}

fn parse(s: &str) -> Pauli {
    let mut cs = s.chars();
    let neg = cs.next() == Some('-');
    let (mut x, mut z) = (vec![], vec![]);
    for c in cs {
        x.push(c == 'X' || c == 'Y');
        z.push(c == 'Z' || c == 'Y');
    }
    Pauli { neg, x, z }
}

/// `self <- other * self`, exact phase (both Hermitian and commuting, so
/// the product is Hermitian with sign ±1).
fn mul_into(h: &mut Pauli, i: &Pauli) {
    let mut e: i32 = 2 * (h.neg as i32) + 2 * (i.neg as i32);
    for q in 0..h.x.len() {
        let (x1, z1, x2, z2) = (i.x[q], i.z[q], h.x[q], h.z[q]);
        e += match (x1, z1) {
            (false, false) => 0,
            (true, true) => z2 as i32 - x2 as i32,
            (true, false) => (z2 as i32) * (2 * x2 as i32 - 1),
            (false, true) => (x2 as i32) * (1 - 2 * z2 as i32),
        };
        h.x[q] ^= x1;
        h.z[q] ^= z1;
    }
    let e = e.rem_euclid(4);
    assert!(
        e == 0 || e == 2,
        "product of commuting Paulis must be Hermitian"
    );
    h.neg = e == 2;
}

/// Fully reduced row-echelon form of the stabilizer group (columns ordered
/// x_0..x_{n-1}, z_0..z_{n-1}), which is unique for a given group.
fn canonical(gens: &[String]) -> Vec<Pauli> {
    let mut rows: Vec<Pauli> = gens.iter().map(|s| parse(s)).collect();
    let n = rows.first().map_or(0, |r| r.x.len());
    let mut r = 0;
    for col in 0..2 * n {
        let bit = |p: &Pauli| if col < n { p.x[col] } else { p.z[col - n] };
        let Some(piv) = (r..rows.len()).find(|&i| bit(&rows[i])) else {
            continue;
        };
        rows.swap(r, piv);
        let pr = rows[r].clone();
        for (i, row) in rows.iter_mut().enumerate() {
            if i != r && bit(row) {
                mul_into(row, &pr);
            }
        }
        r += 1;
    }
    rows
}

fn assert_same(n: usize, prog: &[Step], seed: u64) {
    let (a, mut ta) = run_new(n, prog, seed);
    let (b, mut tb) = run_ref(n, prog, seed);
    assert_eq!(a, b, "outcomes differ: n = {n}, seed = {seed}");
    assert_eq!(
        canonical(&ta.stabilizers()),
        canonical(&tb.stabilizers()),
        "final states differ: n = {n}, seed = {seed}"
    );
}

#[test]
fn identical_outcomes_and_states_on_random_programs() {
    let mut seed = 1000;
    for n in [
        1, 2, 3, 5, 8, 17, 31, 63, 64, 65, 100, 127, 128, 129, 200, 300,
    ] {
        // measurement-heavy, gate-heavy, reset-heavy
        for (pm, pr) in [(0.3, 0.1), (0.05, 0.02), (0.1, 0.3)] {
            let len = (8 * n).clamp(40, 1500);
            let prog = program(n, len, pm, pr, seed);
            assert_same(n, &prog, seed + 1);
            seed += 2;
        }
    }
}

/// Syndrome-extraction shaped workload: Bell/GHZ-style entanglement, then
/// rounds of ancilla parity checks with reset. Most measurements here are
/// deterministic, which is the path the sign tracking speeds up.
#[test]
fn identical_outcomes_on_repeated_parity_checks() {
    for (n_data, seed) in [(5, 1u64), (40, 2), (150, 3)] {
        let n = 2 * n_data;
        let mut prog = vec![];
        for q in 0..n_data {
            prog.push(Step::G(Gate::H(q)));
        }
        for _round in 0..4 {
            for k in 0..n_data {
                let a = n_data + k;
                prog.push(Step::G(Gate::H(a)));
                prog.push(Step::G(Gate::Cnot(a, k)));
                prog.push(Step::G(Gate::Cnot(a, (k + 1) % n_data)));
                prog.push(Step::G(Gate::H(a)));
            }
            for k in 0..n_data {
                prog.push(Step::Reset(n_data + k));
            }
        }
        assert_same(n, &prog, seed);
        // noiseless: rounds after the first repeat the first
        let (out, _) = run_new(n, &prog, seed);
        for r in 1..4 {
            assert_eq!(out[r * n_data..(r + 1) * n_data], out[..n_data]);
        }
    }
}

/// `measure_all` draws from the right affine subspace: forcing the drawn
/// bits on the reference never contradicts a deterministic outcome, and
/// the collapsed state is the drawn basis state.
#[test]
fn measure_all_lands_in_the_support() {
    for (n, seed) in [(3usize, 7u64), (40, 8), (130, 9)] {
        let prog = program(n, 10 * n, 0.05, 0.02, seed);
        let (_, mut t) = run_new(n, &prog, seed);
        let (_, r) = run_ref(n, &prog, seed);
        let mut rng = StdRng::seed_from_u64(seed);
        let bits = t.measure_all(&mut rng);
        let mut r = r.clone();
        for (q, &b) in bits.iter().enumerate() {
            let (got, random) = r.measure_with(q, Some(b), &mut rng);
            assert!(random || got == b, "n = {n}: drew an impossible outcome");
        }
        for (q, &b) in bits.iter().enumerate() {
            assert_eq!(t.peek(q), Some(b));
        }
    }
}

#[test]
fn reset_of_entangled_qubit_leaves_partner_random() {
    let mut rng = StdRng::seed_from_u64(1);
    let mut t = Tableau::new(2);
    t.h(0);
    t.cnot(0, 1);
    t.reset_qubit(0, &mut rng);
    assert_eq!(t.peek(0), Some(false));
    // after the reset channel the partner is maximally mixed in the
    // ensemble; in a trajectory it is collapsed to the drawn outcome
    let mut ones = 0;
    for s in 0..400 {
        let mut rng = StdRng::seed_from_u64(s);
        let mut t = Tableau::new(2);
        t.h(0);
        t.cnot(0, 1);
        t.reset_qubit(0, &mut rng);
        ones += t.measure_qubit(1, &mut rng) as usize;
    }
    assert!((150..250).contains(&ones), "partner ones = {ones}/400");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn prop_identical_outcomes(
        n in 1usize..90,
        len in 1usize..400,
        pm in 0.0f64..0.5,
        pr in 0.0f64..0.3,
        pseed in any::<u64>(),
        rseed in any::<u64>(),
    ) {
        let prog = program(n, len, pm, pr, pseed);
        let (a, mut ta) = run_new(n, &prog, rseed);
        let (b, mut tb) = run_ref(n, &prog, rseed);
        prop_assert_eq!(a, b);
        prop_assert_eq!(canonical(&ta.stabilizers()), canonical(&tb.stabilizers()));
    }
}
