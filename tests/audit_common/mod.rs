#![allow(clippy::needless_range_loop)]
//! Shared pieces of the audit differential fuzz harness: an independent
//! naive reference state vector and edge-biased circuit generators.
//! Used by `tests/differential_fuzz.rs` and by the per-branch adapters in
//! `tools/audit-adapters/`.
#![allow(dead_code)]

use num_complex::Complex64 as C;
use qsim_lab::{Circuit, Gate};
use rand::rngs::StdRng;
use rand::Rng;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Independent reference
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct RefSv {
    pub n: usize,
    pub a: Vec<C>,
}

pub fn cx(re: f64, im: f64) -> C {
    C::new(re, im)
}

impl RefSv {
    pub fn new(n: usize) -> Self {
        let mut a = vec![C::new(0.0, 0.0); 1 << n];
        a[0] = cx(1.0, 0.0);
        RefSv { n, a }
    }

    /// 2x2 matrix on qubit q, written from the textbook definitions.
    pub fn m1(g: &Gate) -> [[C; 2]; 2] {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let z = cx(0.0, 0.0);
        let o = cx(1.0, 0.0);
        let e = |t: f64| cx(t.cos(), t.sin());
        match *g {
            Gate::H(_) => [[cx(s, 0.0), cx(s, 0.0)], [cx(s, 0.0), cx(-s, 0.0)]],
            Gate::X(_) => [[z, o], [o, z]],
            Gate::Y(_) => [[z, cx(0.0, -1.0)], [cx(0.0, 1.0), z]],
            Gate::Z(_) => [[o, z], [z, cx(-1.0, 0.0)]],
            Gate::S(_) => [[o, z], [z, cx(0.0, 1.0)]],
            Gate::Sdg(_) => [[o, z], [z, cx(0.0, -1.0)]],
            Gate::T(_) => [[o, z], [z, e(PI / 4.0)]],
            Gate::Tdg(_) => [[o, z], [z, e(-PI / 4.0)]],
            Gate::Rx(_, t) => {
                let (c, s) = ((t / 2.0).cos(), (t / 2.0).sin());
                [[cx(c, 0.0), cx(0.0, -s)], [cx(0.0, -s), cx(c, 0.0)]]
            }
            Gate::Ry(_, t) => {
                let (c, s) = ((t / 2.0).cos(), (t / 2.0).sin());
                [[cx(c, 0.0), cx(-s, 0.0)], [cx(s, 0.0), cx(c, 0.0)]]
            }
            Gate::Rz(_, t) => [[e(-t / 2.0), z], [z, e(t / 2.0)]],
            Gate::Phase(_, t) => [[o, z], [z, e(t)]],
            _ => unreachable!(),
        }
    }

    pub fn bit(i: usize, q: usize) -> usize {
        (i >> q) & 1
    }

    pub fn apply(&mut self, g: &Gate) {
        let n = self.n;
        let dim = 1usize << n;
        let old = self.a.clone();
        let mut new = vec![cx(0.0, 0.0); dim];
        match *g {
            Gate::Cnot(c, t) => {
                for i in 0..dim {
                    let j = if Self::bit(i, c) == 1 {
                        i ^ (1 << t)
                    } else {
                        i
                    };
                    new[j] = old[i];
                }
            }
            Gate::Cz(a, b) => {
                for i in 0..dim {
                    let s = Self::bit(i, a) & Self::bit(i, b);
                    new[i] = if s == 1 { -old[i] } else { old[i] };
                }
            }
            Gate::CPhase(a, b, t) => {
                for i in 0..dim {
                    let s = Self::bit(i, a) & Self::bit(i, b);
                    new[i] = if s == 1 {
                        old[i] * cx(t.cos(), t.sin())
                    } else {
                        old[i]
                    };
                }
            }
            Gate::Swap(a, b) => {
                for i in 0..dim {
                    let (ba, bb) = (Self::bit(i, a), Self::bit(i, b));
                    let j = (i & !(1 << a) & !(1 << b)) | (ba << b) | (bb << a);
                    new[j] = old[i];
                }
            }
            Gate::Ccx(a, b, t) => {
                for i in 0..dim {
                    let j = if Self::bit(i, a) & Self::bit(i, b) == 1 {
                        i ^ (1 << t)
                    } else {
                        i
                    };
                    new[j] = old[i];
                }
            }
            ref g1 => {
                let q = g1.qubits()[0];
                let m = Self::m1(g1);
                for (i, out) in new.iter_mut().enumerate() {
                    let r = Self::bit(i, q);
                    let i0 = i & !(1 << q);
                    let i1 = i | (1 << q);
                    *out = m[r][0] * old[i0] + m[r][1] * old[i1];
                }
            }
        }
        self.a = new;
    }

    pub fn run(c: &Circuit) -> Self {
        let mut s = RefSv::new(c.num_qubits);
        for g in c.gates() {
            s.apply(g);
        }
        s
    }

    pub fn probs(&self) -> Vec<f64> {
        self.a.iter().map(|x| x.norm_sqr()).collect()
    }

    pub fn prob_one(&self, q: usize) -> f64 {
        self.a
            .iter()
            .enumerate()
            .filter(|(i, _)| Self::bit(*i, q) == 1)
            .map(|(_, x)| x.norm_sqr())
            .sum()
    }

    pub fn collapse(&mut self, q: usize, outcome: bool) {
        let p = if outcome {
            self.prob_one(q)
        } else {
            1.0 - self.prob_one(q)
        };
        let k = 1.0 / p.sqrt();
        for (i, x) in self.a.iter_mut().enumerate() {
            if (Self::bit(i, q) == 1) != outcome {
                *x = cx(0.0, 0.0);
            } else {
                *x *= k;
            }
        }
    }

    /// <ψ| P |ψ> for a Pauli string, qubit 0 first.
    pub fn pauli_expectation(&self, p: &str) -> f64 {
        let mut phi = self.clone();
        for (q, ch) in p.chars().enumerate() {
            match ch {
                'X' => phi.apply(&Gate::X(q)),
                'Y' => phi.apply(&Gate::Y(q)),
                'Z' => phi.apply(&Gate::Z(q)),
                _ => {}
            }
        }
        let v: C = self.a.iter().zip(&phi.a).map(|(x, y)| x.conj() * y).sum();
        assert!(v.im.abs() < 1e-9);
        v.re
    }
}

// ---------------------------------------------------------------------------
// Edge-biased generators
// ---------------------------------------------------------------------------

pub fn iters() -> usize {
    std::env::var("QSIM_FUZZ_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
}

pub fn base_seed() -> u64 {
    std::env::var("QSIM_FUZZ_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0xA0D1_7000)
}

pub fn edge_angle(rng: &mut StdRng) -> f64 {
    const SPECIAL: [f64; 8] = [
        0.0,
        PI / 4.0,
        PI / 2.0,
        PI,
        3.0 * PI / 2.0,
        2.0 * PI,
        -PI / 2.0,
        -PI,
    ];
    match rng.random_range(0..4) {
        0 => SPECIAL[rng.random_range(0..SPECIAL.len())],
        1 => {
            let eps = [1e-15, -1e-15, 1e-9, -1e-9, 1e-6][rng.random_range(0..5)];
            SPECIAL[rng.random_range(0..SPECIAL.len())] + eps
        }
        _ => rng.random_range(-4.0 * PI..4.0 * PI),
    }
}

pub fn edge_qubit(rng: &mut StdRng, n: usize) -> usize {
    match rng.random_range(0..4) {
        0 => 0,
        1 => n - 1,
        _ => rng.random_range(0..n),
    }
}

/// Distinct pair, biased to adjacent and (0, n-1).
pub fn edge_pair(rng: &mut StdRng, n: usize) -> (usize, usize) {
    let (a, b) = match rng.random_range(0..5) {
        0 => (0, n - 1),
        1 => {
            let a = rng.random_range(0..n - 1);
            (a, a + 1)
        }
        2 => (0, 1),
        3 => (n - 2, n - 1),
        _ => {
            let a = rng.random_range(0..n);
            let mut b = rng.random_range(0..n - 1);
            if b >= a {
                b += 1;
            }
            (a, b)
        }
    };
    if rng.random_bool(0.5) {
        (b, a)
    } else {
        (a, b)
    }
}

pub fn distinct3(rng: &mut StdRng, n: usize) -> (usize, usize, usize) {
    let mut v: Vec<usize> = match rng.random_range(0..3) {
        0 => vec![0, n / 2, n - 1],
        1 => {
            let a = rng.random_range(0..n - 2);
            vec![a, a + 1, a + 2]
        }
        _ => {
            let mut v = Vec::new();
            while v.len() < 3 {
                let q = rng.random_range(0..n);
                if !v.contains(&q) {
                    v.push(q);
                }
            }
            v
        }
    };
    // random order (all 6 permutations)
    for i in (1..3).rev() {
        let j = rng.random_range(0..=i);
        v.swap(i, j);
    }
    (v[0], v[1], v[2])
}

pub fn random_gate(rng: &mut StdRng, n: usize, clifford_only: bool, allow_t: bool) -> Gate {
    let q = edge_qubit(rng, n);
    let k = if n == 1 {
        rng.random_range(0..12)
    } else if n == 2 {
        rng.random_range(0..17)
    } else {
        rng.random_range(0..18)
    };
    let g = match k {
        0 => Gate::H(q),
        1 => Gate::X(q),
        2 => Gate::Y(q),
        3 => Gate::Z(q),
        4 => Gate::S(q),
        5 => Gate::Sdg(q),
        6 => Gate::T(q),
        7 => Gate::Tdg(q),
        8 => Gate::Rx(q, edge_angle(rng)),
        9 => Gate::Ry(q, edge_angle(rng)),
        10 => Gate::Rz(q, edge_angle(rng)),
        11 => Gate::Phase(q, edge_angle(rng)),
        12 | 13 => {
            let (a, b) = edge_pair(rng, n);
            Gate::Cnot(a, b)
        }
        14 => {
            let (a, b) = edge_pair(rng, n);
            Gate::Cz(a, b)
        }
        15 => {
            let (a, b) = edge_pair(rng, n);
            Gate::Swap(a, b)
        }
        16 => {
            let (a, b) = edge_pair(rng, n);
            Gate::CPhase(a, b, edge_angle(rng))
        }
        _ => {
            let (a, b, t) = distinct3(rng, n);
            Gate::Ccx(a, b, t)
        }
    };
    if clifford_only && !g.is_clifford() {
        return random_gate(rng, n, clifford_only, allow_t);
    }
    if allow_t && !g.is_clifford() && !g.is_t() {
        return random_gate(rng, n, clifford_only, allow_t);
    }
    g
}

pub fn random_circuit(
    rng: &mut StdRng,
    n: usize,
    depth: usize,
    clifford_only: bool,
    clifford_t: bool,
) -> Circuit {
    let mut c = Circuit::new(n);
    for _ in 0..depth {
        c.gate(random_gate(rng, n, clifford_only, clifford_t));
    }
    c
}

pub fn max_amp_diff(a: &[C], b: impl Iterator<Item = C>) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

pub const SIZES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13];
