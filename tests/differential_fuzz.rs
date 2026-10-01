//! Differential fuzz harness (audit).
//!
//! Every backend is compared against `RefSv`, a deliberately naive dense
//! state vector defined *in this file* with its own gate matrices, so a
//! change to any kernel in `src/` (fused, strided, SIMD, parallel, ...) is
//! checked against code it does not share.
//!
//! Generators are biased towards edge cases: qubit 0 and the top qubit,
//! adjacent and maximally distant pairs, n = 1, and angles near 0, π/2, π
//! and 2π. Tolerances: f64 amplitudes 1e-12 (scaled by depth), f32
//! amplitudes 1e-5, tableau probabilities exact (dyadic), Pauli-path and
//! exact-MPS values 1e-10.
//!
//! `QSIM_FUZZ_ITERS` (default 1) multiplies the number of random cases;
//! `QSIM_FUZZ_SEED` changes the base seed. Failures print the seed and the
//! offending circuit.

use num_complex::Complex64 as C;
use qsim_lab::pauli_path::{self, PauliSum, DEFAULT_MAX_TERMS};
use qsim_lab::{Circuit, Gate, Mps, Simulator, StateVectorF32, StateVectorF64, Tableau};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Independent reference
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct RefSv {
    n: usize,
    a: Vec<C>,
}

fn cx(re: f64, im: f64) -> C {
    C::new(re, im)
}

impl RefSv {
    fn new(n: usize) -> Self {
        let mut a = vec![C::new(0.0, 0.0); 1 << n];
        a[0] = cx(1.0, 0.0);
        RefSv { n, a }
    }

    /// 2x2 matrix on qubit q, written from the textbook definitions.
    fn m1(g: &Gate) -> [[C; 2]; 2] {
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

    fn bit(i: usize, q: usize) -> usize {
        (i >> q) & 1
    }

    fn apply(&mut self, g: &Gate) {
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

    fn run(c: &Circuit) -> Self {
        let mut s = RefSv::new(c.num_qubits);
        for g in c.gates() {
            s.apply(g);
        }
        s
    }

    fn probs(&self) -> Vec<f64> {
        self.a.iter().map(|x| x.norm_sqr()).collect()
    }

    fn prob_one(&self, q: usize) -> f64 {
        self.a
            .iter()
            .enumerate()
            .filter(|(i, _)| Self::bit(*i, q) == 1)
            .map(|(_, x)| x.norm_sqr())
            .sum()
    }

    fn collapse(&mut self, q: usize, outcome: bool) {
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
    fn pauli_expectation(&self, p: &str) -> f64 {
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

fn iters() -> usize {
    std::env::var("QSIM_FUZZ_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
}

fn base_seed() -> u64 {
    std::env::var("QSIM_FUZZ_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0xA0D1_7000)
}

fn edge_angle(rng: &mut StdRng) -> f64 {
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

fn edge_qubit(rng: &mut StdRng, n: usize) -> usize {
    match rng.random_range(0..4) {
        0 => 0,
        1 => n - 1,
        _ => rng.random_range(0..n),
    }
}

/// Distinct pair, biased to adjacent and (0, n-1).
fn edge_pair(rng: &mut StdRng, n: usize) -> (usize, usize) {
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

fn distinct3(rng: &mut StdRng, n: usize) -> (usize, usize, usize) {
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

fn random_gate(rng: &mut StdRng, n: usize, clifford_only: bool, allow_t: bool) -> Gate {
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

fn random_circuit(
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

fn max_amp_diff(a: &[C], b: impl Iterator<Item = C>) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

const SIZES: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13];

// ---------------------------------------------------------------------------
// State vector
// ---------------------------------------------------------------------------

#[test]
fn sv_f64_matches_reference() {
    let mut worst = 0.0f64;
    for it in 0..20 * iters() {
        for &n in &SIZES {
            let seed = base_seed() ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let depth = rng.random_range(1..120);
            let c = random_circuit(&mut rng, n, depth, false, false);
            let r = RefSv::run(&c);
            let mut sv = StateVectorF64::new(n);
            sv.apply_circuit(&c).unwrap();
            let d = max_amp_diff(&r.a, (0..1 << n).map(|i| sv.amplitude(i)));
            worst = worst.max(d);
            assert!(
                d <= 1e-12,
                "f64 Δ={d:e} seed={seed} n={n} circuit={:?}",
                c.ops
            );
            // gate-by-gate path through the Simulator trait must agree too
            let mut sv2 = StateVectorF64::new(n);
            let mut rng2 = StdRng::seed_from_u64(1);
            c.run(&mut sv2, &mut rng2).unwrap();
            let d2 = max_amp_diff(&r.a, (0..1 << n).map(|i| sv2.amplitude(i)));
            assert!(d2 <= 1e-12, "f64 run() Δ={d2:e} seed={seed} n={n}");
        }
    }
    eprintln!("sv_f64 worst |Δamp| = {worst:e}");
}

#[test]
fn sv_f32_matches_reference() {
    let mut worst = 0.0f64;
    for it in 0..20 * iters() {
        for &n in &SIZES {
            let seed = base_seed() ^ 0xF32 ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let depth = rng.random_range(1..60);
            let c = random_circuit(&mut rng, n, depth, false, false);
            let r = RefSv::run(&c);
            let mut sv = StateVectorF32::new(n);
            sv.apply_circuit(&c).unwrap();
            let d = max_amp_diff(&r.a, (0..1 << n).map(|i| sv.amplitude(i)));
            worst = worst.max(d);
            assert!(
                d <= 1e-5,
                "f32 Δ={d:e} seed={seed} n={n} circuit={:?}",
                c.ops
            );
        }
    }
    eprintln!("sv_f32 worst |Δamp| = {worst:e}");
}

/// Larger registers so parallel / blocked code paths (which typically only
/// kick in above a size threshold) are exercised. 16 and 18 qubits.
#[test]
fn sv_large_registers_match_reference() {
    for (k, &n) in [16usize, 18].iter().enumerate() {
        let seed = base_seed() ^ 0x1A46 ^ k as u64;
        let mut rng = StdRng::seed_from_u64(seed);
        let c = random_circuit(&mut rng, n, 40 * iters(), false, false);
        let r = RefSv::run(&c);
        let mut sv = StateVectorF64::new(n);
        sv.apply_circuit(&c).unwrap();
        let d = max_amp_diff(&r.a, (0..1 << n).map(|i| sv.amplitude(i)));
        assert!(d <= 1e-12, "f64 large Δ={d:e} seed={seed} n={n}");
        let mut s32 = StateVectorF32::new(n);
        s32.apply_circuit(&c).unwrap();
        let d32 = max_amp_diff(&r.a, (0..1 << n).map(|i| s32.amplitude(i)));
        assert!(d32 <= 1e-5, "f32 large Δ={d32:e} seed={seed} n={n}");
    }
}

/// Mid-circuit measurement, repeated measurement and reset on the state
/// vector: every outcome must have nonzero reference probability, the
/// post-measurement state must equal the collapsed reference, and repeating
/// a measurement must reproduce it.
#[test]
fn sv_measure_reset_match_reference() {
    for it in 0..30 * iters() {
        for &n in &[1usize, 2, 3, 5, 8] {
            let seed = base_seed() ^ 0x3EA5 ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let mut r = RefSv::new(n);
            let mut sv = StateVectorF64::new(n);
            for _ in 0..rng.random_range(1..8) {
                for _ in 0..rng.random_range(0..10) {
                    let g = random_gate(&mut rng, n, false, false);
                    r.apply(&g);
                    sv.apply_gate(&g).unwrap();
                }
                let q = edge_qubit(&mut rng, n);
                let p1 = r.prob_one(q);
                let reset = rng.random_bool(0.3);
                let mut mrng = StdRng::seed_from_u64(rng.random());
                if reset {
                    Simulator::reset(&mut sv, q, &mut mrng).unwrap();
                    // reference: result of reset is P0 ψ / |..| or X P1 ψ / |..|,
                    // whichever branch was taken; check qubit q is |0> and
                    // the state matches one of the two branches.
                    assert!(
                        sv.prob_one(q) < 1e-12,
                        "reset left P(1)={} seed={seed}",
                        sv.prob_one(q)
                    );
                    let mut b0 = r.clone();
                    let mut b1 = r.clone();
                    let ok0 = p1 < 1.0 - 1e-12 && {
                        b0.collapse(q, false);
                        max_amp_diff(&b0.a, (0..1 << n).map(|i| sv.amplitude(i))) < 1e-10
                    };
                    let ok1 = p1 > 1e-12 && {
                        b1.collapse(q, true);
                        b1.apply(&Gate::X(q));
                        max_amp_diff(&b1.a, (0..1 << n).map(|i| sv.amplitude(i))) < 1e-10
                    };
                    assert!(ok0 || ok1, "reset state matches neither branch seed={seed}");
                    r = if ok0 { b0 } else { b1 };
                } else {
                    let m = Simulator::measure(&mut sv, q, &mut mrng).unwrap();
                    let pm = if m { p1 } else { 1.0 - p1 };
                    assert!(
                        pm > 1e-12,
                        "measured outcome {m} with ref prob {pm} seed={seed}"
                    );
                    r.collapse(q, m);
                    let d = max_amp_diff(&r.a, (0..1 << n).map(|i| sv.amplitude(i)));
                    assert!(d < 1e-10, "post-measure Δ={d:e} seed={seed}");
                    let m2 = Simulator::measure(&mut sv, q, &mut mrng).unwrap();
                    assert_eq!(m, m2, "repeated measurement differs seed={seed}");
                }
            }
        }
    }
}

/// Outcome sampling: chi-square goodness of fit of `sample()` against the
/// reference distribution (8 qubits, 20k shots, very loose p < 1e-6 bound).
#[test]
fn sv_sampling_distribution() {
    let n = 6;
    let seed = base_seed() ^ 0x5A3;
    let mut rng = StdRng::seed_from_u64(seed);
    let c = random_circuit(&mut rng, n, 40, false, false);
    let p = RefSv::run(&c).probs();
    let mut sv = StateVectorF64::new(n);
    sv.apply_circuit(&c).unwrap();
    let shots = 20_000;
    let samples = sv.sample(shots, &mut rng);
    let mut counts = vec![0usize; 1 << n];
    for s in samples {
        counts[s] += 1;
    }
    let mut chi2 = 0.0;
    let mut dof = 0;
    for (i, &pi) in p.iter().enumerate() {
        let e = pi * shots as f64;
        if e < 5.0 {
            assert!(
                pi > 0.0 || counts[i] == 0,
                "sampled outcome {i} with probability 0"
            );
            continue;
        }
        chi2 += (counts[i] as f64 - e).powi(2) / e;
        dof += 1;
    }
    // chi2 < dof + 6 sqrt(2 dof) + 10 is a ~6σ bound
    let bound = dof as f64 + 6.0 * (2.0 * dof as f64).sqrt() + 10.0;
    assert!(chi2 < bound, "chi2={chi2} dof={dof} seed={seed}");
}

// ---------------------------------------------------------------------------
// Stabilizer tableau (exact)
// ---------------------------------------------------------------------------

#[test]
fn tableau_probabilities_exact() {
    for it in 0..25 * iters() {
        for &n in &[1usize, 2, 3, 4, 5, 7, 9, 11] {
            let seed = base_seed() ^ 0x7AB ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let depth = rng.random_range(1..80);
            let c = random_circuit(&mut rng, n, depth, true, false);
            let r = RefSv::run(&c);
            let mut t = Tableau::new(n);
            c.run(&mut t, &mut rng).unwrap();
            for (i, pr) in r.probs().into_iter().enumerate() {
                let pt = t.probability(i);
                // reference probability rounded to the nearest dyadic must
                // equal the tableau's exactly
                let snapped = if pr < 1e-9 {
                    0.0
                } else {
                    2f64.powi(pr.log2().round() as i32)
                };
                assert!(
                    (pr - snapped).abs() < 1e-9,
                    "ref prob {pr} not dyadic seed={seed}"
                );
                assert_eq!(
                    pt, snapped,
                    "outcome {i}: tableau {pt} vs ref {pr} seed={seed} n={n}"
                );
            }
            for s in t.stabilizers() {
                let sign = if s.starts_with('-') { -1.0 } else { 1.0 };
                let ev = r.pauli_expectation(&s[1..]);
                assert!(
                    (ev - sign).abs() < 1e-9,
                    "stabilizer {s} has <P>={ev} seed={seed}"
                );
            }
        }
    }
}

/// Mid-circuit measurements and resets on the tableau: each outcome must be
/// possible under the reference, deterministic outcomes must match, and the
/// collapsed states must keep agreeing.
#[test]
fn tableau_measure_reset_match_reference() {
    for it in 0..30 * iters() {
        for &n in &[1usize, 2, 3, 6, 9] {
            let seed = base_seed() ^ 0x7AB3 ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let mut r = RefSv::new(n);
            let mut t = Tableau::new(n);
            for _ in 0..rng.random_range(1..10) {
                for _ in 0..rng.random_range(0..12) {
                    let g = random_gate(&mut rng, n, true, false);
                    r.apply(&g);
                    t.apply_gate(&g).unwrap();
                }
                let q = edge_qubit(&mut rng, n);
                let p1 = r.prob_one(q);
                assert!(p1 < 1e-9 || (p1 - 0.5).abs() < 1e-9 || p1 > 1.0 - 1e-9);
                let det = t.peek(q);
                match det {
                    Some(b) => assert!(
                        (p1 - if b { 1.0 } else { 0.0 }).abs() < 1e-9,
                        "peek={b} but P1={p1} seed={seed}"
                    ),
                    None => assert!(
                        (p1 - 0.5).abs() < 1e-9,
                        "peek random but P1={p1} seed={seed}"
                    ),
                }
                let m = t.measure_qubit(q, &mut rng);
                assert_eq!(
                    t.measure_qubit(q, &mut rng),
                    m,
                    "repeat differs seed={seed}"
                );
                r.collapse(q, m);
                if rng.random_bool(0.3) {
                    t.reset_qubit(q, &mut rng);
                    if m {
                        r.apply(&Gate::X(q));
                    }
                }
            }
            for (i, pr) in r.probs().into_iter().enumerate() {
                assert!(
                    (t.probability(i) - pr).abs() < 1e-9,
                    "final dist mismatch seed={seed}"
                );
            }
        }
    }
}

/// Tableau shot sampling: every shot in the support, and per-outcome counts
/// consistent with the uniform distribution over the support.
#[test]
fn tableau_sampling_distribution() {
    for k in 0..4 * iters() {
        let n = 5 + k % 4;
        let seed = base_seed() ^ 0x5AB ^ k as u64;
        let mut rng = StdRng::seed_from_u64(seed);
        let c = random_circuit(&mut rng, n, 30, true, false);
        let p = RefSv::run(&c).probs();
        let mut t = Tableau::new(n);
        c.run(&mut t, &mut rng).unwrap();
        let shots = 8000;
        let mut counts = vec![0usize; 1 << n];
        for s in t.sample(shots, &mut rng) {
            let idx = s
                .iter()
                .enumerate()
                .fold(0, |acc, (q, &b)| acc | ((b as usize) << q));
            assert!(
                p[idx] > 1e-9,
                "sampled impossible outcome {idx} seed={seed}"
            );
            counts[idx] += 1;
        }
        let mut chi2 = 0.0;
        let mut dof = 0;
        for (i, &pi) in p.iter().enumerate() {
            if pi > 1e-9 {
                let e = pi * shots as f64;
                chi2 += (counts[i] as f64 - e).powi(2) / e;
                dof += 1;
            }
        }
        let bound = dof as f64 + 6.0 * (2.0 * dof as f64).sqrt() + 10.0;
        assert!(chi2 < bound, "tableau chi2={chi2} dof={dof} seed={seed}");
    }
}

// ---------------------------------------------------------------------------
// Pauli-path (Clifford+T and arbitrary rotations) and MPS
// ---------------------------------------------------------------------------

#[test]
fn pauli_path_matches_reference() {
    let mut worst = 0.0f64;
    for it in 0..15 * iters() {
        for &n in &[1usize, 2, 3, 4, 6, 8] {
            let seed = base_seed() ^ 0xFA7 ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            // mostly Clifford+T, sometimes arbitrary rotations; keep the
            // number of non-Clifford gates small enough for the term cap
            let mut c = Circuit::new(n);
            let mut non_cliff = 0;
            for _ in 0..rng.random_range(1..60) {
                let g = random_gate(&mut rng, n, false, rng.random_bool(0.7));
                if matches!(g, Gate::Ccx(..)) {
                    continue;
                }
                if !g.is_clifford() {
                    if non_cliff >= 8 {
                        continue;
                    }
                    non_cliff += 1;
                }
                c.gate(g);
            }
            let r = RefSv::run(&c);
            let p: String = (0..n)
                .map(|_| ['I', 'X', 'Y', 'Z'][rng.random_range(0..4)])
                .collect();
            let (v, _) =
                pauli_path::expectation(&c, &PauliSum::from_str_single(&p), DEFAULT_MAX_TERMS)
                    .unwrap();
            let e = r.pauli_expectation(&p);
            worst = worst.max((v - e).abs());
            assert!(
                (v - e).abs() < 1e-10,
                "pauli-path <{p}>={v} ref={e} seed={seed} circuit={:?}",
                c.ops
            );
            // marginal distribution on up to 3 qubits
            let k = n.min(3);
            let qs: Vec<usize> = if k == n {
                (0..n).collect()
            } else {
                vec![0, n / 2, n - 1]
            };
            let md = pauli_path::marginal_distribution(&c, &qs).unwrap();
            let probs = r.probs();
            for (b, &mv) in md.iter().enumerate() {
                let rv: f64 = probs
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| {
                        qs.iter()
                            .enumerate()
                            .all(|(j, &q)| ((i >> q) & 1) == ((b >> j) & 1))
                    })
                    .map(|(_, x)| x)
                    .sum();
                assert!(
                    (mv - rv).abs() < 1e-10,
                    "marginal {b}: {mv} vs {rv} seed={seed}"
                );
            }
        }
    }
    eprintln!("pauli_path worst |Δ| = {worst:e}");
}

#[test]
fn mps_exact_matches_reference() {
    for it in 0..10 * iters() {
        for &n in &[1usize, 2, 3, 5, 8, 10] {
            let seed = base_seed() ^ 0x3B5 ^ ((it as u64) << 16) ^ n as u64;
            let mut rng = StdRng::seed_from_u64(seed);
            let depth = rng.random_range(1..60);
            let c = random_circuit(&mut rng, n, depth, false, false);
            let r = RefSv::run(&c);
            let mut m = Mps::new(n, 1 << (n / 2 + 1));
            c.run(&mut m, &mut rng).unwrap();
            let d = max_amp_diff(&r.a, (0..1u128 << n).map(|i| m.amplitude(i)));
            assert!(
                d < 1e-9,
                "mps Δ={d:e} seed={seed} n={n} circuit={:?}",
                c.ops
            );
        }
    }
}

/// Memory caps must turn oversized registers into errors, not OOM aborts.
/// (Only the failing side is checked: allocating the largest allowed
/// register would cost ~0.5 GB on a shared box.)
#[test]
fn memory_caps_reject_oversized_registers() {
    assert!(qsim_lab::StateVector::<f32>::try_new(40).is_err());
    assert!(qsim_lab::StateVector::<f64>::try_new(40).is_err());
    assert!(Tableau::try_new(1_000_000).is_err());
    assert!(Tableau::try_new(64).is_ok());
}
