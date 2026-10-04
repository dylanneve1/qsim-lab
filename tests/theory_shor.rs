//! Executable checks for the theorems of `research/theory-shor.md`.
//!
//! Each test would fail if the corresponding statement were false:
//! * T1 (support law): exact support of the semiclassical work register,
//!   checked in exact cyclotomic arithmetic against the closed-form
//!   divisibility criterion, and against the gate-level bit-sliced engine on
//!   whole measurement trees; closed form of Σ B_i and the peak; the
//!   per-round exception-probability bound 4 / r_odd.
//! * T2 (borrowed magic): exact nullity formula ν = dim aff(S) − dim A_ψ for
//!   branch states (brute force over all 4^n Paulis), ν = 0 at every Toffoli
//!   boundary for equal-weight two-branch inputs of lowered NCT circuits,
//!   interior bound ν ≤ 6, and failure for every other input.
//! * T3 (noise windows): end-window invariance for arbitrary faults
//!   (exact distributions), start-window lower bound for phase-type faults,
//!   textbook sign formula for Z faults, and the dephasing counting bound.
//!
//! Run: `cargo test --release --test theory_shor` (≈ 1–3 min, 2 threads).

use num_complex::Complex64;
use qsim_lab::shor::noisy::{self, Fault, NoiseKind, NoisyCircuit, Pauli, Site};
use qsim_lab::shor::sliced::SlicedState;
use qsim_lab::shor::{Instance, Oracle, OrderFindingState};
use qsim_lab::Gate;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

// ---------------------------------------------------------------- helpers

fn mulmod(a: u64, b: u64, n: u64) -> u64 {
    (u128::from(a) * u128::from(b) % u128::from(n)) as u64
}

fn powmod(mut a: u64, mut e: u128, n: u64) -> u64 {
    let mut r = 1 % n;
    a %= n;
    while e > 0 {
        if e & 1 == 1 {
            r = mulmod(r, a, n);
        }
        a = mulmod(a, a, n);
        e >>= 1;
    }
    r
}

fn order(a: u64, n: u64) -> u64 {
    let mut r = 1;
    let mut x = a % n;
    while x != 1 {
        x = mulmod(x, a, n);
        r += 1;
    }
    r
}

/// R_i = r / gcd(r, 2^(t-i)): order of a^(2^(t-i)).
fn r_i(r: u64, t: usize, i: usize) -> u64 {
    let v = (r.trailing_zeros() as usize).min(t - i);
    r >> v
}

/// B_i = min(2^i, R_i).
fn b_i(r: u64, t: usize, i: usize) -> u64 {
    let ri = r_i(r, t, i);
    if i >= 63 {
        ri
    } else {
        ri.min(1u64 << i)
    }
}

/// Closed form (Theorem 1b) for |S_i| after the measured low bits `y`
/// (`y < 2^i`), with R = R_i. Returns 0 iff the prefix has probability 0.
fn support_closed_form(rr: u64, i: usize, y: u128) -> u64 {
    let p = 1u128 << i;
    let r = u128::from(rr);
    if p <= r {
        return p as u64;
    }
    let s = p % r;
    if (r * y) % p == 0 {
        return rr; // ζ = 1: no cancellation
    }
    let kill_hi = (s * y) % p == 0; // J = q classes (k ≥ s) vanish
    let kill_lo = ((r - s) * y) % p == 0; // J = q+1 classes (k < s) vanish
    assert!(!(kill_hi && kill_lo));
    let mut n = rr;
    if kill_hi {
        n -= (r - s) as u64;
    }
    if kill_lo {
        n -= s as u64;
    }
    n
}

/// Exact support count in Z[ζ], ζ = e^{2πi/2^i}: the coefficient of g^k is
/// c_k = Σ_{m<2^i, m≡k (R)} ζ^{-m y}; reduce with ζ^{2^(i-1)} = -1 and test
/// for the zero vector. Independent of the closed form.
fn support_cyclotomic(rr: u64, i: usize, y: u128) -> u64 {
    if i == 0 {
        return 1;
    }
    let p = 1u128 << i;
    let h = (p / 2) as usize;
    let kmax = (rr as u128).min(p) as u64;
    let mut cnt = 0;
    let mut v = vec![0i64; h];
    for k in 0..kmax {
        v.iter_mut().for_each(|x| *x = 0);
        let mut m = u128::from(k);
        while m < p {
            let e = ((p - (m * y) % p) % p) as usize;
            if e < h {
                v[e] += 1;
            } else {
                v[e - h] -= 1;
            }
            m += u128::from(rr);
        }
        if v.iter().any(|&x| x != 0) {
            cnt += 1;
        }
    }
    cnt
}

/// Exact peak criterion: ∃ s, |y/2^t − s/r| < 1/(2r²)  ⇔  2r·dist(yr, 2^t Z) < 2^t.
fn peak_ok(y: u128, r: u64, t: usize) -> bool {
    let p = 1u128 << t;
    let m = (y * u128::from(r)) % p;
    let d = m.min(p - m);
    2 * u128::from(r) * d < p
}

fn p_ok(dist: &[f64], r: u64, t: usize) -> f64 {
    dist.iter()
        .enumerate()
        .filter(|(y, _)| peak_ok(*y as u128, r, t))
        .map(|(_, p)| p)
        .sum()
}

// ---------------------------------------------------------------- T1

/// T1(b): closed-form divisibility criterion == exact cyclotomic support,
/// exhaustively over all prefixes y < 2^i, i ≤ 10, R ≤ 70.
#[test]
fn t1_support_closed_form_equals_exact_cyclotomic() {
    let mut checked = 0u64;
    let mut deficient = 0u64;
    for rr in 1..=70u64 {
        for i in 0..=10usize {
            for y in 0..(1u128 << i) {
                let a = support_closed_form(rr, i, y);
                let b = support_cyclotomic(rr, i, y);
                assert_eq!(a, b, "R={rr} i={i} y={y}");
                let bi = rr.min(1 << i);
                assert!(b <= bi);
                if b < bi && b > 0 {
                    deficient += 1;
                }
                checked += 1;
            }
        }
    }
    // random larger instances
    let mut rng = StdRng::seed_from_u64(11);
    for _ in 0..3000 {
        let rr = rng.random_range(1..400u64);
        let i = rng.random_range(8..=14usize);
        let y = rng.random_range(0..(1u128 << i));
        // bias towards highly 2-divisible prefixes (where cancellation lives)
        let y = if rng.random_bool(0.5) {
            y >> rng.random_range(0..i) << rng.random_range(0..i)
        } else {
            y
        } % (1u128 << i);
        assert_eq!(
            support_closed_form(rr, i, y),
            support_cyclotomic(rr, i, y),
            "R={rr} i={i} y={y}"
        );
        checked += 1;
    }
    eprintln!("T1b: {checked} (R,i,y) checked, {deficient} deficient (nonzero) cases in the exhaustive part");
    assert!(deficient > 0);
}

/// The semiclassical state before round i, unnormalised, from the formula
/// ψ_i = 2^{-i} Σ_{m<2^i} e^{-2πi m y/2^i} |g^m⟩ (returned per class k < R).
fn formula_amps(rr: u64, i: usize, y: u128) -> Vec<Complex64> {
    let p = 1u128 << i;
    let kmax = (rr as u128).min(p) as usize;
    let mut c = vec![Complex64::new(0.0, 0.0); kmax];
    let mut m = 0u128;
    while m < p {
        let k = (m % u128::from(rr)) as usize;
        let ph = -2.0 * PI * (((m * y) % p) as f64) / (p as f64);
        c[k] += Complex64::from_polar(1.0, ph) / (p as f64);
        m += 1;
    }
    c
}

/// T1(a,b) against the real gate-level circuit: walk the whole measurement
/// tree of the bit-sliced engine (windowed oracle, every gate evaluated) and
/// compare the stored work-register state before every round with the
/// formula: same support (exact count, closed form), amplitudes equal to
/// 1e-12 after normalisation, |S_i| ≤ B_i.
#[test]
fn t1_support_law_on_gate_level_tree() {
    let cases: &[(u64, u64)] = &[
        (15, 7),
        (15, 2),
        (15, 11),
        (21, 2),
        (21, 5),
        (33, 5),
        (35, 2),
        (35, 8),
        (35, 11),
        (39, 5),
        (51, 5),
        (55, 2),
        (57, 5),
        (65, 2),
        (77, 3),
        (85, 3),
        (91, 3),
        (119, 3),
        (143, 2),
        (143, 5),
    ];
    let mut nodes = 0u64;
    let mut defic = 0u64;
    let mut float_residue = 0u64;
    for &(n, a) in cases {
        let inst = Instance::new(n, a, Oracle::Windowed(4));
        let t = inst.t;
        let r = order(a, n);
        // exhaustive for t <= 12, otherwise 200 random root-to-leaf paths
        let mut stack: Vec<(SlicedState<f64>, usize, u128, f64)> =
            vec![(SlicedState::<f64>::new(&inst), 0, 0, 1.0)];
        let mut rng = StdRng::seed_from_u64(n * 1000 + a);
        let exhaustive = t <= 12;
        let mut paths_left = 200;
        while let Some((mut s, i, y, prob)) = stack.pop() {
            // check state before round i
            let rr = r_i(r, t, i);
            let exact = support_closed_form(rr, i, y);
            let amps = formula_amps(rr, i, y);
            let norm2: f64 = amps.iter().map(|c| c.norm_sqr()).sum();
            assert!(
                (norm2 - prob).abs() < 1e-12,
                "P(prefix) N={n} a={a} i={i} y={y}: {norm2} vs {prob}"
            );
            let g = powmod(a, 1u128 << (t - i), n);
            let mut sim: Vec<(u64, Complex64)> = s.work().map(|(k, v)| (k, v)).collect();
            sim.sort_by_key(|e| e.0);
            let mut nonzero = 0u64;
            let sc = 1.0 / prob.sqrt();
            for (k, c) in amps.iter().enumerate() {
                let key = powmod(g, k as u128, n);
                let sv = sim
                    .binary_search_by_key(&key, |e| e.0)
                    .map(|j| sim[j].1)
                    .unwrap_or(Complex64::new(0.0, 0.0));
                assert!(
                    (sv - c * sc).norm() < 1e-9,
                    "amp N={n} a={a} i={i} y={y} k={k}"
                );
                if c.norm() > 1e-9 {
                    nonzero += 1;
                }
            }
            assert_eq!(nonzero, exact, "support N={n} a={a} i={i} y={y}");
            assert!(s.nnz() as u64 >= exact && s.nnz() as u64 <= b_i(r, t, i));
            if s.nnz() as u64 > exact {
                float_residue += 1;
            }
            if exact < b_i(r, t, i) {
                defic += 1;
            }
            nodes += 1;
            if i == t {
                continue;
            }
            s.round(&inst, i, y);
            let p1 = s.prob_one(0);
            let kids: Vec<bool> = if exhaustive {
                vec![false, true]
            } else {
                vec![rng.random::<f64>() < p1]
            };
            for bit in kids {
                let pb = if bit { p1 } else { 1.0 - p1 };
                if pb < 1e-24 {
                    // the formula must say this prefix is impossible
                    let yy = y | (u128::from(bit) << i);
                    assert_eq!(support_closed_form(r_i(r, t, i + 1), i + 1, yy), 0);
                    continue;
                }
                let mut c = s.clone();
                c.keep_final = true;
                c.collapse(0, bit);
                stack.push((c, i + 1, y | (u128::from(bit) << i), prob * pb));
            }
            if !exhaustive && stack.is_empty() && paths_left > 0 {
                paths_left -= 1;
                stack.push((SlicedState::<f64>::new(&inst), 0, 0, 1.0));
            }
        }
    }
    eprintln!("T1 tree: {nodes} states checked, {defic} with exact support < B_i, {float_residue} where the f64 engine kept a rounding residue (stored > exact support)");
    assert!(defic > 0);
}

/// T1(c): Σ_i B_i closed form with exact error term, and peak = max(r_odd, r/2)
/// (r if r odd), for every r < 2^n, n = 2..14.
#[test]
fn t1_sum_and_peak_closed_forms() {
    let mut worst_rel = 0f64;
    for n in 2..=14usize {
        let t = 2 * n;
        for r in 1..(1u64 << n) {
            let nu = r.trailing_zeros() as usize;
            let ro = r >> nu;
            let l = 64 - (ro - 1).leading_zeros() as usize; // ceil(log2 ro); 0 for ro = 1
            let l = if ro == 1 { 0 } else { l };
            let direct: u64 = (0..t).map(|i| b_i(r, t, i)).sum();
            let exact = ro * (t as u64 - nu as u64 - l as u64 - 1) + (1u64 << l) - 1 + r;
            assert_eq!(direct, exact, "n={n} r={r}");
            let approx = ro as f64 * (t as f64 - (r as f64).log2()) + r as f64;
            let e = exact as f64 - approx;
            assert!(
                e <= -1.0 + 1e-9 && e >= -0.0861 * ro as f64 - 1.0 - 1e-9,
                "n={n} r={r} E={e}"
            );
            worst_rel = worst_rel.max(e.abs() / exact as f64);
            let peak = (0..t).map(|i| b_i(r, t, i)).max().unwrap();
            let want = if nu == 0 { r } else { ro.max(r / 2) };
            assert_eq!(peak, want, "peak n={n} r={r}");
        }
    }
    eprintln!("T1c: max relative error of r_odd(2n − log2 r) + r: {worst_rel:.4}");
}

/// T1(d): probability that round i is deficient (|S_i| < B_i) is ≤ 4/r_odd,
/// computed exactly as Σ over deficient prefixes of P(prefix).
#[test]
fn t1_exception_probability_bound() {
    let mut worst = 0f64;
    for n in 3..=7usize {
        let t = 2 * n;
        for r in 2..(1u64 << n) {
            let ro = r >> r.trailing_zeros();
            for i in 1..t.min(13) {
                let rr = r_i(r, t, i);
                let bi = b_i(r, t, i);
                let mut pdef = 0.0;
                let mut ptot = 0.0;
                for y in 0..(1u128 << i) {
                    let amps = formula_amps(rr, i, y);
                    let p: f64 = amps.iter().map(|c| c.norm_sqr()).sum();
                    ptot += p;
                    let s = support_closed_form(rr, i, y);
                    if s > 0 && s < bi {
                        pdef += p;
                    }
                }
                assert!((ptot - 1.0).abs() < 1e-9);
                assert!(
                    pdef <= 4.0 / ro as f64 + 1e-12,
                    "n={n} r={r} i={i} P={pdef}"
                );
                worst = worst.max(pdef * ro as f64);
            }
        }
    }
    eprintln!("T1d: max r_odd·P(deficient round) = {worst:.3} (bound 4)");
}

/// T1(e): the engine's work counter is Σ 2·|S_i|·G_i with |S_i| the exact
/// support of Theorem 1 (along a seeded trajectory).
#[test]
fn t1_work_counter_identity() {
    for &(n, a) in &[(143u64, 5u64), (221, 2), (899, 2), (4087, 3)] {
        let inst = Instance::new(n, a, Oracle::Windowed(4));
        let t = inst.t;
        let r = order(a, n);
        let mut s = SlicedState::<f64>::new(&inst);
        let mut rng = StdRng::seed_from_u64(5);
        let mut y = 0u128;
        let mut w: u128 = 0;
        for i in 0..t {
            let (c, _) = qsim_lab::shor::sliced::oracle_block(&inst, inst.mults[t - 1 - i]);
            let exact = support_closed_form(r_i(r, t, i), i, y);
            w += 2 * u128::from(exact) * c.ops.len() as u128;
            s.round(&inst, i, y);
            let bit = rng.random::<f64>() < s.prob_one(0);
            s.collapse(0, bit);
            y |= u128::from(bit) << i;
        }
        assert_eq!(s.gate_branch_ops, w, "N={n}");
    }
}

// ---------------------------------------------------------------- T2

type Sv = Vec<Complex64>;

fn fwht(v: &mut [Complex64]) {
    let n = v.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (v[j], v[j + h]);
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
        h *= 2;
    }
}

/// Stabilizer nullity by brute force: ν = n − log2 #{(a,b): |⟨ψ|X^a Z^b|ψ⟩| = 1}.
fn nullity_brute(psi: &Sv, n: usize) -> usize {
    let dim = 1usize << n;
    let mut count = 0usize;
    let mut v = vec![Complex64::new(0.0, 0.0); dim];
    for a in 0..dim {
        for x in 0..dim {
            v[x] = psi[x ^ a].conj() * psi[x];
        }
        fwht(&mut v);
        count += v.iter().filter(|z| (z.norm() - 1.0).abs() < 1e-9).count();
    }
    assert!(count.is_power_of_two(), "stabilizer count {count}");
    n - count.trailing_zeros() as usize
}

fn gf2_rank(mut vs: Vec<u64>) -> usize {
    let mut rank = 0;
    for bit in (0..64).rev() {
        if let Some(p) = vs.iter().position(|&v| v >> bit & 1 == 1) {
            let pv = vs.swap_remove(p);
            for v in vs.iter_mut() {
                if *v >> bit & 1 == 1 {
                    *v ^= pv;
                }
            }
            rank += 1;
        }
    }
    rank
}

/// Solvability of { b·u_j = s_j } over GF(2).
fn gf2_solvable(rows: &[(u64, bool)]) -> bool {
    let mut rs: Vec<(u64, bool)> = rows.to_vec();
    let mut out = Vec::new();
    for bit in (0..64).rev() {
        if let Some(p) = rs.iter().position(|r| r.0 >> bit & 1 == 1) {
            let pv = rs.swap_remove(p);
            for r in rs.iter_mut() {
                if r.0 >> bit & 1 == 1 {
                    r.0 ^= pv.0;
                    r.1 ^= pv.1;
                }
            }
            out.push(pv);
        }
    }
    rs.iter().all(|r| r.0 != 0 || !r.1)
}

/// Theorem 2(a): ν = dim aff(S) − dim A_ψ, A_ψ = {a : S⊕a = S, ψ_{x⊕a} = λ(−1)^{b·x} ψ_x on S
/// for some b, λ}. Also returns (d, log2|T(S)|).
fn nullity_formula(psi: &Sv, n: usize) -> (usize, usize, usize) {
    let dim = 1usize << n;
    let supp: Vec<usize> = (0..dim).filter(|&x| psi[x].norm() > 1e-9).collect();
    let s0 = supp[0];
    let d = gf2_rank(supp.iter().map(|&x| (x ^ s0) as u64).collect());
    let in_s = |x: usize| psi[x].norm() > 1e-9;
    let mut a_count = 0usize;
    let mut t_count = 0usize;
    for a in 0..dim {
        if !supp.iter().all(|&x| in_s(x ^ a)) {
            continue;
        }
        t_count += 1;
        // ratio ρ(x) = ψ_{x⊕a}/ψ_x must be λ·(±1) with the sign a character of x⊕s0
        let lam = psi[s0 ^ a] / psi[s0];
        let mut rows = Vec::new();
        let mut ok = true;
        for &x in &supp {
            let q = psi[x ^ a] / psi[x] / lam;
            if (q - 1.0).norm() < 1e-9 {
                rows.push(((x ^ s0) as u64, false));
            } else if (q + 1.0).norm() < 1e-9 {
                rows.push(((x ^ s0) as u64, true));
            } else {
                ok = false;
                break;
            }
        }
        if ok && gf2_solvable(&rows) {
            a_count += 1;
        }
    }
    assert!(a_count.is_power_of_two() && t_count.is_power_of_two());
    (
        d - a_count.trailing_zeros() as usize,
        d,
        t_count.trailing_zeros() as usize,
    )
}

#[derive(Clone, Copy, Debug)]
enum G {
    X(usize),
    Cx(usize, usize),
    Ccx(usize, usize, usize),
}

fn apply_perm(psi: &Sv, g: G) -> Sv {
    let mut out = vec![Complex64::new(0.0, 0.0); psi.len()];
    for (x, &v) in psi.iter().enumerate() {
        let y = match g {
            G::X(q) => x ^ (1 << q),
            G::Cx(c, t) => x ^ (((x >> c) & 1) << t),
            G::Ccx(a, b, t) => x ^ (((x >> a) & (x >> b) & 1) << t),
        };
        out[y] = v;
    }
    out
}

#[derive(Clone, Copy)]
enum Ct {
    H(usize),
    T(usize),
    Tdg(usize),
    Cx(usize, usize),
}

fn apply_ct(psi: &mut Sv, g: Ct) {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    match g {
        Ct::H(q) => {
            for x in 0..psi.len() {
                if x >> q & 1 == 0 {
                    let (a, b) = (psi[x], psi[x | 1 << q]);
                    psi[x] = (a + b) * s;
                    psi[x | 1 << q] = (a - b) * s;
                }
            }
        }
        Ct::T(q) | Ct::Tdg(q) => {
            let ph = Complex64::from_polar(
                1.0,
                if matches!(g, Ct::T(_)) {
                    PI / 4.0
                } else {
                    -PI / 4.0
                },
            );
            for (x, v) in psi.iter_mut().enumerate() {
                if x >> q & 1 == 1 {
                    *v *= ph;
                }
            }
        }
        Ct::Cx(c, t) => *psi = apply_perm(psi, G::Cx(c, t)),
    }
}

/// Nielsen–Chuang Fig. 4.9: Toffoli as 6 CNOT + 7 T/T† + 2 H.
fn lower_toffoli(a: usize, b: usize, c: usize) -> Vec<Ct> {
    use Ct::*;
    vec![
        H(c),
        Cx(b, c),
        Tdg(c),
        Cx(a, c),
        T(c),
        Cx(b, c),
        Tdg(c),
        Cx(a, c),
        T(b),
        T(c),
        H(c),
        Cx(a, b),
        T(a),
        Tdg(b),
        Cx(a, b),
    ]
}

fn random_nct<R: Rng>(n: usize, len: usize, rng: &mut R) -> Vec<G> {
    (0..len)
        .map(|_| {
            let mut q: Vec<usize> = (0..n).collect();
            for i in 0..3 {
                let j = rng.random_range(i..n);
                q.swap(i, j);
            }
            match rng.random_range(0..10) {
                0 => G::X(q[0]),
                1..=3 => G::Cx(q[0], q[1]),
                _ => G::Ccx(q[0], q[1], q[2]),
            }
        })
        .collect()
}

/// T2(a): exact nullity formula and its bounds for random branch states
/// (random supports, flat / ±1 / ±i / random amplitudes), n = 3..6.
#[test]
fn t2_nullity_formula_for_branch_states() {
    let mut rng = StdRng::seed_from_u64(2);
    let mut hist = [0usize; 8];
    let mut log2k_violations = 0;
    for trial in 0..1500 {
        let n = rng.random_range(3..=6usize);
        let dim = 1usize << n;
        let k = rng.random_range(1..=dim.min(16));
        let mut supp: Vec<usize> = (0..dim).collect();
        for i in 0..k {
            let j = rng.random_range(i..dim);
            supp.swap(i, j);
        }
        supp.truncate(k);
        // sometimes force an affine support
        if trial % 3 == 0 {
            let dd = rng.random_range(0..=n.min(4));
            let s0 = rng.random_range(0..dim);
            let gens: Vec<usize> = (0..dd).map(|_| rng.random_range(1..dim)).collect();
            let mut set = vec![s0];
            for g in gens {
                let more: Vec<usize> = set.iter().map(|&x| x ^ g).collect();
                set.extend(more);
                set.sort();
                set.dedup();
            }
            supp = set;
        }
        let kind = rng.random_range(0..4);
        let mut psi = vec![Complex64::new(0.0, 0.0); dim];
        for &x in &supp {
            psi[x] = match kind {
                0 => Complex64::new(1.0, 0.0),
                1 => Complex64::new(if rng.random_bool(0.5) { 1.0 } else { -1.0 }, 0.0),
                2 => Complex64::new(0.0, 1.0).powi(rng.random_range(0..4)),
                _ => Complex64::new(rng.random::<f64>() + 0.1, rng.random::<f64>()),
            };
        }
        let nrm: f64 = psi.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        psi.iter_mut().for_each(|z| *z /= nrm);
        let nb = nullity_brute(&psi, n);
        let (nf, d, lt) = nullity_formula(&psi, n);
        assert_eq!(nb, nf, "trial {trial}");
        let ks = supp.len();
        assert!(nb <= d && d <= n.min(ks - 1));
        assert!(nb + lt >= d);
        if (nb as f64) > (ks as f64).log2() + 1e-9 {
            log2k_violations += 1;
        }
        hist[nb] += 1;
    }
    eprintln!(
        "T2a: nullity histogram {hist:?}; states with ν > log2(#branches): {log2k_violations}"
    );
    // the 'log2(#branches)' bound is false; the affine-dimension bound is right
    assert!(log2k_violations > 0);
    // the explicit counterexample of the write-up: 4 branches {0,e1,e2,e3}, ν = 3
    let mut psi = vec![Complex64::new(0.0, 0.0); 8];
    for x in [0usize, 1, 2, 4] {
        psi[x] = Complex64::new(0.5, 0.0);
    }
    assert_eq!(nullity_brute(&psi, 3), 3);
}

/// T2(b,c): for (|0⟩+|1⟩)/√2 ⊗ |x⟩ ⊗ |0…0⟩ (and any equal-weight two-branch
/// input) every random X/CNOT/Toffoli circuit with Toffolis lowered to the
/// 7-T network has ν = 0 at every Toffoli boundary and ν ≤ 6 inside each
/// network; unequal weights give ν = 1 at every boundary.
#[test]
fn t2_two_branch_inputs_have_zero_magic_at_toffoli_boundaries() {
    // the lowering is a Toffoli
    for x in 0..8usize {
        let mut v = vec![Complex64::new(0.0, 0.0); 8];
        v[x] = Complex64::new(1.0, 0.0);
        for g in lower_toffoli(0, 1, 2) {
            apply_ct(&mut v, g);
        }
        let want = apply_perm(
            &{
                let mut w = vec![Complex64::new(0.0, 0.0); 8];
                w[x] = Complex64::new(1.0, 0.0);
                w
            },
            G::Ccx(0, 1, 2),
        );
        for y in 0..8 {
            assert!((v[y] - want[y]).norm() < 1e-12);
        }
    }
    let mut rng = StdRng::seed_from_u64(3);
    let mut max_interior = 0usize;
    let mut boundaries = 0usize;
    for trial in 0..60 {
        let n = 6 + trial % 2;
        let dim = 1usize << n;
        // 0,1: equal weights, relative phase a power of i (stabilizer input);
        // 2: equal weights, generic phase; 3: unequal weights
        let kind = trial % 4;
        let equal = kind < 2;
        let xin = rng.random_range(0..8usize) << 1; // qubits 1..3 data, rest ancilla 0
        let mut psi = vec![Complex64::new(0.0, 0.0); dim];
        let (w0, w1) = if kind < 3 {
            (0.5f64.sqrt(), 0.5f64.sqrt())
        } else {
            (0.6, 0.8)
        };
        let ph = if kind < 2 {
            rng.random_range(0..4) as f64 * PI / 2.0
        } else {
            0.3 + rng.random::<f64>()
        };
        psi[xin] = Complex64::new(w0, 0.0);
        psi[xin | 1] = Complex64::from_polar(w1, ph);
        for g in random_nct(n, 20, &mut rng) {
            match g {
                G::Ccx(a, b, c) => {
                    // Theorem 2(e): inside the network ν ≤ dim aff(S with the target bit deleted) + 1
                    let supp: Vec<usize> = (0..dim)
                        .filter(|&x| psi[x].norm() > 1e-9)
                        .map(|x| x & !(1 << c))
                        .collect();
                    let d_rest = gf2_rank(supp.iter().map(|&x| (x ^ supp[0]) as u64).collect());
                    for ct in lower_toffoli(a, b, c) {
                        apply_ct(&mut psi, ct);
                        let nu = nullity_brute(&psi, n);
                        assert!(
                            nu <= d_rest + 1,
                            "in-flight ν {nu} > d_rest + 1 = {}",
                            d_rest + 1
                        );
                        max_interior = max_interior.max(nu);
                    }
                }
                _ => psi = apply_perm(&psi, g),
            }
            let nu = nullity_brute(&psi, n);
            assert_eq!(nu, if equal { 0 } else { 1 }, "trial {trial}");
            boundaries += 1;
        }
    }
    assert!(max_interior <= 2);
    eprintln!("T2b: {boundaries} Toffoli boundaries checked; max nullity inside a lowered Toffoli (from ν_b ∈ {{0,1}}): {max_interior}");
}

/// T2(d): conversely, every clean-ancilla input with ≥ 3 branches, or two
/// branches of unequal weight, reaches ν > 0 at some boundary of some NCT
/// circuit (found by random search; the theorem says one always exists).
#[test]
fn t2_only_equal_two_branch_inputs_are_universally_magic_free() {
    let mut rng = StdRng::seed_from_u64(4);
    let n = 5;
    let dim = 1usize << n;
    for trial in 0..60 {
        let k = 3 + trial % 6; // 3..8 branches on the low 4 qubits, qubit 4 ancilla
        let mut supp: Vec<usize> = (0..16).collect();
        for i in 0..k {
            let j = rng.random_range(i..16);
            supp.swap(i, j);
        }
        let mut psi = vec![Complex64::new(0.0, 0.0); dim];
        for &x in &supp[..k] {
            psi[x] = Complex64::new(1.0 / (k as f64).sqrt(), 0.0);
        }
        let mut found = nullity_brute(&psi, n) > 0;
        let _ = trial;
        let mut tries = 0;
        while !found {
            tries += 1;
            assert!(tries < 2000, "no magic-creating circuit found for k = {k}");
            let mut s = psi.clone();
            for g in random_nct(n, 6, &mut rng) {
                s = apply_perm(&s, g);
                if nullity_brute(&s, n) > 0 {
                    found = true;
                    break;
                }
            }
        }
    }
}

// ---------------------------------------------------------------- T3

fn all_control_faults(round: u32) -> Vec<Fault> {
    let mut v = vec![
        Fault {
            round,
            site: Site::Prep,
            pauli: Pauli::X,
        },
        Fault {
            round,
            site: Site::Meas,
            pauli: Pauli::X,
        },
    ];
    for site in [Site::H1, Site::Phase, Site::H2] {
        for pauli in [Pauli::X, Pauli::Y, Pauli::Z] {
            v.push(Fault { round, site, pauli });
        }
    }
    v
}

fn random_gate_fault<R: Rng>(
    nc: &NoisyCircuit,
    round: usize,
    pauli: Option<Pauli>,
    rng: &mut R,
) -> Fault {
    let gates = &nc.rounds[round].gates;
    let g = rng.random_range(0..gates.len());
    let slot = rng.random_range(0..gates[g].qubits().len()) as u8;
    let pauli = pauli.unwrap_or([Pauli::X, Pauli::Y, Pauli::Z][rng.random_range(0..3)]);
    Fault {
        round: round as u32,
        site: Site::Gate {
            gate: g as u32,
            slot,
        },
        pauli,
    }
}

/// T3(a): any fault pattern confined to the last ν2(r) rounds leaves
/// P(peak success) exactly equal to the fault-free S0 (exact distributions);
/// and the window is sharp: some single fault in round t − ν − 1 changes it.
#[test]
fn t3_end_window_is_exactly_harmless() {
    let mut rng = StdRng::seed_from_u64(6);
    for &(n, a) in &[(15u64, 7u64), (21, 2), (35, 8), (51, 2)] {
        let inst = Instance::new(n, a, Oracle::Windowed(4));
        let nc = NoisyCircuit::new(&inst, NoiseKind::Depolarizing);
        let t = inst.t;
        let r = order(a, n);
        let nu = r.trailing_zeros() as usize;
        assert!(nu >= 1);
        let s0 = p_ok(&noisy::trajectory_distribution(&nc, &[]), r, t);
        let mut patterns: Vec<Vec<Fault>> = Vec::new();
        for i in t - nu..t {
            for f in all_control_faults(i as u32) {
                patterns.push(vec![f]);
            }
            for _ in 0..12 {
                patterns.push(vec![random_gate_fault(&nc, i, None, &mut rng)]);
            }
        }
        for _ in 0..6 {
            let i = rng.random_range(t - nu..t);
            let j = rng.random_range(t - nu..t);
            patterns.push(vec![
                random_gate_fault(&nc, i, None, &mut rng),
                random_gate_fault(&nc, j, None, &mut rng),
            ]);
        }
        for p in &patterns {
            let ok = p_ok(&noisy::trajectory_distribution(&nc, p), r, t);
            assert!(
                (ok - s0).abs() < 1e-12,
                "N={n} a={a} {p:?}: {ok} vs S0 {s0}"
            );
        }
        // sharpness
        let i = t - nu - 1;
        let changed = all_control_faults(i as u32)
            .into_iter()
            .any(|f| (p_ok(&noisy::trajectory_distribution(&nc, &[f]), r, t) - s0).abs() > 1e-6);
        assert!(changed, "N={n}: round t-ν-1 should not be in the window");
        // a readout flip of bit t-ν-1 moves yr/2^t by r_odd/2: maps good y to bad y, so P(ok) ≤ 1 − S0
        let f = Fault {
            round: i as u32,
            site: Site::Meas,
            pauli: Pauli::X,
        };
        let pf = p_ok(&noisy::trajectory_distribution(&nc, &[f]), r, t);
        assert!(
            pf <= 1.0 - s0 + 1e-12,
            "N={n} readout flip round {i}: P(ok) = {pf} > 1 - S0"
        );
        eprintln!(
            "T3a N={n} a={a} r={r} ν={nu}: {} patterns, all P(ok) = S0 = {s0:.6}",
            patterns.len()
        );
    }
}

/// Lower bound of Theorem 3(b) for a phase-type fault in round i:
/// Δ' = 2^{t-i-2}/r²; ≥ 1 − 1/(2(⌈Δ'⌉−3)) if ⌈Δ'⌉ ≥ 4, ≥ 8/π² if Δ' ≥ 1, else 0.
fn start_window_bound(t: usize, i: usize, r: u64) -> f64 {
    if i + 2 > t {
        return 0.0;
    }
    let dp = (2f64).powi((t - i - 2) as i32) / (r as f64 * r as f64);
    let c = dp.ceil();
    let mut b = 0.0;
    if dp >= 1.0 {
        b = 8.0 / (PI * PI);
    }
    if c >= 4.0 {
        b = f64::max(b, 1.0 - 1.0 / (2.0 * (c - 3.0)));
    }
    b
}

/// T3(b): phase-type faults (any Z on any oracle qubit, Z/Prep on the
/// control, readout flips) in round i succeed with probability ≥ the
/// start-window bound; X faults in the same rounds are not covered (and do
/// fail much more often).
#[test]
fn t3_start_window_lower_bound_for_phase_faults() {
    let mut rng = StdRng::seed_from_u64(7);
    let mut worst_margin = f64::INFINITY;
    let mut x_below = 0;
    for &(n, a) in &[(35u64, 11u64), (35, 8), (39, 22), (33, 10)] {
        let inst = Instance::new(n, a, Oracle::Windowed(4));
        let nc = NoisyCircuit::new(&inst, NoiseKind::Depolarizing);
        let t = inst.t;
        let r = order(a, n);
        for i in 0..t {
            let b = start_window_bound(t, i, r);
            if b == 0.0 {
                continue;
            }
            let mut pats: Vec<Fault> = vec![
                Fault {
                    round: i as u32,
                    site: Site::Prep,
                    pauli: Pauli::X,
                },
                Fault {
                    round: i as u32,
                    site: Site::Meas,
                    pauli: Pauli::X,
                },
                Fault {
                    round: i as u32,
                    site: Site::H1,
                    pauli: Pauli::Z,
                },
                Fault {
                    round: i as u32,
                    site: Site::Phase,
                    pauli: Pauli::Z,
                },
                Fault {
                    round: i as u32,
                    site: Site::H2,
                    pauli: Pauli::X,
                }, // X after H2 = readout flip
            ];
            for _ in 0..10 {
                pats.push(random_gate_fault(&nc, i, Some(Pauli::Z), &mut rng));
            }
            for f in pats {
                let ok = p_ok(&noisy::trajectory_distribution(&nc, &[f]), r, t);
                assert!(
                    ok >= b - 1e-12,
                    "N={n} a={a} r={r} round {i} {f:?}: {ok} < bound {b}"
                );
                worst_margin = worst_margin.min(ok - b);
            }
            for _ in 0..3 {
                let f = random_gate_fault(&nc, i, Some(Pauli::X), &mut rng);
                let ok = p_ok(&noisy::trajectory_distribution(&nc, &[f]), r, t);
                if ok < b {
                    x_below += 1;
                }
            }
        }
    }
    eprintln!("T3b: min (P(ok) − bound) over phase-type faults = {worst_margin:.4}; X faults below the bound: {x_below}");
    assert!(
        x_below > 0,
        "the hypothesis (phase-type fault) should matter"
    );
}

fn eval_prefix(gates: &[Gate], mut s: u128) -> u128 {
    for g in gates {
        match *g {
            Gate::X(q) => s ^= 1 << q,
            Gate::Cnot(c, t) => s ^= ((s >> c) & 1) << t,
            Gate::Ccx(a, b, t) => s ^= ((s >> a) & (s >> b) & 1) << t,
            Gate::Swap(a, b) => {
                let (x, y) = ((s >> a) & 1, (s >> b) & 1);
                s ^= (x ^ y) << a | (x ^ y) << b;
            }
            ref o => panic!("non-permutation gate {o:?}"),
        }
    }
    s
}

/// T3(c): exact sign formula. A Z fault on qubit q after gate g of round i
/// gives P(y) = Σ_z |2^{-t} Σ_{x: a^x = z} σ(x) e^{-2πi x y / 2^t}|², with
/// σ(x) = (−1)^{f(x_{t−1−i}, a^{2^{t−i}⌊x/2^{t−i}⌋})} and f(c, w) the value of
/// qubit q after gate g on input |c⟩|w⟩|0…0⟩.
#[test]
fn t3_phase_fault_textbook_formula() {
    let mut rng = StdRng::seed_from_u64(8);
    for &(n, a) in &[(15u64, 7u64), (15, 2), (21, 2)] {
        let inst = Instance::new(n, a, Oracle::Windowed(4));
        let nc = NoisyCircuit::new(&inst, NoiseKind::PhaseFlip);
        let t = inst.t;
        let tt = 1usize << t;
        for _ in 0..24 {
            let i = rng.random_range(0..t);
            let f = random_gate_fault(&nc, i, Some(Pauli::Z), &mut rng);
            let (g, q) = match f.site {
                Site::Gate { gate, slot } => {
                    let g = gate as usize;
                    (g, nc.rounds[i].gates[g].qubits()[slot as usize])
                }
                _ => unreachable!(),
            };
            let (c, io) = qsim_lab::shor::sliced::oracle_block(&inst, inst.mults[t - 1 - i]);
            assert_eq!(c.ops.len(), nc.rounds[i].gates.len());
            let prefix = &nc.rounds[i].gates[..=g];
            let fval = |ctrl: u128, w: u64| -> bool {
                let mut s = ctrl << io.ctrl;
                for (j, &xq) in io.x.iter().enumerate() {
                    s |= u128::from((w >> j) & 1) << xq;
                }
                (eval_prefix(prefix, s) >> q) & 1 == 1
            };
            // textbook amplitudes per z
            let mut by_z: std::collections::HashMap<u64, Vec<Complex64>> = Default::default();
            for x in 0..tt {
                let hi = (x >> (t - i)) << (t - i);
                let w = powmod(a, hi as u128, n);
                let ctrl = ((x >> (t - 1 - i)) & 1) as u128;
                let sgn = if fval(ctrl, w) { -1.0 } else { 1.0 };
                let z = powmod(a, x as u128, n);
                let e = by_z
                    .entry(z)
                    .or_insert_with(|| vec![Complex64::new(0.0, 0.0); tt]);
                e[x] += sgn;
            }
            let mut pt = vec![0.0; tt];
            for (_, amp) in by_z {
                for y in 0..tt {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for (x, v) in amp.iter().enumerate() {
                        if v.re != 0.0 {
                            acc += v * Complex64::from_polar(
                                1.0,
                                -2.0 * PI * ((x * y) % tt) as f64 / tt as f64,
                            );
                        }
                    }
                    pt[y] += (acc / tt as f64).norm_sqr();
                }
            }
            let pe = noisy::trajectory_distribution(&nc, &[f]);
            for y in 0..tt {
                assert!(
                    (pt[y] - pe[y]).abs() < 1e-10,
                    "N={n} a={a} {f:?} y={y}: {} vs {}",
                    pt[y],
                    pe[y]
                );
            }
        }
    }
}

/// T3(d): counting bound behind "X/Y faults are fatal": if bits i+1..t−ν−1
/// are i.i.d. uniform (control halves orthogonal in those rounds), then for
/// every recorded prefix y_{≤i} the success probability is ≤ 1/r + r_odd·2^{i+1+ν−t}.
#[test]
fn t3_dephasing_counting_bound() {
    for t in 6..=16usize {
        for r in 2..(1u64 << (t / 2)) {
            let nu = r.trailing_zeros() as usize;
            let ro = r >> nu;
            for i in 0..(t - nu).saturating_sub(1) {
                let m = t - nu - i - 1;
                let bound =
                    1.0 / r as f64 + ro as f64 * (2f64).powi(i as i32 + 1 + nu as i32 - t as i32);
                for ylo in 0..(1u128 << (i + 1)) {
                    let good = (0..(1u128 << m))
                        .filter(|&ym| peak_ok(ylo | ym << (i + 1), r, t))
                        .count();
                    let p = good as f64 / (1u64 << m) as f64;
                    assert!(
                        p <= bound + 1e-12,
                        "t={t} r={r} i={i} y={ylo}: {p} > {bound}"
                    );
                }
            }
        }
    }
}
