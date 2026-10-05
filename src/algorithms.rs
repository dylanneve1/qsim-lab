//! Textbook circuits and algorithms used by the examples, CLI and tests.

use crate::circuit::Circuit;
use crate::engines::blocked::{lower_gate, KOp};
use crate::engines::statevector::{Real, StateVector};
use rand::Rng;
use std::f64::consts::PI;

/// `(|00> + |11>)/√2`.
pub fn bell() -> Circuit {
    let mut c = Circuit::new(2);
    c.h(0).cnot(0, 1);
    c
}

/// `(|0...0> + |1...1>)/√2` with a CNOT chain (depth `n`).
pub fn ghz(n: usize) -> Circuit {
    let mut c = Circuit::new(n);
    c.h(0);
    for q in 1..n {
        c.cnot(q - 1, q);
    }
    c
}

/// A random brickwork circuit: each of the `depth` layers applies
/// `Ry(a) Rz(b)` with random angles to every qubit, then CNOTs on
/// neighbouring pairs `(q, q+1)` starting at `q = layer % 2`.
pub fn random_brickwork<R: Rng + ?Sized>(n: usize, depth: usize, rng: &mut R) -> Circuit {
    let mut c = Circuit::new(n);
    for layer in 0..depth {
        for q in 0..n {
            c.ry(q, rng.random::<f64>() * PI);
            c.rz(q, rng.random::<f64>() * PI);
        }
        for q in (layer % 2..n.saturating_sub(1)).step_by(2) {
            c.cnot(q, q + 1);
        }
    }
    c
}

/// Bernstein–Vazirani for an `n`-bit secret `s`, on `n + 1` qubits (the last
/// one is the phase-kickback ancilla). Measuring qubits `0..n` yields `s`
/// with certainty after a single oracle query.
pub fn bernstein_vazirani(n: usize, secret: u64) -> Circuit {
    let mut c = Circuit::new(n + 1);
    let anc = n;
    c.x(anc).h(anc);
    for q in 0..n {
        c.h(q);
    }
    // oracle |x>|y> -> |x>|y ⊕ s·x>
    for q in 0..n {
        if (secret >> q) & 1 == 1 {
            c.cnot(q, anc);
        }
    }
    for q in 0..n {
        c.h(q);
    }
    for q in 0..n {
        c.measure(q);
    }
    c
}

/// Quantum Fourier transform on qubits `0..n` with qubit 0 as the least
/// significant bit: `|x> -> 2^{-n/2} sum_k e^{2πi xk/2^n} |k>`. Includes
/// the final bit-reversal SWAPs.
pub fn qft(n: usize) -> Circuit {
    let mut c = Circuit::new(n);
    qft_into(&mut c, &(0..n).collect::<Vec<_>>(), false);
    c
}

/// Appends a QFT (or its inverse) on the given qubits (`qs[0]` = least
/// significant) to `c`.
pub fn qft_into(c: &mut Circuit, qs: &[usize], inverse: bool) {
    let n = qs.len();
    let mut sub = Circuit::new(c.num_qubits);
    for j in (0..n).rev() {
        sub.h(qs[j]);
        for k in (0..j).rev() {
            sub.cphase(qs[k], qs[j], PI / (1u64 << (j - k)) as f64);
        }
    }
    for j in 0..n / 2 {
        sub.swap(qs[j], qs[n - 1 - j]);
    }
    if inverse {
        sub = sub.inverse();
    }
    c.append(&sub);
}

/// Grover search for a single marked item among `2^n`, run on a state
/// vector. Returns the measured index and the success probability just
/// before measurement.
pub fn grover<T: Real, R: Rng + ?Sized>(n: usize, marked: usize, rng: &mut R) -> (usize, f64) {
    assert!(n >= 2 && marked < (1 << n));
    let iterations = ((PI / 4.0) * ((1u64 << n) as f64).sqrt()).floor() as usize;
    let s = grover_state::<T>(n, marked, iterations);
    let p = s.amplitude(marked).norm_sqr();
    let found = s.sample(1, rng)[0];
    (found, p)
}

/// The state after the initial Hadamards and `iterations` Grover
/// iterations (oracle marking `marked`, then the diffuser), applied gate by
/// gate with the multi-controlled Z as one diagonal op.
pub fn grover_state<T: Real>(n: usize, marked: usize, iterations: usize) -> StateVector<T> {
    let all: Vec<usize> = (0..n).collect();
    let mut s = StateVector::<T>::new(n);
    let had = |s: &mut StateVector<T>| {
        for q in 0..n {
            s.apply_gate(&crate::Gate::H(q)).expect("valid");
        }
    };
    let flip_zeros = |s: &mut StateVector<T>, pattern: usize| {
        for q in 0..n {
            if (pattern >> q) & 1 == 0 {
                s.apply_gate(&crate::Gate::X(q)).expect("valid");
            }
        }
    };
    had(&mut s);
    for _ in 0..iterations {
        // oracle: phase flip on |marked>
        flip_zeros(&mut s, marked);
        s.apply_mcz(&all);
        flip_zeros(&mut s, marked);
        // diffuser: 2|s><s| - I  (up to global phase)
        had(&mut s);
        flip_zeros(&mut s, 0);
        s.apply_mcz(&all);
        flip_zeros(&mut s, 0);
        had(&mut s);
    }
    s
}

/// [`grover_state`] as executor IR for the cache-blocked executor
/// (`StateVector::apply_kops_blocked`), op for op: the multi-controlled Z
/// on all `n` qubits is a single diagonal [`KOp::Phase`] term.
pub fn grover_kops(n: usize, marked: usize, iterations: usize) -> Vec<KOp> {
    let all = (1usize << n) - 1;
    let mut ops: Vec<KOp> = Vec::new();
    let mcz = KOp::Phase {
        mask: all,
        pat: all,
        f: num_complex::Complex64::new(-1.0, 0.0),
    };
    let had = |ops: &mut Vec<KOp>| {
        for q in 0..n {
            lower_gate(&crate::Gate::H(q), ops);
        }
    };
    let flip_zeros = |ops: &mut Vec<KOp>, pattern: usize| {
        for q in 0..n {
            if (pattern >> q) & 1 == 0 {
                lower_gate(&crate::Gate::X(q), ops);
            }
        }
    };
    had(&mut ops);
    for _ in 0..iterations {
        flip_zeros(&mut ops, marked);
        ops.push(mcz);
        flip_zeros(&mut ops, marked);
        had(&mut ops);
        flip_zeros(&mut ops, 0);
        ops.push(mcz);
        flip_zeros(&mut ops, 0);
        had(&mut ops);
    }
    ops
}

/// Greatest common divisor.
pub fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// `b^e mod m` (products in `u128`, so any `u64` modulus works).
pub fn pow_mod(mut b: u64, mut e: u64, m: u64) -> u64 {
    let mul = |x: u64, y: u64| (u128::from(x) * u128::from(y) % u128::from(m)) as u64;
    let mut r = 1 % m;
    b %= m;
    while e > 0 {
        if e & 1 == 1 {
            r = mul(r, b);
        }
        b = mul(b, b);
        e >>= 1;
    }
    r
}

/// Denominators of the continued-fraction convergents of `x / 2^t`.
pub fn convergent_denominators(x: u64, t: u32) -> Vec<u64> {
    let (mut num, mut den) = (x, 1u64 << t);
    let (mut q_prev, mut q) = (1u64, 0u64);
    let mut out = Vec::new();
    while den != 0 {
        let a = num / den;
        let q_next = a * q + q_prev;
        q_prev = q;
        q = q_next;
        out.push(q);
        let r = num % den;
        num = den;
        den = r;
    }
    out
}

/// Result of one run of Shor's order-finding routine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShorRun {
    /// Base `a` (coprime to `N`) whose order modulo `N` is sought.
    pub a: u64,
    /// Value read from the counting register.
    pub measured: u64,
    /// Order found classically from `measured`, if any.
    pub order: Option<u64>,
    /// A non-trivial factor, if this run produced one.
    pub factor: Option<u64>,
    /// Total qubits simulated: counting plus work register (`3⌈log2 N⌉`).
    pub qubits: usize,
}

/// The textbook order-finding state just before measurement, for `N` with
/// base `a`: `t = 2⌈log2 N⌉` counting qubits (qubits `0..t`, qubit `j`
/// controls `U^(2^j)`), `⌈log2 N⌉` work qubits, controlled modular
/// multiplications as permutation oracles, and an inverse QFT built from
/// H/controlled-phase/SWAP gates. Returns the state and `t`.
pub fn shor_full_state(n_mod: u64, a: u64) -> (StateVector<f64>, usize) {
    assert!(n_mod >= 3 && gcd(a, n_mod) == 1);
    let m = 64 - (n_mod - 1).leading_zeros() as usize; // work qubits
    let t = 2 * m; // counting qubits
    let total = t + m;
    let mut s = StateVector::<f64>::new(total);
    // work register (qubits t..t+m) starts in |1>
    s.apply_gate(&crate::Gate::X(t)).expect("valid");
    for q in 0..t {
        s.apply_gate(&crate::Gate::H(q)).expect("valid");
    }
    let work_mask = ((1usize << m) - 1) << t;
    for j in 0..t {
        // controlled-U^{2^j}: |c=1>|y> -> |c=1>|a^{2^j} y mod N> for y < N
        let mult = pow_mod(a, 1 << j, n_mod) as usize;
        let n = n_mod as usize;
        s.apply_permutation(|i| {
            if (i >> j) & 1 == 0 {
                return i;
            }
            let y = (i & work_mask) >> t;
            if y >= n {
                return i;
            }
            (i & !work_mask) | (((y * mult) % n) << t)
        });
    }
    let mut c = Circuit::new(total);
    qft_into(&mut c, &(0..t).collect::<Vec<_>>(), true);
    s.apply_circuit(&c).expect("valid");
    (s, t)
}

/// Classical post-processing of a measured `t`-bit value: the order from
/// the continued-fraction convergents of `measured / 2^t`, then a factor
/// from `gcd(a^(r/2) ± 1, N)`.
pub fn shor_postprocess(n_mod: u64, a: u64, measured: u64, t: u32) -> (Option<u64>, Option<u64>) {
    let order = convergent_denominators(measured, t)
        .into_iter()
        .find(|&r| r > 0 && r < n_mod && pow_mod(a, r, n_mod) == 1);
    let factor = order.and_then(|r| {
        if r % 2 == 1 {
            return None;
        }
        let y = pow_mod(a, r / 2, n_mod);
        [gcd(y + 1, n_mod), gcd(y + n_mod - 1, n_mod)]
            .into_iter()
            .find(|&f| f > 1 && f < n_mod)
    });
    (order, factor)
}

/// One run of Shor's algorithm for `N` with base `a` on a state vector
/// ([`shor_full_state`], `3⌈log2 N⌉` qubits), sampled once and
/// post-processed with continued fractions.
pub fn shor_order_finding<R: Rng + ?Sized>(n_mod: u64, a: u64, rng: &mut R) -> ShorRun {
    let (s, t) = shor_full_state(n_mod, a);
    let total = s.num_qubits();
    let shot = s.sample(1, rng)[0];
    let measured = (shot & ((1 << t) - 1)) as u64;
    let (order, factor) = shor_postprocess(n_mod, a, measured, t as u32);
    ShorRun {
        a,
        measured,
        order,
        factor,
        qubits: total,
    }
}

/// Factors a small odd composite `N` (not a prime power) with repeated
/// runs of [`shor_order_finding`]. Returns the factor pair and the runs.
pub fn shor_factor<R: Rng + ?Sized>(n_mod: u64, rng: &mut R) -> (Option<(u64, u64)>, Vec<ShorRun>) {
    let mut runs = Vec::new();
    for _ in 0..20 {
        let a = rng.random_range(2..n_mod - 1);
        let g = gcd(a, n_mod);
        if g > 1 {
            // lucky classical guess; skip it so the quantum part is exercised
            continue;
        }
        let run = shor_order_finding(n_mod, a, rng);
        let f = run.factor;
        runs.push(run);
        if let Some(f) = f {
            return (Some((f.min(n_mod / f), f.max(n_mod / f))), runs);
        }
    }
    (None, runs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::statevector::StateVectorF64;
    use num_complex::Complex64;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn qft_of_basis_state() {
        let n = 5;
        for x in [0usize, 1, 7, 19] {
            let mut s = StateVectorF64::basis_state(n, x);
            s.apply_circuit(&qft(n)).unwrap();
            let norm = 1.0 / ((1 << n) as f64).sqrt();
            for k in 0..1 << n {
                let want = Complex64::from_polar(norm, 2.0 * PI * (x * k) as f64 / (1 << n) as f64);
                assert!((s.amplitude(k) - want).norm() < 1e-10, "x={x} k={k}");
            }
        }
    }

    #[test]
    fn convergents() {
        // 3/8 = [0; 2, 1, 2] -> denominators 1, 2, 3, 8
        assert_eq!(convergent_denominators(3, 3), vec![1, 2, 3, 8]);
    }

    #[test]
    fn shor_finds_order_of_7_mod_15() {
        let mut rng = StdRng::seed_from_u64(2);
        let mut found = false;
        for _ in 0..10 {
            let run = shor_order_finding(15, 7, &mut rng);
            assert_eq!(run.qubits, 12);
            // measured/256 must be close to k/4
            assert_eq!(run.measured % 64, 0);
            if run.order == Some(4) {
                assert_eq!(run.factor.map(|f| 15 % f), Some(0));
                found = true;
            }
        }
        assert!(found);
    }
}
