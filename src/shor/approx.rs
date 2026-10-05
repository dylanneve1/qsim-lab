//! Exact simulation of the approximate modular exponentiation of Gidney 2025
//! ("How to factor 2048 bit RSA integers with less than a million noisy
//! qubits", arXiv:2505.15917; code CC-BY-4.0, Zenodo 10.5281/zenodo.15347487),
//! which builds on the approximate residue arithmetic of
//! Chevignard–Fouque–Schrottenloher 2024 (`research/shor/approx-modexp.md`).
//!
//! The paper verifies its construction with a simulator that follows "a few
//! randomly sampled classical trajectories" and says that this cannot verify
//! interference or masking. Here every branch is simulated:
//!
//! * [`ApproxConfig::new`] is a port of the paper's precomputation
//!   (`facto/algorithm/prep`): multiplier windows, the residue number system
//!   search (prime set `P` with `∏P ≥ N^{W1}` and `∏P mod N < N >> gap`, no
//!   prime dividing a multiplier), primitive roots, and the tables of the four
//!   loops (discrete-log differences, windowed modular exponentiation, truncated
//!   residue contributions).
//! * [`plan`] resolves the quint-level program of `approx_modexp` (loop1, loop2,
//!   loop3, loop4, unloop3, unloop2 of `_detailed_example_code.py`) for one
//!   stream of measurement outcomes: every X-basis measurement (lookup outputs,
//!   wrap-around qubits, whole helper registers) gets its sampled outcome, every
//!   deferred phase correction ("vent" tables, pushed/popped uncompute info) is
//!   resolved into a concrete phase lookup, comparison or CZ, exactly as the
//!   paper's code orders its random draws.
//! * [`evaluate`] runs the resolved program on **every** branch `(e, s)` of the
//!   exponent register and the mask: every register value and every sign. It
//!   checks that each ancilla returns to 0, that every branch ends with sign +1
//!   (all phase kickback from measurement-based uncomputation corrected), that
//!   each residue register holds the exact residue, and that the accumulator is
//!   `(s + F̃(e)) mod T`. The X-basis measurements are exact in this
//!   representation because each measured register is a function of the
//!   remaining registers (`e` and the accumulator determine the branch), so no
//!   two branches merge and every outcome has probability `2^{-len}`; the
//!   independent sparse-state-vector check in `research/data/approx-modexp/`
//!   does not assume this.
//! * [`distribution`] turns `F̃` into the exact distribution of the
//!   frequency-basis measurement (QFT of the exponent register after measuring
//!   the accumulator), for one register (Shor-style `g^e`) or two
//!   (Ekerå–Håstad `g^a y^{-b}`).
//! * [`overlap`], [`best_shift`] and [`cond_fidelity`] give the exact fidelity
//!   of the pre-measurement states and of the post-measurement exponent states
//!   with exact arithmetic; [`paper_success`] is the success test of the
//!   paper's own success-rate model.
//! * [`gate_loop4_step`] and [`gate_loop3_pair`] compile one loop4 window step,
//!   and one loop3 step with its unloop3 counterpart, to X/CNOT/Toffoli gates
//!   (plus X-basis measurements and Z/CZ phase fix-ups) with the repo's
//!   measurement-based building blocks, for the gate-level check.

#![allow(clippy::needless_range_loop)]

use crate::algorithms::{gcd, pow_mod};
use crate::shor::mod_inverse;
use num_complex::Complex64;
use rayon::prelude::*;
use std::fmt;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors of the precomputation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApproxError {
    /// A parameter combination is outside what the construction supports.
    Params(String),
    /// The residue-number-system search found no admissible prime set.
    NoRns(String),
}

impl fmt::Display for ApproxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ApproxError::Params(s) => write!(f, "approx-modexp parameters: {s}"),
            ApproxError::NoRns(s) => write!(f, "approx-modexp residue system: {s}"),
        }
    }
}

impl std::error::Error for ApproxError {}

// ---------------------------------------------------------------------------
// Small number theory
// ---------------------------------------------------------------------------

/// Bit length of `x` (Python `int.bit_length`).
pub fn bit_len(x: u64) -> usize {
    64 - x.leading_zeros() as usize
}

fn mask(len: usize) -> u64 {
    if len >= 64 {
        u64::MAX
    } else {
        (1u64 << len) - 1
    }
}

fn parity(x: u64) -> bool {
    x.count_ones() & 1 == 1
}

/// Primes in `[lo, hi)` (simple sieve; `hi` small).
pub fn primes_in(lo: u64, hi: u64) -> Vec<u64> {
    let hi = hi as usize;
    let mut s = vec![true; hi.max(2)];
    s[0] = false;
    if hi > 1 {
        s[1] = false;
    }
    let mut i = 2;
    while i * i < hi {
        if s[i] {
            let mut j = i * i;
            while j < hi {
                s[j] = false;
                j += i;
            }
        }
        i += 1;
    }
    (lo as usize..hi)
        .filter(|&k| s[k])
        .map(|k| k as u64)
        .collect()
}

fn prime_factors(mut x: u64) -> Vec<u64> {
    let mut f = Vec::new();
    let mut d = 2;
    while d * d <= x {
        if x % d == 0 {
            f.push(d);
            while x % d == 0 {
                x /= d;
            }
        }
        d += 1;
    }
    if x > 1 {
        f.push(x);
    }
    f
}

/// The paper's generator choice: the smallest `i ≥ 3` whose order mod the
/// prime `p` is `p − 1` (`find_multiplicative_generator_modulo_prime_number`).
pub fn paper_generator(p: u64) -> u64 {
    let fs = prime_factors(p - 1);
    (3..p)
        .find(|&i| fs.iter().all(|&q| pow_mod(i, (p - 1) / q, p) != 1))
        .expect("prime has a generator")
}

/// Minimal unsigned big integer (little-endian `u32` limbs) for the size
/// check `∏P ≥ N^{W1}`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Big(Vec<u32>);

impl Big {
    fn one() -> Self {
        Big(vec![1])
    }
    fn mul_small(&mut self, m: u64) {
        assert!(m < 1 << 32);
        let mut carry = 0u64;
        for l in self.0.iter_mut() {
            let v = u64::from(*l) * m + carry;
            *l = v as u32;
            carry = v >> 32;
        }
        while carry > 0 {
            self.0.push(carry as u32);
            carry >>= 32;
        }
    }
    fn mul_u64(&mut self, m: u64) {
        self.mul_small(m & 0xFFFF_FFFF);
        let hi = m >> 32;
        if hi > 0 {
            // (lo + hi·2^32): recompute from scratch is simpler
            panic!("Big::mul_u64 only supports 32-bit factors");
        }
    }
    fn trim(&mut self) {
        while self.0.len() > 1 && *self.0.last().unwrap() == 0 {
            self.0.pop();
        }
    }
    fn ge(&self, o: &Big) -> bool {
        let (mut a, mut b) = (self.clone(), o.clone());
        a.trim();
        b.trim();
        if a.0.len() != b.0.len() {
            return a.0.len() > b.0.len();
        }
        for i in (0..a.0.len()).rev() {
            if a.0[i] != b.0[i] {
                return a.0[i] > b.0[i];
            }
        }
        true
    }
    fn bits(&self) -> usize {
        let mut a = self.clone();
        a.trim();
        let top = *a.0.last().unwrap();
        (a.0.len() - 1) * 32 + (32 - top.leading_zeros() as usize)
    }
}

// ---------------------------------------------------------------------------
// Parameters and precomputation
// ---------------------------------------------------------------------------

/// Parameters of one approximate modular exponentiation (the paper's
/// `ProblemConfig`, with the exponent split into registers so that
/// Ekerå–Håstad's `g^a y^{-b}` is expressible).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApproxParams {
    /// The number to factor `N` (`< 2^32`).
    pub n_mod: u64,
    /// Exponent registers, low bits first: `(qubits, base)`. Shor-style:
    /// `[(m, g)]`; Ekerå–Håstad: `[(2m, g), (m, y^{-1})]`.
    pub regs: Vec<(usize, u64)>,
    /// The classical base `g` (for reporting and post-processing).
    pub generator: u64,
    /// Window over the exponent register in loop1 (`window1`).
    pub w1: usize,
    /// Window over the discrete-log register in loop3 (`window3a`).
    pub w3a: usize,
    /// Window over the residue register in loop3 (`window3b`).
    pub w3b: usize,
    /// Window over the residue register in loop4 (`window4`).
    pub w4: usize,
    /// Kept accumulator bits `f` (`len_accumulator`).
    pub len_acc: usize,
    /// Mask width `2^{mask_bits}` in accumulator units (`mask_bits`).
    pub mask_bits: usize,
    /// `min_wraparound_gap`: the residue system must satisfy
    /// `∏P mod N < N >> gap`.
    pub min_gap: usize,
    /// Prime bit length `ℓ` (`None`: the paper's estimate).
    pub prime_bits: Option<usize>,
    /// The paper's search refuses fewer than 100 candidate primes; kept as a
    /// parameter so tiny instances can lower it (documented where used).
    pub min_search_primes: usize,
    /// Use this prime set instead of searching (it must pass the same
    /// admissibility checks: `ℓ`-bit, distinct, no multiplier divisible,
    /// `∏P ≥ N^{W1}`, `∏P mod N < N >> gap`).
    pub forced_periods: Option<Vec<u64>>,
}

impl ApproxParams {
    /// Shor-style parameters: one `m`-qubit exponent register with base `g`.
    #[allow(clippy::too_many_arguments)]
    pub fn shor(
        n_mod: u64,
        g: u64,
        m: usize,
        w: [usize; 4],
        len_acc: usize,
        mask_bits: usize,
    ) -> Self {
        ApproxParams {
            n_mod,
            regs: vec![(m, g)],
            generator: g,
            w1: w[0],
            w3a: w[1],
            w3b: w[2],
            w4: w[3],
            len_acc,
            mask_bits,
            min_gap: len_acc,
            prime_bits: None,
            min_search_primes: 100,
            forced_periods: None,
        }
    }

    /// Ekerå–Håstad parameters (`s = 1`, the convention of
    /// [`crate::shor::ge::eh_regs`]): registers `a` (`2m` qubits, base `g`) and
    /// `b` (`m` qubits, base `y^{-1}`, `y = g^{(N−1)/2}`), `m = ⌈n/2⌉`.
    pub fn eh(n_mod: u64, g: u64, w: [usize; 4], len_acc: usize, mask_bits: usize) -> Self {
        let m = crate::shor::ge::eh_m(n_mod);
        let y = crate::shor::ge::eh_target(n_mod, g);
        let yi = mod_inverse(y, n_mod);
        ApproxParams {
            n_mod,
            regs: vec![(2 * m, g), (m, yi)],
            generator: g,
            w1: w[0],
            w3a: w[1],
            w3b: w[2],
            w4: w[3],
            len_acc,
            mask_bits,
            min_gap: len_acc,
            prime_bits: None,
            min_search_primes: 100,
            forced_periods: None,
        }
    }

    /// Ekerå–Håstad with tradeoff parameter `s` (same convention as
    /// [`ApproxParams::eh`]): `d = (p + q − 2)/2 < 2^m`, `m = ⌈n/2⌉`,
    /// `ℓ = ⌈m/s⌉`; registers `a` (`m + ℓ` qubits, base `g`) and `b`
    /// (`ℓ` qubits, base `y^{-1}`). `s = 1` is [`ApproxParams::eh`]. The
    /// classical post-processing for `s > 1` needs `≥ s + 1` runs.
    pub fn eh_s(
        n_mod: u64,
        g: u64,
        s: usize,
        w: [usize; 4],
        len_acc: usize,
        mask_bits: usize,
    ) -> Self {
        let mut p = Self::eh(n_mod, g, w, len_acc, mask_bits);
        let m = crate::shor::ge::eh_m(n_mod);
        let l = m.div_ceil(s.max(1));
        p.regs[0].0 = m + l;
        p.regs[1].0 = l;
        p
    }

    /// Total exponent qubits `m`.
    pub fn num_input_qubits(&self) -> usize {
        self.regs.iter().map(|r| r.0).sum()
    }
}

/// The paper's `estimated_ideal_mask_bits`: `S = √ε` with
/// `ε = additions · 3 · 2^{−f}`.
pub fn paper_mask_bits(num_periods: usize, nw4: usize, len_acc: usize) -> usize {
    let additions = (nw4 * num_periods) as f64;
    let eps = additions * 3.0 * 2f64.powi(-(len_acc as i32));
    (eps.log2() / 2.0 + len_acc as f64).round().max(0.0) as usize
}

/// The paper's prime-count table `prime_count_and_capacity_at_bit_length`
/// (count, capacity in bits) for `ℓ ≤ 24`, else the prime-number-theorem
/// estimate.
fn prime_stats(bits: usize) -> (u64, u64) {
    const T: [(u64, u64); 25] = [
        (0, 0),
        (0, 0),
        (2, 2),
        (2, 5),
        (2, 7),
        (5, 22),
        (7, 39),
        (13, 84),
        (23, 173),
        (43, 367),
        (75, 716),
        (137, 1444),
        (255, 2945),
        (464, 5823),
        (872, 11817),
        (1612, 23457),
        (3030, 47117),
        (5709, 94496),
        (10749, 188670),
        (20390, 378296),
        (38635, 755437),
        (73586, 1512435),
        (140336, 3024742),
        (268216, 6049260),
        (513708, 12099777),
    ];
    if bits < T.len() {
        return T[bits];
    }
    let n1 = 2f64.powi(bits as i32);
    let n2 = 2f64.powi(bits as i32 - 1);
    (
        (n1 / n1.ln() - n2 / n2.ln()).ceil() as u64,
        (n1 / 4f64.ln()).ceil() as u64,
    )
}

/// Port of `ProblemConfig.estimate_minimum_rns_period_bit_length`.
pub fn estimate_prime_bits(n_mod: u64, nw1: usize) -> usize {
    let max_product_bits = (bit_len(n_mod) * nw1) as f64;
    let mut est = (1.1 * (max_product_bits * 4f64.ln()).log2()).ceil() as usize;
    while prime_stats(est).1 as f64 >= max_product_bits {
        est -= 1;
    }
    while (prime_stats(est).1 as f64) < (max_product_bits + est as f64 * 100.0) * 1.1 {
        est += 1;
    }
    est.max(8)
}

/// Everything precomputed for one run (the paper's `ExecutionConfig`).
#[derive(Clone, Debug)]
pub struct ApproxConfig {
    /// The parameters this was built from.
    pub params: ApproxParams,
    /// Exponent qubits `m`.
    pub m: usize,
    /// loop1 windows over the exponent register: `(bit offset, width)`.
    pub windows1: Vec<(usize, usize)>,
    /// Window multipliers `M[j][k] = base_j^{k}` (`< N`), `2^{w1}` per window.
    pub mults: Vec<Vec<u64>>,
    /// Prime bit length `ℓ`.
    pub ell: usize,
    /// The residue system `P` (ascending).
    pub periods: Vec<u64>,
    /// Primitive root used for each prime.
    pub generators: Vec<u64>,
    /// `table1[i][j][k]`: discrete-log differences (`dlog_i − dlog_{i−1}`,
    /// two's complement, as the paper's `uint32` table), `i = 0..=|P|`.
    pub table1: Vec<Vec<Vec<u64>>>,
    /// `table3a[i][ja][jb][addr]` (negated products, subtracted by loop3).
    pub table3a: Vec<Vec<Vec<Vec<u64>>>>,
    /// `table3b[i][ja][jb][addr]` (inverse products, used by unloop3).
    pub table3b: Vec<Vec<Vec<Vec<u64>>>>,
    /// `table3c[i][k] = g_i^k mod p_i`, `k < 2^{2 w3a}`.
    pub table3c: Vec<Vec<u64>>,
    /// `table4[i][j][k]`: negated truncated contributions (subtracted by loop4).
    pub table4: Vec<Vec<Vec<u64>>>,
    /// Discrete-log accumulator length `ℓ + bitlen(m)`.
    pub len_dlog: usize,
    /// Dropped low bits `t = max(0, n − f)`.
    pub dropped: usize,
    /// Truncated modulus `T = N >> t`.
    pub trunc: u64,
    /// Number of loop1 windows.
    pub nw1: usize,
    /// Number of loop3 discrete-log windows.
    pub nw3a: usize,
    /// Number of loop3 residue windows.
    pub nw3b: usize,
    /// Number of loop4 windows.
    pub nw4: usize,
    /// `L mod N` (`L = ∏P`).
    pub l_mod_n: u64,
    /// Bit length of `L`.
    pub l_bits: usize,
}

impl ApproxConfig {
    /// Runs the precomputation (port of `ExecutionConfig.from_problem_config`).
    pub fn new(p: &ApproxParams) -> Result<Self, ApproxError> {
        let n_mod = p.n_mod;
        if n_mod >= 1 << 32 || n_mod < 15 {
            return Err(ApproxError::Params(format!(
                "N = {n_mod} outside [15, 2^32)"
            )));
        }
        if p.regs.is_empty() || p.regs.iter().any(|r| r.0 == 0) {
            return Err(ApproxError::Params("empty exponent register".into()));
        }
        let m = p.num_input_qubits();
        if m > 40 {
            return Err(ApproxError::Params(format!("m = {m} > 40")));
        }
        if p.w1 == 0 || p.w3a == 0 || p.w3b == 0 || p.w4 == 0 || p.w1 > 8 {
            return Err(ApproxError::Params("windows must be in 1..=8".into()));
        }
        if p.len_acc < 2 || p.len_acc > 31 || p.mask_bits >= p.len_acc {
            return Err(ApproxError::Params(format!(
                "need 2 <= len_acc <= 31 and mask_bits < len_acc (got {}, {})",
                p.len_acc, p.mask_bits
            )));
        }
        // loop1 windows, per register (the paper: one register, ceil(m/w1)).
        let mut windows1 = Vec::new();
        let mut mults = Vec::new();
        let mut off = 0;
        for &(len, base) in &p.regs {
            let mut start = 0;
            while start < len {
                let width = p.w1.min(len - start);
                windows1.push((off + start, width));
                let b = pow_mod(base % n_mod, 1u64 << start.min(63), n_mod);
                // b = base^(2^start): repeated squaring for large starts
                let b = if start < 63 {
                    b
                } else {
                    let mut x = base % n_mod;
                    for _ in 0..start {
                        x = crate::shor::mul_mod(x, x, n_mod);
                    }
                    x
                };
                mults.push((0..1u64 << p.w1).map(|k| pow_mod(b, k, n_mod)).collect());
                start += p.w1;
            }
            off += len;
        }
        let nw1 = windows1.len();
        let ell = p
            .prime_bits
            .unwrap_or_else(|| estimate_prime_bits(n_mod, nw1));
        if !(4..=20).contains(&ell) {
            return Err(ApproxError::Params(format!(
                "prime bit length {ell} outside 4..=20"
            )));
        }
        let len_dlog = ell + bit_len(m as u64);
        if len_dlog > 40 {
            return Err(ApproxError::Params("discrete-log register too long".into()));
        }
        let periods = match &p.forced_periods {
            Some(f) => check_rns(p, &mults, ell, nw1, f)?,
            None => find_rns(p, &mults, ell, nw1)?,
        };
        let generators: Vec<u64> = periods.iter().map(|&q| paper_generator(q)).collect();
        let np = periods.len();
        let w1n = 1usize << p.w1;
        // table1: dlogs, then differences along i (uint32-style wrapping)
        let mut dl = vec![vec![vec![0i64; w1n]; nw1]; np + 1];
        for (i, (&q, &gq)) in periods.iter().zip(&generators).enumerate() {
            let mut log = vec![u64::MAX; q as usize];
            let mut acc = 1u64;
            for k in 0..q - 1 {
                log[acc as usize] = k;
                acc = acc * gq % q;
            }
            for j in 0..nw1 {
                for k in 0..w1n {
                    let v = mults[j][k] % q;
                    assert!(v != 0, "prime {q} divides a multiplier");
                    dl[i][j][k] = log[v as usize] as i64;
                }
            }
        }
        let mut table1 = vec![vec![vec![0u64; w1n]; nw1]; np + 1];
        for i in 0..=np {
            for j in 0..nw1 {
                for k in 0..w1n {
                    let prev = if i == 0 { 0 } else { dl[i - 1][j][k] };
                    table1[i][j][k] = (dl[i][j][k] - prev) as u64;
                }
            }
        }
        let nw3a = ell.div_ceil(p.w3a);
        let nw3b = ell.div_ceil(p.w3b);
        let nw4 = ell.div_ceil(p.w4);
        let a3 = 1usize << (p.w3a + p.w3b);
        let mut table3a = Vec::with_capacity(np);
        let mut table3b = Vec::with_capacity(np);
        let mut table3c = Vec::with_capacity(np);
        for (&q, &gq) in periods.iter().zip(&generators) {
            let mut ta = vec![vec![vec![0u64; a3]; nw3b]; nw3a];
            let mut tb = vec![vec![vec![0u64; a3]; nw3b]; nw3a];
            for (ka, (ra, rb)) in ta.iter_mut().zip(tb.iter_mut()).enumerate() {
                for kb in 0..nw3b {
                    for k4a in 0..1u64 << p.w3a {
                        // v_base = g^(k4a << (ka*w3a)) mod q
                        let e = k4a << (ka * p.w3a);
                        let v_base = pow_mod(gq, e % (q - 1), q);
                        let v2_base = (q - mod_inverse(v_base, q)) % q;
                        let sh = pow_mod(2, (kb * p.w3b) as u64, q);
                        let v_base = v_base * sh % q;
                        let v2_base = v2_base * sh % q;
                        for k4b in 0..1u64 << p.w3b {
                            let addr = ((k4a << p.w3b) | k4b) as usize;
                            ra[kb][addr] = (q - v_base * k4b % q) % q;
                            rb[kb][addr] = (q - v2_base * k4b % q) % q;
                        }
                    }
                }
            }
            let mut tc = vec![0u64; 1 << (2 * p.w3a)];
            let mut acc = 1u64;
            for v in tc.iter_mut() {
                *v = acc;
                acc = acc * gq % q;
            }
            table3a.push(ta);
            table3b.push(tb);
            table3c.push(tc);
        }
        // table4: x = (L/p_i)·((inv_i · k · 2^{j w4}) mod p_i) < L, then mod N, >> t
        let dropped = bit_len(n_mod).saturating_sub(p.len_acc);
        let trunc = n_mod >> dropped;
        let mut table4 = Vec::with_capacity(np);
        for (i, &q) in periods.iter().enumerate() {
            let mut lq_mod_n = 1u64;
            let mut lq_mod_q = 1u64;
            for (i2, &q2) in periods.iter().enumerate() {
                if i2 != i {
                    lq_mod_n = crate::shor::mul_mod(lq_mod_n, q2, n_mod);
                    lq_mod_q = lq_mod_q * (q2 % q) % q;
                }
            }
            let inv = mod_inverse(lq_mod_q, q);
            let mut t4 = vec![vec![0u64; 1 << p.w4]; nw4];
            for (j, row) in t4.iter_mut().enumerate() {
                let sh = pow_mod(2, (j * p.w4) as u64, q);
                for (k, out) in row.iter_mut().enumerate() {
                    let c = inv * sh % q * (k as u64 % q) % q;
                    let x_mod_n = crate::shor::mul_mod(lq_mod_n, c, n_mod);
                    let w = x_mod_n >> dropped;
                    *out = (trunc - w % trunc) % trunc;
                }
            }
            table4.push(t4);
        }
        let mut l_mod_n = 1u64;
        let mut big = Big::one();
        for &q in &periods {
            l_mod_n = crate::shor::mul_mod(l_mod_n, q, n_mod);
            big.mul_u64(q);
        }
        Ok(ApproxConfig {
            params: p.clone(),
            m,
            windows1,
            mults,
            ell,
            periods,
            generators,
            table1,
            table3a,
            table3b,
            table3c,
            table4,
            len_dlog,
            dropped,
            trunc,
            nw1,
            nw3a,
            nw3b,
            nw4,
            l_mod_n,
            l_bits: big.bits(),
        })
    }

    /// Exact `f(e) = ∏_j M[j][e_j] mod N` (`g^e`, or `g^a y^{-b}`).
    pub fn exact(&self, e: u64) -> u64 {
        let n = self.params.n_mod;
        let mut r = 1 % n;
        for (j, &(off, w)) in self.windows1.iter().enumerate() {
            let k = ((e >> off) & mask(w)) as usize;
            r = crate::shor::mul_mod(r, self.mults[j][k], n);
        }
        r
    }

    /// The idealised truncated value `⌊f(e) / 2^t⌋ mod T` (exact arithmetic;
    /// `⌊f/2^t⌋ = T` is possible when the low `t` bits of `N` are non-zero).
    pub fn ideal_trunc(&self, e: u64) -> u64 {
        (self.exact(e) >> self.dropped) % self.trunc
    }

    /// `F̃(e)` from the tables alone (the paper's Eq. 20 with windows):
    /// `Σ_i Σ_j C[i][j][window_j(r_i(e))] mod T`, `r_i(e) = f_L(e) mod p_i`.
    pub fn table_formula(&self, e: u64) -> u64 {
        let t = self.trunc;
        let mut acc = 0u64;
        for (i, &q) in self.periods.iter().enumerate() {
            let mut r = 1u64;
            for (j, &(off, w)) in self.windows1.iter().enumerate() {
                let k = ((e >> off) & mask(w)) as usize;
                r = r * (self.mults[j][k] % q) % q;
            }
            for j in 0..self.nw4 {
                let k = ((r >> (j * self.params.w4)) & mask(self.params.w4)) as usize;
                acc = (acc + (t - self.table4[i][j][k]) % t) % t;
            }
        }
        acc
    }

    /// Number of loop4 additions into the accumulator (`|P|·⌈ℓ/w4⌉`).
    pub fn accumulator_additions(&self) -> usize {
        self.periods.len() * self.nw4
    }

    /// The paper's per-shot deviation model (`probability_of_deviation_failure`):
    /// returns `(ε, S, S + ε/S)` with `ε = additions·(2·2^{−f} + 2^{−gap})` and
    /// `S = 2^{mask − f}`.
    pub fn paper_deviation_model(&self) -> (f64, f64, f64) {
        let p = &self.params;
        let eps = self.accumulator_additions() as f64
            * (2.0 * 2f64.powi(-(p.len_acc as i32)) + 2f64.powi(-(p.min_gap as i32)));
        let s = 2f64.powi(p.mask_bits as i32 - p.len_acc as i32);
        (eps, s, s + eps / s)
    }
}

/// Admissibility checks of a given prime set (the paper's
/// `_verify_rns_solution` constraints plus the no-divisor rule).
fn check_rns(
    p: &ApproxParams,
    mults: &[Vec<u64>],
    ell: usize,
    nw1: usize,
    periods: &[u64],
) -> Result<Vec<u64>, ApproxError> {
    let mut v = periods.to_vec();
    v.sort_unstable();
    v.dedup();
    if v.len() != periods.len() {
        return Err(ApproxError::NoRns("primes are not distinct".into()));
    }
    let primes = primes_in(1u64 << (ell - 1), 1u64 << ell);
    for &q in &v {
        if primes.binary_search(&q).is_err() {
            return Err(ApproxError::NoRns(format!("{q} is not an {ell}-bit prime")));
        }
        if mults.iter().flatten().any(|&f| f % q == 0) {
            return Err(ApproxError::NoRns(format!("{q} divides a multiplier")));
        }
        if p.n_mod % q == 0 {
            return Err(ApproxError::NoRns(format!("{q} divides N")));
        }
    }
    let mut l = Big::one();
    let mut need = Big::one();
    let mut lm = 1u64;
    for &q in &v {
        l.mul_u64(q);
        lm = crate::shor::mul_mod(lm, q, p.n_mod);
    }
    for _ in 0..nw1 {
        need.mul_u64(p.n_mod);
    }
    if !l.ge(&need) {
        return Err(ApproxError::NoRns("product of primes below N^{W1}".into()));
    }
    let dev = lm.min(p.n_mod - lm);
    if dev >= p.n_mod >> p.min_gap {
        return Err(ApproxError::NoRns(format!(
            "deviation {dev} not below N >> {}",
            p.min_gap
        )));
    }
    Ok(v)
}

/// Port of `find_rns_for_conf` with the parallel pair search replaced by a
/// deterministic scan (first pair, in order, whose pruned product meets the
/// gap and size constraints).
fn find_rns(
    p: &ApproxParams,
    mults: &[Vec<u64>],
    ell: usize,
    nw1: usize,
) -> Result<Vec<u64>, ApproxError> {
    let n_mod = p.n_mod;
    let max_product_bits = bit_len(n_mod) * nw1;
    let available = primes_in(1u64 << (ell - 1), 1u64 << ell);
    let mut total = Big::one();
    for &q in &available {
        total.mul_u64(q);
    }
    if total.bits() <= max_product_bits + ell * 100 {
        return Err(ApproxError::NoRns(format!(
            "the product of all {ell}-bit primes has {} bits; the paper's search needs more than {}",
            total.bits(),
            max_product_bits + ell * 100
        )));
    }
    // the paper's rule (no prime divides a multiplier), plus: no prime divides N
    // (only possible at toy sizes, n ≤ 2ℓ; such a prime would put a factor of N
    // into the tables, and the paper's pruning step is invalid for it)
    let mut acceptable: Vec<u64> = available
        .iter()
        .copied()
        .filter(|&q| n_mod % q != 0 && mults.iter().flatten().all(|&f| f % q != 0))
        .collect();
    if acceptable.len() < 4 {
        return Err(ApproxError::NoRns("too few acceptable primes".into()));
    }
    let biggest = acceptable.pop().unwrap();
    let second = acceptable.pop().unwrap();
    let mut fixed = Big::one();
    fixed.mul_u64(acceptable[0]);
    fixed.mul_u64(acceptable[1]);
    let mut contiguous = Vec::new();
    while fixed.bits() <= max_product_bits {
        let Some(q) = acceptable.pop() else {
            return Err(ApproxError::NoRns("ran out of primes".into()));
        };
        contiguous.push(q);
        fixed.mul_u64(q);
    }
    contiguous.push(biggest);
    contiguous.push(second);
    if acceptable.len() < p.min_search_primes {
        return Err(ApproxError::NoRns(format!(
            "{} candidate primes left (the paper requires {})",
            acceptable.len(),
            p.min_search_primes
        )));
    }
    let shifted = n_mod >> p.min_gap;
    // N^{W1} as a big integer
    let mut need = Big::one();
    for _ in 0..nw1 {
        need.mul_u64(n_mod);
    }
    for (i, &p1) in acceptable.iter().enumerate() {
        for &p2 in &acceptable[i + 1..] {
            let mut choice = vec![p1, p2];
            choice.extend_from_slice(&contiguous);
            let mut cand = 1u64;
            for &q in &choice {
                cand = crate::shor::mul_mod(cand, q, n_mod);
            }
            // prune: divide out primes that divide the candidate
            let mut kept: Vec<u64> = choice.clone();
            for &q in &choice {
                if cand % q == 0 {
                    cand /= q;
                    kept.retain(|&x| x != q);
                }
            }
            if cand < shifted {
                // the paper's final check (`_verify_rns_solution`): size and the
                // deviation of the *pruned* set. Pruning `q | (L mod N)` divides
                // `L mod N` by `q` only when `gcd(q, N) = 1`; at toy sizes
                // (n ≤ 2ℓ) a factor of N can be an ℓ-bit prime, so the
                // pruned set must be re-checked (the paper's code asserts here).
                let mut l = Big::one();
                let mut lm = 1u64;
                for &q in &kept {
                    l.mul_u64(q);
                    lm = crate::shor::mul_mod(lm, q, n_mod);
                }
                if l.ge(&need) && lm.min(n_mod - lm) < shifted {
                    kept.sort_unstable();
                    return Ok(kept);
                }
            }
        }
    }
    Err(ApproxError::NoRns(format!(
        "no prime set with L mod N < N >> {} (search exhausted)",
        p.min_gap
    )))
}

// ---------------------------------------------------------------------------
// Resolved quint-level program
// ---------------------------------------------------------------------------

/// Register ids of the branch state.
pub const REG_E: u8 = 0;
/// The output accumulator (`len_acc + 1` qubits, starts as the mask `s`).
pub const REG_ACC: u8 = 1;
/// The discrete-log accumulator.
pub const REG_DLOG: u8 = 2;
/// First residue/helper register (`ℓ + 1` qubits).
pub const REG_R0: u8 = 3;
/// Second residue/helper register.
pub const REG_R1: u8 = 4;

/// A slice `(register, offset, length)` of a register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct View {
    /// Register id.
    pub reg: u8,
    /// Lowest qubit.
    pub off: u8,
    /// Number of qubits.
    pub len: u8,
}

impl View {
    fn new(reg: u8, off: usize, len: usize) -> Self {
        View {
            reg,
            off: off as u8,
            len: len as u8,
        }
    }
    /// The Python slice `x[start:][:width]` of this view.
    fn sub(self, start: usize, width: usize) -> Self {
        let len = self.len as usize;
        let s = start.min(len);
        let w = width.min(len - s);
        View::new(self.reg, self.off as usize + s, w)
    }
    #[inline]
    fn get(self, r: &[u64; 5]) -> u64 {
        (r[self.reg as usize] >> self.off) & mask(self.len as usize)
    }
    #[inline]
    fn set(self, r: &mut [u64; 5], v: u64) {
        let m = mask(self.len as usize) << self.off;
        let x = &mut r[self.reg as usize];
        *x = (*x & !m) | ((v << self.off) & m);
    }
}

/// A lookup address: concatenation of views, `Σ value(view) << shift`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Addr(pub Vec<(View, u8)>);

impl Addr {
    #[inline]
    fn eval(&self, r: &[u64; 5]) -> usize {
        let mut a = 0u64;
        for &(v, s) in &self.0 {
            a |= v.get(r) << s;
        }
        a as usize
    }
    fn reads(&self, reg: u8) -> bool {
        self.0.iter().any(|(v, _)| v.reg == reg)
    }
}

/// How a lookup output is combined into its target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LMode {
    /// `target += T[addr]`.
    Add,
    /// `target -= T[addr]`.
    Sub,
    /// `target ^= T[addr]`.
    Xor,
}

/// One resolved quint-level operation (all measurement outcomes fixed).
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// `t ⊕= T[addr]` (mod `2^len`), then the lookup output is X-measured with
    /// outcome `mx`: kickback `(−1)^{mx·T[addr]}` (corrected by a later
    /// [`Op::PhaseLookup`] on the same address).
    Lookup {
        /// Target.
        t: View,
        /// Address.
        addr: Addr,
        /// Index into [`Plan::vtabs`].
        tab: usize,
        /// Combine mode.
        mode: LMode,
        /// Outcome of the output register's X-basis measurement.
        mx: u64,
    },
    /// `t += c` (mod `2^len`).
    AddConst {
        /// Target.
        t: View,
        /// Constant.
        c: u64,
    },
    /// GHZ lookup `t ±= ctrl·val`, its output X-measured (outcome `mx`) and the
    /// fix-up `Z(ctrl)` applied immediately if `mx·val` is odd.
    Ghz {
        /// Target.
        t: View,
        /// One-qubit control.
        ctrl: View,
        /// Added value.
        val: u64,
        /// Subtract instead of add.
        sub: bool,
        /// Outcome.
        mx: u64,
    },
    /// X-basis measurement of `v` with outcome `mx`, then reset to 0.
    MxRz {
        /// Measured view.
        v: View,
        /// Outcome.
        mx: u64,
    },
    /// Phaseup: `(−1)^{B[addr]}`.
    PhaseLookup {
        /// Address.
        addr: Addr,
        /// Index into [`Plan::btabs`].
        bits: usize,
    },
    /// `(−1)^{[t ≥ T[addr]]}` (`ge`) or `(−1)^{[t < T[addr]]}`, the compared
    /// lookup output X-measured with outcome `mx` (kickback vented).
    PhaseCmp {
        /// Compared register.
        t: View,
        /// Address.
        addr: Addr,
        /// Index into [`Plan::vtabs`].
        tab: usize,
        /// `≥` instead of `<`.
        ge: bool,
        /// Outcome.
        mx: u64,
    },
    /// `(−1)^{popcount(v & mask)}`.
    Cz {
        /// Register.
        v: View,
        /// Classical mask.
        mask: u64,
    },
    /// Global phase `−1`.
    GlobalFlip,
    /// Verification: the view must be 0 here (`del_by_equal_to(0)`).
    AssertZero {
        /// View.
        v: View,
    },
    /// Verification: the view must hold the exact residue for prime `i`.
    CheckResidue {
        /// View.
        v: View,
        /// Prime index.
        i: usize,
    },
}

/// A source of measurement outcomes (`len`-bit uniform values).
pub trait Outcomes {
    /// Next outcome of `len` bits.
    fn draw(&mut self, len: usize) -> u64;
}

/// SplitMix64 outcomes.
#[derive(Clone, Debug)]
pub struct RandomOutcomes(pub u64);

impl Outcomes for RandomOutcomes {
    fn draw(&mut self, len: usize) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) & mask(len)
    }
}

/// Replays a recorded outcome stream (e.g. from the Python cross-check).
#[derive(Clone, Debug)]
pub struct ReplayOutcomes {
    /// `(value, len)` pairs in draw order.
    pub log: Vec<(u64, usize)>,
    /// Next index.
    pub pos: usize,
}

impl Outcomes for ReplayOutcomes {
    fn draw(&mut self, len: usize) -> u64 {
        let (v, l) = self.log[self.pos];
        assert_eq!(
            l, len,
            "outcome {} has length {l}, expected {len}",
            self.pos
        );
        self.pos += 1;
        v
    }
}

/// Constant outcomes (all 0 or all 1).
#[derive(Clone, Debug)]
pub struct ConstOutcomes(pub bool);

impl Outcomes for ConstOutcomes {
    fn draw(&mut self, len: usize) -> u64 {
        if self.0 {
            mask(len)
        } else {
            0
        }
    }
}

/// A resolved program: ops plus the tables they reference.
#[derive(Clone, Debug, Default)]
pub struct Plan {
    /// Operations in order.
    pub ops: Vec<Op>,
    /// Value tables (lookups, comparisons).
    pub vtabs: Vec<Vec<u64>>,
    /// Phase tables (resolved vents).
    pub btabs: Vec<Vec<bool>>,
    /// Number of X-basis measurement outcomes drawn.
    pub draws: usize,
    /// Number of outcome bits drawn.
    pub draw_bits: usize,
    /// Number of conditional fix-ups that fired (comparisons, global flips).
    pub fixups: usize,
}

enum Unc {
    Vent(Vec<bool>),
    WrapVent(u64, Vec<bool>),
    Mask(u64),
}

struct Planner<'a, O: Outcomes + ?Sized> {
    c: &'a ApproxConfig,
    o: &'a mut O,
    plan: Plan,
    stack: Vec<Unc>,
}

fn vent_of(tab: &[u64], mx: u64) -> Vec<bool> {
    tab.iter().map(|&t| parity(t & mx)).collect()
}

fn xor_into(v: &mut [bool], w: &[bool]) {
    for (a, b) in v.iter_mut().zip(w) {
        *a ^= *b;
    }
}

impl<O: Outcomes + ?Sized> Planner<'_, O> {
    fn draw(&mut self, len: usize) -> u64 {
        self.plan.draws += 1;
        self.plan.draw_bits += len;
        self.o.draw(len)
    }
    fn vtab(&mut self, t: Vec<u64>) -> usize {
        self.plan.vtabs.push(t);
        self.plan.vtabs.len() - 1
    }
    fn btab(&mut self, t: Vec<bool>) -> usize {
        self.plan.btabs.push(t);
        self.plan.btabs.len() - 1
    }
    /// `t ⊕= tab[addr]` with the output measured; returns the vent increment.
    fn lookup(&mut self, t: View, addr: Addr, tab: Vec<u64>, mode: LMode) -> Vec<bool> {
        let mx = self.draw(t.len as usize);
        let vent = vent_of(&tab, mx);
        let tab = self.vtab(tab);
        self.plan.ops.push(Op::Lookup {
            t,
            addr,
            tab,
            mode,
            mx,
        });
        vent
    }
    fn ghz(&mut self, t: View, ctrl: View, val: u64, sub: bool) {
        let mx = self.draw(t.len as usize);
        self.plan.ops.push(Op::Ghz {
            t,
            ctrl,
            val,
            sub,
            mx,
        });
    }
    fn mxrz(&mut self, v: View) -> u64 {
        let mx = self.draw(v.len as usize);
        self.plan.ops.push(Op::MxRz { v, mx });
        mx
    }
    fn phase_cmp(&mut self, t: View, addr: Addr, tab: Vec<u64>, ge: bool) -> Vec<bool> {
        let mx = self.draw(t.len as usize);
        let vent = vent_of(&tab, mx);
        let tab = self.vtab(tab);
        self.plan.ops.push(Op::PhaseCmp {
            t,
            addr,
            tab,
            ge,
            mx,
        });
        self.plan.fixups += 1;
        vent
    }
    fn phase_lookup(&mut self, addr: Addr, bits: Vec<bool>) {
        if bits.iter().any(|&b| b) {
            let bits = self.btab(bits);
            self.plan.ops.push(Op::PhaseLookup { addr, bits });
        }
    }
    fn global(&mut self) {
        self.plan.fixups += 1;
        self.plan.ops.push(Op::GlobalFlip);
    }

    fn loop1(&mut self, i: usize, vent: &mut [Vec<bool>]) {
        let c = self.c;
        let dlog = View::new(REG_DLOG, 0, c.len_dlog);
        for j in 0..c.nw1 {
            let (off, w) = c.windows1[j];
            let addr = Addr(vec![(View::new(REG_E, off, w), 0)]);
            let tab: Vec<u64> = c.table1[i][j].clone();
            let v = self.lookup(dlog, addr, tab, LMode::Add);
            xor_into(&mut vent[j], &v);
        }
    }

    fn loop2(&mut self, modulus: u64, comp_len: usize) {
        let mut n = self.c.len_dlog;
        while n > comp_len {
            n -= 1;
            let thr = modulus << (n - comp_len);
            let v = View::new(REG_DLOG, 0, n + 1);
            self.plan.ops.push(Op::AddConst {
                t: v,
                c: thr.wrapping_neg() & mask(n + 1),
            });
            self.ghz(
                View::new(REG_DLOG, 0, n),
                View::new(REG_DLOG, n, 1),
                thr,
                false,
            );
        }
    }

    fn unloop2(&mut self, modulus: u64, comp_len: usize) {
        let mut n = comp_len;
        while n < self.c.len_dlog {
            let thr = modulus << (n - comp_len);
            self.ghz(
                View::new(REG_DLOG, 0, n),
                View::new(REG_DLOG, n, 1),
                thr,
                true,
            );
            self.plan.ops.push(Op::AddConst {
                t: View::new(REG_DLOG, 0, n + 1),
                c: thr,
            });
            n += 1;
        }
    }

    /// Returns the register holding the residue.
    fn loop3(&mut self, i: usize) -> u8 {
        let c = self.c;
        let p = &c.params;
        let q = c.periods[i];
        let ell = c.ell;
        let dlog = View::new(REG_DLOG, 0, ell);
        let mut res = REG_R0;
        let mut hel = REG_R1;
        let full = |r: u8| View::new(r, 0, ell + 1);
        let addr0 = Addr(vec![(dlog.sub(0, 2 * p.w3a), 0)]);
        let v = self.lookup(full(res), addr0, c.table3c[i].clone(), LMode::Xor);
        self.stack.push(Unc::Vent(v));
        for j in 2..c.nw3a {
            let l1 = dlog.sub(j * p.w3a, p.w3a);
            for k in 0..c.nw3b {
                let l0 = full(res).sub(k * p.w3b, p.w3b);
                let addr = Addr(vec![(l0, 0), (l1, p.w3b as u8)]);
                let v = self.lookup(full(hel), addr, c.table3a[i][j][k].clone(), LMode::Sub);
                self.ghz(View::new(hel, 0, ell), View::new(hel, ell, 1), q, false);
                let pw = self.mxrz(View::new(hel, ell, 1));
                self.stack.push(Unc::WrapVent(pw, v));
            }
            std::mem::swap(&mut res, &mut hel);
            let mx = self.mxrz(full(hel));
            self.stack.push(Unc::Mask(mx));
        }
        self.plan.ops.push(Op::AssertZero { v: full(hel) });
        self.plan.ops.push(Op::CheckResidue { v: full(res), i });
        res
    }

    fn loop4(&mut self, i: usize, res: u8) {
        let c = self.c;
        let p = &c.params;
        let t = c.trunc;
        let la = p.len_acc;
        let acc = View::new(REG_ACC, 0, la + 1);
        let low = View::new(REG_ACC, 0, la);
        let top = View::new(REG_ACC, la, 1);
        let rv = View::new(res, 0, c.ell + 1);
        for j in 0..c.nw4 {
            let addr = Addr(vec![(rv.sub(j * p.w4, p.w4), 0)]);
            let tab = c.table4[i][j].clone();
            let tab2: Vec<u64> = tab.iter().map(|&x| t - x).collect();
            let mut vent = self.lookup(acc, addr.clone(), tab, LMode::Sub);
            self.ghz(low, top, t, false);
            if self.mxrz(top) == 1 {
                let v = self.phase_cmp(low, addr.clone(), tab2, true);
                xor_into(&mut vent, &v);
            }
            self.phase_lookup(addr, vent);
        }
    }

    fn unloop3(&mut self, i: usize, res: u8) {
        let c = self.c;
        let p = &c.params;
        let q = c.periods[i];
        let ell = c.ell;
        let dlog = View::new(REG_DLOG, 0, ell);
        let full = |r: u8| View::new(r, 0, ell + 1);
        let mut unres = res;
        let mut hel = if res == REG_R0 { REG_R1 } else { REG_R0 };
        for j in (2..c.nw3a).rev() {
            let l0 = dlog.sub(j * p.w3a, p.w3a);
            for k in (0..c.nw3b).rev() {
                let l1 = full(unres).sub(k * p.w3b, p.w3b);
                let addr = Addr(vec![(l1, 0), (l0, p.w3b as u8)]);
                let tab1 = c.table3b[i][j][k].clone();
                let tab2: Vec<u64> = tab1.iter().map(|&x| q - x).collect();
                let mut vent = self.lookup(full(hel), addr.clone(), tab2, LMode::Sub);
                self.ghz(View::new(hel, 0, ell), View::new(hel, ell, 1), q, false);
                let npw = self.mxrz(View::new(hel, ell, 1));
                if npw == 1 {
                    self.global();
                    let v = self.phase_cmp(View::new(hel, 0, ell), addr.clone(), tab1, false);
                    xor_into(&mut vent, &v);
                }
                self.phase_lookup(addr, vent);
            }
            let Some(Unc::Mask(mx)) = self.stack.pop() else {
                panic!("uncompute stack: expected the helper measurement");
            };
            self.plan.ops.push(Op::Cz {
                v: full(hel),
                mask: mx,
            });
            std::mem::swap(&mut unres, &mut hel);
            for k in (0..c.nw3b).rev() {
                let l1 = full(unres).sub(k * p.w3b, p.w3b);
                let addr = Addr(vec![(l1, 0), (l0, p.w3b as u8)]);
                let Some(Unc::WrapVent(pw, mut ptab)) = self.stack.pop() else {
                    panic!("uncompute stack: expected (phase_wrap, vent)");
                };
                let tab1 = c.table3a[i][j][k].clone();
                let tab2: Vec<u64> = tab1.iter().map(|&x| q - x).collect();
                let v = self.lookup(full(hel), addr.clone(), tab2, LMode::Sub);
                xor_into(&mut ptab, &v);
                self.ghz(View::new(hel, 0, ell), View::new(hel, ell, 1), q, false);
                let npw = self.mxrz(View::new(hel, ell, 1));
                if npw == 1 {
                    self.global();
                }
                if pw ^ npw == 1 {
                    let v = self.phase_cmp(View::new(hel, 0, ell), addr.clone(), tab1, false);
                    xor_into(&mut ptab, &v);
                }
                self.phase_lookup(addr, ptab);
            }
        }
        self.plan.ops.push(Op::AssertZero { v: full(hel) });
        let mx = self.mxrz(full(unres));
        let Some(Unc::Vent(mut vent)) = self.stack.pop() else {
            panic!("uncompute stack: expected the loop3 initial-lookup vent");
        };
        let corr = vent_of(&c.table3c[i], mx);
        xor_into(&mut vent, &corr);
        self.phase_lookup(Addr(vec![(dlog.sub(0, 2 * p.w3a), 0)]), vent);
    }
}

/// Resolves the whole `approx_modexp` program for one outcome stream.
pub fn plan<O: Outcomes + ?Sized>(c: &ApproxConfig, o: &mut O) -> Plan {
    let mut pl = Planner {
        c,
        o,
        plan: Plan::default(),
        stack: Vec::new(),
    };
    let mut loop1_vent = vec![vec![false; 1 << c.params.w1]; c.nw1];
    for i in 0..c.periods.len() {
        let q = c.periods[i];
        pl.loop1(i, &mut loop1_vent);
        pl.loop2(q - 1, c.ell);
        let res = pl.loop3(i);
        pl.loop4(i, res);
        pl.unloop3(i, res);
        pl.unloop2(q - 1, c.ell);
    }
    pl.loop1(c.periods.len(), &mut loop1_vent);
    pl.plan.ops.push(Op::AssertZero {
        v: View::new(REG_DLOG, 0, c.len_dlog),
    });
    for j in 0..c.nw1 {
        let (off, w) = c.windows1[j];
        let v = std::mem::take(&mut loop1_vent[j]);
        pl.phase_lookup(Addr(vec![(View::new(REG_E, off, w), 0)]), v);
    }
    assert!(pl.stack.is_empty(), "uncompute info left over");
    pl.plan
}

// ---------------------------------------------------------------------------
// Exact evaluation on every branch
// ---------------------------------------------------------------------------

/// Result of evaluating a plan on every branch `(e, s)`.
#[derive(Clone, Debug, Default)]
pub struct Verify {
    /// Branches evaluated (`2^m · 2^{mask}`).
    pub branches: u64,
    /// Branches whose final sign is `−1`.
    pub bad_sign: u64,
    /// Branches with a non-zero ancilla (discrete log, residue, helper, or
    /// the accumulator's wrap qubit) at any `AssertZero` or at the end.
    pub dirty: u64,
    /// `(e, prime)` pairs whose residue register was wrong after loop3.
    pub bad_residue: u64,
    /// Branches whose accumulator is not `(s + F̃(e)) mod T`.
    pub bad_acc: u64,
    /// Branches whose exponent register changed.
    pub bad_e: u64,
    /// `e` with `F̃(e)` different from the table formula.
    pub bad_formula: u64,
    /// `max_e |F̃(e) − ⌊f(e)/2^t⌋|` (cyclic, accumulator units).
    pub max_dev: u64,
    /// Mean of `|F̃(e) − ⌊f(e)/2^t⌋|`.
    pub mean_dev: f64,
    /// Histogram of the signed deviation `F̃ − ⌊f/2^t⌋` (index = dev + 64, clamped).
    pub dev_hist: Vec<u64>,
    /// `max_e Δ_N(f(e) − F̃(e)·2^t)` (the paper's modular deviation).
    pub max_mod_dev: f64,
}

/// Cyclic signed difference `a − b` in `Z_t`, in `(−t/2, t/2]`.
pub fn cyc_diff(a: u64, b: u64, t: u64) -> i64 {
    let d = (a + t - b % t) % t;
    if d > t / 2 {
        d as i64 - t as i64
    } else {
        d as i64
    }
}

fn apply_scalar(op: &Op, pl: &Plan, c: &ApproxConfig, r: &mut [u64; 5], sign: &mut bool) -> u32 {
    // returns 1 if a verification check failed
    match op {
        Op::Lookup {
            t,
            addr,
            tab,
            mode,
            mx,
        } => {
            let v = pl.vtabs[*tab][addr.eval(r)] & mask(t.len as usize);
            let x = t.get(r);
            let y = match mode {
                LMode::Add => x.wrapping_add(v),
                LMode::Sub => x.wrapping_sub(v),
                LMode::Xor => x ^ v,
            };
            t.set(r, y);
            *sign ^= parity(mx & v);
        }
        Op::AddConst { t, c } => {
            let x = t.get(r);
            t.set(r, x.wrapping_add(*c));
        }
        Op::Ghz {
            t,
            ctrl,
            val,
            sub,
            mx,
        } => {
            let cb = ctrl.get(r) == 1;
            let v = if cb { val & mask(t.len as usize) } else { 0 };
            let x = t.get(r);
            t.set(
                r,
                if *sub {
                    x.wrapping_sub(v)
                } else {
                    x.wrapping_add(v)
                },
            );
            *sign ^= parity(mx & v); // kickback of the measured lookup output
            if parity(val & mx) {
                *sign ^= ctrl.get(r) == 1; // fix-up Z(ctrl)
            }
        }
        Op::MxRz { v, mx } => {
            *sign ^= parity(mx & v.get(r));
            v.set(r, 0);
        }
        Op::PhaseLookup { addr, bits } => {
            *sign ^= pl.btabs[*bits][addr.eval(r)];
        }
        Op::PhaseCmp {
            t,
            addr,
            tab,
            ge,
            mx,
        } => {
            let tv = pl.vtabs[*tab][addr.eval(r)] & mask(t.len as usize);
            let x = t.get(r);
            *sign ^= if *ge { x >= tv } else { x < tv };
            *sign ^= parity(mx & tv);
        }
        Op::Cz { v, mask } => {
            *sign ^= parity(v.get(r) & mask);
        }
        Op::GlobalFlip => *sign ^= true,
        Op::AssertZero { v } => {
            if v.get(r) != 0 {
                return 1;
            }
        }
        Op::CheckResidue { v, i } => {
            let q = c.periods[*i];
            let mut want = 1u64;
            let e = r[REG_E as usize];
            for (j, &(off, w)) in c.windows1.iter().enumerate() {
                let k = ((e >> off) & mask(w)) as usize;
                want = want * (c.mults[j][k] % q) % q;
            }
            if v.get(r) != want {
                return 2;
            }
        }
    }
    0
}

fn touches_acc(op: &Op) -> bool {
    match op {
        Op::Lookup { t, .. } | Op::AddConst { t, .. } | Op::PhaseCmp { t, .. } => t.reg == REG_ACC,
        Op::Ghz { t, .. } => t.reg == REG_ACC,
        Op::MxRz { v, .. } | Op::Cz { v, .. } | Op::AssertZero { v } => v.reg == REG_ACC,
        Op::CheckResidue { .. } | Op::GlobalFlip | Op::PhaseLookup { .. } => false,
    }
}

/// Per-chunk evaluation: `e` in `[e0, e0 + n)`, every mask value `s`.
/// Returns (partial report, F̃ for the chunk).
fn eval_chunk(c: &ApproxConfig, pl: &Plan, e0: u64, n: usize) -> (Verify, Vec<u32>) {
    let sn = 1usize << c.params.mask_bits;
    let la = c.params.len_acc;
    let t = c.trunc;
    let mut rep = Verify {
        dev_hist: vec![0; 129],
        ..Verify::default()
    };
    let mut ft = vec![0u32; n];
    let mut acc = vec![0u64; sn];
    let mut sg = vec![false; sn];
    for (idx, ftv) in ft.iter_mut().enumerate() {
        let e = e0 + idx as u64;
        let mut r = [e, 0, 0, 0, 0];
        let mut sign = false;
        let mut dirty = false;
        for (s, a) in acc.iter_mut().enumerate() {
            *a = s as u64;
        }
        sg.fill(false);
        for op in &pl.ops {
            if touches_acc(op) {
                // vectorised over s; the address (if any) reads residue registers only
                match op {
                    Op::Lookup {
                        t: tv,
                        addr,
                        tab,
                        mode,
                        mx,
                    } => {
                        debug_assert!(!addr.reads(REG_ACC));
                        let v = pl.vtabs[*tab][addr.eval(&r)] & mask(tv.len as usize);
                        sign ^= parity(mx & v);
                        let mk = mask(tv.len as usize) << tv.off;
                        for a in acc.iter_mut() {
                            let x = (*a >> tv.off) & mask(tv.len as usize);
                            let y = match mode {
                                LMode::Add => x.wrapping_add(v),
                                LMode::Sub => x.wrapping_sub(v),
                                LMode::Xor => x ^ v,
                            };
                            *a = (*a & !mk) | ((y << tv.off) & mk);
                        }
                    }
                    Op::Ghz {
                        t: tv,
                        ctrl,
                        val,
                        sub,
                        mx,
                    } => {
                        assert_eq!(ctrl.reg, REG_ACC);
                        let fix = parity(val & mx);
                        let mk = mask(tv.len as usize) << tv.off;
                        for (a, sgn) in acc.iter_mut().zip(sg.iter_mut()) {
                            let cb = (*a >> ctrl.off) & 1 == 1;
                            let v = if cb { val & mask(tv.len as usize) } else { 0 };
                            let x = (*a >> tv.off) & mask(tv.len as usize);
                            let y = if *sub {
                                x.wrapping_sub(v)
                            } else {
                                x.wrapping_add(v)
                            };
                            *a = (*a & !mk) | ((y << tv.off) & mk);
                            *sgn ^= parity(mx & v);
                            if fix {
                                *sgn ^= (*a >> ctrl.off) & 1 == 1;
                            }
                        }
                    }
                    Op::MxRz { v, mx } => {
                        let mk = mask(v.len as usize);
                        for (a, sgn) in acc.iter_mut().zip(sg.iter_mut()) {
                            *sgn ^= parity(mx & ((*a >> v.off) & mk));
                            *a &= !(mk << v.off);
                        }
                    }
                    Op::PhaseCmp {
                        t: tv,
                        addr,
                        tab,
                        ge,
                        mx,
                    } => {
                        let x0 = pl.vtabs[*tab][addr.eval(&r)] & mask(tv.len as usize);
                        sign ^= parity(mx & x0);
                        for (a, sgn) in acc.iter().zip(sg.iter_mut()) {
                            let x = (*a >> tv.off) & mask(tv.len as usize);
                            *sgn ^= if *ge { x >= x0 } else { x < x0 };
                        }
                    }
                    Op::AssertZero { v } => {
                        if acc.iter().any(|a| (a >> v.off) & mask(v.len as usize) != 0) {
                            dirty = true;
                        }
                    }
                    other => panic!("unexpected accumulator op {other:?}"),
                }
            } else {
                match apply_scalar(op, pl, c, &mut r, &mut sign) {
                    0 => {}
                    1 => dirty = true,
                    _ => rep.bad_residue += 1,
                }
            }
        }
        // final checks
        if r[REG_E as usize] != e {
            rep.bad_e += sn as u64;
        }
        let anc_dirty = dirty || r[2] != 0 || r[3] != 0 || r[4] != 0;
        let f0 = acc[0] & mask(la);
        *ftv = f0 as u32;
        for (s, (&a, &sgn)) in acc.iter().zip(&sg).enumerate() {
            rep.branches += 1;
            if sgn ^ sign {
                rep.bad_sign += 1;
            }
            if anc_dirty || (a >> la) != 0 {
                rep.dirty += 1;
            }
            if a != (s as u64 + f0) % t {
                rep.bad_acc += 1;
            }
        }
        if f0 != c.table_formula(e) {
            rep.bad_formula += 1;
        }
        let ideal = c.ideal_trunc(e);
        let d = cyc_diff(f0, ideal, t);
        rep.max_dev = rep.max_dev.max(d.unsigned_abs());
        rep.mean_dev += d.unsigned_abs() as f64;
        rep.dev_hist[(d.clamp(-64, 64) + 64) as usize] += 1;
        let ex = c.exact(e);
        let n_mod = c.params.n_mod;
        let approx = (f0 << c.dropped) % n_mod;
        let err = (ex + n_mod - approx) % n_mod;
        let md = err.min(n_mod - err) as f64 / n_mod as f64;
        rep.max_mod_dev = rep.max_mod_dev.max(md);
    }
    (rep, ft)
}

fn merge(a: &mut Verify, b: &Verify) {
    a.branches += b.branches;
    a.bad_sign += b.bad_sign;
    a.dirty += b.dirty;
    a.bad_residue += b.bad_residue;
    a.bad_acc += b.bad_acc;
    a.bad_e += b.bad_e;
    a.bad_formula += b.bad_formula;
    a.max_dev = a.max_dev.max(b.max_dev);
    a.mean_dev += b.mean_dev;
    if a.dev_hist.is_empty() {
        a.dev_hist = vec![0; 129];
    }
    for (x, y) in a.dev_hist.iter_mut().zip(&b.dev_hist) {
        *x += y;
    }
    a.max_mod_dev = a.max_mod_dev.max(b.max_mod_dev);
}

/// Evaluates `pl` on every branch: all `e < 2^m`, all `s < 2^{mask}`.
/// Returns the report and `F̃(e)` for every `e` (`acc(e, s) = (s + F̃(e)) mod T`
/// is checked, so `F̃` determines the final state when all checks pass).
pub fn evaluate(c: &ApproxConfig, pl: &Plan) -> (Verify, Vec<u32>) {
    let total = 1u64 << c.m;
    let chunk = 1usize << 10.min(c.m);
    let parts: Vec<(Verify, Vec<u32>)> = (0..total.div_ceil(chunk as u64))
        .into_par_iter()
        .map(|k| {
            let e0 = k * chunk as u64;
            let n = chunk.min((total - e0) as usize);
            eval_chunk(c, pl, e0, n)
        })
        .collect();
    let mut rep = Verify {
        dev_hist: vec![0; 129],
        ..Verify::default()
    };
    let mut ft = Vec::with_capacity(total as usize);
    for (r, f) in parts {
        merge(&mut rep, &r);
        ft.extend_from_slice(&f);
    }
    rep.mean_dev /= total as f64;
    (rep, ft)
}

/// Evaluates one branch `(e, s)` with the full register state at the end
/// (for spot checks and the Python cross-check): returns the five registers
/// and the sign.
pub fn eval_branch(c: &ApproxConfig, pl: &Plan, e: u64, s: u64) -> ([u64; 5], bool, u32) {
    let mut r = [e, s, 0, 0, 0];
    let mut sign = false;
    let mut fails = 0;
    for op in &pl.ops {
        fails += apply_scalar(op, pl, c, &mut r, &mut sign);
    }
    (r, sign, fails)
}

// ---------------------------------------------------------------------------
// Output distribution of the frequency-basis measurement
// ---------------------------------------------------------------------------

/// Twiddle table `e^{−2πik/n}`, `k < n/2`, for [`fft_with`].
pub fn twiddles(n: usize) -> Vec<Complex64> {
    (0..n / 2)
        .map(|k| Complex64::from_polar(1.0, -2.0 * std::f64::consts::PI * k as f64 / n as f64))
        .collect()
}

/// In-place radix-2 FFT with a precomputed [`twiddles`] table of the same size.
pub fn fft_with(a: &mut [Complex64], tw: &[Complex64]) {
    let n = a.len();
    assert!(n.is_power_of_two());
    if n <= 1 {
        return;
    }
    assert_eq!(tw.len(), n / 2);
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if i < j {
            a.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let stride = n / len;
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let u = a[start + k];
                let v = a[start + k + half] * tw[k * stride];
                a[start + k] = u + v;
                a[start + k + half] = u - v;
            }
        }
        len <<= 1;
    }
}

/// In-place radix-2 FFT, `X_j = Σ_x x_x e^{−2πi xj/n}` (`n` a power of two).
pub fn fft(a: &mut [Complex64]) {
    let tw = twiddles(a.len());
    fft_with(a, &tw);
}

/// 2-D FFT of a row-major `rows × cols` array (index `x + cols·y`), both
/// powers of two, with precomputed twiddle tables for both sizes.
pub fn fft2_with(
    a: &mut [Complex64],
    cols: usize,
    rows: usize,
    twc: &[Complex64],
    twr: &[Complex64],
) {
    assert_eq!(a.len(), cols * rows);
    for r in a.chunks_mut(cols) {
        fft_with(r, twc);
    }
    if rows > 1 {
        let mut col = vec![Complex64::new(0.0, 0.0); rows];
        for x in 0..cols {
            for y in 0..rows {
                col[y] = a[x + cols * y];
            }
            fft_with(&mut col, twr);
            for y in 0..rows {
                a[x + cols * y] = col[y];
            }
        }
    }
}

/// 2-D FFT of a row-major `rows × cols` array (index `x + cols·y`), both powers of two.
pub fn fft2(a: &mut [Complex64], cols: usize, rows: usize) {
    let (twc, twr) = (twiddles(cols), twiddles(rows));
    fft2_with(a, cols, rows, &twc, &twr);
}

/// The exact distribution of the frequency-basis measurement after measuring
/// an output register that holds `(s + F(e)) mod t`, `s` uniform in `[0, w)`,
/// `e` uniform over `2^{ma + mb}` values (`e = a + 2^{ma} b`):
/// `P(j, k) = Σ_V |Σ_{(e,s): out = V} ω_A^{aj} ω_B^{bk}|² / (2^{2(ma+mb)} w)`.
/// Returns the distribution (index `j + 2^{ma} k`) and the distribution of `V`.
/// Pairs of real inputs share one complex FFT.
pub fn distribution(f: &[u32], t: u64, w: u64, ma: usize, mb: usize) -> (Vec<f64>, Vec<f64>) {
    let m = ma + mb;
    let size = 1usize << m;
    assert_eq!(f.len(), size);
    assert!(w >= 1 && w <= t);
    // bucket e by F(e)
    let tt = t as usize;
    let mut cnt = vec![0u32; tt + 1];
    for &v in f {
        cnt[v as usize + 1] += 1;
    }
    for i in 0..tt {
        cnt[i + 1] += cnt[i];
    }
    let mut order = vec![0u32; size];
    let mut pos: Vec<u32> = cnt[..tt].to_vec();
    for (e, &v) in f.iter().enumerate() {
        order[pos[v as usize] as usize] = e as u32;
        pos[v as usize] += 1;
    }
    // P(V) = |{(e, s): (s + F(e)) mod t = V}| / (2^m w)
    let norm_v = 1.0 / (size as f64 * w as f64);
    let support = |v: usize| -> std::ops::Range<usize> { cnt[v] as usize..cnt[v + 1] as usize };
    // cyclic window sums of the bucket sizes: |{e : (V − F(e)) mod t < w}|
    let wu = w as usize;
    let mut run: usize = (0..wu).map(|d| support((tt - d % tt) % tt).len()).sum();
    let mut pv = vec![0f64; tt];
    for v in 0..tt {
        if v > 0 {
            run += support(v).len();
            run -= support((v + tt - wu) % tt).len();
        }
        pv[v] = run as f64 * norm_v;
    }
    let vs: Vec<usize> = (0..tt).filter(|&v| pv[v] > 0.0).collect();
    let pairs: Vec<(usize, Option<usize>)> =
        vs.chunks(2).map(|ch| (ch[0], ch.get(1).copied())).collect();
    let nthreads = rayon::current_num_threads().max(1);
    let per = pairs.len().div_ceil(nthreads).max(1);
    let fill = |buf: &mut [Complex64], v: usize, imag: bool| {
        for d in 0..w as usize {
            let u = (v + tt - d % tt) % tt;
            for &e in &order[support(u)] {
                if imag {
                    buf[e as usize].im += 1.0;
                } else {
                    buf[e as usize].re += 1.0;
                }
            }
        }
    };
    let (twc, twr) = (twiddles(1 << ma), twiddles(1 << mb));
    let accs: Vec<Vec<f64>> = pairs
        .par_chunks(per)
        .map(|chunk| {
            let mut buf = vec![Complex64::new(0.0, 0.0); size];
            let mut a = vec![0f64; size];
            for &(v1, v2) in chunk {
                buf.fill(Complex64::new(0.0, 0.0));
                fill(&mut buf, v1, false);
                if let Some(v2) = v2 {
                    fill(&mut buf, v2, true);
                }
                fft2_with(&mut buf, 1 << ma, 1 << mb, &twc, &twr);
                for (x, z) in a.iter_mut().zip(&buf) {
                    *x += z.norm_sqr();
                }
            }
            a
        })
        .collect();
    let mut a = vec![0f64; size];
    for x in &accs {
        for (y, z) in a.iter_mut().zip(x) {
            *y += z;
        }
    }
    // P(j,k) = (A(j,k) + A(−j,−k)) / 2 / norm
    let (na, nb) = (1usize << ma, 1usize << mb);
    let norm = 1.0 / (size as f64 * size as f64 * w as f64);
    let mut p = vec![0f64; size];
    for k in 0..nb {
        for j in 0..na {
            let jm = (na - j) % na;
            let km = (nb - k) % nb;
            p[j + na * k] = 0.5 * (a[j + na * k] + a[jm + na * km]) * norm;
        }
    }
    (p, pv)
}

/// Total-variation distance.
pub fn tv(a: &[f64], b: &[f64]) -> f64 {
    0.5 * a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>()
}

/// `|⟨ψ̃|ψ⟩|` of two masked pre-measurement states
/// `2^{−m/2} w^{−1/2} Σ_{e,s} |e⟩|(s + F(e)) mod t⟩`: for each `e` the two
/// windows overlap in `max(0, w − |F̃(e) − F(e)|)` values.
pub fn overlap(fa: &[u32], fb: &[u32], t: u64, w: u64) -> f64 {
    let s: f64 = fa
        .par_iter()
        .zip(fb)
        .map(|(&a, &b)| {
            let d = cyc_diff(u64::from(a), u64::from(b), t).unsigned_abs();
            w.saturating_sub(d) as f64
        })
        .sum();
    s / (fa.len() as f64 * w as f64)
}

/// The constant shift `c` maximising [`overlap`]`(fa, fb + c)`, and that
/// overlap. A constant shift of the output only relabels the measured value
/// `V` (it changes neither the post-measurement exponent states nor the
/// frequency distribution), so this overlap measures the `e`-dependent part
/// of the deviation; the truncations' systematic rounding-down is the
/// constant part.
pub fn best_shift(fa: &[u32], fb: &[u32], t: u64, w: u64) -> (i64, f64) {
    let mut hist = std::collections::BTreeMap::<i64, u64>::new();
    for (&a, &b) in fa.iter().zip(fb) {
        *hist
            .entry(cyc_diff(u64::from(a), u64::from(b), t))
            .or_default() += 1;
    }
    let lo = *hist.keys().next().unwrap();
    let hi = *hist.keys().next_back().unwrap();
    let mut best = (0i64, -1.0f64);
    for c in lo..=hi {
        let mut s = 0.0;
        for (&d, &n) in &hist {
            s += n as f64 * w.saturating_sub((d - c).unsigned_abs()) as f64;
        }
        let ov = s / (fa.len() as f64 * w as f64);
        if ov > best.1 {
            best = (c, ov);
        }
    }
    best
}

/// `F + c mod t` for every entry (applies a [`best_shift`]).
pub fn shifted(f: &[u32], c: i64, t: u64) -> Vec<u32> {
    let c = c.rem_euclid(t as i64) as u64;
    f.iter().map(|&x| ((u64::from(x) + c) % t) as u32).collect()
}

/// Conditional fidelity of the exponent register after the output
/// measurement: `Σ_V P̃(V) |⟨ψ_V|ψ̃_V⟩|²`, where `ψ_V` (`ψ̃_V`) is the normalised
/// post-measurement state of the exponent register given output `V` for the
/// ideal (actual) circuit, i.e. the uniform superposition over
/// `{e : (V − F(e)) mod t < w}`. Returns `(conditional fidelity,
/// P̃(V outside the ideal support))`. `O(2^m + t)` with difference arrays.
pub fn cond_fidelity(f_ideal: &[u32], f_act: &[u32], t: u64, w: u64) -> (f64, f64) {
    let tt = t as usize;
    let wu = w as usize;
    // cyclic interval [lo, lo + len) of V values, added with +1
    let add = |d: &mut [i64], lo: usize, len: usize| {
        if len == 0 {
            return;
        }
        let hi = lo + len;
        if hi <= tt {
            d[lo] += 1;
            d[hi] -= 1;
        } else {
            d[lo] += 1;
            d[tt] -= 1;
            d[0] += 1;
            d[hi - tt] -= 1;
        }
    };
    let mut di = vec![0i64; tt + 1];
    let mut da = vec![0i64; tt + 1];
    let mut db = vec![0i64; tt + 1];
    for (&a, &b) in f_ideal.iter().zip(f_act) {
        let (a, b) = (a as usize, b as usize);
        add(&mut di, a, wu);
        add(&mut da, b, wu);
        // both windows: V − a and V − b in [0, w): V in [max, min + w) cyclically
        let d = cyc_diff(b as u64, a as u64, t);
        let ov = wu.saturating_sub(d.unsigned_abs() as usize);
        if ov > 0 {
            let lo = if d >= 0 { b } else { a };
            add(&mut db, lo, ov);
        }
    }
    let (mut ci, mut ca, mut cb) = (0i64, 0i64, 0i64);
    let n = f_act.len() as f64 * w as f64;
    let (mut fid, mut outside) = (0.0, 0.0);
    for v in 0..tt {
        ci += di[v];
        ca += da[v];
        cb += db[v];
        if ca > 0 {
            let p = ca as f64 / n;
            if ci == 0 {
                outside += p;
            } else {
                fid += p * (cb * cb) as f64 / (ci as f64 * ca as f64);
            }
        }
    }
    (fid, outside)
}

/// Python's `Fraction(num, den).limit_denominator(max_den)` (exact).
pub fn limit_denominator(num: u128, den: u128, max_den: u128) -> (u128, u128) {
    let g = {
        let (mut a, mut b) = (num, den);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a.max(1)
    };
    let (n0, d0) = (num / g, den / g);
    if d0 <= max_den {
        return (n0, d0);
    }
    let (mut p0, mut q0, mut p1, mut q1) = (0u128, 1u128, 1u128, 0u128);
    let (mut n, mut d) = (n0, d0);
    loop {
        let a = n / d;
        let q2 = q0 + a * q1;
        if q2 > max_den {
            break;
        }
        (p0, q0, p1, q1) = (p1, q1, p0 + a * p1, q2);
        (n, d) = (d, n - a * d);
    }
    let k = (max_den - q0) / q1;
    // bound1 = (p0 + k p1)/(q0 + k q1), bound2 = p1/q1; pick the closer (ties -> bound2)
    let (b1n, b1d) = (p0 + k * p1, q0 + k * q1);
    let x = (n0 as i128, d0 as i128);
    let dist = |pn: u128, pd: u128| -> (i128, i128) {
        // |pn/pd − x| as a fraction (num, den)
        let num = (pn as i128 * x.1 - x.0 * pd as i128).abs();
        (num, pd as i128 * x.1)
    };
    let (e2n, e2d) = dist(p1, q1);
    let (e1n, e1d) = dist(b1n, b1d);
    if e2n * e1d <= e1n * e2d {
        (p1, q1)
    } else {
        (b1n, b1d)
    }
}

/// The success test of the paper's success-rate model
/// (`main1_sample_masked_success_rates.py`) applied to an outcome `j` of a
/// `2^m`-point QFT: `d` = denominator of `j/2^m` limited to `N`, success iff
/// `1 < gcd(g^{⌊d/2⌋} + 1, N) < N`.
pub fn paper_success(j: u64, m: usize, n_mod: u64, g: u64) -> bool {
    let (_, d) = limit_denominator(u128::from(j), 1u128 << m, u128::from(n_mod));
    let y = pow_mod(g, (d / 2) as u64, n_mod);
    let f = gcd(y + 1, n_mod);
    f > 1 && f < n_mod
}

// ---------------------------------------------------------------------------
// Gate level: one loop4 window step
// ---------------------------------------------------------------------------

/// Qubit layout of the gate-level loop4 window step.
#[derive(Clone, Debug)]
pub struct Loop4Layout {
    /// Address (residue window), `w4` qubits.
    pub k: Vec<usize>,
    /// Accumulator, `f + 1` qubits (the top one is the wrap qubit).
    pub acc: Vec<usize>,
    /// Lookup output / constant register, `f` qubits.
    pub tmp: Vec<usize>,
    /// AND ancillas of the unary iteration (`w4`).
    pub and: Vec<usize>,
    /// Carry ancillas of the Gidney adders (`f − 1`).
    pub cy: Vec<usize>,
    /// Total qubits.
    pub nq: usize,
}

impl Loop4Layout {
    /// Layout for window `w4` and accumulator length `f`.
    pub fn new(w4: usize, f: usize) -> Self {
        let mut q = 0;
        let mut take = |n: usize| {
            let v: Vec<usize> = (q..q + n).collect();
            q += n;
            v
        };
        let k = take(w4);
        let acc = take(f + 1);
        let tmp = take(f);
        let and = take(w4.max(1));
        let cy = take(f.saturating_sub(1).max(1));
        Loop4Layout {
            k,
            acc,
            tmp,
            and,
            cy,
            nq: q,
        }
    }
}

/// One loop4 window step as X/CNOT/Toffoli gates, X-basis measurements and
/// Z/CZ fix-ups ([`crate::shor::mbu::MbuOp`]), resolved for the outcomes
/// drawn from `bit`: lookup `tmp = T[k]` (unary iteration, AND ancillas
/// uncomputed by measurement); `acc −= tmp` (Gidney subtractor, `f + 1` bits);
/// measurement-based unlookup of `tmp` (vent); `acc[:f] += acc[f]·T_mod` by a
/// controlled constant load, a Gidney adder and the unload; X-measurement of
/// the wrap qubit `acc[f]` and, if 1, the phase comparison
/// `(−1)^{[acc[:f] ≥ T_mod − T[k]]}` (lookup of `T_mod − T`, phase comparator,
/// global sign, measurement-based unlookup into the same vent); finally the
/// vent phaseup on `k`. `table` holds the paper's negated loop4 entries.
pub fn gate_loop4_step(
    lay: &Loop4Layout,
    table: &[u64],
    trunc: u64,
    bit: &mut dyn FnMut() -> bool,
) -> Vec<crate::shor::mbu::MbuOp> {
    use crate::gate::Gate;
    use crate::shor::mbu::{
        add_g, lookup_ops, phase_lt_g, phase_table, resolve, sub_g, LOp, LookupSpec, MbuOp, NO_CTRL,
    };
    let f = lay.tmp.len();
    let mut out: Vec<MbuOp> = Vec::new();
    let spec = |tab: &[u64]| LookupSpec {
        ctrl: NO_CTRL,
        addr: lay.k.clone(),
        and: lay.and.clone(),
        out: lay.tmp.clone(),
        table: tab.to_vec(),
        mbu_and: true,
        meas_unlookup: true,
    };
    // measurement-based unlookup of `tab` from tmp, returning the vent increment
    let unlookup = |out: &mut Vec<MbuOp>, tab: &[u64], bit: &mut dyn FnMut() -> bool| {
        let mut mx = 0u64;
        for (j, &q) in lay.tmp.iter().enumerate() {
            let m = bit();
            out.push(MbuOp::MeasX(q, m));
            if m {
                mx |= 1 << j;
            }
        }
        tab.iter().map(|&t| parity(t & mx)).collect::<Vec<bool>>()
    };
    // 1. lookup tmp = T[k]
    resolve(&lookup_ops(&spec(table)), bit, &mut out);
    // 2. acc -= tmp (f + 1 bits)
    let mut ops = Vec::new();
    sub_g(&mut ops, &lay.tmp, &lay.acc, &lay.cy);
    resolve(&ops, bit, &mut out);
    // 3. unlookup tmp by measurement into the vent
    let mut vent = unlookup(&mut out, table, bit);
    // 4. acc[:f] += acc[f]·trunc: controlled load, add (mod 2^f), unload
    let top = lay.acc[f];
    let load: Vec<usize> = (0..f).filter(|&j| (trunc >> j) & 1 == 1).collect();
    for &j in &load {
        out.push(MbuOp::G(Gate::Cnot(top, lay.tmp[j])));
    }
    let mut ops = Vec::new();
    if f >= 2 {
        add_g(&mut ops, &lay.tmp[..f - 1], &lay.acc[..f], &lay.cy);
        ops.push(LOp::G(Gate::Cnot(lay.tmp[f - 1], lay.acc[f - 1])));
    } else {
        ops.push(LOp::G(Gate::Cnot(lay.tmp[0], lay.acc[0])));
    }
    resolve(&ops, bit, &mut out);
    for &j in &load {
        out.push(MbuOp::G(Gate::Cnot(top, lay.tmp[j])));
    }
    // 5. X-measure the wrap qubit; if 1: phase (−1)^{[acc_low ≥ trunc − T[k]]}
    let m = bit();
    out.push(MbuOp::MeasX(top, m));
    if m {
        let tab2: Vec<u64> = table.iter().map(|&x| trunc - x).collect();
        resolve(&lookup_ops(&spec(&tab2)), bit, &mut out);
        // (−1)^{[acc ≥ t2]} = −(−1)^{[acc < t2]}
        let mut ops = Vec::new();
        phase_lt_g(&mut ops, &lay.tmp, &lay.acc[..f], &lay.cy);
        ops.push(LOp::GlobalNeg);
        resolve(&ops, bit, &mut out);
        let v = unlookup(&mut out, &tab2, bit);
        xor_into(&mut vent, &v);
    }
    // 6. vent phaseup on k (scratch: AND ancillas + tmp, all clean here)
    if vent.iter().any(|&b| b) {
        let scratch: Vec<usize> = lay.and.iter().chain(&lay.tmp).copied().collect();
        let ops = phase_table(NO_CTRL, &lay.k, &vent, &scratch);
        resolve(&ops, bit, &mut out);
    }
    out
}

/// Qubit layout of the gate-level loop3/unloop3 step pair.
#[derive(Clone, Debug)]
pub struct Loop3Layout {
    /// Discrete-log window (`w3a` qubits).
    pub l1: Vec<usize>,
    /// Residue window (`w3b` qubits) — the address part read from the residue.
    pub r: Vec<usize>,
    /// Helper register, `ℓ + 1` qubits (top = wrap qubit).
    pub hel: Vec<usize>,
    /// Lookup output, `ℓ + 1` qubits.
    pub tmp: Vec<usize>,
    /// AND ancillas (`w3a + w3b`).
    pub and: Vec<usize>,
    /// Carry ancillas (`ℓ`).
    pub cy: Vec<usize>,
    /// Total qubits.
    pub nq: usize,
}

impl Loop3Layout {
    /// Layout for windows `w3a`, `w3b` and prime bit length `ell`.
    pub fn new(w3a: usize, w3b: usize, ell: usize) -> Self {
        let mut q = 0;
        let mut take = |n: usize| {
            let v: Vec<usize> = (q..q + n).collect();
            q += n;
            v
        };
        let r = take(w3b);
        let l1 = take(w3a);
        let hel = take(ell + 1);
        let tmp = take(ell + 1);
        let and = take(w3a + w3b);
        let cy = take(ell);
        Loop3Layout {
            l1,
            r,
            hel,
            tmp,
            and,
            cy,
            nq: q,
        }
    }
}

/// One loop3 inner step and, later, its unloop3 counterpart as gates (the
/// pair whose wrap-qubit phase correction is deferred across subroutines):
///
/// loop3: `tmp = T3a[addr]`; `hel −= tmp` (`ℓ + 1` bits); unlookup `tmp` by
/// measurement into vent `V`; `hel[:ℓ] += hel[ℓ]·p`; X-measure the wrap
/// qubit `hel[ℓ]` (outcome `pw`, correction deferred).
/// unloop3: `tmp = (p − T3a)[addr]`; `hel −= tmp`; unlookup into `V`;
/// `hel[:ℓ] += hel[ℓ]·p`; X-measure the wrap qubit (outcome `npw`); global
/// `−1` if `npw`; if `pw ⊕ npw`: phase `(−1)^{[hel < T3a[addr]]}` (lookup,
/// phase comparator, unlookup into `V`); phaseup of `V` on `addr`.
///
/// `addr = (l1 << w3b) | r`. The composition is the identity on
/// `(l1, r, hel < p)` with every ancilla clean, up to one global sign.
pub fn gate_loop3_pair(
    lay: &Loop3Layout,
    t3a: &[u64],
    p: u64,
    bit: &mut dyn FnMut() -> bool,
) -> Vec<crate::shor::mbu::MbuOp> {
    use crate::gate::Gate;
    use crate::shor::mbu::{
        add_g, lookup_ops, phase_lt_g, phase_table, resolve, sub_g, LOp, LookupSpec, MbuOp, NO_CTRL,
    };
    let ell = lay.hel.len() - 1;
    let addr: Vec<usize> = lay.r.iter().chain(&lay.l1).copied().collect();
    let spec = |tab: &[u64], out: &[usize]| LookupSpec {
        ctrl: NO_CTRL,
        addr: addr.clone(),
        and: lay.and.clone(),
        out: out.to_vec(),
        table: tab.to_vec(),
        mbu_and: true,
        meas_unlookup: true,
    };
    let mut out: Vec<MbuOp> = Vec::new();
    let unlookup =
        |out: &mut Vec<MbuOp>, tab: &[u64], qs: &[usize], bit: &mut dyn FnMut() -> bool| {
            let mut mx = 0u64;
            for (j, &q) in qs.iter().enumerate() {
                let m = bit();
                out.push(MbuOp::MeasX(q, m));
                if m {
                    mx |= 1 << j;
                }
            }
            tab.iter().map(|&t| parity(t & mx)).collect::<Vec<bool>>()
        };
    // hel −= tmp over ℓ + 1 bits: sub_g needs an ℓ-bit subtrahend and an
    // (ℓ + 1)-bit target; T < p < 2^ℓ so tmp[ℓ] stays 0.
    let sub = |out: &mut Vec<MbuOp>, bit: &mut dyn FnMut() -> bool| {
        let mut ops = Vec::new();
        sub_g(&mut ops, &lay.tmp[..ell], &lay.hel, &lay.cy);
        resolve(&ops, bit, out);
    };
    // hel[:ℓ] += hel[ℓ]·p (mod 2^ℓ): controlled load into tmp, add, unload
    let ghz = |out: &mut Vec<MbuOp>, bit: &mut dyn FnMut() -> bool| {
        let top = lay.hel[ell];
        let load: Vec<usize> = (0..ell).filter(|&j| (p >> j) & 1 == 1).collect();
        for &j in &load {
            out.push(MbuOp::G(Gate::Cnot(top, lay.tmp[j])));
        }
        let mut ops = Vec::new();
        add_g(&mut ops, &lay.tmp[..ell - 1], &lay.hel[..ell], &lay.cy);
        ops.push(LOp::G(Gate::Cnot(lay.tmp[ell - 1], lay.hel[ell - 1])));
        resolve(&ops, bit, out);
        for &j in &load {
            out.push(MbuOp::G(Gate::Cnot(top, lay.tmp[j])));
        }
    };
    // ---- loop3 step
    resolve(&lookup_ops(&spec(t3a, &lay.tmp)), bit, &mut out);
    sub(&mut out, bit);
    let mut vent = unlookup(&mut out, t3a, &lay.tmp, bit);
    ghz(&mut out, bit);
    let pw = bit();
    out.push(MbuOp::MeasX(lay.hel[ell], pw));
    // ---- unloop3 counterpart
    let t2: Vec<u64> = t3a.iter().map(|&x| p - x).collect();
    resolve(&lookup_ops(&spec(&t2, &lay.tmp)), bit, &mut out);
    sub(&mut out, bit);
    let v = unlookup(&mut out, &t2, &lay.tmp, bit);
    xor_into(&mut vent, &v);
    ghz(&mut out, bit);
    let npw = bit();
    out.push(MbuOp::MeasX(lay.hel[ell], npw));
    if npw {
        out.push(MbuOp::GlobalNeg);
    }
    if pw ^ npw {
        resolve(&lookup_ops(&spec(t3a, &lay.tmp[..ell])), bit, &mut out);
        let mut ops = Vec::new();
        phase_lt_g(&mut ops, &lay.tmp[..ell], &lay.hel[..ell], &lay.cy);
        resolve(&ops, bit, &mut out);
        let v = unlookup(&mut out, t3a, &lay.tmp[..ell], bit);
        xor_into(&mut vent, &v);
    }
    if vent.iter().any(|&b| b) {
        let scratch: Vec<usize> = lay.and.iter().chain(&lay.tmp).copied().collect();
        resolve(&phase_table(NO_CTRL, &addr, &vent, &scratch), bit, &mut out);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_denominator_matches_python_examples() {
        // Fraction(3141592653589793, 10**15).limit_denominator(1000) == 355/113
        assert_eq!(
            limit_denominator(3141592653589793, 1_000_000_000_000_000, 1000),
            (355, 113)
        );
        // Fraction(1, 3).limit_denominator(2) == 1/2 (tie -> p1/q1 rule)
        assert_eq!(limit_denominator(1, 3, 2), (1, 2));
        assert_eq!(limit_denominator(0, 8, 5), (0, 1));
        assert_eq!(limit_denominator(5, 8, 100), (5, 8));
    }

    #[test]
    fn fft_matches_naive_dft() {
        let n = 16;
        let x: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i * 7 % 5) as f64, (i % 3) as f64))
            .collect();
        let mut y = x.clone();
        fft(&mut y);
        for j in 0..n {
            let mut s = Complex64::new(0.0, 0.0);
            for (i, xi) in x.iter().enumerate() {
                s += xi
                    * Complex64::from_polar(
                        1.0,
                        -2.0 * std::f64::consts::PI * (i * j) as f64 / n as f64,
                    );
            }
            assert!((s - y[j]).norm() < 1e-9);
        }
    }

    #[test]
    fn generator_choice_matches_paper_rule() {
        // smallest i >= 3 of full order: p = 23 -> 5, p = 2053 -> 3? check by brute force
        for p in [23u64, 101, 2053, 4093] {
            let g = paper_generator(p);
            assert!(g >= 3);
            let ord = (1..p).find(|&k| pow_mod(g, k, p) == 1).unwrap();
            assert_eq!(ord, p - 1);
            for i in 3..g {
                let o = (1..p).find(|&k| pow_mod(i, k, p) == 1).unwrap();
                assert!(o < p - 1);
            }
        }
    }
}
