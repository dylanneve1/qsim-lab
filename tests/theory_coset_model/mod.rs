//! Abstract model of the coset-arithmetic windowed Shor circuit of
//! `src/shor/ge.rs` (`GeOpts::coset = c`), independent of the gate-level
//! engine: every window block is the classical permutation
//!   (x, b) -> (x', b') = ((b + A_h(x)) mod M, (x - A_{h^-1}(x')) mod M)
//! of Z_M^2, M = 2^{n+c}, with A_h(x) = sum_k (h * chunk_k(x) * 2^{k w_m} mod N)
//! (plain integer sum of looked-up residues, chunks of w_m bits).
//! Exact arithmetic (c = 0) is x -> h x mod N.
//!
//! The output distribution of phase estimation depends only on the Gram
//! matrix K(E,E') = <Phi_E'|Phi_E> of the work states Phi_E = Pi_E psi_0;
//! here Phi_E is uniform on the set S_E = Pi_E(S_0), so
//! K(E,E') = |S_E ∩ S_E'| / 4^c and
//! p(y) = 4^{-t} 4^{-c} sum_d C(d) cos(2 pi y d / 2^t),
//! C(d) = #{(w, E, E') : w in S_E ∩ S_E', E - E' = d mod 2^t}.
#![allow(dead_code, clippy::needless_range_loop)]

pub fn mulmod(a: u64, b: u64, n: u64) -> u64 {
    (u128::from(a) * u128::from(b) % u128::from(n)) as u64
}
pub fn powmod(mut a: u64, mut e: u64, n: u64) -> u64 {
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
pub fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}
pub fn inv(a: u64, n: u64) -> u64 {
    let (mut t, mut nt, mut r, mut nr) = (0i128, 1i128, n as i128, (a % n) as i128);
    while nr != 0 {
        let q = r / nr;
        (t, nt) = (nt, t - q * nt);
        (r, nr) = (nr, r - q * nr);
    }
    assert_eq!(r, 1);
    t.rem_euclid(n as i128) as u64
}
pub fn order(a: u64, n: u64) -> u64 {
    let mut x = a % n;
    let mut r = 1;
    while x != 1 {
        x = mulmod(x, a, n);
        r += 1;
    }
    r
}
/// `work_bits` of the repo: bit length of N - 1.
pub fn work_bits(n_mod: u64) -> usize {
    64 - (n_mod - 1).leading_zeros() as usize
}

#[derive(Clone, Debug)]
pub struct Model {
    pub n_mod: u64,
    pub n: usize,
    pub c: usize,
    pub nr: usize,
    pub we: usize,
    pub wm: usize,
    pub t: usize,
    /// windows in run order: (i0, w, g = base^{2^{t-i0-w}})
    pub wins: Vec<(usize, usize, u64)>,
}

impl Model {
    pub fn new(n_mod: u64, a: u64, we: usize, wm: usize, c: usize) -> Self {
        Self::with_t(n_mod, a, we, wm, c, 2 * work_bits(n_mod))
    }
    pub fn with_t(n_mod: u64, a: u64, we: usize, wm: usize, c: usize, t: usize) -> Self {
        let n = work_bits(n_mod);
        let mut wins = Vec::new();
        let mut i0 = 0;
        while i0 < t {
            let w = we.min(t - i0);
            let mut g = a % n_mod;
            for _ in 0..t - i0 - w {
                g = mulmod(g, g, n_mod);
            }
            wins.push((i0, w, g));
            i0 += w;
        }
        Model {
            n_mod,
            n,
            c,
            nr: n + c,
            we,
            wm,
            t,
            wins,
        }
    }
    pub fn m(&self) -> u64 {
        1u64 << self.nr
    }
    /// Number of lookup-additions per emult.
    pub fn chunks(&self) -> usize {
        self.nr.div_ceil(self.wm)
    }
    /// A_h(x): sum of the looked-up residues (not reduced).
    pub fn a_sum(&self, h: u64, x: u64) -> u64 {
        let nm = self.n_mod;
        let mut s = 0u64;
        let mut start = 0;
        let mut pw = 1 % nm;
        while start < self.nr {
            let w = self.wm.min(self.nr - start);
            let v = (x >> start) & ((1 << w) - 1);
            s += mulmod(v % nm, mulmod(h, pw, nm), nm);
            for _ in 0..w {
                pw = mulmod(pw, 2, nm);
            }
            start += w;
        }
        s
    }
    /// The individual looked-up residues (for per-addition analysis).
    pub fn a_terms(&self, h: u64, x: u64) -> Vec<u64> {
        let nm = self.n_mod;
        let mut v = Vec::new();
        let mut start = 0;
        let mut pw = 1 % nm;
        while start < self.nr {
            let w = self.wm.min(self.nr - start);
            let ch = (x >> start) & ((1 << w) - 1);
            v.push(mulmod(ch % nm, mulmod(h, pw, nm), nm));
            for _ in 0..w {
                pw = mulmod(pw, 2, nm);
            }
            start += w;
        }
        v
    }
    /// One window block with multiplier h (h^-1 = hi) on (x, b).
    pub fn step(&self, h: u64, hi: u64, x: u64, b: u64) -> (u64, u64) {
        if self.c == 0 {
            return (mulmod(h, x, self.n_mod), 0);
        }
        let m = self.m() as u128;
        let x1 = ((b as u128 + self.a_sum(h, x) as u128) % m) as u64;
        let a2 = self.a_sum(hi, x1) as i128;
        let b1 = (x as i128 - a2).rem_euclid(m as i128) as u64;
        (x1, b1)
    }
    pub fn key(&self, x: u64, b: u64) -> u64 {
        x | (b << self.nr)
    }
    pub fn unkey(&self, k: u64) -> (u64, u64) {
        (k & (self.m() - 1), k >> self.nr)
    }
    /// Initial support S_0 = {(1 + jN, j'N)} (c > 0) or {1}.
    pub fn initial(&self) -> Vec<u64> {
        if self.c == 0 {
            return vec![1];
        }
        let mut v = Vec::new();
        for j in 0..1u64 << self.c {
            for j2 in 0..1u64 << self.c {
                v.push(self.key(1 + j * self.n_mod, j2 * self.n_mod));
            }
        }
        v
    }
    /// Multiplier of window k for digit e.
    pub fn mult(&self, k: usize, e: u64) -> (u64, u64) {
        let g = self.wins[k].2;
        let h = powmod(g, e, self.n_mod);
        (h, inv(h, self.n_mod))
    }
    /// Exponent weight of window k: digit e contributes e * 2^{t - i0 - w}.
    pub fn shift(&self, k: usize) -> usize {
        let (i0, w, _) = self.wins[k];
        self.t - i0 - w
    }
    /// Apply window k with digit e to a support.
    pub fn apply(&self, k: usize, e: u64, s: &[u64]) -> Vec<u64> {
        let (h, hi) = self.mult(k, e);
        s.iter()
            .map(|&key| {
                let (x, b) = if self.c == 0 {
                    (key, 0)
                } else {
                    self.unkey(key)
                };
                let (x1, b1) = self.step(h, hi, x, b);
                if self.c == 0 {
                    x1
                } else {
                    self.key(x1, b1)
                }
            })
            .collect()
    }
    /// All final supports S_E, indexed by E (E < 2^t).
    pub fn supports(&self) -> Vec<Vec<u64>> {
        let mut out = vec![Vec::new(); 1 << self.t];
        self.rec(0, 0, self.initial(), &mut out);
        out
    }
    fn rec(&self, k: usize, e_acc: u64, s: Vec<u64>, out: &mut [Vec<u64>]) {
        if k == self.wins.len() {
            out[e_acc as usize] = s;
            return;
        }
        let w = self.wins[k].1;
        for e in 0..1u64 << w {
            let s2 = self.apply(k, e, &s);
            self.rec(k + 1, e_acc | (e << self.shift(k)), s2, out);
        }
    }
}

/// Output distribution from supports (all of equal size) via the Gram
/// autocorrelation C(d).
pub fn distribution(t: usize, supp: &[Vec<u64>]) -> Vec<f64> {
    let tt = 1usize << t;
    let sz = supp[0].len();
    let mut pairs: Vec<u64> = Vec::with_capacity(tt * sz);
    for (e, s) in supp.iter().enumerate() {
        assert_eq!(s.len(), sz);
        for &k in s {
            pairs.push((k << t) | e as u64);
        }
    }
    pairs.sort_unstable();
    let mut cd = vec![0u64; tt];
    let mask = (tt - 1) as u64;
    let mut i = 0;
    while i < pairs.len() {
        let k = pairs[i] >> t;
        let mut j = i;
        while j < pairs.len() && pairs[j] >> t == k {
            j += 1;
        }
        let g = &pairs[i..j];
        for a in g {
            for b in g {
                cd[((a.wrapping_sub(*b)) & mask) as usize] += 1;
            }
        }
        i = j;
    }
    let norm = 1.0 / ((tt as f64) * (tt as f64) * sz as f64);
    let tw: Vec<f64> = (0..tt)
        .map(|k| (2.0 * std::f64::consts::PI * k as f64 / tt as f64).cos())
        .collect();
    let nz: Vec<(usize, f64)> = cd
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > 0)
        .map(|(d, &v)| (d, v as f64))
        .collect();
    (0..tt)
        .map(|y| {
            let mut s = 0.0;
            for &(d, v) in &nz {
                s += v * tw[(y * d) & (tt - 1)];
            }
            s * norm
        })
        .collect()
}

pub fn tv(p: &[f64], q: &[f64]) -> f64 {
    0.5 * p.iter().zip(q).map(|(a, b)| (a - b).abs()).sum::<f64>()
}

/// TV between coset (c) and exact output distributions.
pub fn tv_coset(n_mod: u64, a: u64, we: usize, wm: usize, c: usize) -> f64 {
    let mc = Model::new(n_mod, a, we, wm, c);
    let me = Model::new(n_mod, a, we, wm, 0);
    tv(
        &distribution(mc.t, &mc.supports()),
        &distribution(me.t, &me.supports()),
    )
}

// ------------------------------------------------------------------ plans
impl Model {
    /// General schedule: registers (len, base) run in order, each windowed
    /// with `we` like `shor_ge::run` (Shor: one register of 2n bits; EH:
    /// (m, y^-1) then (2m, g)).
    pub fn from_regs(n_mod: u64, regs: &[(usize, u64)], we: usize, wm: usize, c: usize) -> Self {
        let n = work_bits(n_mod);
        let mut wins = Vec::new();
        let mut t = 0;
        for &(len, base) in regs {
            let mut i0 = 0;
            while i0 < len {
                let w = we.min(len - i0);
                let mut g = base % n_mod;
                for _ in 0..len - i0 - w {
                    g = mulmod(g, g, n_mod);
                }
                wins.push((i0, w, g));
                i0 += w;
            }
            t += len;
        }
        Model {
            n_mod,
            n,
            c,
            nr: n + c,
            we,
            wm,
            t,
            wins,
        }
    }
}

/// Per-branch trajectory of the coset circuit along a digit sequence.
#[derive(Clone, Debug, Default)]
pub struct Walk {
    /// every lookup read a register congruent to the exact value
    pub faithful: bool,
    /// window index of the first unfaithful lookup
    pub first_bad: Option<usize>,
    /// per window: b register's coset index is outside [0, 2^c) after the
    /// window ("temporarily wrapped": b ≢ 0 mod N as an integer)
    pub b_out: Vec<bool>,
    /// per window: ge-shor's deviant test (b mod N != 0 or x mod N != exact)
    pub deviant: Vec<bool>,
    pub x: u64,
    pub b: u64,
    /// exact residue g^E
    pub u: u64,
    /// signed coset indices at the end (valid when faithful)
    pub jx: i64,
    pub jb: i64,
    /// per window: (J_x, J_b) after the window (signed indices relative to
    /// the exact residue / 0)
    pub traj: Vec<(i64, i64)>,
}

impl Model {
    /// Signed coset index J of value v for residue u: v ≡ u + J N (mod M),
    /// representative in [-M/2, M/2).
    pub fn index(&self, v: u64, u: u64) -> i64 {
        let m = self.m();
        let ninv = inv_pow2(self.n_mod, self.nr);
        let d = v.wrapping_sub(u) & (m - 1);
        let j = (u128::from(d) * u128::from(ninv) % u128::from(m)) as u64;
        if j >= m / 2 {
            j as i64 - m as i64
        } else {
            j as i64
        }
    }
    pub fn walk(&self, digits: &[u64], j: u64, j2: u64) -> Walk {
        let m = self.m();
        let nm = self.n_mod;
        let mut x = 1 + j * nm;
        let mut b = j2 * nm;
        let mut u = 1 % nm;
        let mut w = Walk {
            faithful: true,
            ..Default::default()
        };
        for (k, &e) in digits.iter().enumerate() {
            let (h, hi) = self.mult(k, e);
            if x % nm != u && w.faithful {
                w.faithful = false;
                w.first_bad = Some(k);
            }
            let x1 = ((u128::from(b) + u128::from(self.a_sum(h, x))) % u128::from(m)) as u64;
            let u1 = mulmod(h, u, nm);
            if x1 % nm != u1 && w.faithful {
                w.faithful = false;
                w.first_bad = Some(k);
            }
            let a2 = self.a_sum(hi, x1) as i128;
            b = (x as i128 - a2).rem_euclid(m as i128) as u64;
            x = x1;
            u = u1;
            let jb = self.index(b, 0);
            w.b_out.push(!(0..1i64 << self.c).contains(&jb));
            w.deviant.push(b % nm != 0 || x % nm != u);
            w.traj.push((self.index(x, u), jb));
        }
        w.x = x;
        w.b = b;
        w.u = u;
        w.jx = self.index(x, u);
        w.jb = self.index(b, 0);
        w
    }
    /// Digits of E for each window (Shor/EH layout: window k's digit is
    /// bits [shift, shift + w) of E).
    pub fn digits(&self, e: u64) -> Vec<u64> {
        (0..self.wins.len())
            .map(|k| (e >> self.shift_of(k)) & ((1 << self.wins[k].1) - 1))
            .collect()
    }
    /// bit offset of window k's digit in E (registers concatenated, first
    /// register in the low bits).
    pub fn shift_of(&self, k: usize) -> usize {
        // registers restart at i0 == 0; first register in the low bits
        let start = (0..=k).rev().find(|&i| self.wins[i].0 == 0).unwrap();
        let mut len = self.wins[start].1;
        let mut idx = start + 1;
        while idx < self.wins.len() && self.wins[idx].0 != 0 {
            len += self.wins[idx].1;
            idx += 1;
        }
        let off: usize = self.wins[..start].iter().map(|w| w.1).sum();
        let (i0, w, _) = self.wins[k];
        off + len - i0 - w
    }
}

/// N^{-1} mod 2^bits (N odd).
pub fn inv_pow2(n: u64, bits: usize) -> u64 {
    let mut x: u64 = 1;
    for _ in 0..7 {
        x = x.wrapping_mul(2u64.wrapping_sub(n.wrapping_mul(x)));
    }
    if bits >= 64 {
        x
    } else {
        x & ((1u64 << bits) - 1)
    }
}

// ----------------------------------------------------- Gram / TV lemma

/// Quantities of the TV lemma for a reference family {G_u}: per E,
/// delta_E = 1 - |S_E ∩ G_{u(E)}|/4^c and theta_E = |S_E ∩ ∪_{u'≠u(E)} G_u'|/4^c.
#[derive(Clone, Debug, Default)]
pub struct LemmaStats {
    pub delta_mean: f64,
    pub delta_rms: f64,
    pub delta_max: f64,
    pub theta_mean: f64,
    pub theta_rms: f64,
    /// δ_rms + θ_rms + δ̄ + sqrt(δ̄ θ̄)
    pub bound: f64,
}

pub fn lemma_from(deltas: &[f64], thetas: &[f64]) -> LemmaStats {
    let k = deltas.len() as f64;
    let dm = deltas.iter().sum::<f64>() / k;
    let dr = (deltas.iter().map(|d| d * d).sum::<f64>() / k).sqrt();
    let dx = deltas.iter().cloned().fold(0.0, f64::max);
    let tm = thetas.iter().sum::<f64>() / k;
    let tr = (thetas.iter().map(|d| d * d).sum::<f64>() / k).sqrt();
    LemmaStats {
        delta_mean: dm,
        delta_rms: dr,
        delta_max: dx,
        theta_mean: tm,
        theta_rms: tr,
        bound: dr + tr + dm + (dm * tm).sqrt(),
    }
}

impl Model {
    /// Exact class (g^E mod N) of every E.
    pub fn classes(&self) -> Vec<u64> {
        (0..1u64 << self.t)
            .map(|e| {
                let mut u = 1 % self.n_mod;
                for (k, &d) in self.digits(e).iter().enumerate() {
                    u = mulmod(self.mult(k, d).0, u, self.n_mod);
                }
                u
            })
            .collect()
    }
    /// Lemma stats for the shifted-square reference
    /// G_u = {(u + J N, J' N) mod M : J ∈ [ox, ox+2^c), J' ∈ [ob, ob+2^c)}.
    pub fn lemma_square(&self, supp: &[Vec<u64>], cls: &[u64], ox: i64, ob: i64) -> LemmaStats {
        let side = 1i64 << self.c;
        let q = 4f64.powi(self.c as i32);
        let mut ds = Vec::new();
        let mut ts = Vec::new();
        for (e, s) in supp.iter().enumerate() {
            let u = cls[e];
            let (mut good, mut other) = (0usize, 0usize);
            for &key in s {
                let (x, b) = self.unkey(key);
                let jb = self.index(b, 0);
                if !(ob..ob + side).contains(&jb) {
                    continue;
                }
                // which residue's square holds x? x = u' + J N with J in range
                let jx = self.index(x, u);
                if (ox..ox + side).contains(&jx) {
                    good += 1;
                } else if self.in_any_square(x, ox, side) {
                    other += 1;
                }
            }
            ds.push(1.0 - good as f64 / q);
            ts.push(other as f64 / q);
        }
        lemma_from(&ds, &ts)
    }
    /// x = u' + J N (mod M) for some residue u' and J in [ox, ox+side)?
    pub fn in_any_square(&self, x: u64, ox: i64, side: i64) -> bool {
        let m = self.m() as i128;
        let nm = self.n_mod as i128;
        for s in -1..=1i128 {
            let v = x as i128 + s * m;
            let j = v.div_euclid(nm);
            if (ox as i128..(ox + side) as i128).contains(&j) {
                return true;
            }
        }
        false
    }
    /// Lemma stats for the "majority" reference: G_u = the 4^c keys that
    /// occur in the most supports S_E of class u. Returns None if two
    /// classes' references intersect (lemma needs disjoint G_u).
    pub fn lemma_majority(&self, supp: &[Vec<u64>], cls: &[u64]) -> Option<LemmaStats> {
        use std::collections::HashMap;
        let q = 1usize << (2 * self.c);
        let mut per: HashMap<u64, HashMap<u64, u32>> = HashMap::new();
        for (e, s) in supp.iter().enumerate() {
            let h = per.entry(cls[e]).or_default();
            for &k in s {
                *h.entry(k).or_default() += 1;
            }
        }
        let mut owner: HashMap<u64, u64> = HashMap::new();
        for (&u, h) in &per {
            let mut v: Vec<(u32, u64)> = h.iter().map(|(&k, &c)| (c, k)).collect();
            v.sort_unstable_by(|a, b| b.cmp(a));
            for &(_, k) in v.iter().take(q) {
                if owner.insert(k, u).is_some() {
                    return None;
                }
            }
        }
        let mut ds = Vec::new();
        let mut ts = Vec::new();
        for (e, s) in supp.iter().enumerate() {
            let (mut good, mut other) = (0, 0);
            for k in s {
                match owner.get(k) {
                    Some(&u) if u == cls[e] => good += 1,
                    Some(_) => other += 1,
                    None => {}
                }
            }
            ds.push(1.0 - good as f64 / q as f64);
            ts.push(other as f64 / q as f64);
        }
        Some(lemma_from(&ds, &ts))
    }
}

impl Model {
    /// (number of keys shared by supports of different classes counted over
    /// ordered pairs of classes, number shared within a class by different E)
    pub fn cross_overlap(&self, supp: &[Vec<u64>], cls: &[u64]) -> (u64, u64) {
        use std::collections::HashMap;
        let mut owner: HashMap<u64, Vec<u64>> = HashMap::new();
        for (e, s) in supp.iter().enumerate() {
            for &k in s {
                owner.entry(k).or_default().push(cls[e]);
            }
        }
        let (mut cross, mut within) = (0u64, 0u64);
        for v in owner.values() {
            let mut cnt: HashMap<u64, u64> = HashMap::new();
            for &u in v {
                *cnt.entry(u).or_default() += 1;
            }
            let tot: u64 = v.len() as u64;
            for &c in cnt.values() {
                cross += c * (tot - c);
                within += c * (c - 1);
            }
        }
        (cross, within)
    }
}

impl Model {
    /// For r = ord(a) a power of two dividing 2^t, Phi_E depends only on
    /// E mod r (Theorem B1) and the output distribution is
    /// p(k 2^t / r) = f_k^† κ f_k / r^2 with κ(a,a') = |S_a ∩ S_a'| / 4^c;
    /// returns (TV to the exact uniform-on-multiples law, Σ_{a≠a'} κ(a,a'),
    /// max deviation of S_E from S_{E mod r}).
    pub fn pow2_tv(&self, r: u64) -> (f64, f64, usize) {
        use std::collections::HashMap;
        let supp = self.supports();
        let mut mism = 0;
        for (e, s) in supp.iter().enumerate() {
            let mut a = s.clone();
            let mut b = supp[e % r as usize].clone();
            a.sort_unstable();
            b.sort_unstable();
            if a != b {
                mism += 1;
            }
        }
        let q = 4f64.powi(self.c as i32);
        let mut owner: HashMap<u64, Vec<usize>> = HashMap::new();
        for a in 0..r as usize {
            for &k in &supp[a] {
                owner.entry(k).or_default().push(a);
            }
        }
        let ru = r as usize;
        let mut kappa = vec![0.0; ru * ru];
        for v in owner.values() {
            for &a in v {
                for &b in v {
                    kappa[a * ru + b] += 1.0 / q;
                }
            }
        }
        let off: f64 = (0..ru)
            .flat_map(|a| (0..ru).map(move |b| (a, b)))
            .filter(|(a, b)| a != b)
            .map(|(a, b)| kappa[a * ru + b])
            .sum();
        let mut tvv = 0.0;
        for k in 0..ru {
            let mut s = 0.0;
            for a in 0..ru {
                for b in 0..ru {
                    let ph =
                        2.0 * std::f64::consts::PI * (k as f64) * (a as f64 - b as f64) / r as f64;
                    s += kappa[a * ru + b] * ph.cos();
                }
            }
            tvv += (s / (r * r) as f64 - 1.0 / r as f64).abs();
        }
        (0.5 * tvv, off, mism)
    }
}
