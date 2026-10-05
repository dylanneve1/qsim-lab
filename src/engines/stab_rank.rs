//! Exact "branching-rank" simulator: the state is kept as a short sum of
//! stabilizer states `Ψ = Σ_j c_j |φ_j⟩`, every `|φ_j⟩` in the CH-form of
//! Bravyi, Browne, Calpin, Campbell, Gosset, Howard, "Simulation of quantum
//! circuits by low-rank stabilizer decompositions", Quantum 3, 181 (2019),
//! §4.1 (`|φ⟩ = ω U_C U_H |s⟩`). The CH-form update rules are a port of
//! Qiskit Aer's `chstabilizer.hpp` (Apache-2.0) to any number of qubits
//! (bitsets instead of 64-bit words).
//!
//! What is new here (research/theory/theory-rank.md) is how non-Clifford gates are
//! handled. Every non-Clifford gate of the families we simulate is a
//! *projector gate* `U = I + (λ−1)Π`, `Π = Π_{P_1}⋯Π_{P_m}` the joint +1
//! projector of commuting Hermitian Paulis (`T, Phase, Rz: Π = |1⟩⟨1|`,
//! `CPhase: |11⟩⟨11|`, `Toffoli: |11⟩⟨11|⊗|−⟩⟨−|`). On each term the gate is
//! classified exactly (Theorem R3): the factors are applied one by one to a
//! copy of `φ`; a factor that is deterministic on the current copy is
//! dropped (or kills `Πφ`), the others are "effective". With `m` effective
//! factors, `Uφ` is a stabilizer state iff `m = 0`, or `m = 1` and
//! `λ ∈ {±1, ±i}`, or `m = 2` and `λ = −1`; then the term is updated in
//! place by a Clifford. Only otherwise the term branches into `φ` and
//! `(λ−1)Πφ`. After a branching gate, terms that are the same ray are
//! merged (canonical signed stabilizer group as the key) and zero terms are
//! dropped. The number of terms after gate `k` is the branching rank `r_k`,
//! an upper bound on the stabilizer rank `χ(Ψ_k)`.

#![allow(clippy::needless_range_loop)]

use crate::circuit::Circuit;
use crate::gate::Gate;
use num_complex::Complex64 as C64;
use std::collections::HashMap;
use std::f64::consts::FRAC_PI_4;

// ---------------------------------------------------------------------------
// bitsets

#[inline]
fn get(b: &[u64], i: usize) -> bool {
    (b[i >> 6] >> (i & 63)) & 1 == 1
}
#[inline]
fn flip(b: &mut [u64], i: usize) {
    b[i >> 6] ^= 1 << (i & 63);
}
#[inline]
fn setb(b: &mut [u64], i: usize, v: bool) {
    if get(b, i) != v {
        flip(b, i);
    }
}
#[inline]
fn xor_into(a: &mut [u64], b: &[u64]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x ^= *y;
    }
}
#[inline]
fn popcount(a: &[u64]) -> u32 {
    a.iter().map(|x| x.count_ones()).sum()
}
#[inline]
fn popcount_and(a: &[u64], b: &[u64]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x & y).count_ones()).sum()
}
#[inline]
fn is_zero(a: &[u64]) -> bool {
    a.iter().all(|&x| x == 0)
}
fn first_one(a: &[u64]) -> Option<usize> {
    for (i, &w) in a.iter().enumerate() {
        if w != 0 {
            return Some(i * 64 + w.trailing_zeros() as usize);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// exact scalars eps · 2^{p/2} · e^{iπe/4}

/// An exact scalar `2^{p/2} · e^{iπe/4}`, or zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Scalar {
    /// The scalar is exactly zero (`p` and `e` are then meaningless).
    pub zero: bool,
    /// Magnitude exponent: the modulus is `2^{p/2}`.
    pub p: i32,
    /// Phase exponent in `0..8`: the phase is `e^{iπe/4}`.
    pub e: u8,
}

impl Scalar {
    /// The scalar 1.
    pub const ONE: Scalar = Scalar {
        zero: false,
        p: 0,
        e: 0,
    };
    /// The value as a floating-point complex number.
    pub fn to_c64(self) -> C64 {
        if self.zero {
            return C64::new(0.0, 0.0);
        }
        let m = 2f64.powf(self.p as f64 / 2.0);
        C64::from_polar(m, FRAC_PI_4 * self.e as f64)
    }
    fn mul(self, o: Scalar) -> Scalar {
        Scalar {
            zero: self.zero || o.zero,
            p: self.p + o.p,
            e: (self.e + o.e) % 8,
        }
    }
    fn conj(self) -> Scalar {
        Scalar {
            zero: self.zero,
            p: self.p,
            e: (8 - self.e) % 8,
        }
    }
}

/// `i^e X^x Z^z` on `n` qubits.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Pauli {
    /// X bits (bit `q` = qubit `q`).
    pub x: Vec<u64>,
    /// Z bits (bit `q` = qubit `q`).
    pub z: Vec<u64>,
    /// Phase exponent of `i^e`, in `0..4`.
    pub e: u8,
}

impl Pauli {
    /// The identity on `n` qubits (`e = 0`).
    pub fn identity(n: usize) -> Pauli {
        let w = n.div_ceil(64).max(1);
        Pauli {
            x: vec![0; w],
            z: vec![0; w],
            e: 0,
        }
    }
    /// `sign · Z_q` (sign = +1 or −1).
    pub fn z(n: usize, q: usize, neg: bool) -> Pauli {
        let mut p = Pauli::identity(n);
        flip(&mut p.z, q);
        p.e = if neg { 2 } else { 0 };
        p
    }
    /// `sign · X_q` (sign = +1 or −1).
    pub fn xp(n: usize, q: usize, neg: bool) -> Pauli {
        let mut p = Pauli::identity(n);
        flip(&mut p.x, q);
        p.e = if neg { 2 } else { 0 };
        p
    }
    /// `self ← self · rhs`.
    pub fn mul_assign(&mut self, rhs: &Pauli) {
        let overlap = popcount_and(&self.z, &rhs.x);
        xor_into(&mut self.x, &rhs.x);
        xor_into(&mut self.z, &rhs.z);
        self.e = ((self.e as u32 + rhs.e as u32 + 2 * overlap) % 4) as u8;
    }
}

// ---------------------------------------------------------------------------
// CH-form

/// One stabilizer state `ω U_C U_H |s⟩` (CH-form). Matrices `F, G, M` are
/// stored by columns (`f[j]` = column `j`, a bitset over rows), as in Aer.
/// Row `p` of `G` is the Z-image `U_C^{-1} Z_p U_C = Z^{G_p}`; row `p` of
/// `(F, M, γ)` gives `U_C^{-1} X_p U_C = i^{γ_p} X^{F_p} Z^{M_p}`.
#[derive(Clone, Debug)]
pub struct ChState {
    /// Number of qubits.
    pub n: usize,
    w: usize,
    f: Vec<u64>,
    g: Vec<u64>,
    m: Vec<u64>,
    g1: Vec<u64>,
    g2: Vec<u64>,
    v: Vec<u64>,
    s: Vec<u64>,
    /// The global scalar `ω`.
    pub omega: Scalar,
}

impl ChState {
    /// `|0^n⟩`.
    pub fn zero_state(n: usize) -> ChState {
        let w = n.div_ceil(64).max(1);
        let mut st = ChState {
            n,
            w,
            f: vec![0; n * w],
            g: vec![0; n * w],
            m: vec![0; n * w],
            g1: vec![0; w],
            g2: vec![0; w],
            v: vec![0; w],
            s: vec![0; w],
            omega: Scalar::ONE,
        };
        for q in 0..n {
            flip(&mut st.f[q * w..(q + 1) * w], q);
            flip(&mut st.g[q * w..(q + 1) * w], q);
        }
        st
    }

    #[inline]
    fn col(mat: &[u64], j: usize, w: usize) -> &[u64] {
        &mat[j * w..(j + 1) * w]
    }
    #[inline]
    fn bit(mat: &[u64], row: usize, col: usize, w: usize) -> bool {
        get(&mat[col * w..(col + 1) * w], row)
    }
    fn row(&self, mat: &[u64], r: usize) -> Vec<u64> {
        let mut out = vec![0u64; self.w];
        for j in 0..self.n {
            if Self::bit(mat, r, j, self.w) {
                flip(&mut out, j);
            }
        }
        out
    }
    #[inline]
    fn gamma(&self, q: usize) -> u32 {
        get(&self.g1, q) as u32 + 2 * get(&self.g2, q) as u32
    }

    // ---- right multiplication by C-type gates
    fn right_s(&mut self, q: usize) {
        let w = self.w;
        let fq = Self::col(&self.f, q, w).to_vec();
        xor_into(&mut self.m[q * w..(q + 1) * w], &fq);
        for i in 0..w {
            self.g2[i] ^= fq[i] & !self.g1[i];
            self.g1[i] ^= fq[i];
        }
    }
    fn right_cx(&mut self, q: usize, r: usize) {
        let w = self.w;
        let gr = Self::col(&self.g, r, w).to_vec();
        xor_into(&mut self.g[q * w..(q + 1) * w], &gr);
        let fq = Self::col(&self.f, q, w).to_vec();
        xor_into(&mut self.f[r * w..(r + 1) * w], &fq);
        let mr = Self::col(&self.m, r, w).to_vec();
        xor_into(&mut self.m[q * w..(q + 1) * w], &mr);
    }
    fn right_cz(&mut self, q: usize, r: usize) {
        let w = self.w;
        let fr = Self::col(&self.f, r, w).to_vec();
        let fq = Self::col(&self.f, q, w).to_vec();
        xor_into(&mut self.m[q * w..(q + 1) * w], &fr);
        xor_into(&mut self.m[r * w..(r + 1) * w], &fq);
        for i in 0..w {
            self.g2[i] ^= fq[i] & fr[i];
        }
    }

    // ---- left multiplication (gates)
    /// Left-multiplies by `S` on qubit `q`.
    pub fn s_gate(&mut self, q: usize) {
        let w = self.w;
        for p in 0..self.n {
            if Self::bit(&self.g, q, p, w) {
                flip(&mut self.m[p * w..(p + 1) * w], q);
            }
        }
        flip(&mut self.g1, q);
        if get(&self.g1, q) {
            flip(&mut self.g2, q);
        }
    }
    /// Left-multiplies by `S†` on qubit `q`.
    pub fn sdg_gate(&mut self, q: usize) {
        let w = self.w;
        for p in 0..self.n {
            if Self::bit(&self.g, q, p, w) {
                flip(&mut self.m[p * w..(p + 1) * w], q);
            }
        }
        if get(&self.g1, q) {
            flip(&mut self.g2, q);
        }
        flip(&mut self.g1, q);
    }
    /// Left-multiplies by `Z` on qubit `q`.
    pub fn z_gate(&mut self, q: usize) {
        flip(&mut self.g2, q);
    }
    /// Left-multiplies by `X` on qubit `q`.
    pub fn x_gate(&mut self, q: usize) {
        let xs = self.row(&self.f, q);
        let zs = self.row(&self.m, q);
        let mut phase = 2 * get(&self.g1, q) as u32 + 4 * get(&self.g2, q) as u32;
        let w = self.w;
        for i in 0..w {
            self.s[i] ^= zs[i] & self.v[i];
        }
        let mut par = 0u32;
        for i in 0..w {
            par += (zs[i] & !self.v[i] & self.s[i]).count_ones();
        }
        phase += 4 * (par & 1);
        for i in 0..w {
            self.s[i] ^= xs[i] & !self.v[i];
        }
        let mut par = 0u32;
        for i in 0..w {
            par += (xs[i] & self.v[i] & self.s[i]).count_ones();
        }
        phase += 4 * (par & 1);
        self.omega.e = ((self.omega.e as u32 + phase) % 8) as u8;
    }
    /// Left-multiplies by `Y` on qubit `q`.
    pub fn y_gate(&mut self, q: usize) {
        self.z_gate(q);
        self.x_gate(q);
        self.omega.e = (self.omega.e + 2) % 8;
    }
    /// Left-multiplies by CNOT with control `q` and target `r`.
    pub fn cx_gate(&mut self, q: usize, r: usize) {
        let w = self.w;
        let mut b = false;
        for p in 0..self.n {
            let (gp, fp, mp) = (p * w, p * w, p * w);
            b ^= get(&self.m[mp..mp + w], q) && get(&self.f[fp..fp + w], r);
            if get(&self.g[gp..gp + w], q) {
                flip(&mut self.g[gp..gp + w], r);
            }
            if get(&self.f[fp..fp + w], r) {
                flip(&mut self.f[fp..fp + w], q);
            }
            if get(&self.m[mp..mp + w], r) {
                flip(&mut self.m[mp..mp + w], q);
            }
        }
        if b {
            flip(&mut self.g2, q);
        }
        let carry = get(&self.g1, q) && get(&self.g1, r);
        if get(&self.g1, r) {
            flip(&mut self.g1, q);
        }
        if get(&self.g2, r) {
            flip(&mut self.g2, q);
        }
        if carry {
            flip(&mut self.g2, q);
        }
    }
    /// Left-multiplies by CZ on qubits `q` and `r`.
    pub fn cz_gate(&mut self, q: usize, r: usize) {
        let w = self.w;
        for p in 0..self.n {
            let gp = p * w;
            let gr = get(&self.g[gp..gp + w], r);
            let gq = get(&self.g[gp..gp + w], q);
            if gr {
                flip(&mut self.m[gp..gp + w], q);
            }
            if gq {
                flip(&mut self.m[gp..gp + w], r);
            }
        }
    }
    /// Left-multiplies by `H` on qubit `q`.
    pub fn h_gate(&mut self, q: usize) {
        let rf = self.row(&self.f, q);
        let rg = self.row(&self.g, q);
        let rm = self.row(&self.m, q);
        let w = self.w;
        let mut t = self.s.clone();
        let mut u = self.s.clone();
        let mut alpha = 0u32;
        let mut beta = 0u32;
        for i in 0..w {
            let (v, s) = (self.v[i], self.s[i]);
            t[i] ^= rg[i] & v;
            u[i] ^= (rf[i] & !v) ^ (rm[i] & v);
            alpha += (rg[i] & !v & s).count_ones();
            beta += ((rm[i] & !v & s) ^ (rf[i] & v & (rm[i] ^ s))).count_ones();
        }
        if alpha % 2 == 1 {
            self.omega.e = (self.omega.e + 4) % 8;
        }
        let b = (self.gamma(q) + 2 * alpha + 2 * beta) % 4;
        if t == u {
            self.s = t;
            assert!(b == 1 || b == 3, "CH-form H: unnormalised state");
            self.omega.e = (self.omega.e + if b == 1 { 1 } else { 7 }) % 8;
        } else {
            self.update_svector(&t, &u, b);
        }
    }

    /// Replaces `|s⟩` by `(|t⟩ + i^b |u⟩)/√2` (Bravyi et al. 2019, Prop. 4).
    fn update_svector(&mut self, t: &[u64], u: &[u64], b: u32) {
        let mut b = b % 4;
        if t == u {
            self.s = t.to_vec();
            match b {
                0 => self.omega.p += 1,
                1 => self.omega.e = (self.omega.e + 1) % 8,
                2 => self.omega.zero = true,
                _ => self.omega.e = (self.omega.e + 7) % 8,
            }
            return;
        }
        let w = self.w;
        let mut nu0 = vec![0u64; w];
        let mut nu1 = vec![0u64; w];
        for i in 0..w {
            let ut = u[i] ^ t[i];
            nu0[i] = ut & !self.v[i];
            nu1[i] = ut & self.v[i];
        }
        let q;
        if let Some(q0) = first_one(&nu0) {
            q = q0;
            flip(&mut nu0, q);
            for q1 in q + 1..self.n {
                if get(&nu0, q1) {
                    self.right_cx(q, q1);
                }
            }
            for q1 in 0..self.n {
                if get(&nu1, q1) {
                    self.right_cz(q, q1);
                }
            }
        } else {
            q = first_one(&nu1).expect("t != u");
            flip(&mut nu1, q);
            for q1 in q + 1..self.n {
                if get(&nu1, q1) {
                    self.right_cx(q1, q);
                }
            }
        }
        if get(t, q) {
            self.s = u.to_vec();
            self.omega.e = ((self.omega.e as u32 + 2 * b) % 8) as u8;
            b = (4 - b) % 4;
        } else {
            self.s = t.to_vec();
        }
        let a = get(&self.v, q);
        let e1 = if a {
            (b % 2) * (3 * b).wrapping_sub(2)
        } else {
            0
        };
        let e2 = b % 2 == 1;
        let e3 = !a || b % 2 == 1; // (!a) != (a && b odd)
        let e4 = ((!a) && b >= 2) != (a && (b == 1 || b == 2));
        setb(&mut self.s, q, e4);
        setb(&mut self.v, q, e3);
        self.omega.e = ((self.omega.e as u32 + e1) % 8) as u8;
        if e2 {
            self.right_s(q);
        }
    }

    /// All rows of a column-stored matrix (transpose), `O(n²/64)` words.
    fn rows_of(&self, mat: &[u64]) -> Vec<Vec<u64>> {
        let (n, w) = (self.n, self.w);
        let mut out = vec![vec![0u64; w]; n];
        for j in 0..n {
            let col = Self::col(mat, j, w);
            for (wi, &word) in col.iter().enumerate() {
                let mut wd = word;
                while wd != 0 {
                    let b = wd.trailing_zeros() as usize;
                    wd &= wd - 1;
                    let r = wi * 64 + b;
                    if r < n {
                        out[r][j >> 6] |= 1 << (j & 63);
                    }
                }
            }
        }
        out
    }

    /// `U_C^{-1} X(x) U_C` with precomputed rows of `F` and `M`.
    fn pauli_x_image_rows(&self, x: &[u64], rf: &[Vec<u64>], rm: &[Vec<u64>]) -> Pauli {
        let mut r = Pauli::identity(self.n);
        for pos in 0..self.n {
            if get(x, pos) {
                let ov = popcount_and(&r.z, &rf[pos]);
                xor_into(&mut r.x, &rf[pos]);
                xor_into(&mut r.z, &rm[pos]);
                r.e = ((r.e as u32 + self.gamma(pos) + 2 * ov) % 4) as u8;
            }
        }
        r
    }

    /// `U_C^{-1} X(x) U_C`.
    fn pauli_x_image(&self, x: &[u64]) -> Pauli {
        let mut r = Pauli::identity(self.n);
        for pos in 0..self.n {
            if get(x, pos) {
                let p1 = Pauli {
                    x: self.row(&self.f, pos),
                    z: self.row(&self.m, pos),
                    e: self.gamma(pos) as u8,
                };
                r.mul_assign(&p1);
            }
        }
        r
    }

    /// `R = U_H U_C^{-1} P U_C U_H` (the Pauli as seen by `|s⟩`).
    fn conj_to_s(&self, pp: &Pauli) -> Pauli {
        let w = self.w;
        let mut r = Pauli {
            x: vec![0; w],
            z: vec![0; w],
            e: pp.e,
        };
        for j in 0..self.n {
            if get(&pp.x, j) {
                let rf = self.row(&self.f, j);
                let rm = self.row(&self.m, j);
                let sgn = popcount_and(&r.z, &rf);
                r.e = ((r.e as u32 + 2 * sgn + self.gamma(j)) % 4) as u8;
                xor_into(&mut r.z, &rm);
                xor_into(&mut r.x, &rf);
            }
        }
        for q in 0..self.n {
            if popcount_and(&pp.z, Self::col(&self.g, q, w)) % 2 == 1 {
                flip(&mut r.z, q);
            }
        }
        let mut tx = vec![0u64; w];
        let mut tz = vec![0u64; w];
        let mut y = 0u32;
        for i in 0..w {
            let v = self.v[i];
            tx[i] = (!v & r.x[i]) ^ (v & r.z[i]);
            tz[i] = (!v & r.z[i]) ^ (v & r.x[i]);
            y += (v & r.x[i] & r.z[i]).count_ones();
        }
        r.e = ((r.e as u32 + 2 * y) % 4) as u8;
        r.x = tx;
        r.z = tz;
        r
    }

    /// Deterministic eigenvalue of the Hermitian Pauli `P`, if any.
    pub fn eigenvalue(&self, p: &Pauli) -> Option<bool> {
        let r = self.conj_to_s(p);
        Self::eig_of(&r, &self.s)
    }
    fn eig_of(r: &Pauli, s: &[u64]) -> Option<bool> {
        if !is_zero(&r.x) {
            return None;
        }
        let b = (r.e as u32 + 2 * popcount_and(&r.z, s)) % 4;
        debug_assert!(b.is_multiple_of(2));
        Some(b == 2) // true = eigenvalue −1
    }

    /// `φ ← (I + i^c P) φ / √2` (P Hermitian).
    fn combo_r(&mut self, r: &Pauli, c: u32) {
        let b = (r.e as u32 + c + 2 * popcount_and(&r.z, &self.s)) % 4;
        let mut u = self.s.clone();
        xor_into(&mut u, &r.x);
        let t = self.s.clone();
        self.update_svector(&t, &u, b);
    }
    /// `φ ← (I + i^c P) φ / √2` for a Hermitian Pauli `P`.
    pub fn combo(&mut self, p: &Pauli, c: u32) {
        let r = self.conj_to_s(p);
        self.combo_r(&r, c);
    }
    /// `φ ← (I + P)/2 φ` (may become the zero vector, `omega.zero`).
    pub fn project(&mut self, p: &Pauli) {
        self.combo(p, 0);
        self.omega.p -= 1;
    }
    /// `φ ← P φ`.
    pub fn apply_pauli(&mut self, p: &Pauli) {
        let r = self.conj_to_s(p);
        let k = (r.e as u32 + 2 * popcount_and(&r.z, &self.s)) % 4;
        xor_into(&mut self.s, &r.x);
        self.omega.e = ((self.omega.e as u32 + 2 * k) % 8) as u8;
    }

    /// Exact amplitude `⟨x|φ⟩`.
    pub fn amplitude(&self, x: &[u64]) -> Scalar {
        if self.omega.zero {
            return self.omega;
        }
        let pp = if popcount(x) > 4 {
            let (rf, rm) = (self.rows_of(&self.f), self.rows_of(&self.m));
            self.pauli_x_image_rows(x, &rf, &rm)
        } else {
            self.pauli_x_image(x)
        };
        let mut amp = Scalar {
            zero: false,
            p: -(popcount(&self.v) as i32),
            e: (2 * pp.e) % 8,
        };
        for q in 0..self.n {
            if get(&self.v, q) {
                if get(&self.s, q) && get(&pp.x, q) {
                    amp.e = (amp.e + 4) % 8;
                }
            } else if get(&pp.x, q) != get(&self.s, q) {
                return Scalar {
                    zero: true,
                    p: 0,
                    e: 0,
                };
            }
        }
        amp.conj().mul(self.omega)
    }
    /// `⟨idx|φ⟩` for a basis index that fits in one word (bit `q` = qubit `q`,
    /// so only qubits below 64 can be set).
    pub fn amplitude_index(&self, idx: usize) -> C64 {
        let mut x = vec![0u64; self.w];
        x[0] = idx as u64;
        self.amplitude(&x).to_c64()
    }

    /// A basis state in the support (the `w = 0` sample of Aer's sampler).
    pub fn support_point(&self) -> Vec<u64> {
        let mut x = vec![0u64; self.w];
        for q in 0..self.n {
            if get(&self.s, q) {
                xor_into(&mut x, Self::col(&self.g, q, self.w));
            }
        }
        x
    }

    /// Canonical key of the ray `C·φ`: the signed stabilizer group in fully
    /// reduced row echelon form (rows `x|z` plus the phase), `O(n³/64)`.
    pub fn canonical_key(&self) -> Vec<u64> {
        let tab = self.stabilizer_rref();
        let mut key = Vec::with_capacity(self.n * (2 * self.w + 1));
        for (row, e) in tab {
            key.extend_from_slice(&row);
            key.push(e as u64);
        }
        key
    }

    /// Signed stabilizer generators `(x|z bits, i^e)` in fully reduced row
    /// echelon form (canonical for the ray).
    pub fn stabilizer_rref(&self) -> Vec<(Vec<u64>, u8)> {
        let (n, w) = (self.n, self.w);
        let rows_g = self.rows_of(&self.g);
        let rows_f = self.rows_of(&self.f);
        let rows_m = self.rows_of(&self.m);
        let ginv = gf2_inverse(&rows_g, n);
        let finv = gf2_inverse(&rows_f, n);
        // generators: rows of the 2n-bit tableau, phase e (i^e)
        let mut tab: Vec<(Vec<u64>, u8)> = Vec::with_capacity(n);
        for k in 0..n {
            let neg = get(&self.s, k);
            let mut row = vec![0u64; 2 * w];
            let e: u8 = if !get(&self.v, k) {
                row[w..].copy_from_slice(&ginv[k]);
                0
            } else {
                let x = &finv[k];
                let img = self.pauli_x_image_rows(x, &rows_f, &rows_m);
                // img = i^e X_k Z^z  ⇒  U_C X_k U_C^{-1} = i^{-e} X^x Z^{z G^{-1}}
                let mut zz = vec![0u64; w];
                for p in 0..n {
                    if get(&img.z, p) {
                        xor_into(&mut zz, &ginv[p]);
                    }
                }
                row[..w].copy_from_slice(x);
                row[w..].copy_from_slice(&zz);
                ((4 - img.e as u32) % 4) as u8
            };
            let e = (e + if neg { 2 } else { 0 }) % 4;
            tab.push((row, e));
        }
        // RREF with phase-correct Pauli products
        let mul = |a: &mut (Vec<u64>, u8), b: &(Vec<u64>, u8)| {
            // a ← a·b ;  sign from moving b.x left past a.z
            let ov = popcount_and(&a.0[w..], &b.0[..w]);
            for i in 0..2 * w {
                a.0[i] ^= b.0[i];
            }
            a.1 = ((a.1 as u32 + b.1 as u32 + 2 * ov) % 4) as u8;
        };
        let mut rank = 0;
        for col in 0..2 * n {
            let bit = if col < n { col } else { w * 64 + (col - n) };
            let Some(piv) = (rank..n).find(|&r| get(&tab[r].0, bit)) else {
                continue;
            };
            tab.swap(rank, piv);
            let pr = tab[rank].clone();
            for r in 0..n {
                if r != rank && get(&tab[r].0, bit) {
                    // keep a Hermitian representative: multiply as group
                    // elements (commuting), the product is Hermitian.
                    mul(&mut tab[r], &pr);
                }
            }
            rank += 1;
        }
        debug_assert_eq!(rank, n);
        tab
    }
}

fn gf2_inverse(rows: &[Vec<u64>], n: usize) -> Vec<Vec<u64>> {
    let w = n.div_ceil(64).max(1);
    let mut a: Vec<Vec<u64>> = rows.to_vec();
    let mut inv: Vec<Vec<u64>> = (0..n)
        .map(|i| {
            let mut r = vec![0u64; w];
            flip(&mut r, i);
            r
        })
        .collect();
    for c in 0..n {
        let piv = (c..n).find(|&r| get(&a[r], c)).expect("singular CH matrix");
        a.swap(c, piv);
        inv.swap(c, piv);
        let (ac, ic) = (a[c].clone(), inv[c].clone());
        for r in 0..n {
            if r != c && get(&a[r], c) {
                xor_into(&mut a[r], &ac);
                xor_into(&mut inv[r], &ic);
            }
        }
    }
    inv
}

// ---------------------------------------------------------------------------
// Clifford conjugation of Paulis, diagonalisation, pair merges

type Row = (Vec<u64>, u8);

fn row_pauli(r: &Row, w: usize) -> Pauli {
    Pauli {
        x: r.0[..w].to_vec(),
        z: r.0[w..].to_vec(),
        e: r.1,
    }
}

/// `p ← G p G†` for a Clifford gate `G`.
pub fn conj_gate(p: &mut Pauli, g: &Gate) {
    use Gate::*;
    let add = |p: &mut Pauli, k: u8| p.e = (p.e + k) % 4;
    match *g {
        H(q) => {
            let (x, z) = (get(&p.x, q), get(&p.z, q));
            setb(&mut p.x, q, z);
            setb(&mut p.z, q, x);
            if x && z {
                add(p, 2);
            }
        }
        S(q) => {
            if get(&p.x, q) {
                flip(&mut p.z, q);
                add(p, 1);
            }
        }
        Sdg(q) => {
            if get(&p.x, q) {
                flip(&mut p.z, q);
                add(p, 3);
            }
        }
        X(q) => {
            if get(&p.z, q) {
                add(p, 2);
            }
        }
        Z(q) => {
            if get(&p.x, q) {
                add(p, 2);
            }
        }
        Cnot(c, t) => {
            if get(&p.x, c) {
                flip(&mut p.x, t);
            }
            if get(&p.z, t) {
                flip(&mut p.z, c);
            }
        }
        Cz(a, b) => {
            let (xa, xb) = (get(&p.x, a), get(&p.x, b));
            if xa {
                flip(&mut p.z, b);
            }
            if xb {
                flip(&mut p.z, a);
            }
            if xa && xb {
                add(p, 2);
            }
        }
        Swap(a, b) => {
            conj_gate(p, &Cnot(a, b));
            conj_gate(p, &Cnot(b, a));
            conj_gate(p, &Cnot(a, b));
        }
        Y(q) => {
            if get(&p.x, q) != get(&p.z, q) {
                add(p, 2);
            }
        }
        I(_) => {}
        ref other => panic!("conj_gate: {other:?}"),
    }
}

impl ChState {
    /// Applies a Clifford gate from {H, S, Sdg, X, Y, Z, CNOT, CZ}.
    pub fn apply_clifford(&mut self, g: &Gate) {
        use Gate::*;
        match *g {
            H(q) => self.h_gate(q),
            S(q) => self.s_gate(q),
            Sdg(q) => self.sdg_gate(q),
            X(q) => self.x_gate(q),
            Y(q) => self.y_gate(q),
            Z(q) => self.z_gate(q),
            Cnot(a, b) => self.cx_gate(a, b),
            Cz(a, b) => self.cz_gate(a, b),
            Swap(a, b) => {
                self.cx_gate(a, b);
                self.cx_gate(b, a);
                self.cx_gate(a, b);
            }
            I(_) => {}
            ref other => panic!("apply_clifford: {other:?}"),
        }
    }
}

/// Clifford circuit `V` (time order) with `V g_k V† = +Z_{p_k}` for
/// independent commuting Hermitian Paulis `g_k`; returns `(V, pivots)`.
pub fn diagonalise(gens: &[Pauli], n: usize) -> (Vec<Gate>, Vec<usize>) {
    let mut gs: Vec<Pauli> = gens.to_vec();
    let mut gates = Vec::new();
    let mut piv: Vec<usize> = Vec::new();
    let mut used = vec![false; n];
    for k in 0..gs.len() {
        // remove Z's on earlier pivots using earlier (now +Z_p) generators
        for (kk, &p) in piv.iter().enumerate() {
            if get(&gs[k].z, p) {
                let pk = gs[kk].clone();
                gs[k].mul_assign(&pk);
            }
        }
        let g = gs[k].clone();
        let mut local: Vec<Gate> = Vec::new();
        let xq = (0..n).find(|&q| !used[q] && get(&g.x, q));
        let q;
        if let Some(x0) = xq {
            q = x0;
            for r in 0..n {
                if r != q && !used[r] && get(&g.x, r) {
                    local.push(Gate::Cnot(q, r));
                }
            }
        } else {
            q = (0..n)
                .find(|&q| !used[q] && get(&g.z, q))
                .expect("dependent generator");
            for r in 0..n {
                if r != q && !used[r] && get(&g.z, r) {
                    local.push(Gate::Cnot(r, q));
                }
            }
        }
        let apply = |gl: &Gate, gs: &mut Vec<Pauli>, gates: &mut Vec<Gate>| {
            for p in gs.iter_mut() {
                conj_gate(p, gl);
            }
            gates.push(*gl);
        };
        for gl in &local {
            apply(gl, &mut gs, &mut gates);
        }
        if xq.is_some() {
            if get(&gs[k].z, q) {
                apply(&Gate::S(q), &mut gs, &mut gates);
            }
            let zs: Vec<usize> = (0..n)
                .filter(|&r| r != q && !used[r] && get(&gs[k].z, r))
                .collect();
            for r in zs {
                apply(&Gate::Cz(q, r), &mut gs, &mut gates);
            }
            apply(&Gate::H(q), &mut gs, &mut gates);
        }
        if gs[k].e == 2 {
            apply(&Gate::X(q), &mut gs, &mut gates);
        }
        debug_assert!(gs[k].e == 0 && get(&gs[k].z, q) && popcount(&gs[k].x) == 0);
        used[q] = true;
        piv.push(q);
    }
    (gates, piv)
}

/// Basis of the signed common stabilizer group of two states, given their
/// RREF tableaux (as products of rows of `ti`).
fn common_stabilizers(ti: &[Row], tj: &[Row], phi_j: &ChState, n: usize, w: usize) -> Vec<Pauli> {
    // basis rows: (bits, pivot bit) from tj, then reduce ti rows with tags
    let width = 2 * w;
    let mut basis: Vec<(Vec<u64>, usize, Vec<u64>)> = Vec::new(); // bits, pivot, tag
    let tw = n.div_ceil(64).max(1);
    let pivot_of = |b: &[u64]| first_one(b);
    for r in tj {
        let mut b = r.0.clone();
        for (bb, p, _) in &basis {
            if get(&b, *p) {
                xor_into(&mut b, bb);
            }
        }
        if let Some(p) = pivot_of(&b) {
            basis.push((b, p, vec![0u64; tw]));
        }
    }
    let mut zero_tags: Vec<Vec<u64>> = Vec::new();
    for (k, r) in ti.iter().enumerate() {
        let mut b = r.0.clone();
        let mut tag = vec![0u64; tw];
        flip(&mut tag, k);
        for (bb, p, tg) in &basis {
            if get(&b, *p) {
                xor_into(&mut b, bb);
                xor_into(&mut tag, tg);
            }
        }
        match pivot_of(&b[..width]) {
            Some(p) => basis.push((b, p, tag)),
            None => zero_tags.push(tag),
        }
    }
    let mut out: Vec<Pauli> = zero_tags
        .iter()
        .map(|tag| {
            let mut p = Pauli::identity(n);
            for k in 0..n {
                if get(tag, k) {
                    p.mul_assign(&row_pauli(&ti[k], w));
                }
            }
            p
        })
        .collect();
    // keep the subgroup with eigenvalue +1 on φ_j
    let bad: Vec<usize> = (0..out.len())
        .filter(|&k| phi_j.eigenvalue(&out[k]) == Some(true))
        .collect();
    if let Some(&b0) = bad.first() {
        let pb = out[b0].clone();
        for &k in &bad[1..] {
            out[k].mul_assign(&pb);
        }
        out.remove(b0);
    }
    out
}

/// `rank(⟨ta_a, tb_b⟩_symplectic) > cap`, with early exit.
fn pairing_rank_exceeds(ta: &[Row], tb: &[Row], w: usize, cap: usize) -> bool {
    let n = tb.len();
    let wn = n.div_ceil(64).max(1);
    let mut basis: Vec<(Vec<u64>, usize)> = Vec::new();
    for ra in ta {
        let mut row = vec![0u64; wn];
        for (b, rb) in tb.iter().enumerate() {
            let c = popcount_and(&ra.0[..w], &rb.0[w..]) + popcount_and(&ra.0[w..], &rb.0[..w]);
            if c & 1 == 1 {
                row[b >> 6] |= 1 << (b & 63);
            }
        }
        for (bv, p) in &basis {
            if get(&row, *p) {
                xor_into(&mut row, bv);
            }
        }
        if let Some(p) = first_one(&row) {
            basis.push((row, p));
            if basis.len() > cap {
                return true;
            }
        }
    }
    false
}

/// Tries to replace `ci φi + cj φj` by one term (Some(Some)) or nothing
/// (Some(None), cancellation).
fn try_pair(
    a: &Term,
    b: &Term,
    ta: &[Row],
    tb: &[Row],
    smax: usize,
    tol: f64,
) -> Option<Option<Term>> {
    let n = a.st.n;
    let w = a.st.w;
    // cheap exact filter: for Lagrangian L_a, L_b, dim(L_a ∩ L_b) =
    // n − rank(symplectic pairing matrix); stop as soon as it exceeds smax
    if pairing_rank_exceeds(ta, tb, w, smax) {
        return None;
    }
    let common = common_stabilizers(ta, tb, &b.st, n, w);
    let s = n - common.len();
    if s > smax || s == 0 {
        return None;
    }
    let (v, piv) = diagonalise(&common, n);
    let mut is_piv = vec![false; n];
    for &p in &piv {
        is_piv[p] = true;
    }
    let free: Vec<usize> = (0..n).filter(|&q| !is_piv[q]).collect();
    debug_assert_eq!(free.len(), s);
    let mut sa = a.st.clone();
    let mut sb = b.st.clone();
    for g in &v {
        sa.apply_clifford(g);
        sb.apply_clifford(g);
    }
    let dim = 1usize << s;
    let mut vec_sum = vec![C64::new(0.0, 0.0); dim];
    for y in 0..dim {
        let mut x = vec![0u64; w];
        for (i, &q) in free.iter().enumerate() {
            if (y >> i) & 1 == 1 {
                flip(&mut x, q);
            }
        }
        vec_sum[y] = a.c * sa.amplitude(&x).to_c64() + b.c * sb.amplitude(&x).to_c64();
    }
    let nrm = vec_sum.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    let wmax = a.weight().max(b.weight());
    if nrm <= tol * wmax {
        return Some(None);
    }
    let unit: Vec<C64> = vec_sum.iter().map(|z| z / nrm).collect();
    let k = crate::magic_atlas::stabilizer_synth(&unit, s)?;
    let mut st = ChState::zero_state(n);
    for g in &k {
        let gm = crate::magic_atlas::families::remap(g, &|i| free[i]);
        for h in gm.decompose_to_clifford_rz() {
            st.apply_clifford(&h);
        }
    }
    // exactness guard: the synthesiser accepts states within ~1e-9 of a
    // stabilizer state; require agreement to rounding level
    let mut ov = C64::new(0.0, 0.0);
    let mut dense = vec![C64::new(0.0, 0.0); dim];
    for y in 0..dim {
        let mut x = vec![0u64; w];
        for (i, &q) in free.iter().enumerate() {
            if (y >> i) & 1 == 1 {
                flip(&mut x, q);
            }
        }
        dense[y] = st.amplitude(&x).to_c64();
        ov += dense[y].conj() * unit[y];
    }
    let ph = ov / ov.norm();
    let dev = dense
        .iter()
        .zip(&unit)
        .map(|(d, u)| (d * ph - u).norm())
        .fold(0.0, f64::max);
    if dev > 1e-13 {
        return None;
    }
    for g in v.iter().rev() {
        st.apply_clifford(&g.inverse());
    }
    let x0 = st.support_point();
    let truth = a.c * a.st.amplitude(&x0).to_c64() + b.c * b.st.amplitude(&x0).to_c64();
    let mine = st.amplitude(&x0).to_c64();
    assert!(mine.norm() > 0.0);
    Some(Some(Term {
        c: truth / mine,
        st,
        dirty: true,
    }))
}

// ---------------------------------------------------------------------------
// the sum

/// One term `c · |φ⟩` of the sum (the full coefficient is `c · ω`).
#[derive(Clone, Debug)]
pub struct Term {
    /// Floating-point coefficient (multiplies the CH-form's own `ω`).
    pub c: C64,
    /// The stabilizer state in CH-form.
    pub st: ChState,
    /// changed individually since the last pair-merge pass
    pub dirty: bool,
}

impl Term {
    fn weight(&self) -> f64 {
        self.c.norm() * self.st.omega.to_c64().norm()
    }
}

/// How a projector gate acted on one term.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// `Πφ = 0` or `Πφ = φ` (phase)
    Diagonal,
    /// `Uφ` stabilizer, updated by a Clifford
    Clifford,
    /// split into `φ` and `(λ−1)Πφ`
    Branch,
}

/// Counters of a [`RankState`] run.
#[derive(Clone, Debug, Default)]
pub struct RankStats {
    /// number of terms after every original gate
    pub r: Vec<usize>,
    /// Largest number of terms after any gate.
    pub max_r: usize,
    /// Term updates where a projector gate branched the term in two.
    pub branch_events: usize,
    /// Term updates where a projector gate acted as a Clifford.
    pub clifford_events: usize,
    /// Term updates where a projector gate acted diagonally (`Πφ = 0` or `φ`).
    pub diag_events: usize,
    /// Terms folded into another term of the same ray.
    pub merges: usize,
    /// Pairs of terms merged into one (or cancelled) by the pair merge.
    pub pair_merges: usize,
    /// seconds in the proportional (same-ray) merge
    pub t_merge: f64,
    /// seconds in the pair merge
    pub t_pair: f64,
    /// Pairs of terms tested for a pair merge.
    pub pair_tests: usize,
    /// Terms dropped because their merged weight cancelled to below
    /// `zero_tol` (relative to the largest contribution).
    pub cancellations: usize,
    /// set when `max_terms` was exceeded (simulation stopped)
    pub overflow: bool,
}

/// The state as a sum of CH-form stabilizer states (see the module docs).
#[derive(Clone, Debug)]
pub struct RankState {
    /// Number of qubits.
    pub n: usize,
    /// The terms `Σ_j c_j |φ_j⟩` (zero terms are dropped after merges).
    pub terms: Vec<Term>,
    /// Counters so far.
    pub stats: RankStats,
    /// stop (overflow) when the number of terms would exceed this
    pub max_terms: usize,
    /// relative weight below which a merged term is treated as zero
    pub zero_tol: f64,
    /// pair merges are attempted when two terms share all but `s ≤
    /// pair_merge_s` stabilizer generators (0 disables them)
    pub pair_merge_s: usize,
    /// pair merges are skipped while the rank exceeds this (cost is
    /// O(dirty · r) pair tests)
    pub pair_merge_max_r: usize,
}

fn snap(l: C64) -> C64 {
    for t in [
        C64::new(1.0, 0.0),
        C64::new(-1.0, 0.0),
        C64::new(0.0, 1.0),
        C64::new(0.0, -1.0),
    ] {
        if (l - t).norm() < 1e-12 {
            return t;
        }
    }
    l
}

impl RankState {
    /// `|0^n⟩` as one term, with the default limits (`max_terms = 2^16`,
    /// `zero_tol = 1e-12`, pair merges at `s ≤ 6` while `r ≤ 256`).
    pub fn new(n: usize) -> RankState {
        RankState {
            n,
            terms: vec![Term {
                c: C64::new(1.0, 0.0),
                st: ChState::zero_state(n),
                dirty: false,
            }],
            pair_merge_s: 6,
            pair_merge_max_r: 256,
            stats: RankStats::default(),
            max_terms: 1 << 16,
            zero_tol: 1e-12,
        }
    }

    /// Current number of terms (the branching rank `r`).
    pub fn rank(&self) -> usize {
        self.terms.len()
    }

    fn each(&mut self, f: impl Fn(&mut ChState)) {
        for t in &mut self.terms {
            f(&mut t.st);
        }
    }

    /// Applies one gate. Returns `false` on overflow.
    pub fn apply(&mut self, g: &Gate) -> bool {
        use Gate::*;
        let n = self.n;
        match *g {
            I(_) => {}
            H(q) => self.each(|s| s.h_gate(q)),
            X(q) => self.each(|s| s.x_gate(q)),
            Y(q) => self.each(|s| s.y_gate(q)),
            Z(q) => self.each(|s| s.z_gate(q)),
            S(q) => self.each(|s| s.s_gate(q)),
            Sdg(q) => self.each(|s| s.sdg_gate(q)),
            Cnot(a, b) => self.each(|s| s.cx_gate(a, b)),
            Cz(a, b) => self.each(|s| s.cz_gate(a, b)),
            Swap(a, b) => self.each(|s| {
                s.cx_gate(a, b);
                s.cx_gate(b, a);
                s.cx_gate(a, b);
            }),
            T(q) => {
                return self.projector_gate(
                    C64::from_polar(1.0, FRAC_PI_4),
                    &[Pauli::z(n, q, true)],
                    C64::new(1.0, 0.0),
                )
            }
            Tdg(q) => {
                return self.projector_gate(
                    C64::from_polar(1.0, -FRAC_PI_4),
                    &[Pauli::z(n, q, true)],
                    C64::new(1.0, 0.0),
                )
            }
            Phase(q, t) => {
                return self.projector_gate(
                    C64::from_polar(1.0, t),
                    &[Pauli::z(n, q, true)],
                    C64::new(1.0, 0.0),
                )
            }
            Rz(q, t) => {
                return self.projector_gate(
                    C64::from_polar(1.0, t),
                    &[Pauli::z(n, q, true)],
                    C64::from_polar(1.0, -t / 2.0),
                )
            }
            CPhase(a, b, t) => {
                return self.projector_gate(
                    C64::from_polar(1.0, t),
                    &[Pauli::z(n, a, true), Pauli::z(n, b, true)],
                    C64::new(1.0, 0.0),
                )
            }
            Ccx(a, b, t) => {
                return self.projector_gate(
                    C64::new(-1.0, 0.0),
                    &[
                        Pauli::z(n, a, true),
                        Pauli::z(n, b, true),
                        Pauli::xp(n, t, true),
                    ],
                    C64::new(1.0, 0.0),
                )
            }
            ref other => {
                let d = other.decompose_to_clifford_rz();
                assert!(d.len() != 1 || d[0] != *other, "unsupported gate {other:?}");
                for h in d {
                    if !self.apply(&h) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Runs a circuit, recording the rank after every gate.
    pub fn run(&mut self, c: &Circuit) -> bool {
        for g in c.gates() {
            if !self.apply(g) {
                self.stats.overflow = true;
                return false;
            }
            let r = self.terms.len();
            self.stats.r.push(r);
            self.stats.max_r = self.stats.max_r.max(r);
        }
        true
    }

    /// Classifies a projector gate on one term; returns the action and, for
    /// `Branch`, the term `Πφ`.
    pub fn classify(
        st: &ChState,
        lambda: C64,
        factors: &[Pauli],
    ) -> (Action, Vec<Pauli>, Option<ChState>, bool) {
        let mut psi = st.clone();
        let mut eff: Vec<Pauli> = Vec::new();
        for p in factors {
            let r = psi.conj_to_s(p);
            match ChState::eig_of(&r, &psi.s) {
                Some(false) => {}
                Some(true) => return (Action::Diagonal, eff, None, true), // Πφ = 0
                None => {
                    eff.push(p.clone());
                    psi.combo_r(&r, 0);
                    psi.omega.p -= 1;
                }
            }
        }
        let m = eff.len();
        let l = lambda;
        let is = |t: C64| (l - t).norm() < 1e-12;
        let act = if m == 0 {
            Action::Diagonal
        } else if (m == 1
            && (is(C64::new(-1.0, 0.0)) || is(C64::new(0.0, 1.0)) || is(C64::new(0.0, -1.0))))
            || (m == 2 && is(C64::new(-1.0, 0.0)))
        {
            Action::Clifford
        } else {
            Action::Branch
        };
        (act, eff, Some(psi), false)
    }

    /// `U = global · (I + (λ−1) Π_{P_1}⋯Π_{P_m})`.
    pub fn projector_gate(&mut self, lambda: C64, factors: &[Pauli], global: C64) -> bool {
        let lambda = snap(lambda);
        if (lambda - C64::new(1.0, 0.0)).norm() < 1e-12 {
            for t in &mut self.terms {
                t.c *= global;
            }
            return true;
        }
        let mut new_terms = Vec::new();
        let mut branched = false;
        for t in &mut self.terms {
            let (act, eff, psi, killed) = Self::classify(&t.st, lambda, factors);
            match act {
                Action::Diagonal => {
                    self.stats.diag_events += 1;
                    if !killed {
                        t.c *= lambda;
                        t.dirty = true;
                    }
                }
                Action::Clifford => {
                    self.stats.clifford_events += 1;
                    t.dirty = true;
                    let m1 = C64::new(-1.0, 0.0);
                    if eff.len() == 1 {
                        if (lambda - m1).norm() < 1e-12 {
                            t.st.apply_pauli(&eff[0]);
                            t.c *= m1;
                        } else if lambda.im > 0.0 {
                            t.st.combo(&eff[0], 1);
                            t.c *= C64::from_polar(1.0, FRAC_PI_4);
                        } else {
                            t.st.combo(&eff[0], 3);
                            t.c *= C64::from_polar(1.0, -FRAC_PI_4);
                        }
                    } else {
                        // I − 2Π1Π2 = e^{iπ/4} R(P1) R(P2) R(P1P2), R(Q) = (I+iQ)/√2
                        let mut p12 = eff[0].clone();
                        p12.mul_assign(&eff[1]);
                        t.st.combo(&eff[0], 1);
                        t.st.combo(&eff[1], 1);
                        t.st.combo(&p12, 1);
                        t.c *= C64::from_polar(1.0, FRAC_PI_4);
                    }
                }
                Action::Branch => {
                    self.stats.branch_events += 1;
                    branched = true;
                    new_terms.push(Term {
                        c: t.c * (lambda - 1.0),
                        st: psi.unwrap(),
                        dirty: true,
                    });
                    t.dirty = true;
                }
            }
        }
        self.terms.extend(new_terms);
        for t in &mut self.terms {
            t.c *= global;
        }
        if branched
            || (self.pair_merge_s > 0
                && self.terms.len() <= self.pair_merge_max_r
                && self.terms.iter().any(|t| t.dirty))
        {
            let t0 = std::time::Instant::now();
            let tabs = self.merge();
            let t1 = std::time::Instant::now();
            self.stats.t_merge += (t1 - t0).as_secs_f64();
            if self.pair_merge_s > 0 && self.terms.len() <= self.pair_merge_max_r {
                self.pair_merge(tabs);
            }
            self.stats.t_pair += t1.elapsed().as_secs_f64();
        }
        for t in &mut self.terms {
            t.dirty = false;
        }
        self.terms.len() <= self.max_terms
    }

    /// Merges terms that are the same ray and drops zero terms; returns the
    /// canonical tableaux of the surviving terms.
    pub fn merge(&mut self) -> Vec<Vec<Row>> {
        let mut map: HashMap<Vec<u64>, usize> = HashMap::new();
        let mut out: Vec<Term> = Vec::with_capacity(self.terms.len());
        let mut tabs: Vec<Vec<Row>> = Vec::with_capacity(self.terms.len());
        // largest weight that contributed to each merged term: a merged
        // term is zero iff it cancelled to rounding level relative to it
        let mut contrib: Vec<f64> = Vec::with_capacity(self.terms.len());
        let terms = std::mem::take(&mut self.terms);
        for t in terms {
            if t.st.omega.zero {
                continue;
            }
            let tab = t.st.stabilizer_rref();
            let mut key = Vec::with_capacity(tab.len() * (2 * t.st.w + 1));
            for (row, e) in &tab {
                key.extend_from_slice(row);
                key.push(*e as u64);
            }
            match map.get(&key) {
                Some(&i) => {
                    let x0 = out[i].st.support_point();
                    let ai = out[i].st.amplitude(&x0).to_c64();
                    let aj = t.st.amplitude(&x0).to_c64();
                    assert!(
                        ai.norm() > 0.0 && aj.norm() > 0.0,
                        "support point not in support"
                    );
                    contrib[i] = contrib[i].max(t.weight());
                    out[i].c += t.c * aj / ai;
                    out[i].dirty = true;
                    self.stats.merges += 1;
                }
                None => {
                    map.insert(key, out.len());
                    contrib.push(t.weight());
                    out.push(t);
                    tabs.push(tab);
                }
            }
        }
        let before = out.len();
        let tol = self.zero_tol;
        let mut keep_t = Vec::with_capacity(out.len());
        let mut keep_tab = Vec::with_capacity(out.len());
        for ((t, tab), cm) in out.into_iter().zip(tabs).zip(contrib) {
            if t.weight() > tol * cm {
                keep_t.push(t);
                keep_tab.push(tab);
            }
        }
        self.stats.cancellations += before - keep_t.len();
        self.terms = keep_t;
        keep_tab
    }

    /// Merges pairs `(i, j)` (at least one dirty) whose sum is a single
    /// stabilizer state, or cancels, by reducing both to the
    /// `2^s`-dimensional joint eigenspace of their common signed stabilizer
    /// group (`s ≤ pair_merge_s`). Worklist over dirty terms.
    pub fn pair_merge(&mut self, mut tabs: Vec<Vec<Row>>) {
        loop {
            let Some(i) = self.terms.iter().position(|t| t.dirty) else {
                return;
            };
            let r = self.terms.len();
            let mut done = None;
            for j in 0..r {
                if j == i {
                    continue;
                }
                self.stats.pair_tests += 1;
                if let Some(res) = try_pair(
                    &self.terms[i],
                    &self.terms[j],
                    &tabs[i],
                    &tabs[j],
                    self.pair_merge_s,
                    self.zero_tol,
                ) {
                    done = Some((j, res));
                    break;
                }
            }
            match done {
                None => self.terms[i].dirty = false,
                Some((j, res)) => {
                    let (a, b) = (i.max(j), i.min(j));
                    self.terms.swap_remove(a);
                    tabs.swap_remove(a);
                    self.terms.swap_remove(b);
                    tabs.swap_remove(b);
                    self.stats.pair_merges += 1;
                    if let Some(t) = res {
                        tabs.push(t.st.stabilizer_rref());
                        self.terms.push(t);
                    } else {
                        self.stats.cancellations += 2;
                    }
                }
            }
        }
    }

    /// Exact amplitude `⟨x|Ψ⟩`.
    pub fn amplitude(&self, x: &[u64]) -> C64 {
        self.terms
            .iter()
            .map(|t| t.c * t.st.amplitude(x).to_c64())
            .sum()
    }

    /// Dense state vector (small n).
    pub fn to_statevector(&self) -> Vec<C64> {
        assert!(self.n <= 24);
        let w = self.n.div_ceil(64).max(1);
        (0..1usize << self.n)
            .map(|i| {
                let mut x = vec![0u64; w];
                x[0] = i as u64;
                self.amplitude(&x)
            })
            .collect()
    }
}

/// Bitset from a list of set bits.
pub fn bits(n: usize, ones: &[usize]) -> Vec<u64> {
    let mut b = vec![0u64; n.div_ceil(64).max(1)];
    for &i in ones {
        flip(&mut b, i);
    }
    b
}
