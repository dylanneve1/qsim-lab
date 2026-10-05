//! Exact simulation of **monitored Clifford+T circuits** (random Clifford
//! gates, `T` gates and mid-circuit `Z` measurements with Born-rule
//! outcomes) in the rotation-frame form of [`crate::engines::adaptive`]:
//!
//! ```text
//!     |ψ> = C (|φ>_A ⊗ |0>_{rest})
//! ```
//!
//! `C` is an `n`-qubit Clifford kept as a Schrödinger tableau (rows
//! `D_j = C X_j C†`, `S_j = C Z_j C†`), `A` is the *active register* of
//! `d` virtual qubits and `|φ>` is a dense vector of `2^d` amplitudes.
//!
//! * A Clifford gate is a column update of the tableau (`C ← G C`).
//! * A `T` gate on physical qubit `a` maps to the virtual Pauli
//!   `Q = C† Z_a C`. If `Q` has an `x` bit on an inactive coordinate `v`,
//!   Clifford gates controlled by `v` (which holds `|0>`) are absorbed into
//!   `C` until `Q = ±X_v`, and `v` joins the register: `d ← d + 1`.
//!   Otherwise `exp(-iπ/8 Q)` acts on `|φ>` alone (`d` unchanged).
//! * A `Z_a` measurement with an inactive `x` bit has outcome ±1 with
//!   probability exactly 1/2 and is a pure tableau update (`d` unchanged).
//!   Otherwise `Q` acts on `A` only: the outcome is Born-sampled from `|φ>`,
//!   `|φ>` is projected, a Clifford on `A` rotates `Q` to `Z_u`, and the
//!   factorised coordinate `u` is dropped: `d ← d − 1` (the register is
//!   compacted, as Clifft does at measurements).
//!
//! **Theorem (checked in tests).** The unsigned tableau, hence `d(t)`, does
//! not depend on the measurement outcomes or on the amplitudes of `|φ>`:
//! `n − d` is the size of the stabilizer group `G = <S_j : j ∉ A>`, which
//! evolves exactly like the stabilizer group of a *mixed* stabilizer state
//! under Clifford gates, `Z` measurements and a `Z`-dephasing channel at
//! every `T` location (`G → {g ∈ G : [g, Z_a] = 0}`). So `d` is the
//! entropy of a monitored Clifford circuit with dephasing noise at the `T`
//! sites, computable in polynomial time for any `n` ([`Mode::DimensionOnly`]),
//! while `2^d` is the exact cost of the amplitude simulation.

#![allow(clippy::needless_range_loop, clippy::unnecessary_unwrap)]

pub mod circuit;
pub mod ent;

use crate::engines::adaptive::rotate_dense;
use num_complex::Complex64 as C64;
use rand::Rng;
use std::f64::consts::FRAC_PI_4;

// ---------------------------------------------------------------------------
// Local (≤ 2-qubit) Pauli conjugation on `i^r X^x Z^z` (bit masks).

/// `i^r X^x Z^z` on a small register (bit i = qubit i).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LPauli {
    /// X bits (bit `i` = qubit `i`).
    pub x: u64,
    /// Z bits (bit `i` = qubit `i`).
    pub z: u64,
    /// Phase exponent `r` of `i^r`, in `0..4`.
    pub r: u32,
}

impl LPauli {
    /// `P -> H P H†`.
    pub fn h(&mut self, q: usize) {
        let (xb, zb) = (self.x >> q & 1, self.z >> q & 1);
        self.r = (self.r + 2 * (xb & zb) as u32) % 4;
        self.x = (self.x & !(1 << q)) | (zb << q);
        self.z = (self.z & !(1 << q)) | (xb << q);
    }
    /// `P -> S P S†` (`S X S† = i X Z`).
    pub fn s(&mut self, q: usize) {
        let xb = self.x >> q & 1;
        self.r = (self.r + xb as u32) % 4;
        self.z ^= xb << q;
    }
    /// `P -> CNOT P CNOT`.
    pub fn cnot(&mut self, c: usize, t: usize) {
        self.x ^= (self.x >> c & 1) << t;
        self.z ^= (self.z >> t & 1) << c;
    }
    /// `P -> CZ P CZ`.
    pub fn cz(&mut self, a: usize, b: usize) {
        // X_a -> X_a Z_b: reordering X_b past Z_a costs (-1)^{x_b z_a'}.
        let (xa, xb) = (self.x >> a & 1, self.x >> b & 1);
        // i^r X^x Z^z: X_a X_b Z... -> (X_a Z_b)(Z_a X_b) = X_a Z_b Z_a X_b
        // = (-1)^{1} X_a X_b Z_a Z_b when both x bits set (Z_b X_b = -X_b Z_b).
        if xa & xb == 1 {
            self.r = (self.r + 2) % 4;
        }
        self.z ^= xa << b;
        self.z ^= xb << a;
    }
}

/// A two-qubit Clifford as the action of `P -> G P G†` on the 16 local
/// Paulis `X_0^{b0} Z_0^{b1} X_1^{b2} Z_1^{b3}`: image bits and the phase
/// increment `δ` (image = `i^δ X^.. Z^..`).
#[derive(Clone, Debug)]
pub struct Cliff2 {
    /// Indexed by the local Pauli's bits `b0 | b1<<1 | b2<<2 | b3<<3`
    /// (`X_0, Z_0, X_1, Z_1`): `(image bits in the same encoding, δ mod 4)`.
    pub table: [(u8, u8); 16],
    /// Gate word over {H0, H1, S0, S1, CNOT01} in time order.
    pub word: Vec<u8>,
}

fn apply_gen(p: &mut LPauli, g: u8) {
    match g {
        0 => p.h(0),
        1 => p.h(1),
        2 => p.s(0),
        3 => p.s(1),
        _ => p.cnot(0, 1),
    }
}

fn lp_from(l: usize) -> LPauli {
    LPauli {
        x: (l & 1) as u64 | ((l >> 2 & 1) << 1) as u64,
        z: (l >> 1 & 1) as u64 | ((l >> 3 & 1) << 1) as u64,
        r: 0,
    }
}

fn lp_bits(p: &LPauli) -> u8 {
    ((p.x & 1) | (p.z & 1) << 1 | (p.x >> 1 & 1) << 2 | (p.z >> 1 & 1) << 3) as u8
}

impl Cliff2 {
    fn from_word(word: Vec<u8>) -> Self {
        let mut table = [(0u8, 0u8); 16];
        for (l, t) in table.iter_mut().enumerate() {
            let mut p = lp_from(l);
            for &g in &word {
                apply_gen(&mut p, g);
            }
            *t = (lp_bits(&p), p.r as u8);
        }
        Cliff2 { table, word }
    }

    /// All 11520 elements of the two-qubit Clifford group modulo phase, by
    /// breadth-first search over {H0, H1, S0, S1, CNOT01}.
    pub fn group() -> Vec<Cliff2> {
        use std::collections::HashMap;
        let key = |c: &Cliff2| -> u64 {
            // images of X0, Z0, X1, Z1 with phases determine the element
            let mut k = 0u64;
            for (i, l) in [1usize, 2, 4, 8].iter().enumerate() {
                let (b, r) = c.table[*l];
                k |= ((b as u64) | (r as u64) << 4) << (8 * i);
            }
            k
        };
        let id = Cliff2::from_word(Vec::new());
        let mut seen: HashMap<u64, usize> = HashMap::new();
        seen.insert(key(&id), 0);
        let mut out = vec![id];
        let mut head = 0;
        while head < out.len() {
            for g in 0..5u8 {
                let mut w = out[head].word.clone();
                w.push(g);
                let c = Cliff2::from_word(w);
                let k = key(&c);
                if let std::collections::hash_map::Entry::Vacant(e) = seen.entry(k) {
                    e.insert(out.len());
                    out.push(c);
                }
            }
            head += 1;
        }
        out
    }

    /// The element as physical gates on `(a, b)`, time order.
    pub fn gates(&self, a: usize, b: usize) -> Vec<crate::gate::Gate> {
        use crate::gate::Gate;
        self.word
            .iter()
            .map(|&g| match g {
                0 => Gate::H(a),
                1 => Gate::H(b),
                2 => Gate::S(a),
                3 => Gate::S(b),
                _ => Gate::Cnot(a, b),
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Schrödinger tableau: rows D_j = C X_j C† (row j), S_j = C Z_j C† (row n+j),
// each `i^r X^x Z^z` over the n physical qubits.

/// Schrödinger tableau of an `n`-qubit Clifford `C`: row `j < n` is
/// `D_j = C X_j C†`, row `n + j` is `S_j = C Z_j C†`, each stored as
/// `i^r X^x Z^z` with bit-packed `x`, `z` (bit `q` = physical qubit `q`).
#[derive(Clone, Debug)]
pub struct Tableau {
    /// Number of qubits.
    pub n: usize,
    /// Words per bit row (`ceil(n / 64)`, at least 1).
    pub w: usize,
    x: Vec<u64>,
    z: Vec<u64>,
    r: Vec<u8>,
}

#[inline(always)]
fn bit(v: &[u64], i: usize) -> u64 {
    v[i >> 6] >> (i & 63) & 1
}

impl Tableau {
    /// The identity Clifford on `n` qubits.
    pub fn new(n: usize) -> Self {
        let w = n.div_ceil(64).max(1);
        let mut t = Tableau {
            n,
            w,
            x: vec![0; 2 * n * w],
            z: vec![0; 2 * n * w],
            r: vec![0; 2 * n],
        };
        for j in 0..n {
            t.x[j * w + (j >> 6)] |= 1 << (j & 63);
            t.z[(n + j) * w + (j >> 6)] |= 1 << (j & 63);
        }
        t
    }

    /// X bits of row `i` (`w` words).
    #[inline]
    pub fn row_x(&self, i: usize) -> &[u64] {
        &self.x[i * self.w..(i + 1) * self.w]
    }
    /// Z bits of row `i` (`w` words).
    #[inline]
    pub fn row_z(&self, i: usize) -> &[u64] {
        &self.z[i * self.w..(i + 1) * self.w]
    }
    /// Phase exponent `r` (mod 4) of row `i`.
    #[inline]
    pub fn row_r(&self, i: usize) -> u8 {
        self.r[i]
    }
    /// X bit of physical qubit `q` in row `row` (0 or 1).
    #[inline]
    pub fn xbit(&self, row: usize, q: usize) -> u64 {
        bit(&self.x[row * self.w..], q)
    }
    /// Z bit of physical qubit `q` in row `row` (0 or 1).
    #[inline]
    pub fn zbit(&self, row: usize, q: usize) -> u64 {
        bit(&self.z[row * self.w..], q)
    }

    /// `C ← G C` for a two-qubit Clifford on physical qubits `(a, b)`.
    pub fn apply_cliff2(&mut self, c: &Cliff2, a: usize, b: usize) {
        let w = self.w;
        let (wa, sa, wb, sb) = (a >> 6, a & 63, b >> 6, b & 63);
        for row in 0..2 * self.n {
            let o = row * w;
            let l = (self.x[o + wa] >> sa & 1)
                | (self.z[o + wa] >> sa & 1) << 1
                | (self.x[o + wb] >> sb & 1) << 2
                | (self.z[o + wb] >> sb & 1) << 3;
            if l == 0 {
                continue;
            }
            let (img, dr) = c.table[l as usize];
            let img = img as u64;
            self.x[o + wa] = (self.x[o + wa] & !(1 << sa)) | (img & 1) << sa;
            self.z[o + wa] = (self.z[o + wa] & !(1 << sa)) | (img >> 1 & 1) << sa;
            self.x[o + wb] = (self.x[o + wb] & !(1 << sb)) | (img >> 2 & 1) << sb;
            self.z[o + wb] = (self.z[o + wb] & !(1 << sb)) | (img >> 3 & 1) << sb;
            self.r[row] = (self.r[row] + dr) & 3;
        }
    }

    /// `C ← G C` for a single-qubit Clifford word on `q` (0 = H, 2 = S).
    pub fn apply_1q(&mut self, q: usize, gate: u8) {
        let w = self.w;
        let (wq, sq) = (q >> 6, q & 63);
        for row in 0..2 * self.n {
            let o = row * w;
            let mut p = LPauli {
                x: self.x[o + wq] >> sq & 1,
                z: self.z[o + wq] >> sq & 1,
                r: 0,
            };
            if p.x == 0 && p.z == 0 {
                continue;
            }
            match gate {
                0 => p.h(0),
                2 => p.s(0),
                _ => unreachable!(),
            }
            self.x[o + wq] = (self.x[o + wq] & !(1 << sq)) | p.x << sq;
            self.z[o + wq] = (self.z[o + wq] & !(1 << sq)) | p.z << sq;
            self.r[row] = (self.r[row] + p.r as u8) & 3;
        }
    }

    /// Row `dst ← (i^k) · row dst · row src` (operator product, dst on the left).
    fn mul_into(&mut self, dst: usize, src: usize, k: u8) {
        let w = self.w;
        // (i^a X^p Z^q)(i^b X^s Z^t) = i^{a+b+2 q·s} X^{p+s} Z^{q+t}
        let mut c = 0u32;
        for i in 0..w {
            c += (self.z[dst * w + i] & self.x[src * w + i]).count_ones();
        }
        self.r[dst] = ((self.r[dst] as u32 + self.r[src] as u32 + 2 * c + k as u32) & 3) as u8;
        for i in 0..w {
            self.x[dst * w + i] ^= self.x[src * w + i];
            self.z[dst * w + i] ^= self.z[src * w + i];
        }
    }

    /// Right multiplication `C ← C g` by a virtual gate `g`.
    pub fn right_mul(&mut self, g: VGate) {
        let n = self.n;
        match g {
            VGate::H(q) => {
                // H X H = Z: D_q <-> S_q
                let w = self.w;
                for i in 0..w {
                    self.x.swap(q * w + i, (n + q) * w + i);
                    self.z.swap(q * w + i, (n + q) * w + i);
                }
                self.r.swap(q, n + q);
            }
            // g X_q g† = i X Z (S) / -i X Z (Sdg): D_q <- i^{±1} D_q S_q
            VGate::S(q) => self.mul_into(q, n + q, 1),
            VGate::Sdg(q) => self.mul_into(q, n + q, 3),
            // X Z X = -Z
            VGate::X(q) => self.r[n + q] = (self.r[n + q] + 2) & 3,
            VGate::Z(q) => self.r[q] = (self.r[q] + 2) & 3,
            VGate::Cnot(c, t) => {
                // X_c -> X_c X_t ; Z_t -> Z_c Z_t
                self.mul_into(c, t, 0);
                self.mul_into(n + t, n + c, 0);
            }
            VGate::Cz(a, b) => {
                // X_a -> X_a Z_b ; X_b -> Z_a X_b = X_b Z_a
                self.mul_into(a, n + b, 0);
                self.mul_into(b, n + a, 0);
            }
        }
    }

    /// Virtual decomposition of the physical `Z_a`: (x support, z support).
    pub fn decompose_z(&self, a: usize) -> (Vec<usize>, Vec<usize>) {
        let n = self.n;
        let mut xs = Vec::new();
        let mut zs = Vec::new();
        for j in 0..n {
            if self.xbit(n + j, a) == 1 {
                xs.push(j); // Z_a anticommutes with S_j
            }
            if self.xbit(j, a) == 1 {
                zs.push(j); // Z_a anticommutes with D_j
            }
        }
        (xs, zs)
    }

    /// Sign `ε` with `C Q C† = ε Z_a` for `Q = i^{|x∧z|} X^x Z^z`.
    pub fn sign_of(&self, a: usize, xs: &[usize], zs: &[usize]) -> i32 {
        let w = self.w;
        let mut ax = vec![0u64; w];
        let mut az = vec![0u64; w];
        let mut ar = 0u32;
        let mut mul = |row: usize, ax: &mut Vec<u64>, az: &mut Vec<u64>| {
            let mut c = 0u32;
            for i in 0..w {
                c += (az[i] & self.x[row * w + i]).count_ones();
            }
            ar = (ar + self.r[row] as u32 + 2 * c) & 3;
            for i in 0..w {
                ax[i] ^= self.x[row * w + i];
                az[i] ^= self.z[row * w + i];
            }
        };
        for &j in xs {
            mul(j, &mut ax, &mut az);
        }
        for &j in zs {
            mul(self.n + j, &mut ax, &mut az);
        }
        debug_assert!(ax.iter().all(|&v| v == 0));
        debug_assert!(az
            .iter()
            .enumerate()
            .all(|(i, &v)| v == if i == a >> 6 { 1 << (a & 63) } else { 0 }));
        let xz = xs.iter().filter(|j| zs.contains(j)).count() as u32;
        let k = (ar + xz) & 3;
        debug_assert!(k.is_multiple_of(2), "non-Hermitian image");
        if k == 0 {
            1
        } else {
            -1
        }
    }
}

/// Virtual (right-multiplied) Clifford gates on coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VGate {
    /// Hadamard on a coordinate.
    H(usize),
    /// `S` on a coordinate.
    S(usize),
    /// `S†` on a coordinate.
    Sdg(usize),
    /// Pauli X on a coordinate.
    X(usize),
    /// Pauli Z on a coordinate.
    Z(usize),
    /// CNOT `(control, target)`.
    Cnot(usize, usize),
    /// CZ (symmetric).
    Cz(usize, usize),
}

impl VGate {
    /// The same gate as a [`crate::gate::Gate`] on the same indices.
    pub fn to_gate(self) -> crate::gate::Gate {
        use crate::gate::Gate;
        match self {
            VGate::H(q) => Gate::H(q),
            VGate::S(q) => Gate::S(q),
            VGate::Sdg(q) => Gate::Sdg(q),
            VGate::X(q) => Gate::X(q),
            VGate::Z(q) => Gate::Z(q),
            VGate::Cnot(c, t) => Gate::Cnot(c, t),
            VGate::Cz(a, b) => Gate::Cz(a, b),
        }
    }
}

// ---------------------------------------------------------------------------
// Dense helpers on the active register (bit i = active position i).

pub(crate) fn dense_h(a: &mut [C64], q: usize) {
    let r = std::f64::consts::FRAC_1_SQRT_2;
    let m = 1usize << q;
    for y in 0..a.len() {
        if y & m == 0 {
            let (l, h) = (a[y], a[y | m]);
            a[y] = (l + h) * r;
            a[y | m] = (l - h) * r;
        }
    }
}
pub(crate) fn dense_s(a: &mut [C64], q: usize, dag: bool) {
    let m = 1usize << q;
    let ph = if dag {
        C64::new(0.0, -1.0)
    } else {
        C64::new(0.0, 1.0)
    };
    for (y, v) in a.iter_mut().enumerate() {
        if y & m != 0 {
            *v *= ph;
        }
    }
}
pub(crate) fn dense_x(a: &mut [C64], q: usize) {
    let m = 1usize << q;
    for y in 0..a.len() {
        if y & m == 0 {
            a.swap(y, y | m);
        }
    }
}
pub(crate) fn dense_cnot(a: &mut [C64], c: usize, t: usize) {
    let (mc, mt) = (1usize << c, 1usize << t);
    for y in 0..a.len() {
        if y & mc != 0 && y & mt == 0 {
            a.swap(y, y | mt);
        }
    }
}
pub(crate) fn dense_cz(a: &mut [C64], p: usize, q: usize) {
    let m = (1usize << p) | (1usize << q);
    for (y, v) in a.iter_mut().enumerate() {
        if y & m == m {
            *v = -*v;
        }
    }
}

/// `<φ| i^{|x∧z|} X^x Z^z |φ>` (real for Hermitian strings).
pub(crate) fn dense_pauli_exp(a: &[C64], x: u64, z: u64) -> f64 {
    let r = (x & z).count_ones();
    let mut s = C64::new(0.0, 0.0);
    for (y, &v) in a.iter().enumerate() {
        // (P φ)(y⊕x) gets i^r (-1)^{z·y} φ(y) ... evaluate <φ|P|φ> = Σ_y conj(φ(y⊕x)) (Pφ)(y⊕x)
        let t = a[y ^ x as usize].conj() * v;
        if ((z & y as u64).count_ones() & 1) == 1 {
            s -= t;
        } else {
            s += t;
        }
    }
    let ph = match r % 4 {
        0 => C64::new(1.0, 0.0),
        1 => C64::new(0.0, 1.0),
        2 => C64::new(-1.0, 0.0),
        _ => C64::new(0.0, -1.0),
    };
    (ph * s).re
}

/// `Q φ` for `Q = i^{|x∧z|} X^x Z^z`.
fn dense_apply_pauli(a: &[C64], x: u64, z: u64) -> Vec<C64> {
    let r = (x & z).count_ones();
    let ph = match r % 4 {
        0 => C64::new(1.0, 0.0),
        1 => C64::new(0.0, 1.0),
        2 => C64::new(-1.0, 0.0),
        _ => C64::new(0.0, -1.0),
    };
    let mut out = vec![C64::new(0.0, 0.0); a.len()];
    for (y, &v) in a.iter().enumerate() {
        // X^x Z^z |y> = (-1)^{z·y} |y ⊕ x>
        let s = if ((z & y as u64).count_ones() & 1) == 1 {
            -ph
        } else {
            ph
        };
        out[y ^ x as usize] = s * v;
    }
    out
}

// ---------------------------------------------------------------------------
// The simulator.

/// Amplitude tracking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Exact state: tableau + dense `|φ>` (fails beyond `max_d`).
    Exact,
    /// Tableau only: exact `d(t)` and entanglement bounds at any `n`; the
    /// outcomes of register measurements are drawn 50/50 (they do not affect
    /// `d`, see the module docs).
    DimensionOnly,
}

/// One measurement record.
#[derive(Clone, Copy, Debug)]
pub struct MeasRecord {
    /// Physical qubit measured.
    pub qubit: usize,
    /// Outcome (`true` = `-1` eigenvalue, i.e. bit 1).
    pub outcome: bool,
    /// Born probability of the observed outcome (1/2 for frame-random ones).
    pub prob: f64,
    /// Which case: 0 frame-random, 1 register (projective, d−1), 2 determined.
    pub kind: u8,
}

/// Counters of a [`Monitored`] run.
#[derive(Clone, Debug, Default)]
pub struct MonStats {
    /// Z rotations applied (`T` gates and any [`Monitored::rz`]).
    pub t_gates: usize,
    /// T gates that activated a coordinate (d+1).
    pub t_activating: usize,
    /// T gates that rotated the register in place.
    pub t_register: usize,
    /// Measurements performed.
    pub meas: usize,
    /// Of which frame-random (outcome 50/50, `d` unchanged).
    pub meas_frame: usize,
    /// Of which on the register (projective, `d − 1`).
    pub meas_register: usize,
    /// Of which determined by the state.
    pub meas_determined: usize,
    /// Largest register size `d` reached.
    pub max_d: usize,
    /// Σ 2^d over dense updates.
    pub element_ops: u64,
}

/// A monitored Clifford+T simulation `|ψ> = C (|φ>_A ⊗ |0>)` (see the
/// module docs).
pub struct Monitored {
    /// Number of physical qubits.
    pub n: usize,
    /// The Clifford `C` as a Schrödinger tableau.
    pub tab: Tableau,
    /// active position -> virtual coordinate
    pub active: Vec<usize>,
    /// virtual coordinate -> active position (usize::MAX if inactive)
    pos: Vec<usize>,
    /// The `2^d` register amplitudes `|φ>` (bit `i` = active position `i`);
    /// `None` in [`Mode::DimensionOnly`].
    pub amp: Option<Vec<C64>>,
    /// Largest allowed register size (capped at 34); exceeding it in
    /// [`Mode::Exact`] fails with [`TooLarge`].
    pub max_d: usize,
    /// Counters so far.
    pub stats: MonStats,
    /// Right-multiplied virtual gates, in order (tests: `C = G_T..G_1 R_1 R_2 ..`).
    pub vlog: Option<Vec<VGate>>,
    /// Every measurement record, in order.
    pub records: Vec<MeasRecord>,
}

/// Error: an operation would grow the register beyond `max_d`.
#[derive(Debug)]
pub struct TooLarge {
    /// The register size that was refused.
    pub d: usize,
}

impl Monitored {
    /// `|0^n>` in `mode`, with register size limit `max_d` (capped at 34).
    pub fn new(n: usize, mode: Mode, max_d: usize) -> Self {
        Monitored {
            n,
            tab: Tableau::new(n),
            active: Vec::new(),
            pos: vec![usize::MAX; n],
            amp: match mode {
                Mode::Exact => Some(vec![C64::new(1.0, 0.0)]),
                Mode::DimensionOnly => None,
            },
            max_d: max_d.min(34),
            stats: MonStats::default(),
            vlog: None,
            records: Vec::new(),
        }
    }

    /// Also records every virtual gate in [`Monitored::vlog`].
    pub fn with_log(mut self) -> Self {
        self.vlog = Some(Vec::new());
        self
    }

    /// Current active register size `d` (`2^d` amplitudes in exact mode).
    pub fn d(&self) -> usize {
        self.active.len()
    }

    fn rmul(&mut self, g: VGate) {
        self.tab.right_mul(g);
        if let Some(l) = &mut self.vlog {
            l.push(g);
        }
    }

    /// Physical two-qubit Clifford.
    pub fn cliff2(&mut self, c: &Cliff2, a: usize, b: usize) {
        self.tab.apply_cliff2(c, a, b);
    }

    /// Physical H (0) / S (2).
    pub fn cliff1(&mut self, q: usize, g: u8) {
        self.tab.apply_1q(q, g);
    }

    /// Absorbs `CNOT(v, t)`, `CZ(v, t)` (and `S_v`) into `C`, all controlled
    /// by the inactive `v` (state unchanged), until `C† Z_a C = ±X_v`.
    /// Returns the sign `s` with `Z_a = s · C X_v C†`.
    fn isolate(&mut self, a: usize, v: usize, xs: &[usize]) -> i32 {
        for &t in xs {
            if t != v {
                self.rmul(VGate::Cnot(v, t));
            }
        }
        let (xs2, zs2) = self.tab.decompose_z(a);
        debug_assert!(xs2 == vec![v]);
        for &t in &zs2 {
            if t != v {
                self.rmul(VGate::Cz(v, t));
            }
        }
        if zs2.contains(&v) {
            // Q ∝ Y_v; S_v (acts as identity on |0>_v): S† Y S = X
            self.rmul(VGate::S(v));
        }
        // Now Q = ±X_v, D_v = C X_v C† = ±Z_a.
        let n = self.n;
        debug_assert!(self.tab.row_x(v).iter().all(|&u| u == 0));
        debug_assert!((0..n).all(|q| self.tab.zbit(v, q) == (q == a) as u64));
        let r = self.tab.row_r(v);
        debug_assert!(r.is_multiple_of(2));
        if r == 0 {
            1
        } else {
            -1
        }
    }

    /// Local masks of `xs`, `zs` on the active register (z bits on inactive
    /// coordinates dropped: they act as +1 on |0>).
    fn local(&self, xs: &[usize], zs: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let lx = xs.iter().map(|&j| self.pos[j]).collect::<Vec<_>>();
        let lz = zs
            .iter()
            .filter(|&&j| self.pos[j] != usize::MAX)
            .map(|&j| self.pos[j])
            .collect::<Vec<_>>();
        (lx, lz)
    }

    /// `exp(-i θ Z_a / 2)` (θ = π/4 is `T` up to a global phase).
    pub fn rz(&mut self, a: usize, theta: f64) -> Result<(), TooLarge> {
        self.stats.t_gates += 1;
        let (xs, zs) = self.tab.decompose_z(a);
        if let Some(&v) = xs.iter().find(|&&j| self.pos[j] == usize::MAX) {
            // activate v
            if self.d() + 1 > self.max_d && self.amp.is_some() {
                return Err(TooLarge { d: self.d() + 1 });
            }
            let s = self.isolate(a, v, &xs);
            self.stats.t_activating += 1;
            let p = self.active.len();
            self.active.push(v);
            self.pos[v] = p;
            if let Some(amp) = &mut self.amp {
                // exp(-iθ s X/2)|0> = cos|0> − i s sin|1>
                let (sn, cs) = (theta / 2.0).sin_cos();
                let c1 = C64::new(0.0, -(s as f64) * sn);
                let len = amp.len();
                amp.resize(2 * len, C64::new(0.0, 0.0));
                for y in 0..len {
                    let v0 = amp[y];
                    amp[y] = v0 * cs;
                    amp[y + len] = v0 * c1;
                }
                self.stats.element_ops += 2 * len as u64;
            }
            self.stats.max_d = self.stats.max_d.max(self.d());
            return Ok(());
        }
        // x support inside the register
        let (lx, lz) = self.local(&xs, &zs);
        if lx.is_empty() && lz.is_empty() {
            return Ok(()); // global phase
        }
        self.stats.t_register += 1;
        if self.amp.is_some() {
            let eps = self.tab.sign_of(a, &xs, &zs);
            let amp = self.amp.as_mut().unwrap();
            let (mx, mz) = masks(&lx, &lz);
            rotate_dense(amp, mx, mz, eps as f64 * theta);
            self.stats.element_ops += amp.len() as u64;
        }
        Ok(())
    }

    /// A `T` gate on physical qubit `a` (`rz(a, π/4)`, up to a global phase);
    /// fails if it would activate a coordinate beyond `max_d` in exact mode.
    pub fn t(&mut self, a: usize) -> Result<(), TooLarge> {
        self.rz(a, FRAC_PI_4)
    }

    /// Born-rule `Z_a` measurement. `forced` fixes the outcome (tests; the
    /// returned record then carries that outcome's probability).
    pub fn measure<R: Rng + ?Sized>(
        &mut self,
        a: usize,
        rng: &mut R,
        forced: Option<bool>,
    ) -> MeasRecord {
        self.stats.meas += 1;
        let (xs, zs) = self.tab.decompose_z(a);
        let rec;
        if let Some(&v) = xs.iter().find(|&&j| self.pos[j] == usize::MAX) {
            // frame-random: Z_a = s C X_v C†, v holds |0>
            let s = self.isolate(a, v, &xs);
            let out = forced.unwrap_or_else(|| rng.random::<bool>());
            let lambda = if out { -1 } else { 1 };
            self.rmul(VGate::H(v));
            if lambda * s == -1 {
                self.rmul(VGate::X(v));
            }
            self.stats.meas_frame += 1;
            rec = MeasRecord {
                qubit: a,
                outcome: out,
                prob: 0.5,
                kind: 0,
            };
        } else {
            let (lx, lz) = self.local(&xs, &zs);
            if lx.is_empty() && lz.is_empty() {
                // Z_a ψ = ε ψ
                let eps = if self.amp.is_some() {
                    self.tab.sign_of(a, &xs, &zs)
                } else {
                    1
                };
                let out = eps == -1;
                self.stats.meas_determined += 1;
                rec = MeasRecord {
                    qubit: a,
                    outcome: forced.unwrap_or(out),
                    prob: if forced.is_some_and(|f| f != out) {
                        0.0
                    } else {
                        1.0
                    },
                    kind: 2,
                };
            } else {
                self.stats.meas_register += 1;
                let (out, prob) = if self.amp.is_some() {
                    let eps = self.tab.sign_of(a, &xs, &zs) as f64;
                    let (mx, mz) = masks(&lx, &lz);
                    let amp = self.amp.as_mut().unwrap();
                    let ev = eps * dense_pauli_exp(amp, mx, mz);
                    let p0 = ((1.0 + ev) / 2.0).clamp(0.0, 1.0);
                    let out = forced.unwrap_or_else(|| rng.random::<f64>() >= p0);
                    let prob = if out { 1.0 - p0 } else { p0 };
                    let lam = if out { -1.0 } else { 1.0 };
                    // φ <- (1 + λ ε Q) φ / (2 sqrt(prob))
                    let qa = dense_apply_pauli(amp, mx, mz);
                    let f = 1.0 / (2.0 * prob.max(1e-300).sqrt());
                    for (v, q) in amp.iter_mut().zip(&qa) {
                        *v = (*v + q * (lam * eps)) * f;
                    }
                    self.stats.element_ops += 2 * amp.len() as u64;
                    (out, prob)
                } else {
                    (rng.random::<bool>(), 0.5)
                };
                self.reduce_and_drop(lx, lz);
                rec = MeasRecord {
                    qubit: a,
                    outcome: out,
                    prob,
                    kind: 1,
                };
            }
        }
        self.records.push(rec);
        rec
    }

    /// Virtual gate `g` on active positions: `φ ← g φ`, `C ← C g†`.
    fn vgate(&mut self, g: VGate) {
        // C ← C g† on the virtual coordinates behind the register positions.
        let vc = |p: usize| self.active[p];
        let gdag = match g {
            VGate::H(p) => VGate::H(vc(p)),
            VGate::S(p) => VGate::Sdg(vc(p)),
            VGate::Sdg(p) => VGate::S(vc(p)),
            VGate::X(p) => VGate::X(vc(p)),
            VGate::Z(p) => VGate::Z(vc(p)),
            VGate::Cnot(c, t) => VGate::Cnot(vc(c), vc(t)),
            VGate::Cz(a, b) => VGate::Cz(vc(a), vc(b)),
        };
        self.rmul(gdag);
        if let Some(amp) = &mut self.amp {
            match g {
                VGate::H(p) => dense_h(amp, p),
                VGate::S(p) => dense_s(amp, p, false),
                VGate::Sdg(p) => dense_s(amp, p, true),
                VGate::X(p) => dense_x(amp, p),
                VGate::Z(p) => {
                    let m = 1usize << p;
                    for (y, v) in amp.iter_mut().enumerate() {
                        if y & m != 0 {
                            *v = -*v;
                        }
                    }
                }
                VGate::Cnot(c, t) => dense_cnot(amp, c, t),
                VGate::Cz(a, b) => dense_cz(amp, a, b),
            }
            self.stats.element_ops += amp.len() as u64;
        }
    }

    /// After a register measurement (φ an eigenstate of the local `Q`):
    /// rotate `Q` to `Z_u` by Cliffords on the register, factor out `u`.
    fn reduce_and_drop(&mut self, lx: Vec<usize>, lz: Vec<usize>) {
        // track Q as position lists (unsigned)
        let mut qx: Vec<bool> = vec![false; self.d()];
        let mut qz: Vec<bool> = vec![false; self.d()];
        for &p in &lx {
            qx[p] ^= true;
        }
        for &p in &lz {
            qz[p] ^= true;
        }
        let u;
        if let Some(u0) = qx.iter().position(|&b| b) {
            u = u0;
            for t in 0..qx.len() {
                if t != u && qx[t] {
                    // CNOT(u,t): X_u X_t -> X_u ; Z_t -> Z_u Z_t
                    self.vgate(VGate::Cnot(u, t));
                    qx[t] = false;
                    if qz[t] {
                        qz[u] ^= true;
                    }
                }
            }
            for t in 0..qz.len() {
                if t != u && qz[t] {
                    self.vgate(VGate::Cz(u, t));
                    qz[t] = false;
                }
            }
            if qz[u] {
                self.vgate(VGate::S(u)); // S (XZ) S† ∝ X
                qz[u] = false;
            }
            self.vgate(VGate::H(u));
        } else {
            u = qz.iter().position(|&b| b).expect("nontrivial Q");
            for t in 0..qz.len() {
                if t != u && qz[t] {
                    // CNOT(t,u): Z_t Z_u -> Z_u
                    self.vgate(VGate::Cnot(t, u));
                    qz[t] = false;
                }
            }
        }
        // φ is now an eigenstate of Z_u
        let m = 1usize << u;
        let mut b = false;
        if let Some(amp) = &self.amp {
            let mut w1 = 0.0;
            for (y, v) in amp.iter().enumerate() {
                if y & m != 0 {
                    w1 += v.norm_sqr();
                }
            }
            debug_assert!(!(1e-9..1.0 - 1e-9).contains(&w1), "not an eigenstate: {w1}");
            b = w1 > 0.5;
        }
        if b {
            self.vgate(VGate::X(u));
        }
        if let Some(amp) = &mut self.amp {
            let half = amp.len() / 2;
            let lo_mask = m - 1;
            let mut out = Vec::with_capacity(half);
            for y in 0..half {
                let full = (y & lo_mask) | ((y & !lo_mask) << 1);
                out.push(amp[full]);
            }
            let nrm: f64 = out.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
            for v in out.iter_mut() {
                *v /= nrm;
            }
            *amp = out;
        }
        let vq = self.active.remove(u);
        self.pos[vq] = usize::MAX;
        for (p, &c) in self.active.iter().enumerate() {
            self.pos[c] = p;
        }
    }

    /// The full `2^n` state (tests): `C (φ ⊗ 0)` from the gate logs.
    /// `phys` are the physical gates in time order.
    pub fn to_statevector(&self, phys: &[crate::gate::Gate]) -> crate::StateVectorF64 {
        let n = self.n;
        assert!(n <= 24);
        let amp = self.amp.as_ref().expect("exact mode");
        let mut full = vec![C64::new(0.0, 0.0); 1 << n];
        for (y, &v) in amp.iter().enumerate() {
            let mut idx = 0usize;
            for (p, &c) in self.active.iter().enumerate() {
                if y >> p & 1 == 1 {
                    idx |= 1 << c;
                }
            }
            full[idx] = v;
        }
        let mut sv = crate::StateVectorF64::from_amplitudes(full);
        for g in self.vlog.as_ref().expect("with_log").iter().rev() {
            sv.apply_gate(&g.to_gate()).unwrap();
        }
        for g in phys {
            sv.apply_gate(g).unwrap();
        }
        sv
    }
}

fn masks(lx: &[usize], lz: &[usize]) -> (u64, u64) {
    let mut mx = 0u64;
    let mut mz = 0u64;
    for &p in lx {
        mx ^= 1 << p;
    }
    for &p in lz {
        mz ^= 1 << p;
    }
    (mx, mz)
}
