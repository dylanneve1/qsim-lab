//! Frozen copy of the original CHP tableau implementation, kept as the
//! A/B baseline for benchmarks and as an exactness oracle in tests. Do not
//! optimise this file.
#![allow(dead_code)]

use super::ref_bitmatrix::{self as bitmatrix, BitMatrix};
use super::{tableau_bytes, Layout, MAX_TABLEAU_BYTES};
use crate::circuit::{check_gate, SimError, Simulator};
use crate::gate::{is_multiple_of_half_pi, Gate};
use rand::{Rng, RngCore};
use rayon::prelude::*;
use std::f64::consts::FRAC_PI_2;

/// The CHP stabilizer tableau.
#[derive(Clone, Debug)]
pub struct RefTableau {
    n: usize,
    np: usize,
    w: usize,
    xd: BitMatrix,
    zd: BitMatrix,
    xs: BitMatrix,
    zs: BitMatrix,
    /// Sign bits of destabilizers / stabilizers, one bit per generator.
    rd: Vec<u64>,
    rs: Vec<u64>,
    layout: Layout,
}

#[inline]
fn get_bit(v: &[u64], i: usize) -> bool {
    (v[i / 64] >> (i % 64)) & 1 == 1
}

#[inline]
fn set_bit(v: &mut [u64], i: usize, b: bool) {
    let m = 1u64 << (i % 64);
    if b {
        v[i / 64] |= m;
    } else {
        v[i / 64] &= !m;
    }
}

/// Index of the first set bit at position `>= from`, if any.
fn next_set_bit(v: &[u64], from: usize) -> Option<usize> {
    let mut wi = from / 64;
    if wi >= v.len() {
        return None;
    }
    let mut word = v[wi] & (u64::MAX << (from % 64));
    loop {
        if word != 0 {
            return Some(wi * 64 + word.trailing_zeros() as usize);
        }
        wi += 1;
        if wi == v.len() {
            return None;
        }
        word = v[wi];
    }
}

/// Replaces Pauli row `h` with the product `i * h` (CHP's `rowsum(h, i)`)
/// and returns the new sign bit of `h`.
///
/// The sign of a product of Paulis is `i^k` with `k = 2 r_h + 2 r_i + sum_j
/// g(x_ij, z_ij, x_hj, z_hj) (mod 4)`, where `g` is +1, -1 or 0 per qubit.
/// Instead of looping over qubits, build masks of the +1 and -1 positions
/// for 64 qubits at a time and count them with `popcount`.
#[inline]
fn rowsum(hx: &mut [u64], hz: &mut [u64], hr: bool, ix: &[u64], iz: &[u64], ir: bool) -> bool {
    let mut pos: u64 = 0;
    let mut neg: u64 = 0;
    for k in 0..hx.len() {
        let (x1, z1, x2, z2) = (ix[k], iz[k], hx[k], hz[k]);
        // i-row is Y: g = z2 - x2; X: g = z2(2x2 - 1); Z: g = x2(1 - 2z2)
        let plus = (x1 & z1 & z2 & !x2) | (x1 & !z1 & z2 & x2) | (!x1 & z1 & x2 & !z2);
        let minus = (x1 & z1 & x2 & !z2) | (x1 & !z1 & z2 & !x2) | (!x1 & z1 & x2 & z2);
        pos += plus.count_ones() as u64;
        neg += minus.count_ones() as u64;
        hx[k] = x2 ^ x1;
        hz[k] = z2 ^ z1;
    }
    let total = 2 * (hr as i64) + 2 * (ir as i64) + pos as i64 - neg as i64;
    let m = total.rem_euclid(4);
    debug_assert!(m == 0 || m == 2, "rowsum of anticommuting rows");
    m == 2
}

impl RefTableau {
    /// The state `|0...0>` on `n` qubits. Panics above [`MAX_TABLEAU_BYTES`].
    pub fn new(n: usize) -> Self {
        Self::try_new(n).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_new(n: usize) -> Result<Self, SimError> {
        let bytes = tableau_bytes(n);
        if bytes > MAX_TABLEAU_BYTES {
            return Err(SimError::TooLarge {
                what: "stabilizer tableau",
                bytes,
                limit: MAX_TABLEAU_BYTES,
            });
        }
        let np = n.div_ceil(64).max(1) * 64;
        Ok(RefTableau {
            n,
            np,
            w: np / 64,
            // destabilizer i = X_i, stabilizer i = Z_i; identity in both layouts
            xd: BitMatrix::identity(np),
            zd: BitMatrix::zeros(np),
            xs: BitMatrix::zeros(np),
            zs: BitMatrix::identity(np),
            rd: vec![0; np / 64],
            rs: vec![0; np / 64],
            layout: Layout::QubitMajor,
        })
    }

    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// Bytes of tableau storage.
    pub fn bytes(&self) -> usize {
        self.xd.bytes() * 4 + (self.rd.len() + self.rs.len()) * 8
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    fn set_layout(&mut self, layout: Layout) {
        if self.layout != layout {
            let (xd, zd, xs, zs) = (&mut self.xd, &mut self.zd, &mut self.xs, &mut self.zs);
            rayon::join(
                || rayon::join(|| xd.transpose_in_place(), || zd.transpose_in_place()),
                || rayon::join(|| xs.transpose_in_place(), || zs.transpose_in_place()),
            );
            self.layout = layout;
        }
    }

    /// Applies `f(x_line, z_line, r)` for a single qubit to both halves of
    /// the tableau (qubit-major layout).
    fn for_blocks_1q(&mut self, a: usize, f: impl Fn(&mut [u64], &mut [u64], &mut [u64])) {
        self.set_layout(Layout::QubitMajor);
        f(self.xd.line_mut(a), self.zd.line_mut(a), &mut self.rd);
        f(self.xs.line_mut(a), self.zs.line_mut(a), &mut self.rs);
    }

    /// Two-qubit analogue: `f(xa, za, xb, zb, r)`.
    fn for_blocks_2q(
        &mut self,
        a: usize,
        b: usize,
        f: impl Fn(&mut [u64], &mut [u64], &mut [u64], &mut [u64], &mut [u64]),
    ) {
        self.set_layout(Layout::QubitMajor);
        let (xa, xb) = self.xd.two_lines_mut(a, b);
        let (za, zb) = self.zd.two_lines_mut(a, b);
        f(xa, za, xb, zb, &mut self.rd);
        let (xa, xb) = self.xs.two_lines_mut(a, b);
        let (za, zb) = self.zs.two_lines_mut(a, b);
        f(xa, za, xb, zb, &mut self.rs);
    }

    pub fn h(&mut self, a: usize) {
        self.for_blocks_1q(a, |x, z, r| {
            for k in 0..x.len() {
                r[k] ^= x[k] & z[k];
            }
            x.swap_with_slice(z);
        });
    }

    pub fn s(&mut self, a: usize) {
        self.for_blocks_1q(a, |x, z, r| {
            for k in 0..x.len() {
                r[k] ^= x[k] & z[k];
                z[k] ^= x[k];
            }
        });
    }

    pub fn sdg(&mut self, a: usize) {
        self.for_blocks_1q(a, |x, z, r| {
            for k in 0..x.len() {
                r[k] ^= x[k] & !z[k];
                z[k] ^= x[k];
            }
        });
    }

    pub fn x(&mut self, a: usize) {
        self.for_blocks_1q(a, |_, z, r| r.iter_mut().zip(z).for_each(|(r, z)| *r ^= *z));
    }

    pub fn z(&mut self, a: usize) {
        self.for_blocks_1q(a, |x, _, r| r.iter_mut().zip(x).for_each(|(r, x)| *r ^= *x));
    }

    pub fn y(&mut self, a: usize) {
        self.for_blocks_1q(a, |x, z, r| {
            for k in 0..x.len() {
                r[k] ^= x[k] ^ z[k];
            }
        });
    }

    pub fn cnot(&mut self, c: usize, t: usize) {
        self.for_blocks_2q(c, t, |xc, zc, xt, zt, r| {
            for k in 0..xc.len() {
                r[k] ^= xc[k] & zt[k] & !(xt[k] ^ zc[k]);
                xt[k] ^= xc[k];
                zc[k] ^= zt[k];
            }
        });
    }

    pub fn cz(&mut self, a: usize, b: usize) {
        self.for_blocks_2q(a, b, |xa, za, xb, zb, r| {
            for k in 0..xa.len() {
                r[k] ^= xa[k] & xb[k] & (za[k] ^ zb[k]);
                za[k] ^= xb[k];
                zb[k] ^= xa[k];
            }
        });
    }

    pub fn swap(&mut self, a: usize, b: usize) {
        self.set_layout(Layout::QubitMajor);
        for m in [&mut self.xd, &mut self.zd, &mut self.xs, &mut self.zs] {
            m.swap_lines(a, b);
        }
    }

    /// Applies a gate, or returns [`SimError::Unsupported`] for gates outside
    /// the Clifford group. Phase-type rotations by multiples of π/2 are
    /// accepted (they are powers of S, up to global phase).
    pub fn apply_gate(&mut self, g: &Gate) -> Result<(), SimError> {
        check_gate(g, self.n)?;
        let unsupported = Err(SimError::Unsupported {
            backend: "stabilizer",
            gate: *g,
        });
        match *g {
            Gate::H(a) => self.h(a),
            Gate::S(a) => self.s(a),
            Gate::Sdg(a) => self.sdg(a),
            Gate::X(a) => self.x(a),
            Gate::Y(a) => self.y(a),
            Gate::Z(a) => self.z(a),
            Gate::Cnot(c, t) => self.cnot(c, t),
            Gate::Cz(a, b) => self.cz(a, b),
            Gate::Swap(a, b) => self.swap(a, b),
            Gate::Phase(a, t) | Gate::Rz(a, t) if is_multiple_of_half_pi(t) => {
                let k = (t / FRAC_PI_2).round().rem_euclid(4.0) as usize;
                for _ in 0..k {
                    self.s(a);
                }
            }
            Gate::CPhase(a, b, t) if is_multiple_of_half_pi(t / 2.0) => {
                // multiples of π: identity or CZ
                if (t / std::f64::consts::PI).round().rem_euclid(2.0) == 1.0 {
                    self.cz(a, b);
                }
            }
            _ => return unsupported,
        }
        Ok(())
    }

    /// Returns `Some(outcome)` if measuring qubit `a` is deterministic, or
    /// `None` if the outcome would be uniformly random. Does not change the
    /// state (but may switch the internal layout).
    pub fn peek(&mut self, a: usize) -> Option<bool> {
        self.set_layout(Layout::GeneratorMajor);
        if (0..self.np).any(|i| get_bit(self.xs.line(i), a)) {
            return None;
        }
        Some(self.deterministic_outcome(a))
    }

    fn deterministic_outcome(&self, a: usize) -> bool {
        let w = self.w;
        let mut sx = vec![0u64; w];
        let mut sz = vec![0u64; w];
        let mut sr = false;
        for i in 0..self.np {
            if get_bit(self.xd.line(i), a) {
                sr = rowsum(
                    &mut sx,
                    &mut sz,
                    sr,
                    self.xs.line(i),
                    self.zs.line(i),
                    get_bit(&self.rs, i),
                );
            }
        }
        sr
    }

    /// Measures qubit `a`. If the outcome is random and `forced` is given,
    /// that outcome is selected instead of drawing one. Returns
    /// `(outcome, was_random)`.
    pub fn measure_with<R: Rng + ?Sized>(
        &mut self,
        a: usize,
        forced: Option<bool>,
        rng: &mut R,
    ) -> (bool, bool) {
        assert!(a < self.n, "qubit out of range");
        self.set_layout(Layout::GeneratorMajor);
        let w = self.w;
        let p = match (0..self.np).find(|&i| get_bit(self.xs.line(i), a)) {
            None => return (self.deterministic_outcome(a), false),
            Some(p) => p,
        };
        // Random outcome. Every other generator that anticommutes with Z_a
        // (has an X/Y on qubit a) is multiplied by stabilizer p, so that p
        // is the only one left; then p is replaced by ±Z_a.
        let px = self.xs.line(p).to_vec();
        let pz = self.zs.line(p).to_vec();
        let pr = get_bit(&self.rs, p);
        let (wa, ba) = (a / 64, 1u64 << (a % 64));

        let update = |xm: &mut BitMatrix, zm: &mut BitMatrix, r: &[u64], skip: usize| {
            xm.raw_mut()
                .par_chunks_mut(w)
                .zip(zm.raw_mut().par_chunks_mut(w))
                .enumerate()
                .filter(|(i, (x, _))| *i != skip && x[wa] & ba != 0)
                .map(|(i, (x, z))| (i, rowsum(x, z, get_bit(r, i), &px, &pz, pr)))
                .collect::<Vec<_>>()
        };
        let new_rs = update(&mut self.xs, &mut self.zs, &self.rs, p);
        let new_rd = update(&mut self.xd, &mut self.zd, &self.rd, p);
        for (i, r) in new_rs {
            set_bit(&mut self.rs, i, r);
        }
        for (i, r) in new_rd {
            set_bit(&mut self.rd, i, r);
        }
        // destabilizer p <- old stabilizer p; stabilizer p <- (-1)^outcome Z_a
        self.xd.line_mut(p).copy_from_slice(&px);
        self.zd.line_mut(p).copy_from_slice(&pz);
        set_bit(&mut self.rd, p, pr);
        let outcome = forced.unwrap_or_else(|| rng.random_bool(0.5));
        self.xs.line_mut(p).fill(0);
        let zl = self.zs.line_mut(p);
        zl.fill(0);
        zl[wa] = ba;
        set_bit(&mut self.rs, p, outcome);
        (outcome, true)
    }

    /// Measures qubit `a`, collapsing the state.
    pub fn measure_qubit<R: Rng + ?Sized>(&mut self, a: usize, rng: &mut R) -> bool {
        self.measure_with(a, None, rng).0
    }

    /// Resets qubit `a` to `|0>` (measure, then flip if the outcome was 1).
    pub fn reset_qubit<R: Rng + ?Sized>(&mut self, a: usize, rng: &mut R) -> bool {
        let m = self.measure_qubit(a, rng);
        if m {
            self.x(a);
        }
        m
    }

    /// Measures every qubit in order.
    ///
    /// Measuring qubits one at a time costs `O(n^2)` per deterministic
    /// outcome in CHP (each one multiplies up to `n` rows), so `O(n^3)` for
    /// all of them. Instead this brings the stabilizers into row-echelon form
    /// once (see [`RefTableau::sample`]), draws one bit string from the
    /// resulting affine subspace, and resets the tableau to that basis state.
    pub fn measure_all<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Vec<bool> {
        let bits = self.sample(1, rng).pop().expect("one sample");
        self.reset_to_basis_state(&bits);
        bits
    }

    /// Measures qubits one at a time with the plain CHP procedure (kept for
    /// comparison with [`RefTableau::measure_all`]).
    pub fn measure_all_sequential<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Vec<bool> {
        (0..self.n).map(|q| self.measure_qubit(q, rng)).collect()
    }

    /// Draws `shots` full measurement records without disturbing the state.
    ///
    /// A stabilizer state is a uniform superposition over an affine
    /// subspace `x0 + span(u_1..u_k)` of bit strings. Gaussian elimination
    /// on the stabilizer rows (keeping the destabilizers consistent, so the
    /// state is unchanged) separates `k` generators with X parts, whose X
    /// bits are the `u_i`, from `n - k` Z-only generators `±Z^v`, which fix
    /// `v·x` and give `x0` by back-substitution. Each shot is then
    /// `x0 xor` a random subset of the `u_i`.
    pub fn sample<R: Rng + ?Sized>(&mut self, shots: usize, rng: &mut R) -> Vec<Vec<bool>> {
        self.set_layout(Layout::GeneratorMajor);
        let xp = self.echelon(0, true);
        let kx = xp.len();
        let zp = self.echelon(kx, false);
        debug_assert_eq!(kx + zp.len(), self.np, "stabilizers must be independent");
        let w = self.w;
        let mut x0 = vec![0u64; w];
        for (j, &c) in zp.iter().enumerate().rev() {
            let row = kx + j;
            let parity = self
                .zs
                .line(row)
                .iter()
                .zip(&x0)
                .map(|(a, b)| (a & b).count_ones())
                .sum::<u32>()
                & 1;
            set_bit(&mut x0, c, get_bit(&self.rs, row) ^ (parity == 1));
        }
        (0..shots)
            .map(|_| {
                let mut x = x0.clone();
                for i in 0..kx {
                    if rng.random_bool(0.5) {
                        for (a, b) in x.iter_mut().zip(self.xs.line(i)) {
                            *a ^= b;
                        }
                    }
                }
                (0..self.n).map(|q| get_bit(&x, q)).collect()
            })
            .collect()
    }

    /// Resets the tableau to the computational basis state `|bits>`.
    pub fn reset_to_basis_state(&mut self, bits: &[bool]) {
        assert_eq!(bits.len(), self.n);
        for m in [&mut self.xd, &mut self.zd, &mut self.xs, &mut self.zs] {
            m.raw_mut().fill(0);
        }
        for i in 0..self.np {
            self.xd.set(i, i, true);
            self.zs.set(i, i, true);
        }
        self.rd.fill(0);
        self.rs.fill(0);
        for (q, &b) in bits.iter().enumerate() {
            set_bit(&mut self.rs, q, b);
        }
        // identity blocks look the same in both layouts
    }

    /// `S_i <- S_i S_k` together with `D_k <- D_k D_i`, which keeps every
    /// destabilizer anticommuting with exactly its own stabilizer, so the
    /// tableau still describes the same state. Generator-major layout.
    fn stab_row_mul(&mut self, i: usize, k: usize) {
        let (ri, rk) = (get_bit(&self.rs, i), get_bit(&self.rs, k));
        let (xi, xk) = self.xs.two_lines_mut(i, k);
        let (zi, zk) = self.zs.two_lines_mut(i, k);
        let r = rowsum(xi, zi, ri, xk, zk, rk);
        set_bit(&mut self.rs, i, r);
        let (dk, di) = (get_bit(&self.rd, k), get_bit(&self.rd, i));
        let (xk, xi) = self.xd.two_lines_mut(k, i);
        let (zk, zi) = self.zd.two_lines_mut(k, i);
        let r = rowsum(xk, zk, dk, xi, zi, di);
        set_bit(&mut self.rd, k, r);
    }

    /// Swaps generator pairs `(D_i, S_i)` and `(D_k, S_k)`.
    fn swap_generators(&mut self, i: usize, k: usize) {
        for m in [&mut self.xd, &mut self.zd, &mut self.xs, &mut self.zs] {
            m.swap_lines(i, k);
        }
        for r in [&mut self.rd, &mut self.rs] {
            let (a, b) = (get_bit(r, i), get_bit(r, k));
            set_bit(r, i, b);
            set_bit(r, k, a);
        }
    }

    /// Row-echelon form of stabilizer rows `start..` with respect to their
    /// X bits (`use_x`) or Z bits. Returns the pivot column of each pivot
    /// row; pivot `j` ends up in row `start + j`.
    ///
    /// Finding pivot candidates means scanning one bit column across all
    /// rows, which is strided in this layout. To keep that cheap, the 64
    /// columns of one word are bit-transposed into a small window (64 lines
    /// of `np` bits) once per word, so each column scan reads `np / 64`
    /// contiguous words. Row operations update the window incrementally.
    fn echelon(&mut self, start: usize, use_x: bool) -> Vec<usize> {
        let (np, w) = (self.np, self.w);
        let mut k = start;
        let mut pivots = Vec::new();
        let mut win = vec![0u64; 64 * w];
        let mut blk = [0u64; 64];
        for b in 0..w {
            if k == np {
                break;
            }
            let part = if use_x { &self.xs } else { &self.zs };
            for g in k / 64..w {
                for (j, x) in blk.iter_mut().enumerate() {
                    *x = part.line(64 * g + j)[b];
                }
                bitmatrix::transpose64(&mut blk);
                for (c, x) in blk.iter().enumerate() {
                    win[c * w + g] = *x;
                }
            }
            for c in 0..64 {
                let line = &win[c * w..(c + 1) * w];
                let Some(r) = next_set_bit(line, k) else {
                    continue;
                };
                if r != k {
                    self.swap_generators(r, k);
                    for cc in 0..64 {
                        let l = &mut win[cc * w..(cc + 1) * w];
                        let (a, bb) = (get_bit(l, r), get_bit(l, k));
                        set_bit(l, r, bb);
                        set_bit(l, k, a);
                    }
                }
                let mut targets = Vec::new();
                let mut i = k + 1;
                while let Some(t) = next_set_bit(&win[c * w..(c + 1) * w], i) {
                    targets.push(t);
                    i = t + 1;
                }
                let pk = if use_x {
                    self.xs.line(k)[b]
                } else {
                    self.zs.line(k)[b]
                };
                for &t in &targets {
                    self.stab_row_mul(t, k);
                    let mut m = pk;
                    while m != 0 {
                        let cc = m.trailing_zeros() as usize;
                        win[cc * w + t / 64] ^= 1u64 << (t % 64);
                        m &= m - 1;
                    }
                }
                pivots.push(64 * b + c);
                k += 1;
                if k == np {
                    break;
                }
            }
        }
        pivots
    }

    /// Exact probability of observing `bits` (bit `q` of the index = qubit
    /// `q`) when measuring all qubits. Stabilizer states have a uniform
    /// distribution over an affine subspace, so this is `0` or `2^-k`.
    pub fn probability(&self, bits: usize) -> f64 {
        let mut t = self.clone();
        // outcomes are always forced, so the generator is never consulted
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(0);
        let mut p = 1.0;
        for q in 0..self.n {
            let want = (bits >> q) & 1 == 1;
            let (got, random) = t.measure_with(q, Some(want), &mut rng);
            if random {
                p *= 0.5;
            } else if got != want {
                return 0.0;
            }
        }
        p
    }

    /// The stabilizer generators as strings like `+XXZ` (qubit 0 first).
    pub fn stabilizers(&mut self) -> Vec<String> {
        self.set_layout(Layout::GeneratorMajor);
        (0..self.n)
            .map(|i| {
                let mut s = String::with_capacity(self.n + 1);
                s.push(if get_bit(&self.rs, i) { '-' } else { '+' });
                for q in 0..self.n {
                    s.push(match (self.xs.get(i, q), self.zs.get(i, q)) {
                        (false, false) => 'I',
                        (true, false) => 'X',
                        (true, true) => 'Y',
                        (false, true) => 'Z',
                    });
                }
                s
            })
            .collect()
    }
}

impl Simulator for RefTableau {
    fn name(&self) -> &'static str {
        "stabilizer"
    }
    fn num_qubits(&self) -> usize {
        self.n
    }
    fn apply(&mut self, gate: &Gate) -> Result<(), SimError> {
        self.apply_gate(gate)
    }
    fn measure(&mut self, q: usize, rng: &mut dyn RngCore) -> Result<bool, SimError> {
        if q >= self.n {
            return Err(SimError::QubitOutOfRange {
                qubit: q,
                num_qubits: self.n,
            });
        }
        Ok(self.measure_qubit(q, rng))
    }
    fn reset(&mut self, q: usize, rng: &mut dyn RngCore) -> Result<(), SimError> {
        if q >= self.n {
            return Err(SimError::QubitOutOfRange {
                qubit: q,
                num_qubits: self.n,
            });
        }
        self.reset_qubit(q, rng);
        Ok(())
    }
}
