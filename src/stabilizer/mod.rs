//! Stabilizer simulation with the Aaronson–Gottesman (CHP) tableau.
//!
//! By the Gottesman–Knill theorem, a state produced from `|0...0>` by
//! Clifford gates (H, S, CNOT and their products) is fully described by `n`
//! commuting Pauli operators that stabilise it. The CHP tableau stores those
//! `n` *stabilizers* plus `n` *destabilizers* as rows of `2n` bits (`x` and
//! `z` parts) and a sign bit. Memory is therefore `4n^2` bits instead of
//! `2^n` amplitudes: 10,000 qubits fit in 50 MB, but the quadratic growth
//! still hits a wall (1M qubits would need 125 GB).
//!
//! Implementation notes:
//!
//! * The four `n x n` blocks (destabilizer x/z, stabilizer x/z) are bit
//!   packed into `u64` words, with `n` rounded up to a multiple of 64. Extra
//!   padding qubits simply stay in `|0>`.
//! * The tableau `C` (the Clifford with `D_i = C X_i C†`, `S_i = C Z_i C†`)
//!   is stored *qubit-major*: a line per qubit holding one bit per
//!   generator, so a gate is a handful of word operations over `2n/64`
//!   words. The same lines, read the other way round, are the rows of the
//!   inverse tableau `T = C†`: `T(X_q) = C† X_q C` has x bits `zs[q]` and z
//!   bits `zd[q]`, `T(Z_q)` has x bits `xs[q]` and z bits `xd[q]` (the
//!   symplectic inverse is a transpose with the blocks swapped).
//! * Besides the generator signs, the tableau keeps the signs of the
//!   inverse rows (`sx`, `sz`), as Stim does. A Z measurement of qubit `a`
//!   is deterministic iff `T(Z_a)` has no X part (`xs[a] == 0`), and then
//!   its outcome is just the sign of `T(Z_a)`: `O(n/64)` instead of the
//!   `O(n^2/64)` row products of plain CHP. Gates update the inverse signs
//!   with one word-parallel Pauli product per two-qubit gate.
//! * A random measurement is done Stim-style by inserting gates "at the
//!   beginning of time" (acting on generator indices): CNOTs from the pivot
//!   generator make it the only one anticommuting with `Z_a`, and an H (or
//!   H_YZ) plus an optional X on the pivot collapses the state. All of it is
//!   one pass over the qubit-major lines, so the tableau never has to be
//!   transposed for measurements.
//! * Only [`Tableau::sample`] (and [`Tableau::stabilizers`]) switch to the
//!   generator-major layout, with an in-place 64x64-block bit transpose.
//! * Pauli products count the phase with bit-sliced mod-4 counters, two
//!   popcounts per product instead of two per word.

pub mod bitmatrix;
mod ref_bitmatrix;
#[doc(hidden)]
pub mod reference;
pub mod symphase;

use crate::circuit::{check_gate, SimError, Simulator};
use crate::gate::{is_multiple_of_half_pi, Gate};
use bitmatrix::BitMatrix;
use rand::{Rng, RngCore};
use std::f64::consts::FRAC_PI_2;

/// Upper bound on tableau memory (1 GiB, i.e. up to 46,336 qubits).
pub const MAX_TABLEAU_BYTES: u128 = 1 << 30;

/// Bytes used by the tableau for `n` qubits (four `np x np` bit blocks).
pub fn tableau_bytes(n: usize) -> u128 {
    let np = n.div_ceil(64).max(1) as u128 * 64;
    np * np / 2
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Line `q` of each block holds column `q` (one bit per generator).
    QubitMajor,
    /// Line `i` of each block holds generator `i` (one bit per qubit).
    GeneratorMajor,
}

/// The CHP stabilizer tableau, with inverse-tableau signs.
#[derive(Clone, Debug)]
pub struct Tableau {
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
    /// Sign bits of the inverse rows `C† X_q C` / `C† Z_q C`, one per qubit.
    sx: Vec<u64>,
    sz: Vec<u64>,
    layout: Layout,
    /// Number of layout switches (transposes) so far, for profiling.
    switches: usize,
    /// Whether gates keep `sx`/`sz` up to date. When off, they are stale
    /// and the signs a measurement needs are recomputed on demand.
    track: bool,
    /// Deterministic measurements done without tracking since it was
    /// switched off (for the automatic switch back on).
    untracked_meas: usize,
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

#[inline]
fn flip_bit(v: &mut [u64], i: usize, b: bool) {
    v[i / 64] ^= (b as u64) << (i % 64);
}

/// All-ones if `b` is set.
#[inline(always)]
fn bcast(b: u64) -> u64 {
    0u64.wrapping_sub(b & 1)
}

#[inline(always)]
fn parity(x: u64) -> u64 {
    (x.count_ones() & 1) as u64
}

/// Inclusive prefix XOR inside a word: bit `i` of the result is the XOR of
/// bits `0..=i` of `v`.
#[inline(always)]
fn prefix_xor(mut v: u64) -> u64 {
    v ^= v << 1;
    v ^= v << 2;
    v ^= v << 4;
    v ^= v << 8;
    v ^= v << 16;
    v ^= v << 32;
    v
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

/// Word-parallel phase counter for products of Pauli strings.
///
/// For single-qubit Paulis `a, b` (bits `(x, z)`, with `(1, 1) = Y`),
/// `a·b = i^g (a xor b)` with `g ∈ {0, 1, 3}` (mod 4). `mul` adds `g` at
/// every bit position of a word into a 2-bit counter per position (`c1`
/// low bit, `c2` high bit); `total` sums all positions mod 4. This is the
/// formulation Stim uses: no popcount per word.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Phase {
    c1: u64,
    c2: u64,
}

impl Phase {
    /// Accumulates the phase of `(ax, az) · (bx, bz)` at each bit position.
    #[inline(always)]
    pub fn mul(&mut self, ax: u64, az: u64, bx: u64, bz: u64) {
        let x1z2 = ax & bz;
        let anti = (bx & az) ^ x1z2;
        self.c2 ^= (self.c1 ^ ax ^ bx ^ az ^ bz ^ x1z2) & anti;
        self.c1 ^= anti;
    }

    /// Adds 1 at every set bit of `m`.
    #[inline(always)]
    pub fn add(&mut self, m: u64) {
        self.c2 ^= self.c1 & m;
        self.c1 ^= m;
    }

    /// Sum of two counters.
    #[inline(always)]
    pub fn merge(self, o: Phase) -> Phase {
        Phase {
            c1: self.c1 ^ o.c1,
            c2: self.c2 ^ o.c2 ^ (self.c1 & o.c1),
        }
    }

    /// Total over all bit positions, mod 4.
    #[inline]
    pub fn total(&self) -> u32 {
        (self.c1.count_ones() + 2 * self.c2.count_ones()) & 3
    }
}

/// Phase exponent (mod 4) of the product of the Pauli strings `(ax, az)`
/// and `(bx, bz)`. Read-only, with four independent lane accumulators over
/// `chunks_exact(4)` so the compiler can keep them in vector registers.
#[inline]
fn product_phase(ax: &[u64], az: &[u64], bx: &[u64], bz: &[u64]) -> u32 {
    let mut acc = [Phase::default(); 4];
    let chunks = ax
        .chunks_exact(4)
        .zip(az.chunks_exact(4))
        .zip(bx.chunks_exact(4).zip(bz.chunks_exact(4)));
    for ((ax, az), (bx, bz)) in chunks {
        for l in 0..4 {
            acc[l].mul(ax[l], az[l], bx[l], bz[l]);
        }
    }
    let r = ax.len() - ax.len() % 4;
    for k in r..ax.len() {
        acc[0].mul(ax[k], az[k], bx[k], bz[k]);
    }
    acc[0].merge(acc[1]).merge(acc[2].merge(acc[3])).total()
}

/// Replaces Pauli row `h` with the product `i * h` (CHP's `rowsum(h, i)`)
/// and returns the new sign bit of `h`. Generator-major helper.
#[inline]
fn rowsum(hx: &mut [u64], hz: &mut [u64], hr: bool, ix: &[u64], iz: &[u64], ir: bool) -> bool {
    let mut ph = Phase::default();
    for k in 0..hx.len() {
        let (x2, z2) = (hx[k], hz[k]);
        ph.mul(ix[k], iz[k], x2, z2);
        hx[k] = x2 ^ ix[k];
        hz[k] = z2 ^ iz[k];
    }
    let m = (2 * (hr as u32) + 2 * (ir as u32) + ph.total()) & 3;
    debug_assert!(m == 0 || m == 2, "rowsum of anticommuting rows");
    m == 2
}

/// Bit `p` of every line of `m`, packed as a vector over lines.
fn gather_column(m: &BitMatrix, p: usize) -> Vec<u64> {
    let (np, w) = (m.size(), m.words());
    let (pw, pb) = (p / 64, p % 64);
    let raw = m.raw();
    let mut out = vec![0u64; w];
    for q in 0..np {
        out[q / 64] |= ((raw[q * w + pw] >> pb) & 1) << (q % 64);
    }
    out
}

/// Inverse of [`gather_column`].
fn scatter_column(m: &mut BitMatrix, p: usize, col: &[u64]) {
    let (np, w) = (m.size(), m.words());
    let (pw, pb) = (p / 64, p % 64);
    let raw = m.raw_mut();
    for q in 0..np {
        let word = &mut raw[q * w + pw];
        *word = (*word & !(1u64 << pb)) | (((col[q / 64] >> (q % 64)) & 1) << pb);
    }
}

impl Tableau {
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
        let w = np / 64;
        Ok(Tableau {
            n,
            np,
            w,
            // destabilizer i = X_i, stabilizer i = Z_i; identity in both layouts
            xd: BitMatrix::identity(np),
            zd: BitMatrix::zeros(np),
            xs: BitMatrix::zeros(np),
            zs: BitMatrix::identity(np),
            rd: vec![0; w],
            rs: vec![0; w],
            sx: vec![0; w],
            sz: vec![0; w],
            layout: Layout::QubitMajor,
            switches: 0,
            track: true,
            untracked_meas: 0,
        })
    }

    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// Bytes of tableau storage.
    pub fn bytes(&self) -> usize {
        self.xd.bytes() * 4 + (self.rd.len() * 4) * 8
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// How many times the layout has been switched (each switch transposes
    /// all four blocks).
    pub fn layout_switches(&self) -> usize {
        self.switches
    }

    fn set_layout(&mut self, layout: Layout) {
        if self.layout != layout {
            let (xd, zd, xs, zs) = (&mut self.xd, &mut self.zd, &mut self.xs, &mut self.zs);
            rayon::join(
                || rayon::join(|| xd.transpose_in_place(), || zd.transpose_in_place()),
                || rayon::join(|| xs.transpose_in_place(), || zs.transpose_in_place()),
            );
            self.layout = layout;
            self.switches += 1;
        }
    }

    pub fn h(&mut self, a: usize) {
        self.set_layout(Layout::QubitMajor);
        for (x, z, r) in [
            (&mut self.xd, &mut self.zd, &mut self.rd),
            (&mut self.xs, &mut self.zs, &mut self.rs),
        ] {
            let (x, z) = (x.line_mut(a), z.line_mut(a));
            for k in 0..r.len() {
                r[k] ^= x[k] & z[k];
            }
            x.swap_with_slice(z);
        }
        let (bx, bz) = (get_bit(&self.sx, a), get_bit(&self.sz, a));
        set_bit(&mut self.sx, a, bz);
        set_bit(&mut self.sz, a, bx);
    }

    /// S (`dagger = false`) or S† on qubit `a`.
    fn s_gate(&mut self, a: usize, dagger: bool) {
        self.set_layout(Layout::QubitMajor);
        let (w, track) = (self.w, self.track);
        let (xd, zd) = (&mut self.xd.line_mut(a)[..w], &mut self.zd.line_mut(a)[..w]);
        let (xs, zs) = (&mut self.xs.line_mut(a)[..w], &mut self.zs.line_mut(a)[..w]);
        let (rd, rs) = (&mut self.rd[..w], &mut self.rs[..w]);
        let mut ph = Phase::default();
        for k in 0..w {
            // T(X_a) · T(Z_a), before the update
            if track {
                ph.mul(zs[k], zd[k], xs[k], xd[k]);
            }
            if dagger {
                rd[k] ^= xd[k] & !zd[k];
                rs[k] ^= xs[k] & !zs[k];
            } else {
                rd[k] ^= xd[k] & zd[k];
                rs[k] ^= xs[k] & zs[k];
            }
            zd[k] ^= xd[k];
            zs[k] ^= xs[k];
        }
        // S: T'(X) = T(S† X S) = T(-Y) = -i T(X) T(Z); S†: T'(X) = +i T(X) T(Z)
        let e = 2 * get_bit(&self.sx, a) as u32
            + 2 * get_bit(&self.sz, a) as u32
            + ph.total()
            + if dagger { 1 } else { 3 };
        debug_assert!(!track || e % 2 == 0);
        set_bit(&mut self.sx, a, e & 3 == 2);
    }

    pub fn s(&mut self, a: usize) {
        self.s_gate(a, false);
    }

    pub fn sdg(&mut self, a: usize) {
        self.s_gate(a, true);
    }

    /// Pauli gate with x/z exponents (`X = (1,0)`, `Z = (0,1)`, `Y = (1,1)`).
    fn pauli(&mut self, a: usize, px: bool, pz: bool) {
        self.set_layout(Layout::QubitMajor);
        for (x, z, r) in [
            (&self.xd, &self.zd, &mut self.rd),
            (&self.xs, &self.zs, &mut self.rs),
        ] {
            let (x, z) = (x.line(a), z.line(a));
            for k in 0..r.len() {
                // a row anticommutes with X_a iff it has z on a, etc.
                r[k] ^= (z[k] & bcast(px as u64)) ^ (x[k] & bcast(pz as u64));
            }
        }
        // T'(Z_a) = T(P Z_a P) flips for P = X, Y; T'(X_a) for P = Z, Y
        flip_bit(&mut self.sz, a, px);
        flip_bit(&mut self.sx, a, pz);
    }

    pub fn x(&mut self, a: usize) {
        self.pauli(a, true, false);
    }

    pub fn z(&mut self, a: usize) {
        self.pauli(a, false, true);
    }

    pub fn y(&mut self, a: usize) {
        self.pauli(a, true, true);
    }

    pub fn cnot(&mut self, c: usize, t: usize) {
        self.set_layout(Layout::QubitMajor);
        let w = self.w;
        let (xdc, xdt) = self.xd.two_lines_mut(c, t);
        let (zdc, zdt) = self.zd.two_lines_mut(c, t);
        let (xsc, xst) = self.xs.two_lines_mut(c, t);
        let (zsc, zst) = self.zs.two_lines_mut(c, t);
        let (xdc, xdt, zdc, zdt) = (&mut xdc[..w], &mut xdt[..w], &mut zdc[..w], &mut zdt[..w]);
        let (xsc, xst, zsc, zst) = (&mut xsc[..w], &mut xst[..w], &mut zsc[..w], &mut zst[..w]);
        let (rd, rs) = (&mut self.rd[..w], &mut self.rs[..w]);
        // T(X_c) T(X_t) and T(Z_c) T(Z_t), before the update
        let (ex, ez) = if self.track {
            (
                product_phase(zsc, zdc, zst, zdt),
                product_phase(xsc, xdc, xst, xdt),
            )
        } else {
            (0, 0)
        };
        // one block at a time (fewer concurrent streams, as in plain CHP)
        for k in 0..w {
            rd[k] ^= xdc[k] & zdt[k] & !(xdt[k] ^ zdc[k]);
            xdt[k] ^= xdc[k];
            zdc[k] ^= zdt[k];
        }
        for k in 0..w {
            rs[k] ^= xsc[k] & zst[k] & !(xst[k] ^ zsc[k]);
            xst[k] ^= xsc[k];
            zsc[k] ^= zst[k];
        }
        // T'(X_c) = T(X_c) T(X_t), T'(Z_t) = T(Z_c) T(Z_t)
        debug_assert!(ex % 2 == 0 && ez % 2 == 0);
        let st = get_bit(&self.sx, t);
        flip_bit(&mut self.sx, c, st ^ (ex == 2));
        let sc = get_bit(&self.sz, c);
        flip_bit(&mut self.sz, t, sc ^ (ez == 2));
    }

    pub fn cz(&mut self, a: usize, b: usize) {
        self.set_layout(Layout::QubitMajor);
        let w = self.w;
        let (xda, xdb) = self.xd.two_lines_mut(a, b);
        let (zda, zdb) = self.zd.two_lines_mut(a, b);
        let (xsa, xsb) = self.xs.two_lines_mut(a, b);
        let (zsa, zsb) = self.zs.two_lines_mut(a, b);
        let (xda, xdb, zda, zdb) = (&mut xda[..w], &mut xdb[..w], &mut zda[..w], &mut zdb[..w]);
        let (xsa, xsb, zsa, zsb) = (&mut xsa[..w], &mut xsb[..w], &mut zsa[..w], &mut zsb[..w]);
        let (rd, rs) = (&mut self.rd[..w], &mut self.rs[..w]);
        // T(X_a) T(Z_b) and T(Z_a) T(X_b), before the update
        let (ea, eb) = if self.track {
            (
                product_phase(zsa, zda, xsb, xdb),
                product_phase(xsa, xda, zsb, zdb),
            )
        } else {
            (0, 0)
        };
        for k in 0..w {
            rd[k] ^= xda[k] & xdb[k] & (zda[k] ^ zdb[k]);
            zda[k] ^= xdb[k];
            zdb[k] ^= xda[k];
        }
        for k in 0..w {
            rs[k] ^= xsa[k] & xsb[k] & (zsa[k] ^ zsb[k]);
            zsa[k] ^= xsb[k];
            zsb[k] ^= xsa[k];
        }
        debug_assert!(ea % 2 == 0 && eb % 2 == 0);
        let (sza, szb) = (get_bit(&self.sz, a), get_bit(&self.sz, b));
        flip_bit(&mut self.sx, a, szb ^ (ea == 2));
        flip_bit(&mut self.sx, b, sza ^ (eb == 2));
    }

    pub fn swap(&mut self, a: usize, b: usize) {
        self.set_layout(Layout::QubitMajor);
        for m in [&mut self.xd, &mut self.zd, &mut self.xs, &mut self.zs] {
            m.swap_lines(a, b);
        }
        for s in [&mut self.sx, &mut self.sz] {
            let (x, y) = (get_bit(s, a), get_bit(s, b));
            set_bit(s, a, y);
            set_bit(s, b, x);
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
    /// state.
    pub fn peek(&mut self, a: usize) -> Option<bool> {
        self.set_layout(Layout::QubitMajor);
        if self.xs.line(a).iter().any(|&x| x != 0) {
            None
        } else {
            Some(self.z_sign(a))
        }
    }

    /// Measures qubit `a`. If the outcome is random and `forced` is given,
    /// that outcome is selected instead of drawing one. Returns
    /// `(outcome, was_random)`. A random outcome consumes exactly one
    /// `random_bool(0.5)` from `rng`, as in the original CHP code.
    pub fn measure_with<R: Rng + ?Sized>(
        &mut self,
        a: usize,
        forced: Option<bool>,
        rng: &mut R,
    ) -> (bool, bool) {
        assert!(a < self.n, "qubit out of range");
        self.set_layout(Layout::QubitMajor);
        let w = self.w;
        // T(Z_a) = C† Z_a C has x bits xs[a]: generators anticommuting with Z_a
        let Some(p) = next_set_bit(self.xs.line(a), 0) else {
            let m = self.z_sign(a);
            if !self.track {
                // ski rental: once the on-demand signs have cost as much as
                // recomputing all of them, recompute and track again
                self.untracked_meas += 1;
                if self.untracked_meas > 2 * self.np {
                    self.set_sign_tracking(true);
                }
            }
            return (m, false);
        };
        let mut kmask = self.xs.line(a).to_vec();
        kmask[p / 64] &= !(1u64 << (p % 64));
        if kmask.iter().any(|&k| k != 0) {
            self.fanout_cnot(p, &kmask);
        }
        // Now X/Y of T(Z_a) sits on generator p only. Rotate it to Z there
        // (H or H_YZ "at the beginning of time"), then pick the outcome.
        let mut xdp = gather_column(&self.xd, p);
        let mut zdp = gather_column(&self.zd, p);
        let mut xsp = gather_column(&self.xs, p);
        let mut zsp = gather_column(&self.zs, p);
        let (mut rdp, mut rsp) = (get_bit(&self.rd, p), get_bit(&self.rs, p));
        if !get_bit(&xdp, a) {
            // H on generator p: (D_p, S_p) <- (S_p, D_p); inverse rows with
            // Y on p change sign
            for j in 0..w {
                self.sx[j] ^= zsp[j] & zdp[j];
                self.sz[j] ^= xsp[j] & xdp[j];
            }
            std::mem::swap(&mut xdp, &mut xsp);
            std::mem::swap(&mut zdp, &mut zsp);
            std::mem::swap(&mut rdp, &mut rsp);
        } else {
            // H_YZ on generator p: D_p <- -D_p, S_p <- i D_p S_p; inverse rows
            // with X on p change sign
            let mut ph = Phase::default();
            for j in 0..w {
                self.sx[j] ^= zsp[j] & !zdp[j];
                self.sz[j] ^= xsp[j] & !xdp[j];
                ph.mul(xdp[j], zdp[j], xsp[j], zsp[j]);
                xsp[j] ^= xdp[j];
                zsp[j] ^= zdp[j];
            }
            let e = 2 * rdp as u32 + 2 * rsp as u32 + ph.total() + 1;
            debug_assert_eq!(e % 2, 0);
            rsp = e & 3 == 2;
            rdp = !rdp;
        }
        scatter_column(&mut self.xd, p, &xdp);
        scatter_column(&mut self.zd, p, &zdp);
        scatter_column(&mut self.xs, p, &xsp);
        scatter_column(&mut self.zs, p, &zsp);
        set_bit(&mut self.rd, p, rdp);
        let outcome = forced.unwrap_or_else(|| rng.random_bool(0.5));
        // T(Z_a) is now Z-only; its sign is the outcome without a flip
        set_bit(&mut self.rs, p, rsp);
        if outcome != self.z_sign(a) {
            // X on generator p: S_p <- -S_p; inverse rows with Z/Y on p flip
            rsp = !rsp;
            for j in 0..w {
                self.sx[j] ^= zdp[j];
                self.sz[j] ^= xdp[j];
            }
        }
        set_bit(&mut self.rs, p, rsp);
        debug_assert!(self.xs.line(a).iter().all(|&x| x == 0));
        debug_assert_eq!(self.z_sign(a), outcome);
        (outcome, true)
    }

    /// Sign of `T(Z_a)`: the outcome of measuring qubit `a` when that is
    /// deterministic. `O(n/64)` with tracking, else an `O(n^2/64)` pass.
    fn z_sign(&self, a: usize) -> bool {
        if self.track {
            get_bit(&self.sz, a)
        } else {
            self.inverse_sign(self.xs.line(a), self.xd.line(a))
        }
    }

    /// Whether gates maintain the inverse-row signs (default on).
    pub fn sign_tracking(&self) -> bool {
        self.track
    }

    /// Switches inverse-sign tracking on or off.
    ///
    /// With tracking (the default), every S/CNOT/CZ pays one extra Pauli
    /// product phase over `n/64` words, and a deterministic measurement is
    /// `O(n/64)`. Without it, gates cost the same as plain CHP, and a
    /// deterministic measurement computes the one sign it needs in a single
    /// `O(n^2/64)` pass (still no layout switch). Gate-heavy circuits with
    /// few single-qubit measurements (e.g. GHZ, then [`Tableau::measure_all`])
    /// run faster without it.
    ///
    /// Switching it back on recomputes all `2n` inverse signs (`O(n^3/64)`).
    /// This also happens automatically once `2n` deterministic
    /// measurements have been done untracked, so a wrong choice costs at
    /// most about a factor two.
    pub fn set_sign_tracking(&mut self, on: bool) {
        if on && !self.track {
            self.set_layout(Layout::QubitMajor);
            let mut sx = vec![0u64; self.w];
            let mut sz = vec![0u64; self.w];
            for q in 0..self.np {
                // T(X_q): x = zs[q], z = zd[q]; T(Z_q): x = xs[q], z = xd[q]
                set_bit(
                    &mut sx,
                    q,
                    self.inverse_sign(self.zs.line(q), self.zd.line(q)),
                );
                set_bit(
                    &mut sz,
                    q,
                    self.inverse_sign(self.xs.line(q), self.xd.line(q)),
                );
            }
            self.sx = sx;
            self.sz = sz;
        }
        self.track = on;
        self.untracked_meas = 0;
    }

    /// Sign bit of the inverse row with generator coefficients `u` (X part)
    /// and `v` (Z part), recomputed from the forward tableau.
    ///
    /// If `C† P C = ± prod_i X_i^u_i Z_i^v_i` (tableau convention, `(1, 1)`
    /// meaning `Y_i`), conjugating back by `C` gives
    /// `P = ± i^{|u & v|} (-1)^{u·rd + v·rs} prod_{i in u} D_i prod_{i in v} S_i`
    /// (all destabilizers first; only `D_i`, `S_i` with equal `i`
    /// anticommute, and they stay in that order). The phase of that ordered
    /// product is summed qubit by qubit: for single-qubit factors
    /// `(x_j, z_j)`, `prod_j = i^e (X, Z)` with
    /// `e = sum_j x_j z_j - X Z + 2 sum_{j<k} z_j x_k`. `P` is a single
    /// `X_q` or `Z_q`, so the result has no Y and `X Z = 0`. Qubit-major,
    /// one pass over the lines.
    fn inverse_sign(&self, u: &[u64], v: &[u64]) -> bool {
        let w = self.w;
        let (xd, zd) = (self.xd.raw(), self.zd.raw());
        let (xs, zs) = (self.xs.raw(), self.zs.raw());
        let nzu: Vec<usize> = (0..w).filter(|&j| u[j] != 0).collect();
        let nzv: Vec<usize> = (0..w).filter(|&j| v[j] != 0).collect();
        let mut e: u32 = 0;
        let mut pair: u64 = 0;
        for q in 0..self.np {
            let r = q * w;
            let mut carry = 0u64;
            for (x, z, mask, nz) in [(xd, zd, u, &nzu), (xs, zs, v, &nzv)] {
                for &j in nz {
                    let (xk, zk) = (x[r + j] & mask[j], z[r + j] & mask[j]);
                    e = e.wrapping_add((xk & zk).count_ones());
                    let iz = prefix_xor(zk);
                    pair ^= xk & (iz ^ zk ^ carry);
                    carry ^= bcast(iz >> 63);
                }
            }
        }
        let uv: u32 = u.iter().zip(v).map(|(a, b)| (a & b).count_ones()).sum();
        let ru = u.iter().zip(&self.rd).fold(0, |acc, (a, b)| acc ^ (a & b));
        let rv = v.iter().zip(&self.rs).fold(0, |acc, (a, b)| acc ^ (a & b));
        let e = e
            .wrapping_add(uv)
            .wrapping_add(2 * (parity(pair) ^ parity(ru) ^ parity(rv)) as u32);
        debug_assert_eq!(e % 2, 0, "inverse row must be Hermitian");
        e & 3 == 2
    }

    /// CNOTs from generator `p` to every generator in `kmask`, applied at the
    /// beginning of time (they act on `|0...0>` trivially, so the state is
    /// unchanged): `S_k <- S_k S_p` for each `k`, `D_p <- D_p prod_k D_k`.
    /// One pass over all qubit lines, which also updates the generator signs
    /// and the inverse-row signs.
    fn fanout_cnot(&mut self, p: usize, kmask: &[u64]) {
        let (np, w) = (self.np, self.w);
        let (pw, pb) = (p / 64, p % 64);
        let nz: Vec<usize> = (0..w).filter(|&j| kmask[j] != 0).collect();
        // per generator word: phase of S_k * S_p
        let mut cs = vec![Phase::default(); w];
        // phase of the ordered product D_p D_k1 D_k2 ...: Y count, the
        // x_total*z_total correction, and the (z before x) pair parity
        let mut dy = Phase::default();
        let mut dy_scalar: u32 = 0;
        let mut dxz: u32 = 0;
        let mut dpair: u64 = 0;
        let mut dpair_scalar: u64 = 0;
        let xd = self.xd.raw_mut();
        let zd = self.zd.raw_mut();
        let xs = self.xs.raw_mut();
        let zs = self.zs.raw_mut();
        for q in 0..np {
            let r = q * w..(q + 1) * w;
            let (xd, zd, xs, zs) = (
                &mut xd[r.clone()],
                &mut zd[r.clone()],
                &mut xs[r.clone()],
                &mut zs[r],
            );
            let xdp = (xd[pw] >> pb) & 1;
            let zdp = (zd[pw] >> pb) & 1;
            let xsp = (xs[pw] >> pb) & 1;
            let zsp = (zs[pw] >> pb) & 1;
            let (bxs, bzs, bxd, bzd) = (bcast(xsp), bcast(zsp), bcast(xdp), bcast(zdp));
            let (mut carry_z, mut carry_x) = (0u64, 0u64);
            let (mut parx, mut parz) = (0u64, 0u64);
            let (mut tx, mut tz) = (0u64, 0u64);
            for &j in &nz {
                let km = kmask[j];
                let (xdj, zdj, xsj, zsj) = (xd[j], zd[j], xs[j], zs[j]);
                cs[j].mul(xsj, zsj, bxs, bzs);
                let xk = xdj & km;
                let zk = zdj & km;
                let iz = prefix_xor(zk);
                let zpre = iz ^ zk ^ carry_z;
                carry_z ^= bcast(iz >> 63);
                let ix = prefix_xor(xk);
                let xpre = ix ^ xk ^ carry_x;
                carry_x ^= bcast(ix >> 63);
                dy.add(xk & zk);
                dpair ^= xk & zpre;
                parx ^= xk;
                parz ^= zk;
                // conjugating T(X_q) (x = zs, z = zd) and T(Z_q) (x = xs,
                // z = xd) by the CNOTs: the sign flips by
                // x_p * sum_k z_k (1 + x_k + z_p before k)
                tx ^= zk & !(zsj ^ zpre ^ bzd);
                tz ^= xk & !(xsj ^ xpre ^ bxd);
                xs[j] = xsj ^ (km & bxs);
                zs[j] = zsj ^ (km & bzs);
            }
            let (px, pz) = (parity(parx), parity(parz));
            xd[pw] ^= px << pb;
            zd[pw] ^= pz << pb;
            dy_scalar += (xdp & zdp) as u32;
            dpair_scalar ^= zdp & px;
            dxz += ((xdp ^ px) & (zdp ^ pz)) as u32;
            self.sx[q / 64] ^= (zsp & parity(tx)) << (q % 64);
            self.sz[q / 64] ^= (xsp & parity(tz)) << (q % 64);
        }
        let rsp = bcast(get_bit(&self.rs, p) as u64);
        for &j in &nz {
            debug_assert_eq!(cs[j].c1 & kmask[j], 0, "S_k and S_p must commute");
            self.rs[j] ^= kmask[j] & (cs[j].c2 ^ rsp);
        }
        let e =
            (dy.total() + dy_scalar + 2 * (parity(dpair) ^ dpair_scalar) as u32 + 4 * np as u32
                - dxz % 4)
                & 3;
        debug_assert_eq!(e % 2, 0, "destabilizers must commute");
        let rk = self
            .rd
            .iter()
            .zip(kmask)
            .fold(0u64, |acc, (r, k)| acc ^ (r & k));
        flip_bit(&mut self.rd, p, (parity(rk) == 1) ^ (e == 2));
    }

    /// Measures qubit `a`, collapsing the state.
    pub fn measure_qubit<R: Rng + ?Sized>(&mut self, a: usize, rng: &mut R) -> bool {
        self.measure_with(a, None, rng).0
    }

    /// Resets qubit `a` to `|0>`: measure, then flip if the outcome was 1.
    /// Returns the measurement outcome. (This is the reset channel: if `a`
    /// is entangled, its partners collapse consistently with a random
    /// outcome, exactly as in the state-vector backend. Forcing the outcome
    /// to 0 instead would bias them.)
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
    /// once (see [`Tableau::sample`]), draws one bit string from the
    /// resulting affine subspace, and resets the tableau to that basis state.
    pub fn measure_all<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Vec<bool> {
        let bits = self.sample(1, rng).pop().expect("one sample");
        self.reset_to_basis_state(&bits);
        bits
    }

    /// Measures qubits one at a time (each deterministic outcome is
    /// `O(n/64)` here, each random one a pass over the tableau).
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
        self.sx.fill(0);
        self.sz.fill(0);
        for (q, &b) in bits.iter().enumerate() {
            set_bit(&mut self.rs, q, b);
            set_bit(&mut self.sz, q, b);
        }
        // identity blocks look the same in both layouts
    }

    /// `S_i <- S_i S_k` together with `D_k <- D_k D_i` (a CNOT from
    /// generator `k` to generator `i` at the beginning of time), which keeps
    /// every destabilizer anticommuting with exactly its own stabilizer, so
    /// the tableau still describes the same state. Generator-major layout.
    fn stab_row_mul(&mut self, i: usize, k: usize) {
        // inverse rows: conjugation by CNOT(k -> i) flips the sign by
        // x_k z_i (1 + x_i + z_k); T(X_q) has x = zs, z = zd, T(Z_q) x = xs, z = xd
        {
            let (zsk, zsi, zdk, zdi) = (
                self.zs.line(k),
                self.zs.line(i),
                self.zd.line(k),
                self.zd.line(i),
            );
            let (xsk, xsi, xdk, xdi) = (
                self.xs.line(k),
                self.xs.line(i),
                self.xd.line(k),
                self.xd.line(i),
            );
            for j in 0..self.w {
                self.sx[j] ^= zsk[j] & zdi[j] & !(zsi[j] ^ zdk[j]);
                self.sz[j] ^= xsk[j] & xdi[j] & !(xsi[j] ^ xdk[j]);
            }
        }
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

impl Simulator for Tableau {
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn bell_stabilizers() {
        let mut t = Tableau::new(2);
        t.h(0);
        t.cnot(0, 1);
        let mut s = t.stabilizers();
        s.sort();
        assert_eq!(s, vec!["+XX".to_string(), "+ZZ".to_string()]);
    }

    #[test]
    fn deterministic_and_random() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut t = Tableau::new(3);
        t.x(1);
        assert_eq!(t.peek(0), Some(false));
        assert_eq!(t.peek(1), Some(true));
        t.h(2);
        assert_eq!(t.peek(2), None);
        let m = t.measure_qubit(2, &mut rng);
        assert_eq!(t.peek(2), Some(m));
    }

    #[test]
    fn ghz_outcomes_agree() {
        let mut rng = StdRng::seed_from_u64(2);
        for n in [3, 64, 65, 200] {
            let mut t = Tableau::new(n);
            t.h(0);
            for q in 1..n {
                t.cnot(q - 1, q);
            }
            let bits = t.measure_all(&mut rng);
            assert!(bits.iter().all(|&b| b == bits[0]), "n = {n}");
        }
    }

    #[test]
    fn signs_from_y_and_s() {
        // S H |0> = |+i>, stabilised by +Y; S S H|0> = |->, stabilised by -X
        let mut t = Tableau::new(1);
        t.h(0);
        t.s(0);
        assert_eq!(t.stabilizers(), vec!["+Y"]);
        t.s(0);
        assert_eq!(t.stabilizers(), vec!["-X"]);
        t.h(0);
        assert_eq!(t.peek(0), Some(true));
    }

    #[test]
    fn rejects_t() {
        let mut t = Tableau::new(2);
        assert!(matches!(
            t.apply_gate(&Gate::T(0)),
            Err(SimError::Unsupported { .. })
        ));
        assert!(t.apply_gate(&Gate::Phase(0, FRAC_PI_2)).is_ok());
    }

    #[test]
    fn sampling_keeps_state_and_matches_probabilities() {
        let mut rng = StdRng::seed_from_u64(3);
        for trial in 0..30 {
            let n = 2 + trial % 7;
            let c = crate::circuit::Circuit::random_clifford(n, 6, &mut rng);
            let mut t = Tableau::new(n);
            c.run(&mut t, &mut rng).unwrap();
            let before: Vec<f64> = (0..1 << n).map(|i| t.probability(i)).collect();
            let shots = t.sample(2000, &mut rng);
            let after: Vec<f64> = (0..1 << n).map(|i| t.probability(i)).collect();
            assert_eq!(before, after, "sampling must not change the state");
            let mut counts = vec![0usize; 1 << n];
            for s in shots {
                counts[s
                    .iter()
                    .enumerate()
                    .fold(0, |a, (q, &b)| a | (usize::from(b) << q))] += 1;
            }
            for i in 0..1 << n {
                let f = counts[i] as f64 / 2000.0;
                assert!(
                    (f - before[i]).abs() < 0.06,
                    "trial {trial}: {f} vs {}",
                    before[i]
                );
                assert!(before[i] > 0.0 || counts[i] == 0);
            }
            // gates still work after canonicalisation
            t.h(0);
            let mut sv = crate::StateVectorF64::new(n);
            sv.apply_circuit(&c).unwrap();
            sv.apply_gate(&Gate::H(0)).unwrap();
            for (i, p) in sv.probabilities().into_iter().enumerate() {
                assert!((t.probability(i) - p).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn measure_all_collapses() {
        let mut rng = StdRng::seed_from_u64(4);
        for n in [5, 70, 130] {
            let mut t = Tableau::new(n);
            for q in 0..n {
                t.h(q);
            }
            for q in 0..n - 1 {
                t.cnot(q, q + 1);
            }
            let bits = t.measure_all(&mut rng);
            for (q, &b) in bits.iter().enumerate() {
                assert_eq!(t.peek(q), Some(b));
            }
        }
    }

    #[test]
    fn memory_cap() {
        assert_eq!(tableau_bytes(64), 64 * 64 / 2);
        assert!(tableau_bytes(46_336) <= MAX_TABLEAU_BYTES);
        assert!(Tableau::try_new(46_400).is_err());
    }

    // ---- exactness checks for the inverse-sign / Stim-style measurement ----

    /// A Pauli string with an explicit phase `i^ph` (test-only helper).
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Pauli {
        ph: u8,
        x: Vec<bool>,
        z: Vec<bool>,
    }

    impl Pauli {
        fn identity(n: usize) -> Self {
            Pauli {
                ph: 0,
                x: vec![false; n],
                z: vec![false; n],
            }
        }
        /// `self <- self * o`
        fn mul(&mut self, o: &Pauli) {
            let mut e = self.ph as i32 + o.ph as i32;
            for q in 0..self.x.len() {
                e += g(self.x[q], self.z[q], o.x[q], o.z[q]);
                self.x[q] ^= o.x[q];
                self.z[q] ^= o.z[q];
            }
            self.ph = e.rem_euclid(4) as u8;
        }
        fn parse(s: &str) -> Self {
            let mut p = Pauli::identity(s.len() - 1);
            p.ph = if s.starts_with('-') { 2 } else { 0 };
            for (q, c) in s[1..].chars().enumerate() {
                p.x[q] = c == 'X' || c == 'Y';
                p.z[q] = c == 'Z' || c == 'Y';
            }
            p
        }
    }

    /// Exponent of `i` in the product of single-qubit Paulis `(x1,z1)(x2,z2)`.
    fn g(x1: bool, z1: bool, x2: bool, z2: bool) -> i32 {
        match (x1, z1) {
            (false, false) => 0,
            (true, true) => z2 as i32 - x2 as i32,
            (true, false) => z2 as i32 * (2 * x2 as i32 - 1),
            (false, true) => x2 as i32 * (1 - 2 * z2 as i32),
        }
    }

    #[test]
    fn phase_counter_matches_g() {
        // every pair of single-qubit Paulis, in every bit lane
        for a in 0..4u64 {
            for b in 0..4u64 {
                let (ax, az, bx, bz) = (a & 1, a >> 1, b & 1, b >> 1);
                let want = g(ax == 1, az == 1, bx == 1, bz == 1).rem_euclid(4) as u32;
                for lane in [0, 17, 63] {
                    let mut ph = Phase::default();
                    ph.mul(ax << lane, az << lane, bx << lane, bz << lane);
                    assert_eq!(ph.total(), want, "a={a} b={b}");
                }
            }
        }
        // random words: per-position sum
        let mut rng = StdRng::seed_from_u64(77);
        for _ in 0..200 {
            let v: [u64; 8] = std::array::from_fn(|_| rng.random());
            let mut ph = Phase::default();
            let mut want = 0i32;
            for k in 0..2 {
                let (ax, az, bx, bz) = (v[4 * k], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3]);
                ph.mul(ax, az, bx, bz);
                for i in 0..64 {
                    let bit = |w: u64| (w >> i) & 1 == 1;
                    want += g(bit(ax), bit(az), bit(bx), bit(bz));
                }
            }
            assert_eq!(ph.total() as i32, want.rem_euclid(4));
            let (mut a, mut b) = (Phase::default(), Phase::default());
            a.add(v[0]);
            a.add(v[1]);
            b.add(v[2]);
            let want = (v[0].count_ones() + v[1].count_ones() + v[2].count_ones()) % 4;
            assert_eq!(a.merge(b).total(), want);
        }
        assert_eq!(prefix_xor(0b1011), 0b1001 | (u64::MAX << 4));
    }

    /// Generator `i` of the tableau as an explicit Pauli (destabilizer if
    /// `stab` is false).
    fn generator(t: &mut Tableau, i: usize, stab: bool) -> Pauli {
        t.set_layout(Layout::QubitMajor);
        let (xm, zm, r) = if stab {
            (&t.xs, &t.zs, &t.rs)
        } else {
            (&t.xd, &t.zd, &t.rd)
        };
        let mut p = Pauli::identity(t.np);
        for q in 0..t.np {
            p.x[q] = xm.get(q, i);
            p.z[q] = zm.get(q, i);
        }
        p.ph = 2 * get_bit(r, i) as u8;
        p
    }

    /// Recomputes every inverse-row sign from scratch: `T(P_q)` is the
    /// Pauli over generator indices read off line `q`, and `C(T(P_q))` must
    /// be `±P_q` with the stored sign.
    fn check_inverse_signs(t: &mut Tableau) {
        let np = t.np;
        let gens: Vec<(Pauli, Pauli)> = (0..np)
            .map(|i| (generator(t, i, false), generator(t, i, true)))
            .collect();
        for q in 0..np {
            for (want_x, xm, zm, s) in [(true, &t.zs, &t.zd, &t.sx), (false, &t.xs, &t.xd, &t.sz)] {
                let mut prod = Pauli::identity(np);
                for (i, (d, st)) in gens.iter().enumerate() {
                    let (bx, bz) = (xm.get(q, i), zm.get(q, i));
                    match (bx, bz) {
                        (false, false) => {}
                        (true, false) => prod.mul(d),
                        (false, true) => prod.mul(st),
                        (true, true) => {
                            // C(Y_i) = i D_i S_i
                            let mut y = d.clone();
                            y.mul(st);
                            y.ph = (y.ph + 1) % 4;
                            prod.mul(&y);
                        }
                    }
                }
                let mut want = Pauli::identity(np);
                if want_x {
                    want.x[q] = true;
                } else {
                    want.z[q] = true;
                }
                want.ph = 2 * get_bit(s, q) as u8;
                assert_eq!(
                    prod,
                    want,
                    "inverse row {} of qubit {q}",
                    if want_x { "X" } else { "Z" }
                );
            }
        }
    }

    /// Reduced row-echelon form of a stabilizer group (unique per group).
    fn canonical(stabs: &[String]) -> Vec<Pauli> {
        let mut rows: Vec<Pauli> = stabs.iter().map(|s| Pauli::parse(s)).collect();
        let n = rows.first().map_or(0, |r| r.x.len());
        let mut k = 0;
        for col in 0..2 * n {
            let bit = |p: &Pauli| if col < n { p.x[col] } else { p.z[col - n] };
            let Some(r) = (k..rows.len()).find(|&r| bit(&rows[r])) else {
                continue;
            };
            rows.swap(r, k);
            let piv = rows[k].clone();
            for (i, row) in rows.iter_mut().enumerate() {
                if i != k && bit(row) {
                    row.mul(&piv);
                }
            }
            k += 1;
        }
        rows
    }

    #[test]
    fn matches_reference_on_random_circuits_with_measurements() {
        use super::reference::RefTableau;
        let mut rng = StdRng::seed_from_u64(21);
        for trial in 0..120 {
            let n = [1, 2, 3, 5, 8, 13, 63, 64, 65, 130][trial % 10];
            let depth = 1 + trial % 9;
            let mut t = Tableau::new(n);
            let mut r = RefTableau::new(n);
            let seed: u64 = rng.random();
            let (mut ra, mut rb) = (StdRng::seed_from_u64(seed), StdRng::seed_from_u64(seed));
            for _ in 0..depth {
                let c = crate::circuit::Circuit::random_clifford(n, 1, &mut rng);
                for g in c.gates() {
                    t.apply_gate(g).unwrap();
                    r.apply_gate(g).unwrap();
                }
                for _ in 0..1 + n / 4 {
                    let q = rng.random_range(0..n);
                    assert_eq!(t.peek(q), r.peek(q), "trial {trial}");
                    assert_eq!(
                        t.measure_qubit(q, &mut ra),
                        r.measure_qubit(q, &mut rb),
                        "trial {trial}"
                    );
                }
            }
            if n <= 65 {
                check_inverse_signs(&mut t);
            }
            assert_eq!(
                canonical(&t.stabilizers()),
                canonical(&r.stabilizers()),
                "trial {trial}"
            );
            // sampling still agrees afterwards (and keeps inverse signs valid)
            // (different generator bases give different but equally
            // distributed samples, so check support instead of equality)
            for bits in t.sample(3, &mut ra) {
                let mut c = r.clone();
                for (q, &b) in bits.iter().enumerate() {
                    let (got, random) = c.measure_with(q, Some(b), &mut rb);
                    assert!(random || got == b, "trial {trial}: sample outside support");
                }
            }
            assert_eq!(
                canonical(&t.stabilizers()),
                canonical(&r.stabilizers()),
                "trial {trial}"
            );
            if n <= 65 {
                check_inverse_signs(&mut t);
            }
            for q in 0..n {
                assert_eq!(t.peek(q), r.peek(q), "trial {trial}");
            }
        }
    }

    #[test]
    fn inverse_signs_after_each_gate_kind() {
        let mut rng = StdRng::seed_from_u64(31);
        for trial in 0..40 {
            let n = 1 + trial % 6;
            let mut t = Tableau::new(n);
            let c = crate::circuit::Circuit::random_clifford(n, 4, &mut rng);
            for g in c.gates() {
                t.apply_gate(g).unwrap();
                check_inverse_signs(&mut t);
            }
            for q in 0..n {
                t.measure_qubit(q, &mut rng);
                check_inverse_signs(&mut t);
            }
        }
    }

    #[test]
    fn reset_gives_zero() {
        let mut rng = StdRng::seed_from_u64(8);
        let mut t = Tableau::new(4);
        t.h(0);
        t.cnot(0, 1);
        let m = t.reset_qubit(0, &mut rng);
        assert_eq!(t.peek(0), Some(false));
        assert_eq!(t.peek(1), Some(m));
    }
}
