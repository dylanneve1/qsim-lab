//! The Pauli-rotation ("frame") engine behind [`crate::pauli_path`].
//!
//! The legacy engine pushes the observable backwards through every gate, so a
//! Clifford gate costs one pass over all terms. Here the circuit is first
//! rewritten, exactly, as
//!
//! ```text
//!   U = C · R_m ⋯ R_1,    R_j = exp(-i θ_j Q_j / 2)
//! ```
//!
//! with `C` Clifford and `Q_j` Pauli strings: moving a Z rotation on qubit
//! `a` past the Cliffords `C_k` that precede it turns its axis into
//! `C_k† Z_a C_k`. The axes are found with a Heisenberg tableau (images of
//! the `2n` generators under `P -> C_k† P C_k`), in `O(gates · n)` word
//! operations, independent of the number of Pauli terms. The observable
//! becomes `O' = C† O C`, and the only per-term work left is one pass per
//! rotation:
//!
//! ```text
//!   R† P R = P                          if [P, Q] = 0
//!   R† P R = cos θ P − i sin θ P Q      otherwise
//! ```
//!
//! Two exact reductions then shrink the sum:
//!
//! * **x-span pruning.** `<0|P|0>` is zero unless `P` has no X/Y, and the
//!   remaining rotations `R_k … R_1` can only change the x part of a term by
//!   adding x parts of their axes. A term whose x part is outside
//!   `W_k = span{x(Q_1), …, x(Q_k)}` can never contribute and is dropped.
//! * **z projection.** Conjugating everything by a CNOT network `V`
//!   (`V|0> = |0>`, so expectation values are unchanged) maps
//!   `X^x Z^z -> X^{Lx} Z^{L^{-T} z}`. Choosing `L` so that `W_k` becomes
//!   the span of the first `d_k = dim W_k` unit vectors, for every `k` at
//!   once (the `W_k` are nested), turns the pruning test into a mask test and
//!   shows that qubits `>= d_k` are inert from stage `k` on: they carry no X
//!   in any surviving term or remaining axis, so their Z bits never affect a
//!   commutation, a product phase, or the final value. Those bits are
//!   cleared, which merges terms that differ only there.
//!
//! Optionally, rotations about the same axis are merged when every rotation
//! between them commutes with that axis (an exact phase-polynomial style
//! T-count reduction).
//!
//! Terms live in `S` hash-map shards keyed by a hash of the Pauli string, so
//! a rotation step is two embarrassingly parallel phases: every shard scales
//! its anticommuting terms and emits the new `P·Q` terms into per-destination
//! buffers, then every shard merges the buffers addressed to it. A term `P`
//! receives at most one contribution per step (from `P·Q`), so merging needs
//! no ordering between shards.

use crate::circuit::{check_gate, Circuit, Op, SimError};
use crate::gate::{is_multiple_of_half_pi, Gate};
use crate::pauli_path::{conj_string, PathStats, PauliSum};
use rayon::prelude::*;
use std::collections::HashMap;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

/// Options for the frame engine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameOptions {
    /// Abort when more terms than this are alive.
    pub max_terms: usize,
    /// Drop terms whose x part can no longer be cancelled, and project away
    /// inert z bits (both exact).
    pub prune: bool,
    /// Merge rotations about the same axis separated only by commuting
    /// rotations (exact).
    pub merge_rotations: bool,
    /// Use rayon across hash shards.
    pub parallel: bool,
    /// Fuse a rotation with the stage change that follows it (one pass
    /// instead of two; only surviving branches are created).
    pub fuse: bool,
    /// Terms with `|c| <= drop_below` after a merge are removed (the legacy
    /// engine uses `1e-14`; `0.0` keeps every non-zero term).
    pub drop_below: f64,
}

impl Default for FrameOptions {
    fn default() -> Self {
        FrameOptions {
            max_terms: crate::pauli_path::DEFAULT_MAX_TERMS,
            prune: true,
            merge_rotations: true,
            parallel: true,
            fuse: true,
            drop_below: 1e-14,
        }
    }
}

// ---------------------------------------------------------------------------
// Pauli strings as fixed-size bit arrays.

/// A Hermitian Pauli string `i^{|x∧z|} X^x Z^z` on up to `64 W` qubits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Key<const W: usize> {
    x: [u64; W],
    z: [u64; W],
}

impl<const W: usize> Hash for Key<W> {
    fn hash<H: Hasher>(&self, h: &mut H) {
        for i in 0..W {
            h.write_u64(self.x[i]);
            h.write_u64(self.z[i]);
        }
    }
}

impl<const W: usize> Key<W> {
    const ID: Self = Key {
        x: [0; W],
        z: [0; W],
    };

    #[inline(always)]
    fn anticommutes(&self, o: &Self) -> bool {
        let mut acc = 0u64;
        for i in 0..W {
            acc ^= (self.x[i] & o.z[i]) ^ (self.z[i] & o.x[i]);
        }
        acc.count_ones() & 1 == 1
    }

    /// `self · o = i^e (self ⊕ o)` with `e` returned mod 4. `o_xz` is
    /// `|x(o) ∧ z(o)|`, precomputed by the caller.
    #[inline(always)]
    fn mul(&self, o: &Self, o_xz: u32) -> (Self, u32) {
        // H(x,z) = i^{x·z} X^x Z^z, and X^a Z^b X^c Z^d = (-1)^{b·c} X^{a+c} Z^{b+d}:
        // e = x1·z1 + x2·z2 + 2 z1·x2 − x3·z3 (mod 4).
        let mut r = Self::ID;
        let mut e = o_xz as i32;
        for i in 0..W {
            let (x1, z1) = (self.x[i], self.z[i]);
            let (x3, z3) = (x1 ^ o.x[i], z1 ^ o.z[i]);
            r.x[i] = x3;
            r.z[i] = z3;
            e += (x1 & z1).count_ones() as i32 + 2 * (z1 & o.x[i]).count_ones() as i32
                - (x3 & z3).count_ones() as i32;
        }
        (r, e.rem_euclid(4) as u32)
    }

    fn xz_count(&self) -> u32 {
        (0..W).map(|i| (self.x[i] & self.z[i]).count_ones()).sum()
    }

    fn x_is_zero(&self) -> bool {
        self.x.iter().all(|&w| w == 0)
    }

    /// True if no x bit at position `>= d`.
    #[inline(always)]
    fn x_below(&self, mask: &[u64; W]) -> bool {
        (0..W).all(|i| self.x[i] & !mask[i] == 0)
    }

    #[inline(always)]
    fn project_z(&mut self, mask: &[u64; W]) {
        for (z, m) in self.z.iter_mut().zip(mask) {
            *z &= m;
        }
    }

    fn from_words(k: &[u64], w: usize) -> Self {
        let mut r = Self::ID;
        r.x[..w].copy_from_slice(&k[..w]);
        r.z[..w].copy_from_slice(&k[w..2 * w]);
        r
    }
}

fn mask_below<const W: usize>(d: usize) -> [u64; W] {
    let mut m = [0u64; W];
    for (i, w) in m.iter_mut().enumerate() {
        let lo = i * 64;
        *w = if d >= lo + 64 {
            !0
        } else if d <= lo {
            0
        } else {
            (1u64 << (d - lo)) - 1
        };
    }
    m
}

/// A fast multiplicative hasher (keys are already well mixed by the
/// random Cliffords; the finaliser guards against structured inputs).
#[derive(Default, Clone, Copy)]
struct MixHasher(u64);

impl Hasher for MixHasher {
    #[inline(always)]
    fn write_u64(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(26) ^ v).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.write_u64(*b as u64);
        }
    }
    #[inline(always)]
    fn finish(&self) -> u64 {
        let mut h = self.0;
        h ^= h >> 32;
        h = h.wrapping_mul(0xD6E8_FEB8_6659_FD93);
        h ^= h >> 32;
        h
    }
}

type Map<const W: usize> = HashMap<Key<W>, f64, BuildHasherDefault<MixHasher>>;

#[inline(always)]
fn hash_key<const W: usize>(k: &Key<W>) -> u64 {
    let mut h = MixHasher::default();
    k.hash(&mut h);
    h.finish()
}

// ---------------------------------------------------------------------------
// Compilation: circuit -> (rotations, observable) in the rotation frame.

/// `exp(-i θ Q / 2)` with `Q` a Hermitian Pauli string (sign folded into θ).
#[derive(Clone, Debug)]
struct Rot {
    q: Vec<u64>, // 2w words: x then z
    theta: f64,
}

/// Images of the generators `X_q`, `Z_q` under `P -> C† P C`.
pub(crate) struct HeisenbergTableau {
    w: usize,
    /// `2n` images (X_0..X_{n-1}, Z_0..Z_{n-1}), each `2w` words.
    img: Vec<u64>,
    neg: Vec<bool>,
    n: usize,
}

/// Multiplies Hermitian string `a` (with phase exponent `e`, in units of i)
/// by Hermitian string `b` in place: `i^e a -> i^e a b`.
fn mul_words(a: &mut [u64], e: &mut i32, b: &[u64], w: usize) {
    let mut d = 0i32;
    for i in 0..w {
        let (x1, z1, x2, z2) = (a[i], a[w + i], b[i], b[w + i]);
        let (x3, z3) = (x1 ^ x2, z1 ^ z2);
        d += (x1 & z1).count_ones() as i32
            + (x2 & z2).count_ones() as i32
            + 2 * (z1 & x2).count_ones() as i32
            - (x3 & z3).count_ones() as i32;
        a[i] = x3;
        a[w + i] = z3;
    }
    *e += d;
}

impl HeisenbergTableau {
    /// Words per bit vector (`ceil(n / 64)`, at least 1).
    pub(crate) fn words(&self) -> usize {
        self.w
    }

    pub(crate) fn new(n: usize) -> Self {
        let w = n.div_ceil(64).max(1);
        let mut img = vec![0u64; 2 * n * 2 * w];
        for q in 0..n {
            img[q * 2 * w + q / 64] |= 1 << (q % 64); // X_q
            img[(n + q) * 2 * w + w + q / 64] |= 1 << (q % 64); // Z_q
        }
        HeisenbergTableau {
            w,
            img,
            neg: vec![false; 2 * n],
            n,
        }
    }

    /// Image of the Hermitian string `p` (2w words) with sign: returns
    /// `(negated, image)`.
    pub(crate) fn map(&self, p: &[u64]) -> (bool, Vec<u64>) {
        let w = self.w;
        let mut acc = vec![0u64; 2 * w];
        let mut e: i32 = (0..w).map(|i| (p[i] & p[w + i]).count_ones() as i32).sum();
        // Only the qubits in the support of `p` contribute (same order as a
        // full 0..n scan: X_q then Z_q, ascending q).
        for wi in 0..w {
            let mut sup = p[wi] | p[w + wi];
            while sup != 0 {
                let q = wi * 64 + sup.trailing_zeros() as usize;
                sup &= sup - 1;
                let b = 1u64 << (q % 64);
                for (bitset, gen) in [(p[wi] & b != 0, q), (p[w + wi] & b != 0, self.n + q)] {
                    if bitset {
                        if self.neg[gen] {
                            e += 2;
                        }
                        mul_words(
                            &mut acc,
                            &mut e,
                            &self.img[gen * 2 * w..(gen + 1) * 2 * w],
                            w,
                        );
                    }
                }
            }
        }
        let e = e.rem_euclid(4);
        debug_assert!(e % 2 == 0, "image of a Hermitian string must be Hermitian");
        (e == 2, acc)
    }

    /// `C -> C K` for a Clifford `K` given as gates in time order: every
    /// image becomes `K† img K` (conjugation by `G† · G` for the gates of
    /// `K` from the last to the first).
    pub(crate) fn post_conjugate(&mut self, gates: &[Gate]) {
        let w = self.w;
        for g in gates.iter().rev() {
            let ginv = g.inverse();
            let qs = g.qubits();
            for gen in 0..2 * self.n {
                let im = &mut self.img[gen * 2 * w..(gen + 1) * 2 * w];
                // skip images with no support on the gate's qubits (only
                // Paulis, whose conjugation is a sign, need every image)
                let touches = qs
                    .iter()
                    .any(|&q| (im[q / 64] | im[w + q / 64]) >> (q % 64) & 1 == 1);
                if touches && conj_string(im, w, &ginv) == 1 {
                    self.neg[gen] ^= true;
                }
            }
        }
    }

    /// `C -> G C`: new image of a generator `g` is `old(G† g G)`.
    pub(crate) fn apply_clifford(&mut self, g: &Gate) {
        let w = self.w;
        let ginv = g.inverse();
        let mut updates = Vec::new();
        for q in g.qubits() {
            for (is_z, gen) in [(false, q), (true, self.n + q)] {
                let mut s = vec![0u64; 2 * w];
                s[if is_z { w } else { 0 } + q / 64] |= 1 << (q % 64);
                // conj_string computes G' P G'† with G' = G†, i.e. G† P G.
                let neg = conj_string(&mut s, w, &ginv) == 1;
                let (n2, im) = self.map(&s);
                updates.push((gen, neg ^ n2, im));
            }
        }
        for (gen, neg, im) in updates {
            self.img[gen * 2 * w..(gen + 1) * 2 * w].copy_from_slice(&im);
            self.neg[gen] = neg;
        }
    }
}

/// The circuit as rotations plus the conjugated observable.
struct Compiled {
    n: usize,
    w: usize,
    rots: Vec<Rot>,
    /// `(key, coefficient)` of `C† O C`.
    obs: Vec<(Vec<u64>, f64)>,
    non_clifford: usize,
}

fn compile(circuit: &Circuit, observable: &PauliSum) -> Result<Compiled, SimError> {
    let n = circuit.num_qubits;
    let mut tab = HeisenbergTableau::new(n);
    let w = tab.w;
    let mut rots = Vec::new();
    let mut non_clifford = 0;
    for op in &circuit.ops {
        let g = match op {
            Op::Gate(g) => g,
            _ => panic!("pauli_path::expectation: circuit must be unitary"),
        };
        check_gate(g, n)?;
        for g in g.decompose_to_clifford_rz() {
            if g.is_clifford() {
                tab.apply_clifford(&g);
                continue;
            }
            let (a, theta) = match g {
                Gate::T(a) => (a, FRAC_PI_4),
                Gate::Tdg(a) => (a, -FRAC_PI_4),
                Gate::Phase(a, t) | Gate::Rz(a, t) => (a, t),
                _ => unreachable!("decomposition produced {g:?}"),
            };
            if is_multiple_of_half_pi(theta) {
                let k = (theta / FRAC_PI_2).round().rem_euclid(4.0) as usize;
                for _ in 0..k {
                    tab.apply_clifford(&Gate::S(a));
                }
                continue;
            }
            non_clifford += 1;
            let mut z = vec![0u64; 2 * w];
            z[w + a / 64] |= 1 << (a % 64);
            let (neg, q) = tab.map(&z);
            rots.push(Rot {
                q,
                theta: if neg { -theta } else { theta },
            });
        }
    }
    let ow = observable.w;
    let obs = observable
        .keys
        .chunks(2 * ow)
        .zip(&observable.coefs)
        .map(|(k, &c)| {
            let mut p = vec![0u64; 2 * w];
            p[..w].copy_from_slice(&k[..w]);
            p[w..].copy_from_slice(&k[ow..ow + w]);
            let (neg, im) = tab.map(&p);
            (im, if neg { -c } else { c })
        })
        .collect();
    Ok(Compiled {
        n,
        w,
        rots,
        obs,
        non_clifford,
    })
}

fn anticommutes_words(a: &[u64], b: &[u64], w: usize) -> bool {
    let mut acc = 0u64;
    for i in 0..w {
        acc ^= (a[i] & b[w + i]) ^ (a[w + i] & b[i]);
    }
    acc.count_ones() & 1 == 1
}

/// Merges `exp(-iθ₂Q/2) … exp(-iθ₁Q/2)` when every rotation in between
/// commutes with `Q`. Rotations whose total angle is a multiple of 2π are
/// removed (they are a global phase).
fn merge_rotations(rots: Vec<Rot>, w: usize) -> Vec<Rot> {
    const SCAN: usize = 512;
    let mut out: Vec<Rot> = Vec::with_capacity(rots.len());
    'next: for r in rots {
        for j in (out.len().saturating_sub(SCAN)..out.len()).rev() {
            if out[j].q == r.q {
                out[j].theta += r.theta;
                continue 'next;
            }
            if anticommutes_words(&out[j].q, &r.q, w) {
                break;
            }
        }
        out.push(r);
    }
    out.retain(|r| {
        let k = r.theta / FRAC_PI_2;
        !((k - k.round()).abs() < 1e-12 && (k.round() as i64).rem_euclid(4) == 0)
    });
    out
}

// ---------------------------------------------------------------------------
// The CNOT frame.

/// A linear map `L` on GF(2)^n given by a basis `B` (columns `b_i`):
/// `L x` = coordinates of `x` in that basis, `L^{-T} z = B^T z`.
struct Frame<const W: usize> {
    basis: Vec<[u64; W]>,
    /// Echelon rows: (pivot, vector, combination of basis indices).
    ech: Vec<(usize, [u64; W], [u64; W])>,
}

fn bit<const W: usize>(v: &[u64; W], i: usize) -> bool {
    v[i / 64] >> (i % 64) & 1 == 1
}

impl<const W: usize> Frame<W> {
    fn new() -> Self {
        Frame {
            basis: Vec::new(),
            ech: Vec::new(),
        }
    }

    /// Reduces `v`; returns the residue and the combination used.
    fn reduce(&self, mut v: [u64; W]) -> ([u64; W], [u64; W]) {
        let mut comb = [0u64; W];
        for (p, e, c) in &self.ech {
            if bit(&v, *p) {
                for i in 0..W {
                    v[i] ^= e[i];
                    comb[i] ^= c[i];
                }
            }
        }
        (v, comb)
    }

    /// Adds `v` to the basis if independent. Returns true if added.
    fn push(&mut self, v: [u64; W]) -> bool {
        let (r, mut comb) = self.reduce(v);
        let Some(p) = (0..W * 64).find(|&i| bit(&r, i)) else {
            return false;
        };
        let idx = self.basis.len();
        comb[idx / 64] ^= 1 << (idx % 64);
        self.basis.push(v);
        // Keep the echelon fully reduced at the new pivot.
        for (_, e, c) in self.ech.iter_mut() {
            if bit(e, p) {
                for i in 0..W {
                    e[i] ^= r[i];
                    c[i] ^= comb[i];
                }
            }
        }
        self.ech.push((p, r, comb));
        true
    }

    fn complete(&mut self, n: usize) {
        for q in 0..n {
            let mut e = [0u64; W];
            e[q / 64] |= 1 << (q % 64);
            self.push(e);
        }
    }

    /// Image of a Hermitian string under `V · V†`: `(negated, key)`.
    fn apply(&self, k: &Key<W>) -> (bool, Key<W>) {
        let (r, x2) = self.reduce(k.x);
        debug_assert!(r.iter().all(|&w| w == 0));
        let mut z2 = [0u64; W];
        for (i, b) in self.basis.iter().enumerate() {
            let par = (0..W).map(|j| (b[j] & k.z[j]).count_ones()).sum::<u32>() & 1;
            z2[i / 64] |= (par as u64) << (i % 64);
        }
        let out = Key { x: x2, z: z2 };
        // X^x Z^z maps without a phase; the Hermitian normalisations differ
        // by i^{|x∧z| − |x'∧z'|}, which is ±1.
        let diff = k.xz_count() as i32 - out.xz_count() as i32;
        debug_assert!(diff % 2 == 0);
        (diff.rem_euclid(4) == 2, out)
    }
}

// ---------------------------------------------------------------------------
// The sharded term store.

struct Store<const W: usize> {
    shards: Vec<Map<W>>,
    bits: u32,
    parallel: bool,
    /// Terms discarded by x-span pruning so far.
    pruned: u64,
    bufs: Vec<Vec<Vec<(Key<W>, f64)>>>,
}

impl<const W: usize> Store<W> {
    fn new(parallel: bool) -> Self {
        let bits = if parallel { 6 } else { 0 };
        let s = 1usize << bits;
        Store {
            shards: (0..s).map(|_| Map::default()).collect(),
            bits,
            parallel,
            pruned: 0,
            bufs: (0..s)
                .map(|_| (0..s).map(|_| Vec::new()).collect())
                .collect(),
        }
    }

    #[inline(always)]
    fn shard_of(&self, k: &Key<W>) -> usize {
        if self.bits == 0 {
            0
        } else {
            (hash_key(k) >> 40) as usize & ((1 << self.bits) - 1)
        }
    }

    fn len(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }

    fn add(&mut self, k: Key<W>, c: f64) {
        let s = self.shard_of(&k);
        *self.shards[s].entry(k).or_insert(0.0) += c;
    }

    fn for_each_shard<F>(&mut self, f: F)
    where
        F: Fn(usize, &mut Map<W>, &mut Vec<Vec<(Key<W>, f64)>>) + Sync + Send,
    {
        if self.parallel {
            self.shards
                .par_iter_mut()
                .zip(self.bufs.par_iter_mut())
                .enumerate()
                .for_each(|(i, (sh, b))| f(i, sh, b));
        } else {
            for (i, (sh, b)) in self.shards.iter_mut().zip(self.bufs.iter_mut()).enumerate() {
                f(i, sh, b);
            }
        }
    }

    /// Merges `bufs[src][dst]` into shard `dst`, then removes terms with
    /// `|c| <= drop` that were touched (or every term if `sweep`).
    fn gather(&mut self, drop: f64, sweep: bool) {
        let bufs = &self.bufs;
        let body = |d: usize, sh: &mut Map<W>| {
            for src in bufs.iter() {
                for &(k, v) in &src[d] {
                    match sh.entry(k) {
                        std::collections::hash_map::Entry::Occupied(mut e) => {
                            let c = e.get_mut();
                            *c += v;
                            if !sweep && c.abs() <= drop {
                                e.remove();
                            }
                        }
                        std::collections::hash_map::Entry::Vacant(e) => {
                            if sweep || v.abs() > drop {
                                e.insert(v);
                            }
                        }
                    }
                }
            }
            if sweep {
                sh.retain(|_, c| c.abs() > drop);
            }
        };
        if self.parallel {
            self.shards
                .par_iter_mut()
                .enumerate()
                .for_each(|(d, sh)| body(d, sh));
        } else {
            for (d, sh) in self.shards.iter_mut().enumerate() {
                body(d, sh);
            }
        }
        for b in self.bufs.iter_mut() {
            for v in b.iter_mut() {
                v.clear();
            }
        }
    }

    /// One rotation `exp(-iθQ/2)`: `P -> cos θ P − i sin θ P Q` for every
    /// `P` anticommuting with `Q`.
    fn rotate(&mut self, q: &Key<W>, cs: f64, sn: f64, drop: f64) {
        let q_xz = q.xz_count();
        let bits = self.bits;
        let q = *q;
        self.for_each_shard(|_, sh, out| {
            sh.retain(|k, c| {
                if !k.anticommutes(&q) {
                    return true;
                }
                let (k2, e) = k.mul(&q, q_xz);
                debug_assert!(e & 1 == 1);
                // −i · i^e = i^{e−1} = ±1 for odd e.
                let v = if e == 1 { sn * *c } else { -sn * *c };
                let d = if bits == 0 {
                    0
                } else {
                    (hash_key(&k2) >> 40) as usize & ((1 << bits) - 1)
                };
                out[d].push((k2, v));
                *c *= cs;
                c.abs() > drop
            });
        });
        self.gather(drop, false);
    }

    /// Stage change: drop terms with x outside the first `d` bits and clear
    /// z bits at `>= d`, merging terms that become equal.
    fn project(&mut self, d: usize, drop: f64) {
        let mask = mask_below::<W>(d);
        let bits = self.bits;
        let pruned = AtomicU64::new(0);
        let pruned = &pruned;
        self.for_each_shard(|_, sh, out| {
            let mut np = 0;
            for (mut k, c) in sh.drain() {
                if !k.x_below(&mask) {
                    np += 1;
                    continue;
                }
                k.project_z(&mask);
                let dd = if bits == 0 {
                    0
                } else {
                    (hash_key(&k) >> 40) as usize & ((1 << bits) - 1)
                };
                out[dd].push((k, c));
            }
            pruned.fetch_add(np, Ordering::Relaxed);
        });
        self.pruned += pruned.load(Ordering::Relaxed);
        self.gather(drop, true);
    }

    /// A rotation followed by a stage change to `d` active qubits, in one
    /// pass: only the branches that survive the projection are emitted.
    fn rotate_project(&mut self, q: &Key<W>, cs: f64, sn: f64, d: usize, drop: f64) {
        let q_xz = q.xz_count();
        let mask = mask_below::<W>(d);
        let bits = self.bits;
        let q = *q;
        let pruned = AtomicU64::new(0);
        let pruned = &pruned;
        self.for_each_shard(|_, sh, out| {
            let mut np = 0;
            let mut emit = |mut k: Key<W>, c: f64| {
                if c == 0.0 {
                    return;
                }
                if !k.x_below(&mask) {
                    np += 1;
                    return;
                }
                k.project_z(&mask);
                let dd = if bits == 0 {
                    0
                } else {
                    (hash_key(&k) >> 40) as usize & ((1 << bits) - 1)
                };
                out[dd].push((k, c));
            };
            for (k, c) in sh.drain() {
                if k.anticommutes(&q) {
                    let (k2, e) = k.mul(&q, q_xz);
                    emit(k2, if e == 1 { sn * c } else { -sn * c });
                    emit(k, cs * c);
                } else {
                    emit(k, c);
                }
            }
            pruned.fetch_add(np, Ordering::Relaxed);
        });
        self.pruned += pruned.load(Ordering::Relaxed);
        self.gather(drop, true);
    }

    fn zero_state_value(&self) -> f64 {
        self.shards
            .iter()
            .flat_map(|s| s.iter())
            .filter(|(k, _)| k.x_is_zero())
            .map(|(_, c)| *c)
            .sum()
    }
}

/// Exact `cos θ`, `sin θ` for multiples of π/2 (merged rotations can be
/// Clifford), floating point otherwise.
fn cos_sin(theta: f64) -> (f64, f64) {
    if is_multiple_of_half_pi(theta) {
        match (theta / FRAC_PI_2).round().rem_euclid(4.0) as u32 {
            0 => (1.0, 0.0),
            1 => (0.0, 1.0),
            2 => (-1.0, 0.0),
            _ => (0.0, -1.0),
        }
    } else {
        let (s, c) = theta.sin_cos();
        (c, s)
    }
}

/// Where the Heisenberg sweep stopped when a [`SwitchPolicy`] asked it to
/// hand over to a Schrödinger simulation of the remaining rotations.
///
/// Everything is in the CNOT frame of the pruning proof: rotation `j`
/// (1-based, `j <= stage`) is `exp(-i θ_j Q_j / 2)` with `Q_j` the
/// Hermitian string `i^{|x∧z|} X^x Z^z`, `x` supported on the first `d[j]`
/// qubits and `z` already projected there. The operator `terms` lives on the
/// first `d[stage]` qubits, and the exact value is
/// `<φ| terms |φ>` with `φ = R_stage ⋯ R_1 |0^{d[stage]}>`.
#[derive(Clone, Debug)]
pub(crate) struct SwitchPoint {
    pub stage: usize,
    /// `d[j] = dim span{x(Q_1..Q_j)}` for `j = 0..=stage`.
    pub d: Vec<usize>,
    /// `(x, z, θ)` for rotations `1..=stage` (index `j - 1`).
    pub rots: Vec<(u64, u64, f64)>,
    /// `(x, z, coefficient)` of the propagated operator.
    pub terms: Vec<(u64, u64, f64)>,
    pub stats: PathStats,
}

/// Result of a staged Heisenberg sweep.
pub(crate) enum Staged {
    /// The sweep ran to the end: exact value.
    Done(f64, PathStats),
    /// The policy asked for a switch.
    Switched(Box<SwitchPoint>),
}

/// Called before each rotation `stage` (rotations `1..=stage` remain) with
/// `(stage, live terms, d, term visits so far)`; returns true to stop the
/// Heisenberg sweep and hand over. Only called when pruning is on and
/// `d[stage] <= 62`.
pub(crate) type SwitchPolicy<'a> = &'a mut dyn FnMut(usize, usize, &[usize], u64) -> bool;

fn run<const W: usize>(comp: Compiled, opt: &FrameOptions) -> Result<(f64, PathStats), SimError> {
    match run_staged::<W>(comp, opt, None)? {
        Staged::Done(v, s) => Ok((v, s)),
        Staged::Switched(_) => unreachable!("no policy"),
    }
}

fn low_word<const W: usize>(v: &[u64; W]) -> u64 {
    debug_assert!(v[1..].iter().all(|&w| w == 0));
    v[0]
}

fn switch_point<const W: usize>(
    stage: usize,
    d: &[usize],
    axes: &[Key<W>],
    thetas: &[f64],
    store: &Store<W>,
    stats: PathStats,
) -> Box<SwitchPoint> {
    let rots = (0..stage)
        .map(|j| {
            let mut q = axes[j];
            q.project_z(&mask_below::<W>(d[j + 1]));
            (low_word(&q.x), low_word(&q.z), thetas[j])
        })
        .collect();
    let terms = store
        .shards
        .iter()
        .flat_map(|s| s.iter())
        .map(|(k, &c)| (low_word(&k.x), low_word(&k.z), c))
        .collect();
    Box::new(SwitchPoint {
        stage,
        d: d[..=stage].to_vec(),
        rots,
        terms,
        stats,
    })
}

fn run_staged<const W: usize>(
    comp: Compiled,
    opt: &FrameOptions,
    mut policy: Option<SwitchPolicy<'_>>,
) -> Result<Staged, SimError> {
    let Compiled {
        n, w, rots, obs, ..
    } = comp;
    let m = rots.len();
    let mut axes: Vec<Key<W>> = rots.iter().map(|r| Key::from_words(&r.q, w)).collect();
    let mut thetas: Vec<f64> = rots.iter().map(|r| r.theta).collect();
    let mut obs: Vec<(Key<W>, f64)> = obs
        .iter()
        .map(|(k, c)| (Key::from_words(k, w), *c))
        .collect();
    // d[j] = dim span{x(Q_1..Q_j)} (stage j: rotations 1..=j remain).
    let mut d = vec![n; m + 1];
    if opt.prune {
        let mut fr = Frame::<W>::new();
        d[0] = 0;
        for j in 0..m {
            fr.push(axes[j].x);
            d[j + 1] = fr.basis.len();
        }
        fr.complete(n);
        for (a, t) in axes.iter_mut().zip(thetas.iter_mut()) {
            let (neg, k) = fr.apply(a);
            *a = k;
            if neg {
                *t = -*t;
            }
        }
        for (k, c) in obs.iter_mut() {
            let (neg, k2) = fr.apply(k);
            *k = k2;
            if neg {
                *c = -*c;
            }
        }
    }
    let mut store = Store::<W>::new(opt.parallel);
    let mask = mask_below::<W>(d[m]);
    let mut pruned0 = 0;
    for (mut k, c) in obs {
        if opt.prune {
            if !k.x_below(&mask) {
                pruned0 += 1;
                continue;
            }
            k.project_z(&mask);
        }
        store.add(k, c);
    }
    store
        .shards
        .iter_mut()
        .for_each(|s| s.retain(|_, c| *c != 0.0));
    let mut stats = PathStats {
        peak_terms: store.len(),
        rotations: m,
        ..Default::default()
    };
    for j in (0..m).rev() {
        if let Some(p) = policy.as_mut() {
            if opt.prune && d[j + 1] <= 62 && p(j + 1, store.len(), &d, stats.term_visits) {
                stats.final_terms = store.len();
                stats.pruned_terms = pruned0 + store.pruned;
                return Ok(Staged::Switched(switch_point(
                    j + 1,
                    &d,
                    &axes,
                    &thetas,
                    &store,
                    stats,
                )));
            }
        }
        stats.term_visits += store.len() as u64;
        // rotation j+1 in 1-based stage numbering; stage j+1 -> j.
        let mut q = axes[j];
        if opt.prune {
            q.project_z(&mask_below::<W>(d[j + 1]));
        }
        let (cs, sn) = cos_sin(thetas[j]);
        let stage_change = opt.prune && d[j] < d[j + 1];
        if stage_change && q != Key::ID && opt.fuse {
            store.rotate_project(&q, cs, sn, d[j], opt.drop_below);
        } else {
            if q != Key::ID {
                store.rotate(&q, cs, sn, opt.drop_below);
            }
            if stage_change {
                store.project(d[j], opt.drop_below);
            }
        }
        let len = store.len();
        stats.peak_terms = stats.peak_terms.max(len);
        if len > opt.max_terms {
            // Last chance: a policy may still take over on the (smaller)
            // register of the next stage instead of failing.
            if let Some(p) = policy.as_mut() {
                if opt.prune && j > 0 && d[j] <= 62 && p(j, usize::MAX, &d, stats.term_visits) {
                    stats.final_terms = len;
                    stats.pruned_terms = pruned0 + store.pruned;
                    return Ok(Staged::Switched(switch_point(
                        j, &d, &axes, &thetas, &store, stats,
                    )));
                }
            }
            return Err(SimError::TooManyTerms {
                terms: len,
                limit: opt.max_terms,
            });
        }
        if len == 0 {
            break;
        }
    }
    stats.final_terms = store.len();
    stats.pruned_terms = pruned0 + store.pruned;
    Ok(Staged::Done(store.zero_state_value(), stats))
}

/// [`expectation`] with a [`SwitchPolicy`]: the Heisenberg sweep may stop
/// early and return a [`SwitchPoint`] (see [`crate::adaptive`]).
pub(crate) fn expectation_staged(
    circuit: &Circuit,
    observable: &PauliSum,
    opt: &FrameOptions,
    policy: SwitchPolicy<'_>,
) -> Option<Result<Staged, SimError>> {
    let wn = circuit.num_qubits.div_ceil(64).max(1);
    if wn > 8 {
        return None;
    }
    let mut comp = match compile(circuit, observable) {
        Ok(c) => c,
        Err(e) => return Some(Err(e)),
    };
    let non_clifford = comp.non_clifford;
    if opt.merge_rotations {
        comp.rots = merge_rotations(std::mem::take(&mut comp.rots), comp.w);
    }
    let p = Some(policy);
    let r = match wn {
        1 => run_staged::<1>(comp, opt, p),
        2 => run_staged::<2>(comp, opt, p),
        3 | 4 => run_staged::<4>(comp, opt, p),
        _ => run_staged::<8>(comp, opt, p),
    };
    Some(r.map(|st| match st {
        Staged::Done(v, mut s) => {
            s.non_clifford_gates = non_clifford;
            Staged::Done(v, s)
        }
        Staged::Switched(mut sp) => {
            sp.stats.non_clifford_gates = non_clifford;
            Staged::Switched(sp)
        }
    }))
}

/// Exact `<0| U† O U |0>` with the frame engine. Returns `None` when the
/// register is too wide for the fixed-size keys (more than 512 qubits).
pub(crate) fn expectation(
    circuit: &Circuit,
    observable: &PauliSum,
    opt: &FrameOptions,
) -> Option<Result<(f64, PathStats), SimError>> {
    let wn = circuit.num_qubits.div_ceil(64).max(1);
    if wn > 8 {
        return None;
    }
    let comp = match compile(circuit, observable) {
        Ok(c) => c,
        Err(e) => return Some(Err(e)),
    };
    let non_clifford = comp.non_clifford;
    let mut comp = comp;
    if opt.merge_rotations {
        comp.rots = merge_rotations(std::mem::take(&mut comp.rots), comp.w);
    }
    let r = match wn {
        1 => run::<1>(comp, opt),
        2 => run::<2>(comp, opt),
        3 | 4 => run::<4>(comp, opt),
        _ => run::<8>(comp, opt),
    };
    Some(r.map(|(v, mut s)| {
        s.non_clifford_gates = non_clifford;
        (v, s)
    }))
}
