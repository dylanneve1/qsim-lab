//! Adaptive representation switching for Clifford+T circuits (exact).
//!
//! A circuit `U` of Clifford gates and `m` non-Clifford Z rotations is
//! rewritten, with a Heisenberg tableau, as `U = C · R_m ⋯ R_1` with `C`
//! Clifford and `R_j = exp(-i θ_j Q_j / 2)` (see [`crate::pauli_frame`]).
//! A CNOT network `V` with linear map `L` (`V|y> = |Ly>`) is chosen so that
//! `W_j = span{x(Q_1..Q_j)}` becomes the span of the first `d_j` unit
//! vectors for every `j` at once (the pruning proof in `research/pauli.md`).
//! In that frame every rotation acts on the first `d_j` qubits only, the
//! qubits `>= d_j` are still `|0>` when `R_j` acts (their Z bits act as
//! `+1`), and therefore, exactly,
//!
//! ```text
//!   U|0^n> = C V† (|φ> ⊗ |0^{n-d}>),   |φ> = R'_m ⋯ R'_1 |0^d>,  d = d_m,
//! ```
//!
//! where `R'_j` is `R_j` conjugated by `V` and restricted to `d_j` qubits.
//! `|φ>` is a dense vector of `2^d` amplitudes that grows one qubit at a
//! time (`2^{d_j}` amplitudes while rotation `j` is applied). This is the
//! Jozsa–Van den Nest / Yoganathan–Jozsa–Strelchuk compression, built from
//! the same frame the Pauli-path engine prunes with.
//!
//! Two simulators are built on it:
//!
//! * [`CompressedState`] + [`Sampler`]: Schrödinger picture on the active
//!   register, then exact sampling of all `n` qubits through the Clifford
//!   `D = C V†` (one Gaussian elimination and one diagonalising Clifford on
//!   the dense register; each shot is then a table lookup and `O(n²/64)`
//!   word operations).
//! * [`expectation`]: `<0|U† O U|0>` that starts as a Heisenberg (Pauli-path)
//!   sweep from the end of the circuit and, when a cost model fed by the
//!   live term count says so, hands the propagated operator to a dense
//!   Schrödinger simulation of the remaining rotations on the (smaller)
//!   register of that stage: meet in the middle,
//!   `value = <φ_k| A_k |φ_k>`.
//!
//! Every step is exact (no truncation); the only approximation is IEEE
//! rounding. Global phases (e.g. `T = e^{iπ/8} Rz(π/4)`) are not tracked,
//! as in the rest of the crate's decompositions.

use crate::circuit::{check_gate, Circuit, Op, SimError};
use crate::gate::{is_multiple_of_half_pi, Gate};
use crate::pauli_frame::{expectation_staged, HeisenbergTableau, Staged};
use crate::pauli_path::{FrameOptions, PathStats, PauliSum};
use crate::statevector::{StateVectorF64, MAX_STATE_BYTES};
use num_complex::Complex64;
use rand::Rng;
use rayon::prelude::*;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};
use std::time::Instant;

type C64 = Complex64;

const PAR_MIN: usize = 1 << 14;
const SUB: usize = 1 << 12;

// ---------------------------------------------------------------------------
// Dense kernels on the active register.

#[inline(always)]
fn parity(v: u64) -> bool {
    v.count_ones() & 1 == 1
}

/// Calls `f(y, &mut a[y], &mut a[y ^ x])` once for every unordered pair
/// `{y, y ^ x}` (with `y` the member whose top bit of `x` is clear).
fn for_pairs<F>(a: &mut [C64], x: u64, f: F)
where
    F: Fn(usize, &mut C64, &mut C64) + Sync + Send,
{
    debug_assert!(x != 0 && (x as usize) < a.len());
    let h = 63 - x.leading_zeros() as usize;
    let half = 1usize << h;
    let xl = (x as usize) & (half - 1);
    if a.len() < PAR_MIN {
        for (blk, chunk) in a.chunks_mut(2 * half).enumerate() {
            let (lo, hi) = chunk.split_at_mut(half);
            let base = blk * 2 * half;
            for o in 0..half {
                f(base + o, &mut lo[o], &mut hi[o ^ xl]);
            }
        }
        return;
    }
    // Split every (lo, hi) half into aligned sub-blocks of size `b`: XOR with
    // `xl` maps an aligned sub-block of lo onto an aligned sub-block of hi.
    let b = half.min(SUB);
    let hb = xl / b; // which hi sub-block a lo sub-block pairs with (XOR)
    let xs = xl & (b - 1);
    let mut work: Vec<(usize, &mut [C64], &mut [C64])> = Vec::with_capacity(a.len() / (2 * b));
    for (blk, chunk) in a.chunks_mut(2 * half).enumerate() {
        let (lo, hi) = chunk.split_at_mut(half);
        let mut his: Vec<Option<&mut [C64]>> = hi.chunks_mut(b).map(Some).collect();
        for (i, l) in lo.chunks_mut(b).enumerate() {
            let hchunk = his[i ^ hb].take().expect("bijection");
            work.push((blk * 2 * half + i * b, l, hchunk));
        }
    }
    work.into_par_iter().for_each(|(base, lo, hi)| {
        for o in 0..lo.len() {
            f(base + o, &mut lo[o], &mut hi[o ^ xs]);
        }
    });
}

fn for_each_amp<F>(a: &mut [C64], f: F)
where
    F: Fn(usize, &mut C64) + Sync + Send,
{
    if a.len() < PAR_MIN {
        a.iter_mut().enumerate().for_each(|(y, v)| f(y, v));
    } else {
        a.par_iter_mut()
            .with_min_len(SUB)
            .enumerate()
            .for_each(|(y, v)| f(y, v));
    }
}

/// `i^k` for `k mod 4`.
fn i_pow(k: u32) -> C64 {
    match k % 4 {
        0 => C64::new(1.0, 0.0),
        1 => C64::new(0.0, 1.0),
        2 => C64::new(-1.0, 0.0),
        _ => C64::new(0.0, -1.0),
    }
}

/// `(cos θ/2, sin θ/2)`, exact when `θ/2` is a multiple of π/2.
fn half_cos_sin(theta: f64) -> (f64, f64) {
    let h = theta / 2.0;
    if is_multiple_of_half_pi(h) {
        match (h / FRAC_PI_2).round().rem_euclid(4.0) as u32 {
            0 => (1.0, 0.0),
            1 => (0.0, 1.0),
            2 => (-1.0, 0.0),
            _ => (0.0, -1.0),
        }
    } else {
        let (s, c) = h.sin_cos();
        (c, s)
    }
}

/// `a <- exp(-i θ Q / 2) a` with `Q = i^{|x∧z|} X^x Z^z` (Hermitian).
pub(crate) fn rotate_dense(a: &mut [C64], x: u64, z: u64, theta: f64) {
    let (c, s) = half_cos_sin(theta);
    let r = (x & z).count_ones();
    // (Qa)(y) = i^r (-1)^{z·(y⊕x)} a(y⊕x);  R = c − i s Q.
    let k = i_pow(r + 3) * s;
    if x == 0 {
        let (dp, dm) = (c + k, c - k);
        for_each_amp(a, |y, v| *v *= if parity(z & y as u64) { dm } else { dp });
    } else {
        for_pairs(a, x, |y, lo, hi| {
            let y = y as u64;
            let s_lo = if parity(z & y) { -k } else { k }; // sign at y
            let s_hi = if parity(z & (y ^ x)) { -k } else { k }; // sign at y^x
            let (l, h) = (*lo, *hi);
            *lo = l * c + h * s_hi;
            *hi = h * c + l * s_lo;
        });
    }
}

fn hadamard_dense(a: &mut [C64], q: usize) {
    let r = std::f64::consts::FRAC_1_SQRT_2;
    for_pairs(a, 1 << q, |_, lo, hi| {
        let (l, h) = (*lo, *hi);
        *lo = (l + h) * r;
        *hi = (l - h) * r;
    });
}

/// Unnormalised Walsh–Hadamard transform: `out[z] = Σ_w (-1)^{z·w} a[w]`.
fn fwht(a: &mut [C64]) {
    let d = a.len().trailing_zeros() as usize;
    for q in 0..d {
        for_pairs(a, 1 << q, |_, lo, hi| {
            let (l, h) = (*lo, *hi);
            *lo = l + h;
            *hi = l - h;
        });
    }
}

/// `<φ| Σ c_t P_t |φ>` for Hermitian strings `P_t = i^{|x∧z|} X^x Z^z` on
/// the register of `a`. Terms are grouped by `x`; a group with many `z`
/// values is done with one Walsh–Hadamard transform. Returns the value and
/// the number of dense element operations used.
fn eval_terms(a: &[C64], terms: &mut [(u64, u64, f64)]) -> (f64, u64) {
    let len = a.len();
    let d = len.trailing_zeros() as u64;
    terms.sort_unstable_by_key(|t| (t.0, t.1));
    let mut total = 0.0;
    let mut ops = 0u64;
    let mut i = 0;
    while i < terms.len() {
        let x = terms[i].0;
        let mut j = i;
        while j < terms.len() && terms[j].0 == x {
            j += 1;
        }
        let group = &terms[i..j];
        let fold = |z: u64| -> C64 {
            let body = |w: usize| {
                let v = a[w ^ x as usize].conj() * a[w];
                if parity(z & w as u64) {
                    -v
                } else {
                    v
                }
            };
            if len < PAR_MIN {
                (0..len).map(body).sum()
            } else {
                (0..len).into_par_iter().with_min_len(SUB).map(body).sum()
            }
        };
        if (group.len() as u64) <= d + 1 {
            for &(_, z, c) in group {
                let s = fold(z);
                total += c * (i_pow((x & z).count_ones()) * s).re;
                ops += len as u64;
            }
        } else {
            let mut u: Vec<C64> = if len < PAR_MIN {
                (0..len).map(|w| a[w ^ x as usize].conj() * a[w]).collect()
            } else {
                (0..len)
                    .into_par_iter()
                    .with_min_len(SUB)
                    .map(|w| a[w ^ x as usize].conj() * a[w])
                    .collect()
            };
            fwht(&mut u);
            for &(_, z, c) in group {
                total += c * (i_pow((x & z).count_ones()) * u[z as usize]).re;
            }
            ops += len as u64 * (d + 1);
        }
        i = j;
    }
    (total, ops)
}

fn dense_bytes(d: usize) -> u128 {
    (1u128 << d.min(120)) * std::mem::size_of::<C64>() as u128
}

/// Applies the rotations `(x, z, θ, d_j)` to `|0>`, growing the register to
/// `2^{d_j}` amplitudes before rotation `j`. Returns the vector and the
/// number of element operations.
fn evolve(
    rots: impl Iterator<Item = (u64, u64, f64, usize)>,
    d_final: usize,
    max_qubits: usize,
) -> Result<(Vec<C64>, u64), SimError> {
    let limit = dense_bytes(max_qubits).min(MAX_STATE_BYTES);
    if d_final > 62 || dense_bytes(d_final) > limit {
        return Err(SimError::TooLarge {
            what: "compressed state vector",
            bytes: dense_bytes(d_final),
            limit,
        });
    }
    let mut a = Vec::with_capacity(1 << d_final);
    a.push(C64::new(1.0, 0.0));
    let mut ops = 0u64;
    for (x, z, theta, dj) in rots {
        if (1usize << dj) > a.len() {
            a.resize(1 << dj, C64::new(0.0, 0.0));
        }
        let m = (1u64 << dj) - 1;
        debug_assert!(x & !m == 0);
        rotate_dense(&mut a, x, z & m, theta);
        ops += a.len() as u64;
    }
    if a.len() < (1 << d_final) {
        a.resize(1 << d_final, C64::new(0.0, 0.0));
    }
    Ok((a, ops))
}

// ---------------------------------------------------------------------------
// Pauli rows with explicit phase: i^r X^x Z^z (X before Z on each qubit).

fn get(v: &[u64], i: usize) -> bool {
    v[i / 64] >> (i % 64) & 1 == 1
}

fn flip(v: &mut [u64], i: usize) {
    v[i / 64] ^= 1 << (i % 64);
}

fn xor_into(a: &mut [u64], b: &[u64]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x ^= y;
    }
}

fn and_count(a: &[u64], b: &[u64]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x & y).count_ones()).sum()
}

/// A row of the measurement elimination: `i^r X^x Z^z`, plus the set of
/// measured qubits whose observables multiply to it.
#[derive(Clone, Debug)]
struct Row {
    x: Vec<u64>,
    z: Vec<u64>,
    r: u32,
    comb: Vec<u64>,
}

impl Row {
    /// `self <- self · o`.
    fn mul(&mut self, o: &Row) {
        // (i^a X^p Z^q)(i^b X^s Z^t) = i^{a+b} (-1)^{q·s} X^{p+s} Z^{q+t}
        self.r = (self.r + o.r + 2 * and_count(&self.z, &o.x)) % 4;
        xor_into(&mut self.x, &o.x);
        xor_into(&mut self.z, &o.z);
        xor_into(&mut self.comb, &o.comb);
    }
}

/// Same, on the active register (`d <= 62` qubits).
#[derive(Clone, Copy, Debug)]
struct SRow {
    x: u64,
    z: u64,
    r: u32,
    idx: usize, // index into the comb table
}

impl SRow {
    fn mul(&mut self, o: &SRow, combs: &mut [Vec<u64>]) {
        self.r = (self.r + o.r + 2 * (self.z & o.x).count_ones()) % 4;
        self.x ^= o.x;
        self.z ^= o.z;
        let oc = combs[o.idx].clone();
        xor_into(&mut combs[self.idx], &oc);
    }
    /// Conjugation `P -> G P G†` by a Clifford gate on the register.
    fn h(&mut self, q: usize) {
        let (xb, zb) = (self.x >> q & 1, self.z >> q & 1);
        // H X^a Z^b H = Z^a X^b = (-1)^{ab} X^b Z^a
        self.r = (self.r + 2 * (xb & zb) as u32) % 4;
        self.x = (self.x & !(1 << q)) | (zb << q);
        self.z = (self.z & !(1 << q)) | (xb << q);
    }
    fn cnot(&mut self, c: usize, t: usize) {
        // X_c -> X_c X_t, Z_t -> Z_c Z_t, no phase.
        self.x ^= (self.x >> c & 1) << t;
        self.z ^= (self.z >> t & 1) << c;
    }
    fn cz(&mut self, a: usize, b: usize) {
        // X_a -> X_a Z_b, X_b -> Z_a X_b (no reordering needed).
        self.z ^= (self.x >> a & 1) << b;
        self.z ^= (self.x >> b & 1) << a;
    }
    fn s(&mut self, q: usize) {
        // S X S† = i X Z.
        let xb = self.x >> q & 1;
        self.r = (self.r + xb as u32) % 4;
        self.z ^= xb << q;
    }
}

// ---------------------------------------------------------------------------
// GF(2) frame on Vec<u64>: L x = coordinates of x in a basis built in axis
// order; L^{-T} z = B^T z.

struct Gf2Frame {
    n: usize,
    w: usize,
    basis: Vec<Vec<u64>>,
    /// (pivot, reduced vector, combination of basis indices)
    ech: Vec<(usize, Vec<u64>, Vec<u64>)>,
}

impl Gf2Frame {
    fn new(n: usize, w: usize) -> Self {
        Gf2Frame {
            n,
            w,
            basis: Vec::new(),
            ech: Vec::new(),
        }
    }

    fn reduce(&self, v: &[u64]) -> (Vec<u64>, Vec<u64>) {
        let mut v = v.to_vec();
        let mut comb = vec![0u64; self.w];
        for (p, e, c) in &self.ech {
            if get(&v, *p) {
                xor_into(&mut v, e);
                xor_into(&mut comb, c);
            }
        }
        (v, comb)
    }

    fn push(&mut self, v: &[u64]) -> bool {
        let (r, mut comb) = self.reduce(v);
        let Some(p) = (0..self.n).find(|&i| get(&r, i)) else {
            return false;
        };
        flip(&mut comb, self.basis.len());
        self.basis.push(v.to_vec());
        for (_, e, c) in self.ech.iter_mut() {
            if get(e, p) {
                xor_into(e, &r);
                xor_into(c, &comb);
            }
        }
        self.ech.push((p, r, comb));
        true
    }

    fn complete(&mut self) {
        for q in 0..self.n {
            let mut e = vec![0u64; self.w];
            flip(&mut e, q);
            self.push(&e);
        }
    }

    /// Image of the Hermitian string `i^{|x∧z|} X^x Z^z` (words: x then z)
    /// under `V · V†`: `(negated, x', z')`.
    fn apply(&self, p: &[u64]) -> (bool, Vec<u64>, Vec<u64>) {
        let w = self.w;
        let (x, z) = (&p[..w], &p[w..2 * w]);
        let (r, x2) = self.reduce(x);
        debug_assert!(r.iter().all(|&v| v == 0));
        let mut z2 = vec![0u64; w];
        for (i, b) in self.basis.iter().enumerate() {
            if and_count(b, z) & 1 == 1 {
                flip(&mut z2, i);
            }
        }
        let diff = and_count(x, z) as i32 - and_count(&x2, &z2) as i32;
        debug_assert!(diff % 2 == 0);
        (diff.rem_euclid(4) == 2, x2, z2)
    }
}

// ---------------------------------------------------------------------------
// Compilation for the Schrödinger picture.

struct StateCompiled {
    n: usize,
    w: usize,
    /// Hermitian axis words (x then z) and angle.
    rots: Vec<(Vec<u64>, f64)>,
    tab: HeisenbergTableau,
    /// The Clifford `C`, in time order (used to expand the state in tests).
    cliffords: Vec<Gate>,
}

fn compile_state(circuit: &Circuit) -> Result<StateCompiled, SimError> {
    let n = circuit.num_qubits;
    let mut tab = HeisenbergTableau::new(n);
    let w = tab.words();
    let mut rots = Vec::new();
    let mut cliffords = Vec::new();
    let mut measured = false;
    for op in &circuit.ops {
        let g = match op {
            Op::Gate(g) => {
                assert!(
                    !measured,
                    "adaptive: gates after a measurement are not supported"
                );
                g
            }
            Op::Measure(_) => {
                measured = true;
                continue;
            }
            _ => panic!("adaptive: circuit must be unitary (optionally ending in measurements)"),
        };
        check_gate(g, n)?;
        for g in g.decompose_to_clifford_rz() {
            if g.is_clifford() {
                tab.apply_clifford(&g);
                cliffords.push(g);
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
                    cliffords.push(Gate::S(a));
                }
                continue;
            }
            let mut z = vec![0u64; 2 * w];
            z[w + a / 64] |= 1 << (a % 64);
            let (neg, q) = tab.map(&z);
            rots.push((q, if neg { -theta } else { theta }));
        }
    }
    Ok(StateCompiled {
        n,
        w,
        rots,
        tab,
        cliffords,
    })
}

/// Size `d` of the dense active register [`CompressedState::new`] would
/// need for a unitary `circuit` (an upper bound on its non-Clifford
/// rotation count): the rank of the x-parts of the Heisenberg-mapped
/// rotation axes. Polynomial time (no amplitudes are touched), so planners
/// can call it to decide between the compressed state and a full state
/// vector. Panics on non-unitary circuits like [`CompressedState::new`].
pub fn active_dimension(circuit: &Circuit) -> Result<usize, SimError> {
    let comp = compile_state(circuit)?;
    let mut frame = Gf2Frame::new(comp.n, comp.w);
    for (q, _) in &comp.rots {
        frame.push(&q[..comp.w]);
    }
    Ok(frame.basis.len())
}

/// The active-dimension profile `d_1 ≤ d_2 ≤ … ≤ d_m` (one entry per
/// non-Clifford rotation, after Clifford absorption and merging of
/// half-π multiples), i.e. the `d_profile` [`CompressedState::new`] would
/// record, without touching any amplitudes. `Σ_j 2^{d_j}` is the exact
/// number of amplitude updates of the compressed Schrödinger evolution, so
/// this is an O(gates · n) cost oracle for that engine.
pub fn active_dimension_profile(circuit: &Circuit) -> Result<Vec<usize>, SimError> {
    let comp = compile_state(circuit)?;
    let mut frame = Gf2Frame::new(comp.n, comp.w);
    let mut prof = Vec::with_capacity(comp.rots.len());
    for (q, _) in &comp.rots {
        frame.push(&q[..comp.w]);
        prof.push(frame.basis.len());
    }
    Ok(prof)
}

/// Whether `<0|U† Z_S U|0> = 0` follows from the x-span lemma alone
/// (research/pauli.md §2): the Clifford image `C† Z_S C` has an x part
/// outside the span of every rotation axis' x part, so every Pauli path
/// ends with `x ≠ 0`. O(gates · n) — the first step of the frame engine.
pub fn z_product_vanishes(circuit: &Circuit, qubits: &[usize]) -> Result<bool, SimError> {
    let comp = compile_state(circuit)?;
    let w = comp.w;
    let mut frame = Gf2Frame::new(comp.n, w);
    for (q, _) in &comp.rots {
        frame.push(&q[..w]);
    }
    let mut p = vec![0u64; 2 * w];
    for &q in qubits {
        p[w + q / 64] ^= 1 << (q % 64);
    }
    let (_, img) = comp.tab.map(&p);
    let (rest, _) = frame.reduce(&img[..w]);
    Ok(rest.iter().any(|&v| v != 0))
}

/// Statistics of a compressed-state run.
#[derive(Clone, Debug, Default)]
pub struct CompressedStats {
    /// Non-Clifford rotations applied.
    pub rotations: usize,
    /// Final size of the active register.
    pub active_qubits: usize,
    /// `d_j` for `j = 1..=rotations`.
    pub d_profile: Vec<usize>,
    /// Σ_j 2^{d_j}: amplitudes touched by the rotations.
    pub element_ops: u64,
    /// Wall-clock seconds: compilation (tableau + frame), dense evolution.
    pub compile_secs: f64,
    pub evolve_secs: f64,
}

/// The exact state `C V† (|φ> ⊗ |0>)` of a Clifford+T circuit applied to
/// `|0^n>`, with `|φ>` dense on the `d` active qubits.
pub struct CompressedState {
    n: usize,
    w: usize,
    d: usize,
    amp: Vec<C64>,
    frame: Gf2Frame,
    tab: HeisenbergTableau,
    cliffords: Vec<Gate>,
    pub stats: CompressedStats,
}

impl CompressedState {
    /// Simulates `circuit` (Clifford gates plus any Z rotations; other
    /// gates are decomposed) on `|0^n>`. Fails with `TooLarge` if the
    /// active register would exceed `max_qubits` or the crate's memory cap.
    pub fn new(circuit: &Circuit, max_qubits: usize) -> Result<Self, SimError> {
        let t0 = Instant::now();
        let comp = compile_state(circuit)?;
        let (n, w) = (comp.n, comp.w);
        let mut frame = Gf2Frame::new(n, w);
        let mut dprof = Vec::with_capacity(comp.rots.len());
        for (q, _) in &comp.rots {
            frame.push(&q[..w]);
            dprof.push(frame.basis.len());
        }
        let d = frame.basis.len();
        if d > 62 {
            return Err(SimError::TooLarge {
                what: "compressed state vector",
                bytes: dense_bytes(d),
                limit: dense_bytes(max_qubits).min(MAX_STATE_BYTES),
            });
        }
        frame.complete();
        let mut mapped = Vec::with_capacity(comp.rots.len());
        for ((q, theta), &dj) in comp.rots.iter().zip(&dprof) {
            let (neg, x2, z2) = frame.apply(q);
            debug_assert!(x2[1..].iter().all(|&v| v == 0) && x2[0] >> dj == 0);
            let m = (1u64 << dj) - 1;
            mapped.push((x2[0], z2[0] & m, if neg { -theta } else { *theta }, dj));
        }
        let compile_secs = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let (amp, ops) = evolve(mapped.into_iter(), d, max_qubits)?;
        let stats = CompressedStats {
            rotations: dprof.len(),
            active_qubits: d,
            d_profile: dprof,
            element_ops: ops,
            compile_secs,
            evolve_secs: t1.elapsed().as_secs_f64(),
        };
        Ok(CompressedState {
            n,
            w,
            d,
            amp,
            frame,
            tab: comp.tab,
            cliffords: comp.cliffords,
            stats,
        })
    }

    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// Size `d` of the active register (`2^d` amplitudes are stored).
    pub fn active_qubits(&self) -> usize {
        self.d
    }

    /// The dense active-register amplitudes `|φ>`.
    pub fn active_amplitudes(&self) -> &[C64] {
        &self.amp
    }

    /// Hermitian string `p` (2w words) mapped through `V C† · C V†`.
    fn to_frame(&self, p: &[u64]) -> (bool, Vec<u64>, Vec<u64>) {
        let (neg, im) = self.tab.map(p);
        let (neg2, x, z) = self.frame.apply(&im);
        (neg ^ neg2, x, z)
    }

    /// Exact `<ψ| O |ψ>`.
    pub fn expectation(&self, obs: &PauliSum) -> f64 {
        assert_eq!(obs.n, self.n);
        let w = self.w;
        let mask = (1u64 << self.d) - 1;
        let mut terms = Vec::new();
        for (k, &c) in obs.keys.chunks(2 * obs.w).zip(&obs.coefs) {
            let mut p = vec![0u64; 2 * w];
            p[..w].copy_from_slice(&k[..w]);
            p[w..].copy_from_slice(&k[obs.w..obs.w + w]);
            let (neg, x, z) = self.to_frame(&p);
            // x outside the active register: <0|X..|0> = 0 on the rest.
            if x[1..].iter().any(|&v| v != 0) || x[0] & !mask != 0 {
                continue;
            }
            terms.push((x[0], z[0] & mask, if neg { -c } else { c }));
        }
        eval_terms(&self.amp, &mut terms).0
    }

    /// Expands to a full `2^n` state vector (tests; `n <= 24`). Equal to the
    /// circuit's state up to a global phase.
    pub fn to_statevector(&self) -> StateVectorF64 {
        assert!(self.n <= 24, "to_statevector: too many qubits");
        let mut full = vec![C64::new(0.0, 0.0); 1 << self.n];
        let cols: Vec<usize> = self.frame.basis[..self.d]
            .iter()
            .map(|b| b[0] as usize)
            .collect();
        for (c, &v) in self.amp.iter().enumerate() {
            let mut idx = 0usize;
            for (i, col) in cols.iter().enumerate() {
                if c >> i & 1 == 1 {
                    idx ^= col;
                }
            }
            full[idx] = v;
        }
        let mut sv = StateVectorF64::from_amplitudes(full);
        for g in &self.cliffords {
            sv.apply_gate(g).expect("valid gate");
        }
        sv
    }

    /// Builds an exact sampler of all `n` qubits (consumes the state).
    pub fn sampler(self) -> Sampler {
        Sampler::new(self)
    }
}

// ---------------------------------------------------------------------------
// Sampling all n qubits of C V† (|φ> ⊗ |0>).

/// Statistics of building a [`Sampler`].
#[derive(Clone, Debug, Default)]
pub struct SamplerStats {
    /// Measured outcomes that are uniform and independent of the rest.
    pub random_bits: usize,
    /// Outcomes fixed by the state (given the others).
    pub determined: usize,
    /// Outcomes read from the dense register.
    pub dense_bits: usize,
    /// Dense passes spent diagonalising the measured observables.
    pub dense_passes: usize,
    pub build_secs: f64,
}

/// Exact sampler of computational-basis outcomes of a [`CompressedState`].
///
/// The measured observables `D† Z_q D` (`D = C V†`) are put into a
/// generator form by Gaussian elimination: `r` generators have independent
/// X parts on the inert qubits (their outcomes are uniform and independent),
/// the others act on the active register only. A Clifford `E` on the active
/// register (Hadamards, one CNOT network, one diagonal phase layer) maps
/// those to single-qubit Z's (or to `±I`: determined outcomes), so after
/// `|φ> -> E|φ>` a shot is: draw `y` from `|Eφ(y)|^2`, draw `r` random bits,
/// and map the generator outcomes back to qubit outcomes with a fixed
/// GF(2) matrix.
pub struct Sampler {
    n: usize,
    w: usize,
    /// Cumulative distribution of `|Eφ(y)|^2` (normalised).
    cdf: Vec<f64>,
    /// Outcome vector with every free bit 0 (n bits).
    b0: Vec<u64>,
    /// For each dense pivot: (register bit, column to XOR when it is 1).
    pivot_cols: Vec<(usize, Vec<u64>)>,
    /// Columns to XOR for each uniformly random generator bit.
    random_cols: Vec<Vec<u64>>,
    pub stats: SamplerStats,
}

/// Inverse of an `n × n` GF(2) matrix given by rows (n bits each).
fn gf2_inverse(rows: &[Vec<u64>], n: usize) -> Vec<Vec<u64>> {
    let w = n.div_ceil(64).max(1);
    let mut a: Vec<Vec<u64>> = rows.to_vec();
    let mut inv: Vec<Vec<u64>> = (0..n)
        .map(|i| {
            let mut e = vec![0u64; w];
            flip(&mut e, i);
            e
        })
        .collect();
    for col in 0..n {
        let p = (col..n)
            .find(|&r| get(&a[r], col))
            .expect("measurement generators must be independent");
        a.swap(col, p);
        inv.swap(col, p);
        let (pa, pi) = (a[col].clone(), inv[col].clone());
        for r in 0..n {
            if r != col && get(&a[r], col) {
                xor_into(&mut a[r], &pa);
                xor_into(&mut inv[r], &pi);
            }
        }
    }
    inv
}

impl Sampler {
    fn new(st: CompressedState) -> Self {
        let t0 = Instant::now();
        let (n, w, d) = (st.n, st.w, st.d);
        // 1. The measured observables P_q = D† Z_q D, as i^r X^x Z^z rows.
        let mut rows: Vec<Row> = (0..n)
            .map(|q| {
                let mut p = vec![0u64; 2 * w];
                flip(&mut p[w..], q);
                let (neg, x, z) = st.to_frame(&p);
                let r = (and_count(&x, &z) + if neg { 2 } else { 0 }) % 4;
                let mut comb = vec![0u64; w];
                flip(&mut comb, q);
                Row { x, z, r, comb }
            })
            .collect();
        // 2. Eliminate X on the inert qubits d..n: those pivots are random.
        let mut random: Vec<Row> = Vec::new();
        for col in d..n {
            let Some(pi) = rows.iter().position(|r| get(&r.x, col)) else {
                continue;
            };
            let p = rows.swap_remove(pi);
            for r in rows.iter_mut() {
                if get(&r.x, col) {
                    r.mul(&p);
                }
            }
            random.push(p);
        }
        // 3. The rest act on the active register (Z on inert qubits = +1).
        let mask = if d == 0 { 0 } else { (1u64 << d) - 1 };
        let mut combs: Vec<Vec<u64>> = Vec::with_capacity(rows.len());
        let mut k: Vec<SRow> = rows
            .into_iter()
            .map(|r| {
                debug_assert!(r.x[1..].iter().all(|&v| v == 0) && r.x[0] & !mask == 0);
                combs.push(r.comb);
                SRow {
                    x: r.x[0],
                    z: r.z[0] & mask,
                    r: r.r,
                    idx: combs.len() - 1,
                }
            })
            .collect();
        // 4. Diagonalise them with a Clifford E on the register.
        let mut amp = st.amp;
        let mut passes = 0usize;
        let mut pivots: Vec<(usize, usize)> = Vec::new(); // (row, column)
        let mut is_pivot_col = 0u64;
        let mut determined: Vec<usize> = Vec::new();
        // Phase 1: echelon form in X, with Hadamards where a row is Z-only.
        for i in 0..k.len() {
            for &(pr, pc) in &pivots {
                if k[i].x >> pc & 1 == 1 {
                    let o = k[pr];
                    k[i].mul(&o, &mut combs);
                }
            }
            if k[i].x == 0 {
                let free = k[i].z & !is_pivot_col;
                if free == 0 {
                    debug_assert!(
                        k[i].z == 0,
                        "commuting rows: Z on pivots only implies identity"
                    );
                    debug_assert!(k[i].r % 2 == 0);
                    determined.push(i);
                    continue;
                }
                let c = free.trailing_zeros() as usize;
                for row in k.iter_mut() {
                    row.h(c);
                }
                hadamard_dense(&mut amp, c);
                passes += 1;
            }
            let c = k[i].x.trailing_zeros() as usize;
            debug_assert!(is_pivot_col >> c & 1 == 0);
            for &(pr, _) in &pivots {
                if k[pr].x >> c & 1 == 1 {
                    let o = k[i];
                    k[pr].mul(&o, &mut combs);
                }
            }
            pivots.push((i, c));
            is_pivot_col |= 1 << c;
        }
        // Phase 2: CNOTs clear the non-pivot X bits (pivot rows: X = e_c).
        let mut cnots: Vec<(usize, usize)> = Vec::new();
        for &(pr, pc) in &pivots {
            let mut extra = k[pr].x & !(1u64 << pc);
            while extra != 0 {
                let t = extra.trailing_zeros() as usize;
                extra &= extra - 1;
                for row in k.iter_mut() {
                    row.cnot(pc, t);
                }
                cnots.push((pc, t));
            }
        }
        if !cnots.is_empty() {
            apply_cnot_network(&mut amp, &cnots);
            passes += 1;
        }
        // Phase 3: CZ and S clear the Z parts.
        let mut czs: Vec<(usize, usize)> = Vec::new();
        let mut smask = 0u64;
        for &(pr, pc) in &pivots {
            let mut zs = k[pr].z & !(1u64 << pc);
            while zs != 0 {
                let j = zs.trailing_zeros() as usize;
                zs &= zs - 1;
                for row in k.iter_mut() {
                    row.cz(pc, j);
                }
                czs.push((pc, j));
            }
            if k[pr].z >> pc & 1 == 1 {
                for row in k.iter_mut() {
                    row.s(pc);
                }
                smask |= 1 << pc;
            }
        }
        if !czs.is_empty() || smask != 0 {
            // Phase i^{Σ_S y_c} (-1)^{Σ_{(a,b)} y_a y_b}, the quadratic form
            // stored as one neighbour mask per first endpoint.
            let mut nbr = [0u64; 64];
            let mut firsts = 0u64;
            for &(a, b) in &czs {
                nbr[a] ^= 1 << b;
                firsts |= 1 << a;
            }
            for_each_amp(&mut amp, |y, v| {
                let y = y as u64;
                let mut e = (y & smask).count_ones();
                let mut f = y & firsts;
                while f != 0 {
                    let a = f.trailing_zeros() as usize;
                    f &= f - 1;
                    e += 2 * (y & nbr[a]).count_ones();
                }
                *v *= i_pow(e);
            });
            passes += 1;
        }
        // Phase 4: Hadamards turn ±X_c into ±Z_c.
        for &(pr, pc) in &pivots {
            debug_assert!(k[pr].x == 1 << pc && k[pr].z == 0 && k[pr].r % 2 == 0);
        }
        for &(pr, pc) in &pivots {
            for row in k.iter_mut() {
                row.h(pc);
            }
            debug_assert!(k[pr].x == 0 && k[pr].z == 1 << pc);
            hadamard_dense(&mut amp, pc);
            passes += 1;
        }
        // 5. Distribution of y.
        let mut cdf: Vec<f64> = if amp.len() < PAR_MIN {
            amp.iter().map(|v| v.norm_sqr()).collect()
        } else {
            amp.par_iter()
                .with_min_len(SUB)
                .map(|v| v.norm_sqr())
                .collect()
        };
        drop(amp);
        let mut acc = 0.0;
        for p in cdf.iter_mut() {
            acc += *p;
            *p = acc;
        }
        for p in cdf.iter_mut() {
            *p /= acc;
        }
        // 6. Outcomes: generator rows are random, pivot (dense) and
        // determined; their outcome bits m satisfy m = T b, so b = T^{-1} m.
        let mut trows: Vec<Vec<u64>> = Vec::with_capacity(n);
        let mut m0 = vec![0u64; w]; // fixed part of m (n bits)
        let mut gen = 0usize;
        let mut random_gen = Vec::new();
        for r in &random {
            trows.push(r.comb.clone());
            random_gen.push(gen);
            gen += 1;
        }
        let mut pivot_gen = Vec::new();
        for &(pr, pc) in &pivots {
            trows.push(combs[k[pr].idx].clone());
            if k[pr].r == 2 {
                flip(&mut m0, gen);
            }
            pivot_gen.push((pc, gen));
            gen += 1;
        }
        for &i in &determined {
            trows.push(combs[k[i].idx].clone());
            if k[i].r == 2 {
                flip(&mut m0, gen);
            }
            gen += 1;
        }
        assert_eq!(gen, n);
        let tinv = gf2_inverse(&trows, n);
        let column = |g: usize| -> Vec<u64> {
            let mut c = vec![0u64; w];
            for (q, row) in tinv.iter().enumerate() {
                if get(row, g) {
                    flip(&mut c, q);
                }
            }
            c
        };
        let mut b0 = vec![0u64; w];
        for (q, row) in tinv.iter().enumerate() {
            if and_count(row, &m0) & 1 == 1 {
                flip(&mut b0, q);
            }
        }
        let stats = SamplerStats {
            random_bits: random.len(),
            determined: determined.len(),
            dense_bits: pivots.len(),
            dense_passes: passes,
            build_secs: t0.elapsed().as_secs_f64(),
        };
        Sampler {
            n,
            w,
            cdf,
            b0,
            pivot_cols: pivot_gen.iter().map(|&(pc, g)| (pc, column(g))).collect(),
            random_cols: random_gen.iter().map(|&g| column(g)).collect(),
            stats,
        }
    }

    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// One shot: the `n` measured bits, packed (qubit `q` is bit `q % 64`
    /// of word `q / 64`).
    pub fn sample_packed<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<u64> {
        let u: f64 = rng.random::<f64>();
        let y = self
            .cdf
            .partition_point(|&c| c <= u)
            .min(self.cdf.len() - 1) as u64;
        let mut b = self.b0.clone();
        for (pc, col) in &self.pivot_cols {
            if y >> pc & 1 == 1 {
                xor_into(&mut b, col);
            }
        }
        for col in &self.random_cols {
            if rng.random::<bool>() {
                xor_into(&mut b, col);
            }
        }
        b
    }

    /// `shots` samples as basis-state indices (requires `n <= 64`).
    pub fn sample_indices<R: Rng + ?Sized>(&self, shots: usize, rng: &mut R) -> Vec<u64> {
        assert!(self.n <= 64);
        (0..shots).map(|_| self.sample_packed(rng)[0]).collect()
    }

    /// The exact outcome distribution over all `2^n` bit strings (tests,
    /// `n <= 20`): enumerates every dense outcome and random pattern.
    pub fn distribution(&self) -> Vec<f64> {
        assert!(self.n <= 20 && self.random_cols.len() <= 20);
        let mut dist = vec![0.0; 1 << self.n];
        let pr = 0.5f64.powi(self.random_cols.len() as i32);
        let mut prev = 0.0;
        for (y, &c) in self.cdf.iter().enumerate() {
            let p = c - prev;
            prev = c;
            if p == 0.0 {
                continue;
            }
            let mut b = self.b0[0];
            for (pc, col) in &self.pivot_cols {
                if y >> pc & 1 == 1 {
                    b ^= col[0];
                }
            }
            for pat in 0..1usize << self.random_cols.len() {
                let mut bb = b;
                for (i, col) in self.random_cols.iter().enumerate() {
                    if pat >> i & 1 == 1 {
                        bb ^= col[0];
                    }
                }
                dist[bb as usize] += p * pr;
            }
        }
        dist
    }

    /// Words per sample.
    pub fn words(&self) -> usize {
        self.w
    }
}

/// `|y> -> |M y>` for the CNOT sequence (in time order), as one gather pass.
fn apply_cnot_network(a: &mut Vec<C64>, cnots: &[(usize, usize)]) {
    let d = a.len().trailing_zeros() as usize;
    if cnots.len() <= 2 {
        for &(c, t) in cnots {
            for_pairs(a, 1 << t, |y, lo, hi| {
                if y >> c & 1 == 1 {
                    std::mem::swap(lo, hi);
                }
            });
        }
        return;
    }
    // M^{-1} = the same CNOTs in reverse order; columns of M^{-1}.
    let inv_col = |i: usize| -> u64 {
        let mut y = 1u64 << i;
        for &(c, t) in cnots.iter().rev() {
            y ^= (y >> c & 1) << t;
        }
        y
    };
    let cols: Vec<u64> = (0..d).map(inv_col).collect();
    // Byte tables: M^{-1} y = XOR of table lookups.
    let nt = d.div_ceil(8);
    let tables: Vec<[u64; 256]> = (0..nt)
        .map(|t| {
            let mut tab = [0u64; 256];
            for (v, e) in tab.iter_mut().enumerate() {
                for b in 0..8 {
                    let i = t * 8 + b;
                    if i < d && v >> b & 1 == 1 {
                        *e ^= cols[i];
                    }
                }
            }
            tab
        })
        .collect();
    let src = std::mem::take(a);
    let map = |y: usize| -> C64 {
        let mut s = 0u64;
        for (t, tab) in tables.iter().enumerate() {
            s ^= tab[(y >> (8 * t)) & 0xff];
        }
        src[s as usize]
    };
    *a = if src.len() < PAR_MIN {
        (0..src.len()).map(map).collect()
    } else {
        (0..src.len())
            .into_par_iter()
            .with_min_len(SUB)
            .map(map)
            .collect()
    };
}

// ---------------------------------------------------------------------------
// Expectation values: Heisenberg sweep that may switch to the dense picture.

/// When the Heisenberg sweep hands over to the dense register.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Strategy {
    /// Never switch: the pure rotation-frame Pauli-path engine.
    Frame,
    /// Switch before the first rotation: pure compressed Schrödinger.
    Dense,
    /// Switch when `k` rotations remain (or as soon as possible after).
    SwitchAt(usize),
    /// Cost model fed by the live term count (see [`AdaptiveOptions`]).
    Auto,
}

/// Options for [`expectation`].
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveOptions {
    pub frame: FrameOptions,
    pub strategy: Strategy,
    /// Largest active register the dense side may use.
    pub max_dense_qubits: usize,
    /// Seconds per dense element operation (one amplitude in one pass).
    pub dense_secs_per_op: f64,
    /// Seconds per Pauli-term visit, used until enough visits have been
    /// timed to measure it live.
    pub term_secs_per_visit: f64,
    /// Prior for the log2 growth of the term count per rotation that does
    /// not grow the x-span (random circuits: about 0.5; worst case 1).
    pub growth_prior: f64,
    /// Count span-preserving rotations that leave the live term count
    /// unchanged as zero-growth evidence even below 32 terms (where
    /// doublings are ignored as small-number noise). Without it the growth
    /// meter never sees evidence on circuits whose Heisenberg terms stay
    /// tiny (adders: 4 live terms), keeps the 0.5-per-rotation prior, and
    /// [`Strategy::Auto`] hands over to a 25-qubit register at the start:
    /// 350× slower than the frame (research/planner.md §4). With it, Auto
    /// also keeps sweeping (no hand-over) until 8 rotations have been
    /// observed, as long as a rotation costs less in the frame than in the
    /// dense register.
    pub flat_evidence: bool,
}

impl Default for AdaptiveOptions {
    fn default() -> Self {
        AdaptiveOptions {
            frame: FrameOptions::default(),
            strategy: Strategy::Auto,
            max_dense_qubits: 26,
            dense_secs_per_op: 2e-9,
            term_secs_per_visit: 6e-8,
            growth_prior: 0.5,
            flat_evidence: true,
        }
    }
}

/// What [`expectation`] did.
#[derive(Clone, Debug, Default)]
pub struct AdaptiveReport {
    pub value: f64,
    /// Rotations left to the dense side (`None`: the frame finished).
    pub switched_at: Option<usize>,
    /// Active register size at the switch.
    pub dense_qubits: usize,
    /// Terms handed over at the switch.
    pub handover_terms: usize,
    pub frame_stats: PathStats,
    /// Dense element operations (evolution + evaluation).
    pub dense_ops: u64,
    pub frame_secs: f64,
    pub dense_secs: f64,
}

/// Cost-model decision for [`Strategy::Auto`] at stage `k` (rotations
/// `1..=k` remain, `t` live terms). Projects the Heisenberg cost of
/// continuing to every later stage `k'` and adds the dense cost of
/// finishing from `k'` (`Σ_{j<=k'} 2^{d_j}` amplitude updates plus the
/// handover evaluation, about `terms · 2^{d_k'}`); switches now iff `k' = k`
/// is the cheapest choice, otherwise re-decides at the next stage with the
/// new live term count. Term growth: a rotation that grows the x-span
/// cannot increase the term count (its branch leaves the span and is pruned
/// at the next projection); one that does not grow it multiplies the count
/// by `2^growth`, with `growth` (log2 per such rotation) measured live.
fn auto_decide(
    k: usize,
    t: usize,
    d: &[usize],
    visit_secs: f64,
    growth: f64,
    opt: &AdaptiveOptions,
) -> bool {
    let dense_ok = |j: usize| d[j] <= opt.max_dense_qubits.min(30);
    if t == usize::MAX {
        return dense_ok(k);
    }
    let g = 2f64.powf(growth.clamp(0.0, 1.0));
    let mut prefix = vec![0.0f64; k + 1];
    for i in 1..=k {
        prefix[i] = prefix[i - 1] + (1u64 << d[i].min(62)) as f64;
    }
    let dense_cost = |j: usize, terms: f64| -> f64 {
        let dim = (1u64 << d[j].min(62)) as f64;
        let groups = terms.min(dim);
        let eval = dim * terms.min(groups * (d[j] as f64 + 1.0));
        opt.dense_secs_per_op * (prefix[j] + eval)
    };
    let mut best_j = None;
    let mut best = f64::INFINITY;
    let mut terms = t.max(1) as f64;
    let mut heis = 0.0;
    for j in (0..=k).rev() {
        if dense_ok(j) {
            let c = heis + dense_cost(j, terms);
            if c < best {
                best = c;
                best_j = Some(j);
            }
        }
        if j == 0 {
            // Finishing in the frame.
            if heis < best {
                best_j = None;
            }
            break;
        }
        heis += visit_secs * terms;
        if d[j] == d[j - 1] {
            terms = (terms * g).min(4f64.powi(d[j] as i32));
        }
    }
    best_j == Some(k)
}

/// Live estimate of the term growth per span-preserving rotation (log2),
/// with a prior of [`AdaptiveOptions::growth_prior`] worth 4 observations.
struct GrowthMeter {
    prior: f64,
    flat_evidence: bool,
    sum: f64,
    count: f64,
    last: Option<(usize, usize)>, // (stage, terms)
}

impl GrowthMeter {
    fn observe(&mut self, k: usize, t: usize, d: &[usize]) {
        if let Some((k0, t0)) = self.last {
            // Rotation k0 was just processed (stage k0 -> k0 - 1 = k).
            // Below ~32 terms the first branchings always double the count;
            // that small-number regime says nothing about the growth rate.
            if k0 == k + 1 && d[k0] == d[k] && t > 0 && t != usize::MAX {
                if t0 >= 32 {
                    self.sum += (t as f64 / t0 as f64).log2();
                    self.count += 1.0;
                } else if self.flat_evidence && t <= t0 {
                    // the rotation commuted with every live term
                    self.count += 1.0;
                }
            }
        }
        self.last = Some((k, t));
    }
    fn growth(&self) -> f64 {
        (self.prior * 4.0 + self.sum) / (4.0 + self.count)
    }
}

/// Exact `<0| U† O U |0>`, choosing at run time between the Pauli-path
/// frame and the dense active register (see [`Strategy`]).
pub fn expectation(
    circuit: &Circuit,
    observable: &PauliSum,
    opt: &AdaptiveOptions,
) -> Result<AdaptiveReport, SimError> {
    assert_eq!(observable.n, circuit.num_qubits);
    let mut fopt = opt.frame;
    fopt.prune = true;
    let t0 = Instant::now();
    let strategy = opt.strategy;
    let max_d = opt.max_dense_qubits.min(30);
    let mut meter = GrowthMeter {
        prior: opt.growth_prior,
        flat_evidence: opt.flat_evidence,
        sum: 0.0,
        count: 0.0,
        last: None,
    };
    let mut sweep_start: Option<Instant> = None;
    let mut policy = |k: usize, t: usize, d: &[usize], visits: u64| -> bool {
        match strategy {
            Strategy::Frame => false,
            Strategy::Dense => d[k] <= max_d,
            Strategy::SwitchAt(s) => (k <= s || t == usize::MAX) && d[k] <= max_d,
            Strategy::Auto => {
                let start = *sweep_start.get_or_insert_with(Instant::now);
                let visit = if visits > 100_000 {
                    start.elapsed().as_secs_f64() / visits as f64
                } else {
                    opt.term_secs_per_visit
                };
                meter.observe(k, t, d);
                // Explore before trusting the prior: until 8 rotations have
                // been observed, stay in the frame while one more rotation
                // there is cheaper than one in the dense register and the
                // hand-over evaluation (~ terms · 2^d) is still below the
                // dense evolution itself (so exploring can at most double
                // the cost of switching).
                if opt.flat_evidence && meter.count < 8.0 && t != usize::MAX {
                    let dim = (1u64 << d[k].min(62)) as f64;
                    let evolve: f64 = (1..=k).map(|i| (1u64 << d[i].min(62)) as f64).sum();
                    if (t as f64) * visit < dim * opt.dense_secs_per_op
                        && (t as f64) * dim <= evolve
                    {
                        return false;
                    }
                }
                auto_decide(k, t, d, visit, meter.growth(), opt)
            }
        }
    };
    let staged = expectation_staged(circuit, observable, &fopt, &mut policy)
        .expect("register wider than 512 qubits")?;
    let frame_secs = t0.elapsed().as_secs_f64();
    match staged {
        Staged::Done(value, stats) => Ok(AdaptiveReport {
            value,
            switched_at: None,
            frame_stats: stats,
            frame_secs,
            ..Default::default()
        }),
        Staged::Switched(sp) => {
            let t1 = Instant::now();
            let k = sp.stage;
            let dk = sp.d[k];
            let rots = sp
                .rots
                .iter()
                .enumerate()
                .map(|(j, &(x, z, th))| (x, z, th, sp.d[j + 1]));
            let (amp, ops) = evolve(rots, dk, opt.max_dense_qubits)?;
            let mut terms = sp.terms.clone();
            let (value, eops) = eval_terms(&amp, &mut terms);
            Ok(AdaptiveReport {
                value,
                switched_at: Some(k),
                dense_qubits: dk,
                handover_terms: sp.terms.len(),
                frame_stats: sp.stats,
                dense_ops: ops + eops,
                frame_secs,
                dense_secs: t1.elapsed().as_secs_f64(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairs_cover_every_index_once() {
        for d in [3usize, 15, 16] {
            for x in [1u64, 5, (1 << (d - 1)) | 3, (1 << d) - 1] {
                let mut a = vec![C64::new(0.0, 0.0); 1 << d];
                for_pairs(&mut a, x, |y, lo, hi| {
                    lo.re += 1.0 + y as f64;
                    hi.im += 1.0 + y as f64;
                });
                for (y, v) in a.iter().enumerate() {
                    let top = 63 - x.leading_zeros();
                    if y >> top & 1 == 0 {
                        assert_eq!(v.re, 1.0 + y as f64);
                        assert_eq!(v.im, 0.0);
                    } else {
                        assert_eq!(v.im, 1.0 + (y as u64 ^ x) as f64);
                        assert_eq!(v.re, 0.0);
                    }
                }
            }
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn gf2_inverse_roundtrip() {
        let rows = vec![vec![0b011u64], vec![0b110], vec![0b001]];
        let inv = gf2_inverse(&rows, 3);
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0u32;
                for k in 0..3 {
                    s ^= (get(&rows[i], k) & get(&inv[k], j)) as u32;
                }
                assert_eq!(s, (i == j) as u32);
            }
        }
    }
}
