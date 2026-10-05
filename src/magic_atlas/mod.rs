//! The magic atlas: how much non-stabilizer structure real algorithm
//! circuits accumulate, gate by gate, measured by cheap exact invariants of
//! the rotation frame (research/simulability/magic-atlas.md).
//!
//! For a unitary circuit `U` on `|0^n>` (lowered to Clifford + Z rotations)
//! the rotation frame writes `U_k = C_k R_{m_k} ⋯ R_1` after gate `k`, with
//! `R_j = exp(-iθ_j Q_j/2)` (research/performance/pauli.md, src/engines/adaptive.rs). [`profile`]
//! records, in one O(gates · n/64) pass (plus O(n²·w) for the GF(2) basis):
//!
//! * `d_k = dim span{x(Q_1..Q_{m_k})}` — the active dimension: the exact
//!   register size of the compressed-state engine ([`crate::engines::adaptive`]);
//!   `Σ_j 2^{d_j}` is its exact amplitude-update count;
//! * `f_k` — the **factored** active dimension: in the CNOT frame `V` that
//!   maps `W_m` onto the first `d` coordinates, each rotation acts on the
//!   coordinates in the support of its mapped axis (x part and the z part
//!   restricted to already-active coordinates). Rotations only couple the
//!   coordinates they touch, so `|φ>` is exactly a tensor product over the
//!   connected components of that coupling graph (union-find, monotone).
//!   `f = max component` and `Σ_j 2^{|comp_j|}` is the exact work of
//!   [`FactoredState`];
//! * `E_k` — the stabilizer entanglement (in bits) of the Clifford skeleton
//!   state `C_k|0^n>` across a cut, from a column-major tableau. Since
//!   `U_k|0> = C_k V† (|φ> ⊗ |0>) = Σ_y φ(y) P_y C_k|0>` with Pauli `P_y`,
//!   the true Schmidt rank obeys `log2 χ(U_k|0>) ≤ E_k + d_k` (and
//!   `≤ min(|A|, |B|)`);
//! * the T-count (rotations by odd multiples of π/4) and the number of all
//!   non-Clifford rotations, and the affine support bound of
//!   [`crate::simulability::support_bound`].

#![allow(clippy::needless_range_loop)]

pub mod families;

use crate::circuit::{check_gate, Circuit, Op, SimError};
use crate::engines::adaptive::rotate_dense;
use crate::engines::pauli_frame::HeisenbergTableau;
use crate::gate::{is_multiple_of_half_pi, Gate};
use num_complex::Complex64;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};
use std::time::Instant;

type C64 = Complex64;

// ---------------------------------------------------------------------------
// Incremental GF(2) basis in reduced row-echelon form with pivot lookup.
// Reducing a vector costs O(w · |v ∩ pivots|), not O(w · d).

struct Rank {
    w: usize,
    /// pivot column -> row index (u32::MAX if none)
    piv: Vec<u32>,
    /// reduced rows (each has exactly one pivot column set among pivots)
    rows: Vec<Vec<u64>>,
    /// for each reduced row, which original basis vectors sum to it
    combs: Vec<Vec<u64>>,
    /// cols[q]: set of basis indices i with b_i[q] = 1 (for B^T z)
    cols: Vec<Vec<u64>>,
    d: usize,
}

fn bits(v: &[u64]) -> impl Iterator<Item = usize> + '_ {
    v.iter().enumerate().flat_map(|(wi, &x)| {
        let mut x = x;
        std::iter::from_fn(move || {
            if x == 0 {
                None
            } else {
                let b = x.trailing_zeros() as usize;
                x &= x - 1;
                Some(wi * 64 + b)
            }
        })
    })
}

fn xor_into(a: &mut [u64], b: &[u64]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x ^= y;
    }
}

impl Rank {
    fn new(n: usize, w: usize) -> Self {
        Rank {
            w,
            piv: vec![u32::MAX; n],
            rows: Vec::new(),
            combs: Vec::new(),
            cols: vec![vec![0u64; w]; n],
            d: 0,
        }
    }

    /// Adds `v` (x part, `w` words). Returns its coordinates in the basis
    /// (the original vectors, in insertion order).
    fn push(&mut self, v: &[u64]) -> Vec<u64> {
        let mut r = v.to_vec();
        let mut comb = vec![0u64; self.w];
        for p in bits(v).collect::<Vec<_>>() {
            let k = self.piv[p];
            if k != u32::MAX {
                xor_into(&mut r, &self.rows[k as usize]);
                xor_into(&mut comb, &self.combs[k as usize]);
            }
        }
        let Some(p) = bits(&r).next() else {
            return comb;
        };
        let idx = self.d;
        self.d += 1;
        comb[idx / 64] ^= 1 << (idx % 64);
        for (row, cb) in self.rows.iter_mut().zip(self.combs.iter_mut()) {
            if row[p / 64] >> (p % 64) & 1 == 1 {
                xor_into(row, &r);
                xor_into(cb, &comb);
            }
        }
        self.piv[p] = self.rows.len() as u32;
        self.rows.push(r);
        self.combs.push(comb);
        for q in bits(v).collect::<Vec<_>>() {
            self.cols[q][idx / 64] |= 1 << (idx % 64);
        }
        let mut e = vec![0u64; self.w];
        e[idx / 64] |= 1 << (idx % 64);
        e
    }

    /// `B^T z` restricted to the current basis.
    fn zmap(&self, z: &[u64]) -> Vec<u64> {
        let mut out = vec![0u64; self.w];
        for q in bits(z) {
            xor_into(&mut out, &self.cols[q]);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Union-find over active coordinates.

struct Dsu {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl Dsu {
    fn new(n: usize) -> Self {
        Dsu {
            parent: (0..n as u32).collect(),
            size: vec![1; n],
        }
    }
    fn find(&mut self, mut a: usize) -> usize {
        while self.parent[a] as usize != a {
            let p = self.parent[a] as usize;
            self.parent[a] = self.parent[p];
            a = p;
        }
        a
    }
    fn union(&mut self, a: usize, b: usize) -> usize {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return ra;
        }
        let (big, small) = if self.size[ra] >= self.size[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small] = big as u32;
        self.size[big] += self.size[small];
        big
    }
}

// ---------------------------------------------------------------------------
// Column-major stabilizer tableau of the Clifford skeleton (no signs):
// xcol[q], zcol[q] are bitsets over the n generators C Z_i C†.

struct SkelTab {
    n: usize,
    w: usize,
    xc: Vec<Vec<u64>>,
    zc: Vec<Vec<u64>>,
}

impl SkelTab {
    fn new(n: usize) -> Self {
        let w = n.div_ceil(64).max(1);
        let xc = vec![vec![0u64; w]; n];
        let mut zc = vec![vec![0u64; w]; n];
        for (q, z) in zc.iter_mut().enumerate() {
            z[q / 64] |= 1 << (q % 64);
        }
        SkelTab { n, w, xc, zc }
    }

    fn apply(&mut self, g: &Gate) {
        use Gate::*;
        match *g {
            I(_) | X(_) | Y(_) | Z(_) => {}
            H(q) => std::mem::swap(&mut self.xc[q], &mut self.zc[q]),
            S(q) | Sdg(q) => {
                let x = self.xc[q].clone();
                xor_into(&mut self.zc[q], &x);
            }
            Cnot(c, t) => {
                let xcv = self.xc[c].clone();
                xor_into(&mut self.xc[t], &xcv);
                let ztv = self.zc[t].clone();
                xor_into(&mut self.zc[c], &ztv);
            }
            Cz(a, b) => {
                let xa = self.xc[a].clone();
                let xb = self.xc[b].clone();
                xor_into(&mut self.zc[b], &xa);
                xor_into(&mut self.zc[a], &xb);
            }
            Swap(a, b) => {
                self.xc.swap(a, b);
                self.zc.swap(a, b);
            }
            _ => panic!("SkelTab: non-Clifford or undecomposed gate {g:?}"),
        }
    }

    /// Entanglement (bits) of the stabilizer state across `A = {0..cut}`:
    /// `rank(G|_A) - |A|`, where `G|_A` has the 2|A| columns of `A`.
    fn entanglement(&self, cut: usize) -> usize {
        let mut cols: Vec<Vec<u64>> = Vec::with_capacity(2 * cut);
        for q in 0..cut {
            cols.push(self.xc[q].clone());
            cols.push(self.zc[q].clone());
        }
        let rank = gf2_rank(&mut cols, self.n, self.w);
        rank.saturating_sub(cut)
    }
}

fn gf2_rank(vs: &mut [Vec<u64>], nbits: usize, _w: usize) -> usize {
    let mut rank = 0;
    let mut pivots: Vec<(usize, usize)> = Vec::new(); // (pivot bit, row)
    for i in 0..vs.len() {
        for &(p, r) in &pivots {
            if vs[i][p / 64] >> (p % 64) & 1 == 1 {
                let (a, b) = if r < i {
                    let (l, h) = vs.split_at_mut(i);
                    (&mut h[0], &l[r])
                } else {
                    unreachable!()
                };
                xor_into(a, b);
            }
        }
        if let Some(p) = bits(&vs[i]).next() {
            debug_assert!(p < nbits);
            pivots.push((p, i));
            rank += 1;
        }
    }
    rank
}

// ---------------------------------------------------------------------------

/// Options of [`profile`].
#[derive(Clone, Debug)]
pub struct AtlasOptions {
    /// Number of (roughly evenly spaced) checkpoints at which the skeleton
    /// entanglement is evaluated (0 = only at the end; entanglement is
    /// skipped entirely when `entanglement` is false).
    pub checkpoints: usize,
    /// Compute the skeleton stabilizer entanglement at the checkpoints.
    pub entanglement: bool,
    /// Cut position for the entanglement (default `n / 2`).
    pub cut: Option<usize>,
    /// Compute the affine support bound ([`crate::simulability::support_bound`]).
    pub support: bool,
}

impl Default for AtlasOptions {
    fn default() -> Self {
        AtlasOptions {
            checkpoints: 64,
            entanglement: true,
            cut: None,
            support: true,
        }
    }
}

/// One checkpoint of the profile.
#[derive(Clone, Debug, Default)]
pub struct Checkpoint {
    /// Index of the last original gate applied.
    pub gate: usize,
    /// Non-Clifford rotations so far.
    pub rotations: usize,
    /// T-like rotations so far.
    pub t_count: usize,
    /// Active dimension `d_k` after this gate.
    pub d: usize,
    /// Largest factored component `f` so far.
    pub f: usize,
    /// Skeleton stabilizer entanglement at the cut (bits).
    pub e_stab: usize,
}

/// Result of [`profile`].
#[derive(Clone, Debug, Default)]
pub struct AtlasProfile {
    /// Number of qubits.
    pub n: usize,
    /// Original gates.
    pub gates: usize,
    /// Gates after lowering to Clifford + Z rotations.
    pub lowered: usize,
    /// Two-qubit gates among the original gates.
    pub two_qubit: usize,
    /// Toffoli gates among the original gates.
    pub toffolis: usize,
    /// Non-Clifford rotations (after merging half-π multiples into S).
    pub rotations: usize,
    /// Rotations by odd multiples of π/4 (T-like).
    pub t_count: usize,
    /// `d_j` per rotation.
    pub d_prof: Vec<u32>,
    /// Size of the factored component touched by rotation `j` (0 for a
    /// rotation with empty mapped support, i.e. a global phase).
    pub f_prof: Vec<u32>,
    /// Original gate index of rotation `j`.
    pub rot_gate: Vec<u32>,
    /// Final active dimension `d`.
    pub d: usize,
    /// Largest factored component `f` over the whole circuit.
    pub f: usize,
    /// log2 Σ_j 2^{d_j} (compressed state work), log2 Σ_j 2^{f_j}.
    pub log2_work: f64,
    /// `-1` encodes a circuit without rotations (no dense work) in both fields.
    pub log2_work_f: f64,
    /// Cut position used for the entanglement (`AtlasOptions::cut`, default `n / 2`).
    pub cut: usize,
    /// The recorded checkpoints, in gate order.
    pub checkpoints: Vec<Checkpoint>,
    /// max over checkpoints of the skeleton entanglement, and of the bound
    /// `min(cut, n - cut, E + d)`.
    pub e_stab_max: usize,
    /// See `e_stab_max`.
    pub e_bound_max: usize,
    /// Affine upper bound on log2 of the final support size (`None` unless requested).
    pub support: Option<usize>,
    /// Seconds spent profiling.
    pub secs: f64,
}

fn log2sum_pow2(xs: impl Iterator<Item = u32>) -> f64 {
    // stable log2 Σ 2^x
    let v: Vec<u32> = xs.collect();
    let Some(&m) = v.iter().max() else {
        return f64::NEG_INFINITY;
    };
    let s: f64 = v.iter().map(|&x| (x as f64 - m as f64).exp2()).sum();
    m as f64 + s.log2()
}

fn is_odd_quarter_pi(theta: f64) -> bool {
    let k = theta / FRAC_PI_4;
    (k - k.round()).abs() < 1e-12 && (k.round() as i64).rem_euclid(2) == 1
}

/// The atlas profile of a unitary circuit (see the module docs).
pub fn profile(circuit: &Circuit, opts: &AtlasOptions) -> Result<AtlasProfile, SimError> {
    let t0 = Instant::now();
    let n = circuit.num_qubits;
    let mut tab = HeisenbergTableau::new(n);
    let w = tab.words();
    let mut skel = SkelTab::new(n);
    let mut rank = Rank::new(n, w);
    let mut dsu = Dsu::new(n.max(1));
    let cut = opts.cut.unwrap_or(n / 2).min(n);
    let gates: Vec<Gate> = circuit
        .ops
        .iter()
        .map(|op| match op {
            Op::Gate(g) => Ok(*g),
            _ => Err(SimError::MeasurementNotSupported {
                backend: "magic_atlas",
                op_index: 0,
            }),
        })
        .collect::<Result<_, _>>()?;
    let ng = gates.len();
    let every = if opts.checkpoints == 0 {
        usize::MAX
    } else {
        ng.div_ceil(opts.checkpoints).max(1)
    };
    let mut p = AtlasProfile {
        n,
        gates: ng,
        cut,
        ..Default::default()
    };
    let mut fmax = 0usize;
    for (gi, g) in gates.iter().enumerate() {
        check_gate(g, n)?;
        if g.arity() == 2 {
            p.two_qubit += 1;
        }
        if matches!(g, Gate::Ccx(..)) {
            p.toffolis += 1;
        }
        for g in g.decompose_to_clifford_rz() {
            p.lowered += 1;
            if g.is_clifford() {
                tab.apply_clifford(&g);
                skel.apply(&g);
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
                    skel.apply(&Gate::S(a));
                }
                continue;
            }
            p.rotations += 1;
            if is_odd_quarter_pi(theta) {
                p.t_count += 1;
            }
            let mut z = vec![0u64; 2 * w];
            z[w + a / 64] |= 1 << (a % 64);
            let (_, q) = tab.map(&z);
            let x2 = rank.push(&q[..w]);
            let z2 = rank.zmap(&q[w..]);
            // support in the V frame, restricted to active coordinates
            let mut root = None;
            for c in bits(&x2).chain(bits(&z2)) {
                root = Some(match root {
                    None => dsu.find(c),
                    Some(r) => dsu.union(r, c),
                });
            }
            let fs = match root {
                Some(r) => {
                    let r = dsu.find(r);
                    dsu.size[r] as usize
                }
                None => 0,
            };
            fmax = fmax.max(fs);
            p.d_prof.push(rank.d as u32);
            p.f_prof.push(fs as u32);
            p.rot_gate.push(gi as u32);
        }
        let last = gi + 1 == ng;
        if (gi + 1) % every == 0 || last {
            let e = if opts.entanglement {
                skel.entanglement(cut)
            } else {
                0
            };
            p.checkpoints.push(Checkpoint {
                gate: gi,
                rotations: p.rotations,
                t_count: p.t_count,
                d: rank.d,
                f: fmax,
                e_stab: e,
            });
        }
    }
    p.d = rank.d;
    p.f = fmax;
    // -1 encodes "no rotation" (Clifford circuit: zero dense work)
    p.log2_work = log2sum_pow2(p.d_prof.iter().copied()).max(-1.0);
    p.log2_work_f = log2sum_pow2(p.f_prof.iter().copied().filter(|&x| x > 0)).max(-1.0);
    let half = cut.min(n - cut);
    p.e_stab_max = p.checkpoints.iter().map(|c| c.e_stab).max().unwrap_or(0);
    p.e_bound_max = p
        .checkpoints
        .iter()
        .map(|c| (c.e_stab + c.d).min(half))
        .max()
        .unwrap_or(0);
    if opts.support {
        p.support = Some(crate::simulability::support_bound(n, &gates));
    }
    p.secs = t0.elapsed().as_secs_f64();
    Ok(p)
}

// ---------------------------------------------------------------------------
// Factored compressed state.

/// One tensor factor of the active register: its coordinates (bit `i` of
/// the local index is coordinate `coords[i]`) and amplitudes.
#[derive(Clone, Debug)]
struct Factor {
    coords: Vec<usize>,
    amp: Vec<C64>,
}

/// Statistics of a [`FactoredState`] run.
#[derive(Clone, Debug, Default)]
pub struct FactoredStats {
    /// Rotations applied.
    pub rotations: usize,
    /// Active dimension `d` at the end of compilation.
    pub d: usize,
    /// Largest factor (qubits).
    pub f: usize,
    /// Live factors at the end of compilation.
    pub factors: usize,
    /// Σ_j 2^{|factor_j|} amplitude updates (+ merge copies).
    pub element_ops: u64,
    /// Seconds compiling, evolution excluded.
    pub compile_secs: f64,
    /// Seconds spent evolving the factors.
    pub evolve_secs: f64,
    /// Recycling: factors found in a stabilizer state and absorbed into the
    /// Clifford frame, and the seconds spent testing/absorbing.
    pub absorbed: usize,
    /// See `absorbed`.
    pub recycle_secs: f64,
    /// Free-coordinate compactions (CNOT networks among |0> coordinates
    /// absorbed into the frame).
    pub compactions: usize,
    /// Per original gate (recycling runs): (Σ live factor sizes, largest
    /// live factor) after the gate. `Σ live` bounds the stabilizer nullity.
    pub live: Vec<(u32, u32)>,
}

/// The exact state of a Clifford + Z-rotation circuit on `|0^n>` as
/// `C V† (|φ_1> ⊗ ⋯ ⊗ |φ_r> ⊗ |0^{n-d}>)`: the compressed state of
/// [`crate::engines::adaptive::CompressedState`] with its active register kept as a
/// tensor product of factors that are merged only when a rotation couples
/// them. Exact (no truncation); memory and work are set by the largest
/// factor `f`, not by `d`.
pub struct FactoredState {
    n: usize,
    w: usize,
    tab: HeisenbergTableau,
    rank: Rank,
    /// coordinate -> factor index (usize::MAX if inactive)
    owner: Vec<usize>,
    factors: Vec<Option<Factor>>,
    /// Clifford `K` on the active coordinates absorbed by recycling
    /// (`ψ = C V† K (⊗ factors ⊗ |0>)`), as images `K† P K`; `None` = identity.
    ktab: Option<HeisenbergTableau>,
    /// Factor ids touched since the last recycling check.
    touched: Vec<usize>,
    live_sum: usize,
    /// Run statistics.
    pub stats: FactoredStats,
}

/// `(neg, x', z')` of the Hermitian string `i^{|x∧z|} X^x Z^z` in the V
/// frame, z' restricted to the active coordinates. The sign accounts for
/// re-normalising the Hermitian phase (the CNOT network itself adds none).
fn frame_map(rank: &Rank, x: &[u64], z: &[u64], x2: Vec<u64>) -> (bool, Vec<u64>, Vec<u64>) {
    let z2 = rank.zmap(z);
    let a: u32 = x.iter().zip(z).map(|(p, q)| (p & q).count_ones()).sum();
    let b: u32 = x2.iter().zip(&z2).map(|(p, q)| (p & q).count_ones()).sum();
    let diff = a as i64 - b as i64;
    debug_assert!(diff.rem_euclid(2) == 0);
    (diff.rem_euclid(4) == 2, x2, z2)
}

impl FactoredState {
    /// Simulates `circuit` on `|0^n>`. Fails with `TooLarge` if a factor
    /// would exceed `max_factor` qubits (at most 34).
    pub fn new(circuit: &Circuit, max_factor: usize) -> Result<Self, SimError> {
        Self::with_recycling(circuit, max_factor, 0)
    }

    /// As [`FactoredState::new`], and after every original gate each factor
    /// touched by it with at most `max_check` qubits is tested for being an
    /// exact stabilizer state `K|0^k>`; if so `K` is absorbed into the
    /// Clifford frame and its coordinates return to `|0>` ("magic
    /// recycling"; `max_check = 0` disables it). Exact: the test and the
    /// synthesised `K` are verified to 1e-9 per amplitude.
    pub fn with_recycling(
        circuit: &Circuit,
        max_factor: usize,
        max_check: usize,
    ) -> Result<Self, SimError> {
        let t0 = Instant::now();
        let n = circuit.num_qubits;
        let mut tab = HeisenbergTableau::new(n);
        let w = tab.words();
        let mut rank = Rank::new(n, w);
        let mut st = FactoredState {
            n,
            w,
            tab: HeisenbergTableau::new(n),
            rank: Rank::new(n, w),
            owner: vec![usize::MAX; n],
            factors: Vec::new(),
            ktab: None,
            touched: Vec::new(),
            live_sum: 0,
            stats: FactoredStats::default(),
        };
        let max_factor = max_factor.min(34);
        let mut evolve = 0.0f64;
        for op in &circuit.ops {
            let g = match op {
                Op::Gate(g) => g,
                _ => {
                    return Err(SimError::MeasurementNotSupported {
                        backend: "factored",
                        op_index: 0,
                    })
                }
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
                let mut z = vec![0u64; 2 * w];
                z[w + a / 64] |= 1 << (a % 64);
                let (neg, q) = tab.map(&z);
                let theta = if neg { -theta } else { theta };
                let x2 = rank.push(&q[..w]);
                let (neg2, x2, z2) = frame_map(&rank, &q[..w], &q[w..], x2);
                let theta = if neg2 { -theta } else { theta };
                let t1 = Instant::now();
                st.rotate(&x2, &z2, theta, max_factor, &rank)?;
                evolve += t1.elapsed().as_secs_f64();
            }
            if max_check > 0 {
                let t1 = Instant::now();
                st.recycle(max_check);
                st.stats.recycle_secs += t1.elapsed().as_secs_f64();
                let mx = st
                    .factors
                    .iter()
                    .flatten()
                    .map(|f| f.coords.len())
                    .max()
                    .unwrap_or(0);
                st.stats.live.push((st.live_sum as u32, mx as u32));
            }
        }
        st.tab = tab;
        st.rank = rank;
        st.stats.d = st.rank.d;
        st.stats.factors = st.factors.iter().filter(|f| f.is_some()).count();
        st.stats.evolve_secs = evolve;
        st.stats.compile_secs = t0.elapsed().as_secs_f64() - evolve;
        Ok(st)
    }

    fn rotate(
        &mut self,
        x2: &[u64],
        z2: &[u64],
        theta: f64,
        max_factor: usize,
        rank: &Rank,
    ) -> Result<(), SimError> {
        self.stats.rotations += 1;
        let mut mapped = self.ktab.as_ref().map(|kt| {
            let mut p = x2.to_vec();
            p.extend_from_slice(z2);
            kt.map(&p)
        });
        // Free-coordinate compaction: coordinates holding |0> (recycled) are
        // invariant under any CNOT network W among them, so K -> K W is
        // exact. Choosing W = Π_t CNOT(c*, t) over the free x-bits leaves a
        // single free x-bit (c*): at most one coordinate is activated.
        if let Some((_, im)) = &mapped {
            let fx: Vec<usize> = bits(&im[..self.w])
                .filter(|&c| self.owner[c] == usize::MAX)
                .collect();
            if fx.len() >= 2 {
                let w_gates: Vec<Gate> = fx[1..].iter().map(|&t| Gate::Cnot(fx[0], t)).collect();
                let kt = self.ktab.as_mut().expect("mapped implies ktab");
                kt.post_conjugate(&w_gates);
                self.stats.compactions += 1;
                let mut p = x2.to_vec();
                p.extend_from_slice(z2);
                mapped = Some(kt.map(&p));
            }
        }
        let (theta, x2, z2) = match &mapped {
            Some((neg, im)) => (
                if *neg { -theta } else { theta },
                &im[..self.w],
                &im[self.w..],
            ),
            None => (theta, x2, z2),
        };
        // A z bit on a coordinate holding |0> (inactive or recycled) acts as
        // +1 and is dropped; x bits activate their coordinate.
        let owner = &self.owner;
        let coords: Vec<usize> = bits(x2)
            .chain(bits(z2).filter(|&c| owner[c] != usize::MAX))
            .collect();
        if coords.is_empty() {
            return Ok(()); // global phase
        }
        // activate new coordinates as fresh |0> factors
        let _ = rank;
        for &c in &coords {
            if self.owner[c] == usize::MAX {
                self.owner[c] = self.factors.len();
                self.live_sum += 1;
                self.factors.push(Some(Factor {
                    coords: vec![c],
                    amp: vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
                }));
            }
        }
        let mut ids: Vec<usize> = coords.iter().map(|&c| self.owner[c]).collect();
        ids.sort_unstable();
        ids.dedup();
        let total: usize = ids
            .iter()
            .map(|&i| self.factors[i].as_ref().unwrap().coords.len())
            .sum();
        if total > max_factor {
            return Err(SimError::TooLarge {
                what: "factored compressed state",
                bytes: (1u128 << total.min(120)) * 16,
                limit: (1u128 << max_factor) * 16,
            });
        }
        // merge into ids[0]
        let mut fac = self.factors[ids[0]].take().unwrap();
        for &i in &ids[1..] {
            let o = self.factors[i].take().unwrap();
            let mut amp = vec![C64::new(0.0, 0.0); fac.amp.len() * o.amp.len()];
            let lo = fac.amp.len();
            for (j, &b) in o.amp.iter().enumerate() {
                if b == C64::new(0.0, 0.0) {
                    continue;
                }
                for (k, &a) in fac.amp.iter().enumerate() {
                    amp[j * lo + k] = a * b;
                }
            }
            self.stats.element_ops += amp.len() as u64;
            for &c in &o.coords {
                self.owner[c] = ids[0];
            }
            fac.coords.extend_from_slice(&o.coords);
            fac.amp = amp;
        }
        // local masks
        let mut lx = 0u64;
        let mut lz = 0u64;
        for (i, &c) in fac.coords.iter().enumerate() {
            if x2[c / 64] >> (c % 64) & 1 == 1 {
                lx |= 1 << i;
            }
            if z2[c / 64] >> (c % 64) & 1 == 1 {
                lz |= 1 << i;
            }
        }
        rotate_dense(&mut fac.amp, lx, lz, theta);
        self.stats.element_ops += fac.amp.len() as u64;
        self.stats.f = self.stats.f.max(fac.coords.len());
        self.factors[ids[0]] = Some(fac);
        self.touched.push(ids[0]);
        Ok(())
    }

    /// Tests the factors touched since the last call; absorbs the ones in a
    /// stabilizer state.
    fn recycle(&mut self, max_check: usize) {
        let mut ids = std::mem::take(&mut self.touched);
        ids.sort_unstable();
        ids.dedup();
        for id in ids {
            let Some(fac) = &self.factors[id] else {
                continue;
            };
            let k = fac.coords.len();
            if k > max_check {
                continue;
            }
            let Some(local) = stabilizer_synth(&fac.amp, k) else {
                continue;
            };
            let gates: Vec<Gate> = local
                .iter()
                .map(|g| families::remap(g, &|q| fac.coords[q]))
                .collect();
            let kt = self
                .ktab
                .get_or_insert_with(|| HeisenbergTableau::new(self.n));
            kt.post_conjugate(&gates);
            for &c in &fac.coords {
                self.owner[c] = usize::MAX;
            }
            self.live_sum -= k;
            self.factors[id] = None;
            self.stats.absorbed += 1;
        }
    }

    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// Exact `<ψ|P|ψ>` for a Pauli string given as `(x, z)` bit vectors
    /// (Hermitian `i^{|x∧z|} X^x Z^z`, i.e. `Y` where both are set).
    pub fn pauli_expectation(&self, x: &[bool], z: &[bool]) -> f64 {
        let w = self.w;
        let mut p = vec![0u64; 2 * w];
        for q in 0..self.n {
            if x[q] {
                p[q / 64] |= 1 << (q % 64);
            }
            if z[q] {
                p[w + q / 64] |= 1 << (q % 64);
            }
        }
        let (neg, im) = self.tab.map(&p);
        // x part must lie in the active span
        let mut r = im[..w].to_vec();
        let mut comb = vec![0u64; w];
        for b in bits(&im[..w]).collect::<Vec<_>>() {
            let k = self.rank.piv[b];
            if k != u32::MAX {
                xor_into(&mut r, &self.rank.rows[k as usize]);
                xor_into(&mut comb, &self.rank.combs[k as usize]);
            }
        }
        if r.iter().any(|&v| v != 0) {
            return 0.0;
        }
        let (neg2, x2, z2) = frame_map(&self.rank, &im[..w], &im[w..], comb);
        let (neg3, x2, z2) = match &self.ktab {
            None => (false, x2, z2),
            Some(kt) => {
                let mut p = x2;
                p.extend_from_slice(&z2);
                let (ng, im) = kt.map(&p);
                (ng, im[..w].to_vec(), im[w..].to_vec())
            }
        };
        let mut val = if neg ^ neg2 ^ neg3 { -1.0 } else { 1.0 };
        // group by factor
        let mut per: std::collections::BTreeMap<usize, (u64, u64)> = Default::default();
        for c in bits(&x2).chain(bits(&z2)) {
            let fi = self.owner[c];
            if fi == usize::MAX {
                // inactive or recycled coordinate: it holds |0>, so an x bit
                // gives 0 and a z bit +1.
                if x2[c / 64] >> (c % 64) & 1 == 1 {
                    return 0.0;
                }
                continue;
            }
            let f = self.factors[fi].as_ref().unwrap();
            let i = f.coords.iter().position(|&k| k == c).unwrap();
            let e = per.entry(fi).or_insert((0, 0));
            if x2[c / 64] >> (c % 64) & 1 == 1 {
                e.0 |= 1 << i;
            }
            if z2[c / 64] >> (c % 64) & 1 == 1 {
                e.1 |= 1 << i;
            }
        }
        for (fi, (lx, lz)) in per {
            let a = &self.factors[fi].as_ref().unwrap().amp;
            // <φ| i^{|x∧z|} X^x Z^z |φ> = Σ_y conj(φ(y⊕x)) i^r (-1)^{z·y} φ(y)
            let r = (lx & lz).count_ones();
            let mut s = C64::new(0.0, 0.0);
            for (y, &v) in a.iter().enumerate() {
                let t = a[y ^ lx as usize].conj() * v;
                if (lz & y as u64).count_ones() % 2 == 1 {
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
            val *= (ph * s).re;
        }
        val
    }

    /// The active register as one dense vector over coordinates `0..d`
    /// (Kronecker product of the factors; tests, `d <= 24`). Equal to
    /// `CompressedState::active_amplitudes` (same frame) up to rounding.
    pub fn dense_active(&self) -> Vec<C64> {
        let d = self.rank.d;
        assert!(d <= 24);
        assert!(
            self.ktab.is_none(),
            "dense_active: not available after recycling"
        );
        let mut out = vec![C64::new(0.0, 0.0); 1 << d];
        let fs: Vec<&Factor> = self.factors.iter().flatten().collect();
        for (y, o) in out.iter_mut().enumerate() {
            let mut v = C64::new(1.0, 0.0);
            for f in &fs {
                let mut li = 0usize;
                for (i, &c) in f.coords.iter().enumerate() {
                    li |= (y >> c & 1) << i;
                }
                v *= f.amp[li];
            }
            // coordinates in no factor are |0>
            let covered: usize = fs
                .iter()
                .flat_map(|f| f.coords.iter())
                .fold(0, |m, &c| m | 1 << c);
            if y & !covered != 0 {
                v = C64::new(0.0, 0.0);
            }
            *o = v;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Stabilizer-state recognition and synthesis on a small dense register.

/// If `amp` (`2^k` amplitudes, normalised) is a stabilizer state up to a
/// global phase, returns a Clifford circuit `K` (local qubits `0..k`, time
/// order) with `K|0^k> ∝ amp`; otherwise `None`. Uses the affine-support /
/// quadratic-phase form (Dehaene & De Moor 2003): support `y0 ⊕ span(b)`,
/// uniform modulus, phase `i^{Σ λ_i c_i} (-1)^{Σ q_ij c_i c_j}`. The
/// synthesised circuit is replayed and checked against `amp`.
pub fn stabilizer_synth(amp: &[C64], k: usize) -> Option<Vec<Gate>> {
    let len = amp.len();
    debug_assert_eq!(len, 1 << k);
    let sup: Vec<usize> = (0..len).filter(|&y| amp[y].norm_sqr() > 1e-10).collect();
    let m = sup.len();
    if m == 0 || !m.is_power_of_two() {
        return None;
    }
    let r = m.trailing_zeros() as usize;
    let p0 = 1.0 / m as f64;
    if sup.iter().any(|&y| (amp[y].norm_sqr() - p0).abs() > 1e-9) {
        return None;
    }
    let y0 = sup[0];
    // RREF basis of {y ^ y0}
    let mut rows: Vec<(usize, usize)> = Vec::new(); // (pivot, row)
    for &y in &sup {
        let mut t = y ^ y0;
        for &(p, b) in &rows {
            if t >> p & 1 == 1 {
                t ^= b;
            }
        }
        if t != 0 {
            let p = t.trailing_zeros() as usize;
            for rw in rows.iter_mut() {
                if rw.1 >> p & 1 == 1 {
                    rw.1 ^= t;
                }
            }
            rows.push((p, t));
            if rows.len() > r {
                return None;
            }
        }
    }
    if rows.len() != r {
        return None;
    }
    let coord = |y: usize| -> usize {
        let t = y ^ y0;
        let mut c = 0;
        for (i, &(p, _)) in rows.iter().enumerate() {
            c |= (t >> p & 1) << i;
        }
        c
    };
    let refa = amp[y0];
    let g = |y: usize| amp[y] / refa;
    let quarter = |z: C64| -> Option<u32> {
        let l = (z.arg() / FRAC_PI_2).round().rem_euclid(4.0) as u32;
        let want = [
            C64::new(1.0, 0.0),
            C64::new(0.0, 1.0),
            C64::new(-1.0, 0.0),
            C64::new(0.0, -1.0),
        ][l as usize];
        ((z - want).norm() < 1e-7).then_some(l)
    };
    let mut lam = vec![0u32; r];
    for i in 0..r {
        lam[i] = quarter(g(y0 ^ rows[i].1))?;
    }
    let mut q = vec![vec![false; r]; r];
    for i in 0..r {
        for j in i + 1..r {
            let z = g(y0 ^ rows[i].1 ^ rows[j].1) / (g(y0 ^ rows[i].1) * g(y0 ^ rows[j].1));
            let l = quarter(z)?;
            if l % 2 == 1 {
                return None;
            }
            q[i][j] = l == 2;
        }
    }
    // gates: H, S^λ, CZ, CNOT network M, X^{y0}
    let mut gates = Vec::new();
    for i in 0..r {
        gates.push(Gate::H(i));
    }
    for (i, &l) in lam.iter().enumerate() {
        for _ in 0..l {
            gates.push(Gate::S(i));
        }
    }
    for i in 0..r {
        for j in i + 1..r {
            if q[i][j] {
                gates.push(Gate::Cz(i, j));
            }
        }
    }
    // M: column i = rows[i].1 for i < r, then unit vectors at non-pivots
    let pivots: Vec<usize> = rows.iter().map(|x| x.0).collect();
    let mut cols: Vec<usize> = rows.iter().map(|x| x.1).collect();
    for b in 0..k {
        if !pivots.contains(&b) {
            cols.push(1 << b);
        }
    }
    // row-major matrix: mat[row] bit col
    let mut mat = vec![0usize; k];
    for (c, &v) in cols.iter().enumerate() {
        for (rw, m) in mat.iter_mut().enumerate() {
            if v >> rw & 1 == 1 {
                *m |= 1 << c;
            }
        }
    }
    let mut ops: Vec<(usize, usize)> = Vec::new(); // (control, target): row_t ^= row_c
    for col in 0..k {
        let pr = (col..k).find(|&rw| mat[rw] >> col & 1 == 1)?;
        if pr != col {
            mat[col] ^= mat[pr];
            ops.push((pr, col));
        }
        for rw in 0..k {
            if rw != col && mat[rw] >> col & 1 == 1 {
                mat[rw] ^= mat[col];
                ops.push((col, rw));
            }
        }
    }
    debug_assert!((0..k).all(|i| mat[i] == 1 << i));
    for &(c, t) in ops.iter().rev() {
        gates.push(Gate::Cnot(c, t));
    }
    for b in 0..k {
        if y0 >> b & 1 == 1 {
            gates.push(Gate::X(b));
        }
    }
    // replay and check (global phase aligned on y0)
    let mut v = vec![C64::new(0.0, 0.0); len];
    v[0] = C64::new(1.0, 0.0);
    let r2 = std::f64::consts::FRAC_1_SQRT_2;
    for gt in &gates {
        match *gt {
            Gate::H(a) => {
                for y in 0..len {
                    if y >> a & 1 == 0 {
                        let (l, h) = (v[y], v[y | 1 << a]);
                        v[y] = (l + h) * r2;
                        v[y | 1 << a] = (l - h) * r2;
                    }
                }
            }
            Gate::S(a) => {
                for (y, x) in v.iter_mut().enumerate() {
                    if y >> a & 1 == 1 {
                        *x *= C64::new(0.0, 1.0);
                    }
                }
            }
            Gate::Cz(a, b) => {
                for (y, x) in v.iter_mut().enumerate() {
                    if y >> a & 1 == 1 && y >> b & 1 == 1 {
                        *x = -*x;
                    }
                }
            }
            Gate::Cnot(c, t) => {
                for y in 0..len {
                    if y >> c & 1 == 1 && y >> t & 1 == 0 {
                        v.swap(y, y | 1 << t);
                    }
                }
            }
            Gate::X(a) => {
                for y in 0..len {
                    if y >> a & 1 == 0 {
                        v.swap(y, y | 1 << a);
                    }
                }
            }
            _ => unreachable!(),
        }
    }
    let ph = refa / v[y0];
    if ph.norm() < 1e-12 {
        return None;
    }
    let ph = ph / ph.norm();
    if (0..len).any(|y| (v[y] * ph - amp[y]).norm() > 1e-9) {
        return None;
    }
    let _ = coord;
    Some(gates)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_matches_adaptive_profile() {
        for spec in ["rct:n=12,L=6,t=14", "qft:n=9,in=graph", "qaoa:n=10,p=2"] {
            let c = families::build(spec, 3).unwrap();
            let p = profile(&c, &AtlasOptions::default()).unwrap();
            let want = crate::engines::adaptive::active_dimension_profile(&c).unwrap();
            let got: Vec<usize> = p.d_prof.iter().map(|&x| x as usize).collect();
            assert_eq!(got, want, "{spec}");
        }
    }
}

// ---------------------------------------------------------------------------
// Ground-truth state magic at small n (all 4^n Pauli expectations).

/// State-magic measures of a pure state computed from its full Pauli
/// spectrum: for every `x` the `2^n` values `|<ψ|X^x Z^z|ψ>|` come from one
/// Walsh–Hadamard transform of `conj(ψ(y⊕x)) ψ(y)`, so the cost is
/// `O(4^n · n)` (n ≤ 13 or so).
#[derive(Clone, Copy, Debug, Default)]
pub struct StateMagic {
    /// Stabilizer nullity `ν = n − log2 |{P : |<P>| = 1}|` (Beverland et
    /// al. 2020): 0 iff stabilizer state; `ν ≤ d` always.
    pub nullity: f64,
    /// Stabilizer 2-Rényi entropy `M2 = −log2(Σ_P <P>^4 / 2^n)` (Leone,
    /// Oliviero, Hamma 2022).
    pub m2: f64,
}

/// [`StateMagic`] of the normalised state with amplitudes `amps`.
///  Panics unless the length is a power of two.
pub fn state_magic(amps: &[C64]) -> StateMagic {
    let len = amps.len();
    let n = len.trailing_zeros() as usize;
    assert_eq!(1usize << n, len);
    let mut stab = 0usize;
    let mut s4 = 0.0f64;
    let mut u = vec![C64::new(0.0, 0.0); len];
    for x in 0..len {
        for (y, v) in u.iter_mut().enumerate() {
            *v = amps[y ^ x].conj() * amps[y];
        }
        // in-place FWHT
        let mut h = 1;
        while h < len {
            for i in (0..len).step_by(2 * h) {
                for j in i..i + h {
                    let (a, b) = (u[j], u[j + h]);
                    u[j] = a + b;
                    u[j + h] = a - b;
                }
            }
            h *= 2;
        }
        for v in &u {
            let m = v.norm();
            if m > 1.0 - 1e-9 {
                stab += 1;
            }
            let m2 = m * m;
            s4 += m2 * m2;
        }
    }
    StateMagic {
        nullity: n as f64 - (stab as f64).log2(),
        m2: -(s4 / len as f64).log2(),
    }
}
