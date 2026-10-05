//! Sparse Pauli dynamics (SPD) for kicked-Ising Trotter circuits on
//! heavy-hex lattices: a Heisenberg-picture, truncated Pauli-propagation
//! engine built to reproduce IBM's 127-qubit "utility" experiment
//! (Kim et al., Nature 618, 500 (2023)) classically.
//!
//! # The circuit
//!
//! ```text
//!   U = ( Π_<j,k> exp(+i π/4 Z_j Z_k) · Π_j exp(-i θ_h X_j / 2) )^T     (|0…0> input)
//! ```
//!
//! i.e. every Trotter step is an `RX(θ_h)` kick on every qubit followed by
//! `RZZ(-π/2)` on every edge of the coupling graph. All `RZZ` gates commute,
//! so the order of the three edge-colour layers used on hardware is
//! irrelevant. Optionally a final extra `RX(θ_h)` layer (the weight-17
//! observable of Kim et al. Fig. 4a).
//!
//! # The algorithm
//!
//! A Pauli string is stored as `(x, z)` bit vectors (Hermitian form
//! `i^{|x∧z|} X^x Z^z`, `(1,1)` = Y) with a real coefficient. The observable
//! is pushed backwards through the circuit, one Trotter step at a time:
//!
//! * **ZZ layer (Clifford, one term → one term).** `RZZ(-π/2)` maps an
//!   anticommuting `P` to `i P Z_a Z_b`. `x` never changes, and `P`
//!   anticommutes with `Z_a Z_b` iff `x_a ≠ x_b`, so the whole layer is
//!   `z ← z ⊕ s` with `s_q = deg(q)·x_q ⊕ ⊕_{r~q} x_r` and overall phase
//!   `i^{k + |x∧z| − |x∧z'|}` (`k` = number of cut edges). Cost
//!   `O(|x|)` per term, independent of lattice size.
//! * **RX layer (branching).** `RX(θ)` maps `Z → cos θ Z + sin θ Y` and
//!   `Y → cos θ Y − sin θ Z` (X and I unchanged), so a term with `m` sites
//!   carrying Z or Y has up to `2^m` children. They are enumerated depth
//!   first; because every factor has modulus ≤ 1 the branch coefficient is
//!   monotone along the tree, so a subtree is cut as soon as its coefficient
//!   falls below the threshold `δ` (every child `≥ δ` is still produced).
//!   Children of all parents are then merged in a sharded hash map and
//!   merged coefficients below `δ` are dropped.
//! * **First RX layer (exact, no branching).** `<0| RX† P RX |0>` factorises:
//!   `I → 1, Z → cos θ, Y → −sin θ, X → 0`, so the last backward layer is
//!   evaluated in closed form term by term (no merge pass).
//! * **Light cone.** The support grows by one graph distance per step, so
//!   only qubits within distance `T` of the observable are kept. They are
//!   relabelled in BFS order so the key width `W` (64-bit words) follows the
//!   light cone, not the device (a 1121-qubit device at 20 steps still uses
//!   the ~370-qubit cone of a bulk qubit).
//! * **Truncation.** Coefficient threshold `δ` (as in Begušić & Chan) and an
//!   optional Pauli-weight cap. Every discarded piece is accounted: the
//!   squared norm removed (`discarded_l2sq`, exact per cut since the pruned
//!   leaves are orthogonal) and the l1 mass (a rigorous but loose bound on
//!   the expectation-value error, `|Δ<O>| ≤ Σ|c_discarded|·‖P‖`).
//! * **Noise (optional).** Single-qubit depolarizing channel of strength `p`
//!   on every qubit after each ZZ layer. Its adjoint multiplies a string by
//!   `(1 − 4p/3)^{weight}`, applied exactly.
//!
//! `δ = 0` with no weight cap is exact (differential-tested against the
//! dense state vector and the exact Clifford+Rz Pauli-path engine).

// Bit-array loops over the W words of a key read best indexed.
#![allow(clippy::needless_range_loop)]

use crate::circuit::Circuit;
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::f64::consts::FRAC_PI_2;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Lattices

/// An undirected coupling graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lattice {
    /// Number of vertices (qubits), numbered `0..n`.
    pub n: usize,
    /// Edges `(a, b)` with `a < b`, sorted, no duplicates.
    pub edges: Vec<(usize, usize)>,
    /// Sorted neighbour list of every vertex.
    pub adj: Vec<Vec<usize>>,
}

impl Lattice {
    /// Builds the graph on `n` vertices from `edges` (either orientation;
    /// duplicates are merged). Panics on a self-loop or an out-of-range vertex.
    pub fn from_edges(n: usize, edges: &[(usize, usize)]) -> Self {
        let mut e: Vec<(usize, usize)> = edges
            .iter()
            .map(|&(a, b)| {
                assert!(a != b && a < n && b < n, "bad edge ({a},{b})");
                (a.min(b), a.max(b))
            })
            .collect();
        e.sort_unstable();
        e.dedup();
        let mut adj = vec![Vec::new(); n];
        for &(a, b) in &e {
            adj[a].push(b);
            adj[b].push(a);
        }
        for l in &mut adj {
            l.sort_unstable();
        }
        Lattice { n, edges: e, adj }
    }

    /// IBM heavy-hex family: `rows` rows of `row_len` qubits (the first row
    /// lacks its last qubit, the last row its first), joined by bridge
    /// qubits at row offsets 0, 4, 8, … (even gaps) or 2, 6, 10, … (odd
    /// gaps). Numbered row by row with each gap's bridges after its upper
    /// row. `(7, 15)` is Eagle (127 qubits, the ibm_kyiv / ibm_washington
    /// numbering), `(13, 27)` Osprey-sized (433), `(21, 43)` Condor-sized
    /// (1121).
    pub fn ibm_heavy_hex(rows: usize, row_len: usize) -> Self {
        assert!(rows >= 2 && row_len >= 3 && row_len % 4 == 3);
        assert!(
            rows % 2 == 1,
            "the last gap must be odd so the last row can lack offset 0"
        );
        let mut idx = vec![vec![usize::MAX; row_len]; rows];
        let mut edges = Vec::new();
        let mut next = 0usize;
        let row_ids = |r: usize, next: &mut usize, idx: &mut Vec<Vec<usize>>| {
            for (o, slot) in idx[r].iter_mut().enumerate() {
                let present = !((r == 0 && o == row_len - 1) || (r == rows - 1 && o == 0));
                if present {
                    *slot = *next;
                    *next += 1;
                }
            }
        };
        row_ids(0, &mut next, &mut idx);
        for g in 0..rows - 1 {
            // Bridges of gap g (between rows g and g+1) are numbered before row g+1.
            let start = if g % 2 == 0 { 0 } else { 2 };
            let mut bridges = Vec::new();
            for o in (start..row_len).step_by(4) {
                bridges.push((o, next));
                next += 1;
            }
            row_ids(g + 1, &mut next, &mut idx);
            for (o, b) in bridges {
                let (u, d) = (idx[g][o], idx[g + 1][o]);
                assert!(
                    u != usize::MAX && d != usize::MAX,
                    "bridge to missing qubit"
                );
                edges.push((u, b));
                edges.push((b, d));
            }
        }
        for row in &idx {
            let ids: Vec<usize> = row.iter().copied().filter(|&q| q != usize::MAX).collect();
            for w in ids.windows(2) {
                edges.push((w[0], w[1]));
            }
        }
        Lattice::from_edges(next, &edges)
    }

    /// 127-qubit Eagle (ibm_kyiv / ibm_washington numbering).
    pub fn eagle127() -> Self {
        Self::ibm_heavy_hex(7, 15)
    }
    /// 433-qubit Osprey-sized heavy hex.
    pub fn osprey433() -> Self {
        Self::ibm_heavy_hex(13, 27)
    }
    /// 1121-qubit Condor-sized heavy hex.
    pub fn condor1121() -> Self {
        Self::ibm_heavy_hex(21, 43)
    }

    /// Graph distance from the nearest source (`usize::MAX` if unreachable).
    pub fn distances_from(&self, srcs: &[usize]) -> Vec<usize> {
        let mut d = vec![usize::MAX; self.n];
        let mut q = VecDeque::new();
        for &s in srcs {
            if d[s] != 0 {
                d[s] = 0;
                q.push_back(s);
            }
        }
        while let Some(u) = q.pop_front() {
            for &v in &self.adj[u] {
                if d[v] == usize::MAX {
                    d[v] = d[u] + 1;
                    q.push_back(v);
                }
            }
        }
        d
    }

    /// A connected patch: the first `k` qubits in BFS order from `root`
    /// (induced subgraph, relabelled 0..k). Used for small exact checks.
    pub fn bfs_patch(&self, root: usize, k: usize) -> Lattice {
        let order = self.bfs_order(&[root]);
        let keep: Vec<usize> = order.into_iter().take(k).collect();
        self.induced(&keep).0
    }

    /// Vertices in BFS order from `srcs` (ties by index), reachable only.
    pub fn bfs_order(&self, srcs: &[usize]) -> Vec<usize> {
        let mut seen = vec![false; self.n];
        let mut out = Vec::new();
        let mut q = VecDeque::new();
        let mut s: Vec<usize> = srcs.to_vec();
        s.sort_unstable();
        s.dedup();
        for v in s {
            seen[v] = true;
            q.push_back(v);
        }
        while let Some(u) = q.pop_front() {
            out.push(u);
            for &v in &self.adj[u] {
                if !seen[v] {
                    seen[v] = true;
                    q.push_back(v);
                }
            }
        }
        out
    }

    /// Induced subgraph on `keep` (new label = position in `keep`); also
    /// returns old → new (`usize::MAX` for dropped vertices).
    pub fn induced(&self, keep: &[usize]) -> (Lattice, Vec<usize>) {
        let mut map = vec![usize::MAX; self.n];
        for (i, &v) in keep.iter().enumerate() {
            map[v] = i;
        }
        let e: Vec<(usize, usize)> = self
            .edges
            .iter()
            .filter(|&&(a, b)| map[a] != usize::MAX && map[b] != usize::MAX)
            .map(|&(a, b)| (map[a], map[b]))
            .collect();
        (Lattice::from_edges(keep.len(), &e), map)
    }
}

// ---------------------------------------------------------------------------
// Model and observables

/// The kicked-Ising Trotter circuit of Kim et al.
#[derive(Clone, Debug)]
pub struct KickedIsing {
    /// Coupling graph: one `RZZ` per edge per step.
    pub lattice: Lattice,
    /// Number of Trotter steps `T`.
    pub steps: usize,
    /// Kick angle `θ_h` of the `RX` layer (radians).
    pub theta_h: f64,
    /// One extra `RX(θ_h)` layer after the last step (Kim et al. Fig. 4a).
    pub final_rx: bool,
}

impl KickedIsing {
    /// `steps` Trotter steps on `lattice` with kick angle `theta_h`, without
    /// the final extra `RX` layer.
    pub fn new(lattice: Lattice, steps: usize, theta_h: f64) -> Self {
        KickedIsing {
            lattice,
            steps,
            theta_h,
            final_rx: false,
        }
    }

    /// The gate-level circuit (RZZ(θ) = CNOT · Rz_b(θ) · CNOT), for
    /// validation against the dense state vector and other engines.
    pub fn to_circuit(&self) -> Circuit {
        let n = self.lattice.n;
        let mut c = Circuit::new(n);
        for _ in 0..self.steps {
            for q in 0..n {
                c.rx(q, self.theta_h);
            }
            for &(a, b) in &self.lattice.edges {
                c.cnot(a, b);
                c.rz(b, -FRAC_PI_2);
                c.cnot(a, b);
            }
        }
        if self.final_rx {
            for q in 0..n {
                c.rx(q, self.theta_h);
            }
        }
        c
    }
}

/// A real linear combination of sparse Pauli strings.
#[derive(Clone, Debug, PartialEq)]
pub struct PauliObs {
    /// `(string, coefficient)`; each string is a list of `(qubit, 'X'|'Y'|'Z')`.
    pub terms: Vec<(Vec<(usize, char)>, f64)>,
}

impl PauliObs {
    /// A single Pauli string with coefficient 1.
    pub fn single(s: Vec<(usize, char)>) -> Self {
        PauliObs {
            terms: vec![(s, 1.0)],
        }
    }
    /// `Z_q`.
    pub fn z(q: usize) -> Self {
        Self::single(vec![(q, 'Z')])
    }
    /// Magnetisation `M_z = Σ_q Z_q / n` over `qubits`.
    pub fn magnetisation(qubits: &[usize]) -> Self {
        let w = 1.0 / qubits.len() as f64;
        PauliObs {
            terms: qubits.iter().map(|&q| (vec![(q, 'Z')], w)).collect(),
        }
    }
    /// Parse `"X37 X41 Y75 Z38"` (tokens `<P><qubit>`).
    pub fn parse(s: &str) -> Self {
        let mut v = Vec::new();
        for tok in s.split_whitespace() {
            let mut ch = tok.chars();
            let p = ch.next().unwrap().to_ascii_uppercase();
            assert!("XYZ".contains(p), "bad token {tok}");
            let q: usize = ch.as_str().parse().expect("qubit index");
            v.push((q, p));
        }
        Self::single(v)
    }
    /// Parse a dense string, character `i` = qubit `i` (the Kim et al. data
    /// files use this order).
    pub fn from_dense(s: &str) -> Self {
        let v = s
            .trim()
            .chars()
            .enumerate()
            .filter(|(_, c)| *c != 'I')
            .collect();
        Self::single(v)
    }
    /// Sorted, deduplicated qubits acted on by any term.
    pub fn support(&self) -> Vec<usize> {
        let mut s: Vec<usize> = self
            .terms
            .iter()
            .flat_map(|(t, _)| t.iter().map(|&(q, _)| q))
            .collect();
        s.sort_unstable();
        s.dedup();
        s
    }
}

// ---------------------------------------------------------------------------
// Options and results

/// Truncation, noise and resource options for [`simulate`].
#[derive(Clone, Copy, Debug)]
pub struct SpdOptions {
    /// Coefficient threshold δ (0 = exact), applied to merged coefficients.
    pub delta: f64,
    /// Branch threshold = `branch_factor · δ`, applied to individual
    /// children before they are merged (and in streamed layers). `1` prunes
    /// a path as soon as it falls below δ; smaller values keep sub-δ paths
    /// long enough to merge with others into a coefficient ≥ δ (closer to
    /// per-gate truncation, at higher cost).
    pub branch_factor: f64,
    /// Drop strings of Pauli weight above this after each ZZ layer.
    pub max_weight: usize,
    /// Single-qubit depolarizing probability per qubit per step (after ZZ).
    pub depol: f64,
    /// Abort (result flagged `aborted`) when more terms than this are alive.
    pub max_terms: usize,
    /// Restrict to the backward light cone (exact; on by default).
    pub light_cone: bool,
    /// Number of trailing branching layers evaluated by depth-first
    /// streaming instead of merge passes (default 1). The last layer feeds a
    /// linear closed-form evaluation, so streaming it costs the same work as
    /// merging and removes the largest term table from memory. Larger values
    /// trade time (no deduplication) for memory.
    pub stream: usize,
}

impl Default for SpdOptions {
    fn default() -> Self {
        SpdOptions {
            delta: 0.0,
            branch_factor: 1.0,
            max_weight: usize::MAX,
            depol: 0.0,
            max_terms: 50_000_000,
            light_cone: true,
            stream: 1,
        }
    }
}

/// Result of [`simulate`].
#[derive(Clone, Debug, Default)]
pub struct SpdResult {
    /// The (truncated) expectation value `<0|U† O U|0>`; meaningless if
    /// `aborted`.
    pub value: f64,
    /// Qubits kept (the backward light cone, or all qubits without it).
    pub active_qubits: usize,
    /// Width of a Pauli key in 64-bit words (per `x` and `z` part).
    pub words: usize,
    /// Terms after each merged backward layer (the final layer is not merged).
    pub terms_per_layer: Vec<usize>,
    /// Largest number of live terms after any merged layer.
    pub peak_terms: usize,
    /// Terms entering the closed-form evaluation of the first RX layer.
    pub final_terms: usize,
    /// Σ|c| of discarded pieces (subtree l1 mass): rigorous, loose error bound.
    pub discarded_l1: f64,
    /// Squared Frobenius norm removed by truncation.
    pub discarded_l2sq: f64,
    /// Normalised squared Frobenius norm Σc² of the operator entering the
    /// last layer (1 for a unit-norm Pauli and no truncation or noise).
    pub norm2: f64,
    /// Total wall-clock seconds.
    pub seconds: f64,
    /// `max_terms` was exceeded and the run stopped early; only the
    /// discarded-weight fields and the per-layer counts so far are filled in.
    pub aborted: bool,
}

// ---------------------------------------------------------------------------
// Engine

#[derive(Clone, Copy, PartialEq, Eq)]
struct Key<const W: usize> {
    x: [u64; W],
    z: [u64; W],
}

impl<const W: usize> Hash for Key<W> {
    #[inline]
    fn hash<H: Hasher>(&self, h: &mut H) {
        h.write_u64(key_hash(self));
    }
}

#[inline]
fn fmix(mut k: u64) -> u64 {
    k ^= k >> 33;
    k = k.wrapping_mul(0xff51_afd7_ed55_8ccd);
    k ^= k >> 33;
    k = k.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    k ^ (k >> 33)
}

#[inline]
fn key_hash<const W: usize>(k: &Key<W>) -> u64 {
    let mut h = 0x9e37_79b9_7f4a_7c15u64;
    for i in 0..W {
        h = (h.rotate_left(23) ^ k.x[i]).wrapping_mul(0x517c_c1b7_2722_0a95);
        h = (h.rotate_left(23) ^ k.z[i]).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    fmix(h)
}

#[derive(Default)]
struct PassHasher(u64);
impl Hasher for PassHasher {
    #[inline]
    fn write_u64(&mut self, v: u64) {
        self.0 = v;
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!()
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}
type Map<const W: usize> = HashMap<Key<W>, f64, BuildHasherDefault<PassHasher>>;

const SHARDS: usize = 128;
const FLUSH: usize = 512;

#[inline]
fn shard_of(h: u64) -> usize {
    // Different bits from the ones hashbrown uses (low bits for the bucket,
    // top 7 for the tag).
    ((h >> 40) as usize) % SHARDS
}

/// Per-qubit data in the light-cone labelling.
struct Geometry<const W: usize> {
    /// Bits `{r : r ~ q} ∪ {q if deg(q) odd}`: the z flips caused by `x_q`.
    flip: Vec<[u64; W]>,
    /// Neighbour bitmask (for counting cut edges).
    nbr: Vec<[u64; W]>,
    deg: Vec<u32>,
}

struct Ctx<'a, const W: usize> {
    geo: &'a Geometry<W>,
    cos: f64,
    sin: f64,
    delta: f64,
    branch_delta: f64,
    max_weight: usize,
    damp: f64,
}

#[derive(Default, Clone, Copy)]
struct Acc {
    l1: f64,
    l2sq: f64,
}

impl<'a, const W: usize> Ctx<'a, W> {
    /// Noise damping + ZZ layer + weight cap. Returns `None` if the term is
    /// discarded (accounted in `acc`).
    #[inline]
    fn zz(&self, key: &mut Key<W>, c: &mut f64, acc: &mut Acc) -> bool {
        if self.damp != 1.0 {
            let mut w = 0u32;
            for i in 0..W {
                w += (key.x[i] | key.z[i]).count_ones();
            }
            *c *= self.damp.powi(w as i32);
        }
        let mut s = [0u64; W];
        let mut k: i64 = 0;
        for i in 0..W {
            let mut bits = key.x[i];
            while bits != 0 {
                let q = i * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let f = &self.geo.flip[q];
                for j in 0..W {
                    s[j] ^= f[j];
                }
                // cut edges at q: neighbours with x = 0
                let nb = &self.geo.nbr[q];
                let mut inside = 0u32;
                for j in 0..W {
                    inside += (nb[j] & key.x[j]).count_ones();
                }
                k += (self.geo.deg[q] - inside) as i64;
            }
        }
        // Each cut edge has exactly one endpoint with x=1, counted once. Good.
        let mut xz0 = 0i64;
        let mut xz1 = 0i64;
        for i in 0..W {
            xz0 += (key.x[i] & key.z[i]).count_ones() as i64;
            key.z[i] ^= s[i];
            xz1 += (key.x[i] & key.z[i]).count_ones() as i64;
        }
        let e = (k + xz0 - xz1).rem_euclid(4);
        debug_assert!(e % 2 == 0);
        if e == 2 {
            *c = -*c;
        }
        if self.max_weight != usize::MAX {
            let mut w = 0usize;
            for i in 0..W {
                w += (key.x[i] | key.z[i]).count_ones() as usize;
            }
            if w > self.max_weight {
                acc.l1 += c.abs();
                acc.l2sq += *c * *c;
                return false;
            }
        }
        true
    }

    /// Closed-form `<0| RX† P RX |0>` contribution.
    #[inline]
    fn eval(&self, key: &Key<W>, c: f64) -> f64 {
        let mut nz = 0;
        let mut ny = 0;
        for i in 0..W {
            if key.x[i] & !key.z[i] != 0 {
                return 0.0;
            }
            nz += (key.z[i] & !key.x[i]).count_ones();
            ny += (key.x[i] & key.z[i]).count_ones();
        }
        c * self.cos.powi(nz as i32) * (-self.sin).powi(ny as i32)
    }

    /// RX-layer children of one term, depth first with δ pruning.
    #[inline]
    fn branch(&self, key: Key<W>, c: f64, emit: &mut impl FnMut(Key<W>, f64), acc: &mut Acc) {
        let mut sites = [0u16; 2048];
        let mut m = 0usize;
        for i in 0..W {
            let mut bits = key.z[i];
            while bits != 0 {
                sites[m] = (i * 64 + bits.trailing_zeros() as usize) as u16;
                m += 1;
                bits &= bits - 1;
            }
        }
        self.dfs(&sites[..m], key, c, emit, acc);
    }

    fn dfs(
        &self,
        sites: &[u16],
        key: Key<W>,
        c: f64,
        emit: &mut impl FnMut(Key<W>, f64),
        acc: &mut Acc,
    ) {
        let Some((&q, rest)) = sites.split_first() else {
            emit(key, c);
            return;
        };
        let (wi, b) = ((q / 64) as usize, 1u64 << (q % 64));
        let r = rest.len() as i32;
        // keep: coefficient c·cos
        let ck = c * self.cos;
        if ck != 0.0 {
            if ck.abs() >= self.branch_delta {
                self.dfs(rest, key, ck, emit, acc);
            } else {
                acc.l1 += ck.abs() * (self.cos.abs() + self.sin.abs()).powi(r);
                acc.l2sq += ck * ck;
            }
        }
        // toggle x: Z → +sin Y, Y → −sin Z
        let xbit = key.x[wi] & b != 0;
        let cs = if xbit { -c * self.sin } else { c * self.sin };
        if cs != 0.0 {
            if cs.abs() >= self.branch_delta {
                let mut k2 = key;
                k2.x[wi] ^= b;
                self.dfs(rest, k2, cs, emit, acc);
            } else {
                acc.l1 += cs.abs() * (self.cos.abs() + self.sin.abs()).powi(r);
                acc.l2sq += cs * cs;
            }
        }
    }
}

/// Runs SPD for `<0| U† O U |0>`.
pub fn simulate(model: &KickedIsing, obs: &PauliObs, opt: &SpdOptions) -> SpdResult {
    let t0 = Instant::now();
    let lat = &model.lattice;
    let support = obs.support();
    assert!(
        support.iter().all(|&q| q < lat.n),
        "observable outside lattice"
    );
    // Light cone: ball of radius `steps` (the final RX layer adds nothing).
    let keep: Vec<usize> = if opt.light_cone {
        let d = lat.distances_from(&support);
        lat.bfs_order(&support)
            .into_iter()
            .filter(|&v| d[v] <= model.steps)
            .collect()
    } else {
        let mut o = lat.bfs_order(&support);
        let mut seen = vec![false; lat.n];
        for &v in &o {
            seen[v] = true;
        }
        o.extend((0..lat.n).filter(|&v| !seen[v]));
        o
    };
    let (sub, map) = lat.induced(&keep);
    let words = sub.n.div_ceil(64).max(1);
    macro_rules! go {
        ($($w:literal)*) => {
            match words {
                $( w if w <= $w => run::<$w>(model, &sub, &map, obs, opt), )*
                _ => panic!("light cone of {} qubits too wide", sub.n),
            }
        };
    }
    let mut r = go!(1 2 3 4 5 6 7 8 10 12 14 16 18 20 24 32);
    r.active_qubits = sub.n;
    r.seconds = t0.elapsed().as_secs_f64();
    r
}

fn run<const W: usize>(
    model: &KickedIsing,
    sub: &Lattice,
    map: &[usize],
    obs: &PauliObs,
    opt: &SpdOptions,
) -> SpdResult {
    let n = sub.n;
    let mut geo = Geometry::<W> {
        flip: vec![[0; W]; n],
        nbr: vec![[0; W]; n],
        deg: vec![0; n],
    };
    for q in 0..n {
        let d = sub.adj[q].len();
        geo.deg[q] = d as u32;
        for &r in &sub.adj[q] {
            geo.flip[q][r / 64] ^= 1 << (r % 64);
            geo.nbr[q][r / 64] |= 1 << (r % 64);
        }
        if d % 2 == 1 {
            geo.flip[q][q / 64] ^= 1 << (q % 64);
        }
    }
    let snap = |v: f64| if v.abs() < 1e-14 { 0.0 } else { v };
    let ctx = Ctx::<W> {
        geo: &geo,
        cos: snap(model.theta_h.cos()),
        sin: snap(model.theta_h.sin()),
        delta: opt.delta,
        branch_delta: opt.delta * opt.branch_factor,
        max_weight: opt.max_weight,
        damp: 1.0 - 4.0 * opt.depol / 3.0,
    };

    // Initial terms (merged).
    let mut init: HashMap<Key<W>, f64> = HashMap::new();
    for (s, c) in &obs.terms {
        let mut k = Key::<W> {
            x: [0; W],
            z: [0; W],
        };
        for &(q, p) in s {
            let nq = map[q];
            let (wi, b) = (nq / 64, 1u64 << (nq % 64));
            assert!(k.x[wi] & b == 0 && k.z[wi] & b == 0, "repeated qubit {q}");
            match p {
                'X' => k.x[wi] |= b,
                'Y' => {
                    k.x[wi] |= b;
                    k.z[wi] |= b
                }
                'Z' => k.z[wi] |= b,
                _ => panic!("bad Pauli {p}"),
            }
        }
        *init.entry(k).or_insert(0.0) += c;
    }
    let mut terms: Vec<(Key<W>, f64)> = init.into_iter().filter(|t| t.1 != 0.0).collect();
    let mut res = SpdResult {
        words: W,
        peak_terms: terms.len(),
        ..Default::default()
    };
    let mut acc = Acc::default();

    // Backward passes. Each merged pass = [ZZ (+noise, weight cap)] then RX
    // branching; the first pass has no ZZ when `final_rx`.
    let mut passes: Vec<bool> = Vec::new(); // true = ZZ before RX
    if model.final_rx {
        passes.push(false);
    }
    passes.extend(std::iter::repeat(true).take(model.steps.saturating_sub(1)));
    let nm = passes.len().saturating_sub(opt.stream);
    for &with_zz in &passes[..nm] {
        match merge_pass(&ctx, &terms, with_zz, opt.max_terms) {
            Some((t, a)) => {
                terms = t;
                acc.l1 += a.l1;
                acc.l2sq += a.l2sq;
            }
            None => {
                res.aborted = true;
                res.discarded_l1 = acc.l1;
                res.discarded_l2sq = acc.l2sq;
                return res;
            }
        }
        res.terms_per_layer.push(terms.len());
        res.peak_terms = res.peak_terms.max(terms.len());
    }
    // Remaining layers streamed depth first, then the last step: ZZ and the
    // closed-form first RX layer (or <0|P|0> when there is no Trotter step).
    res.final_terms = terms.len();
    let streamed = &passes[nm..];
    let no_steps = model.steps == 0;
    let (v, n2, a) = terms
        .par_chunks(256)
        .map(|ch| {
            let mut a = Acc::default();
            let mut v = 0.0;
            let mut n2 = 0.0;
            for &(k0, c0) in ch {
                n2 += c0 * c0;
                v += stream_eval(&ctx, streamed, k0, c0, &mut a, no_steps);
            }
            (v, n2, a)
        })
        .reduce(
            || (0.0, 0.0, Acc::default()),
            |x, y| {
                (
                    x.0 + y.0,
                    x.1 + y.1,
                    Acc {
                        l1: x.2.l1 + y.2.l1,
                        l2sq: x.2.l2sq + y.2.l2sq,
                    },
                )
            },
        );
    res.value = v;
    res.norm2 = n2;
    acc.l1 += a.l1;
    acc.l2sq += a.l2sq;
    res.discarded_l1 = acc.l1;
    res.discarded_l2sq = acc.l2sq;
    res
}

/// Depth-first evaluation of the remaining (unmerged) layers of one term.
fn stream_eval<const W: usize>(
    ctx: &Ctx<W>,
    passes: &[bool],
    key: Key<W>,
    c: f64,
    acc: &mut Acc,
    no_steps: bool,
) -> f64 {
    let (mut k, mut c) = (key, c);
    match passes.split_first() {
        None => {
            if no_steps {
                if (0..W).any(|i| k.x[i] != 0) {
                    0.0
                } else {
                    c
                }
            } else if ctx.zz(&mut k, &mut c, acc) {
                ctx.eval(&k, c)
            } else {
                0.0
            }
        }
        Some((&with_zz, rest)) => {
            if with_zz && !ctx.zz(&mut k, &mut c, acc) {
                return 0.0;
            }
            let mut sum = 0.0;
            let mut inner = Acc::default();
            ctx.branch(
                k,
                c,
                &mut |kk, cc| sum += stream_eval(ctx, rest, kk, cc, &mut inner, no_steps),
                acc,
            );
            acc.l1 += inner.l1;
            acc.l2sq += inner.l2sq;
            sum
        }
    }
}

/// One merged backward layer. `None` if `max_terms` was exceeded.
#[allow(clippy::type_complexity)]
fn merge_pass<const W: usize>(
    ctx: &Ctx<W>,
    terms: &[(Key<W>, f64)],
    with_zz: bool,
    max_terms: usize,
) -> Option<(Vec<(Key<W>, f64)>, Acc)> {
    let shards: Vec<Mutex<Map<W>>> = (0..SHARDS).map(|_| Mutex::new(Map::default())).collect();
    let count = AtomicUsize::new(0);
    let abort = AtomicBool::new(false);
    let chunk = 1024;
    let acc = terms
        .par_chunks(chunk)
        .map(|ch| {
            let mut acc = Acc::default();
            if abort.load(Ordering::Relaxed) {
                return acc;
            }
            let mut bufs: Vec<Vec<(Key<W>, f64)>> = (0..SHARDS).map(|_| Vec::new()).collect();
            let flush = |s: usize, buf: &mut Vec<(Key<W>, f64)>| {
                let mut m = shards[s].lock().unwrap();
                let before = m.len();
                for (k, c) in buf.drain(..) {
                    *m.entry(k).or_insert(0.0) += c;
                }
                let added = m.len() - before;
                drop(m);
                if count.fetch_add(added, Ordering::Relaxed) + added > max_terms {
                    abort.store(true, Ordering::Relaxed);
                }
            };
            for &(k0, c0) in ch {
                let (mut k, mut c) = (k0, c0);
                if with_zz && !ctx.zz(&mut k, &mut c, &mut acc) {
                    continue;
                }
                let mut emit = |kk: Key<W>, cc: f64| {
                    let s = shard_of(key_hash(&kk));
                    bufs[s].push((kk, cc));
                    if bufs[s].len() >= FLUSH {
                        let mut b = std::mem::take(&mut bufs[s]);
                        flush(s, &mut b);
                        bufs[s] = b;
                    }
                };
                ctx.branch(k, c, &mut emit, &mut acc);
            }
            for s in 0..SHARDS {
                if !bufs[s].is_empty() {
                    let mut b = std::mem::take(&mut bufs[s]);
                    flush(s, &mut b);
                }
            }
            acc
        })
        .reduce(Acc::default, |a, b| Acc {
            l1: a.l1 + b.l1,
            l2sq: a.l2sq + b.l2sq,
        });
    if abort.load(Ordering::Relaxed) {
        return None;
    }
    let delta = ctx.delta;
    let parts: Vec<(Vec<(Key<W>, f64)>, Acc)> = shards
        .into_par_iter()
        .map(|m| {
            let m = m.into_inner().unwrap();
            let mut a = Acc::default();
            let mut v = Vec::with_capacity(m.len());
            for (k, c) in m {
                if c != 0.0 && c.abs() >= delta {
                    v.push((k, c));
                } else {
                    a.l1 += c.abs();
                    a.l2sq += c * c;
                }
            }
            (v, a)
        })
        .collect();
    let total: usize = parts.iter().map(|p| p.0.len()).sum();
    let mut out = Vec::with_capacity(total);
    let mut acc2 = acc;
    for (v, a) in parts {
        out.extend(v);
        acc2.l1 += a.l1;
        acc2.l2sq += a.l2sq;
    }
    Some((out, acc2))
}

/// Light-cone size (qubits within distance `steps` of `support`).
pub fn light_cone_size(lat: &Lattice, support: &[usize], steps: usize) -> usize {
    lat.distances_from(support)
        .iter()
        .filter(|&&d| d <= steps)
        .count()
}

/// Bytes per stored term for a light cone of `n` qubits (key + coefficient).
pub fn term_bytes(n: usize) -> usize {
    16 * n.div_ceil(64).max(1) + 8
}
