//! Exact hybrid Schrödinger–Feynman (HSF) simulation, in the style of
//! Google's `qsimh` (Markov et al., arXiv:1807.10749).
//!
//! The qubits ("wires") are split into two blocks `A | B`. A gate whose
//! qubits all lie in one block acts on that block's small state vector
//! (Schrödinger part). A gate that crosses the cut is written as an
//! operator-Schmidt sum `U = Σ_k M_k ⊗ N_k` with `M_k` acting on block A and
//! `N_k` on block B; choosing one term per crossing gate gives a *path*, along
//! which the state stays a product `|a_p> ⊗ |b_p>`, and the output is the sum
//! over all paths (Feynman part):
//!
//! ```text
//! <x|U|0> = Σ_p <x_A|a_p> · <x_B|b_p>
//! ```
//!
//! The result is exact (up to floating-point rounding); nothing is truncated.
//! Memory is `O(k · (2^|A| + 2^|B|))` for `k` crossing gates rather than
//! `2^n`, and time is `(Π_i rank_i) × (block work)`.
//!
//! Implementation notes (see `research/performance/hsf.md` for the measurements):
//!
//! * Paths are enumerated **depth first**. Each crossing gate is a level of
//!   the path tree; at each level the walker keeps one checkpoint (a pair of
//!   block vectors) so the siblings can be generated from it, and the last
//!   sibling reuses the parent's buffers. Live memory is `k + 1` pairs per
//!   worker, independent of the number of paths.
//! * Work is parallelised over path *prefixes* (the term choices at the
//!   first few crossing gates). Each worker is an OS thread with its own
//!   preallocated stack of buffers, so the number of live buffers is a hard,
//!   computable bound, which is checked against [`MAX_HSF_BYTES`].
//! * Crossing gates use their true operator-Schmidt rank: CZ, CNOT and
//!   controlled-phase are `|0><0| ⊗ 1 + |1><1| ⊗ V` (rank 2), and a Toffoli
//!   split 1|2 across the cut is also rank 2. Any other two-qubit matrix is
//!   decomposed numerically with an SVD of the reshuffled 4×4 matrix
//!   ([`operator_schmidt`]).
//! * SWAP gates are removed exactly by relabelling wires (a SWAP followed by
//!   the rest of the circuit equals the rest of the circuit with the two
//!   labels exchanged, followed by the SWAP), so they never cross the cut.
//! * Gates are scheduled "as soon as possible" relative to the crossing
//!   gates: a block gate is moved before every crossing gate it commutes with
//!   trivially (disjoint qubits). Work done before the first crossing gate is
//!   done once instead of once per path.
//! * A projector term that annihilates the block state *exactly* (every
//!   kept amplitude is 0.0) prunes the whole subtree; this is exact.
//! * For a batch of amplitudes the last segment of each block can be applied
//!   backwards to the requested basis states once ("bra" leaves) instead of
//!   forwards on every path.

use crate::circuit::{check_gate, Circuit, Op, SimError};
use crate::engines::statevector::{StateVectorF64, MAX_STATE_BYTES};
use crate::gate::{Gate, Mat2, Mat4};
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatMut, MatRef, Par};
use num_complex::Complex64;
use rand::Rng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Memory cap for one HSF computation (all live block vectors, buffers and,
/// for full output, the `2^n` result), mirroring the state vector's cap.
pub const MAX_HSF_BYTES: u128 = 1 << 30;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);
/// Below this many amplitudes loops run on the calling thread.
const PAR_MIN_LEN: usize = 1 << 14;
/// Singular values at or below this (relative to the largest) are treated
/// as exact zeros when computing a numerical operator-Schmidt rank. The
/// singular values of gate matrices are O(1); rounding noise is ~1e-16.
pub const SCHMIDT_ZERO_TOL: f64 = 1e-13;

/// How crossing two-qubit gates are decomposed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchmidtMode {
    /// Closed forms for CZ, CNOT, CPhase (rank 2); SVD for anything else.
    Analytic,
    /// Always use the numerical SVD (for testing the generic path).
    Svd,
    /// The first attempt's expansion `U = Σ_ij E_ij ⊗ U_ij` (always 4
    /// terms, even for rank-2 gates). Kept only for A/B benchmarks.
    MatrixUnits,
}

/// How a path's final block states are turned into amplitudes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafMode {
    /// Apply the last segment to every path and read amplitudes off.
    Forward,
    /// Apply the inverse of the last segment once to each requested basis
    /// state and take inner products with each path's state.
    Bra,
    /// Pick per block by a cost estimate.
    Auto,
}

/// Tuning knobs. The defaults are what the benchmarks call "HSF".
#[derive(Clone, Debug)]
pub struct HsfOptions {
    /// Remove SWAPs by relabelling wires (exact).
    pub eliminate_swaps: bool,
    /// Schedule block gates as early as possible relative to crossing gates.
    pub asap: bool,
    /// How crossing two-qubit gates are decomposed into Schmidt terms.
    pub schmidt: SchmidtMode,
    /// Leaf mode for `amplitudes` (full output always uses `Forward`).
    pub leaf: LeafMode,
    /// Prune paths whose block state becomes exactly zero.
    pub prune_zero: bool,
    /// Number of path workers; 0 = the rayon thread count. Reduced
    /// automatically to respect `max_bytes`.
    pub threads: usize,
    /// Memory cap in bytes.
    pub max_bytes: u128,
    /// Leaves per GEMM batch when accumulating the full output.
    pub gemm_batch: usize,
}

impl Default for HsfOptions {
    fn default() -> Self {
        HsfOptions {
            eliminate_swaps: true,
            asap: true,
            schmidt: SchmidtMode::Analytic,
            leaf: LeafMode::Auto,
            prune_zero: true,
            threads: 0,
            max_bytes: MAX_HSF_BYTES,
            gemm_batch: 64,
        }
    }
}

/// An operation on one block's state vector, with block-local qubit indices.
#[derive(Clone, Debug)]
enum LocalOp {
    Gate(Gate),
    Mat(usize, Mat2),
    /// Keep amplitudes with `index & mask == val`, zero the rest.
    Proj {
        mask: usize,
        val: usize,
    },
    /// Zero amplitudes with `index & mask == val` (the complement of `Proj`).
    ProjNot {
        mask: usize,
        val: usize,
    },
}

/// One crossing gate: its operator-Schmidt terms, each `[op on A, op on B]`
/// (`None` = identity).
#[derive(Clone, Debug)]
struct CutGate {
    gate: Gate,
    terms: Vec<[Option<LocalOp>; 2]>,
}

/// One block: its wires (global index of each local qubit) and its gates
/// grouped into `k + 1` segments separated by the `k` crossing gates.
#[derive(Clone, Debug)]
struct Block {
    wires: Vec<usize>,
    segs: Vec<Vec<Gate>>,
}

/// An exact two-block hybrid Schrödinger–Feynman simulation of one circuit.
#[derive(Clone, Debug)]
pub struct HybridSchrodingerFeynman {
    n: usize,
    /// `in_a[w]`: wire `w` belongs to block A.
    in_a: Vec<bool>,
    /// Wire holding logical qubit `q` at the end of the circuit.
    final_wire: Vec<usize>,
    blocks: [Block; 2],
    cuts: Vec<CutGate>,
    opts: HsfOptions,
}

/// Rewrites the qubit indices of a gate.
pub fn map_gate(g: Gate, f: impl Fn(usize) -> usize) -> Gate {
    use Gate::*;
    match g {
        H(q) => H(f(q)),
        X(q) => X(f(q)),
        Y(q) => Y(f(q)),
        Z(q) => Z(f(q)),
        S(q) => S(f(q)),
        Sdg(q) => Sdg(f(q)),
        T(q) => T(f(q)),
        Tdg(q) => Tdg(f(q)),
        Rx(q, t) => Rx(f(q), t),
        Ry(q, t) => Ry(f(q), t),
        Rz(q, t) => Rz(f(q), t),
        Phase(q, t) => Phase(f(q), t),
        Cnot(a, b) => Cnot(f(a), f(b)),
        Cz(a, b) => Cz(f(a), f(b)),
        Swap(a, b) => Swap(f(a), f(b)),
        CPhase(a, b, t) => CPhase(f(a), f(b), t),
        Ccx(a, b, t) => Ccx(f(a), f(b), f(t)),
        I(q) => I(f(q)),
        Sx(q) => Sx(f(q)),
        Sxdg(q) => Sxdg(f(q)),
        U(q, th, ph, la) => U(f(q), th, ph, la),
        ISwap(a, b) => ISwap(f(a), f(b)),
        ISwapdg(a, b) => ISwapdg(f(a), f(b)),
    }
}

/// Operator-Schmidt decomposition of a two-qubit matrix (indexed by
/// `2·bit(a) + bit(b)`): returns `(M_k, N_k)` with `U = Σ_k M_k ⊗ N_k`, `M_k`
/// acting on the first qubit `a`. The singular values are folded into `M_k`.
/// Terms with singular value `<= SCHMIDT_ZERO_TOL · s_max` are dropped, so
/// the length of the result is the numerical operator-Schmidt rank.
///
/// Reshuffle: `R[(i_a, j_a), (i_b, j_b)] = U[2 i_a + i_b][2 j_a + j_b]`; an SVD
/// `R = Σ s_k u_k v_k^†` gives `M_k[i][j] = s_k u_k[2i + j]` and
/// `N_k[i][j] = conj(v_k[2i + j])`.
pub fn operator_schmidt(m: &Mat4) -> Vec<(Mat2, Mat2)> {
    let r = Mat::<Complex64>::from_fn(4, 4, |row, col| {
        let (ia, ja) = (row >> 1, row & 1);
        let (ib, jb) = (col >> 1, col & 1);
        m[2 * ia + ib][2 * ja + jb]
    });
    let svd = r.thin_svd().expect("4x4 SVD converges");
    let (u, v) = (svd.U(), svd.V());
    let s: Vec<f64> = (0..4).map(|k| svd.S().column_vector()[k].re).collect();
    let smax = s.iter().cloned().fold(0.0, f64::max);
    let mut out = Vec::new();
    for k in 0..4 {
        if s[k] <= SCHMIDT_ZERO_TOL * smax {
            continue;
        }
        let mut ma = [[ZERO; 2]; 2];
        let mut nb = [[ZERO; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                ma[i][j] = u[(2 * i + j, k)] * s[k];
                nb[i][j] = v[(2 * i + j, k)].conj();
            }
        }
        out.push((ma, nb));
    }
    out
}

/// Numerical operator-Schmidt rank of a two-qubit matrix.
pub fn schmidt_rank(m: &Mat4) -> usize {
    operator_schmidt(m).len()
}

/// Exponent of the path count contributed by one gate on wires that are
/// split by the partition (`log2(rank)`), for the cut-selection objective.
fn cut_weight(g: &Gate, opts: &HsfOptions) -> u32 {
    match *g {
        Gate::Cnot(..) | Gate::Cz(..) | Gate::Ccx(..) => 1,
        Gate::CPhase(_, _, t) => u32::from(Complex64::from_polar(1.0, t) != ONE),
        Gate::Swap(..) => {
            if opts.eliminate_swaps {
                0
            } else {
                2
            }
        }
        _ => 0,
    }
}

/// Walks the circuit's gates with SWAP elimination, calling `f` with each
/// remaining gate mapped onto wires. Returns the final logical→wire map.
fn for_each_wire_gate(
    c: &Circuit,
    opts: &HsfOptions,
    mut f: impl FnMut(Gate) -> Result<(), SimError>,
) -> Result<Vec<usize>, SimError> {
    let n = c.num_qubits;
    let mut wire: Vec<usize> = (0..n).collect();
    for (i, op) in c.ops.iter().enumerate() {
        let g = match op {
            Op::Gate(g) => *g,
            // Measurement, reset, noise channels and classically conditioned
            // gates are not unitary; HSF sums amplitudes of a unitary circuit.
            _ => {
                return Err(SimError::MeasurementNotSupported {
                    backend: "hsf",
                    op_index: i,
                })
            }
        };
        check_gate(&g, n)?;
        if let (true, Gate::Swap(a, b)) = (opts.eliminate_swaps, g) {
            wire.swap(a, b);
            continue;
        }
        f(map_gate(g, |q| wire[q]))?;
    }
    Ok(wire)
}

impl HybridSchrodingerFeynman {
    /// Plans an HSF simulation of `circuit` with wire `w` in block A iff
    /// `in_a[w]`. Fails on measurements, bad qubits, or `n > 62`.
    pub fn new(circuit: &Circuit, in_a: &[bool], opts: HsfOptions) -> Result<Self, SimError> {
        let n = circuit.num_qubits;
        assert_eq!(in_a.len(), n, "partition length must equal num_qubits");
        if n > 62 {
            return Err(SimError::TooLarge {
                what: "HSF register (bit strings are 64-bit)",
                bytes: 1u128 << n.min(127),
                limit: 1u128 << 62,
            });
        }
        let side = |w: usize| usize::from(!in_a[w]); // 0 = A, 1 = B
        let mut loc = vec![0usize; n];
        let mut blocks = [
            Block {
                wires: vec![],
                segs: vec![vec![]],
            },
            Block {
                wires: vec![],
                segs: vec![vec![]],
            },
        ];
        for w in 0..n {
            let b = &mut blocks[side(w)];
            loc[w] = b.wires.len();
            b.wires.push(w);
        }
        for b in &blocks {
            let bytes = (1u128 << b.wires.len()) * 16;
            if bytes > MAX_STATE_BYTES {
                return Err(SimError::TooLarge {
                    what: "HSF block state vector",
                    bytes,
                    limit: MAX_STATE_BYTES,
                });
            }
        }
        let mut cuts: Vec<CutGate> = Vec::new();
        // level[w]: earliest segment a gate on wire w may be scheduled in.
        let mut level = vec![0usize; n];
        let final_wire = for_each_wire_gate(circuit, &opts, |g| {
            let ws = g.qubits();
            let s0 = side(ws[0]);
            if ws.iter().all(|&w| side(w) == s0) {
                let lv = if opts.asap {
                    ws.iter().map(|&w| level[w]).max().unwrap_or(0)
                } else {
                    cuts.len()
                };
                for &w in &ws {
                    level[w] = lv;
                }
                blocks[s0].segs[lv].push(map_gate(g, |w| loc[w]));
                return Ok(());
            }
            let Some(terms) = cut_terms(&g, &|w| side(w), &|w| loc[w], opts.schmidt) else {
                return Ok(()); // identity (e.g. CPhase(0))
            };
            let lv = cuts.len();
            for &w in &ws {
                level[w] = lv + 1;
            }
            cuts.push(CutGate { gate: g, terms });
            blocks[0].segs.push(vec![]);
            blocks[1].segs.push(vec![]);
            Ok(())
        })?;
        Ok(HybridSchrodingerFeynman {
            n,
            in_a: in_a.to_vec(),
            final_wire,
            blocks,
            cuts,
            opts,
        })
    }

    /// Plans with an automatically chosen partition ([`auto_partition`]).
    pub fn auto(circuit: &Circuit, opts: HsfOptions) -> Result<Self, SimError> {
        let in_a = auto_partition(circuit, &opts)?;
        Self::new(circuit, &in_a, opts)
    }

    /// Number of qubits (wires) of the planned circuit.
    pub fn num_qubits(&self) -> usize {
        self.n
    }
    /// `in_a[w]` for every wire.
    pub fn partition(&self) -> &[bool] {
        &self.in_a
    }
    /// Sizes of blocks A and B.
    pub fn block_sizes(&self) -> (usize, usize) {
        (self.blocks[0].wires.len(), self.blocks[1].wires.len())
    }
    /// Number of gates crossing the cut (after SWAP elimination).
    pub fn num_cut_gates(&self) -> usize {
        self.cuts.len()
    }
    /// The crossing gates (on wires) and their Schmidt ranks.
    pub fn cut_gates(&self) -> Vec<(Gate, usize)> {
        self.cuts.iter().map(|c| (c.gate, c.terms.len())).collect()
    }
    /// Number of Feynman paths (product of the Schmidt ranks).
    pub fn num_paths(&self) -> u128 {
        self.cuts
            .iter()
            .fold(1u128, |p, c| p.saturating_mul(c.terms.len() as u128))
    }
    /// Gates per block per segment (for diagnostics).
    pub fn segment_sizes(&self) -> [Vec<usize>; 2] {
        [0, 1].map(|s| self.blocks[s].segs.iter().map(|v| v.len()).collect())
    }

    fn pair_bytes(&self) -> u128 {
        let (na, nb) = self.block_sizes();
        ((1u128 << na) + (1u128 << nb)) * 16
    }

    /// Decides the prefix depth and number of workers under the memory cap.
    /// `shared` is memory that does not scale with workers, `per_worker_extra`
    /// is per-worker scratch beyond the buffer stack.
    fn schedule(&self, shared: u128, per_worker_extra: u128) -> Result<(usize, usize), SimError> {
        let k = self.cuts.len();
        let pair = self.pair_bytes();
        let want = if self.opts.threads == 0 {
            rayon::current_num_threads()
        } else {
            self.opts.threads
        }
        .max(1);
        // Smallest prefix depth giving >= 4 tasks per worker (or all levels).
        let mut p = 0;
        let mut prefixes = 1u128;
        while p < k && prefixes < 4 * want as u128 {
            prefixes *= self.cuts[p].terms.len() as u128;
            p += 1;
        }
        let workers = want.min(prefixes.min(usize::MAX as u128) as usize).max(1);
        // shared root pair + per worker: root copy + (k - p) level buffers.
        let per_worker = pair * (1 + (k - p) as u128) + per_worker_extra;
        let base = shared + pair;
        let mut w = workers;
        while w > 1 && base + per_worker * w as u128 > self.opts.max_bytes {
            w -= 1;
        }
        let total = base + per_worker * w as u128;
        if total > self.opts.max_bytes {
            return Err(SimError::TooLarge {
                what: "HSF working set",
                bytes: total,
                limit: self.opts.max_bytes,
            });
        }
        Ok((p, w))
    }

    /// Estimated peak bytes for a batch of `m` amplitudes (an upper bound on
    /// the block vectors and buffers this module allocates).
    pub fn estimated_bytes_amplitudes(&self, m: usize) -> Result<u128, SimError> {
        let shared = self.bra_bytes(m);
        let (p, w) = self.schedule(shared, 0)?;
        Ok(shared + self.pair_bytes() * (1 + w as u128 * (1 + (self.cuts.len() - p) as u128)))
    }

    fn bra_modes(&self, m_distinct: [usize; 2]) -> [bool; 2] {
        let k = self.cuts.len();
        [0, 1].map(|s| match self.opts.leaf {
            LeafMode::Forward => false,
            LeafMode::Bra => true,
            // A bra costs one fused inner-product pass per requested value
            // per path; a forward leaf costs one gate pass per gate in the
            // last segment per path. Inner products are ~1/2 a gate pass.
            LeafMode::Auto => m_distinct[s] < self.blocks[s].segs[k].len() * 2,
        })
    }

    fn bra_bytes(&self, m: usize) -> u128 {
        // upper bound: every requested string distinct on both sides
        let (na, nb) = self.block_sizes();
        let modes = self.bra_modes([m, m]);
        let mut b = 0u128;
        if modes[0] {
            b += m as u128 * (1u128 << na) * 16;
        }
        if modes[1] {
            b += m as u128 * (1u128 << nb) * 16;
        }
        b
    }

    /// Splits a logical bit string into (A-local, B-local) indices.
    fn split_index(&self, x: usize) -> [usize; 2] {
        let mut y = 0usize; // physical (wire) index
        for q in 0..self.n {
            y |= ((x >> q) & 1) << self.final_wire[q];
        }
        [0, 1].map(|s| {
            self.blocks[s]
                .wires
                .iter()
                .enumerate()
                .fold(0, |acc, (j, &w)| acc | (((y >> w) & 1) << j))
        })
    }

    fn apply_seg(&self, s: usize, seg: usize, v: &mut StateVectorF64) {
        for g in &self.blocks[s].segs[seg] {
            v.apply_gate(g).expect("planned gate is valid");
        }
    }

    /// Applies cut `lvl`'s term `t` and segment `lvl + 1` to a pair; returns
    /// false if the path is exactly zero. `skip_last[s]`: leave segment `k`
    /// of block `s` unapplied (bra leaves).
    fn advance(
        &self,
        lvl: usize,
        t: usize,
        pair: &mut [StateVectorF64; 2],
        skip_last: [bool; 2],
    ) -> bool {
        let k = self.cuts.len();
        for (s, v) in pair.iter_mut().enumerate() {
            if let Some(op) = &self.cuts[lvl].terms[t][s] {
                if !apply_local(op, v) && self.opts.prune_zero {
                    return false;
                }
            }
        }
        for (s, v) in pair.iter_mut().enumerate() {
            if !(lvl + 1 == k && skip_last[s]) {
                self.apply_seg(s, lvl + 1, v);
            }
        }
        true
    }

    fn dfs<L: FnMut(&[StateVectorF64; 2])>(
        &self,
        lvl: usize,
        cur: &mut [StateVectorF64; 2],
        bufs: &mut [[StateVectorF64; 2]],
        skip_last: [bool; 2],
        leaf: &mut L,
    ) {
        if lvl == self.cuts.len() {
            leaf(cur);
            return;
        }
        let r = self.cuts[lvl].terms.len();
        let (mine, rest) = bufs.split_first_mut().expect("one buffer per level");
        for t in 0..r {
            if t + 1 < r {
                for s in 0..2 {
                    copy_sv(&mut mine[s], &cur[s]);
                }
                if self.advance(lvl, t, mine, skip_last) {
                    self.dfs(lvl + 1, mine, rest, skip_last, leaf);
                }
            } else if self.advance(lvl, t, cur, skip_last) {
                self.dfs(lvl + 1, cur, rest, skip_last, leaf);
            }
        }
    }

    /// Runs every path. `make` creates a per-worker leaf accumulator; the
    /// accumulators are returned for reduction.
    fn run_paths<A: Send, L>(
        &self,
        skip_last: [bool; 2],
        shared_bytes: u128,
        per_worker_extra: u128,
        make: impl Fn() -> A + Sync,
        leaf: L,
    ) -> Result<Vec<A>, SimError>
    where
        L: Fn(&mut A, &[StateVectorF64; 2]) + Sync,
    {
        let k = self.cuts.len();
        let (p, workers) = self.schedule(shared_bytes, per_worker_extra)?;
        let (na, nb) = self.block_sizes();
        let mut root = [StateVectorF64::new(na), StateVectorF64::new(nb)];
        for (s, v) in root.iter_mut().enumerate() {
            if !(k == 0 && skip_last[s]) {
                self.apply_seg(s, 0, v);
            }
        }
        let radices: Vec<usize> = self.cuts[..p].iter().map(|c| c.terms.len()).collect();
        let prefixes: usize = radices.iter().product();
        let next = AtomicUsize::new(0);
        let worker = || {
            let mut acc = make();
            let mut cur = [StateVectorF64::new(na), StateVectorF64::new(nb)];
            let mut bufs: Vec<[StateVectorF64; 2]> = (p..k)
                .map(|_| [StateVectorF64::new(na), StateVectorF64::new(nb)])
                .collect();
            loop {
                let id = next.fetch_add(1, Ordering::Relaxed);
                if id >= prefixes {
                    break;
                }
                for s in 0..2 {
                    copy_sv(&mut cur[s], &root[s]);
                }
                let mut rem = id;
                let mut alive = true;
                for (lvl, &r) in radices.iter().enumerate() {
                    let t = rem % r;
                    rem /= r;
                    if !self.advance(lvl, t, &mut cur, skip_last) {
                        alive = false;
                        break;
                    }
                }
                if alive {
                    self.dfs(p, &mut cur, &mut bufs, skip_last, &mut |pair| {
                        leaf(&mut acc, pair)
                    });
                }
            }
            acc
        };
        if workers == 1 {
            return Ok(vec![worker()]);
        }
        // OS threads (not rayon tasks) so that a worker blocked inside a
        // parallel gate kernel cannot steal another worker's job and grow a
        // second buffer stack: the number of live stacks is exactly `workers`.
        Ok(std::thread::scope(|sc| {
            let hs: Vec<_> = (0..workers).map(|_| sc.spawn(worker)).collect();
            hs.into_iter()
                .map(|h| h.join().expect("HSF worker panicked"))
                .collect()
        }))
    }

    /// Exact amplitudes `<x|U|0>` for a batch of logical bit strings, computed
    /// in one sweep over the paths.
    pub fn amplitudes(&self, xs: &[usize]) -> Result<Vec<Complex64>, SimError> {
        for &x in xs {
            if self.n < 64 && x >> self.n != 0 {
                return Err(SimError::QubitOutOfRange {
                    qubit: (usize::BITS - x.leading_zeros()) as usize - 1,
                    num_qubits: self.n,
                });
            }
        }
        let m = xs.len();
        if m == 0 {
            return Ok(vec![]);
        }
        // distinct local indices per block, and each request's position in them
        let split: Vec<[usize; 2]> = xs.iter().map(|&x| self.split_index(x)).collect();
        let mut uniq: [Vec<usize>; 2] = [vec![], vec![]];
        let mut pos: Vec<[usize; 2]> = vec![[0, 0]; m];
        for s in 0..2 {
            let mut map: HashMap<usize, usize> = HashMap::new();
            for (j, sp) in split.iter().enumerate() {
                let id = *map.entry(sp[s]).or_insert_with(|| {
                    uniq[s].push(sp[s]);
                    uniq[s].len() - 1
                });
                pos[j][s] = id;
            }
        }
        let bra = self.bra_modes([uniq[0].len(), uniq[1].len()]);
        let k = self.cuts.len();
        let mut bra_bytes = 0u128;
        for s in 0..2 {
            if bra[s] {
                bra_bytes += uniq[s].len() as u128 * (1u128 << self.blocks[s].wires.len()) * 16;
            }
        }
        // memory check before allocating bras
        self.schedule(bra_bytes, 0)?;
        let bras: [Vec<Vec<Complex64>>; 2] = [0, 1].map(|s| {
            if !bra[s] {
                return vec![];
            }
            let nq = self.blocks[s].wires.len();
            uniq[s]
                .iter()
                .map(|&u| {
                    let mut v = StateVectorF64::basis_state(nq, u);
                    for g in self.blocks[s].segs[k].iter().rev() {
                        apply_inverse(g, &mut v);
                    }
                    v.amplitudes().to_vec()
                })
                .collect()
        });
        let make = || {
            (
                vec![ZERO; m],
                [vec![ZERO; uniq[0].len()], vec![ZERO; uniq[1].len()]],
            )
        };
        let leaf = |acc: &mut (Vec<Complex64>, [Vec<Complex64>; 2]), pair: &[StateVectorF64; 2]| {
            let (out, vals) = acc;
            for s in 0..2 {
                let amps = pair[s].amplitudes();
                if bra[s] {
                    multi_dot(&bras[s], amps, &mut vals[s]);
                } else {
                    for (v, &u) in vals[s].iter_mut().zip(&uniq[s]) {
                        *v = amps[u];
                    }
                }
            }
            for (o, p) in out.iter_mut().zip(&pos) {
                *o += vals[0][p[0]] * vals[1][p[1]];
            }
        };
        let accs = self.run_paths(bra, bra_bytes, 0, make, leaf)?;
        let mut out = vec![ZERO; m];
        for (a, _) in accs {
            for (o, v) in out.iter_mut().zip(a) {
                *o += v;
            }
        }
        Ok(out)
    }

    /// One exact amplitude `<x|U|0>`.
    pub fn amplitude(&self, x: usize) -> Result<Complex64, SimError> {
        Ok(self.amplitudes(&[x])?[0])
    }

    /// The full `2^n` output state (logical qubit order, like
    /// [`StateVectorF64::amplitudes`]), accumulated as `Σ_p a_p ⊗ b_p` with
    /// batched complex GEMMs. Subject to the memory cap.
    pub fn state_vector(&self) -> Result<Vec<Complex64>, SimError> {
        let (na, nb) = self.block_sizes();
        let out_bytes = (1u128 << self.n) * 16;
        if out_bytes > MAX_STATE_BYTES {
            return Err(SimError::TooLarge {
                what: "HSF full output",
                bytes: out_bytes,
                limit: MAX_STATE_BYTES,
            });
        }
        let batch = self
            .opts
            .gemm_batch
            .max(1)
            .min(self.num_paths().min(1 << 20) as usize);
        let chunk_bytes = batch as u128 * self.pair_bytes();
        self.schedule(out_bytes, chunk_bytes)?;
        let (la, lb) = (1usize << na, 1usize << nb);
        // psi[ia + la * ib], column-major (la x lb)
        let psi = Mutex::new(vec![ZERO; la * lb]);
        let flush = |ca: &[Complex64], cb: &[Complex64], cols: usize| {
            if cols == 0 {
                return;
            }
            let a = MatRef::from_column_major_slice(&ca[..la * cols], la, cols);
            let b = MatRef::from_column_major_slice(&cb[..lb * cols], lb, cols);
            let mut g = psi.lock().expect("psi lock");
            let dst = MatMut::from_column_major_slice_mut(&mut g[..], la, lb);
            let par = if la * lb >= PAR_MIN_LEN {
                Par::rayon(0)
            } else {
                Par::Seq
            };
            matmul(dst, Accum::Add, a, b.transpose(), ONE, par);
        };
        let make = || (vec![ZERO; la * batch], vec![ZERO; lb * batch], 0usize);
        let leaf = |acc: &mut (Vec<Complex64>, Vec<Complex64>, usize),
                    pair: &[StateVectorF64; 2]| {
            let (ca, cb, cols) = acc;
            let c = *cols;
            ca[c * la..(c + 1) * la].copy_from_slice(pair[0].amplitudes());
            cb[c * lb..(c + 1) * lb].copy_from_slice(pair[1].amplitudes());
            *cols += 1;
            if *cols == batch {
                flush(ca, cb, batch);
                *cols = 0;
            }
        };
        let accs = self.run_paths([false, false], out_bytes, chunk_bytes, make, leaf)?;
        for (ca, cb, cols) in accs {
            flush(&ca, &cb, cols);
        }
        let psi = psi.into_inner().expect("psi lock");
        // storage bit j < na is wire A_j, bit na + j is wire B_j; reorder so
        // bit q is logical qubit q, with in-place bit swaps.
        let mut pos_of_wire = vec![0usize; self.n];
        for (j, &w) in self.blocks[0].wires.iter().enumerate() {
            pos_of_wire[w] = j;
        }
        for (j, &w) in self.blocks[1].wires.iter().enumerate() {
            pos_of_wire[w] = na + j;
        }
        // cur[b] = logical qubit stored at bit b
        let mut cur = vec![0usize; self.n];
        for q in 0..self.n {
            cur[pos_of_wire[self.final_wire[q]]] = q;
        }
        let mut psi = psi;
        for b in 0..self.n {
            while cur[b] != b {
                let t = cur[b];
                swap_index_bits(&mut psi, b, t);
                cur.swap(b, t);
            }
        }
        Ok(psi)
    }
}

/// The Schmidt terms of a crossing gate, as `[A op, B op]` pairs, or `None`
/// if the gate is the identity.
fn cut_terms(
    g: &Gate,
    side: &dyn Fn(usize) -> usize,
    loc: &dyn Fn(usize) -> usize,
    mode: SchmidtMode,
) -> Option<Vec<[Option<LocalOp>; 2]>> {
    let bit = |w: usize| 1usize << loc(w);
    // place (op on side of wire x, op on side of wire y) into [A, B]
    let place = |x: usize, ox: Option<LocalOp>, oy: Option<LocalOp>| -> [Option<LocalOp>; 2] {
        if side(x) == 0 {
            [ox, oy]
        } else {
            [oy, ox]
        }
    };
    let p = |w: usize, v: bool| LocalOp::Proj {
        mask: bit(w),
        val: if v { bit(w) } else { 0 },
    };
    // `|0><0|_c ⊗ 1 + |1><1|_c ⊗ V_t`
    let controlled = |c: usize, v: LocalOp| {
        vec![
            place(c, Some(p(c, false)), None),
            place(c, Some(p(c, true)), Some(v)),
        ]
    };
    match (*g, mode) {
        (Gate::Ccx(c1, c2, t), _) => {
            if side(c1) == side(c2) {
                // controls together, target alone: (1 - P11) ⊗ 1 + P11 ⊗ X
                let mask = bit(c1) | bit(c2);
                Some(vec![
                    place(c1, Some(LocalOp::ProjNot { mask, val: mask }), None),
                    place(
                        c1,
                        Some(LocalOp::Proj { mask, val: mask }),
                        Some(LocalOp::Gate(Gate::X(loc(t)))),
                    ),
                ])
            } else {
                // one control alone, the other control with the target
                let (alone, other) = if side(c1) != side(t) {
                    (c1, c2)
                } else {
                    (c2, c1)
                };
                Some(controlled(
                    alone,
                    LocalOp::Gate(Gate::Cnot(loc(other), loc(t))),
                ))
            }
        }
        (Gate::Cz(a, b), SchmidtMode::Analytic) => {
            Some(controlled(a, LocalOp::Gate(Gate::Z(loc(b)))))
        }
        (Gate::Cnot(c, t), SchmidtMode::Analytic) => {
            Some(controlled(c, LocalOp::Gate(Gate::X(loc(t)))))
        }
        (Gate::CPhase(a, b, th), SchmidtMode::Analytic) => {
            if Complex64::from_polar(1.0, th) == ONE {
                None
            } else {
                Some(controlled(a, LocalOp::Gate(Gate::Phase(loc(b), th))))
            }
        }
        (_, SchmidtMode::MatrixUnits) => {
            let qs = g.qubits();
            let (a, b) = (qs[0], qs[1]);
            let m = g.matrix_2q().expect("two-qubit gate");
            let mut terms = Vec::new();
            for x in 0..4 {
                let (i, j) = (x >> 1, x & 1);
                let mut e = [[ZERO; 2]; 2];
                e[i][j] = ONE;
                let blk: Mat2 = [
                    [m[2 * i][2 * j], m[2 * i][2 * j + 1]],
                    [m[2 * i + 1][2 * j], m[2 * i + 1][2 * j + 1]],
                ];
                terms.push(place(
                    a,
                    Some(LocalOp::Mat(loc(a), e)),
                    Some(LocalOp::Mat(loc(b), blk)),
                ));
            }
            Some(terms)
        }
        _ => {
            let qs = g.qubits();
            let (a, b) = (qs[0], qs[1]);
            let m = g.matrix_2q().expect("two-qubit gate");
            let terms = operator_schmidt(&m);
            if terms.len() == 1 && is_identity_product(&terms[0]) {
                return None;
            }
            Some(
                terms
                    .into_iter()
                    .map(|(ma, nb)| {
                        place(
                            a,
                            Some(LocalOp::Mat(loc(a), ma)),
                            Some(LocalOp::Mat(loc(b), nb)),
                        )
                    })
                    .collect(),
            )
        }
    }
}

fn is_identity_product((m, n): &(Mat2, Mat2)) -> bool {
    // M ⊗ N == 1 ⊗ 1 up to the rounding of the SVD
    (0..16).all(|x| {
        let (i, j, k, l) = (x >> 3, (x >> 2) & 1, (x >> 1) & 1, x & 1);
        let want = if i == j && k == l { ONE } else { ZERO };
        (m[i][j] * n[k][l] - want).norm() < 1e-14
    })
}

/// Exchanges bits `i` and `j` of the index of every element, in place.
fn swap_index_bits(v: &mut [Complex64], i: usize, j: usize) {
    let (lo, hi) = (i.min(j), i.max(j));
    if lo == hi {
        return;
    }
    let f = |chunk: &mut [Complex64]| {
        let (c0, c1) = chunk.split_at_mut(1 << hi);
        // c0: bit hi = 0; swap (bit lo = 1, hi = 0) with (lo = 0, hi = 1)
        for x in (0..c0.len()).filter(|x| x >> lo & 1 == 1) {
            std::mem::swap(&mut c0[x], &mut c1[x ^ (1 << lo)]);
        }
    };
    if v.len() >= PAR_MIN_LEN && v.len() > 1 << (hi + 1) {
        v.par_chunks_mut(1 << (hi + 1)).for_each(f);
    } else {
        v.chunks_mut(1 << (hi + 1)).for_each(f);
    }
}

fn copy_sv(dst: &mut StateVectorF64, src: &StateVectorF64) {
    let (d, s) = (dst.amplitudes_mut(), src.amplitudes());
    if d.len() >= PAR_MIN_LEN {
        d.par_chunks_mut(PAR_MIN_LEN)
            .zip(s.par_chunks(PAR_MIN_LEN))
            .for_each(|(x, y)| x.copy_from_slice(y));
    } else {
        d.copy_from_slice(s);
    }
}

/// Applies a projector; returns whether any kept amplitude is nonzero.
fn project(v: &mut StateVectorF64, mask: usize, val: usize, keep_match: bool) -> bool {
    let f = |off: usize, xs: &mut [Complex64]| {
        let mut any = false;
        for (i, a) in xs.iter_mut().enumerate() {
            if (((off + i) & mask) == val) != keep_match {
                *a = ZERO;
            } else {
                any |= a.re != 0.0 || a.im != 0.0;
            }
        }
        any
    };
    let amps = v.amplitudes_mut();
    if amps.len() >= PAR_MIN_LEN {
        amps.par_chunks_mut(PAR_MIN_LEN)
            .enumerate()
            .map(|(c, xs)| f(c * PAR_MIN_LEN, xs))
            .reduce(|| false, |a, b| a | b)
    } else {
        f(0, amps)
    }
}

/// Applies a local op; returns false only if a projector left exactly zero.
fn apply_local(op: &LocalOp, v: &mut StateVectorF64) -> bool {
    match op {
        LocalOp::Gate(g) => {
            v.apply_gate(g).expect("planned gate is valid");
            true
        }
        LocalOp::Mat(q, m) => {
            v.apply_1q_matrix(*q, m);
            true
        }
        LocalOp::Proj { mask, val } => project(v, *mask, *val, true),
        LocalOp::ProjNot { mask, val } => project(v, *mask, *val, false),
    }
}

fn apply_inverse(g: &Gate, v: &mut StateVectorF64) {
    v.apply_gate(&g.inverse()).expect("valid gate");
}

/// `out[i] = <bras[i] | psi>` in one blocked pass over `psi`.
fn multi_dot(bras: &[Vec<Complex64>], psi: &[Complex64], out: &mut [Complex64]) {
    const B: usize = 2048;
    let block = |start: usize, end: usize, acc: &mut [Complex64]| {
        let p = &psi[start..end];
        for (a, bra) in acc.iter_mut().zip(bras) {
            let b = &bra[start..end];
            let mut s = ZERO;
            for (x, y) in b.iter().zip(p) {
                s += x.conj() * y;
            }
            *a += s;
        }
    };
    out.iter_mut().for_each(|o| *o = ZERO);
    if psi.len() >= PAR_MIN_LEN {
        let nblk = psi.len().div_ceil(B);
        let r = (0..nblk)
            .into_par_iter()
            .fold(
                || vec![ZERO; bras.len()],
                |mut acc, i| {
                    block(i * B, ((i + 1) * B).min(psi.len()), &mut acc);
                    acc
                },
            )
            .reduce(
                || vec![ZERO; bras.len()],
                |mut a, b| {
                    a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                    a
                },
            );
        out.copy_from_slice(&r);
    } else {
        block(0, psi.len(), out);
    }
}

/// Hyperedges of the wire interaction graph after SWAP elimination, each
/// with the number of path-count bits it costs when split by a partition.
fn interaction_edges(c: &Circuit, opts: &HsfOptions) -> Result<Vec<(Vec<usize>, u32)>, SimError> {
    let mut map: HashMap<Vec<usize>, u32> = HashMap::new();
    for_each_wire_gate(c, opts, |g| {
        let w = cut_weight(&g, opts);
        if w > 0 {
            let mut qs = g.qubits();
            qs.sort_unstable();
            *map.entry(qs).or_default() += w;
        }
        Ok(())
    })?;
    let mut v: Vec<_> = map.into_iter().collect();
    v.sort();
    Ok(v)
}

fn split_cost(edges: &[(Vec<usize>, u32)], in_a: &[bool], ids: &[usize]) -> u32 {
    ids.iter()
        .map(|&e| {
            let (ws, w) = &edges[e];
            let s = in_a[ws[0]];
            if ws.iter().all(|&x| in_a[x] == s) {
                0
            } else {
                *w
            }
        })
        .sum()
}

/// Kernighan–Lin refinement of a partition with `|A|` fixed: passes of
/// tentative best pair swaps with locking, keeping the best prefix.
fn kernighan_lin(
    n: usize,
    edges: &[(Vec<usize>, u32)],
    inc: &[Vec<usize>],
    mut in_a: Vec<bool>,
) -> (Vec<bool>, u32) {
    let all: Vec<usize> = (0..edges.len()).collect();
    let mut cost = split_cost(edges, &in_a, &all);
    let na = in_a.iter().filter(|&&x| x).count();
    let steps = na.min(n - na);
    for _pass in 0..50 {
        let start_cost = cost;
        let mut cur = in_a.clone();
        let mut cur_cost = cost;
        let mut locked = vec![false; n];
        let mut best = (cost, in_a.clone());
        for _ in 0..steps {
            let mut pick: Option<(i64, usize, usize)> = None;
            let ca: Vec<usize> = (0..n).filter(|&a| cur[a] && !locked[a]).collect();
            let cb: Vec<usize> = (0..n).filter(|&b| !cur[b] && !locked[b]).collect();
            for &a in &ca {
                for &b in &cb {
                    let mut ids: Vec<usize> = inc[a].iter().chain(&inc[b]).cloned().collect();
                    ids.sort_unstable();
                    ids.dedup();
                    let before = split_cost(edges, &cur, &ids) as i64;
                    cur.swap(a, b);
                    let after = split_cost(edges, &cur, &ids) as i64;
                    cur.swap(a, b);
                    let d = after - before;
                    if pick.map_or(true, |(bd, _, _)| d < bd) {
                        pick = Some((d, a, b));
                    }
                }
            }
            let Some((d, a, b)) = pick else { break };
            cur.swap(a, b);
            locked[a] = true;
            locked[b] = true;
            cur_cost = (cur_cost as i64 + d) as u32;
            if cur_cost < best.0 {
                best = (cur_cost, cur.clone());
            }
        }
        if best.0 < start_cost {
            cost = best.0;
            in_a = best.1;
        } else {
            break;
        }
    }
    (in_a, cost)
}

/// Number of path-count bits (`log2` of the number of paths) a partition
/// costs for a circuit, without building the plan.
pub fn cut_bits(c: &Circuit, in_a: &[bool], opts: &HsfOptions) -> Result<u32, SimError> {
    let edges = interaction_edges(c, opts)?;
    let all: Vec<usize> = (0..edges.len()).collect();
    Ok(split_cost(&edges, in_a, &all))
}

/// Chooses a partition minimising the estimated cost
/// `2^(cut bits) · (2^|A| + 2^|B|)`: for each block size from balanced to
/// mildly unbalanced, Kernighan–Lin from several deterministic starts
/// (contiguous halves, a greedy BFS growth and seeded random splits).
pub fn auto_partition(c: &Circuit, opts: &HsfOptions) -> Result<Vec<bool>, SimError> {
    let n = c.num_qubits;
    let edges = interaction_edges(c, opts)?;
    if n <= 1 {
        return Ok(vec![true; n]);
    }
    let mut inc: Vec<Vec<usize>> = vec![vec![]; n];
    for (e, (ws, _)) in edges.iter().enumerate() {
        for &w in ws {
            inc[w].push(e);
        }
    }
    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(0x5eed);
    let mut best: Option<(f64, Vec<bool>)> = None;
    let half = n / 2;
    let max_block = 25.min(n - 1); // keep each block within the state-vector cap
    for na in (1..=half).rev() {
        let nb = n - na;
        if nb > max_block || nb > half + 4 {
            break;
        }
        let mut starts: Vec<Vec<bool>> = vec![
            (0..n).map(|w| w < na).collect(),
            (0..n).map(|w| w >= nb).collect(),
            (0..n).map(|w| w % 2 == 0 && w / 2 < na).collect(),
        ];
        // greedy growth from wire 0 by connection weight
        let mut g = vec![false; n];
        g[0] = true;
        for _ in 1..na {
            let mut score = vec![0i64; n];
            for (ws, w) in &edges {
                if ws.iter().any(|&x| g[x]) {
                    for &x in ws {
                        score[x] += *w as i64;
                    }
                }
            }
            let pick = (0..n)
                .filter(|&x| !g[x])
                .max_by_key(|&x| (score[x], -(x as i64)))
                .unwrap();
            g[pick] = true;
        }
        starts.push(g);
        for _ in 0..6 {
            let mut idx: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                idx.swap(i, rng.random_range(0..=i));
            }
            let mut s = vec![false; n];
            for &w in &idx[..na] {
                s[w] = true;
            }
            starts.push(s);
        }
        for s in starts {
            // fix the size of contiguous/even starts if they are off
            let cnt = s.iter().filter(|&&x| x).count();
            if cnt != na {
                continue;
            }
            let (p, bits) = kernighan_lin(n, &edges, &inc, s);
            let cost = 2f64.powi(bits as i32) * (2f64.powi(na as i32) + 2f64.powi(nb as i32));
            if best.as_ref().map_or(true, |(bc, _)| cost < *bc) {
                best = Some((cost, p));
            }
        }
    }
    Ok(best
        .map(|b| b.1)
        .unwrap_or_else(|| (0..n).map(|w| w < half).collect()))
}

/// A benchmark/test circuit: two blocks (`A` = qubits `0..na`) of random
/// single-qubit rotations and brickwork CZs, `depth` layers, joined by
/// `cross` gates (CZ, CNOT, CPhase cycling) whose layer is uniform
/// ("spread") or the middle layer (`middle = true`).
pub fn two_block_circuit<R: Rng + ?Sized>(
    n: usize,
    na: usize,
    depth: usize,
    cross: usize,
    middle: bool,
    rng: &mut R,
) -> Circuit {
    assert!(na >= 1 && na < n);
    let mut at: Vec<Vec<(usize, usize)>> = vec![vec![]; depth.max(1)];
    for i in 0..cross {
        let l = if middle {
            depth / 2
        } else {
            rng.random_range(0..depth.max(1))
        };
        let a = if middle {
            i % na
        } else {
            rng.random_range(0..na)
        };
        let b = if middle {
            na + i % (n - na)
        } else {
            rng.random_range(na..n)
        };
        at[l].push((a, b));
    }
    let mut c = Circuit::new(n);
    for (l, cross_here) in at.iter().enumerate() {
        for q in 0..n {
            let th = rng.random_range(-3.1..3.1);
            match rng.random_range(0..4) {
                0 => c.rx(q, th),
                1 => c.ry(q, th),
                2 => c.rz(q, th),
                _ => c.t(q),
            };
        }
        for (lo, hi) in [(0, na), (na, n)] {
            let mut q = lo + l % 2;
            while q + 1 < hi {
                c.cz(q, q + 1);
                q += 2;
            }
        }
        for (i, &(a, b)) in cross_here.iter().enumerate() {
            match i % 3 {
                0 => c.cz(a, b),
                1 => c.cnot(b, a),
                _ => c.cphase(a, b, 0.7),
            };
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kron_sum(terms: &[(Mat2, Mat2)]) -> Mat4 {
        let mut m = [[ZERO; 4]; 4];
        for (a, b) in terms {
            for ia in 0..2 {
                for ja in 0..2 {
                    for ib in 0..2 {
                        for jb in 0..2 {
                            m[2 * ia + ib][2 * ja + jb] += a[ia][ja] * b[ib][jb];
                        }
                    }
                }
            }
        }
        m
    }

    fn max_diff(a: &Mat4, b: &Mat4) -> f64 {
        let mut d = 0.0f64;
        for i in 0..4 {
            for j in 0..4 {
                d = d.max((a[i][j] - b[i][j]).norm());
            }
        }
        d
    }

    #[test]
    fn schmidt_ranks_of_standard_gates() {
        for (g, r) in [
            (Gate::Cz(0, 1), 2),
            (Gate::Cnot(0, 1), 2),
            (Gate::Cnot(1, 0), 2),
            (Gate::CPhase(0, 1, 0.3), 2),
            (Gate::CPhase(0, 1, 1e-6), 2),
            (Gate::CPhase(0, 1, 0.0), 1),
            (Gate::Swap(0, 1), 4),
        ] {
            let m = g.matrix_2q().unwrap();
            let t = operator_schmidt(&m);
            assert_eq!(t.len(), r, "{g:?}");
            assert!(max_diff(&kron_sum(&t), &m) < 1e-14, "{g:?}");
        }
    }

    #[test]
    fn schmidt_of_product_and_random_matrices() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        for _ in 0..50 {
            let mut m = [[ZERO; 4]; 4];
            for row in m.iter_mut() {
                for x in row.iter_mut() {
                    *x = Complex64::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0));
                }
            }
            let t = operator_schmidt(&m);
            assert_eq!(t.len(), 4);
            assert!(max_diff(&kron_sum(&t), &m) < 1e-13);
            // a product matrix has rank 1
            let p = kron_sum(&t[..1]);
            assert_eq!(operator_schmidt(&p).len(), 1);
        }
    }

    #[test]
    fn analytic_terms_reconstruct_gates() {
        // build the analytic terms for a 2-wire partition {0}|{1} and compare
        let side = |w: usize| w;
        let loc = |_w: usize| 0usize;
        for g in [
            Gate::Cz(0, 1),
            Gate::Cnot(0, 1),
            Gate::Cnot(1, 0),
            Gate::CPhase(1, 0, -0.4),
        ] {
            let terms = cut_terms(&g, &side, &loc, SchmidtMode::Analytic).unwrap();
            assert_eq!(terms.len(), 2);
            // embed as 4x4 in (wire0, wire1) order and compare with matrix_2q
            let to_mat = |op: &Option<LocalOp>| -> Mat2 {
                match op {
                    None => [[ONE, ZERO], [ZERO, ONE]],
                    Some(LocalOp::Gate(g)) => g.matrix_1q().unwrap(),
                    Some(LocalOp::Mat(_, m)) => *m,
                    Some(LocalOp::Proj { val, .. }) => {
                        if *val == 0 {
                            [[ONE, ZERO], [ZERO, ZERO]]
                        } else {
                            [[ZERO, ZERO], [ZERO, ONE]]
                        }
                    }
                    Some(LocalOp::ProjNot { .. }) => unreachable!(),
                }
            };
            let pairs: Vec<(Mat2, Mat2)> = terms
                .iter()
                .map(|t| (to_mat(&t[0]), to_mat(&t[1])))
                .collect();
            let mut want = g.matrix_2q().unwrap();
            if g.qubits()[0] == 1 {
                want = crate::gate::mat4_swap_qubits(&want);
            }
            assert!(max_diff(&kron_sum(&pairs), &want) < 1e-15, "{g:?}");
        }
    }
}
