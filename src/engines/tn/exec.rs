//! Execution of a contraction tree (research/simulability/tn.md §4): every
//! pairwise contraction is at most one permutation per input plus a
//! (batched) GEMM through faer, in complex f64 or f32. Sub-trees that do not
//! depend on a sliced index are contracted once and reused by every slice;
//! slices run in parallel (sequential GEMMs) when their working sets fit
//! the memory budget side by side, otherwise one after another with
//! parallel GEMMs and permutations.

use super::network::{Ix, Network};
use super::path::{tree_cost, ContractionTree, Hypergraph};
use crate::circuit::SimError;
use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use num_complex::{Complex, Complex64};
use rayon::prelude::*;
use std::time::Instant;

/// Complex scalar types the executor runs in.
pub trait Scalar:
    faer::traits::ComplexField
    + Copy
    + Send
    + Sync
    + std::fmt::Debug
    + std::ops::Add<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::AddAssign
    + 'static
{
    /// Bytes per entry.
    const BYTES: usize;
    /// Rounds a complex f64 to this type.
    fn from_c64(z: Complex64) -> Self;
    /// Widens to complex f64.
    fn to_c64(self) -> Complex64;
    /// Zero.
    fn zero() -> Self;
}

impl Scalar for Complex<f64> {
    const BYTES: usize = 16;
    fn from_c64(z: Complex64) -> Self {
        z
    }
    fn to_c64(self) -> Complex64 {
        self
    }
    fn zero() -> Self {
        Complex::new(0.0, 0.0)
    }
}

impl Scalar for Complex<f32> {
    const BYTES: usize = 8;
    fn from_c64(z: Complex64) -> Self {
        Complex::new(z.re as f32, z.im as f32)
    }
    fn to_c64(self) -> Complex64 {
        Complex64::new(self.re as f64, self.im as f64)
    }
    fn zero() -> Self {
        Complex::new(0.0, 0.0)
    }
}

/// Below this many entries permutations and GEMMs run on the calling thread.
const PAR_MIN: usize = 1 << 16;
/// Big-times-rank-2 contractions of at least this many entries use the
/// single-pass kernels of [`FastPlan`].
const FAST_MIN: usize = 1 << 10;

/// Copies `dst[j...] = src[base + Σ_j coord_j · sstr_j]` for the row-major
/// destination of shape `dims`.
fn gather<T: Copy + Send + Sync>(
    src: &[T],
    base: usize,
    dims: &[usize],
    sstr: &[usize],
    dst: &mut [T],
    par: bool,
) {
    let len: usize = dims.iter().product();
    debug_assert_eq!(dst.len(), len);
    if dims.is_empty() {
        dst[0] = src[base];
        return;
    }
    // inner block: trailing axes with product <= 4096
    let r = dims.len();
    let mut j0 = r;
    let mut inner = 1usize;
    while j0 > 0 && inner * dims[j0 - 1] <= 4096 {
        j0 -= 1;
        inner *= dims[j0];
    }
    if j0 == r {
        // the last axis alone is larger than 4096
        j0 = r - 1;
        inner = dims[r - 1];
    }
    let mut table = vec![0usize; inner];
    {
        let idims = &dims[j0..];
        let istr = &sstr[j0..];
        let mut coord = vec![0usize; idims.len()];
        let mut off = 0usize;
        for t in table.iter_mut() {
            *t = off;
            let mut k = idims.len();
            while k > 0 {
                k -= 1;
                coord[k] += 1;
                off += istr[k];
                if coord[k] < idims[k] {
                    break;
                }
                off -= istr[k] * idims[k];
                coord[k] = 0;
            }
        }
    }
    let contiguous = table.iter().enumerate().all(|(t, &o)| o == t);
    let odims = &dims[..j0];
    let ostr = &sstr[..j0];
    let outer = len / inner;
    let run = |o_start: usize, chunk: &mut [T]| {
        // decode starting coordinates of outer index o_start
        let mut coord = vec![0usize; odims.len()];
        let mut rem = o_start;
        let mut off = base;
        for k in (0..odims.len()).rev() {
            coord[k] = rem % odims[k];
            rem /= odims[k];
            off += coord[k] * ostr[k];
        }
        for blk in chunk.chunks_mut(inner) {
            if contiguous {
                blk.copy_from_slice(&src[off..off + inner]);
            } else {
                for (d, &t) in blk.iter_mut().zip(&table) {
                    *d = src[off + t];
                }
            }
            let mut k = odims.len();
            while k > 0 {
                k -= 1;
                coord[k] += 1;
                off += ostr[k];
                if coord[k] < odims[k] {
                    break;
                }
                off -= ostr[k] * odims[k];
                coord[k] = 0;
            }
        }
    };
    if par && len >= PAR_MIN && outer > 1 {
        let per = (outer / (rayon::current_num_threads() * 4)).max(1);
        dst.par_chunks_mut(per * inner)
            .enumerate()
            .for_each(|(c, chunk)| run(c * per, chunk));
    } else {
        run(0, dst);
    }
}

fn row_major_strides(dims: &[usize]) -> Vec<usize> {
    let mut s = vec![0; dims.len()];
    let mut acc = 1;
    for k in (0..dims.len()).rev() {
        s[k] = acc;
        acc *= dims[k];
    }
    s
}

/// One pairwise contraction, planned once for every slice.
#[derive(Clone, Debug)]
struct PairPlan {
    /// Node ids of the GEMM's left (A) and right (B) inputs.
    a: u32,
    b: u32,
    /// Positions of A's / B's layout to sum out first (indices nobody else holds).
    a_sum: Vec<usize>,
    b_sum: Vec<usize>,
    /// Permutation of the (summed) input into the GEMM layout, if needed.
    a_perm: Option<Vec<usize>>,
    b_perm: Option<Vec<usize>>,
    /// Layout dims of the inputs before summing.
    a_full_dims: Vec<usize>,
    b_full_dims: Vec<usize>,
    /// Layout dims of the (summed) inputs before permutation.
    a_dims: Vec<usize>,
    b_dims: Vec<usize>,
    nbatch: usize,
    m: usize,
    k: usize,
    n: usize,
    /// A is laid out `[batch, k, m]` instead of `[batch, m, k]`.
    a_kfirst: bool,
    /// B is laid out `[batch, k, n]` (true) or `[batch, n, k]`.
    b_kfirst: bool,
    /// Loop-over-GEMM plan (no permutation); used instead of the fields
    /// above when set.
    lp: Option<LoopPlan>,
    /// Single-pass kernel for a big tensor times a rank-1/2 tensor; used
    /// instead of everything above when set.
    fast: Option<FastPlan>,
}

/// A big tensor A (layout `la`) times a small tensor B with at most two
/// dimension-2 indices: one streaming pass over A, no GEMM, no permutation.
#[derive(Clone, Copy, Debug)]
enum FastPlan {
    /// `B[x]`, `x` kept: `out = A · B[x]` (layout unchanged).
    Diag1 { sx: usize },
    /// `B[x]`, `x` summed: `out = Σ_x A · B[x]` (axis `x` removed).
    Reduce1 { sx: usize },
    /// `B[x, y]`, both kept: `out = A · B[x, y]` (layout unchanged);
    /// `bx`, `by` are B's strides of `x` and `y`.
    Diag2 {
        sx: usize,
        sy: usize,
        bx: usize,
        by: usize,
    },
    /// `B[x, y]`, `x` summed, `y` kept and in A: axis `x` removed.
    Reduce2 {
        sx: usize,
        sy: usize,
        bx: usize,
        by: usize,
    },
    /// `B[x, y]`, `x` summed, `y` new: `y` replaces `x` in place (a gate).
    Gate { sx: usize, bx: usize, by: usize },
    /// `B[x, y]`, `x` kept, `y` new: `y` appended as the fastest axis.
    Expand { sx: usize, bx: usize, by: usize },
}

/// Classifies a big-times-small contraction for [`FastPlan`]; returns the
/// plan and the output layout.
fn fast_plan(
    la: &[Ix],
    lb: &[Ix],
    keep: &[Ix],
    dim: &dyn Fn(Ix) -> usize,
) -> Option<(FastPlan, Vec<Ix>)> {
    if lb.is_empty() || lb.len() > 2 || lb.iter().any(|&i| dim(i) != 2) {
        return None;
    }
    let dims_a: Vec<usize> = la.iter().map(|&i| dim(i)).collect();
    let sa = row_major_strides(&dims_a);
    let pos = |i: Ix| la.iter().position(|&x| x == i);
    let kept = |i: Ix| keep.contains(&i);
    let without = |x: Ix| la.iter().copied().filter(|&i| i != x).collect::<Vec<_>>();
    if lb.len() == 1 {
        let x = lb[0];
        let px = pos(x)?;
        return Some(if kept(x) {
            (FastPlan::Diag1 { sx: sa[px] }, la.to_vec())
        } else {
            (FastPlan::Reduce1 { sx: sa[px] }, without(x))
        });
    }
    // B laid out [u, v]: B[u][v] at 2u + v
    for (x, y, bx, by) in [(lb[0], lb[1], 2usize, 1usize), (lb[1], lb[0], 1, 2)] {
        match (pos(x), pos(y)) {
            (Some(px), Some(py)) => {
                if kept(x) && kept(y) {
                    return Some((
                        FastPlan::Diag2 {
                            sx: sa[px],
                            sy: sa[py],
                            bx,
                            by,
                        },
                        la.to_vec(),
                    ));
                }
                if !kept(x) && kept(y) {
                    return Some((
                        FastPlan::Reduce2 {
                            sx: sa[px],
                            sy: sa[py],
                            bx,
                            by,
                        },
                        without(x),
                    ));
                }
            }
            (Some(px), None) => {
                if kept(x) {
                    let mut out = la.to_vec();
                    out.push(y);
                    return Some((FastPlan::Expand { sx: sa[px], bx, by }, out));
                }
                let out = la.iter().map(|&i| if i == x { y } else { i }).collect();
                return Some((FastPlan::Gate { sx: sa[px], bx, by }, out));
            }
            _ => {}
        }
    }
    None
}

/// Runs a [`FastPlan`]: `a` is the big input, `b` the small one.
fn run_fast<T: Scalar>(f: &FastPlan, a: &[T], b: &[T], out: &mut [T], par: bool) {
    let threads = rayon::current_num_threads();
    let len = a.len();
    // splits [0, n) into chunks aligned to `align` for the closure
    let chunked =
        |n: usize, align: usize, out: &mut [T], per: &(dyn Fn(usize, &mut [T]) + Sync)| {
            if par && n >= PAR_MIN {
                let c = (n / (4 * threads)).max(align).div_ceil(align) * align;
                out.par_chunks_mut(c)
                    .enumerate()
                    .for_each(|(k, o)| per(k * c, o));
            } else {
                per(0, out);
            }
        };
    match *f {
        FastPlan::Diag1 { sx } => {
            let per = |st: usize, o: &mut [T]| {
                for (j, z) in o.iter_mut().enumerate() {
                    let i = st + j;
                    *z = a[i] * b[(i / sx) & 1];
                }
            };
            chunked(len, 1, out, &per);
        }
        FastPlan::Diag2 { sx, sy, bx, by } => {
            let per = |st: usize, o: &mut [T]| {
                for (j, z) in o.iter_mut().enumerate() {
                    let i = st + j;
                    *z = a[i] * b[((i / sx) & 1) * bx + ((i / sy) & 1) * by];
                }
            };
            chunked(len, 1, out, &per);
        }
        FastPlan::Reduce1 { sx } => {
            // out index j <-> A index with a 0 inserted at x
            let per = |st: usize, o: &mut [T]| {
                for (j, z) in o.iter_mut().enumerate() {
                    let jj = st + j;
                    let i0 = (jj / sx) * 2 * sx + jj % sx;
                    *z = a[i0] * b[0] + a[i0 + sx] * b[1];
                }
            };
            chunked(len / 2, 1, out, &per);
        }
        FastPlan::Reduce2 { sx, sy, bx, by } => {
            let per = |st: usize, o: &mut [T]| {
                for (j, z) in o.iter_mut().enumerate() {
                    let jj = st + j;
                    let i0 = (jj / sx) * 2 * sx + jj % sx;
                    let yv = ((i0 / sy) & 1) * by;
                    *z = a[i0] * b[yv] + a[i0 + sx] * b[bx + yv];
                }
            };
            chunked(len / 2, 1, out, &per);
        }
        FastPlan::Gate { sx, bx, by } => {
            // out[.. y ..] = Σ_x B[x, y] A[.. x ..]; bxy = B[x, y]
            let (b00, b10, b01, b11) = (b[0], b[bx], b[by], b[bx + by]);
            let per = |st: usize, o: &mut [T]| {
                for (j, z) in o.iter_mut().enumerate() {
                    let i = st + j;
                    let y = (i / sx) & 1;
                    let i0 = i - y * sx;
                    let (a0, a1) = (a[i0], a[i0 + sx]);
                    // B[x=0, y] a0 + B[x=1, y] a1
                    *z = if y == 0 {
                        b00 * a0 + b10 * a1
                    } else {
                        b01 * a0 + b11 * a1
                    };
                }
            };
            chunked(len, 1, out, &per);
        }
        FastPlan::Expand { sx, bx, by } => {
            let per = |st: usize, o: &mut [T]| {
                for (j, z) in o.iter_mut().enumerate() {
                    let k = st + j;
                    let (i, y) = (k >> 1, k & 1);
                    *z = a[i] * b[((i / sx) & 1) * bx + y * by];
                }
            };
            chunked(2 * len, 2, out, &per);
        }
    }
}

/// A contraction as strided GEMM blocks: rows = one run of A's free
/// indices, inner = one run of shared summed indices (contiguous and in the
/// same order in both inputs), cols = one run of B's free indices; every
/// other index is an explicit loop (summed loops accumulate). The output is
/// laid out `[loop indices, rows, cols]`.
#[derive(Clone, Debug)]
struct LoopPlan {
    rows: usize,
    rs_a: usize,
    inner: usize,
    ks_a: usize,
    ks_b: usize,
    cols: usize,
    cs_b: usize,
    /// Output loops: `(dim, stride in A, stride in B)`, slowest first.
    lo: Vec<(usize, usize, usize)>,
    /// Summed loops: `(dim, stride in A, stride in B)`.
    lk: Vec<(usize, usize, usize)>,
}

impl LoopPlan {
    fn calls(&self) -> usize {
        self.lo.iter().chain(&self.lk).map(|x| x.0).product()
    }
}

/// How pairwise contractions are executed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairStrategy {
    /// Per contraction: strided GEMM blocks when they are large enough,
    /// otherwise permute into a batched GEMM.
    Auto,
    /// Always permute the inputs into batched-GEMM layout.
    Permute,
    /// Always loop over strided GEMM blocks (no permutation).
    Loops,
}

/// Builds the loop-over-GEMM plan for inputs laid out `la`, `lb` (summed
/// single-side indices already removed) and the kept index set; returns it
/// with the output layout.
fn loop_plan(la: &[Ix], lb: &[Ix], keep: &[Ix], dim: &dyn Fn(Ix) -> usize) -> (LoopPlan, Vec<Ix>) {
    #[derive(Clone, Copy, PartialEq)]
    enum C {
        Batch,
        K,
        M,
        N,
    }
    let class_a: Vec<C> = la
        .iter()
        .map(|i| {
            if lb.contains(i) {
                if keep.contains(i) {
                    C::Batch
                } else {
                    C::K
                }
            } else {
                C::M
            }
        })
        .collect();
    let class_b: Vec<C> = lb
        .iter()
        .map(|i| {
            if la.contains(i) {
                if keep.contains(i) {
                    C::Batch
                } else {
                    C::K
                }
            } else {
                C::N
            }
        })
        .collect();
    let dims_a: Vec<usize> = la.iter().map(|&i| dim(i)).collect();
    let dims_b: Vec<usize> = lb.iter().map(|&i| dim(i)).collect();
    let sa = row_major_strides(&dims_a);
    let sb = row_major_strides(&dims_b);
    // largest run of consecutive positions of class `c` in `cls`
    let best_run = |cls: &[C], dims: &[usize], c: C| -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize, usize)> = None; // (size, start, end)
        let mut p = 0;
        while p < cls.len() {
            if cls[p] != c {
                p += 1;
                continue;
            }
            let st = p;
            let mut size = 1;
            while p < cls.len() && cls[p] == c {
                size *= dims[p];
                p += 1;
            }
            if best.is_none_or(|b| size >= b.0) {
                best = Some((size, st, p));
            }
        }
        best.map(|b| (b.1, b.2))
    };
    let r_run = best_run(&class_a, &dims_a, C::M);
    let n_run = best_run(&class_b, &dims_b, C::N);
    // inner run: consecutive K positions in A that are also consecutive, in
    // the same order, in B
    let pos_b = |i: Ix| lb.iter().position(|&x| x == i).unwrap();
    let mut k_run: Option<(usize, usize, usize)> = None; // (size, start, end) in A
    for st in 0..la.len() {
        if class_a[st] != C::K {
            continue;
        }
        let mut en = st + 1;
        let mut size = dims_a[st];
        while en < la.len() && class_a[en] == C::K && pos_b(la[en]) == pos_b(la[en - 1]) + 1 {
            size *= dims_a[en];
            en += 1;
        }
        if k_run.is_none_or(|b| size > b.0) {
            k_run = Some((size, st, en));
        }
    }
    let in_run = |p: usize, run: Option<(usize, usize)>| run.is_some_and(|(a, b)| p >= a && p < b);
    let kr = k_run.map(|(_, a, b)| (a, b));
    let mut lo = Vec::new();
    let mut out: Vec<Ix> = Vec::new();
    for (p, &i) in la.iter().enumerate() {
        if matches!(class_a[p], C::Batch) || (class_a[p] == C::M && !in_run(p, r_run)) {
            let q = lb.iter().position(|&x| x == i);
            lo.push((dims_a[p], sa[p], q.map_or(0, |q| sb[q])));
            out.push(i);
        }
    }
    for (q, &i) in lb.iter().enumerate() {
        if class_b[q] == C::N && !in_run(q, n_run) {
            lo.push((dims_b[q], 0, sb[q]));
            out.push(i);
        }
    }
    let mut lk = Vec::new();
    for (p, &i) in la.iter().enumerate() {
        if class_a[p] == C::K && !in_run(p, kr) {
            lk.push((dims_a[p], sa[p], sb[pos_b(i)]));
        }
    }
    let (rows, rs_a) = match r_run {
        Some((a, b)) => {
            out.extend_from_slice(&la[a..b]);
            (dims_a[a..b].iter().product(), sa[b - 1])
        }
        None => (1, 0),
    };
    let (inner, ks_a, ks_b) = match kr {
        Some((a, b)) => (
            dims_a[a..b].iter().product(),
            sa[b - 1],
            sb[pos_b(la[b - 1])],
        ),
        None => (1, 0, 0),
    };
    let (cols, cs_b) = match n_run {
        Some((a, b)) => {
            out.extend_from_slice(&lb[a..b]);
            (dims_b[a..b].iter().product(), sb[b - 1])
        }
        None => (1, 0),
    };
    (
        LoopPlan {
            rows,
            rs_a,
            inner,
            ks_a,
            ks_b,
            cols,
            cs_b,
            lo,
            lk,
        },
        out,
    )
}

/// A compiled contraction: layouts and pair plans for every node.
pub(crate) struct ExecPlan {
    n_leaves: usize,
    /// Layout (index ids, slowest first) of every node's tensor in a slice.
    layout: Vec<Vec<Ix>>,
    /// Entries of every node's tensor in a slice.
    len: Vec<usize>,
    /// Pair plan of every internal node.
    pair: Vec<Option<PairPlan>>,
    /// Internal nodes in post-order.
    order: Vec<u32>,
    /// Node depends on a sliced index.
    variant: Vec<bool>,
    /// Invariant node whose result must be kept for the slices.
    keep_inv: Vec<bool>,
    root: u32,
    /// Sliced indices and their dimensions.
    sliced: Vec<(Ix, usize)>,
    /// Final permutation of the root layout into the network's output order.
    out_perm: Vec<usize>,
    out_dims: Vec<usize>,
    /// Bytes of one slice's peak working set and of the invariant cache.
    pub(crate) slice_peak_elems: usize,
    pub(crate) cache_elems: usize,
}

fn same_set(a: &[Ix], b: &[Ix]) -> bool {
    a.len() == b.len() && a.iter().all(|x| b.contains(x))
}

/// `Some(kfirst)` when `lay` is `[batch (exactly in that order), then the
/// two blocks `ms` and `ks` contiguous in either order]` and, when `korder`
/// is given, the `ks` block is in exactly that order.
fn gemm_form(
    lay: &[Ix],
    batch: &[Ix],
    ms: &[Ix],
    ks: &[Ix],
    korder: Option<&[Ix]>,
) -> Option<bool> {
    let nb = batch.len();
    if lay.len() != nb + ms.len() + ks.len() || lay[..nb] != *batch {
        return None;
    }
    let rest = &lay[nb..];
    let (nm, nk) = (ms.len(), ks.len());
    let try_form = |kfirst: bool| -> bool {
        let (kb, mb) = if kfirst {
            (&rest[..nk], &rest[nk..])
        } else {
            (&rest[nm..], &rest[..nm])
        };
        same_set(kb, ks) && same_set(mb, ms) && korder.is_none_or(|o| kb == o)
    };
    if try_form(false) {
        Some(false)
    } else if try_form(true) {
        Some(true)
    } else {
        None
    }
}

impl ExecPlan {
    /// Plans the contraction of `nw` along `tree` with `sliced` indices.
    pub(crate) fn new(
        nw: &Network,
        tree: &ContractionTree,
        sliced: &[bool],
        strategy: PairStrategy,
    ) -> Self {
        let hg = Hypergraph::from_network(nw);
        let tc = tree_cost(&hg, tree, &[]);
        let nl = tree.n_leaves;
        let nn = nl + tree.children.len();
        let is_sl = |i: Ix| sliced.get(i as usize).copied().unwrap_or(false);
        let dim = |i: Ix| nw.dims[i as usize];
        let mut layout: Vec<Vec<Ix>> = vec![Vec::new(); nn];
        let mut len = vec![1usize; nn];
        let mut variant = vec![false; nn];
        for l in 0..nl {
            layout[l] = nw.tensors[l]
                .inds
                .iter()
                .copied()
                .filter(|&i| !is_sl(i))
                .collect();
            len[l] = layout[l].iter().map(|&i| dim(i)).product();
            variant[l] = nw.tensors[l].inds.iter().any(|&i| is_sl(i));
        }
        let order = tree.post_order();
        let mut pair: Vec<Option<PairPlan>> = vec![None; nn];
        for &v in &order {
            let [x, y] = tree.kids(v).unwrap();
            variant[v as usize] = variant[x as usize] || variant[y as usize];
            let keep: Vec<Ix> = tc.inds[v as usize]
                .iter()
                .copied()
                .filter(|&i| !is_sl(i))
                .collect();
            // A = the larger input (avoids permuting it when possible)
            let (a, b) = if len[x as usize] >= len[y as usize] {
                (x, y)
            } else {
                (y, x)
            };
            let (la, lb) = (layout[a as usize].clone(), layout[b as usize].clone());
            let a_full_dims: Vec<usize> = la.iter().map(|&i| dim(i)).collect();
            let b_full_dims: Vec<usize> = lb.iter().map(|&i| dim(i)).collect();
            let a_sum: Vec<usize> = (0..la.len())
                .filter(|&p| !lb.contains(&la[p]) && !keep.contains(&la[p]))
                .collect();
            let b_sum: Vec<usize> = (0..lb.len())
                .filter(|&p| !la.contains(&lb[p]) && !keep.contains(&lb[p]))
                .collect();
            let la: Vec<Ix> = (0..la.len())
                .filter(|p| !a_sum.contains(p))
                .map(|p| la[p])
                .collect();
            let lb: Vec<Ix> = (0..lb.len())
                .filter(|p| !b_sum.contains(p))
                .map(|p| lb[p])
                .collect();
            if strategy == PairStrategy::Auto && a_sum.is_empty() && len[a as usize] >= FAST_MIN {
                if let Some((fp, out)) = fast_plan(&la, &lb, &keep, &dim) {
                    debug_assert!(
                        same_set(&out, &keep),
                        "fast layout {out:?} vs keep {keep:?}"
                    );
                    len[v as usize] = out.iter().map(|&i| dim(i)).product();
                    layout[v as usize] = out;
                    pair[v as usize] = Some(PairPlan {
                        a,
                        b,
                        a_full_dims,
                        b_full_dims,
                        a_dims: la.iter().map(|&i| dim(i)).collect(),
                        b_dims: lb.iter().map(|&i| dim(i)).collect(),
                        a_sum,
                        b_sum,
                        a_perm: None,
                        b_perm: None,
                        nbatch: 1,
                        m: 1,
                        k: 1,
                        n: 1,
                        a_kfirst: false,
                        b_kfirst: false,
                        lp: None,
                        fast: Some(fp),
                    });
                    continue;
                }
            }
            if strategy != PairStrategy::Permute {
                let (lp, out) = loop_plan(&la, &lb, &keep, &dim);
                let work = lp.rows * lp.inner * lp.cols;
                let use_loops = strategy == PairStrategy::Loops
                    || lp.calls() <= 16
                    || (work >= 256 && lp.rows * lp.cols >= 8);
                if use_loops {
                    debug_assert!(
                        same_set(&out, &keep),
                        "loop layout {out:?} vs keep {keep:?}"
                    );
                    len[v as usize] = out.iter().map(|&i| dim(i)).product();
                    layout[v as usize] = out;
                    pair[v as usize] = Some(PairPlan {
                        a,
                        b,
                        a_full_dims,
                        b_full_dims,
                        a_dims: la.iter().map(|&i| dim(i)).collect(),
                        b_dims: lb.iter().map(|&i| dim(i)).collect(),
                        a_sum,
                        b_sum,
                        a_perm: None,
                        b_perm: None,
                        nbatch: 1,
                        m: 1,
                        k: 1,
                        n: 1,
                        a_kfirst: false,
                        b_kfirst: false,
                        lp: Some(lp),
                        fast: None,
                    });
                    continue;
                }
            }
            let shared: Vec<Ix> = la.iter().copied().filter(|i| lb.contains(i)).collect();
            let batch_set: Vec<Ix> = shared
                .iter()
                .copied()
                .filter(|i| keep.contains(i))
                .collect();
            let ks: Vec<Ix> = shared
                .iter()
                .copied()
                .filter(|i| !keep.contains(i))
                .collect();
            let ms: Vec<Ix> = la.iter().copied().filter(|i| !lb.contains(i)).collect();
            let ns: Vec<Ix> = lb.iter().copied().filter(|i| !la.contains(i)).collect();
            // A's form: batch first in A's order
            let a_batch: Vec<Ix> = la
                .iter()
                .copied()
                .filter(|i| batch_set.contains(i))
                .collect();
            let (a_target, a_kfirst, a_perm) = match gemm_form(&la, &a_batch, &ms, &ks, None) {
                Some(kf) => (la.clone(), kf, None),
                None => {
                    let mut t = a_batch.clone();
                    t.extend(&ms);
                    t.extend(&ks);
                    let perm = t
                        .iter()
                        .map(|i| la.iter().position(|x| x == i).unwrap())
                        .collect();
                    (t, false, Some(perm))
                }
            };
            let nb = a_batch.len();
            let a_k: Vec<Ix> = if a_kfirst {
                a_target[nb..nb + ks.len()].to_vec()
            } else {
                a_target[nb + ms.len()..].to_vec()
            };
            let a_m: Vec<Ix> = if a_kfirst {
                a_target[nb + ks.len()..].to_vec()
            } else {
                a_target[nb..nb + ms.len()].to_vec()
            };
            let (b_target, b_kfirst, b_perm) = match gemm_form(&lb, &a_batch, &ns, &ks, Some(&a_k))
            {
                // gemm_form's bool is "k block first"
                Some(kf) => (lb.clone(), kf, None),
                None => {
                    let mut t = a_batch.clone();
                    t.extend(&a_k);
                    t.extend(&ns);
                    let perm = t
                        .iter()
                        .map(|i| lb.iter().position(|x| x == i).unwrap())
                        .collect();
                    (t, true, Some(perm))
                }
            };
            let b_n: Vec<Ix> = if b_kfirst {
                b_target[nb + ks.len()..].to_vec()
            } else {
                b_target[nb..nb + ns.len()].to_vec()
            };
            let prod = |v: &[Ix]| v.iter().map(|&i| dim(i)).product::<usize>();
            let mut out = a_batch.clone();
            out.extend(&a_m);
            out.extend(&b_n);
            debug_assert!(same_set(&out, &keep), "layout {out:?} vs keep {keep:?}");
            len[v as usize] = prod(&out);
            layout[v as usize] = out;
            pair[v as usize] = Some(PairPlan {
                a,
                b,
                a_full_dims,
                b_full_dims,
                a_dims: la.iter().map(|&i| dim(i)).collect(),
                b_dims: lb.iter().map(|&i| dim(i)).collect(),
                a_sum,
                b_sum,
                a_perm,
                b_perm,
                nbatch: prod(&a_batch),
                m: prod(&ms),
                k: prod(&ks),
                n: prod(&ns),
                a_kfirst,
                b_kfirst,
                lp: None,
                fast: None,
            });
        }
        let root = tree.root();
        let mut keep_inv = vec![false; nn];
        for &v in &order {
            if variant[v as usize] {
                let [x, y] = tree.kids(v).unwrap();
                for c in [x, y] {
                    if !variant[c as usize] && c as usize >= nl {
                        keep_inv[c as usize] = true;
                    }
                }
            }
        }
        if !variant[root as usize] && root as usize >= nl {
            keep_inv[root as usize] = true;
        }
        let rl = &layout[root as usize];
        let out_perm: Vec<usize> = nw
            .output
            .iter()
            .map(|i| {
                rl.iter()
                    .position(|x| x == i)
                    .expect("output index at the root")
            })
            .collect();
        let out_dims = nw.output.iter().map(|&i| dim(i)).collect();
        let sl: Vec<(Ix, usize)> = (0..nw.dims.len())
            .filter(|&i| is_sl(i as Ix))
            .map(|i| (i as Ix, nw.dims[i]))
            .collect();
        let mut plan = ExecPlan {
            n_leaves: nl,
            layout,
            len,
            pair,
            order,
            variant,
            keep_inv,
            root,
            sliced: sl,
            out_perm,
            out_dims,
            slice_peak_elems: 0,
            cache_elems: 0,
        };
        plan.estimate_memory(tree);
        plan
    }

    /// Peak entries alive while one slice runs (results waiting for their
    /// sibling, plus the current inputs, permutation scratch and output),
    /// and entries of the invariant cache.
    fn estimate_memory(&mut self, tree: &ContractionTree) {
        let mut live = 0usize;
        let mut peak = 0usize;
        let mut cache = 0usize;
        for &v in &self.order {
            if self.keep_inv[v as usize] {
                cache += self.len[v as usize];
            }
        }
        // the two permutation scratch buffers persist at their largest size
        let (mut sa, mut sb) = (0usize, 0usize);
        for &v in &self.order {
            if !self.variant[v as usize] {
                continue;
            }
            let p = self.pair[v as usize].as_ref().unwrap();
            let [x, y] = tree.kids(v).unwrap();
            let mut need = self.len[v as usize];
            if p.a_perm.is_some() {
                sa = sa.max(self.len[p.a as usize]);
            }
            if p.b_perm.is_some() {
                sb = sb.max(self.len[p.b as usize]);
            }
            for c in [p.a, p.b] {
                if (c as usize) < self.n_leaves && self.variant[c as usize] {
                    need += self.len[c as usize]; // sliced leaf copy
                }
            }
            peak = peak.max(live + need);
            for c in [x, y] {
                if c as usize >= self.n_leaves && self.variant[c as usize] {
                    live -= self.len[c as usize];
                }
            }
            live += self.len[v as usize];
        }
        // freed buffers kept by the pool are capped at the live peak
        self.slice_peak_elems = 2 * peak.max(self.len[self.root as usize]) + sa + sb;
        self.cache_elems = cache;
    }

    fn n_slices(&self) -> usize {
        self.sliced.iter().map(|&(_, d)| d).product()
    }
}

/// Free-list of buffers, holding at most `limit` entries in total.
struct Pool<T> {
    free: Vec<Vec<T>>,
    limit: usize,
}

impl<T: Scalar> Pool<T> {
    fn new(limit: usize) -> Self {
        Pool {
            free: Vec::new(),
            limit,
        }
    }
    fn get(&mut self, len: usize) -> Vec<T> {
        // smallest free buffer that is large enough
        let mut best: Option<usize> = None;
        for (k, b) in self.free.iter().enumerate() {
            if b.capacity() >= len && best.is_none_or(|j| self.free[j].capacity() > b.capacity()) {
                best = Some(k);
            }
        }
        let mut v = match best {
            Some(k) => self.free.swap_remove(k),
            None => Vec::with_capacity(len),
        };
        if v.len() > len {
            v.truncate(len);
        } else {
            v.resize(len, T::zero());
        }
        v
    }
    fn put(&mut self, v: Vec<T>) {
        self.free.push(v);
        let mut held: usize = self.free.iter().map(|b| b.capacity()).sum();
        while held > self.limit {
            // drop the smallest buffer
            let k = (0..self.free.len())
                .min_by_key(|&k| self.free[k].capacity())
                .expect("non-empty");
            held -= self.free[k].capacity();
            self.free.swap_remove(k);
        }
    }
}

/// Sums positions `sum` out of a row-major tensor of shape `dims`.
fn sum_out<T: Scalar>(src: &[T], dims: &[usize], sum: &[usize]) -> Vec<T> {
    let keep: Vec<usize> = (0..dims.len()).filter(|p| !sum.contains(p)).collect();
    let kd: Vec<usize> = keep.iter().map(|&p| dims[p]).collect();
    let n: usize = kd.iter().product();
    let ss = row_major_strides(dims);
    let mut out = vec![T::zero(); n];
    let sd: Vec<usize> = sum.iter().map(|&p| dims[p]).collect();
    let ns: usize = sd.iter().product();
    let kstr: Vec<usize> = keep.iter().map(|&p| ss[p]).collect();
    let sstr: Vec<usize> = sum.iter().map(|&p| ss[p]).collect();
    for t in 0..ns {
        let mut base = 0;
        let mut r = t;
        for q in (0..sd.len()).rev() {
            base += (r % sd[q]) * sstr[q];
            r /= sd[q];
        }
        let mut tmp = vec![T::zero(); n];
        gather(src, base, &kd, &kstr, &mut tmp, false);
        for (o, x) in out.iter_mut().zip(tmp) {
            *o += x;
        }
    }
    out
}

/// One strided GEMM block `c (+)= A[oa..] · B[ob..]` of a loop plan.
#[allow(clippy::too_many_arguments)]
fn gemm_block<T: Scalar>(
    lp: &LoopPlan,
    a: &[T],
    oa: usize,
    b: &[T],
    ob: usize,
    c: &mut [T],
    first: bool,
    par: Par,
) {
    let (m, k, n) = (lp.rows, lp.inner, lp.cols);
    if m * k * n <= 64 {
        for r in 0..m {
            for j in 0..n {
                let mut acc = if first { T::zero() } else { c[r * n + j] };
                for kk in 0..k {
                    acc += a[oa + r * lp.rs_a + kk * lp.ks_a] * b[ob + kk * lp.ks_b + j * lp.cs_b];
                }
                c[r * n + j] = acc;
            }
        }
        return;
    }
    let one = T::from_c64(Complex64::new(1.0, 0.0));
    let par = if m * k * n >= PAR_MIN { par } else { Par::Seq };
    // SAFETY: every element addressed below lies inside `a`, `b` and `c`
    // (the strides and extents come from the row-major layouts of the inputs).
    let am = unsafe {
        MatRef::from_raw_parts(a.as_ptr().add(oa), m, k, lp.rs_a as isize, lp.ks_a as isize)
    };
    let bm = unsafe {
        MatRef::from_raw_parts(b.as_ptr().add(ob), k, n, lp.ks_b as isize, lp.cs_b as isize)
    };
    let cm = unsafe { MatMut::from_raw_parts_mut(c.as_mut_ptr(), m, n, n as isize, 1) };
    let acc = if first { Accum::Replace } else { Accum::Add };
    matmul(cm, acc, am, bm, one, par);
}

/// Executes a loop plan into `out` (laid out `[loops, rows, cols]`).
fn run_loops<T: Scalar>(lp: &LoopPlan, a: &[T], b: &[T], out: &mut [T], par: Par, par_loops: bool) {
    let blk = lp.rows * lp.cols;
    let n_lo: usize = lp.lo.iter().map(|x| x.0).product();
    let n_lk: usize = lp.lk.iter().map(|x| x.0).product();
    let body = |i: usize, c: &mut [T], gpar: Par| {
        let (mut oa, mut ob) = (0usize, 0usize);
        let mut r = i;
        for &(d, sa, sb) in lp.lo.iter().rev() {
            let x = r % d;
            r /= d;
            oa += x * sa;
            ob += x * sb;
        }
        for kk in 0..n_lk {
            let (mut ka, mut kb) = (oa, ob);
            let mut r = kk;
            for &(d, sa, sb) in lp.lk.iter().rev() {
                let x = r % d;
                r /= d;
                ka += x * sa;
                kb += x * sb;
            }
            gemm_block(lp, a, ka, b, kb, c, kk == 0, gpar);
        }
    };
    let threads = rayon::current_num_threads();
    if par_loops && n_lo >= 4 * threads && out.len() >= PAR_MIN {
        out.par_chunks_mut(blk)
            .enumerate()
            .for_each(|(i, c)| body(i, c, Par::Seq));
    } else {
        for (i, c) in out.chunks_mut(blk).enumerate() {
            body(i, c, par);
        }
    }
}

struct Worker<T> {
    pool: Pool<T>,
    scratch_a: Vec<T>,
    scratch_b: Vec<T>,
}

impl ExecPlan {
    /// Leaf `l`'s entries for slice values `vals` (one per sliced index).
    fn leaf_slice<T: Scalar>(
        &self,
        nw: &Network,
        leaf_full: &[Vec<T>],
        l: usize,
        vals: &[usize],
        out: &mut [T],
    ) {
        let t = &nw.tensors[l];
        let dims: Vec<usize> = t.inds.iter().map(|&i| nw.dims[i as usize]).collect();
        let ss = row_major_strides(&dims);
        let mut base = 0;
        let mut kd = Vec::new();
        let mut ks = Vec::new();
        for (p, &i) in t.inds.iter().enumerate() {
            match self.sliced.iter().position(|&(s, _)| s == i) {
                Some(q) => base += vals[q] * ss[p],
                None => {
                    kd.push(dims[p]);
                    ks.push(ss[p]);
                }
            }
        }
        gather(&leaf_full[l], base, &kd, &ks, out, false);
    }

    #[allow(clippy::too_many_arguments)]
    fn contract_pair<T: Scalar>(
        &self,
        p: &PairPlan,
        a: &[T],
        b: &[T],
        out: &mut [T],
        w: &mut Worker<T>,
        par: Par,
        par_perm: bool,
    ) {
        let a_red;
        let a = if p.a_sum.is_empty() {
            a
        } else {
            a_red = sum_out(a, &p.a_full_dims, &p.a_sum);
            &a_red[..]
        };
        let b_red;
        let b = if p.b_sum.is_empty() {
            b
        } else {
            b_red = sum_out(b, &p.b_full_dims, &p.b_sum);
            &b_red[..]
        };
        if let Some(f) = &p.fast {
            run_fast(f, a, b, out, par_perm);
            return;
        }
        if let Some(lp) = &p.lp {
            run_loops(lp, a, b, out, par, par_perm);
            return;
        }
        let a = match &p.a_perm {
            None => a,
            Some(perm) => {
                let ss = row_major_strides(&p.a_dims);
                let dims: Vec<usize> = perm.iter().map(|&q| p.a_dims[q]).collect();
                let st: Vec<usize> = perm.iter().map(|&q| ss[q]).collect();
                w.scratch_a.resize(a.len(), T::zero());
                gather(a, 0, &dims, &st, &mut w.scratch_a, par_perm);
                &w.scratch_a[..]
            }
        };
        let b = match &p.b_perm {
            None => b,
            Some(perm) => {
                let ss = row_major_strides(&p.b_dims);
                let dims: Vec<usize> = perm.iter().map(|&q| p.b_dims[q]).collect();
                let st: Vec<usize> = perm.iter().map(|&q| ss[q]).collect();
                w.scratch_b.resize(b.len(), T::zero());
                gather(b, 0, &dims, &st, &mut w.scratch_b, par_perm);
                &w.scratch_b[..]
            }
        };
        let (m, k, n) = (p.m, p.k, p.n);
        let one = T::from_c64(Complex64::new(1.0, 0.0));
        let par = if m * n * k >= PAR_MIN { par } else { Par::Seq };
        for bi in 0..p.nbatch {
            let ab = &a[bi * m * k..(bi + 1) * m * k];
            let bb = &b[bi * k * n..(bi + 1) * k * n];
            let cb = &mut out[bi * m * n..(bi + 1) * m * n];
            // SAFETY: the slices hold exactly m*k, k*n and m*n entries and
            // the strides below address only those.
            let (ars, acs) = if p.a_kfirst {
                (1, m as isize)
            } else {
                (k as isize, 1)
            };
            let (brs, bcs) = if p.b_kfirst {
                (n as isize, 1)
            } else {
                (1, k as isize)
            };
            let am = unsafe { MatRef::from_raw_parts(ab.as_ptr(), m, k, ars, acs) };
            let bm = unsafe { MatRef::from_raw_parts(bb.as_ptr(), k, n, brs, bcs) };
            let cm = unsafe { MatMut::from_raw_parts_mut(cb.as_mut_ptr(), m, n, n as isize, 1) };
            matmul(cm, Accum::Replace, am, bm, one, par);
        }
    }

    /// Runs one slice and returns the root tensor (root layout).
    #[allow(clippy::too_many_arguments)]
    fn run_slice<T: Scalar>(
        &self,
        nw: &Network,
        tree: &ContractionTree,
        leaf_full: &[Vec<T>],
        cache: &[Option<Vec<T>>],
        s: usize,
        w: &mut Worker<T>,
        par: Par,
        par_perm: bool,
    ) -> Vec<T> {
        let mut vals = vec![0usize; self.sliced.len()];
        let mut r = s;
        for q in (0..self.sliced.len()).rev() {
            vals[q] = r % self.sliced[q].1;
            r /= self.sliced[q].1;
        }
        let nn = self.layout.len();
        let mut res: Vec<Option<Vec<T>>> = vec![None; nn];
        let root = self.root as usize;
        if root < self.n_leaves {
            let mut o = w.pool.get(self.len[root]);
            if self.variant[root] {
                self.leaf_slice(nw, leaf_full, root, &vals, &mut o);
            } else {
                o.copy_from_slice(&leaf_full[root]);
            }
            return o;
        }
        if !self.variant[root] {
            return cache[root].clone().expect("cached root");
        }
        for &v in &self.order {
            if !self.variant[v as usize] {
                continue;
            }
            let p = self.pair[v as usize].as_ref().unwrap();
            let fetch = |c: u32,
                         res: &mut Vec<Option<Vec<T>>>,
                         w: &mut Worker<T>|
             -> (Option<Vec<T>>, bool) {
                let c = c as usize;
                if c < self.n_leaves {
                    if self.variant[c] {
                        let mut buf = w.pool.get(self.len[c]);
                        self.leaf_slice(nw, leaf_full, c, &vals, &mut buf);
                        (Some(buf), true)
                    } else {
                        (None, false)
                    }
                } else if self.variant[c] {
                    (res[c].take(), true)
                } else {
                    (None, false)
                }
            };
            let (oa, _) = fetch(p.a, &mut res, w);
            let (ob, _) = fetch(p.b, &mut res, w);
            let a_ref: &[T] = match &oa {
                Some(x) => x,
                None if (p.a as usize) < self.n_leaves => &leaf_full[p.a as usize],
                None => cache[p.a as usize].as_ref().expect("cached"),
            };
            let b_ref: &[T] = match &ob {
                Some(x) => x,
                None if (p.b as usize) < self.n_leaves => &leaf_full[p.b as usize],
                None => cache[p.b as usize].as_ref().expect("cached"),
            };
            let mut out = w.pool.get(self.len[v as usize]);
            self.contract_pair(p, a_ref, b_ref, &mut out, w, par, par_perm);
            if let Some(x) = oa {
                w.pool.put(x);
            }
            if let Some(x) = ob {
                w.pool.put(x);
            }
            res[v as usize] = Some(out);
        }
        let _ = tree;
        res[root].take().expect("root computed")
    }
}

/// How [`contract`] runs.
#[derive(Clone, Copy, Debug)]
pub struct ExecOptions {
    /// Memory budget in bytes for intermediates (all workers together).
    pub max_bytes: u128,
    /// Threads (0: the rayon pool size).
    pub threads: usize,
    /// How pairwise contractions are executed.
    pub strategy: PairStrategy,
}

impl Default for ExecOptions {
    fn default() -> Self {
        ExecOptions {
            max_bytes: 1 << 30,
            threads: 0,
            strategy: PairStrategy::Auto,
        }
    }
}

/// What [`contract`] did.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExecStats {
    /// Slices contracted.
    pub slices: usize,
    /// Workers that ran slices side by side.
    pub workers: usize,
    /// Estimated peak bytes (workers × slice working set + invariant cache).
    pub peak_bytes: u128,
    /// Wall-clock seconds.
    pub secs: f64,
}

/// Contracts `nw` along `tree` with the `sliced` indices summed slice by
/// slice; returns the result laid out over `nw.output` (row-major) and the
/// statistics. Fails with [`SimError::TooLarge`] if one slice's working set
/// (plus the invariant cache) exceeds `opts.max_bytes`.
pub fn contract<T: Scalar>(
    nw: &Network,
    tree: &ContractionTree,
    sliced: &[bool],
    opts: &ExecOptions,
) -> Result<(Vec<Complex64>, ExecStats), SimError> {
    let t0 = Instant::now();
    let out_len: usize = nw.output.iter().map(|&i| nw.dims[i as usize]).product();
    if nw.tensors.is_empty() {
        let mut v = vec![Complex64::new(0.0, 0.0); out_len.max(1)];
        v[0] = nw.scalar;
        return Ok((v, ExecStats::default()));
    }
    if tree.n_leaves != nw.tensors.len() || !tree.is_valid() {
        return Err(SimError::NotSupported {
            what: "tn: the contraction tree does not match the network",
        });
    }
    let plan = ExecPlan::new(nw, tree, sliced, opts.strategy);
    let threads = if opts.threads == 0 {
        rayon::current_num_threads()
    } else {
        opts.threads
    };
    let bytes = |e: usize| e as u128 * T::BYTES as u128;
    let slice_b = bytes(plan.slice_peak_elems) + bytes(out_len);
    let cache_b = bytes(plan.cache_elems);
    if cache_b + slice_b > opts.max_bytes {
        return Err(SimError::TooLarge {
            what: "tn contraction working set",
            bytes: cache_b + slice_b,
            limit: opts.max_bytes,
        });
    }
    let n_slices = plan.n_slices();
    let fit = ((opts.max_bytes - cache_b) / slice_b.max(1)) as usize;
    let workers = threads.min(n_slices).min(fit).max(1);
    let leaf_full: Vec<Vec<T>> = nw
        .tensors
        .iter()
        .map(|t| t.data.iter().map(|&z| T::from_c64(z)).collect())
        .collect();
    // invariant sub-trees, once (parallel GEMMs)
    let mut cache: Vec<Option<Vec<T>>> = vec![None; plan.layout.len()];
    {
        let mut w = Worker {
            pool: Pool::new(plan.slice_peak_elems / 2),
            scratch_a: Vec::new(),
            scratch_b: Vec::new(),
        };
        let mut tmp: Vec<Option<Vec<T>>> = vec![None; plan.layout.len()];
        for &v in &plan.order {
            if plan.variant[v as usize] {
                continue;
            }
            let p = plan.pair[v as usize].as_ref().unwrap();
            let take = |c: u32, tmp: &mut Vec<Option<Vec<T>>>| -> Option<Vec<T>> {
                if (c as usize) < plan.n_leaves {
                    None
                } else {
                    tmp[c as usize].take()
                }
            };
            let oa = take(p.a, &mut tmp);
            let ob = take(p.b, &mut tmp);
            let a_ref: &[T] = match &oa {
                Some(x) => x,
                None => &leaf_full[p.a as usize],
            };
            let b_ref: &[T] = match &ob {
                Some(x) => x,
                None => &leaf_full[p.b as usize],
            };
            let mut out = vec![T::zero(); plan.len[v as usize]];
            plan.contract_pair(p, a_ref, b_ref, &mut out, &mut w, Par::rayon(threads), true);
            if let Some(x) = oa {
                w.pool.put(x);
            }
            if let Some(x) = ob {
                w.pool.put(x);
            }
            if plan.keep_inv[v as usize] {
                cache[v as usize] = Some(out);
            } else {
                tmp[v as usize] = Some(out);
            }
        }
    }
    let root_len = plan.len[plan.root as usize];
    let acc: Vec<T> = if workers > 1 {
        let per_gemm = (threads / workers).max(1);
        let chunks = workers;
        (0..chunks)
            .into_par_iter()
            .map(|c| {
                let mut w = Worker {
                    pool: Pool::new(plan.slice_peak_elems / 2),
                    scratch_a: Vec::new(),
                    scratch_b: Vec::new(),
                };
                let mut acc = vec![T::zero(); root_len];
                let par = if per_gemm > 1 {
                    Par::rayon(per_gemm)
                } else {
                    Par::Seq
                };
                let mut s = c;
                while s < n_slices {
                    let r = plan.run_slice(nw, tree, &leaf_full, &cache, s, &mut w, par, false);
                    for (o, x) in acc.iter_mut().zip(&r) {
                        *o += *x;
                    }
                    w.pool.put(r);
                    s += chunks;
                }
                acc
            })
            .reduce(
                || vec![T::zero(); root_len],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b) {
                        *x += y;
                    }
                    a
                },
            )
    } else {
        let mut w = Worker {
            pool: Pool::new(plan.slice_peak_elems / 2),
            scratch_a: Vec::new(),
            scratch_b: Vec::new(),
        };
        let mut acc = vec![T::zero(); root_len];
        for s in 0..n_slices {
            let r = plan.run_slice(
                nw,
                tree,
                &leaf_full,
                &cache,
                s,
                &mut w,
                Par::rayon(threads),
                true,
            );
            if n_slices == 1 {
                acc = r;
            } else {
                for (o, x) in acc.iter_mut().zip(&r) {
                    *o += *x;
                }
                w.pool.put(r);
            }
        }
        acc
    };
    // root layout -> output order
    let rdims: Vec<usize> = plan.layout[plan.root as usize]
        .iter()
        .map(|&i| nw.dims[i as usize])
        .collect();
    let ss = row_major_strides(&rdims);
    let st: Vec<usize> = plan.out_perm.iter().map(|&q| ss[q]).collect();
    let mut out = vec![T::zero(); out_len];
    gather(&acc, 0, &plan.out_dims, &st, &mut out, true);
    let res = out.into_iter().map(|x| x.to_c64() * nw.scalar).collect();
    Ok((
        res,
        ExecStats {
            slices: n_slices,
            workers,
            peak_bytes: cache_b + slice_b * workers as u128,
            secs: t0.elapsed().as_secs_f64(),
        },
    ))
}
