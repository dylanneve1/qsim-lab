//! Tensor networks of circuits and the exact simplification passes that
//! run before the contraction-tree search
//! (research/simulability/tn.md §2).
//!
//! A [`Network`] is a list of dense [`Tensor`]s whose entries are stored
//! row-major over their index list (last index fastest), a dimension per
//! index id and the ordered list of open (output) indices. An index may be
//! shared by any number of tensors (a hyperedge, created by the diagonal
//! pass); an index that is not an output is summed over once every tensor
//! holding it has been contracted.

use crate::circuit::{check_gate, Circuit, Op, SimError};
use crate::gate::Gate;
use num_complex::Complex64;

/// Index id.
pub type Ix = u32;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);
/// Simplification only merges tensors whose combined index space has at
/// most this many entries (the passes use naive loops).
const SIMPLIFY_MAX_LOG2: u32 = 18;

/// A dense tensor: entries row-major over `inds` (last index fastest).
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    /// Index ids, slowest first. No id appears twice.
    pub inds: Vec<Ix>,
    /// `Π dims` entries.
    pub data: Vec<Complex64>,
}

/// A tensor network: the value is the contraction of all tensors over every
/// non-output index, times `scalar`, laid out row-major over `output`.
#[derive(Clone, Debug)]
pub struct Network {
    /// The tensors.
    pub tensors: Vec<Tensor>,
    /// Dimension of every index id.
    pub dims: Vec<usize>,
    /// Open indices of the result, slowest first.
    pub output: Vec<Ix>,
    /// Global factor (absorbed scalars, or 0 for a network known to vanish).
    pub scalar: Complex64,
}

/// What [`Network::simplify`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SimplifyStats {
    /// Tensors before simplification.
    pub tensors_before: usize,
    /// Tensors after.
    pub tensors_after: usize,
    /// Rank-≤2 absorptions and rank-non-increasing pair merges.
    pub merges: usize,
    /// Indices fixed because a tensor is supported on one value (column reduction).
    pub fixed: usize,
    /// Indices identified because a tensor is diagonal in them (hyperedges).
    pub diagonal: usize,
    /// Tensors split in two along a low-rank bipartition.
    pub splits: usize,
}

/// Which passes [`Network::simplify`] runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimplifyOptions {
    /// Absorb rank ≤ 2 tensors and merge pairs whose merged rank does not grow.
    pub rank: bool,
    /// Fix an index when a tensor is non-zero for only one of its values.
    pub column: bool,
    /// Identify two indices of a tensor that is diagonal in them (creates hyperedges).
    pub diagonal: bool,
    /// Split a tensor in two along an index bipartition whose matrix rank is
    /// below both sides' sizes (an exact SVD; entries below `1e-15·max` of
    /// the factors are rounded to zero).
    pub split: bool,
}

impl Default for SimplifyOptions {
    fn default() -> Self {
        SimplifyOptions {
            rank: true,
            column: true,
            diagonal: true,
            split: true,
        }
    }
}

impl SimplifyOptions {
    /// No pass at all (the raw gate network).
    pub fn none() -> Self {
        SimplifyOptions {
            rank: false,
            column: false,
            diagonal: false,
            split: false,
        }
    }
}

fn strides(inds: &[Ix], dims: &[usize]) -> Vec<usize> {
    let mut s = vec![0; inds.len()];
    let mut acc = 1;
    for k in (0..inds.len()).rev() {
        s[k] = acc;
        acc *= dims[inds[k] as usize];
    }
    s
}

fn numel(inds: &[Ix], dims: &[usize]) -> usize {
    inds.iter().map(|&i| dims[i as usize]).product()
}

/// Mixed-radix odometer over `inds` that tracks one linear offset per
/// stride vector.
struct Odometer {
    radix: Vec<usize>,
    digit: Vec<usize>,
}

impl Odometer {
    fn new(radix: Vec<usize>) -> Self {
        let d = vec![0; radix.len()];
        Odometer { radix, digit: d }
    }
    /// Advances; updates `offs[t] += strides[t][k]` bookkeeping. Returns
    /// false after the last position.
    fn step(&mut self, offs: &mut [usize], st: &[Vec<usize>]) -> bool {
        let mut k = self.radix.len();
        while k > 0 {
            k -= 1;
            self.digit[k] += 1;
            for (o, s) in offs.iter_mut().zip(st) {
                *o += s[k];
            }
            if self.digit[k] < self.radix[k] {
                return true;
            }
            for (o, s) in offs.iter_mut().zip(st) {
                *o -= s[k] * self.radix[k];
            }
            self.digit[k] = 0;
        }
        false
    }
}

/// Contracts `a` and `b` over their shared indices that `keep` rejects;
/// the result holds `a`'s kept indices then `b`'s new kept ones. Naive
/// loops: for small tensors only.
pub(crate) fn contract_small(
    a: &Tensor,
    b: &Tensor,
    dims: &[usize],
    keep: &dyn Fn(Ix) -> bool,
) -> Tensor {
    let mut all: Vec<Ix> = a.inds.clone();
    for &i in &b.inds {
        if !all.contains(&i) {
            all.push(i);
        }
    }
    // indices only in a or only in b that are not kept are summed too
    let out: Vec<Ix> = all.iter().copied().filter(|&i| keep(i)).collect();
    let mut data = vec![ZERO; numel(&out, dims)];
    let pos = |list: &[Ix], st: &[usize], i: Ix| -> usize {
        list.iter().position(|&x| x == i).map_or(0, |p| st[p])
    };
    let sa = strides(&a.inds, dims);
    let sb = strides(&b.inds, dims);
    let so = strides(&out, dims);
    let st: Vec<Vec<usize>> = vec![
        all.iter().map(|&i| pos(&a.inds, &sa, i)).collect(),
        all.iter().map(|&i| pos(&b.inds, &sb, i)).collect(),
        all.iter().map(|&i| pos(&out, &so, i)).collect(),
    ];
    let mut od = Odometer::new(all.iter().map(|&i| dims[i as usize]).collect());
    let mut offs = [0usize; 3];
    loop {
        let x = a.data[offs[0]];
        if x != ZERO {
            data[offs[2]] += x * b.data[offs[1]];
        }
        if !od.step(&mut offs, &st) {
            break;
        }
    }
    Tensor { inds: out, data }
}

/// `t` with index `i` fixed to value `v` (the index is removed).
pub(crate) fn fix_index(t: &Tensor, i: Ix, v: usize, dims: &[usize]) -> Tensor {
    let p = t.inds.iter().position(|&x| x == i).expect("index present");
    let st = strides(&t.inds, dims);
    let inds: Vec<Ix> = t.inds.iter().copied().filter(|&x| x != i).collect();
    let rest: Vec<usize> = (0..t.inds.len()).filter(|&k| k != p).collect();
    let mut data = Vec::with_capacity(numel(&inds, dims));
    let mut od = Odometer::new(rest.iter().map(|&k| dims[t.inds[k] as usize]).collect());
    let stv = vec![rest.iter().map(|&k| st[k]).collect::<Vec<_>>()];
    let mut off = [v * st[p]];
    loop {
        data.push(t.data[off[0]]);
        if !od.step(&mut off, &stv) {
            break;
        }
    }
    Tensor { inds, data }
}

/// `None`: entries are non-zero for several values of index position `p`;
/// `Some(None)`: the tensor is zero; `Some(Some(v))`: only value `v`.
fn single_value(t: &Tensor, p: usize, dims: &[usize]) -> Option<Option<usize>> {
    let st = strides(&t.inds, dims);
    let d = dims[t.inds[p] as usize];
    let mut found: Option<usize> = None;
    for (off, x) in t.data.iter().enumerate() {
        if *x != ZERO {
            let v = (off / st[p]) % d;
            match found {
                None => found = Some(v),
                Some(w) if w != v => return None,
                _ => {}
            }
        }
    }
    Some(found)
}

/// True if `t` vanishes wherever positions `p` and `q` disagree.
fn is_diagonal(t: &Tensor, p: usize, q: usize, dims: &[usize]) -> bool {
    let st = strides(&t.inds, dims);
    let (dp, dq) = (dims[t.inds[p] as usize], dims[t.inds[q] as usize]);
    if dp != dq {
        return false;
    }
    t.data
        .iter()
        .enumerate()
        .all(|(off, x)| *x == ZERO || (off / st[p]) % dp == (off / st[q]) % dq)
}

/// The diagonal of `t` over positions `p` and `q`: index at `q` removed,
/// entries where both agree.
fn take_diagonal(t: &Tensor, p: usize, q: usize, dims: &[usize]) -> Tensor {
    let st = strides(&t.inds, dims);
    let inds: Vec<Ix> = t
        .inds
        .iter()
        .enumerate()
        .filter(|&(k, _)| k != q)
        .map(|(_, &x)| x)
        .collect();
    let rest: Vec<usize> = (0..t.inds.len()).filter(|&k| k != q).collect();
    let mut stv: Vec<usize> = rest.iter().map(|&k| st[k]).collect();
    // moving along p moves along q too
    let pr = rest.iter().position(|&k| k == p).expect("p kept");
    stv[pr] += st[q];
    let mut data = Vec::with_capacity(numel(&inds, dims));
    let mut od = Odometer::new(rest.iter().map(|&k| dims[t.inds[k] as usize]).collect());
    let stv = vec![stv];
    let mut off = [0usize];
    loop {
        data.push(t.data[off[0]]);
        if !od.step(&mut off, &stv) {
            break;
        }
    }
    Tensor { inds, data }
}

/// Sums `t` over index position `p`.
pub(crate) fn sum_index(t: &Tensor, p: usize, dims: &[usize]) -> Tensor {
    let i = t.inds[p];
    let d = dims[i as usize];
    let mut acc = fix_index(t, i, 0, dims);
    for v in 1..d {
        let s = fix_index(t, i, v, dims);
        for (a, b) in acc.data.iter_mut().zip(&s.data) {
            *a += b;
        }
    }
    acc
}

fn gate_tensor(g: &Gate, outs: &[Ix], ins: &[Ix]) -> Option<Tensor> {
    let mut inds: Vec<Ix> = outs.to_vec();
    inds.extend_from_slice(ins);
    if let Some(m) = g.matrix_1q() {
        return Some(Tensor {
            inds,
            data: vec![m[0][0], m[0][1], m[1][0], m[1][1]],
        });
    }
    if let Some(m) = g.matrix_2q() {
        return Some(Tensor {
            inds,
            data: m.iter().flat_map(|r| r.iter().copied()).collect(),
        });
    }
    if let Gate::Ccx(..) = g {
        // row = 4 a + 2 b + t (outputs), col likewise (inputs)
        let mut data = vec![ZERO; 64];
        for col in 0..8usize {
            let row = if col & 6 == 6 { col ^ 1 } else { col };
            data[row * 8 + col] = ONE;
        }
        return Some(Tensor { inds, data });
    }
    None
}

impl Network {
    /// Total number of index ids.
    pub fn num_indices(&self) -> usize {
        self.dims.len()
    }

    /// The network of `<x|C|0^n>` with the qubits in `open` left open:
    /// `bits[q]` is the output bit of every closed qubit `q` (ignored for
    /// open ones). The result is laid out so that `open[j]` is bit `j` of
    /// the result index (`open[0]` least significant).
    pub fn amplitude(c: &Circuit, bits: &[bool], open: &[usize]) -> Result<Network, SimError> {
        let n = c.num_qubits;
        if bits.len() != n {
            return Err(SimError::NotSupported {
                what: "tn: the bit string must have one entry per qubit",
            });
        }
        for &q in open {
            if q >= n {
                return Err(SimError::QubitOutOfRange {
                    qubit: q,
                    num_qubits: n,
                });
            }
        }
        let mut is_open = vec![false; n];
        for &q in open {
            if is_open[q] {
                return Err(SimError::NotSupported {
                    what: "tn: a qubit is listed twice in the open set",
                });
            }
            is_open[q] = true;
        }
        let mut dims: Vec<usize> = Vec::new();
        let fresh = |dims: &mut Vec<usize>| {
            dims.push(2);
            (dims.len() - 1) as Ix
        };
        let mut tensors = Vec::new();
        let mut wire: Vec<Ix> = Vec::with_capacity(n);
        for _ in 0..n {
            let w = fresh(&mut dims);
            tensors.push(Tensor {
                inds: vec![w],
                data: vec![ONE, ZERO],
            });
            wire.push(w);
        }
        for (k, op) in c.ops.iter().enumerate() {
            let g = match op {
                Op::Gate(g) => g,
                _ => {
                    return Err(SimError::MeasurementNotSupported {
                        backend: "tn",
                        op_index: k,
                    })
                }
            };
            check_gate(g, n)?;
            if matches!(g, Gate::I(_)) {
                continue;
            }
            let qs = g.qubits();
            let ins: Vec<Ix> = qs.iter().map(|&q| wire[q]).collect();
            let outs: Vec<Ix> = qs.iter().map(|_| fresh(&mut dims)).collect();
            let t = gate_tensor(g, &outs, &ins).ok_or(SimError::Unsupported {
                backend: "tn",
                gate: *g,
            })?;
            tensors.push(t);
            for (&q, &o) in qs.iter().zip(&outs) {
                wire[q] = o;
            }
        }
        for q in 0..n {
            if !is_open[q] {
                let mut data = vec![ZERO, ZERO];
                data[usize::from(bits[q])] = ONE;
                tensors.push(Tensor {
                    inds: vec![wire[q]],
                    data,
                });
            }
        }
        let output: Vec<Ix> = open.iter().rev().map(|&q| wire[q]).collect();
        Ok(Network {
            tensors,
            dims,
            output,
            scalar: ONE,
        })
    }

    /// Runs the exact simplification passes to a fixed point and
    /// renumbers the indices compactly. The contraction value is unchanged
    /// (up to rounding in the merged tensors).
    pub fn simplify(&mut self, opts: &SimplifyOptions) -> SimplifyStats {
        let mut st = SimplifyStats {
            tensors_before: self.tensors.len(),
            ..Default::default()
        };
        let mut s = Simp::new(self);
        // column and diagonal reductions first, on the raw gate tensors:
        // absorbing a non-diagonal 1-qubit gate first would hide the
        // diagonal structure (quimb's ADCRS order)
        for _round in 0..32 {
            let mut changed = false;
            if opts.column {
                changed |= s.column_reduce(&mut st);
            }
            if s.scalar == ZERO {
                break;
            }
            if opts.diagonal {
                changed |= s.diagonal_reduce(&mut st);
            }
            if opts.rank {
                changed |= s.absorb_and_merge(&mut st);
            }
            if s.scalar == ZERO {
                break;
            }
            if opts.split {
                changed |= s.split_reduce(&mut st);
            }
            if !changed {
                break;
            }
        }
        s.sum_dangling();
        s.finish(self);
        st.tensors_after = self.tensors.len();
        st
    }

    /// Renumbers indices to `0..k` (in order of first appearance) and drops
    /// unused ids.
    pub fn compact(&mut self) {
        let mut map = vec![Ix::MAX; self.dims.len()];
        let mut dims = Vec::new();
        let mut get = |i: Ix, dims: &mut Vec<usize>, old: &[usize]| {
            if map[i as usize] == Ix::MAX {
                map[i as usize] = dims.len() as Ix;
                dims.push(old[i as usize]);
            }
            map[i as usize]
        };
        let old = self.dims.clone();
        for t in self.tensors.iter_mut() {
            for i in t.inds.iter_mut() {
                *i = get(*i, &mut dims, &old);
            }
        }
        for i in self.output.iter_mut() {
            *i = get(*i, &mut dims, &old);
        }
        self.dims = dims;
    }

    /// Contracts the whole network with naive loops, tensor by tensor in
    /// list order (reference for tests; exponential in the width of that
    /// order).
    pub fn contract_naive(&self) -> Vec<Complex64> {
        let mut left = vec![0usize; self.dims.len()];
        for t in &self.tensors {
            for &i in &t.inds {
                left[i as usize] += 1;
            }
        }
        let mut is_out = vec![false; self.dims.len()];
        for &i in &self.output {
            is_out[i as usize] = true;
        }
        let mut acc = Tensor {
            inds: vec![],
            data: vec![self.scalar],
        };
        for t in &self.tensors {
            for &i in &t.inds {
                left[i as usize] -= 1;
            }
            let keep = |i: Ix| is_out[i as usize] || left[i as usize] > 0;
            acc = contract_small(&acc, t, &self.dims, &keep);
        }
        let mut out = vec![ZERO; numel(&self.output, &self.dims)];
        let so = strides(&self.output, &self.dims);
        let sa = strides(&acc.inds, &self.dims);
        let st = vec![
            acc.inds
                .iter()
                .map(|i| so[self.output.iter().position(|o| o == i).expect("out")])
                .collect::<Vec<_>>(),
            sa,
        ];
        let mut od = Odometer::new(acc.inds.iter().map(|&i| self.dims[i as usize]).collect());
        let mut offs = [0usize; 2];
        loop {
            out[offs[0]] = acc.data[offs[1]];
            if !od.step(&mut offs, &st) {
                break;
            }
        }
        out
    }
}

/// Mutable simplification state: tensors as slots, occurrence lists.
struct Simp {
    t: Vec<Option<Tensor>>,
    occ: Vec<Vec<usize>>,
    dims: Vec<usize>,
    is_out: Vec<bool>,
    output: Vec<Ix>,
    scalar: Complex64,
    /// Index sets already split (never split the same set twice).
    split_done: std::collections::HashSet<Vec<Ix>>,
}

impl Simp {
    fn new(nw: &Network) -> Self {
        let mut occ = vec![Vec::new(); nw.dims.len()];
        for (k, t) in nw.tensors.iter().enumerate() {
            for &i in &t.inds {
                occ[i as usize].push(k);
            }
        }
        let mut is_out = vec![false; nw.dims.len()];
        for &i in &nw.output {
            is_out[i as usize] = true;
        }
        Simp {
            t: nw.tensors.iter().cloned().map(Some).collect(),
            occ,
            dims: nw.dims.clone(),
            is_out,
            output: nw.output.clone(),
            scalar: nw.scalar,
            split_done: std::collections::HashSet::new(),
        }
    }

    fn remove(&mut self, k: usize) -> Tensor {
        let t = self.t[k].take().expect("live tensor");
        for &i in &t.inds {
            self.occ[i as usize].retain(|&x| x != k);
        }
        t
    }

    fn insert(&mut self, k: usize, t: Tensor) {
        for &i in &t.inds {
            self.occ[i as usize].push(k);
        }
        self.t[k] = Some(t);
    }

    /// Merged index list of `a` and `b` (indices still needed elsewhere or
    /// in the output) and the log2 size of their union.
    fn merged_inds(&self, a: usize, b: usize) -> (Vec<Ix>, u32) {
        let (ta, tb) = (self.t[a].as_ref().unwrap(), self.t[b].as_ref().unwrap());
        let mut union: Vec<Ix> = ta.inds.clone();
        for &i in &tb.inds {
            if !union.contains(&i) {
                union.push(i);
            }
        }
        let bits: f64 = union
            .iter()
            .map(|&i| (self.dims[i as usize] as f64).log2())
            .sum();
        let kept = union
            .into_iter()
            .filter(|&i| {
                self.is_out[i as usize] || self.occ[i as usize].iter().any(|&x| x != a && x != b)
            })
            .collect();
        (kept, bits.ceil() as u32)
    }

    fn merge(&mut self, a: usize, b: usize) {
        let ta = self.remove(a);
        let tb = self.remove(b);
        let occ = &self.occ;
        let is_out = &self.is_out;
        let keep = |i: Ix| is_out[i as usize] || !occ[i as usize].is_empty();
        let m = contract_small(&ta, &tb, &self.dims, &keep);
        self.insert(a, m);
    }

    fn live(&self) -> Vec<usize> {
        (0..self.t.len()).filter(|&k| self.t[k].is_some()).collect()
    }

    fn neighbours(&self, k: usize) -> Vec<usize> {
        let mut v: Vec<usize> = Vec::new();
        for &i in &self.t[k].as_ref().unwrap().inds {
            for &x in &self.occ[i as usize] {
                if x != k && !v.contains(&x) {
                    v.push(x);
                }
            }
        }
        v
    }

    /// Absorbs scalars and rank ≤ 2 tensors, and merges neighbour pairs whose
    /// merged rank is at most the larger input rank.
    fn absorb_and_merge(&mut self, st: &mut SimplifyStats) -> bool {
        let mut changed = false;
        loop {
            let mut any = false;
            for k in self.live() {
                let Some(t) = self.t[k].as_ref() else {
                    continue;
                };
                if t.inds.is_empty() {
                    let t = self.remove(k);
                    self.scalar *= t.data[0];
                    any = true;
                    continue;
                }
                let rk = t.inds.len();
                let mut best: Option<(usize, usize)> = None;
                for u in self.neighbours(k) {
                    let ru = self.t[u].as_ref().unwrap().inds.len();
                    let (kept, bits) = self.merged_inds(k, u);
                    if bits > SIMPLIFY_MAX_LOG2 || kept.len() > rk.max(ru) {
                        continue;
                    }
                    // prefer the merge that removes the most indices
                    let gain = rk + ru - kept.len();
                    if best.is_none_or(|(_, g)| gain > g) {
                        best = Some((u, gain));
                    }
                }
                if let Some((u, _)) = best {
                    self.merge(u, k);
                    st.merges += 1;
                    any = true;
                }
            }
            if !any {
                break;
            }
            changed = true;
        }
        changed
    }

    fn column_reduce(&mut self, st: &mut SimplifyStats) -> bool {
        let mut changed = false;
        for k in self.live() {
            loop {
                let Some(t) = self.t[k].as_ref() else {
                    break;
                };
                let mut hit = None;
                for (p, &i) in t.inds.iter().enumerate() {
                    if self.is_out[i as usize] {
                        continue;
                    }
                    match single_value(t, p, &self.dims) {
                        None => {}
                        Some(None) => {
                            self.scalar = ZERO;
                            return true;
                        }
                        Some(Some(v)) => {
                            hit = Some((i, v));
                            break;
                        }
                    }
                }
                let Some((i, v)) = hit else {
                    break;
                };
                for u in self.occ[i as usize].clone() {
                    let t = self.remove(u);
                    let f = fix_index(&t, i, v, &self.dims);
                    self.insert(u, f);
                }
                st.fixed += 1;
                changed = true;
            }
        }
        changed
    }

    fn diagonal_reduce(&mut self, st: &mut SimplifyStats) -> bool {
        let mut changed = false;
        for k in self.live() {
            loop {
                let Some(t) = self.t[k].as_ref() else {
                    break;
                };
                let r = t.inds.len();
                let mut hit = None;
                'outer: for p in 0..r {
                    for q in 0..r {
                        if p == q || self.is_out[t.inds[q] as usize] {
                            continue;
                        }
                        if is_diagonal(t, p, q, &self.dims) {
                            hit = Some((p, q));
                            break 'outer;
                        }
                    }
                }
                let Some((p, q)) = hit else {
                    break;
                };
                let (keep_i, drop_j) = (t.inds[p], t.inds[q]);
                let t = self.remove(k);
                let d = take_diagonal(&t, p, q, &self.dims);
                self.insert(k, d);
                // rename j -> i everywhere else
                for u in self.occ[drop_j as usize].clone() {
                    let mut t = self.remove(u);
                    let qj = t.inds.iter().position(|&x| x == drop_j).unwrap();
                    if let Some(pi) = t.inds.iter().position(|&x| x == keep_i) {
                        t = take_diagonal(&t, pi, qj, &self.dims);
                    } else {
                        t.inds[qj] = keep_i;
                    }
                    self.insert(u, t);
                }
                st.diagonal += 1;
                changed = true;
            }
        }
        changed
    }

    /// Splits tensors of rank 3..=6 along the index bipartition with the
    /// smallest matrix rank, when that rank is below both sides' sizes and
    /// the two factors are not larger than the tensor.
    fn split_reduce(&mut self, st: &mut SimplifyStats) -> bool {
        let mut changed = false;
        for k in self.live() {
            let t = self.t[k].as_ref().unwrap();
            let r = t.inds.len();
            if !(3..=6).contains(&r) || t.data.len() > 256 {
                continue;
            }
            let mut key = t.inds.clone();
            key.sort_unstable();
            if self.split_done.contains(&key) {
                continue;
            }
            let dims: Vec<usize> = t.inds.iter().map(|&i| self.dims[i as usize]).collect();
            let st_t = strides(&t.inds, &self.dims);
            // (rank, total size, mask of the left side)
            let mut best: Option<(usize, usize, u32)> = None;
            let mut best_svd: Option<(Vec<Complex64>, Vec<Complex64>)> = None;
            for mask in 1u32..(1 << r) - 1 {
                if mask & 1 == 0 {
                    continue; // position 0 always on the left: each bipartition once
                }
                let left: Vec<usize> = (0..r).filter(|&p| mask >> p & 1 == 1).collect();
                let right: Vec<usize> = (0..r).filter(|&p| mask >> p & 1 == 0).collect();
                let dl: usize = left.iter().map(|&p| dims[p]).product();
                let dr: usize = right.iter().map(|&p| dims[p]).product();
                let m = faer::Mat::<Complex64>::from_fn(dl, dr, |row, col| {
                    let mut off = 0;
                    let mut x = row;
                    for &p in left.iter().rev() {
                        off += (x % dims[p]) * st_t[p];
                        x /= dims[p];
                    }
                    let mut y = col;
                    for &p in right.iter().rev() {
                        off += (y % dims[p]) * st_t[p];
                        y /= dims[p];
                    }
                    t.data[off]
                });
                let Ok(svd) = m.thin_svd() else {
                    continue;
                };
                let sv: Vec<f64> = (0..dl.min(dr))
                    .map(|q| svd.S().column_vector()[q].re)
                    .collect();
                let smax = sv.iter().cloned().fold(0.0, f64::max);
                if smax == 0.0 {
                    continue;
                }
                let rank = sv.iter().filter(|&&x| x > 1e-13 * smax).count();
                let total = rank * (dl + dr);
                if rank >= dl.min(dr) || total > dl * dr {
                    continue;
                }
                if best.is_none_or(|b| (rank, total) < (b.0, b.1)) {
                    // left factor U·S (dl x rank), right factor V^† (rank x dr)
                    let (u, v) = (svd.U(), svd.V());
                    let mut lf = vec![ZERO; dl * rank];
                    let mut rf = vec![ZERO; rank * dr];
                    for row in 0..dl {
                        for q in 0..rank {
                            lf[row * rank + q] = u[(row, q)] * sv[q];
                        }
                    }
                    for q in 0..rank {
                        for col in 0..dr {
                            rf[q * dr + col] = v[(col, q)].conj();
                        }
                    }
                    // round-off-level entries become exact zeros so the
                    // other passes can see the structure
                    let tmax = t.data.iter().map(|z| z.norm()).fold(0.0, f64::max);
                    for z in lf.iter_mut().chain(rf.iter_mut()) {
                        if z.norm() <= 1e-15 * tmax.max(1.0) {
                            *z = ZERO;
                        }
                    }
                    best = Some((rank, total, mask));
                    best_svd = Some((lf, rf));
                }
            }
            let Some((rank, _, mask)) = best else {
                self.split_done.insert(key);
                continue;
            };
            let (lf, rf) = best_svd.unwrap();
            let t = self.remove(k);
            let left: Vec<Ix> = (0..r)
                .filter(|&p| mask >> p & 1 == 1)
                .map(|p| t.inds[p])
                .collect();
            let right: Vec<Ix> = (0..r)
                .filter(|&p| mask >> p & 1 == 0)
                .map(|p| t.inds[p])
                .collect();
            let (mut li, mut ri) = (left.clone(), right.clone());
            if rank > 1 {
                let bond = self.dims.len() as Ix;
                self.dims.push(rank);
                self.is_out.push(false);
                self.occ.push(Vec::new());
                li.push(bond);
                ri.insert(0, bond);
            }
            self.split_done.insert(key);
            self.insert(k, Tensor { inds: li, data: lf });
            self.t.push(None);
            let k2 = self.t.len() - 1;
            self.insert(k2, Tensor { inds: ri, data: rf });
            st.splits += 1;
            changed = true;
        }
        changed
    }

    /// Sums out indices that only one tensor holds and that are not outputs.
    fn sum_dangling(&mut self) {
        for k in self.live() {
            loop {
                let t = self.t[k].as_ref().unwrap();
                let p = t
                    .inds
                    .iter()
                    .position(|&i| !self.is_out[i as usize] && self.occ[i as usize].len() == 1);
                let Some(p) = p else {
                    break;
                };
                let t = self.remove(k);
                let s = sum_index(&t, p, &self.dims);
                self.insert(k, s);
            }
            if self.t[k].as_ref().is_some_and(|t| t.inds.is_empty()) {
                let t = self.remove(k);
                self.scalar *= t.data[0];
            }
        }
    }

    fn finish(self, nw: &mut Network) {
        nw.scalar = self.scalar;
        if self.scalar == ZERO {
            // the value vanishes: keep a zero network with the same outputs
            nw.tensors = self
                .output
                .iter()
                .map(|&i| Tensor {
                    inds: vec![i],
                    data: vec![ZERO; self.dims[i as usize]],
                })
                .collect();
            nw.scalar = ONE;
            if nw.tensors.is_empty() {
                nw.scalar = ZERO;
            }
        } else {
            nw.tensors = self.t.into_iter().flatten().collect();
        }
        nw.dims = self.dims;
        nw.output = self.output;
        nw.compact();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_and_fix_helpers() {
        let dims = vec![2, 2, 2];
        // CZ-like diag(1,1,1,-1) on indices [0,1] as a 2-tensor
        let t = Tensor {
            inds: vec![0, 1],
            data: vec![ONE, ZERO, ZERO, -ONE],
        };
        assert!(is_diagonal(&t, 0, 1, &dims));
        let d = take_diagonal(&t, 0, 1, &dims);
        assert_eq!(d.inds, vec![0]);
        assert_eq!(d.data, vec![ONE, -ONE]);
        let f = fix_index(&t, 1, 1, &dims);
        assert_eq!(f.data, vec![ZERO, -ONE]);
        assert_eq!(single_value(&f, 0, &dims), Some(Some(1)));
    }
}
