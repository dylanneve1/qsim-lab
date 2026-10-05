//! Contraction-tree search: randomised greedy, recursive multilevel
//! Fiduccia–Mattheyses bisection, subtree reconfiguration and
//! memory-bounded slicing, run as a random hyper-parameter search
//! (research/simulability/tn.md §3).
//!
//! Costs follow cotengra's conventions: the cost of one pairwise
//! contraction is the product of the dimensions of every index either
//! input holds (one complex multiply-add each), the size of a tensor is
//! its number of entries, and the cost of a sliced tree is the number of
//! slices times the cost of one slice (no reuse across slices).

use super::network::Network;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::Instant;

/// The structure the search works on: which indices every input tensor
/// holds, their sizes, and the output.
#[derive(Clone, Debug)]
pub struct Hypergraph {
    /// Index ids of every input tensor (leaf).
    pub inputs: Vec<Vec<u32>>,
    /// Output indices.
    pub output: Vec<u32>,
    /// log2 of every index dimension.
    pub log2dim: Vec<f64>,
    /// Number of inputs holding each index.
    pub n_occ: Vec<u32>,
    /// True for output indices.
    pub is_out: Vec<bool>,
}

impl Hypergraph {
    /// The hypergraph of a network.
    pub fn from_network(nw: &Network) -> Self {
        let inputs: Vec<Vec<u32>> = nw.tensors.iter().map(|t| t.inds.clone()).collect();
        Hypergraph::new(
            inputs,
            nw.output.clone(),
            nw.dims.iter().map(|&d| (d as f64).log2()).collect(),
        )
    }

    /// A hypergraph from raw parts (index ids `0..log2dim.len()`).
    pub fn new(inputs: Vec<Vec<u32>>, output: Vec<u32>, log2dim: Vec<f64>) -> Self {
        let mut n_occ = vec![0u32; log2dim.len()];
        for t in &inputs {
            for &i in t {
                n_occ[i as usize] += 1;
            }
        }
        let mut is_out = vec![false; log2dim.len()];
        for &i in &output {
            is_out[i as usize] = true;
        }
        Hypergraph {
            inputs,
            output,
            log2dim,
            n_occ,
            is_out,
        }
    }

    /// Number of leaves.
    pub fn n_leaves(&self) -> usize {
        self.inputs.len()
    }
}

/// A binary contraction tree. Nodes `0..n_leaves` are the inputs; internal
/// node `n_leaves + k` contracts `children[k]`. The root is the last
/// internal node (or leaf 0 when there is a single input).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContractionTree {
    /// Number of input tensors.
    pub n_leaves: usize,
    /// Children of every internal node.
    pub children: Vec<[u32; 2]>,
}

impl ContractionTree {
    /// Id of the root node.
    pub fn root(&self) -> u32 {
        if self.children.is_empty() {
            0
        } else {
            (self.n_leaves + self.children.len() - 1) as u32
        }
    }

    /// Children of node `v` (`None` for a leaf).
    pub fn kids(&self, v: u32) -> Option<[u32; 2]> {
        (v as usize >= self.n_leaves).then(|| self.children[v as usize - self.n_leaves])
    }

    /// Internal nodes in post-order (children before parents).
    pub fn post_order(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.children.len());
        if self.children.is_empty() {
            return out;
        }
        let mut stack = vec![(self.root(), false)];
        while let Some((v, done)) = stack.pop() {
            if let Some([a, b]) = self.kids(v) {
                if done {
                    out.push(v);
                } else {
                    stack.push((v, true));
                    stack.push((b, false));
                    stack.push((a, false));
                }
            }
        }
        out
    }

    /// Checks that every leaf is used exactly once and the root covers all.
    pub fn is_valid(&self) -> bool {
        if self.n_leaves == 0 {
            return self.children.is_empty();
        }
        if self.children.len() + 1 != self.n_leaves {
            return false;
        }
        let mut seen = vec![false; self.n_leaves + self.children.len()];
        let mut stack = vec![self.root()];
        while let Some(v) = stack.pop() {
            if seen[v as usize] {
                return false;
            }
            seen[v as usize] = true;
            if let Some([a, b]) = self.kids(v) {
                stack.push(a);
                stack.push(b);
            }
        }
        seen.iter().all(|&s| s)
    }
}

/// Per-node indices and costs of a tree.
#[derive(Clone, Debug)]
pub struct TreeCost {
    /// Indices of every node's tensor (sorted; sliced indices included).
    pub inds: Vec<Vec<u32>>,
    /// log2 size of every node (sliced indices excluded).
    pub log2size: Vec<f64>,
    /// log2 cost of every internal node's contraction (sliced excluded);
    /// `-inf` for leaves.
    pub log2flops: Vec<f64>,
    /// Internal nodes, post-order.
    pub order: Vec<u32>,
    /// log2 of the summed cost of one slice.
    pub log2_total: f64,
    /// log2 of the largest tensor (leaves included).
    pub log2_max: f64,
    /// log2 of the number of slices.
    pub log2_slices: f64,
}

impl TreeCost {
    /// log10 of the total cost over all slices.
    pub fn log10_total(&self) -> f64 {
        (self.log2_total + self.log2_slices) * std::f64::consts::LOG10_2
    }
}

/// log2(2^a + 2^b).
fn log2_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (hi, lo) = if a > b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp2().ln_1p() / std::f64::consts::LN_2
}

/// Computes node index sets and costs (`sliced[i]`: index `i` is sliced).
pub fn tree_cost(hg: &Hypergraph, tree: &ContractionTree, sliced: &[bool]) -> TreeCost {
    let nn = tree.n_leaves + tree.children.len();
    let w = |i: u32| {
        if sliced.get(i as usize).copied().unwrap_or(false) {
            0.0
        } else {
            hg.log2dim[i as usize]
        }
    };
    let mut cnt: Vec<Vec<(u32, u32)>> = vec![Vec::new(); nn];
    let mut inds: Vec<Vec<u32>> = vec![Vec::new(); nn];
    let mut log2size = vec![0.0; nn];
    let mut log2flops = vec![f64::NEG_INFINITY; nn];
    let mut log2_max = f64::NEG_INFINITY;
    for (l, t) in hg.inputs.iter().enumerate() {
        let mut v: Vec<u32> = t.clone();
        v.sort_unstable();
        cnt[l] = v.iter().map(|&i| (i, 1)).collect();
        log2size[l] = v.iter().map(|&i| w(i)).sum();
        log2_max = log2_max.max(log2size[l]);
        inds[l] = v;
    }
    let order = tree.post_order();
    let mut total = f64::NEG_INFINITY;
    for &v in &order {
        let [a, b] = tree.kids(v).unwrap();
        let (ca, cb) = (&cnt[a as usize], &cnt[b as usize]);
        let mut merged: Vec<(u32, u32)> = Vec::with_capacity(ca.len() + cb.len());
        let (mut x, mut y) = (0, 0);
        let mut fl = 0.0;
        while x < ca.len() || y < cb.len() {
            let take = if y >= cb.len() || (x < ca.len() && ca[x].0 < cb[y].0) {
                let e = ca[x];
                x += 1;
                e
            } else if x >= ca.len() || cb[y].0 < ca[x].0 {
                let e = cb[y];
                y += 1;
                e
            } else {
                let e = (ca[x].0, ca[x].1 + cb[y].1);
                x += 1;
                y += 1;
                e
            };
            fl += w(take.0);
            if hg.is_out[take.0 as usize] || take.1 < hg.n_occ[take.0 as usize] {
                merged.push(take);
            }
        }
        log2flops[v as usize] = fl;
        total = log2_add(total, fl);
        let s: f64 = merged.iter().map(|&(i, _)| w(i)).sum();
        log2size[v as usize] = s;
        log2_max = log2_max.max(s);
        inds[v as usize] = merged.iter().map(|&(i, _)| i).collect();
        cnt[v as usize] = merged;
    }
    let log2_slices = sliced
        .iter()
        .enumerate()
        .filter(|(_, &s)| s)
        .map(|(i, _)| hg.log2dim[i])
        .sum();
    TreeCost {
        inds,
        log2size,
        log2flops,
        order,
        log2_total: total,
        log2_max,
        log2_slices,
    }
}

// ---------------------------------------------------------------------------
// Tree builder and greedy

struct Builder {
    n_leaves: usize,
    children: Vec<[u32; 2]>,
}

impl Builder {
    fn merge(&mut self, a: u32, b: u32) -> u32 {
        self.children.push([a, b]);
        (self.n_leaves + self.children.len() - 1) as u32
    }
}

#[derive(PartialEq)]
struct Cand(f64, u32, u32);
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Cand {
    // min-heap on the score
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.total_cmp(&self.0)
            .then_with(|| o.1.cmp(&self.1))
            .then_with(|| o.2.cmp(&self.2))
    }
}

/// Greedy hyper-parameters (cotengra-style `costmod` and `temperature`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GreedyParams {
    /// Weight of the inputs' sizes: score `size(ab) - alpha (size(a) + size(b))`.
    pub alpha: f64,
    /// Gumbel noise on the (log-scaled) score; 0 is deterministic.
    pub temperature: f64,
}

impl Default for GreedyParams {
    fn default() -> Self {
        GreedyParams {
            alpha: 1.0,
            temperature: 0.0,
        }
    }
}

/// Greedily contracts the nodes `start` (leaf or internal ids already in
/// the builder, with their index-count lists) into one; returns its id.
fn greedy_nodes(
    hg: &Hypergraph,
    start: Vec<(u32, Vec<(u32, u32)>)>,
    b: &mut Builder,
    p: &GreedyParams,
    rng: &mut StdRng,
) -> u32 {
    if start.len() == 1 {
        return start[0].0;
    }
    // index -> total count among the start nodes; kept if this is below
    // the global count or it is an output
    let mut total: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for (_, c) in &start {
        for &(i, k) in c {
            *total.entry(i).or_insert(0) += k;
        }
    }
    let ext = |i: u32, t: &std::collections::HashMap<u32, u32>| {
        hg.is_out[i as usize] || t[&i] < hg.n_occ[i as usize]
    };
    let mut slot_node: Vec<u32> = Vec::new();
    let mut slot_inds: Vec<Option<Vec<(u32, u32)>>> = Vec::new();
    let mut idx_slots: std::collections::HashMap<u32, Vec<u32>> = std::collections::HashMap::new();
    let size = |c: &[(u32, u32)]| -> f64 { c.iter().map(|&(i, _)| hg.log2dim[i as usize]).sum() };
    let merge_list = |x: &[(u32, u32)], y: &[(u32, u32)], t: &std::collections::HashMap<u32, u32>| {
        let mut out = Vec::with_capacity(x.len() + y.len());
        let (mut a, mut c) = (0, 0);
        while a < x.len() || c < y.len() {
            let e = if c >= y.len() || (a < x.len() && x[a].0 < y[c].0) {
                a += 1;
                x[a - 1]
            } else if a >= x.len() || y[c].0 < x[a].0 {
                c += 1;
                y[c - 1]
            } else {
                a += 1;
                c += 1;
                (x[a - 1].0, x[a - 1].1 + y[c - 1].1)
            };
            if ext(e.0, t) || e.1 < t[&e.0] {
                out.push(e);
            }
        }
        out
    };
    for (node, c) in start {
        let s = slot_node.len() as u32;
        for &(i, _) in &c {
            idx_slots.entry(i).or_default().push(s);
        }
        slot_node.push(node);
        slot_inds.push(Some(c));
    }
    let score = |x: &[(u32, u32)],
                 y: &[(u32, u32)],
                 t: &std::collections::HashMap<u32, u32>,
                 rng: &mut StdRng| {
        let m = merge_list(x, y, t);
        let (sx, sy, sm) = (size(x).exp2(), size(y).exp2(), size(&m).exp2());
        let d = sm - p.alpha * (sx + sy);
        let s = d.signum() * (1.0 + d.abs()).log2();
        if p.temperature > 0.0 {
            let u: f64 = rng.random::<f64>().max(1e-300);
            s - p.temperature * (-(-u.ln()).ln())
        } else {
            s
        }
    };
    let mut heap = BinaryHeap::new();
    let n0 = slot_node.len() as u32;
    for s in 0..n0 {
        let mut nb: Vec<u32> = Vec::new();
        for &(i, _) in slot_inds[s as usize].as_ref().unwrap() {
            for &o in &idx_slots[&i] {
                if o > s && !nb.contains(&o) {
                    nb.push(o);
                }
            }
        }
        for o in nb {
            let sc = score(
                slot_inds[s as usize].as_ref().unwrap(),
                slot_inds[o as usize].as_ref().unwrap(),
                &total,
                rng,
            );
            heap.push(Cand(sc, s, o));
        }
    }
    let mut alive = n0 as usize;
    while let Some(Cand(_, x, y)) = heap.pop() {
        if slot_inds[x as usize].is_none() || slot_inds[y as usize].is_none() {
            continue;
        }
        let cx = slot_inds[x as usize].take().unwrap();
        let cy = slot_inds[y as usize].take().unwrap();
        let m = merge_list(&cx, &cy, &total);
        let node = b.merge(slot_node[x as usize], slot_node[y as usize]);
        let z = slot_node.len() as u32;
        slot_node.push(node);
        for &(i, _) in cx.iter().chain(cy.iter()) {
            if let Some(v) = idx_slots.get_mut(&i) {
                v.retain(|&s| s != x && s != y);
            }
        }
        for &(i, _) in &m {
            idx_slots.entry(i).or_default().push(z);
        }
        let mut nb: Vec<u32> = Vec::new();
        for &(i, _) in &m {
            for &o in &idx_slots[&i] {
                if o != z && !nb.contains(&o) {
                    nb.push(o);
                }
            }
        }
        for &o in &nb {
            let sc = score(&m, slot_inds[o as usize].as_ref().unwrap(), &total, rng);
            heap.push(Cand(sc, z, o));
        }
        slot_inds.push(Some(m));
        alive -= 1;
    }
    // disconnected remainder: outer products, smallest first
    let mut rest: Vec<(f64, u32)> = (0..slot_node.len())
        .filter(|&s| slot_inds[s].is_some())
        .map(|s| (size(slot_inds[s].as_ref().unwrap()), slot_node[s]))
        .collect();
    debug_assert_eq!(rest.len(), alive);
    rest.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut acc = rest[0];
    for &(s, n) in &rest[1..] {
        let id = b.merge(acc.1, n);
        acc = (acc.0 + s, id);
    }
    acc.1
}

fn leaf_counts(hg: &Hypergraph, l: u32) -> Vec<(u32, u32)> {
    let mut v: Vec<(u32, u32)> = hg.inputs[l as usize].iter().map(|&i| (i, 1)).collect();
    v.sort_unstable();
    v
}

/// A greedy contraction tree of the whole network.
pub fn greedy(hg: &Hypergraph, p: &GreedyParams, seed: u64) -> ContractionTree {
    let mut b = Builder {
        n_leaves: hg.n_leaves(),
        children: Vec::new(),
    };
    if hg.n_leaves() > 1 {
        let start = (0..hg.n_leaves() as u32)
            .map(|l| (l, leaf_counts(hg, l)))
            .collect();
        let mut rng = StdRng::seed_from_u64(seed);
        greedy_nodes(hg, start, &mut b, p, &mut rng);
    }
    ContractionTree {
        n_leaves: b.n_leaves,
        children: b.children,
    }
}

// ---------------------------------------------------------------------------
// Multilevel FM bisection

/// A small weighted hypergraph for partitioning.
struct PHg {
    vw: Vec<f64>,
    nets: Vec<Vec<u32>>,
    nw: Vec<f64>,
    vnets: Vec<Vec<u32>>,
}

impl PHg {
    fn new(vw: Vec<f64>, nets: Vec<Vec<u32>>, nw: Vec<f64>) -> Self {
        let mut vnets = vec![Vec::new(); vw.len()];
        for (e, pins) in nets.iter().enumerate() {
            for &v in pins {
                vnets[v as usize].push(e as u32);
            }
        }
        PHg {
            vw,
            nets,
            nw,
            vnets,
        }
    }

    fn cut(&self, part: &[u8]) -> f64 {
        self.nets
            .iter()
            .zip(&self.nw)
            .filter(|(pins, _)| {
                let p0 = part[pins[0] as usize];
                pins.iter().any(|&v| part[v as usize] != p0)
            })
            .map(|(_, &w)| w)
            .sum()
    }

    /// Heavy-edge matching; returns the coarse hypergraph and the map.
    fn coarsen(&self, rng: &mut StdRng, max_vw: f64) -> Option<(PHg, Vec<u32>)> {
        let n = self.vw.len();
        let mut order: Vec<u32> = (0..n as u32).collect();
        for i in (1..n).rev() {
            let j = rng.random_range(0..=i);
            order.swap(i, j);
        }
        let mut mate = vec![u32::MAX; n];
        let mut rating = vec![0.0f64; n];
        let mut touched: Vec<u32> = Vec::new();
        for &v in &order {
            if mate[v as usize] != u32::MAX {
                continue;
            }
            for &e in &self.vnets[v as usize] {
                let pins = &self.nets[e as usize];
                if pins.len() > 64 {
                    continue;
                }
                let r = self.nw[e as usize] / (pins.len() - 1).max(1) as f64;
                for &u in pins {
                    if u != v && mate[u as usize] == u32::MAX {
                        if rating[u as usize] == 0.0 {
                            touched.push(u);
                        }
                        rating[u as usize] += r;
                    }
                }
            }
            let mut best = (0.0, u32::MAX);
            for &u in &touched {
                let r = rating[u as usize] * (1.0 + 1e-6 * rng.random::<f64>());
                if r > best.0 && self.vw[v as usize] + self.vw[u as usize] <= max_vw {
                    best = (r, u);
                }
                rating[u as usize] = 0.0;
            }
            touched.clear();
            mate[v as usize] = v;
            if best.1 != u32::MAX {
                mate[v as usize] = best.1;
                mate[best.1 as usize] = v;
            }
        }
        let mut cmap = vec![u32::MAX; n];
        let mut vw = Vec::new();
        for v in 0..n {
            if cmap[v] != u32::MAX {
                continue;
            }
            let c = vw.len() as u32;
            cmap[v] = c;
            let m = mate[v] as usize;
            let mut w = self.vw[v];
            if m != v {
                cmap[m] = c;
                w += self.vw[m];
            }
            vw.push(w);
        }
        if vw.len() as f64 > 0.95 * n as f64 {
            return None;
        }
        let mut nets = Vec::new();
        let mut nw = Vec::new();
        for (e, pins) in self.nets.iter().enumerate() {
            let mut cp: Vec<u32> = pins.iter().map(|&v| cmap[v as usize]).collect();
            cp.sort_unstable();
            cp.dedup();
            if cp.len() >= 2 {
                nets.push(cp);
                nw.push(self.nw[e]);
            }
        }
        Some((PHg::new(vw, nets, nw), cmap))
    }

    /// FM refinement passes; `maxw[s]` caps the weight of side `s`.
    fn fm(&self, part: &mut [u8], maxw: f64, rng: &mut StdRng, passes: usize) {
        let n = self.vw.len();
        let mut cnt: Vec<[u32; 2]> = self
            .nets
            .iter()
            .map(|pins| {
                let mut c = [0u32; 2];
                for &v in pins {
                    c[part[v as usize] as usize] += 1;
                }
                c
            })
            .collect();
        let mut side_w = [0.0f64; 2];
        for v in 0..n {
            side_w[part[v] as usize] += self.vw[v];
        }
        let gain = |v: usize, part: &[u8], cnt: &[[u32; 2]]| -> f64 {
            let s = part[v] as usize;
            let t = 1 - s;
            let mut g = 0.0;
            for &e in &self.vnets[v] {
                let c = cnt[e as usize];
                if c[s] == 1 {
                    g += self.nw[e as usize];
                }
                if c[t] == 0 {
                    g -= self.nw[e as usize];
                }
            }
            g
        };
        for _ in 0..passes {
            let mut stamp = vec![0u32; n];
            let mut locked = vec![false; n];
            let mut heap: BinaryHeap<(OrdF, u32, u32)> = BinaryHeap::new();
            for v in 0..n {
                let g = gain(v, part, &cnt);
                heap.push((OrdF(g + 1e-9 * rng.random::<f64>()), 0, v as u32));
            }
            let mut moves: Vec<u32> = Vec::new();
            let (mut cum, mut best, mut best_len) = (0.0, 0.0, 0usize);
            let limit = 50 + n / 4;
            while let Some((OrdF(g), st, v)) = heap.pop() {
                let v = v as usize;
                if locked[v] || st != stamp[v] {
                    continue;
                }
                let s = part[v] as usize;
                let t = 1 - s;
                if side_w[t] + self.vw[v] > maxw {
                    continue;
                }
                // apply
                locked[v] = true;
                part[v] = t as u8;
                side_w[s] -= self.vw[v];
                side_w[t] += self.vw[v];
                for &e in &self.vnets[v] {
                    cnt[e as usize][s] -= 1;
                    cnt[e as usize][t] += 1;
                }
                cum += g.round_to(1e-6);
                moves.push(v as u32);
                if cum > best + 1e-9 {
                    best = cum;
                    best_len = moves.len();
                }
                if moves.len() - best_len > limit {
                    break;
                }
                for &e in &self.vnets[v] {
                    for &u in &self.nets[e as usize] {
                        let u = u as usize;
                        if !locked[u] {
                            stamp[u] += 1;
                            let gu = gain(u, part, &cnt);
                            heap.push((OrdF(gu + 1e-9 * rng.random::<f64>()), stamp[u], u as u32));
                        }
                    }
                }
            }
            // roll back to the best prefix
            for &v in moves[best_len..].iter().rev() {
                let v = v as usize;
                let t = part[v] as usize;
                let s = 1 - t;
                part[v] = s as u8;
                side_w[t] -= self.vw[v];
                side_w[s] += self.vw[v];
                for &e in &self.vnets[v] {
                    cnt[e as usize][t] -= 1;
                    cnt[e as usize][s] += 1;
                }
            }
            if best <= 1e-9 {
                break;
            }
        }
    }

    fn initial(&self, rng: &mut StdRng, maxw: f64) -> Vec<u8> {
        let n = self.vw.len();
        let total: f64 = self.vw.iter().sum();
        let mut best: Option<(f64, Vec<u8>)> = None;
        for _ in 0..8 {
            // grow part 1 from a random seed by connectivity
            let mut part = vec![0u8; n];
            let mut w1 = 0.0;
            let mut conn = vec![0.0f64; n];
            let seed = rng.random_range(0..n);
            let mut next = Some(seed);
            while let Some(v) = next {
                part[v] = 1;
                w1 += self.vw[v];
                if w1 >= total / 2.0 {
                    break;
                }
                for &e in &self.vnets[v] {
                    for &u in &self.nets[e as usize] {
                        conn[u as usize] += self.nw[e as usize];
                    }
                }
                next = None;
                let mut bc = -1.0;
                for u in 0..n {
                    if part[u] == 0 {
                        let c = conn[u] + 1e-6 * rng.random::<f64>();
                        if c > bc {
                            bc = c;
                            next = Some(u);
                        }
                    }
                }
            }
            self.fm(&mut part, maxw, rng, 8);
            let c = self.cut(&part);
            if best.as_ref().is_none_or(|b| c < b.0) {
                best = Some((c, part));
            }
        }
        best.unwrap().1
    }

    /// Multilevel bisection with imbalance `eps`.
    fn bisect(&self, eps: f64, rng: &mut StdRng) -> Vec<u8> {
        let total: f64 = self.vw.iter().sum();
        let maxw = (total * (1.0 + eps) / 2.0)
            .max(total / 2.0 + 1.0)
            .min((0.95 * total).max(total / 2.0 + 1.0));
        let mut levels: Vec<(PHg, Vec<u32>)> = Vec::new();
        {
            let mut cur: &PHg = self;
            loop {
                if cur.vw.len() <= 48 {
                    break;
                }
                match cur.coarsen(rng, maxw / 3.0) {
                    Some(x) => levels.push(x),
                    None => break,
                }
                cur = &levels.last().unwrap().0;
            }
        }
        let coarsest = levels.last().map_or(self, |x| &x.0);
        let mut part = coarsest.initial(rng, maxw);
        for k in (0..levels.len()).rev() {
            let finer = if k == 0 { self } else { &levels[k - 1].0 };
            let cmap = &levels[k].1;
            let fp: Vec<u8> = cmap.iter().map(|&c| part[c as usize]).collect();
            part = fp;
            finer.fm(&mut part, maxw, rng, 4);
        }
        part
    }
}

#[derive(PartialEq, PartialOrd, Clone, Copy)]
struct OrdF(f64);
impl Eq for OrdF {}
#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for OrdF {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.total_cmp(&o.0)
    }
}

trait RoundTo {
    fn round_to(self, q: f64) -> f64;
}
impl RoundTo for f64 {
    fn round_to(self, q: f64) -> f64 {
        (self / q).round() * q
    }
}

/// Bisection hyper-parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BisectParams {
    /// Allowed imbalance of the two sides at the top level (0 = perfectly
    /// balanced; each side keeps at least 5 % of the tensors).
    pub imbalance: f64,
    /// Depth dependence of the imbalance (cotengra's `imbalance_decay`): for
    /// a sub-network holding a fraction `s` of all tensors the imbalance is
    /// `s^d · imbalance` for `d ≥ 0`, else `1 − s^(−d) (1 − imbalance)`.
    pub imbalance_decay: f64,
    /// Relative noise on the index weights of every partition.
    pub jitter: f64,
    /// Sub-networks of at most this many tensors are finished greedily.
    pub cutoff: usize,
    /// Greedy parameters for the small sub-networks.
    pub greedy: GreedyParams,
}

impl Default for BisectParams {
    fn default() -> Self {
        BisectParams {
            imbalance: 0.2,
            imbalance_decay: 0.0,
            jitter: 0.0,
            cutoff: 12,
            greedy: GreedyParams::default(),
        }
    }
}

fn bisect_rec(
    hg: &Hypergraph,
    leaves: &[u32],
    b: &mut Builder,
    p: &BisectParams,
    rng: &mut StdRng,
) -> u32 {
    if leaves.len() <= p.cutoff.max(2) {
        let start = leaves.iter().map(|&l| (l, leaf_counts(hg, l))).collect();
        return greedy_nodes(hg, start, b, &p.greedy, rng);
    }
    // local hypergraph: nets = indices with >= 2 pins among `leaves`
    let mut local: std::collections::HashMap<u32, Vec<u32>> = std::collections::HashMap::new();
    for (v, &l) in leaves.iter().enumerate() {
        for &i in &hg.inputs[l as usize] {
            local.entry(i).or_default().push(v as u32);
        }
    }
    let mut keys: Vec<u32> = local.keys().copied().collect();
    keys.sort_unstable();
    let mut nets = Vec::new();
    let mut nw = Vec::new();
    for i in keys {
        let pins = &local[&i];
        if pins.len() >= 2 {
            nets.push(pins.clone());
            let j = if p.jitter > 0.0 {
                1.0 + p.jitter * (rng.random::<f64>() - 0.5)
            } else {
                1.0
            };
            nw.push(hg.log2dim[i as usize] * j);
        }
    }
    let ph = PHg::new(vec![1.0; leaves.len()], nets, nw);
    let s = leaves.len() as f64 / hg.n_leaves() as f64;
    let eps = if p.imbalance_decay >= 0.0 {
        s.powf(p.imbalance_decay) * p.imbalance
    } else {
        1.0 - s.powf(-p.imbalance_decay) * (1.0 - p.imbalance.min(1.0))
    };
    let part = ph.bisect(eps.clamp(0.001, 0.9), rng);
    let (mut l0, mut l1) = (Vec::new(), Vec::new());
    for (v, &l) in leaves.iter().enumerate() {
        if part[v] == 0 {
            l0.push(l);
        } else {
            l1.push(l);
        }
    }
    if l0.is_empty() || l1.is_empty() {
        let start = leaves.iter().map(|&l| (l, leaf_counts(hg, l))).collect();
        return greedy_nodes(hg, start, b, &p.greedy, rng);
    }
    let a = bisect_rec(hg, &l0, b, p, rng);
    let c = bisect_rec(hg, &l1, b, p, rng);
    b.merge(a, c)
}

/// A tree from recursive multilevel FM bisection of the network.
pub fn bisection(hg: &Hypergraph, p: &BisectParams, seed: u64) -> ContractionTree {
    let mut b = Builder {
        n_leaves: hg.n_leaves(),
        children: Vec::new(),
    };
    if hg.n_leaves() > 1 {
        let mut rng = StdRng::seed_from_u64(seed);
        let leaves: Vec<u32> = (0..hg.n_leaves() as u32).collect();
        bisect_rec(hg, &leaves, &mut b, p, &mut rng);
    }
    ContractionTree {
        n_leaves: b.n_leaves,
        children: b.children,
    }
}

// ---------------------------------------------------------------------------
// Subtree reconfiguration

/// Re-optimises every subtree of up to `k` frontier tensors exactly (dynamic
/// programming over subsets), minimising the summed contraction cost with
/// every intermediate at most `max_log2size` (when set). Returns the number
/// of improved subtrees.
pub fn reconfigure(
    hg: &Hypergraph,
    tree: &mut ContractionTree,
    sliced: &[bool],
    k: usize,
    max_log2size: Option<f64>,
    passes: usize,
) -> usize {
    let k = k.clamp(3, 12);
    let mut improved = 0;
    for _ in 0..passes {
        let mut tc = tree_cost(hg, tree, sliced);
        // most expensive contractions first
        let mut nodes = tc.order.clone();
        nodes.sort_by(|a, b| tc.log2flops[*b as usize].total_cmp(&tc.log2flops[*a as usize]));
        let mut any = false;
        for v in nodes {
            if reconf_node(hg, tree, &tc, sliced, v, k, max_log2size) {
                improved += 1;
                any = true;
                tc = tree_cost(hg, tree, sliced);
            }
        }
        if !any {
            break;
        }
    }
    improved
}

fn reconf_node(
    hg: &Hypergraph,
    tree: &mut ContractionTree,
    tc: &TreeCost,
    sliced: &[bool],
    v: u32,
    k: usize,
    max_log2size: Option<f64>,
) -> bool {
    // frontier: expand the most expensive internal node until k members
    let mut frontier: Vec<u32> = vec![v];
    let mut internal: Vec<u32> = Vec::new();
    while frontier.len() < k {
        let mut best: Option<(usize, f64)> = None;
        for (p, &f) in frontier.iter().enumerate() {
            if tree.kids(f).is_some() {
                let c = tc.log2flops[f as usize];
                if best.is_none_or(|b| c > b.1) {
                    best = Some((p, c));
                }
            }
        }
        let Some((p, _)) = best else {
            break;
        };
        let f = frontier.swap_remove(p);
        internal.push(f);
        let [a, b] = tree.kids(f).unwrap();
        frontier.push(a);
        frontier.push(b);
    }
    let m = frontier.len();
    if m < 3 {
        return false;
    }
    let old: f64 = internal
        .iter()
        .map(|&x| tc.log2flops[x as usize].exp2())
        .sum();
    // local index numbering (sliced indices dropped)
    let mut local: Vec<u32> = Vec::new();
    let mut fb: Vec<u128> = vec![0; m];
    for (j, &f) in frontier.iter().enumerate() {
        for &i in &tc.inds[f as usize] {
            if sliced.get(i as usize).copied().unwrap_or(false) {
                continue;
            }
            let p = match local.iter().position(|&x| x == i) {
                Some(p) => p,
                None => {
                    if local.len() == 128 {
                        return false;
                    }
                    local.push(i);
                    local.len() - 1
                }
            };
            fb[j] |= 1u128 << p;
        }
    }
    let mut ext: u128 = 0;
    for &i in &tc.inds[v as usize] {
        if let Some(p) = local.iter().position(|&x| x == i) {
            ext |= 1u128 << p;
        }
    }
    let lw: Vec<f64> = local.iter().map(|&i| hg.log2dim[i as usize]).collect();
    let all_two = lw.iter().all(|&w| w == 1.0);
    let weight = |b: u128| -> f64 {
        if all_two {
            b.count_ones() as f64
        } else {
            let mut s = 0.0;
            let mut x = b;
            while x != 0 {
                let t = x.trailing_zeros();
                s += lw[t as usize];
                x &= x - 1;
            }
            s
        }
    };
    let full = (1usize << m) - 1;
    // union of frontier bits per subset and kept bits per subset
    let mut uni = vec![0u128; 1 << m];
    for s in 1..=full {
        let low = s.trailing_zeros() as usize;
        uni[s] = uni[s & (s - 1)] | fb[low];
    }
    let mut kept = vec![0u128; 1 << m];
    let mut ksize = vec![0.0f64; 1 << m];
    for s in 1..=full {
        kept[s] = uni[s] & (ext | uni[full & !s]);
        ksize[s] = weight(kept[s]);
    }
    let lim = max_log2size.unwrap_or(f64::INFINITY);
    let mut best = vec![f64::INFINITY; 1 << m];
    let mut split = vec![0usize; 1 << m];
    for j in 0..m {
        best[1 << j] = 0.0;
    }
    for s in 1..=full {
        if s & (s - 1) == 0 {
            continue;
        }
        if s != full && ksize[s] > lim + 1e-9 {
            continue;
        }
        let low = s & s.wrapping_neg();
        let rest = s ^ low;
        // enumerate sub-masks L of s that contain `low`, L != s
        let mut sub = rest;
        loop {
            let l = sub | low;
            if l != s {
                let r = s ^ l;
                let (bl, br) = (best[l], best[r]);
                if bl.is_finite() && br.is_finite() {
                    let c = bl + br + weight(kept[l] | kept[r]).exp2();
                    if c < best[s] {
                        best[s] = c;
                        split[s] = l;
                    }
                }
            }
            if sub == 0 {
                break;
            }
            sub = (sub - 1) & rest;
        }
    }
    if !(best[full] < old * (1.0 - 1e-9)) {
        return false;
    }
    // rebuild: reuse the internal slots, v stays the subtree root
    let mut slots: Vec<u32> = internal.iter().copied().filter(|&x| x != v).collect();
    fn build(
        s: usize,
        split: &[usize],
        frontier: &[u32],
        slots: &mut Vec<u32>,
        tree: &mut ContractionTree,
        root: Option<u32>,
    ) -> u32 {
        if s & (s - 1) == 0 {
            return frontier[s.trailing_zeros() as usize];
        }
        let l = split[s];
        let a = build(l, split, frontier, slots, tree, None);
        let b = build(s ^ l, split, frontier, slots, tree, None);
        let id = root.unwrap_or_else(|| slots.pop().expect("slot"));
        tree.children[id as usize - tree.n_leaves] = [a, b];
        id
    }
    build(full, &split, &frontier, &mut slots, tree, Some(v));
    debug_assert!(slots.is_empty());
    true
}

// ---------------------------------------------------------------------------
// Slicing

/// Chooses indices to slice until every intermediate has at most
/// `2^target_log2` entries, each time the index that keeps the total
/// (all-slice) cost lowest; with `reconf`, re-optimises subtrees under the
/// sliced sizes after each choice. Returns the sliced flags.
pub fn slice_tree(
    hg: &Hypergraph,
    tree: &mut ContractionTree,
    target_log2: f64,
    reconf: bool,
) -> Vec<bool> {
    let ni = hg.log2dim.len();
    let mut sliced = vec![false; ni];
    loop {
        let tc = tree_cost(hg, tree, &sliced);
        if tc.log2_max <= target_log2 + 1e-9 {
            break;
        }
        // candidate indices: those of tensors above the target
        let mut cand = vec![false; ni];
        let mut any = false;
        for v in 0..tc.inds.len() {
            if tc.log2size[v] > target_log2 + 1e-9 {
                for &i in &tc.inds[v] {
                    if !hg.is_out[i as usize] && !sliced[i as usize] && hg.log2dim[i as usize] > 0.0
                    {
                        cand[i as usize] = true;
                        any = true;
                    }
                }
            }
        }
        if !any {
            break; // only output indices left: cannot slice further
        }
        // S_i = summed cost of the contractions that involve index i
        let mut s_i = vec![f64::NEG_INFINITY; ni];
        for &v in &tc.order {
            let [a, b] = tree.kids(v).unwrap();
            let f = tc.log2flops[v as usize];
            let mut seen: Vec<u32> = tc.inds[a as usize].clone();
            seen.extend_from_slice(&tc.inds[b as usize]);
            seen.sort_unstable();
            seen.dedup();
            for i in seen {
                if cand[i as usize] {
                    s_i[i as usize] = log2_add(s_i[i as usize], f);
                }
            }
        }
        let t = tc.log2_total;
        let mut best: Option<(f64, f64, usize)> = None;
        for i in 0..ni {
            if !cand[i] {
                continue;
            }
            let d = hg.log2dim[i].exp2();
            // new per-slice total: T - S_i + S_i / d; times d slices
            let (tt, si) = (t.exp2(), s_i[i].exp2());
            let new_total = d * (tt - si) + si;
            // how many of the largest tensors it touches (tie-break)
            let touch: f64 = (0..tc.inds.len())
                .filter(|&v| {
                    tc.log2size[v] > target_log2 + 1e-9 && tc.inds[v].binary_search(&(i as u32)).is_ok()
                })
                .map(|v| tc.log2size[v])
                .sum();
            let key = (new_total, -touch, i);
            if best.is_none_or(|b| key.0 < b.0 * (1.0 - 1e-12) || (key.0 <= b.0 * (1.0 + 1e-12) && key.1 < b.1)) {
                best = Some(key);
            }
        }
        let (_, _, i) = best.unwrap();
        sliced[i] = true;
        if reconf {
            let tc2 = tree_cost(hg, tree, &sliced);
            let lim = tc2.log2_max.max(target_log2);
            reconfigure(hg, tree, &sliced, 8, Some(lim), 1);
        }
    }
    sliced
}

// ---------------------------------------------------------------------------
// Hyper-parameter search

/// Options of [`search`].
#[derive(Clone, Debug, PartialEq)]
pub struct PathOptions {
    /// Number of random trees to build.
    pub trials: usize,
    /// Stop starting new trials after this many seconds.
    pub max_secs: f64,
    /// Slice until every intermediate has at most `2^target_log2_size`
    /// entries (`None`: no slicing).
    pub target_log2_size: Option<f64>,
    /// Subtree reconfiguration of the best trees.
    pub reconf: bool,
    /// Frontier size of the reconfiguration.
    pub reconf_k: usize,
    /// Use the randomised greedy method.
    pub greedy: bool,
    /// Use recursive bisection.
    pub bisect: bool,
    /// RNG seed (the search is deterministic for a seed and thread count
    /// when `max_secs` does not bind).
    pub seed: u64,
    /// Number of best trees to refine.
    pub refine_top: usize,
    /// Frontier size of a final reconfiguration of the winning tree (0: none).
    pub polish_k: usize,
}

impl Default for PathOptions {
    fn default() -> Self {
        PathOptions {
            trials: 64,
            max_secs: 30.0,
            target_log2_size: None,
            reconf: true,
            reconf_k: 10,
            greedy: true,
            bisect: true,
            seed: 0x7e55_0001,
            refine_top: 8,
            polish_k: 12,
        }
    }
}

impl PathOptions {
    /// A quick search: a few greedy trees plus one bisection, no refinement
    /// beyond one reconfiguration pass (used by the planner's cost model).
    pub fn quick() -> Self {
        PathOptions {
            trials: 8,
            max_secs: 1.0,
            refine_top: 1,
            reconf_k: 8,
            polish_k: 0,
            ..Default::default()
        }
    }
}

/// What [`search`] found.
#[derive(Clone, Debug)]
pub struct PathStats {
    /// log10 of the cost of the unsliced tree (complex multiply-adds).
    pub log10_flops: f64,
    /// log2 of the largest intermediate of the unsliced tree (entries).
    pub log2_max_size: f64,
    /// log10 of the total cost over all slices.
    pub log10_sliced_flops: f64,
    /// log2 of the largest intermediate of one slice.
    pub log2_sliced_max_size: f64,
    /// Number of slices.
    pub slices: f64,
    /// Sliced total / unsliced cost of the final tree.
    pub overhead: f64,
    /// Trees built.
    pub trials: usize,
    /// Seconds spent searching.
    pub secs: f64,
    /// The method of the winning tree (`"greedy"` or `"bisect"`).
    pub method: &'static str,
}

/// A contraction tree, its sliced indices and their statistics.
#[derive(Clone, Debug)]
pub struct Path {
    /// The tree.
    pub tree: ContractionTree,
    /// Sliced flag of every index id.
    pub sliced: Vec<bool>,
    /// Costs.
    pub stats: PathStats,
}

/// The hyper-parameters of one trial.
#[derive(Clone, Copy, Debug)]
struct TrialParams {
    bisect: Option<BisectParams>,
    greedy: GreedyParams,
    seed: u64,
}

impl TrialParams {
    fn plain_greedy(seed: u64) -> Self {
        TrialParams {
            bisect: None,
            greedy: GreedyParams::default(),
            seed,
        }
    }

    fn random(method: &str, rng: &mut StdRng) -> Self {
        let greedy = GreedyParams {
            alpha: rng.random_range(0.0..1.5),
            temperature: rng.random_range(0.0..1.0f64).powi(2),
        };
        let bisect = (method == "bisect").then(|| BisectParams {
            // log-uniform in [0.01, 0.9]
            imbalance: (rng.random_range(0.01f64.ln()..0.9f64.ln())).exp(),
            imbalance_decay: rng.random_range(-3.0..3.0),
            jitter: rng.random_range(0.0..0.5),
            cutoff: rng.random_range(4..=32),
            greedy: GreedyParams {
                alpha: rng.random_range(0.0..1.2),
                temperature: rng.random_range(0.0..0.5f64).powi(2),
            },
        });
        TrialParams {
            bisect,
            greedy,
            seed: rng.random(),
        }
    }

    fn perturb(&self, rng: &mut StdRng) -> Self {
        let mut j = |x: f64, lo: f64, hi: f64| (x * rng.random_range(0.8..1.25) + rng.random_range(-0.02..0.02)).clamp(lo, hi);
        let greedy = GreedyParams {
            alpha: j(self.greedy.alpha, 0.0, 2.0),
            temperature: j(self.greedy.temperature, 0.0, 1.0),
        };
        let bisect = self.bisect.map(|b| BisectParams {
            imbalance: j(b.imbalance, 0.005, 0.9),
            imbalance_decay: j(b.imbalance_decay, -5.0, 5.0),
            jitter: j(b.jitter, 0.0, 1.0),
            cutoff: ((b.cutoff as f64) * j(1.0, 0.5, 2.0)).round().clamp(2.0, 40.0) as usize,
            greedy: GreedyParams {
                alpha: j(b.greedy.alpha, 0.0, 2.0),
                temperature: j(b.greedy.temperature, 0.0, 1.0),
            },
        });
        TrialParams {
            bisect,
            greedy,
            seed: rng.random(),
        }
    }

    fn build(&self, hg: &Hypergraph) -> ContractionTree {
        match &self.bisect {
            Some(b) => bisection(hg, b, self.seed),
            None => greedy(hg, &self.greedy, self.seed),
        }
    }

    fn method(&self) -> &'static str {
        if self.bisect.is_some() {
            "bisect"
        } else {
            "greedy"
        }
    }
}

#[derive(Clone)]
struct Trial {
    score: f64,
    tree: ContractionTree,
    method: &'static str,
}

fn score_of(hg: &Hypergraph, tree: &ContractionTree, target: Option<f64>) -> f64 {
    match target {
        None => {
            let tc = tree_cost(hg, tree, &[]);
            // flops, with a small penalty on width to break ties
            tc.log2_total + 1e-3 * tc.log2_max
        }
        Some(t) => {
            let mut tr = tree.clone();
            let s = slice_tree(hg, &mut tr, t, false);
            let tc = tree_cost(hg, &tr, &s);
            tc.log2_total + tc.log2_slices
        }
    }
}

/// Random hyper-parameter search over greedy and bisection trees, then
/// reconfiguration and (with a target) slicing of the best ones.
pub fn search(hg: &Hypergraph, opts: &PathOptions) -> Path {
    let t0 = Instant::now();
    let n = hg.n_leaves();
    if n <= 1 {
        let tree = ContractionTree {
            n_leaves: n,
            children: vec![],
        };
        let tc = tree_cost(hg, &tree, &[]);
        let l10 = if n == 0 { 0.0 } else { 0.0f64.max(tc.log2_total) };
        return Path {
            sliced: vec![false; hg.log2dim.len()],
            stats: PathStats {
                log10_flops: l10,
                log2_max_size: tc.log2_max.max(0.0),
                log10_sliced_flops: l10,
                log2_sliced_max_size: tc.log2_max.max(0.0),
                slices: 1.0,
                overhead: 1.0,
                trials: 0,
                secs: t0.elapsed().as_secs_f64(),
                method: "trivial",
            },
            tree,
        };
    }
    let methods: Vec<&'static str> = match (opts.greedy, opts.bisect) {
        (true, true) | (false, false) => vec!["greedy", "bisect"],
        (true, false) => vec!["greedy"],
        (false, true) => vec!["bisect"],
    };
    // rounds: the first half of the trials draws parameters at random; each
    // later round perturbs the best parameter sets so far for half of its
    // trials (a small evolution strategy over the hyper-parameters)
    let total = opts.trials.max(1);
    let rounds: Vec<usize> = if total >= 16 {
        vec![total / 2, total / 4, total - total / 2 - total / 4]
    } else {
        vec![total]
    };
    let mut trials: Vec<(TrialParams, Trial)> = Vec::new();
    let mut next = 0usize;
    for (r, &count) in rounds.iter().enumerate() {
        let mut elite: Vec<TrialParams> = {
            let mut v: Vec<&(TrialParams, Trial)> = trials.iter().collect();
            v.sort_by(|a, b| a.1.score.total_cmp(&b.1.score));
            v.iter().take(4).map(|x| x.0).collect()
        };
        if r == 0 {
            elite.clear();
        }
        let first = next;
        next += count;
        let batch: Vec<(TrialParams, Trial)> = (first..first + count)
            .into_par_iter()
            .filter_map(|t| {
                if t > 0 && t0.elapsed().as_secs_f64() > opts.max_secs {
                    return None;
                }
                let mut rng =
                    StdRng::seed_from_u64(opts.seed ^ (t as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
                let params = if !elite.is_empty() && t % 2 == 0 {
                    elite[(t / 2) % elite.len()].perturb(&mut rng)
                } else if t == 0 {
                    TrialParams::plain_greedy(rng.random())
                } else {
                    TrialParams::random(methods[t % methods.len()], &mut rng)
                };
                let tree = params.build(hg);
                debug_assert!(tree.is_valid());
                let score = score_of(hg, &tree, opts.target_log2_size);
                Some((
                    params,
                    Trial {
                        score,
                        tree,
                        method: params.method(),
                    },
                ))
            })
            .collect();
        trials.extend(batch);
    }
    let trials: Vec<Trial> = trials.into_iter().map(|x| x.1).collect();
    let n_trials = trials.len();
    let mut ranked = trials;
    ranked.sort_by(|a, b| a.score.total_cmp(&b.score));
    ranked.truncate(opts.refine_top.max(1));
    let refined: Vec<(f64, Path)> = ranked
        .into_par_iter()
        .map(|tr| {
            let mut tree = tr.tree;
            if opts.reconf {
                reconfigure(hg, &mut tree, &[], opts.reconf_k, None, 4);
            }
            let unsliced = tree_cost(hg, &tree, &[]);
            let sliced = match opts.target_log2_size {
                Some(t) => slice_tree(hg, &mut tree, t, opts.reconf),
                None => vec![false; hg.log2dim.len()],
            };
            if opts.reconf && opts.target_log2_size.is_some() {
                let lim = tree_cost(hg, &tree, &sliced).log2_max;
                reconfigure(hg, &mut tree, &sliced, opts.reconf_k, Some(lim), 2);
            }
            let tc = tree_cost(hg, &tree, &sliced);
            // unsliced numbers of the final tree
            let un = tree_cost(hg, &tree, &[]);
            let un = if opts.target_log2_size.is_some() { un } else { unsliced };
            let l2 = std::f64::consts::LOG10_2;
            let stats = PathStats {
                log10_flops: un.log2_total * l2,
                log2_max_size: un.log2_max,
                log10_sliced_flops: tc.log10_total(),
                log2_sliced_max_size: tc.log2_max,
                slices: tc.log2_slices.exp2(),
                overhead: ((tc.log2_total + tc.log2_slices) - un.log2_total).exp2(),
                trials: n_trials,
                secs: 0.0,
                method: tr.method,
            };
            let score = tc.log2_total + tc.log2_slices + 1e-3 * tc.log2_max;
            (score, Path { tree, sliced, stats })
        })
        .collect();
    let mut best = refined
        .into_iter()
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .expect("at least one trial")
        .1;
    if opts.reconf && opts.polish_k > opts.reconf_k {
        let tc = tree_cost(hg, &best.tree, &best.sliced);
        let lim = opts.target_log2_size.map(|_| tc.log2_max);
        reconfigure(hg, &mut best.tree, &best.sliced, opts.polish_k, lim, 2);
        let tc = tree_cost(hg, &best.tree, &best.sliced);
        let un = tree_cost(hg, &best.tree, &[]);
        let l2 = std::f64::consts::LOG10_2;
        best.stats.log10_flops = un.log2_total * l2;
        best.stats.log2_max_size = un.log2_max;
        best.stats.log10_sliced_flops = tc.log10_total();
        best.stats.log2_sliced_max_size = tc.log2_max;
        best.stats.overhead = ((tc.log2_total + tc.log2_slices) - un.log2_total).exp2();
    }
    best.stats.secs = t0.elapsed().as_secs_f64();
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(n: usize) -> Hypergraph {
        // tensor k holds indices k and k+1 (mod n), plus an output on 0
        let inputs: Vec<Vec<u32>> = (0..n)
            .map(|k| vec![k as u32, ((k + 1) % n) as u32])
            .collect();
        Hypergraph::new(inputs, vec![], vec![1.0; n])
    }

    #[test]
    fn trees_are_valid_and_reconf_never_worsens() {
        let hg = ring(40);
        for seed in 0..4 {
            let mut t = greedy(
                &hg,
                &GreedyParams {
                    alpha: 0.5,
                    temperature: 0.3,
                },
                seed,
            );
            assert!(t.is_valid());
            let before = tree_cost(&hg, &t, &[]).log2_total;
            reconfigure(&hg, &mut t, &[], 8, None, 3);
            assert!(t.is_valid());
            assert!(tree_cost(&hg, &t, &[]).log2_total <= before + 1e-9);
            let b = bisection(&hg, &BisectParams::default(), seed);
            assert!(b.is_valid());
        }
    }

    #[test]
    fn slicing_meets_the_target() {
        let hg = ring(30);
        let mut t = greedy(&hg, &GreedyParams::default(), 1);
        let s = slice_tree(&hg, &mut t, 1.0, true);
        let tc = tree_cost(&hg, &t, &s);
        assert!(tc.log2_max <= 1.0 + 1e-9);
        assert!(t.is_valid());
    }
}
