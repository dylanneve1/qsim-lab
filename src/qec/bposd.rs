//! Belief propagation + ordered-statistics decoding (BP+OSD-CS) on a
//! circuit-derived detector error model (hypergraph: a mechanism may flip any
//! number of detectors), after Roffe et al., "Decoding across the quantum
//! LDPC code landscape" (arXiv:2005.07016).
//!
//! * BP: flooding normalised min-sum on the Tanner graph of the DEM check
//!   matrix `H` (detectors x mechanisms) with priors `ln((1-p)/p)`.
//! * If the hard decision reproduces the syndrome, it is the correction.
//! * Otherwise OSD: columns sorted by BP posterior (most likely first),
//!   Gaussian elimination picks an information set, OSD-0 solves on it, then
//!   the combination sweep tries flipping each single and each pair of the
//!   `order` most likely non-pivot columns; the lowest-weight solution
//!   (weights = prior LLRs) wins.
//!
//! The decoder returns the predicted observable flips of its correction.
#![allow(clippy::needless_range_loop)]

/// A detector error model as a sparse matrix.
#[derive(Clone, Debug)]
pub struct DemMatrix {
    pub num_detectors: usize,
    /// Detectors flipped by each mechanism.
    pub cols: Vec<Vec<u32>>,
    /// Observable mask of each mechanism.
    pub obs: Vec<u64>,
    /// Probability of each mechanism.
    pub p: Vec<f64>,
}

/// BP+OSD decoder (immutable; per-thread scratch lives in [`BpOsdScratch`]).
#[derive(Clone, Debug)]
pub struct BpOsd {
    m: DemMatrix,
    prior: Vec<f64>,
    pub max_iter: usize,
    pub ms_scale: f64,
    pub osd_order: usize,
    /// edge index layout: edges are numbered column-major; `col_edge[j]` is
    /// the first edge of column j; `row_edges[i]` lists edge ids of row i
    col_edge: Vec<usize>,
    row_edges: Vec<Vec<u32>>,
}

/// Per-thread working memory.
pub struct BpOsdScratch {
    q: Vec<f64>, // variable->check messages (per edge)
    r: Vec<f64>, // check->variable messages (per edge)
    post: Vec<f64>,
    hard: Vec<bool>,
    words: usize,
    mat: Vec<u64>, // dense rows (for OSD), num_detectors x words
}

/// Decoder statistics.
#[derive(Clone, Copy, Debug, Default)]
pub struct DecodeStats {
    pub bp_converged: u64,
    pub osd_calls: u64,
}

impl BpOsd {
    pub fn new(m: DemMatrix, max_iter: usize, ms_scale: f64, osd_order: usize) -> Self {
        let nd = m.num_detectors;
        let mut row_edges: Vec<Vec<u32>> = vec![Vec::new(); nd];
        let mut col_edge = Vec::with_capacity(m.cols.len() + 1);
        let mut e = 0usize;
        for c in m.cols.iter() {
            col_edge.push(e);
            for &i in c {
                row_edges[i as usize].push(e as u32);
                e += 1;
            }
        }
        col_edge.push(e);
        let prior =
            m.p.iter()
                .map(|&p| {
                    let p = p.clamp(1e-15, 0.5 - 1e-12);
                    ((1.0 - p) / p).ln()
                })
                .collect();
        BpOsd {
            m,
            prior,
            max_iter,
            ms_scale,
            osd_order,
            col_edge,
            row_edges,
        }
    }

    pub fn num_mechanisms(&self) -> usize {
        self.m.cols.len()
    }

    pub fn scratch(&self) -> BpOsdScratch {
        let ne = *self.col_edge.last().unwrap();
        let n = self.m.cols.len();
        let words = n.div_ceil(64);
        BpOsdScratch {
            q: vec![0.0; ne],
            r: vec![0.0; ne],
            post: vec![0.0; n],
            hard: vec![false; n],
            words,
            mat: vec![0; self.m.num_detectors * words],
        }
    }

    /// Decodes a syndrome given as the sorted list of fired detectors;
    /// returns the predicted observable mask.
    pub fn decode(&self, fired: &[u32], s: &mut BpOsdScratch, st: &mut DecodeStats) -> u64 {
        if fired.is_empty() {
            st.bp_converged += 1;
            return 0;
        }
        let nd = self.m.num_detectors;
        let n = self.m.cols.len();
        let mut syn = vec![false; nd];
        for &i in fired {
            syn[i as usize] = true;
        }
        // init
        for j in 0..n {
            for e in self.col_edge[j]..self.col_edge[j + 1] {
                s.q[e] = self.prior[j];
            }
        }
        let mut converged = false;
        for _ in 0..self.max_iter {
            // check update (min-sum)
            for i in 0..nd {
                let es = &self.row_edges[i];
                if es.is_empty() {
                    continue;
                }
                let mut sign = syn[i];
                let (mut m1, mut m2) = (f64::INFINITY, f64::INFINITY);
                let mut arg = usize::MAX;
                for (k, &e) in es.iter().enumerate() {
                    let v = s.q[e as usize];
                    if v < 0.0 {
                        sign = !sign;
                    }
                    let a = v.abs();
                    if a < m1 {
                        m2 = m1;
                        m1 = a;
                        arg = k;
                    } else if a < m2 {
                        m2 = a;
                    }
                }
                for (k, &e) in es.iter().enumerate() {
                    let v = s.q[e as usize];
                    let mag = if k == arg { m2 } else { m1 };
                    let sg = sign ^ (v < 0.0);
                    s.r[e as usize] = self.ms_scale * if sg { -mag } else { mag };
                }
            }
            // variable update + hard decision
            for j in 0..n {
                let (a, b) = (self.col_edge[j], self.col_edge[j + 1]);
                let mut tot = self.prior[j];
                for e in a..b {
                    tot += s.r[e];
                }
                s.post[j] = tot;
                s.hard[j] = tot < 0.0;
                for e in a..b {
                    s.q[e] = tot - s.r[e];
                }
            }
            // syndrome check
            let mut ok = true;
            let mut chk = vec![false; nd];
            for j in 0..n {
                if s.hard[j] {
                    for &i in &self.m.cols[j] {
                        chk[i as usize] ^= true;
                    }
                }
            }
            if chk != syn {
                ok = false;
            }
            if ok {
                converged = true;
                break;
            }
        }
        if converged {
            st.bp_converged += 1;
            let mut o = 0u64;
            for j in 0..n {
                if s.hard[j] {
                    o ^= self.m.obs[j];
                }
            }
            return o;
        }
        st.osd_calls += 1;
        self.osd(&syn, s)
    }

    fn osd(&self, syn: &[bool], s: &mut BpOsdScratch) -> u64 {
        let nd = self.m.num_detectors;
        let n = self.m.cols.len();
        let w = s.words;
        // order: most likely error first (smallest posterior LLR)
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by(|&a, &b| s.post[a as usize].partial_cmp(&s.post[b as usize]).unwrap());
        // dense rows over original column indices, plus syndrome
        s.mat.iter_mut().for_each(|x| *x = 0);
        for (j, c) in self.m.cols.iter().enumerate() {
            for &i in c {
                s.mat[i as usize * w + j / 64] |= 1 << (j % 64);
            }
        }
        let mut sv: Vec<bool> = syn.to_vec();
        let mut row_used = vec![false; nd];
        let mut piv_cols: Vec<(u32, u32)> = Vec::new(); // (col, row)
        let mut nonpiv: Vec<u32> = Vec::new();
        for &j in &order {
            let (wj, bj) = (j as usize / 64, 1u64 << (j % 64));
            let mut pr = usize::MAX;
            for i in 0..nd {
                if !row_used[i] && s.mat[i * w + wj] & bj != 0 {
                    pr = i;
                    break;
                }
            }
            if pr == usize::MAX {
                if nonpiv.len() < self.osd_order {
                    nonpiv.push(j);
                }
                continue;
            }
            row_used[pr] = true;
            // eliminate column j from all other rows
            let (before, rest) = s.mat.split_at_mut(pr * w);
            let (prow, after) = rest.split_at_mut(w);
            for i in 0..nd {
                if i == pr {
                    continue;
                }
                let row: &mut [u64] = if i < pr {
                    &mut before[i * w..(i + 1) * w]
                } else {
                    let k = i - pr - 1;
                    &mut after[k * w..(k + 1) * w]
                };
                if row[wj] & bj != 0 {
                    for (a, b) in row.iter_mut().zip(prow.iter()) {
                        *a ^= b;
                    }
                    sv[i] ^= sv[pr];
                }
            }
            piv_cols.push((j, pr as u32));
            if piv_cols.len() == nd {
                break;
            }
        }
        // remaining non-pivot candidates if we stopped early
        if nonpiv.len() < self.osd_order {
            let is_piv: std::collections::HashSet<u32> = piv_cols.iter().map(|x| x.0).collect();
            for &j in &order {
                if nonpiv.len() >= self.osd_order {
                    break;
                }
                if !is_piv.contains(&j) && !nonpiv.contains(&j) {
                    nonpiv.push(j);
                }
            }
        }
        // evaluate candidate: flips T among nonpiv
        let eval = |t: &[u32], s: &BpOsdScratch| -> (f64, u64) {
            let mut cost = 0.0;
            let mut obs = 0u64;
            for &j in t {
                cost += self.prior[j as usize];
                obs ^= self.m.obs[j as usize];
            }
            for &(c, r) in &piv_cols {
                let r = r as usize;
                let mut bit = sv[r];
                for &j in t {
                    if s.mat[r * w + j as usize / 64] >> (j % 64) & 1 == 1 {
                        bit ^= true;
                    }
                }
                if bit {
                    cost += self.prior[c as usize];
                    obs ^= self.m.obs[c as usize];
                }
            }
            (cost, obs)
        };
        let mut best = eval(&[], s);
        for a in 0..nonpiv.len() {
            let c = eval(&[nonpiv[a]], s);
            if c.0 < best.0 {
                best = c;
            }
            for b in a + 1..nonpiv.len() {
                let c = eval(&[nonpiv[a], nonpiv[b]], s);
                if c.0 < best.0 {
                    best = c;
                }
            }
        }
        best.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Repetition code of length 5 (code capacity): decodes all weight <= 2.
    #[test]
    fn repetition_code_corrects_up_to_two() {
        let n = 5;
        let cols: Vec<Vec<u32>> = (0..n)
            .map(|j| {
                let mut v = vec![];
                if j > 0 {
                    v.push(j as u32 - 1);
                }
                if j < n - 1 {
                    v.push(j as u32);
                }
                v
            })
            .collect();
        let obs: Vec<u64> = (0..n).map(|j| (j == 0) as u64).collect();
        let m = DemMatrix {
            num_detectors: n - 1,
            cols: cols.clone(),
            obs: obs.clone(),
            p: vec![0.05; n],
        };
        let dec = BpOsd::new(m, 20, 0.625, 4);
        let mut s = dec.scratch();
        let mut st = DecodeStats::default();
        for e in 0u32..(1 << n) {
            if e.count_ones() > 2 {
                continue;
            }
            let mut syn = vec![false; n - 1];
            let mut o = 0;
            for j in 0..n {
                if e >> j & 1 == 1 {
                    for &i in &cols[j] {
                        syn[i as usize] ^= true;
                    }
                    o ^= obs[j];
                }
            }
            let fired: Vec<u32> = (0..n as u32 - 1).filter(|&i| syn[i as usize]).collect();
            assert_eq!(dec.decode(&fired, &mut s, &mut st), o, "error {e:b}");
        }
    }
}
