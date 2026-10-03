//! Exact circuit-level distance of a detector error model by branch and
//! bound, with an exact count of minimum-weight logical fault sets.
//!
//! A *logical* is a set `S` of mechanisms whose detector sets XOR to empty and
//! whose observable bits XOR to 1. The search for `|S| <= w`:
//!
//! * the smallest-index observable-flipping mechanism `j0` of `S` is chosen at
//!   the root (one branch per `j0`; observable mechanisms below `j0` are then
//!   excluded);
//! * at every node, the fired detector set `F` of the mechanisms chosen so far
//!   is non-empty unless done; the fired detector `f` with the fewest
//!   still-allowed mechanisms is picked, and the node branches on each allowed
//!   mechanism `a` touching `f`. After branch `a` returns, `a` is excluded from
//!   the later sibling branches.
//!
//! Every logical containing the chosen set and avoiding the excluded set
//! contains exactly one "first" allowed mechanism touching `f`, so the branches
//! partition the solutions: each logical of weight `w` is found exactly once,
//! which makes the count exact. Pruning: `|S_sofar| + ceil(|F| / maxdeg) <= w`.
//! A node whose `F` is empty with observable parity 0 is cut: a minimum-weight
//! logical never contains a non-empty zero-syndrome subset.

/// Result of [`min_logical`].
#[derive(Clone, Debug, PartialEq)]
pub struct MinLogical {
    /// Minimum weight (`None` if none up to the limit).
    pub weight: Option<usize>,
    /// Number of distinct minimum-weight logicals (capped at `count_cap`).
    pub count: u64,
    /// One minimum-weight logical (mechanism indices).
    pub example: Vec<usize>,
    /// Search nodes visited.
    pub nodes: u64,
}

struct Search<'a> {
    dets: &'a [Vec<u32>],
    obs: &'a [bool],
    by_det: Vec<Vec<u32>>,
    words: usize,
    maxdeg: usize,
    excluded: Vec<bool>,
    used: Vec<bool>,
    chosen: Vec<usize>,
    count: u64,
    cap: u64,
    example: Vec<usize>,
    nodes: u64,
    node_limit: u64,
}

impl Search<'_> {
    fn toggle(&self, f: &mut [u64], j: usize) {
        for &i in &self.dets[j] {
            f[i as usize / 64] ^= 1u64 << (i % 64);
        }
    }

    fn popcount(f: &[u64]) -> usize {
        f.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Explore with budget `w` (total weight). Returns false if aborted.
    fn dfs(&mut self, f: &mut Vec<u64>, obs: bool, w: usize) -> bool {
        self.nodes += 1;
        if self.nodes > self.node_limit {
            return false;
        }
        let k = self.chosen.len();
        let nf = Self::popcount(f);
        if nf == 0 {
            if obs && k > 0 {
                if self.count == 0 {
                    self.example = self.chosen.clone();
                }
                self.count += 1;
            }
            return true;
        }
        if k + nf.div_ceil(self.maxdeg) > w {
            return true;
        }
        // fired detector with fewest allowed mechanisms
        let mut best_i = usize::MAX;
        let mut best_n = usize::MAX;
        for (wi, &word) in f.iter().enumerate() {
            let mut x = word;
            while x != 0 {
                let i = wi * 64 + x.trailing_zeros() as usize;
                x &= x - 1;
                let n = self.by_det[i]
                    .iter()
                    .filter(|&&a| !self.excluded[a as usize] && !self.used[a as usize])
                    .count();
                if n < best_n {
                    best_n = n;
                    best_i = i;
                    if n <= 1 {
                        break;
                    }
                }
            }
            if best_n <= 1 {
                break;
            }
        }
        if best_n == 0 {
            return true;
        }
        let cands: Vec<u32> = self.by_det[best_i]
            .iter()
            .copied()
            .filter(|&a| !self.excluded[a as usize] && !self.used[a as usize])
            .collect();
        let mut newly_excluded = Vec::new();
        let mut ok = true;
        for &a in &cands {
            let a = a as usize;
            self.used[a] = true;
            self.chosen.push(a);
            self.toggle(f, a);
            let cont = self.dfs(f, obs ^ self.obs[a], w);
            self.toggle(f, a);
            self.chosen.pop();
            self.used[a] = false;
            self.excluded[a] = true;
            newly_excluded.push(a);
            if !cont || self.count >= self.cap {
                ok = cont;
                break;
            }
        }
        for a in newly_excluded {
            self.excluded[a] = false;
        }
        ok
    }
}

/// Minimum weight of a logical in the DEM `(detectors, flips_observable)`,
/// searching weights `1..=max_weight`, counting minimum-weight logicals up to
/// `count_cap`. `node_limit` bounds the total search (returns `weight: None`
/// with `nodes > node_limit` if exceeded).
pub fn min_logical(
    num_detectors: usize,
    dets: &[Vec<u32>],
    obs: &[bool],
    max_weight: usize,
    count_cap: u64,
    node_limit: u64,
) -> MinLogical {
    let mut by_det: Vec<Vec<u32>> = vec![Vec::new(); num_detectors];
    for (j, ds) in dets.iter().enumerate() {
        for &i in ds {
            by_det[i as usize].push(j as u32);
        }
    }
    let words = num_detectors.div_ceil(64).max(1);
    let maxdeg = dets.iter().map(|d| d.len()).max().unwrap_or(1).max(1);
    let mut s = Search {
        dets,
        obs,
        by_det,
        words,
        maxdeg,
        excluded: vec![false; dets.len()],
        used: vec![false; dets.len()],
        chosen: Vec::new(),
        count: 0,
        cap: count_cap,
        example: Vec::new(),
        nodes: 0,
        node_limit,
    };
    let obs_mechs: Vec<usize> = (0..dets.len()).filter(|&j| obs[j]).collect();
    for w in 1..=max_weight {
        // root: branch on the smallest-index observable mechanism j0
        for e in s.excluded.iter_mut() {
            *e = false;
        }
        let mut aborted = false;
        for &j0 in &obs_mechs {
            let mut f = vec![0u64; s.words];
            s.used[j0] = true;
            s.chosen.push(j0);
            s.toggle(&mut f, j0);
            let ok = s.dfs(&mut f, true, w);
            s.chosen.pop();
            s.used[j0] = false;
            s.excluded[j0] = true;
            if !ok {
                aborted = true;
                break;
            }
            if s.count >= s.cap {
                break;
            }
        }
        if aborted {
            return MinLogical {
                weight: None,
                count: 0,
                example: Vec::new(),
                nodes: s.nodes,
            };
        }
        if s.count > 0 {
            return MinLogical {
                weight: Some(w),
                count: s.count,
                example: s.example.clone(),
                nodes: s.nodes,
            };
        }
    }
    MinLogical {
        weight: None,
        count: 0,
        example: Vec::new(),
        nodes: s.nodes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn brute(nd: usize, dets: &[Vec<u32>], obs: &[bool]) -> (Option<usize>, u64) {
        let m = dets.len();
        let mut best = None;
        let mut cnt = 0;
        for mask in 1u32..(1 << m) {
            let mut f = vec![false; nd];
            let mut o = false;
            for j in 0..m {
                if mask >> j & 1 == 1 {
                    for &i in &dets[j] {
                        f[i as usize] ^= true;
                    }
                    o ^= obs[j];
                }
            }
            if o && f.iter().all(|&x| !x) {
                let w = mask.count_ones() as usize;
                match best {
                    None => {
                        best = Some(w);
                        cnt = 1
                    }
                    Some(b) if w < b => {
                        best = Some(w);
                        cnt = 1
                    }
                    Some(b) if w == b => cnt += 1,
                    _ => {}
                }
            }
        }
        (best, cnt)
    }

    #[test]
    fn matches_brute_force_on_random_dems() {
        let mut rng = StdRng::seed_from_u64(3);
        for _ in 0..300 {
            let nd = rng.random_range(1..8usize);
            let m = rng.random_range(1..14usize);
            let dets: Vec<Vec<u32>> = (0..m)
                .map(|_| {
                    let mut v: Vec<u32> = (0..nd as u32).filter(|_| rng.random_bool(0.3)).collect();
                    v.dedup();
                    v
                })
                .collect();
            let obs: Vec<bool> = (0..m).map(|_| rng.random_bool(0.3)).collect();
            let (bw, bc) = brute(nd, &dets, &obs);
            let r = min_logical(nd, &dets, &obs, m, u64::MAX, u64::MAX);
            assert_eq!(r.weight, bw, "{dets:?} {obs:?}");
            if bw.is_some() {
                assert_eq!(r.count, bc, "{dets:?} {obs:?}");
            }
        }
    }
}
