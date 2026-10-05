//! Fast batched detector sampler: Poisson "hits" with uniform locations.
//!
//! [`super::symphase::SymPhaseSampler`] reduces a noisy Clifford circuit to
//! `m = m_ref xor A v` over independent noise variable groups `v`. Its
//! original sampling loop draws every fault by geometric skipping (a float
//! `ln` per fault, plus a separate draw of the Pauli) into a variable buffer
//! and then evaluates `A v`. At surface-code noise rates that loop is 70-80%
//! of the run time. This module draws the same distribution differently.
//!
//! # The hit model (exact)
//!
//! Take a group with `m` non-identity patterns (a bit flip: `m = 1`; 1-qubit
//! depolarizing: `m = 3`; 2-qubit depolarizing: `m = 15`), each pattern with
//! probability `p / m`. The patterns plus identity form the group `Z_2^b`
//! (`m + 1 = 2^b`) under XOR. Replace the group by a Poisson process of
//! *hits*: the number of hits on a cell (one group in one shot) is
//! `Poisson(lambda)`, every hit XORs in an independent pattern drawn
//! uniformly from the `m` non-identity ones. The net pattern of a cell is a
//! continuous-time random walk on `Z_2^b` started at 0. By symmetry (GL(b,2)
//! permutes the non-zero vectors transitively and fixes the step law), every
//! non-identity net pattern has the same probability, and Fourier analysis
//! gives
//!
//! ```text
//! P(net = 0) = 2^-b (1 + m exp(-lambda (m + 1) / m)),
//! ```
//!
//! so `lambda = -(m / (m + 1)) ln(1 - (m + 1) p / m)` reproduces `1 - p`
//! exactly, hence the whole group distribution (`p < m / (m + 1)`; heavier
//! groups take the dense path below). Cells are independent because Poisson
//! counts on disjoint cells of one Poisson process are independent.
//!
//! # Drawing it
//!
//! All groups with the same `(kind, p)` form a *class* of `G` groups. For a
//! batch of `S = 64 W` shots the class is `G * S` cells; a Poisson process
//! with rate `lambda` per cell is equivalently: a total count
//! `K ~ Poisson(G S lambda)`, then `K` independent uniform cells, each with
//! an independent uniform pattern. A (cell, pattern) pair is one uniform
//! integer in `[0, G S m)` (Lemire's multiply-shift with its exact rejection
//! step), split into shot, group and pattern by shifts and one constant
//! division. Each hit XORs the shot bit into the output words of the rows in
//! the columns of the pattern's variables directly: there is no variable
//! buffer to clear or scan, no `ln` per fault, and the Pauli comes from the
//! same random word as the location. All equal-probability locations of a
//! kind share one stream, however they are interleaved in the circuit.
//!
//! Coins (probability 1/2) and groups with large `p` are drawn densely, one
//! random word (coins) or one comparison per shot (heavy groups).
//!
//! The output is not bit-identical to the old path (different draws); its
//! distribution is the same. `tests/engines/fast_sampler.rs` checks the hit algebra
//! exactly, the per-group pattern frequencies and detector statistics
//! against the old sampler, and `research/qec/fast-sampler.md` the 10^6-shot
//! equivalence with Stim.

use super::symphase::{SymPhaseSampler, VarDist};
use rand::RngCore;

/// Groups with `p` above this go to the dense path.
const RARE_P_MAX: f64 = 0.25;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Flip,
    Depol1,
    Depol2,
}

impl Kind {
    fn patterns(self) -> u64 {
        match self {
            Kind::Flip => 1,
            Kind::Depol1 => 3,
            Kind::Depol2 => 15,
        }
    }
}

/// Variable mask of non-identity pattern `k - 1` (`k` = 1..=m), in the
/// variable order of [`VarDist`]: `(x, z)` for depol1, `(x_a, z_a, x_b, z_b)`
/// for depol2.
const fn pauli_bits(k: usize) -> u8 {
    match k {
        1 => 0b01,
        2 => 0b11,
        3 => 0b10,
        _ => 0,
    }
}
const DEPOL1_MASK: [u8; 3] = [pauli_bits(1), pauli_bits(2), pauli_bits(3)];
const DEPOL2_MASK: [u8; 15] = {
    let mut t = [0u8; 15];
    let mut k = 1;
    while k < 16 {
        t[k - 1] = pauli_bits(k / 4) | (pauli_bits(k % 4) << 2);
        k += 1;
    }
    t
};

/// All rare groups of one `(kind, p)`.
#[derive(Clone, Debug)]
struct Class {
    kind: Kind,
    p: f64,
    /// Hit rate per cell.
    lambda: f64,
    /// First variable of each group.
    firsts: Vec<u32>,
    /// Padded hit table: entry `t = group * m + pattern` lists the rows the
    /// hit flips (XOR of the pattern's variable columns) in
    /// `table[t * stride..(t + 1) * stride]`, padded with the sink row.
    /// Empty (`stride == 0`) if some entry is wider than [`MAX_STRIDE`]; the
    /// column-by-column path is used then.
    table: Vec<u32>,
    stride: usize,
    /// Blocked generation (table path): the batch's slots
    /// `t * S + shot` (`S` shots per batch) are cut into blocks of
    /// `2^block_log2`; each block gets an independent `Poisson(mu_b)` count
    /// (`mu_b = lambda / m * 2^block_log2`) from `pois`, and its hits are
    /// uniform within the block, so table accesses walk forward.
    block_log2: u32,
    pois: PoissonTable,
    /// `table` with `u16` rows (when every row and the sink fit), for the
    /// blocked path: half the memory traffic.
    table16: Vec<u16>,
}

/// Per-class constants of [`blocked_hits`].
struct BlockArgs<'a> {
    range: u64,
    shot_bits: u32,
    stride: usize,
    lb: u32,
    rem_mu_per_slot: f64,
    pois: &'a PoissonTable,
}

/// Blocked hit generation for one class with a padded table of width `K`
/// (see [`Class`]): slot = `t * S + shot`, `t = group * m + pattern`.
#[inline(always)]
fn blocked_hits<T: Copy + Into<u64>, R: RngCore + ?Sized, const K: usize>(
    table: &[T],
    a: &BlockArgs,
    rng: &mut R,
    out: &mut [u64],
) {
    let shot_mask = (1u64 << a.shot_bits) - 1;
    let stride = a.stride;
    let hit = |slot: u64, out: &mut [u64]| {
        let shot = slot & shot_mask;
        let t = (slot >> a.shot_bits) as usize;
        let base = (shot >> 6) as usize * stride;
        let bit = 1u64 << (shot & 63);
        let e: &[T; K] = table[t * K..t * K + K].try_into().unwrap();
        let o = &mut out[base..base + stride];
        for &r in e {
            // SAFETY: table rows are <= rows < stride = o.len()
            unsafe {
                *o.get_unchecked_mut(r.into() as usize) ^= bit;
            }
        }
    };
    let lb = a.lb;
    let full = a.range >> lb;
    for b in 0..full {
        let k = a.pois.sample(rng.next_u64());
        for _ in 0..k {
            hit((b << lb) | (rng.next_u64() >> (64 - lb)), out);
        }
    }
    let rem = a.range - (full << lb);
    if rem > 0 {
        for _ in 0..poisson(rng, a.rem_mu_per_slot * rem as f64) {
            hit((full << lb) + uniform_below(rng, rem), out);
        }
    }
}

/// Widest padded hit-table entry.
const MAX_STRIDE: usize = 8;

/// Target mean hits per block of the blocked generator.
const BLOCK_MEAN: f64 = 8.0;

/// Inverse-CDF sampler for one fixed Poisson mean: `k` is the number of
/// thresholds `T_j <= u` for a uniform 64-bit `u`, where
/// `T_j = floor(P(X <= j) * 2^64)`; a 256-entry guide table on the top byte
/// of `u` gives the starting `j`. Exact up to the f64 CDF (relative error
/// ~1e-16) and the 2^-64 grid; the table stops where the remaining tail
/// mass is below 2^-64.
#[derive(Clone, Debug, Default)]
struct PoissonTable {
    thresholds: Vec<u64>,
    guide: Vec<u16>,
}

impl PoissonTable {
    fn new(mu: f64) -> PoissonTable {
        // pmf until it is negligible past the mode, renormalised so the last
        // threshold is exactly 2^64 (stored as u64::MAX)
        let mut pmf = vec![(-mu).exp()];
        let mut k = 0u32;
        while (k as f64) < mu || pmf[k as usize] > 1e-25 {
            k += 1;
            let next = pmf[k as usize - 1] * mu / k as f64;
            pmf.push(next);
        }
        let total: f64 = pmf.iter().sum();
        let mut thresholds = Vec::with_capacity(pmf.len());
        let mut cdf = 0.0;
        for (j, &q) in pmf.iter().enumerate() {
            cdf += q;
            let t = cdf / total * 18446744073709551616.0;
            if j + 1 == pmf.len() || t >= 18446744073709551615.0 {
                thresholds.push(u64::MAX);
                break;
            }
            thresholds.push(t as u64);
        }
        let mut guide = Vec::with_capacity(256);
        let mut j = 0usize;
        for b in 0..256u64 {
            let lo = b << 56;
            while j + 1 < thresholds.len() && thresholds[j] <= lo {
                j += 1;
            }
            guide.push(j as u16);
        }
        PoissonTable { thresholds, guide }
    }

    #[inline(always)]
    fn sample(&self, u: u64) -> usize {
        let mut k = self.guide[(u >> 56) as usize] as usize;
        // the last threshold is u64::MAX; u == u64::MAX (probability 2^-64)
        // stops there too
        while k + 1 < self.thresholds.len() && u >= self.thresholds[k] {
            k += 1;
        }
        k
    }
}

#[derive(Clone, Copy, Debug)]
struct DenseGroup {
    first: u32,
    dist: VarDist,
}

/// Batched sampler; see the module docs.
#[derive(Clone, Debug)]
pub struct FastSampler {
    rows: usize,
    /// `!0` where the reference bit is 1.
    reference: Vec<u64>,
    /// CSC of `A`: rows touched by variable `v` are
    /// `col_rows[col_start[v]..col_start[v + 1]]`.
    col_start: Vec<u32>,
    col_rows: Vec<u32>,
    classes: Vec<Class>,
    dense: Vec<DenseGroup>,
    /// Blocked hit generation on the table path (default; see [`Class`]).
    blocked: bool,
    /// Use the `u16` tables where available (default).
    narrow: bool,
}

/// `lambda` of the hit model for a group with `m` non-identity patterns.
pub fn hit_rate(p: f64, m: u64) -> f64 {
    let m = m as f64;
    -(m / (m + 1.0)) * (-(m + 1.0) * p / m).ln_1p()
}

/// `P(net pattern = identity)` after `Poisson(lambda)` uniform non-identity
/// hits on `Z_2^b`, `m = 2^b - 1` (closed form used in the docs and tests).
pub fn hit_identity_prob(lambda: f64, m: u64) -> f64 {
    let m = m as f64;
    (1.0 + m * (-lambda * (m + 1.0) / m).exp()) / (m + 1.0)
}

impl FastSampler {
    /// Builds the batched sampler for `s` (any row set: raw measurements or
    /// detector parities).
    pub fn new(s: &SymPhaseSampler) -> FastSampler {
        let rows = s.num_measurements();
        let nv = s.num_vars();
        let mut col_start = vec![0u32; nv + 1];
        for j in 0..rows {
            for &v in s.row(j) {
                col_start[v as usize + 1] += 1;
            }
        }
        for i in 0..nv {
            col_start[i + 1] += col_start[i];
        }
        let mut fill = col_start.clone();
        let mut col_rows = vec![0u32; col_start[nv] as usize];
        for j in 0..rows {
            for &v in s.row(j) {
                col_rows[fill[v as usize] as usize] = j as u32;
                fill[v as usize] += 1;
            }
        }
        let mut classes: Vec<Class> = Vec::new();
        let sink = rows as u32;
        let mut dense = Vec::new();
        for g in s.groups() {
            let (kind, p) = match g.dist {
                VarDist::Flip(p) => (Kind::Flip, p),
                VarDist::Depol1(p) => (Kind::Depol1, p),
                VarDist::Depol2(p) => (Kind::Depol2, p),
                VarDist::Coin => {
                    dense.push(DenseGroup {
                        first: g.first,
                        dist: g.dist,
                    });
                    continue;
                }
            };
            if p <= 0.0 {
                continue;
            }
            if p > RARE_P_MAX {
                dense.push(DenseGroup {
                    first: g.first,
                    dist: g.dist,
                });
                continue;
            }
            match classes.iter_mut().find(|c| c.kind == kind && c.p == p) {
                Some(c) => c.firsts.push(g.first),
                None => classes.push(Class {
                    kind,
                    p,
                    lambda: hit_rate(p, kind.patterns()),
                    firsts: vec![g.first],
                    table: Vec::new(),
                    stride: 0,
                    block_log2: 0,
                    pois: PoissonTable::default(),
                    table16: Vec::new(),
                }),
            }
        }
        for c in &mut classes {
            let masks: &[u8] = match c.kind {
                Kind::Flip => &[1],
                Kind::Depol1 => &DEPOL1_MASK,
                Kind::Depol2 => &DEPOL2_MASK,
            };
            // entries as CSR: XOR of the pattern's variable columns (rows
            // appearing an odd number of times), in t = group * m + pattern order
            let n_ent = c.firsts.len() * masks.len();
            let mut ent_start = Vec::with_capacity(n_ent + 1);
            ent_start.push(0usize);
            let mut ent_rows: Vec<u32> = Vec::with_capacity(4 * n_ent);
            let mut e: Vec<u32> = Vec::with_capacity(64);
            for &f in &c.firsts {
                for &mask in masks {
                    e.clear();
                    for k in 0..4 {
                        if mask >> k & 1 == 1 {
                            let v = f as usize + k;
                            e.extend_from_slice(
                                &col_rows[col_start[v] as usize..col_start[v + 1] as usize],
                            );
                        }
                    }
                    e.sort_unstable();
                    let base = ent_rows.len();
                    for &r in &e {
                        if ent_rows.len() > base && ent_rows.last() == Some(&r) {
                            ent_rows.pop();
                        } else {
                            ent_rows.push(r);
                        }
                    }
                    ent_start.push(ent_rows.len());
                }
            }
            let stride = ent_start
                .windows(2)
                .map(|w| w[1] - w[0])
                .max()
                .unwrap_or(0)
                .max(1);
            if stride <= MAX_STRIDE {
                let slot_rate = c.lambda / c.kind.patterns() as f64;
                c.block_log2 = (BLOCK_MEAN / slot_rate).log2().round().clamp(4.0, 40.0) as u32;
                c.pois = PoissonTable::new(slot_rate * (1u64 << c.block_log2) as f64);
                c.stride = stride;
                c.table = vec![sink; n_ent * stride];
                for t in 0..n_ent {
                    let e = &ent_rows[ent_start[t]..ent_start[t + 1]];
                    c.table[t * stride..t * stride + e.len()].copy_from_slice(e);
                }
                if sink < u16::MAX as u32 {
                    c.table16 = c.table.iter().map(|&r| r as u16).collect();
                }
            }
        }
        FastSampler {
            rows,
            reference: s
                .reference()
                .iter()
                .map(|&b| 0u64.wrapping_sub(b as u64))
                .collect(),
            col_start,
            col_rows,
            classes,
            dense,
            blocked: true,
            narrow: true,
        }
    }

    /// Output rows per shot (measurements or detectors).
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Words per 64-shot block in [`Self::sample_batch`]'s buffer:
    /// `rows + 1` (the last word is a scratch sink).
    pub fn stride(&self) -> usize {
        self.rows + 1
    }

    /// Appends the first `blocks` 64-shot blocks of a sampled buffer to
    /// `bytes` in Stim's ptb64 layout (little-endian `u64` per row).
    pub fn ptb64(&self, out: &[u64], blocks: usize, bytes: &mut Vec<u8>) {
        for blk in out.chunks_exact(self.stride()).take(blocks) {
            for w in &blk[..self.rows] {
                bytes.extend_from_slice(&w.to_le_bytes());
            }
        }
    }

    /// Number of rare-path classes (distinct `(kind, p)`) and dense groups.
    pub fn layout(&self) -> (usize, usize) {
        (self.classes.len(), self.dense.len())
    }

    /// Bytes of hit tables used by the default (blocked, `u16` where
    /// possible) path.
    pub fn table_bytes(&self) -> usize {
        self.classes
            .iter()
            .map(|c| {
                if c.table16.is_empty() {
                    4 * c.table.len()
                } else {
                    2 * c.table16.len()
                }
            })
            .sum()
    }

    /// Every distinct non-empty row set a single hit of the rare path can
    /// flip (one per (group, pattern) table entry, sink padding removed),
    /// sorted. Comparable with the error signatures of Stim's DEM.
    pub fn hit_signatures(&self) -> Vec<Vec<u32>> {
        let mut set = std::collections::BTreeSet::new();
        for c in &self.classes {
            for e in c.table.chunks(c.stride.max(1)) {
                let v: Vec<u32> = e
                    .iter()
                    .copied()
                    .filter(|&r| r < self.rows as u32)
                    .collect();
                if !v.is_empty() {
                    set.insert(v);
                }
            }
        }
        set.into_iter().collect()
    }

    /// Drops the padded hit tables so every class takes the column-by-column
    /// path (used when an entry is wider than `MAX_STRIDE`); for tests and
    /// the ablation in `research/qec/fast-sampler.md`.
    #[doc(hidden)]
    pub fn force_column_path(&mut self) {
        for c in &mut self.classes {
            c.table = Vec::new();
            c.table16 = Vec::new();
            c.stride = 0;
        }
    }

    /// One global Poisson count per class and batch with uniform hits over
    /// the whole class (instead of per-block counts); same distribution, for
    /// the ablation.
    #[doc(hidden)]
    pub fn set_blocked(&mut self, on: bool) {
        self.blocked = on;
    }

    /// Use `u32` hit tables even where `u16` ones exist (ablation).
    #[doc(hidden)]
    pub fn set_narrow(&mut self, on: bool) {
        self.narrow = on;
    }

    /// Expected number of hits per shot on the rare path, and the mean number
    /// of output-word XORs per hit.
    pub fn hit_stats(&self) -> (f64, f64) {
        let (mut hits, mut xors) = (0.0, 0.0);
        for c in &self.classes {
            let table: &[u8] = match c.kind {
                Kind::Flip => &[1],
                Kind::Depol1 => &DEPOL1_MASK,
                Kind::Depol2 => &DEPOL2_MASK,
            };
            for &f in &c.firsts {
                let mut w = 0.0;
                for &mask in table {
                    for k in 0..4 {
                        if mask >> k & 1 == 1 {
                            let v = f as usize + k;
                            w += (self.col_start[v + 1] - self.col_start[v]) as f64;
                        }
                    }
                }
                hits += c.lambda;
                xors += c.lambda * w / table.len() as f64;
            }
        }
        (hits, xors / hits.max(1e-300))
    }

    #[inline(always)]
    fn xor_col(&self, v: usize, out: &mut [u64], bit: u64) {
        let (a, b) = (self.col_start[v] as usize, self.col_start[v + 1] as usize);
        for &r in &self.col_rows[a..b] {
            // SAFETY: every row index is < self.rows (built from the
            // sampler's rows) and `out` has at least `self.rows` words
            // (checked in `sample_batch`).
            unsafe {
                *out.get_unchecked_mut(r as usize) ^= bit;
            }
        }
    }

    /// Samples `64 * W` shots, `W = out.len() / stride()` (a power of two):
    /// `out[w * stride() + r]` bit `s` is row `r` of shot `64 w + s`; the
    /// last word of each block is scratch. [`Self::ptb64`] packs it into
    /// Stim's ptb64 layout.
    pub fn sample_batch<R: RngCore + ?Sized>(&self, rng: &mut R, out: &mut [u64]) {
        let stride = self.stride();
        assert!(out.len() % stride == 0);
        let words = out.len() / stride;
        assert!(
            words.is_power_of_two(),
            "words per batch must be a power of two"
        );
        for chunk in out.chunks_exact_mut(stride) {
            chunk[..self.rows].copy_from_slice(&self.reference);
        }
        let shot_bits = words.trailing_zeros() + 6;
        let shot_mask = (1u64 << shot_bits) - 1;
        for c in &self.classes {
            let g = c.firsts.len() as u64;
            let m = c.kind.patterns();
            let range = (g * m) << shot_bits;
            if c.stride > 0 && self.blocked {
                let a = BlockArgs {
                    range,
                    shot_bits,
                    stride,
                    lb: c.block_log2,
                    rem_mu_per_slot: c.lambda / m as f64,
                    pois: &c.pois,
                };
                macro_rules! go {
                    ($t:expr) => {
                        match c.stride {
                            1 => blocked_hits::<_, _, 1>($t, &a, rng, out),
                            2 => blocked_hits::<_, _, 2>($t, &a, rng, out),
                            3 => blocked_hits::<_, _, 3>($t, &a, rng, out),
                            4 => blocked_hits::<_, _, 4>($t, &a, rng, out),
                            5 => blocked_hits::<_, _, 5>($t, &a, rng, out),
                            6 => blocked_hits::<_, _, 6>($t, &a, rng, out),
                            7 => blocked_hits::<_, _, 7>($t, &a, rng, out),
                            _ => blocked_hits::<_, _, 8>($t, &a, rng, out),
                        }
                    };
                }
                if self.narrow && !c.table16.is_empty() {
                    go!(&c.table16[..])
                } else {
                    go!(&c.table[..])
                }
                continue;
            }
            let k = poisson(rng, c.lambda * (g << shot_bits) as f64);
            if c.stride > 0 {
                // one table entry per (group, pattern); rows < stride by construction
                let w = c.stride;
                for _ in 0..k {
                    let idx = uniform_below(rng, range);
                    let shot = idx & shot_mask;
                    let t = (idx >> shot_bits) as usize;
                    let base = (shot >> 6) as usize * stride;
                    let bit = 1u64 << (shot & 63);
                    let e = &c.table[t * w..t * w + w];
                    let o = &mut out[base..base + stride];
                    for &r in e {
                        // SAFETY: table rows are <= self.rows < stride = o.len()
                        unsafe {
                            *o.get_unchecked_mut(r as usize) ^= bit;
                        }
                    }
                }
                continue;
            }
            match c.kind {
                Kind::Flip => {
                    for _ in 0..k {
                        let idx = uniform_below(rng, range);
                        let shot = idx & shot_mask;
                        let grp = (idx >> shot_bits) as usize;
                        let o = &mut out[(shot >> 6) as usize * stride..];
                        self.xor_col(c.firsts[grp] as usize, o, 1u64 << (shot & 63));
                    }
                }
                Kind::Depol1 | Kind::Depol2 => {
                    let table: &[u8] = if c.kind == Kind::Depol1 {
                        &DEPOL1_MASK
                    } else {
                        &DEPOL2_MASK
                    };
                    for _ in 0..k {
                        let idx = uniform_below(rng, range);
                        let shot = idx & shot_mask;
                        let t = idx >> shot_bits;
                        let grp = (t / m) as usize;
                        let mut mask = table[(t % m) as usize];
                        let o = &mut out[(shot >> 6) as usize * stride..];
                        let bit = 1u64 << (shot & 63);
                        let first = c.firsts[grp] as usize;
                        while mask != 0 {
                            self.xor_col(first + mask.trailing_zeros() as usize, o, bit);
                            mask &= mask - 1;
                        }
                    }
                }
            }
        }
        for dg in &self.dense {
            for w in 0..words {
                let o = &mut out[w * stride..];
                match dg.dist {
                    VarDist::Coin => {
                        let x = rng.next_u64();
                        self.xor_col(dg.first as usize, o, x);
                    }
                    VarDist::Flip(p) => {
                        let x = bernoulli_word(rng, p);
                        self.xor_col(dg.first as usize, o, x);
                    }
                    VarDist::Depol1(p) | VarDist::Depol2(p) => {
                        let (m, table): (u64, &[u8]) = match dg.dist {
                            VarDist::Depol1(_) => (3, &DEPOL1_MASK),
                            _ => (15, &DEPOL2_MASK),
                        };
                        let mut x = [0u64; 4];
                        let mut hits = bernoulli_word(rng, p);
                        while hits != 0 {
                            let s = hits.trailing_zeros();
                            let mask = table[uniform_below(rng, m) as usize];
                            for (k, xk) in x.iter_mut().enumerate() {
                                *xk |= (((mask >> k) & 1) as u64) << s;
                            }
                            hits &= hits - 1;
                        }
                        for (k, &xk) in x.iter().enumerate().take(dg.dist.len()) {
                            self.xor_col(dg.first as usize + k, o, xk);
                        }
                    }
                }
            }
        }
    }
}

/// 64 independent Bernoulli(`p`) bits, one comparison of a uniform 64-bit
/// word per bit (exact up to the 2^-64 grid).
fn bernoulli_word<R: RngCore + ?Sized>(rng: &mut R, p: f64) -> u64 {
    if p >= 1.0 {
        return !0;
    }
    let t = (p * 18446744073709551616.0) as u64; // p * 2^64, p < 1
    let mut w = 0u64;
    for s in 0..64 {
        w |= ((rng.next_u64() < t) as u64) << s;
    }
    w
}

/// Uniform integer in `[0, n)`, `n >= 1`: Lemire's multiply-shift with the
/// exact rejection step (no bias).
#[inline(always)]
pub fn uniform_below<R: RngCore + ?Sized>(rng: &mut R, n: u64) -> u64 {
    let mut m = (rng.next_u64() as u128) * (n as u128);
    if (m as u64) < n {
        let t = n.wrapping_neg() % n;
        while (m as u64) < t {
            m = (rng.next_u64() as u128) * (n as u128);
        }
    }
    (m >> 64) as u64
}

/// Uniform `f64` in `[0, 1)` with 53 random bits.
#[inline(always)]
fn unit_f64<R: RngCore + ?Sized>(rng: &mut R) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
}

/// `ln Gamma(x)` for `x >= 1` (Stirling series with upward shift; as in
/// NumPy's `random_loggam`).
fn loggam(x: f64) -> f64 {
    const A: [f64; 10] = [
        8.333333333333333e-02,
        -2.777777777777778e-03,
        7.936507936507937e-04,
        -5.952380952380952e-04,
        8.417508417508418e-04,
        -1.917526917526918e-03,
        6.41025641025641e-03,
        -2.955065359477124e-02,
        1.796443723688307e-01,
        -1.39243221690590e+00,
    ];
    if x == 1.0 || x == 2.0 {
        return 0.0;
    }
    let n = if x < 7.0 { (7.0 - x) as i64 } else { 0 };
    let mut x0 = x + n as f64;
    let x2 = (1.0 / x0) * (1.0 / x0);
    let mut gl0 = A[9];
    for k in (0..9).rev() {
        gl0 = gl0 * x2 + A[k];
    }
    let mut gl = gl0 / x0 + 0.5 * 1.8378770664093453 + (x0 - 0.5) * x0.ln() - x0;
    for _ in 0..n {
        gl -= (x0 - 1.0).ln();
        x0 -= 1.0;
    }
    gl
}

/// Poisson(`lam`) variate: multiplication method for `lam < 10`, Hörmann's
/// PTRS transformed rejection (as in NumPy) above.
pub fn poisson<R: RngCore + ?Sized>(rng: &mut R, lam: f64) -> u64 {
    if lam <= 0.0 {
        return 0;
    }
    if lam < 10.0 {
        let enlam = (-lam).exp();
        let mut x = 0u64;
        let mut prod = 1.0;
        loop {
            prod *= unit_f64(rng);
            if prod > enlam {
                x += 1;
            } else {
                return x;
            }
        }
    }
    let slam = lam.sqrt();
    let loglam = lam.ln();
    let b = 0.931 + 2.53 * slam;
    let a = -0.059 + 0.02483 * b;
    let invalpha = 1.1239 + 1.1328 / (b - 3.4);
    let vr = 0.9277 - 3.6224 / (b - 2.0);
    loop {
        let u = unit_f64(rng) - 0.5;
        let v = unit_f64(rng);
        let us = 0.5 - u.abs();
        let k = ((2.0 * a / us + b) * u + lam + 0.43).floor();
        if us >= 0.07 && v <= vr {
            return k as u64;
        }
        if k < 0.0 || (us < 0.013 && v > us) {
            continue;
        }
        if v.ln() + invalpha.ln() - (a / (us * us) + b).ln() <= -lam + k * loglam - loggam(k + 1.0)
        {
            return k as u64;
        }
    }
}

/// wyrand (Wang Yi, the PRNG of wyhash): one 64x64->128 multiply per
/// output, 64-bit state (period 2^64). Offered next to `rand`'s
/// Xoshiro256++ (`SmallRng`) for the PRNG comparison; Xoshiro256++ is the
/// default in the tools because of its larger state.
#[derive(Clone, Debug)]
pub struct WyRand(pub u64);

impl RngCore for WyRand {
    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0xa076_1d64_78bd_642f);
        let t = (self.0 as u128) * ((self.0 ^ 0xe703_7ed1_a0b4_28db) as u128);
        ((t >> 64) as u64) ^ (t as u64)
    }
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
    fn fill_bytes(&mut self, dst: &mut [u8]) {
        for c in dst.chunks_mut(8) {
            let b = self.next_u64().to_le_bytes();
            c.copy_from_slice(&b[..c.len()]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The inverse-CDF table reproduces the Poisson pmf to f64 accuracy, and
    /// the guide table never changes the result of the plain scan.
    #[test]
    fn poisson_table_matches_pmf_and_scan() {
        for mu in [1e-3, 0.5, 3.0, 8.0, 12.5, 60.0] {
            let t = PoissonTable::new(mu);
            assert_eq!(*t.thresholds.last().unwrap(), u64::MAX);
            let mut prev = 0u64;
            let mut lpk = -mu;
            let mut below = 0.0;
            for (k, &th) in t.thresholds.iter().enumerate() {
                if k > 0 {
                    lpk += mu.ln() - (k as f64).ln();
                }
                let got = (th - prev) as f64 / 18446744073709551616.0;
                let want = lpk.exp();
                if k + 1 < t.thresholds.len() {
                    assert!(
                        (got - want).abs() <= 1e-15 + 1e-12 * want,
                        "mu={mu} k={k}: {got} vs {want}"
                    );
                } else {
                    // the last entry carries the whole remaining tail
                    assert!(
                        (got - (1.0 - below)).abs() < 2e-15,
                        "tail {got} vs {}",
                        1.0 - below
                    );
                }
                below += want;
                prev = th;
            }
            let scan = |u: u64| {
                t.thresholds
                    .iter()
                    .take_while(|&&x| x <= u)
                    .count()
                    .min(t.thresholds.len() - 1)
            };
            let mut rng = WyRand(5);
            for i in 0..200_000u64 {
                let u = match i {
                    0 => 0,
                    1 => u64::MAX,
                    2..=400 => t.thresholds[(i as usize) % t.thresholds.len()].wrapping_sub(i & 1),
                    _ => rng.next_u64(),
                };
                assert_eq!(t.sample(u), scan(u), "mu={mu} u={u}");
            }
        }
    }
}
