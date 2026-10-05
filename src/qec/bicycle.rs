//! Two-block group-algebra (2BGA) codes over abelian groups `Z_l x Z_m`:
//! bivariate-bicycle (BB) codes (Bravyi et al., Nature 627, 778 (2024)),
//! generalised-bicycle (GB) codes (`m = 1`, Kovalev & Pryadko; Panteleev &
//! Kalachev) and coprime-BB codes (`gcd(l, m) = 1`, Wang & Mueller), with
//! exact `[[n, k, d]]`.
//!
//! # Convention
//!
//! Group elements `(i, j)` of `Z_l x Z_m` are indexed `g = i m + j`; the
//! monomial `x^a y^b` is the element `(a, b)`. With `A`, `B` sums of
//! monomials (sets of group elements), the code has `n = 2 l m` qubits, a
//! left block `L` and a right block `R`, each indexed by the group, and
//!
//! ```text
//! H_X = [A | B]        X-check g acts on L{g + a : a in A} and R{g + b : b in B}
//! H_Z = [B^T | A^T]    Z-check h acts on L{h - b : b in B} and R{h - a : a in A}
//! ```
//!
//! which is the matrix convention of Bravyi et al. (`x = S_l (x) I_m`,
//! `y = I_l (x) S_m`, `S` the cyclic shift with `S[i][i+1] = 1`).
//!
//! # Distance
//!
//! [`min_weight_logical`] is an exact branch-and-bound over *connected*
//! error sets (the Dumer–Kovalev–Pryadko cluster idea, implemented like
//! `qec::distance`): a minimum-weight nontrivial logical has no proper
//! non-empty subset in `ker H`, so its support is reached from any of its
//! qubits by repeatedly picking an unsatisfied check and adding one of its
//! qubits. The group acts transitively on each block by translations, which
//! are code automorphisms, so it suffices to root the search at `L0` and at
//! `R0`, and the `R0` search may exclude every `L` qubit (a logical touching
//! `L` has a translate through `L0`, already explored at the same weight).
//! Nontriviality is tested with `k` conjugate logical vectors packed into a
//! `u128` mask per qubit. [`distance_upper_bound`] is a fast randomized
//! information-set search (Leon / Lee–Brickell, `p <= 2`) giving an upper
//! bound and a witness.
#![allow(clippy::needless_range_loop)]

use rand::seq::SliceRandom;
use rand::Rng;

/// A dense GF(2) matrix, row-major, rows packed into `u64` words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gf2Mat {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// `u64` words per row (`ceil(cols / 64)`, at least 1).
    pub words: usize,
    /// Row-major bits: row `r` occupies `data[r * words..(r + 1) * words]`,
    /// column `c` is bit `c % 64` of word `c / 64`.
    pub data: Vec<u64>,
}

impl Gf2Mat {
    /// All-zero `rows × cols` matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let words = cols.div_ceil(64).max(1);
        Gf2Mat {
            rows,
            cols,
            words,
            data: vec![0; rows * words],
        }
    }

    /// Entry `(r, c)`.
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> bool {
        self.data[r * self.words + c / 64] >> (c % 64) & 1 == 1
    }

    /// Toggles entry `(r, c)`.
    #[inline]
    pub fn flip(&mut self, r: usize, c: usize) {
        self.data[r * self.words + c / 64] ^= 1u64 << (c % 64);
    }

    /// Packed words of row `r`.
    pub fn row(&self, r: usize) -> &[u64] {
        &self.data[r * self.words..(r + 1) * self.words]
    }

    /// Support (column indices) of row `r`.
    pub fn row_support(&self, r: usize) -> Vec<usize> {
        bits_of(self.row(r))
    }

    /// Rank over GF(2) (does not modify `self`).
    pub fn rank(&self) -> usize {
        let mut m = self.clone();
        m.rref(None).len()
    }

    /// In-place reduced row echelon form. Columns are visited in `order`
    /// (default `0..cols`). Returns the pivot columns, in order; pivot `t`
    /// sits in row `t`.
    pub fn rref(&mut self, order: Option<&[usize]>) -> Vec<usize> {
        let w = self.words;
        let mut pivots = Vec::new();
        let mut r = 0usize;
        let default: Vec<usize>;
        let order = match order {
            Some(o) => o,
            None => {
                default = (0..self.cols).collect();
                &default
            }
        };
        for &c in order {
            if r == self.rows {
                break;
            }
            let (wc, bc) = (c / 64, 1u64 << (c % 64));
            let Some(p) = (r..self.rows).find(|&i| self.data[i * w + wc] & bc != 0) else {
                continue;
            };
            if p != r {
                for t in 0..w {
                    self.data.swap(p * w + t, r * w + t);
                }
            }
            for i in 0..self.rows {
                if i != r && self.data[i * w + wc] & bc != 0 {
                    for t in 0..w {
                        let v = self.data[r * w + t];
                        self.data[i * w + t] ^= v;
                    }
                }
            }
            pivots.push(c);
            r += 1;
        }
        pivots
    }

    /// A basis of the right kernel `{v : M v = 0}`, as packed vectors.
    pub fn kernel(&self) -> Vec<Vec<u64>> {
        let mut m = self.clone();
        let piv = m.rref(None);
        let mut is_piv = vec![false; self.cols];
        for &c in &piv {
            is_piv[c] = true;
        }
        let w = self.cols.div_ceil(64).max(1);
        let mut out = Vec::new();
        for f in (0..self.cols).filter(|&c| !is_piv[c]) {
            let mut v = vec![0u64; w];
            v[f / 64] |= 1 << (f % 64);
            for (t, &c) in piv.iter().enumerate() {
                if m.get(t, f) {
                    v[c / 64] |= 1 << (c % 64);
                }
            }
            out.push(v);
        }
        out
    }
}

/// Indices of set bits.
pub fn bits_of(v: &[u64]) -> Vec<usize> {
    let mut out = Vec::new();
    for (w, &word) in v.iter().enumerate() {
        let mut x = word;
        while x != 0 {
            out.push(w * 64 + x.trailing_zeros() as usize);
            x &= x - 1;
        }
    }
    out
}

fn dot(a: &[u64], b: &[u64]) -> bool {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x & y).count_ones())
        .sum::<u32>()
        & 1
        == 1
}

/// A two-block code over `Z_l x Z_m` given by the monomial sets `A`, `B`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TwoBlockCode {
    /// Order of the first cyclic factor (`x^l = 1`).
    pub l: usize,
    /// Order of the second cyclic factor (`y^m = 1`; 1 for GB codes).
    pub m: usize,
    /// Monomials `x^a y^b` of `A` as `(a, b)`.
    pub a: Vec<(usize, usize)>,
    /// Monomials of `B`.
    pub b: Vec<(usize, usize)>,
}

impl TwoBlockCode {
    /// Code from monomial lists; exponents are reduced mod `l` and `m` and term
    /// order is preserved (syndrome schedules index terms by position). Panics on
    /// a repeated monomial.
    pub fn new(l: usize, m: usize, a: &[(usize, usize)], b: &[(usize, usize)]) -> Self {
        // term order is kept (syndrome-circuit schedules refer to it)
        let norm = |v: &[(usize, usize)]| -> Vec<(usize, usize)> {
            v.iter().map(|&(i, j)| (i % l, j % m)).collect()
        };
        let (a, b) = (norm(a), norm(b));
        for v in [&a, &b] {
            let mut d = v.clone();
            d.sort_unstable();
            d.dedup();
            assert_eq!(d.len(), v.len(), "repeated monomial");
        }
        TwoBlockCode { l, m, a, b }
    }

    /// Parses `"x^3 + y + y^2"` style polynomials (terms `1`, `x`, `y`,
    /// `x^a`, `y^b`, `x^a y^b`, `x^a*y^b`; for GB codes with `m = 1` use `x`).
    pub fn parse(l: usize, m: usize, a: &str, b: &str) -> Self {
        Self::new(l, m, &parse_poly(a), &parse_poly(b))
    }

    /// Group order `|G| = l m`.
    pub fn order(&self) -> usize {
        self.l * self.m
    }

    /// Number of physical qubits, `2 l m`.
    pub fn n(&self) -> usize {
        2 * self.order()
    }

    fn idx(&self, i: usize, j: usize) -> usize {
        (i % self.l) * self.m + (j % self.m)
    }

    /// `g + (a, b)` (`sign = false`) or `g - (a, b)` (`sign = true`).
    fn shift(&self, g: usize, (a, b): (usize, usize), minus: bool) -> usize {
        let (i, j) = (g / self.m, g % self.m);
        if minus {
            self.idx(i + self.l - a, j + self.m - b)
        } else {
            self.idx(i + a, j + b)
        }
    }

    /// `H_X = [A | B]`.
    pub fn hx(&self) -> Gf2Mat {
        let nn = self.order();
        let mut h = Gf2Mat::zeros(nn, 2 * nn);
        for g in 0..nn {
            for &t in &self.a {
                h.flip(g, self.shift(g, t, false));
            }
            for &t in &self.b {
                h.flip(g, nn + self.shift(g, t, false));
            }
        }
        h
    }

    /// `H_Z = [B^T | A^T]`.
    pub fn hz(&self) -> Gf2Mat {
        let nn = self.order();
        let mut h = Gf2Mat::zeros(nn, 2 * nn);
        for g in 0..nn {
            for &t in &self.b {
                h.flip(g, self.shift(g, t, true));
            }
            for &t in &self.a {
                h.flip(g, nn + self.shift(g, t, true));
            }
        }
        h
    }

    /// Number of logical qubits `n - rank H_X - rank H_Z`.
    pub fn k(&self) -> usize {
        self.n() - self.hx().rank() - self.hz().rank()
    }

    /// Exact `d_Z` (minimum weight of a Z-type logical: `H_X e = 0`,
    /// `e` not in the row space of `H_Z`)... i.e. the distance against the
    /// pair `(hcheck = H_X, hother = H_Z)`; see [`code_distance`].
    pub fn distance(&self, opts: &DistanceOpts) -> DistanceResult {
        code_distance(&self.hx(), &self.hz(), Some(self.order()), opts)
    }

    /// Both CSS distances `(d against H_X, d against H_Z)`.
    pub fn distances_both(&self, opts: &DistanceOpts) -> (DistanceResult, DistanceResult) {
        let (hx, hz) = (self.hx(), self.hz());
        (
            code_distance(&hx, &hz, Some(self.order()), opts),
            code_distance(&hz, &hx, Some(self.order()), opts),
        )
    }

    /// Human-readable polynomial strings `(A, B)`.
    pub fn poly_strings(&self) -> (String, String) {
        (fmt_poly(&self.a, self.m), fmt_poly(&self.b, self.m))
    }
}

/// Formats a monomial set as `x^a y^b + ...` (`m = 1`: univariate).
pub fn fmt_poly(t: &[(usize, usize)], m: usize) -> String {
    let mono = |&(a, b): &(usize, usize)| {
        let mut s = String::new();
        if a > 0 {
            s += "x";
            if a > 1 {
                s += &format!("^{a}");
            }
        }
        if b > 0 && m > 1 {
            if !s.is_empty() {
                s += " ";
            }
            s += "y";
            if b > 1 {
                s += &format!("^{b}");
            }
        }
        if s.is_empty() {
            s = "1".into();
        }
        s
    };
    t.iter().map(mono).collect::<Vec<_>>().join(" + ")
}

/// Parses a sum of monomials in `x`, `y`.
pub fn parse_poly(s: &str) -> Vec<(usize, usize)> {
    s.split('+')
        .map(|term| {
            let term = term.replace(['*', ' '], "");
            let (mut a, mut b) = (0usize, 0usize);
            let bytes: Vec<char> = term.chars().collect();
            let mut i = 0;
            if term == "1" {
                return (0, 0);
            }
            while i < bytes.len() {
                let v = bytes[i];
                assert!(v == 'x' || v == 'y', "bad term {term}");
                i += 1;
                let mut e = 1usize;
                if i < bytes.len() && bytes[i] == '^' {
                    i += 1;
                    let st = i;
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                    e = bytes[st..i].iter().collect::<String>().parse().unwrap();
                }
                if v == 'x' {
                    a += e;
                } else {
                    b += e;
                }
            }
            (a, b)
        })
        .collect()
}

/// Options for [`code_distance`].
#[derive(Clone, Debug)]
pub struct DistanceOpts {
    /// Search weights `1..=max_weight` exactly (also capped by the upper
    /// bound found by the randomized search).
    pub max_weight: usize,
    /// Abort the exact search after this many nodes (per weight level sum).
    pub node_limit: u64,
    /// Iterations of the randomized information-set search.
    pub ub_iters: usize,
    /// Seed of the randomized upper-bound search.
    pub seed: u64,
}

impl Default for DistanceOpts {
    fn default() -> Self {
        DistanceOpts {
            max_weight: 64,
            node_limit: 2_000_000_000,
            ub_iters: 200,
            seed: 1,
        }
    }
}

/// Result of [`code_distance`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DistanceResult {
    /// Number of logical qubits.
    pub k: usize,
    /// Proven lower bound (`d >= lower`).
    pub lower: usize,
    /// Upper bound with a witness (`d <= upper`), `usize::MAX` if none.
    pub upper: usize,
    /// A logical of weight `upper` (qubit indices).
    pub witness: Vec<usize>,
    /// Exact-search nodes used.
    pub nodes: u64,
}

impl DistanceResult {
    /// The distance, if the lower and upper bounds coincide.
    pub fn exact(&self) -> Option<usize> {
        (self.lower == self.upper).then_some(self.upper)
    }
}

/// A basis of `ker(hcheck)` modulo `rowspace(hother)` (`k` vectors): for
/// `(hcheck, hother) = (H_X, H_Z)` these are Z-type logical operators.
pub fn logical_basis(hcheck: &Gf2Mat, hother: &Gf2Mat) -> Vec<Vec<u64>> {
    let w = hcheck.words;
    let mut basis = hother.clone();
    let r0 = basis.rref(None).len();
    let mut red: Vec<(usize, Vec<u64>)> = Vec::new();
    let reduce = |v: &mut Vec<u64>, red: &[(usize, Vec<u64>)]| {
        for (c, r) in red {
            if v[c / 64] >> (c % 64) & 1 == 1 {
                for t in 0..w {
                    v[t] ^= r[t];
                }
            }
        }
    };
    for r in 0..r0 {
        let mut v = basis.row(r).to_vec();
        reduce(&mut v, &red);
        if let Some(&c) = bits_of(&v).first() {
            red.push((c, v));
        }
    }
    let mut out = Vec::new();
    for v0 in hcheck.kernel() {
        let mut v = v0.clone();
        reduce(&mut v, &red);
        if let Some(&c) = bits_of(&v).first() {
            red.push((c, v));
            out.push(v0);
        }
    }
    out
}

/// For the pair `(hcheck, hother)` of a CSS code: per qubit, a `u128` mask of
/// which of `k` conjugate logicals it overlaps. A vector `e` in `ker hcheck`
/// is a nontrivial logical iff the XOR of the masks over its support is
/// non-zero. Returns `(masks, k)`; panics if `k > 64`.
pub fn logical_masks(hcheck: &Gf2Mat, hother: &Gf2Mat) -> (Vec<u128>, usize) {
    let n = hcheck.cols;
    let conj = logical_basis(hother, hcheck);
    let k = conj.len();
    assert!(k <= 128, "k = {k} > 128 unsupported");
    let mut masks = vec![0u128; n];
    for (j, l) in conj.iter().enumerate() {
        for q in bits_of(l) {
            masks[q] |= 1u128 << j;
        }
    }
    (masks, k)
}

/// Randomized information-set upper bound on the minimum weight of a
/// nontrivial logical in `ker hcheck`. Returns `(weight, support)`.
pub fn distance_upper_bound<R: Rng>(
    hcheck: &Gf2Mat,
    masks: &[u128],
    iters: usize,
    rng: &mut R,
) -> (usize, Vec<usize>) {
    let n = hcheck.cols;
    let ker = hcheck.kernel();
    let kd = ker.len();
    let mut best = (usize::MAX, Vec::new());
    if kd == 0 {
        return best;
    }
    let w = n.div_ceil(64).max(1);
    let mut g = Gf2Mat::zeros(kd, n);
    for (r, v) in ker.iter().enumerate() {
        g.data[r * w..(r + 1) * w].copy_from_slice(v);
    }
    let obs_of = |v: &[u64]| bits_of(v).iter().fold(0u128, |o, &q| o ^ masks[q]);
    let mut order: Vec<usize> = (0..n).collect();
    let mut buf = vec![0u64; w];
    for _ in 0..iters {
        order.shuffle(rng);
        let mut m = g.clone();
        let piv = m.rref(Some(&order));
        let rr = piv.len();
        let rows: Vec<&[u64]> = (0..rr).map(|r| m.row(r)).collect();
        let obs: Vec<u128> = rows.iter().map(|r| obs_of(r)).collect();
        let wt: Vec<u32> = rows
            .iter()
            .map(|r| r.iter().map(|x| x.count_ones()).sum())
            .collect();
        for i in 0..rr {
            if obs[i] != 0 && (wt[i] as usize) < best.0 {
                best = (wt[i] as usize, bits_of(rows[i]));
            }
            for j in i + 1..rr {
                if obs[i] ^ obs[j] == 0 {
                    continue;
                }
                let mut c = 0usize;
                for t in 0..w {
                    buf[t] = rows[i][t] ^ rows[j][t];
                    c += buf[t].count_ones() as usize;
                }
                if c < best.0 {
                    best = (c, bits_of(&buf));
                }
            }
        }
    }
    best
}

/// Outcome of [`min_weight_logical`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SearchOutcome {
    /// A nontrivial logical of exactly this weight exists (support given)
    /// and none lighter exists.
    Found(usize, Vec<usize>),
    /// No nontrivial logical of weight `<= max_weight`.
    NoneUpTo(usize),
    /// Node limit hit: no logical of weight `<= proven` exists.
    Aborted {
        /// Largest weight proven to contain no nontrivial logical.
        proven: usize,
    },
}

struct Bb<'a> {
    qchecks: &'a [Vec<u32>],
    cqubits: &'a [Vec<u32>],
    masks: &'a [u128],
    nbr: Vec<Vec<u64>>,
    maxdeg: usize,
    cw: usize,
    banned: Vec<bool>,
    used: Vec<bool>,
    chosen: Vec<usize>,
    nodes: u64,
    limit: u64,
    found: Option<Vec<usize>>,
}

impl Bb<'_> {
    fn toggle(&self, f: &mut [u64], q: usize) {
        for &c in &self.qchecks[q] {
            f[c as usize / 64] ^= 1u64 << (c % 64);
        }
    }

    /// Returns false on abort; sets `found` on success.
    fn dfs(&mut self, f: &mut [u64], obs: u128, w: usize) -> bool {
        self.nodes += 1;
        if self.nodes > self.limit {
            return false;
        }
        let k = self.chosen.len();
        let nf: usize = f.iter().map(|x| x.count_ones() as usize).sum();
        if nf == 0 {
            if obs != 0 {
                self.found = Some(self.chosen.clone());
            }
            return true;
        }
        if k + nf.div_ceil(self.maxdeg) > w {
            return true;
        }
        if k + 1 < w {
            // greedy set of fired checks pairwise sharing no qubit
            let mut blocked = vec![0u64; self.cw];
            let mut lb = 0usize;
            for wi in 0..self.cw {
                let mut x = f[wi] & !blocked[wi];
                while x != 0 {
                    let i = wi * 64 + x.trailing_zeros() as usize;
                    lb += 1;
                    for (b, n) in blocked.iter_mut().zip(&self.nbr[i]) {
                        *b |= n;
                    }
                    x &= !blocked[wi];
                }
            }
            if k + lb > w {
                return true;
            }
        }
        let (mut best_c, mut best_n) = (usize::MAX, usize::MAX);
        'outer: for wi in 0..self.cw {
            let mut x = f[wi];
            while x != 0 {
                let c = wi * 64 + x.trailing_zeros() as usize;
                x &= x - 1;
                let n = self.cqubits[c]
                    .iter()
                    .filter(|&&q| !self.banned[q as usize] && !self.used[q as usize])
                    .count();
                if n < best_n {
                    best_n = n;
                    best_c = c;
                    if n <= 1 {
                        break 'outer;
                    }
                }
            }
        }
        if best_n == 0 {
            return true;
        }
        let cands: Vec<u32> = self.cqubits[best_c]
            .iter()
            .copied()
            .filter(|&q| !self.banned[q as usize] && !self.used[q as usize])
            .collect();
        let mut newly = Vec::with_capacity(cands.len());
        let mut ok = true;
        for &q in &cands {
            let q = q as usize;
            self.used[q] = true;
            self.chosen.push(q);
            self.toggle(f, q);
            let cont = self.dfs(f, obs ^ self.masks[q], w);
            self.toggle(f, q);
            self.chosen.pop();
            self.used[q] = false;
            self.banned[q] = true;
            newly.push(q);
            if !cont || self.found.is_some() {
                ok = cont;
                break;
            }
        }
        for q in newly {
            self.banned[q] = false;
        }
        ok
    }
}

/// Exact minimum weight of a nontrivial logical in `ker hcheck` (masks from
/// [`logical_masks`]), searching weights `start..=max_weight`.
///
/// `roots`: list of `(root qubit, qubits to ban for this root)`. Every
/// minimum-weight logical must contain some root after applying a code
/// automorphism that maps the banned qubits of that root to qubits covered by
/// earlier roots; with `roots = [(q, {0..q})]` for all `q` this is the plain
/// exhaustive search.
pub fn min_weight_logical(
    hcheck: &Gf2Mat,
    masks: &[u128],
    roots: &[(usize, Vec<usize>)],
    start: usize,
    max_weight: usize,
    node_limit: u64,
) -> (SearchOutcome, u64) {
    let n = hcheck.cols;
    let nc = hcheck.rows;
    let mut qchecks: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut cqubits: Vec<Vec<u32>> = vec![Vec::new(); nc];
    for c in 0..nc {
        for q in hcheck.row_support(c) {
            qchecks[q].push(c as u32);
            cqubits[c].push(q as u32);
        }
    }
    let cw = nc.div_ceil(64).max(1);
    let mut nbr = vec![vec![0u64; cw]; nc];
    for qc in &qchecks {
        for &a in qc {
            for &b in qc {
                nbr[a as usize][b as usize / 64] |= 1u64 << (b % 64);
            }
        }
    }
    let maxdeg = qchecks.iter().map(|v| v.len()).max().unwrap_or(1).max(1);
    let mut s = Bb {
        qchecks: &qchecks,
        cqubits: &cqubits,
        masks,
        nbr,
        maxdeg,
        cw,
        banned: vec![false; n],
        used: vec![false; n],
        chosen: Vec::new(),
        nodes: 0,
        limit: node_limit,
        found: None,
    };
    for w in start.max(1)..=max_weight {
        for (root, ban) in roots {
            for b in s.banned.iter_mut() {
                *b = false;
            }
            for &q in ban {
                s.banned[q] = true;
            }
            let mut f = vec![0u64; cw];
            s.used[*root] = true;
            s.chosen.push(*root);
            s.toggle(&mut f, *root);
            let ok = s.dfs(&mut f, masks[*root], w);
            s.chosen.pop();
            s.used[*root] = false;
            if !ok {
                return (SearchOutcome::Aborted { proven: w - 1 }, s.nodes);
            }
            if let Some(found) = s.found.take() {
                return (SearchOutcome::Found(w, found), s.nodes);
            }
        }
    }
    (SearchOutcome::NoneUpTo(max_weight), s.nodes)
}

/// Roots for a two-block code with translation symmetry on blocks of size
/// `half`: `L0` (no ban), then `R0` with all of `L` banned.
pub fn two_block_roots(half: usize) -> Vec<(usize, Vec<usize>)> {
    vec![(0, Vec::new()), (half, (0..half).collect())]
}

/// Plain roots (no symmetry): every qubit, banning the earlier ones.
pub fn all_roots(n: usize) -> Vec<(usize, Vec<usize>)> {
    (0..n).map(|q| (q, (0..q).collect())).collect()
}

/// Distance of the CSS pair `(hcheck, hother)`: minimum weight of `e` with
/// `hcheck e = 0`, `e` not in `rowspace(hother)`. `block` = `Some(|G|)` for
/// two-block codes with translation symmetry (columns `0..|G|` and
/// `|G|..2|G|` are the two regular orbits), `None` for no symmetry.
pub fn code_distance(
    hcheck: &Gf2Mat,
    hother: &Gf2Mat,
    block: Option<usize>,
    opts: &DistanceOpts,
) -> DistanceResult {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    let (masks, k) = logical_masks(hcheck, hother);
    if k == 0 {
        return DistanceResult {
            k,
            lower: usize::MAX,
            upper: usize::MAX,
            witness: Vec::new(),
            nodes: 0,
        };
    }
    let mut rng = StdRng::seed_from_u64(opts.seed);
    let (ub, wit) = distance_upper_bound(hcheck, &masks, opts.ub_iters, &mut rng);
    let roots = match block {
        Some(h) => two_block_roots(h),
        None => all_roots(hcheck.cols),
    };
    let top = opts.max_weight.min(ub.saturating_sub(1));
    let (out, nodes) = min_weight_logical(hcheck, &masks, &roots, 1, top, opts.node_limit);
    match out {
        SearchOutcome::Found(w, sup) => DistanceResult {
            k,
            lower: w,
            upper: w,
            witness: sup,
            nodes,
        },
        SearchOutcome::NoneUpTo(t) => DistanceResult {
            k,
            lower: if t + 1 == ub { ub } else { t + 1 },
            upper: ub,
            witness: wit,
            nodes,
        },
        SearchOutcome::Aborted { proven } => DistanceResult {
            k,
            lower: proven + 1,
            upper: ub,
            witness: wit,
            nodes,
        },
    }
}

/// Checks `e` (support) is a nontrivial logical of `(hcheck, masks)`.
pub fn is_nontrivial_logical(hcheck: &Gf2Mat, masks: &[u128], support: &[usize]) -> bool {
    let mut v = vec![0u64; hcheck.words];
    for &q in support {
        v[q / 64] ^= 1 << (q % 64);
    }
    let syn_zero = (0..hcheck.rows).all(|r| !dot(hcheck.row(r), &v));
    let obs = support.iter().fold(0u128, |o, &q| o ^ masks[q]);
    syn_zero && obs != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toric_code_as_bb() {
        // A = 1 + x, B = 1 + y on Z_L x Z_L is the toric code [[2L^2, 2, L]]
        for l in 2..=5 {
            let c = TwoBlockCode::new(l, l, &[(0, 0), (1, 0)], &[(0, 0), (0, 1)]);
            assert_eq!(c.k(), 2);
            let d = c.distance(&DistanceOpts::default());
            assert_eq!(d.exact(), Some(l), "L={l}");
        }
    }

    #[test]
    fn symmetric_roots_match_plain_search() {
        // random small BB codes: symmetric-root search == plain search
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(7);
        let mut tested = 0;
        while tested < 25 {
            let l = rng.random_range(2..6usize);
            let m = rng.random_range(1..5usize);
            let nn = l * m;
            if nn < 4 {
                continue;
            }
            let pick = |rng: &mut StdRng| {
                let mut v: Vec<usize> = (1..nn).collect();
                v.shuffle(rng);
                let mut t = vec![(0usize, 0usize)];
                t.extend(v[..2.min(nn - 1)].iter().map(|&g| (g / m, g % m)));
                t
            };
            let (a, b) = (pick(&mut rng), pick(&mut rng));
            let c = TwoBlockCode::new(l, m, &a, &b);
            let (hx, hz) = (c.hx(), c.hz());
            let (masks, k) = logical_masks(&hx, &hz);
            if k == 0 {
                continue;
            }
            tested += 1;
            let (o1, _) =
                min_weight_logical(&hx, &masks, &two_block_roots(nn), 1, 2 * nn, u64::MAX);
            let (o2, _) = min_weight_logical(&hx, &masks, &all_roots(2 * nn), 1, 2 * nn, u64::MAX);
            let w = |o: &SearchOutcome| match o {
                SearchOutcome::Found(w, s) => {
                    assert!(is_nontrivial_logical(&hx, &masks, s));
                    *w
                }
                _ => panic!("{o:?}"),
            };
            assert_eq!(w(&o1), w(&o2), "{c:?}");
            // brute force for tiny n
            if 2 * nn <= 16 {
                let mut best = usize::MAX;
                for e in 1u32..(1 << (2 * nn)) {
                    let sup: Vec<usize> = (0..2 * nn).filter(|&q| e >> q & 1 == 1).collect();
                    if is_nontrivial_logical(&hx, &masks, &sup) {
                        best = best.min(sup.len());
                    }
                }
                assert_eq!(best, w(&o1), "{c:?}");
            }
        }
    }

    #[test]
    fn parse_and_format() {
        assert_eq!(parse_poly("x^3 + y + y^2"), vec![(3, 0), (0, 1), (0, 2)]);
        assert_eq!(
            parse_poly("1 + x y^2 + x^2*y"),
            vec![(0, 0), (1, 2), (2, 1)]
        );
        let c = TwoBlockCode::parse(12, 6, "x^3 + y + y^2", "y^3 + x + x^2");
        assert_eq!(c.poly_strings().0, "x^3 + y + y^2");
    }
}
