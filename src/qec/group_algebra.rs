//! Two-block group-algebra (2BGA) codes over arbitrary finite groups,
//! including non-abelian ones (Lin & Pryadko, Phys. Rev. A 109, 022407
//! (2024)), and their coset generalisation ([`CosetCode`], Aydin, Tamo &
//! Barg, arXiv:2606.17268), with exact `[[n, k, d]]` and an enumeration of
//! every code of given weights up to equivalence. The abelian rank <= 2
//! special case (BB / GB codes) lives in `qec::bicycle`; the study, which
//! found the weight-6 codes `[[288,16,16]]` and `[[192,12,14]]` with this
//! module, is `research/qec/code-discovery-2.md`.
//!
//! # Convention
//!
//! A group `G` of order `N` is given by its multiplication table, elements
//! `0..N` with identity `0`. For subsets `A`, `B` of `G` the code has
//! `n = 2N` qubits in two blocks `L`, `R` (each indexed by `G`) and
//!
//! ```text
//! X-check g:  L{g a : a in A}       R{b g : b in B}
//! Z-check h:  L{b^-1 h : b in B}    R{h a^-1 : a in A}
//! ```
//!
//! `A` acts by right and `B` by left multiplication, so the two act on
//! different sides and every X-check meets every Z-check an even number of
//! times (both overlaps count the solutions of `h = b g a`). For abelian
//! `G = Z_l x Z_m` with element `i m + j` this is exactly the convention of
//! `qec::bicycle` (`H_X = [A | B]`, `H_Z = [B^T | A^T]`). Writing elements
//! as their inverses turns it into the left-`A` / right-`B` convention of
//! Lin & Pryadko.
//!
//! # Equivalences
//!
//! Each of these maps sends a code to one with the same `[[n, k, d]]`
//! (a relabelling of qubits and checks; the last two exchange the X and Z
//! sectors):
//!
//! * two-sided translations of `A` and of `B`, independently:
//!   `A -> u A w`, `B -> v B t` (in particular `A` and `B` may be
//!   conjugated independently);
//! * a group automorphism applied to both `A` and `B`;
//! * `(A, B) -> (B^-1, A^-1)` (sectors kept), `(A, B) -> (B, A)` and
//!   `(A, B) -> (A^-1, B^-1)` (sectors exchanged).
//!
//! A *T-class* is a class of subsets under two-sided translation; its
//! identity-containing members are the sets `c(S s^-1)` with `c` an inner
//! automorphism and `s` in `S`. [`Enumeration`] lists T-classes and the
//! orbits of pairs of T-classes under the automorphisms, the swap and the
//! inversion: one code per equivalence class (up to automorphisms of `G`
//! outside the supplied generators).
//!
//! # Distance
//!
//! The exact search of `qec::bicycle` (connected clusters, branch and bound)
//! is rooted at one qubit per orbit of the code automorphisms
//! `L x -> t^-1 x w`, `R x -> v x u^-1` with `u A w = A`, `v B t = B`
//! (each verified against the check sets before use). For abelian groups
//! this is the full translation group (two roots); for non-abelian groups
//! the orbits are smaller and more roots are needed.
#![allow(clippy::needless_range_loop)]

use super::bicycle::{
    distance_upper_bound, logical_masks, min_weight_logical, DistanceOpts, DistanceResult, Gf2Mat,
    SearchOutcome,
};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// Errors from building or parsing a finite group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GroupError {
    /// The table is not a group (message says which axiom failed).
    NotAGroup(String),
    /// Automorphism generator `i` is not an automorphism.
    BadAutomorphism(usize),
    /// Malformed group export text.
    Parse(String),
}

impl fmt::Display for GroupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GroupError::NotAGroup(s) => write!(f, "not a group: {s}"),
            GroupError::BadAutomorphism(i) => write!(f, "automorphism generator {i} is invalid"),
            GroupError::Parse(s) => write!(f, "group export parse error: {s}"),
        }
    }
}

impl std::error::Error for GroupError {}

/// A finite group given by its multiplication table.
#[derive(Clone, Debug)]
pub struct FiniteGroup {
    /// Group order `N`.
    pub order: usize,
    /// Multiplication table, `mul[g * N + h] = g h`; element `0` is the identity.
    pub mul: Vec<u16>,
    /// Inverses.
    pub inv: Vec<u16>,
    /// Generators of a group of automorphisms (each a permutation of the
    /// elements). Enumeration is correct for any subgroup of `Aut(G)`; the
    /// full group only maximises deduplication.
    pub aut_gens: Vec<Vec<u16>>,
    /// Human-readable name, e.g. `SmallGroup(48,3) C4xS3` or `Z7:Z3`.
    pub label: String,
}

impl FiniteGroup {
    /// Builds and verifies a group from its table: identity `0`, a Latin
    /// square, associativity, and each automorphism generator a bijective
    /// homomorphism.
    pub fn from_table(
        order: usize,
        mul: Vec<u16>,
        aut_gens: Vec<Vec<u16>>,
        label: impl Into<String>,
    ) -> Result<Self, GroupError> {
        let n = order;
        if n == 0 || n > u16::MAX as usize || mul.len() != n * n {
            return Err(GroupError::NotAGroup("bad size".into()));
        }
        for g in 0..n {
            if mul[g] as usize != g || mul[g * n] as usize != g {
                return Err(GroupError::NotAGroup("0 is not the identity".into()));
            }
        }
        let mut inv = vec![u16::MAX; n];
        for g in 0..n {
            let mut seen = vec![false; n];
            for h in 0..n {
                let p = mul[g * n + h] as usize;
                if p >= n || seen[p] {
                    return Err(GroupError::NotAGroup(format!("row {g} not a permutation")));
                }
                seen[p] = true;
                if p == 0 {
                    inv[g] = h as u16;
                }
            }
        }
        for g in 0..n {
            for h in 0..n {
                let gh = mul[g * n + h] as usize;
                for k in 0..n {
                    let hk = mul[h * n + k] as usize;
                    if mul[gh * n + k] != mul[g * n + hk] {
                        return Err(GroupError::NotAGroup(format!(
                            "not associative at ({g},{h},{k})"
                        )));
                    }
                }
            }
        }
        let g = FiniteGroup {
            order: n,
            mul,
            inv,
            aut_gens,
            label: label.into(),
        };
        for (i, s) in g.aut_gens.iter().enumerate() {
            if !g.is_automorphism(s) {
                return Err(GroupError::BadAutomorphism(i));
            }
        }
        Ok(g)
    }

    /// `g h`.
    #[inline]
    pub fn mul(&self, g: usize, h: usize) -> usize {
        self.mul[g * self.order + h] as usize
    }

    /// `g^-1`.
    #[inline]
    pub fn inv(&self, g: usize) -> usize {
        self.inv[g] as usize
    }

    /// True if `s` is a bijective homomorphism `G -> G`.
    pub fn is_automorphism(&self, s: &[u16]) -> bool {
        let n = self.order;
        if s.len() != n {
            return false;
        }
        let mut seen = vec![false; n];
        for &x in s {
            if x as usize >= n || seen[x as usize] {
                return false;
            }
            seen[x as usize] = true;
        }
        (0..n).all(|g| {
            (0..n).all(|h| s[self.mul(g, h)] as usize == self.mul(s[g] as usize, s[h] as usize))
        })
    }

    /// True if the group is abelian.
    pub fn is_abelian(&self) -> bool {
        let n = self.order;
        (0..n).all(|g| (0..g).all(|h| self.mul(g, h) == self.mul(h, g)))
    }

    /// Order of the centre.
    pub fn center_order(&self) -> usize {
        let n = self.order;
        (0..n)
            .filter(|&g| (0..n).all(|h| self.mul(g, h) == self.mul(h, g)))
            .count()
    }

    /// Membership vector of the subgroup generated by `gens`.
    pub fn subgroup(&self, gens: &[u16]) -> Vec<bool> {
        let mut inside = vec![false; self.order];
        inside[0] = true;
        let mut stack = vec![0usize];
        while let Some(x) = stack.pop() {
            for &s in gens {
                let y = self.mul(x, s as usize);
                if !inside[y] {
                    inside[y] = true;
                    stack.push(y);
                }
            }
        }
        inside
    }

    /// `Z_l x Z_m` with element `(i, j) = i m + j` (the indexing of
    /// `qec::bicycle::TwoBlockCode`) and its full automorphism group.
    pub fn abelian(l: usize, m: usize) -> Self {
        let n = l * m;
        let mut mul = vec![0u16; n * n];
        for g in 0..n {
            for h in 0..n {
                mul[g * n + h] = (((g / m + h / m) % l) * m + (g % m + h % m) % m) as u16;
            }
        }
        let auts = super::bb_search::AbelianGroup::new(l, m).auts;
        let label = if m == 1 {
            format!("Z{l}")
        } else {
            format!("Z{l}xZ{m}")
        };
        FiniteGroup::from_table(n, mul, auts, label).expect("Z_l x Z_m is a group")
    }

    /// The semidirect product `Z_m : Z_k` with `s r s^-1 = r^q`: elements
    /// `r^i s^j` indexed `j m + i`, `(r^i1 s^j1)(r^i2 s^j2) =
    /// r^(i1 + q^j1 i2) s^(j1 + j2)`. Needs `q^k = 1 (mod m)` and
    /// `gcd(q, m) = 1`. Dihedral: `q = m - 1`, `k = 2`. Automorphisms by brute
    /// force ([`Self::automorphisms_bruteforce`]), so keep `m k` small.
    pub fn metacyclic(m: usize, k: usize, q: usize) -> Result<Self, GroupError> {
        let n = m * k;
        let mut qp = vec![1usize; k + 1];
        for j in 1..=k {
            qp[j] = qp[j - 1] * q % m;
        }
        if qp[k] != 1 % m {
            return Err(GroupError::NotAGroup(format!("{q}^{k} != 1 mod {m}")));
        }
        let mut mul = vec![0u16; n * n];
        for x in 0..n {
            let (i1, j1) = (x % m, x / m);
            for y in 0..n {
                let (i2, j2) = (y % m, y / m);
                let i = (i1 + qp[j1] * i2) % m;
                let j = (j1 + j2) % k;
                mul[x * n + y] = (j * m + i) as u16;
            }
        }
        let mut g = FiniteGroup::from_table(n, mul, Vec::new(), format!("Z{m}:Z{k}(q={q})"))?;
        g.aut_gens = g.automorphisms_bruteforce();
        Ok(g)
    }

    /// Direct product `G x H`, element `(g, h) = g |H| + h`; automorphism
    /// generators are those of the factors (a subgroup of `Aut(G x H)`).
    pub fn direct_product(&self, other: &FiniteGroup) -> Self {
        let (n1, n2) = (self.order, other.order);
        let n = n1 * n2;
        let mut mul = vec![0u16; n * n];
        for x in 0..n {
            for y in 0..n {
                let g = self.mul(x / n2, y / n2);
                let h = other.mul(x % n2, y % n2);
                mul[x * n + y] = (g * n2 + h) as u16;
            }
        }
        let mut auts = Vec::new();
        for s in &self.aut_gens {
            auts.push(
                (0..n)
                    .map(|x| s[x / n2] * n2 as u16 + (x % n2) as u16)
                    .collect(),
            );
        }
        for s in &other.aut_gens {
            auts.push((0..n).map(|x| ((x / n2) * n2) as u16 + s[x % n2]).collect());
        }
        FiniteGroup::from_table(n, mul, auts, format!("{}x{}", self.label, other.label))
            .expect("direct product of groups")
    }

    /// Every automorphism, by brute force over images of a greedy
    /// generating set (exponential in the number of generators: for small
    /// groups and tests).
    pub fn automorphisms_bruteforce(&self) -> Vec<Vec<u16>> {
        let n = self.order;
        let elem_order = |g: usize| {
            let (mut x, mut k) = (g, 1);
            while x != 0 {
                x = self.mul(x, g);
                k += 1;
            }
            k
        };
        // greedy generating set
        let mut gens: Vec<u16> = Vec::new();
        let mut inside = self.subgroup(&gens);
        while let Some(g) = (0..n)
            .filter(|&g| !inside[g])
            .max_by_key(|&g| elem_order(g))
        {
            gens.push(g as u16);
            inside = self.subgroup(&gens);
        }
        // spanning tree: word[x] = (parent, generator index)
        let mut parent = vec![(usize::MAX, usize::MAX); n];
        parent[0] = (0, 0);
        let mut order = vec![0usize];
        let mut i = 0;
        while i < order.len() {
            let x = order[i];
            i += 1;
            for (gi, &s) in gens.iter().enumerate() {
                let y = self.mul(x, s as usize);
                if parent[y].0 == usize::MAX {
                    parent[y] = (x, gi);
                    order.push(y);
                }
            }
        }
        let orders: Vec<usize> = gens.iter().map(|&g| elem_order(g as usize)).collect();
        let cands: Vec<Vec<usize>> = orders
            .iter()
            .map(|&o| (0..n).filter(|&g| elem_order(g) == o).collect())
            .collect();
        let mut out = Vec::new();
        let mut idx = vec![0usize; gens.len()];
        if gens.is_empty() {
            return vec![vec![0]];
        }
        'outer: loop {
            let img: Vec<usize> = idx.iter().zip(&cands).map(|(&i, c)| c[i]).collect();
            let mut s = vec![0u16; n];
            for &x in order.iter().skip(1) {
                let (p, gi) = parent[x];
                s[x] = self.mul(s[p] as usize, img[gi]) as u16;
            }
            if self.is_automorphism(&s) {
                out.push(s);
            }
            for t in 0..idx.len() {
                idx[t] += 1;
                if idx[t] < cands[t].len() {
                    continue 'outer;
                }
                idx[t] = 0;
            }
            break;
        }
        out
    }

    /// Parses the export format of `research/data/code-discovery-2/export_groups.g`:
    /// per group a header `G <N> <id> <|Z|> <|Aut|> <ngens> <name>`, `N` lines
    /// of the multiplication table and `ngens` automorphism lines.
    pub fn parse_export(text: &str) -> Result<Vec<FiniteGroup>, GroupError> {
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let mut out = Vec::new();
        let nums = |l: &str| -> Result<Vec<u16>, GroupError> {
            l.split_whitespace()
                .map(|t| {
                    t.parse::<u16>()
                        .map_err(|e| GroupError::Parse(e.to_string()))
                })
                .collect()
        };
        while let Some(h) = lines.next() {
            let f: Vec<&str> = h.split_whitespace().collect();
            if f.len() < 7 || f[0] != "G" {
                return Err(GroupError::Parse(format!("bad header {h:?}")));
            }
            let p = |s: &str| {
                s.parse::<usize>()
                    .map_err(|e| GroupError::Parse(e.to_string()))
            };
            let (n, id, ngens) = (p(f[1])?, p(f[2])?, p(f[5])?);
            let mut mul = Vec::with_capacity(n * n);
            for _ in 0..n {
                let row = nums(
                    lines
                        .next()
                        .ok_or(GroupError::Parse("short table".into()))?,
                )?;
                if row.len() != n {
                    return Err(GroupError::Parse("bad row length".into()));
                }
                mul.extend(row);
            }
            let mut auts = Vec::new();
            for _ in 0..ngens {
                auts.push(nums(
                    lines.next().ok_or(GroupError::Parse("short auts".into()))?,
                )?);
            }
            out.push(FiniteGroup::from_table(
                n,
                mul,
                auts,
                format!("SmallGroup({n},{id}) {}", f[6..].join(" ")),
            )?);
        }
        Ok(out)
    }
}

/// A two-block group-algebra code: subsets `A`, `B` of a group (see the
/// [module docs](self) for the convention).
#[derive(Clone, Debug)]
pub struct GroupCode<'g> {
    /// The group.
    pub g: &'g FiniteGroup,
    /// Elements of `A`.
    pub a: Vec<u16>,
    /// Elements of `B`.
    pub b: Vec<u16>,
}

/// Rank over GF(2) of rows packed into `W` words (destroys `rows`).
fn rank_rows<const W: usize>(rows: &mut [[u64; W]], ncols: usize) -> usize {
    let nr = rows.len();
    let mut rank = 0;
    for c in 0..ncols {
        let (wc, bc) = (c / 64, 1u64 << (c % 64));
        let Some(p) = (rank..nr).find(|&i| rows[i][wc] & bc != 0) else {
            continue;
        };
        rows.swap(p, rank);
        let pr = rows[rank];
        for i in rank + 1..nr {
            if rows[i][wc] & bc != 0 {
                for t in wc..W {
                    rows[i][t] ^= pr[t];
                }
            }
        }
        rank += 1;
        if rank == nr {
            break;
        }
    }
    rank
}

impl<'g> GroupCode<'g> {
    /// Code from element lists (repeated elements are rejected).
    pub fn new(g: &'g FiniteGroup, a: &[u16], b: &[u16]) -> Self {
        for v in [a, b] {
            let s: HashSet<u16> = v.iter().copied().collect();
            assert_eq!(s.len(), v.len(), "repeated element");
            assert!(v.iter().all(|&x| (x as usize) < g.order));
        }
        GroupCode {
            g,
            a: a.to_vec(),
            b: b.to_vec(),
        }
    }

    /// Group order `N`.
    pub fn order(&self) -> usize {
        self.g.order
    }

    /// Number of physical qubits `2N`.
    pub fn n(&self) -> usize {
        2 * self.g.order
    }

    /// Support of X-check `g` (qubits `0..N` are `L`, `N..2N` are `R`).
    pub fn x_check(&self, g: usize) -> Vec<usize> {
        let n = self.order();
        let mut s: Vec<usize> = self.a.iter().map(|&a| self.g.mul(g, a as usize)).collect();
        s.extend(self.b.iter().map(|&b| n + self.g.mul(b as usize, g)));
        s
    }

    /// Support of Z-check `h`.
    pub fn z_check(&self, h: usize) -> Vec<usize> {
        let n = self.order();
        let mut s: Vec<usize> = self
            .b
            .iter()
            .map(|&b| self.g.mul(self.g.inv(b as usize), h))
            .collect();
        s.extend(
            self.a
                .iter()
                .map(|&a| n + self.g.mul(h, self.g.inv(a as usize))),
        );
        s
    }

    fn matrix(&self, z: bool) -> Gf2Mat {
        let n = self.order();
        let mut h = Gf2Mat::zeros(n, 2 * n);
        for r in 0..n {
            let sup = if z { self.z_check(r) } else { self.x_check(r) };
            for q in sup {
                h.flip(r, q);
            }
        }
        h
    }

    /// `H_X` (`N x 2N`).
    pub fn hx(&self) -> Gf2Mat {
        self.matrix(false)
    }

    /// `H_Z` (`N x 2N`).
    pub fn hz(&self) -> Gf2Mat {
        self.matrix(true)
    }

    /// `(rank H_X, rank H_Z)` over GF(2).
    pub fn ranks(&self) -> (usize, usize) {
        let n = self.order();
        if 2 * n <= 320 {
            let build = |z: bool| -> Vec<[u64; 5]> {
                (0..n)
                    .map(|r| {
                        let mut row = [0u64; 5];
                        let sup = if z { self.z_check(r) } else { self.x_check(r) };
                        for q in sup {
                            row[q / 64] ^= 1 << (q % 64);
                        }
                        row
                    })
                    .collect()
            };
            (
                rank_rows(&mut build(false), 2 * n),
                rank_rows(&mut build(true), 2 * n),
            )
        } else {
            (self.hx().rank(), self.hz().rank())
        }
    }

    /// Number of logical qubits `n - rank H_X - rank H_Z`.
    pub fn k(&self) -> usize {
        let (rx, rz) = self.ranks();
        self.n() - rx - rz
    }

    /// True if the Tanner graph is connected, i.e. `<A'> <B'> = G` for the
    /// subgroups generated by `A a0^-1` and `b0^-1 B`. Otherwise the code is a
    /// disjoint union of smaller codes.
    pub fn is_connected(&self) -> bool {
        let g = self.g;
        let a0 = g.inv(self.a[0] as usize);
        let b0 = g.inv(self.b[0] as usize);
        let ga: Vec<u16> = self
            .a
            .iter()
            .map(|&a| g.mul(a as usize, a0) as u16)
            .collect();
        let gb: Vec<u16> = self
            .b
            .iter()
            .map(|&b| g.mul(b0, b as usize) as u16)
            .collect();
        let (ha, hb) = (g.subgroup(&ga), g.subgroup(&gb));
        let mut prod = vec![false; g.order];
        for x in (0..g.order).filter(|&x| ha[x]) {
            for y in (0..g.order).filter(|&y| hb[y]) {
                prod[g.mul(x, y)] = true;
            }
        }
        prod.iter().all(|&p| p)
    }

    /// True if the qubit permutation `perm` (length `2N`) maps the set of
    /// X-check supports onto itself and the set of Z-check supports onto
    /// itself, i.e. is an automorphism of the CSS code.
    pub fn is_qubit_automorphism(&self, perm: &[usize]) -> bool {
        let n = self.order();
        for z in [false, true] {
            let sup = |r: usize| {
                let mut s = if z { self.z_check(r) } else { self.x_check(r) };
                s.sort_unstable();
                s
            };
            let all: HashSet<Vec<usize>> = (0..n).map(sup).collect();
            for r in 0..n {
                let mut s: Vec<usize> = sup(r).iter().map(|&q| perm[q]).collect();
                s.sort_unstable();
                if !all.contains(&s) {
                    return false;
                }
            }
        }
        true
    }

    /// Code automorphisms from two-sided translations fixing `A` and `B`:
    /// `L x -> t^-1 x w`, `R x -> v x u^-1` for `u A w = A`, `v B t = B`.
    /// Returns verified qubit permutations generating a group whose orbits
    /// are those of the full set (one generator per solution pair with the
    /// other side trivial).
    pub fn translation_automorphisms(&self) -> Vec<Vec<usize>> {
        let g = self.g;
        let n = g.order;
        // all (u, w) with u S w = S
        let stab = |s: &[u16]| -> Vec<(usize, usize)> {
            let set: HashSet<usize> = s.iter().map(|&x| x as usize).collect();
            let mut out = Vec::new();
            for u in 0..n {
                let us0 = g.mul(u, s[0] as usize);
                for &y in s {
                    let w = g.mul(g.inv(us0), y as usize);
                    if s.iter()
                        .all(|&x| set.contains(&g.mul(g.mul(u, x as usize), w)))
                    {
                        out.push((u, w));
                    }
                }
            }
            out
        };
        let mut perms = Vec::new();
        for (u, w) in stab(&self.a) {
            // with (v, t) = (1, 1): L x -> x w, R x -> x u^-1
            let ui = g.inv(u);
            let p: Vec<usize> = (0..n)
                .map(|x| g.mul(x, w))
                .chain((0..n).map(|x| n + g.mul(x, ui)))
                .collect();
            perms.push(p);
        }
        for (v, t) in stab(&self.b) {
            // with (u, w) = (1, 1): L x -> t^-1 x, R x -> v x
            let ti = g.inv(t);
            let p: Vec<usize> = (0..n)
                .map(|x| g.mul(ti, x))
                .chain((0..n).map(|x| n + g.mul(v, x)))
                .collect();
            perms.push(p);
        }
        perms.retain(|p| self.is_qubit_automorphism(p));
        perms
    }

    /// Term-preserving automorphisms, for syndrome circuits that schedule
    /// checks by term: `(u, w)` with `u a w = a` for every `a` in `A` and
    /// `(v, t)` with `v b t = b` for every `b` in `B` act as X-check
    /// `g -> t^-1 g u^-1`, Z-check `h -> v h w`, `L x -> t^-1 x w`,
    /// `R x -> v x u^-1` and send every term of every check to the same term
    /// of the image check. Returns generators as `(X-check permutation,
    /// Z-check permutation, qubit permutation)`, each verified.
    pub fn term_automorphisms(&self) -> Vec<(Vec<usize>, Vec<usize>, Vec<usize>)> {
        let g = self.g;
        let n = g.order;
        let mut out = Vec::new();
        // u-part (v = t = 1): w = a0^-1 u^-1 a0 must satisfy u a w = a for all a
        let a0 = self.a[0] as usize;
        for u in 0..n {
            let ui = g.inv(u);
            let w = g.mul(g.mul(g.inv(a0), ui), a0);
            if self
                .a
                .iter()
                .all(|&a| g.mul(g.mul(u, a as usize), w) == a as usize)
            {
                let xc: Vec<usize> = (0..n).map(|x| g.mul(x, ui)).collect();
                let zc: Vec<usize> = (0..n).map(|h| g.mul(h, w)).collect();
                let q: Vec<usize> = (0..n)
                    .map(|x| g.mul(x, w))
                    .chain((0..n).map(|x| n + g.mul(x, ui)))
                    .collect();
                out.push((xc, zc, q));
            }
        }
        let b0 = self.b[0] as usize;
        for v in 0..n {
            let t = g.mul(g.mul(g.inv(b0), g.inv(v)), b0);
            if self
                .b
                .iter()
                .all(|&b| g.mul(g.mul(v, b as usize), t) == b as usize)
            {
                let ti = g.inv(t);
                let xc: Vec<usize> = (0..n).map(|x| g.mul(ti, x)).collect();
                let zc: Vec<usize> = (0..n).map(|h| g.mul(v, h)).collect();
                let q: Vec<usize> = (0..n)
                    .map(|x| g.mul(ti, x))
                    .chain((0..n).map(|x| n + g.mul(v, x)))
                    .collect();
                out.push((xc, zc, q));
            }
        }
        // verify term by term
        out.retain(|(xc, zc, q)| {
            (0..n).all(|c| {
                let (xs, xi) = (self.x_check(c), self.x_check(xc[c]));
                let (zs, zi) = (self.z_check(c), self.z_check(zc[c]));
                xs.iter().zip(&xi).all(|(&a, &b)| q[a] == b)
                    && zs.iter().zip(&zi).all(|(&a, &b)| q[a] == b)
            })
        });
        out
    }

    /// Roots for `min_weight_logical`: the first qubit of each orbit of the
    /// [translation automorphisms](Self::translation_automorphisms), each
    /// banning all earlier orbits.
    pub fn distance_roots(&self) -> Vec<(usize, Vec<usize>)> {
        orbit_roots(self.n(), &self.translation_automorphisms())
    }

    /// Both CSS distances: `(Z-type, checked by H_X; X-type, checked by
    /// H_Z)`, each exact when its `lower == upper`.
    pub fn distances(&self, opts: &DistanceOpts) -> (DistanceResult, DistanceResult) {
        let (hx, hz) = (self.hx(), self.hz());
        let roots = self.distance_roots();
        (
            code_distance_roots(&hx, &hz, &roots, opts),
            code_distance_roots(&hz, &hx, &roots, opts),
        )
    }
}

/// Orbits of the group generated by `perms` on `0..n`, as roots for
/// `min_weight_logical`: `(first element of orbit i, all elements of orbits
/// < i)`.
pub fn orbit_roots(n: usize, perms: &[Vec<usize>]) -> Vec<(usize, Vec<usize>)> {
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for p in perms {
        for x in 0..n {
            let (a, b) = (find(&mut parent, x), find(&mut parent, p[x]));
            if a != b {
                parent[a.max(b)] = a.min(b);
            }
        }
    }
    let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
    for x in 0..n {
        let r = find(&mut parent, x);
        members.entry(r).or_default().push(x);
    }
    let mut orbits: Vec<Vec<usize>> = members.into_values().collect();
    orbits.sort();
    let mut roots = Vec::new();
    let mut banned = Vec::new();
    for o in orbits {
        roots.push((o[0], banned.clone()));
        banned.extend(o);
    }
    roots
}

/// Like `qec::bicycle::code_distance`, with explicit search roots (see
/// [`orbit_roots`]).
pub fn code_distance_roots(
    hcheck: &Gf2Mat,
    hother: &Gf2Mat,
    roots: &[(usize, Vec<usize>)],
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
    let top = opts.max_weight.min(ub.saturating_sub(1));
    let (out, nodes) = min_weight_logical(hcheck, &masks, roots, 1, top, opts.node_limit);
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

/// Right cosets `H x` of a subgroup `H` of a [`FiniteGroup`].
#[derive(Clone, Debug)]
pub struct Cosets {
    /// Elements of `H`.
    pub h: Vec<u16>,
    /// Index of the coset `H x` of every element `x`.
    pub coset_of: Vec<u16>,
    /// One representative per coset.
    pub reps: Vec<u16>,
}

impl Cosets {
    /// Right cosets of `h`, which must be a subgroup (checked).
    pub fn new(g: &FiniteGroup, h: &[u16]) -> Result<Self, GroupError> {
        let n = g.order;
        let inside: HashSet<usize> = h.iter().map(|&x| x as usize).collect();
        if !inside.contains(&0)
            || h.iter().any(|&x| {
                h.iter()
                    .any(|&y| !inside.contains(&g.mul(x as usize, y as usize)))
            })
        {
            return Err(GroupError::NotAGroup("H is not a subgroup".into()));
        }
        let mut coset_of = vec![u16::MAX; n];
        let mut reps = Vec::new();
        for x in 0..n {
            if coset_of[x] != u16::MAX {
                continue;
            }
            let id = reps.len() as u16;
            for &y in h {
                coset_of[g.mul(y as usize, x)] = id;
            }
            reps.push(x as u16);
        }
        Ok(Cosets {
            h: h.to_vec(),
            coset_of,
            reps,
        })
    }

    /// Number of cosets `[G : H]`.
    pub fn len(&self) -> usize {
        self.reps.len()
    }

    /// True for the trivial group (never: `H` has at least one coset).
    pub fn is_empty(&self) -> bool {
        self.reps.is_empty()
    }

    /// Membership vector of the normaliser `N_G(H) = {x : x H x^-1 = H}`.
    pub fn normalizer(&self, g: &FiniteGroup) -> Vec<bool> {
        let inside: HashSet<u16> = self.h.iter().copied().collect();
        (0..g.order)
            .map(|x| {
                let xi = g.inv(x);
                self.h
                    .iter()
                    .all(|&y| inside.contains(&(g.mul(g.mul(x, y as usize), xi) as u16)))
            })
            .collect()
    }

    /// True if `H` is normal in `G`.
    pub fn is_normal(&self, g: &FiniteGroup) -> bool {
        self.normalizer(g).iter().all(|&b| b)
    }
}

/// A coset code (the construction of Aydin, Tamo & Barg, arXiv:2606.17268,
/// written in this module's convention): qubits and checks are the right
/// cosets `H x`; `A` (any elements of `G`) acts by right multiplication and
/// `B` (elements of the normaliser `N_G(H)`, so that left multiplication is
/// well defined on cosets) by left multiplication:
///
/// ```text
/// X-check Hg:  L{H g a}       R{H b g}
/// Z-check Hh:  L{H b^-1 h}    R{H h a^-1}
/// ```
///
/// Both overlaps of X-check `Hg` and Z-check `Hh` count the pairs with
/// `H b g a = H h`, so the checks commute. With `H = {1}` this is
/// [`GroupCode`]; with `H` normal it is the [`GroupCode`] of `G / H`.
#[derive(Clone, Debug)]
pub struct CosetCode<'g> {
    /// The group.
    pub g: &'g FiniteGroup,
    /// The cosets of `H`.
    pub c: &'g Cosets,
    /// Elements of `A`.
    pub a: Vec<u16>,
    /// Elements of `B` (in `N_G(H)`).
    pub b: Vec<u16>,
}

impl<'g> CosetCode<'g> {
    /// Code from element lists; panics if an element of `B` does not
    /// normalise `H`.
    pub fn new(g: &'g FiniteGroup, c: &'g Cosets, a: &[u16], b: &[u16]) -> Self {
        let nor = c.normalizer(g);
        assert!(b.iter().all(|&x| nor[x as usize]), "B must normalise H");
        CosetCode {
            g,
            c,
            a: a.to_vec(),
            b: b.to_vec(),
        }
    }

    /// Number of cosets `N = [G : H]` (checks of each type).
    pub fn order(&self) -> usize {
        self.c.len()
    }

    /// Number of physical qubits `2N`.
    pub fn n(&self) -> usize {
        2 * self.order()
    }

    /// Support of X-check `i` (qubits `0..N` are `L`, `N..2N` are `R`).
    pub fn x_check(&self, i: usize) -> Vec<usize> {
        let (g, n, r) = (self.g, self.order(), self.c.reps[i] as usize);
        let co = |x: usize| self.c.coset_of[x] as usize;
        let mut s: Vec<usize> = self.a.iter().map(|&a| co(g.mul(r, a as usize))).collect();
        s.extend(self.b.iter().map(|&b| n + co(g.mul(b as usize, r))));
        s
    }

    /// Support of Z-check `i`.
    pub fn z_check(&self, i: usize) -> Vec<usize> {
        let (g, n, r) = (self.g, self.order(), self.c.reps[i] as usize);
        let co = |x: usize| self.c.coset_of[x] as usize;
        let mut s: Vec<usize> = self
            .b
            .iter()
            .map(|&b| co(g.mul(g.inv(b as usize), r)))
            .collect();
        s.extend(self.a.iter().map(|&a| n + co(g.mul(r, g.inv(a as usize)))));
        s
    }

    /// The code as explicit check supports.
    pub fn explicit(&self) -> ExplicitCode {
        ExplicitCode {
            n: self.n(),
            x: (0..self.order()).map(|i| self.x_check(i)).collect(),
            z: (0..self.order()).map(|i| self.z_check(i)).collect(),
        }
    }

    /// Candidate automorphisms from two-sided translations fixing `A` (any
    /// `u, w`) or `B` (`v, t` in the normaliser), as qubit permutations;
    /// only verified ones are returned.
    pub fn translation_automorphisms(&self) -> Vec<Vec<usize>> {
        let (g, n) = (self.g, self.order());
        let gn = g.order;
        let nor = self.c.normalizer(g);
        let co = |x: usize| self.c.coset_of[x] as usize;
        let stab = |s: &[u16], restrict: bool| -> Vec<(usize, usize)> {
            let set: HashSet<usize> = s.iter().map(|&x| x as usize).collect();
            let mut out = Vec::new();
            for u in (0..gn).filter(|&u| !restrict || nor[u]) {
                let us0 = g.mul(u, s[0] as usize);
                for &y in s {
                    let w = g.mul(g.inv(us0), y as usize);
                    if (!restrict || nor[w])
                        && s.iter()
                            .all(|&x| set.contains(&g.mul(g.mul(u, x as usize), w)))
                    {
                        out.push((u, w));
                    }
                }
            }
            out
        };
        let reps = &self.c.reps;
        let mut perms = Vec::new();
        for (u, w) in stab(&self.a, false) {
            let ui = g.inv(u);
            perms.push(
                (0..n)
                    .map(|i| co(g.mul(reps[i] as usize, w)))
                    .chain((0..n).map(|i| n + co(g.mul(reps[i] as usize, ui))))
                    .collect::<Vec<usize>>(),
            );
        }
        for (v, t) in stab(&self.b, true) {
            let ti = g.inv(t);
            perms.push(
                (0..n)
                    .map(|i| co(g.mul(ti, reps[i] as usize)))
                    .chain((0..n).map(|i| n + co(g.mul(v, reps[i] as usize))))
                    .collect::<Vec<usize>>(),
            );
        }
        let e = self.explicit();
        perms.retain(|p| e.is_automorphism(p));
        perms
    }
}

/// A CSS code given by explicit check supports (`x[i]`, `z[i]`) on `n`
/// qubits, with the generic tools used by the searches.
#[derive(Clone, Debug)]
pub struct ExplicitCode {
    /// Number of qubits.
    pub n: usize,
    /// X-check supports (a qubit listed twice cancels).
    pub x: Vec<Vec<usize>>,
    /// Z-check supports.
    pub z: Vec<Vec<usize>>,
}

impl ExplicitCode {
    fn matrix(&self, z: bool) -> Gf2Mat {
        let rows = if z { &self.z } else { &self.x };
        let mut h = Gf2Mat::zeros(rows.len(), self.n);
        for (r, sup) in rows.iter().enumerate() {
            for &q in sup {
                h.flip(r, q);
            }
        }
        h
    }

    /// `H_X`.
    pub fn hx(&self) -> Gf2Mat {
        self.matrix(false)
    }

    /// `H_Z`.
    pub fn hz(&self) -> Gf2Mat {
        self.matrix(true)
    }

    /// True if every X-check meets every Z-check an even number of times.
    pub fn commutes(&self) -> bool {
        let (hx, hz) = (self.hx(), self.hz());
        (0..hx.rows).all(|i| {
            (0..hz.rows).all(|j| {
                hx.row(i)
                    .iter()
                    .zip(hz.row(j))
                    .map(|(a, b)| (a & b).count_ones())
                    .sum::<u32>()
                    % 2
                    == 0
            })
        })
    }

    /// `(rank H_X, rank H_Z)`.
    pub fn ranks(&self) -> (usize, usize) {
        if self.n <= 320 {
            let build = |rows: &[Vec<usize>]| -> Vec<[u64; 5]> {
                rows.iter()
                    .map(|sup| {
                        let mut row = [0u64; 5];
                        for &q in sup {
                            row[q / 64] ^= 1 << (q % 64);
                        }
                        row
                    })
                    .collect()
            };
            (
                rank_rows(&mut build(&self.x), self.n),
                rank_rows(&mut build(&self.z), self.n),
            )
        } else {
            (self.hx().rank(), self.hz().rank())
        }
    }

    /// Number of logical qubits.
    pub fn k(&self) -> usize {
        let (rx, rz) = self.ranks();
        self.n - rx - rz
    }

    /// True if the Tanner graph (qubits and both check types) is connected.
    pub fn is_connected(&self) -> bool {
        let mut qchecks: Vec<Vec<usize>> = vec![Vec::new(); self.n];
        let all: Vec<&Vec<usize>> = self.x.iter().chain(self.z.iter()).collect();
        for (c, sup) in all.iter().enumerate() {
            for &q in sup.iter() {
                qchecks[q].push(c);
            }
        }
        let mut seen = vec![false; self.n];
        let mut cseen = vec![false; all.len()];
        let mut stack = vec![0usize];
        seen[0] = true;
        while let Some(q) = stack.pop() {
            for &c in &qchecks[q] {
                if !cseen[c] {
                    cseen[c] = true;
                    for &q2 in all[c].iter() {
                        if !seen[q2] {
                            seen[q2] = true;
                            stack.push(q2);
                        }
                    }
                }
            }
        }
        seen.iter().all(|&b| b)
    }

    /// True if `perm` maps the X-check supports onto themselves and the
    /// Z-check supports onto themselves (as sets of sets, after cancelling
    /// repeated qubits).
    pub fn is_automorphism(&self, perm: &[usize]) -> bool {
        if perm.len() != self.n {
            return false;
        }
        let mut seen = vec![false; self.n];
        for &p in perm {
            if p >= self.n || seen[p] {
                return false;
            }
            seen[p] = true;
        }
        let norm = |s: &[usize]| {
            let mut v = s.to_vec();
            v.sort_unstable();
            let mut out: Vec<usize> = Vec::new();
            for q in v {
                if out.last() == Some(&q) {
                    out.pop();
                } else {
                    out.push(q);
                }
            }
            out
        };
        for rows in [&self.x, &self.z] {
            let all: HashSet<Vec<usize>> = rows.iter().map(|s| norm(s)).collect();
            for sup in rows {
                let img: Vec<usize> = sup.iter().map(|&q| perm[q]).collect();
                if !all.contains(&norm(&img)) {
                    return false;
                }
            }
        }
        true
    }

    /// Both CSS distances `(Z-type, X-type)` with the given search roots.
    pub fn distances(
        &self,
        roots: &[(usize, Vec<usize>)],
        opts: &DistanceOpts,
    ) -> (DistanceResult, DistanceResult) {
        let (hx, hz) = (self.hx(), self.hz());
        (
            code_distance_roots(&hx, &hz, roots, opts),
            code_distance_roots(&hz, &hx, roots, opts),
        )
    }
}

/// The quotient `N_G(H) / H` as a group, with one lift (an element of
/// `N_G(H)`) per element of the quotient; the identity's lift is `0`.
/// Automorphism generators are not computed (none are needed to enumerate
/// T-classes).
pub fn normalizer_quotient(g: &FiniteGroup, c: &Cosets) -> (FiniteGroup, Vec<u16>) {
    let nor = c.normalizer(g);
    let mut lift: Vec<u16> = Vec::new();
    let mut idx: HashMap<u16, u16> = HashMap::new();
    for x in (0..g.order).filter(|&x| nor[x]) {
        let k = c.coset_of[x];
        if let std::collections::hash_map::Entry::Vacant(e) = idx.entry(k) {
            e.insert(lift.len() as u16);
            lift.push(x as u16);
        }
    }
    let q = lift.len();
    let mut mul = vec![0u16; q * q];
    for i in 0..q {
        for j in 0..q {
            let p = g.mul(lift[i] as usize, lift[j] as usize);
            mul[i * q + j] = idx[&c.coset_of[p]];
        }
    }
    let grp = FiniteGroup::from_table(q, mul, Vec::new(), "N(H)/H").expect("quotient is a group");
    (grp, lift)
}

/// T-classes of identity-containing `w`-subsets and orbits of pairs of
/// T-classes: one representative `(A, B)` per equivalence class of codes
/// (see the [module docs](self)).
pub struct Enumeration {
    /// Subset weight `w`.
    pub weight: usize,
    /// Representative (sorted, identity-containing) of each T-class.
    pub reps: Vec<Vec<u16>>,
    /// T-class of every identity-containing sorted subset.
    pub class_of: HashMap<Vec<u16>, u32>,
}

impl Enumeration {
    /// All T-classes of `w`-subsets of `g`.
    pub fn new(g: &FiniteGroup, w: usize) -> Self {
        let n = g.order;
        let mut class_of: HashMap<Vec<u16>, u32> = HashMap::new();
        let mut reps = Vec::new();
        let mut cur: Vec<u16> = vec![0];
        fn subsets(
            n: usize,
            w: usize,
            start: usize,
            cur: &mut Vec<u16>,
            f: &mut dyn FnMut(&[u16]),
        ) {
            if cur.len() == w {
                f(cur);
                return;
            }
            for x in start..n {
                cur.push(x as u16);
                subsets(n, w, x + 1, cur, f);
                cur.pop();
            }
        }
        let mut all: Vec<Vec<u16>> = Vec::new();
        subsets(n, w, 1, &mut cur, &mut |s| all.push(s.to_vec()));
        for s in &all {
            if class_of.contains_key(s) {
                continue;
            }
            let id = reps.len() as u32;
            let mut best = s.clone();
            for &x in s.iter() {
                let xi = g.inv(x as usize);
                // S x^-1, then all conjugates
                let t: Vec<usize> = s.iter().map(|&y| g.mul(y as usize, xi)).collect();
                for c in 0..n {
                    let ci = g.inv(c);
                    let mut v: Vec<u16> =
                        t.iter().map(|&y| g.mul(g.mul(c, y), ci) as u16).collect();
                    v.sort_unstable();
                    if v < best {
                        best = v.clone();
                    }
                    class_of.insert(v, id);
                }
            }
            reps.push(best);
        }
        Enumeration {
            weight: w,
            reps,
            class_of,
        }
    }

    /// T-class of an identity-containing subset (any order).
    pub fn class(&self, s: &[u16]) -> u32 {
        let mut v = s.to_vec();
        v.sort_unstable();
        self.class_of[&v]
    }

    /// T-class of an arbitrary subset (translated to contain the identity).
    pub fn class_any(&self, g: &FiniteGroup, s: &[u16]) -> u32 {
        let x0 = g.inv(s[0] as usize);
        let v: Vec<u16> = s.iter().map(|&y| g.mul(y as usize, x0) as u16).collect();
        self.class(&v)
    }

    /// Image of every T-class under each automorphism generator, and under
    /// inversion `S -> S^-1`.
    fn actions(&self, g: &FiniteGroup) -> (Vec<Vec<u32>>, Vec<u32>) {
        let auts = g
            .aut_gens
            .iter()
            .map(|s| {
                self.reps
                    .iter()
                    .map(|r| {
                        let v: Vec<u16> = r.iter().map(|&x| s[x as usize]).collect();
                        self.class(&v)
                    })
                    .collect()
            })
            .collect();
        let inv = self
            .reps
            .iter()
            .map(|r| {
                let v: Vec<u16> = r.iter().map(|&x| g.inv[x as usize]).collect();
                self.class(&v)
            })
            .collect();
        (auts, inv)
    }

    /// One representative `(class of A, class of B)` per orbit of pairs
    /// under the automorphism generators, the swap and the inversion, with
    /// the orbit size.
    pub fn pair_orbits(&self, g: &FiniteGroup) -> Vec<(u32, u32, usize)> {
        let m = self.reps.len();
        let (auts, inv) = self.actions(g);
        let mut parent: Vec<u32> = (0..(m * m) as u32).collect();
        fn find(p: &mut [u32], mut x: u32) -> u32 {
            while p[x as usize] != x {
                p[x as usize] = p[p[x as usize] as usize];
                x = p[x as usize];
            }
            x
        }
        let join = |p: &mut Vec<u32>, x: usize, y: usize| {
            let (a, b) = (find(p, x as u32), find(p, y as u32));
            if a != b {
                p[a.max(b) as usize] = a.min(b);
            }
        };
        for i in 0..m {
            for j in 0..m {
                let x = i * m + j;
                for s in &auts {
                    join(&mut parent, x, s[i] as usize * m + s[j] as usize);
                }
                join(&mut parent, x, j * m + i);
                join(&mut parent, x, inv[i] as usize * m + inv[j] as usize);
            }
        }
        let mut size: HashMap<u32, usize> = HashMap::new();
        for x in 0..m * m {
            *size.entry(find(&mut parent, x as u32)).or_default() += 1;
        }
        let mut out: Vec<(u32, u32, usize)> = size
            .into_iter()
            .map(|(r, s)| (r / m as u32, r % m as u32, s))
            .collect();
        out.sort_unstable();
        out
    }
}
