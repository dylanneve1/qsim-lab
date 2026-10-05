//! Signature-tensor reduction: TODD (Heyfron & Campbell, "An efficient
//! quantum compiler that reduces T count", arXiv:1712.01557) with a
//! FastTODD-style kernel computation.
//!
//! A CNOT+T region applies `ω^{Σ_j p_j·u}` (ω = e^{iπ/4}) for a list of
//! parities `p_j ∈ F_2^d` (the columns of a `d × m` matrix `A`), one T gate
//! each, up to Clifford gates. Two lists are equal up to a diagonal
//! Clifford (S, Z and CZ on parities) exactly when their *signature
//! tensors* `S = Σ_j p_j ⊗ p_j ⊗ p_j (mod 2)` agree, so the T-count of
//! the region is the symmetric tensor rank of `S` over F_2.
//!
//! TODD lowers `m` by updates `A → A + z yᵀ` that keep `S`. With
//! `B(y) = A·diag(y)·Aᵀ`, the update keeps `S` when `A y = 0`, `|y|` is
//! even and `B(y) = z wᵀ + w zᵀ` for some `w` (the "χ(A, z)" condition of
//! the paper, written as a bilinear form). Choosing `z = p_a + p_b` and a
//! `y` with `y_a ≠ y_b` makes columns `a` and `b` equal, so both go
//! (one comes back as `z` when `|y|` is odd): each step removes at least
//! one T gate.
//!
//! Kernel computation: the system in `(y, w)` is `A y = 0`,
//! `(A_α ∧ A_β)·y + z_α w_β + z_β w_α = 0` for `α < β`. The `y` part does
//! not depend on `z`, so it is row-reduced once per step with the `w`
//! coefficients of every unit `z = e_i` carried along; each candidate `z`
//! then costs one small `d`-column elimination instead of a full one
//! (the idea behind FastTODD, Vandaele et al. arXiv:2407.08695).

use super::gf2::{kernel, rref, Bits};
use rand::seq::{IndexedRandom, SliceRandom};
use rand::Rng;
use std::collections::HashMap;

/// Removes zero columns and pairs of equal columns (neither changes the
/// signature tensor). The result is sorted.
pub fn clean(cols: Vec<Bits>) -> Vec<Bits> {
    let mut v: Vec<Bits> = cols.into_iter().filter(|c| !c.is_zero()).collect();
    v.sort_unstable();
    let mut out = Vec::with_capacity(v.len());
    let mut i = 0;
    while i < v.len() {
        let mut j = i + 1;
        while j < v.len() && v[j] == v[i] {
            j += 1;
        }
        if (j - i) % 2 == 1 {
            out.push(v[i].clone());
        }
        i = j;
    }
    out
}

/// The nonzero entries `(α ≤ β ≤ γ)` of the signature tensor of `cols`
/// (vectors in `F_2^d`), sorted. `O(m d³)`: for tests and checks.
pub fn signature(cols: &[Bits], d: usize) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    for a in 0..d {
        for b in a..d {
            for c in b..d {
                let s = cols
                    .iter()
                    .filter(|p| p.get(a) && p.get(b) && p.get(c))
                    .count();
                if s % 2 == 1 {
                    out.push((a, b, c));
                }
            }
        }
    }
    out
}

/// The diagonal Clifford `D` with `Σ_old (p·u) = Σ_new (p·u) + D(u)
/// (mod 8)` for two column lists with the same signature tensor, as
/// parity terms `(q, k)` meaning `ω^{k·(q·u)}` with `k` even.
///
/// Returns `None` if the linear or quadratic parts show that the
/// signature tensors differ (the cubic part is not checked here; the
/// whole-circuit path-sum check covers it).
pub fn clifford_correction(old: &[Bits], new: &[Bits], d: usize) -> Option<Vec<(Bits, u8)>> {
    let mut lin = vec![0i64; d];
    let mut quad = vec![0i64; d * d];
    for (cols, sign) in [(old, 1i64), (new, -1i64)] {
        for p in cols {
            let ones: Vec<usize> = p.ones().collect();
            for (i, &a) in ones.iter().enumerate() {
                lin[a] += sign;
                for &b in &ones[i + 1..] {
                    quad[a * d + b] += sign;
                }
            }
        }
    }
    let mut terms = Vec::new();
    for a in 0..d {
        let l = lin[a].rem_euclid(8);
        if l % 2 != 0 {
            return None;
        }
        if l != 0 {
            terms.push((Bits::unit(d, a), l as u8));
        }
    }
    for a in 0..d {
        for b in a + 1..d {
            // coefficient of u_a u_b in Σ (p·u) is -2·(count) (mod 8)
            let q = (-2 * quad[a * d + b]).rem_euclid(8);
            match q {
                0 => {}
                4 => {
                    // 4 u_a u_b = 2 u_a + 2 u_b - 2 (u_a ⊕ u_b)
                    let mut ab = Bits::unit(d, a);
                    ab.set(b, true);
                    terms.push((Bits::unit(d, a), 2));
                    terms.push((Bits::unit(d, b), 2));
                    terms.push((ab, 6));
                }
                _ => return None,
            }
        }
    }
    Some(terms)
}

/// Options for [`todd`].
#[derive(Clone, Copy, Debug)]
pub struct ToddParams {
    /// Visit column pairs and kernel vectors in random order (for
    /// randomised restarts); otherwise deterministic.
    pub randomize: bool,
    /// Upper bound on the number of reduction steps (`usize::MAX`: none).
    pub max_steps: usize,
}

impl Default for ToddParams {
    fn default() -> Self {
        ToddParams {
            randomize: false,
            max_steps: usize::MAX,
        }
    }
}

/// Reduces the number of columns of `cols` (vectors in `F_2^d`) while
/// keeping their signature tensor, until no TODD step applies.
pub fn todd<R: Rng>(cols: Vec<Bits>, d: usize, params: &ToddParams, rng: &mut R) -> Vec<Bits> {
    let mut cols = clean(cols);
    let mut steps = 0;
    while steps < params.max_steps {
        if params.randomize {
            cols.shuffle(rng);
        }
        let Some((z, y)) = find_reduction(&cols, d, params.randomize, rng) else {
            break;
        };
        let before = cols.len();
        apply(&mut cols, &z, &y);
        debug_assert!(cols.len() < before, "a TODD step must remove a column");
        steps += 1;
    }
    cols
}

/// `A ← A + z yᵀ`, appending `z` when `|y|` is odd, then [`clean`].
fn apply(cols: &mut Vec<Bits>, z: &Bits, y: &Bits) {
    let m = cols.len();
    let mut odd = false;
    for j in y.ones() {
        if j < m {
            cols[j].xor_with(z);
            odd = !odd;
        }
    }
    if odd {
        cols.push(z.clone());
    }
    *cols = clean(std::mem::take(cols));
}

/// The row-reduced system of one TODD step.
struct System {
    m: usize,
    d: usize,
    /// Words of the `y` part of a row.
    wm: usize,
    /// Words of one `w` block.
    wd: usize,
    rows: Vec<Bits>,
    pivots: Vec<usize>,
    /// Basis of the `z`-independent kernel `K0 = {y: Ay = 0, B(y) = 0}`.
    k0: Vec<Bits>,
}

impl System {
    fn build(cols: &[Bits], d: usize) -> Self {
        let m = cols.len();
        let wm = m.div_ceil(64).max(1);
        let wd = d.div_ceil(64).max(1);
        let width = wm + d * wd;
        let mut lin: Vec<Vec<u64>> = vec![vec![0u64; wm]; d];
        for (j, c) in cols.iter().enumerate() {
            for a in c.ones() {
                lin[a][j >> 6] |= 1u64 << (j & 63);
            }
        }
        let mut rows: Vec<Bits> = Vec::with_capacity(d + d * (d.saturating_sub(1)) / 2);
        for row in lin.iter() {
            let mut r = vec![0u64; width];
            r[..wm].copy_from_slice(row);
            rows.push(Bits(r));
        }
        for a in 0..d {
            for b in a + 1..d {
                let mut r = vec![0u64; width];
                for k in 0..wm {
                    r[k] = lin[a][k] & lin[b][k];
                }
                // z_a w_b (block a, bit b) + z_b w_a (block b, bit a)
                let off_a = wm + a * wd;
                let off_b = wm + b * wd;
                r[off_a + (b >> 6)] |= 1u64 << (b & 63);
                r[off_b + (a >> 6)] |= 1u64 << (a & 63);
                rows.push(Bits(r));
            }
        }
        let pivots = rref(&mut rows, m);
        let mut is_pivot = vec![false; m];
        for &p in &pivots {
            is_pivot[p] = true;
        }
        let mut k0 = Vec::new();
        for f in (0..m).filter(|&f| !is_pivot[f]) {
            let mut y = Bits::unit(m, f);
            for (r, &p) in pivots.iter().enumerate() {
                if rows[r].get(f) {
                    y.set(p, true);
                }
            }
            k0.push(y);
        }
        System {
            m,
            d,
            wm,
            wd,
            rows,
            pivots,
            k0,
        }
    }

    /// `Σ_{i ∈ supp z}` of the `w` blocks of row `r`.
    fn block_sum(&self, r: usize, z: &Bits) -> Bits {
        let mut out = Bits(vec![0u64; self.wd]);
        let row = &self.rows[r].0;
        for i in z.ones() {
            let off = self.wm + i * self.wd;
            for k in 0..self.wd {
                out.0[k] ^= row[off + k];
            }
        }
        out
    }

    /// The `y` vectors spanning the solutions for this `z` beyond `K0`.
    fn z_solutions(&self, z: &Bits) -> Vec<Bits> {
        let rank = self.pivots.len();
        let mut g: Vec<Bits> = (rank..self.rows.len())
            .map(|r| self.block_sum(r, z))
            .filter(|b| !b.is_zero())
            .collect();
        let ws = kernel(&mut g, self.d);
        if ws.is_empty() {
            return Vec::new();
        }
        let tops: Vec<Bits> = (0..rank).map(|r| self.block_sum(r, z)).collect();
        ws.iter()
            .map(|w| {
                let mut y = Bits::zeros(self.m);
                for (r, t) in tops.iter().enumerate() {
                    if t.dot(w) {
                        y.set(self.pivots[r], true);
                    }
                }
                y
            })
            .filter(|y| !y.is_zero())
            .collect()
    }
}

/// From a family spanning the admissible `y`s, one with `y_a ≠ y_b`,
/// preferring even weight.
fn pick<R: Rng>(
    family: &[&Bits],
    a: usize,
    b: usize,
    randomize: bool,
    rng: &mut R,
) -> Option<Bits> {
    let split: Vec<&Bits> = family
        .iter()
        .copied()
        .filter(|y| y.get(a) != y.get(b))
        .collect();
    if split.is_empty() {
        return None;
    }
    let mut y = if randomize {
        (*split.choose(rng).expect("nonempty")).clone()
    } else {
        split[0].clone()
    };
    if randomize {
        // a random element of the affine space {y : y_a ≠ y_b}
        for f in family {
            if rng.random_bool(0.5) && f.get(a) == f.get(b) {
                y.xor_with(f);
            }
        }
    }
    if y.count_ones() % 2 == 1 {
        if let Some(f) = family
            .iter()
            .find(|f| f.get(a) == f.get(b) && f.count_ones() % 2 == 1)
        {
            y.xor_with(f);
        }
    }
    Some(y)
}

/// One TODD step: `(z, y)` such that `A + z yᵀ` (with a zero column
/// appended when `|y|` is odd) has the same signature tensor and two
/// equal columns.
fn find_reduction<R: Rng>(
    cols: &[Bits],
    d: usize,
    randomize: bool,
    rng: &mut R,
) -> Option<(Bits, Bits)> {
    let m = cols.len();
    if m < 2 {
        return None;
    }
    let sys = System::build(cols, d);
    // Fast path: a z-independent kernel vector that is not constant.
    let mut k0_order: Vec<usize> = (0..sys.k0.len()).collect();
    if randomize {
        k0_order.shuffle(rng);
    }
    for &i in &k0_order {
        let k = &sys.k0[i];
        let w = k.count_ones();
        if w == 0 || w == m {
            continue;
        }
        let ones: Vec<usize> = k.ones().collect();
        let zeros: Vec<usize> = (0..m).filter(|&j| !k.get(j)).collect();
        let (a, b) = if randomize {
            (*ones.choose(rng).unwrap(), *zeros.choose(rng).unwrap())
        } else {
            (ones[0], zeros[0])
        };
        let family: Vec<&Bits> = sys.k0.iter().collect();
        let y = pick(&family, a, b, randomize, rng)?;
        let mut z = cols[a].clone();
        z.xor_with(&cols[b]);
        return Some((z, y));
    }
    // General path: z = p_a + p_b for every pair.
    let mut pairs: Vec<(usize, usize)> = (0..m)
        .flat_map(|a| (a + 1..m).map(move |b| (a, b)))
        .collect();
    if randomize {
        pairs.shuffle(rng);
    }
    let mut cache: HashMap<Bits, Vec<Bits>> = HashMap::new();
    for (a, b) in pairs {
        let mut z = cols[a].clone();
        z.xor_with(&cols[b]);
        if z.is_zero() {
            continue;
        }
        let sols = cache
            .entry(z.clone())
            .or_insert_with(|| sys.z_solutions(&z));
        if sols.is_empty() {
            continue;
        }
        let family: Vec<&Bits> = sys.k0.iter().chain(sols.iter()).collect();
        if let Some(y) = pick(&family, a, b, randomize, rng) {
            return Some((z, y));
        }
    }
    None
}

/// A random signature-preserving update `A → A + z yᵀ` that need not
/// lower the column count (it may raise it by one): `z` is the sum of two
/// random columns, `y` a random nonzero element of the admissible space.
/// Used to leave a TODD fixed point before reducing again.
pub fn random_neutral_move<R: Rng>(cols: &[Bits], d: usize, rng: &mut R) -> Option<Vec<Bits>> {
    let m = cols.len();
    if m < 2 {
        return None;
    }
    let sys = System::build(cols, d);
    for _ in 0..8 {
        let a = rng.random_range(0..m);
        let b = rng.random_range(0..m);
        if a == b {
            continue;
        }
        let mut z = cols[a].clone();
        z.xor_with(&cols[b]);
        if z.is_zero() {
            continue;
        }
        let sols = sys.z_solutions(&z);
        let family: Vec<&Bits> = sys.k0.iter().chain(sols.iter()).collect();
        if family.is_empty() {
            continue;
        }
        let mut y = Bits::zeros(m);
        while y.is_zero() {
            for f in &family {
                if rng.random_bool(0.5) {
                    y.xor_with(f);
                }
            }
        }
        let mut out = cols.to_vec();
        apply(&mut out, &z, &y);
        return Some(out);
    }
    None
}

/// Large-neighbourhood search: from `start` (already TODD-reduced),
/// repeatedly apply one to three random neutral moves and a randomised
/// TODD, keeping the result when it has no more columns (ties accepted, so
/// the search drifts across plateaus). Returns the smallest list seen.
pub fn lns<R: Rng>(start: Vec<Bits>, d: usize, rounds: usize, rng: &mut R) -> Vec<Bits> {
    let mut cur = start;
    let mut best = cur.clone();
    let params = ToddParams {
        randomize: true,
        ..Default::default()
    };
    for _ in 0..rounds {
        let mut cand = cur.clone();
        for _ in 0..rng.random_range(1..=3) {
            if let Some(next) = random_neutral_move(&cand, d, rng) {
                cand = next;
            }
        }
        let cand = todd(cand, d, &params, rng);
        if cand.len() <= cur.len() {
            cur = cand;
            if cur.len() < best.len() {
                best = cur.clone();
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn col(d: usize, bits: &[usize]) -> Bits {
        let mut b = Bits::zeros(d);
        for &i in bits {
            b.set(i, true);
        }
        b
    }

    #[test]
    fn ccz_is_irreducible_and_doubled_ccz_vanishes() {
        // CCZ on 3 qubits: the 7 nonzero parities.
        let d = 3;
        let ccz: Vec<Bits> = (1u64..8).map(|v| Bits(vec![v])).collect();
        let mut rng = StdRng::seed_from_u64(1);
        let out = todd(ccz.clone(), d, &ToddParams::default(), &mut rng);
        assert_eq!(out.len(), 7);
        assert_eq!(signature(&out, d), signature(&ccz, d));
        // two CCZ on the same qubits: identity
        let mut two = ccz.clone();
        two.extend(ccz.clone());
        assert!(clean(two).is_empty());
    }

    #[test]
    fn todd_keeps_signature_on_random_inputs() {
        let mut rng = StdRng::seed_from_u64(7);
        for trial in 0..40 {
            let d = 3 + trial % 5;
            let m = 5 + trial % 17;
            let cols: Vec<Bits> = (0..m)
                .map(|_| {
                    let mut b = Bits::zeros(d);
                    for i in 0..d {
                        if rng.random_bool(0.5) {
                            b.set(i, true);
                        }
                    }
                    b
                })
                .collect();
            let params = ToddParams {
                randomize: trial % 2 == 1,
                ..Default::default()
            };
            let out = todd(cols.clone(), d, &params, &mut rng);
            assert!(out.len() <= clean(cols.clone()).len());
            assert_eq!(signature(&out, d), signature(&cols, d), "trial {trial}");
            let corr = clifford_correction(&cols, &out, d).expect("same signature");
            assert!(corr.iter().all(|(_, k)| k % 2 == 0));
        }
    }

    #[test]
    fn lns_keeps_signature() {
        let mut rng = StdRng::seed_from_u64(11);
        for trial in 0..10 {
            let d = 4 + trial % 3;
            let cols: Vec<Bits> = (0..12)
                .map(|_| Bits(vec![rng.random_range(1..(1u64 << d))]))
                .collect();
            let start = todd(cols.clone(), d, &ToddParams::default(), &mut rng);
            let out = lns(start.clone(), d, 20, &mut rng);
            assert!(out.len() <= start.len());
            assert_eq!(signature(&out, d), signature(&cols, d));
        }
    }

    #[test]
    fn two_overlapping_toffolis_reduce() {
        // Two CCZ sharing two qubits: CCZ(0,1,2)·CCZ(0,1,3) = CCZ on
        // (0,1, 2⊕3) up to Clifford: 14 -> 7 terms.
        let d = 4;
        let mut cols = Vec::new();
        for t in [2usize, 3] {
            for s in 1u64..8 {
                let mut v = Vec::new();
                if s & 1 != 0 {
                    v.push(0);
                }
                if s & 2 != 0 {
                    v.push(1);
                }
                if s & 4 != 0 {
                    v.push(t);
                }
                cols.push(col(d, &v));
            }
        }
        let mut rng = StdRng::seed_from_u64(3);
        let out = todd(cols.clone(), d, &ToddParams::default(), &mut rng);
        assert_eq!(signature(&out, d), signature(&cols, d));
        assert!(out.len() <= 7, "got {}", out.len());
    }
}
