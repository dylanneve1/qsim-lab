// chi(|H>^{⊗5}) >= 6 (and the n = 4 / F^{⊗5} validation runs): complete search for k-term
// stabilizer decompositions of psi^{⊗n} one step past chi(psi^{⊗(n-1)}).
// Standalone, std only.  Build from this directory:
//     rustc -O -C target-cpu=native stabrank5.rs -o stabrank5
// It includes ../stabrank-lower/stabrank.rs as a module (stabilizer-state table, symmetry group,
// least squares, and the older search / glue / degenerate-search code used for cross-checks).
// Write-up: research/theory/stabrank5.md.
//
// Ingredients (proofs in the write-up):
//  * Galois lemma.  If psi^{⊗n} = sum_i c_i phi_i with linearly independent stabilizer states phi_i
//    and psi in {H, F}, then span(phi_i) also contains (psi^perp)^{⊗n}.  So the k-term minimal
//    decompositions are the k-dimensional spans of stabilizer states containing the 2-dimensional
//    space U = span(psi^{⊗n}, psi^perp^{⊗n}).
//  * `gsearch` enumerates all independent k-sets (k = 4, 5) of stabilizer states whose span
//    contains U, up to the symmetry group G: fix k-3 terms (the first is the orbit representative
//    of the minimal orbit among the terms; for k = 5 the second is independent of it modulo U and
//    minimal in its Stab(phi_1)-orbit); the remaining three then project to a common line (or to 0)
//    modulo W = U + span(fixed terms), which is found by a projective hash.
//  * Completions.  (|0>A + |1>z)/sqrt2 is a stabilizer state iff z = w X^v Z^u A with w in
//    {±1, ±i} (controlled-Pauli argument).  `bottoms(A)` lists 0 (the product term |0>A) and these.
//  * Lift (Proposition 7 of stabrank-lower.md).  Restrict the last qubit by <0|.  The restricted
//    k-tuple of a k-term decomposition of psi^{⊗n}, k = chi(psi^{⊗(n-1)}) + 1, is either
//      I   a minimal k-term decomposition A of psi^{⊗(n-1)}: then sum_i a_i z_i = t psi^{⊗(n-1)},
//          t = psi_1/psi_0, z_i in bottoms(A_i) -- meet in the middle on two random functionals;
//      II  an optimal (k-1)-term decomposition D plus a zero: t psi - sum d_i z_i ∝ a stabilizer
//          state -- brute force over bottoms with a support/magnitude filter;
//      IIIa D plus a term parallel to D_j: sum d_i z_i - t psi = s (z_j - z_tau) -- meet in the
//          middle on the 2x2-determinant scalar condition;
//      IIIb D plus a stabilizer state tau in span(D) parallel to no D_j -- brute force on scalars.
//    D and A range over orbit representatives under G_{n-1} (every element extends to a symmetry
//    of psi^{⊗n} that fixes the <0| restriction of the last qubit).
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

#[allow(dead_code, unused_imports, unused_variables, unused_mut, clippy::all)]
#[path = "../stabrank-lower/stabrank.rs"]
pub mod stabrank;
pub use stabrank::*;

// ---------------------------------------------------------------- small helpers
pub fn cj(z: C) -> C {
    C::new(z.re, -z.im)
}
pub fn csc(z: C, s: f64) -> C {
    C::new(z.re * s, z.im * s)
}
pub fn cdiv(a: C, b: C) -> C {
    let d = b.n2();
    let p = a * cj(b);
    C::new(p.re / d, p.im / d)
}
pub fn nthreads() -> usize {
    std::env::var("THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4)
}
/// The orthogonal single-qubit state psi^perp = (-conj psi_1, conj psi_0).
pub fn perp1(p: &[C; 2]) -> [C; 2] {
    [-cj(p[1]), cj(p[0])]
}
fn xorshift(seed: &mut u64) -> f64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    (*seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}
pub fn random_functional(dim: usize, seed: u64) -> Vec<C> {
    let mut s = seed | 1;
    (0..dim)
        .map(|_| {
            let a = xorshift(&mut s);
            C::new(a, xorshift(&mut s))
        })
        .collect()
}

// ---------------------------------------------------------------- symmetry helpers
/// Orbit representative (smallest index) per orbit id.
pub fn orbit_reps_idx(orbit: &[u32]) -> Vec<u32> {
    let no = *orbit.iter().max().unwrap() as usize + 1;
    let mut rep = vec![u32::MAX; no];
    for (i, &o) in orbit.iter().enumerate() {
        if rep[o as usize] == u32::MAX {
            rep[o as usize] = i as u32;
        }
    }
    rep
}
/// Canonical form of a set of state indices under the explicit group: lexicographically least
/// sorted image.  Also returns the size of the set-stabilizer.
pub fn canon_set(s: &[u32], group: &[Vec<u16>]) -> (Vec<u32>, usize) {
    let mut best: Vec<u32> = vec![u32::MAX; s.len()];
    let mut sorted = s.to_vec();
    sorted.sort();
    let mut stab = 0;
    let mut im = vec![0u32; s.len()];
    for g in group {
        for (j, &x) in s.iter().enumerate() {
            im[j] = g[x as usize] as u32;
        }
        im.sort();
        if im == sorted {
            stab += 1;
        }
        if im < best {
            best.copy_from_slice(&im);
        }
    }
    (best, stab)
}

/// Dedupe found sets to one canonical representative per G-orbit; returns (reps, total count of
/// sets = sum of orbit sizes).
pub fn dedupe_orbits(found: &[Vec<u32>], group: &[Vec<u16>]) -> (Vec<Vec<u32>>, usize) {
    let mut seen: HashMap<Vec<u32>, usize> = HashMap::new();
    for s in found {
        let (c, st) = canon_set(s, group);
        seen.entry(c).or_insert(group.len() / st);
    }
    let mut reps: Vec<(Vec<u32>, usize)> = seen.into_iter().collect();
    reps.sort();
    let total = reps.iter().map(|r| r.1).sum();
    (reps.into_iter().map(|r| r.0).collect(), total)
}

// ---------------------------------------------------------------- Galois-pair search
pub struct GStats {
    pub pairs: u64,
    pub cand: u64,
    pub verified: u64,
}

/// All linearly independent k-sets (k = 4 or 5) of stabilizer states in `t` whose span contains
/// both `psi` and `psip` (normalised, orthogonal), with every coefficient of `psi` non-zero.
/// Returns the sets (sorted indices) found; every G-orbit of such sets is hit at least once.
pub fn gsearch(
    t: &Table,
    psi: &[C],
    psip: &[C],
    k: usize,
    orbit: &[u32],
    group: &[Vec<u16>],
    st: &mut GStats,
) -> Vec<Vec<u32>> {
    assert!(k == 4 || k == 5);
    let nst = t.states.len();
    let dim = psi.len();
    let rep = orbit_reps_idx(orbit);
    let norbits = rep.len();
    let q0: Vec<C> = psi.to_vec();
    let q1: Vec<C> = psip.to_vec();
    assert!((norm2(&q0) - 1.0).abs() < 1e-12 && (norm2(&q1) - 1.0).abs() < 1e-12);
    assert!(dot(&q0, &q1).abs() < 1e-12);
    let ip0: Vec<C> = t.states.iter().map(|v| dot(&q0, v)).collect();
    let ip1: Vec<C> = t.states.iter().map(|v| dot(&q1, v)).collect();
    // no stabilizer state lies in U (needed: phi_1 is independent of U)
    for v in 0..nst {
        assert!(1.0 - ip0[v].n2() - ip1[v].n2() > 1e-6, "stabilizer state in U");
    }
    let f1 = random_functional(dim, 0x9E3779B97F4A7C15);
    let f2 = random_functional(dim, 0x2545F4914F6CDD1D);
    let fv1: Vec<C> = t.states.iter().map(|v| dot(&f1, v)).collect();
    let fv2: Vec<C> = t.states.iter().map(|v| dot(&f2, v)).collect();
    let stabs: Vec<Vec<usize>> = (0..norbits)
        .map(|o| {
            (0..group.len())
                .filter(|&e| group[e][rep[o] as usize] as u32 == rep[o])
                .collect()
        })
        .collect();
    let next = AtomicUsize::new(0);
    let out = Mutex::new((Vec::<Vec<u32>>::new(), 0u64, 0u64, 0u64));
    std::thread::scope(|sc| {
        for _ in 0..nthreads() {
            sc.spawn(|| {
                let mut lfound: Vec<Vec<u32>> = vec![];
                let mut lseen: std::collections::HashSet<Vec<u32>> = Default::default();
                let (mut lpairs, mut lcand, mut lver) = (0u64, 0u64, 0u64);
                loop {
                    let o = next.fetch_add(1, Ordering::SeqCst);
                    if o >= norbits {
                        break;
                    }
                    let r = rep[o] as usize;
                    let cands: Vec<usize> = (0..nst).filter(|&v| orbit[v] as usize >= o).collect();
                    // q2 from r
                    let mut q2 = t.states[r].clone();
                    for x in 0..dim {
                        q2[x] = q2[x] - ip0[r] * q0[x] - ip1[r] * q1[x];
                    }
                    let n2 = norm2(&q2).sqrt();
                    for z in q2.iter_mut() {
                        *z = csc(*z, 1.0 / n2);
                    }
                    let mut ip2 = vec![C::default(); nst];
                    for &v in &cands {
                        ip2[v] = dot(&q2, &t.states[v]);
                    }
                    if k == 4 {
                        let qs = [&q0, &q1, &q2];
                        let ips = [&ip0, &ip1, &ip2];
                        lpairs += 1;
                        hash_step(
                            t, psi, psip, &[r as u32], &qs, &ips, &cands, &f1, &f2, &fv1, &fv2,
                            &mut lfound, &mut lcand, &mut lver,
                        );
                        for s in lfound.drain(..) {
                            lseen.insert(s);
                        }
                        continue;
                    }
                    // k == 5: second fixed term
                    let mut ip3 = vec![C::default(); nst];
                    for &v2 in &cands {
                        if v2 == r {
                            continue;
                        }
                        if stabs[o]
                            .iter()
                            .any(|&e| (group[e][v2] as usize) < v2)
                        {
                            continue;
                        }
                        let c0 = ip0[v2];
                        let c1 = ip1[v2];
                        let c2 = ip2[v2];
                        let rn2 = 1.0 - c0.n2() - c1.n2() - c2.n2();
                        if rn2 < 1e-9 {
                            continue; // v2 in U + C r: not independent modulo U
                        }
                        let mut q3 = t.states[v2].clone();
                        for x in 0..dim {
                            q3[x] = q3[x] - c0 * q0[x] - c1 * q1[x] - c2 * q2[x];
                        }
                        let n3 = norm2(&q3).sqrt();
                        for z in q3.iter_mut() {
                            *z = csc(*z, 1.0 / n3);
                        }
                        for &v in &cands {
                            ip3[v] = dot(&q3, &t.states[v]);
                        }
                        lpairs += 1;
                        let qs = [&q0, &q1, &q2, &q3];
                        let ips = [&ip0, &ip1, &ip2, &ip3];
                        hash_step(
                            t,
                            psi,
                            psip,
                            &[r as u32, v2 as u32],
                            &qs,
                            &ips,
                            &cands,
                            &f1,
                            &f2,
                            &fv1,
                            &fv2,
                            &mut lfound,
                            &mut lcand,
                            &mut lver,
                        );
                        for s in lfound.drain(..) {
                            lseen.insert(s);
                        }
                    }
                    if std::env::var("PROGRESS").is_ok() {
                        eprintln!("orbit {}/{} distinct found so far {}", o, norbits, lseen.len());
                    }
                }
                let mut g = out.lock().unwrap();
                g.0.extend(lseen);
                g.1 += lpairs;
                g.2 += lcand;
                g.3 += lver;
            });
        }
    });
    let g = out.into_inner().unwrap();
    st.pairs += g.1;
    st.cand += g.2;
    st.verified += g.3;
    g.0
}

/// Given the fixed terms (whose span with U is W, orthonormal basis `qs`), find all triples of
/// further candidates whose projections onto W^perp are pairwise parallel or zero (at least one
/// non-zero), and keep the sets that are decompositions.
#[allow(clippy::too_many_arguments)]
fn hash_step(
    t: &Table,
    psi: &[C],
    psip: &[C],
    fixed: &[u32],
    qs: &[&Vec<C>],
    ips: &[&Vec<C>],
    cands: &[usize],
    f1: &[C],
    f2: &[C],
    fv1: &[C],
    fv2: &[C],
    found: &mut Vec<Vec<u32>>,
    ncand: &mut u64,
    nver: &mut u64,
) {
    let fq1: Vec<C> = qs.iter().map(|q| dot(f1, q)).collect();
    let fq2: Vec<C> = qs.iter().map(|q| dot(f2, q)).collect();
    let mut zero: Vec<u32> = vec![];
    let mut pts: Vec<(f64, f64, f64, u32)> = Vec::with_capacity(cands.len());
    for &v in cands {
        if fixed.contains(&(v as u32)) {
            continue;
        }
        *ncand += 1;
        let mut pn2 = 1.0;
        let mut a = fv1[v];
        let mut b = fv2[v];
        for j in 0..qs.len() {
            let c = ips[j][v];
            pn2 -= c.n2();
            a = a - fq1[j] * c;
            b = b - fq2[j] * c;
        }
        if pn2 < 1e-9 {
            zero.push(v as u32);
            continue;
        }
        let s = a.n2() + b.n2();
        let ab = a * cj(b);
        pts.push((2.0 * ab.re / s, 2.0 * ab.im / s, (a.n2() - b.n2()) / s, v as u32));
    }
    pts.sort_unstable_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    // union-find of near-coincident projective points
    let np = pts.len();
    let mut parent: Vec<usize> = (0..np).collect();
    fn find(p: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while p[r] != r {
            r = p[r];
        }
        let mut y = x;
        while p[y] != r {
            let nx = p[y];
            p[y] = r;
            y = nx;
        }
        r
    }
    let eps = 1e-6;
    let mut paired = vec![false; np];
    for i in 0..np {
        let mut j = i + 1;
        while j < np && pts[j].0 - pts[i].0 < eps {
            if (pts[j].1 - pts[i].1).abs() + (pts[j].2 - pts[i].2).abs() < 2.0 * eps {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                if a != b {
                    parent[a.max(b)] = a.min(b);
                }
                paired[i] = true;
                paired[j] = true;
            }
            j += 1;
        }
    }
    // A valid triple is: 3 parallel non-zero projections, or 2 parallel + 1 zero, or 1 non-zero +
    // 2 zero (all three zero would put 5 terms in the 4-dimensional W).
    let z = &zero;
    let try_set = |extra: &[u32], found: &mut Vec<Vec<u32>>, nver: &mut u64| {
        let mut s: Vec<u32> = fixed.to_vec();
        s.extend_from_slice(extra);
        *nver += 1;
        if let Some(d) = check_set(t, psi, &s) {
            // Galois lemma sanity: psip is in the span too
            let vs: Vec<&[C]> = d.idx.iter().map(|&i| t.states[i as usize].as_slice()).collect();
            let (_, res) = lsq(&vs, psip);
            assert!(res < 1e-7, "Galois lemma violated?");
            found.push(d.idx);
        }
    };
    // classes with >= 2 members
    let mut classes: HashMap<usize, Vec<u32>> = HashMap::new();
    for i in 0..np {
        if paired[i] {
            let r = find(&mut parent, i);
            classes.entry(r).or_default().push(pts[i].3);
        }
    }
    for cl in classes.values() {
        let m = cl.len();
        for a in 0..m {
            for b in (a + 1)..m {
                for c in (b + 1)..m {
                    try_set(&[cl[a], cl[b], cl[c]], found, nver);
                }
                for &zz in z.iter() {
                    try_set(&[cl[a], cl[b], zz], found, nver);
                }
            }
        }
    }
    // one non-zero projection (any point) + two zero projections
    if z.len() >= 2 {
        for p in &pts {
            for i in 0..z.len() {
                for j in (i + 1)..z.len() {
                    try_set(&[p.3, z[i], z[j]], found, nver);
                }
            }
        }
    }
}

/// Projective points (on the Riemann sphere) of the projections of `cands` onto W^perp, where W
/// has orthonormal basis `qs`; returns (zero set, points sorted by first coordinate).
#[allow(clippy::too_many_arguments)]
fn project_points(
    qs: &[&Vec<C>],
    ips: &[&Vec<C>],
    cands: &[usize],
    f1: &[C],
    f2: &[C],
    fv1: &[C],
    fv2: &[C],
    dg: &mut Diag,
) -> (Vec<u32>, Vec<(f64, f64, f64, u32)>) {
    let fq1: Vec<C> = qs.iter().map(|q| dot(f1, q)).collect();
    let fq2: Vec<C> = qs.iter().map(|q| dot(f2, q)).collect();
    let mut zero = vec![];
    let mut pts = Vec::with_capacity(cands.len());
    for &v in cands {
        let mut pn2 = 1.0;
        let mut a = fv1[v];
        let mut b = fv2[v];
        for j in 0..qs.len() {
            let c = ips[j][v];
            pn2 -= c.n2();
            a = a - fq1[j] * c;
            b = b - fq2[j] * c;
        }
        if pn2 < 1e-9 {
            dg.zero_pn2_max = dg.zero_pn2_max.max(pn2);
            zero.push(v as u32);
            continue;
        }
        dg.nonzero_pn2_min = dg.nonzero_pn2_min.min(pn2);
        let s = a.n2() + b.n2();
        let ab = a * cj(b);
        pts.push((2.0 * ab.re / s, 2.0 * ab.im / s, (a.n2() - b.n2()) / s, v as u32));
    }
    pts.sort_unstable_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    (zero, pts)
}

/// Classes (size >= 2) of near-coincident projective points.
fn point_classes(pts: &[(f64, f64, f64, u32)]) -> Vec<Vec<u32>> {
    let np = pts.len();
    let mut parent: Vec<usize> = (0..np).collect();
    fn find(p: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while p[r] != r {
            r = p[r];
        }
        let mut y = x;
        while p[y] != r {
            let nx = p[y];
            p[y] = r;
            y = nx;
        }
        r
    }
    let eps = 1e-6;
    let mut paired = vec![false; np];
    for i in 0..np {
        let mut j = i + 1;
        while j < np && pts[j].0 - pts[i].0 < eps {
            if (pts[j].1 - pts[i].1).abs() + (pts[j].2 - pts[i].2).abs() < 2.0 * eps {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                if a != b {
                    parent[a.max(b)] = a.min(b);
                }
                paired[i] = true;
                paired[j] = true;
            }
            j += 1;
        }
    }
    let mut classes: HashMap<usize, Vec<u32>> = HashMap::new();
    for i in 0..np {
        if paired[i] {
            let r = find(&mut parent, i);
            classes.entry(r).or_default().push(pts[i].3);
        }
    }
    classes.into_values().collect()
}

/// Margins of the numerical decisions (reported so that the tolerances can be audited).
#[derive(Clone, Debug)]
pub struct Diag {
    pub zero_pn2_max: f64,    // largest |P v|^2 classified as zero (in W)
    pub nonzero_pn2_min: f64, // smallest |P v|^2 classified as non-zero
    pub acc_gram_min: f64,    // smallest Gram determinant among accepted sets
    pub acc_coef_min: f64,    // smallest |coefficient| among accepted sets
    pub acc_res_max: f64,     // largest residual among accepted sets
    pub ambiguous: u64,       // rejected sets with residual < 1e-6 and (Gram < 1e-6 or min|c| < 1e-6)
    pub rej_res_min: f64,     // smallest residual among sets rejected only for the residual
}
impl Diag {
    pub fn new() -> Diag {
        Diag {
            zero_pn2_max: 0.0,
            nonzero_pn2_min: f64::MAX,
            acc_gram_min: f64::MAX,
            acc_coef_min: f64::MAX,
            acc_res_max: 0.0,
            ambiguous: 0,
            rej_res_min: f64::MAX,
        }
    }
    pub fn merge(&mut self, o: &Diag) {
        self.zero_pn2_max = self.zero_pn2_max.max(o.zero_pn2_max);
        self.nonzero_pn2_min = self.nonzero_pn2_min.min(o.nonzero_pn2_min);
        self.acc_gram_min = self.acc_gram_min.min(o.acc_gram_min);
        self.acc_coef_min = self.acc_coef_min.min(o.acc_coef_min);
        self.acc_res_max = self.acc_res_max.max(o.acc_res_max);
        self.ambiguous += o.ambiguous;
        self.rej_res_min = self.rej_res_min.min(o.rej_res_min);
    }
}
/// check_set with margin bookkeeping: same accept rule (Gram > 1e-10, residual < 1e-8, all
/// |c| > 1e-8, distinct indices).
pub fn check_set_diag(t: &Table, psi: &[C], set: &[u32], dg: &mut Diag) -> Option<Dec> {
    let mut s = set.to_vec();
    s.sort();
    s.dedup();
    if s.len() != set.len() {
        return None;
    }
    let vs: Vec<&[C]> = s.iter().map(|&i| t.states[i as usize].as_slice()).collect();
    let g = gram_det(&vs);
    let (c, res) = if g > 1e-14 { lsq(&vs, psi) } else { (vec![], f64::INFINITY) };
    let cmin = c.iter().map(|z| z.abs()).fold(f64::MAX, f64::min);
    if g > 1e-10 && res < 1e-8 && cmin > 1e-8 {
        dg.acc_gram_min = dg.acc_gram_min.min(g);
        dg.acc_coef_min = dg.acc_coef_min.min(cmin);
        dg.acc_res_max = dg.acc_res_max.max(res);
        return Some(Dec { idx: s, coef: c });
    }
    if res < 1e-6 && (g < 1e-6 || cmin < 1e-6) && g > 1e-14 && cmin > 1e-14 {
        dg.ambiguous += 1;
    }
    if g > 1e-6 && cmin > 1e-6 && res >= 1e-8 {
        dg.rej_res_min = dg.rej_res_min.min(res);
    }
    None
}

/// Refined k = 5 Galois search (same output as `gsearch(.., 5, ..)` up to repeats).  phi_2 is the
/// minimal-orbit term among those independent of phi_1 modulo U.  Terms in W = U + span(phi_1,
/// phi_2) are exactly Z1 (states in U + C phi_1) plus the level-1 projective class of phi_2, so only
/// candidates with orbit >= orbit(phi_2) need the level-2 hash.
pub fn gsearch5r(
    t: &Table,
    psi: &[C],
    psip: &[C],
    orbit: &[u32],
    group: &[Vec<u16>],
    st: &mut GStats,
) -> Vec<Vec<u32>> {
    let nst = t.states.len();
    let dim = psi.len();
    let rep = orbit_reps_idx(orbit);
    let norbits = rep.len();
    let q0: Vec<C> = psi.to_vec();
    let q1: Vec<C> = psip.to_vec();
    let ip0: Vec<C> = t.states.iter().map(|v| dot(&q0, v)).collect();
    let ip1: Vec<C> = t.states.iter().map(|v| dot(&q1, v)).collect();
    for v in 0..nst {
        assert!(1.0 - ip0[v].n2() - ip1[v].n2() > 1e-6, "stabilizer state in U");
    }
    let f1 = random_functional(dim, 0x9E3779B97F4A7C15);
    let f2 = random_functional(dim, 0x2545F4914F6CDD1D);
    let fv1: Vec<C> = t.states.iter().map(|v| dot(&f1, v)).collect();
    let fv2: Vec<C> = t.states.iter().map(|v| dot(&f2, v)).collect();
    // work items: (orbit o, chunk of v2 candidates) for load balance
    let next = AtomicUsize::new(0);
    let out = Mutex::new((Vec::<Vec<u32>>::new(), 0u64, 0u64, 0u64, 0u64));
    let diag = Mutex::new(Diag::new());
    std::thread::scope(|sc| {
        for _ in 0..nthreads() {
            sc.spawn(|| {
                let mut lfound: std::collections::HashSet<Vec<u32>> = Default::default();
                let (mut lpairs, mut lcand, mut lver, mut lodd) = (0u64, 0u64, 0u64, 0u64);
                let mut ldiag = Diag::new();
                let mut ip2 = vec![C::default(); nst];
                let mut ip3 = vec![C::default(); nst];
                loop {
                    let o = next.fetch_add(1, Ordering::SeqCst);
                    if o >= norbits {
                        break;
                    }
                    let r = rep[o] as usize;
                    let stab: Vec<usize> = (0..group.len())
                        .filter(|&e| group[e][r] as usize == r)
                        .collect();
                    let cands: Vec<usize> = (0..nst)
                        .filter(|&v| v != r && orbit[v] as usize >= o)
                        .collect();
                    let mut q2 = t.states[r].clone();
                    for x in 0..dim {
                        q2[x] = q2[x] - ip0[r] * q0[x] - ip1[r] * q1[x];
                    }
                    let n2 = norm2(&q2).sqrt();
                    for z in q2.iter_mut() {
                        *z = csc(*z, 1.0 / n2);
                    }
                    for &v in &cands {
                        ip2[v] = dot(&q2, &t.states[v]);
                    }
                    // level 1: Z1 and projective classes
                    let (z1, pts1) = project_points(
                        &[&q0, &q1, &q2],
                        &[&ip0, &ip1, &ip2],
                        &cands,
                        &f1,
                        &f2,
                        &fv1,
                        &fv2,
                        &mut ldiag,
                    );
                    let classes1 = point_classes(&pts1);
                    let mut class_of: HashMap<u32, usize> = HashMap::new();
                    for (ci, cl) in classes1.iter().enumerate() {
                        for &v in cl {
                            class_of.insert(v, ci);
                        }
                    }
                    let z1set: std::collections::HashSet<u32> = z1.iter().copied().collect();
                    for &v2 in &cands {
                        if z1set.contains(&(v2 as u32)) {
                            continue; // not independent of r modulo U
                        }
                        if stab.iter().any(|&e| (group[e][v2] as usize) < v2) {
                            continue;
                        }
                        let o2 = orbit[v2];
                        // zero set Z: Z1 plus the level-1 class of v2
                        let mut zero: Vec<u32> = z1.clone();
                        let mut inclass: std::collections::HashSet<u32> = Default::default();
                        if let Some(&ci) = class_of.get(&(v2 as u32)) {
                            for &w in &classes1[ci] {
                                if w as usize != v2 {
                                    zero.push(w);
                                    inclass.insert(w);
                                }
                            }
                        }
                        let hc: Vec<usize> = cands
                            .iter()
                            .copied()
                            .filter(|&v| {
                                v != v2
                                    && orbit[v] >= o2
                                    && !z1set.contains(&(v as u32))
                                    && !inclass.contains(&(v as u32))
                            })
                            .collect();
                        let c0 = ip0[v2];
                        let c1 = ip1[v2];
                        let c2 = ip2[v2];
                        let mut q3 = t.states[v2].clone();
                        for x in 0..dim {
                            q3[x] = q3[x] - c0 * q0[x] - c1 * q1[x] - c2 * q2[x];
                        }
                        let n3 = norm2(&q3).sqrt();
                        for z in q3.iter_mut() {
                            *z = csc(*z, 1.0 / n3);
                        }
                        for &v in &hc {
                            ip3[v] = dot(&q3, &t.states[v]);
                        }
                        lpairs += 1;
                        lcand += hc.len() as u64;
                        let (zz, pts) = project_points(
                            &[&q0, &q1, &q2, &q3],
                            &[&ip0, &ip1, &ip2, &ip3],
                            &hc,
                            &f1,
                            &f2,
                            &fv1,
                            &fv2,
                            &mut ldiag,
                        );
                        lodd += zz.len() as u64; // should be 0: such states are in the class of v2
                        zero.extend(zz);
                        let fixed = [r as u32, v2 as u32];
                        let mut try_set = |extra: &[u32]| {
                            let mut s: Vec<u32> = fixed.to_vec();
                            s.extend_from_slice(extra);
                            lver += 1;
                            if let Some(d) = check_set_diag(t, psi, &s, &mut ldiag) {
                                let vs: Vec<&[C]> =
                                    d.idx.iter().map(|&i| t.states[i as usize].as_slice()).collect();
                                let (_, res) = lsq(&vs, psip);
                                assert!(res < 1e-7, "Galois lemma violated?");
                                lfound.insert(d.idx);
                            }
                        };
                        for cl in point_classes(&pts) {
                            let m = cl.len();
                            for a in 0..m {
                                for b in (a + 1)..m {
                                    for c in (b + 1)..m {
                                        try_set(&[cl[a], cl[b], cl[c]]);
                                    }
                                    for &w in zero.iter() {
                                        try_set(&[cl[a], cl[b], w]);
                                    }
                                }
                            }
                        }
                        if zero.len() >= 2 {
                            for p in &pts {
                                for i in 0..zero.len() {
                                    for j in (i + 1)..zero.len() {
                                        try_set(&[p.3, zero[i], zero[j]]);
                                    }
                                }
                            }
                        }
                    }
                    if std::env::var("PROGRESS").is_ok() {
                        eprintln!("orbit {}/{} (|cands| {}) found so far {}", o, norbits, cands.len(), lfound.len());
                    }
                }
                let mut g = out.lock().unwrap();
                g.0.extend(lfound);
                g.1 += lpairs;
                g.2 += lcand;
                g.3 += lver;
                g.4 += lodd;
                diag.lock().unwrap().merge(&ldiag);
            });
        }
    });
    let g = out.into_inner().unwrap();
    st.pairs += g.1;
    st.cand += g.2;
    st.verified += g.3;
    if g.4 > 0 {
        eprintln!("  note: {} level-2 zero projections outside the level-1 class (kept as zeros)", g.4);
    }
    eprintln!("  margins: {:?}", diag.into_inner().unwrap());
    g.0
}

// ---------------------------------------------------------------- completions
/// Bottom options for a term whose <0|-restriction of the new last qubit is the normalised
/// stabilizer state `a` (on nq qubits): index 0 is the zero vector (product term |0>a, x = 1);
/// the others are the distinct vectors w X^v Z^u a (term (a, z)/sqrt2, x = 1/sqrt2), z = y/x.
pub fn bottoms(a: &[C], nq: usize) -> Vec<Vec<C>> {
    let dim = 1usize << nq;
    assert_eq!(a.len(), dim);
    let mut out: Vec<Vec<C>> = vec![vec![C::default(); dim]];
    let mut seen: std::collections::HashSet<Vec<i64>> = std::collections::HashSet::new();
    let phases = [
        C::new(1., 0.),
        C::new(0., 1.),
        C::new(-1., 0.),
        C::new(0., -1.),
    ];
    for v in 0..dim {
        for u in 0..dim {
            // (X^v Z^u a)[x ^ v] = (-1)^{u.x} a[x]
            let mut b = vec![C::default(); dim];
            for x in 0..dim {
                let sgn = if (u & x).count_ones() % 2 == 1 { -1.0 } else { 1.0 };
                b[x ^ v] = csc(a[x], sgn);
            }
            for &w in &phases {
                let bw: Vec<C> = b.iter().map(|&z| z * w).collect();
                if seen.insert(key_of(&bw)) {
                    out.push(bw);
                }
            }
        }
    }
    out
}

/// The full n-qubit term for top state a and bottom option z (normalised).
pub fn term(a: &[C], z: &[C]) -> Vec<C> {
    let zero = norm2(z) < 1e-20;
    let s = if zero { 1.0 } else { 1.0 / 2f64.sqrt() };
    let mut v: Vec<C> = a.iter().map(|&x| csc(x, s)).collect();
    v.extend(z.iter().map(|&x| csc(x, s)));
    v
}

/// Verify a candidate n-qubit decomposition: every term a stabilizer state, independent, target
/// in the span with all coefficients non-zero.  Returns the coefficients.
pub fn verify_terms(terms: &[Vec<C>], target: &[C]) -> Option<Vec<C>> {
    if !terms.iter().all(|v| is_stabilizer(v)) {
        return None;
    }
    let refs: Vec<&[C]> = terms.iter().map(|v| v.as_slice()).collect();
    if gram_det(&refs) < 1e-10 {
        return None;
    }
    let (c, res) = lsq(&refs, target);
    if res < 1e-8 && c.iter().all(|z| z.abs() > 1e-8) {
        Some(c)
    } else {
        None
    }
}

// ---------------------------------------------------------------- lifts
pub struct LiftCtx {
    pub nq: usize,       // qubits of the restricted level (n-1)
    pub tpsi: Vec<C>,    // t * psi^{⊗(n-1)}, t = psi_1/psi_0
    pub target: Vec<C>,  // psi^{⊗n}
    pub f: Vec<C>,       // random functionals on C^{2^(n-1)}
    pub g: Vec<C>,
}
impl LiftCtx {
    pub fn new(p: &[C; 2], n: usize) -> LiftCtx {
        LiftCtx::with_last(p, n, p)
    }
    /// Target psi^{⊗(n-1)} ⊗ last (last qubit = most significant bit), restricted by <0|.  With
    /// last = u psi for a single-qubit Clifford u with u|s> = |0>, this is the <s|-restriction of
    /// the decompositions of psi^{⊗n} (apply u to the last qubit of every term).
    pub fn with_last(p: &[C; 2], n: usize, last: &[C; 2]) -> LiftCtx {
        let nq = n - 1;
        let psi = tensor_power(p, nq);
        let t = cdiv(last[1], last[0]);
        let mut target: Vec<C> = psi.iter().map(|&z| z * last[0]).collect();
        target.extend(psi.iter().map(|&z| z * last[1]));
        LiftCtx {
            nq,
            tpsi: psi.iter().map(|&z| z * t).collect(),
            target,
            f: random_functional(1 << nq, 0x51ED2701),
            g: random_functional(1 << nq, 0xC0FFEE11),
        }
    }
}

/// Type I: all bottom choices z_i in bottoms(A_i) with sum_i a_i z_i = t psi.  Returns the
/// verified n-qubit decompositions (term vectors).
pub fn lift_type1(cx: &LiftCtx, tops: &[&[C]], a: &[C]) -> Vec<Vec<Vec<C>>> {
    let k = tops.len();
    assert!(k >= 3);
    let opts: Vec<Vec<Vec<C>>> = tops.iter().map(|v| bottoms(v, cx.nq)).collect();
    let sf: Vec<Vec<C>> = (0..k)
        .map(|i| opts[i].iter().map(|z| a[i] * dot(&cx.f, z)).collect())
        .collect();
    let sg: Vec<Vec<C>> = (0..k)
        .map(|i| opts[i].iter().map(|z| a[i] * dot(&cx.g, z)).collect())
        .collect();
    let tf = dot(&cx.f, &cx.tpsi);
    let tg = dot(&cx.g, &cx.tpsi);
    // left: terms 0, 1
    let mut left: Vec<(f64, C, C, u16, u16)> = vec![];
    for o0 in 0..opts[0].len() {
        for o1 in 0..opts[1].len() {
            let kf = sf[0][o0] + sf[1][o1];
            left.push((kf.re, kf, sg[0][o0] + sg[1][o1], o0 as u16, o1 as u16));
        }
    }
    left.sort_unstable_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    let lkeys: Vec<f64> = left.iter().map(|x| x.0).collect();
    let eps = 1e-8;
    let mut out = vec![];
    // right: terms 2..k, odometer
    let rk = k - 2;
    let mut ch = vec![0usize; rk];
    loop {
        let mut kf = tf;
        let mut kg = tg;
        for (j, &c) in ch.iter().enumerate() {
            kf = kf - sf[j + 2][c];
            kg = kg - sg[j + 2][c];
        }
        let lo = lkeys.partition_point(|&x| x < kf.re - eps);
        let mut i = lo;
        while i < left.len() && left[i].0 <= kf.re + eps {
            let l = &left[i];
            if (l.1 - kf).abs() < eps && (l.2 - kg).abs() < eps {
                let mut choice = vec![l.3 as usize, l.4 as usize];
                choice.extend_from_slice(&ch);
                // full check
                let mut res = cx.tpsi.clone();
                for j in 0..k {
                    for x in 0..res.len() {
                        res[x] = res[x] - a[j] * opts[j][choice[j]][x];
                    }
                }
                if norm2(&res).sqrt() < 1e-8 {
                    let terms: Vec<Vec<C>> =
                        (0..k).map(|j| term(tops[j], &opts[j][choice[j]])).collect();
                    if verify_terms(&terms, &cx.target).is_some() {
                        out.push(terms);
                    }
                }
            }
            i += 1;
        }
        let mut j = 0;
        while j < rk {
            ch[j] += 1;
            if ch[j] < opts[j + 2].len() {
                break;
            }
            ch[j] = 0;
            j += 1;
        }
        if j == rk {
            break;
        }
    }
    out
}

/// Quick necessary test for "v is proportional to a stabilizer state" (support size a power of
/// two, equal magnitudes), followed by the exact test.
fn stab_like(v: &[C]) -> bool {
    let mut mx = 0.0f64;
    for z in v {
        mx = mx.max(z.n2());
    }
    if mx < 1e-18 {
        return false;
    }
    let mut cnt = 0usize;
    for z in v {
        let m = z.n2();
        if m > 1e-9 * mx {
            if (m - mx).abs() > 1e-6 * mx {
                return false;
            }
            cnt += 1;
        }
    }
    cnt.is_power_of_two() && is_stabilizer(v)
}

/// Types II, IIIa, IIIb for one optimal (k-1)-term decomposition D (tops, coefficients d).
/// Returns (found decompositions, counts per type of full candidate checks).
pub fn lift_degenerate(
    cx: &LiftCtx,
    t: &Table,
    tops: &[&[C]],
    d: &[C],
) -> (Vec<Vec<Vec<C>>>, [u64; 3], usize) {
    let m = tops.len(); // k-1
    let dim = 1usize << cx.nq;
    let opts: Vec<Vec<Vec<C>>> = tops.iter().map(|v| bottoms(v, cx.nq)).collect();
    let mut out = vec![];
    let mut checks = [0u64; 3];
    // ---- type II: t psi - sum d_i z_i ∝ stabilizer state B (non-zero); term k = (0, B)
    {
        let mut partial: Vec<Vec<C>> = vec![cx.tpsi.clone(); m + 1];
        let mut ch = vec![0usize; m];
        // recursive odometer with partial sums: partial[j+1] = partial[j] - d_j z_j
        fn rec(
            j: usize,
            m: usize,
            d: &[C],
            opts: &[Vec<Vec<C>>],
            partial: &mut Vec<Vec<C>>,
            ch: &mut Vec<usize>,
            hits: &mut Vec<(Vec<usize>, Vec<C>)>,
            cnt: &mut u64,
        ) {
            if j == m {
                *cnt += 1;
                if stab_like(&partial[m]) {
                    hits.push((ch.clone(), partial[m].clone()));
                }
                return;
            }
            for o in 0..opts[j].len() {
                ch[j] = o;
                let (a, b) = partial.split_at_mut(j + 1);
                let src = &a[j];
                let dst = &mut b[0];
                for x in 0..src.len() {
                    dst[x] = src[x] - d[j] * opts[j][o][x];
                }
                rec(j + 1, m, d, opts, partial, ch, hits, cnt);
            }
        }
        let mut hits = vec![];
        let mut cnt = 0u64;
        rec(0, m, d, &opts, &mut partial, &mut ch, &mut hits, &mut cnt);
        checks[0] = cnt;
        for (ch, r) in hits {
            let rn = norm2(&r).sqrt();
            let mut terms: Vec<Vec<C>> = (0..m).map(|j| term(tops[j], &opts[j][ch[j]])).collect();
            let mut last = vec![C::default(); dim];
            last.extend(r.iter().map(|&z| csc(z, 1.0 / rn)));
            terms.push(last);
            if verify_terms(&terms, &cx.target).is_some() {
                out.push(terms);
            }
        }
    }
    // scalar tables
    let ff: Vec<Vec<C>> = opts.iter().map(|o| o.iter().map(|z| dot(&cx.f, z)).collect()).collect();
    let gg: Vec<Vec<C>> = opts.iter().map(|o| o.iter().map(|z| dot(&cx.g, z)).collect()).collect();
    let tf = dot(&cx.f, &cx.tpsi);
    let tg = dot(&cx.g, &cx.tpsi);
    let eps = 1e-8;
    // ---- type IIIa: tau parallel to D_j.  X = sum d_i z_i - t psi = s (z_j - z_tau)
    for j in 0..m {
        let others: Vec<usize> = (0..m).filter(|&i| i != j).collect();
        let no = opts[j].len();
        for oj in 0..no {
            for ot in 0..no {
                if ot == oj {
                    continue;
                }
                let fd = ff[j][oj] - ff[j][ot];
                let gd = gg[j][oj] - gg[j][ot];
                // sum_{i != j} d_i (gd f(z_i) - fd g(z_i)) = gd tf - fd tg - d_j (gd f(z_j) - fd g(z_j))
                let h = |i: usize, o: usize| d[i] * (gd * ff[i][o] - fd * gg[i][o]);
                let tt = gd * tf - fd * tg - d[j] * (gd * ff[j][oj] - fd * gg[j][oj]);
                // meet in the middle over the m-1 other indices: left = first (m-2), right = last
                let nl = others.len() - 1;
                let mut left: Vec<(f64, C, Vec<u16>)> = vec![];
                let mut ch = vec![0usize; nl];
                loop {
                    let mut s = C::default();
                    for (q, &c) in ch.iter().enumerate() {
                        s = s + h(others[q], c);
                    }
                    left.push((s.re, s, ch.iter().map(|&c| c as u16).collect()));
                    let mut q = 0;
                    while q < nl {
                        ch[q] += 1;
                        if ch[q] < opts[others[q]].len() {
                            break;
                        }
                        ch[q] = 0;
                        q += 1;
                    }
                    if q == nl {
                        break;
                    }
                }
                left.sort_unstable_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
                let lk: Vec<f64> = left.iter().map(|x| x.0).collect();
                let last = others[nl];
                for ol in 0..opts[last].len() {
                    let need = tt - h(last, ol);
                    let scale = 1.0 + need.abs();
                    let lo = lk.partition_point(|&x| x < need.re - eps * scale);
                    let mut i = lo;
                    while i < left.len() && left[i].0 <= need.re + eps * scale {
                        if (left[i].1 - need).abs() < eps * scale {
                            checks[1] += 1;
                            let mut choice = vec![0usize; m];
                            choice[j] = oj;
                            for (q, &c) in left[i].2.iter().enumerate() {
                                choice[others[q]] = c as usize;
                            }
                            choice[last] = ol;
                            // full check: X ∥ Delta, s != 0, s != d_j
                            let mut xv: Vec<C> = cx.tpsi.iter().map(|&z| -z).collect();
                            for q in 0..m {
                                for x in 0..dim {
                                    xv[x] = xv[x] + d[q] * opts[q][choice[q]][x];
                                }
                            }
                            let delta: Vec<C> =
                                (0..dim).map(|x| opts[j][oj][x] - opts[j][ot][x]).collect();
                            let dn = norm2(&delta);
                            let s = cdiv(dot(&delta, &xv), C::new(dn, 0.0));
                            let mut res = 0.0;
                            for x in 0..dim {
                                res += (xv[x] - s * delta[x]).n2();
                            }
                            if res.sqrt() < 1e-8 && s.abs() > 1e-9 && (s - d[j]).abs() > 1e-9 {
                                let mut terms: Vec<Vec<C>> =
                                    (0..m).map(|q| term(tops[q], &opts[q][choice[q]])).collect();
                                terms.push(term(tops[j], &opts[j][ot]));
                                if verify_terms(&terms, &cx.target).is_some() {
                                    out.push(terms);
                                }
                            }
                        }
                        i += 1;
                    }
                }
            }
        }
    }
    // ---- type IIIb: tau in span(D), a stabilizer state parallel to no D_j
    let refs: Vec<&[C]> = tops.to_vec();
    let mut taus: Vec<(usize, Vec<C>)> = vec![];
    for (ix, s) in t.states.iter().enumerate() {
        if tops.iter().any(|tp| parallel(tp, s)) {
            continue;
        }
        let (e, res) = lsq(&refs, s);
        if res < 1e-8 {
            taus.push((ix, e));
        }
    }
    let ntaus = taus.len();
    for (ix, e) in &taus {
        let tau = &t.states[*ix];
        let topt = bottoms(tau, cx.nq);
        let tff: Vec<C> = topt.iter().map(|z| dot(&cx.f, z)).collect();
        let tgg: Vec<C> = topt.iter().map(|z| dot(&cx.g, z)).collect();
        // odometer over the m tops: fX = sum d f - tf, fY' = sum e f
        let mut ch = vec![0usize; m];
        loop {
            let mut fx = -tf;
            let mut gx = -tg;
            let mut fy = C::default();
            let mut gy = C::default();
            for q in 0..m {
                fx = fx + d[q] * ff[q][ch[q]];
                gx = gx + d[q] * gg[q][ch[q]];
                fy = fy + e[q] * ff[q][ch[q]];
                gy = gy + e[q] * gg[q][ch[q]];
            }
            for ot in 0..topt.len() {
                let fyy = fy - tff[ot];
                let gyy = gy - tgg[ot];
                let det = fx * gyy - gx * fyy;
                let scale = 1.0 + (fx.abs() + gx.abs()) * (fyy.abs() + gyy.abs());
                if det.abs() < eps * scale {
                    checks[2] += 1;
                    let mut xv: Vec<C> = cx.tpsi.iter().map(|&z| -z).collect();
                    let mut yv: Vec<C> = topt[ot].iter().map(|&z| -z).collect();
                    for q in 0..m {
                        for x in 0..dim {
                            xv[x] = xv[x] + d[q] * opts[q][ch[q]][x];
                            yv[x] = yv[x] + e[q] * opts[q][ch[q]][x];
                        }
                    }
                    let yn = norm2(&yv);
                    if yn < 1e-18 {
                        continue; // X = s*0 forces X = 0, i.e. s free: then s is not determined
                    }
                    let s = cdiv(dot(&yv, &xv), C::new(yn, 0.0));
                    let mut res = 0.0;
                    for x in 0..dim {
                        res += (xv[x] - s * yv[x]).n2();
                    }
                    if res.sqrt() < 1e-8
                        && s.abs() > 1e-9
                        && (0..m).all(|q| (d[q] - s * e[q]).abs() > 1e-9)
                    {
                        let mut terms: Vec<Vec<C>> =
                            (0..m).map(|q| term(tops[q], &opts[q][ch[q]])).collect();
                        terms.push(term(tau, &topt[ot]));
                        if verify_terms(&terms, &cx.target).is_some() {
                            out.push(terms);
                        }
                    }
                }
            }
            let mut q = 0;
            while q < m {
                ch[q] += 1;
                if ch[q] < opts[q].len() {
                    break;
                }
                ch[q] = 0;
                q += 1;
            }
            if q == m {
                break;
            }
        }
    }
    (out, checks, ntaus)
}

// ---------------------------------------------------------------- driver
pub struct Level {
    pub t: Table,
    pub psi: Vec<C>,
    pub psip: Vec<C>,
    pub gens: Vec<Vec<u32>>,
    pub orbit: Vec<u32>,
    pub group: Vec<Vec<u16>>,
}
impl Level {
    pub fn new(kind: &str, n: usize) -> Level {
        let p = psi1(kind);
        let t = Table::new(n);
        let psi = tensor_power(&p, n);
        let psip = tensor_power(&perp1(&p), n);
        let gens = sym_generators(&t, &p);
        let orbit = orbits(t.states.len(), &gens);
        let group = group_closure(&gens, 100_000).expect("group too large");
        Level { t, psi, psip, gens, orbit, group }
    }
}

/// Decompositions (as index sets) -> coefficient vectors for the normalised target.
pub fn coefs(lv: &Level, s: &[u32]) -> Vec<C> {
    check_set(&lv.t, &lv.psi, s).expect("not a decomposition").coef
}

fn write_sets(path: &str, lv: &Level, sets: &[Vec<u32>], header: &str) {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    writeln!(f, "# {}  format: idx:re,im per term (idx = position in enum_states(n))", header).unwrap();
    for s in sets {
        let c = coefs(lv, s);
        let parts: Vec<String> = s
            .iter()
            .zip(&c)
            .map(|(i, z)| format!("{}:{:.15},{:.15}", i, z.re, z.im))
            .collect();
        writeln!(f, "{}", parts.join(" ")).unwrap();
    }
}

/// All optimal / minimal k-term decompositions at level lv, as G-orbit representatives, plus the
/// total number of decompositions.  k = 3 uses the older projective-hash search of stabrank.rs,
/// k = 4, 5 the Galois search.
pub fn decs(lv: &Level, k: usize) -> (Vec<Vec<u32>>, usize, f64) {
    let t0 = std::time::Instant::now();
    let found: Vec<Vec<u32>> = if k == 3 {
        let mut st = SearchStats { w_count: 0, cand_count: 0, verified: 0 };
        let g16 = lv.group.clone();
        search(&lv.t, &lv.psi, 3, &lv.orbit, Some(&g16), &mut st)
            .into_iter()
            .map(|d| d.idx)
            .collect()
    } else {
        let mut st = GStats { pairs: 0, cand: 0, verified: 0 };
        let f = if k == 5 && std::env::var("SIMPLE").is_err() {
            gsearch5r(&lv.t, &lv.psi, &lv.psip, &lv.orbit, &lv.group, &mut st)
        } else {
            gsearch(&lv.t, &lv.psi, &lv.psip, k, &lv.orbit, &lv.group, &mut st)
        };
        eprintln!(
            "  gsearch k={}: pairs={} candidates={} verified={} raw found={}",
            k, st.pairs, st.cand, st.verified, f.len()
        );
        f
    };
    let (reps, total) = dedupe_orbits(&found, &lv.group);
    (reps, total, t0.elapsed().as_secs_f64())
}

/// Lift every representative (type I for `mins`, degenerate types for `opts`) to level n.
/// Returns the found n-qubit decompositions (term vectors).
pub fn lift_all(
    kind: &str,
    n: usize,
    lv: &Level,
    opts_reps: &[Vec<u32>],
    mins_reps: &[Vec<u32>],
) -> Vec<Vec<Vec<C>>> {
    let p = psi1(kind);
    let last = last_state(kind);
    let cx = LiftCtx::with_last(&p, n, &last);
    let found = Mutex::new(vec![]);
    let stats = Mutex::new(([0u64; 3], 0usize));
    // degenerate part
    let t0 = std::time::Instant::now();
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..nthreads() {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                if i >= opts_reps.len() {
                    break;
                }
                let s = &opts_reps[i];
                let tops: Vec<&[C]> = s.iter().map(|&x| lv.t.states[x as usize].as_slice()).collect();
                let d = coefs(lv, s);
                let (f, ch, nt) = lift_degenerate(&cx, &lv.t, &tops, &d);
                let mut st = stats.lock().unwrap();
                for q in 0..3 {
                    st.0[q] += ch[q];
                }
                st.1 += nt;
                drop(st);
                if !f.is_empty() {
                    found.lock().unwrap().extend(f);
                }
            });
        }
    });
    let st = stats.into_inner().unwrap();
    let ndeg = found.lock().unwrap().len();
    println!(
        "  degenerate part: {} optimal reps, II combos {}, IIIa scalar hits {}, IIIb taus {} (scalar hits {}), found {} ({:.1}s)",
        opts_reps.len(),
        st.0[0],
        st.0[1],
        st.1,
        st.0[2],
        ndeg,
        t0.elapsed().as_secs_f64()
    );
    // type I part
    let t1 = std::time::Instant::now();
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..nthreads() {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                if i >= mins_reps.len() {
                    break;
                }
                let s = &mins_reps[i];
                let tops: Vec<&[C]> = s.iter().map(|&x| lv.t.states[x as usize].as_slice()).collect();
                let a = coefs(lv, s);
                let f = lift_type1(&cx, &tops, &a);
                if !f.is_empty() {
                    found.lock().unwrap().extend(f);
                }
                let dn = done.fetch_add(1, Ordering::SeqCst) + 1;
                if std::env::var("PROGRESS").is_ok() && dn % 1000 == 0 {
                    eprintln!("  type I lifted {}/{} ({:.0}s)", dn, mins_reps.len(), t1.elapsed().as_secs_f64());
                }
            });
        }
    });
    let out = found.into_inner().unwrap();
    println!(
        "  type-I part: {} minimal reps, found {} ({:.1}s)",
        mins_reps.len(),
        out.len() - ndeg,
        t1.elapsed().as_secs_f64()
    );
    out
}

/// The last-qubit target u psi for the restriction bra BRA (index into the six single-qubit
/// stabilizer states of Table::new(1); default 0 = |0>), u a Clifford with u|s> = |0>.
pub fn last_state(kind: &str) -> [C; 2] {
    let p = psi1(kind);
    let s: usize = std::env::var("BRA").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    if s == 0 {
        return p;
    }
    let t1 = Table::new(1);
    let u = cliffords1()
        .into_iter()
        .find(|m| apply1(m, &t1.states[s])[0].abs() > 1.0 - 1e-9)
        .unwrap();
    apply1(&u, &p)
}

/// Set key of an n-qubit decomposition (sorted canonical vector keys).
pub fn set_key(terms: &[Vec<C>]) -> Vec<Vec<i64>> {
    let mut ks: Vec<Vec<i64>> = terms.iter().map(|v| key_of(&canonical(v))).collect();
    ks.sort();
    ks
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");
    match cmd {
        // bcheck <n>: bottoms() equals the is_stabilizer-based completions() of stabrank.rs
        "bcheck" => {
            let n: usize = args[2].parse().unwrap();
            let step: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
            let t = Table::new(n);
            let mut checked = 0;
            for (ix, a) in t.states.iter().enumerate().step_by(step) {
                let b = bottoms(a, n);
                let c = completions(&t, a);
                let mut kb: Vec<Vec<i64>> = b.iter().map(|z| key_of(z)).collect();
                let mut kc: Vec<Vec<i64>> = c
                    .iter()
                    .map(|(x, y)| key_of(&y.iter().map(|&z| csc(z, 1.0 / x)).collect::<Vec<C>>()))
                    .collect();
                kb.sort();
                kc.sort();
                assert_eq!(kb, kc, "state {}", ix);
                assert_eq!(b.len(), 1 + 4 * (1 << n));
                checked += 1;
            }
            println!("bcheck n={}: {} states, bottoms == completions (each 1 + 4*2^n = {})", n, checked, 1 + 4 * (1 << n));
        }
        // galois <kind> <n> <k>: Galois lemma on the optimal decompositions found by the old search
        "galois" => {
            let kind = &args[2];
            let n: usize = args[3].parse().unwrap();
            let k: usize = args[4].parse().unwrap();
            let lv = Level::new(kind, n);
            let mut st = SearchStats { w_count: 0, cand_count: 0, verified: 0 };
            let g16 = lv.group.clone();
            let reps = search(&lv.t, &lv.psi, k, &lv.orbit, Some(&g16), &mut st);
            let all = expand(&lv.t, &lv.psi, &reps, &lv.gens);
            let mut worst = 0.0f64;
            for d in &all {
                let vs: Vec<&[C]> = d.idx.iter().map(|&i| lv.t.states[i as usize].as_slice()).collect();
                let (_, res) = lsq(&vs, &lv.psip);
                worst = worst.max(res);
            }
            println!("galois {}^{} k={}: {} decompositions, max residual of psi_perp^n in their span = {:.2e}", kind, n, k, all.len(), worst);
        }
        // gsearch <kind> <n> <k>: all minimal k-term decompositions (Galois search), counts + reps file
        "gsearch" => {
            let kind = &args[2];
            let n: usize = args[3].parse().unwrap();
            let k: usize = args[4].parse().unwrap();
            let lv = Level::new(kind, n);
            println!(
                "{}^{}: {} states, {} orbits, |G| = {}",
                kind,
                n,
                lv.t.states.len(),
                lv.orbit.iter().max().unwrap() + 1,
                lv.group.len()
            );
            let (reps, total, secs) = decs(&lv, k);
            let l1: Vec<f64> = reps.iter().map(|s| coefs(&lv, s).iter().map(|z| z.abs()).sum()).collect();
            let (mn, mx) = l1.iter().fold((f64::MAX, 0f64), |(a, b), &x| (a.min(x), b.max(x)));
            println!(
                "{}^{} k={}: {} decompositions in {} orbits ({:.1}s); l1 range [{:.6}, {:.6}], ratio {:.4}",
                kind, n, k, total, reps.len(), secs, mn, mx, mx / mn
            );
            write_sets(
                &format!("min_{}{}_k{}_reps.txt", kind, n, k),
                &lv,
                &reps,
                &format!("{}^{} k={}: {} decompositions, {} orbit reps", kind, n, k, total, reps.len()),
            );
        }
        // pipeline <kind> <n> <k>: all k-term decompositions of psi^{⊗n}, k = chi(n-1)+1
        "pipeline" => {
            let kind = args[2].clone();
            let n: usize = args[3].parse().unwrap();
            let k: usize = args[4].parse().unwrap();
            let t0 = std::time::Instant::now();
            let lv = Level::new(&kind, n - 1);
            println!(
                "level {}^{}: {} states, {} orbits, |G| = {}",
                kind,
                n - 1,
                lv.t.states.len(),
                lv.orbit.iter().max().unwrap() + 1,
                lv.group.len()
            );
            // the symmetry reduction needs every antiunitary element of G_{n-1} to extend to the last
            // qubit by a diagonal Clifford d with d conj(last) ∥ last (so <0| is preserved)
            {
                let last = last_state(&kind);
                let ok = (0..4).any(|m| {
                    let ph = [C::new(1., 0.), C::new(0., 1.), C::new(-1., 0.), C::new(0., -1.)][m];
                    parallel(&[cj(last[0]), ph * cj(last[1])], &last)
                });
                assert!(ok, "antiunitary symmetries do not extend for this bra");
                println!("  restriction bra #{} (last-qubit target {:.6}{:+.6}i, {:.6}{:+.6}i)",
                    std::env::var("BRA").unwrap_or("0".into()), last[0].re, last[0].im, last[1].re, last[1].im);
            }
            let (orep, ototal, osecs) = decs(&lv, k - 1);
            println!(
                "  optimal ({}-term) decompositions of {}^{}: {} in {} orbits ({:.1}s)",
                k - 1, kind, n - 1, ototal, orep.len(), osecs
            );
            // chi(n-1) = k-1: no (k-2)-term decomposition (k-2 = 2 or 3 via the old search)
            {
                let mut st = SearchStats { w_count: 0, cand_count: 0, verified: 0 };
                let g16 = lv.group.clone();
                let lower = search(&lv.t, &lv.psi, k - 2, &lv.orbit, Some(&g16), &mut st);
                assert!(lower.is_empty(), "chi(n-1) < k-1");
                println!("  no {}-term decomposition of {}^{} (old search)", k - 2, kind, n - 1);
            }
            let (mrep, mtotal, msecs) = decs(&lv, k);
            println!(
                "  minimal {}-term decompositions of {}^{}: {} in {} orbits ({:.1}s)",
                k, kind, n - 1, mtotal, mrep.len(), msecs
            );
            write_sets(
                &format!("min_{}{}_k{}_reps.txt", kind, n - 1, k),
                &lv,
                &mrep,
                &format!("{}^{} k={}: {} decompositions, {} orbit reps", kind, n - 1, k, mtotal, mrep.len()),
            );
            let found = lift_all(&kind, n, &lv, &orep, &mrep);
            let mut sets: std::collections::HashSet<Vec<Vec<i64>>> = std::collections::HashSet::new();
            for f in &found {
                sets.insert(set_key(f));
            }
            println!(
                "{}^{}: {}-term decompositions found (up to G_{}): {} raw, {} distinct sets  [total {:.1}s]",
                kind, n, k, n - 1, found.len(), sets.len(), t0.elapsed().as_secs_f64()
            );
            if n <= 4 && !found.is_empty() {
                // expand under the full n-qubit symmetry group and count
                let tn = Table::new(n);
                let p = psi1(&kind);
                let psin = tensor_power(&p, n);
                let gn = sym_generators(&tn, &p);
                let ds: Vec<Dec> = found
                    .iter()
                    .map(|f| {
                        let mut ix: Vec<u32> = f.iter().map(|v| tn.lookup(v).unwrap()).collect();
                        ix.sort();
                        check_set(&tn, &psin, &ix).unwrap()
                    })
                    .collect();
                let all = expand(&tn, &psin, &ds, &gn);
                println!("{}^{}: total {} decompositions with {} terms (expanded under G_{})", kind, n, all.len(), k, n);
                let path = format!("../stabrank-lower/dec_{}{}_k{}.txt", kind, n, k);
                if let Ok(txt) = std::fs::read_to_string(&path) {
                    let mut old: Vec<Vec<u32>> = txt
                        .lines()
                        .filter(|l| !l.starts_with('#'))
                        .map(|l| {
                            let mut v: Vec<u32> =
                                l.split_whitespace().map(|p| p.split(':').next().unwrap().parse().unwrap()).collect();
                            v.sort();
                            v
                        })
                        .collect();
                    old.sort();
                    let mut new: Vec<Vec<u32>> = all.iter().map(|d| d.idx.clone()).collect();
                    new.sort();
                    println!("  compared with {}: {} sets there, identical: {}", path, old.len(), old == new);
                }
            }
            for f in found.iter().take(3) {
                println!("  example decomposition:");
                let c = verify_terms(f, &tensor_power(&psi1(&kind), n)).unwrap();
                for (v, z) in f.iter().zip(&c) {
                    println!(
                        "    c={:.12}{:+.12}i  v={}",
                        z.re,
                        z.im,
                        canonical(v).iter().map(|z| format!("({:.4},{:.4})", z.re, z.im)).collect::<Vec<_>>().join(" ")
                    );
                }
            }
        }
        // orbitsizes <kind> <n>: orbit sizes of the n-qubit stabilizer states under G
        "orbitsizes" => {
            let lv = Level::new(&args[2], args[3].parse().unwrap());
            let rep = orbit_reps_idx(&lv.orbit);
            let mut sizes = vec![0usize; rep.len()];
            for &o in &lv.orbit {
                sizes[o as usize] += 1;
            }
            // predicted relative work of the refined k = 5 search per orbit
            let nst = lv.t.states.len();
            let mut ge = vec![0usize; rep.len() + 1];
            for o in (0..rep.len()).rev() {
                ge[o] = ge[o + 1] + sizes[o];
            }
            let mut tot = 0f64;
            let mut work = vec![];
            for o in 0..rep.len() {
                // sum over v2 (orbit o2 >= o, weight |O_o|/|G| for Stab-minimality) of |{orbit >= o2}|
                let mut w = 0f64;
                for o2 in o..rep.len() {
                    w += sizes[o2] as f64 * ge[o2] as f64;
                }
                w *= sizes[o] as f64 / lv.group.len() as f64;
                tot += w;
                work.push(w);
            }
            let mut acc = 0f64;
            for o in 0..rep.len() {
                acc += work[o];
                if o % 10 == 0 || o + 1 == rep.len() {
                    println!("orbit {:3} size {:4} cumulative predicted work {:.3}", o, sizes[o], acc / tot);
                }
            }
            println!("states {}, orbits {}, |G| {}", nst, rep.len(), lv.group.len());
        }
        // olddeg <kind> <n> <k>: cross-check of the degenerate part with the independent brute-force
        // degenerate_search() of stabrank.rs, one thread per optimal-decomposition representative.
        "olddeg" => {
            let kind = args[2].clone();
            let n: usize = args[3].parse().unwrap();
            let k: usize = args[4].parse().unwrap();
            let t0 = std::time::Instant::now();
            let lv = Level::new(&kind, n - 1);
            let last = last_state(&kind);
            let (orep, ototal, _) = decs(&lv, k - 1);
            println!(
                "olddeg {}^{} k={} bra #{}: {} optimal decompositions in {} orbits",
                kind, n, k, std::env::var("BRA").unwrap_or("0".into()), ototal, orep.len()
            );
            let next = AtomicUsize::new(0);
            let res = Mutex::new((0usize, 0u64));
            std::thread::scope(|sc| {
                for _ in 0..nthreads() {
                    sc.spawn(|| loop {
                        let i = next.fetch_add(1, Ordering::SeqCst);
                        if i >= orep.len() {
                            break;
                        }
                        let d = Dec { idx: orep[i].clone(), coef: coefs(&lv, &orep[i]) };
                        let (f, combos) = degenerate_search(&lv.t, &[d], last[0], last[1], k, true);
                        eprintln!("  rep {}: combos {} found {} ({:.0}s)", i, combos, f.len(), t0.elapsed().as_secs_f64());
                        let mut r = res.lock().unwrap();
                        r.0 += f.len();
                        r.1 += combos;
                    });
                }
            });
            let r = res.into_inner().unwrap();
            println!("olddeg {}^{} k={}: combos {} found {} [{:.0}s]", kind, n, k, r.1, r.0, t0.elapsed().as_secs_f64());
        }
        // gluesym <kind> <n> <k> <reps file>: the all-type-I case of the symmetric split, by the
        // independent glue_from() of stabrank.rs.  If every single-qubit stabilizer restriction is of
        // type I, the <0|- and <1|-restrictions A, B of the last qubit are both minimal k-term
        // decompositions with l1(B) = l1(A)/|psi_1/psi_0|; glue_from tries every such pair (A over
        // orbit reps, B over the full orbits of the reps whose l1 can match) and every bijection.
        "gluesym" => {
            let kind = args[2].clone();
            let n: usize = args[3].parse().unwrap();
            let k: usize = args[4].parse().unwrap();
            let path = &args[5];
            let t0 = std::time::Instant::now();
            let lv = Level::new(&kind, n - 1);
            let p = psi1(&kind);
            let rabs = cdiv(p[1], p[0]).abs();
            let txt = std::fs::read_to_string(path).expect("reps file");
            let reps: Vec<Dec> = txt
                .lines()
                .filter(|l| !l.starts_with('#'))
                .map(|l| {
                    let mut v: Vec<u32> = l
                        .split_whitespace()
                        .map(|p| p.split(':').next().unwrap().parse().unwrap())
                        .collect();
                    v.sort();
                    assert_eq!(v.len(), k);
                    check_set(&lv.t, &lv.psi, &v).expect("rep is not a decomposition")
                })
                .collect();
            let l1: Vec<f64> = l1_norms(&reps);
            let (mn, mx) = l1.iter().fold((f64::MAX, 0f64), |(a, b), &x| (a.min(x), b.max(x)));
            println!(
                "gluesym {}^{} k={}: {} reps, l1 in [{:.6}, {:.6}], ratio {:.4} (gluing needs ratio 1/|psi1/psi0| = {:.4})",
                kind, n, k, reps.len(), mn, mx, mx / mn, 1.0 / rabs
            );
            // A side: reps with l1(A)/rabs <= max; B side: reps with l1(B) >= min/rabs
            let aside: Vec<Dec> = reps
                .iter()
                .zip(&l1)
                .filter(|(_, &x)| x / rabs <= mx * (1.0 + 1e-9))
                .map(|(d, _)| d.clone())
                .collect();
            let bsel: Vec<Dec> = reps
                .iter()
                .zip(&l1)
                .filter(|(_, &x)| x >= mn / rabs * (1.0 - 1e-9))
                .map(|(d, _)| d.clone())
                .collect();
            let bfull = expand(&lv.t, &lv.psi, &bsel, &lv.gens);
            println!(
                "  A side: {} reps; B side: {} reps -> {} decompositions after expansion",
                aside.len(),
                bsel.len(),
                bfull.len()
            );
            let g = if aside.is_empty() || bfull.is_empty() {
                vec![]
            } else {
                glue_from(&lv.t, &aside, &bfull, &p, n)
            };
            println!(
                "gluesym {}^{} k={}: glued decompositions (all restrictions of type I): {}  [{:.1}s]",
                kind, n, k, g.len(), t0.elapsed().as_secs_f64()
            );
        }
        // symsplit <kind> <n> <k> <reps file>: Algorithm 2.  Case A: some single-qubit stabilizer
        // restriction is degenerate -- WLOG (symmetry of psi^{⊗n}) the last qubit and a bra from each
        // orbit of the six bras: lift_degenerate.  Case B: all restrictions of type I -- glue_from on
        // the <0|/<1| pair (as gluesym).  For n <= 4 the union is expanded and compared with the
        // known list.
        "symsplit" => {
            let kind = args[2].clone();
            let n: usize = args[3].parse().unwrap();
            let k: usize = args[4].parse().unwrap();
            let path = &args[5];
            let t0 = std::time::Instant::now();
            let lv = Level::new(&kind, n - 1);
            let p = psi1(&kind);
            let (orep, ototal, _) = decs(&lv, k - 1);
            println!("symsplit {}^{} k={}: {} optimal ({}-term) decompositions in {} orbits", kind, n, k, ototal, k - 1, orep.len());
            let t1 = Table::new(1);
            let o1 = orbits(6, &sym_generators(&t1, &p));
            let cl = cliffords1();
            let mut all_found: Vec<Vec<Vec<C>>> = vec![];
            for oid in 0..=*o1.iter().max().unwrap() {
                let sidx = (0..6).find(|&i| o1[i] == oid).unwrap();
                let u = *cl.iter().find(|m| apply1(m, &t1.states[sidx])[0].abs() > 1.0 - 1e-9).unwrap();
                let last = apply1(&u, &p);
                let ok = (0..4).any(|m| {
                    let ph = [C::new(1., 0.), C::new(0., 1.), C::new(-1., 0.), C::new(0., -1.)][m];
                    parallel(&[cj(last[0]), ph * cj(last[1])], &last)
                });
                assert!(ok, "antiunitary symmetries do not extend for bra {}", sidx);
                let cx = LiftCtx::with_last(&p, n, &last);
                let next = AtomicUsize::new(0);
                let found = Mutex::new(vec![]);
                std::thread::scope(|sc| {
                    for _ in 0..nthreads() {
                        sc.spawn(|| loop {
                            let i = next.fetch_add(1, Ordering::SeqCst);
                            if i >= orep.len() {
                                break;
                            }
                            let tops: Vec<&[C]> = orep[i].iter().map(|&x| lv.t.states[x as usize].as_slice()).collect();
                            let d = coefs(&lv, &orep[i]);
                            let (f, _, _) = lift_degenerate(&cx, &lv.t, &tops, &d);
                            found.lock().unwrap().extend(f);
                        });
                    }
                });
                let f = found.into_inner().unwrap();
                println!("  case A, bra #{}: {} degenerate-restriction decompositions [{:.1}s]", sidx, f.len(), t0.elapsed().as_secs_f64());
                let udag: M2 = [cj(u[0]), cj(u[2]), cj(u[1]), cj(u[3])];
                for terms in f {
                    all_found.push(terms.iter().map(|v| apply_local(&udag, n - 1, v)).collect());
                }
            }
            // case B
            let rabs = cdiv(p[1], p[0]).abs();
            let txt = std::fs::read_to_string(path).expect("reps file");
            let reps: Vec<Dec> = txt
                .lines()
                .filter(|l| !l.starts_with('#'))
                .map(|l| {
                    let mut v: Vec<u32> = l.split_whitespace().map(|p| p.split(':').next().unwrap().parse().unwrap()).collect();
                    v.sort();
                    check_set(&lv.t, &lv.psi, &v).expect("rep is not a decomposition")
                })
                .collect();
            let l1 = l1_norms(&reps);
            let (mn, mx) = l1.iter().fold((f64::MAX, 0f64), |(a, b), &x| (a.min(x), b.max(x)));
            let aside: Vec<Dec> = reps.iter().zip(&l1).filter(|(_, &x)| x / rabs <= mx * (1.0 + 1e-9)).map(|(d, _)| d.clone()).collect();
            let bsel: Vec<Dec> = reps.iter().zip(&l1).filter(|(_, &x)| x >= mn / rabs * (1.0 - 1e-9)).map(|(d, _)| d.clone()).collect();
            let bfull = expand(&lv.t, &lv.psi, &bsel, &lv.gens);
            let g = if aside.is_empty() || bfull.is_empty() { vec![] } else { glue_from(&lv.t, &aside, &bfull, &p, n) };
            println!(
                "  case B (all type I): {} reps (l1 in [{:.6}, {:.6}]), A side {}, B side {} reps / {} expanded; glued {} [{:.1}s]",
                reps.len(), mn, mx, aside.len(), bsel.len(), bfull.len(), g.len(), t0.elapsed().as_secs_f64()
            );
            for (terms, _) in g {
                all_found.push(terms);
            }
            for f in &all_found {
                assert!(verify_terms(f, &tensor_power(&p, n)).is_some(), "found set does not verify");
            }
            println!("symsplit {}^{} k={}: {} decompositions found (raw)", kind, n, k, all_found.len());
            if n <= 4 && !all_found.is_empty() {
                let tn = Table::new(n);
                let psin = tensor_power(&p, n);
                let gn = sym_generators(&tn, &p);
                let ds: Vec<Dec> = all_found
                    .iter()
                    .map(|f| {
                        let mut ix: Vec<u32> = f.iter().map(|v| tn.lookup(v).unwrap()).collect();
                        ix.sort();
                        check_set(&tn, &psin, &ix).unwrap()
                    })
                    .collect();
                let all = expand(&tn, &psin, &ds, &gn);
                let mut newl: Vec<Vec<u32>> = all.iter().map(|d| d.idx.clone()).collect();
                newl.sort();
                print!("symsplit {}^{}: total {} decompositions (expanded under G_{})", kind, n, all.len(), n);
                let opath = format!("../stabrank-lower/dec_{}{}_k{}.txt", kind, n, k);
                if let Ok(txt) = std::fs::read_to_string(&opath) {
                    let mut old: Vec<Vec<u32>> = txt
                        .lines()
                        .filter(|l| !l.starts_with('#'))
                        .map(|l| {
                            let mut v: Vec<u32> = l.split_whitespace().map(|p| p.split(':').next().unwrap().parse().unwrap()).collect();
                            v.sort();
                            v
                        })
                        .collect();
                    old.sort();
                    print!("; identical to {}: {}", opath, old == newl);
                }
                println!();
            }
        }
        // liftfile <kind> <n> <k> <reps file> [deg|type1|all]: lifts from a saved list of minimal
        // k-term decompositions of psi^{⊗(n-1)} (orbit reps, as written by `gsearch`), with the
        // restriction bra BRA (default 0); the optimal (k-1)-term list is recomputed.
        "liftfile" => {
            let kind = args[2].clone();
            let n: usize = args[3].parse().unwrap();
            let k: usize = args[4].parse().unwrap();
            let path = &args[5];
            let mode = args.get(6).map(|s| s.as_str()).unwrap_or("all");
            let t0 = std::time::Instant::now();
            let lv = Level::new(&kind, n - 1);
            let last = last_state(&kind);
            let ok = (0..4).any(|m| {
                let ph = [C::new(1., 0.), C::new(0., 1.), C::new(-1., 0.), C::new(0., -1.)][m];
                parallel(&[cj(last[0]), ph * cj(last[1])], &last)
            });
            assert!(ok, "antiunitary symmetries do not extend for this bra");
            println!(
                "level {}^{}: |G| = {}; restriction bra #{} (last-qubit target {:.6}{:+.6}i, {:.6}{:+.6}i)",
                kind, n - 1, lv.group.len(), std::env::var("BRA").unwrap_or("0".into()),
                last[0].re, last[0].im, last[1].re, last[1].im
            );
            let (orep, ototal, _) = decs(&lv, k - 1);
            println!("  optimal ({}-term) decompositions: {} in {} orbits", k - 1, ototal, orep.len());
            let txt = std::fs::read_to_string(path).expect("reps file");
            let mut header = String::new();
            let mut mrep: Vec<Vec<u32>> = vec![];
            for l in txt.lines() {
                if l.starts_with('#') {
                    header = l.to_string();
                    continue;
                }
                let mut v: Vec<u32> = l
                    .split_whitespace()
                    .map(|p| p.split(':').next().unwrap().parse().unwrap())
                    .collect();
                v.sort();
                assert_eq!(v.len(), k);
                mrep.push(v);
            }
            println!("  loaded {} minimal {}-term reps from {} ({})", mrep.len(), k, path, header);
            let (o2, m2): (&[Vec<u32>], &[Vec<u32>]) = match mode {
                "deg" => (&orep, &[]),
                "type1" => (&[], &mrep),
                _ => (&orep, &mrep),
            };
            let found = lift_all(&kind, n, &lv, o2, m2);
            let mut sets: std::collections::HashSet<Vec<Vec<i64>>> = std::collections::HashSet::new();
            for f in &found {
                sets.insert(set_key(f));
            }
            if n <= 4 && !found.is_empty() {
                // rotate back (u^dagger on the last qubit), expand under G_n, compare with the old list
                let s: usize = std::env::var("BRA").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
                let t1 = Table::new(1);
                let u = cliffords1()
                    .into_iter()
                    .find(|m| apply1(m, &t1.states[s])[0].abs() > 1.0 - 1e-9)
                    .unwrap();
                let udag: M2 = [cj(u[0]), cj(u[2]), cj(u[1]), cj(u[3])];
                let tn = Table::new(n);
                let p = psi1(&kind);
                let psin = tensor_power(&p, n);
                let gn = sym_generators(&tn, &p);
                let ds: Vec<Dec> = found
                    .iter()
                    .map(|f| {
                        let mut ix: Vec<u32> = f
                            .iter()
                            .map(|v| tn.lookup(&apply_local(&udag, n - 1, v)).unwrap())
                            .collect();
                        ix.sort();
                        check_set(&tn, &psin, &ix).unwrap()
                    })
                    .collect();
                let all = expand(&tn, &psin, &ds, &gn);
                let mut newl: Vec<Vec<u32>> = all.iter().map(|d| d.idx.clone()).collect();
                newl.sort();
                print!("{}^{}: total {} decompositions with {} terms (expanded under G_{})", kind, n, all.len(), k, n);
                let path = format!("../stabrank-lower/dec_{}{}_k{}.txt", kind, n, k);
                if let Ok(txt) = std::fs::read_to_string(&path) {
                    let mut old: Vec<Vec<u32>> = txt
                        .lines()
                        .filter(|l| !l.starts_with('#'))
                        .map(|l| {
                            let mut v: Vec<u32> = l
                                .split_whitespace()
                                .map(|p| p.split(':').next().unwrap().parse().unwrap())
                                .collect();
                            v.sort();
                            v
                        })
                        .collect();
                    old.sort();
                    print!("; identical to {}: {}", path, old == newl);
                }
                println!();
            }
            println!(
                "{}^{} (bra #{}, mode {}): {}-term decompositions found: {} raw, {} distinct  [total {:.1}s]",
                kind, n, std::env::var("BRA").unwrap_or("0".into()), mode, k, found.len(), sets.len(),
                t0.elapsed().as_secs_f64()
            );
        }
        _ => println!("commands: bcheck n [step] | galois K n k | gsearch K n k | pipeline K n k | liftfile K n k file [deg|type1|all]"),
    }
}
