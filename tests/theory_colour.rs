//! Executable checks for research/theory-colour.md (triangular 6.6.6 colour
//! code, one auxiliary per plaquette).
//!
//! * Theorem 1 (corner lemma) and Theorem 2 (boundary lemma). The explicit
//!   weight-d logicals of the proofs (families I and II, their 120-degree
//!   rotations and one-plaquette flips) are built for every odd d up to 41.
//!   Each must be a logical of weight d meeting its plaquette in exactly the
//!   claimed pair. The certified malign pairs of every boundary plaquette must
//!   leave no sequential CNOT order without a malign hook.
//! * Exactness for small d. All weight-d logicals are enumerated. The exact
//!   malign hook classes must contain the certified ones, no boundary plaquette
//!   may have a safe order, and every interior plaquette has exactly 48 safe
//!   orders.
//! * Circuit level. For actual circuits (K-F, tri-optimal and the d = 9 global
//!   schedule), the (d-1)-fault logical of Theorem 2 is assembled from
//!   mechanisms of the circuit's own detector error model, at every boundary
//!   plaquette.
//! * Theorem 3. Deleting every circuit mechanism whose residual is not
//!   equivalent to weight <= 1 leaves circuit distance exactly d.
use qsim_lab::qec::color::{ColorCode, ColorNoise, ColorSchedule, KF_SCHEDULE, TRI_OPTIMAL};
use qsim_lab::qec::distance::min_logical;
use std::collections::{BTreeSet, HashMap, HashSet};

type Pt = (i32, i32);
type Set = BTreeSet<usize>;

/// Lattice offsets of positions a..f (= color.rs OFFSETS in (u, v) coordinates).
const NB: [Pt; 6] = [(-1, 1), (0, 1), (1, 0), (1, -1), (0, -1), (-1, 0)];

/// The code in lattice coordinates u = (x - 2y)/4, v = y: the triangle
/// u, v >= 0, u + v <= L = 3m, plaquettes where u - v = 2 (mod 3).
struct Tri {
    d: usize,
    m: i32,
    l: i32,
    q_of: HashMap<Pt, usize>,
    p_uv: Vec<Pt>,
    p_of: HashMap<Pt, usize>,
    supp: Vec<Vec<usize>>,
    bottom: Set,
}

impl Tri {
    fn new(d: usize) -> Self {
        let cc = ColorCode::new(d);
        let m = ((d - 1) / 2) as i32;
        let l = 3 * m;
        let uv = |x: i32, y: i32| ((x - 2 * y) / 4, y);
        let q_of: HashMap<Pt, usize> = cc
            .data
            .iter()
            .enumerate()
            .map(|(i, &(x, y))| (uv(x, y), i))
            .collect();
        let p_uv: Vec<Pt> = cc.plaquettes.iter().map(|p| uv(p.x, p.y)).collect();
        let p_of = p_uv.iter().enumerate().map(|(i, &p)| (p, i)).collect();
        let mut supp = Vec::new();
        for (pi, p) in cc.plaquettes.iter().enumerate() {
            let (u, v) = p_uv[pi];
            assert_eq!((u - v).rem_euclid(3), 2, "plaquette site rule");
            for k in 0..6 {
                let site = (u + NB[k].0, v + NB[k].1);
                assert_eq!(p.data[k], q_of.get(&site).copied(), "offset map");
            }
            supp.push(p.data.iter().flatten().copied().collect());
        }
        for &(u, v) in q_of.keys() {
            assert!(u >= 0 && v >= 0 && u + v <= l && (u - v).rem_euclid(3) != 2);
        }
        let bottom = (0..cc.data.len()).filter(|&i| cc.data[i].1 == 0).collect();
        Tri { d, m, l, q_of, p_uv, p_of, supp, bottom }
    }
    fn q(&self, p: Pt) -> usize {
        *self.q_of.get(&p).unwrap_or_else(|| panic!("{p:?} is not a data qubit"))
    }
    fn set<I: IntoIterator<Item = Pt>>(&self, pts: I) -> Set {
        pts.into_iter().map(|p| self.q(p)).collect()
    }
    fn plaq(&self, p: Pt) -> usize {
        self.p_of[&p]
    }
    fn is_logical(&self, s: &Set) -> bool {
        self.supp
            .iter()
            .all(|sp| sp.iter().filter(|q| s.contains(q)).count() % 2 == 0)
            && s.intersection(&self.bottom).count() % 2 == 1
    }
    fn uv_of(&self, q: usize) -> Pt {
        *self.q_of.iter().find(|(_, &i)| i == q).unwrap().0
    }
    fn rho(&self, p: Pt) -> Pt {
        (p.1, self.l - p.0 - p.1)
    }
    fn rho_k(&self, mut p: Pt, k: usize) -> Pt {
        for _ in 0..k % 3 {
            p = self.rho(p);
        }
        p
    }
    fn rot(&self, s: &Set, k: usize) -> Set {
        s.iter().map(|&q| self.q(self.rho_k(self.uv_of(q), k))).collect()
    }
    fn times(&self, s: &Set, plaquette: Pt) -> Set {
        s.symmetric_difference(&self.supp[self.plaq(plaquette)].iter().copied().collect())
            .copied()
            .collect()
    }
    /// Family I_k (k = 0..m-1): red string from the bottom at u = 3k+1 up to
    /// the left side, plus the top part of the left side.
    fn fam_i(&self, k: i32) -> Set {
        let c = 3 * k + 1;
        let mut pts = vec![(c, 0), (c, 1)];
        for t in 0..k {
            pts.push((3 * (k - t), 3 * t + 3));
            pts.push((3 * (k - t) - 1, 3 * t + 4));
        }
        pts.extend((3 * k + 3..=self.l).filter(|v| v % 3 != 1).map(|v| (0, v)));
        self.set(pts)
    }
    /// Family II_j (j = 0..m-1): right part of the bottom side plus a green
    /// string from the trapezoid (3j+2, 0) up-left to the left side.
    fn fam_ii(&self, j: i32) -> Set {
        let mut pts: Vec<Pt> = (3 * j + 3..=self.l).filter(|u| u % 3 != 2).map(|u| (u, 0)).collect();
        for t in 0..j {
            pts.push((3 * j + 1 - 3 * t, 3 * t + 1));
            pts.push((3 * j - 3 * t, 3 * t + 2));
        }
        pts.push((1, 3 * j));
        pts.push((0, 3 * j));
        self.set(pts)
    }
    fn a_side(&self) -> Set {
        self.bottom.clone()
    }
}

/// (plaquette, [(pair, logical, name)]) for the bottom side: corners (Theorem 1)
/// and all other bottom boundary plaquettes (Theorem 2).
#[allow(clippy::type_complexity)]
fn bottom_certificates(t: &Tri) -> Vec<(usize, Vec<(Set, Set, String)>)> {
    let m = t.m;
    let tr = |j: i32| (3 * j + 2, 0);
    let mut out = Vec::new();
    let pair = |p: Pt, a: usize, b: usize| t.set([(p.0 + NB[a].0, p.1 + NB[a].1), (p.0 + NB[b].0, p.1 + NB[b].1)]);
    let (a, b, c, d, e, f) = (0, 1, 2, 3, 4, 5);
    // Theorem 1: bottom-left corner (0,1) = {b=(0,2), c=(1,1), d=(1,0), e=(0,0)}.
    let p0 = (0, 1);
    out.push((
        t.plaq(p0),
        vec![
            (pair(p0, d, e), t.a_side(), "A".into()),
            (pair(p0, b, e), t.rot(&t.a_side(), 1), "rho A (left side)".into()),
            (pair(p0, c, e), t.times(&t.a_side(), tr(0)), "A.t0".into()),
        ],
    ));
    // Theorem 2: non-corner trapezoids t_j, j = 0..m-2 (positions a, b, c, f).
    for j in 0..m - 1 {
        let p = tr(j);
        let ac = if j >= 1 {
            (t.fam_ii(j), format!("II_{j}"))
        } else {
            (t.rot(&t.fam_i(m - 1), 1), format!("rho I_{}", m - 1))
        };
        out.push((
            t.plaq(p),
            vec![
                (pair(p, c, f), t.a_side(), "A".into()),
                (pair(p, a, f), t.fam_i(j), format!("I_{j}")),
                (pair(p, a, c), ac.0, ac.1),
            ],
        ));
    }
    // Theorem 2: boundary hexagons h_j = (3j, 1), j = 1..m-1: all ten pairs of {b,c,d,e,f}.
    for j in 1..m {
        let p = (3 * j, 1);
        let r2 = |s: Set| t.rot(&s, 2);
        let mut v = vec![
            (pair(p, d, e), t.a_side(), "A".to_string()),
            (pair(p, d, f), t.times(&t.a_side(), tr(j - 1)), format!("A.t{}", j - 1)),
            (pair(p, c, e), t.times(&t.a_side(), tr(j)), format!("A.t{j}")),
            (pair(p, c, f), t.times(&t.times(&t.a_side(), tr(j - 1)), tr(j)), format!("A.t{}.t{j}", j - 1)),
            (pair(p, c, d), t.fam_i(j), format!("I_{j}")),
            (pair(p, b, c), t.fam_ii(j), format!("II_{j}")),
            (pair(p, e, f), r2(t.fam_ii(m - j)), format!("rho^2 II_{}", m - j)),
        ];
        if j <= m - 2 {
            v.push((pair(p, b, d), t.times(&t.fam_ii(j), tr(j)), format!("II_{j}.t{j}")));
            v.push((pair(p, b, e), r2(t.fam_i(m - 1 - j)), format!("rho^2 I_{}", m - 1 - j)));
            v.push((pair(p, b, f), t.times(&r2(t.fam_i(m - 1 - j)), tr(j - 1)), format!("rho^2 I_{}.t{}", m - 1 - j, j - 1)));
        } else {
            let k = (3 * m - 2, 2);
            v.push((pair(p, b, d), t.times(&t.fam_i(m - 1), k), format!("I_{}.S{k:?}", m - 1)));
            v.push((pair(p, b, e), t.times(&r2(t.fam_i(0)), k), format!("rho^2 I_0.S{k:?}")));
            v.push((pair(p, b, f), t.rot(&t.fam_ii(m - 1), 1), format!("rho II_{}", m - 1)));
        }
        out.push((t.plaq(p), v));
    }
    out
}

/// Certificates for all 3d-6 boundary plaquettes (bottom ones and their rotations).
#[allow(clippy::type_complexity)]
fn all_certificates(t: &Tri) -> HashMap<usize, Vec<(Set, Set, String)>> {
    let mut all: HashMap<usize, Vec<(Set, Set, String)>> = HashMap::new();
    for (pi, certs) in bottom_certificates(t) {
        for k in 0..3 {
            let rp = t.plaq(t.rho_k(t.p_uv[pi], k));
            for (pr, l, name) in &certs {
                all.entry(rp).or_default().push((t.rot(pr, k), t.rot(l, k), format!("rho^{k}({name})")));
            }
        }
    }
    all
}

fn permutations(v: &[usize]) -> Vec<Vec<usize>> {
    if v.len() <= 1 {
        return vec![v.to_vec()];
    }
    let mut out = Vec::new();
    for i in 0..v.len() {
        let mut rest = v.to_vec();
        let x = rest.remove(i);
        for mut p in permutations(&rest) {
            p.insert(0, x);
            out.push(p);
        }
    }
    out
}

/// Multi-qubit hooks of a sequential order: suffixes of size 2..=w-2.
fn hooks(order: &[usize]) -> Vec<Set> {
    let w = order.len();
    (2..=w.saturating_sub(2)).map(|k| order[k..].iter().copied().collect()).collect()
}

/// Number of orders of `supp` none of whose hooks (or their complements) is in `malign`.
fn safe_orders(supp: &[usize], malign: &HashSet<Set>) -> usize {
    let full: Set = supp.iter().copied().collect();
    permutations(supp)
        .iter()
        .filter(|o| {
            hooks(o).iter().all(|h| {
                let c: Set = full.difference(h).copied().collect();
                !malign.contains(h) && !malign.contains(&c)
            })
        })
        .count()
}

fn is_boundary(t: &Tri, pi: usize) -> bool {
    let deg = |q: usize| t.supp.iter().filter(|s| s.contains(&q)).count();
    t.supp[pi].iter().any(|&q| deg(q) < 3)
}

#[test]
fn families_are_weight_d_logicals() {
    for d in (3..=41).step_by(2) {
        let t = Tri::new(d);
        assert!(t.is_logical(&t.a_side()) && t.a_side().len() == d);
        for k in 0..t.m {
            for (name, s) in [("I", t.fam_i(k)), ("II", t.fam_ii(k))] {
                for r in 0..3 {
                    let s = t.rot(&s, r);
                    assert!(t.is_logical(&s), "d={d} rho^{r} {name}_{k} not a logical");
                    assert_eq!(s.len(), d, "d={d} rho^{r} {name}_{k}");
                }
            }
        }
    }
}

#[test]
fn rotation_is_a_code_automorphism() {
    for d in (3..=41).step_by(2) {
        let t = Tri::new(d);
        for (pi, sp) in t.supp.iter().enumerate() {
            let rp = t.plaq(t.rho(t.p_uv[pi]));
            let img: Set = sp.iter().map(|&q| t.q(t.rho(t.uv_of(q)))).collect();
            assert_eq!(img, t.supp[rp].iter().copied().collect::<Set>());
        }
        assert_eq!(t.rot(&t.a_side(), 3), t.a_side());
    }
}

#[test]
fn corner_and_boundary_lemmas_hold_with_explicit_certificates() {
    for d in (3..=41).step_by(2) {
        let t = Tri::new(d);
        let certs = all_certificates(&t);
        let nbnd = (0..t.supp.len()).filter(|&p| is_boundary(&t, p)).count();
        assert_eq!(nbnd, 3 * d - 6, "d={d}: boundary plaquettes");
        assert_eq!(certs.len(), 3 * d - 6, "d={d}: every boundary plaquette certified");
        for (&pi, list) in &certs {
            assert!(is_boundary(&t, pi));
            let sp: Set = t.supp[pi].iter().copied().collect();
            let mut malign = HashSet::new();
            for (pr, l, name) in list {
                assert!(t.is_logical(l) && l.len() == d, "d={d} {name}");
                let meet: Set = l.intersection(&sp).copied().collect();
                assert_eq!(&meet, pr, "d={d} plaquette {:?} {name}", t.p_uv[pi]);
                malign.insert(pr.clone());
            }
            assert_eq!(safe_orders(&t.supp[pi], &malign), 0, "d={d} plaquette {:?}", t.p_uv[pi]);
        }
    }
}

/// All weight-d logicals (as bitmasks), by Gray-code enumeration of the
/// stabilizer coset of the bottom side; also asserts the code distance is d.
fn weight_d_logicals(t: &Tri) -> Vec<u128> {
    let n = t.q_of.len();
    assert!(n <= 128);
    let rows: Vec<u128> = t.supp.iter().map(|s| s.iter().fold(0u128, |a, &q| a | 1 << q)).collect();
    let mut cur: u128 = t.bottom.iter().fold(0, |a, &q| a | 1 << q);
    let mut out = vec![cur];
    for k in 1u64..(1u64 << rows.len()) {
        cur ^= rows[k.trailing_zeros() as usize];
        let w = cur.count_ones() as usize;
        assert!(w >= t.d, "logical of weight {w} < d");
        if w == t.d {
            out.push(cur);
        }
    }
    out
}

fn exact_check(d: usize, expect_logicals: usize) {
    let t = Tri::new(d);
    let logs = weight_d_logicals(&t);
    eprintln!("d={d}: {} weight-d logicals", logs.len());
    assert_eq!(logs.len(), expect_logicals, "number of weight-{d} logicals");
    let certs = all_certificates(&t);
    for (pi, sp) in t.supp.iter().enumerate() {
        // exact malign classes: a hook residual H (|H| = 2, 3 or w-2) gives a logical
        // with d-1 faults iff H or supp\H lies inside a weight-d logical
        let mut malign = HashSet::new();
        for size in 2..=3 {
            for o in permutations(sp) {
                let h: Set = o[..size].iter().copied().collect();
                let hm = h.iter().fold(0u128, |a, &q| a | 1 << q);
                if logs.iter().any(|&l| l & hm == hm) {
                    malign.insert(h);
                }
            }
        }
        assert!(malign.iter().all(|h| h.len() == 2), "d={d}: a weight-3 hook is malign");
        let safe = safe_orders(sp, &malign);
        if is_boundary(&t, pi) {
            assert_eq!(safe, 0, "d={d} boundary plaquette {:?}", t.p_uv[pi]);
            for (pr, _, _) in &certs[&pi] {
                assert!(malign.contains(pr));
            }
        } else {
            assert_eq!(safe, 48, "d={d} interior plaquette {:?}", t.p_uv[pi]);
        }
    }
}

#[test]
fn malign_hooks_exact_d5_d7() {
    exact_check(5, 36);
    exact_check(7, 140);
}

#[test]
#[ignore = "2^30 stabilizer products; ~20 s in release"]
fn malign_hooks_exact_d9() {
    exact_check(9, 464);
}

fn load_schedule(text: &str) -> ColorSchedule {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: Vec<u8> = l.split_whitespace().map(|t| t.parse().unwrap()).collect();
            [v[0], v[1], v[2], v[3], v[4], v[5]]
        })
        .collect()
}

/// Theorem 2 at circuit level: for every boundary plaquette, the actual X-half
/// order of `s` has a hook which, with d-2 single data errors, is an
/// undetectable logical built from mechanisms of the circuit's own DEM.
fn circuit_witness(d: usize, s: &ColorSchedule) {
    let t = Tri::new(d);
    let cc = ColorCode::new(d);
    let np = cc.plaquettes.len();
    let mem = cc.memory(s, 1, ColorNoise::Cnot(0.001));
    let z = mem.z_sector();
    let index: HashMap<(Vec<u32>, bool), usize> =
        z.dets.iter().cloned().zip(z.obs.iter().copied()).enumerate().map(|(i, k)| (k, i)).collect();
    // a clean X error on data set Q during the X half of the last round flips the
    // final-layer detectors (layer 1) of the plaquettes meeting Q oddly
    let sig = |qs: &Set| -> (Vec<u32>, bool) {
        let dets = (0..np)
            .filter(|&pi| t.supp[pi].iter().filter(|q| qs.contains(q)).count() % 2 == 1)
            .map(|pi| (np + pi) as u32)
            .collect();
        (dets, qs.intersection(&t.bottom).count() % 2 == 1)
    };
    let certs = all_certificates(&t);
    for (&pi, list) in &certs {
        let p = &cc.plaquettes[pi];
        let mut pos: Vec<usize> = (0..6).filter(|&k| p.data[k].is_some()).collect();
        pos.sort_by_key(|&k| s[pi][k]);
        let order: Vec<usize> = pos.iter().map(|&k| p.data[k].unwrap()).collect();
        let sp: Set = order.iter().copied().collect();
        let mut found = false;
        'h: for h in hooks(&order) {
            let c: Set = sp.difference(&h).copied().collect();
            for (pr, l, name) in list {
                if *pr != h && *pr != c {
                    continue;
                }
                // hook H plus singles on L \ pr: product L (pr = H) or L.S_p (pr = supp \ H)
                let singles: Set = l.symmetric_difference(pr).copied().collect();
                let mut faults = vec![index.get(&sig(&h)).copied()];
                faults.extend(singles.iter().map(|&q| index.get(&sig(&[q].into())).copied()));
                assert!(faults.iter().all(|f| f.is_some()), "d={d}: missing mechanism ({name})");
                let faults: Vec<usize> = faults.into_iter().map(|f| f.unwrap()).collect();
                assert!(faults.iter().all(|&f| z.pure[f]));
                let mut acc: BTreeSet<u32> = BTreeSet::new();
                let mut ob = false;
                for &f in &faults {
                    for &dd in &z.dets[f] {
                        if !acc.remove(&dd) {
                            acc.insert(dd);
                        }
                    }
                    ob ^= z.obs[f];
                }
                assert!(acc.is_empty() && ob, "d={d}: not an undetectable logical ({name})");
                assert_eq!(faults.len(), d - 1, "d={d}: {name}, hook {h:?}");
                found = true;
                break 'h;
            }
        }
        assert!(found, "d={d}: no certified hook at plaquette {:?}", t.p_uv[pi]);
    }
}

#[test]
fn boundary_lemma_circuit_level_d5_d7() {
    for d in [5, 7] {
        let cc = ColorCode::new(d);
        circuit_witness(d, &cc.uniform_schedule(KF_SCHEDULE));
        circuit_witness(d, &cc.uniform_schedule([TRI_OPTIMAL; 3]));
    }
}

#[test]
fn boundary_lemma_circuit_level_d9_global_schedule() {
    let s = load_schedule(include_str!("../research/data/colour-global/schedules/d9_global_D8.sched"));
    circuit_witness(9, &s);
    circuit_witness(9, &ColorCode::new(9).uniform_schedule(KF_SCHEDULE));
}

/// Theorem 3: drop every Z-sector mechanism whose time-projected syndrome is
/// not that of an error of weight <= 1. The rest has circuit distance >= d
/// (here exactly d). Every mechanism kept by the filter has a weight <= 1
/// residual, so this is the theorem's hypothesis applied to the real DEM.
#[test]
fn hook_free_residuals_give_full_distance() {
    for (d, rounds) in [(3, 3), (5, 2), (5, 5), (7, 2)] {
        let t = Tri::new(d);
        let cc = ColorCode::new(d);
        let np = cc.plaquettes.len();
        let mem = cc.memory(&cc.uniform_schedule(KF_SCHEDULE), rounds, ColorNoise::Cnot(0.001));
        let z = mem.z_sector();
        // (syndrome, observable parity) of every single-qubit X error
        let single: HashSet<(Vec<u32>, bool)> = (0..t.q_of.len())
            .map(|q| {
                let syn = (0..np).filter(|&pi| t.supp[pi].contains(&q)).map(|pi| pi as u32).collect();
                (syn, t.bottom.contains(&q))
            })
            .collect();
        let (mut dets, mut obs) = (Vec::new(), Vec::new());
        let mut dropped = 0;
        for (ds, &ob) in z.dets.iter().zip(&z.obs) {
            let mut proj = vec![false; np];
            for &dd in ds {
                proj[z.plaquette[dd as usize]] ^= true;
            }
            let proj: Vec<u32> = (0..np).filter(|&p| proj[p]).map(|p| p as u32).collect();
            if (proj.is_empty() && !ob) || single.contains(&(proj, ob)) {
                dets.push(ds.clone());
                obs.push(ob);
            } else {
                dropped += 1;
            }
        }
        assert!(dropped > 0, "the K-F circuit has multi-qubit hooks");
        let r = min_logical(z.num_detectors, &dets, &obs, d, 1, u64::MAX);
        assert_eq!(r.weight, Some(d), "d={d} rounds={rounds}");
    }
}
