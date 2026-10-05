//! Two-block group-algebra codes over non-abelian groups
//! (`qec::group_algebra`): commutation, ranks, agreement with
//! `qec::bicycle` on abelian groups, symmetry-rooted distance vs plain and
//! brute-force search, the claimed code equivalences, and completeness of
//! the enumeration against an independent union-find over all pairs.
use qsim_lab::qec::bicycle::{
    all_roots, is_nontrivial_logical, logical_masks, min_weight_logical, DistanceOpts, Gf2Mat,
    SearchOutcome, TwoBlockCode,
};
use qsim_lab::qec::group_algebra::{Enumeration, FiniteGroup, GroupCode};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet};

fn commute(c: &GroupCode) -> bool {
    let (hx, hz) = (c.hx(), c.hz());
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

fn random_subset(rng: &mut StdRng, n: usize, w: usize) -> Vec<u16> {
    let mut v: Vec<u16> = (0..n as u16).collect();
    v.shuffle(rng);
    v.truncate(w);
    v
}

#[test]
fn groups_build_and_aut_counts() {
    // |Aut(S3)| = 6, |Aut(D4)| = 8, |Aut(Z7:Z3)| = 42, |Aut(Q8)|: not metacyclic split
    assert_eq!(FiniteGroup::metacyclic(3, 2, 2).unwrap().aut_gens.len(), 6);
    assert_eq!(FiniteGroup::metacyclic(4, 2, 3).unwrap().aut_gens.len(), 8);
    assert_eq!(FiniteGroup::metacyclic(7, 3, 2).unwrap().aut_gens.len(), 42);
    assert!(FiniteGroup::metacyclic(7, 3, 3).is_err());
    let s3 = FiniteGroup::metacyclic(3, 2, 2).unwrap();
    assert!(!s3.is_abelian());
    assert_eq!(s3.center_order(), 1);
    let p = s3.direct_product(&FiniteGroup::abelian(2, 1));
    assert_eq!(p.order, 12);
    assert_eq!(p.center_order(), 2);
}

#[test]
fn matches_bicycle_on_abelian_groups() {
    // same matrices, k and d as qec::bicycle for Z_l x Z_m
    let c0 = TwoBlockCode::parse(6, 6, "x^3+y+y^2", "y^3+x+x^2");
    let g = FiniteGroup::abelian(6, 6);
    let idx = |(i, j): (usize, usize)| (i * 6 + j) as u16;
    let a: Vec<u16> = c0.a.iter().map(|&t| idx(t)).collect();
    let b: Vec<u16> = c0.b.iter().map(|&t| idx(t)).collect();
    let c = GroupCode::new(&g, &a, &b);
    assert_eq!(c.hx(), c0.hx());
    assert_eq!(c.hz(), c0.hz());
    assert_eq!(c.k(), 12);
    assert_eq!(c.distance_roots().len(), 2);
    let (dz, dx) = c.distances(&DistanceOpts::default());
    assert_eq!((dz.exact(), dx.exact()), (Some(6), Some(6)));
}

#[test]
fn checks_commute_and_ranks_match() {
    let mut rng = StdRng::seed_from_u64(3);
    for (m, k, q) in [
        (3, 2, 2),
        (5, 4, 2),
        (7, 3, 2),
        (9, 2, 8),
        (13, 3, 3),
        (4, 4, 3),
    ] {
        let g = FiniteGroup::metacyclic(m, k, q).unwrap();
        for _ in 0..20 {
            let w = rng.random_range(2..5usize);
            let a = random_subset(&mut rng, g.order, w);
            let wb = rng.random_range(2..5usize);
            let b = random_subset(&mut rng, g.order, wb);
            let c = GroupCode::new(&g, &a, &b);
            assert!(commute(&c), "{m} {k} {q} {a:?} {b:?}");
            let (rx, rz) = c.ranks();
            assert_eq!((rx, rz), (c.hx().rank(), c.hz().rank()));
        }
    }
}

/// (k, d_Z, d_X) by brute force over all errors (tiny codes only).
fn brute(c: &GroupCode) -> (usize, usize, usize) {
    let (hx, hz) = (c.hx(), c.hz());
    let n = c.n();
    let d = |h: &Gf2Mat, o: &Gf2Mat| {
        let (masks, k) = logical_masks(h, o);
        if k == 0 {
            return usize::MAX;
        }
        let mut best = usize::MAX;
        for e in 1u32..(1 << n) {
            let w = e.count_ones() as usize;
            if w >= best {
                continue;
            }
            let sup: Vec<usize> = (0..n).filter(|&q| e >> q & 1 == 1).collect();
            if is_nontrivial_logical(h, &masks, &sup) {
                best = w;
            }
        }
        best
    };
    (c.k(), d(&hx, &hz), d(&hz, &hx))
}

#[test]
fn symmetric_roots_match_plain_and_brute_force() {
    let mut rng = StdRng::seed_from_u64(11);
    let groups = [
        FiniteGroup::metacyclic(3, 2, 2).unwrap(),
        FiniteGroup::metacyclic(4, 2, 3).unwrap(),
        FiniteGroup::metacyclic(5, 2, 4).unwrap(),
        FiniteGroup::metacyclic(7, 3, 2).unwrap(),
        FiniteGroup::metacyclic(3, 4, 2).unwrap(),
        FiniteGroup::metacyclic(9, 2, 8).unwrap(),
        FiniteGroup::metacyclic(3, 2, 2)
            .unwrap()
            .direct_product(&FiniteGroup::abelian(3, 1)),
    ];
    let mut tested = 0;
    for g in &groups {
        let mut found = 0;
        for _ in 0..400 {
            if found == 6 {
                break;
            }
            let a = random_subset(&mut rng, g.order, 3);
            let b = random_subset(&mut rng, g.order, 3);
            let c = GroupCode::new(g, &a, &b);
            if c.k() == 0 {
                continue;
            }
            found += 1;
            let (hx, hz) = (c.hx(), c.hz());
            let roots = c.distance_roots();
            for (h, o) in [(&hx, &hz), (&hz, &hx)] {
                let (masks, _) = logical_masks(h, o);
                let w = |r: &[(usize, Vec<usize>)]| match min_weight_logical(
                    h,
                    &masks,
                    r,
                    1,
                    c.n(),
                    u64::MAX,
                )
                .0
                {
                    SearchOutcome::Found(w, s) => {
                        assert!(is_nontrivial_logical(h, &masks, &s));
                        w
                    }
                    o => panic!("{o:?}"),
                };
                assert_eq!(w(&roots), w(&all_roots(c.n())), "{} {a:?} {b:?}", g.label);
            }
            if c.n() <= 16 {
                let (k, dz, dx) = brute(&c);
                let (rz, rx) = c.distances(&DistanceOpts::default());
                assert_eq!(k, rz.k);
                assert_eq!(
                    (Some(dz), Some(dx)),
                    (rz.exact(), rx.exact()),
                    "{a:?} {b:?}"
                );
            }
            tested += 1;
        }
    }
    assert!(tested >= 25, "{tested}");
}

/// Applies a random equivalence and checks (k, d_Z, d_X) transform as
/// claimed in the module docs.
#[test]
fn equivalences_preserve_parameters() {
    let mut rng = StdRng::seed_from_u64(5);
    let g = FiniteGroup::metacyclic(7, 3, 2).unwrap();
    let n = g.order;
    let inv = |s: &[u16]| -> Vec<u16> { s.iter().map(|&x| g.inv[x as usize]).collect() };
    let tr = |s: &[u16], u: usize, w: usize| -> Vec<u16> {
        s.iter()
            .map(|&x| g.mul(g.mul(u, x as usize), w) as u16)
            .collect()
    };
    let params = |a: &[u16], b: &[u16]| {
        let c = GroupCode::new(&g, a, b);
        let (dz, dx) = c.distances(&DistanceOpts::default());
        (c.k(), dz.exact().unwrap_or(0), dx.exact().unwrap_or(0))
    };
    let mut done = 0;
    while done < 8 {
        let a = random_subset(&mut rng, n, 3);
        let b = random_subset(&mut rng, n, 3);
        let (k, dz, dx) = params(&a, &b);
        if k == 0 {
            continue;
        }
        done += 1;
        let (u, w, v, t) = (
            rng.random_range(0..n),
            rng.random_range(0..n),
            rng.random_range(0..n),
            rng.random_range(0..n),
        );
        assert_eq!(params(&tr(&a, u, w), &tr(&b, v, t)), (k, dz, dx));
        let s = &g.aut_gens[rng.random_range(0..g.aut_gens.len())];
        let ap: Vec<u16> = a.iter().map(|&x| s[x as usize]).collect();
        let bp: Vec<u16> = b.iter().map(|&x| s[x as usize]).collect();
        assert_eq!(params(&ap, &bp), (k, dz, dx));
        assert_eq!(params(&inv(&b), &inv(&a)), (k, dz, dx));
        assert_eq!(params(&b, &a), (k, dx, dz));
        assert_eq!(params(&inv(&a), &inv(&b)), (k, dx, dz));
    }
}

/// The number of enumerated orbits equals the number of orbits of ALL
/// pairs of 3-subsets under the generators of the equivalence group
/// (translations by group generators on either side of `A` and of `B`,
/// automorphisms, swap, inversion), computed independently by
/// union-find; parameters are constant on random pairs of one orbit.
#[test]
fn enumeration_matches_brute_force_orbits() {
    for g in [
        FiniteGroup::metacyclic(3, 2, 2).unwrap(),
        FiniteGroup::metacyclic(4, 2, 3).unwrap(),
        FiniteGroup::metacyclic(3, 4, 2).unwrap(),
        FiniteGroup::metacyclic(7, 3, 2).unwrap(),
        FiniteGroup::abelian(6, 1),
        FiniteGroup::abelian(3, 3),
    ] {
        let n = g.order;
        let e = Enumeration::new(&g, 3);
        let orbits = e.pair_orbits(&g);
        let m = e.reps.len();
        assert_eq!(orbits.iter().map(|o| o.2).sum::<usize>(), m * m);
        for (i, r) in e.reps.iter().enumerate() {
            assert_eq!(e.class(r) as usize, i);
        }
        assert_eq!(e.class_of.len(), (n - 1) * (n - 2) / 2);
        // brute force over all 3-subsets
        let mut subs: Vec<[u16; 3]> = Vec::new();
        for x in 0..n {
            for y in x + 1..n {
                for z in y + 1..n {
                    subs.push([x as u16, y as u16, z as u16]);
                }
            }
        }
        let index: HashMap<[u16; 3], usize> =
            subs.iter().enumerate().map(|(i, s)| (*s, i)).collect();
        let key = |v: [usize; 3]| {
            let mut s = [v[0] as u16, v[1] as u16, v[2] as u16];
            s.sort_unstable();
            index[&s]
        };
        let ms = subs.len();
        let map = |f: &dyn Fn(usize) -> usize| -> Vec<usize> {
            subs.iter()
                .map(|s| key([f(s[0] as usize), f(s[1] as usize), f(s[2] as usize)]))
                .collect()
        };
        let gens: Vec<usize> = (1..n).collect();
        let left: Vec<Vec<usize>> = gens.iter().map(|&u| map(&|x| g.mul(u, x))).collect();
        let right: Vec<Vec<usize>> = gens.iter().map(|&u| map(&|x| g.mul(x, u))).collect();
        let auts: Vec<Vec<usize>> = g.aut_gens.iter().map(|s| map(&|x| s[x] as usize)).collect();
        let inv = map(&|x| g.inv(x));
        let mut parent: Vec<usize> = (0..ms * ms).collect();
        fn find(p: &mut [usize], mut x: usize) -> usize {
            while p[x] != x {
                p[x] = p[p[x]];
                x = p[x];
            }
            x
        }
        let join = |p: &mut Vec<usize>, x: usize, y: usize| {
            let (a, b) = (find(p, x), find(p, y));
            if a != b {
                p[a.max(b)] = a.min(b);
            }
        };
        for i in 0..ms {
            for j in 0..ms {
                let x = i * ms + j;
                for t in left.iter().chain(right.iter()) {
                    join(&mut parent, x, t[i] * ms + j);
                    join(&mut parent, x, i * ms + t[j]);
                }
                for s in &auts {
                    join(&mut parent, x, s[i] * ms + s[j]);
                }
                join(&mut parent, x, j * ms + i);
                join(&mut parent, x, inv[i] * ms + inv[j]);
            }
        }
        let roots: HashSet<usize> = (0..ms * ms).map(|x| find(&mut parent, x)).collect();
        assert_eq!(roots.len(), orbits.len(), "{}", g.label);
        // parameters agree between a random pair and its orbit representative
        let param = |a: &[u16], b: &[u16]| {
            let c = GroupCode::new(&g, a, b);
            let k = c.k();
            if k == 0 {
                return (0, 0, 0);
            }
            let (dz, dx) = c.distances(&DistanceOpts::default());
            let (x, y) = (dz.exact().unwrap(), dx.exact().unwrap());
            (k, x.min(y), x.max(y))
        };
        let mut rep_of_root: HashMap<usize, (usize, usize, usize)> = HashMap::new();
        for &(i, j, _) in &orbits {
            let (a, b) = (&e.reps[i as usize], &e.reps[j as usize]);
            let x = key([a[0] as usize, a[1] as usize, a[2] as usize]) * ms
                + key([b[0] as usize, b[1] as usize, b[2] as usize]);
            let r = find(&mut parent, x);
            assert!(
                rep_of_root.insert(r, param(a, b)).is_none(),
                "two reps in one orbit"
            );
        }
        let mut rng = StdRng::seed_from_u64(9);
        for _ in 0..40 {
            let (i, j) = (rng.random_range(0..ms), rng.random_range(0..ms));
            let r = find(&mut parent, i * ms + j);
            let p = param(&subs[i], &subs[j]);
            assert_eq!(
                rep_of_root[&r], p,
                "{} {:?} {:?}",
                g.label, subs[i], subs[j]
            );
        }
    }
}

/// Coset codes: `H = {1}` gives the group code; every random coset code with
/// `B` in the normaliser has commuting checks; the symmetry-rooted distance
/// equals the plain search; a normal `H` gives the code of `G / H`.
#[test]
fn coset_codes() {
    use qsim_lab::qec::group_algebra::{normalizer_quotient, orbit_roots, CosetCode, Cosets};
    let mut rng = StdRng::seed_from_u64(21);
    // H = {1}
    let g = FiniteGroup::metacyclic(7, 3, 2).unwrap();
    let triv = Cosets::new(&g, &[0]).unwrap();
    let (a, b) = (vec![0u16, 1, 5], vec![0u16, 3, 10]);
    let cc = CosetCode::new(&g, &triv, &a, &b).explicit();
    let gc = GroupCode::new(&g, &a, &b);
    for i in 0..g.order {
        assert_eq!(cc.x[i], gc.x_check(i));
        assert_eq!(cc.z[i], gc.z_check(i));
    }
    // non-normal H in S3 x Z_m and D4 x Z_3, random A and B in N_G(H)
    let s3 = FiniteGroup::metacyclic(3, 2, 2).unwrap();
    let d4 = FiniteGroup::metacyclic(4, 2, 3).unwrap();
    let mut tested = 0;
    for (k, m) in [(&s3, 4usize), (&s3, 5), (&d4, 3), (&s3, 7)] {
        let g = FiniteGroup::abelian(m, 1).direct_product(k);
        // H = <s> with s the reflection (element (0, 0 + m_k * 1) = index k.order/2? use j = 1, i = 0)
        let s = (k.order / 2) as u16; // r^0 s^1 in metacyclic indexing (j m + i)
        let h: Vec<u16> = (0..g.order as u16)
            .filter(|&x| g.subgroup(&[s])[x as usize])
            .collect();
        let cos = Cosets::new(&g, &h).unwrap();
        assert!(!cos.is_normal(&g));
        let nor: Vec<u16> = (0..g.order as u16)
            .filter(|&x| cos.normalizer(&g)[x as usize])
            .collect();
        let (q, lift) = normalizer_quotient(&g, &cos);
        assert_eq!(q.order * h.len(), nor.len());
        assert_eq!(lift[0], 0);
        let mut found = 0;
        for _ in 0..300 {
            if found == 5 {
                break;
            }
            let a = random_subset(&mut rng, g.order, 3);
            let mut bn = nor.clone();
            bn.shuffle(&mut rng);
            let b: Vec<u16> = bn[..3].to_vec();
            let c = CosetCode::new(&g, &cos, &a, &b);
            let ex = c.explicit();
            assert!(ex.commutes(), "{} {a:?} {b:?}", g.label);
            let (rx, rz) = ex.ranks();
            assert_eq!((rx, rz), (ex.hx().rank(), ex.hz().rank()));
            if ex.n - rx - rz == 0 || ex.n > 44 {
                continue;
            }
            found += 1;
            let perms = c.translation_automorphisms();
            assert!(perms.iter().all(|p| ex.is_automorphism(p)));
            let roots = orbit_roots(ex.n, &perms);
            let (hx, hz) = (ex.hx(), ex.hz());
            for (hc, ho) in [(&hx, &hz), (&hz, &hx)] {
                let (masks, _) = logical_masks(hc, ho);
                let w = |r: &[(usize, Vec<usize>)]| match min_weight_logical(
                    hc,
                    &masks,
                    r,
                    1,
                    ex.n,
                    u64::MAX,
                )
                .0
                {
                    SearchOutcome::Found(w, s) => {
                        assert!(is_nontrivial_logical(hc, &masks, &s));
                        w
                    }
                    o => panic!("{o:?}"),
                };
                assert_eq!(w(&roots), w(&all_roots(ex.n)));
            }
            tested += 1;
        }
    }
    assert!(tested >= 8, "{tested}");
    // normal H: Z_6 x S3 with H = Z_2 (central) gives the code over Z_3 x S3
    let g = FiniteGroup::abelian(6, 1).direct_product(&s3);
    let z2: Vec<u16> = vec![0, (3 * s3.order) as u16];
    let cos = Cosets::new(&g, &z2).unwrap();
    assert!(cos.is_normal(&g));
    let quo = FiniteGroup::abelian(3, 1).direct_product(&s3);
    // (z, x) -> (z mod 3, x)
    let down = |e: u16| ((e as usize / s3.order % 3) * s3.order + e as usize % s3.order) as u16;
    for _ in 0..10 {
        let a = random_subset(&mut rng, g.order, 3);
        let b = random_subset(&mut rng, g.order, 3);
        let (ad, bd): (Vec<u16>, Vec<u16>) = (
            a.iter().map(|&x| down(x)).collect(),
            b.iter().map(|&x| down(x)).collect(),
        );
        if HashSet::<u16>::from_iter(ad.iter().copied()).len() < 3
            || HashSet::<u16>::from_iter(bd.iter().copied()).len() < 3
        {
            continue;
        }
        let ex = CosetCode::new(&g, &cos, &a, &b).explicit();
        let gq = GroupCode::new(&quo, &ad, &bd);
        assert_eq!(ex.k(), gq.k());
    }
}

/// The group generated by permutations of `0..d` (GAP's convention: `(p q)(i) =
/// q(p(i))`), elements numbered by sorting the permutations (identity first).
fn perm_group(gens: &[&[u8]]) -> (FiniteGroup, HashMap<Vec<u8>, u16>) {
    let d = gens[0].len();
    let id: Vec<u8> = (0..d as u8).collect();
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    seen.insert(id.clone());
    let mut stack = vec![id];
    while let Some(p) = stack.pop() {
        for g in gens {
            let q: Vec<u8> = (0..d).map(|i| g[p[i] as usize]).collect();
            if seen.insert(q.clone()) {
                stack.push(q);
            }
        }
    }
    let mut els: Vec<Vec<u8>> = seen.into_iter().collect();
    els.sort();
    let index: HashMap<Vec<u8>, u16> = els
        .iter()
        .enumerate()
        .map(|(i, p)| (p.clone(), i as u16))
        .collect();
    let n = els.len();
    let mut mul = vec![0u16; n * n];
    for (i, p) in els.iter().enumerate() {
        for (j, q) in els.iter().enumerate() {
            let r: Vec<u8> = (0..d).map(|t| q[p[t] as usize]).collect();
            mul[i * n + j] = index[&r];
        }
    }
    (
        FiniteGroup::from_table(n, mul, vec![], "perm").unwrap(),
        index,
    )
}

/// Rebuilds a code from a permutation representation of its group (printed by
/// `research/data/code-discovery-2/perm_rep.g`) and checks [[n, k, d]] exactly
/// in both sectors.
fn pin(gens: &[&[u8]], a: &[&[u8]], b: &[&[u8]], n: usize, k: usize, d: usize) {
    let (g, index) = perm_group(gens);
    let el = |p: &[u8]| index[&p.to_vec()];
    let (av, bv): (Vec<u16>, Vec<u16>) = (
        a.iter().map(|p| el(p)).collect(),
        b.iter().map(|p| el(p)).collect(),
    );
    let c = GroupCode::new(&g, &av, &bv);
    assert_eq!(c.n(), n);
    assert!(c.is_connected());
    assert_eq!(c.k(), k);
    let (dz, dx) = c.distances(&DistanceOpts::default());
    assert_eq!(
        (dz.exact(), dx.exact()),
        (Some(d), Some(d)),
        "{dz:?} {dx:?}"
    );
}

/// [[288,16,16]] over SmallGroup(144,167) = Z6 x (C3 : D8): k d^2/n = 14.2,
/// above every weight-6 code in published papers with n <= 288 (the same
/// parameters were posted to the qLDPC challenge board with d only an upper
/// bound; see code-discovery-2.md). About 4 s in release.
#[test]
fn new_code_288_16_16() {
    pin(
        &[
            &[1, 0, 2, 4, 3, 5, 6, 7, 9, 8, 11, 10],
            &[1, 0, 2, 3, 4, 5, 6, 7, 10, 11, 8, 9],
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 10, 9],
            &[0, 1, 2, 3, 4, 6, 7, 5, 8, 9, 10, 11],
            &[0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 8, 9],
            &[0, 1, 3, 4, 2, 5, 6, 7, 8, 9, 10, 11],
        ],
        &[
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            &[1, 0, 2, 4, 3, 6, 7, 5, 9, 8, 11, 10],
            &[0, 1, 3, 4, 2, 7, 5, 6, 8, 11, 10, 9],
        ],
        &[
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            &[0, 1, 3, 4, 2, 6, 7, 5, 10, 11, 8, 9],
            &[0, 1, 2, 4, 3, 7, 5, 6, 9, 10, 11, 8],
        ],
        288,
        16,
        16,
    );
}

/// [[192,12,14]] over SmallGroup(96,17) = C3 : (Q8 : C4): k d^2/n = 12.25
/// (also in Lin & Pryadko's 2BGA dataset, with a randomized distance).
#[test]
fn new_code_192_12_14() {
    pin(
        &[
            &[0, 2, 1, 6, 5, 4, 3, 8, 10, 14, 12, 9, 7, 11, 13],
            &[0, 1, 2, 4, 5, 6, 3, 7, 14, 13, 10, 12, 11, 9, 8],
            &[0, 1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14],
            &[0, 1, 2, 5, 6, 3, 4, 9, 11, 10, 13, 12, 14, 7, 8],
            &[0, 1, 2, 3, 4, 5, 6, 10, 12, 13, 7, 14, 8, 9, 11],
            &[1, 2, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        ],
        &[
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            &[0, 2, 1, 6, 5, 4, 3, 8, 10, 14, 12, 9, 7, 11, 13],
            &[1, 2, 0, 4, 5, 6, 3, 7, 14, 13, 10, 12, 11, 9, 8],
        ],
        &[
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            &[1, 2, 0, 5, 6, 3, 4, 9, 11, 10, 13, 12, 14, 7, 8],
            &[2, 0, 1, 4, 5, 6, 3, 9, 8, 7, 13, 14, 12, 10, 11],
        ],
        192,
        12,
        14,
    );
}

/// [[192,16,12]] (SmallGroup(96,12)) and [[200,16,12]] (D10 x D10), both in
/// Lin & Pryadko's dataset with randomized distances, and the new Pareto point
/// [[224,18,12]] (C7 x ((C4 x C2) : C2)).
#[test]
fn new_codes_k16_k18_d12() {
    pin(
        &[
            &[0, 2, 1, 4, 3, 10, 8, 9, 6, 7, 5],
            &[0, 1, 2, 6, 7, 9, 3, 8, 10, 5, 4],
            &[0, 1, 2, 3, 8, 5, 6, 10, 4, 9, 7],
            &[0, 1, 2, 5, 7, 6, 9, 8, 10, 3, 4],
            &[0, 1, 2, 6, 8, 9, 3, 10, 4, 5, 7],
            &[1, 2, 0, 3, 4, 5, 6, 7, 8, 9, 10],
        ],
        &[
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            &[0, 2, 1, 7, 6, 4, 10, 5, 3, 8, 9],
            &[1, 0, 2, 8, 3, 7, 4, 9, 6, 10, 5],
        ],
        &[
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            &[1, 2, 0, 6, 10, 9, 3, 4, 7, 5, 8],
            &[2, 0, 1, 5, 8, 6, 9, 10, 4, 3, 7],
        ],
        192,
        16,
        12,
    );
    pin(
        &[
            &[0, 1, 2, 3, 4, 5, 9, 8, 7, 6],
            &[0, 4, 3, 2, 1, 5, 6, 7, 8, 9],
            &[1, 2, 3, 4, 0, 5, 6, 7, 8, 9],
            &[0, 1, 2, 3, 4, 6, 7, 8, 9, 5],
        ],
        &[
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            &[1, 2, 3, 4, 0, 5, 9, 8, 7, 6],
            &[2, 3, 4, 0, 1, 6, 7, 8, 9, 5],
        ],
        &[
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            &[0, 4, 3, 2, 1, 6, 7, 8, 9, 5],
            &[1, 2, 3, 4, 0, 7, 8, 9, 5, 6],
        ],
        200,
        16,
        12,
    );
    pin(
        &[
            &[1, 2, 3, 0, 4, 7, 6, 5, 8, 9, 10, 11, 12, 13, 14],
            &[0, 1, 2, 3, 5, 4, 7, 6, 8, 9, 10, 11, 12, 13, 14],
            &[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 8],
            &[0, 1, 2, 3, 6, 7, 4, 5, 8, 9, 10, 11, 12, 13, 14],
            &[2, 3, 0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        ],
        &[
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            &[1, 2, 3, 0, 4, 7, 6, 5, 9, 10, 11, 12, 13, 14, 8],
            &[0, 1, 2, 3, 5, 4, 7, 6, 11, 12, 13, 14, 8, 9, 10],
        ],
        &[
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            &[2, 3, 0, 1, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 8],
            &[0, 1, 2, 3, 6, 7, 4, 5, 11, 12, 13, 14, 8, 9, 10],
        ],
        224,
        18,
        12,
    );
}

/// [[288,34,8]] over A4 x A4 (SmallGroup(144,184)), a new high-rate Pareto point.
#[test]
fn new_code_288_34_8() {
    pin(
        &[
            &[0, 3, 1, 2, 4, 5, 6, 7],
            &[0, 1, 2, 3, 4, 7, 5, 6],
            &[0, 1, 2, 3, 5, 4, 7, 6],
            &[0, 1, 2, 3, 6, 7, 4, 5],
            &[1, 0, 3, 2, 4, 5, 6, 7],
            &[2, 3, 0, 1, 4, 5, 6, 7],
        ],
        &[
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[0, 3, 1, 2, 5, 4, 7, 6],
            &[1, 3, 2, 0, 6, 7, 4, 5],
        ],
        &[
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 0, 3, 2, 4, 7, 5, 6],
            &[3, 2, 1, 0, 5, 7, 6, 4],
        ],
        288,
        34,
        8,
    );
}
