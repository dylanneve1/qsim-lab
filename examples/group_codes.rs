//! Two-block group-algebra codes over arbitrary finite groups (including
//! non-abelian ones): exhaustive search against the known frontier, and
//! exact parameters of single codes. See `research/qec/code-discovery-2.md`.
//!
//! ```text
//! group_codes search <threshold.tsv> <threads> <node_limit> [--skip done.txt] <groups.txt>...
//! group_codes csearch <threshold.tsv> <threads> <node_limit> <Nmin> <Nmax> <maxK> <maxH> <groups.txt>...
//! group_codes params <groups.txt> <gap id> <A> <B> [node_limit]
//! group_codes cparams <groups.txt> <gap id> <H> <A> <B> [node_limit]
//! group_codes schedsearch <groups.txt> <gap id> <A> <B> <rounds> <max_w> [node_limit] [stride]
//! group_codes cdist <groups.txt> <gap id> <A> <B> <sched|ibm|auto> <rounds> [max_w] [node_limit] [x]
//! group_codes ler <groups.txt> <gap id> <A> <B> <sched|ibm|auto> <rounds> <p> <shots> <seed> [threads] [osd_order] [x]
//! ```
//!
//! The circuit commands take a coset code instead when the environment
//! variable `SUBGROUP` lists the elements of `H` (comma-separated). They use
//! the depth-7 schedules of `qec::bb_circuit`
//! (uniform circuit noise, BP+OSD-CS on the memory-basis sector), as
//! `examples/bb_codes.rs` does for abelian codes; the circuit distance is
//! rooted with the term-preserving code automorphisms
//! (`GroupCode::term_automorphisms`), falling back to observable mechanisms.
//!
//! `groups.txt` files are written by
//! `research/data/code-discovery-2/export_groups.g` (GAP SmallGroups). `A`
//! and `B` are comma-separated element indices of that export.
//!
//! `search` enumerates every inequivalent weight-(3, 3) code over every
//! group in the files and, for each connected class with `k > 0`, decides
//! whether `d >= T(n, k)` (the domination threshold of known codes, see
//! `known_codes.py`): `below` (a nontrivial logical of weight `< T` exists;
//! `d_up` is its weight), `tie` (`d = T` exactly), `new` (`d > T`, exact) or
//! `undecided` (node limit). One JSON line per class with `k > 0`.
use qsim_lab::qec::bicycle::{
    distance_upper_bound, logical_masks, min_weight_logical, DistanceOpts, Gf2Mat, SearchOutcome,
};
use qsim_lab::qec::group_algebra::{
    normalizer_quotient, orbit_roots, CosetCode, Cosets, Enumeration, FiniteGroup, GroupCode,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use qsim_lab::engines::stabilizer::fast_sampler::{FastSampler, WyRand};
use qsim_lab::engines::stabilizer::symphase::SymPhaseSampler;
use qsim_lab::qec::bb_circuit::{
    memory, schedule_valid, valid_schedules, BbMemory, BbSchedule, TwoBlockLayout,
};
use qsim_lab::qec::bposd::{BpOsd, DecodeStats, DemMatrix};
use qsim_lab::qec::color::circuit_dem;
use rand::RngCore;
use std::collections::HashMap;
use std::io::Write;
use std::sync::Mutex;
use std::time::Instant;

fn load_groups(paths: &[String]) -> Vec<(usize, FiniteGroup)> {
    let mut out = Vec::new();
    for p in paths {
        let text = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{p}: {e}"));
        for g in FiniteGroup::parse_export(&text).unwrap_or_else(|e| panic!("{p}: {e}")) {
            let id: usize = g
                .label
                .split(['(', ',', ')'])
                .nth(2)
                .and_then(|s| s.parse().ok())
                .unwrap();
            out.push((id, g));
        }
    }
    out
}

fn elems(s: &str) -> Vec<u16> {
    s.split(',').map(|t| t.trim().parse().unwrap()).collect()
}

struct Sector {
    h: Gf2Mat,
    masks: Vec<u128>,
}

/// Is there a nontrivial logical of weight <= w in either sector?
/// Ok(Some(weight)) found, Ok(None) none, Err(proven) aborted.
fn any_logical_upto(
    secs: &[Sector],
    roots: &[(usize, Vec<usize>)],
    w: usize,
    limit: u64,
) -> Result<Option<usize>, usize> {
    let mut best: Option<usize> = None;
    for s in secs {
        match min_weight_logical(&s.h, &s.masks, roots, w, w, limit).0 {
            SearchOutcome::Found(_, sup) => {
                best = Some(best.map_or(sup.len(), |b: usize| b.min(sup.len())));
            }
            SearchOutcome::NoneUpTo(_) => {}
            SearchOutcome::Aborted { proven } => return Err(proven),
        }
    }
    Ok(best)
}

/// Decides whether a code with known-threshold `t` (= T(n, k)) beats every
/// known code: `below` / `le_T` (a nontrivial logical of weight `< t` /
/// `= t` exists: randomized ISD first, then one exhaustive DFS level `t`),
/// `new` (no logical of weight `<= t` in either sector; then exact `d`
/// from `t + 1`), `undecided` (node limit). Returns (status, d_lo, d_up,
/// d_Z, d_X, roots used).
fn classify<F: FnOnce() -> Vec<(usize, Vec<usize>)>>(
    hx: &Gf2Mat,
    hz: &Gf2Mat,
    roots_fn: F,
    t: usize,
    limit: u64,
    rng: &mut StdRng,
) -> (&'static str, usize, usize, usize, usize, usize) {
    let secs: Vec<Sector> = [(hx, hz), (hz, hx)]
        .iter()
        .map(|(h, o)| Sector {
            h: (*h).clone(),
            masks: logical_masks(h, o).0,
        })
        .collect();
    // randomized information-set search in two stages: most codes are far
    // below the threshold and are dismissed after 10 iterations
    let lab = |w: usize| if w < t { "below" } else { "le_T" };
    let mut ubs: Vec<usize> = secs
        .iter()
        .map(|s| distance_upper_bound(&s.h, &s.masks, 10, rng).0)
        .collect();
    if t >= 1 && ubs[0].min(ubs[1]) < t {
        let ub = ubs[0].min(ubs[1]);
        return (lab(ub), 0, ub, 0, 0, 0);
    }
    for (i, s) in secs.iter().enumerate() {
        ubs[i] = ubs[i].min(distance_upper_bound(&s.h, &s.masks, 90, rng).0);
    }
    let ub = ubs[0].min(ubs[1]);
    if t >= 1 && ub <= t {
        return (lab(ub), 0, ub, 0, 0, 0);
    }
    let roots = roots_fn();
    let nr = roots.len();
    if t >= 1 {
        match any_logical_upto(&secs, &roots, t, limit) {
            Ok(Some(w)) => return (lab(w), 0, w.min(ub), 0, 0, nr),
            Err(p) => return ("undecided", p + 1, ub, 0, 0, nr),
            Ok(None) => {}
        }
    }
    // d > t in both sectors: exact from t + 1
    let mut d = [0usize; 2];
    let mut lo = [0usize; 2];
    for (i, s) in secs.iter().enumerate() {
        if ubs[i] <= t + 1 {
            d[i] = ubs[i];
            lo[i] = ubs[i];
            continue;
        }
        match min_weight_logical(&s.h, &s.masks, &roots, t + 1, ubs[i] - 1, limit).0 {
            SearchOutcome::Found(w, _) => {
                d[i] = w;
                lo[i] = w;
            }
            SearchOutcome::NoneUpTo(_) => {
                d[i] = ubs[i];
                lo[i] = ubs[i];
            }
            SearchOutcome::Aborted { proven } => {
                d[i] = ubs[i];
                lo[i] = proven + 1;
            }
        }
    }
    let (l, u) = (lo[0].min(lo[1]), d[0].min(d[1]));
    if l == u {
        ("new", l, u, d[0], d[1], nr)
    } else {
        ("new_bounds", l, u, d[0], d[1], nr)
    }
}

fn search_group(
    id: usize,
    g: &FiniteGroup,
    thr: &HashMap<(usize, usize), usize>,
    limit: u64,
) -> (String, String) {
    let t0 = Instant::now();
    let nn = g.order;
    let n = 2 * nn;
    let e = Enumeration::new(g, 3);
    let orbits = e.pair_orbits(g);
    let mut out = String::new();
    let mut rng = StdRng::seed_from_u64(nn as u64 * 1000 + id as u64);
    let (mut nk, mut ndisc, mut nbelow, mut ntie, mut nnew, mut nund) = (0, 0, 0, 0, 0, 0);
    let mut big_k = 0;
    for &(i, j, _) in &orbits {
        let (a, b) = (&e.reps[i as usize], &e.reps[j as usize]);
        let c = GroupCode::new(g, a, b);
        let (rx, rz) = c.ranks();
        let k = n - rx - rz;
        if k == 0 {
            continue;
        }
        nk += 1;
        let conn = c.is_connected();
        if !conn {
            ndisc += 1;
            continue;
        }
        if k > 128 {
            big_k += 1;
            continue;
        }
        let tc = Instant::now();
        let t = thr.get(&(n, k)).copied().unwrap_or(0);
        let (hx, hz) = (c.hx(), c.hz());
        let (status, lo, up, dz, dx, nroots) =
            classify(&hx, &hz, || c.distance_roots(), t, limit, &mut rng);
        match status {
            "below" => nbelow += 1,
            "le_T" | "tie" => ntie += 1,
            "new" => nnew += 1,
            _ => nund += 1,
        }
        let fmt = |v: &[u16]| {
            v.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        out += &format!(
            "{{\"N\":{nn},\"id\":{id},\"n\":{n},\"k\":{k},\"T\":{t},\"status\":\"{status}\",\"d_lo\":{lo},\"d_up\":{up},\"dz\":{dz},\"dx\":{dx},\"A\":\"{}\",\"B\":\"{}\",\"roots\":{},\"s\":{:.3}}}\n",
            fmt(a),
            fmt(b),
            nroots,
            tc.elapsed().as_secs_f64()
        );
    }
    let summary = format!(
        "N={nn} id={id} {} |Z|={} tclasses={} orbits={} k>0={nk} disconnected={ndisc} k>128={big_k} below={nbelow} le_T={ntie} new={nnew} undecided={nund} t={:.1}s",
        g.label,
        g.center_order(),
        e.reps.len(),
        orbits.len(),
        t0.elapsed().as_secs_f64()
    );
    (out, summary)
}

fn search(a: &[String]) {
    let mut thr: HashMap<(usize, usize), usize> = HashMap::new();
    for line in std::fs::read_to_string(&a[0]).unwrap().lines() {
        let v: Vec<usize> = line.split_whitespace().map(|x| x.parse().unwrap()).collect();
        if v.len() == 3 {
            thr.insert((v[0], v[1]), v[2]);
        }
    }
    let threads: usize = a[1].parse().unwrap();
    let limit: u64 = a[2].parse().unwrap();
    let (skip, files) = if a.get(3).is_some_and(|x| x == "--skip") {
        let done: std::collections::HashSet<(usize, usize)> = std::fs::read_to_string(&a[4])
            .unwrap()
            .lines()
            .filter_map(|l| {
                let v: Vec<usize> = l.split_whitespace().filter_map(|x| x.parse().ok()).collect();
                (v.len() == 2).then(|| (v[0], v[1]))
            })
            .collect();
        (done, &a[5..])
    } else {
        (Default::default(), &a[3..])
    };
    let groups: Vec<(usize, FiniteGroup)> = load_groups(files)
        .into_iter()
        .filter(|(id, g)| !skip.contains(&(g.order, *id)))
        .collect();
    eprintln!("{} groups ({} skipped as done)", groups.len(), skip.len());
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    let lock = Mutex::new(());
    // largest groups first for load balance
    let mut order: Vec<usize> = (0..groups.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(groups[i].1.order));
    order.par_iter().for_each(|&i| {
        let (id, g) = &groups[i];
        let (lines, summary) = search_group(*id, g, &thr, limit);
        let _l = lock.lock().unwrap();
        let so = std::io::stdout();
        let mut so = so.lock();
        so.write_all(lines.as_bytes()).unwrap();
        so.flush().unwrap();
        eprintln!("{summary}");
    });
}

fn params(a: &[String]) {
    let groups = load_groups(&a[..1]);
    let id: usize = a[1].parse().unwrap();
    let g = &groups.iter().find(|(i, _)| *i == id).expect("group id").1;
    let (av, bv) = (elems(&a[2]), elems(&a[3]));
    let limit: u64 = a.get(4).map_or(u64::MAX, |s| s.parse().unwrap());
    let c = GroupCode::new(g, &av, &bv);
    let t = Instant::now();
    let (rx, rz) = c.ranks();
    let k = c.n() - rx - rz;
    let roots = c.distance_roots();
    let opts = DistanceOpts {
        node_limit: limit,
        ub_iters: 1000,
        ..Default::default()
    };
    let (dz, dx) = c.distances(&opts);
    println!(
        "{{\"group\":\"{}\",\"n\":{},\"k\":{k},\"rank_hx\":{rx},\"rank_hz\":{rz},\"connected\":{},\"roots\":{},\"dz_lo\":{},\"dz_up\":{},\"dx_lo\":{},\"dx_up\":{},\"nodes\":{},\"wz\":{:?},\"wx\":{:?},\"s\":{:.3}}}",
        g.label,
        c.n(),
        c.is_connected(),
        roots.len(),
        dz.lower,
        dz.upper,
        dx.lower,
        dx.upper,
        dz.nodes + dx.nodes,
        dz.witness,
        dx.witness,
        t.elapsed().as_secs_f64()
    );
}

/// Sector DEM: (sector detector ids, merged columns, obs masks, probabilities).
fn sector_dem(m: &BbMemory) -> (Vec<usize>, DemMatrix) {
    let sdet: Vec<usize> = (0..m.detectors.len()).filter(|&i| m.in_sector[i]).collect();
    let mut map = vec![u32::MAX; m.detectors.len()];
    for (k, &i) in sdet.iter().enumerate() {
        map[i] = k as u32;
    }
    let dem = circuit_dem(&m.circuit, &m.noise, &m.detectors, &m.observables);
    let mut merged: HashMap<(Vec<u32>, u64), f64> = HashMap::new();
    for e in &dem {
        let zs: Vec<u32> = e
            .detectors
            .iter()
            .filter_map(|&i| {
                let z = map[i as usize];
                (z != u32::MAX).then_some(z)
            })
            .collect();
        if zs.is_empty() {
            assert_eq!(e.observables, 0, "undetectable logical mechanism");
            continue;
        }
        let v = merged.entry((zs, e.observables)).or_insert(0.0);
        *v = *v * (1.0 - e.p) + e.p * (1.0 - *v);
    }
    let mut ents: Vec<_> = merged.into_iter().collect();
    ents.sort_by(|x, y| x.0.cmp(&y.0));
    (
        sdet.clone(),
        DemMatrix {
            num_detectors: sdet.len(),
            cols: ents.iter().map(|e| e.0 .0.clone()).collect(),
            obs: ents.iter().map(|e| e.0 .1).collect(),
            p: ents.iter().map(|e| e.1).collect(),
        },
    )
}

/// A code for the circuit commands: a group code, or a coset code when the
/// environment variable `SUBGROUP` holds the elements of `H`.
enum AnyCode<'g> {
    Group(GroupCode<'g>),
    Coset(CosetCode<'g>),
}

impl TwoBlockLayout for AnyCode<'_> {
    fn order(&self) -> usize {
        match self {
            AnyCode::Group(c) => c.order(),
            AnyCode::Coset(c) => c.order(),
        }
    }
    fn weights(&self) -> (usize, usize) {
        match self {
            AnyCode::Group(c) => TwoBlockLayout::weights(c),
            AnyCode::Coset(c) => TwoBlockLayout::weights(c),
        }
    }
    fn x_term(&self, g: usize, t: usize) -> usize {
        match self {
            AnyCode::Group(c) => c.x_term(g, t),
            AnyCode::Coset(c) => c.x_term(g, t),
        }
    }
    fn z_term(&self, h: usize, t: usize) -> usize {
        match self {
            AnyCode::Group(c) => c.z_term(h, t),
            AnyCode::Coset(c) => c.z_term(h, t),
        }
    }
    fn hx(&self) -> Gf2Mat {
        match self {
            AnyCode::Group(c) => TwoBlockLayout::hx(c),
            AnyCode::Coset(c) => TwoBlockLayout::hx(c),
        }
    }
    fn hz(&self) -> Gf2Mat {
        match self {
            AnyCode::Group(c) => TwoBlockLayout::hz(c),
            AnyCode::Coset(c) => TwoBlockLayout::hz(c),
        }
    }
}

impl AnyCode<'_> {
    fn n(&self) -> usize {
        2 * TwoBlockLayout::order(self)
    }
    fn label(&self) -> String {
        match self {
            AnyCode::Group(c) => c.g.label.clone(),
            AnyCode::Coset(c) => format!("{} / H(|H|={})", c.g.label, c.c.h.len()),
        }
    }
    /// Check permutations of term-preserving automorphisms for the memory
    /// basis (group codes: `term_automorphisms`; coset codes: right
    /// translations by central elements, `Hx -> Hxz`).
    fn detector_perms(&self, x_basis: bool) -> Vec<Vec<usize>> {
        match self {
            AnyCode::Group(c) => c
                .term_automorphisms()
                .into_iter()
                .map(|(xc, zc, _)| if x_basis { xc } else { zc })
                .collect(),
            AnyCode::Coset(c) => {
                let g = c.g;
                (1..g.order)
                    .filter(|&z| (0..g.order).all(|y| g.mul(z, y) == g.mul(y, z)))
                    .map(|z| {
                        (0..c.order())
                            .map(|i| c.c.coset_of[g.mul(c.c.reps[i] as usize, z)] as usize)
                            .collect()
                    })
                    .collect()
            }
        }
    }
}

fn code_sched<'g>(groups: &'g [(usize, FiniteGroup)], a: &[String]) -> (AnyCode<'g>, BbSchedule) {
    let id: usize = a[0].parse().unwrap();
    let g = &groups.iter().find(|(i, _)| *i == id).expect("group id").1;
    let c = match std::env::var("SUBGROUP") {
        Ok(h) => {
            let cos: &'g Cosets = Box::leak(Box::new(Cosets::new(g, &elems(&h)).unwrap()));
            AnyCode::Coset(CosetCode::new(g, cos, &elems(&a[1]), &elems(&a[2])))
        }
        Err(_) => AnyCode::Group(GroupCode::new(g, &elems(&a[1]), &elems(&a[2]))),
    };
    let s = match a[3].as_str() {
        "auto" => valid_schedules(&c)[0],
        x => BbSchedule::parse(x),
    };
    assert!(schedule_valid(&c, &s), "invalid schedule {}", s.spec());
    (c, s)
}

/// Circuit distance of the sector DEM of `m`: mechanisms are grouped into
/// orbits of the term-preserving automorphisms acting on detectors (`r N +
/// h -> r N + perm(h)`, `perm` the Z- or X-check permutation of the memory
/// basis), one root per orbit, each banning the earlier orbits; falls back to
/// observable-mechanism roots if mechanisms are not determined by their
/// detectors. Returns (lower, upper or None, nodes, mechanisms, roots).
fn circuit_distance(
    c: &AnyCode,
    m: &BbMemory,
    x_basis: bool,
    max_w: usize,
    limit: u64,
) -> (usize, Option<usize>, u64, usize, usize) {
    let (_sdet, dm) = sector_dem(m);
    let nn = TwoBlockLayout::order(c);
    let ncol = dm.cols.len();
    let mut h = Gf2Mat::zeros(dm.num_detectors, ncol);
    for (j, col) in dm.cols.iter().enumerate() {
        for &i in col {
            h.flip(i as usize, j);
        }
    }
    let masks: Vec<u128> = dm.obs.iter().map(|&o| o as u128).collect();
    let mut index: HashMap<Vec<u32>, usize> = HashMap::new();
    let mut unique = true;
    for (j, col) in dm.cols.iter().enumerate() {
        if index.insert(col.clone(), j).is_some() {
            unique = false;
        }
    }
    let perms = c.detector_perms(x_basis);
    let mut col_perms: Vec<Vec<usize>> = Vec::new();
    if unique {
        'p: for p in &perms {
            let mut cp = Vec::with_capacity(ncol);
            for col in &dm.cols {
                let mut v: Vec<u32> = col
                    .iter()
                    .map(|&d| {
                        let (r, hh) = (d as usize / nn, d as usize % nn);
                        (r * nn + p[hh]) as u32
                    })
                    .collect();
                v.sort_unstable();
                match index.get(&v) {
                    Some(&t) => cp.push(t),
                    None => {
                        unique = false;
                        break 'p;
                    }
                }
            }
            col_perms.push(cp);
        }
    }
    let roots: Vec<(usize, Vec<usize>)> = if unique {
        qsim_lab::qec::group_algebra::orbit_roots(ncol, &col_perms)
            .into_iter()
            .collect()
    } else {
        let obs_mechs: Vec<usize> = (0..ncol).filter(|&j| dm.obs[j] != 0).collect();
        obs_mechs
            .iter()
            .enumerate()
            .map(|(i, &j)| (j, obs_mechs[..i].to_vec()))
            .collect()
    };
    let nroots = roots.len();
    let (out, nodes) = min_weight_logical(&h, &masks, &roots, 1, max_w, limit);
    match out {
        SearchOutcome::Found(w, _) => (w, Some(w), nodes, ncol, nroots),
        SearchOutcome::NoneUpTo(w) => (w + 1, None, nodes, ncol, nroots),
        SearchOutcome::Aborted { proven } => (proven + 1, None, nodes, ncol, nroots),
    }
}

fn cdist(a: &[String]) {
    let groups = load_groups(&a[..1]);
    let (c, s) = code_sched(&groups, &a[1..]);
    let rounds: usize = a[5].parse().unwrap();
    let max_w: usize = a.get(6).map_or(30, |s| s.parse().unwrap());
    let limit: u64 = a.get(7).map_or(1_000_000_000, |s| s.parse().unwrap());
    let x_basis = a.get(8).is_some_and(|s| s == "x");
    let t0 = Instant::now();
    let m = memory(&c, &s, rounds, 0.001, x_basis);
    let (lo, up, nodes, ncol, nroots) = circuit_distance(&c, &m, x_basis, max_w, limit);
    println!(
        "{{\"n\":{},\"sched\":\"{}\",\"rounds\":{rounds},\"basis\":\"{}\",\"mechanisms\":{ncol},\"roots\":{nroots},\"dcirc_lo\":{lo},\"dcirc_up\":{},\"nodes\":{nodes},\"s\":{:.1}}}",
        c.n(),
        s.spec(),
        if x_basis { "x" } else { "z" },
        up.map_or("null".to_string(), |u| u.to_string()),
        t0.elapsed().as_secs_f64()
    );
}

fn schedsearch(a: &[String]) {
    let groups = load_groups(&a[..1]);
    let mut spec: Vec<String> = a[1..4].to_vec();
    spec.push("auto".into());
    let (c, _) = code_sched(&groups, &spec);
    let rounds: usize = a[4].parse().unwrap();
    let max_w: usize = a[5].parse().unwrap();
    let limit: u64 = a.get(6).map_or(200_000_000, |s| s.parse().unwrap());
    let stride: usize = a.get(7).map_or(1, |s| s.parse().unwrap());
    let all = valid_schedules(&c);
    eprintln!("{} valid schedules", all.len());
    for s in all.iter().step_by(stride) {
        let t0 = Instant::now();
        let mz = memory(&c, s, rounds, 0.001, false);
        let (zl, zu, zn, _, _) = circuit_distance(&c, &mz, false, max_w, limit);
        let mx = memory(&c, s, rounds, 0.001, true);
        let (xl, xu, xn, _, _) = circuit_distance(&c, &mx, true, max_w, limit);
        let f = |u: Option<usize>| u.map_or("null".to_string(), |u| u.to_string());
        println!(
            "{{\"sched\":\"{}\",\"z_lo\":{zl},\"z_up\":{},\"x_lo\":{xl},\"x_up\":{},\"nodes\":{},\"s\":{:.1}}}",
            s.spec(),
            f(zu),
            f(xu),
            zn + xn,
            t0.elapsed().as_secs_f64()
        );
    }
}

fn ler(a: &[String]) {
    let groups = load_groups(&a[..1]);
    let (c, s) = code_sched(&groups, &a[1..]);
    let rounds: usize = a[5].parse().unwrap();
    let p: f64 = a[6].parse().unwrap();
    let shots: usize = a[7].parse().unwrap();
    let seed: u64 = a[8].parse().unwrap();
    let threads: usize = a.get(9).map_or(1, |s| s.parse().unwrap());
    let order: usize = a.get(10).map_or(10, |s| s.parse().unwrap());
    let x_basis = a.get(11).is_some_and(|s| s == "x");
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    let t0 = Instant::now();
    let m = memory(&c, &s, rounds, p, x_basis);
    let k = m.observables.len();
    assert!(k <= 64);
    let (sdet, dm) = sector_dem(&m);
    let nmech = dm.cols.len();
    let dec = BpOsd::new(dm, 100, 0.625, order);
    let sets: Vec<Vec<usize>> = sdet
        .iter()
        .map(|&i| m.detectors[i].clone())
        .chain(m.observables.iter().cloned())
        .collect();
    let smp = SymPhaseSampler::new(&m.circuit, &m.noise)
        .unwrap()
        .with_parities(&sets)
        .relative_to_reference();
    let f = FastSampler::new(&smp);
    let nz = sdet.len();
    let t_setup = t0.elapsed().as_secs_f64();
    let words = 4usize;
    let per_task = 64 * words * 8;
    let tasks: Vec<usize> = (0..shots.div_ceil(per_task)).collect();
    let t1 = Instant::now();
    let res: Vec<(u64, u64, DecodeStats)> = tasks
        .par_iter()
        .map(|&t| {
            let mut rng = WyRand(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (t as u64 + 1));
            let _ = rng.next_u64();
            let mut out = vec![0u64; f.stride() * words];
            let mut sc = dec.scratch();
            let mut st = DecodeStats::default();
            let (mut fails, mut n) = (0u64, 0u64);
            let mut fired: Vec<Vec<u32>> = vec![Vec::new(); 64];
            for _ in 0..8 {
                f.sample_batch(&mut rng, &mut out);
                for w in 0..words {
                    let blk = &out[w * f.stride()..(w + 1) * f.stride()];
                    for v in fired.iter_mut() {
                        v.clear();
                    }
                    for (r, &x0) in blk[..nz].iter().enumerate() {
                        let mut x = x0;
                        while x != 0 {
                            fired[x.trailing_zeros() as usize].push(r as u32);
                            x &= x - 1;
                        }
                    }
                    for (sh, fd) in fired.iter().enumerate() {
                        let mut actual = 0u64;
                        for j in 0..k {
                            actual |= (blk[nz + j] >> sh & 1) << j;
                        }
                        let pred = dec.decode(fd, &mut sc, &mut st);
                        fails += (pred != actual) as u64;
                        n += 1;
                    }
                }
            }
            (fails, n, st)
        })
        .collect();
    let (mut fails, mut n, mut st) = (0u64, 0u64, DecodeStats::default());
    for (fl, kk, s2) in res {
        fails += fl;
        n += kk;
        st.bp_converged += s2.bp_converged;
        st.osd_calls += s2.osd_calls;
    }
    let pl = fails as f64 / n as f64;
    let z = 1.96f64;
    let nf = n as f64;
    let den = 1.0 + z * z / nf;
    let cc = (pl + z * z / (2.0 * nf)) / den;
    let hw = z * (pl * (1.0 - pl) / nf + z * z / (4.0 * nf * nf)).sqrt() / den;
    let per_round = |x: f64| 1.0 - (1.0 - x).max(0.0).powf(1.0 / rounds as f64);
    println!(
        "{{\"group\":\"{}\",\"n\":{},\"k\":{k},\"A\":\"{}\",\"B\":\"{}\",\"sched\":\"{}\",\"basis\":\"{}\",\"rounds\":{rounds},\"p\":{p},\"shots\":{n},\"fails\":{fails},\"p_L\":{pl:.6e},\"ci95\":[{:.6e},{:.6e}],\"p_L_round\":{:.6e},\"ci95_round\":[{:.6e},{:.6e}],\"mechanisms\":{nmech},\"detectors\":{nz},\"bp_converged\":{},\"osd_calls\":{},\"osd_order\":{order},\"setup_s\":{t_setup:.2},\"decode_s\":{:.2},\"threads\":{threads}}}",
        c.label(),
        c.n(),
        a[2],
        a[3],
        s.spec(),
        if x_basis { "x" } else { "z" },
        cc - hw,
        cc + hw,
        per_round(pl),
        per_round(cc - hw),
        per_round(cc + hw),
        st.bp_converged,
        st.osd_calls,
        t1.elapsed().as_secs_f64()
    );
}

/// Coset codes over `G = Z_m x K` with `H` a cyclic non-normal subgroup of
/// `K` (one per orbit of `Aut(K)` on such subgroups), `N = m [K : H]` in
/// `[Nmin, Nmax]`: every pair (T-class of `A` in `G`, T-class of `B` in
/// `N_G(H) / H`), deduplicated under the automorphisms of `Z_m` and the
/// automorphism generators of `K` that fix `H`; classified like `search`.
fn csearch(a: &[String]) {
    let mut thr: HashMap<(usize, usize), usize> = HashMap::new();
    for line in std::fs::read_to_string(&a[0]).unwrap().lines() {
        let v: Vec<usize> = line.split_whitespace().map(|x| x.parse().unwrap()).collect();
        if v.len() == 3 {
            thr.insert((v[0], v[1]), v[2]);
        }
    }
    let threads: usize = a[1].parse().unwrap();
    let limit: u64 = a[2].parse().unwrap();
    let p = |i: usize| a[i].parse::<usize>().unwrap();
    let (nmin, nmax, maxk, maxh) = (p(3), p(4), p(5), p(6));
    let ks: Vec<(usize, FiniteGroup)> = load_groups(&a[7..])
        .into_iter()
        .filter(|(_, g)| g.order <= maxk && !g.is_abelian())
        .collect();
    // jobs: (K index, H, m)
    let mut jobs: Vec<(usize, Vec<u16>, usize)> = Vec::new();
    for (ki, (_, k)) in ks.iter().enumerate() {
        let kn = k.order;
        let mut seen: std::collections::HashSet<Vec<u16>> = Default::default();
        for x in 1..kn {
            let mut h: Vec<u16> = (0..kn)
                .filter(|&y| k.subgroup(&[x as u16])[y])
                .map(|y| y as u16)
                .collect();
            h.sort_unstable();
            if h.len() > maxh || seen.contains(&h) {
                continue;
            }
            // orbit of h under Aut(K) (generators)
            let mut orbit = vec![h.clone()];
            seen.insert(h.clone());
            let mut i = 0;
            while i < orbit.len() {
                for s in &k.aut_gens {
                    let mut v: Vec<u16> = orbit[i].iter().map(|&y| s[y as usize]).collect();
                    v.sort_unstable();
                    if seen.insert(v.clone()) {
                        orbit.push(v);
                    }
                }
                i += 1;
            }
            let c = Cosets::new(k, &h).unwrap();
            if c.is_normal(k) {
                continue;
            }
            let idx = kn / h.len();
            for m in 1..=nmax / idx {
                if m * idx >= nmin {
                    jobs.push((ki, h.clone(), m));
                }
            }
        }
    }
    eprintln!("{} K groups, {} (K, H, m) jobs", ks.len(), jobs.len());
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    let lock = Mutex::new(());
    jobs.sort_by_key(|j| std::cmp::Reverse(j.2 * ks[j.0].1.order));
    jobs.par_iter().for_each(|(ki, h, m)| {
        let (kid, k) = (&ks[*ki].0, &ks[*ki].1);
        let (lines, summary) = csearch_job(*kid, k, h, *m, &thr, limit);
        let _l = lock.lock().unwrap();
        let so = std::io::stdout();
        let mut so = so.lock();
        so.write_all(lines.as_bytes()).unwrap();
        so.flush().unwrap();
        eprintln!("{summary}");
    });
}

fn csearch_job(
    kid: usize,
    k: &FiniteGroup,
    h: &[u16],
    m: usize,
    thr: &HashMap<(usize, usize), usize>,
    limit: u64,
) -> (String, String) {
    let t0 = Instant::now();
    let kn = k.order;
    let g = FiniteGroup::abelian(m, 1).direct_product(k);
    let hh: Vec<u16> = h.to_vec(); // z = 0 component: element (0, h) = h
    let cos = Cosets::new(&g, &hh).unwrap();
    let nn = cos.len();
    let n = 2 * nn;
    let ea = Enumeration::new(&g, 3);
    let (q, lift) = normalizer_quotient(&g, &cos);
    let eb = Enumeration::new(&q, 3);
    let mut q_of_coset: HashMap<u16, u16> = HashMap::new();
    for (i, &l) in lift.iter().enumerate() {
        q_of_coset.insert(cos.coset_of[l as usize], i as u16);
    }
    let hset: std::collections::HashSet<u16> = hh.iter().copied().collect();
    // automorphisms of G fixing H: units of Z_m (all) and Aut(K) generators fixing H
    let auts: Vec<&Vec<u16>> = g
        .aut_gens
        .iter()
        .filter(|s| {
            let mut v: Vec<u16> = hh.iter().map(|&x| s[x as usize]).collect();
            v.sort_unstable();
            let mut w = hh.clone();
            w.sort_unstable();
            v == w
        })
        .collect();
    let (ma, mb) = (ea.reps.len(), eb.reps.len());
    let act_a: Vec<Vec<u32>> = auts
        .iter()
        .map(|s| {
            ea.reps
                .iter()
                .map(|r| ea.class(&r.iter().map(|&x| s[x as usize]).collect::<Vec<u16>>()))
                .collect()
        })
        .collect();
    let act_b: Vec<Vec<u32>> = auts
        .iter()
        .map(|s| {
            eb.reps
                .iter()
                .map(|r| {
                    let v: Vec<u16> = r
                        .iter()
                        .map(|&x| q_of_coset[&cos.coset_of[s[lift[x as usize] as usize] as usize]])
                        .collect();
                    eb.class(&v)
                })
                .collect()
        })
        .collect();
    let mut parent: Vec<u32> = (0..(ma * mb) as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    for i in 0..ma {
        for j in 0..mb {
            for (sa, sb) in act_a.iter().zip(&act_b) {
                let (x, y) = (
                    find(&mut parent, (i * mb + j) as u32),
                    find(&mut parent, (sa[i] as usize * mb + sb[j] as usize) as u32),
                );
                if x != y {
                    parent[x.max(y) as usize] = x.min(y);
                }
            }
        }
    }
    let mut out = String::new();
    let mut rng = StdRng::seed_from_u64((m * 1000 + kn) as u64 * 7919 + kid as u64);
    let (mut npairs, mut nk, mut ndisc, mut nlow, mut nbelow, mut nle, mut nnew, mut nund) =
        (0, 0, 0, 0, 0, 0, 0, 0);
    let _ = &hset;
    for x in 0..ma * mb {
        if find(&mut parent, x as u32) != x as u32 {
            continue;
        }
        npairs += 1;
        let (i, j) = (x / mb, x % mb);
        let av = &ea.reps[i];
        let bv: Vec<u16> = eb.reps[j].iter().map(|&y| lift[y as usize]).collect();
        let c = CosetCode::new(&g, &cos, av, &bv);
        let ex = c.explicit();
        // keep genuine weight-6 codes (no qubit repeated in a check)
        let full = ex.x.iter().chain(ex.z.iter()).all(|s| {
            let mut v = s.clone();
            v.sort_unstable();
            v.dedup();
            v.len() == 6
        });
        if !full {
            nlow += 1;
            continue;
        }
        let (rx, rz) = ex.ranks();
        let kk = n - rx - rz;
        if kk == 0 {
            continue;
        }
        nk += 1;
        if !ex.is_connected() {
            ndisc += 1;
            continue;
        }
        if kk > 128 {
            continue;
        }
        let tc = Instant::now();
        let t = thr.get(&(n, kk)).copied().unwrap_or(0);
        let (hx, hz) = (ex.hx(), ex.hz());
        let (status, lo, up, dz, dx, nroots) = classify(
            &hx,
            &hz,
            || orbit_roots(n, &c.translation_automorphisms()),
            t,
            limit,
            &mut rng,
        );
        match status {
            "below" => nbelow += 1,
            "le_T" => nle += 1,
            "new" | "new_bounds" => nnew += 1,
            _ => nund += 1,
        }
        let fmt = |v: &[u16]| {
            v.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        out += &format!(
            "{{\"K\":\"{}\",\"Kid\":{kid},\"Korder\":{kn},\"H\":\"{}\",\"m\":{m},\"n\":{n},\"k\":{kk},\"T\":{t},\"status\":\"{status}\",\"d_lo\":{lo},\"d_up\":{up},\"dz\":{dz},\"dx\":{dx},\"A\":\"{}\",\"B\":\"{}\",\"roots\":{nroots},\"s\":{:.3}}}\n",
            k.label,
            fmt(h),
            fmt(av),
            fmt(&bv),
            tc.elapsed().as_secs_f64()
        );
    }
    let summary = format!(
        "K={} |H|={} H={} m={m} N={nn} Aclasses={ma} Bclasses={mb} pairs={npairs} lowweight={nlow} k>0={nk} disconnected={ndisc} below={nbelow} le_T={nle} new={nnew} undecided={nund} t={:.1}s",
        k.label,
        h.len(),
        h.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","),
        t0.elapsed().as_secs_f64()
    );
    (out, summary)
}

/// Exact parameters of one coset code: `cparams <groups.txt> <gap id> <H> <A> <B> [node_limit]`
/// (comma-separated element indices of the export).
fn cparams(a: &[String]) {
    let groups = load_groups(&a[..1]);
    let id: usize = a[1].parse().unwrap();
    let g = &groups.iter().find(|(i, _)| *i == id).expect("group id").1;
    let h = elems(&a[2]);
    let cos = Cosets::new(g, &h).expect("H is a subgroup");
    let (av, bv) = (elems(&a[3]), elems(&a[4]));
    let limit: u64 = a.get(5).map_or(u64::MAX, |s| s.parse().unwrap());
    let c = CosetCode::new(g, &cos, &av, &bv);
    let ex = c.explicit();
    let t = Instant::now();
    let (rx, rz) = ex.ranks();
    let k = ex.n - rx - rz;
    let roots = orbit_roots(ex.n, &c.translation_automorphisms());
    let opts = DistanceOpts {
        node_limit: limit,
        ub_iters: 1000,
        ..Default::default()
    };
    let (dz, dx) = ex.distances(&roots, &opts);
    println!(
        "{{\"group\":\"{}\",\"H\":{},\"normal\":{},\"n\":{},\"k\":{k},\"commute\":{},\"connected\":{},\"roots\":{},\"dz_lo\":{},\"dz_up\":{},\"dx_lo\":{},\"dx_up\":{},\"nodes\":{},\"wz\":{:?},\"wx\":{:?},\"s\":{:.3}}}",
        g.label,
        h.len(),
        cos.is_normal(g),
        ex.n,
        ex.commutes(),
        ex.is_connected(),
        roots.len(),
        dz.lower,
        dz.upper,
        dx.lower,
        dx.upper,
        dz.nodes + dx.nodes,
        dz.witness,
        dx.witness,
        t.elapsed().as_secs_f64()
    );
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a[1].as_str() {
        "search" => search(&a[2..]),
        "csearch" => csearch(&a[2..]),
        "cparams" => cparams(&a[2..]),
        "params" => params(&a[2..]),
        "cdist" => cdist(&a[2..]),
        "schedsearch" => schedsearch(&a[2..]),
        "ler" => ler(&a[2..]),
        x => panic!("unknown command {x}"),
    }
}
