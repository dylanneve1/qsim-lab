//! Two-block (BB / GB / coprime-BB) codes: parameters, search, circuit LER.
//!
//! ```text
//! bb_codes params <l> <m> <A> <B> [max_weight] [node_limit]
//! bb_codes search <Nmin> <Nmax> <wa> <wb> <worker> <workers> [node_limit]
//! bb_codes schedules <l> <m> <A> <B>
//! bb_codes ler <l> <m> <A> <B> <sched|ibm> <rounds> <p> <shots> <seed> [threads] [osd_order] [x]
//! bb_codes cdist <l> <m> <A> <B> <sched|ibm> <rounds> [max_w] [node_limit] [x]
//! bb_codes schedsearch <l> <m> <A> <B> <rounds> <max_w> [node_limit] [stride]
//! ```
//! `search` enumerates every inequivalent two-block code over every abelian
//! group of rank <= 2 and order `N` in `[Nmin, Nmax]` (`N % workers ==
//! worker`), with `|A| = wa`, `|B| = wb`, and prints one JSON line per class
//! with `k > 0`: exact `d` when `d_lower == d_upper`.
use qsim_lab::qec::bb_search::{enumerate_codes, groups_of_order, AbelianGroup};
use qsim_lab::qec::bicycle::{
    code_distance, distance_upper_bound, logical_masks, DistanceOpts, TwoBlockCode,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

use qsim_lab::qec::bb_circuit::{
    memory, schedule_valid, valid_schedules, BbMemory, BbSchedule, IBM_SCHEDULE,
};
use qsim_lab::qec::bicycle::{min_weight_logical, Gf2Mat, SearchOutcome};
use qsim_lab::qec::bposd::{BpOsd, DecodeStats, DemMatrix};
use qsim_lab::qec::color::circuit_dem;
use qsim_lab::stabilizer::fast_sampler::{FastSampler, WyRand};
use qsim_lab::stabilizer::symphase::SymPhaseSampler;
use rand::RngCore;
use rayon::prelude::*;

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

fn code_and_sched(a: &[String]) -> (TwoBlockCode, BbSchedule) {
    let c = TwoBlockCode::parse(a[0].parse().unwrap(), a[1].parse().unwrap(), &a[2], &a[3]);
    let s = match a[4].as_str() {
        "auto" => valid_schedules(&c)[0],
        x => BbSchedule::parse(x),
    };
    assert!(schedule_valid(&c, &s), "invalid schedule {}", s.spec());
    (c, s)
}

fn ler(a: &[String]) {
    let (c, s) = code_and_sched(a);
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
    let h = z * (pl * (1.0 - pl) / nf + z * z / (4.0 * nf * nf)).sqrt() / den;
    let per_round = |x: f64| 1.0 - (1.0 - x).max(0.0).powf(1.0 / rounds as f64);
    let (pa, pb) = c.poly_strings();
    println!(
        "{{\"n\":{},\"k\":{k},\"l\":{},\"m\":{},\"A\":\"{pa}\",\"B\":\"{pb}\",\"sched\":\"{}\",\"basis\":\"{}\",\"rounds\":{rounds},\"p\":{p},\"shots\":{n},\"fails\":{fails},\"p_L\":{pl:.6e},\"ci95\":[{:.6e},{:.6e}],\"p_L_round\":{:.6e},\"ci95_round\":[{:.6e},{:.6e}],\"mechanisms\":{nmech},\"detectors\":{nz},\"bp_converged\":{},\"osd_calls\":{},\"osd_order\":{order},\"setup_s\":{t_setup:.2},\"decode_s\":{:.2},\"threads\":{threads}}}",
        c.n(), c.l, c.m, s.spec(), if x_basis { "x" } else { "z" },
        cc - h, cc + h, per_round(pl), per_round(cc - h), per_round(cc + h),
        st.bp_converged, st.osd_calls, t1.elapsed().as_secs_f64()
    );
}

/// Circuit distance of the sector DEM of `m`, using the translation symmetry
/// of the circuit: mechanisms are grouped into orbits under the group
/// (detector `r N + h -> r N + (h + g)`), one root per orbit, each root
/// banning all earlier orbits. Falls back to observable-mechanism roots if
/// two mechanisms share a detector set. Returns (lower, upper or None, nodes).
fn circuit_distance(
    c: &TwoBlockCode,
    m: &BbMemory,
    max_w: usize,
    limit: u64,
) -> (usize, Option<usize>, u64, usize) {
    let (_sdet, dm) = sector_dem(m);
    let nn = c.order();
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
    let shift = |col: &[u32], gi: usize, gj: usize| -> Vec<u32> {
        let mut v: Vec<u32> = col
            .iter()
            .map(|&d| {
                let (r, hh) = (d as usize / nn, d as usize % nn);
                let (i, j) = (hh / c.m, hh % c.m);
                (r * nn + ((i + gi) % c.l) * c.m + (j + gj) % c.m) as u32
            })
            .collect();
        v.sort_unstable();
        v
    };
    let mut orbit = vec![usize::MAX; ncol];
    let mut reps: Vec<Vec<usize>> = Vec::new();
    if unique {
        'outer: for j0 in 0..ncol {
            if orbit[j0] != usize::MAX {
                continue;
            }
            let id = reps.len();
            let mut members = vec![j0];
            orbit[j0] = id;
            let mut st = vec![j0];
            while let Some(j) = st.pop() {
                for (gi, gj) in [(1, 0), (0, 1)] {
                    match index.get(&shift(&dm.cols[j], gi, gj)) {
                        Some(&t) => {
                            if orbit[t] == usize::MAX {
                                orbit[t] = id;
                                members.push(t);
                                st.push(t);
                            }
                        }
                        None => {
                            unique = false;
                            break 'outer;
                        }
                    }
                }
            }
            reps.push(members);
        }
    }
    let roots: Vec<(usize, Vec<usize>)> = if unique {
        let mut banned = Vec::new();
        let mut roots = Vec::new();
        for mem in &reps {
            roots.push((mem[0], banned.clone()));
            banned.extend_from_slice(mem);
        }
        roots
    } else {
        let obs_mechs: Vec<usize> = (0..ncol).filter(|&j| dm.obs[j] != 0).collect();
        obs_mechs
            .iter()
            .enumerate()
            .map(|(i, &j)| (j, obs_mechs[..i].to_vec()))
            .collect()
    };
    let (out, nodes) = min_weight_logical(&h, &masks, &roots, 1, max_w, limit);
    match out {
        SearchOutcome::Found(w, _) => (w, Some(w), nodes, ncol),
        SearchOutcome::NoneUpTo(w) => (w + 1, None, nodes, ncol),
        SearchOutcome::Aborted { proven } => (proven + 1, None, nodes, ncol),
    }
}

fn cdist(a: &[String]) {
    let (c, s) = code_and_sched(a);
    let rounds: usize = a[5].parse().unwrap();
    let max_w: usize = a.get(6).map_or(30, |s| s.parse().unwrap());
    let limit: u64 = a.get(7).map_or(1_000_000_000, |s| s.parse().unwrap());
    let x_basis = a.get(8).is_some_and(|s| s == "x");
    let t0 = Instant::now();
    let m = memory(&c, &s, rounds, 0.001, x_basis);
    let (lo, up, nodes, ncol) = circuit_distance(&c, &m, max_w, limit);
    println!(
        "{{\"n\":{},\"sched\":\"{}\",\"rounds\":{rounds},\"basis\":\"{}\",\"mechanisms\":{ncol},\"dcirc_lo\":{lo},\"dcirc_up\":{},\"nodes\":{nodes},\"s\":{:.1}}}",
        c.n(), s.spec(), if x_basis { "x" } else { "z" },
        up.map_or("null".to_string(), |u| u.to_string()),
        t0.elapsed().as_secs_f64()
    );
}

/// For every valid schedule: circuit distance in both bases (`rounds`
/// cycles, weights up to `max_w`); prints one JSON line per schedule.
fn schedsearch(a: &[String]) {
    let c = TwoBlockCode::parse(a[0].parse().unwrap(), a[1].parse().unwrap(), &a[2], &a[3]);
    let rounds: usize = a[4].parse().unwrap();
    let max_w: usize = a[5].parse().unwrap();
    let limit: u64 = a.get(6).map_or(200_000_000, |s| s.parse().unwrap());
    let stride: usize = a.get(7).map_or(1, |s| s.parse().unwrap());
    let all = valid_schedules(&c);
    eprintln!("{} valid schedules", all.len());
    for s in all.iter().step_by(stride) {
        let t0 = Instant::now();
        let mz = memory(&c, s, rounds, 0.001, false);
        let (zl, zu, zn, _) = circuit_distance(&c, &mz, max_w, limit);
        let mx = memory(&c, s, rounds, 0.001, true);
        let (xl, xu, xn, _) = circuit_distance(&c, &mx, max_w, limit);
        let f = |u: Option<usize>| u.map_or("null".to_string(), |u| u.to_string());
        println!(
            "{{\"sched\":\"{}\",\"z_lo\":{zl},\"z_up\":{},\"x_lo\":{xl},\"x_up\":{},\"nodes\":{},\"s\":{:.1}}}",
            s.spec(), f(zu), f(xu), zn + xn, t0.elapsed().as_secs_f64()
        );
    }
}
