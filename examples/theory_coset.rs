//! Data for research/theory/theory-coset.md (see tests/theory/theory_coset.rs for the
//! exact checks).
//!   theory_coset tv   <N> <a> <we> <wm> <cmax>     exact TV + lemma bounds + walk stats
//!   theory_coset zero <Nmax> <cmax> <we> <wm>      TV = 0 sweep over all N, all bases
//!   theory_coset mc   <N> <regs:shor|eh> <a> <we> <wm> <c> <E samples> <branches> <seed> [ox ob]
#![allow(clippy::needless_range_loop)]
#[path = "../tests/theory_coset_model/mod.rs"]
mod model;
use model::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn arg<T: std::str::FromStr>(v: &[String], i: usize) -> T
where
    T::Err: std::fmt::Debug,
{
    v[i].parse().unwrap()
}

fn walk_stats(m: &Model) -> (f64, Vec<f64>, Vec<f64>, f64, f64, Vec<f64>) {
    // exhaustive over E and branches: unfaithful fraction, mean deviant
    // weight per window, mean b_out per window, mean final |jx - j|, |jb - j'|
    let wn = m.wins.len();
    let mut unf = 0.0;
    let mut dev = vec![0.0; wn];
    let mut bo = vec![0.0; wn];
    let (mut dx, mut db) = (0.0, 0.0);
    let mut fb = vec![0.0; wn];
    let side = 1u64 << m.c;
    let tot = ((1u64 << m.t) * side * side) as f64;
    for e in 0..1u64 << m.t {
        let d = m.digits(e);
        for j in 0..side {
            for j2 in 0..side {
                let w = m.walk(&d, j, j2);
                if !w.faithful {
                    unf += 1.0;
                }
                for k in 0..wn {
                    dev[k] += w.deviant[k] as u8 as f64;
                    bo[k] += w.b_out[k] as u8 as f64;
                }
                if let Some(k) = w.first_bad {
                    fb[k] += 1.0;
                }
                if w.faithful {
                    // registers swap every window
                    let (ix, ib) = if wn.is_multiple_of(2) {
                        (j, j2)
                    } else {
                        (j2, j)
                    };
                    dx += (w.jx - ix as i64).abs() as f64;
                    db += (w.jb - ib as i64).abs() as f64;
                }
            }
        }
    }
    (
        unf / tot,
        dev.iter().map(|v| v / tot).collect(),
        bo.iter().map(|v| v / tot).collect(),
        dx / tot,
        db / tot,
        fb.iter().map(|v| v / tot).collect(),
    )
}

fn main() {
    let v: Vec<String> = std::env::args().collect();
    match v[1].as_str() {
        "tv" => {
            let (nm, a, we, wm, cmax): (u64, u64, usize, usize, usize) =
                (arg(&v, 2), arg(&v, 3), arg(&v, 4), arg(&v, 5), arg(&v, 6));
            let me = Model::new(nm, a, we, wm, 0);
            let pe = distribution(me.t, &me.supports());
            println!(
                "N={nm} a={a} r={} n={} t={} we={we} wm={wm} windows={} 2^n/N={:.3}",
                order(a, nm),
                me.n,
                me.t,
                me.wins.len(),
                (1u64 << me.n) as f64 / nm as f64
            );
            for c in 1..=cmax {
                let t0 = std::time::Instant::now();
                let mc = Model::new(nm, a, we, wm, c);
                let supp = mc.supports();
                let cls = mc.classes();
                let tvv = tv(&distribution(mc.t, &supp), &pe);
                let sq0 = mc.lemma_square(&supp, &cls, 0, 0);
                // best shifted square (grid), honest: it is a valid reference for any shift
                let k = mc.chunks() as i64;
                let mut best = (f64::INFINITY, 0, 0);
                for ox in -k..=k {
                    for ob in -k..=k {
                        let s = mc.lemma_square(&supp, &cls, ox, ob);
                        if s.bound < best.0 {
                            best = (s.bound, ox, ob);
                        }
                    }
                }
                let sqb = mc.lemma_square(&supp, &cls, best.1, best.2);
                let maj = mc.lemma_majority(&supp, &cls);
                let (unf, dev, bo, dx, db, fb) = walk_stats(&mc);
                let f = (1u64 << c) as f64;
                println!(
                    "c={c} K={} TV={tvv:.6} TV*2^c={:.3} | square(0,0): dbar={:.5} drms={:.5} bound={:.5} | square({},{}): dbar={:.5} drms={:.5} thbar={:.2e} bound={:.5} bound*2^c={:.2} | majority: {} | unfaithful={:.5} (*2^c={:.3}) mean|dJx|={dx:.3} mean|dJb|={db:.3} ({:.1}s)",
                    mc.chunks(), tvv * f, sq0.delta_mean, sq0.delta_rms, sq0.bound, best.1, best.2, sqb.delta_mean, sqb.delta_rms, sqb.theta_mean, sqb.bound, sqb.bound * f,
                    match maj { Some(s) => format!("dbar={:.5} drms={:.5} thbar={:.2e} bound={:.5}", s.delta_mean, s.delta_rms, s.theta_mean, s.bound), None => "overlap".into() },
                    unf, unf * f, t0.elapsed().as_secs_f64()
                );
                println!(
                    "   deviant/window*2^c: {}",
                    dev.iter()
                        .map(|d| format!("{:.2}", d * f))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
                println!(
                    "   first unfaithful lookup per window*2^c: {}",
                    fb.iter()
                        .map(|d| format!("{:.2}", d * f))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
                println!(
                    "   b_out/window*2^c:   {}",
                    bo.iter()
                        .map(|d| format!("{:.2}", d * f))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }
        }
        "cross" => {
            let (nm, a, we, wm, c): (u64, u64, usize, usize, usize) =
                (arg(&v, 2), arg(&v, 3), arg(&v, 4), arg(&v, 5), arg(&v, 6));
            let mc = if v.len() > 7 {
                Model::with_t(nm, a, we, wm, c, arg(&v, 7))
            } else {
                Model::new(nm, a, we, wm, c)
            };
            let supp = mc.supports();
            let (x, w) = mc.cross_overlap(&supp, &mc.classes());
            println!(
                "N={nm} a={a} r={} c={c}: cross-class shared (ordered pairs)={x} within-class={w}",
                order(a, nm)
            );
        }
        "probe" => {
            let (nm, a, we, wm, c): (u64, u64, usize, usize, usize) =
                (arg(&v, 2), arg(&v, 3), arg(&v, 4), arg(&v, 5), arg(&v, 6));
            let mc = Model::new(nm, a, we, wm, c);
            let side = 1u64 << c;
            let mut seen = std::collections::BTreeSet::new();
            for e in 0..1u64 << mc.t {
                let d = mc.digits(e);
                for j in 0..side {
                    for j2 in 0..side {
                        let w = mc.walk(&d, j, j2);
                        if !w.faithful {
                            seen.insert((w.u, w.x % nm, w.b % nm, w.first_bad.unwrap(), w.x, w.b));
                        }
                    }
                }
            }
            for s in seen.iter().take(60) {
                println!("{s:?}");
            }
            println!("M mod N = {}", mc.m() % nm);
        }
        "pow2" => {
            // all N in [lo, hi], all bases of power-of-two order r >= 2
            let (lo, hi, cmax, we, wm): (u64, u64, usize, usize, usize) =
                (arg(&v, 2), arg(&v, 3), arg(&v, 4), arg(&v, 5), arg(&v, 6));
            let mut tally = std::collections::BTreeMap::new();
            for nm in (lo | 1..=hi).step_by(2) {
                for a in 2..nm - 1 {
                    if gcd(a, nm) != 1 {
                        continue;
                    }
                    let r = order(a, nm);
                    if !r.is_power_of_two() {
                        continue;
                    }
                    let mut line = String::new();
                    let mut z = true;
                    for c in 1..=cmax {
                        let mc = Model::new(nm, a, we, wm, c);
                        let (tvv, off, mism) = mc.pow2_tv(r);
                        assert_eq!(mism, 0, "Phi_E not a function of E mod r");
                        z &= off == 0.0;
                        line.push_str(&format!(" c={c}:TV={tvv:.2e},offdiag={off:.2e}"));
                    }
                    *tally.entry((r, z)).or_insert(0u32) += 1;
                    println!("N={nm} a={a} r={r} we={we}{line}");
                }
            }
            println!("tally (r, disjoint for all c): {tally:?}");
        }
        "scan" => {
            // TV vs number of windows (t exponent bits, we = 1 or 2) at fixed N, c
            let (nm, a, we, wm, c, tmax): (u64, u64, usize, usize, usize, usize) = (
                arg(&v, 2),
                arg(&v, 3),
                arg(&v, 4),
                arg(&v, 5),
                arg(&v, 6),
                arg(&v, 7),
            );
            for t in (we..=tmax).step_by(we) {
                let mc = Model::with_t(nm, a, we, wm, c, t);
                let me = Model::with_t(nm, a, we, wm, 0, t);
                let supp = mc.supports();
                let cls = mc.classes();
                let tvv = tv(&distribution(t, &supp), &distribution(t, &me.supports()));
                let sq = mc.lemma_square(&supp, &cls, 0, 0);
                let maj = mc
                    .lemma_majority(&supp, &cls)
                    .map(|s| s.bound)
                    .unwrap_or(f64::NAN);
                let (unf, _, _, _, _, _) = walk_stats(&mc);
                let f = (1u64 << c) as f64;
                println!("N={nm} a={a} we={we} wm={wm} c={c} K={} t={t} windows={} additions={} TV*2^c={:.3} square-bound*2^c={:.3} majority-bound*2^c={:.3} dbar(square)*2^c={:.3} unfaithful*2^c={:.3}",
                    mc.chunks(), mc.wins.len(), 2 * mc.chunks() * mc.wins.len(), tvv * f, sq.bound * f, maj * f, sq.delta_mean * f, unf * f);
            }
        }
        "zero" => {
            let (nmax, cmax, we, wm): (u64, usize, usize, usize) =
                (arg(&v, 2), arg(&v, 3), arg(&v, 4), arg(&v, 5));
            let mut stats = std::collections::BTreeMap::new();
            for nm in (5..=nmax).step_by(2) {
                for a in 2..nm - 1 {
                    if gcd(a, nm) != 1 {
                        continue;
                    }
                    let r = order(a, nm);
                    let pow2 = r.is_power_of_two();
                    let me = Model::new(nm, a, we, wm, 0);
                    let pe = distribution(me.t, &me.supports());
                    let mut zs = String::new();
                    let mut allzero = true;
                    for c in 1..=cmax {
                        let mc = Model::new(nm, a, we, wm, c);
                        let supp = mc.supports();
                        let tvv = tv(&distribution(mc.t, &supp), &pe);
                        let z = tvv < 1e-9;
                        allzero &= z;
                        zs.push_str(&format!(" {tvv:.4}"));
                    }
                    *stats.entry((pow2, allzero)).or_insert(0u32) += 1;
                    println!(
                        "N={nm} a={a} r={r} pow2={pow2} TV(c=1..{cmax}):{zs}{}",
                        if allzero != pow2 {
                            "  <-- MISMATCH"
                        } else {
                            ""
                        }
                    );
                }
            }
            println!("summary (r power of two, TV==0 for all c): {stats:?}");
        }
        "mc" => {
            let nm: u64 = arg(&v, 2);
            let kind = v[3].as_str();
            let a: u64 = arg(&v, 4);
            let (we, wm, c, ne, nb, seed): (usize, usize, usize, usize, usize, u64) = (
                arg(&v, 5),
                arg(&v, 6),
                arg(&v, 7),
                arg(&v, 8),
                arg(&v, 9),
                arg(&v, 10),
            );
            let n = work_bits(nm);
            let regs: Vec<(usize, u64)> = if kind == "shor" {
                vec![(2 * n, a)]
            } else {
                let mb = 64 - nm.leading_zeros() as usize;
                let m = mb.div_ceil(2);
                let y = powmod(a, (nm - 1) / 2, nm);
                vec![(m, inv(y, nm)), (2 * m, a)]
            };
            let md = Model::from_regs(nm, &regs, we, wm, c);
            let wn = md.wins.len();
            let mut rng = StdRng::seed_from_u64(seed);
            let side = 1i64 << c;
            let f = side as f64;
            let draw = |rng: &mut StdRng| -> Vec<u64> {
                md.wins
                    .iter()
                    .map(|w| rng.random_range(0..1u64 << w.1))
                    .collect()
            };
            // pilot sample (independent of the main sample): choose, for every
            // prefix length k, the square offset minimising the mean bad fraction
            let kk = md.chunks() as i64 * 2 + 2;
            let mut pil: Vec<Vec<(i64, i64, i64, i64)>> = vec![Vec::new(); wn];
            for _ in 0..(ne / 4).max(50) {
                let d = draw(&mut rng);
                for _ in 0..(nb / 4).max(50) {
                    let (j, j2) = (
                        rng.random_range(0..side as u64),
                        rng.random_range(0..side as u64),
                    );
                    let w = md.walk(&d, j, j2);
                    for k in 0..wn {
                        let (ix, ib) = if (k + 1) % 2 == 0 {
                            (j as i64, j2 as i64)
                        } else {
                            (j2 as i64, j as i64)
                        };
                        pil[k].push((w.traj[k].0, w.traj[k].1, ix, ib));
                    }
                }
            }
            let offs: Vec<(i64, i64)> = (0..wn)
                .map(|k| {
                    let mut best = (usize::MAX, 0, 0);
                    for ox in -kk..=kk {
                        for ob in -kk..=kk {
                            let bad = pil[k]
                                .iter()
                                .filter(|p| {
                                    !((ox..ox + side).contains(&p.0)
                                        && (ob..ob + side).contains(&p.1))
                                })
                                .count();
                            if bad < best.0 {
                                best = (bad, ox, ob);
                            }
                        }
                    }
                    (best.1, best.2)
                })
                .collect();
            // main sample
            let mut ds = vec![Vec::new(); wn];
            let mut ts = Vec::new();
            let mut unf = vec![0.0; wn];
            let mut disp = vec![(0.0, 0.0); wn];
            let mut dev = vec![0.0; wn];
            let mut nf = vec![0.0; wn];
            for _ in 0..ne {
                let d = draw(&mut rng);
                let mut bad = vec![0usize; wn];
                let mut oth = 0usize;
                for _ in 0..nb {
                    let (j, j2) = (
                        rng.random_range(0..side as u64),
                        rng.random_range(0..side as u64),
                    );
                    let w = md.walk(&d, j, j2);
                    for k in 0..wn {
                        let (ox, ob) = offs[k];
                        let (jx, jb) = w.traj[k];
                        if !((ox..ox + side).contains(&jx) && (ob..ob + side).contains(&jb)) {
                            bad[k] += 1;
                        }
                        if w.first_bad.is_some_and(|fb| fb <= k) {
                            unf[k] += 1.0;
                        }
                        let (ix, ib) = if (k + 1) % 2 == 0 {
                            (j as i64, j2 as i64)
                        } else {
                            (j2 as i64, j as i64)
                        };
                        if w.first_bad.is_none_or(|fb| fb > k) {
                            disp[k].0 += (jx - ix) as f64;
                            disp[k].1 += ((jx - ix) as f64).powi(2);
                            nf[k] += 1.0;
                        }
                        let _ = ib;
                        dev[k] += w.deviant[k] as u8 as f64;
                    }
                    // theta: final state in another class's square
                    let (ox, ob) = offs[wn - 1];
                    let (jx, jb) = w.traj[wn - 1];
                    if (ob..ob + side).contains(&jb)
                        && !(ox..ox + side).contains(&jx)
                        && md.in_any_square(w.x, ox, side)
                    {
                        oth += 1;
                    }
                }
                for k in 0..wn {
                    ds[k].push(bad[k] as f64 / nb as f64);
                }
                ts.push(oth as f64 / nb as f64);
            }
            let tot = (ne * nb) as f64;
            let est = |xs: &[f64]| -> (f64, f64, f64) {
                let m = xs.iter().sum::<f64>() / xs.len() as f64;
                let m2 = xs.iter().map(|d| d * d).sum::<f64>() / xs.len() as f64;
                let m2u = ((m2 - m / nb as f64) / (1.0 - 1.0 / nb as f64)).max(0.0);
                let se = (xs.iter().map(|d| (d - m).powi(2)).sum::<f64>()
                    / (xs.len() as f64 * (xs.len() as f64 - 1.0)))
                    .sqrt();
                (m, m2u.sqrt(), se)
            };
            println!("N={nm} {kind} a={a} n={n} t={} windows={wn} we={we} wm={wm} c={c} K={} E={ne} branches={nb} seed={seed}", md.t, md.chunks());
            println!("  k  offset     dbar*2^c  (se)    drms*2^c  unfaithful*2^c  mean dJ  var dJ  deviant*2^c");
            for k in 0..wn {
                let (dm, dr, se) = est(&ds[k]);
                let mj = disp[k].0 / nf[k];
                println!(
                    "  {:2} ({:3},{:3}) {:9.3} ({:5.3}) {:9.3} {:12.3} {:9.3} {:8.3} {:8.3}",
                    k + 1,
                    offs[k].0,
                    offs[k].1,
                    dm * f,
                    se * f,
                    dr * f,
                    unf[k] / tot * f,
                    mj,
                    disp[k].1 / nf[k] - mj * mj,
                    dev[k] / tot * f
                );
            }
            let (dm, dr, se) = est(&ds[wn - 1]);
            let (tm, tr, _) = est(&ts);
            let bound = dr + tr + dm + (dm * tm).sqrt();
            println!("FINAL c={c}: dbar={dm:.4e} (se {se:.1e}) drms={dr:.4e} thetabar={tm:.2e} thetarms={tr:.2e} | lemma bound TV <= drms+thrms+dbar+sqrt(dbar*thbar) = {bound:.4e} (bound*2^c={:.2}); +2se: {:.4e}", bound * f, bound + 4.0 * se);
        }
        _ => panic!("usage"),
    }
}
