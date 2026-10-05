//! Two-block (BB / GB / coprime-BB) codes: parameters, search, circuit LER.
//!
//! ```text
//! bb_codes params <l> <m> <A> <B> [max_weight] [node_limit]
//! bb_codes search <Nmin> <Nmax> <wa> <wb> <worker> <workers> [node_limit]
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

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a[1].as_str() {
        "params" => {
            let l: usize = a[2].parse().unwrap();
            let m: usize = a[3].parse().unwrap();
            let c = TwoBlockCode::parse(l, m, &a[4], &a[5]);
            let mut o = DistanceOpts::default();
            if let Some(w) = a.get(6) {
                o.max_weight = w.parse().unwrap();
            }
            if let Some(w) = a.get(7) {
                o.node_limit = w.parse().unwrap();
            }
            let t = Instant::now();
            let k = c.k();
            let tk = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let d = c.distance(&o);
            println!(
                "{{\"l\":{l},\"m\":{m},\"A\":\"{}\",\"B\":\"{}\",\"n\":{},\"k\":{k},\"d_lower\":{},\"d_upper\":{},\"nodes\":{},\"k_s\":{tk:.6},\"d_s\":{:.3}}}",
                a[4], a[5], c.n(), d.lower, d.upper, d.nodes, t.elapsed().as_secs_f64()
            );
        }
        "search" => search(&a[2..]),
        x => panic!("unknown command {x}"),
    }
}

fn search(a: &[String]) {
    let p = |i: usize| a[i].parse::<usize>().unwrap();
    let (nmin, nmax, wa, wb, worker, workers) = (p(0), p(1), p(2), p(3), p(4), p(5));
    let node_limit: u64 = a.get(6).map_or(200_000_000, |s| s.parse().unwrap());
    let out = std::io::stdout();
    let mut out = out.lock();
    for nn in nmin..=nmax {
        if nn % workers != worker {
            continue;
        }
        for (l, m) in groups_of_order(nn) {
            let t0 = Instant::now();
            let g = AbelianGroup::new(l, m);
            let mut best: HashMap<usize, usize> = HashMap::new();
            let mut rng = StdRng::seed_from_u64((nn * 1000 + m) as u64);
            let (mut exact, mut pruned, mut aborted) = (0u64, 0u64, 0u64);
            let (ranked, found) = enumerate_codes(&g, wa, wb, |c| {
                let code = c.code();
                if c.k > 128 {
                    let (pa, pb) = code.poly_strings();
                    writeln!(out, "{{\"n\":{},\"k\":{},\"d_lo\":0,\"d_up\":0,\"l\":{l},\"m\":{m},\"wa\":{wa},\"wb\":{wb},\"A\":\"{pa}\",\"B\":\"{pb}\",\"skipped\":true}}", 2 * nn, c.k).unwrap();
                    return;
                }
                let (hx, hz) = (code.hx(), code.hz());
                let (masks, k) = logical_masks(&hx, &hz);
                assert_eq!(k, c.k);
                let (ub, _) = distance_upper_bound(&hx, &masks, 12, &mut rng);
                let b = best.get(&k).copied().unwrap_or(0);
                let (lo, up) = if ub < b {
                    pruned += 1;
                    (0, ub)
                } else {
                    let o = DistanceOpts {
                        max_weight: ub,
                        node_limit,
                        ub_iters: 30,
                        seed: 7,
                    };
                    let r = code_distance(&hx, &hz, Some(nn), &o);
                    if r.lower == r.upper {
                        exact += 1;
                        best.insert(k, b.max(r.lower));
                    } else {
                        aborted += 1;
                    }
                    (r.lower, r.upper)
                };
                let (pa, pb) = code.poly_strings();
                writeln!(
                    out,
                    "{{\"n\":{},\"k\":{k},\"d_lo\":{lo},\"d_up\":{up},\"l\":{l},\"m\":{m},\"wa\":{wa},\"wb\":{wb},\"A\":\"{pa}\",\"B\":\"{pb}\"}}",
                    2 * nn
                )
                .unwrap();
            });
            eprintln!(
                "N={nn} G=Z{l}xZ{m} |Aut|={} ranked={ranked} classes={found} exact={exact} pruned={pruned} aborted={aborted} t={:.1}s",
                g.auts.len(),
                t0.elapsed().as_secs_f64()
            );
        }
    }
}
