//! Driver for the exact simulation of Gidney 2025's approximate modular
//! exponentiation (`research/shor/approx-modexp.md`, `src/shor/approx.rs`).
//!
//! ```text
//! cargo run --release --example approx_modexp -- config  N=3127 g=3122 mode=shor m=24 f=10 mask=paper
//! cargo run --release --example approx_modexp -- verify  N=... seeds=16
//! cargo run --release --example approx_modexp -- dist    N=... [unmasked=1] [succ=paper|repo|eh]
//! cargo run --release --example approx_modexp -- dump    N=... out=DIR      (tables + F̃ for the Python checks)
//! cargo run --release --example approx_modexp -- replay  N=... log=FILE out=FILE
//! cargo run --release --example approx_modexp -- gate    w4=2 f=6 [trials=..]
//! ```
//!
//! Parameters are `key=value`: `N`, `g`, `mode` (`shor` | `eh`), `m` (Shor
//! exponent qubits, default `2n`), `w1 w3a w3b w4` (windows, default 2),
//! `f` (accumulator bits), `mask` (bits or `paper`), `gap` (default `f`),
//! `ell` (prime bits), `minprimes` (default 100), `seed`.

use qsim_lab::algorithms::{gcd, pow_mod};
use qsim_lab::shor::approx::{
    best_shift, cond_fidelity, distribution, evaluate, gate_loop4_step, overlap,
    paper_mask_bits, paper_success, plan, shifted, tv,
    ApproxConfig, ApproxParams, ConstOutcomes, Loop4Layout, Outcomes, RandomOutcomes,
    ReplayOutcomes, Verify,
};
use qsim_lab::shor::ge;
use qsim_lab::shor::mbu::{eval_on_key, MbuOp};
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

fn kv(args: &[String]) -> HashMap<String, String> {
    args.iter()
        .filter_map(|a| a.split_once('=').map(|(k, v)| (k.to_string(), v.to_string())))
        .collect()
}

fn get<T: std::str::FromStr>(m: &HashMap<String, String>, k: &str, d: T) -> T {
    m.get(k).and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn params(m: &HashMap<String, String>) -> ApproxParams {
    let n_mod: u64 = get(m, "N", 3127);
    let n = 64 - n_mod.leading_zeros() as usize;
    let g: u64 = get(m, "g", 2);
    assert_eq!(gcd(g, n_mod), 1, "g must be coprime to N");
    let w = [
        get(m, "w1", 2usize),
        get(m, "w3a", 2usize),
        get(m, "w3b", 2usize),
        get(m, "w4", 2usize),
    ];
    let f: usize = get(m, "f", n.saturating_sub(2).max(4));
    let mode = m.get("mode").map(String::as_str).unwrap_or("shor");
    let mut p = match mode {
        "eh" => ApproxParams::eh_s(n_mod, g, get(m, "s", 1usize), w, f, 0),
        _ => ApproxParams::shor(n_mod, g, get(m, "m", 2 * n), w, f, 0),
    };
    p.min_gap = get(m, "gap", f);
    p.prime_bits = m.get("ell").and_then(|v| v.parse().ok());
    p.min_search_primes = get(m, "minprimes", 100);
    if let Some(v) = m.get("primes") {
        p.forced_periods = Some(v.split(',').map(|x| x.parse().expect("primes")).collect());
    }
    // the paper (§2.4): if the prime search fails, increment the prime length ℓ
    if p.prime_bits.is_none() {
        let mut q = p.clone();
        q.mask_bits = 0;
        let est = {
            let nw1: usize = q.regs.iter().map(|r| r.0.div_ceil(q.w1)).sum();
            qsim_lab::shor::approx::estimate_prime_bits(q.n_mod, nw1)
        };
        for ell in est..est + 6 {
            q.prime_bits = Some(ell);
            match ApproxConfig::new(&q) {
                Ok(_) => {
                    p.prime_bits = Some(ell);
                    break;
                }
                Err(e) => eprintln!("# ell={ell}: {e}; trying ell={}", ell + 1),
            }
        }
    }
    // mask bits: a number, or "paper" (the paper's estimated_ideal_mask_bits)
    match m.get("mask").map(String::as_str) {
        Some("paper") | None => {
            let c = ApproxConfig::new(&ApproxParams {
                mask_bits: 0,
                ..p.clone()
            })
            .expect("precompute");
            p.mask_bits = paper_mask_bits(c.periods.len(), c.nw4, f).min(f - 1);
        }
        Some(v) => p.mask_bits = v.parse().expect("mask"),
    }
    p
}

fn describe(c: &ApproxConfig) {
    let p = &c.params;
    let (eps, s, pdev) = c.paper_deviation_model();
    println!(
        "N={} n={} g={} regs={:?} m={} w=({},{},{},{}) f={} mask={} gap={} ell={} |P|={} \
         L_bits={} LmodN={} dropped={} T={} nw1={} nw3a={} nw3b={} nw4={} len_dlog={} additions={}",
        p.n_mod,
        64 - p.n_mod.leading_zeros(),
        p.generator,
        p.regs,
        c.m,
        p.w1,
        p.w3a,
        p.w3b,
        p.w4,
        p.len_acc,
        p.mask_bits,
        p.min_gap,
        c.ell,
        c.periods.len(),
        c.l_bits,
        c.l_mod_n,
        c.dropped,
        c.trunc,
        c.nw1,
        c.nw3a,
        c.nw3b,
        c.nw4,
        c.len_dlog,
        c.accumulator_additions()
    );
    println!("paper_model eps={eps:.4e} S={s:.4e} P_deviant_bound={pdev:.4e}");
    println!("periods={:?}", c.periods);
}

fn print_verify(tag: &str, v: &Verify, secs: f64) {
    println!(
        "{tag} branches={} bad_sign={} dirty={} bad_residue={} bad_acc={} bad_e={} bad_formula={} \
         max_dev={} mean_dev={:.4} max_mod_dev={:.4e} secs={secs:.2}",
        v.branches,
        v.bad_sign,
        v.dirty,
        v.bad_residue,
        v.bad_acc,
        v.bad_e,
        v.bad_formula,
        v.max_dev,
        v.mean_dev,
        v.max_mod_dev
    );
}

fn hist_str(h: &[u64]) -> String {
    h.iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(i, c)| format!("{}:{}", i as i64 - 64, c))
        .collect::<Vec<_>>()
        .join(",")
}

/// (mean, std, min, max) of the signed deviation from its histogram (±64 clamp).
fn hist_stats(h: &[u64]) -> (f64, f64, i64, i64) {
    let n: u64 = h.iter().sum();
    let mut mean = 0.0;
    let (mut lo, mut hi) = (i64::MAX, i64::MIN);
    for (i, &c) in h.iter().enumerate() {
        let d = i as i64 - 64;
        mean += d as f64 * c as f64;
        if c > 0 {
            lo = lo.min(d);
            hi = hi.max(d);
        }
    }
    mean /= n.max(1) as f64;
    let var = h
        .iter()
        .enumerate()
        .map(|(i, &c)| (i as f64 - 64.0 - mean).powi(2) * c as f64)
        .sum::<f64>()
        / n.max(1) as f64;
    (mean, var.sqrt(), lo, hi)
}

/// Order of g mod N.
fn order(g: u64, n: u64) -> u64 {
    let mut x = g % n;
    let mut r = 1;
    while x != 1 {
        x = (u128::from(x) * u128::from(g) % u128::from(n)) as u64;
        r += 1;
    }
    r
}

/// Textbook Shor distribution of a `2^m`-point QFT for `f(e) = g^e` (closed form).
fn shor_unmasked(m: usize, r: u64) -> Vec<f64> {
    let size = 1u64 << m;
    let q = size / r;
    let rem = size % r;
    let pi = std::f64::consts::PI;
    (0..size)
        .into_par_iter()
        .map(|j| {
            let x = (u128::from(r) * u128::from(j) % u128::from(size)) as f64;
            let g2 = |n: u64| -> f64 {
                if x == 0.0 {
                    (n * n) as f64
                } else {
                    let a = (pi * n as f64 * x / size as f64).sin();
                    let b = (pi * x / size as f64).sin();
                    (a / b) * (a / b)
                }
            };
            (rem as f64 * g2(q + 1) + (r - rem) as f64 * g2(q)) / (size as f64 * size as f64)
        })
        .collect()
}

struct Dists {
    actual: Vec<f64>,
    ideal: Vec<f64>,
    pv_actual: Vec<f64>,
    pv_ideal: Vec<f64>,
}

fn run_dists(c: &ApproxConfig, ft: &[u32]) -> Dists {
    let p = &c.params;
    let (ma, mb) = if p.regs.len() == 2 {
        (p.regs[0].0, p.regs[1].0)
    } else {
        (c.m, 0)
    };
    let w = 1u64 << p.mask_bits;
    let fi: Vec<u32> = (0..1u64 << c.m)
        .into_par_iter()
        .map(|e| c.ideal_trunc(e) as u32)
        .collect();
    let (actual, pv_actual) = distribution(ft, c.trunc, w, ma, mb);
    let (ideal, pv_ideal) = distribution(&fi, c.trunc, w, ma, mb);
    Dists {
        actual,
        ideal,
        pv_actual,
        pv_ideal,
    }
}

fn success_shor(m: usize, n_mod: u64, g: u64, d: &[f64], repo: bool) -> f64 {
    d.par_iter()
        .enumerate()
        .filter(|(_, &p)| p > 0.0)
        .map(|(j, &p)| {
            let ok = if repo {
                qsim_lab::shor::postprocess(n_mod, g, j as u128, m as u32)
                    .1
                    .is_some()
            } else {
                paper_success(j as u64, m, n_mod, g)
            };
            if ok {
                p
            } else {
                0.0
            }
        })
        .sum()
}

fn success_eh(ma: usize, n_mod: u64, g: u64, ds: &[&[f64]], floor: f64) -> (Vec<f64>, f64) {
    // one post-processing per (j, k) with mass above `floor` in any distribution
    let na = 1usize << ma;
    let size = ds[0].len();
    let ok: Vec<bool> = (0..size)
        .into_par_iter()
        .map(|idx| {
            if ds.iter().all(|d| d[idx] <= floor) {
                return false;
            }
            let (j, k) = ((idx % na) as u128, (idx / na) as u128);
            ge::eh_postprocess(n_mod, g, j, k, 4096).0.is_some()
        })
        .collect();
    let succ = ds
        .iter()
        .map(|d| d.iter().zip(&ok).filter(|(_, &o)| o).map(|(p, _)| p).sum())
        .collect();
    // mass skipped (below floor) bounds the error of each success value
    let skipped = ds
        .iter()
        .map(|d| d.iter().filter(|&&p| p <= floor).sum::<f64>())
        .fold(0.0, f64::max);
    (succ, skipped)
}

/// Peak masses for Shor-style outcomes: peak `k = round(j r / 2^m) mod r`.
fn peaks(d: &[f64], m: usize, r: u64) -> Vec<f64> {
    let size = 1u128 << m;
    let mut q = vec![0f64; r as usize];
    for (j, &p) in d.iter().enumerate() {
        let k = ((j as u128 * u128::from(r) * 2 + size) / (2 * size)) % u128::from(r);
        q[k as usize] += p;
    }
    q
}

fn cmd_dist(a: &HashMap<String, String>) {
    let p = params(a);
    let t0 = Instant::now();
    let c = ApproxConfig::new(&p).expect("precompute");
    describe(&c);
    let seed: u64 = get(a, "seed", 1);
    let pl = plan(&c, &mut RandomOutcomes(seed));
    let (v, ft) = evaluate(&c, &pl);
    print_verify("verify", &v, t0.elapsed().as_secs_f64());
    println!("dev_hist {}", hist_str(&v.dev_hist));
    let (dm, ds, dlo, dhi) = hist_stats(&v.dev_hist);
    println!("dev_stats mean={dm:.4} std={ds:.4} min={dlo} max={dhi} (units of 2^t; spread max-min={})", dhi - dlo);
    let t1 = Instant::now();
    let d = run_dists(&c, &ft);
    let w = 1u64 << p.mask_bits;
    let fi: Vec<u32> = (0..1u64 << c.m).map(|e| c.ideal_trunc(e) as u32).collect();
    let ov = overlap(&ft, &fi, c.trunc, w);
    println!(
        "fidelity |<ideal|actual>|={ov:.10} infidelity(1-|.|^2)={:.6e} tv_bound(sqrt(1-F))={:.6e} max_dev/W={:.6e} mean_dev/W={:.6e}",
        1.0 - ov * ov,
        (1.0 - ov * ov).max(0.0).sqrt(),
        v.max_dev as f64 / w as f64,
        v.mean_dev / w as f64
    );
    let (cf, outside) = cond_fidelity(&fi, &ft, c.trunc, w);
    println!("conditional exponent-state fidelity sum_V P(V)|<psi_V|psi~_V>|^2={cf:.10} P(V outside ideal support)={outside:.3e}");
    let (shift, ovs) = best_shift(&ft, &fi, c.trunc, w);
    let fis = shifted(&fi, shift, c.trunc);
    let (cfs, outs) = cond_fidelity(&fis, &ft, c.trunc, w);
    println!(
        "best constant shift c={shift}: |<ideal+c|actual>|={ovs:.10} infidelity={:.6e} tv_bound={:.6e} cond_fidelity={cfs:.10} P(outside)={outs:.3e}",
        1.0 - ovs * ovs,
        (1.0 - ovs * ovs).max(0.0).sqrt()
    );
    println!(
        "tv(actual,ideal_masked)={:.6e} tv_V(actual,ideal)={:.6e} dist_secs={:.1}",
        tv(&d.actual, &d.ideal),
        tv(&d.pv_actual, &d.pv_ideal),
        t1.elapsed().as_secs_f64()
    );
    let n_mod = p.n_mod;
    let g = p.generator;
    let r = order(g, n_mod);
    println!("order r={r}");
    let unmasked = get(a, "unmasked", 1u32) == 1;
    if p.regs.len() == 1 {
        let un = if unmasked {
            Some(shor_unmasked(c.m, r))
        } else {
            None
        };
        for (name, repo) in [("paper", false), ("repo", true)] {
            let sa = success_shor(c.m, n_mod, g, &d.actual, repo);
            let si = success_shor(c.m, n_mod, g, &d.ideal, repo);
            let su = un.as_ref().map(|u| success_shor(c.m, n_mod, g, u, repo));
            println!(
                "success[{name}] actual={sa:.10} ideal_masked={si:.10} unmasked={}",
                su.map_or("-".into(), |x| format!("{x:.10}"))
            );
        }
        if let Some(u) = &un {
            println!(
                "tv(actual,unmasked)={:.6e} tv(ideal_masked,unmasked)={:.6e}",
                tv(&d.actual, u),
                tv(&d.ideal, u)
            );
            if r <= 4096 {
                let qa = peaks(&d.actual, c.m, r);
                let qi = peaks(&d.ideal, c.m, r);
                let qu = peaks(u, c.m, r);
                println!(
                    "peaks k0: actual={:.6} ideal={:.6} unmasked={:.6} (Eq.42 random-R model w/P = {:.6})",
                    qa[0],
                    qi[0],
                    qu[0],
                    // w/P with w = |R| ~ r·S: the expected kept fraction
                    (w as f64 * (1u64 << c.dropped) as f64 / n_mod as f64).min(1.0)
                );
                println!("peaks tv(actual,ideal)={:.6e} tv(actual,unmasked)={:.6e} tv(ideal,unmasked)={:.6e}",
                    tv(&qa, &qi), tv(&qa, &qu), tv(&qi, &qu));
                if let Some(path) = a.get("peaks_out") {
                    let mut fh = std::fs::File::create(path).unwrap();
                    writeln!(fh, "k,actual,ideal_masked,unmasked").unwrap();
                    for k in 0..r as usize {
                        writeln!(fh, "{k},{:.12e},{:.12e},{:.12e}", qa[k], qi[k], qu[k]).unwrap();
                    }
                }
            }
        }
    } else {
        let ma = p.regs[0].0;
        let mut ds: Vec<&[f64]> = vec![&d.actual, &d.ideal];
        let un;
        if unmasked {
            let fe: Vec<u32> = (0..1u64 << c.m)
                .into_par_iter()
                .map(|e| c.exact(e) as u32)
                .collect();
            un = distribution(&fe, n_mod, 1, ma, p.regs[1].0).0;
            ds.push(&un);
            println!(
                "tv(actual,unmasked)={:.6e} tv(ideal_masked,unmasked)={:.6e}",
                tv(&d.actual, &un),
                tv(&d.ideal, &un)
            );
        }
        // Ekerå–Håstad pair quality: alpha = {d j + 2^m k} mod 2^{m+l}, centred
        let mbits = ge::eh_m(n_mod);
        let pf = (2..n_mod).find(|&x| n_mod % x == 0).unwrap();
        let d = (pf + n_mod / pf - 2) / 2;
        let lb = p.regs[1].0;
        let modl = 1u128 << (mbits + lb);
        let na = 1usize << ma;
        for (name, dd) in ["actual", "ideal_masked", "unmasked"].iter().zip(&ds) {
            let mut q = [0f64; 4];
            for (idx, &pr) in dd.iter().enumerate() {
                let (j, k) = ((idx % na) as u128, (idx / na) as u128);
                let a = (u128::from(d) * j + (k << mbits)) % modl;
                let a = a.min(modl - a);
                for (t, qq) in q.iter_mut().enumerate() {
                    if a <= (1u128 << (mbits + t)) >> 2 {
                        *qq += pr;
                    }
                }
            }
            println!(
                "eh_alpha[{name}] P(|a|<=2^(m-2))={:.8} P(<=2^(m-1))={:.8} P(<=2^m)={:.8} P(<=2^(m+1))={:.8}",
                q[0], q[1], q[2], q[3]
            );
        }
        if lb == mbits {
            let (succ, skipped) = success_eh(ma, n_mod, g, &ds, 1e-13);
            println!(
                "success[eh] actual={:.10} ideal_masked={:.10} unmasked={} (mass below 1e-13 skipped <= {skipped:.2e})",
                succ[0],
                succ[1],
                succ.get(2).map_or("-".into(), |x| format!("{x:.10}"))
            );
        }
    }
    if let Some(path) = a.get("dist_out") {
        let mut fh = std::fs::File::create(path).unwrap();
        writeln!(fh, "j,actual,ideal_masked").unwrap();
        for (j, (x, y)) in d.actual.iter().zip(&d.ideal).enumerate() {
            if *x > 0.0 || *y > 0.0 {
                writeln!(fh, "{j},{x:.15e},{y:.15e}").unwrap();
            }
        }
    }
    println!("total_secs={:.1}", t0.elapsed().as_secs_f64());
}

fn cmd_verify(a: &HashMap<String, String>) {
    let p = params(a);
    let c = ApproxConfig::new(&p).expect("precompute");
    describe(&c);
    let seeds: u64 = get(a, "seeds", 8);
    let base: u64 = get(a, "seed", 1);
    let mut all_ok = true;
    let mut f0: Option<Vec<u32>> = None;
    let runs: Vec<(String, Box<dyn Outcomes>)> = {
        let mut v: Vec<(String, Box<dyn Outcomes>)> = vec![
            ("zero".into(), Box::new(ConstOutcomes(false))),
            ("one".into(), Box::new(ConstOutcomes(true))),
        ];
        for s in 0..seeds {
            v.push((
                format!("seed{}", base + s),
                Box::new(RandomOutcomes(base + s)),
            ));
        }
        v
    };
    for (name, mut o) in runs {
        let t0 = Instant::now();
        let pl = plan(&c, &mut *o);
        let (v, ft) = evaluate(&c, &pl);
        print_verify(
            &format!(
                "verify[{name}] ops={} draws={} bits={} fixups={}",
                pl.ops.len(),
                pl.draws,
                pl.draw_bits,
                pl.fixups
            ),
            &v,
            t0.elapsed().as_secs_f64(),
        );
        let ok = v.bad_sign == 0
            && v.dirty == 0
            && v.bad_residue == 0
            && v.bad_acc == 0
            && v.bad_e == 0
            && v.bad_formula == 0;
        all_ok &= ok;
        if let Some(f) = &f0 {
            if *f != ft {
                println!("  F̃ differs between outcome streams!");
                all_ok = false;
            }
        } else {
            println!("dev_hist {}", hist_str(&v.dev_hist));
            f0 = Some(ft);
        }
    }
    println!("ALL_OK={all_ok}");
}

fn cmd_dump(a: &HashMap<String, String>) {
    let p = params(a);
    let c = ApproxConfig::new(&p).expect("precompute");
    let dir = a.get("out").expect("out=DIR");
    std::fs::create_dir_all(dir).unwrap();
    let mut fh = std::fs::File::create(format!("{dir}/rust_config.txt")).unwrap();
    let p = &c.params;
    writeln!(fh, "modulus = {}", p.n_mod).unwrap();
    writeln!(fh, "generator = {}", p.generator).unwrap();
    writeln!(fh, "num_input_qubits = {}", c.m).unwrap();
    writeln!(fh, "window1 = {}", p.w1).unwrap();
    writeln!(fh, "window3a = {}", p.w3a).unwrap();
    writeln!(fh, "window3b = {}", p.w3b).unwrap();
    writeln!(fh, "window4 = {}", p.w4).unwrap();
    writeln!(fh, "len_accumulator = {}", p.len_acc).unwrap();
    writeln!(fh, "mask_bits = {}", p.mask_bits).unwrap();
    writeln!(fh, "min_wraparound_gap = {}", p.min_gap).unwrap();
    writeln!(fh, "rns_primes_bit_length = {}", c.ell).unwrap();
    writeln!(fh, "periods = {:?}", c.periods).unwrap();
    writeln!(fh, "generators = {:?}", c.generators).unwrap();
    let flat1: Vec<u64> = c
        .table1
        .iter()
        .flatten()
        .flatten()
        .map(|&x| x & 0xFFFF_FFFF)
        .collect();
    writeln!(fh, "table1 = {flat1:?}").unwrap();
    let f3a: Vec<u64> = c.table3a.iter().flatten().flatten().flatten().copied().collect();
    let f3b: Vec<u64> = c.table3b.iter().flatten().flatten().flatten().copied().collect();
    let f3c: Vec<u64> = c.table3c.iter().flatten().copied().collect();
    let f4: Vec<u64> = c.table4.iter().flatten().flatten().copied().collect();
    writeln!(fh, "table3a = {f3a:?}").unwrap();
    writeln!(fh, "table3b = {f3b:?}").unwrap();
    writeln!(fh, "table3c = {f3c:?}").unwrap();
    writeln!(fh, "table4 = {f4:?}").unwrap();
    println!("wrote {dir}/rust_config.txt");
}

/// Replays outcomes recorded by the Python quantum backend and writes the
/// final `(e, s) -> (acc, sign)` map.
fn cmd_replay(a: &HashMap<String, String>) {
    let p = params(a);
    let c = ApproxConfig::new(&p).expect("precompute");
    let text = std::fs::read_to_string(a.get("log").expect("log=FILE")).unwrap();
    let log: Vec<(u64, usize)> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let (v, n) = l.split_once(',').unwrap();
            (v.trim().parse().unwrap(), n.trim().parse().unwrap())
        })
        .collect();
    let mut o = ReplayOutcomes { log, pos: 0 };
    let pl = plan(&c, &mut o);
    assert_eq!(o.pos, o.log.len(), "not every recorded outcome was used");
    let (v, ft) = evaluate(&c, &pl);
    print_verify("replay", &v, 0.0);
    let mut fh = std::fs::File::create(a.get("out").expect("out=FILE")).unwrap();
    writeln!(fh, "e,F").unwrap();
    for (e, f) in ft.iter().enumerate() {
        writeln!(fh, "{e},{f}").unwrap();
    }
    // also the final per-branch sign for every (e, s), as a check of the
    // all-branch evaluator against the scalar one
    let mut bad = 0;
    for e in 0..1u64 << c.m {
        for s in 0..1u64 << p.mask_bits {
            let (r, sign, fails) = qsim_lab::shor::approx::eval_branch(&c, &pl, e, s);
            if sign || fails != 0 || r[1] != (s + u64::from(ft[e as usize])) % c.trunc {
                bad += 1;
            }
        }
    }
    println!("scalar_recheck_bad={bad}");
}

fn cmd_gate(a: &HashMap<String, String>) {
    // one loop4 window step at the gate level vs the quint-level semantics,
    // on every input (k, acc) and for many outcome streams
    let w4: usize = get(a, "w4", 2);
    let f: usize = get(a, "f", 6);
    let trials: u64 = get(a, "trials", 64);
    let trunc: u64 = get(a, "T", (1u64 << f) - 3);
    assert!(trunc < 1 << f && trunc >= 1 << (f - 1));
    let lay = Loop4Layout::new(w4, f);
    let mut bad = 0u64;
    let mut checked = 0u64;
    let mut counts = qsim_lab::shor::mbu::MbuCounts::default();
    for trial in 0..trials {
        // a random table of negated entries < T (as table4)
        let mut rng = RandomOutcomes(trial * 7919 + 13);
        let table: Vec<u64> = (0..1u64 << w4).map(|_| rng.draw(f) % trunc).collect();
        let mut bits = RandomOutcomes(trial * 31 + 1);
        let ops: Vec<MbuOp> = gate_loop4_step(&lay, &table, trunc, &mut || bits.draw(1) == 1);
        if trial == 0 {
            counts = qsim_lab::shor::mbu::MbuCounts::of(&ops);
        }
        let mut sign0 = None;
        for k in 0..1u64 << w4 {
            for acc in 0..trunc {
                let mut key = 0u128;
                for (i, &q) in lay.k.iter().enumerate() {
                    key |= u128::from((k >> i) & 1) << q;
                }
                for (i, &q) in lay.acc.iter().enumerate() {
                    key |= u128::from((acc >> i) & 1) << q;
                }
                let (out, s) = eval_on_key(&ops, key);
                // quint level: acc' = (acc − T[k]) mod trunc, everything else unchanged
                let want_acc = (acc + trunc - table[k as usize]) % trunc;
                let mut want = 0u128;
                for (i, &q) in lay.k.iter().enumerate() {
                    want |= u128::from((k >> i) & 1) << q;
                }
                for (i, &q) in lay.acc.iter().enumerate() {
                    want |= u128::from((want_acc >> i) & 1) << q;
                }
                checked += 1;
                let s0 = *sign0.get_or_insert(s);
                if out != want || s != s0 {
                    bad += 1;
                }
            }
        }
    }
    println!(
        "gate_loop4 w4={w4} f={f} T={trunc} qubits={} trials={trials} inputs_checked={checked} bad={bad} \
         counts(trial0): toffoli={} cnot={} x={} fixup={} meas={}",
        lay.nq, counts.toffoli, counts.cnot, counts.x, counts.fixup, counts.meas
    );
}

fn cmd_config(a: &HashMap<String, String>) {
    let p = params(a);
    match ApproxConfig::new(&p) {
        Ok(c) => {
            describe(&c);
            let pl = plan(&c, &mut RandomOutcomes(1));
            println!(
                "plan ops={} draws={} draw_bits={} fixups={}",
                pl.ops.len(),
                pl.draws,
                pl.draw_bits,
                pl.fixups
            );
        }
        Err(e) => println!("ERROR {e}"),
    }
}

/// Parameter sweep: one CSV row per configuration (verification + exact
/// distributions + success).
fn cmd_sweep(a: &HashMap<String, String>) {
    let key = a.get("key").expect("key=f|mask|w1|w3|w4").clone();
    let vals: Vec<String> = a
        .get("vals")
        .expect("vals=a,b,c")
        .split(',')
        .map(String::from)
        .collect();
    let out = a.get("out").cloned();
    let mut rows = Vec::new();
    let header = "key,val,N,g,mode,m,f,mask,w1,w3a,w3b,w4,ell,P,additions,max_dev,mean_dev,max_mod_dev,\
                  paper_eps,paper_S,paper_Pdev,overlap,infidelity,shift,overlap_shift,infidelity_shift,\
                  tv_actual_ideal,succ_actual,succ_ideal,bad_sign,dirty,bad_acc,bad_formula,secs,\
                  dev_mean,dev_std,dev_min,dev_max";
    println!("{header}");
    for v in vals {
        let mut b = a.clone();
        match key.as_str() {
            "w3" => {
                b.insert("w3a".into(), v.clone());
                b.insert("w3b".into(), v.clone());
            }
            k => {
                b.insert(k.into(), v.clone());
            }
        }
        let t0 = Instant::now();
        let p = params(&b);
        let c = match ApproxConfig::new(&p) {
            Ok(c) => c,
            Err(e) => {
                println!("# {key}={v}: {e}");
                continue;
            }
        };
        let pl = plan(&c, &mut RandomOutcomes(get(&b, "seed", 1)));
        let (ver, ft) = evaluate(&c, &pl);
        let d = run_dists(&c, &ft);
        let w = 1u64 << p.mask_bits;
        let fi: Vec<u32> = (0..1u64 << c.m).map(|e| c.ideal_trunc(e) as u32).collect();
        let ov = overlap(&ft, &fi, c.trunc, w);
        let (shift, ovs) = best_shift(&ft, &fi, c.trunc, w);
        let (sa, si) = if p.regs.len() == 1 {
            (
                success_shor(c.m, p.n_mod, p.generator, &d.actual, false),
                success_shor(c.m, p.n_mod, p.generator, &d.ideal, false),
            )
        } else {
            let (s, _) = success_eh(p.regs[0].0, p.n_mod, p.generator, &[&d.actual, &d.ideal], 1e-13);
            (s[0], s[1])
        };
        let (eps, s, pdev) = c.paper_deviation_model();
        let hs = hist_stats(&ver.dev_hist);
        let row = format!(
            "{key},{v},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.6},{:.6e},{:.6e},{:.6e},{:.6e},{:.10},{:.6e},{},{:.10},{:.6e},{:.6e},{:.10},{:.10},{},{},{},{},{:.1},{:.4},{:.4},{},{}",
            p.n_mod,
            p.generator,
            if p.regs.len() == 2 { "eh" } else { "shor" },
            c.m,
            p.len_acc,
            p.mask_bits,
            p.w1,
            p.w3a,
            p.w3b,
            p.w4,
            c.ell,
            c.periods.len(),
            c.accumulator_additions(),
            ver.max_dev,
            ver.mean_dev,
            ver.max_mod_dev,
            eps,
            s,
            pdev,
            ov,
            1.0 - ov * ov,
            shift,
            ovs,
            1.0 - ovs * ovs,
            tv(&d.actual, &d.ideal),
            sa,
            si,
            ver.bad_sign,
            ver.dirty,
            ver.bad_acc,
            ver.bad_formula,
            t0.elapsed().as_secs_f64(),
            hs.0,
            hs.1,
            hs.2,
            hs.3
        );
        println!("{row}");
        rows.push(row);
    }
    if let Some(path) = out {
        let mut fh = std::fs::File::create(path).unwrap();
        writeln!(fh, "{header}").unwrap();
        for r in rows {
            writeln!(fh, "{r}").unwrap();
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().cloned().unwrap_or_default();
    let a = kv(&args[1.min(args.len())..]);
    match cmd.as_str() {
        "config" => cmd_config(&a),
        "verify" => cmd_verify(&a),
        "dist" => cmd_dist(&a),
        "dump" => cmd_dump(&a),
        "replay" => cmd_replay(&a),
        "gate" => cmd_gate(&a),
        "sweep" => cmd_sweep(&a),
        _ => eprintln!("usage: approx_modexp config|verify|dist|dump|replay|gate|sweep key=value..."),
    }
    let _ = pow_mod(2, 3, 5);
}
