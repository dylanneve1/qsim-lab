//! Gidney–Ekerå techniques in the exact gate-level Shor simulation
//! (exp/ge-shor, research/shor/ge-shor.md).
//!
//! ```text
//! ge_shor run    <N> <seed> <we> <wm> <lookups|all|none> [shor|eh|eh-odd|shor-odd] [f32|f64] [runs]
//! ge_shor counts <N> <seed> <we> <wm> <lookups|all|none> [shor|eh|eh-odd] [coset c]
//! ge_shor coset  <N> <a> <we> <wm> <cmax>          exact distributions (t ≤ 26)
//! ge_shor cosetmc <N> <a> <we> <wm> <c> <paths> <seed>
//! ge_shor ehmc   <N> <runs> <seed> <we> <wm>       Monte-Carlo EH vs Shor success
//! ge_shor info   <N> <seed>                        the base rule's h, g = h^(2^n), y (no factors used)
//! ge_shor dump   <N> <seed> <we> <wm> <lookups|all|none> <shor|eh|eh-odd|shor-odd> <window>
//!                                                  resolved ops of one window block (text)
//! ```
//! `run` picks the base like `qsim run shor --seed S --tries 1`
//! (`StdRng::seed_from_u64(S)`, `random_range(2..N−1)`) and draws the
//! measurement outcomes from the same stream. `eh-odd` and `shor-odd` use
//! `g = h^(2^n) mod N` (odd order; no factorisation needed); `shor-odd`
//! runs Shor's order finding on `g` and factors with Miller's reduction
//! from the multiple `2^n·ord(g)` of `ord(h)` (research/shor/shor-xl.md).
use qsim_lab::algorithms::gcd;
use qsim_lab::shor;
use qsim_lab::shor::ge::{self as shor_ge, GeOpts};
use qsim_lab::shor::mbu::MbuOpts;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn mbu(s: &str) -> MbuOpts {
    match s {
        "all" => MbuOpts::ALL,
        "lookups" => MbuOpts::LOOKUPS,
        "none" => MbuOpts::NONE,
        _ => panic!("mbu option: all | lookups | none"),
    }
}

fn arg<T: std::str::FromStr>(v: &[String], i: usize) -> T
where
    T::Err: std::fmt::Debug,
{
    v[i].parse().unwrap()
}

fn base(n_mod: u64, seed: u64) -> (u64, StdRng) {
    let mut rng = StdRng::seed_from_u64(seed);
    let a = rng.random_range(2..n_mod - 1);
    assert_eq!(gcd(a, n_mod), 1, "seed {seed} draws a non-coprime base");
    (a, rng)
}

fn order(n_mod: u64, a: u64) -> u64 {
    let mut r = 1;
    let mut x = a % n_mod;
    while x != 1 {
        x = shor::mul_mod(x, a, n_mod);
        r += 1;
    }
    r
}

fn print_run(r: &shor_ge::GeRun, secs: f64) {
    println!(
        "qubits={}  windows={}  toffoli={}  gates={}  meas={}  fixups={}  steps={}  peak_support={}  peak_branches={}  gate_branch_ops={:.3e}",
        r.qubits,
        r.windows,
        r.counts.toffoli,
        r.counts.total,
        r.counts.meas,
        r.counts.fixup,
        r.steps,
        r.peak,
        r.peak_branches,
        r.gate_branch_ops as f64
    );
    println!(
        "time {:.3} s  (build {:.3}, eval {:.3}, sort {:.3}, probs {:.3}, new state {:.3})",
        secs, r.prof[0], r.prof[1], r.prof[2], r.prof[3], r.prof[4]
    );
}

fn slicebench<const L: usize>(
    prog: &qsim_lab::shor::sliced::SlicedProgram,
    lay: &shor_ge::GeLayout,
    n_mod: u64,
    reps: usize,
) {
    use qsim_lab::shor::sliced::{SliceBuf, SliceIsa};
    let mut rng = StdRng::seed_from_u64(99);
    let mut w0 = vec![[0u64; L]; lay.nq + 2];
    w0[lay.nq] = [u64::MAX; L];
    for i in 0..64 * L {
        let x = rng.random_range(0..n_mod);
        for (j, &q) in lay.x.iter().enumerate() {
            w0[q][i / 64] |= ((x >> j) & 1) << (i % 64);
        }
        for &q in &lay.e {
            w0[q][i / 64] |= u64::from(rng.random::<bool>()) << (i % 64);
        }
    }
    let mut want = None;
    for isa in [SliceIsa::Scalar, SliceIsa::Avx2, SliceIsa::Avx512] {
        if !isa.available() || (isa == SliceIsa::Avx512 && L % 8 != 0) {
            continue;
        }
        let mut buf = SliceBuf::<L>::new(lay.nq + 2);
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t0 = Instant::now();
            for _ in 0..reps {
                buf.words().copy_from_slice(&w0);
                prog.eval_with::<L>(isa, buf.words());
            }
            best = best.min(t0.elapsed().as_secs_f64());
        }
        let out = buf.words().to_vec();
        match &want {
            None => want = Some(out),
            Some(wv) => assert!(*wv == out, "{isa:?} differs"),
        }
        let words = (reps * prog.len() * L) as f64;
        println!(
            "L={L:2} {isa:?}: {:.3} ns per step per 64-branch word, {:.3e} gate-branch ops/s per thread",
            best / words * 1e9,
            words * 64.0 / best
        );
    }
}

fn main() {
    let v: Vec<String> = std::env::args().collect();
    match v.get(1).map(String::as_str) {
        Some("run") => {
            let n_mod: u64 = arg(&v, 2);
            let seed: u64 = arg(&v, 3);
            let o = GeOpts {
                we: arg(&v, 4),
                wm: arg(&v, 5),
                mbu: mbu(&v[6]),
                coset: 0,
            };
            let var = v.get(7).map(String::as_str).unwrap_or("shor");
            let eh = var.starts_with("eh");
            let f32 = v.get(8).map(String::as_str) != Some("f64");
            // optional: up to `runs` runs on the same base until one factors N
            // (each later run continues the same measurement stream)
            let runs: usize = v.get(9).map_or(1, |s| s.parse().unwrap());
            let (h, mut rng) = base(n_mod, seed);
            let mut a = h;
            if var == "eh-odd" || var == "shor-odd" {
                // EH accepts any base: g = h^(2^n) has odd order (no
                // factorisation used), which keeps the simulated support
                // at r_odd (see research/shor/ge-shor.md)
                a = shor_ge::pow2k(a, shor::work_bits(n_mod), n_mod);
            }
            for run in 1..=runs {
                let t0 = Instant::now();
                let mut draw = || rng.random::<f64>();
                let done = if var == "shor-odd" {
                    let (r, order, _) = if f32 {
                        shor_ge::shor_run::<f32>(n_mod, a, &o, &mut draw)
                    } else {
                        shor_ge::shor_run::<f64>(n_mod, a, &o, &mut draw)
                    };
                    let secs = t0.elapsed().as_secs_f64();
                    let k = shor::work_bits(n_mod);
                    let factor =
                        order.and_then(|r| shor_ge::factor_from_power_order(n_mod, h, k, r));
                    println!(
                        "Shor-odd run {run}  N={n_mod}  h={h}  g=h^(2^{k})={a}  measured={}  order(g)={order:?}  factor={factor:?}",
                        r.y[0]
                    );
                    print_run(&r, secs);
                    factor.is_some()
                } else if eh {
                    let (r, f) = if f32 {
                        shor_ge::eh_run::<f32>(n_mod, a, &o, &mut draw)
                    } else {
                        shor_ge::eh_run::<f64>(n_mod, a, &o, &mut draw)
                    };
                    let secs = t0.elapsed().as_secs_f64();
                    println!(
                        "EH run {run}  N={n_mod}  h={h}  g={a}  m={}  exponent bits={}  j={}  k={}  factors={f:?}",
                        shor_ge::eh_m(n_mod),
                        3 * shor_ge::eh_m(n_mod),
                        r.y[1],
                        r.y[0]
                    );
                    print_run(&r, secs);
                    f.is_some()
                } else {
                    let (r, order, factor) = if f32 {
                        shor_ge::shor_run::<f32>(n_mod, a, &o, &mut draw)
                    } else {
                        shor_ge::shor_run::<f64>(n_mod, a, &o, &mut draw)
                    };
                    let secs = t0.elapsed().as_secs_f64();
                    println!(
                        "Shor run {run}  N={n_mod}  a={a}  measured={}  order={order:?}  factor={factor:?}",
                        r.y[0]
                    );
                    print_run(&r, secs);
                    factor.is_some()
                };
                if done {
                    break;
                }
            }
        }
        Some("info") => {
            // the base rule only: nothing here uses the factors of N
            let n_mod: u64 = arg(&v, 2);
            let seed: u64 = arg(&v, 3);
            let (h, _) = base(n_mod, seed);
            let k = shor::work_bits(n_mod);
            let g = shor_ge::pow2k(h, k, n_mod);
            println!(
                "N={n_mod} bits={} n={k} seed={seed} h={h} g=h^(2^{k})={g} eh_m={} y=g^((N-1)/2)={}",
                64 - n_mod.leading_zeros(),
                shor_ge::eh_m(n_mod),
                shor_ge::eh_target(n_mod, g)
            );
        }
        Some("dump") => {
            let n_mod: u64 = arg(&v, 2);
            let seed: u64 = arg(&v, 3);
            let o = GeOpts {
                we: arg(&v, 4),
                wm: arg(&v, 5),
                mbu: mbu(&v[6]),
                coset: 0,
            };
            let var = v[7].as_str();
            let want: usize = arg(&v, 8);
            let (mut a, _) = base(n_mod, seed);
            if var == "eh-odd" || var == "shor-odd" {
                a = shor_ge::pow2k(a, shor::work_bits(n_mod), n_mod);
            }
            let regs = if var.starts_with("eh") {
                shor_ge::eh_regs(n_mod, a)
            } else {
                shor_ge::shor_regs(n_mod, a)
            };
            let lay = shor_ge::GeLayout::new(shor::work_bits(n_mod), &o);
            let mut wi = 0usize;
            for reg in &regs {
                for (i0, w) in shor_ge::windows(reg.len, o.we) {
                    if wi == want {
                        let g = shor_ge::pow2k(reg.base, reg.len - i0 - w, n_mod);
                        let mut oc = qsim_lab::shor::mbu::Outcomes::from_env(
                            shor_ge::outcome_seed(n_mod, g, wi as u64),
                        );
                        let ops = shor_ge::window_block(&lay, g, n_mod, &o, &mut oc);
                        let list = |q: &[usize]| {
                            q.iter().map(usize::to_string).collect::<Vec<_>>().join(",")
                        };
                        println!(
                            "nq {} n {} N {n_mod} window {want} w {w} g {g} e {} x {}",
                            lay.nq,
                            lay.n,
                            list(&lay.e[..w]),
                            list(&lay.stored())
                        );
                        use qsim_lab::gate::Gate;
                        use qsim_lab::shor::mbu::MbuOp;
                        for op in &ops {
                            match *op {
                                MbuOp::G(Gate::X(t)) => println!("X {t}"),
                                MbuOp::G(Gate::Cnot(c, t)) => println!("CX {c} {t}"),
                                MbuOp::G(Gate::Ccx(a, b, t)) => println!("CCX {a} {b} {t}"),
                                MbuOp::G(Gate::Swap(a, b)) => println!("SWAP {a} {b}"),
                                MbuOp::G(Gate::Z(q)) => println!("Z {q}"),
                                MbuOp::G(Gate::Cz(a, b)) => println!("CZ {a} {b}"),
                                MbuOp::MeasX(q, m) => println!("MX {q} {}", u8::from(m)),
                                MbuOp::GlobalNeg => println!("GNEG"),
                                MbuOp::G(g) => panic!("unexpected gate {g:?}"),
                            }
                        }
                        return;
                    }
                    wi += 1;
                }
            }
            panic!("window {want} out of range ({wi} windows)");
        }
        Some("slicebench") => {
            // single-thread throughput of the slice evaluator per tier on one
            // real window block (random valid inputs, 64·L branches per batch)
            let n_mod: u64 = arg(&v, 2);
            let seed: u64 = arg(&v, 3);
            let o = GeOpts {
                we: arg(&v, 4),
                wm: arg(&v, 5),
                mbu: mbu(&v[6]),
                coset: 0,
            };
            let var = v[7].as_str();
            let window: usize = arg(&v, 8);
            let reps: usize = arg(&v, 9);
            let (mut a, _) = base(n_mod, seed);
            if var == "eh-odd" || var == "shor-odd" {
                a = shor_ge::pow2k(a, shor::work_bits(n_mod), n_mod);
            }
            let regs = if var.starts_with("eh") {
                shor_ge::eh_regs(n_mod, a)
            } else {
                shor_ge::shor_regs(n_mod, a)
            };
            let lay = shor_ge::GeLayout::new(shor::work_bits(n_mod), &o);
            let (reg, i0, w) = regs
                .iter()
                .flat_map(|r| {
                    shor_ge::windows(r.len, o.we)
                        .into_iter()
                        .map(move |(i0, w)| (r, i0, w))
                })
                .nth(window)
                .expect("window index");
            let g = shor_ge::pow2k(reg.base, reg.len - i0 - w, n_mod);
            let mut oc = qsim_lab::shor::mbu::Outcomes::from_env(shor_ge::outcome_seed(
                n_mod,
                g,
                window as u64,
            ));
            let ops = shor_ge::window_block(&lay, g, n_mod, &o, &mut oc);
            let prog = qsim_lab::shor::sliced::SlicedProgram::compile_ops(lay.nq, &ops).unwrap();
            println!(
                "N={n_mod} window {window}: qubits={} steps={} reps={reps}",
                lay.nq,
                prog.len()
            );
            slicebench::<8>(&prog, &lay, n_mod, reps);
            slicebench::<16>(&prog, &lay, n_mod, reps);
            slicebench::<32>(&prog, &lay, n_mod, reps);
            slicebench::<64>(&prog, &lay, n_mod, reps);
        }
        Some("counts") => {
            let n_mod: u64 = arg(&v, 2);
            let seed: u64 = arg(&v, 3);
            let coset = if v.get(8).map(String::as_str) == Some("coset") {
                arg(&v, 9)
            } else {
                0
            };
            let o = GeOpts {
                we: arg(&v, 4),
                wm: arg(&v, 5),
                mbu: mbu(&v[6]),
                coset,
            };
            let var = v.get(7).map(String::as_str).unwrap_or("shor");
            let eh = var.starts_with("eh");
            let (mut a, _) = base(n_mod, seed);
            if var == "eh-odd" || var == "shor-odd" {
                a = shor_ge::pow2k(a, shor::work_bits(n_mod), n_mod);
            }
            let regs = if eh {
                shor_ge::eh_regs(n_mod, a)
            } else {
                shor_ge::shor_regs(n_mod, a)
            };
            let bits: usize = regs.iter().map(|r| r.len).sum();
            let (c, steps, nq) = shor_ge::schedule_counts(n_mod, &regs, &o);
            println!(
                "N={n_mod} n={} {} we={} wm={} mbu={} coset={} exp_bits={bits} qubits={nq} toffoli={} gates={} cnot={} meas={} fixups={} steps={}",
                shor::work_bits(n_mod),
                var,
                o.we,
                o.wm,
                v[6],
                coset,
                c.toffoli,
                c.total,
                c.cnot,
                c.meas,
                c.fixup,
                steps
            );
        }
        Some("coset") => {
            let n_mod: u64 = arg(&v, 2);
            let a: u64 = arg(&v, 3);
            let (we, wm, cmax): (usize, usize, usize) = (arg(&v, 4), arg(&v, 5), arg(&v, 6));
            let regs = shor_ge::shor_regs(n_mod, a);
            let t = regs[0].len as u32;
            let exact = shor::full_qft_distribution(n_mod, a);
            let r = order(n_mod, a);
            // strict: the true order is a convergent denominator of y/2^t
            // (no multiples tried); loose: shor::postprocess finds a factor
            let succ = |d: &[f64]| -> (f64, f64) {
                let mut st = (0.0, 0.0);
                for (y, &p) in d.iter().enumerate() {
                    if p <= 0.0 {
                        continue;
                    }
                    if shor::convergents(y as u128, t).contains(&u128::from(r)) {
                        st.0 += p;
                    }
                    if shor::postprocess(n_mod, a, y as u128, t).1.is_some() {
                        st.1 += p;
                    }
                }
                st
            };
            let s0 = succ(&exact);
            println!(
                "N={n_mod} a={a} r={r} we={we} wm={wm}: exact P(strict)={:.6} P(factor)={:.6}",
                s0.0, s0.1
            );
            for c in 0..=cmax {
                let o = GeOpts {
                    we,
                    wm,
                    mbu: MbuOpts::ALL,
                    coset: c,
                };
                let t0 = Instant::now();
                let d = shor_ge::distribution(n_mod, &regs, &o, 1e-15);
                let tv: f64 = 0.5
                    * exact
                        .iter()
                        .zip(&d)
                        .map(|(x, y)| (x - y).abs())
                        .sum::<f64>();
                let (cnt, _, nq) = shor_ge::schedule_counts(n_mod, &regs, &o);
                println!(
                    "c={c} qubits={nq} toffoli={} TV={tv:.6} P(strict)={:.6} P(factor)={:.6} Σ={:.12} ({:.1} s)",
                    cnt.toffoli,
                    succ(&d).0,
                    succ(&d).1,
                    d.iter().sum::<f64>(),
                    t0.elapsed().as_secs_f64()
                );
            }
        }
        Some("cosetmc") => {
            let n_mod: u64 = arg(&v, 2);
            let a: u64 = arg(&v, 3);
            let o = GeOpts {
                we: arg(&v, 4),
                wm: arg(&v, 5),
                mbu: MbuOpts::ALL,
                coset: arg(&v, 6),
            };
            let paths: usize = arg(&v, 7);
            let seed: u64 = arg(&v, 8);
            let t = 2 * shor::work_bits(n_mod) as u32;
            let mut rng = StdRng::seed_from_u64(seed);
            let r = order(n_mod, a);
            let (mut sq, mut sp, mut tv, mut fin) = (0.0, 0.0, 0.0, Vec::new());
            let mut peak = 0;
            let t0 = Instant::now();
            for _ in 0..paths {
                let p = shor_ge::coset_path(n_mod, a, &o, &mut || rng.random::<f64>());
                let ok = shor::convergents(p.y, t).contains(&u128::from(r));
                let w = (p.ln_p - p.ln_q).exp();
                sq += f64::from(u8::from(ok));
                sp += f64::from(u8::from(ok)) * w;
                tv += (1.0 - w).max(0.0);
                fin.push(p.infidelity.clone());
                peak = peak.max(p.peak);
            }
            let k = paths as f64;
            let nw = fin[0].len();
            let mean_last: f64 = fin
                .iter()
                .map(|f| f[nw - 1])
                .filter(|x| x.is_finite())
                .sum::<f64>()
                / fin.iter().filter(|f| f[nw - 1].is_finite()).count().max(1) as f64;
            let mean_first: f64 = fin.iter().map(|f| f[0]).sum::<f64>() / k;
            println!(
                "N={n_mod} a={a} {o:?} paths={paths}: P_strict(coset)={:.4} P_strict(exact, IS)={:.4} TV={:.4} deviant weight after window 1={mean_first:.3e} after last={mean_last:.3e} peak={peak} ({:.1} s)",
                sq / k,
                sp / k,
                tv / k,
                t0.elapsed().as_secs_f64()
            );
            let mut prof = String::new();
            for i in 0..nw {
                let xs: Vec<f64> = fin.iter().map(|f| f[i]).filter(|x| x.is_finite()).collect();
                prof += &format!(" {:.3e}", xs.iter().sum::<f64>() / xs.len().max(1) as f64);
            }
            println!("mean deviant weight per window:{prof}");
        }
        Some("ehmc") => {
            let n_mod: u64 = arg(&v, 2);
            let runs: usize = arg(&v, 3);
            let seed: u64 = arg(&v, 4);
            let o = GeOpts {
                we: arg(&v, 5),
                wm: arg(&v, 6),
                mbu: MbuOpts::LOOKUPS,
                coset: 0,
            };
            let mut rng = StdRng::seed_from_u64(seed);
            let (mut eh_ok, mut shor_ord, mut shor_fac, mut done) = (0, 0, 0, 0);
            let mut eh_odd_ok = 0;
            let t0 = Instant::now();
            while done < runs {
                let g = rng.random_range(2..n_mod - 1);
                if gcd(g, n_mod) > 1 {
                    continue;
                }
                done += 1;
                let (_, f) = shor_ge::eh_run::<f64>(n_mod, g, &o, &mut || rng.random::<f64>());
                eh_ok += usize::from(f.is_some());
                let go = shor_ge::pow2k(g, shor::work_bits(n_mod), n_mod);
                if go != 1 {
                    let (_, f) = shor_ge::eh_run::<f64>(n_mod, go, &o, &mut || rng.random::<f64>());
                    eh_odd_ok += usize::from(f.is_some());
                }
                let (_, ord, fac) =
                    shor_ge::shor_run::<f64>(n_mod, g, &o, &mut || rng.random::<f64>());
                // "order found" = the true order (verified a^r = 1, minimal)
                shor_ord += usize::from(ord.is_some());
                shor_fac += usize::from(fac.is_some());
            }
            let k = runs as f64;
            let ci = |x: usize| {
                let p = x as f64 / k;
                1.96 * (p * (1.0 - p) / k).sqrt()
            };
            println!(
                "N={n_mod} runs={runs} we={} wm={}: EH P(factor | 1 run)={:.4}±{:.4}  EH odd-order base {:.4}±{:.4}   Shor P(order)={:.4}±{:.4} P(factor | 1 run)={:.4}±{:.4}   ({:.1} s)",
                o.we,
                o.wm,
                eh_ok as f64 / k,
                ci(eh_ok),
                eh_odd_ok as f64 / k,
                ci(eh_odd_ok),
                shor_ord as f64 / k,
                ci(shor_ord),
                shor_fac as f64 / k,
                ci(shor_fac),
                t0.elapsed().as_secs_f64()
            );
        }
        _ => {
            eprintln!("see the doc comment of examples/ge_shor.rs");
            std::process::exit(2);
        }
    }
}
