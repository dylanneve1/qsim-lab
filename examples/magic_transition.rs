//! Campaign driver for research/simulability/magic-transition.md.
//!
//! cargo run --release --example magic_transition -- scan n=64 pm=0.1,0.2 pt=0.05 samples=20
//!   key=value options:
//!     n, depth (default 4n), pm (comma list), pt (number) or ptn (c: p_t = c/n),
//!     samples, seed0, mode (dim|exact), maxd (exact register cap, default 26),
//!     ent (none|half|i3), magic (0|1: ν, M2 of the register when d <= magicmax=10),
//!     window (time-average window for d, default n layers), series (0|1)
//! cargo run --release --example magic_transition -- validate n=18 seeds=4
//!   state-vector cross-check at larger n (same seeds, Born outcomes).

use qsim_lab::magic_atlas::state_magic;
use qsim_lab::monitored::circuit::{self, MOp, Params};
use qsim_lab::monitored::ent::cut_entropy;
use qsim_lab::monitored::{Cliff2, Mode, Monitored};
use qsim_lab::StateVectorF64;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;
use std::time::Instant;

fn kv(args: &[String]) -> HashMap<String, String> {
    args.iter()
        .filter_map(|a| {
            a.split_once('=')
                .map(|(k, v)| (k.to_string(), v.to_string()))
        })
        .collect()
}

fn get<T: std::str::FromStr>(m: &HashMap<String, String>, k: &str, d: T) -> T {
    m.get(k).and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("scan");
    let m = kv(&args[2..]);
    match cmd {
        "validate" => validate(&m),
        _ => scan(&m),
    }
}

fn region(n: usize, lo: usize, hi: usize) -> Vec<bool> {
    (0..n).map(|q| q >= lo && q < hi).collect()
}

fn scan(m: &HashMap<String, String>) {
    let n: usize = get(m, "n", 32);
    let depth: usize = get(m, "depth", 4 * n);
    let pms: Vec<f64> = m
        .get("pm")
        .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or(vec![0.1]);
    let p_t: f64 = if let Some(c) = m.get("ptn") {
        c.parse::<f64>().unwrap() / n as f64
    } else {
        get(m, "pt", 0.0)
    };
    let samples: u64 = get(m, "samples", 10);
    let seed0: u64 = get(m, "seed0", 0);
    let mode = match m.get("mode").map(|s| s.as_str()) {
        Some("exact") => Mode::Exact,
        _ => Mode::DimensionOnly,
    };
    let maxd: usize = get(m, "maxd", 26);
    let ent = m.get("ent").cloned().unwrap_or("half".into());
    let magic: u32 = get(m, "magic", 0);
    let magicmax: usize = get(m, "magicmax", 10);
    let window: usize = get(m, "window", n).min(depth);
    let series: u32 = get(m, "series", 0);
    let tag = m.get("tag").cloned().unwrap_or_default();
    let group = Cliff2::group();
    if get::<u32>(m, "header", 1) == 1 {
        println!("tag,n,depth,p_m,p_t,seed,mode,d_final,d_avg,d_max,s_lo,s_hi,s_exact,i3_lo,i3_hi,i3_exact,nu,m2,t_gates,t_act,t_reg,m_frame,m_reg,m_det,elem_ops,secs,failed");
    }
    for &p_m in &pms {
        for s in 0..samples {
            let seed = seed0 + s;
            let p = Params {
                n,
                depth,
                p_m,
                p_t,
                periodic: true,
            };
            let t0 = Instant::now();
            let mut dsum = 0u64;
            let (sim, tr) = circuit::run(
                &p,
                mode,
                maxd,
                seed.wrapping_mul(0x1000_0001)
                    ^ (p_m * 1e6) as u64
                    ^ ((p_t * 1e9) as u64) << 20
                    ^ (n as u64) << 50,
                seed,
                &group,
                |t, sim| {
                    if t + window >= depth {
                        dsum += sim.d() as u64;
                    }
                },
            );
            let failed = tr.failed_at.is_some();
            let d_avg = if failed {
                f64::NAN
            } else {
                dsum as f64 / window as f64
            };
            let (mut s_lo, mut s_hi, mut s_ex) = (f64::NAN, f64::NAN, f64::NAN);
            let (mut i3_lo, mut i3_hi, mut i3_ex) = (f64::NAN, f64::NAN, f64::NAN);
            if !failed && ent != "none" {
                let ce = cut_entropy(&sim, &region(n, 0, n / 2), 34);
                s_lo = ce.lower;
                s_hi = ce.upper;
                s_ex = ce.s2.unwrap_or(f64::NAN);
                if ent == "i3" {
                    let q = n / 4;
                    // A,B,C = quarters 0,1,2 (D = 3), periodic
                    let regs: [(Vec<bool>, f64); 7] = [
                        (region(n, 0, q), 1.0),
                        (region(n, q, 2 * q), 1.0),
                        (region(n, 2 * q, 3 * q), 1.0),
                        (region(n, 0, 2 * q), -1.0),
                        (region(n, q, 3 * q), -1.0),
                        (
                            (0..n).map(|x| x < q || (x >= 2 * q && x < 3 * q)).collect(),
                            -1.0,
                        ),
                        (region(n, 0, 3 * q), 1.0),
                    ];
                    let (mut lo, mut hi, mut ex) = (0.0, 0.0, 0.0);
                    for (r, sg) in &regs {
                        let c = cut_entropy(&sim, r, 34);
                        if *sg > 0.0 {
                            lo += c.lower;
                            hi += c.upper;
                        } else {
                            lo -= c.upper;
                            hi -= c.lower;
                        }
                        ex += sg * c.s2.unwrap_or(f64::NAN);
                    }
                    i3_lo = lo;
                    i3_hi = hi;
                    i3_ex = ex;
                }
            }
            let (mut nu, mut m2) = (f64::NAN, f64::NAN);
            if magic == 1 && !failed && sim.d() <= magicmax {
                if let Some(a) = &sim.amp {
                    let sm = state_magic(a);
                    nu = sm.nullity;
                    m2 = sm.m2;
                }
            }
            let st = &sim.stats;
            println!(
                "{tag},{n},{depth},{p_m},{p_t},{seed},{},{},{d_avg},{},{s_lo},{s_hi},{s_ex},{i3_lo},{i3_hi},{i3_ex},{nu},{m2},{},{},{},{},{},{},{},{:.4},{}",
                if mode == Mode::Exact { "exact" } else { "dim" },
                sim.d(),
                tr.d.iter().max().copied().unwrap_or(0),
                st.t_gates,
                st.t_activating,
                st.t_register,
                st.meas_frame,
                st.meas_register,
                st.meas_determined,
                st.element_ops,
                t0.elapsed().as_secs_f64(),
                failed as u8
            );
            if series == 1 {
                let s: Vec<String> = tr.d.iter().map(|v| v.to_string()).collect();
                eprintln!("SERIES,{n},{p_m},{p_t},{seed},{}", s.join(" "));
            }
        }
    }
}

/// Larger-n cross-check against the dense state vector.
fn validate(m: &HashMap<String, String>) {
    let n: usize = get(m, "n", 16);
    let seeds: u64 = get(m, "seeds", 3);
    let depth: usize = get(m, "depth", 2 * n);
    let group = Cliff2::group();
    for seed in 0..seeds {
        let p_m = [0.1, 0.2, 0.3][seed as usize % 3];
        let p_t = [0.1, 0.05, 0.2][seed as usize % 3];
        let p = Params {
            n,
            depth,
            p_m,
            p_t,
            periodic: true,
        };
        let t0 = Instant::now();
        let mut crng = StdRng::seed_from_u64(1000 + seed);
        let mut brng = StdRng::seed_from_u64(2000 + seed);
        let mut sim = Monitored::new(n, Mode::Exact, 30).with_log();
        let mut sv = StateVectorF64::new(n);
        let mut cl = Vec::new();
        let mut maxdp = 0.0f64;
        let mut nm = 0;
        let mut maxd = 0;
        for t in 0..depth {
            for op in circuit::layer(&p, t, group.len(), &mut crng) {
                match op {
                    MOp::C2(k, a, b) => {
                        sim.cliff2(&group[k as usize], a, b);
                        for g in group[k as usize].gates(a, b) {
                            sv.apply_gate(&g).unwrap();
                            cl.push(g);
                        }
                    }
                    MOp::T(a) => {
                        sim.t(a).unwrap();
                        sv.apply_gate(&qsim_lab::gate::Gate::T(a)).unwrap();
                    }
                    MOp::M(a) => {
                        let r = sim.measure(a, &mut brng, None);
                        let ps = sv.collapse(a, r.outcome);
                        maxdp = maxdp.max((ps - r.prob).abs());
                        nm += 1;
                    }
                }
                maxd = maxd.max(sim.d());
            }
        }
        let f = sim.to_statevector(&cl).fidelity(&sv);
        let half = region(n, 0, n / 2);
        let ce = cut_entropy(&sim, &half, 40);
        println!(
            "validate n={n} seed={seed} p_m={p_m} p_t={p_t} depth={depth}: meas={nm} max|Δp|={maxdp:.2e} fidelity-1={:.2e} d_final={} d_max={maxd} S2_half={:?} [{}, {}] ({:.1}s)",
            f - 1.0,
            sim.d(),
            ce.s2,
            ce.lower,
            ce.upper,
            t0.elapsed().as_secs_f64()
        );
    }
}
