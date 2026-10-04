//! Campaign driver for research/transition-theory.md (d-only, polynomial).
//!
//! `d(t)` of the monitored Clifford+T engine is the entropy of the same
//! monitored Clifford circuit with every `T` replaced by full Z-dephasing
//! (research/magic-transition.md §1.1), so every mode here runs in
//! [`Mode::DimensionOnly`].
//!
//! ```text
//! transition_theory steady n=256 pm=0.15,0.16 eta=1 pattern=poisson samples=10
//!     pattern = poisson : T on each qubit w.p. eta/n per layer (= magic_transition's ptn=eta;
//!                         same circuit seeds, so it reproduces raw.csv)
//!               exact   : floor(eta) (+1 w.p. frac(eta)) T per layer at distinct random sites
//!               fixed   : k = ceil(eta) sites spaced n/k apart, each dephased w.p. eta/k per layer
//!                         (noise on fixed timelike lines: a line defect)
//!     init = zero (|0^n>) | mixed (maximally mixed, d = n)
//!     depth (default 4n), window (default n): d_avg over the last window layers,
//!     d_prev over the window before it (stationarity check).
//! transition_theory survival n=1024 pm=0.16 burn=512 tmax=512 samples=50 gap=32
//!     pure Clifford steady state; inject one dephasing at a random site,
//!     record the layer at which the entropy dies (d back to 0); after a
//!     death wait `gap` layers and inject again, until `tmax` is exceeded by
//!     a survivor (censored) or `depth` runs out.
//! transition_theory decay n=512 pm=0.16 depth=1024 samples=20
//!     maximally mixed initial state, no injection, d(t) at all t (mean/sem
//!     over samples printed per t).
//! ```

use qsim_lab::monitored::circuit::{self, MOp, Params};
use qsim_lab::monitored::{Cliff2, Mode, Monitored};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
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

fn list(m: &HashMap<String, String>, k: &str, d: f64) -> Vec<f64> {
    m.get(k)
        .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or(vec![d])
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("steady");
    let m = kv(&args[2..]);
    match cmd {
        "survival" => survival(&m),
        "decay" => decay(&m),
        _ => steady(&m),
    }
}

/// Same seed mixing as examples/magic_transition.rs.
fn circ_seed(seed: u64, p_m: f64, p_t: f64, n: usize) -> u64 {
    seed.wrapping_mul(0x1000_0001) ^ (p_m * 1e6) as u64 ^ ((p_t * 1e9) as u64) << 20 ^ (n as u64) << 50
}

/// Clifford brickwork layer `t` followed by `Z` measurements w.p. `p_m`
/// (no T): the ops of `circuit::layer` with `p_t = 0`, in the same order.
fn clifford_layer<R: Rng + ?Sized>(n: usize, t: usize, p_m: f64, ng: usize, rng: &mut R) -> (Vec<MOp>, Vec<MOp>) {
    let mut g = Vec::with_capacity(n / 2 + 1);
    let off = t % 2;
    let mut i = off;
    while i + 1 < n {
        g.push(MOp::C2(rng.random_range(0..ng) as u16, i, i + 1));
        i += 2;
    }
    if off == 1 && n % 2 == 0 && n > 2 {
        g.push(MOp::C2(rng.random_range(0..ng) as u16, n - 1, 0));
    }
    let mut ms = Vec::new();
    for q in 0..n {
        if rng.random::<f64>() < p_m {
            ms.push(MOp::M(q));
        }
    }
    (g, ms)
}

fn make_mixed(sim: &mut Monitored) {
    for q in 0..sim.n {
        sim.cliff1(q, 0);
        sim.t(q).unwrap();
    }
    assert_eq!(sim.d(), sim.n);
}

fn steady(m: &HashMap<String, String>) {
    let n: usize = get(m, "n", 64);
    let depth: usize = get(m, "depth", 4 * n);
    let window: usize = get(m, "window", n).min(depth / 2);
    let pms = list(m, "pm", 0.16);
    let etas = list(m, "eta", 1.0);
    let samples: u64 = get(m, "samples", 10);
    let seed0: u64 = get(m, "seed0", 0);
    let pattern = m.get("pattern").cloned().unwrap_or("poisson".into());
    let init = m.get("init").cloned().unwrap_or("zero".into());
    let tag = m.get("tag").cloned().unwrap_or_default();
    let group = Cliff2::group();
    let ng = group.len();
    if get::<u32>(m, "header", 1) == 1 {
        println!("tag,n,depth,window,p_m,eta,pattern,init,seed,d_avg,d_prev,d_final,t_gates,t_act,secs");
    }
    for &eta in &etas {
        for &p_m in &pms {
            for s in 0..samples {
                let seed = seed0 + s;
                let p_t = eta / n as f64;
                let t0 = Instant::now();
                let cs = circ_seed(seed, p_m, p_t, n) ^ if pattern == "poisson" { 0 } else { 0x5bd1_e995 };
                let mut crng = StdRng::seed_from_u64(cs);
                let mut brng = StdRng::seed_from_u64(seed ^ 0x9e37_79b9_7f4a_7c15);
                let mut sim = Monitored::new(n, Mode::DimensionOnly, 0);
                if init == "mixed" {
                    make_mixed(&mut sim);
                }
                let par = Params { n, depth, p_m, p_t, periodic: true };
                let k_fixed = eta.ceil().max(1.0) as usize;
                let fixed_sites: Vec<usize> = (0..k_fixed).map(|j| j * n / k_fixed).collect();
                let (mut dsum, mut psum) = (0u64, 0u64);
                for t in 0..depth {
                    let ops: Vec<MOp> = match pattern.as_str() {
                        "poisson" => circuit::layer(&par, t, ng, &mut crng),
                        _ => {
                            let (mut o, ms) = clifford_layer(n, t, p_m, ng, &mut crng);
                            if pattern == "fixed" {
                                for &q in &fixed_sites {
                                    if crng.random::<f64>() < eta / k_fixed as f64 {
                                        o.push(MOp::T(q));
                                    }
                                }
                            } else {
                                let mut k = eta.floor() as usize;
                                if crng.random::<f64>() < eta - eta.floor() {
                                    k += 1;
                                }
                                let mut chosen: Vec<usize> = Vec::with_capacity(k);
                                while chosen.len() < k.min(n) {
                                    let q = crng.random_range(0..n);
                                    if !chosen.contains(&q) {
                                        chosen.push(q);
                                    }
                                }
                                chosen.sort_unstable();
                                o.extend(chosen.into_iter().map(MOp::T));
                            }
                            o.extend(ms);
                            o
                        }
                    };
                    for op in ops {
                        circuit::apply(&mut sim, op, &group, &mut brng).unwrap();
                    }
                    if t + window >= depth {
                        dsum += sim.d() as u64;
                    } else if t + 2 * window >= depth {
                        psum += sim.d() as u64;
                    }
                }
                println!(
                    "{tag},{n},{depth},{window},{p_m},{eta},{pattern},{init},{seed},{},{},{},{},{},{:.3}",
                    dsum as f64 / window as f64,
                    psum as f64 / window as f64,
                    sim.d(),
                    sim.stats.t_gates,
                    sim.stats.t_activating,
                    t0.elapsed().as_secs_f64()
                );
            }
        }
    }
}

fn survival(m: &HashMap<String, String>) {
    let n: usize = get(m, "n", 256);
    let burn: usize = get(m, "burn", n);
    let tmax: usize = get(m, "tmax", n / 2);
    let gap: usize = get(m, "gap", 32);
    let depth: usize = get(m, "depth", burn + 8 * tmax);
    let pms = list(m, "pm", 0.16);
    let samples: u64 = get(m, "samples", 10);
    let seed0: u64 = get(m, "seed0", 0);
    let group = Cliff2::group();
    let ng = group.len();
    if get::<u32>(m, "header", 1) == 1 {
        println!("n,p_m,seed,k,t_inj,activated,tau,censored");
    }
    for &p_m in &pms {
        for s in 0..samples {
            let seed = seed0 + s;
            let mut crng = StdRng::seed_from_u64(circ_seed(seed, p_m, 0.0, n) ^ 0x7f4a_7c15);
            let mut brng = StdRng::seed_from_u64(seed ^ 0x9e37_79b9_7f4a_7c15);
            let mut sim = Monitored::new(n, Mode::DimensionOnly, 0);
            let mut next_inj = burn;
            let mut alive_since: Option<usize> = None;
            let mut k = 0;
            for t in 0..depth {
                let (g, ms) = clifford_layer(n, t, p_m, ng, &mut crng);
                for op in g {
                    circuit::apply(&mut sim, op, &group, &mut brng).unwrap();
                }
                if alive_since.is_none() && t == next_inj {
                    let q = crng.random_range(0..n);
                    sim.t(q).unwrap();
                    if sim.d() == 1 {
                        alive_since = Some(t);
                    } else {
                        println!("{n},{p_m},{seed},{k},{t},0,0,0");
                        k += 1;
                        next_inj = t + 1;
                    }
                }
                for op in ms {
                    circuit::apply(&mut sim, op, &group, &mut brng).unwrap();
                }
                if let Some(t0) = alive_since {
                    let age = t + 1 - t0; // layers survived, counting the injection layer
                    if sim.d() == 0 {
                        println!("{n},{p_m},{seed},{k},{t0},1,{age},0");
                        k += 1;
                        alive_since = None;
                        next_inj = t + gap;
                    } else if age >= tmax {
                        println!("{n},{p_m},{seed},{k},{t0},1,{age},1");
                        break;
                    }
                }
                if next_inj + tmax > depth && alive_since.is_none() {
                    break;
                }
            }
        }
    }
}

fn decay(m: &HashMap<String, String>) {
    let n: usize = get(m, "n", 256);
    let depth: usize = get(m, "depth", 2 * n);
    let pms = list(m, "pm", 0.16);
    let samples: u64 = get(m, "samples", 10);
    let seed0: u64 = get(m, "seed0", 0);
    let group = Cliff2::group();
    let ng = group.len();
    if get::<u32>(m, "header", 1) == 1 {
        println!("n,p_m,seed,t,d");
    }
    for &p_m in &pms {
        for s in 0..samples {
            let seed = seed0 + s;
            let mut crng = StdRng::seed_from_u64(circ_seed(seed, p_m, 0.0, n) ^ 0x3c6e_f372);
            let mut brng = StdRng::seed_from_u64(seed ^ 0x9e37_79b9_7f4a_7c15);
            let mut sim = Monitored::new(n, Mode::DimensionOnly, 0);
            make_mixed(&mut sim);
            let mut last = usize::MAX;
            for t in 0..depth {
                let (g, ms) = clifford_layer(n, t, p_m, ng, &mut crng);
                for op in g.into_iter().chain(ms) {
                    circuit::apply(&mut sim, op, &group, &mut brng).unwrap();
                }
                let d = sim.d();
                // record every change (run-length), plus the last layer
                if d != last || t + 1 == depth {
                    println!("{n},{p_m},{seed},{},{d}", t + 1);
                    last = d;
                }
                if d == 0 {
                    break;
                }
            }
        }
    }
}
