//! Noisy gate-level Shor at scale: Monte-Carlo trajectories of the
//! windowed semiclassical circuit under stochastic Pauli noise
//! (`qsim_lab::shor::noisy`), written as one CSV row per trajectory.
//!
//! ```text
//! # fault-count stratified: k = 0..=kmax faults at uniformly random locations
//! cargo run --release --example shor_noise -- strat N a w kind kmax M seed [cap]
//! # direct: independent faults with probability p per location
//! cargo run --release --example shor_noise -- direct N a w kind p M seed [cap]
//! # instance info only
//! cargo run --release --example shor_noise -- info N a w
//! ```
//! Columns: `order_strict` = r is a continued-fraction convergent denominator
//! of y/2^t (textbook criterion); `order_ok` / `factor_ok` = the repo's
//! `shor::postprocess` (which also tries multiples k·q, k ≤ 256, of each
//! convergent) returns r / a nontrivial factor.
//! `kind` = depol | bitflip | phaseflip. `cap` bounds the support of a
//! trajectory (default 2^26 branches); a capped trajectory has unknown
//! outcome and is reported as such (`capped_round >= 0`).
//! Env: `QSIM_NOISE_CONC` = trajectories run concurrently (default: threads),
//! `QSIM_NOISE_F32` = f32 amplitudes.

use qsim_lab::shor::noisy::{self, NoiseKind, NoisyCircuit, Site};
use qsim_lab::shor::{Instance, Oracle};
use qsim_lab::shor_window::WindowLayout;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::io::Write;

fn seed_of(seed: u64, k: u64, j: u64) -> u64 {
    // splitmix64 of the triple
    let mut z = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(k.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(j.wrapping_mul(0x94D0_49BB_1331_11EB))
        .wrapping_add(0x2545_F491_4F6C_DD1D);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().map(String::as_str).unwrap_or("info");
    let n_mod: u64 = args[1].parse().unwrap();
    let a: u64 = args[2].parse().unwrap();
    let w: usize = args[3].parse().unwrap();
    let inst = Instance::new(n_mod, a, Oracle::Windowed(w));
    let r = noisy::order_of(a, n_mod);
    let lay = WindowLayout::new(inst.m, w);
    if mode == "info" {
        for kind in [NoiseKind::Depolarizing, NoiseKind::BitFlip, NoiseKind::PhaseFlip] {
            let nc = NoisyCircuit::new(&inst, kind);
            let gates: usize = nc.rounds.iter().map(|r| r.gates.len()).sum();
            println!(
                "N={n_mod} n={} a={a} r={r} w={w} qubits={} kind={} gates={gates} locations={}",
                inst.m,
                nc.nq,
                kind.name(),
                nc.num_locations()
            );
        }
        return;
    }
    let kind = NoiseKind::parse(&args[4]).expect("kind");
    let nc = NoisyCircuit::new(&inst, kind);
    let m: u64 = args[6].parse().unwrap();
    let seed: u64 = args[7].parse().unwrap();
    let cap: usize = args.get(8).map_or(1 << 26, |s| s.parse().unwrap());
    let gates: usize = nc.rounds.iter().map(|r| r.gates.len()).sum();
    let mut jobs: Vec<(u64, u64)> = Vec::new(); // (k or u64::MAX for direct, j)
    let (ks, p) = match mode {
        "strat" => {
            let kmax: u64 = args[5].parse().unwrap();
            ((0..=kmax).collect::<Vec<_>>(), None)
        }
        "direct" => (vec![u64::MAX], Some(args[5].parse::<f64>().unwrap())),
        _ => panic!("mode {mode}"),
    };
    for &k in &ks {
        for j in 0..m {
            jobs.push((k, j));
        }
    }
    let stdout = std::io::stdout();
    {
        let mut o = stdout.lock();
        writeln!(
            o,
            "# N={n_mod} n={} a={a} r={r} w={w} qubits={} kind={} gates={gates} locations={} t={} mode={mode} p={:?} cap={cap} seed={seed}",
            inst.m,
            nc.nq,
            kind.name(),
            nc.num_locations(),
            inst.t,
            p
        )
        .unwrap();
        writeln!(o, "n,N,a,r,kind,k,traj,measured,order_strict,order_ok,factor_ok,capped_round,peak,dirty_from,work_ops,secs,faults").unwrap();
    }
    let t_all = std::time::Instant::now();
    // at most `conc` trajectories at a time (memory: a capped trajectory
    // holds a few times `cap` branches); each trajectory is itself parallel
    let conc: usize = std::env::var("QSIM_NOISE_CONC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(rayon::current_num_threads());
    let f32_amps = std::env::var_os("QSIM_NOISE_F32").is_some();
    jobs.chunks(conc.max(1)).for_each(|batch| batch.par_iter().for_each(|&(k, j)| {
        let mut rng = StdRng::seed_from_u64(seed_of(seed, k, j));
        let faults = match p {
            Some(p) => nc.sample_p(p, &mut rng),
            None => nc.sample_k(k as usize, &mut rng),
        };
        let t0 = std::time::Instant::now();
        let tr = if f32_amps {
            noisy::run_trajectory::<f32, _>(&nc, &faults, cap, &mut rng)
        } else {
            noisy::run_trajectory::<f64, _>(&nc, &faults, cap, &mut rng)
        };
        let secs = t0.elapsed().as_secs_f64();
        let fdesc: Vec<String> = faults
            .iter()
            .map(|f| {
                let (gname, q, role, gfrac) = match (f.site, nc.gate_qubit(f)) {
                    (Site::Gate { gate, .. }, Some((g, q))) => {
                        let name = match g {
                            qsim_lab::Gate::X(_) => "x",
                            qsim_lab::Gate::Cnot(..) => "cx",
                            qsim_lab::Gate::Ccx(..) => "ccx",
                            qsim_lab::Gate::Swap(..) => "swap",
                            _ => "?",
                        };
                        let frac = gate as f64 / nc.rounds[f.round as usize].gates.len() as f64;
                        (name, q as i64, noisy::windowed_role(&lay, q), frac)
                    }
                    _ => ("-", 0, "ctrl", -1.0),
                };
                format!(
                    "{}/{}/{}/{}/{}/{}/{:.5}",
                    f.round,
                    f.site.kind_name(),
                    gname,
                    q,
                    role,
                    f.pauli.name(),
                    gfrac
                )
            })
            .collect();
        let kk = if p.is_some() { faults.len() as u64 } else { k };
        let order_ok = tr.order == Some(r);
        // textbook criterion: r is a continued-fraction convergent denominator of y / 2^t
        let strict = tr.measured.is_some_and(|y| {
            qsim_lab::shor::convergents(y, inst.t as u32).contains(&u128::from(r))
        });
        let line = format!(
            "{},{n_mod},{a},{r},{},{kk},{j},{},{},{},{},{},{},{},{},{secs:.4},{}",
            inst.m,
            kind.name(),
            tr.measured.map_or(-1i128, |y| y as i128),
            u8::from(strict),
            u8::from(order_ok),
            u8::from(tr.factor.is_some()),
            tr.capped.map_or(-1i64, |c| c.round as i64),
            tr.peak,
            tr.dirty_from.map_or(-1i64, |d| d as i64),
            tr.work_ops,
            fdesc.join("|")
        );
        let mut o = stdout.lock();
        writeln!(o, "{line}").unwrap();
    }));
    eprintln!("total {:.2}s for {} trajectories", t_all.elapsed().as_secs_f64(), jobs.len());
}
