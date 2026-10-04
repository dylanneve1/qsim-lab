//! Noisy gate-level Shor on every oracle of the repo (`shor::noisy_gen`):
//! Monte-Carlo trajectories, one CSV row per trajectory.
//!
//! ```text
//! cargo run --release --example noise_oracles -- info   ORACLE N a
//! cargo run --release --example noise_oracles -- strat  ORACLE N a kind kmax M seed [cap]
//! cargo run --release --example noise_oracles -- direct ORACLE N a kind p    M seed [cap]
//! ```
//! `ORACLE` = `windowed:W` | `opt:W` | `mbul:W` (measurement-based lookups) |
//! `mbu:W` (all MBU constructions). `kind` = depol | bitflip | phaseflip.
//! Columns: `ok` = peak criterion of research/shor-noise.md, `strict` = r is
//! a convergent denominator of y/2^t, `weight` = importance weight of the
//! uniformly drawn recorded X-basis outcomes (1 unless branches collided;
//! estimators are weighted means), `L` = locations of this trajectory's
//! resolved circuit (MBU streams depend on the recorded outcomes).
//! Env: `QSIM_NOISE_CONC` (trajectories at a time), `QSIM_NOISE_F32`,
//! `QSIM_NOISE_RESET` (ideal reset of every ancilla after each round),
//! `QSIM_NOISE_KMIN`, `QSIM_NOISE_DESIGN` = `round` | `window` (Z-basis resets of
//! should-be-clean ancillas at every round end / after every window, with
//! reset-flip locations; `noisy_gen::ResetMode`).

use qsim_lab::shor::noisy::{self, NoiseKind, Site};
use qsim_lab::shor::noisy_gen::{self, tag, GenCircuit, Key, NOp, ResetMode, K192};
use qsim_lab::shor::{Instance, Oracle};
use qsim_lab::shor_window::WindowLayout;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::io::Write;

fn seed_of(seed: u64, k: u64, j: u64) -> u64 {
    let mut z = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(k.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(j.wrapping_mul(0x94D0_49BB_1331_11EB))
        .wrapping_add(0x2545_F491_4F6C_DD1D);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub fn parse_oracle(s: &str) -> Oracle {
    let (name, w) = s.split_once(':').unwrap_or((s, "4"));
    let w: usize = w.parse().expect("window");
    match name {
        "windowed" => Oracle::Windowed(w),
        "opt" | "windowed-opt" => Oracle::WindowedOpt(w),
        "mbul" | "mbu-lookup" => Oracle::WindowedMbuLookup(w),
        "mbu" => Oracle::WindowedMbu(w),
        _ => panic!("oracle {s}"),
    }
}

fn gate_name(op: &NOp) -> &'static str {
    match op {
        NOp::G(qsim_lab::Gate::X(_)) => "x",
        NOp::G(qsim_lab::Gate::Cnot(..)) => "cx",
        NOp::G(qsim_lab::Gate::Ccx(..)) => "ccx",
        NOp::G(qsim_lab::Gate::Swap(..)) => "swap",
        NOp::G(qsim_lab::Gate::Z(_)) => "z",
        NOp::G(qsim_lab::Gate::Cz(..)) => "cz",
        NOp::MeasX(..) => "measx",
        NOp::ResetZ(..) => "resetz",
        _ => "?",
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().map(String::as_str).unwrap_or("info");
    let oracle = parse_oracle(&args[1]);
    let n_mod: u64 = args[2].parse().unwrap();
    let a: u64 = args[3].parse().unwrap();
    let inst = Instance::new(n_mod, a, oracle);
    let r = noisy::order_of(a, n_mod);
    let w = match oracle {
        Oracle::Windowed(w)
        | Oracle::WindowedOpt(w)
        | Oracle::WindowedMbu(w)
        | Oracle::WindowedMbuLookup(w) => w,
        _ => 4,
    };
    let lay = WindowLayout::new(inst.m, w);
    if mode == "info" {
        let mut rng = StdRng::seed_from_u64(1);
        for kind in [
            NoiseKind::Depolarizing,
            NoiseKind::BitFlip,
            NoiseKind::PhaseFlip,
        ] {
            let gc = GenCircuit::new(&inst, kind);
            let mut ls = Vec::new();
            let mut stats = Vec::new();
            for _ in 0..if gc.is_mbu() { 8 } else { 1 } {
                let res = gc.resolve_rng(&mut rng);
                let ops: usize = res.rounds.iter().map(|r| r.ops.len()).sum();
                let meas: usize = res.rounds.iter().map(|r| r.measurements()).sum();
                let ccx: usize = res
                    .rounds
                    .iter()
                    .flat_map(|r| &r.ops)
                    .filter(|o| matches!(o, NOp::G(qsim_lab::Gate::Ccx(..))))
                    .count();
                ls.push(res.num_locations() as f64);
                stats.push((ops, meas, ccx));
            }
            let mean = ls.iter().sum::<f64>() / ls.len() as f64;
            let sd = (ls.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / (ls.len().max(2) - 1) as f64)
                .sqrt();
            println!(
                "N={n_mod} n={} a={a} r={r} oracle={oracle:?} qubits={} kind={} ops={} meas={} ccx={} L_mean={mean:.0} L_sd={sd:.1}",
                inst.m,
                gc.nq,
                kind.name(),
                stats[0].0,
                stats[0].1,
                stats[0].2
            );
        }
        // locations per block tag (depolarizing, one resolved stream)
        let gc = GenCircuit::new(&inst, NoiseKind::Depolarizing);
        let res = gc.resolve_rng(&mut rng);
        let mut by: std::collections::BTreeMap<String, (u64, u64)> = Default::default();
        for rd in &res.rounds {
            for (op, &t) in rd.ops.iter().zip(&rd.tags) {
                let e = by.entry(tag::name(t)).or_default();
                e.0 += 1;
                e.1 += match op {
                    NOp::G(g) => g.arity() as u64,
                    NOp::MeasX(..) => 2,
                    NOp::ResetZ(g) => rd.groups[*g as usize].len() as u64,
                };
            }
        }
        for (k, (ops, locs)) in by {
            println!("tag {k} ops={ops} locations={locs}");
        }
        return;
    }
    let kind = NoiseKind::parse(&args[4]).expect("kind");
    let design = match std::env::var("QSIM_NOISE_DESIGN").as_deref() {
        Ok("round") => ResetMode::Round,
        Ok("window") => ResetMode::Window,
        _ => ResetMode::None,
    };
    let gc = GenCircuit::with_resets(&inst, kind, design);
    let m: u64 = args[6].parse().unwrap();
    let seed: u64 = args[7].parse().unwrap();
    let cap: usize = args.get(8).map_or(1 << 26, |s| s.parse().unwrap());
    let (ks, p) = match mode {
        "strat" => {
            let kmax: u64 = args[5].parse().unwrap();
            let kmin: u64 = std::env::var("QSIM_NOISE_KMIN")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            ((kmin..=kmax).collect::<Vec<_>>(), None)
        }
        "direct" => (vec![u64::MAX], Some(args[5].parse::<f64>().unwrap())),
        _ => panic!("mode {mode}"),
    };
    let mut jobs: Vec<(u64, u64)> = Vec::new();
    for &k in &ks {
        for j in 0..m {
            jobs.push((k, j));
        }
    }
    let reset = std::env::var_os("QSIM_NOISE_RESET").is_some();
    let f32_amps = std::env::var_os("QSIM_NOISE_F32").is_some();
    let wide = gc.nq - 1 > 128;
    let stdout = std::io::stdout();
    {
        let mut o = stdout.lock();
        writeln!(
            o,
            "# N={n_mod} n={} a={a} r={r} oracle={oracle:?} qubits={} kind={} t={} mode={mode} p={:?} cap={cap} seed={seed} reset_ancillas={reset} wide_keys={wide} design={design:?}",
            inst.m,
            gc.nq,
            kind.name(),
            inst.t,
            p
        )
        .unwrap();
        writeln!(o, "n,N,a,r,oracle,kind,k,traj,measured,ok,strict,weight,collisions,capped_round,peak,dirty_from,L,secs,support,faults").unwrap();
    }
    let conc: usize = std::env::var("QSIM_NOISE_CONC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(rayon::current_num_threads());
    let oname = format!("{oracle:?}");
    let t_all = std::time::Instant::now();
    jobs.chunks(conc.max(1)).for_each(|batch| {
        batch.par_iter().for_each(|&(k, j)| {
            let mut rng = StdRng::seed_from_u64(seed_of(seed, k, j));
            let t0 = std::time::Instant::now();
            let res = gc.resolve_rng(&mut rng);
            let faults = match p {
                Some(p) => res.sample_p(p, &mut rng),
                None => res.sample_k(k as usize, &mut rng),
            };
            fn go<K: Key>(
                gc: &GenCircuit,
                res: &noisy_gen::Resolved,
                f: &[noisy::Fault],
                cap: usize,
                reset: bool,
                f32: bool,
                rng: &mut StdRng,
            ) -> noisy_gen::GenTrajectory {
                if f32 {
                    noisy_gen::run_trajectory::<K, f32, _>(gc, res, f, cap, reset, rng)
                } else {
                    noisy_gen::run_trajectory::<K, f64, _>(gc, res, f, cap, reset, rng)
                }
            }
            let tr = if wide {
                go::<K192>(&gc, &res, &faults, cap, reset, f32_amps, &mut rng)
            } else {
                go::<u128>(&gc, &res, &faults, cap, reset, f32_amps, &mut rng)
            };
            let secs = t0.elapsed().as_secs_f64();
            let fdesc: Vec<String> = faults
                .iter()
                .map(|f| {
                    let (gname, q, role, tg, gfrac) = match (f.site, res.op_qubit(f)) {
                        (Site::Gate { gate, slot }, Some((op, q, t))) => {
                            let mut name = gate_name(&op).to_string();
                            if let NOp::MeasX(..) = op {
                                name = if slot == 0 { "measx-ro" } else { "measx-reset" }.into();
                            }
                            if let NOp::ResetZ(..) = op {
                                name = "resetz-flip".into();
                            }
                            let frac =
                                gate as f64 / res.rounds[f.round as usize].ops.len() as f64;
                            (name, q as i64, noisy_gen::role(&lay, q), tag::name(t), frac)
                        }
                        _ => ("-".into(), 0, "ctrl", "control".into(), -1.0),
                    };
                    format!(
                        "{}/{}/{}/{}/{}/{}/{}/{:.5}",
                        f.round,
                        f.site.kind_name(),
                        gname,
                        q,
                        role,
                        tg,
                        f.pauli.name(),
                        gfrac
                    )
                })
                .collect();
            let kk = if p.is_some() { faults.len() as u64 } else { k };
            let ok = tr.measured.is_some_and(|y| noisy_gen::peak_ok(y, inst.t, r));
            let strict = tr.measured.is_some_and(|y| {
                qsim_lab::shor::convergents(y, inst.t as u32).contains(&u128::from(r))
            });
            let line = format!(
                "{},{n_mod},{a},{r},{oname},{},{kk},{j},{},{},{},{:.6e},{},{},{},{},{},{secs:.4},{},{}",
                inst.m,
                kind.name(),
                tr.measured.map_or(-1i128, |y| y as i128),
                u8::from(ok),
                u8::from(strict),
                tr.weight,
                tr.collision_rounds,
                tr.capped.map_or(-1i64, |c| c.round as i64),
                tr.peak,
                tr.dirty_from.map_or(-1i64, |d| d as i64),
                res.num_locations(),
                tr.support_trace
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(";"),
                fdesc.join("|")
            );
            let mut o = stdout.lock();
            writeln!(o, "{line}").unwrap();
        })
    });
    eprintln!(
        "total {:.2}s for {} trajectories",
        t_all.elapsed().as_secs_f64(),
        jobs.len()
    );
}
