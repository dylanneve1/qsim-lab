//! Driver binary for the simulability study (research/simulability.md).
//!
//! ```text
//! simulability features SPEC SEED [nohsf] [OBS]     -> one JSON line of features
//! simulability run ENGINE SPEC SEED MEM_BYTES [OBS]  -> one JSON line (value, secs, ...)
//! simulability mpscost SPEC SEED                   -> replayed MPS cost per bound
//! simulability mpstrace SPEC SEED [PROBE_CAP]       -> real exact MPS run: stats, trace
//!                                                     check, capped-probe prediction
//! OBS: all (Z on every qubit, default) | mid2 | mid4
//! ```
//! SPEC is `family:key=value,...`, e.g. `ct:n=24,L=8,t=20,nn=1`.

use qsim_lab::mps::{Mps, MpsStats};
use qsim_lab::mps_cost::{replay, BondSource, Estimator, ReplayCost};
use qsim_lab::simulability::{build, features_for, observable_qubits, run_engine_obs, Spec};
use std::time::Instant;

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: simulability features SPEC SEED [nohsf] | run ENGINE SPEC SEED MEM");
        std::process::exit(2);
    }
    match args[1].as_str() {
        "features" => {
            let spec = Spec::parse(&args[2]).expect("spec");
            let seed: u64 = args[3].parse().expect("seed");
            let with_hsf = !args[4..].iter().any(|s| s == "nohsf");
            let obs_name = args[4..].iter().find(|s| *s != "nohsf").map(String::as_str);
            let c = build(&spec, seed).expect("build");
            let obs = observable_qubits(obs_name.unwrap_or("all"), c.num_qubits).expect("obs");
            let f = features_for(&c, with_hsf, &obs).expect("features");
            println!(
                "{{\"n\":{},\"gates\":{},\"g2\":{},\"g3\":{},\"depth2\":{},\"t_count\":{},\"rotations\":{},\"d\":{},\"dense_l\":{:.4},\"redundant\":{},\"obs_zero\":{},\"frame_l\":{:.4},\"chi_bits\":{},\"mps_l\":{:.4},\"hsf_k\":{},\"hsf_na\":{},\"hsf_nb\":{},\"hsf_l\":{:.4},\"sup\":{},\"chi_bits0\":{},\"mps_l0\":{:.4},\"hsf_keff\":{},\"hsf_l0\":{:.4},\"sparse_l\":{:.4},\"sv_l\":{:.4},\"feat_secs\":{:.6},\"feat_secs_frame\":{:.6},\"feat_secs_hsf\":{:.6}}}",
                f.n, f.gates, f.g2, f.g3, f.depth2, f.t_count, f.rotations, f.d, f.dense_l,
                f.redundant, f.obs_zero, f.frame_l, f.chi_bits, f.mps_l, f.hsf_k, f.hsf_na, f.hsf_nb,
                f.hsf_l, f.sup, f.chi_bits0, f.mps_l0, f.hsf_keff, f.hsf_l0, f.sparse_l, f.sv_l, f.secs, f.secs_frame, f.secs_hsf
            );
        }
        "run" => {
            let engine = &args[2];
            let spec = Spec::parse(&args[3]).expect("spec");
            let seed: u64 = args[4].parse().expect("seed");
            let mem: u128 = args
                .get(5)
                .map(|s| s.parse().expect("mem"))
                .unwrap_or(1 << 30);
            let t0 = Instant::now();
            let c = build(&spec, seed).expect("build");
            let obs = observable_qubits(
                args.get(6).map(String::as_str).unwrap_or("all"),
                c.num_qubits,
            )
            .expect("obs");
            let build_secs = t0.elapsed().as_secs_f64();
            match run_engine_obs(engine, &c, mem, &obs) {
                Ok(r) => println!(
                    "{{\"ok\":true,\"value\":{:.15e},\"secs\":{:.6},\"size\":{},\"build_secs\":{:.6},\"note\":\"{}\"}}",
                    r.value, r.secs, r.size, build_secs, esc(&r.note)
                ),
                Err(e) => println!(
                    "{{\"ok\":false,\"error\":\"{}\",\"build_secs\":{:.6}}}",
                    esc(&format!("{e:?}")),
                    build_secs
                ),
            }
        }
        "mpscost" => {
            let spec = Spec::parse(&args[2]).expect("spec");
            let seed: u64 = args[3].parse().expect("seed");
            let c = build(&spec, seed).expect("build");
            let mut parts = Vec::new();
            for e in Estimator::ALL {
                let r = replay(&c, BondSource::Bound(e)).expect("replay");
                parts.push(format!("\"{}\":{}", e.name(), cost_json(&r)));
            }
            println!("{{{}}}", parts.join(","));
        }
        "mpstrace" => {
            let spec = Spec::parse(&args[2]).expect("spec");
            let seed: u64 = args[3].parse().expect("seed");
            let caps: Vec<u32> = args
                .get(4)
                .map(|s| s.split(',').map(|x| x.parse().expect("cap")).collect())
                .unwrap_or_else(|| vec![16]);
            let c = build(&spec, seed).expect("build");
            let max_secs: f64 = std::env::var("MPS_MAX_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(30.0);
            // capped probe runs
            let mut probes = Vec::new();
            for &cap in &caps {
                let t0 = Instant::now();
                let mut p = Mps::new(c.num_qubits, cap as usize);
                p.enable_trace();
                let mut done = true;
                for g in c.gates() {
                    p.apply_gate(g).expect("gate");
                    if t0.elapsed().as_secs_f64() > max_secs {
                        done = false;
                        break;
                    }
                }
                let probe_secs = t0.elapsed().as_secs_f64();
                let r = replay(&c, BondSource::Probe(p.trace(), cap)).expect("replay");
                let rx = replay(&c, BondSource::ProbeExtrapolate(p.trace(), cap)).expect("replay");
                probes.push(format!(
                    "\"{}\":{{\"done\":{},\"secs\":{:.6},\"truncations\":{},\"pred\":{},\"predx\":{}}}",
                    cap,
                    done,
                    probe_secs,
                    p.truncation_count(),
                    cost_json(&r),
                    cost_json(&rx)
                ));
            }
            // exact run
            let t1 = Instant::now();
            let mut m = Mps::new(c.num_qubits, 1 << 20);
            if let Some(cut) = std::env::var("MPS_CUTOFF")
                .ok()
                .and_then(|s| s.parse().ok())
            {
                m.set_cutoff(cut);
            }
            m.enable_trace();
            let mut done = true;
            for g in c.gates() {
                m.apply_gate(g).expect("gate");
                if t1.elapsed().as_secs_f64() > max_secs {
                    done = false;
                    break;
                }
            }
            let secs = t1.elapsed().as_secs_f64();
            // bound-capped exact run
            let bt = qsim_lab::mps_cost::replay_traced(&c, BondSource::Bound(Estimator::Best))
                .expect("replay");
            let t2 = Instant::now();
            let mut mb = Mps::new(c.num_qubits, 1 << 20);
            mb.set_step_caps(bt.trace.clone());
            mb.enable_trace();
            let mut bdone = true;
            for g in c.gates() {
                mb.apply_gate(g).expect("gate");
                if t2.elapsed().as_secs_f64() > max_secs {
                    bdone = false;
                    break;
                }
            }
            let bsecs = t2.elapsed().as_secs_f64();
            let capped = format!(
                "{{\"done\":{},\"secs\":{:.6},\"trace_max\":{},\"discarded\":{:.3e},\"stats\":{},\"zall_diff\":{:.3e}}}",
                bdone,
                bsecs,
                mb.trace().iter().max().copied().unwrap_or(1),
                1.0 - mb.fidelity_estimate(),
                stats_json(&mb.stats()),
                if done && bdone {
                    let all: Vec<usize> = (0..c.num_qubits).collect();
                    (m.expectation_z_product(&all) - mb.expectation_z_product(&all)).abs()
                } else {
                    f64::NAN
                }
            );
            let mut check = String::from("null");
            let mut viol = String::from("null");
            if done {
                let r = replay(&c, BondSource::Trace(m.trace())).expect("replay");
                check = format!("{}", r.stats == m.stats());
                // the best bound must dominate the real trace step by step
                let b = qsim_lab::mps_cost::replay_traced(&c, BondSource::Bound(Estimator::Best))
                    .expect("replay");
                let bad: Vec<(usize, u32, u32)> = b
                    .trace
                    .iter()
                    .zip(m.trace())
                    .enumerate()
                    .filter(|(_, (x, y))| x < y)
                    .map(|(i, (x, y))| (i, *x, *y))
                    .collect();
                if !bad.is_empty() {
                    eprintln!(
                        "violations (step, bound, real): {:?}",
                        &bad[..bad.len().min(12)]
                    );
                }
                let v = bad.len();
                viol = format!("{}", v + usize::from(b.trace.len() != m.trace().len()));
            }
            println!(
                "{{\"done\":{},\"secs\":{:.6},\"max_bond_final\":{},\"trace_max\":{},\"trace_sum_log\":{:.3},\"stats\":{},\"replay_matches\":{},\"bound_violations\":{},\"capped\":{},\"probes\":{{{}}}}}",
                done,
                secs,
                m.max_bond_dim(),
                m.trace().iter().max().copied().unwrap_or(1),
                m.trace().iter().map(|&x| (x as f64).log2()).sum::<f64>(),
                stats_json(&m.stats()),
                check,
                viol,
                capped,
                probes.join(",")
            );
        }
        _ => {
            eprintln!("unknown command");
            std::process::exit(2);
        }
    }
}

fn stats_json(s: &MpsStats) -> String {
    format!(
        "{{\"svd_calls\":{},\"svd_work\":{:.6e},\"qr_calls\":{},\"qr_work\":{:.6e},\"mm_work\":{:.6e},\"oneq_work\":{:.6e}}}",
        s.svd_calls, s.svd_work, s.qr_calls, s.qr_work, s.mm_work, s.oneq_work
    )
}

fn cost_json(r: &ReplayCost) -> String {
    format!(
        "{{\"stats\":{},\"max_bond\":{},\"final_max_bond\":{},\"sum_log_bond\":{:.3},\"secs\":{:.6}}}",
        stats_json(&r.stats),
        r.max_bond,
        r.final_max_bond,
        r.sum_log_bond,
        r.secs
    )
}
