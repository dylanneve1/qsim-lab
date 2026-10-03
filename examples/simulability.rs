//! Driver binary for the simulability study (research/simulability.md).
//!
//! ```text
//! simulability features SPEC SEED [nohsf]      -> one JSON line of features
//! simulability run ENGINE SPEC SEED MEM_BYTES  -> one JSON line (value, secs, ...)
//! ```
//! SPEC is `family:key=value,...`, e.g. `ct:n=24,L=8,t=20,nn=1`.

use qsim_lab::simulability::{build, features, run_engine, Spec};
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
            let with_hsf = args.get(4).map(|s| s != "nohsf").unwrap_or(true);
            let c = build(&spec, seed).expect("build");
            let f = features(&c, with_hsf).expect("features");
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
            let build_secs = t0.elapsed().as_secs_f64();
            match run_engine(engine, &c, mem) {
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
        _ => {
            eprintln!("unknown command");
            std::process::exit(2);
        }
    }
}
