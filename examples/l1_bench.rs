//! Interleaved A/B timing of blocked-executor configurations on one circuit:
//! each repetition runs every configuration once (round robin), and the
//! minimum over repetitions is reported, so slow drifts of a shared machine
//! hit all configurations equally.
//!
//! usage: l1_bench <brick|qft> <n,...> <f32|f64> <reps> <depth> <cfg> [<cfg> ...]
//! cfg  = name[:key=val[,key=val...]]   keys: block_kib slots tile_kib tile_b
//!        simd fuse sched. The name `ref` runs gate-by-gate `apply_circuit`.
//! env: GHZ (assumed clock, default 3.228) for the cycles column;
//!      CSV=path appends raw rows `workload,n,prec,cfg,rep,seconds`.

use num_complex::Complex;
use qsim_lab::algorithms;
use qsim_lab::blocked::{fuse_1q, lower_gates, tile_stats, BlockConfig, KOp};
use qsim_lab::circuit::{Circuit, Op};
use qsim_lab::statevector::{Real, StateVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::io::Write;
use std::time::Instant;

fn parse_cfg(spec: &str) -> (String, Option<BlockConfig>) {
    let (name, kv) = spec.split_once(':').unwrap_or((spec, ""));
    if name == "ref" {
        return (name.to_string(), None);
    }
    let mut cfg = BlockConfig::default();
    for p in kv.split(',').filter(|s| !s.is_empty()) {
        let (k, v) = p.split_once('=').unwrap();
        match k {
            "block_kib" => cfg.block_bytes = v.parse::<usize>().unwrap() << 10,
            "slots" => cfg.slots = v.parse().unwrap(),
            "tile_kib" => cfg.l1_tile_bytes = v.parse::<usize>().unwrap() << 10,
            "tile_b" => cfg.l1_tile_bytes = v.parse().unwrap(),
            "simd" => cfg.simd = v == "1",
            "fuse" => cfg.fuse_1q = v == "1",
            "sched" => cfg.schedule_diag = v == "1",
            _ => panic!("unknown key {k}"),
        }
    }
    (name.to_string(), Some(cfg))
}

fn workload(name: &str, n: usize, depth: usize) -> Circuit {
    match name {
        "brick" => {
            let mut rng = StdRng::seed_from_u64(42);
            algorithms::random_brickwork(n, depth, &mut rng)
        }
        "qft" => algorithms::qft(n),
        _ => panic!("unknown workload"),
    }
}

fn kops(c: &Circuit) -> Vec<KOp> {
    let gates: Vec<_> = c
        .ops
        .iter()
        .map(|o| match o {
            Op::Gate(g) => *g,
            _ => panic!(),
        })
        .collect();
    lower_gates(&gates)
}

/// Amplitude-updates of the fused op list (a controlled op touches only the
/// amplitudes where its controls are set).
fn amp_ops(ops: &[KOp], n: usize) -> f64 {
    ops.iter()
        .map(|o| match o {
            KOp::U1 { ctrl, .. } => (1u64 << n >> ctrl.count_ones()) as f64,
            KOp::Phase { mask, .. } => (1u64 << n >> mask.count_ones()) as f64,
            KOp::Swap { .. } => (1u64 << n) as f64 / 2.0,
        })
        .sum()
}

fn maxdiff<T: Real>(a: &[Complex<T>], b: &[Complex<T>]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = *x - *y;
            (d.re.to_f64().powi(2) + d.im.to_f64().powi(2)).sqrt()
        })
        .fold(0.0, f64::max)
}

fn bench<T: Real>(wl: &str, n: usize, depth: usize, reps: usize, specs: &[String], prec: &str) {
    let c = workload(wl, n, depth);
    let ops = kops(&c);
    let cfgs: Vec<(String, Option<BlockConfig>)> = specs.iter().map(|s| parse_cfg(s)).collect();
    let ghz: f64 = std::env::var("GHZ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3.228);
    let threads = rayon::current_num_threads();
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); cfgs.len()];
    let mut last: Vec<Option<StateVector<T>>> = (0..cfgs.len()).map(|_| None).collect();
    let mut csv = std::env::var("CSV").ok().map(|p| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .unwrap()
    });
    for rep in 0..reps {
        for (i, (name, cfg)) in cfgs.iter().enumerate() {
            let mut s = StateVector::<T>::basis_state(n, 0);
            let t = Instant::now();
            match cfg {
                None => s.apply_circuit(&c).unwrap(),
                Some(cfg) => s.apply_circuit_blocked(&c, cfg).unwrap(),
            }
            let dt = t.elapsed().as_secs_f64();
            times[i].push(dt);
            if let Some(f) = csv.as_mut() {
                writeln!(f, "{wl},{n},{prec},{name},{rep},{dt:.6}").unwrap();
            }
            last[i] = Some(s);
        }
    }
    for (i, (name, cfg)) in cfgs.iter().enumerate() {
        let min = times[i].iter().cloned().fold(f64::INFINITY, f64::min);
        // work in the same units for every config: fused ops of the default config
        let fused = fuse_1q(&ops, n, false);
        let work = amp_ops(&fused, n);
        let cyc = min * ghz * 1e9 * threads as f64 / work;
        let d = maxdiff(
            last[0].as_ref().unwrap().amplitudes(),
            last[i].as_ref().unwrap().amplitudes(),
        );
        let extra = match cfg {
            Some(cfg) if cfg.l1_tile_bytes > 0 => {
                let st = tile_stats::<T>(&ops, n, cfg);
                format!(
                    "k={} tiled/full ops {}/{} runs {}",
                    st.tile_bits, st.tiled_ops, st.full_ops, st.runs
                )
            }
            _ => String::new(),
        };
        println!(
            "| {wl} | {n} | {prec} | {name} | {:.4} | {cyc:.3} | {d:.1e} | {} | {extra} |",
            min,
            times[i]
                .iter()
                .map(|t| format!("{t:.3}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let wl = a[1].as_str();
    let ns: Vec<usize> = a[2].split(',').map(|x| x.parse().unwrap()).collect();
    let prec = a[3].as_str();
    let reps: usize = a[4].parse().unwrap();
    let depth: usize = a[5].parse().unwrap();
    let specs: Vec<String> = a[6..].to_vec();
    println!(
        "threads {} (rayon); cycles column = wall x clock x threads / amplitude-updates of the fused op list",
        rayon::current_num_threads()
    );
    println!("| workload | n | prec | config | min wall s | core-cyc/amp-op | max dAmp vs first | all wall s | tiling |");
    println!("|---|---|---|---|---|---|---|---|---|");
    for &n in &ns {
        match prec {
            "f32" => bench::<f32>(wl, n, depth, reps, &specs, prec),
            _ => bench::<f64>(wl, n, depth, reps, &specs, prec),
        }
    }
}
