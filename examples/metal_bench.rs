//! GPU (Metal, f32) vs CPU (blocked executor, f32) timing on one shared
//! buffer, interleaved: every repetition runs each mode once, the order
//! rotating per repetition; the minimum is reported.
//!
//! usage:
//!   metal_bench <qft|brick> <n,...> <reps> <depth> <mode> [<mode> ...]
//!     mode = cpu | naive | gpu[:tg=12,slots=6,thr=512,fuse=1,sched=1]
//!   metal_bench bw <n> <passes>      in-place scale bandwidth, GPU and CPU
//!
//! CPU threads: RAYON_NUM_THREADS. Times: `setup` = MTLBuffer allocation +
//! |0> init on the GPU (once per n); `plan` = lowering + fusion + stage
//! planning; `run` = command buffer submit..completion. CPU `run` includes
//! its own planning, like `apply_circuit_blocked`.

use num_complex::Complex32;
use qsim_lab::algorithms;
use qsim_lab::blocked::{lower_gates, BlockConfig, BlockedChunkExecutor, KOp};
use qsim_lab::circuit::Circuit;
use qsim_lab::metal_sv::{circuit_gates, MetalConfig, MetalSim, MetalState};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::time::Instant;

#[derive(Clone, Debug)]
enum Mode {
    Cpu,
    Naive,
    Gpu(MetalConfig),
}

fn parse_mode(s: &str) -> (String, Mode) {
    let (name, kv) = s.split_once(':').unwrap_or((s, ""));
    let m = match name {
        "cpu" => Mode::Cpu,
        "naive" => Mode::Naive,
        "gpu" => {
            let mut c = MetalConfig::default();
            for p in kv.split(',').filter(|x| !x.is_empty()) {
                let (k, v) = p.split_once('=').unwrap();
                match k {
                    "tg" => c.tg_bits = v.parse().unwrap(),
                    "slots" => c.slots = v.parse().unwrap(),
                    "thr" => c.threads = v.parse().unwrap(),
                    "fuse" => c.fuse_1q = v == "1",
                    "sched" => c.schedule_diag = v == "1",
                    _ => panic!("unknown key {k}"),
                }
            }
            Mode::Gpu(c)
        }
        _ => panic!("unknown mode {name}"),
    };
    (s.to_string(), m)
}

fn workload(name: &str, n: usize, depth: usize) -> Circuit {
    match name {
        "brick" => {
            let mut rng = StdRng::seed_from_u64(42);
            algorithms::random_brickwork(n, depth, &mut rng)
        }
        "qft" => algorithms::qft(n),
        _ => panic!("unknown workload {name}"),
    }
}

/// Fingerprint for cross-mode checks without a second state in memory:
/// 65536 evenly spaced amplitudes plus the norm.
fn fingerprint(a: &[Complex32]) -> (Vec<Complex32>, f64) {
    let stride = (a.len() / 65536).max(1);
    let samp = a.iter().step_by(stride).copied().collect();
    let norm = a
        .par_iter()
        .map(|z| (z.re as f64).powi(2) + (z.im as f64).powi(2))
        .sum();
    (samp, norm)
}

fn bw(sim: &MetalSim, n: usize, passes: usize) {
    let mut st = sim.alloc(n).unwrap();
    let bytes = 2.0 * 8.0 * (1u64 << n) as f64 * passes as f64;
    sim.scale_passes(&st, 1); // warm (page-in)
    let mut g = f64::INFINITY;
    let mut c = f64::INFINITY;
    for _ in 0..3 {
        g = g.min(sim.scale_passes(&st, passes));
        let a = st.amplitudes_mut();
        let t = Instant::now();
        for _ in 0..passes {
            a.par_chunks_mut(1 << 14).for_each(|ch| {
                for z in ch {
                    *z *= 1.0f32;
                }
            });
        }
        c = c.min(t.elapsed().as_secs_f64());
    }
    println!(
        "bw n={n} passes={passes} bytes/pass={:.3e}: GPU {:.1} GB/s ({:.4} s)  CPU({} thr) {:.1} GB/s ({:.4} s)",
        bytes / passes as f64,
        bytes / g / 1e9,
        g,
        rayon::current_num_threads(),
        bytes / c / 1e9,
        c
    );
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let sim = MetalSim::new().unwrap();
    if a[1] == "bw" {
        bw(&sim, a[2].parse().unwrap(), a[3].parse().unwrap());
        return;
    }
    let wl = a[1].as_str();
    let ns: Vec<usize> = a[2].split(',').map(|x| x.parse().unwrap()).collect();
    let reps: usize = a[3].parse().unwrap();
    let depth: usize = a[4].parse().unwrap();
    let modes: Vec<(String, Mode)> = a[5..].iter().map(|s| parse_mode(s)).collect();
    println!(
        "device {} | cpu threads {} | workload {wl} depth {depth}",
        sim.device_name(),
        rayon::current_num_threads()
    );
    for &n in &ns {
        let c = workload(wl, n, depth);
        let gates = circuit_gates(&c).unwrap();
        let t = Instant::now();
        let mut st: MetalState = sim.alloc(n).unwrap();
        let setup = t.elapsed().as_secs_f64();
        // touch once more so the first timed run doesn't pay page-in
        sim.scale_passes(&st, 1);
        let mut times = vec![Vec::new(); modes.len()];
        let mut plan_t = vec![0.0f64; modes.len()];
        let mut stages = vec![0usize; modes.len()];
        let mut fps: Vec<Option<(Vec<Complex32>, f64)>> = vec![None; modes.len()];
        for rep in 0..reps {
            for k in 0..modes.len() {
                let i = (k + rep) % modes.len();
                sim.set_basis(&st, 0);
                let (_, mode) = &modes[i];
                let dt = match mode {
                    Mode::Cpu => {
                        let cfg = BlockConfig::default();
                        let t = Instant::now();
                        let ops: Vec<KOp> = lower_gates(&gates);
                        let ex = BlockedChunkExecutor::<f32>::from_kops(&ops, n, &cfg);
                        ex.apply_to_chunk(st.amplitudes_mut());
                        t.elapsed().as_secs_f64()
                    }
                    Mode::Naive => {
                        let ops = lower_gates(&gates);
                        let t = Instant::now();
                        sim.apply_kops_naive(&mut st, &ops).unwrap();
                        stages[i] = ops.len();
                        t.elapsed().as_secs_f64()
                    }
                    Mode::Gpu(cfg) => {
                        let t0 = Instant::now();
                        let plan = sim.compile(n, &lower_gates(&gates), cfg).unwrap();
                        let t1 = Instant::now();
                        sim.run(&mut st, &plan).unwrap();
                        plan_t[i] = (t1 - t0).as_secs_f64();
                        stages[i] = plan.num_stages();
                        t1.elapsed().as_secs_f64()
                    }
                };
                times[i].push(dt);
                if rep == 0 {
                    fps[i] = Some(fingerprint(st.amplitudes()));
                }
            }
        }
        let (ref0, _) = fps[0].clone().unwrap();
        println!("| wl | n | mode | min run s | plan s | passes | eff GB/s | max dSample vs first | norm | setup s | all run s |");
        for (i, (name, _)) in modes.iter().enumerate() {
            let min = times[i].iter().cloned().fold(f64::INFINITY, f64::min);
            let (fp, norm) = fps[i].as_ref().unwrap();
            let d = fp
                .iter()
                .zip(&ref0)
                .map(|(x, y)| (x - y).norm() as f64)
                .fold(0.0, f64::max);
            let passes = stages[i];
            let gbs = if passes > 0 {
                format!("{:.1}", 16.0 * (1u64 << n) as f64 * passes as f64 / min / 1e9)
            } else {
                "-".into()
            };
            println!(
                "| {wl} | {n} | {name} | {min:.4} | {:.4} | {} | {gbs} | {d:.1e} | {norm:.6} | {setup:.4} | {} |",
                plan_t[i],
                if passes > 0 { passes.to_string() } else { "-".into() },
                times[i]
                    .iter()
                    .map(|t| format!("{t:.4}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
        drop(st);
    }
}
