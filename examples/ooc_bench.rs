//! Out-of-core state vector benchmark: one configuration per invocation, one CSV
//! line on stdout (so every run can go through `bench.sh`).
//!
//! ```text
//! ooc_bench header
//! ooc_bench ram   <workload> <n> <f32|f64> [reps]
//! ooc_bench ooc   <workload> <n> <f32|f64> <swap|window> <chunk_bits> <group_bits> <overlap 0|1> [verify 0|1]
//! ```
//!
//! Workloads: `qft`, `brick` (4 layers), `brick16` (16 layers), `ghz`.
//! Scratch directory: `$OOC_SCRATCH` (default `/tmp/ooc-scratch`). With
//! `verify=1` and `n <= 26` the final state is compared chunk by chunk with the
//! in-RAM blocked executor (max |Δamp|, streamed so RAM stays at one state).

use qsim_lab::algorithms;
use qsim_lab::blocked::BlockConfig;
use qsim_lab::circuit::Circuit;
use qsim_lab::ooc::{OocConfig, OocScheduler, OocStateVector};
use qsim_lab::statevector::{Real, StateVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::path::PathBuf;
use std::time::Instant;

fn workload(name: &str, n: usize) -> Circuit {
    let mut rng = StdRng::seed_from_u64(42 + n as u64);
    match name {
        "ghz" => algorithms::ghz(n),
        "qft" => algorithms::qft(n),
        "brick" => algorithms::random_brickwork(n, 4, &mut rng),
        "brick16" => algorithms::random_brickwork(n, 16, &mut rng),
        _ => panic!("unknown workload: {name}"),
    }
}

fn scratch_path() -> PathBuf {
    PathBuf::from(std::env::var("OOC_SCRATCH").unwrap_or_else(|_| "/tmp/ooc-scratch".into()))
}

const HEADER: &str = "kind,workload,n,prec,sched,chunk_bits,group_bits,overlap,file_mb,passes,gate_passes,perm_passes,read_mb,written_mb,wall_ms,io_busy_ms,compute_ms,perm_ms,stall_ms,gb_per_s,max_err";

fn run_ram<T: Real>(wl: &str, n: usize, prec: &str, reps: usize) {
    let circ = workload(wl, n);
    let bcfg = BlockConfig::default();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let mut sv = StateVector::<T>::new(n);
        let t0 = Instant::now();
        sv.apply_circuit_blocked(&circ, &bcfg).unwrap();
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
    }
    let mb = ((1usize << n) * std::mem::size_of::<num_complex::Complex<T>>()) as f64 / 1048576.0;
    println!("ram,{wl},{n},{prec},blocked,,,,{mb:.0},,,,,,{best:.1},,,,,,");
}

#[allow(clippy::too_many_arguments)]
fn run_ooc<T: Real>(
    wl: &str,
    n: usize,
    prec: &str,
    sched: OocScheduler,
    c: usize,
    k: usize,
    overlap: bool,
    verify: bool,
) {
    let circ = workload(wl, n);
    let bcfg = BlockConfig::default();
    let cfg = OocConfig {
        chunk_bits: c,
        scratch_dir: Some(scratch_path()),
        block_config: bcfg.clone(),
        restore_order: true,
        scheduler: sched,
        group_bits: k,
        overlap_io: overlap,
    };
    let mut ooc = OocStateVector::<T>::temp(n, c, cfg).unwrap();
    let stats = ooc.simulate_circuit(&circ).unwrap();

    let mut max_err = f64::NAN;
    if verify {
        let mut sv = StateVector::<T>::new(n);
        sv.apply_circuit_blocked(&circ, &bcfg).unwrap();
        let amps = sv.amplitudes();
        let mut buf = vec![num_complex::Complex::<T>::default(); 1 << c];
        let mut err = 0.0f64;
        for ci in 0..ooc.num_chunks() {
            ooc.read_chunk(ci, &mut buf).unwrap();
            for (j, z) in buf.iter().enumerate() {
                let d = *z - amps[(ci << c) | j];
                err = err.max(d.norm().to_f64());
            }
        }
        max_err = err;
    } else {
        let norm = ooc.state_norm().unwrap();
        assert!((norm - 1.0).abs() < 1e-3, "state norm {norm} diverged from 1.0");
    }

    let moved = (stats.bytes_read + stats.bytes_written) as f64;
    let wall = stats.wall_time.as_secs_f64();
    println!(
        "ooc,{wl},{n},{prec},{},{c},{},{},{:.0},{},{},{},{:.0},{:.0},{:.1},{:.1},{:.1},{:.1},{:.1},{:.2},{:e}",
        if sched == OocScheduler::Swap { "swap" } else { "window" },
        if sched == OocScheduler::Swap { 0 } else { k },
        overlap as u8,
        ooc.total_bytes() as f64 / 1048576.0,
        stats.file_passes,
        stats.local_passes,
        stats.swap_passes,
        stats.bytes_read as f64 / 1048576.0,
        stats.bytes_written as f64 / 1048576.0,
        wall * 1e3,
        stats.io_time.as_secs_f64() * 1e3,
        stats.compute_time.as_secs_f64() * 1e3,
        stats.perm_time.as_secs_f64() * 1e3,
        stats.stall_time.as_secs_f64() * 1e3,
        moved / 1e9 / wall,
        max_err
    );
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    match a.first().map(String::as_str) {
        Some("header") => println!("{HEADER}"),
        Some("ram") => {
            let n: usize = a[2].parse().unwrap();
            let reps: usize = a.get(4).map_or(1, |x| x.parse().unwrap());
            if a[3] == "f32" {
                run_ram::<f32>(&a[1], n, "f32", reps)
            } else {
                run_ram::<f64>(&a[1], n, "f64", reps)
            }
        }
        Some("ooc") => {
            let n: usize = a[2].parse().unwrap();
            let sched = if a[4] == "swap" {
                OocScheduler::Swap
            } else {
                OocScheduler::Window
            };
            let c: usize = a[5].parse().unwrap();
            let k: usize = a[6].parse().unwrap();
            let ov = a[7] == "1";
            let verify = a.get(8).is_some_and(|x| x == "1");
            if a[3] == "f32" {
                run_ooc::<f32>(&a[1], n, "f32", sched, c, k, ov, verify)
            } else {
                run_ooc::<f64>(&a[1], n, "f64", sched, c, k, ov, verify)
            }
        }
        _ => eprintln!("usage: see the module docs of examples/ooc_bench.rs"),
    }
}
