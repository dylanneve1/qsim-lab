//! Comprehensive benchmark suite for out-of-core state vector simulation.
//!
//! Measures:
//! - Total file passes, local passes, and swap passes.
//! - Bytes read and written, effective I/O throughput.
//! - Wall-clock time, I/O time, CPU compute time.
//! - Head-to-head comparisons against in-RAM blocked executor at 16..=24 qubits.
//! - Scaling up to n=29 qubits (4 GiB state vector) breaking the in-RAM 1 GiB limit.

use qsim_lab::algorithms;
use qsim_lab::blocked::BlockConfig;
use qsim_lab::circuit::Circuit;
use qsim_lab::ooc::{OocConfig, OocStateVector};
use qsim_lab::statevector::{Real, StateVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

fn workload(name: &str, n: usize) -> Circuit {
    match name {
        "ghz" => algorithms::ghz(n),
        "qft" => algorithms::qft(n),
        "brick" => {
            let mut rng = StdRng::seed_from_u64(42 + n as u64);
            // 4 layers of brickwork gates across all adjacent pairs
            algorithms::random_brickwork(n, 4, &mut rng)
        }
        _ => panic!("unknown workload: {name}"),
    }
}

fn scratch_path() -> PathBuf {
    PathBuf::from("/mnt/HC_Volume_106989832/dylan/qsim-swarm/ooc/scratch")
}

struct RunResult {
    workload: String,
    n: usize,
    c: usize,
    prec: &'static str,
    file_bytes: u64,
    passes: usize,
    local_passes: usize,
    swap_passes: usize,
    bytes_moved: u64,
    wall_ms: f64,
    io_ms: f64,
    compute_ms: f64,
    throughput_gb_s: f64,
    in_ram_ms: Option<f64>,
    max_err: Option<f64>,
}

fn bench_ooc<T: Real>(
    workload_name: &str,
    n: usize,
    c: usize,
    prec_name: &'static str,
    compare_ram: bool,
) -> RunResult {
    let circ = workload(workload_name, n);
    let bcfg = BlockConfig::default();

    let mut in_ram_ms = None;
    let mut in_ram_amps = None;

    if compare_ram && n <= 22 {
        let t0 = Instant::now();
        let mut in_ram = StateVector::<T>::new(n);
        in_ram.apply_circuit_blocked(&circ, &bcfg).unwrap();
        in_ram_ms = Some(t0.elapsed().as_secs_f64() * 1e3);
        in_ram_amps = Some(in_ram.amplitudes().to_vec());
    }

    let ooc_cfg = OocConfig {
        chunk_bits: c,
        scratch_dir: Some(scratch_path()),
        block_config: bcfg,
        restore_order: true,
    };

    let mut ooc = OocStateVector::<T>::temp(n, c, ooc_cfg).unwrap();
    let stats = ooc.simulate_circuit(&circ).unwrap();

    let mut max_err = None;
    if let Some(ref ram_amps) = in_ram_amps {
        let ooc_amps = ooc.read_amplitudes().unwrap();
        let mut err = 0.0f64;
        for (x, y) in ooc_amps.iter().zip(ram_amps.iter()) {
            let d = (*x - *y).norm().to_f64();
            if d > err {
                err = d;
            }
        }
        max_err = Some(err);
    } else {
        // Verify norm without loading full vector into RAM
        let norm = ooc.state_norm().unwrap();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "State norm {norm} diverged from 1.0!"
        );
    }

    let bytes_moved = stats.bytes_read + stats.bytes_written;
    let wall_s = stats.wall_time.as_secs_f64();
    let throughput_gb_s = if wall_s > 0.0 {
        (bytes_moved as f64) / (1024.0 * 1024.0 * 1024.0) / wall_s
    } else {
        0.0
    };

    RunResult {
        workload: workload_name.to_string(),
        n,
        c,
        prec: prec_name,
        file_bytes: ooc.total_bytes(),
        passes: stats.file_passes,
        local_passes: stats.local_passes,
        swap_passes: stats.swap_passes,
        bytes_moved,
        wall_ms: wall_s * 1e3,
        io_ms: stats.io_time.as_secs_f64() * 1e3,
        compute_ms: stats.compute_time.as_secs_f64() * 1e3,
        throughput_gb_s,
        in_ram_ms,
        max_err,
    }
}

fn main() {
    println!("==========================================================================================");
    println!("qsim-lab: Out-of-Core State Vector Simulation Benchmark Suite");
    println!("==========================================================================================");

    let mut results: Vec<RunResult> = Vec::new();

    // 1. Head-to-Head & Accuracy Sweeps: n = 16..22 (f64, forced small chunks)
    println!("\n--- Part 1: Exactness & Head-to-Head vs in-RAM Blocked (f64, small chunks) ---");
    for &n in &[16, 18, 20, 22] {
        let c = n - 4; // forced 16 chunks
        for wl in &["qft", "brick"] {
            let res = bench_ooc::<f64>(wl, n, c, "f64", true);
            println!(
                "[{}] n={:2}, c={:2} | passes={:2} (loc={:2}, swap={:2}) | data={:6.1} MB | OOC={:7.1} ms, RAM={:7.1} ms | max_err={:e}",
                res.workload,
                res.n,
                res.c,
                res.passes,
                res.local_passes,
                res.swap_passes,
                (res.bytes_moved as f64) / (1024.0 * 1024.0),
                res.wall_ms,
                res.in_ram_ms.unwrap_or(0.0),
                res.max_err.unwrap_or(0.0)
            );
            results.push(res);
        }
    }

    // 2. Scaling beyond physical RAM & in-RAM memory cap: n = 24..29 (f32)
    // 26 qubits = 512 MB, 27 qubits = 1.0 GB (cap!), 28 qubits = 2.0 GB, 29 qubits = 4.0 GB!
    println!("\n--- Part 2: Scaling to High Qubit Counts (f32, up to n=29, NVMe out-of-core) ---");
    for &n in &[24, 26, 27, 28, 29] {
        let c = 22.min(n - 1); // 2^22 chunk = 16 MiB per chunk in f32
        for wl in &["qft", "brick"] {
            let res = bench_ooc::<f32>(wl, n, c, "f32", false);
            println!(
                "[{}] n={:2}, c={:2} | file={:6.1} MB | passes={:2} (loc={:2}, swap={:2}) | moved={:7.1} MB | wall={:8.1} ms (io={:7.1} ms, comp={:7.1} ms) | {:5.2} GB/s",
                res.workload,
                res.n,
                res.c,
                (res.file_bytes as f64) / (1024.0 * 1024.0),
                res.passes,
                res.local_passes,
                res.swap_passes,
                (res.bytes_moved as f64) / (1024.0 * 1024.0),
                res.wall_ms,
                res.io_ms,
                res.compute_ms,
                res.throughput_gb_s
            );
            results.push(res);
        }
    }

    // Write CSV data
    let csv_path = PathBuf::from("research/data/ooc/scaling.csv");
    let mut f = File::create(&csv_path).expect("failed to create CSV output file");
    writeln!(
        f,
        "workload,n,c,precision,file_bytes,passes,local_passes,swap_passes,bytes_moved,wall_ms,io_ms,compute_ms,throughput_gb_s,in_ram_ms,max_err"
    )
    .unwrap();

    for r in &results {
        writeln!(
            f,
            "{},{},{},{},{},{},{},{},{},{:.2},{:.2},{:.2},{:.3},{:.2},{:e}",
            r.workload,
            r.n,
            r.c,
            r.prec,
            r.file_bytes,
            r.passes,
            r.local_passes,
            r.swap_passes,
            r.bytes_moved,
            r.wall_ms,
            r.io_ms,
            r.compute_ms,
            r.throughput_gb_s,
            r.in_ram_ms.unwrap_or(0.0),
            r.max_err.unwrap_or(0.0)
        )
        .unwrap();
    }
    println!("\nBenchmark results written to {}", csv_path.display());
}
