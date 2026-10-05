//! Times the blocked state-vector executor on a circuit read from a gate-list
//! file, so that qsim-lab and other simulators run *identical* circuits
//! (`research/data/avx512/gen_circuits.py` writes the files; the format is
//! documented there and in `research/performance/avx512.md`).
//!
//! usage: sv_file_bench <file> <f32|f64> <reps> [key=val ...]
//! keys:  block_kib slots fuse sched simd avx512 dense dmin tile_kib (BlockConfig),
//!        mode=blocked|ref (ref = gate-by-gate `apply_gate`),
//!        dump=<path> (final state as little-endian f64 re/im pairs, index bit k = qubit k).
//! Threads: `RAYON_NUM_THREADS`. Timed: `apply_gates_blocked` (lowering, fusion,
//! planning and execution) on a preallocated, already touched |0..0> state.
//! States above the crate's 1 GiB cap are allocated directly; the run refuses
//! to start unless MemAvailable stays >= 6 GiB after the allocation.

use num_complex::Complex;
use qsim_lab::engines::blocked::BlockConfig;
use qsim_lab::engines::statevector::{Real, StateVector};
use qsim_lab::Gate;
use rayon::prelude::*;
use std::io::Write;
use std::time::Instant;

fn parse(path: &str) -> (usize, Vec<Gate>) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut n = 0usize;
    let mut gates = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        let q = |i: usize| -> usize { f[i].parse().unwrap() };
        let x = |i: usize| -> f64 { f[i].parse().unwrap() };
        match f[0] {
            "n" => n = q(1),
            "h" => gates.push(Gate::H(q(1))),
            "x" => gates.push(Gate::X(q(1))),
            "rx" => gates.push(Gate::Rx(q(1), x(2))),
            "ry" => gates.push(Gate::Ry(q(1), x(2))),
            "rz" => gates.push(Gate::Rz(q(1), x(2))),
            "u3" => gates.push(Gate::U(q(1), x(2), x(3), x(4))),
            "cx" => gates.push(Gate::Cnot(q(1), q(2))),
            "cz" => gates.push(Gate::Cz(q(1), q(2))),
            "cp" => gates.push(Gate::CPhase(q(1), q(2), x(3))),
            "swap" => gates.push(Gate::Swap(q(1), q(2))),
            // exp(-i θ/2 Z⊗Z) = e^{-iθ/2} diag(1, e^{iθ}, e^{iθ}, 1): global phase dropped
            "rzz" => {
                let (a, b, t) = (q(1), q(2), x(3));
                gates.push(Gate::Phase(a, t));
                gates.push(Gate::Phase(b, t));
                gates.push(Gate::CPhase(a, b, -2.0 * t));
            }
            g => panic!("{path}:{}: unsupported gate {g} (use the decomposed file)", ln + 1),
        }
    }
    assert!(n > 0, "{path}: missing `n` line");
    (n, gates)
}

fn mem_available() -> u64 {
    let s = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    s.lines()
        .find(|l| l.starts_with("MemAvailable:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse::<u64>().ok())
        .map(|kb| kb << 10)
        .unwrap_or(u64::MAX)
}

fn zero_state<T: Real>(n: usize) -> StateVector<T> {
    let bytes = (1u64 << n) * std::mem::size_of::<Complex<T>>() as u64;
    let avail = mem_available();
    assert!(
        bytes <= 12 << 30 && avail >= bytes + (6 << 30),
        "refusing: state {bytes} B, MemAvailable {avail} B (rule: keep >= 6 GiB free, <= 12 GiB)"
    );
    let mut v: Vec<Complex<T>> = (0..1usize << n)
        .into_par_iter()
        .map(|_| Complex::new(T::zero(), T::zero()))
        .collect();
    v[0] = Complex::new(T::one(), T::zero());
    StateVector::from_amplitudes(v)
}

fn reset<T: Real>(s: &mut StateVector<T>) {
    let a = s.amplitudes_mut();
    a.par_iter_mut()
        .for_each(|z| *z = Complex::new(T::zero(), T::zero()));
    a[0] = Complex::new(T::one(), T::zero());
}

fn run<T: Real>(path: &str, prec: &str, reps: usize, cfg: &BlockConfig, mode: &str, dump: Option<&str>) {
    let (n, gates) = parse(path);
    let mut s = zero_state::<T>(n);
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        reset(&mut s);
        let t = Instant::now();
        if mode == "ref" {
            for g in &gates {
                s.apply_gate(g).unwrap();
            }
        } else {
            s.apply_gates_blocked(&gates, cfg).unwrap();
        }
        times.push(t.elapsed().as_secs_f64());
    }
    let mut sorted = times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let name = std::path::Path::new(path)
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_default();
    println!(
        "| {name} | {n} | {prec} | {} | {mode} | {:.4} | {:.4} | {} | norm {:.6} |",
        rayon::current_num_threads(),
        sorted[0],
        sorted[sorted.len() / 2],
        times
            .iter()
            .map(|t| format!("{t:.4}"))
            .collect::<Vec<_>>()
            .join(" "),
        s.norm_sqr()
    );
    if let Some(p) = dump {
        let mut f = std::io::BufWriter::new(std::fs::File::create(p).unwrap());
        for z in s.amplitudes() {
            f.write_all(&z.re.to_f64().to_le_bytes()).unwrap();
            f.write_all(&z.im.to_f64().to_le_bytes()).unwrap();
        }
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: sv_file_bench <file> <f32|f64> <reps> [key=val ...]");
        std::process::exit(2);
    }
    let (path, prec) = (a[1].as_str(), a[2].as_str());
    let reps: usize = a[3].parse().unwrap();
    let mut cfg = BlockConfig::default();
    let mut mode = "blocked".to_string();
    let mut dump = None;
    for kv in &a[4..] {
        let (k, v) = kv.split_once('=').expect("key=val");
        match k {
            "block_kib" => cfg.block_bytes = v.parse::<usize>().unwrap() << 10,
            "slots" => cfg.slots = v.parse().unwrap(),
            "fuse" => cfg.fuse_1q = v == "1",
            "sched" => cfg.schedule_diag = v == "1",
            "simd" => cfg.simd = v == "1",
            "dense" => cfg.dense_fusion = v.parse().unwrap(),
            "dmin" => cfg.dense_min_ops = v.parse().unwrap(),
            "tile_kib" => cfg.l1_tile_bytes = v.parse::<usize>().unwrap() << 10,
            "mode" => mode = v.to_string(),
            "dump" => dump = Some(v.to_string()),
            _ => panic!("unknown key {k}"),
        }
    }
    match prec {
        "f32" => run::<f32>(path, prec, reps, &cfg, &mode, dump.as_deref()),
        "f64" => run::<f64>(path, prec, reps, &cfg, &mode, dump.as_deref()),
        _ => panic!("precision must be f32 or f64"),
    }
}
