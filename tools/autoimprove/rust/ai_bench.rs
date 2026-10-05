//! autoimprove bench driver (copied into `examples/ai_bench.rs` of the tree
//! under test by `tools/autoimprove/autoimprove.py`; not part of the crate).
//!
//! usage: ai_bench <qft|brick|clifft> <n> <f32|f64> <reps> [key=value ...]
//! keys: block_kib, slots, fuse, sched, simd, tile_kib, small_n,
//!       dense_k, dense_min (only when the tree has dense fusion).
//!
//! Prints one JSON line: wall-clock seconds of `apply_circuit_blocked` per
//! rep, at most `reps` reps and no more once they total 0.4 s (the state is re-initialised, outside the timer, before each rep),
//! plus a fingerprint of the final state (a fixed pseudo-random linear
//! functional and the norm) so the harness can check that baseline and
//! candidate computed the same state at full size.

use num_complex::Complex;
use qsim_lab::algorithms;
use qsim_lab::engines::blocked::{lower_gates, tile_stats, BlockConfig};
use qsim_lab::circuit::Circuit;
use qsim_lab::engines::statevector::{Real, StateVector};
use qsim_lab::Gate;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

/// Random Clifford+T: `depth` layers of a random 1q gate from
/// {H, S, Sdg, T, Tdg, X} on every qubit, then CNOT or CZ on a random
/// perfect matching.
fn clifft(n: usize, depth: usize, seed: u64) -> Circuit {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut c = Circuit::new(n);
    for _ in 0..depth {
        for q in 0..n {
            let g = match rng.random_range(0..6) {
                0 => Gate::H(q),
                1 => Gate::S(q),
                2 => Gate::Sdg(q),
                3 => Gate::T(q),
                4 => Gate::Tdg(q),
                _ => Gate::X(q),
            };
            c.gate(g);
        }
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.random_range(0..=i);
            perm.swap(i, j);
        }
        for p in perm.chunks_exact(2) {
            if rng.random_bool(0.5) {
                c.gate(Gate::Cnot(p[0], p[1]));
            } else {
                c.gate(Gate::Cz(p[0], p[1]));
            }
        }
    }
    c
}

fn workload(name: &str, n: usize) -> (Circuit, usize) {
    match name {
        "qft" => (algorithms::qft(n), 0x5A5A_5A5A & ((1 << n) - 1)),
        "brick" => {
            let mut rng = StdRng::seed_from_u64(42);
            (algorithms::random_brickwork(n, 20, &mut rng), 0)
        }
        // starts in |+...+>-ish: an H layer makes every amplitude matter
        "clifft" => {
            let mut c = Circuit::new(n);
            for q in 0..n {
                c.gate(Gate::H(q));
            }
            c.append(&clifft(n, 16, 7));
            (c, 0)
        }
        _ => panic!("unknown workload {name}"),
    }
}

fn config(args: &[String]) -> BlockConfig {
    let mut cfg = BlockConfig::default();
    for a in args {
        let (k, v) = a.split_once('=').expect("key=value");
        let u = || v.parse::<usize>().expect("integer");
        match k {
            "block_kib" => cfg.block_bytes = u() << 10,
            "slots" => cfg.slots = u(),
            "fuse" => cfg.fuse_1q = u() != 0,
            "sched" => cfg.schedule_diag = u() != 0,
            "simd" => cfg.simd = u() != 0,
            "tile_kib" => cfg.l1_tile_bytes = u() << 10,
            "small_n" => cfg.small_n = u(),
            "dense_k" => cfg.dense_fusion = u(),    // @dense
            "dense_min" => cfg.dense_min_ops = u(), // @dense
            _ => panic!("unknown key {k}"),
        }
    }
    cfg
}

#[repr(C)]
struct Timeval {
    sec: i64,
    usec: i32,
}

#[repr(C)]
struct Rusage {
    utime: Timeval,
    stime: Timeval,
    rest: [i64; 14],
}

extern "C" {
    fn getrusage(who: i32, usage: *mut Rusage) -> i32;
}

/// Process CPU time (user + system, all threads) in seconds, microsecond
/// resolution. Single-threaded runs measured this way do not count the time
/// the thread spends descheduled, which on a loaded machine is most of the
/// wall-clock noise.
fn cpu_time() -> f64 {
    let mut r = Rusage {
        utime: Timeval { sec: 0, usec: 0 },
        stime: Timeval { sec: 0, usec: 0 },
        rest: [0; 14],
    };
    // SAFETY: getrusage(RUSAGE_SELF) fills a `struct rusage`, whose layout on
    // 64-bit macOS and Linux is two `timeval`s (16 bytes each) and 14 longs.
    unsafe { getrusage(0, &mut r) };
    let tv = |t: &Timeval| t.sec as f64 + t.usec as f64 * 1e-6;
    tv(&r.utime) + tv(&r.stime)
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// `sum_k w_k a_k` for fixed pseudo-random weights in [-1, 1]^2, and the norm.
fn fingerprint<T: Real>(amps: &[Complex<T>]) -> (f64, f64, f64) {
    let (mut fr, mut fi, mut nn) = (0.0, 0.0, 0.0);
    for (k, a) in amps.iter().enumerate() {
        let h = splitmix(k as u64);
        let wr = (h & 0xFFFF_FFFF) as f64 / 2147483648.0 - 1.0;
        let wi = (h >> 32) as f64 / 2147483648.0 - 1.0;
        let (ar, ai) = (a.re.to_f64(), a.im.to_f64());
        fr += wr * ar - wi * ai;
        fi += wr * ai + wi * ar;
        nn += ar * ar + ai * ai;
    }
    (fr, fi, nn)
}

type Fp = (f64, f64, f64);

fn bench<T: Real>(
    c: &Circuit,
    init: usize,
    reps: usize,
    cfg: &BlockConfig,
) -> (Vec<f64>, Vec<f64>, Fp) {
    // warm the thread pool and code paths on a small register
    let mut w = StateVector::<T>::new(12.min(c.num_qubits));
    w.apply_circuit_blocked(&algorithms::qft(w.num_qubits()), cfg)
        .unwrap();
    let mut s = StateVector::<T>::new(c.num_qubits);
    let mut times = Vec::new();
    let mut cpu = Vec::new();
    // up to `reps` timings, stopping once they add up to MIN_TOTAL seconds:
    // one timing for the big cases, several for the millisecond ones
    const MIN_TOTAL: f64 = 0.4;
    for _ in 0..reps {
        if times.iter().sum::<f64>() >= MIN_TOTAL {
            break;
        }
        // explicit writes: every page is faulted in before the timer starts
        let amps = s.amplitudes_mut();
        amps.fill(Complex::new(T::zero(), T::zero()));
        amps[init] = Complex::new(T::one(), T::zero());
        let (t, c0) = (Instant::now(), cpu_time());
        s.apply_circuit_blocked(c, cfg).unwrap();
        times.push(t.elapsed().as_secs_f64());
        cpu.push(cpu_time() - c0);
    }
    (times, cpu, fingerprint(s.amplitudes()))
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        eprintln!("usage: ai_bench <qft|brick|clifft> <n> <f32|f64> <reps> [key=value ...]");
        std::process::exit(2);
    }
    let (wl, n, prec, reps) = (&a[1], a[2].parse().unwrap(), &a[3], a[4].parse().unwrap());
    let cfg = config(&a[5..]);
    let (c, init) = workload(wl, n);
    let (times, cpu, (fr, fi, nn)) = match prec.as_str() {
        "f32" => bench::<f32>(&c, init, reps, &cfg),
        _ => bench::<f64>(&c, init, reps, &cfg),
    };
    // plan shape: stages (full-state sweeps) and block-level passes
    let st = match prec.as_str() {
        "f32" => tile_stats::<f32>(&lower_gates(c.gates()), n, &cfg),
        _ => tile_stats::<f64>(&lower_gates(c.gates()), n, &cfg),
    };
    let fmt = |v: &[f64]| {
        v.iter()
            .map(|t| format!("{t:.6}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    println!(
        "{{\"wl\":\"{wl}\",\"n\":{n},\"prec\":\"{prec}\",\"gates\":{},\"times\":[{}],\"cpu\":[{}],\"fp\":[{fr:.12e},{fi:.12e}],\"norm\":{nn:.12e},\"stages\":{},\"passes\":{}}}",
        c.gates().count(),
        fmt(&times),
        fmt(&cpu),
        st.stages,
        st.full_ops + st.tiled_ops
    );
}
