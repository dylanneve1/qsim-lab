//! Timing harness for the state-vector executors (min of `reps` runs).
//!
//! usage: sv_speed <ghz|qft|brick|grover> <n,...> [f32|f64] [base|blocked|both]
//!                 [reps] [key=value ...]
//! keys: block_kib, slots, fuse (0/1), small_n

use num_complex::Complex;
use qsim_lab::algorithms;
use qsim_lab::blocked::BlockConfig;
use qsim_lab::circuit::Circuit;
use qsim_lab::statevector::{Real, StateVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

fn workload(name: &str, n: usize) -> (Circuit, usize) {
    match name {
        "ghz" => (algorithms::ghz(n), 0),
        "qft" => (algorithms::qft(n), 0x5A5A_5A5A & ((1 << n) - 1)),
        "brick" => {
            let mut rng = StdRng::seed_from_u64(42);
            (algorithms::random_brickwork(n, 20, &mut rng), 0)
        }
        "grover" => {
            // one Grover iteration written as plain gates (MCZ via CCX-free
            // phase: built from H/X and a multi-controlled X is not a Gate,
            // so use the textbook layer structure with CZ on qubit pairs)
            panic!("grover is timed separately")
        }
        w if w.starts_with("rep:") => {
            // rep:<gate>:<q>[:<q2>] -> 200 copies of one gate (microbenchmark)
            let f: Vec<&str> = w.split(':').collect();
            let q: usize = f[2].parse().unwrap();
            let q2: usize = f.get(3).map(|x| x.parse().unwrap()).unwrap_or(0);
            let g = match f[1] {
                "h" => qsim_lab::Gate::H(q),
                "x" => qsim_lab::Gate::X(q),
                "t" => qsim_lab::Gate::T(q),
                "rx" => qsim_lab::Gate::Rx(q, 0.3),
                "cnot" => qsim_lab::Gate::Cnot(q, q2),
                "cz" => qsim_lab::Gate::Cz(q, q2),
                "cp" => qsim_lab::Gate::CPhase(q, q2, 0.3),
                "swap" => qsim_lab::Gate::Swap(q, q2),
                _ => panic!(),
            };
            let mut c = Circuit::new(n);
            for _ in 0..200 {
                c.gate(g);
            }
            (c, 0)
        }
        _ => panic!("unknown workload {name}"),
    }
}

/// Process CPU time (user + system, all threads) in seconds, from
/// /proc/self/stat (10 ms resolution). Less sensitive to time slicing by
/// other processes than wall-clock time on the shared machine.
fn cpu_time() -> f64 {
    let s = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let rest = s.rsplit_once(')').map(|x| x.1).unwrap_or("");
    let f: Vec<&str> = rest.split_whitespace().collect();
    let tick = |i: usize| f.get(i).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0);
    (tick(11) + tick(12)) / 100.0
}

fn run<T: Real>(
    c: &Circuit,
    init: usize,
    mode: &str,
    cfg: &BlockConfig,
) -> (f64, f64, StateVector<T>) {
    let mut s = StateVector::<T>::basis_state(c.num_qubits, init);
    let c0 = cpu_time();
    let t = Instant::now();
    match mode {
        "base" => s.apply_circuit(c).unwrap(),
        _ => s.apply_circuit_blocked(c, cfg).unwrap(),
    }
    (t.elapsed().as_secs_f64(), cpu_time() - c0, s)
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

fn bench<T: Real>(
    wl: &str,
    ns: &[usize],
    modes: &[&str],
    reps: usize,
    cfg: &BlockConfig,
    prec: &str,
) {
    println!("| workload | n | prec | gates | mode | min wall s | min cpu s | all wall |");
    println!("|---|---|---|---|---|---|---|---|");
    for &n in ns {
        let (c, init) = workload(wl, n);
        let mut last: Vec<StateVector<T>> = Vec::new();
        for &mode in modes {
            let mut times = Vec::new();
            let mut cpus = Vec::new();
            let mut st = None;
            for _ in 0..reps {
                let (dt, cpu, s) = run::<T>(&c, init, mode, cfg);
                times.push(dt);
                cpus.push(cpu);
                st = Some(s);
            }
            let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
            let cmin = cpus.iter().cloned().fold(f64::INFINITY, f64::min);
            println!(
                "| {wl} | {n} | {prec} | {} | {mode} | {min:.4} | {cmin:.2} | {} |",
                c.num_gates(),
                times
                    .iter()
                    .map(|t| format!("{t:.3}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            last.push(st.unwrap());
        }
        if last.len() == 2 {
            println!(
                "max |Δamp| {} vs {}: {:.3e}",
                modes[0],
                modes[1],
                maxdiff(last[0].amplitudes(), last[1].amplitudes())
            );
        }
    }
}

fn parse_cfg<'a>(kvs: impl Iterator<Item = &'a str>) -> BlockConfig {
    let mut cfg = BlockConfig::default();
    for kv in kvs {
        let (k, v) = kv.split_once('=').unwrap();
        match k {
            "block_kib" => cfg.block_bytes = v.parse::<usize>().unwrap() << 10,
            "slots" => cfg.slots = v.parse().unwrap(),
            "fuse" => cfg.fuse_1q = v == "1",
            "small_n" => cfg.small_n = v.parse().unwrap(),
            "split" => cfg.split_phases = v == "1",
            "sched" => cfg.schedule_diag = v == "1",
            "tile" => cfg.tile_u1 = v == "1",
            "simd" => cfg.simd = v == "1",
            "fusion" => cfg.max_fusion = v.parse().unwrap(),
            _ => panic!("unknown key {k}"),
        }
    }
    cfg
}

/// Interleaved A/B: `sv_speed ab <wl> <n> <f32|f64> <rounds> <cfg>...` where
/// each cfg is `key=val,key=val` (or `base` for the gate-by-gate path).
/// Runs the configs round-robin and prints min / median wall per config and
/// the max |Δamp| against the first config.
fn ab<T: Real>(wl: &str, n: usize, rounds: usize, cfgs: &[String]) {
    let (c, init) = workload(wl, n);
    let parsed: Vec<Option<BlockConfig>> = cfgs
        .iter()
        .map(|s| if s == "base" { None } else { Some(parse_cfg(s.split(',').filter(|x| !x.is_empty()))) })
        .collect();
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); cfgs.len()];
    let mut first: Option<StateVector<T>> = None;
    let mut diffs = vec![0.0f64; cfgs.len()];
    for round in 0..rounds {
        for (i, p) in parsed.iter().enumerate() {
            let mut s = StateVector::<T>::basis_state(c.num_qubits, init);
            let t = Instant::now();
            match p {
                None => s.apply_circuit(&c).unwrap(),
                Some(cfg) => s.apply_circuit_blocked(&c, cfg).unwrap(),
            }
            times[i].push(t.elapsed().as_secs_f64());
            if round == 0 {
                match &first {
                    None => first = Some(s),
                    Some(f) => diffs[i] = maxdiff(f.amplitudes(), s.amplitudes()),
                }
            }
        }
    }
    println!("| workload | n | config | min wall s | median wall s | max abs diff vs first |");
    println!("|---|---|---|---|---|---|");
    for (i, name) in cfgs.iter().enumerate() {
        let mut v = times[i].clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "| {wl} | {n} | {name} | {:.4} | {:.4} | {:.2e} |",
            v[0],
            v[v.len() / 2],
            diffs[i]
        );
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a[1] == "ab" {
        let n: usize = a[3].parse().unwrap();
        let rounds: usize = a[5].parse().unwrap();
        match a[4].as_str() {
            "f32" => ab::<f32>(&a[2], n, rounds, &a[6..]),
            _ => ab::<f64>(&a[2], n, rounds, &a[6..]),
        }
        return;
    }
    let wl = a[1].as_str();
    let ns: Vec<usize> = a[2].split(',').map(|x| x.parse().unwrap()).collect();
    let prec = a.get(3).map(|s| s.as_str()).unwrap_or("f32");
    let mode = a.get(4).map(|s| s.as_str()).unwrap_or("both");
    let reps: usize = a.get(5).map(|s| s.parse().unwrap()).unwrap_or(5);
    let cfg = parse_cfg(a.iter().skip(6).map(|x| x.as_str()));
    let modes: Vec<&str> = match mode {
        "both" => vec!["base", "blocked"],
        m => vec![m],
    };
    match prec {
        "f32" => bench::<f32>(wl, &ns, &modes, reps, &cfg, prec),
        _ => bench::<f64>(wl, &ns, &modes, reps, &cfg, prec),
    }
}
