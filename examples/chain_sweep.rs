//! Chain-sweep amplitudes for the IBM doped-Clifford circuit (and its
//! truncations): validation against the state vector, timing, and the
//! fidelity of dropping bond slices. See `research/chain-sweep/README.md`.
//!
//! ```text
//! cargo run --release --example chain_sweep -- info     --n 70 --d 70
//! cargo run --release --example chain_sweep -- validate --n 22 --d 40 --k 20
//! cargo run --release --example chain_sweep -- bench    --n 70 --d 40 --reps 3 --backend cpu32
//! cargo run --release --example chain_sweep -- fidsv    --n 20 --d 20 --s 8 --trials 4
//! cargo run --release --example chain_sweep -- fidmitm  --n 70 --d 40 --s 8 --k 400
//! cargo run --release --example chain_sweep -- lowprec  --n 70 --d 40 --k 200 --formats bf16,fp16:b1024
//! ```
//! `--backend metal` needs `--features metal` on macOS. `--qasm PATH`
//! overrides the circuit (default: the bundled nq70 depth-70 file).

use num_complex::Complex64;
use qsim_lab::circuit::Circuit;
use qsim_lab::engines::blocked::BlockConfig;
use qsim_lab::engines::chain_sweep::{self, compile, sliced_ops, truncate, ChainCircuit};
use qsim_lab::engines::statevector::StateVector;
use qsim_lab::gate::Gate;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::time::Instant;

const QASM: &str = include_str!("../research/chain-sweep/nq70_depth70_checks27_doped.qasm");

struct Args(Vec<String>);
impl Args {
    fn get<T: std::str::FromStr>(&self, k: &str, d: T) -> T {
        let key = format!("--{k}");
        self.0
            .iter()
            .position(|a| *a == key)
            .and_then(|i| self.0.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    }
    fn s(&self, k: &str, d: &str) -> String {
        self.get(k, d.to_string())
    }
}

fn circuit(a: &Args) -> Circuit {
    let src = match a.0.iter().position(|x| x == "--qasm") {
        Some(i) => std::fs::read_to_string(&a.0[i + 1]).expect("read qasm"),
        None => QASM.to_string(),
    };
    let c = Circuit::from_qasm(&src).expect("parse qasm");
    if a.0.iter().any(|x| x == "--tail") {
        // the last d CZ layers (of D_total = --dtot, default 70) instead of the first d
        let (dt, d) = (a.get("dtot", 70usize), a.get("d", 70usize));
        return chain_sweep::truncate_window(&c, a.get("n", 70), dt.saturating_sub(d), dt);
    }
    truncate(&c, a.get("n", 70), a.get("d", 70))
}

fn rand_x(rng: &mut StdRng, n: usize) -> u128 {
    let mut x = 0u128;
    for i in 0..n {
        if rng.random_bool(0.5) {
            x |= 1 << i;
        }
    }
    x
}

/// Runs a compiled plan on the chosen backend; returns (amplitude, seconds
/// of the run proper, seconds of backend compilation).
fn block_cfg() -> BlockConfig {
    let mut cfg = BlockConfig::default();
    let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok());
    if let Some(v) = env("CS_SIMD") {
        cfg.simd = v != 0;
    }
    if let Some(v) = env("CS_AVX512") {
        cfg.avx512 = v != 0;
    }
    if let Some(v) = env("CS_DENSE") {
        cfg.dense_fusion = v;
    }
    if let Some(v) = env("CS_FUSE") {
        cfg.fuse_1q = v != 0;
    }
    if let Some(v) = env("CS_SLOTS") {
        cfg.slots = v;
    }
    if let Some(v) = env("CS_BLOCK_BYTES") {
        cfg.block_bytes = v;
    }
    if let Some(v) = env("CS_DIAG") {
        cfg.schedule_diag = v != 0;
    }
    cfg
}

fn run(plan: &chain_sweep::SweepPlan, backend: &str) -> (Complex64, f64, f64) {
    let cfg = block_cfg();
    match backend {
        "cpu64" => {
            let t = Instant::now();
            let a = chain_sweep::amplitude_cpu::<f64>(plan, &cfg).unwrap();
            (a, t.elapsed().as_secs_f64(), 0.0)
        }
        "cpu32" => {
            let t = Instant::now();
            let a = chain_sweep::amplitude_cpu::<f32>(plan, &cfg).unwrap();
            (a, t.elapsed().as_secs_f64(), 0.0)
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        "metal" => metal::run(plan),
        b => panic!("unknown backend {b} (metal needs --features metal on macOS)"),
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal {
    use super::*;
    use qsim_lab::metal_sv::{MetalConfig, MetalSim};
    use std::sync::OnceLock;
    static SIM: OnceLock<MetalSim> = OnceLock::new();
    pub fn cfg() -> MetalConfig {
        let mut cfg = MetalConfig::default();
        let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok());
        if let Some(v) = env("CS_TG") {
            cfg.tg_bits = v;
        }
        if let Some(v) = env("CS_SLOTS") {
            cfg.slots = v;
        }
        if let Some(v) = env("CS_THREADS") {
            cfg.threads = v;
        }
        if let Some(v) = env("CS_REGS") {
            cfg.regs = v != 0;
        }
        if let Some(v) = env("CS_BATCH") {
            cfg.batch = v;
        }
        cfg
    }
    pub fn run(plan: &chain_sweep::SweepPlan) -> (Complex64, f64, f64) {
        let sim = SIM.get_or_init(|| MetalSim::new().expect("metal device"));
        let cfg = cfg();
        let w = plan.width.max(2);
        let mut s = sim.alloc(w).expect("alloc");
        let t0 = Instant::now();
        let p = sim.compile(w, &plan.ops, &cfg).expect("compile");
        let tc = t0.elapsed().as_secs_f64();
        if std::env::var_os("CS_STAGES").is_some() {
            eprintln!("metal stages={} cfg={cfg:?}", p.num_stages());
        }
        if std::env::var_os("CS_PROF").is_some() {
            let ls: f64 = sim.profile(&mut s, &p, true).unwrap().iter().sum();
            let full: f64 = sim.profile(&mut s, &p, false).unwrap().iter().sum();
            let gb = p.num_stages() as f64 * 16.0 * (1u64 << w) as f64 / 1e9;
            eprintln!(
                "metal profile: load/store only {ls:.3}s ({:.0} GB/s), full per-stage {full:.3}s",
                gb / ls
            );
            sim.set_basis(&s, 0);
        }
        let t1 = Instant::now();
        sim.run(&mut s, &p).expect("run");
        let tr = t1.elapsed().as_secs_f64();
        let a = s.amplitudes()[0];
        (
            Complex64::new(a.re as f64, a.im as f64) * plan.scale,
            tr,
            tc,
        )
    }
}

fn info(a: &Args) {
    let c = circuit(a);
    let cc = ChainCircuit::from_circuit(&c).unwrap();
    let plan = compile(&cc, 0, &HashMap::new());
    let bw = plan.backward.iter().filter(|&&b| b).count();
    println!(
        "n={} bonds={} width={} ops={} backward_qubits={} cut_widths={:?}",
        cc.n,
        cc.num_bonds(),
        plan.width,
        plan.ops.len(),
        bw,
        plan.cut_width
    );
}

fn validate(a: &Args) {
    let c = circuit(a);
    let n = c.num_qubits;
    let k: usize = a.get("k", 10);
    let backends = a.s("backends", "cpu64,cpu32");
    let mut sv = StateVector::<f64>::new(n);
    let t = Instant::now();
    sv.apply_circuit(&c).unwrap();
    let tsv = t.elapsed().as_secs_f64();
    let cc = ChainCircuit::from_circuit(&c).unwrap();
    let mut rng = StdRng::seed_from_u64(a.get("seed", 1));
    let rms = (1.0 / (1u64 << n) as f64).sqrt();
    // half the bitstrings uniform, half drawn from |amp|^2 (heavy outputs)
    let samples = sv.sample(k / 2, &mut rng);
    let mut xs: Vec<u128> = (0..k - k / 2).map(|_| rand_x(&mut rng, n)).collect();
    xs.extend(samples.iter().map(|&s| s as u128));
    for b in backends.split(',') {
        let (mut max_abs, mut max_rel, mut tt) = (0.0f64, 0.0f64, 0.0);
        let mut width = 0;
        for &x in &xs {
            let plan = compile(&cc, x, &HashMap::new());
            width = plan.width;
            let (amp, tr, _) = run(&plan, b);
            tt += tr;
            let e = sv.amplitude(x as usize);
            max_abs = max_abs.max((amp - e).norm() / rms);
            // exact zeros (Clifford-like truncations) only enter max_abs
            if e.norm() > 1e-3 * rms {
                max_rel = max_rel.max((amp - e).norm() / e.norm());
            }
        }
        println!(
            "validate n={n} d={} width={width} backend={b} k={} max|err|/rms={max_abs:.3e} max_rel={max_rel:.3e} t_amp={:.3}s t_sv={tsv:.2}s",
            a.get::<usize>("d", 70),
            xs.len(),
            tt / xs.len() as f64
        );
    }
}

fn bench(a: &Args) {
    let c = circuit(a);
    let n = c.num_qubits;
    let cc = ChainCircuit::from_circuit(&c).unwrap();
    let reps: usize = a.get("reps", 2);
    let backend = a.s("backend", "cpu32");
    let mut rng = StdRng::seed_from_u64(a.get("seed", 1));
    for r in 0..reps {
        let x = rand_x(&mut rng, n);
        let t = Instant::now();
        let plan = compile(&cc, x, &HashMap::new());
        let tcomp = t.elapsed().as_secs_f64();
        let (amp, trun, tbc) = run(&plan, &backend);
        println!(
            "bench n={n} d={} width={} ops={} backend={backend} rep={r} compile={tcomp:.3}s backend_compile={tbc:.3}s run={trun:.3}s amp={:.6e}{:+.6e}i |amp|^2*2^n={:.4}",
            a.get::<usize>("d", 70),
            plan.width,
            plan.ops.len(),
            amp.re,
            amp.im,
            amp.norm_sqr() * 2f64.powi(n as i32)
        );
    }
}

/// Exact fidelity of keeping a random subset of the slices over `s` bonds:
/// every slice state is computed once by the state vector (stored in f32),
/// then `F(K) = |Σ_{k∈K} <ψ|ψ_k>|^2 / ||Σ_{k∈K} ψ_k||^2` from the overlaps
/// and the Gram matrix, for many random subsets per kept fraction.
fn fidsv(a: &Args) {
    use num_complex::Complex32;
    let c = circuit(a);
    let n = c.num_qubits;
    let cc = ChainCircuit::from_circuit(&c).unwrap();
    let s: usize = a.get("s", 6);
    let subsets: usize = a.get("subsets", 200);
    let mode = a.s("mode", "edge");
    let e = n / 2 - 1;
    let bonds: Vec<usize> = match mode.as_str() {
        // one bond per edge around the middle, at the middle layer
        "spread" => (0..s)
            .map(|j| {
                let edge = (e + j).saturating_sub(s / 2).min(n - 2);
                let eb = cc.edge_bonds(edge);
                eb[eb.len() / 2]
            })
            .collect(),
        // s bonds of the middle edge, spread over its time range
        _ => {
            let eb = cc.edge_bonds(e);
            (0..s).map(|j| eb[j * eb.len() / s]).collect()
        }
    };
    let mut full = StateVector::<f64>::new(n);
    full.apply_circuit(&c).unwrap();
    let ns = 1usize << s;
    let t0 = Instant::now();
    let states: Vec<Vec<Complex32>> = (0..ns)
        .map(|k| {
            let mut st = StateVector::<f64>::new(n);
            for (q, m, cz) in sliced_ops(&c, &bonds, k) {
                match cz {
                    Some(r) => st.apply_gate(&Gate::Cz(q, r)).unwrap(),
                    None => st.apply_1q_matrix(q, &m),
                }
            }
            st.amplitudes()
                .iter()
                .map(|z| Complex32::new(z.re as f32, z.im as f32))
                .collect()
        })
        .collect();
    let tstates = t0.elapsed().as_secs_f64();
    let ov: Vec<Complex64> = states
        .iter()
        .map(|st| {
            full.amplitudes()
                .iter()
                .zip(st)
                .map(|(a, b)| a.conj() * Complex64::new(b.re as f64, b.im as f64))
                .sum()
        })
        .collect();
    let mut g = vec![Complex64::new(0.0, 0.0); ns * ns];
    for k in 0..ns {
        for l in k..ns {
            let v: Complex64 = states[k]
                .iter()
                .zip(&states[l])
                .map(|(a, b)| {
                    Complex64::new(a.re as f64, -a.im as f64)
                        * Complex64::new(b.re as f64, b.im as f64)
                })
                .sum();
            g[k * ns + l] = v;
            g[l * ns + k] = v.conj();
        }
    }
    let diag_sum: f64 = (0..ns).map(|k| g[k * ns + k].re).sum();
    let mut off = 0.0f64;
    for k in 0..ns {
        for l in 0..ns {
            if k != l {
                off +=
                    g[k * ns + l].norm() / (g[k * ns + k].re * g[l * ns + l].re).sqrt().max(1e-300);
            }
        }
    }
    let norms: Vec<f64> = (0..ns).map(|k| g[k * ns + k].re).collect();
    let (nmin, nmax) = norms
        .iter()
        .fold((f64::MAX, 0.0f64), |(lo, hi), &x| (lo.min(x), hi.max(x)));
    println!(
        "fidsv n={n} d={} s={s} mode={mode} bonds={bonds:?} slices={ns} t_states={tstates:.1}s sum_k||psi_k||^2={diag_sum:.6} slice_norm_min={nmin:.3e} max={nmax:.3e} mean|cos|_offdiag={:.4}",
        a.get::<usize>("d", 70),
        off / (ns * (ns - 1)).max(1) as f64
    );
    let mut rng = StdRng::seed_from_u64(a.get("seed", 1));
    for j in 0..=s {
        let m = 1usize << j;
        let mut fids = Vec::with_capacity(subsets);
        for _ in 0..subsets {
            let mut perm: Vec<usize> = (0..ns).collect();
            perm.shuffle(&mut rng);
            let kset = &perm[..m];
            let num: Complex64 = kset.iter().map(|&k| ov[k]).sum();
            let mut den = 0.0f64;
            for &k in kset {
                for &l in kset {
                    den += g[k * ns + l].re;
                }
            }
            fids.push(num.norm_sqr() / den.max(1e-300));
        }
        let mean = fids.iter().sum::<f64>() / fids.len() as f64;
        let sd = (fids.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / fids.len() as f64).sqrt();
        println!(
            "fidsv kept={m}/{ns} frac={:.5} fidelity_mean={mean:.5} sd={sd:.5}",
            m as f64 / ns as f64
        );
    }
}

/// Fidelity of dropping slices on one cut, estimated over `k` uniformly
/// random bitstrings of the full-width circuit via meet-in-the-middle.
fn fidmitm(a: &Args) {
    let c = circuit(a);
    let n = c.num_qubits;
    let cc = ChainCircuit::from_circuit(&c).unwrap();
    let s: usize = a.get("s", 6);
    let k: usize = a.get("k", 100);
    let subsets: usize = a.get("subsets", 8);
    let e: usize = a.get("e", n / 2 - 1);
    let cfg = BlockConfig::default();
    let mut rng = StdRng::seed_from_u64(a.get("seed", 1));
    let nb = cc.edge_bonds(e).len();
    let pos: Vec<usize> = (0..s).map(|j| j * nb / s).collect();
    let ns = 1usize << s;
    // random nested subsets: perms[t][..m]
    let perms: Vec<Vec<usize>> = (0..subsets)
        .map(|_| {
            let mut p: Vec<usize> = (0..ns).collect();
            p.shuffle(&mut rng);
            p
        })
        .collect();
    let marks: Vec<usize> = (0..=s).map(|j| 1 << j).collect();
    // accumulators per (subset, mark): Σ conj(a) a_K, Σ |a_K|^2
    let mut ov = vec![vec![Complex64::new(0.0, 0.0); marks.len()]; subsets];
    let mut nk = vec![vec![0.0f64; marks.len()]; subsets];
    let (mut nfull, mut xeb) = (0.0f64, 0.0f64);
    let t0 = Instant::now();
    for i in 0..k {
        let x = rand_x(&mut rng, n);
        let ct = chain_sweep::cut_tensors_cpu::<f64>(&cc, x, e, &cfg).unwrap();
        let sums = ct.slice_sums(&pos);
        let amp: Complex64 = sums.iter().sum();
        nfull += amp.norm_sqr();
        xeb += amp.norm_sqr() * 2f64.powi(n as i32);
        for (t, p) in perms.iter().enumerate() {
            let mut part = Complex64::new(0.0, 0.0);
            let mut mi = 0;
            for (cnt, &sl) in p.iter().enumerate() {
                part += sums[sl];
                if cnt + 1 == marks[mi] {
                    ov[t][mi] += amp.conj() * part;
                    nk[t][mi] += part.norm_sqr();
                    mi += 1;
                }
            }
        }
        if i == 0 {
            println!(
                "fidmitm n={n} d={} e={e} edge_bonds={nb} s={s} first_bitstring={:.2}s",
                a.get::<usize>("d", 70),
                t0.elapsed().as_secs_f64()
            );
        }
    }
    println!(
        "fidmitm k={k} mean 2^n|a|^2 = {:.4} (1 for uniform x) total {:.1}s",
        xeb / k as f64,
        t0.elapsed().as_secs_f64()
    );
    for (mi, &m) in marks.iter().enumerate() {
        let fids: Vec<f64> = (0..subsets)
            .map(|t| ov[t][mi].norm_sqr() / (nfull * nk[t][mi]).max(1e-300))
            .collect();
        let mean = fids.iter().sum::<f64>() / subsets as f64;
        let sd = (fids.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / subsets as f64).sqrt();
        let normr = (0..subsets).map(|t| nk[t][mi]).sum::<f64>() / subsets as f64 / nfull;
        println!(
            "fidmitm kept={m}/{ns} frac={:.5} fidelity_mean={mean:.5} sd_over_subsets={sd:.5} norm_kept/norm={normr:.5}",
            m as f64 / ns as f64
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let a = Args(args.clone());
    match args.first().map(String::as_str) {
        Some("info") => info(&a),
        Some("validate") => validate(&a),
        Some("bench") => bench(&a),
        Some("fidsv") => fidsv(&a),
        Some("fidmitm") => fidmitm(&a),
        Some("lowprec") => lowprec(&a),
        _ => eprintln!(
            "usage: chain_sweep info|validate|bench|fidsv|fidmitm|lowprec [--n N --d D ...]"
        ),
    }
}

/// Fidelity of storing the bond register in a reduced-precision format:
/// `k` uniform bitstrings, the exact f64 amplitude against the emulated
/// low-precision sweep for every format in `--formats`, rounding after each
/// pass (`--gran stage|op|qubit`). Prints the overlap fidelity
/// `|Σ e* l|^2 / (Σ|e|^2 Σ|l|^2)`, the linear-XEB ratio of sampling from
/// `|l|^2` (vs from `|e|^2`) and the rms relative error, with jackknife
/// errors. `--trace J` also prints, for the first bitstring, the register
/// fidelity against f64 every `J` passes.
fn lowprec(a: &Args) {
    use qsim_lab::engines::chain_lowprec::{passes, run_lowprec, Granularity, LowPrec};
    let c = circuit(a);
    let n = c.num_qubits;
    let d: usize = a.get("d", 70);
    let cc = ChainCircuit::from_circuit(&c).unwrap();
    let k: usize = a.get("k", 50);
    let trace: usize = a.get("trace", 0);
    let report: usize = a.get("report", 25);
    let gs = a.s("gran", "stage");
    let gran = match gs.as_str() {
        "op" => Granularity::Op,
        "qubit" => Granularity::Qubit,
        "stage" => Granularity::Stage,
        g => Granularity::Every(
            g.strip_prefix("every:")
                .and_then(|m| m.parse().ok())
                .expect("--gran op|qubit|stage|every:M"),
        ),
    };
    let fmts: Vec<LowPrec> = a
        .s("formats", "bf16,bf16:b1024,fp16:g,fp16:b1024")
        .split(',')
        .map(|f| LowPrec::parse(f).unwrap_or_else(|| panic!("bad format {f}")))
        .collect();
    let cfg = block_cfg();
    if a.0.iter().any(|x| x == "--count") {
        let plan = compile(&cc, 0, &HashMap::new());
        for g in [Granularity::Op, Granularity::Stage, Granularity::Qubit] {
            println!(
                "passes n={n} d={d} width={} block_bytes={} gran={g:?} passes={}",
                plan.width,
                cfg.block_bytes,
                passes(&plan, &cfg, g).len()
            );
        }
        return;
    }
    let mut rng = StdRng::seed_from_u64(a.get("seed", 1));
    let mut ex: Vec<Complex64> = Vec::new();
    let mut lps: Vec<Vec<Complex64>> = vec![Vec::new(); fmts.len()];
    let mut tsec = vec![0.0f64; fmts.len()];
    let t0 = Instant::now();
    for i in 0..k {
        let x = rand_x(&mut rng, n);
        let plan = compile(&cc, x, &HashMap::new());
        let e = chain_sweep::amplitude_cpu::<f64>(&plan, &cfg).unwrap();
        let ps = passes(&plan, &cfg, gran);
        let npass = ps.len();
        ex.push(e);
        for (j, f) in fmts.iter().enumerate() {
            let t = Instant::now();
            let tr = if i == 0 { trace } else { 0 };
            let r = run_lowprec(&plan, &ps, f, tr).unwrap();
            tsec[j] += t.elapsed().as_secs_f64();
            lps[j].push(r.amp);
            if !r.trace.is_empty() {
                let pts: Vec<String> = r
                    .trace
                    .iter()
                    .map(|(p, fi)| format!("{p}:{:.6}", fi))
                    .collect();
                println!("trace d={d} fmt={f} passes={} {}", r.passes, pts.join(" "));
            }
        }
        if (i + 1) % report == 0 || i + 1 == k {
            println!(
                "lowprec n={n} d={d} width={} gran={gran:?} passes={npass} k={} elapsed={:.0}s",
                plan.width,
                i + 1,
                t0.elapsed().as_secs_f64()
            );
            for (j, f) in fmts.iter().enumerate() {
                let st = lp_stats(&ex, &lps[j], n);
                println!(
                    "  fmt={:<16} bits={:.3} F={:.6} ±{:.6}  1-F={:.3e}  xeb_ratio={:.4} ±{:.4}  rms_rel={:.3e}  t/amp={:.2}s",
                    f.to_string(),
                    f.bits_per_component(),
                    st[0],
                    st[1],
                    1.0 - st[0],
                    st[2],
                    st[3],
                    st[4],
                    tsec[j] / (i + 1) as f64
                );
            }
        }
    }
}

/// (F, jackknife se, xeb ratio, se, rms relative error).
fn lp_stats(e: &[Complex64], l: &[Complex64], n: usize) -> [f64; 5] {
    let k = e.len();
    let two_n = 2f64.powi(n as i32);
    let fid = |skip: usize| -> (f64, f64) {
        let (mut ov, mut ne, mut nl) = (Complex64::new(0.0, 0.0), 0.0, 0.0);
        let (mut pl, mut ple, mut pe2) = (0.0, 0.0, 0.0);
        for i in (0..k).filter(|&i| i != skip) {
            ov += e[i].conj() * l[i];
            let (pe, pli) = (e[i].norm_sqr(), l[i].norm_sqr());
            ne += pe;
            nl += pli;
            pl += pli;
            ple += pli * pe;
            pe2 += pe * pe;
        }
        let f = ov.norm_sqr() / (ne * nl);
        let xl = two_n * ple / pl - 1.0;
        let xi = two_n * pe2 / ne - 1.0;
        (f, xl / xi)
    };
    let (f, x) = fid(usize::MAX);
    let (mut sf, mut sx) = (0.0, 0.0);
    if k > 1 {
        let jk: Vec<(f64, f64)> = (0..k).map(fid).collect();
        let mf = jk.iter().map(|v| v.0).sum::<f64>() / k as f64;
        let mx = jk.iter().map(|v| v.1).sum::<f64>() / k as f64;
        let c = (k - 1) as f64 / k as f64;
        sf = (c * jk.iter().map(|v| (v.0 - mf).powi(2)).sum::<f64>()).sqrt();
        sx = (c * jk.iter().map(|v| (v.1 - mx).powi(2)).sum::<f64>()).sqrt();
    }
    let num: f64 = e.iter().zip(l).map(|(a, b)| (a - b).norm_sqr()).sum();
    let den: f64 = e.iter().map(|a| a.norm_sqr()).sum();
    [f, sf, x, sx, (num / den).sqrt()]
}
