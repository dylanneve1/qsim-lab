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
//! cargo run --release --example chain_sweep -- packed   --n 70 --d 44 --tail --k 100 --formats int4:b16:h,int5:b64
//! cargo run --release --example chain_sweep -- tailinfo  --n 70 --d 70 --m 8 --format int4:b64
//! cargo run --release --example chain_sweep -- tailcheck --n 20 --d 40 --m 8 --backends cpu64,cpu32,int6:b64
//! cargo run --release --example chain_sweep -- run --d 70 --format int6:b64 --jobs cal_jobs.jsonl --out cal.jsonl --heartbeat hb.txt
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
        Some("packed") => packed(&a),
        Some("packedgpu") => packedgpu(&a),
        Some("tailinfo") => tailinfo(&a),
        Some("tailcheck") => tailcheck(&a),
        Some("run") => runloop(&a),
        Some("tailbench") => tailbench(&a),
        _ => eprintln!(
            "usage: chain_sweep info|validate|bench|fidsv|fidmitm|lowprec|packed|packedgpu|tailinfo|tailcheck|run [--n N --d D ...]"
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

/// Packed low-precision storage (`chain_packed`): the register is held as
/// packed `b`-bit ints + block scales and streamed once per stage of the
/// big-buffer planner (`--l` buffer bits, `--slots` gathered bits; the ops
/// are 1q-fused first unless `CS_FUSE=0`). With `--k K` (default 20) it
/// prints the fidelity against the exact f64 amplitude (as `lowprec`); with
/// `--noexact` it skips the exact runs (timing only); `--emul` also runs
/// the emulated reference on the same stages and counts bit-exact matches;
/// `--exact32` times the exact f32 sweep too; `--count` only plans.
fn packed(a: &Args) {
    use qsim_lab::engines::blocked::fuse_1q;
    use qsim_lab::engines::chain_lowprec::LowPrec;
    use qsim_lab::engines::chain_packed::{
        packed_stages, run_emulated_stages, run_packed, PackedStore,
    };
    let c = circuit(a);
    let n = c.num_qubits;
    let d: usize = a.get("d", 70);
    let cc = ChainCircuit::from_circuit(&c).unwrap();
    let k: usize = a.get("k", 20);
    let l: usize = a.get("l", 22);
    let slots: usize = a.get("slots", 14);
    let flag = |f: &str| a.0.iter().any(|x| x == f);
    let fmts: Vec<LowPrec> = a
        .s("formats", "int4:b16:h,int5:b64")
        .split(',')
        .map(|f| LowPrec::parse(f).unwrap_or_else(|| panic!("bad format {f}")))
        .collect();
    let cfg = block_cfg();
    let fuse = std::env::var("CS_FUSE").map(|v| v != "0").unwrap_or(true);
    let stages_for = |plan: &chain_sweep::SweepPlan| {
        let ops = if fuse {
            fuse_1q(&plan.ops, plan.width, false)
        } else {
            plan.ops.clone()
        };
        packed_stages(&ops, plan.width, l, slots)
    };
    let threads = rayon::current_num_threads();
    if flag("--count") {
        let plan = compile(&cc, 0, &HashMap::new());
        let st = stages_for(&plan);
        let minrun = st
            .iter()
            .map(|s| {
                s.inner
                    .iter()
                    .enumerate()
                    .take_while(|(j, q)| j == *q)
                    .count()
            })
            .min()
            .unwrap_or(0);
        println!(
            "packed-count n={n} d={d} width={} ops={} l={l} slots={slots} passes={} min_run_bits={minrun}",
            plan.width,
            plan.ops.len(),
            st.len()
        );
        for f in &fmts {
            println!(
                "  fmt={f} store={:.3} GiB  + buffers {:.3} GiB ({threads} threads)",
                PackedStore::bytes_for(plan.width, f).unwrap_or(0) as f64 / (1u64 << 30) as f64,
                (threads << l) as f64 * 8.0 / (1u64 << 30) as f64
            );
        }
        return;
    }
    let noexact = flag("--noexact");
    let mut rng = StdRng::seed_from_u64(a.get("seed", 1));
    let mut ex: Vec<Complex64> = Vec::new();
    let mut lps: Vec<Vec<Complex64>> = vec![Vec::new(); fmts.len()];
    let mut tsec = vec![0.0f64; fmts.len()];
    let mut tparts = vec![[0.0f64; 3]; fmts.len()];
    let mut exact_ok = vec![0usize; fmts.len()];
    let t0 = Instant::now();
    for i in 0..k {
        let x = rand_x(&mut rng, n);
        let plan = compile(&cc, x, &HashMap::new());
        let st = stages_for(&plan);
        if flag("--exact32") {
            let t = Instant::now();
            let e32 = chain_sweep::amplitude_cpu::<f32>(&plan, &cfg).unwrap();
            println!(
                "exact32 d={d} width={} t={:.2}s amp={e32:.4e}",
                plan.width,
                t.elapsed().as_secs_f64()
            );
        }
        if !noexact {
            ex.push(chain_sweep::amplitude_cpu::<f64>(&plan, &cfg).unwrap());
        }
        for (j, f) in fmts.iter().enumerate() {
            let r = run_packed(&plan, &st, f, &cfg).unwrap_or_else(|e| panic!("{f}: {e}"));
            tsec[j] += r.secs;
            for (t, s) in tparts[j].iter_mut().zip(r.thread_secs) {
                *t += s;
            }
            if r.underflow + r.overflow > 0 {
                println!(
                    "  WARNING fmt={f} underflow={} overflow={}",
                    r.underflow, r.overflow
                );
            }
            if flag("--emul") {
                let e = run_emulated_stages(&plan, &st, f, &cfg).unwrap();
                exact_ok[j] += (e == r.amp) as usize;
            }
            lps[j].push(r.amp);
            if noexact {
                println!(
                    "packed-time n={n} d={d} width={} fmt={f} passes={} store={:.3} GiB t={:.2}s unpack/compute/pack thread-s={:.1}/{:.1}/{:.1} threads={threads} amp={:.4e}",
                    plan.width,
                    r.passes,
                    r.store_bytes as f64 / (1u64 << 30) as f64,
                    r.secs,
                    r.thread_secs[0],
                    r.thread_secs[1],
                    r.thread_secs[2],
                    r.amp
                );
            }
        }
        if !noexact && ((i + 1) % a.get("report", 25usize) == 0 || i + 1 == k) {
            println!(
                "packed n={n} d={d} width={} passes={} l={l} slots={slots} k={} elapsed={:.0}s",
                plan.width,
                st.len(),
                i + 1,
                t0.elapsed().as_secs_f64()
            );
            for (j, f) in fmts.iter().enumerate() {
                let s = lp_stats(&ex, &lps[j], n);
                let r = -s[0].ln() / st.len() as f64;
                println!(
                    "  fmt={:<12} bits={:.3} F={:.6} ±{:.6} r=-lnF/R={:.3e} xeb_ratio={:.4} ±{:.4} rms_rel={:.3e} t/amp={:.3}s{}",
                    f.to_string(),
                    f.bits_per_component(),
                    s[0],
                    s[1],
                    r,
                    s[2],
                    s[3],
                    s[4],
                    tsec[j] / (i + 1) as f64,
                    if flag("--emul") {
                        format!(" bit-exact={}/{}", exact_ok[j], i + 1)
                    } else {
                        String::new()
                    }
                );
            }
        }
    }
}

// ----- tail-open batches and the sampling run (RUNPLAN §2.1, §6) ----------

#[cfg(windows)]
mod keepawake {
    #[link(name = "kernel32")]
    extern "system" {
        fn SetThreadExecutionState(flags: u32) -> u32;
    }
    /// Keeps the machine from sleeping while this thread lives (ES_CONTINUOUS | ES_SYSTEM_REQUIRED).
    pub fn on() {
        // SAFETY: plain Win32 call with constant flags.
        unsafe {
            SetThreadExecutionState(0x8000_0000 | 0x0000_0001);
        }
    }
}
#[cfg(not(windows))]
mod keepawake {
    pub fn on() {}
}

/// The register backend named by `--format`: `cpu64`, `cpu32`, or a packed
/// `intB:bN[:h]` format (with `--l`, `--slots`).
fn tail_backend(a: &Args) -> Box<dyn qsim_lab::engines::chain_tail::SweepBackend> {
    use qsim_lab::engines::chain_lowprec::LowPrec;
    use qsim_lab::engines::chain_tail::{CpuExact, CpuPacked};
    let f = a.s("format", "int6:b64");
    let cfg = block_cfg();
    match f.as_str() {
        "cpu64" => Box::new(CpuExact::<f64>::new(cfg)),
        "cpu32" => Box::new(CpuExact::<f32>::new(cfg)),
        f => {
            let lp = LowPrec::parse(f).unwrap_or_else(|| panic!("bad format {f}"));
            let mut be = CpuPacked::new(lp, cfg, a.get("l", 22), a.get("slots", 14));
            be.fuse = std::env::var("CS_FUSE").map(|v| v != "0").unwrap_or(true);
            Box::new(be)
        }
    }
}

/// Plan facts for a tail-open run: register width, open bonds and their
/// register bits, passes R of the chosen backend, tail-pass cost.
fn tailinfo(a: &Args) {
    use qsim_lab::engines::chain_tail::TailPlan;
    let c = circuit(a);
    let cc = ChainCircuit::from_circuit(&c).unwrap();
    let m: usize = a.get("m", 6);
    let tp = TailPlan::new(&cc, 0, m);
    let be = tail_backend(a);
    let r = be.count_passes(&tp.sweep);
    println!(
        "tailinfo n={} d={} m={m} width={} ops={} open_bonds={} R={} backend=\"{}\" tail_cmacs={:.3e}",
        cc.n,
        a.get::<usize>("d", 70),
        tp.sweep.width,
        tp.sweep.ops.len(),
        tp.num_bonds(),
        r,
        be.describe(),
        tp.tail_cost()
    );
    println!("  open bond slots (time order): {:?}", tp.slots);
    if let Some(lp) = qsim_lab::engines::chain_lowprec::LowPrec::parse(&a.s("format", "int6:b64")) {
        if let Some(b) =
            qsim_lab::engines::chain_packed::PackedStore::bytes_for(tp.sweep.width, &lp)
        {
            println!("  store {:.3} GiB", b as f64 / (1u64 << 30) as f64);
        }
    }
}

/// Validation of the tail batch: `--k` random suffixes, every backend in
/// `--backends` (cpu64, cpu32, packed formats) against the dense state
/// vector (n <= 26) or, at larger n, the exact chain sweep of every
/// `--every`-th completion. Prints max relative error and batch fidelity.
fn tailcheck(a: &Args) {
    use qsim_lab::engines::chain_tail::{run_tail, CpuExact, CpuPacked, SweepBackend, TailPlan};
    let c = circuit(a);
    let n = c.num_qubits;
    let cc = ChainCircuit::from_circuit(&c).unwrap();
    let m: usize = a.get("m", 6);
    let k: usize = a.get("k", 3);
    let every: usize = a.get("every", 1);
    let mut rng = StdRng::seed_from_u64(a.get("seed", 1));
    let sv = (n <= 26).then(|| {
        let mut sv = StateVector::<f64>::new(n);
        sv.apply_circuit(&c).unwrap();
        sv
    });
    let full = |x: u128, j: usize| -> u128 {
        let mut y = x & !((1u128 << m) - 1);
        for i in 0..m {
            y |= (((j >> (m - 1 - i)) & 1) as u128) << i;
        }
        y
    };
    let backends = a.s("backends", "cpu64,cpu32");
    for t in 0..k {
        let x = rand_x(&mut rng, n);
        let tp = TailPlan::new(&cc, x, m);
        let js: Vec<usize> = (0..1usize << m).step_by(every).collect();
        let reference: Vec<Complex64> = js
            .iter()
            .map(|&j| match &sv {
                Some(sv) => sv.amplitude(full(x, j) as usize),
                None => chain_sweep::amplitude(&cc, full(x, j)).unwrap(),
            })
            .collect();
        let refnorm = reference.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        for b in backends.split(',') {
            let mut be: Box<dyn SweepBackend> = match b {
                "cpu64" => Box::new(CpuExact::<f64>::new(block_cfg())),
                "cpu32" => Box::new(CpuExact::<f32>::new(block_cfg())),
                f => {
                    let lp = qsim_lab::engines::chain_lowprec::LowPrec::parse(f).expect("format");
                    Box::new(CpuPacked::new(
                        lp,
                        block_cfg(),
                        a.get("l", 22),
                        a.get("slots", 14),
                    ))
                }
            };
            let (amps, st, ttail) = run_tail(be.as_mut(), &tp, &mut |_, _| {}).unwrap();
            let got: Vec<Complex64> = js.iter().map(|&j| amps[j]).collect();
            let err = got
                .iter()
                .zip(&reference)
                .map(|(g, e)| (g - e).norm())
                .fold(0.0f64, f64::max)
                / (refnorm / (js.len() as f64).sqrt());
            let ov: Complex64 = reference.iter().zip(&got).map(|(e, l)| e.conj() * l).sum();
            let nl: f64 = got.iter().map(|z| z.norm_sqr()).sum();
            let fid = ov.norm_sqr() / (refnorm * refnorm * nl);
            println!(
                "tailcheck n={n} d={} m={m} trial={t} backend={b} width={} R={} max|err|/rms={err:.3e} F={fid:.9} t_sweep={:.2}s t_tail={ttail:.2}s ref={}",
                a.get::<usize>("d", 70),
                tp.sweep.width,
                st.passes,
                st.secs,
                if sv.is_some() { "statevector" } else { "chain-sweep" }
            );
        }
    }
}

/// Times the tail pass alone on a synthetic register (hash values, no
/// store): `--d D --m M`; prints seconds and complex MACs per second.
fn tailbench(a: &Args) {
    use qsim_lab::engines::chain_tail::{tail_amplitudes, TailPlan};
    let c = circuit(a);
    let cc = ChainCircuit::from_circuit(&c).unwrap();
    let m: usize = a.get("m", 8);
    let tp = TailPlan::new(&cc, 0, m);
    let reg = |i: usize| {
        let h = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        Complex64::new(
            (h >> 40) as f64 * 1e-7 - 0.8,
            ((h >> 16) & 0xffffff) as f64 * 1e-7 - 0.8,
        )
    };
    let t = Instant::now();
    let v = tail_amplitudes(&tp, &reg);
    let s = t.elapsed().as_secs_f64();
    println!(
        "tailbench d={} m={m} K={} cmacs={:.3e} t={s:.2}s rate={:.3e} cMAC/s threads={} amp0={:.3e}",
        a.get::<usize>("d", 70),
        tp.num_bonds(),
        tp.tail_cost(),
        tp.tail_cost() / s,
        rayon::current_num_threads(),
        v[0]
    );
}

/// The sampling / calibration run loop (RUNPLAN §6 E3/E4). One process,
/// one register allocation; for every job not yet done in `--out` (resume
/// by key, in job order, a crashed job is redone): append a `start` line,
/// sweep, tail pass, append the result record (fsync); `--heartbeat FILE`
/// is rewritten after every pass. Jobs come from `--jobs FILE` (lines of
/// `analyze.py prefixes` / `calrows`) or from `--seed-file F --count N
/// [--start S] --m M` (the same prefixes as `analyze.py prefixes`).
/// `--take K` stops after K new jobs.
fn runloop(a: &Args) {
    use qsim_lab::engines::chain_run::{
        bitstring, draw, hex, jamps, jnum, json_field, parse_job, read_seed, seed_jobs, sha256, Job,
    };
    use qsim_lab::engines::chain_tail::{tail_amplitudes, TailPlan};
    use std::io::Write;
    keepawake::on();
    let c = circuit(a);
    let n = c.num_qubits;
    let d: usize = a.get("d", 70);
    let cc = ChainCircuit::from_circuit(&c).unwrap();
    let out = a.s("out", "run.jsonl");
    let hb = a.s("heartbeat", "heartbeat.txt");
    let take: usize = a.get("take", usize::MAX);
    let mut seed_hash = String::new();
    let jobs: Vec<Job> = if let Some(p) = a.0.iter().position(|x| x == "--jobs") {
        let txt = std::fs::read_to_string(&a.0[p + 1]).expect("read jobs");
        txt.lines().filter_map(parse_job).collect()
    } else {
        let p =
            a.0.iter()
                .position(|x| x == "--seed-file")
                .expect("--jobs FILE or --seed-file F --count N --m M");
        let seed = read_seed(&std::fs::read(&a.0[p + 1]).expect("read seed"));
        seed_hash = hex(&sha256(&seed));
        seed_jobs(
            &seed,
            a.get("m", 6),
            n,
            a.get("start", 0u64),
            a.get("count", 0u64),
        )
    };
    for j in &jobs {
        j.validate(n).unwrap_or_else(|e| panic!("{e}"));
    }
    let mut keys = std::collections::HashSet::new();
    for j in &jobs {
        assert!(keys.insert(j.key()), "duplicate job {}", j.key());
    }
    // resume: keys with a result record
    let mut done = std::collections::HashSet::new();
    if let Ok(txt) = std::fs::read_to_string(&out) {
        for l in txt.lines() {
            match json_field(l, "kind") {
                Some("sample") => {
                    done.insert(format!("s{}", json_field(l, "i").unwrap_or("?")));
                }
                Some("calib") => {
                    done.insert(format!("c{}", json_field(l, "row").unwrap_or("?")));
                }
                _ => {}
            }
        }
    }
    let todo: Vec<&Job> = jobs
        .iter()
        .filter(|j| !done.contains(&j.key()))
        .take(take)
        .collect();
    let mut be = tail_backend(a);
    let host = std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_default();
    eprintln!(
        "run: n={n} d={d} jobs={} done={} todo={} backend=\"{}\" out={out}",
        jobs.len(),
        done.len(),
        todo.len(),
        be.describe()
    );
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out)
        .expect("open out");
    let now = || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
    };
    let write_line = |f: &mut std::fs::File, s: &str| {
        writeln!(f, "{s}").expect("write record");
        f.sync_all().expect("fsync record");
    };
    let beat = |job: &str, phase: &str, pass: usize, total: usize, t0: f64| {
        let tmp = format!("{hb}.tmp");
        let txt = format!(
            "time_unix={:.0} pid={} job={job} phase={phase} pass={pass}/{total} elapsed_s={:.0}\n",
            now(),
            std::process::id(),
            now() - t0
        );
        if std::fs::write(&tmp, txt).is_ok() {
            let _ = std::fs::rename(&tmp, &hb);
        }
    };
    for job in todo {
        let m = job.tail_m();
        let key = job.key();
        let t0 = now();
        beat(&key, "compile", 0, 0, t0);
        let tp = TailPlan::new(&cc, job.x(), m);
        let id = match (job.i, job.row) {
            (Some(i), _) => format!("\"i\":{i}"),
            (None, Some(r)) => format!("\"row\":{r}"),
            _ => unreachable!(),
        };
        write_line(
            &mut f,
            &format!(
                "{{\"kind\":\"start\",{id},\"t_unix\":{:.0},\"pid\":{}}}",
                t0,
                std::process::id()
            ),
        );
        let ts = Instant::now();
        let st = be
            .sweep(&tp.sweep, &mut |p, tot| beat(&key, "sweep", p, tot, t0))
            .unwrap_or_else(|e| panic!("sweep {key}: {e}"));
        beat(&key, "tail", st.passes, st.passes, t0);
        let tt = Instant::now();
        let bref: &dyn qsim_lab::engines::chain_tail::SweepBackend = be.as_ref();
        let amps = tail_amplitudes(&tp, &|i| bref.amp(i));
        let ttail = tt.elapsed().as_secs_f64();
        let tsweep = ts.elapsed().as_secs_f64();
        let fmt = a.s("format", "int6:b64");
        let mut rec = if job.i.is_some() {
            let u = job.u_tail.unwrap();
            let jj = draw(&amps, u);
            format!(
                "{{\"kind\":\"sample\",{id},\"u_tail\":{},\"bitstring_q0_first\":\"{}\",\"j_tail\":{jj},",
                jnum(u),
                bitstring(job, jj)
            )
        } else {
            format!("{{\"kind\":\"calib\",{id},")
        };
        rec += &format!(
            "\"tail_m\":{m},\"prefix_bits\":\"{}\",\"format\":\"{fmt}\",\"R\":{},\"n\":{n},\"d\":{d},\"t_sweep_s\":{:.1},\"t_tail_s\":{:.1},\"underflow\":{},\"overflow\":{},\"backend\":\"{}\",\"host\":\"{host}\",\"seed_sha256\":\"{seed_hash}\",\"t_end_unix\":{:.0},\"amps\":{}}}",
            job.prefix_bits,
            st.passes,
            tsweep,
            ttail,
            st.underflow,
            st.overflow,
            be.describe(),
            now(),
            jamps(&amps)
        );
        write_line(&mut f, &rec);
        beat(&key, "done", st.passes, st.passes, t0);
        eprintln!(
            "run: {key} done R={} t_sweep={:.1}s (passes {:.1}s, tail {ttail:.1}s) underflow={} overflow={}",
            st.passes, tsweep, st.secs, st.underflow, st.overflow
        );
    }
    beat("-", "idle", 0, 0, now());
}

/// GPU backend of the packed sweep (`chain_packed_gpu`, `--features wgpu`).
/// Stages as `packed` (`--l`, `--slots`, 1q-fused unless `CS_FUSE=0`), run
/// with the FMA-tier config the GPU reproduces: nested cache blocks of
/// `2^--nb` amplitudes (default 12) with `--nslots` gathered bits (default
/// 6), no AVX-512 / dense fusion / tiling. `--count`: plan statistics only;
/// `--emul`: the CPU emulation of the GPU pipeline; `--cpu`: also the CPU
/// packed path (timing + bit-exact check); `--nogpu`: skip the GPU.
/// `--chunk B` streams 2^B amplitudes per chunk, `--inflight S` chunks in
/// flight; `--k K` bitstrings (default 1).
fn packedgpu(a: &Args) {
    use qsim_lab::engines::blocked::fuse_1q;
    use qsim_lab::engines::chain_lowprec::LowPrec;
    use qsim_lab::engines::chain_packed::{packed_stages, run_packed};
    use qsim_lab::engines::chain_packed_gpu::{emulate_packed, gpu_plans, plan_stats, Codec};
    let c = circuit(a);
    let n = c.num_qubits;
    let d: usize = a.get("d", 70);
    let cc = ChainCircuit::from_circuit(&c).unwrap();
    let k: usize = a.get("k", 1);
    let l: usize = a.get("l", 26);
    let slots: usize = a.get("slots", 14);
    let nb: usize = a.get("nb", 12);
    let flag = |f: &str| a.0.iter().any(|x| x == f);
    let lp = LowPrec::parse(&a.s("fmt", "int4:b16:h")).expect("bad --fmt");
    let cfg = BlockConfig {
        block_bytes: 8 << nb,
        slots: a.get("nslots", 6),
        fuse_1q: false,
        avx512: false,
        dense_fusion: 0,
        l1_tile_bytes: 0,
        ..BlockConfig::default()
    };
    let fuse = std::env::var("CS_FUSE").map(|v| v != "0").unwrap_or(true);
    let stages_for = |plan: &chain_sweep::SweepPlan| {
        let ops = if fuse {
            fuse_1q(&plan.ops, plan.width, false)
        } else {
            plan.ops.clone()
        };
        packed_stages(&ops, plan.width, l, slots)
    };
    if flag("--count") {
        let plan = compile(&cc, 0, &HashMap::new());
        let st = stages_for(&plan);
        let t = Instant::now();
        let gp = gpu_plans(&st, plan.width, &cfg).expect("export");
        let ps = plan_stats(&gp);
        let codec = Codec::new(&lp).expect("format");
        let amps = (1u64 << plan.width) as f64;
        let packed =
            amps * (codec.bits as f64 / 4.0 + codec.scale_bytes() as f64 / codec.block as f64);
        let gib = (1u64 << 30) as f64;
        println!(
            "packedgpu-count n={n} d={d} width={} ops={} l={l} slots={slots} nb={nb} passes={} subs={} (per pass {:.1}) ops_exported={} max_sub_l={} min_bc={} max_conds={} max_table={:.1} KiB export={:.2}s",
            plan.width,
            plan.ops.len(),
            ps.passes,
            ps.subs,
            ps.subs as f64 / ps.passes as f64,
            ps.ops,
            ps.max_sub_l,
            ps.min_bc,
            ps.max_conds,
            ps.max_table_bytes as f64 / 1024.0,
            t.elapsed().as_secs_f64()
        );
        for rb in [3, 4, 5, 6] {
            let (o, loc, ph) = qsim_lab::engines::chain_packed_gpu::reg_stats(&gp, rb);
            println!("  register bits {rb}: {loc}/{o} ops register-local, {ph} shared phases");
        }
        println!(
            "  store {:.2} GiB; PCIe per pass {:.2} GiB each way, per sweep {:.0} GiB each way; VRAM f32 traffic per pass {:.0} GiB",
            packed / gib,
            packed / gib,
            packed * ps.passes as f64 / gib,
            amps * 8.0 * (2.0 + 2.0 * ps.subs as f64 / ps.passes as f64) / gib
        );
        return;
    }
    #[cfg(feature = "wgpu")]
    let gpu = if flag("--nogpu") {
        None
    } else {
        use qsim_lab::engines::chain_packed_gpu::gpu::{GpuOptions, GpuSweeper};
        let opts = GpuOptions {
            nested_bits: nb,
            reg_bits: a.get("rb", 0),
            chunk_amps: 1usize << a.get("chunk", 27usize),
            slots: a.get("inflight", 3),
            progress: std::env::var_os("CS_PROGRESS").is_some(),
            ..GpuOptions::default()
        };
        let g = GpuSweeper::new(&opts).expect("GPU adapter");
        println!("adapter: {}", g.adapter_info());
        Some(g)
    };
    let mut rng = StdRng::seed_from_u64(a.get("seed", 1));
    for _ in 0..k {
        let x = rand_x(&mut rng, n);
        let plan = compile(&cc, x, &HashMap::new());
        let st = stages_for(&plan);
        let mut cpu_amp = None;
        if flag("--cpu") {
            let r = run_packed(&plan, &st, &lp, &cfg).expect("cpu packed");
            println!(
                "cpu  n={n} d={d} width={} passes={} t={:.2}s amp={:.6e}",
                plan.width, r.passes, r.secs, r.amp
            );
            cpu_amp = Some(r.amp);
        }
        if flag("--emul") {
            let r = emulate_packed(&plan, &st, &lp, &cfg).expect("emulate");
            println!(
                "emul n={n} d={d} width={} passes={} t={:.2}s amp={:.6e}{}",
                plan.width,
                r.passes,
                r.secs,
                r.amp,
                cpu_amp.map_or(String::new(), |c| format!(" bit-exact={}", c == r.amp))
            );
        }
        #[cfg(feature = "wgpu")]
        if let Some(g) = &gpu {
            let r = g.run(&plan, &st, &lp, &cfg).expect("gpu run");
            let mut ps = r.pass_secs.clone();
            ps.sort_by(f64::total_cmp);
            println!(
                "gpu  n={n} d={d} width={} passes={} store={:.2} GiB t={:.2}s (pass median {:.2}s max {:.2}s) uf/of/inexact={}/{}/{} amp={:.6e}{}",
                plan.width,
                r.passes,
                r.store_bytes as f64 / (1u64 << 30) as f64,
                r.secs,
                ps[ps.len() / 2],
                ps[ps.len() - 1],
                r.underflow,
                r.overflow,
                r.inexact,
                r.amp,
                cpu_amp.map_or(String::new(), |c| format!(" bit-exact={}", c == r.amp))
            );
        }
    }
}
