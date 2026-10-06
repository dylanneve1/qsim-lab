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

/// Writes the left and right boundary tensors of edge `e` for `k` random
/// bitstrings as raw little-endian f64 (re, im) pairs, for offline spectra.
fn dumpcut(a: &Args) {
    let c = circuit(a);
    let n = c.num_qubits;
    let cc = ChainCircuit::from_circuit(&c).unwrap();
    let e: usize = a.get("e", n / 2 - 1);
    let k: usize = a.get("k", 1);
    let out = a.s("out", "cut");
    let cfg = BlockConfig::default();
    let mut rng = StdRng::seed_from_u64(a.get("seed", 1));
    for i in 0..k {
        let x = rand_x(&mut rng, n);
        let ct = chain_sweep::cut_tensors_cpu::<f64>(&cc, x, e, &cfg).unwrap();
        for (tag, v) in [("L", &ct.left), ("R", &ct.right)] {
            let bytes: Vec<u8> = v
                .iter()
                .flat_map(|z| [z.re.to_le_bytes(), z.im.to_le_bytes()].concat())
                .collect();
            std::fs::write(format!("{out}_{i}_{tag}.bin"), bytes).unwrap();
        }
        let layers: Vec<usize> = ct.bonds.iter().map(|&b| cc.bond_layer[b]).collect();
        println!(
            "dumpcut i={i} x={x} e={e} bonds={} layers={layers:?} amp={}",
            ct.bonds.len(),
            ct.amplitude()
        );
    }
}

/// Exact amplitudes of `k` uniformly random bitstrings, one per line
/// (`x re im seconds`), for the boundary-MPS fidelity runs.
fn exactamps(a: &Args) {
    let c = circuit(a);
    let n = c.num_qubits;
    let cc = ChainCircuit::from_circuit(&c).unwrap();
    let k: usize = a.get("k", 10);
    let backend = a.s("backend", "cpu32");
    let mut rng = StdRng::seed_from_u64(a.get("seed", 1));
    for _ in 0..k {
        let x = rand_x(&mut rng, n);
        let plan = compile(&cc, x, &HashMap::new());
        let (amp, t, _) = run(&plan, &backend);
        println!("{x} {:.12e} {:.12e} {t:.3}", amp.re, amp.im);
    }
}

fn parse_chis(a: &Args) -> Vec<usize> {
    a.s("chis", "16,64,256")
        .split(',')
        .map(|v| v.parse().expect("chi"))
        .collect()
}

/// Boundary-MPS amplitudes against exact ones (from `exactamps`, or computed
/// here on the CPU if `--amps` is not given): fidelity, truncation estimate,
/// time and memory per bond cap.
fn mpsfid(a: &Args) {
    use rayon::prelude::*;
    faer::set_global_parallelism(faer::Par::Seq);
    let c = circuit(a);
    let n = c.num_qubits;
    let cc = ChainCircuit::from_circuit(&c).unwrap();
    let cutoff: f64 = a.get("cutoff", 1e-13);
    let pairs: Vec<(u128, Complex64)> = match a.0.iter().position(|x| x == "--amps") {
        Some(i) => std::fs::read_to_string(&a.0[i + 1])
            .expect("read amps")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let f: Vec<&str> = l.split_whitespace().collect();
                (
                    f[0].parse().unwrap(),
                    Complex64::new(f[1].parse().unwrap(), f[2].parse().unwrap()),
                )
            })
            .collect(),
        None => {
            // --noexact: truncation estimates only (exact amplitudes set to 0)
            let exact = !a.0.iter().any(|x| x == "--noexact");
            let mut rng = StdRng::seed_from_u64(a.get("seed", 1));
            (0..a.get("k", 8))
                .map(|_| {
                    let x = rand_x(&mut rng, n);
                    let plan = compile(&cc, x, &HashMap::new());
                    let amp = if exact {
                        chain_sweep::amplitude_cpu::<f64>(&plan, &BlockConfig::default()).unwrap()
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    (x, amp)
                })
                .collect()
        }
    };
    let d: usize = a.get("d", 70);
    println!(
        "mpsfid n={n} d={d} bitstrings={} cutoff={cutoff:e}",
        pairs.len()
    );
    for chi in parse_chis(a) {
        let t0 = Instant::now();
        let res: Vec<_> = pairs
            .par_iter()
            .map(|&(x, ex)| {
                let plan = compile(&cc, x, &HashMap::new());
                let t = Instant::now();
                let r = qsim_lab::engines::chain_mps::amplitude_mps(&plan, chi, cutoff);
                (ex, r, t.elapsed().as_secs_f64())
            })
            .collect();
        let wall = t0.elapsed().as_secs_f64();
        let ov: Complex64 = res.iter().map(|(ex, r, _)| ex.conj() * r.amp).sum();
        let ne: f64 = res.iter().map(|(ex, _, _)| ex.norm_sqr()).sum();
        let na: f64 = res.iter().map(|(_, r, _)| r.amp.norm_sqr()).sum();
        let fid = ov.norm_sqr() / (ne * na).max(1e-300);
        // bootstrap standard error of F over the bitstrings
        let mut brng = StdRng::seed_from_u64(99);
        let boots: Vec<f64> = (0..200)
            .map(|_| {
                let (mut o, mut e2, mut a2) = (Complex64::new(0.0, 0.0), 0.0, 0.0);
                for _ in 0..res.len() {
                    let (ex, r, _) = &res[brng.random_range(0..res.len())];
                    o += ex.conj() * r.amp;
                    e2 += ex.norm_sqr();
                    a2 += r.amp.norm_sqr();
                }
                o.norm_sqr() / (e2 * a2).max(1e-300)
            })
            .collect();
        let bm = boots.iter().sum::<f64>() / boots.len() as f64;
        let fse = (boots.iter().map(|f| (f - bm).powi(2)).sum::<f64>() / boots.len() as f64).sqrt();
        let fe: Vec<f64> = res.iter().map(|(_, r, _)| r.fid_est).collect();
        let fe_mean = fe.iter().sum::<f64>() / fe.len() as f64;
        let lfe = fe.iter().map(|f| f.max(1e-300).log10()).sum::<f64>() / fe.len() as f64;
        let tmean = res.iter().map(|r| r.2).sum::<f64>() / res.len() as f64;
        let pchi = res.iter().map(|r| r.1.counters.peak_chi).max().unwrap();
        let pbytes = res.iter().map(|r| r.1.counters.peak_bytes).max().unwrap();
        let svds = res.iter().map(|r| r.1.counters.svds).sum::<usize>() / res.len();
        let swaps = res.iter().map(|r| r.1.counters.swaps).sum::<usize>() / res.len();
        // error over the RMS amplitude 2^-n/2 (exact zeros are common)
        let rms = 0.5f64.powf(n as f64 / 2.0);
        let maxrel = res
            .iter()
            .map(|(ex, r, _)| (r.amp - ex).norm() / rms)
            .fold(0.0f64, f64::max);
        println!(
            "chi={chi} F={fid:.5} F_se={fse:.5} norm_ratio={:.5} fid_est_mean={fe_mean:.4e} log10_fid_est_mean={lfe:.3} \
             max_err/rms={maxrel:.2e} t_per_amp={tmean:.3}s wall={wall:.1}s peak_chi={pchi} \
             peak_mib={:.1} svds={svds} swaps={swaps}",
            na / ne,
            pbytes as f64 / 1048576.0
        );
    }
}

/// Bond dimensions and the truncation estimate after every qubit of one
/// bitstring's sweep.
fn mpsprofile(a: &Args) {
    faer::set_global_parallelism(faer::Par::Seq);
    let c = circuit(a);
    let n = c.num_qubits;
    let cc = ChainCircuit::from_circuit(&c).unwrap();
    let mut rng = StdRng::seed_from_u64(a.get("seed", 1));
    let x = rand_x(&mut rng, n);
    let plan = compile(&cc, x, &HashMap::new());
    let chi: usize = a.get("chi", 64);
    let cutoff: f64 = a.get("cutoff", 1e-13);
    let mut m = qsim_lab::engines::chain_mps::BoundaryMps::new(plan.width, chi, cutoff);
    let t = Instant::now();
    for i in 0..n {
        for op in &plan.ops[plan.qubit_ops[i]..plan.qubit_ops[i + 1]] {
            m.apply_kop(op);
        }
        let bd = m.bond_dims();
        let lb: Vec<String> = bd
            .iter()
            .map(|&b| format!("{:.1}", (b as f64).log2()))
            .collect();
        println!(
            "q={i:2} cut_width={} fid_est={:.4e} log2_chi=[{}] t={:.1}s",
            plan.cut_width[i],
            m.fid_est(),
            lb.join(" "),
            t.elapsed().as_secs_f64()
        );
    }
    println!(
        "amp={} counters={:?}",
        m.zero_amplitude() * plan.scale,
        m.counters
    );
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
        Some("dumpcut") => dumpcut(&a),
        Some("exactamps") => exactamps(&a),
        Some("mpsfid") => mpsfid(&a),
        Some("mpsprofile") => mpsprofile(&a),
        _ => eprintln!("usage: chain_sweep info|validate|bench|fidsv|fidmitm [--n N --d D ...]"),
    }
}
