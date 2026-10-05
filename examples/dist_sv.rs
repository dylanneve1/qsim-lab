//! Two-node distributed state vector driver (see `qsim_lab::engines::dist` and
//! `research/performance/distributed-sv.md`).
//!
//! Both nodes run the same command line except for `--node` and the
//! connection side; node 0 normally listens, node 1 connects.
//!
//! ```text
//! dist_sv link --listen 127.0.0.1:47100 [--mb 64] [--pings 40]
//! dist_sv link --connect 127.0.0.1:47100 [--mb 64] [--pings 40]
//! dist_sv run  --node 0 --listen ADDR  --workload qft --n 26 --prec f32 \
//!              --local-bits 23 --owner 00000001 [--restore 1] [--free 1] [--fold 1] \
//!              [--basis X] [--verify qft|norm] [--msg-kib 4096] [--verbose 1]
//! dist_sv run  --node 1 --connect ADDR ... (same flags)
//! dist_sv plan --workload qft --n 30 --local-bits 27 [--restore 1]
//! dist_sv ram  --workload qft --n 26 --prec f32 [--reps 3]
//! ```
//!
//! `--verify`: `norm` (default), `qft` (each node checks its amplitudes
//! against the analytic QFT of `--basis X`), `ref` (gather on node 0 and
//! compare with the single-node blocked executor; small n only).
//!
//! Workloads: `qft`, `brick` (4 layers), `brick16`, `brickd` (`--depth`
//! layers, default 2n), `ghz`. Output: one
//! `key=value` line per node on stdout.

use num_complex::{Complex, Complex64};
use qsim_lab::algorithms;
use qsim_lab::circuit::Circuit;
use qsim_lab::engines::blocked::BlockConfig;
use qsim_lab::engines::dist::{
    barrier, fingerprint, handshake, plan_circuit, DistConfig, DistState, DistStep, Link, PipeLink,
    PlanOptions, TcpLink,
};
use qsim_lab::engines::statevector::{Real, StateVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::net::TcpListener;
use std::os::fd::FromRawFd;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

fn args() -> (String, HashMap<String, String>) {
    let mut it = std::env::args().skip(1);
    let mode = it.next().unwrap_or_else(|| "help".into());
    let mut kv = HashMap::new();
    while let Some(k) = it.next() {
        let k = k.trim_start_matches("--").to_string();
        let v = it
            .next()
            .unwrap_or_else(|| panic!("missing value for --{k}"));
        kv.insert(k, v);
    }
    (mode, kv)
}

fn get<T: std::str::FromStr>(kv: &HashMap<String, String>, k: &str, d: T) -> T {
    kv.get(k)
        .map(|v| v.parse().unwrap_or_else(|_| panic!("bad --{k}")))
        .unwrap_or(d)
}

fn workload(name: &str, n: usize, depth: usize) -> Circuit {
    let mut rng = StdRng::seed_from_u64(42 + n as u64);
    match name {
        "brickd" => algorithms::random_brickwork(n, depth, &mut rng),
        "ghz" => algorithms::ghz(n),
        "qft" => algorithms::qft(n),
        "brick" => algorithms::random_brickwork(n, 4, &mut rng),
        "brick16" => algorithms::random_brickwork(n, 16, &mut rng),
        _ => panic!("unknown workload {name}"),
    }
}

/// The link to the other node, and the spawned peer process (if any).
struct Conn {
    link: Box<dyn Link>,
    child: Option<Child>,
    /// Our stdout is the link (`--stdio 1`): report on stderr instead.
    stdio: bool,
}

impl Conn {
    fn report(&self, line: &str) {
        if self.stdio {
            eprintln!("{line}");
        } else {
            println!("{line}");
        }
    }
    fn finish(mut self) {
        drop(self.link);
        if let Some(c) = self.child.as_mut() {
            let st = c.wait().expect("wait for peer");
            assert!(st.success(), "peer exited with {st}");
        }
    }
}

/// `--listen ADDR` / `--connect ADDR` (TCP), `--stdio 1` (our stdin/stdout,
/// when started by the peer through `ssh host cmd`), or `--spawn CMD` (start
/// the peer with `sh -c CMD` and talk over its stdin/stdout).
fn connect(kv: &HashMap<String, String>) -> Conn {
    let tcp = |link: TcpLink| Conn {
        link: Box::new(link),
        child: None,
        stdio: false,
    };
    if let Some(a) = kv.get("listen") {
        let l = TcpListener::bind(a).expect("bind");
        eprintln!("listening on {a}");
        tcp(TcpLink::accept(&l).expect("accept"))
    } else if let Some(a) = kv.get("connect") {
        tcp(TcpLink::connect(a.as_str(), Duration::from_secs(120)).expect("connect"))
    } else if kv.get("stdio").map(|s| s.as_str()) == Some("1") {
        // SAFETY: fds 0 and 1 are open for the life of the process and are
        // used only through these handles from here on.
        let (r, w) = unsafe { (File::from_raw_fd(0), File::from_raw_fd(1)) };
        Conn {
            link: Box::new(PipeLink::new(r, w)),
            child: None,
            stdio: true,
        }
    } else if let Some(cmd) = kv.get("spawn") {
        let mut child = Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("spawn peer");
        let r = child.stdout.take().unwrap();
        let w = child.stdin.take().unwrap();
        Conn {
            link: Box::new(PipeLink::new(r, w)),
            child: Some(child),
            stdio: false,
        }
    } else {
        panic!("need --listen, --connect, --stdio 1 or --spawn CMD")
    }
}

/// Node role for the link benchmark: the side that listens or spawns is 0.
fn role(kv: &HashMap<String, String>) -> u8 {
    if kv.contains_key("listen") || kv.contains_key("spawn") {
        0
    } else {
        1
    }
}

fn opts(kv: &HashMap<String, String>) -> PlanOptions {
    PlanOptions {
        free_initial_layout: get(kv, "free", 1u8) == 1,
        fold_swaps: get(kv, "fold", 1u8) == 1,
        restore_order: get(kv, "restore", 0u8) == 1,
    }
}

fn link_bench(kv: &HashMap<String, String>) {
    let mut conn = connect(kv);
    let link: &mut dyn Link = conn.link.as_mut();
    let me = role(kv);
    let mb: usize = get(kv, "mb", 64);
    let pings: usize = get(kv, "pings", 40);
    barrier(link).unwrap();
    // Latency: node 0 pings.
    let mut rtts = Vec::new();
    let mut b = [0u8; 8];
    for _ in 0..pings {
        if me == 0 {
            let t = Instant::now();
            link.send_all(&b).unwrap();
            link.recv_exact(&mut b).unwrap();
            rtts.push(t.elapsed().as_secs_f64() * 1e3);
        } else {
            link.recv_exact(&mut b).unwrap();
            link.send_all(&b).unwrap();
        }
    }
    let bytes = mb << 20;
    let blk = vec![0x5Au8; 1 << 20];
    let one_way = |link: &mut dyn Link, sender: bool| -> f64 {
        barrier(link).unwrap();
        let t = Instant::now();
        if sender {
            for _ in 0..mb {
                link.send_all(&blk).unwrap();
            }
            let mut ack = [0u8; 1];
            link.recv_exact(&mut ack).unwrap();
        } else {
            let mut rbuf = vec![0u8; 1 << 20];
            for _ in 0..mb {
                link.recv_exact(&mut rbuf).unwrap();
            }
            link.send_all(&[1]).unwrap();
        }
        bytes as f64 / t.elapsed().as_secs_f64() / 1048576.0
    };
    let up = one_way(link, me == 1); // node 1 -> node 0
    let down = one_way(link, me == 0); // node 0 -> node 1
    barrier(link).unwrap();
    let t = Instant::now();
    {
        let (w, r) = link.split();
        std::thread::scope(|s| {
            s.spawn(|| {
                for _ in 0..mb {
                    w.write_all(&blk).unwrap();
                }
                w.flush().unwrap();
            });
            let mut rb = vec![0u8; 1 << 20];
            for _ in 0..mb {
                r.read_exact(&mut rb).unwrap();
            }
        });
    }
    barrier(link).unwrap();
    let duplex = bytes as f64 / t.elapsed().as_secs_f64() / 1048576.0;
    rtts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = rtts.get(rtts.len() / 2).copied().unwrap_or(f64::NAN);
    let line = format!(
        "link node={me} mb={mb} rtt_ms_med={med:.3} rtt_ms_min={:.3} node1_to_node0_MiBps={up:.1} node0_to_node1_MiBps={down:.1} duplex_each_way_MiBps={duplex:.1}",
        rtts.first().copied().unwrap_or(f64::NAN)
    );
    conn.report(&line);
    conn.finish();
}

/// Max |amp - QFT|x>| over this node's amplitudes, in the current layout.
fn qft_error<T: Real>(st: &DistState<T>, x: usize) -> f64 {
    let n = st.num_qubits();
    let v2p = st.layout().to_vec();
    let lb = st
        .owned()
        .next()
        .map(|(_, d)| d.len().trailing_zeros() as usize);
    let Some(lb) = lb else { return 0.0 };
    let mut p2v = vec![0usize; n];
    for (v, &p) in v2p.iter().enumerate() {
        p2v[p] = v;
    }
    let h = lb / 2;
    let tab = |bits: std::ops::Range<usize>| -> Vec<usize> {
        let w = bits.len();
        (0..1usize << w)
            .map(|t| {
                let mut x = 0;
                for (j, p) in bits.clone().enumerate() {
                    x |= ((t >> j) & 1) << p2v[p];
                }
                x
            })
            .collect()
    };
    let lo = tab(0..h);
    let hi = tab(h..lb);
    let mask = (1usize << h) - 1;
    let nn = (1usize << n) as f64;
    let amp = nn.powf(-0.5);
    let nmask = (1usize << n) - 1;
    st.owned()
        .map(|(r, d)| {
            let mut rp = 0;
            for (j, &v) in p2v[lb..].iter().enumerate() {
                rp |= ((r >> j) & 1) << v;
            }
            d.par_iter()
                .enumerate()
                .map(|(off, a)| {
                    let k = rp | lo[off & mask] | hi[off >> h];
                    let ph = (x.wrapping_mul(k) & nmask) as f64 / nn * std::f64::consts::TAU;
                    let w = Complex64::from_polar(amp, ph);
                    let a = Complex64::new(a.re.to_f64(), a.im.to_f64());
                    (a - w).norm()
                })
                .reduce(|| 0.0, f64::max)
        })
        .fold(0.0, f64::max)
}

fn run_node<T: Real>(kv: &HashMap<String, String>, prec: &str) {
    let node: u8 = get(kv, "node", 0);
    let wl: String = get(kv, "workload", "qft".to_string());
    let n: usize = get(kv, "n", 20);
    let l: usize = get(kv, "local-bits", n - 1);
    let owner: Vec<u8> = get(kv, "owner", "01".to_string())
        .bytes()
        .map(|b| b - b'0')
        .collect();
    let x: usize = get(kv, "basis", 0);
    let verify: String = get(kv, "verify", "norm".to_string());
    let verbose: u8 = get(kv, "verbose", 0);
    let cfg = DistConfig {
        block: BlockConfig::default(),
        msg_amps: (get::<usize>(kv, "msg-kib", 4096) << 10) / std::mem::size_of::<Complex<T>>(),
    };
    let circ = workload(&wl, n, get(kv, "depth", 2 * n));
    let t = Instant::now();
    let o = opts(kv);
    let plan = plan_circuit(&circ, l, &o).expect("plan");
    let plan_ms = t.elapsed().as_secs_f64() * 1e3;
    let mut conn = connect(kv);
    let link: &mut dyn Link = conn.link.as_mut();
    let fp = fingerprint(&plan, &owner, std::mem::size_of::<Complex<T>>());
    handshake(link, node, fp).expect("handshake");
    let mut st = DistState::<T>::new_basis(n, l, node, owner.clone(), &plan.initial_v2p, x, cfg);
    let local_gib = st.local_bytes() as f64 / (1u64 << 30) as f64;
    barrier(link).unwrap();
    let t0 = Instant::now();
    if verbose == 1 {
        for (i, step) in plan.steps.iter().enumerate() {
            let ts = Instant::now();
            match step {
                DistStep::Local(g) => st.apply_local(g),
                DistStep::Swap { local, global } => st.swap_qubits(*local, *global, link).unwrap(),
                DistStep::Relabel(p) => st.relabel(p),
                DistStep::Rename { a, b } => st.rename(*a, *b),
            }
            eprintln!(
                "[node {node}] step {i}/{} {} {:.2}s (t={:.1}s)",
                plan.steps.len(),
                match step {
                    DistStep::Local(g) => format!("local {} gates", g.len()),
                    DistStep::Swap { local, global } => format!("swap {local}<->{global}"),
                    DistStep::Relabel(p) => format!("relabel {} pairs", p.len()),
                    DistStep::Rename { a, b } => format!("rename {a}<->{b}"),
                },
                ts.elapsed().as_secs_f64(),
                t0.elapsed().as_secs_f64()
            );
        }
    } else {
        st.run(&plan, link).expect("run");
    }
    barrier(link).unwrap();
    let wall = t0.elapsed().as_secs_f64();
    let norm = st.norm_sqr(link).unwrap();
    let err = if verify == "ref" {
        // Gather on node 0 and compare with the single-node blocked executor.
        match st.gather(link).unwrap() {
            Some(got) => {
                let mut v = vec![Complex::new(T::zero(), T::zero()); 1 << n];
                let mut phys = 0usize;
                for q in 0..n {
                    phys |= ((x >> q) & 1) << q;
                }
                v[phys] = Complex::new(T::one(), T::zero());
                let mut sv = StateVector::<T>::from_amplitudes(v);
                sv.apply_circuit_blocked(&circ, &BlockConfig::default())
                    .unwrap();
                let e = got
                    .iter()
                    .zip(sv.amplitudes())
                    .map(|(a, b)| {
                        let d = *a - *b;
                        (d.re.to_f64().powi(2) + d.im.to_f64().powi(2)).sqrt()
                    })
                    .fold(0.0, f64::max);
                link.send_all(&e.to_le_bytes()).unwrap();
                e
            }
            None => {
                let mut b = [0u8; 8];
                link.recv_exact(&mut b).unwrap();
                f64::from_le_bytes(b)
            }
        }
    } else if verify == "qft" {
        let e = qft_error(&st, x);
        // max over both nodes
        link.send_all(&e.to_le_bytes()).unwrap();
        let mut b = [0u8; 8];
        link.recv_exact(&mut b).unwrap();
        e.max(f64::from_le_bytes(b))
    } else {
        f64::NAN
    };
    let s = st.stats();
    let owner_s: String = owner.iter().map(|o| (b'0' + o) as char).collect();
    let line = format!(
        "dist node={node} workload={wl} n={n} prec={prec} L={l} owner={owner_s} restore={} free={} fold={} \
         plan_swaps={} plan_runs={} folded={} plan_ms={plan_ms:.1} local_gib={local_gib:.3} cross_pairs={} local_pairs={} \
         sent_mib={:.1} recv_mib={:.1} wall_s={wall:.3} compute_s={:.3} exchange_s={:.3} local_swap_s={:.3} \
         norm={norm:.9} verify={verify} max_err={err:.3e}",
        o.restore_order as u8,
        o.free_initial_layout as u8,
        o.fold_swaps as u8,
        plan.swaps,
        plan.runs,
        plan.folded_swaps,
        s.cross_pairs,
        s.local_pairs,
        s.bytes_sent as f64 / 1048576.0,
        s.bytes_recv as f64 / 1048576.0,
        s.compute.as_secs_f64(),
        s.exchange.as_secs_f64(),
        s.local_swap.as_secs_f64(),
    );
    drop(st);
    conn.report(&line);
    conn.finish();
}

fn plan_only(kv: &HashMap<String, String>) {
    let wl: String = get(kv, "workload", "qft".to_string());
    let n: usize = get(kv, "n", 20);
    let o = opts(kv);
    for l in (n.saturating_sub(4).max(3)..n).rev() {
        let plan = plan_circuit(&workload(&wl, n, get(kv, "depth", 2 * n)), l, &o).unwrap();
        println!(
            "plan workload={wl} n={n} L={l} G={} swaps={} runs={} folded={} restore={}",
            n - l,
            plan.swaps,
            plan.runs,
            plan.folded_swaps,
            o.restore_order as u8
        );
    }
}

fn ram<T: Real>(kv: &HashMap<String, String>, prec: &str) {
    let wl: String = get(kv, "workload", "qft".to_string());
    let n: usize = get(kv, "n", 20);
    let reps: usize = get(kv, "reps", 3);
    let c = workload(&wl, n, get(kv, "depth", 2 * n));
    let cfg = BlockConfig::default();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        // `from_amplitudes` bypasses the 1 GiB guard of `StateVector::new`.
        let mut v = vec![Complex::new(T::zero(), T::zero()); 1 << n];
        v[0] = Complex::new(T::one(), T::zero());
        let mut sv = StateVector::<T>::from_amplitudes(v);
        let t = Instant::now();
        sv.apply_circuit_blocked(&c, &cfg).unwrap();
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("ram workload={wl} n={n} prec={prec} best_s={best:.3}");
}

fn main() {
    let (mode, kv) = args();
    let prec: String = get(&kv, "prec", "f32".to_string());
    match (mode.as_str(), prec.as_str()) {
        ("link", _) => link_bench(&kv),
        ("plan", _) => plan_only(&kv),
        ("run", "f32") => run_node::<f32>(&kv, "f32"),
        ("run", "f64") => run_node::<f64>(&kv, "f64"),
        ("ram", "f32") => ram::<f32>(&kv, "f32"),
        ("ram", "f64") => ram::<f64>(&kv, "f64"),
        _ => eprintln!("usage: see the module docs of examples/dist_sv.rs"),
    }
}
