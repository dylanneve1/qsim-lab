//! Driver for the tensor-network engine studies (research/simulability/tn.md).
//!
//! ```text
//! tn_bench syc FILE [--bits HEX] [--open N] [--trials N] [--secs S] [--target L2]
//!                   [--seed S] [--no-reconf] [--greedy] [--bisect] [--dump JSON]
//!                   [--run] [--f32] [--threads T] [--mem-gb G] [--reps R]
//!     one amplitude <bits|C|0> of a circuit file (format: research/data/tn/sycamore.py);
//!     prints the simplified network, the path statistics and (--run) the
//!     contraction time, as one JSON line per stage
//! tn_bench ising FIG THETA [--steps T] [same search/run flags]
//!     kicked-Ising expectation values on the 127-qubit Eagle lattice through
//!     the light cone: FIG = 3a (M_z, 5 steps) | 3b | 3c | 4a | z<q>
//! tn_bench json NETWORK.json [same flags]
//!     a network written by research/data/tn/cotengra_on_network.py (structure;
//!     entries from the .arrays.json sidecar when present, else random)
//! tn_bench svcal FAMILY N DEPTH SEED [--reps R]
//!     state-vector evolution time here vs the planner's M1 model (machine ratio)
//! tn_bench plan FAMILY N DEPTH SEED [--mem-gb G]
//!     planner::amplitudes of one basis state with and without Engine::Tn
//! tn_bench calib FAMILY N DEPTH SEED [same flags]
//!     one random instance (FAMILY = brick | sycpat | qaoa | rqc): path cost and
//!     contraction time for the planner's cost model
//! ```

use num_complex::Complex64;
use qsim_lab::circuit::Circuit;
use qsim_lab::engines::spd::{KickedIsing, Lattice, PauliObs};
use qsim_lab::engines::tn::{
    self, contract, network_json, ExecOptions, Hypergraph, Network, PairStrategy, PathOptions,
    Pauli, Precision, SimplifyOptions, TnOptions,
};
use qsim_lab::gate::Gate;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};
use std::time::Instant;

const W10: &str = "X13 X29 X31 Y9 Y30 Z8 Z12 Z17 Z28 Z32";
const W17: &str = "X37 X41 X52 X56 X57 X58 X62 X79 Y75 Z38 Z40 Z42 Z63 Z72 Z80 Z90 Z91";
const W17B: &str = "X37 X41 X52 X56 X57 X58 X62 X79 Y38 Y40 Y42 Y63 Y72 Y80 Y90 Y91 Z75";

/// Parses the circuit format of research/data/tn/sycamore.py into qsim-lab
/// gates (exact, global phase included).
fn parse_circuit(src: &str) -> Circuit {
    let mut c = Circuit::new(0);
    for line in src.lines() {
        let line = line.split('#').next().unwrap().trim();
        if line.is_empty() {
            continue;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        let q = |k: usize| -> usize { t[k].parse().expect("qubit") };
        let f = |k: usize| -> f64 { t[k].parse().expect("angle") };
        match t[0] {
            "qubits" => c = Circuit::new(q(1)),
            "sx" => {
                c.sx(q(1));
            }
            // sqrt(Y) = e^{i pi/4} Ry(pi/2) = Phase(pi/2) Rz(-pi/2) Ry(pi/2)
            "sy" => {
                c.u(q(1), FRAC_PI_2, 0.0, 0.0)
                    .rz(q(1), -FRAC_PI_2)
                    .phase(q(1), FRAC_PI_2);
            }
            // sqrt(W) = Z^{1/4} X^{1/2} Z^{-1/4}
            "sw" => {
                c.phase(q(1), -FRAC_PI_4).sx(q(1)).phase(q(1), FRAC_PI_4);
            }
            // fSim(pi/2, phi) = CPhase(-phi) iSWAP^dagger
            "fsim" => {
                let (th, ph) = (f(3), f(4));
                assert!(
                    (th - FRAC_PI_2).abs() < 1e-12,
                    "only fSim(pi/2, phi) is supported"
                );
                c.iswapdg(q(1), q(2)).cphase(q(1), q(2), -ph);
            }
            "h" => {
                c.h(q(1));
            }
            "x" => {
                c.x(q(1));
            }
            "z" => {
                c.z(q(1));
            }
            "t" => {
                c.t(q(1));
            }
            "rz" => {
                c.rz(q(1), f(2));
            }
            "rx" => {
                c.rx(q(1), f(2));
            }
            "cz" => {
                c.cz(q(1), q(2));
            }
            "cnot" | "cx" => {
                c.cnot(q(1), q(2));
            }
            other => panic!("unknown gate {other}"),
        }
    }
    c
}

struct Args(Vec<String>);
impl Args {
    fn get(&self, k: &str) -> Option<String> {
        self.0
            .iter()
            .position(|a| a == k)
            .map(|i| self.0[i + 1].clone())
    }
    fn has(&self, k: &str) -> bool {
        self.0.iter().any(|a| a == k)
    }
    fn num<T: std::str::FromStr>(&self, k: &str, d: T) -> T {
        self.get(k).map_or(d, |s| s.parse().ok().expect("number"))
    }
}

fn options(a: &Args) -> TnOptions {
    let mem_gb: f64 = a.num("--mem-gb", 2.0);
    TnOptions {
        precision: if a.has("--f32") {
            Precision::F32
        } else {
            Precision::F64
        },
        max_bytes: (mem_gb * (1u64 << 30) as f64) as u128,
        simplify: if a.has("--no-simplify") {
            SimplifyOptions::none()
        } else if a.has("--no-diag") {
            SimplifyOptions {
                diagonal: false,
                ..SimplifyOptions::default()
            }
        } else {
            SimplifyOptions::default()
        },
        path: PathOptions {
            trials: a.num("--trials", 64),
            max_secs: a.num("--secs", 60.0),
            target_log2_size: a.get("--target").map(|s| s.parse().unwrap()),
            reconf: !a.has("--no-reconf"),
            reconf_k: a.num("--reconf-k", 10),
            greedy: !a.has("--bisect"),
            bisect: !a.has("--greedy"),
            seed: a.num("--seed", 0x7e55_0001u64),
            refine_top: a.num("--refine", 8),
            polish_k: a.num("--polish", 12),
        },
        threads: a.num("--threads", 0),
        strategy: match a.get("--strategy").as_deref() {
            Some("permute") => PairStrategy::Permute,
            Some("loops") => PairStrategy::Loops,
            _ => PairStrategy::Auto,
        },
    }
}

fn load() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .map(|s| s.split_whitespace().next().unwrap_or("?").to_string())
        .unwrap_or_else(|_| "?".into())
}

/// Simplify + search + (optionally) contract one network; prints JSON lines.
fn run(label: &str, mut nw: Network, a: &Args) -> Option<Vec<Complex64>> {
    let o = options(a);
    let raw = nw.tensors.len();
    let t0 = Instant::now();
    let st = nw.simplify(&o.simplify);
    let simp_secs = t0.elapsed().as_secs_f64();
    if let Some(p) = a.get("--dump") {
        std::fs::write(&p, network_json(&nw)).expect("write network json");
    }
    let hg = Hypergraph::from_network(&nw);
    let mut po = o.path.clone();
    if po.target_log2_size.is_none() {
        po.target_log2_size = Some(tn::default_target_log2(&o));
    }
    let path = tn::search(&hg, &po);
    let s = &path.stats;
    println!(
        "{{\"stage\":\"path\",\"label\":\"{label}\",\"raw_tensors\":{raw},\"tensors\":{},\"indices\":{},\"simplify_secs\":{simp_secs:.3},\"log10_flops\":{:.4},\"log2_max_size\":{:.2},\"target\":{},\"log10_sliced_flops\":{:.4},\"log2_sliced_max_size\":{:.2},\"slices\":{},\"overhead\":{:.4},\"trials\":{},\"search_secs\":{:.2},\"method\":\"{}\",\"threads\":{},\"load\":\"{}\"}}",
        st.tensors_after,
        nw.dims.len(),
        s.log10_flops,
        s.log2_max_size,
        po.target_log2_size.unwrap(),
        s.log10_sliced_flops,
        s.log2_sliced_max_size,
        s.slices,
        s.overhead,
        s.trials,
        s.secs,
        s.method,
        rayon::current_num_threads(),
        load()
    );
    if !a.has("--run") {
        return None;
    }
    let reps: usize = a.num("--reps", 1);
    let eo = ExecOptions {
        max_bytes: o.max_bytes,
        threads: o.threads,
        strategy: o.strategy,
    };
    let mut out = None;
    for r in 0..reps {
        let l0 = load();
        let res = match o.precision {
            Precision::F64 => contract::<Complex64>(&nw, &path.tree, &path.sliced, &eo),
            Precision::F32 => {
                contract::<num_complex::Complex<f32>>(&nw, &path.tree, &path.sliced, &eo)
            }
        };
        match res {
            Ok((v, es)) => {
                let flops = 10f64.powf(s.log10_sliced_flops);
                println!(
                    "{{\"stage\":\"run\",\"label\":\"{label}\",\"rep\":{r},\"secs\":{:.4},\"value\":[{:.15e},{:.15e}],\"slices\":{},\"workers\":{},\"peak_bytes\":{},\"precision\":\"{:?}\",\"gflop_rate\":{:.2},\"load_before\":\"{l0}\",\"load_after\":\"{}\"}}",
                    es.secs,
                    v[0].re,
                    v[0].im,
                    es.slices,
                    es.workers,
                    es.peak_bytes,
                    o.precision,
                    8.0 * flops / es.secs / 1e9,
                    load()
                );
                out = Some(v);
            }
            Err(e) => {
                println!("{{\"stage\":\"run\",\"label\":\"{label}\",\"error\":\"{e}\"}}");
                return None;
            }
        }
    }
    out
}

/// Minimal JSON values (enough for the network files of
/// research/data/tn/cotengra_on_network.py).
#[derive(Debug)]
enum J {
    Num(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
    Other,
}

fn parse_json(s: &[u8], i: &mut usize) -> J {
    let ws = |i: &mut usize| {
        while *i < s.len() && (s[*i] as char).is_whitespace() {
            *i += 1;
        }
    };
    ws(i);
    match s[*i] {
        b'[' => {
            *i += 1;
            let mut v = Vec::new();
            loop {
                ws(i);
                if s[*i] == b']' {
                    *i += 1;
                    break;
                }
                v.push(parse_json(s, i));
                ws(i);
                if s[*i] == b',' {
                    *i += 1;
                }
            }
            J::Arr(v)
        }
        b'{' => {
            *i += 1;
            let mut v = Vec::new();
            loop {
                ws(i);
                if s[*i] == b'}' {
                    *i += 1;
                    break;
                }
                let J::Str(k) = parse_json(s, i) else {
                    panic!("object key")
                };
                ws(i);
                *i += 1; // ':'
                let val = parse_json(s, i);
                v.push((k, val));
                ws(i);
                if s[*i] == b',' {
                    *i += 1;
                }
            }
            J::Obj(v)
        }
        b'"' => {
            *i += 1;
            let st = *i;
            while s[*i] != b'"' {
                if s[*i] == b'\\' {
                    *i += 1;
                }
                *i += 1;
            }
            let out = String::from_utf8_lossy(&s[st..*i]).to_string();
            *i += 1;
            J::Str(out)
        }
        b'n' | b't' | b'f' => {
            while *i < s.len() && (s[*i] as char).is_ascii_alphabetic() {
                *i += 1;
            }
            J::Other
        }
        _ => {
            let st = *i;
            while *i < s.len() && matches!(s[*i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                *i += 1;
            }
            J::Num(
                std::str::from_utf8(&s[st..*i])
                    .unwrap()
                    .parse()
                    .expect("number"),
            )
        }
    }
}

impl J {
    fn get(&self, k: &str) -> Option<&J> {
        match self {
            J::Obj(v) => v.iter().find(|(x, _)| x == k).map(|(_, j)| j),
            _ => None,
        }
    }
    fn arr(&self) -> &[J] {
        match self {
            J::Arr(v) => v,
            _ => panic!("array expected"),
        }
    }
    fn num(&self) -> f64 {
        match self {
            J::Num(x) => *x,
            _ => panic!("number expected"),
        }
    }
}

/// A network from cotengra_on_network.py's JSON (structure) and, if the
/// `.arrays.json` sidecar exists, its tensor entries (random entries otherwise).
fn json_network(file: &str) -> Network {
    let src = std::fs::read(file).expect("network json");
    let j = parse_json(&src, &mut 0);
    let inputs: Vec<Vec<u32>> = j
        .get("inputs")
        .unwrap()
        .arr()
        .iter()
        .map(|t| t.arr().iter().map(|x| x.num() as u32).collect())
        .collect();
    let output: Vec<u32> = j
        .get("output")
        .unwrap()
        .arr()
        .iter()
        .map(|x| x.num() as u32)
        .collect();
    let mut dims: Vec<usize> = Vec::new();
    if let Some(J::Obj(sd)) = j.get("size_dict") {
        for (k, v) in sd {
            let i: usize = k.parse().expect("int index id");
            if dims.len() <= i {
                dims.resize(i + 1, 1);
            }
            dims[i] = v.num() as usize;
        }
    }
    let side = file.replace(".json", ".arrays.json");
    let mut scalar = Complex64::new(1.0, 0.0);
    let data: Vec<Vec<Complex64>> = match std::fs::read(&side) {
        Ok(b) => {
            let a = parse_json(&b, &mut 0);
            if let Some(e) = a.get("exponent") {
                scalar = Complex64::new(10f64.powf(e.num()), 0.0);
            }
            a.get("tensors")
                .unwrap()
                .arr()
                .iter()
                .map(|t| {
                    let re = t.get("re").unwrap().arr();
                    let im = t.get("im").unwrap().arr();
                    re.iter()
                        .zip(im)
                        .map(|(r, i)| Complex64::new(r.num(), i.num()))
                        .collect()
                })
                .collect()
        }
        Err(_) => {
            let mut rng = StdRng::seed_from_u64(1);
            inputs
                .iter()
                .map(|t| {
                    let n: usize = t.iter().map(|&i| dims[i as usize]).product();
                    (0..n)
                        .map(|_| Complex64::new(rng.random(), rng.random()))
                        .collect()
                })
                .collect()
        }
    };
    let tensors = inputs
        .into_iter()
        .zip(data)
        .map(|(inds, data)| tn::Tensor { inds, data })
        .collect();
    let mut nw = Network {
        tensors,
        dims,
        output,
        scalar,
    };
    nw.compact();
    nw
}

fn json_cmd(a: &Args) {
    let file = &a.0[1];
    let nw = json_network(file);
    let label = std::path::Path::new(file)
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    run(&label, nw, a);
}

fn syc(a: &Args) {
    let file = &a.0[1];
    let src = std::fs::read_to_string(file).expect("circuit file");
    let c = parse_circuit(&src);
    let n = c.num_qubits;
    let x = a
        .get("--bits")
        .map_or(0u128, |s| u128::from_str_radix(&s, 16).expect("hex"));
    let bits = tn::bits_of(x, n);
    let nopen: usize = a.num("--open", 0);
    let open: Vec<usize> = (0..nopen).collect();
    let nw = Network::amplitude(&c, &bits, &open).expect("network");
    let label = std::path::Path::new(file)
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    run(&label, nw, a);
}

fn ising(a: &Args) {
    let fig = a.0[1].as_str();
    let theta: f64 = a.0[2].parse().expect("theta");
    let lat = Lattice::eagle127();
    let (obs, mut steps, final_rx) = match fig {
        "3a" | "mz" => (
            PauliObs::magnetisation(&(0..lat.n).collect::<Vec<_>>()),
            5,
            false,
        ),
        "3b" => (PauliObs::parse(W10), 5, false),
        "3c" => (PauliObs::parse(W17), 5, false),
        "4a" => (PauliObs::parse(W17B), 5, true),
        s if s.starts_with('z') => (PauliObs::z(s[1..].parse().unwrap()), 5, false),
        _ => panic!("unknown figure {fig}"),
    };
    if let Some(s) = a.get("--steps") {
        steps = s.parse().unwrap();
    }
    // RZZ(-pi/2) as Phase(-pi/2) Phase(-pi/2) CPhase(pi): the same circuit
    // as KickedIsing::to_circuit up to a global phase (which cancels in
    // <O>), written with diagonal gates so the light cone can drop the ones
    // that commute with a Z-type observable
    let model = KickedIsing::new(lat, steps, theta);
    let n = model.lattice.n;
    let mut c = Circuit::new(n);
    for _ in 0..steps {
        for q in 0..n {
            c.rx(q, theta);
        }
        for &(x, y) in &model.lattice.edges {
            c.phase(x, -FRAC_PI_2).phase(y, -FRAC_PI_2).cphase(x, y, PI);
        }
    }
    if final_rx {
        for q in 0..n {
            c.rx(q, theta);
        }
    }
    let t0 = Instant::now();
    let mut total = 0.0;
    for (term, w) in &obs.terms {
        let p: Vec<(usize, Pauli)> = term
            .iter()
            .map(|&(q, ch)| {
                (
                    q,
                    match ch {
                        'X' => Pauli::X,
                        'Y' => Pauli::Y,
                        _ => Pauli::Z,
                    },
                )
            })
            .collect();
        let (d, cone) = tn::expectation_circuit(&c, &p).expect("doubled circuit");
        let nw = Network::amplitude(&d, &vec![false; d.num_qubits], &[]).expect("network");
        let label = format!("ising-{fig}-T{steps}-theta{theta:.4}-cone{}", cone.len());
        let v = run(&label, nw, a);
        if let Some(v) = v {
            total += w * v[0].re;
        }
    }
    if a.has("--run") {
        println!(
            "{{\"stage\":\"value\",\"fig\":\"{fig}\",\"steps\":{steps},\"theta\":{theta},\"value\":{total:.12},\"secs\":{:.3}}}",
            t0.elapsed().as_secs_f64()
        );
    }
}

/// One random instance of a calibration family.
fn build_family(fam: &str, n: usize, depth: usize, seed: u64) -> (Circuit, u128) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut c = Circuit::new(n);
    match fam {
        // 1D brickwork of random U + CZ
        "brick" => {
            for l in 0..depth {
                for q in 0..n {
                    c.u(
                        q,
                        rng.random_range(0.0..PI),
                        rng.random_range(-PI..PI),
                        rng.random_range(-PI..PI),
                    );
                }
                let mut q = l % 2;
                while q + 1 < n {
                    c.cz(q, q + 1);
                    q += 2;
                }
            }
        }
        // 2D grid (rows of 6), alternating four coupler directions, sqrt-gates + iSWAP-like
        "sycpat" => {
            let w = 6usize;
            let rows = n.div_ceil(w);
            let one = [Gate::Sx(0), Gate::H(0), Gate::T(0)];
            for l in 0..depth {
                for q in 0..n {
                    match one[rng.random_range(0..3)] {
                        Gate::Sx(_) => c.sx(q),
                        Gate::H(_) => c.h(q),
                        _ => c.ry(q, rng.random_range(-PI..PI)),
                    };
                }
                let dir = l % 4;
                for r in 0..rows {
                    for col in 0..w {
                        let q = r * w + col;
                        let (r2, c2) = match dir {
                            0 => (r + 1, col),
                            1 => (r, col + 1),
                            2 => (r + 1, col + 1),
                            _ => (r + 1, col.wrapping_sub(1)),
                        };
                        let ok = match dir {
                            0 | 2 => (r + col) % 2 == l / 4 % 2,
                            _ => (r + col + 1) % 2 == l / 4 % 2,
                        };
                        if ok && r2 < rows && c2 < w {
                            let q2 = r2 * w + c2;
                            if q2 < n {
                                c.iswapdg(q, q2).cphase(q, q2, -PI / 6.0);
                            }
                        }
                    }
                }
            }
        }
        // QAOA MaxCut on a random 3-regular-ish graph
        "qaoa" => {
            for q in 0..n {
                c.h(q);
            }
            let mut edges = Vec::new();
            for q in 0..n {
                for _ in 0..2 {
                    let r = rng.random_range(0..n);
                    if r != q && !edges.contains(&(q.min(r), q.max(r))) {
                        edges.push((q.min(r), q.max(r)));
                    }
                }
            }
            for _ in 0..depth {
                let g: f64 = rng.random_range(0.0..PI);
                for &(x, y) in &edges {
                    c.cnot(x, y).rz(y, g).cnot(x, y);
                }
                let b: f64 = rng.random_range(0.0..PI);
                for q in 0..n {
                    c.rx(q, b);
                }
            }
        }
        // random gates on random pairs
        _ => {
            for _ in 0..depth * n / 2 {
                let x = rng.random_range(0..n);
                let mut y = rng.random_range(0..n - 1);
                if y >= x {
                    y += 1;
                }
                c.u(x, rng.random(), rng.random(), rng.random());
                c.cz(x, y);
            }
        }
    }
    let x: u128 = rng.random::<u128>()
        & if n >= 128 {
            u128::MAX
        } else {
            (1u128 << n) - 1
        };
    (c, x)
}

/// Random instances for the cost model.
fn calib(a: &Args) {
    let fam = a.0[1].as_str();
    let n: usize = a.0[2].parse().unwrap();
    let depth: usize = a.0[3].parse().unwrap();
    let seed: u64 = a.0[4].parse().unwrap();
    let (c, x) = build_family(fam, n, depth, seed);
    let nw = Network::amplitude(&c, &tn::bits_of(x, n), &[]).expect("network");
    let label = format!("{fam}:n={n},d={depth},s={seed}");
    run(&label, nw, a);
}

/// State-vector evolution time on this machine against the planner's
/// M1-fitted evolution model (the machine-speed ratio kappa).
fn svcal(a: &Args) {
    let fam = a.0[1].as_str();
    let n: usize = a.0[2].parse().unwrap();
    let depth: usize = a.0[3].parse().unwrap();
    let seed: u64 = a.0[4].parse().unwrap();
    let (c, _) = build_family(fam, n, depth, seed);
    let m = qsim_lab::planner::CostModel::mac_m1().state[0];
    let sv_l = (c.num_gates().max(1) as f64).log2() + n as f64;
    let pred = (m.a + m.b * sv_l).exp2();
    let reps: usize = a.num("--reps", 3);
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        let mut sv = qsim_lab::StateVectorF64::try_new(n).expect("state");
        sv.apply_circuit_blocked(&c, &qsim_lab::engines::blocked::BlockConfig::default())
            .expect("evolve");
        best = best.min(t0.elapsed().as_secs_f64());
        std::hint::black_box(sv.amplitude(0));
    }
    println!(
        "{{\"stage\":\"svcal\",\"label\":\"{fam}:n={n},d={depth},s={seed}\",\"gates\":{},\"secs\":{best:.5},\"m1_model_secs\":{pred:.5},\"ratio\":{:.4},\"threads\":{},\"load\":\"{}\"}}",
        c.num_gates(),
        best / pred,
        rayon::current_num_threads(),
        load()
    );
}

/// End to end: the planner's amplitude request with and without the TN
/// candidate, both timed here.
fn plan_cmd(a: &Args) {
    let fam = a.0[1].as_str();
    let n: usize = a.0[2].parse().unwrap();
    let depth: usize = a.0[3].parse().unwrap();
    let seed: u64 = a.0[4].parse().unwrap();
    let (c, x) = build_family(fam, n, depth, seed);
    let mem_gb: f64 = a.num("--mem-gb", 1.0);
    for tn_on in [true, false] {
        qsim_lab::planner::clear_cache();
        let cfg = qsim_lab::planner::PlannerConfig {
            tn: tn_on,
            cache: false,
            mem_bytes: (mem_gb * (1u64 << 30) as f64) as u128,
            ..Default::default()
        };
        let t0 = Instant::now();
        let r = qsim_lab::planner::amplitudes(&c, &[x], &cfg);
        let secs = t0.elapsed().as_secs_f64();
        match r {
            Ok(r) => println!(
                "{{\"stage\":\"plan\",\"label\":\"{fam}:n={n},d={depth},s={seed}\",\"tn\":{tn_on},\"engine\":\"{}\",\"secs\":{secs:.5},\"plan_secs\":{:.5},\"aborted\":{},\"value\":[{:.12e},{:.12e}],\"load\":\"{}\"}}",
                r.engine.name(),
                r.plan_secs,
                r.aborted.len(),
                r.amplitudes[0].re,
                r.amplitudes[0].im,
                load()
            ),
            Err(e) => println!(
                "{{\"stage\":\"plan\",\"label\":\"{fam}:n={n},d={depth},s={seed}\",\"tn\":{tn_on},\"error\":\"{e}\",\"secs\":{secs:.5}}}"
            ),
        }
    }
}

fn main() {
    let a = Args(std::env::args().skip(1).collect());
    match a.0.first().map(String::as_str) {
        Some("syc") => syc(&a),
        Some("ising") => ising(&a),
        Some("calib") => calib(&a),
        Some("json") => json_cmd(&a),
        Some("svcal") => svcal(&a),
        Some("plan") => plan_cmd(&a),
        _ => eprintln!("usage: see the module docs of examples/tn_bench.rs"),
    }
}
