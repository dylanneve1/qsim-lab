//! Stabilizer benchmarks for the `exp/stab` experiments.
//!
//! cargo run --release --example stab_bench -- <workload> <impl> [args]
//!
//! impl: `ref` (frozen original tableau) or `new` (current `Tableau`).
//! Every timing is the minimum over `REPS` runs.

use qsim_lab::circuit::{Circuit, Op, Simulator};
use qsim_lab::stabilizer::reference::RefTableau;
use qsim_lab::{Gate, Tableau};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn reps() -> usize {
    std::env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5)
}

fn min_time<F: FnMut() -> T, T>(mut f: F) -> (f64, T) {
    let mut best = f64::INFINITY;
    let mut out = None;
    for _ in 0..reps() {
        let t = Instant::now();
        let r = f();
        best = best.min(t.elapsed().as_secs_f64());
        out = Some(r);
    }
    (best, out.unwrap())
}

/// Common surface for the two implementations.
trait Tab: Simulator {
    fn new_tab(n: usize) -> Self;
    fn h_(&mut self, q: usize);
    fn cnot_(&mut self, a: usize, b: usize);
    fn measure_all_(&mut self, rng: &mut StdRng) -> Vec<bool>;
    fn reset_(&mut self, q: usize, rng: &mut StdRng) -> bool {
        let m = self.measure(q, rng).unwrap();
        if m {
            self.apply(&Gate::X(q)).unwrap();
        }
        m
    }
    fn stats(&self) -> String {
        String::new()
    }
}

impl Tab for RefTableau {
    fn new_tab(n: usize) -> Self {
        RefTableau::new(n)
    }
    fn h_(&mut self, q: usize) {
        self.h(q)
    }
    fn cnot_(&mut self, a: usize, b: usize) {
        self.cnot(a, b)
    }
    fn measure_all_(&mut self, rng: &mut StdRng) -> Vec<bool> {
        self.measure_all(rng)
    }
}

impl Tab for Tableau {
    fn new_tab(n: usize) -> Self {
        Tableau::new(n)
    }
    fn h_(&mut self, q: usize) {
        self.h(q)
    }
    fn cnot_(&mut self, a: usize, b: usize) {
        self.cnot(a, b)
    }
    fn measure_all_(&mut self, rng: &mut StdRng) -> Vec<bool> {
        self.measure_all(rng)
    }
    fn reset_(&mut self, q: usize, rng: &mut StdRng) -> bool {
        self.reset_qubit(q, rng)
    }
    fn stats(&self) -> String {
        format!("switches={}", self.layout_switches())
    }
}

fn ghz<T: Tab>(sizes: &[usize]) {
    println!("| n | prepare (s) | measure all (s) | total (s) |");
    println!("|---|---|---|---|");
    for &n in sizes {
        let mut rng = StdRng::seed_from_u64(2);
        let mut best = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
        for _ in 0..reps() {
            let t0 = Instant::now();
            let mut t = T::new_tab(n);
            t.h_(0);
            for q in 1..n {
                t.cnot_(q - 1, q);
            }
            let tp = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let bits = t.measure_all_(&mut rng);
            let tm = t1.elapsed().as_secs_f64();
            let tt = t0.elapsed().as_secs_f64();
            assert!(bits.iter().all(|&b| b == bits[0]));
            drop(t);
            best = (best.0.min(tp), best.1.min(tm), best.2.min(tt));
        }
        println!("| {n} | {:.4} | {:.4} | {:.4} |", best.0, best.1, best.2);
    }
}

/// Random Clifford layers; after each layer, `m` random qubits are measured.
fn mixed_circuit(n: usize, depth: usize, m: usize, seed: u64) -> Circuit {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut c = Circuit::new(n);
    for _ in 0..depth {
        c.append(&Circuit::random_clifford(n, 1, &mut rng));
        for _ in 0..m {
            c.measure(rng.random_range(0..n));
        }
    }
    c
}

fn run_circuit<T: Tab>(c: &Circuit, seed: u64) -> (Vec<bool>, String) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut t = T::new_tab(c.num_qubits);
    let out = c.run(&mut t, &mut rng).unwrap();
    (out, t.stats())
}

fn mixed<T: Tab>(sizes: &[usize], depth: usize, frac: f64) {
    println!("| n | depth | measurements/layer | time (s) | outcome hash | stats |");
    println!("|---|---|---|---|---|---|");
    for &n in sizes {
        let m = ((n as f64 * frac).ceil() as usize).max(1);
        let c = mixed_circuit(n, depth, m, 7);
        let (t, (out, stats)) = min_time(|| run_circuit::<T>(&c, 11));
        println!(
            "| {n} | {depth} | {m} | {t:.4} | {:016x} | {stats} |",
            hash(&out)
        );
    }
}

fn hash(bits: &[bool]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bits {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    h
}

/// Rotated surface code of distance `d` (d odd): `d*d` data qubits on a
/// grid, `d*d - 1` ancillas. Returns (num_qubits, x_ancillas, z_ancillas)
/// where each ancilla lists its data neighbours in CNOT order (None = no
/// neighbour on that corner).
pub struct Surface {
    pub n: usize,
    pub x_anc: Vec<(usize, [Option<usize>; 4])>,
    pub z_anc: Vec<(usize, [Option<usize>; 4])>,
}

pub fn surface(d: usize) -> Surface {
    let data = |i: isize, j: isize| -> Option<usize> {
        (i >= 0 && j >= 0 && (i as usize) < d && (j as usize) < d)
            .then(|| i as usize * d + j as usize)
    };
    let mut next = d * d;
    let (mut x_anc, mut z_anc) = (vec![], vec![]);
    for i in 0..=d {
        for j in 0..=d {
            let is_x = (i + j) % 2 == 0;
            let bulk = (1..d).contains(&i) && (1..d).contains(&j);
            let tb = (i == 0 || i == d) && (1..d).contains(&j);
            let lr = (j == 0 || j == d) && (1..d).contains(&i);
            if !(bulk || (tb && is_x) || (lr && !is_x)) {
                continue;
            }
            let (ii, jj) = (i as isize, j as isize);
            let nw = data(ii - 1, jj - 1);
            let ne = data(ii - 1, jj);
            let sw = data(ii, jj - 1);
            let se = data(ii, jj);
            let a = next;
            next += 1;
            if is_x {
                x_anc.push((a, [nw, ne, sw, se]));
            } else {
                z_anc.push((a, [nw, sw, ne, se]));
            }
        }
    }
    Surface {
        n: next,
        x_anc,
        z_anc,
    }
}

fn syndrome_round<T: Tab>(t: &mut T, s: &Surface, rng: &mut StdRng, out: &mut Vec<bool>) {
    for &(a, _) in &s.x_anc {
        t.h_(a);
    }
    for k in 0..4 {
        for &(a, nb) in &s.x_anc {
            if let Some(q) = nb[k] {
                t.cnot_(a, q);
            }
        }
        for &(a, nb) in &s.z_anc {
            if let Some(q) = nb[k] {
                t.cnot_(q, a);
            }
        }
    }
    for &(a, _) in &s.x_anc {
        t.h_(a);
    }
    for &(a, _) in s.x_anc.iter().chain(&s.z_anc) {
        out.push(t.reset_(a, rng));
    }
}

fn syndrome<T: Tab>(ds: &[usize], rounds: usize) {
    println!("| d | qubits | rounds | time (s) | per round (ms) | outcome hash | stats |");
    println!("|---|---|---|---|---|---|---|");
    for &d in ds {
        let s = surface(d);
        let (t, (out, stats)) = min_time(|| {
            let mut rng = StdRng::seed_from_u64(5);
            let mut t = T::new_tab(s.n);
            let mut out = vec![];
            for _ in 0..rounds {
                syndrome_round(&mut t, &s, &mut rng, &mut out);
            }
            (out, t.stats())
        });
        // noiseless: every round after the first repeats the first
        let per = s.x_anc.len() + s.z_anc.len();
        for r in 1..rounds {
            assert_eq!(out[r * per..(r + 1) * per], out[..per]);
        }
        println!(
            "| {d} | {} | {rounds} | {t:.4} | {:.3} | {:016x} | {stats} |",
            s.n,
            1e3 * t / rounds as f64,
            hash(&out)
        );
    }
}

fn parse_list(s: &str) -> Vec<usize> {
    s.split(',').map(|x| x.parse().unwrap()).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let w = args.get(1).map(String::as_str).unwrap_or("ghz");
    let imp = args.get(2).map(String::as_str).unwrap_or("new");
    let arg = |i: usize, d: &str| args.get(i).cloned().unwrap_or_else(|| d.to_string());
    macro_rules! dispatch {
        ($f:ident ( $($a:expr),* )) => {
            match imp {
                "ref" => $f::<RefTableau>($($a),*),
                _ => $f::<Tableau>($($a),*),
            }
        };
    }
    println!("## {w} ({imp})\n");
    match w {
        "ghz" => dispatch!(ghz(&parse_list(&arg(3, "1000,5000,10000,20000,46336")))),
        "mixed" => dispatch!(mixed(
            &parse_list(&arg(3, "100,500,1000,2000")),
            arg(4, "50").parse().unwrap(),
            arg(5, "0.05").parse().unwrap()
        )),
        "syndrome" => dispatch!(syndrome(
            &parse_list(&arg(3, "5,11,21,31")),
            arg(4, "10").parse().unwrap()
        )),
        _ => panic!("unknown workload {w}"),
    }
    let _ = Op::Measure(0);
}
