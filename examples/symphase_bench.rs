//! Symbolic-phase sampler vs shot-by-shot tableau simulation on the
//! rotated surface-code memory circuit (`SurfaceCode::build_circuit`) with
//! circuit-level noise.
//!
//! cargo run --release --example symphase_bench -- [d,d,...] [p] [sym_batches] [tab_shots]
//!
//! Interleaved: each repetition times the tableau shots, then the sampler
//! batches. Prints min per-shot times and the speedup.

use qsim_lab::stabilizer::symphase::SymPhaseSampler;
use qsim_lab::{NoiseModel, SurfaceCode, Tableau};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ds: Vec<usize> = args
        .get(1)
        .map_or("3,5,7,11", String::as_str)
        .split(',')
        .map(|x| x.parse().unwrap())
        .collect();
    let p: f64 = args.get(2).map_or(0.001, |s| s.parse().unwrap());
    let batches: usize = args.get(3).map_or(200, |s| s.parse().unwrap());
    let tab_shots: usize = args.get(4).map_or(100, |s| s.parse().unwrap());
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let noise = NoiseModel::circuit_level(p, p);
    println!("p = {p}, rounds = d, sampler shots = 64 x {batches}, tableau shots = {tab_shots}\n");
    println!(
        "| d | qubits | measurements | vars | nnz(A) | compile (ms) | tableau (us/shot) | sampler (us/shot) | speedup/shot | mean ones/shot tab, sym | detector nnz | detector sampler (us/shot) |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|");
    for &d in &ds {
        let code = SurfaceCode::new(d, d);
        let c = code.build_circuit();
        let mut compile = f64::INFINITY;
        let mut s = None;
        for _ in 0..reps {
            let t = Instant::now();
            s = Some(SymPhaseSampler::new(&c, &noise).unwrap());
            compile = compile.min(t.elapsed().as_secs_f64());
        }
        let s = s.unwrap();
        let m = s.num_measurements();
        // detector-like parities: each ancilla against the same ancilla one
        // round earlier; data measurements as they are
        let na = SurfaceCode::num_ancillas(d);
        let sets: Vec<Vec<usize>> = (0..m)
            .map(|j| {
                if (na..d * na).contains(&j) {
                    vec![j - na, j]
                } else {
                    vec![j]
                }
            })
            .collect();
        let ds = s.with_parities(&sets);
        let mut dvals = vec![0u64; ds.num_vars()];
        let (mut t_tab, mut t_sym, mut t_det) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let (mut ones_tab, mut ones_sym) = (0usize, 0usize);
        let mut rng = StdRng::seed_from_u64(1);
        let mut vals = vec![0u64; s.num_vars()];
        let mut out = vec![0u64; m];
        for _ in 0..reps {
            let t = Instant::now();
            ones_tab = 0;
            for _ in 0..tab_shots {
                let mut tab = Tableau::new(c.num_qubits);
                let r = c.run_noisy(&mut tab, &noise, &mut rng).unwrap();
                ones_tab += r.iter().filter(|&&b| b).count();
            }
            t_tab = t_tab.min(t.elapsed().as_secs_f64() / tab_shots as f64);
            let t = Instant::now();
            ones_sym = 0;
            for _ in 0..batches {
                s.sample_batch(&mut rng, &mut vals, &mut out);
                ones_sym += out.iter().map(|w| w.count_ones() as usize).sum::<usize>();
            }
            t_sym = t_sym.min(t.elapsed().as_secs_f64() / (64 * batches) as f64);
            let t = Instant::now();
            for _ in 0..batches {
                ds.sample_batch(&mut rng, &mut dvals, &mut out);
            }
            t_det = t_det.min(t.elapsed().as_secs_f64() / (64 * batches) as f64);
        }
        println!(
            "| {d} | {} | {m} | {} | {} | {:.1} | {:.1} | {:.3} | {:.0}x | {:.1}, {:.1} | {} | {:.3} |",
            c.num_qubits,
            s.num_vars(),
            s.nnz(),
            1e3 * compile,
            1e6 * t_tab,
            1e6 * t_sym,
            t_tab / t_sym,
            ones_tab as f64 / tab_shots as f64,
            ones_sym as f64 / (64 * batches) as f64,
            ds.nnz(),
            1e6 * t_det,
        );
    }
}
