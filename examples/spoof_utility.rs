//! Classical reproduction of IBM's 127-qubit kicked-Ising "utility"
//! experiment (Kim et al., Nature 618, 500 (2023)) with sparse Pauli
//! dynamics (`qsim_lab::spd`), plus the same circuits on larger heavy-hex
//! lattices.
//!
//! ```text
//! cargo run --release --example spoof_utility -- <fig> [--delta D[,D2,...]] [--thetas t1,t2,...]
//!        [--steps T] [--lattice 127|433|1121] [--depol p] [--max-weight w] [--mem-gb G] [--stream k]
//! fig: 3a (M_z, 5 steps) | 3b (weight 10, 5 steps) | 3c (weight 17, 5 steps)
//!      | 4a (weight 17, 5 steps + RX) | 4b (Z_62, 20 steps) | z<q> (Z_q) | mz (M_z)
//! ```
//! One JSON object per (θ, δ) on stdout. Set RAYON_NUM_THREADS to bound cores.

use qsim_lab::spd::{light_cone_size, simulate, term_bytes, KickedIsing, Lattice, PauliObs, SpdOptions};
use std::f64::consts::PI;

const W10: &str = "X13 X29 X31 Y9 Y30 Z8 Z12 Z17 Z28 Z32";
const W17: &str = "X37 X41 X52 X56 X57 X58 X62 X79 Y75 Z38 Z40 Z42 Z63 Z72 Z80 Z90 Z91";
const W17B: &str = "X37 X41 X52 X56 X57 X58 X62 X79 Y38 Y40 Y42 Y63 Y72 Y80 Y90 Y91 Z75";

fn list(s: &str) -> Vec<f64> {
    s.split(',')
        .map(|t| {
            let t = t.trim();
            if let Some(k) = t.strip_suffix("pi/32") {
                k.parse::<f64>().unwrap() * PI / 32.0
            } else if t == "pi/2" {
                PI / 2.0
            } else {
                t.parse().unwrap()
            }
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let fig = args.first().cloned().unwrap_or_else(|| "3a".into());
    let get = |k: &str| args.iter().position(|a| a == k).map(|i| args[i + 1].clone());
    let deltas = get("--delta").map(|s| list(&s)).unwrap_or(vec![1e-4]);
    let lattice_n: usize = get("--lattice").map(|s| s.parse().unwrap()).unwrap_or(127);
    let lat = match lattice_n {
        127 => Lattice::eagle127(),
        433 => Lattice::osprey433(),
        1121 => Lattice::condor1121(),
        _ => panic!("lattice must be 127, 433 or 1121"),
    };
    let depol: f64 = get("--depol").map(|s| s.parse().unwrap()).unwrap_or(0.0);
    let max_weight: usize = get("--max-weight").map(|s| s.parse().unwrap()).unwrap_or(usize::MAX);
    let stream: usize = get("--stream").map(|s| s.parse().unwrap()).unwrap_or(1);
    let mem_gb: f64 = get("--mem-gb").map(|s| s.parse().unwrap()).unwrap_or(2.0);
    let default_thetas: Vec<f64> = (0..=16).map(|k| k as f64 * PI / 32.0).collect();
    let thetas = get("--thetas").map(|s| list(&s)).unwrap_or(default_thetas);

    let (obs, mut steps, final_rx) = match fig.as_str() {
        "3a" | "mz" => (PauliObs::magnetisation(&(0..lat.n).collect::<Vec<_>>()), 5, false),
        "3b" => (PauliObs::parse(W10), 5, false),
        "3c" => (PauliObs::parse(W17), 5, false),
        "4a" => (PauliObs::parse(W17B), 5, true),
        "4b" => (PauliObs::z(62), 20, false),
        s if s.starts_with('z') => (PauliObs::z(s[1..].parse().unwrap()), 20, false),
        _ => panic!("unknown figure {fig}"),
    };
    if let Some(s) = get("--steps") {
        steps = s.parse().unwrap();
    }
    let cone = light_cone_size(&lat, &obs.support(), steps);
    let max_terms = ((mem_gb * 1e9) / (3.0 * term_bytes(cone) as f64)) as usize;
    eprintln!(
        "fig {fig}: lattice {} qubits, {} edges, steps {steps}, final_rx {final_rx}, light cone {cone}, max_terms {max_terms}",
        lat.n,
        lat.edges.len()
    );
    for &theta in &thetas {
        for &delta in &deltas {
            let mut m = KickedIsing::new(lat.clone(), steps, theta);
            m.final_rx = final_rx;
            let r = simulate(
                &m,
                &obs,
                &SpdOptions {
                    delta,
                    max_weight,
                    depol,
                    max_terms,
                    light_cone: true,
                    stream,
                },
            );
            println!(
                "{{\"fig\":\"{fig}\",\"lattice\":{},\"steps\":{steps},\"theta\":{theta:.6},\"delta\":{delta:e},\"depol\":{depol},\"stream\":{stream},\"max_weight\":{},\"value\":{:.10},\"aborted\":{},\"cone\":{},\"words\":{},\"peak_terms\":{},\"final_terms\":{},\"terms_per_layer\":{:?},\"discarded_l1\":{:.6e},\"discarded_l2sq\":{:.6e},\"norm2\":{:.8},\"seconds\":{:.3}}}",
                lat.n,
                if max_weight == usize::MAX { -1 } else { max_weight as i64 },
                r.value,
                r.aborted,
                r.active_qubits,
                r.words,
                r.peak_terms,
                r.final_terms,
                r.terms_per_layer,
                r.discarded_l1,
                r.discarded_l2sq,
                r.norm2,
                r.seconds
            );
        }
    }
}
