//! `qsim`: run examples and benchmarks from the command line.

use clap::{Parser, Subcommand, ValueEnum};
use qsim_lab::algorithms;
use qsim_lab::bench;
use qsim_lab::circuit::{Circuit, Simulator};
use qsim_lab::{Mps, StateVectorF64, Tableau};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::BTreeMap;

#[derive(Parser)]
#[command(
    name = "qsim",
    about = "Educational quantum circuit simulator",
    version
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run timing benchmarks and print markdown tables.
    Bench {
        #[command(subcommand)]
        which: BenchCmd,
    },
    /// Run an example circuit and print its measurement statistics.
    Run {
        example: Example,
        /// Number of qubits (where the example takes one).
        #[arg(short, long)]
        qubits: Option<usize>,
        /// Number of shots.
        #[arg(short, long, default_value_t = 1000)]
        shots: usize,
        /// Backend for circuit examples (bell, ghz, bv).
        #[arg(short, long, value_enum, default_value_t = Backend::Sv)]
        backend: Backend,
        /// Number to factor (shor example).
        #[arg(long, default_value_t = 15)]
        modulus: u64,
        /// RNG seed.
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
}

#[derive(Subcommand)]
enum BenchCmd {
    /// GHZ on the state vector, f32 and f64.
    Sv {
        #[arg(long, default_value_t = 16)]
        min: usize,
        /// Capped at 26 (f32) / 25 (f64) by the memory limit.
        #[arg(long, default_value_t = 26)]
        max: usize,
    },
    /// QFT on the state vector.
    Qft {
        #[arg(long, default_value_t = 16)]
        min: usize,
        #[arg(long, default_value_t = 26)]
        max: usize,
    },
    /// GHZ + measure-all on the stabilizer tableau.
    Stab {
        /// Comma-separated qubit counts.
        #[arg(long, value_delimiter = ',', default_values_t = [1000, 2000, 5000, 10000, 20000, 30000, 40000, 46336])]
        sizes: Vec<usize>,
    },
    /// Cost of Clifford+T circuits vs number of T gates (Pauli paths).
    CliffordT {
        #[arg(long, default_value_t = 64)]
        qubits: usize,
        #[arg(long, default_value_t = 3)]
        depth: usize,
        #[arg(long, default_value_t = 40)]
        max_t: usize,
        #[arg(long, default_value_t = 2)]
        step: usize,
        #[arg(long, default_value_t = 1 << 22)]
        max_terms: usize,
    },
    /// Hybrid Schrödinger–Feynman vs the f64 state vector, as the number of
    /// gates crossing the cut grows.
    HsfCrossover {
        #[arg(long, default_value_t = 20)]
        n: usize,
        #[arg(long, value_delimiter = ',', default_values_t = [0, 2, 4, 6, 8, 10, 12, 14])]
        ks: Vec<usize>,
        #[arg(long, default_value_t = 8)]
        depth: usize,
        #[arg(long, default_value_t = 1000)]
        amps: usize,
        #[arg(long, default_value_t = 3)]
        reps: usize,
        /// Skip the full-output measurement.
        #[arg(long)]
        no_full: bool,
    },
    /// HSF amplitude batches beyond the state vector's memory cap.
    HsfBig {
        #[arg(long, value_delimiter = ',', default_values_t = [32, 36, 40])]
        ns: Vec<usize>,
        #[arg(long, value_delimiter = ',', default_values_t = [4, 8])]
        ks: Vec<usize>,
        #[arg(long, default_value_t = 6)]
        depth: usize,
        #[arg(long, default_value_t = 64)]
        amps: usize,
        /// Put all crossing gates in the middle layer instead of spreading them.
        #[arg(long)]
        middle: bool,
    },
    /// A/B of HSF design choices on one circuit.
    HsfAblation {
        #[arg(long, default_value_t = 22)]
        n: usize,
        #[arg(long, default_value_t = 8)]
        k: usize,
        #[arg(long, default_value_t = 8)]
        depth: usize,
        #[arg(long, default_value_t = 1000)]
        amps: usize,
        #[arg(long, default_value_t = 3)]
        reps: usize,
        #[arg(long)]
        middle: bool,
    },
    /// MPS: GHZ at large n, then random circuits.
    Mps {
        #[arg(long, default_value_t = 24)]
        random_qubits: usize,
        #[arg(long, default_value_t = 256)]
        max_bond: usize,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum Example {
    Bell,
    Ghz,
    Bv,
    Grover,
    Qft,
    Shor,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Backend {
    Sv,
    Stab,
    Mps,
}

fn histogram(c: &Circuit, backend: Backend, shots: usize, rng: &mut StdRng) {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for _ in 0..shots {
        let mut sim: Box<dyn Simulator> = match backend {
            Backend::Sv => Box::new(StateVectorF64::new(c.num_qubits)),
            Backend::Stab => Box::new(Tableau::new(c.num_qubits)),
            Backend::Mps => Box::new(Mps::new(c.num_qubits, 64)),
        };
        let bits = c.run(sim.as_mut(), rng).expect("circuit runs");
        // print with the last-measured bit leftmost, like |q_{n-1} ... q_0>
        let s: String = bits
            .iter()
            .rev()
            .map(|&b| if b { '1' } else { '0' })
            .collect();
        *counts.entry(s).or_default() += 1;
    }
    for (k, v) in counts {
        println!("  {k}: {v}");
    }
}

fn main() {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Bench { which } => match which {
            BenchCmd::Sv { min, max } => {
                println!("## GHZ, state vector, single precision\n");
                bench::sv_ghz::<f32>(&(min..=max.min(26)).collect::<Vec<_>>(), "f32");
                println!("\n## GHZ, state vector, double precision\n");
                bench::sv_ghz::<f64>(&(min..=max.min(25)).collect::<Vec<_>>(), "f64");
            }
            BenchCmd::Qft { min, max } => {
                println!("## QFT, state vector, single precision\n");
                bench::sv_qft::<f32>(&(min..=max.min(26)).step_by(2).collect::<Vec<_>>(), "f32");
            }
            BenchCmd::Stab { sizes } => {
                println!("## GHZ, stabilizer tableau\n");
                bench::stab_ghz(&sizes);
            }
            BenchCmd::CliffordT {
                qubits,
                depth,
                max_t,
                step,
                max_terms,
            } => {
                println!("## Clifford+T, Pauli-path summation\n");
                let ts: Vec<usize> = (0..=max_t).step_by(step.max(1)).collect();
                bench::clifford_t(qubits, depth, &ts, max_terms);
            }
            BenchCmd::HsfCrossover {
                n,
                ks,
                depth,
                amps,
                reps,
                no_full,
            } => {
                println!("## HSF vs state vector, n = {n}, depth {depth}\n");
                bench::hsf_crossover(n, &ks, depth, amps, reps, !no_full);
            }
            BenchCmd::HsfBig {
                ns,
                ks,
                depth,
                amps,
                middle,
            } => {
                println!("## HSF beyond the state vector, depth {depth}, middle = {middle}\n");
                bench::hsf_big(&ns, &ks, depth, amps, middle);
            }
            BenchCmd::HsfAblation {
                n,
                k,
                depth,
                amps,
                reps,
                middle,
            } => {
                println!("## HSF ablation, n = {n}, k = {k}, depth {depth}, middle = {middle}\n");
                bench::hsf_ablation(n, k, depth, amps, reps, middle);
            }
            BenchCmd::Mps {
                random_qubits,
                max_bond,
            } => {
                println!("## GHZ, MPS\n");
                bench::mps_ghz(&[100, 1000, 10000]);
                println!("\n## Random brickwork circuits, MPS\n");
                bench::mps_random(random_qubits, max_bond, &[2, 4, 8, 12, 16, 20, 24, 32]);
                println!("\n## Random brickwork circuits, MPS with a small bond cap\n");
                bench::mps_random(60, 32, &[4, 8, 16, 32]);
            }
        },
        Cmd::Run {
            example,
            qubits,
            shots,
            backend,
            modulus,
            seed,
        } => {
            let mut rng = StdRng::seed_from_u64(seed);
            match example {
                Example::Bell => {
                    let mut c = algorithms::bell();
                    c.measure_all();
                    println!("Bell state, {shots} shots:");
                    histogram(&c, backend, shots, &mut rng);
                }
                Example::Ghz => {
                    let n = qubits.unwrap_or(5);
                    let mut c = algorithms::ghz(n);
                    c.measure_all();
                    println!("{n}-qubit GHZ state, {shots} shots:");
                    histogram(&c, backend, shots, &mut rng);
                }
                Example::Bv => {
                    let n = qubits.unwrap_or(8);
                    let secret = 0b1011_0110_1101_0011u64 & ((1u64 << n) - 1);
                    let c = algorithms::bernstein_vazirani(n, secret);
                    println!("Bernstein-Vazirani, secret {secret:0n$b}, {shots} shots:");
                    histogram(&c, backend, shots.min(100), &mut rng);
                }
                Example::Grover => {
                    let n = qubits.unwrap_or(10);
                    let marked = 0x2A5 & ((1 << n) - 1);
                    let (found, p) = algorithms::grover::<f64, _>(n, marked, &mut rng);
                    println!(
                        "Grover over 2^{n} items, marked {marked}: measured {found}, \
                         success probability {p:.4}"
                    );
                }
                Example::Qft => {
                    let n = qubits.unwrap_or(4);
                    let x = 5 % (1 << n);
                    let mut s = StateVectorF64::basis_state(n, x);
                    s.apply_circuit(&algorithms::qft(n)).expect("valid");
                    println!("QFT|{x}> on {n} qubits (amplitude = 2^(-n/2) e^(2πi·{x}k/2^n)):");
                    for k in 0..(1usize << n).min(16) {
                        let a = s.amplitude(k);
                        println!(
                            "  k={k:2}: |a|={:.4}  phase/2π={:.4}",
                            a.norm(),
                            (a.arg() / (2.0 * std::f64::consts::PI)).rem_euclid(1.0)
                        );
                    }
                }
                Example::Shor => {
                    let n = modulus;
                    let (f, runs) = algorithms::shor_factor(n, &mut rng);
                    for r in &runs {
                        println!(
                            "a={:2}  qubits={}  measured={:4}  order={:?}  factor={:?}",
                            r.a, r.qubits, r.measured, r.order, r.factor
                        );
                    }
                    match f {
                        Some((p, q)) => println!("{n} = {p} x {q}"),
                        None => println!("no factor found"),
                    }
                }
            }
        }
    }
}
