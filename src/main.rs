//! `qsim`: run examples and benchmarks from the command line.

use clap::{Parser, Subcommand, ValueEnum};
use qsim_lab::algorithms;
use qsim_lab::bench;
use qsim_lab::bench::adaptive as adaptive_bench;
use qsim_lab::circuit::{Circuit, Simulator};
use qsim_lab::shor;
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
    /// Adaptive tableau -> Pauli frame -> compressed state vector
    /// experiments (one method per invocation, for interleaved A/B runs).
    Adaptive {
        #[command(subcommand)]
        which: AdaptiveCmd,
    },
    /// Export a circuit as an ONNX graph for viewing in Netron
    /// (https://netron.app): one node per operation, one wire tensor per
    /// qubit segment, measurement records and detectors as classical tensors.
    Export {
        /// Output file (conventionally `.onnx`).
        #[arg(short, long)]
        out: std::path::PathBuf,
        /// Read an OpenQASM 2.0 file.
        #[arg(long, conflicts_with_all = ["stim", "example"])]
        qasm: Option<std::path::PathBuf>,
        /// Read a `.stim` file (its detectors and observables become nodes).
        #[arg(long, conflicts_with_all = ["qasm", "example"])]
        stim: Option<std::path::PathBuf>,
        /// Export a built-in circuit instead of a file.
        #[arg(long, value_enum)]
        example: Option<ExportExample>,
        /// Number of qubits (bell/ghz/bv/qft/brickwork).
        #[arg(short, long)]
        qubits: Option<usize>,
        /// Brickwork depth.
        #[arg(long, default_value_t = 4)]
        depth: usize,
        /// Modulus N (shor examples).
        #[arg(long, default_value_t = 15)]
        modulus: u64,
        /// Base a (shor examples; default: the smallest a >= 2 coprime to N).
        #[arg(long)]
        base: Option<u64>,
        /// Code distance (surface / repetition examples).
        #[arg(long, default_value_t = 3)]
        distance: usize,
        /// Syndrome rounds (surface / repetition examples; default: distance).
        #[arg(long)]
        rounds: Option<usize>,
        /// Export only the first N operations (large circuits are slow to lay out in viewers).
        #[arg(long)]
        max_ops: Option<usize>,
        /// RNG seed (brickwork, bv).
        #[arg(long, default_value_t = 1)]
        seed: u64,
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
        /// shor: one recycled control qubit (semiclassical QFT) instead of
        /// 2n counting qubits.
        #[arg(long)]
        semiclassical: bool,
        /// shor (semiclassical): exact sparse state instead of dense.
        #[arg(long)]
        sparse: bool,
        /// shor (semiclassical, permutation oracle): fused rounds that keep
        /// only the work register (control qubit handled analytically).
        #[arg(long)]
        fused: bool,
        /// shor (semiclassical, dense): f32 amplitudes.
        #[arg(long)]
        f32: bool,
        /// shor (semiclassical): modular-multiplication oracle.
        #[arg(long, value_enum, default_value_t = OracleArg::Permutation)]
        oracle: OracleArg,
        /// shor: base `a` (default: random bases until a factor is found).
        #[arg(long)]
        base: Option<u64>,
        /// shor (gate-level oracle, dense): one apply_gate pass per gate
        /// instead of the cache-blocked executor.
        #[arg(long)]
        no_blocked: bool,
        /// shor (ripple oracle): evaluate gates one by one instead of reversible block evaluation.
        #[arg(long)]
        gate_by_gate: bool,
        /// shor: maximum number of order-finding runs.
        #[arg(long, default_value_t = 20)]
        tries: usize,
        /// shor (ripple / windowed oracle): bit-sliced branch tracking of the
        /// gate-level circuit (exact; `--f32` for f32 amplitudes).
        #[arg(long)]
        sliced: bool,
        /// shor (windowed oracle): lookup window size in bits.
        #[arg(long, default_value_t = 4)]
        window: usize,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum OracleArg {
    Permutation,
    Beauregard,
    Ripple,
    Windowed,
    /// windowed oracle with the superoptimised blocks (exp/superopt)
    WindowedOpt,
    /// windowed-opt with measurement-based uncomputation: temporary-AND
    /// lookups, measurement-based unlookup, Gidney adders (exp/mbu-shor)
    WindowedMbu,
    /// windowed-opt with measurement-based lookups/unlookups only
    WindowedMbuLookup,
}

/// Peak resident set size of this process in MiB (Linux `VmHWM`).
fn peak_rss_mib() -> Option<f64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = s.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024.0)
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
        #[arg(long, default_value_t = 0)]
        min_t: usize,
        /// legacy | frame (variants: frame-noprune, frame-nomerge, frame-serial)
        #[arg(long, default_value = "frame")]
        engine: String,
        /// Stop after the first circuit slower than this (seconds).
        #[arg(long, default_value_t = f64::INFINITY)]
        time_limit: f64,
        /// Report the minimum time over this many runs.
        #[arg(long, default_value_t = 1)]
        repeat: usize,
        /// z0 (<Z_0>, exactly 0 on these circuits) | stab (a stabilizer of
        /// the Clifford skeleton, generically non-zero)
        #[arg(long, default_value = "z0")]
        observable: String,
    },
    /// Pauli paths on a Cuccaro ripple-carry adder (structured Toffoli
    /// circuit) of growing width.
    Adder {
        #[arg(long, value_delimiter = ',', default_values_t = [2, 4, 8, 16, 32, 64])]
        bits: Vec<usize>,
        #[arg(long, default_value_t = 1 << 22)]
        max_terms: usize,
        #[arg(long, default_value = "frame")]
        engine: String,
        #[arg(long, default_value_t = 1)]
        repeat: usize,
        #[arg(long, default_value_t = 30.0)]
        time_limit: f64,
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
    /// One HSF or state-vector run per process (for peak RSS).
    HsfPoint {
        #[arg(long)]
        n: usize,
        #[arg(long)]
        k: usize,
        #[arg(long, default_value_t = 8)]
        depth: usize,
        #[arg(long, default_value_t = 1000)]
        amps: usize,
        /// sv, full or amps
        #[arg(long)]
        mode: String,
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

#[derive(Subcommand)]
enum AdaptiveCmd {
    /// Exact <O> by one method: sv | legacy | frame | dense | auto | switch:K
    /// (comma-separated list allowed; `sweep` = switch:0..=t step --step).
    Expect {
        /// random | two-phase
        #[arg(long, default_value = "random")]
        family: String,
        #[arg(long, default_value_t = 50)]
        qubits: usize,
        /// T gates (random) or core T gates (two-phase); comma-separated.
        #[arg(long, value_delimiter = ',', default_values_t = [30])]
        t: Vec<usize>,
        #[arg(long, default_value_t = 20)]
        core: usize,
        #[arg(long, default_value_t = 20)]
        t_tail: usize,
        #[arg(long, default_value_t = 3)]
        depth: usize,
        #[arg(long, default_value = "auto")]
        methods: String,
        #[arg(long, default_value = "stab")]
        observable: String,
        #[arg(long, default_value_t = 1 << 22)]
        max_terms: usize,
        #[arg(long, default_value_t = 26)]
        max_dense: usize,
        #[arg(long, default_value_t = 1)]
        repeat: usize,
        #[arg(long, default_value_t = 4)]
        step: usize,
        /// `sweep` stops after a switch point slower than this (seconds).
        #[arg(long, default_value_t = 10.0)]
        time_limit: f64,
        /// Print the table header.
        #[arg(long)]
        header: bool,
    },
    /// Sampling all qubits: sv | compressed.
    Sample {
        #[arg(long, default_value = "random")]
        family: String,
        #[arg(long, default_value_t = 50)]
        qubits: usize,
        #[arg(long, value_delimiter = ',', default_values_t = [10, 20])]
        t: Vec<usize>,
        #[arg(long, default_value_t = 20)]
        core: usize,
        #[arg(long, default_value_t = 0)]
        t_tail: usize,
        #[arg(long, default_value_t = 3)]
        depth: usize,
        #[arg(long, default_value = "compressed")]
        methods: String,
        #[arg(long, default_value_t = 100_000)]
        shots: usize,
        #[arg(long, default_value_t = 26)]
        max_dense: usize,
        #[arg(long)]
        header: bool,
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

/// Built-in circuits that `qsim export --example` can write.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ExportExample {
    Bell,
    Ghz,
    Bv,
    Qft,
    Brickwork,
    /// Semiclassical Shor with Beauregard's gate-level oracle (one recycled control).
    ShorSemiclassical,
    /// Semiclassical Shor with the Cuccaro ripple-carry oracle (X/CNOT/Toffoli).
    ShorRipple,
    /// Rotated surface-code memory circuit (`--distance`, `--rounds`).
    Surface,
    /// Repetition-code memory circuit (`--distance`, `--rounds`).
    Repetition,
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
                min_t,
                engine,
                time_limit,
                repeat,
                observable,
            } => {
                println!("## Clifford+T, Pauli-path summation ({engine}, {observable})\n");
                let ts: Vec<usize> = (min_t..=max_t).step_by(step.max(1)).collect();
                bench::clifford_t_with(
                    qubits,
                    depth,
                    &ts,
                    max_terms,
                    &engine,
                    time_limit,
                    repeat,
                    &observable,
                );
            }
            BenchCmd::Adder {
                bits,
                max_terms,
                engine,
                repeat,
                time_limit,
            } => {
                println!("## Cuccaro adder, Pauli-path summation ({engine})\n");
                bench::adder(&bits, max_terms, &engine, repeat, time_limit);
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
            BenchCmd::HsfPoint {
                n,
                k,
                depth,
                amps,
                mode,
            } => bench::hsf_point(n, k, depth, amps, &mode),
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
        Cmd::Adaptive { which } => match which {
            AdaptiveCmd::Expect {
                family,
                qubits,
                t,
                core,
                t_tail,
                depth,
                methods,
                observable,
                max_terms,
                max_dense,
                repeat,
                step,
                time_limit,
                header,
            } => {
                if header {
                    adaptive_bench::expect_header();
                }
                for &tt in &t {
                    let c = adaptive_bench::family(&family, qubits, tt, core, t_tail, depth);
                    let obs = adaptive_bench::observable(&c, &observable);
                    let label = format!("{family} n={qubits} t={tt}");
                    for m in methods.split(',') {
                        if m == "sweep" {
                            // From the end of the circuit backwards; stops
                            // once a switch point is slower than --time-limit.
                            let tot = c.t_count();
                            for k in (0..=tot).rev().step_by(step.max(1)) {
                                let dt = adaptive_bench::expect_row(
                                    &c,
                                    &label,
                                    &obs,
                                    &format!("switch:{k}"),
                                    max_terms,
                                    max_dense,
                                    repeat,
                                );
                                if dt > time_limit {
                                    break;
                                }
                            }
                        } else {
                            adaptive_bench::expect_row(
                                &c, &label, &obs, m, max_terms, max_dense, repeat,
                            );
                        }
                    }
                }
            }
            AdaptiveCmd::Sample {
                family,
                qubits,
                t,
                core,
                t_tail,
                depth,
                methods,
                shots,
                max_dense,
                header,
            } => {
                if header {
                    adaptive_bench::sample_header();
                }
                for &tt in &t {
                    let c = adaptive_bench::family(&family, qubits, tt, core, t_tail, depth);
                    let label = format!("{family} n={qubits} t={tt}");
                    for m in methods.split(',') {
                        adaptive_bench::sample_row(&c, &label, shots, m, max_dense);
                    }
                }
            }
        },
        Cmd::Export {
            out,
            qasm,
            stim,
            example,
            qubits,
            depth,
            modulus,
            base,
            distance,
            rounds,
            max_ops,
            seed,
        } => {
            use qsim_lab::io::onnx;
            let mut opts = onnx::OnnxOptions {
                max_ops,
                ..Default::default()
            };
            let circuit = if let Some(path) = qasm {
                let src = std::fs::read_to_string(&path).unwrap_or_else(|e| {
                    eprintln!("cannot read {}: {e}", path.display());
                    std::process::exit(2)
                });
                opts.name = path.file_stem().map(|s| s.to_string_lossy().into_owned());
                opts.metadata
                    .push(("source".into(), path.display().to_string()));
                qsim_lab::io::qasm::from_qasm(&src).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(2)
                })
            } else if let Some(path) = stim {
                let src = std::fs::read_to_string(&path).unwrap_or_else(|e| {
                    eprintln!("cannot read {}: {e}", path.display());
                    std::process::exit(2)
                });
                let prog = qsim_lab::io::stim::parse_stim(&src).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(2)
                });
                opts.name = path.file_stem().map(|s| s.to_string_lossy().into_owned());
                opts.metadata
                    .push(("source".into(), path.display().to_string()));
                opts.detectors = prog.detectors;
                opts.observables = prog.observables;
                prog.circuit
            } else {
                let Some(ex) = example else {
                    eprintln!("give one of --qasm, --stim or --example");
                    std::process::exit(2)
                };
                let mut rng = StdRng::seed_from_u64(seed);
                let a = base.unwrap_or_else(|| {
                    (2..modulus)
                        .find(|&a| algorithms::gcd(a, modulus) == 1)
                        .unwrap_or(2)
                });
                let rounds = rounds.unwrap_or(distance);
                let (name, c) = match ex {
                    ExportExample::Bell => ("bell".to_string(), algorithms::bell()),
                    ExportExample::Ghz => {
                        let n = qubits.unwrap_or(5);
                        (format!("ghz_{n}"), algorithms::ghz(n))
                    }
                    ExportExample::Bv => {
                        let n = qubits.unwrap_or(6);
                        let secret = rand::Rng::random::<u64>(&mut rng) & ((1u64 << n) - 1);
                        (
                            format!("bv_{n}"),
                            algorithms::bernstein_vazirani(n, secret),
                        )
                    }
                    ExportExample::Qft => {
                        let n = qubits.unwrap_or(5);
                        (format!("qft_{n}"), algorithms::qft(n))
                    }
                    ExportExample::Brickwork => {
                        let n = qubits.unwrap_or(6);
                        (
                            format!("brickwork_{n}x{depth}"),
                            algorithms::random_brickwork(n, depth, &mut rng),
                        )
                    }
                    ExportExample::ShorSemiclassical => (
                        format!("shor_semiclassical_N{modulus}_a{a}"),
                        shor::semiclassical_circuit(modulus, a),
                    ),
                    ExportExample::ShorRipple => (
                        format!("shor_ripple_N{modulus}_a{a}"),
                        shor::semiclassical_ripple_circuit(modulus, a),
                    ),
                    ExportExample::Surface => (
                        format!("surface_d{distance}_r{rounds}"),
                        qsim_lab::qec::surface::SurfaceCode::new(distance, rounds).build_circuit(),
                    ),
                    ExportExample::Repetition => (
                        format!("repetition_d{distance}_r{rounds}"),
                        qsim_lab::qec::repetition::RepetitionCode::new(distance, rounds)
                            .build_circuit(),
                    ),
                };
                opts.name = Some(name);
                c
            };
            let bytes = onnx::to_onnx(&circuit, &opts).unwrap_or_else(|e| {
                eprintln!("cannot export: {e}");
                std::process::exit(1)
            });
            std::fs::write(&out, &bytes).unwrap_or_else(|e| {
                eprintln!("cannot write {}: {e}", out.display());
                std::process::exit(1)
            });
            let shown = max_ops.map_or(circuit.ops.len(), |m| m.min(circuit.ops.len()));
            println!(
                "wrote {} ({} bytes): {} qubits, {} of {} ops; open it in Netron (https://netron.app)",
                out.display(),
                bytes.len(),
                circuit.num_qubits,
                shown,
                circuit.ops.len()
            );
        }
        Cmd::Run {
            example,
            qubits,
            shots,
            backend,
            modulus,
            seed,
            semiclassical,
            sparse,
            f32,
            fused,
            oracle,
            base,
            tries,
            no_blocked,
            gate_by_gate,
            sliced,
            window,
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
                    let t0 = std::time::Instant::now();
                    let oracle_kind = match oracle {
                        OracleArg::Permutation => shor::Oracle::Permutation,
                        OracleArg::Beauregard => shor::Oracle::Beauregard,
                        OracleArg::Ripple => shor::Oracle::Ripple,
                        OracleArg::Windowed => shor::Oracle::Windowed(window),
                        OracleArg::WindowedOpt => shor::Oracle::WindowedOpt(window),
                        OracleArg::WindowedMbu => shor::Oracle::WindowedMbu(window),
                        OracleArg::WindowedMbuLookup => shor::Oracle::WindowedMbuLookup(window),
                    };
                    let path = qsim_lab::pipeline::choose_shor_path(
                        n,
                        oracle_kind,
                        semiclassical,
                        sparse,
                        f32,
                        qsim_lab::engines::statevector::MAX_STATE_BYTES,
                    );
                    if path.overridden {
                        eprintln!(
                            "note: the dense register for N={n} would exceed the {} MiB memory cap; \
                             running {}{} (exact; pass --semiclassical/--sparse to silence this)",
                            qsim_lab::engines::statevector::MAX_STATE_BYTES >> 20,
                            if path.semiclassical { "semiclassical" } else { "gate-level" },
                            if path.sparse { " sparse" } else { "" },
                        );
                    }
                    let (semiclassical, sparse) = (path.semiclassical, path.sparse);
                    if !semiclassical {
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
                    } else {
                        let oracle = oracle_kind;
                        let backend = match (fused || sliced, sparse && !sliced, f32) {
                            _ if sliced && f32 => shor::Backend::SlicedF32,
                            _ if sliced => shor::Backend::SlicedF64,
                            (false, true, _) => shor::Backend::Sparse,
                            (false, false, true) => shor::Backend::DenseF32,
                            (false, false, false) => shor::Backend::DenseF64,
                            (true, true, _) => shor::Backend::FusedSparse,
                            (true, false, true) => shor::Backend::FusedF32,
                            (true, false, false) => shor::Backend::FusedF64,
                        };
                        let (f, runs) = match base {
                            Some(a) => {
                                let mut inst = shor::Instance::new(n, a, oracle);
                                inst.blocked = !no_blocked;
                                inst.gate_by_gate = gate_by_gate;
                                let r = shor::order_finding(&inst, backend, &mut rng);
                                let f = r.factor.map(|f| (f.min(n / f), f.max(n / f)));
                                (f, vec![r])
                            }
                            None => shor::factor_semiclassical_with_options(
                                n,
                                oracle,
                                backend,
                                tries,
                                !no_blocked,
                                gate_by_gate,
                                &mut rng,
                            ),
                        };
                        for r in &runs {
                            println!(
                                "a={}  qubits={}  measured={}  order={:?}  factor={:?}  peak_amplitudes={}  peak_amp_bytes={}  total_gates={}  toffoli_gates={}  mbu_measurements={}  gate_branch_ops={:.3e}",
                                r.a, r.qubits, r.measured, r.order, r.factor, r.peak_stored, r.peak_bytes, r.total_gates, r.toffoli_gates, r.measurements, r.work_ops as f64
                            );
                        }
                        match f {
                            Some((p, q)) => println!("{n} = {p} x {q}"),
                            None => println!("no factor found"),
                        }
                    }
                    println!(
                        "time {:.3} s  peak RSS {:.1} MiB",
                        t0.elapsed().as_secs_f64(),
                        peak_rss_mib().unwrap_or(f64::NAN)
                    );
                }
            }
        }
    }
}
