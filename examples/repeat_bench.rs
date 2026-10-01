//! Benchmarks of the repeat pass (`compile::repeat`). Run inside the swarm's
//! bench lock, one suite per invocation:
//!
//! ```sh
//! bench.sh cargo run --release --example repeat_bench -- qec|clifford|trotter|qaoa|brickwork|grover|diag|detect
//! ```
//!
//! Variants are interleaved and the minimum of 3 runs is reported as CSV:
//! `suite,workload,n,r,variant,min_s`.

use qsim_lab::algorithms::random_brickwork;
use qsim_lab::circuit::{Circuit, Op, Simulator};
use qsim_lab::compile::repeat::cliff::{power_gates, sample_program};
use qsim_lab::compile::repeat::exec::{run_dense, ExecOptions};
use qsim_lab::compile::repeat::workloads::*;
use qsim_lab::compile::repeat::{detect, DetectOptions, Node, Program};
use qsim_lab::gate::Gate;
use qsim_lab::{StateVectorF64, Tableau};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

type Variant<'a> = (&'a str, Box<dyn FnMut() + 'a>);

fn race(suite: &str, workload: &str, n: usize, r: usize, mut vs: Vec<Variant>) {
    let mut best = vec![f64::INFINITY; vs.len()];
    for _ in 0..3 {
        for (i, (_, f)) in vs.iter_mut().enumerate() {
            let t = Instant::now();
            f();
            best[i] = best[i].min(t.elapsed().as_secs_f64());
        }
    }
    for (i, (name, _)) in vs.iter().enumerate() {
        println!("{suite},{workload},{n},{r},{name},{:.6}", best[i]);
    }
}

fn maxdiff(a: &StateVectorF64, b: &StateVectorF64) -> f64 {
    a.amplitudes()
        .iter()
        .zip(b.amplitudes())
        .map(|(x, y)| (*x - *y).norm())
        .fold(0.0, f64::max)
}

fn gates_of(c: &Circuit) -> Vec<Gate> {
    c.gates().copied().collect()
}

/// Plain streaming execution of a program on a tableau (no materialisation).
fn plain_tableau(p: &Program, rng: &mut StdRng) -> Vec<bool> {
    fn go(nodes: &[Node], t: &mut Tableau, rng: &mut StdRng, out: &mut Vec<bool>) {
        for n in nodes {
            match n {
                Node::Ops(ops) => {
                    for op in ops {
                        match op {
                            Op::Gate(g) => t.apply_gate(g).unwrap(),
                            Op::Measure(q) => out.push(t.measure_qubit(*q, rng)),
                            Op::Reset(q) => {
                                t.reset_qubit(*q, rng);
                            }
                            _ => unreachable!(),
                        }
                    }
                }
                Node::Repeat { body, reps } => {
                    for _ in 0..*reps {
                        go(body, t, rng, out);
                    }
                }
                Node::Param { .. } => unreachable!(),
            }
        }
    }
    let mut t = Tableau::new(p.num_qubits);
    let mut out = Vec::new();
    go(&p.nodes, &mut t, rng, &mut out);
    out
}

fn qec() {
    for (d, plus) in [(5usize, true), (9, true)] {
        for rounds in [1_000usize, 10_000, 100_000, 1_000_000] {
            let p = qec_program(d, rounds, plus);
            let n = p.num_qubits;
            // the plain baseline streams the same ops; at 1e6 rounds a single
            // shot takes seconds, so one shot is timed per variant
            let name = format!("rep-d{d}{}", if plus { "-plus" } else { "" });
            let mut a = StdRng::seed_from_u64(1);
            let mut b = StdRng::seed_from_u64(1);
            let want = plain_tableau(&p, &mut a);
            let (got, stats) = sample_program(&p, true, true, &mut b).unwrap();
            assert_eq!(want, got);
            eprintln!("# {name} r={rounds} {stats:?}");
            race(
                "qec",
                &name,
                n,
                rounds,
                vec![
                    (
                        "plain",
                        Box::new(|| {
                            let mut r = StdRng::seed_from_u64(1);
                            std::hint::black_box(plain_tableau(&p, &mut r));
                        }),
                    ),
                    (
                        "repeat",
                        Box::new(|| {
                            let mut r = StdRng::seed_from_u64(1);
                            std::hint::black_box(sample_program(&p, true, true, &mut r));
                        }),
                    ),
                ],
            );
        }
    }
}

fn clifford() {
    // unitary Clifford block repeated r times on the CHP tableau
    for n in [50usize, 200] {
        let mut rng = StdRng::seed_from_u64(7);
        let body = random_clifford_gates(n, 4 * n, &mut rng);
        for r in [1_000usize, 10_000, 100_000, 1_000_000] {
            let mut variants: Vec<Variant> = Vec::new();
            // plain only where it finishes in reasonable time
            if r * body.len() * n <= 2_000_000_000 {
                variants.push((
                    "plain",
                    Box::new(|| {
                        let mut t = Tableau::new(n);
                        for _ in 0..r {
                            for g in &body {
                                t.apply_gate(g).unwrap();
                            }
                        }
                        std::hint::black_box(&t);
                    }),
                ));
            }
            variants.push((
                "power",
                Box::new(|| {
                    let g = power_gates(&body, r as u64).unwrap();
                    let mut t = Tableau::new(n);
                    for x in &g {
                        t.apply_gate(x).unwrap();
                    }
                    std::hint::black_box(&t);
                }),
            ));
            race("clifford", "random-clifford-block", n, r, variants);
        }
    }
}

fn dense_variants<'a>(
    n: usize,
    prog: &'a Program,
    full: &'a Circuit,
    body: &'a [Gate],
    reps: usize,
    opts: ExecOptions,
) -> Vec<Variant<'a>> {
    let fg = gates_of(full);
    vec![
        (
            "gatewise",
            Box::new(move || {
                let mut sv = StateVectorF64::new(n);
                sv.apply_circuit(full).unwrap();
                std::hint::black_box(&sv);
            }),
        ),
        (
            "plain-batch",
            Box::new(move || {
                let mut sv = StateVectorF64::new(n);
                sv.apply_gates(&fg).unwrap();
                std::hint::black_box(&sv);
            }),
        ),
        (
            "plain-per-copy",
            Box::new(move || {
                let mut sv = StateVectorF64::new(n);
                for _ in 0..reps {
                    sv.apply_gates(body).unwrap();
                }
                std::hint::black_box(&sv);
            }),
        ),
        (
            "repeat",
            Box::new(move || {
                let mut sv = StateVectorF64::new(n);
                run_dense(prog, &mut sv, &opts).unwrap();
                std::hint::black_box(&sv);
            }),
        ),
        (
            "repeat-force-small",
            Box::new(move || {
                let mut sv = StateVectorF64::new(n);
                let o = ExecOptions {
                    force_small: true,
                    ..ExecOptions::default()
                };
                run_dense(prog, &mut sv, &o).unwrap();
                std::hint::black_box(&sv);
            }),
        ),
    ]
}

fn repeat_prog(n: usize, prefix: &Circuit, body: &Circuit, reps: usize) -> Program {
    Program {
        num_qubits: n,
        nodes: vec![
            Node::Ops(prefix.ops.clone()),
            Node::Repeat {
                body: vec![Node::Ops(body.ops.clone())],
                reps,
            },
        ],
    }
}

fn trotter_suite() {
    for n in [6usize, 10, 14] {
        for r in [1_000usize, 10_000] {
            if n == 14 && r > 1_000 {
                continue;
            }
            let step = tfim_step(n, 0.05, 0.04);
            let mut prefix = Circuit::new(n);
            for q in 0..n {
                prefix.h(q);
            }
            let prog = repeat_prog(n, &prefix, &step, r);
            let mut full = prefix.clone();
            full.append(&repeated(&step, r));
            let body = gates_of(&step);
            let mut a = StateVectorF64::new(n);
            a.apply_circuit(&full).unwrap();
            let mut b = StateVectorF64::new(n);
            run_dense(&prog, &mut b, &ExecOptions::default()).unwrap();
            eprintln!("# trotter n={n} r={r} max|Δamp| = {:e}", maxdiff(&a, &b));
            race(
                "trotter",
                "tfim-chain",
                n,
                r,
                dense_variants(n, &prog, &full, &body, r, ExecOptions::default()),
            );
        }
    }
}

fn qaoa_suite() {
    for (n, p) in [(12usize, 20usize), (18, 20)] {
        let gam: Vec<f64> = (0..p).map(|i| 0.1 + 0.03 * i as f64).collect();
        let bet: Vec<f64> = (0..p).map(|i| 0.7 - 0.02 * i as f64).collect();
        let c = qaoa_ring(n, &gam, &bet);
        let prog = detect(&c, &DetectOptions::default());
        let canon_g = gates_of(&prog.to_circuit());
        let cg = gates_of(&c);
        eprintln!("# qaoa n={n} p={p} {:?}", prog.report());
        let mut a = StateVectorF64::new(n);
        a.apply_circuit(&c).unwrap();
        let mut b = StateVectorF64::new(n);
        run_dense(&prog, &mut b, &ExecOptions::default()).unwrap();
        eprintln!("# qaoa max|Δamp| = {:e}", maxdiff(&a, &b));
        race(
            "qaoa",
            "ring",
            n,
            p,
            vec![
                (
                    "gatewise",
                    Box::new(|| {
                        let mut sv = StateVectorF64::new(n);
                        sv.apply_circuit(&c).unwrap();
                        std::hint::black_box(&sv);
                    }),
                ),
                (
                    "plain-batch",
                    Box::new(|| {
                        let mut sv = StateVectorF64::new(n);
                        sv.apply_gates(&cg).unwrap();
                        std::hint::black_box(&sv);
                    }),
                ),
                (
                    "plain-canonical-order",
                    Box::new(|| {
                        let mut sv = StateVectorF64::new(n);
                        sv.apply_gates(&canon_g).unwrap();
                        std::hint::black_box(&sv);
                    }),
                ),
                (
                    "detect-only",
                    Box::new(|| {
                        std::hint::black_box(detect(&c, &DetectOptions::default()));
                    }),
                ),
                (
                    "detect+repeat",
                    Box::new(|| {
                        let prog = detect(&c, &DetectOptions::default());
                        let mut sv = StateVectorF64::new(n);
                        run_dense(&prog, &mut sv, &ExecOptions::default()).unwrap();
                        std::hint::black_box(&sv);
                    }),
                ),
            ],
        );
    }
}

fn brickwork_suite() {
    for (n, depth, r) in [(20usize, 4usize, 20usize), (22, 4, 10), (22, 8, 6)] {
        let mut rng = StdRng::seed_from_u64(3);
        let body = random_brickwork(n, depth, &mut rng);
        let prefix = Circuit::new(n);
        let prog = repeat_prog(n, &prefix, &body, r);
        let full = repeated(&body, r);
        let bg = gates_of(&body);
        let mut a = StateVectorF64::new(n);
        a.apply_circuit(&full).unwrap();
        let mut b = StateVectorF64::new(n);
        let st = run_dense(&prog, &mut b, &ExecOptions::default()).unwrap();
        eprintln!(
            "# brickwork n={n} depth={depth} r={r} {st:?} max|Δamp| = {:e}",
            maxdiff(&a, &b)
        );
        // compile vs run split
        {
            use qsim_lab::blocked::{lower_gates, BlockConfig};
            let ops = lower_gates(bg.iter());
            let t = Instant::now();
            let plan = StateVectorF64::compile_kops(n, &ops, &BlockConfig::default());
            let tc = t.elapsed().as_secs_f64();
            let mut sv = StateVectorF64::new(n);
            let t = Instant::now();
            sv.run_compiled(&plan);
            let tr = t.elapsed().as_secs_f64();
            println!("brickwork,compile-vs-run,{n},{depth},compile_s,{tc:.6}");
            println!("brickwork,compile-vs-run,{n},{depth},run_s,{tr:.6}");
        }
        race(
            "brickwork",
            &format!("depth{depth}"),
            n,
            r,
            dense_variants(n, &prog, &full, &bg, r, ExecOptions::default()),
        );
    }
}

fn grover_suite() {
    for (n, it) in [(7usize, 200usize), (9, 100)] {
        let full = grover(n, 5, it);
        let nq = full.num_qubits;
        let fg = gates_of(&full);
        let p = detect(&full, &DetectOptions::default());
        eprintln!("# grover n={n} (qubits {nq}) it={it} {:?}", p.report());
        let mut a = StateVectorF64::new(nq);
        a.apply_circuit(&full).unwrap();
        let mut b = StateVectorF64::new(nq);
        run_dense(&p, &mut b, &ExecOptions::default()).unwrap();
        eprintln!("# grover max|Δamp| = {:e}", maxdiff(&a, &b));
        race(
            "grover",
            "mcz-ladder",
            nq,
            it,
            vec![
                (
                    "gatewise",
                    Box::new(|| {
                        let mut sv = StateVectorF64::new(nq);
                        sv.apply_circuit(&full).unwrap();
                        std::hint::black_box(&sv);
                    }),
                ),
                (
                    "plain-batch",
                    Box::new(|| {
                        let mut sv = StateVectorF64::new(nq);
                        sv.apply_gates(&fg).unwrap();
                        std::hint::black_box(&sv);
                    }),
                ),
                (
                    "repeat",
                    Box::new(|| {
                        let mut sv = StateVectorF64::new(nq);
                        run_dense(&p, &mut sv, &ExecOptions::default()).unwrap();
                        std::hint::black_box(&sv);
                    }),
                ),
            ],
        );
    }
}

fn diag_suite() {
    for (n, r) in [(14usize, 30_000usize), (20, 200)] {
        let mut b = Circuit::new(n);
        for q in 0..n {
            b.rz(q, 0.01 * (q + 1) as f64);
        }
        for q in 0..n - 1 {
            b.cphase(q, q + 1, 0.02);
        }
        let mut prefix = Circuit::new(n);
        for q in 0..n {
            prefix.h(q);
        }
        let prog = repeat_prog(n, &prefix, &b, r);
        let mut full = prefix.clone();
        let bg = gates_of(&b);
        full.append(&repeated(&b, r));
        let mut a = StateVectorF64::new(n);
        a.apply_circuit(&full).unwrap();
        let mut c = StateVectorF64::new(n);
        run_dense(&prog, &mut c, &ExecOptions::default()).unwrap();
        eprintln!("# diag n={n} r={r} max|Δamp| = {:e}", maxdiff(&a, &c));
        race(
            "diag",
            "phase-layer",
            n,
            r,
            dense_variants(n, &prog, &full, &bg, r, ExecOptions::default()),
        );
    }
}

fn detect_suite() {
    let gam: Vec<f64> = (0..20).map(|i| 0.1 + 0.03 * i as f64).collect();
    let bet: Vec<f64> = (0..20).map(|i| 0.7 - 0.02 * i as f64).collect();
    let cases: Vec<(&str, Circuit)> = vec![
        ("trotter-n12-r1000", trotter(12, 1000, 0.05, 0.04)),
        ("qaoa-n12-p20", qaoa_ring(12, &gam, &bet)),
        ("grover-n7-it200", grover(7, 5, 200)),
        ("qec-d5-r1000", qec_memory(5, 1000, true)),
        ("qec-d9-r1000", qec_memory(9, 1000, true)),
        ("brickwork-n20-d4-r40", {
            let mut rng = StdRng::seed_from_u64(3);
            repeated(&random_brickwork(20, 4, &mut rng), 40)
        }),
        ("random-brickwork-d200 (no repeats)", {
            let mut rng = StdRng::seed_from_u64(3);
            random_brickwork(20, 200, &mut rng)
        }),
        ("shuffled-trotter-n8-r200", {
            let mut rng = StdRng::seed_from_u64(9);
            let step = tfim_step(8, 0.05, 0.04);
            let mut c = Circuit::new(8);
            for _ in 0..200 {
                c.append(&shuffle_commuting(&step, &mut rng));
            }
            c
        }),
    ];
    for (name, c) in cases {
        for (label, o) in [
            (
                "as-written",
                DetectOptions {
                    layered: false,
                    parameterised: false,
                    ..Default::default()
                },
            ),
            (
                "layered",
                DetectOptions {
                    parameterised: false,
                    ..Default::default()
                },
            ),
            ("layered+param", DetectOptions::default()),
        ] {
            let t = Instant::now();
            let p = detect(&c, &o);
            let dt = t.elapsed().as_secs_f64();
            let r = p.report();
            println!(
                "detect,{name},{},{},{label},gates={} covered={:.4} saved={:.4} repeats={} param={} time_s={dt:.4}",
                c.num_qubits,
                c.ops.len(),
                r.total_gates,
                r.coverage(),
                r.saved_gates as f64 / r.total_gates.max(1) as f64,
                r.repeats,
                r.param_repeats
            );
        }
    }
}

fn main() {
    let suite = std::env::args().nth(1).unwrap_or_default();
    match suite.as_str() {
        "qec" => qec(),
        "clifford" => clifford(),
        "trotter" => trotter_suite(),
        "qaoa" => qaoa_suite(),
        "brickwork" => brickwork_suite(),
        "grover" => grover_suite(),
        "diag" => diag_suite(),
        "detect" => detect_suite(),
        _ => {
            eprintln!("usage: repeat_bench qec|clifford|trotter|qaoa|brickwork|grover|diag|detect")
        }
    }
}
