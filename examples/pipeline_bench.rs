//! End-to-end benchmark of `qsim_lab::pipeline::simulate` against plain
//! `Circuit::run`-style simulation. Run inside the swarm's bench lock:
//!
//! ```sh
//! bench.sh cargo run --release --example pipeline_bench [filter]
//! ```
//!
//! For every family the table reports min-of-N seconds of
//!   * `gatewise`: the pre-pipeline path (one `apply_gate` pass per gate),
//!     then `sample(shots)` on the final vector (terminal circuits) or one
//!     gate-by-gate run per shot (mid-circuit circuits);
//!   * `run`: `Circuit::run` (blocked executor for gate segments), same
//!     sampling;
//!   * `simulate`: `pipeline::simulate(Request::Samples)`.

use qsim_lab::algorithms;
use qsim_lab::circuit::{Circuit, Op, Simulator};
use qsim_lab::pipeline::{simulate, Budget, Request};
use qsim_lab::{Gate, StateVectorF64};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn min_time<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

fn dense_layers(n: usize, depth: usize, qs: &[usize], rng: &mut StdRng, c: &mut Circuit) {
    let _ = n;
    for _ in 0..depth {
        for &q in qs {
            match rng.random_range(0..4) {
                0 => c.gate(Gate::Rx(q, rng.random_range(0.1..3.0))),
                1 => c.gate(Gate::Ry(q, rng.random_range(0.1..3.0))),
                2 => c.gate(Gate::Rz(q, rng.random_range(0.1..3.0))),
                _ => c.gate(Gate::H(q)),
            };
        }
        for w in qs.windows(2).step_by(2) {
            c.cnot(w[0], w[1]);
        }
        for w in qs.windows(2).skip(1).step_by(2) {
            c.cz(w[0], w[1]);
        }
    }
    let _ = qs;
}

fn random_dense(n: usize, depth: usize, seed: u64) -> Circuit {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut c = Circuit::new(n);
    let qs: Vec<usize> = (0..n).collect();
    dense_layers(n, depth, &qs, &mut rng, &mut c);
    c.measure_all();
    c
}

fn two_blocks(n: usize, depth: usize, seed: u64) -> Circuit {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut c = Circuit::new(n);
    let a: Vec<usize> = (0..n / 2).collect();
    let b: Vec<usize> = (n / 2..n).collect();
    dense_layers(n, depth, &a, &mut rng, &mut c);
    dense_layers(n, depth, &b, &mut rng, &mut c);
    c.measure_all();
    c
}

fn qft_family(n: usize, seed: u64) -> Circuit {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.gate(Gate::Ry(q, rng.random_range(0.2..2.9)));
    }
    c.append(&algorithms::qft(n));
    c.measure_all();
    c
}

fn ghz(n: usize) -> Circuit {
    let mut c = algorithms::ghz(n);
    c.measure_all();
    c
}

/// Clifford layers with `t` T gates interleaved.
fn clifford_t(n: usize, t: usize, seed: u64) -> Circuit {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut c = Circuit::new(n);
    for _ in 0..t {
        c.append(&Circuit::random_clifford(n, 4, &mut rng));
        c.gate(Gate::T(rng.random_range(0..n)));
    }
    c.append(&Circuit::random_clifford(n, 4, &mut rng));
    c.measure_all();
    c
}

/// Rounds of entangling, mid-circuit measurement, reset and classically
/// controlled correction on a ring of dense gates.
fn midcircuit(n: usize, rounds: usize, seed: u64) -> Circuit {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut c = Circuit::new(n);
    for meas in 0..rounds {
        let qs: Vec<usize> = (0..n).collect();
        dense_layers(n, 3, &qs, &mut rng, &mut c);
        let q = rng.random_range(0..n);
        c.measure(q);
        c.classic_controlled(Gate::X((q + 1) % n), meas, true);
        c.reset(q);
    }
    c.measure_all();
    c
}

fn has_nonunitary_mid(c: &Circuit) -> bool {
    let last_gate = c.ops.iter().rposition(|o| matches!(o, Op::Gate(_)));
    c.ops.iter().enumerate().any(|(i, o)| {
        !matches!(o, Op::Gate(_)) && !matches!(o, Op::Measure(_)) || {
            matches!(o, Op::Measure(_)) && last_gate.is_some_and(|g| g > i)
        }
    })
}

/// The pre-pipeline behaviour: every gate is one pass over the vector.
fn run_gatewise(c: &Circuit, s: &mut StateVectorF64, rng: &mut StdRng) {
    for op in &c.ops {
        match op {
            Op::Gate(g) => s.apply_gate(g).unwrap(),
            Op::Measure(q) => {
                s.measure(*q, rng).unwrap();
            }
            Op::Reset(q) => s.reset(*q, rng).unwrap(),
            Op::ClassicControlled { .. } => {}
            _ => {}
        }
    }
}

fn row(name: &str, c: &Circuit, shots: usize, reps: usize) {
    let n = c.num_qubits;
    let mid = has_nonunit_or_mid(c);
    let (tg, tr);
    if !mid {
        let gates_only = Circuit {
            num_qubits: n,
            ops: c
                .ops
                .iter()
                .filter(|o| matches!(o, Op::Gate(_)))
                .copied()
                .collect(),
        };
        tg = min_time(reps, || {
            let mut s = StateVectorF64::new(n);
            for g in gates_only.gates() {
                s.apply_gate(g).unwrap();
            }
            std::hint::black_box(s.sample(shots, &mut StdRng::seed_from_u64(1)));
        });
        tr = min_time(reps, || {
            let mut s = StateVectorF64::new(n);
            gates_only
                .run(&mut s, &mut StdRng::seed_from_u64(1))
                .unwrap();
            std::hint::black_box(s.sample(shots, &mut StdRng::seed_from_u64(1)));
        });
    } else {
        let per = shots.min(20);
        tg = min_time(reps.min(2), || {
            for i in 0..per {
                let mut s = StateVectorF64::new(n);
                run_gatewise_ctrl(c, &mut s, &mut StdRng::seed_from_u64(i as u64));
            }
        }) * (shots as f64 / per as f64);
        tr = min_time(reps.min(2), || {
            for i in 0..per {
                let mut s = StateVectorF64::new(n);
                c.run(&mut s, &mut StdRng::seed_from_u64(i as u64)).unwrap();
            }
        }) * (shots as f64 / per as f64);
    }
    let mut engines = String::new();
    let ts = min_time(reps.min(if mid { 1 } else { reps }), || {
        let r = simulate(c, &Request::Samples { shots, seed: 3 }, &Budget::default()).unwrap();
        engines = format!("{:?}", r.engines.iter().map(|e| e.2).collect::<Vec<_>>());
    });
    if engines.len() > 60 {
        engines.truncate(57);
        engines.push_str("...");
    }
    println!(
        "{name:<26} n={n:<3} gates={:<5} shots={shots:<5} gatewise {tg:>9.4}s  run {tr:>9.4}s  simulate {ts:>9.4}s  | run {:>6.2}x  simulate {:>7.2}x  {engines}",
        c.num_gates(),
        tg / tr,
        tg / ts,
    );
}

fn has_nonunit_or_mid(c: &Circuit) -> bool {
    has_nonunitary_mid(c)
}

fn run_gatewise_ctrl(c: &Circuit, s: &mut StateVectorF64, rng: &mut StdRng) {
    let mut out: Vec<bool> = Vec::new();
    for op in &c.ops {
        match op {
            Op::Gate(g) => s.apply_gate(g).unwrap(),
            Op::Measure(q) => out.push(s.measure(*q, rng).unwrap()),
            Op::Reset(q) => s.reset(*q, rng).unwrap(),
            Op::ClassicControlled {
                gate,
                meas_index,
                target_value,
            } => {
                if out[*meas_index] == *target_value {
                    s.apply_gate(gate).unwrap();
                }
            }
            _ => {}
        }
    }
    let _ = run_gatewise;
}

/// Adaptive vs dense state vector for the same compile plan, over (n, t).
fn threshold_sweep() {
    use qsim_lab::compile::plan::{compile_sampling, AdaptiveRule, Backend};
    use qsim_lab::pipeline::plan_options;
    println!("threshold sweep: terminal Clifford+T sampling, 1000 shots, min of 3");
    println!("n   t   d   dense_s   adaptive_s  ratio  rule_picks");
    for n in [10usize, 12, 14, 16, 18, 20] {
        for t in [2usize, 4, 8, 12, 16] {
            let c = clifford_t(n, t, 10 + t as u64);
            let mut off = plan_options();
            off.adaptive = None;
            let mut on = plan_options();
            on.adaptive = Some(AdaptiveRule {
                min_qubits: 1,
                margin: 0,
                max_active: 30,
            });
            let d = qsim_lab::adaptive::active_dimension(&Circuit {
                num_qubits: n,
                ops: c
                    .ops
                    .iter()
                    .filter(|o| matches!(o, Op::Gate(_)))
                    .copied()
                    .collect(),
            })
            .unwrap();
            let dense = min_time(3, || {
                let p = compile_sampling(&c, off).unwrap();
                std::hint::black_box(
                    p.sample::<f64, _>(1000, &mut StdRng::seed_from_u64(1))
                        .unwrap(),
                );
            });
            let mut picked = false;
            let ad = min_time(3, || {
                let p = compile_sampling(&c, on).unwrap();
                picked = p
                    .components()
                    .iter()
                    .any(|c| c.backend == Backend::Adaptive);
                std::hint::black_box(
                    p.sample::<f64, _>(1000, &mut StdRng::seed_from_u64(1))
                        .unwrap(),
                );
            });
            let rule = plan_options();
            let pick = {
                let p = compile_sampling(&c, rule).unwrap();
                p.components()
                    .iter()
                    .any(|c| c.backend == Backend::Adaptive)
            };
            println!(
                "{n:<3} {t:<3} {d:<3} {dense:>9.5} {ad:>11.5} {:>6.2}  forced_adaptive_used={picked} rule_adaptive={pick}",
                dense / ad
            );
        }
    }
}

/// Z-product expectation: adaptive on/off.
fn expectation_sweep() {
    use qsim_lab::compile::plan::{expectation_z_product, AdaptiveRule};
    use qsim_lab::pipeline::plan_options;
    println!("expectation sweep: <Z0 Z1 Z2> after Clifford+T, min of 3");
    println!("n   t   dense_or_pauli_s   adaptive_s  ratio");
    for n in [12usize, 16, 20, 24] {
        for t in [4usize, 8, 12, 20, 30] {
            let mut c = clifford_t(n, t, 30 + t as u64);
            c.ops.retain(|o| matches!(o, Op::Gate(_)));
            let mut off = plan_options();
            off.adaptive = None;
            let mut on = plan_options();
            on.adaptive = Some(AdaptiveRule {
                min_qubits: 1,
                margin: 0,
                max_active: 26,
            });
            let a = min_time(3, || {
                std::hint::black_box(expectation_z_product(&c, &[0, 1, 2], off).unwrap());
            });
            let b = min_time(3, || {
                std::hint::black_box(expectation_z_product(&c, &[0, 1, 2], on).unwrap());
            });
            println!("{n:<3} {t:<3} {a:>16.5} {b:>12.5} {:>6.2}", a / b);
        }
    }
}

fn main() {
    let filter = std::env::args().nth(1).unwrap_or_default();
    let want = |name: &str| filter.is_empty() || name.contains(&filter);
    let reps = 3;
    if filter == "thr" {
        threshold_sweep();
        expectation_sweep();
        return;
    }
    println!("(min of {reps}; shared VM; shots=1000 unless noted)");
    if want("dense") {
        row("random dense", &random_dense(20, 20, 1), 1000, reps);
        row("random dense", &random_dense(22, 12, 1), 1000, reps);
    }
    if want("qft") {
        row("qft", &qft_family(22, 2), 1000, reps);
    }
    if want("blocks") {
        row("two independent blocks", &two_blocks(24, 14, 3), 1000, reps);
    }
    if want("ghz") {
        row("ghz", &ghz(24), 1000, reps);
    }
    if want("cliffordt") {
        row("clifford+T t=4", &clifford_t(22, 4, 4), 1000, reps);
        row("clifford+T t=8", &clifford_t(22, 8, 5), 1000, reps);
        row("clifford+T t=14", &clifford_t(22, 14, 6), 1000, reps);
    }
    if want("mid") {
        row(
            "mid-circuit measure/reset",
            &midcircuit(16, 6, 7),
            100,
            reps,
        );
    }
}
