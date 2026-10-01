//! Benchmarks the compiler passes against "always simulate the original
//! circuit on one state vector". Run inside the swarm's bench lock:
//!
//! ```sh
//! bench.sh cargo run --release --example compile_bench [filter]
//! ```

use qsim_lab::algorithms;
use qsim_lab::circuit::Circuit;
use qsim_lab::compile::plan::{compile_sampling, compile_unitary, PlanOptions};
use qsim_lab::compile::{expectation_z_product, optimize};
use qsim_lab::Gate;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;
use std::time::Instant;

const REPS: usize = 5;
const SHOTS: usize = 1000;

fn min_time<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

/// Grover search for `marked` on `n` data qubits; the multi-controlled Z is
/// a Toffoli ladder on `n - 3` ancillas (qubits `n..`). Measures the data.
fn grover_circuit(n: usize, marked: usize) -> Circuit {
    let m = n - 1; // controls of the MCX onto data qubit n-1
    let anc = |i: usize| n + i;
    let mut c = Circuit::new(n + m - 2);
    let t = n - 1;
    let mcz = |c: &mut Circuit| {
        c.h(t);
        let mut ladder = vec![(0, 1, anc(0))];
        for i in 2..m - 1 {
            ladder.push((i, anc(i - 2), anc(i - 1)));
        }
        for &(a, b, x) in &ladder {
            c.ccx(a, b, x);
        }
        c.ccx(m - 1, anc(m - 3), t);
        for &(a, b, x) in ladder.iter().rev() {
            c.ccx(a, b, x);
        }
        c.h(t);
    };
    let flip = |c: &mut Circuit, pat: usize| {
        for q in 0..n {
            if (pat >> q) & 1 == 0 {
                c.x(q);
            }
        }
    };
    for q in 0..n {
        c.h(q);
    }
    let iters = ((PI / 4.0) * ((1u64 << n) as f64).sqrt()).floor() as usize;
    for _ in 0..iters {
        flip(&mut c, marked);
        mcz(&mut c);
        flip(&mut c, marked);
        for q in 0..n {
            c.h(q);
        }
        flip(&mut c, 0);
        mcz(&mut c);
        flip(&mut c, 0);
        for q in 0..n {
            c.h(q);
        }
    }
    for q in 0..n {
        c.measure(q);
    }
    c
}

/// Hardware-efficient ansatz: `layers` of Ry/Rz on every qubit then a
/// brickwork of CNOTs between neighbours.
fn hea(n: usize, layers: usize, rng: &mut StdRng) -> Circuit {
    let mut c = Circuit::new(n);
    for l in 0..layers {
        for q in 0..n {
            c.ry(q, rng.random::<f64>() * PI);
            c.rz(q, rng.random::<f64>() * PI);
        }
        for q in (l % 2..n - 1).step_by(2) {
            c.cnot(q, q + 1);
        }
    }
    c
}

/// `k` independent registers of `w` qubits, each a random circuit.
fn registers(k: usize, w: usize, depth: usize, rng: &mut StdRng) -> Circuit {
    let mut c = Circuit::new(k * w);
    for r in 0..k {
        let sub = hea(w, depth, rng);
        for g in sub.gates() {
            c.gate(qsim_lab::compile::analysis::relabel(g, |q| q + r * w));
        }
    }
    c.measure_all();
    c
}

fn row(name: &str, c: &Circuit, base: Option<f64>, comp: f64, compile: f64, extra: &str) {
    let speed = base.map_or("baseline impossible".to_string(), |b| {
        format!("{:.1}x", b / comp)
    });
    println!(
        "| {name} | {} | {} | {} | {comp:.4} | {compile:.4} | {speed} | {extra} |",
        c.num_qubits,
        c.num_gates(),
        base.map_or("-".into(), |b| format!("{b:.4}")),
    );
}

fn sampling_case(name: &str, c: &Circuit, baseline: bool) {
    let mut rng = StdRng::seed_from_u64(7);
    let compile_t = min_time(REPS, || {
        std::hint::black_box(compile_sampling(c, PlanOptions::default()));
    });
    let plan = compile_sampling(c, PlanOptions::default());
    let comp_t = min_time(REPS, || {
        let p = compile_sampling(c, PlanOptions::default());
        std::hint::black_box(p.sample::<f32, _>(SHOTS, &mut rng).unwrap());
    });
    let base_t = baseline.then(|| {
        let p0 = compile_sampling(c, PlanOptions::none());
        min_time(REPS, || {
            std::hint::black_box(p0.sample::<f32, _>(SHOTS, &mut rng).unwrap());
        })
    });
    let s = &plan.stats;
    let comps: Vec<String> = s
        .components
        .iter()
        .map(|(q, g, b)| format!("{q}q/{g}g/{b:?}"))
        .collect();
    let extra = format!(
        "gates {}→{} (peephole) →{} (cone) ; suffix {} ; comps [{}]",
        s.gates_in,
        s.gates_after_peephole,
        s.gates_after_light_cone,
        s.suffix_gates,
        comps.join(", ")
    );
    row(name, c, base_t, comp_t, compile_t, &extra);
}

fn main() {
    let filter = std::env::args().nth(1).unwrap_or_default();
    let want = |name: &str| filter.is_empty() || name.contains(&filter);
    println!(
        "min of {REPS} runs; sampling cases draw {SHOTS} shots with f32 amplitudes; \
         compiled time includes compilation.\n"
    );
    println!("| workload | qubits | gates | always-SV (s) | compiled (s) | compile only (s) | speedup | notes |");
    println!("|---|---|---|---|---|---|---|---|");
    let mut rng = StdRng::seed_from_u64(1);

    if want("qft") {
        for n in [20, 22] {
            let mut c = algorithms::qft(n);
            c.measure_all();
            sampling_case(&format!("qft{n}+measure"), &c, true);
        }
    }
    if want("bv") {
        for n in [16, 23] {
            let secret = 0x2D_5A5Bu64 & ((1 << n) - 1);
            let c = algorithms::bernstein_vazirani(n, secret);
            sampling_case(&format!("bv{n}"), &c, true);
        }
    }
    if want("ghz") {
        for n in [20, 24] {
            let mut c = algorithms::ghz(n);
            c.measure_all();
            sampling_case(&format!("ghz{n}"), &c, true);
        }
    }
    if want("grover") {
        for n in [9, 11] {
            let c = grover_circuit(n, 0b1011 & ((1 << n) - 1));
            sampling_case(&format!("grover{n} (+{} anc)", n - 3), &c, true);
        }
    }
    if want("cliffordt") {
        for (n, d, p) in [(20, 20, 0.02), (22, 30, 0.01)] {
            let mut c = Circuit::random_clifford_t(n, d, p, &mut rng);
            c.measure_all();
            sampling_case(&format!("rand-clifford+T n{n} d{d} p{p}"), &c, true);
        }
        // too wide for a state vector; a few measured qubits
        let mut c = Circuit::random_clifford_t(40, 6, 0.01, &mut rng);
        c.measure(0).measure(1).measure(2);
        sampling_case("rand-clifford+T n40 d6, measure 3", &c, false);
    }
    if want("registers") {
        sampling_case("3 x 8q registers", &registers(3, 8, 10, &mut rng), true);
        sampling_case("4 x 6q registers", &registers(4, 6, 10, &mut rng), true);
        sampling_case(
            "4 x 8q registers (32q)",
            &registers(4, 8, 10, &mut rng),
            false,
        );
    }
    if want("idle") {
        // A 14-qubit computation on a 24-qubit register.
        let mut c = Circuit::new(24);
        for g in hea(14, 8, &mut rng).gates() {
            c.gate(*g);
        }
        c.measure_all();
        sampling_case("hea14 on 24q register", &c, true);
    }
    if want("midcircuit") {
        // measure-and-continue: every layer measures two qubits mid-circuit
        let n = 16;
        let mut c = Circuit::new(n);
        for q in 0..n {
            c.h(q);
        }
        for q in 0..n - 1 {
            c.cnot(q, q + 1);
        }
        for l in 0..6 {
            for q in 0..n {
                c.ry(q, 0.1 * (q + l) as f64).t(q);
            }
            for q in (l % 2..n - 1).step_by(2) {
                c.cz(q, q + 1);
            }
            c.measure((3 * l) % n).measure((3 * l + 7) % n);
        }
        // uncompute-style tail on measured-out qubits that nobody reads
        for q in 0..4 {
            c.h(q).t(q);
        }
        sampling_case("midcircuit 16q", &c, true);
    }
    if want("unitary") {
        let n = 22;
        let mut c = algorithms::qft(n);
        c.append(&algorithms::qft(n).inverse());
        let t0 = min_time(3, || {
            let mut s = qsim_lab::StateVectorF32::new(n);
            s.apply_circuit(&c).unwrap();
            std::hint::black_box(s);
        });
        let t1 = min_time(REPS, || {
            let p = compile_unitary(&c, PlanOptions::default());
            std::hint::black_box(p.statevector::<f32>().unwrap());
        });
        let o = optimize(&c);
        row(
            "qft·qft† n22 (state vector)",
            &c,
            Some(t0),
            t1,
            0.0,
            &format!("gates → {}", o.circuit.num_gates()),
        );

        // Clifford prefix absorption: deep random Clifford then a T layer
        let n = 22;
        let mut c = Circuit::random_clifford(n, 30, &mut rng);
        for q in 0..n {
            c.t(q).h(q);
        }
        let t0 = min_time(3, || {
            let mut s = qsim_lab::StateVectorF32::new(n);
            s.apply_circuit(&c).unwrap();
            std::hint::black_box(s);
        });
        let t1 = min_time(REPS, || {
            let p = compile_unitary(&c, PlanOptions::default());
            std::hint::black_box(p.statevector::<f32>().unwrap());
        });
        row(
            "clifford(d30)+T layer n22 (state vector)",
            &c,
            Some(t0),
            t1,
            0.0,
            "",
        );
    }
    if want("expect") {
        for n in [20, 24] {
            let c = hea(n, 4, &mut rng);
            let q = n / 2;
            let t0 = min_time(3, || {
                let mut s = qsim_lab::StateVectorF32::new(n);
                s.apply_circuit(&c).unwrap();
                std::hint::black_box(s.expectation_z(q));
            });
            let t1 = min_time(REPS, || {
                std::hint::black_box(
                    expectation_z_product(&c, &[q], PlanOptions::default()).unwrap(),
                );
            });
            row(&format!("<Z_mid> hea n{n} L4"), &c, Some(t0), t1, 0.0, "");
        }
        let c = hea(64, 4, &mut rng);
        let t1 = min_time(REPS, || {
            std::hint::black_box(expectation_z_product(&c, &[32], PlanOptions::default()).unwrap());
        });
        row("<Z_32> hea n64 L4", &c, None, t1, 0.0, "");
    }
    let _ = Gate::H(0);
}
