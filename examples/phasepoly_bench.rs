//! T-count / non-Clifford-count table for the DAG peephole vs phase folding
//! (and their composition), on random Clifford+T circuits and arithmetic
//! circuits lowered to Clifford+T. Also writes each circuit as OpenQASM to
//! `research/data/phasepoly/circuits/` so PyZX can be run on the same
//! inputs (see `research/data/phasepoly/pyzx_compare.py`), and checks every
//! n <= 12 result against the state vector.
//!
//! Usage: cargo run --release --example phasepoly_bench [outdir]
//! Timings (`time` mode) must go through the swarm's bench.sh lock.

use qsim_lab::algorithms::qft;
use qsim_lab::compile::{optimize, phase_fold};
use qsim_lab::gate::toffoli_clifford_t;
use qsim_lab::qasm::to_qasm;
use qsim_lab::shor_ripple::{add_mod, controlled_ua, cuccaro_add, RippleLayout};
use qsim_lab::{Circuit, Gate, Op, StateVectorF64};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::f64::consts::PI;
use std::time::Instant;

fn lower(c: &Circuit) -> Circuit {
    let mut o = Circuit::new(c.num_qubits);
    for op in &c.ops {
        match *op {
            Op::Gate(Gate::Ccx(a, b, t)) => {
                for g in toffoli_clifford_t(a, b, t) {
                    o.gate(g);
                }
            }
            Op::Gate(Gate::CPhase(a, b, th)) => {
                o.phase(a, th / 2.0)
                    .cnot(a, b)
                    .phase(b, -th / 2.0)
                    .cnot(a, b)
                    .phase(b, th / 2.0);
            }
            op => o.ops.push(op),
        }
    }
    o
}

fn toffoli_ladder(n: usize, reps: usize) -> Circuit {
    // compute-uncompute AND ladder: a0 a1 -> t0, t0 a2 -> t1, ...
    let k = n - 2;
    let mut c = Circuit::new(n + k);
    for q in 0..n {
        c.h(q);
    }
    for _ in 0..reps {
        c.ccx(0, 1, n);
        for i in 1..k {
            c.ccx(n + i - 1, i + 1, n + i);
        }
        c.cnot(n + k - 1, 0);
        for i in (1..k).rev() {
            c.ccx(n + i - 1, i + 1, n + i);
        }
        c.ccx(0, 1, n);
    }
    c
}

/// Approximate QFT: controlled phases π/2^m only for m <= max_m.
fn aqft(n: usize, max_m: usize) -> Circuit {
    let mut c = Circuit::new(n);
    for j in (0..n).rev() {
        c.h(j);
        for k in (0..j).rev() {
            let m = j - k;
            if m <= max_m {
                c.cphase(k, j, PI / (1u64 << m) as f64);
            }
        }
    }
    for j in 0..n / 2 {
        c.swap(j, n - 1 - j);
    }
    c
}

fn nc(c: &Circuit) -> usize {
    c.gates().filter(|g| !g.is_clifford()).count()
}

fn check(name: &str, c: &Circuit, o: &Circuit, phase: f64) {
    if c.num_qubits > 22 {
        return;
    }
    // arithmetic circuits are permutations on |0>: use a random-ish input
    let mut rng = StdRng::seed_from_u64(99);
    let prep = qsim_lab::Circuit::random_clifford_t(c.num_qubits, 6, 0.3, &mut rng);
    let run = |b: &Circuit| {
        let mut s = StateVectorF64::new(c.num_qubits);
        s.apply_circuit(&prep).unwrap();
        s.apply_circuit(b).unwrap();
        s
    };
    let (a, b) = (run(c), run(o));
    let ph = num_complex::Complex64::from_polar(1.0, phase);
    let d = a
        .amplitudes()
        .iter()
        .zip(b.amplitudes())
        .map(|(x, y)| (x - ph * y).norm())
        .fold(0.0, f64::max);
    assert!(d < 1e-10, "{name}: state mismatch {d}");
}

fn main() {
    let outdir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "research/data/phasepoly".into());
    let _ = std::fs::create_dir_all(format!("{outdir}/circuits"));
    let mut rng = StdRng::seed_from_u64(2024);
    let mut cases: Vec<(String, Circuit)> = Vec::new();
    for (n, d) in [(8, 50), (12, 80), (16, 100), (32, 200)] {
        let c = Circuit::random_clifford_t(n, d, 0.25, &mut rng);
        cases.push((format!("random_ct_n{n}_d{d}"), c));
    }
    for n in [8usize, 16, 32] {
        // a (n), b (n), carry-in, carry-out
        let a: Vec<usize> = (0..n).collect();
        let b: Vec<usize> = (n..2 * n + 1).collect();
        let mut c = Circuit::new(2 * n + 2);
        cuccaro_add(&mut c, &a, &b, 2 * n + 1);
        cases.push((format!("cuccaro_add_n{n}"), lower(&c)));
    }
    for (n, r) in [(6, 1), (10, 2), (16, 2)] {
        cases.push((format!("toffoli_ladder_n{n}_r{r}"), lower(&toffoli_ladder(n, r))));
    }
    for (nm, a, nmod) in [(3usize, 5u64, 7u64), (4, 7, 15), (5, 11, 21), (6, 13, 55)] {
        let lay = RippleLayout::new(nm);
        let mut c = Circuit::new(lay.num_qubits());
        add_mod(&mut c, &lay, a % nmod, nmod, &[lay.ctrl]);
        cases.push((format!("ripple_addmod_n{nm}"), lower(&c)));
        let ua = controlled_ua(&lay, lay.ctrl, a, nmod);
        cases.push((format!("ripple_ctrl_ua_n{nm}"), lower(&ua)));
    }
    for (n, m) in [(8usize, 3usize), (16, 3), (16, 5), (32, 4)] {
        cases.push((format!("aqft_n{n}_m{m}"), lower(&aqft(n, m))));
    }
    cases.push(("qft_n12_full".into(), lower(&qft(12))));

    println!("circuit,qubits,gates,nc_orig,nc_peephole,nc_fold,nc_peep_fold_peep,t_orig,t_peephole,t_fold,t_pfp,fold_ms");
    let mut csv = String::from("circuit,qubits,gates,nc_orig,nc_peephole,nc_fold,nc_peep_fold_peep,t_orig,t_peephole,t_fold,t_pfp,fold_ms\n");
    for (name, c) in &cases {
        let _ = std::fs::write(format!("{outdir}/circuits/{name}.qasm"), to_qasm(c).unwrap());
        let p = optimize(c);
        let t0 = Instant::now();
        let f = phase_fold(c);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let f2 = phase_fold(&p.circuit);
        let p2 = optimize(&f2.circuit);
        let pfp_phase = p.global_phase + f2.global_phase + p2.global_phase;
        check(name, c, &p.circuit, p.global_phase);
        check(name, c, &f.circuit, f.global_phase);
        check(name, c, &p2.circuit, pfp_phase);
        let line = format!(
            "{name},{},{},{},{},{},{},{},{},{},{},{ms:.2}",
            c.num_qubits,
            c.num_gates(),
            nc(c),
            nc(&p.circuit),
            nc(&f.circuit),
            nc(&p2.circuit),
            c.t_count(),
            p.circuit.t_count(),
            f.circuit.t_count(),
            p2.circuit.t_count(),
        );
        println!("{line}");
        csv.push_str(&line);
        csv.push('\n');
    }
    std::fs::write(format!("{outdir}/counts.csv"), csv).unwrap();
}
