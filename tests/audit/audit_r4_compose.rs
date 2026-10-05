//! Round-4 audit: phase folding (`compile::phase_fold`) and the repeat pass
//! (`compile::repeat`) composed, in both orders, against the naive
//! reference (whole instrument incl. global phase), plus how much repeat
//! structure folding leaves behind (printed with `--nocapture`).

#![allow(clippy::needless_range_loop)]

#[path = "../audit_common/mod.rs"]
mod audit_common;
#[path = "../audit_r4/mod.rs"]
mod audit_r4;

use audit_common::iters;
use audit_r4::{compare, random_state};
use qsim_lab::compile::phase_fold;
use qsim_lab::compile::repeat::workloads::{grover, qec_memory, repeated, trotter};
use qsim_lab::compile::repeat::{detect, rewrite, DetectOptions};
use qsim_lab::gate::toffoli_clifford_t;
use qsim_lab::{Circuit, Gate, Op};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn block(rng: &mut StdRng, n: usize, len: usize, measure: bool) -> Circuit {
    let mut c = Circuit::new(n);
    for _ in 0..len {
        let q = rng.random_range(0..n);
        let p = (q + 1 + rng.random_range(0..n - 1)) % n;
        match rng.random_range(0..10) {
            0 | 1 => {
                c.t(q);
            }
            2 => {
                c.tdg(q);
            }
            3 => {
                c.h(q);
            }
            4 => {
                c.rz(q, rng.random_range(-3.0..3.0));
            }
            5 => {
                c.s(q);
            }
            _ => {
                c.cnot(q, p);
            }
        }
    }
    if measure {
        let q = rng.random_range(0..n);
        c.measure(q).reset(q);
    }
    c
}

/// fold(rewrite(detect(c))) and rewrite(detect(fold(c))) are both exact.
#[test]
fn audit_compose_fold_and_repeat_exact() {
    let mut rng = StdRng::seed_from_u64(audit_common::base_seed() ^ 0xC0_4F05E);
    for it in 0..80 * iters() {
        let n = rng.random_range(2..=5);
        let measure = it % 3 == 2;
        let len = rng.random_range(3..12);
        let b = block(&mut rng, n, len, measure);
        let reps = if measure { 3 } else { [2, 5, 9][it % 3] };
        let mut c = Circuit::new(n);
        c.h(0);
        c.append(&repeated(&b, reps));
        c.t(n - 1);
        let init = random_state(n, &mut rng);
        let opts = DetectOptions {
            min_ops: 4,
            ..DetectOptions::default()
        };
        for allow in [false, true] {
            // repeat first, then fold
            let rw = rewrite(&detect(&c, &opts), allow);
            let f = phase_fold(&rw.circuit);
            let ph = rw.global_phase + f.global_phase;
            if rw.phase_exact {
                let d = compare(&c, &f.circuit, ph, &init);
                assert!(d < 1e-9, "#{it} repeat->fold: {d}");
            }
            // fold first, then repeat
            let f = phase_fold(&c);
            let rw = rewrite(&detect(&f.circuit, &opts), allow);
            if rw.phase_exact {
                let d = compare(&c, &rw.circuit, f.global_phase + rw.global_phase, &init);
                assert!(d < 1e-9, "#{it} fold->repeat: {d}");
            }
        }
    }
}

fn lower(c: &Circuit) -> Circuit {
    let mut o = Circuit::new(c.num_qubits);
    for op in &c.ops {
        match *op {
            Op::Gate(Gate::Ccx(a, b, t)) => {
                for g in toffoli_clifford_t(a, b, t) {
                    o.gate(g);
                }
            }
            op => o.ops.push(op),
        }
    }
    o
}

/// Folding moves merged rotations to their first occurrence, which can
/// break the bit-identical copies the repeat detector needs. Report the
/// repeat coverage before and after folding on the repeat workloads.
#[test]
fn audit_compose_fold_vs_repeat_coverage() {
    let opts = DetectOptions::default();
    let mut ladder_block = Circuit::new(5);
    ladder_block
        .h(0)
        .ccx(0, 1, 3)
        .ccx(3, 2, 4)
        .cnot(4, 0)
        .ccx(3, 2, 4)
        .ccx(0, 1, 3);
    let cases: Vec<(&str, Circuit)> = vec![
        ("trotter n=8 x200", trotter(8, 200, 0.05, 0.04)),
        ("grover n=5 x20", grover(5, 3, 20)),
        ("qec d=5 x100", qec_memory(5, 100, true)),
        (
            "toffoli block x50 (lowered)",
            lower(&repeated(&ladder_block, 50)),
        ),
    ];
    for (name, c) in cases {
        let before = detect(&c, &opts).report();
        let f = phase_fold(&c);
        let after = detect(&f.circuit, &opts).report();
        let nc = |c: &Circuit| c.gates().filter(|g| !g.is_clifford()).count();
        eprintln!(
            "{name}: gates {} -> {} (non-Clifford {} -> {}); repeat-saved gates {} -> {} (coverage {:.1}% -> {:.1}%)",
            c.num_gates(),
            f.circuit.num_gates(),
            nc(&c),
            nc(&f.circuit),
            before.saved_gates,
            after.saved_gates,
            100.0 * before.coverage(),
            100.0 * after.coverage()
        );
    }
}
