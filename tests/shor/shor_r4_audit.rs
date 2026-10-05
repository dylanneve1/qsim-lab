//! Independent audit of round 4 (exp/shor-r4-audit): the sliced backend's
//! safety checks must actually fire (they are `assert!`, not logging), and the
//! sliced windowed run must match the dense textbook state-vector reference.

use qsim_lab::algorithms::gcd;
use qsim_lab::shor::sliced::{
    eval_block, eval_block_into, BlockOut, SliceIo, SlicedProgram, SlicedState,
};
use qsim_lab::shor::window::{controlled_ua, WindowLayout};
use qsim_lab::shor::{self, Instance, Oracle};
use qsim_lab::{Circuit, Gate, Op};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn panic_msg(r: std::thread::Result<()>) -> String {
    match r {
        Ok(()) => String::from("<no panic>"),
        Err(e) => e
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default(),
    }
}

fn block(n_mod: u64, a: u64, w: usize) -> (Circuit, SliceIo, Vec<u64>) {
    let lay = WindowLayout::new(shor::work_bits(n_mod), w);
    let c = controlled_ua(&lay, a, n_mod);
    let io = SliceIo {
        ctrl: lay.ctrl,
        x: lay.x.clone(),
    };
    (c, io, (0..n_mod).collect())
}

#[test]
fn compile_rejects_non_permutation_gates() {
    for g in [Gate::H(3), Gate::Phase(0, 0.3), Gate::Z(1), Gate::T(2)] {
        let mut c = Circuit::new(5);
        c.gate(Gate::Ccx(0, 1, 2));
        c.gate(g);
        assert!(SlicedProgram::compile(&c).is_err(), "{g:?} accepted");
    }
}

#[test]
fn untouched_circuit_passes_all_checks() {
    let (c, io, xs) = block(35, 2, 2);
    let prog = SlicedProgram::compile(&c).unwrap();
    let ys = eval_block(&prog, &io, true, &xs);
    for (x, y) in xs.iter().zip(&ys) {
        assert_eq!(*y, 2 * x % 35);
    }
    eval_block_into::<u64>(&prog, &io, false, &xs, BlockOut::Identity);
}

/// Dropping one uncompute Toffoli of a table lookup leaves an AND ancilla set:
/// the evaluator must panic, not silently drop the ancilla.
#[test]
fn dirty_ancilla_panics() {
    let (mut c, io, xs) = block(35, 2, 2);
    let lay = WindowLayout::new(shor::work_bits(35), 2);
    let last_and = c
        .ops
        .iter()
        .rposition(|op| matches!(op, Op::Gate(Gate::Ccx(_, _, t)) if lay.and.contains(t)))
        .unwrap();
    c.ops.remove(last_and);
    let prog = SlicedProgram::compile(&c).unwrap();
    let m = panic_msg(catch_unwind(AssertUnwindSafe(|| {
        eval_block(&prog, &io, true, &xs);
    })));
    assert!(m.contains("ancillas did not return to 0"), "got: {m}");
}

#[test]
fn changed_control_panics() {
    let (mut c, io, xs) = block(35, 2, 2);
    c.gate(Gate::X(0));
    let prog = SlicedProgram::compile(&c).unwrap();
    let m = panic_msg(catch_unwind(AssertUnwindSafe(|| {
        eval_block(&prog, &io, true, &xs);
    })));
    assert!(m.contains("control qubit changed"), "got: {m}");
}

/// An unconditional flip of a work bit is invisible to the ancilla check but
/// the control-0 half must catch it.
#[test]
fn control0_not_identity_panics() {
    let (mut c, io, xs) = block(35, 2, 2);
    c.gate(Gate::X(io.x[0]));
    let prog = SlicedProgram::compile(&c).unwrap();
    let m = panic_msg(catch_unwind(AssertUnwindSafe(|| {
        eval_block_into::<u64>(&prog, &io, false, &xs, BlockOut::Identity);
    })));
    assert!(m.contains("not the identity"), "got: {m}");
}

/// Sliced windowed (every gate of the X/CNOT/CCX circuit) vs the dense
/// textbook state vector (`algorithms::shor_full_state`, 3n qubits, full QFT).
#[test]
fn sliced_windowed_equals_dense_textbook_distribution() {
    for (n_mod, count) in [(15u64, 6), (21, 4), (33, 2), (35, 2), (39, 2)] {
        let bases: Vec<u64> = (2..n_mod - 1)
            .filter(|&a| gcd(a, n_mod) == 1)
            .take(count)
            .collect();
        for a in bases {
            let full = shor::full_qft_distribution(n_mod, a);
            for w in 1..=4 {
                let inst = Instance::new(n_mod, a, Oracle::Windowed(w));
                let d =
                    shor::semiclassical_distribution(&inst, SlicedState::<f64>::new(&inst), 0.0);
                let diff = full
                    .iter()
                    .zip(&d)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0, f64::max);
                assert!(diff < 1e-12, "N={n_mod} a={a} w={w}: {diff:e}");
            }
        }
    }
}
