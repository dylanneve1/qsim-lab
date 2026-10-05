#![allow(clippy::field_reassign_with_default)]
mod audit_common;
use audit_common::*;
use qsim_lab::blocked::BlockConfig;
use qsim_lab::Gate::*;
use qsim_lab::{Circuit, Gate, StateVectorF64};

fn delta(gs: &[Gate], n: usize, cfg: &BlockConfig) -> f64 {
    let mut c = Circuit::new(n);
    for g in gs { c.gate(*g); }
    let r = RefSv::run(&c);
    let mut sv = StateVectorF64::new(n);
    sv.apply_circuit_blocked(&c, cfg).unwrap();
    max_amp_diff(&r.a, (0..1 << n).map(|i| sv.amplitude(i)))
}

#[test]
fn variants() {
    let full = vec![Y(0), Cnot(0, 4), Rx(0, 0.7853981633974493), Z(0), H(0), X(0), S(0), Z(0), T(0), H(0)];
    let cases: Vec<(&str, Vec<Gate>)> = vec![
        ("full", full.clone()),
        ("no Y0", full[1..].to_vec()),
        ("no cnot", [&full[..1], &full[2..]].concat()),
        ("1q run only after |+>", vec![H(0), Rx(0, 0.7853981633974493), Z(0), H(0), X(0), S(0), Z(0), T(0), H(0)]),
        ("run only", full[2..].to_vec()),
    ];
    for (name, gs) in &cases {
        for n in [1usize, 2, 5] {
            if gs.iter().any(|g| g.qubits().iter().any(|&q| q >= n)) { continue; }
            for (fuse, split, sched) in [(true, true, true), (false, true, true), (true, false, true), (true, true, false), (true, false, false)] {
                let mut cfg = BlockConfig::default();
                cfg.fuse_1q = fuse; cfg.split_phases = split; cfg.schedule_diag = sched;
                let d = delta(gs, n, &cfg);
                if d > 1e-9 { eprintln!("{name} n={n} fuse={fuse} split={split} sched={sched}: Δ={d:.3e}"); }
            }
        }
    }
    eprintln!("done");
}
