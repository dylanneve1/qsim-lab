//! Whole-run gate / Toffoli / measurement counts of the measurement-based
//! oracles vs windowed-opt, over all 2n controlled-U rounds of the record
//! instances (exp/mbu-shor). Usage:
//! `cargo run --release --example mbu_counts [N a] [w...]`
use qsim_lab::shor::sliced::{oracle_ops, SlicedProgram};
use qsim_lab::shor::{Instance, Oracle};
use qsim_lab::shor_mbu::MbuCounts;

fn run(n: u64, a: u64, oracle: Oracle) -> (MbuCounts, usize, usize) {
    let inst = Instance::new(n, a, oracle);
    let mut tot = MbuCounts::default();
    let mut steps = 0usize;
    for i in 0..inst.t {
        let (ops, _, nq) = oracle_ops(&inst, inst.mults[inst.t - 1 - i]);
        tot.add(&MbuCounts::of(&ops));
        steps += SlicedProgram::compile_ops(nq, &ops).unwrap().len();
    }
    (tot, steps, inst.qubits())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let insts: Vec<(u64, u64)> = if args.len() >= 2 {
        vec![(args[0].parse().unwrap(), args[1].parse().unwrap())]
    } else {
        vec![
            (1_005_973, 980_062),
            (10_161_323, 9_899_614),
            (221_643_407, 215_934_921),
            (1_537_596_787, 457_167_243),
        ]
    };
    let ws: Vec<usize> = if args.len() > 2 {
        args[2..].iter().map(|s| s.parse().unwrap()).collect()
    } else {
        vec![4]
    };
    for &(n, a) in &insts {
        for &w in &ws {
            let (base, bsteps, bq) = run(n, a, Oracle::WindowedOpt(w));
            println!("N={n} a={a} w={w}");
            println!(
                "  {:<18} qubits={bq:3} gates={:9} toffoli={:8} meas={:7} fixup={:7} slice_steps={:9}",
                "windowed-opt", base.total, base.toffoli, base.meas, base.fixup, bsteps
            );
            for (name, o) in [
                ("mbu-lookup", Oracle::WindowedMbuLookup(w)),
                ("mbu", Oracle::WindowedMbu(w)),
            ] {
                let (c, s, q) = run(n, a, o);
                let pct = |x: usize, y: usize| 100.0 * (x as f64 / y as f64 - 1.0);
                println!(
                    "  {name:<18} qubits={q:3} gates={:9} toffoli={:8} meas={:7} fixup={:7} slice_steps={s:9}  (gates {:+.1}%, toffoli {:+.1}%, steps {:+.1}%)  cnot={} x={}",
                    c.total,
                    c.toffoli,
                    c.meas,
                    c.fixup,
                    pct(c.total, base.total),
                    pct(c.toffoli, base.toffoli),
                    pct(s, bsteps),
                    c.cnot,
                    c.x
                );
            }
        }
    }
}
