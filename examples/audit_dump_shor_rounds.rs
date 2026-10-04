//! Audit helper (research/audit.md §16): dumps the per-round windowed oracle
//! blocks of a Shor instance as plain text, so an independent (Python)
//! noisy simulator can replay them.
//! `cargo run --release --example audit_dump_shor_rounds -- N a w > out.txt`
use qsim_lab::shor::sliced::oracle_block;
use qsim_lab::shor::{Instance, Oracle};
use qsim_lab::{Gate, Op};

fn main() {
    let a: Vec<u64> = std::env::args()
        .skip(1)
        .map(|s| s.parse().unwrap())
        .collect();
    let inst = Instance::new(a[0], a[1], Oracle::Windowed(a[2] as usize));
    println!("nq {} t {} m {}", inst.qubits(), inst.t, inst.m);
    for i in 0..inst.t {
        let mult = inst.mults[inst.t - 1 - i];
        let (c, io) = oracle_block(&inst, mult);
        println!(
            "round {i} mult {mult} ctrl {} x {:?} gates {}",
            io.ctrl,
            io.x,
            c.ops.len()
        );
        for op in &c.ops {
            match op {
                Op::Gate(Gate::X(t)) => println!("X {t}"),
                Op::Gate(Gate::Cnot(c, t)) => println!("CX {c} {t}"),
                Op::Gate(Gate::Ccx(a, b, t)) => println!("CCX {a} {b} {t}"),
                Op::Gate(Gate::Swap(a, b)) => println!("SWAP {a} {b}"),
                o => panic!("{o:?}"),
            }
        }
    }
}
