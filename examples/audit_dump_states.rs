//! Audit helper (research/process/audit.md §16): builds a magic-atlas instance and
//! writes the dense state after every original gate (raw little-endian
//! f64 re,im pairs, 2^n per gate), for an independent stabilizer-nullity
//! computation in Python.
//! `cargo run --release --example audit_dump_states -- SPEC OUT.bin` (or OUT.qasm: the circuit)
use qsim_lab::magic_atlas::families;
use qsim_lab::{Op, StateVectorF64};
use std::io::Write;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let c = families::build(&a[0], 1).unwrap();
    if a[1].ends_with(".qasm") {
        std::fs::write(&a[1], c.to_qasm().unwrap()).unwrap();
        return;
    }
    let mut sv = StateVectorF64::new(c.num_qubits);
    let mut f = std::io::BufWriter::new(std::fs::File::create(&a[1]).unwrap());
    let mut k = 0usize;
    for op in &c.ops {
        let Op::Gate(g) = op else {
            panic!("non-gate op {op:?}")
        };
        sv.apply_gate(g).unwrap();
        for z in sv.amplitudes() {
            f.write_all(&z.re.to_le_bytes()).unwrap();
            f.write_all(&z.im.to_le_bytes()).unwrap();
        }
        k += 1;
    }
    eprintln!("n {} gates {}", c.num_qubits, k);
}
