//! Why one T gate matters: the stabilizer tableau refuses it, and the
//! Pauli-path simulator's cost doubles (at most) with each one.

use qsim_lab::circuit::Circuit;
use qsim_lab::pauli_path::{self, PauliSum};
use qsim_lab::{Gate, Tableau};

fn main() {
    let n = 50;
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    for q in 0..n - 1 {
        c.cnot(q, q + 1);
    }
    let mut t = Tableau::new(n);
    for g in c.gates() {
        t.apply_gate(g).unwrap();
    }
    println!("Clifford circuit on {n} qubits: the tableau handles it.");
    println!("Adding T(0): {}", t.apply_gate(&Gate::T(0)).unwrap_err());

    // A layered circuit where each T acts on a qubit carrying X/Y terms.
    println!("\nT gates  <X_0 ... X_9>   Pauli terms");
    for layers in 0..=12 {
        let mut c = Circuit::new(n);
        for l in 0..layers {
            for q in 0..10 {
                c.h(q);
            }
            c.t(l % 10);
            for q in 0..9 {
                c.cnot(q, q + 1);
            }
        }
        let obs = PauliSum::from_str_single(&format!("{}{}", "X".repeat(10), "I".repeat(n - 10)));
        let (v, st) = pauli_path::expectation(&c, &obs, 1 << 22).unwrap();
        println!("{:7}  {v:+.6}       {}", c.t_count(), st.peak_terms);
    }
}
