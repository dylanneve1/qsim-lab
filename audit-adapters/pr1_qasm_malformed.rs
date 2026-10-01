//! PR 1 @ dd5528d: malformed OpenQASM input. Prints what from_qasm does; at dd5528d wrong arity, extra params, out-of-range index into the next qreg and duplicate operands are silently accepted.
use qsim_lab::circuit::Circuit;
#[test]
fn malformed() {
    for g in ["rz() q[0]", "cx q[0]", "u3(1) q[0]", "h q[0],q[1]", "rx(1,2) q[0]", "h q[2]", "cx q[0],q[0]", "h q", "barrier q"] {
        let s = format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nqreg r[2];\ncreg c[2];\n{g};\n");
        let r = Circuit::from_qasm(&s);
        println!("{g:>16} -> {:?}", r.as_ref().map(|c| format!("{:?}", c.ops)).map_err(|e| e.to_string()));
    }
}
