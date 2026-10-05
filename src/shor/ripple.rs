//! Gate-level modular arithmetic for Shor's algorithm using Cuccaro
//! ripple-carry adders (quant-ph/0410184) and Vedral–Barenco–Ekert (1996)
//! modular reduction.
//!
//! All operations are built strictly from reversible classical gates:
//! [`Gate::X`], [`Gate::Cnot`], and [`Gate::Ccx`] (Toffoli).
//!
//! Qubit layout for an `n`-bit modulus:
//! - Qubit 0: control
//! - Qubits `1..=n`: work register `x` (`n` qubits, LSB first)
//! - Qubits `n+1..=2n+1`: accumulator register `b` (`n + 1` qubits, LSB first)
//! - Qubits `2n+2..=3n+1`: constant register `a` (`n` qubits, LSB first)
//! - Qubit `3n+2`: Cuccaro carry-in `c0` (always 0 on entry and exit)
//! - Qubit `3n+3`: modular comparison flag `t` (always 0 on entry and exit)
//!
//! Total qubits: `3n + 4`.

use crate::circuit::{Circuit, Op};
use crate::engines::sparse::SparseState;
use crate::gate::Gate;

/// Qubit layout for an `n`-bit modulus in Cuccaro ripple-carry arithmetic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RippleLayout {
    pub n: usize,
    pub ctrl: usize,
    pub x: Vec<usize>,
    pub b: Vec<usize>,
    pub a: Vec<usize>,
    pub c0: usize,
    pub t: usize,
}

impl RippleLayout {
    pub fn new(n: usize) -> Self {
        assert!(n >= 1, "n must be at least 1");
        let ctrl = 0;
        let x: Vec<usize> = (1..=n).collect();
        let b: Vec<usize> = (n + 1..=2 * n + 1).collect();
        let a: Vec<usize> = (2 * n + 2..=3 * n + 1).collect();
        let c0 = 3 * n + 2;
        let t = 3 * n + 3;
        Self {
            n,
            ctrl,
            x,
            b,
            a,
            c0,
            t,
        }
    }

    /// Total number of qubits: `3n + 4`.
    pub fn num_qubits(&self) -> usize {
        3 * self.n + 4
    }

    /// Bitmask of all ancilla qubits that must be 0 before and after modular multiplication.
    /// Ancilla qubits are:
    /// - accumulator `b`: qubits `n+1..=2n+1`
    /// - constant register `a`: qubits `2n+2..=3n+1`
    /// - carry-in `c0`: qubit `3n+2`
    /// - modular flag `t`: qubit `3n+3`
    pub fn ancilla_mask(&self) -> u64 {
        let mut mask = 0u64;
        for &q in &self.b {
            mask |= 1u64 << q;
        }
        for &q in &self.a {
            mask |= 1u64 << q;
        }
        mask |= 1u64 << self.c0;
        mask |= 1u64 << self.t;
        mask
    }
}

/// In-place Cuccaro ripple-carry addition:
/// `(a, b, c0=0) -> (a, (b + a) mod 2^(n+1), c0=0)`.
/// `a` has `n` qubits, `b` has `n + 1` qubits, `c0` is 1 ancilla qubit.
/// `a` and `c0` are restored to their initial values.
pub fn cuccaro_add(c: &mut Circuit, a: &[usize], b: &[usize], c0: usize) {
    let n = a.len();
    assert_eq!(b.len(), n + 1, "b must have n + 1 qubits");
    if n == 0 {
        return;
    }

    // MAJ(x, y, z): CNOT(z, y), CNOT(z, x), CCX(x, y, z)
    let maj = |c: &mut Circuit, x: usize, y: usize, z: usize| {
        c.cnot(z, y).cnot(z, x).gate(Gate::Ccx(x, y, z));
    };
    // UMA(x, y, z): CCX(x, y, z), CNOT(z, x), CNOT(x, y)
    let uma = |c: &mut Circuit, x: usize, y: usize, z: usize| {
        c.gate(Gate::Ccx(x, y, z)).cnot(z, x).cnot(x, y);
    };

    // Forward ripple pass: carries compute up into a[i]
    maj(c, c0, b[0], a[0]);
    for i in 1..n {
        maj(c, a[i - 1], b[i], a[i]);
    }

    // Carry out into b[n]
    c.cnot(a[n - 1], b[n]);

    // Backward uncompute pass: restores a[i] and computes sum into b[i]
    for i in (1..n).rev() {
        uma(c, a[i - 1], b[i], a[i]);
    }
    uma(c, c0, b[0], a[0]);
}

/// Inverse of [`cuccaro_add`]:
/// `(a, b, c0=0) -> (a, (b - a) mod 2^(n+1), c0=0)`.
pub fn cuccaro_sub(c: &mut Circuit, a: &[usize], b: &[usize], c0: usize) {
    let n = a.len();
    assert_eq!(b.len(), n + 1, "b must have n + 1 qubits");
    if n == 0 {
        return;
    }

    let uma_inv = |c: &mut Circuit, x: usize, y: usize, z: usize| {
        c.cnot(x, y).cnot(z, x).gate(Gate::Ccx(x, y, z));
    };
    let maj_inv = |c: &mut Circuit, x: usize, y: usize, z: usize| {
        c.gate(Gate::Ccx(x, y, z)).cnot(z, x).cnot(z, y);
    };

    // Forward pass
    uma_inv(c, c0, b[0], a[0]);
    for i in 1..n {
        uma_inv(c, a[i - 1], b[i], a[i]);
    }

    // Carry out into b[n]
    c.cnot(a[n - 1], b[n]);

    // Backward pass
    for i in (1..n).rev() {
        maj_inv(c, a[i - 1], b[i], a[i]);
    }
    maj_inv(c, c0, b[0], a[0]);
}

/// Loads a classical constant `val` into register `a` conditioned on `ctrls`.
/// Since each gate is self-inverse, calling this twice uncomputes `a` back to 0.
pub fn load_constant(c: &mut Circuit, a: &[usize], val: u64, ctrls: &[usize]) {
    for (j, &q) in a.iter().enumerate() {
        if (val >> j) & 1 == 1 {
            match ctrls {
                [] => {
                    c.x(q);
                }
                [c1] => {
                    c.cnot(*c1, q);
                }
                [c1, c2] => {
                    c.gate(Gate::Ccx(*c1, *c2, q));
                }
                _ => panic!("load_constant supports at most 2 controls"),
            }
        }
    }
}

/// Doubly or singly controlled modular adder:
/// `b -> (b + val) mod N` on `0 <= b < N`, conditioned on `ctrls`.
/// If `ctrls` is false, `b -> b`.
/// All ancilla qubits (`a`, `c0`, `t`, `b[n]`) start at 0 and return to 0.
pub fn add_mod(c: &mut Circuit, lay: &RippleLayout, val: u64, n_mod: u64, ctrls: &[usize]) {
    let a = &lay.a;
    let b = &lay.b;
    let c0 = lay.c0;
    let t = lay.t;
    let bn = b[lay.n];

    // 1. Add val to b (controlled on ctrls)
    load_constant(c, a, val, ctrls);
    cuccaro_add(c, a, b, c0);
    load_constant(c, a, val, ctrls);

    // 2. Subtract N from b (unconditional)
    load_constant(c, a, n_mod, &[]);
    cuccaro_sub(c, a, b, c0);
    load_constant(c, a, n_mod, &[]);

    // 3. Flag t = bn (indicates underflow: t=1 if b + val < N, t=0 if b + val >= N)
    c.cnot(bn, t);

    // 4. If t == 1, add N back to b (controlled on t)
    load_constant(c, a, n_mod, &[t]);
    cuccaro_add(c, a, b, c0);
    load_constant(c, a, n_mod, &[t]);

    // 5. Subtract val from b (controlled on ctrls)
    load_constant(c, a, val, ctrls);
    cuccaro_sub(c, a, b, c0);
    load_constant(c, a, val, ctrls);

    // 6. Uncompute flag t using NOT(bn)
    c.x(bn);
    c.cnot(bn, t);
    c.x(bn);

    // 7. Add val back to b (controlled on ctrls)
    load_constant(c, a, val, ctrls);
    cuccaro_add(c, a, b, c0);
    load_constant(c, a, val, ctrls);
}

/// Controlled modular multiplier:
/// `|ctrl>|x>|b=0> -> |ctrl>|x>|ctrl · a · x mod N>`.
pub fn cmult(lay: &RippleLayout, ctrl: usize, a: u64, n_mod: u64) -> Circuit {
    let mut c = Circuit::new(lay.num_qubits());
    let mut cur = a % n_mod;
    for &xi in &lay.x {
        add_mod(&mut c, lay, cur, n_mod, &[ctrl, xi]);
        cur = (u128::from(cur) * 2 % u128::from(n_mod)) as u64;
    }
    c
}

/// Controlled modular multiplication block `U_a`:
/// `|ctrl>|x>|0> -> |ctrl>|a^ctrl · x mod N>|0>` for `x < N`.
/// All ancilla qubits start at 0 and return to 0.
pub fn controlled_ua(lay: &RippleLayout, ctrl: usize, a: u64, n_mod: u64) -> Circuit {
    let inv = crate::shor::mod_inverse(a, n_mod);
    let mut c = cmult(lay, ctrl, a, n_mod);
    for i in 0..lay.n {
        let (x, b) = (lay.x[i], lay.b[i]);
        // CSWAP(ctrl, x, b)
        c.cnot(b, x);
        c.gate(Gate::Ccx(ctrl, x, b));
        c.cnot(b, x);
    }
    c.append(&cmult(lay, ctrl, inv, n_mod).inverse());
    c
}

/// Total gate count and Toffoli count in a circuit.
pub fn gate_counts(c: &Circuit) -> (usize, usize) {
    let mut total = 0;
    let mut toffoli = 0;
    for op in &c.ops {
        if let Op::Gate(g) = op {
            total += 1;
            if matches!(g, Gate::Ccx(..)) {
                toffoli += 1;
            }
        }
    }
    (total, toffoli)
}

/// Evaluates a single reversible classical gate on a 64-bit basis index.
#[inline(always)]
pub fn eval_gate_on_key(k: u64, g: &Gate) -> u64 {
    match *g {
        Gate::X(q) => k ^ (1u64 << q),
        Gate::Cnot(c, t) => {
            let bit = (k >> c) & 1;
            k ^ (bit << t)
        }
        Gate::Ccx(c1, c2, t) => {
            let bit = ((k >> c1) & (k >> c2)) & 1;
            k ^ (bit << t)
        }
        Gate::Swap(q1, q2) => {
            let diff = ((k >> q1) ^ (k >> q2)) & 1;
            k ^ ((diff << q1) | (diff << q2))
        }
        _ => panic!("eval_gate_on_key only supports reversible permutation gates"),
    }
}

/// Evaluates a whole circuit of reversible classical gates on a 64-bit basis index.
#[inline]
pub fn eval_circuit_on_key(mut k: u64, c: &Circuit) -> u64 {
    for op in &c.ops {
        match op {
            Op::Gate(g) => k = eval_gate_on_key(k, g),
            _ => panic!("eval_circuit_on_key only supports gate ops"),
        }
    }
    k
}

/// Evaluates a reversible circuit block on each basis key of a [`SparseState`]
/// in parallel across keys using Rayon, verifying that all ancillas return to 0.
pub fn apply_reversible_block(state: &mut SparseState, c: &Circuit, ancilla_mask: u64) {
    state.apply_permutation_par(|k| {
        let k_new = eval_circuit_on_key(k, c);
        assert_eq!(
            k_new & ancilla_mask,
            0,
            "ancillas did not return to 0! k={k:x} -> k_new={k_new:x}, ancillas={:x}",
            k_new & ancilla_mask
        );
        k_new
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::gcd;

    #[test]
    fn cuccaro_adder_and_subtractor_exhaustive_small() {
        for n in 1..=6 {
            let a_qs: Vec<usize> = (0..n).collect();
            let b_qs: Vec<usize> = (n..2 * n + 1).collect();
            let c0 = 2 * n + 1;
            let num_q = 2 * n + 2;

            let mut add_c = Circuit::new(num_q);
            cuccaro_add(&mut add_c, &a_qs, &b_qs, c0);

            let mut sub_c = Circuit::new(num_q);
            cuccaro_sub(&mut sub_c, &a_qs, &b_qs, c0);

            let modulus_b = 1u64 << (n + 1);
            let modulus_a = 1u64 << n;

            for a_val in 0..modulus_a {
                for b_val in 0..modulus_b {
                    let k = a_val | (b_val << n);

                    // Test Add
                    let k_after_add = eval_circuit_on_key(k, &add_c);
                    let a_out = k_after_add & (modulus_a - 1);
                    let b_out = (k_after_add >> n) & (modulus_b - 1);
                    let c0_out = (k_after_add >> (2 * n + 1)) & 1;

                    assert_eq!(a_out, a_val, "a preserved");
                    assert_eq!(c0_out, 0, "c0 preserved");
                    assert_eq!(b_out, (b_val + a_val) % modulus_b, "sum correct");

                    // Test Sub
                    let k_after_sub = eval_circuit_on_key(k, &sub_c);
                    let a_out_sub = k_after_sub & (modulus_a - 1);
                    let b_out_sub = (k_after_sub >> n) & (modulus_b - 1);
                    let c0_out_sub = (k_after_sub >> (2 * n + 1)) & 1;

                    assert_eq!(a_out_sub, a_val, "a preserved");
                    assert_eq!(c0_out_sub, 0, "c0 preserved");
                    assert_eq!(
                        b_out_sub,
                        (b_val + modulus_b - a_val) % modulus_b,
                        "difference correct"
                    );

                    // Round-trip
                    let k_rt = eval_circuit_on_key(k_after_add, &sub_c);
                    assert_eq!(k_rt, k, "add then sub is identity");
                }
            }
        }
    }

    #[test]
    fn add_mod_exhaustive_small() {
        for n in 2..=6 {
            let lay = RippleLayout::new(n);
            let n_mod = (1u64 << n) - 1; // odd modulus

            for val in 0..n_mod {
                // Test uncontrolled add_mod
                let mut c_uncontrolled = Circuit::new(lay.num_qubits());
                add_mod(&mut c_uncontrolled, &lay, val, n_mod, &[]);

                // Test controlled add_mod (ctrl=qubit 0)
                let mut c_controlled = Circuit::new(lay.num_qubits());
                add_mod(&mut c_controlled, &lay, val, n_mod, &[lay.ctrl]);

                for b_val in 0..n_mod {
                    // 1. Uncontrolled
                    let k_in = b_val << (lay.n + 1);
                    let k_out = eval_circuit_on_key(k_in, &c_uncontrolled);
                    assert_eq!(
                        k_out & lay.ancilla_mask() & !(((1u64 << (lay.n + 1)) - 1) << (lay.n + 1)),
                        0
                    );
                    let b_out = (k_out >> (lay.n + 1)) & ((1u64 << (lay.n + 1)) - 1);
                    assert_eq!(b_out, (b_val + val) % n_mod);
                    // ancilla clean
                    assert_eq!(
                        k_out & lay.ancilla_mask(),
                        (b_out << (lay.n + 1)) & lay.ancilla_mask()
                    );

                    // 2. Controlled with ctrl=0
                    let k_ctrl0 = k_in;
                    let k_ctrl0_out = eval_circuit_on_key(k_ctrl0, &c_controlled);
                    assert_eq!(k_ctrl0_out, k_ctrl0, "ctrl=0 should be identity");

                    // 3. Controlled with ctrl=1
                    let k_ctrl1 = k_in | 1;
                    let k_ctrl1_out = eval_circuit_on_key(k_ctrl1, &c_controlled);
                    let b_ctrl1_out = (k_ctrl1_out >> (lay.n + 1)) & ((1u64 << (lay.n + 1)) - 1);
                    assert_eq!(b_ctrl1_out, (b_val + val) % n_mod);
                    assert_eq!(k_ctrl1_out & 1, 1);
                }
            }
        }
    }

    #[test]
    fn controlled_ua_exhaustive_small() {
        for n_mod in [15u64, 21, 33, 35, 55, 63] {
            let n = crate::shor::work_bits(n_mod);
            let lay = RippleLayout::new(n);

            for a in 2..n_mod {
                if gcd(a, n_mod) != 1 {
                    continue;
                }
                let c = controlled_ua(&lay, lay.ctrl, a, n_mod);

                for ctrl in [0u64, 1] {
                    for x in 1..n_mod {
                        let k_in = ctrl | (x << 1);
                        let k_out = eval_circuit_on_key(k_in, &c);

                        // All ancillas must be 0
                        assert_eq!(
                            k_out & lay.ancilla_mask(),
                            0,
                            "ancillas dirty: N={n_mod} a={a} ctrl={ctrl} x={x}"
                        );

                        let ctrl_out = k_out & 1;
                        let x_out = (k_out >> 1) & ((1u64 << n) - 1);

                        assert_eq!(ctrl_out, ctrl);
                        let expected_x = if ctrl == 1 { (x * a) % n_mod } else { x };
                        assert_eq!(
                            x_out, expected_x,
                            "N={n_mod} a={a} ctrl={ctrl} x={x}: got {x_out}, expected {expected_x}"
                        );
                    }
                }
            }
        }
    }
}
