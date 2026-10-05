//! Gate-level modular arithmetic for Shor's algorithm, after Beauregard,
//! "Circuit for Shor's algorithm using 2n+3 qubits" (2003):
//!
//! * `φADD(a)`: add a classical constant to a register held in Fourier
//!   space (Draper's adder) — one phase rotation per qubit, optionally with
//!   one or two quantum controls;
//! * `φADD(a)MOD(N)`: doubly controlled modular adder (5 adders, 4 QFTs, one
//!   ancilla that is returned to `|0>`);
//! * `CMULT(a)MOD(N)`: `|c>|x>|b> -> |c>|x>|b + c·a·x mod N>`;
//! * controlled `U_a`: `CMULT(a)`, controlled SWAP, `CMULT(a^-1)^-1`, which
//!   maps `|c>|x>|0> -> |c>|a^c x mod N>|0>` for `x < N`.
//!
//! Only `Phase`, `CPhase`, `H`, `X`, `Cnot` and `Ccx` gates are emitted; the
//! doubly controlled phases are decomposed exactly into `CPhase` + `Cnot`.
//! The QFTs omit the final SWAPs, so in Fourier space qubit `j` of a
//! register carries the phase `2π b / 2^(j+1)`.

use crate::circuit::Circuit;
use crate::gate::Gate;
use std::f64::consts::PI;

/// Qubit layout for an `n`-bit modulus: the work register `x` is qubits
/// `1..=n`, the accumulator `b` is `n+1..=2n+1` (`n + 1` qubits, LSB first),
/// the ancilla is `2n + 2`. Qubit 0 is left for the control.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BeauregardLayout {
    /// Width of the modulus (work-register bits).
    pub n: usize,
    /// Work-register qubits, LSB first (`1..=n`).
    pub x: Vec<usize>,
    /// Fourier-space accumulator qubits, LSB first (`n+1..=2n+1`).
    pub b: Vec<usize>,
    /// Ancilla qubit of the modular adder (`2n + 2`).
    pub anc: usize,
}

impl BeauregardLayout {
    /// Layout for an `n`-bit modulus.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            x: (1..=n).collect(),
            b: (n + 1..=2 * n + 1).collect(),
            anc: 2 * n + 2,
        }
    }

    /// Total qubits including the control: `2n + 3`.
    pub fn num_qubits(&self) -> usize {
        2 * self.n + 3
    }
}

/// QFT without the final bit-reversal SWAPs on `qs` (`qs[0]` = LSB).
pub fn qft_noswap(c: &mut Circuit, qs: &[usize]) {
    for j in (0..qs.len()).rev() {
        c.h(qs[j]);
        for k in (0..j).rev() {
            c.cphase(qs[k], qs[j], PI / (1u64 << (j - k)) as f64);
        }
    }
}

/// Inverse of [`qft_noswap`].
pub fn iqft_noswap(c: &mut Circuit, qs: &[usize]) {
    for j in 0..qs.len() {
        for k in 0..j {
            c.cphase(qs[k], qs[j], -PI / (1u64 << (j - k)) as f64);
        }
        c.h(qs[j]);
    }
}

/// The rotation angles of `φADD(sign · a)` on an `l`-qubit Fourier
/// register: qubit `j` gets `2π (a mod 2^(j+1)) / 2^(j+1)` (zero angles are
/// skipped by the callers).
fn add_angles(a: u64, l: usize, negate: bool) -> Vec<f64> {
    (0..l)
        .map(|j| {
            let modulus = 1u128 << (j + 1);
            let r = (u128::from(a) % modulus) as f64;
            let th = 2.0 * PI * r / modulus as f64;
            if negate {
                -th
            } else {
                th
            }
        })
        .collect()
}

/// `φADD(±a)` on the Fourier-space register `qs`, controlled by zero, one
/// or two qubits.
pub fn phi_add(c: &mut Circuit, qs: &[usize], a: u64, negate: bool, controls: &[usize]) {
    let th = add_angles(a, qs.len(), negate);
    let nz = || th.iter().zip(qs).filter(|(t, _)| **t != 0.0);
    match *controls {
        [] => {
            for (&t, &q) in nz() {
                c.phase(q, t);
            }
        }
        [c1] => {
            for (&t, &q) in nz() {
                c.cphase(c1, q, t);
            }
        }
        [c1, c2] => {
            // CCP(θ)(c1,c2,q) = CP(θ/2)(c2,q) CX(c1,c2) CP(-θ/2)(c2,q)
            //                   CX(c1,c2) CP(θ/2)(c1,q);
            // the phases on different targets commute, so the two CNOTs are
            // shared by the whole register.
            if nz().next().is_none() {
                return;
            }
            for (&t, &q) in nz() {
                c.cphase(c2, q, t / 2.0);
            }
            c.cnot(c1, c2);
            for (&t, &q) in nz() {
                c.cphase(c2, q, -t / 2.0);
            }
            c.cnot(c1, c2);
            for (&t, &q) in nz() {
                c.cphase(c1, q, t / 2.0);
            }
        }
        _ => panic!("phi_add supports at most two controls"),
    }
}

/// Doubly controlled `φADD(a)MOD(N)` on the Fourier-space accumulator
/// (requires `a < N`, `b < N` on input, ancilla `|0>`; leaves the ancilla in
/// `|0>`).
pub fn phi_add_mod(
    c: &mut Circuit,
    lay: &BeauregardLayout,
    a: u64,
    n_mod: u64,
    c1: usize,
    c2: usize,
) {
    let b = &lay.b;
    let msb = *b.last().expect("non-empty");
    phi_add(c, b, a, false, &[c1, c2]);
    phi_add(c, b, n_mod, true, &[]);
    iqft_noswap(c, b);
    c.cnot(msb, lay.anc);
    qft_noswap(c, b);
    phi_add(c, b, n_mod, false, &[lay.anc]);
    phi_add(c, b, a, true, &[c1, c2]);
    iqft_noswap(c, b);
    c.x(msb).cnot(msb, lay.anc).x(msb);
    qft_noswap(c, b);
    phi_add(c, b, a, false, &[c1, c2]);
}

/// `CMULT(a)MOD(N)`: `|ctrl>|x>|b> -> |ctrl>|x>|b + ctrl·a·x mod N>`.
pub fn cmult(lay: &BeauregardLayout, ctrl: usize, a: u64, n_mod: u64) -> Circuit {
    let mut c = Circuit::new(lay.num_qubits());
    qft_noswap(&mut c, &lay.b);
    let mut ai = a % n_mod;
    for &xi in &lay.x {
        phi_add_mod(&mut c, lay, ai, n_mod, ctrl, xi);
        ai = (u128::from(ai) * 2 % u128::from(n_mod)) as u64;
    }
    iqft_noswap(&mut c, &lay.b);
    c
}

/// Controlled `U_a`: `|ctrl>|x>|0>|0> -> |ctrl>|a^ctrl · x mod N>|0>|0>` for
/// `x < N`, as a gate-level circuit on `2n + 3` qubits.
pub fn controlled_ua(lay: &BeauregardLayout, ctrl: usize, a: u64, n_mod: u64) -> Circuit {
    let inv = crate::shor::mod_inverse(a, n_mod);
    let mut c = cmult(lay, ctrl, a, n_mod);
    for i in 0..lay.n {
        let (x, b) = (lay.x[i], lay.b[i]);
        // Fredkin(ctrl; x, b)
        c.cnot(b, x);
        c.gate(Gate::Ccx(ctrl, x, b));
        c.cnot(b, x);
    }
    c.append(&cmult(lay, ctrl, inv, n_mod).inverse());
    c
}
