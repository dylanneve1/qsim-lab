//! Windowed gate-level modular multiplication (Gidney 2019, "Windowed
//! quantum arithmetic", arXiv:1905.07682), built only from X, CNOT and CCX.
//!
//! The ripple oracle ([`crate::shor::ripple`]) does one doubly-controlled
//! modular addition per bit of `x`. Here the bits of `x` are grouped into
//! windows of `w` bits; per window a *table lookup* (a QROM: for every
//! address `v`, an AND-chain computes `[ctrl ∧ x_window = v]` into an
//! ancilla, which CNOTs the classical constant `T[v] = v·a·2^(kw) mod N`
//! into a lookup register `L`) is followed by ONE quantum–quantum modular
//! addition `b += L mod N` and the inverse lookup. That replaces `w`
//! modular additions with one, at the price of two lookups of `2^w`
//! entries.
//!
//! Qubit layout for an `n`-bit modulus and window `w`:
//! - qubit 0: control;
//! - `1..=n`: work register `x`;
//! - `n+1..=2n+1`: accumulator `b` (`n + 1` qubits);
//! - `2n+2..=3n+1`: lookup register `L`;
//! - `3n+2..=4n+1`: constant register `K` (holds `N` during the reduction);
//! - `4n+2`: Cuccaro carry-in `c0`; `4n+3`: comparison flag `t`;
//! - `4n+4..4n+4+w`: AND-chain ancillas of the lookup.
//!
//! Total `4n + 4 + w` qubits. All ancillas return to 0.

use crate::circuit::Circuit;
use crate::gate::Gate;
use crate::shor::ripple::{cuccaro_add, cuccaro_sub, load_constant};

/// Qubit layout of the windowed multiplier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowLayout {
    pub n: usize,
    pub w: usize,
    pub ctrl: usize,
    pub x: Vec<usize>,
    pub b: Vec<usize>,
    pub l: Vec<usize>,
    pub k: Vec<usize>,
    pub c0: usize,
    pub t: usize,
    pub and: Vec<usize>,
}

impl WindowLayout {
    pub fn new(n: usize, w: usize) -> Self {
        assert!(n >= 1 && w >= 1, "n and w must be positive");
        let w = w.min(n);
        Self {
            n,
            w,
            ctrl: 0,
            x: (1..=n).collect(),
            b: (n + 1..=2 * n + 1).collect(),
            l: (2 * n + 2..=3 * n + 1).collect(),
            k: (3 * n + 2..=4 * n + 1).collect(),
            c0: 4 * n + 2,
            t: 4 * n + 3,
            and: (4 * n + 4..4 * n + 4 + w).collect(),
        }
    }

    pub fn num_qubits(&self) -> usize {
        4 * self.n + 4 + self.w
    }
}

/// Table lookup: `L ^= T[v]` where `v` is the value of the address qubits
/// `addr` (LSB first), conditioned on `ctrl`. Self-inverse.
///
/// For each address `v` the address bits that must be 0 are flipped with X,
/// an AND chain `and[0] = ctrl ∧ addr[0]`, `and[j] = and[j-1] ∧ addr[j]` is
/// computed with Toffolis, `and[last]` is CNOT-ed into the set bits of
/// `T[v]`, and the chain is uncomputed. Consecutive addresses share the
/// common chain prefix (only the suffix that changes is recomputed), and
/// X flips are only emitted when a bit's required polarity changes.
pub fn lookup(
    c: &mut Circuit,
    ctrl: usize,
    addr: &[usize],
    and: &[usize],
    l: &[usize],
    table: &[u64],
) {
    let w = addr.len();
    assert!(w >= 1 && and.len() >= w && table.len() == 1 << w);
    // flipped[j]: addr[j] currently X-ed (so it reads 1 when the bit is 0)
    let mut flipped = vec![false; w];
    // depth of the AND chain currently computed (and[0..depth] valid)
    let mut depth = 0usize;
    let mut chain_v: u64 = 0; // address prefix the chain was computed for
    for (v, &val) in table.iter().enumerate() {
        let v = v as u64;
        if val == 0 {
            continue;
        }
        // longest prefix (bits 0..p) shared with the computed chain
        let mut p = 0;
        while p < depth && ((v >> p) & 1) == ((chain_v >> p) & 1) {
            p += 1;
        }
        // uncompute chain levels p..depth (top down)
        for j in (p..depth).rev() {
            let prev = if j == 0 { ctrl } else { and[j - 1] };
            c.gate(Gate::Ccx(prev, addr[j], and[j]));
        }
        // set polarities for bits p..w and compute the chain
        for j in p..w {
            let need = (v >> j) & 1 == 0;
            if flipped[j] != need {
                c.x(addr[j]);
                flipped[j] = need;
            }
            let prev = if j == 0 { ctrl } else { and[j - 1] };
            c.gate(Gate::Ccx(prev, addr[j], and[j]));
        }
        depth = w;
        chain_v = v;
        for (i, &q) in l.iter().enumerate() {
            if (val >> i) & 1 == 1 {
                c.cnot(and[w - 1], q);
            }
        }
    }
    for j in (0..depth).rev() {
        let prev = if j == 0 { ctrl } else { and[j - 1] };
        c.gate(Gate::Ccx(prev, addr[j], and[j]));
    }
    for (j, f) in flipped.iter().enumerate() {
        if *f {
            c.x(addr[j]);
        }
    }
}

/// `b -> (b + L) mod N` for `b, L < N`, with `K`, `c0`, `t`, `b[n]` clean
/// on entry and exit (VBE / Beauregard structure with ripple adders).
pub fn add_mod_reg(c: &mut Circuit, lay: &WindowLayout, n_mod: u64) {
    let (l, b, k, c0, t) = (&lay.l, &lay.b, &lay.k, lay.c0, lay.t);
    let bn = b[lay.n];
    cuccaro_add(c, l, b, c0);
    load_constant(c, k, n_mod, &[]);
    cuccaro_sub(c, k, b, c0);
    load_constant(c, k, n_mod, &[]);
    c.cnot(bn, t);
    load_constant(c, k, n_mod, &[t]);
    cuccaro_add(c, k, b, c0);
    load_constant(c, k, n_mod, &[t]);
    cuccaro_sub(c, l, b, c0);
    c.x(bn);
    c.cnot(bn, t);
    c.x(bn);
    cuccaro_add(c, l, b, c0);
}

/// Controlled windowed multiply-add: `|c>|x>|b=0> -> |c>|x>|c·a·x mod N>`.
pub fn cmult(lay: &WindowLayout, a: u64, n_mod: u64) -> Circuit {
    let mut c = Circuit::new(lay.num_qubits());
    let n = lay.n;
    let mut start = 0;
    // a · 2^start mod N
    let mut base = a % n_mod;
    while start < n {
        let w = lay.w.min(n - start);
        let table: Vec<u64> = (0..1u64 << w)
            .map(|v| (u128::from(v) * u128::from(base) % u128::from(n_mod)) as u64)
            .collect();
        let addr = &lay.x[start..start + w];
        lookup(&mut c, lay.ctrl, addr, &lay.and, &lay.l, &table);
        add_mod_reg(&mut c, lay, n_mod);
        lookup(&mut c, lay.ctrl, addr, &lay.and, &lay.l, &table);
        for _ in 0..w {
            base = (u128::from(base) * 2 % u128::from(n_mod)) as u64;
        }
        start += w;
    }
    c
}

/// Controlled `U_a`: `|c>|x>|0…> -> |c>|a^c x mod N>|0…>` for `x < N`.
pub fn controlled_ua(lay: &WindowLayout, a: u64, n_mod: u64) -> Circuit {
    let inv = crate::shor::mod_inverse(a, n_mod);
    let mut c = cmult(lay, a, n_mod);
    for i in 0..lay.n {
        let (x, b) = (lay.x[i], lay.b[i]);
        c.cnot(b, x);
        c.gate(Gate::Ccx(lay.ctrl, x, b));
        c.cnot(b, x);
    }
    c.append(&cmult(lay, inv, n_mod).inverse());
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::gcd;
    use crate::shor::ripple::eval_circuit_on_key;

    #[test]
    fn lookup_exhaustive() {
        for w in 1..=4usize {
            let addr: Vec<usize> = (1..=w).collect();
            let and: Vec<usize> = (w + 1..=2 * w).collect();
            let l: Vec<usize> = (2 * w + 1..2 * w + 7).collect();
            let table: Vec<u64> = (0..1u64 << w).map(|v| (v * 37 + 5) % 61 % 64).collect();
            let mut c = Circuit::new(2 * w + 7);
            lookup(&mut c, 0, &addr, &and, &l, &table);
            for ctrl in 0..2u64 {
                for v in 0..1u64 << w {
                    for l0 in [0u64, 0b101101] {
                        let k = ctrl | (v << 1) | (l0 << (2 * w + 1));
                        let out = eval_circuit_on_key(k, &c);
                        let want = if ctrl == 1 {
                            l0 ^ table[v as usize]
                        } else {
                            l0
                        };
                        assert_eq!(out & ((1 << (2 * w + 1)) - 1), k & ((1 << (2 * w + 1)) - 1));
                        assert_eq!(out >> (2 * w + 1), want, "w={w} ctrl={ctrl} v={v}");
                    }
                }
            }
        }
    }

    #[test]
    fn windowed_controlled_ua_exhaustive_small() {
        for n_mod in [15u64, 21, 33, 35, 55, 63] {
            let n = crate::shor::work_bits(n_mod);
            for w in 1..=4 {
                let lay = WindowLayout::new(n, w);
                assert!(lay.num_qubits() <= 64);
                let anc_mask: u64 =
                    !((1u64 << (n + 1)) - 1) & (u64::MAX >> (64 - lay.num_qubits()));
                for a in (2..n_mod).filter(|&a| gcd(a, n_mod) == 1).take(5) {
                    let c = controlled_ua(&lay, a, n_mod);
                    for ctrl in [0u64, 1] {
                        for x in 0..n_mod {
                            let k_out = eval_circuit_on_key(ctrl | (x << 1), &c);
                            assert_eq!(
                                k_out & anc_mask,
                                0,
                                "dirty: N={n_mod} w={w} a={a} c={ctrl} x={x}"
                            );
                            assert_eq!(k_out & 1, ctrl);
                            let want = if ctrl == 1 { x * a % n_mod } else { x };
                            assert_eq!(
                                (k_out >> 1) & ((1 << n) - 1),
                                want,
                                "N={n_mod} w={w} a={a} x={x}"
                            );
                        }
                    }
                }
            }
        }
    }
}
