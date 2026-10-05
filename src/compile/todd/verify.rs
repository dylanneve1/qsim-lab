//! Exact equivalence checks for Clifford+T circuits, written independently
//! of the optimiser's own term bookkeeping.
//!
//! * [`path_sum`]: the sum-over-paths form
//!   `U|x> = 2^{-h/2} Σ_y ω^{P(x,y)} |f(x,y)>` (one path variable per
//!   Hadamard), with the phase `P` in canonical multilinear form over
//!   `Z_8` (monomials of degree ≤ 3; higher ones vanish mod 8) and the
//!   output `f` as affine functions. Two circuits with the same number of
//!   Hadamards, the same `f` and the same `P` up to its constant term are
//!   equal up to a global phase `ω^{Δ}` ([`equivalent`]). The check is
//!   sufficient, not necessary: it is meant for optimisers that keep every
//!   Hadamard in place, which is what this module's optimiser does.
//! * [`simulate_basis`]: exact sparse simulation of one computational-basis
//!   input over `Z[ω]` with a common `1/√2^k` (no floating point), for a
//!   second, semantics-only check ([`basis_equivalent`]).

use super::gf2::Bits;
use crate::circuit::{Circuit, Op};
use crate::gate::Gate;
use std::collections::HashMap;

/// Gates the checks understand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VGate {
    /// Hadamard.
    H(usize),
    /// Pauli X.
    X(usize),
    /// `Cnot(control, target)`.
    Cnot(usize, usize),
    /// SWAP.
    Swap(usize, usize),
    /// `diag(1, ω^k)`, `k` mod 8 (T = 1, S = 2, Z = 4, S† = 6, T† = 7).
    Phase(usize, u8),
    /// Controlled Z.
    Cz(usize, usize),
    /// Doubly controlled Z.
    Ccz(usize, usize, usize),
    /// Toffoli (checked as `H·CCZ·H` on the target).
    Ccx(usize, usize, usize),
    /// Multiplies the whole state by `ω^k`.
    Global(u8),
}

/// Why a gate list could not be checked.
#[derive(Clone, Debug, PartialEq)]
pub struct UnsupportedGate(pub String);

impl std::fmt::Display for UnsupportedGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "gate not supported by the Clifford+T checker: {}",
            self.0
        )
    }
}

impl std::error::Error for UnsupportedGate {}

/// The check gates of a [`Circuit`] over `{H, X, Y, Z, S, S†, T, T†,
/// CNOT, CZ, SWAP, Toffoli}`.
pub fn vgates_from_circuit(c: &Circuit) -> Result<Vec<VGate>, UnsupportedGate> {
    let mut out = Vec::with_capacity(c.ops.len());
    for op in &c.ops {
        let Op::Gate(g) = op else {
            return Err(UnsupportedGate(format!("{op:?}")));
        };
        match *g {
            Gate::I(_) => {}
            Gate::H(q) => out.push(VGate::H(q)),
            Gate::X(q) => out.push(VGate::X(q)),
            Gate::Y(q) => {
                // Y = i·X·Z (Z first)
                out.push(VGate::Phase(q, 4));
                out.push(VGate::X(q));
                out.push(VGate::Global(2));
            }
            Gate::Z(q) => out.push(VGate::Phase(q, 4)),
            Gate::S(q) => out.push(VGate::Phase(q, 2)),
            Gate::Sdg(q) => out.push(VGate::Phase(q, 6)),
            Gate::T(q) => out.push(VGate::Phase(q, 1)),
            Gate::Tdg(q) => out.push(VGate::Phase(q, 7)),
            Gate::Cnot(a, b) => out.push(VGate::Cnot(a, b)),
            Gate::Cz(a, b) => out.push(VGate::Cz(a, b)),
            Gate::Swap(a, b) => out.push(VGate::Swap(a, b)),
            Gate::Ccx(a, b, t) => out.push(VGate::Ccx(a, b, t)),
            other => return Err(UnsupportedGate(format!("{other:?}"))),
        }
    }
    Ok(out)
}

/// The check gates of a `.qc` circuit (`CCZ` native, Toffoli as
/// [`VGate::Ccx`]).
pub fn vgates_from_qc(c: &crate::io::qc::QcCircuit) -> Vec<VGate> {
    use crate::io::qc::QcGate as Q;
    let mut out = Vec::with_capacity(c.gates.len());
    for g in &c.gates {
        match *g {
            Q::H(q) => out.push(VGate::H(q)),
            Q::X(q) => out.push(VGate::X(q)),
            Q::Y(q) => {
                out.push(VGate::Phase(q, 4));
                out.push(VGate::X(q));
                out.push(VGate::Global(2));
            }
            Q::Z(q) => out.push(VGate::Phase(q, 4)),
            Q::S(q) => out.push(VGate::Phase(q, 2)),
            Q::Sdg(q) => out.push(VGate::Phase(q, 6)),
            Q::T(q) => out.push(VGate::Phase(q, 1)),
            Q::Tdg(q) => out.push(VGate::Phase(q, 7)),
            Q::Cnot(a, b) => out.push(VGate::Cnot(a, b)),
            Q::Cz(a, b) => out.push(VGate::Cz(a, b)),
            Q::Ccz(a, b, t) => out.push(VGate::Ccz(a, b, t)),
            Q::Toffoli(a, b, t) => out.push(VGate::Ccx(a, b, t)),
            Q::Swap(a, b) => out.push(VGate::Swap(a, b)),
        }
    }
    out
}

/// A monomial `x_a x_b x_c` with `a < b < c`; unused slots are `u32::MAX`.
type Mono = [u32; 3];

/// The sum-over-paths form of a circuit (see the module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PathSum {
    /// Number of Hadamard path variables `h`.
    pub hadamards: usize,
    /// Constant term of `P` (global phase `ω^constant`).
    pub constant: u8,
    /// Nonzero monomial coefficients of `P` mod 8.
    pub poly: HashMap<Mono, u8>,
    /// Output wire values: linear part over the variables (inputs first,
    /// then path variables in Hadamard order) and constant bit.
    pub outputs: Vec<(Bits, bool)>,
}

struct PsBuilder {
    constant: u8,
    poly: HashMap<Mono, u8>,
}

impl PsBuilder {
    fn add(&mut self, m: Mono, k: u8) {
        let k = k % 8;
        if k == 0 {
            return;
        }
        let e = self.poly.entry(m).or_insert(0);
        *e = (*e + k) % 8;
        if *e == 0 {
            self.poly.remove(&m);
        }
    }

    /// Adds `k·(c ⊕ ⊕_{i ∈ vars} x_i)` using
    /// `⊕ x_i = Σ_{∅≠T} (-2)^{|T|-1} x^T` (terms with |T| ≥ 4 vanish mod 8).
    fn add_parity(&mut self, vars: &[u32], c: bool, k: u8) {
        let mut k = k % 8;
        if k == 0 {
            return;
        }
        if c {
            self.constant = (self.constant + k) % 8;
            k = (8 - k) % 8;
        }
        for (i, &a) in vars.iter().enumerate() {
            self.add([a, u32::MAX, u32::MAX], k);
            if k % 4 != 0 {
                for (j, &b) in vars.iter().enumerate().skip(i + 1) {
                    self.add([a, b, u32::MAX], (16 - 2 * k) % 8);
                    if k % 2 == 1 {
                        for &cc in &vars[j + 1..] {
                            self.add([a, b, cc], 4);
                        }
                    }
                }
            }
        }
    }
}

/// Computes the path sum of `gates` on `n` qubits. Qubits with
/// `zero_init[q]` set start as the constant 0 (no input variable), so
/// equality of two path sums then means equality on that subspace only.
pub fn path_sum(n: usize, gates: &[VGate], zero_init: Option<&[bool]>) -> PathSum {
    let nh: usize = gates
        .iter()
        .map(|g| match g {
            VGate::H(_) => 1,
            VGate::Ccx(..) => 2,
            _ => 0,
        })
        .sum();
    let nv = n + nh;
    let mut lin: Vec<Bits> = (0..n)
        .map(|q| {
            if zero_init.is_some_and(|z| z[q]) {
                Bits::zeros(nv)
            } else {
                Bits::unit(nv, q)
            }
        })
        .collect();
    let mut cst = vec![false; n];
    let mut b = PsBuilder {
        constant: 0,
        poly: HashMap::new(),
    };
    let mut next = n;
    let vars = |v: &Bits| -> Vec<u32> { v.ones().map(|i| i as u32).collect() };
    let mut hadamard = |q: usize, lin: &mut Vec<Bits>, cst: &mut Vec<bool>, b: &mut PsBuilder| {
        let y = next as u32;
        next += 1;
        // phase 4·(c ⊕ l)·y = 4cy + 4·Σ_{i∈l} x_i y (mod 8)
        if cst[q] {
            b.add([y, u32::MAX, u32::MAX], 4);
        }
        for i in lin[q].ones() {
            b.add([i as u32, y, u32::MAX], 4);
        }
        lin[q] = Bits::unit(nv, y as usize);
        cst[q] = false;
    };
    for g in gates {
        match *g {
            VGate::H(q) => hadamard(q, &mut lin, &mut cst, &mut b),
            VGate::X(q) => cst[q] = !cst[q],
            VGate::Cnot(a, t) => {
                let s = lin[a].clone();
                lin[t].xor_with(&s);
                cst[t] ^= cst[a];
            }
            VGate::Swap(a, t) => {
                lin.swap(a, t);
                cst.swap(a, t);
            }
            VGate::Global(k) => b.constant = (b.constant + k) % 8,
            VGate::Phase(q, k) => b.add_parity(&vars(&lin[q]), cst[q], k),
            VGate::Cz(a, t) => {
                // 4ab = 2a + 2b - 2(a ⊕ b)
                let mut ab = lin[a].clone();
                ab.xor_with(&lin[t]);
                b.add_parity(&vars(&lin[a]), cst[a], 2);
                b.add_parity(&vars(&lin[t]), cst[t], 2);
                b.add_parity(&vars(&ab), cst[a] ^ cst[t], 6);
            }
            VGate::Ccz(p, q, r) => ccz(&lin, &cst, p, q, r, &mut b, &vars),
            VGate::Ccx(p, q, r) => {
                hadamard(r, &mut lin, &mut cst, &mut b);
                ccz(&lin, &cst, p, q, r, &mut b, &vars);
                hadamard(r, &mut lin, &mut cst, &mut b);
            }
        }
    }
    PathSum {
        hadamards: nh,
        constant: b.constant,
        poly: b.poly,
        outputs: lin.into_iter().zip(cst).collect(),
    }
}

fn ccz(
    lin: &[Bits],
    cst: &[bool],
    p: usize,
    q: usize,
    r: usize,
    b: &mut PsBuilder,
    vars: &dyn Fn(&Bits) -> Vec<u32>,
) {
    // 4pqr = p + q + r - (p⊕q) - (p⊕r) - (q⊕r) + (p⊕q⊕r)
    for (set, k) in [
        (&[p][..], 1u8),
        (&[q][..], 1),
        (&[r][..], 1),
        (&[p, q][..], 7),
        (&[p, r][..], 7),
        (&[q, r][..], 7),
        (&[p, q, r][..], 1),
    ] {
        let mut l = lin[set[0]].clone();
        let mut c = cst[set[0]];
        for &w in &set[1..] {
            l.xor_with(&lin[w]);
            c ^= cst[w];
        }
        b.add_parity(&vars(&l), c, k);
    }
}

/// `Some(Δ)` when the two path sums are identical except for their
/// constant terms (then `U_a = ω^Δ U_b`), `None` otherwise.
pub fn equivalent(a: &PathSum, b: &PathSum) -> Option<u8> {
    if a.hadamards == b.hadamards && a.outputs == b.outputs && a.poly == b.poly {
        Some((a.constant + 8 - b.constant) % 8)
    } else {
        None
    }
}

/// An element `a0 + a1 ω + a2 ω² + a3 ω³` of `Z[ω]`, `ω = e^{iπ/4}`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ZOmega(pub [i128; 4]);

impl ZOmega {
    /// `self · ω^k`.
    pub fn mul_omega(self, k: u8) -> Self {
        let mut a = self.0;
        for _ in 0..(k % 8) {
            // ω·(a0, a1, a2, a3) = (-a3, a0, a1, a2)
            a = [-a[3], a[0], a[1], a[2]];
        }
        ZOmega(a)
    }

    fn add(self, o: Self) -> Self {
        ZOmega([
            self.0[0] + o.0[0],
            self.0[1] + o.0[1],
            self.0[2] + o.0[2],
            self.0[3] + o.0[3],
        ])
    }

    fn is_zero(&self) -> bool {
        self.0 == [0; 4]
    }

    /// `self / √2` if it lies in `Z[ω]` (`√2 = ω - ω³`).
    fn div_sqrt2(self) -> Option<Self> {
        let [a0, a1, a2, a3] = self.0;
        if (a0 - a2) % 2 != 0 || (a1 - a3) % 2 != 0 {
            return None;
        }
        Some(ZOmega([
            (a1 - a3) / 2,
            (a0 + a2) / 2,
            (a1 + a3) / 2,
            (a2 - a0) / 2,
        ]))
    }

    fn mul_sqrt2(self) -> Self {
        // (b0 + b1ω + b2ω² + b3ω³)(ω - ω³)
        let [b0, b1, b2, b3] = self.0;
        ZOmega([b1 - b3, b0 + b2, b1 + b3, b2 - b0])
    }
}

/// An exact sparse state `Σ_x amps[x] |x> / √2^k`, normalised so that not
/// every amplitude is divisible by `√2` (unless `k = 0`), sorted by basis
/// index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactState {
    /// The power of `1/√2`.
    pub k: u32,
    /// `(basis index, amplitude numerator)`, sorted, no zeros.
    pub amps: Vec<(u128, ZOmega)>,
}

/// Runs `gates` exactly on the basis input `input` (`n ≤ 128` qubits).
pub fn simulate_basis(n: usize, gates: &[VGate], input: u128) -> ExactState {
    assert!(n <= 128, "simulate_basis supports up to 128 qubits");
    let mut amps: HashMap<u128, ZOmega> = HashMap::new();
    amps.insert(input, ZOmega([1, 0, 0, 0]));
    let mut k: u32 = 0;
    let bit = |x: u128, q: usize| (x >> q) & 1 == 1;
    let permute = |amps: &mut HashMap<u128, ZOmega>, f: &dyn Fn(u128) -> u128| {
        let old = std::mem::take(amps);
        for (x, a) in old {
            amps.insert(f(x), a);
        }
    };
    let phase = |amps: &mut HashMap<u128, ZOmega>, f: &dyn Fn(u128) -> u8| {
        for (x, a) in amps.iter_mut() {
            *a = a.mul_omega(f(*x));
        }
    };
    let hadamard = |amps: &mut HashMap<u128, ZOmega>, k: &mut u32, q: usize| {
        let old = std::mem::take(amps);
        for (x, a) in old {
            let x0 = x & !(1u128 << q);
            let x1 = x | (1u128 << q);
            let e0 = amps.entry(x0).or_default();
            *e0 = e0.add(a);
            let neg = if bit(x, q) { a.mul_omega(4) } else { a };
            let e1 = amps.entry(x1).or_default();
            *e1 = e1.add(neg);
        }
        amps.retain(|_, a| !a.is_zero());
        *k += 1;
        // normalise
        while *k > 0 && amps.values().all(|a| a.div_sqrt2().is_some()) {
            for a in amps.values_mut() {
                *a = a.div_sqrt2().expect("checked");
            }
            *k -= 1;
        }
    };
    let mut global = 0u8;
    for g in gates {
        match *g {
            VGate::H(q) => hadamard(&mut amps, &mut k, q),
            VGate::X(q) => permute(&mut amps, &|x| x ^ (1u128 << q)),
            VGate::Cnot(a, t) => permute(&mut amps, &|x| {
                if bit(x, a) {
                    x ^ (1u128 << t)
                } else {
                    x
                }
            }),
            VGate::Swap(a, t) => permute(&mut amps, &|x| {
                if bit(x, a) != bit(x, t) {
                    x ^ (1u128 << a) ^ (1u128 << t)
                } else {
                    x
                }
            }),
            VGate::Phase(q, kk) => phase(&mut amps, &|x| if bit(x, q) { kk } else { 0 }),
            VGate::Cz(a, t) => phase(&mut amps, &|x| if bit(x, a) && bit(x, t) { 4 } else { 0 }),
            VGate::Ccz(a, b2, t) => phase(&mut amps, &|x| {
                if bit(x, a) && bit(x, b2) && bit(x, t) {
                    4
                } else {
                    0
                }
            }),
            VGate::Ccx(a, b2, t) => permute(&mut amps, &|x| {
                if bit(x, a) && bit(x, b2) {
                    x ^ (1u128 << t)
                } else {
                    x
                }
            }),
            VGate::Global(kk) => global = (global + kk) % 8,
        }
    }
    let mut v: Vec<(u128, ZOmega)> = amps
        .into_iter()
        .map(|(x, a)| (x, a.mul_omega(global)))
        .collect();
    v.sort_unstable_by_key(|e| e.0);
    let _ = n;
    ExactState { k, amps: v }
}

/// `Some(j)` if `a = ω^j b` exactly.
pub fn exact_phase_relation(a: &ExactState, b: &ExactState) -> Option<u8> {
    if a.amps.len() != b.amps.len() || a.amps.is_empty() {
        return None;
    }
    // bring both to the same power of 1/√2
    let lift = |s: &ExactState, to: u32| -> Vec<(u128, ZOmega)> {
        s.amps
            .iter()
            .map(|&(x, mut v)| {
                for _ in s.k..to {
                    v = v.mul_sqrt2();
                }
                (x, v)
            })
            .collect()
    };
    let kk = a.k.max(b.k);
    let (va, vb) = (lift(a, kk), lift(b, kk));
    let j = (0..8u8).find(|&j| va[0].0 == vb[0].0 && va[0].1 == vb[0].1.mul_omega(j))?;
    va.iter()
        .zip(&vb)
        .all(|(p, q)| p.0 == q.0 && p.1 == q.1.mul_omega(j))
        .then_some(j)
}

/// Checks `A|x> = ω^j B|x>` with one common `j` for every basis input in
/// `inputs`; returns `j`, or the first failing input.
pub fn basis_equivalent(n: usize, a: &[VGate], b: &[VGate], inputs: &[u128]) -> Result<u8, u128> {
    use rayon::prelude::*;
    let Some(&first) = inputs.first() else {
        return Ok(0);
    };
    let j0 = exact_phase_relation(&simulate_basis(n, a, first), &simulate_basis(n, b, first))
        .ok_or(first)?;
    let bad = inputs.par_iter().find_first(|&&x| {
        exact_phase_relation(&simulate_basis(n, a, x), &simulate_basis(n, b, x)) != Some(j0)
    });
    match bad {
        Some(&x) => Err(x),
        None => Ok(j0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ccz_decompositions_agree() {
        // CCZ as 7 T gates (Amy et al.) vs the native gate.
        let t = |q| VGate::Phase(q, 1);
        let td = |q| VGate::Phase(q, 7);
        let dec = vec![
            t(0),
            t(1),
            t(2),
            VGate::Cnot(0, 1),
            td(1),
            VGate::Cnot(0, 1),
            VGate::Cnot(0, 2),
            td(2),
            VGate::Cnot(0, 2),
            VGate::Cnot(1, 2),
            td(2),
            VGate::Cnot(1, 2),
            VGate::Cnot(0, 2),
            VGate::Cnot(1, 2),
            t(2),
            VGate::Cnot(1, 2),
            VGate::Cnot(0, 2),
        ];
        let native = vec![VGate::Ccz(0, 1, 2)];
        assert_eq!(
            equivalent(&path_sum(3, &dec, None), &path_sum(3, &native, None)),
            Some(0)
        );
        let inputs: Vec<u128> = (0..8).collect();
        assert_eq!(basis_equivalent(3, &dec, &native, &inputs), Ok(0));
        // a wrong decomposition is caught by both checks
        let mut bad = dec.clone();
        bad[0] = td(0);
        assert_eq!(
            equivalent(&path_sum(3, &bad, None), &path_sum(3, &native, None)),
            None
        );
        assert!(basis_equivalent(3, &bad, &native, &inputs).is_err());
    }

    #[test]
    fn toffoli_is_h_ccz_h_and_hh_is_exact() {
        let a = vec![VGate::Ccx(0, 1, 2)];
        let b = vec![VGate::H(2), VGate::Ccz(0, 1, 2), VGate::H(2)];
        assert_eq!(
            equivalent(&path_sum(3, &a, None), &path_sum(3, &b, None)),
            Some(0)
        );
        let inputs: Vec<u128> = (0..8).collect();
        assert_eq!(basis_equivalent(3, &a, &b, &inputs), Ok(0));
        let hh = vec![VGate::H(0), VGate::Phase(0, 1), VGate::H(0)];
        let s = simulate_basis(1, &hh, 0);
        assert!(s.k <= 2);
    }
}
