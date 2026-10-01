//! Scaling Shor's order finding.
//!
//! [`crate::algorithms::shor_order_finding`] is the textbook circuit: `2n`
//! counting qubits, `n` work qubits, a full inverse QFT, `3n` qubits in total.
//! This module adds:
//!
//! * **Semiclassical QFT with one recycled control qubit** (Griffiths–Niu
//!   1996; Mosca–Ekert 1998; Beauregard 2003). The `2n` counting qubits are
//!   replaced by a single qubit that is reset, put in `|+>`, used as control
//!   of `U^(2^k)`, rotated by a phase that depends on the bits measured so far,
//!   Hadamard-ed and measured, `2n` times. The distribution of the measured
//!   integer is *identical* to the full-QFT circuit (deferred-measurement
//!   principle); [`semiclassical_distribution`] enumerates it exactly so the
//!   tests can check that to 1e-12.
//! * Two backends behind [`OrderFindingState`]: the dense
//!   [`StateVector`] and the exact [`SparseState`].
//! * Oracles: the permutation oracle `|1>|y> -> |1>|a y mod N>` (a lookup
//!   table, as in `algorithms`), and gate-level Beauregard arithmetic
//!   ([`crate::shor_arith`]).
//!
//! Qubit layout (all paths): qubit 0 is the recycled control, the work
//! register `x` is qubits `1..=n`; the gate-level oracle adds `n + 1` qubits
//! for the Fourier-space accumulator `b` and one ancilla.

use crate::algorithms::{gcd, pow_mod, shor_postprocess};
use crate::circuit::Circuit;
use crate::gate::Gate;
use crate::shor_arith::{self, BeauregardLayout};
use crate::sparse::SparseState;
use crate::statevector::{Real, StateVector};
use num_complex::Complex;
use rand::Rng;
use rayon::prelude::*;
use std::f64::consts::PI;

/// Which implementation of controlled `U_a: |y> -> |a y mod N>` to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Oracle {
    /// A basis-state permutation (classical lookup table), `n + 1` qubits.
    Permutation,
    /// Beauregard's gate-level circuit (QFT adders, modular adder,
    /// controlled multiplier, controlled swap), `2n + 3` qubits.
    Beauregard,
}

/// A simulator state that can run semiclassical order finding.
pub trait OrderFindingState: Clone {
    fn gate(&mut self, g: &Gate);
    /// `|c=1>|y> -> |c=1>|mult * y mod N>` for `y < N` (control = qubit 0,
    /// work register = qubits `1..=m`); identity elsewhere. `inv` is
    /// `mult^-1 mod N`.
    fn ctrl_mul(&mut self, m: usize, mult: u64, inv: u64, n_mod: u64);
    fn prob_one(&self, q: usize) -> f64;
    fn collapse(&mut self, q: usize, outcome: bool);
    /// Bytes held by the amplitudes.
    fn bytes(&self) -> usize;
    /// Stored amplitudes (dense: `2^n`).
    fn stored(&self) -> usize;
}

impl<T: Real> OrderFindingState for StateVector<T> {
    fn gate(&mut self, g: &Gate) {
        self.apply_gate(g).expect("valid gate");
    }
    fn ctrl_mul(&mut self, m: usize, _mult: u64, inv: u64, n_mod: u64) {
        assert!(n_mod < 1 << 32);
        let mask = (1usize << m) - 1;
        let amps = self.amplitudes_mut();
        // gather: the new amplitude of |1>|z> is the old one of |1>|inv z>
        let odd: Vec<Complex<T>> = amps.par_chunks(2).map(|c| c[1]).collect();
        let n = n_mod as usize;
        let inv = inv as usize;
        amps.par_chunks_mut(2).enumerate().for_each(|(j, c)| {
            let y = j & mask;
            if y < n {
                c[1] = odd[(j & !mask) | (y * inv % n)];
            }
        });
    }
    fn prob_one(&self, q: usize) -> f64 {
        StateVector::prob_one(self, q)
    }
    fn collapse(&mut self, q: usize, outcome: bool) {
        StateVector::collapse(self, q, outcome);
    }
    fn bytes(&self) -> usize {
        StateVector::bytes(self)
    }
    fn stored(&self) -> usize {
        1 << self.num_qubits()
    }
}

impl OrderFindingState for SparseState {
    fn gate(&mut self, g: &Gate) {
        self.apply_gate(g).expect("valid gate");
    }
    fn ctrl_mul(&mut self, m: usize, mult: u64, _inv: u64, n_mod: u64) {
        assert!(n_mod < 1 << 32);
        let mask = (1u64 << m) - 1;
        self.apply_permutation(|k| {
            if k & 1 == 0 {
                return k;
            }
            let y = (k >> 1) & mask;
            if y >= n_mod {
                return k;
            }
            (k & !(mask << 1)) | ((y * mult % n_mod) << 1)
        });
    }
    fn prob_one(&self, q: usize) -> f64 {
        SparseState::prob_one(self, q)
    }
    fn collapse(&mut self, q: usize, outcome: bool) {
        SparseState::collapse(self, q, outcome);
    }
    fn bytes(&self) -> usize {
        SparseState::bytes(self)
    }
    fn stored(&self) -> usize {
        self.nnz()
    }
}

/// Number of bits of `N - 1`, i.e. the work-register width `n`.
pub fn work_bits(n_mod: u64) -> usize {
    64 - (n_mod - 1).leading_zeros() as usize
}

/// Modular inverse of `a` mod `n` (`gcd(a, n) = 1`).
pub fn mod_inverse(a: u64, n: u64) -> u64 {
    let (mut t, mut new_t) = (0i128, 1i128);
    let (mut r, mut new_r) = (n as i128, (a % n) as i128);
    while new_r != 0 {
        let q = r / new_r;
        (t, new_t) = (new_t, t - q * new_t);
        (r, new_r) = (new_r, r - q * new_r);
    }
    assert_eq!(r, 1, "{a} is not invertible mod {n}");
    t.rem_euclid(n as i128) as u64
}

/// Everything fixed for one order-finding instance.
#[derive(Clone, Debug)]
pub struct Instance {
    pub n_mod: u64,
    pub a: u64,
    /// Work-register width `n`.
    pub m: usize,
    /// Number of measured bits `t = 2n`.
    pub t: usize,
    pub oracle: Oracle,
    /// `a^(2^k) mod N` for `k = 0..t`.
    mults: Vec<u64>,
}

impl Instance {
    pub fn new(n_mod: u64, a: u64, oracle: Oracle) -> Self {
        assert!(n_mod >= 3 && gcd(a, n_mod) == 1);
        let m = work_bits(n_mod);
        let t = 2 * m;
        assert!(t <= 62, "N too large for u64 phase bookkeeping");
        let mults = (0..t).map(|k| pow_mod(a, 1 << k, n_mod)).collect();
        Self {
            n_mod,
            a,
            m,
            t,
            oracle,
            mults,
        }
    }

    /// Total number of qubits the semiclassical circuit needs.
    pub fn qubits(&self) -> usize {
        match self.oracle {
            Oracle::Permutation => self.m + 1,
            Oracle::Beauregard => 2 * self.m + 3,
        }
    }

    fn layout(&self) -> BeauregardLayout {
        BeauregardLayout::new(self.m)
    }

    /// Basis index of the initial state: control 0, work register `|1>`.
    pub fn initial_index(&self) -> u64 {
        1 << 1
    }

    /// The phase correction applied before the final H of step `i` (which
    /// measures bit `i` of the result), given the lower bits `y_low`
    /// already measured: `-2π · y_low / 2^(i+1)`.
    pub fn correction(i: usize, y_low: u64) -> f64 {
        -PI * (y_low as f64) / ((1u64 << i) as f64)
    }

    /// Applies one round before the measurement: `H`, controlled
    /// `U^(2^(t-1-i))`, phase correction, `H` on the control qubit.
    pub fn round<S: OrderFindingState>(&self, s: &mut S, i: usize, y_low: u64) {
        let k = self.t - 1 - i;
        s.gate(&Gate::H(0));
        let mult = self.mults[k];
        match self.oracle {
            Oracle::Permutation => {
                s.ctrl_mul(self.m, mult, mod_inverse(mult, self.n_mod), self.n_mod)
            }
            Oracle::Beauregard => {
                let c = shor_arith::controlled_ua(&self.layout(), 0, mult, self.n_mod);
                for g in c.gates() {
                    s.gate(g);
                }
            }
        }
        if y_low != 0 {
            s.gate(&Gate::Phase(0, Self::correction(i, y_low)));
        }
        s.gate(&Gate::H(0));
    }
}

/// Outcome of one semiclassical run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemiRun {
    pub a: u64,
    pub measured: u64,
    pub order: Option<u64>,
    pub factor: Option<u64>,
    pub qubits: usize,
    /// Largest number of stored amplitudes seen during the run.
    pub peak_stored: usize,
    /// Largest amplitude memory seen during the run (bytes).
    pub peak_bytes: usize,
}

/// One semiclassical order-finding run on the state `s` (which must be the
/// initial state [`Instance::initial_index`] on [`Instance::qubits`] qubits).
pub fn run_semiclassical<S: OrderFindingState, R: Rng + ?Sized>(
    inst: &Instance,
    mut s: S,
    rng: &mut R,
) -> SemiRun {
    let mut y = 0u64;
    let (mut peak_stored, mut peak_bytes) = (s.stored(), s.bytes());
    for i in 0..inst.t {
        inst.round(&mut s, i, y);
        peak_stored = peak_stored.max(s.stored());
        peak_bytes = peak_bytes.max(s.bytes());
        // same random draw as StateVector::measure_qubit
        let p1 = s.prob_one(0);
        let bit = rng.random::<f64>() < p1;
        s.collapse(0, bit);
        if bit {
            y |= 1 << i;
            s.gate(&Gate::X(0)); // recycle: reset the control to |0>
        }
    }
    let (order, factor) = shor_postprocess(inst.n_mod, inst.a, y, inst.t as u32);
    SemiRun {
        a: inst.a,
        measured: y,
        order,
        factor,
        qubits: inst.qubits(),
        peak_stored,
        peak_bytes,
    }
}

/// Dense initial state for `inst`.
pub fn dense_initial<T: Real>(inst: &Instance) -> StateVector<T> {
    StateVector::<T>::basis_state(inst.qubits(), inst.initial_index() as usize)
}

/// Sparse initial state for `inst`.
pub fn sparse_initial(inst: &Instance) -> SparseState {
    SparseState::basis_state(inst.qubits(), inst.initial_index())
}

/// The exact distribution of the measured `t`-bit integer, by walking the
/// whole tree of measurement outcomes (each branch's probability is taken
/// from the state before collapse). Branches with probability below
/// `prune` (e.g. `1e-15`) are skipped; their leaves are reported as 0.
pub fn semiclassical_distribution<S: OrderFindingState>(
    inst: &Instance,
    s: S,
    prune: f64,
) -> Vec<f64> {
    let mut out = vec![0.0; 1usize << inst.t];
    fn walk<S: OrderFindingState>(
        inst: &Instance,
        mut s: S,
        i: usize,
        y: u64,
        p: f64,
        prune: f64,
        out: &mut [f64],
    ) {
        if i == inst.t {
            out[y as usize] = p;
            return;
        }
        inst.round(&mut s, i, y);
        let p1 = s.prob_one(0);
        for bit in [false, true] {
            let pb = if bit { p1 } else { 1.0 - p1 };
            if pb <= prune {
                continue;
            }
            let mut c = s.clone();
            c.collapse(0, bit);
            if bit {
                c.gate(&Gate::X(0));
            }
            walk(
                inst,
                c,
                i + 1,
                y | (u64::from(bit) << i),
                p * pb,
                prune,
                out,
            );
        }
    }
    walk(inst, s, 0, 0, 1.0, prune, &mut out);
    out
}

/// The same distribution for the textbook `3n`-qubit circuit
/// ([`crate::algorithms::shor_full_state`]): the marginal of the counting
/// register.
pub fn full_qft_distribution(n_mod: u64, a: u64) -> Vec<f64> {
    let (s, t) = crate::algorithms::shor_full_state(n_mod, a);
    let mut out = vec![0.0; 1 << t];
    let mask = (1usize << t) - 1;
    for (i, amp) in s.amplitudes().iter().enumerate() {
        out[i & mask] += amp.norm_sqr();
    }
    out
}

/// The semiclassical circuit as a plain [`Circuit`] (gate-level oracle
/// only): reset (as a classically controlled `X`), `H`, controlled `U^(2^k)`, the phase correction as one
/// classically controlled `Phase(-π/2^d)` per earlier bit (the correction
/// factorises over the measured bits, so the existing single-bit
/// `ClassicControlled` op suffices), `H`, `Measure`. Measurement `i` is bit
/// `i` of the result. Runs on any [`crate::Simulator`] via [`Circuit::run`].
pub fn semiclassical_circuit(n_mod: u64, a: u64) -> Circuit {
    let inst = Instance::new(n_mod, a, Oracle::Beauregard);
    let lay = inst.layout();
    let mut c = Circuit::new(inst.qubits());
    c.x(1); // work register |1>
    for i in 0..inst.t {
        let k = inst.t - 1 - i;
        if i > 0 {
            // recycle the control: it was just measured, so a classically
            // controlled X resets it deterministically (an `Op::Reset`
            // would re-measure it, which is equivalent but draws randomness)
            c.c_if(i - 1, Gate::X(0));
        }
        c.h(0);
        c.append(&shor_arith::controlled_ua(&lay, 0, inst.mults[k], n_mod));
        for l in 0..i {
            // bit l contributes -2π · 2^l / 2^(i+1) = -π / 2^(i-l)
            c.c_if(l, Gate::Phase(0, -PI / ((1u64 << (i - l)) as f64)));
        }
        c.h(0);
        c.measure(0);
    }
    c
}

/// Which simulator backs a semiclassical factoring attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    DenseF64,
    DenseF32,
    Sparse,
}

/// Runs one semiclassical order finding with the given backend.
pub fn order_finding<R: Rng + ?Sized>(inst: &Instance, backend: Backend, rng: &mut R) -> SemiRun {
    match backend {
        Backend::DenseF64 => run_semiclassical(inst, dense_initial::<f64>(inst), rng),
        Backend::DenseF32 => run_semiclassical(inst, dense_initial::<f32>(inst), rng),
        Backend::Sparse => run_semiclassical(inst, sparse_initial(inst), rng),
    }
}

/// Factors `N` (odd, not a prime power) with up to `tries` semiclassical
/// runs, picking random bases like [`crate::algorithms::shor_factor`].
pub fn factor_semiclassical<R: Rng + ?Sized>(
    n_mod: u64,
    oracle: Oracle,
    backend: Backend,
    tries: usize,
    rng: &mut R,
) -> (Option<(u64, u64)>, Vec<SemiRun>) {
    let mut runs = Vec::new();
    for _ in 0..tries {
        let a = rng.random_range(2..n_mod - 1);
        if gcd(a, n_mod) > 1 {
            continue; // lucky classical guess; skip so the quantum part runs
        }
        let inst = Instance::new(n_mod, a, oracle);
        let run = order_finding(&inst, backend, rng);
        let f = run.factor;
        runs.push(run);
        if let Some(f) = f {
            return (Some((f.min(n_mod / f), f.max(n_mod / f))), runs);
        }
    }
    (None, runs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse() {
        for n in [15u64, 21, 143, 1_000_003] {
            for a in 2..50 {
                if gcd(a, n) == 1 {
                    assert_eq!(a * mod_inverse(a, n) % n, 1);
                }
            }
        }
    }
}
