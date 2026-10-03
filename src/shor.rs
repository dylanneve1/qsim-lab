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

pub mod fused;
pub mod sliced;

use crate::algorithms::{gcd, pow_mod};
use crate::blocked::BlockConfig;
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
    /// Cuccaro ripple-carry gate-level circuit (X, CNOT, CCX only),
    /// `3n + 4` qubits.
    Ripple,
    /// Windowed (table-lookup) ripple-carry circuit, window `w`
    /// ([`crate::shor_window`]), `4n + 4 + w` qubits; X, CNOT, CCX only.
    Windowed(usize),
}

/// A simulator state that can run semiclassical order finding.
pub trait OrderFindingState: Clone {
    fn gate(&mut self, g: &Gate);
    /// Applies a gate sequence; `blocked` lets a dense state use the
    /// cache-blocked fused executor ([`crate::blocked`]).
    fn gates(&mut self, gs: &[Gate], blocked: bool) {
        let _ = blocked;
        for g in gs {
            self.gate(g);
        }
    }
    /// Applies a reversible circuit block to the state.
    fn apply_block(&mut self, c: &Circuit, ancilla_mask: u64) {
        let _ = ancilla_mask;
        for g in c.gates() {
            self.gate(g);
        }
    }
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
    /// One round before the measurement of bit `i` (see
    /// [`Instance::round`]); fused states override it.
    fn round(&mut self, inst: &Instance, i: usize, y_low: u128)
    where
        Self: Sized,
    {
        inst.round(self, i, y_low);
    }
    /// Recycles the control after it was measured as `bit`.
    fn reset_control(&mut self, bit: bool) {
        if bit {
            self.gate(&Gate::X(0));
        }
    }
}

impl<T: Real> OrderFindingState for StateVector<T> {
    fn gate(&mut self, g: &Gate) {
        self.apply_gate(g).expect("valid gate");
    }
    fn gates(&mut self, gs: &[Gate], blocked: bool) {
        if blocked {
            self.apply_gates_blocked(gs, &BlockConfig::default())
                .expect("valid gates");
        } else {
            for g in gs {
                self.gate(g);
            }
        }
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
    fn apply_block(&mut self, c: &Circuit, ancilla_mask: u64) {
        crate::shor_ripple::apply_reversible_block(self, c, ancilla_mask);
    }
    fn ctrl_mul(&mut self, m: usize, mult: u64, _inv: u64, n_mod: u64) {
        let mask = (1u64 << m) - 1;
        self.apply_permutation(|k| {
            if k & 1 == 0 {
                return k;
            }
            let y = (k >> 1) & mask;
            if y >= n_mod {
                return k;
            }
            (k & !(mask << 1)) | (mul_mod(y, mult, n_mod) << 1)
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
        self.peak_nnz()
    }
}

/// `x · y mod n` (64-bit product when it cannot overflow).
#[inline]
pub fn mul_mod(x: u64, y: u64, n: u64) -> u64 {
    if n <= 1 << 32 {
        x * y % n
    } else {
        (u128::from(x) * u128::from(y) % u128::from(n)) as u64
    }
}

/// Denominators of the continued-fraction convergents of `x / 2^t`
/// (`t <= 126`).
pub fn convergents(x: u128, t: u32) -> Vec<u128> {
    let (mut num, mut den) = (x, 1u128 << t);
    let (mut q_prev, mut q) = (1u128, 0u128);
    let mut out = Vec::new();
    while den != 0 {
        let a = num / den;
        let q_next = a.saturating_mul(q).saturating_add(q_prev);
        q_prev = q;
        q = q_next;
        out.push(q);
        let r = num % den;
        num = den;
        den = r;
    }
    out
}

/// How many multiples `k·q` of each convergent denominator `q` are tried.
pub const ORDER_MULTIPLES: u64 = 256;

/// Classical post-processing for the scaled paths. Like
/// [`crate::algorithms::shor_postprocess`], plus the standard fix for
/// `measured/2^t ≈ s/r` with `gcd(s, r) = g > 1` (the convergent is `r/g`):
/// try `k·q` for `k = 1..=ORDER_MULTIPLES`, take the first `r` with
/// `a^r = 1 mod N`, and strip small prime factors that keep `a^r = 1`.
/// Purely classical; it does not change the quantum distribution, only how
/// many runs are needed.
pub fn postprocess(n_mod: u64, a: u64, measured: u128, t: u32) -> (Option<u64>, Option<u64>) {
    let mut order = None;
    'outer: for q in convergents(measured, t) {
        if q == 0 || q >= u128::from(n_mod) {
            continue;
        }
        let q = q as u64;
        for k in 1..=ORDER_MULTIPLES {
            let Some(r) = q.checked_mul(k).filter(|&r| r < n_mod) else {
                break;
            };
            if pow_mod(a, r, n_mod) == 1 {
                let mut r = r;
                for p in 2..1000u64 {
                    while r % p == 0 && pow_mod(a, r / p, n_mod) == 1 {
                        r /= p;
                    }
                }
                order = Some(r);
                break 'outer;
            }
        }
    }
    let factor = order.and_then(|r| {
        if r % 2 == 1 {
            return None;
        }
        let y = pow_mod(a, r / 2, n_mod);
        [gcd(y + 1, n_mod), gcd(y + n_mod - 1, n_mod)]
            .into_iter()
            .find(|&f| f > 1 && f < n_mod)
    });
    (order, factor)
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
    /// Dense states run gate-level rounds with the cache-blocked executor
    /// (default `true`; `false` = one `apply_gate` pass per gate).
    pub blocked: bool,
    /// Ripple oracle: apply gates one by one instead of reversible block evaluation.
    pub gate_by_gate: bool,
    /// `a^(2^k) mod N` for `k = 0..t`.
    pub mults: Vec<u64>,
}

impl Instance {
    pub fn new(n_mod: u64, a: u64, oracle: Oracle) -> Self {
        assert!(n_mod >= 3 && gcd(a, n_mod) == 1);
        let m = work_bits(n_mod);
        let t = 2 * m;
        assert!(
            m <= 63,
            "the work register plus control must fit in a u64 key"
        );
        // a^(2^k) by repeated squaring (2^k overflows u64 for k >= 64)
        let mut mults = Vec::with_capacity(t);
        let mut cur = a % n_mod;
        for _ in 0..t {
            mults.push(cur);
            cur = pow_mod(cur, 2, n_mod);
        }
        Self {
            n_mod,
            a,
            m,
            t,
            oracle,
            blocked: true,
            gate_by_gate: false,
            mults,
        }
    }

    /// Total number of qubits the semiclassical circuit needs.
    pub fn qubits(&self) -> usize {
        match self.oracle {
            Oracle::Permutation => self.m + 1,
            Oracle::Beauregard => 2 * self.m + 3,
            Oracle::Ripple => 3 * self.m + 4,
            Oracle::Windowed(w) => 4 * self.m + 4 + w.min(self.m),
        }
    }

    pub fn layout(&self) -> BeauregardLayout {
        BeauregardLayout::new(self.m)
    }

    pub fn ripple_layout(&self) -> crate::shor_ripple::RippleLayout {
        crate::shor_ripple::RippleLayout::new(self.m)
    }

    /// Basis index of the initial state: control 0, work register `|1>`.
    pub fn initial_index(&self) -> u64 {
        1 << 1
    }

    /// The phase correction applied before the final H of step `i` (which
    /// measures bit `i` of the result), given the lower bits `y_low`
    /// already measured: `-2π · y_low / 2^(i+1)`.
    pub fn correction(i: usize, y_low: u128) -> f64 {
        -PI * (y_low as f64) / ((1u128 << i) as f64)
    }

    /// Applies one round before the measurement: `H`, controlled
    /// `U^(2^(t-1-i))`, phase correction, `H` on the control qubit.
    pub fn round<S: OrderFindingState>(&self, s: &mut S, i: usize, y_low: u128) {
        let k = self.t - 1 - i;
        let mult = self.mults[k];
        let corr = (y_low != 0).then(|| Gate::Phase(0, Self::correction(i, y_low)));
        match self.oracle {
            Oracle::Permutation => {
                s.gate(&Gate::H(0));
                s.ctrl_mul(self.m, mult, mod_inverse(mult, self.n_mod), self.n_mod);
                if let Some(g) = corr {
                    s.gate(&g);
                }
                s.gate(&Gate::H(0));
            }
            Oracle::Beauregard => {
                let c = shor_arith::controlled_ua(&self.layout(), 0, mult, self.n_mod);
                let mut gs = Vec::with_capacity(c.ops.len() + 3);
                gs.push(Gate::H(0));
                gs.extend(c.gates().copied());
                gs.extend(corr);
                gs.push(Gate::H(0));
                s.gates(&gs, self.blocked);
            }
            Oracle::Ripple => {
                let lay = self.ripple_layout();
                let c = crate::shor_ripple::controlled_ua(&lay, 0, mult, self.n_mod);
                s.gate(&Gate::H(0));
                if self.gate_by_gate {
                    for g in c.gates() {
                        s.gate(g);
                    }
                } else {
                    s.apply_block(&c, lay.ancilla_mask());
                }
                if let Some(g) = corr {
                    s.gate(&g);
                }
                s.gate(&Gate::H(0));
            }
            Oracle::Windowed(_) => {
                let (c, _) = sliced::oracle_block(self, mult);
                s.gate(&Gate::H(0));
                for g in c.gates() {
                    s.gate(g);
                }
                if let Some(g) = corr {
                    s.gate(&g);
                }
                s.gate(&Gate::H(0));
            }
        }
    }
}

/// Outcome of one semiclassical run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemiRun {
    pub a: u64,
    /// The measured `2n`-bit integer (bit `i` = measurement `i`).
    pub measured: u128,
    pub order: Option<u64>,
    pub factor: Option<u64>,
    pub qubits: usize,
    /// Largest number of stored amplitudes seen during the run.
    pub peak_stored: usize,
    /// Largest amplitude memory seen during the run (bytes).
    pub peak_bytes: usize,
    pub total_gates: usize,
    pub toffoli_gates: usize,
}

/// One semiclassical order-finding run on the state `s` (which must be the
/// initial state [`Instance::initial_index`] on [`Instance::qubits`] qubits).
pub fn run_semiclassical<S: OrderFindingState, R: Rng + ?Sized>(
    inst: &Instance,
    mut s: S,
    rng: &mut R,
) -> SemiRun {
    let mut y = 0u128;
    let (mut peak_stored, mut peak_bytes) = (s.stored(), s.bytes());
    let mut total_gates = 0usize;
    let mut toffoli_gates = 0usize;
    for i in 0..inst.t {
        let k = inst.t - 1 - i;
        let mult = inst.mults[k];
        match inst.oracle {
            Oracle::Permutation => {}
            Oracle::Beauregard => {
                let lay = inst.layout();
                let c = shor_arith::controlled_ua(&lay, 0, mult, inst.n_mod);
                total_gates += c.ops.len() + 2 + usize::from(y != 0);
            }
            Oracle::Ripple | Oracle::Windowed(_) => {
                let (c, _) = sliced::oracle_block(inst, mult);
                let (g_tot, g_tof) = crate::shor_ripple::gate_counts(&c);
                total_gates += g_tot + 2 + usize::from(y != 0);
                toffoli_gates += g_tof;
            }
        }
        s.round(inst, i, y);
        peak_stored = peak_stored.max(s.stored());
        peak_bytes = peak_bytes.max(s.bytes());
        // same random draw as StateVector::measure_qubit
        let p1 = s.prob_one(0);
        let bit = rng.random::<f64>() < p1;
        s.collapse(0, bit);
        s.reset_control(bit); // recycle the control qubit
        if bit {
            y |= 1 << i;
            if matches!(
                inst.oracle,
                Oracle::Beauregard | Oracle::Ripple | Oracle::Windowed(_)
            ) {
                total_gates += 1;
            }
        }
    }
    let (order, factor) = postprocess(inst.n_mod, inst.a, y, inst.t as u32);
    SemiRun {
        a: inst.a,
        measured: y,
        order,
        factor,
        qubits: inst.qubits(),
        peak_stored,
        peak_bytes,
        total_gates,
        toffoli_gates,
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
        y: u128,
        p: f64,
        prune: f64,
        out: &mut [f64],
    ) {
        if i == inst.t {
            out[y as usize] = p;
            return;
        }
        s.round(inst, i, y);
        let p1 = s.prob_one(0);
        for bit in [false, true] {
            let pb = if bit { p1 } else { 1.0 - p1 };
            if pb <= prune {
                continue;
            }
            let mut c = s.clone();
            c.collapse(0, bit);
            c.reset_control(bit);
            walk(
                inst,
                c,
                i + 1,
                y | (u128::from(bit) << i),
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

/// The semiclassical circuit as a plain [`Circuit`] with the Cuccaro ripple-carry oracle.
pub fn semiclassical_ripple_circuit(n_mod: u64, a: u64) -> Circuit {
    let inst = Instance::new(n_mod, a, Oracle::Ripple);
    let lay = inst.ripple_layout();
    let mut c = Circuit::new(inst.qubits());
    c.x(1); // work register |1>
    for i in 0..inst.t {
        let k = inst.t - 1 - i;
        if i > 0 {
            c.c_if(i - 1, Gate::X(0));
        }
        c.h(0);
        c.append(&crate::shor_ripple::controlled_ua(
            &lay,
            0,
            inst.mults[k],
            n_mod,
        ));
        for l in 0..i {
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
    /// [`fused::FusedDense`] (permutation oracle only).
    FusedF64,
    FusedF32,
    /// [`fused::FusedSparse`] (permutation oracle only).
    FusedSparse,
    /// [`sliced::SlicedState`] f64 amplitudes (reversible gate-level oracles).
    SlicedF64,
    /// [`sliced::SlicedState`] f32 amplitudes.
    SlicedF32,
}

/// Runs one semiclassical order finding with the given backend.
pub fn order_finding<R: Rng + ?Sized>(inst: &Instance, backend: Backend, rng: &mut R) -> SemiRun {
    match backend {
        Backend::DenseF64 => run_semiclassical(inst, dense_initial::<f64>(inst), rng),
        Backend::DenseF32 => run_semiclassical(inst, dense_initial::<f32>(inst), rng),
        Backend::Sparse => run_semiclassical(inst, sparse_initial(inst), rng),
        Backend::FusedF64 => run_semiclassical(inst, fused::FusedDense::<f64>::new(inst), rng),
        Backend::FusedF32 => run_semiclassical(inst, fused::FusedDense::<f32>::new(inst), rng),
        Backend::FusedSparse => run_semiclassical(inst, fused::FusedSparse::new(inst), rng),
        Backend::SlicedF64 => run_semiclassical(inst, sliced::SlicedState::<f64>::new(inst), rng),
        Backend::SlicedF32 => run_semiclassical(inst, sliced::SlicedState::<f32>::new(inst), rng),
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
    factor_semiclassical_with_options(n_mod, oracle, backend, tries, true, false, rng)
}

/// Same as [`factor_semiclassical`] with explicit execution options (`blocked`, `gate_by_gate`).
pub fn factor_semiclassical_with_options<R: Rng + ?Sized>(
    n_mod: u64,
    oracle: Oracle,
    backend: Backend,
    tries: usize,
    blocked: bool,
    gate_by_gate: bool,
    rng: &mut R,
) -> (Option<(u64, u64)>, Vec<SemiRun>) {
    let mut runs = Vec::new();
    for _ in 0..tries {
        let a = rng.random_range(2..n_mod - 1);
        if gcd(a, n_mod) > 1 {
            continue; // lucky classical guess; skip so the quantum part runs
        }
        let mut inst = Instance::new(n_mod, a, oracle);
        inst.blocked = blocked;
        inst.gate_by_gate = gate_by_gate;
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
    fn convergents_match_the_u64_version() {
        for (x, t) in [(3u64, 3u32), (64444, 16), (35145706927192, 48)] {
            let old = crate::algorithms::convergent_denominators(x, t);
            let new: Vec<u64> = convergents(u128::from(x), t)
                .into_iter()
                .map(|q| q as u64)
                .collect();
            assert_eq!(old, new);
        }
    }

    #[test]
    fn postprocess_recovers_order_when_s_and_r_share_a_factor() {
        // N = 143, a = 2 has order 60; s = 18 shares 6 with 60, so the
        // continued fractions of 18/60 only give 10.
        let (n, a, r, t) = (143u64, 2u64, 60u64, 16u32);
        let measured = ((18u128 << t) + u128::from(r) / 2) / u128::from(r);
        let (old, _) = crate::algorithms::shor_postprocess(n, a, measured as u64, t);
        assert_ne!(old, Some(60));
        let (order, factor) = postprocess(n, a, measured, t);
        assert_eq!(order, Some(60));
        assert!(matches!(factor, Some(11) | Some(13)));
    }

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
