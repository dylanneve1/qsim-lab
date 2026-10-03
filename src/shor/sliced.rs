//! Bit-sliced branch tracking for the gate-level reversible oracles
//! (ripple-carry [`crate::shor_ripple`] and windowed [`crate::shor_window`]).
//!
//! # What is simulated
//!
//! The semiclassical circuit is unchanged: one recycled control (qubit 0), the
//! work register `x` (qubits `1..=n`) and every ancilla of the gate-level
//! oracle. Each round is `H(0)`, the controlled-`U` circuit (X / CNOT / CCX
//! only), `Phase(0, φ)`, `H(0)`, measure qubit 0, recycle it.
//!
//! The controlled-`U` circuit contains only permutation gates, so it maps
//! every computational basis state to one basis state; the exact state is a
//! list of (basis state, amplitude) "branches", and applying the circuit means
//! pushing every branch through every gate — what [`crate::sparse::SparseState`]
//! does gate by gate. Here the branches are *bit-sliced*: a batch of `64·L`
//! branches is stored as one `[u64; L]` word per qubit, and a gate becomes
//! `w[t] ^= w[c1] & w[c2]` on those words (X and CNOT use an all-ones word as
//! the missing controls). Every gate of the circuit is applied to every
//! branch; nothing about modular multiplication is computed classically.
//!
//! Between rounds the control is `|0>` and (checked, see below) every
//! ancilla is `|0>`, so only the work-register value of each branch is
//! stored. A round evaluates the circuit on both `|0>|x>|0…>` and
//! `|1>|x>|0…>` for every stored `x`, **checks** that the control came back
//! unchanged and every ancilla is 0 in every output (otherwise it panics —
//! the dropped ancilla qubits would then be entangled and the bookkeeping
//! below would be wrong), and that the control-0 branch returned `x`.
//! The control-qubit algebra is then exactly that of [`super::fused`]:
//! `P(1) = ‖ψ − e^{iφ}Uψ‖²/4`, collapse to `(ψ ± e^{iφ}Uψ)/(2√p)`, where
//! `Uψ` is assembled from the *gate-evaluated* outputs (a sort-merge join on
//! the output keys).
//!
//! # Cost
//!
//! `O(|supp ψ| · G)` bit operations per round (`G` = gates in the round's
//! controlled-`U`), divided by the slice width `64·L`; `|supp ψ| ≤ r`, the
//! order of `a`. For a generic semiprime and random base `r ≈ N/c`, so this
//! is exponential in the bit length — it is the sparse state of the real
//! circuit, made fast, not a factoring speed-up.

use super::{Instance, Oracle, OrderFindingState};
use crate::circuit::{Circuit, Op};
use crate::gate::Gate;
use crate::statevector::Real;
use num_complex::{Complex, Complex64};
use num_traits::Zero;
use rayon::prelude::*;

/// Lane count: one slice holds `64 * LANES` branches.
pub const LANES: usize = 8;

/// A reversible circuit compiled to `w[t] ^= w[c1] & w[c2]` steps.
#[derive(Clone, Debug)]
pub struct SlicedProgram {
    /// Number of circuit qubits; word `nq` is the all-ones word.
    pub nq: usize,
    ops: Vec<[u32; 3]>,
    /// Gates in the source circuit (a SWAP counts once).
    pub gates: usize,
}

impl SlicedProgram {
    /// Compiles a circuit of X / CNOT / CCX / SWAP gates. Returns an error for
    /// any other gate or non-gate operation.
    pub fn compile(c: &Circuit) -> Result<Self, String> {
        let nq = c.num_qubits;
        let one = nq as u32;
        let mut ops = Vec::with_capacity(c.ops.len());
        let mut gates = 0;
        let q = |i: usize| -> Result<u32, String> {
            if i < nq {
                Ok(i as u32)
            } else {
                Err(format!("qubit {i} out of range ({nq} qubits)"))
            }
        };
        for op in &c.ops {
            let Op::Gate(g) = op else {
                return Err(format!("non-gate op {op:?} in a reversible block"));
            };
            gates += 1;
            match *g {
                Gate::X(t) => ops.push([q(t)?, one, one]),
                Gate::Cnot(c, t) => ops.push([q(t)?, q(c)?, one]),
                Gate::Ccx(c1, c2, t) => ops.push([q(t)?, q(c1)?, q(c2)?]),
                Gate::Swap(a, b) => {
                    let (a, b) = (q(a)?, q(b)?);
                    ops.push([a, b, one]);
                    ops.push([b, a, one]);
                    ops.push([a, b, one]);
                }
                ref g => return Err(format!("gate {g:?} is not a basis-state permutation")),
            }
        }
        Ok(Self { nq, ops, gates })
    }

    /// Number of slice steps.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Applies the program to `w` (`nq + 1` words, the last all ones).
    #[inline]
    pub fn eval<const L: usize>(&self, w: &mut [[u64; L]]) {
        assert!(w.len() > self.nq);
        assert!(w[self.nq].iter().all(|&x| x == u64::MAX));
        for &[t, a, b] in &self.ops {
            // SAFETY: every index was checked against nq at compile time and
            // w.len() > nq.
            unsafe {
                let x = *w.get_unchecked(a as usize);
                let y = *w.get_unchecked(b as usize);
                let wt = w.get_unchecked_mut(t as usize);
                for l in 0..L {
                    wt[l] ^= x[l] & y[l];
                }
            }
        }
    }
}

/// Which qubits are read and written between rounds; every other qubit is an
/// ancilla that must be 0 before and after the block.
#[derive(Clone, Debug)]
pub struct SliceIo {
    pub ctrl: usize,
    /// Work register, LSB first.
    pub x: Vec<usize>,
}

/// Evaluates the block on `|ctrl>|x>|0…>` for every `x` in `xs`, gate by
/// gate on bit slices, and returns the output work-register values.
/// Panics if any output has a changed control or a non-zero ancilla.
pub fn eval_block(prog: &SlicedProgram, io: &SliceIo, ctrl: bool, xs: &[u64]) -> Vec<u64> {
    const B: usize = 64 * LANES;
    let nq = prog.nq;
    assert!(io.x.len() <= 64);
    let mut is_reg = vec![false; nq];
    is_reg[io.ctrl] = true;
    for &q in &io.x {
        is_reg[q] = true;
    }
    let anc: Vec<usize> = (0..nq).filter(|&q| !is_reg[q]).collect();
    let mut out = vec![0u64; xs.len()];
    xs.par_chunks(B)
        .zip(out.par_chunks_mut(B))
        .for_each_init(
            || vec![[0u64; LANES]; nq + 1],
            |w, (inp, outp)| {
                for wq in w.iter_mut() {
                    *wq = [0; LANES];
                }
                w[nq] = [u64::MAX; LANES];
                // valid-lane mask
                let mut valid = [0u64; LANES];
                for (l, v) in valid.iter_mut().enumerate() {
                    let lo = l * 64;
                    if inp.len() > lo {
                        let k = (inp.len() - lo).min(64);
                        *v = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };
                    }
                }
                if ctrl {
                    w[io.ctrl] = valid;
                }
                // transpose in
                for (j, &q) in io.x.iter().enumerate() {
                    let wq = &mut w[q];
                    for (i, &x) in inp.iter().enumerate() {
                        wq[i >> 6] |= ((x >> j) & 1) << (i & 63);
                    }
                }
                prog.eval(w);
                // checks: control unchanged, ancillas zero (valid lanes)
                for l in 0..LANES {
                    let want = if ctrl { valid[l] } else { 0 };
                    assert_eq!(
                        w[io.ctrl][l] & valid[l],
                        want,
                        "control qubit changed by the oracle block"
                    );
                    let mut dirty = 0u64;
                    for &q in &anc {
                        dirty |= w[q][l];
                    }
                    assert_eq!(
                        dirty & valid[l],
                        0,
                        "ancillas did not return to 0 (lane word {l})"
                    );
                }
                // transpose out
                for o in outp.iter_mut() {
                    *o = 0;
                }
                for (j, &q) in io.x.iter().enumerate() {
                    let wq = &w[q];
                    for (i, o) in outp.iter_mut().enumerate() {
                        *o |= ((wq[i >> 6] >> (i & 63)) & 1) << j;
                    }
                }
            },
        );
    out
}

/// The controlled-`U_mult` gate-level block for a reversible oracle.
pub fn oracle_block(inst: &Instance, mult: u64) -> (Circuit, SliceIo) {
    match inst.oracle {
        Oracle::Ripple => {
            let lay = inst.ripple_layout();
            let c = crate::shor_ripple::controlled_ua(&lay, 0, mult, inst.n_mod);
            (
                c,
                SliceIo {
                    ctrl: lay.ctrl,
                    x: lay.x.clone(),
                },
            )
        }
        Oracle::Windowed(w) => {
            let lay = crate::shor_window::WindowLayout::new(inst.m, w);
            let c = crate::shor_window::controlled_ua(&lay, mult, inst.n_mod);
            (
                c,
                SliceIo {
                    ctrl: lay.ctrl,
                    x: lay.x.clone(),
                },
            )
        }
        o => panic!("sliced branch tracking needs a reversible oracle, not {o:?}"),
    }
}

/// Exact state of the semiclassical circuit with a reversible oracle,
/// stored as the work-register support (sorted) and amplitudes.
#[derive(Clone, Debug)]
pub struct SlicedState<T: Real> {
    keys: Vec<u64>,
    amps: Vec<Complex<T>>,
    /// Output keys of the control-1 branches, sorted, with source index.
    ukeys: Vec<(u64, u32)>,
    ph: Complex64,
    p1: f64,
    peak: usize,
    /// Total gate applications (gates × branches) performed.
    pub gate_branch_ops: u128,
    /// Skip evaluating the control-0 branches (they are the identity on a
    /// clean input; this halves the work but no longer *checks* it).
    pub skip_ctrl0: bool,
}

fn cvt<T: Real>(z: Complex64) -> Complex<T> {
    Complex::new(T::from_f64(z.re), T::from_f64(z.im))
}
fn c64<T: Real>(z: Complex<T>) -> Complex64 {
    Complex64::new(z.re.to_f64(), z.im.to_f64())
}

impl<T: Real> SlicedState<T> {
    pub fn new(inst: &Instance) -> Self {
        assert!(matches!(inst.oracle, Oracle::Ripple | Oracle::Windowed(_)));
        Self {
            keys: vec![1],
            amps: vec![Complex::new(T::one(), T::zero())],
            ukeys: Vec::new(),
            ph: Complex64::new(1.0, 0.0),
            p1: 0.0,
            peak: 1,
            gate_branch_ops: 0,
            skip_ctrl0: false,
        }
    }

    pub fn nnz(&self) -> usize {
        self.keys.len()
    }

    /// The stored work-register amplitudes `(x, ψ_x)`, sorted by `x`.
    pub fn work(&self) -> impl Iterator<Item = (u64, Complex<T>)> + '_ {
        self.keys.iter().copied().zip(self.amps.iter().copied())
    }

    /// Splits the key space into chunks for a parallel merge of `keys` and
    /// `ukeys`: returns `(a_lo, a_hi, b_lo, b_hi)` per chunk.
    fn chunks(&self) -> Vec<(usize, usize, usize, usize)> {
        let p = (rayon::current_num_threads() * 8).max(1);
        let n = self.keys.len();
        let mut bounds: Vec<u64> = (1..p).map(|i| self.keys[n * i / p]).collect();
        bounds.dedup();
        let mut out = Vec::with_capacity(bounds.len() + 1);
        let (mut a0, mut b0) = (0, 0);
        for &k in &bounds {
            let a1 = self.keys.partition_point(|&x| x < k);
            let b1 = self.ukeys.partition_point(|e| e.0 < k);
            out.push((a0, a1, b0, b1));
            (a0, b0) = (a1, b1);
        }
        out.push((a0, n, b0, self.ukeys.len()));
        out
    }

    /// Visits the union of `supp ψ` and `supp Uψ` in key order on
    /// `[a0,a1) × [b0,b1)`, calling `f(key, ψ_key, (Uψ)_key)`.
    fn merge(&self, (a0, a1, b0, b1): (usize, usize, usize, usize), mut f: impl FnMut(u64, Complex64, Complex64)) {
        let z = Complex64::zero();
        let (mut i, mut j) = (a0, b0);
        while i < a1 || j < b1 {
            let ka = if i < a1 { self.keys[i] } else { u64::MAX };
            let kb = if j < b1 { self.ukeys[j].0 } else { u64::MAX };
            if i < a1 && (j >= b1 || ka < kb) {
                f(ka, c64(self.amps[i]), z);
                i += 1;
            } else if j < b1 && (i >= a1 || kb < ka) {
                f(kb, z, c64(self.amps[self.ukeys[j].1 as usize]));
                j += 1;
            } else {
                f(ka, c64(self.amps[i]), c64(self.amps[self.ukeys[j].1 as usize]));
                i += 1;
                j += 1;
            }
        }
    }
}

impl<T: Real> OrderFindingState for SlicedState<T> {
    fn gate(&mut self, _g: &Gate) {
        panic!("SlicedState only runs whole rounds")
    }
    fn ctrl_mul(&mut self, _m: usize, _mult: u64, _inv: u64, _n: u64) {
        panic!("SlicedState only runs gate-level oracle rounds")
    }
    fn round(&mut self, inst: &Instance, i: usize, y_low: u128) {
        let mult = inst.mults[inst.t - 1 - i];
        let (c, io) = oracle_block(inst, mult);
        let prog = SlicedProgram::compile(&c).expect("reversible oracle");
        assert!(self.keys.len() < u32::MAX as usize);
        // control = 1 branches: the gate-level circuit computes U x
        let ys = eval_block(&prog, &io, true, &self.keys);
        let mut branches = self.keys.len() as u128;
        if !self.skip_ctrl0 {
            // control = 0 branches: same circuit, must give x back
            let back = eval_block(&prog, &io, false, &self.keys);
            assert!(
                back == self.keys,
                "controlled-U with control 0 is not the identity"
            );
            branches *= 2;
        }
        self.gate_branch_ops += branches * prog.gates as u128;
        let mut uk: Vec<(u64, u32)> = ys
            .into_iter()
            .enumerate()
            .map(|(j, y)| (y, j as u32))
            .collect();
        uk.par_sort_unstable_by_key(|e| e.0);
        // U is a permutation: distinct inputs must give distinct outputs
        assert!(
            uk.windows(2).all(|w| w[0].0 != w[1].0),
            "oracle block is not injective on the support"
        );
        self.ukeys = uk;
        let phi = if y_low != 0 {
            Instance::correction(i, y_low)
        } else {
            0.0
        };
        self.ph = Complex64::from_polar(1.0, phi);
        let ph = self.ph;
        let s: f64 = self
            .chunks()
            .into_par_iter()
            .map(|ch| {
                let mut acc = 0.0;
                self.merge(ch, |_, a, b| acc += (a - ph * b).norm_sqr());
                acc
            })
            .sum();
        self.p1 = s / 4.0;
        self.peak = self.peak.max(self.keys.len() + self.ukeys.len());
    }
    fn prob_one(&self, q: usize) -> f64 {
        assert_eq!(q, 0);
        self.p1
    }
    fn collapse(&mut self, q: usize, outcome: bool) {
        assert_eq!(q, 0);
        let p = if outcome { self.p1 } else { 1.0 - self.p1 };
        assert!(p > 0.0, "cannot collapse onto a zero-probability outcome");
        let ph = if outcome { -self.ph } else { self.ph };
        let k = 0.5 / p.sqrt();
        let parts: Vec<(Vec<u64>, Vec<Complex<T>>)> = self
            .chunks()
            .into_par_iter()
            .map(|ch| {
                let mut ks = Vec::new();
                let mut vs = Vec::new();
                self.merge(ch, |key, a, b| {
                    let v = (a + ph * b) * k;
                    let v = cvt::<T>(v);
                    if v != Complex::zero() {
                        ks.push(key);
                        vs.push(v);
                    }
                });
                (ks, vs)
            })
            .collect();
        let total: usize = parts.iter().map(|p| p.0.len()).sum();
        let mut keys = Vec::with_capacity(total);
        let mut amps = Vec::with_capacity(total);
        for (ks, vs) in parts {
            keys.extend(ks);
            amps.extend(vs);
        }
        self.keys = keys;
        self.amps = amps;
        self.ukeys = Vec::new();
    }
    fn reset_control(&mut self, _bit: bool) {}
    fn bytes(&self) -> usize {
        self.keys.capacity() * (8 + std::mem::size_of::<Complex<T>>())
            + self.ukeys.capacity() * 16
    }
    fn stored(&self) -> usize {
        self.peak
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shor_ripple::{controlled_ua, eval_circuit_on_key, RippleLayout};

    #[test]
    fn sliced_eval_matches_per_key_eval() {
        for n_mod in [15u64, 21, 35, 143] {
            let n = super::super::work_bits(n_mod);
            let lay = RippleLayout::new(n);
            let c = controlled_ua(&lay, 0, 2, n_mod);
            let prog = SlicedProgram::compile(&c).unwrap();
            let io = SliceIo {
                ctrl: 0,
                x: lay.x.clone(),
            };
            let xs: Vec<u64> = (0..n_mod).collect();
            for ctrl in [false, true] {
                let ys = eval_block(&prog, &io, ctrl, &xs);
                for (&x, &y) in xs.iter().zip(&ys) {
                    let k = eval_circuit_on_key(u64::from(ctrl) | (x << 1), &c);
                    assert_eq!(k & 1, u64::from(ctrl));
                    assert_eq!((k >> 1) & ((1 << n) - 1), y, "N={n_mod} x={x}");
                }
            }
        }
    }
}
