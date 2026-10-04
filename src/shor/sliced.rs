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

/// Default lane count: one slice holds `64 * LANES` branches. Override with
/// the environment variable `QSIM_SLICE_LANES` (4, 8, 16 or 32).
pub const LANES: usize = 16;

fn lanes() -> usize {
    static L: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *L.get_or_init(|| {
        std::env::var("QSIM_SLICE_LANES")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|l| [4usize, 8, 16, 32].contains(l))
            .unwrap_or(LANES)
    })
}

#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    static A: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *A.get_or_init(|| {
        std::env::var_os("QSIM_NO_AVX2").is_none() && std::arch::is_x86_feature_detected!("avx2")
    })
}

/// In-place 64×64 bit-matrix transpose: afterwards bit `i` of `a[j]` is
/// bit `j` of the old `a[i]`.
#[inline]
pub fn transpose64(a: &mut [u64; 64]) {
    let mut j = 32usize;
    let mut m: u64 = 0x0000_0000_FFFF_FFFF;
    while j != 0 {
        let mut k = 0usize;
        while k < 64 {
            let t = ((a[k] >> j) ^ a[k + j]) & m;
            a[k + j] ^= t;
            a[k] ^= t << j;
            k = (k + j + 1) & !j;
        }
        j >>= 1;
        m ^= m << j;
    }
}

/// Applies raw `w[t] ^= w[a] & w[b]` steps with the same AVX2 runtime
/// dispatch as [`SlicedProgram::eval`]. Used by the noisy engine
/// ([`super::noisy`]), whose programs also address a sign word.
///
/// # Safety
/// Every index in `ops` must be `< w.len()` (the noisy engine checks this
/// once when it builds a program).
pub(crate) unsafe fn eval_raw_unchecked<const L: usize>(ops: &[[u32; 3]], w: &mut [[u64; L]]) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 support was detected at run time; indices checked by the caller.
        unsafe { eval_avx2::<L>(ops, w) };
        return;
    }
    eval_body::<L>(ops, w);
}

/// A reversible circuit compiled to `w[t] ^= w[c1] & w[c2]` steps.
#[derive(Clone, Debug)]
pub struct SlicedProgram {
    /// Number of circuit qubits; word `nq` is the all-ones word.
    pub nq: usize,
    ops: Vec<[u32; 3]>,
    /// Gates in the source circuit (a SWAP counts once).
    pub gates: usize,
}

#[inline(always)]
fn eval_body<const L: usize>(ops: &[[u32; 3]], w: &mut [[u64; L]]) {
    for &[t, a, b] in ops {
        // SAFETY: every index was checked against nq at compile time and
        // the caller checked w.len() > nq.
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn eval_avx2<const L: usize>(ops: &[[u32; 3]], w: &mut [[u64; L]]) {
    eval_body::<L>(ops, w);
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
        #[cfg(target_arch = "x86_64")]
        if has_avx2() {
            // SAFETY: AVX2 support was detected at run time.
            unsafe { eval_avx2::<L>(&self.ops, w) };
            return;
        }
        eval_body::<L>(&self.ops, w);
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

/// An output slot that receives a work-register value.
pub trait KeySlot: Send {
    fn set_key(&mut self, y: u64);
}
impl KeySlot for u64 {
    #[inline]
    fn set_key(&mut self, y: u64) {
        *self = y;
    }
}
impl<P: Send> KeySlot for (u64, P) {
    #[inline]
    fn set_key(&mut self, y: u64) {
        self.0 = y;
    }
}

/// What [`eval_block_into`] does with the output work-register values.
pub enum BlockOut<'a, S: KeySlot> {
    /// Write `U x_i` into slot `i` (e.g. the key of a `(key, amplitude)`
    /// pair, so the join needs no index indirection).
    Keys(&'a mut [S]),
    /// Assert `U x_i = x_i` (control-0 branches).
    Identity,
}

/// Evaluates the block on `|ctrl>|x>|0…>` for every `x` in `xs`, gate by
/// gate on bit slices, and returns the output work-register values.
/// Panics if any output has a changed control or a non-zero ancilla.
pub fn eval_block(prog: &SlicedProgram, io: &SliceIo, ctrl: bool, xs: &[u64]) -> Vec<u64> {
    let mut out = vec![0u64; xs.len()];
    eval_block_into(prog, io, ctrl, xs, BlockOut::Keys(&mut out[..]));
    out
}

/// [`eval_block`] with an explicit output mode.
pub fn eval_block_into<S: KeySlot>(
    prog: &SlicedProgram,
    io: &SliceIo,
    ctrl: bool,
    xs: &[u64],
    out: BlockOut<S>,
) {
    match lanes() {
        4 => eval_block_l::<4, S>(prog, io, ctrl, xs, out),
        8 => eval_block_l::<8, S>(prog, io, ctrl, xs, out),
        32 => eval_block_l::<32, S>(prog, io, ctrl, xs, out),
        _ => eval_block_l::<16, S>(prog, io, ctrl, xs, out),
    }
}

fn eval_block_l<const L: usize, S: KeySlot>(
    prog: &SlicedProgram,
    io: &SliceIo,
    ctrl: bool,
    xs: &[u64],
    out: BlockOut<S>,
) {
    let b = 64 * L;
    let nq = prog.nq;
    assert!(io.x.len() <= 64);
    let mut is_reg = vec![false; nq];
    is_reg[io.ctrl] = true;
    for &q in &io.x {
        is_reg[q] = true;
    }
    let anc: Vec<usize> = (0..nq).filter(|&q| !is_reg[q]).collect();
    let batch = |w: &mut Vec<[u64; L]>, inp: &[u64]| -> Vec<u64> {
        for wq in w.iter_mut() {
            *wq = [0; L];
        }
        w[nq] = [u64::MAX; L];
        // valid-lane mask
        let mut valid = [0u64; L];
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
        // transpose in (64x64 bit blocks)
        let mut blk = [0u64; 64];
        for (l, c) in inp.chunks(64).enumerate() {
            blk[..c.len()].copy_from_slice(c);
            blk[c.len()..].fill(0);
            transpose64(&mut blk);
            for (j, &q) in io.x.iter().enumerate() {
                w[q][l] = blk[j];
            }
        }
        prog.eval(w);
        // checks: control unchanged, ancillas zero (valid lanes)
        for l in 0..L {
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
        let mut o = vec![0u64; inp.len()];
        for (l, oc) in o.chunks_mut(64).enumerate() {
            blk.fill(0);
            for (j, &q) in io.x.iter().enumerate() {
                blk[j] = w[q][l];
            }
            transpose64(&mut blk);
            oc.copy_from_slice(&blk[..oc.len()]);
        }
        o
    };
    let init = || vec![[0u64; L]; nq + 1];
    match out {
        BlockOut::Keys(out) => {
            assert_eq!(out.len(), xs.len());
            xs.par_chunks(b)
                .zip(out.par_chunks_mut(b))
                .for_each_init(init, |w, (inp, outp)| {
                    for (o, y) in outp.iter_mut().zip(batch(w, inp)) {
                        o.set_key(y);
                    }
                });
        }
        BlockOut::Identity => {
            xs.par_chunks(b).for_each_init(init, |w, inp| {
                assert!(
                    batch(w, inp) == inp,
                    "controlled-U with control 0 is not the identity"
                );
            });
        }
    }
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
        Oracle::WindowedOpt(w) => {
            let lay = crate::shor_window::WindowLayout::new(inst.m, w);
            let c = crate::shor_superopt::controlled_ua(
                &lay,
                mult,
                inst.n_mod,
                &crate::shor_superopt::Opts::ALL,
            );
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
    /// `(U x, ψ_x)` for every stored `x`, sorted by `U x` (the control-1
    /// branches after the block).
    uk: Vec<(u64, Complex<T>)>,
    ph: Complex64,
    p1: f64,
    peak: usize,
    /// Total gate applications (gates × branches) performed.
    pub gate_branch_ops: u128,
    /// Skip evaluating the control-0 branches (they are the identity on a
    /// clean input; this halves the work but no longer *checks* it).
    pub skip_ctrl0: bool,
    /// Also materialise the post-measurement state after the last round
    /// (default `false`: the measured integer is complete after the last
    /// `P(1)`, so the final collapse — the largest support, `≈ r` — is
    /// skipped and the state is dropped).
    pub keep_final: bool,
    final_round: bool,
    /// Seconds spent in: building+compiling the circuit, control-1 gate
    /// evaluation, control-0 gate evaluation, sort, P(1) merge, collapse.
    pub prof: [f64; 6],
    /// `P(control = 1)` of every round, in order.
    pub p1_trace: Vec<f64>,
    /// `|supp ψ|` at the start of every round (index `i` = after `i`
    /// measured bits); the support after the last collapse is `nnz()`.
    pub support_trace: Vec<usize>,
}

impl<T: Real> Drop for SlicedState<T> {
    fn drop(&mut self) {
        if std::env::var_os("QSIM_SLICE_PROFILE").is_some() && self.gate_branch_ops > 0 {
            let p = self.prof;
            eprintln!(
                "[sliced profile] build {:.3}s  eval c=1 {:.3}s  eval c=0 {:.3}s  sort {:.3}s  p1 {:.3}s  collapse {:.3}s  peak support {}",
                p[0], p[1], p[2], p[3], p[4], p[5], self.peak
            );
        }
        if std::env::var_os("QSIM_SLICE_TRACE").is_some() && self.gate_branch_ops > 0 {
            let t: Vec<String> = self.p1_trace.iter().map(|p| format!("{p:.17e}")).collect();
            eprintln!("[sliced p1] {}", t.join(" "));
            let t: Vec<String> = self.support_trace.iter().map(|p| p.to_string()).collect();
            eprintln!("[sliced support] {} {}", t.join(" "), self.keys.len());
        }
    }
}

fn cvt<T: Real>(z: Complex64) -> Complex<T> {
    Complex::new(T::from_f64(z.re), T::from_f64(z.im))
}
fn c64<T: Real>(z: Complex<T>) -> Complex64 {
    Complex64::new(z.re.to_f64(), z.im.to_f64())
}

impl<T: Real> SlicedState<T> {
    pub fn new(inst: &Instance) -> Self {
        assert!(matches!(
            inst.oracle,
            Oracle::Ripple | Oracle::Windowed(_) | Oracle::WindowedOpt(_)
        ));
        Self {
            keys: vec![1],
            amps: vec![Complex::new(T::one(), T::zero())],
            uk: Vec::new(),
            ph: Complex64::new(1.0, 0.0),
            p1: 0.0,
            peak: 1,
            gate_branch_ops: 0,
            skip_ctrl0: false,
            keep_final: false,
            final_round: false,
            prof: [0.0; 6],
            p1_trace: Vec::new(),
            support_trace: Vec::new(),
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
    /// `uk`: returns `(a_lo, a_hi, b_lo, b_hi)` per chunk.
    fn chunks(&self) -> Vec<(usize, usize, usize, usize)> {
        let p = (rayon::current_num_threads() * 8).max(1);
        let n = self.keys.len();
        let mut bounds: Vec<u64> = (1..p).map(|i| self.keys[n * i / p]).collect();
        bounds.dedup();
        let mut out = Vec::with_capacity(bounds.len() + 1);
        let (mut a0, mut b0) = (0, 0);
        for &k in &bounds {
            let a1 = self.keys.partition_point(|&x| x < k);
            let b1 = self.uk.partition_point(|e| e.0 < k);
            out.push((a0, a1, b0, b1));
            (a0, b0) = (a1, b1);
        }
        out.push((a0, n, b0, self.uk.len()));
        out
    }

    /// Visits the union of `supp ψ` and `supp Uψ` in key order on
    /// `[a0,a1) × [b0,b1)`, calling `f(key, ψ_key, (Uψ)_key)`.
    fn merge(
        &self,
        (a0, a1, b0, b1): (usize, usize, usize, usize),
        mut f: impl FnMut(u64, Complex64, Complex64),
    ) {
        let z = Complex64::zero();
        let (mut i, mut j) = (a0, b0);
        while i < a1 || j < b1 {
            let ka = if i < a1 { self.keys[i] } else { u64::MAX };
            let kb = if j < b1 { self.uk[j].0 } else { u64::MAX };
            if i < a1 && (j >= b1 || ka < kb) {
                f(ka, c64(self.amps[i]), z);
                i += 1;
            } else if j < b1 && (i >= a1 || kb < ka) {
                f(kb, z, c64(self.uk[j].1));
                j += 1;
            } else {
                f(ka, c64(self.amps[i]), c64(self.uk[j].1));
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
        self.support_trace.push(self.keys.len());
        self.final_round = i + 1 == inst.t;
        let t0 = std::time::Instant::now();
        let (c, io) = oracle_block(inst, mult);
        let prog = SlicedProgram::compile(&c).expect("reversible oracle");
        drop(c);
        let t1 = std::time::Instant::now();
        assert!(self.keys.len() < u32::MAX as usize);
        // control = 1 branches: the gate-level circuit computes U x
        self.uk = Vec::new();
        let mut uk: Vec<(u64, Complex<T>)> = self.amps.par_iter().map(|&a| (0u64, a)).collect();
        eval_block_into(&prog, &io, true, &self.keys, BlockOut::Keys(&mut uk[..]));
        let t2 = std::time::Instant::now();
        let mut branches = self.keys.len() as u128;
        if !self.skip_ctrl0 {
            // control = 0 branches: same circuit, must give x back
            eval_block_into::<u64>(&prog, &io, false, &self.keys, BlockOut::Identity);
            branches *= 2;
        }
        let t3 = std::time::Instant::now();
        self.gate_branch_ops += branches * prog.gates as u128;
        uk.par_sort_unstable_by_key(|e| e.0);
        // U is a permutation: distinct inputs must give distinct outputs
        assert!(
            uk.par_windows(2).all(|w| w[0].0 != w[1].0),
            "oracle block is not injective on the support"
        );
        self.uk = uk;
        let t4 = std::time::Instant::now();
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
        self.p1_trace.push(self.p1);
        self.peak = self.peak.max(self.keys.len());
        let t5 = std::time::Instant::now();
        let d = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64();
        self.prof[0] += d(t0, t1);
        self.prof[1] += d(t1, t2);
        self.prof[2] += d(t2, t3);
        self.prof[3] += d(t3, t4);
        self.prof[4] += d(t4, t5);
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
        if self.final_round && !self.keep_final {
            self.keys = Vec::new();
            self.amps = Vec::new();
            self.uk = Vec::new();
            return;
        }
        let t0 = std::time::Instant::now();
        let parts: Vec<(Vec<u64>, Vec<Complex<T>>)> = self
            .chunks()
            .into_par_iter()
            .map(|ch| {
                let cap = (ch.1 - ch.0) + (ch.3 - ch.2);
                let mut ks = Vec::with_capacity(cap);
                let mut vs = Vec::with_capacity(cap);
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
        self.keys = Vec::new();
        self.amps = Vec::new();
        self.uk = Vec::new();
        let total: usize = parts.iter().map(|p| p.0.len()).sum();
        // parallel concatenation into exact-size buffers
        let mut keys = vec![0u64; total];
        let mut amps = vec![Complex::<T>::zero(); total];
        {
            let mut kd: &mut [u64] = &mut keys;
            let mut ad: &mut [Complex<T>] = &mut amps;
            let mut jobs = Vec::with_capacity(parts.len());
            for (ks, vs) in &parts {
                let (k0, k1) = std::mem::take(&mut kd).split_at_mut(ks.len());
                let (a0, a1) = std::mem::take(&mut ad).split_at_mut(vs.len());
                jobs.push((k0, a0, ks, vs));
                kd = k1;
                ad = a1;
            }
            jobs.into_par_iter().for_each(|(k0, a0, ks, vs)| {
                k0.copy_from_slice(ks);
                a0.copy_from_slice(vs);
            });
        }
        drop(parts);
        self.peak = self.peak.max(keys.len());
        self.keys = keys;
        self.amps = amps;
        self.prof[5] += t0.elapsed().as_secs_f64();
    }
    fn reset_control(&mut self, _bit: bool) {}
    fn bytes(&self) -> usize {
        self.keys.capacity() * (8 + std::mem::size_of::<Complex<T>>())
            + self.uk.capacity() * std::mem::size_of::<(u64, Complex<T>)>()
    }
    /// Peak support size `|supp ψ|` (work-register values); each round
    /// evaluates the circuit on `2 |supp ψ|` basis states.
    fn stored(&self) -> usize {
        self.peak
    }
    fn work_ops(&self) -> u128 {
        self.gate_branch_ops
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shor_ripple::{controlled_ua, eval_circuit_on_key, RippleLayout};

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn transpose64_is_a_transpose() {
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut a = [0u64; 64];
        for x in a.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *x = s;
        }
        let orig = a;
        transpose64(&mut a);
        for i in 0..64 {
            for j in 0..64 {
                assert_eq!((a[j] >> i) & 1, (orig[i] >> j) & 1);
            }
        }
    }

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
