//! Dense state-vector simulation.
//!
//! An `n`-qubit state is stored as its `2^n` complex amplitudes. A gate on
//! qubit `q` never builds a `2^n x 2^n` matrix: it pairs up the amplitudes
//! whose indices differ only in bit `q` (indices `i` and `i | 1<<q`) and
//! applies the 2x2 matrix to each pair. Memory traffic, not arithmetic, is
//! the bottleneck, so the kernels are written as straight loops over
//! contiguous, equally long slices (which LLVM vectorises) and the outer
//! loops are split across threads with rayon.
//!
//! Gates are dispatched to special cases where that saves work: diagonal
//! gates only touch the half of the amplitudes whose target bit is 1, X/CNOT/
//! SWAP are pure memory swaps, and CZ/controlled-phase touch a quarter of
//! the vector.

use crate::circuit::{check_gate, Circuit, SimError, Simulator};
use crate::gate::{mat4_swap_qubits, Gate, Mat2, Mat4};
use num_complex::{Complex, Complex64};
use num_traits::{Float, One, Zero};
use rand::seq::SliceRandom;
use rand::{Rng, RngCore};
use rayon::prelude::*;
use std::fmt::Debug;
use std::iter::Sum;

/// Floating-point type usable for amplitudes (`f32` or `f64`).
pub trait Real: Float + Send + Sync + Default + Debug + Sum + 'static {
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl Real for f32 {
    #[inline]
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl Real for f64 {
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

/// Upper bound on the size of a state vector this crate will allocate.
///
/// 1 GiB allows 26 qubits in single precision (`Complex<f32>`, 8 bytes per
/// amplitude: 512 MiB) or 25 qubits in double precision (512 MiB); one more
/// qubit doubles the size to 1 GiB, which is still accepted. Raise it if your
/// machine has the memory.
pub const MAX_STATE_BYTES: u128 = 1 << 30;

/// Below this many amplitudes everything runs on the calling thread.
const PAR_MIN_LEN: usize = 1 << 14;
/// Unit of work handed to a rayon task (in amplitudes).
const BLOCK: usize = 1 << 14;

#[inline]
fn cvt<T: Real>(z: Complex64) -> Complex<T> {
    Complex::new(T::from_f64(z.re), T::from_f64(z.im))
}

/// A pure state of `n` qubits stored as `2^n` amplitudes.
#[derive(Clone, Debug, PartialEq)]
pub struct StateVector<T: Real = f64> {
    n: usize,
    amps: Vec<Complex<T>>,
}

/// Single-precision state vector.
pub type StateVectorF32 = StateVector<f32>;
/// Double-precision state vector.
pub type StateVectorF64 = StateVector<f64>;

/// Bytes needed for an `n`-qubit state vector with amplitude type `T`.
pub fn state_bytes<T: Real>(n: usize) -> u128 {
    (1u128 << n.min(127)) * std::mem::size_of::<Complex<T>>() as u128
}

/// Calls `f(lo, hi)` on matching runs of amplitudes whose index has bit `q`
/// clear (`lo`) and set (`hi`). `lo[i]` and `hi[i]` always form a pair.
fn for_each_pair<T, F>(amps: &mut [Complex<T>], q: usize, f: F)
where
    T: Real,
    F: Fn(&mut [Complex<T>], &mut [Complex<T>]) + Sync,
{
    let s = 1usize << q;
    let split = |chunk: &mut [Complex<T>]| {
        let (lo, hi) = chunk.split_at_mut(s);
        f(lo, hi)
    };
    if amps.len() < PAR_MIN_LEN {
        amps.chunks_mut(2 * s).for_each(split);
    } else if 2 * s <= BLOCK {
        // Many small pairs: give each task a block containing many of them.
        amps.par_chunks_mut(BLOCK)
            .for_each(|blk| blk.chunks_mut(2 * s).for_each(&split));
    } else {
        // Few, long runs: split each run into blocks as well.
        amps.par_chunks_mut(2 * s).for_each(|chunk| {
            let (lo, hi) = chunk.split_at_mut(s);
            lo.par_chunks_mut(BLOCK / 2)
                .zip(hi.par_chunks_mut(BLOCK / 2))
                .for_each(|(a, b)| f(a, b));
        });
    }
}

/// Two-qubit analogue of [`for_each_pair`] for bits `l < h`: calls
/// `f(a00, a01, a10, a11)` where `aXY` holds the amplitudes with bit `h` = X
/// and bit `l` = Y, element-aligned.
fn for_each_quad<T, F>(amps: &mut [Complex<T>], l: usize, h: usize, f: F)
where
    T: Real,
    F: Fn(&mut [Complex<T>], &mut [Complex<T>], &mut [Complex<T>], &mut [Complex<T>]) + Sync,
{
    debug_assert!(l < h);
    let sl = 1usize << l;
    let sh = 1usize << h;
    // Given the h=0 and h=1 halves of a run, walk the l-structure inside them.
    let inner = |c0: &mut [Complex<T>], c1: &mut [Complex<T>]| {
        for (x0, x1) in c0.chunks_mut(2 * sl).zip(c1.chunks_mut(2 * sl)) {
            let (a00, a01) = x0.split_at_mut(sl);
            let (a10, a11) = x1.split_at_mut(sl);
            f(a00, a01, a10, a11);
        }
    };
    let split_h = |chunk: &mut [Complex<T>]| {
        let (c0, c1) = chunk.split_at_mut(sh);
        inner(c0, c1)
    };
    if amps.len() < PAR_MIN_LEN {
        amps.chunks_mut(2 * sh).for_each(split_h);
    } else if 2 * sh <= BLOCK {
        amps.par_chunks_mut(BLOCK)
            .for_each(|blk| blk.chunks_mut(2 * sh).for_each(&split_h));
    } else if 2 * sl <= BLOCK / 2 {
        amps.par_chunks_mut(2 * sh).for_each(|chunk| {
            let (c0, c1) = chunk.split_at_mut(sh);
            c0.par_chunks_mut(BLOCK / 2)
                .zip(c1.par_chunks_mut(BLOCK / 2))
                .for_each(|(x, y)| inner(x, y));
        });
    } else {
        let p = BLOCK / 4;
        amps.par_chunks_mut(2 * sh).for_each(|chunk| {
            let (c0, c1) = chunk.split_at_mut(sh);
            c0.par_chunks_mut(2 * sl)
                .zip(c1.par_chunks_mut(2 * sl))
                .for_each(|(x0, x1)| {
                    let (a00, a01) = x0.split_at_mut(sl);
                    let (a10, a11) = x1.split_at_mut(sl);
                    (
                        a00.par_chunks_mut(p),
                        a01.par_chunks_mut(p),
                        a10.par_chunks_mut(p),
                        a11.par_chunks_mut(p),
                    )
                        .into_par_iter()
                        .for_each(|(w, x, y, z)| f(w, x, y, z));
                });
        });
    }
}

#[inline]
fn scale_slice<T: Real>(xs: &mut [Complex<T>], k: Complex<T>) {
    for x in xs {
        *x = *x * k;
    }
}

impl<T: Real> StateVector<T> {
    /// `|0...0>` on `n` qubits. Panics if the vector would exceed
    /// [`MAX_STATE_BYTES`]; use [`StateVector::try_new`] to handle that.
    pub fn new(n: usize) -> Self {
        Self::try_new(n).unwrap_or_else(|e| panic!("{e}"))
    }

    /// `|0...0>` on `n` qubits, or an error if it would exceed the memory cap.
    pub fn try_new(n: usize) -> Result<Self, SimError> {
        let bytes = state_bytes::<T>(n);
        if n >= 63 || bytes > MAX_STATE_BYTES {
            return Err(SimError::TooLarge {
                what: "state vector",
                bytes,
                limit: MAX_STATE_BYTES,
            });
        }
        let len = 1usize << n;
        let mut amps: Vec<Complex<T>> = if len >= PAR_MIN_LEN {
            // Parallel first touch: the zeroing pass is itself memory bound.
            (0..len).into_par_iter().map(|_| Complex::zero()).collect()
        } else {
            vec![Complex::zero(); len]
        };
        amps[0] = Complex::one();
        Ok(StateVector { n, amps })
    }

    /// Builds a state from explicit amplitudes (length must be a power of 2).
    pub fn from_amplitudes(amps: Vec<Complex<T>>) -> Self {
        assert!(amps.len().is_power_of_two(), "length must be a power of 2");
        let n = amps.len().trailing_zeros() as usize;
        StateVector { n, amps }
    }

    /// The computational basis state `|index>`.
    pub fn basis_state(n: usize, index: usize) -> Self {
        let mut s = Self::new(n);
        s.amps[0] = Complex::zero();
        s.amps[index] = Complex::one();
        s
    }

    pub fn num_qubits(&self) -> usize {
        self.n
    }

    pub fn amplitudes(&self) -> &[Complex<T>] {
        &self.amps
    }

    pub(crate) fn amplitudes_mut(&mut self) -> &mut [Complex<T>] {
        &mut self.amps
    }

    pub fn amplitude(&self, index: usize) -> Complex64 {
        let a = self.amps[index];
        Complex64::new(a.re.to_f64(), a.im.to_f64())
    }

    /// Memory used by the amplitudes, in bytes.
    pub fn bytes(&self) -> usize {
        std::mem::size_of_val(self.amps.as_slice())
    }

    /// Squared norm (should stay 1 up to rounding).
    pub fn norm_sqr(&self) -> f64 {
        if self.amps.len() < PAR_MIN_LEN {
            self.amps.iter().map(|a| a.norm_sqr().to_f64()).sum()
        } else {
            self.amps
                .par_chunks(BLOCK)
                .map(|c| c.iter().map(|a| a.norm_sqr().to_f64()).sum::<f64>())
                .sum()
        }
    }

    /// `<self|other>`.
    pub fn inner(&self, other: &Self) -> Complex64 {
        assert_eq!(self.n, other.n);
        self.amps
            .par_iter()
            .zip(other.amps.par_iter())
            .map(|(a, b)| {
                let p = a.conj() * *b;
                Complex64::new(p.re.to_f64(), p.im.to_f64())
            })
            .sum()
    }

    /// `|<self|other>|^2`.
    pub fn fidelity(&self, other: &Self) -> f64 {
        self.inner(other).norm_sqr()
    }

    /// All `2^n` outcome probabilities (intended for small `n`).
    pub fn probabilities(&self) -> Vec<f64> {
        self.amps.iter().map(|a| a.norm_sqr().to_f64()).collect()
    }

    // ----- gates ---------------------------------------------------------

    /// Applies a gate. Every [`Gate`] is supported.
    pub fn apply_gate(&mut self, g: &Gate) -> Result<(), SimError> {
        check_gate(g, self.n)?;
        match *g {
            Gate::H(q) => self.apply_h(q),
            Gate::X(q) => for_each_pair(&mut self.amps, q, |lo, hi| lo.swap_with_slice(hi)),
            Gate::Y(q) => self.apply_y(q),
            Gate::Cnot(c, t) => self.apply_cnot(c, t),
            Gate::Cz(a, b) => self.apply_cphase_raw(a, b, Complex::new(-T::one(), T::zero())),
            Gate::CPhase(a, b, th) => {
                self.apply_cphase_raw(a, b, cvt(Complex64::from_polar(1.0, th)))
            }
            Gate::Swap(a, b) => self.apply_swap(a, b),
            Gate::Ccx(a, b, t) => {
                let x = Gate::X(0).matrix_1q().expect("1q");
                self.apply_multi_controlled_1q(&[a, b], t, &x)
            }
            ref g1 => {
                let q = g1.qubits()[0];
                if let Some((d0, d1)) = g1.diagonal_1q() {
                    self.apply_diagonal(q, d0, d1);
                } else {
                    let m = g1.matrix_1q().expect("single-qubit gate");
                    self.apply_1q_matrix(q, &m);
                }
            }
        }
        Ok(())
    }

    /// Applies every gate of a circuit; measurements are not allowed here
    /// (use [`Circuit::run`] for circuits with measurements).
    pub fn apply_circuit(&mut self, c: &Circuit) -> Result<(), SimError> {
        for op in &c.ops {
            match op {
                crate::circuit::Op::Gate(g) => self.apply_gate(g)?,
                crate::circuit::Op::Measure(_) => {
                    panic!("apply_circuit: use Circuit::run for measurements")
                }
            }
        }
        Ok(())
    }

    /// Applies an arbitrary 2x2 unitary to qubit `q`.
    pub fn apply_1q_matrix(&mut self, q: usize, m: &Mat2) {
        let (m00, m01, m10, m11) = (
            cvt::<T>(m[0][0]),
            cvt::<T>(m[0][1]),
            cvt::<T>(m[1][0]),
            cvt::<T>(m[1][1]),
        );
        for_each_pair(&mut self.amps, q, |lo, hi| {
            for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
                let (x, y) = (*a, *b);
                *a = m00 * x + m01 * y;
                *b = m10 * x + m11 * y;
            }
        });
    }

    fn apply_h(&mut self, q: usize) {
        let h = T::from_f64(std::f64::consts::FRAC_1_SQRT_2);
        for_each_pair(&mut self.amps, q, |lo, hi| {
            for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
                let (x, y) = (*a, *b);
                *a = (x + y).scale(h);
                *b = (x - y).scale(h);
            }
        });
    }

    fn apply_y(&mut self, q: usize) {
        // Y|0> = i|1>, Y|1> = -i|0>
        for_each_pair(&mut self.amps, q, |lo, hi| {
            for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
                let (x, y) = (*a, *b);
                *a = Complex::new(y.im, -y.re); // -i * y
                *b = Complex::new(-x.im, x.re); //  i * x
            }
        });
    }

    fn apply_diagonal(&mut self, q: usize, d0: Complex64, d1: Complex64) {
        let (d0, d1) = (cvt::<T>(d0), cvt::<T>(d1));
        if d0 == Complex::one() {
            for_each_pair(&mut self.amps, q, |_, hi| scale_slice(hi, d1));
        } else {
            for_each_pair(&mut self.amps, q, |lo, hi| {
                scale_slice(lo, d0);
                scale_slice(hi, d1);
            });
        }
    }

    fn apply_cnot(&mut self, c: usize, t: usize) {
        let (l, h) = (c.min(t), c.max(t));
        if c == h {
            // control is the high bit: swap the l-bit within the h=1 half
            for_each_quad(&mut self.amps, l, h, |_, _, a10, a11| {
                a10.swap_with_slice(a11)
            });
        } else {
            for_each_quad(&mut self.amps, l, h, |_, a01, _, a11| {
                a01.swap_with_slice(a11)
            });
        }
    }

    fn apply_cphase_raw(&mut self, a: usize, b: usize, phase: Complex<T>) {
        let (l, h) = (a.min(b), a.max(b));
        for_each_quad(&mut self.amps, l, h, |_, _, _, a11| scale_slice(a11, phase));
    }

    fn apply_swap(&mut self, a: usize, b: usize) {
        let (l, h) = (a.min(b), a.max(b));
        for_each_quad(&mut self.amps, l, h, |_, a01, a10, _| {
            a01.swap_with_slice(a10)
        });
    }

    /// Applies an arbitrary 4x4 unitary to qubits `(a, b)` (matrix indexed
    /// by `2*bit(a) + bit(b)`).
    pub fn apply_2q_matrix(&mut self, a: usize, b: usize, m: &Mat4) {
        assert_ne!(a, b);
        let (l, h) = (a.min(b), a.max(b));
        // for_each_quad orders slices as 2*bit(h) + bit(l)
        let m = if a == h { *m } else { mat4_swap_qubits(m) };
        let mt: [[Complex<T>; 4]; 4] = m.map(|row| row.map(cvt::<T>));
        for_each_quad(&mut self.amps, l, h, |a00, a01, a10, a11| {
            for i in 0..a00.len() {
                let v = [a00[i], a01[i], a10[i], a11[i]];
                let o = |r: usize| {
                    mt[r][0] * v[0] + mt[r][1] * v[1] + mt[r][2] * v[2] + mt[r][3] * v[3]
                };
                a00[i] = o(0);
                a01[i] = o(1);
                a10[i] = o(2);
                a11[i] = o(3);
            }
        });
    }

    /// Applies `m` to `target` on the subspace where all `controls` are 1.
    pub fn apply_multi_controlled_1q(&mut self, controls: &[usize], target: usize, m: &Mat2) {
        let mask: usize = controls.iter().map(|&c| 1usize << c).sum();
        assert_eq!(mask & (1 << target), 0, "target cannot be a control");
        let (m00, m01, m10, m11) = (
            cvt::<T>(m[0][0]),
            cvt::<T>(m[0][1]),
            cvt::<T>(m[1][0]),
            cvt::<T>(m[1][1]),
        );
        let s = 1usize << target;
        // `base` is the global index of lo[0]; lo[j]/hi[j] form a pair.
        let kernel = |base: usize, lo: &mut [Complex<T>], hi: &mut [Complex<T>]| {
            for j in 0..lo.len() {
                if (base + j) & mask == mask {
                    let (x, y) = (lo[j], hi[j]);
                    lo[j] = m00 * x + m01 * y;
                    hi[j] = m10 * x + m11 * y;
                }
            }
        };
        if 2 * s <= BLOCK {
            let bs = BLOCK.min(self.amps.len());
            self.amps
                .par_chunks_mut(bs)
                .enumerate()
                .for_each(|(k, blk)| {
                    for (i, chunk) in blk.chunks_mut(2 * s).enumerate() {
                        let (lo, hi) = chunk.split_at_mut(s);
                        kernel(k * bs + i * 2 * s, lo, hi);
                    }
                });
        } else {
            self.amps
                .par_chunks_mut(2 * s)
                .enumerate()
                .for_each(|(k, chunk)| {
                    let (lo, hi) = chunk.split_at_mut(s);
                    lo.par_chunks_mut(BLOCK / 2)
                        .zip(hi.par_chunks_mut(BLOCK / 2))
                        .enumerate()
                        .for_each(|(e, (a, b))| kernel(k * 2 * s + e * BLOCK / 2, a, b));
                });
        }
    }

    /// Multi-controlled Z: flips the sign of every amplitude whose index has
    /// all of `qubits` set. Used as the Grover oracle/diffuser core.
    pub fn apply_mcz(&mut self, qubits: &[usize]) {
        let mask: usize = qubits.iter().map(|&c| 1usize << c).sum();
        let bs = BLOCK.min(self.amps.len());
        self.amps
            .par_chunks_mut(bs)
            .enumerate()
            .for_each(|(k, blk)| {
                let base = k * bs;
                for (j, a) in blk.iter_mut().enumerate() {
                    if (base + j) & mask == mask {
                        *a = -*a;
                    }
                }
            });
    }

    /// Applies a classical reversible function `|i> -> |f(i)>`. `f` must be
    /// a bijection on `0..2^n`. Used for oracles such as modular
    /// multiplication in Shor's algorithm; costs one extra state vector.
    pub fn apply_permutation(&mut self, f: impl Fn(usize) -> usize) {
        let mut out = vec![Complex::zero(); self.amps.len()];
        let mut seen = vec![false; self.amps.len()];
        for (i, a) in self.amps.iter().enumerate() {
            let j = f(i);
            assert!(!seen[j], "apply_permutation: f is not a bijection");
            seen[j] = true;
            out[j] = *a;
        }
        self.amps = out;
    }

    // ----- measurement ---------------------------------------------------

    /// Probability that measuring qubit `q` gives 1.
    pub fn prob_one(&self, q: usize) -> f64 {
        let s = 1usize << q;
        let hi_sum = |chunk: &[Complex<T>]| -> f64 {
            chunk
                .chunks(2 * s)
                .map(|c| c[s..].iter().map(|a| a.norm_sqr().to_f64()).sum::<f64>())
                .sum()
        };
        if self.amps.len() < PAR_MIN_LEN {
            hi_sum(&self.amps)
        } else if 2 * s <= BLOCK {
            self.amps.par_chunks(BLOCK).map(hi_sum).sum()
        } else {
            // Every block lies entirely inside a run with bit q fixed.
            self.amps
                .par_chunks(BLOCK)
                .enumerate()
                .filter(|(k, _)| (k * BLOCK) & s != 0)
                .map(|(_, c)| c.iter().map(|a| a.norm_sqr().to_f64()).sum::<f64>())
                .sum()
        }
    }

    /// `<Z_q>`.
    pub fn expectation_z(&self, q: usize) -> f64 {
        1.0 - 2.0 * self.prob_one(q)
    }

    /// Projects qubit `q` onto `outcome` and renormalises. Returns the
    /// probability of that outcome before projection.
    pub fn collapse(&mut self, q: usize, outcome: bool) -> f64 {
        let p1 = self.prob_one(q);
        let p = if outcome { p1 } else { 1.0 - p1 };
        assert!(p > 0.0, "cannot collapse onto a zero-probability outcome");
        let k = Complex::new(T::from_f64(1.0 / p.sqrt()), T::zero());
        let z = Complex::zero();
        for_each_pair(&mut self.amps, q, |lo, hi| {
            let (keep, kill) = if outcome { (hi, lo) } else { (lo, hi) };
            kill.fill(z);
            scale_slice(keep, k);
        });
        p
    }

    /// Measures qubit `q`, collapsing the state.
    pub fn measure_qubit<R: Rng + ?Sized>(&mut self, q: usize, rng: &mut R) -> bool {
        let p1 = self.prob_one(q);
        let outcome = rng.random::<f64>() < p1;
        self.collapse(q, outcome);
        outcome
    }

    /// Draws `shots` samples of all qubits without collapsing the state.
    /// Each sample is a basis index (bit `q` = outcome of qubit `q`).
    ///
    /// Cost: one parallel pass to get block weights, then a single merged
    /// walk over sorted random numbers, so `O(2^n + shots log shots)`.
    pub fn sample<R: Rng + ?Sized>(&self, shots: usize, rng: &mut R) -> Vec<usize> {
        let len = self.amps.len();
        let bs = BLOCK.min(len);
        let sums: Vec<f64> = self
            .amps
            .par_chunks(bs)
            .map(|c| c.iter().map(|a| a.norm_sqr().to_f64()).sum())
            .collect();
        let total: f64 = sums.iter().sum();
        let mut rs: Vec<f64> = (0..shots).map(|_| rng.random::<f64>() * total).collect();
        rs.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        let mut out = Vec::with_capacity(shots);
        let (mut ci, mut acc_c) = (0usize, 0.0f64);
        let (mut j, mut acc_j) = (0usize, 0.0f64);
        for r in rs {
            while ci + 1 < sums.len() && acc_c + sums[ci] <= r {
                acc_c += sums[ci];
                ci += 1;
                j = 0;
                acc_j = 0.0;
            }
            let chunk = &self.amps[ci * bs..(ci + 1) * bs];
            while j + 1 < chunk.len() && acc_c + acc_j + chunk[j].norm_sqr().to_f64() <= r {
                acc_j += chunk[j].norm_sqr().to_f64();
                j += 1;
            }
            out.push(ci * bs + j);
        }
        out.shuffle(rng);
        out
    }
}

impl<T: Real> Simulator for StateVector<T> {
    fn name(&self) -> &'static str {
        "state-vector"
    }
    fn num_qubits(&self) -> usize {
        self.n
    }
    fn apply(&mut self, gate: &Gate) -> Result<(), SimError> {
        self.apply_gate(gate)
    }
    fn measure(&mut self, q: usize, rng: &mut dyn RngCore) -> Result<bool, SimError> {
        if q >= self.n {
            return Err(SimError::QubitOutOfRange {
                qubit: q,
                num_qubits: self.n,
            });
        }
        Ok(self.measure_qubit(q, rng))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn size_cap() {
        // (no 512 MiB allocation in unit tests; just check the arithmetic)
        assert_eq!(state_bytes::<f32>(26), 512 << 20);
        assert_eq!(StateVector::<f32>::try_new(10).unwrap().bytes(), 8 << 10);
        assert!(StateVector::<f32>::try_new(28).is_err());
        assert!(StateVector::<f64>::try_new(27).is_err());
        assert_eq!(state_bytes::<f64>(25), 512 << 20);
    }

    #[test]
    fn bell_state() {
        let mut s = StateVectorF64::new(2);
        s.apply_gate(&Gate::H(0)).unwrap();
        s.apply_gate(&Gate::Cnot(0, 1)).unwrap();
        let h = std::f64::consts::FRAC_1_SQRT_2;
        assert!((s.amplitude(0).re - h).abs() < 1e-12);
        assert!((s.amplitude(3).re - h).abs() < 1e-12);
        assert!(s.amplitude(1).norm() < 1e-12 && s.amplitude(2).norm() < 1e-12);
    }

    #[test]
    fn sampling_follows_distribution() {
        let mut s = StateVectorF64::new(3);
        s.apply_gate(&Gate::Ry(0, 1.0)).unwrap();
        s.apply_gate(&Gate::H(2)).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        let shots = 40_000;
        let samples = s.sample(shots, &mut rng);
        let probs = s.probabilities();
        let mut counts = [0usize; 8];
        for x in samples {
            counts[x] += 1;
        }
        for i in 0..8 {
            let f = counts[i] as f64 / shots as f64;
            assert!((f - probs[i]).abs() < 0.01, "{i}: {f} vs {}", probs[i]);
        }
    }

    #[test]
    fn large_strides_match_small_strides() {
        // 16 qubits exercises the parallel paths (including the 4-slice
        // split for high qubit pairs); compare against a run with the
        // qubits relabelled so everything happens at low strides.
        let n = 16;
        let mut rng = StdRng::seed_from_u64(3);
        let mut a = StateVectorF64::new(n);
        let mut b = StateVectorF64::new(n);
        let rev = |q: usize| n - 1 - q;
        for q in 0..n {
            a.apply_gate(&Gate::Ry(q, 0.1 * q as f64 + 0.3)).unwrap();
            b.apply_gate(&Gate::Ry(rev(q), 0.1 * q as f64 + 0.3))
                .unwrap();
        }
        for _ in 0..40 {
            let x = rng.random_range(0..n);
            let mut y = rng.random_range(0..n);
            while y == x {
                y = rng.random_range(0..n);
            }
            let th = rng.random::<f64>();
            for g in [
                Gate::Cnot(x, y),
                Gate::CPhase(x, y, th),
                Gate::Swap(x, y),
                Gate::Rx(x, th),
                Gate::T(y),
                Gate::Y(x),
                Gate::Ccx(x, y, (x.max(y) + 1) % n),
            ] {
                if check_gate(&g, n).is_err() {
                    continue;
                }
                a.apply_gate(&g).unwrap();
                let gb = match g {
                    Gate::Cnot(p, q) => Gate::Cnot(rev(p), rev(q)),
                    Gate::CPhase(p, q, t) => Gate::CPhase(rev(p), rev(q), t),
                    Gate::Swap(p, q) => Gate::Swap(rev(p), rev(q)),
                    Gate::Rx(p, t) => Gate::Rx(rev(p), t),
                    Gate::T(p) => Gate::T(rev(p)),
                    Gate::Y(p) => Gate::Y(rev(p)),
                    Gate::Ccx(p, q, r) => Gate::Ccx(rev(p), rev(q), rev(r)),
                    _ => unreachable!(),
                };
                b.apply_gate(&gb).unwrap();
            }
        }
        let rev_index = |i: usize| (0..n).fold(0, |acc, q| acc | (((i >> q) & 1) << rev(q)));
        for i in 0..(1 << n) {
            assert!((a.amplitude(i) - b.amplitude(rev_index(i))).norm() < 1e-9);
        }
        assert!((a.norm_sqr() - 1.0).abs() < 1e-9);
        for q in [0, 7, 15] {
            assert!((a.prob_one(q) - b.prob_one(rev(q))).abs() < 1e-9);
        }
    }
}
