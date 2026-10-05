//! Pauli-frame detector sampler for short runs (notebook:
//! `research/qec/sampler-x.md` §2.3).
//!
//! [`super::fast_sampler::FastSampler`] wins per shot by a wide margin, but
//! it needs a compile (a backward sweep over the whole circuit, see
//! [`super::detector_compiler`]) whose sparse set merges cost more per gate
//! than simply pushing a 64-shot Pauli frame through the gate. For a run of
//! a few hundred or thousand shots the frame simulation is the cheaper
//! exact method, so `sample-x` uses it there.
//!
//! The method is the standard one (Gidney, Stim, arXiv:2103.02202):
//! * a frame holds, per qubit, the X and Z components of the Pauli error
//!   relative to a noiseless reference run, for `64 W` shots at once (bit
//!   `s` of word `w` = shot `64 w + s`);
//! * Clifford gates conjugate the frame word-parallel (`CX c t`: `x[t] ^=
//!   x[c]`, `z[c] ^= z[t]`, ...);
//! * a Z-basis measurement records `x[q]` (the outcome flip relative to the
//!   reference) XOR its readout flip, then randomises `z[q]` (a coin: the
//!   qubit is in a Z eigenstate, so a random Z is free and re-randomises
//!   later outcomes the reference fixed arbitrarily); a reset clears `x[q]`
//!   and randomises `z[q]`; the initial frame has random `z`;
//! * a detector is the XOR of the recorded flips it reads, an observable
//!   likewise (detection events relative to the reference, Stim's
//!   convention).
//!
//! Noise uses FastSampler's exact hit model: a channel with `m`
//! non-identity Paulis and probability `p < m / (m + 1)` receives
//! `Poisson(lambda)` hits per shot (`lambda` = [`hit_rate`]`(p, m)`), each a
//! uniform non-identity Pauli, and hits on one shot XOR. Per target the hit
//! count over all `64 W` shots is one lookup in a precomputed inverse-CDF
//! table and each hit one random word (shot from its top bits, the batch
//! being a power of two), so no logarithm is evaluated per fault. Channels
//! at or beyond full mixing (`p >= m / (m + 1)`) are sampled literally:
//! Bernoulli(`p`) cells found by geometric skipping, carried across
//! targets. Both are exactly the channel, in the convention of
//! [`super::symphase::VarDist`].
//!
//! The program is walked straight from the parsed [`StimCircuit`]
//! (`REPEAT` blocks unrolled on the fly), so there is no compile at all; the
//! measurement record is a ring of `max_lookback` slots (rounded up to a
//! power of two), so it stays in cache however long the circuit is.
//! [`FrameSampler::set_simd`] runs the same code compiled with AVX-512
//! enabled (the word loops are vectorised by the compiler); same random
//! stream, so the output is bit-identical.

use super::fast_sampler::{batch_rng, hit_rate, uniform_below, PoissonTable, WyRand};
use crate::gate::Gate;
use crate::io::stim::{StimCircuit, StimOp};
use rand::RngCore;
use std::io::Write;

/// The faults of one channel kind (`m` non-identity Paulis) at one
/// probability `p`, for targets of `cells` cells (shots) each.
#[derive(Clone, Debug)]
struct Stream {
    p_bits: u64,
    m: u64,
    /// Hit model (`p < m / (m + 1)`): the number of hits on a target is
    /// `Poisson(lambda * cells)` (this table), each at a uniform shot with a
    /// uniform non-identity Pauli; hits on one cell XOR (exact, as in
    /// [`super::fast_sampler`]'s module docs).
    pois: Option<PoissonTable>,
    /// Otherwise Bernoulli(`p`) cells by geometric skipping: `ln(1 - p)`
    /// (`-inf` for `p = 1`) and the cells left before the next fault.
    ln_q: f64,
    gap: u64,
}

/// Reusable buffers of [`FrameSampler::sample_batch`].
#[derive(Clone, Debug, Default)]
pub struct FrameState {
    x: Vec<u64>,
    z: Vec<u64>,
    rec: Vec<u64>,
    streams: Vec<Stream>,
    last: usize,
}

/// Frame-simulation detector sampler over a parsed `.stim` program.
#[derive(Clone, Debug)]
pub struct FrameSampler<'a> {
    prog: &'a StimCircuit,
    simd: bool,
}

/// Cells to skip before the next fault of a Bernoulli stream.
#[inline(always)]
fn geometric(ln_q: f64, rng: &mut WyRand) -> u64 {
    // U in (0, 1]
    let u = 1.0 - (rng.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0);
    let k = (u.ln() / ln_q).floor();
    if k >= 1.8e19 || k.is_nan() {
        u64::MAX / 4
    } else {
        k as u64
    }
}

impl FrameState {
    /// The stream of a channel with `m` Paulis at probability `p` (created
    /// on first use in a batch).
    #[inline(always)]
    fn stream(&mut self, p: f64, m: u64, cells: u64, rng: &mut WyRand) -> usize {
        let pb = p.to_bits();
        let hit = |s: &Stream| s.p_bits == pb && s.m == m;
        if self.last < self.streams.len() && hit(&self.streams[self.last]) {
            return self.last;
        }
        let i = match self.streams.iter().position(hit) {
            Some(i) => i,
            None => {
                let mf = m as f64;
                let pois = (p < mf / (mf + 1.0) * (1.0 - 1e-9))
                    .then(|| PoissonTable::new(hit_rate(p, m) * cells as f64));
                let ln_q = (-p).ln_1p();
                let gap = if pois.is_some() || p >= 1.0 {
                    0
                } else {
                    geometric(ln_q, rng)
                };
                self.streams.push(Stream {
                    p_bits: pb,
                    m,
                    pois,
                    ln_q,
                    gap,
                });
                self.streams.len() - 1
            }
        };
        self.last = i;
        i
    }

    /// The faults of one target (`cells` shots, a power of two) in stream
    /// `i`: calls `hit(shot, pauli)` with `pauli` in `1..=m` for every hit
    /// (hit model) or faulty cell (Bernoulli model).
    #[inline(always)]
    fn faults(&mut self, i: usize, cells: u64, rng: &mut WyRand, mut hit: impl FnMut(u64, u64)) {
        let st = &mut self.streams[i];
        let m = st.m;
        let pauli = |rng: &mut WyRand| if m == 1 { 1 } else { 1 + uniform_below(rng, m) };
        if let Some(t) = &st.pois {
            let shift = 64 - cells.trailing_zeros();
            for _ in 0..t.sample(rng.next_u64()) {
                let s = rng.next_u64() >> shift;
                hit(s, pauli(rng));
            }
            return;
        }
        if st.gap >= cells {
            st.gap -= cells;
            return;
        }
        let mut s = st.gap;
        let ln_q = st.ln_q;
        loop {
            hit(s, pauli(rng));
            let g = if ln_q == f64::NEG_INFINITY {
                0
            } else {
                geometric(ln_q, rng)
            };
            s = s.saturating_add(1).saturating_add(g);
            if s >= cells {
                self.streams[i].gap = s - cells;
                return;
            }
        }
    }
}

/// `v[dst line] ^= v[src line]` for `w`-word lines `dst != src` (split
/// borrows, so the loop vectorises).
#[inline(always)]
fn xor_line(v: &mut [u64], dst: usize, src: usize, w: usize) {
    let (d, s) = if dst < src {
        let (a, b) = v.split_at_mut(src * w);
        (&mut a[dst * w..dst * w + w], &b[..w])
    } else {
        let (a, b) = v.split_at_mut(dst * w);
        (&mut b[..w], &a[src * w..src * w + w])
    };
    for (x, y) in d.iter_mut().zip(s) {
        *x ^= y;
    }
}

/// `a[line q] ^= b[line r]` for two different vectors.
#[inline(always)]
fn xor_cross(a: &mut [u64], q: usize, b: &[u64], r: usize, w: usize) {
    for (x, y) in a[q * w..q * w + w].iter_mut().zip(&b[r * w..r * w + w]) {
        *x ^= y;
    }
}

/// Pauli index (1 = X, 2 = Y, 3 = Z) to `(x, z)` flips.
#[inline(always)]
fn pauli_xz(k: u64) -> (bool, bool) {
    (k == 1 || k == 2, k == 2 || k == 3)
}

impl<'a> FrameSampler<'a> {
    /// A sampler for the detectors and observables of `prog`.
    pub fn new(prog: &'a StimCircuit) -> FrameSampler<'a> {
        FrameSampler { prog, simd: false }
    }

    /// Output rows per shot: detectors, then observables.
    pub fn rows(&self) -> usize {
        self.prog.num_detectors() + self.prog.num_observables()
    }

    /// Runs the word loops compiled for AVX-512 when the CPU has it
    /// (returns whether it will). Bit-identical output.
    pub fn set_simd(&mut self, on: bool) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            self.simd = on
                && std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vl");
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = on;
            self.simd = false;
        }
        self.simd
    }

    /// Samples `64 * words` shots (`words` a power of two): `out` (at least
    /// `words * rows()` words) gets `out[w * rows() + r]` bit `s` = row `r`
    /// of shot `64 w + s`, i.e. Stim's ptb64 layout block by block.
    pub fn sample_batch(&self, words: usize, rng: &mut WyRand, st: &mut FrameState, out: &mut [u64]) {
        #[cfg(target_arch = "x86_64")]
        if self.simd {
            // SAFETY: `simd` is only set when the CPU has these features
            unsafe { self.batch_avx512(words, rng, st, out) };
            return;
        }
        self.batch_impl(words, rng, st, out);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,avx512f,avx512bw,avx512vl")]
    unsafe fn batch_avx512(&self, words: usize, rng: &mut WyRand, st: &mut FrameState, out: &mut [u64]) {
        self.batch_impl(words, rng, st, out);
    }

    #[inline(always)]
    fn batch_impl(&self, w: usize, rng: &mut WyRand, st: &mut FrameState, out: &mut [u64]) {
        let n = self.prog.num_qubits();
        let rows = self.rows();
        let nd = self.prog.num_detectors();
        let cells = 64 * w as u64;
        assert!(w.is_power_of_two(), "words per batch must be a power of two");
        assert!(out.len() >= w * rows);
        out[..w * rows].fill(0);
        st.x.clear();
        st.x.resize(n * w, 0);
        st.z.clear();
        st.z.extend((0..n * w).map(|_| rng.next_u64()));
        // the record is a ring of `ring` measurement slots: nothing reads
        // further back than max_lookback
        let ring = self.prog.max_lookback().max(1).next_power_of_two();
        let rmask = ring - 1;
        st.rec.clear();
        st.rec.resize(ring * w, 0);
        st.streams.clear();
        // split borrows: the streams move into `st2`, the frame words stay
        let mut st2 = FrameState {
            streams: std::mem::take(&mut st.streams),
            ..Default::default()
        };
        let FrameState { x, z, rec, .. } = &mut *st;
        let line = |q: usize| q * w..(q + 1) * w;
        self.prog.for_each_op(|op| match op {
            StimOp::Gate(g) => match g {
                Gate::H(q) => {
                    let (xs, zs) = (&mut x[line(q)], &mut z[line(q)]);
                    xs.swap_with_slice(zs);
                }
                Gate::S(q) | Gate::Sdg(q) => xor_cross(z, q, x, q, w),
                Gate::Cnot(c, t) => {
                    xor_line(x, t, c, w);
                    xor_line(z, c, t, w);
                }
                Gate::Cz(a, b) => {
                    xor_cross(z, a, x, b, w);
                    xor_cross(z, b, x, a, w);
                }
                Gate::Swap(a, b) => {
                    let (lo, hi) = (a.min(b), a.max(b));
                    for v in [&mut *x, &mut *z] {
                        let (p, q) = v.split_at_mut(hi * w);
                        p[lo * w..lo * w + w].swap_with_slice(&mut q[..w]);
                    }
                }
                // Paulis commute with the frame up to sign
                _ => {}
            },
            StimOp::Measure { qubit: q, index, p } => {
                let r = (index & rmask) * w;
                rec[r..r + w].copy_from_slice(&x[line(q)]);
                if p > 0.0 {
                    let i = st2.stream(p, 1, cells, rng);
                    st2.faults(i, cells, rng, |s, _| {
                        rec[r + (s >> 6) as usize] ^= 1 << (s & 63);
                    });
                }
                for zi in &mut z[line(q)] {
                    *zi = rng.next_u64();
                }
            }
            StimOp::Reset(q) => {
                x[line(q)].fill(0);
                for zi in &mut z[line(q)] {
                    *zi = rng.next_u64();
                }
            }
            StimOp::XFlip(q, p) | StimOp::YFlip(q, p) | StimOp::ZFlip(q, p) if p > 0.0 => {
                let (fx, fz) = match op {
                    StimOp::XFlip(..) => (true, false),
                    StimOp::YFlip(..) => (true, true),
                    _ => (false, true),
                };
                let i = st2.stream(p, 1, cells, rng);
                st2.faults(i, cells, rng, |s, _| {
                    let (k, b) = (q * w + (s >> 6) as usize, 1u64 << (s & 63));
                    if fx {
                        x[k] ^= b;
                    }
                    if fz {
                        z[k] ^= b;
                    }
                });
            }
            StimOp::Depolarize1(q, p) if p > 0.0 => {
                let i = st2.stream(p, 3, cells, rng);
                st2.faults(i, cells, rng, |s, pk| {
                    let (fx, fz) = pauli_xz(pk);
                    let (k, b) = (q * w + (s >> 6) as usize, 1u64 << (s & 63));
                    x[k] ^= if fx { b } else { 0 };
                    z[k] ^= if fz { b } else { 0 };
                });
            }
            StimOp::Depolarize2(qa, qb, p) if p > 0.0 => {
                let i = st2.stream(p, 15, cells, rng);
                st2.faults(i, cells, rng, |s, pk| {
                    let (ax, az) = pauli_xz(pk / 4);
                    let (bx, bz) = pauli_xz(pk % 4);
                    let (o, b) = ((s >> 6) as usize, 1u64 << (s & 63));
                    x[qa * w + o] ^= if ax { b } else { 0 };
                    z[qa * w + o] ^= if az { b } else { 0 };
                    x[qb * w + o] ^= if bx { b } else { 0 };
                    z[qb * w + o] ^= if bz { b } else { 0 };
                });
            }
            StimOp::Detector {
                row,
                base,
                lookbacks,
            } => {
                for i in 0..w {
                    let mut acc = 0u64;
                    for &k in lookbacks {
                        acc ^= rec[((base - k as usize) & rmask) * w + i];
                    }
                    out[i * rows + row] = acc;
                }
            }
            StimOp::Observable {
                index,
                base,
                lookbacks,
            } => {
                for i in 0..w {
                    let mut acc = 0u64;
                    for &k in lookbacks {
                        acc ^= rec[((base - k as usize) & rmask) * w + i];
                    }
                    out[i * rows + nd + index] ^= acc;
                }
            }
            _ => {}
        });
        st.streams = st2.streams;
    }

    /// Samples `shots` shots (rounded up to a multiple of 64) in batches of
    /// `64 * words` shots (`words` a power of two; the output depends on it)
    /// and writes ptb64 to `w`. Batch `b` uses [`batch_rng`]`(seed, b)`.
    pub fn write_ptb64<W: Write + ?Sized>(
        &self,
        shots: usize,
        seed: u64,
        words: usize,
        w: &mut W,
    ) -> std::io::Result<()> {
        let groups = shots.div_ceil(64);
        let words = words.max(1);
        let batches = groups.div_ceil(words);
        let rows = self.rows();
        let mut st = FrameState::default();
        let mut out = vec![0u64; words * rows];
        for b in 0..batches {
            let mut rng = batch_rng(seed, b as u64);
            self.sample_batch(words, &mut rng, &mut st, &mut out);
            let take = words.min(groups - b * words);
            let bytes: &[u64] = &out[..take * rows];
            if cfg!(target_endian = "little") {
                // SAFETY: plain u64 words; on little-endian targets their bytes
                // are the ptb64 encoding
                w.write_all(unsafe {
                    std::slice::from_raw_parts(bytes.as_ptr() as *const u8, bytes.len() * 8)
                })?;
            } else {
                for v in bytes {
                    w.write_all(&v.to_le_bytes())?;
                }
            }
        }
        Ok(())
    }
}
