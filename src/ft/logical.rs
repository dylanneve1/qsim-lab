//! Logical-level drivers: one interface ([`Logical`]) for an algorithm written
//! in Clifford+T with measurements, three implementations:
//!
//! * [`Encoded`] on the frame backend: every logical qubit is a level-K
//!   concatenated Steane block; the *ideal* logical state lives in a small
//!   dense vector ([`LSv`]) while all physical noise lives in the Pauli frame.
//!   A logical measurement samples the ideal outcome from the vector and XORs
//!   the decoded frame flip; feed-forward (the S correction of a T gadget,
//!   semiclassical phase corrections) uses the recorded outcome, so logical
//!   errors act on the vector exactly as they would physically (e.g. an X_L
//!   error before a T gadget becomes the wrong S correction).
//! * [`Encoded`] on the dense backend (validation): no logical vector, the
//!   physical state vector holds everything and measurements are real.
//! * [`Unencoded`]: the same algorithm on bare qubits with the same
//!   circuit-level noise (T gates physical).

use super::backends::{DenseBackend, FrameBackend};
use super::core::*;
use super::machine::{ideal_logical, FtConfig, Machine, Phys};
use num_complex::Complex64;

/// Small dense logical state (≤ ~16 qubits).
#[derive(Clone, Debug)]
pub struct LSv {
    /// Number of logical qubits.
    pub n: usize,
    /// `2^n` amplitudes; basis index bit `q` is qubit `q` (qubit 0 = least
    /// significant bit).
    pub a: Vec<Complex64>,
}

impl LSv {
    /// `n`-qubit register in |0…0⟩.
    pub fn new(n: usize) -> Self {
        let mut a = vec![Complex64::new(0.0, 0.0); 1 << n];
        a[0] = Complex64::new(1.0, 0.0);
        LSv { n, a }
    }
    /// Pauli X on qubit `q`.
    pub fn x(&mut self, q: usize) {
        let m = 1 << q;
        for i in 0..self.a.len() {
            if i & m == 0 {
                self.a.swap(i, i | m);
            }
        }
    }
    /// Pauli Z on qubit `q`.
    pub fn z(&mut self, q: usize) {
        self.phase(q, Complex64::new(-1.0, 0.0));
    }
    /// Multiplies every amplitude with qubit `q` = 1 by `ph` (diag(1, ph)).
    pub fn phase(&mut self, q: usize, ph: Complex64) {
        let m = 1 << q;
        for (i, v) in self.a.iter_mut().enumerate() {
            if i & m != 0 {
                *v *= ph;
            }
        }
    }
    /// S = diag(1, i) on qubit `q`.
    pub fn s(&mut self, q: usize) {
        self.phase(q, Complex64::new(0.0, 1.0));
    }
    /// S† = diag(1, −i) on qubit `q`.
    pub fn sdg(&mut self, q: usize) {
        self.phase(q, Complex64::new(0.0, -1.0));
    }
    /// T = diag(1, e^{iπ/4}) on qubit `q`.
    pub fn t(&mut self, q: usize) {
        self.phase(q, Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4));
    }
    /// T† = diag(1, e^{−iπ/4}) on qubit `q`.
    pub fn tdg(&mut self, q: usize) {
        self.phase(q, Complex64::from_polar(1.0, -std::f64::consts::FRAC_PI_4));
    }
    /// Hadamard on qubit `q`.
    pub fn h(&mut self, q: usize) {
        let m = 1 << q;
        let r = std::f64::consts::FRAC_1_SQRT_2;
        for i in 0..self.a.len() {
            if i & m == 0 {
                let (u, v) = (self.a[i], self.a[i | m]);
                self.a[i] = (u + v) * r;
                self.a[i | m] = (u - v) * r;
            }
        }
    }
    /// CNOT with control `c` and target `t`.
    pub fn cnot(&mut self, c: usize, t: usize) {
        let (mc, mt) = (1 << c, 1 << t);
        for i in 0..self.a.len() {
            if i & mc != 0 && i & mt == 0 {
                self.a.swap(i, i | mt);
            }
        }
    }
    /// Toffoli: flips `t` when both `a` and `b` are 1.
    pub fn ccx(&mut self, a: usize, b: usize, t: usize) {
        let (ma, mb, mt) = (1 << a, 1 << b, 1 << t);
        for i in 0..self.a.len() {
            if i & ma != 0 && i & mb != 0 && i & mt == 0 {
                self.a.swap(i, i | mt);
            }
        }
    }
    /// Applies a Pauli code (bit 0 = X, bit 1 = Z, 3 = Y up to phase) to qubit
    /// `q`; Z is applied before X.
    pub fn pauli(&mut self, q: usize, code: u8) {
        if code & PZ != 0 {
            self.z(q);
        }
        if code & PX != 0 {
            self.x(q);
        }
    }
    /// Probability that qubit `q` measures 1 (assumes a normalised state).
    pub fn prob1(&self, q: usize) -> f64 {
        let m = 1 << q;
        self.a
            .iter()
            .enumerate()
            .filter(|(i, _)| i & m != 0)
            .map(|(_, v)| v.norm_sqr())
            .sum()
    }
    /// Born-rule measurement of qubit q with collapse.
    pub fn measure(&mut self, q: usize, rng: &mut Xoshiro) -> bool {
        let p1 = self.prob1(q);
        let r = rng.f64() < p1;
        let m = 1 << q;
        let norm = if r { p1 } else { 1.0 - p1 }.sqrt();
        for (i, v) in self.a.iter_mut().enumerate() {
            if ((i & m) != 0) == r {
                *v /= norm;
            } else {
                *v = Complex64::new(0.0, 0.0);
            }
        }
        r
    }
    /// Reset qubit q to |b⟩ (measure, then flip if needed).
    pub fn reset(&mut self, q: usize, b: bool, rng: &mut Xoshiro) {
        if self.measure(q, rng) != b {
            self.x(q);
        }
    }
}

/// A Clifford+T logical machine with measurements.
pub trait Logical {
    /// Prepares logical qubit `q` in |0⟩ (`bit = false`) or |1⟩ (`bit = true`).
    fn prep(&mut self, q: usize, bit: bool);
    /// Logical Hadamard.
    fn h(&mut self, q: usize);
    /// Logical S.
    fn s(&mut self, q: usize);
    /// Logical S†.
    fn sdg(&mut self, q: usize);
    /// Logical T.
    fn t(&mut self, q: usize);
    /// Logical T†.
    fn tdg(&mut self, q: usize);
    /// Logical CNOT, control `c`, target `t`.
    fn cnot(&mut self, c: usize, t: usize);
    /// Measures logical qubit `q` in the Z basis and returns the recorded outcome.
    fn meas(&mut self, q: usize) -> bool;
    /// Toffoli; default: the standard 7-T decomposition (Nielsen & Chuang
    /// Fig. 4.9: 6 CNOT, 7 T/T†, 2 H).
    fn ccx(&mut self, c1: usize, c2: usize, t: usize) {
        self.h(t);
        self.cnot(c2, t);
        self.tdg(t);
        self.cnot(c1, t);
        self.t(t);
        self.cnot(c2, t);
        self.tdg(t);
        self.cnot(c1, t);
        self.t(c2);
        self.t(t);
        self.h(t);
        self.cnot(c1, c2);
        self.t(c1);
        self.tdg(c2);
        self.cnot(c1, c2);
    }
    /// Classically controlled S† in a fixed slot: when `apply` is false the
    /// slot is a noisy identity, so the location structure of a run does not
    /// depend on measurement outcomes (needed for the clean-run estimator).
    fn sdg_slot(&mut self, q: usize, apply: bool) {
        if apply {
            self.sdg(q);
        }
    }
    /// Controlled swap.
    fn cswap(&mut self, c: usize, a: usize, b: usize) {
        self.cnot(b, a);
        self.ccx(c, a, b);
        self.cnot(b, a);
    }
}

/// How magic states are supplied to the encoded T gadget.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MagicMode {
    /// Non-FT injection simulated at the circuit level (+ optional
    /// post-selection), no distillation.
    Raw,
    /// A noiselessly encoded |T⟩_L with a logical Z error of probability ε
    /// (twirled model of a distilled state, e.g. ε = 35 ε_in³ for 15-to-1).
    Model(f64),
}

/// Statistics of the encoded run.
#[derive(Clone, Debug, Default)]
pub struct RunCounts {
    /// Number of T/T† gadgets executed (one magic state consumed each).
    pub t_gadgets: u64,
    /// Number of logical gates applied (H, S, S†, T, T†, CNOT and empty
    /// S†-slots; preparations and measurements are not counted).
    pub logical_gates: u64,
    /// Frame mode: set as soon as any logical-level error is present (a
    /// non-zero decoded flip of a top-level measurement, or a data block whose
    /// residual frame decodes to a logical Pauli after a top-level operation).
    /// Conservative: some of these errors are harmless for the output.
    pub logical_fault: bool,
}

/// Encoded logical machine at level `k` (1 = Steane [[7,1,3]], 2 = [[49,1,9]]).
pub struct Encoded<B: Phys> {
    /// Concatenated-Steane machine that owns all physical qubits.
    pub m: Machine<B>,
    /// Concatenation level (1 = [[7,1,3]], 2 = [[49,1,9]]).
    pub k: usize,
    /// Index of the first physical qubit of each logical data block (one level-`k`
    /// block of `7^k` qubits per logical qubit).
    pub blocks: Vec<usize>,
    /// Ideal logical state (frame backend only); the last qubit is the magic
    /// slot.
    pub sv: Option<LSv>,
    /// RNG for sampling the ideal logical measurement outcomes and the
    /// `MagicMode::Model` logical Z errors.
    pub mrng: Xoshiro,
    /// How magic states for T gadgets are produced.
    pub magic: MagicMode,
    /// Run statistics.
    pub counts: RunCounts,
    /// Logical qubits currently holding state (prepared, not yet measured).
    pub live: Vec<bool>,
}

impl Encoded<FrameBackend> {
    /// Frame-backend machine with `nlog` logical qubits at level `k`; the ideal
    /// logical vector has `nlog + 1` qubits (the extra one is the magic slot).
    /// `seed` seeds the logical-measurement RNG; physical faults come from `noise`.
    pub fn frame(
        k: usize,
        nlog: usize,
        noise: Noise,
        cfg: FtConfig,
        magic: MagicMode,
        seed: u64,
    ) -> Self {
        let mut m = Machine::new(FrameBackend::default(), noise, cfg, k);
        let blocks = (0..nlog).map(|_| m.alloc(k)).collect();
        Encoded {
            m,
            k,
            blocks,
            sv: Some(LSv::new(nlog + 1)),
            mrng: Xoshiro::new(seed ^ 0x5EED_CAFE),
            magic,
            counts: RunCounts::default(),
            live: vec![false; nlog],
        }
    }
}

impl Encoded<DenseBackend> {
    /// Dense-backend machine (validation) with `nlog` logical qubits at level `k`
    /// on a `cap`-qubit physical state vector (`cap` must cover all data and
    /// ancilla blocks, otherwise it panics). `seed` seeds both the backend's
    /// measurement RNG and the magic-model RNG.
    pub fn dense(
        k: usize,
        nlog: usize,
        cap: usize,
        noise: Noise,
        cfg: FtConfig,
        magic: MagicMode,
        seed: u64,
    ) -> Self {
        let mut m = Machine::new(DenseBackend::new(cap, seed), noise, cfg, k);
        let blocks = (0..nlog).map(|_| m.alloc(k)).collect();
        Encoded {
            m,
            k,
            blocks,
            sv: None,
            mrng: Xoshiro::new(seed ^ 0x5EED_CAFE),
            magic,
            counts: RunCounts::default(),
            live: vec![false; nlog],
        }
    }
}

impl Encoded<FrameBackend> {
    /// Update `counts.logical_fault` from the frames of the live data blocks
    /// (a measured block is dead until it is prepared again).
    pub fn check_frames(&mut self) {
        if self.counts.logical_fault {
            return;
        }
        for (q, &b) in self.blocks.iter().enumerate() {
            if self.live[q] && ideal_logical(&self.m.b.frame, self.k, b) != (false, false) {
                self.counts.logical_fault = true;
                return;
            }
        }
    }
}

impl<B: Phys> Encoded<B> {
    fn magic_slot(&self) -> usize {
        self.blocks.len()
    }
    fn t_gadget(&mut self, q: usize, dagger: bool) {
        self.counts.t_gadgets += 1;
        let k = self.k;
        let mb = self.m.alloc(k);
        match self.magic {
            MagicMode::Raw => self.m.inject(k, mb),
            MagicMode::Model(eps) => {
                self.m.inject_perfect(k, mb);
                if eps > 0.0 && self.mrng.f64() < eps {
                    self.m.pauli_block(k, mb, PZ);
                }
            }
        }
        let ms = self.magic_slot();
        if let Some(sv) = &mut self.sv {
            sv.reset(ms, false, &mut self.mrng);
            sv.h(ms);
            sv.t(ms);
        }
        let d = self.blocks[q];
        self.m.cnot(k, d, mb);
        if let Some(sv) = &mut self.sv {
            sv.cnot(q, ms);
        }
        let rec = self.meas_block(ms, mb);
        self.m.release(k, mb);
        // T: outcome 1 leaves T†, fix with S. T†: outcome 0 leaves T, fix with S†.
        let apply = rec != dagger;
        if apply {
            if dagger {
                self.m.sdg(k, d);
            } else {
                self.m.s(k, d);
            }
            if let Some(sv) = &mut self.sv {
                if dagger {
                    sv.sdg(q)
                } else {
                    sv.s(q)
                }
            }
        } else {
            self.m.id(k, d);
        }
    }
    /// Measure block `b` holding logical slot `slot`: recorded outcome.
    fn meas_block(&mut self, slot: usize, b: usize) -> bool {
        let r = self.m.meas_z(self.k, b);
        if r && self.sv.is_some() {
            self.counts.logical_fault = true;
        }
        match &mut self.sv {
            Some(sv) => sv.measure(slot, &mut self.mrng) ^ r,
            None => r,
        }
    }
}

impl<B: Phys> Logical for Encoded<B> {
    fn prep(&mut self, q: usize, bit: bool) {
        let (k, b) = (self.k, self.blocks[q]);
        self.m.prep0(k, b);
        self.live[q] = true;
        if self.sv.is_some() {
            // Z_L stabilises the freshly prepared |0⟩_L / |1⟩_L: a decoded Z_L
            // in the frame is no error; absorb it (exact rewrite of the frame)
            // so that the logical-fault indicator only sees real errors.
            if let Some(fr) = self.m.b.frame_ref() {
                if ideal_logical(fr, k, b).1 {
                    self.m.pauli_block(k, b, PZ);
                }
            }
        }
        match &mut self.sv {
            // frame model: the ideal state carries the |1⟩, the frame only errors
            Some(sv) => sv.reset(q, bit, &mut self.mrng),
            // dense model: apply the (noiseless, Pauli-frame) logical X
            None => {
                if bit {
                    self.m.pauli_block(k, b, PX);
                }
            }
        }
    }
    fn h(&mut self, q: usize) {
        self.counts.logical_gates += 1;
        self.m.h(self.k, self.blocks[q]);
        if let Some(sv) = &mut self.sv {
            sv.h(q);
        }
    }
    fn s(&mut self, q: usize) {
        self.counts.logical_gates += 1;
        self.m.s(self.k, self.blocks[q]);
        if let Some(sv) = &mut self.sv {
            sv.s(q);
        }
    }
    fn sdg(&mut self, q: usize) {
        self.counts.logical_gates += 1;
        self.m.sdg(self.k, self.blocks[q]);
        if let Some(sv) = &mut self.sv {
            sv.sdg(q);
        }
    }
    fn t(&mut self, q: usize) {
        self.counts.logical_gates += 1;
        self.t_gadget(q, false);
    }
    fn tdg(&mut self, q: usize) {
        self.counts.logical_gates += 1;
        self.t_gadget(q, true);
    }
    fn cnot(&mut self, c: usize, t: usize) {
        self.counts.logical_gates += 1;
        self.m.cnot(self.k, self.blocks[c], self.blocks[t]);
        if let Some(sv) = &mut self.sv {
            sv.cnot(c, t);
        }
    }
    fn meas(&mut self, q: usize) -> bool {
        let b = self.blocks[q];
        self.live[q] = false;
        self.meas_block(q, b)
    }
    fn sdg_slot(&mut self, q: usize, apply: bool) {
        if apply {
            self.sdg(q);
        } else {
            self.counts.logical_gates += 1;
            self.m.id(self.k, self.blocks[q]);
        }
    }
}

/// Bare (unencoded) qubits with the same circuit-level noise: preparation and
/// measurement flips p, single-qubit depolarizing p after every 1q gate
/// (including T), two-qubit depolarizing p after every CNOT. With
/// `native_ccx`, a Toffoli is one gate followed by independent single-qubit
/// depolarizing p on each of its three qubits (the `shor-noise` model).
pub struct Unencoded {
    /// Ideal state of the `nlog` bare qubits (noise is applied directly to it).
    pub sv: LSv,
    /// Circuit-level noise source.
    pub noise: Noise,
    /// RNG for Born-rule measurements and resets.
    pub mrng: Xoshiro,
    /// Treat a Toffoli as a single noisy gate instead of the 7-T decomposition.
    pub native_ccx: bool,
    /// Number of noise locations visited so far.
    pub locations: u64,
}

impl Unencoded {
    /// `nlog` bare qubits in |0…0⟩; `seed` seeds the measurement RNG.
    pub fn new(nlog: usize, noise: Noise, native_ccx: bool, seed: u64) -> Self {
        Unencoded {
            sv: LSv::new(nlog),
            noise,
            mrng: Xoshiro::new(seed ^ 0x5EED_CAFE),
            native_ccx,
            locations: 0,
        }
    }
    fn n1(&mut self, q: usize) {
        self.locations += 1;
        let c = self.noise.loc1();
        if c != 0 {
            self.sv.pauli(q, c);
        }
    }
}

impl Logical for Unencoded {
    fn prep(&mut self, q: usize, bit: bool) {
        self.locations += 1;
        let f = self.noise.flip();
        self.sv.reset(q, bit ^ f, &mut self.mrng);
    }
    fn h(&mut self, q: usize) {
        self.sv.h(q);
        self.n1(q);
    }
    fn s(&mut self, q: usize) {
        self.sv.s(q);
        self.n1(q);
    }
    fn sdg(&mut self, q: usize) {
        self.sv.sdg(q);
        self.n1(q);
    }
    fn t(&mut self, q: usize) {
        self.sv.t(q);
        self.n1(q);
    }
    fn tdg(&mut self, q: usize) {
        self.sv.tdg(q);
        self.n1(q);
    }
    fn cnot(&mut self, c: usize, t: usize) {
        self.sv.cnot(c, t);
        self.locations += 1;
        let code = self.noise.loc2();
        if code != 0 {
            self.sv.pauli(c, code & 3);
            self.sv.pauli(t, code >> 2);
        }
    }
    fn meas(&mut self, q: usize) -> bool {
        self.locations += 1;
        let r = self.sv.measure(q, &mut self.mrng);
        r ^ self.noise.flip()
    }
    fn sdg_slot(&mut self, q: usize, apply: bool) {
        if apply {
            self.sdg(q);
        } else {
            self.n1(q);
        }
    }
    fn ccx(&mut self, c1: usize, c2: usize, t: usize) {
        if !self.native_ccx {
            // default decomposition
            self.h(t);
            self.cnot(c2, t);
            self.tdg(t);
            self.cnot(c1, t);
            self.t(t);
            self.cnot(c2, t);
            self.tdg(t);
            self.cnot(c1, t);
            self.t(c2);
            self.t(t);
            self.h(t);
            self.cnot(c1, c2);
            self.t(c1);
            self.tdg(c2);
            self.cnot(c1, c2);
            return;
        }
        self.sv.ccx(c1, c2, t);
        self.n1(c1);
        self.n1(c2);
        self.n1(t);
    }
}

/// Encoded frame machine that records logical faults after every operation.
pub struct Checked(pub Encoded<FrameBackend>);
impl Logical for Checked {
    fn prep(&mut self, q: usize, b: bool) {
        self.0.prep(q, b);
        self.0.check_frames();
    }
    fn h(&mut self, q: usize) {
        self.0.h(q);
        self.0.check_frames();
    }
    fn s(&mut self, q: usize) {
        self.0.s(q);
        self.0.check_frames();
    }
    fn sdg(&mut self, q: usize) {
        self.0.sdg(q);
        self.0.check_frames();
    }
    fn t(&mut self, q: usize) {
        self.0.t(q);
        self.0.check_frames();
    }
    fn tdg(&mut self, q: usize) {
        self.0.tdg(q);
        self.0.check_frames();
    }
    fn cnot(&mut self, c: usize, t: usize) {
        self.0.cnot(c, t);
        self.0.check_frames();
    }
    fn meas(&mut self, q: usize) -> bool {
        let r = self.0.meas(q);
        self.0.check_frames();
        r
    }
    fn sdg_slot(&mut self, q: usize, apply: bool) {
        self.0.sdg_slot(q, apply);
        self.0.check_frames();
    }
}

/// Logical error of injected |T⟩ states: (pX, pY, pZ, eps_twirled, accept).
pub fn inject_errors(
    level: usize,
    p: f64,
    trials: u64,
    seed: u64,
    ps: bool,
) -> (f64, f64, f64, f64, f64) {
    let cfg = FtConfig {
        ec: true,
        inject_postselect: ps,
    };
    let mut m = Machine::new(FrameBackend::default(), Noise::new(p, seed), cfg, level);
    let (mut nx, mut ny, mut nz) = (0u64, 0u64, 0u64);
    for _ in 0..trials {
        let q = m.alloc(level);
        m.inject(level, q);
        match ideal_logical(&m.b.frame, level, q) {
            (true, false) => nx += 1,
            (true, true) => ny += 1,
            (false, true) => nz += 1,
            _ => {}
        }
        m.release(level, q);
    }
    let t = trials as f64;
    let (px, py, pz) = (nx as f64 / t, ny as f64 / t, nz as f64 / t);
    let acc = trials as f64 / m.stats.inject_attempts.max(1) as f64;
    (px, py, pz, pz + 0.5 * (px + py), acc)
}
