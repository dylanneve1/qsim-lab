//! Concatenated Steane-code machine. A level-k block is `7^k` consecutive
//! physical qubits; sub-block j of a level-k block starting at `b` starts at
//! `b + j·7^(k−1)`. Every level-k operation is built from level-(k−1)
//! operations exactly as in Aliferis–Gottesman–Preskill concatenation:
//!
//! * gates: transversal (H, S = transversal S†, CNOT), then Steane EC on
//!   every output block;
//! * preparation of |0⟩_L / |+⟩_L: non-FT encoder + one verification qubit
//!   measuring a weight-3 logical operator (Goto 2016), repeat until it
//!   passes; data preparations are followed by EC;
//! * Steane EC: couple a verified |+⟩_L (X errors) and then a verified |0⟩_L
//!   (Z errors) transversally, measure them transversally, decode each with
//!   the Hamming lookup (hierarchically: the 7 bits are level-(k−1) decoded
//!   outcomes), apply the correction as a (noiseless) Pauli-frame update;
//! * measurement: transversal, hierarchical hard-decision decoding;
//! * magic-state injection: a level-(k−1) |T⟩ on sub-block 2, the |ψ⟩ encoder,
//!   one EC (optionally post-selected on a trivial syndrome).
//!
//! The machine is generic over the physical backend [`Phys`]: the exact Pauli
//! frame ([`crate::ft::backends::FrameBackend`], used for the experiments) or a
//! dense state vector with real measurements ([`crate::ft::backends::DenseBackend`],
//! used only to validate the frame model). Noise is drawn by the machine, so
//! both backends see identical fault realisations for the same seed.
//! Measurement methods return the *recorded decoded value* on the dense
//! backend and the *flip relative to the ideal value* on the frame backend;
//! the decoding code is linear, so the same code serves both.

use super::core::*;

/// Physical backend: a Pauli frame or a dense state.
pub trait Phys {
    fn ensure(&mut self, n: usize);
    fn prep0(&mut self, q: usize);
    fn prep_plus(&mut self, q: usize);
    fn prep_t(&mut self, q: usize);
    fn h(&mut self, q: usize);
    fn s(&mut self, q: usize);
    fn sdg(&mut self, q: usize);
    fn cnot(&mut self, c: usize, t: usize);
    fn meas_z(&mut self, q: usize) -> bool;
    fn meas_x(&mut self, q: usize) -> bool;
    /// Apply a Pauli (code: bit0 X, bit1 Z).
    fn pauli(&mut self, q: usize, code: u8);
}

#[derive(Clone, Copy, Debug)]
pub struct FtConfig {
    /// Steane EC after every gate / data preparation (always on in the
    /// experiments; off only in cheap dense-validation tests).
    pub ec: bool,
    /// Post-select magic-state injection on a trivial syndrome of its first EC.
    pub inject_postselect: bool,
}

impl Default for FtConfig {
    fn default() -> Self {
        FtConfig {
            ec: true,
            inject_postselect: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub prep_rejects: u64,
    pub inject_rejects: u64,
    pub inject_attempts: u64,
    pub phys_prep: u64,
    pub phys_1q: u64,
    pub phys_2q: u64,
    pub phys_meas: u64,
    pub ec_calls: u64,
}

pub const POW7: [usize; 5] = [1, 7, 49, 343, 2401];

pub struct Machine<B: Phys> {
    pub b: B,
    pub noise: Noise,
    pub cfg: FtConfig,
    /// The top (logical) level: component tags are only set by operations at
    /// this level.
    pub top: usize,
    free: Vec<Vec<usize>>,
    next: usize,
    pub stats: Stats,
}

#[inline]
fn sub(k: usize, q: usize, j: usize) -> usize {
    q + j * POW7[k - 1]
}

impl<B: Phys> Machine<B> {
    pub fn new(b: B, noise: Noise, cfg: FtConfig, top: usize) -> Self {
        Machine {
            b,
            noise,
            cfg,
            top,
            free: vec![Vec::new(); 5],
            next: 0,
            stats: Stats::default(),
        }
    }
    /// Number of physical qubits ever allocated (high-water mark).
    pub fn phys_qubits(&self) -> usize {
        self.next
    }
    pub fn alloc(&mut self, k: usize) -> usize {
        if let Some(q) = self.free[k].pop() {
            return q;
        }
        let q = self.next;
        self.next += POW7[k];
        self.b.ensure(self.next);
        q
    }
    pub fn release(&mut self, k: usize, q: usize) {
        self.free[k].push(q);
    }
    #[inline]
    fn tag(&mut self, k: usize, c: Comp) {
        if k == self.top {
            self.noise.comp = c;
        }
    }
    #[inline]
    fn n1(&mut self, q: usize) {
        let c = self.noise.loc1();
        if c != 0 {
            self.b.pauli(q, c);
        }
    }
    #[inline]
    fn n2(&mut self, a: usize, t: usize) {
        let c = self.noise.loc2();
        if c != 0 {
            if c & 3 != 0 {
                self.b.pauli(a, c & 3);
            }
            if c >> 2 != 0 {
                self.b.pauli(t, c >> 2);
            }
        }
    }

    /// Noiseless Pauli on a whole level-k block (logical X_L = X^⊗7^k, Z_L
    /// likewise): Pauli-frame corrections and Pauli "gates".
    pub fn pauli_block(&mut self, k: usize, q: usize, code: u8) {
        for i in q..q + POW7[k] {
            self.b.pauli(i, code);
        }
    }

    // ------------------------------------------------------------ preparation
    pub fn prep0(&mut self, k: usize, q: usize) {
        if k == 0 {
            self.stats.phys_prep += 1;
            self.b.prep0(q);
            if self.noise.flip() {
                self.b.pauli(q, PX);
            }
        } else {
            self.tag(k, Comp::Prep);
            self.prep_raw(k, q, false);
            if self.cfg.ec {
                self.tag(k, Comp::Ec);
                self.ec(k, q);
            }
        }
    }
    pub fn prep_plus(&mut self, k: usize, q: usize) {
        if k == 0 {
            self.stats.phys_prep += 1;
            self.b.prep_plus(q);
            if self.noise.flip() {
                self.b.pauli(q, PZ);
            }
        } else {
            self.tag(k, Comp::Prep);
            self.prep_raw(k, q, true);
            if self.cfg.ec {
                self.tag(k, Comp::Ec);
                self.ec(k, q);
            }
        }
    }
    /// Verified |0⟩_L (`plus = false`) or |+⟩_L (`plus = true`), no trailing EC.
    pub fn prep_raw(&mut self, k: usize, q: usize, plus: bool) {
        debug_assert!(k >= 1);
        loop {
            for j in 0..7 {
                let pivot = STEANE_PIVOTS.contains(&j);
                if pivot != plus {
                    self.prep_plus(k - 1, sub(k, q, j));
                } else {
                    self.prep0(k - 1, sub(k, q, j));
                }
            }
            for &(p, t) in STEANE_ENC.iter() {
                if plus {
                    self.cnot(k - 1, sub(k, q, t), sub(k, q, p));
                } else {
                    self.cnot(k - 1, sub(k, q, p), sub(k, q, t));
                }
            }
            let v = self.alloc(k - 1);
            let bad = if plus {
                self.prep_plus(k - 1, v);
                for &r in STEANE_VERIFY.iter() {
                    self.cnot(k - 1, v, sub(k, q, r));
                }
                self.meas_x(k - 1, v)
            } else {
                self.prep0(k - 1, v);
                for &r in STEANE_VERIFY.iter() {
                    self.cnot(k - 1, sub(k, q, r), v);
                }
                self.meas_z(k - 1, v)
            };
            self.release(k - 1, v);
            if !bad {
                return;
            }
            self.stats.prep_rejects += 1;
        }
    }

    /// Encode |T⟩ at level k (non-fault-tolerant injection).
    pub fn inject(&mut self, k: usize, q: usize) {
        if k == 0 {
            self.stats.phys_prep += 1;
            self.b.prep_t(q);
            self.n1(q);
            return;
        }
        loop {
            self.tag(k, Comp::Inject);
            self.stats.inject_attempts += u64::from(k == self.top);
            let (inp, cp) = STEANE_INJECT;
            self.inject(k - 1, sub(k, q, inp));
            for j in 0..7 {
                if j == inp {
                    continue;
                }
                if STEANE_PIVOTS.contains(&j) {
                    self.prep_plus(k - 1, sub(k, q, j));
                } else {
                    self.prep0(k - 1, sub(k, q, j));
                }
            }
            for &t in cp.iter() {
                self.cnot(k - 1, sub(k, q, inp), sub(k, q, t));
            }
            for &(p, t) in STEANE_ENC.iter() {
                self.cnot(k - 1, sub(k, q, p), sub(k, q, t));
            }
            self.tag(k, Comp::Inject);
            let nontrivial = self.cfg.ec && self.ec_inner(k, q);
            if !(self.cfg.inject_postselect && nontrivial) {
                return;
            }
            if k == self.top {
                self.stats.inject_rejects += 1;
            }
        }
    }

    /// A noiseless encoded |T⟩ (for the "ideal / distilled magic" modes).
    pub fn inject_perfect(&mut self, k: usize, q: usize) {
        let s = self.noise.suspended;
        self.noise.suspended = true;
        self.inject(k, q);
        self.noise.suspended = s;
    }

    // ------------------------------------------------------------------ gates
    pub fn h(&mut self, k: usize, q: usize) {
        if k == 0 {
            self.stats.phys_1q += 1;
            self.b.h(q);
            self.n1(q);
        } else {
            self.tag(k, Comp::Gate);
            for j in 0..7 {
                self.h(k - 1, sub(k, q, j));
            }
            self.trailing_ec(k, q);
        }
    }
    /// Logical S (transversal S† on the level below for the Steane code).
    pub fn s(&mut self, k: usize, q: usize) {
        if k == 0 {
            self.stats.phys_1q += 1;
            self.b.s(q);
            self.n1(q);
        } else {
            self.tag(k, Comp::Gate);
            for j in 0..7 {
                self.sdg(k - 1, sub(k, q, j));
            }
            self.trailing_ec(k, q);
        }
    }
    pub fn sdg(&mut self, k: usize, q: usize) {
        if k == 0 {
            self.stats.phys_1q += 1;
            self.b.sdg(q);
            self.n1(q);
        } else {
            self.tag(k, Comp::Gate);
            for j in 0..7 {
                self.s(k - 1, sub(k, q, j));
            }
            self.trailing_ec(k, q);
        }
    }
    /// Identity gate slot (a noisy location that does nothing): used for the
    /// not-applied branch of a classically controlled S so that the location
    /// structure does not depend on measurement outcomes.
    pub fn id(&mut self, k: usize, q: usize) {
        if k == 0 {
            self.stats.phys_1q += 1;
            self.n1(q);
        } else {
            self.tag(k, Comp::Gate);
            for j in 0..7 {
                self.id(k - 1, sub(k, q, j));
            }
            self.trailing_ec(k, q);
        }
    }
    pub fn cnot(&mut self, k: usize, c: usize, t: usize) {
        if k == 0 {
            self.stats.phys_2q += 1;
            self.b.cnot(c, t);
            self.n2(c, t);
        } else {
            self.tag(k, Comp::Gate);
            for j in 0..7 {
                self.cnot(k - 1, sub(k, c, j), sub(k, t, j));
            }
            if self.cfg.ec {
                self.tag(k, Comp::Ec);
                self.ec(k, c);
                self.ec(k, t);
            }
        }
    }
    fn trailing_ec(&mut self, k: usize, q: usize) {
        if self.cfg.ec {
            self.tag(k, Comp::Ec);
            self.ec(k, q);
        }
    }

    // ----------------------------------------------------------- measurement
    pub fn meas_z(&mut self, k: usize, q: usize) -> bool {
        if k == 0 {
            self.stats.phys_meas += 1;
            let r = self.b.meas_z(q);
            r ^ self.noise.flip()
        } else {
            self.tag(k, Comp::Meas);
            let mut f = 0u8;
            for j in 0..7 {
                if self.meas_z(k - 1, sub(k, q, j)) {
                    f |= 1 << j;
                }
            }
            steane_decode(f)
        }
    }
    pub fn meas_x(&mut self, k: usize, q: usize) -> bool {
        if k == 0 {
            self.stats.phys_meas += 1;
            let r = self.b.meas_x(q);
            r ^ self.noise.flip()
        } else {
            self.tag(k, Comp::Meas);
            let mut f = 0u8;
            for j in 0..7 {
                if self.meas_x(k - 1, sub(k, q, j)) {
                    f |= 1 << j;
                }
            }
            steane_decode(f)
        }
    }

    // --------------------------------------------------------- Steane EC
    /// Steane EC on a level-k block; returns whether any syndrome was
    /// non-trivial. The component tag is left to the caller.
    pub fn ec(&mut self, k: usize, q: usize) -> bool {
        self.ec_inner(k, q)
    }
    fn ec_inner(&mut self, k: usize, q: usize) -> bool {
        debug_assert!(k >= 1);
        self.stats.ec_calls += 1;
        let saved = self.noise.comp;
        // X errors: |+>_L ancilla, CNOT data -> ancilla, measure ancilla in Z.
        let a = self.alloc(k);
        self.prep_raw(k, a, true);
        self.noise.comp = saved;
        for j in 0..7 {
            self.cnot(k - 1, sub(k, q, j), sub(k, a, j));
        }
        let mut f = 0u8;
        for j in 0..7 {
            if self.meas_z(k - 1, sub(k, a, j)) {
                f |= 1 << j;
            }
        }
        self.noise.comp = saved;
        self.release(k, a);
        let sx = steane_syndrome(f);
        if sx != 0 {
            self.pauli_block(k - 1, sub(k, q, (sx - 1) as usize), PX);
        }
        // Z errors: |0>_L ancilla, CNOT ancilla -> data, measure ancilla in X.
        let a = self.alloc(k);
        self.prep_raw(k, a, false);
        self.noise.comp = saved;
        for j in 0..7 {
            self.cnot(k - 1, sub(k, a, j), sub(k, q, j));
        }
        let mut g = 0u8;
        for j in 0..7 {
            if self.meas_x(k - 1, sub(k, a, j)) {
                g |= 1 << j;
            }
        }
        self.noise.comp = saved;
        self.release(k, a);
        let sz = steane_syndrome(g);
        if sz != 0 {
            self.pauli_block(k - 1, sub(k, q, (sz - 1) as usize), PZ);
        }
        sx != 0 || sz != 0
    }
}

/// Ideal hierarchical decoding of a Pauli frame on a level-k block: the
/// logical (X part, Z part) the block would read out under a perfect
/// transversal measurement. Frame backend only.
pub fn ideal_logical(frame: &[u8], k: usize, q: usize) -> (bool, bool) {
    if k == 0 {
        return (frame[q] & 1 == 1, frame[q] & 2 == 2);
    }
    let (mut fx, mut fz) = (0u8, 0u8);
    for j in 0..7 {
        let (x, z) = ideal_logical(frame, k - 1, sub(k, q, j));
        fx |= (x as u8) << j;
        fz |= (z as u8) << j;
    }
    (steane_decode(fx), steane_decode(fz))
}

#[cfg(test)]
mod tests {
    use super::super::backends::FrameBackend;
    use super::*;

    fn machine(noise: Noise, top: usize) -> Machine<FrameBackend> {
        Machine::new(FrameBackend::default(), noise, FtConfig::default(), top)
    }

    /// Run `body` once noiselessly to count its locations, then once per
    /// (location, Pauli) single fault; `check` must accept every outcome.
    fn all_single_faults(
        top: usize,
        body: &dyn Fn(&mut Machine<FrameBackend>) -> bool,
    ) -> (u64, u64) {
        let mut m = machine(Noise::scripted(vec![]), top);
        assert!(body(&mut m), "noiseless run must pass");
        let nloc = m.noise.loc;
        let mut tried = 0;
        let mut failed = 0;
        for l in 0..nloc {
            for code in 1..=15u8 {
                let mut m = machine(Noise::scripted(vec![(l, code)]), top);
                tried += 1;
                if !body(&mut m) {
                    failed += 1;
                }
            }
        }
        (tried, failed)
    }

    fn clean(m: &Machine<FrameBackend>, k: usize, q: usize) -> bool {
        ideal_logical(&m.b.frame, k, q) == (false, false)
    }

    #[test]
    fn prep_is_fault_tolerant() {
        // verified prep + EC, then a perfect readout: no single fault may
        // leave a logical error (Z/X class) on |0>_L or |+>_L.
        for plus in [false, true] {
            let (t, f) = all_single_faults(1, &|m| {
                let q = m.alloc(1);
                if plus {
                    m.prep_plus(1, q)
                } else {
                    m.prep0(1, q)
                }
                // a second EC: the output of the first must be correctable
                m.ec(1, q);
                let (x, z) = ideal_logical(&m.b.frame, 1, q);
                if plus { !z } else { !x }
            });
            assert!(t > 1000);
            assert_eq!(f, 0, "plus={plus}: {f} of {t} single faults fail");
        }
    }

    #[test]
    fn cnot_exrec_is_fault_tolerant() {
        let (t, f) = all_single_faults(1, &|m| {
            let a = m.alloc(1);
            let b = m.alloc(1);
            m.noise.suspended = true;
            m.prep0(1, a);
            m.prep0(1, b);
            m.noise.suspended = false;
            m.ec(1, a);
            m.ec(1, b);
            m.cnot(1, a, b);
            m.h(1, a);
            m.s(1, b);
            m.noise.suspended = true;
            m.ec(1, a);
            m.ec(1, b);
            clean(m, 1, a) && clean(m, 1, b)
        });
        assert_eq!(f, 0, "{f} of {t} single faults fail");
    }

    #[test]
    fn measurement_exrec_is_fault_tolerant() {
        for xb in [false, true] {
            let (t, f) = all_single_faults(1, &|m| {
                let a = m.alloc(1);
                m.noise.suspended = true;
                m.prep0(1, a);
                m.noise.suspended = false;
                m.ec(1, a);
                !(if xb { m.meas_x(1, a) } else { m.meas_z(1, a) })
            });
            assert_eq!(f, 0, "{f} of {t}");
        }
    }

    #[test]
    fn level2_tolerates_random_three_faults() {
        // A level-2 1-exRec only fails if two level-1 exRecs inside it fail,
        // each needing two faults: any three faults must be harmless.
        let body = |m: &mut Machine<FrameBackend>| {
            let a = m.alloc(2);
            let b = m.alloc(2);
            m.noise.suspended = true;
            m.prep0(2, a);
            m.prep0(2, b);
            m.noise.suspended = false;
            m.cnot(2, a, b);
            m.noise.suspended = true;
            m.ec(2, a);
            m.ec(2, b);
            clean(m, 2, a) && clean(m, 2, b)
        };
        let mut m = machine(Noise::scripted(vec![]), 2);
        assert!(body(&mut m));
        let nloc = m.noise.loc;
        let mut rng = Xoshiro::new(5);
        for _ in 0..300 {
            let mut fl = Vec::new();
            for _ in 0..3 {
                fl.push((rng.next_u64() % nloc, 1 + (rng.next_u64() % 15) as u8));
            }
            fl.sort();
            fl.dedup_by_key(|x| x.0);
            let mut m = machine(Noise::scripted(fl.clone()), 2);
            assert!(body(&mut m), "faults {fl:?} broke a level-2 exRec");
        }
    }
}
