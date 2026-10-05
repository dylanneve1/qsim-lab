//! Shared pieces of the fault-tolerant simulator: a fast RNG, the circuit-level
//! noise source (random or scripted faults, per-component masks) and the
//! Steane-code tables.

/// xoshiro256++ (fast, deterministic per seed).
#[derive(Clone, Debug)]
pub struct Xoshiro {
    s: [u64; 4],
}

impl Xoshiro {
    /// Seeds the four state words from `seed` via SplitMix64, so every seed
    /// (including 0) gives a valid non-zero state.
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut s = [0u64; 4];
        for w in &mut s {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *w = x ^ (x >> 31);
        }
        Xoshiro { s }
    }
    /// Next 64 uniformly distributed bits (xoshiro256++ output function).
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let r = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }
    /// Uniform in [0, 1).
    #[inline]
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Which top-level component a physical location belongs to (for the failure
/// breakdown). Locations inside nested gadgets inherit the tag of the
/// top-level operation that created them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Comp {
    /// Verified preparation of a top-level data block (|0⟩_L, |1⟩_L).
    Prep = 0,
    /// The transversal part of a top-level Clifford gate (incl. the S/I slot
    /// of a T gadget).
    Gate = 1,
    /// Top-level Steane error correction (ancilla preparation, coupling,
    /// ancilla measurement) after gates and preparations.
    Ec = 2,
    /// Magic-state injection (encoding a noisy |T⟩ plus its first EC).
    Inject = 3,
    /// Transversal top-level measurement.
    Meas = 4,
}
/// Number of [`Comp`] variants.
pub const N_COMP: usize = 5;
/// Display names of the components, indexed by `Comp as usize`.
pub const COMP_NAMES: [&str; N_COMP] = ["prep", "gate", "ec", "inject", "meas"];
/// Bit mask with every component enabled (bit `i` = `Comp` with discriminant `i`).
pub const ALL_COMPS: u32 = (1 << N_COMP) - 1;

/// Pauli codes: bit 0 = X part, bit 1 = Z part (1 = X, 2 = Z, 3 = Y). Two-qubit
/// codes: low two bits act on the first qubit, high two bits on the second.
pub const PX: u8 = 1;
/// Z bit of a single-qubit Pauli code (see [`PX`]).
pub const PZ: u8 = 2;

/// Circuit-level noise source. Every physical location calls one of
/// [`Noise::loc1`], [`Noise::loc2`], [`Noise::flip`]; the location counter
/// advances by one each time (unless suspended), so a scripted fault list
/// addresses locations by index.
#[derive(Clone, Debug)]
pub struct Noise {
    /// Physical error rate per location (see [`Noise::loc1`], [`Noise::loc2`],
    /// [`Noise::flip`] for how it is split between Paulis).
    pub p: f64,
    /// Bit mask over [`Comp`]: faults only occur in enabled components (the
    /// RNG is consumed either way).
    pub mask: u32,
    rng: Xoshiro,
    /// Scripted faults (location index, Pauli code), sorted by index. When
    /// present the RNG is not used.
    script: Option<Vec<(u64, u8)>>,
    cursor: usize,
    /// Index of the next location (incremented once per non-suspended draw).
    pub loc: u64,
    /// Component that subsequent locations are attributed to; set by the caller.
    pub comp: Comp,
    /// While `true`, draws return no fault and neither the location counter nor
    /// the per-component counts advance.
    pub suspended: bool,
    /// Number of faults actually injected per component (only enabled ones).
    pub faults: [u64; N_COMP],
    /// Number of locations visited per component.
    pub locs: [u64; N_COMP],
}

impl Noise {
    /// Random noise at rate `p`, all components enabled, RNG seeded with `seed`.
    pub fn new(p: f64, seed: u64) -> Self {
        Noise {
            p,
            mask: ALL_COMPS,
            rng: Xoshiro::new(seed),
            script: None,
            cursor: 0,
            loc: 0,
            comp: Comp::Gate,
            suspended: false,
            faults: [0; N_COMP],
            locs: [0; N_COMP],
        }
    }
    /// Noise that injects exactly the given `(location index, Pauli code)`
    /// faults and nothing else. Codes are reduced onto each location's Pauli set
    /// (`1 + (c - 1) % k` for a location with `k` non-trivial Paulis); code 0
    /// means no fault. The list is sorted, so order does not matter.
    pub fn scripted(mut faults: Vec<(u64, u8)>) -> Self {
        faults.sort();
        let mut n = Noise::new(0.0, 0);
        n.script = Some(faults);
        n
    }
    /// Rewinds the location counter and script cursor and zeros the
    /// per-component fault/location counts (the RNG state is not reset).
    pub fn reset_counters(&mut self) {
        self.loc = 0;
        self.cursor = 0;
        self.faults = [0; N_COMP];
        self.locs = [0; N_COMP];
    }
    /// Draw the fault (0 = none) at one location with `k` non-trivial Paulis.
    #[inline]
    fn draw(&mut self, k: u32) -> u8 {
        if self.suspended {
            return 0;
        }
        let ci = self.comp as usize;
        self.locs[ci] += 1;
        let idx = self.loc;
        self.loc += 1;
        let code = if let Some(s) = &self.script {
            if self.cursor < s.len() && s[self.cursor].0 == idx {
                let c = s[self.cursor].1;
                self.cursor += 1;
                // map any code onto this location's Pauli set
                if c == 0 {
                    0
                } else {
                    1 + (c - 1) % k as u8
                }
            } else {
                0
            }
        } else {
            let u = self.rng.f64();
            if u < self.p {
                let v = ((u / self.p) * k as f64) as u32;
                (1 + v.min(k - 1)) as u8
            } else {
                0
            }
        };
        if code != 0 && (self.mask >> ci) & 1 == 1 {
            self.faults[ci] += 1;
            code
        } else {
            0
        }
    }
    /// Single-qubit depolarizing location: X, Z or Y with p/3 each.
    #[inline]
    pub fn loc1(&mut self) -> u8 {
        self.draw(3)
    }
    /// Two-qubit depolarizing location: each of the 15 non-identity Paulis
    /// with p/15.
    #[inline]
    pub fn loc2(&mut self) -> u8 {
        self.draw(15)
    }
    /// Preparation / measurement flip with probability p.
    #[inline]
    pub fn flip(&mut self) -> bool {
        self.draw(1) != 0
    }
}

// ---------------------------------------------------------------- Steane code

/// Steane [[7,1,3]]: qubit j has syndrome j + 1 (Hamming code). Stabilizer
/// supports (both X and Z type): {0,2,4,6}, {1,2,5,6}, {3,4,5,6}.
pub const STEANE_CHECKS: [u8; 3] = [0b1010101, 0b1100110, 0b1111000];

/// |0⟩_L encoder: pivots 0, 1, 3 start in |+⟩, the rest in |0⟩, then these
/// nine CNOTs (pivot → target). (This order also admits Goto's one-qubit
/// verification on [`STEANE_VERIFY`], but that leaves correlated X_a Z_b
/// errors that break transversal S; the machine uses Steane's checker-block
/// verification instead.)
pub const STEANE_PIVOTS: [usize; 3] = [0, 1, 3];
/// The nine encoder CNOTs as `(control, target)`, applied in this order.
pub const STEANE_ENC: [(usize, usize); 9] = [
    (0, 2),
    (0, 6),
    (0, 4),
    (1, 2),
    (1, 6),
    (1, 5),
    (3, 4),
    (3, 6),
    (3, 5),
];
/// Weight-3 logical representative (Goto-style verification; unused).
pub const STEANE_VERIFY: [usize; 3] = [2, 4, 5];
/// Weight-3 X_L representative avoiding the pivots: the injected qubit (2) is
/// copied to 4 and 5 before the pivot CNOTs.
pub const STEANE_INJECT: (usize, [usize; 2]) = (2, [4, 5]);

/// Syndrome of a 7-bit X (or Z) error pattern `f` (bit `j` = qubit `j`): the
/// XOR of `j + 1` over flipped qubits, so a single error on qubit `j` gives
/// `j + 1` and 0 means no detectable error. Bit `r` of the result is check
/// `STEANE_CHECKS[r]`.
#[inline]
pub fn steane_syndrome(f: u8) -> u8 {
    let mut s = 0u8;
    for j in 0..7 {
        if (f >> j) & 1 == 1 {
            s ^= (j + 1) as u8;
        }
    }
    s
}

/// Hard-decision decode of 7 measured bits: correct the single bit the
/// syndrome points to, return the parity (the logical value).
#[inline]
pub fn steane_decode(mut f: u8) -> bool {
    let s = steane_syndrome(f);
    if s != 0 {
        f ^= 1 << (s - 1);
    }
    f.count_ones() & 1 == 1
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checks_match_syndromes() {
        for j in 0..7u8 {
            let mut s = 0;
            for (r, c) in STEANE_CHECKS.iter().enumerate() {
                if (c >> j) & 1 == 1 {
                    s |= 1 << r;
                }
            }
            assert_eq!(s, j + 1);
            assert_eq!(steane_syndrome(1 << j), j + 1);
        }
        // the verification and injection supports are logical (weight-3
        // codewords of the Hamming code, odd parity)
        for sup in [STEANE_VERIFY, [2, 4, 5]] {
            let w: u8 = sup.iter().map(|&q| 1u8 << q).sum();
            assert_eq!(steane_syndrome(w), 0);
        }
    }
    #[test]
    fn noise_rates() {
        let mut n = Noise::new(0.1, 7);
        let mut c = [0u32; 16];
        for _ in 0..200_000 {
            c[n.loc2() as usize] += 1;
        }
        let tot: u32 = c[1..].iter().sum();
        assert!((tot as f64 / 2e5 - 0.1).abs() < 0.003);
        for &x in &c[1..] {
            assert!((x as f64 / tot as f64 - 1.0 / 15.0).abs() < 0.01);
        }
    }
}
