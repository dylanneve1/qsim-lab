//! Symbolic-phase sampling of noisy Clifford circuits (after SymPhase,
//! arXiv:2311.03906).
//!
//! With Pauli noise, only the *signs* of a stabilizer simulation depend on
//! the random choices; the X/Z structure is the same in every shot. So every
//! measurement outcome is an affine GF(2) function of a fixed set of random
//! variables:
//!
//! ```text
//! m = m_ref xor A v
//! ```
//!
//! where `m_ref` is one noiseless reference sample and `v` collects
//! * one or more *fault* variables per noise location (a Pauli X/Y/Z flip,
//!   the two bits of a 1-qubit depolarizing error, the four bits of a
//!   2-qubit one, a measurement or reset flip), sampled from the noise
//!   distribution, and
//! * one uniformly random *coin* per point where a qubit is known to be in a
//!   Z eigenstate (start of the circuit, after each measurement and reset).
//!   Applying `Z` there with probability 1/2 leaves the state unchanged and
//!   re-randomises every later outcome that the reference sample fixed
//!   arbitrarily (Gidney, Stim, arXiv:2103.02202, §2.3).
//!
//! `A` is found once, by pushing a *symbolic* Pauli frame through the
//! circuit: for each qubit, the X and Z components of the frame are bit sets
//! over the variables, Clifford gates permute/XOR them, noise and coins add
//! new variables, and a measurement reads the X component of the measured
//! qubit. Each batch of 64 shots is then a sparse GF(2) matrix-vector
//! product on `u64` words (bit `s` = shot `s`).
//!
//! The distribution is exactly that of running the circuit shot by shot
//! with [`crate::Circuit::run_noisy`] on a [`super::Tableau`]
//! (`tests/symphase.rs` compares full outcome distributions exactly on small
//! circuits).

use super::Tableau;
use crate::circuit::{Circuit, Op, SimError};
use crate::gate::Gate;
use crate::noise::NoiseModel;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::{FRAC_PI_2, PI};

/// Upper bound on the symbolic frame (`2 n V` bits for `V` variables).
pub const MAX_FRAME_BYTES: u128 = 1 << 30;

/// Distribution of one group of variables.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VarDist {
    /// One variable, uniform.
    Coin,
    /// One variable, 1 with probability `p`.
    Flip(f64),
    /// Two variables `(x, z)`: X, Y, Z each with probability `p / 3`.
    Depol1(f64),
    /// Four variables `(x_a, z_a, x_b, z_b)`: each of the 15 non-identity
    /// two-qubit Paulis with probability `p / 15`.
    Depol2(f64),
}

impl VarDist {
    /// Number of variables in the group.
    pub fn len(&self) -> usize {
        match self {
            VarDist::Coin | VarDist::Flip(_) => 1,
            VarDist::Depol1(_) => 2,
            VarDist::Depol2(_) => 4,
        }
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    /// All outcomes of the group with their probabilities (bit `i` of the
    /// pattern = variable `i` of the group).
    pub fn outcomes(&self) -> Vec<(u32, f64)> {
        match *self {
            VarDist::Coin => vec![(0, 0.5), (1, 0.5)],
            VarDist::Flip(p) => vec![(0, 1.0 - p), (1, p)],
            VarDist::Depol1(p) => {
                let mut v = vec![(0, 1.0 - p)];
                v.extend((1..4).map(|k| (pauli_bits(k), p / 3.0)));
                v
            }
            VarDist::Depol2(p) => {
                let mut v = vec![(0, 1.0 - p)];
                v.extend((1..16).map(|k| (pauli_bits(k / 4) | (pauli_bits(k % 4) << 2), p / 15.0)));
                v
            }
        }
    }
}

/// Pauli index (1 = X, 2 = Y, 3 = Z, as in [`crate::noise`]) to `(x, z)`
/// bits `x | z << 1`.
fn pauli_bits(k: usize) -> u32 {
    match k {
        1 => 0b01,
        2 => 0b11,
        3 => 0b10,
        _ => 0,
    }
}

/// A group of consecutive variables with a joint distribution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VarGroup {
    pub first: u32,
    pub dist: VarDist,
}

/// Precomputed sampler for a noisy Clifford circuit.
#[derive(Clone, Debug)]
pub struct SymPhaseSampler {
    reference: Vec<bool>,
    /// CSR rows: variables whose XOR flips measurement `j`.
    row_start: Vec<u32>,
    row_vars: Vec<u32>,
    groups: Vec<VarGroup>,
    num_vars: usize,
}

/// Symbolic Pauli frame: per qubit, X and Z components as bit sets over the
/// variables.
struct Frame {
    words: usize,
    x: Vec<u64>,
    z: Vec<u64>,
}

impl Frame {
    fn line(v: &[u64], q: usize, w: usize) -> &[u64] {
        &v[q * w..(q + 1) * w]
    }

    /// `v[dst] ^= v[src]` for lines of `w` words.
    fn xor_line(v: &mut [u64], dst: usize, src: usize, w: usize) {
        debug_assert_ne!(dst, src);
        let (d, s) = if dst < src {
            let (a, b) = v.split_at_mut(src * w);
            (&mut a[dst * w..(dst + 1) * w], &b[..w])
        } else {
            let (a, b) = v.split_at_mut(dst * w);
            (&mut b[..w], &a[src * w..(src + 1) * w])
        };
        for (a, b) in d.iter_mut().zip(s) {
            *a ^= b;
        }
    }

    /// `x[q] <-> z[q]`.
    fn swap_xz(&mut self, q: usize) {
        let w = self.words;
        self.x[q * w..(q + 1) * w].swap_with_slice(&mut self.z[q * w..(q + 1) * w]);
    }

    /// `z[q] ^= x[q]` (conjugation by S or S†).
    fn s(&mut self, q: usize) {
        let w = self.words;
        let (x, z) = (&self.x[q * w..(q + 1) * w], &mut self.z[q * w..(q + 1) * w]);
        for (a, b) in z.iter_mut().zip(x) {
            *a ^= b;
        }
    }

    fn cnot(&mut self, c: usize, t: usize) {
        let w = self.words;
        Self::xor_line(&mut self.x, t, c, w);
        Self::xor_line(&mut self.z, c, t, w);
    }

    fn cz(&mut self, a: usize, b: usize) {
        let w = self.words;
        // z_a ^= x_b, z_b ^= x_a
        for (q, r) in [(a, b), (b, a)] {
            let (src, dst) = (&self.x[r * w..(r + 1) * w], &mut self.z[q * w..(q + 1) * w]);
            for (d, s) in dst.iter_mut().zip(src) {
                *d ^= s;
            }
        }
    }

    fn swap(&mut self, a: usize, b: usize) {
        let w = self.words;
        for v in [&mut self.x, &mut self.z] {
            for k in 0..w {
                v.swap(a * w + k, b * w + k);
            }
        }
    }

    fn flip(v: &mut [u64], q: usize, w: usize, var: u32) {
        v[q * w + var as usize / 64] ^= 1u64 << (var % 64);
    }

    /// Heisenberg-picture update of the frame by a Clifford gate (signs are
    /// irrelevant for a frame).
    fn apply(&mut self, g: &Gate) {
        match *g {
            Gate::H(a) => self.swap_xz(a),
            Gate::S(a) | Gate::Sdg(a) => self.s(a),
            Gate::X(_) | Gate::Y(_) | Gate::Z(_) => {}
            Gate::Cnot(c, t) => self.cnot(c, t),
            Gate::Cz(a, b) => self.cz(a, b),
            Gate::Swap(a, b) => self.swap(a, b),
            Gate::Phase(a, t) | Gate::Rz(a, t) => {
                if (t / FRAC_PI_2).round().rem_euclid(2.0) == 1.0 {
                    self.s(a);
                }
            }
            Gate::CPhase(a, b, t) => {
                if (t / PI).round().rem_euclid(2.0) == 1.0 {
                    self.cz(a, b);
                }
            }
            _ => unreachable!("non-Clifford gates are rejected by the reference run"),
        }
    }
}

fn unsupported(g: Gate) -> SimError {
    SimError::Unsupported {
        backend: "symphase",
        gate: g,
    }
}

impl SymPhaseSampler {
    /// Compiles `circuit` under `noise` (with the same semantics as
    /// [`Circuit::run_noisy`]). Classically controlled operations must be
    /// Pauli gates without gate noise (a conditioned fault would make the
    /// outcomes non-affine in the variables).
    pub fn new(circuit: &Circuit, noise: &NoiseModel) -> Result<Self, SimError> {
        let n = circuit.num_qubits;
        let gate_noise = |g: &Gate| -> Option<VarDist> {
            match g.qubits().len() {
                1 if noise.p_1q > 0.0 => Some(VarDist::Depol1(noise.p_1q)),
                2 if noise.p_2q > 0.0 => Some(VarDist::Depol2(noise.p_2q)),
                _ => None,
            }
        };

        // 1. Noiseless reference sample (random outcomes forced to 0), which
        //    also validates every gate.
        let mut tab = Tableau::try_new(n)?;
        let mut zero = StdRng::seed_from_u64(0);
        let mut reference = Vec::new();
        let mut num_vars = n; // initial coins
        for op in &circuit.ops {
            match *op {
                Op::Gate(g) => {
                    tab.apply_gate(&g)?;
                    num_vars += gate_noise(&g).map_or(0, |d| d.len());
                }
                Op::Measure(q) => {
                    check_qubit(q, n)?;
                    reference.push(tab.measure_with(q, Some(false), &mut zero).0);
                    num_vars += 1 + (noise.p_meas > 0.0) as usize;
                }
                Op::Reset(q) => {
                    check_qubit(q, n)?;
                    if tab.measure_with(q, Some(false), &mut zero).0 {
                        tab.x(q);
                    }
                    num_vars += 1 + (noise.p_reset > 0.0) as usize;
                }
                Op::ClassicControlled {
                    gate,
                    meas_index,
                    target_value,
                } => {
                    if !matches!(gate, Gate::X(_) | Gate::Y(_) | Gate::Z(_))
                        || gate_noise(&gate).is_some()
                    {
                        return Err(unsupported(gate));
                    }
                    if meas_index >= reference.len() {
                        return Err(SimError::ClassicalBitOutOfRange {
                            bit: meas_index,
                            available: reference.len(),
                        });
                    }
                    if reference[meas_index] == target_value {
                        tab.apply_gate(&gate)?;
                    }
                }
                Op::XFlip(q, p) | Op::YFlip(q, p) | Op::ZFlip(q, p) => {
                    check_qubit(q, n)?;
                    num_vars += (p > 0.0) as usize;
                }
                Op::Depolarize1q(q, p) => {
                    check_qubit(q, n)?;
                    num_vars += 2 * (p > 0.0) as usize;
                }
                Op::Depolarize2q(a, b, p) => {
                    check_qubit(a, n)?;
                    check_qubit(b, n)?;
                    num_vars += 4 * (p > 0.0) as usize;
                }
            }
        }

        // 2. Symbolic frame.
        let words = num_vars.div_ceil(64).max(1);
        let bytes = 2 * (n as u128) * (words as u128) * 8;
        if bytes > MAX_FRAME_BYTES {
            return Err(SimError::TooLarge {
                what: "symbolic Pauli frame",
                bytes,
                limit: MAX_FRAME_BYTES,
            });
        }
        let w = words;
        let mut f = Frame {
            words,
            x: vec![0; n * w],
            z: vec![0; n * w],
        };
        let mut groups = Vec::new();
        let mut next: u32 = 0;
        let mut alloc = |dist: VarDist, groups: &mut Vec<VarGroup>| -> u32 {
            let first = next;
            groups.push(VarGroup { first, dist });
            next += dist.len() as u32;
            first
        };
        for q in 0..n {
            let c = alloc(VarDist::Coin, &mut groups);
            Frame::flip(&mut f.z, q, w, c);
        }
        let mut row_start = vec![0u32];
        let mut row_vars: Vec<u32> = Vec::new();
        let mut scratch = vec![0u64; w];
        let push_row = |bits: &[u64], row_vars: &mut Vec<u32>, row_start: &mut Vec<u32>| {
            for (k, &word) in bits.iter().enumerate() {
                let mut m = word;
                while m != 0 {
                    row_vars.push((64 * k) as u32 + m.trailing_zeros());
                    m &= m - 1;
                }
            }
            row_start.push(row_vars.len() as u32);
        };
        let add_depol1 = |f: &mut Frame, q: usize, v: u32| {
            Frame::flip(&mut f.x, q, w, v);
            Frame::flip(&mut f.z, q, w, v + 1);
        };
        for op in &circuit.ops {
            match *op {
                Op::Gate(g) => {
                    f.apply(&g);
                    match gate_noise(&g) {
                        Some(d @ VarDist::Depol1(_)) => {
                            let v = alloc(d, &mut groups);
                            add_depol1(&mut f, g.qubits()[0], v);
                        }
                        Some(d @ VarDist::Depol2(_)) => {
                            let v = alloc(d, &mut groups);
                            let qs = g.qubits();
                            add_depol1(&mut f, qs[0], v);
                            add_depol1(&mut f, qs[1], v + 2);
                        }
                        _ => {}
                    }
                }
                Op::Measure(q) => {
                    scratch.copy_from_slice(Frame::line(&f.x, q, w));
                    if noise.p_meas > 0.0 {
                        let v = alloc(VarDist::Flip(noise.p_meas), &mut groups);
                        scratch[v as usize / 64] ^= 1u64 << (v % 64);
                    }
                    push_row(&scratch, &mut row_vars, &mut row_start);
                    let c = alloc(VarDist::Coin, &mut groups);
                    Frame::flip(&mut f.z, q, w, c);
                }
                Op::Reset(q) => {
                    f.x[q * w..(q + 1) * w].fill(0);
                    f.z[q * w..(q + 1) * w].fill(0);
                    let c = alloc(VarDist::Coin, &mut groups);
                    Frame::flip(&mut f.z, q, w, c);
                    if noise.p_reset > 0.0 {
                        let v = alloc(VarDist::Flip(noise.p_reset), &mut groups);
                        Frame::flip(&mut f.x, q, w, v);
                    }
                }
                Op::ClassicControlled {
                    gate, meas_index, ..
                } => {
                    // applied in a shot iff applied in the reference xor
                    // (that measurement differs from the reference)
                    let (s, e) = (
                        row_start[meas_index] as usize,
                        row_start[meas_index + 1] as usize,
                    );
                    let q = gate.qubits()[0];
                    let (fx, fz) = match gate {
                        Gate::X(_) => (true, false),
                        Gate::Y(_) => (true, true),
                        _ => (false, true),
                    };
                    for &v in &row_vars[s..e] {
                        if fx {
                            Frame::flip(&mut f.x, q, w, v);
                        }
                        if fz {
                            Frame::flip(&mut f.z, q, w, v);
                        }
                    }
                }
                Op::XFlip(q, p) | Op::YFlip(q, p) | Op::ZFlip(q, p) if p > 0.0 => {
                    let v = alloc(VarDist::Flip(p), &mut groups);
                    if !matches!(op, Op::ZFlip(..)) {
                        Frame::flip(&mut f.x, q, w, v);
                    }
                    if !matches!(op, Op::XFlip(..)) {
                        Frame::flip(&mut f.z, q, w, v);
                    }
                }
                Op::Depolarize1q(q, p) if p > 0.0 => {
                    let v = alloc(VarDist::Depol1(p), &mut groups);
                    add_depol1(&mut f, q, v);
                }
                Op::Depolarize2q(a, b, p) if p > 0.0 => {
                    let v = alloc(VarDist::Depol2(p), &mut groups);
                    add_depol1(&mut f, a, v);
                    add_depol1(&mut f, b, v + 2);
                }
                _ => {}
            }
        }
        debug_assert_eq!(next as usize, num_vars);

        // 3. Drop groups none of whose variables reach a measurement.
        let mut s = SymPhaseSampler {
            reference,
            row_start,
            row_vars,
            groups,
            num_vars,
        };
        s.prune();
        Ok(s)
    }

    /// A sampler for parities of measurement sets (e.g. QEC detectors:
    /// `m[r][k] xor m[r-1][k]`) instead of raw measurements. Row `i` of the
    /// new `A` is the XOR of the rows in `sets[i]`, so a fault that flips a
    /// stabilizer outcome in every later round touches only the detectors
    /// where it starts and stops: far fewer non-zeros than the raw rows.
    /// Variable groups no row needs any more are dropped.
    pub fn with_parities(&self, sets: &[Vec<usize>]) -> SymPhaseSampler {
        let mut parity = vec![false; self.num_vars];
        let mut touched: Vec<u32> = Vec::new();
        let mut reference = Vec::with_capacity(sets.len());
        let mut row_start = vec![0u32];
        let mut row_vars: Vec<u32> = Vec::new();
        for set in sets {
            let mut r = false;
            for &j in set {
                r ^= self.reference[j];
                for &v in self.row(j) {
                    if !parity[v as usize] {
                        touched.push(v);
                    }
                    parity[v as usize] ^= true;
                }
            }
            touched.sort_unstable();
            for &v in &touched {
                if std::mem::take(&mut parity[v as usize]) {
                    row_vars.push(v);
                }
            }
            touched.clear();
            reference.push(r);
            row_start.push(row_vars.len() as u32);
        }
        let mut out = SymPhaseSampler {
            reference,
            row_start,
            row_vars,
            groups: self.groups.clone(),
            num_vars: self.num_vars,
        };
        out.prune();
        out
    }

    /// Drops variable groups that no row uses and renumbers the rest.
    fn prune(&mut self) {
        let mut used = vec![false; self.num_vars];
        for &v in &self.row_vars {
            used[v as usize] = true;
        }
        let mut remap = vec![u32::MAX; self.num_vars];
        let mut kept = Vec::new();
        let mut m: u32 = 0;
        for g in &self.groups {
            let r = g.first as usize..g.first as usize + g.dist.len();
            if used[r.clone()].iter().any(|&u| u) {
                for (k, v) in r.enumerate() {
                    remap[v] = m + k as u32;
                }
                kept.push(VarGroup {
                    first: m,
                    dist: g.dist,
                });
                m += g.dist.len() as u32;
            }
        }
        for v in &mut self.row_vars {
            *v = remap[*v as usize];
        }
        self.groups = kept;
        self.num_vars = m as usize;
    }

    /// Number of measurements per shot.
    pub fn num_measurements(&self) -> usize {
        self.reference.len()
    }

    /// Number of random variables after pruning.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Non-zeros of `A` (word XORs per measurement word per 64 shots).
    pub fn nnz(&self) -> usize {
        self.row_vars.len()
    }

    /// The reference sample `m_ref`.
    pub fn reference(&self) -> &[bool] {
        &self.reference
    }

    /// Variables of row `j` of `A`.
    pub fn row(&self, j: usize) -> &[u32] {
        &self.row_vars[self.row_start[j] as usize..self.row_start[j + 1] as usize]
    }

    /// Variable groups and their distributions.
    pub fn groups(&self) -> &[VarGroup] {
        &self.groups
    }

    /// Draws all variables for 64 shots into `vals` (bit `s` = shot `s`).
    ///
    /// Coins are one random word each. Rare faults are drawn by geometric
    /// skipping over the (group, shot) grid, so their cost is proportional
    /// to the number of faults that actually occur.
    pub fn sample_vars<R: Rng + ?Sized>(&self, rng: &mut R, vals: &mut [u64]) {
        vals.fill(0);
        self.draw_vars(rng, vals, None);
    }

    /// Like [`Self::sample_vars`] (same draws, same result for the same RNG
    /// stream) but without clearing all `num_vars` words: `touched` must list
    /// every non-zero word of `vals` on entry (as left by the previous call,
    /// or empty with `vals` all zero); only those are cleared. At low noise
    /// this replaces an `O(num_vars)` memset per 64 shots by `O(#faults)`.
    pub fn sample_vars_sparse<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        vals: &mut [u64],
        touched: &mut Vec<u32>,
    ) {
        for &i in touched.iter() {
            vals[i as usize] = 0;
        }
        touched.clear();
        self.draw_vars(rng, vals, Some(touched));
    }

    fn draw_vars<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        vals: &mut [u64],
        mut touched: Option<&mut Vec<u32>>,
    ) {
        let mut i = 0;
        while i < self.groups.len() {
            let g = self.groups[i];
            let p = match g.dist {
                VarDist::Coin => {
                    vals[g.first as usize] = rng.random();
                    if let Some(t) = touched.as_deref_mut() {
                        t.push(g.first);
                    }
                    i += 1;
                    continue;
                }
                VarDist::Flip(p) | VarDist::Depol1(p) | VarDist::Depol2(p) => p,
            };
            // run of consecutive groups with the same distribution
            let mut j = i + 1;
            while j < self.groups.len() && self.groups[j].dist == g.dist {
                j += 1;
            }
            let cells = (j - i) * 64;
            let mut c = 0usize;
            loop {
                c += geometric_skip(p, rng);
                if c >= cells {
                    break;
                }
                let grp = self.groups[i + c / 64];
                let bit = 1u64 << (c % 64);
                let pattern = match grp.dist {
                    VarDist::Flip(_) => 1,
                    VarDist::Depol1(_) => pauli_bits(rng.random_range(1..4usize)),
                    VarDist::Depol2(_) => {
                        let k = rng.random_range(1..16usize);
                        pauli_bits(k / 4) | (pauli_bits(k % 4) << 2)
                    }
                    VarDist::Coin => unreachable!(),
                };
                let mut m = pattern;
                while m != 0 {
                    let v = grp.first as usize + m.trailing_zeros() as usize;
                    if vals[v] == 0 {
                        if let Some(t) = touched.as_deref_mut() {
                            t.push(v as u32);
                        }
                    }
                    vals[v] |= bit;
                    m &= m - 1;
                }
                c += 1;
            }
            i = j;
        }
    }

    /// Measurement records for 64 shots: `out[j]` bit `s` is measurement
    /// `j` of shot `s`. `vals` is scratch space of [`Self::num_vars`] words.
    pub fn sample_batch<R: Rng + ?Sized>(&self, rng: &mut R, vals: &mut [u64], out: &mut [u64]) {
        self.sample_vars(rng, vals);
        self.eval(vals, out);
    }

    /// Column (variable -> rows) view of `A`, for [`Self::eval_sparse`].
    pub fn column_view(&self) -> ColumnView {
        let mut start = vec![0u32; self.num_vars + 1];
        for &v in &self.row_vars {
            start[v as usize + 1] += 1;
        }
        for i in 0..self.num_vars {
            start[i + 1] += start[i];
        }
        let mut fill = start.clone();
        let mut rows = vec![0u32; self.row_vars.len()];
        for j in 0..self.num_measurements() {
            for &v in self.row(j) {
                rows[fill[v as usize] as usize] = j as u32;
                fill[v as usize] += 1;
            }
        }
        ColumnView { start, rows }
    }

    /// Same result as [`Self::eval`], computed column-wise from the
    /// non-zero words listed in `touched` (as produced by
    /// [`Self::sample_vars_sparse`]): `O(#faults x column weight)` instead of
    /// `O(nnz)` per 64 shots.
    pub fn eval_sparse(&self, cv: &ColumnView, vals: &[u64], touched: &[u32], out: &mut [u64]) {
        for (j, o) in out.iter_mut().enumerate() {
            *o = 0u64.wrapping_sub(self.reference[j] as u64);
        }
        for &v in touched {
            let w = vals[v as usize];
            let (a, b) = (
                cv.start[v as usize] as usize,
                cv.start[v as usize + 1] as usize,
            );
            for &r in &cv.rows[a..b] {
                out[r as usize] ^= w;
            }
        }
    }

    /// `out = m_ref xor A vals`, 64 shots per word.
    pub fn eval(&self, vals: &[u64], out: &mut [u64]) {
        for (j, o) in out.iter_mut().enumerate() {
            let mut acc = 0u64.wrapping_sub(self.reference[j] as u64);
            for &v in self.row(j) {
                acc ^= vals[v as usize];
            }
            *o = acc;
        }
    }

    /// `shots` measurement records, one `Vec<bool>` per shot.
    pub fn sample<R: Rng + ?Sized>(&self, shots: usize, rng: &mut R) -> Vec<Vec<bool>> {
        let m = self.num_measurements();
        let mut vals = vec![0u64; self.num_vars];
        let mut out = vec![0u64; m];
        let mut res = Vec::with_capacity(shots);
        while res.len() < shots {
            self.sample_batch(rng, &mut vals, &mut out);
            for s in 0..64.min(shots - res.len()) {
                res.push(out.iter().map(|w| (w >> s) & 1 == 1).collect());
            }
        }
        res
    }
}

/// Transposed (CSC) form of a sampler's matrix `A`.
#[derive(Clone, Debug)]
pub struct ColumnView {
    start: Vec<u32>,
    rows: Vec<u32>,
}

/// Number of failures before the next success of a Bernoulli(`p`) process.
fn geometric_skip<R: Rng + ?Sized>(p: f64, rng: &mut R) -> usize {
    if p >= 1.0 {
        return 0;
    }
    // U in (0, 1]
    let u: f64 = 1.0 - rng.random::<f64>();
    let k = (u.ln() / (1.0 - p).ln()).floor();
    if k >= usize::MAX as f64 {
        usize::MAX / 2
    } else {
        k as usize
    }
}

fn check_qubit(q: usize, n: usize) -> Result<(), SimError> {
    if q >= n {
        Err(SimError::QubitOutOfRange {
            qubit: q,
            num_qubits: n,
        })
    } else {
        Ok(())
    }
}
