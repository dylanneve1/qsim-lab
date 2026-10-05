//! Exact noisy trajectories for **every** gate-level Shor oracle of the repo
//! that is a permutation circuit up to X-basis measurements: the round-4
//! windowed oracle, the superoptimised `windowed-opt` oracle and the two
//! measurement-based-uncomputation oracles (`windowed-mbu-lookup`,
//! `windowed-mbu`). Generalises [`super::noisy`] (same fault model, same
//! control algebra) in three ways:
//!
//! 1. **Op streams with X-basis measurements.** A round is a list of
//!    [`NOp`]s: X / CNOT / CCX / SWAP / Z / CZ gates and `MeasX(q, m)`, the
//!    X-basis measurement of `q` with *recorded* outcome `m` followed by a
//!    reset to `|0⟩`. Classically controlled fix-ups (the CZ of a temporary
//!    AND, the phase lookup of a measured unlookup, the phase comparator of
//!    a measured flag) are resolved from the recorded outcomes, exactly as
//!    `shor_mbu::resolve` does, so the op stream of a trajectory depends on
//!    its recorded outcomes.
//! 2. **Exact measurement semantics under faults, without per-measurement
//!    sorting.** On a branch list the X-basis projection onto outcome `m`
//!    maps `|…q…⟩ ↦ (−1)^{m·q}|…0…⟩` (times `1/√2`). Without faults the
//!    measured qubit is a function of the other qubits on every branch, so
//!    no two branches collide and `P(m) = ½`. A fault can break this (two
//!    branches that differ only in `q` then *merge*, and interfere). The
//!    engine keeps the branch list as a **multiset** (duplicates allowed,
//!    every op is linear and acts per branch) and sums duplicate keys once,
//!    at the end of the round, where it merges the two control halves
//!    anyway. The recorded outcomes are drawn uniformly (`P = ½`, the
//!    fault-free law) and the trajectory carries the **importance weight**
//!    `W = Π_rounds ‖state‖²` (unnormalised projections, no `1/√2`), which
//!    equals `Π_j 2·P(m_j | history)`: the estimator `E[W·ok]` is exact and
//!    `W = 1` on every trajectory without collisions.
//! 3. **Wide keys.** A branch is keyed by all non-control qubits in a
//!    [`Key`]: `u128` (≤ 128 non-control qubits) or [`K192`] (≤ 192), so the
//!    full-MBU oracle (`5n + 3 + w` qubits) runs at n = 24.
//!
//! # Fault locations
//!
//! As in [`super::noisy`]: per round `Prep`, `H1`, gate locations, `Phase`,
//! `H2`, `Meas` on the control; a Pauli after every gate on each of its
//! qubits (Z and CZ fix-ups are gates: 1 and 2 locations). An X-basis
//! measurement has **two** locations (depolarizing and bit-flip channels;
//! none under phase-flip noise, where a Z before the measurement — the
//! location after the previous gate on `q` — already flips the outcome):
//! slot 0, a readout flip (the recorded outcome, which selects the fix-up,
//! differs from the projection), and slot 1, a reset flip (`q` left in
//! `|1⟩`). Each is a flip with probability `p`, like `Prep` / `Meas`.
//!
//! Every location also carries a block tag ([`tag`]): lookup / unlookup /
//! modular adder / controlled swap, and the part (unary-iteration AND
//! chain, fan-out, measurement, fix-up, adder, reduction, comparator, flag),
//! for the per-block fatality map.

use super::noisy::{Capped, Fault, NoiseKind, Pauli, Site};
use super::sliced::{eval_raw_unchecked, transpose64};
use super::{postprocess, Instance, Oracle};
use crate::engines::statevector::Real;
use crate::gate::Gate;
use crate::shor::mbu::{self as shor_mbu, LOp, MbuLayout, MbuOpts};
use crate::shor::window::WindowLayout;
use num_complex::{Complex, Complex64};
use num_traits::Zero;
use rand::Rng;
use rayon::prelude::*;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Keys
// ---------------------------------------------------------------------------

/// A branch key: the non-control qubits (key bit `j` = qubit `j + 1`).
pub trait Key:
    Copy + Ord + Eq + Default + Send + Sync + std::fmt::Debug + std::hash::Hash + 'static
{
    /// 64-bit words.
    const WORDS: usize;
    /// Word `i` (`i < WORDS`), least significant first.
    fn word(&self, i: usize) -> u64;
    /// Overwrites word `i` with `v`.
    fn set_word(&mut self, i: usize, v: u64);
    /// Key whose low 64 bits are `v` (all higher bits 0).
    fn from_low(v: u64) -> Self {
        let mut k = Self::default();
        k.set_word(0, v);
        k
    }
    /// Bit `j` (= qubit `j + 1`).
    fn bit(&self, j: usize) -> bool {
        (self.word(j / 64) >> (j % 64)) & 1 == 1
    }
    /// `(bits < n, bits >= n)`.
    fn split(&self, n: usize) -> (Self, Self) {
        let mut lo = Self::default();
        let mut hi = Self::default();
        for i in 0..Self::WORDS {
            let w = self.word(i);
            let base = 64 * i;
            let m = if n <= base {
                0
            } else if n >= base + 64 {
                u64::MAX
            } else {
                (1u64 << (n - base)) - 1
            };
            lo.set_word(i, w & m);
            hi.set_word(i, w & !m);
        }
        (lo, hi)
    }
    /// True if any bit at position `>= n` is set.
    fn any_from(&self, n: usize) -> bool {
        self.split(n).1 != Self::default()
    }
    /// Low 64 bits.
    fn low64(&self) -> u64 {
        self.word(0)
    }
}

impl Key for u128 {
    const WORDS: usize = 2;
    fn word(&self, i: usize) -> u64 {
        (*self >> (64 * i)) as u64
    }
    fn set_word(&mut self, i: usize, v: u64) {
        let m = !(u128::from(u64::MAX) << (64 * i));
        *self = (*self & m) | (u128::from(v) << (64 * i));
    }
}

/// A 192-bit key (three words, least significant first).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct K192(pub [u64; 3]);

impl Key for K192 {
    const WORDS: usize = 3;
    fn word(&self, i: usize) -> u64 {
        self.0[i]
    }
    fn set_word(&mut self, i: usize, v: u64) {
        self.0[i] = v;
    }
}

// ---------------------------------------------------------------------------
// Ops, tags, rounds
// ---------------------------------------------------------------------------

/// A resolved op of a round.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NOp {
    /// X, CNOT, CCX, SWAP, Z or CZ.
    G(Gate),
    /// X-basis measurement of the qubit with the *recorded* outcome, then
    /// reset to `|0⟩`.
    MeasX(u32, bool),
    /// Z-basis measure-and-reset of the qubit group `Round::groups[i]`
    /// (design variants: mid-circuit reset of qubits that should be clean;
    /// the outcome is discarded). One location per qubit: a reset flip.
    ResetZ(u32),
    /// A control-site location of exponent qubit `j` in a windowed
    /// exponentiation round ([`GenCircuit::new_ge`]): kind 0 = preparation
    /// flip (X before the first H), 1 = Pauli after the first H, 2 = Pauli
    /// after the phase correction, 3 = Pauli after the second H, 4 = readout
    /// flip. No operation by itself (the H / phase / measurement of the
    /// exponent qubits are applied by the engine after the block).
    Ctrl(u8, u8),
}

/// Block tags: `block | part | INV`.
pub mod tag {
    /// The table lookup that computes the window's entry.
    pub const LOOKUP: u8 = 0x00;
    /// The table lookup that uncomputes it (coherent or measured).
    pub const UNLOOKUP: u8 = 0x10;
    /// The modular adder `b ← b + L mod N`.
    pub const MODADD: u8 = 0x20;
    /// The controlled swap of `x` and `b`.
    pub const SWAP: u8 = 0x30;
    /// Untagged.
    pub const OTHER: u8 = 0x40;
    /// A [`super::NOp::ResetZ`] (design variants).
    pub const RESET: u8 = 0x50;
    /// Exponent-qubit control sites (windowed exponentiation).
    pub const CTRL: u8 = 0x60;
    /// Mask of the block field.
    pub const BLOCK: u8 = 0x70;

    /// Lookup: unary-iteration / AND-chain gates (target an AND ancilla,
    /// temporary-AND measurements and their CZ).
    pub const UNARY: u8 = 0;
    /// Lookup: fan-out CNOTs into the output register.
    pub const FANOUT: u8 = 1;
    /// Measured unlookup: X-measurement of the output register.
    pub const MEAS: u8 = 2;
    /// Measured unlookup: the phase-lookup fix-up.
    pub const FIX: u8 = 3;
    /// Modular adder: the adder `b += L` (before the first gate on `K`).
    pub const ADD: u8 = 4;
    /// Modular adder: the reduction (first to last gate on `K`).
    pub const REDUCE: u8 = 5;
    /// Modular adder: the flag comparator (after the last gate on `K`).
    pub const CMP: u8 = 6;
    /// Modular adder: X-measurement of the flag.
    pub const FLAGMEAS: u8 = 7;
    /// Modular adder: the flag's phase-comparator fix-up.
    pub const FLAGFIX: u8 = 8;
    /// Anything else (controlled swap gates).
    pub const GATE: u8 = 9;
    /// Mask of the part field.
    pub const PART: u8 = 0x0f;
    /// The op belongs to the inverse multiplier (`a⁻¹`, second half).
    pub const INV: u8 = 0x80;

    /// Name of the block field of `t` (`lookup`, `unlookup`, `modadd`, `swap`,
    /// `reset`, `ctrl`, else `other`).
    pub fn block_name(t: u8) -> &'static str {
        match t & BLOCK {
            LOOKUP => "lookup",
            UNLOOKUP => "unlookup",
            MODADD => "modadd",
            SWAP => "swap",
            RESET => "reset",
            CTRL => "ctrl",
            _ => "other",
        }
    }
    /// Name of the part field of `t` (`unary`, `fanout`, `meas`, … ; `gate` for
    /// [`GATE`] and unknown parts).
    pub fn part_name(t: u8) -> &'static str {
        match t & PART {
            UNARY => "unary",
            FANOUT => "fanout",
            MEAS => "meas",
            FIX => "fix",
            ADD => "add",
            REDUCE => "reduce",
            CMP => "cmp",
            FLAGMEAS => "flagmeas",
            FLAGFIX => "flagfix",
            _ => "gate",
        }
    }
    /// `block.part` plus `.inv` for the inverse multiplier.
    pub fn name(t: u8) -> String {
        format!(
            "{}.{}{}",
            block_name(t),
            part_name(t),
            if t & INV != 0 { ".inv" } else { "" }
        )
    }
}

fn op_slots(op: &NOp, groups: &[Vec<u32>], kind: NoiseKind) -> u64 {
    let flips = !matches!(kind, NoiseKind::PhaseFlip);
    match op {
        NOp::G(g) => g.arity() as u64,
        NOp::MeasX(..) => 2 * u64::from(flips),
        NOp::ResetZ(gi) => groups[*gi as usize].len() as u64 * u64::from(flips),
        NOp::Ctrl(k, _) => {
            if matches!(k, 1..=3) {
                1
            } else {
                u64::from(flips)
            }
        }
    }
}

/// The ops of one round with their tags and location offsets.
#[derive(Clone, Debug)]
pub struct Round {
    /// The ops in application order.
    pub ops: Vec<NOp>,
    /// Block tag of every op ([`tag`]), parallel to `ops`.
    pub tags: Vec<u8>,
    /// Qubit groups of the [`NOp::ResetZ`] ops.
    pub groups: Vec<Vec<u32>>,
    slot_prefix: Vec<u64>,
}

impl Round {
    /// A round without [`NOp::ResetZ`] groups; panics if `ops` and `tags` differ
    /// in length.
    pub fn new(ops: Vec<NOp>, tags: Vec<u8>, kind: NoiseKind) -> Self {
        Self::with_groups(ops, tags, Vec::new(), kind)
    }
    /// A round whose [`NOp::ResetZ`] ops index `groups`; panics if `ops` and
    /// `tags` differ in length.
    pub fn with_groups(
        ops: Vec<NOp>,
        tags: Vec<u8>,
        groups: Vec<Vec<u32>>,
        kind: NoiseKind,
    ) -> Self {
        assert_eq!(ops.len(), tags.len());
        let mut slot_prefix = Vec::with_capacity(ops.len() + 1);
        let mut s = 0u64;
        slot_prefix.push(0);
        for op in &ops {
            s += op_slots(op, &groups, kind);
            slot_prefix.push(s);
        }
        Self {
            ops,
            tags,
            groups,
            slot_prefix,
        }
    }
    /// Number of op fault locations in the round under the round's noise kind
    /// (gate arities, two per X-measurement and one per reset qubit for the
    /// flip channels, the control sites of [`NOp::Ctrl`]).
    pub fn gate_slots(&self) -> u64 {
        *self.slot_prefix.last().unwrap()
    }
    /// Number of X-basis measurements.
    pub fn measurements(&self) -> usize {
        self.ops
            .iter()
            .filter(|o| matches!(o, NOp::MeasX(..)))
            .count()
    }
}

/// The resolved circuit of one trajectory (every round's op stream) with
/// its fault locations.
#[derive(Clone, Debug)]
pub struct Resolved {
    /// The noise channel.
    pub kind: NoiseKind,
    /// The rounds, in measurement order.
    pub rounds: Vec<Round>,
    loc_prefix: Vec<u64>,
    /// Per-round control sites (`Prep`, `H1`, `Phase`, `H2`, `Meas`); false
    /// for windowed exponentiation, whose control sites are [`NOp::Ctrl`].
    ctrl_sites: bool,
    /// Exponent qubits (windowed exponentiation; empty otherwise).
    pub eq: Vec<u32>,
}

impl Resolved {
    /// Fault locations of `rounds` with the per-round control sites (`Prep`, `H1`,
    /// `Phase`, `H2`, `Meas`) and no exponent qubits.
    pub fn new(rounds: Vec<Round>, kind: NoiseKind) -> Self {
        let pm = u64::from(has_prep_meas(kind));
        let mut loc_prefix = vec![0u64];
        for r in &rounds {
            let n = pm + 1 + r.gate_slots() + 2 + pm;
            loc_prefix.push(loc_prefix.last().unwrap() + n);
        }
        Self {
            kind,
            rounds,
            loc_prefix,
            ctrl_sites: true,
            eq: Vec::new(),
        }
    }

    /// Windows of a windowed exponentiation: every location is an op slot.
    pub fn new_windowed(rounds: Vec<Round>, kind: NoiseKind, eq: Vec<u32>) -> Self {
        let mut loc_prefix = vec![0u64];
        for r in &rounds {
            loc_prefix.push(loc_prefix.last().unwrap() + r.gate_slots());
        }
        Self {
            kind,
            rounds,
            loc_prefix,
            ctrl_sites: false,
            eq,
        }
    }

    /// Total number of fault locations `L`.
    pub fn num_locations(&self) -> u64 {
        *self.loc_prefix.last().unwrap()
    }

    /// Round and site of global location `g < L`.
    pub fn site_of(&self, g: u64) -> (usize, Site) {
        let i = self.loc_prefix.partition_point(|&x| x <= g) - 1;
        let mut l = g - self.loc_prefix[i];
        let pm = has_prep_meas(self.kind);
        let rc = &self.rounds[i];
        if !self.ctrl_sites {
            let gi = rc.slot_prefix.partition_point(|&x| x <= l) - 1;
            let slot = (l - rc.slot_prefix[gi]) as u8;
            return (
                i,
                Site::Gate {
                    gate: gi as u32,
                    slot,
                },
            );
        }
        if pm {
            if l == 0 {
                return (i, Site::Prep);
            }
            l -= 1;
        }
        if l == 0 {
            return (i, Site::H1);
        }
        l -= 1;
        let slots = rc.gate_slots();
        if l < slots {
            let gi = rc.slot_prefix.partition_point(|&x| x <= l) - 1;
            let slot = (l - rc.slot_prefix[gi]) as u8;
            return (
                i,
                Site::Gate {
                    gate: gi as u32,
                    slot,
                },
            );
        }
        l -= slots;
        match l {
            0 => (i, Site::Phase),
            1 => (i, Site::H2),
            2 if pm => (i, Site::Meas),
            _ => unreachable!("location {g} out of range"),
        }
    }

    fn fault_at<R: Rng + ?Sized>(&self, g: u64, rng: &mut R) -> Fault {
        let (round, site) = self.site_of(g);
        let pauli = match site {
            Site::Prep | Site::Meas => Pauli::X,
            Site::Gate { gate, .. }
                if matches!(
                    self.rounds[round].ops[gate as usize],
                    NOp::MeasX(..) | NOp::ResetZ(..) | NOp::Ctrl(0 | 4, _)
                ) =>
            {
                Pauli::X
            }
            _ => sample_pauli(self.kind, rng),
        };
        Fault {
            round: round as u32,
            site,
            pauli,
        }
    }

    /// Exactly `k` faults at a uniformly random `k`-subset of the locations.
    pub fn sample_k<R: Rng + ?Sized>(&self, k: usize, rng: &mut R) -> Vec<Fault> {
        let l = self.num_locations();
        assert!((k as u64) <= l);
        let mut picked = std::collections::BTreeSet::new();
        while picked.len() < k {
            picked.insert(rng.random_range(0..l));
        }
        picked.into_iter().map(|g| self.fault_at(g, rng)).collect()
    }

    /// Independent faults with probability `p` at every location.
    pub fn sample_p<R: Rng + ?Sized>(&self, p: f64, rng: &mut R) -> Vec<Fault> {
        let l = self.num_locations();
        let mut out = Vec::new();
        if p <= 0.0 {
            return out;
        }
        if p >= 1.0 {
            return (0..l).map(|g| self.fault_at(g, rng)).collect();
        }
        let ln1p = (-p).ln_1p();
        let mut g: u64 = 0;
        loop {
            let u: f64 = 1.0 - rng.random::<f64>();
            let skip = (u.ln() / ln1p).floor();
            if !skip.is_finite() || g as f64 + skip >= l as f64 {
                break;
            }
            g += skip as u64;
            out.push(self.fault_at(g, rng));
            g += 1;
            if g >= l {
                break;
            }
        }
        out
    }

    /// The op, the qubit and the tag hit by a gate-site fault.
    pub fn op_qubit(&self, f: &Fault) -> Option<(NOp, usize, u8)> {
        match f.site {
            Site::Gate { gate, slot } => {
                let r = &self.rounds[f.round as usize];
                let op = r.ops[gate as usize];
                let q = match op {
                    NOp::G(g) => g.qubits()[slot as usize],
                    NOp::MeasX(q, _) => q as usize,
                    NOp::ResetZ(g) => r.groups[g as usize][slot as usize] as usize,
                    NOp::Ctrl(_, j) => self.eq[j as usize] as usize,
                };
                Some((op, q, r.tags[gate as usize]))
            }
            _ => None,
        }
    }

    /// Compiles round `i` with its faults spliced in. Words: `0..nq` qubits,
    /// `nq` all-ones, `nq + 1` the sign.
    fn program(&self, nq: usize, i: usize, faults: &[Fault]) -> (Vec<Seg>, PostFaults) {
        let (segs, post, _) = self.program_w(nq, i, faults);
        (segs, post)
    }

    /// [`Self::program`] plus the post-block faults of every exponent qubit
    /// (windowed exponentiation).
    fn program_w(
        &self,
        nq: usize,
        i: usize,
        faults: &[Fault],
    ) -> (Vec<Seg>, PostFaults, Vec<PostFaults>) {
        let nqu = nq as u32;
        let (one, sign) = (nqu, nqu + 1);
        let rc = &self.rounds[i];
        let mut gf: Vec<(u32, u8, Pauli)> = Vec::new();
        let mut post = PostFaults::default();
        let mut wpost = vec![PostFaults::default(); self.eq.len()];
        let mut ops = Vec::with_capacity(rc.ops.len() + rc.ops.len() / 4 + 8);
        let mut segs: Vec<Seg> = Vec::new();
        let push_pauli = |ops: &mut Vec<[u32; 3]>, q: u32, p: Pauli| {
            if matches!(p, Pauli::Z | Pauli::Y) {
                ops.push([sign, q, one]);
            }
            if matches!(p, Pauli::X | Pauli::Y) {
                ops.push([q, one, one]);
            }
        };
        let mut prep = false;
        let mut h1 = None;
        for f in faults {
            assert_eq!(f.round as usize, i);
            match f.site {
                Site::Prep => prep = true,
                Site::H1 => h1 = Some(f.pauli),
                Site::Gate { gate, slot } => gf.push((gate, slot, f.pauli)),
                Site::Phase => post.phase = Some(f.pauli),
                Site::H2 => post.h2 = Some(f.pauli),
                Site::Meas => post.meas = true,
            }
        }
        if prep {
            push_pauli(&mut ops, 0, Pauli::Z); // X before H = Z after H
        }
        if let Some(p) = h1 {
            push_pauli(&mut ops, 0, p);
        }
        gf.sort_unstable_by_key(|&(g, s, _)| (g, s));
        let mut fi = 0;
        for (gi, op) in rc.ops.iter().enumerate() {
            let mut here: [bool; 2] = [false; 2];
            let start = fi;
            while fi < gf.len() && gf[fi].0 as usize == gi {
                fi += 1;
            }
            let fs = &gf[start..fi];
            match *op {
                NOp::G(g) => {
                    let q = |x: usize| x as u32;
                    match g {
                        Gate::X(t) => ops.push([q(t), one, one]),
                        Gate::Cnot(c, t) => ops.push([q(t), q(c), one]),
                        Gate::Ccx(a, b, t) => ops.push([q(t), q(a), q(b)]),
                        Gate::Swap(a, b) => {
                            let (a, b) = (q(a), q(b));
                            ops.push([a, b, one]);
                            ops.push([b, a, one]);
                            ops.push([a, b, one]);
                        }
                        Gate::Z(t) => ops.push([sign, q(t), one]),
                        Gate::Cz(a, b) => ops.push([sign, q(a), q(b)]),
                        g => panic!("{g:?} is not a supported gate"),
                    }
                    let qs = g.qubits();
                    for &(_, slot, p) in fs {
                        push_pauli(&mut ops, qs[slot as usize] as u32, p);
                    }
                }
                NOp::MeasX(t, m) => {
                    for &(_, slot, _) in fs {
                        here[slot as usize] = true;
                    }
                    // slot 0: readout flip -> projection = recorded ^ 1
                    if m ^ here[0] {
                        ops.push([sign, t, one]);
                    }
                    ops.push([t, t, t]);
                    if here[1] {
                        ops.push([t, one, one]);
                    }
                }
                NOp::Ctrl(k, j) => {
                    let q = self.eq[j as usize];
                    for &(_, _, p) in fs {
                        match k {
                            0 => push_pauli(&mut ops, q, Pauli::Z),
                            1 => push_pauli(&mut ops, q, p),
                            2 => wpost[j as usize].phase = Some(p),
                            3 => wpost[j as usize].h2 = Some(p),
                            _ => wpost[j as usize].meas = true,
                        }
                    }
                }
                NOp::ResetZ(g) => {
                    let grp = rc.groups[g as usize].clone();
                    let mut flips = Vec::new();
                    for &(_, slot, _) in fs {
                        flips.push(grp[slot as usize]);
                    }
                    segs.push(Seg {
                        ops: std::mem::take(&mut ops),
                        reset: Some((grp, flips)),
                    });
                }
            }
        }
        assert_eq!(fi, gf.len(), "gate fault beyond the round's ops");
        segs.push(Seg { ops, reset: None });
        assert!(segs
            .iter()
            .flat_map(|s| &s.ops)
            .flatten()
            .all(|&x| x <= sign));
        (segs, post, wpost)
    }
}

fn has_prep_meas(kind: NoiseKind) -> bool {
    !matches!(kind, NoiseKind::PhaseFlip)
}

fn sample_pauli<R: Rng + ?Sized>(kind: NoiseKind, rng: &mut R) -> Pauli {
    match kind {
        NoiseKind::Depolarizing => [Pauli::X, Pauli::Y, Pauli::Z][rng.random_range(0..3)],
        NoiseKind::BitFlip => Pauli::X,
        NoiseKind::PhaseFlip => Pauli::Z,
    }
}

/// A straight-line piece of a round's program, optionally followed by a
/// Z-basis reset of a qubit group (and the reset flips of that reset).
#[derive(Clone, Debug)]
struct Seg {
    ops: Vec<[u32; 3]>,
    reset: Option<(Vec<u32>, Vec<u32>)>,
}

#[derive(Clone, Copy, Debug, Default)]
struct PostFaults {
    phase: Option<Pauli>,
    h2: Option<Pauli>,
    meas: bool,
}

// ---------------------------------------------------------------------------
// Building and tagging the rounds
// ---------------------------------------------------------------------------

/// How the rounds of a [`GenCircuit`] are produced.
#[derive(Clone, Debug)]
enum Source {
    /// One fixed op stream (reversible oracles), shared by all trajectories.
    Fixed(Arc<Resolved>),
    /// Measurement-based oracle: resolved per trajectory from its recorded
    /// outcomes.
    Mbu(MbuOpts, usize),
    /// Gidney–Ekerå windowed exponentiation (`shor_ge`, exact arithmetic):
    /// one engine round per exponent window of `we` rounds.
    Ge(crate::shor::ge::GeOpts),
}

/// Where the design variants put Z-basis measure-and-resets
/// ([`NOp::ResetZ`]) of qubits that should be clean.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResetMode {
    /// No resets (the oracles as built).
    None,
    /// At the end of every round: every ancilla (all qubits above the work
    /// register).
    Round,
    /// After every window's unlookup: the window-clean ancillas (lookup
    /// register, AND ancillas, constant register, carry, flag, top bit of
    /// the accumulator, Gidney carries); plus the end-of-round reset.
    Window,
}

/// Inserts the [`ResetMode`] resets into a tagged round.
fn insert_resets(
    ops: Vec<NOp>,
    tags: Vec<u8>,
    mode: ResetMode,
    inst: &Instance,
    nq: usize,
) -> (Vec<NOp>, Vec<u8>, Vec<Vec<u32>>) {
    if mode == ResetMode::None {
        return (ops, tags, Vec::new());
    }
    let m = inst.m;
    let all: Vec<u32> = (m as u32 + 1..nq as u32).collect();
    let w = match inst.oracle {
        Oracle::Windowed(w)
        | Oracle::WindowedOpt(w)
        | Oracle::WindowedMbu(w)
        | Oracle::WindowedMbuLookup(w) => w,
        o => panic!("resets need a windowed layout, not {o:?}"),
    };
    let lay = WindowLayout::new(m, w);
    let mut win: Vec<u32> = lay
        .l
        .iter()
        .chain(&lay.and)
        .chain(&lay.k)
        .chain([&lay.c0, &lay.t, &lay.b[m]])
        .map(|&q| q as u32)
        .collect();
    // Gidney carries (full-MBU layout) sit above the windowed layout
    win.extend(lay.num_qubits() as u32..nq as u32);
    win.sort_unstable();
    let groups = vec![win, all];
    let mut o = Vec::with_capacity(ops.len() + 64);
    let mut t = Vec::with_capacity(ops.len() + 64);
    let n = ops.len();
    for j in 0..n {
        o.push(ops[j]);
        t.push(tags[j]);
        let is_ul = tags[j] & tag::BLOCK == tag::UNLOOKUP;
        let next_ul = j + 1 < n && tags[j + 1] & tag::BLOCK == tag::UNLOOKUP;
        if mode == ResetMode::Window && is_ul && !next_ul && j + 1 < n {
            o.push(NOp::ResetZ(0));
            t.push(tag::RESET | tag::GATE);
        }
    }
    o.push(NOp::ResetZ(1));
    t.push(tag::RESET | tag::GATE);
    (o, t, groups)
}

/// A noisy semiclassical Shor circuit for any supported oracle.
#[derive(Clone, Debug)]
pub struct GenCircuit {
    /// The order-finding instance.
    pub inst: Instance,
    /// The noise channel.
    pub kind: NoiseKind,
    /// Qubits (control = qubit 0).
    pub nq: usize,
    src: Source,
    /// Design-variant resets inserted into the rounds.
    pub resets: ResetMode,
    /// Exponent window (1 = one recycled control per round).
    pub we: usize,
}

impl GenCircuit {
    /// [`GenCircuit::new`] with the design variant's resets.
    pub fn with_resets(inst: &Instance, kind: NoiseKind, resets: ResetMode) -> Self {
        let mut gc = Self::new(inst, kind);
        gc.resets = resets;
        if let Source::Fixed(r) = &gc.src {
            let rounds = r
                .rounds
                .iter()
                .map(|rd| {
                    let (o, t, g) =
                        insert_resets(rd.ops.clone(), rd.tags.clone(), resets, inst, gc.nq);
                    Round::with_groups(o, t, g, kind)
                })
                .collect();
            gc.src = Source::Fixed(Arc::new(Resolved::new(rounds, kind)));
        }
        gc
    }

    /// The circuit of `inst` without extra resets. Supports the `Ripple`,
    /// `Windowed`, `WindowedOpt`, `WindowedMbu` and `WindowedMbuLookup` oracles
    /// (panics otherwise) with at most 193 qubits.
    pub fn new(inst: &Instance, kind: NoiseKind) -> Self {
        let nq = inst.qubits();
        assert!(nq <= 193, "keys hold at most 192 non-control qubits");
        let src = match inst.oracle {
            Oracle::Ripple | Oracle::Windowed(_) | Oracle::WindowedOpt(_) => {
                let rounds = (0..inst.t)
                    .map(|i| {
                        let mult = inst.mults[inst.t - 1 - i];
                        let (ops, tags) = plain_round(inst, mult);
                        Round::new(ops, tags, kind)
                    })
                    .collect();
                Source::Fixed(Arc::new(Resolved::new(rounds, kind)))
            }
            Oracle::WindowedMbu(w) => Source::Mbu(MbuOpts::ALL, w),
            Oracle::WindowedMbuLookup(w) => Source::Mbu(MbuOpts::LOOKUPS, w),
            o => panic!("the noisy engine needs a gate-level permutation oracle, not {o:?}"),
        };
        Self {
            inst: inst.clone(),
            kind,
            nq,
            src,
            resets: ResetMode::None,
            we: 1,
        }
    }

    /// Windowed exponentiation (research/shor/ge-shor.md): `we` exponent qubits
    /// per window (`we` divides `t = 2n`), `wm`-bit multiplicand windows,
    /// all MBU constructions, exact modular arithmetic.
    pub fn new_ge(n_mod: u64, a: u64, we: usize, wm: usize, kind: NoiseKind) -> Self {
        use crate::shor::ge::{GeLayout, GeOpts};
        let inst = Instance::new(n_mod, a, Oracle::WindowedMbu(wm));
        assert!(
            we >= 1 && inst.t.is_multiple_of(we),
            "the window must divide t = 2n"
        );
        let o = GeOpts {
            we,
            wm,
            mbu: MbuOpts::ALL,
            coset: 0,
        };
        let lay = GeLayout::new(inst.m, &o);
        assert!(lay.nq <= 193);
        Self {
            inst,
            kind,
            nq: lay.nq,
            src: Source::Ge(o),
            resets: ResetMode::None,
            we,
        }
    }

    /// A circuit with explicitly given rounds (custom op streams, e.g. for
    /// tests of the measurement semantics); `nq` qubits, control = qubit 0.
    pub fn with_rounds(inst: &Instance, kind: NoiseKind, nq: usize, rounds: Vec<Round>) -> Self {
        assert_eq!(rounds.len(), inst.t);
        Self {
            inst: inst.clone(),
            kind,
            nq,
            src: Source::Fixed(Arc::new(Resolved::new(rounds, kind))),
            resets: ResetMode::None,
            we: 1,
        }
    }

    /// Whether the op stream depends on recorded measurement outcomes.
    pub fn is_mbu(&self) -> bool {
        matches!(self.src, Source::Mbu(..) | Source::Ge(..))
    }

    /// The trajectory's resolved circuit, with every recorded X-basis
    /// outcome drawn from `bit` (uniform for the importance-sampled
    /// engine).
    pub fn resolve(&self, bit: &mut dyn FnMut() -> bool) -> Arc<Resolved> {
        match &self.src {
            Source::Fixed(r) => r.clone(),
            Source::Mbu(o, w) => {
                let inst = &self.inst;
                let rounds = (0..inst.t)
                    .map(|i| {
                        let mult = inst.mults[inst.t - 1 - i];
                        let (ops, tags) = mbu_round(inst, *w, o, mult, bit);
                        let (ops, tags, g) = insert_resets(ops, tags, self.resets, inst, self.nq);
                        Round::with_groups(ops, tags, g, self.kind)
                    })
                    .collect();
                Arc::new(Resolved::new(rounds, self.kind))
            }
            Source::Ge(o) => {
                let inst = &self.inst;
                let lay = crate::shor::ge::GeLayout::new(inst.m, o);
                let we = o.we;
                let rounds = (0..inst.t / we)
                    .map(|k| {
                        let i0 = k * we;
                        let g = inst.mults[inst.t - i0 - we];
                        let (body, btags) = ge_round(&lay, o, inst.n_mod, g, bit);
                        let mut ops = Vec::with_capacity(body.len() + 5 * we);
                        let mut tags = Vec::with_capacity(body.len() + 5 * we);
                        for j in 0..we as u8 {
                            ops.extend([NOp::Ctrl(0, j), NOp::Ctrl(1, j)]);
                        }
                        ops.extend(body);
                        tags.resize(2 * we, tag::CTRL | tag::GATE);
                        tags.extend(btags);
                        for jj in 0..we {
                            let j = (we - 1 - jj) as u8;
                            ops.extend([NOp::Ctrl(2, j), NOp::Ctrl(3, j), NOp::Ctrl(4, j)]);
                            tags.extend([tag::CTRL | tag::GATE; 3]);
                        }
                        Round::new(ops, tags, self.kind)
                    })
                    .collect();
                let eq = lay.e.iter().map(|&q| q as u32).collect();
                Arc::new(Resolved::new_windowed(rounds, self.kind, eq))
            }
        }
    }

    /// Resolved circuit with uniformly random recorded outcomes from `rng`.
    pub fn resolve_rng<R: Rng + ?Sized>(&self, rng: &mut R) -> Arc<Resolved> {
        self.resolve(&mut || rng.random::<bool>())
    }
}

/// Role of a qubit in the windowed / MBU layouts.
pub fn role(lay: &WindowLayout, q: usize) -> &'static str {
    if q == lay.ctrl {
        "ctrl"
    } else if lay.x.contains(&q) {
        "x"
    } else if lay.b.contains(&q) {
        "b"
    } else if lay.l.contains(&q) {
        "lookup"
    } else if lay.k.contains(&q) {
        "const"
    } else if q == lay.c0 {
        "carry"
    } else if q == lay.t {
        "flag"
    } else if lay.and.contains(&q) {
        "and"
    } else {
        "cy"
    }
}

/// Modular-adder part of each op of one contiguous modular-adder run, from
/// its position relative to the gates on the constant register `K`.
fn modadd_parts(ops: &[NOp], k: &[usize], inv: bool) -> Vec<u8> {
    let touches_k = |op: &NOp| match op {
        NOp::G(g) => g.qubits().iter().any(|q| k.contains(q)),
        NOp::MeasX(q, _) => k.contains(&(*q as usize)),
        NOp::ResetZ(_) | NOp::Ctrl(..) => false,
    };
    let first = ops.iter().position(touches_k);
    let last = ops.iter().rposition(touches_k);
    let (head, tail) = if inv {
        (tag::CMP, tag::ADD)
    } else {
        (tag::ADD, tag::CMP)
    };
    (0..ops.len())
        .map(|j| match (first, last) {
            (Some(f), Some(_)) if j < f => head,
            (Some(_), Some(l)) if j > l => tail,
            (Some(_), Some(_)) => tag::REDUCE,
            _ => tag::GATE,
        })
        .collect()
}

/// The fixed op stream of a reversible oracle, tagged.
fn plain_round(inst: &Instance, mult: u64) -> (Vec<NOp>, Vec<u8>) {
    match inst.oracle {
        Oracle::WindowedOpt(w) => {
            let lay = WindowLayout::new(inst.m, w);
            let (gates, tags) = windowed_opt_tagged(&lay, mult, inst.n_mod);
            (gates.into_iter().map(NOp::G).collect(), tags)
        }
        Oracle::Windowed(w) => {
            let lay = WindowLayout::new(inst.m, w);
            let (c, _) = super::sliced::oracle_block(inst, mult);
            let gates: Vec<Gate> = c.gates().copied().collect();
            let tags = role_tags(&lay, &gates);
            (gates.into_iter().map(NOp::G).collect(), tags)
        }
        _ => {
            let (c, _) = super::sliced::oracle_block(inst, mult);
            let gates: Vec<NOp> = c.gates().map(|g| NOp::G(*g)).collect();
            let n = gates.len();
            (gates, vec![tag::OTHER | tag::GATE; n])
        }
    }
}

/// Role-based tags for the round-4 windowed oracle: swap gates (target `x`,
/// or `CCX(ctrl, x, b)`), lookup gates (touch an AND ancilla or the
/// control; fan-out if the target is in `L` / `b`), modular adder (the
/// rest, parts by the `K` rule per run). Lookups and unlookups of adjacent
/// windows are contiguous here, so both are tagged `lookup`.
fn role_tags(lay: &WindowLayout, gates: &[Gate]) -> Vec<u8> {
    let half_at = gates
        .iter()
        .position(|g| matches!(*g, Gate::Ccx(c, x, b) if c == lay.ctrl && lay.x.contains(&x) && lay.b.contains(&b)))
        .unwrap_or(gates.len());
    let mut tags = vec![0u8; gates.len()];
    let mut j = 0;
    while j < gates.len() {
        let g = &gates[j];
        let qs = g.qubits();
        let t = *qs.last().unwrap();
        let inv = if j > half_at { tag::INV } else { 0 };
        let is_swap = lay.x.contains(&t)
            || matches!(*g, Gate::Ccx(c, x, b) if c == lay.ctrl && lay.x.contains(&x) && lay.b.contains(&b));
        if is_swap {
            tags[j] = tag::SWAP | tag::GATE;
            j += 1;
            continue;
        }
        if qs.iter().any(|q| lay.and.contains(q) || *q == lay.ctrl) {
            let part = if lay.l.contains(&t) || lay.b.contains(&t) {
                tag::FANOUT
            } else {
                tag::UNARY
            };
            tags[j] = tag::LOOKUP | part | inv;
            j += 1;
            continue;
        }
        // a modular-adder run
        let s = j;
        while j < gates.len() {
            let g = &gates[j];
            let qs = g.qubits();
            let t = *qs.last().unwrap();
            let sw = lay.x.contains(&t)
                || matches!(*g, Gate::Ccx(c, x, b) if c == lay.ctrl && lay.x.contains(&x) && lay.b.contains(&b));
            if sw || qs.iter().any(|q| lay.and.contains(q) || *q == lay.ctrl) {
                break;
            }
            j += 1;
        }
        let run: Vec<NOp> = gates[s..j].iter().map(|g| NOp::G(*g)).collect();
        let inv_run = s > half_at;
        for (o, p) in modadd_parts(&run, &lay.k, inv_run).into_iter().enumerate() {
            tags[s + o] = tag::MODADD | p | if inv_run { tag::INV } else { 0 };
        }
    }
    tags
}

/// The `windowed-opt` controlled-`U_a` (`Opts::ALL`, block passes) emitted
/// window by window with exact block tags; asserted equal, gate for gate,
/// to [`crate::shor::superopt::controlled_ua`].
pub fn windowed_opt_tagged(lay: &WindowLayout, a: u64, n_mod: u64) -> (Vec<Gate>, Vec<u8>) {
    use crate::circuit::Circuit;
    use crate::shor::superopt::{self as so, Opts};
    let o = Opts::ALL;
    assert!(o.block_passes && o.direct_first && !o.window_dp);
    let block = so::modadd_block(lay, n_mod, &o);
    let blen = block.ops.len();
    let madd = so::madd_mask(lay);
    let lk_part = |g: &Gate, out: &[usize]| {
        let t = *g.qubits().last().unwrap();
        if out.contains(&t) {
            tag::FANOUT
        } else {
            tag::UNARY
        }
    };
    // forward-orientation tags of one multiply-add
    let cm = |a: u64| -> (Vec<Gate>, Vec<u8>) {
        let mut gs = Vec::new();
        let mut ts = Vec::new();
        let mut start = 0;
        let mut base = a % n_mod;
        for w in so::window_sizes(lay, a, n_mod, &o) {
            let mut c = Circuit::new(lay.num_qubits());
            so::emit_window(&mut c, lay, n_mod, &o, start, w, base, &madd, Some(&block));
            let g: Vec<Gate> = c.gates().copied().collect();
            if start == 0 {
                for x in &g {
                    ts.push(tag::LOOKUP | lk_part(x, &lay.b));
                }
            } else {
                let keep = (g.len() - blen) / 2;
                assert_eq!(2 * keep + blen, g.len());
                for x in &g[..keep] {
                    ts.push(tag::LOOKUP | lk_part(x, &lay.l));
                }
                let run: Vec<NOp> = g[keep..keep + blen].iter().map(|x| NOp::G(*x)).collect();
                for p in modadd_parts(&run, &lay.k, false) {
                    ts.push(tag::MODADD | p);
                }
                for x in &g[keep + blen..] {
                    ts.push(tag::UNLOOKUP | lk_part(x, &lay.l));
                }
            }
            gs.extend(g);
            for _ in 0..w {
                base = (u128::from(base) * 2 % u128::from(n_mod)) as u64;
            }
            start += w;
        }
        (gs, ts)
    };
    let inv = crate::shor::mod_inverse(a, n_mod);
    let (mut gates, mut tags) = cm(a);
    for i in 0..lay.n {
        let (x, b) = (lay.x[i], lay.b[i]);
        gates.extend([
            Gate::Cnot(b, x),
            Gate::Ccx(lay.ctrl, x, b),
            Gate::Cnot(b, x),
        ]);
        tags.extend([tag::SWAP | tag::GATE; 3]);
    }
    let (g2, t2) = cm(inv);
    for (g, t) in g2.into_iter().rev().zip(t2.into_iter().rev()) {
        // reversed: the uncompute of the forward lookup is the compute here
        let b = match t & tag::BLOCK {
            tag::LOOKUP => tag::UNLOOKUP,
            tag::UNLOOKUP => tag::LOOKUP,
            x => x,
        };
        gates.push(g);
        tags.push(b | (t & tag::PART) | tag::INV);
    }
    let reference = so::controlled_ua(lay, a, n_mod, &o);
    let rg: Vec<Gate> = reference.gates().copied().collect();
    assert_eq!(
        rg, gates,
        "tagged windowed-opt emission differs from controlled_ua"
    );
    (gates, tags)
}

/// Context of the tagged resolver.
#[derive(Clone, Copy)]
struct Ctx<'a> {
    /// Block bits (`LOOKUP` / `UNLOOKUP` / `MODADD` / `SWAP`) | `INV`.
    base: u8,
    /// Lookup output register (fan-out vs unary).
    out: Option<&'a [usize]>,
    /// Force this part.
    part: Option<u8>,
}

/// Tagged mirror of [`shor_mbu::resolve`] (same op stream for the same
/// outcome bits, minus `GlobalNeg`, which is a global phase); modular-adder
/// gates get the provisional part `GATE` (partitioned later).
fn resolve_tagged(
    ops: &[LOp],
    bit: &mut dyn FnMut() -> bool,
    out: &mut Vec<NOp>,
    tags: &mut Vec<u8>,
    cx: Ctx,
) {
    let part_of = |t: usize, cx: &Ctx| -> u8 {
        if let Some(p) = cx.part {
            return p;
        }
        match cx.out {
            Some(o) if o.contains(&t) => tag::FANOUT,
            Some(_) => tag::UNARY,
            None => tag::GATE,
        }
    };
    for op in ops {
        match op {
            LOp::G(g) => {
                let t = *g.qubits().last().unwrap();
                out.push(NOp::G(*g));
                tags.push(cx.base | part_of(t, &cx));
            }
            LOp::GlobalNeg => {}
            LOp::And(a, b, t) => {
                out.push(NOp::G(Gate::Ccx(*a, *b, *t)));
                tags.push(cx.base | part_of(*t, &cx));
            }
            LOp::UnAnd(a, b, t) => {
                let m = bit();
                let p = part_of(*t, &cx);
                out.push(NOp::MeasX(*t as u32, m));
                tags.push(cx.base | p);
                if m {
                    out.push(NOp::G(Gate::Cz(*a, *b)));
                    tags.push(cx.base | p);
                }
            }
            LOp::Lookup(s) => {
                let c = Ctx {
                    base: cx.base,
                    out: Some(&s.out),
                    part: cx.part,
                };
                resolve_tagged(&shor_mbu::lookup_ops(s), bit, out, tags, c);
            }
            LOp::FlagCompute(f) => {
                let c = Ctx {
                    part: Some(cx.part.unwrap_or(tag::CMP)),
                    ..cx
                };
                resolve_tagged(&f.compute, bit, out, tags, c);
            }
            LOp::FlagUncompute(f) => {
                let m = bit();
                out.push(NOp::MeasX(f.t as u32, m));
                tags.push(cx.base | cx.part.unwrap_or(tag::FLAGMEAS));
                if m {
                    let c = Ctx {
                        part: Some(cx.part.unwrap_or(tag::FLAGFIX)),
                        ..cx
                    };
                    resolve_tagged(&f.fix, bit, out, tags, c);
                }
            }
            LOp::Unlookup(s) => {
                if !s.meas_unlookup {
                    let c = Ctx {
                        base: cx.base,
                        out: Some(&s.out),
                        part: cx.part,
                    };
                    resolve_tagged(&shor_mbu::lookup_ops(s), bit, out, tags, c);
                    continue;
                }
                let mut mask = 0u64;
                for (j, &q) in s.out.iter().enumerate() {
                    let m = bit();
                    out.push(NOp::MeasX(q as u32, m));
                    tags.push(cx.base | cx.part.unwrap_or(tag::MEAS));
                    if m {
                        mask |= 1 << j;
                    }
                }
                let g: Vec<bool> = s
                    .table
                    .iter()
                    .map(|&t| (t & mask).count_ones() & 1 == 1)
                    .collect();
                if g.iter().any(|&b| b) {
                    let scratch: Vec<usize> = s.and.iter().chain(&s.out).copied().collect();
                    let c = Ctx {
                        base: cx.base,
                        out: None,
                        part: Some(cx.part.unwrap_or(tag::FIX)),
                    };
                    resolve_tagged(
                        &shor_mbu::phase_table(s.ctrl, &s.addr, &g, &scratch),
                        bit,
                        out,
                        tags,
                        c,
                    );
                }
            }
        }
    }
}

/// One MBU round, resolved and tagged.
fn mbu_round(
    inst: &Instance,
    w: usize,
    o: &MbuOpts,
    mult: u64,
    bit: &mut dyn FnMut() -> bool,
) -> (Vec<NOp>, Vec<u8>) {
    let lay = MbuLayout::new(inst.m, w, o);
    let win = &lay.win;
    let n_mod = inst.n_mod;
    let lops = shor_mbu::controlled_ua_ops(&lay, mult, n_mod, o);
    let modadd = shor_mbu::modadd_ops(&lay, n_mod, o);
    let len1 = shor_mbu::cmult(&lay, mult, n_mod, o, &modadd).len();
    let swap_end = len1 + 3 * win.n;
    tag_top(&lops, len1, swap_end, &win.k, bit)
}

/// One window of the Gidney–Ekerå windowed exponentiation, resolved and
/// tagged (multiply-add by `g^e`, swap, inverse multiply-add by `g^{−e}`).
fn ge_round(
    lay: &crate::shor::ge::GeLayout,
    o: &crate::shor::ge::GeOpts,
    n_mod: u64,
    g: u64,
    bit: &mut dyn FnMut() -> bool,
) -> (Vec<NOp>, Vec<u8>) {
    let ml = lay.mbu.as_ref().expect("exact arithmetic");
    let lops = crate::shor::ge::window_ops(lay, g, n_mod, o);
    let modadd = shor_mbu::modadd_ops(ml, n_mod, &o.mbu);
    let len1 = crate::shor::ge::emult(lay, g, n_mod, o, &modadd).len();
    let swap_end = len1 + 3 * lay.nr;
    tag_top(&lops, len1, swap_end, &ml.win.k, bit)
}

/// Resolves and tags a controlled-`U` / window block whose top-level ops are
/// `[multiply-add (lookups, modular adders, unlookups)][swap: len1..swap_end]
/// [inverse multiply-add]`.
fn tag_top(
    lops: &[LOp],
    len1: usize,
    swap_end: usize,
    k: &[usize],
    bit: &mut dyn FnMut() -> bool,
) -> (Vec<NOp>, Vec<u8>) {
    assert!(lops.len() >= swap_end);
    let mut ops = Vec::new();
    let mut tags = Vec::new();
    let mut j = 0;
    while j < lops.len() {
        let inv = if j >= swap_end { tag::INV } else { 0 };
        if (len1..swap_end).contains(&j) {
            let c = Ctx {
                base: tag::SWAP,
                out: None,
                part: Some(tag::GATE),
            };
            resolve_tagged(&lops[j..j + 1], bit, &mut ops, &mut tags, c);
            j += 1;
            continue;
        }
        match &lops[j] {
            LOp::Lookup(_) => {
                let c = Ctx {
                    base: tag::LOOKUP | inv,
                    out: None,
                    part: None,
                };
                resolve_tagged(&lops[j..j + 1], bit, &mut ops, &mut tags, c);
                j += 1;
            }
            LOp::Unlookup(_) => {
                let c = Ctx {
                    base: tag::UNLOOKUP | inv,
                    out: None,
                    part: None,
                };
                resolve_tagged(&lops[j..j + 1], bit, &mut ops, &mut tags, c);
                j += 1;
            }
            _ => {
                // a modular-adder run (up to the next lookup / the swap)
                let s = j;
                while j < lops.len()
                    && !matches!(lops[j], LOp::Lookup(_) | LOp::Unlookup(_))
                    && !(len1..swap_end).contains(&j)
                    && (j < swap_end || s >= swap_end)
                {
                    j += 1;
                }
                let o0 = ops.len();
                let c = Ctx {
                    base: tag::MODADD | inv,
                    out: None,
                    part: None,
                };
                resolve_tagged(&lops[s..j], bit, &mut ops, &mut tags, c);
                let parts = modadd_parts(&ops[o0..], k, inv != 0);
                for (t, p) in tags[o0..].iter_mut().zip(parts) {
                    if *t & tag::PART == tag::GATE {
                        *t = (*t & !tag::PART) | p;
                    }
                }
            }
        }
    }
    (ops, tags)
}

// ---------------------------------------------------------------------------
// The state
// ---------------------------------------------------------------------------

type OutBranch<K, T> = (K, Complex<T>, bool);

fn eval_half<const L: usize, K: Key, T: Real>(
    ops: &[[u32; 3]],
    nq: usize,
    keys: &[K],
    amps: &[Complex<T>],
    ctrls: &[bool],
    out: &mut [OutBranch<K, T>],
) {
    let b = 64 * L;
    let sign = nq + 1;
    keys.par_chunks(b)
        .zip(amps.par_chunks(b))
        .zip(ctrls.par_chunks(b))
        .zip(out.par_chunks_mut(b))
        .for_each_init(
            || vec![[0u64; L]; nq + 2],
            |w, (((ks, am), cs), os)| {
                for x in w.iter_mut() {
                    *x = [0; L];
                }
                w[nq] = [u64::MAX; L];
                let mut blk = [0u64; 64];
                for (l, (kc, cc)) in ks.chunks(64).zip(cs.chunks(64)).enumerate() {
                    let mut cw = 0u64;
                    for (j, &c) in cc.iter().enumerate() {
                        cw |= u64::from(c) << j;
                    }
                    w[0][l] = cw;
                    for half in 0..K::WORDS {
                        let lo = 1 + 64 * half;
                        if lo >= nq {
                            break;
                        }
                        for (j, k) in kc.iter().enumerate() {
                            blk[j] = k.word(half);
                        }
                        blk[kc.len()..].fill(0);
                        transpose64(&mut blk);
                        for (bit, &v) in blk.iter().enumerate() {
                            let q = lo + bit;
                            if q >= nq {
                                break;
                            }
                            w[q][l] = v;
                        }
                    }
                }
                // SAFETY: every index of `ops` is <= nq + 1 < w.len()
                // (checked in `Resolved::program`).
                unsafe { eval_raw_unchecked::<L>(ops, w) };
                for (l, ((kc, ac), oc)) in ks
                    .chunks(64)
                    .zip(am.chunks(64))
                    .zip(os.chunks_mut(64))
                    .enumerate()
                {
                    let mut kout = [K::default(); 64];
                    for half in 0..K::WORDS {
                        let lo = 1 + 64 * half;
                        if lo >= nq {
                            break;
                        }
                        for (bit, v) in blk.iter_mut().enumerate() {
                            let q = lo + bit;
                            *v = if q < nq { w[q][l] } else { 0 };
                        }
                        transpose64(&mut blk);
                        for (j, ko) in kout.iter_mut().enumerate().take(kc.len()) {
                            ko.set_word(half, blk[j]);
                        }
                    }
                    let (cw, sw) = (w[0][l], w[sign][l]);
                    for j in 0..kc.len() {
                        let a = ac[j];
                        let a = if (sw >> j) & 1 == 1 { -a } else { a };
                        oc[j] = (kout[j], a, (cw >> j) & 1 == 1);
                    }
                }
            },
        );
}

fn eval_dispatch<K: Key, T: Real>(
    ops: &[[u32; 3]],
    nq: usize,
    keys: &[K],
    amps: &[Complex<T>],
    ctrls: &[bool],
    out: &mut [OutBranch<K, T>],
) {
    match keys.len() {
        0..=64 => eval_half::<1, K, T>(ops, nq, keys, amps, ctrls, out),
        65..=512 => eval_half::<4, K, T>(ops, nq, keys, amps, ctrls, out),
        _ => eval_half::<16, K, T>(ops, nq, keys, amps, ctrls, out),
    }
}

/// `key` with only the bits of `qubits` (key bit = qubit − 1).
fn group_bits<K: Key>(key: &K, mask: &K) -> K {
    let mut o = K::default();
    for i in 0..K::WORDS {
        o.set_word(i, key.word(i) & mask.word(i));
    }
    o
}

fn group_mask<K: Key>(qubits: &[u32]) -> K {
    let mut m = K::default();
    for &q in qubits {
        let j = q as usize - 1;
        m.set_word(j / 64, m.word(j / 64) | (1u64 << (j % 64)));
    }
    m
}

/// Z-basis measure-and-reset of the group `mask` on a branch list (both
/// control values): merges duplicate branches, picks a group value with
/// `choose(probabilities)` → `(index, weight factor)`, keeps its branches
/// (norm preserved), clears the group bits. Returns the weight factor.
fn reset_group<K: Key, T: Real>(
    v: &mut Vec<OutBranch<K, T>>,
    mask: &K,
    choose: &mut dyn FnMut(&[f64]) -> (usize, f64),
) -> f64 {
    let z = K::default();
    if !v.par_iter().any(|e| group_bits(&e.0, mask) != z) {
        return 1.0;
    }
    v.par_sort_unstable_by_key(|e| (e.0, e.2));
    let mut m: Vec<OutBranch<K, T>> = Vec::with_capacity(v.len());
    for e in v.drain(..) {
        match m.last_mut() {
            Some(l) if l.0 == e.0 && l.2 == e.2 => l.1 = l.1 + e.1,
            _ => m.push(e),
        }
    }
    let mut mass: std::collections::BTreeMap<K, f64> = Default::default();
    for e in &m {
        *mass.entry(group_bits(&e.0, mask)).or_insert(0.0) += c64(e.1).norm_sqr();
    }
    let total: f64 = mass.values().sum();
    let vals: Vec<K> = mass.keys().copied().collect();
    let probs: Vec<f64> = mass.values().map(|x| x / total).collect();
    let (idx, wf) = choose(&probs);
    let pick = vals[idx];
    let scale = T::from_f64((1.0 / probs[idx]).sqrt());
    let mut inv = K::default();
    for i in 0..K::WORDS {
        inv.set_word(i, !mask.word(i));
    }
    *v = m
        .into_iter()
        .filter(|e| group_bits(&e.0, mask) == pick)
        .map(|e| (group_bits(&e.0, &inv), e.1 * scale, e.2))
        .collect();
    wf
}

fn cvt<T: Real>(z: Complex64) -> Complex<T> {
    Complex::new(T::from_f64(z.re), T::from_f64(z.im))
}
fn c64<T: Real>(z: Complex<T>) -> Complex64 {
    Complex64::new(z.re.to_f64(), z.im.to_f64())
}

/// Sorts by key and sums the amplitudes of equal keys; returns the number
/// of merged duplicates.
fn sort_merge<K: Key, T: Real>(v: &mut Vec<OutBranch<K, T>>) -> usize {
    v.par_sort_unstable_by_key(|e| e.0);
    let dup = v.par_windows(2).filter(|w| w[0].0 == w[1].0).count();
    if dup == 0 {
        return 0;
    }
    let mut out: Vec<OutBranch<K, T>> = Vec::with_capacity(v.len() - dup);
    for e in v.drain(..) {
        match out.last_mut() {
            Some(l) if l.0 == e.0 => l.1 = l.1 + e.1,
            _ => out.push(e),
        }
    }
    *v = out;
    dup
}

/// Exact state of one trajectory: branches keyed by every non-control
/// qubit.
#[derive(Clone, Debug)]
pub struct GenState<K: Key, T: Real> {
    keys: Vec<K>,
    amps: Vec<Complex<T>>,
    merged: Vec<(K, Complex<T>, Complex<T>)>,
    p1_raw: f64,
    p0_raw: f64,
    /// Largest support seen (after a merge).
    pub peak: usize,
    /// Gate × branch applications.
    pub work_ops: u128,
    /// First round after which some branch had a non-zero ancilla.
    pub dirty_from: Option<usize>,
    /// Support at the start of every round.
    pub support_trace: Vec<usize>,
    /// `ln` of the importance weight `Π_rounds ‖state‖²`.
    pub log_weight: f64,
    /// Rounds in which branches collided (some X-measured qubit was not a
    /// function of the rest).
    pub collision_rounds: usize,
    /// Work-register width (for the dirty-ancilla check of windowed rounds).
    pub work_bits: usize,
}

impl<K: Key, T: Real> Default for GenState<K, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Key, T: Real> GenState<K, T> {
    /// Control `|0>`, work register `|1>`, ancillas `|0>`.
    pub fn new() -> Self {
        Self {
            keys: vec![K::from_low(1)],
            amps: vec![Complex::new(T::one(), T::zero())],
            merged: Vec::new(),
            p1_raw: 0.0,
            p0_raw: 0.0,
            peak: 1,
            work_ops: 0,
            dirty_from: None,
            support_trace: Vec::new(),
            log_weight: 0.0,
            collision_rounds: 0,
            work_bits: 0,
        }
    }

    /// Number of stored branches (duplicates counted).
    pub fn nnz(&self) -> usize {
        self.keys.len()
    }

    /// The stored branches `(key, amplitude)`.
    pub fn branches(&self) -> impl Iterator<Item = (K, Complex<T>)> + '_ {
        self.keys.iter().copied().zip(self.amps.iter().copied())
    }

    /// Runs round `i` and returns `(P(1), readout flipped, W)` with `W` the
    /// squared norm of the unnormalised post-round state.
    #[allow(clippy::too_many_arguments)]
    pub fn round(
        &mut self,
        res: &Resolved,
        inst: &Instance,
        nq: usize,
        i: usize,
        y_low: u128,
        faults: &[Fault],
        cap: usize,
        choose: &mut dyn FnMut(&[f64]) -> (usize, f64),
    ) -> Result<(f64, bool, f64), Capped> {
        self.support_trace.push(self.keys.len());
        let (segs, post) = res.program(nq, i, faults);
        let s = self.keys.len();
        let mut out: Vec<OutBranch<K, T>> = vec![(K::default(), Complex::zero(), false); 2 * s];
        let mut reset_w = 1.0f64;
        if segs.len() == 1 {
            let ops = &segs[0].ops;
            let (o0, o1) = out.split_at_mut(s);
            eval_dispatch(ops, nq, &self.keys, &self.amps, &vec![false; s], o0);
            eval_dispatch(ops, nq, &self.keys, &self.amps, &vec![true; s], o1);
        } else {
            let mut keys: Vec<K> = self.keys.iter().chain(&self.keys).copied().collect();
            let mut amps: Vec<Complex<T>> = self.amps.iter().chain(&self.amps).copied().collect();
            let mut ctrls: Vec<bool> = (0..2 * s).map(|j| j >= s).collect();
            for (si, seg) in segs.iter().enumerate() {
                out = vec![(K::default(), Complex::zero(), false); keys.len()];
                eval_dispatch(&seg.ops, nq, &keys, &amps, &ctrls, &mut out);
                if let Some((grp, flips)) = &seg.reset {
                    reset_w *= reset_group(&mut out, &group_mask::<K>(grp), choose);
                    if !flips.is_empty() {
                        let fm: K = group_mask(flips);
                        for e in out.iter_mut() {
                            for wi in 0..K::WORDS {
                                e.0.set_word(wi, e.0.word(wi) ^ fm.word(wi));
                            }
                        }
                    }
                }
                if si + 1 < segs.len() {
                    keys = out.iter().map(|e| e.0).collect();
                    amps = out.iter().map(|e| e.1).collect();
                    ctrls = out.iter().map(|e| e.2).collect();
                }
            }
        }
        self.work_ops += 2 * s as u128 * res.rounds[i].ops.len() as u128;
        let (mut a1, mut a0): (Vec<_>, Vec<_>) = out.into_par_iter().partition(|e| e.2);
        let dups = sort_merge(&mut a0) + sort_merge(&mut a1);
        if dups > 0 {
            self.collision_rounds += 1;
        }
        let phi = if y_low != 0 {
            Instance::correction(i, y_low)
        } else {
            0.0
        };
        let ph = Complex64::from_polar(1.0, phi);
        let z = Complex64::zero();
        let mut merged = Vec::with_capacity(a0.len().max(a1.len()));
        let (mut x, mut y) = (0, 0);
        let (mut p1, mut p0) = (0.0f64, 0.0f64);
        while x < a0.len() || y < a1.len() {
            let ka = a0.get(x).map(|e| e.0);
            let kb = a1.get(y).map(|e| e.0);
            let key = match (ka, kb) {
                (Some(a), Some(b)) => a.min(b),
                (Some(a), None) => a,
                (None, Some(b)) => b,
                _ => unreachable!(),
            };
            let (mut u0, mut u1) = (z, z);
            if ka == Some(key) {
                u0 = c64(a0[x].1);
                x += 1;
            }
            if kb == Some(key) {
                u1 = ph * c64(a1[y].1);
                y += 1;
            }
            if let Some(p) = post.phase {
                if matches!(p, Pauli::Z | Pauli::Y) {
                    u1 = -u1;
                }
                if matches!(p, Pauli::X | Pauli::Y) {
                    std::mem::swap(&mut u0, &mut u1);
                }
            }
            let mut o0 = (u0 + u1) * 0.5;
            let mut o1 = (u0 - u1) * 0.5;
            if let Some(p) = post.h2 {
                if matches!(p, Pauli::Z | Pauli::Y) {
                    o1 = -o1;
                }
                if matches!(p, Pauli::X | Pauli::Y) {
                    std::mem::swap(&mut o0, &mut o1);
                }
            }
            p1 += o1.norm_sqr();
            p0 += o0.norm_sqr();
            merged.push((key, cvt::<T>(o0), cvt::<T>(o1)));
        }
        drop(a0);
        drop(a1);
        self.peak = self.peak.max(merged.len());
        let n = inst.m;
        if self.dirty_from.is_none() && merged.iter().any(|e| e.0.any_from(n)) {
            self.dirty_from = Some(i);
        }
        if merged.len() > cap {
            return Err(Capped {
                round: i,
                support: merged.len(),
            });
        }
        let wn = p0 + p1;
        self.keys = Vec::new();
        self.amps = Vec::new();
        self.merged = merged;
        self.p1_raw = p1;
        self.p0_raw = p0;
        let pr1 = if wn > 0.0 { p1 / wn } else { 0.0 };
        Ok((pr1, post.meas, wn * reset_w))
    }

    /// Collapses the control measured after the last round and renormalises.
    pub fn collapse(&mut self, outcome: bool) {
        let p = if outcome { self.p1_raw } else { self.p0_raw };
        assert!(p > 0.0, "cannot collapse onto a zero-probability outcome");
        let k = 1.0 / p.sqrt();
        let mut keys = Vec::with_capacity(self.merged.len());
        let mut amps = Vec::with_capacity(self.merged.len());
        for &(key, o0, o1) in &self.merged {
            let v = c64(if outcome { o1 } else { o0 });
            // exact cancellations from colliding branches leave rounding dust
            if v.norm_sqr() * k * k > 1e-28 {
                keys.push(key);
                amps.push(cvt::<T>(v * k));
            }
        }
        self.merged = Vec::new();
        self.keys = keys;
        self.amps = amps;
    }

    /// Ideal measure-and-reset of every ancilla (every key bit `>= n`).
    pub fn reset_ancillas<R: Rng + ?Sized>(&mut self, n: usize, rng: &mut R) -> bool {
        if self.keys.iter().all(|k| !k.any_from(n)) {
            return false;
        }
        let mut w: std::collections::BTreeMap<K, f64> = std::collections::BTreeMap::new();
        for (k, a) in self.keys.iter().zip(&self.amps) {
            *w.entry(k.split(n).1).or_insert(0.0) += c64(*a).norm_sqr();
        }
        let total: f64 = w.values().sum();
        let mut u = rng.random::<f64>() * total;
        let mut pick = *w.keys().next_back().unwrap();
        for (&anc, &p) in &w {
            if u < p {
                pick = anc;
                break;
            }
            u -= p;
        }
        let norm = T::from_f64((total / w[&pick]).sqrt());
        let mut kv: Vec<(K, Complex<T>)> = self
            .keys
            .iter()
            .zip(&self.amps)
            .filter(|(k, _)| k.split(n).1 == pick)
            .map(|(k, a)| (k.split(n).0, *a * norm))
            .collect();
        kv.sort_unstable_by_key(|e| e.0);
        self.keys = kv.iter().map(|e| e.0).collect();
        self.amps = kv.iter().map(|e| e.1).collect();
        pick != K::default()
    }
}

/// The state of a windowed-exponentiation round after its block: for every
/// rest-of-register key (exponent qubits removed), the `2^we` amplitudes of
/// the exponent register. The exponent qubits are then phase-corrected,
/// H-transformed and measured one at a time ([`WinState::prepare`],
/// [`WinState::collapse`]).
#[derive(Clone, Debug)]
pub struct WinState<K: Key> {
    rest: Vec<K>,
    amps: Vec<Complex64>,
    ne: usize,
    post: Vec<PostFaults>,
    total: f64,
}

impl<K: Key> WinState<K> {
    /// Phase correction `phi`, the post-phase fault, H and the post-H fault
    /// on exponent bit `j`; returns `(P(bit j = 1), readout flipped)`.
    pub fn prepare(&mut self, j: usize, phi: f64) -> (f64, bool) {
        let ne = self.ne;
        let ph = Complex64::from_polar(1.0, phi);
        let pf = self.post[j];
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let m = 1usize << j;
        let fix = |a0: &mut Complex64, a1: &mut Complex64, p: Option<Pauli>| {
            if let Some(p) = p {
                if matches!(p, Pauli::Z | Pauli::Y) {
                    *a1 = -*a1;
                }
                if matches!(p, Pauli::X | Pauli::Y) {
                    std::mem::swap(a0, a1);
                }
            }
        };
        self.amps.par_chunks_mut(ne).for_each(|v| {
            for e in 0..ne {
                if e & m != 0 {
                    continue;
                }
                let (mut a0, mut a1) = (v[e], v[e | m] * ph);
                fix(&mut a0, &mut a1, pf.phase);
                let (mut b0, mut b1) = ((a0 + a1) * h, (a0 - a1) * h);
                fix(&mut b0, &mut b1, pf.h2);
                v[e] = b0;
                v[e | m] = b1;
            }
        });
        let p1: f64 = self
            .amps
            .par_chunks(ne)
            .map(|v| {
                (0..ne)
                    .filter(|e| e & m != 0)
                    .map(|e| v[e].norm_sqr())
                    .sum::<f64>()
            })
            .sum();
        (p1 / self.total, pf.meas)
    }

    /// Projects exponent bit `j` onto `bit` and resets it to 0.
    pub fn collapse(&mut self, j: usize, bit: bool) {
        let ne = self.ne;
        let m = 1usize << j;
        let mass: f64 = self
            .amps
            .par_chunks(ne)
            .map(|v| {
                (0..ne)
                    .filter(|e| (e & m != 0) == bit)
                    .map(|e| v[e].norm_sqr())
                    .sum::<f64>()
            })
            .sum();
        assert!(
            mass > 0.0,
            "cannot collapse onto a zero-probability outcome"
        );
        let k = 1.0 / mass.sqrt();
        self.amps.par_chunks_mut(ne).for_each(|v| {
            for e in 0..ne {
                if e & m != 0 {
                    continue;
                }
                let src = if bit { v[e | m] } else { v[e] };
                v[e] = src * k;
                v[e | m] = Complex64::zero();
            }
        });
        self.total = 1.0;
    }
}

impl<K: Key, T: Real> GenState<K, T> {
    /// Runs the block of window `k` of a windowed exponentiation on every
    /// `(e, x)` branch; returns the exponent-register state and `W`.
    #[allow(clippy::too_many_arguments)]
    pub fn window_block(
        &mut self,
        res: &Resolved,
        nq: usize,
        k: usize,
        we: usize,
        faults: &[Fault],
        cap: usize,
        choose: &mut dyn FnMut(&[f64]) -> (usize, f64),
    ) -> Result<(WinState<K>, f64), Capped> {
        self.support_trace.push(self.keys.len());
        let (segs, _, wpost) = res.program_w(nq, k, faults);
        let eq = &res.eq;
        assert_eq!(eq.len(), we);
        assert_eq!(eq[0], 0, "exponent bit 0 is the control word");
        let ne = 1usize << we;
        let s = self.keys.len();
        let emask: K = group_mask(&eq[1..]);
        let mut keys: Vec<K> = Vec::with_capacity(ne * s);
        let mut ctrls: Vec<bool> = Vec::with_capacity(ne * s);
        let mut amps: Vec<Complex<T>> = Vec::with_capacity(ne * s);
        for e in 0..ne {
            let mut eb = K::default();
            for (j, &q) in eq.iter().enumerate().skip(1) {
                if (e >> j) & 1 == 1 {
                    let b = q as usize - 1;
                    eb.set_word(b / 64, eb.word(b / 64) | (1u64 << (b % 64)));
                }
            }
            for (kk, a) in self.keys.iter().zip(&self.amps) {
                let mut kk = *kk;
                for wi in 0..K::WORDS {
                    kk.set_word(wi, kk.word(wi) | eb.word(wi));
                }
                keys.push(kk);
                ctrls.push(e & 1 == 1);
                amps.push(*a);
            }
        }
        let mut reset_w = 1.0f64;
        let mut out: Vec<OutBranch<K, T>> = Vec::new();
        for (si, seg) in segs.iter().enumerate() {
            out = vec![(K::default(), Complex::zero(), false); keys.len()];
            eval_dispatch(&seg.ops, nq, &keys, &amps, &ctrls, &mut out);
            if let Some((grp, flips)) = &seg.reset {
                reset_w *= reset_group(&mut out, &group_mask::<K>(grp), choose);
                if !flips.is_empty() {
                    let fm: K = group_mask(flips);
                    for e in out.iter_mut() {
                        for wi in 0..K::WORDS {
                            e.0.set_word(wi, e.0.word(wi) ^ fm.word(wi));
                        }
                    }
                }
            }
            if si + 1 < segs.len() {
                keys = out.iter().map(|e| e.0).collect();
                amps = out.iter().map(|e| e.1).collect();
                ctrls = out.iter().map(|e| e.2).collect();
            }
        }
        drop(keys);
        drop(amps);
        drop(ctrls);
        self.work_ops += (ne * s) as u128 * res.rounds[k].ops.len() as u128;
        let mut inv = K::default();
        for wi in 0..K::WORDS {
            inv.set_word(wi, !emask.word(wi));
        }
        // (rest, e, amplitude)
        let mut v: Vec<(K, usize, Complex<T>)> = out
            .into_par_iter()
            .map(|(key, a, c)| {
                let mut e = usize::from(c);
                for (j, &q) in eq.iter().enumerate().skip(1) {
                    if key.bit(q as usize - 1) {
                        e |= 1 << j;
                    }
                }
                (group_bits(&key, &inv), e, a)
            })
            .collect();
        v.par_sort_unstable_by_key(|x| (x.0, x.1));
        let scale = (1.0 / ne as f64).sqrt();
        let mut rest: Vec<K> = Vec::new();
        let mut wamps: Vec<Complex64> = Vec::new();
        let mut collided = false;
        let mut last: Option<(K, usize)> = None;
        for (key, e, a) in v {
            if rest.last() != Some(&key) {
                rest.push(key);
                wamps.resize(wamps.len() + ne, Complex64::zero());
            } else if last == Some((key, e)) {
                collided = true;
            }
            last = Some((key, e));
            let base = (rest.len() - 1) * ne;
            wamps[base + e] += c64(a) * scale;
        }
        if collided {
            self.collision_rounds += 1;
        }
        self.peak = self.peak.max(rest.len());
        if self.dirty_from.is_none() {
            // exponent bits are measured and reset; key bits >= n of the
            // rest are ancillas
            let n = self.work_bits;
            if rest.iter().any(|kk| kk.any_from(n)) {
                self.dirty_from = Some(k);
            }
        }
        if rest.len() > cap {
            return Err(Capped {
                round: k,
                support: rest.len(),
            });
        }
        let total: f64 = wamps.par_iter().map(|a| a.norm_sqr()).sum();
        self.keys = Vec::new();
        self.amps = Vec::new();
        Ok((
            WinState {
                rest,
                amps: wamps,
                ne,
                post: wpost,
                total,
            },
            total * reset_w,
        ))
    }

    /// Takes the state back from a fully measured window.
    pub fn finish_window(&mut self, ws: WinState<K>) {
        let ne = ws.ne;
        let mut keys = Vec::with_capacity(ws.rest.len());
        let mut amps = Vec::with_capacity(ws.rest.len());
        for (r, key) in ws.rest.iter().enumerate() {
            let a = ws.amps[r * ne];
            if a.norm_sqr() > 1e-28 {
                keys.push(*key);
                amps.push(cvt::<T>(a));
            }
        }
        self.keys = keys;
        self.amps = amps;
    }
}

/// [`run_trajectory`] for a windowed exponentiation.
fn run_trajectory_windowed<K: Key, T: Real, R: Rng + ?Sized>(
    gc: &GenCircuit,
    res: &Resolved,
    faults: &[Fault],
    cap: usize,
    rng: &mut R,
) -> GenTrajectory {
    let inst = &gc.inst;
    let we = gc.we;
    let nw = inst.t / we;
    let mut by_round: Vec<Vec<Fault>> = vec![Vec::new(); nw];
    for f in faults {
        by_round[f.round as usize].push(*f);
    }
    let mut s = GenState::<K, T>::new();
    s.work_bits = inst.m;
    let mut y = 0u128;
    for (k, fs) in by_round.iter().enumerate() {
        let mut choose = |pr: &[f64]| {
            let mut u = rng.random::<f64>();
            for (j, &x) in pr.iter().enumerate() {
                if u < x {
                    return (j, 1.0);
                }
                u -= x;
            }
            (pr.len() - 1, 1.0)
        };
        match s.window_block(res, gc.nq, k, we, fs, cap, &mut choose) {
            Ok((mut ws, wn)) => {
                s.log_weight += wn.ln();
                if wn <= 0.0 {
                    return GenTrajectory {
                        measured: Some(y),
                        order: None,
                        factor: None,
                        peak: s.peak,
                        capped: None,
                        dirty_from: s.dirty_from,
                        work_ops: s.work_ops,
                        support_trace: std::mem::take(&mut s.support_trace),
                        weight: 0.0,
                        collision_rounds: s.collision_rounds,
                    };
                }
                for jj in 0..we {
                    let i = k * we + jj;
                    let j = we - 1 - jj;
                    let (p1, flip) = ws.prepare(j, Instance::correction(i, y));
                    let bit = rng.random::<f64>() < p1;
                    ws.collapse(j, bit);
                    if bit ^ flip {
                        y |= 1 << i;
                    }
                }
                s.finish_window(ws);
            }
            Err(c) => {
                let mut st = std::mem::take(&mut s.support_trace);
                st.push(c.support);
                return GenTrajectory {
                    support_trace: st,
                    measured: None,
                    order: None,
                    factor: None,
                    peak: s.peak,
                    capped: Some(c),
                    dirty_from: s.dirty_from,
                    work_ops: s.work_ops,
                    weight: s.log_weight.exp(),
                    collision_rounds: s.collision_rounds,
                };
            }
        }
    }
    let (order, factor) = postprocess(inst.n_mod, inst.a, y, inst.t as u32);
    GenTrajectory {
        support_trace: std::mem::take(&mut s.support_trace),
        measured: Some(y),
        order,
        factor,
        peak: s.peak,
        capped: None,
        dirty_from: s.dirty_from,
        work_ops: s.work_ops,
        weight: s.log_weight.exp(),
        collision_rounds: s.collision_rounds,
    }
}

/// [`trajectory_distribution`] for a windowed exponentiation.
fn trajectory_distribution_windowed<K: Key>(
    gc: &GenCircuit,
    res: &Resolved,
    faults: &[Fault],
) -> Vec<f64> {
    let t = gc.inst.t;
    let we = gc.we;
    let nw = t / we;
    let mut by_round: Vec<Vec<Fault>> = vec![Vec::new(); nw];
    for f in faults {
        by_round[f.round as usize].push(*f);
    }
    let mut out = vec![0.0; 1usize << t];
    #[allow(clippy::too_many_arguments)]
    fn bits<K: Key>(
        gc: &GenCircuit,
        res: &Resolved,
        by_round: &[Vec<Fault>],
        s: &GenState<K, f64>,
        ws: WinState<K>,
        k: usize,
        jj: usize,
        y: u128,
        p: f64,
        out: &mut [f64],
    ) {
        let we = gc.we;
        if jj == we {
            let mut c = s.clone();
            c.finish_window(ws);
            walk(gc, res, by_round, c, k + 1, y, p, out);
            return;
        }
        let i = k * we + jj;
        let j = we - 1 - jj;
        let mut ws = ws;
        let (p1, flip) = ws.prepare(j, Instance::correction(i, y));
        for bit in [false, true] {
            let pb = if bit { p1 } else { 1.0 - p1 };
            if pb <= 1e-300 {
                continue;
            }
            let mut w2 = ws.clone();
            w2.collapse(j, bit);
            let rec = u128::from(bit ^ flip) << i;
            bits(gc, res, by_round, s, w2, k, jj + 1, y | rec, p * pb, out);
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn walk<K: Key>(
        gc: &GenCircuit,
        res: &Resolved,
        by_round: &[Vec<Fault>],
        mut s: GenState<K, f64>,
        k: usize,
        y: u128,
        p: f64,
        out: &mut [f64],
    ) {
        if k == gc.inst.t / gc.we {
            out[y as usize] += p;
            return;
        }
        let (ws, wn) = s
            .window_block(
                res,
                gc.nq,
                k,
                gc.we,
                &by_round[k],
                usize::MAX,
                &mut argmax_choice,
            )
            .unwrap();
        if wn <= 1e-300 {
            return;
        }
        bits(gc, res, by_round, &s, ws, k, 0, y, p * wn, out);
    }
    let mut s0 = GenState::<K, f64>::new();
    s0.work_bits = gc.inst.m;
    walk::<K>(gc, res, &by_round, s0, 0, 0, 1.0, &mut out);
    out
}

/// Outcome of one trajectory.
#[derive(Clone, Debug)]
pub struct GenTrajectory {
    /// The recorded `t`-bit integer (`None` if the run was capped).
    pub measured: Option<u128>,
    /// Order recovered from `measured` by [`postprocess`], if any.
    pub order: Option<u64>,
    /// Nontrivial factor of `N` derived from the order, if any.
    pub factor: Option<u64>,
    /// Largest support seen ([`GenState::peak`]).
    pub peak: usize,
    /// Where the support exceeded the cap, if it did.
    pub capped: Option<Capped>,
    /// First round after which some branch had a non-zero ancilla.
    pub dirty_from: Option<usize>,
    /// Gate × branch applications.
    pub work_ops: u128,
    /// Support at the start of every round that ran (plus the support that
    /// exceeded the cap, if capped).
    pub support_trace: Vec<usize>,
    /// Importance weight `Π 2P(m_j)` of the uniformly drawn recorded
    /// X-basis outcomes (1 unless branches collided).
    pub weight: f64,
    /// Rounds in which branches collided ([`GenState::collision_rounds`]).
    pub collision_rounds: usize,
}

/// Runs one trajectory on a resolved circuit with the given faults.
#[allow(clippy::too_many_arguments)]
pub fn run_trajectory<K: Key, T: Real, R: Rng + ?Sized>(
    gc: &GenCircuit,
    res: &Resolved,
    faults: &[Fault],
    cap: usize,
    reset_ancillas: bool,
    rng: &mut R,
) -> GenTrajectory {
    assert!(gc.nq - 1 <= 64 * K::WORDS, "key too narrow");
    if gc.we > 1 {
        assert!(!reset_ancillas, "no ideal reset in windowed exponentiation");
        return run_trajectory_windowed::<K, T, R>(gc, res, faults, cap, rng);
    }
    let inst = &gc.inst;
    let mut by_round: Vec<Vec<Fault>> = vec![Vec::new(); inst.t];
    for f in faults {
        by_round[f.round as usize].push(*f);
    }
    let mut s = GenState::<K, T>::new();
    let mut y = 0u128;
    for (i, fs) in by_round.iter().enumerate() {
        let mut choose = |pr: &[f64]| {
            let mut u = rng.random::<f64>();
            for (j, &x) in pr.iter().enumerate() {
                if u < x {
                    return (j, 1.0);
                }
                u -= x;
            }
            (pr.len() - 1, 1.0)
        };
        let step = s.round(res, inst, gc.nq, i, y, fs, cap, &mut choose);
        match step {
            Ok((p1, flip, wn)) => {
                s.log_weight += wn.ln();
                if wn <= 0.0 {
                    // every branch cancelled: weight 0, outcome irrelevant
                    return GenTrajectory {
                        measured: Some(y),
                        order: None,
                        factor: None,
                        peak: s.peak,
                        capped: None,
                        dirty_from: s.dirty_from,
                        work_ops: s.work_ops,
                        support_trace: std::mem::take(&mut s.support_trace),
                        weight: 0.0,
                        collision_rounds: s.collision_rounds,
                    };
                }
                let bit = rng.random::<f64>() < p1;
                if i + 1 < inst.t {
                    s.collapse(bit);
                    if reset_ancillas {
                        s.reset_ancillas(inst.m, rng);
                    }
                }
                if bit ^ flip {
                    y |= 1 << i;
                }
            }
            Err(c) => {
                let mut st = std::mem::take(&mut s.support_trace);
                st.push(c.support);
                return GenTrajectory {
                    support_trace: st,
                    measured: None,
                    order: None,
                    factor: None,
                    peak: s.peak,
                    capped: Some(c),
                    dirty_from: s.dirty_from,
                    work_ops: s.work_ops,
                    weight: s.log_weight.exp(),
                    collision_rounds: s.collision_rounds,
                };
            }
        }
    }
    let (order, factor) = postprocess(inst.n_mod, inst.a, y, inst.t as u32);
    GenTrajectory {
        support_trace: std::mem::take(&mut s.support_trace),
        measured: Some(y),
        order,
        factor,
        peak: s.peak,
        capped: None,
        dirty_from: s.dirty_from,
        work_ops: s.work_ops,
        weight: s.log_weight.exp(),
        collision_rounds: s.collision_rounds,
    }
}

/// Exact distribution of the recorded integer for a fixed fault pattern on
/// a fixed resolved circuit (fixed recorded X-basis outcomes), weighted by
/// `W`: `Σ_y` of the result is `Π 2P(m_j)` of the recorded outcomes given
/// the faults (1 without collisions). Small instances only.
pub fn trajectory_distribution<K: Key>(
    gc: &GenCircuit,
    res: &Resolved,
    faults: &[Fault],
) -> Vec<f64> {
    if gc.we > 1 {
        return trajectory_distribution_windowed::<K>(gc, res, faults);
    }
    let t = gc.inst.t;
    let mut by_round: Vec<Vec<Fault>> = vec![Vec::new(); t];
    for f in faults {
        by_round[f.round as usize].push(*f);
    }
    let mut out = vec![0.0; 1usize << t];
    #[allow(clippy::too_many_arguments)]
    fn walk<K: Key>(
        gc: &GenCircuit,
        res: &Resolved,
        by_round: &[Vec<Fault>],
        mut s: GenState<K, f64>,
        i: usize,
        y: u128,
        p: f64,
        out: &mut [f64],
    ) {
        if i == gc.inst.t {
            out[y as usize] += p;
            return;
        }
        let (p1, flip, wn) = s
            .round(
                res,
                &gc.inst,
                gc.nq,
                i,
                y,
                &by_round[i],
                usize::MAX,
                &mut argmax_choice,
            )
            .unwrap();
        if wn <= 1e-300 {
            return;
        }
        for bit in [false, true] {
            let pb = if bit { p1 } else { 1.0 - p1 };
            if pb <= 1e-300 {
                continue;
            }
            let mut c = s.clone();
            c.collapse(bit);
            let rec = u128::from(bit ^ flip) << i;
            walk(gc, res, by_round, c, i + 1, y | rec, p * pb * wn, out);
        }
    }
    walk::<K>(gc, res, &by_round, GenState::new(), 0, 0, 1.0, &mut out);
    out
}

/// Deterministic reset outcome for exact distributions: the most probable
/// group value, with its probability as the path weight.
/// Ties (within 1e-9) go to the smallest group value.
pub fn argmax_choice(p: &[f64]) -> (usize, f64) {
    let mx = p.iter().copied().fold(0.0, f64::max);
    let best = p.iter().position(|&x| x >= mx - 1e-9).unwrap();
    (best, p[best])
}

/// The success criterion of research/shor/shor-noise.md ("peak"):
/// `|y/2^t − s/r| < 1/(2r²)` for the nearest `s`.
pub fn peak_ok(y: u128, t: usize, r: u64) -> bool {
    // dist(y·r/2^t, Z) < 1/(2r)  <=>  |y·r − s·2^t| · 2r < 2^t
    let yr = y * u128::from(r);
    let m = 1u128 << t;
    let rem = yr % m;
    let d = rem.min(m - rem);
    d * 2 * u128::from(r) < m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shor::noisy::{self, NoisyCircuit};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn keys_round_trip() {
        let mut k = K192::default();
        k.set_word(2, 5);
        k.set_word(0, 1 << 63);
        assert!(k.bit(63) && k.bit(128) && !k.bit(129) && k.bit(130));
        let (lo, hi) = k.split(100);
        assert_eq!(lo, K192([1 << 63, 0, 0]));
        assert_eq!(hi, K192([0, 0, 5]));
        assert!(k.any_from(130) && !k.any_from(131));
        let u: u128 = (1u128 << 100) | 7;
        assert_eq!(u.split(64), (7, 1u128 << 100));
        let mut v = 0u128;
        v.set_word(1, 3);
        assert_eq!(v, 3u128 << 64);
    }

    #[test]
    fn peak_matches_float() {
        for (t, r) in [(8usize, 6u64), (10, 21), (12, 40)] {
            for y in 0..(1u128 << t) {
                let f = y as f64 * r as f64 / (1u64 << t) as f64;
                let d = (f - f.round()).abs();
                if (d - 1.0 / (2.0 * r as f64)).abs() > 1e-9 {
                    assert_eq!(peak_ok(y, t, r), d < 1.0 / (2.0 * r as f64), "{y} {t} {r}");
                }
            }
        }
    }

    #[test]
    fn mbu_tagged_resolution_equals_shor_mbu() {
        for (n_mod, a, oracle) in [
            (15u64, 7u64, Oracle::WindowedMbu(2)),
            (21, 2, Oracle::WindowedMbuLookup(2)),
            (55, 2, Oracle::WindowedMbu(3)),
            (143, 5, Oracle::WindowedMbuLookup(4)),
        ] {
            let inst = Instance::new(n_mod, a, oracle);
            let (w, o) = match oracle {
                Oracle::WindowedMbu(w) => (w, MbuOpts::ALL),
                Oracle::WindowedMbuLookup(w) => (w, MbuOpts::LOOKUPS),
                _ => unreachable!(),
            };
            let lay = MbuLayout::new(inst.m, w, &o);
            for seed in 0..4u64 {
                for &mult in &inst.mults {
                    let mut r1 = StdRng::seed_from_u64(seed);
                    let mut r2 = StdRng::seed_from_u64(seed);
                    let (ops, tags) = mbu_round(&inst, w, &o, mult, &mut || r1.random());
                    let lops = shor_mbu::controlled_ua_ops(&lay, mult, n_mod, &o);
                    let mut refops = Vec::new();
                    shor_mbu::resolve(&lops, &mut || r2.random(), &mut refops);
                    let refops: Vec<NOp> = refops
                        .into_iter()
                        .filter_map(|o| match o {
                            shor_mbu::MbuOp::G(g) => Some(NOp::G(g)),
                            shor_mbu::MbuOp::MeasX(q, m) => Some(NOp::MeasX(q as u32, m)),
                            shor_mbu::MbuOp::GlobalNeg => None,
                        })
                        .collect();
                    assert_eq!(ops, refops);
                    assert!(tags.iter().all(|t| t & tag::BLOCK != tag::OTHER));
                }
            }
        }
    }

    #[test]
    fn windowed_opt_tags_cover_blocks() {
        let inst = Instance::new(143, 5, Oracle::WindowedOpt(2));
        let gc = GenCircuit::new(&inst, NoiseKind::Depolarizing);
        let res = gc.resolve(&mut || false);
        let mut seen = std::collections::BTreeSet::new();
        for r in &res.rounds {
            for &t in &r.tags {
                seen.insert(tag::name(t));
            }
        }
        for want in [
            "lookup.unary",
            "lookup.fanout",
            "unlookup.unary",
            "modadd.add",
            "modadd.reduce",
            "modadd.cmp",
            "swap.gate",
            "modadd.add.inv",
            "lookup.fanout.inv",
        ] {
            assert!(seen.contains(want), "{want} missing: {seen:?}");
        }
    }

    /// The generic engine reproduces the original engine (research/
    /// shor-noise.md) exactly on the reversible oracles.
    #[test]
    fn matches_original_engine_on_reversible_oracles() {
        let mut rng = StdRng::seed_from_u64(7);
        for (n_mod, a, oracle) in [
            (15u64, 7u64, Oracle::Windowed(2)),
            (21, 2, Oracle::WindowedOpt(2)),
            (35, 3, Oracle::WindowedOpt(3)),
        ] {
            let inst = Instance::new(n_mod, a, oracle);
            for kind in [
                NoiseKind::Depolarizing,
                NoiseKind::BitFlip,
                NoiseKind::PhaseFlip,
            ] {
                let old = NoisyCircuit::new(&inst, kind);
                let gc = GenCircuit::new(&inst, kind);
                let res = gc.resolve(&mut || false);
                assert_eq!(old.num_locations(), res.num_locations());
                for k in [0usize, 1, 1, 2, 3] {
                    let faults = res.sample_k(k, &mut rng);
                    let d1 = noisy::trajectory_distribution(&old, &faults);
                    let d2 = trajectory_distribution::<u128>(&gc, &res, &faults);
                    let d3 = trajectory_distribution::<K192>(&gc, &res, &faults);
                    for ((x, y), z) in d1.iter().zip(&d2).zip(&d3) {
                        assert!((x - y).abs() < 1e-12 && (x - z).abs() < 1e-12);
                    }
                }
            }
        }
    }
}
