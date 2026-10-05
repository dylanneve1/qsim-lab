//! Exact Monte-Carlo trajectories of the gate-level semiclassical Shor
//! circuit under stochastic Pauli noise, on bit slices.
//!
//! # Why this stays exact at the cost of the noiseless engine
//!
//! The oracle block of every round contains only X, CNOT and CCX, so the
//! state of the whole register (control, work register, every ancilla) is a
//! list of basis-state branches with amplitudes. A Pauli fault keeps it so:
//!
//! * `X_q` flips bit `q` of every branch — one more `w[q] ^= 1` step on the
//!   slices, a basis-state permutation like the gates themselves;
//! * `Z_q` multiplies each branch by `(-1)^{b_q}` — tracked *exactly* as a
//!   per-branch sign bit, one more slice word updated by `sign ^= w[q]`;
//! * `Y_q = i X_q Z_q`: the `Z` step then the `X` step (the factor `i` is a
//!   global phase of the trajectory).
//!
//! So a trajectory with any fault pattern costs one bit-sliced evaluation of
//! the round per branch, exactly as the noiseless engine
//! ([`super::sliced`]). What changes is the bookkeeping: a fault can leave
//! ancillas dirty or the control flipped, so a branch is keyed by **all**
//! non-control qubits (a `u128`, `4n + 4 + w ≤ 129` qubits, `n ≤ 30`), and
//! the control-0 half of a round is a genuine map (no identity shortcut).
//! The control algebra is the general form of the noiseless one: after the
//! block, `A_c(k)` is the (signed) amplitude of the branch with control `c`
//! and rest `k` (a permutation, so at most one input maps there), and
//! `P(1) = Σ_k |A_0(k) − e^{iφ} A_1(k)|² / 4` before any post-phase fault.
//!
//! # Fault locations (all at the same rate `p` in [`NoiseKind`])
//!
//! Per round `i`: `Prep` (the recycled control starts in `|1>`: an X flip),
//! `H1` (Pauli after the first H), `Gate{g, slot}` (Pauli on each qubit of
//! every oracle gate, after the gate), `Phase` (after the phase correction),
//! `H2` (after the second H), `Meas` (readout flip of the recorded bit, which
//! the classical controller then feeds forward). State preparation of the
//! work register and ancillas is ideal and there is no idle noise.
//!
//! Conditional on exactly `k` faults, the faulted locations are a uniform
//! `k`-subset ([`NoisyCircuit::sample_k`]); that is what makes the fault-count
//! stratified estimate of `P_succ(p)` exact in law.

use super::sliced::{eval_raw_unchecked, oracle_block, transpose64};
use super::{postprocess, Instance, Oracle};
use crate::circuit::{Circuit, Op};
use crate::engines::statevector::Real;
use crate::gate::Gate;
use num_complex::{Complex, Complex64};
use num_traits::Zero;
use rand::Rng;
use rayon::prelude::*;
use std::f64::consts::PI;

/// A single-qubit Pauli fault.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pauli {
    /// Bit flip.
    X,
    /// Bit and phase flip (`Y`).
    Y,
    /// Phase flip.
    Z,
}

impl Pauli {
    /// `"X"`, `"Y"` or `"Z"`.
    pub fn name(self) -> &'static str {
        match self {
            Pauli::X => "X",
            Pauli::Y => "Y",
            Pauli::Z => "Z",
        }
    }
    fn has_x(self) -> bool {
        !matches!(self, Pauli::Z)
    }
    fn has_z(self) -> bool {
        !matches!(self, Pauli::X)
    }
}

/// Which Pauli channel acts at every location.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoiseKind {
    /// X, Y, Z with probability `p/3` each at gate / H / phase locations;
    /// X (flip) at `Prep` and `Meas`.
    Depolarizing,
    /// X with probability `p` at every location.
    BitFlip,
    /// Z with probability `p` at gate / H / phase locations (no `Prep` or
    /// `Meas` locations).
    PhaseFlip,
}

impl NoiseKind {
    /// Parses `depol` / `depolarizing`, `bitflip` / `x`, `phaseflip` / `z`;
    /// `None` for anything else.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "depol" | "depolarizing" => Some(Self::Depolarizing),
            "bitflip" | "x" => Some(Self::BitFlip),
            "phaseflip" | "z" => Some(Self::PhaseFlip),
            _ => None,
        }
    }
    /// Short name (`depol`, `bitflip`, `phaseflip`), accepted by [`NoiseKind::parse`].
    pub fn name(self) -> &'static str {
        match self {
            Self::Depolarizing => "depol",
            Self::BitFlip => "bitflip",
            Self::PhaseFlip => "phaseflip",
        }
    }
    fn has_prep_meas(self) -> bool {
        !matches!(self, Self::PhaseFlip)
    }
    fn sample_pauli<R: Rng + ?Sized>(self, rng: &mut R) -> Pauli {
        match self {
            Self::Depolarizing => [Pauli::X, Pauli::Y, Pauli::Z][rng.random_range(0..3)],
            Self::BitFlip => Pauli::X,
            Self::PhaseFlip => Pauli::Z,
        }
    }
}

/// A fault location inside one round.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Site {
    /// The recycled control starts the round in `|1>` (X before the first H).
    Prep,
    /// Pauli on the control after the first H.
    H1,
    /// Pauli on qubit `slot` of oracle gate `gate`, after the gate.
    Gate {
        /// Index of the gate in [`RoundCirc::gates`].
        gate: u32,
        /// Index into the gate's qubit list (`Gate::qubits`).
        slot: u8,
    },
    /// Pauli on the control after the phase correction.
    Phase,
    /// Pauli on the control after the second H.
    H2,
    /// The recorded bit is flipped (readout error).
    Meas,
}

impl Site {
    /// Lower-case name of the site kind (`prep`, `h1`, `gate`, `phase`, `h2`,
    /// `meas`).
    pub fn kind_name(&self) -> &'static str {
        match self {
            Site::Prep => "prep",
            Site::H1 => "h1",
            Site::Gate { .. } => "gate",
            Site::Phase => "phase",
            Site::H2 => "h2",
            Site::Meas => "meas",
        }
    }
}

/// One fault of a trajectory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fault {
    /// Round index (`0..t`, in measurement order).
    pub round: u32,
    /// Where in the round the fault acts.
    pub site: Site,
    /// For `Prep` and `Meas` this is always `X` (a flip).
    pub pauli: Pauli,
}

/// The oracle gates of one round and the slot offsets of their qubits.
#[derive(Clone, Debug)]
pub struct RoundCirc {
    /// The round's oracle gates (X, CNOT, CCX, SWAP only), in application order.
    pub gates: Vec<Gate>,
    /// `slot_prefix[g]` = number of gate-qubit locations before gate `g`.
    slot_prefix: Vec<u64>,
}

impl RoundCirc {
    /// Number of gate-qubit fault locations in the round (Σ of gate arities).
    pub fn gate_slots(&self) -> u64 {
        *self.slot_prefix.last().unwrap()
    }
}

/// The semiclassical circuit of one instance with its fault locations.
#[derive(Clone, Debug)]
pub struct NoisyCircuit {
    /// The order-finding instance.
    pub inst: Instance,
    /// The noise channel.
    pub kind: NoiseKind,
    /// Qubits of the circuit (control = qubit 0).
    pub nq: usize,
    /// The oracle block of every round (`t` entries, round `i` multiplies by
    /// `a^(2^(t−1−i))`).
    pub rounds: Vec<RoundCirc>,
    /// Global location offset of every round (`t + 1` entries).
    loc_prefix: Vec<u64>,
}

impl NoisyCircuit {
    /// Builds the rounds and fault locations of `inst`. Panics unless the oracle
    /// is `Ripple`, `Windowed` or `WindowedOpt` and the circuit has at most 129
    /// qubits.
    pub fn new(inst: &Instance, kind: NoiseKind) -> Self {
        assert!(matches!(
            inst.oracle,
            Oracle::Ripple | Oracle::Windowed(_) | Oracle::WindowedOpt(_)
        ));
        let nq = inst.qubits();
        assert!(
            nq <= 129,
            "the noisy engine keys the {} non-control qubits in a u128",
            nq - 1
        );
        let mut rounds = Vec::with_capacity(inst.t);
        for i in 0..inst.t {
            let mult = inst.mults[inst.t - 1 - i];
            let (c, io) = oracle_block(inst, mult);
            assert_eq!(io.ctrl, 0, "the control must be qubit 0");
            assert_eq!(c.num_qubits, nq);
            let gates: Vec<Gate> = c
                .ops
                .iter()
                .map(|op| match op {
                    Op::Gate(
                        g @ (Gate::X(_) | Gate::Cnot(..) | Gate::Ccx(..) | Gate::Swap(..)),
                    ) => *g,
                    o => panic!("{o:?} is not a basis-state permutation gate"),
                })
                .collect();
            let mut slot_prefix = Vec::with_capacity(gates.len() + 1);
            let mut s = 0u64;
            slot_prefix.push(0);
            for g in &gates {
                s += g.arity() as u64;
                slot_prefix.push(s);
            }
            rounds.push(RoundCirc { gates, slot_prefix });
        }
        let pm = u64::from(kind.has_prep_meas());
        let mut loc_prefix = vec![0u64];
        for r in &rounds {
            let n = pm + 1 + r.gate_slots() + 2 + pm;
            loc_prefix.push(loc_prefix.last().unwrap() + n);
        }
        Self {
            inst: inst.clone(),
            kind,
            nq,
            rounds,
            loc_prefix,
        }
    }

    /// Total number of fault locations `L`.
    pub fn num_locations(&self) -> u64 {
        *self.loc_prefix.last().unwrap()
    }

    /// Fault locations of round `i`.
    pub fn round_locations(&self, i: usize) -> u64 {
        self.loc_prefix[i + 1] - self.loc_prefix[i]
    }

    /// Round and site of global location `g < L`.
    pub fn site_of(&self, g: u64) -> (usize, Site) {
        let i = self.loc_prefix.partition_point(|&x| x <= g) - 1;
        let mut l = g - self.loc_prefix[i];
        let pm = self.kind.has_prep_meas();
        let rc = &self.rounds[i];
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
            _ => self.kind.sample_pauli(rng),
        };
        Fault {
            round: round as u32,
            site,
            pauli,
        }
    }

    /// Exactly `k` faults at a uniformly random `k`-subset of the
    /// locations, each with a Pauli drawn from the location's channel.
    pub fn sample_k<R: Rng + ?Sized>(&self, k: usize, rng: &mut R) -> Vec<Fault> {
        let l = self.num_locations();
        assert!((k as u64) <= l);
        let mut picked = std::collections::BTreeSet::new();
        while picked.len() < k {
            picked.insert(rng.random_range(0..l));
        }
        picked.into_iter().map(|g| self.fault_at(g, rng)).collect()
    }

    /// Independent faults with probability `p` at every location
    /// (geometric skipping).
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
            let u: f64 = 1.0 - rng.random::<f64>(); // (0, 1]
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

    /// The gate and qubit hit by a gate fault.
    pub fn gate_qubit(&self, f: &Fault) -> Option<(Gate, usize)> {
        match f.site {
            Site::Gate { gate, slot } => {
                let g = self.rounds[f.round as usize].gates[gate as usize];
                Some((g, g.qubits()[slot as usize]))
            }
            _ => None,
        }
    }

    /// Compiles round `i` with the faults of that round spliced in. Words:
    /// `0..nq` qubits, `nq` all-ones, `nq + 1` the sign. Returns the program
    /// and the post-block faults (phase, H2, readout flip).
    fn program(&self, i: usize, faults: &[Fault]) -> (Vec<[u32; 3]>, PostFaults) {
        let nq = self.nq as u32;
        let (one, sign) = (nq, nq + 1);
        let rc = &self.rounds[i];
        let mut gf: Vec<(u32, u8, Pauli)> = Vec::new();
        let mut post = PostFaults::default();
        let mut ops = Vec::with_capacity(rc.gates.len() + 8);
        let push_pauli = |ops: &mut Vec<[u32; 3]>, q: u32, p: Pauli| {
            if p.has_z() {
                ops.push([sign, q, one]);
            }
            if p.has_x() {
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
        // X before H = Z after H (HX = ZH)
        if prep {
            push_pauli(&mut ops, 0, Pauli::Z);
        }
        if let Some(p) = h1 {
            push_pauli(&mut ops, 0, p);
        }
        gf.sort_unstable_by_key(|&(g, s, _)| (g, s));
        let mut fi = 0;
        for (gi, g) in rc.gates.iter().enumerate() {
            match *g {
                Gate::X(t) => ops.push([t as u32, one, one]),
                Gate::Cnot(c, t) => ops.push([t as u32, c as u32, one]),
                Gate::Ccx(a, b, t) => ops.push([t as u32, a as u32, b as u32]),
                Gate::Swap(a, b) => {
                    let (a, b) = (a as u32, b as u32);
                    ops.push([a, b, one]);
                    ops.push([b, a, one]);
                    ops.push([a, b, one]);
                }
                _ => unreachable!(),
            }
            while fi < gf.len() && gf[fi].0 as usize == gi {
                let q = g.qubits()[gf[fi].1 as usize] as u32;
                push_pauli(&mut ops, q, gf[fi].2);
                fi += 1;
            }
        }
        assert_eq!(fi, gf.len(), "gate fault beyond the round's gates");
        assert!(ops.iter().flatten().all(|&x| x <= sign));
        (ops, post)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct PostFaults {
    phase: Option<Pauli>,
    h2: Option<Pauli>,
    meas: bool,
}

/// One block output branch: rest-of-register key, signed amplitude, control.
type OutBranch<T> = (u128, Complex<T>, bool);

/// Evaluates the faulted round program on `|ctrl>|key>` for every stored
/// key, `64·L` branches per slice batch.
fn eval_half<const L: usize, T: Real>(
    ops: &[[u32; 3]],
    nq: usize,
    keys: &[u128],
    amps: &[Complex<T>],
    ctrl: bool,
    out: &mut [OutBranch<T>],
) {
    let b = 64 * L;
    let sign = nq + 1;
    keys.par_chunks(b)
        .zip(amps.par_chunks(b))
        .zip(out.par_chunks_mut(b))
        .for_each_init(
            || vec![[0u64; L]; nq + 2],
            |w, ((ks, am), os)| {
                for x in w.iter_mut() {
                    *x = [0; L];
                }
                w[nq] = [u64::MAX; L];
                let mut blk = [0u64; 64];
                for (l, kc) in ks.chunks(64).enumerate() {
                    let valid = if kc.len() == 64 {
                        u64::MAX
                    } else {
                        (1u64 << kc.len()) - 1
                    };
                    if ctrl {
                        w[0][l] = valid;
                    }
                    for half in 0..2usize {
                        let lo = 1 + 64 * half;
                        if lo >= nq {
                            break;
                        }
                        for (j, &k) in kc.iter().enumerate() {
                            blk[j] = (k >> (64 * half)) as u64;
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
                // (checked in `NoisyCircuit::program`).
                unsafe { eval_raw_unchecked::<L>(ops, w) };
                for (l, ((kc, ac), oc)) in ks
                    .chunks(64)
                    .zip(am.chunks(64))
                    .zip(os.chunks_mut(64))
                    .enumerate()
                {
                    let mut kout = [0u128; 64];
                    for half in 0..2usize {
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
                            *ko |= u128::from(blk[j]) << (64 * half);
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

fn eval_dispatch<T: Real>(
    ops: &[[u32; 3]],
    nq: usize,
    keys: &[u128],
    amps: &[Complex<T>],
    ctrl: bool,
    out: &mut [OutBranch<T>],
) {
    match keys.len() {
        0..=64 => eval_half::<1, T>(ops, nq, keys, amps, ctrl, out),
        65..=512 => eval_half::<4, T>(ops, nq, keys, amps, ctrl, out),
        _ => eval_half::<16, T>(ops, nq, keys, amps, ctrl, out),
    }
}

fn cvt<T: Real>(z: Complex64) -> Complex<T> {
    Complex::new(T::from_f64(z.re), T::from_f64(z.im))
}
fn c64<T: Real>(z: Complex<T>) -> Complex64 {
    Complex64::new(z.re.to_f64(), z.im.to_f64())
}

/// The support grew beyond the cap given to [`NoisyState::round`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capped {
    /// Round whose support exceeded the cap.
    pub round: usize,
    /// Support size that exceeded the cap.
    pub support: usize,
}

/// Exact state of one noisy trajectory: branches keyed by every
/// non-control qubit (key bit `j` = qubit `j + 1`).
#[derive(Clone, Debug)]
pub struct NoisyState<T: Real> {
    keys: Vec<u128>,
    amps: Vec<Complex<T>>,
    /// After a round: `(key, amplitude of control 0, of control 1)` just
    /// before the measurement.
    merged: Vec<(u128, Complex<T>, Complex<T>)>,
    p1: f64,
    /// Largest support seen (after a merge).
    pub peak: usize,
    /// Gate × branch applications.
    pub work_ops: u128,
    /// First round after which some branch had a non-zero ancilla.
    pub dirty_from: Option<usize>,
    /// Support at the start of every round.
    pub support_trace: Vec<usize>,
}

impl<T: Real> NoisyState<T> {
    /// Control `|0>`, work register `|1>`, ancillas `|0>`.
    pub fn new() -> Self {
        Self {
            keys: vec![1],
            amps: vec![Complex::new(T::one(), T::zero())],
            merged: Vec::new(),
            p1: 0.0,
            peak: 1,
            work_ops: 0,
            dirty_from: None,
            support_trace: Vec::new(),
        }
    }

    /// Number of stored branches.
    pub fn nnz(&self) -> usize {
        self.keys.len()
    }

    /// The stored branches `(rest-of-register key, amplitude)`.
    pub fn branches(&self) -> impl Iterator<Item = (u128, Complex<T>)> + '_ {
        self.keys.iter().copied().zip(self.amps.iter().copied())
    }

    /// Runs round `i` (H, faulted oracle block, phase correction for the
    /// recorded low bits `y_low`, H) and returns `P(qubit reads 1)` and
    /// whether the readout of this round is flipped. Errors if the support
    /// after the round would exceed `cap`.
    pub fn round(
        &mut self,
        nc: &NoisyCircuit,
        i: usize,
        y_low: u128,
        faults: &[Fault],
        cap: usize,
    ) -> Result<(f64, bool), Capped> {
        self.support_trace.push(self.keys.len());
        let (ops, post) = nc.program(i, faults);
        let nq = nc.nq;
        let s = self.keys.len();
        let mut out: Vec<OutBranch<T>> = vec![(0, Complex::zero(), false); 2 * s];
        {
            let (o0, o1) = out.split_at_mut(s);
            eval_dispatch(&ops, nq, &self.keys, &self.amps, false, o0);
            eval_dispatch(&ops, nq, &self.keys, &self.amps, true, o1);
        }
        self.work_ops += 2 * s as u128 * nc.rounds[i].gates.len() as u128;
        let (mut a1, mut a0): (Vec<_>, Vec<_>) = out.into_par_iter().partition(|e| e.2);
        a0.par_sort_unstable_by_key(|e| e.0);
        a1.par_sort_unstable_by_key(|e| e.0);
        assert!(
            a0.par_windows(2).all(|w| w[0].0 != w[1].0)
                && a1.par_windows(2).all(|w| w[0].0 != w[1].0),
            "the faulted round is not a permutation"
        );
        let phi = if y_low != 0 {
            Instance::correction(i, y_low)
        } else {
            0.0
        };
        let ph = Complex64::from_polar(1.0, phi);
        let z = Complex64::zero();
        let mut merged = Vec::with_capacity(a0.len().max(a1.len()));
        let (mut x, mut y) = (0, 0);
        let mut p1 = 0.0f64;
        while x < a0.len() || y < a1.len() {
            let ka = a0.get(x).map_or(u128::MAX, |e| e.0);
            let kb = a1.get(y).map_or(u128::MAX, |e| e.0);
            let key = ka.min(kb);
            let (mut u0, mut u1) = (z, z);
            if x < a0.len() && ka == key {
                u0 = c64(a0[x].1);
                x += 1;
            }
            if y < a1.len() && kb == key {
                u1 = ph * c64(a1[y].1);
                y += 1;
            }
            if let Some(p) = post.phase {
                if p.has_z() {
                    u1 = -u1;
                }
                if p.has_x() {
                    std::mem::swap(&mut u0, &mut u1);
                }
            }
            let mut o0 = (u0 + u1) * 0.5;
            let mut o1 = (u0 - u1) * 0.5;
            if let Some(p) = post.h2 {
                if p.has_z() {
                    o1 = -o1;
                }
                if p.has_x() {
                    std::mem::swap(&mut o0, &mut o1);
                }
            }
            p1 += o1.norm_sqr();
            merged.push((key, cvt::<T>(o0), cvt::<T>(o1)));
        }
        drop(a0);
        drop(a1);
        self.peak = self.peak.max(merged.len());
        let n = nc.inst.m;
        if self.dirty_from.is_none() && merged.iter().any(|e| e.0 >> n != 0) {
            self.dirty_from = Some(i);
        }
        if merged.len() > cap {
            return Err(Capped {
                round: i,
                support: merged.len(),
            });
        }
        self.keys = Vec::new();
        self.amps = Vec::new();
        self.merged = merged;
        self.p1 = p1;
        Ok((p1, post.meas))
    }

    /// Collapses the qubit measured after the last round onto `outcome`.
    pub fn collapse(&mut self, outcome: bool) {
        let p = if outcome { self.p1 } else { 1.0 - self.p1 };
        assert!(p > 0.0, "cannot collapse onto a zero-probability outcome");
        let k = T::from_f64(1.0 / p.sqrt());
        let mut keys = Vec::with_capacity(self.merged.len());
        let mut amps = Vec::with_capacity(self.merged.len());
        for &(key, o0, o1) in &self.merged {
            let v = if outcome { o1 } else { o0 };
            if v != Complex::zero() {
                keys.push(key);
                amps.push(v * k);
            }
        }
        self.merged = Vec::new();
        self.keys = keys;
        self.amps = amps;
    }
}

impl<T: Real> NoisyState<T> {
    /// Measures every ancilla (every non-control qubit above the `n`-bit work
    /// register) and resets it to `|0>`: samples the ancilla value `a` with
    /// probability `Σ_{x} |ψ(x, a)|²`, keeps those branches, clears their
    /// ancilla bits and renormalises. Returns `true` if a non-zero ancilla
    /// value was found (the reset then removed dirt).
    pub fn reset_ancillas<R: Rng + ?Sized>(&mut self, n: usize, rng: &mut R) -> bool {
        if self.keys.iter().all(|&k| k >> n == 0) {
            return false;
        }
        let mut w: std::collections::BTreeMap<u128, f64> = std::collections::BTreeMap::new();
        for (k, a) in self.keys.iter().zip(&self.amps) {
            *w.entry(k >> n).or_insert(0.0) += c64(*a).norm_sqr();
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
        let mask = (1u128 << n) - 1;
        let mut kv: Vec<(u128, Complex<T>)> = self
            .keys
            .iter()
            .zip(&self.amps)
            .filter(|(k, _)| *k >> n == pick)
            .map(|(k, a)| (k & mask, *a * norm))
            .collect();
        kv.sort_unstable_by_key(|e| e.0);
        self.keys = kv.iter().map(|e| e.0).collect();
        self.amps = kv.iter().map(|e| e.1).collect();
        pick != 0
    }
}

impl<T: Real> Default for NoisyState<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Outcome of one noisy trajectory.
#[derive(Clone, Debug)]
pub struct Trajectory {
    /// The recorded `t`-bit integer (`None` if the run was capped).
    pub measured: Option<u128>,
    /// Order recovered from `measured` by [`postprocess`], if any.
    pub order: Option<u64>,
    /// Nontrivial factor of `N` derived from the order, if any.
    pub factor: Option<u64>,
    /// Largest support seen ([`NoisyState::peak`]).
    pub peak: usize,
    /// Where the support exceeded the cap, if it did.
    pub capped: Option<Capped>,
    /// First round after which some branch had a non-zero ancilla
    /// ([`NoisyState::dirty_from`]).
    pub dirty_from: Option<usize>,
    /// Gate × branch applications.
    pub work_ops: u128,
    /// Support at the start of every round that ran (plus the support
    /// that exceeded the cap, if capped).
    pub support_trace: Vec<usize>,
}

/// Runs one trajectory with the given faults (any order). Gives up with
/// `capped` set if the support exceeds `cap` (its outcome is then unknown).
pub fn run_trajectory<T: Real, R: Rng + ?Sized>(
    nc: &NoisyCircuit,
    faults: &[Fault],
    cap: usize,
    rng: &mut R,
) -> Trajectory {
    run_trajectory_opts::<T, R>(nc, faults, cap, false, rng)
}

/// [`run_trajectory`] with an optional (ideal) measure-and-reset of every
/// ancilla after each round's control measurement — what a device with
/// mid-circuit reset could do for qubits that should be clean anyway.
pub fn run_trajectory_opts<T: Real, R: Rng + ?Sized>(
    nc: &NoisyCircuit,
    faults: &[Fault],
    cap: usize,
    reset_ancillas: bool,
    rng: &mut R,
) -> Trajectory {
    let inst = &nc.inst;
    let mut by_round: Vec<Vec<Fault>> = vec![Vec::new(); inst.t];
    for f in faults {
        by_round[f.round as usize].push(*f);
    }
    let mut s = NoisyState::<T>::new();
    let mut y = 0u128;
    for (i, fs) in by_round.iter().enumerate() {
        match s.round(nc, i, y, fs, cap) {
            Ok((p1, flip)) => {
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
                return Trajectory {
                    support_trace: st,
                    measured: None,
                    order: None,
                    factor: None,
                    peak: s.peak,
                    capped: Some(c),
                    dirty_from: s.dirty_from,
                    work_ops: s.work_ops,
                };
            }
        }
    }
    let (order, factor) = postprocess(inst.n_mod, inst.a, y, inst.t as u32);
    Trajectory {
        support_trace: std::mem::take(&mut s.support_trace),
        measured: Some(y),
        order,
        factor,
        peak: s.peak,
        capped: None,
        dirty_from: s.dirty_from,
        work_ops: s.work_ops,
    }
}

/// Exact distribution of the recorded integer for a fixed fault pattern
/// (whole measurement tree; small instances only).
pub fn trajectory_distribution(nc: &NoisyCircuit, faults: &[Fault]) -> Vec<f64> {
    let t = nc.inst.t;
    let mut by_round: Vec<Vec<Fault>> = vec![Vec::new(); t];
    for f in faults {
        by_round[f.round as usize].push(*f);
    }
    let mut out = vec![0.0; 1usize << t];
    #[allow(clippy::too_many_arguments)]
    fn walk(
        nc: &NoisyCircuit,
        by_round: &[Vec<Fault>],
        mut s: NoisyState<f64>,
        i: usize,
        y: u128,
        p: f64,
        out: &mut [f64],
    ) {
        if i == nc.inst.t {
            out[y as usize] += p;
            return;
        }
        let (p1, flip) = s.round(nc, i, y, &by_round[i], usize::MAX).unwrap();
        for bit in [false, true] {
            let pb = if bit { p1 } else { 1.0 - p1 };
            if pb <= 1e-300 {
                continue;
            }
            let mut c = s.clone();
            c.collapse(bit);
            let rec = u128::from(bit ^ flip) << i;
            walk(nc, by_round, c, i + 1, y | rec, p * pb, out);
        }
    }
    walk(nc, &by_round, NoisyState::new(), 0, 0, 1.0, &mut out);
    out
}

/// The same noisy circuit as a plain [`Circuit`] with the stock stochastic
/// noise ops of [`crate::noise`] (`Depolarize1q` / `XFlip` / `ZFlip`, each
/// sampled independently by [`Circuit::run`]): the independent reference
/// for the trajectory sampler. Measurement `i` is bit `i` of the result.
pub fn reference_circuit(nc: &NoisyCircuit, p: f64) -> Circuit {
    reference_circuit_opts(nc, p, false)
}

/// [`reference_circuit`], optionally with an `Op::Reset` of every ancilla
/// after each control measurement (matches [`run_trajectory_opts`]).
pub fn reference_circuit_opts(nc: &NoisyCircuit, p: f64, reset_ancillas: bool) -> Circuit {
    let inst = &nc.inst;
    let mut c = Circuit::new(nc.nq);
    let noise = |c: &mut Circuit, q: usize| {
        c.ops.push(match nc.kind {
            NoiseKind::Depolarizing => Op::Depolarize1q(q, p),
            NoiseKind::BitFlip => Op::XFlip(q, p),
            NoiseKind::PhaseFlip => Op::ZFlip(q, p),
        });
    };
    let pm = nc.kind.has_prep_meas();
    c.x(1);
    for i in 0..inst.t {
        if i > 0 {
            c.c_if(i - 1, Gate::X(0)); // recycle (true reset: the record equals the qubit)
        }
        if pm {
            c.ops.push(Op::XFlip(0, p));
        }
        c.h(0);
        noise(&mut c, 0);
        for g in &nc.rounds[i].gates {
            c.gate(*g);
            for q in g.qubits() {
                noise(&mut c, q);
            }
        }
        for l in 0..i {
            c.c_if(l, Gate::Phase(0, -PI / ((1u64 << (i - l)) as f64)));
        }
        noise(&mut c, 0);
        c.h(0);
        noise(&mut c, 0);
        if pm {
            c.ops.push(Op::XFlip(0, p));
        }
        c.measure(0);
        if reset_ancillas && i + 1 < inst.t {
            for q in inst.m + 1..nc.nq {
                c.reset(q);
            }
        }
    }
    c
}

/// The multiplicative order of `a` mod `n` (brute force; `n < 2^40`).
pub fn order_of(a: u64, n: u64) -> u64 {
    let mut x = a % n;
    let mut r = 1;
    while x != 1 {
        x = super::mul_mod(x, a, n);
        r += 1;
    }
    r
}

/// Role of a qubit in the windowed layout (for fault attribution).
pub fn windowed_role(lay: &crate::shor::window::WindowLayout, q: usize) -> &'static str {
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
        "?"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn noiseless_trajectory_matches_sliced_distribution() {
        for (n, a) in [(15u64, 7u64), (21, 2)] {
            let inst = Instance::new(n, a, Oracle::Windowed(2));
            let nc = NoisyCircuit::new(&inst, NoiseKind::Depolarizing);
            let d = trajectory_distribution(&nc, &[]);
            let s = super::super::semiclassical_distribution(
                &inst,
                super::super::sliced::SlicedState::<f64>::new(&inst),
                0.0,
            );
            let m = d
                .iter()
                .zip(&s)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f64::max);
            assert!(m < 1e-12, "N={n}: {m:e}");
        }
    }

    #[test]
    fn locations_round_trip_and_sampling() {
        let inst = Instance::new(21, 2, Oracle::Windowed(2));
        for kind in [
            NoiseKind::Depolarizing,
            NoiseKind::BitFlip,
            NoiseKind::PhaseFlip,
        ] {
            let nc = NoisyCircuit::new(&inst, kind);
            let l = nc.num_locations();
            let mut kinds = std::collections::HashMap::new();
            for g in 0..l {
                let (_, s) = nc.site_of(g);
                *kinds.entry(s.kind_name()).or_insert(0u64) += 1;
            }
            let t = inst.t as u64;
            assert_eq!(kinds["h1"], t);
            assert_eq!(
                kinds.get("prep").copied().unwrap_or(0),
                if kind == NoiseKind::PhaseFlip { 0 } else { t }
            );
            let slots: u64 = nc.rounds.iter().map(|r| r.gate_slots()).sum();
            assert_eq!(kinds["gate"], slots);
            let mut rng = StdRng::seed_from_u64(3);
            let f = nc.sample_k(5, &mut rng);
            assert_eq!(f.len(), 5);
            // geometric skipping: mean count p·L
            let p = 3.0 / l as f64;
            let tot: usize = (0..4000).map(|_| nc.sample_p(p, &mut rng).len()).sum();
            let mean = tot as f64 / 4000.0;
            assert!((mean - 3.0).abs() < 0.15, "mean {mean}");
        }
    }
}
