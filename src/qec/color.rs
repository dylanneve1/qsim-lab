//! Triangular 6.6.6 colour code memory circuits with one auxiliary qubit per
//! plaquette and a per-plaquette CNOT schedule.
//!
//! Layout: identical to `color-code-stim` (S.-H. Lee) as used by Kishony &
//! Fowler, "Color code off-the-hook" (arXiv:2603.28852): with
//! `L = 3(d-1)/2`, rows `y = 0..=L`, columns `x = 2y, 2y+4, .., 4L-2y`; the
//! site `(x, y)` holds a plaquette (auxiliary qubit) iff
//! `((x/2 - y)/2) mod 3 == pos(y)` with `pos = [2, 0, 1][y mod 3]` and colour
//! `[g, b, r][y mod 3]`, otherwise a data qubit. A plaquette's data qubits
//! sit at the six offsets [`OFFSETS`] (positions `a..f`); boundary
//! plaquettes miss some. The Z logical is the `y = 0` row (the red boundary).
//!
//! A *schedule* gives, per plaquette and offset position, the CNOT time step
//! `1..=6`. One syndrome round (Kishony–Fowler's parallel construction):
//! `CX data->anc` at steps 1..6, `M` anc, `RX` anc, `CX anc->data` at steps
//! 1..6 again, `MX` anc, `R` anc (not after the last round). Z detectors
//! every round, X detectors from the second round on, final data `M`.
//!
//! Noise is written as explicit ops (so [`crate::io::stim::to_stim`] exports
//! exactly the circuit that is sampled); only the readout flip lives in the
//! returned [`NoiseModel`].
#![allow(clippy::needless_range_loop)]

use crate::circuit::{Circuit, Op};
use crate::engines::stabilizer::symphase::SymPhaseSampler;
use crate::gate::Gate;
use crate::noise::NoiseModel;

/// Data-qubit offsets of a plaquette, positions `a..f`.
pub const OFFSETS: [(i32, i32); 6] = [(-2, 1), (2, 1), (4, 0), (2, -1), (-2, -1), (-4, 0)];

/// One plaquette (stabilizer face).
#[derive(Clone, Debug, PartialEq)]
pub struct Plaquette {
    /// Column of the plaquette's auxiliary site on the layout grid.
    pub x: i32,
    /// Row of the plaquette's auxiliary site (`0..=L`, `y = 0` is the red
    /// boundary).
    pub y: i32,
    /// 0 = red, 1 = green, 2 = blue.
    pub color: u8,
    /// Data-qubit index at each offset position, if present.
    pub data: [Option<usize>; 6],
}

impl Plaquette {
    /// Number of data qubits the plaquette acts on (6 in the bulk, fewer on the
    /// boundary).
    pub fn weight(&self) -> usize {
        self.data.iter().filter(|d| d.is_some()).count()
    }
}

/// Triangular colour code of odd distance `d`.
#[derive(Clone, Debug)]
pub struct ColorCode {
    /// Code distance (odd, ≥ 3).
    pub d: usize,
    /// `(x, y)` of each data qubit.
    pub data: Vec<(i32, i32)>,
    /// All plaquettes with at least one data qubit, in row-major site order
    /// (`y`, then `x`); plaquette `i` uses auxiliary qubit `num_data() + i` in
    /// the memory circuits.
    pub plaquettes: Vec<Plaquette>,
}

/// Time step (`1..=6` in Kishony-Fowler's design; up to [`MAX_STEP`] for
/// deeper schedules) per plaquette and offset position; entries for absent
/// positions are ignored. Each half of a round has as many CNOT layers as the
/// largest step used (at least 6).
pub type ColorSchedule = Vec<[u8; 6]>;

/// Largest supported CNOT time step (layers per half-round).
pub const MAX_STEP: usize = 12;

/// Circuit-level noise models (all explicit ops except the readout flip).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColorNoise {
    /// `DEPOLARIZE2(p)` after every CNOT, nothing else (K–F "noisy CNOT").
    Cnot(f64),
    /// Uniform depolarizing (as K–F's tqec model): `DEPOLARIZE2(p)` after
    /// CNOTs, `DEPOLARIZE1(p)` on every qubit idle in a moment, readout flips
    /// `p`, `X_ERROR(p)` after `R`, `Z_ERROR(p)` after `RX`.
    Uniform(f64),
}

/// A built memory experiment.
#[derive(Clone, Debug)]
pub struct ColorMemory {
    /// The noisy memory circuit: data qubits `0..nd`, plaquette auxiliaries
    /// `nd..nd + np`, then any flag qubits.
    pub circuit: Circuit,
    /// Readout-flip noise only (`p_meas`; 0 for [`ColorNoise::Cnot`]); every other
    /// noise channel is an explicit op in `circuit`.
    pub noise: NoiseModel,
    /// Detectors as lists of measurement-record indices whose parity is
    /// deterministic without noise.
    pub detectors: Vec<Vec<usize>>,
    /// The single logical observable: final data measurements on the `y = 0` row.
    pub observables: Vec<Vec<usize>>,
    /// Per detector: (plaquette, is_x_type, round).
    pub detector_info: Vec<(usize, bool, usize)>,
    /// Memory basis: false = Z (|0>, Z logical), true = X (|+>, X logical).
    pub x_basis: bool,
    /// Per detector: whether it is a flag-qubit detector.
    pub flag_detector: Vec<bool>,
}

impl ColorCode {
    /// Builds the layout for odd distance `d ≥ 3` (panics otherwise). Data
    /// qubits are indexed in row-major site order (`y`, then `x`).
    pub fn new(d: usize) -> Self {
        assert!(d % 2 == 1 && d >= 3, "odd d >= 3");
        let l = (3 * (d - 1) / 2) as i32;
        let mut data = Vec::new();
        let mut plaq_sites = Vec::new();
        for y in 0..=l {
            let (color, pos) = match y % 3 {
                0 => (1u8, 2),
                1 => (2u8, 0),
                _ => (0u8, 1),
            };
            let mut x = 2 * y;
            while x <= 4 * l - 2 * y {
                if ((x / 2 - y) / 2).rem_euclid(3) != pos {
                    data.push((x, y));
                } else {
                    plaq_sites.push((x, y, color));
                }
                x += 4;
            }
        }
        let find = |x: i32, y: i32| data.iter().position(|&p| p == (x, y));
        let plaquettes = plaq_sites
            .into_iter()
            .map(|(x, y, color)| {
                let mut dq = [None; 6];
                for (k, (dx, dy)) in OFFSETS.iter().enumerate() {
                    dq[k] = find(x + dx, y + dy);
                }
                Plaquette {
                    x,
                    y,
                    color,
                    data: dq,
                }
            })
            .filter(|p| p.weight() > 0)
            .collect();
        ColorCode {
            d,
            data,
            plaquettes,
        }
    }

    /// Number of data qubits.
    pub fn num_data(&self) -> usize {
        self.data.len()
    }

    /// Data qubits of the Z (and X) logical: the `y = 0` row.
    pub fn logical_support(&self) -> Vec<usize> {
        (0..self.data.len())
            .filter(|&i| self.data[i].1 == 0)
            .collect()
    }

    /// Same schedule for every plaquette of a colour.
    pub fn uniform_schedule(&self, by_color: [[u8; 6]; 3]) -> ColorSchedule {
        self.plaquettes
            .iter()
            .map(|p| by_color[p.color as usize])
            .collect()
    }

    /// `(step, data qubit, plaquette, plaquette')` for every time-step
    /// collision, plus any plaquette using a step twice or out of range.
    pub fn collisions(&self, s: &ColorSchedule) -> Vec<String> {
        let mut out = Vec::new();
        let mut used: Vec<[Option<usize>; MAX_STEP + 1]> =
            vec![[None; MAX_STEP + 1]; self.data.len()];
        for (pi, p) in self.plaquettes.iter().enumerate() {
            let mut seen = [false; MAX_STEP + 1];
            for k in 0..6 {
                if let Some(q) = p.data[k] {
                    let t = s[pi][k] as usize;
                    if !(1..=MAX_STEP).contains(&t) {
                        out.push(format!("plaquette {pi} pos {k}: step {t} out of range"));
                        continue;
                    }
                    if seen[t] {
                        out.push(format!("plaquette {pi}: step {t} used twice"));
                    }
                    seen[t] = true;
                    if let Some(o) = used[q][t] {
                        out.push(format!(
                            "step {t}: data {q} used by plaquettes {o} and {pi}"
                        ));
                    }
                    used[q][t] = Some(pi);
                }
            }
        }
        out
    }

    /// Builds the Z-basis memory experiment (`rounds >= 1`).
    pub fn memory(&self, s: &ColorSchedule, rounds: usize, noise: ColorNoise) -> ColorMemory {
        self.memory_basis(s, rounds, noise, false)
    }

    /// Builds a memory experiment in the Z basis (`x_basis = false`: data
    /// start in |0>, Z logical) or the X basis (data start in |+> via `RX`,
    /// final `MX`, X logical on the same `y = 0` row). Detectors of the
    /// memory's own type start in round 0, the other type from round 1.
    pub fn memory_basis(
        &self,
        s: &ColorSchedule,
        rounds: usize,
        noise: ColorNoise,
        x_basis: bool,
    ) -> ColorMemory {
        self.memory_flagged(s, &[], rounds, noise, x_basis)
    }

    /// Plaquettes that touch a boundary data qubit (a data qubit in fewer
    /// than three plaquettes): `3d - 6` of them for `d >= 5`.
    pub fn boundary_plaquettes(&self) -> Vec<bool> {
        let mut deg = vec![0usize; self.data.len()];
        for p in &self.plaquettes {
            for q in p.data.iter().flatten() {
                deg[*q] += 1;
            }
        }
        self.plaquettes
            .iter()
            .map(|p| p.data.iter().flatten().any(|&q| deg[q] < 3))
            .collect()
    }

    /// Flag-CNOT slots for the flagged plaquettes (see [`Self::memory_flagged`]):
    /// per plaquette `None` (unflagged) or `Some((s1, s2))` with `s1 < t_2`
    /// and `s2 > t_{w-1}` (`t_k` = the plaquette's k-th data-CNOT step), each
    /// a step at which the auxiliary is idle. Step `0` is an extra CNOT layer
    /// before the data layers, step `T + 1` one after them (`T` = data layers
    /// per half). Idle in-schedule steps are preferred (no extra depth), then
    /// the step closest to the protected window.
    pub fn flag_slots(&self, s: &ColorSchedule, flagged: &[bool]) -> Vec<Option<(u8, u8)>> {
        let t = self.num_steps(s) as u8;
        self.plaquettes
            .iter()
            .enumerate()
            .map(|(pi, p)| {
                if !flagged.get(pi).copied().unwrap_or(false) {
                    return None;
                }
                let mut ts: Vec<u8> = (0..6)
                    .filter(|&k| p.data[k].is_some())
                    .map(|k| s[pi][k])
                    .collect();
                ts.sort_unstable();
                let w = ts.len();
                assert!(w >= 4, "flag on a plaquette of weight {w}");
                let busy = |x: u8| ts.contains(&x);
                // s1 in [0, t_2): latest idle in-schedule step, else 0
                let s1 = (1..ts[1]).rev().find(|&x| !busy(x)).unwrap_or(0);
                // s2 in (t_{w-1}, T+1]: earliest idle in-schedule step, else
                // T+1. (Not just after t_{w-2}: an X/Z error on the auxiliary
                // right after the second flag CNOT, e.g. from that CNOT's own
                // DEPOLARIZE2, would otherwise spread unflagged to the data
                // qubits of the remaining two CNOTs.)
                let s2 = (ts[w - 2] + 1..=t).find(|&x| !busy(x)).unwrap_or(t + 1);
                Some((s1, s2))
            })
            .collect()
    }

    /// Data-CNOT layers per half-round: the largest step used, at least 6.
    pub fn num_steps(&self, s: &ColorSchedule) -> usize {
        self.plaquettes
            .iter()
            .enumerate()
            .flat_map(|(pi, p)| {
                (0..6)
                    .filter(move |&k| p.data[k].is_some())
                    .map(move |k| s[pi][k] as usize)
            })
            .max()
            .unwrap_or(6)
            .max(6)
    }

    /// Memory experiment with one flag qubit on each plaquette with
    /// `flagged[pi]` (empty slice: no flags; the circuit is then exactly
    /// [`Self::memory_basis`]'s).
    ///
    /// A flagged plaquette's flag qubit is reset and measured together with
    /// its auxiliary and serves both halves of a round:
    /// - Z half (auxiliary in |0>, `CX data->anc`): flag in |+> (`RX`),
    ///   `CX flag->anc` at the two flag slots, flag measured in X. It catches
    ///   a Z error on the auxiliary between the slots (a Z hook; X-type
    ///   sector).
    /// - X half (auxiliary in |+>, `CX anc->data`): flag in |0>,
    ///   `CX anc->flag` at the same slots, flag measured in Z. It catches an X
    ///   error on the auxiliary between the slots (an X hook; Z-type sector).
    ///
    /// Slots come from [`Self::flag_slots`]: every auxiliary error that
    /// would spread to `2..=w-2` data qubits (all multi-qubit hooks up to
    /// stabilizer equivalence), including errors from the flag CNOTs
    /// themselves, flips the flag. Every flag measurement is its own
    /// detector. Flag CNOTs carry the same `DEPOLARIZE2` as data CNOTs.
    pub fn memory_flagged(
        &self,
        s: &ColorSchedule,
        flagged: &[bool],
        rounds: usize,
        noise: ColorNoise,
        x_basis: bool,
    ) -> ColorMemory {
        assert!(rounds >= 1);
        assert_eq!(s.len(), self.plaquettes.len());
        let nd = self.data.len();
        let np = self.plaquettes.len();
        let slots = self.flag_slots(s, flagged);
        let flag_of: Vec<Option<usize>> = {
            let mut k = 0;
            slots
                .iter()
                .map(|sl| {
                    sl.map(|_| {
                        k += 1;
                        nd + np + k - 1
                    })
                })
                .collect()
        };
        let nf = flag_of.iter().flatten().count();
        let n = nd + np + nf;
        let anc = |pi: usize| nd + pi;
        let (p2, p_idle, p_meas, p_reset) = match noise {
            ColorNoise::Cnot(p) => (p, 0.0, 0.0, 0.0),
            ColorNoise::Uniform(p) => (p, p, p, p),
        };
        let mut ops: Vec<Op> = Vec::new();
        // a moment: ops on a set of qubits; idle noise for the others
        let idle = |ops: &mut Vec<Op>, busy: &[bool]| {
            if p_idle > 0.0 {
                for (q, &b) in busy.iter().enumerate() {
                    if !b {
                        ops.push(Op::Depolarize1q(q, p_idle));
                    }
                }
            }
        };
        let nsteps = self.num_steps(s);
        assert!(nsteps <= MAX_STEP, "step {nsteps} > MAX_STEP");
        // layer index = step (0 = pre-layer, nsteps + 1 = post-layer);
        // entries (other qubit, plaquette): a data qubit or a flag qubit
        let mut cx_layers: Vec<Vec<(usize, usize)>> = vec![Vec::new(); nsteps + 2];
        for (pi, p) in self.plaquettes.iter().enumerate() {
            for k in 0..6 {
                if let Some(q) = p.data[k] {
                    cx_layers[s[pi][k] as usize].push((q, pi));
                }
            }
        }
        let mut is_flag = vec![false; n];
        for (pi, sl) in slots.iter().enumerate() {
            if let (Some((s1, s2)), Some(f)) = (sl, flag_of[pi]) {
                cx_layers[*s1 as usize].push((f, pi));
                cx_layers[*s2 as usize].push((f, pi));
                is_flag[f] = true;
            }
        }
        let used_layers: Vec<usize> = (0..nsteps + 2)
            .filter(|&l| (1..=nsteps).contains(&l) || !cx_layers[l].is_empty())
            .collect();
        let reset_moment = |ops: &mut Vec<Op>, qs: &[(usize, bool)]| {
            let mut busy = vec![false; n];
            for &(q, x_basis) in qs {
                ops.push(Op::Reset(q));
                if x_basis {
                    ops.push(Op::Gate(Gate::H(q)));
                }
                if p_reset > 0.0 {
                    ops.push(if x_basis {
                        Op::ZFlip(q, p_reset)
                    } else {
                        Op::XFlip(q, p_reset)
                    });
                }
                busy[q] = true;
            }
            idle(ops, &busy);
        };
        let measure_moment = |ops: &mut Vec<Op>, qs: &[(usize, bool)]| {
            let mut busy = vec![false; n];
            for &(q, x_basis) in qs {
                if x_basis {
                    ops.push(Op::Gate(Gate::H(q)));
                }
                ops.push(Op::Measure(q));
                if x_basis {
                    ops.push(Op::Gate(Gate::H(q)));
                }
                busy[q] = true;
            }
            idle(ops, &busy);
        };
        let cx_moment = |ops: &mut Vec<Op>, layer: &[(usize, usize)], to_anc: bool| {
            let mut busy = vec![false; n];
            for &(q, pi) in layer {
                let (c, t) = if to_anc { (q, anc(pi)) } else { (anc(pi), q) };
                ops.push(Op::Gate(Gate::Cnot(c, t)));
                if p2 > 0.0 {
                    ops.push(Op::Depolarize2q(c, t, p2));
                }
                busy[c] = true;
                busy[t] = true;
            }
            idle(ops, &busy);
        };
        let ancs: Vec<usize> = (0..np).map(anc).collect();
        let flags: Vec<usize> = flag_of.iter().flatten().copied().collect();
        let all: Vec<usize> = (0..n).collect();
        let mut num_meas = 0usize;
        let mut z_rec = vec![vec![0usize; np]; rounds];
        let mut x_rec = vec![vec![0usize; np]; rounds];
        // flag records per round: Z-half flags (X-measured), X-half flags (Z-measured)
        let mut fz_rec = vec![vec![0usize; nf]; rounds];
        let mut fx_rec = vec![vec![0usize; nf]; rounds];
        // initial reset: data in the memory basis, Z-half flags in |+>
        let init: Vec<(usize, bool)> = all
            .iter()
            .map(|&q| (q, (x_basis && q < nd) || is_flag[q]))
            .collect();
        reset_moment(&mut ops, &init);
        // Z half: auxiliaries |0>, flags |+>; X half: auxiliaries |+>, flags |0>
        let z_half_q: Vec<(usize, bool)> = ancs
            .iter()
            .map(|&q| (q, false))
            .chain(flags.iter().map(|&q| (q, true)))
            .collect();
        let x_half_q: Vec<(usize, bool)> = ancs
            .iter()
            .map(|&q| (q, true))
            .chain(flags.iter().map(|&q| (q, false)))
            .collect();
        for r in 0..rounds {
            if r > 0 {
                reset_moment(&mut ops, &z_half_q);
            }
            for &l in &used_layers {
                cx_moment(&mut ops, &cx_layers[l], true);
            }
            measure_moment(&mut ops, &z_half_q);
            for (pi, rec) in z_rec[r].iter_mut().enumerate() {
                *rec = num_meas + pi;
            }
            for (k, rec) in fz_rec[r].iter_mut().enumerate() {
                *rec = num_meas + np + k;
            }
            num_meas += np + nf;
            reset_moment(&mut ops, &x_half_q);
            for &l in &used_layers {
                cx_moment(&mut ops, &cx_layers[l], false);
            }
            // the last X measurement shares its moment with the final data
            // measurement (as in Kishony-Fowler's circuit)
            let mut xm = x_half_q.clone();
            if r + 1 == rounds {
                xm.extend((0..nd).map(|q| (q, x_basis)));
            }
            measure_moment(&mut ops, &xm);
            for (pi, rec) in x_rec[r].iter_mut().enumerate() {
                *rec = num_meas + pi;
            }
            for (k, rec) in fx_rec[r].iter_mut().enumerate() {
                *rec = num_meas + np + k;
            }
            num_meas += np + nf;
        }
        let flag_plaq: Vec<usize> = (0..np).filter(|&pi| flag_of[pi].is_some()).collect();
        let data_base = num_meas;
        let mut detectors = Vec::new();
        let mut info = Vec::new();
        let mut flag_det: Vec<bool> = Vec::new();
        // own type (deterministic from round 0) first, then the other type
        let (own, other) = if x_basis {
            (&x_rec, &z_rec)
        } else {
            (&z_rec, &x_rec)
        };
        for r in 0..rounds {
            for pi in 0..np {
                let mut v = vec![own[r][pi]];
                if r > 0 {
                    v.push(own[r - 1][pi]);
                }
                detectors.push(v);
                info.push((pi, x_basis, r));
            }
            if r > 0 {
                for pi in 0..np {
                    detectors.push(vec![other[r][pi], other[r - 1][pi]]);
                    info.push((pi, !x_basis, r));
                }
            }
            // flag detectors (single deterministic measurements): the X-half
            // flag sees X errors (Z-type sector), the Z-half flag Z errors
            for (k, &pi) in flag_plaq.iter().enumerate() {
                detectors.push(vec![fx_rec[r][k]]);
                info.push((pi, false, r));
                detectors.push(vec![fz_rec[r][k]]);
                info.push((pi, true, r));
                flag_det.resize(detectors.len() - 2, false);
                flag_det.extend([true, true]);
            }
        }
        for (pi, p) in self.plaquettes.iter().enumerate() {
            let mut v: Vec<usize> = p.data.iter().flatten().map(|&q| data_base + q).collect();
            v.push(own[rounds - 1][pi]);
            detectors.push(v);
            info.push((pi, x_basis, rounds));
        }
        flag_det.resize(detectors.len(), false);
        let observables = vec![self
            .logical_support()
            .iter()
            .map(|&q| data_base + q)
            .collect()];
        ColorMemory {
            circuit: Circuit { num_qubits: n, ops },
            noise: NoiseModel {
                p_meas,
                ..NoiseModel::none()
            },
            detectors,
            observables,
            detector_info: info,
            x_basis,
            flag_detector: flag_det,
        }
    }
}

/// One distinct single-fault signature of a circuit's detector error model.
#[derive(Clone, Debug, PartialEq)]
pub struct DemEntry {
    /// Flipped detectors (sorted).
    pub detectors: Vec<u32>,
    /// Flipped observables (bit mask).
    pub observables: u64,
    /// Probability that an odd number of the merged faults fire.
    pub p: f64,
}

/// The circuit-derived detector error model of `m`, read off the SymPhase
/// sampler (each outcome of each noise location is one fault; faults with the
/// same signature are merged as independent events). Panics if a detector or
/// observable is not deterministic.
pub fn circuit_dem(
    circuit: &Circuit,
    noise: &NoiseModel,
    detectors: &[Vec<usize>],
    observables: &[Vec<usize>],
) -> Vec<DemEntry> {
    use crate::engines::stabilizer::symphase::VarDist;
    use std::collections::HashMap;
    let sets: Vec<Vec<usize>> = detectors.iter().chain(observables).cloned().collect();
    let s = SymPhaseSampler::new(circuit, noise)
        .expect("Clifford circuit")
        .with_parities(&sets);
    assert!(
        s.reference().iter().all(|&b| !b),
        "non-deterministic reference"
    );
    let nd = detectors.len();
    let mut cols: Vec<Vec<u32>> = vec![Vec::new(); s.num_vars()];
    for j in 0..s.num_measurements() {
        for &v in s.row(j) {
            cols[v as usize].push(j as u32);
        }
    }
    let mut map: HashMap<(Vec<u32>, u64), f64> = HashMap::new();
    for g in s.groups() {
        assert!(
            g.dist != VarDist::Coin,
            "a detector depends on a random outcome"
        );
        for (pat, p) in g.dist.outcomes() {
            if pat == 0 {
                continue;
            }
            let mut sig: Vec<u32> = Vec::new();
            for k in 0..g.dist.len() {
                if pat >> k & 1 == 1 {
                    for &r in &cols[g.first as usize + k] {
                        match sig.binary_search(&r) {
                            Ok(i) => {
                                sig.remove(i);
                            }
                            Err(i) => sig.insert(i, r),
                        }
                    }
                }
            }
            if sig.is_empty() {
                continue;
            }
            let mut obs = 0u64;
            let dets: Vec<u32> = sig
                .into_iter()
                .filter(|&r| {
                    if (r as usize) >= nd {
                        obs ^= 1 << (r as usize - nd);
                        false
                    } else {
                        true
                    }
                })
                .collect();
            let e = map.entry((dets, obs)).or_insert(0.0);
            *e = *e * (1.0 - p) + p * (1.0 - *e);
        }
    }
    let mut v: Vec<DemEntry> = map
        .into_iter()
        .map(|((detectors, observables), p)| DemEntry {
            detectors,
            observables,
            p,
        })
        .collect();
    v.sort_by(|a, b| (&a.detectors, a.observables).cmp(&(&b.detectors, b.observables)));
    v
}

/// The Kishony–Fowler colour-dependent schedule as published in the paper's
/// companion code (`results/zero_collision_schedules.csv`, index 0, the one
/// `benchmark_circuits.py` loads): steps for positions `a..f`.
pub const KF_SCHEDULE: [[u8; 6]; 3] = [
    [1, 3, 2, 5, 4, 6], // red
    [5, 1, 4, 3, 6, 2], // green
    [1, 5, 3, 6, 2, 4], // blue
];

/// Lee et al.'s uniform "tri-optimal" schedule (same for every colour).
pub const TRI_OPTIMAL: [u8; 6] = [2, 3, 6, 5, 4, 1];

/// The memory-basis sector of a memory experiment's DEM (detectors of the
/// memory's own type and the observable; mechanisms merged by sector
/// signature). Named after the default Z-basis memory.
#[derive(Clone, Debug)]
pub struct ZSector {
    /// Number of memory-type detectors.
    pub num_detectors: usize,
    /// Per merged mechanism: memory-type detectors it flips (indices into this
    /// sector, `0..num_detectors`).
    pub dets: Vec<Vec<u32>>,
    /// Per merged mechanism: whether it flips the observable.
    pub obs: Vec<bool>,
    /// Whether some fault with exactly this Z-sector signature flips no
    /// X-type detector (needed to certify that a Z-sector logical is a
    /// full-DEM logical of the same weight).
    pub pure: Vec<bool>,
    /// Plaquette of each Z-sector detector.
    pub plaquette: Vec<usize>,
}

impl ColorMemory {
    /// Projects the circuit's DEM onto the memory-basis sector: drops the other
    /// type's detectors, discards mechanisms that then do nothing, and merges
    /// mechanisms with the same `(detectors, observable)` signature (their
    /// probabilities are not kept).
    pub fn z_sector(&self) -> ZSector {
        let dem = circuit_dem(
            &self.circuit,
            &self.noise,
            &self.detectors,
            &self.observables,
        );
        let mut zmap = vec![u32::MAX; self.detectors.len()];
        let mut plaquette = Vec::new();
        for (i, inf) in self.detector_info.iter().enumerate() {
            if inf.1 == self.x_basis {
                zmap[i] = plaquette.len() as u32;
                plaquette.push(inf.0);
            }
        }
        let mut index = std::collections::HashMap::new();
        let (mut dets, mut obs, mut pure) = (Vec::new(), Vec::new(), Vec::new());
        for e in &dem {
            let zs: Vec<u32> = e
                .detectors
                .iter()
                .filter_map(|&i| {
                    let z = zmap[i as usize];
                    (z != u32::MAX).then_some(z)
                })
                .collect();
            let ob = e.observables & 1 == 1;
            if zs.is_empty() && !ob {
                continue;
            }
            let is_pure = zs.len() == e.detectors.len();
            let k = *index.entry((zs.clone(), ob)).or_insert_with(|| {
                dets.push(zs);
                obs.push(ob);
                pure.push(false);
                dets.len() - 1
            });
            pure[k] |= is_pure;
        }
        ZSector {
            num_detectors: plaquette.len(),
            dets,
            obs,
            pure,
            plaquette,
        }
    }

    /// Exact Z-memory circuit distance and number of minimum-weight logicals
    /// (see [`crate::qec::distance`]); `certified` is true if the example
    /// logical lifts to the full DEM (it then equals the full-DEM distance,
    /// since dropping X detectors can only lower the minimum).
    pub fn z_distance(
        &self,
        count_cap: u64,
        node_limit: u64,
    ) -> (crate::qec::distance::MinLogical, bool) {
        let z = self.z_sector();
        let r = crate::qec::distance::min_logical(
            z.num_detectors,
            &z.dets,
            &z.obs,
            64,
            count_cap,
            node_limit,
        );
        let cert = !r.example.is_empty() && r.example.iter().all(|&j| z.pure[j]);
        (r, cert)
    }
}

/// Parses a schedule spec: `kf`, `tri`, or a file with one line per
/// plaquette `t_a t_b t_c t_d t_e t_f [F]` (absent positions: anything, e.g.
/// 0; a trailing `F` flags the plaquette). A `+bflags` suffix (e.g.
/// `kf+bflags`) additionally flags every boundary-touching plaquette.
/// Returns the schedule and the per-plaquette flag mask.
pub fn parse_schedule_spec(cc: &ColorCode, spec: &str) -> (ColorSchedule, Vec<bool>) {
    let (base, bflags) = match spec.strip_suffix("+bflags") {
        Some(b) => (b, true),
        None => (spec, false),
    };
    let np = cc.plaquettes.len();
    let (s, mut f) = match base {
        "kf" => (cc.uniform_schedule(KF_SCHEDULE), vec![false; np]),
        "tri" => (cc.uniform_schedule([TRI_OPTIMAL; 3]), vec![false; np]),
        path => {
            let text = std::fs::read_to_string(path).expect("schedule file");
            let mut s = ColorSchedule::new();
            let mut f = Vec::new();
            for l in text.lines().filter(|l| !l.trim().is_empty()) {
                let tok: Vec<&str> = l.split_whitespace().collect();
                let v: Vec<u8> = tok[..6].iter().map(|t| t.parse().unwrap()).collect();
                s.push([v[0], v[1], v[2], v[3], v[4], v[5]]);
                f.push(tok.get(6).is_some_and(|t| *t == "F"));
            }
            assert_eq!(s.len(), np, "schedule lines != plaquettes");
            (s, f)
        }
    };
    if bflags {
        for (x, b) in f.iter_mut().zip(cc.boundary_plaquettes()) {
            *x |= b;
        }
    }
    (s, f)
}

/// Resource count of a (possibly flagged) memory circuit, per round.
#[derive(Clone, Debug, PartialEq)]
pub struct ColorResources {
    /// Number of data qubits.
    pub data: usize,
    /// Number of plaquette auxiliary qubits.
    pub aux: usize,
    /// Number of flag qubits.
    pub flags: usize,
    /// CNOT layers per round (both halves).
    pub cnot_layers: usize,
    /// CNOTs per round (both halves).
    pub cnots: usize,
}

impl ColorCode {
    /// Qubits, CNOT layers and CNOT count per round of
    /// [`Self::memory_flagged`] with this schedule and flag mask.
    pub fn resources(&self, s: &ColorSchedule, flagged: &[bool]) -> ColorResources {
        let t = self.num_steps(s);
        let slots = self.flag_slots(s, flagged);
        let pre = slots.iter().flatten().any(|sl| sl.0 == 0);
        let post = slots.iter().flatten().any(|sl| sl.1 as usize == t + 1);
        let nf = slots.iter().flatten().count();
        let data_cx: usize = self.plaquettes.iter().map(|p| p.weight()).sum();
        ColorResources {
            data: self.data.len(),
            aux: self.plaquettes.len(),
            flags: nf,
            cnot_layers: 2 * (t + pre as usize + post as usize),
            cnots: 2 * (data_cx + 2 * nf),
        }
    }
}
