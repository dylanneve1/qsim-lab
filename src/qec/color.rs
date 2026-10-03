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
//! Noise is written as explicit ops (so [`crate::stim_io::to_stim`] exports
//! exactly the circuit that is sampled); only the readout flip lives in the
//! returned [`NoiseModel`].
#![allow(clippy::needless_range_loop)]

use crate::circuit::{Circuit, Op};
use crate::gate::Gate;
use crate::noise::NoiseModel;
use crate::stabilizer::symphase::SymPhaseSampler;

/// Data-qubit offsets of a plaquette, positions `a..f`.
pub const OFFSETS: [(i32, i32); 6] = [(-2, 1), (2, 1), (4, 0), (2, -1), (-2, -1), (-4, 0)];

/// One plaquette (stabilizer face).
#[derive(Clone, Debug, PartialEq)]
pub struct Plaquette {
    pub x: i32,
    pub y: i32,
    /// 0 = red, 1 = green, 2 = blue.
    pub color: u8,
    /// Data-qubit index at each offset position, if present.
    pub data: [Option<usize>; 6],
}

impl Plaquette {
    pub fn weight(&self) -> usize {
        self.data.iter().filter(|d| d.is_some()).count()
    }
}

/// Triangular colour code of odd distance `d`.
#[derive(Clone, Debug)]
pub struct ColorCode {
    pub d: usize,
    /// `(x, y)` of each data qubit.
    pub data: Vec<(i32, i32)>,
    pub plaquettes: Vec<Plaquette>,
}

/// Time step (`1..=6`) per plaquette and offset position; entries for absent
/// positions are ignored.
pub type ColorSchedule = Vec<[u8; 6]>;

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
    pub circuit: Circuit,
    pub noise: NoiseModel,
    pub detectors: Vec<Vec<usize>>,
    pub observables: Vec<Vec<usize>>,
    /// Per detector: (plaquette, is_x_type, round).
    pub detector_info: Vec<(usize, bool, usize)>,
}

impl ColorCode {
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
        let mut used: Vec<[Option<usize>; 7]> = vec![[None; 7]; self.data.len()];
        for (pi, p) in self.plaquettes.iter().enumerate() {
            let mut seen = [false; 7];
            for k in 0..6 {
                if let Some(q) = p.data[k] {
                    let t = s[pi][k] as usize;
                    if !(1..=6).contains(&t) {
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

    /// Builds the memory experiment (Z basis, `rounds >= 1`).
    pub fn memory(&self, s: &ColorSchedule, rounds: usize, noise: ColorNoise) -> ColorMemory {
        assert!(rounds >= 1);
        assert_eq!(s.len(), self.plaquettes.len());
        let nd = self.data.len();
        let np = self.plaquettes.len();
        let n = nd + np;
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
        let mut cx_layers: Vec<Vec<(usize, usize)>> = vec![Vec::new(); 7]; // (data, plaquette)
        for (pi, p) in self.plaquettes.iter().enumerate() {
            for k in 0..6 {
                if let Some(q) = p.data[k] {
                    cx_layers[s[pi][k] as usize].push((q, pi));
                }
            }
        }
        let reset_moment = |ops: &mut Vec<Op>, qs: &[usize], x_basis: bool| {
            let mut busy = vec![false; n];
            for &q in qs {
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
        let all: Vec<usize> = (0..n).collect();
        let mut num_meas = 0usize;
        let mut z_rec = vec![vec![0usize; np]; rounds];
        let mut x_rec = vec![vec![0usize; np]; rounds];
        reset_moment(&mut ops, &all, false);
        for r in 0..rounds {
            if r > 0 {
                reset_moment(&mut ops, &ancs, false);
            }
            for layer in &cx_layers[1..=6] {
                cx_moment(&mut ops, layer, true);
            }
            let zm: Vec<(usize, bool)> = ancs.iter().map(|&q| (q, false)).collect();
            measure_moment(&mut ops, &zm);
            for (pi, rec) in z_rec[r].iter_mut().enumerate() {
                *rec = num_meas + pi;
            }
            num_meas += np;
            reset_moment(&mut ops, &ancs, true);
            for layer in &cx_layers[1..=6] {
                cx_moment(&mut ops, layer, false);
            }
            // the last X measurement shares its moment with the final data
            // measurement (as in Kishony-Fowler's circuit)
            let mut xm: Vec<(usize, bool)> = ancs.iter().map(|&q| (q, true)).collect();
            if r + 1 == rounds {
                xm.extend((0..nd).map(|q| (q, false)));
            }
            measure_moment(&mut ops, &xm);
            for (pi, rec) in x_rec[r].iter_mut().enumerate() {
                *rec = num_meas + pi;
            }
            num_meas += np;
        }
        let data_base = num_meas;
        let mut detectors = Vec::new();
        let mut info = Vec::new();
        for r in 0..rounds {
            for pi in 0..np {
                let mut v = vec![z_rec[r][pi]];
                if r > 0 {
                    v.push(z_rec[r - 1][pi]);
                }
                detectors.push(v);
                info.push((pi, false, r));
            }
            if r > 0 {
                for pi in 0..np {
                    detectors.push(vec![x_rec[r][pi], x_rec[r - 1][pi]]);
                    info.push((pi, true, r));
                }
            }
        }
        for (pi, p) in self.plaquettes.iter().enumerate() {
            let mut v: Vec<usize> = p.data.iter().flatten().map(|&q| data_base + q).collect();
            v.push(z_rec[rounds - 1][pi]);
            detectors.push(v);
            info.push((pi, false, rounds));
        }
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
    use crate::stabilizer::symphase::VarDist;
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

/// The Z sector of a memory experiment's DEM (Z-type detectors and the
/// observable; mechanisms merged by Z-sector signature).
#[derive(Clone, Debug)]
pub struct ZSector {
    pub num_detectors: usize,
    pub dets: Vec<Vec<u32>>,
    pub obs: Vec<bool>,
    /// Whether some fault with exactly this Z-sector signature flips no
    /// X-type detector (needed to certify that a Z-sector logical is a
    /// full-DEM logical of the same weight).
    pub pure: Vec<bool>,
    /// Plaquette of each Z-sector detector.
    pub plaquette: Vec<usize>,
}

impl ColorMemory {
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
            if !inf.1 {
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
