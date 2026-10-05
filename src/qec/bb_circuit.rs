//! Syndrome-extraction memory circuits for weight-6 two-block codes
//! (`|A| = |B| = 3`), generalising the depth-7 CNOT schedule of Bravyi et
//! al. (Nature 627, 778 (2024)).
//!
//! # Layout and schedule
//!
//! Qubits: data `L` = `0..N`, data `R` = `N..2N`, X-check ancillas
//! `2N..3N`, Z-check ancillas `3N..4N` (`N = |G|`). A schedule assigns, to
//! each of 7 CNOT layers, at most one *term* per check type:
//!
//! * X-check `g`, term `t`: `t < 3` -> `L(g + a_t)`, else `R(g + b_{t-3})`
//!   (CNOT ancilla -> data);
//! * Z-check `h`, term `t`: `t < 3` -> `L(h - b_t)`, else `R(h - a_{t-3})`
//!   (CNOT data -> ancilla).
//!
//! Every term is a bijection between checks and one data block, so a layer
//! is conflict-free iff the X and Z terms used in it act on different
//! blocks. [`IBM_SCHEDULE`] is the schedule of Bravyi et al.
//! (`sX = [idle, 1, 4, 3, 5, 0, 2]`, `sZ = [3, 5, 0, 1, 2, 4, idle]`).
//!
//! # Cycle and noise
//!
//! Each cycle: reset all ancillas (X ancillas in `|+>`), 7 CNOT layers,
//! measure all ancillas (X ancillas in the X basis). Noise ("uniform", as in
//! Bravyi et al.): two-qubit depolarizing `p` after every CNOT, single-qubit
//! depolarizing `p` on every qubit idle in a CNOT layer, preparation flips
//! `p`, readout flips `p`. Basis-change Hadamards are part of the noiseless
//! preparation / measurement. Data qubits are initialised and finally
//! measured in the memory basis.
#![allow(clippy::needless_range_loop)]

use super::bicycle::{bits_of, logical_basis, TwoBlockCode};
use crate::circuit::{Circuit, Op};
use crate::gate::Gate;
use crate::noise::NoiseModel;

/// A depth-7 schedule: per layer, the X-check term and the Z-check term
/// (`None` = idle).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BbSchedule {
    /// X-check term used in each of the 7 CNOT layers (0–2 = `L` via `a_t`,
    /// 3–5 = `R` via `b_{t-3}`).
    pub sx: [Option<u8>; 7],
    /// Z-check term used in each of the 7 CNOT layers (0–2 = `L` via `b_t`,
    /// 3–5 = `R` via `a_{t-3}`).
    pub sz: [Option<u8>; 7],
}

/// The schedule of Bravyi et al.
pub const IBM_SCHEDULE: BbSchedule = BbSchedule {
    sx: [None, Some(1), Some(4), Some(3), Some(5), Some(0), Some(2)],
    sz: [Some(3), Some(5), Some(0), Some(1), Some(2), Some(4), None],
};

impl BbSchedule {
    /// Compact text form `"<sx>/<sz>"`: one character per layer, the term digit
    /// or `-` for idle (e.g. `-143502/350124-` for [`IBM_SCHEDULE`]).
    pub fn spec(&self) -> String {
        let f = |s: &[Option<u8>; 7]| {
            s.iter()
                .map(|t| t.map_or("-".to_string(), |t| t.to_string()))
                .collect::<String>()
        };
        format!("{}/{}", f(&self.sx), f(&self.sz))
    }

    /// Parses `"-143502/350124-"` style specs (as printed by [`Self::spec`],
    /// seven characters per half, no spaces) or `ibm`.
    pub fn parse(s: &str) -> Self {
        if s == "ibm" {
            return IBM_SCHEDULE;
        }
        let (a, b) = s.split_once('/').expect("sx/sz");
        let p = |x: &str| {
            let v: Vec<Option<u8>> = x.chars().map(|c| c.to_digit(10).map(|d| d as u8)).collect();
            let arr: [Option<u8>; 7] = v.try_into().expect("7 layers");
            arr
        };
        BbSchedule { sx: p(a), sz: p(b) }
    }
}

/// Data qubit touched by X-check `g` through term `t`.
pub fn x_term_qubit(c: &TwoBlockCode, g: usize, t: usize) -> usize {
    let nn = c.order();
    if t < 3 {
        add(c, g, c.a[t], false)
    } else {
        nn + add(c, g, c.b[t - 3], false)
    }
}

/// Data qubit touched by Z-check `h` through term `t`.
pub fn z_term_qubit(c: &TwoBlockCode, h: usize, t: usize) -> usize {
    let nn = c.order();
    if t < 3 {
        add(c, h, c.b[t], true)
    } else {
        nn + add(c, h, c.a[t - 3], true)
    }
}

fn add(c: &TwoBlockCode, g: usize, (a, b): (usize, usize), minus: bool) -> usize {
    let (l, m) = (c.l, c.m);
    let (i, j) = (g / m, g % m);
    if minus {
        ((i + l - a) % l) * m + (j + m - b) % m
    } else {
        ((i + a) % l) * m + (j + b) % m
    }
}

/// True if no data qubit is used twice in a layer and every term is used
/// exactly once per check type.
pub fn schedule_well_formed(s: &BbSchedule) -> bool {
    let mut seen_x = [false; 6];
    let mut seen_z = [false; 6];
    for layer in 0..7 {
        if let Some(t) = s.sx[layer] {
            if t > 5 || seen_x[t as usize] {
                return false;
            }
            seen_x[t as usize] = true;
        }
        if let Some(t) = s.sz[layer] {
            if t > 5 || seen_z[t as usize] {
                return false;
            }
            seen_z[t as usize] = true;
        }
        if let (Some(x), Some(z)) = (s.sx[layer], s.sz[layer]) {
            if (x < 3) == (z < 3) {
                return false;
            }
        }
    }
    seen_x.iter().all(|&b| b) && seen_z.iter().all(|&b| b)
}

/// True if the schedule measures the stabilizers: for every X-check /
/// Z-check pair, the number of shared data qubits on which the X-check's
/// CNOT comes first is even.
pub fn schedule_valid(c: &TwoBlockCode, s: &BbSchedule) -> bool {
    if !schedule_well_formed(s) {
        return false;
    }
    let nn = c.order();
    let mut tx = [0usize; 6];
    let mut tz = [0usize; 6];
    for layer in 0..7 {
        if let Some(t) = s.sx[layer] {
            tx[t as usize] = layer;
        }
        if let Some(t) = s.sz[layer] {
            tz[t as usize] = layer;
        }
    }
    // per data qubit: the Z-check and time touching it, per term
    let mut ztouch: Vec<Vec<(usize, usize)>> = vec![Vec::new(); 2 * nn];
    for h in 0..nn {
        for t in 0..6 {
            ztouch[z_term_qubit(c, h, t)].push((h, tz[t]));
        }
    }
    let mut cnt = vec![0u8; nn];
    for g in 0..nn {
        cnt.iter_mut().for_each(|x| *x = 0);
        for t in 0..6 {
            let q = x_term_qubit(c, g, t);
            for &(h, tzh) in &ztouch[q] {
                if tx[t] < tzh {
                    cnt[h] ^= 1;
                }
            }
        }
        if cnt.iter().any(|&x| x != 0) {
            return false;
        }
    }
    true
}

/// All valid schedules of the IBM shape (X idle in layer 0, Z idle in
/// layer 6).
pub fn valid_schedules(c: &TwoBlockCode) -> Vec<BbSchedule> {
    fn perms(v: &mut Vec<u8>, k: usize, out: &mut Vec<Vec<u8>>) {
        if k == v.len() {
            out.push(v.clone());
            return;
        }
        for i in k..v.len() {
            v.swap(k, i);
            perms(v, k + 1, out);
            v.swap(k, i);
        }
    }
    let mut ps = Vec::new();
    perms(&mut (0..6).collect(), 0, &mut ps);
    let mut out = Vec::new();
    for px in &ps {
        for pz in &ps {
            let mut s = BbSchedule {
                sx: [None; 7],
                sz: [None; 7],
            };
            for i in 0..6 {
                s.sx[i + 1] = Some(px[i]);
                s.sz[i] = Some(pz[i]);
            }
            if schedule_well_formed(&s) && schedule_valid(c, &s) {
                out.push(s);
            }
        }
    }
    out
}

/// A built memory experiment.
#[derive(Clone, Debug)]
pub struct BbMemory {
    /// The noisy memory circuit (all qubits start in |0⟩; ancilla layout as in the
    /// module docs). Gate noise is explicit in the ops.
    pub circuit: Circuit,
    /// Readout noise: measurement flips with probability `p` (all other noise
    /// channels are disabled; gate noise is in the circuit ops).
    pub noise: NoiseModel,
    /// Detectors as lists of measurement-record indices whose parity is
    /// deterministic in the absence of noise.
    pub detectors: Vec<Vec<usize>>,
    /// One observable per logical qubit (`k` of them).
    pub observables: Vec<Vec<usize>>,
    /// Per detector: true if it belongs to the memory-basis sector (Z-type
    /// checks for a Z-basis memory), the sector that is decoded.
    pub in_sector: Vec<bool>,
}

/// Builds a `rounds`-cycle memory experiment in the Z (`x_basis = false`) or
/// X basis with uniform circuit noise `p`.
pub fn memory(c: &TwoBlockCode, s: &BbSchedule, rounds: usize, p: f64, x_basis: bool) -> BbMemory {
    assert!(rounds >= 1);
    assert_eq!((c.a.len(), c.b.len()), (3, 3), "weight-6 codes only");
    let nn = c.order();
    let nd = 2 * nn;
    let n = 4 * nn;
    let xa = |g: usize| nd + g;
    let za = |h: usize| nd + nn + h;
    let mut ops: Vec<Op> = Vec::new();
    // initial data reset
    for q in 0..nd {
        ops.push(Op::Reset(q));
        if x_basis {
            ops.push(Op::Gate(Gate::H(q)));
        }
        if p > 0.0 {
            ops.push(if x_basis {
                Op::ZFlip(q, p)
            } else {
                Op::XFlip(q, p)
            });
        }
    }
    let mut nmeas = 0usize;
    let mut xrec = vec![vec![0usize; nn]; rounds];
    let mut zrec = vec![vec![0usize; nn]; rounds];
    for r in 0..rounds {
        for g in 0..nn {
            ops.push(Op::Reset(xa(g)));
            ops.push(Op::Gate(Gate::H(xa(g))));
            if p > 0.0 {
                ops.push(Op::ZFlip(xa(g), p));
            }
            ops.push(Op::Reset(za(g)));
            if p > 0.0 {
                ops.push(Op::XFlip(za(g), p));
            }
        }
        for layer in 0..7 {
            let mut busy = vec![false; n];
            if let Some(t) = s.sx[layer] {
                for g in 0..nn {
                    let q = x_term_qubit(c, g, t as usize);
                    ops.push(Op::Gate(Gate::Cnot(xa(g), q)));
                    if p > 0.0 {
                        ops.push(Op::Depolarize2q(xa(g), q, p));
                    }
                    busy[xa(g)] = true;
                    busy[q] = true;
                }
            }
            if let Some(t) = s.sz[layer] {
                for h in 0..nn {
                    let q = z_term_qubit(c, h, t as usize);
                    ops.push(Op::Gate(Gate::Cnot(q, za(h))));
                    if p > 0.0 {
                        ops.push(Op::Depolarize2q(q, za(h), p));
                    }
                    busy[za(h)] = true;
                    busy[q] = true;
                }
            }
            if p > 0.0 {
                for q in 0..n {
                    if !busy[q] {
                        ops.push(Op::Depolarize1q(q, p));
                    }
                }
            }
        }
        for g in 0..nn {
            ops.push(Op::Gate(Gate::H(xa(g))));
            ops.push(Op::Measure(xa(g)));
            xrec[r][g] = nmeas;
            nmeas += 1;
        }
        for h in 0..nn {
            ops.push(Op::Measure(za(h)));
            zrec[r][h] = nmeas;
            nmeas += 1;
        }
    }
    let mut drec = vec![0usize; nd];
    for q in 0..nd {
        if x_basis {
            ops.push(Op::Gate(Gate::H(q)));
        }
        ops.push(Op::Measure(q));
        drec[q] = nmeas;
        nmeas += 1;
    }
    let (own, other) = if x_basis {
        (&xrec, &zrec)
    } else {
        (&zrec, &xrec)
    };
    let (hx, hz) = (c.hx(), c.hz());
    let hown = if x_basis { &hx } else { &hz };
    let mut detectors = Vec::new();
    let mut in_sector = Vec::new();
    for r in 0..rounds {
        for i in 0..nn {
            let mut v = vec![own[r][i]];
            if r > 0 {
                v.push(own[r - 1][i]);
            }
            detectors.push(v);
            in_sector.push(true);
        }
        if r > 0 {
            for i in 0..nn {
                detectors.push(vec![other[r][i], other[r - 1][i]]);
                in_sector.push(false);
            }
        }
    }
    for i in 0..nn {
        let mut v: Vec<usize> = hown.row_support(i).iter().map(|&q| drec[q]).collect();
        v.push(own[rounds - 1][i]);
        detectors.push(v);
        in_sector.push(true);
    }
    // observables: logicals of the memory-basis type measured on the data
    let logs = if x_basis {
        logical_basis(&hz, &hx)
    } else {
        logical_basis(&hx, &hz)
    };
    let observables = logs
        .iter()
        .map(|l| bits_of(l).iter().map(|&q| drec[q]).collect())
        .collect();
    BbMemory {
        circuit: Circuit { num_qubits: n, ops },
        noise: NoiseModel {
            p_meas: p,
            ..NoiseModel::none()
        },
        detectors,
        observables,
        in_sector,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qec::color::circuit_dem;

    #[test]
    fn ibm_schedule_valid_on_published_codes() {
        for (l, m, a, b) in [
            (6, 6, "x^3+y+y^2", "y^3+x+x^2"),
            (15, 3, "x^9+y+y^2", "1+x^2+x^7"),
            (12, 6, "x^3+y+y^2", "y^3+x+x^2"),
        ] {
            let c = TwoBlockCode::parse(l, m, a, b);
            assert!(schedule_valid(&c, &IBM_SCHEDULE), "{l} {m}");
        }
    }

    #[test]
    fn memory_detectors_deterministic_and_dem_has_no_undetectable_logical() {
        let c = TwoBlockCode::parse(6, 6, "x^3+y+y^2", "y^3+x+x^2");
        for x_basis in [false, true] {
            let mem = memory(&c, &IBM_SCHEDULE, 2, 0.001, x_basis);
            assert_eq!(mem.observables.len(), 12);
            // circuit_dem panics on non-deterministic detectors/observables
            let dem = circuit_dem(&mem.circuit, &mem.noise, &mem.detectors, &mem.observables);
            assert!(dem
                .iter()
                .all(|e| !e.detectors.is_empty() || e.observables == 0));
        }
    }

    #[test]
    fn invalid_schedule_is_rejected_by_both_checks() {
        let c = TwoBlockCode::parse(6, 6, "x^3+y+y^2", "y^3+x+x^2");
        let all = valid_schedules(&c);
        assert!(all.contains(&IBM_SCHEDULE));
        // a well-formed but invalid schedule makes the circuit non-deterministic
        let mut found = None;
        'o: for i in 0..6u8 {
            for j in 0..6u8 {
                let mut s = IBM_SCHEDULE;
                s.sz.swap(i as usize, j as usize);
                if schedule_well_formed(&s) && !schedule_valid(&c, &s) {
                    found = Some(s);
                    break 'o;
                }
            }
        }
        let s = found.expect("an invalid well-formed variant");
        let mem = memory(&c, &s, 2, 0.0, false);
        let r = std::panic::catch_unwind(|| {
            circuit_dem(&mem.circuit, &mem.noise, &mem.detectors, &mem.observables)
        });
        assert!(r.is_err(), "invalid schedule should give random detectors");
    }
}
