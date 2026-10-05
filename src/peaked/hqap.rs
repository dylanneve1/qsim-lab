//! Heuristic peaked circuits built by identity insertion (HQAP), reduced to their peaked core.
//!
//! The construction (arXiv:2510.25838) trains a shallow peaked circuit `R ▷ P`,
//! inserts identity blocks `U ▷ U†` between `R` and `P`, and obfuscates the
//! result (wire swaps, re-trained parameters, re-synthesised patches). Every
//! inserted block acts as a permutation of the wires, so the circuit equals
//! `R ▷ Π ▷ P` up to the obfuscation's small approximation error, and `R ▷ Π ▷ P`
//! is shallow enough to contract exactly. [`solve`] does this in four steps:
//!
//! 1. **Units.** The circuit is a list of two-qubit units
//!    `u a; u b; cz a,b; u a; u b` ([`units`]).
//! 2. **Parts.** The file lists the circuit part by part, each part as layers of
//!    disjoint units that end in a tail of short layers ([`parts`]). `R` is the
//!    first part and `P` the last.
//! 3. **Wire permutations.** A unit the obfuscation left untouched reappears in
//!    the inverse half of its block, reversed and inverted, so the rotation
//!    between two consecutive CZs on a wire in `U` is the exact inverse of a
//!    rotation in `U†`. Such pairs, when unique in the circuit, are *anchors*
//!    ([`anchors`]); each pairs a wire of `U` with the wire that carries the same
//!    qubit in `U†`, and the units touching an anchor pair the neighbouring
//!    wires too. A part with many anchors is an inserted block, and these pairs
//!    give its wire permutation ([`block_permutations`]).
//! 4. **Core and peak.** Everything between `R` and `P` is replaced by the
//!    composed permutation and [`super::find_peak`] is run on `R ▷ Π ▷ P`.
//!    Re-synthesised patches can straddle the boundary between `R` and the
//!    middle (or the middle and `P`) in the file order, so units at those
//!    boundaries are moved into the core one at a time when that raises the
//!    exact peak probability (the trained core is the most peaked).
//!
//! The target bitstring is never used. Lab notebook:
//! `research/simulability/peaked-circuits.md`.

use super::{find_peak, Peak};
use crate::circuit::{Circuit, Op, SimError};
use crate::engines::tn::TnOptions;
use crate::gate::{Gate, Mat2};
use num_complex::Complex64;
use std::collections::{BTreeSet, HashMap};

/// A part with at least this many anchors is an inserted identity block.
pub const MIN_ANCHORS: usize = 50;

/// One two-qubit unit `u a; u b; cz a,b; u a; u b`.
#[derive(Clone, Debug)]
pub struct Unit {
    /// First qubit of the CZ.
    pub a: usize,
    /// Second qubit of the CZ.
    pub b: usize,
    /// The five gates of the unit, in circuit order.
    pub ops: [Op; 5],
    /// Rotation applied to `a` before the CZ.
    pub pre_a: Mat2,
    /// Rotation applied to `b` before the CZ.
    pub pre_b: Mat2,
    /// Rotation applied to `a` after the CZ.
    pub post_a: Mat2,
    /// Rotation applied to `b` after the CZ.
    pub post_b: Mat2,
}

fn err(msg: impl Into<String>) -> SimError {
    SimError::QasmError(format!("hqap: {}", msg.into()))
}

/// Splits `c` into its units; any other shape is an error.
pub fn units(c: &Circuit) -> Result<Vec<Unit>, SimError> {
    if !c.ops.len().is_multiple_of(5) {
        return Err(err(
            "the circuit is not a sequence of 'u; u; cz; u; u' units",
        ));
    }
    let one = |op: &Op| -> Option<(usize, Mat2)> {
        match op {
            Op::Gate(g @ Gate::U(q, ..)) => Some((*q, g.matrix_1q()?)),
            _ => None,
        }
    };
    let mut out = Vec::with_capacity(c.ops.len() / 5);
    for (i, w) in c.ops.chunks(5).enumerate() {
        let bad = || err(format!("unit {i} is not 'u; u; cz; u; u'"));
        let (q0, m0) = one(&w[0]).ok_or_else(bad)?;
        let (q1, m1) = one(&w[1]).ok_or_else(bad)?;
        let (a, b) = match w[2] {
            Op::Gate(Gate::Cz(a, b)) => (a, b),
            _ => return Err(bad()),
        };
        let (q3, m3) = one(&w[3]).ok_or_else(bad)?;
        let (q4, m4) = one(&w[4]).ok_or_else(bad)?;
        let pick = |qa: usize, ma: Mat2, mb: Mat2| if qa == a { (ma, mb) } else { (mb, ma) };
        if ([q0, q1] != [a, b] && [q0, q1] != [b, a]) || ([q3, q4] != [a, b] && [q3, q4] != [b, a])
        {
            return Err(bad());
        }
        let (pre_a, pre_b) = pick(q0, m0, m1);
        let (post_a, post_b) = pick(q3, m3, m4);
        out.push(Unit {
            a,
            b,
            ops: [w[0], w[1], w[2], w[3], w[4]],
            pre_a,
            pre_b,
            post_a,
            post_b,
        });
    }
    Ok(out)
}

/// The parts of the file: greedy layers of disjoint units in file order; a new
/// part starts where a long layer (>= 9 units) follows a short one (<= 3 units).
/// Returns inclusive unit-index ranges.
pub fn parts(units: &[Unit]) -> Vec<(usize, usize)> {
    let mut layers: Vec<Vec<usize>> = vec![vec![]];
    let mut used = BTreeSet::new();
    for (k, u) in units.iter().enumerate() {
        if used.contains(&u.a) || used.contains(&u.b) {
            layers.push(vec![]);
            used.clear();
        }
        layers.last_mut().unwrap().push(k);
        used.insert(u.a);
        used.insert(u.b);
    }
    let mut out = Vec::new();
    let mut start = 0;
    for i in 1..layers.len() {
        if layers[i].len() >= 9 && layers[i - 1].len() <= 3 {
            out.push((start, layers[i][0] - 1));
            start = layers[i][0];
        }
    }
    out.push((start, units.len() - 1));
    out
}

/// An exact-inverse pair of rotations: on wire `left_wire` between units
/// `left.0` and `left.1`, and on wire `right_wire` between `right.0` and `right.1`.
#[derive(Clone, Copy, Debug)]
pub struct Anchor {
    /// Wire of the earlier rotation.
    pub left_wire: usize,
    /// Units before and after the earlier rotation.
    pub left: (usize, usize),
    /// Wire of the later rotation.
    pub right_wire: usize,
    /// Units before and after the later rotation.
    pub right: (usize, usize),
}

fn mul(x: &Mat2, y: &Mat2) -> Mat2 {
    let mut m = [[Complex64::new(0.0, 0.0); 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            m[i][j] = x[i][0] * y[0][j] + x[i][1] * y[1][j];
        }
    }
    m
}

/// SU(2) quaternion with a canonical sign, as an exact-match key (1e-8 grid).
fn key(m: &Mat2, inverse: bool) -> ([i64; 4], bool) {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let s = det.sqrt();
    let (u00, u01) = (m[0][0] / s, m[0][1] / s);
    let mut q = [u00.re, u00.im, u01.re, u01.im];
    if inverse {
        q[1] = -q[1];
        q[2] = -q[2];
        q[3] = -q[3];
    }
    if let Some(&x) = q.iter().find(|x| x.abs() > 1e-6) {
        if x < 0.0 {
            q.iter_mut().for_each(|v| *v = -*v);
        }
    }
    let identity = (q[0].abs() - 1.0).abs() < 1e-9;
    (q.map(|v| (v * 1e8).round() as i64), identity)
}

/// All anchors: rotations that are unique in the circuit and whose exact
/// inverse occurs exactly once, later in the circuit.
pub fn anchors(n: usize, units: &[Unit]) -> Vec<Anchor> {
    // (wire, unit before, unit after, rotation between them)
    let mut segs: Vec<(usize, usize, usize, Mat2)> = Vec::new();
    let mut last: Vec<Option<(usize, Mat2)>> = vec![None; n];
    for (k, u) in units.iter().enumerate() {
        for (q, pre, post) in [(u.a, u.pre_a, u.post_a), (u.b, u.pre_b, u.post_b)] {
            if let Some((kp, post_prev)) = last[q] {
                segs.push((q, kp, k, mul(&pre, &post_prev)));
            }
            last[q] = Some((k, post));
        }
    }
    let mut count: HashMap<[i64; 4], Vec<usize>> = HashMap::new();
    for (i, s) in segs.iter().enumerate() {
        count.entry(key(&s.3, false).0).or_default().push(i);
    }
    let mut out = Vec::new();
    for (i, s) in segs.iter().enumerate() {
        let (k, identity) = key(&s.3, false);
        if identity || count[&k].len() != 1 {
            continue;
        }
        let ki = key(&s.3, true).0;
        if let Some(js) = count.get(&ki) {
            if js.len() == 1 && js[0] != i {
                let t = &segs[js[0]];
                if s.2 <= t.1 {
                    out.push(Anchor {
                        left_wire: s.0,
                        left: (s.1, s.2),
                        right_wire: t.0,
                        right: (t.1, t.2),
                    });
                }
            }
        }
    }
    out
}

/// An inserted identity block and its wire permutation.
#[derive(Clone, Debug)]
pub struct Block {
    /// Index of the part that holds the block.
    pub part: usize,
    /// Anchors whose seam lies in that part.
    pub anchors: usize,
    /// `perm[a]` = the wire that carries, after the block, the qubit that was on wire `a`.
    pub perm: Vec<usize>,
    /// Wires whose image came from anchors on that wire itself (the rest from neighbouring units).
    pub anchored_wires: usize,
}

/// The identity blocks (parts with at least [`MIN_ANCHORS`] anchors) and their
/// wire permutations, from votes: an anchor votes `left_wire -> right_wire`
/// (weight 10), and the units on either side of it vote for their other wires
/// (weight 1). Each wire takes its most voted image; ties, conflicts, or more
/// than one wire without votes are errors.
pub fn block_permutations(
    n: usize,
    units: &[Unit],
    parts: &[(usize, usize)],
    anchors: &[Anchor],
) -> Result<Vec<Block>, SimError> {
    let part_of = |k: usize| {
        parts
            .iter()
            .position(|&(lo, hi)| lo <= k && k <= hi)
            .unwrap()
    };
    let mut by_part: HashMap<usize, Vec<&Anchor>> = HashMap::new();
    for a in anchors {
        by_part
            .entry(part_of((a.left.1 + a.right.0) / 2))
            .or_default()
            .push(a);
    }
    let other = |k: usize, w: usize| {
        if units[k].a == w {
            units[k].b
        } else {
            units[k].a
        }
    };
    let mut blocks = Vec::new();
    let mut keys: Vec<usize> = by_part.keys().copied().collect();
    keys.sort();
    for p in keys {
        let anc = &by_part[&p];
        if anc.len() < MIN_ANCHORS {
            continue;
        }
        let mut votes: Vec<HashMap<usize, usize>> = vec![HashMap::new(); n];
        let mut anchored = BTreeSet::new();
        for a in anc {
            *votes[a.left_wire].entry(a.right_wire).or_default() += 10;
            anchored.insert(a.left_wire);
            // the unit after the earlier rotation mirrors the unit before the later one, and vice versa
            for (kl, kr) in [(a.left.1, a.right.0), (a.left.0, a.right.1)] {
                let (x, y) = (other(kl, a.left_wire), other(kr, a.right_wire));
                *votes[x].entry(y).or_default() += 1;
            }
        }
        let mut perm = vec![usize::MAX; n];
        for w in 0..n {
            let mut v: Vec<(usize, usize)> = votes[w].iter().map(|(&t, &c)| (c, t)).collect();
            v.sort_by(|x, y| y.cmp(x));
            if let Some(&(c, t)) = v.first() {
                if v.len() > 1 && v[1].0 == c {
                    return Err(err(format!("block in part {p}: tied images for wire {w}")));
                }
                perm[w] = t;
            }
        }
        let mut seen = vec![false; n];
        for &t in perm.iter().filter(|&&t| t != usize::MAX) {
            if seen[t] {
                return Err(err(format!("block in part {p}: two wires map to wire {t}")));
            }
            seen[t] = true;
        }
        let free_src: Vec<usize> = (0..n).filter(|&w| perm[w] == usize::MAX).collect();
        let free_dst: Vec<usize> = (0..n).filter(|&t| !seen[t]).collect();
        match free_src.len() {
            0 => {}
            1 => perm[free_src[0]] = free_dst[0],
            m => {
                return Err(err(format!(
                    "block in part {p}: {m} wires without evidence"
                )))
            }
        }
        blocks.push(Block {
            part: p,
            anchors: anc.len(),
            perm,
            anchored_wires: anchored.len(),
        });
    }
    Ok(blocks)
}

/// The core `R ▷ Π ▷ P` as a circuit, and the label map: output wire `w` of the
/// original circuit carries the core's qubit `labels[w]`.
pub fn core_circuit(
    n: usize,
    units: &[Unit],
    r_units: &BTreeSet<usize>,
    p_units: &BTreeSet<usize>,
    blocks: &[Block],
) -> (Circuit, Vec<usize>) {
    // after a block the qubit that was on wire a is on wire perm[a]: wire w holds labels[perm^-1(w)]
    let mut labels: Vec<usize> = (0..n).collect();
    for b in blocks {
        let mut inv = vec![0; n];
        for (a, &t) in b.perm.iter().enumerate() {
            inv[t] = a;
        }
        labels = (0..n).map(|w| labels[inv[w]]).collect();
    }
    let mut c = Circuit::new(n);
    for &k in r_units {
        c.ops.extend_from_slice(&units[k].ops);
    }
    for &k in p_units {
        for op in &units[k].ops {
            c.ops.push(match *op {
                Op::Gate(Gate::U(q, t, p, l)) => Op::Gate(Gate::U(labels[q], t, p, l)),
                Op::Gate(Gate::Cz(a, b)) => Op::Gate(Gate::Cz(labels[a], labels[b])),
                other => other,
            });
        }
    }
    (c, labels)
}

/// The outcome of [`solve`].
#[derive(Clone, Debug)]
pub struct Solution {
    /// The peak of the original circuit (`bits[w]` = bit of output wire `w`).
    pub bits: Vec<bool>,
    /// Exact peak probability of the core `R ▷ Π ▷ P`.
    pub core_probability: f64,
    /// The core's peak (in core labels).
    pub core_peak: Peak,
    /// Units of the core.
    pub core_units: usize,
    /// One line per step.
    pub log: Vec<String>,
}

/// Runs the four steps on an HQAP circuit (see the module documentation).
pub fn solve(c: &Circuit, opts: &TnOptions) -> Result<Solution, SimError> {
    let n = c.num_qubits;
    let mut log = Vec::new();
    let us = units(c)?;
    log.push(format!(
        "units: {} two-qubit units on {} qubits",
        us.len(),
        n
    ));
    let ps = parts(&us);
    if ps.len() < 3 {
        return Err(err(format!(
            "expected R, a middle and P; found {} part(s)",
            ps.len()
        )));
    }
    log.push(format!(
        "parts: {}",
        ps.iter()
            .map(|(a, b)| format!("{a}..{b}"))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    let anc = anchors(n, &us);
    let blocks = block_permutations(n, &us, &ps, &anc)?;
    log.push(format!(
        "anchors: {} exact-inverse rotation pairs",
        anc.len()
    ));
    for b in &blocks {
        log.push(format!(
            "identity block in part {} ({}..{}): {} anchors, permutation of {} wires ({} from their own anchors)",
            b.part,
            ps[b.part].0,
            ps[b.part].1,
            b.anchors,
            b.perm.iter().enumerate().filter(|&(a, &t)| a != t).count(),
            b.anchored_wires
        ));
    }
    let last = ps.len() - 1;
    let mut r_units: BTreeSet<usize> = (ps[0].0..=ps[0].1).collect();
    let mut p_units: BTreeSet<usize> = (ps[last].0..=ps[last].1).collect();
    let peak_of =
        |r: &BTreeSet<usize>, p: &BTreeSet<usize>| -> Result<(Peak, Vec<usize>), SimError> {
            let (core, labels) = core_circuit(n, &us, r, p, &blocks);
            Ok((find_peak(&core, opts)?, labels))
        };
    let (mut best, mut labels) = peak_of(&r_units, &p_units)?;
    log.push(format!(
        "core R ▷ Π ▷ P: {} units, peak probability {:.4}",
        r_units.len() + p_units.len(),
        best.probability
    ));
    // boundary refinement: candidates are the first units of part 1 and the last units of part last-1
    let on_wires = |k: usize| [us[k].a, us[k].b];
    loop {
        let mut cands: Vec<(bool, usize)> = Vec::new();
        let (lo1, hi1) = ps[1];
        for k in lo1..=hi1 {
            if r_units.contains(&k) {
                continue;
            }
            let ws = on_wires(k);
            if (lo1..k).all(|j| r_units.contains(&j) || !on_wires(j).iter().any(|w| ws.contains(w)))
            {
                cands.push((true, k));
            }
        }
        let (lo2, hi2) = ps[last - 1];
        for k in lo2..=hi2 {
            if p_units.contains(&k) {
                continue;
            }
            let ws = on_wires(k);
            if (k + 1..=hi2)
                .all(|j| p_units.contains(&j) || !on_wires(j).iter().any(|w| ws.contains(w)))
            {
                cands.push((false, k));
            }
        }
        let mut improved: Option<(f64, bool, usize, Peak, Vec<usize>)> = None;
        for &(to_r, k) in &cands {
            let (mut r2, mut p2) = (r_units.clone(), p_units.clone());
            if to_r {
                r2.insert(k);
            } else {
                p2.insert(k);
            }
            let (pk, lb) = peak_of(&r2, &p2)?;
            let bar = improved.as_ref().map_or(best.probability, |x| x.0);
            if pk.probability > bar {
                improved = Some((pk.probability, to_r, k, pk, lb));
            }
        }
        match improved {
            Some((p, to_r, k, pk, lb)) => {
                if to_r {
                    r_units.insert(k);
                } else {
                    p_units.insert(k);
                }
                log.push(format!(
                    "boundary: unit {k} moved into {}: peak probability {:.4} -> {:.4}",
                    if to_r { "R" } else { "P" },
                    best.probability,
                    p
                ));
                best = pk;
                labels = lb;
            }
            None => break,
        }
    }
    let bits: Vec<bool> = (0..n).map(|w| best.bits[labels[w]]).collect();
    Ok(Solution {
        bits,
        core_probability: best.probability,
        core_units: r_units.len() + p_units.len(),
        core_peak: best,
        log,
    })
}
