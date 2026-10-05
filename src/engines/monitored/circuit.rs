//! Random monitored brickwork circuits: uniform two-qubit Cliffords in a
//! brickwork, then `T` with probability `p_t` on every qubit, then a `Z`
//! measurement with probability `p_m` on every qubit (Bejan–McLauchlan–Béri
//! PRX Quantum 5, 030332 (2024), uncorrelated monitoring; Li–Chen–Fisher /
//! Gullans–Huse for `p_t = 0`). The circuit randomness (gates, `T` and
//! measurement locations) comes from `circ_seed`; Born outcomes from a
//! separate `born_seed`, so `d(t)` can be compared across outcome records.

use super::{Cliff2, MeasRecord, Mode, Monitored, TooLarge};
use crate::gate::Gate;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// One op of a monitored brickwork layer.
#[derive(Clone, Copy, Debug)]
pub enum MOp {
    /// Two-qubit Clifford: index into the Clifford group list ([`Cliff2::group`]),
    /// then the qubits `(a, b)` it acts on.
    C2(u16, usize, usize),
    /// `T` gate on a qubit.
    T(usize),
    /// `Z` measurement of a qubit.
    M(usize),
}

/// Parameters of the random monitored brickwork.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// Number of qubits (a chain).
    pub n: usize,
    /// Number of layers.
    pub depth: usize,
    /// Probability of a `Z` measurement per qubit per layer.
    pub p_m: f64,
    /// Probability of a `T` gate per qubit per layer.
    pub p_t: f64,
    /// Close the chain into a ring (adds the `(n-1, 0)` gate on odd layers
    /// when `n` is even and greater than 2).
    pub periodic: bool,
}

/// The ops of layer `t`.
pub fn layer<R: Rng + ?Sized>(p: &Params, t: usize, ngroup: usize, rng: &mut R) -> Vec<MOp> {
    let n = p.n;
    let mut ops = Vec::with_capacity(n);
    let off = t % 2;
    let mut i = off;
    while i + 1 < n {
        ops.push(MOp::C2(rng.random_range(0..ngroup) as u16, i, i + 1));
        i += 2;
    }
    if p.periodic && off == 1 && n.is_multiple_of(2) && n > 2 {
        ops.push(MOp::C2(rng.random_range(0..ngroup) as u16, n - 1, 0));
    }
    for q in 0..n {
        if rng.random::<f64>() < p.p_t {
            ops.push(MOp::T(q));
        }
    }
    for q in 0..n {
        if rng.random::<f64>() < p.p_m {
            ops.push(MOp::M(q));
        }
    }
    ops
}

/// Physical gates of an op list (for the reference state vector); `None`
/// marks a measurement.
pub fn to_gates(ops: &[MOp], group: &[Cliff2]) -> Vec<Option<Vec<Gate>>> {
    ops.iter()
        .map(|op| match *op {
            MOp::C2(k, a, b) => Some(group[k as usize].gates(a, b)),
            MOp::T(a) => Some(vec![Gate::T(a)]),
            MOp::M(_) => None,
        })
        .collect()
}

/// Applies one op.
pub fn apply<R: Rng + ?Sized>(
    sim: &mut Monitored,
    op: MOp,
    group: &[Cliff2],
    born: &mut R,
) -> Result<Option<MeasRecord>, TooLarge> {
    match op {
        MOp::C2(k, a, b) => {
            sim.cliff2(&group[k as usize], a, b);
            Ok(None)
        }
        MOp::T(a) => sim.t(a).map(|_| None),
        MOp::M(a) => Ok(Some(sim.measure(a, born, None))),
    }
}

/// Per-layer observables of one trajectory.
#[derive(Clone, Debug, Default)]
pub struct Trajectory {
    /// `d` after every layer.
    pub d: Vec<u32>,
    /// Number of measurement records made.
    pub records: usize,
    /// Layer at which the run stopped because the register would exceed
    /// `max_d` (`None`: it completed).
    pub failed_at: Option<usize>,
}

/// Runs a trajectory; `observe(t, &sim)` is called after every layer.
pub fn run<F: FnMut(usize, &Monitored)>(
    p: &Params,
    mode: Mode,
    max_d: usize,
    circ_seed: u64,
    born_seed: u64,
    group: &[Cliff2],
    mut observe: F,
) -> (Monitored, Trajectory) {
    let mut crng = StdRng::seed_from_u64(circ_seed);
    let mut brng = StdRng::seed_from_u64(born_seed ^ 0x9e37_79b9_7f4a_7c15);
    let mut sim = Monitored::new(p.n, mode, max_d);
    let mut tr = Trajectory::default();
    'outer: for t in 0..p.depth {
        for op in layer(p, t, group.len(), &mut crng) {
            if apply(&mut sim, op, group, &mut brng).is_err() {
                tr.failed_at = Some(t);
                break 'outer;
            }
        }
        tr.d.push(sim.d() as u32);
        observe(t, &sim);
    }
    tr.records = sim.records.len();
    (sim, tr)
}
