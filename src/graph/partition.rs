//! Subgraph partitioning across engines.
//!
//! Beyond independent components: cut the qubits into `A | B` so that only a
//! few *controlled* gates (`CZ`, `CNOT`, `CPhase`) cross, run each side on
//! the engine that suits it (e.g. an MPS for a long shallow chain, a dense
//! state vector for a small deep core) and join the sides by a path sum
//! over the cut. A controlled gate `C-U` with control `c` and target `t` is
//!
//! ```text
//! C-U = |0><0|_c ⊗ 1 + |1><1|_c ⊗ U_t = ½ Σ_{s,u ∈ {0,1}} (-1)^{s·u} Z_c^s ⊗ U_t^u
//! ```
//!
//! so with `c` cut gates each side is run for `2^c` *unitary* insertion
//! patterns (an engine never sees a projector) and
//!
//! ```text
//! <x|ψ> = 2^{-c} Σ_{s,u} (-1)^{s·u} <x_A|A_s> <x_B|B_u>
//! ```
//!
//! (`s·u` taken per cut gate; the side that holds the control gets the `Z`
//! insertions). The sum is a Walsh–Hadamard transform of the `B` vector, so
//! the join costs `c 2^c` per amplitude on top of the `2^{c+1}` side runs.
//!
//! The partition is chosen by pricing candidates with the planner:
//! `2^c × (predicted A + predicted B)` against the best single engine.
//! Candidates come from greedy min-cut growth of `B` from several seeds
//! (add the qubit with the most gates into `B`, ties to fewer gates out).

use crate::circuit::{Circuit, Op, SimError};
use crate::gate::Gate;
use crate::planner::{self, Engine, PlanRequest, PlannerConfig};
use num_complex::Complex64;

/// A priced partition.
#[derive(Clone, Debug)]
pub struct CutPlan {
    /// `true` = qubit in side A.
    pub in_a: Vec<bool>,
    /// Number of cut gates.
    pub cut: usize,
    /// Engines and predicted seconds of one run of each side.
    pub side_a: (Engine, f64),
    /// Engine and predicted seconds of one run of side B.
    pub side_b: (Engine, f64),
    /// `2^c × (side A + side B)`.
    pub predicted_secs: f64,
    /// The planner's best single engine for the whole circuit (`None`: no
    /// engine fits the budget).
    pub single: Option<(Engine, f64)>,
    /// Seconds spent choosing the partition.
    pub plan_secs: f64,
}

fn is_cuttable(g: &Gate) -> bool {
    matches!(g, Gate::Cz(..) | Gate::Cnot(..) | Gate::CPhase(..))
}

/// Number of cut gates of a partition (`None` if some gate crossing it is
/// not a controlled gate the path sum handles).
pub fn cut_size(c: &Circuit, in_a: &[bool]) -> Option<usize> {
    let mut k = 0;
    for g in c.gates() {
        let qs = g.qubits();
        let a = qs.iter().filter(|&&q| in_a[q]).count();
        if a != 0 && a != qs.len() {
            if !is_cuttable(g) {
                return None;
            }
            k += 1;
        }
    }
    Some(k)
}

/// Side circuits for one insertion pattern (bit `k` of `pat` = choice at
/// cut gate `k`), with qubits relabelled to `0..|side|`.
fn side_circuit(c: &Circuit, in_a: &[bool], side_a: bool, pat: usize) -> Circuit {
    let mut local = vec![usize::MAX; c.num_qubits];
    let mut m = 0;
    for (q, &a) in in_a.iter().enumerate() {
        if a == side_a {
            local[q] = m;
            m += 1;
        }
    }
    let mut out = Circuit::new(m);
    let mut k = 0;
    for op in &c.ops {
        let Op::Gate(g) = op else { continue };
        let qs = g.qubits();
        let mine = qs.iter().filter(|&&q| in_a[q] == side_a).count();
        if mine == qs.len() {
            out.gate(crate::engines::hsf::map_gate(*g, |q| local[q]));
        } else if mine > 0 {
            if pat >> k & 1 == 1 {
                // this side holds the control (Z) or the target (U)
                let ins = match *g {
                    Gate::Cz(a, b) => {
                        let q = if in_a[a] == side_a { a } else { b };
                        Gate::Z(local[q])
                    }
                    Gate::Cnot(ct, t) => {
                        if in_a[ct] == side_a {
                            Gate::Z(local[ct])
                        } else {
                            Gate::X(local[t])
                        }
                    }
                    Gate::CPhase(a, b, th) => {
                        // symmetric: the lower qubit acts as control
                        let (ct, t) = (a.min(b), a.max(b));
                        if in_a[ct] == side_a {
                            Gate::Z(local[ct])
                        } else {
                            Gate::Phase(local[t], th)
                        }
                    }
                    _ => unreachable!("checked by cut_size"),
                };
                out.gate(ins);
            }
            k += 1;
        }
    }
    out
}

fn project(x: u128, in_a: &[bool], side_a: bool) -> u128 {
    let mut y = 0u128;
    let mut j = 0;
    for (q, &a) in in_a.iter().enumerate() {
        if a == side_a {
            y |= ((x >> q) & 1) << j;
            j += 1;
        }
    }
    y
}

fn predict(c: &Circuit, m: usize, cfg: &PlannerConfig) -> Option<(Engine, f64)> {
    let p = planner::plan(c, &PlanRequest::Amplitudes(m), cfg).ok()?;
    p.ranked
        .iter()
        .copied()
        .find(|&(e, _)| !matches!(e, Engine::Zero | Engine::Tableau | Engine::Compressed))
}

/// Greedy min-cut candidates: for each seed, grow `B` one qubit at a time
/// (most gates into `B`, then fewest out) and keep every prefix.
pub fn candidates(c: &Circuit, seeds: usize, max_b: usize) -> Vec<Vec<bool>> {
    let n = c.num_qubits;
    let mut w = vec![vec![0usize; n]; n];
    for g in c.gates() {
        let qs = g.qubits();
        for i in 0..qs.len() {
            for j in i + 1..qs.len() {
                w[qs[i]][qs[j]] += 1;
                w[qs[j]][qs[i]] += 1;
            }
        }
    }
    let deg: Vec<usize> = (0..n).map(|q| w[q].iter().sum()).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&q| std::cmp::Reverse(deg[q]));
    let mut out = Vec::new();
    for &seed in order.iter().take(seeds) {
        let mut in_b = vec![false; n];
        in_b[seed] = true;
        let mut size = 1;
        loop {
            out.push(in_b.iter().map(|&b| !b).collect());
            if size >= max_b.min(n - 1) {
                break;
            }
            let best = (0..n).filter(|&q| !in_b[q]).max_by_key(|&q| {
                let into: usize = (0..n).filter(|&r| in_b[r]).map(|r| w[q][r]).sum();
                (into, std::cmp::Reverse(deg[q] - into), std::cmp::Reverse(q))
            });
            match best {
                Some(q) => {
                    in_b[q] = true;
                    size += 1;
                }
                None => break,
            }
        }
    }
    out
}

/// Prices the candidates of [`candidates`] (cut at most `max_cut` gates)
/// and the single engines; returns the cheapest partition.
pub fn plan_cut(
    c: &Circuit,
    m: usize,
    cfg: &PlannerConfig,
    max_cut: usize,
    max_b: usize,
) -> Result<CutPlan, SimError> {
    let t0 = std::time::Instant::now();
    let single = predict(c, m, cfg);
    let mut best: Option<CutPlan> = None;
    let mut seen = std::collections::HashSet::new();
    for in_a in candidates(c, 4, max_b) {
        if !seen.insert(in_a.clone()) {
            continue;
        }
        let Some(k) = cut_size(c, &in_a) else {
            continue;
        };
        if k > max_cut || in_a.iter().all(|&a| a) || in_a.iter().all(|&a| !a) {
            continue;
        }
        // cheap lower bound before calling the planner: 2^k side runs
        let reps = (1u64 << k) as f64;
        if let Some(b) = &best {
            if reps * 1e-6 > b.predicted_secs {
                continue;
            }
        }
        let ca = side_circuit(c, &in_a, true, 0);
        let cb = side_circuit(c, &in_a, false, 0);
        let (Some(pa), Some(pb)) = (predict(&ca, m, cfg), predict(&cb, m, cfg)) else {
            continue;
        };
        let t = reps * (pa.1 + pb.1);
        if best.as_ref().map_or(true, |b| t < b.predicted_secs) {
            best = Some(CutPlan {
                in_a,
                cut: k,
                side_a: pa,
                side_b: pb,
                predicted_secs: t,
                single,
                plan_secs: 0.0,
            });
        }
    }
    let mut b = best.ok_or(SimError::NotSupported {
        what: "partition: no candidate cut within the limits",
    })?;
    b.plan_secs = t0.elapsed().as_secs_f64();
    Ok(b)
}

/// Exact amplitudes `<x|ψ>` (global phase included) through the cut.
pub fn cut_amplitudes(
    c: &Circuit,
    plan: &CutPlan,
    xs: &[u128],
    cfg: &PlannerConfig,
) -> Result<Vec<Complex64>, SimError> {
    let k = cut_size(c, &plan.in_a).ok_or(SimError::NotSupported {
        what: "partition: a non-controlled gate crosses the cut",
    })?;
    let paths = 1usize << k;
    let side = |side_a: bool| -> Result<Vec<Vec<Complex64>>, SimError> {
        let ys: Vec<u128> = xs.iter().map(|&x| project(x, &plan.in_a, side_a)).collect();
        (0..paths)
            .map(|pat| {
                let sc = side_circuit(c, &plan.in_a, side_a, pat);
                Ok(planner::amplitudes(&sc, &ys, cfg)?.amplitudes)
            })
            .collect()
    };
    let a = side(true)?; // a[s][i]
    let b = side(false)?; // b[u][i]
    let scale = 1.0 / paths as f64;
    Ok((0..xs.len())
        .map(|i| {
            // Walsh–Hadamard transform of b over u: h[s] = Σ_u (-1)^{s·u} b[u]
            let mut h: Vec<Complex64> = (0..paths).map(|u| b[u][i]).collect();
            let mut len = 1;
            while len < paths {
                for blk in (0..paths).step_by(2 * len) {
                    for j in blk..blk + len {
                        let (x, y) = (h[j], h[j + len]);
                        h[j] = x + y;
                        h[j + len] = x - y;
                    }
                }
                len *= 2;
            }
            (0..paths).map(|s| a[s][i] * h[s]).sum::<Complex64>() * scale
        })
        .collect())
}
