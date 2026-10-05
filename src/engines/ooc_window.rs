//! Windowed pass scheduler for the out-of-core state vector.
//!
//! The first out-of-core scheduler ([`crate::engines::ooc::schedule_ooc`]) models every
//! change of the local qubit set as a *global<->local qubit swap*, one full
//! pass over the file each, plus one more pass for every local run in between.
//! This module replaces that with **windowed passes**:
//!
//! * The file is split into chunks of `2^c` amplitudes (the I/O granule).
//!   Physical qubits `0..c` live inside a chunk, physical qubits `c..n` pick
//!   the chunk.
//! * One *pass* chooses a set `H` of `k` high physical qubits and visits every
//!   group of `2^k` chunks that differ only in the `H` bits. A group is
//!   gathered into one RAM buffer of `2^(c+k)` amplitudes, so a pass can apply
//!   **any** gate whose qubits are in the *window* `W = {0..c} ∪ H`
//!   (`c + k` qubits), however many gates that is.
//! * Before the group is written back, the buffer's `c + k` bit positions may
//!   be permuted. That relocates logical qubits between low and high physical
//!   positions at zero extra I/O: it is how a "swap" is paid for, fused into a
//!   pass that was happening anyway.
//! * `Swap` gates are free: they are folded into the logical->physical map
//!   (no data moves) and the final layout is restored by the same fused
//!   permutations.
//!
//! Every pass reads and writes the whole file exactly once, so the number of
//! passes is the whole cost model. The scheduler is greedy with lookahead: it
//! drains every gate that is executable in the current window, and when stuck
//! it adds to `H` the high qubits whose inclusion unlocks the most gates per
//! added qubit (evaluated by simulating the drain), then picks the post-pass
//! layout with Belady's rule (the `c` window qubits that are needed soonest
//! become the low ones).
//!
//! Exactness: gates are applied by the same cache-blocked kernels as the
//! in-RAM executor, and layout changes are pure index permutations, so the
//! result is identical to the in-RAM run up to the floating-point rounding of
//! the gate kernels (which see the same inputs in the same order per
//! amplitude pair).

use crate::circuit::{check_gate, Circuit, Op, SimError};
use crate::dag::gate_qubits;
use crate::engines::ooc::remap_gate;
use crate::gate::Gate;
use rayon::prelude::*;

/// One windowed pass over the file.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowPass {
    /// Sorted high physical qubits (each `>= c`) gathered into the window.
    /// Buffer bit `c + j` is physical qubit `high[j]`; buffer bits `0..c` are
    /// physical qubits `0..c`.
    pub high: Vec<usize>,
    /// Gates in *buffer* coordinates (all `< c + high.len()`), in order.
    pub gates: Vec<Gate>,
    /// Optional bit permutation of the buffer applied after the gates:
    /// new bit `b` takes its value from old bit `perm[b]`.
    pub perm: Option<Vec<usize>>,
}

/// A complete windowed plan.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowPlan {
    /// The passes, in order.
    pub passes: Vec<WindowPass>,
    /// Logical -> physical map after the plan (identity when order is restored).
    pub final_v2p: Vec<usize>,
    /// `Swap` gates folded into the layout instead of being executed.
    pub elided_swaps: usize,
    /// Passes that apply at least one gate.
    pub gate_passes: usize,
    /// Passes that change the physical layout (apply a buffer permutation).
    pub perm_passes: usize,
}

/// How the scheduler treats `Swap` gates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwapPolicy {
    /// Always fold into the logical->physical map (never executed; the final
    /// order is restored by buffer permutations).
    Always,
    /// Execute as an in-buffer gate when both qubits are in the window, fold
    /// into the layout otherwise (a swap that reaches outside the window is
    /// free; one inside is cheaper as a cache-blocked kernel than as a
    /// strided copy later).
    Hybrid,
    /// Always execute as a gate (needs both qubits in the window).
    Never,
}

/// Knobs for [`schedule_window`].
#[derive(Clone, Debug)]
pub struct WindowOptions {
    /// Number of high qubits `k` per window (buffer is `2^(c+k)` amplitudes).
    pub extra_bits: usize,
    /// How `Swap` gates are handled.
    pub swaps: SwapPolicy,
    /// Put the layout back to canonical order at the end.
    pub restore_order: bool,
    /// Candidate score is `drained / |needed|^alpha` (tuned per strategy; the
    /// planner tries several and keeps the plan with the fewest passes).
    pub alphas: Vec<f64>,
}

impl Default for WindowOptions {
    fn default() -> Self {
        WindowOptions {
            extra_bits: 3,
            swaps: SwapPolicy::Hybrid,
            restore_order: true,
            alphas: vec![1.0, 0.5, 0.0, 2.0],
        }
    }
}

/// Dependency structure of a gate list (qubit wires).
struct Deps {
    qubits: Vec<([usize; 3], usize)>,
    succ: Vec<Vec<u32>>,
    indeg0: Vec<u32>,
    swap: Vec<bool>,
}

impl Deps {
    fn build(gates: &[Gate], n: usize, policy: SwapPolicy) -> Deps {
        let mut last: Vec<Option<u32>> = vec![None; n];
        let mut succ: Vec<Vec<u32>> = vec![Vec::new(); gates.len()];
        let mut indeg0 = vec![0u32; gates.len()];
        let mut qubits = Vec::with_capacity(gates.len());
        let mut swap = Vec::with_capacity(gates.len());
        for (i, g) in gates.iter().enumerate() {
            let (qs, nq) = gate_qubits(g);
            let mut preds: Vec<u32> = Vec::with_capacity(3);
            for &q in &qs[..nq] {
                if let Some(p) = last[q] {
                    if !preds.contains(&p) {
                        preds.push(p);
                    }
                }
                last[q] = Some(i as u32);
            }
            for p in preds {
                succ[p as usize].push(i as u32);
                indeg0[i] += 1;
            }
            qubits.push((qs, nq));
            swap.push(policy != SwapPolicy::Never && matches!(g, Gate::Swap(..)));
        }
        Deps {
            qubits,
            succ,
            indeg0,
            swap,
        }
    }
}

const DONE: u32 = u32::MAX;

#[derive(Clone)]
struct State {
    indeg: Vec<u32>,
    ready: Vec<u32>,
    v2p: Vec<usize>,
    remaining: usize,
}

impl State {
    fn new(deps: &Deps, v2p: Vec<usize>) -> State {
        let ready = (0..deps.indeg0.len() as u32)
            .filter(|&i| deps.indeg0[i as usize] == 0)
            .collect();
        State {
            indeg: deps.indeg0.clone(),
            ready,
            v2p,
            remaining: deps.indeg0.len(),
        }
    }

    /// Executes every ready gate whose qubits all sit inside the window
    /// (`in_w[physical]`), cascading. `emit(node, physical_qubits)` is called
    /// for every executed non-elided gate. Returns the number of gates (and
    /// elided swaps) retired.
    fn drain(
        &mut self,
        deps: &Deps,
        in_w: &[bool],
        hybrid: bool,
        mut emit: impl FnMut(u32, &[usize; 3]),
    ) -> usize {
        let mut count = 0;
        loop {
            let mut progressed = false;
            let mut i = 0;
            while i < self.ready.len() {
                let id = self.ready[i];
                let (qs, nq) = deps.qubits[id as usize];
                let both_in = in_w[self.v2p[qs[0]]] && in_w[self.v2p[qs[1]]];
                if deps.swap[id as usize] && !(hybrid && both_in) {
                    self.v2p.swap(qs[0], qs[1]);
                } else {
                    let mut ph = [0usize; 3];
                    let mut ok = true;
                    for j in 0..nq {
                        ph[j] = self.v2p[qs[j]];
                        ok &= in_w[ph[j]];
                    }
                    if !ok {
                        i += 1;
                        continue;
                    }
                    emit(id, &ph);
                }
                self.ready.swap_remove(i);
                self.indeg[id as usize] = DONE;
                self.remaining -= 1;
                count += 1;
                progressed = true;
                for &s in &deps.succ[id as usize] {
                    let d = &mut self.indeg[s as usize];
                    *d -= 1;
                    if *d == 0 {
                        self.ready.push(s);
                    }
                }
            }
            if !progressed {
                break;
            }
        }
        count
    }

    /// First remaining use (topological index) of every logical qubit.
    fn next_use(&self, deps: &Deps, n: usize) -> Vec<usize> {
        let mut nu = vec![usize::MAX; n];
        let mut left = n;
        for id in 0..deps.qubits.len() {
            if self.indeg[id] == DONE {
                continue;
            }
            let (qs, nq) = deps.qubits[id];
            for &q in &qs[..nq] {
                if nu[q] == usize::MAX {
                    nu[q] = id;
                    left -= 1;
                }
            }
            if left == 0 {
                break;
            }
        }
        nu
    }
}

fn p2v_of(v2p: &[usize]) -> Vec<usize> {
    let mut p2v = vec![0; v2p.len()];
    for (v, &p) in v2p.iter().enumerate() {
        p2v[p] = v;
    }
    p2v
}

/// Windowed plan for a unitary gate list, starting from layout `v2p0`.
pub fn schedule_window_gates(
    gates: &[Gate],
    n: usize,
    c: usize,
    v2p0: &[usize],
    opts: &WindowOptions,
) -> Result<WindowPlan, SimError> {
    for g in gates {
        check_gate(g, n)?;
    }
    let mut best: Option<WindowPlan> = None;
    let alphas: &[f64] = if opts.alphas.is_empty() {
        &[1.0]
    } else {
        &opts.alphas
    };
    for &alpha in alphas {
        let plan = schedule_one(gates, n, c, v2p0, opts, alpha)?;
        if best
            .as_ref()
            .is_none_or(|b| plan.passes.len() < b.passes.len())
        {
            best = Some(plan);
        }
    }
    Ok(best.expect("at least one strategy"))
}

/// Windowed plan for a circuit of unitary gates, starting from the canonical layout.
pub fn schedule_window(
    circuit: &Circuit,
    c: usize,
    opts: &WindowOptions,
) -> Result<WindowPlan, SimError> {
    let n = circuit.num_qubits;
    let mut gates = Vec::with_capacity(circuit.ops.len());
    for op in &circuit.ops {
        match op {
            Op::Gate(g) => gates.push(*g),
            _ => {
                return Err(SimError::NotSupported {
                    what: "out-of-core simulation currently supports unitary circuits",
                })
            }
        }
    }
    let v2p0: Vec<usize> = (0..n).collect();
    schedule_window_gates(&gates, n, c, &v2p0, opts)
}

fn schedule_one(
    gates: &[Gate],
    n: usize,
    c: usize,
    v2p0: &[usize],
    opts: &WindowOptions,
    alpha: f64,
) -> Result<WindowPlan, SimError> {
    let deps = Deps::build(gates, n, opts.swaps);
    let hybrid = opts.swaps == SwapPolicy::Hybrid;
    let mut st = State::new(&deps, v2p0.to_vec());
    let mut passes: Vec<WindowPass> = Vec::new();
    let mut elided = 0usize;
    let c = c.min(n);
    // Window of high qubits; if everything is local there is nothing to window.
    let k_max = opts.extra_bits.min(n - c);

    let mut guard = 0usize;
    loop {
        let finished = st.remaining == 0;
        if finished && (!opts.restore_order || st.v2p.iter().enumerate().all(|(q, &p)| p == q)) {
            break;
        }
        guard += 1;
        if guard > 4 * gates.len() + 4 * n + 64 {
            return Err(SimError::NotSupported {
                what: "out-of-core window scheduler failed to make progress",
            });
        }

        let mut in_w = vec![false; n];
        in_w[..c].fill(true);
        let mut high: Vec<usize> = Vec::new();
        let mut emitted: Vec<Gate> = Vec::new();

        // ---- gate phase: drain, extend the window when stuck ----
        loop {
            let mut local_emitted: Vec<Gate> = Vec::new();
            st.drain(&deps, &in_w, hybrid, |id, ph| {
                local_emitted.push(remap_physical(&gates[id as usize], ph));
            });
            emitted.extend(local_emitted);
            if st.ready.is_empty() {
                break;
            }
            // Stuck: every ready gate needs high qubits outside the window.
            let mut cands: Vec<Vec<usize>> = Vec::new();
            for &id in &st.ready {
                let (qs, nq) = deps.qubits[id as usize];
                let mut need: Vec<usize> = qs[..nq]
                    .iter()
                    .map(|&q| st.v2p[q])
                    .filter(|&p| !in_w[p])
                    .collect();
                need.sort_unstable();
                need.dedup();
                if need.is_empty() || high.len() + need.len() > k_max {
                    continue;
                }
                if !cands.contains(&need) {
                    cands.push(need);
                }
            }
            if cands.is_empty() {
                if high.is_empty() {
                    return Err(SimError::NotSupported {
                        what: "out-of-core window too small for a gate (increase group_bits)",
                    });
                }
                break;
            }
            let mut best: Option<(f64, usize, Vec<usize>)> = None;
            for need in cands {
                let mut sim = st.clone();
                let mut w2 = in_w.clone();
                for &p in &need {
                    w2[p] = true;
                }
                let drained = sim.drain(&deps, &w2, hybrid, |_, _| {});
                let score = drained as f64 / (need.len() as f64).powf(alpha);
                let better = match &best {
                    None => true,
                    Some((bs, bl, _)) => {
                        score > *bs + 1e-12 || ((score - *bs).abs() <= 1e-12 && need.len() < *bl)
                    }
                };
                if better {
                    best = Some((score, need.len(), need));
                }
            }
            let (_, _, need) = best.unwrap();
            for p in need {
                in_w[p] = true;
                high.push(p);
            }
        }

        // Swaps retired inside this pass are free; count them for stats.
        // (They are not in `emitted`.)
        let finished_now = st.remaining == 0;

        // ---- fill the window up to k_max ----
        let p2v = p2v_of(&st.v2p);
        if finished_now {
            if opts.restore_order {
                // Prefer high positions that are out of place.
                let mut wrong: Vec<usize> = (c..n).filter(|&p| p2v[p] != p && !in_w[p]).collect();
                // Follow chains so cycles close inside one window.
                let mut order: Vec<usize> = Vec::new();
                while let Some(&start) = wrong.first() {
                    let mut cur = start;
                    while let Some(pos) = wrong.iter().position(|&x| x == cur) {
                        wrong.remove(pos);
                        order.push(cur);
                        let l = p2v[cur];
                        if l >= c {
                            cur = l;
                        } else {
                            break;
                        }
                    }
                }
                for p in order {
                    if high.len() >= k_max {
                        break;
                    }
                    in_w[p] = true;
                    high.push(p);
                }
            }
        } else {
            let nu = st.next_use(&deps, n);
            let mut cand: Vec<usize> = (c..n).filter(|&p| !in_w[p]).collect();
            cand.sort_by_key(|&p| nu[p2v[p]]);
            for p in cand {
                if high.len() >= k_max {
                    break;
                }
                in_w[p] = true;
                high.push(p);
            }
        }
        high.sort_unstable();

        // ---- choose the post-pass layout ----
        let k = high.len();
        let m = c + k;
        // buffer bit -> physical position
        let bit_pos: Vec<usize> = (0..c).chain(high.iter().copied()).collect();
        let nu = st.next_use(&deps, n);
        let assign = if finished_now {
            if opts.restore_order {
                Some(arrange_restore(&bit_pos, c, &p2v))
            } else {
                None
            }
        } else {
            arrange_belady(&bit_pos, c, &p2v, &nu)
        };
        let mut perm: Option<Vec<usize>> = None;
        if let Some(assign) = assign {
            // assign[b] = logical qubit that should sit at buffer bit b afterwards.
            let mut pm = vec![0usize; m];
            let mut identity = true;
            let old_bit_of_pos = |pos: usize| bit_pos.iter().position(|&x| x == pos).unwrap();
            for b in 0..m {
                let l = assign[b];
                let ob = old_bit_of_pos(st.v2p[l]);
                pm[b] = ob;
                if ob != b {
                    identity = false;
                }
            }
            if !identity {
                for b in 0..m {
                    st.v2p[assign[b]] = bit_pos[b];
                }
                perm = Some(pm);
            }
        }

        // Gates: physical -> buffer coordinates.
        let mut to_bit = vec![usize::MAX; n];
        for (b, &p) in bit_pos.iter().enumerate() {
            to_bit[p] = b;
        }
        let gates_buf: Vec<Gate> = emitted.iter().map(|g| remap_gate(g, &to_bit)).collect();

        if gates_buf.is_empty() && perm.is_none() {
            // A pass that does nothing: only possible when stuck without progress.
            if st.remaining == 0 {
                break;
            }
            return Err(SimError::NotSupported {
                what: "out-of-core window scheduler failed to make progress",
            });
        }
        passes.push(WindowPass {
            high,
            gates: gates_buf,
            perm,
        });
    }

    elided += gates.len() - passes.iter().map(|p| p.gates.len()).sum::<usize>();
    let gate_passes = passes.iter().filter(|p| !p.gates.is_empty()).count();
    let perm_passes = passes.iter().filter(|p| p.perm.is_some()).count();
    Ok(WindowPlan {
        passes,
        final_v2p: st.v2p,
        elided_swaps: elided,
        gate_passes,
        perm_passes,
    })
}

/// Rewrites a gate onto the physical qubits `ph` (in the gate's argument order).
fn remap_physical(g: &Gate, ph: &[usize; 3]) -> Gate {
    let mut out = *g;
    set_qubits(&mut out, ph);
    out
}

fn set_qubits(g: &mut Gate, ph: &[usize; 3]) {
    match g {
        Gate::I(q)
        | Gate::X(q)
        | Gate::Y(q)
        | Gate::Z(q)
        | Gate::H(q)
        | Gate::S(q)
        | Gate::Sdg(q)
        | Gate::T(q)
        | Gate::Tdg(q)
        | Gate::Sx(q)
        | Gate::Sxdg(q)
        | Gate::Rx(q, _)
        | Gate::Ry(q, _)
        | Gate::Rz(q, _)
        | Gate::Phase(q, _)
        | Gate::U(q, _, _, _) => *q = ph[0],
        Gate::Cnot(a, b)
        | Gate::Cz(a, b)
        | Gate::Swap(a, b)
        | Gate::ISwap(a, b)
        | Gate::ISwapdg(a, b)
        | Gate::CPhase(a, b, _) => {
            *a = ph[0];
            *b = ph[1];
        }
        Gate::Ccx(a, b, t) => {
            *a = ph[0];
            *b = ph[1];
            *t = ph[2];
        }
    }
}

/// Belady layout: the `c` window qubits needed soonest become the low ones.
/// Returns `assign[b]` = logical qubit at buffer bit `b` afterwards, or `None`
/// if the set of low qubits does not change (low-bit order is left alone).
fn arrange_belady(bit_pos: &[usize], c: usize, p2v: &[usize], nu: &[usize]) -> Option<Vec<usize>> {
    let m = bit_pos.len();
    let cur: Vec<usize> = (0..m).map(|b| p2v[bit_pos[b]]).collect(); // logical per buffer bit
                                                                     // Rank window logicals by next use; ties keep the currently-low ones first.
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by_key(|&b| (nu[cur[b]], b));
    let mut want_low = vec![false; m];
    for &b in idx.iter().take(c) {
        want_low[b] = true;
    }
    // evicted: currently low (b < c) but not wanted low; incoming: high, wanted low.
    let evicted: Vec<usize> = (0..c).filter(|&b| !want_low[b]).collect();
    let incoming: Vec<usize> = (c..m).filter(|&b| want_low[b]).collect();
    debug_assert_eq!(evicted.len(), incoming.len());
    if evicted.is_empty() {
        return None;
    }
    let mut assign = cur.clone();
    // Prefer matches that put a logical qubit on the bit equal to its id.
    let mut slots_low = evicted.clone();
    let mut slots_high = incoming.clone();
    let mut pend_in: Vec<usize> = Vec::new();
    let mut pend_ev: Vec<usize> = Vec::new();
    for &ib in &incoming {
        let l = cur[ib];
        if let Some(i) = slots_low.iter().position(|&b| bit_pos[b] == l) {
            let b = slots_low.remove(i);
            assign[b] = l;
        } else {
            pend_in.push(ib);
        }
    }
    for &eb in &evicted {
        let l = cur[eb];
        if let Some(i) = slots_high.iter().position(|&b| bit_pos[b] == l) {
            let b = slots_high.remove(i);
            assign[b] = l;
        } else {
            pend_ev.push(eb);
        }
    }
    for (ib, b) in pend_in.into_iter().zip(slots_low) {
        assign[b] = cur[ib];
    }
    for (eb, b) in pend_ev.into_iter().zip(slots_high) {
        assign[b] = cur[eb];
    }
    Some(assign)
}

/// Restoration layout: every window logical qubit whose home position is in the
/// window goes home; low-home qubits fill low slots; the rest stay put if their
/// slot is free, else fill leftovers.
fn arrange_restore(bit_pos: &[usize], c: usize, p2v: &[usize]) -> Vec<usize> {
    let m = bit_pos.len();
    let cur: Vec<usize> = (0..m).map(|b| p2v[bit_pos[b]]).collect();
    let mut assign: Vec<Option<usize>> = vec![None; m];
    let mut placed = vec![false; m]; // by index into cur
    let bit_of_pos = |pos: usize| bit_pos.iter().position(|&x| x == pos);
    // Pass 1: qubit l goes to position l when that position is in the window.
    for i in 0..m {
        let l = cur[i];
        if let Some(b) = bit_of_pos(l) {
            if assign[b].is_none() {
                assign[b] = Some(l);
                placed[i] = true;
            }
        }
    }
    // Pass 2: stay put when free.
    for i in 0..m {
        if !placed[i] && assign[i].is_none() {
            assign[i] = Some(cur[i]);
            placed[i] = true;
        }
    }
    // Pass 3: leftovers into remaining slots (low slots first for low-home qubits).
    let mut left: Vec<usize> = (0..m).filter(|&i| !placed[i]).map(|i| cur[i]).collect();
    left.sort_by_key(|&l| (l >= c, l));
    let mut free: Vec<usize> = (0..m).filter(|&b| assign[b].is_none()).collect();
    free.sort_by_key(|&b| (b >= c, b));
    for (l, b) in left.into_iter().zip(free) {
        assign[b] = Some(l);
    }
    assign.into_iter().map(|a| a.unwrap()).collect()
}

/// Applies the bit permutation `perm` (new bit `b` takes old bit `perm[b]`) to
/// a buffer of `2^perm.len()` elements, writing the result to `dst`.
pub fn permute_bits<T: Copy + Send + Sync>(src: &[T], dst: &mut [T], perm: &[usize]) {
    let m = perm.len();
    assert_eq!(src.len(), 1 << m);
    assert_eq!(dst.len(), 1 << m);
    let r = (0..m).take_while(|&b| perm[b] == b).count();
    if r == m {
        dst.copy_from_slice(src);
        return;
    }
    let run = 1usize << r;
    let hi = m - r;
    let half = hi / 2;
    let rest = hi - half;
    // lo table: bits r..r+half of the new index; hi table: the remaining bits.
    let mut lo = vec![0usize; 1 << half];
    for (t, slot) in lo.iter_mut().enumerate() {
        let mut v = 0usize;
        for j in 0..half {
            v |= ((t >> j) & 1) << perm[r + j];
        }
        *slot = v;
    }
    let mut hi_t = vec![0usize; 1 << rest];
    for (t, slot) in hi_t.iter_mut().enumerate() {
        let mut v = 0usize;
        for j in 0..rest {
            v |= ((t >> j) & 1) << perm[r + half + j];
        }
        *slot = v;
    }
    let lomask = (1usize << half) - 1;
    let par_len = run.max(1 << 14);
    dst.par_chunks_mut(par_len)
        .enumerate()
        .for_each(|(ci, out)| {
            let t0 = ci * par_len / run;
            for (j, o) in out.chunks_mut(run).enumerate() {
                let t = t0 + j;
                let s = lo[t & lomask] | hi_t[t >> half];
                o.copy_from_slice(&src[s..s + run]);
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(src: &[u32], perm: &[usize]) -> Vec<u32> {
        let m = perm.len();
        let mut out = vec![0u32; 1 << m];
        for (j, o) in out.iter_mut().enumerate() {
            let mut old = 0usize;
            for (b, &pb) in perm.iter().enumerate() {
                old |= ((j >> b) & 1) << pb;
            }
            *o = src[old];
        }
        out
    }

    #[test]
    fn permute_matches_naive() {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for m in 1..=12usize {
            for _ in 0..8 {
                let mut perm: Vec<usize> = (0..m).collect();
                if m > 3 && rand::random::<bool>() {
                    perm[3..].shuffle(&mut rng);
                } else {
                    perm.shuffle(&mut rng);
                }
                let src: Vec<u32> = (0..1u32 << m).map(|x| x.wrapping_mul(2654435761)).collect();
                let mut dst = vec![0u32; 1 << m];
                permute_bits(&src, &mut dst, &perm);
                assert_eq!(dst, naive(&src, &perm), "m={m} perm={perm:?}");
            }
        }
    }
}
