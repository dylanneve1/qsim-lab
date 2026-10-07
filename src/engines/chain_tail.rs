//! Tail-open chain sweep: all `2^m` amplitudes `<x_0..x_(m-1), x_m..x_(n-1)| U |0^n>`
//! for a fixed suffix `x_m..x_(n-1)` from one sweep.
//!
//! The sweep runs on the mirrored chain ([`chain_sweep::mirror`]): qubits
//! `n-1, n-2, .., m` in that order, and stops at the clean cut on edge
//! `(m-1, m)`. The bond register then holds `R(β)`, a function of the
//! values `β_t` of qubit `m` at the `K` CZs of that edge (on the mirrored
//! chain qubit `m` is the projector side of those CZs). The other halves of
//! those CZs are `Z^{β_t}` on qubit `m-1`, so with the tail circuit
//! `T(β) = G_K Z_{m-1}^{β_K} G_(K-1) .. Z_{m-1}^{β_1} G_0 |0^m>` (the `G_t`
//! are the tail's own gates between consecutive bonds, CZs inside the tail
//! included):
//!
//! `amp(x_tail) = scale · <x_tail| Σ_β R(β) T(β)`.
//!
//! [`tail_amplitudes`] evaluates that sum in one read-only pass over the
//! register: a depth-first walk over `β` in time order that carries the
//! `2^m`-vector `G_t Z^{β_t} .. |0>` and, at the last bond, adds
//! `R(β) ·` the vector into an accumulator; the last segment `G_K` is
//! applied once at the end. Everything after the sweep is f64, so the pass
//! adds no rounding: the number of roundings of a low-precision run is the
//! sweep's pass count only.
//!
//! Output order: entry `j` is the completion whose bits `q0..q(m-1)` read
//! `j` with **q0 the most significant bit** (`j = int(bits_q0_first[..m], 2)`),
//! the convention of SUTD's amplitude batches (Zenodo 10.5281/zenodo.21912448).
//!
//! The register is read through [`SweepBackend::amp`], so any backend that
//! can run a [`SweepPlan`] and answer random reads (CPU f64/f32, the packed
//! store, a GPU with a host-side copy) plugs in.

use crate::circuit::SimError;
use crate::engines::blocked::BlockConfig;
use crate::engines::chain_sweep::{
    compile_prefix, mirror, reverse_bits, ChainCircuit, Ev, SweepPlan,
};
use crate::engines::statevector::{Real, StateVector};
use crate::gate::Mat2;
use num_complex::Complex64;
use rayon::prelude::*;
use std::collections::HashMap;

const C0: Complex64 = Complex64::new(0.0, 0.0);
const C1: Complex64 = Complex64::new(1.0, 0.0);

/// One operation of the tail circuit on the `2^m`-vector (bit `i` of the
/// index = qubit `i`).
#[derive(Clone, Debug)]
enum TOp {
    /// Dense 1q matrix on qubit `q`.
    U(usize, Mat2),
    /// Diagonal over the whole `2^m` space (fused CZs and diagonal 1q gates).
    Diag(Vec<Complex64>),
}

/// A compiled tail-open amplitude batch.
#[derive(Clone, Debug)]
pub struct TailPlan {
    /// Open qubits `q0..q(m-1)`.
    pub m: usize,
    /// The sweep of qubits `n-1..m` on the mirrored chain.
    pub sweep: SweepPlan,
    /// Bonds of edge `(m-1, m)` in time order (= bond id order).
    pub bonds: Vec<usize>,
    /// Register bit of each of `bonds`.
    pub slots: Vec<usize>,
    /// `segs[t]`: tail gates after bond `t-1` and before bond `t`
    /// (`segs[K]` after the last bond).
    segs: Vec<Vec<TOp>>,
}

fn is_diag(m: &Mat2) -> bool {
    m[0][1] == C0 && m[1][0] == C0
}

impl TailPlan {
    /// Compiles the batch for the suffix bits of `x` (bit `i` = qubit `i`,
    /// Qiskit's little-endian integer; bits `0..m` are ignored).
    pub fn new(cc: &ChainCircuit, x: u128, m: usize) -> Self {
        let n = cc.n;
        assert!(m >= 1 && m < n, "tail size must be in 1..n");
        assert!(m <= 16, "tail of 2^m amplitudes: m <= 16");
        let mc = mirror(cc);
        let none = HashMap::new();
        let (sweep, open) = compile_prefix(&mc, reverse_bits(x, n), &none, n - m);
        let bonds: Vec<usize> = open.iter().map(|&(b, _)| b).collect();
        let slots: Vec<usize> = open.iter().map(|&(_, s)| s).collect();
        debug_assert!(bonds.iter().all(|&b| cc.bond_edge[b] == m - 1));
        debug_assert_eq!(bonds, cc.edge_bonds(m - 1));
        let segs = tail_segments(cc, m, &bonds);
        TailPlan {
            m,
            sweep,
            bonds,
            slots,
            segs,
        }
    }

    /// Number of open bonds (`K`).
    pub fn num_bonds(&self) -> usize {
        self.bonds.len()
    }

    /// Complex MACs of the tail pass: the GEMM form when the register is
    /// bond-ordered (`2^K · 2^m`), else the depth-first form.
    pub fn tail_cost(&self) -> f64 {
        if gemm_ok(self) {
            return 2f64.powi(self.bonds.len() as i32) * (1usize << self.m) as f64;
        }
        self.tail_cost_dfs()
    }

    /// Complex MACs of the depth-first tail pass (dense ops count 2 per entry).
    pub fn tail_cost_dfs(&self) -> f64 {
        let dim = (1usize << self.m) as f64;
        let k = self.bonds.len();
        let seg = |t: usize| -> f64 {
            self.segs[t]
                .iter()
                .map(|o| match o {
                    TOp::U(..) => 2.0,
                    TOp::Diag(_) => 1.0,
                })
                .sum::<f64>()
        };
        // node at depth t (t bonds fixed) applies segs[t]; leaves: 2 reads, 1 MAC per entry
        let mut c = 0.0;
        for t in 1..k {
            c += 2f64.powi(t as i32) * (seg(t) + 0.5) * dim;
        }
        c + 2f64.powi(k as i32 - 1) * dim
    }
}

/// The tail circuit split at the bonds of edge `(m-1, m)`.
fn tail_segments(cc: &ChainCircuit, m: usize, bonds: &[usize]) -> Vec<Vec<TOp>> {
    let dim = 1usize << m;
    let mut segs: Vec<Vec<TOp>> = vec![Vec::new()];
    let mut pos = vec![0usize; m];
    let lines = &cc.lines[..m];
    let push = |segs: &mut Vec<Vec<TOp>>, op: TOp| {
        let seg = segs.last_mut().unwrap();
        match (seg.last_mut(), op) {
            (Some(TOp::Diag(d)), TOp::Diag(e)) => {
                for (a, b) in d.iter_mut().zip(e) {
                    *a *= b;
                }
            }
            (_, op) => seg.push(op),
        }
    };
    let diag1 = |q: usize, a: Complex64, b: Complex64| -> Vec<Complex64> {
        (0..dim)
            .map(|s| if (s >> q) & 1 == 0 { a } else { b })
            .collect()
    };
    let mut done_bonds = 0usize;
    // merge the worldlines in a causal order: a tail-internal CZ fires when
    // both of its qubits have reached it
    loop {
        let mut progressed = false;
        for q in 0..m {
            while pos[q] < lines[q].len() {
                match lines[q][pos[q]] {
                    Ev::G(u) => {
                        if is_diag(&u) {
                            push(&mut segs, TOp::Diag(diag1(q, u[0][0], u[1][1])));
                        } else {
                            push(&mut segs, TOp::U(q, u));
                        }
                        pos[q] += 1;
                        progressed = true;
                    }
                    Ev::P(b) if q == m - 1 => {
                        // a CZ to qubit m: Z^{β} on this qubit, summed by the DFS
                        assert_eq!(b, bonds[done_bonds], "open bonds out of time order");
                        done_bonds += 1;
                        segs.push(Vec::new());
                        pos[q] += 1;
                        progressed = true;
                    }
                    Ev::P(b) => {
                        // CZ(q, q+1): fires when q+1 is at Z(b)
                        if matches!(lines[q + 1].get(pos[q + 1]), Some(Ev::Z(bb)) if *bb == b) {
                            let d: Vec<Complex64> = (0..dim)
                                .map(|s| if (s >> q) & 3 == 3 { -C1 } else { C1 })
                                .collect();
                            push(&mut segs, TOp::Diag(d));
                            pos[q] += 1;
                            pos[q + 1] += 1;
                            progressed = true;
                        } else {
                            break;
                        }
                    }
                    Ev::Z(_) => break, // waits for its left partner
                }
            }
        }
        if pos.iter().zip(lines).all(|(p, l)| *p == l.len()) {
            break;
        }
        assert!(
            progressed,
            "tail worldlines deadlocked (not a valid chain circuit)"
        );
    }
    assert_eq!(done_bonds, bonds.len());
    segs
}

#[inline]
fn apply_ops(ops: &[TOp], v: &mut [Complex64]) {
    for op in ops {
        match op {
            TOp::Diag(d) => {
                for (a, b) in v.iter_mut().zip(d) {
                    *a *= b;
                }
            }
            TOp::U(q, u) => {
                let st = 1usize << q;
                let mut i = 0;
                while i < v.len() {
                    for j in i..i + st {
                        let (a, b) = (v[j], v[j + st]);
                        v[j] = u[0][0] * a + u[0][1] * b;
                        v[j + st] = u[1][0] * a + u[1][1] * b;
                    }
                    i += 2 * st;
                }
            }
        }
    }
}

/// Bits of the DFS fixed by the parallel split (2^PAR tasks).
const PAR: usize = 10;

/// The `2^m` amplitudes of the batch (SUTD order, see the module docs),
/// reading the swept register through `reg` (raw register values; the
/// sweep's scale is applied here). Uses the GEMM form ([`tail_amplitudes_gemm`])
/// when the open bonds sit on register bits `0..K` in time order (the chain
/// sweep's allocator gives that), else the depth-first form.
pub fn tail_amplitudes(tp: &TailPlan, reg: &(dyn Fn(usize) -> Complex64 + Sync)) -> Vec<Complex64> {
    if gemm_ok(tp) {
        let runs = |start: usize, out: &mut [Complex64]| {
            for (k, o) in out.iter_mut().enumerate() {
                *o = reg(start + k);
            }
        };
        tail_amplitudes_gemm(tp, &runs, None)
    } else {
        tail_amplitudes_dfs(tp, reg)
    }
}

/// [`tail_amplitudes`] reading a backend: contiguous rows through
/// [`SweepBackend::read_run`] (block decodes, no per-entry overhead).
pub fn tail_amplitudes_backend<B: SweepBackend + ?Sized>(tp: &TailPlan, be: &B) -> Vec<Complex64> {
    if gemm_ok(tp) {
        tail_amplitudes_gemm(tp, &|start, out| be.read_run(start, out), None)
    } else {
        tail_amplitudes_dfs(tp, &|i| be.amp(i))
    }
}

fn gemm_ok(tp: &TailPlan) -> bool {
    tp.slots.iter().enumerate().all(|(t, &s)| s == t) && tp.sweep.width == tp.slots.len().max(1)
}

/// Depth-first form: walks `β` in time order carrying the `2^m`-vector;
/// cost `~2^K · |seg| · 2^m` (random register reads). The reference for the
/// GEMM form, and the fallback for any register layout.
pub fn tail_amplitudes_dfs(
    tp: &TailPlan,
    reg: &(dyn Fn(usize) -> Complex64 + Sync),
) -> Vec<Complex64> {
    let m = tp.m;
    let dim = 1usize << m;
    let k = tp.bonds.len();
    let w = m - 1; // wire carrying Z^β
    let mut v0 = vec![C0; dim];
    v0[0] = C1;
    apply_ops(&tp.segs[0], &mut v0);
    let mut acc = if k == 0 {
        v0.iter().map(|a| a * reg(0)).collect::<Vec<_>>()
    } else {
        let p = PAR.min(k - 1);
        // one accumulator per top task, summed in task order: the batch is
        // bit-reproducible whatever the thread count or scheduling
        let parts: Vec<Vec<Complex64>> = (0..1usize << p)
            .into_par_iter()
            .map(|top| {
                let mut acc = vec![C0; dim];
                let mut v = v0.clone();
                let mut idx = 0usize;
                for t in 0..p {
                    let b = (top >> (p - 1 - t)) & 1;
                    if b == 1 {
                        flip(&mut v, w);
                        idx |= 1 << tp.slots[t];
                    }
                    apply_ops(&tp.segs[t + 1], &mut v);
                }
                let mut bufs = vec![vec![C0; dim]; k - p];
                dfs(tp, reg, p, &v, idx, &mut bufs, &mut acc);
                acc
            })
            .collect();
        let mut acc = vec![C0; dim];
        for part in parts {
            for (x, y) in acc.iter_mut().zip(part) {
                *x += y;
            }
        }
        acc
    };
    apply_ops(&tp.segs[k], &mut acc);
    let mut out = vec![C0; dim];
    for (s, a) in acc.iter().enumerate() {
        out[reverse_bits(s as u128, m) as usize] = a * tp.sweep.scale;
    }
    out
}

#[inline]
fn flip(v: &mut [Complex64], w: usize) {
    for (s, a) in v.iter_mut().enumerate() {
        if (s >> w) & 1 == 1 {
            *a = -*a;
        }
    }
}

/// `v`: the vector with bonds `0..t` fixed (encoded in `idx`) and
/// `segs[t]` applied; bond `t` is next. `bufs` holds one scratch vector
/// per remaining level.
fn dfs(
    tp: &TailPlan,
    reg: &(dyn Fn(usize) -> Complex64 + Sync),
    t: usize,
    v: &[Complex64],
    idx: usize,
    bufs: &mut [Vec<Complex64>],
    acc: &mut [Complex64],
) {
    let k = tp.bonds.len();
    let w = tp.m - 1;
    if t + 1 == k {
        let r0 = reg(idx);
        let r1 = reg(idx | 1 << tp.slots[t]);
        let (p, q) = (r0 + r1, r0 - r1);
        for (s, (a, x)) in acc.iter_mut().zip(v).enumerate() {
            *a += if (s >> w) & 1 == 0 { p * x } else { q * x };
        }
        return;
    }
    let (cur, rest) = bufs.split_first_mut().unwrap();
    for b in 0..2 {
        cur.copy_from_slice(v);
        if b == 1 {
            flip(cur, w);
        }
        apply_ops(&tp.segs[t + 1], cur);
        dfs(tp, reg, t + 1, cur, idx | b << tp.slots[t], rest, acc);
    }
}

/// What a sweep did.
#[derive(Clone, Debug, Default)]
pub struct SweepStats {
    /// Memory passes = roundings of the register (1 for full-precision backends).
    pub passes: usize,
    /// Wall-clock seconds of the sweep proper.
    pub secs: f64,
    /// Bytes of the register store.
    pub store_bytes: usize,
    /// Packed `:h` exponent-window underflow / overflow counts (0 elsewhere).
    pub underflow: u64,
    /// See `underflow`.
    pub overflow: u64,
}

/// A register backend for tail-open runs: runs a [`SweepPlan`] from
/// `|0..0>` and then answers random reads of the swept register (for the
/// read-only tail pass). Implementations may keep their allocation between
/// sweeps (the run loop allocates once).
pub trait SweepBackend: Sync {
    /// Short description (format, precision, device).
    fn describe(&self) -> String;
    /// Runs `plan.ops` on a fresh `|0..0>` register; calls `on_pass(j, total)`
    /// after memory pass `j` (1-based).
    fn sweep(
        &mut self,
        plan: &SweepPlan,
        on_pass: &mut dyn FnMut(usize, usize),
    ) -> Result<SweepStats, SimError>;
    /// Raw register value at index `i` after the last sweep (no plan scale).
    fn amp(&self, i: usize) -> Complex64;
    /// Raw register values `start..start + out.len()` (the tail pass reads
    /// rows of `2^11..2^13` entries this way). The default calls `amp` per
    /// entry; block-coded stores should decode whole blocks.
    fn read_run(&self, start: usize, out: &mut [Complex64]) {
        for (k, o) in out.iter_mut().enumerate() {
            *o = self.amp(start + k);
        }
    }
    /// Passes (roundings) the backend would use for `plan`, without running it.
    fn count_passes(&self, plan: &SweepPlan) -> usize;
}

/// One tail-open batch: sweep, then the tail pass.
pub fn run_tail<B: SweepBackend + ?Sized>(
    be: &mut B,
    tp: &TailPlan,
    on_pass: &mut dyn FnMut(usize, usize),
) -> Result<(Vec<Complex64>, SweepStats, f64), SimError> {
    let st = be.sweep(&tp.sweep, on_pass)?;
    let t = std::time::Instant::now();
    let amps = tail_amplitudes_backend(tp, &*be);
    Ok((amps, st, t.elapsed().as_secs_f64()))
}

/// Full-precision CPU register (f64 or f32), the cache-blocked executor.
pub struct CpuExact<T: Real> {
    /// Executor configuration.
    pub cfg: BlockConfig,
    sv: Option<StateVector<T>>,
}

impl<T: Real> CpuExact<T> {
    /// A backend with executor configuration `cfg`.
    pub fn new(cfg: BlockConfig) -> Self {
        CpuExact { cfg, sv: None }
    }
}

impl<T: Real> SweepBackend for CpuExact<T> {
    fn describe(&self) -> String {
        format!("cpu-{}", 8 * std::mem::size_of::<T>())
    }
    fn sweep(
        &mut self,
        plan: &SweepPlan,
        on_pass: &mut dyn FnMut(usize, usize),
    ) -> Result<SweepStats, SimError> {
        let t = std::time::Instant::now();
        self.sv = None;
        // the state-vector cap (MAX_STATE_BYTES) is for interactive use; a
        // reference sweep may want a bigger register, so allocate directly
        let len = 1usize << plan.width;
        let mut amps: Vec<num_complex::Complex<T>> = Vec::new();
        amps.try_reserve_exact(len)
            .map_err(|_| SimError::TooLarge {
                what: "exact chain-sweep register (allocation failed)",
                bytes: (len * 2 * std::mem::size_of::<T>()) as u128,
                limit: 0,
            })?;
        amps.resize(len, num_complex::Complex::new(T::zero(), T::zero()));
        amps[0] = num_complex::Complex::new(T::one(), T::zero());
        let mut sv = StateVector::<T>::from_amplitudes(amps);
        sv.apply_kops_blocked(&plan.ops, &self.cfg);
        on_pass(1, 1);
        let bytes = sv.amplitudes().len() * 2 * std::mem::size_of::<T>();
        self.sv = Some(sv);
        Ok(SweepStats {
            passes: 1,
            secs: t.elapsed().as_secs_f64(),
            store_bytes: bytes,
            ..Default::default()
        })
    }
    fn amp(&self, i: usize) -> Complex64 {
        self.sv.as_ref().expect("sweep first").amplitude(i)
    }
    fn count_passes(&self, _plan: &SweepPlan) -> usize {
        1
    }
}

/// The packed low-precision store ([`crate::engines::chain_packed`]),
/// allocated once and re-zeroed between sweeps.
pub struct CpuPacked {
    /// Storage format.
    pub lp: crate::engines::chain_lowprec::LowPrec,
    /// Executor configuration (kernels, stage preparation).
    pub cfg: BlockConfig,
    /// Gather-buffer bits of the big-buffer planner.
    pub l: usize,
    /// Gathered (non-contiguous) bits per stage.
    pub slots: usize,
    /// Fuse 1q gates before planning.
    pub fuse: bool,
    store: Option<crate::engines::chain_packed::PackedStore>,
}

impl CpuPacked {
    /// A packed backend (store allocated on the first sweep, or with [`Self::allocate`]).
    pub fn new(
        lp: crate::engines::chain_lowprec::LowPrec,
        cfg: BlockConfig,
        l: usize,
        slots: usize,
    ) -> Self {
        CpuPacked {
            lp,
            cfg,
            l,
            slots,
            fuse: true,
            store: None,
        }
    }

    /// Allocates the store for a `width`-bit register now (fails cleanly).
    pub fn allocate(&mut self, width: usize) -> Result<usize, SimError> {
        use crate::engines::chain_packed::PackedStore;
        if self.store.as_ref().map(|s| s.width()) != Some(width) {
            self.store = None;
            self.store = Some(PackedStore::new(width, &self.lp)?);
        }
        Ok(self.store.as_ref().unwrap().bytes())
    }

    /// The store (after a sweep).
    pub fn store(&self) -> Option<&crate::engines::chain_packed::PackedStore> {
        self.store.as_ref()
    }

    /// The memory passes of `plan`.
    pub fn stages(&self, plan: &SweepPlan) -> Vec<crate::engines::blocked::Stage> {
        let ops = if self.fuse {
            crate::engines::blocked::fuse_1q(&plan.ops, plan.width, false)
        } else {
            plan.ops.clone()
        };
        crate::engines::chain_packed::packed_stages(&ops, plan.width, self.l, self.slots)
    }
}

impl SweepBackend for CpuPacked {
    fn describe(&self) -> String {
        format!("cpu-packed {} l={} slots={}", self.lp, self.l, self.slots)
    }
    fn sweep(
        &mut self,
        plan: &SweepPlan,
        on_pass: &mut dyn FnMut(usize, usize),
    ) -> Result<SweepStats, SimError> {
        let stages = self.stages(plan);
        self.allocate(plan.width)?;
        let cfg = self.cfg.clone();
        let store = self.store.as_mut().unwrap();
        store.reset();
        let t = std::time::Instant::now();
        let n = stages.len();
        for (j, st) in stages.iter().enumerate() {
            store.run_stage(st, &cfg)?;
            on_pass(j + 1, n);
        }
        Ok(SweepStats {
            passes: n,
            secs: t.elapsed().as_secs_f64(),
            store_bytes: store.bytes(),
            underflow: store.underflow(),
            overflow: store.overflow(),
        })
    }
    fn amp(&self, i: usize) -> Complex64 {
        let z = self.store.as_ref().expect("sweep first").get(i);
        Complex64::new(z.re as f64, z.im as f64)
    }
    fn read_run(&self, start: usize, out: &mut [Complex64]) {
        self.store
            .as_ref()
            .expect("sweep first")
            .read_run(start, out);
    }
    fn count_passes(&self, plan: &SweepPlan) -> usize {
        self.stages(plan).len()
    }
}

/// Applies `segs[t+1] (left + Z_w right)`: the sum over bond `t` of two
/// sibling vectors (`left` for `β_t = 0`), result in `right`.
#[inline]
fn combine(tp: &TailPlan, t: usize, left: &[Complex64], right: &mut [Complex64]) {
    let w = tp.m - 1;
    for (s, (r, l)) in right.iter_mut().zip(left).enumerate() {
        *r = if (s >> w) & 1 == 0 { *l + *r } else { *l - *r };
    }
    apply_ops(&tp.segs[t + 1], right);
}

/// Binary-counter tree reduction over consecutive bonds `base, base+1, ..`:
/// vectors are pushed in increasing order of their bond bits (bond `base`
/// = least significant).
struct Reducer {
    base: usize,
    stack: Vec<Option<Vec<Complex64>>>,
}

impl Reducer {
    fn new(base: usize, levels: usize) -> Self {
        Reducer {
            base,
            stack: vec![None; levels + 1],
        }
    }
    fn push(&mut self, tp: &TailPlan, mut node: Vec<Complex64>) {
        let mut j = 0;
        while let Some(left) = self.stack[j].take() {
            combine(tp, self.base + j, &left, &mut node);
            j += 1;
        }
        self.stack[j] = Some(node);
    }
    fn finish(mut self) -> Vec<Complex64> {
        let top = self.stack.len() - 1;
        let r = self.stack[top].take().expect("a power of two of pushes");
        assert!(self.stack.iter().all(|x| x.is_none()));
        r
    }
}

/// GEMM form, for registers whose bit `t` is bond `t` (time order). With
/// the early bonds `0..h` as the low register bits: the `2^h` tail vectors
/// `v_E` after the early bonds form a `2^m × 2^h` matrix, each row block of
/// the register (`β_L` fixed, `2^h` contiguous entries) times it is the
/// tail state `u(β_L)` before bond `h`, and a binary-counter tree over the
/// late bonds sums `segs .. Z^{β_t} u`. Cost `2^K · 2^m` complex MACs as a
/// GEMM plus `2^(K-h) · |seg| · 2^m`; the register is read once, in order.
/// `h` defaults to `max(0, 19 - m)` (V of 8 MiB) capped at `K`.
pub fn tail_amplitudes_gemm(
    tp: &TailPlan,
    reg: &(dyn Fn(usize, &mut [Complex64]) + Sync),
    h: Option<usize>,
) -> Vec<Complex64> {
    use faer::linalg::matmul::matmul;
    use faer::{Accum, MatMut, MatRef, Par};
    assert!(gemm_ok(tp), "register layout is not bond-ordered");
    let m = tp.m;
    let dim = 1usize << m;
    let k = tp.bonds.len();
    let h = h.unwrap_or(19usize.saturating_sub(m)).min(k);
    let w = m - 1;
    // V: v_E for E in 0..2^h (bit t of E = β_t), each contiguous
    let mut v = vec![C0; dim];
    v[0] = C1;
    apply_ops(&tp.segs[0], &mut v);
    for t in 0..h {
        let half = v.len();
        v.extend_from_within(..);
        for (e, x) in v.chunks_exact_mut(dim).enumerate() {
            if e >= half / dim {
                flip(x, w);
            }
            apply_ops(&tp.segs[t + 1], x);
        }
    }
    let ve = 1usize << h;
    let vt = MatRef::from_column_major_slice(&v[..], dim, ve);
    let late = k - h;
    let top = late.min(10);
    let per = late - top; // late bits inside one chunk
    let rows_per_chunk = 1usize << per;
    let batch = rows_per_chunk.min(64);
    let chunk = |c: usize| -> Vec<Complex64> {
        let mut red = Reducer::new(h, per);
        let mut rb = vec![C0; ve * batch];
        let mut ub = vec![C0; dim * batch];
        let mut r0 = 0;
        while r0 < rows_per_chunk {
            for (b, col) in rb.chunks_exact_mut(ve).enumerate() {
                reg(((c * rows_per_chunk) + r0 + b) << h, col);
            }
            let rhs = MatRef::from_column_major_slice(&rb[..], ve, batch);
            let dst = MatMut::from_column_major_slice_mut(&mut ub[..], dim, batch);
            matmul(dst, Accum::Replace, vt, rhs, C1, Par::Seq);
            for u in ub.chunks_exact(dim) {
                red.push(tp, u.to_vec());
            }
            r0 += batch;
        }
        red.finish()
    };
    let parts: Vec<Vec<Complex64>> = (0..1usize << top).into_par_iter().map(chunk).collect();
    let mut red = Reducer::new(h + per, top);
    for p in parts {
        red.push(tp, p);
    }
    let acc = red.finish();
    let mut out = vec![C0; dim];
    for (s, a) in acc.iter().enumerate() {
        out[reverse_bits(s as u128, m) as usize] = a * tp.sweep.scale;
    }
    out
}
