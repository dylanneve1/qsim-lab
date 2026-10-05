//! Compile once, bind many times.
//!
//! [`CompiledCircuit::compile`] runs every *structural* decision once:
//! light cone with respect to the observable, independent components,
//! lowering to executor ops, single-qubit fusion, the blocked executor's
//! stage plan and diagonal schedule, and the per-stage preparation of every
//! stage that does not depend on a parameter (constant folding: runs of
//! fixed gates are multiplied out at compile time). It also runs the
//! constant prefix (everything before the first parameter-dependent stage)
//! once and keeps that state.
//!
//! [`CompiledCircuit::bind`] then only evaluates the numeric recipes of the
//! parameter-dependent executor ops (a few 2x2 products and `e^{iθ}`s) and
//! re-prepares the stages that contain one. Nothing else is recomputed.
//!
//! Exactness: the structure of the plan never depends on a numeric value
//! that a parameter can change. Decisions the plain executor takes from
//! values (drop a phase equal to 1, treat a product as diagonal, pick the
//! real or `X` kernel) are taken here from the gate *kinds* for
//! parameter-dependent ops (a run is diagonal iff every factor is diagonal
//! for all angles), and kernel kinds are re-derived from the bound numbers
//! by the stage preparation. `tests/graph.rs` checks every path against an
//! independent reference state vector.

use super::observable::{diagonal_expectation, pauli_expectation, Observable};
use super::param::{Angle, POp, ParamCircuit};
use crate::blocked::{
    lower_gate, plan_stages, prepare_stage_cfg, prepare_stage_mapped, run_prepared_stage,
    schedule_diag_order, BlockConfig, KOp, OpLoc, PreparedStage, Stage,
};
use crate::circuit::SimError;
use crate::gate::{mat2_mul, Gate, Mat2};
use crate::statevector::{state_bytes, StateVectorF64, MAX_STATE_BYTES};
use num_complex::Complex64;
use rayon::prelude::*;

const C0: Complex64 = Complex64::new(0.0, 0.0);
const C1: Complex64 = Complex64::new(1.0, 0.0);

/// Options of [`CompiledCircuit::compile`].
#[derive(Clone, Debug)]
pub struct GraphOptions {
    /// Drop ops outside the observable's backward light cone.
    pub light_cone: bool,
    /// Drop trailing diagonal ops when the observable is diagonal.
    pub diagonal_suffix: bool,
    /// Simulate independent components separately.
    pub components: bool,
    /// Keep the state after the constant prefix and start every bind there.
    pub prefix_cache: bool,
    /// Executor options (block size, fusion, diagonal scheduling, SIMD).
    pub block: BlockConfig,
    /// Largest dense register (bytes).
    pub mem_bytes: u128,
    /// Phase gadgets (`POp::ZString`) up to this width become diagonal
    /// terms; wider ones run as CNOT ladders.
    pub max_zstring: usize,
    /// Bind by patching numbers into the prepared stages in place (`false`:
    /// re-prepare every parameter-dependent stage on each bind).
    pub patch_bind: bool,
    /// Phase-polynomial region rewrite ([`super::rewrite`]); the rewritten
    /// circuit is used only if its executor plan is cheaper (fewer stages,
    /// then fewer executor ops, summed over the parts).
    pub rewrite: Option<super::rewrite::RewriteOptions>,
}

impl Default for GraphOptions {
    fn default() -> Self {
        GraphOptions {
            light_cone: true,
            diagonal_suffix: true,
            components: true,
            prefix_cache: true,
            block: BlockConfig::default(),
            mem_bytes: MAX_STATE_BYTES,
            max_zstring: 6,
            patch_bind: true,
            rewrite: Some(super::rewrite::RewriteOptions::default()),
        }
    }
}

/// Rewrites phase gadgets wider than `max_k` (and of width 1) into
/// CNOT ladders around an `Rz`, so only narrow gadgets become diagonal
/// terms (`2^(k-1)` phase terms each).
fn expand_wide(ops: &[POp], max_k: usize) -> Vec<POp> {
    let mut out = Vec::with_capacity(ops.len());
    for op in ops {
        match op {
            POp::ZString(qs, a) if qs.len() == 1 => out.push(POp::Rz(qs[0], a.clone())),
            POp::ZString(qs, _) if qs.is_empty() => {}
            POp::ZString(qs, a) if qs.len() > max_k => {
                for w in qs.windows(2) {
                    out.push(POp::Fixed(Gate::Cnot(w[0], w[1])));
                }
                out.push(POp::Rz(*qs.last().expect("nonempty"), a.clone()));
                for w in qs.windows(2).rev() {
                    out.push(POp::Fixed(Gate::Cnot(w[0], w[1])));
                }
            }
            _ => out.push(op.clone()),
        }
    }
    out
}

/// What a compiled op's numbers are made of.
#[derive(Clone, Debug)]
enum Fac {
    Const(Mat2),
    /// A single-qubit parameterised op (bound through [`Gate::matrix_1q`]).
    Op(POp),
}

#[derive(Clone, Debug)]
enum Recipe {
    /// `U1` whose matrix is the product of the factors (first applied first).
    U1(Vec<Fac>),
    /// A fused diagonal 1q run `diag(d0, d1)`: the op is the phase term
    /// `d1/d0` on `|1>` and `d0` goes to the global phase.
    DiagRun(Vec<Fac>),
    /// Phase term `c · e^{i a}`.
    Phase(Complex64, Angle),
}

fn fac_matrix(f: &Fac, params: &[f64]) -> Mat2 {
    match f {
        Fac::Const(m) => *m,
        Fac::Op(op) => {
            let mut g = Vec::with_capacity(1);
            op.bind_into(params, &mut g);
            g[0].matrix_1q().expect("single-qubit op")
        }
    }
}

fn product(fs: &[Fac], params: &[f64]) -> Mat2 {
    let mut m = fac_matrix(&fs[0], params);
    for f in &fs[1..] {
        m = mat2_mul(&fac_matrix(f, params), &m);
    }
    m
}

/// Statistics of a compilation.
#[derive(Clone, Debug, Default)]
pub struct GraphStats {
    pub ops_in: usize,
    pub ops_after_cone: usize,
    /// `(qubits, executor ops, stages, parameter-dependent stages)` per part.
    pub parts: Vec<(usize, usize, usize, usize)>,
    pub compile_secs: f64,
    /// The phase-region rewrite was applied.
    pub rewritten: bool,
}

/// One independent component, compiled for the blocked executor.
#[derive(Clone)]
struct DenseProgram {
    n: usize,
    /// Template ops (numbers at compile time; parameter-dependent ones are
    /// overwritten on bind).
    kops: Vec<KOp>,
    /// `(kop index, recipe)` for every parameter-dependent op.
    recipes: Vec<(usize, Recipe)>,
    stages: Vec<(Vec<usize>, Vec<usize>)>, // (inner, kop order)
    param_stage: Vec<bool>,
    /// Constant stages, prepared with every executor option.
    prepared: Vec<Option<PreparedStage<f64>>>,
    /// Parameter-dependent stages: prepared once (ops in schedule order,
    /// no fusion) with the location of every op, patched on bind.
    templates: Vec<Option<PreparedStage<f64>>>,
    /// `(stage, location)` of every recipe's op.
    recipe_loc: Vec<(usize, OpLoc)>,
    /// Patch numbers in place (`false`: re-prepare bound stages).
    patch: bool,
    block: BlockConfig,
    /// Global phase `c · e^{i a}` from the ops (diagonal runs add theirs on bind).
    global: (Complex64, Angle),
    /// First parameter-dependent stage, and the state just before it.
    prefix_end: usize,
    prefix_state: Option<Vec<Complex64>>,
    simd: bool,
}

/// A part's bound numbers.
struct BoundPart {
    stages: Vec<Option<PreparedStage<f64>>>,
    phase: Complex64,
}

impl DenseProgram {
    fn compile(n: usize, ops: &[POp], opts: &GraphOptions) -> Result<Self, SimError> {
        let bytes = state_bytes::<f64>(n);
        let cap = opts.mem_bytes.min(MAX_STATE_BYTES);
        if bytes > cap {
            return Err(SimError::TooLarge {
                what: "graph compiler: dense component",
                bytes,
                limit: cap,
            });
        }
        let mut kops: Vec<KOp> = Vec::new();
        let mut rec: Vec<Option<Recipe>> = Vec::new();
        let mut global = (C1, Angle::constant(0.0));
        // pending 1q run per qubit
        let mut pending: Vec<Vec<Fac>> = vec![Vec::new(); n];
        let flush = |q: usize,
                     run: &mut Vec<Fac>,
                     kops: &mut Vec<KOp>,
                     rec: &mut Vec<Option<Recipe>>,
                     global: &mut (Complex64, Angle)| {
            if run.is_empty() {
                return;
            }
            let fs = std::mem::take(run);
            // merge adjacent constants (constant folding)
            let mut folded: Vec<Fac> = Vec::with_capacity(fs.len());
            for f in fs {
                match (folded.last_mut(), f) {
                    (Some(Fac::Const(a)), Fac::Const(b)) => *a = mat2_mul(&b, a),
                    (_, f) => folded.push(f),
                }
            }
            let param = folded.iter().any(|f| matches!(f, Fac::Op(_)));
            if !param {
                let m = match folded[0] {
                    Fac::Const(m) => m,
                    _ => unreachable!(),
                };
                if m[0][1] == C0 && m[1][0] == C0 {
                    let (d0, d1) = (m[0][0], m[1][1]);
                    if d0 != C1 {
                        global.0 *= d0;
                    }
                    let f = d1 / d0;
                    if f != C1 {
                        kops.push(KOp::Phase {
                            mask: 1 << q,
                            pat: 1 << q,
                            f,
                        });
                        rec.push(None);
                    }
                } else {
                    kops.push(KOp::U1 { q, m, ctrl: 0 });
                    rec.push(None);
                }
                return;
            }
            let diag = folded.iter().all(|f| match f {
                Fac::Const(m) => m[0][1] == C0 && m[1][0] == C0,
                Fac::Op(op) => op.is_diagonal(),
            });
            if diag {
                kops.push(KOp::Phase {
                    mask: 1 << q,
                    pat: 1 << q,
                    f: Complex64::new(0.6, 0.8),
                });
                rec.push(Some(Recipe::DiagRun(folded)));
            } else {
                kops.push(KOp::U1 {
                    q,
                    m: [[C0, C1], [C1, C0]],
                    ctrl: 0,
                });
                rec.push(Some(Recipe::U1(folded)));
            }
        };
        let push_phase = |kops: &mut Vec<KOp>,
                          rec: &mut Vec<Option<Recipe>>,
                          mask: usize,
                          pat: usize,
                          a: &Angle| {
            if a.is_const() {
                kops.push(KOp::Phase {
                    mask,
                    pat,
                    f: Complex64::from_polar(1.0, a.c0),
                });
                rec.push(None);
            } else {
                kops.push(KOp::Phase {
                    mask,
                    pat,
                    f: Complex64::new(0.6, 0.8),
                });
                rec.push(Some(Recipe::Phase(C1, a.clone())));
            }
        };
        let ops = expand_wide(ops, opts.max_zstring);
        let mut tmp = Vec::new();
        for op in &ops {
            // single-qubit ops accumulate
            match op {
                POp::Fixed(g) if g.arity() == 1 => {
                    let q = g.qubits()[0];
                    pending[q].push(Fac::Const(g.matrix_1q().expect("1q")));
                    continue;
                }
                POp::Rx(q, _)
                | POp::Ry(q, _)
                | POp::Rz(q, _)
                | POp::Phase(q, _)
                | POp::U(q, ..) => {
                    if op.is_param() {
                        pending[*q].push(Fac::Op(op.clone()));
                    } else {
                        let mut g = Vec::new();
                        op.bind_into(&[], &mut g);
                        pending[*q].push(Fac::Const(g[0].matrix_1q().expect("1q")));
                    }
                    continue;
                }
                _ => {}
            }
            let h = Gate::H(0).matrix_1q().expect("H");
            if let POp::Rxx(a, b, _) = op {
                pending[*a].push(Fac::Const(h));
                pending[*b].push(Fac::Const(h));
            }
            for q in op.qubits() {
                flush(q, &mut pending[q], &mut kops, &mut rec, &mut global);
            }
            match op {
                POp::Fixed(g) => {
                    tmp.clear();
                    lower_gate(g, &mut tmp);
                    for k in &tmp {
                        kops.push(*k);
                        rec.push(None);
                    }
                }
                POp::CPhase(a, b, ang) => {
                    let m = (1 << a) | (1 << b);
                    push_phase(&mut kops, &mut rec, m, m, ang);
                }
                POp::Rzz(a, b, ang) | POp::Rxx(a, b, ang) => {
                    // e^{-iθ/2} diag(1, e^{iθ}, e^{iθ}, 1) (Rxx: H⊗H queued
                    // before and after, see above)
                    global.1 = global.1.plus(&ang.times(-0.5));
                    let m = (1 << a) | (1 << b);
                    for pat in [1 << a, 1 << b] {
                        push_phase(&mut kops, &mut rec, m, pat, ang);
                    }
                    if matches!(op, POp::Rxx(..)) {
                        let h = Gate::H(0).matrix_1q().expect("H");
                        pending[*a].push(Fac::Const(h));
                        pending[*b].push(Fac::Const(h));
                    }
                }
                POp::ZString(qs, ang) => {
                    // e^{-iθ/2} on even parity, e^{iθ/2} on odd parity
                    global.1 = global.1.plus(&ang.times(-0.5));
                    let m: usize = qs.iter().map(|&q| 1usize << q).sum();
                    let k = qs.len();
                    for sub in 0..1usize << k {
                        if sub.count_ones() & 1 == 1 {
                            let pat: usize = (0..k)
                                .filter(|j| sub >> j & 1 == 1)
                                .map(|j| 1usize << qs[j])
                                .sum();
                            push_phase(&mut kops, &mut rec, m, pat, ang);
                        }
                    }
                }
                _ => unreachable!("single-qubit ops handled above"),
            }
        }
        for q in 0..n {
            let mut run = std::mem::take(&mut pending[q]);
            flush(q, &mut run, &mut kops, &mut rec, &mut global);
        }
        let mut recipes = Vec::new();
        let mut slot = vec![usize::MAX; kops.len()];
        for (i, r) in rec.into_iter().enumerate() {
            if let Some(r) = r {
                slot[i] = recipes.len();
                recipes.push((i, r));
            }
        }
        // stage plan: contiguous ranges of the op list (plan_stages keeps order)
        let l = opts.block.block_bits(n, std::mem::size_of::<Complex64>());
        let planned = plan_stages(&kops, n, l, opts.block.slots);
        let mut stages = Vec::with_capacity(planned.len());
        let mut start = 0usize;
        for st in &planned {
            let range: Vec<usize> = (start..start + st.ops.len()).collect();
            start += st.ops.len();
            let order = if opts.block.schedule_diag {
                schedule_diag_order(&st.ops)
                    .into_iter()
                    .map(|k| range[k])
                    .collect()
            } else {
                range
            };
            stages.push((st.inner.clone(), order));
        }
        let param_stage: Vec<bool> = stages
            .iter()
            .map(|(_, ord)| ord.iter().any(|&k| slot[k] != usize::MAX))
            .collect();
        let mut prog = DenseProgram {
            n,
            kops,
            recipes,
            stages,
            param_stage,
            prepared: Vec::new(),
            templates: Vec::new(),
            recipe_loc: Vec::new(),
            patch: opts.patch_bind,
            block: opts.block.clone(),
            global,
            prefix_end: 0,
            prefix_state: None,
            simd: opts.block.simd,
        };
        prog.prepared = (0..prog.stages.len())
            .map(|s| (!prog.param_stage[s]).then(|| prog.prepare(s, &prog.kops)))
            .collect();
        // where each parameter-dependent op lands in its stage
        let mut at = vec![(usize::MAX, usize::MAX); prog.kops.len()];
        for (s, (_, order)) in prog.stages.iter().enumerate() {
            for (j, &k) in order.iter().enumerate() {
                at[k] = (s, j);
            }
        }
        let mut templates: Vec<Option<PreparedStage<f64>>> = vec![None; prog.stages.len()];
        let mut locs: Vec<Vec<OpLoc>> = vec![Vec::new(); prog.stages.len()];
        if prog.patch {
            for s in 0..prog.stages.len() {
                if prog.param_stage[s] {
                    let (inner, order) = &prog.stages[s];
                    let (p, l) = prepare_stage_mapped(
                        &Stage {
                            inner: inner.clone(),
                            ops: order.iter().map(|&k| prog.kops[k]).collect(),
                        },
                        n,
                    );
                    templates[s] = Some(p);
                    locs[s] = l;
                }
            }
            prog.recipe_loc = prog
                .recipes
                .iter()
                .map(|(k, _)| {
                    let (s, j) = at[*k];
                    (s, locs[s][j])
                })
                .collect();
        }
        prog.templates = templates;
        prog.prefix_end = prog
            .param_stage
            .iter()
            .position(|&p| p)
            .unwrap_or(prog.stages.len());
        Ok(prog)
    }

    /// Runs the constant prefix once and keeps the state.
    fn compute_prefix(&mut self) -> Result<(), SimError> {
        if self.prefix_end == 0 || self.prefix_state.is_some() {
            return Ok(());
        }
        let mut sv = StateVectorF64::try_new(self.n)?;
        for s in 0..self.prefix_end {
            run_prepared_stage(
                sv.amplitudes_mut(),
                self.n,
                self.prepared[s].as_ref().expect("constant stage"),
                self.simd,
            );
        }
        self.prefix_state = Some(sv.amplitudes().to_vec());
        Ok(())
    }

    fn prepare(&self, s: usize, kops: &[KOp]) -> PreparedStage<f64> {
        let (inner, order) = &self.stages[s];
        prepare_stage_cfg(
            &Stage {
                inner: inner.clone(),
                ops: order.iter().map(|&k| kops[k]).collect(),
            },
            self.n,
            &self.block,
        )
    }

    /// Bind by patching the prepared templates in place.
    fn bind_patch(&self, params: &[f64]) -> BoundPart {
        let mut phase = self.global.0 * Complex64::from_polar(1.0, self.global.1.eval(params));
        let mut stages = self.templates.clone();
        for ((_, r), &(s, loc)) in self.recipes.iter().zip(&self.recipe_loc) {
            let st = stages[s].as_mut().expect("template");
            match r {
                Recipe::U1(fs) => st.set_u1(loc, &product(fs, params)),
                Recipe::DiagRun(fs) => {
                    let m = product(fs, params);
                    phase *= m[0][0];
                    st.set_phase(loc, m[1][1] / m[0][0]);
                }
                Recipe::Phase(c, a) => {
                    st.set_phase(loc, c * Complex64::from_polar(1.0, a.eval(params)))
                }
            }
        }
        BoundPart { stages, phase }
    }

    fn bind(&self, params: &[f64]) -> BoundPart {
        if self.patch {
            return self.bind_patch(params);
        }
        let mut kops = self.kops.clone();
        let mut phase = self.global.0 * Complex64::from_polar(1.0, self.global.1.eval(params));
        for (k, r) in &self.recipes {
            match r {
                Recipe::U1(fs) => {
                    if let KOp::U1 { m, .. } = &mut kops[*k] {
                        *m = product(fs, params);
                    }
                }
                Recipe::DiagRun(fs) => {
                    let m = product(fs, params);
                    phase *= m[0][0];
                    if let KOp::Phase { f, .. } = &mut kops[*k] {
                        *f = m[1][1] / m[0][0];
                    }
                }
                Recipe::Phase(c, a) => {
                    if let KOp::Phase { f, .. } = &mut kops[*k] {
                        *f = c * Complex64::from_polar(1.0, a.eval(params));
                    }
                }
            }
        }
        let stages = (0..self.stages.len())
            .map(|s| self.param_stage[s].then(|| self.prepare(s, &kops)))
            .collect();
        BoundPart { stages, phase }
    }

    fn run(&self, b: &BoundPart) -> Result<StateVectorF64, SimError> {
        let (mut sv, from) = match &self.prefix_state {
            Some(a) => (StateVectorF64::from_amplitudes(a.clone()), self.prefix_end),
            None => (StateVectorF64::try_new(self.n)?, 0),
        };
        let amps = sv.amplitudes_mut();
        for s in from..self.stages.len() {
            let p = b.stages[s]
                .as_ref()
                .or(self.prepared[s].as_ref())
                .expect("stage prepared");
            run_prepared_stage(amps, self.n, p, self.simd);
        }
        Ok(sv)
    }
}

/// A parameterised circuit compiled for one observable (or for the full
/// state when the observable is `None`).
#[derive(Clone)]
pub struct CompiledCircuit {
    num_qubits: usize,
    num_params: usize,
    /// Global qubits of each part (local index = position).
    parts: Vec<(Vec<usize>, DenseProgram)>,
    /// Qubits with no surviving op (state |0>).
    idle: Vec<usize>,
    obs: Option<Observable>,
    gphase: Angle,
    stats: GraphStats,
}

/// A [`CompiledCircuit`] with numbers for one parameter vector.
pub struct BoundCircuit<'a> {
    cc: &'a CompiledCircuit,
    parts: Vec<BoundPart>,
    gphase: Complex64,
}

impl CompiledCircuit {
    /// Compiles `pc` for expectation values of `obs` (or, with `None`, for
    /// the full state / amplitudes: no light cone).
    pub fn compile(
        pc: &ParamCircuit,
        obs: Option<&Observable>,
        opts: &GraphOptions,
    ) -> Result<Self, SimError> {
        let t0 = std::time::Instant::now();
        let plain = Self::compile_one(pc, obs, opts)?;
        let mut best = plain;
        if let Some(ro) = &opts.rewrite {
            let (rc, st) = super::rewrite::phase_regions(pc, ro);
            if st.rewritten > 0 {
                let cand = Self::compile_one(&rc, obs, opts)?;
                let cost = |c: &CompiledCircuit| {
                    c.stats
                        .parts
                        .iter()
                        .fold((0usize, 0usize), |a, p| (a.0 + p.2, a.1 + p.1))
                };
                if cost(&cand) < cost(&best) {
                    best = cand;
                    best.stats.rewritten = true;
                }
            }
        }
        if opts.prefix_cache {
            for (_, p) in best.parts.iter_mut() {
                p.compute_prefix()?;
            }
        }
        best.stats.compile_secs = t0.elapsed().as_secs_f64();
        Ok(best)
    }

    fn compile_one(
        pc: &ParamCircuit,
        obs: Option<&Observable>,
        opts: &GraphOptions,
    ) -> Result<Self, SimError> {
        let t0 = std::time::Instant::now();
        let n = pc.num_qubits;
        if n > 128 {
            return Err(SimError::NotSupported {
                what: "graph compiler: more than 128 qubits",
            });
        }
        // global phase ops are not part of the graph
        let mut gphase = Angle::constant(0.0);
        let mut keep: Vec<bool> = pc
            .ops
            .iter()
            .map(|o| match o {
                POp::Global(a) => {
                    gphase = gphase.plus(a);
                    false
                }
                _ => true,
            })
            .collect();
        if let (Some(o), true) = (obs, opts.light_cone || opts.diagonal_suffix) {
            let sup = o.support();
            let mut live: Vec<bool> = (0..n).map(|q| sup >> q & 1 == 1).collect();
            if !opts.light_cone {
                live = vec![true; n];
            }
            // qubits whose remaining suffix is diagonal (Z observable)
            let mut diag_tail = vec![opts.diagonal_suffix && o.is_diagonal(); n];
            for (i, op) in pc.ops.iter().enumerate().rev() {
                if !keep[i] {
                    continue;
                }
                let qs = op.qubits();
                if !qs.iter().any(|&q| live[q]) {
                    keep[i] = false;
                    continue;
                }
                if op.is_diagonal() && qs.iter().all(|&q| diag_tail[q]) {
                    keep[i] = false;
                    continue;
                }
                for &q in &qs {
                    live[q] = true;
                    diag_tail[q] = false;
                }
            }
        }
        let kept: Vec<&POp> = pc
            .ops
            .iter()
            .zip(&keep)
            .filter(|(_, &k)| k)
            .map(|(o, _)| o)
            .collect();
        // components (union-find over the kept ops)
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut [usize], mut x: usize) -> usize {
            while p[x] != x {
                p[x] = p[p[x]];
                x = p[x];
            }
            x
        }
        let mut used = vec![false; n];
        for op in &kept {
            let qs = op.qubits();
            for &q in &qs {
                used[q] = true;
            }
            if opts.components {
                for w in qs.windows(2) {
                    let (a, b) = (find(&mut parent, w[0]), find(&mut parent, w[1]));
                    parent[a] = b;
                }
            } else {
                for &q in &qs {
                    let (a, b) = (find(&mut parent, q), find(&mut parent, 0));
                    parent[a] = b;
                }
            }
        }
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut gid = vec![usize::MAX; n];
        for q in 0..n {
            if !used[q] {
                continue;
            }
            let r = find(&mut parent, q);
            if gid[r] == usize::MAX {
                gid[r] = groups.len();
                groups.push(Vec::new());
            }
            groups[gid[r]].push(q);
        }
        let mut part_ops: Vec<Vec<POp>> = vec![Vec::new(); groups.len()];
        let mut local = vec![0usize; n];
        for g in &groups {
            for (j, &q) in g.iter().enumerate() {
                local[q] = j;
            }
        }
        for op in &kept {
            let q0 = op.qubits()[0];
            let g = gid[find(&mut parent, q0)];
            part_ops[g].push(op.map_qubits(|q| local[q]));
        }
        let mut parts = Vec::with_capacity(groups.len());
        let mut stats = GraphStats {
            ops_in: pc.ops.len(),
            ops_after_cone: kept.len(),
            ..Default::default()
        };
        for (g, ops) in groups.into_iter().zip(part_ops) {
            let prog = DenseProgram::compile(g.len(), &ops, opts)?;
            stats.parts.push((
                g.len(),
                prog.kops.len(),
                prog.stages.len(),
                prog.param_stage.iter().filter(|&&p| p).count(),
            ));
            parts.push((g, prog));
        }
        let idle = (0..n).filter(|&q| !used[q]).collect();
        stats.compile_secs = t0.elapsed().as_secs_f64();
        Ok(CompiledCircuit {
            num_qubits: n,
            num_params: pc.num_params,
            parts,
            idle,
            obs: obs.cloned(),
            gphase,
            stats,
        })
    }

    pub fn stats(&self) -> &GraphStats {
        &self.stats
    }

    pub fn num_params(&self) -> usize {
        self.num_params
    }

    /// Evaluates the numeric recipes for `params`.
    pub fn bind(&self, params: &[f64]) -> Result<BoundCircuit<'_>, SimError> {
        if params.len() != self.num_params {
            return Err(SimError::NotSupported {
                what: "CompiledCircuit::bind: wrong number of parameters",
            });
        }
        Ok(BoundCircuit {
            cc: self,
            parts: self.parts.iter().map(|(_, p)| p.bind(params)).collect(),
            gphase: Complex64::from_polar(1.0, self.gphase.eval(params)),
        })
    }

    /// `<obs>` at every parameter vector. Binds run in parallel (each on
    /// its own state, the executor parallelises inside) as long as one
    /// state per worker fits in [`GraphOptions::mem_bytes`]-sized budget of
    /// 1 GiB; otherwise one after the other.
    pub fn sweep_expectation(&self, params: &[Vec<f64>]) -> Result<Vec<f64>, SimError> {
        let per: u128 = self
            .parts
            .iter()
            .map(|(g, _)| 2 * state_bytes::<f64>(g.len()))
            .sum();
        let workers = rayon::current_num_threads() as u128;
        let one = |p: &Vec<f64>| self.bind(p)?.expectation();
        if per * workers <= MAX_STATE_BYTES {
            params.par_iter().map(one).collect()
        } else {
            params.iter().map(one).collect()
        }
    }
}

impl BoundCircuit<'_> {
    fn check_full(&self) -> Result<(), SimError> {
        if self.cc.obs.is_some() {
            return Err(SimError::NotSupported {
                what:
                    "compiled for an observable (light cone applied): compile with None for states",
            });
        }
        Ok(())
    }

    /// Per-part dense states (local qubit order) with global phases applied.
    fn part_states(&self) -> Result<Vec<StateVectorF64>, SimError> {
        self.cc
            .parts
            .iter()
            .zip(&self.parts)
            .map(|((_, p), b)| p.run(b))
            .collect()
    }

    /// `<obs>` for the observable the circuit was compiled for.
    pub fn expectation(&self) -> Result<f64, SimError> {
        let obs = self.cc.obs.as_ref().ok_or(SimError::NotSupported {
            what: "BoundCircuit::expectation: compiled without an observable",
        })?;
        self.expectation_of(obs)
    }

    /// `<o>` for any observable (must lie in the compiled light cone if the
    /// circuit was compiled for a different one).
    pub fn expectation_of(&self, o: &Observable) -> Result<f64, SimError> {
        let states = self.part_states()?;
        let n = self.cc.num_qubits;
        // local index of every qubit per part
        let locals: Vec<Vec<Option<usize>>> = self
            .cc
            .parts
            .iter()
            .map(|(g, _)| {
                let mut l = vec![None; n];
                for (j, &q) in g.iter().enumerate() {
                    l[q] = Some(j);
                }
                l
            })
            .collect();
        let idle_mask: u128 = self.cc.idle.iter().fold(0, |m, &q| m | 1u128 << q);
        // diagonal terms of a single part are batched into one pass
        let mut total = 0.0;
        let mut batched: Vec<Vec<(f64, u64)>> = vec![Vec::new(); states.len()];
        for t in &o.terms {
            // idle qubits are |0>: X/Y there give 0, Z gives 1
            if t.x & idle_mask != 0 {
                continue;
            }
            let mut touched = Vec::new();
            for (pi, l) in locals.iter().enumerate() {
                let (x, z) = Observable::restrict(t, l);
                if x | z != 0 {
                    touched.push((pi, x, z));
                }
            }
            match touched.as_slice() {
                [] => total += t.coef,
                [(pi, 0, z)] => batched[*pi].push((t.coef, *z)),
                _ => {
                    let mut v = t.coef;
                    for &(pi, x, z) in &touched {
                        v *= pauli_expectation(states[pi].amplitudes(), x, z);
                    }
                    total += v;
                }
            }
        }
        for (pi, terms) in batched.iter().enumerate() {
            if !terms.is_empty() {
                total += diagonal_expectation(states[pi].amplitudes(), terms);
            }
        }
        Ok(total)
    }

    /// The full `2^n` state (qubit `q` = bit `q`), global phase included.
    pub fn statevector(&self) -> Result<StateVectorF64, SimError> {
        self.check_full()?;
        let n = self.cc.num_qubits;
        let bytes = state_bytes::<f64>(n);
        if bytes > MAX_STATE_BYTES {
            return Err(SimError::TooLarge {
                what: "BoundCircuit::statevector",
                bytes,
                limit: MAX_STATE_BYTES,
            });
        }
        let states = self.part_states()?;
        let mut phase = self.gphase;
        for b in &self.parts {
            phase *= b.phase;
        }
        let mut out = vec![C0; 1usize << n];
        // tensor product: iterate over the parts' basis states
        let parts: Vec<(&Vec<usize>, &StateVectorF64)> =
            self.cc.parts.iter().map(|(g, _)| g).zip(&states).collect();
        fn rec(
            parts: &[(&Vec<usize>, &StateVectorF64)],
            idx: usize,
            amp: Complex64,
            out: &mut [Complex64],
        ) {
            match parts.split_first() {
                None => out[idx] = amp,
                Some(((g, sv), rest)) => {
                    for (y, a) in sv.amplitudes().iter().enumerate() {
                        if *a == C0 {
                            continue;
                        }
                        let mut x = idx;
                        for (j, &q) in g.iter().enumerate() {
                            if y >> j & 1 == 1 {
                                x |= 1 << q;
                            }
                        }
                        rec(rest, x, amp * a, out);
                    }
                }
            }
        }
        rec(&parts, 0, phase, &mut out);
        Ok(StateVectorF64::from_amplitudes(out))
    }

    /// `<x|ψ>` for basis states `x`, global phase included.
    pub fn amplitudes(&self, xs: &[u128]) -> Result<Vec<Complex64>, SimError> {
        self.check_full()?;
        let states = self.part_states()?;
        let mut phase = self.gphase;
        for b in &self.parts {
            phase *= b.phase;
        }
        let idle_mask: u128 = self.cc.idle.iter().fold(0, |m, &q| m | 1u128 << q);
        Ok(xs
            .iter()
            .map(|&x| {
                if x & idle_mask != 0 || (self.cc.num_qubits < 128 && x >> self.cc.num_qubits != 0)
                {
                    return C0;
                }
                let mut a = phase;
                for ((g, _), sv) in self.cc.parts.iter().zip(&states) {
                    let y = g
                        .iter()
                        .enumerate()
                        .fold(0usize, |y, (j, &q)| y | (((x >> q) & 1) as usize) << j);
                    a *= sv.amplitudes()[y];
                }
                a
            })
            .collect())
    }
}
