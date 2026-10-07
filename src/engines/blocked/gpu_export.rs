//! Export of a compiled f32 plan to a flat, GPU-friendly form.
//!
//! The GPU backend of the packed chain sweep (`chain_packed_gpu`) must
//! reproduce the CPU kernels bit for bit. Rather than re-deriving the
//! plan, it runs the CPU's own compiled plan: the cache-blocked stages
//! ([`Prepared`]) with their typed ops ([`LOp`]), exported here as
//! [`GpuSubStage`]s. Every per-element formula of the FMA kernel tier
//! (`F = true`) is a short fixed sequence of f32 `fma`/`mul` operations
//! that a shader can repeat exactly; the only part computed in f64 on the
//! CPU, the tables of the diagonal groups, is precomputed here for every
//! combination of the group's outer-conditioned terms ([`GpuDiagGroup`]),
//! so the shader only does f32 arithmetic.
//!
//! The reference semantics are those of the FMA tier without the AVX-512
//! preparation (no single-pass diagonals, no pairs on the lowest buffer
//! bits) and without dense fusion or L1 tiling: export fails on anything
//! else. [`run_sub_stage_ref`] is a scalar CPU interpreter of the exported
//! form, used by the tests to check the export against the executor.

use super::{
    product_table, BlockConfig, CompiledKOps, DiagGroup, LOp, Prepared, PreparedStage, UKind, C1,
    LO_BITS,
};
use num_complex::{Complex32, Complex64};

/// Most outer-conditioned terms one diagonal group may have (its tables are
/// precomputed for each of the `2^k` activity patterns).
pub const MAX_COND_TERMS: usize = 12;

/// Kind of a 2x2 gate: 0 = X (a swap), 1 = real matrix, 2 = complex.
pub type GpuKind = u8;

/// One op of a sub-stage, on buffer bits of the sub-stage's `2^l` buffer.
#[derive(Clone, Debug)]
pub enum GpuOp {
    /// `m` on buffer bit `t` where the buffer bits `cin` are all 1, applied
    /// only in sub-blocks whose register index has all bits of `cout` set.
    /// `m` = re of m00, m01, m10, m11, then im of the same.
    U1 {
        /// Target bit.
        t: u32,
        /// Coefficients.
        m: [f32; 8],
        /// Kernel kind.
        kind: GpuKind,
        /// Inner control mask (buffer bits).
        cin: u32,
        /// Outer control mask (register bits).
        cout: u64,
    },
    /// Exchange of buffer bits `a < b`.
    Swap {
        /// Lower bit.
        a: u32,
        /// Higher bit.
        b: u32,
    },
    /// `m1` on bit `t1`, `m2` on bit `t2` (`t1 < t2`), then a CNOT
    /// (`cx` = 1: control `t1`; 2: control `t2`; 0: none). `real*`: the
    /// real-matrix formula (also used for X here).
    Pair {
        /// First target.
        t1: u32,
        /// First matrix.
        m1: [f32; 8],
        /// Real formula for `m1`.
        real1: bool,
        /// Second target.
        t2: u32,
        /// Second matrix.
        m2: [f32; 8],
        /// Real formula for `m2`.
        real2: bool,
        /// CNOT folded in.
        cx: u8,
    },
    /// A diagonal block: its groups applied in order.
    Diag(Vec<GpuDiagGroup>),
}

/// Factor tables of one diagonal group for one activity pattern.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuDiagTables {
    /// Low table (buffer bits `0..lb`), re / im.
    pub lor: Vec<f32>,
    /// See `lor`.
    pub loi: Vec<f32>,
    /// Row factors (buffer bits `lb..l`), re / im.
    pub hr: Vec<f32>,
    /// See `hr`.
    pub hi: Vec<f32>,
}

/// A diagonal group: factor `lo[x] * hi[h]` (one f32 complex product, then
/// the amplitude times it) on the buffer indices `j = h << lb | x` with
/// `(j & cmask) == cpat`.
#[derive(Clone, Debug)]
pub struct GpuDiagGroup {
    /// Condition mask on buffer bits.
    pub cmask: u32,
    /// Condition pattern.
    pub cpat: u32,
    /// Low-table bits.
    pub lb: u32,
    /// `(omask, opat)` of the outer-conditioned terms, in term order: bit
    /// `k` of the activity pattern is `(base & omask) == opat`.
    pub conds: Vec<(u64, u64)>,
    /// Tables per activity pattern; `None`: no term active (group skipped).
    pub variants: Vec<Option<GpuDiagTables>>,
}

/// One cache-blocked stage: buffer bit `j` is register bit `j`-th set bit
/// of `inner_mask`; the remaining register bits select the sub-block.
#[derive(Clone, Debug)]
pub struct GpuSubStage {
    /// Buffer bits.
    pub l: usize,
    /// Register bits of the buffer.
    pub inner_mask: usize,
    /// Ops in order.
    pub ops: Vec<GpuOp>,
}

/// Why a plan cannot be exported.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExportError {
    /// AVX-512-only, dense or tiled forms.
    Unsupported(&'static str),
    /// A diagonal group with too many outer-conditioned terms.
    TooManyConds(usize),
}

impl std::fmt::Display for ExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExportError::Unsupported(w) => write!(f, "GPU export: unsupported {w}"),
            ExportError::TooManyConds(k) => write!(
                f,
                "GPU export: a diagonal group has {k} outer-conditioned terms (max {MAX_COND_TERMS})"
            ),
        }
    }
}

fn kind(k: UKind) -> GpuKind {
    match k {
        UKind::X => 0,
        UKind::Real => 1,
        UKind::Complex => 2,
    }
}

/// Tables of group `g` on a buffer of `2^l` for the activity pattern
/// `pat` of its conditioned terms (exactly as `apply_diag_group`).
fn group_tables(g: &DiagGroup, l: usize, pat: usize) -> Option<GpuDiagTables> {
    let mut e = [[C1; 2]; 64];
    let mut s = C1;
    let mut any = false;
    let mut k = 0;
    for t in &g.terms {
        let active = if t.omask == 0 {
            true
        } else {
            let a = pat >> k & 1 == 1;
            k += 1;
            a
        };
        if !active {
            continue;
        }
        any = true;
        match t.bit {
            None => s *= t.f,
            Some((j, v)) => e[j][v as usize] *= t.f,
        }
    }
    if !any {
        return None;
    }
    let lb = l.min(LO_BITS);
    let (mut lo, mut hi) = (Vec::new(), Vec::new());
    product_table(&mut lo, &e[..lb], C1);
    product_table(&mut hi, &e[lb..l], s);
    let c32 = |v: &[Complex64], im: bool| -> Vec<f32> {
        v.iter()
            .map(|z| if im { z.im as f32 } else { z.re as f32 })
            .collect()
    };
    Some(GpuDiagTables {
        lor: c32(&lo, false),
        loi: c32(&lo, true),
        hr: c32(&hi, false),
        hi: c32(&hi, true),
    })
}

fn export_prepared(p: &Prepared<f32>) -> Result<GpuSubStage, ExportError> {
    if p.tile.is_some() {
        return Err(ExportError::Unsupported("L1 tiling"));
    }
    let mut ops = Vec::with_capacity(p.ops.len());
    for op in &p.ops {
        ops.push(match op {
            LOp::U1 {
                t,
                m,
                kind: k,
                cin,
                cout,
            } => GpuOp::U1 {
                t: *t as u32,
                m: *m,
                kind: kind(*k),
                cin: *cin as u32,
                cout: *cout as u64,
            },
            LOp::Swap { a, b } => GpuOp::Swap {
                a: *a as u32,
                b: *b as u32,
            },
            LOp::Pair {
                t1,
                m1,
                k1,
                t2,
                m2,
                k2,
                cx,
            } => GpuOp::Pair {
                t1: *t1 as u32,
                m1: *m1,
                real1: *k1 != UKind::Complex,
                t2: *t2 as u32,
                m2: *m2,
                real2: *k2 != UKind::Complex,
                cx: *cx,
            },
            LOp::Diag(d) => {
                let mut gs = Vec::with_capacity(d.groups.len());
                for g in &d.groups {
                    let conds: Vec<(u64, u64)> = g
                        .terms
                        .iter()
                        .filter(|t| t.omask != 0)
                        .map(|t| (t.omask as u64, t.opat as u64))
                        .collect();
                    if conds.len() > MAX_COND_TERMS {
                        return Err(ExportError::TooManyConds(conds.len()));
                    }
                    let variants = (0..1usize << conds.len())
                        .map(|pat| group_tables(g, p.l, pat))
                        .collect();
                    gs.push(GpuDiagGroup {
                        cmask: g.cmask as u32,
                        cpat: g.cpat as u32,
                        lb: p.l.min(LO_BITS) as u32,
                        conds,
                        variants,
                    });
                }
                GpuOp::Diag(gs)
            }
            LOp::DiagPass(_) => return Err(ExportError::Unsupported("single-pass diagonal")),
            LOp::Dense { .. } => return Err(ExportError::Unsupported("dense fusion")),
        });
    }
    Ok(GpuSubStage {
        l: p.l,
        inner_mask: p.inner_mask,
        ops,
    })
}

/// Whether `cfg` gives plans the GPU export reproduces: FMA tier (no
/// AVX-512 kernels or preparation), no dense fusion, no L1 tiling, SIMD on.
pub fn cfg_exportable(cfg: &BlockConfig) -> bool {
    cfg.simd
        && !(cfg.avx512 && super::avx512_available())
        && cfg.dense_fusion < 2
        && cfg.tile_bits(30, 8) == 0
        && super::simd_available()
}

/// Exports a compiled plan.
pub fn export_compiled(c: &CompiledKOps<f32>) -> Result<Vec<GpuSubStage>, ExportError> {
    if c.isa != super::Isa::Fma {
        return Err(ExportError::Unsupported("kernel tier other than FMA"));
    }
    c.stages.iter().map(export_prepared).collect()
}

/// Exports one prepared stage (run with `run_prepared_on_block`, whose
/// kernel tier is the FMA one only when AVX-512 is unavailable or disabled
/// with `QSIM_NO_AVX512`).
pub fn export_prepared_stage(p: &PreparedStage<f32>) -> Result<GpuSubStage, ExportError> {
    if super::avx512_available() || !super::simd_available() {
        return Err(ExportError::Unsupported("kernel tier other than FMA"));
    }
    export_prepared(&p.0)
}

// ----- scalar reference interpreter --------------------------------------

#[inline]
fn cmul2(m: [f32; 4], xr: f32, xi: f32, yr: f32, yi: f32) -> (f32, f32) {
    let [m0r, m0i, m1r, m1i] = m;
    let re = m0r.mul_add(xr, (-m0i).mul_add(xi, m1r.mul_add(yr, -(m1i * yi))));
    let im = m0r.mul_add(xi, m0i.mul_add(xr, m1r.mul_add(yi, m1i * yr)));
    (re, im)
}

/// `(x, y) <- (m0 x + m1 y, m2 x + m3 y)` with the real or complex formula.
#[inline]
pub fn mat_apply_ref(
    m: &[f32; 8],
    real: bool,
    x: Complex32,
    y: Complex32,
) -> (Complex32, Complex32) {
    let [m0r, m1r, m2r, m3r, m0i, m1i, m2i, m3i] = *m;
    if real {
        (
            Complex32::new(m0r.mul_add(x.re, m1r * y.re), m0r.mul_add(x.im, m1r * y.im)),
            Complex32::new(m2r.mul_add(x.re, m3r * y.re), m2r.mul_add(x.im, m3r * y.im)),
        )
    } else {
        let (a, b) = cmul2([m0r, m0i, m1r, m1i], x.re, x.im, y.re, y.im);
        let (c, d) = cmul2([m2r, m2i, m3r, m3i], x.re, x.im, y.re, y.im);
        (Complex32::new(a, b), Complex32::new(c, d))
    }
}

/// Activity pattern of a group's conditioned terms in the sub-block whose
/// register index (outer bits) is `base`.
pub fn activity(conds: &[(u64, u64)], base: u64) -> usize {
    conds
        .iter()
        .enumerate()
        .map(|(k, &(m, p))| (((base & m) == p) as usize) << k)
        .sum()
}

/// Runs the ops of `s` on one gathered buffer (`buf.len() == 2^s.l`) of
/// the sub-block with register index `base` (its outer bits).
pub fn run_sub_stage_ref(s: &GpuSubStage, buf: &mut [Complex32], base: u64) {
    let n = buf.len();
    assert_eq!(n, 1 << s.l);
    for op in &s.ops {
        match op {
            GpuOp::U1 {
                t,
                m,
                kind,
                cin,
                cout,
            } => {
                if base & cout != *cout {
                    continue;
                }
                let st = 1usize << t;
                for i in 0..n {
                    if i & st != 0 || i & *cin as usize != *cin as usize {
                        continue;
                    }
                    let (x, y) = (buf[i], buf[i | st]);
                    let (a, b) = if *kind == 0 {
                        (y, x)
                    } else {
                        mat_apply_ref(m, *kind == 1, x, y)
                    };
                    buf[i] = a;
                    buf[i | st] = b;
                }
            }
            GpuOp::Swap { a, b } => {
                let (sa, sb) = (1usize << a, 1usize << b);
                for i in 0..n {
                    if i & sa != 0 && i & sb == 0 {
                        buf.swap(i, i ^ sa ^ sb);
                    }
                }
            }
            GpuOp::Pair {
                t1,
                m1,
                real1,
                t2,
                m2,
                real2,
                cx,
            } => {
                let (s1, s2) = (1usize << t1, 1usize << t2);
                for i in 0..n {
                    if i & (s1 | s2) != 0 {
                        continue;
                    }
                    let a = [buf[i], buf[i | s1], buf[i | s2], buf[i | s1 | s2]];
                    let (b0, b1) = mat_apply_ref(m1, *real1, a[0], a[1]);
                    let (b2, b3) = mat_apply_ref(m1, *real1, a[2], a[3]);
                    let (c0, c2) = mat_apply_ref(m2, *real2, b0, b2);
                    let (mut c1, mut c3) = mat_apply_ref(m2, *real2, b1, b3);
                    let mut c2 = c2;
                    if *cx == 1 {
                        std::mem::swap(&mut c1, &mut c3);
                    } else if *cx == 2 {
                        std::mem::swap(&mut c2, &mut c3);
                    }
                    buf[i] = c0;
                    buf[i | s1] = c1;
                    buf[i | s2] = c2;
                    buf[i | s1 | s2] = c3;
                }
            }
            GpuOp::Diag(gs) => {
                for g in gs {
                    let Some(tb) = &g.variants[activity(&g.conds, base)] else {
                        continue;
                    };
                    let lb = g.lb as usize;
                    for (j, z) in buf.iter_mut().enumerate() {
                        if j as u32 & g.cmask != g.cpat {
                            continue;
                        }
                        let (x, h) = (j & ((1 << lb) - 1), j >> lb);
                        let (lr, li, hr, hi) = (tb.lor[x], tb.loi[x], tb.hr[h], tb.hi[h]);
                        let fr = lr.mul_add(hr, -(li * hi));
                        let fi = lr.mul_add(hi, li * hr);
                        let (xr, xi) = (z.re, z.im);
                        *z = Complex32::new(xr.mul_add(fr, -(xi * fi)), xr.mul_add(fi, xi * fr));
                    }
                }
            }
        }
    }
}
