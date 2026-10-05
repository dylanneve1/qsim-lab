//! Gidney–Ekerå 2019 (arXiv:1905.09749) techniques in the exact gate-level
//! Shor simulation (exp/ge-shor, `research/shor/ge-shor.md`).
//!
//! Three changes to *what* the circuit computes, each simulated gate by
//! gate on every branch:
//!
//! 1. **Exponent windowing.** Instead of one controlled multiplication per
//!    exponent bit, `w_e` exponent qubits are kept live at once and ONE
//!    multiplication `x → g^e·x mod N` (`e` = the window's value, `g` = the
//!    window's lowest power of the base) is done with lookups addressed by the
//!    `w_e` exponent qubits *and* `w_m` multiplicand bits:
//!    `T[e, v] = g^e·v·2^{k w_m} mod N`. The lookups are uncontrolled (the
//!    exponent is in the address), so the controlled swap becomes a plain
//!    swap. The semiclassical QFT is unchanged in distribution: the window's
//!    `w_e` qubits are measured one after another, highest power first, each
//!    with the Griffiths–Niu phase correction of all bits measured before it
//!    (including the ones of the same window). [`GeState`] evaluates the block
//!    on all `2^{w_e}·|supp ψ|` branches `(e, x)` and then runs the `w_e`
//!    measurements exactly.
//! 2. **Ekerå–Håstad short exponent** ([`eh_run`], [`eh_postprocess`]):
//!    factoring an RSA integer `N = pq` as the short discrete logarithm
//!    `d = (p + q − 2)/2` of `y = g^{(N−1)/2}` to the base `g`. Two exponent
//!    registers of `2m` and `m` bits (`m` ≈ `n/2`, `s = 1`): `3m ≈ 1.5n`
//!    controlled multiplications instead of `2n`; classical post-processing by
//!    2-D lattice reduction and enumeration.
//! 3. **Coset representation** (Zalka 2006; GE19 §2.4) ([`GeOpts::coset`]):
//!    registers of `n + c` qubits holding `Σ_j |x + jN⟩`; the modular adder
//!    becomes a plain `(n + c)`-bit addition. This is *approximate*: the
//!    simulation tracks every coset branch, so it measures the exact effect
//!    of the approximation on the output distribution.

use crate::algorithms::{gcd, pow_mod};
use crate::engines::statevector::Real;
use crate::gate::Gate;
use crate::shor::mbu::{
    add_g, inverse, modadd_ops, resolve, LOp, LookupSpec, MbuCounts, MbuLayout, MbuOp, MbuOpts,
    Outcomes, NO_CTRL,
};
use crate::shor::sliced::{transpose64, SlicedProgram};
use crate::shor::window::WindowLayout;
use crate::shor::{mod_inverse, mul_mod};
use num_complex::{Complex, Complex64};
use num_traits::Zero;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::rc::Rc;

/// Parameters of the Gidney–Ekerå style oracle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeOpts {
    /// Exponent window `w_e` (1 = one recycled control per round).
    pub we: usize,
    /// Multiplicand window `w_m`.
    pub wm: usize,
    /// Measurement-based uncomputation options (as in `shor_mbu`).
    pub mbu: MbuOpts,
    /// Coset padding `c` (0 = exact modular arithmetic). With `c > 0` the
    /// work and accumulator registers have `n + c` qubits and every modular
    /// addition is a plain `(n + c)`-bit addition (Gidney adders required).
    pub coset: usize,
}

/// Qubit layout.
#[derive(Clone, Debug)]
pub struct GeLayout {
    /// Modulus bits `n`.
    pub n: usize,
    /// Register width: `n` (exact) or `n + c` (coset).
    pub nr: usize,
    pub we: usize,
    pub wm: usize,
    /// Exponent window qubits (LSB first).
    pub e: Vec<usize>,
    /// Work register (`nr` qubits).
    pub x: Vec<usize>,
    /// Accumulator: `n + 1` qubits (exact) / `nr` qubits (coset).
    pub b: Vec<usize>,
    /// Lookup output: `n` (exact) / `nr − 1` (coset; the top `c − 1` stay 0).
    pub l: Vec<usize>,
    /// AND ancillas of the lookups (`w_e + w_m`).
    pub and: Vec<usize>,
    /// Carry ancillas of the Gidney adders.
    pub cy: Vec<usize>,
    /// Exact arithmetic: the windowed/MBU layout the modular adders use.
    pub mbu: Option<MbuLayout>,
    pub nq: usize,
}

impl GeLayout {
    pub fn new(n: usize, o: &GeOpts) -> Self {
        assert!(n >= 2 && (1..=6).contains(&o.we) && o.wm >= 1);
        let wa = o.we + o.wm;
        if o.coset == 0 {
            let win = WindowLayout {
                n,
                w: wa,
                ctrl: 0,
                x: (1..=n).collect(),
                b: (n + 1..=2 * n + 1).collect(),
                l: (2 * n + 2..=3 * n + 1).collect(),
                k: (3 * n + 2..=4 * n + 1).collect(),
                c0: 4 * n + 2,
                t: 4 * n + 3,
                and: (4 * n + 4..4 * n + 4 + wa).collect(),
            };
            let base = win.num_qubits();
            let cy: Vec<usize> = if o.mbu.adders {
                (base..base + n - 1).collect()
            } else {
                Vec::new()
            };
            let mut nq = base + cy.len();
            let mut e = vec![win.ctrl];
            for _ in 1..o.we {
                e.push(nq);
                nq += 1;
            }
            Self {
                n,
                nr: n,
                we: o.we,
                wm: o.wm,
                e,
                x: win.x.clone(),
                b: win.b.clone(),
                l: win.l.clone(),
                and: win.and.clone(),
                cy: cy.clone(),
                mbu: Some(MbuLayout { win, cy }),
                nq,
            }
        } else {
            assert!(o.mbu.adders, "coset arithmetic uses Gidney adders");
            let nr = n + o.coset;
            let mut q = 0;
            let mut take = |k: usize| {
                let v: Vec<usize> = (q..q + k).collect();
                q += k;
                v
            };
            let x = take(nr);
            let b = take(nr);
            let l = take(nr - 1);
            let and = take(wa);
            let cy = take(nr - 2);
            let e = take(o.we);
            Self {
                n,
                nr,
                we: o.we,
                wm: o.wm,
                e,
                x,
                b,
                l,
                and,
                cy,
                mbu: None,
                nq: q,
            }
        }
    }

    /// The qubits stored between windows (the work register, plus the
    /// accumulator in coset mode, whose coset index is not an ancilla).
    pub fn stored(&self) -> Vec<usize> {
        if self.mbu.is_some() {
            self.x.clone()
        } else {
            self.x.iter().chain(&self.b).copied().collect()
        }
    }
}

/// Windowed multiply-add `|e⟩|x⟩|b⟩ → |e⟩|x⟩|b + g^e·x⟩` (exact: `mod N`
/// with `b = 0` before; coset: plain `(n + c)`-bit additions of the
/// looked-up residues).
pub fn emult(lay: &GeLayout, g: u64, n_mod: u64, o: &GeOpts, modadd: &[LOp]) -> Vec<LOp> {
    let n = lay.n;
    let nr = lay.nr;
    let ge: Vec<u64> = {
        let mut v = vec![1 % n_mod];
        for _ in 1..1usize << lay.we {
            v.push(mul_mod(*v.last().unwrap(), g % n_mod, n_mod));
        }
        v
    };
    let mut ops = Vec::new();
    let mut start = 0;
    let mut pw = 1 % n_mod; // 2^start mod N
    while start < nr {
        let w = lay.wm.min(nr - start);
        let mut table = vec![0u64; 1 << (w + lay.we)];
        for (e, &gee) in ge.iter().enumerate() {
            let ge_pw = mul_mod(gee, pw, n_mod);
            for v in 0..1u64 << w {
                table[(v as usize) | (e << w)] = mul_mod(v % n_mod, ge_pw, n_mod);
            }
        }
        let mut addr = lay.x[start..start + w].to_vec();
        addr.extend(&lay.e);
        let direct = lay.mbu.is_some() && start == 0;
        let out = if direct {
            lay.b[..n].to_vec()
        } else {
            lay.l[..n].to_vec()
        };
        let spec = Rc::new(LookupSpec {
            ctrl: NO_CTRL,
            addr,
            and: lay.and.clone(),
            out,
            table,
            mbu_and: o.mbu.lookup_and,
            meas_unlookup: o.mbu.unlookup,
        });
        ops.push(LOp::Lookup(spec.clone()));
        if !direct {
            if lay.mbu.is_some() {
                ops.extend(modadd.iter().cloned());
            } else {
                add_g(&mut ops, &lay.l, &lay.b, &lay.cy[..nr - 2]);
            }
            ops.push(LOp::Unlookup(spec));
        }
        for _ in 0..w {
            pw = mul_mod(pw, 2, n_mod);
        }
        start += w;
    }
    ops
}

/// The window block `|e⟩|x⟩ → |e⟩|g^e x mod N⟩` as logical ops:
/// multiply-add by `g^e`, swap, exact inverse of the multiply-add by
/// `g^{−e}`.
pub fn window_ops(lay: &GeLayout, g: u64, n_mod: u64, o: &GeOpts) -> Vec<LOp> {
    let modadd = match &lay.mbu {
        Some(m) => modadd_ops(m, n_mod, &o.mbu),
        None => Vec::new(),
    };
    let ginv = mod_inverse(g, n_mod);
    let mut ops = emult(lay, g, n_mod, o, &modadd);
    for i in 0..lay.nr {
        let (x, b) = (lay.x[i], lay.b[i]);
        ops.push(LOp::G(Gate::Cnot(x, b)));
        ops.push(LOp::G(Gate::Cnot(b, x)));
        ops.push(LOp::G(Gate::Cnot(x, b)));
    }
    ops.extend(inverse(&emult(lay, ginv, n_mod, o, &modadd)));
    ops
}

/// Resolved window block for one outcome stream.
pub fn window_block(
    lay: &GeLayout,
    g: u64,
    n_mod: u64,
    o: &GeOpts,
    outc: &mut Outcomes,
) -> Vec<MbuOp> {
    let ops = window_ops(lay, g, n_mod, o);
    let mut out = Vec::with_capacity(ops.len() * 2);
    resolve(&ops, &mut || outc.next_bit(), &mut out);
    out
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// A sorted list of `(stored key, amplitude)` branches.
pub type Arr<T> = Vec<(u64, Complex<T>)>;

fn lanes() -> usize {
    std::env::var("QSIM_SLICE_LANES")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|l| [4usize, 8, 16, 32].contains(l))
        .unwrap_or(16)
}

/// Evaluates the resolved window program on `|e⟩|key⟩|0…⟩` for every input
/// branch, gate by gate on bit slices; returns `(U_e key, amplitude)`.
/// Panics unless the exponent qubits are unchanged, every ancilla is 0 and
/// every branch carries the program's global sign.
pub fn eval_e<T: Real>(
    prog: &SlicedProgram,
    eq: &[usize],
    xq: &[usize],
    e: u64,
    inp: &[(u64, Complex<T>)],
) -> Arr<T> {
    let mut out = inp.to_vec();
    let _ = eval_e_inplace(prog, eq, xq, e, &mut out);
    out
}

/// [`eval_e`] in place: every key of `buf` is replaced by its image.
pub fn eval_e_inplace<T: Real>(
    prog: &SlicedProgram,
    eq: &[usize],
    xq: &[usize],
    e: u64,
    buf: &mut [(u64, Complex<T>)],
) -> bool {
    match lanes() {
        4 => eval_e_l::<4, T>(prog, eq, xq, e, buf),
        8 => eval_e_l::<8, T>(prog, eq, xq, e, buf),
        32 => eval_e_l::<32, T>(prog, eq, xq, e, buf),
        _ => eval_e_l::<16, T>(prog, eq, xq, e, buf),
    }
}

fn eval_e_l<const L: usize, T: Real>(
    prog: &SlicedProgram,
    eq: &[usize],
    xq: &[usize],
    e: u64,
    out: &mut [(u64, Complex<T>)],
) -> bool {
    let nq = prog.nq;
    assert!(xq.len() <= 64);
    let mut is_reg = vec![false; nq];
    for &q in eq.iter().chain(xq) {
        is_reg[q] = true;
    }
    let anc: Vec<usize> = (0..nq).filter(|&q| !is_reg[q]).collect();
    out.par_chunks_mut(64 * L)
        .map_init(
            || vec![[0u64; L]; nq + 2],
            |w, chunk| {
                for wq in w.iter_mut() {
                    *wq = [0; L];
                }
                w[nq] = [u64::MAX; L];
                let mut valid = [0u64; L];
                for (l, v) in valid.iter_mut().enumerate() {
                    let lo = l * 64;
                    if chunk.len() > lo {
                        let k = (chunk.len() - lo).min(64);
                        *v = if k == 64 { u64::MAX } else { (1u64 << k) - 1 };
                    }
                }
                for (j, &q) in eq.iter().enumerate() {
                    if (e >> j) & 1 == 1 {
                        w[q] = valid;
                    }
                }
                let mut blk = [0u64; 64];
                for (l, c) in chunk.chunks(64).enumerate() {
                    for (b, x) in blk.iter_mut().zip(c) {
                        *b = x.0;
                    }
                    blk[c.len()..].fill(0);
                    transpose64(&mut blk);
                    for (j, &q) in xq.iter().enumerate() {
                        w[q][l] = blk[j];
                    }
                }
                prog.eval(w);
                let mut same = true;
                for l in 0..L {
                    for (j, &q) in eq.iter().enumerate() {
                        let want = if (e >> j) & 1 == 1 { valid[l] } else { 0 };
                        assert_eq!(
                            w[q][l] & valid[l],
                            want,
                            "exponent qubit changed by the window block"
                        );
                    }
                    let mut dirty = 0u64;
                    for &q in &anc {
                        dirty |= w[q][l];
                    }
                    assert_eq!(dirty & valid[l], 0, "ancillas did not return to 0");
                    if prog.signed {
                        let want = if prog.global_neg { valid[l] } else { 0 };
                        assert_eq!(
                            w[nq + 1][l] & valid[l],
                            want,
                            "measurement-based uncomputation left a relative sign"
                        );
                    }
                }
                for (l, oc) in chunk.chunks_mut(64).enumerate() {
                    blk.fill(0);
                    for (j, &q) in xq.iter().enumerate() {
                        blk[j] = w[q][l];
                    }
                    transpose64(&mut blk);
                    for (o, &k) in oc.iter_mut().zip(blk.iter()) {
                        same &= o.0 == k;
                        o.0 = k;
                    }
                }
                same
            },
        )
        .reduce(|| true, |a, b| a && b)
}

fn cvt<T: Real>(z: Complex64) -> Complex<T> {
    Complex::new(T::from_f64(z.re), T::from_f64(z.im))
}
fn c64<T: Real>(z: Complex<T>) -> Complex64 {
    Complex64::new(z.re.to_f64(), z.im.to_f64())
}

/// The branches of one window after the block, and the semiclassical
/// measurement of its exponent qubits in progress.
///
/// `ent` holds every evaluated branch `(e, V^e x)` as one array sorted by
/// the packed key `(V^e x) << w | e`, so the `2^w` values of one
/// work-register key are adjacent. The measurement is applied **lazily**:
/// for every key the `2^w` values are combined level by level exactly as
/// `C' = (C_lo ± e^{iφ} C_hi)/√(2p)` would combine materialised arrays (the
/// same floating-point operations in the same order, rounded to `T` after
/// every level), so nothing but the evaluated branches and, at the end,
/// the new state is stored.
#[derive(Clone, Debug)]
pub struct WindowArrays<T: Real> {
    /// Window bits `w`.
    pub w: usize,
    pub ent: std::sync::Arc<Vec<(u64, Complex<T>)>>,
    /// `ent[..split]` and `ent[split..]` are each sorted (`split > 0`: the
    /// `e = 0` branches are `ψ` itself, already sorted, and only the others
    /// were sorted).
    pub split: usize,
    /// Measured levels, highest exponent bit first: `(±e^{iφ}, 1/√(2p))`.
    pub levels: Vec<(Complex64, f64)>,
}

impl<T: Real> WindowArrays<T> {
    /// Number of exponent qubits still unmeasured.
    pub fn bits_left(&self) -> usize {
        self.w - self.levels.len()
    }
    /// Total number of evaluated `(e, x)` branches.
    pub fn branches(&self) -> usize {
        self.ent.len()
    }
    /// Aligned chunks `(a_lo, a_hi, b_lo, b_hi)` of the two sorted runs
    /// `A = ent[..split]`, `B = ent[split..]`, cut at work-register key
    /// boundaries.
    fn chunks(&self) -> Vec<(usize, usize, usize, usize)> {
        let ent = &self.ent;
        let (sa, n) = (self.split, ent.len());
        let w = self.w;
        let p = (rayon::current_num_threads() * 8).max(1);
        let nb = n - sa;
        let mut cuts: Vec<u64> = (1..p)
            .filter_map(|j| {
                let i = sa + nb * j / p;
                (i > sa && i < n).then(|| ent[i].0 >> w)
            })
            .collect();
        cuts.dedup();
        let mut out = Vec::with_capacity(cuts.len() + 1);
        let (mut a0, mut b0) = (0, sa);
        for &k in &cuts {
            let a1 = ent[..sa].partition_point(|v| v.0 >> w < k);
            let b1 = sa + ent[sa..].partition_point(|v| v.0 >> w < k);
            if a1 > a0 || b1 > b0 {
                out.push((a0, a1, b0, b1));
            }
            (a0, b0) = (a1, b1);
        }
        out.push((a0, sa, b0, n));
        out
    }
    /// Visits the entries of one chunk in packed-key order (a 2-way merge
    /// of the two runs).
    #[inline]
    fn stream(
        &self,
        (a0, a1, b0, b1): (usize, usize, usize, usize),
        mut f: impl FnMut(&(u64, Complex<T>)),
    ) {
        let ent: &[(u64, Complex<T>)] = &self.ent;
        let (mut i, mut j) = (a0, b0);
        while i < a1 && j < b1 {
            if ent[i].0 < ent[j].0 {
                f(&ent[i]);
                i += 1;
            } else {
                f(&ent[j]);
                j += 1;
            }
        }
        for v in &ent[i..a1] {
            f(v);
        }
        for v in &ent[j..b1] {
            f(v);
        }
    }
    /// Calls `f(key, vals)` for every work-register key of a chunk,
    /// `vals[e]` = the amplitude of `(e, key)` (0 if absent).
    #[inline]
    fn groups(&self, ch: (usize, usize, usize, usize), mut f: impl FnMut(u64, &mut [Complex64])) {
        let w = self.w;
        let mask = (1u64 << w) - 1;
        let mut vals = vec![Complex64::zero(); 1 << w];
        let mut cur: Option<u64> = None;
        self.stream(ch, |v| {
            let key = v.0 >> w;
            if cur != Some(key) {
                if let Some(k) = cur {
                    f(k, &mut vals);
                    vals.iter_mut().for_each(|v| *v = Complex64::zero());
                }
                cur = Some(key);
            }
            vals[(v.0 & mask) as usize] = c64(v.1);
        });
        if let Some(k) = cur {
            f(k, &mut vals);
        }
    }
    /// Applies the initial `2^{−w/2}` and the measured levels to the values
    /// at one key; returns how many entries of `vals` are live.
    #[inline]
    fn reduce(&self, vals: &mut [Complex64]) -> usize {
        const NORMS: [f64; 7] = [
            1.0,
            std::f64::consts::FRAC_1_SQRT_2,
            0.5,
            0.353_553_390_593_273_8,
            0.25,
            0.176_776_695_296_636_9,
            0.125,
        ];
        let norm = NORMS[self.w];
        for v in vals.iter_mut() {
            *v = c64(cvt::<T>(*v * norm));
        }
        let mut len = vals.len();
        for &(ph, k) in &self.levels {
            let h = len / 2;
            for e in 0..h {
                vals[e] = c64(cvt::<T>((vals[e] + ph * vals[e + h]) * k));
            }
            len = h;
        }
        len
    }
    /// Outcome probabilities `(P(0), P(1))` of measuring the highest
    /// remaining exponent qubit after the phase `φ` on its `|1⟩` and `H`.
    pub fn probs(&self, phi: f64) -> (f64, f64) {
        assert!(self.bits_left() > 0);
        let ph = Complex64::from_polar(1.0, phi);
        self.chunks()
            .into_par_iter()
            .map(|ch| {
                let (mut p0, mut p1) = (0.0, 0.0);
                self.groups(ch, |_, vals| {
                    let len = self.reduce(vals);
                    let h = len / 2;
                    for e in 0..h {
                        p0 += (vals[e] + ph * vals[e + h]).norm_sqr() / 2.0;
                        p1 += (vals[e] - ph * vals[e + h]).norm_sqr() / 2.0;
                    }
                });
                (p0, p1)
            })
            .reduce(|| (0.0, 0.0), |u, v| (u.0 + v.0, u.1 + v.1))
    }
    /// Joint distribution of all remaining exponent bits in one pass:
    /// `out[b]` = P(next bits = `b`), bit `j` of `b` = the `j`-th remaining
    /// measurement (highest remaining power first), where measurement `j`
    /// gets the phase `phi(j, b mod 2^j)` (Griffiths–Niu: it depends on the
    /// outcomes of the earlier measurements of the same window).
    pub fn joint_probs(&self, phi: &(dyn Fn(usize, u64) -> f64 + Sync)) -> Vec<f64> {
        let bl = self.bits_left();
        // phases per (level j, prefix): 2^bl − 1 of them
        let mut phs = Vec::new();
        for j in 0..bl {
            for pre in 0..1u64 << j {
                phs.push(Complex64::from_polar(1.0, phi(j, pre)));
            }
        }
        let r = std::f64::consts::FRAC_1_SQRT_2;
        let nb = 1usize << bl;
        let ne = 1usize << self.w;
        // The map from the 2^w values at a key to the 2^bl outcome
        // amplitudes is linear: tabulate it (column e = image of δ_e under
        // the measured levels and the butterfly over the remaining bits).
        let mut mat = vec![Complex64::zero(); nb * ne];
        {
            let mut cur = vec![Complex64::zero(); nb];
            let mut nxt = vec![Complex64::zero(); nb];
            for e in 0..ne {
                let mut vals = vec![Complex64::zero(); ne];
                vals[e] = Complex64::new(1.0, 0.0);
                // same reduction as `reduce`, without rounding to T
                let mut len = ne;
                for v in vals.iter_mut() {
                    *v *= (1.0 / ne as f64).sqrt();
                }
                for &(ph, k) in &self.levels {
                    let h = len / 2;
                    for i in 0..h {
                        vals[i] = (vals[i] + ph * vals[i + h]) * k;
                    }
                    len = h;
                }
                cur.copy_from_slice(&vals[..nb]);
                for j in 0..bl {
                    let lj = nb >> j;
                    let h = lj / 2;
                    for pre in 0..1usize << j {
                        let ph = phs[(1usize << j) - 1 + pre];
                        for b in 0..2usize {
                            let s = if b == 1 { -ph } else { ph };
                            let o = (pre | (b << j)) * h;
                            for i in 0..h {
                                nxt[o + i] = (cur[pre * lj + i] + s * cur[pre * lj + i + h]) * r;
                            }
                        }
                    }
                    std::mem::swap(&mut cur, &mut nxt);
                }
                for b in 0..nb {
                    mat[b * ne + e] = cur[b];
                }
            }
        }
        let w = self.w;
        let mask = (1u64 << w) - 1;
        self.chunks()
            .into_par_iter()
            .map(|ch| {
                let mut acc = vec![0.0f64; nb];
                let mut amp = vec![Complex64::zero(); nb];
                let mut key = u64::MAX;
                self.stream(ch, |v| {
                    let k = v.0 >> w;
                    if k != key {
                        for (a, z) in acc.iter_mut().zip(amp.iter_mut()) {
                            *a += z.norm_sqr();
                            *z = Complex64::zero();
                        }
                        key = k;
                    }
                    let e = (v.0 & mask) as usize;
                    let x = c64(v.1);
                    for (b, z) in amp.iter_mut().enumerate() {
                        *z += mat[b * ne + e] * x;
                    }
                });
                for (a, z) in acc.iter_mut().zip(amp.iter()) {
                    *a += z.norm_sqr();
                }
                acc
            })
            .reduce(
                || vec![0.0; nb],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b) {
                        *x += y;
                    }
                    a
                },
            )
    }
    /// Records the outcome `bit` (probability `p`) of the highest remaining
    /// exponent qubit.
    pub fn collapse(&mut self, phi: f64, bit: bool, p: f64) {
        assert!(p > 0.0, "cannot collapse onto a zero-probability outcome");
        assert!(self.bits_left() > 0);
        let ph = Complex64::from_polar(1.0, phi) * if bit { -1.0 } else { 1.0 };
        self.levels.push((ph, 1.0 / (2.0 * p).sqrt()));
    }
    /// The arrays `C_{e''}` of the not-yet-measured exponent values `e''`
    /// (exact zeros dropped).
    pub fn materialize(&self) -> Vec<Arr<T>> {
        let nl = 1usize << self.bits_left();
        let ne = 1usize << self.w;
        let parts: Vec<Vec<Arr<T>>> = self
            .chunks()
            .into_par_iter()
            .map(|ch| {
                let est = (ch.1 - ch.0) + (ch.3 - ch.2);
                let mut out: Vec<Arr<T>> =
                    (0..nl).map(|_| Vec::with_capacity(est / ne + 16)).collect();
                self.groups(ch, |key, vals| {
                    let len = self.reduce(vals);
                    debug_assert_eq!(len, nl);
                    for (o, v) in out.iter_mut().zip(vals[..nl].iter()) {
                        let z = cvt::<T>(*v);
                        if z != Complex::zero() {
                            o.push((key, z));
                        }
                    }
                });
                out
            })
            .collect();
        (0..nl)
            .map(|e| {
                let total: usize = parts.iter().map(|p| p[e].len()).sum();
                let mut v = Vec::with_capacity(total);
                for p in &parts {
                    v.extend_from_slice(&p[e]);
                }
                v
            })
            .collect()
    }
}

/// Exact state of the windowed semiclassical circuit: the support of the
/// stored registers (sorted) with amplitudes.
#[derive(Clone, Debug)]
pub struct GeState<T: Real> {
    pub psi: Arr<T>,
    /// Peak number of stored branches between windows.
    pub peak: usize,
    /// Peak number of `(e, x)` branches inside a window.
    pub peak_branches: usize,
    /// Σ over windows of (branches evaluated) × (block ops).
    pub gate_branch_ops: u128,
    /// Seconds: build, eval, sort, window probabilities, new state.
    pub prof: [f64; 5],
}

/// One window's program and I/O.
pub struct WindowProg {
    pub prog: SlicedProgram,
    pub e: Vec<usize>,
    pub x: Vec<usize>,
    pub counts: MbuCounts,
}

impl WindowProg {
    pub fn new(lay: &GeLayout, ops: &[MbuOp]) -> Self {
        let prog = SlicedProgram::compile_ops(lay.nq, ops).expect("window block");
        Self {
            prog,
            e: lay.e.clone(),
            x: lay.stored(),
            counts: MbuCounts::of(ops),
        }
    }
}

impl<T: Real> GeState<T> {
    /// Basis state `key` (amplitude 1).
    pub fn basis(key: u64) -> Self {
        Self::from_branches(vec![(key, Complex::new(T::one(), T::zero()))])
    }
    pub fn from_branches(mut psi: Arr<T>) -> Self {
        psi.sort_unstable_by_key(|e| e.0);
        let peak = psi.len();
        Self {
            psi,
            peak,
            peak_branches: peak,
            gate_branch_ops: 0,
            prof: [0.0; 5],
        }
    }

    /// Runs the window block on every `(e, x)` branch (`e < 2^w`) and
    /// returns them, sorted by work-register key (see [`WindowArrays`]).
    /// The array is grown from `ψ`'s own buffer: segment `e` is a copy of
    /// `ψ` evaluated in place under exponent value `e` (`e = 0` last).
    pub fn window(&mut self, wp: &WindowProg, w_used: usize) -> WindowArrays<T> {
        let t0 = std::time::Instant::now();
        let ne = 1usize << w_used;
        let n_in = self.psi.len();
        // memory guard: QSIM_GE_MAX_GB caps the window's branch array
        if let Some(gb) = std::env::var("QSIM_GE_MAX_GB")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
        {
            let need = (ne * n_in * std::mem::size_of::<(u64, Complex<T>)>()) as f64 / 1e9;
            assert!(
                need <= gb,
                "window needs {need:.2} GB of branches (2^{w_used} × {n_in}), over QSIM_GE_MAX_GB = {gb}"
            );
        }
        let mut t_eval = 0.0;
        let mut ent = std::mem::take(&mut self.psi);
        ent.reserve_exact((ne - 1) * n_in);
        for e in 1..ne {
            ent.extend_from_within(0..n_in);
            let seg = ent.len() - n_in;
            let ta = std::time::Instant::now();
            eval_e_inplace(&wp.prog, &wp.e, &wp.x, e as u64, &mut ent[seg..seg + n_in]);
            t_eval += ta.elapsed().as_secs_f64();
        }
        // e = 0 last, in place; when it is the identity on every branch
        // (exact arithmetic) segment 0 is still ψ, already sorted
        let ta = std::time::Instant::now();
        let id = eval_e_inplace(&wp.prog, &wp.e, &wp.x, 0, &mut ent[..n_in]);
        t_eval += ta.elapsed().as_secs_f64();
        let wsh = w_used as u32;
        ent.par_chunks_mut(n_in.max(1))
            .enumerate()
            .for_each(|(e, seg)| {
                for v in seg.iter_mut() {
                    assert!(
                        v.0 >> (64 - wsh) == 0,
                        "key too wide to pack the exponent value"
                    );
                    v.0 = (v.0 << wsh) | e as u64;
                }
            });
        let split = if id { n_in } else { 0 };
        ent[split..].par_sort_unstable_by_key(|v| v.0);
        // V^e is a permutation: no (e, key) may appear twice
        assert!(
            ent[split..].par_windows(2).all(|w| w[0].0 != w[1].0),
            "window block is not injective on the support"
        );
        self.gate_branch_ops += (ne * n_in) as u128 * wp.prog.gates as u128;
        self.peak_branches = self.peak_branches.max(ne * n_in);
        self.prof[1] += t_eval;
        self.prof[2] += t0.elapsed().as_secs_f64() - t_eval;
        WindowArrays {
            w: w_used,
            ent: std::sync::Arc::new(ent),
            split,
            levels: Vec::new(),
        }
    }

    /// Takes the state back from a fully measured window.
    pub fn finish(&mut self, wa: WindowArrays<T>) {
        assert_eq!(wa.bits_left(), 0);
        let chunks4 = wa.chunks();
        let chunks: Vec<(usize, usize)> = chunks4.iter().map(|c| (c.2, c.3)).collect();
        let levels = wa.levels.clone();
        let (w, split) = (wa.w, wa.split);
        self.psi = if split > 0 {
            let tmp = WindowArrays {
                w,
                ent: std::sync::Arc::new(Vec::new()),
                split,
                levels: levels.clone(),
            };
            match std::sync::Arc::try_unwrap(wa.ent) {
                Ok(ent) => finish_runs_in_place(&tmp, ent, split, &chunks4),
                Err(ent) => {
                    let wa = WindowArrays { ent, ..tmp };
                    wa.materialize().pop().unwrap()
                }
            }
        } else {
            match std::sync::Arc::try_unwrap(wa.ent) {
                // sole owner, one run: the new state in place
                Ok(ent) => {
                    let tmp = WindowArrays {
                        w,
                        ent: std::sync::Arc::new(Vec::new()),
                        split: 0,
                        levels,
                    };
                    finish_in_place(&tmp, ent, &chunks)
                }
                Err(ent) => {
                    let tmp = WindowArrays {
                        w,
                        ent,
                        split: 0,
                        levels,
                    };
                    tmp.materialize().pop().unwrap()
                }
            }
        };
        self.peak = self.peak.max(self.psi.len());
    }
}

/// The new state of a fully measured window, written over the window's
/// own branch array: every key group of `ent` (sorted by packed key)
/// yields at most one output, written at or before the group's first
/// index, so each chunk compacts in place; the chunks are then moved
/// together. Same arithmetic as [`WindowArrays::materialize`].
fn finish_in_place<T: Real>(
    wa: &WindowArrays<T>,
    mut ent: Vec<(u64, Complex<T>)>,
    chunks: &[(usize, usize)],
) -> Arr<T> {
    let w = wa.w;
    let ne = 1usize << w;
    let mask = (1u64 << w) - 1;
    // split ent into the chunk slices
    let mut slices = Vec::with_capacity(chunks.len());
    {
        let mut rest: &mut [(u64, Complex<T>)] = &mut ent;
        let mut at = 0;
        for &(lo, hi) in chunks {
            let (_, r) = std::mem::take(&mut rest).split_at_mut(lo - at);
            let (c, r) = r.split_at_mut(hi - lo);
            slices.push(c);
            rest = r;
            at = hi;
        }
    }
    let counts: Vec<usize> = slices
        .into_par_iter()
        .map(|c| {
            let mut vals = [Complex64::zero(); 64];
            let mut out = 0usize;
            let n = c.len();
            let mut i = 0;
            while i < n {
                let key = c[i].0 >> w;
                while i < n && c[i].0 >> w == key {
                    vals[(c[i].0 & mask) as usize] = c64(c[i].1);
                    i += 1;
                }
                let len = wa.reduce(&mut vals[..ne]);
                debug_assert_eq!(len, 1);
                let z = cvt::<T>(vals[0]);
                if z != Complex::zero() {
                    c[out] = (key, z);
                    out += 1;
                }
                vals[..ne].iter_mut().for_each(|v| *v = Complex64::zero());
            }
            out
        })
        .collect();
    let mut dst = 0;
    for (&(lo, _), &cnt) in chunks.iter().zip(&counts) {
        if lo != dst {
            ent.copy_within(lo..lo + cnt, dst);
        }
        dst += cnt;
    }
    ent.truncate(dst);
    ent.shrink_to_fit();
    ent
}

/// [`finish_in_place`] for two sorted runs `ent[..split]` (`e = 0`, i.e.
/// `ψ`) and `ent[split..]`: outputs are written into the `e = 0` run behind
/// its read pointer; a chunk whose outputs would overtake that pointer
/// (the support grows in this window) continues in a side buffer.
fn finish_runs_in_place<T: Real>(
    wa: &WindowArrays<T>,
    mut ent: Vec<(u64, Complex<T>)>,
    split: usize,
    chunks: &[(usize, usize, usize, usize)],
) -> Arr<T> {
    let w = wa.w;
    let ne = 1usize << w;
    let mask = (1u64 << w) - 1;
    let results: Vec<(usize, Arr<T>)> = {
        let (a, b) = ent.split_at_mut(split);
        let b: &[(u64, Complex<T>)] = b;
        let mut jobs = Vec::with_capacity(chunks.len());
        let mut rest: &mut [(u64, Complex<T>)] = a;
        let mut at = 0;
        for &(a0, a1, b0, b1) in chunks {
            let (_, r) = std::mem::take(&mut rest).split_at_mut(a0 - at);
            let (c, r) = r.split_at_mut(a1 - a0);
            jobs.push((c, &b[b0 - split..b1 - split]));
            rest = r;
            at = a1;
        }
        jobs.into_par_iter()
            .map(|(a, b)| {
                let mut vals = [Complex64::zero(); 64];
                let (mut ia, mut ib, mut o) = (0usize, 0usize, 0usize);
                let mut ovf: Arr<T> = Vec::new();
                while ia < a.len() || ib < b.len() {
                    let ka = if ia < a.len() { a[ia].0 >> w } else { u64::MAX };
                    let kb = if ib < b.len() { b[ib].0 >> w } else { u64::MAX };
                    let key = ka.min(kb);
                    while ia < a.len() && a[ia].0 >> w == key {
                        vals[(a[ia].0 & mask) as usize] = c64(a[ia].1);
                        ia += 1;
                    }
                    while ib < b.len() && b[ib].0 >> w == key {
                        vals[(b[ib].0 & mask) as usize] = c64(b[ib].1);
                        ib += 1;
                    }
                    let len = wa.reduce(&mut vals[..ne]);
                    debug_assert_eq!(len, 1);
                    let z = cvt::<T>(vals[0]);
                    if z != Complex::zero() {
                        if ovf.is_empty() && o < ia {
                            a[o] = (key, z);
                            o += 1;
                        } else {
                            ovf.push((key, z));
                        }
                    }
                    vals[..ne].iter_mut().for_each(|v| *v = Complex64::zero());
                }
                (o, ovf)
            })
            .collect()
    };
    if results.iter().all(|r| r.1.is_empty()) {
        let mut dst = 0;
        for (&(a0, _, _, _), &(cnt, _)) in chunks.iter().zip(&results) {
            if a0 != dst {
                ent.copy_within(a0..a0 + cnt, dst);
            }
            dst += cnt;
        }
        ent.truncate(dst);
        ent.shrink_to_fit();
        ent
    } else {
        let total: usize = results.iter().map(|r| r.0 + r.1.len()).sum();
        let mut out = Vec::with_capacity(total);
        for (&(a0, _, _, _), (cnt, ovf)) in chunks.iter().zip(&results) {
            out.extend_from_slice(&ent[a0..a0 + cnt]);
            out.extend_from_slice(ovf);
        }
        out
    }
}

/// One exponent register of a phase-estimation schedule.
#[derive(Clone, Debug)]
pub struct ExpReg {
    /// Number of bits (rounds).
    pub len: usize,
    /// The base `b`: round `i` of this register applies `b^{2^(len−1−i)}`.
    pub base: u64,
}

/// Per-run statistics.
#[derive(Clone, Debug, Default)]
pub struct GeRun {
    /// Measured integer of each register (bit `i` = round `i`).
    pub y: Vec<u128>,
    pub counts: MbuCounts,
    /// Slice steps (all windows).
    pub steps: usize,
    pub windows: usize,
    pub peak: usize,
    pub peak_branches: usize,
    pub gate_branch_ops: u128,
    pub qubits: usize,
    pub prof: [f64; 5],
}

/// The windows of a register of `len` rounds with window `we`: round
/// ranges `[i0, i0 + w)`, in round order.
pub fn windows(len: usize, we: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    let mut i0 = 0;
    while i0 < len {
        let w = we.min(len - i0);
        v.push((i0, w));
        i0 += w;
    }
    v
}

/// `base^(2^k) mod N`.
pub fn pow2k(base: u64, k: usize, n_mod: u64) -> u64 {
    let mut g = base % n_mod;
    for _ in 0..k {
        g = mul_mod(g, g, n_mod);
    }
    g
}

/// Griffiths–Niu phase correction for round `i` given the bits `y_low`
/// already measured in this register.
pub fn correction(i: usize, y_low: u128) -> f64 {
    if y_low == 0 {
        0.0
    } else {
        -PI * (y_low as f64) / ((1u128 << i) as f64)
    }
}

/// Initial stored branches: `x = 1` (exact) or the coset states
/// `Σ_j |1 + jN⟩ ⊗ Σ_j' |j'N⟩ / 2^c` (coset).
pub fn initial<T: Real>(lay: &GeLayout, n_mod: u64, c: usize) -> Arr<T> {
    if c == 0 {
        return vec![(1, Complex::new(T::one(), T::zero()))];
    }
    let nr = lay.nr;
    let amp = T::from_f64(1.0 / (1u64 << c) as f64);
    let mut v = Vec::with_capacity(1 << (2 * c));
    for j in 0..1u64 << c {
        for j2 in 0..1u64 << c {
            let x = 1 + j * n_mod;
            let b = j2 * n_mod;
            assert!(x < 1 << nr && b < 1 << nr);
            v.push((x | (b << nr), Complex::new(amp, T::zero())));
        }
    }
    v.sort_unstable_by_key(|e| e.0);
    v
}

/// Mixes `(N, g, window)` into an outcome-stream seed.
fn outcome_seed(n_mod: u64, g: u64, k: u64) -> u64 {
    n_mod.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ g.rotate_left(17) ^ k.wrapping_mul(0x9E37_79B9)
}

/// One sampled run of the windowed phase estimation over `regs` (each
/// register its own semiclassical QFT), drawing outcomes from `rng`
/// (a uniform `[0,1)` source).
pub fn run<T: Real>(
    n_mod: u64,
    regs: &[ExpReg],
    o: &GeOpts,
    rng: &mut dyn FnMut() -> f64,
) -> GeRun {
    let n = crate::shor::work_bits(n_mod);
    let lay = GeLayout::new(n, o);
    let mut st = GeState::<T>::from_branches(initial(&lay, n_mod, o.coset));
    let mut r = GeRun {
        qubits: lay.nq,
        ..GeRun::default()
    };
    let mut wi = 0u64;
    for reg in regs {
        let mut y = 0u128;
        let wins = windows(reg.len, o.we);
        let last_reg = std::ptr::eq(reg, regs.last().unwrap());
        for (k, &(i0, w)) in wins.iter().enumerate() {
            let t0 = std::time::Instant::now();
            // lowest power of the window: 2^(len − i0 − w)
            let g = pow2k(reg.base, reg.len - i0 - w, n_mod);
            let mut oc = Outcomes::from_env(outcome_seed(n_mod, g, wi));
            wi += 1;
            let ops = window_block(&lay, g, n_mod, o, &mut oc);
            let wp = WindowProg::new(&lay, &ops);
            drop(ops);
            r.counts.add(&wp.counts);
            r.steps += wp.prog.len();
            r.windows += 1;
            st.prof[0] += t0.elapsed().as_secs_f64();
            let mut wa = st.window(&wp, w);
            let tm = std::time::Instant::now();
            let final_window = last_reg && k + 1 == wins.len();
            // the joint distribution of the window's w bits in one pass,
            // then one conditional draw per round (as in
            // `shor::run_semiclassical`)
            let y0 = y;
            let jp = wa.joint_probs(&|j, pre| correction(i0 + j, y0 | (u128::from(pre) << i0)));
            let t_joint = tm.elapsed().as_secs_f64();
            let tot: f64 = jp.iter().sum();
            assert!((tot - 1.0).abs() < 1e-6, "window norm {tot}");
            let mut pre = 0u64;
            for j in 0..w {
                let i = i0 + j;
                let phi = correction(i, y);
                let (mut q0, mut q1) = (0.0, 0.0);
                for (b, &p) in jp.iter().enumerate() {
                    let b = b as u64;
                    if b & ((1 << j) - 1) == pre {
                        if (b >> j) & 1 == 1 {
                            q1 += p;
                        } else {
                            q0 += p;
                        }
                    }
                }
                let s = q0 + q1;
                let bit = rng() < q1 / s;
                if bit {
                    y |= 1 << i;
                    pre |= 1 << j;
                }
                // conditional probability of this outcome given the prefix
                let pc = if bit { q1 } else { q0 } / s;
                wa.collapse(phi, bit, pc);
            }
            // the measured integer is complete after the last P(1): the
            // final state (the largest support) is not materialised
            if !final_window {
                st.finish(wa);
            }
            st.prof[3] += t_joint;
            st.prof[4] += tm.elapsed().as_secs_f64() - t_joint;
            if std::env::var_os("QSIM_GE_PROFILE").is_some() {
                eprintln!(
                    "[ge window {k}] in={} joint {:.3}s finish {:.3}s out={}",
                    r.peak_branches,
                    t_joint,
                    tm.elapsed().as_secs_f64() - t_joint,
                    st.psi.len()
                );
            }
        }
        r.y.push(y);
    }
    r.peak = st.peak;
    r.peak_branches = st.peak_branches;
    r.gate_branch_ops = st.gate_branch_ops;
    r.prof = st.prof;
    r
}

/// The exact distribution of the measured registers (concatenated, first
/// register in the low bits), by walking the whole tree of outcomes.
pub fn distribution(n_mod: u64, regs: &[ExpReg], o: &GeOpts, prune: f64) -> Vec<f64> {
    let n = crate::shor::work_bits(n_mod);
    let lay = GeLayout::new(n, o);
    let total: usize = regs.iter().map(|r| r.len).sum();
    assert!(total <= 26);
    // windows in order: (reg idx, i0, w, g, offset of the register in y)
    let mut plan = Vec::new();
    let mut off = 0;
    let mut wi = 0u64;
    for (ri, reg) in regs.iter().enumerate() {
        for (i0, w) in windows(reg.len, o.we) {
            let g = pow2k(reg.base, reg.len - i0 - w, n_mod);
            let mut oc = Outcomes::from_env(outcome_seed(n_mod, g, wi));
            wi += 1;
            let ops = window_block(&lay, g, n_mod, o, &mut oc);
            plan.push((ri, i0, w, off, WindowProg::new(&lay, &ops)));
        }
        off += reg.len;
    }
    let mut out = vec![0.0; 1 << total];
    #[allow(clippy::too_many_arguments)]
    fn walk(
        plan: &[(usize, usize, usize, usize, WindowProg)],
        k: usize,
        st: GeState<f64>,
        y: u128,
        p: f64,
        prune: f64,
        out: &mut [f64],
    ) {
        if k == plan.len() {
            out[y as usize] += p;
            return;
        }
        let (_, _, w, _, ref wp) = plan[k];
        let mut st = st;
        let wa = st.window(wp, w);
        #[allow(clippy::too_many_arguments)]
        fn inner(
            plan: &[(usize, usize, usize, usize, WindowProg)],
            k: usize,
            st: &GeState<f64>,
            wa: WindowArrays<f64>,
            j: usize,
            y: u128,
            p: f64,
            prune: f64,
            out: &mut [f64],
        ) {
            let (_, i0, w, off, _) = plan[k];
            if j == w {
                let mut s2 = st.clone();
                s2.finish(wa);
                walk(plan, k + 1, s2, y, p, prune, out);
                return;
            }
            let i = i0 + j;
            let y_reg = (y >> off) & ((1u128 << i) - 1);
            let phi = correction(i, y_reg);
            let (p0, p1) = wa.probs(phi);
            for (bit, pb) in [(false, p0), (true, p1)] {
                if pb <= prune {
                    continue;
                }
                let mut w2 = wa.clone();
                w2.collapse(phi, bit, pb);
                inner(
                    plan,
                    k,
                    st,
                    w2,
                    j + 1,
                    y | (u128::from(bit) << (off + i)),
                    p * pb,
                    prune,
                    out,
                );
            }
        }
        inner(plan, k, &st, wa, 0, y, p, prune, out);
    }
    let st = GeState::<f64>::from_branches(initial(&lay, n_mod, o.coset));
    walk(&plan, 0, st, 0, 1.0, prune, &mut out);
    out
}

/// The same distribution through a genuinely quantum gate-by-gate
/// reference: a [`crate::engines::sparse::SparseState`] on all `nq ≤ 64` qubits,
/// real `H` on the exponent qubits, every block gate applied in turn, every
/// X-basis measurement as `H` + projection (its probability asserted to be
/// 1/2), then the window's semiclassical measurements as `Phase`, `H` and a
/// projective measurement of each exponent qubit, highest power first.
pub fn distribution_sparse(n_mod: u64, regs: &[ExpReg], o: &GeOpts, prune: f64) -> Vec<f64> {
    use crate::engines::sparse::SparseState;
    let n = crate::shor::work_bits(n_mod);
    let lay = GeLayout::new(n, o);
    assert!(lay.nq <= 64);
    let total: usize = regs.iter().map(|r| r.len).sum();
    let mut plan = Vec::new();
    let mut off = 0;
    let mut wi = 0u64;
    for (ri, reg) in regs.iter().enumerate() {
        for (i0, w) in windows(reg.len, o.we) {
            let g = pow2k(reg.base, reg.len - i0 - w, n_mod);
            let mut oc = Outcomes::from_env(outcome_seed(n_mod, g, wi));
            wi += 1;
            plan.push((ri, i0, w, off, window_block(&lay, g, n_mod, o, &mut oc)));
        }
        off += reg.len;
    }
    let stored = lay.stored();
    let init: Vec<(u64, Complex64)> = initial::<f64>(&lay, n_mod, o.coset)
        .into_iter()
        .map(|(k, a)| {
            let mut key = 0u64;
            for (j, &q) in stored.iter().enumerate() {
                key |= ((k >> j) & 1) << q;
            }
            (key, a)
        })
        .collect();
    let s = SparseState::from_amplitudes(lay.nq, init);
    let mut out = vec![0.0; 1 << total];
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn walk(
        plan: &[(usize, usize, usize, usize, Vec<MbuOp>)],
        e: &[usize],
        k: usize,
        j: usize,
        s: SparseState,
        y: u128,
        p: f64,
        prune: f64,
        out: &mut [f64],
    ) {
        if k == plan.len() {
            out[y as usize] += p;
            return;
        }
        let (_, i0, w, off, ref ops) = plan[k];
        let mut s = s;
        if j == 0 {
            for &q in &e[..w] {
                s.apply_gate(&Gate::H(q)).unwrap();
            }
            for op in ops {
                match *op {
                    MbuOp::G(g) => s.apply_gate(&g).unwrap(),
                    MbuOp::MeasX(q, m) => {
                        s.apply_gate(&Gate::H(q)).unwrap();
                        let pm = s.collapse(q, m);
                        assert!((pm - 0.5).abs() < 1e-9, "P(m) = {pm}");
                        if m {
                            s.apply_gate(&Gate::X(q)).unwrap();
                        }
                    }
                    MbuOp::GlobalNeg => {}
                }
            }
        }
        if j == w {
            walk(plan, e, k + 1, 0, s, y, p, prune, out);
            return;
        }
        let i = i0 + j;
        let q = e[w - 1 - j]; // highest remaining power first
        let y_reg = (y >> off) & ((1u128 << i) - 1);
        let phi = correction(i, y_reg);
        if phi != 0.0 {
            s.apply_gate(&Gate::Phase(q, phi)).unwrap();
        }
        s.apply_gate(&Gate::H(q)).unwrap();
        let p1 = s.prob_one(q);
        for (bit, pb) in [(false, 1.0 - p1), (true, p1)] {
            if pb <= prune {
                continue;
            }
            let mut c = s.clone();
            c.collapse(q, bit);
            if bit {
                c.apply_gate(&Gate::X(q)).unwrap(); // reset
            }
            walk(
                plan,
                e,
                k,
                j + 1,
                c,
                y | (u128::from(bit) << (off + i)),
                p * pb,
                prune,
                out,
            );
        }
    }
    walk(&plan, &lay.e, 0, 0, s, 0, 1.0, prune, &mut out);
    out
}

// ---------------------------------------------------------------------------
// Shor and Ekerå–Håstad schedules
// ---------------------------------------------------------------------------

/// Shor's order finding: one register of `2n` bits, base `a`.
pub fn shor_regs(n_mod: u64, a: u64) -> Vec<ExpReg> {
    vec![ExpReg {
        len: 2 * crate::shor::work_bits(n_mod),
        base: a,
    }]
}

/// Ekerå–Håstad parameters for `N` with `n` bits: `m = ⌈n/2⌉`, a bound on
/// the bit length of `d = (p + q − 2)/2` whenever `p, q < 2^{⌈n/2⌉}`
/// (balanced RSA-type semiprimes); `ℓ = m` (`s = 1`).
pub fn eh_m(n_mod: u64) -> usize {
    let n = 64 - n_mod.leading_zeros() as usize;
    n.div_ceil(2)
}

/// `y = g^{(N−1)/2} mod N`, whose discrete logarithm to the base `g` is
/// `d = (p + q − 2)/2` (`λ(N)` divides `φ(N)/2 = (N − 1)/2 − d`).
pub fn eh_target(n_mod: u64, g: u64) -> u64 {
    pow_mod(g, (n_mod - 1) / 2, n_mod)
}

/// Ekerå–Håstad registers, in the order they are run: the `m`-bit register
/// `b` with base `y^{−1}` first, then the `2m`-bit register `a` with base
/// `g` (computing `g^a y^{−b}`). The two registers' controlled
/// multiplications commute and each has its own semiclassical QFT, so the
/// order does not change the distribution of `(j, k)`; it changes the
/// simulator's support. Running `b` first keeps the support inside
/// `{y^{−b}}·⟨g^{2^{2m−i}}⟩`, which is `r_odd`-sized for most of the run
/// when `y = g^d` lies in the odd-order part (`d ≡ 0 mod 2^{ν₂(r)}`), and
/// at most `r` otherwise; `a` first reaches the full group `⟨g⟩` (`r`)
/// after `2m` rounds and stays there for all `m` rounds of `b`.
pub fn eh_regs(n_mod: u64, g: u64) -> Vec<ExpReg> {
    let m = eh_m(n_mod);
    let y = eh_target(n_mod, g);
    vec![
        ExpReg {
            len: m,
            base: mod_inverse(y, n_mod),
        },
        ExpReg {
            len: 2 * m,
            base: g,
        },
    ]
}

fn isqrt(v: u128) -> u128 {
    if v < 2 {
        return v;
    }
    let mut x = (v as f64).sqrt() as u128;
    while x * x > v {
        x -= 1;
    }
    while (x + 1) * (x + 1) <= v {
        x += 1;
    }
    x
}

/// `p, q` from `p + q = s` and `pq = N`, if integral.
pub fn split_from_sum(n_mod: u64, s: u128) -> Option<(u64, u64)> {
    let n = u128::from(n_mod);
    let disc = s.checked_mul(s)?.checked_sub(4 * n)?;
    let r = isqrt(disc);
    if r * r != disc || (s + r) % 2 != 0 {
        return None;
    }
    let p = (s - r) / 2;
    let q = (s + r) / 2;
    (p > 1 && p * q == n).then_some((p as u64, q as u64))
}

/// Ekerå–Håstad classical post-processing for `s = 1` (Ekerå–Håstad 2017,
/// §4–5; Ekerå 2020, "On post-processing in the quantum algorithm for
/// computing short discrete logarithms", Des. Codes Cryptogr. 88): the
/// lattice `L` spanned by `(j, 1)` and `(2^{ℓ+m}, 0)` contains
/// `v = ((d j) mod 2^{ℓ+m}, d)`, which for a good pair lies within
/// `≈ 2^m` of `u = (−2^m k mod 2^{ℓ+m}, 0)`. We Lagrange-reduce the basis
/// and enumerate **every** lattice vector `v` with `|v_1 − u_1| ≤ 2^{m+1}`
/// and `0 < v_2 < 2^m` (at most `max_cands`), accepting the first `d' = v_2`
/// with `g^{d'} = y (mod N)` for which `p + q = 2d' + 2` splits `N`.
/// Returns the factors and the number of candidates tested.
pub fn eh_postprocess(
    n_mod: u64,
    g: u64,
    j: u128,
    k: u128,
    max_cands: usize,
) -> (Option<(u64, u64)>, usize) {
    let m = eh_m(n_mod) as u32;
    let y = eh_target(n_mod, g);
    let modl = 1i128 << (2 * m);
    let u1 = (-((k as i128) << m)).rem_euclid(modl);
    // basis rows (x, z)
    let mut b1 = (j as i128 % modl, 1i128);
    let mut b2 = (modl, 0i128);
    let dot = |a: (i128, i128), b: (i128, i128)| a.0 * b.0 + a.1 * b.1;
    // Lagrange reduction (exact integer arithmetic)
    loop {
        if dot(b1, b1) > dot(b2, b2) {
            std::mem::swap(&mut b1, &mut b2);
        }
        let n1 = dot(b1, b1);
        let num = dot(b1, b2);
        // nearest integer to num / n1
        let q = (2 * num + n1).div_euclid(2 * n1);
        if q == 0 {
            break;
        }
        b2 = (b2.0 - q * b1.0, b2.1 - q * b1.1);
        if dot(b2, b2) >= n1 {
            break;
        }
    }
    // enumerate lattice points v = a·b1 + c·b2 in the box
    let d = 1i128 << (m + 1);
    let box_x = (u1 - d, u1 + d);
    let box_z = (1i128, (1i128 << m) - 1);
    // coefficient ranges from the inverse basis over the box corners
    let det = b1.0 * b2.1 - b1.1 * b2.0;
    assert!(det != 0);
    let coef = |x: i128, z: i128| {
        // solve a*b1 + c*b2 = (x, z)
        let a = (x as f64 * b2.1 as f64 - z as f64 * b2.0 as f64) / det as f64;
        let c = (b1.0 as f64 * z as f64 - b1.1 as f64 * x as f64) / det as f64;
        (a, c)
    };
    let corners = [
        coef(box_x.0, box_z.0),
        coef(box_x.0, box_z.1),
        coef(box_x.1, box_z.0),
        coef(box_x.1, box_z.1),
    ];
    let amin = corners
        .iter()
        .map(|c| c.0)
        .fold(f64::INFINITY, f64::min)
        .floor() as i128
        - 1;
    let amax = corners
        .iter()
        .map(|c| c.0)
        .fold(f64::NEG_INFINITY, f64::max)
        .ceil() as i128
        + 1;
    let cmin = corners
        .iter()
        .map(|c| c.1)
        .fold(f64::INFINITY, f64::min)
        .floor() as i128
        - 1;
    let cmax = corners
        .iter()
        .map(|c| c.1)
        .fold(f64::NEG_INFINITY, f64::max)
        .ceil() as i128
        + 1;
    if (amax - amin + 1).saturating_mul(cmax - cmin + 1) > (max_cands as i128) * 64 {
        return (None, 0);
    }
    let mut tested = 0;
    for a in amin..=amax {
        for c in cmin..=cmax {
            let v = (a * b1.0 + c * b2.0, a * b1.1 + c * b2.1);
            // translate x into the box (v_x is only defined mod 2^{2m} up to
            // the lattice, which contains (2^{2m}, 0))
            if v.0 < box_x.0 || v.0 > box_x.1 || v.1 < box_z.0 || v.1 > box_z.1 {
                continue;
            }
            tested += 1;
            if tested > max_cands {
                return (None, tested);
            }
            let dd = v.1 as u64;
            if pow_mod(g, dd, n_mod) == y {
                if let Some(f) = split_from_sum(n_mod, 2 * u128::from(dd) + 2) {
                    return (Some(f), tested);
                }
            }
        }
    }
    (None, tested)
}

/// One Ekerå–Håstad factoring run: returns the run statistics and the
/// factors found by the post-processing (if any).
pub fn eh_run<T: Real>(
    n_mod: u64,
    g: u64,
    o: &GeOpts,
    rng: &mut dyn FnMut() -> f64,
) -> (GeRun, Option<(u64, u64)>) {
    assert_eq!(gcd(g, n_mod), 1);
    let r = run::<T>(n_mod, &eh_regs(n_mod, g), o, rng);
    // registers run b first: y[0] = k (m bits), y[1] = j (2m bits)
    let (f, _) = eh_postprocess(n_mod, g, r.y[1], r.y[0], 4096);
    (r, f)
}

/// One Shor order-finding run with exponent windows; returns the run and
/// `(order, factor)` from the standard post-processing.
pub fn shor_run<T: Real>(
    n_mod: u64,
    a: u64,
    o: &GeOpts,
    rng: &mut dyn FnMut() -> f64,
) -> (GeRun, Option<u64>, Option<u64>) {
    let r = run::<T>(n_mod, &shor_regs(n_mod, a), o, rng);
    let t = 2 * crate::shor::work_bits(n_mod) as u32;
    let (order, factor) = crate::shor::postprocess(n_mod, a, r.y[0], t);
    (r, order, factor)
}

/// Gate counts of all window blocks of a schedule (outcomes from the same
/// streams as [`run`]), without simulating.
pub fn schedule_counts(n_mod: u64, regs: &[ExpReg], o: &GeOpts) -> (MbuCounts, usize, usize) {
    let n = crate::shor::work_bits(n_mod);
    let lay = GeLayout::new(n, o);
    let mut c = MbuCounts::default();
    let mut steps = 0;
    let mut wi = 0u64;
    for reg in regs {
        for (i0, w) in windows(reg.len, o.we) {
            let g = pow2k(reg.base, reg.len - i0 - w, n_mod);
            let mut oc = Outcomes::from_env(outcome_seed(n_mod, g, wi));
            wi += 1;
            let ops = window_block(&lay, g, n_mod, o, &mut oc);
            c.add(&MbuCounts::of(&ops));
            steps += SlicedProgram::compile_ops(lay.nq, &ops).unwrap().len();
        }
    }
    (c, steps, lay.nq)
}

/// Statistics of one coset-arithmetic path sampled in lockstep with the
/// exact-arithmetic circuit ([`coset_path`]).
#[derive(Clone, Debug, Default)]
pub struct CosetPath {
    /// Measured integer (drawn from the coset circuit's own distribution).
    pub y: u128,
    /// `ln Q(y)` (coset circuit) and `ln P(y)` (exact circuit; `−∞` if the
    /// exact circuit cannot produce `y`).
    pub ln_q: f64,
    pub ln_p: f64,
    /// Deviant weight after every window: the probability on coset branches
    /// `|x_r⟩|b_r⟩` that are *not* congruent to the exact circuit's state
    /// after the same outcomes, i.e. `b_r ≢ 0 (mod N)` or `x_r mod N` outside
    /// the exact support (a wrapped `(n + c)`-bit addition). A shift of the
    /// coset index without a wrap is not a deviation.
    pub infidelity: Vec<f64>,
    /// Peak coset support.
    pub peak: usize,
}

/// Weight of the coset branches not congruent to the exact state.
fn coset_bad_weight(n_mod: u64, nr: usize, exact: &Arr<f64>, cos: &Arr<f64>) -> f64 {
    let mask = (1u64 << nr) - 1;
    let mut bad = 0.0;
    for &(k, a) in cos {
        let (xr, br) = (k & mask, k >> nr);
        if br % n_mod != 0 || exact.binary_search_by_key(&(xr % n_mod), |e| e.0).is_err() {
            bad += a.norm_sqr();
        }
    }
    bad
}

/// Samples one path of the coset circuit (`o.coset = c > 0`) and runs the
/// exact circuit (`coset = 0`, same windows) on the same outcomes, so that
/// `Q(y)`, `P(y)` and the state infidelity after every window are exact
/// for this path. `TV(P, Q) = E_{y∼Q}[(1 − P(y)/Q(y))_+]` and
/// `P_succ(exact) = E_{y∼Q}[succ(y) P(y)/Q(y)]` are then unbiased path
/// averages.
pub fn coset_path(n_mod: u64, a: u64, o: &GeOpts, rng: &mut dyn FnMut() -> f64) -> CosetPath {
    assert!(o.coset > 0);
    let n = crate::shor::work_bits(n_mod);
    let oe = GeOpts { coset: 0, ..*o };
    let lay_c = GeLayout::new(n, o);
    let lay_e = GeLayout::new(n, &oe);
    let mut sc = GeState::<f64>::from_branches(initial(&lay_c, n_mod, o.coset));
    let mut se = GeState::<f64>::basis(1);
    let mut alive = true;
    let mut out = CosetPath::default();
    let t = 2 * n;
    let wins = windows(t, o.we);
    for (k, &(i0, w)) in wins.iter().enumerate() {
        let g = pow2k(a, t - i0 - w, n_mod);
        let mut oc = Outcomes::from_env(outcome_seed(n_mod, g, k as u64));
        let wpc = WindowProg::new(&lay_c, &window_block(&lay_c, g, n_mod, o, &mut oc));
        let mut wac = sc.window(&wpc, w);
        let mut wae = if alive {
            let mut oc = Outcomes::from_env(outcome_seed(n_mod, g, k as u64));
            let wpe = WindowProg::new(&lay_e, &window_block(&lay_e, g, n_mod, &oe, &mut oc));
            Some(se.window(&wpe, w))
        } else {
            None
        };
        for j in 0..w {
            let i = i0 + j;
            let phi = correction(i, out.y);
            let (q0, q1) = wac.probs(phi);
            let bit = rng() * (q0 + q1) < q1;
            let qb = if bit { q1 } else { q0 } / (q0 + q1);
            out.ln_q += qb.ln();
            wac.collapse(phi, bit, qb * (q0 + q1));
            if let Some(wa) = wae.as_mut() {
                let (p0, p1) = wa.probs(phi);
                let pb = if bit { p1 } else { p0 };
                if pb <= 1e-300 {
                    alive = false;
                    out.ln_p = f64::NEG_INFINITY;
                } else {
                    out.ln_p += (pb / (p0 + p1)).ln();
                    wa.collapse(phi, bit, pb);
                }
            }
            if !alive {
                wae = None;
            }
            if bit {
                out.y |= 1 << i;
            }
        }
        sc.finish(wac);
        if let Some(wa) = wae {
            se.finish(wa);
            out.infidelity
                .push(coset_bad_weight(n_mod, lay_c.nr, &se.psi, &sc.psi));
        } else {
            out.infidelity.push(f64::NAN);
        }
    }
    out.peak = sc.peak;
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shor::mbu::eval_on_key;

    fn check_block(n_mod: u64, o: &GeOpts, mode: u8) {
        let n = crate::shor::work_bits(n_mod);
        let lay = GeLayout::new(n, o);
        assert!(lay.nq <= 128);
        for g in (2..n_mod).filter(|&a| gcd(a, n_mod) == 1).take(2) {
            let mut oc = Outcomes::new(g * 31 + n_mod, mode);
            let ops = window_block(&lay, g, n_mod, o, &mut oc);
            let sign0 = eval_on_key(&ops, 0).1;
            for e in 0..1u64 << o.we {
                let ge = pow_mod(g, e, n_mod);
                for x in 0..n_mod {
                    let mut k = 0u128;
                    for (j, &q) in lay.e.iter().enumerate() {
                        k |= u128::from((e >> j) & 1) << q;
                    }
                    let mut want = k;
                    for (j, &q) in lay.x.iter().enumerate() {
                        k |= u128::from((x >> j) & 1) << q;
                        want |= u128::from((mul_mod(ge, x, n_mod) >> j) & 1) << q;
                    }
                    let (out, s) = eval_on_key(&ops, k);
                    assert_eq!(s, sign0, "N={n_mod} {o:?} e={e} x={x}: sign");
                    assert_eq!(out, want, "N={n_mod} {o:?} g={g} e={e} x={x}");
                }
            }
        }
    }

    #[test]
    fn window_block_exhaustive_small() {
        for n_mod in [15u64, 21, 35, 55, 77] {
            for we in 1..=3 {
                for wm in 1..=3 {
                    for mbu in [MbuOpts::ALL, MbuOpts::LOOKUPS, MbuOpts::NONE] {
                        let o = GeOpts {
                            we,
                            wm,
                            mbu,
                            coset: 0,
                        };
                        for mode in 0..3 {
                            check_block(n_mod, &o, mode);
                        }
                    }
                }
            }
        }
    }

    /// Coset blocks are not exact, but every coset branch whose additions
    /// do not wrap maps `x + jN` to a value `≡ g^e x (mod N)` and leaves the
    /// accumulator `≡ 0`.
    #[test]
    fn coset_block_is_congruent() {
        for n_mod in [15u64, 21, 35] {
            let n = crate::shor::work_bits(n_mod);
            for (we, wm, c) in [(1, 2, 3), (2, 2, 4), (2, 3, 3)] {
                let o = GeOpts {
                    we,
                    wm,
                    mbu: MbuOpts::ALL,
                    coset: c,
                };
                let lay = GeLayout::new(n, &o);
                let g = 2;
                let mut oc = Outcomes::new(5, 0);
                let ops = window_block(&lay, g, n_mod, &o, &mut oc);
                let sign0 = eval_on_key(&ops, 0).1;
                let mut good = 0;
                let mut tot = 0;
                for e in 0..1u64 << we {
                    for x in 0..n_mod {
                        for j in 0..1u64 << c {
                            for j2 in 0..1u64 << c {
                                let xv = x + j * n_mod;
                                let bv = j2 * n_mod;
                                let mut k = 0u128;
                                for (i, &q) in lay.e.iter().enumerate() {
                                    k |= u128::from((e >> i) & 1) << q;
                                }
                                for i in 0..lay.nr {
                                    k |= u128::from((xv >> i) & 1) << lay.x[i];
                                    k |= u128::from((bv >> i) & 1) << lay.b[i];
                                }
                                let (out, s) = eval_on_key(&ops, k);
                                assert_eq!(s, sign0);
                                let rd = |reg: &[usize]| {
                                    reg.iter()
                                        .enumerate()
                                        .map(|(i, &q)| (((out >> q) & 1) as u64) << i)
                                        .sum::<u64>()
                                };
                                let (xo, bo) = (rd(&lay.x), rd(&lay.b));
                                tot += 1;
                                if xo % n_mod == mul_mod(pow_mod(g, e, n_mod), x, n_mod)
                                    && bo % n_mod == 0
                                {
                                    good += 1;
                                }
                            }
                        }
                    }
                }
                // most branches are congruent; wrapped ones are the deviation
                assert!(good * 10 >= tot * 7, "N={n_mod} c={c}: {good}/{tot}");
            }
        }
    }

    #[test]
    fn eh_postprocess_finds_d_from_ideal_pairs() {
        // N = 1 005 973 = 997 × 1009; g = 2; d = (p + q − 2)/2 = 1002
        let n_mod = 1_005_973u64;
        let g = 2u64;
        let m = eh_m(n_mod) as u32;
        let d = 1002u128;
        assert_eq!(pow_mod(g, d as u64, n_mod), eh_target(n_mod, g));
        let mut ok = 0;
        for j in (1u128..1 << (2 * m)).step_by(997) {
            // the k that makes {d j + 2^m k} smallest
            let modl = 1u128 << (2 * m);
            let r = (d * j) % modl;
            let k = ((modl - r + (1 << (m - 1))) >> m) % (1 << m);
            let (f, _) = eh_postprocess(n_mod, g, j, k, 4096);
            if f == Some((997, 1009)) {
                ok += 1;
            }
        }
        assert!(ok > 0);
    }
}
