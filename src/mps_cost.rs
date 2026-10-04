//! Predicting the cost of an exact MPS run without running it
//! (research/planner.md §2).
//!
//! Two pieces:
//!
//! * [`BondBounds`]: rigorous upper bounds on the Schmidt rank of the
//!   evolving state across any bipartition, maintained gate by gate in
//!   O(n) to O(m) word operations per gate:
//!   - `Cut`: `min(|A|, |B|)`;
//!   - `Cross`: the time-resolved crossing count (each two-qubit gate adds
//!     its operator-Schmidt bits to every line cut it straddles), extended
//!     to non-prefix bipartitions by `χ(A ∪ {q}) ≤ 2 χ(A)`;
//!   - `Stab`: write the state as `|ψ_t> = O_t |S_t>` with `|S_t>` the
//!     stabilizer state of the Clifford part and `O_t` a product of Pauli
//!     rotations whose axes are pushed through every later Clifford.
//!     `χ_A ≤ 2^{e_A(S_t)} · OSR(O_t) ≤ 2^{e_A(S_t) + s_A(t)}`, where
//!     `e_A` is the exact stabilizer entanglement (rank of the stabilizer
//!     group projected on `A`, minus `|A|`) and `s_A` counts the rotation
//!     axes that act on both sides of the cut;
//!   - `Coset`: `|ψ_t>` lies in `span{g|S_t> : g ∈ ⟨axes⟩}`; grouping `g`
//!     by its coset modulo the stabilizer projected on `A` gives
//!     `χ_A ≤ 2^{rank π_A(Stab + ⟨axes⟩) − |A|}` (and the same for `B`);
//!   - `Affine`: every wire is a constant, an affine function of the
//!     branching variables (one per non-monomial one-qubit gate) or opaque;
//!     the Schmidt rank is at most the number of distinct `A`-parts of the
//!     support, `2^{rank(affine forms on A) + #opaque on A}` (and `B`).
//!
//!   `Best` is the minimum of all of them.
//! * [`replay`]: a symbolic re-run of [`crate::mps::Mps::apply_gate`]'s exact
//!   control flow (orthogonality-centre moves, SWAP routing, Toffoli
//!   decomposition) that tracks only bond dimensions. Every SVD keeps
//!   `min(2 dl, 2 dr, bound)`, so the replayed bond profile upper-bounds the
//!   real one step by step, and the replay returns the same [`MpsStats`]
//!   the real engine counts. Fed the real engine's bond trace it reproduces
//!   the real counts exactly (tested); fed a bound it predicts them.

use crate::circuit::{Circuit, Op, SimError};
use crate::gate::{is_multiple_of_half_pi, Gate};
use crate::mps::{matmul_work, qr_work, svd_work, MpsStats};
use std::f64::consts::FRAC_PI_2;

/// Largest register [`BondBounds`] handles with its Pauli bit masks; above
/// it only `Cut` and `Cross` are used.
pub const MAX_QUBITS: usize = 64;

/// Which bound feeds the replay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Estimator {
    Cut,
    Cross,
    Stab,
    Coset,
    Affine,
    Best,
}

impl Estimator {
    pub const ALL: [Estimator; 6] = [
        Estimator::Cut,
        Estimator::Cross,
        Estimator::Stab,
        Estimator::Coset,
        Estimator::Affine,
        Estimator::Best,
    ];
    pub fn name(&self) -> &'static str {
        match self {
            Estimator::Cut => "cut",
            Estimator::Cross => "cross",
            Estimator::Stab => "stab",
            Estimator::Coset => "coset",
            Estimator::Affine => "affine",
            Estimator::Best => "best",
        }
    }
}

/// A wire of the affine support tracker: the same known value on every
/// branch, an affine function of the branching variables (the form only;
/// constant offsets do not change any rank), or an opaque function of them.
#[derive(Clone, Debug)]
enum Wire {
    /// Constant on every branch; `None`: value not tracked.
    Const(Option<bool>),
    Affine(Vec<u64>),
    Opaque,
}

/// log2 of the operator-Schmidt rank of a two-qubit gate across its qubits.
fn schmidt_bits(g: &Gate) -> usize {
    match g {
        Gate::Cnot(..) | Gate::Cz(..) | Gate::CPhase(..) => 1,
        _ => 2,
    }
}

fn is_branching(g: &Gate) -> bool {
    g.arity() == 1
        && g.diagonal_1q().is_none()
        && !matches!(g, Gate::X(_) | Gate::Y(_) | Gate::I(_))
}

/// Rank over GF(2) of 128-bit vectors (leading-bit pivot table).
fn rank128(vs: impl Iterator<Item = u128>) -> usize {
    rank128_upto(vs, usize::MAX)
}

/// [`rank128`], stopping as soon as the rank reaches `stop`.
fn rank128_upto(vs: impl Iterator<Item = u128>, stop: usize) -> usize {
    let mut piv = [0u128; 128];
    let mut r = 0;
    for mut v in vs {
        if r >= stop {
            break;
        }
        while v != 0 {
            let b = 127 - v.leading_zeros() as usize;
            if piv[b] == 0 {
                piv[b] = v;
                r += 1;
                break;
            }
            v ^= piv[b];
        }
    }
    r
}

/// Rank over GF(2) of bit vectors (padded to a common length), stopping
/// as soon as it reaches `stop`.
fn rank_vecs<'a>(vs: impl Iterator<Item = &'a Vec<u64>>, stop: usize) -> usize {
    let vs: Vec<&Vec<u64>> = vs.collect();
    let w = vs.iter().map(|v| v.len()).max().unwrap_or(0);
    let mut basis: Vec<(usize, Vec<u64>)> = Vec::new();
    for v in vs {
        if basis.len() >= stop {
            break;
        }
        let mut v = v.clone();
        v.resize(w, 0);
        for (p, b) in &basis {
            if v[p / 64] >> (p % 64) & 1 == 1 {
                for (x, y) in v.iter_mut().zip(b) {
                    *x ^= y;
                }
            }
        }
        if let Some(p) = lead(&v) {
            basis.push((p, v));
        }
    }
    basis.len()
}

fn lead(v: &[u64]) -> Option<usize> {
    v.iter()
        .enumerate()
        .find(|(_, &x)| x != 0)
        .map(|(i, &x)| i * 64 + x.trailing_zeros() as usize)
}

/// Rigorous Schmidt-rank bounds of the evolving state (see module docs).
#[derive(Clone, Debug)]
pub struct BondBounds {
    n: usize,
    full: u64,
    /// `cross[k]`: crossing bits of the prefix cut `{0..=k} | rest`.
    cross: Vec<usize>,
    /// Pauli masks are only tracked for `n <= MAX_QUBITS`.
    pauli: bool,
    /// Stabilizer generators of the Clifford part (signs dropped).
    gens: Vec<(u64, u64)>,
    /// Every non-Clifford rotation axis so far, in the current frame.
    axes: Vec<(u64, u64)>,
    /// A basis of the span of `axes` (current frame).
    axbasis: Vec<(u64, u64)>,
    wires: Vec<Wire>,
    nvars: usize,
    saved: Option<(Vec<Wire>, usize)>,
}

#[inline]
fn pack(x: u64, z: u64, a: u64) -> u128 {
    ((x & a) as u128) | (((z & a) as u128) << 64)
}

impl BondBounds {
    pub fn new(n: usize) -> Self {
        let pauli = n <= MAX_QUBITS;
        BondBounds {
            n,
            full: if n >= 64 { u64::MAX } else { (1u64 << n) - 1 },
            cross: vec![0; n.saturating_sub(1)],
            pauli,
            gens: if pauli {
                (0..n).map(|q| (0, 1u64 << q)).collect()
            } else {
                Vec::new()
            },
            axes: Vec::new(),
            axbasis: Vec::new(),
            wires: vec![Wire::Const(Some(false)); n],
            nvars: 0,
            saved: None,
        }
    }

    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// Non-Clifford rotation axes seen so far.
    pub fn rotations(&self) -> usize {
        self.axes.len()
    }

    // ----- Pauli frame ---------------------------------------------------

    fn conj_all(&mut self, f: impl Fn(&mut (u64, u64))) {
        for p in self.gens.iter_mut() {
            f(p);
        }
        for p in self.axes.iter_mut() {
            f(p);
        }
        for p in self.axbasis.iter_mut() {
            f(p);
        }
    }

    fn cl_h(&mut self, q: usize) {
        self.conj_all(|p| {
            let (x, z) = (p.0 >> q & 1, p.1 >> q & 1);
            if x != z {
                p.0 ^= 1 << q;
                p.1 ^= 1 << q;
            }
        });
    }
    fn cl_s(&mut self, q: usize) {
        self.conj_all(|p| p.1 ^= p.0 & (1 << q));
    }
    fn cl_sx(&mut self, q: usize) {
        self.conj_all(|p| p.0 ^= p.1 & (1 << q));
    }
    fn cl_cnot(&mut self, c: usize, t: usize) {
        self.conj_all(|p| {
            p.0 ^= (p.0 >> c & 1) << t;
            p.1 ^= (p.1 >> t & 1) << c;
        });
    }
    fn cl_cz(&mut self, a: usize, b: usize) {
        self.conj_all(|p| {
            let (xa, xb) = (p.0 >> a & 1, p.0 >> b & 1);
            p.1 ^= (xb << a) | (xa << b);
        });
    }
    fn cl_swap(&mut self, a: usize, b: usize) {
        self.conj_all(|p| {
            for v in [&mut p.0, &mut p.1] {
                let (ba, bb) = (*v >> a & 1, *v >> b & 1);
                if ba != bb {
                    *v ^= (1 << a) | (1 << b);
                }
            }
        });
    }

    fn add_axis(&mut self, x: u64, z: u64) {
        self.axes.push((x, z));
        if self.axbasis.len() < 2 * self.n {
            let r0 = rank128(self.axbasis.iter().map(|&(x, z)| pack(x, z, u64::MAX)));
            let r1 = rank128(
                self.axbasis
                    .iter()
                    .chain(std::iter::once(&(x, z)))
                    .map(|&(x, z)| pack(x, z, u64::MAX)),
            );
            if r1 > r0 {
                self.axbasis.push((x, z));
            }
        }
    }

    /// One primitive (Clifford or Z rotation) in the Pauli frame.
    fn frame_primitive(&mut self, g: &Gate) {
        use Gate::*;
        match *g {
            I(_) | X(_) | Y(_) | Z(_) => {}
            H(q) => self.cl_h(q),
            S(q) | Sdg(q) => self.cl_s(q),
            Sx(q) | Sxdg(q) => self.cl_sx(q),
            Cnot(c, t) => self.cl_cnot(c, t),
            Cz(a, b) => self.cl_cz(a, b),
            Swap(a, b) => self.cl_swap(a, b),
            T(q) | Tdg(q) => self.add_axis(0, 1 << q),
            Rz(q, th) | Phase(q, th) => {
                if is_multiple_of_half_pi(th) {
                    let k = (th / FRAC_PI_2).round() as i64;
                    if k.rem_euclid(2) == 1 {
                        self.cl_s(q);
                    }
                } else {
                    self.add_axis(0, 1 << q);
                }
            }
            ref other => {
                for h in other.decompose_to_clifford_rz() {
                    debug_assert!(h != *other);
                    self.frame_primitive(&h);
                }
            }
        }
    }

    // ----- affine support ------------------------------------------------

    fn fresh_var(&mut self) -> Wire {
        let mut v = vec![0u64; self.nvars / 64 + 1];
        v[self.nvars / 64] |= 1 << (self.nvars % 64);
        self.nvars += 1;
        Wire::Affine(v)
    }

    fn xor(a: &Wire, b: &Wire) -> Wire {
        match (a, b) {
            (Wire::Const(x), Wire::Const(y)) => Wire::Const(match (x, y) {
                (Some(x), Some(y)) => Some(x ^ y),
                _ => None,
            }),
            (Wire::Const(_), x) | (x, Wire::Const(_)) => x.clone(),
            (Wire::Affine(u), Wire::Affine(v)) => {
                let (long, short) = if u.len() >= v.len() { (u, v) } else { (v, u) };
                let mut s = long.clone();
                for (x, y) in s.iter_mut().zip(short) {
                    *x ^= y;
                }
                if s.iter().all(|&x| x == 0) {
                    // equal forms: constant on every branch, value unknown
                    // (offsets are not tracked).
                    Wire::Const(None)
                } else {
                    Wire::Affine(s)
                }
            }
            _ => Wire::Opaque,
        }
    }

    /// The wire is either `t` or `t ⊕ o` (a control with an untracked
    /// constant value).
    fn either(t: &Wire, o: &Wire) -> Wire {
        match o {
            Wire::Const(_) => match t {
                Wire::Const(_) => Wire::Const(None),
                x => x.clone(),
            },
            _ => Wire::Opaque,
        }
    }

    fn flip(w: &mut Wire) {
        if let Wire::Const(Some(v)) = w {
            *v = !*v;
        }
    }

    fn affine_part(&mut self, g: &Gate) {
        match *g {
            Gate::X(q) | Gate::Y(q) => Self::flip(&mut self.wires[q]),
            Gate::Cnot(c, t) => self.wires[t] = Self::xor(&self.wires[t], &self.wires[c]),
            Gate::Swap(a, b) | Gate::ISwap(a, b) | Gate::ISwapdg(a, b) => self.wires.swap(a, b),
            Gate::Ccx(a, b, t) => {
                let cv = |w: &Wire| match w {
                    Wire::Const(v) => Some(*v),
                    _ => None,
                };
                match (cv(&self.wires[a]), cv(&self.wires[b])) {
                    (Some(Some(x)), Some(Some(y))) => {
                        if x && y {
                            Self::flip(&mut self.wires[t]);
                        }
                    }
                    (Some(Some(true)), _) => {
                        self.wires[t] = Self::xor(&self.wires[t], &self.wires[b]);
                    }
                    (_, Some(Some(true))) => {
                        self.wires[t] = Self::xor(&self.wires[t], &self.wires[a]);
                    }
                    (Some(Some(false)), _) | (_, Some(Some(false))) => {}
                    (Some(None), Some(None)) => {
                        // t or NOT t: same form, unknown offset
                        if let Wire::Const(_) = self.wires[t] {
                            self.wires[t] = Wire::Const(None);
                        }
                    }
                    (Some(None), _) => {
                        self.wires[t] = Self::either(&self.wires[t], &self.wires[b]);
                    }
                    (_, Some(None)) => {
                        self.wires[t] = Self::either(&self.wires[t], &self.wires[a]);
                    }
                    _ => self.wires[t] = Wire::Opaque,
                }
            }
            ref h if is_branching(h) => {
                let q = h.qubits()[0];
                self.wires[q] = self.fresh_var();
            }
            _ => {} // diagonal: support unchanged
        }
    }

    // ----- gate interface --------------------------------------------------

    /// Applies one gate exactly as the MPS engine applies it (a one- or
    /// two-qubit matrix, or one part of a decomposed gate).
    pub fn apply_part(&mut self, g: &Gate) {
        let qs = g.qubits();
        if qs.len() == 2 {
            let (lo, hi) = (qs[0].min(qs[1]), qs[0].max(qs[1]));
            let r = schmidt_bits(g);
            for k in lo..hi {
                self.cross[k] = (self.cross[k] + r).min((k + 1).min(self.n - 1 - k));
            }
        }
        if self.pauli {
            self.frame_primitive(g);
        }
        self.affine_part(g);
    }

    /// Before the parts of a decomposed gate (Toffoli).
    pub fn begin_composite(&mut self, _g: &Gate) {
        self.saved = Some((self.wires.clone(), self.nvars));
    }

    /// After the parts of a decomposed gate: the affine tracker treats the
    /// whole gate as one permutation (tighter than its parts).
    pub fn end_composite(&mut self, g: &Gate) {
        if let Some((w, nv)) = self.saved.take() {
            self.wires = w;
            self.nvars = nv;
            self.affine_part(g);
        }
    }

    // ----- queries -------------------------------------------------------

    fn cut_bits(&self, a: u64) -> usize {
        let k = (a & self.full).count_ones() as usize;
        k.min(self.n - k)
    }

    fn cross_bits(&self, a: u64) -> usize {
        // prefix P_k = {0..k-1}, crossing bits c(P_0) = c(P_n) = 0.
        let a = a & self.full;
        let mut best = usize::MAX;
        for k in 0..=self.n {
            let pk = if k >= 64 { u64::MAX } else { (1u64 << k) - 1 } & self.full;
            let c = if k == 0 || k == self.n {
                0
            } else {
                self.cross[k - 1]
            };
            best = best.min(c + (a ^ pk).count_ones() as usize);
        }
        best
    }

    fn stab_ent(&self, a: u64) -> usize {
        let r = rank128(self.gens.iter().map(|&(x, z)| pack(x, z, a)));
        r - (a & self.full).count_ones() as usize
    }

    fn straddle(&self, a: u64) -> usize {
        let b = !a & self.full;
        self.axes
            .iter()
            .filter(|&&(x, z)| (x | z) & a != 0 && (x | z) & b != 0)
            .count()
    }

    /// Coset bound, capped at `lim` (the computation stops there).
    fn coset_bits(&self, a: u64, lim: usize) -> usize {
        let side = |m: u64, lim: usize| {
            let k = m.count_ones() as usize;
            let r = rank128_upto(
                self.gens
                    .iter()
                    .chain(self.axbasis.iter())
                    .map(|&(x, z)| pack(x, z, m)),
                k.saturating_add(lim),
            );
            (r - k).min(lim)
        };
        let a = a & self.full;
        let sa = side(a, lim);
        side(!a & self.full, sa)
    }

    /// Affine bound, capped at `lim` (the computation stops there).
    fn affine_bits(&self, a: u64, lim: usize) -> usize {
        let side = |m: u64, lim: usize| {
            let mut opaque = 0;
            let forms: Vec<&Vec<u64>> = (0..self.n)
                .filter(|&q| m >> q & 1 == 1)
                .filter_map(|q| match &self.wires[q] {
                    Wire::Affine(v) => Some(v),
                    Wire::Opaque => {
                        opaque += 1;
                        None
                    }
                    Wire::Const(_) => None,
                })
                .collect();
            if opaque >= lim {
                return lim;
            }
            (rank_vecs(forms.into_iter(), lim - opaque) + opaque)
                .min(self.nvars)
                .min(lim)
        };
        let a = a & self.full;
        let sa = side(a, lim);
        side(!a & self.full, sa)
    }

    /// log2 bound on the Schmidt rank across `A | rest` (`a`: bit mask of
    /// the qubits in `A`) by one estimator.
    pub fn bits(&self, a: u64, est: Estimator) -> usize {
        let cut = self.cut_bits(a);
        let v = match est {
            Estimator::Cut => cut,
            Estimator::Cross => self.cross_bits(a),
            Estimator::Stab if self.pauli => self.stab_ent(a) + self.straddle(a),
            Estimator::Coset if self.pauli => self.coset_bits(a, cut),
            Estimator::Affine => self.affine_bits(a, cut),
            Estimator::Best => {
                // cheapest first; each later bound only has to beat `b`
                // and stops computing once it cannot.
                let mut b = cut.min(self.cross_bits(a));
                if self.pauli && b > 0 {
                    b = b.min(self.stab_ent(a) + self.straddle(a));
                }
                if self.pauli && b > 0 {
                    b = self.coset_bits(a, b);
                }
                if b > 0 {
                    b = self.affine_bits(a, b);
                }
                b
            }
            _ => cut,
        };
        v.min(cut)
    }
}

/// How the replay chooses the bond after each SVD.
#[derive(Clone, Copy, Debug)]
pub enum BondSource<'a> {
    /// A rigorous bound.
    Bound(Estimator),
    /// The real engine's trace (reproduces its counts exactly).
    Trace(&'a [u32]),
    /// A capped probe run's trace: below the cap take it, at the cap fall
    /// back to the `Best` bound.
    Probe(&'a [u32], u32),
    /// Like `Probe`, but at the cap extrapolate: `cap · 2^Δ`, where `Δ` is
    /// how many bits the `Best` bound of the same bipartition has grown
    /// since the probe first saturated there (an estimate, not a bound).
    ProbeExtrapolate(&'a [u32], u32),
}

/// What [`replay`] predicts.
#[derive(Clone, Debug, Default)]
pub struct ReplayCost {
    pub stats: MpsStats,
    /// Largest bond at any time.
    pub max_bond: usize,
    /// Largest bond of the final state.
    pub final_max_bond: usize,
    /// `Σ log2 χ` over the SVDs (mean = this / svd_calls).
    pub sum_log_bond: f64,
    pub secs: f64,
    /// Bond kept after every SVD (only with [`replay_traced`]).
    pub trace: Vec<u32>,
}

struct Sym<'a> {
    n: usize,
    bond: Vec<usize>,
    center: usize,
    /// site -> logical qubit
    perm: Vec<usize>,
    bounds: BondBounds,
    src: BondSource<'a>,
    step: usize,
    record: bool,
    /// ProbeExtrapolate: bound bits at the first saturation per bipartition.
    sat: std::collections::HashMap<u64, usize>,
    out: ReplayCost,
}

impl Sym<'_> {
    fn dl(&self, i: usize) -> usize {
        if i == 0 {
            1
        } else {
            self.bond[i - 1]
        }
    }
    fn dr(&self, i: usize) -> usize {
        if i + 1 == self.n {
            1
        } else {
            self.bond[i]
        }
    }
    fn move_center(&mut self, to: usize) {
        while self.center < to {
            let i = self.center;
            let (m, k, k2) = (2 * self.dl(i), self.dr(i), 2 * self.dr(i + 1));
            self.out.stats.qr_calls += 1;
            self.out.stats.qr_work += qr_work(m, k) + matmul_work(m.min(k), k, k2);
            self.bond[i] = m.min(k);
            self.center += 1;
        }
        while self.center > to {
            let i = self.center;
            let (m, k, k2) = (2 * self.dr(i), self.dl(i), 2 * self.dl(i - 1));
            self.out.stats.qr_calls += 1;
            self.out.stats.qr_work += qr_work(m, k) + matmul_work(k2, k, m.min(k));
            self.bond[i - 1] = m.min(k);
            self.center -= 1;
        }
    }
    fn mask_upto(&self, i: usize) -> u64 {
        self.perm[..=i]
            .iter()
            .fold(0u64, |m, &q| m | (1u64 << (q % 64)))
    }
    fn adjacent(&mut self, i: usize) {
        self.move_center(i);
        let (dl, dm, dr) = (self.dl(i), self.dr(i), self.dr(i + 1));
        self.out.stats.mm_work += matmul_work(2 * dl, dm, 2 * dr) + (4 * dl * dr) as f64;
        self.out.stats.svd_calls += 1;
        self.out.stats.svd_work += svd_work(2 * dl, 2 * dr);
        let lim = (2 * dl).min(2 * dr);
        let est = match self.src {
            BondSource::Trace(t) => t.get(self.step).map(|&x| x as usize).unwrap_or(lim),
            BondSource::Probe(t, cap) => match t.get(self.step) {
                Some(&x) if x < cap => x as usize,
                _ => self.bound_bond(i, Estimator::Best, lim),
            },
            BondSource::Bound(e) => self.bound_bond(i, e, lim),
            BondSource::ProbeExtrapolate(t, cap) => match t.get(self.step) {
                Some(&x) if x < cap => x as usize,
                _ => {
                    let b = self.bound_bond(i, Estimator::Best, lim);
                    let bits = if b == usize::MAX {
                        40
                    } else {
                        b.trailing_zeros() as usize
                    };
                    let mask = self.mask_upto(i);
                    let b0 = *self.sat.entry(mask).or_insert(bits);
                    let ext = (cap as usize) << bits.saturating_sub(b0).min(30);
                    ext.min(b)
                }
            },
        };
        let keep = est.clamp(1, lim);
        self.step += 1;
        self.bond[i] = keep;
        self.center = i + 1;
        self.out.max_bond = self.out.max_bond.max(keep);
        self.out.sum_log_bond += (keep as f64).log2();
        if self.record {
            self.out.trace.push(keep as u32);
        }
    }
    fn bound_bond(&self, i: usize, e: Estimator, lim: usize) -> usize {
        if lim <= 1 {
            return 1;
        }
        let b = if self.n <= MAX_QUBITS {
            self.bounds.bits(self.mask_upto(i), e)
        } else {
            // no masks: only the line cut / crossing count of the prefix
            let k = i + 1;
            let c = (k.min(self.n - k)).min(self.bounds.cross[i]);
            if matches!(e, Estimator::Cut) {
                k.min(self.n - k)
            } else {
                c
            }
        };
        if b >= 40 {
            usize::MAX
        } else {
            1usize << b
        }
    }
    fn apply_2q(&mut self, g: &Gate) {
        let qs = g.qubits();
        let (lo, hi) = (qs[0].min(qs[1]), qs[0].max(qs[1]));
        for k in (lo + 1..hi).rev() {
            self.perm.swap(k, k + 1);
            self.adjacent(k);
        }
        self.bounds.apply_part(g);
        self.adjacent(lo);
        for k in lo + 1..hi {
            self.perm.swap(k, k + 1);
            self.adjacent(k);
        }
    }
    fn apply_gate(&mut self, g: &Gate) {
        if matches!(g, Gate::I(_)) {
            return;
        }
        if g.matrix_1q().is_some() {
            let q = g.qubits()[0];
            self.out.stats.oneq_work += (self.dl(q) * self.dr(q)) as f64;
            self.bounds.apply_part(g);
        } else if g.matrix_2q().is_some() {
            self.apply_2q(g);
        } else {
            self.bounds.begin_composite(g);
            for h in g.decompose_to_clifford_rz() {
                self.apply_gate(&h);
            }
            self.bounds.end_composite(g);
        }
    }
}

/// Symbolic replay of an exact MPS run of `c` from `|0^n>` (see module
/// docs). Errors on non-unitary circuits.
pub fn replay(c: &Circuit, src: BondSource) -> Result<ReplayCost, SimError> {
    replay_impl(c, src, false)
}

/// [`replay`] that also records the bond after every SVD
/// ([`ReplayCost::trace`]); with a rigorous bound every entry upper-bounds
/// the real engine's bond at the same step.
pub fn replay_traced(c: &Circuit, src: BondSource) -> Result<ReplayCost, SimError> {
    replay_impl(c, src, true)
}

fn replay_impl(c: &Circuit, src: BondSource, record: bool) -> Result<ReplayCost, SimError> {
    let t0 = std::time::Instant::now();
    let n = c.num_qubits;
    let mut s = Sym {
        n,
        bond: vec![1; n.saturating_sub(1)],
        center: 0,
        perm: (0..n).collect(),
        bounds: BondBounds::new(n),
        src,
        step: 0,
        record,
        sat: std::collections::HashMap::new(),
        out: ReplayCost::default(),
    };
    for op in &c.ops {
        match op {
            Op::Gate(g) => s.apply_gate(g),
            _ => {
                return Err(SimError::NotSupported {
                    what: "mps_cost::replay needs a unitary circuit",
                })
            }
        }
    }
    s.out.max_bond = s.out.max_bond.max(1);
    s.out.final_max_bond = s.bond.iter().copied().max().unwrap_or(1);
    s.out.secs = t0.elapsed().as_secs_f64();
    Ok(s.out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank128_basic() {
        assert_eq!(rank128([1u128, 2, 3].into_iter()), 2);
        assert_eq!(rank128([1u128 << 127, 1].into_iter()), 2);
    }

    #[test]
    fn bell_pair_bounds() {
        let mut b = BondBounds::new(2);
        b.apply_part(&Gate::H(0));
        for e in Estimator::ALL {
            // `Cut` only knows the cut size
            let want = usize::from(e == Estimator::Cut);
            assert_eq!(b.bits(1, e), want, "{e:?}");
        }
        b.apply_part(&Gate::Cnot(0, 1));
        for e in Estimator::ALL {
            assert_eq!(b.bits(1, e), 1, "{e:?}");
        }
        b.apply_part(&Gate::Cnot(0, 1));
        assert_eq!(b.bits(1, Estimator::Stab), 0);
        assert_eq!(b.bits(1, Estimator::Coset), 0);
        assert_eq!(b.bits(1, Estimator::Affine), 0);
        assert_eq!(b.bits(1, Estimator::Cross), 1); // crossing count can't undo
    }
}
