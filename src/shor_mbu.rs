//! Measurement-based uncomputation (MBU) in the windowed Shor oracle
//! (exp/mbu-shor, `research/mbu-shor.md`).
//!
//! The superoptimised windowed oracle ([`crate::shor_superopt`]) is a pure
//! X/CNOT/CCX permutation. Fault-tolerant constructions (Gidney 2017,
//! Babbush et al. 2018, Gidney 2019, Gidney–Ekerå 2019) uncompute
//! temporaries by **measuring them in the X basis** instead of running the
//! computation backwards:
//!
//! * **temporary logical-AND** (Gidney, arXiv:1709.06648): `t = a ∧ b` is
//!   computed into a clean `t` (one Toffoli) and later uncomputed by
//!   measuring `t` in the X basis: outcome `m` is uniform, the branch picks up
//!   `(−1)^{m·(a∧b)}`, and a classically controlled `CZ(a, b)` (if `m = 1`)
//!   removes it. 0 Toffolis for the uncompute.
//! * **measurement-based unlookup** (Berry et al. 2019, Gidney 2019 windowed
//!   arithmetic): the lookup output register `out = ctrl·T[v]` is measured in
//!   the X basis (outcomes `m ∈ F₂ⁿ`), which leaves the phase
//!   `(−1)^{ctrl·(m·T[v])}` on every branch; a *phase lookup* of the 1-bit
//!   table `g(v) = m·T[v] mod 2` cancels it. Here the phase lookup splits the
//!   `w` address bits into `w − k` high bits (unary iteration, temporary ANDs)
//!   and `k` low bits whose monomials are precomputed (the algebraic normal
//!   form of `g` restricted to each high leaf), so it costs
//!   `(2^{w−k} − 1) + (2^k − k − 1)` Toffolis (4 for `w = 4`) instead of the
//!   `2^w − 1` of a second lookup.
//! * **Gidney adders** (`MbuOpts::adders`): every Cuccaro ripple adder and the
//!   comparator of the modular adder are replaced by the temporary-AND
//!   adder: `n` Toffolis per addition instead of `2n`, at the price of
//!   `n − 1` carry ancillas.
//!
//! # Exactness in the branch-state engine
//!
//! An X-basis measurement of a qubit `q` that is a *deterministic function of
//! the other qubits in every branch* maps `Σ_x α_x |x⟩|f(x)⟩` to
//! `Σ_x α_x (−1)^{m f(x)} |x⟩|m⟩` with `P(m) = 1/2` exactly; after resetting
//! `q` the branch set is unchanged (no two branches merge) and every branch
//! carries a sign. So a measured block is still a map on basis branches plus
//! a per-branch sign bit. The circuit is built as logical operations
//! ([`LOp`], with compute/uncompute pairs and lookups as units so that the
//! inverse multiplier is the exact inverse: reverse the list and swap
//! `And ↔ UnAnd`, `Lookup ↔ Unlookup`), then **resolved** for sampled
//! outcomes into [`MbuOp`]s: gates from {X, CNOT, CCX, Z, CZ} and
//! `MeasX(q, m)` (= X-basis measurement of `q` with outcome `m`, then reset
//! to `|0⟩`). The bit-sliced engine ([`crate::shor::sliced`]) evaluates Z /
//! CZ / `MeasX` on a sign word (`sign ^= w[q]`, `sign ^= w[a] & w[b]`,
//! `w[q] = 0`) and **asserts** that at the end of every block every branch
//! has sign `+1`, as it already asserts that every ancilla is clean and the
//! block is injective on the support (which is what fails if a measured
//! qubit was *not* a deterministic function: two branches would collide).
//! The gate-by-gate reference path instead applies a real `H`, checks
//! `P(m) = 1/2`, projects and renormalises.

use crate::circuit::Circuit;
use crate::gate::Gate;
use crate::shor_superopt::{modadd_block, FanoutPlan, Opts};
use crate::shor_window::WindowLayout;
use std::rc::Rc;

/// A resolved operation: a gate from {X, CNOT, CCX, SWAP, Z, CZ} or an
/// X-basis measurement with its (sampled) outcome followed by a reset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MbuOp {
    G(Gate),
    /// X-basis measurement of the qubit with the given outcome, then reset
    /// to `|0⟩` (an `X` if the outcome is 1).
    MeasX(usize, bool),
}

/// Which measurement-based constructions to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MbuOpts {
    /// Lookups uncompute their address-tree ANDs by measurement
    /// (`2^w − 1` Toffolis instead of `2(2^w − 1)`).
    pub lookup_and: bool,
    /// Measurement-based unlookup (X-measure the output, phase fix-up).
    pub unlookup: bool,
    /// Gidney temporary-AND adders and comparator in the modular adder
    /// (`4n` Toffolis instead of `8n`; `n − 1` extra carry qubits).
    pub adders: bool,
    /// Uncompute the modular adder's flag `t = [b_new < L]` by X-basis
    /// measurement; the fix-up (outcome 1, probability 1/2) is a *phase*
    /// comparator `(−1)^{[b < L]}` (Gidney 2025's trick for the flag). The
    /// comparator then runs on half of the shots only.
    pub flag: bool,
}

impl MbuOpts {
    /// Everything on.
    pub const ALL: MbuOpts = MbuOpts {
        lookup_and: true,
        unlookup: true,
        adders: true,
        flag: true,
    };
    /// Lookups and unlookups only: same qubits as the windowed oracle.
    pub const LOOKUPS: MbuOpts = MbuOpts {
        lookup_and: true,
        unlookup: true,
        adders: false,
        flag: true,
    };
    /// Nothing measured: the `windowed-opt` oracle (as logical ops).
    pub const NONE: MbuOpts = MbuOpts {
        lookup_and: false,
        unlookup: false,
        adders: false,
        flag: false,
    };
}

/// Windowed layout plus the carry ancillas of the Gidney adders.
#[derive(Clone, Debug)]
pub struct MbuLayout {
    pub win: WindowLayout,
    /// `n − 1` carry ancillas (empty without `adders`).
    pub cy: Vec<usize>,
}

impl MbuLayout {
    pub fn new(n: usize, w: usize, o: &MbuOpts) -> Self {
        let win = WindowLayout::new(n, w);
        let base = win.num_qubits();
        let cy = if o.adders {
            (base..base + n.saturating_sub(1)).collect()
        } else {
            Vec::new()
        };
        Self { win, cy }
    }
    pub fn num_qubits(&self) -> usize {
        self.win.num_qubits() + self.cy.len()
    }
}

/// A table lookup `out ^= ctrl·T[addr]` (by unary iteration).
#[derive(Clone, Debug)]
pub struct LookupSpec {
    pub ctrl: usize,
    /// Address, LSB first.
    pub addr: Vec<usize>,
    /// Clean AND ancillas (at least `addr.len()`).
    pub and: Vec<usize>,
    pub out: Vec<usize>,
    pub table: Vec<u64>,
    /// Uncompute the address-tree ANDs by measurement.
    pub mbu_and: bool,
    /// As an unlookup: measure `out` in the X basis and fix the phase.
    pub meas_unlookup: bool,
}

/// A flag qubit `t` that some reversible `compute` sequence XORs a value
/// `c` into (from clean), and a `fix` sequence applying the phase
/// `(−1)^c` (used only when the X-basis measurement of `t` gives 1).
#[derive(Clone, Debug)]
pub struct FlagSpec {
    pub t: usize,
    pub compute: Vec<LOp>,
    pub fix: Vec<LOp>,
}

/// A logical operation (before measurement outcomes are sampled).
#[derive(Clone, Debug)]
pub enum LOp {
    /// X / CNOT / CCX (self-inverse).
    G(Gate),
    /// `t = a ∧ b` into a clean `t` (one Toffoli).
    And(usize, usize, usize),
    /// Uncompute `t = a ∧ b` (by measurement: X-measure `t`, `CZ(a, b)` if 1).
    UnAnd(usize, usize, usize),
    /// `out ^= ctrl·T[addr]` with `out` clean before.
    Lookup(Rc<LookupSpec>),
    /// `out ^= ctrl·T[addr]` with `out = ctrl·T[addr]` before (so after it
    /// `out` is clean): measurement-based if `meas_unlookup`.
    Unlookup(Rc<LookupSpec>),
    /// `t ^= c` on a clean `t` (runs `compute`).
    FlagCompute(Rc<FlagSpec>),
    /// Uncompute `t = c` by measurement: X-measure `t`, `fix` if 1.
    FlagUncompute(Rc<FlagSpec>),
}

/// The exact inverse of a logical op sequence.
pub fn inverse(ops: &[LOp]) -> Vec<LOp> {
    ops.iter()
        .rev()
        .map(|op| match op {
            LOp::G(g) => {
                assert!(matches!(g, Gate::X(_) | Gate::Cnot(..) | Gate::Ccx(..)));
                LOp::G(*g)
            }
            LOp::And(a, b, t) => LOp::UnAnd(*a, *b, *t),
            LOp::UnAnd(a, b, t) => LOp::And(*a, *b, *t),
            LOp::Lookup(s) => LOp::Unlookup(s.clone()),
            LOp::Unlookup(s) => LOp::Lookup(s.clone()),
            LOp::FlagCompute(f) => LOp::FlagUncompute(f.clone()),
            LOp::FlagUncompute(f) => LOp::FlagCompute(f.clone()),
        })
        .collect()
}

/// Unary iteration over the address tree of `plan` (MSB first), calling
/// `emit(ops, flag, mask)` at every node with a non-empty mask while the
/// node's flag (`ctrl ∧ address prefix`) is live.
#[allow(clippy::too_many_arguments)]
fn unary(
    ops: &mut Vec<LOp>,
    plan: &FanoutPlan,
    d: usize,
    h: usize,
    p: usize,
    addr: &[usize],
    and: &[usize],
    mbu: bool,
    emit: &mut dyn FnMut(&mut Vec<LOp>, usize, u64),
) {
    let m = plan.use_[d][h];
    if m != 0 {
        emit(ops, p, m);
    }
    if d == plan.w {
        return;
    }
    let (left, right) = (plan.nonempty(d + 1, 2 * h), plan.nonempty(d + 1, 2 * h + 1));
    if !left && !right {
        return;
    }
    let bit = addr[plan.w - 1 - d];
    let f = and[d];
    ops.push(LOp::And(p, bit, f));
    if right {
        unary(ops, plan, d + 1, 2 * h + 1, f, addr, and, mbu, emit);
    }
    if left {
        ops.push(LOp::G(Gate::Cnot(p, f)));
        unary(ops, plan, d + 1, 2 * h, f, addr, and, mbu, emit);
        ops.push(LOp::G(Gate::Cnot(p, f)));
    }
    ops.push(if mbu {
        LOp::UnAnd(p, bit, f)
    } else {
        LOp::G(Gate::Ccx(p, bit, f))
    });
}

/// The lowered lookup (unary iteration, optimal fan-out).
pub fn lookup_ops(s: &LookupSpec) -> Vec<LOp> {
    let plan = FanoutPlan::optimal(&s.table, s.out.len());
    let mut ops = Vec::new();
    let out = s.out.clone();
    unary(
        &mut ops,
        &plan,
        0,
        0,
        s.ctrl,
        &s.addr,
        &s.and,
        s.mbu_and,
        &mut |ops, p, m| {
            for (i, &q) in out.iter().enumerate() {
                if (m >> i) & 1 == 1 {
                    ops.push(LOp::G(Gate::Cnot(p, q)));
                }
            }
        },
    );
    ops
}

/// Toffolis (`And` and `CCX`) of a logical op list (lookups not expanded).
fn and_count(ops: &[LOp]) -> usize {
    ops.iter()
        .filter(|o| matches!(o, LOp::And(..) | LOp::G(Gate::Ccx(..))))
        .count()
}

/// Phase lookup `|v⟩ → (−1)^{ctrl·g(v)} |v⟩` with the `k` low address bits
/// handled by precomputed monomials (algebraic normal form per high leaf)
/// and the `w − k` high bits by unary iteration. `scratch` must hold at
/// least `(w − k) + (2^k − k − 1)` clean qubits; all return clean.
pub fn phase_table_k(ctrl: usize, addr: &[usize], g: &[bool], scratch: &[usize], k: usize) -> Vec<LOp> {
    let w = addr.len();
    assert_eq!(g.len(), 1 << w);
    assert!(k <= w && k <= 6);
    let hw = w - k;
    let mut anf = vec![0u64; 1 << hw];
    for (h, a_h) in anf.iter_mut().enumerate() {
        let mut a: Vec<u8> = (0..1usize << k).map(|l| u8::from(g[(h << k) | l])).collect();
        for i in 0..k {
            for s in 0..1usize << k {
                if (s >> i) & 1 == 1 {
                    a[s] ^= a[s ^ (1 << i)];
                }
            }
        }
        for (s, &v) in a.iter().enumerate() {
            if v == 1 {
                *a_h |= 1 << s;
            }
        }
    }
    let plan = FanoutPlan::optimal(&anf, 1 << k);
    let used = plan.use_.iter().flatten().fold(0u64, |a, &b| a | b);
    // monomials of degree >= 2 that are needed, closed under "drop top bit"
    let mut need = vec![false; 1 << k];
    for s in (0..1usize << k).rev() {
        if s.count_ones() >= 2 && ((used >> s) & 1 == 1 || need[s]) {
            need[s] = true;
            let top = usize::BITS - 1 - s.leading_zeros();
            let s2 = s & !(1 << top);
            if s2.count_ones() >= 2 {
                need[s2] = true;
            }
        }
    }
    let mut ops = Vec::new();
    let (tree, pool) = scratch.split_at(hw);
    let mut mon = vec![usize::MAX; 1 << k];
    let mut ands = Vec::new();
    let mut next = 0;
    for s in 0..1usize << k {
        if !need[s] {
            continue;
        }
        let top = (usize::BITS - 1 - s.leading_zeros()) as usize;
        let s2 = s & !(1 << top);
        let src = if s2.count_ones() == 1 {
            addr[s2.trailing_zeros() as usize]
        } else {
            mon[s2]
        };
        let anc = pool[next];
        next += 1;
        mon[s] = anc;
        ops.push(LOp::And(src, addr[top], anc));
        ands.push((src, addr[top], anc));
    }
    unary(
        &mut ops,
        &plan,
        0,
        0,
        ctrl,
        &addr[k..],
        tree,
        true,
        &mut |ops, p, m| {
            for s in 0..1usize << k {
                if (m >> s) & 1 == 0 {
                    continue;
                }
                ops.push(LOp::G(match s.count_ones() {
                    0 => Gate::Z(p),
                    1 => Gate::Cz(p, addr[s.trailing_zeros() as usize]),
                    _ => Gate::Cz(p, mon[s]),
                }));
            }
        },
    );
    for &(a, b, t) in ands.iter().rev() {
        ops.push(LOp::UnAnd(a, b, t));
    }
    ops
}

/// [`phase_table_k`] with the split `k` that minimises Toffolis, then ops.
pub fn phase_table(ctrl: usize, addr: &[usize], g: &[bool], scratch: &[usize]) -> Vec<LOp> {
    let w = addr.len();
    let mut best: Option<(usize, usize, Vec<LOp>)> = None;
    for k in 0..=w.min(6) {
        if (w - k) + (1usize << k) - k - 1 > scratch.len() {
            continue;
        }
        let ops = phase_table_k(ctrl, addr, g, scratch, k);
        let key = (and_count(&ops), ops.len());
        if best.as_ref().map_or(true, |b| key < (b.0, b.1)) {
            best = Some((key.0, key.1, ops));
        }
    }
    best.expect("not enough scratch qubits for the phase fix-up").2
}

/// Resolves logical ops into [`MbuOp`]s, drawing every measurement outcome
/// from `rng`.
pub fn resolve(ops: &[LOp], rng: &mut dyn FnMut() -> bool, out: &mut Vec<MbuOp>) {
    for op in ops {
        match op {
            LOp::G(g) => out.push(MbuOp::G(*g)),
            LOp::And(a, b, t) => out.push(MbuOp::G(Gate::Ccx(*a, *b, *t))),
            LOp::UnAnd(a, b, t) => {
                let m = rng();
                out.push(MbuOp::MeasX(*t, m));
                if m {
                    out.push(MbuOp::G(Gate::Cz(*a, *b)));
                }
            }
            LOp::Lookup(s) => resolve(&lookup_ops(s), rng, out),
            LOp::FlagCompute(f) => resolve(&f.compute, rng, out),
            LOp::FlagUncompute(f) => {
                let m = rng();
                out.push(MbuOp::MeasX(f.t, m));
                if m {
                    resolve(&f.fix, rng, out);
                }
            }
            LOp::Unlookup(s) => {
                if !s.meas_unlookup {
                    resolve(&lookup_ops(s), rng, out);
                    continue;
                }
                let mut mask = 0u64;
                for (j, &q) in s.out.iter().enumerate() {
                    let m = rng();
                    out.push(MbuOp::MeasX(q, m));
                    if m {
                        mask |= 1 << j;
                    }
                }
                let g: Vec<bool> = s
                    .table
                    .iter()
                    .map(|&t| (t & mask).count_ones() & 1 == 1)
                    .collect();
                if g.iter().any(|&b| b) {
                    let scratch: Vec<usize> = s.and.iter().chain(&s.out).copied().collect();
                    resolve(&phase_table(s.ctrl, &s.addr, &g, &scratch), rng, out);
                }
            }
        }
    }
}

/// Gidney's temporary-AND adder: `b += a (mod 2^{n+1})` for `n`-bit `a`
/// and `(n+1)`-bit `b`, `cy` (`n − 1` clean carries). `n` Toffolis.
pub fn add_g(ops: &mut Vec<LOp>, a: &[usize], b: &[usize], cy: &[usize]) {
    let n = a.len();
    assert_eq!(b.len(), n + 1);
    assert!(n >= 1 && cy.len() + 1 >= n);
    let g = |ops: &mut Vec<LOp>, x: Gate| ops.push(LOp::G(x));
    if n == 1 {
        g(ops, Gate::Ccx(a[0], b[0], b[1]));
        g(ops, Gate::Cnot(a[0], b[0]));
        return;
    }
    ops.push(LOp::And(a[0], b[0], cy[0]));
    for i in 1..n - 1 {
        let c = cy[i - 1];
        g(ops, Gate::Cnot(c, a[i]));
        g(ops, Gate::Cnot(c, b[i]));
        ops.push(LOp::And(a[i], b[i], cy[i]));
        g(ops, Gate::Cnot(c, cy[i]));
    }
    let (i, c) = (n - 1, cy[n - 2]);
    g(ops, Gate::Cnot(c, a[i]));
    g(ops, Gate::Cnot(c, b[i]));
    g(ops, Gate::Ccx(a[i], b[i], b[n]));
    g(ops, Gate::Cnot(c, b[n]));
    g(ops, Gate::Cnot(c, a[i]));
    g(ops, Gate::Cnot(a[i], b[i]));
    for i in (1..n - 1).rev() {
        let c = cy[i - 1];
        g(ops, Gate::Cnot(c, cy[i]));
        ops.push(LOp::UnAnd(a[i], b[i], cy[i]));
        g(ops, Gate::Cnot(c, a[i]));
        g(ops, Gate::Cnot(a[i], b[i]));
    }
    ops.push(LOp::UnAnd(a[0], b[0], cy[0]));
    g(ops, Gate::Cnot(a[0], b[0]));
}

/// `b -= a (mod 2^{n+1})`: the exact inverse of [`add_g`].
pub fn sub_g(ops: &mut Vec<LOp>, a: &[usize], b: &[usize], cy: &[usize]) {
    let mut f = Vec::new();
    add_g(&mut f, a, b, cy);
    ops.extend(inverse(&f));
}

/// `t ^= [b < a]` for `n`-bit `a`, `b`: the carry-out of `a + ¬b`, computed
/// with temporary ANDs and uncomputed by measurement. `n` Toffolis.
pub fn cmp_lt_g(ops: &mut Vec<LOp>, a: &[usize], b: &[usize], cy: &[usize], t: usize) {
    let n = a.len();
    assert_eq!(b.len(), n);
    let g = |ops: &mut Vec<LOp>, x: Gate| ops.push(LOp::G(x));
    for &q in b {
        g(ops, Gate::X(q));
    }
    if n == 1 {
        g(ops, Gate::Ccx(a[0], b[0], t));
    } else {
        ops.push(LOp::And(a[0], b[0], cy[0]));
        for i in 1..n - 1 {
            let c = cy[i - 1];
            g(ops, Gate::Cnot(c, a[i]));
            g(ops, Gate::Cnot(c, b[i]));
            ops.push(LOp::And(a[i], b[i], cy[i]));
            g(ops, Gate::Cnot(c, cy[i]));
        }
        let (i, c) = (n - 1, cy[n - 2]);
        g(ops, Gate::Cnot(c, a[i]));
        g(ops, Gate::Cnot(c, b[i]));
        g(ops, Gate::Ccx(a[i], b[i], t));
        g(ops, Gate::Cnot(c, t));
        g(ops, Gate::Cnot(c, b[i]));
        g(ops, Gate::Cnot(c, a[i]));
        for i in (1..n - 1).rev() {
            let c = cy[i - 1];
            g(ops, Gate::Cnot(c, cy[i]));
            ops.push(LOp::UnAnd(a[i], b[i], cy[i]));
            g(ops, Gate::Cnot(c, b[i]));
            g(ops, Gate::Cnot(c, a[i]));
        }
        ops.push(LOp::UnAnd(a[0], b[0], cy[0]));
    }
    for &q in b {
        g(ops, Gate::X(q));
    }
}

/// The phase `(−1)^{[b < a]}`: the carry chain of `a + ¬b` with temporary
/// ANDs, the top carry kicked back as `CZ(a', b') · Z(c)` (since
/// `MAJ(a, b, c) = (a ⊕ c)(b ⊕ c) ⊕ c`), the chain uncomputed by
/// measurement. `n − 1` Toffolis.
pub fn phase_lt_g(ops: &mut Vec<LOp>, a: &[usize], b: &[usize], cy: &[usize]) {
    let n = a.len();
    assert_eq!(b.len(), n);
    let g = |ops: &mut Vec<LOp>, x: Gate| ops.push(LOp::G(x));
    for &q in b {
        g(ops, Gate::X(q));
    }
    if n == 1 {
        g(ops, Gate::Cz(a[0], b[0]));
    } else {
        ops.push(LOp::And(a[0], b[0], cy[0]));
        for i in 1..n - 1 {
            let c = cy[i - 1];
            g(ops, Gate::Cnot(c, a[i]));
            g(ops, Gate::Cnot(c, b[i]));
            ops.push(LOp::And(a[i], b[i], cy[i]));
            g(ops, Gate::Cnot(c, cy[i]));
        }
        let (i, c) = (n - 1, cy[n - 2]);
        g(ops, Gate::Cnot(c, a[i]));
        g(ops, Gate::Cnot(c, b[i]));
        g(ops, Gate::Cz(a[i], b[i]));
        g(ops, Gate::Z(c));
        g(ops, Gate::Cnot(c, b[i]));
        g(ops, Gate::Cnot(c, a[i]));
        for i in (1..n - 1).rev() {
            let c = cy[i - 1];
            g(ops, Gate::Cnot(c, cy[i]));
            ops.push(LOp::UnAnd(a[i], b[i], cy[i]));
            g(ops, Gate::Cnot(c, b[i]));
            g(ops, Gate::Cnot(c, a[i]));
        }
        ops.push(LOp::UnAnd(a[0], b[0], cy[0]));
    }
    for &q in b {
        g(ops, Gate::X(q));
    }
}

/// `b → (b + L) mod N` for `b, L < N` with Gidney adders (`4n` Toffolis):
/// the structure of [`crate::shor_superopt::add_mod_reg`] (comparator flag
/// uncompute, `K = t·N` flip).
pub fn add_mod_g(ops: &mut Vec<LOp>, lay: &MbuLayout, n_mod: u64, flag: bool) {
    let w = &lay.win;
    let (l, b, k, t, cy) = (&w.l, &w.b, &w.k, w.t, &lay.cy);
    let n = w.n;
    let g = |ops: &mut Vec<LOp>, x: Gate| ops.push(LOp::G(x));
    let set: Vec<usize> = (0..n).filter(|&j| (n_mod >> j) & 1 == 1).map(|j| k[j]).collect();
    add_g(ops, l, b, cy);
    for &q in &set {
        g(ops, Gate::X(q));
    }
    sub_g(ops, k, b, cy);
    g(ops, Gate::Cnot(b[n], t));
    g(ops, Gate::X(t));
    for &q in &set {
        g(ops, Gate::Cnot(t, q));
    }
    g(ops, Gate::X(t));
    add_g(ops, k, b, cy);
    for &q in &set {
        g(ops, Gate::Cnot(t, q));
    }
    if flag {
        // t = [b_new < L] after the X; uncomputed by measurement
        g(ops, Gate::X(t));
        let mut compute = Vec::new();
        cmp_lt_g(&mut compute, l, &b[..n], cy, t);
        let mut fix = Vec::new();
        phase_lt_g(&mut fix, l, &b[..n], cy);
        ops.push(LOp::FlagUncompute(Rc::new(FlagSpec { t, compute, fix })));
    } else {
        cmp_lt_g(ops, l, &b[..n], cy, t);
        g(ops, Gate::X(t));
    }
}

/// The Cuccaro modular adder of [`crate::shor_superopt::add_mod_reg`] (all
/// options) with its flag uncomputed by measurement: the reversible part
/// (up to `X(t)`, `t = [b_new < L]`) gets the peephole and SAT-rule passes;
/// the comparator only runs as the `compute` of the flag (inverse
/// multiplier) and, as a phase comparator (`Z` on the borrow instead of the
/// `CNOT` into `t`), as the fix-up.
pub fn add_mod_cuccaro_flag(ops: &mut Vec<LOp>, lay: &MbuLayout, n_mod: u64) {
    use crate::shor_ripple::{cuccaro_add, cuccaro_sub, load_constant};
    use crate::shor_superopt::{compare_lt, reversible_peephole, sat_peephole_init};
    let w = &lay.win;
    let (l, b, k, c0, t) = (&w.l, &w.b, &w.k, w.c0, w.t);
    let n = w.n;
    let bn = b[n];
    let nq = w.num_qubits();
    let mut c = Circuit::new(nq);
    cuccaro_add(&mut c, l, b, c0);
    load_constant(&mut c, k, n_mod, &[]);
    cuccaro_sub(&mut c, k, b, c0);
    c.cnot(bn, t);
    c.x(t);
    load_constant(&mut c, k, n_mod, &[t]);
    c.x(t);
    cuccaro_add(&mut c, k, b, c0);
    load_constant(&mut c, k, n_mod, &[t]);
    c.x(t);
    let mut init = vec![None; nq];
    for &q in k.iter().chain([&c0, &t, &bn]) {
        init[q] = Some(false);
    }
    let c = reversible_peephole(&sat_peephole_init(&reversible_peephole(&c), &init));
    ops.extend(c.gates().map(|g| LOp::G(*g)));
    let mut cc = Circuit::new(nq);
    compare_lt(&mut cc, l, &b[..n], c0, t);
    let mut init = vec![None; nq];
    init[c0] = Some(false);
    init[t] = Some(false);
    let cc = reversible_peephole(&sat_peephole_init(&reversible_peephole(&cc), &init));
    let compute: Vec<LOp> = cc.gates().map(|g| LOp::G(*g)).collect();
    // phase comparator: compare_lt with Z(borrow) for CNOT(borrow, t)
    let mut pc = Circuit::new(nq);
    compare_lt(&mut pc, l, &b[..n], c0, t);
    let fix: Vec<LOp> = pc
        .gates()
        .map(|g| match *g {
            Gate::Cnot(a, tt) if tt == t => LOp::G(Gate::Z(a)),
            g => LOp::G(g),
        })
        .collect();
    ops.push(LOp::FlagUncompute(Rc::new(FlagSpec { t, compute, fix })));
}

/// The modular-adder block as logical ops.
pub fn modadd_ops(lay: &MbuLayout, n_mod: u64, o: &MbuOpts) -> Vec<LOp> {
    let mut ops = Vec::new();
    if o.adders {
        add_mod_g(&mut ops, lay, n_mod, o.flag);
    } else if o.flag {
        add_mod_cuccaro_flag(&mut ops, lay, n_mod);
    } else {
        let c: Circuit = modadd_block(&lay.win, n_mod, &Opts::ALL);
        ops.extend(c.gates().map(|g| LOp::G(*g)));
    }
    ops
}

/// Controlled windowed multiply-add `|c⟩|x⟩|b=0⟩ → |c⟩|x⟩|c·a·x mod N⟩`.
pub fn cmult(lay: &MbuLayout, a: u64, n_mod: u64, o: &MbuOpts, modadd: &[LOp]) -> Vec<LOp> {
    let win = &lay.win;
    let n = win.n;
    let mut ops = Vec::new();
    let mut start = 0;
    let mut base = a % n_mod;
    while start < n {
        let w = win.w.min(n - start);
        let table: Vec<u64> = (0..1u64 << w)
            .map(|v| (u128::from(v) * u128::from(base) % u128::from(n_mod)) as u64)
            .collect();
        let out = if start == 0 {
            win.b[..n].to_vec()
        } else {
            win.l.clone()
        };
        let spec = Rc::new(LookupSpec {
            ctrl: win.ctrl,
            addr: win.x[start..start + w].to_vec(),
            and: win.and.clone(),
            out,
            table,
            mbu_and: o.lookup_and,
            meas_unlookup: o.unlookup,
        });
        ops.push(LOp::Lookup(spec.clone()));
        if start > 0 {
            ops.extend(modadd.iter().cloned());
            ops.push(LOp::Unlookup(spec));
        }
        for _ in 0..w {
            base = (u128::from(base) * 2 % u128::from(n_mod)) as u64;
        }
        start += w;
    }
    ops
}

/// Controlled `U_a` as logical ops: multiply-add by `a`, controlled swap,
/// the exact inverse of the multiply-add by `a⁻¹`.
pub fn controlled_ua_ops(lay: &MbuLayout, a: u64, n_mod: u64, o: &MbuOpts) -> Vec<LOp> {
    let win = &lay.win;
    let inv = crate::shor::mod_inverse(a, n_mod);
    let modadd = modadd_ops(lay, n_mod, o);
    let mut ops = cmult(lay, a, n_mod, o, &modadd);
    for i in 0..win.n {
        let (x, b) = (win.x[i], win.b[i]);
        ops.push(LOp::G(Gate::Cnot(b, x)));
        ops.push(LOp::G(Gate::Ccx(win.ctrl, x, b)));
        ops.push(LOp::G(Gate::Cnot(b, x)));
    }
    ops.extend(inverse(&cmult(lay, inv, n_mod, o, &modadd)));
    ops
}

/// SplitMix64 outcome stream.
pub struct Outcomes {
    s: u64,
    mode: u8,
}

impl Outcomes {
    /// `mode`: 0 = uniform random, 1 = all 0, 2 = all 1.
    pub fn new(seed: u64, mode: u8) -> Self {
        Self { s: seed, mode }
    }
    /// From the environment: `QSIM_MBU_OUTCOMES` = `random` (default),
    /// `zero` or `one`; `QSIM_MBU_SEED` (default 0) is mixed into `seed`.
    pub fn from_env(seed: u64) -> Self {
        let mode = match std::env::var("QSIM_MBU_OUTCOMES").as_deref() {
            Ok("zero") => 1,
            Ok("one") => 2,
            _ => 0,
        };
        let salt: u64 = std::env::var("QSIM_MBU_SEED")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        Self::new(seed ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15), mode)
    }
    pub fn next_bit(&mut self) -> bool {
        match self.mode {
            1 => false,
            2 => true,
            _ => {
                self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = self.s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                (z ^ (z >> 31)) & 1 == 1
            }
        }
    }
}

/// The resolved controlled-`U_a` block for one outcome stream.
pub fn controlled_ua(lay: &MbuLayout, a: u64, n_mod: u64, o: &MbuOpts, outc: &mut Outcomes) -> Vec<MbuOp> {
    let ops = controlled_ua_ops(lay, a, n_mod, o);
    let mut out = Vec::with_capacity(ops.len() * 2);
    resolve(&ops, &mut || outc.next_bit(), &mut out);
    out
}

/// Gate counts of a resolved block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MbuCounts {
    /// Every op: gates, fix-up Z/CZ and measurements.
    pub total: usize,
    pub toffoli: usize,
    pub cnot: usize,
    pub x: usize,
    /// Classically controlled phase fix-ups (Z and CZ).
    pub fixup: usize,
    pub meas: usize,
}

impl MbuCounts {
    pub fn of(ops: &[MbuOp]) -> Self {
        let mut c = MbuCounts::default();
        for op in ops {
            c.total += 1;
            match op {
                MbuOp::MeasX(..) => c.meas += 1,
                MbuOp::G(Gate::Ccx(..)) => c.toffoli += 1,
                MbuOp::G(Gate::Cnot(..)) => c.cnot += 1,
                MbuOp::G(Gate::X(_)) => c.x += 1,
                MbuOp::G(Gate::Z(_) | Gate::Cz(..)) => c.fixup += 1,
                MbuOp::G(g) => panic!("unexpected gate {g:?}"),
            }
        }
        c
    }
    pub fn add(&mut self, o: &MbuCounts) {
        self.total += o.total;
        self.toffoli += o.toffoli;
        self.cnot += o.cnot;
        self.x += o.x;
        self.fixup += o.fixup;
        self.meas += o.meas;
    }
}

/// Evaluates a resolved block on one basis key (`u128`, qubit `q` = bit
/// `q`), returning the output key and the accumulated sign bit. Panics if a
/// measured qubit is not reset to 0 by the op itself (it always is).
pub fn eval_on_key(ops: &[MbuOp], mut k: u128) -> (u128, bool) {
    let bit = |k: u128, q: usize| (k >> q) & 1 == 1;
    let mut sign = false;
    for op in ops {
        match *op {
            MbuOp::G(Gate::X(q)) => k ^= 1 << q,
            MbuOp::G(Gate::Cnot(c, t)) => {
                if bit(k, c) {
                    k ^= 1 << t
                }
            }
            MbuOp::G(Gate::Ccx(a, b, t)) => {
                if bit(k, a) && bit(k, b) {
                    k ^= 1 << t
                }
            }
            MbuOp::G(Gate::Swap(a, b)) => {
                if bit(k, a) != bit(k, b) {
                    k ^= (1 << a) | (1 << b)
                }
            }
            MbuOp::G(Gate::Z(q)) => sign ^= bit(k, q),
            MbuOp::G(Gate::Cz(a, b)) => sign ^= bit(k, a) && bit(k, b),
            MbuOp::MeasX(q, m) => {
                sign ^= m && bit(k, q);
                k &= !(1 << q);
            }
            MbuOp::G(g) => panic!("unexpected gate {g:?}"),
        }
    }
    (k, sign)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::SparseState;
    use num_complex::Complex64;

    fn mulmod(a: u64, b: u64, n: u64) -> u64 {
        (u128::from(a) * u128::from(b) % u128::from(n)) as u64
    }

    #[test]
    fn gidney_adder_and_comparator_exhaustive() {
        for n in 1..=5usize {
            // a: 0..n, b: n..2n+1, cy: 2n+1..3n, t: 3n
            let a: Vec<usize> = (0..n).collect();
            let b: Vec<usize> = (n..2 * n + 1).collect();
            let cy: Vec<usize> = (2 * n + 1..3 * n).collect();
            let t = 3 * n;
            for mode in 0..3u8 {
                let mut oc = Outcomes::new(n as u64 * 7 + 1, mode);
                let mut add = Vec::new();
                add_g(&mut add, &a, &b, &cy);
                let mut sub = Vec::new();
                sub_g(&mut sub, &a, &b, &cy);
                let mut cmp = Vec::new();
                cmp_lt_g(&mut cmp, &a, &b[..n], &cy, t);
                assert_eq!(and_count(&add), n);
                assert_eq!(and_count(&cmp), n);
                let mut ra = Vec::new();
                resolve(&add, &mut || oc.next_bit(), &mut ra);
                let mut rs = Vec::new();
                resolve(&sub, &mut || oc.next_bit(), &mut rs);
                let mut rc = Vec::new();
                resolve(&cmp, &mut || oc.next_bit(), &mut rc);
                let mut ph = Vec::new();
                phase_lt_g(&mut ph, &a, &b[..n], &cy);
                assert_eq!(and_count(&ph), n - 1);
                let mut rp = Vec::new();
                resolve(&ph, &mut || oc.next_bit(), &mut rp);
                for av in 0u128..1 << n {
                    for bv in 0u128..1 << (n + 1) {
                        let k = av | (bv << n);
                        let (o, s) = eval_on_key(&ra, k);
                        assert!(!s);
                        assert_eq!(o, av | (((av + bv) & ((2 << n) - 1)) << n));
                        let (o, s) = eval_on_key(&rs, k);
                        assert!(!s);
                        let d = (bv + (2 << n) - av) & ((2 << n) - 1);
                        assert_eq!(o, av | (d << n));
                        for tv in 0..2u128 {
                            if bv >> n == 1 {
                                continue;
                            }
                            let (o, s) = eval_on_key(&rc, k | (tv << t));
                            assert!(!s);
                            let want = tv ^ u128::from(bv < av);
                            assert_eq!(o, k | (want << t), "n={n} a={av} b={bv}");
                            let (o, s) = eval_on_key(&rp, k | (tv << t));
                            assert_eq!(o, k | (tv << t));
                            assert_eq!(s, bv < av, "phase n={n} a={av} b={bv}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn mbu_lookup_unlookup_exhaustive() {
        for w in 1..=5usize {
            let nout = 5;
            // ctrl 0, addr 1..=w, and w+1..=2w, out 2w+1..
            let addr: Vec<usize> = (1..=w).collect();
            let and: Vec<usize> = (w + 1..=2 * w).collect();
            let out: Vec<usize> = (2 * w + 1..2 * w + 1 + nout).collect();
            for seed in 0..6u64 {
                let mut oc = Outcomes::new(seed, (seed % 3) as u8);
                let table: Vec<u64> = (0..1u64 << w)
                    .map(|v| (v.wrapping_mul(0x9E37_79B9) ^ seed.wrapping_mul(77)) % 32)
                    .collect();
                let spec = Rc::new(LookupSpec {
                    ctrl: 0,
                    addr: addr.clone(),
                    and: and.clone(),
                    out: out.clone(),
                    table: table.clone(),
                    mbu_and: true,
                    meas_unlookup: true,
                });
                let lk = lookup_ops(&spec);
                assert_eq!(and_count(&lk), (1 << w) - 1);
                let mut ops = Vec::new();
                resolve(&[LOp::Lookup(spec.clone())], &mut || oc.next_bit(), &mut ops);
                let mut un = Vec::new();
                resolve(&[LOp::Unlookup(spec)], &mut || oc.next_bit(), &mut un);
                for ctrl in 0..2u128 {
                    for v in 0u128..1 << w {
                        let k = ctrl | (v << 1);
                        let (o, s) = eval_on_key(&ops, k);
                        assert!(!s);
                        let y = if ctrl == 1 { u128::from(table[v as usize]) } else { 0 };
                        assert_eq!(o, k | (y << (2 * w + 1)));
                        let (o2, s2) = eval_on_key(&un, o);
                        assert!(!s2, "w={w} seed={seed} ctrl={ctrl} v={v}: phase not fixed");
                        assert_eq!(o2, k);
                    }
                }
            }
        }
    }

    #[test]
    fn phase_table_all_splits() {
        for w in 0..=5usize {
            let addr: Vec<usize> = (1..=w).collect();
            let scratch: Vec<usize> = (w + 1..w + 1 + 40).collect();
            for seed in 0..20u64 {
                let mut oc = Outcomes::new(seed + 100, 0);
                let g: Vec<bool> = (0..1usize << w).map(|_| oc.next_bit()).collect();
                for k in 0..=w {
                    let ops = phase_table_k(0, &addr, &g, &scratch, k);
                    let mut r = Vec::new();
                    resolve(&ops, &mut || oc.next_bit(), &mut r);
                    for ctrl in 0..2u128 {
                        for v in 0u128..1 << w {
                            let key = ctrl | (v << 1);
                            let (o, s) = eval_on_key(&r, key);
                            assert_eq!(o, key);
                            assert_eq!(s, ctrl == 1 && g[v as usize], "w={w} k={k} v={v}");
                        }
                    }
                }
                if w == 4 {
                    let best = phase_table(0, &addr, &g, &scratch);
                    assert!(and_count(&best) <= 4);
                }
            }
        }
    }

    fn check_block(n_mod: u64, w: usize, o: &MbuOpts, mode: u8) {
        let n = crate::shor::work_bits(n_mod);
        let lay = MbuLayout::new(n, w, o);
        assert!(lay.num_qubits() <= 128);
        for a in (2..n_mod).filter(|&a| crate::algorithms::gcd(a, n_mod) == 1).take(3) {
            let mut oc = Outcomes::new(a * 1000 + n_mod, mode);
            let ops = controlled_ua(&lay, a, n_mod, o, &mut oc);
            for ctrl in 0..2u128 {
                for x in 0..n_mod {
                    let k = ctrl | (u128::from(x) << 1);
                    let (out, s) = eval_on_key(&ops, k);
                    assert!(!s, "N={n_mod} w={w} {o:?} a={a} x={x}: sign");
                    let y = if ctrl == 1 { mulmod(a, x, n_mod) } else { x };
                    assert_eq!(out, ctrl | (u128::from(y) << 1), "N={n_mod} w={w} {o:?} a={a} x={x}");
                }
            }
        }
    }

    #[test]
    fn controlled_ua_exhaustive_small() {
        for n_mod in [15u64, 21, 35, 55, 63, 77] {
            for w in 1..=4 {
                let nf = |o: MbuOpts| MbuOpts { flag: false, ..o };
                for o in [
                    MbuOpts::ALL,
                    MbuOpts::LOOKUPS,
                    MbuOpts::NONE,
                    nf(MbuOpts::ALL),
                    nf(MbuOpts::LOOKUPS),
                ] {
                    for mode in 0..3 {
                        check_block(n_mod, w, &o, mode);
                    }
                }
            }
        }
    }

    /// Genuine quantum check (no determinism assumed): a superposition of
    /// every `x < N` with random amplitudes and control `|+⟩` through the
    /// resolved block on a sparse state vector, each `MeasX` as `H`,
    /// projection (probability must be 1/2) and reset.
    #[test]
    fn controlled_ua_on_sparse_state_with_real_measurements() {
        for (n_mod, w) in [(15u64, 2usize), (21, 3), (35, 4)] {
            let n = crate::shor::work_bits(n_mod);
            for o in [MbuOpts::ALL, MbuOpts::LOOKUPS] {
                let lay = MbuLayout::new(n, w, &o);
                let nq = lay.num_qubits();
                assert!(nq <= 64);
                let a = 2;
                let mut oc = Outcomes::new(n_mod + w as u64, 0);
                let ops = controlled_ua(&lay, a, n_mod, &o, &mut oc);
                let mut amps = Vec::new();
                let mut z = 0.0f64;
                for c in 0..2u64 {
                    for x in 0..n_mod {
                        let amp = Complex64::new(
                            ((x * 7 + c * 3 + 1) % 11) as f64 - 5.0,
                            ((x * 5 + c) % 7) as f64 - 3.0,
                        );
                        z += amp.norm_sqr();
                        amps.push((c | (x << 1), amp));
                    }
                }
                let mut dense = std::collections::HashMap::new();
                for &(k, a) in &amps {
                    dense.insert(k, a / z.sqrt());
                }
                let mut s = SparseState::from_amplitudes(nq, dense.clone());
                for op in &ops {
                    match *op {
                        MbuOp::G(g) => s.apply_gate(&g).unwrap(),
                        MbuOp::MeasX(q, m) => {
                            s.apply_gate(&Gate::H(q)).unwrap();
                            let p = s.collapse(q, m);
                            assert!((p - 0.5).abs() < 1e-12, "P(m) = {p}");
                            if m {
                                s.apply_gate(&Gate::X(q)).unwrap();
                            }
                        }
                    }
                }
                for (&k, &amp) in &dense {
                    let c = k & 1;
                    let x = k >> 1;
                    let y = if c == 1 { mulmod(a, x, n_mod) } else { x };
                    let got = s.amplitude(c | (y << 1));
                    assert!((got - amp).norm() < 1e-12, "N={n_mod} {o:?} x={x} c={c}");
                }
                assert!((s.norm_sqr() - 1.0).abs() < 1e-12);
            }
        }
    }
}
