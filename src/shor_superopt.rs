//! Superoptimised windowed oracle (exp/superopt, `research/shor/superopt.md`).
//!
//! Same qubit layout and the same arithmetic as [`crate::shor_window`]
//! (Gidney-style windowed modular multiplication built from X, CNOT and
//! CCX only), with every building block replaced by a cheaper one that is
//! proved correct (exhaustively for small widths in the tests, and by the
//! inductive argument in `research/shor/superopt.md`). Each improvement can be
//! switched on separately through [`Opts`] so its effect can be measured:
//!
//! * `unary`: the table lookup walks the address tree MSB-first with
//!   Babbush et al.'s unary iteration (arXiv:1805.03662, Fig. 7) and the
//!   sibling-switch `CNOT(parent, flag)` instead of recomputing the AND
//!   chain for every address. The baseline lookup shares chain prefixes
//!   LSB-first, which in counting order never shares anything: it costs
//!   about `2w` Toffolis per address (`124` for `w = 4`); unary iteration
//!   costs `2(2^w − 1)` (`30`).
//! * `fanout`: every node of the address tree carries a live flag
//!   (`ctrl ∧ address prefix`) at some point, and the flag of a node is the
//!   XOR of its leaves' flags. For each output bit the set of addresses
//!   whose table entry has that bit set is written as the XOR of the
//!   fewest tree nodes (an exact tree DP: each node is either used, and its
//!   children must then produce the complement, or not). The CNOT fan-out
//!   drops from `Σ popcount(T[v])` to the DP optimum.
//! * `comparator`: the modular adder's flag `t` is uncomputed by a
//!   comparator (`t ^= [b < L]`, half a Cuccaro subtraction, the borrow
//!   copied out, the half undone) instead of a subtraction and an addition:
//!   `10n → 8n` Toffolis per modular addition.
//! * `kflip`: the constant register `K` goes from `N` to `t·N` with
//!   `X(t) CNOT(t, K_j) X(t)` instead of unloading and reloading.
//! * `direct_first`: the accumulator is 0 before the first window, so the
//!   first lookup writes straight into `b` (no modular addition, no
//!   unlookup).
//! * `keep_chain`: the trailing gates of a lookup that only touch the
//!   control, the address and the AND ancillas commute with the modular
//!   addition (disjoint qubits) and cancel against the head of the
//!   unlookup; they are dropped from both.

use crate::circuit::Circuit;
use crate::gate::Gate;
use crate::shor_ripple::{cuccaro_add, cuccaro_sub, load_constant};
use crate::shor_window::WindowLayout;

/// Which optimisations to apply (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Opts {
    pub unary: bool,
    pub fanout: bool,
    pub comparator: bool,
    pub kflip: bool,
    pub direct_first: bool,
    pub keep_chain: bool,
    /// Finish with the generic commutation-aware peephole pass
    /// ([`crate::compile::peephole`]): cancels the inverse pairs left at
    /// block junctions (e.g. `CNOT(c0, b0)` between consecutive adders).
    pub peephole: bool,
    /// Choose the window sizes (each at most `lay.w`) per multiplier by an
    /// exact gate-count DP instead of uniform windows.
    pub window_dp: bool,
    /// Apply the SAT-derived window rewrite rules
    /// ([`sat_peephole`], `src/shor_superopt_rules.txt`).
    pub sat_rules: bool,
    /// Run the peephole / SAT-rule passes once on the modular-adder block
    /// (identical in every window) instead of on the whole controlled-U
    /// circuit: much cheaper to build, and the passes then miss only the
    /// rewrites that straddle a block junction.
    pub block_passes: bool,
}

impl Opts {
    /// Everything off: gate-for-gate the circuit of [`crate::shor_window`].
    pub const BASELINE: Opts = Opts {
        unary: false,
        fanout: false,
        comparator: false,
        kflip: false,
        direct_first: false,
        keep_chain: false,
        peephole: false,
        window_dp: false,
        sat_rules: false,
        block_passes: false,
    };
    /// Everything on.
    pub const ALL: Opts = Opts {
        unary: true,
        fanout: true,
        comparator: true,
        kflip: true,
        direct_first: true,
        keep_chain: true,
        peephole: true,
        window_dp: false,
        sat_rules: true,
        block_passes: true,
    };
}

impl Default for Opts {
    fn default() -> Self {
        Opts::ALL
    }
}

/// Per-node fan-out plan of a lookup: `use_[d][h]` is the bit mask of
/// output bits CNOT-ed from the flag of tree node `(d, h)` (depth `d`,
/// address high bits `h`; depth 0 is the root = the control, depth `w` the
/// leaves).
#[derive(Clone, Debug)]
pub struct FanoutPlan {
    pub w: usize,
    pub use_: Vec<Vec<u64>>,
}

impl FanoutPlan {
    /// Plain fan-out: every leaf `v` CNOTs its own entry `T[v]`.
    pub fn leaves(table: &[u64]) -> Self {
        let w = table.len().trailing_zeros() as usize;
        assert_eq!(table.len(), 1 << w);
        let mut use_: Vec<Vec<u64>> = (0..=w).map(|d| vec![0; 1 << d]).collect();
        use_[w].copy_from_slice(table);
        FanoutPlan { w, use_ }
    }

    /// Optimal fan-out over the tree nodes, bit by bit (exact DP).
    pub fn optimal(table: &[u64], nbits: usize) -> Self {
        let w = table.len().trailing_zeros() as usize;
        assert_eq!(table.len(), 1 << w);
        let mut use_: Vec<Vec<u64>> = (0..=w).map(|d| vec![0; 1 << d]).collect();
        for i in 0..nbits {
            // cost[d][h][flip]
            let mut cost: Vec<Vec<[u32; 2]>> = (0..=w).map(|d| vec![[0; 2]; 1 << d]).collect();
            for v in 0..1usize << w {
                let s = ((table[v] >> i) & 1) as u32;
                cost[w][v] = [s, 1 - s];
            }
            for d in (0..w).rev() {
                for h in 0..1usize << d {
                    let (l, r) = (cost[d + 1][2 * h], cost[d + 1][2 * h + 1]);
                    for f in 0..2 {
                        let keep = l[f] + r[f];
                        let flip = 1 + l[1 - f] + r[1 - f];
                        cost[d][h][f] = keep.min(flip);
                    }
                }
            }
            // top-down choice
            let mut stack = vec![(0usize, 0usize, 0usize)];
            while let Some((d, h, f)) = stack.pop() {
                if d == w {
                    if cost[w][h][f] == 1 {
                        use_[w][h] |= 1 << i;
                    }
                    continue;
                }
                let (l, r) = (cost[d + 1][2 * h], cost[d + 1][2 * h + 1]);
                let keep = l[f] + r[f];
                let flip = 1 + l[1 - f] + r[1 - f];
                let g = if flip < keep {
                    use_[d][h] |= 1 << i;
                    1 - f
                } else {
                    f
                };
                stack.push((d + 1, 2 * h, g));
                stack.push((d + 1, 2 * h + 1, g));
            }
        }
        FanoutPlan { w, use_ }
    }

    /// Number of fan-out CNOTs.
    pub fn cnots(&self) -> usize {
        self.use_
            .iter()
            .flatten()
            .map(|m| m.count_ones() as usize)
            .sum()
    }

    /// The table this plan writes (XOR of node masks over each address's
    /// ancestors including itself).
    pub fn table(&self) -> Vec<u64> {
        (0..1usize << self.w)
            .map(|v| {
                (0..=self.w)
                    .map(|d| self.use_[d][v >> (self.w - d)])
                    .fold(0, |a, b| a ^ b)
            })
            .collect()
    }

    /// Whether node `(d, h)` or any node below it has a non-empty mask.
    pub fn nonempty(&self, d: usize, h: usize) -> bool {
        if self.use_[d][h] != 0 {
            return true;
        }
        d < self.w && (self.nonempty(d + 1, 2 * h) || self.nonempty(d + 1, 2 * h + 1))
    }
}

/// Unary-iteration table lookup: `out ^= T[v]` (as encoded by `plan`)
/// where `v` is the value of `addr` (LSB first), conditioned on `ctrl`.
/// `and[0..w]` are clean ancillas and return clean. Self-inverse as a
/// unitary; the gate list reversed is its inverse.
pub fn lookup_unary(
    c: &mut Circuit,
    ctrl: usize,
    addr: &[usize],
    and: &[usize],
    out: &[usize],
    plan: &FanoutPlan,
) {
    let w = addr.len();
    assert_eq!(plan.w, w);
    assert!(and.len() >= w);
    #[allow(clippy::too_many_arguments)]
    fn rec(
        c: &mut Circuit,
        d: usize,
        h: usize,
        p: usize,
        addr: &[usize],
        and: &[usize],
        out: &[usize],
        plan: &FanoutPlan,
    ) {
        let w = plan.w;
        let m = plan.use_[d][h];
        for (i, &q) in out.iter().enumerate() {
            if (m >> i) & 1 == 1 {
                c.cnot(p, q);
            }
        }
        assert!(m >> out.len() == 0, "table entry wider than the output");
        if d == w {
            return;
        }
        let bit = addr[w - 1 - d];
        let f = and[d];
        let (left, right) = (plan.nonempty(d + 1, 2 * h), plan.nonempty(d + 1, 2 * h + 1));
        match (right, left) {
            (false, false) => {}
            (true, false) => {
                c.gate(Gate::Ccx(p, bit, f));
                rec(c, d + 1, 2 * h + 1, f, addr, and, out, plan);
                c.gate(Gate::Ccx(p, bit, f));
            }
            (false, true) => {
                c.gate(Gate::Ccx(p, bit, f));
                c.cnot(p, f);
                rec(c, d + 1, 2 * h, f, addr, and, out, plan);
                c.cnot(p, f);
                c.gate(Gate::Ccx(p, bit, f));
            }
            (true, true) => {
                c.gate(Gate::Ccx(p, bit, f));
                rec(c, d + 1, 2 * h + 1, f, addr, and, out, plan);
                c.cnot(p, f);
                rec(c, d + 1, 2 * h, f, addr, and, out, plan);
                c.cnot(p, f);
                c.gate(Gate::Ccx(p, bit, f));
            }
        }
    }
    rec(c, 0, 0, ctrl, addr, and, out, plan);
}

/// `t ^= [b < a]` for `n`-bit `a`, `b` (LSB first), `c0` a clean ancilla:
/// the first half of a Cuccaro subtraction (whose top carry is the
/// borrow), one CNOT, and the half undone. `2n` Toffolis, `4n + 1` CNOTs.
pub fn compare_lt(c: &mut Circuit, a: &[usize], b: &[usize], c0: usize, t: usize) {
    let n = a.len();
    assert_eq!(b.len(), n);
    assert!(n >= 1);
    let uma_inv = |c: &mut Circuit, x: usize, y: usize, z: usize| {
        c.cnot(x, y).cnot(z, x).gate(Gate::Ccx(x, y, z));
    };
    let uma = |c: &mut Circuit, x: usize, y: usize, z: usize| {
        c.gate(Gate::Ccx(x, y, z)).cnot(z, x).cnot(x, y);
    };
    uma_inv(c, c0, b[0], a[0]);
    for i in 1..n {
        uma_inv(c, a[i - 1], b[i], a[i]);
    }
    c.cnot(a[n - 1], t);
    for i in (1..n).rev() {
        uma(c, a[i - 1], b[i], a[i]);
    }
    uma(c, c0, b[0], a[0]);
}

/// `b -> (b + L) mod N` for `b, L < N` (as [`crate::shor_window::add_mod_reg`]).
pub fn add_mod_reg(c: &mut Circuit, lay: &WindowLayout, n_mod: u64, o: &Opts) {
    let (l, b, k, c0, t) = (&lay.l, &lay.b, &lay.k, lay.c0, lay.t);
    let n = lay.n;
    let bn = b[n];
    cuccaro_add(c, l, b, c0);
    load_constant(c, k, n_mod, &[]);
    cuccaro_sub(c, k, b, c0);
    if o.kflip {
        // K = N, t = 0 -> t = [b + L < N], K = t·N
        c.cnot(bn, t);
        c.x(t);
        load_constant(c, k, n_mod, &[t]);
        c.x(t);
    } else {
        load_constant(c, k, n_mod, &[]);
        c.cnot(bn, t);
        load_constant(c, k, n_mod, &[t]);
    }
    cuccaro_add(c, k, b, c0);
    load_constant(c, k, n_mod, &[t]);
    if o.comparator {
        // t = [no reduction] = [b_new >= L] = NOT [b_new < L]
        compare_lt(c, l, &b[..n], c0, t);
        c.x(t);
    } else {
        cuccaro_sub(c, l, b, c0);
        c.x(bn);
        c.cnot(bn, t);
        c.x(bn);
        cuccaro_add(c, l, b, c0);
    }
}

/// Emits one lookup into `out` with table `table`.
fn emit_lookup(
    c: &mut Circuit,
    lay: &WindowLayout,
    addr: &[usize],
    out: &[usize],
    table: &[u64],
    o: &Opts,
) {
    if o.unary {
        let plan = if o.fanout {
            FanoutPlan::optimal(table, out.len())
        } else {
            FanoutPlan::leaves(table)
        };
        lookup_unary(c, lay.ctrl, addr, &lay.and, out, &plan);
    } else {
        crate::shor_window::lookup(c, lay.ctrl, addr, &lay.and, out, table);
    }
}

/// Emits one window (bits `start..start + w` of `x`, table
/// `T[v] = v·base mod N` with `base = a·2^start mod N`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_window(
    c: &mut Circuit,
    lay: &WindowLayout,
    n_mod: u64,
    o: &Opts,
    start: usize,
    w: usize,
    base: u64,
    madd: &[bool],
    block: Option<&Circuit>,
) {
    let n = lay.n;
    let table: Vec<u64> = (0..1u64 << w)
        .map(|v| (u128::from(v) * u128::from(base) % u128::from(n_mod)) as u64)
        .collect();
    let addr = &lay.x[start..start + w];
    if start == 0 && o.direct_first {
        emit_lookup(c, lay, addr, &lay.b[..n], &table, o);
        return;
    }
    let mut lk = Circuit::new(lay.num_qubits());
    emit_lookup(&mut lk, lay, addr, &lay.l, &table, o);
    let gates: Vec<Gate> = lk.gates().copied().collect();
    let mut keep = gates.len();
    if o.keep_chain {
        while keep > 0 && gates[keep - 1].qubits().iter().all(|&q| !madd[q]) {
            keep -= 1;
        }
    }
    for g in &gates[..keep] {
        c.gate(*g);
    }
    match block {
        Some(b) => {
            c.append(b);
        }
        None => add_mod_reg(c, lay, n_mod, o),
    }
    if o.keep_chain {
        for g in gates[..keep].iter().rev() {
            c.gate(*g);
        }
    } else {
        // the lookup is self-inverse: emit it again (as the baseline)
        emit_lookup(c, lay, addr, &lay.l, &table, o);
    }
}

/// Window sizes for a multiply-add by `a`: uniform `lay.w` (last window
/// shorter), or, with `o.window_dp`, the split of the `n` bits into windows
/// of at most `lay.w` bits that minimises the emitted gate count (exact
/// dynamic programme over window start and size; every window is emitted
/// and counted).
pub fn window_sizes(lay: &WindowLayout, a: u64, n_mod: u64, o: &Opts) -> Vec<usize> {
    let n = lay.n;
    if !o.window_dp {
        let mut v = Vec::new();
        let mut s = 0;
        while s < n {
            let w = lay.w.min(n - s);
            v.push(w);
            s += w;
        }
        return v;
    }
    let mut base = vec![a % n_mod; n + 1];
    for s in 1..=n {
        base[s] = (u128::from(base[s - 1]) * 2 % u128::from(n_mod)) as u64;
    }
    let madd = madd_mask(lay);
    let mut best = vec![(usize::MAX, 0usize); n + 1];
    best[n] = (0, 0);
    for s in (0..n).rev() {
        for w in 1..=lay.w.min(n - s) {
            if best[s + w].0 == usize::MAX {
                continue;
            }
            let mut c = Circuit::new(lay.num_qubits());
            emit_window(&mut c, lay, n_mod, o, s, w, base[s], &madd, None);
            let cost = c.ops.len() + best[s + w].0;
            if cost < best[s].0 {
                best[s] = (cost, w);
            }
        }
    }
    let mut v = Vec::new();
    let mut s = 0;
    while s < n {
        v.push(best[s].1);
        s += best[s].1;
    }
    v
}

pub(crate) fn madd_mask(lay: &WindowLayout) -> Vec<bool> {
    // qubits the modular adder touches (for keep_chain)
    let mut madd = vec![false; lay.num_qubits()];
    for &q in lay.b.iter().chain(&lay.l).chain(&lay.k) {
        madd[q] = true;
    }
    madd[lay.c0] = true;
    madd[lay.t] = true;
    madd
}

/// Controlled windowed multiply-add `|c>|x>|b=0> -> |c>|x>|c·a·x mod N>`.
pub fn cmult(lay: &WindowLayout, a: u64, n_mod: u64, o: &Opts) -> Circuit {
    let mut c = Circuit::new(lay.num_qubits());
    let madd = madd_mask(lay);
    let block = o.block_passes.then(|| modadd_block(lay, n_mod, o));
    let mut start = 0;
    let mut base = a % n_mod;
    for w in window_sizes(lay, a, n_mod, o) {
        emit_window(&mut c, lay, n_mod, o, start, w, base, &madd, block.as_ref());
        for _ in 0..w {
            base = (u128::from(base) * 2 % u128::from(n_mod)) as u64;
        }
        start += w;
    }
    c
}

/// The modular adder with the peephole / SAT-rule passes applied to it
/// alone. At its start `K`, `c0`, `t` and `b[n]` are 0 (known constants for
/// the rules); `b`, `L` and everything else are unknown.
pub fn modadd_block(lay: &WindowLayout, n_mod: u64, o: &Opts) -> Circuit {
    let mut c = Circuit::new(lay.num_qubits());
    add_mod_reg(&mut c, lay, n_mod, o);
    if o.peephole {
        c = reversible_peephole(&c);
    }
    if o.sat_rules {
        let mut init = vec![None; lay.num_qubits()];
        for &q in lay.k.iter().chain([&lay.c0, &lay.t, &lay.b[lay.n]]) {
            init[q] = Some(false);
        }
        c = sat_peephole_init(&c, &init);
        if o.peephole {
            c = reversible_peephole(&c);
        }
    }
    c
}

/// Controlled `U_a`: `|c>|x>|0…> -> |c>|a^c x mod N>|0…>` for `x < N`.
pub fn controlled_ua(lay: &WindowLayout, a: u64, n_mod: u64, o: &Opts) -> Circuit {
    let inv = crate::shor::mod_inverse(a, n_mod);
    let mut c = cmult(lay, a, n_mod, o);
    for i in 0..lay.n {
        let (x, b) = (lay.x[i], lay.b[i]);
        c.cnot(b, x);
        c.gate(Gate::Ccx(lay.ctrl, x, b));
        c.cnot(b, x);
    }
    c.append(&cmult(lay, inv, n_mod, o).inverse());
    if o.block_passes {
        return c;
    }
    if o.peephole {
        c = reversible_peephole(&c);
    }
    if o.sat_rules {
        c = sat_peephole(&c, lay.n + 1);
        if o.peephole {
            c = reversible_peephole(&c);
        }
    }
    c
}

/// Local gate of a rewrite rule (wire indices inside the window).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Lg {
    X(u8),
    Cx(u8, u8),
    Ccx(u8, u8, u8),
}

/// Packed window key: gates (12 bits each, type code 1..3 + three 3-bit
/// wire indices) and (wire count, constant pattern 2 bits per wire).
type RuleKey = (u128, u16);

fn lg_code(g: &Lg) -> u128 {
    let (t, a, b, c) = match *g {
        Lg::X(q) => (1u128, q, 0, 0),
        Lg::Cx(x, y) => (2, x, y, 0),
        Lg::Ccx(x, y, z) => (3, x, y, z),
    };
    t | (u128::from(a) << 2) | (u128::from(b) << 5) | (u128::from(c) << 8)
}

fn rule_key(gates: &[Lg], consts: &[Option<bool>]) -> RuleKey {
    let mut k = 0u128;
    for (i, g) in gates.iter().enumerate() {
        k |= lg_code(g) << (12 * i);
    }
    let mut m = consts.len() as u16;
    for (i, c) in consts.iter().enumerate() {
        let v: u16 = match c {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        };
        m |= v << (3 + 2 * i);
    }
    (k, m)
}

/// Window shape of the SAT peephole (must match `tools/superopt/peep.py`).
pub const SAT_Q: usize = 5;
pub const SAT_LMAX: usize = 10;

fn parse_lgs(s: &str) -> Vec<Lg> {
    s.split(';')
        .filter(|t| !t.is_empty())
        .map(|t| {
            let p: Vec<&str> = t.split_whitespace().collect();
            let n = |i: usize| p[i].parse::<u8>().unwrap();
            match p[0] {
                "X" => Lg::X(n(1)),
                "CX" => Lg::Cx(n(1), n(2)),
                "CCX" => Lg::Ccx(n(1), n(2), n(3)),
                g => panic!("bad gate {g}"),
            }
        })
        .collect()
}

fn sim_lgs(gs: &[Lg], mut v: u32) -> u32 {
    for g in gs {
        match *g {
            Lg::X(q) => v ^= 1 << q,
            Lg::Cx(c, t) => v ^= ((v >> c) & 1) << t,
            Lg::Ccx(a, b, t) => v ^= ((v >> a) & (v >> b) & 1) << t,
        }
    }
    v
}

/// The rule table, parsed once and **verified**: every rule's replacement
/// is checked against its pattern on every input consistent with the
/// rule's known constants, so correctness does not rest on the SAT solver.
fn sat_rules() -> &'static std::collections::HashMap<RuleKey, Vec<Lg>> {
    static R: std::sync::OnceLock<std::collections::HashMap<RuleKey, Vec<Lg>>> =
        std::sync::OnceLock::new();
    R.get_or_init(|| {
        let mut m = std::collections::HashMap::new();
        for line in include_str!("shor_superopt_rules.txt").lines() {
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            let f: Vec<&str> = line.split('|').collect();
            let nw: usize = f[0].parse().unwrap();
            let consts: Vec<Option<bool>> = f[1]
                .chars()
                .map(|ch| match ch {
                    '0' => Some(false),
                    '1' => Some(true),
                    _ => None,
                })
                .collect();
            assert_eq!(consts.len(), nw);
            let (pat, rep) = (parse_lgs(f[2]), parse_lgs(f[3]));
            assert!(rep.len() < pat.len(), "rule does not shrink: {line}");
            for v in 0u32..1 << nw {
                let ok = consts
                    .iter()
                    .enumerate()
                    .all(|(i, c)| !matches!(c, Some(b) if ((v >> i) & 1 == 1) != *b));
                if ok {
                    assert_eq!(sim_lgs(&pat, v), sim_lgs(&rep, v), "invalid rule {line}");
                }
            }
            assert!(pat.len() <= SAT_LMAX && nw <= SAT_Q);
            m.insert(rule_key(&pat, &consts), rep);
        }
        m
    })
}

fn gate_qs(g: &Gate) -> ([usize; 3], usize) {
    match *g {
        Gate::X(q) => ([q, 0, 0], 1),
        Gate::Cnot(c, t) => ([c, t, 0], 2),
        Gate::Ccx(a, b, t) => ([a, b, t], 3),
        _ => panic!("sat_peephole: X/CNOT/CCX only"),
    }
}

fn const_step(val: &mut [Option<bool>], g: &Gate) {
    let flip = |val: &mut [Option<bool>], t: usize| {
        if let Some(b) = val[t] {
            val[t] = Some(!b);
        }
    };
    match *g {
        Gate::X(q) => flip(val, q),
        Gate::Cnot(c, t) => match val[c] {
            Some(false) => {}
            Some(true) => flip(val, t),
            None => val[t] = None,
        },
        Gate::Ccx(a, b, t) => match (val[a], val[b]) {
            (Some(false), _) | (_, Some(false)) => {}
            (Some(true), Some(true)) => flip(val, t),
            _ => val[t] = None,
        },
        _ => unreachable!(),
    }
}

/// Rewrites the circuit with the SAT-derived window rules: from each gate,
/// the longest run of consecutive gates on at most [`SAT_Q`] wires (and at
/// most [`SAT_LMAX`] gates) is looked up, together with the wires whose
/// value is a known constant at that point (constant propagation from
/// "qubits `>= anc_from` start at 0"), and replaced by the rule's shorter
/// equivalent. Repeated until nothing changes.
pub fn sat_peephole(c: &Circuit, anc_from: usize) -> Circuit {
    let init: Vec<Option<bool>> = (0..c.num_qubits)
        .map(|q| if q < anc_from { None } else { Some(false) })
        .collect();
    sat_peephole_init(c, &init)
}

/// [`sat_peephole`] with an explicit initial constant map (`Some(b)`: the
/// qubit is known to hold `b` on every input the circuit is used on).
pub fn sat_peephole_init(c: &Circuit, init: &[Option<bool>]) -> Circuit {
    let rules = sat_rules();
    let mut gates: Vec<Gate> = c.gates().copied().collect();
    for _pass in 0..8 {
        let mut val: Vec<Option<bool>> = init.to_vec();
        let mut out: Vec<Gate> = Vec::with_capacity(gates.len());
        let mut changed = false;
        let mut i = 0;
        while i < gates.len() {
            let mut wires = [0usize; SAT_Q];
            let mut nw = 0usize;
            let mut local = [Lg::X(0); SAT_LMAX];
            let mut j = i;
            'grow: while j < gates.len() && j - i < SAT_LMAX {
                let (qs, k) = gate_qs(&gates[j]);
                let mut idx = [0u8; 3];
                let mut add = [0usize; 3];
                let mut nadd = 0;
                for (s, &q) in qs[..k].iter().enumerate() {
                    if let Some(p) = wires[..nw].iter().position(|&w| w == q) {
                        idx[s] = p as u8;
                    } else if let Some(p) = add[..nadd].iter().position(|&w| w == q) {
                        idx[s] = (nw + p) as u8;
                    } else {
                        if nw + nadd + 1 > SAT_Q {
                            break 'grow;
                        }
                        add[nadd] = q;
                        idx[s] = (nw + nadd) as u8;
                        nadd += 1;
                    }
                }
                wires[nw..nw + nadd].copy_from_slice(&add[..nadd]);
                nw += nadd;
                local[j - i] = match k {
                    1 => Lg::X(idx[0]),
                    2 => Lg::Cx(idx[0], idx[1]),
                    _ => Lg::Ccx(idx[0], idx[1], idx[2]),
                };
                j += 1;
            }
            if j - i >= 2 {
                let mut consts = [None; SAT_Q];
                for (k, &q) in wires[..nw].iter().enumerate() {
                    consts[k] = val[q];
                }
                if let Some(rep) = rules.get(&rule_key(&local[..j - i], &consts[..nw])) {
                    for g in rep {
                        let g = match *g {
                            Lg::X(q) => Gate::X(wires[q as usize]),
                            Lg::Cx(c, t) => Gate::Cnot(wires[c as usize], wires[t as usize]),
                            Lg::Ccx(a, b, t) => {
                                Gate::Ccx(wires[a as usize], wires[b as usize], wires[t as usize])
                            }
                        };
                        const_step(&mut val, &g);
                        out.push(g);
                    }
                    i = j;
                    changed = true;
                    continue;
                }
            }
            const_step(&mut val, &gates[i]);
            out.push(gates[i]);
            i += 1;
        }
        gates = out;
        if std::env::var_os("QSIM_SATPEEP_DEBUG").is_some() {
            eprintln!(
                "sat_peephole pass {_pass}: {} gates, changed={changed}",
                gates.len()
            );
        }
        if !changed {
            break;
        }
    }
    let mut r = Circuit::new(c.num_qubits);
    for g in gates {
        r.gate(g);
    }
    r
}

/// The peephole pass restricted to X/CNOT/CCX circuits. Its output equals
/// the input up to a global phase; both are permutation matrices built from
/// X/CNOT/CCX only (checked), so they are the same permutation.
pub fn reversible_peephole(c: &Circuit) -> Circuit {
    let p = crate::compile::peephole::optimize(c).circuit;
    for g in p.gates() {
        assert!(
            matches!(g, Gate::X(_) | Gate::Cnot(..) | Gate::Ccx(..)),
            "peephole produced a non-reversible gate {g:?}"
        );
    }
    assert_eq!(p.gates().count(), p.ops.len());
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::gcd;
    use crate::shor_ripple::{eval_circuit_on_key, gate_counts};

    fn lcg(s: &mut u64) -> u64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *s >> 11
    }

    #[test]
    fn fanout_plan_reproduces_table_and_is_no_worse() {
        let mut s = 7u64;
        for w in 1..=6usize {
            for _ in 0..50 {
                let nb = 1 + (lcg(&mut s) % 40) as usize;
                let mut table: Vec<u64> = (0..1usize << w)
                    .map(|_| lcg(&mut s) & ((1u64 << nb) - 1))
                    .collect();
                table[0] = 0;
                let p = FanoutPlan::optimal(&table, nb);
                assert_eq!(p.table(), table);
                assert!(p.cnots() <= FanoutPlan::leaves(&table).cnots());
            }
        }
    }

    #[test]
    fn fanout_plan_optimal_bruteforce_small() {
        // exhaustive over every 1-bit table on w <= 3: DP optimum equals
        // the minimum over all subsets of tree nodes
        for w in 1..=3usize {
            let nodes: Vec<(usize, usize)> = (0..=w)
                .flat_map(|d| (0..1usize << d).map(move |h| (d, h)))
                .collect();
            let cover = |d: usize, h: usize| -> u64 {
                let mut m = 0u64;
                for v in 0..1usize << w {
                    if v >> (w - d) == h {
                        m |= 1 << v;
                    }
                }
                m
            };
            let mut best = vec![u32::MAX; 1 << (1 << w)];
            for sub in 0u64..1 << nodes.len() {
                let mut m = 0;
                for (j, &(d, h)) in nodes.iter().enumerate() {
                    if (sub >> j) & 1 == 1 {
                        m ^= cover(d, h);
                    }
                }
                best[m as usize] = best[m as usize].min(sub.count_ones());
            }
            for s in 0u64..1 << (1 << w) {
                let table: Vec<u64> = (0..1usize << w).map(|v| (s >> v) & 1).collect();
                let p = FanoutPlan::optimal(&table, 1);
                assert_eq!(p.cnots() as u32, best[s as usize], "w={w} s={s:b}");
            }
        }
    }

    #[test]
    fn lookup_unary_exhaustive() {
        let mut s = 3u64;
        for w in 1..=5usize {
            for fan in [false, true] {
                let addr: Vec<usize> = (1..=w).collect();
                let and: Vec<usize> = (w + 1..=2 * w).collect();
                let nb = 9;
                let l: Vec<usize> = (2 * w + 1..2 * w + 1 + nb).collect();
                let mut table: Vec<u64> = (0..1u64 << w).map(|_| lcg(&mut s) % 512).collect();
                if w % 2 == 0 {
                    table[0] = 0;
                }
                let plan = if fan {
                    FanoutPlan::optimal(&table, nb)
                } else {
                    FanoutPlan::leaves(&table)
                };
                let mut c = Circuit::new(2 * w + 1 + nb);
                lookup_unary(&mut c, 0, &addr, &and, &l, &plan);
                let (_, tof) = gate_counts(&c);
                assert!(tof <= 2 * ((1 << w) - 1));
                let lo = (1u64 << (2 * w + 1)) - 1;
                for ctrl in 0..2u64 {
                    for v in 0..1u64 << w {
                        for l0 in [0u64, 0b101101011] {
                            let k = ctrl | (v << 1) | (l0 << (2 * w + 1));
                            let out = eval_circuit_on_key(k, &c);
                            let want = if ctrl == 1 {
                                l0 ^ table[v as usize]
                            } else {
                                l0
                            };
                            assert_eq!(out & lo, k & lo);
                            assert_eq!(out >> (2 * w + 1), want, "w={w} c={ctrl} v={v}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn compare_lt_exhaustive() {
        for n in 1..=5usize {
            // qubits: a 0..n, b n..2n, c0 2n, t 2n+1
            let a: Vec<usize> = (0..n).collect();
            let b: Vec<usize> = (n..2 * n).collect();
            let mut c = Circuit::new(2 * n + 2);
            compare_lt(&mut c, &a, &b, 2 * n, 2 * n + 1);
            for x in 0..1u64 << n {
                for y in 0..1u64 << n {
                    for t in 0..2u64 {
                        let k = x | (y << n) | (t << (2 * n + 1));
                        let out = eval_circuit_on_key(k, &c);
                        let want = k ^ (u64::from(y < x) << (2 * n + 1));
                        assert_eq!(out, want, "n={n} a={x} b={y}");
                    }
                }
            }
        }
    }

    fn all_opts() -> Vec<Opts> {
        let mut v = vec![Opts::BASELINE, Opts::ALL];
        v.push(Opts {
            peephole: true,
            ..Opts::BASELINE
        });
        v.push(Opts {
            block_passes: false,
            ..Opts::ALL
        });
        v.push(Opts {
            window_dp: true,
            ..Opts::ALL
        });
        for i in 0..6 {
            let mut o = Opts::BASELINE;
            match i {
                0 => o.unary = true,
                1 => {
                    o.unary = true;
                    o.fanout = true
                }
                2 => o.comparator = true,
                3 => o.kflip = true,
                4 => o.direct_first = true,
                _ => {
                    o.unary = true;
                    o.keep_chain = true
                }
            }
            v.push(o);
        }
        v
    }

    #[test]
    fn sat_rules_parse_and_verify() {
        // panics on any rule whose replacement differs from its pattern
        let _ = sat_rules().len();
    }

    #[test]
    fn baseline_opts_reproduce_shor_window_exactly() {
        for (n_mod, w) in [(15u64, 2usize), (35, 3), (221, 4)] {
            let n = crate::shor::work_bits(n_mod);
            let lay = WindowLayout::new(n, w);
            let a = 2;
            let x = crate::shor_window::controlled_ua(&lay, a, n_mod);
            let y = controlled_ua(&lay, a, n_mod, &Opts::BASELINE);
            assert_eq!(x.ops.len(), y.ops.len());
            assert!(x.ops == y.ops, "baseline differs from shor_window");
        }
    }

    #[test]
    fn add_mod_reg_exhaustive() {
        for n_mod in [3u64, 5, 7, 11, 13, 21] {
            let n = crate::shor::work_bits(n_mod);
            let lay = WindowLayout::new(n, 1);
            for o in all_opts() {
                let mut c = Circuit::new(lay.num_qubits());
                add_mod_reg(&mut c, &lay, n_mod, &o);
                for b in 0..n_mod {
                    for l in 0..n_mod {
                        let mut k = 0u64;
                        for i in 0..n {
                            k |= ((b >> i) & 1) << lay.b[i];
                            k |= ((l >> i) & 1) << lay.l[i];
                        }
                        let out = eval_circuit_on_key(k, &c);
                        let mut want = 0u64;
                        let s = (b + l) % n_mod;
                        for i in 0..n {
                            want |= ((s >> i) & 1) << lay.b[i];
                            want |= ((l >> i) & 1) << lay.l[i];
                        }
                        assert_eq!(out, want, "N={n_mod} b={b} l={l} {o:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn controlled_ua_exhaustive_small_all_opts() {
        for n_mod in [15u64, 21, 33, 35, 55, 63] {
            let n = crate::shor::work_bits(n_mod);
            for w in 1..=4 {
                let lay = WindowLayout::new(n, w);
                let anc_mask: u64 =
                    !((1u64 << (n + 1)) - 1) & (u64::MAX >> (64 - lay.num_qubits()));
                for o in all_opts() {
                    for a in (2..n_mod).filter(|&a| gcd(a, n_mod) == 1).take(3) {
                        let c = controlled_ua(&lay, a, n_mod, &o);
                        for ctrl in [0u64, 1] {
                            for x in 0..n_mod {
                                let k_out = eval_circuit_on_key(ctrl | (x << 1), &c);
                                assert_eq!(k_out & anc_mask, 0, "dirty N={n_mod} w={w} {o:?}");
                                assert_eq!(k_out & 1, ctrl);
                                let want = if ctrl == 1 { x * a % n_mod } else { x };
                                assert_eq!((k_out >> 1) & ((1 << n) - 1), want);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn all_opts_cut_gates_and_toffolis() {
        let (n_mod, a) = (1_005_973u64, 980_062u64);
        let n = crate::shor::work_bits(n_mod);
        let lay = WindowLayout::new(n, 4);
        let (g0, t0) = gate_counts(&controlled_ua(&lay, a, n_mod, &Opts::BASELINE));
        let (g1, t1) = gate_counts(&controlled_ua(&lay, a, n_mod, &Opts::ALL));
        assert!(g1 < g0 && t1 < t0, "{g0}/{t0} -> {g1}/{t1}");
    }
}
