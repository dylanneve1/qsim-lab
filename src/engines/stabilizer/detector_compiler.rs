//! Detector compiler: a backward sensitivity sweep that turns a noisy
//! Clifford circuit with detectors straight into the fault -> detector
//! columns of [`super::fast_sampler::FastSampler`] (notebook:
//! `research/qec/sampler-x.md`).
//!
//! The old route ([`super::symphase::SymPhaseSampler`]) pushes a *forward*
//! symbolic Pauli frame over all noise variables (dense bit lines of
//! `#variables` bits per qubit), reads raw measurement rows, then XORs them
//! into detector rows. Its cost grows like `gates x variables / 64`: 52 ms at
//! d = 15 and 1.1 s at d = 25 for Stim's rotated surface-code memory.
//!
//! This module sweeps the circuit *backwards*, as Stim's error analyzer does,
//! keeping for every qubit the set of detector/observable rows that an X or a
//! Z flip at the current point would toggle (its *sensitivity*):
//!
//! * a `DETECTOR`/`OBSERVABLE_INCLUDE` adds its row to the pending set of
//!   each measurement it reads;
//! * a Z-basis measurement of `q` XORs its pending set into `xs[q]` (an X
//!   flip before it flips the outcome);
//! * a reset clears `xs[q]` and `zs[q]`;
//! * a Clifford gate `G` maps sensitivities by `sens_before(P) =
//!   sens_after(G P G^dagger)` (e.g. CX(c, t): `xs[c] ^= xs[t]`,
//!   `zs[t] ^= zs[c]`; H swaps `xs` and `zs`);
//! * a noise channel reads its columns directly: an X flip toggles `xs[q]`,
//!   a Z flip `zs[q]`, a Y flip both.
//!
//! Sets are tiny in QEC circuits (detectors are local in time), so they are
//! kept as sorted `u32` lists in one arena and XOR-merged; no per-op
//! allocation, no reference tableau (detection events are frame flips, so
//! the noiseless reference sample is never needed), and the circuit is never
//! materialised as `#variables`-wide bit lines.
//!
//! The output ([`Columns`]) is, column for column and group for group, what
//! the old route produces after `with_parities(..).relative_to_reference()`
//! and pruning: the same variable groups in the same order, the same rows per
//! variable. `tests/engines/detector_compiler.rs` asserts that equality (and
//! the equality of the resulting `FastSampler`s, hit tables included) on
//! every circuit family in the test suite. The semantics of every operation
//! mirror `SymPhaseSampler::new`: gate noise after the gate, a measurement's
//! readout flip before and its coin after the outcome, a reset's coin before
//! its flip.

use super::symphase::{VarDist, VarGroup};
use crate::circuit::{check_gate, Circuit, Op, SimError};
use crate::gate::{is_multiple_of_half_pi, Gate};
use crate::io::stim::{StimCircuit, StimOp};
use crate::noise::NoiseModel;
use std::f64::consts::{FRAC_PI_2, PI};

/// The detector matrix in column form: `rows` output rows (detectors, then
/// observables); variable groups in program order (each group's variables
/// are consecutive, as in [`super::symphase::SymPhaseSampler`]); variable `v`
/// toggles rows `col_rows[col_start[v]..col_start[v + 1]]` (sorted).
#[derive(Clone, Debug, PartialEq)]
pub struct Columns {
    /// Number of output rows (detectors + observables).
    pub rows: usize,
    /// Variable groups that reach at least one row, in program order.
    pub groups: Vec<VarGroup>,
    /// CSC column starts (`#variables + 1` entries).
    pub col_start: Vec<u32>,
    /// CSC row indices, sorted within each column.
    pub col_rows: Vec<u32>,
}

impl Columns {
    /// Number of variables.
    pub fn num_vars(&self) -> usize {
        self.col_start.len() - 1
    }

    /// Rows toggled by variable `v`.
    pub fn col(&self, v: usize) -> &[u32] {
        &self.col_rows[self.col_start[v] as usize..self.col_start[v + 1] as usize]
    }

    /// The same matrix read off a [`super::symphase::SymPhaseSampler`]
    /// (rows = its measurement or parity rows): the transpose of its CSR.
    pub fn from_symphase(s: &super::symphase::SymPhaseSampler) -> Columns {
        let rows = s.num_measurements();
        let nv = s.num_vars();
        let mut col_start = vec![0u32; nv + 1];
        for j in 0..rows {
            for &v in s.row(j) {
                col_start[v as usize + 1] += 1;
            }
        }
        for i in 0..nv {
            col_start[i + 1] += col_start[i];
        }
        let mut fill = col_start.clone();
        let mut col_rows = vec![0u32; col_start[nv] as usize];
        for j in 0..rows {
            for &v in s.row(j) {
                col_rows[fill[v as usize] as usize] = j as u32;
                fill[v as usize] += 1;
            }
        }
        Columns {
            rows,
            groups: s.groups().to_vec(),
            col_start,
            col_rows,
        }
    }
}

/// A sorted `u32` set stored in the arena: `data[off..off + len]`.
type Desc = (u32, u32);

const EMPTY: Desc = (0, 0);

/// The second operand of [`merge_append`].
#[derive(Clone, Copy)]
enum Operand<'a> {
    /// A set in the arena.
    Arena(Desc),
    /// A set outside the arena.
    Slice(&'a [u32]),
}

/// Appends `a xor b` (symmetric difference of two sorted sets; `a` in the
/// arena) to `data` and returns its descriptor. The merge loop is
/// branch-free apart from its exit test: the smaller head is written, the
/// output cursor only advances when the heads differ, and equal heads are
/// both consumed.
#[inline(always)]
fn merge_append(data: &mut Vec<u32>, a: Desc, b: Operand<'_>) -> Desc {
    let na = a.1 as usize;
    let nb = match b {
        Operand::Arena(d) => d.1 as usize,
        Operand::Slice(s) => s.len(),
    };
    data.reserve(na + nb);
    let start = data.len();
    // SAFETY: after the reserve, `data` has room for na + nb more entries
    // and does not reallocate below; `a` (and `b` when in the arena) lie in
    // the initialised part, all pointers derive from one base pointer, and
    // the output region [start, start + na + nb) is disjoint from both
    // inputs; set_len covers only written entries.
    unsafe {
        let base = data.as_mut_ptr();
        let ap = base.add(a.0 as usize) as *const u32;
        let bp = match b {
            Operand::Arena(d) => base.add(d.0 as usize) as *const u32,
            Operand::Slice(s) => s.as_ptr(),
        };
        let o = base.add(start);
        let (mut i, mut j, mut k) = (0usize, 0usize, 0usize);
        while i < na && j < nb {
            let (x, y) = (*ap.add(i), *bp.add(j));
            *o.add(k) = x.min(y);
            k += (x != y) as usize;
            i += (x <= y) as usize;
            j += (y <= x) as usize;
        }
        std::ptr::copy_nonoverlapping(ap.add(i), o.add(k), na - i);
        k += na - i;
        std::ptr::copy_nonoverlapping(bp.add(j), o.add(k), nb - j);
        k += nb - j;
        data.set_len(start + k);
        (start as u32, k as u32)
    }
}

/// Which Pauli a flip channel applies.
#[derive(Clone, Copy)]
enum Flip {
    X,
    Y,
    Z,
}

/// The backward sweep. Every set lives in one append-only arena `data`:
/// an update appends the new set and repoints the descriptor (the old
/// version is garbage), so sets are immutable once written. That makes an
/// empty destination free to *share* its source's storage, and recording a
/// column is a descriptor push, not a copy.
struct Sweep {
    data: Vec<u32>,
    /// `sens[2q]`: rows toggled by an X flip on `q` here; `sens[2q + 1]`:
    /// by a Z flip.
    sens: Vec<Desc>,
    /// Per measurement: rows that read it (plus feedback targets).
    pend: Vec<Desc>,
    /// Recorded groups and their columns, reverse program order.
    dists: Vec<VarDist>,
    cols: Vec<Desc>,
}

impl Sweep {
    fn new(num_qubits: usize, num_meas: usize) -> Sweep {
        Sweep {
            data: Vec::with_capacity(1 << 16),
            sens: vec![EMPTY; 2 * num_qubits],
            pend: vec![EMPTY; num_meas],
            dists: Vec::new(),
            cols: Vec::new(),
        }
    }

    /// The descriptor of `a xor b`, both in the arena.
    #[inline(always)]
    fn xor(&mut self, a: Desc, b: Desc) -> Desc {
        if b.1 == 0 {
            return a;
        }
        if a.1 == 0 {
            return b;
        }
        merge_append(&mut self.data, a, Operand::Arena(b))
    }

    /// `sens[dst] ^= sens[src]`.
    #[inline(always)]
    fn sens_xor(&mut self, dst: usize, src: usize) {
        self.sens[dst] = self.xor(self.sens[dst], self.sens[src]);
    }

    /// Inserts `v` into pending set `m`, or removes it if present.
    #[inline(always)]
    fn toggle(&mut self, m: usize, v: u32) {
        let a = self.pend[m];
        self.pend[m] = merge_append(&mut self.data, a, Operand::Slice(&[v]));
    }

    #[inline(always)]
    fn record(&mut self, dist: VarDist, cols: &[Desc]) {
        if cols.iter().all(|c| c.1 == 0) {
            return;
        }
        self.dists.push(dist);
        self.cols.extend_from_slice(cols);
    }

    // ------------------------------------------------------------ gates

    #[inline(always)]
    fn h(&mut self, q: usize) {
        self.sens.swap(2 * q, 2 * q + 1);
    }

    /// S or S^dagger: X -> Y, so `xs ^= zs`.
    #[inline(always)]
    fn s(&mut self, q: usize) {
        self.sens_xor(2 * q, 2 * q + 1);
    }

    /// sqrt(X) or its inverse: Z -> Y, so `zs ^= xs`.
    #[inline(always)]
    fn sx(&mut self, q: usize) {
        self.sens_xor(2 * q + 1, 2 * q);
    }

    #[inline(always)]
    fn cx(&mut self, c: usize, t: usize) {
        self.sens_xor(2 * c, 2 * t);
        self.sens_xor(2 * t + 1, 2 * c + 1);
    }

    #[inline(always)]
    fn cz(&mut self, a: usize, b: usize) {
        self.sens_xor(2 * a, 2 * b + 1);
        self.sens_xor(2 * b, 2 * a + 1);
    }

    #[inline(always)]
    fn swap(&mut self, a: usize, b: usize) {
        self.sens.swap(2 * a, 2 * b);
        self.sens.swap(2 * a + 1, 2 * b + 1);
    }

    /// Backward update by a Clifford gate (already validated). Gates made of
    /// several primitives undo them in reverse order.
    #[inline(always)]
    fn gate(&mut self, g: &Gate) {
        match *g {
            Gate::I(_) | Gate::X(_) | Gate::Y(_) | Gate::Z(_) => {}
            Gate::H(a) => self.h(a),
            Gate::S(a) | Gate::Sdg(a) => self.s(a),
            Gate::Sx(a) | Gate::Sxdg(a) => self.sx(a),
            Gate::Phase(a, t) | Gate::Rz(a, t) => {
                if (t / FRAC_PI_2).round().rem_euclid(2.0) == 1.0 {
                    self.s(a);
                }
            }
            Gate::Cnot(c, t) => self.cx(c, t),
            Gate::Cz(a, b) => self.cz(a, b),
            Gate::Swap(a, b) => self.swap(a, b),
            // forward: swap, cz, s(a), s(b) (as in the tableau)
            Gate::ISwap(a, b) | Gate::ISwapdg(a, b) => {
                self.s(b);
                self.s(a);
                self.cz(a, b);
                self.swap(a, b);
            }
            Gate::CPhase(a, b, t) => {
                if (t / PI).round().rem_euclid(2.0) == 1.0 {
                    self.cz(a, b);
                }
            }
            _ => unreachable!("validated by clifford_check"),
        }
    }

    // ------------------------------------------------------------ noise

    #[inline(always)]
    fn flip(&mut self, q: usize, which: Flip, p: f64) {
        let col = match which {
            Flip::X => self.sens[2 * q],
            Flip::Z => self.sens[2 * q + 1],
            Flip::Y => self.xor(self.sens[2 * q], self.sens[2 * q + 1]),
        };
        self.record(VarDist::Flip(p), &[col]);
    }

    #[inline(always)]
    fn depol1(&mut self, q: usize, p: f64) {
        let c = [self.sens[2 * q], self.sens[2 * q + 1]];
        self.record(VarDist::Depol1(p), &c);
    }

    #[inline(always)]
    fn depol2(&mut self, a: usize, b: usize, p: f64) {
        let c = [
            self.sens[2 * a],
            self.sens[2 * a + 1],
            self.sens[2 * b],
            self.sens[2 * b + 1],
        ];
        self.record(VarDist::Depol2(p), &c);
    }

    /// Z-basis measurement number `m` of `q` with readout flip `p`: forward
    /// order is [flip, outcome, coin], so the coin (a Z on `q` just after
    /// the outcome) is recorded first.
    #[inline(always)]
    fn measure(&mut self, q: usize, m: usize, p: f64) {
        let coin = self.sens[2 * q + 1];
        self.record(VarDist::Coin, &[coin]);
        let rows = self.pend[m];
        if rows.1 > 0 {
            if p > 0.0 {
                self.record(VarDist::Flip(p), &[rows]);
            }
            self.sens[2 * q] = self.xor(self.sens[2 * q], rows);
            self.pend[m] = EMPTY;
        }
    }

    /// Reset of `q` with flip `p`: forward order is [coin, flip].
    #[inline(always)]
    fn reset(&mut self, q: usize, p: f64) {
        if p > 0.0 {
            let x = self.sens[2 * q];
            self.record(VarDist::Flip(p), &[x]);
        }
        let z = self.sens[2 * q + 1];
        self.record(VarDist::Coin, &[z]);
        self.sens[2 * q] = EMPTY;
        self.sens[2 * q + 1] = EMPTY;
    }

    /// The initial coins (|0> is a Z eigenstate), recorded last so that
    /// they come first in program order.
    fn initial_coins(&mut self, n: usize) {
        for q in (0..n).rev() {
            let z = self.sens[2 * q + 1];
            self.record(VarDist::Coin, &[z]);
        }
    }

    /// Reverses the recording into program order.
    fn finish(self, rows: usize) -> Columns {
        let mut groups = Vec::with_capacity(self.dists.len());
        let mut col_start = Vec::with_capacity(self.cols.len() + 1);
        col_start.push(0u32);
        let total: usize = self.cols.iter().map(|c| c.1 as usize).sum();
        let mut col_rows = Vec::with_capacity(total);
        let mut ce = self.cols.len();
        let mut first = 0u32;
        for &dist in self.dists.iter().rev() {
            let k = dist.len();
            for &c in &self.cols[ce - k..ce] {
                col_rows.extend_from_slice(&self.data[c.0 as usize..(c.0 + c.1) as usize]);
                col_start.push(col_rows.len() as u32);
            }
            groups.push(VarGroup { first, dist });
            first += k as u32;
            ce -= k;
        }
        Columns {
            rows,
            groups,
            col_start,
            col_rows,
        }
    }
}

/// Accepts exactly the gates the stabilizer tableau accepts (and with the
/// same error), so that the compiler and `SymPhaseSampler::new` agree on
/// which circuits are valid.
fn clifford_check(g: &Gate, n: usize) -> Result<(), SimError> {
    check_gate(g, n)?;
    let ok = match *g {
        Gate::I(_)
        | Gate::H(_)
        | Gate::S(_)
        | Gate::Sdg(_)
        | Gate::X(_)
        | Gate::Y(_)
        | Gate::Z(_)
        | Gate::Sx(_)
        | Gate::Sxdg(_)
        | Gate::Cnot(..)
        | Gate::Cz(..)
        | Gate::Swap(..)
        | Gate::ISwap(..)
        | Gate::ISwapdg(..) => true,
        Gate::Phase(_, t) | Gate::Rz(_, t) => is_multiple_of_half_pi(t),
        Gate::CPhase(_, _, t) => is_multiple_of_half_pi(t / 2.0),
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(SimError::Unsupported {
            backend: "stabilizer",
            gate: *g,
        })
    }
}

/// Qubits of a 1- or 2-qubit gate without allocating.
#[inline(always)]
fn gate_qubits(g: &Gate) -> (usize, usize, usize) {
    match *g {
        Gate::Cnot(a, b)
        | Gate::Cz(a, b)
        | Gate::Swap(a, b)
        | Gate::ISwap(a, b)
        | Gate::ISwapdg(a, b)
        | Gate::CPhase(a, b, _) => (2, a, b),
        Gate::I(a)
        | Gate::H(a)
        | Gate::X(a)
        | Gate::Y(a)
        | Gate::Z(a)
        | Gate::S(a)
        | Gate::Sdg(a)
        | Gate::T(a)
        | Gate::Tdg(a)
        | Gate::Sx(a)
        | Gate::Sxdg(a)
        | Gate::Rx(a, _)
        | Gate::Ry(a, _)
        | Gate::Rz(a, _)
        | Gate::Phase(a, _)
        | Gate::U(a, ..) => (1, a, a),
        Gate::Ccx(..) => (3, 0, 0),
    }
}

fn check_qubit(q: usize, n: usize) -> Result<(), SimError> {
    if q >= n {
        Err(SimError::QubitOutOfRange {
            qubit: q,
            num_qubits: n,
        })
    } else {
        Ok(())
    }
}

/// Compiles `circuit` under `noise` (the semantics of
/// [`crate::Circuit::run_noisy`]) into the detector matrix of the rows
/// `detectors` then `observables` (absolute measurement indices), reported
/// relative to the noiseless reference (Stim's detection-event convention).
/// Equal to `Columns::from_symphase(&SymPhaseSampler::new(circuit,
/// noise)?.with_parities(&sets).relative_to_reference())`.
pub fn compile_circuit(
    circuit: &Circuit,
    noise: &NoiseModel,
    detectors: &[Vec<usize>],
    observables: &[Vec<usize>],
) -> Result<Columns, SimError> {
    let n = circuit.num_qubits;
    let num_meas = circuit
        .ops
        .iter()
        .filter(|o| matches!(o, Op::Measure(_)))
        .count();
    let mut sw = Sweep::new(n, num_meas);
    for (row, set) in detectors.iter().chain(observables).enumerate() {
        for &m in set {
            if m >= num_meas {
                return Err(SimError::ClassicalBitOutOfRange {
                    bit: m,
                    available: num_meas,
                });
            }
            sw.toggle(m, row as u32);
        }
    }
    let gate_noise = |arity: usize| -> Option<VarDist> {
        match arity {
            1 if noise.p_1q > 0.0 => Some(VarDist::Depol1(noise.p_1q)),
            2 if noise.p_2q > 0.0 => Some(VarDist::Depol2(noise.p_2q)),
            _ => None,
        }
    };
    let mut m = num_meas;
    for op in circuit.ops.iter().rev() {
        match *op {
            Op::Gate(g) => {
                clifford_check(&g, n)?;
                let (arity, a, b) = gate_qubits(&g);
                match gate_noise(arity) {
                    Some(VarDist::Depol1(p)) => sw.depol1(a, p),
                    Some(VarDist::Depol2(p)) => sw.depol2(a, b, p),
                    _ => {}
                }
                sw.gate(&g);
            }
            Op::Measure(q) => {
                check_qubit(q, n)?;
                m -= 1;
                sw.measure(q, m, noise.p_meas);
            }
            Op::Reset(q) => {
                check_qubit(q, n)?;
                sw.reset(q, noise.p_reset);
            }
            Op::ClassicControlled {
                gate, meas_index, ..
            } => {
                // a Pauli applied iff measurement `meas_index` flipped relative
                // to the reference: that measurement's flip also toggles what
                // the Pauli toggles here
                let (fx, fz, q) = match gate {
                    Gate::X(q) => (true, false, q),
                    Gate::Y(q) => (true, true, q),
                    Gate::Z(q) => (false, true, q),
                    g => {
                        return Err(SimError::Unsupported {
                            backend: "symphase",
                            gate: g,
                        })
                    }
                };
                if gate_noise(1).is_some() {
                    return Err(SimError::Unsupported {
                        backend: "symphase",
                        gate,
                    });
                }
                check_qubit(q, n)?;
                if meas_index >= m {
                    return Err(SimError::ClassicalBitOutOfRange {
                        bit: meas_index,
                        available: m,
                    });
                }
                let sens = match (fx, fz) {
                    (true, false) => sw.sens[2 * q],
                    (false, true) => sw.sens[2 * q + 1],
                    _ => sw.xor(sw.sens[2 * q], sw.sens[2 * q + 1]),
                };
                sw.pend[meas_index] = sw.xor(sw.pend[meas_index], sens);
            }
            Op::XFlip(q, p) | Op::YFlip(q, p) | Op::ZFlip(q, p) => {
                check_qubit(q, n)?;
                if p > 0.0 {
                    let which = match op {
                        Op::XFlip(..) => Flip::X,
                        Op::YFlip(..) => Flip::Y,
                        _ => Flip::Z,
                    };
                    sw.flip(q, which, p);
                }
            }
            Op::Depolarize1q(q, p) => {
                check_qubit(q, n)?;
                if p > 0.0 {
                    sw.depol1(q, p);
                }
            }
            Op::Depolarize2q(a, b, p) => {
                check_qubit(a, n)?;
                check_qubit(b, n)?;
                if p > 0.0 {
                    sw.depol2(a, b, p);
                }
            }
        }
    }
    sw.initial_coins(n);
    Ok(sw.finish(detectors.len() + observables.len()))
}

/// Compiles a parsed `.stim` program into the detector matrix of its
/// detectors then observables (detection events relative to the noiseless
/// reference, as Stim reports them). Same result as [`compile_circuit`] on
/// `prog.to_program()`, without unrolling the program into ops, and with
/// per-instruction readout-flip probabilities.
pub fn compile_stim(prog: &StimCircuit) -> Columns {
    compile_stim_timed(prog).0
}

/// [`compile_stim`] plus the wall time of the backward sweep and of the
/// final reversal into program order, in seconds (for the profile in
/// `research/qec/sampler-x.md`).
#[doc(hidden)]
pub fn compile_stim_timed(prog: &StimCircuit) -> (Columns, f64, f64) {
    let t0 = std::time::Instant::now();
    let n = prog.num_qubits();
    let nd = prog.num_detectors();
    let mut sw = Sweep::new(n, prog.num_measurements());
    prog.for_each_op_rev(|op| match op {
        // the parser only produces valid H S S_DAG X Y Z CX CZ SWAP on
        // distinct in-range qubits
        StimOp::Gate(g) => sw.gate(&g),
        StimOp::Measure { qubit, index, p } => sw.measure(qubit, index, p),
        StimOp::Reset(q) => sw.reset(q, 0.0),
        StimOp::XFlip(q, p) if p > 0.0 => sw.flip(q, Flip::X, p),
        StimOp::YFlip(q, p) if p > 0.0 => sw.flip(q, Flip::Y, p),
        StimOp::ZFlip(q, p) if p > 0.0 => sw.flip(q, Flip::Z, p),
        StimOp::Depolarize1(q, p) if p > 0.0 => sw.depol1(q, p),
        StimOp::Depolarize2(a, b, p) if p > 0.0 => sw.depol2(a, b, p),
        StimOp::Detector {
            row,
            base,
            lookbacks,
        } => {
            for &k in lookbacks {
                sw.toggle(base - k as usize, row as u32);
            }
        }
        StimOp::Observable {
            index,
            base,
            lookbacks,
        } => {
            for &k in lookbacks {
                sw.toggle(base - k as usize, (nd + index) as u32);
            }
        }
        _ => {}
    });
    sw.initial_coins(n);
    let t_sweep = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    let c = sw.finish(nd + prog.num_observables());
    (c, t_sweep, t1.elapsed().as_secs_f64())
}

/// Expected hits drawn per table entry at which building the padded hit
/// tables starts to pay off (see [`tables_pay_off`]).
pub const TABLE_PAYOFF_HITS_PER_ENTRY: f64 = 1.0;

/// Whether a run of `shots` shots should build the padded hit tables
/// (`FastSampler::from_columns(c, true)`) rather than sample through the
/// columns. Building costs about one write per (group, pattern) entry; every
/// hit sampled through a table instead of the columns saves a few
/// nanoseconds. Tables are built when the run is expected to draw at least
/// [`TABLE_PAYOFF_HITS_PER_ENTRY`] hits per entry. Only the speed depends on
/// this choice, never the distribution.
pub fn tables_pay_off(c: &Columns, shots: usize) -> bool {
    tables_pay_off_with(c, shots, TABLE_PAYOFF_HITS_PER_ENTRY)
}

/// [`tables_pay_off`] with the threshold (expected hits per table entry) as
/// a parameter, for calibration.
pub fn tables_pay_off_with(c: &Columns, shots: usize, hits_per_entry: f64) -> bool {
    use super::fast_sampler::hit_rate;
    // (m, p bits) -> number of rare groups
    let mut classes: Vec<(u64, u64, f64)> = Vec::new();
    for g in &c.groups {
        let (m, p) = match g.dist {
            VarDist::Flip(p) => (1, p),
            VarDist::Depol1(p) => (3, p),
            VarDist::Depol2(p) => (15, p),
            VarDist::Coin => continue,
        };
        if p <= 0.0 || p > 0.25 {
            continue;
        }
        match classes
            .iter_mut()
            .find(|(cm, cp, _)| *cm == m && *cp == p.to_bits())
        {
            Some(e) => e.2 += 1.0,
            None => classes.push((m, p.to_bits(), 1.0)),
        }
    }
    let (mut hits, mut entries) = (0.0, 0.0);
    for &(m, pb, n) in &classes {
        hits += n * hit_rate(f64::from_bits(pb), m);
        entries += n * m as f64;
    }
    shots as f64 * hits >= hits_per_entry * entries
}
