//! T-count optimisation of Clifford+T circuits by phase-polynomial
//! re-synthesis: phase folding into Hadamard-delimited slots, TODD on each
//! slot's signature tensor, and exact verification of every output.
//! Notebook: `research/compiler/todd.md`.
//!
//! **Model.** Every Hadamard stays where it is. Between Hadamards the
//! circuit is a CNOT/X/SWAP network with diagonal phases, so the whole
//! circuit is a sum over paths whose phase is a sum of terms
//! `k·(p·v)` (ω = e^{iπ/4}) over affine parities `p` of the *variables*
//! (the inputs and one variable per Hadamard). A term with odd `k` costs
//! one T gate. A parity is *available* in the slots (stretches between
//! consecutive Hadamards) where it is a XOR of the current wire values;
//! this is an interval of slots ([`Term::birth`]..=[`Term::death`]).
//!
//! **Algorithm** ([`optimize`]).
//! 1. [`scan`]: merge equal parities (phase folding, Amy–Maslov–Mosca)
//!    and compute each term's availability interval.
//! 2. Assign every odd term to one slot inside its interval (greedy
//!    interval stabbing, then optional local search).
//! 3. In each slot, write the assigned odd terms as columns `p_j` in the
//!    coordinates of the wire values there and minimise their number with
//!    [`tensor::todd`] (randomised restarts on request), adding the
//!    diagonal Clifford correction [`tensor::clifford_correction`].
//! 4. Re-emit: the Hadamard/CNOT/X/SWAP skeleton unchanged, each slot's
//!    terms as `CNOT·T^k·CNOT` parity blocks, even (Clifford) terms at
//!    their first occurrence.
//!
//! **Verification** ([`verify`]): the output must have the same path sum
//! as the input (exact, all inputs, polynomial time), checked by code that
//! does not share the optimiser's term bookkeeping; tests add exact
//! basis-state simulation over `Z[ω]` and the state-vector engine.

pub mod gf2;
pub mod pauli;
pub mod tensor;
pub mod verify;

use crate::circuit::{Circuit, Op};
use crate::gate::Gate;
use crate::io::qc::{QcCircuit, QcGate};
use gf2::{Bits, SpanSolver};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt;
use tensor::{clifford_correction, todd, ToddParams};
use verify::VGate;

/// Errors of the T-count optimiser.
#[derive(Clone, Debug, PartialEq)]
pub enum ToddError {
    /// The circuit contains a gate outside `{H, X, Y, Z, S, S†, T, T†,
    /// CNOT, CZ, CCZ, Toffoli, SWAP}` (or a non-unitary operation).
    Unsupported(String),
    /// An output failed verification (a bug; never returned for a correct
    /// optimiser run).
    NotEquivalent(String),
}

impl fmt::Display for ToddError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToddError::Unsupported(s) => write!(f, "T-count optimiser: unsupported {s}"),
            ToddError::NotEquivalent(s) => write!(f, "T-count optimiser: verification failed: {s}"),
        }
    }
}

impl std::error::Error for ToddError {}

impl From<ToddError> for crate::error::Error {
    fn from(e: ToddError) -> Self {
        crate::error::Error::Parse(e.to_string())
    }
}

/// A gate of the optimiser's input representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PGate {
    /// Hadamard (kept in place by the optimiser).
    H(usize),
    /// Pauli X.
    X(usize),
    /// `Cnot(control, target)`.
    Cnot(usize, usize),
    /// SWAP.
    Swap(usize, usize),
    /// `diag(1, ω^k)` with `k` mod 8 (T = 1, S = 2, Z = 4, S† = 6, T† = 7).
    Phase(usize, u8),
    /// Controlled Z.
    Cz(usize, usize),
    /// Doubly controlled Z.
    Ccz(usize, usize, usize),
}

/// A circuit over [`PGate`] times a global phase `ω^global`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhaseCircuit {
    /// Number of qubits.
    pub num_qubits: usize,
    /// Gates in time order.
    pub gates: Vec<PGate>,
    /// Global phase in units of π/4.
    pub global: u8,
}

impl PhaseCircuit {
    /// From a `.qc` circuit (Toffoli = `H·CCZ·H` on the target).
    pub fn from_qc(c: &QcCircuit) -> Self {
        let mut out = PhaseCircuit {
            num_qubits: c.num_qubits(),
            gates: Vec::with_capacity(c.gates.len()),
            global: 0,
        };
        for g in &c.gates {
            match *g {
                QcGate::H(q) => out.gates.push(PGate::H(q)),
                QcGate::X(q) => out.gates.push(PGate::X(q)),
                QcGate::Y(q) => {
                    out.gates.push(PGate::Phase(q, 4));
                    out.gates.push(PGate::X(q));
                    out.global = (out.global + 2) % 8;
                }
                QcGate::Z(q) => out.gates.push(PGate::Phase(q, 4)),
                QcGate::S(q) => out.gates.push(PGate::Phase(q, 2)),
                QcGate::Sdg(q) => out.gates.push(PGate::Phase(q, 6)),
                QcGate::T(q) => out.gates.push(PGate::Phase(q, 1)),
                QcGate::Tdg(q) => out.gates.push(PGate::Phase(q, 7)),
                QcGate::Cnot(a, b) => out.gates.push(PGate::Cnot(a, b)),
                QcGate::Cz(a, b) => out.gates.push(PGate::Cz(a, b)),
                QcGate::Ccz(a, b, t) => out.gates.push(PGate::Ccz(a, b, t)),
                QcGate::Toffoli(a, b, t) => {
                    out.gates.push(PGate::H(t));
                    out.gates.push(PGate::Ccz(a, b, t));
                    out.gates.push(PGate::H(t));
                }
                QcGate::Swap(a, b) => out.gates.push(PGate::Swap(a, b)),
            }
        }
        out
    }

    /// From a [`Circuit`] over `{I, H, X, Y, Z, S, S†, T, T†, CNOT, CZ,
    /// SWAP, Toffoli}`; `Phase`/`Rz` angles must be multiples of π/4.
    pub fn from_circuit(c: &Circuit) -> Result<Self, ToddError> {
        let mut out = PhaseCircuit {
            num_qubits: c.num_qubits,
            gates: Vec::with_capacity(c.ops.len()),
            global: 0,
        };
        let units = |t: f64| -> Option<u8> {
            let k = (t / std::f64::consts::FRAC_PI_4).round();
            ((t - k * std::f64::consts::FRAC_PI_4).abs() < 1e-12)
                .then(|| (k as i64).rem_euclid(8) as u8)
        };
        for op in &c.ops {
            let Op::Gate(g) = op else {
                return Err(ToddError::Unsupported(format!("operation {op:?}")));
            };
            match *g {
                Gate::I(_) => {}
                Gate::H(q) => out.gates.push(PGate::H(q)),
                Gate::X(q) => out.gates.push(PGate::X(q)),
                Gate::Y(q) => {
                    out.gates.push(PGate::Phase(q, 4));
                    out.gates.push(PGate::X(q));
                    out.global = (out.global + 2) % 8;
                }
                Gate::Z(q) => out.gates.push(PGate::Phase(q, 4)),
                Gate::S(q) => out.gates.push(PGate::Phase(q, 2)),
                Gate::Sdg(q) => out.gates.push(PGate::Phase(q, 6)),
                Gate::T(q) => out.gates.push(PGate::Phase(q, 1)),
                Gate::Tdg(q) => out.gates.push(PGate::Phase(q, 7)),
                Gate::Phase(q, t) => match units(t) {
                    Some(k) => out.gates.push(PGate::Phase(q, k)),
                    None => return Err(ToddError::Unsupported(format!("gate {g:?}"))),
                },
                Gate::Rz(q, t) => match (units(t), units(-t / 2.0)) {
                    // Rz(t) = e^{-it/2} diag(1, e^{it})
                    (Some(k), Some(gk)) => {
                        out.gates.push(PGate::Phase(q, k));
                        out.global = (out.global + gk) % 8;
                    }
                    _ => return Err(ToddError::Unsupported(format!("gate {g:?}"))),
                },
                Gate::Cnot(a, b) => out.gates.push(PGate::Cnot(a, b)),
                Gate::Cz(a, b) => out.gates.push(PGate::Cz(a, b)),
                Gate::Swap(a, b) => out.gates.push(PGate::Swap(a, b)),
                Gate::Ccx(a, b, t) => {
                    out.gates.push(PGate::H(t));
                    out.gates.push(PGate::Ccz(a, b, t));
                    out.gates.push(PGate::H(t));
                }
                other => return Err(ToddError::Unsupported(format!("gate {other:?}"))),
            }
        }
        Ok(out)
    }

    /// T-count: odd phases count 1, each `CCZ` counts 7.
    pub fn t_count(&self) -> usize {
        self.gates
            .iter()
            .map(|g| match *g {
                PGate::Phase(_, k) if k % 2 == 1 => 1,
                PGate::Ccz(..) => 7,
                _ => 0,
            })
            .sum()
    }

    /// Number of Hadamard gates.
    pub fn hadamard_count(&self) -> usize {
        self.gates
            .iter()
            .filter(|g| matches!(g, PGate::H(_)))
            .count()
    }

    /// The gates in the checker's representation.
    pub fn to_vgates(&self) -> Vec<VGate> {
        let mut v: Vec<VGate> = self
            .gates
            .iter()
            .map(|g| match *g {
                PGate::H(q) => VGate::H(q),
                PGate::X(q) => VGate::X(q),
                PGate::Cnot(a, b) => VGate::Cnot(a, b),
                PGate::Swap(a, b) => VGate::Swap(a, b),
                PGate::Phase(q, k) => VGate::Phase(q, k),
                PGate::Cz(a, b) => VGate::Cz(a, b),
                PGate::Ccz(a, b, c) => VGate::Ccz(a, b, c),
            })
            .collect();
        if self.global != 0 {
            v.push(VGate::Global(self.global));
        }
        v
    }

    /// The circuit as a [`Circuit`] (each `CCZ` as the standard 7-T
    /// decomposition) and its global phase in units of π/4 (`self =
    /// ω^global · circuit`).
    pub fn to_circuit(&self) -> (Circuit, u8) {
        let mut c = Circuit::new(self.num_qubits);
        for g in &self.gates {
            match *g {
                PGate::H(q) => {
                    c.h(q);
                }
                PGate::X(q) => {
                    c.x(q);
                }
                PGate::Cnot(a, b) => {
                    c.cnot(a, b);
                }
                PGate::Swap(a, b) => {
                    c.swap(a, b);
                }
                PGate::Cz(a, b) => {
                    c.cz(a, b);
                }
                PGate::Phase(q, k) => push_phase(&mut c, q, k),
                PGate::Ccz(a, b, t) => {
                    for (set, k) in [
                        (&[a][..], 1u8),
                        (&[b][..], 1),
                        (&[t][..], 1),
                        (&[a, b][..], 7),
                        (&[a, t][..], 7),
                        (&[b, t][..], 7),
                        (&[a, b, t][..], 1),
                    ] {
                        push_parity_phase(&mut c, set, k);
                    }
                }
            }
        }
        (c, self.global)
    }

    /// Removes Hadamards around CNOT/SWAP blocks with the identity
    /// `H^{⊗W} · B · H^{⊗W} = B'`, where `B'` is the block with every
    /// CNOT reversed (`W` = wires of the block). With `S` (resp. `T`) the
    /// wires of `W` whose neighbouring gate before (resp. after) the block
    /// is a Hadamard, `[H_S] B [H_T] = [H_{W∖S}] B' [H_{W∖T}]`; the rewrite
    /// is applied when it removes Hadamards (`|S| + |T| > |W|`), to a
    /// fixed point, followed by [`Self::cancel_hadamard_pairs`]. Returns
    /// the number of Hadamards removed.
    pub fn reduce_hadamards(&mut self) -> usize {
        let before = self.hadamard_count();
        loop {
            let mut changed = self.cancel_hadamard_pairs() > 0;
            changed |= self.hadamard_target_rule() > 0;
            let n = self.num_qubits;
            let g = &self.gates;
            let mut i = 0;
            #[allow(clippy::type_complexity)]
            let mut rewrite: Option<(
                usize,
                usize,
                Vec<usize>,
                Vec<usize>,
                Vec<usize>,
            )> = None;
            while i < g.len() {
                if !matches!(g[i], PGate::Cnot(..) | PGate::Swap(..)) {
                    i += 1;
                    continue;
                }
                let mut j = i;
                let mut wires = vec![false; n];
                while j < g.len() {
                    match g[j] {
                        PGate::Cnot(a, b) | PGate::Swap(a, b) => {
                            wires[a] = true;
                            wires[b] = true;
                            j += 1;
                        }
                        _ => break,
                    }
                }
                // block g[i..j]; neighbouring gates on each wire
                let w: Vec<usize> = (0..n).filter(|&q| wires[q]).collect();
                let touches = |gate: &PGate, q: usize| -> bool {
                    match *gate {
                        PGate::H(a) | PGate::X(a) | PGate::Phase(a, _) => a == q,
                        PGate::Cnot(a, b) | PGate::Swap(a, b) | PGate::Cz(a, b) => a == q || b == q,
                        PGate::Ccz(a, b, c) => a == q || b == q || c == q,
                    }
                };
                let mut s_idx = Vec::new();
                let mut t_idx = Vec::new();
                for &q in &w {
                    if let Some(p) = (0..i).rev().find(|&p| touches(&g[p], q)) {
                        if g[p] == PGate::H(q) {
                            s_idx.push(p);
                        }
                    }
                    if let Some(p) = (j..g.len()).find(|&p| touches(&g[p], q)) {
                        if g[p] == PGate::H(q) {
                            t_idx.push(p);
                        }
                    }
                }
                if s_idx.len() + t_idx.len() > w.len() {
                    rewrite = Some((i, j, w, s_idx, t_idx));
                    break;
                }
                i = j;
            }
            if let Some((i, j, w, s_idx, t_idx)) = rewrite {
                let s_wires: Vec<usize> = s_idx
                    .iter()
                    .map(|&p| match self.gates[p] {
                        PGate::H(q) => q,
                        _ => unreachable!(),
                    })
                    .collect();
                let t_wires: Vec<usize> = t_idx
                    .iter()
                    .map(|&p| match self.gates[p] {
                        PGate::H(q) => q,
                        _ => unreachable!(),
                    })
                    .collect();
                let mut block: Vec<PGate> = Vec::new();
                for &q in w.iter().filter(|q| !s_wires.contains(q)) {
                    block.push(PGate::H(q));
                }
                for gate in &self.gates[i..j] {
                    block.push(match *gate {
                        PGate::Cnot(a, b) => PGate::Cnot(b, a),
                        other => other,
                    });
                }
                for &q in w.iter().filter(|q| !t_wires.contains(q)) {
                    block.push(PGate::H(q));
                }
                let mut out = Vec::with_capacity(self.gates.len());
                for (p, &gate) in self.gates.iter().enumerate() {
                    if p == i {
                        out.extend(block.iter().copied());
                    }
                    if (i..j).contains(&p) || s_idx.contains(&p) || t_idx.contains(&p) {
                        continue;
                    }
                    out.push(gate);
                }
                self.gates = out;
                changed = true;
            }
            if !changed {
                break;
            }
        }
        before - self.hadamard_count()
    }

    /// Removes pairs of Hadamards on a wire `t` when every gate between
    /// them that touches `t` is a CNOT targeting `t` or an `X` on `t`:
    /// `H·CNOT(c,t)·H = CZ(c,t)` and `H·X·H = Z`. Returns the number of
    /// pairs removed.
    pub fn hadamard_target_rule(&mut self) -> usize {
        let n = self.num_qubits;
        let mut removed = 0;
        let mut open: Vec<Option<usize>> = vec![None; n];
        let mut ok = vec![false; n];
        let mut i = 0;
        while i < self.gates.len() {
            let g = self.gates[i];
            match g {
                PGate::H(t) => {
                    if let (Some(start), true) = (open[t], ok[t]) {
                        // rewrite gates start+1..i on wire t
                        for p in start + 1..i {
                            self.gates[p] = match self.gates[p] {
                                PGate::Cnot(c, tt) if tt == t => PGate::Cz(c, t),
                                PGate::X(tt) if tt == t => PGate::Phase(t, 4),
                                other => other,
                            };
                        }
                        self.gates.remove(i);
                        self.gates.remove(start);
                        removed += 1;
                        // indices shifted; restart the scan
                        open = vec![None; n];
                        ok = vec![false; n];
                        i = 0;
                        continue;
                    }
                    open[t] = Some(i);
                    ok[t] = true;
                }
                PGate::Cnot(c, t) => {
                    ok[c] = false;
                    open[c] = None;
                    let _ = t; // a CNOT target keeps the rule applicable
                }
                PGate::X(_) => {}
                PGate::Phase(q, _) => {
                    ok[q] = false;
                    open[q] = None;
                }
                PGate::Swap(a, b) | PGate::Cz(a, b) => {
                    for q in [a, b] {
                        ok[q] = false;
                        open[q] = None;
                    }
                }
                PGate::Ccz(a, b, c) => {
                    for q in [a, b, c] {
                        ok[q] = false;
                        open[q] = None;
                    }
                }
            }
            i += 1;
        }
        removed
    }

    /// Cancels pairs of Hadamards on one wire with no gate on that wire
    /// in between. Returns how many pairs were removed.
    pub fn cancel_hadamard_pairs(&mut self) -> usize {
        let n = self.num_qubits;
        let mut keep: Vec<Option<PGate>> = Vec::with_capacity(self.gates.len());
        let mut stack: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut removed = 0;
        for &g in &self.gates {
            let qs: Vec<usize> = match g {
                PGate::H(q) | PGate::X(q) | PGate::Phase(q, _) => vec![q],
                PGate::Cnot(a, b) | PGate::Swap(a, b) | PGate::Cz(a, b) => vec![a, b],
                PGate::Ccz(a, b, c) => vec![a, b, c],
            };
            if let PGate::H(q) = g {
                if let Some(&i) = stack[q].last() {
                    if keep[i] == Some(PGate::H(q)) {
                        keep[i] = None;
                        stack[q].pop();
                        removed += 1;
                        continue;
                    }
                }
            }
            let idx = keep.len();
            keep.push(Some(g));
            for q in qs {
                stack[q].push(idx);
            }
        }
        self.gates = keep.into_iter().flatten().collect();
        removed
    }
}

/// Appends `diag(1, ω^k)` on `q` as Clifford+T gates (one T/T† when `k`
/// is odd).
fn push_phase(c: &mut Circuit, q: usize, k: u8) {
    let gs: &[Gate] = match k % 8 {
        0 => &[],
        1 => &[Gate::T(q)],
        2 => &[Gate::S(q)],
        3 => &[Gate::S(q), Gate::T(q)],
        4 => &[Gate::Z(q)],
        5 => &[Gate::Z(q), Gate::T(q)],
        6 => &[Gate::Sdg(q)],
        _ => &[Gate::Tdg(q)],
    };
    for &g in gs {
        c.gate(g);
    }
}

/// Appends `ω^{k·(⊕_{w ∈ wires} u_w)}` as a CNOT ladder onto the first
/// wire, the phase, and the ladder undone.
fn push_parity_phase(c: &mut Circuit, wires: &[usize], k: u8) {
    if k.is_multiple_of(8) || wires.is_empty() {
        return;
    }
    let t = wires[0];
    for &w in &wires[1..] {
        c.cnot(w, t);
    }
    push_phase(c, t, k);
    for &w in wires[1..].iter().rev() {
        c.cnot(w, t);
    }
}

/// A merged phase term of the scan.
#[derive(Clone, Debug)]
pub struct Term {
    /// Linear parity over the variables (inputs, then one per Hadamard).
    pub parity: Bits,
    /// Coefficient mod 8 of `ω^{coef·(parity·v)}`.
    pub coef: u8,
    /// Index of the gate where the parity first occurs.
    pub first_gate: usize,
    /// First slot where the parity is available.
    pub birth: usize,
    /// Last slot where the parity is available.
    pub death: usize,
}

/// Result of [`scan`].
#[derive(Clone, Debug)]
pub struct Scan {
    /// Number of variables (qubits + Hadamards).
    pub nvars: usize,
    /// Merged terms (including ones whose coefficient cancelled to 0).
    pub terms: Vec<Term>,
    /// Global phase collected from constant parities, units of π/4.
    pub global: u8,
    /// Gate index of the `s`-th Hadamard; slot `s` ends just before it.
    pub hadamards: Vec<usize>,
}

/// Phase folding with availability intervals (see the module docs).
pub fn scan(c: &PhaseCircuit) -> Scan {
    let n = c.num_qubits;
    let nh = c.hadamard_count();
    let nv = n + nh;
    let mut lin: Vec<Bits> = (0..n).map(|q| Bits::unit(nv, q)).collect();
    let mut cst = vec![false; n];
    let mut terms: Vec<Term> = Vec::new();
    let mut coords: Vec<Bits> = Vec::new();
    let mut live: Vec<usize> = Vec::new();
    let mut index: HashMap<Bits, usize> = HashMap::new();
    let mut global = c.global;
    let mut hadamards = Vec::with_capacity(nh);
    let birth_of = |p: &Bits| -> usize {
        match p.last_one() {
            Some(i) if i >= n => i - n + 1,
            _ => 0,
        }
    };
    #[allow(clippy::too_many_arguments)]
    fn add(
        gi: usize,
        p: Bits,
        coord: Bits,
        cst1: bool,
        k: u8,
        global: &mut u8,
        terms: &mut Vec<Term>,
        coords: &mut Vec<Bits>,
        live: &mut Vec<usize>,
        index: &mut HashMap<Bits, usize>,
        birth: usize,
    ) {
        let k = k % 8;
        if k == 0 {
            return;
        }
        if p.is_zero() {
            if cst1 {
                *global = (*global + k) % 8;
            }
            return;
        }
        let kk = if cst1 {
            *global = (*global + k) % 8;
            (8 - k) % 8
        } else {
            k
        };
        match index.get(&p) {
            Some(&id) => terms[id].coef = (terms[id].coef + kk) % 8,
            None => {
                let id = terms.len();
                terms.push(Term {
                    parity: p.clone(),
                    coef: kk,
                    first_gate: gi,
                    birth,
                    death: usize::MAX,
                });
                coords.push(coord);
                live.push(id);
                index.insert(p, id);
            }
        }
    }
    let mut slot = 0usize;
    for (gi, g) in c.gates.iter().enumerate() {
        match *g {
            PGate::X(q) => cst[q] = !cst[q],
            PGate::Cnot(a, t) => {
                let s = lin[a].clone();
                lin[t].xor_with(&s);
                cst[t] ^= cst[a];
                for &id in &live {
                    if coords[id].get(t) {
                        coords[id].flip(a);
                    }
                }
            }
            PGate::Swap(a, t) => {
                lin.swap(a, t);
                cst.swap(a, t);
                for &id in &live {
                    let (x, y) = (coords[id].get(a), coords[id].get(t));
                    coords[id].set(a, y);
                    coords[id].set(t, x);
                }
            }
            PGate::H(q) => {
                live.retain(|&id| {
                    if coords[id].get(q) {
                        terms[id].death = slot;
                        false
                    } else {
                        true
                    }
                });
                lin[q] = Bits::unit(nv, n + slot);
                cst[q] = false;
                hadamards.push(gi);
                slot += 1;
            }
            PGate::Phase(q, k) => {
                let p = lin[q].clone();
                let b = birth_of(&p);
                add(
                    gi,
                    p,
                    Bits::unit(n, q),
                    cst[q],
                    k,
                    &mut global,
                    &mut terms,
                    &mut coords,
                    &mut live,
                    &mut index,
                    b,
                );
            }
            PGate::Cz(a, t) | PGate::Ccz(a, t, _) => {
                let set: Vec<(Vec<usize>, u8)> = match *g {
                    PGate::Cz(..) => vec![(vec![a], 2), (vec![t], 2), (vec![a, t], 6)],
                    PGate::Ccz(_, _, r) => vec![
                        (vec![a], 1),
                        (vec![t], 1),
                        (vec![r], 1),
                        (vec![a, t], 7),
                        (vec![a, r], 7),
                        (vec![t, r], 7),
                        (vec![a, t, r], 1),
                    ],
                    _ => unreachable!(),
                };
                for (wires, k) in set {
                    let mut p = Bits::zeros(nv);
                    let mut co = Bits::zeros(n);
                    let mut c1 = false;
                    for &w in &wires {
                        p.xor_with(&lin[w]);
                        co.set(w, true);
                        c1 ^= cst[w];
                    }
                    let b = birth_of(&p);
                    add(
                        gi,
                        p,
                        co,
                        c1,
                        k,
                        &mut global,
                        &mut terms,
                        &mut coords,
                        &mut live,
                        &mut index,
                        b,
                    );
                }
            }
        }
    }
    for &id in &live {
        terms[id].death = slot;
    }
    Scan {
        nvars: nv,
        terms,
        global,
        hadamards,
    }
}

/// Options of [`optimize`].
#[derive(Clone, Debug)]
pub struct ToddOptions {
    /// Run TODD at all (false: phase folding into slots only).
    pub todd: bool,
    /// Randomised TODD restarts per slot, best kept (0: one deterministic run).
    pub restarts: usize,
    /// Seed of the randomised restarts and the local search.
    pub seed: u64,
    /// Local-search passes that move odd terms between the slots they are
    /// available in (0: greedy assignment only).
    pub reassign_passes: usize,
    /// Wall-clock budget of the local search, in seconds.
    pub reassign_seconds: f64,
    /// Pauli-frame mode only: push Clifford rotations created by merging
    /// to the end (conjugating later rotations) and merge again.
    pub absorb_cliffords: bool,
    /// Rounds of large-neighbourhood search ([`tensor::lns`]) per group
    /// after the restarts (0: none).
    pub lns_rounds: usize,
}

impl Default for ToddOptions {
    fn default() -> Self {
        ToddOptions {
            todd: true,
            restarts: 0,
            seed: 1,
            reassign_passes: 0,
            reassign_seconds: 60.0,
            absorb_cliffords: true,
            lns_rounds: 0,
        }
    }
}

/// What [`optimize`] did.
#[derive(Clone, Debug, Default)]
pub struct ToddReport {
    /// T-count of the input.
    pub t_input: usize,
    /// T-count after phase folding (odd merged terms).
    pub t_folded: usize,
    /// T-count of the output circuit.
    pub t_output: usize,
    /// Number of Hadamards (unchanged).
    pub hadamards: usize,
    /// Slots that received odd terms, with `(slot, terms in, T out)`.
    pub groups: Vec<(usize, usize, usize)>,
    /// CNOT count of the output.
    pub cnots: usize,
    /// Global phase in units of π/4: `input = ω^global_phase · output`.
    pub global_phase: u8,
}

/// A slot group after TODD: the terms to emit as `(coordinates over the
/// wires, coefficient)`, and the global phase picked up.
struct GroupOut {
    terms: Vec<(Bits, u8)>,
    global: u8,
    t_out: usize,
}

/// Wire coordinates of the odd terms of one slot: `(columns, extra even
/// terms, global phase)`.
fn slot_columns(
    ids: &[usize],
    sc: &Scan,
    lin: &[Bits],
    cst: &[bool],
) -> (Vec<Bits>, Vec<(Bits, u8)>, u8) {
    let n = lin.len();
    let solver = SpanSolver::new(lin);
    let mut cstb = Bits::zeros(n);
    for (q, &c) in cst.iter().enumerate() {
        cstb.set(q, c);
    }
    let mut cols = Vec::with_capacity(ids.len());
    let mut extra = Vec::new();
    let mut global = 0u8;
    for &id in ids {
        let t = &sc.terms[id];
        let c = solver
            .solve(&t.parity)
            .expect("an assigned term is available in its slot");
        let k = if c.dot(&cstb) {
            global = (global + t.coef) % 8;
            (8 - t.coef) % 8
        } else {
            t.coef
        };
        debug_assert!(k % 2 == 1);
        if k != 1 {
            extra.push((c.clone(), (k + 7) % 8));
        }
        cols.push(c);
    }
    (cols, extra, global)
}

/// Deterministic TODD T-count of one slot group.
fn group_cost(ids: &[usize], sc: &Scan, snap: &(Vec<Bits>, Vec<bool>), n: usize) -> usize {
    if ids.is_empty() {
        return 0;
    }
    let (cols, _, _) = slot_columns(ids, sc, &snap.0, &snap.1);
    todd(
        cols,
        n,
        &ToddParams::default(),
        &mut StdRng::seed_from_u64(0),
    )
    .len()
}

/// First-improvement local search over the assignment of odd terms to
/// cut slots (see [`ToddOptions::reassign_passes`]).
fn reassign(
    sc: &Scan,
    groups: &mut HashMap<usize, Vec<usize>>,
    cuts: &[usize],
    snaps: &HashMap<usize, (Vec<Bits>, Vec<bool>)>,
    n: usize,
    opts: &ToddOptions,
) {
    use rand::seq::SliceRandom;
    let start = std::time::Instant::now();
    let mut rng = StdRng::seed_from_u64(opts.seed ^ 0xA55A_5AA5);
    let mut cost: HashMap<usize, usize> = cuts
        .par_iter()
        .map(|&s| (s, group_cost(&groups[&s], sc, &snaps[&s], n)))
        .collect();
    let mut slot_of: HashMap<usize, usize> = HashMap::new();
    for (&s, ids) in groups.iter() {
        for &i in ids {
            slot_of.insert(i, s);
        }
    }
    let cands_of = |t: usize| -> Vec<usize> {
        cuts.iter()
            .copied()
            .filter(|&s| sc.terms[t].birth <= s && s <= sc.terms[t].death)
            .collect()
    };
    let movable: Vec<usize> = slot_of
        .keys()
        .copied()
        .filter(|&t| cands_of(t).len() >= 2)
        .collect();
    let mut movable = movable;
    movable.sort_unstable();
    for _ in 0..opts.reassign_passes {
        let mut improved = false;
        let mut order = movable.clone();
        order.shuffle(&mut rng);
        for &t in &order {
            if start.elapsed().as_secs_f64() > opts.reassign_seconds {
                return;
            }
            let from = slot_of[&t];
            let cands: Vec<usize> = cands_of(t).into_iter().filter(|&s| s != from).collect();
            let mut a = groups[&from].clone();
            a.retain(|&x| x != t);
            let ca = group_cost(&a, sc, &snaps[&from], n);
            let best = cands
                .par_iter()
                .map(|&s| {
                    let mut b = groups[&s].clone();
                    b.push(t);
                    b.sort_unstable();
                    let cb = group_cost(&b, sc, &snaps[&s], n);
                    (cb as i64 - cost[&s] as i64, s, cb)
                })
                .min();
            if let Some((db, s, cb)) = best {
                let delta = db + ca as i64 - cost[&from] as i64;
                if delta < 0 {
                    groups.insert(from, a);
                    let g = groups.get_mut(&s).expect("cut slot");
                    g.push(t);
                    g.sort_unstable();
                    cost.insert(from, ca);
                    cost.insert(s, cb);
                    slot_of.insert(t, s);
                    improved = true;
                }
            }
        }
        if !improved {
            break;
        }
    }
    groups.retain(|_, v| !v.is_empty());
}

/// Best TODD result over `restarts` randomised runs plus one deterministic run.
fn best_todd(cols: &[Bits], d: usize, restarts: usize, lns_rounds: usize, seed: u64) -> Vec<Bits> {
    let first = best_of_restarts(cols, d, restarts, seed);
    if lns_rounds == 0 || first.len() < 2 {
        return first;
    }
    // independent LNS chains in parallel, best kept
    let chains = rayon::current_num_threads().clamp(1, 8);
    (0..chains)
        .into_par_iter()
        .map(|c| {
            let mut rng = StdRng::seed_from_u64(seed ^ 0x51ED_270B_2738_6A5B ^ (c as u64) << 32);
            tensor::lns(first.clone(), d, lns_rounds, &mut rng)
        })
        .min_by_key(|v| v.len())
        .unwrap_or(first)
}

/// Best of one deterministic TODD run and `restarts` randomised ones.
fn best_of_restarts(cols: &[Bits], d: usize, restarts: usize, seed: u64) -> Vec<Bits> {
    let det = todd(
        cols.to_vec(),
        d,
        &ToddParams::default(),
        &mut StdRng::seed_from_u64(seed),
    );
    if restarts == 0 {
        return det;
    }
    let best = (0..restarts)
        .into_par_iter()
        .map(|r| {
            let mut rng =
                StdRng::seed_from_u64(seed ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul(r as u64 + 1)));
            let p = ToddParams {
                randomize: true,
                ..Default::default()
            };
            todd(cols.to_vec(), d, &p, &mut rng)
        })
        .min_by_key(|v| v.len())
        .expect("restarts > 0");
    if best.len() < det.len() {
        best
    } else {
        det
    }
}

fn group_terms(
    cols: &[Bits],
    new: Vec<Bits>,
    extra: Vec<(Bits, u8)>,
    d: usize,
) -> (Vec<(Bits, u8)>, usize) {
    let corr = clifford_correction(cols, &new, d).expect("TODD keeps the signature tensor");
    let mut map: HashMap<Bits, u8> = HashMap::new();
    let t_out = new.len();
    for c in new {
        *map.entry(c).or_insert(0) += 1;
    }
    for (c, k) in corr.into_iter().chain(extra) {
        let e = map.entry(c).or_insert(0);
        *e = (*e + k) % 8;
    }
    let mut v: Vec<(Bits, u8)> = map.into_iter().filter(|(_, k)| k % 8 != 0).collect();
    v.sort();
    (v, t_out)
}

/// Optimises the T-count of `c`; returns the output circuit and a report.
/// The output keeps every Hadamard and is exactly equal to the input up to
/// the reported global phase (checked by [`verify_output`]).
pub fn optimize(c: &PhaseCircuit, opts: &ToddOptions) -> (Circuit, ToddReport) {
    let n = c.num_qubits;
    let sc = scan(c);
    let nh = sc.hadamards.len();
    let odd: Vec<usize> = (0..sc.terms.len())
        .filter(|&i| sc.terms[i].coef % 2 == 1)
        .collect();
    let even: Vec<usize> = (0..sc.terms.len())
        .filter(|&i| sc.terms[i].coef.is_multiple_of(2) && sc.terms[i].coef != 0)
        .collect();

    // Greedy interval stabbing: cut at the earliest death.
    let mut order = odd.clone();
    order.sort_by_key(|&i| (sc.terms[i].death, sc.terms[i].birth));
    let mut slot_of: HashMap<usize, usize> = HashMap::new();
    let mut last: Option<usize> = None;
    for &i in &order {
        let t = &sc.terms[i];
        match last {
            Some(cut) if cut >= t.birth => {
                slot_of.insert(i, cut);
            }
            _ => {
                last = Some(t.death);
                slot_of.insert(i, t.death);
            }
        }
    }
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (&i, &s) in &slot_of {
        groups.entry(s).or_default().push(i);
    }
    for v in groups.values_mut() {
        v.sort_unstable();
    }

    // Wire values at every slot's emission point (just before the
    // Hadamard that closes the slot, or the end).
    let pos_of_slot = |s: usize| -> usize {
        if s < nh {
            sc.hadamards[s]
        } else {
            c.gates.len()
        }
    };
    let cut_slots: Vec<usize> = {
        let mut v: Vec<usize> = groups.keys().copied().collect();
        v.sort_unstable();
        v
    };
    let mut snapshots: HashMap<usize, (Vec<Bits>, Vec<bool>)> = HashMap::new();
    {
        let wanted: HashMap<usize, usize> =
            cut_slots.iter().map(|&s| (pos_of_slot(s), s)).collect();
        let mut lin: Vec<Bits> = (0..n).map(|q| Bits::unit(sc.nvars, q)).collect();
        let mut cst = vec![false; n];
        let mut hcount = 0;
        for p in 0..=c.gates.len() {
            if let Some(&s) = wanted.get(&p) {
                snapshots.insert(s, (lin.clone(), cst.clone()));
            }
            if p == c.gates.len() {
                break;
            }
            step_wires(&c.gates[p], n, &mut lin, &mut cst, &mut hcount);
        }
    }

    // Local search over the slot assignment: move an odd term to another
    // cut slot inside its availability interval when deterministic TODD
    // says the total T-count drops.
    if opts.todd && opts.reassign_passes > 0 {
        reassign(&sc, &mut groups, &cut_slots, &snapshots, n, opts);
    }

    // TODD per slot (in parallel).
    let mut slots: Vec<usize> = groups.keys().copied().collect();
    slots.sort_unstable();
    let outs: Vec<(usize, usize, GroupOut)> = slots
        .par_iter()
        .map(|&s| {
            let ids = &groups[&s];
            let (lin, cst) = &snapshots[&s];
            let (cols, extra, global) = slot_columns(ids, &sc, lin, cst);
            let new = if opts.todd {
                best_todd(
                    &cols,
                    n,
                    opts.restarts,
                    opts.lns_rounds,
                    opts.seed ^ (s as u64).wrapping_mul(0x51_7CC1_B727_220A),
                )
            } else {
                tensor::clean(cols.clone())
            };
            let (terms, t_out) = group_terms(&cols, new, extra, n);
            (
                s,
                ids.len(),
                GroupOut {
                    terms,
                    global,
                    t_out,
                },
            )
        })
        .collect();

    // Emission.
    let mut at_pos: HashMap<usize, Vec<(Bits, u8)>> = HashMap::new();
    let mut global =
        (sc.global + outs.iter().map(|o| o.2.global as usize).sum::<usize>() as u8 % 8) % 8;
    let mut report = ToddReport {
        t_input: c.t_count(),
        t_folded: odd.len(),
        hadamards: nh,
        ..Default::default()
    };
    for (s, m_in, g) in outs {
        report.groups.push((s, m_in, g.t_out));
        at_pos.entry(pos_of_slot(s)).or_default().extend(g.terms);
    }
    let mut even_at: HashMap<usize, Vec<usize>> = HashMap::new();
    for &i in &even {
        even_at.entry(sc.terms[i].first_gate).or_default().push(i);
    }
    let mut out = Circuit::new(n);
    let mut lin: Vec<Bits> = (0..n).map(|q| Bits::unit(sc.nvars, q)).collect();
    let mut cst = vec![false; n];
    let mut hcount = 0;
    for p in 0..=c.gates.len() {
        if let Some(ids) = even_at.get(&p) {
            let solver = SpanSolver::new(&lin);
            for &i in ids {
                let t = &sc.terms[i];
                let co = solver
                    .solve(&t.parity)
                    .expect("term available at its first gate");
                let k1 = co.ones().fold(false, |a, q| a ^ cst[q]);
                let k = if k1 {
                    global = (global + t.coef) % 8;
                    (8 - t.coef) % 8
                } else {
                    t.coef
                };
                let wires: Vec<usize> = co.ones().collect();
                push_parity_phase(&mut out, &wires, k);
            }
        }
        if let Some(terms) = at_pos.get(&p) {
            for (co, k) in terms {
                let wires: Vec<usize> = co.ones().collect();
                push_parity_phase(&mut out, &wires, *k);
            }
        }
        if p == c.gates.len() {
            break;
        }
        let g = &c.gates[p];
        match *g {
            PGate::H(q) => {
                out.h(q);
            }
            PGate::X(q) => {
                out.x(q);
            }
            PGate::Cnot(a, b) => {
                out.cnot(a, b);
            }
            PGate::Swap(a, b) => {
                out.swap(a, b);
            }
            PGate::Phase(..) | PGate::Cz(..) | PGate::Ccz(..) => {}
        }
        step_wires(g, n, &mut lin, &mut cst, &mut hcount);
    }
    report.t_output = out
        .gates()
        .filter(|g| matches!(g, Gate::T(_) | Gate::Tdg(_)))
        .count();
    report.cnots = out.gates().filter(|g| matches!(g, Gate::Cnot(..))).count();
    report.global_phase = global;
    report.groups.sort_unstable();
    (out, report)
}

/// Advances the wire values over one gate (phases leave them unchanged).
fn step_wires(g: &PGate, n: usize, lin: &mut [Bits], cst: &mut [bool], hcount: &mut usize) {
    match *g {
        PGate::X(q) => cst[q] = !cst[q],
        PGate::Cnot(a, t) => {
            let s = lin[a].clone();
            lin[t].xor_with(&s);
            cst[t] ^= cst[a];
        }
        PGate::Swap(a, t) => {
            lin.swap(a, t);
            cst.swap(a, t);
        }
        PGate::H(q) => {
            let nv = lin[q].0.len() * 64;
            lin[q] = Bits::unit(nv, n + *hcount);
            cst[q] = false;
            *hcount += 1;
        }
        PGate::Phase(..) | PGate::Cz(..) | PGate::Ccz(..) => {}
    }
}

/// Checks `output` against `input` with the path-sum canonical form and
/// returns the global phase `Δ` (units of π/4) with `input = ω^Δ · output`.
pub fn verify_output(input: &PhaseCircuit, output: &Circuit) -> Result<u8, ToddError> {
    let vo = verify::vgates_from_circuit(output).map_err(|e| ToddError::Unsupported(e.0))?;
    let a = verify::path_sum(input.num_qubits, &input.to_vgates(), None);
    let b = verify::path_sum(output.num_qubits, &vo, None);
    verify::equivalent(&a, &b).ok_or_else(|| {
        ToddError::NotEquivalent(format!(
            "path sums differ (hadamards {} vs {}, {} vs {} monomials, outputs equal: {})",
            a.hadamards,
            b.hadamards,
            a.poly.len(),
            b.poly.len(),
            a.outputs == b.outputs
        ))
    })
}
