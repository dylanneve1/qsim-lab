//! Exact tensor-network contraction: amplitudes `<x|C|0^n>`, batches of
//! amplitudes over a set of open qubits, and Pauli expectation values
//! `<0|C† P C|0>` of unitary circuits, without truncation
//! (research/simulability/tn.md).
//!
//! ```text
//! circuit -> Network (gate tensors, |0> inputs, <x| outputs folded in)
//!         -> simplify: absorb rank <= 2 tensors, merge pairs whose rank does
//!            not grow, fix single-valued indices, identify diagonal index
//!            pairs (hyperedges)            [network.rs]
//!         -> tree search: randomised greedy + recursive multilevel FM
//!            bisection, subtree reconfiguration, memory-bounded slicing
//!                                          [path.rs]
//!         -> execute: permute + GEMM (faer, complex f64 / f32), slices in
//!            parallel or GEMMs in parallel, slice-invariant subtrees once
//!                                          [exec.rs]
//! ```
//!
//! Expectation values first drop every gate outside the backward light cone
//! of the observable (they cancel against their inverses), then contract the
//! doubled network `<0|C_cone† P C_cone|0>`.
//!
//! Everything is exact: the only approximation is floating-point rounding
//! (f64: ~1e-15 relative per operation; f32: ~1e-7).

pub mod exec;
pub mod network;
pub mod path;

pub use exec::{contract, ExecOptions, ExecStats, PairStrategy, Scalar};
pub use network::{Ix, Network, SimplifyOptions, SimplifyStats, Tensor};
pub use path::{
    bisection, greedy, reconfigure, search, slice_tree, tree_cost, BisectParams, ContractionTree,
    GreedyParams, Hypergraph, Path, PathOptions, PathStats, TreeCost,
};

use crate::circuit::{check_gate, Circuit, Op, SimError};
use crate::gate::Gate;
use num_complex::Complex64;
use std::time::Instant;

/// Arithmetic of the contraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// Complex f64 (default; differential tests at 1e-12).
    F64,
    /// Complex f32 (half the memory and about twice the GEMM rate).
    F32,
}

/// Options of the tensor-network engine.
#[derive(Clone, Debug)]
pub struct TnOptions {
    /// Arithmetic of the contraction.
    pub precision: Precision,
    /// Memory budget for the contraction's intermediates, in bytes. Unless
    /// `path.target_log2_size` is set, the largest intermediate is sliced
    /// down to `max_bytes / (8 · entry size)` entries.
    pub max_bytes: u128,
    /// Simplification passes.
    pub simplify: SimplifyOptions,
    /// Contraction-tree search.
    pub path: PathOptions,
    /// Threads (0: the rayon pool).
    pub threads: usize,
    /// How pairwise contractions are executed.
    pub strategy: PairStrategy,
}

impl Default for TnOptions {
    fn default() -> Self {
        TnOptions {
            precision: Precision::F64,
            max_bytes: 1 << 30,
            simplify: SimplifyOptions::default(),
            path: PathOptions::default(),
            threads: 0,
            strategy: PairStrategy::Auto,
        }
    }
}

/// What a run did.
#[derive(Clone, Debug)]
pub struct TnReport {
    /// Qubits of the (light-cone reduced) circuit that was contracted.
    pub qubits: usize,
    /// Gates of that circuit (doubled for expectation values).
    pub gates: usize,
    /// Simplification statistics.
    pub simplify: SimplifyStats,
    /// Contraction-tree statistics.
    pub path: PathStats,
    /// Execution statistics.
    pub exec: ExecStats,
    /// Seconds spent building and simplifying the network.
    pub build_secs: f64,
    /// Seconds spent searching for the tree.
    pub search_secs: f64,
    /// Seconds spent contracting.
    pub contract_secs: f64,
}

/// A Pauli operator on one qubit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pauli {
    /// Pauli X.
    X,
    /// Pauli Y.
    Y,
    /// Pauli Z.
    Z,
}

impl Pauli {
    fn gate(self, q: usize) -> Gate {
        match self {
            Pauli::X => Gate::X(q),
            Pauli::Y => Gate::Y(q),
            Pauli::Z => Gate::Z(q),
        }
    }
}

/// Parses `"XIZY"` (character `k` acts on qubit `k`; `I` or `_` is the
/// identity) into a list of `(qubit, Pauli)`.
pub fn parse_pauli(s: &str) -> Result<Vec<(usize, Pauli)>, SimError> {
    let mut out = Vec::new();
    for (q, ch) in s.chars().enumerate() {
        match ch {
            'X' | 'x' => out.push((q, Pauli::X)),
            'Y' | 'y' => out.push((q, Pauli::Y)),
            'Z' | 'z' => out.push((q, Pauli::Z)),
            'I' | 'i' | '_' => {}
            _ => {
                return Err(SimError::NotSupported {
                    what: "tn: Pauli strings use the characters I, X, Y, Z",
                })
            }
        }
    }
    Ok(out)
}

/// The `n` low bits of `x` as a bit string (qubit `q` = bit `q`).
pub fn bits_of(x: u128, n: usize) -> Vec<bool> {
    (0..n).map(|q| q < 128 && (x >> q) & 1 == 1).collect()
}

fn require_unitary(c: &Circuit) -> Result<(), SimError> {
    for (k, op) in c.ops.iter().enumerate() {
        match op {
            Op::Gate(g) => check_gate(g, c.num_qubits)?,
            _ => {
                return Err(SimError::MeasurementNotSupported {
                    backend: "tn",
                    op_index: k,
                })
            }
        }
    }
    Ok(())
}

/// Bytes per entry of a precision.
fn entry_bytes(p: Precision) -> u128 {
    match p {
        Precision::F64 => 16,
        Precision::F32 => 8,
    }
}

/// The slicing target used when `opts.path.target_log2_size` is unset.
pub fn default_target_log2(opts: &TnOptions) -> f64 {
    let entries = (opts.max_bytes / (8 * entry_bytes(opts.precision))).max(2);
    (entries as f64).log2().floor()
}

/// Simplifies `nw`, searches a tree and contracts it: the result laid out
/// over `nw.output` (row-major), and the report (with `qubits` and `gates`
/// zero; the circuit-level functions fill them in).
pub fn run_network(
    mut nw: Network,
    opts: &TnOptions,
) -> Result<(Vec<Complex64>, TnReport), SimError> {
    let t0 = Instant::now();
    let simplify = nw.simplify(&opts.simplify);
    let build_secs = t0.elapsed().as_secs_f64();
    let hg = Hypergraph::from_network(&nw);
    let mut popts = opts.path.clone();
    if popts.target_log2_size.is_none() {
        popts.target_log2_size = Some(default_target_log2(opts));
    }
    let threads = opts.threads;
    let run = |popts: &PathOptions| -> Result<(Path, Vec<Complex64>, ExecStats), SimError> {
        let path = search(&hg, popts);
        let eo = ExecOptions {
            max_bytes: opts.max_bytes,
            threads,
            strategy: opts.strategy,
        };
        let (v, st) = match opts.precision {
            Precision::F64 => contract::<Complex64>(&nw, &path.tree, &path.sliced, &eo)?,
            Precision::F32 => {
                contract::<num_complex::Complex<f32>>(&nw, &path.tree, &path.sliced, &eo)?
            }
        };
        Ok((path, v, st))
    };
    // if the working set of a slice is still too large, slice harder
    let mut attempt = 0;
    let (path, value, exec) = loop {
        match run(&popts) {
            Ok(x) => break x,
            Err(SimError::TooLarge { .. })
                if attempt < 6 && opts.path.target_log2_size.is_none() =>
            {
                attempt += 1;
                popts.target_log2_size = popts.target_log2_size.map(|t| t - 2.0);
            }
            Err(e) => return Err(e),
        }
    };
    Ok((
        value,
        TnReport {
            qubits: 0,
            gates: 0,
            simplify,
            search_secs: path.stats.secs,
            contract_secs: exec.secs,
            path: path.stats,
            exec,
            build_secs,
        },
    ))
}

/// The amplitude `<x|C|0^n>` (global phase included); `bits[q]` is bit `q`
/// of `x`.
pub fn amplitude(
    c: &Circuit,
    bits: &[bool],
    opts: &TnOptions,
) -> Result<(Complex64, TnReport), SimError> {
    let (v, r) = amplitudes(c, bits, &[], opts)?;
    Ok((v[0], r))
}

/// The `2^k` amplitudes `<x|C|0^n>` where the closed qubits take their bit
/// from `bits` and the open qubits `open[0..k]` run over all values: entry
/// `j` of the result has bit `t` of `j` on qubit `open[t]`.
pub fn amplitudes(
    c: &Circuit,
    bits: &[bool],
    open: &[usize],
    opts: &TnOptions,
) -> Result<(Vec<Complex64>, TnReport), SimError> {
    require_unitary(c)?;
    let t0 = Instant::now();
    let nw = Network::amplitude(c, bits, open)?;
    let pre = t0.elapsed().as_secs_f64();
    let (v, mut r) = run_network(nw, opts)?;
    r.qubits = c.num_qubits;
    r.gates = c.num_gates();
    r.build_secs += pre;
    Ok((v, r))
}

/// Gates of `c` in the backward light cone of `support`, relabelled onto
/// the cone's qubits; also returns the cone (global qubit of each local one).
pub fn light_cone(c: &Circuit, support: &[usize]) -> Result<(Circuit, Vec<usize>), SimError> {
    require_unitary(c)?;
    let n = c.num_qubits;
    let mut live = vec![false; n];
    for &q in support {
        if q >= n {
            return Err(SimError::QubitOutOfRange {
                qubit: q,
                num_qubits: n,
            });
        }
        live[q] = true;
    }
    let gates: Vec<Gate> = c.gates().copied().collect();
    let mut keep = vec![false; gates.len()];
    for (k, g) in gates.iter().enumerate().rev() {
        let qs = g.qubits();
        if qs.iter().any(|&q| live[q]) {
            keep[k] = true;
            for q in qs {
                live[q] = true;
            }
        }
    }
    let cone: Vec<usize> = (0..n).filter(|&q| live[q]).collect();
    let mut map = vec![usize::MAX; n];
    for (i, &q) in cone.iter().enumerate() {
        map[q] = i;
    }
    let mut out = Circuit::new(cone.len());
    for (k, g) in gates.iter().enumerate() {
        if keep[k] {
            out.gate(relabel(g, &map));
        }
    }
    Ok((out, cone))
}

fn relabel(g: &Gate, m: &[usize]) -> Gate {
    use Gate::*;
    match *g {
        I(q) => I(m[q]),
        H(q) => H(m[q]),
        X(q) => X(m[q]),
        Y(q) => Y(m[q]),
        Z(q) => Z(m[q]),
        S(q) => S(m[q]),
        Sdg(q) => Sdg(m[q]),
        T(q) => T(m[q]),
        Tdg(q) => Tdg(m[q]),
        Sx(q) => Sx(m[q]),
        Sxdg(q) => Sxdg(m[q]),
        Rx(q, t) => Rx(m[q], t),
        Ry(q, t) => Ry(m[q], t),
        Rz(q, t) => Rz(m[q], t),
        Phase(q, t) => Phase(m[q], t),
        U(q, a, b, c) => U(m[q], a, b, c),
        Cnot(a, b) => Cnot(m[a], m[b]),
        Cz(a, b) => Cz(m[a], m[b]),
        Swap(a, b) => Swap(m[a], m[b]),
        ISwap(a, b) => ISwap(m[a], m[b]),
        ISwapdg(a, b) => ISwapdg(m[a], m[b]),
        CPhase(a, b, t) => CPhase(m[a], m[b], t),
        Ccx(a, b, t) => Ccx(m[a], m[b], m[t]),
    }
}

fn is_diagonal_gate(g: &Gate) -> bool {
    matches!(
        g,
        Gate::I(_)
            | Gate::Z(_)
            | Gate::S(_)
            | Gate::Sdg(_)
            | Gate::T(_)
            | Gate::Tdg(_)
            | Gate::Rz(..)
            | Gate::Phase(..)
            | Gate::Cz(..)
            | Gate::CPhase(..)
    )
}

/// The backward light cone of a Pauli string, refined by commutation: a
/// diagonal gate whose qubits all carry a diagonal factor (`Z` or `I`) of
/// the Heisenberg-evolved observable commutes with it and is dropped. (The
/// observable stays a product of its original `Z`/`I` factors on every
/// qubit no kept gate has touched, so the test is exact.) Returns the kept
/// gates relabelled onto the cone and the cone.
pub fn pauli_light_cone(
    c: &Circuit,
    pauli: &[(usize, Pauli)],
) -> Result<(Circuit, Vec<usize>), SimError> {
    require_unitary(c)?;
    let n = c.num_qubits;
    let mut live = vec![false; n];
    let mut diag = vec![true; n];
    for &(q, p) in pauli {
        if q >= n {
            return Err(SimError::QubitOutOfRange {
                qubit: q,
                num_qubits: n,
            });
        }
        live[q] = true;
        diag[q] = p == Pauli::Z;
    }
    let gates: Vec<Gate> = c.gates().copied().collect();
    let mut keep = vec![false; gates.len()];
    for (k, g) in gates.iter().enumerate().rev() {
        let qs = g.qubits();
        if !qs.iter().any(|&q| live[q]) {
            continue;
        }
        if is_diagonal_gate(g) && qs.iter().all(|&q| diag[q]) {
            continue; // commutes with the observable
        }
        keep[k] = true;
        for q in qs {
            live[q] = true;
            diag[q] = false;
        }
    }
    let cone: Vec<usize> = (0..n).filter(|&q| live[q]).collect();
    let mut map = vec![usize::MAX; n];
    for (i, &q) in cone.iter().enumerate() {
        map[q] = i;
    }
    let mut out = Circuit::new(cone.len());
    for (k, g) in gates.iter().enumerate() {
        if keep[k] {
            out.gate(relabel(g, &map));
        }
    }
    Ok((out, cone))
}

/// The doubled circuit `C_cone · P · C_cone†` of `<0|C† P C|0>` after the
/// light-cone reduction ([`pauli_light_cone`]), and the cone.
pub fn expectation_circuit(
    c: &Circuit,
    pauli: &[(usize, Pauli)],
) -> Result<(Circuit, Vec<usize>), SimError> {
    let mut seen = vec![false; c.num_qubits];
    for &q in pauli.iter().map(|(q, _)| q) {
        if q < seen.len() && seen[q] {
            return Err(SimError::NotSupported {
                what: "tn: a qubit appears twice in the Pauli string",
            });
        }
        if q < seen.len() {
            seen[q] = true;
        }
    }
    let (cone_c, cone) = pauli_light_cone(c, pauli)?;
    let mut d = cone_c.clone();
    for &(q, p) in pauli {
        let local = cone
            .iter()
            .position(|&x| x == q)
            .expect("support is in the cone");
        d.gate(p.gate(local));
    }
    d.append(&cone_c.inverse());
    Ok((d, cone))
}

/// The expectation value `<0|C† P C|0>` of a Pauli string (real for every
/// Hermitian `P`; the imaginary rounding residue is dropped).
pub fn expectation(
    c: &Circuit,
    pauli: &[(usize, Pauli)],
    opts: &TnOptions,
) -> Result<(f64, TnReport), SimError> {
    require_unitary(c)?;
    let t0 = Instant::now();
    let (d, cone) = expectation_circuit(c, pauli)?;
    let bits = vec![false; d.num_qubits];
    let nw = Network::amplitude(&d, &bits, &[])?;
    let pre = t0.elapsed().as_secs_f64();
    let (v, mut r) = run_network(nw, opts)?;
    r.qubits = cone.len();
    r.gates = d.num_gates();
    r.build_secs += pre;
    Ok((v[0].re, r))
}

/// Contraction cost of one amplitude of `c` from a quick tree search (no
/// contraction): used by the planner's cost model.
pub fn estimate_amplitude(
    c: &Circuit,
    opts: &TnOptions,
) -> Result<(PathStats, SimplifyStats), SimError> {
    require_unitary(c)?;
    let bits = vec![false; c.num_qubits];
    let mut nw = Network::amplitude(c, &bits, &[])?;
    let st = nw.simplify(&opts.simplify);
    let hg = Hypergraph::from_network(&nw);
    let mut popts = opts.path.clone();
    if popts.target_log2_size.is_none() {
        popts.target_log2_size = Some(default_target_log2(opts));
    }
    Ok((search(&hg, &popts).stats, st))
}

/// The network as JSON `{"inputs": [[ids]], "output": [ids], "size_dict":
/// {"id": dim}}` (index ids as integers), for comparison with other
/// contraction-path optimisers.
pub fn network_json(nw: &Network) -> String {
    let inputs: Vec<String> = nw
        .tensors
        .iter()
        .map(|t| {
            let v: Vec<String> = t.inds.iter().map(|i| i.to_string()).collect();
            format!("[{}]", v.join(","))
        })
        .collect();
    let out: Vec<String> = nw.output.iter().map(|i| i.to_string()).collect();
    let sizes: Vec<String> = nw
        .dims
        .iter()
        .enumerate()
        .map(|(i, d)| format!("\"{i}\":{d}"))
        .collect();
    format!(
        "{{\"inputs\":[{}],\"output\":[{}],\"size_dict\":{{{}}}}}",
        inputs.join(","),
        out.join(","),
        sizes.join(",")
    )
}
