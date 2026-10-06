//! Chain-sweep ("temporal boundary") amplitudes for open 1D CZ circuits.
//!
//! For a circuit whose only two-qubit gates are `CZ(i, i+1)` on an open
//! chain, write every CZ as `Σ_b P_b ⊗ Z^b` (`P_b = |b><b|`): the left qubit
//! carries the projector, the right qubit the `Z^b`. The amplitude
//! `<x|U|0^n>` is then `Σ_bonds Π_i <x_i| W_i(bonds) |0>`, where qubit `i`'s
//! worldline `W_i` only sees the bonds on its two edges. Sweeping the chain
//! from qubit 0 to qubit `n-1` and summing the bonds of edge `i-1` while
//! creating those of edge `i` gives an exact contraction whose live tensor is
//! indexed by the unsliced bonds of one edge (plus at most one qubit wire),
//! so its width is about `D/2` for CZ-depth `D` (Manabe, Gu, Pan,
//! arXiv:2608.13110, use the same sweep).
//!
//! The sweep is compiled to executor ops ([`KOp`]) on a `W`-bit "bond
//! register" that starts in `|0..0>`: a bond is a bit of the register, a
//! free slot is a bit in `|0>`. Every step is a (generally non-unitary)
//! single-bit matrix, a controlled X or a diagonal term, so the existing
//! cache-blocked CPU executor and the Metal kernels run it unchanged. At
//! the end every slot is free again and the amplitude is the register's
//! `|0..0>` component times a host-side scale factor.
//!
//! Qubit `i`'s worldline is processed forwards (from `|0>` to `<x_i|`) or
//! backwards (from `<x_i|` with transposed gates to `|0>`), whichever needs
//! fewer live bits; that keeps the register at `ceil(D/2)` bits for
//! brickwork, as in the paper.
//!
//! Slicing: any CZ can be fixed to bond value `k`; it then becomes the local
//! operators `P_k` and `Z^k` and leaves the register. The sum over all
//! `2^s` assignments of `s` sliced bonds is the exact amplitude. Note that
//! slicing a bond only narrows the cut it crosses, not the others.

use crate::circuit::{Circuit, Op, SimError};
use crate::engines::blocked::KOp;
use crate::gate::{mat2_mul, Gate, Mat2};
use num_complex::Complex64;
use std::collections::HashMap;

const C0: Complex64 = Complex64::new(0.0, 0.0);
const C1: Complex64 = Complex64::new(1.0, 0.0);
/// Gain of the bond sums. `1` (unnormalised `[[1,1],[1,-1]]`) keeps the
/// register norm roughly constant from qubit to qubit, which matters in f32.
const S2: f64 = 1.0;

/// One step of a qubit's worldline.
#[derive(Clone, Copy, Debug)]
pub enum Ev {
    /// Single-qubit gate (any 2x2 matrix).
    G(Mat2),
    /// CZ with bond `b` to the right neighbour; this qubit is the projector side.
    P(usize),
    /// CZ with bond `b` to the left neighbour; this qubit gets `Z^b`.
    Z(usize),
}

/// A circuit in chain form: per-qubit worldlines plus bond metadata.
#[derive(Clone, Debug)]
pub struct ChainCircuit {
    /// Number of qubits.
    pub n: usize,
    /// Worldline of each qubit, in program order.
    pub lines: Vec<Vec<Ev>>,
    /// Left qubit of each bond (the bond lies on edge `(e, e+1)`).
    pub bond_edge: Vec<usize>,
    /// CZ-layer (ASAP over CZs, 1-based) of each bond.
    pub bond_layer: Vec<usize>,
}

impl ChainCircuit {
    /// Converts a circuit of single-qubit gates and nearest-neighbour CZs.
    pub fn from_circuit(c: &Circuit) -> Result<Self, SimError> {
        let n = c.num_qubits;
        let mut lines = vec![Vec::new(); n];
        let mut bond_edge = Vec::new();
        let mut bond_layer = Vec::new();
        let mut depth = vec![0usize; n];
        for (op_index, op) in c.ops.iter().enumerate() {
            let g = match op {
                Op::Gate(g) => g,
                Op::Measure(_) => continue,
                _ => {
                    return Err(SimError::MeasurementNotSupported {
                        backend: "chain_sweep",
                        op_index,
                    })
                }
            };
            match *g {
                Gate::Cz(a, b) => {
                    let (l, r) = if a < b { (a, b) } else { (b, a) };
                    if r != l + 1 {
                        return Err(SimError::Unsupported {
                            backend: "chain_sweep (CZ must be nearest-neighbour)",
                            gate: *g,
                        });
                    }
                    let id = bond_edge.len();
                    let layer = depth[l].max(depth[r]) + 1;
                    depth[l] = layer;
                    depth[r] = layer;
                    bond_edge.push(l);
                    bond_layer.push(layer);
                    lines[l].push(Ev::P(id));
                    lines[r].push(Ev::Z(id));
                }
                ref g1 if g1.arity() == 1 => {
                    let m = g1.matrix_1q().expect("1q gate");
                    lines[g1.qubits()[0]].push(Ev::G(m));
                }
                _ => {
                    return Err(SimError::Unsupported {
                        backend: "chain_sweep",
                        gate: *g,
                    })
                }
            }
        }
        Ok(ChainCircuit {
            n,
            lines,
            bond_edge,
            bond_layer,
        })
    }

    /// Number of bonds (CZ gates).
    pub fn num_bonds(&self) -> usize {
        self.bond_edge.len()
    }

    /// Bonds on edge `(e, e+1)`, in program order.
    pub fn edge_bonds(&self, e: usize) -> Vec<usize> {
        (0..self.num_bonds())
            .filter(|&b| self.bond_edge[b] == e)
            .collect()
    }
}

/// Keeps the first `n` qubits and the first `d` CZ layers of a circuit of
/// 1q gates and CZs (layers are ASAP over the CZs). A single-qubit gate is
/// kept when the last CZ before it on its qubit is in a kept layer (or there
/// is none); CZs touching qubits `>= n` are dropped.
pub fn truncate(c: &Circuit, n: usize, d: usize) -> Circuit {
    let mut out = Circuit::new(n);
    let mut depth = vec![0usize; c.num_qubits];
    for op in &c.ops {
        let Op::Gate(g) = op else { continue };
        match *g {
            Gate::Cz(a, b) => {
                let layer = depth[a].max(depth[b]) + 1;
                depth[a] = layer;
                depth[b] = layer;
                if layer <= d && a < n && b < n {
                    out.gate(*g);
                }
            }
            ref g1 => {
                let q = g1.qubits();
                if q.iter().all(|&q| q < n && depth[q] <= d) {
                    out.gate(*g);
                }
            }
        }
    }
    out
}

/// A compiled amplitude: run `ops` on a `width`-bit register from `|0..0>`,
/// read the `|0..0>` component and multiply by `scale`.
#[derive(Clone, Debug)]
pub struct SweepPlan {
    /// Register width (bits).
    pub width: usize,
    /// Executor ops.
    pub ops: Vec<KOp>,
    /// Host-side factor (f64).
    pub scale: Complex64,
    /// Op index at which each qubit's worldline starts (len `n + 1`).
    pub qubit_ops: Vec<usize>,
    /// Live register bits after each qubit.
    pub cut_width: Vec<usize>,
    /// Whether each qubit was swept backwards in time.
    pub backward: Vec<bool>,
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    /// Qubit unentangled with the register, state `ψ`.
    Product([Complex64; 2]),
    /// Qubit state is `U |bit p>` for register slot `p` (a bond).
    Tied(usize, Mat2),
    /// Qubit lives in slot `p`.
    Live(usize),
}

const ID: Mat2 = [[C1, C0], [C0, C1]];
const XM: Mat2 = [[C0, C1], [C1, C0]];

fn mat_vec(m: &Mat2, v: &[Complex64; 2]) -> [Complex64; 2] {
    [
        m[0][0] * v[0] + m[0][1] * v[1],
        m[1][0] * v[0] + m[1][1] * v[1],
    ]
}

fn transpose(m: &Mat2) -> Mat2 {
    [[m[0][0], m[1][0]], [m[0][1], m[1][1]]]
}

fn diag(a: Complex64, b: Complex64) -> Mat2 {
    [[a, C0], [C0, b]]
}

struct Builder<'a> {
    ops: Vec<KOp>,
    emit: bool,
    scale: Complex64,
    slot_of: HashMap<usize, usize>,
    used: Vec<bool>,
    live: usize,
    peak: usize,
    sliced: &'a HashMap<usize, u8>,
}

impl Builder<'_> {
    fn alloc(&mut self) -> usize {
        let s = match self.used.iter().position(|u| !u) {
            Some(s) => s,
            None => {
                self.used.push(false);
                self.used.len() - 1
            }
        };
        self.used[s] = true;
        self.live += 1;
        self.peak = self.peak.max(self.live);
        s
    }
    fn free(&mut self, s: usize) {
        debug_assert!(self.used[s]);
        self.used[s] = false;
        self.live -= 1;
    }
    fn push(&mut self, op: KOp) {
        if self.emit {
            self.ops.push(op);
        }
    }
    fn u1(&mut self, q: usize, m: Mat2) {
        self.push(KOp::U1 { q, m, ctrl: 0 });
    }
    fn phase(&mut self, mask: usize, pat: usize, f: Complex64) {
        if f != C1 {
            self.push(KOp::Phase { mask, pat, f });
        }
    }

    /// Processes one worldline; `init` is the initial ket, `fin` the final bra.
    fn line(&mut self, evs: &[Ev], backward: bool, init: [Complex64; 2], fin: [Complex64; 2]) {
        let mut mode = Mode::Product(init);
        let n = evs.len();
        for k in 0..n {
            let ev = if backward { evs[n - 1 - k] } else { evs[k] };
            // sliced CZs become local operators
            let ev = match ev {
                Ev::P(b) if self.sliced.contains_key(&b) => {
                    let v = self.sliced[&b];
                    Ev::G(if v == 0 { diag(C1, C0) } else { diag(C0, C1) })
                }
                Ev::Z(b) if self.sliced.contains_key(&b) => {
                    let v = self.sliced[&b];
                    Ev::G(if v == 0 { ID } else { diag(C1, -C1) })
                }
                e => e,
            };
            mode = match ev {
                Ev::G(m) => {
                    let m = if backward { transpose(&m) } else { m };
                    match mode {
                        Mode::Product(psi) => Mode::Product(mat_vec(&m, &psi)),
                        Mode::Tied(p, u) => Mode::Tied(p, mat2_mul(&m, &u)),
                        Mode::Live(p) => {
                            self.u1(p, m);
                            Mode::Live(p)
                        }
                    }
                }
                Ev::Z(b) => {
                    let a = self.slot_of.remove(&b).expect("left bond in register");
                    match mode {
                        Mode::Product(psi) => {
                            // M[q][b] = psi_q (-1)^{bq} / sqrt2
                            let m = [[psi[0] * S2, psi[0] * S2], [psi[1] * S2, -psi[1] * S2]];
                            self.u1(a, m);
                            Mode::Live(a)
                        }
                        Mode::Tied(p, u) => {
                            // out(p, q) = u[q][p] * Σ_b (-1)^{bq} v(p, b)
                            let mask = (1 << p) | (1 << a);
                            if u.iter().flatten().all(|z| z.norm() > 1e-12) {
                                // u[q][p] = A_p B_q C^{pq}: the 1-bit factors fuse
                                // with neighbouring 1q matrices, one 2-bit term left
                                let (b0, b1) = (u[0][0], u[1][0]);
                                let a1 = u[0][1] / u[0][0];
                                let cc = u[1][1] * u[0][0] / (u[1][0] * u[0][1]);
                                self.u1(a, [[b0 * S2, b0 * S2], [b1 * S2, -b1 * S2]]);
                                self.phase(1 << p, 1 << p, a1);
                                if (cc - C1).norm() > 1e-13 {
                                    self.phase(mask, mask, cc);
                                }
                            } else {
                                let h = [[C1 * S2, C1 * S2], [C1 * S2, -C1 * S2]];
                                self.u1(a, h);
                                for bp in 0..2 {
                                    for q in 0..2 {
                                        self.phase(mask, (bp << p) | (q << a), u[q][bp]);
                                    }
                                }
                            }
                            Mode::Live(a)
                        }
                        Mode::Live(p) => {
                            let mask = (1 << p) | (1 << a);
                            self.phase(mask, mask, -C1);
                            self.u1(a, [[C1 * S2, C1 * S2], [C0, C0]]);
                            self.free(a);
                            Mode::Live(p)
                        }
                    }
                }
                Ev::P(b) => match mode {
                    Mode::Product(psi) => {
                        let c = self.alloc();
                        self.slot_of.insert(b, c);
                        self.u1(c, [[psi[0], C0], [psi[1], C0]]);
                        Mode::Tied(c, ID)
                    }
                    Mode::Tied(p, u) => {
                        let c = self.alloc();
                        self.slot_of.insert(b, c);
                        self.push(KOp::U1 {
                            q: c,
                            m: XM,
                            ctrl: 1 << p,
                        });
                        if u != ID {
                            self.u1(c, u);
                        }
                        Mode::Tied(c, ID)
                    }
                    Mode::Live(p) => {
                        self.slot_of.insert(b, p);
                        Mode::Tied(p, ID)
                    }
                },
            };
        }
        match mode {
            Mode::Product(psi) => self.scale *= fin[0] * psi[0] + fin[1] * psi[1],
            Mode::Tied(p, u) => {
                for bp in 0..2 {
                    let f = fin[0] * u[0][bp] + fin[1] * u[1][bp];
                    self.phase(1 << p, bp << p, f);
                }
            }
            Mode::Live(p) => {
                self.u1(p, [[fin[0], fin[1]], [C0, C0]]);
                self.free(p);
            }
        }
    }
}

/// Compiles the amplitude `<x|U|0^n>` (bit `i` of `x` = qubit `i`, i.e.
/// Qiskit's little-endian integer) with the bonds in `sliced` fixed.
pub fn compile(cc: &ChainCircuit, x: u128, sliced: &HashMap<usize, u8>) -> SweepPlan {
    compile_prefix(cc, x, sliced, cc.n).0
}

/// Compiles the sweep over qubits `0..upto` only. The register then holds
/// the unsliced bonds of edge `(upto-1, upto)` (values of qubit `upto-1`
/// at those CZs); returns the plan and the `(bond, register bit)` pairs.
/// With `upto = n` this is [`compile`] and the pair list is empty.
pub fn compile_prefix(
    cc: &ChainCircuit,
    x: u128,
    sliced: &HashMap<usize, u8>,
    upto: usize,
) -> (SweepPlan, Vec<(usize, usize)>) {
    let mut b = Builder {
        ops: Vec::new(),
        emit: true,
        scale: C1,
        slot_of: HashMap::new(),
        used: Vec::new(),
        live: 0,
        peak: 0,
        sliced,
    };
    let zero = [C1, C0];
    let mut qubit_ops = Vec::with_capacity(upto + 1);
    let mut cut_width = Vec::with_capacity(upto);
    let mut backward = Vec::with_capacity(upto);
    for i in 0..upto {
        let xi = if (x >> i) & 1 == 1 {
            [C0, C1]
        } else {
            [C1, C0]
        };
        // dry runs: pick the direction with the smaller peak
        let mut best = (usize::MAX, false);
        for bw in [false, true] {
            let mut t = Builder {
                ops: Vec::new(),
                emit: false,
                scale: C1,
                slot_of: b.slot_of.clone(),
                used: b.used.clone(),
                live: b.live,
                peak: b.live,
                sliced,
            };
            let (init, fin) = if bw { (xi, zero) } else { (zero, xi) };
            t.line(&cc.lines[i], bw, init, fin);
            if t.peak < best.0 {
                best = (t.peak, bw);
            }
        }
        let bw = best.1;
        let (init, fin) = if bw { (xi, zero) } else { (zero, xi) };
        qubit_ops.push(b.ops.len());
        b.line(&cc.lines[i], bw, init, fin);
        cut_width.push(b.live);
        backward.push(bw);
    }
    qubit_ops.push(b.ops.len());
    let mut open: Vec<(usize, usize)> = b.slot_of.iter().map(|(&k, &v)| (k, v)).collect();
    open.sort_unstable();
    debug_assert!(upto < cc.n || b.live == 0);
    (
        SweepPlan {
            width: b.used.len().max(1),
            ops: b.ops,
            scale: b.scale,
            qubit_ops,
            cut_width,
            backward,
        },
        open,
    )
}

/// The chain read from the other end: qubit `i` becomes `n-1-i` (so `x`
/// must be bit-reversed accordingly, see [`reverse_bits`]); bond ids are
/// kept, and the projector side of every CZ moves to the other qubit.
pub fn mirror(cc: &ChainCircuit) -> ChainCircuit {
    let n = cc.n;
    let lines = (0..n)
        .map(|i| {
            cc.lines[n - 1 - i]
                .iter()
                .map(|e| match *e {
                    Ev::P(b) => Ev::Z(b),
                    Ev::Z(b) => Ev::P(b),
                    g => g,
                })
                .collect()
        })
        .collect();
    ChainCircuit {
        n,
        lines,
        bond_edge: cc.bond_edge.iter().map(|&e| n - 2 - e).collect(),
        bond_layer: cc.bond_layer.clone(),
    }
}

/// Reverses the low `n` bits of `x`.
pub fn reverse_bits(x: u128, n: usize) -> u128 {
    (0..n).fold(0, |acc, i| acc | (((x >> i) & 1) << (n - 1 - i)))
}

/// Boundary tensors of a cut, for meet-in-the-middle amplitudes and free
/// slice sums on the cut's edge.
#[derive(Clone, Debug)]
pub struct CutTensors {
    /// The bonds of the cut edge, in index order (bit `j` of an index =
    /// value of `bonds[j]`).
    pub bonds: Vec<usize>,
    /// Left half, indexed by the bond values (values of qubit `e`).
    pub left: Vec<Complex64>,
    /// Right half, Walsh-Hadamard transformed so that the amplitude is
    /// `Σ_b left[b] * right[b]`.
    pub right: Vec<Complex64>,
}

impl CutTensors {
    /// The full amplitude.
    pub fn amplitude(&self) -> Complex64 {
        self.left.iter().zip(&self.right).map(|(a, b)| a * b).sum()
    }

    /// The slice sums for the bonds at positions `pos` of `self.bonds`:
    /// entry `s` is the amplitude with `bonds[pos[j]] = bit j of s`.
    pub fn slice_sums(&self, pos: &[usize]) -> Vec<Complex64> {
        let mut out = vec![C0; 1 << pos.len()];
        for (i, (a, b)) in self.left.iter().zip(&self.right).enumerate() {
            let s = pos
                .iter()
                .enumerate()
                .fold(0, |acc, (j, &p)| acc | (((i >> p) & 1) << j));
            out[s] += a * b;
        }
        out
    }
}

/// Reads the bond-indexed tensor out of a register: `slots[j]` is the
/// register bit of bond `j`; every other register bit must be 0.
fn gather(amps: impl Fn(usize) -> Complex64, slots: &[usize], scale: Complex64) -> Vec<Complex64> {
    (0..1usize << slots.len())
        .map(|i| {
            let idx = slots
                .iter()
                .enumerate()
                .fold(0, |acc, (j, &s)| acc | (((i >> j) & 1) << s));
            amps(idx) * scale
        })
        .collect()
}

/// Splits the amplitude of `x` at edge `(e, e+1)`: sweeps qubits `0..=e`
/// from the left and `e+1..n` from the right (CPU, precision `T`).
pub fn cut_tensors_cpu<T: crate::engines::statevector::Real>(
    cc: &ChainCircuit,
    x: u128,
    e: usize,
    cfg: &crate::engines::blocked::BlockConfig,
) -> Result<CutTensors, SimError> {
    let none = HashMap::new();
    let (pl, open_l) = compile_prefix(cc, x, &none, e + 1);
    let m = mirror(cc);
    let (pr, open_r) = compile_prefix(&m, reverse_bits(x, cc.n), &none, cc.n - 1 - e);
    let bonds: Vec<usize> = open_l.iter().map(|&(b, _)| b).collect();
    debug_assert_eq!(bonds, open_r.iter().map(|&(b, _)| b).collect::<Vec<_>>());
    let run = |p: &SweepPlan, open: &[(usize, usize)]| -> Result<Vec<Complex64>, SimError> {
        let mut sv = crate::engines::statevector::StateVector::<T>::try_new(p.width)?;
        sv.apply_kops_blocked(&p.ops, cfg);
        let slots: Vec<usize> = open.iter().map(|&(_, s)| s).collect();
        Ok(gather(|i| sv.amplitude(i), &slots, p.scale))
    };
    let left = run(&pl, &open_l)?;
    let mut right = run(&pr, &open_r)?;
    // the right half holds values c of qubit e+1; the CZs contribute
    // (-1)^{b.c}: right'[b] = Σ_c (-1)^{b.c} right[c]
    let m = bonds.len();
    for k in 0..m {
        for i in 0..right.len() {
            if (i >> k) & 1 == 0 {
                let j = i | (1 << k);
                let (a, b) = (right[i], right[j]);
                right[i] = a + b;
                right[j] = a - b;
            }
        }
    }
    Ok(CutTensors { bonds, left, right })
}

/// Exact amplitude on the CPU (cache-blocked executor, precision `T`).
pub fn amplitude_cpu<T: crate::engines::statevector::Real>(
    plan: &SweepPlan,
    cfg: &crate::engines::blocked::BlockConfig,
) -> Result<Complex64, SimError> {
    let mut sv = crate::engines::statevector::StateVector::<T>::try_new(plan.width)?;
    sv.apply_kops_blocked(&plan.ops, cfg);
    Ok(sv.amplitude(0) * plan.scale)
}

/// Convenience: exact amplitude of `x`, unsliced, f64 on the CPU.
pub fn amplitude(cc: &ChainCircuit, x: u128) -> Result<Complex64, SimError> {
    let plan = compile(cc, x, &HashMap::new());
    amplitude_cpu::<f64>(&plan, &Default::default())
}

/// Sum over every assignment of the bonds in `bonds` (each slice compiled
/// and run separately); returns the per-slice amplitudes, indexed by the
/// assignment (bit `j` = value of `bonds[j]`).
pub fn slice_amplitudes_cpu<T: crate::engines::statevector::Real>(
    cc: &ChainCircuit,
    x: u128,
    bonds: &[usize],
    cfg: &crate::engines::blocked::BlockConfig,
) -> Result<Vec<Complex64>, SimError> {
    let mut out = Vec::with_capacity(1 << bonds.len());
    for s in 0..1usize << bonds.len() {
        let sl: HashMap<usize, u8> = bonds
            .iter()
            .enumerate()
            .map(|(j, &b)| (b, ((s >> j) & 1) as u8))
            .collect();
        let plan = compile(cc, x, &sl);
        out.push(amplitude_cpu::<T>(&plan, cfg)?);
    }
    Ok(out)
}

/// The circuit with the CZs of `bonds` replaced by `P_k ⊗ Z^k` for the
/// assignment `s` (bit `j` = value of `bonds[j]`), as single-qubit
/// matrices: for checking slices against a state vector. Returns the gate
/// list in program order with each sliced CZ replaced by two 1q matrices.
pub fn sliced_ops(c: &Circuit, bonds: &[usize], s: usize) -> Vec<(usize, Mat2, Option<usize>)> {
    // (qubit, matrix, Some(other) for an unsliced CZ)
    let mut out = Vec::new();
    let mut id = 0usize;
    for op in &c.ops {
        let Op::Gate(g) = op else { continue };
        match *g {
            Gate::Cz(a, b) => {
                let (l, r) = if a < b { (a, b) } else { (b, a) };
                if let Some(j) = bonds.iter().position(|&x| x == id) {
                    let v = (s >> j) & 1;
                    out.push((l, if v == 0 { diag(C1, C0) } else { diag(C0, C1) }, None));
                    out.push((r, if v == 0 { ID } else { diag(C1, -C1) }, None));
                } else {
                    out.push((l, ID, Some(r)));
                }
                id += 1;
            }
            ref g1 => out.push((g1.qubits()[0], g1.matrix_1q().unwrap(), None)),
        }
    }
    out
}
