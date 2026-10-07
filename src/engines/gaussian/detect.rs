//! Free-fermion structure detection: block fusion, the matchgate test, the
//! Jordan–Wigner ordering, and the compiled Gaussian program
//! (docs/ENGINE_GAUSSIAN.md §3).
//!
//! The pipeline is
//!
//! 1. **Relabel and fuse.** Explicit `SWAP` gates only rename wires. Every
//!    other gate is assigned to the wire ("slot") it acts on after the
//!    renaming. Consecutive two-qubit gates on the same pair, together with
//!    the one-qubit gates between them, fuse into one 4x4 block. The
//!    one-qubit gates between blocks with different partners ("tails") are
//!    split between the block before and the block after: for every block
//!    the detector tries each cut point of its closing tails and keeps the
//!    one that makes the block Gaussian (or diagonal), smallest residual
//!    first. A block that equals `SWAP · V` with `V` Gaussian or diagonal is
//!    replaced by `V` followed by a renaming, like an explicit `SWAP`.
//! 2. **Classify.** A diagonal two-qubit block is two one-site phases times
//!    `exp(i g n_a n_b)`; `g` is its *interaction phase*. A non-diagonal
//!    block is tested for the matchgate property: conjugation must map the
//!    four local Majorana operators into their own span. The residual is the
//!    largest normalised Hilbert–Schmidt distance from that span.
//! 3. **Order.** Non-diagonal Gaussian blocks need their two wires to be
//!    adjacent in the Jordan–Wigner order. The detector keeps the given qubit
//!    order if it already works; otherwise it orders the wires along the
//!    paths of the graph of such blocks (or a greedy maximum-weight path
//!    cover when that graph is not a union of paths) and keeps whichever
//!    order leaves more blocks adjacent.
//! 4. **Compile.** Each Gaussian block becomes an orthogonal map of its 2 or
//!    4 Majorana operators (plus a sign on all later modes when the block
//!    flips fermion parity), each interaction phase becomes an
//!    [`InteractionPhase`] record.

use crate::circuit::Circuit;
use crate::gate::{mat4_swap_qubits, Gate, Mat2, Mat4};
use num_complex::Complex64 as C;

const Z: C = C::new(0.0, 0.0);
const O: C = C::new(1.0, 0.0);
const I: C = C::new(0.0, 1.0);

/// Detector settings.
#[derive(Clone, Copy, Debug)]
pub struct DetectOptions {
    /// A block is Gaussian (diagonal) when its residual (off-diagonal norm)
    /// is at most `tol`, and an interaction phase with `|g| <= tol` counts as
    /// zero. The engine is exact only at round-off level, so the default is
    /// `1e-10`.
    pub tol: f64,
    /// Treat `SWAP` gates, and fused blocks equal to `SWAP` times a Gaussian
    /// or diagonal gate, as wire renamings (default `true`).
    pub relabel_swaps: bool,
    /// Search for a Jordan–Wigner order other than the qubit order
    /// (default `true`).
    pub reorder: bool,
}

impl Default for DetectOptions {
    fn default() -> Self {
        DetectOptions {
            tol: 1e-10,
            relabel_swaps: true,
            reorder: true,
        }
    }
}

/// How the Jordan–Wigner order was chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ordering {
    /// The circuit's own qubit order (wire `k` is mode `k`).
    Identity,
    /// Concatenated paths of the graph of non-diagonal Gaussian blocks.
    Paths,
    /// A greedy maximum-weight path cover of that graph (it had a vertex of
    /// degree > 2 or a cycle): some blocks are left non-adjacent.
    GreedyCover,
}

/// A non-Gaussian diagonal interaction `exp(i g n_a n_b)` found in a
/// diagonal two-qubit block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InteractionPhase {
    /// Index of the block in time order (see [`GaussianReport::blocks`]).
    pub block: usize,
    /// The two wires, as initial qubit indices (wire `w` starts on qubit `w`).
    pub wires: (usize, usize),
    /// The Jordan–Wigner modes of the two wires.
    pub modes: (usize, usize),
    /// The interaction phase, wrapped to `(-π, π]`.
    pub g: f64,
}

/// What the detector found (see the module docs).
#[derive(Clone, Debug, Default)]
pub struct GaussianReport {
    /// Number of qubits.
    pub n: usize,
    /// Fused blocks: one-qubit, two-qubit and opaque (three-qubit) blocks.
    pub blocks: usize,
    /// Two-qubit blocks among `blocks`.
    pub blocks_2q: usize,
    /// Blocks that are exactly Gaussian (to `tol`): one-site maps, diagonal
    /// blocks without interaction phase, and adjacent matchgates.
    pub gaussian_blocks: usize,
    /// `gaussian_blocks / blocks` (1 for an empty circuit).
    pub gaussian_fraction: f64,
    /// Largest residual of a block that is not an interaction block: the
    /// matchgate residual of a non-Gaussian block, 1 for a matchgate whose
    /// wires end up non-adjacent and for a three-qubit block, and the
    /// round-off residual of the Gaussian ones.
    pub max_residual: f64,
    /// Blocks that are neither Gaussian nor interaction blocks.
    pub non_gaussian: usize,
    /// Of those, matchgates whose wires are not adjacent in the order.
    pub nonadjacent: usize,
    /// Diagonal blocks with a non-zero interaction phase.
    pub interactions: Vec<InteractionPhase>,
    /// `Σ |g|` over `interactions`.
    pub interaction_total: f64,
    /// `max |g|` over `interactions`.
    pub interaction_max: f64,
    /// SWAP gates and SWAP-equivalent blocks turned into wire renamings.
    pub swaps_relabelled: usize,
    /// How the order was chosen.
    pub ordering: Option<Ordering>,
    /// `order[k]` = the wire (initial qubit) on Jordan–Wigner mode `k`.
    pub order: Vec<usize>,
    /// The chains of the order (paths of the block graph), each a run of
    /// consecutive modes; every path starts at its smaller end wire and the
    /// paths are sorted by first wire. For [`Ordering::Identity`] this is
    /// one path `0..n`.
    pub paths: Vec<Vec<usize>>,
    /// `mode_of_qubit[q]` = the Jordan–Wigner mode held by qubit `q` at the
    /// end of the circuit.
    pub mode_of_qubit: Vec<usize>,
    /// Every Gaussian block commutes with the total particle number
    /// (`U(n)` rather than `O(2n)`). A property of the fused blocks: an `X`
    /// left as a block of its own makes it false even when the circuit as a
    /// whole conserves the particle number.
    pub number_conserving: bool,
    /// Gaussian with no interaction phase: the engine is exact.
    pub exact: bool,
    /// Gaussian up to interaction phases (dropping them leaves a Gaussian
    /// circuit).
    pub free: bool,
    /// Seconds spent detecting.
    pub secs: f64,
}

/// One step of a compiled Gaussian circuit, on Jordan–Wigner modes.
///
/// `q` maps the local Majorana operators: with `c` the operators of the
/// block (`γ_{2k}, γ_{2k+1}` of each mode, in mode order), `U† c_i U =
/// Σ_j q[i][j] c_j`. Every Majorana operator of a later mode is multiplied
/// by `det q` (the block flips fermion parity when `det q = -1`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GaussOp {
    /// A one-mode map on mode `mode`.
    One {
        /// The mode.
        mode: usize,
        /// The 2x2 orthogonal map.
        q: [[f64; 2]; 2],
    },
    /// A two-mode map on modes `mode` and `mode + 1`.
    Two {
        /// The lower mode.
        mode: usize,
        /// The 4x4 orthogonal map.
        q: [[f64; 4]; 4],
    },
    /// An interaction phase `exp(i g n_a n_b)` (not Gaussian; the engine
    /// refuses or drops it, see `InteractionPolicy`).
    Interaction(InteractionPhase),
}

/// The detector's report plus the compiled program (empty unless every
/// block is Gaussian or an interaction block).
#[derive(Clone, Debug, Default)]
pub struct GaussianProgram {
    /// The report.
    pub report: GaussianReport,
    /// Gaussian steps in time order (`report.free` is true), else empty.
    pub ops: Vec<GaussOp>,
}

// ---------------------------------------------------------------------------
// small complex matrices

fn mul4(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut m = [[Z; 4]; 4];
    for i in 0..4 {
        for k in 0..4 {
            let x = a[i][k];
            if x == Z {
                continue;
            }
            for j in 0..4 {
                m[i][j] += x * b[k][j];
            }
        }
    }
    m
}

fn dag4(a: &Mat4) -> Mat4 {
    let mut m = [[Z; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            m[i][j] = a[j][i].conj();
        }
    }
    m
}

fn mul2(a: &Mat2, b: &Mat2) -> Mat2 {
    let mut m = [[Z; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            m[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
        }
    }
    m
}

fn dag2(a: &Mat2) -> Mat2 {
    [
        [a[0][0].conj(), a[1][0].conj()],
        [a[0][1].conj(), a[1][1].conj()],
    ]
}

const ID2: Mat2 = [[O, Z], [Z, O]];

/// `a ⊗ b` with `a` on the first (more significant) qubit.
fn kron(a: &Mat2, b: &Mat2) -> Mat4 {
    let mut m = [[Z; 4]; 4];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    m[2 * i + j][2 * k + l] = a[i][k] * b[j][l];
                }
            }
        }
    }
    m
}

const PX: Mat2 = [[Z, O], [O, Z]];
const PY: Mat2 = [[Z, C::new(0.0, -1.0)], [I, Z]];
const PZ: Mat2 = [[O, Z], [Z, C::new(-1.0, 0.0)]];

/// The four local Majorana operators of two Jordan–Wigner-adjacent modes,
/// the first qubit of the matrix being the lower mode.
fn majoranas2() -> [Mat4; 4] {
    [
        kron(&PX, &ID2),
        kron(&PY, &ID2),
        kron(&PZ, &PX),
        kron(&PZ, &PY),
    ]
}

/// `U† c_i U = Σ_j q_ij c_j` on two adjacent modes, and the residual
/// `max_i ||U† c_i U − Σ_j q_ij c_j||_F / ||c_i||_F`.
pub(crate) fn majorana_map2(u: &Mat4) -> ([[f64; 4]; 4], f64) {
    let c = majoranas2();
    let ud = dag4(u);
    let mut q = [[0.0; 4]; 4];
    let mut res: f64 = 0.0;
    for i in 0..4 {
        let w = mul4(&ud, &mul4(&c[i], u));
        for j in 0..4 {
            // Tr(c_j w) / 4; c_j is Hermitian and w is Hermitian
            let mut t = Z;
            for k in 0..4 {
                for l in 0..4 {
                    t += c[j][k][l] * w[l][k];
                }
            }
            q[i][j] = t.re / 4.0;
        }
        let mut d2 = 0.0;
        for k in 0..4 {
            for l in 0..4 {
                let mut r = w[k][l];
                for j in 0..4 {
                    r -= c[j][k][l] * q[i][j];
                }
                d2 += r.norm_sqr();
            }
        }
        res = res.max((d2 / 4.0).sqrt());
    }
    (q, res)
}

/// The one-mode version of [`majorana_map2`] (`c = X, Y`).
pub(crate) fn majorana_map1(u: &Mat2) -> ([[f64; 2]; 2], f64) {
    let c = [PX, PY];
    let ud = dag2(u);
    let mut q = [[0.0; 2]; 2];
    let mut res: f64 = 0.0;
    for i in 0..2 {
        let w = mul2(&ud, &mul2(&c[i], u));
        for j in 0..2 {
            let mut t = Z;
            for k in 0..2 {
                for l in 0..2 {
                    t += c[j][k][l] * w[l][k];
                }
            }
            q[i][j] = t.re / 2.0;
        }
        let mut d2 = 0.0;
        for k in 0..2 {
            for l in 0..2 {
                let mut r = w[k][l];
                for j in 0..2 {
                    r -= c[j][k][l] * q[i][j];
                }
                d2 += r.norm_sqr();
            }
        }
        res = res.max((d2 / 2.0).sqrt());
    }
    (q, res)
}

fn wrap(t: f64) -> f64 {
    use std::f64::consts::PI;
    let mut x = t.rem_euclid(2.0 * PI);
    if x > PI {
        x -= 2.0 * PI;
    }
    x
}

/// Off-diagonal Frobenius norm of a 4x4 block.
fn offdiag4(u: &Mat4) -> f64 {
    let mut s = 0.0;
    for (i, row) in u.iter().enumerate() {
        for (j, x) in row.iter().enumerate() {
            if i != j {
                s += x.norm_sqr();
            }
        }
    }
    s.sqrt()
}

/// One-site phases `(α_first, α_second)` and interaction phase `g` of a
/// diagonal block: `U = e^{iφ00} e^{iα_first n_first} e^{iα_second n_second}
/// e^{i g n_first n_second}`.
fn diag_phases(u: &Mat4) -> (f64, f64, f64) {
    let p: Vec<f64> = (0..4).map(|k| u[k][k].arg()).collect();
    (
        wrap(p[2] - p[0]),
        wrap(p[1] - p[0]),
        wrap(p[0] - p[1] - p[2] + p[3]),
    )
}

fn swap4() -> Mat4 {
    let mut m = [[Z; 4]; 4];
    m[0][0] = O;
    m[1][2] = O;
    m[2][1] = O;
    m[3][3] = O;
    m
}

/// Classification of one two-qubit block.
#[derive(Clone, Copy, Debug)]
enum Kind2 {
    /// Diagonal: one-site phases and the interaction phase `g`.
    Diag { g: f64 },
    /// A matchgate (needs adjacent modes).
    Match,
    /// Neither.
    NonGaussian,
}

#[derive(Clone, Copy, Debug)]
struct Class2 {
    kind: Kind2,
    /// The block is `SWAP · v`: apply `v`, then rename.
    swap: bool,
    /// Matchgate residual (0 for diagonal blocks).
    residual: f64,
    /// The block to apply (`u`, or `SWAP · u` when `swap`).
    v: Mat4,
}

fn classify2(u: &Mat4, opts: &DetectOptions) -> Class2 {
    let tol = opts.tol;
    let mk = |kind: Kind2, swap: bool, residual: f64, v: &Mat4| Class2 {
        kind,
        swap,
        residual,
        v: *v,
    };
    let sw = opts.relabel_swaps.then(|| mul4(&swap4(), u));
    // 1. diagonal without interaction: one-site phases (any two wires)
    let diag_u = offdiag4(u) <= tol;
    if diag_u {
        let (_, _, g) = diag_phases(u);
        if g.abs() <= tol {
            return mk(Kind2::Diag { g }, false, 0.0, u);
        }
    }
    // 2. a matchgate (adjacent wires)
    let (_, r0) = majorana_map2(u);
    if r0 <= tol {
        return mk(Kind2::Match, false, r0, u);
    }
    // 3. SWAP times a matchgate or a phase-only block (CZ = SWAP · fSWAP:
    //    an interaction phase of π is a fermionic SWAP plus a renaming)
    let mut r = r0;
    if let Some(v) = &sw {
        if offdiag4(v) <= tol {
            let (_, _, g) = diag_phases(v);
            if g.abs() <= tol {
                return mk(Kind2::Diag { g }, true, 0.0, v);
            }
        }
        let (_, r1) = majorana_map2(v);
        if r1 <= tol {
            return mk(Kind2::Match, true, r1, v);
        }
        r = r.min(r1);
    }
    // 4. a genuine interaction phase
    if diag_u {
        let (_, _, g) = diag_phases(u);
        return mk(Kind2::Diag { g }, false, 0.0, u);
    }
    if let Some(v) = &sw {
        if offdiag4(v) <= tol {
            let (_, _, g) = diag_phases(v);
            return mk(Kind2::Diag { g }, true, 0.0, v);
        }
    }
    mk(Kind2::NonGaussian, false, r, u)
}

// ---------------------------------------------------------------------------
// fusion

/// A run of one-qubit gates on one slot between two blocks.
struct Seg {
    gates: Vec<Mat2>,
    prev: Option<usize>,
    /// `gates[..split]` belong to `prev`, the rest to the next block (or
    /// form a standalone one-qubit block).
    split: usize,
}

#[allow(clippy::large_enum_variant)] // one per block, short-lived
enum RawKind {
    /// Two-qubit block on slots `(a, b)`, `a` the first qubit of `core`.
    Two { a: usize, b: usize, core: Mat4 },
    /// A gate the detector does not fuse (three-qubit).
    Opaque { slots: Vec<usize> },
}

struct Raw {
    kind: RawKind,
    /// Opening segment of each slot (in the order of the slots).
    open: Vec<usize>,
    /// Closing segment of each slot, filled when the next segment starts.
    close: Vec<Option<usize>>,
}

/// One fused, classified block in terms of wires (after every renaming).
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)] // one per block
enum Item {
    One {
        w: usize,
        u: Mat2,
    },
    Two {
        a: usize,
        b: usize,
        v: Mat4,
        kind: Kind2,
        residual: f64,
    },
    Opaque,
}

/// A cut choice: (score, gates taken from tail a, from tail b, class).
type Candidate = ((u8, f64, usize), usize, usize, Class2);

/// Longest prefix/suffix of a tail whose cut points are all tried.
const CUT_WINDOW: usize = 8;

fn time_product(gates: &[Mat2]) -> Mat2 {
    gates.iter().fold(ID2, |acc, g| mul2(g, &acc))
}

/// Merges consecutive diagonal 2x2 matrices (they commute with the split
/// choice: a phase is Gaussian on either side).
fn compress(gates: Vec<Mat2>) -> Vec<Mat2> {
    let diag = |m: &Mat2| m[0][1] == Z && m[1][0] == Z;
    let mut out: Vec<Mat2> = Vec::with_capacity(gates.len());
    for g in gates {
        if let Some(last) = out.last_mut() {
            if diag(last) && diag(&g) {
                *last = mul2(&g, last);
                continue;
            }
        }
        out.push(g);
    }
    out
}

struct Fused {
    items: Vec<Item>,
    /// Final slot of each physical qubit (explicit SWAPs).
    slot_of_qubit: Vec<usize>,
    /// Content (wire) in each slot at the end (SWAP-equivalent blocks).
    content: Vec<usize>,
    swaps: usize,
}

fn fuse(c: &Circuit, opts: &DetectOptions) -> Fused {
    let n = c.num_qubits;
    let mut lab: Vec<usize> = (0..n).collect(); // physical -> slot
    let mut tail: Vec<Vec<Mat2>> = vec![Vec::new(); n];
    let mut last: Vec<Option<usize>> = vec![None; n];
    let mut open_blk: Vec<Option<usize>> = vec![None; n];
    let mut raws: Vec<Raw> = Vec::new();
    let mut segs: Vec<Seg> = Vec::new();
    let mut swaps = 0usize;

    // starts a new block on `slots`: closes the tails into segments
    let start = |slots: &[usize],
                 kind: RawKind,
                 tail: &mut Vec<Vec<Mat2>>,
                 last: &mut Vec<Option<usize>>,
                 raws: &mut Vec<Raw>,
                 segs: &mut Vec<Seg>|
     -> usize {
        let k = raws.len();
        let mut open = Vec::with_capacity(slots.len());
        for &w in slots {
            let s = segs.len();
            let prev = last[w];
            segs.push(Seg {
                gates: compress(std::mem::take(&mut tail[w])),
                prev,
                split: 0,
            });
            if let Some(p) = prev {
                let r: &mut Raw = &mut raws[p];
                let i = slots_of(&r.kind)
                    .iter()
                    .position(|&x| x == w)
                    .expect("slot");
                r.close[i] = Some(s);
            }
            open.push(s);
            last[w] = Some(k);
        }
        raws.push(Raw {
            kind,
            open,
            close: vec![None; slots.len()],
        });
        k
    };

    for g in c.gates() {
        match *g {
            Gate::Swap(p, q) if opts.relabel_swaps => {
                lab.swap(p, q);
                swaps += 1;
            }
            ref g1 if g1.arity() == 1 => {
                let w = lab[g1.qubits()[0]];
                tail[w].push(g1.matrix_1q().expect("one-qubit gate"));
            }
            ref g2 if g2.arity() == 2 => {
                let qs = g2.qubits();
                let (a, b) = (lab[qs[0]], lab[qs[1]]);
                let m = g2.matrix_2q().expect("two-qubit gate");
                match (open_blk[a], open_blk[b]) {
                    (Some(k), Some(k2)) if k == k2 => {
                        let RawKind::Two { a: a0, core, .. } = &mut raws[k].kind else {
                            unreachable!("open blocks are two-qubit blocks")
                        };
                        // gate in the block's orientation
                        let (m, ta, tb) = if *a0 == a {
                            (m, a, b)
                        } else {
                            (mat4_swap_qubits(&m), b, a)
                        };
                        let t = kron(&time_product(&tail[ta]), &time_product(&tail[tb]));
                        tail[ta].clear();
                        tail[tb].clear();
                        *core = mul4(&m, &mul4(&t, core));
                    }
                    _ => {
                        let k = start(
                            &[a, b],
                            RawKind::Two { a, b, core: m },
                            &mut tail,
                            &mut last,
                            &mut raws,
                            &mut segs,
                        );
                        open_blk[a] = Some(k);
                        open_blk[b] = Some(k);
                    }
                }
            }
            ref g3 => {
                let slots: Vec<usize> = g3.qubits().iter().map(|&q| lab[q]).collect();
                start(
                    &slots,
                    RawKind::Opaque {
                        slots: slots.clone(),
                    },
                    &mut tail,
                    &mut last,
                    &mut raws,
                    &mut segs,
                );
                for &w in &slots {
                    open_blk[w] = None;
                }
            }
        }
    }
    // final tails
    let mut final_seg: Vec<usize> = Vec::with_capacity(n);
    for w in 0..n {
        let s = segs.len();
        let prev = last[w];
        segs.push(Seg {
            gates: compress(std::mem::take(&mut tail[w])),
            prev,
            split: 0,
        });
        if let Some(p) = prev {
            let r = &mut raws[p];
            let i = slots_of(&r.kind)
                .iter()
                .position(|&x| x == w)
                .expect("slot");
            r.close[i] = Some(s);
        }
        final_seg.push(s);
    }

    // Resolve the splits block by block (creation order is a topological
    // order) and classify; apply SWAP-equivalent renamings on the way.
    let mut content: Vec<usize> = (0..n).collect(); // slot -> wire
    let mut items: Vec<Item> = Vec::with_capacity(raws.len() + n);
    for raw in &raws {
        let open: Vec<Mat2> = raw
            .open
            .iter()
            .map(|&s| time_product(&segs[s].gates[segs[s].split..]))
            .collect();
        match &raw.kind {
            RawKind::Opaque { .. } => {
                // everything after an opaque block goes to the next block
                for s in raw.close.iter().flatten() {
                    segs[*s].split = 0;
                }
                items.push(Item::Opaque);
            }
            RawKind::Two { a, b, core } => {
                let base = mul4(core, &kron(&open[0], &open[1]));
                // candidate cut points: all of them for short tails, the
                // first and last `CUT_WINDOW` for long ones
                let prefixes = |s: Option<usize>| -> Vec<(usize, Mat2)> {
                    let mut v = vec![(0, ID2)];
                    if let Some(s) = s {
                        let len = segs[s].gates.len();
                        let mut acc = ID2;
                        for (k, g) in segs[s].gates.iter().enumerate() {
                            acc = mul2(g, &acc);
                            if k < CUT_WINDOW || k + 1 + CUT_WINDOW >= len {
                                v.push((k + 1, acc));
                            }
                        }
                    }
                    v
                };
                let (pa, pb) = (prefixes(raw.close[0]), prefixes(raw.close[1]));
                // score: (0 Gaussian / 1 interaction / 2 neither, residual,
                // gates taken), lexicographic
                let mut best: Option<Candidate> = None;
                for &(ka, ref ma) in &pa {
                    for &(kb, ref mb) in &pb {
                        let u = mul4(&kron(ma, mb), &base);
                        let cl = classify2(&u, opts);
                        let rank = match cl.kind {
                            Kind2::Diag { g } if g.abs() > opts.tol => 1,
                            Kind2::NonGaussian => 2,
                            _ => 0,
                        };
                        let score = (rank, if rank == 2 { cl.residual } else { 0.0 }, ka + kb);
                        let better = best.as_ref().is_none_or(|(s, ..)| {
                            score.0 < s.0
                                || (score.0 == s.0
                                    && (score.1 < s.1 || (score.1 == s.1 && score.2 < s.2)))
                        });
                        if better {
                            best = Some((score, ka, kb, cl));
                        }
                    }
                }
                let (_, ka, kb, cl) = best.expect("at least one candidate");
                if let Some(s) = raw.close[0] {
                    segs[s].split = ka;
                }
                if let Some(s) = raw.close[1] {
                    segs[s].split = kb;
                }
                let (wa, wb) = (content[*a], content[*b]);
                items.push(Item::Two {
                    a: wa,
                    b: wb,
                    v: cl.v,
                    kind: cl.kind,
                    residual: cl.residual,
                });
                if cl.swap {
                    content.swap(*a, *b);
                    swaps += 1;
                }
            }
        }
    }
    // standalone one-qubit remainders (after the last block on the slot)
    for (w, &s) in final_seg.iter().enumerate() {
        let seg = &segs[s];
        let rest = &seg.gates[if seg.prev.is_some() { seg.split } else { 0 }..];
        if !rest.is_empty() {
            items.push(Item::One {
                w: content[w],
                u: time_product(rest),
            });
        }
    }
    Fused {
        items,
        slot_of_qubit: lab,
        content,
        swaps,
    }
}

fn slots_of(k: &RawKind) -> Vec<usize> {
    match k {
        RawKind::Two { a, b, .. } => vec![*a, *b],
        RawKind::Opaque { slots } => slots.clone(),
    }
}

// ---------------------------------------------------------------------------
// ordering

struct Dsu(Vec<usize>);
impl Dsu {
    fn find(&mut self, x: usize) -> usize {
        let mut r = x;
        while self.0[r] != r {
            r = self.0[r];
        }
        let mut y = x;
        while self.0[y] != r {
            let nx = self.0[y];
            self.0[y] = r;
            y = nx;
        }
        r
    }
}

/// Orders `n` wires along a path cover of the weighted edges (greedy,
/// heaviest first); returns the order and whether the cover kept every edge.
fn path_order(n: usize, edges: &[((usize, usize), usize)]) -> (Vec<usize>, Vec<Vec<usize>>, bool) {
    let mut sorted: Vec<&((usize, usize), usize)> = edges.iter().collect();
    sorted.sort_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
    let mut deg = vec![0usize; n];
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut dsu = Dsu((0..n).collect());
    let mut all = true;
    for &&((a, b), _) in &sorted {
        if deg[a] >= 2 || deg[b] >= 2 || dsu.find(a) == dsu.find(b) {
            all = false;
            continue;
        }
        let (ra, rb) = (dsu.find(a), dsu.find(b));
        dsu.0[ra] = rb;
        deg[a] += 1;
        deg[b] += 1;
        adj[a].push(b);
        adj[b].push(a);
    }
    let mut seen = vec![false; n];
    let mut paths: Vec<Vec<usize>> = Vec::new();
    for s in 0..n {
        if seen[s] || deg[s] > 1 {
            continue;
        }
        // s is an end (degree 0 or 1) not yet visited: the smaller end
        let mut p = vec![s];
        seen[s] = true;
        let mut cur = s;
        while let Some(&nx) = adj[cur].iter().find(|&&x| !seen[x]) {
            seen[nx] = true;
            p.push(nx);
            cur = nx;
        }
        paths.push(p);
    }
    debug_assert!(seen.iter().all(|&x| x), "a path cover has no cycles");
    let order = paths.iter().flatten().copied().collect();
    (order, paths, all)
}

// ---------------------------------------------------------------------------
// the detector

/// Runs the detector (block fusion, classification, ordering) and compiles
/// the Gaussian program when the circuit is Gaussian up to interaction
/// phases. Only `Op::Gate` operations are considered.
pub fn compile(c: &Circuit, opts: &DetectOptions) -> GaussianProgram {
    let t0 = std::time::Instant::now();
    let n = c.num_qubits;
    let tol = opts.tol;
    let fused = fuse(c, opts);
    let items = &fused.items;

    // edges of non-diagonal Gaussian blocks
    let mut w: std::collections::BTreeMap<(usize, usize), usize> = Default::default();
    for it in items {
        if let Item::Two {
            a,
            b,
            kind: Kind2::Match,
            ..
        } = it
        {
            *w.entry(((*a).min(*b), (*a).max(*b))).or_insert(0) += 1;
        }
    }
    let edges: Vec<((usize, usize), usize)> = w.into_iter().collect();
    let covered = |pos: &[usize]| -> usize {
        edges
            .iter()
            .filter(|((a, b), _)| pos[*a].abs_diff(pos[*b]) == 1)
            .map(|(_, k)| *k)
            .sum()
    };
    let inverse = |order: &[usize]| -> Vec<usize> {
        let mut pos = vec![0; n];
        for (k, &x) in order.iter().enumerate() {
            pos[x] = k;
        }
        pos
    };
    let ident: Vec<usize> = (0..n).collect();
    let total: usize = edges.iter().map(|e| e.1).sum();
    let id_cov = covered(&ident);
    let first_path = if n == 0 { vec![] } else { vec![ident.clone()] };
    let (mut order, mut paths, mut kind) = (ident.clone(), first_path, Ordering::Identity);
    if opts.reorder && id_cov < total {
        let (o, p, all) = path_order(n, &edges);
        if covered(&inverse(&o)) > id_cov {
            order = o;
            paths = p;
            kind = if all {
                Ordering::Paths
            } else {
                Ordering::GreedyCover
            };
        }
    }
    let pos = inverse(&order);

    let mut rep = GaussianReport {
        n,
        blocks: items.len(),
        ordering: Some(kind),
        swaps_relabelled: fused.swaps,
        number_conserving: true,
        ..Default::default()
    };
    let mut ops: Vec<GaussOp> = Vec::with_capacity(items.len() * 2);
    let phase_q = |alpha: f64| -> [[f64; 2]; 2] {
        // e^{iα n}: rotation of (γ_2k, γ_2k+1) by α
        let u: Mat2 = [[O, Z], [Z, C::from_polar(1.0, alpha)]];
        majorana_map1(&u).0
    };
    let commutes_j2 =
        |q: &[[f64; 2]; 2]| (q[0][0] - q[1][1]).abs() < 1e-9 && (q[0][1] + q[1][0]).abs() < 1e-9;
    for (idx, it) in items.iter().enumerate() {
        match it {
            Item::One { w, u } => {
                let (q, r) = majorana_map1(u);
                rep.max_residual = rep.max_residual.max(r);
                if r <= tol {
                    rep.gaussian_blocks += 1;
                    if !commutes_j2(&q) {
                        rep.number_conserving = false;
                    }
                    ops.push(GaussOp::One { mode: pos[*w], q });
                } else {
                    rep.non_gaussian += 1;
                }
            }
            Item::Opaque => {
                rep.non_gaussian += 1;
                rep.max_residual = rep.max_residual.max(1.0);
            }
            Item::Two {
                a,
                b,
                v,
                kind,
                residual,
            } => {
                rep.blocks_2q += 1;
                match kind {
                    Kind2::Diag { g } => {
                        let (al_a, al_b, _) = diag_phases(v);
                        for (wire, al) in [(*a, al_a), (*b, al_b)] {
                            if al != 0.0 {
                                ops.push(GaussOp::One {
                                    mode: pos[wire],
                                    q: phase_q(al),
                                });
                            }
                        }
                        if g.abs() <= tol {
                            rep.gaussian_blocks += 1;
                        } else {
                            let ip = InteractionPhase {
                                block: idx,
                                wires: (*a, *b),
                                modes: (pos[*a], pos[*b]),
                                g: *g,
                            };
                            rep.interaction_total += g.abs();
                            rep.interaction_max = rep.interaction_max.max(g.abs());
                            rep.interactions.push(ip);
                            ops.push(GaussOp::Interaction(ip));
                        }
                    }
                    Kind2::Match => {
                        let (pa, pb) = (pos[*a], pos[*b]);
                        let v = if pa + 1 == pb {
                            Some(*v)
                        } else if pb + 1 == pa {
                            Some(mat4_swap_qubits(v))
                        } else {
                            None
                        };
                        match v {
                            Some(v) => {
                                let (q, r) = majorana_map2(&v);
                                rep.max_residual = rep.max_residual.max(r.max(*residual));
                                rep.gaussian_blocks += 1;
                                if !number_conserving4(&q) {
                                    rep.number_conserving = false;
                                }
                                ops.push(GaussOp::Two {
                                    mode: pa.min(pb),
                                    q,
                                });
                            }
                            None => {
                                rep.non_gaussian += 1;
                                rep.nonadjacent += 1;
                                rep.max_residual = rep.max_residual.max(1.0);
                            }
                        }
                    }
                    Kind2::NonGaussian => {
                        rep.non_gaussian += 1;
                        rep.max_residual = rep.max_residual.max(*residual);
                    }
                }
            }
        }
    }
    rep.gaussian_fraction = if rep.blocks == 0 {
        1.0
    } else {
        rep.gaussian_blocks as f64 / rep.blocks as f64
    };
    rep.free = rep.non_gaussian == 0;
    rep.exact = rep.free && rep.interactions.is_empty();
    rep.mode_of_qubit = (0..n)
        .map(|q| pos[fused.content[fused.slot_of_qubit[q]]])
        .collect();
    rep.order = order;
    rep.paths = paths;
    if !rep.free {
        ops.clear();
    }
    rep.secs = t0.elapsed().as_secs_f64();
    GaussianProgram { report: rep, ops }
}

/// `q` commutes with the complex structure of both modes and has det +1.
fn number_conserving4(q: &[[f64; 4]; 4]) -> bool {
    // J = diag(j, j), j = [[0, 1], [-1, 0]]; (qJ)_ik = Σ_j q_ij J_jk
    let jm = |i: usize, k: usize| -> f64 {
        if i / 2 != k / 2 {
            return 0.0;
        }
        match (i % 2, k % 2) {
            (0, 1) => 1.0,
            (1, 0) => -1.0,
            _ => 0.0,
        }
    };
    for i in 0..4 {
        for k in 0..4 {
            let qj: f64 = (0..4).map(|j| q[i][j] * jm(j, k)).sum();
            let jq: f64 = (0..4).map(|j| jm(i, j) * q[j][k]).sum();
            if (qj - jq).abs() > 1e-9 {
                return false;
            }
        }
    }
    true
}

/// The detector's report alone (see [`compile`]).
pub fn detect(c: &Circuit, opts: &DetectOptions) -> GaussianReport {
    compile(c, opts).report
}
