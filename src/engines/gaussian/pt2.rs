//! Exact second-order perturbation theory in weak interaction phases
//! (docs/ENGINE_GAUSSIAN.md §6), for number-conserving circuits.
//!
//! A circuit that is Gaussian up to interaction phases is
//! `U(λ) = F_m V_m F_{m-1} … V_1 F_0` with free (Gaussian) parts `F` and
//! vertices `V_k = exp(iλ g_k n_a n_b)`. Because `(n_a n_b)^2 = n_a n_b`,
//! `V_k = 1 + c_k P_k` exactly, with `c_k = e^{iλ g_k} − 1`. In the
//! interaction picture of the free evolution, `U(λ) = F ∏_k (1 + c_k P_k)`
//! (later vertices to the left), where `P_k = ñ_a(t_k) ñ_b(t_k)` and
//! `ñ_a(t) = c_a(t)† c_a(t)` with the free Heisenberg operator
//! `c_a(t) = Σ_l W(t)_{al} c_l`. Expanding `⟨O⟩(λ) = a0 + a1 λ + a2 λ² +
//! O(λ³)` for a one-body observable `O` (evaluated at the end of the
//! circuit, `O_H = F† O F`) in the initial basis state gives
//!
//! * `a0 = ⟨O_H⟩`,
//! * `a1 = −2 Σ_k g_k Im⟨O_H P_k⟩`,
//! * `a2 = −Σ_k g_k² Re⟨O_H P_k⟩ + Σ_{k,j} g_k g_j Re⟨P_k O_H P_j⟩
//!   − 2 Re Σ_{k>j} g_k g_j ⟨O_H P_k P_j⟩`.
//!
//! Every expectation value is a product of at most five fermion bilinears
//! in a Slater (Fock) state, evaluated with Wick's theorem for
//! number-conserving states: `⟨Π_i c†(x_i) c(y_i)⟩ = det G̃` with
//! `G̃_ij = ⟨c†(x_i) c(y_j)⟩` for `i ≤ j` and `−⟨c(y_j) c†(x_i)⟩` for
//! `i > j`. The one-body `O_H = Σ_{lm} M_lm c†_l c_m` is not a single
//! bilinear; its row and column of `G̃` are linear in `l` and `m`, so the
//! terms of the determinant that use both are contracted with `M` first
//! (the matrices `S^{AB}` below). The single-particle propagator `W(t)`
//! is the product of the blocks' `n x n` single-particle unitaries, read off
//! the Majorana maps of the compiled program.
//!
//! The coefficients are exact (no truncation, round-off only); only
//! `total = a0 + a1 λ + a2 λ²` is an approximation of `⟨O⟩(λ)`, and
//! `est_error` is a heuristic for the size of the neglected terms, **not a
//! bound**. Cost: `O(K n²)` for the propagation, `O(K² n)` for the
//! correlation matrices and `O(K_lc K)` five-point Wick sums, with `K` the
//! number of interaction phases and `K_lc ≤ K` the number inside the free
//! light cone of the observable; memory `5 (2K)² · 16` bytes.

use super::detect::{compile, DetectOptions, GaussOp, GaussianProgram, GaussianReport};
use crate::circuit::{Circuit, Op, SimError};
use crate::gate::Gate;
use num_complex::Complex64 as C;

const CZ: C = C::new(0.0, 0.0);

/// A one-body observable `constant + Σ_i c_i n_i` on Jordan–Wigner modes
/// (in [`pt2`]) or on qubits at the end of the circuit (in
/// [`pt2_circuit`]). `Z_i = 1 − 2 n_i` is added with [`OneBody::z`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OneBody {
    /// The constant term.
    pub constant: f64,
    /// Density terms `(index, c_i)`; repeated indices add up.
    pub density: Vec<(usize, f64)>,
}

impl OneBody {
    /// The zero observable.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `c n_i`.
    pub fn n(mut self, i: usize, c: f64) -> Self {
        self.density.push((i, c));
        self
    }

    /// Adds `c Z_i = c (1 − 2 n_i)`.
    pub fn z(mut self, i: usize, c: f64) -> Self {
        self.constant += c;
        self.density.push((i, -2.0 * c));
        self
    }
}

/// Settings of [`pt2`] and [`pt2_circuit`].
#[derive(Clone, Copy, Debug)]
pub struct Pt2Options {
    /// The coupling `λ` at which `total` and `est_error` are evaluated (the
    /// circuit itself is `λ = 1`).
    pub lambda: f64,
    /// Largest admissible leakage of a block out of the number-conserving
    /// set (the norm of the `c†` part of `U† c U`), default `1e-10`.
    pub leak_tol: f64,
    /// Largest working memory (bytes) of the correlation matrices.
    pub max_bytes: u128,
    /// Detector settings ([`pt2_circuit`] only).
    pub detect: DetectOptions,
}

impl Default for Pt2Options {
    fn default() -> Self {
        Pt2Options {
            lambda: 1.0,
            leak_tol: 1e-10,
            max_bytes: 2 << 30,
            detect: DetectOptions::default(),
        }
    }
}

/// Second-order expansion `⟨O⟩(λ) ≈ a0 + a1 λ + a2 λ²` of one observable.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pt2Result {
    /// `⟨O⟩` of the free circuit (all `g = 0`).
    pub a0: f64,
    /// Exact first-order coefficient.
    pub a1: f64,
    /// Exact second-order coefficient.
    pub a2: f64,
    /// `a0 + a1 λ + a2 λ²` at `λ = Pt2Options::lambda`.
    pub total: f64,
    /// Heuristic size of the neglected third-order term,
    /// `|a2| λ³ max|g| sqrt(light_cone)`: an estimate, **not a bound**.
    pub est_error: f64,
    /// Interaction phases inside the free light cone of the observable
    /// (the others do not contribute to `a1` or `a2`).
    pub light_cone: usize,
}

/// The result of [`pt2_circuit`].
#[derive(Clone, Debug)]
pub struct Pt2Run {
    /// One entry per observable.
    pub results: Vec<Pt2Result>,
    /// Detector report of the circuit without its initial `X` layer.
    pub report: GaussianReport,
    /// Initial occupation of each qubit (from the leading `X` gates).
    pub initial: Vec<bool>,
}

/// Splits off the basis-state preparation: every `X` gate on a qubit that
/// no other operation has touched yet flips that qubit's initial
/// occupation and is removed. Returns the remaining circuit and the
/// initial occupations (qubit `q` = bit `q`).
pub fn split_initial_x(c: &Circuit) -> (Circuit, Vec<bool>) {
    let n = c.num_qubits;
    let mut occ = vec![false; n];
    let mut touched = vec![false; n];
    let mut rest = Circuit::new(n);
    for op in &c.ops {
        match op {
            Op::Gate(Gate::X(q)) if !touched[*q] => occ[*q] ^= true,
            Op::Gate(g) => {
                for q in g.qubits() {
                    touched[q] = true;
                }
                rest.ops.push(op.clone());
            }
            _ => {
                touched.iter_mut().for_each(|t| *t = true);
                rest.ops.push(op.clone());
            }
        }
    }
    (rest, occ)
}

/// Heisenberg map `U† c_r U = Σ_s w_rs c_s` of a block with Majorana map
/// `q` (`2k x 2k`, row-major, `γ_{2r} = c_r + c_r†`,
/// `γ_{2r+1} = i(c_r† − c_r)`) on `k` modes, and the leakage: the largest
/// coefficient of a `c_s†` (zero iff the block conserves the particle
/// number).
fn sp_map(q: &[f64], k: usize) -> ([[C; 2]; 2], f64) {
    let d = 2 * k;
    let at = |i: usize, j: usize| q[i * d + j];
    let mut w = [[CZ; 2]; 2];
    let mut leak: f64 = 0.0;
    for r in 0..k {
        for s in 0..k {
            let (a, b, c, e) = (
                at(2 * r, 2 * s),
                at(2 * r, 2 * s + 1),
                at(2 * r + 1, 2 * s),
                at(2 * r + 1, 2 * s + 1),
            );
            w[r][s] = C::new(0.5 * (a + e), 0.5 * (c - b));
            leak = leak.max(0.5 * C::new(a - e, c + b).norm());
        }
    }
    (w, leak)
}

/// All permutations of `0..n` with their signs and inverses.
fn permutations(n: usize) -> Vec<(f64, Vec<usize>, Vec<usize>)> {
    fn rec(
        n: usize,
        cur: &mut Vec<usize>,
        used: &mut Vec<bool>,
        out: &mut Vec<(f64, Vec<usize>, Vec<usize>)>,
    ) {
        if cur.len() == n {
            let mut inv = vec![0; n];
            let mut inversions = 0;
            for i in 0..n {
                inv[cur[i]] = i;
                for j in i + 1..n {
                    if cur[i] > cur[j] {
                        inversions += 1;
                    }
                }
            }
            let s = if inversions % 2 == 0 { 1.0 } else { -1.0 };
            out.push((s, cur.clone(), inv));
            return;
        }
        for v in 0..n {
            if !used[v] {
                used[v] = true;
                cur.push(v);
                rec(n, cur, used, out);
                cur.pop();
                used[v] = false;
            }
        }
    }
    let mut out = Vec::new();
    rec(n, &mut Vec::new(), &mut vec![false; n], &mut out);
    out
}

/// Correlation matrices over the `2K` vertex vectors `φ_α` (rows of `W`
/// at the vertices) for one observable.
struct Wick<'a> {
    dim: usize,
    /// `Gp[α][β] = ⟨c†(φ_α) c(φ_β)⟩`.
    gp: &'a [C],
    /// `Gm[α][β] = ⟨c(φ_β) c†(φ_α)⟩`.
    gm: &'a [C],
    /// `S^{AB}[α][β] = Σ_lm φ_αl A_l M_lm B_m conj(φ_βm)` with `A, B` the
    /// occupation `D` or `D̄ = 1 − D`: `S^{DD}`, `S^{D̄D̄}`, `S^{DD̄}`
    /// (`S^{D̄D}[α][β] = conj(S^{DD̄}[β][α])`).
    sdd: &'a [C],
    sbb: &'a [C],
    sdb: &'a [C],
    /// `⟨O_H⟩` without the constant.
    d: C,
}

impl Wick<'_> {
    /// `⟨Π_i B_i⟩` for the bilinears `seq[i]`: `Some(α)` is `ñ(φ_α)`,
    /// `None` (exactly once) is `O_H`. `perms` are the permutations of
    /// `seq.len()` elements.
    fn corr(&self, seq: &[Option<usize>], perms: &[(f64, Vec<usize>, Vec<usize>)]) -> C {
        let nn = seq.len();
        debug_assert!(nn <= 5);
        let p = seq.iter().position(|x| x.is_none()).expect("one observable");
        let dim = self.dim;
        let mut gt = [[CZ; 5]; 5];
        let mut xm = [[CZ; 5]; 5];
        for i in 0..nn {
            let Some(ai) = seq[i] else { continue };
            for j in 0..nn {
                let Some(aj) = seq[j] else { continue };
                gt[i][j] = if i <= j {
                    self.gp[ai * dim + aj]
                } else {
                    -self.gm[ai * dim + aj]
                };
            }
        }
        // xm[j][i]: entry (p, j) of G̃ times entry (i, p), summed against M.
        // Row p: ⟨c†_l c(y_j)⟩ = D_l y_jl (j > p) or −D̄_l y_jl (j < p);
        // column p: ⟨c†(x_i) c_m⟩ = conj(x_im) D_m (i < p) or −D̄_m (i > p).
        for j in 0..nn {
            let Some(aj) = seq[j] else { continue };
            for i in 0..nn {
                let Some(ai) = seq[i] else { continue };
                let s = match (j > p, i < p) {
                    (true, true) => self.sdd[aj * dim + ai],
                    (true, false) => -self.sdb[aj * dim + ai],
                    (false, true) => -self.sdb[ai * dim + aj].conj(),
                    (false, false) => self.sbb[aj * dim + ai],
                };
                xm[j][i] = s;
            }
        }
        let mut tot = CZ;
        for (sign, sig, inv) in perms {
            let mut t;
            if sig[p] == p {
                t = self.d;
                for i in 0..nn {
                    if i != p {
                        t *= gt[i][sig[i]];
                    }
                }
            } else {
                let c = inv[p];
                t = xm[sig[p]][c];
                for i in 0..nn {
                    if i != p && i != c {
                        t *= gt[i][sig[i]];
                    }
                }
            }
            tot += t * *sign;
        }
        tot
    }
}

/// Hermitian `A[α][β] = Σ_t w_t conj(X[α][t]) X[β][t]` for the rows of `x`
/// (`rows x cols`, row-major).
fn gram(x: &[C], rows: usize, cols: usize, w: &[f64], conj_first: bool) -> Vec<C> {
    let mut out = vec![CZ; rows * rows];
    for a in 0..rows {
        let xa = &x[a * cols..(a + 1) * cols];
        for b in a..rows {
            let xb = &x[b * cols..(b + 1) * cols];
            let mut s = CZ;
            for t in 0..cols {
                if w[t] != 0.0 {
                    s += xa[t].conj() * xb[t] * w[t];
                }
            }
            if !conj_first {
                s = s.conj();
            }
            out[a * rows + b] = s;
            out[b * rows + a] = s.conj();
        }
    }
    out
}

/// Second-order perturbation theory for the interaction phases of a
/// compiled program that is Gaussian up to interaction phases and whose
/// every Gaussian step conserves the particle number.
///
/// `occupation[k]` is the initial occupation of Jordan–Wigner mode `k`;
/// the observables are on modes at the end of the program. Errors:
/// `NotSupported` when the program is not free, a block is not
/// number-conserving (leakage above `opts.leak_tol`) or the sizes do not
/// match; `QubitOutOfRange` for an observable mode out of range;
/// `TooLarge` when the correlation matrices exceed `opts.max_bytes`.
pub fn pt2(
    prog: &GaussianProgram,
    occupation: &[bool],
    observables: &[OneBody],
    opts: &Pt2Options,
) -> Result<Vec<Pt2Result>, SimError> {
    let r = &prog.report;
    if !r.free {
        return Err(SimError::NotSupported {
            what: "gaussian pt2: the circuit is not Gaussian up to interaction phases",
        });
    }
    let n = r.n;
    if occupation.len() != n {
        return Err(SimError::NotSupported {
            what: "gaussian pt2: one initial occupation per mode is needed",
        });
    }
    for o in observables {
        for &(i, _) in &o.density {
            if i >= n {
                return Err(SimError::QubitOutOfRange {
                    qubit: i,
                    num_qubits: n,
                });
            }
        }
    }
    let k_tot = r.interactions.len();
    let dim = 2 * k_tot;
    let bytes = 5 * (dim as u128) * (dim as u128) * 16 + 16 * (dim as u128) * (n as u128);
    if bytes > opts.max_bytes {
        return Err(SimError::TooLarge {
            what: "gaussian pt2 correlation matrices",
            bytes,
            limit: opts.max_bytes,
        });
    }

    // free single-particle propagation; W row-major n x n
    let mut w = vec![CZ; n * n];
    for i in 0..n {
        w[i * n + i] = C::new(1.0, 0.0);
    }
    let mut phi: Vec<C> = Vec::with_capacity(dim * n);
    let mut g: Vec<f64> = Vec::with_capacity(k_tot);
    let not_nc = SimError::NotSupported {
        what: "gaussian pt2: unsupported: not number-conserving",
    };
    let apply = |w: &mut Vec<C>, mode: usize, u: &[[C; 2]; 2], k: usize| {
        for c in 0..n {
            let old = [w[mode * n + c], if k == 2 { w[(mode + 1) * n + c] } else { CZ }];
            for (rr, row) in u.iter().enumerate().take(k) {
                w[(mode + rr) * n + c] = row[0] * old[0] + row[1] * old[1];
            }
        }
    };
    for op in &prog.ops {
        match op {
            GaussOp::One { mode, q } => {
                let qf = [q[0][0], q[0][1], q[1][0], q[1][1]];
                let (u, leak) = sp_map(&qf, 1);
                if leak > opts.leak_tol {
                    return Err(not_nc);
                }
                apply(&mut w, *mode, &u, 1);
            }
            GaussOp::Two { mode, q } => {
                let qf: Vec<f64> = q.iter().flatten().copied().collect();
                let (u, leak) = sp_map(&qf, 2);
                if leak > opts.leak_tol {
                    return Err(not_nc);
                }
                apply(&mut w, *mode, &u, 2);
            }
            GaussOp::Interaction(ip) => {
                let (a, b) = ip.modes;
                phi.extend_from_slice(&w[a * n..(a + 1) * n]);
                phi.extend_from_slice(&w[b * n..(b + 1) * n]);
                g.push(ip.g);
            }
        }
    }
    debug_assert_eq!(g.len(), k_tot);
    let occ: Vec<f64> = occupation
        .iter()
        .map(|&b| if b { 1.0 } else { 0.0 })
        .collect();
    let unocc: Vec<f64> = occ.iter().map(|x| 1.0 - x).collect();
    // Gp, Gm: Gram matrices of conj(φ) weighted by D, D̄
    let gp = gram(&phi, dim, n, &occ, true);
    let gm = gram(&phi, dim, n, &unocc, true);
    let perms3 = permutations(3);
    let perms5 = permutations(5);
    let gmax = g.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let lam = opts.lambda;

    let mut out = Vec::with_capacity(observables.len());
    for o in observables {
        // merged density terms
        let mut coef = vec![0.0; n];
        for &(i, c) in &o.density {
            coef[i] += c;
        }
        let modes: Vec<usize> = (0..n).filter(|&i| coef[i] != 0.0).collect();
        let cw: Vec<f64> = modes.iter().map(|&i| coef[i]).collect();
        let m = modes.len();
        // d = Σ_i c_i Σ_l |W_il|² D_l
        let mut d = 0.0;
        for (t, &i) in modes.iter().enumerate() {
            let s: f64 = (0..n).map(|l| w[i * n + l].norm_sqr() * occ[l]).sum();
            d += cw[t] * s;
        }
        let a0 = o.constant + d;
        // Y_A[α][t] = Σ_l φ_αl A_l conj(W_{i_t l})
        let mut yd = vec![CZ; dim * m];
        let mut yb = vec![CZ; dim * m];
        for a in 0..dim {
            let ph = &phi[a * n..(a + 1) * n];
            for (t, &i) in modes.iter().enumerate() {
                let wi = &w[i * n..(i + 1) * n];
                let (mut sd, mut sb) = (CZ, CZ);
                for l in 0..n {
                    let x = ph[l] * wi[l].conj();
                    if occ[l] != 0.0 {
                        sd += x;
                    } else {
                        sb += x;
                    }
                }
                yd[a * m + t] = sd;
                yb[a * m + t] = sb;
            }
        }
        // light cone: vertices whose both vectors are orthogonal to every
        // observed mode commute with O_H and drop out when they are the
        // later vertex of a term
        let inside: Vec<bool> = (0..k_tot)
            .map(|k| {
                (2 * k..2 * k + 2).any(|a| {
                    (0..m)
                        .map(|t| (yd[a * m + t] + yb[a * m + t]).norm_sqr())
                        .sum::<f64>()
                        > 1e-26
                })
            })
            .collect();
        let k_lc = inside.iter().filter(|&&x| x).count();
        if k_lc == 0 {
            out.push(Pt2Result {
                a0,
                a1: 0.0,
                a2: 0.0,
                total: a0,
                est_error: 0.0,
                light_cone: 0,
            });
            continue;
        }
        // S^{AB}[α][β] = Σ_t c_t Y_A[α][t] conj(Y_B[β][t])
        let sdd = gram(&yd, dim, m, &cw, false);
        let sbb = gram(&yb, dim, m, &cw, false);
        let mut sdb = vec![CZ; dim * dim];
        for a in 0..dim {
            for b in 0..dim {
                let mut s = CZ;
                for t in 0..m {
                    s += yd[a * m + t] * yb[b * m + t].conj() * cw[t];
                }
                sdb[a * dim + b] = s;
            }
        }
        let wk = Wick {
            dim,
            gp: &gp,
            gm: &gm,
            sdd: &sdd,
            sbb: &sbb,
            sdb: &sdb,
            d: C::new(d, 0.0),
        };
        let (mut a1, mut a2) = (0.0, 0.0);
        for k in 0..k_tot {
            if !inside[k] {
                continue;
            }
            let (ka, kb) = (Some(2 * k), Some(2 * k + 1));
            let op = wk.corr(&[None, ka, kb], &perms3);
            a1 += -2.0 * g[k] * op.im;
            a2 += -g[k] * g[k] * op.re;
            for j in 0..=k {
                let (ja, jb) = (Some(2 * j), Some(2 * j + 1));
                let pop = wk.corr(&[ka, kb, None, ja, jb], &perms5);
                if j == k {
                    a2 += g[k] * g[k] * pop.re;
                } else {
                    let opp = wk.corr(&[None, ka, kb, ja, jb], &perms5);
                    a2 += 2.0 * g[k] * g[j] * (pop.re - opp.re);
                }
            }
        }
        out.push(Pt2Result {
            a0,
            a1,
            a2,
            total: a0 + a1 * lam + a2 * lam * lam,
            est_error: a2.abs() * lam.abs().powi(3) * gmax * (k_lc as f64).sqrt(),
            light_cone: k_lc,
        });
    }
    Ok(out)
}

/// [`pt2`] on a circuit: the leading `X` gates prepare the initial basis
/// state ([`split_initial_x`]), the rest is compiled by the detector, and
/// the observables are on **qubits** at the end of the circuit (mapped to
/// modes through [`GaussianReport::mode_of_qubit`]). Errors as in [`pt2`],
/// plus `MeasurementNotSupported` for a non-gate operation.
pub fn pt2_circuit(
    c: &Circuit,
    observables: &[OneBody],
    opts: &Pt2Options,
) -> Result<Pt2Run, SimError> {
    if let Some(i) = c.ops.iter().position(|o| !matches!(o, Op::Gate(_))) {
        return Err(SimError::MeasurementNotSupported {
            backend: "gaussian pt2",
            op_index: i,
        });
    }
    let (rest, initial) = split_initial_x(c);
    let prog = compile(&rest, &opts.detect);
    let rep = &prog.report;
    let n = c.num_qubits;
    let occ_mode: Vec<bool> = if rep.free {
        rep.order.iter().map(|&wire| initial[wire]).collect()
    } else {
        vec![false; n]
    };
    let mut obs = Vec::with_capacity(observables.len());
    for o in observables {
        let mut m = OneBody {
            constant: o.constant,
            density: Vec::with_capacity(o.density.len()),
        };
        for &(q, x) in &o.density {
            if q >= n {
                return Err(SimError::QubitOutOfRange {
                    qubit: q,
                    num_qubits: n,
                });
            }
            m.density.push((rep.mode_of_qubit[q], x));
        }
        obs.push(m);
    }
    let results = pt2(&prog, &occ_mode, &obs, opts)?;
    Ok(Pt2Run {
        results,
        report: prog.report,
        initial,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_particle_map_of_a_phase() {
        // e^{iα n}: U† c U = e^{iα} c
        let a: f64 = 0.37;
        let q = [a.cos(), -a.sin(), a.sin(), a.cos()];
        let (w, leak) = sp_map(&q, 1);
        assert!(leak < 1e-15);
        assert!((w[0][0] - C::from_polar(1.0, a)).norm() < 1e-15);
        // X: parity-odd, fully leaking
        let (_, leak) = sp_map(&[1.0, 0.0, 0.0, -1.0], 1);
        assert!((leak - 1.0).abs() < 1e-15);
    }

    #[test]
    fn permutation_signs() {
        let p = permutations(3);
        assert_eq!(p.len(), 6);
        assert_eq!(p.iter().map(|x| x.0).sum::<f64>(), 0.0);
        assert_eq!(permutations(5).len(), 120);
    }
}
