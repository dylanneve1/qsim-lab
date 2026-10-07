//! The Gaussian state: the Majorana covariance matrix of `U|0^n⟩` for a
//! free-fermion circuit `U`, and its read-outs (docs/ENGINE_GAUSSIAN.md §2, §4).
//!
//! With `γ_{2k} = Z_0…Z_{k-1} X_k` and `γ_{2k+1} = Z_0…Z_{k-1} Y_k` on the
//! Jordan–Wigner modes `k`, the state is described by the real antisymmetric
//! `2n x 2n` matrix `M_ab = -i ⟨γ_a γ_b⟩` (`a ≠ b`), so that `⟨Z_k⟩ =
//! M_{2k,2k+1}` and the vacuum is `⊕_k [[0, 1], [-1, 0]]`. A Gaussian block
//! with Majorana map `U† γ_a U = Σ_b Q_ab γ_b` sends `M → Q M Qᵀ`: O(n) per
//! block, because `Q` is the identity outside the block's 2 or 4 operators
//! except for a sign `det Q` on every later mode. That sign is kept lazily in
//! a vector `d` (the true matrix is `D M D`), so parity-odd blocks also cost
//! O(n).

use super::detect::{compile, DetectOptions, GaussOp, GaussianProgram, GaussianReport};
use super::InteractionPhase;
use crate::circuit::{Circuit, Op, SimError};
use rand::Rng;

/// What to do with interaction phases `exp(i g n_a n_b)` (the only
/// non-Gaussian blocks the engine knows how to separate).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum InteractionPolicy {
    /// Refuse circuits with a non-zero interaction phase (exact engine).
    #[default]
    Refuse,
    /// Drop them (`g → 0`) and keep the one-site phases of the block: the
    /// free-fermion part of the circuit. Not exact when `g ≠ 0`.
    Drop,
}

/// Settings of [`GaussianState::from_circuit`].
#[derive(Clone, Copy, Debug)]
pub struct GaussianOptions {
    /// Detector settings.
    pub detect: DetectOptions,
    /// Interaction phases.
    pub interactions: InteractionPolicy,
    /// Largest covariance matrix (bytes, `32 n^2`) the engine may allocate.
    pub max_bytes: u128,
}

impl Default for GaussianOptions {
    fn default() -> Self {
        GaussianOptions {
            detect: DetectOptions::default(),
            interactions: InteractionPolicy::Refuse,
            max_bytes: 1 << 30,
        }
    }
}

/// Bytes of the covariance matrix of `n` modes.
pub fn covariance_bytes(n: usize) -> u128 {
    8 * (2 * n as u128) * (2 * n as u128)
}

/// A fermionic Gaussian state on `n` Jordan–Wigner modes, read out in terms
/// of the circuit's qubits at the end of the circuit.
#[derive(Clone, Debug)]
pub struct GaussianState {
    n: usize,
    /// `2n x 2n`, row-major; the covariance is `D m D`.
    m: Vec<f64>,
    /// The diagonal of `D` (±1).
    d: Vec<f64>,
    /// Mode of each qubit at the end of the circuit.
    mode_of_qubit: Vec<usize>,
}

fn not_gaussian(r: &GaussianReport) -> SimError {
    if r.nonadjacent > 0 && r.non_gaussian == r.nonadjacent {
        SimError::NotSupported {
            what: "gaussian: the matchgates do not fit one Jordan-Wigner order",
        }
    } else {
        SimError::NotSupported {
            what: "gaussian: the circuit is not Gaussian (free-fermion) in any order the detector tried",
        }
    }
}

impl GaussianState {
    /// The vacuum `|0^n⟩` with the identity mode order.
    pub fn vacuum(n: usize) -> Self {
        let dim = 2 * n;
        let mut m = vec![0.0; dim * dim];
        for k in 0..n {
            m[(2 * k) * dim + 2 * k + 1] = 1.0;
            m[(2 * k + 1) * dim + 2 * k] = -1.0;
        }
        GaussianState {
            n,
            m,
            d: vec![1.0; dim],
            mode_of_qubit: (0..n).collect(),
        }
    }

    /// Number of modes (= qubits).
    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// The mode qubit `q` holds at the end of the circuit.
    pub fn mode_of_qubit(&self, q: usize) -> usize {
        self.mode_of_qubit[q]
    }

    /// Detects, compiles and evolves `c|0^n⟩`. Fails with `NotSupported`
    /// when the circuit is not Gaussian (or has interaction phases under
    /// [`InteractionPolicy::Refuse`]), `MeasurementNotSupported` on a
    /// non-gate operation, and `TooLarge` when the covariance exceeds
    /// `opts.max_bytes`.
    pub fn from_circuit(
        c: &Circuit,
        opts: &GaussianOptions,
    ) -> Result<(GaussianState, GaussianReport), SimError> {
        if let Some(i) = c.ops.iter().position(|o| !matches!(o, Op::Gate(_))) {
            return Err(SimError::MeasurementNotSupported {
                backend: "gaussian",
                op_index: i,
            });
        }
        let prog = compile(c, &opts.detect);
        let st = Self::evolve(&prog, opts)?;
        Ok((st, prog.report))
    }

    /// Evolves the vacuum through a compiled program (see
    /// [`Self::from_circuit`] for the errors).
    pub fn evolve(
        prog: &GaussianProgram,
        opts: &GaussianOptions,
    ) -> Result<GaussianState, SimError> {
        let r = &prog.report;
        if !r.free {
            return Err(not_gaussian(r));
        }
        if !r.interactions.is_empty() && opts.interactions == InteractionPolicy::Refuse {
            return Err(SimError::NotSupported {
                what: "gaussian: the circuit has non-zero interaction phases \
                       (InteractionPolicy::Drop simulates its free-fermion part)",
            });
        }
        Self::evolve_with(prog, opts.max_bytes, |_, _| {})
    }

    /// Evolves the vacuum through a compiled program that is Gaussian up to
    /// interaction phases, calling `on_interaction(state, phase)` at each
    /// interaction phase with the state just before it; the phase itself is
    /// then dropped. This is the hook a perturbative treatment of the
    /// interactions attaches to (docs/ENGINE_GAUSSIAN.md §5).
    pub fn evolve_with<F>(
        prog: &GaussianProgram,
        max_bytes: u128,
        mut on_interaction: F,
    ) -> Result<GaussianState, SimError>
    where
        F: FnMut(&GaussianState, &InteractionPhase),
    {
        let r = &prog.report;
        if !r.free {
            return Err(not_gaussian(r));
        }
        let bytes = covariance_bytes(r.n);
        if bytes > max_bytes {
            return Err(SimError::TooLarge {
                what: "gaussian covariance matrix",
                bytes,
                limit: max_bytes,
            });
        }
        let mut st = GaussianState::vacuum(r.n);
        for op in &prog.ops {
            match op {
                GaussOp::Interaction(ip) => on_interaction(&st, ip),
                op => st.apply(op),
            }
        }
        st.mode_of_qubit = r.mode_of_qubit.clone();
        Ok(st)
    }

    /// Applies one Gaussian step (an interaction phase is ignored).
    pub fn apply(&mut self, op: &GaussOp) {
        match op {
            GaussOp::One { mode, q } => {
                let qf = [q[0][0], q[0][1], q[1][0], q[1][1]];
                self.apply_local::<2>(2 * mode, &qf);
            }
            GaussOp::Two { mode, q } => {
                let mut qf = [0.0; 16];
                for i in 0..4 {
                    for j in 0..4 {
                        qf[4 * i + j] = q[i][j];
                    }
                }
                self.apply_local::<4>(2 * mode, &qf);
            }
            GaussOp::Interaction(_) => {}
        }
    }

    /// `M → Q M Qᵀ` on the `K` indices starting at `s`, then the parity
    /// sign on every later index.
    fn apply_local<const K: usize>(&mut self, s: usize, q: &[f64]) {
        let dim = 2 * self.n;
        let m = &mut self.m;
        // fold the lazy signs of the block's indices into m
        for i in s..s + K {
            if self.d[i] < 0.0 {
                for c in 0..dim {
                    m[i * dim + c] = -m[i * dim + c];
                    m[c * dim + i] = -m[c * dim + i];
                }
                self.d[i] = 1.0;
            }
        }
        // rows: m[s+i][c] = Σ_j q_ij m[s+j][c]
        let mut v = [0.0f64; K];
        for c in 0..dim {
            for (j, x) in v.iter_mut().enumerate() {
                *x = m[(s + j) * dim + c];
            }
            for i in 0..K {
                let mut acc = 0.0;
                for j in 0..K {
                    acc += q[i * K + j] * v[j];
                }
                m[(s + i) * dim + c] = acc;
            }
        }
        // columns: m[r][s+i] = Σ_j m[r][s+j] q_ij
        for r in 0..dim {
            let row = &mut m[r * dim + s..r * dim + s + K];
            v.copy_from_slice(row);
            for i in 0..K {
                let mut acc = 0.0;
                for j in 0..K {
                    acc += q[i * K + j] * v[j];
                }
                row[i] = acc;
            }
        }
        if det::<K>(q) < 0.0 {
            for x in &mut self.d[s + K..] {
                *x = -*x;
            }
        }
    }

    /// `M_ab` with the lazy signs applied.
    #[inline]
    fn get(&self, a: usize, b: usize) -> f64 {
        self.m[a * 2 * self.n + b] * self.d[a] * self.d[b]
    }

    /// The covariance matrix `M_ab = -i⟨γ_a γ_b⟩` (`2n x 2n`, row-major,
    /// Jordan–Wigner mode order).
    pub fn covariance(&self) -> Vec<f64> {
        let dim = 2 * self.n;
        let mut out = vec![0.0; dim * dim];
        for a in 0..dim {
            for b in 0..dim {
                out[a * dim + b] = self.get(a, b);
            }
        }
        out
    }

    fn check(&self, q: usize) -> Result<usize, SimError> {
        self.mode_of_qubit
            .get(q)
            .copied()
            .ok_or(SimError::QubitOutOfRange {
                qubit: q,
                num_qubits: self.n,
            })
    }

    /// `⟨Z_q⟩` of qubit `q` at the end of the circuit.
    pub fn expectation_z(&self, q: usize) -> Result<f64, SimError> {
        let k = self.check(q)?;
        Ok(self.get(2 * k, 2 * k + 1))
    }

    /// `⟨Z_i Z_j⟩` by Wick's theorem (a 4x4 Pfaffian).
    pub fn z_correlation(&self, i: usize, j: usize) -> Result<f64, SimError> {
        self.expectation_z_product(&[i, j])
    }

    /// `⟨Π_{q ∈ qs} Z_q⟩` (a qubit listed twice cancels) by Wick's theorem:
    /// the Pfaffian of the covariance restricted to the modes' Majorana
    /// operators, O(|qs|^3).
    pub fn expectation_z_product(&self, qs: &[usize]) -> Result<f64, SimError> {
        let mut modes: Vec<usize> = Vec::with_capacity(qs.len());
        for &q in qs {
            let k = self.check(q)?;
            if let Some(p) = modes.iter().position(|&x| x == k) {
                modes.swap_remove(p);
            } else {
                modes.push(k);
            }
        }
        modes.sort_unstable();
        let idx: Vec<usize> = modes.iter().flat_map(|&k| [2 * k, 2 * k + 1]).collect();
        let dim = idx.len();
        let mut a = vec![0.0; dim * dim];
        for (r, &x) in idx.iter().enumerate() {
            for (c, &y) in idx.iter().enumerate() {
                a[r * dim + c] = self.get(x, y);
            }
        }
        Ok(pfaffian(&mut a, dim))
    }

    /// Probability that the qubits `qs` read `bits` (marginal over the
    /// others): `±Pf((M_S + B)/2)` with `B` the covariance of the basis
    /// state (sign: the parity of `bits`), O(|qs|^3).
    pub fn marginal_probability(&self, qs: &[usize], bits: &[bool]) -> Result<f64, SimError> {
        if qs.len() != bits.len() {
            return Err(SimError::NotSupported {
                what: "gaussian: one bit per qubit is needed",
            });
        }
        let mut pairs: Vec<(usize, bool)> = Vec::with_capacity(qs.len());
        for (&q, &b) in qs.iter().zip(bits) {
            let k = self.check(q)?;
            if let Some(&(_, b0)) = pairs.iter().find(|x| x.0 == k) {
                if b0 != b {
                    return Ok(0.0);
                }
                continue;
            }
            pairs.push((k, b));
        }
        pairs.sort_unstable();
        // Π_k (1 + s_k Z_k)/2 expands to Σ_S Π_{k∈S} s_k Pf(M_S) / 2^m, and
        // Pf(M + B) = Π_k s_k · Σ_S Π_{k∈S} s_k Pf(M_S) for B = ⊕ s_k J
        let sign: f64 = if pairs.iter().filter(|x| x.1).count() % 2 == 1 {
            -1.0
        } else {
            1.0
        };
        let dim = 2 * pairs.len();
        let mut a = vec![0.0; dim * dim];
        for (r, &(k, b)) in pairs.iter().enumerate() {
            for (c, &(l, _)) in pairs.iter().enumerate() {
                for (u, v) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    a[(2 * r + u) * dim + 2 * c + v] = 0.5 * self.get(2 * k + u, 2 * l + v);
                }
            }
            let s = if b { -0.5 } else { 0.5 };
            a[(2 * r) * dim + 2 * r + 1] += s;
            a[(2 * r + 1) * dim + 2 * r] -= s;
        }
        Ok((sign * pfaffian(&mut a, dim)).max(0.0))
    }

    /// Probability of the basis state `x` (bit `q` = qubit `q`, `n ≤ 128`).
    pub fn probability(&self, x: u128) -> Result<f64, SimError> {
        if self.n > 128 {
            return Err(SimError::NotSupported {
                what: "gaussian: basis-state indices need n <= 128",
            });
        }
        let qs: Vec<usize> = (0..self.n).collect();
        let bits: Vec<bool> = (0..self.n).map(|q| (x >> q) & 1 == 1).collect();
        self.marginal_probability(&qs, &bits)
    }

    /// `shots` samples of every qubit (bit `q` = qubit `q`, `n ≤ 128`):
    /// the modes are measured one after the other, each projection a rank-2
    /// update of the remaining covariance (O(n^3) per shot).
    pub fn sample<R: Rng + ?Sized>(
        &self,
        shots: usize,
        rng: &mut R,
    ) -> Result<Vec<u128>, SimError> {
        if self.n > 128 {
            return Err(SimError::NotSupported {
                what: "gaussian: samples need n <= 128",
            });
        }
        let n = self.n;
        let dim = 2 * n;
        let cov = self.covariance();
        let mut qubit_of_mode = vec![0usize; n];
        for (q, &k) in self.mode_of_qubit.iter().enumerate() {
            qubit_of_mode[k] = q;
        }
        let mut out = Vec::with_capacity(shots);
        let mut m = cov.clone();
        for _ in 0..shots {
            m.copy_from_slice(&cov);
            let mut x = 0u128;
            for (k, &qk) in qubit_of_mode.iter().enumerate() {
                let (a, b) = (2 * k, 2 * k + 1);
                let mab = m[a * dim + b].clamp(-1.0, 1.0);
                let p0 = 0.5 * (1.0 + mab);
                let one = rng.random::<f64>() >= p0;
                let s = if one { -1.0 } else { 1.0 };
                if one {
                    x |= 1u128 << qk;
                }
                let den = 1.0 + s * mab;
                if den <= 1e-300 {
                    continue;
                }
                let f = s / den;
                for c in b + 1..dim {
                    let (mca, mcb) = (m[c * dim + a], m[c * dim + b]);
                    for e in c + 1..dim {
                        let (mea, meb) = (m[e * dim + a], m[e * dim + b]);
                        let v = m[c * dim + e] + f * (mcb * mea - mca * meb);
                        m[c * dim + e] = v;
                        m[e * dim + c] = -v;
                    }
                }
            }
            out.push(x);
        }
        Ok(out)
    }
}

fn det<const K: usize>(q: &[f64]) -> f64 {
    match K {
        2 => q[0] * q[3] - q[1] * q[2],
        _ => {
            // Laplace-free: LU with partial pivoting on a copy
            let mut a = [0.0f64; 16];
            a[..16].copy_from_slice(&q[..16]);
            let mut d = 1.0;
            for k in 0..4 {
                let p = (k..4)
                    .max_by(|&i, &j| a[i * 4 + k].abs().total_cmp(&a[j * 4 + k].abs()))
                    .unwrap_or(k);
                if a[p * 4 + k] == 0.0 {
                    return 0.0;
                }
                if p != k {
                    for c in 0..4 {
                        a.swap(p * 4 + c, k * 4 + c);
                    }
                    d = -d;
                }
                d *= a[k * 4 + k];
                for i in k + 1..4 {
                    let f = a[i * 4 + k] / a[k * 4 + k];
                    for c in k..4 {
                        a[i * 4 + c] -= f * a[k * 4 + c];
                    }
                }
            }
            d
        }
    }
}

/// Pfaffian of a real antisymmetric `dim x dim` matrix (row-major,
/// overwritten): skew-symmetric Gaussian elimination with pivoting,
/// O(dim^3). Returns 0 for odd `dim`.
pub fn pfaffian(a: &mut [f64], dim: usize) -> f64 {
    if dim % 2 == 1 {
        return 0.0;
    }
    let mut pf = 1.0;
    let mut k = 0;
    while k < dim {
        // pivot: largest |a[k][j]|, j > k
        let mut piv = k + 1;
        let mut best = a[k * dim + k + 1].abs();
        for j in k + 2..dim {
            let v = a[k * dim + j].abs();
            if v > best {
                best = v;
                piv = j;
            }
        }
        if best == 0.0 {
            return 0.0;
        }
        if piv != k + 1 {
            let p = piv;
            let q = k + 1;
            for c in 0..dim {
                a.swap(p * dim + c, q * dim + c);
            }
            for r in 0..dim {
                a.swap(r * dim + p, r * dim + q);
            }
            pf = -pf;
        }
        let akk = a[k * dim + k + 1];
        pf *= akk;
        for i in k + 2..dim {
            let ti = a[k * dim + i] / akk;
            let rowi = a[(k + 1) * dim + i];
            for j in k + 2..dim {
                let tj = a[k * dim + j] / akk;
                a[i * dim + j] += -ti * a[(k + 1) * dim + j] + tj * rowi;
            }
        }
        k += 2;
    }
    pf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pfaffian_small() {
        // [[0, a], [-a, 0]] -> a
        let mut m = vec![0.0, 3.0, -3.0, 0.0];
        assert_eq!(pfaffian(&mut m, 2), 3.0);
        // 4x4: Pf = a01 a23 - a02 a13 + a03 a12
        let (a01, a02, a03, a12, a13, a23) = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let mut m = vec![
            0.0, a01, a02, a03, //
            -a01, 0.0, a12, a13, //
            -a02, -a12, 0.0, a23, //
            -a03, -a13, -a23, 0.0,
        ];
        let want = a01 * a23 - a02 * a13 + a03 * a12;
        assert!((pfaffian(&mut m, 4) - want).abs() < 1e-12);
    }

    #[test]
    fn pfaffian_squared_is_determinant() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let dim = 8;
        let mut a = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in i + 1..dim {
                let v: f64 = rng.random_range(-1.0..1.0);
                a[i * dim + j] = v;
                a[j * dim + i] = -v;
            }
        }
        // determinant by elimination
        let mut b = a.clone();
        let mut det = 1.0;
        for k in 0..dim {
            let p = (k..dim)
                .max_by(|&i, &j| b[i * dim + k].abs().total_cmp(&b[j * dim + k].abs()))
                .unwrap();
            if p != k {
                for c in 0..dim {
                    b.swap(p * dim + c, k * dim + c);
                }
                det = -det;
            }
            det *= b[k * dim + k];
            for i in k + 1..dim {
                let f = b[i * dim + k] / b[k * dim + k];
                for c in k..dim {
                    b[i * dim + c] -= f * b[k * dim + c];
                }
            }
        }
        let pf = pfaffian(&mut a, dim);
        assert!((pf * pf - det).abs() < 1e-10 * det.abs().max(1.0));
    }
}
