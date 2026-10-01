//! Exact hybrid Schrödinger--Feynman circuit cutting.
//!
//! The state is represented as a sum of products of two smaller state
//! vectors.  A gate crossing the cut is expanded into four operator products
//! (the generic operator-basis expansion; common gates usually have rank 2).

use crate::circuit::{check_gate, Circuit, SimError};
use crate::gate::{Gate, Mat2};
use crate::statevector::StateVectorF64;
use num_complex::Complex64;

#[derive(Clone)]
struct Path {
    a: StateVectorF64,
    b: StateVectorF64,
    coeff: Complex64,
}

/// Exact two-block Schrödinger--Feynman simulator.
#[derive(Clone, Debug)]
pub struct HybridSchrodingerFeynman {
    n: usize,
    cut: Vec<bool>,
}

impl HybridSchrodingerFeynman {
    /// Constructs a simulator with the first `left` qubits in block A.
    pub fn new(n: usize, left: usize) -> Self {
        assert!(left <= n);
        let mut cut = vec![false; n];
        for x in cut.iter_mut().take(left) {
            *x = true;
        }
        Self { n, cut }
    }

    /// Chooses a balanced partition using a deterministic local-search pass.
    pub fn automatic(n: usize, circuit: &Circuit) -> Self {
        let left = n / 2;
        let mut s = Self::new(n, left);
        // Swapping one A/B pair until no single swap improves the cut.
        loop {
            let old = s.cut_count(circuit);
            let mut best = None;
            for a in 0..n {
                if s.cut[a] {
                    for b in 0..n {
                        if !s.cut[b] {
                            s.cut.swap(a, b);
                            let v = s.cut_count(circuit);
                            s.cut.swap(a, b);
                            if v < old {
                                best = Some((a, b, v));
                            }
                        }
                    }
                }
            }
            if let Some((a, b, _)) = best {
                s.cut.swap(a, b);
            } else {
                break;
            }
        }
        s
    }

    pub fn num_qubits(&self) -> usize {
        self.n
    }
    pub fn cut(&self) -> &[bool] {
        &self.cut
    }
    pub fn cut_count(&self, c: &Circuit) -> usize {
        c.gates()
            .filter(|g| {
                g.arity() == 2 && {
                    let q = g.qubits();
                    self.cut[q[0]] != self.cut[q[1]]
                }
            })
            .count()
    }

    fn local_index(&self, q: usize) -> usize {
        self.cut[..q].iter().filter(|&&x| x).count()
    }
    fn block_sizes(&self) -> (usize, usize) {
        let a = self.cut.iter().filter(|&&x| x).count();
        (a, self.n - a)
    }
    fn remap(&self, g: Gate) -> Gate {
        let q = |x| self.local_index(x);
        match g {
            Gate::H(x) => Gate::H(q(x)),
            Gate::X(x) => Gate::X(q(x)),
            Gate::Y(x) => Gate::Y(q(x)),
            Gate::Z(x) => Gate::Z(q(x)),
            Gate::S(x) => Gate::S(q(x)),
            Gate::Sdg(x) => Gate::Sdg(q(x)),
            Gate::T(x) => Gate::T(q(x)),
            Gate::Tdg(x) => Gate::Tdg(q(x)),
            Gate::Rx(x, t) => Gate::Rx(q(x), t),
            Gate::Ry(x, t) => Gate::Ry(q(x), t),
            Gate::Rz(x, t) => Gate::Rz(q(x), t),
            Gate::Phase(x, t) => Gate::Phase(q(x), t),
            Gate::Cnot(a, b) => Gate::Cnot(q(a), q(b)),
            Gate::Cz(a, b) => Gate::Cz(q(a), q(b)),
            Gate::Swap(a, b) => Gate::Swap(q(a), q(b)),
            Gate::CPhase(a, b, t) => Gate::CPhase(q(a), q(b), t),
            Gate::Ccx(a, b, t) => Gate::Ccx(q(a), q(b), q(t)),
        }
    }
    fn apply_local(path: &mut Path, g: Gate, in_a: bool) -> Result<(), SimError> {
        if in_a {
            path.a.apply_gate(&g)
        } else {
            path.b.apply_gate(&g)
        }
    }
    fn cross_terms(&self, g: Gate) -> Result<Vec<(Mat2, Mat2, Complex64)>, SimError> {
        let qs = g.qubits();
        let m = g.matrix_2q().ok_or(SimError::Unsupported {
            backend: "hsf",
            gate: g,
        })?;
        // U = sum_ij E_ij (on first argument) tensor B_ij. This is exact
        // for every two-qubit gate and has at most four paths per cut gate.
        let z = Complex64::new(0.0, 0.0);
        let mut out = Vec::new();
        for i in 0..2 {
            for j in 0..2 {
                let mut a = [[z; 2]; 2];
                let mut b = [[z; 2]; 2];
                a[i][j] = Complex64::new(1.0, 0.0);
                for ob in 0..2 {
                    for ib in 0..2 {
                        b[ob][ib] = m[i * 2 + ob][j * 2 + ib];
                    }
                }
                out.push((a, b, Complex64::new(1.0, 0.0)));
            }
        }
        let _ = qs;
        Ok(out)
    }

    fn paths(&self, c: &Circuit) -> Result<Vec<Path>, SimError> {
        let (na, nb) = self.block_sizes();
        let mut paths = vec![Path {
            a: StateVectorF64::new(na),
            b: StateVectorF64::new(nb),
            coeff: Complex64::new(1.0, 0.0),
        }];
        for op in &c.ops {
            let g = match op {
                crate::circuit::Op::Gate(g) => *g,
                crate::circuit::Op::Measure(_) => {
                    return Err(SimError::Unsupported {
                        backend: "hsf",
                        gate: Gate::H(0),
                    })
                }
            };
            check_gate(&g, self.n)?;
            let qs = g.qubits();
            if g.arity() == 1 {
                let in_a = self.cut[qs[0]];
                let lg = self.remap(g);
                for p in &mut paths {
                    Self::apply_local(p, lg, in_a)?;
                }
            } else if g.arity() == 2 && self.cut[qs[0]] == self.cut[qs[1]] {
                let lg = self.remap(g);
                let in_a = self.cut[qs[0]];
                for p in &mut paths {
                    Self::apply_local(p, lg, in_a)?;
                }
            } else {
                // cross-cut two-qubit operator expansion
                let terms = self.cross_terms(g)?;
                let mut next = Vec::with_capacity(paths.len() * 4);
                for p in paths {
                    for (ma, mb, k) in &terms {
                        let mut x = p.clone();
                        x.coeff = p.coeff * *k;
                        let (qa, qb) = (qs[0], qs[1]);
                        let (a_gate, b_gate) = if self.cut[qa] { (ma, mb) } else { (mb, ma) };
                        x.a.apply_1q_matrix(
                            self.local_index(if self.cut[qa] { qa } else { qb }),
                            a_gate,
                        );
                        x.b.apply_1q_matrix(
                            self.local_index(if self.cut[qa] { qb } else { qa }),
                            b_gate,
                        );
                        next.push(x);
                    }
                }
                paths = next;
            }
        }
        Ok(paths)
    }

    /// Computes one exact computational-basis amplitude without allocating 2^n output.
    pub fn amplitude(&self, c: &Circuit, index: usize) -> Result<Complex64, SimError> {
        if index >= (1usize << self.n) {
            return Err(SimError::QubitOutOfRange {
                qubit: self.n,
                num_qubits: self.n,
            });
        }
        let paths = self.paths(c);
        let mut out = Complex64::new(0.0, 0.0);
        let mut ia = 0;
        let mut ib = 0;
        let mut ba = 0;
        let mut bb = 0;
        for q in 0..self.n {
            if self.cut[q] {
                ia |= ((index >> q) & 1) << ba;
                ba += 1;
            } else {
                ib |= ((index >> q) & 1) << bb;
                bb += 1;
            }
        }
        for p in paths? {
            out += p.coeff * p.a.amplitude(ia) * p.b.amplitude(ib);
        }
        Ok(out)
    }

    /// Reconstructs the full output when it fits the normal 1 GiB cap.
    pub fn simulate(&self, c: &Circuit) -> Result<Vec<Complex64>, SimError> {
        let bytes = (1u128 << self.n) * 16;
        if self.n >= 63 || bytes > (1u128 << 30) {
            return Err(SimError::TooLarge {
                what: "HSF full output",
                bytes,
                limit: 1u128 << 30,
            });
        }
        (0..(1usize << self.n))
            .map(|i| self.amplitude(c, i))
            .collect()
    }
}
