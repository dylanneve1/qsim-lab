//! Gate definitions shared by every backend.
//!
//! Conventions used throughout the crate:
//!
//! * Qubit `q` is bit `q` of a computational-basis index, so qubit 0 is the
//!   least significant bit. The basis state `|q2 q1 q0>` has index
//!   `4*q2 + 2*q1 + q0`.
//! * A two-qubit gate acting on `(a, b)` is described by a 4x4 matrix whose
//!   row/column index is `2*bit(a) + bit(b)`, i.e. the first qubit argument
//!   is the more significant one *in the matrix*, as in most textbooks. For
//!   `Cnot(c, t)` this gives the familiar matrix that swaps `|10>` and `|11>`.

use num_complex::Complex64;
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_4};

/// A 2x2 complex matrix, row-major.
pub type Mat2 = [[Complex64; 2]; 2];
/// A 4x4 complex matrix, row-major, indexed by `2*bit(a) + bit(b)`.
pub type Mat4 = [[Complex64; 4]; 4];

/// A quantum gate together with the qubits it acts on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Gate {
    I(usize),
    H(usize),
    X(usize),
    Y(usize),
    Z(usize),
    S(usize),
    Sdg(usize),
    T(usize),
    Tdg(usize),
    Sx(usize),
    Sxdg(usize),
    /// `exp(-i θ X / 2)`
    Rx(usize, f64),
    /// `exp(-i θ Y / 2)`
    Ry(usize, f64),
    /// `exp(-i θ Z / 2)`
    Rz(usize, f64),
    /// `diag(1, e^{iθ})`
    Phase(usize, f64),
    /// `U(θ, φ, λ)` universal single-qubit gate.
    U(usize, f64, f64, f64),
    /// `Cnot(control, target)`
    Cnot(usize, usize),
    Cz(usize, usize),
    Swap(usize, usize),
    ISwap(usize, usize),
    ISwapdg(usize, usize),
    /// Controlled phase `diag(1, 1, 1, e^{iθ})`.
    CPhase(usize, usize, f64),
    /// Toffoli: `Ccx(control1, control2, target)`.
    Ccx(usize, usize, usize),
}

const fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

const ZERO: Complex64 = c(0.0, 0.0);
const ONE: Complex64 = c(1.0, 0.0);

impl Gate {
    /// The qubits the gate acts on, in argument order.
    pub fn qubits(&self) -> Vec<usize> {
        use Gate::*;
        match *self {
            I(q) | H(q) | X(q) | Y(q) | Z(q) | S(q) | Sdg(q) | T(q) | Tdg(q) | Sx(q)
            | Sxdg(q) => vec![q],
            Rx(q, _) | Ry(q, _) | Rz(q, _) | Phase(q, _) | U(q, _, _, _) => vec![q],
            Cnot(a, b) | Cz(a, b) | Swap(a, b) | ISwap(a, b) | ISwapdg(a, b)
            | CPhase(a, b, _) => vec![a, b],
            Ccx(a, b, t) => vec![a, b, t],
        }
    }

    /// Number of qubits the gate acts on.
    pub fn arity(&self) -> usize {
        self.qubits().len()
    }

    /// True for gates in the Clifford group (simulable with a tableau).
    pub fn is_clifford(&self) -> bool {
        use Gate::*;
        matches!(
            self,
            I(_) | H(_)
                | X(_)
                | Y(_)
                | Z(_)
                | S(_)
                | Sdg(_)
                | Sx(_)
                | Sxdg(_)
                | Cnot(..)
                | Cz(..)
                | Swap(..)
                | ISwap(..)
                | ISwapdg(..)
        )
    }

    /// True for the T and T† gates.
    pub fn is_t(&self) -> bool {
        matches!(self, Gate::T(_) | Gate::Tdg(_))
    }

    /// The inverse gate.
    pub fn inverse(&self) -> Gate {
        use Gate::*;
        match *self {
            S(q) => Sdg(q),
            Sdg(q) => S(q),
            T(q) => Tdg(q),
            Tdg(q) => T(q),
            Sx(q) => Sxdg(q),
            Sxdg(q) => Sx(q),
            Rx(q, t) => Rx(q, -t),
            Ry(q, t) => Ry(q, -t),
            Rz(q, t) => Rz(q, -t),
            Phase(q, t) => Phase(q, -t),
            U(q, th, ph, lam) => U(q, -th, -lam, -ph),
            CPhase(a, b, t) => CPhase(a, b, -t),
            ISwap(a, b) => ISwapdg(a, b),
            ISwapdg(a, b) => ISwap(a, b),
            g => g, // I, H, X, Y, Z, CNOT, CZ, SWAP, CCX are self-inverse
        }
    }

    /// If the gate is diagonal and acts on one qubit, returns `(d0, d1)`.
    pub fn diagonal_1q(&self) -> Option<(Complex64, Complex64)> {
        use Gate::*;
        Some(match *self {
            I(_) => (ONE, ONE),
            Z(_) => (ONE, c(-1.0, 0.0)),
            S(_) => (ONE, c(0.0, 1.0)),
            Sdg(_) => (ONE, c(0.0, -1.0)),
            T(_) => (ONE, Complex64::from_polar(1.0, FRAC_PI_4)),
            Tdg(_) => (ONE, Complex64::from_polar(1.0, -FRAC_PI_4)),
            Phase(_, t) => (ONE, Complex64::from_polar(1.0, t)),
            Rz(_, t) => (
                Complex64::from_polar(1.0, -t / 2.0),
                Complex64::from_polar(1.0, t / 2.0),
            ),
            _ => return None,
        })
    }

    /// The 2x2 unitary of a single-qubit gate (`None` for multi-qubit gates).
    pub fn matrix_1q(&self) -> Option<Mat2> {
        use Gate::*;
        if let Some((d0, d1)) = self.diagonal_1q() {
            return Some([[d0, ZERO], [ZERO, d1]]);
        }
        let h = FRAC_1_SQRT_2;
        Some(match *self {
            H(_) => [[c(h, 0.0), c(h, 0.0)], [c(h, 0.0), c(-h, 0.0)]],
            X(_) => [[ZERO, ONE], [ONE, ZERO]],
            Y(_) => [[ZERO, c(0.0, -1.0)], [c(0.0, 1.0), ZERO]],
            Sx(_) => [
                [c(0.5, 0.5), c(0.5, -0.5)],
                [c(0.5, -0.5), c(0.5, 0.5)],
            ],
            Sxdg(_) => [
                [c(0.5, -0.5), c(0.5, 0.5)],
                [c(0.5, 0.5), c(0.5, -0.5)],
            ],
            Rx(_, t) => {
                let (s, co) = (t / 2.0).sin_cos();
                [[c(co, 0.0), c(0.0, -s)], [c(0.0, -s), c(co, 0.0)]]
            }
            Ry(_, t) => {
                let (s, co) = (t / 2.0).sin_cos();
                [[c(co, 0.0), c(-s, 0.0)], [c(s, 0.0), c(co, 0.0)]]
            }
            U(_, th, ph, lam) => {
                let (s, co) = (th / 2.0).sin_cos();
                let m00 = c(co, 0.0);
                let m01 = -Complex64::from_polar(s, lam);
                let m10 = Complex64::from_polar(s, ph);
                let m11 = Complex64::from_polar(co, ph + lam);
                [[m00, m01], [m10, m11]]
            }
            _ => return None,
        })
    }

    /// The 4x4 unitary of a two-qubit gate, indexed by `2*bit(a) + bit(b)`.
    pub fn matrix_2q(&self) -> Option<Mat4> {
        use Gate::*;
        let mut m = [[ZERO; 4]; 4];
        match *self {
            Cnot(..) => {
                m[0][0] = ONE;
                m[1][1] = ONE;
                m[2][3] = ONE;
                m[3][2] = ONE;
            }
            Cz(..) => {
                m[0][0] = ONE;
                m[1][1] = ONE;
                m[2][2] = ONE;
                m[3][3] = c(-1.0, 0.0);
            }
            Swap(..) => {
                m[0][0] = ONE;
                m[1][2] = ONE;
                m[2][1] = ONE;
                m[3][3] = ONE;
            }
            ISwap(..) => {
                m[0][0] = ONE;
                m[1][2] = c(0.0, 1.0);
                m[2][1] = c(0.0, 1.0);
                m[3][3] = ONE;
            }
            ISwapdg(..) => {
                m[0][0] = ONE;
                m[1][2] = c(0.0, -1.0);
                m[2][1] = c(0.0, -1.0);
                m[3][3] = ONE;
            }
            CPhase(_, _, t) => {
                m[0][0] = ONE;
                m[1][1] = ONE;
                m[2][2] = ONE;
                m[3][3] = Complex64::from_polar(1.0, t);
            }
            _ => return None,
        }
        Some(m)
    }

    /// Rewrites the gate into an equivalent sequence (equal up to global
    /// phase) using only Clifford gates and `Rz`/`Phase`/`T` rotations.
    ///
    /// Backends that only understand Pauli rotations about Z (the Pauli-path
    /// simulator) or only one- and two-qubit gates (MPS) use this.
    pub fn decompose_to_clifford_rz(&self) -> Vec<Gate> {
        use Gate::*;
        match *self {
            I(_) => vec![],
            Sx(q) => vec![H(q), S(q), H(q)],
            Sxdg(q) => vec![H(q), Sdg(q), H(q)],
            ISwap(a, b) => vec![Swap(a, b), Cz(a, b), S(a), S(b)],
            ISwapdg(a, b) => vec![Swap(a, b), Cz(a, b), Sdg(a), Sdg(b)],
            U(q, th, ph, lam) => vec![
                Rz(q, lam),
                Sdg(q),
                H(q),
                Rz(q, th),
                H(q),
                S(q),
                Rz(q, ph),
            ],
            Rx(q, t) => vec![H(q), Rz(q, t), H(q)],
            // Ry(θ) = S Rx(θ) S†, applied in time order S†, Rx, S.
            Ry(q, t) => vec![Sdg(q), H(q), Rz(q, t), H(q), S(q)],
            CPhase(a, b, t) => vec![
                Phase(a, t / 2.0),
                Phase(b, t / 2.0),
                Cnot(a, b),
                Phase(b, -t / 2.0),
                Cnot(a, b),
            ],
            Ccx(a, b, t) => toffoli_clifford_t(a, b, t),
            g => vec![g],
        }
    }
}

/// The standard 7-T-gate decomposition of the Toffoli gate
/// (Nielsen & Chuang, Fig. 4.9).
pub fn toffoli_clifford_t(a: usize, b: usize, t: usize) -> Vec<Gate> {
    use Gate::*;
    vec![
        H(t),
        Cnot(b, t),
        Tdg(t),
        Cnot(a, t),
        T(t),
        Cnot(b, t),
        Tdg(t),
        Cnot(a, t),
        T(b),
        T(t),
        H(t),
        Cnot(a, b),
        T(a),
        Tdg(b),
        Cnot(a, b),
    ]
}

/// Angle helper: `true` if `t` is an integer multiple of π/2 (to 1e-12).
pub fn is_multiple_of_half_pi(t: f64) -> bool {
    let k = t / FRAC_PI_2;
    (k - k.round()).abs() < 1e-12
}

/// Multiplies two 2x2 matrices.
pub fn mat2_mul(a: &Mat2, b: &Mat2) -> Mat2 {
    let mut m = [[ZERO; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            m[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
        }
    }
    m
}

/// Conjugates a 4x4 two-qubit matrix by SWAP, i.e. exchanges which qubit is
/// treated as the more significant one.
pub fn mat4_swap_qubits(m: &Mat4) -> Mat4 {
    let p = |i: usize| ((i & 1) << 1) | (i >> 1);
    let mut out = [[ZERO; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[p(i)][p(j)] = m[i][j];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq2(a: &Mat2, b: &Mat2) -> bool {
        (0..2).all(|i| (0..2).all(|j| (a[i][j] - b[i][j]).norm() < 1e-12))
    }

    fn mat(g: Gate) -> Mat2 {
        g.matrix_1q().unwrap()
    }

    #[test]
    fn single_qubit_identities() {
        let id = [[ONE, ZERO], [ZERO, ONE]];
        assert!(approx_eq2(
            &mat2_mul(&mat(Gate::H(0)), &mat(Gate::H(0))),
            &id
        ));
        assert!(approx_eq2(
            &mat2_mul(&mat(Gate::S(0)), &mat(Gate::S(0))),
            &mat(Gate::Z(0))
        ));
        assert!(approx_eq2(
            &mat2_mul(&mat(Gate::T(0)), &mat(Gate::T(0))),
            &mat(Gate::S(0))
        ));
        assert!(approx_eq2(
            &mat2_mul(&mat(Gate::T(0)), &mat(Gate::Tdg(0))),
            &id
        ));
        // HZH = X
        let hzh = mat2_mul(
            &mat(Gate::H(0)),
            &mat2_mul(&mat(Gate::Z(0)), &mat(Gate::H(0))),
        );
        assert!(approx_eq2(&hzh, &mat(Gate::X(0))));
    }

    #[test]
    fn inverse_is_inverse() {
        let gates = [
            Gate::H(0),
            Gate::S(0),
            Gate::T(0),
            Gate::Rx(0, 0.3),
            Gate::Ry(0, -1.1),
            Gate::Rz(0, 2.0),
            Gate::Phase(0, 0.7),
            Gate::I(0),
            Gate::Sx(0),
            Gate::Sxdg(0),
            Gate::U(0, 1.2, -0.5, 0.8),
        ];
        let id = [[ONE, ZERO], [ZERO, ONE]];
        for g in gates {
            let p = mat2_mul(&mat(g), &mat(g.inverse()));
            assert!(approx_eq2(&p, &id), "{g:?}");
        }
    }

    #[test]
    fn sx_squared_is_x() {
        let sx = mat(Gate::Sx(0));
        let sx2 = mat2_mul(&sx, &sx);
        // Sx^2 = X up to global phase, check equality
        assert!(approx_eq2(&sx2, &mat(Gate::X(0))));
    }

    #[test]
    fn iswap_matrix() {
        let m = Gate::ISwap(0, 1).matrix_2q().unwrap();
        assert_eq!(m[0][0], ONE);
        assert_eq!(m[1][2], c(0.0, 1.0));
        assert_eq!(m[2][1], c(0.0, 1.0));
        assert_eq!(m[3][3], ONE);

        let mdg = Gate::ISwapdg(0, 1).matrix_2q().unwrap();
        assert_eq!(mdg[0][0], ONE);
        assert_eq!(mdg[1][2], c(0.0, -1.0));
        assert_eq!(mdg[2][1], c(0.0, -1.0));
        assert_eq!(mdg[3][3], ONE);
    }

    #[test]
    fn swap_qubits_of_cnot_reverses_roles() {
        let m = Gate::Cnot(0, 1).matrix_2q().unwrap();
        let r = mat4_swap_qubits(&m);
        // Reversed CNOT swaps |01> and |11> (the control is now the low bit).
        assert_eq!(r[1][3], ONE);
        assert_eq!(r[3][1], ONE);
        assert_eq!(r[0][0], ONE);
        assert_eq!(r[2][2], ONE);
    }

    #[test]
    fn half_pi_detection() {
        assert!(is_multiple_of_half_pi(std::f64::consts::PI));
        assert!(is_multiple_of_half_pi(-FRAC_PI_2));
        assert!(!is_multiple_of_half_pi(FRAC_PI_4));
    }
}
