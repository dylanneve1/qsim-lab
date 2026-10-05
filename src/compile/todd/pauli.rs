//! Pauli-frame front end: a Clifford+T circuit as a sequence of `π/8`
//! Pauli rotations followed by a Clifford (`U = C · Π_j e^{-iπ/8 · k_j P_j}`
//! up to a global phase), with every rotation axis `P_j` written in the
//! input frame. Rotations whose axes commute can be reordered, merged
//! (equal axes) and grouped; a group of pairwise commuting axes is
//! diagonalised by a Clifford and becomes one phase polynomial, which is
//! where TODD applies. This generalises the slot model of the parent
//! module, which only groups terms between the circuit's own Hadamards.

use super::gf2::Bits;
use super::PGate;

/// A Hermitian Pauli string `(-1)^sign · i^{|x∧z|} X^x Z^z` (so `x = z = 1`
/// on a qubit means `Y`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Pauli {
    /// X part.
    pub x: Bits,
    /// Z part.
    pub z: Bits,
    /// Sign bit.
    pub sign: bool,
}

impl Pauli {
    /// `Z_q` on `n` qubits.
    pub fn z(n: usize, q: usize) -> Self {
        Pauli {
            x: Bits::zeros(n),
            z: Bits::unit(n, q),
            sign: false,
        }
    }

    /// `X_q` on `n` qubits.
    pub fn x(n: usize, q: usize) -> Self {
        Pauli {
            x: Bits::unit(n, q),
            z: Bits::zeros(n),
            sign: false,
        }
    }

    /// True if the two strings commute.
    pub fn commutes(&self, o: &Pauli) -> bool {
        !(self.x.dot(&o.z) ^ self.z.dot(&o.x))
    }

    /// True if the string has no X/Y factor.
    pub fn is_diagonal(&self) -> bool {
        self.x.is_zero()
    }

    /// The product `self · o` of two commuting Hermitian strings (again
    /// Hermitian).
    pub fn mul_commuting(&self, o: &Pauli) -> Pauli {
        debug_assert!(self.commutes(o));
        // X^a Z^b X^c Z^d = (-1)^{b·c} X^{a+c} Z^{b+d}; with the i^{|x∧z|}
        // convention the phase of the product is i^{e} with
        // e = |a∧b| + |c∧d| + 2(b·c) - |(a+c)∧(b+d)|  (mod 4), which is even
        // for commuting strings.
        let ab = self.x.and(&self.z).count_ones() as i64;
        let cd = o.x.and(&o.z).count_ones() as i64;
        let bc = self.z.and(&o.x).count_ones() as i64;
        let mut x = self.x.clone();
        x.xor_with(&o.x);
        let mut z = self.z.clone();
        z.xor_with(&o.z);
        let xz = x.and(&z).count_ones() as i64;
        let e = (ab + cd + 2 * bc - xz).rem_euclid(4);
        debug_assert!(e % 2 == 0, "commuting Hermitian strings multiply to a Hermitian string");
        Pauli {
            x,
            z,
            sign: self.sign ^ o.sign ^ (e == 2),
        }
    }
}

/// The images `C† Z_q C` and `C† X_q C` of the accumulated Clifford `C`
/// (the inverse tableau), updated gate by gate.
#[derive(Clone, Debug)]
pub struct Frame {
    /// `C† Z_q C`.
    pub zs: Vec<Pauli>,
    /// `C† X_q C`.
    pub xs: Vec<Pauli>,
}

impl Frame {
    /// The identity frame on `n` qubits.
    pub fn new(n: usize) -> Self {
        Frame {
            zs: (0..n).map(|q| Pauli::z(n, q)).collect(),
            xs: (0..n).map(|q| Pauli::x(n, q)).collect(),
        }
    }

    /// Appends a Clifford gate `G` (`C ← G·C`): the new images are
    /// `C† (G† P G) C`.
    pub fn apply(&mut self, g: &PGate) {
        match *g {
            PGate::H(q) => std::mem::swap(&mut self.zs[q], &mut self.xs[q]),
            PGate::X(q) => {
                // X† Z X = -Z
                self.zs[q].sign = !self.zs[q].sign;
            }
            PGate::Phase(q, k) => {
                debug_assert!(k % 2 == 0, "only Clifford phases move the frame");
                for _ in 0..(k / 2) % 4 {
                    // S† X S = -Y = -(i X Z) ; Y as Hermitian string: X Z with
                    // i-convention, so image(X) <- -(image(X)·image(Z)) in
                    // the Hermitian product sense with an extra sign.
                    let y = mul_anticommuting(&self.xs[q], &self.zs[q]);
                    let mut y = y;
                    y.sign = !y.sign;
                    self.xs[q] = y;
                }
            }
            PGate::Cnot(c, t) => {
                // CNOT† Z_t CNOT = Z_c Z_t ; CNOT† X_c CNOT = X_c X_t
                let zt = self.zs[c].mul_commuting(&self.zs[t]);
                let xc = self.xs[c].mul_commuting(&self.xs[t]);
                self.zs[t] = zt;
                self.xs[c] = xc;
            }
            PGate::Cz(a, b) => {
                // CZ† X_a CZ = X_a Z_b
                let xa = self.xs[a].mul_commuting(&self.zs[b]);
                let xb = self.xs[b].mul_commuting(&self.zs[a]);
                self.xs[a] = xa;
                self.xs[b] = xb;
            }
            PGate::Swap(a, b) => {
                self.zs.swap(a, b);
                self.xs.swap(a, b);
            }
            PGate::Ccz(..) => panic!("CCZ is not Clifford"),
        }
    }

    /// The input-frame axis of `Z` on the parity of `wires` at the
    /// current point (`C† Z_{w1} ⋯ Z_{wk} C`).
    pub fn z_axis(&self, wires: &[usize]) -> Pauli {
        let mut p = self.zs[wires[0]].clone();
        for &w in &wires[1..] {
            p = p.mul_commuting(&self.zs[w]);
        }
        p
    }
}

/// `i · a · b` for two anticommuting Hermitian strings, which is Hermitian
/// (used for `Y = i X Z`).
fn mul_anticommuting(a: &Pauli, b: &Pauli) -> Pauli {
    debug_assert!(!a.commutes(b));
    let ab = a.x.and(&a.z).count_ones() as i64;
    let cd = b.x.and(&b.z).count_ones() as i64;
    let bc = a.z.and(&b.x).count_ones() as i64;
    let mut x = a.x.clone();
    x.xor_with(&b.x);
    let mut z = a.z.clone();
    z.xor_with(&b.z);
    let xz = x.and(&z).count_ones() as i64;
    // a·b = i^{ab + cd + 2bc - xz} X^x Z^z (times signs); times i:
    let e = (ab + cd + 2 * bc - xz + 1).rem_euclid(4);
    debug_assert!(e % 2 == 0);
    Pauli {
        x,
        z,
        sign: a.sign ^ b.sign ^ (e == 2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_images_match_textbook_conjugations() {
        let n = 2;
        let mut f = Frame::new(n);
        f.apply(&PGate::H(0));
        assert_eq!(f.zs[0], Pauli::x(n, 0));
        let mut f = Frame::new(n);
        f.apply(&PGate::Cnot(0, 1));
        let z0z1 = Pauli::z(n, 0).mul_commuting(&Pauli::z(n, 1));
        assert_eq!(f.zs[1], z0z1);
        // S then S = Z: S†S† X S S = Z X Z = -X
        let mut f = Frame::new(1);
        f.apply(&PGate::Phase(0, 4));
        let mut minus_x = Pauli::x(1, 0);
        minus_x.sign = true;
        assert_eq!(f.xs[0], minus_x);
        // S† X S = -Y
        let mut f = Frame::new(1);
        f.apply(&PGate::Phase(0, 2));
        assert_eq!(f.xs[0].x, Bits::unit(1, 0));
        assert_eq!(f.xs[0].z, Bits::unit(1, 0));
        assert!(f.xs[0].sign);
    }
}
