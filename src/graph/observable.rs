//! Pauli-sum observables `Σ_t c_t P_t` and their exact expectation values
//! on a dense state.

use crate::circuit::SimError;
use num_complex::Complex64;
use rayon::prelude::*;

/// One Pauli string with a real coefficient. `x`/`z` are bit masks over the
/// qubits (`Y` sets both).
#[derive(Clone, Debug, PartialEq)]
pub struct PauliTerm {
    /// Real coefficient `c_t`.
    pub coef: f64,
    /// X mask: bit `q` set when the factor on qubit `q` is `X` or `Y`.
    pub x: u128,
    /// Z mask: bit `q` set when the factor on qubit `q` is `Z` or `Y`.
    pub z: u128,
}

impl PauliTerm {
    /// Qubits the term acts on.
    pub fn support(&self) -> u128 {
        self.x | self.z
    }

    /// Number of `Y` factors.
    pub fn num_y(&self) -> u32 {
        (self.x & self.z).count_ones()
    }

    /// Diagonal (only `Z`s and identities).
    pub fn is_diagonal(&self) -> bool {
        self.x == 0
    }
}

/// `Σ_t c_t P_t`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Observable {
    /// The terms of the sum, in insertion order.
    pub terms: Vec<PauliTerm>,
}

impl Observable {
    /// The zero observable (no terms).
    pub fn new() -> Self {
        Observable::default()
    }

    /// Adds `coef · P` for a sparse string like `"X0 Y3 Z5"` (empty = identity).
    pub fn add(&mut self, coef: f64, s: &str) -> Result<&mut Self, SimError> {
        let (mut x, mut z) = (0u128, 0u128);
        for tok in s.split_whitespace() {
            let (p, q) = tok.split_at(1);
            let q: usize = q.parse().map_err(|_| SimError::NotSupported {
                what: "Observable::add: bad Pauli token",
            })?;
            if q >= 128 {
                return Err(SimError::QubitOutOfRange {
                    qubit: q,
                    num_qubits: 128,
                });
            }
            let b = 1u128 << q;
            if (x | z) & b != 0 {
                return Err(SimError::NotSupported {
                    what: "Observable::add: repeated qubit",
                });
            }
            match p {
                "X" | "x" => x |= b,
                "Y" | "y" => {
                    x |= b;
                    z |= b
                }
                "Z" | "z" => z |= b,
                "I" | "i" => {}
                _ => {
                    return Err(SimError::NotSupported {
                        what: "Observable::add: bad Pauli letter",
                    })
                }
            }
        }
        self.terms.push(PauliTerm { coef, x, z });
        Ok(self)
    }

    /// Adds `coef · Z_a Z_b`.
    pub fn zz(&mut self, coef: f64, a: usize, b: usize) -> &mut Self {
        self.terms.push(PauliTerm {
            coef,
            x: 0,
            z: (1u128 << a) | (1u128 << b),
        });
        self
    }

    /// Union of the supports.
    pub fn support(&self) -> u128 {
        self.terms.iter().fold(0, |s, t| s | t.support())
    }

    /// Only `Z`s.
    pub fn is_diagonal(&self) -> bool {
        self.terms.iter().all(|t| t.is_diagonal())
    }

    /// Restriction of the term to the given qubits, relabelled to local
    /// indices (`local[q]` = position, `None` = not in the part).
    pub(crate) fn restrict(t: &PauliTerm, local: &[Option<usize>]) -> (u64, u64) {
        let (mut x, mut z) = (0u64, 0u64);
        for (q, l) in local.iter().enumerate() {
            if let Some(l) = l {
                if t.x >> q & 1 == 1 {
                    x |= 1 << l;
                }
                if t.z >> q & 1 == 1 {
                    z |= 1 << l;
                }
            }
        }
        (x, z)
    }
}

#[inline]
fn parity(x: u64) -> f64 {
    if x.count_ones() & 1 == 0 {
        1.0
    } else {
        -1.0
    }
}

/// `<ψ|P|ψ>` for local masks `(x, z)` on a normalised dense state.
///
/// `P|y> = i^{#Y} (-1)^{z·y} |y ⊕ x>`, so `<ψ|P|ψ> = i^{#Y} Σ_y conj(ψ[y⊕x])
/// (-1)^{z·y} ψ[y]`.
pub fn pauli_expectation(amps: &[Complex64], x: u64, z: u64) -> f64 {
    let ny = (x & z).count_ones();
    if x == 0 {
        return amps
            .par_iter()
            .with_min_len(1 << 12)
            .enumerate()
            .map(|(y, a)| parity(y as u64 & z) * a.norm_sqr())
            .sum();
    }
    let s: Complex64 = amps
        .par_iter()
        .with_min_len(1 << 12)
        .enumerate()
        .map(|(y, a)| amps[y ^ x as usize].conj() * a * parity(y as u64 & z))
        .sum();
    let ph = match ny % 4 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, -1.0),
    };
    (ph * s).re
}

/// Several diagonal terms in one pass: `Σ_t c_t <Z_{z_t}>`.
pub fn diagonal_expectation(amps: &[Complex64], terms: &[(f64, u64)]) -> f64 {
    amps.par_iter()
        .with_min_len(1 << 10)
        .enumerate()
        .map(|(y, a)| {
            let p = a.norm_sqr();
            if p == 0.0 {
                return 0.0;
            }
            let mut s = 0.0;
            for &(c, z) in terms {
                s += c * parity(y as u64 & z);
            }
            s * p
        })
        .sum()
}
