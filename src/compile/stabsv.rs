//! A phase-exact stabilizer simulator for up to 64 qubits, used to replace
//! the Clifford prefix of a circuit by a single state-vector write.
//!
//! The state is described by `n` stabilizer generators `P = i^e X^x Z^z`
//! (bit masks `x`, `z`, phase exponent `e` mod 4) plus one *reference*
//! basis state `r` in the support together with its exact amplitude
//! `ψ(r)`. The generators fix the state up to a global phase; the reference
//! amplitude pins that phase down, so the conversion reproduces the state
//! vector exactly (not just up to phase).
//!
//! Since `P|y> = i^e (-1)^{z·y} |y xor x>` and `P|ψ> = |ψ>`, every
//! stabilizer gives `ψ(y xor x) = i^e (-1)^{z·y} ψ(y)`. This both tracks
//! the reference through Hadamards (which mix `ψ(r)` with `ψ(r xor e_j)`)
//! and enumerates the whole support `r + span{x_i}` with a Gray code in
//! `O(2^k)` steps for a support of size `2^k`. Every amplitude is the
//! reference amplitude times a power of `i`, so there is no rounding drift.

use crate::circuit::{Circuit, Op};
use crate::gate::Gate;
use crate::statevector::{Real, StateVector};
use num_complex::{Complex, Complex64};
use rayon::prelude::*;
use std::f64::consts::FRAC_1_SQRT_2;

/// Generator `i^e X^x Z^z`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Row {
    x: u64,
    z: u64,
    e: u8,
}

impl Row {
    /// The product `self * other`.
    #[inline]
    fn mul(self, o: Row) -> Row {
        // X^x1 Z^z1 X^x2 Z^z2 = (-1)^{|z1 & x2|} X^{x1^x2} Z^{z1^z2}
        let sign = 2 * ((self.z & o.x).count_ones() as u8 & 1);
        Row {
            x: self.x ^ o.x,
            z: self.z ^ o.z,
            e: (self.e + o.e + sign) & 3,
        }
    }

    /// The power of `i` relating `ψ(y xor x)` to `ψ(y)`.
    #[inline]
    fn ratio_exp(&self, y: u64) -> u8 {
        (self.e + 2 * ((self.z & y).count_ones() as u8 & 1)) & 3
    }
}

/// Multiplies by `i^p` exactly.
#[inline]
fn times_i_pow(a: Complex64, p: u8) -> Complex64 {
    match p & 3 {
        0 => a,
        1 => Complex64::new(-a.im, a.re),
        2 => -a,
        _ => Complex64::new(a.im, -a.re),
    }
}

/// A stabilizer state with an exact global phase.
#[derive(Clone, Debug)]
pub struct PhaseStabilizer {
    n: usize,
    rows: Vec<Row>,
    r: u64,
    amp: Complex64,
}

impl PhaseStabilizer {
    /// `|0...0>` on `n <= 64` qubits.
    pub fn new(n: usize) -> Self {
        assert!(n <= 64, "PhaseStabilizer supports at most 64 qubits");
        PhaseStabilizer {
            n,
            rows: (0..n)
                .map(|q| Row {
                    x: 0,
                    z: 1 << q,
                    e: 0,
                })
                .collect(),
            r: 0,
            amp: Complex64::new(1.0, 0.0),
        }
    }

    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// The exact amplitude `<y|ψ>`.
    pub fn amplitude(&self, y: u64) -> Complex64 {
        let ech = self.echelon();
        // Express y xor r as a combination of the X parts (pivot = highest bit).
        let mut d = y ^ self.r;
        let mut cur = self.r;
        let mut a = self.amp;
        for row in &ech {
            if row.x == 0 {
                break;
            }
            let p = 63 - row.x.leading_zeros();
            if (d >> p) & 1 == 1 {
                a = times_i_pow(a, row.ratio_exp(cur));
                cur ^= row.x;
                d ^= row.x;
            }
        }
        if d != 0 {
            Complex64::new(0.0, 0.0)
        } else {
            a
        }
    }

    /// Applies a Clifford gate. Panics on non-Clifford gates.
    pub fn apply(&mut self, g: &Gate) {
        use Gate::*;
        let bit = |m: u64, q: usize| ((m >> q) & 1) as u8;
        match *g {
            X(j) => {
                self.r ^= 1 << j;
                for w in &mut self.rows {
                    w.e = (w.e + 2 * bit(w.z, j)) & 3;
                }
            }
            Z(j) => {
                if bit(self.r, j) == 1 {
                    self.amp = -self.amp;
                }
                for w in &mut self.rows {
                    w.e = (w.e + 2 * bit(w.x, j)) & 3;
                }
            }
            Y(j) => {
                // Y|b> = i (-1)^b |b xor 1>
                self.amp = times_i_pow(self.amp, 1 + 2 * bit(self.r, j));
                self.r ^= 1 << j;
                for w in &mut self.rows {
                    w.e = (w.e + 2 * (bit(w.x, j) ^ bit(w.z, j))) & 3;
                }
            }
            S(j) | Sdg(j) => {
                let dag = matches!(g, Sdg(_));
                if bit(self.r, j) == 1 {
                    self.amp = times_i_pow(self.amp, if dag { 3 } else { 1 });
                }
                let k = if dag { 3 } else { 1 };
                for w in &mut self.rows {
                    let xj = bit(w.x, j);
                    w.e = (w.e + k * xj) & 3;
                    w.z ^= (xj as u64) << j;
                }
            }
            H(j) => self.hadamard(j),
            Cnot(c, t) => {
                if bit(self.r, c) == 1 {
                    self.r ^= 1 << t;
                }
                for w in &mut self.rows {
                    w.x ^= (bit(w.x, c) as u64) << t;
                    w.z ^= (bit(w.z, t) as u64) << c;
                }
            }
            Cz(a, b) => {
                if bit(self.r, a) & bit(self.r, b) == 1 {
                    self.amp = -self.amp;
                }
                for w in &mut self.rows {
                    let (xa, xb) = (bit(w.x, a), bit(w.x, b));
                    w.e = (w.e + 2 * (xa & xb)) & 3;
                    w.z ^= ((xa as u64) << b) | ((xb as u64) << a);
                }
            }
            Swap(a, b) => {
                let sw = |m: u64| {
                    if ((m >> a) ^ (m >> b)) & 1 == 1 {
                        m ^ ((1 << a) | (1 << b))
                    } else {
                        m
                    }
                };
                self.r = sw(self.r);
                for w in &mut self.rows {
                    w.x = sw(w.x);
                    w.z = sw(w.z);
                }
            }
            _ => panic!("PhaseStabilizer: {g:?} is not Clifford"),
        }
    }

    fn hadamard(&mut self, j: usize) {
        // ψ(r xor e_j) relative to ψ(r), from a stabilizer with X part e_j.
        let ech = self.echelon();
        let lambda = ech
            .iter()
            .find(|w| w.x == 1 << j)
            .map(|w| w.ratio_exp(self.r));
        let rj = (self.r >> j) & 1;
        let a = self.amp;
        let other = lambda.map_or(Complex64::new(0.0, 0.0), |p| times_i_pow(a, p));
        let (u0, u1) = if rj == 0 { (a, other) } else { (other, a) };
        let v0 = (u0 + u1) * FRAC_1_SQRT_2;
        let v1 = (u0 - u1) * FRAC_1_SQRT_2;
        if v0.norm_sqr() >= v1.norm_sqr() {
            self.r &= !(1 << j);
            self.amp = v0;
        } else {
            self.r |= 1 << j;
            self.amp = v1;
        }
        for w in &mut self.rows {
            let (xj, zj) = ((w.x >> j) & 1, (w.z >> j) & 1);
            w.e = (w.e + 2 * (xj & zj) as u8) & 3;
            w.x = (w.x & !(1 << j)) | (zj << j);
            w.z = (w.z & !(1 << j)) | (xj << j);
        }
    }

    /// Generators in echelon form on the X parts with the pivot of each row
    /// its highest set bit (rows with X parts first, by decreasing pivot,
    /// each pivot cleared from all other rows), then the Z-only rows.
    fn echelon(&self) -> Vec<Row> {
        let mut rows = self.rows.clone();
        let mut rank = 0;
        for col in (0..self.n).rev() {
            let m = 1u64 << col;
            let Some(p) = (rank..rows.len()).find(|&i| rows[i].x & m != 0) else {
                continue;
            };
            rows.swap(rank, p);
            let piv = rows[rank];
            for (i, w) in rows.iter_mut().enumerate() {
                if i != rank && w.x & m != 0 {
                    *w = w.mul(piv);
                }
            }
            rank += 1;
        }
        rows
    }

    /// The state as a dense vector, written in parallel.
    pub fn to_statevector<T: Real>(&self) -> StateVector<T> {
        let n = self.n;
        let len = 1usize << n;
        let ech = self.echelon();
        let k = ech.iter().take_while(|w| w.x != 0).count();
        let xrows = &ech[..k];
        // Split the index space into 2^m chunks by the top m bits. Rows whose
        // pivot (highest bit) is below n - m never change the top bits, so
        // each chunk is filled independently: pick the top rows that reach
        // it, then Gray-code over the low rows.
        let m = n.min(8);
        let low_bits = n - m;
        let top_rows: Vec<&Row> = xrows
            .iter()
            .filter(|w| 63 - w.x.leading_zeros() as usize >= low_bits)
            .collect();
        let low_rows: Vec<&Row> = xrows
            .iter()
            .filter(|w| (63 - w.x.leading_zeros() as usize) < low_bits)
            .collect();
        let chunk = 1usize << low_bits;
        let zero = Complex::new(T::zero(), T::zero());
        let cvt = |a: Complex64| Complex::new(T::from_f64(a.re), T::from_f64(a.im));
        let mut amps: Vec<Complex<T>> = (0..len).into_par_iter().map(|_| zero).collect();
        amps.par_chunks_mut(chunk).enumerate().for_each(|(t, out)| {
            let base = (t as u64) << low_bits;
            let mut d = (base ^ self.r) >> low_bits << low_bits;
            let mut y = self.r;
            let mut a = self.amp;
            for w in &top_rows {
                let p = 63 - w.x.leading_zeros();
                if (d >> p) & 1 == 1 {
                    a = times_i_pow(a, w.ratio_exp(y));
                    y ^= w.x;
                    d ^= w.x;
                }
            }
            if d >> low_bits != 0 {
                return; // no support point has these top bits
            }
            let mask = (chunk - 1) as u64;
            out[(y & mask) as usize] = cvt(a);
            for s in 1..1usize << low_rows.len() {
                let w = low_rows[s.trailing_zeros() as usize];
                a = times_i_pow(a, w.ratio_exp(y));
                y ^= w.x;
                out[(y & mask) as usize] = cvt(a);
            }
        });
        StateVector::from_amplitudes(amps)
    }
}

/// Runs the Clifford circuit `prefix` (no measurements) on `|0...0>` and
/// returns the exact state vector.
pub fn clifford_statevector<T: Real>(prefix: &Circuit) -> StateVector<T> {
    let mut s = PhaseStabilizer::new(prefix.num_qubits);
    for op in &prefix.ops {
        match op {
            Op::Gate(g) => s.apply(g),
            // Anything else (measurement, reset, noise, classical control)
            // is not a unitary gate: callers pass `clifford_prefix` output.
            other => panic!("clifford_statevector: prefix must be unitary, got {other:?}"),
        }
    }
    s.to_statevector()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statevector::StateVectorF64;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn matches_statevector_exactly() {
        let mut rng = StdRng::seed_from_u64(5);
        for trial in 0..200 {
            let n = 1 + trial % 11;
            let c = Circuit::random_clifford(n, 1 + trial % 9, &mut rng);
            let got: StateVectorF64 = clifford_statevector(&c);
            let mut want = StateVectorF64::new(n);
            want.apply_circuit(&c).unwrap();
            let mut st = PhaseStabilizer::new(n);
            for g in c.gates() {
                st.apply(g);
            }
            for i in 0..1 << n {
                let d = (got.amplitude(i) - want.amplitude(i)).norm();
                assert!(d < 1e-12, "trial {trial} n={n} i={i}: {d}");
                let d2 = (st.amplitude(i as u64) - want.amplitude(i)).norm();
                assert!(d2 < 1e-12, "amplitude(): trial {trial} i={i}: {d2}");
            }
        }
    }
}
