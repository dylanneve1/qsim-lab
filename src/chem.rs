//! Low-magic quantum chemistry on the frame engines (research/simulability/lowmagic-chem.md).
//!
//! * [`Fcidump`]: molecular integrals in the FCIDUMP text format (as written by
//!   PySCF's `tools.fcidump`), chemist's notation `(pq|rs)`, 8-fold symmetry.
//! * [`Program`]: a circuit of Pauli rotations `exp(-i θ/2 P)` plus a few
//!   Clifford gates (the Hartree–Fock `X` gates, `H` for phase estimation), with
//!   optional variational parameters (`θ = mult · param[k]`). Written by the
//!   Python driver (`research/data/lowmagic-chem/chem.py`) from OpenFermion's
//!   Jordan–Wigner / Bravyi–Kitaev transforms.
//! * [`jw_hamiltonian`]: the Jordan–Wigner qubit Hamiltonian (interleaved spin
//!   orbitals `2p + σ`) generated directly from the integrals, optionally
//!   restricted to fermionic monomials whose `x` part lies in a given GF(2)
//!   span. For a program whose Clifford part only flips/phases qubits
//!   (`X`/`Z`/`S`/`CZ`, e.g. a determinant reference), a Pauli with `x` outside
//!   the span of the rotation axes has zero expectation in the compressed state,
//!   so the energy of a 100-qubit state only needs the few monomials inside the
//!   span ([`Span::from_program`]).
//! * [`energy`] (compressed state), [`rotosolve`] (exact coordinate descent for
//!   excitation generators, whose energy is a trigonometric polynomial of degree
//!   2 in each angle).

use crate::circuit::{Circuit, SimError};
use crate::engines::adaptive::CompressedState;
use crate::engines::pauli_path::PauliSum;
use crate::gate::Gate;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// FCIDUMP

/// One- and two-electron integrals over `norb` spatial orbitals.
#[derive(Clone, Debug)]
pub struct Fcidump {
    /// Number of spatial orbitals (`NORB`).
    pub norb: usize,
    /// Number of electrons (`NELEC`).
    pub nelec: usize,
    /// Twice the spin projection, `2·S_z` (`MS2`; 0 if absent).
    pub ms2: i64,
    /// Constant (nuclear repulsion + frozen core).
    pub ecore: f64,
    /// `h[p * norb + q]`.
    pub h1: Vec<f64>,
    /// `(pq|rs)` at `((p * norb + q) * norb + r) * norb + s`, all 8 symmetric copies filled.
    pub eri: Vec<f64>,
}

impl Fcidump {
    /// Parses the FCIDUMP text format (1-based indices, `0 0 0 0` = core).
    pub fn parse(text: &str) -> Result<Fcidump, String> {
        let up = text.to_ascii_uppercase();
        let end = up
            .find("&END")
            .or_else(|| up.find("/\n"))
            .ok_or("FCIDUMP: no &END")?;
        let header = &up[..end];
        let key = |k: &str| -> Option<i64> {
            let i = header.find(&format!("{k}="))?;
            let rest = &header[i + k.len() + 1..];
            let num: String = rest
                .trim_start()
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '-')
                .collect();
            num.parse().ok()
        };
        let norb = key("NORB").ok_or("FCIDUMP: NORB")? as usize;
        let nelec = key("NELEC").ok_or("FCIDUMP: NELEC")? as usize;
        let ms2 = key("MS2").unwrap_or(0);
        let body_start = text[end..]
            .find('\n')
            .map(|i| end + i + 1)
            .unwrap_or(text.len());
        let n2 = norb * norb;
        let mut fd = Fcidump {
            norb,
            nelec,
            ms2,
            ecore: 0.0,
            h1: vec![0.0; n2],
            eri: vec![0.0; n2 * n2],
        };
        for line in text[body_start..].lines() {
            let t: Vec<&str> = line.split_whitespace().collect();
            if t.len() < 5 {
                continue;
            }
            let v: f64 = t[0]
                .replace(['D', 'd'], "E")
                .parse()
                .map_err(|e| format!("FCIDUMP value {:?}: {e}", t[0]))?;
            let idx: Vec<usize> = t[1..5]
                .iter()
                .map(|s| {
                    s.parse::<usize>()
                        .map_err(|e| format!("FCIDUMP index: {e}"))
                })
                .collect::<Result<_, _>>()?;
            let (i, j, k, l) = (idx[0], idx[1], idx[2], idx[3]);
            if i > norb || j > norb || k > norb || l > norb {
                return Err(format!("FCIDUMP index out of range in {line:?}"));
            }
            match (i, j, k, l) {
                (0, 0, 0, 0) => fd.ecore = v,
                (_, _, 0, 0) => {
                    let (p, q) = (i - 1, j - 1);
                    fd.h1[p * norb + q] = v;
                    fd.h1[q * norb + p] = v;
                }
                _ => {
                    let (p, q, r, s) = (i - 1, j - 1, k - 1, l - 1);
                    for (a, b, c, d) in [
                        (p, q, r, s),
                        (q, p, r, s),
                        (p, q, s, r),
                        (q, p, s, r),
                        (r, s, p, q),
                        (s, r, p, q),
                        (r, s, q, p),
                        (s, r, q, p),
                    ] {
                        fd.eri[((a * norb + b) * norb + c) * norb + d] = v;
                    }
                }
            }
        }
        Ok(fd)
    }

    /// One-electron integral `h_pq` (0-based spatial orbitals).
    #[inline]
    pub fn h(&self, p: usize, q: usize) -> f64 {
        self.h1[p * self.norb + q]
    }

    /// Two-electron integral `(pq|rs)` in chemist's notation (0-based spatial orbitals).
    #[inline]
    pub fn g(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        let n = self.norb;
        self.eri[((p * n + q) * n + r) * n + s]
    }

    /// Number of spin orbitals (= qubits).
    pub fn qubits(&self) -> usize {
        2 * self.norb
    }

    /// Energy of the determinant with the given occupied spin orbitals
    /// (interleaved `2p + σ`), Slater–Condon diagonal rule.
    pub fn determinant_energy(&self, occ: &[usize]) -> f64 {
        let mut e = self.ecore;
        for &i in occ {
            e += self.h(i / 2, i / 2);
        }
        for (a, &i) in occ.iter().enumerate() {
            for &j in &occ[a + 1..] {
                let (p, q) = (i / 2, j / 2);
                e += self.g(p, p, q, q);
                if i % 2 == j % 2 {
                    e -= self.g(p, q, q, p);
                }
            }
        }
        e
    }
}

// ---------------------------------------------------------------------------
// Programs of Pauli rotations

/// A Pauli string as `(qubit, 'X' | 'Y' | 'Z')`.
pub type PauliString = Vec<(usize, u8)>;

/// One operation of a [`Program`].
#[derive(Clone, Debug)]
pub enum POp {
    /// A Clifford gate applied as-is.
    Clifford(Gate),
    /// `exp(-i θ/2 P)` with `θ = angle + mult · params[param]` (if any).
    Rot {
        /// Rotation axis.
        pauli: PauliString,
        /// Fixed angle offset (radians).
        angle: f64,
        /// Index into [`Program::params`], or `None` for a fixed rotation.
        param: Option<usize>,
        /// Multiplier applied to the parameter (unused when `param` is `None`).
        mult: f64,
    },
}

/// A circuit of Pauli rotations with optional parameters.
#[derive(Clone, Debug, Default)]
pub struct Program {
    /// Number of qubits.
    pub n: usize,
    /// Operations in application order.
    pub ops: Vec<POp>,
    /// Initial parameter values.
    pub params: Vec<f64>,
    /// Free-form `# key value` metadata lines.
    pub meta: Vec<(String, String)>,
}

fn parse_pauli(s: &[&str]) -> Result<PauliString, String> {
    let mut v = Vec::new();
    for t in s {
        let (c, q) = t.split_at(1);
        let c = c.as_bytes()[0].to_ascii_uppercase();
        if !matches!(c, b'X' | b'Y' | b'Z') {
            return Err(format!("bad Pauli factor {t:?}"));
        }
        v.push((q.parse::<usize>().map_err(|e| format!("{t:?}: {e}"))?, c));
    }
    Ok(v)
}

impl Program {
    /// Line format:
    /// `n N` · `param K VALUE` · `x q` / `h q` / `s q` / `sdg q` / `z q` / `cx a b` / `cz a b` ·
    /// `rot ANGLE X0 Y3 Z4 …` · `prot K MULT X0 Y3 …` (θ = MULT · param K) · `# key value`.
    pub fn parse(text: &str) -> Result<Program, String> {
        let mut p = Program::default();
        for (ln, line) in text.lines().enumerate() {
            let t: Vec<&str> = line.split_whitespace().collect();
            if t.is_empty() {
                continue;
            }
            let err = |e: String| format!("program line {}: {e}", ln + 1);
            let num = |s: &str| s.parse::<usize>().map_err(|e| err(e.to_string()));
            let flt = |s: &str| s.parse::<f64>().map_err(|e| err(e.to_string()));
            match t[0] {
                "#" => {
                    if t.len() >= 2 {
                        p.meta.push((t[1].to_string(), t[2..].join(" ")));
                    }
                }
                "n" => p.n = num(t[1])?,
                "param" => {
                    let k = num(t[1])?;
                    if p.params.len() <= k {
                        p.params.resize(k + 1, 0.0);
                    }
                    p.params[k] = flt(t[2])?;
                }
                "x" => p.ops.push(POp::Clifford(Gate::X(num(t[1])?))),
                "z" => p.ops.push(POp::Clifford(Gate::Z(num(t[1])?))),
                "h" => p.ops.push(POp::Clifford(Gate::H(num(t[1])?))),
                "s" => p.ops.push(POp::Clifford(Gate::S(num(t[1])?))),
                "sdg" => p.ops.push(POp::Clifford(Gate::Sdg(num(t[1])?))),
                "cx" => p
                    .ops
                    .push(POp::Clifford(Gate::Cnot(num(t[1])?, num(t[2])?))),
                "cz" => p.ops.push(POp::Clifford(Gate::Cz(num(t[1])?, num(t[2])?))),
                "rot" => p.ops.push(POp::Rot {
                    angle: flt(t[1])?,
                    pauli: parse_pauli(&t[2..]).map_err(err)?,
                    param: None,
                    mult: 0.0,
                }),
                "prot" => p.ops.push(POp::Rot {
                    angle: 0.0,
                    param: Some(num(t[1])?),
                    mult: flt(t[2])?,
                    pauli: parse_pauli(&t[3..]).map_err(err)?,
                }),
                other => return Err(err(format!("unknown op {other:?}"))),
            }
        }
        for op in &p.ops {
            match op {
                POp::Rot { pauli, param, .. } => {
                    if let Some(k) = param {
                        if *k >= p.params.len() {
                            p.params.resize(k + 1, 0.0);
                        }
                    }
                    if pauli.iter().any(|&(q, _)| q >= p.n) {
                        return Err("Pauli qubit out of range".into());
                    }
                }
                POp::Clifford(g) => {
                    if g.qubits().iter().any(|&q| q >= p.n) {
                        return Err("Clifford qubit out of range".into());
                    }
                }
            }
        }
        Ok(p)
    }

    /// Number of [`POp::Rot`] operations (fixed and parametrised).
    pub fn rotations(&self) -> usize {
        self.ops
            .iter()
            .filter(|o| matches!(o, POp::Rot { .. }))
            .count()
    }

    /// True if every Clifford gate maps `x` parts to themselves (`X, Z, S,
    /// Sdg, CZ`): then observables keep their physical `x` vector in the
    /// compressed frame and [`Span::from_program`] is the active span.
    pub fn x_preserving(&self) -> bool {
        self.ops.iter().all(|o| match o {
            POp::Clifford(g) => matches!(
                g,
                Gate::X(_) | Gate::Y(_) | Gate::Z(_) | Gate::S(_) | Gate::Sdg(_) | Gate::Cz(..)
            ),
            POp::Rot { .. } => true,
        })
    }

    /// The circuit for parameters `theta` (`None` = the stored ones).
    pub fn circuit(&self, theta: Option<&[f64]>) -> Circuit {
        let th = theta.unwrap_or(&self.params);
        let mut c = Circuit::new(self.n);
        for op in &self.ops {
            match op {
                POp::Clifford(g) => {
                    c.gate(*g);
                }
                POp::Rot {
                    pauli,
                    angle,
                    param,
                    mult,
                } => {
                    let a = angle + param.map_or(0.0, |k| mult * th[k]);
                    pauli_rotation(&mut c, pauli, a);
                }
            }
        }
        c
    }
}

/// Appends `exp(-i θ/2 P)`: basis change, CNOT ladder, `Rz(θ)`, uncompute.
pub fn pauli_rotation(c: &mut Circuit, p: &[(usize, u8)], theta: f64) {
    if p.is_empty() {
        return; // global phase
    }
    for &(q, s) in p {
        match s {
            b'X' => {
                c.h(q);
            }
            b'Y' => {
                c.sdg(q);
                c.h(q);
            }
            _ => {}
        }
    }
    let qs: Vec<usize> = p.iter().map(|&(q, _)| q).collect();
    for w in qs.windows(2) {
        c.gate(Gate::Cnot(w[0], w[1]));
    }
    let last = *qs.last().unwrap();
    c.gate(Gate::Rz(last, theta));
    for w in qs.windows(2).rev() {
        c.gate(Gate::Cnot(w[0], w[1]));
    }
    for &(q, s) in p.iter().rev() {
        match s {
            b'X' => {
                c.h(q);
            }
            b'Y' => {
                c.h(q);
                c.s(q);
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// GF(2) span of x vectors

/// A GF(2) span of `n`-bit vectors (reduced echelon basis).
#[derive(Clone, Debug)]
pub struct Span {
    /// Number of bits per vector.
    pub n: usize,
    w: usize,
    /// (pivot, row) pairs; rows reduced against each other.
    rows: Vec<(usize, Vec<u64>)>,
}

impl Span {
    /// The zero span of `n`-bit vectors.
    pub fn new(n: usize) -> Span {
        Span {
            n,
            w: n.div_ceil(64).max(1),
            rows: Vec::new(),
        }
    }

    /// Dimension of the span (number of basis rows).
    pub fn dim(&self) -> usize {
        self.rows.len()
    }

    fn reduce(&self, v: &mut [u64]) {
        for (p, r) in &self.rows {
            if v[p / 64] >> (p % 64) & 1 == 1 {
                for (a, b) in v.iter_mut().zip(r) {
                    *a ^= b;
                }
            }
        }
    }

    /// True if `v` (packed little-endian into [`Span::words`] `u64`s, bit `i` = qubit `i`) lies in the span.
    pub fn contains(&self, v: &[u64]) -> bool {
        let mut v = v.to_vec();
        self.reduce(&mut v);
        v.iter().all(|&x| x == 0)
    }

    /// Adds `v`; returns true if the dimension grew.
    pub fn push(&mut self, v: &[u64]) -> bool {
        let mut v = v.to_vec();
        self.reduce(&mut v);
        let Some(p) = (0..self.n).find(|&i| v[i / 64] >> (i % 64) & 1 == 1) else {
            return false;
        };
        for (_, r) in self.rows.iter_mut() {
            if r[p / 64] >> (p % 64) & 1 == 1 {
                for (a, b) in r.iter_mut().zip(&v) {
                    *a ^= b;
                }
            }
        }
        self.rows.push((p, v));
        true
    }

    /// Span of the physical `x` vectors of all rotation axes of `prog`
    /// (equals the compressed-state active span when `prog.x_preserving()`).
    pub fn from_program(prog: &Program) -> Span {
        let mut s = Span::new(prog.n);
        for op in &prog.ops {
            if let POp::Rot { pauli, .. } = op {
                let mut v = vec![0u64; s.w];
                for &(q, c) in pauli {
                    if c != b'Z' {
                        v[q / 64] ^= 1 << (q % 64);
                    }
                }
                s.push(&v);
            }
        }
        s
    }

    /// Number of `u64` words per packed vector, `max(1, ceil(n / 64))`.
    pub fn words(&self) -> usize {
        self.w
    }
}

// ---------------------------------------------------------------------------
// Jordan–Wigner Hamiltonian

/// `c · X^x Z^z` (not the Hermitian convention) as a sparse sum.
type Op = Vec<(Vec<u64>, num_complex::Complex64)>;

fn ladder(j: usize, dagger: bool, w: usize) -> Op {
    use num_complex::Complex64 as C;
    // a_j = ½ X_j Z_{<j} − ½ X_j Z_{≤j};  a†_j = ½ X_j Z_{<j} + ½ X_j Z_{≤j}
    let mut x = vec![0u64; w];
    x[j / 64] |= 1 << (j % 64);
    let mut m = vec![0u64; w];
    for q in 0..j {
        m[q / 64] |= 1 << (q % 64);
    }
    let mut me = m.clone();
    me[j / 64] |= 1 << (j % 64);
    let mut k1 = x.clone();
    k1.extend_from_slice(&m);
    let mut k2 = x;
    k2.extend_from_slice(&me);
    let s = if dagger { 0.5 } else { -0.5 };
    vec![(k1, C::new(0.5, 0.0)), (k2, C::new(s, 0.0))]
}

fn mul_ops(a: &Op, b: &Op, w: usize) -> Op {
    let mut out: Op = Vec::with_capacity(a.len() * b.len());
    for (ka, ca) in a {
        for (kb, cb) in b {
            // (X^x1 Z^z1)(X^x2 Z^z2) = (-1)^{z1·x2} X^{x1^x2} Z^{z1^z2}
            let mut sign = 0u32;
            let mut k = vec![0u64; 2 * w];
            for i in 0..w {
                sign += (ka[w + i] & kb[i]).count_ones();
                k[i] = ka[i] ^ kb[i];
                k[w + i] = ka[w + i] ^ kb[w + i];
            }
            let c = ca * cb * if sign % 2 == 1 { -1.0 } else { 1.0 };
            out.push((k, c));
        }
    }
    out
}

/// Statistics of a Hamiltonian build.
#[derive(Clone, Debug, Default)]
pub struct HamStats {
    /// Fermionic monomials considered (above the integral tolerance).
    pub monomials_total: u64,
    /// Monomials kept by the span filter.
    pub monomials_kept: u64,
    /// Pauli terms in the resulting Hamiltonian (after dropping near-zero coefficients).
    pub pauli_terms: usize,
    /// Wall-clock build time in seconds.
    pub secs: f64,
}

/// The Jordan–Wigner Hamiltonian
/// `E0 + Σ h_pq a†_{pσ} a_{qσ} + ½ Σ (pq|rs) a†_{pσ} a†_{rτ} a_{sτ} a_{qσ}`
/// on `2·norb` qubits (spin orbital `2p + σ`), restricted to monomials whose
/// `x` vector (`e_i ⊕ e_j ⊕ e_k ⊕ e_l`) lies in `span` (if given). The
/// returned sum includes the constant as an identity term.
pub fn jw_hamiltonian(fd: &Fcidump, span: Option<&Span>, tol: f64) -> (PauliSum, HamStats) {
    use num_complex::Complex64 as C;
    let t0 = std::time::Instant::now();
    let n = fd.qubits();
    let w = n.div_ceil(64).max(1);
    let no = fd.norb;
    let mut acc: HashMap<Vec<u64>, C> = HashMap::new();
    let mut st = HamStats::default();
    acc.insert(vec![0u64; 2 * w], C::new(fd.ecore, 0.0));
    let lad: Vec<[Op; 2]> = (0..n)
        .map(|j| [ladder(j, false, w), ladder(j, true, w)])
        .collect();
    let xvec = |idx: &[usize]| {
        let mut v = vec![0u64; w];
        for &i in idx {
            v[i / 64] ^= 1 << (i % 64);
        }
        v
    };
    let keep = |idx: &[usize]| match span {
        None => true,
        Some(s) => s.contains(&xvec(idx)),
    };
    let add = |op: Op, coef: f64, acc: &mut HashMap<Vec<u64>, C>| {
        for (k, c) in op {
            *acc.entry(k).or_insert(C::new(0.0, 0.0)) += c * coef;
        }
    };
    // one-body
    for p in 0..no {
        for q in 0..no {
            let v = fd.h(p, q);
            if v.abs() < tol {
                continue;
            }
            for s in 0..2 {
                let (i, j) = (2 * p + s, 2 * q + s);
                st.monomials_total += 1;
                if !keep(&[i, j]) {
                    continue;
                }
                st.monomials_kept += 1;
                let op = mul_ops(&lad[i][1], &lad[j][0], w);
                add(op, v, &mut acc);
            }
        }
    }
    // two-body: ½ (pq|rs) a†_{pσ} a†_{rτ} a_{sτ} a_{qσ}
    for p in 0..no {
        for q in 0..no {
            for r in 0..no {
                for s in 0..no {
                    let v = fd.g(p, q, r, s);
                    if v.abs() < tol {
                        continue;
                    }
                    for sg in 0..2 {
                        for tu in 0..2 {
                            let (i, j, k, l) = (2 * p + sg, 2 * q + sg, 2 * r + tu, 2 * s + tu);
                            if i == k || j == l {
                                continue;
                            }
                            st.monomials_total += 1;
                            if !keep(&[i, j, k, l]) {
                                continue;
                            }
                            st.monomials_kept += 1;
                            let a = mul_ops(&lad[i][1], &lad[k][1], w);
                            let b = mul_ops(&lad[l][0], &lad[j][0], w);
                            add(mul_ops(&a, &b, w), 0.5 * v, &mut acc);
                        }
                    }
                }
            }
        }
    }
    // to the Hermitian convention: X^x Z^z = i^{-|x∧z|} P
    let mut keys = Vec::new();
    let mut coefs = Vec::new();
    let mut entries: Vec<_> = acc.into_iter().collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    for (k, c) in entries {
        let y: u32 = (0..w).map(|i| (k[i] & k[w + i]).count_ones()).sum();
        let ph = match y % 4 {
            0 => C::new(1.0, 0.0),
            1 => C::new(0.0, -1.0),
            2 => C::new(-1.0, 0.0),
            _ => C::new(0.0, 1.0),
        };
        let c = c * ph;
        debug_assert!(c.im.abs() < 1e-9, "non-Hermitian coefficient {c}");
        if c.re.abs() < 1e-14 {
            continue;
        }
        keys.extend_from_slice(&k);
        coefs.push(c.re);
    }
    st.pauli_terms = coefs.len();
    st.secs = t0.elapsed().as_secs_f64();
    (PauliSum { n, w, keys, coefs }, st)
}

/// `<ψ|H|ψ>` for a [`PauliSum`] on a dense little-endian state vector (tests).
pub fn sv_expectation(amps: &[num_complex::Complex64], h: &PauliSum) -> f64 {
    use num_complex::Complex64 as C;
    let (n, w) = (h.n, h.w);
    assert!(n <= 30 && amps.len() == 1 << n);
    let mut e = 0.0;
    for (k, &c) in h.keys.chunks(2 * w).zip(&h.coefs) {
        let (x, z) = (k[0] as usize, k[w] as usize);
        let y = (x & z).count_ones();
        let ph = [
            C::new(1.0, 0.0),
            C::new(0.0, 1.0),
            C::new(-1.0, 0.0),
            C::new(0.0, -1.0),
        ][(y % 4) as usize];
        // <ψ| i^y X^x Z^z |ψ> = Σ_b conj(ψ[b ^ x]) i^y (-1)^{z·b} ψ[b]
        let mut s = C::new(0.0, 0.0);
        for (b, &a) in amps.iter().enumerate() {
            let t = amps[b ^ x].conj() * a;
            if (z & b).count_ones() % 2 == 1 {
                s -= t;
            } else {
                s += t;
            }
        }
        e += c * (ph * s).re;
    }
    e
}

// ---------------------------------------------------------------------------
// Energies and optimisation

/// Exact energy of `prog(theta)` with the compressed state; `h` should be
/// built with the program's span (or unfiltered).
pub fn energy(
    prog: &Program,
    theta: Option<&[f64]>,
    h: &PauliSum,
    max_d: usize,
) -> Result<(f64, CompressedState), SimError> {
    let c = prog.circuit(theta);
    let st = CompressedState::new(&c, max_d)?;
    let e = st.expectation(h);
    Ok((e, st))
}

/// Minimises `a0 + a1 cos t + b1 sin t + a2 cos 2t + b2 sin 2t` given its
/// values at `t0 + 2πk/5`; returns `(t*, f(t*))`.
pub fn trig2_min(t0: f64, vals: &[f64; 5]) -> (f64, f64) {
    let step = 2.0 * std::f64::consts::PI / 5.0;
    let mut c = [0.0f64; 5]; // a0, a1, b1, a2, b2
    for (k, &v) in vals.iter().enumerate() {
        let t = t0 + step * k as f64;
        c[0] += v / 5.0;
        c[1] += 2.0 * v * t.cos() / 5.0;
        c[2] += 2.0 * v * t.sin() / 5.0;
        c[3] += 2.0 * v * (2.0 * t).cos() / 5.0;
        c[4] += 2.0 * v * (2.0 * t).sin() / 5.0;
    }
    let f = |t: f64| {
        c[0] + c[1] * t.cos() + c[2] * t.sin() + c[3] * (2.0 * t).cos() + c[4] * (2.0 * t).sin()
    };
    let df = |t: f64| {
        -c[1] * t.sin() + c[2] * t.cos() - 2.0 * c[3] * (2.0 * t).sin()
            + 2.0 * c[4] * (2.0 * t).cos()
    };
    let d2f = |t: f64| {
        -c[1] * t.cos()
            - c[2] * t.sin()
            - 4.0 * c[3] * (2.0 * t).cos()
            - 4.0 * c[4] * (2.0 * t).sin()
    };
    let mut best = (0.0, f64::INFINITY);
    for i in 0..720 {
        let t = -std::f64::consts::PI + i as f64 * std::f64::consts::PI / 360.0;
        let v = f(t);
        if v < best.1 {
            best = (t, v);
        }
    }
    let mut t = best.0;
    for _ in 0..30 {
        let h2 = d2f(t);
        if h2 <= 0.0 {
            break;
        }
        let dt = df(t) / h2;
        t -= dt;
        if dt.abs() < 1e-15 {
            break;
        }
    }
    if f(t) <= best.1 {
        (t, f(t))
    } else {
        best
    }
}

/// Rotosolve-style exact coordinate descent: each parameter of an
/// excitation generator `exp(θ G)` (`G³ = −G`) enters the energy as a
/// degree-2 trigonometric polynomial, fixed by 5 evaluations. Returns the
/// optimised parameters and the energy after each sweep. Only parameters
/// `0..nopt` are optimised (the rest keep their initial values).
pub fn rotosolve(
    prog: &Program,
    h: &PauliSum,
    sweeps: usize,
    tol: f64,
    max_d: usize,
    nopt: usize,
) -> Result<(Vec<f64>, Vec<f64>, usize), SimError> {
    let mut th = prog.params.clone();
    let step = 2.0 * std::f64::consts::PI / 5.0;
    let mut hist = Vec::new();
    let mut evals = 0usize;
    let mut cur = energy(prog, Some(&th), h, max_d)?.0;
    evals += 1;
    hist.push(cur);
    for _ in 0..sweeps {
        for k in 0..th.len().min(nopt) {
            let t0 = th[k];
            let mut vals = [0.0f64; 5];
            vals[0] = cur;
            for (j, v) in vals.iter_mut().enumerate().skip(1) {
                th[k] = t0 + step * j as f64;
                *v = energy(prog, Some(&th), h, max_d)?.0;
                evals += 1;
            }
            let (t, fmin) = trig2_min(t0, &vals);
            // keep the angle in (-π, π]; avoid exact Clifford points (a zero
            // angle would drop a rotation and shrink the frame, harmlessly)
            th[k] = t;
            cur = fmin;
        }
        let e = energy(prog, Some(&th), h, max_d)?.0;
        evals += 1;
        cur = e;
        let done = hist.last().is_some_and(|&p| (p - e).abs() < tol);
        hist.push(e);
        if done {
            break;
        }
    }
    Ok((th, hist, evals))
}

// ---------------------------------------------------------------------------
// Best state in the register (a classical comparator)

/// `H` restricted to the compressed register, grouped by `x`: for each `x`,
/// the `(z, c·i^{|x∧z|})` pairs.
pub struct RegisterHamiltonian {
    /// Number of active register qubits (`2^d` amplitudes).
    pub d: usize,
    groups: Vec<(u64, Vec<(u64, num_complex::Complex64)>)>,
}

impl RegisterHamiltonian {
    /// Restricts `h` to the active register of `st` and groups the terms by `x`.
    pub fn new(st: &CompressedState, h: &PauliSum) -> RegisterHamiltonian {
        use num_complex::Complex64 as C;
        let mut terms = st.register_terms(h);
        terms.sort_unstable_by_key(|t| (t.0, t.1));
        let mut groups: Vec<(u64, Vec<(u64, C)>)> = Vec::new();
        for (x, z, c) in terms {
            let y = (x & z).count_ones();
            let ph = [
                C::new(1.0, 0.0),
                C::new(0.0, 1.0),
                C::new(-1.0, 0.0),
                C::new(0.0, -1.0),
            ][(y % 4) as usize];
            match groups.last_mut() {
                Some((gx, v)) if *gx == x => v.push((z, ph * c)),
                _ => groups.push((x, vec![(z, ph * c)])),
            }
        }
        RegisterHamiltonian {
            d: st.active_qubits(),
            groups,
        }
    }

    /// Number of distinct `x` groups.
    pub fn groups(&self) -> usize {
        self.groups.len()
    }

    /// `out = H v` on the `2^d` register.
    pub fn apply(&self, v: &[num_complex::Complex64], out: &mut [num_complex::Complex64]) {
        use num_complex::Complex64 as C;
        let len = v.len();
        assert_eq!(len, 1 << self.d);
        out.iter_mut().for_each(|o| *o = C::new(0.0, 0.0));
        let mut diag = vec![C::new(0.0, 0.0); len];
        for (x, zs) in &self.groups {
            diag.iter_mut().for_each(|o| *o = C::new(0.0, 0.0));
            if zs.len() <= self.d + 1 {
                for &(z, c) in zs {
                    for (b, dv) in diag.iter_mut().enumerate() {
                        if (z & b as u64).count_ones() % 2 == 1 {
                            *dv -= c;
                        } else {
                            *dv += c;
                        }
                    }
                }
            } else {
                // D(b) = Σ_z c_z (-1)^{z·b}: a Walsh–Hadamard transform of c
                for &(z, c) in zs {
                    diag[z as usize] += c;
                }
                let mut h = 1;
                while h < len {
                    for i in (0..len).step_by(2 * h) {
                        for j in i..i + h {
                            let (a, b) = (diag[j], diag[j + h]);
                            diag[j] = a + b;
                            diag[j + h] = a - b;
                        }
                    }
                    h *= 2;
                }
            }
            let x = *x as usize;
            for b in 0..len {
                out[b ^ x] += diag[b] * v[b];
            }
        }
    }
}

/// Lowest eigenvalue of `H` restricted to the compressed register of `st`,
/// in the symmetry sector of the register state `φ` (plain Lanczos started
/// from `φ`, no re-orthogonalisation: the lowest Ritz value converges, ghost
/// copies are harmless). Every circuit whose rotations stay in this span has
/// energy `>=` this number (variational principle). Returns the Ritz value
/// after each iteration.
pub fn register_ground(
    st: &CompressedState,
    h: &PauliSum,
    iters: usize,
    tol: f64,
) -> (Vec<f64>, usize) {
    use num_complex::Complex64 as C;
    let rh = RegisterHamiltonian::new(st, h);
    let mut v: Vec<C> = st.active_amplitudes().to_vec();
    let nrm = v.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
    v.iter_mut().for_each(|a| *a /= nrm);
    let len = v.len();
    let mut vprev = vec![C::new(0.0, 0.0); len];
    let mut w = vec![C::new(0.0, 0.0); len];
    let (mut alphas, mut betas) = (Vec::new(), Vec::new());
    let mut hist = Vec::new();
    let mut beta = 0.0f64;
    for it in 0..iters {
        rh.apply(&v, &mut w);
        let alpha: f64 = v.iter().zip(&w).map(|(a, b)| (a.conj() * b).re).sum();
        for i in 0..len {
            w[i] -= v[i] * alpha + vprev[i] * beta;
        }
        alphas.push(alpha);
        let ritz = lowest_tridiag(&alphas, &betas);
        hist.push(ritz);
        beta = w.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        let conv = it > 2 && (hist[it - 1] - ritz).abs() < tol;
        if beta < 1e-12 || conv {
            break;
        }
        betas.push(beta);
        std::mem::swap(&mut vprev, &mut v);
        for i in 0..len {
            v[i] = w[i] / beta;
        }
    }
    (hist, rh.groups())
}

/// Lowest eigenvalue of the symmetric tridiagonal matrix (bisection).
fn lowest_tridiag(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for i in 0..n {
        let r =
            (if i > 0 { b[i - 1].abs() } else { 0.0 }) + (if i + 1 < n { b[i].abs() } else { 0.0 });
        lo = lo.min(a[i] - r);
        hi = hi.max(a[i] + r);
    }
    // count eigenvalues < x (Sturm sequence)
    let count = |x: f64| {
        let mut c = 0;
        let mut q = 1.0f64;
        for i in 0..n {
            let off = if i > 0 { b[i - 1] * b[i - 1] } else { 0.0 };
            q = a[i] - x - if i > 0 { off / q } else { 0.0 };
            if q == 0.0 {
                q = 1e-300;
            }
            if q < 0.0 {
                c += 1;
            }
        }
        c
    };
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if count(mid) >= 1 {
            hi = mid;
        } else {
            lo = mid;
        }
        if hi - lo < 1e-13 * (1.0 + hi.abs()) {
            break;
        }
    }
    0.5 * (lo + hi)
}
