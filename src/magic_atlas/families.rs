//! Circuit families of the magic atlas: real algorithm circuits at any `n`,
//! built only from gates the rotation frame understands (Cliffords, Z
//! rotations, `Rx/Ry/U`, `CPhase`, Toffoli; everything is lowered to
//! Clifford + Z rotations by `Gate::decompose_to_clifford_rz`).
//!
//! A family is selected by a spec string `name:key=value,...`, e.g.
//! `qft:n=64,cut=0,in=basis`. Unknown keys are an error.

use crate::circuit::Circuit;
use crate::gate::Gate;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;
use std::f64::consts::PI;

/// A parsed family spec.
#[derive(Clone, Debug)]
pub struct Spec {
    pub family: String,
    pub params: BTreeMap<String, String>,
}

impl Spec {
    pub fn parse(s: &str) -> Result<Spec, String> {
        let (family, rest) = match s.split_once(':') {
            Some((f, r)) => (f.to_string(), r),
            None => (s.to_string(), ""),
        };
        let mut params = BTreeMap::new();
        for kv in rest.split(',').filter(|x| !x.is_empty()) {
            let (k, v) = kv
                .split_once('=')
                .ok_or_else(|| format!("bad parameter {kv:?} in {s:?}"))?;
            params.insert(k.trim().to_string(), v.trim().to_string());
        }
        Ok(Spec { family, params })
    }

    fn take<T: std::str::FromStr>(
        &self,
        used: &mut Vec<String>,
        k: &str,
        default: T,
    ) -> Result<T, String> {
        used.push(k.to_string());
        match self.params.get(k) {
            None => Ok(default),
            Some(v) => v
                .parse()
                .map_err(|_| format!("bad value {v:?} for {k} in {}", self.family)),
        }
    }
}

/// Builds the circuit for `spec` (all randomness from `seed`).
pub fn build(spec: &str, seed: u64) -> Result<Circuit, String> {
    let sp = Spec::parse(spec)?;
    let mut used = Vec::new();
    let mut rng = StdRng::seed_from_u64(seed);
    let u = &mut used;
    let c = match sp.family.as_str() {
        "qft" => {
            let n: usize = sp.take(u, "n", 16)?;
            let cut: usize = sp.take(u, "cut", 0)?;
            let inp: String = sp.take(u, "in", "basis".to_string())?;
            let mut c = Circuit::new(n);
            prep(&mut c, &(0..n).collect::<Vec<_>>(), &inp, &mut rng)?;
            qft(&mut c, &(0..n).collect::<Vec<_>>(), cut, false, true);
            c
        }
        "cuccaro" | "gidney" | "draper" => {
            let bits: usize = sp.take(u, "bits", 8)?;
            let inp: String = sp.take(u, "in", "basis".to_string())?;
            let cut: usize = sp.take(u, "cut", 0)?;
            adder(&sp.family, bits, &inp, cut, &mut rng)?
        }
        "shorwin" => {
            let nb: usize = sp.take(u, "nbits", 6)?;
            let w: usize = sp.take(u, "w", 2)?;
            let inp: String = sp.take(u, "in", "one".to_string())?;
            shor_window_oracle(nb, w, &inp)?
        }
        "shor" => {
            let nb: usize = sp.take(u, "nbits", 4)?;
            let w: usize = sp.take(u, "w", 2)?;
            let cnt: usize = sp.take(u, "cnt", 2 * nb)?;
            shor_full(nb, w, cnt)?
        }
        "grover" => {
            let n: usize = sp.take(u, "n", 10)?;
            let it: usize = sp.take(u, "it", 1)?;
            grover(n, it, &mut rng)
        }
        "ising" => {
            let n: usize = sp.take(u, "n", 16)?;
            let steps: usize = sp.take(u, "steps", 4)?;
            let dt: f64 = sp.take(u, "dt", 0.1)?;
            let j: f64 = sp.take(u, "J", 1.0)?;
            let h: f64 = sp.take(u, "h", 1.0)?;
            let inp: String = sp.take(u, "in", "zero".to_string())?;
            let mut c = Circuit::new(n);
            prep(&mut c, &(0..n).collect::<Vec<_>>(), &inp, &mut rng)?;
            for _ in 0..steps {
                ising_step(&mut c, &(0..n).collect::<Vec<_>>(), j * dt, h * dt, None);
            }
            c
        }
        "heis" => {
            let n: usize = sp.take(u, "n", 16)?;
            let steps: usize = sp.take(u, "steps", 4)?;
            let dt: f64 = sp.take(u, "dt", 0.1)?;
            let inp: String = sp.take(u, "in", "neel".to_string())?;
            let mut c = Circuit::new(n);
            prep(&mut c, &(0..n).collect::<Vec<_>>(), &inp, &mut rng)?;
            for _ in 0..steps {
                heis_step(&mut c, n, dt);
            }
            c
        }
        "qaoa" => {
            let n: usize = sp.take(u, "n", 16)?;
            let p: usize = sp.take(u, "p", 1)?;
            let graph: String = sp.take(u, "graph", "reg3".to_string())?;
            qaoa(n, p, &graph, &mut rng)?
        }
        "hea" => {
            let n: usize = sp.take(u, "n", 16)?;
            let layers: usize = sp.take(u, "layers", 1)?;
            hea(n, layers, &mut rng)
        }
        "qpe" => {
            let t: usize = sp.take(u, "t", 8)?;
            let s: usize = sp.take(u, "s", 8)?;
            let kind: String = sp.take(u, "kind", "stab".to_string())?;
            qpe(t, s, &kind, &mut rng)?
        }
        "walk" => {
            let m: usize = sp.take(u, "m", 4)?;
            let steps: usize = sp.take(u, "steps", 2)?;
            walk(m, steps)
        }
        "hhl" => {
            let t: usize = sp.take(u, "t", 4)?;
            let m: usize = sp.take(u, "m", 2)?;
            hhl(t, m)?
        }
        "rct" => {
            // random Clifford+T: `layers` of random 1q Cliffords + a CNOT
            // brickwork, `t` T gates (each followed by H) at random slots
            // (same as simulability's `ct` family, NN).
            let n: usize = sp.take(u, "n", 16)?;
            let layers: usize = sp.take(u, "L", 8)?;
            let t: usize = sp.take(u, "t", 16)?;
            rct(n, layers, t, &mut rng)
        }
        f => return Err(format!("unknown family {f:?}")),
    };
    for k in sp.params.keys() {
        if !used.contains(k) {
            return Err(format!("unknown parameter {k:?} for family {}", sp.family));
        }
    }
    Ok(c)
}

/// Input preparation on `qs`: `zero` (nothing), `basis` (random classical
/// bits), `plus` (H on all), `neel` (X on odd), `graph` (H on all + CZ
/// chain: a cluster state, a generic stabilizer input).
pub fn prep(c: &mut Circuit, qs: &[usize], inp: &str, rng: &mut StdRng) -> Result<(), String> {
    match inp {
        "zero" => {}
        "basis" => {
            for &q in qs {
                if rng.random::<bool>() {
                    c.x(q);
                }
            }
        }
        "plus" => {
            for &q in qs {
                c.h(q);
            }
        }
        "neel" => {
            for &q in qs.iter().skip(1).step_by(2) {
                c.x(q);
            }
        }
        "graph" => {
            for &q in qs {
                c.h(q);
            }
            for w in qs.windows(2) {
                c.cz(w[0], w[1]);
            }
        }
        _ => return Err(format!("unknown input {inp:?}")),
    }
    Ok(())
}

/// QFT on `qs` (`qs[0]` = LSB). `cut > 0` drops controlled rotations by
/// angles `π/2^k` with `k > cut` (the approximate QFT of Coppersmith /
/// Barenco et al.; `cut = 1` keeps only the CZs, i.e. Clifford).
/// `swaps` adds the bit reversal. `inverse` gives the inverse transform.
pub fn qft(c: &mut Circuit, qs: &[usize], cut: usize, inverse: bool, swaps: bool) {
    let n = qs.len();
    let mut sub = Circuit::new(c.num_qubits);
    for j in (0..n).rev() {
        sub.h(qs[j]);
        for k in (0..j).rev() {
            let dist = j - k;
            if cut > 0 && dist > cut {
                continue;
            }
            sub.cphase(qs[k], qs[j], PI / 2f64.powi(dist as i32));
        }
    }
    if swaps {
        for j in 0..n / 2 {
            sub.swap(qs[j], qs[n - 1 - j]);
        }
    }
    if inverse {
        sub = sub.inverse();
    }
    c.append(&sub);
}

/// Q#'s `ApplyAnd` (Jones 2013 / Gidney 2018): `|a,b,0> -> |a,b,ab>` with
/// 4 T gates. Only valid when the target is `|0>` on entry.
pub fn and_compute(c: &mut Circuit, a: usize, b: usize, t: usize) {
    c.h(t).t(t).cnot(a, t).cnot(b, t);
    c.cnot(t, a).cnot(t, b);
    c.gate(Gate::Tdg(a)).gate(Gate::Tdg(b)).t(t);
    c.cnot(t, b).cnot(t, a);
    c.h(t).s(t);
}

/// Unitary inverse of [`and_compute`] (Gidney's adder uncomputes by
/// measurement; a unitary circuit needs the adjoint, 4 more T gates).
pub fn and_uncompute(c: &mut Circuit, a: usize, b: usize, t: usize) {
    let mut s = Circuit::new(c.num_qubits);
    and_compute(&mut s, a, b, t);
    c.append(&s.inverse());
}

/// Gidney's adder (arXiv:1709.06648) `b += a mod 2^bits`, carries in
/// `anc` (`bits - 1` ancillas, |0> in and out).
pub fn gidney_add(c: &mut Circuit, a: &[usize], b: &[usize], anc: &[usize], toffoli: bool) {
    let n = a.len();
    assert!(b.len() == n && anc.len() + 1 >= n);
    let and = |c: &mut Circuit, x, y, t| {
        if toffoli {
            c.ccx(x, y, t);
        } else {
            and_compute(c, x, y, t);
        }
    };
    let unand = |c: &mut Circuit, x, y, t| {
        if toffoli {
            c.ccx(x, y, t);
        } else {
            and_uncompute(c, x, y, t);
        }
    };
    // anc[i] holds carry c_{i+1}
    for i in 0..n.saturating_sub(1) {
        if i > 0 {
            c.cnot(anc[i - 1], a[i]).cnot(anc[i - 1], b[i]);
        }
        and(c, a[i], b[i], anc[i]);
        if i > 0 {
            c.cnot(anc[i - 1], anc[i]);
        }
    }
    if n >= 2 {
        c.cnot(anc[n - 2], b[n - 1]);
    }
    c.cnot(a[n - 1], b[n - 1]);
    for i in (0..n.saturating_sub(1)).rev() {
        if i > 0 {
            c.cnot(anc[i - 1], anc[i]);
        }
        unand(c, a[i], b[i], anc[i]);
        if i > 0 {
            c.cnot(anc[i - 1], a[i]);
        }
        c.cnot(a[i], b[i]);
    }
}

/// Draper's QFT adder `b += a mod 2^bits` (quantum-quantum, no ancilla):
/// QFT(b), controlled phases from `a`, inverse QFT. `cut` as in [`qft`]
/// (applied to the transforms and to the adder phases).
pub fn draper_add(c: &mut Circuit, a: &[usize], b: &[usize], cut: usize) {
    let n = b.len();
    qft(c, b, cut, false, false);
    // after qft_noswap-style transform, b[j] carries phase 2π b / 2^{j+1}
    for j in 0..n {
        for (i, &ai) in a.iter().enumerate().take(j + 1) {
            let dist = j - i; // angle 2π 2^i / 2^{j+1} = π / 2^{j-i}
            if cut > 0 && dist > cut {
                continue;
            }
            c.cphase(ai, b[j], PI / 2f64.powi(dist as i32));
        }
    }
    qft(c, b, cut, true, false);
}

fn adder(
    kind: &str,
    bits: usize,
    inp: &str,
    cut: usize,
    rng: &mut StdRng,
) -> Result<Circuit, String> {
    // layout: a = 0..bits, b = bits..2bits (+1 for cuccaro), ancillas after
    let a: Vec<usize> = (0..bits).collect();
    let bl = if kind == "cuccaro" { bits + 1 } else { bits };
    let b: Vec<usize> = (bits..bits + bl).collect();
    let nanc = match kind {
        "cuccaro" => 1,
        "gidney" => bits.saturating_sub(1),
        _ => 0,
    };
    let anc: Vec<usize> = (bits + bl..bits + bl + nanc).collect();
    let mut c = Circuit::new(bits + bl + nanc);
    match inp {
        "basis" => {
            prep(&mut c, &a, "basis", rng)?;
            prep(&mut c, &b[..bits], "basis", rng)?;
        }
        "plusa" => {
            prep(&mut c, &a, "plus", rng)?;
            prep(&mut c, &b[..bits], "basis", rng)?;
        }
        "plusab" => {
            prep(&mut c, &a, "plus", rng)?;
            prep(&mut c, &b[..bits], "plus", rng)?;
        }
        _ => return Err(format!("unknown adder input {inp:?}")),
    }
    match kind {
        "cuccaro" => crate::shor_ripple::cuccaro_add(&mut c, &a, &b, anc[0]),
        "gidney" => gidney_add(&mut c, &a, &b, &anc, false),
        "draper" => draper_add(&mut c, &a, &b, cut),
        _ => unreachable!(),
    }
    Ok(c)
}

fn shor_modulus(nb: usize) -> (u64, u64) {
    // largest odd n_mod < 2^nb with 2^(nb-1) <= n_mod, not a prime power
    // test-free choice: n_mod = 2^nb - 1 (odd; composite for nb >= 4 except
    // Mersenne primes, harmless here: only the gate structure matters);
    // a = smallest base >= 2 coprime to n_mod.
    let n_mod = (1u64 << nb) - 1;
    let a = (2..n_mod)
        .find(|&a| crate::algorithms::gcd(a, n_mod) == 1)
        .unwrap_or(2);
    (n_mod.max(3), a)
}

/// One controlled `U_a` of the windowed oracle (shor_window.rs), control in
/// `|+>`, work register `x = 1` (`in=one`) or a uniform superposition of
/// its low half (`in=half`).
fn shor_window_oracle(nb: usize, w: usize, inp: &str) -> Result<Circuit, String> {
    let (n_mod, a) = shor_modulus(nb);
    let lay = crate::shor_window::WindowLayout::new(nb, w);
    let mut c = Circuit::new(lay.num_qubits());
    c.h(lay.ctrl);
    match inp {
        "one" => {
            c.x(lay.x[0]);
        }
        "half" => {
            for &q in &lay.x[..nb / 2] {
                c.h(q);
            }
        }
        _ => return Err(format!("unknown shorwin input {inp:?}")),
    }
    c.append(&crate::shor_window::controlled_ua(&lay, a, n_mod));
    Ok(c)
}

/// Full order finding: `cnt` counting qubits in `|+>`, `x = 1`, controlled
/// `U_{a^{2^j}}` (windowed oracle, its control remapped to counting qubit
/// `j`), inverse QFT on the counting register.
fn shor_full(nb: usize, w: usize, cnt: usize) -> Result<Circuit, String> {
    let (n_mod, a) = shor_modulus(nb);
    let lay = crate::shor_window::WindowLayout::new(nb, w);
    let nw = lay.num_qubits(); // includes the oracle's own control slot 0
    let n = cnt + nw - 1;
    // map oracle qubit q (q >= 1) -> cnt + q - 1; control 0 -> counting j
    let mut c = Circuit::new(n);
    for j in 0..cnt {
        c.h(j);
    }
    c.x(cnt + lay.x[0] - 1);
    let mut aj = a % n_mod;
    for j in 0..cnt {
        let o = crate::shor_window::controlled_ua(&lay, aj, n_mod);
        for g in o.gates() {
            let m = |q: usize| if q == 0 { j } else { cnt + q - 1 };
            c.gate(remap(g, &m));
        }
        aj = (u128::from(aj) * u128::from(aj) % u128::from(n_mod)) as u64;
    }
    qft(&mut c, &(0..cnt).collect::<Vec<_>>(), 0, true, true);
    Ok(c)
}

/// Gate with every qubit index passed through `m`.
pub fn remap(g: &Gate, m: &dyn Fn(usize) -> usize) -> Gate {
    use Gate::*;
    match *g {
        I(q) => I(m(q)),
        H(q) => H(m(q)),
        X(q) => X(m(q)),
        Y(q) => Y(m(q)),
        Z(q) => Z(m(q)),
        S(q) => S(m(q)),
        Sdg(q) => Sdg(m(q)),
        T(q) => T(m(q)),
        Tdg(q) => Tdg(m(q)),
        Sx(q) => Sx(m(q)),
        Sxdg(q) => Sxdg(m(q)),
        Rx(q, t) => Rx(m(q), t),
        Ry(q, t) => Ry(m(q), t),
        Rz(q, t) => Rz(m(q), t),
        Phase(q, t) => Phase(m(q), t),
        U(q, a, b, l) => U(m(q), a, b, l),
        Cnot(a, b) => Cnot(m(a), m(b)),
        Cz(a, b) => Cz(m(a), m(b)),
        Swap(a, b) => Swap(m(a), m(b)),
        ISwap(a, b) => ISwap(m(a), m(b)),
        ISwapdg(a, b) => ISwapdg(m(a), m(b)),
        CPhase(a, b, t) => CPhase(m(a), m(b), t),
        Ccx(a, b, t) => Ccx(m(a), m(b), m(t)),
    }
}

/// Multi-controlled Z on `qs` with a Toffoli ladder into `anc`
/// (`qs.len() - 2` clean ancillas).
pub fn mcz(c: &mut Circuit, qs: &[usize], anc: &[usize]) {
    let k = qs.len();
    match k {
        0 => {}
        1 => {
            c.z(qs[0]);
        }
        2 => {
            c.cz(qs[0], qs[1]);
        }
        _ => {
            c.ccx(qs[0], qs[1], anc[0]);
            for i in 2..k - 1 {
                c.ccx(anc[i - 2], qs[i], anc[i - 1]);
            }
            c.cz(anc[k - 3], qs[k - 1]);
            for i in (2..k - 1).rev() {
                c.ccx(anc[i - 2], qs[i], anc[i - 1]);
            }
            c.ccx(qs[0], qs[1], anc[0]);
        }
    }
}

/// Grover on `n` search qubits (+ `n - 2` ladder ancillas): `it`
/// iterations of (phase oracle for a random marked string, diffusion).
fn grover(n: usize, it: usize, rng: &mut StdRng) -> Circuit {
    assert!(n >= 2);
    let qs: Vec<usize> = (0..n).collect();
    let anc: Vec<usize> = (n..n + n.saturating_sub(2)).collect();
    let mut c = Circuit::new(n + anc.len());
    let marked: Vec<bool> = (0..n).map(|_| rng.random()).collect();
    for &q in &qs {
        c.h(q);
    }
    for _ in 0..it {
        for (&q, &m) in qs.iter().zip(&marked) {
            if !m {
                c.x(q);
            }
        }
        mcz(&mut c, &qs, &anc);
        for (&q, &m) in qs.iter().zip(&marked) {
            if !m {
                c.x(q);
            }
        }
        for &q in &qs {
            c.h(q);
            c.x(q);
        }
        mcz(&mut c, &qs, &anc);
        for &q in &qs {
            c.x(q);
            c.h(q);
        }
    }
    c
}

/// `exp(-i θ Z_a Z_b / 2)`.
fn zz(c: &mut Circuit, a: usize, b: usize, theta: f64) {
    c.cnot(a, b).rz(b, theta).cnot(a, b);
}

/// One first-order Trotter step of `H = J Σ Z_i Z_{i+1} + h Σ X_i` on a
/// chain: `exp(-i J dt ZZ)` on every bond, then `exp(-i h dt X)`. With a
/// `ctrl`, every rotation is controlled (for phase estimation).
pub fn ising_step(c: &mut Circuit, qs: &[usize], jdt: f64, hdt: f64, ctrl: Option<usize>) {
    for w in qs.windows(2) {
        match ctrl {
            None => zz(c, w[0], w[1], 2.0 * jdt),
            Some(k) => {
                c.cnot(w[0], w[1]);
                crz(c, k, w[1], 2.0 * jdt);
                c.cnot(w[0], w[1]);
            }
        }
    }
    for &q in qs {
        match ctrl {
            None => {
                c.rx(q, 2.0 * hdt);
            }
            Some(k) => {
                c.h(q);
                crz(c, k, q, 2.0 * hdt);
                c.h(q);
            }
        }
    }
}

/// Controlled `Rz(θ)` on `t`.
pub fn crz(c: &mut Circuit, k: usize, t: usize, theta: f64) {
    c.rz(t, theta / 2.0)
        .cnot(k, t)
        .rz(t, -theta / 2.0)
        .cnot(k, t);
}

/// One Trotter step of the XXX Heisenberg chain, even then odd bonds:
/// `exp(-i dt (XX + YY + ZZ))` per bond.
fn heis_step(c: &mut Circuit, n: usize, dt: f64) {
    for parity in 0..2 {
        for i in (parity..n.saturating_sub(1)).step_by(2) {
            let (a, b) = (i, i + 1);
            zz(c, a, b, 2.0 * dt);
            c.h(a).h(b);
            zz(c, a, b, 2.0 * dt);
            c.h(a).h(b);
            c.sdg(a).sdg(b).h(a).h(b);
            zz(c, a, b, 2.0 * dt);
            c.h(a).h(b).s(a).s(b);
        }
    }
}

fn graph_edges(n: usize, graph: &str, rng: &mut StdRng) -> Result<Vec<(usize, usize)>, String> {
    let mut e = Vec::new();
    match graph {
        "ring" => {
            for i in 0..n {
                e.push((i, (i + 1) % n));
            }
        }
        "reg3" => {
            // ring + random perfect matching (3-regular up to collisions)
            for i in 0..n {
                e.push((i, (i + 1) % n));
            }
            let mut p: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = rng.random_range(0..=i);
                p.swap(i, j);
            }
            for k in (0..n - 1).step_by(2) {
                let (a, b) = (p[k].min(p[k + 1]), p[k].max(p[k + 1]));
                if b - a != 1 && !(a == 0 && b == n - 1) {
                    e.push((a, b));
                }
            }
        }
        _ => return Err(format!("unknown graph {graph:?}")),
    }
    Ok(e)
}

fn qaoa(n: usize, p: usize, graph: &str, rng: &mut StdRng) -> Result<Circuit, String> {
    let edges = graph_edges(n, graph, rng)?;
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    for _ in 0..p {
        let gamma: f64 = rng.random_range(0.1..1.4);
        let beta: f64 = rng.random_range(0.1..1.4);
        for &(a, b) in &edges {
            zz(&mut c, a, b, 2.0 * gamma);
        }
        for q in 0..n {
            c.rx(q, 2.0 * beta);
        }
    }
    Ok(c)
}

/// Hardware-efficient ansatz: per layer `Ry Rz` on every qubit and a CNOT
/// ladder; a final `Ry` layer. Random angles.
fn hea(n: usize, layers: usize, rng: &mut StdRng) -> Circuit {
    let mut c = Circuit::new(n);
    for _ in 0..layers {
        for q in 0..n {
            c.ry(q, rng.random_range(-PI..PI));
            c.rz(q, rng.random_range(-PI..PI));
        }
        for q in 0..n.saturating_sub(1) {
            c.cnot(q, q + 1);
        }
    }
    for q in 0..n {
        c.ry(q, rng.random_range(-PI..PI));
    }
    c
}

/// Phase estimation with `t` counting qubits on an `s`-qubit system.
/// `kind=stab`: the system is a GHZ state, an eigenstate of
/// `U = exp(-iθ X^{⊗s}) Π_i exp(-iφ_i Z_i Z_{i+1})` (random angles);
/// `controlled-U^{2^k}` is a product of controlled Pauli rotations.
/// `kind=trotter`: `U` = one transverse-field Ising Trotter step on `|0^s>`
/// (not an eigenstate), `U^{2^k}` by repetition (keep `t` small).
fn qpe(t: usize, s: usize, kind: &str, rng: &mut StdRng) -> Result<Circuit, String> {
    let n = t + s;
    let sys: Vec<usize> = (t..n).collect();
    let mut c = Circuit::new(n);
    for k in 0..t {
        c.h(k);
    }
    match kind {
        "stab" => {
            c.h(sys[0]);
            for w in sys.windows(2) {
                c.cnot(w[0], w[1]);
            }
            let theta: f64 = rng.random_range(0.1..3.0);
            let phis: Vec<f64> = (0..s - 1).map(|_| rng.random_range(0.1..3.0)).collect();
            for k in 0..t {
                let f = (1u64 << k.min(62)) as f64;
                // controlled exp(-i f θ X^{⊗s}): H all, CNOT parity onto last
                for &q in &sys {
                    c.h(q);
                }
                for w in sys.windows(2) {
                    c.cnot(w[0], w[1]);
                }
                crz(&mut c, k, sys[s - 1], 2.0 * f * theta);
                for w in sys.windows(2).rev() {
                    c.cnot(w[0], w[1]);
                }
                for &q in &sys {
                    c.h(q);
                }
                for (i, w) in sys.windows(2).enumerate() {
                    c.cnot(w[0], w[1]);
                    crz(&mut c, k, w[1], 2.0 * f * phis[i]);
                    c.cnot(w[0], w[1]);
                }
            }
        }
        "trotter" => {
            for k in 0..t {
                for _ in 0..(1usize << k) {
                    ising_step(&mut c, &sys, 0.3, 0.3, Some(k));
                }
            }
        }
        _ => return Err(format!("unknown qpe kind {kind:?}")),
    }
    qft(&mut c, &(0..t).collect::<Vec<_>>(), 0, true, true);
    Ok(c)
}

/// Multi-controlled X (`ctrls` -> `t`) with a Toffoli ladder into `anc`
/// (`ctrls.len() - 2` clean ancillas).
pub fn mcx(c: &mut Circuit, ctrls: &[usize], t: usize, anc: &[usize]) {
    match ctrls.len() {
        0 => {
            c.x(t);
        }
        1 => {
            c.cnot(ctrls[0], t);
        }
        2 => {
            c.ccx(ctrls[0], ctrls[1], t);
        }
        k => {
            c.ccx(ctrls[0], ctrls[1], anc[0]);
            for i in 2..k - 1 {
                c.ccx(anc[i - 2], ctrls[i], anc[i - 1]);
            }
            c.ccx(anc[k - 3], ctrls[k - 1], t);
            for i in (2..k - 1).rev() {
                c.ccx(anc[i - 2], ctrls[i], anc[i - 1]);
            }
            c.ccx(ctrls[0], ctrls[1], anc[0]);
        }
    }
}

/// Coined discrete-time walk on a cycle of `2^m` sites: coin qubit 0,
/// position `1..=m`, ladder ancillas after. Step: H(coin); if coin=1
/// increment else decrement (controlled ±1 by multi-controlled X ladders).
fn walk(m: usize, steps: usize) -> Circuit {
    let coin = 0;
    let pos: Vec<usize> = (1..=m).collect();
    let anc: Vec<usize> = (m + 1..m + 1 + m.saturating_sub(1)).collect();
    let mut c = Circuit::new(m + 1 + anc.len());
    let inc = |c: &mut Circuit| {
        // controlled increment: bit i flips iff coin and all lower bits are 1
        for i in (0..m).rev() {
            let mut ctrls = vec![coin];
            ctrls.extend_from_slice(&pos[..i]);
            mcx(c, &ctrls, pos[i], &anc);
        }
    };
    for _ in 0..steps {
        c.h(coin);
        inc(&mut c);
        // decrement when coin = 0: X coin, inverse increment, X coin
        c.x(coin);
        let mut d = Circuit::new(c.num_qubits);
        inc(&mut d);
        c.append(&d.inverse());
        c.x(coin);
    }
    c
}

/// Uniformly controlled `Ry` ("multiplexor"): `target` gets `Ry(α_v)` when
/// the controls read `v` (`ctrls[0]` = LSB). Gray-code decomposition
/// (Möttönen et al. 2004): `2^k` `Ry` and `2^k` CNOTs.
pub fn multiplexed_ry(c: &mut Circuit, ctrls: &[usize], target: usize, alpha: &[f64]) {
    let k = ctrls.len();
    let len = 1usize << k;
    assert_eq!(alpha.len(), len);
    if k == 0 {
        c.ry(target, alpha[0]);
        return;
    }
    let gray = |i: usize| i ^ (i >> 1);
    // θ_i = 2^{-k} Σ_v (-1)^{popcount(v & gray(i))} α_v
    let theta: Vec<f64> = (0..len)
        .map(|i| {
            let g = gray(i);
            alpha
                .iter()
                .enumerate()
                .map(|(v, &a)| if (v & g).count_ones() % 2 == 1 { -a } else { a })
                .sum::<f64>()
                / len as f64
        })
        .collect();
    for i in 0..len {
        c.ry(target, theta[i]);
        let changed = gray(i) ^ gray((i + 1) % len);
        let bit = changed.trailing_zeros() as usize;
        c.cnot(ctrls[bit], target);
    }
}

/// Toy HHL: clock `t` qubits (0..t), system `m` qubits, ancilla last.
/// `A = H^{⊗m} diag(λ) H^{⊗m}` with `λ(x) = x + 1` (exact in the clock if
/// `2^m < 2^t`), `|b> = |0^m>`, evolution `e^{2πi A k / 2^t}`, exact
/// eigenvalue inversion `Ry(2 arcsin(1/λ))` by a multiplexor, then the
/// inverse phase estimation.
fn hhl(t: usize, m: usize) -> Result<Circuit, String> {
    if m >= t {
        return Err("hhl needs m < t".into());
    }
    let sys: Vec<usize> = (t..t + m).collect();
    let anc = t + m;
    let mut c = Circuit::new(t + m + 1);
    let mut pe = Circuit::new(t + m + 1);
    for k in 0..t {
        pe.h(k);
    }
    for &q in &sys {
        pe.h(q);
    }
    let base = 2.0 * PI / 2f64.powi(t as i32);
    for k in 0..t {
        // controlled e^{i α λ(x)}, α = base·2^k, λ = 1 + Σ 2^i x_i
        let alpha = base * 2f64.powi(k as i32);
        pe.phase(k, alpha);
        for (i, &q) in sys.iter().enumerate() {
            pe.cphase(k, q, alpha * 2f64.powi(i as i32));
        }
    }
    qft(&mut pe, &(0..t).collect::<Vec<_>>(), 0, true, true);
    c.append(&pe);
    // clock now holds λ (qubit 0 = LSB)
    let clock: Vec<usize> = (0..t).collect();
    let alpha: Vec<f64> = (0..1usize << t)
        .map(|v| {
            let lam = v as f64;
            if lam >= 1.0 {
                2.0 * (1.0 / lam).asin()
            } else {
                0.0
            }
        })
        .collect();
    multiplexed_ry(&mut c, &clock, anc, &alpha);
    c.append(&pe.inverse());
    Ok(c)
}

pub fn reverse_bits(v: usize, k: usize) -> usize {
    (0..k).fold(0, |acc, i| acc | (((v >> i) & 1) << (k - 1 - i)))
}

/// Random Clifford+T (simulability's `ct` family, NN brickwork).
fn rct(n: usize, layers: usize, t: usize, rng: &mut StdRng) -> Circuit {
    let mut c = Circuit::new(n);
    let total = layers * n;
    let mut slots: Vec<usize> = (0..t).map(|_| rng.random_range(0..total.max(1))).collect();
    slots.sort_unstable();
    let mut si = 0;
    for l in 0..layers {
        for q in 0..n {
            match rng.random_range(0..4) {
                0 => {}
                1 => {
                    c.h(q);
                }
                2 => {
                    c.s(q);
                }
                _ => {
                    c.h(q).s(q);
                }
            }
            while si < slots.len() && slots[si] == l * n + q {
                c.t(q).h(q);
                si += 1;
            }
        }
        for q in ((l % 2)..n.saturating_sub(1)).step_by(2) {
            c.cnot(q, q + 1);
        }
    }
    c
}
