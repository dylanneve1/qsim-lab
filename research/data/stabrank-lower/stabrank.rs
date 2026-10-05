// Exact small-n stabilizer-rank lower bounds by exhaustive search plus "plateau gluing".
// Standalone, std only.  Build: rustc -O -C target-cpu=native stabrank.rs -o stabrank
// Run:   ./stabrank <cmd> ...   (see main()).
//
// Conventions: an n-qubit vector has index x in [0, 2^n); qubit j is bit j of x.  "Restricting
// the last qubit" (qubit n-1, the most significant bit) to |0>/|1> gives the first/second half of
// the vector.  Stabilizer states are stored canonically: normalised, and the amplitude at the
// smallest support point is real positive.
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::io::Write;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct C {
    pub re: f64,
    pub im: f64,
}
impl C {
    pub fn new(re: f64, im: f64) -> C {
        C { re, im }
    }
    fn conj(self) -> C {
        C::new(self.re, -self.im)
    }
    pub fn n2(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    pub fn abs(self) -> f64 {
        self.n2().sqrt()
    }
    fn scale(self, s: f64) -> C {
        C::new(self.re * s, self.im * s)
    }
    fn div(self, o: C) -> C {
        let d = o.n2();
        let p = self * o.conj();
        C::new(p.re / d, p.im / d)
    }
    fn expi(t: f64) -> C {
        C::new(t.cos(), t.sin())
    }
}
impl std::ops::Add for C {
    type Output = C;
    fn add(self, o: C) -> C {
        C::new(self.re + o.re, self.im + o.im)
    }
}
impl std::ops::Sub for C {
    type Output = C;
    fn sub(self, o: C) -> C {
        C::new(self.re - o.re, self.im - o.im)
    }
}
impl std::ops::Mul for C {
    type Output = C;
    fn mul(self, o: C) -> C {
        C::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}
impl std::ops::Neg for C {
    type Output = C;
    fn neg(self) -> C {
        C::new(-self.re, -self.im)
    }
}

pub fn dot(a: &[C], b: &[C]) -> C {
    // <a|b>
    let mut s = C::default();
    for i in 0..a.len() {
        s = s + a[i].conj() * b[i];
    }
    s
}
pub fn norm2(a: &[C]) -> f64 {
    a.iter().map(|z| z.n2()).sum()
}

const TOL: f64 = 1e-9;

pub fn canonical(v: &[C]) -> Vec<C> {
    let nrm = norm2(v).sqrt();
    let first = v.iter().find(|z| z.abs() > 1e-7 * nrm).copied().unwrap();
    let ph = first.conj().scale(1.0 / (first.abs() * nrm));
    v.iter().map(|&z| z * ph).collect()
}
pub fn key_of(v: &[C]) -> Vec<i64> {
    let mut k = Vec::with_capacity(2 * v.len());
    for z in v {
        k.push((z.re * 1e6).round() as i64);
        k.push((z.im * 1e6).round() as i64);
    }
    k
}

// ---------------------------------------------------------------- stabilizer states
pub fn subspaces(n: usize) -> Vec<Vec<usize>> {
    // all linear subspaces of F_2^n, each as a basis
    let mut seen: HashSet<u64> = HashSet::new();
    let mut out = vec![];
    let mut frontier: Vec<(u64, Vec<usize>)> = vec![(1u64, vec![])];
    seen.insert(1);
    while let Some((set, basis)) = frontier.pop() {
        out.push(basis.clone());
        for v in 0..(1usize << n) {
            if set >> v & 1 == 1 {
                continue;
            }
            let mut ns = set;
            for x in 0..(1usize << n) {
                if set >> x & 1 == 1 {
                    ns |= 1u64 << (x ^ v);
                }
            }
            if seen.insert(ns) {
                let mut b = basis.clone();
                b.push(v);
                frontier.push((ns, b));
            }
        }
    }
    out
}

pub fn enum_states(n: usize) -> Vec<Vec<C>> {
    let dim = 1usize << n;
    let mut out = vec![];
    for basis in subspaces(n) {
        let k = basis.len();
        // span
        let mut span = vec![0usize; 1 << k];
        for y in 0..(1usize << k) {
            let mut x = 0;
            for j in 0..k {
                if y >> j & 1 == 1 {
                    x ^= basis[j];
                }
            }
            span[y] = x;
        }
        let amp = (2f64).powf(-(k as f64) / 2.0);
        let npairs = k * (k.saturating_sub(1)) / 2;
        for x0 in 0..dim {
            if span.iter().any(|&s| (s ^ x0) < x0) {
                continue; // x0 must be the minimum of its coset
            }
            for l in 0..(1usize << k) {
                for ql in 0..(1usize << k) {
                    for qq in 0..(1usize << npairs) {
                        let mut v = vec![C::default(); dim];
                        for y in 0..(1usize << k) {
                            let mut f = 0usize; // Z4 phase exponent
                            for j in 0..k {
                                if y >> j & 1 == 1 {
                                    f += (l >> j & 1) + 2 * (ql >> j & 1);
                                }
                            }
                            let mut p = 0;
                            for a in 0..k {
                                for b in (a + 1)..k {
                                    if (y >> a & 1 == 1) && (y >> b & 1 == 1) && (qq >> p & 1 == 1)
                                    {
                                        f += 2;
                                    }
                                    p += 1;
                                }
                            }
                            let ph = match f % 4 {
                                0 => C::new(1.0, 0.0),
                                1 => C::new(0.0, 1.0),
                                2 => C::new(-1.0, 0.0),
                                _ => C::new(0.0, -1.0),
                            };
                            v[x0 ^ span[y]] = ph.scale(amp);
                        }
                        out.push(v);
                    }
                }
            }
        }
    }
    out
}

pub fn n_stab(n: usize) -> u64 {
    let mut c = 1u64 << n;
    for k in 1..=n {
        c *= (1u64 << k) + 1;
    }
    c
}

/// Exact-up-to-tolerance test: is v (any normalisation) a stabilizer state?
pub fn is_stabilizer(v: &[C]) -> bool {
    let nrm = norm2(v).sqrt();
    if nrm < TOL {
        return false;
    }
    let supp: Vec<usize> = (0..v.len()).filter(|&x| v[x].abs() > 1e-7 * nrm).collect();
    let s = supp.len();
    if !s.is_power_of_two() {
        return false;
    }
    let x0 = supp[0];
    let inset: HashSet<usize> = supp.iter().copied().collect();
    // basis of directions
    let mut basis: Vec<usize> = vec![];
    let mut spanset: Vec<usize> = vec![0];
    for &x in &supp {
        let d = x ^ x0;
        if !spanset.contains(&d) {
            basis.push(d);
            let cur = spanset.clone();
            for c in cur {
                spanset.push(c ^ d);
            }
        }
    }
    if spanset.len() != s {
        return false;
    }
    for d in &spanset {
        if !inset.contains(&(x0 ^ d)) {
            return false;
        }
    }
    let a0 = v[x0];
    let k = basis.len();
    // phases in Z4
    let mut f = vec![0usize; 1 << k];
    for y in 0..(1usize << k) {
        let mut x = x0;
        for j in 0..k {
            if y >> j & 1 == 1 {
                x ^= basis[j];
            }
        }
        let r = v[x].div(a0);
        if (r.abs() - 1.0).abs() > 1e-6 {
            return false;
        }
        let e = if (r - C::new(1.0, 0.0)).abs() < 1e-6 {
            0
        } else if (r - C::new(0.0, 1.0)).abs() < 1e-6 {
            1
        } else if (r - C::new(-1.0, 0.0)).abs() < 1e-6 {
            2
        } else if (r - C::new(0.0, -1.0)).abs() < 1e-6 {
            3
        } else {
            return false;
        };
        f[y] = e;
    }
    // fit f(y) = sum_j c_j y_j + 2 sum_{a<b} q_ab y_a y_b  (mod 4), c_j in Z4
    let c: Vec<usize> = (0..k).map(|j| f[1 << j]).collect();
    let mut q = vec![vec![0usize; k]; k];
    for a in 0..k {
        for b in (a + 1)..k {
            let d = (f[(1 << a) | (1 << b)] + 8 - c[a] - c[b]) % 4;
            if d % 2 != 0 {
                return false;
            }
            q[a][b] = d / 2;
        }
    }
    for y in 0..(1usize << k) {
        let mut g = 0usize;
        for j in 0..k {
            if y >> j & 1 == 1 {
                g += c[j];
                for b in (j + 1)..k {
                    if y >> b & 1 == 1 {
                        g += 2 * q[j][b];
                    }
                }
            }
        }
        if g % 4 != f[y] {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------- single-qubit Cliffords
pub type M2 = [C; 4]; // row-major
fn m2mul(a: &M2, b: &M2) -> M2 {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    ]
}
fn m2canon(a: &M2) -> M2 {
    let f = a.iter().find(|z| z.abs() > 1e-9).copied().unwrap();
    let ph = f.conj().scale(1.0 / f.abs());
    [a[0] * ph, a[1] * ph, a[2] * ph, a[3] * ph]
}
pub fn cliffords1() -> Vec<M2> {
    let s = 1.0 / 2f64.sqrt();
    let h: M2 = [C::new(s, 0.), C::new(s, 0.), C::new(s, 0.), C::new(-s, 0.)];
    let ph: M2 = [C::new(1., 0.), C::default(), C::default(), C::new(0., 1.)];
    let id: M2 = [C::new(1., 0.), C::default(), C::default(), C::new(1., 0.)];
    let mut out: Vec<M2> = vec![id];
    let mut keys: HashSet<Vec<i64>> = HashSet::new();
    keys.insert(key_of(&id));
    let mut i = 0;
    while i < out.len() {
        for g in [&h, &ph] {
            let m = m2canon(&m2mul(g, &out[i]));
            if keys.insert(key_of(&m)) {
                out.push(m);
            }
        }
        i += 1;
    }
    assert_eq!(out.len(), 24);
    out
}
pub fn apply1(m: &M2, v: &[C]) -> [C; 2] {
    [m[0] * v[0] + m[1] * v[1], m[2] * v[0] + m[3] * v[1]]
}
pub fn parallel(a: &[C], b: &[C]) -> bool {
    let d = dot(a, b);
    (d.n2() - norm2(a) * norm2(b)).abs() < 1e-9
}

// ---------------------------------------------------------------- targets
pub fn psi1(kind: &str) -> [C; 2] {
    match kind {
        "H" => [C::new((PI / 8.).cos(), 0.), C::new((PI / 8.).sin(), 0.)],
        // face state: Bloch vector (1,1,1)/sqrt3
        "F" => {
            let beta = 0.5 * (1.0 / 3f64.sqrt()).acos();
            [C::new(beta.cos(), 0.), C::expi(PI / 4.).scale(beta.sin())]
        }
        _ => panic!("kind"),
    }
}
/// Non-product targets (permutation-symmetric, real): "W" and "D<k>" (Dicke with weight k).
pub fn special_target(kind: &str, n: usize) -> Option<Vec<C>> {
    let w: Option<u32> = if kind == "W" {
        Some(1)
    } else if let Some(r) = kind.strip_prefix('D') {
        r.parse().ok()
    } else {
        None
    };
    let w = w?;
    let v: Vec<C> = (0..(1usize << n))
        .map(|x| {
            if x.count_ones() == w {
                C::new(1., 0.)
            } else {
                C::default()
            }
        })
        .collect();
    let nr = norm2(&v).sqrt();
    Some(v.iter().map(|z| z.scale(1.0 / nr)).collect())
}
pub fn perm_conj_generators(t: &Table) -> Vec<Vec<u32>> {
    let n = t.n;
    let mut maps: Vec<Box<dyn Fn(&[C]) -> Vec<C>>> =
        vec![Box::new(|v: &[C]| v.iter().map(|z| z.conj()).collect())];
    if n >= 2 {
        let mut sw: Vec<usize> = (0..n).collect();
        sw.swap(0, 1);
        maps.push(Box::new(move |v: &[C]| permute_qubits(&sw, v)));
        let cyc: Vec<usize> = (0..n).map(|j| (j + 1) % n).collect();
        maps.push(Box::new(move |v: &[C]| permute_qubits(&cyc, v)));
    }
    maps.iter()
        .map(|f| t.states.iter().map(|s| t.lookup(&f(s)).unwrap()).collect())
        .collect()
}
pub fn tensor_power(p: &[C; 2], n: usize) -> Vec<C> {
    (0..(1usize << n))
        .map(|x| {
            let mut a = C::new(1., 0.);
            for j in 0..n {
                a = a * p[(x >> j) & 1];
            }
            a
        })
        .collect()
}

// ---------------------------------------------------------------- symmetry group
pub fn apply_local(m: &M2, q: usize, v: &[C]) -> Vec<C> {
    let mut o = v.to_vec();
    for x in 0..v.len() {
        if x >> q & 1 == 0 {
            let y = x | (1 << q);
            let r = apply1(m, &[v[x], v[y]]);
            o[x] = r[0];
            o[y] = r[1];
        }
    }
    o
}
fn permute_qubits(perm: &[usize], v: &[C]) -> Vec<C> {
    // qubit j -> perm[j]
    let mut o = vec![C::default(); v.len()];
    for x in 0..v.len() {
        let mut y = 0;
        for j in 0..perm.len() {
            if x >> j & 1 == 1 {
                y |= 1 << perm[j];
            }
        }
        o[y] = v[x];
    }
    o
}

pub struct Table {
    pub n: usize,
    pub states: Vec<Vec<C>>,
    pub index: HashMap<Vec<i64>, u32>,
}
impl Table {
    pub fn new(n: usize) -> Table {
        let states = enum_states(n);
        assert_eq!(states.len() as u64, n_stab(n));
        let mut index = HashMap::new();
        for (i, s) in states.iter().enumerate() {
            let c = canonical(s);
            assert!(index.insert(key_of(&c), i as u32).is_none());
        }
        Table { n, states, index }
    }
    /// Only the 6^n product stabilizer states (for the product-stabilizer-rank model).
    pub fn products(n: usize) -> Table {
        let s = 1.0 / 2f64.sqrt();
        let one = [
            [C::new(1., 0.), C::default()],
            [C::default(), C::new(1., 0.)],
            [C::new(s, 0.), C::new(s, 0.)],
            [C::new(s, 0.), C::new(-s, 0.)],
            [C::new(s, 0.), C::new(0., s)],
            [C::new(s, 0.), C::new(0., -s)],
        ];
        let mut states = vec![];
        let mut index = HashMap::new();
        for code in 0..6usize.pow(n as u32) {
            let mut v = vec![C::new(1., 0.)];
            let mut c = code;
            for _ in 0..n {
                let f = one[c % 6];
                c /= 6;
                // new qubit becomes the most significant bit
                let mut w = vec![C::default(); v.len() * 2];
                for x in 0..v.len() {
                    w[x] = v[x] * f[0];
                    w[x + v.len()] = v[x] * f[1];
                }
                v = w;
            }
            index.insert(key_of(&canonical(&v)), states.len() as u32);
            states.push(canonical(&v));
        }
        Table { n, states, index }
    }
    pub fn lookup(&self, v: &[C]) -> Option<u32> {
        self.index.get(&key_of(&canonical(v))).copied()
    }
}

/// Generators of the symmetry group of psi1^{⊗n} acting on stabilizer states, as permutations.
pub fn sym_generators(t: &Table, p: &[C; 2]) -> Vec<Vec<u32>> {
    let n = t.n;
    let mut gens: Vec<Vec<u32>> = vec![];
    let cl = cliffords1();
    let mut maps: Vec<Box<dyn Fn(&[C]) -> Vec<C>>> = vec![];
    for m in &cl {
        let im = apply1(m, p);
        if parallel(&im, p) && (m[1].abs() > 1e-9 || (m[3] - m[0]).abs() > 1e-9) {
            let m = *m;
            maps.push(Box::new(move |v: &[C]| apply_local(&m, 0, v)));
        }
        // antiunitary: v -> (m^{⊗n}) conj(v)
        let pc = [p[0].conj(), p[1].conj()];
        let im2 = apply1(m, &pc);
        if parallel(&im2, p) {
            let m = *m;
            maps.push(Box::new(move |v: &[C]| {
                let mut w: Vec<C> = v.iter().map(|z| z.conj()).collect();
                for q in 0..n {
                    w = apply_local(&m, q, &w);
                }
                w
            }));
        }
    }
    if n >= 2 {
        let mut sw: Vec<usize> = (0..n).collect();
        sw.swap(0, 1);
        maps.push(Box::new(move |v: &[C]| permute_qubits(&sw, v)));
        let cyc: Vec<usize> = (0..n).map(|j| (j + 1) % n).collect();
        maps.push(Box::new(move |v: &[C]| permute_qubits(&cyc, v)));
    }
    // sanity: each map fixes psi^{⊗n} projectively
    let psi = tensor_power(p, n);
    for f in &maps {
        assert!(parallel(&f(&psi), &psi));
        let perm: Vec<u32> = t
            .states
            .iter()
            .map(|s| t.lookup(&f(s)).expect("not a stabilizer state"))
            .collect();
        gens.push(perm);
    }
    gens
}

pub fn orbits(nst: usize, gens: &[Vec<u32>]) -> Vec<u32> {
    // orbit id per state, orbits numbered by ascending size (ties: by min element)
    let mut parent: Vec<u32> = (0..nst as u32).collect();
    fn find(p: &mut Vec<u32>, x: u32) -> u32 {
        let mut r = x;
        while p[r as usize] != r {
            r = p[r as usize];
        }
        let mut y = x;
        while p[y as usize] != r {
            let nx = p[y as usize];
            p[y as usize] = r;
            y = nx;
        }
        r
    }
    for g in gens {
        for i in 0..nst {
            let a = find(&mut parent, i as u32);
            let b = find(&mut parent, g[i]);
            if a != b {
                parent[a.max(b) as usize] = a.min(b);
            }
        }
    }
    let roots: Vec<u32> = (0..nst as u32).map(|i| find(&mut parent, i)).collect();
    let mut size: HashMap<u32, usize> = HashMap::new();
    for &r in &roots {
        *size.entry(r).or_default() += 1;
    }
    let mut rs: Vec<(usize, u32)> = size.iter().map(|(&r, &s)| (s, r)).collect();
    rs.sort();
    let id: HashMap<u32, u32> = rs
        .iter()
        .enumerate()
        .map(|(i, &(_, r))| (r, i as u32))
        .collect();
    roots.iter().map(|r| id[r]).collect()
}

pub fn group_closure(gens: &[Vec<u32>], cap: usize) -> Option<Vec<Vec<u16>>> {
    let nst = gens[0].len();
    if nst > 65535 {
        return None;
    }
    let idp: Vec<u16> = (0..nst as u16).collect();
    let mut seen: HashSet<Vec<u16>> = HashSet::new();
    seen.insert(idp.clone());
    let mut out = vec![idp];
    let mut i = 0;
    while i < out.len() {
        for g in gens {
            let h: Vec<u16> = out[i].iter().map(|&x| g[x as usize] as u16).collect();
            if !seen.contains(&h) {
                seen.insert(h.clone());
                out.push(h);
                if out.len() > cap {
                    return None;
                }
            }
        }
        i += 1;
    }
    Some(out)
}

// ---------------------------------------------------------------- linear algebra
/// Least squares: coefficients c minimising |psi - sum c_i v_i|; returns (c, residual norm).
pub fn lsq(vs: &[&[C]], psi: &[C]) -> (Vec<C>, f64) {
    let k = vs.len();
    let mut a = vec![vec![C::default(); k + 1]; k];
    for i in 0..k {
        for j in 0..k {
            a[i][j] = dot(vs[i], vs[j]);
        }
        a[i][k] = dot(vs[i], psi);
    }
    // gaussian elimination with partial pivoting
    for col in 0..k {
        let piv = (col..k)
            .max_by(|&x, &y| a[x][col].abs().partial_cmp(&a[y][col].abs()).unwrap())
            .unwrap();
        a.swap(col, piv);
        let d = a[col][col];
        if d.abs() < 1e-14 {
            return (vec![], f64::INFINITY);
        }
        for r in 0..k {
            if r != col {
                let f = a[r][col].div(d);
                for cc in col..=k {
                    let t = a[col][cc];
                    a[r][cc] = a[r][cc] - f * t;
                }
            }
        }
    }
    let c: Vec<C> = (0..k).map(|i| a[i][k].div(a[i][i])).collect();
    let mut r = psi.to_vec();
    for i in 0..k {
        for x in 0..r.len() {
            r[x] = r[x] - c[i] * vs[i][x];
        }
    }
    (c, norm2(&r).sqrt())
}
pub fn gram_det(vs: &[&[C]]) -> f64 {
    // |det Gram| via elimination
    let k = vs.len();
    let mut a = vec![vec![C::default(); k]; k];
    for i in 0..k {
        for j in 0..k {
            a[i][j] = dot(vs[i], vs[j]);
        }
    }
    let mut det = 1.0;
    for col in 0..k {
        let piv = (col..k)
            .max_by(|&x, &y| a[x][col].abs().partial_cmp(&a[y][col].abs()).unwrap())
            .unwrap();
        a.swap(col, piv);
        let d = a[col][col];
        det *= d.abs();
        if d.abs() < 1e-15 {
            return 0.0;
        }
        for r in (col + 1)..k {
            let f = a[r][col].div(d);
            for cc in col..k {
                let t = a[col][cc];
                a[r][cc] = a[r][cc] - f * t;
            }
        }
    }
    det
}

/// A decomposition: sorted state indices with coefficients for the normalised target.
#[derive(Clone, Debug)]
pub struct Dec {
    pub idx: Vec<u32>,
    pub coef: Vec<C>,
}

/// Check a candidate set; returns the decomposition if psi is in the span, all coefficients
/// nonzero and the set independent.
pub fn check_set(t: &Table, psi: &[C], set: &[u32]) -> Option<Dec> {
    let mut s = set.to_vec();
    s.sort();
    s.dedup();
    if s.len() != set.len() {
        return None;
    }
    let vs: Vec<&[C]> = s.iter().map(|&i| t.states[i as usize].as_slice()).collect();
    if gram_det(&vs) < 1e-10 {
        return None;
    }
    let (c, res) = lsq(&vs, psi);
    if res > 1e-8 || c.iter().any(|z| z.abs() < 1e-8) {
        return None;
    }
    Some(Dec { idx: s, coef: c })
}

// ---------------------------------------------------------------- exhaustive search
pub struct SearchStats {
    pub w_count: u64,
    pub cand_count: u64,
    pub verified: u64,
}

/// Find all independent k-sets of stabilizer states (k >= 2) whose span contains psi
/// (normalised), i.e. all rank-k decompositions with no vanishing coefficient, up to the symmetry
/// group; returns orbit representatives (possibly with repeats) found in canonical (min-orbit-index
/// first) form.  Completeness argument: see research/theory/stabrank-lower.md, "Search".
pub fn search(
    t: &Table,
    psi: &[C],
    k: usize,
    orbit: &[u32],
    group: Option<&Vec<Vec<u16>>>,
    st: &mut SearchStats,
) -> Vec<Dec> {
    let nst = t.states.len();
    let dim = psi.len();
    let mut found = vec![];
    let norbits = *orbit.iter().max().unwrap() as usize + 1;
    let mut rep = vec![u32::MAX; norbits];
    for i in 0..nst {
        let o = orbit[i] as usize;
        if rep[o] == u32::MAX {
            rep[o] = i as u32;
        }
    }
    // fixed random functionals for the projective hash
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let f1: Vec<C> = (0..dim).map(|_| C::new(rnd(), rnd())).collect();
    let f2: Vec<C> = (0..dim).map(|_| C::new(rnd(), rnd())).collect();
    // per-state <f|v>
    let fv1: Vec<C> = t.states.iter().map(|v| dot(&f1, v)).collect();
    let fv2: Vec<C> = t.states.iter().map(|v| dot(&f2, v)).collect();
    // orthonormal basis q_0 = psi
    let pn = norm2(psi).sqrt();
    let q0: Vec<C> = psi.iter().map(|z| z.scale(1.0 / pn)).collect();
    let ip0: Vec<C> = t.states.iter().map(|v| dot(&q0, v)).collect();

    // recursive: chosen = indices so far; qs = orthonormal basis; ips[j][v] = <q_j|v>
    struct Ctx<'a> {
        t: &'a Table,
        psi: &'a [C],
        k: usize,
        orbit: &'a [u32],
        fv1: Vec<C>,
        fv2: Vec<C>,
        f1: Vec<C>,
        f2: Vec<C>,
    }
    fn rec(
        cx: &Ctx,
        chosen: &mut Vec<u32>,
        qs: &mut Vec<Vec<C>>,
        ips: &mut Vec<Vec<C>>,
        minidx: u32,
        allowed: &dyn Fn(usize, u32) -> bool,
        found: &mut Vec<Dec>,
        st: &mut SearchStats,
    ) {
        let nst = cx.t.states.len();
        if chosen.len() + 2 == cx.k {
            // hash step: candidates v with orbit >= minidx, not chosen
            st.w_count += 1;
            // <f|q_j>
            let fq1: Vec<C> = qs.iter().map(|q| dot(&cx.f1, q)).collect();
            let fq2: Vec<C> = qs.iter().map(|q| dot(&cx.f2, q)).collect();
            let mut pts: Vec<(f64, f64, f64, u32)> = Vec::new();
            for v in 0..nst {
                if cx.orbit[v] < minidx || chosen.contains(&(v as u32)) {
                    continue;
                }
                st.cand_count += 1;
                let mut pn2 = 1.0;
                let mut a = cx.fv1[v];
                let mut b = cx.fv2[v];
                for j in 0..qs.len() {
                    let c = ips[j][v];
                    pn2 -= c.n2();
                    a = a - fq1[j] * c;
                    b = b - fq2[j] * c;
                }
                if pn2 < 1e-9 {
                    continue; // v in W: dependent set or smaller decomposition
                }
                let s = a.n2() + b.n2();
                let ab = a * b.conj();
                pts.push((
                    2.0 * ab.re / s,
                    2.0 * ab.im / s,
                    (a.n2() - b.n2()) / s,
                    v as u32,
                ));
            }
            pts.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
            let eps = 1e-6;
            for i in 0..pts.len() {
                let mut j = i + 1;
                while j < pts.len() && pts[j].0 - pts[i].0 < eps {
                    let d = (pts[j].1 - pts[i].1).abs() + (pts[j].2 - pts[i].2).abs();
                    if d < 2.0 * eps {
                        let mut set = chosen.clone();
                        set.push(pts[i].3);
                        set.push(pts[j].3);
                        st.verified += 1;
                        if let Some(dc) = check_set(cx.t, cx.psi, &set) {
                            found.push(dc);
                        }
                    }
                    j += 1;
                }
            }
            return;
        }
        // choose next element
        let level = chosen.len();
        for v in 0..nst {
            let o = cx.orbit[v];
            if o < minidx || chosen.contains(&(v as u32)) || !allowed(level, v as u32) {
                continue;
            }
            // Gram-Schmidt
            let mut w = cx.t.states[v].clone();
            for j in 0..qs.len() {
                let c = ips[j][v];
                for x in 0..w.len() {
                    w[x] = w[x] - c * qs[j][x];
                }
            }
            let wn = norm2(&w).sqrt();
            if wn < 1e-6 {
                continue;
            }
            for z in w.iter_mut() {
                *z = z.scale(1.0 / wn);
            }
            let ipw: Vec<C> = cx.t.states.iter().map(|s| dot(&w, s)).collect();
            chosen.push(v as u32);
            qs.push(w);
            ips.push(ipw);
            rec(cx, chosen, qs, ips, o, allowed, found, st);
            chosen.pop();
            qs.pop();
            ips.pop();
        }
    }
    let cx = Ctx {
        t,
        psi,
        k,
        orbit,
        fv1,
        fv2,
        f1,
        f2,
    };
    assert!(k >= 2);
    if k == 2 {
        // psi in span(a,b): b in span(psi,a)
        // simple direct: for each a, project all onto span(psi,a)^perp, zero ones are partners
        for a in 0..nst {
            let set_a = &t.states[a];
            let mut q1 = set_a.clone();
            let c = ip0[a];
            for x in 0..dim {
                q1[x] = q1[x] - c * q0[x];
            }
            let n1 = norm2(&q1).sqrt();
            if n1 < 1e-6 {
                continue;
            }
            for z in q1.iter_mut() {
                *z = z.scale(1.0 / n1);
            }
            for b in (a + 1)..nst {
                let c0 = ip0[b];
                let c1 = dot(&q1, &t.states[b]);
                if 1.0 - c0.n2() - c1.n2() < 1e-9 {
                    if let Some(dc) = check_set(t, psi, &[a as u32, b as u32]) {
                        found.push(dc);
                    }
                }
            }
        }
        return found;
    }
    let stabs: Vec<Vec<usize>> = match group {
        Some(g) => (0..norbits)
            .map(|o| {
                (0..g.len())
                    .filter(|&e| g[e][rep[o] as usize] as u32 == rep[o])
                    .collect()
            })
            .collect(),
        None => vec![],
    };
    let nthreads: usize = std::env::var("THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = std::sync::Mutex::new((found, 0u64, 0u64, 0u64));
    std::thread::scope(|sc| {
        for _ in 0..nthreads {
            sc.spawn(|| {
                let mut lst = SearchStats {
                    w_count: 0,
                    cand_count: 0,
                    verified: 0,
                };
                let mut lfound = vec![];
                loop {
                    let o = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if o >= norbits {
                        break;
                    }
                    let r1 = rep[o] as usize;
                    let mut chosen = vec![];
                    let mut qs = vec![q0.clone()];
                    let mut ips = vec![ip0.clone()];
                    // level 0 must be the orbit representative r1
                    let stab_o = if group.is_some() {
                        Some(&stabs[o])
                    } else {
                        None
                    };
                    let allowed = |level: usize, v: u32| -> bool {
                        if level == 0 {
                            return v as usize == r1;
                        }
                        if level == 1 {
                            if let (Some(g), Some(sb)) = (group, stab_o) {
                                // v must be minimal in its Stab(r1)-orbit
                                for &e in sb.iter() {
                                    if (g[e][v as usize] as u32) < v {
                                        return false;
                                    }
                                }
                            }
                        }
                        true
                    };
                    let w0 = lst.w_count;
                    rec(
                        &cx,
                        &mut chosen,
                        &mut qs,
                        &mut ips,
                        o as u32,
                        &allowed,
                        &mut lfound,
                        &mut lst,
                    );
                    if std::env::var("PROGRESS").is_ok() {
                        eprintln!(
                            "orbit {}/{} W={} found_so_far={}",
                            o,
                            norbits,
                            lst.w_count - w0,
                            lfound.len()
                        );
                    }
                }
                let mut r = results.lock().unwrap();
                r.0.extend(lfound);
                r.1 += lst.w_count;
                r.2 += lst.cand_count;
                r.3 += lst.verified;
            });
        }
    });
    let r = results.into_inner().unwrap();
    st.w_count += r.1;
    st.cand_count += r.2;
    st.verified += r.3;
    r.0
}

/// Expand representatives to the full set of decompositions: orbit of each set under the group
/// generated by `gens` (BFS on sets, so the group itself need not be stored).
pub fn expand(t: &Table, psi: &[C], reps: &[Dec], gens: &[Vec<u32>]) -> Vec<Dec> {
    let mut seen: HashSet<Vec<u32>> = HashSet::new();
    let mut out = vec![];
    for d in reps {
        let mut s0 = d.idx.clone();
        s0.sort();
        if !seen.insert(s0.clone()) {
            continue;
        }
        let mut stack = vec![s0];
        while let Some(s) = stack.pop() {
            out.push(
                check_set(t, psi, &s).expect("image of a decomposition must be a decomposition"),
            );
            for g in gens {
                let mut im: Vec<u32> = s.iter().map(|&i| g[i as usize]).collect();
                im.sort();
                if seen.insert(im.clone()) {
                    stack.push(im);
                }
            }
        }
    }
    out
}

/// l1 norms of the coefficient vectors of decompositions (normalised target, normalised terms).
pub fn l1_norms(ds: &[Dec]) -> Vec<f64> {
    ds.iter()
        .map(|d| d.coef.iter().map(|z| z.abs()).sum())
        .collect()
}
/// The l1-ratio criterion (research/theory/stabrank-lower.md, Cor. 7): returns true if NO two optimal
/// decompositions have l1 norms in ratio r (so a plateau chi_n = chi_{n-1} is impossible).
pub fn l1_ratio_obstruction(ds: &[Dec], r: f64) -> bool {
    let l = l1_norms(ds);
    !l.iter()
        .any(|&a| l.iter().any(|&b| (b - r * a).abs() < 1e-7 * b.max(1.0)))
}

/// One representative per orbit of decompositions under the group generated by `gens`.
pub fn orbit_reps(t: &Table, psi: &[C], all: &[Dec], gens: &[Vec<u32>]) -> Vec<Dec> {
    let mut seen: HashSet<Vec<u32>> = HashSet::new();
    let mut reps = vec![];
    for d in all {
        if seen.contains(&d.idx) {
            continue;
        }
        for o in expand(t, psi, std::slice::from_ref(d), gens) {
            seen.insert(o.idx);
        }
        reps.push(d.clone());
    }
    reps
}

// ---------------------------------------------------------------- gluing (plateau lemma)
/// Given ALL rank-k decompositions D of psi1^{⊗(n-1)} (k = chi(psi1^{⊗(n-1)})), return all
/// rank-k decompositions of psi1^{⊗n} as lists of n-qubit vectors with coefficients.
/// Valid by the plateau lemma: every term of such a decomposition has both restrictions of the
/// last qubit nonzero, and the restrictions form decompositions in D.
pub fn glue(t: &Table, dset: &[Dec], p: &[C; 2], n: usize) -> Vec<(Vec<Vec<C>>, Vec<C>)> {
    glue_from(t, dset, dset, p, n)
}
/// As `glue`, with the A-side (<0|-restriction) decompositions taken from `aside` only (e.g. orbit
/// representatives under the symmetry group of psi^{⊗(n-1)}, which fixes the last qubit).
pub fn glue_from(
    t: &Table,
    aside: &[Dec],
    dset: &[Dec],
    p: &[C; 2],
    n: usize,
) -> Vec<(Vec<Vec<C>>, Vec<C>)> {
    let k = dset[0].idx.len();
    let ratio = p[1].div(p[0]); // psi_1/psi_0
    let rabs = ratio.abs();
    // index decompositions by sum of |coef|
    let mut bucket: HashMap<i64, Vec<usize>> = HashMap::new();
    let q = 1e6;
    for (i, d) in dset.iter().enumerate() {
        let s: f64 = d.coef.iter().map(|z| z.abs()).sum();
        bucket.entry((s * q).round() as i64).or_default().push(i);
    }
    let perms = permutations(k);
    let psin = tensor_power(p, n);
    let mut out = vec![];
    for d in aside {
        // need D' with |a'_{σi}| = |a_i| / rabs   (|omega| = 1)
        let s: f64 = d.coef.iter().map(|z| z.abs()).sum::<f64>() / rabs;
        let key = (s * q).round() as i64;
        for kk in [key - 1, key, key + 1] {
            if let Some(list) = bucket.get(&kk) {
                for &j in list {
                    let e = &dset[j];
                    for sg in &perms {
                        let mut ok = true;
                        let mut terms = vec![];
                        let mut coefs = vec![];
                        for i in 0..k {
                            let a = d.coef[i];
                            let b = e.coef[sg[i]];
                            let om = ratio * b.div(a);
                            if (om.abs() - 1.0).abs() > 1e-7 {
                                ok = false;
                                break;
                            }
                            let av = &t.states[d.idx[i] as usize];
                            let bv = &t.states[e.idx[sg[i]] as usize];
                            let mut v: Vec<C> = av.clone();
                            v.extend(bv.iter().map(|&z| z * om));
                            let v: Vec<C> = v.iter().map(|z| z.scale(1.0 / 2f64.sqrt())).collect();
                            if !is_stabilizer(&v) {
                                ok = false;
                                break;
                            }
                            terms.push(v);
                            coefs.push(a.scale(2f64.sqrt()) * p[0]);
                        }
                        if ok {
                            // verify the reconstruction
                            let mut r = psin.clone();
                            for i in 0..k {
                                for x in 0..r.len() {
                                    r[x] = r[x] - coefs[i] * terms[i][x];
                                }
                            }
                            assert!(norm2(&r).sqrt() < 1e-8, "glue reconstruction failed");
                            out.push((terms, coefs));
                        }
                    }
                }
            }
        }
    }
    out
}

pub fn permutations(k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    let mut out = vec![];
    for p in permutations(k - 1) {
        for pos in 0..=p.len() {
            let mut q = p.clone();
            q.insert(pos, k - 1);
            out.push(q);
        }
    }
    out
}

/// m-uniformity of a (stabilizer) state: every m-qubit marginal maximally mixed.  Tested via:
/// <s|_M v != 0 for every product of single-qubit stabilizer states s on every m-subset M.
pub fn min_restriction_ok(v: &[C], n: usize, m: usize) -> bool {
    let s = 1.0 / 2f64.sqrt();
    let one = [
        [C::new(1., 0.), C::default()],
        [C::default(), C::new(1., 0.)],
        [C::new(s, 0.), C::new(s, 0.)],
        [C::new(s, 0.), C::new(-s, 0.)],
        [C::new(s, 0.), C::new(0., s)],
        [C::new(s, 0.), C::new(0., -s)],
    ];
    // iterate subsets of size m and product bras
    for mask in 0..(1usize << n) {
        if mask.count_ones() as usize != m {
            continue;
        }
        let qs: Vec<usize> = (0..n).filter(|&j| mask >> j & 1 == 1).collect();
        let mut choice = vec![0usize; m];
        loop {
            // compute norm of <s|_M v
            let mut w = v.to_vec();
            for (t, &qb) in qs.iter().enumerate() {
                let br = one[choice[t]];
                let mut nw = vec![C::default(); w.len()];
                for x in 0..w.len() {
                    if x >> qb & 1 == 0 {
                        let y = x | (1 << qb);
                        nw[x] = br[0].conj() * w[x] + br[1].conj() * w[y];
                    }
                }
                w = nw;
            }
            if norm2(&w) < 1e-10 {
                return false;
            }
            let mut t = 0;
            loop {
                if t == m {
                    break;
                }
                choice[t] += 1;
                if choice[t] < 6 {
                    break;
                }
                choice[t] = 0;
                t += 1;
            }
            if t == m {
                break;
            }
        }
    }
    true
}

fn write_decs(path: &str, t: &Table, ds: &[Dec]) {
    // compact: state indices refer to enum_states(n) order (deterministic); coefficients for the
    // normalised target.
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    writeln!(
        f,
        "# n={} k={} count={}  format: idx:re,im per term (idx = position in enum_states(n))",
        t.n,
        ds.first().map(|d| d.idx.len()).unwrap_or(0),
        ds.len()
    )
    .unwrap();
    for d in ds {
        let parts: Vec<String> = d
            .idx
            .iter()
            .zip(&d.coef)
            .map(|(&i, c)| format!("{}:{:.15},{:.15}", i, c.re, c.im))
            .collect();
        writeln!(f, "{}", parts.join(" ")).unwrap();
    }
}

// ---------------------------------------------------------------- one-above-plateau search
/// Options for a term whose last-qubit restriction <0| is (proportional to) the (n-1)-qubit
/// stabilizer state `d`: returns (top scale x, bottom vector y) with phi = (x d, y) a normalised
/// n-qubit stabilizer state; includes the product option (1, 0).
pub fn completions(t: &Table, d: &[C]) -> Vec<(f64, Vec<C>)> {
    let s = 1.0 / 2f64.sqrt();
    let mut out = vec![(1.0, vec![C::default(); d.len()])];
    let sd = d.iter().filter(|z| z.abs() > 1e-9).count();
    let phases = [
        C::new(1., 0.),
        C::new(0., 1.),
        C::new(-1., 0.),
        C::new(0., -1.),
    ];
    for b in &t.states {
        if b.iter().filter(|z| z.abs() > 1e-9).count() != sd {
            continue;
        }
        for om in phases {
            let mut v: Vec<C> = d.iter().map(|z| z.scale(s)).collect();
            v.extend(b.iter().map(|&z| (z * om).scale(s)));
            if is_stabilizer(&v) {
                out.push((s, b.iter().map(|&z| (z * om).scale(s)).collect()));
            }
        }
    }
    out
}

/// Exhaustive search for decompositions of psi^{⊗(n-1)} ⊗ (alpha|0> + beta|1>) with k terms, in
/// which the <0|-restriction of the last qubit is DEGENERATE (contains a zero, two parallel
/// vectors, or is linearly dependent), given all optimal (k-1)-term decompositions `dopt` of
/// psi^{⊗(n-1)} (k-1 = chi(psi^{⊗(n-1)})).  Returns the decompositions found (term vectors).
pub fn degenerate_search(
    t: &Table,
    dopt: &[Dec],
    alpha: C,
    beta: C,
    k: usize,
    taus: bool,
) -> (Vec<Vec<Vec<C>>>, u64) {
    let dim = t.states[0].len();
    let psi_prev: Vec<C> = {
        // reconstruct psi^{⊗(n-1)} from the first decomposition
        let d = &dopt[0];
        let mut v = vec![C::default(); dim];
        for (i, &ix) in d.idx.iter().enumerate() {
            for x in 0..dim {
                v[x] = v[x] + d.coef[i] * t.states[ix as usize][x];
            }
        }
        v
    };
    let mut found = vec![];
    let mut combos = 0u64;
    let mut cache: HashMap<u32, Vec<(f64, Vec<C>)>> = HashMap::new();
    for d in dopt {
        for &ix in &d.idx {
            cache
                .entry(ix)
                .or_insert_with(|| completions(t, &t.states[ix as usize]));
        }
    }
    let target_full = |terms: &Vec<Vec<C>>| -> bool {
        // verify: psi_prev ⊗ (alpha, beta) in span, all coefficients nonzero, independent
        let mut tg: Vec<C> = psi_prev.iter().map(|&z| z * alpha).collect();
        tg.extend(psi_prev.iter().map(|&z| z * beta));
        let refs: Vec<&[C]> = terms.iter().map(|v| v.as_slice()).collect();
        if gram_det(&refs) < 1e-10 {
            return false;
        }
        let (c, res) = lsq(&refs, &tg);
        res < 1e-8 && c.iter().all(|z| z.abs() > 1e-8)
    };
    let mk = |x: f64, top: &[C], y: &[C]| -> Vec<C> {
        let mut v: Vec<C> = top.iter().map(|z| z.scale(x)).collect();
        v.extend_from_slice(y);
        v
    };
    for d in dopt {
        let m = d.idx.len(); // = k-1
        assert_eq!(m + 1, k);
        let opts: Vec<&Vec<(f64, Vec<C>)>> = d.idx.iter().map(|ix| &cache[ix]).collect();
        let tops: Vec<&Vec<C>> = d.idx.iter().map(|&ix| &t.states[ix as usize]).collect();
        // ---- type II: restrictions (D_1..D_m, 0); phi_k = (0, B_k)
        let mut choice = vec![0usize; m];
        loop {
            combos += 1;
            // c_i = alpha d_i / x_i ; R = beta psi - sum c_i y_i must be prop. to a stabilizer state
            let mut r: Vec<C> = psi_prev.iter().map(|&z| z * beta).collect();
            for i in 0..m {
                let (x, y) = &opts[i][choice[i]];
                let c = (alpha * d.coef[i]).scale(1.0 / x);
                for q in 0..dim {
                    r[q] = r[q] - c * y[q];
                }
            }
            if norm2(&r) > 1e-12 && is_stabilizer(&r) {
                let mut terms: Vec<Vec<C>> = (0..m)
                    .map(|i| {
                        let (x, y) = &opts[i][choice[i]];
                        mk(*x, tops[i], y)
                    })
                    .collect();
                let rn = norm2(&r).sqrt();
                let mut last = vec![C::default(); dim];
                last.extend(r.iter().map(|z| z.scale(1.0 / rn)));
                terms.push(last);
                if target_full(&terms) {
                    found.push(terms);
                }
            }
            let mut j = 0;
            while j < m {
                choice[j] += 1;
                if choice[j] < opts[j].len() {
                    break;
                }
                choice[j] = 0;
                j += 1;
            }
            if j == m {
                break;
            }
        }
        // ---- types IIIa / IIIb: restrictions (D_1..D_m, tau) with tau = sum_j e_j D_j
        // (IIIa: tau = D_j0, e = unit vector).  c_j = (alpha d_j - t e_j)/x_j, c_tau = t/x_tau;
        // bottom: R(t) = beta psi - sum_j alpha d_j y_j/x_j - t (y_tau/x_tau - sum_j e_j y_j/x_j) = 0
        let mut tau_list: Vec<(Vec<C>, Vec<C>)> = vec![]; // (tau vector, e)
        for j0 in 0..m {
            let mut e = vec![C::default(); m];
            e[j0] = C::new(1., 0.);
            tau_list.push((tops[j0].clone(), e));
        }
        if taus {
            for (ix, st) in t.states.iter().enumerate() {
                if d.idx.contains(&(ix as u32)) {
                    continue;
                }
                let refs: Vec<&[C]> = tops.iter().map(|v| v.as_slice()).collect();
                let (e, res) = lsq(&refs, st);
                if res < 1e-8 {
                    tau_list.push((st.clone(), e));
                }
            }
        }
        for (tau, e) in &tau_list {
            let topts = completions(t, tau);
            let mut choice = vec![0usize; m];
            loop {
                // base R0 and partial Delta
                let mut r0: Vec<C> = psi_prev.iter().map(|&z| z * beta).collect();
                let mut dl = vec![C::default(); dim];
                for i in 0..m {
                    let (x, y) = &opts[i][choice[i]];
                    let c = (alpha * d.coef[i]).scale(1.0 / x);
                    let ex = e[i].scale(1.0 / x);
                    for q in 0..dim {
                        r0[q] = r0[q] - c * y[q];
                        dl[q] = dl[q] - ex * y[q];
                    }
                }
                for (xt, yt) in &topts {
                    combos += 1;
                    let delta: Vec<C> = (0..dim).map(|q| dl[q] + yt[q].scale(1.0 / xt)).collect();
                    // need r0 = t * delta
                    let dn = norm2(&delta);
                    let ok = if dn < 1e-12 {
                        norm2(&r0) < 1e-12
                    } else {
                        let tt = dot(&delta, &r0).div(C::new(dn, 0.));
                        let mut res = 0.0;
                        for q in 0..dim {
                            res += (r0[q] - tt * delta[q]).n2();
                        }
                        res < 1e-12
                    };
                    if ok {
                        let mut terms: Vec<Vec<C>> = (0..m)
                            .map(|i| {
                                let (x, y) = &opts[i][choice[i]];
                                mk(*x, tops[i], y)
                            })
                            .collect();
                        terms.push(mk(*xt, tau, yt));
                        if target_full(&terms) {
                            found.push(terms);
                        }
                    }
                }
                let mut j = 0;
                while j < m {
                    choice[j] += 1;
                    if choice[j] < opts[j].len() {
                        break;
                    }
                    choice[j] = 0;
                    j += 1;
                }
                if j == m {
                    break;
                }
            }
        }
    }
    (found, combos)
}

/// ALL minimal k-term decompositions of psi^{⊗n} with k = chi(psi^{⊗(n-1)}) + 1, up to the
/// symmetry group (as lists of term vectors, possibly with repeats): the degenerate part
/// (degenerate_search over every bra-orbit representative) plus the non-degenerate part (glue of
/// all minimal k-term decompositions of psi^{⊗(n-1)}).  Proposition 7 of the write-up.
pub fn past_plateau(kind: &str, n: usize, verbose: bool) -> (usize, Vec<Vec<Vec<C>>>) {
    let p = psi1(kind);
    let t = Table::new(n - 1);
    let psi = tensor_power(&p, n - 1);
    let gens = sym_generators(&t, &p);
    let orb = orbits(t.states.len(), &gens);
    let group = group_closure(&gens, 1000);
    let mut st = SearchStats {
        w_count: 0,
        cand_count: 0,
        verified: 0,
    };
    let mut km1 = 2;
    let reps = loop {
        let r = search(&t, &psi, km1, &orb, group.as_ref(), &mut st);
        if !r.is_empty() {
            break r;
        }
        km1 += 1;
    };
    let dopt = expand(&t, &psi, &reps, &gens);
    let k = km1 + 1;
    if verbose {
        println!(
            "{}^{}: chi = {} ({} optimal decompositions)",
            kind,
            n - 1,
            km1,
            dopt.len()
        );
    }
    let mut found: Vec<Vec<Vec<C>>> = vec![];
    let t1 = Table::new(1);
    let g1 = sym_generators(&t1, &p);
    let o1 = orbits(6, &g1);
    let cl = cliffords1();
    for oid in 0..=*o1.iter().max().unwrap() {
        let sidx = (0..6).find(|&i| o1[i] == oid).unwrap();
        let u = *cl
            .iter()
            .find(|m| apply1(m, &t1.states[sidx])[0].abs() > 1.0 - 1e-9)
            .unwrap();
        let udag: M2 = [u[0].conj(), u[2].conj(), u[1].conj(), u[3].conj()];
        let up = apply1(&u, &p);
        let (fd, combos) = degenerate_search(&t, &dopt, up[0], up[1], k, true);
        if verbose {
            println!(
                "  degenerate, bra #{}: combos {} found {}",
                sidx,
                combos,
                fd.len()
            );
        }
        for f in fd {
            found.push(f.iter().map(|v| apply_local(&udag, n - 1, v)).collect());
        }
    }
    let mins = expand(
        &t,
        &psi,
        &search(&t, &psi, k, &orb, group.as_ref(), &mut st),
        &gens,
    );
    let mreps = orbit_reps(&t, &psi, &mins, &gens);
    let g = glue_from(&t, &mreps, &mins, &p, n);
    if verbose {
        println!(
            "  non-degenerate: {} minimal {}-term decompositions of {}^{} ({} orbits), glued {}",
            mins.len(),
            k,
            kind,
            n - 1,
            mreps.len(),
            g.len()
        );
    }
    for (terms, _) in g {
        found.push(terms);
    }
    let psin = tensor_power(&p, n);
    for f in &found {
        let refs: Vec<&[C]> = f.iter().map(|v| v.as_slice()).collect();
        let (c, res) = lsq(&refs, &psin);
        assert!(
            res < 1e-8 && c.iter().all(|z| z.abs() > 1e-8) && f.iter().all(|v| is_stabilizer(v))
        );
    }
    (k, found)
}

// ---------------------------------------------------------------- annealing (upper bounds)
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unif(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}
fn apply_gate(v: &mut [C], n: usize, g: usize, a: usize, b: usize) {
    let s = 1.0 / 2f64.sqrt();
    match g {
        0 => {
            // H on a
            for x in 0..v.len() {
                if x >> a & 1 == 0 {
                    let y = x | 1 << a;
                    let (p, q) = (v[x], v[y]);
                    v[x] = (p + q).scale(s);
                    v[y] = (p - q).scale(s);
                }
            }
        }
        1 => {
            // S on a
            for x in 0..v.len() {
                if x >> a & 1 == 1 {
                    v[x] = v[x] * C::new(0., 1.);
                }
            }
        }
        2 => {
            // CNOT a->b
            if a == b {
                return;
            }
            for x in 0..v.len() {
                if x >> a & 1 == 1 && x >> b & 1 == 0 {
                    v.swap(x, x | 1 << b);
                }
            }
        }
        _ => {
            // X on a
            for x in 0..v.len() {
                if x >> a & 1 == 0 {
                    v.swap(x, x | 1 << a);
                }
            }
        }
    }
    let _ = n;
}
/// 1 - |P psi|^2 where P projects on span(vs).
pub fn residual(vs: &[Vec<C>], psi: &[C]) -> f64 {
    let mut qs: Vec<Vec<C>> = vec![];
    for v in vs {
        let mut w = v.clone();
        for q in &qs {
            let c = dot(q, &w);
            for x in 0..w.len() {
                w[x] = w[x] - c * q[x];
            }
        }
        let wn = norm2(&w).sqrt();
        if wn < 1e-7 {
            continue;
        }
        for z in w.iter_mut() {
            *z = z.scale(1.0 / wn);
        }
        qs.push(w);
    }
    let mut r = 1.0;
    for q in &qs {
        r -= dot(q, psi).n2();
    }
    r.max(0.0)
}
pub fn anneal(psi: &[C], n: usize, k: usize, steps: u64, seed: u64) -> Option<Vec<Vec<C>>> {
    let mut rng = Rng(seed | 1);
    let dim = 1usize << n;
    let mut cur: Vec<Vec<C>> = (0..k)
        .map(|_| {
            let mut v = vec![C::default(); dim];
            v[0] = C::new(1., 0.);
            for _ in 0..(10 * n * n) {
                let (g, a, b) = (rng.below(4), rng.below(n), rng.below(n));
                apply_gate(&mut v, n, g, a, b);
            }
            v
        })
        .collect();
    let mut e = residual(&cur, psi);
    let beta0: f64 = 1.0;
    let beta1: f64 = 4000.0;
    for st in 0..steps {
        let beta = beta0 * (beta1 / beta0).powf(st as f64 / steps as f64);
        let i = rng.below(k);
        let (g, a, b) = (rng.below(4), rng.below(n), rng.below(n));
        let old = cur[i].clone();
        apply_gate(&mut cur[i], n, g, a, b);
        let e2 = residual(&cur, psi);
        if e2 <= e || rng.unif() < (-(e2 - e) * beta).exp() {
            e = e2;
        } else {
            cur[i] = old;
        }
        if e < 1e-12 {
            return Some(cur);
        }
    }
    None
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");
    match cmd {
        "count" => {
            for n in 1..=4 {
                let t = Table::new(n);
                println!("n={} stabilizer states={}", n, t.states.len());
            }
        }
        // search <kind> <n> <k> [prod]
        "search" => {
            let kind = &args[2];
            let n: usize = args[3].parse().unwrap();
            let k: usize = args[4].parse().unwrap();
            let t = if args.get(5).map(|s| s == "prod").unwrap_or(false) {
                Table::products(n)
            } else {
                Table::new(n)
            };
            let (psi, gens) = match special_target(kind, n) {
                Some(v) => (v, perm_conj_generators(&t)),
                None => {
                    let p = psi1(kind);
                    (tensor_power(&p, n), sym_generators(&t, &p))
                }
            };
            let orb = orbits(t.states.len(), &gens);
            let cap: usize = std::env::var("GROUP_CAP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1000);
            let group = group_closure(&gens, cap);
            eprintln!(
                "n={} states={} orbits={} |G|={:?}",
                n,
                t.states.len(),
                orb.iter().max().unwrap() + 1,
                group.as_ref().map(|g| g.len())
            );
            let mut st = SearchStats {
                w_count: 0,
                cand_count: 0,
                verified: 0,
            };
            let t0 = std::time::Instant::now();
            let reps = search(&t, &psi, k, &orb, group.as_ref(), &mut st);
            let all = expand(&t, &psi, &reps, &gens);
            println!(
                "{}^{} k={}: decompositions found: {} (reps {}), W={} cand={} verified={} time={:.1}s",
                kind, n, k, all.len(), reps.len(), st.w_count, st.cand_count, st.verified, t0.elapsed().as_secs_f64()
            );
            let path = format!("dec_{}{}_k{}.txt", kind, n, k);
            write_decs(&path, &t, &all);
            // l1 norms of coefficient vectors (normalised target and terms): symmetry invariant
            let mut l1: Vec<i64> = all
                .iter()
                .map(|d| (d.coef.iter().map(|z| z.abs()).sum::<f64>() * 1e9).round() as i64)
                .collect();
            l1.sort();
            let mut hist: Vec<(f64, usize)> = vec![];
            for v in l1 {
                if let Some(last) = hist.last_mut() {
                    if (last.0 * 1e9).round() as i64 == v {
                        last.1 += 1;
                        continue;
                    }
                }
                hist.push((v as f64 / 1e9, 1));
            }
            if hist.len() <= 40 {
                println!(
                    "l1 norms: {}",
                    hist.iter()
                        .map(|(v, c)| format!("{:.9}x{}", v, c))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            } else {
                println!(
                    "l1 norms: {} distinct values, min {:.9} max {:.9}",
                    hist.len(),
                    hist[0].0,
                    hist.last().unwrap().0
                );
            }
        }
        // anneal <kind> <n> <k> <steps> <restarts>
        "anneal" => {
            let kind = &args[2];
            let n: usize = args[3].parse().unwrap();
            let k: usize = args[4].parse().unwrap();
            let steps: u64 = args[5].parse().unwrap();
            let restarts: u64 = args[6].parse().unwrap();
            let psi = match special_target(kind, n) {
                Some(v) => v,
                None => tensor_power(&psi1(kind), n),
            };
            for r in 0..restarts {
                if let Some(vs) = anneal(&psi, n, k, steps, 0x1234567 + 7919 * r) {
                    let refs: Vec<&[C]> = vs.iter().map(|v| v.as_slice()).collect();
                    let (c, res) = lsq(&refs, &psi);
                    assert!(vs.iter().all(|v| is_stabilizer(v)));
                    println!(
                        "FOUND {}^{} rank<= {} at restart {} residual {:.2e}",
                        kind, n, k, r, res
                    );
                    for (v, cc) in vs.iter().zip(&c) {
                        let cv = canonical(v);
                        println!(
                            "  c={:.12}{:+.12}i v={}",
                            cc.re,
                            cc.im,
                            cv.iter()
                                .map(|z| format!("({:.6},{:.6})", z.re, z.im))
                                .collect::<Vec<_>>()
                                .join(" ")
                        );
                    }
                    return;
                }
            }
            println!(
                "not found: {}^{} k={} steps={} restarts={}",
                kind, n, k, steps, restarts
            );
        }
        // pblock <kind> <b>: kill probability for the block-local restricted model (uniform mu over
        // b-qubit stabilizer states non-orthogonal to psi^{⊗b}); prints p and exponent per qubit.
        "pblock" => {
            let kind = &args[2];
            let b: usize = args[3].parse().unwrap();
            let t = Table::new(b);
            let psi = tensor_power(&psi1(kind), b);
            let mu: Vec<usize> = (0..t.states.len())
                .filter(|&i| dot(&t.states[i], &psi).abs() > 1e-9)
                .collect();
            let mut minkill = usize::MAX;
            for tau in &t.states {
                let kill = mu
                    .iter()
                    .filter(|&&i| dot(&t.states[i], tau).abs() < 1e-9)
                    .count();
                minkill = minkill.min(kill);
            }
            let p = minkill as f64 / mu.len() as f64;
            println!("{} b={}: |Stab_b|={} |mu|={} min kills={} p={:.6} => R_b >= (1/(1-p))^(n/b), exponent {:.4} bits/qubit; chi(psi^b)-block upper bound exponent for comparison", kind, b, t.states.len(), mu.len(), minkill, p, -(1.0 - p).log2() / b as f64);
        }
        // past <kind> <n>: ALL k-term decompositions of psi^{⊗n}, k = chi(n-1)+1, up to symmetry:
        // degenerate part (deg, every bra-orbit representative) + non-degenerate part (gluemin).
        // For n <= 4 the result is expanded under the symmetry group and written out.
        "past" => {
            let kind = &args[2];
            let n: usize = args[3].parse().unwrap();
            let p = psi1(kind);
            let (k, found) = past_plateau(kind, n, true);
            println!(
                "{}^{}: {}-term decompositions found (up to symmetry, with repeats): {}",
                kind,
                n,
                k,
                found.len()
            );
            if n <= 4 && !found.is_empty() {
                let psin = tensor_power(&p, n);
                let tn = Table::new(n);
                let gn = sym_generators(&tn, &p);
                let ds: Vec<Dec> = found
                    .iter()
                    .map(|f| {
                        let mut ix: Vec<u32> = f.iter().map(|v| tn.lookup(v).unwrap()).collect();
                        ix.sort();
                        check_set(&tn, &psin, &ix).unwrap()
                    })
                    .collect();
                let all = expand(&tn, &psin, &ds, &gn);
                let mut l1: Vec<i64> = l1_norms(&all)
                    .iter()
                    .map(|x| (x * 1e9).round() as i64)
                    .collect();
                l1.sort();
                let mut hist: Vec<(i64, usize)> = vec![];
                for v in l1 {
                    if let Some(last) = hist.last_mut() {
                        if last.0 == v {
                            last.1 += 1;
                            continue;
                        }
                    }
                    hist.push((v, 1));
                }
                println!(
                    "{}^{}: total {} decompositions with {} terms; l1 norms: {}",
                    kind,
                    n,
                    all.len(),
                    k,
                    hist.iter()
                        .map(|(v, c)| format!("{:.9}x{}", *v as f64 / 1e9, c))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
                write_decs(&format!("dec_{}{}_k{}.txt", kind, n, k), &tn, &all);
                let oreps = orbit_reps(&tn, &psin, &all, &gn);
                write_decs(&format!("dec_{}{}_k{}_reps.txt", kind, n, k), &tn, &oreps);
                println!("  {} orbits; representatives written", oreps.len());
                if std::env::var("NEXT").is_ok() {
                    // non-degenerate half of the k-term search at n+1 (valid when chi(psi^{⊗n}) = k-1,
                    // i.e. these are the minimal k-term decompositions one past the plateau)
                    let g = glue_from(&tn, &oreps, &all, &p, n + 1);
                    println!(
                        "{}^{}: non-degenerate {}-term decompositions (A side up to symmetry): {}",
                        kind,
                        n + 1,
                        k,
                        g.len()
                    );
                }
            }
        }
        // deg <kind> <n> : the degenerate-restriction half of the k = chi(n-1)+1 search at n
        // (k-1 = chi(psi^{⊗(n-1)}) is taken from an exhaustive search).  Bras s run over orbit
        // representatives of the 6 single-qubit stabilizer states under the symmetry of psi.
        "deg" => {
            let kind = &args[2];
            let n: usize = args[3].parse().unwrap();
            let p = psi1(kind);
            let t = Table::new(n - 1);
            let psi = tensor_power(&p, n - 1);
            let gens = sym_generators(&t, &p);
            let orb = orbits(t.states.len(), &gens);
            let cap: usize = std::env::var("GROUP_CAP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1000);
            let group = group_closure(&gens, cap);
            let mut st = SearchStats {
                w_count: 0,
                cand_count: 0,
                verified: 0,
            };
            // chi at n-1
            let mut km1 = 2;
            let reps = loop {
                let r = search(&t, &psi, km1, &orb, group.as_ref(), &mut st);
                if !r.is_empty() {
                    break r;
                }
                km1 += 1;
            };
            // orbit representatives of optimal decompositions under the (n-1)-qubit symmetry group
            let all = expand(&t, &psi, &reps, &gens);
            let mut seen: HashSet<Vec<u32>> = HashSet::new();
            let mut orbreps = vec![];
            for dd in &all {
                if seen.contains(&dd.idx) {
                    continue;
                }
                let orbit_sets = expand(&t, &psi, std::slice::from_ref(dd), &gens);
                for o in &orbit_sets {
                    seen.insert(o.idx.clone());
                }
                orbreps.push(dd.clone());
            }
            println!(
                "{}^{}: chi = {}, {} optimal decompositions in {} orbits",
                kind,
                n - 1,
                km1,
                all.len(),
                orbreps.len()
            );
            // bra representatives
            let t1 = Table::new(1);
            let g1 = sym_generators(&t1, &p);
            let o1 = orbits(6, &g1);
            let cl = cliffords1();
            let taus = std::env::var("NO_TAUS").is_err();
            let mut total = 0;
            for oid in 0..=*o1.iter().max().unwrap() {
                let sidx = (0..6).find(|&i| o1[i] == oid).unwrap();
                let sv = &t1.states[sidx];
                let u = cl
                    .iter()
                    .find(|m| apply1(m, sv)[0].abs() > 1.0 - 1e-9)
                    .unwrap();
                let up = apply1(u, &p);
                // all optimal decompositions (not orbit representatives): an antiunitary symmetry of
                // psi^{⊗(n-1)} need not fix the transformed target, so no reduction is used here.
                let (found, combos) = degenerate_search(&t, &all, up[0], up[1], km1 + 1, taus);
                println!("  bra s = state #{} (|<0|U s>|=1): alpha={:.6}{:+.6}i beta={:.6}{:+.6}i  combos={} found={}", sidx, up[0].re, up[0].im, up[1].re, up[1].im, combos, found.len());
                total += found.len();
                if let Some(f) = found.first() {
                    for v in f {
                        println!(
                            "    term {}",
                            canonical(v)
                                .iter()
                                .map(|z| format!("({:.4},{:.4})", z.re, z.im))
                                .collect::<Vec<_>>()
                                .join(" ")
                        );
                    }
                }
            }
            println!(
                "degenerate-restriction decompositions with {} terms: {}",
                km1 + 1,
                total
            );
        }
        // degcheck <kind> <n> <bra index 0..5>: validate degenerate_search against a direct exhaustive
        // search (no symmetry) for the transformed target at small n.
        "degcheck" => {
            let kind = &args[2];
            let n: usize = args[3].parse().unwrap();
            let sidx: usize = args[4].parse().unwrap();
            let p = psi1(kind);
            let tp = Table::new(n - 1);
            let psi = tensor_power(&p, n - 1);
            let gens = sym_generators(&tp, &p);
            let orb = orbits(tp.states.len(), &gens);
            let mut st = SearchStats {
                w_count: 0,
                cand_count: 0,
                verified: 0,
            };
            let mut km1 = 2;
            let reps = loop {
                let r = search(&tp, &psi, km1, &orb, None, &mut st);
                if !r.is_empty() {
                    break r;
                }
                km1 += 1;
            };
            let all = expand(&tp, &psi, &reps, &gens);
            let t1 = Table::new(1);
            let u = cliffords1()
                .into_iter()
                .find(|m| apply1(m, &t1.states[sidx])[0].abs() > 1.0 - 1e-9)
                .unwrap();
            let up = apply1(&u, &p);
            let (found, _) = degenerate_search(&tp, &all, up[0], up[1], km1 + 1, true);
            let mut fs: HashSet<Vec<Vec<i64>>> = HashSet::new();
            for f in &found {
                let mut ks: Vec<Vec<i64>> = f.iter().map(|v| key_of(&canonical(v))).collect();
                ks.sort();
                fs.insert(ks);
            }
            // direct: target psi^{⊗(n-1)} ⊗ (up0, up1), last qubit = MSB
            let t = Table::new(n);
            let mut tg: Vec<C> = psi.iter().map(|&z| z * up[0]).collect();
            tg.extend(psi.iter().map(|&z| z * up[1]));
            let triv: Vec<u32> = (0..t.states.len() as u32).collect();
            let orbt = orbits(t.states.len(), &[triv.clone()]);
            let direct = search(&t, &tg, km1 + 1, &orbt, None, &mut st);
            let direct = expand(&t, &tg, &direct, &[triv]);
            let half = 1usize << (n - 1);
            let mut ndeg = 0;
            let mut ds: HashSet<Vec<Vec<i64>>> = HashSet::new();
            for d in &direct {
                let tops: Vec<Vec<C>> = d
                    .idx
                    .iter()
                    .map(|&i| t.states[i as usize][..half].to_vec())
                    .collect();
                let nz: Vec<&[C]> = tops
                    .iter()
                    .filter(|v| norm2(v) > 1e-12)
                    .map(|v| v.as_slice())
                    .collect();
                let degen = nz.len() < tops.len() || gram_det(&nz) < 1e-10;
                if degen {
                    ndeg += 1;
                    let mut ks: Vec<Vec<i64>> = d
                        .idx
                        .iter()
                        .map(|&i| key_of(&t.states[i as usize]))
                        .collect();
                    ks.sort();
                    ds.insert(ks);
                }
            }
            println!("{}^{} bra#{}: direct {} decompositions with {} terms, of which degenerate {}; degenerate_search found {} (sets equal: {})", kind, n, sidx, direct.len(), km1+1, ndeg, fs.len(), fs == ds);
        }
        // gluecheck <kind> <n> <k>: glue() over ALL minimal k-term decompositions at n-1 must give
        // exactly the k-term decompositions at n whose two last-qubit restrictions are both
        // nondegenerate (validation of the non-plateau use of glue()).
        "gluecheck" => {
            let kind = &args[2];
            let n: usize = args[3].parse().unwrap();
            let k: usize = args[4].parse().unwrap();
            let p = psi1(kind);
            let tp = Table::new(n - 1);
            let psip = tensor_power(&p, n - 1);
            let gp = sym_generators(&tp, &p);
            let op = orbits(tp.states.len(), &gp);
            let grp = group_closure(&gp, 1000);
            let mut st = SearchStats {
                w_count: 0,
                cand_count: 0,
                verified: 0,
            };
            let dp = expand(
                &tp,
                &psip,
                &search(&tp, &psip, k, &op, grp.as_ref(), &mut st),
                &gp,
            );
            let g = glue(&tp, &dp, &p, n);
            let mut gs: HashSet<Vec<Vec<i64>>> = HashSet::new();
            for (terms, _) in &g {
                let mut ks: Vec<Vec<i64>> = terms.iter().map(|v| key_of(&canonical(v))).collect();
                ks.sort();
                gs.insert(ks);
            }
            let t = Table::new(n);
            let psi = tensor_power(&p, n);
            let gn = sym_generators(&t, &p);
            let on = orbits(t.states.len(), &gn);
            let grn = group_closure(&gn, 1000);
            let direct = expand(
                &t,
                &psi,
                &search(&t, &psi, k, &on, grn.as_ref(), &mut st),
                &gn,
            );
            let half = 1usize << (n - 1);
            let mut ds: HashSet<Vec<Vec<i64>>> = HashSet::new();
            for d in &direct {
                let mut ok = true;
                for part in 0..2 {
                    let v: Vec<Vec<C>> = d
                        .idx
                        .iter()
                        .map(|&i| t.states[i as usize][part * half..(part + 1) * half].to_vec())
                        .collect();
                    let nz: Vec<&[C]> = v
                        .iter()
                        .filter(|x| norm2(x) > 1e-12)
                        .map(|x| x.as_slice())
                        .collect();
                    if nz.len() < v.len() || gram_det(&nz) < 1e-10 {
                        ok = false;
                    }
                }
                if ok {
                    let mut ks: Vec<Vec<i64>> = d
                        .idx
                        .iter()
                        .map(|&i| key_of(&t.states[i as usize]))
                        .collect();
                    ks.sort();
                    ds.insert(ks);
                }
            }
            println!("{}^{} k={}: |D_min(n-1)|={} glue sets={} direct={} direct-nondegenerate={} equal={}", kind, n, k, dp.len(), gs.len(), direct.len(), ds.len(), gs == ds);
        }
        // gluemin <kind> <n> <k>: the type-I x type-I half of the k = chi(n-1)+1 search at n:
        // glue over all minimal k-term decompositions of psi^{⊗(n-1)} (A side: orbit reps).
        "gluemin" => {
            let kind = &args[2];
            let n: usize = args[3].parse().unwrap();
            let k: usize = args[4].parse().unwrap();
            let p = psi1(kind);
            let t = Table::new(n - 1);
            let psi = tensor_power(&p, n - 1);
            let gens = sym_generators(&t, &p);
            let orb = orbits(t.states.len(), &gens);
            let cap: usize = std::env::var("GROUP_CAP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1000);
            let group = group_closure(&gens, cap);
            eprintln!("|G|={:?}", group.as_ref().map(|g| g.len()));
            let mut st = SearchStats {
                w_count: 0,
                cand_count: 0,
                verified: 0,
            };
            let t0 = std::time::Instant::now();
            let reps = search(&t, &psi, k, &orb, group.as_ref(), &mut st);
            let all = expand(&t, &psi, &reps, &gens);
            let oreps = orbit_reps(&t, &psi, &all, &gens);
            println!(
                "{}^{}: {} minimal {}-term decompositions in {} orbits (search {:.0}s, W={})",
                kind,
                n - 1,
                all.len(),
                k,
                oreps.len(),
                t0.elapsed().as_secs_f64(),
                st.w_count
            );
            write_decs(
                &format!("dec_{}{}_k{}_reps.txt", kind, n - 1, k),
                &t,
                &oreps,
            );
            let g = glue_from(&t, &oreps, &all, &p, n);
            println!(
                "{}^{}: type-I x type-I {}-term decompositions (A side up to symmetry): {}",
                kind,
                n,
                k,
                g.len()
            );
            if let Some((terms, coefs)) = g.first() {
                for (v, c) in terms.iter().zip(coefs) {
                    println!(
                        "  c={:.12}{:+.12}i v={}",
                        c.re,
                        c.im,
                        canonical(v)
                            .iter()
                            .map(|z| format!("({:.4},{:.4})", z.re, z.im))
                            .collect::<Vec<_>>()
                            .join(" ")
                    );
                }
            }
        }
        // chain <kind> <n0> <k> <nmax>: all rank-k decompositions at n0 by exhaustive search, then
        // plateau-gluing up to nmax.  Valid only when chi(psi^{⊗n0}) = k (checked: search k-1 empty).
        "chain" => {
            let kind = &args[2];
            let n0: usize = args[3].parse().unwrap();
            let k: usize = args[4].parse().unwrap();
            let nmax: usize = args[5].parse().unwrap();
            let p = psi1(kind);
            let mut t = Table::new(n0);
            let psi = tensor_power(&p, n0);
            let gens = sym_generators(&t, &p);
            let orb = orbits(t.states.len(), &gens);
            let cap: usize = std::env::var("GROUP_CAP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1000);
            let group = group_closure(&gens, cap);
            let mut st = SearchStats {
                w_count: 0,
                cand_count: 0,
                verified: 0,
            };
            if k > 2 {
                let lower = search(&t, &psi, k - 1, &orb, group.as_ref(), &mut st);
                assert!(lower.is_empty(), "chi < k at n0");
                println!(
                    "{}^{}: no rank-{} decomposition (exhaustive)",
                    kind,
                    n0,
                    k - 1
                );
            }
            let reps = search(&t, &psi, k, &orb, group.as_ref(), &mut st);
            let mut d = expand(&t, &psi, &reps, &gens);
            println!("{}^{}: {} rank-{} decompositions", kind, n0, d.len(), k);
            for n in (n0 + 1)..=nmax {
                let g = glue(&t, &d, &p, n);
                // plateau lemma consistency: every term must be 1-uniform
                let mut nonuni = 0;
                for (terms, _) in &g {
                    for v in terms {
                        if !min_restriction_ok(v, n, 1) {
                            nonuni += 1;
                        }
                    }
                }
                // dedupe as sets of canonical vectors
                let mut sets: HashSet<Vec<Vec<i64>>> = HashSet::new();
                for (terms, _) in &g {
                    let mut ks: Vec<Vec<i64>> =
                        terms.iter().map(|v| key_of(&canonical(v))).collect();
                    ks.sort();
                    sets.insert(ks);
                }
                println!("{}^{}: gluing gives {} rank-{} decompositions (raw {}), non-1-uniform terms: {}", kind, n, sets.len(), k, g.len(), nonuni);
                if g.is_empty() {
                    println!("=> chi({}^{}) >= {}", kind, n, k + 1);
                    break;
                }
                if n == nmax {
                    let (terms, coefs) = &g[0];
                    for (v, c) in terms.iter().zip(coefs) {
                        println!(
                            "  c={:.12}{:+.12}i  v={:?}",
                            c.re,
                            c.im,
                            v.iter()
                                .map(|z| (format!("{:.4}", z.re), format!("{:.4}", z.im)))
                                .collect::<Vec<_>>()
                        );
                    }
                    break;
                }
                // convert to Dec over the n-qubit table
                let tn = Table::new(n);
                let psin = tensor_power(&p, n);
                let mut nd = vec![];
                let mut seen: HashSet<Vec<u32>> = HashSet::new();
                for (terms, _) in &g {
                    let idx: Vec<u32> = terms
                        .iter()
                        .map(|v| tn.lookup(v).expect("glued term not stabilizer"))
                        .collect();
                    let mut s2 = idx.clone();
                    s2.sort();
                    if seen.insert(s2.clone()) {
                        nd.push(check_set(&tn, &psin, &s2).expect("glued set must decompose"));
                    }
                }
                write_decs(&format!("dec_{}{}_k{}.txt", kind, n, k), &tn, &nd);
                d = nd;
                t = tn;
            }
        }
        // brute <kind> <n> <k>: naive enumeration of all k-subsets (k<=3), cross-check of `search`
        "brute" => {
            let kind = &args[2];
            let n: usize = args[3].parse().unwrap();
            let k: usize = args[4].parse().unwrap();
            let p = psi1(kind);
            let t = Table::new(n);
            let psi = tensor_power(&p, n);
            let nst = t.states.len();
            let mut cnt = 0u64;
            if k == 2 {
                for a in 0..nst {
                    for b in (a + 1)..nst {
                        if check_set(&t, &psi, &[a as u32, b as u32]).is_some() {
                            cnt += 1;
                        }
                    }
                }
            } else if k == 4 {
                for a in 0..nst {
                    for b in (a + 1)..nst {
                        for c in (b + 1)..nst {
                            for d in (c + 1)..nst {
                                if check_set(&t, &psi, &[a as u32, b as u32, c as u32, d as u32])
                                    .is_some()
                                {
                                    cnt += 1;
                                }
                            }
                        }
                    }
                }
            } else {
                for a in 0..nst {
                    for b in (a + 1)..nst {
                        // orthonormal basis of span(psi,a,b)
                        let mut qs: Vec<Vec<C>> = vec![];
                        for v in [&psi, &t.states[a], &t.states[b]] {
                            let mut w = v.clone();
                            for q in &qs {
                                let c = dot(q, &w);
                                for x in 0..w.len() {
                                    w[x] = w[x] - c * q[x];
                                }
                            }
                            let wn = norm2(&w).sqrt();
                            if wn < 1e-6 {
                                break;
                            }
                            for z in w.iter_mut() {
                                *z = z.scale(1.0 / wn);
                            }
                            qs.push(w);
                        }
                        if qs.len() < 3 {
                            continue;
                        }
                        for c in (b + 1)..nst {
                            let v = &t.states[c];
                            let r: f64 = 1.0 - qs.iter().map(|q| dot(q, v).n2()).sum::<f64>();
                            if r < 1e-9
                                && check_set(&t, &psi, &[a as u32, b as u32, c as u32]).is_some()
                            {
                                cnt += 1;
                            }
                        }
                    }
                }
            }
            println!("brute {}^{} k={}: {} decompositions", kind, n, k, cnt);
        }
        _ => println!("commands: count | search K n k | glue ..."),
    }
}
