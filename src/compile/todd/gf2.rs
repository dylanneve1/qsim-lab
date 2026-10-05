//! Dense GF(2) vectors and row reduction for the T-count optimiser.

/// A GF(2) vector stored in 64-bit words (bit `i` is bit `i % 64` of word
/// `i / 64`). The length is implicit: all vectors that are combined have
/// the same number of words.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Bits(pub Vec<u64>);

impl Bits {
    /// The zero vector with room for `nbits` bits (at least one word).
    pub fn zeros(nbits: usize) -> Self {
        Bits(vec![0; nbits.div_ceil(64).max(1)])
    }

    /// The unit vector `e_i` with room for `nbits` bits.
    pub fn unit(nbits: usize, i: usize) -> Self {
        let mut b = Bits::zeros(nbits);
        b.set(i, true);
        b
    }

    /// Bit `i`.
    #[inline]
    pub fn get(&self, i: usize) -> bool {
        (self.0[i >> 6] >> (i & 63)) & 1 == 1
    }

    /// Sets bit `i` to `v`.
    #[inline]
    pub fn set(&mut self, i: usize, v: bool) {
        let m = 1u64 << (i & 63);
        if v {
            self.0[i >> 6] |= m;
        } else {
            self.0[i >> 6] &= !m;
        }
    }

    /// Flips bit `i`.
    #[inline]
    pub fn flip(&mut self, i: usize) {
        self.0[i >> 6] ^= 1u64 << (i & 63);
    }

    /// `self ^= other`.
    #[inline]
    pub fn xor_with(&mut self, other: &Bits) {
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            *a ^= *b;
        }
    }

    /// `self & other`.
    #[inline]
    pub fn and(&self, other: &Bits) -> Bits {
        Bits(self.0.iter().zip(&other.0).map(|(a, b)| a & b).collect())
    }

    /// True if every bit is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&w| w == 0)
    }

    /// Number of one bits.
    #[inline]
    pub fn count_ones(&self) -> usize {
        self.0.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Index of the lowest one bit.
    #[inline]
    pub fn first_one(&self) -> Option<usize> {
        self.0
            .iter()
            .enumerate()
            .find(|(_, &w)| w != 0)
            .map(|(i, &w)| i * 64 + w.trailing_zeros() as usize)
    }

    /// Index of the highest one bit.
    #[inline]
    pub fn last_one(&self) -> Option<usize> {
        self.0
            .iter()
            .enumerate()
            .rev()
            .find(|(_, &w)| w != 0)
            .map(|(i, &w)| i * 64 + 63 - w.leading_zeros() as usize)
    }

    /// Indices of the one bits, ascending.
    pub fn ones(&self) -> impl Iterator<Item = usize> + '_ {
        self.0.iter().enumerate().flat_map(|(i, &w)| {
            let mut w = w;
            std::iter::from_fn(move || {
                if w == 0 {
                    None
                } else {
                    let t = w.trailing_zeros() as usize;
                    w &= w - 1;
                    Some(i * 64 + t)
                }
            })
        })
    }

    /// Inner product over GF(2).
    #[inline]
    pub fn dot(&self, other: &Bits) -> bool {
        self.0
            .iter()
            .zip(&other.0)
            .fold(0u32, |acc, (a, b)| acc ^ (a & b).count_ones())
            & 1
            == 1
    }
}

/// Reduces `rows` in place to reduced row echelon form, pivoting only on
/// the columns `0..ncols` (bits beyond `ncols` are carried along, which is
/// how callers track row operations). Returns the pivot column of each of
/// the first `rank` rows; rows `rank..` are zero on `0..ncols`.
pub fn rref(rows: &mut [Bits], ncols: usize) -> Vec<usize> {
    let mut pivots = Vec::new();
    let mut r = 0;
    for col in 0..ncols {
        if r == rows.len() {
            break;
        }
        let Some(p) = (r..rows.len()).find(|&i| rows[i].get(col)) else {
            continue;
        };
        rows.swap(r, p);
        let (head, tail) = rows.split_at_mut(r);
        let (pivot, tail) = tail.split_first_mut().expect("row r exists");
        for row in head.iter_mut().chain(tail.iter_mut()) {
            if row.get(col) {
                row.xor_with(pivot);
            }
        }
        pivots.push(col);
        r += 1;
    }
    pivots
}

/// A basis of `{y : M y = 0}` for the matrix whose rows are `rows`
/// (columns `0..ncols`). `rows` is consumed (reduced in place).
pub fn kernel(rows: &mut [Bits], ncols: usize) -> Vec<Bits> {
    let pivots = rref(rows, ncols);
    let mut is_pivot = vec![false; ncols];
    for &p in &pivots {
        is_pivot[p] = true;
    }
    let mut basis = Vec::new();
    for f in (0..ncols).filter(|&f| !is_pivot[f]) {
        let mut y = Bits::unit(ncols, f);
        for (r, &p) in pivots.iter().enumerate() {
            if rows[r].get(f) {
                y.set(p, true);
            }
        }
        basis.push(y);
    }
    basis
}

/// Incremental membership test for the span of a set of vectors, also
/// returning the combination that produces a member.
pub struct SpanSolver {
    /// Reduced basis rows: `(vector, combination of the inputs)`.
    rows: Vec<(Bits, Bits, usize)>,
    ninputs: usize,
}

impl SpanSolver {
    /// Builds the solver from `vectors` (each of the same width).
    pub fn new(vectors: &[Bits]) -> Self {
        let k = vectors.len();
        let mut rows: Vec<(Bits, Bits, usize)> = Vec::new();
        for (i, v) in vectors.iter().enumerate() {
            let mut v = v.clone();
            let mut comb = Bits::unit(k, i);
            for (b, c, p) in &rows {
                if v.get(*p) {
                    v.xor_with(b);
                    comb.xor_with(c);
                }
            }
            if let Some(p) = v.first_one() {
                // keep the basis reduced on pivot columns
                for (b, c, _) in rows.iter_mut() {
                    if b.get(p) {
                        b.xor_with(&v);
                        c.xor_with(&comb);
                    }
                }
                rows.push((v, comb, p));
            }
        }
        SpanSolver { rows, ninputs: k }
    }

    /// The combination (a subset of the input vectors, as a bit vector over
    /// their indices) whose sum is `target`, or `None` if `target` is not
    /// in the span.
    pub fn solve(&self, target: &Bits) -> Option<Bits> {
        let mut v = target.clone();
        let mut comb = Bits::zeros(self.ninputs);
        for (b, c, p) in &self.rows {
            if v.get(*p) {
                v.xor_with(b);
                comb.xor_with(c);
            }
        }
        if v.is_zero() {
            Some(comb)
        } else {
            None
        }
    }

    /// Dimension of the span.
    pub fn rank(&self) -> usize {
        self.rows.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_basics() {
        let mut a = Bits::zeros(130);
        a.set(3, true);
        a.set(129, true);
        assert_eq!(a.ones().collect::<Vec<_>>(), vec![3, 129]);
        assert_eq!(a.first_one(), Some(3));
        assert_eq!(a.last_one(), Some(129));
        assert_eq!(a.count_ones(), 2);
        let b = Bits::unit(130, 129);
        assert!(a.dot(&b));
        a.xor_with(&b);
        assert_eq!(a.ones().collect::<Vec<_>>(), vec![3]);
    }

    #[test]
    fn kernel_and_span() {
        // rows: [1 1 0], [0 1 1] -> kernel spanned by [1 1 1]
        let mut rows = vec![Bits(vec![0b011]), Bits(vec![0b110])];
        let k = kernel(&mut rows, 3);
        assert_eq!(k, vec![Bits(vec![0b111])]);
        let s = SpanSolver::new(&[Bits(vec![0b011]), Bits(vec![0b110])]);
        assert_eq!(s.solve(&Bits(vec![0b101])), Some(Bits(vec![0b11])));
        assert_eq!(s.solve(&Bits(vec![0b001])), None);
    }
}
