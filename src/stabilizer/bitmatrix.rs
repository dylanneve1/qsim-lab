//! Square, bit-packed boolean matrices with an in-place transpose.

/// An `np x np` bit matrix (`np` a multiple of 64) stored as `np` lines of
/// `np / 64` words. Bit `j` of line `i` is `data[i * w + j / 64] >> (j % 64)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitMatrix {
    np: usize,
    w: usize,
    data: Vec<u64>,
}

impl BitMatrix {
    pub fn zeros(np: usize) -> Self {
        assert_eq!(np % 64, 0, "size must be a multiple of 64");
        let w = np / 64;
        BitMatrix {
            np,
            w,
            data: vec![0; np * w],
        }
    }

    pub fn identity(np: usize) -> Self {
        let mut m = Self::zeros(np);
        for i in 0..np {
            m.set(i, i, true);
        }
        m
    }

    /// Resets all elements to zero without reallocating.
    pub fn reset_zeros(&mut self) {
        self.data.fill(0);
    }

    /// Resets this matrix to the identity matrix without reallocating.
    pub fn reset_identity(&mut self) {
        self.reset_zeros();
        for i in 0..self.np {
            self.set(i, i, true);
        }
    }

    /// Number of lines (and bits per line).
    pub fn size(&self) -> usize {
        self.np
    }

    /// Words per line.
    pub fn words(&self) -> usize {
        self.w
    }

    pub fn bytes(&self) -> usize {
        self.data.len() * 8
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> bool {
        (self.data[i * self.w + j / 64] >> (j % 64)) & 1 == 1
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: bool) {
        let word = &mut self.data[i * self.w + j / 64];
        let bit = 1u64 << (j % 64);
        if v {
            *word |= bit;
        } else {
            *word &= !bit;
        }
    }

    #[inline]
    pub fn line(&self, i: usize) -> &[u64] {
        &self.data[i * self.w..(i + 1) * self.w]
    }

    #[inline]
    pub fn line_mut(&mut self, i: usize) -> &mut [u64] {
        &mut self.data[i * self.w..(i + 1) * self.w]
    }

    /// Mutable access to two distinct lines at once.
    pub fn two_lines_mut(&mut self, a: usize, b: usize) -> (&mut [u64], &mut [u64]) {
        assert_ne!(a, b);
        let w = self.w;
        if a < b {
            let (left, right) = self.data.split_at_mut(b * w);
            (&mut left[a * w..(a + 1) * w], &mut right[..w])
        } else {
            let (left, right) = self.data.split_at_mut(a * w);
            (&mut right[..w], &mut left[b * w..(b + 1) * w])
        }
    }

    pub fn swap_lines(&mut self, a: usize, b: usize) {
        if a != b {
            let (x, y) = self.two_lines_mut(a, b);
            x.swap_with_slice(y);
        }
    }

    /// All lines as consecutive `w`-word chunks.
    pub fn raw_mut(&mut self) -> &mut [u64] {
        &mut self.data
    }

    /// Transposes the matrix in place, 64x64 blocks at a time.
    pub fn transpose_in_place(&mut self) {
        let w = self.w;
        let mut a = [0u64; 64];
        let mut b = [0u64; 64];
        for bi in 0..w {
            self.gather(bi, bi, &mut a);
            transpose64(&mut a);
            self.scatter(bi, bi, &a);
            for bj in bi + 1..w {
                self.gather(bi, bj, &mut a);
                self.gather(bj, bi, &mut b);
                transpose64(&mut a);
                transpose64(&mut b);
                self.scatter(bi, bj, &b);
                self.scatter(bj, bi, &a);
            }
        }
    }

    #[inline]
    fn gather(&self, bi: usize, bj: usize, out: &mut [u64; 64]) {
        for (k, o) in out.iter_mut().enumerate() {
            *o = self.data[(bi * 64 + k) * self.w + bj];
        }
    }

    #[inline]
    fn scatter(&mut self, bi: usize, bj: usize, src: &[u64; 64]) {
        for (k, s) in src.iter().enumerate() {
            self.data[(bi * 64 + k) * self.w + bj] = *s;
        }
    }
}

/// Transposes a 64x64 bit block: bit `j` of word `i` moves to bit `i` of
/// word `j`. Recursive block swap, 6 rounds of 32 word pairs.
pub fn transpose64(a: &mut [u64; 64]) {
    let mut j = 32usize;
    let mut m: u64 = 0x0000_0000_FFFF_FFFF;
    while j != 0 {
        let mut k = 0usize;
        while k < 64 {
            let t = ((a[k] >> j) ^ a[k + j]) & m;
            a[k] ^= t << j;
            a[k + j] ^= t;
            k = (k + j + 1) & !j;
        }
        j >>= 1;
        m ^= m << j;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn transpose64_matches_naive() {
        let mut rng = StdRng::seed_from_u64(5);
        let mut a = [0u64; 64];
        for x in a.iter_mut() {
            *x = rng.random();
        }
        let orig = a;
        transpose64(&mut a);
        for (i, row) in orig.iter().enumerate() {
            for (j, col) in a.iter().enumerate() {
                assert_eq!((col >> i) & 1, (row >> j) & 1);
            }
        }
    }

    #[test]
    fn transpose_in_place_matches_naive() {
        let mut rng = StdRng::seed_from_u64(9);
        let np = 192;
        let mut m = BitMatrix::zeros(np);
        for i in 0..np {
            for j in 0..np {
                m.set(i, j, rng.random_bool(0.3));
            }
        }
        let orig = m.clone();
        m.transpose_in_place();
        for i in 0..np {
            for j in 0..np {
                assert_eq!(m.get(i, j), orig.get(j, i));
            }
        }
        m.transpose_in_place();
        assert_eq!(m, orig);
    }

    #[test]
    fn two_lines_both_orders() {
        let mut m = BitMatrix::identity(128);
        let (a, b) = m.two_lines_mut(70, 3);
        assert_eq!(a[1], 1 << 6);
        assert_eq!(b[0], 1 << 3);
        m.swap_lines(3, 70);
        assert!(m.get(3, 70) && m.get(70, 3));
    }
}
