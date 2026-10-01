#![allow(dead_code)]
//! Frozen copy of `bitmatrix.rs` used by the reference tableau.

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
