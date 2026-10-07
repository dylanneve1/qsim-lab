//! Bookkeeping for the long tail-open sampling run (see the doped-Clifford
//! RUNPLAN): seed commitment and the seeded prefix / tail streams (bit for
//! bit the same as `runplan/analyze.py`), job and record lines (flat JSON,
//! no dependencies), and the suffix draw of the tail sampler.
//!
//! Conventions: bitstrings are written q0-first; a job's `prefix_bits` has
//! the `m` open tail qubits as dots (`"......0110..."`, 70 chars); a tail
//! batch entry `j` is the completion with bits `q0..q(m-1)` = `j`, q0 the
//! most significant bit.

use num_complex::Complex64;

/// SHA-256 (FIPS 180-4), for the seed commitment and the seeded streams.
pub fn sha256(msg: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut data = msg.to_vec();
    let bitlen = (msg.len() as u64).wrapping_mul(8);
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&bitlen.to_be_bytes());
    for chunk in data.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, c) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [
                t1.wrapping_add(t2),
                v[0],
                v[1],
                v[2],
                v[3].wrapping_add(t1),
                v[4],
                v[5],
                v[6],
            ];
        }
        for (a, b) in h.iter_mut().zip(v) {
            *a = a.wrapping_add(b);
        }
    }
    let mut out = [0u8; 32];
    for (o, x) in out.chunks_exact_mut(4).zip(h) {
        o.copy_from_slice(&x.to_be_bytes());
    }
    out
}

/// Lower-case hex.
pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// The seed as `analyze.py` reads it: the file's bytes with ASCII
/// whitespace stripped at both ends.
pub fn read_seed(bytes: &[u8]) -> Vec<u8> {
    let s = bytes
        .iter()
        .position(|c| !c.is_ascii_whitespace())
        .unwrap_or(bytes.len());
    let e = bytes
        .iter()
        .rposition(|c| !c.is_ascii_whitespace())
        .map_or(s, |e| e + 1);
    bytes[s..e.max(s)].to_vec()
}

/// `analyze.py stream`: `nbits` bits ('0'/'1') of sha256(seed|tag|i|ctr), MSB first.
pub fn stream(seed: &[u8], tag: &str, i: u64, nbits: usize) -> String {
    let mut out = String::new();
    let mut ctr = 0u64;
    while out.len() < nbits {
        let mut msg = seed.to_vec();
        msg.extend_from_slice(format!("|{tag}|{i}|{ctr}").as_bytes());
        for b in sha256(&msg) {
            out.push_str(&format!("{b:08b}"));
        }
        ctr += 1;
    }
    out.truncate(nbits);
    out
}

/// `analyze.py unif`: a uniform double in [0, 1) from 53 stream bits.
pub fn unif(seed: &[u8], tag: &str, i: u64) -> f64 {
    let s = stream(seed, tag, i, 53);
    u64::from_str_radix(&s, 2).unwrap() as f64 / (1u64 << 53) as f64
}

/// One job of the run.
#[derive(Clone, Debug, PartialEq)]
pub struct Job {
    /// Sample index (production) or `None` for a calibration job.
    pub i: Option<u64>,
    /// SUTD row (calibration) or `None`.
    pub row: Option<u64>,
    /// q0-first, tail as dots.
    pub prefix_bits: String,
    /// Uniform for the suffix draw (samples).
    pub u_tail: Option<f64>,
}

impl Job {
    /// Open tail size (leading dots).
    pub fn tail_m(&self) -> usize {
        self.prefix_bits.chars().take_while(|&c| c == '.').count()
    }
    /// The little-endian suffix integer (bit `i` = qubit `i`; tail bits 0).
    pub fn x(&self) -> u128 {
        self.prefix_bits
            .chars()
            .enumerate()
            .fold(0u128, |a, (i, c)| a | (((c == '1') as u128) << i))
    }
    /// Resume key: `s<i>` or `c<row>`.
    pub fn key(&self) -> String {
        match (self.i, self.row) {
            (Some(i), _) => format!("s{i}"),
            (None, Some(r)) => format!("c{r}"),
            _ => String::from("?"),
        }
    }
    /// Checks the prefix: `n` chars of '0'/'1' after `m >= 1` dots.
    pub fn validate(&self, n: usize) -> Result<(), String> {
        let m = self.tail_m();
        let ok = self.prefix_bits.len() == n
            && m >= 1
            && self.prefix_bits[m..].chars().all(|c| c == '0' || c == '1');
        if !ok {
            return Err(format!(
                "job {}: prefix_bits must be {n} chars, leading dots then 0/1",
                self.key()
            ));
        }
        if self.i.is_none() && self.row.is_none() {
            return Err("job has neither \"i\" nor \"row\"".into());
        }
        if self.i.is_some() && self.u_tail.is_none() {
            return Err(format!("sample job {}: missing u_tail", self.key()));
        }
        Ok(())
    }
}

/// Production jobs `start..start+n` from the seed (= `analyze.py prefixes`).
pub fn seed_jobs(seed: &[u8], m: usize, n_qubits: usize, start: u64, n: u64) -> Vec<Job> {
    (start..start + n)
        .map(|i| Job {
            i: Some(i),
            row: None,
            prefix_bits: ".".repeat(m) + &stream(seed, "prefix", i, n_qubits - m),
            u_tail: Some(unif(seed, "tail", i)),
        })
        .collect()
}

/// The raw text of a field of a flat JSON object line (number, `true`,
/// or a string without escapes), or `None`.
pub fn json_field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("\"{key}\"");
    let mut from = 0;
    while let Some(p) = line[from..].find(&pat) {
        let after = from + p + pat.len();
        let rest = line[after..].trim_start();
        if let Some(rest) = rest.strip_prefix(':') {
            let rest = rest.trim_start();
            if let Some(s) = rest.strip_prefix('"') {
                return s.find('"').map(|e| &s[..e]);
            }
            let e = rest
                .find(|c: char| c == ',' || c == '}' || c.is_whitespace())
                .unwrap_or(rest.len());
            return Some(&rest[..e]);
        }
        from = after;
    }
    None
}

/// Parses a job line (`analyze.py prefixes` / `calrows` output).
pub fn parse_job(line: &str) -> Option<Job> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    Some(Job {
        i: json_field(line, "i").and_then(|v| v.parse().ok()),
        row: json_field(line, "row").and_then(|v| v.parse().ok()),
        prefix_bits: json_field(line, "prefix_bits")?.to_string(),
        u_tail: json_field(line, "u_tail").and_then(|v| v.parse().ok()),
    })
}

/// Suffix draw of the tail sampler: the smallest `j` whose cumulative
/// weight `Σ_{k<=j} |l_k|^2` exceeds `u · Σ |l|^2`.
pub fn draw(amps: &[Complex64], u: f64) -> usize {
    let tot: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
    let mut c = 0.0;
    for (j, a) in amps.iter().enumerate() {
        c += a.norm_sqr();
        if c > u * tot {
            return j;
        }
    }
    amps.len() - 1
}

/// The full q0-first bitstring for tail entry `j` of a job.
pub fn bitstring(job: &Job, j: usize) -> String {
    let m = job.tail_m();
    let head: String = (0..m)
        .map(|i| {
            if (j >> (m - 1 - i)) & 1 == 1 {
                '1'
            } else {
                '0'
            }
        })
        .collect();
    head + &job.prefix_bits[m..]
}

/// f64 as JSON (shortest round-trip form; non-finite as null).
pub fn jnum(x: f64) -> String {
    if x.is_finite() {
        format!("{x:?}")
    } else {
        "null".into()
    }
}

/// Amplitudes as a JSON array of `[re, im]` pairs (full f64 precision).
pub fn jamps(amps: &[Complex64]) -> String {
    let v: Vec<String> = amps
        .iter()
        .map(|a| format!("[{},{}]", jnum(a.re), jnum(a.im)))
        .collect();
    format!("[{}]", v.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_vectors() {
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let long = vec![b'a'; 1000];
        assert_eq!(
            hex(&sha256(&long)),
            "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3"
        );
    }

    /// Same streams as `analyze.py` (values computed with Python hashlib).
    #[test]
    fn streams_match_analyze_py() {
        let seed = read_seed(b"  deadbeef\n");
        assert_eq!(seed, b"deadbeef");
        // python: analyze.stream(b'deadbeef', 'prefix', 3, 300), analyze.unif(b'deadbeef', 'tail', 3)
        let want = "110011100110101111111011111111000001011111111101000111000111101110101101111010101111000001000010011111001010011111001100100111001000011010110111000010000100111100001110100100010011011000011001100001011011001001001001011010111011011010001110011001110111100000001010100000100011101110100011011011011001";
        assert_eq!(stream(&seed, "prefix", 3, 300), want);
        assert_eq!(unif(&seed, "tail", 3), 0.05918210170950411);
    }

    #[test]
    fn jobs_and_draw() {
        let j = parse_job(r#"{"i": 7, "prefix_bits": "..0101", "u_tail": 0.25}"#).unwrap();
        assert_eq!(j.i, Some(7));
        assert_eq!(j.tail_m(), 2);
        assert_eq!(j.x(), 0b101000);
        assert!(j.validate(6).is_ok());
        let c = parse_job(r#"{"row": 12, "prefix_bits": "...1"}"#).unwrap();
        assert_eq!((c.i, c.row, c.key().as_str()), (None, Some(12), "c12"));
        let a = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 1.0),
        ];
        assert_eq!(draw(&a, 0.0), 0);
        assert_eq!(draw(&a, 0.3), 1);
        assert_eq!(draw(&a, 0.6), 3);
        assert_eq!(bitstring(&j, 2), "100101");
    }
}
