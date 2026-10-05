use qsim_lab::ft::core::Noise;
use qsim_lab::ft::logical::{Checked, Encoded, MagicMode};
use qsim_lab::ft::machine::FtConfig;
use qsim_lab::ft::shor::{run_shor15, NLOG15};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let p: f64 = a[1].parse().unwrap();
    let n: u64 = a[2].parse().unwrap();
    let seed: u64 = a[3].parse().unwrap();
    let mut h = [0u64; 8];
    let mut hf = [0u64; 8];
    for s in 0..n {
        let sd = seed * 1_000_003 + s;
        let mut c = Checked(Encoded::frame(
            1,
            NLOG15,
            Noise::new(p, sd),
            FtConfig::default(),
            MagicMode::Raw,
            sd,
        ));
        let y = run_shor15(&mut c, 7, 3);
        if c.0.counts.logical_fault {
            hf[y as usize] += 1
        } else {
            h[y as usize] += 1
        }
    }
    println!("p={p} clean={:?} faulty={:?}", h, hf);
}
