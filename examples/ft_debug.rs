use qsim_lab::ft::core::Noise;
use qsim_lab::ft::logical::{Encoded, Logical, MagicMode, Unencoded};
use qsim_lab::ft::machine::FtConfig;
fn main() {
    // controlled-swap on |c=+>|a=1>|b=0>: compare
    let mut e = Encoded::frame(1, 3, Noise::new(0.0, 1), FtConfig::default(), MagicMode::Model(0.0), 1);
    let mut u = Unencoded::new(3, Noise::new(0.0, 1), false, 1);
    for l in [&mut e as &mut dyn Logical, &mut u as &mut dyn Logical] {
        l.prep(0, false);
        l.prep(1, true);
        l.prep(2, false);
        l.h(0);
        l.t(1);
        l.cswap(0, 1, 2);
    }
    let sv = e.sv.as_ref().unwrap();
    for i in 0..16 {
        let a = sv.a[i];
        if a.norm() > 1e-9 { println!("enc {i:04b} {a}"); }
    }
    for i in 0..8 {
        let a = u.sv.a[i];
        if a.norm() > 1e-9 { println!("unenc {i:03b} {a}"); }
    }
}
