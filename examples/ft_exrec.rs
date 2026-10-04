//! Per-gadget logical failure rates (1-exRec style) at levels 1 and 2:
//! ft_exrec <level> <p> <trials> <seed> [gadget]
//! Each trial: perfect inputs, a noisy leading EC, the gadget (with its
//! trailing EC), then ideal decoding.
use qsim_lab::ft::backends::FrameBackend;
use qsim_lab::ft::core::Noise;
use qsim_lab::ft::machine::{ideal_logical, FtConfig, Machine};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let k: usize = a[1].parse().unwrap();
    let p: f64 = a[2].parse().unwrap();
    let n: u64 = a[3].parse().unwrap();
    let seed: u64 = a[4].parse().unwrap();
    let names = ["prep0", "prep+", "ec", "h", "s", "id", "cnot", "measZ", "measX", "inject"];
    let only: Option<String> = a.get(5).cloned();
    for (oi, name) in names.iter().enumerate() {
        if let Some(o) = &only {
            if o != name {
                continue;
            }
        }
        let mut m = Machine::new(FrameBackend::default(), Noise::new(p, seed + oi as u64), FtConfig::default(), k);
        let x = m.alloc(k);
        let y = m.alloc(k);
        let mut fail = 0u64;
        let mut locs = 0u64;
        for _ in 0..n {
            m.noise.suspended = true;
            m.prep0(k, x);
            m.prep0(k, y);
            m.noise.suspended = false;
            let l0 = m.noise.loc;
            // leading EC on the input blocks (1-exRec = leading EC + gadget + trailing EC)
            if oi != 9 {
                m.ec(k, x);
                m.ec(k, y);
            }
            let mut flip = false;
            match oi {
                0 => m.prep0(k, x),
                1 => m.prep_plus(k, x),
                2 => {
                    m.ec(k, x);
                }
                3 => m.h(k, x),
                4 => m.s(k, x),
                5 => m.id(k, x),
                6 => m.cnot(k, x, y),
                7 => flip = m.meas_z(k, x),
                8 => {
                    // ideal |+>_L then measure X
                    m.noise.suspended = true;
                    m.prep_plus(k, x);
                    m.noise.suspended = false;
                    flip = m.meas_x(k, x)
                }
                _ => m.inject(k, x),
            }
            locs += m.noise.loc - l0;
            let bad = if oi >= 7 && oi <= 8 {
                flip
            } else {
                let (lx, lz) = ideal_logical(&m.b.frame, k, x);
                let (my, mz) = ideal_logical(&m.b.frame, k, y);
                match oi {
                    0 => lx,
                    1 => lz,
                    9 => lx || lz,
                    _ => lx || lz || my || mz,
                }
            };
            fail += bad as u64;
        }
        println!("level={k} p={p:e} gadget={name} trials={n} fail={fail} rate={:.3e} locs={:.0}", fail as f64 / n as f64, locs as f64 / n as f64);
    }
}
