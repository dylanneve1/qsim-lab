//! Reading and writing Stim's `.stim` circuit format (a Clifford + Pauli
//! noise subset), so that qsim-lab and Stim can sample the *same* circuit.
//!
//! * [`to_stim`] serialises a [`Circuit`] plus a [`NoiseModel`] by walking
//!   the actual op list and making every implicit noise location of the model
//!   explicit, in the order [`Circuit::run_noisy`] and the SymPhase sampler
//!   use: `DEPOLARIZE1`/`DEPOLARIZE2` after each gate, `M(p)` for a readout
//!   flip, `X_ERROR(p)` after each `R`. Detectors and observables (absolute
//!   measurement indices) become `DETECTOR`/`OBSERVABLE_INCLUDE` lines with
//!   `rec[-k]` lookbacks at the end of the file.
//! * [`parse_stim`] reads a `.stim` file (with `REPEAT` blocks, as written by
//!   `stim.Circuit.generated`) into a [`Circuit`] whose noise is explicit
//!   [`Op`]s, plus a [`NoiseModel`] that only carries the readout flip
//!   probability (Stim's `M(p)`; it must be the same for every measurement).
//!
//! Supported instructions: `I H S S_DAG SQRT_Z SQRT_Z_DAG X Y Z CX CNOT ZCX
//! CZ ZCZ SWAP R RZ RX M MZ MX MR MRZ MRX X_ERROR Y_ERROR Z_ERROR DEPOLARIZE1
//! DEPOLARIZE2 DETECTOR OBSERVABLE_INCLUDE TICK QUBIT_COORDS SHIFT_COORDS
//! REPEAT`. Anything else is an error, never silently dropped.
//!
//! Consecutive ops of one kind on disjoint qubits are written on one line
//! (`CX 0 1 2 3`); that is the same circuit, because ops on disjoint qubits
//! commute, and it lets Stim process them as one instruction.

use crate::circuit::{Circuit, Op};
use crate::gate::Gate;
use crate::noise::NoiseModel;
use std::f64::consts::{FRAC_PI_2, PI};
use std::fmt::Write as _;

/// A parsed `.stim` program.
#[derive(Clone, Debug, PartialEq)]
pub struct StimProgram {
    /// Gates, measurements, resets and explicit noise ops.
    pub circuit: Circuit,
    /// Only `p_meas` can be non-zero (from `M(p)`); gate and reset noise are
    /// explicit ops in `circuit`.
    pub noise: NoiseModel,
    /// Each detector as a set of absolute measurement indices.
    pub detectors: Vec<Vec<usize>>,
    /// `observables[k]` = measurement indices of `OBSERVABLE_INCLUDE(k)`.
    pub observables: Vec<Vec<usize>>,
}

/// Error from [`parse_stim`] or [`to_stim`].
#[derive(Clone, Debug, PartialEq)]
pub struct StimError(pub String);

impl std::fmt::Display for StimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stim format: {}", self.0)
    }
}

impl std::error::Error for StimError {}

fn err<T>(msg: impl Into<String>) -> Result<T, StimError> {
    Err(StimError(msg.into()))
}

// ---------------------------------------------------------------- writer

/// One output "item": instruction name (with argument) and its targets.
struct Item {
    name: String,
    targets: Vec<usize>,
}

/// Stim name of a Clifford gate (`None` for an identity that still carries
/// a noise location).
fn stim_gate(g: &Gate) -> Result<(&'static str, Vec<usize>), StimError> {
    use Gate::*;
    let quarter = |t: f64| -> Result<i64, StimError> {
        let k = t / FRAC_PI_2;
        if (k - k.round()).abs() > 1e-9 {
            return err(format!("non-Clifford angle in {g:?}"));
        }
        Ok((k.round() as i64).rem_euclid(4))
    };
    Ok(match *g {
        I(a) => ("I", vec![a]),
        H(a) => ("H", vec![a]),
        X(a) => ("X", vec![a]),
        Y(a) => ("Y", vec![a]),
        Z(a) => ("Z", vec![a]),
        S(a) => ("S", vec![a]),
        Sdg(a) => ("S_DAG", vec![a]),
        Sx(a) => ("SQRT_X", vec![a]),
        Sxdg(a) => ("SQRT_X_DAG", vec![a]),
        Phase(a, t) | Rz(a, t) => {
            let name = ["I", "S", "Z", "S_DAG"][quarter(t)? as usize];
            (name, vec![a])
        }
        Cnot(c, t) => ("CX", vec![c, t]),
        Cz(a, b) => ("CZ", vec![a, b]),
        Swap(a, b) => ("SWAP", vec![a, b]),
        CPhase(a, b, t) => {
            let k = t / PI;
            if (k - k.round()).abs() > 1e-9 {
                return err(format!("non-Clifford angle in {g:?}"));
            }
            if (k.round() as i64).rem_euclid(2) == 1 {
                ("CZ", vec![a, b])
            } else {
                // identity on two qubits; keep the noise location
                ("I", vec![a, b])
            }
        }
        _ => return err(format!("gate {g:?} is not supported by the .stim writer")),
    })
}

fn fmt_p(p: f64) -> String {
    // shortest repr that round-trips exactly
    format!("{p:?}")
}

/// Serialises `circuit` under `noise` (implicit noise made explicit) with the
/// given detectors and observables (absolute measurement indices).
pub fn to_stim(
    circuit: &Circuit,
    noise: &NoiseModel,
    detectors: &[Vec<usize>],
    observables: &[Vec<usize>],
) -> Result<String, StimError> {
    let mut items: Vec<Item> = Vec::new();
    let mut num_meas = 0usize;
    let push = |items: &mut Vec<Item>, name: String, targets: Vec<usize>| {
        items.push(Item { name, targets });
    };
    for op in &circuit.ops {
        match *op {
            Op::Gate(g) => {
                let (name, qs) = stim_gate(&g)?;
                let qs2 = qs.clone();
                if name == "I" && qs.len() == 2 {
                    // two-qubit identity: two I's, then the 2q noise
                    push(&mut items, "I".into(), vec![qs[0]]);
                    push(&mut items, "I".into(), vec![qs[1]]);
                } else {
                    push(&mut items, name.into(), qs);
                }
                match qs2.len() {
                    1 if noise.p_1q > 0.0 => push(
                        &mut items,
                        format!("DEPOLARIZE1({})", fmt_p(noise.p_1q)),
                        qs2,
                    ),
                    2 if noise.p_2q > 0.0 => push(
                        &mut items,
                        format!("DEPOLARIZE2({})", fmt_p(noise.p_2q)),
                        qs2,
                    ),
                    _ => {}
                }
            }
            Op::Measure(q) => {
                let name = if noise.p_meas > 0.0 {
                    format!("M({})", fmt_p(noise.p_meas))
                } else {
                    "M".into()
                };
                push(&mut items, name, vec![q]);
                num_meas += 1;
            }
            Op::Reset(q) => {
                push(&mut items, "R".into(), vec![q]);
                if noise.p_reset > 0.0 {
                    push(
                        &mut items,
                        format!("X_ERROR({})", fmt_p(noise.p_reset)),
                        vec![q],
                    );
                }
            }
            Op::ClassicControlled { .. } => {
                return err("classically controlled ops are not supported by the .stim writer")
            }
            Op::XFlip(q, p) => push(&mut items, format!("X_ERROR({})", fmt_p(p)), vec![q]),
            Op::YFlip(q, p) => push(&mut items, format!("Y_ERROR({})", fmt_p(p)), vec![q]),
            Op::ZFlip(q, p) => push(&mut items, format!("Z_ERROR({})", fmt_p(p)), vec![q]),
            Op::Depolarize1q(q, p) => {
                push(&mut items, format!("DEPOLARIZE1({})", fmt_p(p)), vec![q])
            }
            Op::Depolarize2q(a, b, p) => {
                push(&mut items, format!("DEPOLARIZE2({})", fmt_p(p)), vec![a, b])
            }
        }
    }

    // Batch into lines. A *layer* is a maximal run of items whose qubits are
    // pairwise disjoint *across different groups*: group consecutive
    // (gate, its noise) pairs so that one line per instruction name is
    // written in first-appearance order. Within a layer all ops act on
    // disjoint qubit sets except the noise that follows its own gate, so the
    // reordering only commutes ops on disjoint qubits.
    let mut out = String::new();
    let mut i = 0;
    while i < items.len() {
        // unit = a gate/measure/reset item plus any directly following noise
        // items on exactly the same qubits
        let unit_end = |s: usize| -> usize {
            let mut e = s + 1;
            while e < items.len()
                && is_noise(&items[e].name)
                && !is_noise(&items[s].name)
                && items[e].targets == items[s].targets
            {
                e += 1;
            }
            e
        };
        let first_end = unit_end(i);
        let shape: Vec<&str> = items[i..first_end]
            .iter()
            .map(|it| it.name.as_str())
            .collect();
        let mut used: Vec<usize> = items[i..first_end]
            .iter()
            .flat_map(|it| it.targets.iter().copied())
            .collect();
        used.sort_unstable();
        used.dedup();
        let mut units = vec![(i, first_end)];
        let mut j = first_end;
        while j < items.len() {
            let e = unit_end(j);
            let same_shape =
                e - j == shape.len() && items[j..e].iter().zip(&shape).all(|(it, s)| it.name == *s);
            if !same_shape {
                break;
            }
            let qs = &items[j].targets;
            if qs.iter().any(|q| used.binary_search(q).is_ok()) {
                break;
            }
            for &q in qs {
                let pos = used.binary_search(&q).unwrap_err();
                used.insert(pos, q);
            }
            units.push((j, e));
            j = e;
        }
        for (k, name) in shape.iter().enumerate() {
            let mut line = String::from(*name);
            for &(s, _) in &units {
                for q in &items[s + k].targets {
                    write!(line, " {q}").unwrap();
                }
            }
            out.push_str(&line);
            out.push('\n');
        }
        i = j;
    }

    let num_meas_i = num_meas as isize;
    let rec = |m: usize| -> Result<String, StimError> {
        if m >= num_meas {
            return err(format!("record {m} out of range ({num_meas} measurements)"));
        }
        Ok(format!(" rec[{}]", m as isize - num_meas_i))
    };
    for det in detectors {
        let mut line = String::from("DETECTOR");
        for &m in det {
            line.push_str(&rec(m)?);
        }
        out.push_str(&line);
        out.push('\n');
    }
    for (k, obs) in observables.iter().enumerate() {
        let mut line = format!("OBSERVABLE_INCLUDE({k})");
        for &m in obs {
            line.push_str(&rec(m)?);
        }
        out.push_str(&line);
        out.push('\n');
    }
    Ok(out)
}

fn is_noise(name: &str) -> bool {
    name.starts_with("DEPOLARIZE") || name.contains("_ERROR")
}

// ------------------------------------------------- reference parser (old)
//
// The original line-by-line parser: it re-parses every line of a REPEAT body
// on every iteration. Kept (hidden) as the reference the fast parser below is
// tested against.

struct Parser {
    ops: Vec<Op>,
    max_qubit: Option<usize>,
    num_meas: usize,
    meas_p: Option<f64>,
    detectors: Vec<Vec<usize>>,
    observables: Vec<Vec<usize>>,
}

/// One instruction line, already split.
struct Line<'a> {
    name: String,
    args: Vec<f64>,
    targets: Vec<&'a str>,
}

fn split_line(raw: &str) -> Result<Option<Line<'_>>, StimError> {
    let s = raw.split('#').next().unwrap().trim();
    if s.is_empty() {
        return Ok(None);
    }
    // name, optional [tag], optional (args), targets
    let name_end = s
        .find(|c: char| c == '(' || c == '[' || c.is_whitespace())
        .unwrap_or(s.len());
    let name = s[..name_end].to_ascii_uppercase();
    let mut rest = &s[name_end..];
    if rest.starts_with('[') {
        let close = rest
            .find(']')
            .ok_or_else(|| StimError(format!("bad tag: {raw}")))?;
        rest = &rest[close + 1..];
    }
    let mut args = Vec::new();
    if rest.starts_with('(') {
        let close = rest
            .find(')')
            .ok_or_else(|| StimError(format!("bad args: {raw}")))?;
        for a in rest[1..close].split(',') {
            let a = a.trim();
            if !a.is_empty() {
                args.push(
                    a.parse::<f64>()
                        .map_err(|_| StimError(format!("bad argument {a:?} in {raw:?}")))?,
                );
            }
        }
        rest = &rest[close + 1..];
    }
    let targets = rest.split_whitespace().collect();
    Ok(Some(Line {
        name,
        args,
        targets,
    }))
}

impl Parser {
    fn qubit(&mut self, t: &str) -> Result<usize, StimError> {
        let q: usize = t
            .parse()
            .map_err(|_| StimError(format!("unsupported target {t:?}")))?;
        self.max_qubit = Some(self.max_qubit.map_or(q, |m| m.max(q)));
        Ok(q)
    }

    fn rec(&self, t: &str) -> Result<usize, StimError> {
        let inner = t
            .strip_prefix("rec[")
            .and_then(|r| r.strip_suffix(']'))
            .ok_or_else(|| StimError(format!("expected rec[-k], got {t:?}")))?;
        let k: isize = inner
            .parse()
            .map_err(|_| StimError(format!("bad record target {t:?}")))?;
        if k >= 0 || (-k) as usize > self.num_meas {
            return err(format!(
                "record {t} out of range ({} so far)",
                self.num_meas
            ));
        }
        Ok((self.num_meas as isize + k) as usize)
    }

    fn prob(line: &Line) -> Result<f64, StimError> {
        match line.args.as_slice() {
            [p] if (0.0..=1.0).contains(p) => Ok(*p),
            _ => err(format!("{} needs one probability", line.name)),
        }
    }

    fn measure(&mut self, q: usize, p: f64) -> Result<(), StimError> {
        match self.meas_p {
            None => self.meas_p = Some(p),
            Some(prev) if prev == p => {}
            Some(prev) => {
                return err(format!(
                    "measurement flip probabilities differ ({prev} vs {p}); qsim-lab's \
                     NoiseModel has one p_meas"
                ))
            }
        }
        self.ops.push(Op::Measure(q));
        self.num_meas += 1;
        Ok(())
    }

    fn instruction(&mut self, line: &Line) -> Result<(), StimError> {
        let name = line.name.as_str();
        let one_q = |g: fn(usize) -> Gate| g;
        let gate1: Option<fn(usize) -> Gate> = match name {
            "I" => Some(one_q(Gate::I)),
            "H" | "H_XZ" => Some(one_q(Gate::H)),
            "X" => Some(one_q(Gate::X)),
            "Y" => Some(one_q(Gate::Y)),
            "Z" => Some(one_q(Gate::Z)),
            "S" | "SQRT_Z" => Some(one_q(Gate::S)),
            "S_DAG" | "SQRT_Z_DAG" => Some(one_q(Gate::Sdg)),
            _ => None,
        };
        if let Some(g) = gate1 {
            if !line.args.is_empty() {
                return err(format!("{name} takes no arguments"));
            }
            for t in &line.targets {
                let q = self.qubit(t)?;
                if name != "I" {
                    self.ops.push(Op::Gate(g(q)));
                }
            }
            return Ok(());
        }
        let gate2: Option<fn(usize, usize) -> Gate> = match name {
            "CX" | "CNOT" | "ZCX" => Some(Gate::Cnot),
            "CZ" | "ZCZ" => Some(Gate::Cz),
            "SWAP" => Some(Gate::Swap),
            _ => None,
        };
        if let Some(g) = gate2 {
            if !line.targets.len().is_multiple_of(2) || !line.args.is_empty() {
                return err(format!("bad {name} line"));
            }
            for pair in line.targets.chunks(2) {
                let a = self.qubit(pair[0])?;
                let b = self.qubit(pair[1])?;
                if a == b {
                    return err(format!("{name} on a repeated qubit {a}"));
                }
                self.ops.push(Op::Gate(g(a, b)));
            }
            return Ok(());
        }
        match name {
            "TICK" | "QUBIT_COORDS" | "SHIFT_COORDS" => Ok(()),
            "R" | "RZ" | "RX" => {
                for t in &line.targets {
                    let q = self.qubit(t)?;
                    self.ops.push(Op::Reset(q));
                    if name == "RX" {
                        self.ops.push(Op::Gate(Gate::H(q)));
                    }
                }
                Ok(())
            }
            "M" | "MZ" | "MX" | "MR" | "MRZ" | "MRX" => {
                let p = match line.args.as_slice() {
                    [] => 0.0,
                    [p] => *p,
                    _ => return err(format!("bad {name} arguments")),
                };
                let x_basis = name.ends_with('X');
                let reset = name.starts_with("MR");
                for t in &line.targets {
                    if t.starts_with('!') {
                        return err("inverted measurement targets are not supported");
                    }
                    let q = self.qubit(t)?;
                    if x_basis {
                        self.ops.push(Op::Gate(Gate::H(q)));
                    }
                    self.measure(q, p)?;
                    if reset {
                        self.ops.push(Op::Reset(q));
                    }
                    if x_basis {
                        self.ops.push(Op::Gate(Gate::H(q)));
                    }
                }
                Ok(())
            }
            "X_ERROR" | "Y_ERROR" | "Z_ERROR" | "DEPOLARIZE1" => {
                let p = Self::prob(line)?;
                for t in &line.targets {
                    let q = self.qubit(t)?;
                    self.ops.push(match name {
                        "X_ERROR" => Op::XFlip(q, p),
                        "Y_ERROR" => Op::YFlip(q, p),
                        "Z_ERROR" => Op::ZFlip(q, p),
                        _ => Op::Depolarize1q(q, p),
                    });
                }
                Ok(())
            }
            "DEPOLARIZE2" => {
                let p = Self::prob(line)?;
                if !line.targets.len().is_multiple_of(2) {
                    return err("DEPOLARIZE2 needs pairs");
                }
                for pair in line.targets.chunks(2) {
                    let a = self.qubit(pair[0])?;
                    let b = self.qubit(pair[1])?;
                    self.ops.push(Op::Depolarize2q(a, b, p));
                }
                Ok(())
            }
            "DETECTOR" => {
                let mut recs = Vec::new();
                for t in &line.targets {
                    recs.push(self.rec(t)?);
                }
                self.detectors.push(recs);
                Ok(())
            }
            "OBSERVABLE_INCLUDE" => {
                let k = match line.args.as_slice() {
                    [k] if *k >= 0.0 && k.fract() == 0.0 => *k as usize,
                    _ => return err("OBSERVABLE_INCLUDE needs an index"),
                };
                if self.observables.len() <= k {
                    self.observables.resize(k + 1, Vec::new());
                }
                for t in &line.targets {
                    let m = self.rec(t)?;
                    self.observables[k].push(m);
                }
                Ok(())
            }
            _ => err(format!("unsupported instruction {name}")),
        }
    }

    fn block(&mut self, lines: &[&str]) -> Result<(), StimError> {
        let mut i = 0;
        while i < lines.len() {
            let raw = lines[i];
            let trimmed = raw.split('#').next().unwrap().trim();
            if trimmed.is_empty() {
                i += 1;
                continue;
            }
            if trimmed == "}" {
                return err("unbalanced }");
            }
            if trimmed.to_ascii_uppercase().starts_with("REPEAT") && trimmed.ends_with('{') {
                let count: usize = trimmed[6..trimmed.len() - 1]
                    .trim()
                    .rsplit(' ')
                    .next()
                    .unwrap_or("")
                    .parse()
                    .map_err(|_| StimError(format!("bad REPEAT line {raw:?}")))?;
                // find matching brace
                let mut depth = 1;
                let mut j = i + 1;
                while j < lines.len() {
                    let t = lines[j].split('#').next().unwrap().trim();
                    if t.ends_with('{') {
                        depth += 1;
                    } else if t == "}" {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    j += 1;
                }
                if j == lines.len() {
                    return err("unterminated REPEAT block");
                }
                for _ in 0..count {
                    self.block(&lines[i + 1..j])?;
                }
                i = j + 1;
                continue;
            }
            if let Some(line) = split_line(raw)? {
                self.instruction(&line)?;
            }
            i += 1;
        }
        Ok(())
    }
}

/// The original `.stim` parser (same subset and result as [`parse_stim`]),
/// kept as the reference in tests.
#[doc(hidden)]
pub fn parse_stim_reference(text: &str) -> Result<StimProgram, StimError> {
    let mut p = Parser {
        ops: Vec::new(),
        max_qubit: None,
        num_meas: 0,
        meas_p: None,
        detectors: Vec::new(),
        observables: Vec::new(),
    };
    let lines: Vec<&str> = text.lines().collect();
    p.block(&lines)?;
    let n = p.max_qubit.map_or(0, |m| m + 1);
    let noise = NoiseModel {
        p_meas: p.meas_p.unwrap_or(0.0),
        ..NoiseModel::none()
    };
    Ok(StimProgram {
        circuit: Circuit {
            num_qubits: n,
            ops: p.ops,
        },
        noise,
        detectors: p.detectors,
        observables: p.observables,
    })
}

// ------------------------------------------------------------ fast parser
//
// Each line is parsed once; REPEAT blocks are kept as blocks (not unrolled),
// and [`StimCircuit::for_each_op`] / [`StimCircuit::for_each_op_rev`] unroll
// them on the fly. Same subset and same semantics as the reference parser
// (`tests/core/stim_io.rs` checks `parse_stim == parse_stim_reference` on
// Stim's circuit zoo and on random programs).

/// Single-qubit gates of the subset, after aliasing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum G1 {
    H,
    S,
    Sdg,
    X,
    Y,
    Z,
}

/// Two-qubit gates of the subset, after aliasing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum G2 {
    Cx,
    Cz,
    Swap,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Gate1(G1),
    Gate2(G2),
    /// `R`/`RZ` (`x = false`) or `RX`.
    Reset {
        x: bool,
    },
    /// `M`/`MZ`, `MX`, `MR`/`MRZ`, `MRX`.
    Measure {
        x: bool,
        reset: bool,
    },
    XErr,
    YErr,
    ZErr,
    Depol1,
    Depol2,
    Detector,
    Observable(u32),
    Repeat {
        count: u64,
        block: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Inst {
    kind: Kind,
    /// Probability argument (noise channels, measurement flips).
    p: f64,
    /// Targets are `targets[start..end]`: qubits, or `k` of `rec[-k]`.
    start: u32,
    end: u32,
}

/// A parsed `.stim` program (the subset in the module docs) with `REPEAT`
/// blocks kept as blocks: every line is parsed once, and the walkers
/// [`Self::for_each_op`] / [`Self::for_each_op_rev`] unroll the blocks on
/// the fly. [`parse_stim`] is `parse_stim_circuit(text)?.to_program()`.
#[derive(Clone, Debug, PartialEq)]
pub struct StimCircuit {
    /// `blocks[0]` is the top level; `Kind::Repeat` refers to a body block.
    blocks: Vec<Vec<Inst>>,
    targets: Vec<u32>,
    num_qubits: usize,
    num_measurements: usize,
    num_detectors: usize,
    num_observables: usize,
    max_lookback: usize,
}

/// One primitive operation of a [`StimCircuit`], in program order (or in
/// reverse for [`StimCircuit::for_each_op_rev`]). Multi-target instructions
/// are split per target, X-basis measurements and resets become `H`
/// conjugations, `MR` becomes `M` then `R`: exactly the [`Op`] sequence
/// [`parse_stim`] produces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StimOp<'a> {
    /// A Clifford gate (`H S S_DAG X Y Z CX CZ SWAP` and their aliases).
    Gate(Gate),
    /// Z-basis measurement number `index` (in program order) of `qubit`,
    /// with readout flip probability `p`.
    Measure {
        /// Measured qubit.
        qubit: usize,
        /// Position of this outcome in the measurement record.
        index: usize,
        /// Probability of reporting the flipped outcome (`M(p)`).
        p: f64,
    },
    /// Reset to |0>.
    Reset(usize),
    /// `X_ERROR(p)` on one qubit.
    XFlip(usize, f64),
    /// `Y_ERROR(p)` on one qubit.
    YFlip(usize, f64),
    /// `Z_ERROR(p)` on one qubit.
    ZFlip(usize, f64),
    /// `DEPOLARIZE1(p)` on one qubit.
    Depolarize1(usize, f64),
    /// `DEPOLARIZE2(p)` on one pair.
    Depolarize2(usize, usize, f64),
    /// Detector number `row`: the parity of measurements `base - k` for
    /// every `k` in `lookbacks` (`rec[-k]`).
    Detector {
        /// Detector index (program order).
        row: usize,
        /// Number of measurements before this instruction.
        base: usize,
        /// The `k` of each `rec[-k]` target.
        lookbacks: &'a [u32],
    },
    /// `OBSERVABLE_INCLUDE(index)` of measurements `base - k`.
    Observable {
        /// Observable index.
        index: usize,
        /// Number of measurements before this instruction.
        base: usize,
        /// The `k` of each `rec[-k]` target.
        lookbacks: &'a [u32],
    },
}

/// Walker state: measurements and detectors so far.
struct Walk {
    m: usize,
    det: usize,
}

fn gate1(g: G1, q: usize) -> Gate {
    match g {
        G1::H => Gate::H(q),
        G1::S => Gate::S(q),
        G1::Sdg => Gate::Sdg(q),
        G1::X => Gate::X(q),
        G1::Y => Gate::Y(q),
        G1::Z => Gate::Z(q),
    }
}

fn gate2(g: G2, a: usize, b: usize) -> Gate {
    match g {
        G2::Cx => Gate::Cnot(a, b),
        G2::Cz => Gate::Cz(a, b),
        G2::Swap => Gate::Swap(a, b),
    }
}

impl StimCircuit {
    /// Number of qubits (largest qubit target + 1; `QUBIT_COORDS` targets
    /// are not counted).
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Number of measurements (with `REPEAT` blocks unrolled).
    pub fn num_measurements(&self) -> usize {
        self.num_measurements
    }

    /// Number of `DETECTOR`s (with `REPEAT` blocks unrolled).
    pub fn num_detectors(&self) -> usize {
        self.num_detectors
    }

    /// Number of observables (largest `OBSERVABLE_INCLUDE` index + 1).
    pub fn num_observables(&self) -> usize {
        self.num_observables
    }

    /// Largest `k` of any `rec[-k]` target (0 if there is none): a
    /// detector or observable never reads further back than this.
    pub fn max_lookback(&self) -> usize {
        self.max_lookback
    }

    /// Calls `f` on every primitive operation in program order.
    pub fn for_each_op<F: FnMut(StimOp<'_>)>(&self, mut f: F) {
        let mut w = Walk { m: 0, det: 0 };
        self.walk(0, &mut w, &mut f);
    }

    /// Calls `f` on every primitive operation in reverse program order
    /// (`Measure::index` and `Detector::row` count down from the totals).
    pub fn for_each_op_rev<F: FnMut(StimOp<'_>)>(&self, mut f: F) {
        let mut w = Walk {
            m: self.num_measurements,
            det: self.num_detectors,
        };
        self.walk_rev(0, &mut w, &mut f);
    }

    fn walk<F: FnMut(StimOp<'_>)>(&self, b: usize, w: &mut Walk, f: &mut F) {
        for inst in &self.blocks[b] {
            let ts = &self.targets[inst.start as usize..inst.end as usize];
            let p = inst.p;
            match inst.kind {
                Kind::Repeat { count, block } => {
                    for _ in 0..count {
                        self.walk(block as usize, w, f);
                    }
                }
                Kind::Gate1(g) => ts
                    .iter()
                    .for_each(|&q| f(StimOp::Gate(gate1(g, q as usize)))),
                Kind::Gate2(g) => ts
                    .chunks_exact(2)
                    .for_each(|t| f(StimOp::Gate(gate2(g, t[0] as usize, t[1] as usize)))),
                Kind::Reset { x } => {
                    for &q in ts {
                        f(StimOp::Reset(q as usize));
                        if x {
                            f(StimOp::Gate(Gate::H(q as usize)));
                        }
                    }
                }
                Kind::Measure { x, reset } => {
                    for &q in ts {
                        let q = q as usize;
                        if x {
                            f(StimOp::Gate(Gate::H(q)));
                        }
                        f(StimOp::Measure {
                            qubit: q,
                            index: w.m,
                            p,
                        });
                        w.m += 1;
                        if reset {
                            f(StimOp::Reset(q));
                        }
                        if x {
                            f(StimOp::Gate(Gate::H(q)));
                        }
                    }
                }
                Kind::XErr => ts.iter().for_each(|&q| f(StimOp::XFlip(q as usize, p))),
                Kind::YErr => ts.iter().for_each(|&q| f(StimOp::YFlip(q as usize, p))),
                Kind::ZErr => ts.iter().for_each(|&q| f(StimOp::ZFlip(q as usize, p))),
                Kind::Depol1 => ts
                    .iter()
                    .for_each(|&q| f(StimOp::Depolarize1(q as usize, p))),
                Kind::Depol2 => ts
                    .chunks_exact(2)
                    .for_each(|t| f(StimOp::Depolarize2(t[0] as usize, t[1] as usize, p))),
                Kind::Detector => {
                    f(StimOp::Detector {
                        row: w.det,
                        base: w.m,
                        lookbacks: ts,
                    });
                    w.det += 1;
                }
                Kind::Observable(k) => f(StimOp::Observable {
                    index: k as usize,
                    base: w.m,
                    lookbacks: ts,
                }),
            }
        }
    }

    fn walk_rev<F: FnMut(StimOp<'_>)>(&self, b: usize, w: &mut Walk, f: &mut F) {
        for inst in self.blocks[b].iter().rev() {
            let ts = &self.targets[inst.start as usize..inst.end as usize];
            let p = inst.p;
            match inst.kind {
                Kind::Repeat { count, block } => {
                    for _ in 0..count {
                        self.walk_rev(block as usize, w, f);
                    }
                }
                Kind::Gate1(g) => ts
                    .iter()
                    .rev()
                    .for_each(|&q| f(StimOp::Gate(gate1(g, q as usize)))),
                Kind::Gate2(g) => ts
                    .chunks_exact(2)
                    .rev()
                    .for_each(|t| f(StimOp::Gate(gate2(g, t[0] as usize, t[1] as usize)))),
                Kind::Reset { x } => {
                    for &q in ts.iter().rev() {
                        if x {
                            f(StimOp::Gate(Gate::H(q as usize)));
                        }
                        f(StimOp::Reset(q as usize));
                    }
                }
                Kind::Measure { x, reset } => {
                    for &q in ts.iter().rev() {
                        let q = q as usize;
                        if x {
                            f(StimOp::Gate(Gate::H(q)));
                        }
                        if reset {
                            f(StimOp::Reset(q));
                        }
                        w.m -= 1;
                        f(StimOp::Measure {
                            qubit: q,
                            index: w.m,
                            p,
                        });
                        if x {
                            f(StimOp::Gate(Gate::H(q)));
                        }
                    }
                }
                Kind::XErr => ts
                    .iter()
                    .rev()
                    .for_each(|&q| f(StimOp::XFlip(q as usize, p))),
                Kind::YErr => ts
                    .iter()
                    .rev()
                    .for_each(|&q| f(StimOp::YFlip(q as usize, p))),
                Kind::ZErr => ts
                    .iter()
                    .rev()
                    .for_each(|&q| f(StimOp::ZFlip(q as usize, p))),
                Kind::Depol1 => ts
                    .iter()
                    .rev()
                    .for_each(|&q| f(StimOp::Depolarize1(q as usize, p))),
                Kind::Depol2 => ts
                    .chunks_exact(2)
                    .rev()
                    .for_each(|t| f(StimOp::Depolarize2(t[0] as usize, t[1] as usize, p))),
                Kind::Detector => {
                    w.det -= 1;
                    f(StimOp::Detector {
                        row: w.det,
                        base: w.m,
                        lookbacks: ts,
                    });
                }
                Kind::Observable(k) => f(StimOp::Observable {
                    index: k as usize,
                    base: w.m,
                    lookbacks: ts,
                }),
            }
        }
    }

    /// Unrolls the program into a [`StimProgram`] (`Circuit` ops plus
    /// absolute detector and observable record sets). Errors if the
    /// measurement flip probabilities differ, since a [`NoiseModel`] has one
    /// `p_meas`.
    pub fn to_program(&self) -> Result<StimProgram, StimError> {
        let mut ops = Vec::new();
        let mut detectors = Vec::with_capacity(self.num_detectors);
        let mut observables: Vec<Vec<usize>> = vec![Vec::new(); self.num_observables];
        let mut meas_p: Option<f64> = None;
        let mut bad: Option<StimError> = None;
        self.for_each_op(|op| match op {
            StimOp::Gate(g) => ops.push(Op::Gate(g)),
            StimOp::Measure { qubit, p, .. } => {
                match meas_p {
                    None => meas_p = Some(p),
                    Some(prev) if prev == p => {}
                    Some(prev) => {
                        if bad.is_none() {
                            bad = Some(StimError(format!(
                                "measurement flip probabilities differ ({prev} vs {p}); \
                                 qsim-lab's NoiseModel has one p_meas"
                            )));
                        }
                    }
                }
                ops.push(Op::Measure(qubit));
            }
            StimOp::Reset(q) => ops.push(Op::Reset(q)),
            StimOp::XFlip(q, p) => ops.push(Op::XFlip(q, p)),
            StimOp::YFlip(q, p) => ops.push(Op::YFlip(q, p)),
            StimOp::ZFlip(q, p) => ops.push(Op::ZFlip(q, p)),
            StimOp::Depolarize1(q, p) => ops.push(Op::Depolarize1q(q, p)),
            StimOp::Depolarize2(a, b, p) => ops.push(Op::Depolarize2q(a, b, p)),
            StimOp::Detector {
                base, lookbacks, ..
            } => detectors.push(lookbacks.iter().map(|&k| base - k as usize).collect()),
            StimOp::Observable {
                index,
                base,
                lookbacks,
            } => observables[index].extend(lookbacks.iter().map(|&k| base - k as usize)),
        });
        if let Some(e) = bad {
            return Err(e);
        }
        Ok(StimProgram {
            circuit: Circuit {
                num_qubits: self.num_qubits,
                ops,
            },
            noise: NoiseModel {
                p_meas: meas_p.unwrap_or(0.0),
                ..NoiseModel::none()
            },
            detectors,
            observables,
        })
    }
}

/// Parser state for [`parse_stim_circuit`].
struct Builder {
    blocks: Vec<Vec<Inst>>,
    targets: Vec<u32>,
    /// Open blocks (innermost last) with their REPEAT count and the
    /// measurement count at the start of their first iteration.
    stack: Vec<(usize, u64, u64)>,
    /// Measurements per iteration of each block, and detectors.
    block_meas: Vec<u64>,
    block_dets: Vec<u64>,
    /// Measurements so far on the first pass through every open block (the
    /// smallest record a `rec[-k]` inside can see).
    meas_first: u64,
    max_qubit: Option<usize>,
    num_observables: usize,
    max_lookback: u32,
    args: Vec<f64>,
}

/// Parses a non-negative decimal integer (an optional leading `+`, as
/// `str::parse::<usize>` accepts).
fn parse_uint(t: &str) -> Option<usize> {
    let b = t.as_bytes();
    let b = b.strip_prefix(b"+").unwrap_or(b);
    if b.is_empty() || b.len() > 19 {
        return t.parse().ok();
    }
    let mut v: usize = 0;
    for &c in b {
        if !c.is_ascii_digit() {
            return None;
        }
        v = v * 10 + (c - b'0') as usize;
    }
    Some(v)
}

impl Builder {
    fn cur(&mut self) -> usize {
        self.stack.last().map_or(0, |s| s.0)
    }

    fn qubit(&mut self, t: &str) -> Result<u32, StimError> {
        let q = parse_uint(t).ok_or_else(|| StimError(format!("unsupported target {t:?}")))?;
        if q > u32::MAX as usize / 2 {
            return err(format!("qubit target {t:?} too large"));
        }
        self.max_qubit = Some(self.max_qubit.map_or(q, |m| m.max(q)));
        Ok(q as u32)
    }

    fn rec(&mut self, t: &str) -> Result<u32, StimError> {
        let inner = t
            .strip_prefix("rec[")
            .and_then(|r| r.strip_suffix(']'))
            .ok_or_else(|| StimError(format!("expected rec[-k], got {t:?}")))?;
        let k: isize = inner
            .parse()
            .map_err(|_| StimError(format!("bad record target {t:?}")))?;
        if k >= 0 || (-k) as u64 > self.meas_first {
            return err(format!(
                "record {t} out of range ({} so far)",
                self.meas_first
            ));
        }
        self.max_lookback = self.max_lookback.max((-k) as u32);
        Ok((-k) as u32)
    }

    fn push(&mut self, kind: Kind, p: f64, start: usize) {
        let b = self.cur();
        let end = self.targets.len();
        self.blocks[b].push(Inst {
            kind,
            p,
            start: start as u32,
            end: end as u32,
        });
    }

    fn prob(&self, name: &str) -> Result<f64, StimError> {
        match self.args.as_slice() {
            [p] if (0.0..=1.0).contains(p) => Ok(*p),
            _ => err(format!("{name} needs one probability")),
        }
    }

    fn add_meas(&mut self, n: u64) -> Result<(), StimError> {
        let b = self.cur();
        self.block_meas[b] += n;
        self.meas_first = self
            .meas_first
            .checked_add(n)
            .ok_or_else(|| StimError("too many measurements".into()))?;
        Ok(())
    }

    fn instruction(&mut self, raw: &str, s: &str) -> Result<(), StimError> {
        // name, optional [tag], optional (args), targets
        let name_end = s
            .find(|c: char| c == '(' || c == '[' || c.is_whitespace())
            .unwrap_or(s.len());
        let mut buf = [0u8; 24];
        let name_raw = &s[..name_end];
        if name_raw.len() > buf.len() {
            return err(format!(
                "unsupported instruction {}",
                name_raw.to_ascii_uppercase()
            ));
        }
        let nb = &mut buf[..name_raw.len()];
        nb.copy_from_slice(name_raw.as_bytes());
        nb.make_ascii_uppercase();
        let name = std::str::from_utf8(nb).unwrap_or("");
        let mut rest = &s[name_end..];
        if rest.starts_with('[') {
            let close = rest
                .find(']')
                .ok_or_else(|| StimError(format!("bad tag: {raw}")))?;
            rest = &rest[close + 1..];
        }
        self.args.clear();
        if rest.starts_with('(') {
            let close = rest
                .find(')')
                .ok_or_else(|| StimError(format!("bad args: {raw}")))?;
            for a in rest[1..close].split(',') {
                let a = a.trim();
                if !a.is_empty() {
                    self.args.push(
                        a.parse::<f64>()
                            .map_err(|_| StimError(format!("bad argument {a:?} in {raw:?}")))?,
                    );
                }
            }
            rest = &rest[close + 1..];
        }
        let targets = rest.split_whitespace();
        let start = self.targets.len();
        let g1 = match name {
            "I" => Some(None),
            "H" | "H_XZ" => Some(Some(G1::H)),
            "X" => Some(Some(G1::X)),
            "Y" => Some(Some(G1::Y)),
            "Z" => Some(Some(G1::Z)),
            "S" | "SQRT_Z" => Some(Some(G1::S)),
            "S_DAG" | "SQRT_Z_DAG" => Some(Some(G1::Sdg)),
            _ => None,
        };
        if let Some(g) = g1 {
            if !self.args.is_empty() {
                return err(format!("{name} takes no arguments"));
            }
            for t in targets {
                let q = self.qubit(t)?;
                if g.is_some() {
                    self.targets.push(q);
                }
            }
            if let Some(g) = g {
                self.push(Kind::Gate1(g), 0.0, start);
            }
            return Ok(());
        }
        let g2 = match name {
            "CX" | "CNOT" | "ZCX" => Some(G2::Cx),
            "CZ" | "ZCZ" => Some(G2::Cz),
            "SWAP" => Some(G2::Swap),
            _ => None,
        };
        if let Some(g) = g2 {
            let ts: Vec<&str> = targets.collect();
            if !ts.len().is_multiple_of(2) || !self.args.is_empty() {
                return err(format!("bad {name} line"));
            }
            for pair in ts.chunks(2) {
                let a = self.qubit(pair[0])?;
                let b = self.qubit(pair[1])?;
                if a == b {
                    return err(format!("{name} on a repeated qubit {a}"));
                }
                self.targets.push(a);
                self.targets.push(b);
            }
            self.push(Kind::Gate2(g), 0.0, start);
            return Ok(());
        }
        match name {
            "TICK" | "QUBIT_COORDS" | "SHIFT_COORDS" => Ok(()),
            "R" | "RZ" | "RX" => {
                for t in targets {
                    let q = self.qubit(t)?;
                    self.targets.push(q);
                }
                self.push(Kind::Reset { x: name == "RX" }, 0.0, start);
                Ok(())
            }
            "M" | "MZ" | "MX" | "MR" | "MRZ" | "MRX" => {
                let p = match self.args.as_slice() {
                    [] => 0.0,
                    [p] if (0.0..=1.0).contains(p) => *p,
                    _ => return err(format!("bad {name} arguments")),
                };
                let mut n = 0u64;
                for t in targets {
                    if t.starts_with('!') {
                        return err("inverted measurement targets are not supported");
                    }
                    let q = self.qubit(t)?;
                    self.targets.push(q);
                    n += 1;
                }
                self.add_meas(n)?;
                let kind = Kind::Measure {
                    x: name.ends_with('X'),
                    reset: name.starts_with("MR"),
                };
                self.push(kind, p, start);
                Ok(())
            }
            "X_ERROR" | "Y_ERROR" | "Z_ERROR" | "DEPOLARIZE1" => {
                let p = self.prob(name)?;
                for t in targets {
                    let q = self.qubit(t)?;
                    self.targets.push(q);
                }
                let kind = match name {
                    "X_ERROR" => Kind::XErr,
                    "Y_ERROR" => Kind::YErr,
                    "Z_ERROR" => Kind::ZErr,
                    _ => Kind::Depol1,
                };
                self.push(kind, p, start);
                Ok(())
            }
            "DEPOLARIZE2" => {
                let p = self.prob(name)?;
                let ts: Vec<&str> = targets.collect();
                if !ts.len().is_multiple_of(2) {
                    return err("DEPOLARIZE2 needs pairs");
                }
                for t in ts {
                    let q = self.qubit(t)?;
                    self.targets.push(q);
                }
                self.push(Kind::Depol2, p, start);
                Ok(())
            }
            "DETECTOR" => {
                for t in targets {
                    let k = self.rec(t)?;
                    self.targets.push(k);
                }
                let b = self.cur();
                self.block_dets[b] += 1;
                self.push(Kind::Detector, 0.0, start);
                Ok(())
            }
            "OBSERVABLE_INCLUDE" => {
                let k = match self.args.as_slice() {
                    [k] if *k >= 0.0 && k.fract() == 0.0 && *k < 1e9 => *k as u32,
                    _ => return err("OBSERVABLE_INCLUDE needs an index"),
                };
                for t in targets {
                    let r = self.rec(t)?;
                    self.targets.push(r);
                }
                self.num_observables = self.num_observables.max(k as usize + 1);
                self.push(Kind::Observable(k), 0.0, start);
                Ok(())
            }
            _ => err(format!("unsupported instruction {name}")),
        }
    }
}

/// Parses a `.stim` circuit (see the module docs for the supported subset)
/// without unrolling `REPEAT` blocks.
pub fn parse_stim_circuit(text: &str) -> Result<StimCircuit, StimError> {
    let mut b = Builder {
        blocks: vec![Vec::new()],
        targets: Vec::new(),
        stack: Vec::new(),
        block_meas: vec![0],
        block_dets: vec![0],
        meas_first: 0,
        max_qubit: None,
        num_observables: 0,
        max_lookback: 0,
        args: Vec::new(),
    };
    for raw in text.lines() {
        let s = raw.split('#').next().unwrap_or("").trim();
        if s.is_empty() {
            continue;
        }
        if s == "}" {
            let Some((body, count, first)) = b.stack.pop() else {
                return err("unbalanced }");
            };
            // the remaining iterations of the body
            let more = b.block_meas[body]
                .checked_mul(count - 1)
                .ok_or_else(|| StimError("too many measurements".into()))?;
            b.meas_first = first
                .checked_add(b.block_meas[body])
                .and_then(|x| x.checked_add(more))
                .ok_or_else(|| StimError("too many measurements".into()))?;
            let parent = b.cur();
            let (m, d) = (
                b.block_meas[body].checked_mul(count),
                b.block_dets[body].checked_mul(count),
            );
            let (Some(m), Some(d)) = (m, d) else {
                return err("REPEAT count too large");
            };
            b.block_meas[parent] = b.block_meas[parent]
                .checked_add(m)
                .ok_or_else(|| StimError("too many measurements".into()))?;
            b.block_dets[parent] = b.block_dets[parent]
                .checked_add(d)
                .ok_or_else(|| StimError("too many detectors".into()))?;
            b.blocks[parent].push(Inst {
                kind: Kind::Repeat {
                    count,
                    block: body as u32,
                },
                p: 0.0,
                start: 0,
                end: 0,
            });
            continue;
        }
        if s.len() >= 6 && s[..6].eq_ignore_ascii_case("REPEAT") && s.ends_with('{') {
            let count: u64 = s[6..s.len() - 1]
                .trim()
                .rsplit(' ')
                .next()
                .unwrap_or("")
                .parse()
                .map_err(|_| StimError(format!("bad REPEAT line {raw:?}")))?;
            if count == 0 {
                return err(format!("REPEAT 0 is not supported: {raw:?}"));
            }
            let body = b.blocks.len();
            b.blocks.push(Vec::new());
            b.block_meas.push(0);
            b.block_dets.push(0);
            b.stack.push((body, count, b.meas_first));
            continue;
        }
        b.instruction(raw, s)?;
    }
    if !b.stack.is_empty() {
        return err("unterminated REPEAT block");
    }
    let lim = u32::MAX as u64;
    if b.block_meas[0] > lim || b.block_dets[0] > lim || b.targets.len() as u64 > lim {
        return err("program too large (more than 2^32 measurements, detectors or targets)");
    }
    Ok(StimCircuit {
        blocks: b.blocks,
        targets: b.targets,
        num_qubits: b.max_qubit.map_or(0, |m| m + 1),
        num_measurements: b.block_meas[0] as usize,
        num_detectors: b.block_dets[0] as usize,
        num_observables: b.num_observables,
        max_lookback: b.max_lookback as usize,
    })
}

/// Parses a `.stim` circuit (see the module docs for the supported subset)
/// into a [`StimProgram`] with `REPEAT` blocks unrolled.
pub fn parse_stim(text: &str) -> Result<StimProgram, StimError> {
    parse_stim_circuit(text)?.to_program()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_repeat_and_records() {
        let text = "R 0 1\nX_ERROR(0.01) 0 1\nREPEAT 3 {\n  CX 0 1\n  DEPOLARIZE2(0.002) 0 1\n  MR 1\n  DETECTOR rec[-1]\n}\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n";
        let p = parse_stim(text).unwrap();
        assert_eq!(p.circuit.num_qubits, 2);
        assert_eq!(p.detectors, vec![vec![0], vec![1], vec![2]]);
        assert_eq!(p.observables, vec![vec![3]]);
        // M(0.003) and MR (p=0) differ -> error
        assert!(parse_stim("M(0.1) 0\nM 0\n").is_err());
        assert!(parse_stim("FOO 0\n").is_err());
    }

    #[test]
    fn writer_batches_only_disjoint_ops() {
        let mut c = Circuit::new(4);
        c.ops = vec![
            Op::Gate(Gate::Cnot(0, 1)),
            Op::Gate(Gate::Cnot(2, 3)),
            Op::Gate(Gate::Cnot(1, 2)),
            Op::Measure(0),
            Op::Measure(1),
        ];
        let noise = NoiseModel::gate_depolarizing(0.0, 0.01);
        let s = to_stim(&c, &noise, &[vec![0, 1]], &[]).unwrap();
        assert_eq!(
            s,
            "CX 0 1 2 3\nDEPOLARIZE2(0.01) 0 1 2 3\nCX 1 2\nDEPOLARIZE2(0.01) 1 2\nM 0 1\nDETECTOR rec[-2] rec[-1]\n"
        );
    }
}
