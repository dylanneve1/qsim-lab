# Chain-sweep amplitudes for IBM's doped-Clifford circuit (nq70, depth 70)

Status: experiment (October 2026). The engine is exact and validated, and its
cost is measured up to CZ-depth 58 on an M1 Pro. The full depth-70 circuit
does **not** fit on a 16 GB machine, and bond slicing cannot make it fit (see
below).

Circuit: `nq70_depth70_checks27_doped.qasm` (quantum-advantage-tracker issue
228). 70 qubits on an open chain, 2415 CZ in 70 brickwork layers (only
`(i, i+1)` pairs), 468 `rz(pi/4)`, other single-qubit gates Clifford.
Amplitude convention `<x|U|0^70>` with bit `i` of `x` = qubit `i` (Qiskit's
little-endian integer).

Prior art: Manabe, Gu, Pan, arXiv:2608.13110 contract the same network along
the chain (their "temporal-boundary" sweep, width `ceil(d/2) = 35`). They
distribute the 2^35-entry (256 GiB) tensor over 8 H100s, at 32 s per
256-amplitude batch, and found that slicing that tensor down even by one bit
has a rapidly growing overhead. Everything here reproduces that picture on
small hardware. It is not a new method.

## Method (`src/engines/chain_sweep.rs`)

- Each `CZ(i, i+1) = Σ_b P_b ⊗ Z^b`: the left qubit gets the projector, the
  right qubit gets `Z^b`. A bond `b` is one bit. The amplitude is
  `Σ_bonds Π_i <x_i| W_i(bonds) |0>`.
- The sweep runs from qubit 0 to qubit 69 on a register of bond bits that
  starts in `|0..0>`. Processing qubit `i` sums out the bonds of edge
  `(i-1, i)` and creates those of edge `(i, i+1)`. The qubit's own wire is
  kept "tied" to the last bond it created whenever possible, so the register
  never needs more than `ceil(D/2)` bits: each qubit is processed forwards or
  backwards in time, whichever is narrower.
- Every step is a single-bit matrix (generally non-unitary), a controlled X
  or a diagonal term, i.e. an ordinary `KOp`. The existing cache-blocked CPU
  executor and the Metal stage kernels therefore run the sweep **unchanged**.
  At the end every bit is free again and the amplitude is the `|0..0>`
  component times a scale factor tracked on the host.
- The bond sums use the unnormalised `[[1,1],[1,-1]]`. The register norm then
  shrinks by exactly half a bit per qubit (the amplitude's own `2^-n/2`
  scale), which keeps f32 safe at n = 70. A normalised Hadamard underflowed
  f32 by qubit ~20.
- In a "tied" consume, the `2x2` diagonal is factored as `A_p B_q C^{pq}`, so
  only one 2-bit phase remains. This cuts KOps by a third and Metal time by
  24%.
- Slicing: any CZ can be fixed to `b = k`, which turns it into the local
  operators `P_k`, `Z^k` and removes its bit. `cut_tensors_cpu` sweeps from
  both ends to one edge and Walsh–Hadamard transforms the right half. This
  gives the amplitude as a dot product and **all** slice sums over that
  edge's bonds for free (used for the fidelity runs).

Example driver: `examples/chain_sweep.rs` (`info`, `validate`, `bench`,
`fidsv`, `fidmitm`). Tests: `tests/engines/chain_sweep.rs`.

## Validation (exactness)

Random bitstrings (half uniform, half sampled from `|ψ|^2`) of the circuit
truncated to the first `n` qubits and the first `D` CZ layers, compared with
the dense f64 state vector. Errors are `|a - a_sv|` over the RMS amplitude
`2^-n/2`, and relative errors over amplitudes that are not exact zeros.

| machine | n | D | width | backend | max err / rms | max rel |
|---|---|---|---|---|---|---|
| VPS | 20 | 20 | 10 | CPU f64 / f32 | 5.0e-15 / 2.1e-7 | 3.6e-15 / 1.5e-7 |
| VPS | 20 | 30 | 15 | CPU f64 / f32 | 7.8e-15 / 4.1e-7 | 3.9e-15 / 2.1e-7 |
| VPS | 22 | 40 | 20 | CPU f64 / f32 | 5.6e-15 / 4.5e-7 | 4.0e-15 / 3.2e-7 |
| VPS | 24 | 24 | 12 | CPU f64 / f32 | 6.2e-15 / 2.1e-7 | 5.1e-15 / 2.7e-7 |
| VPS | 24 | 44 | 22 | CPU f64 / f32 | 8.2e-15 / 3.2e-7 | 4.5e-15 / 2.0e-7 |
| M1 Pro | 20 | 20 | 10 | Metal f32 | 2.1e-7 | 1.5e-7 |
| M1 Pro | 22 | 40 | 20 | Metal f32 | 1.0e-6 | 6.8e-7 |
| M1 Pro | 24 | 44 | 22 | Metal f32 | 5.5e-7 | 3.6e-7 |

At n = 70, where no state vector exists, the CPU (f32) and Metal amplitudes
of the same bitstrings agree to about 1e-6 relative (D = 30 to 52). The unit
tests cover random brickwork circuits with arbitrary rotations (every
amplitude, n ≤ 8), the truncated IBM circuit, slicing (slice sum = amplitude,
each slice = the state vector with that CZ replaced by `P_k ⊗ Z^k`) and the
meet-in-the-middle split.

## Cost (n = 70, unsliced, one amplitude)

Width = `D/2` at every cut (35 at D = 70; checked by `info`). Times are the
run only. Backend compilation is ≤ 0.02 s. The M1 Pro (16 GB, 2021) was in
use with other CPU jobs running. The VPS is 4 shared vCPUs.

| D | width | f32 state | M1 Pro Metal (s) | Metal stages | M1 Pro CPU f32 (s) | VPS CPU f32 (s) |
|---|---|---|---|---|---|---|
| 30 | 15 | 256 KiB | 0.01 | | | 0.10 |
| 36 | 18 | 2 MiB | 0.04 | | | 0.43 |
| 40 | 20 | 8 MiB | 0.10 | 246 | | 1.64 |
| 44 | 22 | 32 MiB | 0.43 | 315 | | 6.25 |
| 48 | 24 | 128 MiB | 1.81 | 316 | 6.1 | 20.8 |
| 50 | 25 | 256 MiB | 3.82 | 350 | | |
| 52 | 26 | 512 MiB | 8.0 | 385 | 25.0 | |
| 54 | 27 | 1 GiB | 16.5 | 385 | | |
| 56 | 28 | 2 GiB | 34.3 | 420 | | |
| 58 | 29 | 4 GiB | 71.6 | 420 | | |

The Metal time doubles per width bit (×2.07 from 28 to 29). At width 25, a
per-stage profile gives 1.45 s of pure load/store (129 GB/s) out of 4.1 s.
The rest is in-stage arithmetic, mostly diagonal terms. That leaves roughly
2.5–3× headroom from kernel work before the 350 passes are bandwidth-bound.

### Extrapolation to D = 70 (width 35)

- **Memory: 2^35 complex64 = 256 GiB.** The M1 Pro has 16 GB, and the Metal
  backend's 32-bit indexing caps it at 2^30 anyway. The largest exact depth
  on this Mac is D = 58 (4 GiB), or D = 60 (8 GiB) on an otherwise idle
  machine.
- **Time, if the memory existed:** 71.6 s × 2^6 × (≈525/420 stages; the
  stage count grows by ~35 per 2 width bits) ≈ 5.7e3 s ≈ **1.6 h per
  amplitude** at the measured M1 Pro efficiency. The uncertainty is about
  ×1.5 either way: a 256 GiB stream is not a 4 GiB one, and the stage count
  is itself extrapolated. The pure load/store floor of ~525 passes over
  512 GiB at 129 GB/s is about 0.6 h. For scale, the paper
  reports 32 s per batch of 256 amplitudes on 8 H100s.
- **Slicing does not rescue it.** In a chain sweep a bond belongs to exactly
  one cut, so fixing it narrows only that cut. Bringing every one of the 69
  cuts from 35 to 30 bits (8 GiB) means slicing ≥ 5 bonds per edge, ≥ 345
  bonds in all: 2^345 slices. Generic slicing of the whole network (the
  paper's cotengra study) has the same problem: the overhead explodes even
  for one bit. Dropping slices to save time is no help either, because the
  fidelity equals the fraction kept (next section), so 2^-345 of the slices
  is zero fidelity.
- Out-of-core is not possible here either: 256 GiB against ~13–17 GB of free
  disk.

## Does fidelity equal the fraction of slices kept?

`fidsv` builds every slice state of `s` sliced bonds with the f64 state
vector. It then takes the overlaps `<ψ|ψ_k>` and the Gram matrix, and
evaluates `F(K) = |Σ_K <ψ|ψ_k>|^2 / ||Σ_K ψ_k||^2` for 300 random subsets
`K` per kept fraction. `fidmitm` estimates the same quantity at n = 70 from
k uniform bitstrings via the meet-in-the-middle cut tensors (ratio
estimator; biased upward by about 1/k at small F). All runs were on the M1
Pro CPU.

| circuit | n | D | sliced bonds | Σ‖ψ_k‖² | slice norms | mean \|cos\| off-diag | F at f = 1/256, 1/16, 1/4, 1/2 |
|---|---|---|---|---|---|---|---|
| IBM, middle edge | 20 | 20 | 8 | 1.000000 | all 2^-8 | 0.0000 | 0.0039, 0.0625, 0.2500, 0.5000 (sd 0) |
| IBM, middle edge | 20 | 40 | 8 | 1.000000 | all 2^-8 | 0.0000 | 0.0039, 0.0625, 0.2500, 0.5000 (sd 0) |
| IBM, 8 different edges | 20 | 40 | 8 | 1.000000 | all 2^-8 | 0.0000 | 0.0039, 0.0625, 0.2500, 0.5000 (sd 0) |
| IBM, middle edge (k = 300 bitstrings) | 70 | 40 | 8 of 20 | | | | 0.008, 0.069, 0.262, 0.505 (sd over subsets 0.006–0.024) |
| random U3 brickwork (control) | 20 | 40 | 8 | 1.000000 | 1.7e-4 … 1.8e-2 | 0.011 | 0.0067, 0.062, 0.248, 0.502 (sd 0.011–0.055) |

- On IBM's circuit, the slices of CZ bonds are, to f32 precision, exactly
  orthogonal and of exactly equal norm `2^-s`. Every subset of a fraction `f`
  of the slices then has fidelity exactly `f`. The n = 70 estimate agrees
  within its sampling error. A plausible reason, not proven here: the only
  non-Clifford gates are diagonal `T`s, and the projector side of each CZ is
  then an unbiased, uncorrelated Z bit, as in a stabilizer state.
- On a generic random circuit the rule only holds on average: slice norms
  vary by 100×, and individual subsets scatter widely around `f`.
- Either way, `F ≈ f` buys time, not memory. With bond slicing it cannot
  bring this circuit onto a 16 GB machine.

## Next bottlenecks / options

1. **Memory is the wall** for the exact D = 70 amplitude: 256 GiB, about 16
   Macs' worth of RAM. Any route onto one machine has to be approximate in
   the boundary itself. One example is truncating the boundary tensor as an
   MPS along the time direction (a boundary-MPS / transverse contraction).
   Its fidelity cost is unmeasured.
2. Speed at D ≤ 58: (a) in-stage diagonal work (~65% of Metal time at
   width 25); (b) pass count, about 6 stages per qubit. A light-cone-aware
   stage planner could merge successive qubits' staircases (the paper's
   "merge small branch tensors"), plausibly 2–4× fewer passes.
3. Batches: `cut_tensors_cpu` turns `K` left halves × `J` right halves into
   `K·J` amplitudes for `K + J` half sweeps plus `K·J` dot products. Open
   output bits as in the paper (3-region scheme) are not implemented.

## Reproduce

```text
cargo test --release --test chain_sweep
cargo run --release --example chain_sweep -- validate --n 22 --d 40 --k 8 --backends cpu64,cpu32
cargo run --release --features metal --example chain_sweep -- bench --n 70 --d 58 --reps 2 --backend metal
cargo run --release --example chain_sweep -- fidsv --n 20 --d 40 --s 8 --subsets 300 [--mode spread]
cargo run --release --example chain_sweep -- fidmitm --n 70 --d 40 --s 8 --k 300 --subsets 16
```
