# Dense k-qubit fusion in the blocked executor (integration pass, 4 Oct 2026)

Source: `exp/sv-monomial` (7be7697, `src/dense_fusion.rs`) and
`wip/fusion-avx2` (327b346, lane-exchange kernels `src/dense_kernels.rs`),
both unreviewed WIP left at agent timeouts on 1 Oct. Ported onto main,
reviewed, differential-tested and measured by the integration agent.
Both branches are archived (`research/ARCHIVE.md`).

## What is on main

`BlockConfig::dense_fusion` (width `k` = 2 or 3; 0/1 = off) and
`BlockConfig::dense_min_ops` (cost rule, 0 = auto). Default: **k = 2 on
aarch64, off on x86_64** (§4).

* **Pass** (`dense_fusion::fuse_stage`): runs per stage of the blocked
  executor, after `schedule_diag`, on the stage's `KOp`s. Greedy: each
  qubit is owned by at most one pending group; a gate whose qubits are all
  cached in the block and whose union with the groups it touches has at most
  `k` qubits is multiplied into them (groups on disjoint qubits are merged
  first); any other gate flushes the groups on its qubits and is emitted
  as is. Diagonal phases are absorbed only into an existing group that
  already covers them. Correctness argument: a pending group only contains
  gates on its own qubits and every later gate touching those qubits either
  joins or flushes it, so everything emitted meanwhile acts on disjoint
  qubits and commutes with it.
* **Cost rule**: a group on `k >= 2` qubits is emitted as a dense op only if
  it contains at least `2^k` uncontrolled 1-qubit gates whose 2x2 matrix has
  no zero entry ("heavy" gates); otherwise its gates are emitted unchanged.
  Rationale in §3. `dense_min_ops = 1` forces every group of two or more
  gates (tests, experiments).
* **Kernels** (`dense_kernels.rs`, from `wip/fusion-avx2`, used unchanged
  apart from wiring): split re/im buffers, tiles of `2^k` vectors of 8
  amplitudes; targets below bit 3 are exchanged with free higher index bits
  by constant shuffles so the `2^k x 2^k` matrix-vector product is purely
  vertical; scalar gather path when the block has fewer than `3 + k` bits.
  Monomorphised over main's FMA switch, so they run under the AVX2+FMA
  dispatch on x86_64 and with NEON `fmla` on aarch64.
* Not ported from the WIP: `KOp::Dense` in the executor IR (fusion now runs
  per stage and `KOp` stays `Copy`), the WIP's `tile_u1` lane-exchange
  1-qubit kernel (superseded by main's `exp/simd` kernels), `sv_micro`.

## 1. Review findings on the WIP

* The fusion algorithm was correct in principle (same ownership argument),
  but it fused every group of two or more gates, including a lone `U1 + CNOT`
  and pure permutation chains, and it fused across qubits the stage does
  not cache only by luck of `needs_inner` (controls outside the block were
  allowed into dense ops). The port restricts dense ops to cached qubits.
* `DenseOp::lift` / `compose` (now `then`) and `u1_to_dense` were correct;
  `swap_to_dense` and `phase_to_dense` too. Unit tests kept (H·H = I, Bell
  column) and extended (lone ops pass through, outer qubits not fused,
  cost rule).
* `dense_kernels.rs` had no tests in the WIP. It is now covered by the
  differential tests below for every target pattern `PM` (targets below
  bit 3), both the tile and the scalar path, f32/f64, portable and FMA.

## 2. Tests

`tests/dense_fusion.rs` (all also run on aarch64):

* `fused_matches_audit_reference`: against the independent naive state
  vector `audit_common::RefSv` (edge-case angles and qubits, SWAP/CCX/CPhase),
  n = 2..12, 6 block geometries (incl. gathered blocks and L1 tiling on),
  portable and SIMD kernels, (k, rule) in {(2, forced), (3, forced),
  (3, default)}: max |Δamp| <= 1e-12 (f64), <= 1e-5 (f32).
* proptests `random_circuits_agree` (random universal circuits) and
  `layered_circuits_agree` (brickwork + CNOT/CPhase/Rx/SWAP/CCX layers),
  against gate-by-gate and the unfused blocked executor, same tolerances.
* `algorithms_agree` (GHZ, QFT, brickwork, n = 4..13), `fusion_engages`
  (fusion actually produces dense ops on the test circuits, including
  3-qubit groups in a single-block register).
* `tests/blocked.rs` configs now include forced dense fusion (k = 2, 3).

## 3. Measurements (Apple M1 Pro, 8 cores; interleaved A/B, min of 3)

Harness: `examples/l1_bench.rs` (`dense=k`, `dmin=m` keys; workloads
`brick` = random brickwork depth 20, `su4` = brickwork of generic 2-qubit
unitaries written as 3 x (Ry Rz on both qubits, CNOT), `qft`, `ghz`,
`adder` = repeated Cuccaro adders). Each workload is one locked chunk under
the swarm's Mac bench lock; configurations run round-robin per repetition.
**The Mac was heavily shared during all runs (load average 11-24 on 8
cores)**; in the same chunk, configurations that execute an identical op
list differ by up to 3-10% at n = 24 and 15-18% for runs under 0.1 s, which
is the noise floor. Raw data and drivers: `research/data/dense-fusion/`.

`raw` = every group of two or more gates fused (`dmin=2`, the WIP's
behaviour minus single-gate groups); `rule` = the cost rule of §0. Speedup
= unfused / fused.

| workload | n | prec | threads | unfused s | k=2 raw | k=2 rule | k=3 raw | passes unfused → k=2 raw |
|---|---|---|---|---|---|---|---|---|
| brick | 24 | f32 | 6 | 1.180 | 0.98x | (no dense op) | 0.96x | 692 → 273 |
| brick | 24 | f64 | 6 | 1.388 | **0.66x** | (no dense op) | 0.66x | 692 → 272 |
| brick | 26 | f32 | 8 | 3.983 | 0.88x | (no dense op) | 0.89x | 752 → 299 |
| brick | 20 | f32 | 6 | 0.065 | 0.87x | (no dense op) | 0.95x | 572 → 219 |
| su4 | 24 | f32 | 6 | 2.155 | **2.03x** | **2.07x** | 2.04x | 2070 → 243 |
| su4 | 24 | f64 | 6 | 4.152 | 1.48x | **1.49x** | 1.49x | 2070 → 243 |
| su4 | 26 | f32 | 8 | 9.491 | 2.07x | **2.09x** | 2.06x | 2250 → 269 |
| su4 | 20 | f32 | 6 | 0.122 | 2.02x | **2.00x** | 1.98x | 1710 → 209 |
| qft | 24 | f32 | 6 | 0.098 | 1.02x (1 dense op) | (no dense op) | 0.94x | 59 → 58 |
| ghz | 24 | f32 | 6 | 0.058 | 0.96x (1 dense op) | (no dense op) | **0.48x** (10 dense3) | 24 → 13 (k=3) |

(First, separate run at load 15-24: brick-24 f32 k=2 raw 0.87x, su4-24 f32
2.05x, QFT-24 0.91x; same picture.)

Reading:

* **Flops, not passes, decide.** A dense 4x4 product costs 4 complex
  multiply-adds per amplitude, exactly the cost of two separate 1-qubit
  gates. Brickwork's groups are `U1 U1 CNOT`: the same flops in one pass
  instead of three, and the fused kernel loses (0.66-0.98x) because it is
  less efficient per flop than the specialised 1-qubit kernels (gathering
  4 vectors, lane exchanges for low targets) and the CNOT pass it saves is
  nearly free. In f64 the loss is worst (0.66x); a likely cause (not
  profiled) is register pressure: a k = 2 tile is 4 x 8 complex lanes,
  twice the registers in f64. This confirms the earlier
  VPS result in `research/sv-monomial.md` §2 (WIP k=2 slower on brickwork)
  on a second architecture with a corrected implementation.
* su4 groups contain six heavy 1-qubit gates and three CNOTs per pair: 3x
  fewer flops and 9x fewer passes after fusion, and the fused executor is
  **2.0-2.1x faster in f32, 1.5x in f64** on the M1 Pro. Generic 2-qubit
  unitaries are what quantum-volume and random-circuit-sampling workloads
  are made of, and what qsim's own fusion is built for.
* **k = 3 adds nothing in practice.** With n larger than the block, each
  stage of the blocked planner spans about one brick layer, and 3-qubit
  groups need two consecutive layers in one stage; they appear only when the
  whole register is one block (n <= 12 by default) or for chains (GHZ),
  where they are a 2x loss. Fusion is bounded by stage boundaries; a planner
  that cuts stages around fusion groups (or fuses before staging) is the
  open next step.
* Permutation chains must never be fused (GHZ k=3: 0.48x). Hence the cost
  rule counts only 1-qubit gates with a dense 2x2 matrix: CNOT, CCX, X and
  swaps (arithmetic circuits) never trigger fusion.

Cost rule check: under the rule brickwork, QFT, GHZ and the adder produce
exactly the unfused op list (zero dense ops, same pass count), so their
time is the unfused time up to the fusion pass itself; su4 keeps the full
gain. See §3a for the confirmation run of the final rule.

## 3a. Confirmation of the final rule (7329bd3)

Same harness, one chunk per row, `off2` a second copy of the unfused
config in the same chunk (noise floor), load 13-20:

| workload | n | prec | unfused s (off / off2) | k=2 rule | k=3 rule | dense ops (k=2) |
|---|---|---|---|---|---|---|
| su4 | 24 | f32 | 2.550 / 2.570 | **2.15x** | 2.05x | 230 |
| su4 | 24 | f64 | 3.430 / 3.428 | **1.50x** | 1.58x | 230 |
| brick | 24 | f32 | 1.058 / 1.067 | 1.06x (identical ops) | 1.06x | 0 |
| adder | 24 | f32 | 1.564 / 1.517 | 1.02x (identical ops) | 1.01x | 0 |
| qft | 24 | f32 | 0.122 / 0.124 | 1.00x (identical ops) | 0.98x | 0 |
| ghz | 24 | f32 | 0.052 / 0.044 | 0.98x (identical ops) | 1.24x | 0 |

Where the rule fuses nothing, the stage keeps its original op list, so the
result is bit-identical to the unfused executor
(`tests/dense_fusion.rs::no_dense_op_is_bit_identical`) and the timing
differences in those rows are noise (compare off vs off2).

## 4. Verdict and defaults

* **Merged.** Correct (differential tests above, x86_64 and aarch64),
  and a real gain on the circuit class it targets: **1.5x (f64) to 2.15x
  (f32) on brickwork of generic two-qubit unitaries at n = 20-26 on the M1
  Pro**, with no measured regression because the cost rule leaves every
  other workload we tried bit-identical.
* **Default on aarch64: `dense_fusion = 2`** (meets the bar of >= 1.1x
  without regressions). k = 3 is available but adds nothing measurable (no
  3-qubit groups form across stages; see §3).
* **Default on x86_64: off.** Not measured: the VPS load average was 10-27
  for the whole integration window (no timing claims above load 4). The
  kernels run there under the AVX2+FMA dispatch and are tested; turning
  the default on needs a quiet-window A/B of `su4` and `brick` with
  `l1_bench ... off:dense=0 k2:dense=2`.
* The plain brickwork claim of `research/sv-monomial.md` §2 stands on both
  machines: dense 2-qubit fusion does not close the gap to qsim on
  brickwork (it is a loss there), because brickwork's two-qubit gates are
  CNOTs that cost nothing to apply separately.
* Open: fusion is bounded by the stages of the blocked planner (each stage
  ~ one brick layer when n exceeds the block); planning stages around
  fusion groups, a per-kernel cost model instead of the gate-count rule,
  and a register-blocked f64 kernel are the next steps. (A separate agent,
  `exp/graph-compiler`, is building cost-model fusion on top of this.)
