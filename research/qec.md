# QEC lab notebook: circuit-derived detector error model and surface-code threshold

Machine: shared 4-vCPU AMD EPYC-Rome VPS (AVX2), 7.7 GB RAM, with other agents
compiling at the same time (load average 7–11 during these runs). All Monte
Carlo runs went through `qsim-swarm/bench.sh` (flock). Timings are only
indicative because the VM is shared. Rust 1.93.1, release profile.

## 1. Problem

`main` @ 86e5d67 had a "fast" surface-code sampler, `run_experiment_fast`, that
drew from a **hand-written phenomenological** mechanism list: one X flip per
data qubit per round plus one measurement flip per check, all at rate `p`. The
circuit and the `NoiseModel` were not used at all. `run_experiment` switched to
it silently when `shots > 300`. Its "agreement" test only checked that both
rates were < 0.15. The audit (`exp/audit`, `tests/qec_dem_audit.rs`) measured
the damage at d=3, rounds=3, `circuit_level(0.005, 0.005)`, 20k shots per side:
decoded logical error 0.43% (fast) vs 2.50% (full tableau), with 12 of 16
detectors off by up to 17.7σ.

The previous agent's WIP (100c2dc) replaced the list with a circuit-derived DEM
built by forward Pauli-frame propagation, but it had not been tested. My review
found:

| # | bug in 100c2dc | effect |
|---|---|---|
| 1 | each fault was propagated through `ops[op_idx..]`, i.e. starting **with** the op that produced it | reset errors were cleared by their own `Reset` and never reached the DEM; gate errors were conjugated by their own gate (harmless for depolarizing noise, wrong in general) |
| 2 | `SurfaceCode::new` built the DEM with `NoiseModel::uniform(1.0)`, and `run_experiment_fast(_noise)` ignored its argument | the noiseless test gave 262/500 logical errors (also seen by the auditor) |
| 3 | merged mechanisms were sampled as independent events | the mutually exclusive X/Y/Z (or 15-way) choice became independent events, an O(p²) distribution change |
| 4 | the decoder still used the hand-built phenomenological graph | no diagonal (space-time) edges, see §4 |
| 5 | unused variables | clippy `-D warnings` failed |

The auditor found bug 1 independently (exp/audit `research/audit.md` §6):
with reset-only noise, 12 detectors were off by up to 28.6σ.

## 2. What the code does now (`src/qec/dem.rs`, `src/qec/surface.rs`)

**Noise locations.** These mirror `Circuit::run_noisy` exactly, and
`tests/surface.rs::fault_locations_cover_every_noisy_op` checks the list
op by op.

| location | outcomes (mutually exclusive) | probability each |
|---|---|---|
| after every 1-qubit gate | X, Y, Z | `p_1q / 3` |
| after every 2-qubit gate | the 15 non-identity Paulis on (a, b) | `p_2q / 15` |
| every measurement | classical flip of that record | `p_meas` |
| after every reset | X on the qubit | `p_reset` |

**Signatures by a backward sensitivity sweep.** One right-to-left pass keeps,
for every qubit, the bit-packed set of detectors (plus the observable) that an
X or a Z inserted *at that point* would flip:

- measuring q (Z basis) at record m: `sx[q] ^= D(m)`, where D(m) is the set of
  detectors containing m;
- reset: `sx[q] = sz[q] = 0`;
- H: swap `sx`/`sz`;
- CNOT(c,t): `sx[c] ^= sx[t]`, `sz[t] ^= sz[c]`;
- S/S†: `sx ^= sz`;
- CZ: `sx[a] ^= sz[b]`, `sx[b] ^= sz[a]`;
- SWAP: swap.

A fault's signature is the XOR of the sensitivities of its Pauli's X and Z
parts, read just after the op that produced it. A readout flip's signature is
D(m) itself. The cost is a single pass over the circuit: building the d=9
model takes 13 ms.

**Independent forward check.** `propagate_forward` is the textbook rule set:

- the fault is inserted after its op;
- H swaps X↔Z;
- CNOT copies X from control to target and Z from target to control;
- a Z-basis measurement record flips iff the frame has an X component on that qubit;
- reset clears the frame on its qubit;
- a readout fault flips the record directly.

`backward_sweep_matches_forward_pauli_frames_on_every_fault` checks that both
give identical signatures for **every outcome of every location** at d=3 r=3
and d=5 r=2.

**Exact sampler.** `DemSampler` groups locations by firing probability and
draws the firing ones with geometric skipping, using `ln_1p` for tiny p. Each
firing location picks one of its outcomes uniformly, which is the same
exclusive choice `run_noisy` makes, and the signatures are XORed. This is the
distribution of (detector events, observable flip) that the tableau produces,
not an approximation of it. No merging is used for sampling; merged
mechanisms (`detector_error_model()`) exist only for inspection.

**Explicit sampling method.** `run_experiment(noise, shots,
SamplingMethod::{Tableau, DetectorErrorModel}, rng)` replaces the old call.
The `shots > 300` switch is gone. `run_experiment_tableau` and
`run_experiment_dem` are thin wrappers.

**Detectors.** They are defined once, as lists of measurement records
(`detector_records()`, `observable_records()`). Both `extract_z_defects`
(tableau path) and the DEM use these lists, so the two paths cannot disagree
on what a detector is.

## 3. Hook errors: the CNOT schedule was distance-reducing

With the original X-check order (NW, NE, SW, SE), an X fault on an X-check
ancilla after two CNOTs spreads to the last two data qubits, SW and SE. Those
two qubits are horizontal, parallel to the X-type logical (left→right for
this layout, where Z_L is column 0). The circuit-derived graph showed what
that costs:

- `every_single_circuit_fault_is_corrected` failed at d=3: such a hook fault
  has the same single-detector signature as a logical boundary error;
- the graph-like circuit distance, `DecodingGraph::min_logical_weight`, was
  **3 at d=5**.

I changed the X-check order to NW, SW, NE, SE. The hook pair is then
vertical, and the graph distance is d for d = 3, 5, 7, 9
(`circuit_derived_graph_has_full_distance`). Z-check hooks are Z errors and
do not affect a Z-basis memory experiment. Syndrome extraction is sequential
per check (not interleaved), so no commutation constraint fixes the order.

## 4. Decoding graph built from the circuit; >2-detector mechanisms

`SurfaceCode::new` now builds the Union-Find graph from the fault list
(`decoding_graph_from_faults`):

- a signature with 2 detectors becomes an edge;
- a signature with 1 detector becomes an edge to the single boundary node;
- duplicate edges are merged;
- an edge's logical flag is the observable flip of the faults that produce it;
- if faults with **both** flags produce the same edge, which can happen at
  boundary edges because one boundary node stands for both sides, the more
  probable flag is kept and the conflict is counted;
- signatures with 3+ detectors (hyperedges) are not added. Union-Find only
  handles graphs. The report checks that every hyperedge decomposes into
  existing edges with a consistent logical flag. Sampling is unaffected
  either way.

Measured (`examples/surface_threshold report d`):

| d (=rounds) | qubits | detectors | fault locations | distinct signatures | 1-det | 2-det | ≥3-det | flag conflicts | undetectable logical | graph distance (circuit / phenomenological) |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 17 | 16 | 145 | 55 | 16 | 39 | 0 | 0 | 0 | 3 / 3 |
| 5 | 49 | 72 | 761 | 301 | 36 | 265 | 0 | 0 | 0 | 5 / 5 |
| 7 | 97 | 192 | 2185 | 883 | 64 | 819 | 0 | 0 | 0 | 7 / 7 |
| 9 | 161 | 400 | 4753 | 1945 | 100 | 1845 | 0 | 0 | 0 | 9 / 9 |

For this Z-memory circuit **no fault flips more than two Z detectors**, and,
with the fixed schedule, no edge has conflicting logical flags. A Y on a data
qubit acts like its X part. A two-qubit fault on a data–ancilla CNOT combines
a data error and an ancilla flip, and the shared detector cancels. The
hyperedge and conflict handling is therefore documented and counted, but it
never triggers here. (With the old CNOT order there *were* conflicts: see §3.)

The old hand-built graph is still available, as
`SurfaceCode::with_phenomenological_decoder`, for A/B comparison. It has one
space edge per data qubit per round and one time edge per check, and no
diagonal edges. Same DEM samples, 50k shots, seed 3
(`research/data/qec/ab_graph_and_speed.csv`):

| d | p | circuit graph p_L | phenomenological graph p_L |
|---|---|---|---|
| 3 | 0.3% | 0.464% | 0.416% |
| 5 | 0.3% | 0.286% | 0.478% |
| 7 | 0.3% | 0.114% | 0.244% |
| 3 | 0.5% | 1.226% | 1.130% |
| 5 | 0.5% | 1.258% | 1.606% |
| 7 | 0.5% | 0.860% | 1.106% |

At d=3 the two graphs are within noise (the phenomenological graph is
nominally better). From d=5 up, the circuit-derived graph is clearly better:
it halves p_L at d=7, p=0.3%.

## 5. Validation: DEM sampler vs full tableau (acceptance test)

`tests/qec_dem_audit.rs` comes from `exp/audit`. Only `fast_stats()` was
changed, to drive the new `dem_sampler`. I also added a pooled check on the
mean number of detection events per shot (Welch z) and the env overrides
`QSIM_DEM_D`/`QSIM_DEM_P`. The test does a two-proportion z-test per detector
plus the raw and decoded logical rates, Bonferroni-corrected, with
|z| < 4.9 required everywhere.

```
cargo test --release --test qec_dem_audit -- --ignored --nocapture
QSIM_DEM_P=0.01 QSIM_DEM_SHOTS=50000 cargo test --release --test qec_dem_audit -- --ignored --nocapture
QSIM_DEM_D=5 cargo test --release --test qec_dem_audit -- --ignored --nocapture
```

| model | d | p | shots/side | worst detector \|z\| | detection events/shot (tableau / DEM) | decoded p_L tableau | decoded p_L DEM | z | verdict |
|---|---|---|---|---|---|---|---|---|---|
| OLD (main 86e5d67, auditor's run) | 3 | 0.5% | 20k | 17.7 | – | 2.495% | 0.425% | ≈17 | **FAIL** |
| OLD (main 86e5d67, my re-run, original audit file) | 3 | 0.5% | 20k | 17.7 (12 of 16 beyond 4.9σ) | – | 2.495% | 0.425% | +17.3 | **FAIL** |
| NEW | 3 | 0.5% | 20k | 1.7 | 0.6404 / 0.6258 | 1.190% | 1.250% | −0.5 | PASS |
| NEW | 3 | 1.0% | 50k | 1.6 | 1.2114 / 1.2180 | 4.234% | 4.104% | +1.0 | PASS |
| NEW | 5 | 0.5% | 20k | 2.5 | 3.5934 / 3.6055 | 1.350% | 1.225% | +1.1 | PASS |

Raw outputs are in `research/data/qec/audit_*.txt`. The tableau p_L under the
new code (1.19% at d=3) is lower than the auditor's 2.5% on main. The noise
and circuit are the same, but the decoder graph is now circuit-derived and the
hook order is fixed. Both sides of the test use the same decoder, so the test
compares samplers, not decoders.

A quick, always-run version (3000 shots, 5σ) lives in
`tests/surface.rs::dem_sampling_matches_tableau_quick`.

### 5.1 Old model re-run

I re-ran the auditor's unmodified `tests/qec_dem_audit.rs` against a worktree
of `main` @ 86e5d67 (same seed 7, 20k shots). It reproduces the audit
exactly:

- 12 detectors beyond 4.9σ, worst 17.7σ;
- raw logical flip: tableau 7.665% vs fast 5.725% (z = +7.8);
- decoded: tableau 2.495% vs fast 0.425% (z = +17.3, 5.9× optimistic).

Output: `research/data/qec/audit_OLD_main86e5d67_d3_p0.005_20k.txt`.

## 6. Discriminating noise tests (`tests/noise.rs`)

- `single_qubit_depolarizing_and_readout_from_zero_in_z_and_x_basis` starts
  from |0⟩ and runs on the tableau, the state vector and the MPS:
  - Z basis: `Z(0)` (trivial on |0⟩, but it carries 1q noise), then measure;
  - X basis: `H, H`, then measure;
  - each is compared with an exact 2×2 density-matrix calculation within
    5 binomial σ (20k shots);
  - the test also asserts that "no noise", "no gate noise" and "no readout
    noise" would each be more than 8σ away, so a backend that ignored any
    channel fails.
- `two_qubit_depolarizing_after_cnot_zz_and_xx_parity`: a Bell pair from
  H+CNOT with p_2q = 0.15 (1q noise off). The ZZ parity and the XX parity
  each flip with 8p/15 on the tableau AND the state vector. A channel
  inserting only X-type or only Z-type errors fails one of the two.

## 7. Cost

| d | method | shots | wall (1 thread, shared VM) |
|---|---|---|---|
| 3 | tableau | 5000 | 0.25 s |
| 3 | DEM | 5000 | 0.01 s |
| 5 | tableau | 2000 | 0.55 s |
| 5 | DEM | 2000 | 0.02 s |
| 9 | DEM | 100000 | 4.7 s |

At d=9 the time goes to the Union-Find decode, not to sampling.
