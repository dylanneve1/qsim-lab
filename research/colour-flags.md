# Colour-code flags: full circuit distance as real circuits, and what it buys

Branch `exp/colour-flags` (from `main` = e7e102d). Code: `src/qec/color.rs` (`memory_flagged`, `flag_slots`, `resources`, `parse_schedule_spec`), `examples/color_search.rs`, `examples/color_ler.rs`, `tests/colour_flags.rs`. Data and scripts: `research/data/colour-flags/`.

This follows `research/colour-global.md` §6. There, full circuit distance d for the triangular 6.6.6 colour code (Kishony–Fowler (K–F) layout, arXiv:2603.28852) needed every boundary-touching plaquette (3d − 6 of them) to be hook-free, plus a 7th CNOT layer from d = 9. Those schedules were checked only in an idealised symbolic model (flagged hooks treated as *absent*, flag CNOTs not in the timing). The Rust generator had no flag qubits. Noise is the noisy-CNOT model (`DEPOLARIZE2(p)` after every CNOT) unless stated. Memory runs over d rounds.

## Headline

- **Full circuit distance as real circuits.** Put one flag qubit on each of the 3d − 6 boundary-touching plaquettes and use colour-global's hook-free boundary schedules. The circuits then reach **d_circ = d exactly** at d = 5, 7 and 9:
  - in both bases, over d rounds;
  - certified exact with our branch and bound;
  - cross-checked by the same search on Stim's own DEM, and by Stim's upper bound at fewer rounds.

  K–F has d − ⌊(d+3)/6⌋ = 4 / 6 / 7. The cost at d = 9 is +21 qubits (91 → 112), 312 → 396 CNOTs and 12 → 18 CNOT layers per round.
- **Under K–F's noisy-CNOT model the flagged circuits cut the logical error per round 3–10× against K–F:**
  - Tesseract (orders 4, beam 8), p = 0.3%: **0.26× [0.21, 0.33] at d = 5 and 0.27× [0.16, 0.45] at d = 7**;
  - Tesseract, d = 5, p = 0.1%: **0.09× [0.03, 0.21]**;
  - BP+OSD at d = 9: **0.42× [0.34, 0.53] at p = 0.3% and 0.28× [0.19, 0.40] at p = 0.2%**.

  They also win at **equal qubit count**: 0.35–0.48× against K–F interpolated in qubits (BP+OSD), and HF d = 9 (112 qubits) is 0.47× K–F d = 11 (136 qubits).
- **But most of that gain is the flags, not the odd distance step.** K–F's own schedule with the same flags has d_circ = 5 / 6 / 8 and identical resources.
  - At d = 5 and 7 it is statistically indistinguishable from the d_circ = d circuit: HF / K–F + flags = 0.96 [0.75, 1.23] (d = 5) and 1.34 [0.64, 2.81] (d = 7), Tesseract, p = 0.3%.
  - The odd step pays only at d = 9: **0.69× [0.55, 0.86] at p = 0.3% and 0.55× [0.36, 0.82] at p = 0.2%** against the same-resource d − 1 circuit. That is measured with BP+OSD; Tesseract at d = 9 was not affordable, §5.1.
- **The gain reverses under uniform depolarizing noise.** At p = 0.1% the flagged circuits are 1.24× [1.09, 1.41] (d = 5) and 1.47× [1.24, 1.75] (d = 7) *worse* than K–F with BP+OSD, and 1.18× [0.92, 1.53] / 1.00× [0.43, 2.31] with Tesseract. The extra layers bring idle noise, and the flag CNOTs and readouts add fault locations; d_circ is unchanged.
- **New structural results:**
  - colour-global's absent-hook model is optimistic: it certifies some d − 1 circuits as d. An exact flag model now matches the circuit DEM.
  - **At d = 11 boundary flags cannot reach d**: DRAT-verified UNSAT even with any depth and decoupled X/Z schedules. Flagging all 45 plaquettes reaches 11 at 1 round.
  - **Light Tesseract (1 order, beam 5), used in colour-global, is decoder-limited at d = 9**: every one of its failures examined was corrected by a heavier setting.

## 1. Construction: one flag qubit per boundary plaquette

`ColorCode::memory_flagged(schedule, flagged, rounds, noise, basis)`. With no flags the circuit is op-for-op identical to `memory_basis`; `tests/colour_flags.rs::no_flags_is_the_unflagged_circuit` checks this.

Each flagged plaquette p gets one flag qubit. It is reset and measured in the same moments as p's auxiliary and serves both halves of every round:

| half | auxiliary | flag | flag CNOTs | catches |
|---|---|---|---|---|
| Z half (`CX data→anc`) | `R` (\|0⟩) | `RX` (\|+⟩), measured in X | `CX flag→anc` at slots s1, s2 | Z on the auxiliary between s1 and s2 (a Z hook; X-type sector) |
| X half (`CX anc→data`) | `RX` (\|+⟩) | `R` (\|0⟩), measured in Z | `CX anc→flag` at s1, s2 | X on the auxiliary between s1 and s2 (an X hook; Z-type sector) |

Every flag measurement is its own detector (deterministic: the flag is fresh each half). Flag CNOTs get the same `DEPOLARIZE2(p)` as data CNOTs, and flags get idle noise under the uniform model.

**Where the flag CNOTs go** (`flag_slots`). Let t_1 < … < t_w be p's data-CNOT steps. Then:
- s1 < t_2;
- s2 > t_{w−1}, **not** just > t_{w−2};
- each slot is a step at which p's auxiliary is idle;
- idle in-schedule steps are preferred (no extra depth); otherwise step 0 is used (one extra CNOT layer before the data layers) or step T + 1 (one after them).

The s2 rule matters. My first version used s2 > t_{w−2}, which seems enough: hooks after the (w−1)-th CNOT are single-qubit. But an X on the auxiliary created *by the second flag CNOT's own fault* still spreads to the last two data qubits, unflagged. That version gave d_circ = 6 at d = 7. The circuit DEM showed the unflagged weight-2 hook directly (§3).

**Resources per round** (`color_search resources`):

| d | circuit | qubits (data + aux + flags) | CNOT layers per round | CNOTs per round | d_circ |
|---|---|---|---|---|---|
| 5 | K–F | 28 (19 + 9 + 0) | 12 | 84 | 4 |
| 5 | **HF** (flags + hook-free schedule, 6 data layers) | **37** (+ 9 flags) | **16** | **120** | **5** |
| 5 | K–F + boundary flags | 37 | 16 | 120 | 5 |
| 7 | K–F | 55 | 12 | 180 | 6 |
| 7 | **HF** (6 data layers) | **70** (+ 15) | **16** | **240** | **7** |
| 7 | K–F + boundary flags | 70 | 16 | 240 | 6 |
| 9 | K–F | 91 | 12 | 312 | 7 |
| 9 | global D8 (colour-global, K–F space) | 91 | 12 | 312 | 8 |
| 9 | **HF** (7 data layers) | **112** (+ 21) | **18** | **396** | **9** |
| 9 | K–F + boundary flags | 112 | 16 | 396 | 8 |
| 11 | K–F | 136 | 12 | 480 | 9 |
| 11 | all 45 plaquettes flagged (6 data layers) | 181 (+ 45) | 16 | 660 | 11 (1 round, §4) |
| 13 | K–F | 190 | 12 | 684 | 10 |

So the cost is 3d − 6 flag qubits (+33%, +27% and +23% qubits at d = 5, 7, 9), +4 CNOT per flag per round (+43%, +33%, +27% CNOTs), and 4 extra CNOT layers per round (6 at d = 9, which also needs a 7th data layer). Under the noisy-CNOT model the extra layers cost nothing (no idle noise); the extra CNOTs and their faults are fully counted. Under uniform depolarizing noise the extra layers add idle errors on every data qubit, and the flagged circuits lose at p = 0.1% (§5.4); SI1000 was not run.

HF schedules: `schedules/d5_hf_T6.sched`, `d7_hf_T6.sched`, `d9_hf_T7.sched`. Each is one line per plaquette with the steps for positions a–f, and a trailing `F` marks a flag. They are colour-global's `*_hookfree_boundary_*.found.json` schedules, unchanged, and the flags sit exactly on the 3d − 6 boundary-touching plaquettes (`ColorCode::boundary_plaquettes`, test `flagged_plaquettes_are_the_boundary_ones`).

## 2. Verification: d_circ = d exactly

Z- and X-basis memory, noisy-CNOT model. "Ours" is the exact branch and bound (`src/qec/distance.rs`) on our circuit DEM (SymPhase), memory-basis sector, *certified* (the example logical lifts to the full DEM). So it gives the exact distance and an explicit weight-d logical. "Stim DEM ≥" runs the same branch and bound on **Stim's own DEM** of the exported `.stim` (sector projection: lower bound). "Stim ≤" is Stim 1.16 `search_for_undetectable_logical_errors` (ev ≤ 4, deg ≤ 10; upper bound).

| d | rounds | basis | ours (exact, certified) | N_min | Stim DEM ≥ | Stim ≤ |
|---|---|---|---|---|---|---|
| 5 | 1 | Z / X | **5 / 5** | 483 / 508 | | |
| 5 | **5** | Z / X | **5 / 5** | 3,470 / 3,501 | 5 / 5 | 5 / 5 |
| 7 | 1 | Z / X | **7 / 7** | 5,074 / 4,927 | | |
| 7 | 3 | Z / X | | | 7 / 7 | 7 / 7 |
| 7 | **7** | Z / X | **7 / 7** (35 s each) | 80,664 / 80,331 | 7 / 7 | not run (Stim's search needs > 4 GB here) |
| 9 | 1 | Z / X | **9 / 9** (100 s) | 61,573 / 58,111 | 9 / 9 | 9 / 9 |
| 9 | 3 | Z / X | **9 / 9** (250 s) | | | |
| 9 | **9** | Z / X | **9 / 9** (25 min each) | | 9 / 9 (25–30 min each) | not run |
| 11 | 1 | Z | **11** (all 45 plaquettes flagged, 27 min) | | | |

Also checked:
- **Detectors are deterministic**, flags included. Both bases, both noise models, 1–2 rounds, at d = 5, 7, for HF and for K–F + boundary flags (`tests/colour_flags.rs`). Stim also builds every exported DEM, and it rejects non-deterministic detectors.
- **K–F's own schedule + the same boundary flags** reaches 5 at d = 5, but only **6 at d = 7** (7 rounds) and **8 at d = 9** (1 round). Flags alone are not enough: the schedule has to change too, as colour-global's model said.
- The minimum-weight count N_min of the HF circuits is large (61,573 at d = 9 over 1 round, against K–F's 36 at weight 7). Most weight-d logicals are plain boundary strings, repeated over space-time positions. N_min was not optimised. A second d = 7 schedule found by the flag-aware search (§3; `d7_fpk_T6.sched`) has 68,091 against 80,664 over 7 rounds, so the count is mostly intrinsic.

Pinned in `tests/colour_flags.rs`:
- d = 5 over 5 rounds, both bases;
- d = 7 over 1 round, both bases;
- K–F + flags = 6 at d = 7;
- d = 7 over 7 rounds and d = 9 over 1 round, both bases (`#[ignore]`, release, about 1–4 min).

## 3. The absent-hook model is optimistic; an exact flag model

colour-global's `--hookfree` treats a flagged plaquette as having *no* multi-qubit hooks. In a real flag circuit the hooks are still there, now also flipping the flag detector, and a flag-only fault exists (an X on the flag from a flag CNOT's `DEPOLARIZE2`). So 2 faults on one flagged plaquette in one round give either:
- any suffix of its X-half order (a hook plus a flag-only fault), or
- any contiguous segment of it (two hooks).

On a hexagon that is a weight-3 data error for 2 faults.

`cgf_sat.py` is colour-global's CEGAR with this exact flag model. For a flagged plaquette p and layer l, the unflagged multi-qubit hooks are replaced by:
- `syn(S)@l + F(p,l)` for every suffix S of p's order, 1 ≤ |S| ≤ w − 1;
- `F(p,l)` alone.

`cgf_sat.py verify` compares the model's signature set with the Rust flagged-circuit DEM. The sets are **identical** for the HF, `hfpk`, `fpk` and K–F + flags schedules at d = 5, 7 and 9, over 1 and 2 rounds.

Consequences:
- **The absent-hook model's FOUND results are not sufficient.** Schedules that it certifies at D = d, found with K–F decision phases (`--phase-kf`; `schedules/d7_hfpk_T6.sched`, `d9_hfpk_T7.sched`), are **d − 1 as circuits**: 6 at d = 7 and 8 at d = 9. The minimum logicals use two flagged hooks of one plaquette in one round, e.g. `[f10@0 5@1 7@1 10@1] [f10@0 12@1]`. colour-global's specific `*_hookfree_boundary_*` schedules happen to be fine (§2).
- **Its UNSAT results transfer.** The flagged circuit's DEM contains the absent-hook model's DEM, so "no schedule reaches D" in the relaxed model holds for real flag circuits too.
- With the exact model the CEGAR finds valid flagged schedules directly. At d = 7 it found `d7_fpk_T6.sched` (d_circ = 7 over 7 rounds, certified) in 12 iterations.

## 4. d = 11: boundary flags are not enough

Each query asks whether a schedule with D = 11 exists over 1 round (which bounds any number of rounds), with all 27 boundary-touching plaquettes hook-free in the absent-hook model. Since that model is optimistic, the UNSATs hold for flag circuits.

| space | result | iterations / cuts / time | certificate |
|---|---|---|---|
| one schedule, 7 + 7 layers | **UNSAT** | 124 / 15,301 / 324 s | `certs/d11_D11_hfbnd_kfT7`, drat-trim **VERIFIED** |
| one schedule, 8 + 8 layers | **UNSAT** | 134 / 17,470 / 362 s | `d11_D11_hfbnd_kfT8`, **VERIFIED** |
| any orders, any depth, X/Z decoupled | **UNSAT** | 104 / 14,262 / 322 s | `d11_D11_hfbnd_free`, **VERIFIED** |
| exact flag model, **all 45 plaquettes flagged**, one 7-layer-capable schedule | **FOUND** at iteration 1 (the first candidate already has distance 11). It uses only steps 1–6. | 1,163 s (one distance call) | `schedules/d11_allflags_T6.sched`; Rust circuit DEM, Z, 1 round: **11, certified** (1,625 s) |

So colour-global's §6 pattern (3d − 6 boundary flags + depth) stops at d = 9. At d = 11 even any-depth, decoupled schedules cannot reach 11 with only the boundary ring hook-free. Flagging every plaquette works with 6 data layers. The minimal flag set at d = 11 is open: a scan over K hook-free plaquettes chosen by the solver was started and stopped for CPU. The all-flags circuit costs 181 qubits, about K–F's d = 13 (190). Only 1 round and Z were checked.

## 5. Logical error rate: does the odd step pay?

Three arms at every (d, p), independent samples, same decoder per table:
- **K–F**;
- **HF** (d_circ = d);
- **K–F + boundary flags**: the same qubits, flags and CNOT count as HF, but d_circ = 5 / 6 / 8. This isolates the schedule (and the odd step) from the flags.

At d = 9 the colour-global D8 schedule is added (K–F's resources, d_circ = 8). Per-round p_L = (1 − (1 − 2P)^{1/R})/2 with Wilson 95% CIs. Ratio CIs use the log-ratio normal approximation (`make_tables.py`).

### 5.1 A decoder caveat first: light Tesseract is not near-optimal here

colour-global's Tesseract comparison used `det_orders = 1, det_beam = 5`. At d = 9, p = 0.3% I re-decoded every shot that this light setting got wrong with the heavier settings (`redecode.py`, 16,384-shot chunks):
- **all 5 failures on K–F's circuit and all 8 on the HF circuit are corrected** by orders 4 / beam 8, and by K–F's 16 / 15;
- **so light-Tesseract failures at d = 9 are mostly decoder failures**, and it cannot resolve schedule differences there. colour-global's "Tesseract sees no LER gain for D8 (0.95× [0.61, 1.48])" used that setting and should be read as decoder-limited. Re-measuring D8 at d = 9 with Tesseract (4/8) was not affordable (5.2), so only the BP+OSD comparison (5.3) remains.
- At d = 5 light, medium and full gave identical failures on 6,144-shot samples.

All Tesseract numbers below use **orders 4, beam 8, beam climbing, full DEM (all detectors incl. flags)**. They are labelled "Tesseract (4/8)".

Decoding only the memory-basis sector, as our BP+OSD does, makes Tesseract about 70× faster at d = 9 but about 3× worse in absolute terms at d = 5 (148 vs 52 failures on the same 24,576 K–F shots). The sector restriction, not BP+OSD itself, is most of the gap to Tesseract.

### 5.2 Tesseract (4/8), full DEM

Pooled over the VPS (x86) and the Mac (M1); the two machines draw independent samples.

| d | p | arm | fails / shots | p_L per round [95% CI] | ratio vs K–F [95% CI] |
|---|---|---|---|---|---|
| 5 | 0.1% | K–F | 58 / 491,520 | 2.36e-05 [1.83e-05, 3.05e-05] | — |
| 5 | 0.1% | HF (flags + hook-free schedule) | 5 / 491,520 | 2.03e-06 [8.69e-07, 4.76e-06] | **0.09** [0.03, 0.21] |
| 5 | 0.1% | K–F + boundary flags | 6 / 393,216 | 3.05e-06 [1.40e-06, 6.66e-06] | **0.13** [0.06, 0.30] |
| 5 | 0.2% | K–F | 142 / 196,608 | 1.45e-04 [1.23e-04, 1.70e-04] | — |
| 5 | 0.2% | HF (flags + hook-free schedule) | 75 / 540,672 | 2.77e-05 [2.21e-05, 3.48e-05] | **0.19** [0.15, 0.25] |
| 5 | 0.2% | K–F + boundary flags | 52 / 491,520 | 2.12e-05 [1.61e-05, 2.77e-05] | **0.15** [0.11, 0.20] |
| 5 | 0.3% | K–F | 161 / 98,304 | 3.28e-04 [2.81e-04, 3.83e-04] | — |
| 5 | 0.3% | HF (flags + hook-free schedule) | 127 / 294,912 | 8.62e-05 [7.24e-05, 1.03e-04] | **0.26** [0.21, 0.33] |
| 5 | 0.3% | K–F + boundary flags | 132 / 294,912 | 8.96e-05 [7.55e-05, 1.06e-04] | **0.27** [0.22, 0.34] |
| 7 | 0.3% | K–F | 64 / 229,376 | 3.99e-05 [3.12e-05, 5.09e-05] | — |
| 7 | 0.3% | HF (flags + hook-free schedule) | 17 / 229,376 | 1.06e-05 [6.61e-06, 1.70e-05] | **0.27** [0.16, 0.45] |
| 7 | 0.3% | K–F + boundary flags | 12 / 217,088 | 7.90e-06 [4.52e-06, 1.38e-05] | **0.20** [0.11, 0.37] |

At d = 7, p = 0.3% the failures that Tesseract (4/8) leaves on the HF circuit are genuine. On a Mac chunk with 5 failures, all 5 persist at 16 / 15 and at 32 / 30 (`redecode.py`, `BASE=4,8`); their syndrome weights are 15–40.

**d = 9 with a near-optimal decoder was not affordable.** Tesseract (4/8) on the full d = 9 DEM decodes about 3.4 shots/s on 3 VPS cores at p = 0.5%. K–F gave 1 failure in 3,072 shots there, against about 14 expected from the light setting's rate. Getting about 50 failures per arm at p = 0.3% would take on the order of a day of the shared machines; the Mac was saturated by other agents and held under the bench lock for most of the run. So d = 9 LER is BP+OSD only (5.3).

**Light Tesseract (1/5) on the same circuits** (`t_` tags, kept for the record). At d = 9 it sees *no* difference (HF / K–F = 1.11 [0.61, 2.00]); §5.1 shows its failures there are decoder failures:

| d | p | arm | fails / shots | p_L per round [95% CI] | ratio vs K–F [95% CI] |
|---|---|---|---|---|---|
| 7 | 0.3% | K–F | 43 / 92,160 | 6.67e-05 [4.95e-05, 8.98e-05] | — |
| 7 | 0.3% | HF (flags + hook-free schedule) | 15 / 61,440 | 3.49e-05 [2.11e-05, 5.76e-05] | **0.52** [0.29, 0.94] |
| 7 | 0.3% | K–F + boundary flags | 6 / 61,440 | 1.40e-05 [6.39e-06, 3.04e-05] | **0.21** [0.09, 0.49] |
| 9 | 0.3% | K–F | 23 / 69,632 | 3.67e-05 [2.45e-05, 5.51e-05] | — |
| 9 | 0.3% | HF (flags + hook-free schedule) | 21 / 57,344 | 4.07e-05 [2.66e-05, 6.22e-05] | **1.11** [0.61, 2.00] |
| 9 | 0.3% | K–F + boundary flags | 5 / 24,576 | 2.26e-05 [9.66e-06, 5.29e-05] | **0.62** [0.23, 1.62] |
| 9 | 0.3% | global D8 (colour-global) | 7 / 12,288 | 6.33e-05 [3.07e-05, 1.31e-04] | **1.73** [0.74, 4.02] |

### 5.3 BP+OSD-CS (order 100, memory-basis sector incl. flag detectors)

| d | p | arm | fails / shots | p_L per round [95% CI] | ratio vs K–F [95% CI] |
|---|---|---|---|---|---|
| 5 | 0.1% | K–F | 197 / 384,000 | 1.03e-04 [8.93e-05, 1.18e-04] | — |
| 5 | 0.1% | HF (flags + hook-free schedule) | 150 / 2,816,000 | 1.07e-05 [9.08e-06, 1.25e-05] | **0.10** [0.08, 0.13] |
| 5 | 0.1% | K–F + boundary flags | 159 / 2,304,000 | 1.38e-05 [1.18e-05, 1.61e-05] | **0.13** [0.11, 0.17] |
| 5 | 0.2% | K–F | 283 / 128,000 | 4.43e-04 [3.94e-04, 4.98e-04] | — |
| 5 | 0.2% | HF (flags + hook-free schedule) | 213 / 448,000 | 9.51e-05 [8.32e-05, 1.09e-04] | **0.21** [0.18, 0.26] |
| 5 | 0.2% | K–F + boundary flags | 224 / 448,000 | 1.00e-04 [8.78e-05, 1.14e-04] | **0.23** [0.19, 0.27] |
| 5 | 0.3% | K–F | 341 / 64,000 | 1.07e-03 [9.62e-04, 1.19e-03] | — |
| 5 | 0.3% | HF (flags + hook-free schedule) | 329 / 192,000 | 3.43e-04 [3.08e-04, 3.82e-04] | **0.32** [0.28, 0.37] |
| 5 | 0.3% | K–F + boundary flags | 370 / 256,000 | 2.89e-04 [2.61e-04, 3.20e-04] | **0.27** [0.23, 0.31] |
| 7 | 0.1% | K–F | 102 / 1,177,600 | 1.24e-05 [1.02e-05, 1.50e-05] | — |
| 7 | 0.1% | HF (flags + hook-free schedule) | 28 / 1,638,400 | 2.44e-06 [1.69e-06, 3.53e-06] | **0.20** [0.13, 0.30] |
| 7 | 0.1% | K–F + boundary flags | 18 / 1,638,400 | 1.57e-06 [9.93e-07, 2.48e-06] | **0.13** [0.08, 0.21] |
| 7 | 0.2% | K–F | 166 / 307,200 | 7.72e-05 [6.63e-05, 8.99e-05] | — |
| 7 | 0.2% | HF (flags + hook-free schedule) | 131 / 819,200 | 2.28e-05 [1.93e-05, 2.71e-05] | **0.30** [0.24, 0.37] |
| 7 | 0.2% | K–F + boundary flags | 132 / 819,200 | 2.30e-05 [1.94e-05, 2.73e-05] | **0.30** [0.24, 0.37] |
| 7 | 0.3% | K–F | 230 / 102,400 | 3.21e-04 [2.83e-04, 3.66e-04] | — |
| 7 | 0.3% | HF (flags + hook-free schedule) | 213 / 307,200 | 9.91e-05 [8.67e-05, 1.13e-04] | **0.31** [0.26, 0.37] |
| 7 | 0.3% | K–F + boundary flags | 217 / 256,000 | 1.21e-04 [1.06e-04, 1.38e-04] | **0.38** [0.31, 0.45] |
| 9 | 0.2% | K–F | 102 / 320,000 | 3.54e-05 [2.92e-05, 4.30e-05] | — |
| 9 | 0.2% | HF (flags + hook-free schedule) | 36 / 409,600 | 9.77e-06 [7.05e-06, 1.35e-05] | **0.28** [0.19, 0.40] |
| 9 | 0.2% | K–F + boundary flags | 66 / 409,600 | 1.79e-05 [1.41e-05, 2.28e-05] | **0.51** [0.37, 0.69] |
| 9 | 0.2% | global D8 (colour-global) | 66 / 409,600 | 1.79e-05 [1.41e-05, 2.28e-05] | **0.51** [0.37, 0.69] |
| 9 | 0.3% | K–F | 150 / 102,400 | 1.63e-04 [1.39e-04, 1.91e-04] | — |
| 9 | 0.3% | HF (flags + hook-free schedule) | 151 / 243,200 | 6.90e-05 [5.89e-05, 8.10e-05] | **0.42** [0.34, 0.53] |
| 9 | 0.3% | K–F + boundary flags | 161 / 179,200 | 9.99e-05 [8.56e-05, 1.17e-04] | **0.61** [0.49, 0.77] |
| 9 | 0.3% | global D8 (colour-global) | 158 / 166,400 | 1.06e-04 [9.04e-05, 1.23e-04] | **0.65** [0.52, 0.81] |
| 11 | 0.2% | K–F | 34 / 147,200 | 2.10e-05 [1.50e-05, 2.93e-05] | — |
| 11 | 0.3% | K–F | 104 / 64,000 | 1.48e-04 [1.22e-04, 1.79e-04] | — |

At d = 9 the two d − 1 = 8 circuits are indistinguishable: K–F + flags / global D8 = 0.95 [0.76, 1.18] at 0.3% and 1.00 [0.71, 1.41] at 0.2%. With K–F's resources, the flags alone buy nothing at d = 9 over a better schedule. The d_circ = 9 circuit beats both:
- HF / K–F + flags = **0.69 [0.55, 0.86]** (0.3%) and **0.55 [0.36, 0.82]** (0.2%);
- HF / D8 = 0.65 [0.52, 0.82] and 0.55 [0.36, 0.82].

At d = 5 and 7 HF / K–F + flags ranges over 0.77–1.56, with no consistent sign (`ratio_pair.py`).

### 5.4 Uniform depolarizing noise (p = 0.1%; idle, reset and readout noise as K–F's uniform model)

d_circ is unchanged by this noise model: HF 5 / 7, K–F 4 / 6 (`NOISE=uniform ./dist.sh`).

BP+OSD:

| d | p | arm | fails / shots | p_L per round [95% CI] | ratio vs K–F [95% CI] |
|---|---|---|---|---|---|
| 5 | 0.1% | K–F | 409 / 64,000 | 1.28e-03 [1.17e-03, 1.42e-03] | — |
| 5 | 0.1% | HF (flags + hook-free schedule) | 506 / 64,000 | 1.59e-03 [1.46e-03, 1.74e-03] | **1.24** [1.09, 1.41] |
| 5 | 0.1% | K–F + boundary flags | 596 / 64,000 | 1.88e-03 [1.73e-03, 2.03e-03] | **1.46** [1.29, 1.66] |
| 7 | 0.1% | K–F | 247 / 102,400 | 3.45e-04 [3.05e-04, 3.91e-04] | — |
| 7 | 0.1% | HF (flags + hook-free schedule) | 272 / 76,800 | 5.07e-04 [4.51e-04, 5.72e-04] | **1.47** [1.24, 1.75] |
| 7 | 0.1% | K–F + boundary flags | 270 / 76,800 | 5.04e-04 [4.47e-04, 5.68e-04] | **1.46** [1.23, 1.73] |

Tesseract (4/8):

| d | p | arm | fails / shots | p_L per round [95% CI] | ratio vs K–F [95% CI] |
|---|---|---|---|---|---|
| 5 | 0.1% | K–F | 124 / 32,768 | 7.59e-04 [6.37e-04, 9.05e-04] | — |
| 5 | 0.1% | HF (flags + hook-free schedule) | 110 / 24,576 | 8.98e-04 [7.45e-04, 1.08e-03] | **1.18** [0.92, 1.53] |
| 5 | 0.1% | K–F + boundary flags | 106 / 24,576 | 8.66e-04 [7.16e-04, 1.05e-03] | **1.14** [0.88, 1.48] |
| 7 | 0.1% | K–F | 11 / 16,384 | 9.60e-05 [5.36e-05, 1.72e-04] | — |
| 7 | 0.1% | HF (flags + hook-free schedule) | 11 / 16,384 | 9.60e-05 [5.36e-05, 1.72e-04] | **1.00** [0.43, 2.31] |
| 7 | 0.1% | K–F + boundary flags | 14 / 16,384 | 1.22e-04 [7.28e-05, 2.05e-04] | **1.27** [0.58, 2.80] |

The flags do not pay here at p = 0.1%:
- every data qubit idles through 4 extra layers per round (16 vs 12), about +33% data-qubit fault locations per round;
- the flags add reset, readout and idle faults of their own.

At this p that outweighs the extra distance. A lower p, or flag CNOTs merged into existing layers (only possible where a plaquette has idle in-schedule steps, §1), would be needed. SI1000 was not run.

### 5.5 Equal footprint

K–F's per-round p_L interpolated log-linearly in total qubits between adjacent K–F distances (28 / 55 / 91 / 136 qubits at d = 5 / 7 / 9 / 11), evaluated at each HF circuit's qubit count (`footprint.py`; noisy-CNOT model).

| decoder | p | HF d = 5 (37 q) vs K–F at 37 q | HF d = 7 (70 q) vs K–F at 70 q | HF d = 9 (112 q) vs K–F at 112 q | HF d = 9 vs K–F d = 11 (136 q) |
|---|---|---|---|---|---|
| BP+OSD | 0.3% | 0.48 [0.42, 0.55] | 0.41 [0.35, 0.48] | 0.44 [0.36, 0.54] | 0.47 |
| BP+OSD | 0.2% | 0.38 [0.33, 0.45] | 0.41 [0.33, 0.50] | 0.35 [0.24, 0.51] | 0.47 |
| BP+OSD | 0.1% | 0.21 [0.17, 0.26] | — | — | — |
| Tesseract (4/8) | 0.3% | 0.53 [0.43, 0.66] | — (no K–F d = 9) | — | — |

So, under noisy-CNOT noise, the flagged circuits are 2–5× better than K–F at the same footprint, and HF d = 9 beats K–F d = 11 with 24 fewer qubits (BP+OSD). The d = 5 HF circuit does **not** beat K–F d = 7, which has more qubits: HF d = 5 / K–F d = 7 is 1.07 (BP+OSD) and 2.16 (Tesseract) at 0.3%. In the "next distance up" sense the flags win only from d = 7 on. With Tesseract only the d = 5 point exists. Under uniform depolarizing noise (5.4) the footprint comparison is unfavourable, since the flagged circuits already lose at equal d.

## 6. Chromobius

Chromobius needs every detector annotated with colour and basis. Flag detectors were annotated −1 ("ignore"); `chromobius_ler.py` does this.
- **d = 5 HF:** it compiles, since every plaquette is flagged and the remaining hooks decompose. But it decodes badly: p_L = 1.34% per shot at p = 0.3%, against 0.04% (Tesseract) and 0.17% (BP+OSD) on the same circuit. It throws away the flag information, so the flagged hooks look like ordinary correlated errors.
- **d = 7, 9 HF and K–F + flags (all d):** it fails to compile ("Failed to decompose a complex error instruction"), on the unflagged interior hexagon hooks. This is the same failure as on K–F's own circuit.

So Chromobius works on neither circuit in a useful way. A flag-aware Chromobius would be needed.

## 7. Caveats

- **Noise models.**
  - The noisy-CNOT model (K–F's headline model for hooks, and colour-global's) is where flags pay.
  - Under uniform depolarizing noise at p = 0.1% they do not (§5.4). Only one p was run there.
  - SI1000 was not run. K–F's own comparison is per footprint, under all three models; our flagged circuits would need lower p or a shallower flag layout to compete under idle noise.
- **d = 9 LER is BP+OSD only.**
  - BP+OSD decodes the memory-basis sector, so it is not near-optimal: about 3× worse in absolute terms than Tesseract on the full DEM at d = 5.
  - The schedule ratios from the two decoders agree where both exist (d = 5, 7: HF / K–F 0.26 vs 0.32 at d = 5, 0.27 vs 0.31 at d = 7, p = 0.3%).
  - The d = 9 odd-step gain (0.55–0.69×) is therefore *not* confirmed with a near-optimal decoder.
- Tesseract at orders 4 / beam 8 is "near-optimal" only in the sense of §5.1 and §5.2. It corrected every light failure examined at d = 9, and its own failures at d = 7 survive 32 / 30.
- Z memory for LER. X memory was checked for distance only (both bases certified d at full rounds).
- The HF schedules are colour-global's first-found ones, with N_min not optimised. K–F + flags has fewer minimum-weight logicals (2,344 vs 3,470 at d = 5 over 5 rounds), which is consistent with it matching or beating HF at d ≤ 7.
- Statistical arms ran on two machines (x86 VPS, M1 Mac). Stim's sampler output is platform-dependent for the same seed, so the pooled chunks are independent; seeds never repeat within a tag.
- Footprint comparisons interpolate log p_L linearly in qubit count between adjacent K–F distances.
- d = 11: circuit distance checked at 1 round, Z only, for the all-flags schedule. No LER.
- Stim's undetectable-logical search (an upper bound) was run up to d = 7 over 3 rounds and d = 9 over 1 round. At d rounds it needs more than 4 GB, which this shared VPS could not spare. The explicit certified weight-d logical from our branch and bound is the upper bound there.

## 8. Known vs new

**Known.**
- K–F give d_circ = d − ⌊(d+3)/6⌋ and suggest boundary flags at a sub-leading qubit cost as future work.
- Flag qubits for hook errors: Chao–Reichardt; Chamberland et al. for colour codes.
- colour-global: 3d − 6 hook-free boundary plaquettes are necessary, and sufficient at d ≤ 9 in an idealised model.

**New here.**
1. Flag qubits in the generator, both halves and both bases. The construction comes with an explicit flag-slot rule (s1 < t_2, s2 > t_{w−1}) and the counterexample that shows why the obvious s2 > t_{w−2} fails.
2. **d_circ = d exactly as real circuits** at d = 5, 7, 9 over d rounds in both bases. It is exact and certified with our tool, cross-checked on Stim's own DEM, with Stim upper bounds at smaller round counts. K–F + the same flags reaches only 6 / 8 at d = 7 / 9.
3. An exact flag model for the CEGAR. Its signature sets equal the circuit DEM. It shows the absent-hook relaxation is optimistic (it certifies d − 1 circuits as d) while its UNSATs still transfer.
4. **d = 11: the boundary-ring pattern fails.** The UNSATs are DRAT-verified for 7 and 8 layers and for any depth with decoupled X/Z schedules. Flagging all 45 plaquettes reaches 11 (1 round).
5. **LER, measured.**
   - Under noisy-CNOT noise, boundary flags give 3–10× lower logical error per round than K–F at d = 5, 7 with a near-optimal decoder, and 2–5× at equal footprint (BP+OSD).
   - The odd distance step itself pays only at d = 9 (0.55–0.69×, BP+OSD). At d = 5, 7 K–F's schedule with the same flags (d_circ = d or d − 1) does as well.
   - Under uniform depolarizing noise at p = 0.1% the flagged circuits are worse than K–F (1.2–1.5×, BP+OSD).
6. **Decoder caveat.**
   - Light Tesseract (1 order, beam 5) is decoder-limited at d = 9: all 13 of its failures examined were corrected by a heavier setting.
   - So colour-global's "Tesseract sees no gain for D8 (0.95×)" does not establish the absence of a gain.
   - With BP+OSD, D8 is 0.65× [0.52, 0.81] (0.3%) and 0.51× [0.37, 0.69] (0.2%) of K–F here. That reproduces colour-global's BP+OSD numbers.
7. **Decoding only the memory-basis sector loses about 3×** in logical error against full-DEM Tesseract at d = 5, but is about 70× faster at d = 9. Most of the "BP+OSD is 2.5–3× weaker than Tesseract" gap in qec-r4 is this sector restriction.
8. Chromobius works on neither the K–F nor the flagged circuits at d ≥ 7. At d = 5 it runs on the flagged circuit, but only by ignoring the flags, and it is then about 30× worse than Tesseract.

## 9. Reproduce

```bash
export CARGO_TARGET_DIR=/tmp/cf-target CARGO_INCREMENTAL=0
cargo build --release --example color_search --example color_ler --example dem_distance
cargo test --release --test colour_flags            # + -- --ignored (d = 7 x 7 rounds, d = 9)
cd research/data/colour-flags                       # python: stim 1.16, tesseract-decoder, python-sat, chromobius
CS=$CARGO_TARGET_DIR/release/examples/color_search
$CS resources 9 schedules/d9_hf_T7.sched            # 112 qubits, 18 layers, 396 CNOTs
./dist.sh 9 9 schedules/d9_hf_T7.sched z            # 9, certified (~25 min)
$CS export 9 1 cnot 0.001 schedules/d9_hf_T7.sched /tmp/d9.stim && python stim_verify.py /tmp/d9.stim z 8
python cgf_sat.py verify 9 2 schedules/d9_hf_T7.sched   # exact flag model == circuit DEM
python cgf_sat.py 7 7 1 kf --T 6 --warm --sym --phase-kf --out /tmp/d7.sched
./certify_flags.sh python /path/to/drat-trim        # d = 11 UNSAT certificates
python campaign.py jobs_vps_bposd2.json out.jsonl   # BP+OSD arms (light Tesseract: jobs_mac_A3.json, jobs_vps_tess.json; uniform: jobs_vps_uniform*.json); jobs_vps_M7.json / jobs_mac_M4.json: Tesseract (4/8)
python make_tables.py m_ runs/vps_tess_M.jsonl runs/mac_tess_M.jsonl
python ratio_pair.py b_ hf kfflag runs/vps_bposd.jsonl
python footprint.py b_ 0.003 runs/vps_bposd.jsonl
BASE=4,8 python redecode.py 7 7 0.003 schedules/d7_hf_T6.sched 12288 <seed>   # re-decode a chunk's failures
```
