# Colour-code flags: full circuit distance as real circuits, and what it buys

Branch `exp/colour-flags` (from `main` = e7e102d). Code: `src/qec/color.rs` (`memory_flagged`, `flag_slots`, `resources`, `parse_schedule_spec`), `examples/color_search.rs`, `examples/color_ler.rs`, `tests/colour_flags.rs`. Data and scripts: `research/data/colour-flags/`.

This follows `research/colour-global.md` §6. There, full circuit distance d for the triangular 6.6.6 colour code (Kishony–Fowler (K–F) layout, arXiv:2603.28852) needed every boundary-touching plaquette (3d − 6 of them) to be hook-free, plus a 7th CNOT layer from d = 9. Those schedules were checked only in an idealised symbolic model (flagged hooks treated as *absent*, flag CNOTs not in the timing). The Rust generator had no flag qubits. Noise is the noisy-CNOT model (`DEPOLARIZE2(p)` after every CNOT) unless stated. Memory runs over d rounds.

## Headline

__HEADLINE__

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

So the cost is 3d − 6 flag qubits (+33%, +27% and +23% qubits at d = 5, 7, 9), +4 CNOT per flag per round (+43%, +33%, +27% CNOTs), and 4 extra CNOT layers per round (6 at d = 9, which also needs a 7th data layer). Under the noisy-CNOT model the extra layers cost nothing (no idle noise); the extra CNOTs and their faults are fully counted. Under uniform or SI1000 noise the extra layers would add idle errors on every data qubit; that was not measured (§7).

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
| 9 | **9** | Z / X | **9 / 9** (25 min each) | | __D9R9STIM__ | not run |
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
- **so light-Tesseract failures at d = 9 are mostly decoder failures**, and it cannot resolve schedule differences there. colour-global's "Tesseract sees no LER gain for D8 (0.95× [0.61, 1.48])" used that setting and should be read as decoder-limited (re-measured in 5.2).
- At d = 5 light, medium and full gave identical failures on 6,144-shot samples.

All Tesseract numbers below use **orders 4, beam 8, beam climbing, full DEM (all detectors incl. flags)**. They are labelled "Tesseract (4/8)".

Decoding only the memory-basis sector, as our BP+OSD does, makes Tesseract about 70× faster at d = 9 but about 3× worse in absolute terms at d = 5 (148 vs 52 failures on the same 24,576 K–F shots). The sector restriction, not BP+OSD itself, is most of the gap to Tesseract.

### 5.2 Tesseract (4/8), full DEM

__TESS_TABLE__

### 5.3 BP+OSD-CS (order 100, memory-basis sector incl. flag detectors)

__BPOSD_TABLE__

### 5.4 Equal footprint

__FOOTPRINT__

## 6. Chromobius

Chromobius needs every detector annotated with colour and basis. Flag detectors were annotated −1 ("ignore"); `chromobius_ler.py` does this.
- **d = 5 HF:** it compiles, since every plaquette is flagged and the remaining hooks decompose. But it decodes badly: p_L = 1.34% per shot at p = 0.3%, against 0.04% (Tesseract) and 0.17% (BP+OSD) on the same circuit. It throws away the flag information, so the flagged hooks look like ordinary correlated errors.
- **d = 7, 9 HF and K–F + flags (all d):** it fails to compile ("Failed to decompose a complex error instruction"), on the unflagged interior hexagon hooks. This is the same failure as on K–F's own circuit.

So Chromobius works on neither circuit in a useful way. A flag-aware Chromobius would be needed.

## 7. Caveats

- Noisy-CNOT model only, the model K–F's boundary analysis and colour-global's search used. The HF circuits have 16–18 CNOT layers per round against K–F's 12. Under uniform depolarizing or SI1000 noise the extra layers add idle errors on all data qubits, and the comparison could shift. Not measured.
- Z memory for LER; X memory checked for distance only (both bases certified d at full rounds).
- The HF schedules are colour-global's first-found ones; N_min was not optimised.
- Tesseract at orders 4 / beam 8 is "near-optimal" only in the sense of §5.1: it corrected every light failure examined. A full-setting (16 / 15) run at d = 9 over enough shots was not affordable (about 0.2 s per shot per core).
- Footprint comparisons interpolate log p_L linearly in qubit count between adjacent K–F distances (K–F's own figures use √qubits; the conclusion is the same, §5.4).
- d = 11: circuit distance checked at 1 round, Z only, for the all-flags schedule. No LER.

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
5. The LER answer (§5) and the decoder caveat (§5.1): light Tesseract is decoder-limited at d = 9, so colour-global's "no LER gain" for D8 is not established either way at that setting.

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
python campaign.py jobs_vps_bposd.json out.jsonl    # BP+OSD arms; jobs_*_M*.json for Tesseract (4/8)
python make_tables.py m_ runs/*tess_M*.jsonl
```
