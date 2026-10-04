# Colour-code syndrome schedules: global search and optimality certificates

Branch `exp/colour-global` (from `main` = dd26f4b). Code and data: `research/data/colour-global/`.
This follows `research/qec-r4.md` Part 2. Setting: triangular 6.6.6 colour code, Kishony–Fowler (K–F, arXiv:2603.28852) layout and round structure, Z memory, noisy-CNOT model (`DEPOLARIZE2(p)` after every CNOT) unless stated.

**K–F's design space** (what qec-r4's local search explored):
- one auxiliary per plaquette;
- the same time step t_p(q) ∈ {1..6} for the X half and the Z half, i.e. 6 + 6 CNOT layers per round;
- collision-free: no data qubit in two CNOTs at the same step.

K–F's schedule has d_circ = d − ⌊(d+3)/6⌋: 4, 6, 7, 9, 10 at d = 5, 7, 9, 11, 13.

## Headline

1. **Every single-auxiliary schedule has d_circ ≤ d − 1.** This holds for any order, any depth, any collision pattern, with the X and Z halves decoupled.
   - It follows from a corner lemma (§2), checked for d = 3–15, and independently from DRAT-verified UNSAT certificates (d = 5, 7, 9).
   - So K–F is optimal at d = 5 and 7.
2. **d = 9: K–F is *not* optimal in their own design space.** An exact counterexample-guided SAT search found a schedule with **d_circ = 8** (K–F: 7), which is optimal by item 1. Independent checks:
   - the circuit-level Rust branch and bound gives 8, certified, in both bases over 9 rounds;
   - Stim's undetectable-logical search finds 8 against 7 for K–F, over 1 and 3 rounds;
   - the uniform depolarizing model also gives 8.

   Logical error per round, same decoder, compared with K–F: **0.58× [0.54, 0.62] at p = 0.3%, 0.50× [0.44, 0.57] at 0.2%, 0.41× [0.33, 0.50] at 0.15%**. The ratio falls with p, as an extra unit of distance should. qec-r4's N_min-halving schedule gave 1.02× at p = 0.3%.
3. **d = 11: K–F *is* optimal in their design space**: D = 10 is UNSAT (DRAT-verified; 140,531 cuts). The cheapest relaxation that recovers d − 1 = 10 is **one extra CNOT layer per half** (7 + 7, still one collision-free schedule for both halves). Its circuit-level check gives 10 in both bases (1 round; Stim agrees). Decoupling the X and Z schedules at 6 layers does not help (UNSAT).
4. **Why qec-r4's local search never raised d_circ at d = 9.** Freezing the interior plaquettes at K–F and freeing the 21 boundary-touching ones (exactly that search space) makes 8 impossible (DRAT-verified UNSAT). Interior plaquettes next to the boundary must change.
5. **Reaching the full distance d** needs every boundary-touching plaquette (3d − 6 of them) to be **fully hook-free** (flag-protected, or ≥ 3 auxiliaries on hexagons).
   - 3d − 7 is UNSAT for any choice of plaquettes, any order and any depth (d = 5, 7, 9).
   - The 3 corners alone are not enough.
   - A two-auxiliary cat split, even on every plaquette, is not enough.
   - With all boundary plaquettes hook-free, d is reached in 6 + 6 layers at d = 5 and 7. At d = 9 a 7th layer is also needed: 6 layers is UNSAT even with decoupled X/Z schedules.

## 1. Method

### 1.1 Exact symbolic DEM (`cg_model.py`)

For Z memory under noisy-CNOT noise, every Z-sector mechanism is one of three kinds.

- **Clean data error (layer l).** It flips the layer-l detectors with odd overlap. It is either:
  - a single X (always present), or
  - a *hook*: X on an X-type auxiliary after its k-th CNOT spreads to that plaquette's data qubits met *after* step k, a proper suffix S of its X-half order. S and its complement have the same signature.
- **Partial data error** (Z half of round r). X on data q after its CNOT with plaquette p flips the layer-r detectors of the plaquettes q meets later and the layer-(r+1) detectors of those it met so far (a prefix E of q's Z-half meeting order). An X ⊗ X fault after q's first CNOT gives E = ∅ in round 0.
- **Time-like**: X on a Z-type auxiliary flips layers r and r+1 of that plaquette.

So the DEM depends on the schedule only through two kinds of order literal:
- `bx[p,a,b]`: plaquette p touches a before b in the X half;
- `bz[q,a,b]`: q meets plaquette a before b in the Z half.

Every potential mechanism has an exact *presence condition*, a conjunction of these literals. Two checks against the Rust circuit DEM's Z sector (`color_search dem`), each on 30 schedules (K–F plus random collision-free ones):
- `python cg_model.py 5` (d = 3–9, 1–3 rounds) prints `ALL EQUAL`: the model's signature set equals the circuit's.
- `python test_presence.py 4` (d = 3–9, 1–2 rounds) prints `ALL OK`. For every signature of the universe (all hooks of all plaquettes, all partial errors), "some presence condition used in the CNF holds" ⇔ "the signature is in the circuit DEM", and no circuit signature lies outside the universe. This directly validates the clauses the certificates are built from.

### 1.2 Counterexample-guided SAT (`cg_sat.py`)

The query is "is there a schedule with d_circ ≥ D?". Schedule spaces:

| mode | X-half order | Z-half meeting order | constraint |
|---|---|---|---|
| `kf --T 6` | one time table t_p(q) ∈ 1..T for both halves | same table | collision-free (K–F's space at T = 6) |
| `sep --T 6` | table t^X | independent table t^Z | each collision-free, 1..T |
| `free` | any total order per plaquette | any total order per qubit | none (any depth) |

The time tables use an order encoding; `bx`/`bz` are tied to them by ↔ clauses.

The loop:
1. One variable y_sig per DEM signature is forced true by each of its presence conditions.
2. CaDiCaL 1.9.5 proposes a schedule.
3. `dem_distance` (the exact branch and bound of `src/qec/distance.rs` behind a new stdin CLI, `examples/dem_distance.rs`) lists every minimum-weight logical of weight < D.
4. Each listed logical L adds the clause ∨_{sig ∈ L, conditional} ¬y_sig. This is sound: it removes exactly the schedules that contain L.
5. With `--sym`, the 120° and 240° rotations of L are also added. Each image is asserted to be a logical; that held in every run. This cuts iterations about 2.5–3×.

The loop ends either with a schedule that has no logical of weight < D (**FOUND**, then verified with the circuit-level tools) or with **UNSAT**. On UNSAT the final CNF (encoding + cuts) is re-solved by Glucose with a DRAT proof and checked by `drat-trim`.

- Rounds = 1 is used in the loop. Every 1-round logical maps onto the last round of a longer circuit, so 1-round UNSAT means UNSAT for any number of rounds.
- `--space` (one layer, singles and hooks only) relaxes further, so its UNSATs are stronger.

### 1.3 Relaxation models

- **`--hookfree K`**: up to K plaquettes, chosen by the solver, have no multi-qubit hook (flag-protected, or ≥ 3 auxiliaries on a hexagon). Flag CNOTs are *not* added to the timing, which is optimistic.
- **`--split K`**: two auxiliaries (a cat pair), each measuring half the plaquette. 2 + 2 on a weight-4 plaquette is hook-free. 3 + 3 on a hexagon leaves the last pair of each triple as a hook, and the solver chooses the split and orders. Cat-preparation faults are ignored, which is optimistic, so its UNSATs stand.
- **`--T 7`**: a 7th CNOT layer per half.

## 2. The corner lemma: d_circ ≤ d − 1 for every single-auxiliary schedule

A weight-4 corner plaquette measured by one auxiliary with sequential CNOTs o1 o2 o3 o4 always has the hook X_{o3 o4}, which is X_{o1 o2} times the stabilizer. There are only three pairings {o1 o2 | o3 o4}.

`corner_bound.py` shows that for **each** pairing and each of the three corners, single data errors plus that one hook already give a logical of weight d − 1. All of those faults are clean X-half errors of one round. Hence d_circ ≤ d − 1 for any order, depth, collision pattern, X/Z coupling and number of rounds.

Verified for d = 3, 5, 7, 9, 11, 13, 15 (`runs/corner_bound.txt`). The SAT certificates of §3 reproduce it independently; at d = 5 a deletion-minimal unsat core is 3 logicals, one per corner pairing (`cg_core.py`).

- K–F is therefore optimal at d = 3, 5, 7.
- Their formula sits below the bound from d = 9 on: 7 < 8, 9 < 10, 10 < 12.

## 3. Certificates (all DRAT-verified)

`certify.sh` regenerates the CNFs, `verify_certs.sh` runs `drat-trim`, and `certs/verified.txt` holds the verdicts. The full CNFs, DRAT proofs and cut logicals are in `certs.tar.xz`.

Every row means "no schedule in this space reaches D". The proof is over 1 round, so it holds for any number of rounds.

| certificate | space | D | CNF vars / clauses | cuts | drat-trim |
|---|---|---|---|---|---|
| `d5_D5_space_free` | any orders and depth, space-only | 5 | 300 / 1,040 | 86 | VERIFIED |
| `d7_D7_space_free` | same | 7 | 738 / 3,403 | 967 | VERIFIED |
| `d9_D9_space_free` | same | 9 | 1,368 / 13,761 | 9,171 | VERIFIED |
| `d9_D8_kf_fixint` | K–F space, 9 interior plaquettes fixed to K–F | 8 | 3,504 / 21,631 | 2,488 | VERIFIED |
| `d11_D10_kfT6` | **K–F space (6 + 6 layers)** | 10 | 5,490 / 170,561 | 140,531 | VERIFIED |
| `d11_D10_sepT6` | independent X/Z schedules, 6 layers each | 10 | D11SEPVARS | D11SEPCUTS | D11SEPV |
| `d5_D5_hf8`, `d7_D7_hf14`, `d9_D9_hf20` | any orders, space-only, ≤ 3d − 7 hook-free plaquettes | d | 309 / 991; 812 / 3,197; 1,598 / 8,052 | 36; 659; 3,072 | VERIFIED |
| `d7_D7_hf3corners` | the 3 corners hook-free | 7 | 741 / 3,072 | 636 | VERIFIED |
| `d5/7/9_D=d_split_all` | every plaquette two-auxiliary (cat split) | d | 576 / 1,923; 1,557 / 5,728; 3,000 / 14,741 | 33; 484; 4,535 | VERIFIED |
| `d9_D9_hfbnd_kfT6` | all 21 boundary plaquettes hook-free, K–F timing | 9 | 3,525 / 19,995 | 951 | VERIFIED |
| `d9_D9_hfbnd_sepT6` | same, independent X/Z 6-layer schedules | 9 | 5,397 / 26,394 | 918 | VERIFIED |

What the certificates rely on:
- the order encoding of each space (`cg_sat.py`, `Enc`);
- the presence conditions (§1.1, checked exactly against the circuit DEM);
- each cut being a logical (asserted when generated; stored next to each CNF).

The UNSAT itself is solver-independent, checked by drat-trim.

## 4. d = 9: d_circ = 8 inside K–F's space

`schedules/d9_global_D8.sched` has one line per plaquette with the steps for positions a–f (0 = absent); `.found.json` has the orders. Found by `cg_sat.py 9 8 1 kf --warm` in 195 iterations, 16,876 cuts and 96 s on one VPS core; with `--sym` it takes 77 iterations.

| check | tool | K–F | new |
|---|---|---|---|
| collision-free, steps 1–6, deterministic detectors | `color_search collisions`, `tests/colour_global.rs` | yes | yes |
| Z memory, 1 round | Rust circuit DEM + exact B&B, certified | 7 (N = 36) | **8** (N = 10,119) |
| Z memory, 3 rounds | same | 7 | **8** (19 s) |
| Z memory, 9 rounds | same | 7 (N = 492) | **8** (275 s) |
| X memory, 1 / 9 rounds | same | 7 | **8 / 8** (266 s) |
| uniform depolarizing, Z, 1 round | same | 7 | **8** |
| Stim 1.16 `search_for_undetectable_logical_errors` on the exported circuit (upper bound), 1 / 3 / 9 rounds | Stim | 7 / 7 / 7 | **8 / 8** / (the 9-round search ran out of memory on the VPS) |

The schedule changes almost every plaquette, interior ones included; `d9_D8_kf_fixint` shows the interior *must* change. The price is more minimum-weight logicals at the higher weight (10,119 at weight 8 against 36 at weight 7, over 1 round). That is why the gain grows as p falls.

### 4.1 Logical error rate (d = 9, 9 rounds, noisy CNOT, Z memory)

- Same BP+OSD-CS decoder on both arms (order 100, Z sector, `examples/color_ler.rs`), independent samples.
- Mac M1 Pro, 8 threads. Each ≤ 150 s chunk held the bench lock; arms interleaved chunk by chunk (`ler_chunks.sh`; raw data `cg_ler_d9.jsonl`).
- Ratio CI from the log-ratio normal approximation (`ler_summary.py`).

| p | K–F p_L/round (fails / shots) | new p_L/round (fails / shots) | ratio new / K–F [95% CI] |
|---|---|---|---|
| 0.30% | 2.000e-04 (2157 / 1200128) | 1.162e-04 (1254 / 1200128) | **0.581 [0.542, 0.623]** |
| 0.20% | 3.806e-05 (685 / 2000128) | 1.900e-05 (342 / 2000128) | **0.499 [0.439, 0.568]** |
| 0.15% | 1.262e-05 (318 / 2800128) | 5.119e-06 (129 / 2800128) | **0.406 [0.331, 0.498]** |

For comparison, the qec-r4 LNS schedule (N_min 492 → 255 at the same d_circ = 7) gave 1.02× [0.94, 1.11] at p = 0.3%: a flat ratio. Here it falls 0.58 → 0.50 → 0.41, the signature of the distance gain.

## 5. d = 11 and beyond

| space (d = 11, D = 10, 1 round) | result | iterations / cuts / time | evidence |
|---|---|---|---|
| K–F: 6 + 6 layers, one schedule | **UNSAT** (K–F's 9 is optimal here) | 356 / 140,531 / 942 s | DRAT VERIFIED |
| independent X/Z schedules, 6 layers each | **UNSAT** | 376 / 139,327 / 737 s | D11SEPV |
| **7 + 7 layers, one schedule, collision-free** | **FOUND** | 387 / 146,980 / 1,199 s | `schedules/d11_T7_D10.sched` |
| any depth, independent orders | FOUND | 158 / 95,854 / 835 s | `runs/d11_free_found.json` |

Checks of the 7-layer d = 11 schedule (14 CNOT layers per round against K–F's 12):

| check | K–F | 7-layer |
|---|---|---|
| Rust circuit DEM, Z and X memory, 1 round, certified | 9 | **10 / 10** (≈ 150 s each) |
| Stim upper bound, 1 round | 9 | **10** |
| Rust circuit DEM, Z memory, 3 rounds | 9 (formula) | D11R3 |

So the boundary deficit is not monotone in the design space. At d = 9 K–F's 6 + 6-layer space has slack (8 is reachable). At d = 11 it does not, and one extra layer per half restores d − 1.

**d = 13**: D13STATUS

## 6. Relaxations: what unlocks more distance

| relaxation | d = 5 | d = 7 | d = 9 | d = 11 | evidence |
|---|---|---|---|---|---|
| none (K–F space) | 4 = d − 1 | 6 = d − 1 | **8 = d − 1** (K–F: 7) | 9 (K–F, optimal) | §2–§5 |
| 7 + 7 layers | ≤ d − 1 | ≤ d − 1 | ≤ d − 1 (already reached at 6) | **10 = d − 1** | corner lemma; §5 |
| any depth, decoupled X/Z | ≤ d − 1 | ≤ d − 1 | ≤ d − 1 | ≥ 10 at 1 round | corner lemma; §5 |
| corners hook-free (3 flags) | — | ≤ 6 (UNSAT) | — | — | `d7_D7_hf3corners` |
| two auxiliaries (cat split) on every plaquette | ≤ 4 | ≤ 6 | ≤ 8 | — | `*_split_all` |
| ≤ 3d − 7 hook-free plaquettes, any orders and depth | ≤ 4 | ≤ 6 | ≤ 8 | — | `*_hf8/14/20` |
| all 3d − 6 boundary plaquettes hook-free, 6 + 6 layers | **5** (5 rounds) | **7** (7 rounds) | ≤ 8 (UNSAT; also with decoupled X/Z) | — | `schedules/d5,d7_hookfree_boundary_T6`; `d9_D9_hfbnd_*` |
| same with 7 + 7 layers | | | **9** (1, 3 and 9 rounds) | — | `schedules/d9_hookfree_boundary_T7.found.json` |

The cheapest relaxations:
- **To d − 1**, free at d ≤ 9 (K–F's space suffices); **+1 CNOT layer per half at d = 11**.
- **To d**: fully hook-free measurement of all 3d − 6 boundary-touching plaquettes (minimal in count), plus a 7th layer per half from d = 9. More layers, decoupled schedules or cat-pair auxiliaries alone never get past d − 1.

## 7. Caveats

- Noisy-CNOT model (the uniform depolarizing model was spot-checked at d = 9). Only Z memory was optimised; X memory was checked for the d = 9 schedule (8 over 9 rounds) and the d = 11 schedule (10 over 1 round). SI1000 not run.
- The hook-free relaxation is idealised (flag CNOTs not in the timing, their faults assumed caught). The hook-free schedules (`*_hookfree_*`) were verified with the symbolic model only, because the Rust generator has no flag qubits.
- `free`/`sep` keep K–F's round structure: Z half, then X half, one sequential auxiliary per plaquette per half.
- The certificates assume the symbolic DEM is exact. That is checked exactly against the circuit DEM on 60 schedule instances (§1.1). Every FOUND single-schedule result was re-verified with the circuit-level Rust tool and Stim, not with the model.
- The d = 11 free-mode schedule was verified over 1 round only.
- LER uses our BP+OSD, about 2.5–3× weaker than Tesseract in absolute terms (qec-r4 §2.2), so the ratios are decoder-relative. No d = 11 LER was measured.
- N counts merged-DEM mechanism sets, unweighted.

## 8. Reproduce

```bash
export CARGO_TARGET_DIR=/tmp/cg-target CARGO_INCREMENTAL=0
cargo build --release --example color_search --example dem_distance --example color_ler
cd research/data/colour-global                  # python needs python-sat; Stim checks need stim
python cg_model.py 5 && python test_presence.py 4 # symbolic DEM == circuit DEM
python corner_bound.py 3,5,7,9,11,13,15          # corner lemma
python cg_sat.py 9 8 1 kf --warm --sym           # FOUND: d_circ = 8 at d = 9 in K-F's space
python cg_sat.py 11 10 1 kf --warm --sym         # UNSAT (~15 min)
python cg_sat.py 11 10 1 kf --T 7 --warm --sym   # FOUND with 7 + 7 layers (~20 min)
./certify.sh python drat-trim && ./verify_certs.sh drat-trim
$CARGO_TARGET_DIR/release/examples/color_search distance 9 9 schedules/d9_global_D8.sched 1   # 8, certified
$CARGO_TARGET_DIR/release/examples/color_search export 9 3 cnot 0.001 schedules/d9_global_D8.sched /tmp/x.stim && python stim_check.py /tmp/x.stim 4 6
cargo test --release --test colour_global       # + -- --ignored for the d = 11 distance
```

`src/qec/color.rs` now accepts steps up to `MAX_STEP = 12` (layers per half = largest step used, at least 6). K–F circuits are unchanged.
