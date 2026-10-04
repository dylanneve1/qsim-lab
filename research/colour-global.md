# Colour-code syndrome schedules: global search and optimality certificates

Branch `exp/colour-global` (from `main` = dd26f4b). Code and data: `research/data/colour-global/`.
This follows `research/qec-r4.md` Part 2. Setting: triangular 6.6.6 colour code, Kishony–Fowler (K–F, arXiv:2603.28852) layout and round structure, Z-memory, noisy-CNOT model (`DEPOLARIZE2(p)` after every CNOT) unless stated.

## Headline

1. **K–F's circuit distance is not optimal at d = 9.**
   - A schedule inside K–F's *exact* design space reaches **d_circ = 8**, against K–F's d − ⌊(d+3)/6⌋ = 7. The design space is one auxiliary per plaquette, the same 6 time steps for the X and Z halves (6 + 6 CNOT layers), and a collision-free schedule.
   - It was found by an exact counterexample-guided SAT search.
   - Independent checks on the exported circuit:
     - the Rust circuit-DEM branch and bound gives 8, certified, in both Z and X memory over 9 rounds;
     - Stim's undetectable-logical search finds 8 for this schedule and 7 for K–F;
     - the uniform depolarizing model also gives 8.
   - Logical error per round at p = 0.3% is **0.58× K–F's** [0.54, 0.62], with the same BP+OSD decoder and 1.2 M shots per arm. The earlier N_min-halving local-search schedule gave no gain at this setting (1.02×).
2. **8 is optimal at d = 9, and K–F's d − 1 is optimal at d = 5 and 7, for every single-auxiliary schedule.** This holds for any CNOT order and any depth, with no collision constraint and with the X and Z halves scheduled independently.
   - It follows from a corner lemma: every possible hook of a corner plaquette, plus d − 2 single data errors, makes a logical. This was checked for d = 3–15.
   - It also follows from DRAT-verified UNSAT certificates (d = 5, 7, 9).
3. **Why the earlier local search could not raise d_circ.** Freezing the 9 interior plaquettes at K–F's schedule and freeing the 21 boundary-touching ones (exactly the earlier LNS space) makes d_circ = 8 impossible at d = 9 (DRAT-verified UNSAT). The interior plaquettes adjacent to the boundary have to change too.
4. **Cheapest relaxation that unlocks d_circ = d.** More CNOT layers, separate X/Z schedules, or any depth cannot help (corner lemma). Two auxiliaries per plaquette (a 2 + 2 or 3 + 3 cat split) cannot help either, even on every plaquette (UNSAT, d = 5, 7, 9). What works:
   - make every boundary-touching plaquette **fully hook-free** (flag-protected, or ≥ 3 auxiliaries on hexagons): that is 3d − 6 plaquettes, i.e. 9, 15 and 21 at d = 5, 7, 9;
   - one fewer is UNSAT at each d (DRAT), and the three corners alone are not enough;
   - with that, d is reached in K–F's 6 + 6 layers at d = 5 and 7. At d = 9 it additionally needs a **7th CNOT layer** per half: 6 layers is UNSAT even with independent X/Z schedules.
5. **d = 11**: (see §6; search status at the time of writing).

## 1. Method

### 1.1 Exact symbolic DEM (`cg_model.py`)

For Z-memory under noisy-CNOT noise, every Z-sector fault mechanism is one of three kinds.

- **Clean data error (layer l).** It flips the layer-l detectors of the plaquettes with odd overlap. It is either:
  - a single X (always present), or
  - a *hook*: X on an X-type auxiliary after its k-th CNOT spreads to the plaquette's data qubits *after* step k, i.e. a proper suffix S of that plaquette's X-half order. S and its complement have the same signature.
- **Partial data error** (Z half, round r). X on data q after its CNOT with plaquette p flips the layer-r detectors of the plaquettes q meets *later* and the layer-(r+1) detectors of those it met *so far* (a prefix E of q's Z-half meeting order). An X ⊗ X fault after q's first CNOT gives E = ∅ in round 0.
- **Time-like**: X on a Z-type auxiliary flips layers r and r+1 of that plaquette.

So the DEM depends on the schedule **only** through two kinds of order literal:
- `bx[p,a,b]`: plaquette p touches a before b in the X half;
- `bz[q,a,b]`: q meets plaquette a before b in the Z half.

Every potential mechanism has an exact presence condition, a conjunction of these literals. The model's signature set equals the Rust circuit DEM's Z sector (`color_search dem`) on 30 schedules (K–F plus random collision-free ones; d = 3, 5, 7, 9; 1–3 rounds): `python cg_model.py 5` prints `ALL EQUAL`.

### 1.2 Counterexample-guided SAT (`cg_sat.py`)

The query is "does a schedule with d_circ ≥ D exist?". Schedule spaces:

| mode | X-half order | Z-half meeting order | constraint |
|---|---|---|---|
| `kf` | one time table t_p(q) ∈ 1..T for both halves | same table | collision-free; T = 6 is K–F's space |
| `sep` | table t^X | independent table t^Z | each collision-free, steps 1..T |
| `free` | any total order per plaquette | any total order per qubit | none (any depth) |

The time tables use an order encoding; `bx`/`bz` are tied to them by ↔ clauses.

The loop:
1. One `y_sig` variable per DEM signature is forced true by each presence condition.
2. The SAT solver (CaDiCaL 1.9.5) proposes a schedule.
3. `dem_distance` (the exact branch and bound of `src/qec/distance.rs`, new CLI `examples/dem_distance.rs`) lists every logical of weight < D in its DEM.
4. Each listed logical L adds the clause ∨_{sig∈L, conditional} ¬y_sig. This is sound: it removes exactly the schedules that contain L.
5. With `--sym`, the 120° and 240° rotations of each logical are added too. Each image is asserted to be a logical; that held every time.

The loop ends either with a schedule that has no logical of weight < D (**FOUND**, then verified independently), or with **UNSAT**. On UNSAT the final CNF (encoding + cuts) is re-solved by Glucose with a DRAT proof and checked by `drat-trim`.

- Rounds = 1 is used in the loop. Any 1-round logical maps onto the last round of a longer circuit, so 1-round UNSAT is UNSAT for every number of rounds, and FOUND schedules are re-verified over d rounds.
- `--space` (one layer, singles + hooks only) is a further relaxation, so its UNSATs are stronger statements.

### 1.3 Relaxation models

- **`--hookfree K`**: up to K plaquettes (chosen by the solver) have no multi-qubit hook. This models a flag-protected plaquette, or ≥ 3 auxiliaries on a hexagon, with flag/cat faults assumed caught. Flag CNOTs are *not* added to the timing, which is optimistic.
- **`--split K`**: two auxiliaries (a cat pair), each measuring half the plaquette. 2 + 2 on a weight-4 plaquette is hook-free. 3 + 3 on a hexagon leaves the last pair of each triple as a hook, and the solver chooses the split and orders.

## 2. Upper bound for every single-auxiliary schedule: the corner lemma

A weight-4 corner plaquette measured by one auxiliary with sequential CNOTs o1 o2 o3 o4 always has the hook X_{o3 o4}, which is X_{o1 o2} times the stabilizer. There are only three pairings {o1 o2 | o3 o4}.

`corner_bound.py` shows that for **each** pairing and each of the three corners, the minimum logical with single data errors plus that one hook has weight d − 1. Every one of those faults is a clean X-half error of a single round. So

  **d_circ ≤ d − 1 for every single-auxiliary schedule, whatever the order, depth, collision pattern or X/Z coupling, and for any number of rounds.**

Verified for d = 3, 5, 7, 9, 11, 13, 15 (`runs/corner_bound.txt`). The SAT certificates in §3 reproduce it independently. A deletion-minimal unsat core at d = 5 is just 3 logicals, one per corner pairing (`cg_core.py`).

Consequences:
- K–F (d − ⌊(d+3)/6⌋) is optimal at d = 3, 5, 7.
- From d = 9 on their formula sits below the bound (7 < 8 at d = 9, 9 < 10 at d = 11, 10 < 12 at d = 13). §4 shows the gap at d = 9 is real slack.

## 3. Certificates (all DRAT-verified)

`certify.sh` regenerates every CNF from scratch; `verify_certs.sh` runs `drat-trim`; results are in `certs/verified.txt`. Each line is: query → UNSAT means "no schedule in this space reaches D", for 1 round and hence for any number of rounds.

| certificate | space | D | CNF vars / clauses | cuts | verdict |
|---|---|---|---|---|---|
| `d5_D5_space_free` | any orders, space-only | 5 | 300 / 1,040 | 86 | VERIFIED |
| `d7_D7_space_free` | any orders, space-only | 7 | 738 / 3,403 | 967 | VERIFIED |
| `d9_D9_space_free` | any orders, space-only | 9 | 1,368 / 13,761 | 9,171 | VERIFIED |
| `d9_D8_kf_fixint` | K–F space, 9 interior plaquettes fixed to K–F, 1 round | 8 | 3,504 / 21,631 | 2,488 | VERIFIED |
| `d5_D5_hf8`, `d7_D7_hf14`, `d9_D9_hf20` | any orders, space-only, ≤ 3d − 7 hook-free plaquettes | d | 309 / 991; 812 / 3,197; 1,598 / 8,052 | 36; 659; 3,072 | VERIFIED |
| `d7_D7_hf3corners` | the 3 corners hook-free | 7 | 741 / 3,072 | 636 | VERIFIED |
| `d5/d7/d9_D=d_split_all` | **every** plaquette two-auxiliary (cat split) | d | 576 / 1,923; 1,557 / 5,728; 3,000 / 14,741 | 33; 484; 4,535 | VERIFIED |
| `d9_D9_hfbnd_kfT6` | all 21 boundary plaquettes hook-free, K–F 6-layer timing, 1 round | 9 | 3,525 / 19,995 | 951 | VERIFIED |
| `d9_D9_hfbnd_sepT6` | same, independent X/Z 6-layer schedules | 9 | 5,397 / 26,394 | 918 | VERIFIED |

What the certificates rely on:
- the order encoding of each space (straightforward, in `cg_sat.py`);
- the presence conditions (the symbolic DEM, §1.1, equal to the circuit DEM on 30 schedules);
- each cut being a logical (asserted when generated; the logicals are stored next to each CNF).

The DRAT check then makes the UNSAT itself independent of the SAT solver.

## 4. d = 9: a schedule with d_circ = 8 in K–F's space

`schedules/d9_global_D8.sched` (one line per plaquette, steps for positions a–f, 0 = absent; `d9_global_D8.found.json` has the orders). Found by `cg_sat.py 9 8 1 kf --warm`: 195 iterations, 16,876 cuts, 96 s single-core VPS. With `--sym` it takes 77 iterations.

| check | tool | K–F | new |
|---|---|---|---|
| collision-free, steps 1–6, deterministic detectors | `color_search collisions`, `tests/colour_global.rs` | yes | yes |
| Z-memory d_circ, 1 round | Rust circuit DEM + B&B (certified) | 7 (N = 36) | **8** (N = 10,119) |
| Z-memory d_circ, 3 rounds | same | 7 | **8** (19 s) |
| Z-memory d_circ, 9 rounds | same | 7 (N = 492) | **8** (275 s) |
| X-memory d_circ, 1 / 9 rounds | same | 7 | **8 / 8** (266 s) |
| uniform depolarizing, Z, 1 round | same | 7 | **8** |
| Stim `search_for_undetectable_logical_errors`, 1 round (upper bound) | Stim 1.16 on the exported circuit | 7 | 8 |
| Stim, 9 rounds | same | (pending) | (pending) |

The new schedule changes almost every plaquette, interior ones included. Certificate `d9_D8_kf_fixint` shows that the interior *must* change. The leading-order cost is a larger count of minimum-weight logicals at the higher weight, 10,119 at weight 8 against 36 at weight 7 (1 round). So the gain grows as p falls.

### 4.1 Logical error rate (d = 9, 9 rounds, noisy-CNOT, Z memory)

- Same BP+OSD-CS decoder on both arms (order 100, Z sector; `examples/color_ler.rs`), independent samples.
- Mac M1 Pro, 8 threads; each ≤ 150 s chunk held the bench lock; arms interleaved chunk by chunk (`ler_chunks.sh`, `cg_ler_d9.jsonl`).
- Ratio CI from the log-ratio normal approximation (`ler_summary.py`).

| p | K–F p_L/round (fails / shots) | new p_L/round (fails / shots) | ratio new / K–F [95% CI] |
|---|---|---|---|
LER_TABLE

For comparison, the qec-r4 LNS schedule (N_min 492 → 255 at the same d_circ = 7) gave 1.02× [0.94, 1.11] at p = 0.3%.

## 5. Relaxations: what unlocks d_circ = d

| relaxation | d = 5 | d = 7 | d = 9 | evidence |
|---|---|---|---|---|
| none (K–F space) | 4 = d − 1 (optimal) | 6 = d − 1 (optimal) | **8 = d − 1** (new; K–F 7) | §2, §4 |
| 7 + 7 layers, any depth, independent X/Z schedules | ≤ d − 1 | ≤ d − 1 | ≤ d − 1 | corner lemma; `*_space_free` certificates |
| corners hook-free (3 flags) | — | ≤ 6 (UNSAT) | — | `d7_D7_hf3corners` |
| two auxiliaries (cat split) on every plaquette | ≤ 4 (UNSAT) | ≤ 6 (UNSAT) | ≤ 8 (UNSAT) | `*_split_all` |
| ≤ 3d − 7 hook-free plaquettes, any orders and depth | ≤ 4 | ≤ 6 | ≤ 8 | `*_hf8/14/20` |
| all 3d − 6 boundary plaquettes hook-free, K–F 6 + 6 layers | **5** (5 rounds) | **7** (7 rounds) | ≤ 8 (UNSAT, also with independent X/Z schedules) | `hf_kf_d5/d7`; `d9_D9_hfbnd_kfT6/sepT6` |
| all boundary plaquettes hook-free, **7 + 7 layers** | | | **9** (1 and 3 rounds; 9 rounds: HFD9R9) | `schedules/d9_hookfree_boundary_T7.found.json` |

The cheapest relaxation that reaches the full distance is **fully hook-free measurement of every boundary-touching plaquette** (3d − 6 of them). At d = 9 a 7th CNOT layer per half is also needed. It is minimal in the count of hook-free plaquettes: 3d − 7 is UNSAT for any choice of which plaquettes, with any order and depth. Merely doubling the auxiliaries (cat pairs) is not enough anywhere.

## 6. Scaling: d = 11

D11_STATUS

## 7. Caveats

- Noisy-CNOT and uniform-depolarizing models only. The optimisation used Z memory; X memory was checked for the d = 9 schedule (8 over 9 rounds) but not optimised. SI1000 not run.
- The "hook-free" relaxation is idealised: flag or extra-auxiliary CNOTs are not added to the timing, and their own faults are assumed caught. The `split` model ignores cat-preparation faults, which only makes it optimistic, so its UNSATs stand.
- The `free`/`sep` spaces keep K–F's round structure: Z half, then X half, one sequential auxiliary per plaquette per half.
- The certificates assume the symbolic DEM is exact. It was validated against the circuit DEM on 30 schedules, and every FOUND schedule was re-verified with the circuit-level Rust tool (and Stim at d = 9), not with the model.
- LER uses our BP+OSD, which is about 2.5–3× weaker than Tesseract in absolute terms (qec-r4 §2.2), so the ratios are decoder-relative.
- N counts merged-DEM mechanism sets, unweighted.

## 8. Reproduce

```bash
export CARGO_TARGET_DIR=/tmp/cg-target
cargo build --release --example color_search --example dem_distance --example color_ler
cd research/data/colour-global              # needs python-sat (pysat) for the SAT parts
python cg_model.py 5                        # symbolic DEM == circuit DEM (30 schedules)
python corner_bound.py 3,5,7,9,11,13,15     # corner lemma
python cg_sat.py 9 8 1 kf --warm --sym      # finds a d_circ = 8 schedule at d = 9
python cg_sat.py 9 9 1 free --space         # UNSAT: 9 is impossible for any single-auxiliary schedule
./certify.sh python drat-trim && ./verify_certs.sh drat-trim
/tmp/cg-target/release/examples/color_search distance 9 9 schedules/d9_global_D8.sched 1   # 8, certified
python stim_check.py <exported.stim> 4 6     # Stim upper bound
cargo test --release --test colour_global
```
