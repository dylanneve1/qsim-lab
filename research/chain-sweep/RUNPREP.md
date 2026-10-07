# Run preparation: tail-open batches, run loop, rehearsal, D = 70 calibration

Branch `exp/runprep` (on `exp/packed-lowprec`). Protocol: the doped-Clifford RUNPLAN (§2.1 tail-open
pass, §6 run-book). Reference amplitudes: SUTD, Zenodo 10.5281/zenodo.21912448 (2051 IBM rows × 256 exact
amplitudes over q0..q7, q8..q69 fixed to the row).

## What was built

- `engines::chain_tail`
  - `TailPlan::new(cc, x, m)`: sweeps the **mirrored** chain over q69..q_m and stops at the clean cut on
    edge (m−1, m). The register then holds R(β), β = the values of q_m at the K CZs of that edge.
  - Tail pass: amp(x_tail) = scale · ⟨x_tail| Σ_β R(β) T(β), with T(β) the m-qubit tail circuit where
    each CZ to q_m becomes Z^β on q_(m−1). Everything after the sweep is f64, so the pass adds no
    rounding: R counts the sweep's passes only.
  - Output index j = int(bits q0..q(m−1)), **q0 the most significant bit** (SUTD's rule).
  - Two forms:
    - depth-first (`tail_amplitudes_dfs`): any register layout, ~2^K·|seg|·2^m;
    - GEMM (`tail_amplitudes_gemm`, used automatically): on the mirrored D = 70 chain the K = 35 open
      bonds sit on register bits 0..34 in time order. Hence the early bonds form 2^h contiguous entries:
      each register row (late bonds fixed) times the 2^m × 2^h matrix of tail vectors gives the tail
      state, and a binary-counter tree sums over the late bonds. Cost: 2^K·2^m complex MACs. The register
      is read once, in order. At D = 70, m = 8 that is 8.8e12 cMACs instead of 4.3e14 for the DFS. The
      result is bit-reproducible (fixed reduction order).
  - `SweepBackend` trait (`describe`, `sweep(plan, on_pass)`, `amp(i)`, `count_passes`), with `CpuExact<f64|f32>` and
    `CpuPacked` (allocated once, `PackedStore::reset` between sweeps). The run loop and the tail pass use
    only the trait, so a GPU backend plugs in.
- `engines::chain_packed`: `PackedStore::get(i)` (random-access decode, bit-identical to the block
  decoder), `reset`, `decode_all`.
- `engines::chain_run`
  - SHA-256 and the seeded streams, bit for bit those of `runplan/analyze.py` (`prefix`, `tail`).
  - Job and record lines; the tail sampler's draw (smallest j with cumulative |l|² > u·Σ|l|²).
- `examples/chain_sweep.rs`:
  - `tailinfo`: width, open bonds, slots, R, tail cost, store size.
  - `tailcheck`: a tail batch vs the state vector / exact chain sweep, per backend.
  - `tailbench`: the tail pass alone.
  - `run`: the production loop.
    - One process, one allocation. Jobs come from `--jobs FILE` (analyze.py `prefixes` / `calrows` /
      `rowjobs`) or `--seed-file F --count N --m M`.
    - Per job it appends `{"kind":"start",…}`, sweeps, runs the tail pass, and appends the RUNPLAN §6
      record (`kind`, `i`|`row`, `tail_m`, `prefix_bits`, `amps` [[re,im]×2^m] in SUTD order, `format`,
      `R`, `t_sweep_s`, `t_tail_s`; samples add `u_tail`, `j_tail`, `bitstring_q0_first`;
      `seed_sha256`), with an fsync.
    - It rewrites `--heartbeat` after every pass (tmp + rename).
    - Resume is by key: any job without a result record is redone, in job order, never skipped.
    - RAM only: the register never touches disk.
    - On Windows it calls `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`.
- `runplan/analyze.py` (outside this repo) gained:
  - `rowjobs`: calibration jobs for chosen SUTD rows;
  - `compare`: batch fidelity of one record file against another, matched by key;
  - in `calib`: a bootstrap error over amplitudes (usable with 1–2 records) and convention diagnostics
    (F if conjugated / tail index bit-reversed / both).

## Exactness

| check | where | result |
|---|---|---|
| tail batch vs dense state vector, m = 1..8, n = 12–14 (doped truncations + random brickwork) | VPS test | < 1e-10·rms, every entry |
| tail batch vs exact n = 70 chain sweep per completion, D = 16/24/30, m = 8/6/8 | VPS test | < 1e-9 relative (f64) |
| f32 register | VPS test, win | ≤ 8e-6 relative |
| GEMM form vs DFS form, every split h | VPS test | < 1e-12 relative |
| packed: `get` vs block decoder; tail over the store vs over its decoded copy | VPS test, 7 formats | bit-identical |
| `tailcheck` n = 22 D = 40 m = 8 vs state vector; n = 70 D = 40 m = 6 vs chain sweep | win (Windows build) | f64 1e-14, f32 2e-6 |
| packed fidelity at the production config (l = 22, slots = 14, R = 63, m = 8), D = 50, vs exact f64, 512 amplitudes | VPS | int4:b64 0.507, int4:b16:h 0.588, int6:b64 0.958 (pred. 0.47 / 0.57 / 0.962) |
| int6:b64 vs int8:b64 at D = 56, m = 6, R = 65, 192 amplitudes | win | 0.957 (pred. 0.960) |

Plan facts at D = 70 (`tailinfo`, l = 22, slots = 14):
- width 35, K = 35 open bonds on bits 0..34;
- **R = 63 at m = 8, 65 at m = 6** (71 for the full chain);
- store 34 GiB (int4:b64), 36 GiB (int4:b16:h), 50 GiB (int6:b64).

## Rehearsal (win, D = 56, int6:b64, m = 6, rehearsal seed sha256 5d414e8c…)

- 10 sweeps at 20.8 s each (20.4 s of passes, 0.3 s tail), R = 65; peak working set 1.4 GiB.
- No paging (monitor: pages-in ≈ 0, pagefile flat).
- Forced kill during s4 (pass 37/65). The relaunch found 4 done, redid s4, and finished s4..s9. The file
  holds two `start` lines for s4 and one result each for s0..s9.
- Prefixes derived on Windows from the seed are identical to `analyze.py prefixes`.
- `analyze.py samples` reports no missing indices.

## D = 70 calibration (go/no-go)

See the section appended after the run.
