# research/data/avx512: baseline comparison data and scripts

Data and drivers for `research/performance/avx512.md` (qsim-lab against qsimcirq, qulacs,
qiskit-aer and PennyLane lightning.qubit on the Xeon Gold 6548Y+, identical circuits).

## Scripts

| file | what it does |
|---|---|
| `gen_circuits.py` | deterministic generator of the five workloads (`qft`, `brick_cz`, `brick_su4`, `qv`, `qaoa`) in one gate-list text format (documented in its docstring); SU(4) workloads are written as an exact KAK decomposition (`<wl>_<n>.txt`, u3 + cx) and as dense 4x4 gates (`<wl>_<n>.dense.txt`); `--selftest` checks the Sycamore 1q gates, the KAK decompositions (1e-12 up to phase) and dense-vs-decomposed equivalence; also contains the numpy reference simulator used by `validate.py` |
| `baselines.py` | runs one framework on one circuit file and prints one JSON line per repetition; the timer semantics per framework are in its docstring; `baselines.py info` prints the SIMD module / kernels each framework dispatches to |
| `validate.py` | cross-validation: every framework's final state (same bit layout) against the numpy reference |
| `bench_matrix.py` | interleaved timing driver (ABBA rounds of fresh processes, `taskset` pinning, holds of the global bench lock of at most ~7 min, load gating at load1 <= 16, `uptime`/`free -g` logged per hold in `holds.log`) |
| `plans.py` | writes the cell lists (JSON) for the knob sweep and the timing matrix |
| `pick_best.py` | picks each framework's fastest knob setting per (workload, precision) from the sweep |
| `summarize.py` | best time per framework per cell, best baseline, qsim-lab ratio, load flags |

## Raw data

| file | content |
|---|---|
| `validation_n12.csv`, `validation_n14.csv` | fidelity and phase-aligned max amplitude error of every framework / precision / workload vs the numpy reference |
| `sweep_n24_raw.csv` | per-framework knob sweep at n = 24, 8 threads (one row per timing) |
| `baselines_raw.csv` | the timing matrix (one row per timing; columns documented in `bench_matrix.py`) |
| `holds.log` | `uptime` and `free -g` before and after every hold of the bench lock |

## Rerun

```sh
. /dev/shm/qsim/venv/bin/activate
python gen_circuits.py --selftest
python gen_circuits.py /dev/shm/qsim/avx512/circuits all 12,14,24,26,28
python validate.py /dev/shm/qsim/avx512/circuits 12 validation_n12.csv
python plans.py sweep /dev/shm/qsim/avx512/circuits plan_sweep.json
python bench_matrix.py plan_sweep.json sweep_n24_raw.csv --reps 3 --rounds 1
python pick_best.py sweep_n24_raw.csv best.json --report sweep_summary.md
python plans.py matrix /dev/shm/qsim/avx512/circuits best.json plan.json 24,26 both 8
python bench_matrix.py plan.json baselines_raw.csv --reps 2 --rounds 2
python summarize.py baselines_raw.csv --md baselines_summary.md
```

qsim-lab is run through `examples/sv_file_bench.rs` (`QSIMLAB_BIN` points at the binary;
the "main" numbers use a binary built from main 6b21728).
