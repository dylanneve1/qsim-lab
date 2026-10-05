# autoimprove ledger

Raw data for [research/performance/autoimprove.md](../../performance/autoimprove.md).

| file | contents |
|---|---|
| `ledger.jsonl(.xz)` | one JSON object per evaluation (candidates, A/A calibrations, knob sweeps, the multi-thread confirmation, CI checks): the full diff, build and test results, every timing (per process: all in-process timings, CPU and wall), load averages, plan shape (stages, block passes), the verdict |
| `ledger-v0.jsonl` | the A/A runs of the two earlier timing protocols (4-thread wall clock), which motivated the single-thread CPU-time protocol |

`tools/datafiles.py unpack` restores `ledger.jsonl`. Then
`tools/autoimprove/report.py research/data/autoimprove/ledger.jsonl --cases combo6@c0403bb`
prints the notebook's tables.
