# exp/superopt data (Mac M1 Pro)
- `counts_final.txt`: whole-run gate/CCX counts, ablations (incl. global vs block passes), 20/24/28/31-bit records (`examples/superopt_counts N a 4`).
- `counts_31.txt`, `counts_20_28.txt`: earlier ablation runs (global passes default; 20–28 with the generic-peephole column).
- `bench.log` (opt = ALL with window DP and no SAT rules; my own builds overlapped rep 1 of the 31-bit run, so treat rep 1 as contaminated), `bench2.log` (global passes), `bench3.log` (final `Opts::ALL`, block passes): interleaved timings under the bench lock.
- `sat1/2/3.jsonl` + `.err`: SAT certificates (`tools/superopt/blocks.py`); `sat3_partial.err` has the unfinished n = 3 comparator (k ≤ 8 refuted). `sat2.err` also has the Q=6/LMAX=12 peephole window log.
- `peep.jsonl/.err`, `rules1*.`, `rules2*`: SAT peephole runs (`tools/superopt/peep.py`); `rules_merged.txt` is the merged rule table before the CCX-non-increasing filter (`src/shor/superopt_rules.txt` is the filtered one).
