# mpou2: our MPO + unswapping solver for HQAP-style peaked circuits

This solver produced the blind P5 validation (22 s) and our P9 solve (3 s per run on an M1 Pro). Both match the graded
or independent answers, which are not stored here; see `../PORTAL.md`. Full log: `NOTES.md`.

- **mpou2.py**: 1D MPO with one output and one input leg per site, exact relabelling (layout / output / input legs),
  QR-reduced two-site updates (complex128), greedy and hot-bond unswapping. Tested in `test_mpou2.py`: operator error
  2e-14.
- **centre.py / tno_centre.py**: an *exact centre block*. The central swap-network band is contracted exactly (dense per
  group, or with the swap-aware TNO from `../generic/tno.py`) and converted to an MPO. That is what prevents the blow-up
  at the swap network.
- **Mirror-partner scheduling (run2.py `--sched pair`)**: after each absorbed gate, absorb its mirror partner on the
  other side, the gate acting on the same two MPO sites, as soon as it is available. On P9, W stays at bond ≤ 8 through
  all 1885 blocks and the final state has bond 1. Without the centre block, or with plain layer order, it blows up after
  80–200 blocks.
- **match2.py**: matching unswap (Pauli-transfer score plus assignment), with a confidence gate.
- **endgame.py**: exact marginals and amplitudes of the outer layers (quimb/cotengra, light-cone pruned).
- P9 command: `run2.py P9 --m 49 --band 45 55 --tno_band --tno_cut 1e-3 --mode rel --eps E --maxb 512 --tau 1e5
  --match_hi 0 --sched pair`. The same peak comes out for E in {1e-2, 3e-3, 1e-3, 3e-4}, p ≈ 0.100.
- `ringsim.py`: ring-block state solver tried on P6. It runs out of memory at multi-ring clusters; P6 was solved by
  `../p6-rings/`.

Ideas credited: Kremer–Dupuis arXiv 2604.21908 (midpoint MPO + unswapping), Sabre routing (optional).
