# P11 / P12: v1 (section repair) vs v2 (max-peak)

Both solvers stay in the tree.  `solve_peaked.py` (v1) is unchanged and is what the 5 Oct tracker
comments describe.  `solve_peaked_v2.py` replaces only the boundary choice.

| | v1 `solve_peaked.py` | v2 `solve_peaked_v2.py` |
|---|---|---|
| structure (anchors, sections, wire maps) | same | same (strict involution check) |
| boundary choice | weak-wire filter `|<Z>|<0.6`, accept if p grows >= 1.3x | every frontier gate, take max exact peak p, stop when none improves |
| tuned constants | 0.6, 1.3 | none |
| determinism | iterates `set(cands)`: path depends on PYTHONHASHSEED (P12: gates 60+108 or 60+64) | fixed order (p desc, side, index) |
| target string used at runtime | no | no |

## Results (VPS, quimb 1.11.2 / cotengra 0.8.2)

| circuit | v1 core p | v2 core p | v2 gates added | bits off vs Helios peak | v2 time |
|---|---|---|---|---|---|
| P11 | 0.3029 | 0.3029 | R59 | 0 | 306 s |
| P12 | 0.2047 (seed 0) | 0.2366 | R60, R64 | 0 | 356 s |

Same peak under both methods; v2's repair is the threshold-free one and is order-stable.

## What v2 does NOT claim
* "Exact" applies to the reduced core only.  Sweeping and masking in these circuits are lossy
  by construction (arXiv 2510.25838 s3.1), and an exact-cancellation rewrite of the middle finds
  nothing: 0/1999 (P11) and 0/2457 (P12) CZs cancel, exact inverse pairs exist only inside the
  two inner blocks.
* P12's first step is a thin call: R60 0.0542 vs R77 0.0409 (1.32x).

## Branch audit (can the objective be fooled by a wrong first step?)
Force a different first gate, then let v2 finish greedily (`--start side:k`):

| circuit | forced start | final core p | peak |
|---|---|---|---|
| P11 | (none, v2) | 0.3029 | correct |
| P11 | R63 | 0.2877 | correct |
| P11 | R73 | 0.2503 | correct |
| P12 | (none, v2) | 0.2366 | correct |
| P12 | R77 | 0.0882 | wrong (several bits) |
| P12 | P2153 | 0.1021 | wrong (several bits) |

On P12 the wrong first steps end 2.3-2.7x lower, so ranking complete paths by exact peak
probability picks the right one even though the first step is close.  On P11 every branch lands
on the same peak.  Bit comparison to the hardware peak is post hoc only.

Raw logs/JSON: `v2-runs/`.
