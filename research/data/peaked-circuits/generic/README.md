# Generic peaked-circuit solver

Research code for classically extracting the peak of "Heuristic Quantum Advantage" peaked circuits
(arXiv 2510.25838): `T[R] ▷ T[U] ▷ U† ▷ P`, where R ▷ P is a shallow trained peaked circuit and the
middle is an obfuscated identity (swaps, sweeping, masking). The solver never reads a target string.
Reference peaks were compared only after solving (`compare.py`).

> `private/` holds answer strings of Peak Portal problems (P2–P10). **Never commit it.**
> Results below give probabilities and evidence only, never the answer strings, for those problems.

## Pipeline (`solve.py` dispatches)

1. **Front end (`gparse.py`)**: QASM 2 and 3; `qreg` / `qubit[n]`, multiple registers, user `gate` macros
   (e.g. `iswap`). 1-qubit gates: u, u3, U, u2, u1, p, rz, rx, ry, x, y, z, h, s, sdg, t, tdg, sx, sxdg, id.
   2-qubit gates: cz, cx/cnot, cy, swap (as 3 cx), rzz / cp / crz (any diagonal gate with interaction phase π);
   non-CZ-class gates are rejected. Every 2-qubit gate becomes exactly one CZ plus local gates.
   Two foldings of the 1-qubit gates are provided. `parse`: everything pending is folded into the next unit.
   `parse_nearest`: each gate is attached to the nearest 2-qubit gate in file order; for P11/P12 this reproduces
   the strict 5-line units bit for bit. `test_gparse.py` checks state fidelity = 1 (to 1e-12) on synthetic
   circuits with every gate type (QASM 2 and 3), on the iSWAP macro, and on random ≤11-qubit sub-circuits of
   every test file.
2. **Anchor route (`solve_anchor.py`)**: used when exact inverse segment pairs ("anchors") exist (P11, P12).
   - Blocks: sort anchors by mirror centre; a new block starts exactly when an anchor contradicts the current
     involution (`blocks.py`).
   - Generation sections: serial-layer size jumps split by Otsu's two-class rule (`sections2.py`); only the
     first section (R) and the last section (P) are used.
   - Core: R ▷ π ▷ P, with π = the composition of the block involutions; it is contracted exactly
     (quimb/cotengra).
   - Boundary: greedy over the full DAG frontier, adding the gate with the maximum exact peak probability,
     until nothing improves.
3. **Operator route (`solve_generic.py`, `tnoq.py`)**: used when there are no anchors (every segment re-trained).
   - Mirror centre: for each candidate cut (coarse grid, then refined) the window operator W is grown
     layer-synchronously with canonised arbitrary-geometry compression (`tensor_network_ag_compress`). The
     score is the number of gates absorbed before W exceeds a memory budget; the centre is the argmax.
   - Grow W from the centre over the whole circuit, then contract `P ▷ W ▷ R |0⟩` exactly. The peak is the
     sign of the exact marginals; the probability is |amplitude|² / norm.
   - The middle-out cancellation idea is the one used by Kremer and Dupuis (arXiv 2604.21908; tracker #47,
     #105). This is an independent implementation: data-driven centre, our units, exact final contraction.
   - `tno.py` / `tnos.py` are an experimental swap-aware variant for relabelled mirrors. Each tensor carries
     one input leg and one output leg of *any* wires, so a permutation is free. Gate splits try both leg
     assignments, and `unswap_pass` exchanges legs of bonded pairs when that lowers the rank (an
     arbitrary-geometry analogue of "unswapping"). It is tested exactly in `test_tno.py`, but it does not yet
     crack P9.
   - `mirror.py` / `pauli.py` hold a Pauli-image (Heisenberg) mirror detector, used for diagnosis only.

## Reproduce

```
PY=/tmp/pk/research/data/peaked-circuits/.venv/bin/python   # quimb 1.11.2, cotengra 0.8.2, kahypar
prlimit --as=2700000000 $PY test_gparse.py; $PY test_tno.py
prlimit --as=2700000000 $PY solve.py FILE.qasm --json private/out.json
prlimit --as=2700000000 $PY robust.py FILE.qasm TAG c1,c2 1e-3,1e-4,1e-5      # robustness sweep
$PY compare.py private/out.json            # post hoc only
```

## Results (this machine: 4 vCPU, 7.7 GB shared; every run capped at 2.5 GB)

See `NOTES.md` for the running log and `logs_*.log` for raw output. Answer strings for portal problems are only in `private/`.

| circuit | parsed | route / structure found | peak vs reference (post hoc) | core p | time | peak RSS |
|---|---|---|---|---|---|---|
| P11 98q/1999 (tracker; = portal P11) | yes | anchor: 419 anchors → 2 involution blocks, Otsu sections = old sections, +R59 | identical to our earlier P11 peak (hardware-confirmed) | 0.3029 | 279 s | 284 MB |
| P12 98q/2457 (tracker; = portal P12) | yes | anchor: 569 anchors → 2 blocks, +R60 +R64 | identical to our earlier P12 peak | 0.2366 | 336 s | 305 MB |
| heavy_hex 49q/5072 (tracker) | yes | operator: centre 140 (scan 143:5072 vs ~180), whole circuit absorbed, W bond ≤ 8 | **Hamming 0/49** vs tracker #105 / KD notebook / hardware top count (Qiskit order) | 0.6973 | 91 s | 207 MB |
| heavy_hex 49q/4020 (tracker; = portal P10) | yes | operator: centre 112 (scan 108:4020 vs ~180), whole circuit absorbed | no public reference; portal-graded correct (per parent) | 0.695–0.702 over 2 centres × cutoffs 1e-3/1e-4 | 92 s | 212 MB |
| portal P8 40q grid / 888 iSWAP | yes (gate macros) | operator: centre 59 (1649 vs ≤160), whole circuit absorbed; exact final contraction too wide (32.5) → candidate from bond-capped W\|0⟩ (χ=4,2,1 agree) | portal-graded correct (per parent) | 0.0841 (unnormalised \|⟨s\|W0⟩\|², uncapped state; all 1-flips ≤ 0.0013) | 1226 s | 739 MB |
| P9 56q/1917 rzz (tracker; = portal P9) | yes | **not solved**: 0 unique anchors; centre ≈ layer 50 holds a swap network (layers 46–55); unswapping passes the band but W regrows outside it | — | — | — | ≤ 1.1 GB per attempt |

Scaling and bottlenecks (`logs_scaling_p11p12.log`, solver logs):
- Anchor route: the final P11/P12 cores are tiny (amplitude width 12 / 8, log2 flops 17.9 / 15.4; the worst
  marginal has width 11, log2 flops ~17). The 5 minutes are spent on the number of evaluations, not on flops:
  ~95 frontier candidates × 2–3 steps × up to 98 marginals (cached by light cone), each a small quimb contraction
  with Python/path overhead.
- Operator route on heavy-hex: W growth is ~0.025 s per layer pair once the centre is known (whole 4020 circuit in
  ~3 s, bond ≤ 8). The centre scan dominates (67–74 s of 92 s). Final contraction width 12–16, log2 flops 19–23.
- Bottleneck 1 (P8): when the peaking layer P is deep on a 2D lattice (~16 grid layers), the exact final contraction
  of W|0⟩ is width 32.5 / log2 flops 41.6 (norm). That needs either bond-capped approximation (used) or sliced
  contraction beyond this machine.
- Bottleneck 2 (P9): relabelling mirrors with an explicit central swap network. Arbitrary-geometry unswapping
  removes the band; afterwards the operator still grows ×1.3–1.6 per layer (53k elements at [37,57)), so memory and
  time are exponential in the uncancelled depth.

Repository hygiene: commit the code, `README.md`, `NOTES.md`, `logs_*.log` and `results/` (public circuits only).
Do **not** commit `private/` (portal answers, including the parent's P4/P7 logs moved there) or `refs/` (third-party
clones and reference strings used only for post-hoc comparison).
