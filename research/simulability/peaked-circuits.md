# Cracking the 98-qubit peaked circuits P11 and P12 classically

Branch `exp/peaked-crack`, 5 Oct 2026, Xeon Gold 6548Y+ VM (16 vCPU, shared). Code and log:
[`research/data/peaked-circuits/`](../data/peaked-circuits/) (`solve_peaked.py`, `run.log`).

## Headline

The two 98-qubit heuristic peaked circuits of the
[Quantum Advantage Tracker](https://quantum-advantage-tracker.github.io/),
`peaked_circuit_P11_Hqap_98x1999` and `peaked_circuit_P12_Hqap_98x2457` (BlueQubit's Peak Portal
problems 11 and 12, listed with classical runtime "unsolved"), are solved **classically and blind**
in **89 s and 182 s** of wall time in one Python process on one shared VM. The recovered 98-bit peaks
equal, bit for bit, the strings that Quantinuum Helios-1 runs recovered and the BlueQubit portal
accepted (tracker issues #246 and #247; the quantum submission #179 quotes 7,200 s of quantum
runtime for P11). The target bitstrings are not used by the solver; they are only compared at the end.

## What the circuits are, and the weakness

The construction ([arXiv:2510.25838](https://arxiv.org/abs/2510.25838)) trains a shallow peaked circuit
R ▷ P, inserts an identity U ▷ U† between R and P, and obfuscates with wire swaps, re-trained
("swept") parameters and re-synthesised ("masked") patches: T[R] ▷ T[U] ▷ U† ▷ P. In P11 and P12 the
identity is nested: an outer U ▷ U† whose two halves each contain an inner identity block.

1. **Fingerprints.** Every 2-qubit gate is the atomic unit `u a; u b; cz a,b; u a; u b`. The
   single-qubit rotation between consecutive CZs on a wire is an SU(2) element with continuous
   angles. Unswept gates appear in U† as exact inverses of their U partners, so unique exact inverse
   pairs ("anchors": 419 in P11, 569 in P12) mark mirror centres. There are two, one per inner block.
2. **Absorbed swaps.** Inside each inner block the map "wire of a U gate → wire of its U† partner" is a
   single, consistent **involution with no fixed points** (49 transpositions; 95-98 of 98 wires
   anchored, zero conflicts). The CZs adjacent to anchors obey the mirror rule under this map (203 of
   211 in P11), and ~50 qubit pairs carry exactly three CZs each, i.e. a SWAP network sits in the
   seam. So each inner block equals the wire permutation f.
3. **The outer identity.** The serialised circuit splits into generation sections (each ends in a
   tail of tiny layers). With the inner blocks replaced by f, the outer U† half mirrors the outer
   T[U] half **with the identity map** (294 of 318 gate pairs in P11, 397 of 450 in P12; the rest are
   masked patches).
4. **Reduction.** The whole middle (1,716 of 1,999 gates in P11) is a permutation, which in a
   tensor network is free re-wiring. What remains is R (first section, 59/60 gates: one disjoint
   layer plus a small masked patch) and P (last section, ~110 two-CZ gates). Its gate-network
   treewidth is ~15, and quimb/cotengra contract every single-qubit marginal exactly in ~4-5 s.
5. **Boundary repair.** Obfuscation patches can straddle the R / T[U] boundary, which leaves a few
   marginals weak (P12: 9 wires with |<Z>| < 0.6, peak probability 0.005). The solver greedily moves
   frontier gates across the boundaries and keeps a move only if the exact peak probability of the
   core grows by at least 30 %. One or two gates suffice (peak probability 0.30 for P11, 0.20 for
   P12).

## Results

| instance | qubits × CZ | blind runtime | core peak probability | peak = Helios-1 / portal-accepted |
|---|---|---|---|---|
| P11 `98x1999` | 98 × 1999 | 88.6 s | 0.303 | yes, 98/98 bits |
| P12 `98x2457` | 98 × 2457 | 181.9 s | 0.205 | yes, 98/98 bits |

Full output: [`run.log`](../data/peaked-circuits/run.log).

## Caveats

* This exploits the published construction (identity insertion), like IBM's "MPO + unswapping"
  attack on the 56-qubit P9 ([arXiv:2604.21908](https://arxiv.org/abs/2604.21908)); it says nothing
  about peaked circuits built differently.
* The peak probabilities are those of the de-obfuscated core R ▷ Π ▷ P, not of the obfuscated
  98-qubit circuit (whose swept and masked patches are lossy by design and which is not exactly
  simulable here). The external check is the match with the hardware peaks.
* Development history, stated plainly: the boundary-repair step was added after a first P12 run
  differed from the hardware peak in 4 bits. The final solver's objective is only the exact peak
  probability of the core; it never reads the target.
* Timings are wall time on a shared VM (1-min load 6-13), single run, Python; no optimisation.
* Not tested here: the 56-qubit P9 (RZZ-based, already solved classically by others) and the
  heavy-hex instances.

## Reproduction

One command, from a clean checkout (Python >= 3.10; pinned `requirements.txt`; downloads the two
circuits at a pinned tracker commit and checks their sha256; about 5 minutes on one CPU):

```sh
research/data/peaked-circuits/reproduce.sh
```

It prints both peaks and confirms they match the Helios-1 results. Verified from scratch in a fresh
virtual environment on 5 Oct 2026 (98.5 s and 166.2 s).

Manual route:

```sh
pip install numpy scipy quimb cotengra
git clone --depth 1 https://github.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io qat
python3 research/data/peaked-circuits/solve_peaked.py \
  qat/data/classically-verifiable-problems/circuit-models/peaked_circuit/peaked_circuit_P11_Hqap_98x1999.qasm \
  qat/data/classically-verifiable-problems/circuit-models/peaked_circuit/peaked_circuit_P12_Hqap_98x2457.qasm
```
