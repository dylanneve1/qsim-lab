# Peaked circuits: open problems

The mirror-fold solvers here (`solve_peaked.py`, `solve_peaked_v2.py`, and the generic mirror-centre
solver) crack the Peak Portal / HQAP circuits we have tried, but only because those circuits share one
construction. These are the known limits, stated as open problems. Each comes with the evidence that
motivated it. Related prior work: Kremer & Dupuis, arXiv:2604.21908 (mirror cancellation into an MPO
plus greedy "unswapping").

## 1. Peaked circuits without a mirror
Every method here begins by finding a centre layer that the circuit folds around. A circuit peaked some
other way (no inverse-pair structure, or a reflection hidden by more than a wire permutation) gives the
solver nothing to fold, and it falls back to an ordinary contraction.
*Open:* a structure-free method (or a hardness argument) for peaked circuits with no reflection symmetry.

## 2. Bounding the loss from compression
"Exact" covers only the reduced core. Folding the window operator W is lossy, and when it loses norm
the peak becomes unreliable. P4 (48q, 5096 CZ units): norm 0.867 and min|<Z>| 0.158 at the
scan-selected centre.
*Open:* a rigorous bound linking lost norm and marginal margins to how reliable the peak is, so a
solve comes with a certificate rather than a heuristic threshold.

## 3. Greedy boundary choice can be misled
v2 picks each boundary gate by maximum exact peak. On P12 the first step is a narrow call (R60 0.0542 vs
R77 0.0409, 1.32x). Forcing R77 or P2153 first converges to a wrong peak (see `V2_MAXP.md`, branch
audit).
*Open:* a non-greedy or backtracking search with a stopping criterion that rules out a wrong basin,
or a proof that the max-peak objective is unimodal for this construction.

## 4. Scaling of the folded operator
P9 (56q, HQAP 1917) has defeated several centre choices. P6 (62q, ~10.5k gates) is untested at the
time of writing. Memory is set by W's bond dimension, and nothing predicts in advance whether it
stays small.
*Open:* predict W's maximum bond dimension from the circuit before running, and find which features
of the construction control it.

## 5. Certifying the global peak
`parent_verify.py` shows that no single-bit flip of a candidate scores higher. That is a local
maximum test. It does not prove the candidate is the global argmax.
*Open:* a cheap global certificate, e.g. a bound on the total weight outside a Hamming ball, from
quantities the fold already computes.
