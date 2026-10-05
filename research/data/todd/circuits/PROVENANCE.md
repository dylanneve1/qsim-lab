# Benchmark circuits (provenance)

The `.qc` files in this folder are the standard Clifford+T T-count benchmarks
(Amy, Maslov & Mosca 2014; used by Feynman, PyZX, TODD and AlphaTensor-Quantum),
copied unmodified from Feynman (`github.com/meamy/feynman`, directory
`benchmarks/qc`) at commit `90cb0c807321fb356587c7be623d1d5607666c2d`
(2024-11-14), fetched 5 Oct 2026. Feynman is distributed under the BSD-3-Clause
licence reproduced in [FEYNMAN_LICENSE.md](FEYNMAN_LICENSE.md); some files carry
their original authors' headers, which are kept.

Four circuits of the suite are **not** committed because their headers forbid
redistribution: `hwb6`, `hwb8`, `gf2^16_mult` and `gf2^32_mult`. Run
`../fetch_circuits.sh` to download them from the same commit; it checks the
sha256 sums in `../fetched.sha256` (and those of the committed files in
`../circuits.sha256`).
