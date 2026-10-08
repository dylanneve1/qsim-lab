# Doped-Clifford production results (tracker issue 228)

- `out_final74.jsonl`: all run records (4 calibration batches c555, c1784, c1180, c495 + 74 production samples s0–s73), sha256 c221f294295a3bb3c31476a78520c3746db7b8432791eb89135e6932051fb7f6.
- `prod-seed-REVEALED.txt`: production seed. `python3 ../runplan/analyze.py commit --seed-file prod-seed-REVEALED.txt` prints b9d2c1262c305229e5c2a12a1e0c4cad1eb2f933226b767878547950b37a2cf2, the hash published before the first sample (2026-10-07). Prefixes: `analyze.py prefixes --seed-file prod-seed-REVEALED.txt --m 6 --n 74`.
- `s1-calseed.txt`, `s2-calseed.txt`: calibration seeds (rows via `analyze.py calrows --m 6 --k 2`).
- Scoring vs SUTD exact amplitudes (Zenodo 10.5281/zenodo.21912448): `analyze.py calib out_final74.jsonl`; predicted XEB: `analyze.py samples out_final74.jsonl --xhat 0.822 --xse 0.060`.
- Bitstrings q0-first (Qiskit little-endian = reversed). Full report: ../WRITEUP.md.
