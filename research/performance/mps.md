# MPS Backend Optimization Lab Notebook

**Agent**: `agt_305f0c3d` (qsim-mps)  
**Branch**: `exp/mps`  
**Base commit**: `e86e91b`  
**Hardware**: 4 vCPU (AMD EPYC-Rome, AVX2, no AVX-512), 7.7 GB RAM (shared VM)  

---

## 1. Baseline Performance

Measured using `/mnt/HC_Volume_106989832/dylan/qsim-swarm/bench.sh ./target/release/qsim bench mps`.

### GHZ
| n | max bond | memory | time (s) |
|---|---|---|---|
| 100 | 2 | 12.4 KiB | 0.050 |
| 1000 | 2 | 124.9 KiB | 0.100 |
| 10000 | 2 | 1.2 MiB | 0.419 |

### Random Brickwork Circuits (n=24, χ_cap=256)
| n | depth | χ cap | max bond | fidelity est. | memory | time (s) |
|---|---|---|---|---|---|---|
| 24 | 2 | 256 | 2 | 1.000000 | 2.9 KiB | 0.001 |
| 24 | 4 | 256 | 4 | 1.000000 | 10.6 KiB | 0.005 |
| 24 | 8 | 256 | 16 | 1.000000 | 138.6 KiB | 0.015 |
| 24 | 12 | 256 | 64 | 1.000000 | 1.6 MiB | 0.308 |
| 24 | 16 | 256 | 241 | 1.000000 | 12.2 MiB | 4.691 |
| 24 | 20 | 256 | 256 | 0.999994 | 18.7 MiB | 28.540 |
| 24 | 24 | 256 | 256 | 0.997527 | 18.7 MiB | 47.323 |
| 24 | 32 | 256 | 256 | 0.909861 | 18.7 MiB | 90.616 |

### Random Brickwork Circuits (n=60, χ_cap=32)
| n | depth | χ cap | max bond | fidelity est. | memory | time (s) |
|---|---|---|---|---|---|---|
| 60 | 4 | 32 | 4 | 1.000000 | 28.6 KiB | 0.008 |
| 60 | 8 | 32 | 16 | 1.000000 | 426.6 KiB | 0.049 |
| 60 | 16 | 32 | 32 | 0.734902 | 1.6 MiB | 1.681 |
| 60 | 32 | 32 | 32 | 0.000088 | 1.6 MiB | 3.783 |
