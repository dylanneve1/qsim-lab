[bench.sh] load at start: 17.23
| circuit | method | active qubits | memory | state (s) | sampler setup (s) | shots (s) | shots/s |
|---|---|---|---|---|---|---|---|
| random n=50 t=8 | compressed | 6 | 0 MiB | 0.0014 | 0.0001 (rand 44 / dense 6 / det 0, 7 passes) | 0.0374 | 2.672e6 |
| random n=50 t=12 | compressed | 9 | 0 MiB | 0.0021 | 0.0001 (rand 41 / dense 9 / det 0, 10 passes) | 0.0393 | 2.546e6 |
| random n=50 t=16 | compressed | 14 | 0 MiB | 0.0038 | 0.0013 (rand 36 / dense 14 / det 0, 15 passes) | 0.0428 | 2.339e6 |
| random n=50 t=20 | compressed | 19 | 8 MiB | 0.0082 | 0.0619 (rand 31 / dense 19 / det 0, 20 passes) | 0.0633 | 1.581e6 |
| random n=50 t=22 | compressed | 20 | 16 MiB | 0.0243 | 0.1028 (rand 30 / dense 20 / det 0, 22 passes) | 0.0832 | 1.201e6 |
| random n=50 t=24 | compressed | 23 | 128 MiB | 0.0713 | 0.8069 (rand 27 / dense 23 / det 0, 25 passes) | 0.1625 | 6.152e5 |
| random n=50 t=25 | compressed | 23 | 128 MiB | 0.0906 | 0.8528 (rand 27 / dense 23 / det 0, 25 passes) | 0.1753 | 5.706e5 |
[bench.sh] load at start: 16.81
| random n=64 t=25 | compressed | 24 | 256 MiB | 0.1728 | 1.5175 (rand 40 / dense 24 / det 0, 25 passes) | 0.1879 | 5.321e5 |
