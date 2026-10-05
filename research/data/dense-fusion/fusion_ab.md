## 2026-10-04T22:16:13Z threads=6 load={ 15.39 23.87 22.50 } args: brick 24 f32 3 20 off:dense=0 k2:dense=2 k3:dense=3
threads 6 (rayon); cycles column = wall x clock x threads / amplitude-updates of the fused op list
| workload | n | prec | config | min wall s | core-cyc/amp-op | max dAmp vs first | all wall s | tiling |
|---|---|---|---|---|---|---|---|---|
| brick | 24 | f32 | off | 0.9314 | 1.864 | 0.0e0 | 1.376 0.931 0.945 | passes 692 |
| brick | 24 | f32 | k2 | 1.0650 | 2.131 | 1.4e-9 | 1.206 1.081 1.065 | passes 273 dense2 210 dense3 0 |
| brick | 24 | f32 | k3 | 1.0447 | 2.090 | 1.4e-9 | 1.045 1.071 1.049 | passes 273 dense2 210 dense3 0 |
## end 2026-10-04T22:16:23Z load={ 14.78 23.46 22.37 }
## 2026-10-04T22:17:28Z threads=6 load={ 11.34 20.68 21.40 } args: su4 24 f32 3 20 off:dense=0 k2:dense=2 k3:dense=3
threads 6 (rayon); cycles column = wall x clock x threads / amplitude-updates of the fused op list
| workload | n | prec | config | min wall s | core-cyc/amp-op | max dAmp vs first | all wall s | tiling |
|---|---|---|---|---|---|---|---|---|
| su4 | 24 | f32 | off | 2.4259 | 1.624 | 0.0e0 | 2.880 2.619 2.426 | passes 2070 |
| su4 | 24 | f32 | k2 | 1.1821 | 0.791 | 2.3e-9 | 1.324 1.282 1.182 | passes 243 dense2 230 dense3 0 |
| su4 | 24 | f32 | k3 | 1.1627 | 0.778 | 2.3e-9 | 1.359 1.227 1.163 | passes 243 dense2 230 dense3 0 |
## end 2026-10-04T22:17:44Z load={ 12.14 20.40 21.29 }
## 2026-10-04T22:18:49Z threads=6 load={ 10.08 18.26 20.42 } args: qft 24 f32 3 1 off:dense=0 k2:dense=2 k3:dense=3
threads 6 (rayon); cycles column = wall x clock x threads / amplitude-updates of the fused op list
| workload | n | prec | config | min wall s | core-cyc/amp-op | max dAmp vs first | all wall s | tiling |
|---|---|---|---|---|---|---|---|---|
| qft | 24 | f32 | off | 0.0930 | 1.084 | 0.0e0 | 0.137 0.105 0.093 | passes 59 |
| qft | 24 | f32 | k2 | 0.1019 | 1.188 | 0.0e0 | 0.103 0.102 0.119 | passes 58 dense2 1 dense3 0 |
| qft | 24 | f32 | k3 | 0.0960 | 1.119 | 0.0e0 | 0.096 0.101 0.111 | passes 58 dense2 1 dense3 0 |
## end 2026-10-04T22:18:50Z load={ 10.08 18.26 20.42 }
