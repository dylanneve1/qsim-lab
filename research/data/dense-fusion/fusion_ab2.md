## 2026-10-04T22:30:09Z threads=6 load={ 10.66 11.41 15.31 } args: brick 24 f32 3 20 off:dense=0 k2raw:dense=2,dmin=2 k2:dense=2 k3raw:dense=3,dmin=2 k3:dense=3
threads 6 (rayon); cycles column = wall x clock x threads / amplitude-updates of the fused op list
| workload | n | prec | config | min wall s | core-cyc/amp-op | max dAmp vs first | all wall s | tiling |
|---|---|---|---|---|---|---|---|---|
| brick | 24 | f32 | off | 1.1801 | 2.361 | 0.0e0 | 1.945 1.180 1.191 | passes 692 |
| brick | 24 | f32 | k2raw | 1.2065 | 2.414 | 1.4e-9 | 1.609 1.242 1.206 | passes 273 dense2 210 dense3 0 |
| brick | 24 | f32 | k2 | 1.1182 | 2.237 | 0.0e0 | 1.136 1.140 1.118 | passes 692 dense2 0 dense3 0 |
| brick | 24 | f32 | k3raw | 1.2333 | 2.467 | 1.4e-9 | 1.247 1.240 1.233 | passes 273 dense2 210 dense3 0 |
| brick | 24 | f32 | k3 | 1.0698 | 2.140 | 0.0e0 | 1.102 1.070 1.113 | passes 692 dense2 0 dense3 0 |
## end 2026-10-04T22:30:29Z load={ 13.05 11.93 15.40 }
## 2026-10-04T22:31:34Z threads=6 load={ 14.11 12.59 15.40 } args: su4 24 f32 3 20 off:dense=0 k2raw:dense=2,dmin=2 k2:dense=2 k3raw:dense=3,dmin=2 k3:dense=3
threads 6 (rayon); cycles column = wall x clock x threads / amplitude-updates of the fused op list
| workload | n | prec | config | min wall s | core-cyc/amp-op | max dAmp vs first | all wall s | tiling |
|---|---|---|---|---|---|---|---|---|
| su4 | 24 | f32 | off | 2.1552 | 1.442 | 0.0e0 | 2.443 2.194 2.155 | passes 2070 |
| su4 | 24 | f32 | k2raw | 1.0617 | 0.711 | 2.3e-9 | 1.093 1.062 1.081 | passes 243 dense2 230 dense3 0 |
| su4 | 24 | f32 | k2 | 1.0420 | 0.697 | 2.3e-9 | 1.091 1.042 1.065 | passes 243 dense2 230 dense3 0 |
| su4 | 24 | f32 | k3raw | 1.0578 | 0.708 | 2.3e-9 | 1.108 1.074 1.058 | passes 243 dense2 230 dense3 0 |
| su4 | 24 | f32 | k3 | 1.0536 | 0.705 | 2.3e-9 | 1.078 1.054 1.064 | passes 243 dense2 230 dense3 0 |
## end 2026-10-04T22:31:54Z load={ 14.30 12.73 15.38 }
## 2026-10-04T22:32:59Z threads=6 load={ 11.16 12.12 14.96 } args: qft 24 f32 3 1 off:dense=0 k2raw:dense=2,dmin=2 k2:dense=2 k3raw:dense=3,dmin=2 k3:dense=3
threads 6 (rayon); cycles column = wall x clock x threads / amplitude-updates of the fused op list
| workload | n | prec | config | min wall s | core-cyc/amp-op | max dAmp vs first | all wall s | tiling |
|---|---|---|---|---|---|---|---|---|
| qft | 24 | f32 | off | 0.0977 | 1.139 | 0.0e0 | 0.145 0.098 0.121 | passes 59 |
| qft | 24 | f32 | k2raw | 0.0960 | 1.119 | 0.0e0 | 0.108 0.096 0.110 | passes 58 dense2 1 dense3 0 |
| qft | 24 | f32 | k2 | 0.1029 | 1.200 | 0.0e0 | 0.117 0.103 0.129 | passes 59 dense2 0 dense3 0 |
| qft | 24 | f32 | k3raw | 0.1041 | 1.214 | 0.0e0 | 0.118 0.111 0.104 | passes 58 dense2 1 dense3 0 |
| qft | 24 | f32 | k3 | 0.0976 | 1.138 | 0.0e0 | 0.100 0.111 0.098 | passes 59 dense2 0 dense3 0 |
## end 2026-10-04T22:33:01Z load={ 11.16 12.12 14.96 }
## 2026-10-04T22:34:06Z threads=6 load={ 12.04 12.23 14.79 } args: brick 24 f64 3 20 off:dense=0 k2raw:dense=2,dmin=2 k2:dense=2 k3raw:dense=3,dmin=2 k3:dense=3
threads 6 (rayon); cycles column = wall x clock x threads / amplitude-updates of the fused op list
| workload | n | prec | config | min wall s | core-cyc/amp-op | max dAmp vs first | all wall s | tiling |
|---|---|---|---|---|---|---|---|---|
| brick | 24 | f64 | off | 1.3882 | 2.777 | 0.0e0 | 2.098 1.433 1.388 | passes 692 |
| brick | 24 | f64 | k2raw | 2.1181 | 4.238 | 2.8e-18 | 2.271 2.118 2.121 | passes 272 dense2 210 dense3 0 |
| brick | 24 | f64 | k2 | 1.3962 | 2.794 | 0.0e0 | 1.396 1.410 1.408 | passes 692 dense2 0 dense3 0 |
| brick | 24 | f64 | k3raw | 2.0882 | 4.178 | 2.8e-18 | 2.094 2.088 2.189 | passes 272 dense2 210 dense3 0 |
| brick | 24 | f64 | k3 | 1.3980 | 2.797 | 0.0e0 | 1.398 1.424 1.444 | passes 692 dense2 0 dense3 0 |
## end 2026-10-04T22:34:32Z load={ 12.29 12.29 14.74 }
## 2026-10-04T22:35:37Z threads=6 load={ 15.13 13.10 14.86 } args: su4 24 f64 3 20 off:dense=0 k2raw:dense=2,dmin=2 k2:dense=2 k3raw:dense=3,dmin=2 k3:dense=3
threads 6 (rayon); cycles column = wall x clock x threads / amplitude-updates of the fused op list
| workload | n | prec | config | min wall s | core-cyc/amp-op | max dAmp vs first | all wall s | tiling |
|---|---|---|---|---|---|---|---|---|
| su4 | 24 | f64 | off | 4.1516 | 2.778 | 0.0e0 | 4.701 4.190 4.152 | passes 2070 |
| su4 | 24 | f64 | k2raw | 2.8052 | 1.877 | 5.6e-18 | 2.831 2.805 2.809 | passes 243 dense2 230 dense3 0 |
| su4 | 24 | f64 | k2 | 2.7922 | 1.869 | 5.6e-18 | 2.826 2.806 2.792 | passes 243 dense2 230 dense3 0 |
| su4 | 24 | f64 | k3raw | 2.7883 | 1.866 | 5.6e-18 | 2.788 2.796 2.814 | passes 243 dense2 230 dense3 0 |
| su4 | 24 | f64 | k3 | 2.7903 | 1.867 | 5.6e-18 | 2.790 2.813 2.813 | passes 243 dense2 230 dense3 0 |
## end 2026-10-04T22:36:24Z load={ 17.39 13.97 15.08 }
## 2026-10-04T22:37:29Z threads=8 load={ 13.81 13.56 14.84 } args: brick 26 f32 3 20 off:dense=0 k2raw:dense=2,dmin=2 k2:dense=2 k3raw:dense=3,dmin=2 k3:dense=3
threads 8 (rayon); cycles column = wall x clock x threads / amplitude-updates of the fused op list
| workload | n | prec | config | min wall s | core-cyc/amp-op | max dAmp vs first | all wall s | tiling |
|---|---|---|---|---|---|---|---|---|
| brick | 26 | f32 | off | 3.9827 | 2.444 | 0.0e0 | 4.547 4.170 3.983 | passes 752 |
| brick | 26 | f32 | k2raw | 4.5023 | 2.763 | 9.8e-10 | 4.920 4.656 4.502 | passes 299 dense2 227 dense3 0 |
| brick | 26 | f32 | k2 | 3.9148 | 2.403 | 0.0e0 | 4.136 3.915 3.987 | passes 752 dense2 0 dense3 0 |
| brick | 26 | f32 | k3raw | 4.4545 | 2.734 | 9.8e-10 | 4.688 4.455 4.465 | passes 299 dense2 227 dense3 0 |
| brick | 26 | f32 | k3 | 3.9208 | 2.406 | 0.0e0 | 4.165 3.921 4.175 | passes 752 dense2 0 dense3 0 |
## end 2026-10-04T22:38:35Z load={ 19.16 15.20 15.36 }
## 2026-10-04T22:40:25Z threads=8 load={ 12.40 13.63 14.70 } args: su4 26 f32 3 20 off:dense=0 k2raw:dense=2,dmin=2 k2:dense=2 k3raw:dense=3,dmin=2 k3:dense=3
threads 8 (rayon); cycles column = wall x clock x threads / amplitude-updates of the fused op list
| workload | n | prec | config | min wall s | core-cyc/amp-op | max dAmp vs first | all wall s | tiling |
|---|---|---|---|---|---|---|---|---|
| su4 | 26 | f32 | off | 9.4907 | 1.948 | 0.0e0 | 10.906 9.491 10.106 | passes 2250 |
| su4 | 26 | f32 | k2raw | 4.5955 | 0.943 | 1.4e-9 | 5.152 4.596 4.723 | passes 269 dense2 250 dense3 0 |
| su4 | 26 | f32 | k2 | 4.5315 | 0.930 | 1.4e-9 | 4.666 4.531 4.672 | passes 269 dense2 250 dense3 0 |
| su4 | 26 | f32 | k3raw | 4.6064 | 0.945 | 1.4e-9 | 4.606 5.031 4.717 | passes 269 dense2 250 dense3 0 |
| su4 | 26 | f32 | k3 | 4.5520 | 0.934 | 1.4e-9 | 4.705 5.060 4.552 | passes 269 dense2 250 dense3 0 |
## end 2026-10-04T22:41:53Z load={ 20.97 16.47 15.71 }
## 2026-10-04T22:42:58Z threads=6 load={ 13.36 15.10 15.25 } args: brick 20 f32 5 20 off:dense=0 k2raw:dense=2,dmin=2 k2:dense=2 k3raw:dense=3,dmin=2 k3:dense=3
threads 6 (rayon); cycles column = wall x clock x threads / amplitude-updates of the fused op list
| workload | n | prec | config | min wall s | core-cyc/amp-op | max dAmp vs first | all wall s | tiling |
|---|---|---|---|---|---|---|---|---|
| brick | 20 | f32 | off | 0.0651 | 2.520 | 0.0e0 | 0.125 0.104 0.089 0.183 0.065 | passes 572 |
| brick | 20 | f32 | k2raw | 0.0749 | 2.899 | 4.6e-9 | 0.160 0.075 0.079 0.129 0.082 | passes 219 dense2 177 dense3 0 |
| brick | 20 | f32 | k2 | 0.0688 | 2.663 | 0.0e0 | 0.226 0.095 0.069 0.074 0.075 | passes 572 dense2 0 dense3 0 |
| brick | 20 | f32 | k3raw | 0.0684 | 2.649 | 4.6e-9 | 0.148 0.088 0.078 0.068 0.076 | passes 219 dense2 177 dense3 0 |
| brick | 20 | f32 | k3 | 0.0586 | 2.269 | 0.0e0 | 0.149 0.099 0.118 0.059 0.070 | passes 572 dense2 0 dense3 0 |
## end 2026-10-04T22:43:01Z load={ 13.36 15.10 15.25 }
## 2026-10-04T22:44:06Z threads=6 load={ 11.40 14.22 14.90 } args: su4 20 f32 5 20 off:dense=0 k2raw:dense=2,dmin=2 k2:dense=2 k3raw:dense=3,dmin=2 k3:dense=3
threads 6 (rayon); cycles column = wall x clock x threads / amplitude-updates of the fused op list
| workload | n | prec | config | min wall s | core-cyc/amp-op | max dAmp vs first | all wall s | tiling |
|---|---|---|---|---|---|---|---|---|
| su4 | 20 | f32 | off | 0.1218 | 1.578 | 0.0e0 | 0.270 0.275 0.199 0.140 0.122 | passes 1710 |
| su4 | 20 | f32 | k2raw | 0.0603 | 0.781 | 6.8e-9 | 0.140 0.115 0.072 0.060 0.060 | passes 209 dense2 190 dense3 0 |
| su4 | 20 | f32 | k2 | 0.0610 | 0.791 | 6.8e-9 | 0.171 0.109 0.066 0.061 0.083 | passes 209 dense2 190 dense3 0 |
| su4 | 20 | f32 | k3raw | 0.0616 | 0.799 | 6.8e-9 | 0.143 0.133 0.063 0.062 0.064 | passes 209 dense2 190 dense3 0 |
| su4 | 20 | f32 | k3 | 0.0637 | 0.826 | 6.8e-9 | 0.154 0.086 0.066 0.072 0.064 | passes 209 dense2 190 dense3 0 |
## end 2026-10-04T22:44:09Z load={ 11.93 14.28 14.92 }
## 2026-10-04T22:45:14Z threads=6 load={ 13.28 14.20 14.84 } args: ghz 24 f32 3 1 off:dense=0 k2raw:dense=2,dmin=2 k2:dense=2 k3raw:dense=3,dmin=2 k3:dense=3
threads 6 (rayon); cycles column = wall x clock x threads / amplitude-updates of the fused op list
| workload | n | prec | config | min wall s | core-cyc/amp-op | max dAmp vs first | all wall s | tiling |
|---|---|---|---|---|---|---|---|---|
| ghz | 24 | f32 | off | 0.0576 | 5.317 | 0.0e0 | 0.100 0.066 0.058 | passes 24 |
| ghz | 24 | f32 | k2raw | 0.0598 | 5.520 | 0.0e0 | 0.095 0.060 0.065 | passes 23 dense2 1 dense3 0 |
| ghz | 24 | f32 | k2 | 0.0565 | 5.221 | 0.0e0 | 0.102 0.059 0.057 | passes 24 dense2 0 dense3 0 |
| ghz | 24 | f32 | k3raw | 0.1189 | 10.978 | 0.0e0 | 0.171 0.152 0.119 | passes 13 dense2 0 dense3 10 |
| ghz | 24 | f32 | k3 | 0.0443 | 4.089 | 0.0e0 | 0.056 0.044 0.064 | passes 24 dense2 0 dense3 0 |
## end 2026-10-04T22:45:16Z load={ 13.28 14.20 14.84 }
DONE
