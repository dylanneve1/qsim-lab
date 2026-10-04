#!/bin/bash
X=/tmp/qsim-wt/tgt-shor-noise/release/examples/shor_noise
cd /tmp/qsim-wt/vpsdata
run(){ name=$1; shift; [ -f $name.csv ] && return; RAYON_NUM_THREADS=2 nice -n 15 $X "$@" > $name.tmp 2>>vps.err && mv $name.tmp $name.csv; }
run z12 strat 2867 1388 4 phaseflip 3 300 412 8388608
run z14 strat 8453 8448 4 phaseflip 3 300 414 8388608
run z16 strat 34387 25590 4 phaseflip 3 300 416 8388608
run z18 strat 200479 103279 4 phaseflip 3 300 418 8388608
run z20 strat 821749 118176 4 phaseflip 3 300 420 8388608
run x12 strat 2867 1388 4 bitflip 1 300 512 8388608
run z22 strat 2957047 555422 4 phaseflip 3 200 422 8388608
run z24 strat 10161323 321017 4 phaseflip 3 200 424 8388608
echo VPSQ-DONE >> vps.err
