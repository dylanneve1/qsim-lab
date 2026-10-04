#!/bin/zsh
# usage: bench.sh <label> <N> <seed> <prec f32|f64> <reps> <config>...
# config: old:<oracle>  |  ge:<we>:<wm>:<mbu>:<shor|eh>
Q=~/qsim-ge/target/release/qsim
G=~/qsim-ge/target/release/examples/ge_shor
label=$1; N=$2; seed=$3; prec=$4; reps=$5; shift 5
until mkdir /tmp/qsim-mac-bench.lock 2>/dev/null; do sleep 5; done
echo "== $label  $(date -u +%H:%M:%SZ)  load: $(uptime | sed 's/.*averages: //')"
for r in $(seq 1 $reps); do
  for cfg in "$@"; do
    if [[ $cfg == old:* ]]; then
      o=${cfg#old:}; extra=""; [[ $prec == f32 ]] && extra="--f32"
      echo "-- $cfg rep $r"
      QSIM_SLICE_PROFILE=1 /usr/bin/time -l $Q run shor --modulus $N --semiclassical --sliced --window 4 --oracle $o $extra --seed $seed --tries 1 2>&1 | egrep -i "measured|factor|time|profile|maximum resident|gates|real" | head -12
    else
      IFS=: read -r _ we wm mbu var <<< "$cfg"
      echo "-- $cfg rep $r"
      /usr/bin/time -l $G run $N $seed $we $wm $mbu $var $prec 2>&1 | egrep "Shor|EH|qubits|time|maximum resident" 
    fi
  done
done
rmdir /tmp/qsim-mac-bench.lock
echo "== released $(date -u +%H:%M:%SZ)"
