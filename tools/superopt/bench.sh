#!/bin/bash
# Interleaved A/B timings, windowed vs windowed-opt (exp/superopt), Mac, under the bench lock.
Q=~/qsim-so-target/release/qsim
LOG=~/qsim-so-data/bench.log
run() { echo "== $* (load $(sysctl -n vm.loadavg)) $(date +%T)"; QSIM_SLICE_PROFILE=1 /usr/bin/time -l $Q run shor --semiclassical --sliced --window 4 "$@" 2>&1 | grep -v "^note" | grep -E "profile|a=|time |maximum resident"; }
lock() { until mkdir /tmp/qsim-mac-bench.lock 2>/dev/null; do sleep 5; done; echo "lock $1 $(date)"; }
unlock() { rmdir /tmp/qsim-mac-bench.lock; echo "unlock $(date)"; sleep 40; }
{
cd ~/qsim-so && git log --oneline -1
lock S1
for rep in 1 2 3; do for o in windowed windowed-opt; do run --oracle $o --modulus 10161323 --seed 1 --tries 1; done; done
for rep in 1 2 3; do for o in windowed windowed-opt; do run --oracle $o --modulus 221643407 --seed 1 --tries 1; done; done
unlock
for rep in 1 2 3; do
  for o in windowed windowed-opt; do
    lock "31-$rep-$o"
    run --oracle $o --f32 --modulus 1537596787 --seed 2 --tries 1
    unlock
  done
done
echo done $(date)
} >> $LOG 2>&1
