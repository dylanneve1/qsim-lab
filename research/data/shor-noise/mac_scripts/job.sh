#!/bin/bash
# usage: job.sh outname <shor_noise args...>   (one locked run, hard limit 178 s)
out=~/qsim-noise-data/$1.csv; shift
until mkdir /tmp/qsim-mac-bench.lock 2>/dev/null; do sleep 5; done
start=$(date +%s)
echo "# load $(sysctl -n vm.loadavg) start $(date -u +%FT%TZ)" > $out.tmp
env QSIM_NOISE_KMIN=${KMIN:-0} ${RST:+QSIM_NOISE_RESET=1} QSIM_NOISE_CONC=${CONC:-4} perl -e 'alarm shift; exec @ARGV' 178 ~/qsim-noise-target/release/examples/shor_noise "$@" >> $out.tmp 2>> ~/qsim-noise-data/err.log
rc=$?
rmdir /tmp/qsim-mac-bench.lock
end=$(date +%s)
if [ $rc -eq 0 ]; then mv $out.tmp $out; echo "ok $out $((end-start))s"; else mv $out.tmp $out.partial; echo "FAIL rc=$rc $out $((end-start))s"; fi
