#!/bin/zsh
memok() { local f=$(vm_stat | awk '/Pages free/ {f=$3} /Pages inactive/ {i=$3} END {gsub("\\.","",f); gsub("\\.","",i); print (f+i)*16384/1e9}'); echo "free+inactive ${f} GB (need $1)"; [[ ${f%.*} -ge $1 ]]; }
for rep in 1 2 3; do QSIM_SLICE_LANES=32 ~/qsim-ge-data/bench.sh "24bit-eh lanes=32" 10161323 1 f64 1 ge:2:3:lookups:eh ge:2:3:lookups:eh-odd; sleep 65; done
for rep in 1 2 3; do
  until memok 5; do sleep 30; done
  QSIM_SLICE_LANES=32 ~/qsim-ge-data/bench.sh "28bit lanes=32" 221643407 1 f64 1 ge:2:3:lookups:shor old:windowed-mbu-lookup ge:2:4:all:shor ge:2:3:lookups:eh-odd
  sleep 65
  until memok 5; do sleep 30; done
  ~/qsim-ge-data/bench.sh "28bit lanes=16" 221643407 1 f64 1 old:windowed-mbu-lookup
  sleep 65
done
for cfg in ge:2:3:lookups:eh-odd ge:2:3:lookups:eh; do
  until memok 6; do sleep 30; done
  QSIM_SLICE_LANES=32 ~/qsim-ge-data/bench.sh "31bit lanes=32" 1537596787 2 f32 1 $cfg
  sleep 70
done
echo DONE
