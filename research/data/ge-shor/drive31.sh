#!/bin/zsh
# 31-bit timing campaign: one run per locked chunk, >= 65 s gaps, memory check before each
memok() { local f=$(vm_stat | awk '/Pages free/ {f=$3} /Pages inactive/ {i=$3} END {gsub("\\.","",f); gsub("\\.","",i); print (f+i)*16384/1e9}'); echo "free+inactive ${f} GB"; [[ ${f%.*} -ge 6 ]]; }
run() { # label lanes config
  until memok; do sleep 30; done
  QSIM_SLICE_LANES=$2 ~/qsim-ge-data/bench.sh "$1 lanes=$2" 1537596787 2 f32 1 $3
  sleep 70
}
for rep in 1 2 3; do
  run 31bit 32 ge:2:3:lookups:shor
  run 31bit 16 old:windowed-mbu-lookup
done
run 31bit 32 ge:2:4:all:shor
run 31bit 32 old:windowed-mbu-lookup
echo DONE
