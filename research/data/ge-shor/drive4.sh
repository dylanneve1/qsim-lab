#!/bin/zsh
while pgrep -f drive3.sh >/dev/null; do sleep 20; done
sleep 70
memok() { local f=$(vm_stat | awk '/Pages free/ {f=$3} /Pages inactive/ {i=$3} END {gsub("\\.","",f); gsub("\\.","",i); print (f+i)*16384/1e9}'); echo "free+inactive ${f} GB (need $1)"; [[ ${f%.*} -ge $1 ]]; }
for rep in 1 2; do
  until memok 6; do sleep 30; done
  QSIM_SLICE_LANES=32 ~/qsim-ge-data/bench.sh "31bit lanes=32" 1537596787 2 f32 1 ge:2:3:lookups:shor
  sleep 70
  until memok 6; do sleep 30; done
  ~/qsim-ge-data/bench.sh "31bit lanes=16" 1537596787 2 f32 1 old:windowed-mbu-lookup
  sleep 70
done
echo DONE
