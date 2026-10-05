#!/bin/zsh
# usage: timed.sh <cmd...>   (run INSIDE flock /dev/shm/qsim/bench.lock)
# Prints load/memory before and after, peak RSS and wall time of the command.
# Address space capped at 13 GB (prlimit) so a mistake cannot OOM other users.
echo "== $(date '+%F %T %Z')  threads=${RAYON_NUM_THREADS:-16(default)}  $*"
echo "before: $(uptime | sed 's/.*load average/load average/')  | $(free -g | awk '/Mem:/{print "avail " $7 " GB, used " $3 " GB"}')"
prlimit --as=13958643712 /usr/bin/time -f "wall %e s  user %U s  sys %S s  maxrss %M KB" "$@" 2>&1
echo "after:  $(uptime | sed 's/.*load average/load average/')  | $(free -g | awk '/Mem:/{print "avail " $7 " GB, used " $3 " GB"}')"
