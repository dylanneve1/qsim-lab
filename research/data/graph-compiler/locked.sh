#!/bin/zsh
# usage: locked.sh <log> <cmd...>   (takes the swarm bench lock, max 140 s)
log=$1; shift
until mkdir /tmp/qsim-mac-bench.lock 2>/dev/null; do sleep 5; done
trap 'rmdir /tmp/qsim-mac-bench.lock 2>/dev/null' EXIT
echo "# $(date -u +%FT%TZ) load $(sysctl -n vm.loadavg) :: $*" >> $log
timeout 140 "$@" >> $log 2>&1
echo "# exit $? $(date -u +%FT%TZ)" >> $log
