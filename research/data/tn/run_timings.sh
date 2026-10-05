#!/usr/bin/env bash
# Timed quimb contractions (quimb_timings.py) under the global bench lock (waits for >= 10 GB available),
# prlimit --as=$AS (default 3e9; the e2e search uses COTENGRA_NUM_WORKERS=2), 8 BLAS threads pinned to
# the 8 physical cores (even logical CPUs 0,2,...,14), nice 15. (Runs before 12:35 used --as=12 GiB and the e2e
# search used 8 workers; the JSON lines record rlimit_as_bytes when known.)
# Pauses run_paths.sh (PAUSE file) and waits for any running search to finish first.
# Usage: bash run_timings.sh <circuit stem> [contract|slices|e2e] [dtypes...]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-/dev/shm/qsim/ext/tn-py/work}
STEM=$1; MODE=${2:-contract}; shift 2 || shift $#
DTYPES=${*:-complex64}
export PYTHONPATH=/dev/shm/qsim/ext/tn-py OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 COTENGRA_NUM_WORKERS=2
AS=${AS:-3000000000}
touch "$WORK/PAUSE"
while [ "$(free -m | awk '/^Mem:/ {print $7}')" -lt 10000 ] || pgrep -f "^/dev/shm/qsim/venv/bin/python [^ ]*quimb_paths.py" > /dev/null; do sleep 10; done
echo "== $(date -Iseconds) $STEM $MODE $DTYPES"; uptime; free -g | head -2
flock /dev/shm/qsim/bench.lock bash -c "
  echo \"lock acquired \$(date -Iseconds)\"; uptime; free -g | head -2
  taskset -c 0,2,4,6,8,10,12,14 nice -n 15 prlimit --as=$AS /dev/shm/qsim/venv/bin/python \
    '$HERE/quimb_timings.py' $MODE $STEM --dtypes $DTYPES --reps 3 --work '$WORK' --out '$HERE/quimb_timings.jsonl'
  echo \"rc=\$? releasing lock \$(date -Iseconds)\"; uptime; free -g | head -2"
