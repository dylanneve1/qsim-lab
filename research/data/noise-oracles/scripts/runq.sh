#!/bin/bash
# usage: runq.sh BIN QUEUE OUTDIR CONC THREADS [mac]
# Runs the queue's jobs one after another (skipping finished ones). With
# `mac`: SIGSTOPs the job while anyone holds /tmp/qsim-mac-bench.lock (the
# swarm's timing lock) or while free+inactive memory < 3 GB; stops after the
# current job if OUTDIR/STOP exists.
BIN=$1; Q=$2; OUT=$3; CONC=$4; TH=$5; MAC=$6
mkdir -p "$OUT"
memfree_gb() {
  vm_stat | awk '/Pages free/ {f=$3} /Pages inactive/ {i=$3} END {gsub("\\.","",f); gsub("\\.","",i); printf "%d", (f+i)*16384/1073741824}'
}
while read -r name rest; do
  [ -z "$name" ] && continue
  [ -e "$OUT/STOP" ] && { echo "STOP file, exiting"; exit 0; }
  [ -e "$OUT/$name.csv.gz" ] && continue
  envs=(); args=()
  for tok in $rest; do
    if [ ${#args[@]} -eq 0 ] && [[ "$tok" == *=* ]]; then envs+=("$tok"); else args+=("$tok"); fi
  done
  start=$(date +%s)
  env "${envs[@]}" QSIM_NOISE_CONC=$CONC RAYON_NUM_THREADS=$TH nice -n 15 "$BIN" "${args[@]}" > "$OUT/$name.csv.tmp" 2>> "$OUT/err.log" &
  pid=$!
  if [ "$MAC" = mac ]; then
    stopped=0
    while kill -0 $pid 2>/dev/null; do
      if [ -d /tmp/qsim-mac-bench.lock ] || [ "$(memfree_gb)" -lt 3 ]; then
        [ $stopped -eq 0 ] && { kill -STOP $pid; stopped=1; }
      else
        [ $stopped -eq 1 ] && { kill -CONT $pid; stopped=0; }
      fi
      sleep 2
    done
  fi
  wait $pid; rc=$?
  end=$(date +%s)
  if [ $rc -eq 0 ]; then gzip -c "$OUT/$name.csv.tmp" > "$OUT/$name.csv.gz" && rm "$OUT/$name.csv.tmp"; echo "ok $name $((end-start))s"; else mv "$OUT/$name.csv.tmp" "$OUT/$name.csv.partial"; echo "FAIL rc=$rc $name $((end-start))s"; fi
done < "$Q"
echo queue done
