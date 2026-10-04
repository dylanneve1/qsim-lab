#!/bin/bash
# One locked run of `planner_v2 feat` (planner features + per-tier timings) on the Mac.
B=$1; OUT=$2
"$B" feat instances.txt > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"
