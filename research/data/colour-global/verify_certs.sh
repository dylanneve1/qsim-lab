#!/bin/bash
# drat-trim every certs/*.cnf against its .drat; prints one verdict line per certificate
DT=${1:-drat-trim}
for c in certs/*.cnf; do
  n=${c%.cnf}
  [ -f $n.drat ] || { echo "$(basename $n): no proof"; continue; }
  v=$($DT $c $n.drat -t 2000 2>&1 | grep -o "s VERIFIED\|s NOT VERIFIED" | head -1)
  echo "$(basename $n): $(sed -n 2p $c) -> ${v:-no verdict}"
done
