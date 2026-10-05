#!/bin/sh
# Downloads the four benchmark circuits that are not committed here (their
# headers forbid redistribution: hwb6, hwb8, gf2^16_mult, gf2^32_mult) from
# Feynman (github.com/meamy/feynman, benchmarks/qc) at the pinned commit
# into circuits/ next to this script, and checks their sha256 sums.
# The committed circuits come from the same commit (circuits.sha256).
set -eu
here=$(cd "$(dirname "$0")" && pwd)
sha=90cb0c807321fb356587c7be623d1d5607666c2d
for f in hwb6 hwb8 'gf2^16_mult' 'gf2^32_mult'; do
  enc=$(printf '%s' "$f.qc" | sed 's/\^/%5E/g')
  curl -sfL "https://raw.githubusercontent.com/meamy/feynman/$sha/benchmarks/qc/$enc" \
    -o "$here/circuits/$f.qc"
done
cd "$here/circuits" && sha256sum -c ../fetched.sha256 && sha256sum -c ../circuits.sha256 >/dev/null && echo "all circuits verified"
