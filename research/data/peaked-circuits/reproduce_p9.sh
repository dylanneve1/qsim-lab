#!/bin/sh
# Reproduce the P9 (peaked_circuit_P9_Hqap_56x1917) peak with mpou2 (MPO + exact centre block + mirror-partner
# scheduling) from scratch. Python >= 3.10. About 3-10 s per cutoff once the venv exists.
# The solver never reads a target string. The public reference (tracker #106 / #153 / #241, Kremer-Dupuis notebook)
# is compared only after all four runs have finished.
# eps=3e-4 is much slower than the others (46 s on an M1 Pro, ~14 min on a loaded 4-vCPU VM); the headline runtime
# is the single eps=1e-3 run.
set -e
cd "$(dirname "$0")"
here=$(pwd)
t_start=$(date +%s)
python3 -m venv .venv && . .venv/bin/activate && pip install -q -r requirements.txt
base=https://raw.githubusercontent.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io/1db844f1540a198c5620af49247e09fc28e7f61b/data/classically-verifiable-problems/circuit-models/peaked_circuit
f=peaked_circuit_P9_Hqap_56x1917.qasm
curl -sLO $base/$f
shacheck() { if command -v sha256sum >/dev/null; then sha256sum -c; else shasum -a 256 -c; fi; }
echo "f043c6cdf9e1a1ba9b3e68ea3b893277e8acacab80319d6a8524cc022886a3f9  $f" | shacheck
mkdir -p out
export PYTHONPATH="$here/mpou2:$here/generic"
for e in 1e-2 3e-3 1e-3 3e-4; do
  t0=$(python3 -c 'import time; print(time.time())')
  python3 -u mpou2/run2.py $f --m 49 --band 45 55 --tno_band --tno_cut 1e-3 --mode rel --eps $e --maxb 512 \
    --tau 1e5 --match_hi 0 --sched pair --log_every 400 --out out/p9_eps$e.json > out/p9_eps$e.log 2>&1
  t1=$(python3 -c 'import time; print(time.time())')
  python3 - "$e" "$t0" "$t1" <<'PY'
import json, sys
e, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
r = json.load(open(f"out/p9_eps{e}.json"))
print(f"eps={e}: peak (qubit 0 leftmost) {r['peak']}  p={r['p']:.4f}  max single flip {r['max_flip']:.2e}"
      f"  flips above peak {r['n_flips_higher']}  wall {t1 - t0:.1f} s")
PY
done
python3 - <<'PY'
import json
peaks = {json.load(open(f"out/p9_eps{e}.json"))["peak"] for e in ["1e-2", "3e-3", "1e-3", "3e-4"]}
print("all four cutoffs agree" if len(peaks) == 1 else f"cutoffs DISAGREE: {len(peaks)} distinct peaks")
pk = peaks.pop()
print("peak, qubit 0 leftmost:            ", pk)
print("peak, Qiskit order (qubit 0 rightmost):", pk[::-1])
# Post hoc only: the public P9 reference from the Kremer-Dupuis unswapping notebook (cells 13-14; also tracker #241).
# That string comes from MPS samples in qubit order, i.e. qubit 0 LEFTMOST (not Qiskit order).
ref = "01101110111001100000100000001010011100101101010111110111"
print("matches the public P9 reference (qubit 0 leftmost)" if pk == ref else
      f"differs from the public P9 reference in {sum(a != b for a, b in zip(pk, ref))} bits")
PY
echo "total wall clock (including venv and download): $(( $(date +%s) - t_start )) s"
