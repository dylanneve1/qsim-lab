#!/bin/sh
# Reproduce the heavy_hex 49x4020 and 49x5072 peaks with the anchor-free operator route (generic/solve_generic.py)
# from scratch. Python >= 3.10. About 90 s per circuit on one CPU once the venv exists.
# The solver never reads a target string. The 5072 result is compared with the public reference (tracker #105)
# only after it has finished. 49x4020 has no public reference string (it is BlueQubit Peak Portal problem P10), so
# this script prints only a sha256 of that peak; set SHOW_PEAK=1 to print the string itself.
set -e
cd "$(dirname "$0")"
here=$(pwd)
t_start=$(date +%s)
python3 -m venv .venv && . .venv/bin/activate && pip install -q -r requirements.txt
base=https://raw.githubusercontent.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io/1db844f1540a198c5620af49247e09fc28e7f61b/data/classically-verifiable-problems/circuit-models/peaked_circuit
shacheck() { python3 -c 'import hashlib,sys
for line in sys.stdin:
    h,f=line.split()
    d=hashlib.sha256(open(f,"rb").read()).hexdigest()
    print(f+": "+("OK" if d==h else "FAILED")); sys.exit(0 if d==h else 1)'; }
curl -sLO $base/peaked_circuit_heavy_hex_49x4020.qasm
curl -sLO $base/peaked_circuit_heavy_hex_49x5072.qasm
echo "ef92424a8e365905037e2a610a50ec8cd46b032a0fb01fd12b80d2d8ed0f112f  peaked_circuit_heavy_hex_49x4020.qasm" | shacheck
echo "a9c2409bad096273b64c722a29cf493f0dd621a80760e87f4b778f0bbb70ddc9  peaked_circuit_heavy_hex_49x5072.qasm" | shacheck
mkdir -p out
export PYTHONPATH="$here/generic"
for g in 4020 5072; do
  t0=$(python3 -c 'import time; print(time.time())')
  python3 -u generic/solve_generic.py peaked_circuit_heavy_hex_49x$g.qasm --json out/hh$g.json > out/hh$g.log 2>&1
  t1=$(python3 -c 'import time; print(time.time())')
  python3 - "$g" "$t0" "$t1" <<'PY'
import hashlib, json, os, sys
g, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
r = json.load(open(f"out/hh{g}.json"))
q = r["peak"][::-1]  # Qiskit order (qubit 0 rightmost), as on the tracker
show = g == "5072" or os.environ.get("SHOW_PEAK") == "1"
pk = f"peak (Qiskit order) {q}" if show else f"peak sha256 {hashlib.sha256(q.encode()).hexdigest()[:16]} (Qiskit order)"
print(f"heavy_hex_49x{g}: centre layer {r['centre']}, {r['absorbed']}/{r['gates']} gates absorbed, W max bond "
      f"{r['W_max_bond']}; {pk}; p={r['p']:.4f} (norm {r['norm']:.4f}, min|<Z>| {r['min_abs_z']:.3f}); wall {t1 - t0:.1f} s")
if g == "5072":  # post hoc only: public reference from tracker #105 (Qiskit order)
    ref = "1001100011100000001010100001011111111011110010110"
    print("  matches the public 49x5072 reference (#105)" if q == ref else
          f"  differs from the public reference in {sum(a != b for a, b in zip(q, ref))} bits")
PY
done
echo "total wall clock (including venv and download): $(( $(date +%s) - t_start )) s"
