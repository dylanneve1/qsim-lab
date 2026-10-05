#!/bin/sh
# Reproduce the P11 / P12 peaks from scratch (Python >= 3.10). About 5 minutes on one CPU.
set -e
cd "$(dirname "$0")"
python3 -m venv .venv && . .venv/bin/activate && pip install -q -r requirements.txt
base=https://raw.githubusercontent.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io/1db844f1540a198c5620af49247e09fc28e7f61b/data/classically-verifiable-problems/circuit-models/peaked_circuit
curl -sLO $base/peaked_circuit_P11_Hqap_98x1999.qasm
curl -sLO $base/peaked_circuit_P12_Hqap_98x2457.qasm
echo "1373d50c8a42b1ca745d202391767c417ddac56db95182ac2aced019231b3372  peaked_circuit_P11_Hqap_98x1999.qasm" | sha256sum -c
echo "868ff86a396f86a8cbca48f7127e49c4c4f951a5a955295e5010a92b76be961d  peaked_circuit_P12_Hqap_98x2457.qasm" | sha256sum -c
python3 solve_peaked.py peaked_circuit_P11_Hqap_98x1999.qasm peaked_circuit_P12_Hqap_98x2457.qasm | tee out.log
grep -q 10101110111010011111100010110011101011101011111001010101101100001110101110010000010100001001100000 out.log && echo "P11 peak matches the Helios-1 result"
grep -q 10100011110010100111000100011100110001011111011100111001010110101011001001000000101000100000111100 out.log && echo "P12 peak matches the Helios-1 result"
