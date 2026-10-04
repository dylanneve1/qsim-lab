#!/bin/bash
# DRAT certificates for the d = 11 hook-free-boundary UNSAT claims of research/colour-flags.md.
# Same pipeline as ../colour-global/certify.sh: CEGAR CNF (encoding + cuts) -> Glucose DRAT -> drat-trim.
# usage: certify_flags.sh <python> <drat-trim>   (run from this directory; writes certs/)
PY=${1:-python3}; DT=${2:-drat-trim}
mkdir -p certs
run() {
  name=$1; shift
  (cd ../colour-global && $PY cg_sat.py "$@" --cnf ../colour-flags/certs/$name.cnf) > certs/$name.log 2>&1
  res=$(grep RESULT certs/$name.log)
  if echo "$res" | grep -q UNSAT; then
    $PY -c "
from pysat.formula import CNF; from pysat.solvers import Solver
f=CNF(from_file='certs/$name.cnf'); s=Solver(name='glucose4',bootstrap_with=f.clauses,with_proof=True)
assert not s.solve(); open('certs/$name.drat','w').write('\n'.join(s.get_proof())+'\n')"
    v=$($DT certs/$name.cnf certs/$name.drat -t 4000 | grep '^s ')
    nv=$(head -1 certs/$name.cnf)
  else v="(not UNSAT)"; nv=""; fi
  echo "$name | $* | $res | $nv | $v" | tee -a certs/summary.txt
}
run d11_D11_hfbnd_free 11 11 1 free --warm --sym --hookfree 27 --hookfree-set boundary
run d11_D11_hfbnd_kfT7 11 11 1 kf --T 7 --warm --sym --hookfree 27 --hookfree-set boundary
run d11_D11_hfbnd_kfT8 11 11 1 kf --T 8 --warm --sym --hookfree 27 --hookfree-set boundary
