#!/bin/bash
# Re-derive every UNSAT claim of research/colour-global.md as a CNF (schedule-space encoding +
# CEGAR cuts), re-solve it with Glucose emitting a DRAT proof, and check it with drat-trim.
# usage: certify.sh <python> <drat-trim>   (writes certs/<name>.{cnf,drat,log} and certs/summary.txt)
PY=${1:-python3}; DT=${2:-drat-trim}
mkdir -p certs
run() {
  name=$1; shift
  $PY cg_sat.py "$@" --cnf certs/$name.cnf > certs/$name.log 2>&1
  res=$(grep RESULT certs/$name.log)
  if echo "$res" | grep -q UNSAT; then
    $PY -c "
from pysat.formula import CNF; from pysat.solvers import Solver
f=CNF(from_file='certs/$name.cnf'); s=Solver(name='glucose4',bootstrap_with=f.clauses,with_proof=True)
assert not s.solve(); open('certs/$name.drat','w').write('\n'.join(s.get_proof())+'\n')"
    v=$($DT certs/$name.cnf certs/$name.drat -t 1000 | grep '^s ')
    nv=$(head -2 certs/$name.cnf | tail -1)
  else v="(not UNSAT)"; nv=""; fi
  echo "$name | $* | $res | $nv | $v" | tee -a certs/summary.txt
}
run d5_D5_space_free 5 5 1 free --space
run d7_D7_space_free 7 7 1 free --space
run d9_D9_space_free 9 9 1 free --space
run d9_D8_kf_fixint 9 8 1 kf --warm --fix-interior
run d5_D5_hf8 5 5 1 free --space --hookfree 8
run d7_D7_hf14 7 7 1 free --space --hookfree 14
run d9_D9_hf20 9 9 1 free --space --hookfree 20
run d7_D7_hf3corners 7 7 1 free --space --hookfree 3 --hookfree-set corners
run d5_D5_split_all 5 5 1 free --space --split 9
run d7_D7_split_all 7 7 1 free --space --split 18
run d9_D9_split_all 9 9 1 free --space --split 30
run d9_D9_hfbnd_kfT6 9 9 1 kf --warm --hookfree 21 --hookfree-set boundary
run d9_D9_hfbnd_sepT6 9 9 1 sep --warm --hookfree 21 --hookfree-set boundary
