#!/bin/bash
# usage: dist.sh d rounds spec basis  -> one-line summary + JSON appended to runs/dist.jsonl
B=${CS:-/tmp/cf-target/release/examples/color_search}
out=$(nice -n 15 $B distance $1 $2 $3 ${CAP:-100000} 18446744073709551615 ${NOISE:-cnot} $4)
mkdir -p runs; echo "$out" | python3 -c "import json,sys; j=json.load(sys.stdin); j['basis']='$4'; open('runs/dist.jsonl','a').write(json.dumps(j)+'\n'); print('d=%d R=%d %s %s: d_circ=%s N=%s cert=%s %.1fs  %s'%(j['d'],j['rounds'],j['schedule'],'$4',j['distance'],j['count'],j['certified'],j['seconds'],j['example'][:200]))"
