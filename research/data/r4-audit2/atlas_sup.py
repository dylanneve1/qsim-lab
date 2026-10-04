import csv,json,subprocess
from concurrent.futures import ThreadPoolExecutor
B='/tmp/qsim-wt/r4-audit2-target/release/examples/magic_atlas'
R=[r for r in csv.DictReader(open('/tmp/qsim-wt/r4-audit2/research/data/magic-atlas/atlas.csv')) if int(r['toffolis'])>0]
def f(r):
    o=subprocess.run(['nice','-n','15',B,'profile',r['spec'],r['seed']],capture_output=True,text=True,timeout=600).stdout
    j=json.loads(o.strip().splitlines()[-1]); return r['spec'],int(r['support']),j['support'],int(r['n'])
with ThreadPoolExecutor(2) as ex: res=list(ex.map(f,R))
ch=[x for x in res if x[1]!=x[2]]
print('toffoli rows',len(res),'changed',len(ch))
for x in ch: print(x)
