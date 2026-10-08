import json
g = json.load(open('/tmp/su2-254-mp/work/gauss.json'))
print("SCV step 0 (1st in array):", g['free']['SCV'][0]['stag'])
print("Mes step 0:", g['free']['meson'][0]['stag'])
