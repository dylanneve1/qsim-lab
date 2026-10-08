import gauss
rec, _, _ = gauss.run('SCV', mode='free')
for r in rec:
    if r['step'] == 20: print("SCV 20:", r['stag'])
