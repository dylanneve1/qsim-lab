from chain import *
import sys
n0, ops = parse()
D = int(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 70
w = window(n0, ops, n, 70-D, 70); lines, be = chain(n, w)
p = compile_plan(lines, 0x5a5a5a5a5a5a5a5a5 & ((1 << n)-1))
print('W', p['width'], 'ops', len(p['ops']), 'backward', ''.join('B' if b else 'f' for b in p['backward']))
print('cut', p['cut_width'])
for i in [0, 1, 2, 3, 30, 31, 32, 68, 69][:]:
    if i >= n: continue
    seq = []
    for op in p['ops'][p['qubit_ops'][i]:p['qubit_ops'][i+1]]:
        if op[0] == 'U':
            m = op[2]; nd = abs(m[0, 1]) > 0 or abs(m[1, 0]) > 0
            seq.append(('u%d' % op[1] if nd else 'd%d' % op[1]) + ('c%s' % [b for b in range(64) if op[3] >> b & 1] if op[3] else ''))
        else:
            bits = [b for b in range(64) if op[1] >> b & 1]
            if len(bits) > 1: seq.append('Z' + '-'.join(map(str, bits)))
    print(i, 'B' if p['backward'][i] else 'f', ' '.join(seq))
