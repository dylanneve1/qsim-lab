import sys, re, collections
import numpy as np

QASM = '/home/dylan/.talon/workspace/research/doped-clifford/nq70_depth70_checks27_doped.qasm'
TWEIGHTS = '/home/dylan/.talon/workspace/research/doped-clifford/nq70_depth70_checks27_doped.qasm.tweights.tsv'

def load_circuit():
    with open(QASM) as f:
        lines = [l.strip() for l in f if l.strip()]
        
    ops = []
    q_depth = [0]*70
    t_gates = [] # (qubit, cz_layer, global_op_idx)
    cz_gates = [] # (q1, q2, cz_layer, global_op_idx)
    
    for idx, l in enumerate(lines):
        if l.startswith('cz'):
            m = re.match(r'cz q\[(\d+)\],\s*q\[(\d+)\];', l)
            q1, q2 = sorted([int(m.group(1)), int(m.group(2))])
            d = max(q_depth[q1], q_depth[q2]) + 1
            q_depth[q1] = d
            q_depth[q2] = d
            cz_gates.append((q1, q2, d, idx))
            ops.append(('cz', (q1, q2), d))
        elif l.startswith('rz'):
            m = re.match(r'rz\(pi/4\)\s+q\[(\d+)\];', l)
            q = int(m.group(1))
            d = q_depth[q] # cz layer preceding this T gate
            t_gates.append((q, d, idx))
            ops.append(('t', (q,), d))
        elif any(l.startswith(k) for k in ['h ', 's ', 'sx ', 'sxdg ', 'sdg ']):
            m = re.match(r'([a-z]+)\s+q\[(\d+)\];', l)
            name, q = m.group(1), int(m.group(2))
            ops.append((name, (q,), q_depth[q]))
            
    return ops, t_gates, cz_gates

def analyze_h2():
    ops, t_gates, cz_gates = load_circuit()
    
    # 1. Heat map of T gates: qubit x CZ layer
    # CZ layers are 0..70 (0 means before any CZ, 70 means after 70th CZ)
    grid = np.zeros((70, 71), dtype=int)
    for q, d, _ in t_gates:
        grid[q, d] += 1
        
    print(f"Total T gates loaded: {len(t_gates)}")
    
    # Analyze distribution across depth
    print("\n=== T Gate Distribution by Depth (CZ layers) ===")
    depth_brackets = [(0, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60), (61, 63), (64, 70)]
    for lo, hi in depth_brackets:
        cnt = np.sum(grid[:, lo:hi+1])
        active_q = np.where(np.sum(grid[:, lo:hi+1], axis=1) > 0)[0]
        q_range = f"q[{min(active_q):02d}..{max(active_q):02d}]" if len(active_q) > 0 else "None"
        print(f"Layers {lo:02d}..{hi:02d}: {cnt:3d} T gates | active qubits: {len(active_q):2d} ({q_range})")
        
    # T gates in last 7 layers (64..70)
    cnt_last7 = np.sum(grid[:, 64:71])
    cnt_last8 = np.sum(grid[:, 63:71])
    print(f"\nLast 7 layers (64..70): {cnt_last7} T gates ({cnt_last7/468*100:.1f}%)")
    print(f"Last 8 layers (63..70): {cnt_last8} T gates ({cnt_last8/468*100:.1f}%)")
    
    # Analyze distribution across qubits
    print("\n=== T Gate Distribution by Qubits ===")
    for q_start in range(0, 70, 10):
        sub = np.sum(grid[q_start:q_start+10, :])
        early_sub = np.sum(grid[q_start:q_start+10, 0:50]) # early layers
        late_sub = np.sum(grid[q_start:q_start+10, 50:71])  # late layers
        print(f"q[{q_start:02d}..{q_start+9:02d}]: total {sub:3d} (layers 0..49: {early_sub:2d}, layers 50..70: {late_sub:3d})")
        
    # Early layers T distribution (layers 0..30)
    early_t_by_q = np.sum(grid[:, 0:31], axis=1)
    print("\nEarly T gates (layers 0..30) by qubit:")
    print("Nonzero qubits in layers 0..30:")
    for q in range(70):
        if early_t_by_q[q] > 0:
            print(f"  q[{q:02d}]: {early_t_by_q[q]} T gates (layers: {np.where(grid[q, 0:31] > 0)[0].tolist()})")
            
    # Past light cone for each chain cut e (bonds between e and e+1)
    # A CZ on (e, e+1) at layer L:
    # Backwards causal cone from edge (e, e+1) at layer L spreads by 1 qubit per CZ layer
    # Since brickwork CZ has speed 1 site per layer, from (e, e+1) at layer L,
    # the lightcone backwards to layer l <= L covers qubits [e - (L - l), e + 1 + (L - l)]
    # In the chain sweep, the boundary state at edge e comes from contracting ALL gates on qubits 0..e
    # BUT let's compute both:
    # (A) All T gates on qubits 0..e (the region contracted by the left sweep)
    # (B) T gates strictly in the backward lightcone of the 35 CZ bonds on edge (e, e+1)
    
    # Compute backward lightcone in 1D brickwork:
    # For a bond at (e, e+1) at layer L:
    # Any gate at (q, l) with l <= L is in its backward light cone iff
    # if q <= e: e - q <= L - l  (i.e. q >= e - (L - l))
    # if q > e:  q - (e+1) <= L - l (i.e. q <= e + 1 + (L - l))
    
    # Since the 35 CZ bonds on edge e sit at layers L = 1, 3, 5... or 2, 4, 6... up to 70:
    # The latest bond sits at layer L_max = 69 or 70.
    # From layer 70, backward light cone reaches distance 70 in both directions!
    # Since n=70, the backward light cone from layer 70 covers the ENTIRE CIRCUIT!
    
    print("\n=== T Gate Counts for Chain Cuts e in {5, 10, 15, 20, 25, 30, 34, 40, 45, 50, 55, 60, 65} ===")
    print("cut e | T in L (q<=e) | T in R (q>e) | T in L (past cone) | T in L (layers 0..59) | T in L (last 10)")
    cuts_to_test = [5, 10, 15, 20, 25, 30, 34, 40, 45, 50, 55, 60, 65]
    
    # Find layers of CZs on each edge
    cz_layers_by_edge = collections.defaultdict(list)
    for q1, q2, d, _ in cz_gates:
        cz_layers_by_edge[q1].append(d)
        
    for e in cuts_to_test:
        bonds_L = cz_layers_by_edge[e] # layers of CZs on edge e
        t_in_left = sum(1 for q, d, _ in t_gates if q <= e)
        t_in_right = sum(1 for q, d, _ in t_gates if q > e)
        t_left_early = sum(1 for q, d, _ in t_gates if q <= e and d < 60)
        t_left_late = sum(1 for q, d, _ in t_gates if q <= e and d >= 60)
        
        # Backward light cone from the CZ bonds on edge e:
        # A T gate at (q, d) with q <= e is in the past light cone if
        # there exists some bond layer L in bonds_L such that d <= L and (e - q) <= (L - d)
        in_past_cone = 0
        for q, d, _ in t_gates:
            if q <= e:
                # check if reachable
                reachable = any(d <= L and (e - q) <= (L - d) for L in bonds_L)
                if reachable:
                    in_past_cone += 1
                    
        print(f" {e:4d} | {t_in_left:12d} | {t_in_right:12d} | {in_past_cone:18d} | {t_left_early:21d} | {t_left_late:16d}")

if __name__ == '__main__':
    analyze_h2()
