import numpy as np

def build_circuit(L, steps, dt, U, t_h):
    # qubits: 0 to 2L-1
    # layout at start:
    # even i: q_{2i} = dn, q_{2i+1} = up
    # odd i: q_{2i} = up, q_{2i+1} = dn
    gates = []
    
    # Let's say layer sequence for step 1:
    # 1. Rz (chemical potential, if any. mu=0 here, so empty)
    # 2. Short hopping
    # 3. Onsite U
    # 4. fSWAP
    
    def layer_hopping(frame_is_fswapped):
        # frame_is_fswapped = False:
        # even i: dn, up
        # odd i: up, dn
        # pairs (2i+1, 2i+2) connect up_i, up_{i+1} if i is even
        # and dn_i, dn_{i+1} if i is odd.
        # This is EXACTLY short hopping!
        layer = []
        for i in range(L-1):
            layer.append(('hop', 2*i+1, 2*i+2, -t_h * dt))
        return layer
        
    def layer_onsite():
        layer = []
        for i in range(L):
            layer.append(('U', 2*i, 2*i+1, U * dt))
        return layer
        
    def layer_fswap():
        layer = []
        for i in range(L):
            layer.append(('fswap', 2*i, 2*i+1))
        return layer

    # If it is mirrored:
    # Step 1: hop, U, fswap
    # Step 2: fswap, U, hop
    # But wait, if Step 1 ends with fswap, the spins are swapped!
    # If Step 2 starts with fswap, it swaps them back!
    # Then Step 2 does U and hop.
    # At the end of Step 2, the frame is back to normal!
    # Let's verify this.
    for k in range(steps):
        if k % 2 == 0:
            gates.extend(layer_hopping(False))
            gates.extend(layer_onsite())
            gates.extend(layer_fswap())
        else:
            gates.extend(layer_fswap())
            gates.extend(layer_onsite())
            gates.extend(layer_hopping(False))
            
    return gates

gates = build_circuit(4, 2, 0.2, -2.0, 1.0)
print(gates[:10])
