import numpy as np
import gauss

def sign_n(x, C, A):
    ans = A.bit_count() + (C >> (x + 1)).bit_count() + (A >> (x + 1)).bit_count()
    return 1.0 if ans % 2 == 0 else -1.0

def sign_swap(old, new, S):
    S_rest = S & ~(1 << old)
    i = (S & ((1 << old) - 1)).bit_count()
    j = (S_rest & ((1 << new) - 1)).bit_count()
    return 1.0 if abs(i - j) % 2 == 0 else -1.0

def step_hop(C, A, x, y, V):
    # V is 2x2. Returns list of (new_C, new_A, coeff)
    # Actually we do C and A independently!
    
    st_c = ((C >> x) & 1) | (((C >> y) & 1) << 1)
    res_c = []
    if st_c == 0:
        res_c.append((C, 1.0))
    elif st_c == 1:
        res_c.append((C, V[0,0]))
        C_new = (C & ~(1 << x)) | (1 << y)
        res_c.append((C_new, V[0,1] * sign_swap(x, y, C)))
    elif st_c == 2:
        C_new = (C & ~(1 << y)) | (1 << x)
        res_c.append((C_new, V[1,0] * sign_swap(y, x, C)))
        res_c.append((C, V[1,1]))
    else:
        detV = V[0,0]*V[1,1] - V[0,1]*V[1,0]
        res_c.append((C, detV))
        
    st_a = ((A >> x) & 1) | (((A >> y) & 1) << 1)
    res_a = []
    Vc = np.conj(V)
    if st_a == 0:
        res_a.append((A, 1.0))
    elif st_a == 1:
        res_a.append((A, Vc[0,0]))
        A_new = (A & ~(1 << x)) | (1 << y)
        res_a.append((A_new, Vc[0,1] * sign_swap(x, y, A)))
    elif st_a == 2:
        A_new = (A & ~(1 << y)) | (1 << x)
        res_a.append((A_new, Vc[1,0] * sign_swap(y, x, A)))
        res_a.append((A, Vc[1,1]))
    else:
        detVc = Vc[0,0]*Vc[1,1] - Vc[0,1]*Vc[1,0]
        res_a.append((A, detVc))
        
    out = []
    for c_new, c_coeff in res_c:
        if abs(c_coeff) < 1e-14: continue
        for a_new, a_coeff in res_a:
            if abs(a_coeff) < 1e-14: continue
            out.append((c_new, a_new, c_coeff * a_coeff))
    return out


def step_interact(C0, A0, C1, A1, a, b, g, a0_phase, a1_phase):
    # phases:
    # the vertex is U = exp(i a0 n_a) exp(i a1 n_b) exp(i g n_a n_b)
    # So U^\dag O U applies these phases.
    # Single site phase: U^\dag c^\dag U = c^\dag e^{i phi}
    
    # First, let's just do the single site phases mathematically:
    # If a in C0, coeff *= exp(i a0_phase). If a in A0, coeff *= exp(-i a0_phase)
    # same for b.
    ph = 1.0
    if (C0 >> a) & 1: ph *= np.exp(1j * a0_phase)
    if (A0 >> a) & 1: ph *= np.exp(-1j * a0_phase)
    if (C1 >> b) & 1: ph *= np.exp(1j * a1_phase)
    if (A1 >> b) & 1: ph *= np.exp(-1j * a1_phase)
    
    # Now interaction exp(i g n_a n_b)
    st_a = (((C0 >> a) & 1) << 1) | ((A0 >> a) & 1)
    st_b = (((C1 >> b) & 1) << 1) | ((A1 >> b) & 1)
    
    # states: 0=I, 1=A, 2=C, 3=N
    exp_ig = np.exp(1j * g)
    exp_mig = np.exp(-1j * g)
    Z = exp_ig - 1.0
    Z_star = exp_mig - 1.0
    
    # Return list of (C0_new, A0_new, C1_new, A1_new, coeff)
    out = []
    
    if st_a == 0:
        if st_b == 0: out.append((C0, A0, C1, A1, 1.0))
        elif st_b == 3: out.append((C0, A0, C1, A1, 1.0))
        elif st_b == 2:
            out.append((C0, A0, C1, A1, 1.0))
            out.append((C0 | (1<<a), A0 | (1<<a), C1, A1, Z_star * sign_n(a, C0, A0)))
        elif st_b == 1:
            out.append((C0, A0, C1, A1, 1.0))
            out.append((C0 | (1<<a), A0 | (1<<a), C1, A1, Z * sign_n(a, C0, A0)))
            
    elif st_a == 3:
        if st_b == 0: out.append((C0, A0, C1, A1, 1.0))
        elif st_b == 3: out.append((C0, A0, C1, A1, 1.0))
        elif st_b == 2: out.append((C0, A0, C1, A1, exp_mig))
        elif st_b == 1: out.append((C0, A0, C1, A1, exp_ig))
        
    elif st_a == 2:
        if st_b == 0:
            out.append((C0, A0, C1, A1, 1.0))
            out.append((C0, A0, C1 | (1<<b), A1 | (1<<b), Z_star * sign_n(b, C1, A1)))
        elif st_b == 3: out.append((C0, A0, C1, A1, exp_mig))
        elif st_b == 2: out.append((C0, A0, C1, A1, exp_mig))
        elif st_b == 1: out.append((C0, A0, C1, A1, 1.0))
            
    elif st_a == 1:
        if st_b == 0:
            out.append((C0, A0, C1, A1, 1.0))
            out.append((C0, A0, C1 | (1<<b), A1 | (1<<b), Z * sign_n(b, C1, A1)))
        elif st_b == 3: out.append((C0, A0, C1, A1, exp_ig))
        elif st_b == 2: out.append((C0, A0, C1, A1, 1.0))
        elif st_b == 1: out.append((C0, A0, C1, A1, exp_ig))
        
    return [(c0, a0, c1, a1, c * ph) for (c0, a0, c1, a1, c) in out]


def expectation(C0, A0, C1, A1, occ0_mask, occ1_mask):
    if C0 != A0 or C1 != A1: return 0.0
    if (C0 & occ0_mask) != C0: return 0.0
    if (C1 & occ1_mask) != C1: return 0.0
    k0 = C0.bit_count()
    k1 = C1.bit_count()
    sign = 1
    if (k0 * (k0 - 1) // 2) % 2 != 0: sign *= -1
    if (k1 * (k1 - 1) // 2) % 2 != 0: sign *= -1
    return sign


def run_sim(circ, max_steps=20, epsilon=1e-12, k_max=6):
    import time
    from collections import defaultdict
    import gauss
    D = '/tmp/su2-254/circuits/'
    occ, m, out = gauss.blocks(D + f'x_100_{circ}.qasm')
    
    # Identify which site is on which chain
    occ0_mask = 0
    occ1_mask = 0
    for w in range(120):
        s, l = m[w]
        if occ[w]:
            if l == 0: occ0_mask |= (1 << s)
            else: occ1_mask |= (1 << s)
            
    # Reverse the gates since Heisenberg picture goes backwards
    # Wait, out has steps.
    # A step is a list of blocks? No, out is a flat list of blocks, with {'step': i} markers.
    
    # We want observables: stag_SCV = sum_r (-1)^r (n_{i,r} + n_{o,r})
    # Since Heisenberg is linear, we can just track the evolution of each n_{l,r} independently
    # and sum their expectation values at the end!
    # Even better, we can evolve the SUM of operators if we just put them all in one dict?
    # No, they have different starting positions, but yes, it's just a sum of terms!
    # Wait, the number of terms might grow. Is it better to evolve stag all at once?
    # Yes! A sum of terms is just a dictionary.
    
    terms = defaultdict(complex)
    for r in range(60):
        c = 1.0 if r % 2 == 0 else -1.0
        terms[(1<<r, 1<<r, 0, 0)] += c
        terms[(0, 0, 1<<r, 1<<r)] += c
        
    print(f"Initial terms: {len(terms)}")
    
    # To propagate backwards, we need to apply the blocks in REVERSE order.
    # Wait, does `out` contain ALL 120 steps?
    # We only care about step 20 to step 1.
    # The observable is at `step` = T. We propagate it back to `step` = 0.
    # Let's collect blocks per step.
    steps = []
    curr = []
    for b in out:
        if 'step' in b:
            steps.append(curr)
            curr = []
        else:
            curr.append(b)
            
    print(f"Total steps found: {len(steps)}")
    
    # We want expectations at step 1..20
    # To get expectation at step T, we take the observable, and apply gates from step T back to 1.
    # So we should start with O, then for t in T, T-1, ... 1:
    #   O = U_t^\dag O U_t
    # Then evaluate.
    # Alternatively, we can start with O, and apply gates backward step by step,
    # and AT EACH STEP t (which corresponds to pushing the observable back to t-1),
    # the expectation of this back-propagated operator on the INITIAL state gives the expectation
    # of the observable at step T - t.
    # Wait, no. If we want observable at step 1, 2, ..., 20.
    # The observable at step 1 is O(1) = U_1^\dag O U_1.
    # Observable at step 2 is O(2) = U_1^\dag U_2^\dag O U_2 U_1.
    # So if we start with O, and apply U_20^\dag O U_20, we get O_1.
    # Then U_19^\dag O_1 U_19 -> O_2...
    # This gives O(20) at the end.
    pass


def propagate_back(terms, blocks_seq, m, epsilon, k_max):
    # blocks_seq is in REVERSE chronological order
    for b in blocks_seq:
        w = b['w']
        U = b['U']
        sl = [m[x] for x in w]
        if len(w) == 1:
            s, l = sl[0]
            ph = U[1,1]/U[0,0]
            new_terms = {}
            for (C0, A0, C1, A1), coef in terms.items():
                if l == 0:
                    c = coef
                    if (C0 >> s) & 1: c *= ph
                    if (A0 >> s) & 1: c *= np.conj(ph)
                else:
                    c = coef
                    if (C1 >> s) & 1: c *= ph
                    if (A1 >> s) & 1: c *= np.conj(ph)
                new_terms[(C0, A0, C1, A1)] = c
            terms = new_terms
        else:
            (s1, l1), (s2, l2) = sl
            if l1 == l2:
                # hop
                V = U[1:3, 1:3]/U[0,0]
                Vsp = np.array([[V[1,1], V[1,0]], [V[0,1], V[0,0]]])
                new_terms = defaultdict(complex)
                for (C0, A0, C1, A1), coef in terms.items():
                    if l1 == 0:
                        res = step_hop(C0, A0, s1, s2, Vsp)
                        for nc, na, c in res:
                            new_terms[(nc, na, C1, A1)] += coef * c
                    else:
                        res = step_hop(C1, A1, s1, s2, Vsp)
                        for nc, na, c in res:
                            new_terms[(C0, A0, nc, na)] += coef * c
                terms = new_terms
            else:
                # interaction
                ph = np.angle(np.diag(U))
                a0 = ph[2] - ph[0]
                a1 = ph[1] - ph[0]
                g = ph[3] - ph[2] - ph[1] + ph[0]
                g = (g + np.pi) % (2 * np.pi) - np.pi
                
                new_terms = defaultdict(complex)
                for (C0, A0, C1, A1), coef in terms.items():
                    res = step_interact(C0, A0, C1, A1, s1, s2, g, a0, a1)
                    for nc0, na0, nc1, na1, c in res:
                        new_terms[(nc0, na0, nc1, na1)] += coef * c
                terms = new_terms
                
        # truncate
        terms = {k: v for k, v in terms.items() if abs(v) > epsilon}
        # degree truncation:
        if k_max is not None:
            terms = {k: v for k, v in terms.items() if k[0].bit_count() + k[2].bit_count() <= k_max}
            
    return terms

def evaluate(terms, occ0_mask, occ1_mask):
    ans = 0.0
    for (C0, A0, C1, A1), coef in terms.items():
        ans += coef * expectation(C0, A0, C1, A1, occ0_mask, occ1_mask)
    return ans.real

def get_stag_initial():
    terms = defaultdict(complex)
    for r in range(60):
        c = 1.0 if r % 2 == 0 else -1.0
        terms[(1<<r, 1<<r, 0, 0)] += c
        terms[(0, 0, 1<<r, 1<<r)] += c
    return terms

def run_sim(circ, max_steps=20, epsilon=1e-12, k_max=6):
    import time
    import gauss
    D = '/tmp/su2-254/circuits/'
    occ, m, out = gauss.blocks(D + f'x_100_{circ}.qasm')
    
    occ0_mask = 0
    occ1_mask = 0
    for w in range(120):
        s, l = m[w]
        if occ[w]:
            if l == 0: occ0_mask |= (1 << s)
            else: occ1_mask |= (1 << s)
            
    steps = []
    curr = []
    for b in out:
        if 'step' in b:
            if curr:
                steps.append(curr)
            curr = []
        else:
            curr.append(b)
            
    print(f"[{circ}] Found {len(steps)} steps")
    
    results = []
    for t in range(1, min(max_steps, len(steps)) + 1):
        t0 = time.time()
        terms = get_stag_initial()
        
        # We want observable at step t, so we apply gates from step t down to 1 in reverse
        # The blocks for step t is steps[t-1].
        # We need to reverse the order of blocks WITHIN each step?
        # Yes, U_step = U_1 U_2 ... U_k (where U_k is the last block in the step).
        # So U_step^\dag O U_step = U_k^\dag ... U_1^\dag O U_1 ... U_k.
        # This means we process blocks in REVERSE order of how they appear in `out`.
        blocks_seq = []
        for i in range(t-1, -1, -1):
            blocks_seq.extend(reversed(steps[i]))
            
        final_terms = propagate_back(terms, blocks_seq, m, epsilon, k_max)
        val = evaluate(final_terms, occ0_mask, occ1_mask)
        t1 = time.time()
        print(f"Step {t:2d}: {val:10.5f} | terms: {len(final_terms):6d} | time: {t1-t0:.2f}s")
        results.append(val)
        
    return results

if __name__ == '__main__':
    from collections import defaultdict
    run_sim('SCV', max_steps=4, epsilon=1e-8, k_max=4)
