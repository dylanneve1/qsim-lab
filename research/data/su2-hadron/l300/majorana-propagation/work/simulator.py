import numpy as np

def sign_n(x, C, A):
    ans = len(A) + sum(1 for c in C if c > x) + sum(1 for a in A if a > x)
    return 1 if ans % 2 == 0 else -1

# Let's test the free hop as well!
