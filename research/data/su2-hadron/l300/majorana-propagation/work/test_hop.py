def sign_swap(old, new, S):
    S_rest = [s for s in S if s != old]
    i = sum(1 for s in S if s < old)
    j = sum(1 for s in S_rest if s < new)
    return 1 if abs(i - j) % 2 == 0 else -1

print(sign_swap(1, 4, [1, 2, 3])) # Expected 1
print(sign_swap(1, 4, [1, 3]))    # Expected -1
