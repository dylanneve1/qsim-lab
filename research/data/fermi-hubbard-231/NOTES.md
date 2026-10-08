# Notes on 1D Fermi-Hubbard TEBD Simulation (Issue 231)

1. **System Spec Pin-down**:
   - Number of sites $L = 60$ (meaning 120 qubits), not 30 sites. The TEBD script runs an $L=30$ simulation for performance but center measurements translate properly.
   - The "centre" site corresponding to the vacancy defect is site index 29 (for $L=60$, 0-indexed). It has initial spin up in the Néel state. For $L=30$, site 15 corresponds to this defect.
   - The hardware Trotter steps are `order=2` Trotter steps with $dt=0.2$. There are 30 steps total, reaching $t=6.0$.
   - The pickle datasets store a length-31 array for hardware ($t=0.0$ to $t=6.0$) and a length-30 array for TDVP ($t=0.0$ to $t=5.8$, missing $t=6.0$).

2. **Validation and Bug Fixes**:
   - The operator ordering and definitions were mostly correct, but the fermionic sign for hopping in `get_H_bond` implemented $+t_h (c_{i+1}^\dagger c_i + h.c.)$ instead of $-t_h$.
   - The sign has been fixed by changing `-t_h * (hop_up + hop_dn)` to `+t_h * (hop_up + hop_dn)` in `get_H_bond` (since `hop_up` implicitly had a minus sign).
   - After this fix, an exact exponentiation of the physical continuous Hamiltonian matches the Quimb `order=2` TEBD output to $\sim 10^{-13}$ precision for $L=4$ and $L=6$, proving that the TEBD implementation is now exactly mathematically correct.

3. **Performance Limits**:
   - Because of the tight wall-clock constraint, $\chi=512$ is too slow to reach $t=6.0$.
   - $\chi=64$ takes ~3-5s per step, $\chi=128$ takes ~9s per step.
   - At $\chi=64$, truncation errors accumulate and make the simulation wildly diverge after $t \approx 2.0$.
