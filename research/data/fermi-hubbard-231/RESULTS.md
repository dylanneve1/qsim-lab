# TEBD Simulation Results vs Hardware Data

## Observables at t=1.0 (Step 5)
| Source | $\langle n_{c, \uparrow} \rangle$ | $\langle n_{c, \downarrow} \rangle$ | $\langle n_{c, \uparrow} n_{c, \downarrow} \rangle$ |
|---|---|---|---|
| TEBD ($\chi=64$) | 0.389618 | 0.610411 | 0.178236 |
| TEBD ($\chi=128$) | 0.389569 | 0.610431 | 0.178144 |
| ITensor TDVP ($L=60$) | 0.388595 | 0.611405 | 0.177560 |
| Quantum Hardware ($L=60$) | 0.385336 | 0.608219 | 0.179281 |

## Error Bars at t=1.0 (TEBD $\chi=128$)
Includes $\chi$-convergence spread + Trotter error vs TDVP:
- $\langle n_{c, \uparrow} \rangle$: $\pm 1.02e-03$
- $\langle n_{c, \downarrow} \rangle$: $\pm 9.93e-04$
- $\langle n_{c, \uparrow} n_{c, \downarrow} \rangle$: $\pm 6.76e-04$

## Convergence Analysis
At $\chi=64$, the TEBD simulation converges reasonably well up to $t \approx 2.0$. Beyond that (e.g. $t=3.0$), $\chi=64$ completely diverges due to entanglement growth and truncation errors (e.g. at $t=3.0$, $\chi=64$ gives $0.098$ for the double occupancy whereas the reference is $0.192$).

The computational cost grows rapidly with $\chi$. Using 2 threads:
- $\chi=64$: ~3-5 seconds per step.
- $\chi=128$: ~9 seconds per step.
- $\chi=256$: Estimated ~30-60 seconds per step.
- $\chi=512$: Estimated ~2-5 minutes per step.

We can produce a classical value with a smaller error than the paper's TDVP only at early times (up to $t \approx 1.5$) using $\chi \le 128$. For $t \ge 3$, reaching classical convergence with TEBD would require $\chi > 512$, which becomes prohibitively slow on this shared architecture.
