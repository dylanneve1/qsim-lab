# TDPBF16PS issue rate vs spacing (micro5.rs, one thread, register-only, 4 independent accumulators)
# columns: ns per TDPBF16PS with the products back to back, with 8 / 2 dependent register adds or 16 NOPs between them, and with the asm block unrolled x2.
# Clock ~3.2-3.3 GHz (dependent register add chain). First block on vCPU 4, second on vCPU 10. Load 18-21, under the bench lock.

 12:07:17 up 136 days, 13:16,  4 users,  load average: 18.04, 28.29, 30.67
rep 0: ns/dp: back-to-back 13.89 | +8 adds 8.32 | +2 adds 14.17 | +16 nops 7.21 | unrolled x2 14.11
rep 1: ns/dp: back-to-back 13.60 | +8 adds 7.83 | +2 adds 13.57 | +16 nops 7.35 | unrolled x2 13.85
rep 2: ns/dp: back-to-back 13.73 | +8 adds 7.94 | +2 adds 13.99 | +16 nops 7.29 | unrolled x2 14.09
rep 0: ns/dp: back-to-back 14.80 | +8 adds 7.91 | +2 adds 15.51 | +16 nops 7.26 | unrolled x2 14.72
rep 1: ns/dp: back-to-back 14.19 | +8 adds 7.87 | +2 adds 15.37 | +16 nops 7.41 | unrolled x2 14.42
rep 2: ns/dp: back-to-back 14.42 | +8 adds 8.01 | +2 adds 14.47 | +16 nops 7.49 | unrolled x2 14.55
 12:14:39 up 136 days, 13:23,  4 users,  load average: 21.03, 23.19, 27.45
