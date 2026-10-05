Mac (M1 Pro, one thread, RAYON_NUM_THREADS=1, bench lock). Single shots. research/simulability/planner-v2.md.
- req.jsonl: read-out session (`planner_v2 req ENGINE`), 1770 (instance, engine) rows; evolve + every read-out
  timed separately. Load 6.7-12 (chunks 1-2, another agent's LER campaign running), 4.1-12.2 (chunks 3-4).
  Pairs the round-4 sweep measured as censored or > 4 s are `skipped` (counted as censored).
- hsfamp.jsonl: HSF amplitude-only runs for the 81 instances whose HSF full output was skipped (8 s timeout).
- feat_mac.jsonl: planner features, per-tier timings, v1 / v2 / rule choices (commit 612ceb4 constants).
- e2e.jsonl: first end-to-end session (v1:e, v2nc:*, rule:*), load 15-49: compared against req.jsonl, confounded.
- e2e2.jsonl, e2e3.jsonl, e2e4.jsonl: end to end with the in-session oracle (force-ENGINE), per gen_e2e_jobs.py:
  e2e2 = v2 at b560df4^ (load 4-21), e2e3 = b560df4 (load 6-8), e2e4 = 657dfeb, final (load 15-35).
- auto3/: Strategy::Auto with exploration, first re-time (before the frame-store change).
- autoab/: Auto A/B (auto vs auto0 = no exploration), same binary, load 17-49.
