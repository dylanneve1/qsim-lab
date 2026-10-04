Mac (M1 Pro, one thread, bench lock, driver.py CSV format). Single shots.
- retime_skirental_*.csv: frame, dense, auto re-time with the rejected ski-rental Auto (frame/dense are the control for auto2).
- auto2_*.csv: auto with the shipped flat-evidence Auto.
- plan_*.csv: planx/planp/plan/mpsb with the first (unstaged) planner, 314 dataset instances.
- plan2_* / plan3_* / plan4_*.csv: planner after staging (2), the HSF-fits rule (3), and the faster replay (4, final).
- new_hea.csv, new_qft.csv: held-out families, every engine incl. the unstaged planner.
- new2/3/4_*.csv: planner-only re-runs on the held-out families, matching plan2/3/4.
