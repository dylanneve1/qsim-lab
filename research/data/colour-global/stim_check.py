#!/usr/bin/env python3
"""Independent upper bound on circuit distance with Stim's search_for_undetectable_logical_errors
(heuristic search: it finds *a* small undetectable logical; it cannot prove a lower bound).
usage: stim_check.py file.stim [max_event_set] [max_edge_degree]"""
import sys, time, stim
c = stim.Circuit.from_file(sys.argv[1])
ev = int(sys.argv[2]) if len(sys.argv) > 2 else 6
deg = int(sys.argv[3]) if len(sys.argv) > 3 else 9

t = time.time()
err = c.search_for_undetectable_logical_errors(
    dont_explore_detection_event_sets_with_size_above=ev,
    dont_explore_edges_with_degree_above=deg,
    dont_explore_edges_increasing_symptom_degree=False,
    canonicalize_circuit_errors=True)
print(sys.argv[1], "stim upper bound:", len(err), f"({time.time()-t:.1f}s, ev<={ev}, deg<={deg})", flush=True)
