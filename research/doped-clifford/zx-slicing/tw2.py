"""Treewidth (min-fill upper bound, MMD+ lower bound) of the ZX spider graph for several reduction levels,
and of the raw quimb TN primal graph."""
import sys, pickle, numpy as np, networkx as nx, pyzx as zx
from networkx.algorithms.approximation import treewidth_min_fill_in, treewidth_min_degree
from circ import *
def mmd_plus(G):
    G = G.copy(); lb = 0
    while G.number_of_nodes() > 1:
        v = min(G.nodes, key=G.degree); dv = G.degree(v); lb = max(lb, dv)
        if dv == 0: G.remove_node(v); continue
        u = min(G.neighbors(v), key=lambda w: len(set(G.neighbors(w)) & set(G.neighbors(v))))
        G = nx.contracted_nodes(G, u, v, self_loops=False)
    return lb
def spider_graph(g):
    G = nx.Graph(); G.add_nodes_from(g.vertices())
    for e in g.edges():
        u, v = g.edge_st(e)
        if u != v: G.add_edge(u, v)
    return G
D = int(sys.argv[1]); n = 70; gates = load(D, n); x = np.random.default_rng(7).integers(0, 2, n)
for lvl in sys.argv[2].split(','):
    g, before = zx_graph(gates, n, x, 'none')
    zx.to_gh(g); zx.spider_simp(g); zx.id_simp(g); zx.spider_simp(g)
    if lvl == 'interior': zx.simplify.interior_clifford_simp(g)
    elif lvl == 'clifford': zx.clifford_simp(g)
    elif lvl == 'full': zx.full_reduce(g)
    G = spider_graph(g)
    ub = treewidth_min_fill_in(G)[0] if G.number_of_nodes() < 3000 else treewidth_min_degree(G)[0]
    print(f'D={D} zx[{lvl}]: V={G.number_of_nodes()} E={G.number_of_edges()} tw_ub={ub} tw_lb(MMD+)={mmd_plus(G)}', flush=True)
