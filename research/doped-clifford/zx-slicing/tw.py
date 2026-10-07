"""Treewidth upper bounds of the reduced ZX spider graph (primal graph of the hyperindex network)."""
import sys, pickle, networkx as nx
from networkx.algorithms.approximation import treewidth_min_degree, treewidth_min_fill_in
for D in map(int, sys.argv[1:]):
    d = pickle.load(open(f'/tmp/doped-zx/net_D{D}.pkl', 'rb'))['zx']
    G = nx.Graph(); G.add_nodes_from({i for t in d['inputs'] for i in t})
    for t in d['inputs']:
        for a in t:
            for b in t:
                if a < b: G.add_edge(a, b)
    w1, _ = treewidth_min_degree(G); w2, _ = treewidth_min_fill_in(G)
    print(f'D={D} zx primal graph: V={G.number_of_nodes()} E={G.number_of_edges()} density={2*G.number_of_edges()/G.number_of_nodes()/(G.number_of_nodes()-1):.2f} tw_mindeg<={w1} tw_minfill<={w2}', flush=True)
