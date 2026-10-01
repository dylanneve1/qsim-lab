//! A fast Union-Find decoder (Delfosse & Nickerson) for quantum error correction.
//!
//! The decoder operates on a decoding graph `G = (V, E)`. Vertices correspond to
//! detector events (syndrome defects). Edges represent physical error mechanisms
//! (space-like data errors or time-like measurement errors). A designated boundary
//! vertex absorbs defects at the open boundaries of the code.

use std::collections::VecDeque;

/// An edge in the decoding graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphEdge {
    pub u: usize,
    pub v: usize,
    /// Whether this error mechanism changes the logical observable.
    pub flips_logical: bool,
    /// Edge weight / length (usually 1 for nearest-neighbour spacetime lattice).
    pub weight: usize,
}

/// A decoding graph representing the error mechanisms of a QEC code.
#[derive(Clone, Debug)]
pub struct DecodingGraph {
    pub num_nodes: usize,
    pub boundary_node: usize,
    pub edges: Vec<GraphEdge>,
    pub adj: Vec<Vec<usize>>, // edge indices incident to each node
}

impl DecodingGraph {
    pub fn new(num_nodes: usize, boundary_node: usize) -> Self {
        Self {
            num_nodes,
            boundary_node,
            edges: Vec::new(),
            adj: vec![Vec::new(); num_nodes],
        }
    }

    pub fn add_edge(&mut self, u: usize, v: usize, flips_logical: bool, weight: usize) {
        assert!(u < self.num_nodes && v < self.num_nodes);
        let edge_idx = self.edges.len();
        self.edges.push(GraphEdge {
            u,
            v,
            flips_logical,
            weight,
        });
        self.adj[u].push(edge_idx);
        self.adj[v].push(edge_idx);
    }

    /// Minimum number of edges in a set that triggers no detector but flips
    /// the logical observable (the graph-like circuit distance): a shortest
    /// walk from the boundary back to the boundary with odd logical parity.
    /// Returns `None` if no such set exists.
    pub fn min_logical_weight(&self) -> Option<usize> {
        let n = self.num_nodes;
        let b = self.boundary_node;
        let mut dist = vec![usize::MAX; 2 * n];
        let mut q = VecDeque::new();
        dist[2 * b] = 0;
        q.push_back((b, 0usize));
        while let Some((u, par)) = q.pop_front() {
            let du = dist[2 * u + par];
            for &e in &self.adj[u] {
                let edge = &self.edges[e];
                let v = if edge.u == u { edge.v } else { edge.u };
                let np = par ^ edge.flips_logical as usize;
                if dist[2 * v + np] == usize::MAX {
                    dist[2 * v + np] = du + 1;
                    if v == b && np == 1 {
                        return Some(du + 1);
                    }
                    q.push_back((v, np));
                }
            }
        }
        None
    }
}

/// Disjoint Set Union (DSU) structure tracking parity and boundary connectivity.
struct Dsu {
    parent: Vec<usize>,
    defect_count: Vec<usize>,
    has_boundary: Vec<bool>,
}

impl Dsu {
    fn new(n: usize, boundary_node: usize, defects: &[usize]) -> Self {
        let mut defect_count = vec![0; n];
        for &d in defects {
            if d < n && d != boundary_node {
                defect_count[d] += 1;
            }
        }
        let mut has_boundary = vec![false; n];
        if boundary_node < n {
            has_boundary[boundary_node] = true;
        }
        Self {
            parent: (0..n).collect(),
            defect_count,
            has_boundary,
        }
    }

    fn find(&mut self, mut i: usize) -> usize {
        let mut root = i;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        while self.parent[i] != root {
            let next = self.parent[i];
            self.parent[i] = root;
            i = next;
        }
        root
    }

    fn union(&mut self, i: usize, j: usize) -> (usize, usize) {
        let root_i = self.find(i);
        let root_j = self.find(j);
        if root_i == root_j {
            return (root_i, root_j);
        }
        // Attach j to i
        self.parent[root_j] = root_i;
        self.defect_count[root_i] += self.defect_count[root_j];
        self.has_boundary[root_i] |= self.has_boundary[root_j];
        (root_i, root_j)
    }

    fn is_odd(&mut self, i: usize) -> bool {
        let root = self.find(i);
        !self.has_boundary[root] && (self.defect_count[root] % 2 != 0)
    }
}

/// A Union-Find decoder for standard QEC codes.
#[derive(Clone, Debug)]
pub struct UnionFindDecoder {
    pub graph: DecodingGraph,
}

impl UnionFindDecoder {
    pub fn new(graph: DecodingGraph) -> Self {
        Self { graph }
    }

    /// Decodes a set of defective syndrome nodes.
    ///
    /// Returns `true` if the predicted correction flips the logical observable.
    pub fn decode(&self, defects: &[usize]) -> bool {
        let n = self.graph.num_nodes;
        let b = self.graph.boundary_node;
        let num_edges = self.graph.edges.len();

        let mut dsu = Dsu::new(n, b, defects);

        // Track cluster members for each DSU root
        let mut cluster_members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

        // Edge growth: an edge is fully grown when total growth >= 2
        let mut edge_growth = vec![0usize; num_edges];
        let mut edge_merged = vec![false; num_edges];
        let mut sides = vec![0usize; num_edges];
        let mut frontier: Vec<usize> = Vec::new();

        // Spanning forest of merged tree edges: (neighbor, flips_logical)
        let mut spanning_forest_adj: Vec<Vec<(usize, bool)>> = vec![Vec::new(); n];

        // Edge growth phase: grow active (odd) clusters by half-edges
        loop {
            let mut active_roots = Vec::new();
            for i in 0..n {
                if dsu.parent[i] == i && dsu.is_odd(i) {
                    active_roots.push(i);
                }
            }

            if active_roots.is_empty() {
                break;
            }

            // Frontier: every unmerged edge leaving an active cluster, with the
            // number of active sides growing into it (1, or 2 when it joins two
            // different active clusters).
            frontier.clear();
            for &root in &active_roots {
                for &u in &cluster_members[root] {
                    for &e_idx in &self.graph.adj[u] {
                        if edge_merged[e_idx] {
                            continue;
                        }
                        let edge = self.edges(e_idx);
                        let v = if edge.u == u { edge.v } else { edge.u };
                        if dsu.find(v) != root {
                            if sides[e_idx] == 0 {
                                frontier.push(e_idx);
                            }
                            sides[e_idx] += 1;
                        }
                    }
                }
            }

            // Every active side grows each frontier edge by one half-unit per
            // step; an edge of weight w is fully grown at 2w. No cluster changes
            // until the first edge completes, so jump straight to that step
            // (identical to unit stepping; for weight-1 graphs this is the
            // classic half-edge growth).
            let mut delta = usize::MAX;
            for &e in &frontier {
                let need = 2 * self.graph.edges[e].weight.max(1) - edge_growth[e];
                delta = delta.min(need.div_ceil(sides[e]));
            }
            let mut edges_to_merge = Vec::new();
            for &e in &frontier {
                edge_growth[e] += delta * sides[e];
                sides[e] = 0;
                if edge_growth[e] >= 2 * self.graph.edges[e].weight.max(1) {
                    edge_merged[e] = true;
                    let edge = self.edges(e);
                    edges_to_merge.push((edge.u, edge.v, edge.flips_logical));
                }
            }
            if frontier.is_empty() {
                break; // odd cluster with no way out (cannot happen with a boundary)
            }

            for (u, v, flips_logical) in edges_to_merge {
                let root_u = dsu.find(u);
                let root_v = dsu.find(v);
                if root_u != root_v {
                    let (winner, loser) = dsu.union(root_u, root_v);
                    spanning_forest_adj[u].push((v, flips_logical));
                    spanning_forest_adj[v].push((u, flips_logical));

                    let mut loser_members = std::mem::take(&mut cluster_members[loser]);
                    cluster_members[winner].append(&mut loser_members);
                }
            }
        }

        // Peeling phase: peel the spanning forest to find the correction
        self.peel(&spanning_forest_adj, defects)
    }

    fn edges(&self, idx: usize) -> &GraphEdge {
        &self.graph.edges[idx]
    }

    fn peel(&self, tree_adj: &[Vec<(usize, bool)>], defects: &[usize]) -> bool {
        let n = self.graph.num_nodes;
        let b = self.graph.boundary_node;

        let mut node_defect = vec![false; n];
        for &d in defects {
            if d < n && d != b {
                node_defect[d] = !node_defect[d];
            }
        }

        let mut visited = vec![false; n];
        let mut logical_flip = false;

        // Peel each connected component in the forest.
        // Root components that touch the boundary at the boundary node,
        // so defects peel directly into the boundary sink.
        let mut starts = Vec::with_capacity(n);
        if b < n {
            starts.push(b);
        }
        for i in 0..n {
            if i != b {
                starts.push(i);
            }
        }

        for start in starts {
            if visited[start] || tree_adj[start].is_empty() {
                continue;
            }

            // Find BFS ordering from `start`
            let mut order = Vec::new();
            let mut parent: Vec<Option<(usize, bool)>> = vec![None; n];
            let mut q = VecDeque::new();

            visited[start] = true;
            q.push_back(start);

            while let Some(u) = q.pop_front() {
                order.push(u);
                for &(v, flips) in &tree_adj[u] {
                    if !visited[v] {
                        visited[v] = true;
                        parent[v] = Some((u, flips));
                        q.push_back(v);
                    }
                }
            }

            // Peel from leaves up to root in reverse BFS order
            for &u in order.iter().rev() {
                if let Some((p, edge_flips_logical)) = parent[u] {
                    if node_defect[u] {
                        // Flip edge (u, p)
                        node_defect[u] = false;
                        node_defect[p] = !node_defect[p];
                        if edge_flips_logical {
                            logical_flip = !logical_flip;
                        }
                    }
                }
            }
        }

        logical_flip
    }
}
