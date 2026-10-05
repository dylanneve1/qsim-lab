/* Independent exact check that a CSS code has no nontrivial logical of weight <= W.
 *
 * Written from scratch in C for code-discovery-2 (no code shared with qsim_lab or
 * verify_code.py). Input (text, whitespace separated):
 *
 *   n  rc  ro                      qubits, rows of the checking matrix, rows of the other matrix
 *   rc lines: w q1 .. qw           supports of the checks (e.g. H_X for Z-type logicals)
 *   ro lines: w q1 .. qw           supports of the other type (e.g. H_Z), whose row space is "trivial"
 *
 * usage: mwlogical W < code.txt
 * Prints "none <= W" (and the node count) if every e with H e = 0, 0 < |e| <= W lies in
 * rowspace(other); otherwise prints the first nontrivial logical found.
 *
 * Method: a minimum-weight nontrivial logical E has no proper non-empty subset in ker H
 * (such a subset or its complement in E would be a lighter nontrivial logical, or a
 * stabilizer), so from any of its qubits it is reached by repeatedly adding a qubit of
 * an unsatisfied check. Every qubit is tried as a root, in order, with all earlier roots
 * banned; inside a root the branches of one check ban their earlier siblings. Pruning:
 * |S| + ceil(#unsatisfied / max column weight) > W. No symmetry is used.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 640
#define QW ((MAXN + 63) / 64)
#define MAXC 320
#define CW ((MAXC + 63) / 64)

static int n, rc, ro, W, maxdeg;
static int csup[MAXC][16], cdeg[MAXC];
static int qchk[MAXN][16], qdeg[MAXN];
static uint64_t basis[MAXN][QW]; /* reduced rows of rowspace(other), basis[pivot] */
static int has_pivot[MAXN];
static int banned[MAXN], used[MAXN], chosen[64], nch;
static long long nodes;

static int top_bit(const uint64_t *v) {
    for (int w = QW - 1; w >= 0; w--)
        if (v[w]) return w * 64 + 63 - __builtin_clzll(v[w]);
    return -1;
}

static void add_basis(uint64_t *v) {
    int h;
    while ((h = top_bit(v)) >= 0) {
        if (!has_pivot[h]) {
            memcpy(basis[h], v, sizeof(uint64_t) * QW);
            has_pivot[h] = 1;
            return;
        }
        for (int w = 0; w < QW; w++) v[w] ^= basis[h][w];
    }
}

static int trivial(const uint64_t *e) {
    uint64_t v[QW];
    memcpy(v, e, sizeof v);
    int h;
    while ((h = top_bit(v)) >= 0) {
        if (!has_pivot[h]) return 0;
        for (int w = 0; w < QW; w++) v[w] ^= basis[h][w];
    }
    return 1;
}

static int dfs(uint64_t *syn, uint64_t *vec) {
    nodes++;
    int unsat = 0;
    for (int w = 0; w < CW; w++) unsat += __builtin_popcountll(syn[w]);
    if (unsat == 0) return !trivial(vec); /* zero syndrome: found iff nontrivial */
    if (nch + (unsat + maxdeg - 1) / maxdeg > W) return 0;
    int best = -1, bestn = 1 << 30;
    for (int w = 0; w < CW && bestn > 1; w++) {
        uint64_t x = syn[w];
        while (x) {
            int c = w * 64 + __builtin_ctzll(x);
            x &= x - 1;
            int f = 0;
            for (int i = 0; i < cdeg[c]; i++) {
                int q = csup[c][i];
                if (!banned[q] && !used[q]) f++;
            }
            if (f < bestn) {
                bestn = f;
                best = c;
                if (f <= 1) break;
            }
        }
    }
    if (bestn == 0) return 0;
    int newly[16], nn = 0, hit = 0;
    for (int i = 0; i < cdeg[best]; i++) {
        int q = csup[best][i];
        if (banned[q] || used[q]) continue;
        used[q] = 1;
        chosen[nch++] = q;
        for (int j = 0; j < qdeg[q]; j++) syn[qchk[q][j] / 64] ^= 1ULL << (qchk[q][j] % 64);
        vec[q / 64] ^= 1ULL << (q % 64);
        hit = dfs(syn, vec);
        if (hit) return 1; /* keep `chosen` for printing */
        vec[q / 64] ^= 1ULL << (q % 64);
        for (int j = 0; j < qdeg[q]; j++) syn[qchk[q][j] / 64] ^= 1ULL << (qchk[q][j] % 64);
        nch--;
        used[q] = 0;
        banned[q] = 1;
        newly[nn++] = q;
    }
    for (int i = 0; i < nn; i++) banned[newly[i]] = 0;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: mwlogical W < code.txt\n");
        return 2;
    }
    W = atoi(argv[1]);
    if (scanf("%d %d %d", &n, &rc, &ro) != 3 || n > MAXN || rc > MAXC || W >= 64) return 2;
    for (int c = 0; c < rc; c++) {
        int w;
        if (scanf("%d", &w) != 1 || w > 16) return 2;
        for (int i = 0; i < w; i++) {
            int q;
            if (scanf("%d", &q) != 1 || q < 0 || q >= n) return 2;
            /* a qubit listed twice cancels */
            int dup = -1;
            for (int j = 0; j < cdeg[c]; j++)
                if (csup[c][j] == q) dup = j;
            if (dup >= 0) csup[c][dup] = csup[c][--cdeg[c]];
            else csup[c][cdeg[c]++] = q;
        }
        for (int i = 0; i < cdeg[c]; i++) {
            int q = csup[c][i];
            if (qdeg[q] >= 16) return 3;
            qchk[q][qdeg[q]++] = c;
        }
    }
    for (int r = 0; r < ro; r++) {
        int w;
        uint64_t v[QW];
        memset(v, 0, sizeof v);
        if (scanf("%d", &w) != 1) return 2;
        for (int i = 0; i < w; i++) {
            int q;
            if (scanf("%d", &q) != 1) return 2;
            v[q / 64] ^= 1ULL << (q % 64);
        }
        add_basis(v);
    }
    maxdeg = 1;
    for (int q = 0; q < n; q++)
        if (qdeg[q] > maxdeg) maxdeg = qdeg[q];
    for (int root = 0; root < n; root++) {
        uint64_t syn[CW], vec[QW];
        memset(syn, 0, sizeof syn);
        memset(vec, 0, sizeof vec);
        for (int q = 0; q < n; q++) banned[q] = q < root, used[q] = 0;
        used[root] = 1;
        nch = 0;
        chosen[nch++] = root;
        for (int j = 0; j < qdeg[root]; j++) syn[qchk[root][j] / 64] ^= 1ULL << (qchk[root][j] % 64);
        vec[root / 64] ^= 1ULL << (root % 64);
        if (W >= 1 && dfs(syn, vec)) {
            printf("found weight %d:", nch);
            for (int i = 0; i < nch; i++) printf(" %d", chosen[i]);
            printf("\nnodes %lld\n", nodes);
            return 1;
        }
    }
    printf("none <= %d\nnodes %lld\n", W, nodes);
    return 0;
}
