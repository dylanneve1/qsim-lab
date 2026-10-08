#include <stdint.h>
#include <stdlib.h>
#include <math.h>
typedef struct { uint64_t mask; double r; double i; } HopBranch;
typedef struct { int n0; int n1; double r; double i; } ResBranch;
#include <string.h>
#include <stdio.h>

#define HASH_SIZE 16777215  // 2^24 - 1
#define MAX_NEW_TERMS 5000000

typedef struct {
    uint64_t c0, a0, c1, a1;
    double re, im;
} Term;

typedef struct {
    uint64_t c0, a0, c1, a1;
    double re, im;
    uint32_t next;
} HashEntry;

HashEntry* hash_table;
uint32_t* hash_heads;
uint32_t hash_count;

Term* terms;
int num_terms;

void init() {
    hash_table = (HashEntry*)malloc(sizeof(HashEntry) * MAX_NEW_TERMS);
    hash_heads = (uint32_t*)calloc(HASH_SIZE + 1, sizeof(uint32_t));
    terms = (Term*)malloc(sizeof(Term) * MAX_NEW_TERMS);
    num_terms = 0;
    hash_count = 0;
}

static inline uint32_t hash_func(uint64_t c0, uint64_t a0, uint64_t c1, uint64_t a1) {
    uint64_t h = c0 ^ (a0 * 11) ^ (c1 * 131) ^ (a1 * 1313);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccd;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53;
    h ^= h >> 33;
    return h & HASH_SIZE;
}
void clear_hash() {
    for (uint32_t i = 0; i < hash_count; i++) {
        uint32_t idx = hash_table[i].next;
        // actually we just clear the heads we used
        // wait, better to just memset the whole heads array if it's small, or just traverse
    }
    for (uint32_t i = 0; i < hash_count; i++) { uint32_t h = hash_func(hash_table[i].c0, hash_table[i].a0, hash_table[i].c1, hash_table[i].a1); hash_heads[h] = 0; }
    hash_count = 0;
}


void add_term(uint64_t c0, uint64_t a0, uint64_t c1, uint64_t a1, double re, double im) {
    if (fabs(re) < 1e-15 && fabs(im) < 1e-15) return;
    uint32_t h = hash_func(c0, a0, c1, a1);
    uint32_t head = hash_heads[h];
    while (head != 0) {
        HashEntry* e = &hash_table[head - 1];
        if (e->c0 == c0 && e->a0 == a0 && e->c1 == c1 && e->a1 == a1) {
            e->re += re;
            e->im += im;
            return;
        }
        head = e->next;
    }
    if (hash_count >= MAX_NEW_TERMS) {
        printf("Error: hash_count exceeded %d\n", MAX_NEW_TERMS);
        return;
    }
    HashEntry* e = &hash_table[hash_count];
    e->c0 = c0; e->a0 = a0; e->c1 = c1; e->a1 = a1;
    e->re = re; e->im = im;
    e->next = hash_heads[h];
    hash_heads[h] = hash_count + 1;
    hash_count++;
}

void finish_step(double eps, int k_max) {
    num_terms = 0;
    for (uint32_t i = 0; i < hash_count; i++) {
        HashEntry* e = &hash_table[i];
        if (fabs(e->re) > eps || fabs(e->im) > eps) {
            int k = __builtin_popcountll(e->c0) + __builtin_popcountll(e->c1);
            if (k <= k_max) {
                terms[num_terms].c0 = e->c0;
                terms[num_terms].a0 = e->a0;
                terms[num_terms].c1 = e->c1;
                terms[num_terms].a1 = e->a1;
                terms[num_terms].re = e->re;
                terms[num_terms].im = e->im;
                num_terms++;
            }
        }
    }
    clear_hash();
}

static inline double sign_n(int x, uint64_t C, uint64_t A) {
    int count = __builtin_popcountll(A) + __builtin_popcountll(C >> (x + 1)) + __builtin_popcountll(A >> (x + 1));
    return (count % 2 == 0) ? 1.0 : -1.0;
}

static inline double sign_swap(int old_pos, int new_pos, uint64_t S) {
    uint64_t S_rest = S & ~(1ULL << old_pos);
    int i = __builtin_popcountll(S & ((1ULL << old_pos) - 1));
    int j = __builtin_popcountll(S_rest & ((1ULL << new_pos) - 1));
    return (abs(i - j) % 2 == 0) ? 1.0 : -1.0;
}

void apply_phase(int s, int l, double ph_re, double ph_im) {
    for (int i = 0; i < num_terms; i++) {
        uint64_t C = (l == 0) ? terms[i].c0 : terms[i].c1;
        uint64_t A = (l == 0) ? terms[i].a0 : terms[i].a1;
        double re = terms[i].re, im = terms[i].im;
        double mul_re = 1.0, mul_im = 0.0;
        
        if ((C >> s) & 1) {
            double nr = mul_re * ph_re - mul_im * ph_im;
            double ni = mul_re * ph_im + mul_im * ph_re;
            mul_re = nr; mul_im = ni;
        }
        if ((A >> s) & 1) {
            double nr = mul_re * ph_re + mul_im * ph_im;
            double ni = mul_re * (-ph_im) + mul_im * ph_re;
            mul_re = nr; mul_im = ni;
        }
        
        double nre = re * mul_re - im * mul_im;
        double nim = re * mul_im + im * mul_re;
        add_term(terms[i].c0, terms[i].a0, terms[i].c1, terms[i].a1, nre, nim);
    }
    finish_step(0.0, 999);
}

void apply_hop(int s1, int s2, int l, double v00r, double v00i, double v01r, double v01i, double v10r, double v10i, double v11r, double v11i) {
    double detr = v00r*v11r - v00i*v11i - (v01r*v10r - v01i*v10i);
    double deti = v00r*v11i + v00i*v11r - (v01r*v10i + v01i*v10r);
    
    for (int i = 0; i < num_terms; i++) {
        uint64_t C = (l == 0) ? terms[i].c0 : terms[i].c1;
        uint64_t A = (l == 0) ? terms[i].a0 : terms[i].a1;
        
        int st_c = (((C >> s1) & 1) | (((C >> s2) & 1) << 1));
        int st_a = (((A >> s1) & 1) | (((A >> s2) & 1) << 1));
        
        // 4 branches for C, 4 for A
        HopBranch bc[2], ba[2];
        int nc = 0, na = 0;
        
        if (st_c == 0) { bc[nc++] = (HopBranch){ C, 1.0, 0.0 }; }
        else if (st_c == 1) {
            bc[nc++] = (HopBranch){ C, v00r, v00i };
            double sgn = sign_swap(s1, s2, C);
            bc[nc++] = (HopBranch){ (C & ~(1ULL<<s1)) | (1ULL<<s2), v01r * sgn, v01i * sgn };
        } else if (st_c == 2) {
            double sgn = sign_swap(s2, s1, C);
            bc[nc++] = (HopBranch){ (C & ~(1ULL<<s2)) | (1ULL<<s1), v10r * sgn, v10i * sgn };
            bc[nc++] = (HopBranch){ C, v11r, v11i };
        } else {
            bc[nc++] = (HopBranch){ C, detr, deti };
        }
        
        if (st_a == 0) { ba[na++] = (HopBranch){ A, 1.0, 0.0 }; }
        else if (st_a == 1) {
            ba[na++] = (HopBranch){ A, v00r, -v00i };
            double sgn = sign_swap(s1, s2, A);
            ba[na++] = (HopBranch){ (A & ~(1ULL<<s1)) | (1ULL<<s2), v01r * sgn, -v01i * sgn };
        } else if (st_a == 2) {
            double sgn = sign_swap(s2, s1, A);
            ba[na++] = (HopBranch){ (A & ~(1ULL<<s2)) | (1ULL<<s1), v10r * sgn, -v10i * sgn };
            ba[na++] = (HopBranch){ A, v11r, -v11i };
        } else {
            ba[na++] = (HopBranch){ A, detr, -deti };
        }
        
        for (int cidx = 0; cidx < nc; cidx++) {
            for (int aidx = 0; aidx < na; aidx++) {
                double r = bc[cidx].r * ba[aidx].r - bc[cidx].i * ba[aidx].i;
                double im = bc[cidx].r * ba[aidx].i + bc[cidx].i * ba[aidx].r;
                double tre = terms[i].re * r - terms[i].im * im;
                double tim = terms[i].re * im + terms[i].im * r;
                
                uint64_t c0 = (l == 0) ? bc[cidx].mask : terms[i].c0;
                uint64_t a0 = (l == 0) ? ba[aidx].mask : terms[i].a0;
                uint64_t c1 = (l == 1) ? bc[cidx].mask : terms[i].c1;
                uint64_t a1 = (l == 1) ? ba[aidx].mask : terms[i].a1;
                
                add_term(c0, a0, c1, a1, tre, tim);
            }
        }
    }
    finish_step(0.0, 999);
}

void apply_interact(int s1, int s2, double a0_ph, double a1_ph, double g) {
    double exp_ig_r = cos(g), exp_ig_i = sin(g);
    double exp_mig_r = cos(-g), exp_mig_i = sin(-g);
    double z_r = exp_ig_r - 1.0, z_i = exp_ig_i;
    double zs_r = exp_mig_r - 1.0, zs_i = exp_mig_i;
    
    for (int i = 0; i < num_terms; i++) {
        uint64_t c0 = terms[i].c0, a0 = terms[i].a0;
        uint64_t c1 = terms[i].c1, a1 = terms[i].a1;
        double re = terms[i].re, im = terms[i].im;
        
        // Single site phases
        double ph_r = 1.0, ph_i = 0.0;
        if ((c0 >> s1) & 1) { double nr = ph_r*cos(a0_ph) - ph_i*sin(a0_ph); double ni = ph_r*sin(a0_ph) + ph_i*cos(a0_ph); ph_r = nr; ph_i = ni; }
        if ((a0 >> s1) & 1) { double nr = ph_r*cos(-a0_ph) - ph_i*sin(-a0_ph); double ni = ph_r*sin(-a0_ph) + ph_i*cos(-a0_ph); ph_r = nr; ph_i = ni; }
        if ((c1 >> s2) & 1) { double nr = ph_r*cos(a1_ph) - ph_i*sin(a1_ph); double ni = ph_r*sin(a1_ph) + ph_i*cos(a1_ph); ph_r = nr; ph_i = ni; }
        if ((a1 >> s2) & 1) { double nr = ph_r*cos(-a1_ph) - ph_i*sin(-a1_ph); double ni = ph_r*sin(-a1_ph) + ph_i*cos(-a1_ph); ph_r = nr; ph_i = ni; }
        
        double nre = re * ph_r - im * ph_i;
        double nim = re * ph_i + im * ph_r;
        
        int st_a = (((c0 >> s1) & 1) << 1) | ((a0 >> s1) & 1);
        int st_b = (((c1 >> s2) & 1) << 1) | ((a1 >> s2) & 1);
        
        ResBranch res[2];
        int n_res = 0;
        
        if (st_a == 0) {
            if (st_b == 0 || st_b == 3) { res[n_res++] = (ResBranch){0, 0, 1.0, 0.0}; }
            else if (st_b == 2) {
                res[n_res++] = (ResBranch){0, 0, 1.0, 0.0};
                res[n_res++] = (ResBranch){1, 0, zs_r, zs_i};
            } else {
                res[n_res++] = (ResBranch){0, 0, 1.0, 0.0};
                res[n_res++] = (ResBranch){1, 0, z_r, z_i};
            }
        } else if (st_a == 3) {
            if (st_b == 0 || st_b == 3) { res[n_res++] = (ResBranch){0, 0, 1.0, 0.0}; }
            else if (st_b == 2) { res[n_res++] = (ResBranch){0, 0, exp_mig_r, exp_mig_i}; }
            else { res[n_res++] = (ResBranch){0, 0, exp_ig_r, exp_ig_i}; }
        } else if (st_a == 2) {
            if (st_b == 0) {
                res[n_res++] = (ResBranch){0, 0, 1.0, 0.0};
                res[n_res++] = (ResBranch){0, 1, zs_r, zs_i};
            } else if (st_b == 3 || st_b == 2) { res[n_res++] = (ResBranch){0, 0, exp_mig_r, exp_mig_i}; }
            else { res[n_res++] = (ResBranch){0, 0, 1.0, 0.0}; }
        } else {
            if (st_b == 0) {
                res[n_res++] = (ResBranch){0, 0, 1.0, 0.0};
                res[n_res++] = (ResBranch){0, 1, z_r, z_i};
            } else if (st_b == 3 || st_b == 1) { res[n_res++] = (ResBranch){0, 0, exp_ig_r, exp_ig_i}; }
            else { res[n_res++] = (ResBranch){0, 0, 1.0, 0.0}; }
        }
        
        for (int k = 0; k < n_res; k++) {
            uint64_t nc0 = c0, na0 = a0, nc1 = c1, na1 = a1;
            double sgn = 1.0;
            if (res[k].n0) { nc0 |= (1ULL<<s1); na0 |= (1ULL<<s1); sgn *= sign_n(s1, c0, a0); }
            if (res[k].n1) { nc1 |= (1ULL<<s2); na1 |= (1ULL<<s2); sgn *= sign_n(s2, c1, a1); }
            
            double cr = res[k].r * sgn, ci = res[k].i * sgn;
            double tr = nre * cr - nim * ci;
            double ti = nre * ci + nim * cr;
            add_term(nc0, na0, nc1, na1, tr, ti);
        }
    }
    finish_step(0.0, 999);
}

void do_truncation(double eps, int k_max) {
    for (int i = 0; i < num_terms; i++) {
        add_term(terms[i].c0, terms[i].a0, terms[i].c1, terms[i].a1, terms[i].re, terms[i].im);
    }
    finish_step(eps, k_max);
}

double get_eval(uint64_t occ0, uint64_t occ1) {
    double sum = 0;
    for (int i = 0; i < num_terms; i++) {
        if (terms[i].c0 != terms[i].a0 || terms[i].c1 != terms[i].a1) continue;
        if ((terms[i].c0 & occ0) != terms[i].c0) continue;
        if ((terms[i].c1 & occ1) != terms[i].c1) continue;
        
        int k0 = __builtin_popcountll(terms[i].c0);
        int k1 = __builtin_popcountll(terms[i].c1);
        int sign = 1;
        if ((k0 * (k0 - 1) / 2) % 2 != 0) sign *= -1;
        if ((k1 * (k1 - 1) / 2) % 2 != 0) sign *= -1;
        
        sum += terms[i].re * sign;
    }
    return sum;
}
