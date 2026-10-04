// Metal compute kernels for the f32 dense state vector (src/metal_sv.rs).
//
// Amplitudes are float2 (re, im). Apple GPUs have no f64, so this backend is
// single precision only.
//
// `stage` is the GPU counterpart of the CPU blocked executor: one threadgroup
// gathers the 2^l amplitudes that share one assignment of the outer qubits
// into threadgroup memory, applies every op of the stage there, and writes
// them back. DRAM is streamed once per stage instead of once per gate.
//
// TG_BITS (max inner qubits, threadgroup buffer size) is set by the host when
// it compiles this source.

#include <metal_stdlib>
using namespace metal;

#ifndef TG_BITS
#define TG_BITS 12
#endif
#define TG_SIZE (1u << TG_BITS)

#define MAX_OUTER 32
#define MAX_HI 256

struct StageHdr {
    uint l;      // inner qubits of this stage (<= TG_BITS)
    uint b;      // inner qubits 0..b are the physical qubits 0..b (contiguous)
    uint nout;   // number of outer qubits
    uint nops;
    uint outer[MAX_OUTER];  // physical position of outer bit k of the group id
    uint hioff[MAX_HI];     // physical offset of buffer index bits b..l-1
};

// kind: 0 complex 2x2, 1 real 2x2, 2 X (controlled), 3 swap, 4 diagonal run,
//       5 dense 2-qubit (unused by the planner; kept for direct calls)
struct OpG {
    uint kind;
    uint t;      // target buffer bit (kinds 0-2); swap: lower bit
    uint cin;    // control mask on buffer bits
    uint omask;  // outer control mask (physical bits)
    uint opat;   // outer control pattern
    uint a;      // diag: first term; swap: upper bit
    uint nterms; // diag: number of terms
    uint pad;
    float m[8];  // row-major m00 m01 m10 m11 as (re, im) pairs
};

struct Term {
    uint imask;  // condition on buffer bits
    uint ipat;
    uint omask;  // condition on outer (physical) bits
    uint opat;
    float lnmag; // factor = exp(lnmag + i theta)
    float theta;
    uint pad0;
    uint pad1;
};

static inline uint insert0(uint p, uint t) {
    return ((p >> t) << (t + 1)) | (p & ((1u << t) - 1u));
}

static inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

kernel void stage(device float2* psi [[buffer(0)]],
                  constant StageHdr& h [[buffer(1)]],
                  device const OpG* ops [[buffer(2)]],
                  device const Term* terms [[buffer(3)]],
                  uint g [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]],
                  uint tpg [[threads_per_threadgroup]])
{
    threadgroup float2 buf[TG_SIZE];
    uint base = 0;
    for (uint k = 0; k < h.nout; k++) {
        base |= ((g >> k) & 1u) << h.outer[k];
    }
    const uint size = 1u << h.l;
    const uint b = h.b;
    const uint lowmask = (1u << b) - 1u;
    for (uint e = tid; e < size; e += tpg) {
        buf[e] = psi[base + (e & lowmask) + h.hioff[e >> b]];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint half_ = size >> 1;
    for (uint o = 0; o < h.nops; o++) {
        const OpG op = ops[o];
        // outer controls are uniform over the threadgroup
        if ((base & op.omask) != op.opat) {
            continue;
        }
        const uint t = op.t;
        const uint cin = op.cin;
        switch (op.kind) {
        case 0: {
            const float2 m00 = float2(op.m[0], op.m[1]);
            const float2 m01 = float2(op.m[2], op.m[3]);
            const float2 m10 = float2(op.m[4], op.m[5]);
            const float2 m11 = float2(op.m[6], op.m[7]);
            for (uint p = tid; p < half_; p += tpg) {
                const uint i0 = insert0(p, t);
                if ((i0 & cin) != cin) continue;
                const uint i1 = i0 | (1u << t);
                const float2 x = buf[i0];
                const float2 y = buf[i1];
                buf[i0] = cmul(m00, x) + cmul(m01, y);
                buf[i1] = cmul(m10, x) + cmul(m11, y);
            }
            break;
        }
        case 1: {
            const float r00 = op.m[0], r01 = op.m[2], r10 = op.m[4], r11 = op.m[6];
            for (uint p = tid; p < half_; p += tpg) {
                const uint i0 = insert0(p, t);
                if ((i0 & cin) != cin) continue;
                const uint i1 = i0 | (1u << t);
                const float2 x = buf[i0];
                const float2 y = buf[i1];
                buf[i0] = r00 * x + r01 * y;
                buf[i1] = r10 * x + r11 * y;
            }
            break;
        }
        case 2: {
            for (uint p = tid; p < half_; p += tpg) {
                const uint i0 = insert0(p, t);
                if ((i0 & cin) != cin) continue;
                const uint i1 = i0 | (1u << t);
                const float2 x = buf[i0];
                buf[i0] = buf[i1];
                buf[i1] = x;
            }
            break;
        }
        case 3: {
            const uint a = t, bb = op.a; // a < bb
            for (uint q = tid; q < (size >> 2); q += tpg) {
                const uint i = insert0(insert0(q, a), bb);
                const uint x = i | (1u << a);
                const uint y = i | (1u << bb);
                const float2 v = buf[x];
                buf[x] = buf[y];
                buf[y] = v;
            }
            break;
        }
        case 4: {
            const uint t0 = op.a, t1 = op.a + op.nterms;
            for (uint e = tid; e < size; e += tpg) {
                float ln = 0.0f, th = 0.0f;
                bool any = false;
                for (uint k = t0; k < t1; k++) {
                    const Term tm = terms[k];
                    if ((base & tm.omask) == tm.opat && (e & tm.imask) == tm.ipat) {
                        ln += tm.lnmag;
                        th += tm.theta;
                        any = true;
                    }
                }
                if (any) {
                    float c;
                    const float s = precise::sincos(th, c);
                    buf[e] = cmul(buf[e], precise::exp(ln) * float2(c, s));
                }
            }
            break;
        }
        default:
            break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint e = tid; e < size; e += tpg) {
        psi[base + (e & lowmask) + h.hioff[e >> b]] = buf[e];
    }
}

// ----- one dispatch per gate (no fusion): the "naive" GPU baseline --------

struct GHdr {
    uint kind;
    uint t;
    uint ctrl;   // physical control mask (U1); swap: upper qubit in `a`
    uint a;
    uint mask;   // phase: (i & mask) == pat
    uint pat;
    float fre;
    float fim;
    float m[8];
};

kernel void gate_pairs(device float2* psi [[buffer(0)]],
                       constant GHdr& h [[buffer(1)]],
                       uint p [[thread_position_in_grid]])
{
    const uint t = h.t;
    if (h.kind == 3) {
        // swap: p enumerates indices with both bits clear (grid = 2^(n-2))
        const uint i = insert0(insert0(p, t), h.a);
        const uint x = i | (1u << t);
        const uint y = i | (1u << h.a);
        const float2 v = psi[x];
        psi[x] = psi[y];
        psi[y] = v;
        return;
    }
    const uint i0 = insert0(p, t);
    if ((i0 & h.ctrl) != h.ctrl) return;
    const uint i1 = i0 | (1u << t);
    const float2 x = psi[i0];
    const float2 y = psi[i1];
    const float2 m00 = float2(h.m[0], h.m[1]);
    const float2 m01 = float2(h.m[2], h.m[3]);
    const float2 m10 = float2(h.m[4], h.m[5]);
    const float2 m11 = float2(h.m[6], h.m[7]);
    psi[i0] = cmul(m00, x) + cmul(m01, y);
    psi[i1] = cmul(m10, x) + cmul(m11, y);
}

kernel void gate_phase(device float2* psi [[buffer(0)]],
                       constant GHdr& h [[buffer(1)]],
                       uint i [[thread_position_in_grid]])
{
    if ((i & h.mask) == h.pat) {
        psi[i] = cmul(psi[i], float2(h.fre, h.fim));
    }
}

// ----- utilities ----------------------------------------------------------

kernel void init_basis(device float4* psi [[buffer(0)]],
                       constant uint& idx [[buffer(1)]],
                       uint i [[thread_position_in_grid]])
{
    // two amplitudes per thread
    float4 v = float4(0.0f);
    if ((idx >> 1) == i) {
        if (idx & 1u) v.z = 1.0f; else v.x = 1.0f;
    }
    psi[i] = v;
}

// Bandwidth probe: in-place scale, one read + one write of every amplitude.
kernel void scale4(device float4* psi [[buffer(0)]],
                   constant float& s [[buffer(1)]],
                   uint i [[thread_position_in_grid]])
{
    psi[i] = psi[i] * s;
}
